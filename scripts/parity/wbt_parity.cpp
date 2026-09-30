// SPDX-FileCopyrightText: 2026 deploy_ws
//
// OFFLINE numerical parity test: State_WBT's computational core (WBTPolicy,
// the real ObservationManager / ActionManager / OrtRunner and
// unitree::BaseArticulation::update) vs scripts/parity/sim_reference.py
// (UniLab tmp/sim_prototype.py), fed the same sensor trace.
//
// THIS PROGRAM DOES NOT TOUCH DDS, MOTORS, OR ANY ROBOT INTERFACE: the
// articulation reads a plain struct that mimics the unitree_hg LowState_
// accessors.  One WBTPolicy instance serves every episode, so each new
// episode is an exit + re-entry (State_WBT::enter -> WBTPolicy::begin).
//
// Build + run: scripts/parity/run_parity.sh
#include "WBT_Policy.h"
#include "unitree_articulation.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <mutex>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

struct FakeMotorState {
    float q_ = 0.0f, dq_ = 0.0f;
    float q()  const { return q_; }
    float dq() const { return dq_; }
};
struct FakeImuState {
    std::array<float, 4> quaternion_{1.0f, 0.0f, 0.0f, 0.0f};   // wxyz
    std::array<float, 3> gyroscope_{};
    const std::array<float, 4>& quaternion() const { return quaternion_; }
    const std::array<float, 3>& gyroscope()  const { return gyroscope_; }
};
struct FakeLowStateMsg {
    FakeImuState imu_;
    std::array<FakeMotorState, 35> motors_;
    const FakeImuState& imu_state() const { return imu_; }
    const std::array<FakeMotorState, 35>& motor_state() const { return motors_; }
};
struct FakeLowState {
    std::mutex mutex_;
    FakeLowStateMsg msg_;
    unitree::common::UnitreeJoystick joystick;
};

constexpr int kJ = 29;
constexpr int kSensorDim = 4 + 3 + kJ + kJ;   // quat, gyro, q, dq

struct Episode {
    int start = 0, steps = 0;
    std::vector<float> sensors;   // steps x kSensorDim
};

std::vector<float> read_floats(std::ifstream& f, std::size_t n)
{
    std::vector<float> v(n);
    f.read(reinterpret_cast<char*>(v.data()), n * sizeof(float));
    if (f.gcount() != static_cast<std::streamsize>(n * sizeof(float))) {
        throw std::runtime_error("short read");
    }
    return v;
}

float max_abs_diff(const float* a, const float* b, int n)
{
    float m = 0.0f;
    for (int i = 0; i < n; ++i) m = std::max(m, std::fabs(a[i] - b[i]));
    return m;
}

}  // namespace


int main(int argc, char** argv)
{
    if (argc != 7) {
        std::fprintf(stderr, "usage: %s deploy_cfg.yaml fsm_config.yaml policy.onnx motion.bin "
                             "parity_dir(trace.bin, ref.bin) fsm_state_name\n", argv[0]);
        return 2;
    }
    try {
    const YAML::Node deploy_cfg = YAML::LoadFile(argv[1]);
    const YAML::Node fsm_cfg    = YAML::LoadFile(argv[2])["FSM"][argv[6]];
    const std::string dir       = argv[5];
    const int obs_dim = deploy_cfg["obs_dim"].as<int>();
    const int row_dim = obs_dim + 2 * kJ;

    auto low   = std::make_shared<FakeLowState>();
    auto robot = std::make_shared<unitree::BaseArticulation<std::shared_ptr<FakeLowState>>>(low);
    WBTPolicy policy(deploy_cfg,
                     fsm_cfg["kp"].as<std::vector<float>>(), fsm_cfg["kd"].as<std::vector<float>>(),
                     argv[4], argv[3], robot);

    bool ok = true;
    try {   // no inference / no target before an episode has begun
        policy.step();
        std::printf("FAIL: step() before begin() did not throw\n");
        ok = false;
    } catch (const std::logic_error&) {}

    std::ifstream trace(dir + "/trace.bin", std::ios::binary);
    std::ifstream ref_f(dir + "/ref.bin", std::ios::binary);
    if (!trace || !ref_f) throw std::runtime_error("cannot open trace.bin / ref.bin in " + dir);
    std::int32_t n_ep = 0;
    trace.read(reinterpret_cast<char*>(&n_ep), sizeof(n_ep));
    std::vector<Episode> episodes(n_ep);
    for (auto& ep : episodes) {
        std::int32_t hdr[2];
        trace.read(reinterpret_cast<char*>(hdr), sizeof(hdr));
        ep.start = hdr[0];
        ep.steps = hdr[1];
        ep.sensors = read_floats(trace, static_cast<std::size_t>(ep.steps) * kSensorDim);
    }

    // obs segment table from the deploy contract
    struct Seg { std::string name; int off, len; float diff = 0.0f; };
    std::vector<Seg> segs;
    int off = 0;
    for (const auto& s : deploy_cfg["obs_layout"]) {
        const int len = s["dim"].as<int>() * (s["history_length"] ? s["history_length"].as<int>() : 1);
        segs.push_back({s["name"].as<std::string>(), off, len});
        off += len;
    }

    const int last = policy.motion().last_frame();
    const int max_future = deploy_cfg["future_steps"].as<std::vector<int>>().back();
    const float tol = 1e-4f;
    float g_obs = 0, g_act = 0, g_q = 0;
    std::vector<std::vector<float>> cpp_rows;   // all episodes, for the re-entry check
    std::vector<double> step_us;                // WBTPolicy::step() wall time (obs + ONNX + action)
    std::ofstream cpp_out(dir + "/cpp.bin", std::ios::binary);

    std::printf("%-3s %-26s %6s %12s %12s %12s\n", "ep", "checkpoint", "frame", "obs", "raw_action", "q_target");
    for (int e = 0; e < n_ep; ++e) {
        const auto& ep = episodes[e];
        float e_obs = 0, e_act = 0, e_q = 0, t_obs = 0, t_act = 0, t_q = 0;
        for (int s = 0; s < ep.steps; ++s) {
            const float* in = &ep.sensors[static_cast<std::size_t>(s) * kSensorDim];
            {
                std::lock_guard<std::mutex> lock(low->mutex_);
                for (int i = 0; i < 4; ++i) low->msg_.imu_.quaternion_[i] = in[i];
                for (int i = 0; i < 3; ++i) low->msg_.imu_.gyroscope_[i]  = in[4 + i];
                for (int j = 0; j < kJ; ++j) {
                    low->msg_.motors_[j].q_  = in[7 + j];
                    low->msg_.motors_[j].dq_ = in[7 + kJ + j];
                }
            }
            policy.env()->robot->update();
            if (s == 0) {
                policy.begin(ep.start);
                if (policy.has_action() || !policy.q_target().empty()) {
                    std::printf("FAIL: ep %d has a q_target before its first inference\n", e);
                    ok = false;
                }
                const auto& iq = policy.init_quat();
                std::printf("%-3d init_quat(wxyz)=[%+.7f %+.7f %+.7f %+.7f]\n",
                            e, iq.w(), iq.x(), iq.y(), iq.z());
            }
            const int frame = policy.frame();
            const auto t0 = std::chrono::steady_clock::now();
            policy.step();
            step_us.push_back(std::chrono::duration<double, std::micro>(
                std::chrono::steady_clock::now() - t0).count());

            std::vector<float> row(policy.obs());
            row.insert(row.end(), policy.raw_action().begin(), policy.raw_action().end());
            row.insert(row.end(), policy.q_target().begin(), policy.q_target().end());
            if (static_cast<int>(row.size()) != row_dim) throw std::runtime_error("row size mismatch");
            cpp_out.write(reinterpret_cast<const char*>(row.data()), row.size() * sizeof(float));
            const auto ref = read_floats(ref_f, row_dim);

            const float d_obs = max_abs_diff(row.data(), ref.data(), obs_dim);
            const float d_act = max_abs_diff(row.data() + obs_dim, ref.data() + obs_dim, kJ);
            const float d_q   = max_abs_diff(row.data() + obs_dim + kJ, ref.data() + obs_dim + kJ, kJ);
            for (auto& sg : segs) {
                sg.diff = std::max(sg.diff, max_abs_diff(row.data() + sg.off, ref.data() + sg.off, sg.len));
            }
            e_obs = std::max(e_obs, d_obs);  e_act = std::max(e_act, d_act);  e_q = std::max(e_q, d_q);
            if (frame + max_future > last) {
                t_obs = std::max(t_obs, d_obs);  t_act = std::max(t_act, d_act);  t_q = std::max(t_q, d_q);
            }
            const char* tag = s == 0 ? "reset + first frame" : s == 9 ? "10th frame" : nullptr;
            if (tag) std::printf("%-3d %-26s %6d %12.3e %12.3e %12.3e\n", e, tag, frame, d_obs, d_act, d_q);
            cpp_rows.push_back(std::move(row));
        }
        if (t_obs > 0 || t_act > 0 || t_q > 0 || episodes[e].start + ep.steps - 1 + max_future > last) {
            std::printf("%-3d %-26s %6s %12.3e %12.3e %12.3e\n", e, "tail (future clamped)",
                        (std::to_string(last - max_future + 1) + "+").c_str(), t_obs, t_act, t_q);
        }
        std::printf("%-3d %-26s %6s %12.3e %12.3e %12.3e\n", e, "episode max",
                    (std::to_string(ep.steps) + "st").c_str(), e_obs, e_act, e_q);
        g_obs = std::max(g_obs, e_obs);  g_act = std::max(g_act, e_act);  g_q = std::max(g_q, e_q);
    }

    // Re-entry: the last episode replays episode 0's first sensor rows after a
    // mid-clip exit, so its C++ output must equal episode 0's bit-for-bit.
    if (n_ep >= 2 && episodes.back().start == episodes[0].start) {
        const int n = episodes.back().steps;
        const std::size_t base = cpp_rows.size() - n;
        float d = 0.0f;
        for (int s = 0; s < n; ++s) {
            d = std::max(d, max_abs_diff(cpp_rows[base + s].data(), cpp_rows[s].data(), row_dim));
        }
        std::printf("re-entry (ep %d vs ep 0, %d steps): max_abs_diff = %.3e\n", n_ep - 1, n, d);
        if (d != 0.0f) ok = false;
    }

    std::printf("per-segment obs max_abs_diff (all steps):\n");
    for (const auto& sg : segs) {
        std::printf("  [%4d,%4d) %-22s %.3e\n", sg.off, sg.off + sg.len, sg.name.c_str(), sg.diff);
    }
    if (!step_us.empty()) {
        auto sorted = step_us;
        std::sort(sorted.begin(), sorted.end());
        double mean = 0.0;
        for (double v : sorted) mean += v;
        mean /= sorted.size();
        std::printf("WBTPolicy::step() time over %zu steps: mean %.2f ms  p99 %.2f ms  max %.2f ms\n",
                    sorted.size(), mean * 1e-3, sorted[sorted.size() * 99 / 100] * 1e-3, sorted.back() * 1e-3);
    }
    std::printf("OVERALL max_abs_diff: obs %.3e  raw_action %.3e  q_target %.3e  (tol %.0e)\n",
                g_obs, g_act, g_q, tol);
    ok = ok && g_obs <= tol && g_act <= tol && g_q <= tol;
    std::printf("%s\n", ok ? "PARITY PASS" : "PARITY FAIL");
    return ok ? 0 : 1;

    } catch (const std::exception& e) {
        std::fprintf(stderr, "FAIL: %s\n", e.what());
        return 1;
    }
}
