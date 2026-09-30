// SPDX-FileCopyrightText: 2026 deploy_ws
//
// OFFLINE test of what the g1_29dof Velocity state (State_RLBase) publishes
// right after enter(): nothing of this state (gains or q) may reach LowCmd
// before the first inference of the CURRENT entry.  Checked on first entry
// (action buffer still all zeros) and on re-entry (buffer holds the previous
// entry's target).  Uses the real State_RLBase + velocity ONNX, with fake
// LowCmd/LowState structs (scripts/parity/fake_sdk/Types.h): NO DDS, NO robot.
//
// Build + run: scripts/parity/run_rlbase_gate_test.sh
#include "FSM/State_RLBase.h"

#include <chrono>
#include <cmath>
#include <cstdio>
#include <thread>

std::unique_ptr<LowCmd_t>   FSMState::lowcmd   = nullptr;
std::shared_ptr<LowState_t> FSMState::lowstate = nullptr;
std::shared_ptr<Keyboard>   FSMState::keyboard = nullptr;

namespace {

constexpr int   kN = 29;
constexpr float kSentinelKp = -1.0f;
constexpr float kSentinelQ  = 99.0f;

struct EntryResult {
    int published = 0, untouched = 0, valid = 0, bad = 0;
    int first_valid_ms = -1;
    double enter_ms = 0.0;
    std::vector<float> first_q, last_q;
};

// One FSM entry: arm LowCmd with sentinels (= "previous state's command"),
// enter(), tick pre_run/run/post_run at ~1 kHz for `ms`, exit().
EntryResult run_entry(State_RLBase& st, const std::vector<float>& kp, const std::vector<float>& stale_q, int ms)
{
    auto& cmd = *FSMState::lowcmd;
    for (int i = 0; i < kN; ++i) {
        cmd.msg_.motor_cmd()[i].kp() = kSentinelKp;
        cmd.msg_.motor_cmd()[i].q()  = kSentinelQ;
    }
    cmd.published.clear();

    const auto t0 = std::chrono::steady_clock::now();
    st.enter();
    const double enter_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    for (int t = 0; t < ms; ++t) {
        st.pre_run();
        st.run();
        st.post_run();
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    st.exit();

    EntryResult r;
    r.enter_ms = enter_ms;
    std::lock_guard<std::mutex> lock(cmd.log_mtx);
    r.published = static_cast<int>(cmd.published.size());
    for (int m = 0; m < r.published; ++m) {
        auto& mc = cmd.published[m].motor_cmd();
        bool untouched = true, gains_ok = true, q_finite = true, q_zero = true, q_stale = !stale_q.empty();
        std::vector<float> q(kN);
        for (int i = 0; i < kN; ++i) {
            q[i] = mc[i].q();
            untouched = untouched && mc[i].kp() == kSentinelKp && q[i] == kSentinelQ;
            gains_ok  = gains_ok && mc[i].kp() == kp[i];
            q_finite  = q_finite && std::isfinite(q[i]) && q[i] != kSentinelQ;
            q_zero    = q_zero && q[i] == 0.0f;
            if (!stale_q.empty()) q_stale = q_stale && q[i] == stale_q[i];
        }
        if (untouched) {
            ++r.untouched;
        } else if (gains_ok && q_finite && !q_zero && !q_stale) {
            if (r.valid++ == 0) { r.first_valid_ms = m; r.first_q = q; }
            r.last_q = q;
        } else {
            ++r.bad;
        }
    }
    return r;
}

}  // namespace

int main(int argc, char** argv)
{
    if (argc != 2) {
        std::fprintf(stderr, "usage: %s <g1_29dof project dir>\n", argv[0]);
        return 2;
    }
    try {
    param::proj_dir = argv[1];
    param::config   = YAML::LoadFile((param::proj_dir / "config" / "config.yaml").string());
    FSMStringMap.insert({1, "Passive"});

    FSMState::lowcmd   = std::make_unique<LowCmd_t>();
    FSMState::lowstate = std::make_shared<LowState_t>();
    const auto vel_cfg = YAML::LoadFile((param::proj_dir / "config/policy/velocity/v0/params/deploy.yaml").string());
    const auto kp  = vel_cfg["stiffness"].as<std::vector<float>>();
    const auto ids = vel_cfg["joint_ids_map"].as<std::vector<int>>();
    const auto q0  = vel_cfg["default_joint_pos"].as<std::vector<float>>();
    auto set_stance = [&](float offset) {       // upright, Velocity nominal stance (+offset)
        std::lock_guard<std::mutex> lock(FSMState::lowstate->mutex_);
        for (int i = 0; i < kN; ++i) FSMState::lowstate->msg_.motors_[ids[i]].q_ = q0[i] + offset;
    };

    State_RLBase st(3, "Velocity");
    bool ok = true;

    set_stance(0.0f);
    const auto e1 = run_entry(st, kp, {}, 120);
    set_stance(0.05f);                           // different snapshot -> different action on re-entry
    const auto e2 = run_entry(st, kp, e1.last_q, 120);

    auto report = [&](const char* tag, const EntryResult& r) {
        std::printf("%-9s published %3d | previous-state cmd kept %3d | INVALID (gain set, q zero/stale) %3d | valid %3d, first valid at tick %d | enter() %.1f ms\n",
                    tag, r.published, r.untouched, r.bad, r.valid, r.first_valid_ms, r.enter_ms);
        if (r.bad != 0 || r.valid == 0) ok = false;
    };
    report("entry 1", e1);
    report("re-entry", e2);
    if (!e1.first_q.empty() && !e2.first_q.empty()) {
        float d = 0.0f;
        for (int i = 0; i < kN; ++i) d = std::max(d, std::fabs(e2.first_q[i] - e1.last_q[i]));
        std::printf("re-entry first valid q vs entry-1 last q: max|diff| = %.4f (must be fresh, > 0)\n", d);
        if (!(d > 0.0f)) ok = false;
    }
    std::printf("%s\n", ok ? "GATE PASS" : "GATE FAIL");
    return ok ? 0 : 1;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "FAIL: %s\n", e.what());
        return 1;
    }
}
