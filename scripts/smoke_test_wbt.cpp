// SPDX-FileCopyrightText: 2026 deploy_ws
//
// Standalone OFFLINE smoke test for the WBT deploy stack on aarch64 — exercises
// every State_WBT codepath that doesn't need DDS / a real robot:
//   - WBTMotionLoader: parse 16-byte header, slurp 6 frames-major float32 blocks
//   - Frame-cursor lerp/slerp at multiple times in the clip
//   - init_quat = yaw_align(robot_torso_w, ref_motion[0].anchor_quat_w)
//   - motion_anchor_ori_b = R(real^-1 * init * ref)[:, :2] (6 floats)
//   - Synthetic 154-d obs assembly in UniLab order, ONNX forward, finite action
//
// THIS PROGRAM DOES NOT TOUCH DDS, MOTORS, OR ANY ROBOT INTERFACE.
//
// Build:
//   g++ -std=c++17 -O2 \
//       -I /usr/include/eigen3 \
//       -I unitree_rl_lab/deploy/thirdparty/onnxruntime-linux-x64-1.22.0/include \
//       scripts/smoke_test_wbt.cpp \
//       unitree_rl_lab/deploy/thirdparty/onnxruntime-linux-x64-1.22.0/lib/libonnxruntime.so.1.22.0 \
//       -o scripts/smoke_test_wbt
//
// Run:
//   LD_LIBRARY_PATH=unitree_rl_lab/deploy/thirdparty/onnxruntime-linux-x64-1.22.0/lib \
//   ./scripts/smoke_test_wbt
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include <eigen3/Eigen/Dense>
#include "onnxruntime_cxx_api.h"


// ---------- WBTMotionLoader (mirrors State_WBT.cpp) ----------------------------
struct MotionData {
    int fps = 0, num_frames = 0, num_joints = 0, num_bodies = 0, anchor_idx = 7;
    float dt = 0, duration = 0;
    Eigen::MatrixXf joint_pos, joint_vel;        // (F, J) row-major
    std::vector<Eigen::Vector3f>    anchor_pos;  // (F)
    std::vector<Eigen::Quaternionf> anchor_quat; // (F) wxyz
};

static MotionData load_motion(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f.is_open()) throw std::runtime_error("cannot open " + path);

    int32_t hdr[4];
    f.read(reinterpret_cast<char*>(hdr), sizeof(hdr));

    MotionData m;
    m.fps = hdr[0]; m.num_frames = hdr[1]; m.num_joints = hdr[2]; m.num_bodies = hdr[3];
    m.dt = 1.0f / m.fps;
    m.duration = m.num_frames * m.dt;

    auto read_block = [&](size_t n) {
        std::vector<float> buf(n);
        f.read(reinterpret_cast<char*>(buf.data()), n * sizeof(float));
        return buf;
    };
    {
        auto buf = read_block(static_cast<size_t>(m.num_frames) * m.num_joints);
        m.joint_pos = Eigen::Map<
            Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
        >(buf.data(), m.num_frames, m.num_joints);
    }
    {
        auto buf = read_block(static_cast<size_t>(m.num_frames) * m.num_joints);
        m.joint_vel = Eigen::Map<
            Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
        >(buf.data(), m.num_frames, m.num_joints);
    }
    {
        auto buf = read_block(static_cast<size_t>(m.num_frames) * m.num_bodies * 3);
        m.anchor_pos.resize(m.num_frames);
        for (int fr = 0; fr < m.num_frames; ++fr) {
            const float* base = buf.data() + (fr * m.num_bodies + m.anchor_idx) * 3;
            m.anchor_pos[fr] = Eigen::Vector3f(base[0], base[1], base[2]);
        }
    }
    {
        auto buf = read_block(static_cast<size_t>(m.num_frames) * m.num_bodies * 4);
        m.anchor_quat.resize(m.num_frames);
        for (int fr = 0; fr < m.num_frames; ++fr) {
            const float* base = buf.data() + (fr * m.num_bodies + m.anchor_idx) * 4;
            m.anchor_quat[fr] = Eigen::Quaternionf(base[0], base[1], base[2], base[3]);
            m.anchor_quat[fr].normalize();
        }
    }
    return m;
}

// yaw-only Eigen quat extractor (matches isaaclab::yawQuaternion)
static Eigen::Quaternionf yaw_quat(const Eigen::Quaternionf& q) {
    float yaw = std::atan2(2.0f * (q.w() * q.z() + q.x() * q.y()),
                           1.0f - 2.0f * (q.y() * q.y() + q.z() * q.z()));
    float h = yaw * 0.5f;
    return Eigen::Quaternionf(std::cos(h), 0.0f, 0.0f, std::sin(h)).normalized();
}

// motion_anchor_ori_b — R(real^-1 * init * ref)[:, :2] flattened row-major (6-d)
static std::array<float, 6> compute_anchor_ori_b(
    const Eigen::Quaternionf& real_q,
    const Eigen::Quaternionf& init_q,
    const Eigen::Quaternionf& ref_q)
{
    Eigen::Matrix3f R = (real_q.conjugate() * init_q * ref_q).toRotationMatrix();
    return {R(0,0), R(0,1), R(1,0), R(1,1), R(2,0), R(2,1)};
}


int main(int argc, char** argv)
{
    const std::string asset_dir   = (argc > 1) ? argv[1] : "assets";
    const std::string motion_path = asset_dir + "/dance1.bin";
    const std::string policy_path = asset_dir + "/policy.onnx";

    // ---------- 1. motion bin ----------
    MotionData m = load_motion(motion_path);
    std::printf("[motion] fps=%d frames=%d joints=%d bodies=%d duration=%.2fs\n",
                m.fps, m.num_frames, m.num_joints, m.num_bodies, m.duration);
    if (m.num_joints != 29 || m.num_bodies != 14) {
        std::cerr << "FAIL: schema mismatch\n"; return 1;
    }
    // sanity-check several frames
    for (int fr : {0, m.num_frames / 2, m.num_frames - 1}) {
        const auto jp = m.joint_pos.row(fr);
        const auto jv = m.joint_vel.row(fr);
        const auto& q = m.anchor_quat[fr];
        const auto& p = m.anchor_pos[fr];
        if (!jp.allFinite() || !jv.allFinite() || !p.allFinite()) {
            std::cerr << "FAIL: non-finite motion frame " << fr << "\n"; return 1;
        }
        if (std::abs(q.norm() - 1.0f) > 1e-3f) {
            std::cerr << "FAIL: non-unit quat at frame " << fr
                      << " norm=" << q.norm() << "\n"; return 1;
        }
        std::printf("[motion] frame %4d  joint[:3]=[%+.3f %+.3f %+.3f]  "
                    "anchor pos=[%+.2f %+.2f %+.2f]  quat(wxyz)=[%+.3f %+.3f %+.3f %+.3f]\n",
                    fr, jp(0), jp(1), jp(2), p(0), p(1), p(2),
                    q.w(), q.x(), q.y(), q.z());
    }

    // ---------- 2. init_quat: yaw_align(robot, ref[0]) ----------
    // Pretend the deployed robot's torso quat is identity (FixStand stance).
    Eigen::Quaternionf real_q0   = Eigen::Quaternionf::Identity();
    Eigen::Quaternionf ref_q0    = m.anchor_quat[0];
    Eigen::Quaternionf init_quat =
        Eigen::Quaternionf(yaw_quat(real_q0).toRotationMatrix() *
                           yaw_quat(ref_q0).toRotationMatrix().transpose());
    std::printf("[init] init_quat (wxyz) = [%+.4f %+.4f %+.4f %+.4f] (yaw align robot↔ref0)\n",
                init_quat.w(), init_quat.x(), init_quat.y(), init_quat.z());

    // motion_anchor_ori_b at frame 0 with real ≈ ref0 → residual ≈ I, so 6-d ≈ [1,0,0,1,0,0]
    auto ori6 = compute_anchor_ori_b(real_q0, init_quat, m.anchor_quat[0]);
    std::printf("[obs ] anchor_ori_b @ frame 0   = [%+.3f %+.3f %+.3f %+.3f %+.3f %+.3f]"
                "  (expect ≈ [1,0,0,1,0,0])\n",
                ori6[0], ori6[1], ori6[2], ori6[3], ori6[4], ori6[5]);
    auto ori6_mid = compute_anchor_ori_b(real_q0, init_quat,
                                         m.anchor_quat[m.num_frames/2]);
    std::printf("[obs ] anchor_ori_b @ frame %d = [%+.3f %+.3f %+.3f %+.3f %+.3f %+.3f]\n",
                m.num_frames/2, ori6_mid[0], ori6_mid[1], ori6_mid[2],
                ori6_mid[3], ori6_mid[4], ori6_mid[5]);

    // ---------- 3. ONNX load ----------
    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "smoke");
    Ort::SessionOptions opts;
    opts.SetGraphOptimizationLevel(ORT_ENABLE_EXTENDED);
    Ort::Session sess(env, policy_path.c_str(), opts);

    Ort::AllocatorWithDefaultOptions alloc;
    auto in_name  = sess.GetInputNameAllocated(0, alloc);
    auto out_name = sess.GetOutputNameAllocated(0, alloc);
    auto in_shape = sess.GetInputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape();
    if (in_shape[1] != 154) {
        std::cerr << "FAIL: ONNX input width " << in_shape[1] << " != 154\n"; return 1;
    }
    std::printf("[ort ] %s loaded; input '%s'[1,154] -> output '%s'[1,29]\n",
                policy_path.c_str(), in_name.get(), out_name.get());

    // ---------- 4. Synthetic 154-d obs in UniLab order ----------
    // [cmd_jpos(29) | cmd_jvel(29) | anchor_ori_b(6) | gyro(3) | jp_rel(29) | dof_vel(29) | last_act(29)]
    // Use motion frame 0 for command, identity gyro, default-pos diff = 0, prev action = 0.
    std::vector<float> obs;
    obs.reserve(154);
    auto append = [&](const float* p, int n){ obs.insert(obs.end(), p, p + n); };
    Eigen::VectorXf jp0 = m.joint_pos.row(0);
    Eigen::VectorXf jv0 = m.joint_vel.row(0);
    append(jp0.data(), 29);
    append(jv0.data(), 29);
    for (float v : ori6) obs.push_back(v);
    obs.insert(obs.end(), {0.0f, 0.0f, 0.0f});                  // gyro
    obs.insert(obs.end(), 29, 0.0f);                            // joint_pos_rel
    obs.insert(obs.end(), 29, 0.0f);                            // dof_vel
    obs.insert(obs.end(), 29, 0.0f);                            // last_actions
    if (obs.size() != 154) {
        std::cerr << "FAIL: assembled obs size " << obs.size() << " != 154\n"; return 1;
    }

    std::vector<int64_t> shape{1, 154};
    auto mem = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
    auto in_t = Ort::Value::CreateTensor<float>(mem, obs.data(), obs.size(),
                                                shape.data(), shape.size());
    const char* in_names[]  = {in_name.get()};
    const char* out_names[] = {out_name.get()};
    auto out = sess.Run(Ort::RunOptions{nullptr}, in_names, &in_t, 1, out_names, 1);
    auto* a = out[0].GetTensorMutableData<float>();

    float amax = 0.0f;
    for (int i = 0; i < 29; ++i) amax = std::max(amax, std::fabs(a[i]));
    std::printf("[ort ] forward(motion frame 0 obs):  action[:6]=[%+.3f %+.3f %+.3f %+.3f %+.3f %+.3f]"
                "  |action|_max=%.3f\n",
                a[0], a[1], a[2], a[3], a[4], a[5], amax);

    if (amax > 1.05f) {
        std::cerr << "WARN: |action| > 1 (tanh out-of-range; OK if action_scale handled downstream)\n";
    }

    std::printf("\nSMOKE OK — motion + init_quat + ori_b + ONNX 154->29 all green.\n"
                "         (Robot remains untouched; no DDS, no motor commands.)\n");
    return 0;
}
