// SPDX-FileCopyrightText: 2026 deploy_ws (parallel WBT chain).
#include "WBT_Policy.h"

#include "isaaclab/envs/mdp/observations/observations.h"
#include "isaaclab/envs/mdp/actions/joint_actions.h"
#include "isaaclab/utils/utils.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <map>
#include <set>
#include <stdexcept>


// ---------------------------------------------------------------------------
// WBTMotionLoader
// ---------------------------------------------------------------------------

WBTMotionLoader::WBTMotionLoader(const std::string& path, int anchor_body_idx, int root_body_idx)
: anchor_idx_(anchor_body_idx), root_idx_(root_body_idx)
{
    std::ifstream f(path, std::ios::binary);
    if (!f.is_open()) {
        throw std::runtime_error("WBTMotionLoader: cannot open " + path);
    }

    std::int32_t hdr[4];
    f.read(reinterpret_cast<char*>(hdr), sizeof(hdr));
    if (f.gcount() != sizeof(hdr)) {
        throw std::runtime_error("WBTMotionLoader: short header in " + path);
    }
    fps_        = hdr[0];
    num_frames_ = hdr[1];
    num_joints_ = hdr[2];
    num_bodies_ = hdr[3];

    if (fps_ <= 0 || num_frames_ <= 0 || num_joints_ <= 0 || num_bodies_ <= 0) {
        throw std::runtime_error("WBTMotionLoader: bad header in " + path);
    }
    if (anchor_idx_ < 0 || anchor_idx_ >= num_bodies_ || root_idx_ < 0 || root_idx_ >= num_bodies_) {
        throw std::runtime_error("WBTMotionLoader: anchor idx " + std::to_string(anchor_idx_)
                                 + " out of range [0," + std::to_string(num_bodies_) + ")");
    }

    auto read_block = [&](std::size_t count) {
        std::vector<float> buf(count);
        f.read(reinterpret_cast<char*>(buf.data()), count * sizeof(float));
        if (f.gcount() != static_cast<std::streamsize>(count * sizeof(float))) {
            throw std::runtime_error("WBTMotionLoader: short data block in " + path);
        }
        for (float v : buf) {
            if (!std::isfinite(v)) {
                throw std::runtime_error("WBTMotionLoader: non-finite value in " + path);
            }
        }
        return buf;
    };

    const std::size_t F = num_frames_, J = num_joints_, B = num_bodies_;
    joint_pos_ = read_block(F * J);
    joint_vel_ = read_block(F * J);
    {
        const auto buf = read_block(F * B * 3);
        root_pos_w_.resize(F);
        for (std::size_t frame = 0; frame < F; ++frame) {
            const float* p = buf.data() + frame * B * 3;
            root_pos_w_[frame] = Eigen::Vector3f(p[root_idx_ * 3],
                                                 p[root_idx_ * 3 + 1],
                                                 p[root_idx_ * 3 + 2]);
        }
    }
    {
        // body_quat_w wxyz: keep the anchor column exactly as stored (the
        // training command and sim_prototype.py do not re-normalise).
        const auto buf = read_block(F * B * 4);
        anchor_quat_w_.resize(F);
        for (std::size_t frame = 0; frame < F; ++frame) {
            const float* q = buf.data() + (frame * B + anchor_idx_) * 4;
            anchor_quat_w_[frame] = Eigen::Quaternionf(q[0], q[1], q[2], q[3]);
            if (std::abs(anchor_quat_w_[frame].norm() - 1.0f) > 1e-3f) {
                throw std::runtime_error("WBTMotionLoader: non-unit anchor quaternion at frame "
                                         + std::to_string(frame) + " in " + path);
            }
        }
    }
    (void)read_block(F * B * 3);   // body_lin_vel_w
    (void)read_block(F * B * 3);   // body_ang_vel_w

    if (f.peek() != EOF) {
        throw std::runtime_error("WBTMotionLoader: trailing bytes after expected payload in " + path);
    }
}


// ---------------------------------------------------------------------------
// Observation registrars.  `wbt_` prefixes avoid collisions with upstream
// registrations while keeping the deploy contract explicit.
// ---------------------------------------------------------------------------

WBTPolicy* WBTPolicy::active = nullptr;

namespace isaaclab { namespace mdp {

// MimicLite's `base_ang_vel` is the MuJoCo torso gyro; the upstream
// `base_ang_vel` (pelvis IMU) stays untouched for the IsaacLab policies.
REGISTER_OBSERVATION(wbt_torso_ang_vel)
{
    const auto& data = env->robot->data.torso_ang_vel_b;
    return std::vector<float>(data.data(), data.data() + data.size());
}
REGISTER_OBSERVATION(wbt_ref_root_pos_future_local)
{
    if (!WBTPolicy::active) {
        throw std::runtime_error("wbt_ref_root_pos_future_local: no active WBTPolicy");
    }
    return WBTPolicy::active->ref_root_pos_future_local(env);
}
REGISTER_OBSERVATION(wbt_ref_root_ori_future_b)
{
    if (!WBTPolicy::active) {
        throw std::runtime_error("wbt_ref_root_ori_future_b: no active WBTPolicy");
    }
    return WBTPolicy::active->ref_root_ori_future_b(env);
}
REGISTER_OBSERVATION(wbt_ref_joint_pos_future)
{
    if (!WBTPolicy::active) {
        throw std::runtime_error("wbt_ref_joint_pos_future: no active WBTPolicy");
    }
    return WBTPolicy::active->ref_joint_pos_future(env);
}

}}  // namespace isaaclab::mdp


// ---------------------------------------------------------------------------
// WBTPolicy
// ---------------------------------------------------------------------------

Eigen::Quaternionf WBTPolicy::torso_quat_w(const isaaclab::ArticulationData& data)
{
    return data.root_quat_w
         * Eigen::AngleAxisf(data.joint_pos[12], Eigen::Vector3f::UnitZ())
         * Eigen::AngleAxisf(data.joint_pos[13], Eigen::Vector3f::UnitX())
         * Eigen::AngleAxisf(data.joint_pos[14], Eigen::Vector3f::UnitY());
}

Eigen::Quaternionf WBTPolicy::projected_yaw_quat(const Eigen::Quaternionf& quat)
{
    const Eigen::Matrix3f R = quat.toRotationMatrix();
    const Eigen::Vector3f x = R.col(0), z = R.col(2);
    Eigen::Vector2f heading;
    if (x.head<2>().norm() > 0.1f) {
        heading = x.head<2>();
    } else {
        if (x.z() < 0.0f) { heading = z.head<2>(); } else { heading = -z.head<2>(); }
    }
    return Eigen::Quaternionf(Eigen::AngleAxisf(std::atan2(heading.y(), heading.x()),
                                                 Eigen::Vector3f::UnitZ()));
}

static int clamp_motion_frame(int frame, int step, int last)
{
    return std::clamp(frame + step, 0, last);
}

std::vector<float> WBTPolicy::ref_root_pos_future_local(const isaaclab::ManagerBasedRLEnv* env) const
{
    const auto root_quat = env->robot->data.root_quat_w;
    const auto root_yaw = projected_yaw_quat(root_quat);
    const int last = motion_->last_frame();
    const auto current_ref = motion_->root_pos(frame_);
    std::vector<float> out;
    out.reserve(future_steps_.size() * 3);
    // No global root-position sensor exists on the G1 low state.  The
    // deployment reference is therefore anchored at the current clip root;
    // future displacements retain the training term's local-frame semantics.
    // Displacements are clip-frame vectors: rotate them by init_quat_ into the
    // robot's world (as ref_root_ori_future_b does for orientations) before
    // projecting onto the robot's yaw.
    for (const int step : future_steps_) {
        const auto p = motion_->root_pos(clamp_motion_frame(frame_, step, last));
        const Eigen::Vector3f local = root_yaw.conjugate() * (init_quat_ * (p - current_ref));
        out.insert(out.end(), local.data(), local.data() + 3);
    }
    return out;
}

std::vector<float> WBTPolicy::ref_root_ori_future_b(const isaaclab::ManagerBasedRLEnv* env) const
{
    const Eigen::Quaternionf anchor_quat = anchor_is_pelvis_
        ? env->robot->data.root_quat_w : torso_quat_w(env->robot->data);
    const Eigen::Quaternionf robot_inv = anchor_quat.conjugate();
    const int last = motion_->last_frame();
    std::vector<float> out;
    out.reserve(future_steps_.size() * 6);
    for (const int step : future_steps_) {
        const int f = clamp_motion_frame(frame_, step, last);
        const Eigen::Matrix3f R = (robot_inv * (init_quat_ * motion_->anchor_quat(f))).toRotationMatrix();
        out.push_back(R(0, 0)); out.push_back(R(0, 1)); out.push_back(R(0, 2));
        out.push_back(R(1, 0)); out.push_back(R(1, 1)); out.push_back(R(1, 2));
    }
    return out;
}

std::vector<float> WBTPolicy::ref_joint_pos_future(const isaaclab::ManagerBasedRLEnv*) const
{
    const int last = motion_->last_frame();
    const int J = motion_->num_joints();
    std::vector<float> out;
    out.reserve(future_steps_.size() * J);
    for (const int step : future_steps_) {
        const int f = clamp_motion_frame(frame_, step, last);
        out.insert(out.end(), motion_->joint_pos(f), motion_->joint_pos(f) + J);
    }
    return out;
}

YAML::Node WBTPolicy::build_env_cfg(const YAML::Node& deploy_cfg,
                                    const std::vector<float>& kp,
                                    const std::vector<float>& kd)
{
    // Map the UniLab deploy_config.yaml -> the schema ManagerBasedRLEnv parses.
    const auto  default_angles = deploy_cfg["default_angles"].as<std::vector<float>>();
    const float action_scale   = deploy_cfg["action_scale"].as<float>();

    // Training joint order (task base.yaml joint_names) == G1 SDK motor order,
    // so policy index == motor index.
    std::vector<float> joint_ids_map(default_angles.size());
    for (std::size_t i = 0; i < joint_ids_map.size(); ++i) joint_ids_map[i] = static_cast<float>(i);

    YAML::Node cfg;
    cfg["step_dt"]           = deploy_cfg["ctrl_dt"].as<float>();
    cfg["joint_ids_map"]     = joint_ids_map;
    cfg["default_joint_pos"] = default_angles;
    cfg["stiffness"]         = kp;
    cfg["damping"]           = kd;

    // -- action: MotionJointPositionAction = raw * action_scale + default_angles.
    // No clip (training applies none; the actuators enforce their own limits).
    YAML::Node jpa;
    jpa["joint_names"] = std::vector<std::string>{".*"};
    jpa["scale"]       = std::vector<float>(default_angles.size(), action_scale);
    jpa["offset"]      = default_angles;
    jpa["clip"]        = YAML::Node();   // null -> no clip
    jpa["joint_ids"]   = YAML::Node();   // null -> all joints
    jpa["simulate_action_latency"] = deploy_cfg["simulate_action_latency"].IsDefined()
        ? deploy_cfg["simulate_action_latency"].as<bool>() : false;
    cfg["actions"]["JointPositionAction"] = jpa;

    // -- observations, in the exact obs_layout order (yaml-cpp maps preserve
    // insertion order; ObservationManager concatenates in that order).
    // UniLab actor term -> framework registrar.
    static const std::map<std::string, std::string> alias = {
        {"base_ang_vel",         "wbt_torso_ang_vel"},         // torso gyro (pelvis IMU + waist rates)
        {"joint_pos",            "joint_pos"},                 // raw q (MimicLite)
        {"joint_vel",            "joint_vel_rel"},             // raw dq
        {"last_action",          "last_action"},
        {"ref_root_pos_future_local", "wbt_ref_root_pos_future_local"},
        {"ref_root_ori_future_b", "wbt_ref_root_ori_future_b"},
        {"ref_joint_pos_future", "wbt_ref_joint_pos_future"},
        {"projected_gravity",    "projected_gravity"},         // root_quat_w^-1 * [0,0,-1]
    };

    YAML::Node obs;
    std::set<std::string> seen;
    int obs_total = 0;
    for (const auto& seg : deploy_cfg["obs_layout"]) {
        const auto name = seg["name"].as<std::string>();
        const int  dim  = seg["dim"].as<int>();
        const int  hist = seg["history_length"] ? seg["history_length"].as<int>() : 1;
        const auto it = alias.find(name);
        if (it == alias.end()) {
            throw std::runtime_error("WBTPolicy: no framework term for obs segment '" + name + "'");
        }
        if (!seen.insert(it->second).second) {
            throw std::runtime_error("WBTPolicy: obs segment '" + name + "' listed twice");
        }
        if (hist < 1) {
            throw std::runtime_error("WBTPolicy: bad history_length for '" + name + "'");
        }
        YAML::Node term;
        term["params"]         = YAML::Node(YAML::NodeType::Map);   // {}
        term["clip"]           = YAML::Node();                       // null: no clip
        term["scale"]          = YAML::Node();                       // null: no scale
        term["history_length"] = hist;
        obs[it->second]        = term;
        spdlog::info("WBTPolicy: obs [{:4d},{:4d}) {:<22} {:3d} x {:2d}",
                     obs_total, obs_total + dim * hist, name, dim, hist);
        obs_total += dim * hist;
    }
    cfg["observations"] = obs;

    if (obs_total != deploy_cfg["obs_dim"].as<int>()) {
        throw std::runtime_error("WBTPolicy: obs_layout sum (" + std::to_string(obs_total)
                                 + ") != obs_dim (" + deploy_cfg["obs_dim"].as<std::string>() + ")");
    }
    return cfg;
}

WBTPolicy::WBTPolicy(const YAML::Node& deploy_cfg,
                     const std::vector<float>& kp,
                     const std::vector<float>& kd,
                     const std::string& motion_path,
                     const std::string& policy_path,
                     std::shared_ptr<isaaclab::Articulation> robot)
{
    // Contract checks: this build implements exactly the training semantics.
    if (deploy_cfg["use_gym_history"] && deploy_cfg["use_gym_history"].as<bool>()) {
        throw std::runtime_error("WBTPolicy: use_gym_history=true is not supported");
    }
    if (deploy_cfg["ema_alpha"] && deploy_cfg["ema_alpha"].as<float>() != 1.0f) {
        throw std::runtime_error("WBTPolicy: ema_alpha != 1 (action filter) is not supported");
    }
    anchor_is_pelvis_ = !deploy_cfg["anchor_body_name"]
        || deploy_cfg["anchor_body_name"].as<std::string>() == "pelvis";

    obs_dim_      = deploy_cfg["obs_dim"].as<int>();
    action_dim_   = deploy_cfg["action_dim"].as<int>();
    future_steps_ = deploy_cfg["future_steps"].as<std::vector<int>>();
    if (future_steps_.empty()) {
        throw std::runtime_error("WBTPolicy: future_steps must be non-empty");
    }
    default_angles_ = deploy_cfg["default_angles"].as<std::vector<float>>();
    action_scale_ = deploy_cfg["action_scale"].as<float>();
    if (!(action_scale_ > 0.0f) || !std::isfinite(action_scale_)) {
        throw std::runtime_error("WBTPolicy: action_scale must be finite and positive");
    }
    if (static_cast<int>(default_angles_.size()) != action_dim_
        || static_cast<int>(kp.size()) != action_dim_
        || static_cast<int>(kd.size()) != action_dim_) {
        throw std::runtime_error("WBTPolicy: default_angles / kp / kd must have action_dim entries");
    }

    motion_ = std::make_shared<WBTMotionLoader>(
        motion_path,
        deploy_cfg["anchor_body_idx_in_motion"].as<int>(),
        deploy_cfg["root_body_idx_in_motion"].as<int>());
    if (motion_->num_joints() != action_dim_) {
        throw std::runtime_error("WBTPolicy: motion joints (" + std::to_string(motion_->num_joints())
                                 + ") != action_dim (" + std::to_string(action_dim_) + ")");
    }
    const float ctrl_dt = deploy_cfg["ctrl_dt"].as<float>();
    if (std::abs(motion_->fps() * ctrl_dt - 1.0f) > 1e-4f) {
        throw std::runtime_error("WBTPolicy: motion fps " + std::to_string(motion_->fps())
                                 + " != 1/ctrl_dt (one motion frame per policy step)");
    }
    for (const auto& seg : deploy_cfg["obs_layout"]) {
        const auto name = seg["name"].as<std::string>();
        const int expected = name == "ref_root_pos_future_local"
            ? static_cast<int>(future_steps_.size()) * 3
            : name == "ref_root_ori_future_b"
                ? static_cast<int>(future_steps_.size()) * 6
                : name == "ref_joint_pos_future"
                    ? static_cast<int>(future_steps_.size()) * action_dim_
                    : seg["dim"].as<int>();
        if (seg["dim"].as<int>() != expected) {
            throw std::runtime_error("WBTPolicy: observation dimension mismatch for '" + name + "'");
        }
    }

    active = this;   // ObservationManager evaluates every term once while constructing
    env_ = std::make_unique<isaaclab::ManagerBasedRLEnv>(build_env_cfg(deploy_cfg, kp, kd),
                                                         std::move(robot));
    try {
        env_->alg = std::make_unique<isaaclab::OrtRunner>(policy_path);
    } catch (const std::exception& e) {
        throw std::runtime_error("WBTPolicy: failed to load " + policy_path + " (external weights "
                                 "policy.onnx.data must sit next to it): " + e.what());
    }

    const auto& ort = static_cast<const isaaclab::OrtRunner&>(*env_->alg);
    const auto& in_shapes = ort.get_input_shapes();
    const auto& out_shape = ort.get_output_shape();
    if (in_shapes.size() != 1 || std::string(ort.get_input_names()[0]) != "obs"
        || in_shapes[0] != std::vector<int64_t>{1, obs_dim_}
        || out_shape != std::vector<int64_t>{1, action_dim_}) {
        throw std::runtime_error("WBTPolicy: ONNX I/O must be obs[1," + std::to_string(obs_dim_)
                                 + "] -> [1," + std::to_string(action_dim_) + "]");
    }
    spdlog::info("WBTPolicy: policy '{}' obs[1,{}] -> action[1,{}]; motion fps={} frames={} bodies={}",
                 policy_path, obs_dim_, action_dim_,
                 motion_->fps(), motion_->num_frames(), motion_->num_bodies());
}

void WBTPolicy::begin(int start_frame)
{
    if (start_frame < 0 || start_frame > motion_->last_frame()) {
        throw std::runtime_error("WBTPolicy: start frame " + std::to_string(start_frame)
                                 + " outside the clip");
    }
    active = this;
    frame_ = start_frame;

    // Yaw-only alignment of the reference to the robot's current torso heading
    // (training starts from RSI, so robot and reference headings coincide at
    // t=0; the real robot's IMU yaw is arbitrary).  Identity when they already
    // coincide, e.g. in the sim prototype.
    const Eigen::Quaternionf anchor_quat = anchor_is_pelvis_
        ? env_->robot->data.root_quat_w : torso_quat_w(env_->robot->data);
    const Eigen::Quaternionf robot_yaw = projected_yaw_quat(anchor_quat);
    const Eigen::Quaternionf ref_yaw   = projected_yaw_quat(motion_->anchor_quat(frame_));
    init_quat_ = robot_yaw * ref_yaw.conjugate();

    env_->episode_length = 0;
    env_->global_phase   = 0.0f;
    // Historical deployment semantics: clear last_action and the one-step
    // latency slot, then run policy inference on the next control tick.
    env_->action_manager->reset();
    env_->observation_manager->reset();   // every history row := current snapshot
    has_action_ = false;
    begun_      = true;
    obs_.clear();
    raw_action_.clear();
    q_target_.clear();
}

void WBTPolicy::step()
{
    if (!begun_) {
        throw std::logic_error("WBTPolicy::step() called before begin()");
    }
    active = this;

    auto obs_map = env_->observation_manager->compute();
    const auto& obs = obs_map.at("obs");
    if (static_cast<int>(obs.size()) != obs_dim_) {
        throw std::runtime_error("WBTPolicy: assembled obs has " + std::to_string(obs.size())
                                 + " floats, expected " + std::to_string(obs_dim_));
    }
    for (std::size_t i = 0; i < obs.size(); ++i) {
        if (!std::isfinite(obs[i])) {
            throw std::runtime_error("WBTPolicy: non-finite obs[" + std::to_string(i) + "]");
        }
    }

    const auto action = env_->alg->act(obs_map);
    if (static_cast<int>(action.size()) != action_dim_) {
        throw std::runtime_error("WBTPolicy: policy returned " + std::to_string(action.size())
                                 + " actions, expected " + std::to_string(action_dim_));
    }
    for (std::size_t i = 0; i < action.size(); ++i) {
        if (!std::isfinite(action[i])) {
            throw std::runtime_error("WBTPolicy: non-finite action[" + std::to_string(i) + "]");
        }
    }
    env_->action_manager->process_action(action);

    obs_        = std::move(obs_map.at("obs"));
    raw_action_ = action;
    q_target_ = env_->action_manager->processed_actions();
    env_->episode_length += 1;
    ++frame_;
    has_action_ = true;
}
