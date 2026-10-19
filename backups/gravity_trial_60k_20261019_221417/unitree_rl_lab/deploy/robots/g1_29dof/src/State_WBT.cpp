// SPDX-FileCopyrightText: 2026 deploy_ws (parallel WBT chain).
#include "State_WBT.h"

#include "unitree_articulation.h"
#include "isaaclab/envs/mdp/terminations.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <unistd.h>   // [DIAG] fsync
#include <iomanip>   // [DIAG]
#include <sstream>   // [DIAG]
#include <stdexcept>
#include <ctime>     // [DIAG]


static std::filesystem::path resolve_path(const std::string& s)
{
    std::filesystem::path p(s);
    return p.is_absolute() ? p : (param::proj_dir / p);
}

State_WBT::State_WBT(int state_mode, std::string state_string)
: FSMState(state_mode, state_string)
{
    auto fsm_cfg = param::config["FSM"][state_string];
    if (!fsm_cfg) {
        throw std::runtime_error("State_WBT: no FSM[" + state_string + "] config block");
    }

    const auto deploy_cfg_path = resolve_path(fsm_cfg["deploy_config"].as<std::string>());
    const auto policy_path     = resolve_path(fsm_cfg["policy"       ].as<std::string>());
    const auto motion_path     = resolve_path(fsm_cfg["motion"       ].as<std::string>());

    if (!std::filesystem::exists(deploy_cfg_path)) {
        throw std::runtime_error("State_WBT: deploy_config missing: " + deploy_cfg_path.string());
    }
    if (!std::filesystem::exists(policy_path)) {
        throw std::runtime_error("State_WBT: policy missing: " + policy_path.string());
    }
    if (!std::filesystem::exists(motion_path)) {
        throw std::runtime_error("State_WBT: motion missing: " + motion_path.string());
    }

    auto articulation = std::make_shared<unitree::BaseArticulation<LowState_t::SharedPtr>>(
        FSMState::lowstate);
    policy_ = std::make_unique<WBTPolicy>(
        YAML::LoadFile(deploy_cfg_path.string()),
        fsm_cfg["kp"].as<std::vector<float>>(),
        fsm_cfg["kd"].as<std::vector<float>>(),
        motion_path.string(), policy_path.string(), articulation);

    // time_start / time_end: clip time in seconds -> inclusive frame range.
    const auto& motion = policy_->motion();
    auto to_frame = [&](float t) {
        return std::clamp(static_cast<int>(std::lround(t * motion.fps())), 0, motion.last_frame());
    };
    start_frame_ = fsm_cfg["time_start"] ? to_frame(fsm_cfg["time_start"].as<float>()) : 0;
    end_frame_   = (fsm_cfg["time_end"] && !fsm_cfg["time_end"].IsNull())
        ? std::max(start_frame_, to_frame(fsm_cfg["time_end"].as<float>()))
        : motion.last_frame();
    spdlog::info("State_WBT: motion '{}' frames [{}, {}] of {} @ {} fps",
                 motion_path.filename().string(), start_frame_, end_frame_,
                 motion.num_frames(), motion.fps());

    // Checks are evaluated in order; the first that fires wins.
    if (FSMStringMap.right.count("Passive")) {
        // Policy thread failed (non-finite obs/action, exception) -> damping.
        this->registered_checks.emplace_back(
            std::make_pair([this]() -> bool { return fault_.load(); },
                           FSMStringMap.right.at("Passive")));
        // Bad-orientation safety (same threshold as State_Mimic), evaluated on
        // this entry's snapshots only.
        this->registered_checks.emplace_back(
            std::make_pair([this]() -> bool { return bad_orientation_.load(); },
                           FSMStringMap.right.at("Passive")));
    }
    // Clip end -> Velocity (active-balance RL locomotion policy; FixStand's
    // static joint-space PD interpolation cannot absorb residual momentum).
    if (FSMStringMap.right.count("Velocity")) {
        this->registered_checks.emplace_back(
            std::make_pair([this]() -> bool { return motion_done_.load(); },
                           FSMStringMap.right.at("Velocity")));
    }
}

void State_WBT::enter()
{
    if (policy_thread_.joinable()) {   // defensive: never run two policy threads
        policy_thread_running_.store(false);
        policy_thread_.join();
    }
    target_ready_.store(false);
    motion_done_.store(false);
    bad_orientation_.store(false);
    fault_.store(false);

    // Gains are applied together with the first valid q target in run(); the
    // previous state's command stays on the bus until then.

    // [DIAG] open a fresh csv for this entry: log/wbt_diag_<time>.csv
    // FILE-based, per-row fflush + periodic fsync: survives crash and unclean
    // poweroff (the 2026-10-19 session lost all rows to a power cut).
    {
        std::time_t now = std::time(nullptr);
        std::tm tmv{};
        localtime_r(&now, &tmv);
        char ts[32];
        std::strftime(ts, sizeof(ts), "%Y%m%d_%H%M%S", &tmv);
        auto dir = param::proj_dir / "log";
        std::error_code ec;
        std::filesystem::create_directories(dir, ec);
        auto path = dir / ("wbt_diag_" + std::string(ts) + ".csv");
        diag_fs_ = std::fopen(path.c_str(), "w");
        diag_rows_ = 0;
        diag_t0_ = std::chrono::steady_clock::now();
        if (diag_fs_) {
            const auto& d = policy_->env()->robot->data;
            std::fprintf(diag_fs_, "# [DIAG] WBT per-step log, t_ms since entry; J=29 policy order\n# kp:");
            for (std::size_t i = 0; i < d.joint_stiffness.size(); ++i) std::fprintf(diag_fs_, ",%g", d.joint_stiffness[i]);
            std::fprintf(diag_fs_, "\n# kd:");
            for (std::size_t i = 0; i < d.joint_damping.size(); ++i) std::fprintf(diag_fs_, ",%g", d.joint_damping[i]);
            std::fprintf(diag_fs_, "\nt_ms,frame");
            for (const char* p : {"tq","a","mq","mdq","tau","ojp","ola"}) {
                for (int j = 0; j < 29; ++j) std::fprintf(diag_fs_, ",%s%d", p, j);
            }
            std::fprintf(diag_fs_, "\n");
            std::fflush(diag_fs_);
            ::fsync(::fileno(diag_fs_));
            spdlog::info("State_WBT: [DIAG] per-step csv enabled: {}", path.string());
        } else {
            spdlog::warn("State_WBT: [DIAG] cannot open csv, continuing without");
        }
    }

    policy_thread_running_.store(true);
    policy_thread_ = std::thread(&State_WBT::policy_loop, this);
}

void State_WBT::policy_loop()
{
    using clock = std::chrono::steady_clock;
    const auto dt = std::chrono::duration_cast<clock::duration>(
        std::chrono::duration<double>(policy_->step_dt()));

    auto* env = policy_->env();
    bool first = true;
    auto next = clock::now();
    try {
        while (policy_thread_running_.load()) {
            if (first || policy_->frame() <= end_frame_) {
                env->robot->update();
                if (isaaclab::mdp::bad_orientation(env, 1.0f)) {
                    // No new target; the FSM drops to Passive on its next tick.
                    bad_orientation_.store(true);
                } else {
                    if (first) { spdlog::info("State_WBT: [DIAG] first robot update ok");
                        // Fresh episode on this snapshot: history primed, last
                        // action zeroed, reference yaw-aligned to the torso.
                        policy_->begin(start_frame_);
                        first = false;
                    }
                    policy_->step();
                    {
                        std::lock_guard<std::mutex> lock(target_mtx_);
                        q_target_ = policy_->q_target();
                    }
                    target_ready_.store(true);

                    // [DIAG] one csv row per policy step (policy thread).
                    if (diag_fs_) {
                        const auto t_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                            std::chrono::steady_clock::now() - diag_t0_).count();
                        const auto& d = env->robot->data;
                        const auto& tq = policy_->q_target();
                        const auto& ra = policy_->raw_action();
                        const auto& ob = policy_->obs();
                        std::fprintf(diag_fs_, "%lld,%d",
                                     static_cast<long long>(t_ms), policy_->frame());
                        for (float v : tq) std::fprintf(diag_fs_, ",%.5f", v);
                        for (float v : ra) std::fprintf(diag_fs_, ",%.5f", v);
                        for (std::size_t i = 0; i < tq.size(); ++i) std::fprintf(diag_fs_, ",%.5f", d.joint_pos[i]);
                        for (std::size_t i = 0; i < tq.size(); ++i) std::fprintf(diag_fs_, ",%.5f", d.joint_vel[i]);
                        for (std::size_t i = 0; i < tq.size(); ++i)
                            std::fprintf(diag_fs_, ",%.3f",
                                d.joint_stiffness[i] * (tq[i] - d.joint_pos[i])
                                - d.joint_damping[i] * d.joint_vel[i]);
                        for (std::size_t i = 339; i < 368 && i < ob.size(); ++i) std::fprintf(diag_fs_, ",%.5f", ob[i]);
                        for (std::size_t i = 0; i < 29 && i < ob.size(); ++i) std::fprintf(diag_fs_, ",%.5f", ob[i]);
                        std::fprintf(diag_fs_, "\n");
                        std::fflush(diag_fs_);
                        if (++diag_rows_ % 25 == 0) ::fsync(::fileno(diag_fs_));
                        if (diag_rows_ == 1) spdlog::info("State_WBT: [DIAG] first policy step ok (frame {})", policy_->frame());
                    }
                }
            } else {
                // The last frame's target was held for its full step.
                motion_done_.store(true);
            }
            next += dt;
            std::this_thread::sleep_until(next);
        }
    } catch (const std::exception& e) {
        spdlog::error("State_WBT: policy step failed at motion frame {}: {}", policy_->frame(), e.what());
        if (diag_fs_) { std::fflush(diag_fs_); ::fsync(::fileno(diag_fs_)); }   // [DIAG]
        fault_.store(true);
    }
}

void State_WBT::run()
{
    if (!target_ready_.load()) {
        return;
    }
    std::vector<float> q;
    {
        std::lock_guard<std::mutex> lock(target_mtx_);
        q = q_target_;
    }
    // Identity joint_ids_map for UniLab => motor index == policy index.  We
    // still go through joint_ids_map so a future non-identity remap works.
    const auto& data = policy_->env()->robot->data;
    for (std::size_t i = 0; i < data.joint_stiffness.size(); ++i) {
        auto& motor = lowcmd->msg_.motor_cmd()[i];
        motor.kp()  = data.joint_stiffness[i];
        motor.kd()  = data.joint_damping[i];
        motor.dq()  = 0.0f;
        motor.tau() = 0.0f;
    }
    const auto& ids = data.joint_ids_map;
    for (std::size_t i = 0; i < ids.size(); ++i) {
        lowcmd->msg_.motor_cmd()[static_cast<int>(ids[i])].q() = q[i];
    }
}

void State_WBT::exit()
{
    policy_thread_running_.store(false);
    // [DIAG] flush, fsync and close the per-step csv.
    if (diag_fs_) { std::fflush(diag_fs_); ::fsync(::fileno(diag_fs_)); std::fclose(diag_fs_); diag_fs_ = nullptr; }
    if (policy_thread_.joinable()) {
        policy_thread_.join();
    }
    target_ready_.store(false);
}
