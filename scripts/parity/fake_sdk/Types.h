// OFFLINE TEST STUB — replaces g1_29dof/include/Types.h for rlbase_gate_test.
// Plain structs with the accessors the FSM / articulation code uses; nothing
// here creates a DDS participant, publisher or subscriber.
#pragma once

#include <array>
#include <cstdint>
#include <memory>
#include <mutex>
#include <vector>

#include "unitree/dds_wrapper/common/unitree_joystick.hpp"

namespace fake {
struct MotorCmd {
    float q_ = 0, dq_ = 0, tau_ = 0, kp_ = 0, kd_ = 0;
    uint8_t mode_ = 0;
    float& q() { return q_; }
    float& dq() { return dq_; }
    float& tau() { return tau_; }
    float& kp() { return kp_; }
    float& kd() { return kd_; }
    uint8_t& mode() { return mode_; }
};
struct LowCmdMsg {
    std::array<MotorCmd, 35> motors_;
    std::array<MotorCmd, 35>& motor_cmd() { return motors_; }
};
struct MotorState {
    float q_ = 0, dq_ = 0;
    float q() const { return q_; }
    float dq() const { return dq_; }
};
struct ImuState {
    std::array<float, 4> quaternion_{1, 0, 0, 0};
    std::array<float, 3> gyroscope_{};
    const std::array<float, 4>& quaternion() const { return quaternion_; }
    const std::array<float, 3>& gyroscope() const { return gyroscope_; }
};
struct LowStateMsg {
    ImuState imu_;
    std::array<MotorState, 35> motors_;
    const ImuState& imu_state() const { return imu_; }
    const std::array<MotorState, 35>& motor_state() const { return motors_; }
};
}  // namespace fake

// "publish" = append a copy of msg_ (what the RealTimePublisher would send).
struct FakeLowCmd {
    fake::LowCmdMsg msg_;
    std::mutex log_mtx;
    std::vector<fake::LowCmdMsg> published;
    void unlockAndPublish() {
        std::lock_guard<std::mutex> lock(log_mtx);
        published.push_back(msg_);
    }
};
struct FakeLowState {
    using SharedPtr = std::shared_ptr<FakeLowState>;
    std::mutex mutex_;
    fake::LowStateMsg msg_;
    unitree::common::UnitreeJoystick joystick;
    void update() {}
    bool isTimeout() const { return false; }
};

using LowCmd_t = FakeLowCmd;
using LowState_t = FakeLowState;
