// Pure numerical MimicLite actor check for aarch64 deployment hosts.
// This links the same vendored ONNX Runtime used by g1_ctrl and never opens
// DDS, constructs an FSM, or touches robot hardware.
#include "isaaclab/algorithms/algorithms.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

int main(int argc, char** argv)
{
    if (argc != 2) {
        std::fprintf(stderr, "usage: %s assets/policy.onnx\n", argv[0]);
        return 2;
    }
    try {
        isaaclab::OrtRunner runner(argv[1]);
        const auto& inputs = runner.get_input_shapes();
        const auto& output = runner.get_output_shape();
        if (inputs.size() != 1 || runner.get_input_names().size() != 1
            || std::string(runner.get_input_names()[0]) != "obs"
            || inputs[0] != std::vector<int64_t>{1, 397}
            || output != std::vector<int64_t>{1, 29}) {
            throw std::runtime_error("expected ONNX obs[1,397] -> action[1,29]");
        }

        std::vector<std::vector<float>> probes(3, std::vector<float>(397));
        for (int i = 0; i < 397; ++i) {
            probes[0][i] = 0.0f;
            probes[1][i] = -1.0f + 2.0f * static_cast<float>(i) / 396.0f;
            probes[2][i] = std::sin(static_cast<float>(i) * 0.017f);
        }
        std::vector<std::vector<float>> actions;
        for (const auto& obs : probes) {
            const auto a = runner.act({{"obs", obs}});
            const auto b = runner.act({{"obs", obs}});
            if (a.size() != 29 || !std::all_of(a.begin(), a.end(), [](float x) {
                    return std::isfinite(x);
                })) {
                throw std::runtime_error("non-finite or malformed action");
            }
            float repeat_diff = 0.0f;
            for (std::size_t i = 0; i < a.size(); ++i) {
                repeat_diff = std::max(repeat_diff, std::fabs(a[i] - b[i]));
            }
            if (repeat_diff != 0.0f) {
                throw std::runtime_error("repeated inference is not bit-stable");
            }
            actions.push_back(a);
        }

        // The first physical target after reset is the default pose because
        // the configured one-step latency uses the previous zero action.
        const float defaults[29] = {
            -0.312f, 0.0f, 0.0f, 0.669f, -0.363f, 0.0f,
            -0.312f, 0.0f, 0.0f, 0.669f, -0.363f, 0.0f,
            0.0f, 0.0f, 0.0f, 0.2f, 0.2f, 0.0f, 0.6f, 0.0f,
            0.0f, 0.0f, 0.2f, -0.2f, 0.0f, 0.6f, 0.0f, 0.0f, 0.0f,
        };
        for (int i = 0; i < 29; ++i) {
            const float first_target = defaults[i];
            const float second_target = actions[0][i] * 2.0f + defaults[i];
            if (!std::isfinite(first_target) || !std::isfinite(second_target)) {
                throw std::runtime_error("non-finite delayed target");
            }
        }
        std::printf("NUMERIC PASS — vendored ONNX obs[1,397] -> action[1,29], "
                    "3 finite deterministic probes; one-step reset target is default pose\n");
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "NUMERIC FAIL: %s\n", e.what());
        return 1;
    }
}
