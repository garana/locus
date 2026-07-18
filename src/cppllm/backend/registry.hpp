#pragma once

#include <span>
#include <string_view>

#include "cppllm/backend/cpu_ops.hpp"

namespace cppllm::backend {

/** Op table a backend provides to the model runtime. Ops not in
 * the table (rmsnorm, rope, ...) are negligible and stay scalar. */
struct Ops {
    void (*matvec)(const Mat&, std::span<const float>,
                   std::span<float>);
    void (*dequant_row)(const Mat&, std::uint32_t,
                        std::span<float>);
};

/**
 * One selectable math implementation. All variants are always
 * compiled in (pbw pattern); `available` reflects what the running
 * machine supports, `selectable` whether the backend can serve
 * inference today (the Vulkan entry is listed but not selectable
 * until M5 completes).
 */
struct Backend {
    std::string_view name;
    std::string_view description;
    bool available;
    bool selectable;
    Ops ops;
};

/** @returns All known backends, best-first for this build. */
std::span<const Backend> backends();

/** @returns The backend named `name`, or nullptr. */
const Backend* find_backend(std::string_view name);

/** @returns The best available, selectable backend. */
const Backend& best_backend();

/**
 * Resolves the backend to use: explicit choice > CPPLLM_BACKEND
 * env var > best available.
 *
 * @param choice Backend name from the CLI, or empty.
 * @throws std::invalid_argument on unknown or unusable names.
 */
const Backend& resolve_backend(std::string_view choice);

}  // namespace cppllm::backend
