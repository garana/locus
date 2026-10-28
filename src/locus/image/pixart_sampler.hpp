#pragma once

#include <cstdint>
#include <functional>
#include <span>
#include <vector>

#include "locus/backend/registry.hpp"
#include "locus/gguf/gguf.hpp"
#include "locus/image/pixart_dit.hpp"

namespace locus::image {

/**
 * diffusers' DPMSolverMultistepScheduler as PixArt-alpha configures it
 * (i#146; docs/models/pixart-alpha.md section 5): DPM-Solver++ of
 * order 2 with the midpoint update, epsilon prediction, linear betas,
 * linspace timesteps, a first-order first and last step, and a final
 * sigma of 0.
 *
 * "sigma" is the noise-to-signal ratio sqrt((1 - abar) / abar) at a
 * training timestep. Each step turns the noise prediction into a
 * prediction of the clean latent, x0 = (x - sigma_s eps) / alpha_s,
 * and moves x from sigma_i to sigma_{i+1}. The last step lands on
 * sigma 0, where the result is the last x0.
 *
 * The scalars follow the reference's float32 steps (the schedule's
 * cumulative product in double, cast to float, as torch's CPU cumprod
 * does).
 */
class DpmSolverPP {
  public:
    /**
     * @param steps Inference steps (8 for the reference), 1 to 1000.
     * @param train_steps The training schedule length (1000).
     * @param beta_start, beta_end The linear beta range.
     * @throws std::invalid_argument on a bad step count or range.
     */
    DpmSolverPP(std::uint32_t steps, std::uint32_t train_steps = 1000,
                float beta_start = 0.0001f, float beta_end = 0.02f);

    /**
     * Reads the pixart.sampler.* keys (convert-pixart.py) and refuses a
     * configuration this class does not implement.
     *
     * @throws gguf::Error on a missing key or another algorithm.
     */
    static DpmSolverPP from_gguf(const gguf::GgufFile& g,
                                 std::uint32_t steps);

    /** @returns The steps' timesteps, high to low (999 ... 125). */
    const std::vector<std::int32_t>& timesteps() const { return t_; }
    /** @returns steps + 1 sigmas, the last 0. */
    const std::vector<float>& sigmas() const { return sigma_; }
    /** @returns The index of the next step (0 to steps). */
    std::uint32_t index() const { return index_; }

    /**
     * One step: x moves from sigmas()[index()] to the next sigma.
     *
     * @param eps The noise prediction for x, same size as x.
     * @param x The latent, updated in place.
     * @throws std::logic_error after the last step, and
     *     std::invalid_argument on a size change.
     */
    void step(std::span<const float> eps, std::span<float> x);

  private:
    std::vector<std::int32_t> t_;
    std::vector<float> sigma_;
    std::uint32_t index_ = 0;
    std::vector<float> prev_x0_;  ///< the previous step's x0
};

/**
 * The pipeline's denoising loop (section 5.5): per step, the DiT on
 * the negative and the prompt T5 rows, classifier-free guidance
 * e = E_neg + guidance (E_pos - E_neg) on the 4 noise channels, then a
 * sampler step.
 *
 * @param dit The denoiser.
 * @param sched A fresh sampler; all its steps run.
 * @param latent The start latent [4][S][S], denoised in place.
 * @param neg, n_neg The valid T5 rows of the negative prompt.
 * @param pos, n_pos The valid T5 rows of the prompt.
 * @param guidance The CFG scale (4.5).
 * @param op The backend ops.
 * @param keep_going Called before each DiT call with the step index,
 *     may be empty; returning false stops the loop (a client that went
 *     away).
 * @returns true when every step ran, false when keep_going stopped it.
 */
bool pixart_denoise(
    const PixArtDit& dit, DpmSolverPP& sched, std::span<float> latent,
    std::span<const float> neg, std::uint32_t n_neg,
    std::span<const float> pos, std::uint32_t n_pos, float guidance,
    const backend::Ops& op,
    const std::function<bool(std::uint32_t)>& keep_going = {});

}  // namespace locus::image
