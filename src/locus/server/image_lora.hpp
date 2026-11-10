#pragma once

#include <vector>

#include "json.hpp"
#include "locus/image/lora.hpp"

namespace locus::server {

/** The most LoRAs one image request may name. */
inline constexpr std::size_t kMaxRequestLoras = 8;

/**
 * Reads the LoRA fields of a FLUX image request (i#372):
 *
 * - "lora": a LoRA name, or an array of 1 to kMaxRequestLoras distinct
 *   names (the --lora-dir file names without ".gguf");
 * - "lora_scale": the strength, a number from -4 to 4 (default 1); an
 *   array of as many numbers when "lora" is an array, or one number for
 *   all of them.
 *
 * @returns The picks in request order; empty when there is no "lora".
 * @throws std::invalid_argument naming the bad field.
 */
std::vector<image::LoraPick> parse_lora_fields(const nlohmann::json& body);

}  // namespace locus::server
