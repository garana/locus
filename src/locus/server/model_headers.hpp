#pragma once

#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "locus/gguf/gguf.hpp"
#include "locus/image/lora.hpp"

namespace locus::server {

/**
 * The X-Model and X-Model-Files headers of one kind of reply (i#1107),
 * built once at startup. Empty when that kind is not served.
 */
struct ModelHeaders {
    /** X-Model: what ran, as one token ("qwen2", "flux", "ltx-video"). */
    std::string model;
    /** X-Model-Files: model_file_item for each file, joined by ", ". */
    std::string files;
};

/**
 * Quotes text as a structured field string (RFC 8941): in double
 * quotes, with a backslash before each quote and backslash. Other
 * bytes outside printable ASCII become "?", which a header can not
 * carry.
 *
 * @param s The text.
 * @returns The quoted string.
 */
std::string sf_string(std::string_view s);

/**
 * One item of the X-Model-Files header: the file name, quoted, then
 * `name` (general.name), `version` (general.version, else the first 12
 * hex digits of general.source.revision) when the GGUF has them, and
 * `type`, the tensor type that holds the most bytes (whether and how
 * the weights are quantized).
 *
 * Example: "flux-q8_0.gguf";name="flux1-dev";type=q8_0
 *
 * @param file The file's name, without its directory.
 * @param g The parsed file.
 * @returns The item.
 */
std::string model_file_item(std::string_view file, const gguf::GgufFile& g);

/**
 * Builds the headers of a reply from the GGUFs it uses. Reads only the
 * files' metadata and tensor tables (they are mapped, not read).
 *
 * @param model The X-Model token.
 * @param paths The GGUF paths, in the order to list them.
 * @returns The headers.
 */
ModelHeaders model_headers(std::string model,
                           std::span<const std::string> paths);

/**
 * @param dir A model directory.
 * @returns Its "*.gguf" files, sorted by name, so the header lists
 *     them in a stable order.
 */
std::vector<std::string> gguf_files_in(const std::string& dir);

/**
 * The X-Image-Lora header of a FLUX reply: each LoRA's name, quoted,
 * and its scale, as "line-art";scale=0.6, joined by ", ".
 *
 * @param picks The LoRAs the request applied, in its order.
 * @returns The header value; empty for no LoRAs.
 */
std::string lora_header(std::span<const image::LoraPick> picks);

}  // namespace locus::server
