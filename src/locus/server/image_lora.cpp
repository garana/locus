#include "locus/server/image_lora.hpp"

#include <cmath>
#include <set>
#include <stdexcept>
#include <string>

namespace locus::server {

namespace {

/** @returns v as a strength, checked to be a number from -4 to 4. */
float strength(const nlohmann::json& v) {
    if (!v.is_number()) {
        throw std::invalid_argument("lora_scale must be a number, or an "
                                    "array of numbers");
    }
    const double s = v.get<double>();
    if (!std::isfinite(s) || s < -4.0 || s > 4.0) {
        throw std::invalid_argument("lora_scale must be from -4 to 4");
    }
    return static_cast<float>(s);
}

}  // namespace

std::vector<image::LoraPick> parse_lora_fields(const nlohmann::json& body) {
    std::vector<image::LoraPick> picks;
    if (!body.contains("lora")) {
        if (body.contains("lora_scale")) {
            throw std::invalid_argument("lora_scale needs lora");
        }
        return picks;
    }
    const nlohmann::json& l = body["lora"];
    const auto name = [](const nlohmann::json& v) {
        if (!v.is_string() || v.get<std::string>().empty()) {
            throw std::invalid_argument("lora must be a non-empty name, or "
                                        "an array of them");
        }
        return v.get<std::string>();
    };
    if (l.is_array()) {
        if (l.empty() || l.size() > kMaxRequestLoras) {
            throw std::invalid_argument(
                "lora must name 1 to " + std::to_string(kMaxRequestLoras) +
                " LoRAs");
        }
        std::set<std::string> seen;
        for (const auto& v : l) {
            picks.push_back({name(v), 1.0f});
            if (!seen.insert(picks.back().name).second) {
                throw std::invalid_argument("lora names " +
                                            picks.back().name + " twice");
            }
        }
    } else {
        picks.push_back({name(l), 1.0f});
    }
    if (body.contains("lora_scale")) {
        const nlohmann::json& s = body["lora_scale"];
        if (s.is_array()) {
            if (!l.is_array() || s.size() != picks.size()) {
                throw std::invalid_argument(
                    "lora_scale must have one number per lora");
            }
            for (std::size_t i = 0; i < picks.size(); ++i) {
                picks[i].strength = strength(s[i]);
            }
        } else {
            const float v = strength(s);
            for (auto& p : picks) p.strength = v;
        }
    }
    return picks;
}

}  // namespace locus::server
