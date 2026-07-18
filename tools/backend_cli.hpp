#pragma once

#include <cstdio>
#include <string>
#include <string_view>
#include <vector>

#include "cppllm/backend/registry.hpp"

namespace cppllm_tools {

/** Prints the backend registry, marking the auto-pick with '*'. */
inline void print_backends() {
    const auto& best = cppllm::backend::best_backend();
    std::printf("backends (best first):\n");
    for (const auto& b : cppllm::backend::backends()) {
        const char* note = !b.available      ? "  [unavailable]"
                           : !b.selectable   ? "  [not selectable]"
                           : &b == &best     ? "  [default]"
                                             : "";
        std::printf("%c %-8s %s%s\n", &b == &best ? '*' : ' ',
                    std::string(b.name).c_str(),
                    std::string(b.description).c_str(), note);
    }
}

/** Flag lines shared by every tool's --help output. */
inline constexpr const char* kCommonHelp =
    "options:\n"
    "  --backend NAME   select the math backend (see --backends)\n"
    "  --backends       list available backends and exit\n"
    "  -h, --help       show this help and exit\n"
    "\n"
    "environment:\n"
    "  CPPLLM_BACKEND   backend to use when --backend is not\n"
    "                   given; same names as --backends\n";

/** Backend flags peeled off argv; the rest stays positional. */
struct BackendArgs {
    std::vector<std::string> positional;
    /** From --backend NAME or --backend=NAME; empty if absent. */
    std::string choice;
    /** --backends was given: list and exit. */
    bool list = false;
    /** --help / -h was given: print usage and exit. */
    bool help = false;
};

inline BackendArgs parse_backend_args(int argc, char** argv) {
    BackendArgs out;
    for (int i = 1; i < argc; ++i) {
        const std::string_view a = argv[i];
        if (a == "--backends") {
            out.list = true;
        } else if (a == "--help" || a == "-h") {
            out.help = true;
        } else if (a == "--backend" && i + 1 < argc) {
            out.choice = argv[++i];
        } else if (a.rfind("--backend=", 0) == 0) {
            out.choice = std::string(a.substr(10));
        } else {
            out.positional.emplace_back(a);
        }
    }
    return out;
}

}  // namespace cppllm_tools
