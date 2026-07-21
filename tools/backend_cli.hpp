#pragma once

#include <cstdio>
#include <string>
#include <string_view>
#include <vector>

#include "locus/backend/registry.hpp"
#include "locus/model/arch.hpp"

namespace locus_tools {

/** Prints the supported model architectures. */
inline void print_archs() {
    std::printf("architectures:\n");
    for (const auto& a : locus::model::archs()) {
        std::printf("  %-10s %s\n",
                    std::string(a.name).c_str(),
                    std::string(a.description).c_str());
    }
}

/** Prints the backend registry, marking the auto-pick with '*'. */
inline void print_backends() {
    const auto& best = locus::backend::best_backend();
    std::printf("backends (best first):\n");
    for (const auto& b : locus::backend::backends()) {
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
    "  --archs          list supported architectures and exit\n"
    "  -h, --help       show this help and exit\n"
    "\n"
    "environment:\n"
    "  LOCUS_BACKEND   backend to use when --backend is not\n"
    "                   given; same names as --backends\n";

/** Backend flags peeled off argv; the rest stays positional. */
struct BackendArgs {
    std::vector<std::string> positional;
    /** From --backend NAME or --backend=NAME; empty if absent. */
    std::string choice;
    /** --backends was given: list and exit. */
    bool list = false;
    /** --archs was given: list architectures and exit. */
    bool list_archs = false;
    /** --help / -h was given: print usage and exit. */
    bool help = false;
};

inline BackendArgs parse_backend_args(int argc, char** argv) {
    BackendArgs out;
    for (int i = 1; i < argc; ++i) {
        const std::string_view a = argv[i];
        if (a == "--backends") {
            out.list = true;
        } else if (a == "--archs") {
            out.list_archs = true;
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

}  // namespace locus_tools
