#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <string>
#include <utility>
#include <vector>

#include "locus/gguf/gguf.hpp"
#include "locus/tok/tokenizer.hpp"

namespace {

using locus::gguf::GgufFile;
using locus::gguf::TensorInfo;
using locus::gguf::TensorType;

/** Short ggml type name for a tensor's storage type. */
const char* type_name(TensorType t) {
    switch (t) {
        case TensorType::kF32: return "F32";
        case TensorType::kF16: return "F16";
        case TensorType::kQ4_0: return "Q4_0";
        case TensorType::kQ4_1: return "Q4_1";
        case TensorType::kQ5_0: return "Q5_0";
        case TensorType::kQ5_1: return "Q5_1";
        case TensorType::kQ8_0: return "Q8_0";
        case TensorType::kQ2_K: return "Q2_K";
        case TensorType::kQ3_K: return "Q3_K";
        case TensorType::kQ4_K: return "Q4_K";
        case TensorType::kQ5_K: return "Q5_K";
        case TensorType::kQ6_K: return "Q6_K";
        case TensorType::kQ8_K: return "Q8_K";
        case TensorType::kIQ2_XXS: return "IQ2_XXS";
        case TensorType::kIQ3_XXS: return "IQ3_XXS";
        case TensorType::kIQ1_S: return "IQ1_S";
        case TensorType::kIQ4_XS: return "IQ4_XS";
        default: return "?";
    }
}

/** Human-readable byte size, e.g. "172 MB". */
std::string human_bytes(std::uint64_t b) {
    char buf[32];
    const double d = static_cast<double>(b);
    if (b >= (1ull << 30)) {
        std::snprintf(buf, sizeof buf, "%.2f GB", d / (1ull << 30));
    } else if (b >= (1ull << 20)) {
        std::snprintf(buf, sizeof buf, "%.1f MB", d / (1ull << 20));
    } else if (b >= (1ull << 10)) {
        std::snprintf(buf, sizeof buf, "%.1f KB", d / (1ull << 10));
    } else {
        std::snprintf(buf, sizeof buf, "%llu B",
                      static_cast<unsigned long long>(b));
    }
    return buf;
}

/** Human-readable parameter count, auto-scaled: 260K, 1.24B. */
std::string human_count(std::uint64_t n) {
    char buf[32];
    const double d = static_cast<double>(n);
    if (n >= 1000000000ull) {
        std::snprintf(buf, sizeof buf, "%.2fB", d / 1e9);
    } else if (n >= 1000000ull) {
        std::snprintf(buf, sizeof buf, "%.2fM", d / 1e6);
    } else if (n >= 1000ull) {
        std::snprintf(buf, sizeof buf, "%.1fK", d / 1e3);
    } else {
        std::snprintf(buf, sizeof buf, "%llu",
                      static_cast<unsigned long long>(n));
    }
    return buf;
}

/** Non-unit dims joined by x, e.g. [2048,8192,1,1] -> "2048x8192". */
std::string shape_str(const std::uint64_t ne[4]) {
    std::string s;
    for (int i = 0; i < 4; ++i) {
        if (ne[i] == 1 && i > 0) {
            continue;
        }
        if (!s.empty()) {
            s += "x";
        }
        s += std::to_string(ne[i]);
    }
    return s;
}

std::uint64_t elems(const TensorInfo& t) {
    return t.ne[0] * t.ne[1] * t.ne[2] * t.ne[3];
}

/** blk.<n>. prefix -> layer index and suffix; false for non-block. */
bool block_tensor(const std::string& name, std::uint64_t& layer,
                  std::string& suffix) {
    if (name.rfind("blk.", 0) != 0) {
        return false;
    }
    const std::size_t dot = name.find('.', 4);
    if (dot == std::string::npos) {
        return false;
    }
    layer = std::strtoull(name.c_str() + 4, nullptr, 10);
    suffix = name.substr(dot + 1);
    return true;
}

/** Ordering priority so a block reads norm -> qkv -> out -> ffn. */
int role_rank(const std::string& suffix) {
    static const std::vector<std::pair<const char*, int>> order = {
        {"attn_norm", 0},     {"attn_q", 1},
        {"attn_q_a", 1},      {"attn_q_b", 2},
        {"attn_kv_a", 3},     {"attn_k", 3},
        {"attn_k_b", 3},      {"attn_kv_b", 4},
        {"attn_v", 4},        {"attn_v_b", 4},
        {"attn_output", 9},   {"ffn_norm", 20},
        {"ffn_gate_inp", 21}, {"ffn_gate", 22},
        {"ffn_up", 23},       {"ffn_down", 24},
        {"ffn_gate_shexp", 25}, {"ffn_up_shexp", 26},
        {"ffn_down_shexp", 27}};
    int best = 50;
    for (const auto& [k, r] : order) {
        if (suffix.rfind(k, 0) == 0) {
            best = std::min(best, r);
        }
    }
    return best;
}

// Dataflow stage of a block tensor, for wiring intra-block edges.
enum Stage {
    S_ATTN_NORM,
    S_ATTN_PROJ,
    S_ATTN_OUT,
    S_FFN_NORM,
    S_FFN_ROUTER,
    S_FFN_IN,
    S_FFN_OUT,
    S_OTHER
};

int stage_of(const std::string& s) {
    if (s.rfind("attn_norm", 0) == 0) return S_ATTN_NORM;
    if (s.rfind("attn_output", 0) == 0) return S_ATTN_OUT;
    if (s.rfind("attn", 0) == 0) return S_ATTN_PROJ;
    if (s.rfind("ffn_norm", 0) == 0) return S_FFN_NORM;
    if (s.rfind("ffn_gate_inp", 0) == 0) return S_FFN_ROUTER;
    if (s.rfind("ffn_down", 0) == 0) return S_FFN_OUT;
    if (s.rfind("ffn_gate", 0) == 0 || s.rfind("ffn_up", 0) == 0)
        return S_FFN_IN;
    return S_OTHER;
}

// Which FFN sub-path a tensor belongs to, so a gate/up feeds only
// its own down: routed experts, shared experts, or a dense FFN.
std::string ffn_family(const std::string& s) {
    if (s.find("_exps") != std::string::npos) return "exps";
    if (s.find("_shexp") != std::string::npos) return "shexp";
    return "dense";
}

bool has_output(const std::string& name) {
    return name.find("output") != std::string::npos;
}

/** Renders a mermaid flowchart of the model's block structure. */
void emit_mermaid(const GgufFile& g) {
    const std::string arch = std::string(
        g.get_string("general.architecture").value_or("?"));
    const std::string name =
        std::string(g.get_string("general.name").value_or(""));

    std::uint64_t total_bytes = 0, total_params = 0;
    std::map<std::uint64_t, std::vector<const TensorInfo*>> layers;
    std::vector<const TensorInfo*> globals;
    for (const auto& t : g.tensors()) {
        total_bytes += t.nbytes;
        total_params += elems(t);
        std::uint64_t l;
        std::string suf;
        if (block_tensor(t.name, l, suf)) {
            layers[l].push_back(&t);
        } else {
            globals.push_back(&t);
        }
    }

    // Collapse identical layers by their sorted suffix set.
    auto signature = [](const std::vector<const TensorInfo*>& ts) {
        std::vector<std::string> sufs;
        for (const auto* t : ts) {
            std::uint64_t l;
            std::string suf;
            block_tensor(t->name, l, suf);
            sufs.push_back(suf);
        }
        std::sort(sufs.begin(), sufs.end());
        std::string s;
        for (const auto& x : sufs) {
            s += x + ";";
        }
        return s;
    };
    struct Group {
        std::vector<const TensorInfo*> rep;
        std::uint64_t count = 0;
        std::uint64_t bytes = 0;
    };
    std::map<std::uint64_t, Group> groups;  // keyed by first layer
    std::map<std::string, std::uint64_t> sig_first;
    for (auto& [l, ts] : layers) {
        const std::string sig = signature(ts);
        auto it = sig_first.find(sig);
        if (it == sig_first.end()) {
            sig_first[sig] = l;
            auto& gr = groups[l];
            gr.rep = ts;
            for (const auto* t : ts) {
                gr.bytes += t->nbytes;
            }
            gr.count = 1;
        } else {
            groups[it->second].count++;
        }
    }

    std::printf(
        "%%%% %s  arch=%s  layers=%zu  params=%s  size=%s\n",
        name.empty() ? "(model)" : name.c_str(), arch.c_str(),
        layers.size(), human_count(total_params).c_str(),
        human_bytes(total_bytes).c_str());
    std::printf("flowchart TD\n");

    int nid = 0;
    auto full_node = [&](const TensorInfo* t) {
        const int id = nid++;
        std::printf("  n%d[\"%s<br/>%s | %s | %s\"]\n", id,
                    t->name.c_str(), shape_str(t->ne).c_str(),
                    type_name(t->type),
                    human_bytes(t->nbytes).c_str());
        return id;
    };

    std::sort(globals.begin(), globals.end(),
              [](const TensorInfo* a, const TensorInfo* b) {
                  auto rank = [](const std::string& n) {
                      if (n.find("token_embd") != std::string::npos)
                          return 0;
                      if (n.find("output_norm") != std::string::npos)
                          return 8;
                      if (has_output(n)) return 9;
                      return 5;
                  };
                  return rank(a->name) < rank(b->name);
              });

    // Input-side globals (embeddings) chained top to bottom.
    int prev = -1;
    for (const auto* t : globals) {
        if (has_output(t->name)) {
            continue;
        }
        const int id = full_node(t);
        if (prev >= 0) {
            std::printf("  n%d --> n%d\n", prev, id);
        }
        prev = id;
    }

    // One subgraph per distinct block signature.
    int last_blk = -1;
    int sg = 0;
    for (auto& [first, gr] : groups) {
        (void)first;
        const int me = sg++;
        const bool moe = std::any_of(
            gr.rep.begin(), gr.rep.end(), [](const TensorInfo* t) {
                return t->name.find("_exps") != std::string::npos ||
                       t->name.find("gate_inp") !=
                           std::string::npos;
            });
        std::printf(
            "  subgraph blk%d[\"block x%llu -- %s each%s\"]\n", me,
            static_cast<unsigned long long>(gr.count),
            human_bytes(gr.bytes).c_str(), moe ? " (MoE)" : "");
        std::printf("    direction TB\n");
        auto rep = gr.rep;
        std::sort(rep.begin(), rep.end(),
                  [](const TensorInfo* a, const TensorInfo* b) {
                      std::uint64_t l;
                      std::string sa, sb;
                      block_tensor(a->name, l, sa);
                      block_tensor(b->name, l, sb);
                      const int ra = role_rank(sa),
                                rb = role_rank(sb);
                      return ra != rb ? ra < rb : sa < sb;
                  });
        struct BN {
            int id;
            int stage;
            std::string fam;
        };
        std::vector<BN> bn;
        for (const auto* t : rep) {
            std::uint64_t l;
            std::string suf;
            block_tensor(t->name, l, suf);
            const int id = nid++;
            std::printf("    n%d[\"%s<br/>%s | %s | %s\"]\n", id,
                        suf.c_str(), shape_str(t->ne).c_str(),
                        type_name(t->type),
                        human_bytes(t->nbytes).c_str());
            bn.push_back({id, stage_of(suf), ffn_family(suf)});
        }
        // Intra-block dataflow: norm -> projections -> attn_output
        // -> ffn_norm -> (router ->) gate/up -> down, matched by
        // FFN family so experts and shared experts wire separately.
        auto e = [](int a, int b) {
            if (a >= 0 && b >= 0) {
                std::printf("    n%d --> n%d\n", a, b);
            }
        };
        auto first_stage = [&](int st) {
            for (const auto& x : bn) {
                if (x.stage == st) {
                    return x.id;
                }
            }
            return -1;
        };
        const int an = first_stage(S_ATTN_NORM);
        const int ao = first_stage(S_ATTN_OUT);
        const int fn = first_stage(S_FFN_NORM);
        const int fr = first_stage(S_FFN_ROUTER);
        for (const auto& x : bn) {
            if (x.stage == S_ATTN_PROJ) {
                e(an, x.id);
                e(x.id, ao);
            }
        }
        e(ao, fn);
        e(fn, fr);
        for (const auto& x : bn) {
            if (x.stage != S_FFN_IN) {
                continue;
            }
            e(x.fam == "exps" && fr >= 0 ? fr : fn, x.id);
            for (const auto& y : bn) {
                if (y.stage == S_FFN_OUT && y.fam == x.fam) {
                    e(x.id, y.id);
                }
            }
        }
        std::printf("  end\n");
        if (prev >= 0) {
            std::printf("  n%d --> blk%d\n", prev, me);
        }
        last_blk = me;
    }

    // Output-side globals (output_norm, output) after the blocks.
    prev = -1;
    for (const auto* t : globals) {
        if (!has_output(t->name)) {
            continue;
        }
        const int id = full_node(t);
        if (prev < 0) {
            if (last_blk >= 0) {
                std::printf("  blk%d --> n%d\n", last_blk, id);
            }
        } else {
            std::printf("  n%d --> n%d\n", prev, id);
        }
        prev = id;
    }
}

}  // namespace

/**
 * GGUF inspection tool (stable CLI for probing models):
 *   locus-gguf <model.gguf> info            metadata + counts
 *   locus-gguf <model.gguf> tensor <name>   one tensor's shape
 *   locus-gguf <model.gguf> tokens <text>   token ids for text
 *   locus-gguf <model.gguf> mermaid         block-structure diagram
 */
int main(int argc, char** argv) {
    if (argc < 3) {
        std::fprintf(stderr,
                     "usage: %s <model.gguf> info\n"
                     "       %s <model.gguf> tensor <name>\n"
                     "       %s <model.gguf> tokens <text>\n"
                     "       %s <model.gguf> mermaid\n",
                     argv[0], argv[0], argv[0], argv[0]);
        return 2;
    }
    try {
        auto g = locus::gguf::GgufFile::open(argv[1]);
        const std::string cmd = argv[2];
        if (cmd == "info") {
            for (const auto& [k, v] : g.metadata()) {
                std::printf("%s = u%llu i%lld f%g '%s'\n",
                            k.c_str(),
                            (unsigned long long)v.u,
                            (long long)v.i, v.f, v.s.c_str());
            }
            std::printf("total tensors: %zu\n",
                        g.total_tensor_count());
        } else if (cmd == "tensor" && argc > 3) {
            const auto* t = g.find_tensor(argv[3]);
            if (t == nullptr) {
                std::printf("not found: %s\n", argv[3]);
                return 1;
            }
            std::printf(
                "%s ne=[%llu,%llu,%llu,%llu] type=%u "
                "bytes=%llu first=%02x\n",
                t->name.c_str(),
                (unsigned long long)t->ne[0],
                (unsigned long long)t->ne[1],
                (unsigned long long)t->ne[2],
                (unsigned long long)t->ne[3], (unsigned)t->type,
                (unsigned long long)t->nbytes,
                (unsigned)g.tensor_data(*t)[0]);
        } else if (cmd == "tokens" && argc > 3) {
            auto tok = locus::tok::tokenizer_from_gguf(g);
            for (auto id : tok->encode(argv[3], true)) {
                std::printf("%d ", id);
            }
            std::printf("\n");
        } else if (cmd == "mermaid") {
            emit_mermaid(g);
        } else {
            std::fprintf(stderr, "unknown command: %s\n",
                         cmd.c_str());
            return 2;
        }
    } catch (const std::exception& e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }
    return 0;
}
