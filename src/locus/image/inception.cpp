#include "locus/image/inception.hpp"

#include <stdexcept>

namespace locus::image {

namespace {

/** Inception-v3's conv count: the stem's 5, A 7 x 3, B 4, C 10 x 4,
 *  D 6, E 9 x 2. */
constexpr std::size_t kConvs = 94;

/** Whether this thread's run() puts its convs on Vulkan: set there
 *  for the blocks' conv() calls, so they need not all pass it. */
thread_local bool t_gpu = false;

}  // namespace

Inception Inception::load(const gguf::GgufFile& g) {
    const std::string p = "inception.";
    Inception m;
    m.pre_ = cnn::read_preprocess(g, p);
    m.labels_ = cnn::read_labels(g, p);
    for (const auto& ti : g.tensors()) {
        const std::string& n = ti.name;
        constexpr std::string_view kW = ".weight";
        if (n.size() > kW.size() && n.ends_with(kW) && ti.n_dims == 4) {
            const std::string base = n.substr(0, n.size() - kW.size());
            m.convs_.emplace(base, cnn::read_conv(g, base));
        }
    }
    if (m.convs_.size() != kConvs) {
        throw gguf::Error("inception: " + std::to_string(m.convs_.size()) +
                          " convs, not " + std::to_string(kConvs));
    }
    m.fc_w_ = cnn::read_f32(g, "fc.weight",
                            std::size_t{m.num_classes()} * 2048);
    m.fc_b_ = cnn::read_f32(g, "fc.bias", m.num_classes());
    return m;
}

Inception::Map Inception::conv(const std::string& name, const Map& in,
                               std::uint32_t stride, bool same) const {
    const auto it = convs_.find(name);
    if (it == convs_.end()) {
        throw std::runtime_error("inception: no conv " + name);
    }
    const cnn::Conv& c = it->second;
    if (c.in != in.c) {
        throw std::runtime_error("inception: " + name + " takes " +
                                 std::to_string(c.in) + " channels, not " +
                                 std::to_string(in.c));
    }
    Map out{{}, c.out, in.h, in.w};
    out.x = cnn::conv(c, in.x, out.h, out.w, stride, same ? c.kh / 2 : 0,
                      same ? c.kw / 2 : 0, true, t_gpu);
    return out;
}

namespace {

/** Concatenates feature maps of one size on the channel axis. */
template <typename M>
M cat(std::initializer_list<const M*> parts) {
    M out;
    out.h = (*parts.begin())->h;
    out.w = (*parts.begin())->w;
    for (const M* p : parts) {
        if (p->h != out.h || p->w != out.w) {
            throw std::runtime_error("inception: branch sizes differ");
        }
        out.x.insert(out.x.end(), p->x.begin(), p->x.end());
        out.c += p->c;
    }
    return out;
}

}  // namespace

Inception::Map Inception::block_a(const std::string& p, const Map& x) const {
    const Map b1 = conv(p + "branch1x1", x, 1, true);
    const Map b5 = conv(p + "branch5x5_2", conv(p + "branch5x5_1", x, 1, true),
                        1, true);
    Map d = conv(p + "branch3x3dbl_1", x, 1, true);
    d = conv(p + "branch3x3dbl_2", d, 1, true);
    d = conv(p + "branch3x3dbl_3", d, 1, true);
    const Map pool{cnn::avg_pool3(x.x, x.c, x.h, x.w), x.c, x.h, x.w};
    const Map bp = conv(p + "branch_pool", pool, 1, true);
    return cat<Map>({&b1, &b5, &d, &bp});
}

Inception::Map Inception::block_b(const std::string& p, const Map& x) const {
    const Map b3 = conv(p + "branch3x3", x, 2, false);
    Map d = conv(p + "branch3x3dbl_1", x, 1, true);
    d = conv(p + "branch3x3dbl_2", d, 1, true);
    d = conv(p + "branch3x3dbl_3", d, 2, false);
    Map mp{{}, x.c, x.h, x.w};
    mp.x = cnn::max_pool3(x.x, x.c, mp.h, mp.w, 2, 0);
    return cat<Map>({&b3, &d, &mp});
}

Inception::Map Inception::block_c(const std::string& p, const Map& x) const {
    const Map b1 = conv(p + "branch1x1", x, 1, true);
    Map b7 = conv(p + "branch7x7_1", x, 1, true);
    b7 = conv(p + "branch7x7_2", b7, 1, true);
    b7 = conv(p + "branch7x7_3", b7, 1, true);
    Map d = conv(p + "branch7x7dbl_1", x, 1, true);
    for (const char* n : {"branch7x7dbl_2", "branch7x7dbl_3", "branch7x7dbl_4",
                          "branch7x7dbl_5"}) {
        d = conv(p + n, d, 1, true);
    }
    const Map pool{cnn::avg_pool3(x.x, x.c, x.h, x.w), x.c, x.h, x.w};
    const Map bp = conv(p + "branch_pool", pool, 1, true);
    return cat<Map>({&b1, &b7, &d, &bp});
}

Inception::Map Inception::block_d(const std::string& p, const Map& x) const {
    const Map b3 = conv(p + "branch3x3_2", conv(p + "branch3x3_1", x, 1, true),
                        2, false);
    Map b7 = conv(p + "branch7x7x3_1", x, 1, true);
    b7 = conv(p + "branch7x7x3_2", b7, 1, true);
    b7 = conv(p + "branch7x7x3_3", b7, 1, true);
    b7 = conv(p + "branch7x7x3_4", b7, 2, false);
    Map mp{{}, x.c, x.h, x.w};
    mp.x = cnn::max_pool3(x.x, x.c, mp.h, mp.w, 2, 0);
    return cat<Map>({&b3, &b7, &mp});
}

Inception::Map Inception::block_e(const std::string& p, const Map& x) const {
    const Map b1 = conv(p + "branch1x1", x, 1, true);
    const Map b3 = conv(p + "branch3x3_1", x, 1, true);
    const Map b3a = conv(p + "branch3x3_2a", b3, 1, true);
    const Map b3b = conv(p + "branch3x3_2b", b3, 1, true);
    const Map b3c = cat<Map>({&b3a, &b3b});
    Map d = conv(p + "branch3x3dbl_1", x, 1, true);
    d = conv(p + "branch3x3dbl_2", d, 1, true);
    const Map da = conv(p + "branch3x3dbl_3a", d, 1, true);
    const Map db = conv(p + "branch3x3dbl_3b", d, 1, true);
    const Map dc = cat<Map>({&da, &db});
    const Map pool{cnn::avg_pool3(x.x, x.c, x.h, x.w), x.c, x.h, x.w};
    const Map bp = conv(p + "branch_pool", pool, 1, true);
    return cat<Map>({&b1, &b3c, &dc, &bp});
}

Inception::Output Inception::run(std::span<const float> chw,
                                 bool gpu) const {
    if (chw.size() != std::size_t{3} * pre_.size * pre_.size) {
        throw std::invalid_argument("inception: need [3][size][size]");
    }
    struct Restore {
        bool prev;
        ~Restore() { t_gpu = prev; }
    } restore{t_gpu};
    t_gpu = gpu;
    Map x{{chw.begin(), chw.end()}, 3, pre_.size, pre_.size};
    x = conv("Conv2d_1a_3x3", x, 2, false);
    x = conv("Conv2d_2a_3x3", x, 1, false);
    x = conv("Conv2d_2b_3x3", x, 1, true);
    x.x = cnn::max_pool3(x.x, x.c, x.h, x.w, 2, 0);
    x = conv("Conv2d_3b_1x1", x, 1, false);
    x = conv("Conv2d_4a_3x3", x, 1, false);
    x.x = cnn::max_pool3(x.x, x.c, x.h, x.w, 2, 0);
    for (const char* b : {"Mixed_5b.", "Mixed_5c.", "Mixed_5d."}) {
        x = block_a(b, x);
    }
    x = block_b("Mixed_6a.", x);
    for (const char* b : {"Mixed_6b.", "Mixed_6c.", "Mixed_6d.", "Mixed_6e."}) {
        x = block_c(b, x);
    }
    x = block_d("Mixed_7a.", x);
    for (const char* b : {"Mixed_7b.", "Mixed_7c."}) x = block_e(b, x);
    if (x.c != 2048) {
        throw std::runtime_error("inception: the last block gives " +
                                 std::to_string(x.c) + " channels");
    }
    Output out;
    out.embedding = cnn::global_avg(x.x, x.c, std::size_t{x.h} * x.w);
    out.logits = cnn::linear(fc_w_, fc_b_, out.embedding);
    return out;
}

}  // namespace locus::image
