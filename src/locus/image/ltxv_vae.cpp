#include "locus/image/ltxv_vae.hpp"

#include <algorithm>
#include <cmath>
#include <functional>
#include <stdexcept>
#include <string>

#include "locus/image/pixart_dit.hpp"
#include "locus/sys/thread_pool.hpp"

namespace locus::image {

namespace {

/** Runs fn(i) for i in [0, n) on the thread pool. */
void parallel(std::size_t n, const std::function<void(std::size_t)>& fn) {
    sys::ThreadPool::instance().parallel_for(n, fn);
}

float silu(float v) { return v / (1.0f + std::exp(-v)); }

/**
 * The reference's PixelNorm (eps 1e-8) over the channels of x
 * [ch][plane], in place: each position divided by the RMS of its ch
 * values.
 */
void pixel_norm(std::span<float> x, std::uint32_t ch, std::size_t plane) {
    constexpr std::size_t kChunk = 1024;
    parallel((plane + kChunk - 1) / kChunk, [&](std::size_t item) {
        const std::size_t p0 = item * kChunk;
        const std::size_t p1 = std::min(plane, p0 + kChunk);
        float ss[kChunk] = {};
        for (std::uint32_t c = 0; c < ch; ++c) {
            const float* row = &x[c * plane];
            for (std::size_t p = p0; p < p1; ++p) ss[p - p0] += row[p] * row[p];
        }
        for (std::size_t p = p0; p < p1; ++p) {
            ss[p - p0] =
                1.0f / std::sqrt(ss[p - p0] / static_cast<float>(ch) + 1e-8f);
        }
        for (std::uint32_t c = 0; c < ch; ++c) {
            float* row = &x[c * plane];
            for (std::size_t p = p0; p < p1; ++p) row[p] *= ss[p - p0];
        }
    });
}

/**
 * x[c][p] = silu(x[c][p] * (1 + scale[c]) + shift[c]) over x
 * [ch][plane], in place.
 */
void modulate_silu(std::span<float> x, std::uint32_t ch, std::size_t plane,
                   const float* shift, const float* scale) {
    parallel(ch, [&](std::size_t c) {
        float* row = &x[c * plane];
        const float s = 1.0f + scale[c], b = shift[c];
        for (std::size_t p = 0; p < plane; ++p) row[p] = silu(row[p] * s + b);
    });
}

/**
 * The reference's PixelShuffleND(3, (2, 2, 2)) of x [ch * 8][f][h][w]
 * into [ch][2 f - 1][2 h][2 w], dropping the first output frame:
 * out[c][2 d + p1 - 1][2 y + p2][2 x + p3] = x[8 c + 4 p1 + 2 p2 + p3]
 * [d][y][x].
 */
std::vector<float> shuffle_drop(std::span<const float> x, std::uint32_t ch,
                                std::uint32_t f, std::uint32_t h,
                                std::uint32_t w) {
    const std::uint32_t of = 2 * f - 1, oh = 2 * h, ow = 2 * w;
    std::vector<float> out(std::size_t{ch} * of * oh * ow);
    const std::size_t in_plane = std::size_t{f} * h * w;
    parallel(ch, [&](std::size_t c) {
        for (std::uint32_t k = 0; k < 8; ++k) {
            const std::uint32_t p1 = k >> 2, p2 = (k >> 1) & 1, p3 = k & 1;
            const float* src = &x[(c * 8 + k) * in_plane];
            for (std::uint32_t d = 0; d < f; ++d) {
                const std::uint32_t od = 2 * d + p1;
                if (od == 0) continue;  // the dropped first frame
                for (std::uint32_t y = 0; y < h; ++y) {
                    float* dst = &out[((c * of + od - 1) * oh + 2 * y + p2) *
                                          ow +
                                      p3];
                    const float* s = &src[(std::size_t{d} * h + y) * w];
                    for (std::uint32_t xx = 0; xx < w; ++xx) {
                        dst[2 * xx] = s[xx];
                    }
                }
            }
        }
    });
    return out;
}

/**
 * diffusers' TimestepEmbedding after Timesteps(256, flip_sin_to_cos,
 * shift 0): linear_1, SiLU, linear_2, on the scaled timestep t.
 */
std::vector<float> time_mlp(std::span<const float> w1,
                            std::span<const float> b1,
                            std::span<const float> w2,
                            std::span<const float> b2, std::uint32_t out,
                            float t) {
    const auto tf = PixArtDit::timestep_features(t, 256);
    std::vector<float> h(out), o(out);
    parallel(out, [&](std::size_t r) {
        float s = b1[r];
        for (std::uint32_t c = 0; c < 256; ++c) s += w1[r * 256 + c] * tf[c];
        h[r] = silu(s);
    });
    parallel(out, [&](std::size_t r) {
        float s = b2[r];
        for (std::uint32_t c = 0; c < out; ++c) s += w2[r * out + c] * h[c];
        o[r] = s;
    });
    return o;
}

}  // namespace

void LtxvVae::conv3d_k3(std::span<const float> in, std::uint32_t ci,
                        std::uint32_t f, std::uint32_t h, std::uint32_t w,
                        std::span<const float> weight,
                        std::span<const float> bias, std::uint32_t co,
                        std::span<float> out) {
    const std::size_t plane = std::size_t{h} * w, vol = plane * f;
    if (f == 0 || h == 0 || w == 0 || in.size() != ci * vol ||
        weight.size() != std::size_t{co} * ci * 27 || bias.size() != co ||
        out.size() != co * vol) {
        throw std::invalid_argument("ltxv_vae: conv3d sizes");
    }
    const std::size_t rows = std::size_t{f} * h;  // rows per channel
    parallel(std::size_t{co} * rows, [&](std::size_t item) {
        const auto oc = static_cast<std::uint32_t>(item / rows);
        const auto rest = static_cast<std::uint32_t>(item % rows);
        const std::uint32_t od = rest / h;
        const auto oy = static_cast<std::int64_t>(rest % h);
        float* o = &out[oc * vol + std::size_t{od} * plane + oy * w];
        for (std::uint32_t x = 0; x < w; ++x) o[x] = bias[oc];
        for (std::uint32_t c = 0; c < ci; ++c) {
            const float* wk = &weight[(std::size_t{oc} * ci + c) * 27];
            for (std::uint32_t kt = 0; kt < 3; ++kt) {
                // The edge frame repeats in time (causal=False padding).
                const std::int64_t d = std::clamp<std::int64_t>(
                    static_cast<std::int64_t>(od) + kt - 1, 0, f - 1);
                const float* ip = &in[c * vol + d * plane];
                for (std::uint32_t ky = 0; ky < 3; ++ky) {
                    const std::int64_t iy = oy + ky - 1;
                    if (iy < 0 || iy >= h) continue;
                    const float* irow = ip + iy * w;
                    for (std::uint32_t kx = 0; kx < 3; ++kx) {
                        const float wv = wk[(kt * 3 + ky) * 3 + kx];
                        const std::int64_t shift =
                            static_cast<std::int64_t>(kx) - 1;
                        const std::int64_t lo =
                            std::max<std::int64_t>(0, -shift);
                        const std::int64_t hi = std::min<std::int64_t>(
                            w, static_cast<std::int64_t>(w) - shift);
                        for (std::int64_t x = lo; x < hi; ++x) {
                            o[x] += wv * irow[x + shift];
                        }
                    }
                }
            }
        }
    });
}

LtxvVae LtxvVae::load(const gguf::GgufFile& g) {
    if (g.get_string("general.architecture") != "ltxv_vae") {
        throw gguf::Error("ltxv_vae: general.architecture is not ltxv_vae");
    }
    LtxvVae v;
    auto u = [&](std::string_view k) -> std::uint32_t {
        auto x = g.get_uint(k);
        if (!x || *x == 0 || *x > (1u << 16)) {
            throw gguf::Error("ltxv_vae: missing or bad " + std::string(k));
        }
        return static_cast<std::uint32_t>(*x);
    };
    v.latent_ch_ = u("ltxv_vae.latent_channels");
    v.out_ch_ = u("ltxv_vae.out_channels");
    v.patch_ = u("ltxv_vae.patch_size");
    if (u("ltxv_vae.decoder.kernel_size") != 3 || v.patch_ > 16) {
        throw gguf::Error("ltxv_vae: only 3 x 3 x 3 kernels, patch <= 16");
    }
    const gguf::Value* kinds = g.find("ltxv_vae.decoder.block_kinds");
    const gguf::Value* params = g.find("ltxv_vae.decoder.block_params");
    if (kinds == nullptr || params == nullptr ||
        kinds->type != gguf::ValueType::kArray ||
        params->type != gguf::ValueType::kArray || kinds->arr.empty() ||
        kinds->arr.size() != params->arr.size() || kinds->arr.size() > 64) {
        throw gguf::Error("ltxv_vae: bad decoder block lists");
    }

    const auto info = [&](const std::string& name) {
        const gguf::TensorInfo* ti = g.find_tensor(name);
        if (ti == nullptr) {
            throw gguf::Error("ltxv_vae: missing tensor " + name);
        }
        if (ti->type != gguf::TensorType::kF32) {
            throw gguf::Error("ltxv_vae: " + name + " is not F32");
        }
        return ti;
    };
    const auto floats = [&](const gguf::TensorInfo* ti) {
        const auto bytes = g.tensor_data(*ti);
        return std::span<const float>(
            reinterpret_cast<const float*>(bytes.data()), ti->nelements());
    };
    const auto vec = [&](const std::string& name, std::size_t n) {
        const gguf::TensorInfo* ti = info(name);
        if (ti->nelements() != n) {
            throw gguf::Error("ltxv_vae: " + name + " must have " +
                              std::to_string(n) + " values");
        }
        return floats(ti);
    };
    // A conv weight [co][ci][27] (GGUF ne = 27, ci, co) and its bias.
    const auto conv = [&](const std::string& name, std::uint32_t ci) {
        const gguf::TensorInfo* ti = info(name + ".weight");
        if (ti->n_dims != 3 || ti->ne[0] != 27 || ti->ne[1] != ci ||
            ti->ne[2] == 0 || ti->ne[2] > (1u << 16)) {
            throw gguf::Error("ltxv_vae: " + name + ".weight is not [co][" +
                              std::to_string(ci) + "][27]");
        }
        Conv c;
        c.ci = ci;
        c.co = static_cast<std::uint32_t>(ti->ne[2]);
        c.w = floats(ti);
        c.b = vec(name + ".bias", c.co);
        return c;
    };
    const auto mlp = [&](const std::string& name, std::uint32_t out) {
        TimeMlp m;
        m.out = out;
        m.w1 = vec(name + ".timestep_embedder.linear_1.weight",
                   std::size_t{out} * 256);
        m.b1 = vec(name + ".timestep_embedder.linear_1.bias", out);
        m.w2 = vec(name + ".timestep_embedder.linear_2.weight",
                   std::size_t{out} * out);
        m.b2 = vec(name + ".timestep_embedder.linear_2.bias", out);
        return m;
    };

    const std::string d = "decoder.";
    v.t_scale_ = vec(d + "timestep_scale_multiplier", 1)[0];
    v.mean_ = vec("per_channel_statistics.mean-of-means", v.latent_ch_);
    v.std_ = vec("per_channel_statistics.std-of-means", v.latent_ch_);
    v.conv_in_ = conv(d + "conv_in.conv", v.latent_ch_);
    std::uint32_t ch = v.conv_in_.co;
    // up_blocks run in the reverse of the config's list.
    const std::size_t nb = kinds->arr.size();
    for (std::size_t i = 0; i < nb; ++i) {
        const gguf::Value& kv = kinds->arr[nb - 1 - i];
        const gguf::Value& pv = params->arr[nb - 1 - i];
        const std::int64_t p = pv.type == gguf::ValueType::kInt32 ||
                                       pv.type == gguf::ValueType::kInt64
                                   ? pv.i
                                   : static_cast<std::int64_t>(pv.u);
        if (kv.type != gguf::ValueType::kString || p <= 0 || p > 64) {
            throw gguf::Error("ltxv_vae: bad decoder block entry");
        }
        const std::string b = d + "up_blocks." + std::to_string(i) + ".";
        Block blk;
        blk.ch = ch;
        if (kv.s == "res_x") {
            blk.res = true;
            blk.time = mlp(b + "time_embedder", 4 * ch);
            for (std::int64_t j = 0; j < p; ++j) {
                const std::string r =
                    b + "res_blocks." + std::to_string(j) + ".";
                Res res;
                res.conv1 = conv(r + "conv1.conv", ch);
                res.conv2 = conv(r + "conv2.conv", ch);
                if (res.conv1.co != ch || res.conv2.co != ch) {
                    throw gguf::Error("ltxv_vae: " + r + " changes channels");
                }
                res.table = vec(r + "scale_shift_table", std::size_t{4} * ch);
                blk.layers.push_back(res);
            }
        } else if (kv.s == "compress_all") {
            blk.reduce = static_cast<std::uint32_t>(p);
            blk.conv = conv(b + "conv.conv", ch);
            if (ch % 8 != 0 || 8 % blk.reduce != 0 ||
                blk.conv.co != 8 * ch / blk.reduce) {
                throw gguf::Error("ltxv_vae: bad compress_all at " + b);
            }
            ch /= blk.reduce;
            v.time_scale_ *= 2;
            v.space_scale_ *= 2;
        } else {
            throw gguf::Error("ltxv_vae: unsupported block " + kv.s);
        }
        v.blocks_.push_back(std::move(blk));
    }
    v.last_time_ = mlp(d + "last_time_embedder", 2 * ch);
    v.last_table_ = vec(d + "last_scale_shift_table", std::size_t{2} * ch);
    v.conv_out_ = conv(d + "conv_out.conv", ch);
    if (v.conv_out_.co != v.out_ch_ * v.patch_ * v.patch_) {
        throw gguf::Error("ltxv_vae: conv_out is not out x patch^2");
    }
    v.space_scale_ *= v.patch_;
    return v;
}

std::vector<float> LtxvVae::decode(std::span<const float> latent,
                                   std::uint32_t f, std::uint32_t h,
                                   std::uint32_t w, float timestep) const {
    if (f == 0 || h == 0 || w == 0 || f > 1024 || h > 4096 || w > 4096 ||
        latent.size() != std::size_t{latent_ch_} * f * h * w) {
        throw std::invalid_argument("ltxv_vae: latent is not C x f x h x w");
    }
    std::size_t plane = std::size_t{f} * h * w;

    // Un-normalize, then conv_in.
    std::vector<float> z(latent.begin(), latent.end());
    for (std::uint32_t c = 0; c < latent_ch_; ++c) {
        for (std::size_t p = 0; p < plane; ++p) {
            z[c * plane + p] = z[c * plane + p] * std_[c] + mean_[c];
        }
    }
    std::uint32_t ch = conv_in_.co;
    std::vector<float> x(ch * plane);
    conv3d_k3(z, latent_ch_, f, h, w, conv_in_.w, conv_in_.b, ch, x);

    const float st = timestep * t_scale_;
    std::vector<float> a, b;
    for (const Block& blk : blocks_) {
        if (blk.res) {
            const auto emb = time_mlp(blk.time.w1, blk.time.b1, blk.time.w2,
                                      blk.time.b2, blk.time.out, st);
            std::vector<float> ada(std::size_t{4} * ch);
            for (const Res& r : blk.layers) {
                // Rows shift1, scale1, shift2, scale2: table + emb.
                for (std::size_t i = 0; i < ada.size(); ++i) {
                    ada[i] = r.table[i] + emb[i];
                }
                a = x;
                pixel_norm(a, ch, plane);
                modulate_silu(a, ch, plane, &ada[0], &ada[ch]);
                b.resize(a.size());
                conv3d_k3(a, ch, f, h, w, r.conv1.w, r.conv1.b, ch, b);
                pixel_norm(b, ch, plane);
                modulate_silu(b, ch, plane, &ada[2 * ch], &ada[3 * ch]);
                conv3d_k3(b, ch, f, h, w, r.conv2.w, r.conv2.b, ch, a);
                for (std::size_t i = 0; i < x.size(); ++i) x[i] += a[i];
            }
        } else {
            const std::uint32_t oc = blk.conv.co;
            a.resize(std::size_t{oc} * plane);
            conv3d_k3(x, ch, f, h, w, blk.conv.w, blk.conv.b, oc, a);
            auto y = shuffle_drop(a, oc / 8, f, h, w);
            const auto res = shuffle_drop(x, ch / 8, f, h, w);
            const std::uint32_t nch = ch / blk.reduce;
            const std::uint32_t rch = ch / 8;
            const std::size_t np =
                std::size_t{2 * f - 1} * (2 * h) * (2 * w);
            // The residual tiles the shuffled input along the channels.
            parallel(nch, [&](std::size_t c) {
                float* yr = &y[c * np];
                const float* rr = &res[(c % rch) * np];
                for (std::size_t p = 0; p < np; ++p) yr[p] += rr[p];
            });
            x = std::move(y);
            ch = nch;
            f = 2 * f - 1;
            h *= 2;
            w *= 2;
            plane = np;
        }
    }

    // Last: pixel norm, shift and scale, SiLU, conv_out, unpatchify.
    pixel_norm(x, ch, plane);
    const auto emb = time_mlp(last_time_.w1, last_time_.b1, last_time_.w2,
                              last_time_.b2, last_time_.out, st);
    std::vector<float> shift(ch), scale(ch);
    for (std::uint32_t c = 0; c < ch; ++c) {
        shift[c] = last_table_[c] + emb[c];
        scale[c] = last_table_[ch + c] + emb[ch + c];
    }
    modulate_silu(x, ch, plane, shift.data(), scale.data());
    const std::uint32_t pc = conv_out_.co;
    a.resize(std::size_t{pc} * plane);
    conv3d_k3(x, ch, f, h, w, conv_out_.w, conv_out_.b, pc, a);

    // in[c * p^2 + r p + q][f][y][x] -> out[c][f][y p + q][x p + r].
    const std::uint32_t p = patch_, oh = h * p, ow = w * p;
    std::vector<float> out(std::size_t{out_ch_} * f * oh * ow);
    for (std::uint32_t c = 0; c < out_ch_; ++c) {
        for (std::uint32_t r = 0; r < p; ++r) {
            for (std::uint32_t q = 0; q < p; ++q) {
                const float* src = &a[(c * p * p + r * p + q) * plane];
                for (std::uint32_t d = 0; d < f; ++d) {
                    for (std::uint32_t y = 0; y < h; ++y) {
                        const float* s = &src[(std::size_t{d} * h + y) * w];
                        float* o = &out[((std::size_t{c} * f + d) * oh +
                                         y * p + q) *
                                            ow +
                                        r];
                        for (std::uint32_t xx = 0; xx < w; ++xx) {
                            o[xx * p] = s[xx];
                        }
                    }
                }
            }
        }
    }
    return out;
}

}  // namespace locus::image
