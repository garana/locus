#include "locus/image/jpeg_decode.hpp"

#include <algorithm>
#include <array>
#include <climits>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

namespace locus::image {

namespace {

[[noreturn]] void bad(const std::string& what) {
    throw std::invalid_argument("jpeg: " + what);
}

// Zigzag position -> row-major index in the 8x8 block.
constexpr std::uint8_t kDezigzag[64] = {
    0,  1,  8,  16, 9,  2,  3,  10, 17, 24, 32, 25, 18, 11, 4,  5,
    12, 19, 26, 33, 40, 48, 41, 34, 27, 20, 13, 6,  7,  14, 21, 28,
    35, 42, 49, 56, 57, 50, 43, 36, 29, 22, 15, 23, 30, 37, 44, 51,
    58, 59, 52, 45, 38, 31, 39, 46, 53, 60, 61, 54, 47, 55, 62, 63};

constexpr int kMaxDim = 16384;  // as the stb_image build this replaces

/** A Huffman table (JPEG F.2.2.3: per-length code ranges). */
struct Huffman {
    bool defined = false;
    std::array<std::uint8_t, 256> values{};
    std::array<int, 17> mincode{}, valptr{};
    std::array<int, 18> maxcode{};
};

void build(Huffman& h, const std::uint8_t* counts) {
    int code = 0, k = 0;
    for (int l = 1; l <= 16; ++l) {
        h.valptr[l] = k;
        h.mincode[l] = code;
        code += counts[l - 1];
        k += counts[l - 1];
        h.maxcode[l] = counts[l - 1] ? code - 1 : -1;
        if (code > (1 << l)) bad("bad Huffman code lengths");
        code <<= 1;
    }
    h.maxcode[17] = INT_MAX;
    h.defined = true;
}

/**
 * Entropy-coded bits, MSB first, with 0xFF00 unstuffed. A marker or the
 * end of the data stops the input; from then on it reads zero bits, as
 * stb_image does, and remembers the marker.
 */
class Bits {
  public:
    Bits(const std::uint8_t* p, std::size_t n, std::size_t& pos)
        : p_(p), n_(n), pos_(pos) {}

    int marker() const { return marker_; }
    void reset() {
        buf_ = 0;
        cnt_ = 0;
        nomore_ = false;
        marker_ = -1;
    }
    std::uint32_t bit() {
        if (cnt_ < 1) fill();
        const std::uint32_t v = buf_ >> 31;
        buf_ <<= 1;
        --cnt_;
        return v;
    }
    int bits(int n) {
        if (n == 0) return 0;
        if (cnt_ < n) fill();
        const std::uint32_t v = buf_ >> (32 - n);
        buf_ <<= n;
        cnt_ -= n;
        return static_cast<int>(v);
    }
    /** receive + extend (F.2.2.1): n bits as a signed value. */
    int extend(int n) {
        if (n == 0) return 0;
        const int v = bits(n);
        return v < (1 << (n - 1)) ? v - (1 << n) + 1 : v;
    }
    int decode(const Huffman& h) {
        if (!h.defined) bad("scan uses an undefined Huffman table");
        int code = 0;
        for (int l = 1; l <= 16; ++l) {
            code = (code << 1) | static_cast<int>(bit());
            if (code <= h.maxcode[l]) {
                const int i = h.valptr[l] + code - h.mincode[l];
                if (i < 0 || i > 255) bad("bad Huffman code");
                return h.values[i];
            }
        }
        bad("bad Huffman code");
    }
    /** Loads whole bytes until more than 24 bits are buffered. */
    void fill() {
        while (cnt_ <= 24) {
            std::uint32_t b = 0;
            if (!nomore_) {
                if (pos_ >= n_) {
                    nomore_ = true;
                } else {
                    b = p_[pos_++];
                    if (b == 0xFF) {
                        int c = pos_ < n_ ? p_[pos_++] : 0;
                        while (c == 0xFF && pos_ < n_) c = p_[pos_++];
                        if (c != 0) {
                            marker_ = c;
                            nomore_ = true;
                            b = 0;
                        }
                    }
                }
            }
            buf_ |= b << (24 - cnt_);
            cnt_ += 8;
        }
    }

  private:
    const std::uint8_t* p_;
    std::size_t n_;
    std::size_t& pos_;
    std::uint32_t buf_ = 0;
    int cnt_ = 0;
    bool nomore_ = false;
    int marker_ = -1;
};

struct Component {
    int id = 0, h = 1, v = 1, tq = 0;
    int hd = 0, ha = 0;  ///< Huffman tables of the current scan
    int x = 0, y = 0;    ///< samples covering the image
    int w2 = 0, h2 = 0;  ///< samples covering whole MCUs
    int dc_pred = 0;
    std::vector<std::uint8_t> data;   ///< w2 x h2 samples
    std::vector<std::int16_t> coeff;  ///< progressive: 64 per block
    int coeff_w = 0;                  ///< blocks per coefficient row
};

std::uint8_t clamp8(std::int64_t x) {
    return static_cast<std::uint8_t>(x < 0 ? 0 : x > 255 ? 255 : x);
}

// stb_image's integer IDCT (derived from libjpeg's jidctint, ISLOW):
// constants scaled by 4096, two extra bits kept between the passes.
// The sums are 64-bit: a valid file never leaves int range, so the
// result is stb_image's, and corrupt coefficients cannot overflow
// (undefined behavior in int, which stb_image's int math has).
using Acc = std::int64_t;
constexpr Acc f2f(double x) { return static_cast<int>(x * 4096 + 0.5); }

struct Idct1d {
    Acc t0, t1, t2, t3, x0, x1, x2, x3;
    Idct1d(Acc s0, Acc s1, Acc s2, Acc s3, Acc s4, Acc s5, Acc s6, Acc s7) {
        Acc p2 = s2, p3 = s6;
        Acc p1 = (p2 + p3) * f2f(0.5411961);
        t2 = p1 + p3 * f2f(-1.847759065);
        t3 = p1 + p2 * f2f(0.765366865);
        p2 = s0;
        p3 = s4;
        t0 = (p2 + p3) * 4096;
        t1 = (p2 - p3) * 4096;
        x0 = t0 + t3;
        x3 = t0 - t3;
        x1 = t1 + t2;
        x2 = t1 - t2;
        t0 = s7;
        t1 = s5;
        t2 = s3;
        t3 = s1;
        p3 = t0 + t2;
        Acc p4 = t1 + t3;
        p1 = t0 + t3;
        p2 = t1 + t2;
        const Acc p5 = (p3 + p4) * f2f(1.175875602);
        t0 = t0 * f2f(0.298631336);
        t1 = t1 * f2f(2.053119869);
        t2 = t2 * f2f(3.072711026);
        t3 = t3 * f2f(1.501321110);
        p1 = p5 + p1 * f2f(-0.899976223);
        p2 = p5 + p2 * f2f(-2.562915447);
        p3 = p3 * f2f(-1.961570560);
        p4 = p4 * f2f(-0.390180644);
        t3 += p1 + p4;
        t2 += p2 + p3;
        t1 += p2 + p4;
        t0 += p1 + p3;
    }
};

void idct_block(std::uint8_t* out, int stride, const std::int16_t* d) {
    Acc val[64];
    for (int i = 0; i < 8; ++i) {
        const std::int16_t* c = d + i;
        Acc* v = val + i;
        if (c[8] == 0 && c[16] == 0 && c[24] == 0 && c[32] == 0 &&
            c[40] == 0 && c[48] == 0 && c[56] == 0) {
            const Acc dc = c[0] * 4;
            for (int k = 0; k < 8; ++k) v[k * 8] = dc;
            continue;
        }
        Idct1d t(c[0], c[8], c[16], c[24], c[32], c[40], c[48], c[56]);
        t.x0 += 512;
        t.x1 += 512;
        t.x2 += 512;
        t.x3 += 512;
        v[0] = (t.x0 + t.t3) >> 10;
        v[56] = (t.x0 - t.t3) >> 10;
        v[8] = (t.x1 + t.t2) >> 10;
        v[48] = (t.x1 - t.t2) >> 10;
        v[16] = (t.x2 + t.t1) >> 10;
        v[40] = (t.x2 - t.t1) >> 10;
        v[24] = (t.x3 + t.t0) >> 10;
        v[32] = (t.x3 - t.t0) >> 10;
    }
    for (int i = 0; i < 8; ++i) {
        const Acc* v = val + i * 8;
        std::uint8_t* o = out + i * stride;
        Idct1d t(v[0], v[1], v[2], v[3], v[4], v[5], v[6], v[7]);
        // 1 << 17 to remove (4096, the 2 kept bits, sqrt(8)^2), with
        // rounding and the +128 level shift folded in.
        constexpr Acc kBias = 65536 + (128 << 17);
        t.x0 += kBias;
        t.x1 += kBias;
        t.x2 += kBias;
        t.x3 += kBias;
        o[0] = clamp8((t.x0 + t.t3) >> 17);
        o[7] = clamp8((t.x0 - t.t3) >> 17);
        o[1] = clamp8((t.x1 + t.t2) >> 17);
        o[6] = clamp8((t.x1 - t.t2) >> 17);
        o[2] = clamp8((t.x2 + t.t1) >> 17);
        o[5] = clamp8((t.x2 - t.t1) >> 17);
        o[3] = clamp8((t.x3 + t.t0) >> 17);
        o[4] = clamp8((t.x3 - t.t0) >> 17);
    }
}

/** stb_image's check that a * b fits a short (both taken as shorts). */
bool mul_fits(std::int16_t a, std::int16_t b) {
    if (b == 0) return true;
    if ((a >= 0) == (b >= 0)) return a <= SHRT_MAX / b;
    return a >= SHRT_MIN / b;
}

/** The decoder state for one file. */
class Decoder {
  public:
    Decoder(const std::uint8_t* p, std::size_t n, const DecodeLimits& lim)
        : p_(p), n_(n), lim_(lim) {}

    Rgb8 run();

  private:
    int u8() {
        if (pos_ >= n_) bad("truncated");
        return p_[pos_++];
    }
    int u16() {
        const int hi = u8();
        return (hi << 8) | u8();
    }
    void skip(std::size_t k) {
        if (k > n_ - pos_) bad("truncated");
        pos_ += k;
    }
    /** @returns The next marker, or -1 when the next byte is not 0xFF. */
    int marker() {
        if (cached_ >= 0) {
            const int m = cached_;
            cached_ = -1;
            return m;
        }
        if (pos_ >= n_) return -1;
        int x = p_[pos_++];
        if (x != 0xFF) return -1;
        while (x == 0xFF) {
            if (pos_ >= n_) return -1;
            x = p_[pos_++];
        }
        return x;
    }
    void table_marker(int m);
    void frame(int m);
    void scan();
    void decode_block(Bits& b, std::int16_t* data, Component& c);
    void prog_dc(Bits& b, std::int16_t* data, Component& c);
    void prog_ac(Bits& b, std::int16_t* data, Component& c);
    Rgb8 convert();

    const std::uint8_t* p_;
    std::size_t n_;
    std::size_t pos_ = 0;
    DecodeLimits lim_;
    int cached_ = -1;  ///< a marker the entropy decoder stopped at

    std::array<std::array<std::uint16_t, 64>, 4> dequant_{};
    std::array<Huffman, 4> huff_dc_{}, huff_ac_{};
    int restart_ = 0;
    bool jfif_ = false;
    int app14_ = -1;  ///< Adobe color transform, -1 when absent

    bool have_frame_ = false, progressive_ = false;
    int w_ = 0, h_ = 0;
    std::vector<Component> comp_;
    int hmax_ = 1, vmax_ = 1, mcu_x_ = 0, mcu_y_ = 0;
    int rgb_ids_ = 0;

    // Current scan.
    std::array<int, 4> order_{};
    int scan_n_ = 0;
    int ss_ = 0, se_ = 63, ah_ = 0, al_ = 0;
    int eob_run_ = 0;
    int scans_ = 0;
};

void Decoder::table_marker(int m) {
    if (m == 0xDD) {  // DRI
        if (u16() != 4) bad("bad DRI length");
        restart_ = u16();
        return;
    }
    if (m == 0xDB) {  // DQT
        int len = u16() - 2;
        while (len > 0) {
            const int q = u8();
            const int prec = q >> 4, t = q & 15;
            if (prec > 1) bad("bad DQT precision");
            if (t > 3) bad("bad DQT table");
            for (int i = 0; i < 64; ++i) {
                dequant_[t][kDezigzag[i]] =
                    static_cast<std::uint16_t>(prec ? u16() : u8());
            }
            len -= prec ? 129 : 65;
        }
        if (len != 0) bad("bad DQT length");
        return;
    }
    if (m == 0xC4) {  // DHT
        int len = u16() - 2;
        while (len > 0) {
            const int q = u8();
            const int tc = q >> 4, th = q & 15;
            if (tc > 1 || th > 3) bad("bad DHT header");
            std::uint8_t counts[16];
            int total = 0;
            for (int i = 0; i < 16; ++i) {
                counts[i] = static_cast<std::uint8_t>(u8());
                total += counts[i];
            }
            if (total > 256) bad("bad DHT header");
            Huffman& h = tc == 0 ? huff_dc_[th] : huff_ac_[th];
            build(h, counts);
            for (int i = 0; i < total; ++i) {
                h.values[i] = static_cast<std::uint8_t>(u8());
            }
            len -= 17 + total;
        }
        if (len != 0) bad("bad DHT length");
        return;
    }
    if ((m >= 0xE0 && m <= 0xEF) || m == 0xFE) {  // APPn, COM
        int len = u16();
        if (len < 2) bad("bad APP or COM length");
        len -= 2;
        if (m == 0xE0 && len >= 5) {
            static constexpr std::uint8_t kJfif[5] = {'J', 'F', 'I', 'F', 0};
            bool ok = true;
            for (int i = 0; i < 5; ++i) ok = (u8() == kJfif[i]) && ok;
            len -= 5;
            if (ok) jfif_ = true;
        } else if (m == 0xEE && len >= 12) {
            static constexpr std::uint8_t kAdobe[6] = {'A', 'd', 'o',
                                                       'b', 'e', 0};
            bool ok = true;
            for (int i = 0; i < 6; ++i) ok = (u8() == kAdobe[i]) && ok;
            len -= 6;
            if (ok) {
                skip(5);  // version, flags0, flags1
                app14_ = u8();
                len -= 6;
            }
        }
        skip(static_cast<std::size_t>(len));
        return;
    }
    bad("unknown marker " + std::to_string(m));
}

void Decoder::frame(int m) {
    if (have_frame_) bad("second frame header");
    if (m != 0xC0 && m != 0xC1 && m != 0xC2) {
        bad("unsupported JPEG type (SOF" + std::to_string(m - 0xC0) +
            "): only baseline, extended and progressive Huffman");
    }
    progressive_ = m == 0xC2;
    const int len = u16();
    if (len < 11) bad("bad SOF length");
    if (u8() != 8) bad("only 8-bit precision");
    h_ = u16();
    w_ = u16();
    if (h_ == 0) bad("zero height (DNL-defined height is unsupported)");
    if (w_ == 0) bad("zero width");
    if (w_ > kMaxDim || h_ > kMaxDim ||
        std::uint64_t(w_) * std::uint64_t(h_) > lim_.max_pixels) {
        bad("dimensions over the limit");
    }
    const int nc = u8();
    if (nc != 1 && nc != 3 && nc != 4) bad("bad component count");
    if (len != 8 + 3 * nc) bad("bad SOF length");
    comp_.assign(nc, Component{});
    static constexpr std::uint8_t kRgb[3] = {'R', 'G', 'B'};
    for (int i = 0; i < nc; ++i) {
        Component& c = comp_[i];
        c.id = u8();
        if (nc == 3 && c.id == kRgb[i]) ++rgb_ids_;
        const int q = u8();
        c.h = q >> 4;
        c.v = q & 15;
        if (c.h < 1 || c.h > 4) bad("bad H sampling factor");
        if (c.v < 1 || c.v > 4) bad("bad V sampling factor");
        c.tq = u8();
        if (c.tq > 3) bad("bad quantization table id");
        hmax_ = std::max(hmax_, c.h);
        vmax_ = std::max(vmax_, c.v);
    }
    for (const Component& c : comp_) {
        if (hmax_ % c.h != 0 || vmax_ % c.v != 0) {
            bad("non-integer sampling ratio");
        }
    }
    mcu_x_ = (w_ + hmax_ * 8 - 1) / (hmax_ * 8);
    mcu_y_ = (h_ + vmax_ * 8 - 1) / (vmax_ * 8);
    for (Component& c : comp_) {
        c.x = (w_ * c.h + hmax_ - 1) / hmax_;
        c.y = (h_ * c.v + vmax_ - 1) / vmax_;
        c.w2 = mcu_x_ * c.h * 8;
        c.h2 = mcu_y_ * c.v * 8;
        c.data.assign(std::size_t(c.w2) * c.h2, 0);
        if (progressive_) {
            c.coeff_w = c.w2 / 8;
            c.coeff.assign(std::size_t(c.w2) * c.h2, 0);
        }
    }
    have_frame_ = true;
}

void Decoder::decode_block(Bits& b, std::int16_t* data, Component& c) {
    const int t = b.decode(huff_dc_[c.hd]);
    if (t > 15) bad("bad DC code");
    std::fill_n(data, 64, std::int16_t{0});
    const int diff = b.extend(t);
    const long long dc = static_cast<long long>(c.dc_pred) + diff;
    if (dc > INT_MAX || dc < INT_MIN) bad("bad DC delta");
    c.dc_pred = static_cast<int>(dc);
    const auto& dq = dequant_[c.tq];
    if (!mul_fits(static_cast<std::int16_t>(c.dc_pred),
                  static_cast<std::int16_t>(dq[0]))) {
        bad("DC coefficient overflow");
    }
    data[0] = static_cast<std::int16_t>(c.dc_pred * dq[0]);
    const Huffman& ac = huff_ac_[c.ha];
    for (int k = 1; k < 64;) {
        const int rs = b.decode(ac);
        const int s = rs & 15, r = rs >> 4;
        if (s == 0) {
            if (rs != 0xF0) break;  // end of block
            k += 16;
            continue;
        }
        k += r;
        if (k > 63) bad("AC run past the block");
        const int z = kDezigzag[k++];
        data[z] = static_cast<std::int16_t>(b.extend(s) * dq[z]);
    }
}

void Decoder::prog_dc(Bits& b, std::int16_t* data, Component& c) {
    if (se_ != 0) bad("DC scan with AC coefficients");
    if (ah_ == 0) {
        std::fill_n(data, 64, std::int16_t{0});
        const int t = b.decode(huff_dc_[c.hd]);
        if (t > 15) bad("bad DC code");
        const long long dc = static_cast<long long>(c.dc_pred) + b.extend(t);
        if (dc > INT_MAX || dc < INT_MIN) bad("bad DC delta");
        c.dc_pred = static_cast<int>(dc);
        if (!mul_fits(static_cast<std::int16_t>(c.dc_pred),
                      static_cast<std::int16_t>(1 << al_))) {
            bad("DC coefficient overflow");
        }
        data[0] = static_cast<std::int16_t>(c.dc_pred * (1 << al_));
    } else if (b.bit()) {
        data[0] = static_cast<std::int16_t>(data[0] + (1 << al_));
    }
}

void Decoder::prog_ac(Bits& b, std::int16_t* data, Component& c) {
    if (ss_ == 0) bad("AC scan with the DC coefficient");
    const Huffman& ac = huff_ac_[c.ha];
    if (ah_ == 0) {  // first scan of these coefficients
        if (eob_run_ > 0) {
            --eob_run_;
            return;
        }
        for (int k = ss_; k <= se_;) {
            const int rs = b.decode(ac);
            const int s = rs & 15, r = rs >> 4;
            if (s == 0) {
                if (r < 15) {
                    eob_run_ = (1 << r) - 1;
                    if (r) eob_run_ += b.bits(r);
                    break;
                }
                k += 16;
                continue;
            }
            k += r;
            if (k > 63) bad("AC run past the block");
            data[kDezigzag[k++]] =
                static_cast<std::int16_t>(b.extend(s) * (1 << al_));
        }
        return;
    }
    // Refinement: one more bit for every nonzero coefficient in the band,
    // and new coefficients of magnitude 1 << al.
    const int bit = 1 << al_;
    auto refine = [&](std::int16_t& p) {
        if (b.bit() && (p & bit) == 0) {
            p = static_cast<std::int16_t>(p > 0 ? p + bit : p - bit);
        }
    };
    if (eob_run_ > 0) {
        --eob_run_;
        for (int k = ss_; k <= se_; ++k) {
            std::int16_t& p = data[kDezigzag[k]];
            if (p != 0) refine(p);
        }
        return;
    }
    for (int k = ss_; k <= se_;) {
        const int rs = b.decode(ac);
        int s = rs & 15, r = rs >> 4;
        if (s == 0) {
            if (r < 15) {
                eob_run_ = (1 << r) - 1;
                if (r) eob_run_ += b.bits(r);
                r = 64;  // to the end of the band
            }  // r == 15: skip 16 zero coefficients
        } else {
            if (s != 1) bad("bad refinement code");
            s = b.bit() ? bit : -bit;
        }
        while (k <= se_) {
            std::int16_t& p = data[kDezigzag[k++]];
            if (p != 0) {
                refine(p);
            } else {
                if (r == 0) {
                    p = static_cast<std::int16_t>(s);
                    break;
                }
                --r;
            }
        }
    }
}

void Decoder::scan() {
    if (!have_frame_) bad("scan before the frame header");
    const int len = u16();
    scan_n_ = u8();
    if (scan_n_ < 1 || scan_n_ > 4 ||
        scan_n_ > static_cast<int>(comp_.size())) {
        bad("bad scan component count");
    }
    if (len != 6 + 2 * scan_n_) bad("bad SOS length");
    for (int i = 0; i < scan_n_; ++i) {
        const int id = u8(), q = u8();
        int which = 0;
        while (which < static_cast<int>(comp_.size()) && comp_[which].id != id) {
            ++which;
        }
        if (which == static_cast<int>(comp_.size())) bad("scan names no component");
        comp_[which].hd = q >> 4;
        comp_[which].ha = q & 15;
        if (comp_[which].hd > 3 || comp_[which].ha > 3) bad("bad table id");
        order_[i] = which;
    }
    ss_ = u8();
    se_ = u8();
    const int a = u8();
    ah_ = a >> 4;
    al_ = a & 15;
    if (progressive_) {
        if (ss_ > 63 || se_ > 63 || ss_ > se_ || ah_ > 13 || al_ > 13) {
            bad("bad progressive scan parameters");
        }
    } else {
        if (ss_ != 0 || ah_ != 0 || al_ != 0) bad("bad SOS parameters");
        se_ = 63;
    }
    ++scans_;

    Bits b(p_, n_, pos_);
    auto restart = [&] {
        b.reset();
        for (Component& c : comp_) c.dc_pred = 0;
        eob_run_ = 0;
    };
    restart();
    int todo = restart_ ? restart_ : INT_MAX;
    // Counts down one MCU; @returns false when the scan must stop (the
    // interval ended without a restart marker: corrupt data, kept).
    auto next_mcu = [&] {
        if (--todo > 0) return true;
        b.fill();
        if (b.marker() < 0xD0 || b.marker() > 0xD7) return false;
        restart();
        todo = restart_ ? restart_ : INT_MAX;
        return true;
    };
    std::int16_t block[64];
    if (scan_n_ == 1) {  // one component, its blocks in raster order
        Component& c = comp_[order_[0]];
        const int bw = (c.x + 7) >> 3, bh = (c.y + 7) >> 3;
        for (int j = 0; j < bh; ++j) {
            for (int i = 0; i < bw; ++i) {
                if (!progressive_) {
                    decode_block(b, block, c);
                    idct_block(c.data.data() + std::size_t(c.w2) * j * 8 + i * 8,
                               c.w2, block);
                } else {
                    std::int16_t* d =
                        c.coeff.data() + 64 * (std::size_t(i) + std::size_t(j) * c.coeff_w);
                    if (ss_ == 0) {
                        prog_dc(b, d, c);
                    } else {
                        prog_ac(b, d, c);
                    }
                }
                if (!next_mcu()) goto done;
            }
        }
    } else {  // interleaved MCUs
        for (int j = 0; j < mcu_y_; ++j) {
            for (int i = 0; i < mcu_x_; ++i) {
                for (int k = 0; k < scan_n_; ++k) {
                    Component& c = comp_[order_[k]];
                    for (int y = 0; y < c.v; ++y) {
                        for (int x = 0; x < c.h; ++x) {
                            const int bx = i * c.h + x, by = j * c.v + y;
                            if (!progressive_) {
                                decode_block(b, block, c);
                                idct_block(c.data.data() +
                                               std::size_t(c.w2) * by * 8 + bx * 8,
                                           c.w2, block);
                            } else {
                                prog_dc(b,
                                        c.coeff.data() +
                                            64 * (std::size_t(bx) +
                                                  std::size_t(by) * c.coeff_w),
                                        c);
                            }
                        }
                    }
                }
                if (!next_mcu()) goto done;
            }
        }
    }
done:
    cached_ = b.marker();
    if (cached_ < 0) {
        // Skip junk to the next real marker (0xFF then not 0x00 / 0xFF).
        while (pos_ < n_) {
            int x = p_[pos_++];
            while (x == 0xFF && pos_ < n_) {
                x = p_[pos_++];
                if (x != 0x00 && x != 0xFF) {
                    cached_ = x;
                    return;
                }
            }
        }
    }
}

Rgb8 Decoder::convert() {
    if (progressive_) {
        for (Component& c : comp_) {
            const int bw = (c.x + 7) >> 3, bh = (c.y + 7) >> 3;
            const auto& dq = dequant_[c.tq];
            for (int j = 0; j < bh; ++j) {
                for (int i = 0; i < bw; ++i) {
                    std::int16_t* d =
                        c.coeff.data() + 64 * (std::size_t(i) + std::size_t(j) * c.coeff_w);
                    for (int k = 0; k < 64; ++k) {
                        d[k] = static_cast<std::int16_t>(d[k] * dq[k]);
                    }
                    idct_block(c.data.data() + std::size_t(c.w2) * j * 8 + i * 8,
                               c.w2, d);
                }
            }
        }
    }
    const int nc = static_cast<int>(comp_.size());
    const bool is_rgb = nc == 3 && (rgb_ids_ == 3 || (app14_ == 0 && !jfif_));

    // stb_image's resampler state per component.
    struct Resample {
        int hs, vs, ystep, w_lores, ypos;
        const std::uint8_t* line0;
        const std::uint8_t* line1;
        std::vector<std::uint8_t> buf;
    };
    std::vector<Resample> rs(nc);
    for (int k = 0; k < nc; ++k) {
        Resample& r = rs[k];
        r.hs = hmax_ / comp_[k].h;
        r.vs = vmax_ / comp_[k].v;
        r.ystep = r.vs >> 1;
        r.w_lores = (w_ + r.hs - 1) / r.hs;
        r.ypos = 0;
        r.line0 = r.line1 = comp_[k].data.data();
        r.buf.assign(std::size_t(w_) + 3, 0);
    }
    auto div4 = [](int x) { return static_cast<std::uint8_t>(x >> 2); };
    auto div16 = [](int x) { return static_cast<std::uint8_t>(x >> 4); };
    // One output row of a component, upsampled to the image width.
    auto resample = [&](Resample& r, const std::uint8_t* near,
                        const std::uint8_t* far) -> const std::uint8_t* {
        std::uint8_t* out = r.buf.data();
        const int w = r.w_lores;
        if (r.hs == 1 && r.vs == 1) return near;
        if (r.hs == 1 && r.vs == 2) {
            for (int i = 0; i < w; ++i) out[i] = div4(3 * near[i] + far[i] + 2);
            return out;
        }
        if (r.hs == 2 && r.vs == 1) {
            if (w == 1) {
                out[0] = out[1] = near[0];
                return out;
            }
            out[0] = near[0];
            out[1] = div4(near[0] * 3 + near[1] + 2);
            int i = 1;
            for (; i < w - 1; ++i) {
                const int n = 3 * near[i] + 2;
                out[i * 2] = div4(n + near[i - 1]);
                out[i * 2 + 1] = div4(n + near[i + 1]);
            }
            out[i * 2] = div4(near[w - 2] * 3 + near[w - 1] + 2);
            out[i * 2 + 1] = near[w - 1];
            return out;
        }
        if (r.hs == 2 && r.vs == 2) {
            if (w == 1) {
                out[0] = out[1] = div4(3 * near[0] + far[0] + 2);
                return out;
            }
            int t1 = 3 * near[0] + far[0];
            out[0] = div4(t1 + 2);
            for (int i = 1; i < w; ++i) {
                const int t0 = t1;
                t1 = 3 * near[i] + far[i];
                out[i * 2 - 1] = div16(3 * t0 + t1 + 8);
                out[i * 2] = div16(3 * t1 + t0 + 8);
            }
            out[w * 2 - 1] = div4(t1 + 2);
            return out;
        }
        for (int i = 0; i < w; ++i) {  // nearest
            for (int j = 0; j < r.hs; ++j) out[i * r.hs + j] = near[i];
        }
        return out;
    };
    // stb_image's reduced-precision fixed point YCbCr -> RGB.
    auto fx = [](double x) { return static_cast<int>(x * 4096.0 + 0.5) << 8; };
    const int kCr = fx(1.40200), kCrG = -fx(0.71414), kCbG = -fx(0.34414),
              kCb = fx(1.77200);
    auto ycc = [&](std::uint8_t* o, int y, int cb, int cr) {
        const int yf = (y << 20) + (1 << 19);
        cr -= 128;
        cb -= 128;
        o[0] = clamp8((yf + cr * kCr) >> 20);
        o[1] = clamp8((yf + cr * kCrG + static_cast<int>(
                                           static_cast<unsigned>(cb * kCbG) &
                                           0xffff0000u)) >> 20);
        o[2] = clamp8((yf + cb * kCb) >> 20);
    };
    auto blinn = [](int x, int y) {
        const unsigned t = static_cast<unsigned>(x * y + 128);
        return static_cast<std::uint8_t>((t + (t >> 8)) >> 8);
    };

    Rgb8 out{static_cast<std::uint32_t>(w_), static_cast<std::uint32_t>(h_),
             std::vector<std::uint8_t>(std::size_t(w_) * h_ * 3)};
    std::array<const std::uint8_t*, 4> row{};
    for (int j = 0; j < h_; ++j) {
        for (int k = 0; k < nc; ++k) {
            Resample& r = rs[k];
            const bool bottom = r.ystep >= (r.vs >> 1);
            row[k] = resample(r, bottom ? r.line1 : r.line0,
                              bottom ? r.line0 : r.line1);
            if (++r.ystep >= r.vs) {
                r.ystep = 0;
                r.line0 = r.line1;
                if (++r.ypos < comp_[k].y) r.line1 += comp_[k].w2;
            }
        }
        std::uint8_t* o = out.rgb.data() + std::size_t(j) * w_ * 3;
        for (int i = 0; i < w_; ++i, o += 3) {
            if (nc == 1) {
                o[0] = o[1] = o[2] = row[0][i];
            } else if (nc == 3) {
                if (is_rgb) {
                    o[0] = row[0][i];
                    o[1] = row[1][i];
                    o[2] = row[2][i];
                } else {
                    ycc(o, row[0][i], row[1][i], row[2][i]);
                }
            } else if (app14_ == 0) {  // CMYK
                const int m = row[3][i];
                o[0] = blinn(row[0][i], m);
                o[1] = blinn(row[1][i], m);
                o[2] = blinn(row[2][i], m);
            } else if (app14_ == 2) {  // YCCK
                ycc(o, row[0][i], row[1][i], row[2][i]);
                const int m = row[3][i];
                o[0] = blinn(255 - o[0], m);
                o[1] = blinn(255 - o[1], m);
                o[2] = blinn(255 - o[2], m);
            } else {  // YCbCr with a fourth channel, ignored
                ycc(o, row[0][i], row[1][i], row[2][i]);
            }
        }
    }
    return out;
}

Rgb8 Decoder::run() {
    if (marker() != 0xD8) bad("no SOI marker");
    // Tables and application data until the frame header; junk bytes
    // between markers are skipped, as stb_image does.
    for (;;) {
        int m = marker();
        while (m < 0) {
            if (pos_ >= n_) bad("no frame header");
            m = marker();
        }
        if (m >= 0xC0 && m <= 0xCF && m != 0xC4 && m != 0xC8 && m != 0xCC) {
            frame(m);
            break;
        }
        table_marker(m);
    }
    // Scans and tables until EOI. Like stb_image, the end of the data or
    // an unexpected marker after a scan ends the image there.
    for (int m = marker(); m != 0xD9 && m >= 0;) {
        if (m == 0xDA) {
            scan();
            m = marker();
            if (m >= 0xD0 && m <= 0xD7) m = marker();  // a trailing RST
        } else if (m == 0xDC) {  // DNL
            if (u16() != 4) bad("bad DNL length");
            if (u16() != h_) bad("bad DNL height");
            m = marker();
        } else if (m == 0xDB || m == 0xC4 || m == 0xDD ||
                   (m >= 0xE0 && m <= 0xEF) || m == 0xFE) {
            table_marker(m);
            m = marker();
        } else {
            break;
        }
    }
    if (scans_ == 0) bad("no scan");
    return convert();
}

}  // namespace

bool is_jpeg(std::span<const std::byte> data) {
    return data.size() >= 2 && std::to_integer<int>(data[0]) == 0xFF &&
           std::to_integer<int>(data[1]) == 0xD8;
}

Rgb8 decode_jpeg(std::span<const std::byte> data, const DecodeLimits& lim) {
    if (data.size() > lim.max_bytes) bad("over the size limit");
    Decoder d(reinterpret_cast<const std::uint8_t*>(data.data()), data.size(),
              lim);
    return d.run();
}

}  // namespace locus::image
