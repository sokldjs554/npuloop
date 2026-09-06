// npuloop integer kernels: bit-exact with the NumPy engine and with gemmlowp/TFLite reference semantics.
// Tensors are NCHW; codes are carried as int32 (values fit uint8/int8); accumulators are int32 with
// explicit saturation; requantization uses the fixed-point multiplier + shift scheme.
#include <cstdint>
#include <cstdlib>
#include <algorithm>

extern "C" {

enum Rounding { R_TFLITE = 0, R_HALF_EVEN = 1, R_TRUNCATE = 2, R_FLOOR = 3, R_SINGLE = 4 };

static inline int32_t srdhm(int32_t a, int32_t b) {
    // SaturatingRoundingDoublingHighMul (gemmlowp)
    bool overflow = (a == b) && (a == INT32_MIN);
    int64_t ab = (int64_t)a * (int64_t)b;
    int64_t nudge = ab >= 0 ? (1LL << 30) : (1 - (1LL << 30));
    int64_t v = ab + nudge;
    int32_t hi = (int32_t)(v / (1LL << 31));   // C++ division truncates toward zero
    return overflow ? INT32_MAX : hi;
}

static inline int32_t rdbpot(int32_t x, int exponent, int rounding) {
    if (exponent <= 0) return x;
    if (rounding == R_TRUNCATE) return x >= 0 ? (x >> exponent) : -((-x) >> exponent);
    if (rounding == R_FLOOR) return x >> exponent;
    int32_t mask = (int32_t)((1LL << exponent) - 1);
    int32_t remainder = x & mask;
    if (rounding == R_HALF_EVEN) {
        int32_t half = (mask + 1) >> 1;
        int32_t q = x >> exponent;
        bool up = remainder > half || (remainder == half && (q & 1));
        return q + (up ? 1 : 0);
    }
    int32_t threshold = (mask >> 1) + (x < 0 ? 1 : 0);
    return (x >> exponent) + (remainder > threshold ? 1 : 0);
}

static inline int64_t rdbpot64(int64_t x, int exponent) {
    if (exponent <= 0) return x;
    int64_t mask = (1LL << exponent) - 1;
    int64_t remainder = x & mask;
    int64_t threshold = (mask >> 1) + (x < 0 ? 1 : 0);
    return (x >> exponent) + (remainder > threshold ? 1 : 0);
}

static inline int32_t mbqm(int32_t x, int32_t mult, int shift, int rounding) {
    if (rounding == R_SINGLE) {   // TFLITE_SINGLE_ROUNDING: one 64-bit product, one rounding
        int64_t v = rdbpot64((int64_t)x * (int64_t)mult, 31 - shift);
        if (v > INT32_MAX) v = INT32_MAX;
        if (v < INT32_MIN) v = INT32_MIN;
        return (int32_t)v;
    }
    int left = shift > 0 ? shift : 0;
    int right = shift > 0 ? 0 : -shift;
    int64_t xs = (int64_t)x * (1LL << left);
    if (xs > INT32_MAX) xs = INT32_MAX;
    if (xs < INT32_MIN) xs = INT32_MIN;
    return rdbpot(srdhm((int32_t)xs, mult), right, rounding);
}

static inline int32_t sat(int64_t v, int bits) {
    int64_t lo = -(1LL << (bits - 1)), hi = (1LL << (bits - 1)) - 1;
    return (int32_t)(v < lo ? lo : (v > hi ? hi : v));
}

// Direct convolution with integer accumulation and fused requantization.
void conv_requant(const int32_t* x, int N, int C, int H, int W,
                  const int8_t* w, int Cout, int kh, int kw, int sh, int sw, int ph, int pw, int groups,
                  int zp_in, const int32_t* bias, const int32_t* mult, const int32_t* shift,
                  int zp_out, int qmin, int qmax, int rounding, int acc_bits,
                  int32_t* out, int Ho, int Wo, int64_t* n_saturated) {
    int cin_g = C / groups, cout_g = Cout / groups;
    int64_t nsat = 0;
    for (int n = 0; n < N; ++n) {
        for (int oc = 0; oc < Cout; ++oc) {
            int g = oc / cout_g;
            const int8_t* wk = w + (int64_t)oc * cin_g * kh * kw;
            for (int oy = 0; oy < Ho; ++oy) {
                for (int ox = 0; ox < Wo; ++ox) {
                    int64_t acc = 0;
                    for (int ic = 0; ic < cin_g; ++ic) {
                        int c = g * cin_g + ic;
                        const int32_t* xc = x + ((int64_t)n * C + c) * H * W;
                        for (int ky = 0; ky < kh; ++ky) {
                            int iy = oy * sh - ph + ky;
                            if (iy < 0 || iy >= H) continue;
                            for (int kx = 0; kx < kw; ++kx) {
                                int ix = ox * sw - pw + kx;
                                if (ix < 0 || ix >= W) continue;
                                acc += (int64_t)(xc[iy * W + ix] - zp_in) * (int64_t)wk[(ic * kh + ky) * kw + kx];
                            }
                        }
                    }
                    int32_t a = sat(acc, acc_bits);
                    if ((int64_t)a != acc) ++nsat;
                    int64_t ab = (int64_t)a + bias[oc];
                    int32_t a2 = sat(ab, 32);
                    int32_t y = mbqm(a2, mult[oc], shift[oc], rounding) + zp_out;
                    out[(((int64_t)n * Cout + oc) * Ho + oy) * Wo + ox] = std::min(std::max(y, qmin), qmax);
                }
            }
        }
    }
    if (n_saturated) *n_saturated = nsat;
}

void linear_requant(const int32_t* x, int N, int K, const int8_t* w, int M, int zp_in,
                    const int32_t* bias, const int32_t* mult, const int32_t* shift,
                    int zp_out, int qmin, int qmax, int rounding, int acc_bits, int32_t* out) {
    for (int n = 0; n < N; ++n)
        for (int m = 0; m < M; ++m) {
            int64_t acc = 0;
            for (int k = 0; k < K; ++k) acc += (int64_t)(x[n * K + k] - zp_in) * (int64_t)w[m * K + k];
            int32_t a = sat(acc, acc_bits);
            int32_t a2 = sat((int64_t)a + bias[m], 32);
            int32_t y = mbqm(a2, mult[m], shift[m], rounding) + zp_out;
            out[n * M + m] = std::min(std::max(y, qmin), qmax);
        }
}

// TFLite-style add: rescale both operands into a 20-bit-shifted common domain, add, requantize once.
void add_requant(const int32_t* a, const int32_t* b, int64_t n, int zp1, int zp2, int left_shift,
                 int32_t m1, int s1, int32_t m2, int s2, int32_t mo, int so,
                 int zp_out, int qmin, int qmax, int rounding, int32_t* out) {
    for (int64_t i = 0; i < n; ++i) {
        int32_t x1 = (a[i] - zp1) * (1 << left_shift);
        int32_t x2 = (b[i] - zp2) * (1 << left_shift);
        int32_t y1 = mbqm(x1, m1, s1, rounding);
        int32_t y2 = mbqm(x2, m2, s2, rounding);
        int32_t raw = y1 + y2;
        int32_t y = mbqm(raw, mo, so, rounding) + zp_out;
        out[i] = std::min(std::max(y, qmin), qmax);
    }
}

// Global average pool, TFLite semantics (round half away from zero, scale unchanged).
void global_avgpool(const int32_t* x, int N, int C, int HW, int qmin, int qmax, int32_t* out) {
    for (int n = 0; n < N; ++n)
        for (int c = 0; c < C; ++c) {
            int64_t s = 0;
            const int32_t* p = x + ((int64_t)n * C + c) * HW;
            for (int i = 0; i < HW; ++i) s += p[i];
            int64_t y = s >= 0 ? (s + HW / 2) / HW : -((-s + HW / 2) / HW);
            out[n * C + c] = (int32_t)std::min<int64_t>(std::max<int64_t>(y, qmin), qmax);
        }
}

void lut_apply(const int32_t* x, int64_t n, const int32_t* table, int qmin_in, int32_t* out) {
    for (int64_t i = 0; i < n; ++i) out[i] = table[x[i] - qmin_in];
}

// Exposed for unit tests
int32_t test_mbqm(int32_t x, int32_t mult, int shift, int rounding) { return mbqm(x, mult, shift, rounding); }
int32_t test_srdhm(int32_t a, int32_t b) { return srdhm(a, b); }
int32_t test_rdbpot(int32_t x, int e, int rounding) { return rdbpot(x, e, rounding); }

}
