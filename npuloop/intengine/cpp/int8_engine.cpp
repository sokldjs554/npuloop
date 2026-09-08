// npuloop integer kernels: bit-exact with the NumPy engine and with gemmlowp/TFLite reference semantics.
// Tensors are NCHW; codes are carried as int32 (values fit uint8/int8); accumulators are int32 with
// explicit saturation; requantization uses the fixed-point multiplier + shift scheme.
#include <cstdint>
#include <cstdlib>
#include <algorithm>
#include <cmath>

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
    int64_t mask = (1LL << exponent) - 1;
    int64_t remainder = (int64_t)x & mask;
    if (rounding == R_HALF_EVEN) {
        int64_t half = (mask + 1) >> 1;
        int32_t q = x >> exponent;
        bool up = remainder > half || (remainder == half && (q & 1));
        return q + (up ? 1 : 0);
    }
    int64_t threshold = (mask >> 1) + (x < 0 ? 1 : 0);
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

// Token mean over the second axis, TFLite average semantics (round half away from zero).
void token_mean(const int32_t* x, int N, int T, int C, int qmin, int qmax, int32_t* out) {
    for (int n = 0; n < N; ++n)
        for (int c = 0; c < C; ++c) {
            int64_t s = 0;
            for (int t = 0; t < T; ++t) s += x[((int64_t)n * T + t) * C + c];
            int64_t y = s >= 0 ? (s + T / 2) / T : -((-s + T / 2) / T);
            out[(int64_t)n * C + c] = (int32_t)std::min<int64_t>(std::max<int64_t>(y, qmin), qmax);
        }
}

// Activation x activation matmul: (batch, M, K) @ (batch, K, N). No weight reuse, both operands
// carry a zero point, so the accumulator needs the full (a - za)(b - zb) product.
void matmul_requant(const int32_t* a, const int32_t* b, int batch, int M, int K, int N,
                    int zp_a, int zp_b, int32_t mult, int shift, int zp_out, int qmin, int qmax,
                    int rounding, int acc_bits, int32_t* out, int64_t* n_saturated) {
    int64_t nsat = 0;
    for (int g = 0; g < batch; ++g) {
        const int32_t* A = a + (int64_t)g * M * K;
        const int32_t* B = b + (int64_t)g * K * N;
        int32_t* O = out + (int64_t)g * M * N;
        for (int i = 0; i < M; ++i)
            for (int j = 0; j < N; ++j) {
                int64_t acc = 0;
                for (int k = 0; k < K; ++k)
                    acc += (int64_t)(A[i * K + k] - zp_a) * (int64_t)(B[k * N + j] - zp_b);
                int32_t s = sat(acc, acc_bits);
                if ((int64_t)s != acc) ++nsat;
                int32_t y = mbqm(s, mult, shift, rounding) + zp_out;
                O[i * N + j] = std::min(std::max(y, qmin), qmax);
            }
    }
    if (n_saturated) *n_saturated = nsat;
}

// One concat input rescaled into the shared output scale.
void concat_requant(const int32_t* x, int64_t n, int zp_in, int32_t mult, int shift,
                    int zp_out, int qmin, int qmax, int rounding, int32_t* out) {
    for (int64_t i = 0; i < n; ++i) {
        int32_t y = mbqm(x[i] - zp_in, mult, shift, rounding) + zp_out;
        out[i] = std::min(std::max(y, qmin), qmax);
    }
}

static inline int64_t round_div64(int64_t num, int64_t den) {
    int64_t half = den / 2;
    return num >= 0 ? (num + half) / den : -((-num + half) / den);
}

// exp from a Q15 table indexed by (q - rowmax), then an exact integer normalization.
void softmax_int(const int32_t* x, int64_t outer, int D, int64_t inner, const int32_t* table, int offset,
                 int32_t mult, int shift, int zp_out, int qmin, int qmax, int rounding, int32_t* out) {
    for (int64_t o = 0; o < outer; ++o)
        for (int64_t i = 0; i < inner; ++i) {
            int32_t mx = INT32_MIN;
            for (int j = 0; j < D; ++j) {
                int32_t v = x[(o * D + j) * inner + i];
                if (v > mx) mx = v;
            }
            int64_t total = 0;
            for (int j = 0; j < D; ++j) total += table[x[(o * D + j) * inner + i] - mx + offset];
            if (total < 1) total = 1;
            for (int j = 0; j < D; ++j) {
                int64_t e = table[x[(o * D + j) * inner + i] - mx + offset];
                int64_t v = round_div64(e << 15, total);
                int32_t y = mbqm(sat(v, 32), mult, shift, rounding) + zp_out;
                out[(o * D + j) * inner + i] = std::min(std::max(y, qmin), qmax);
            }
        }
}

// floor(sqrt(x)) for x >= 0, exactly (double sqrt then an integer correction).
static inline int64_t isqrt64c(int64_t x) {
    int64_t r = (int64_t)std::sqrt((double)x) - 2;
    if (r < 0) r = 0;
    while ((r + 1) * (r + 1) <= x) ++r;
    return r;
}

// LayerNorm over the last axis in integer arithmetic: int64 sums, exact integer sqrt, per-channel
// requantization carrying gamma; beta is added in the output domain.
void layernorm_int(const int32_t* x, int64_t rows, int C, int zp_in, int64_t eps_int,
                   const int32_t* mult, const int32_t* shift, const int32_t* bias,
                   int zp_out, int qmin, int qmax, int rounding, int32_t* out) {
    for (int64_t r = 0; r < rows; ++r) {
        const int32_t* px = x + r * C;
        int32_t* po = out + r * C;
        int64_t s = 0, sq = 0;
        for (int c = 0; c < C; ++c) { int64_t d = px[c] - zp_in; s += d; sq += d * d; }
        int64_t var_num = (int64_t)C * sq - s * s;
        int64_t denom = isqrt64c(var_num + eps_int);
        if (denom < 1) denom = 1;
        for (int c = 0; c < C; ++c) {
            int64_t centered = ((int64_t)px[c] - zp_in) * C - s;
            int64_t t = round_div64(centered << 15, denom);
            int32_t y = mbqm(sat(t, 32), mult[c], shift[c], rounding) + bias[c] + zp_out;
            po[c] = std::min(std::max(y, qmin), qmax);
        }
    }
}

// Exposed for unit tests
int32_t test_mbqm(int32_t x, int32_t mult, int shift, int rounding) { return mbqm(x, mult, shift, rounding); }
int32_t test_srdhm(int32_t a, int32_t b) { return srdhm(a, b); }
int32_t test_rdbpot(int32_t x, int e, int rounding) { return rdbpot(x, e, rounding); }

}
