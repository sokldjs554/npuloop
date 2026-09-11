// npuloop integer kernels: bit-exact with the NumPy engine and with gemmlowp/TFLite reference semantics.
// Tensors are NCHW; codes are carried as int32 (values fit uint8/int8); accumulators are exact integer
// sums (int32 when provably safe, int64 otherwise) saturated to acc_bits; requantization uses the
// fixed-point multiplier + shift scheme.
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

// Largest |x - zp| in a tensor: decides whether int32 partial sums are provably exact for a layer.
static inline int64_t max_abs_centered(const int32_t* x, int64_t n, int zp) {
    int64_t m = 0;
    for (int64_t i = 0; i < n; ++i) { int64_t v = x[i] - zp; if (v < 0) v = -v; if (v > m) m = v; }
    return m;
}

// Epilogue shared by conv/linear: saturate the accumulator to acc_bits, add bias, requantize, clamp.
static inline int32_t requant_out(int64_t acc, int acc_bits, int32_t bias, int32_t mult, int32_t shift, int rounding,
                                  int zp_out, int qmin, int qmax, int64_t& nsat) {
    int32_t a = sat(acc, acc_bits);
    if ((int64_t)a != acc) ++nsat;
    int32_t a2 = sat((int64_t)a + bias, 32);
    int32_t y = mbqm(a2, mult, shift, rounding) + zp_out;
    return std::min(std::max(y, qmin), qmax);
}

// Convolution as im2col + integer GEMM with fused requantization.
//
// For one image and one group the zero-point-subtracted input is unfolded into Xt (K x P, P = Ho*Wo output
// pixels, K = cin_g*kh*kw); each output channel is then K axpy passes over a P-long int32 row, which the
// compiler vectorizes (no horizontal reductions). Partial sums are kept in int32 whenever K * max|x - zp| * 128 < 2^31 — true for
// every 8-bit layer — and in int64 otherwise, so the result equals the naive int64 sum either way; the
// epilogue (saturate to acc_bits, bias, requantize) is shared.
}  // extern "C" (templates cannot have C linkage)

static void im2col_t(const int32_t* xm, int cin_g, int H, int W, int kh, int kw, int sh, int sw, int ph, int pw,
                     int Ho, int Wo, int32_t* Xt) {
    // Xt is K x P (one row per (ic, ky, kx) tap, P = Ho*Wo pixels) so the GEMM below streams over pixels.
    const int64_t P = (int64_t)Ho * Wo;
    for (int ic = 0; ic < cin_g; ++ic)
        for (int ky = 0; ky < kh; ++ky)
            for (int kx = 0; kx < kw; ++kx) {
                int32_t* row = Xt + (int64_t)((ic * kh + ky) * kw + kx) * P;
                for (int oy = 0; oy < Ho; ++oy) {
                    int iy = oy * sh - ph + ky;
                    for (int ox = 0; ox < Wo; ++ox) {
                        int ix = ox * sw - pw + kx;
                        row[(int64_t)oy * Wo + ox] = (iy < 0 || iy >= H || ix < 0 || ix >= W) ? 0 : xm[((int64_t)ic * H + iy) * W + ix];
                    }
                }
            }
}

template <typename Acc>
static void gemm_requant(const int32_t* Xt, const int32_t* Wm, int P, int K, int cout_g, int oc0,
                         int acc_bits, const int32_t* bias, const int32_t* mult, const int32_t* shift, int rounding,
                         int zp_out, int qmin, int qmax, int32_t* out, Acc* acc, int64_t& nsat) {
    for (int oc = 0; oc < cout_g; ++oc) {
        const int32_t* wr = Wm + (int64_t)oc * K;
        for (int p = 0; p < P; ++p) acc[p] = 0;
        int k = 0;
        for (; k + 4 <= K; k += 4) {          // four taps per pass: one accumulator load/store per 4 MACs
            const Acc w0 = (Acc)wr[k], w1 = (Acc)wr[k + 1], w2 = (Acc)wr[k + 2], w3 = (Acc)wr[k + 3];
            const int32_t* __restrict xr = Xt + (int64_t)k * P;
            Acc* __restrict ar = acc;
            for (int p = 0; p < P; ++p)
                ar[p] += (Acc)xr[p] * w0 + (Acc)xr[P + p] * w1 + (Acc)xr[2 * P + p] * w2 + (Acc)xr[3 * P + p] * w3;
        }
        for (; k < K; ++k) {
            const Acc wv = (Acc)wr[k];
            if (wv == 0) continue;
            const int32_t* __restrict xr = Xt + (int64_t)k * P;
            Acc* __restrict ar = acc;
            for (int p = 0; p < P; ++p) ar[p] += (Acc)xr[p] * wv;
        }
        int32_t* o = out + (int64_t)oc * P;
        for (int p = 0; p < P; ++p) {
            int32_t s = sat((int64_t)acc[p], acc_bits);
            if ((int64_t)s != (int64_t)acc[p]) ++nsat;
            int32_t s2 = sat((int64_t)s + bias[oc0 + oc], 32);
            int32_t y = mbqm(s2, mult[oc0 + oc], shift[oc0 + oc], rounding) + zp_out;
            o[p] = std::min(std::max(y, qmin), qmax);
        }
    }
}

extern "C" {

void conv_requant(const int32_t* x, int N, int C, int H, int W,
                  const int8_t* w, int Cout, int kh, int kw, int sh, int sw, int ph, int pw, int groups,
                  int zp_in, const int32_t* bias, const int32_t* mult, const int32_t* shift,
                  int zp_out, int qmin, int qmax, int rounding, int acc_bits,
                  int32_t* out, int Ho, int Wo, int64_t* n_saturated) {
    const int cin_g = C / groups, cout_g = Cout / groups;
    const int64_t HW = (int64_t)H * W, P = (int64_t)Ho * Wo, K = (int64_t)cin_g * kh * kw;
    const bool exact32 = K * max_abs_centered(x, (int64_t)N * C * HW, zp_in) * 128 < (1LL << 31);
    int32_t* xm = (int32_t*)std::malloc(sizeof(int32_t) * (size_t)(cin_g * HW));
    int32_t* X = (int32_t*)std::malloc(sizeof(int32_t) * (size_t)(P * K));
    int32_t* Wm = (int32_t*)std::malloc(sizeof(int32_t) * (size_t)((int64_t)Cout * K));
    int32_t* acc32 = exact32 ? (int32_t*)std::malloc(sizeof(int32_t) * (size_t)P) : nullptr;
    int64_t* acc64 = exact32 ? nullptr : (int64_t*)std::malloc(sizeof(int64_t) * (size_t)P);
    for (int64_t i = 0; i < (int64_t)Cout * K; ++i) Wm[i] = w[i];
    int64_t nsat = 0;
    for (int n = 0; n < N; ++n)
        for (int g = 0; g < groups; ++g) {
            const int32_t* xg = x + ((int64_t)n * C + (int64_t)g * cin_g) * HW;
            for (int64_t i = 0; i < cin_g * HW; ++i) xm[i] = xg[i] - zp_in;
            im2col_t(xm, cin_g, H, W, kh, kw, sh, sw, ph, pw, Ho, Wo, X);
            int32_t* o = out + ((int64_t)n * Cout + (int64_t)g * cout_g) * P;
            if (exact32) gemm_requant<int32_t>(X, Wm + (int64_t)g * cout_g * K, (int)P, (int)K, cout_g, g * cout_g, acc_bits, bias, mult, shift, rounding, zp_out, qmin, qmax, o, acc32, nsat);
            else         gemm_requant<int64_t>(X, Wm + (int64_t)g * cout_g * K, (int)P, (int)K, cout_g, g * cout_g, acc_bits, bias, mult, shift, rounding, zp_out, qmin, qmax, o, acc64, nsat);
        }
    std::free(xm); std::free(X); std::free(Wm); std::free(acc32); std::free(acc64);
    if (n_saturated) *n_saturated = nsat;
}

void linear_requant(const int32_t* x, int N, int K, const int8_t* w, int M, int zp_in,
                    const int32_t* bias, const int32_t* mult, const int32_t* shift,
                    int zp_out, int qmin, int qmax, int rounding, int acc_bits, int32_t* out) {
    const bool exact32 = (int64_t)K * max_abs_centered(x, (int64_t)N * K, zp_in) * 128 < (1LL << 31);
    int32_t* xm = (int32_t*)std::malloc(sizeof(int32_t) * (size_t)K);
    int64_t nsat = 0;
    for (int n = 0; n < N; ++n) {
        for (int k = 0; k < K; ++k) xm[k] = x[(int64_t)n * K + k] - zp_in;
        for (int m = 0; m < M; ++m) {
            const int8_t* __restrict wm = w + (int64_t)m * K;
            int64_t acc;
            if (exact32) { int32_t a = 0; for (int k = 0; k < K; ++k) a += xm[k] * (int32_t)wm[k]; acc = a; }
            else         { int64_t a = 0; for (int k = 0; k < K; ++k) a += (int64_t)xm[k] * (int64_t)wm[k]; acc = a; }
            out[(int64_t)n * M + m] = requant_out(acc, acc_bits, bias[m], mult[m], shift[m], rounding, zp_out, qmin, qmax, nsat);
        }
    }
    std::free(xm);
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

// Global average pool with its own output scale (TFLite MEAN over H,W): one requantization of the centred sum.
void global_avgpool_requant(const int32_t* x, int N, int C, int HW, int zp_in, int32_t mult, int shift,
                            int zp_out, int qmin, int qmax, int rounding, int32_t* out) {
    for (int n = 0; n < N; ++n)
        for (int c = 0; c < C; ++c) {
            int64_t s = 0;
            const int32_t* p = x + ((int64_t)n * C + c) * HW;
            for (int i = 0; i < HW; ++i) s += p[i] - zp_in;
            int32_t y = mbqm(sat(s, 32), mult, shift, rounding) + zp_out;
            out[n * C + c] = std::min(std::max(y, qmin), qmax);
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
// carry a zero point, so the accumulator needs the full (a - za)(b - zb) product. The k loop is outside
// the contiguous j loop (an int32 axpy per row), with the same int32/int64 exactness rule as conv.
}  // extern "C"

template <typename Acc>
static void matmul_rows(const int32_t* A, const int32_t* B, int M, int K, int N, int zp_a, int zp_b,
                        int32_t* Bm, Acc* acc, int32_t mult, int shift, int zp_out, int qmin, int qmax,
                        int rounding, int acc_bits, int32_t* O, int64_t& nsat) {
    for (int64_t i = 0; i < (int64_t)K * N; ++i) Bm[i] = B[i] - zp_b;
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) acc[j] = 0;
        for (int k = 0; k < K; ++k) {
            Acc av = (Acc)(A[(int64_t)i * K + k] - zp_a);
            if (av == 0) continue;
            const int32_t* __restrict br = Bm + (int64_t)k * N;
            for (int j = 0; j < N; ++j) acc[j] += av * (Acc)br[j];
        }
        for (int j = 0; j < N; ++j) {
            int64_t a = acc[j];
            int32_t s = sat(a, acc_bits);
            if ((int64_t)s != a) ++nsat;
            int32_t y = mbqm(s, mult, shift, rounding) + zp_out;
            O[(int64_t)i * N + j] = std::min(std::max(y, qmin), qmax);
        }
    }
}

extern "C" {

void matmul_requant(const int32_t* a, const int32_t* b, int batch, int M, int K, int N,
                    int zp_a, int zp_b, int32_t mult, int shift, int zp_out, int qmin, int qmax,
                    int rounding, int acc_bits, int32_t* out, int64_t* n_saturated) {
    int64_t nsat = 0;
    const int64_t ma = max_abs_centered(a, (int64_t)batch * M * K, zp_a), mb = max_abs_centered(b, (int64_t)batch * K * N, zp_b);
    const bool exact32 = (int64_t)K * ma * mb < (1LL << 31);
    int32_t* Bm = (int32_t*)std::malloc(sizeof(int32_t) * (size_t)K * N);
    int32_t* acc32 = exact32 ? (int32_t*)std::malloc(sizeof(int32_t) * (size_t)N) : nullptr;
    int64_t* acc64 = exact32 ? nullptr : (int64_t*)std::malloc(sizeof(int64_t) * (size_t)N);
    for (int g = 0; g < batch; ++g) {
        const int32_t* A = a + (int64_t)g * M * K;
        const int32_t* B = b + (int64_t)g * K * N;
        int32_t* O = out + (int64_t)g * M * N;
        if (exact32) matmul_rows<int32_t>(A, B, M, K, N, zp_a, zp_b, Bm, acc32, mult, shift, zp_out, qmin, qmax, rounding, acc_bits, O, nsat);
        else         matmul_rows<int64_t>(A, B, M, K, N, zp_a, zp_b, Bm, acc64, mult, shift, zp_out, qmin, qmax, rounding, acc_bits, O, nsat);
    }
    std::free(Bm); std::free(acc32); std::free(acc64);
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
