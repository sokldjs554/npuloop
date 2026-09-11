// int8_runner: execute a .npuloop file (see intengine/serialize.py) with no Python at all.
//
//   int8_runner model.npuloop input.bin --float --n 64 --out codes.bin [--dump DIR] [--argmax]
//
// input.bin is raw little-endian data in NCHW order: float32 (--float; quantized here with the file's
// input scale/zero-point, round half to even) or int32 codes. The output tensor's codes are written to
// --out as int32; --dump writes every node's codes to DIR/<node>.i32 so a test can compare all of them
// with the Python engines. Same kernels as the ctypes engine (int8_engine.cpp is included below).
#include "int8_engine.cpp"
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

// ---------------------------------------------------------------- minimal JSON
struct J {
    enum T { NUL, BOOL, NUM, STR, ARR, OBJ } t = NUL;
    bool b = false; double n = 0; std::string s;
    std::vector<J> a; std::vector<std::pair<std::string, J>> o;
    const J* get(const std::string& k) const { for (auto& kv : o) if (kv.first == k) return &kv.second; return nullptr; }
    const J& operator[](const std::string& k) const { const J* v = get(k); if (!v) throw std::runtime_error("missing key " + k); return *v; }
    const J& operator[](size_t i) const { return a.at(i); }
    size_t size() const { return t == ARR ? a.size() : o.size(); }
    long long i() const { return (long long)std::llround(n); }
};

struct Parser {
    const std::string& s; size_t p = 0;
    explicit Parser(const std::string& src) : s(src) {}
    void ws() { while (p < s.size() && (s[p] == ' ' || s[p] == '\n' || s[p] == '\r' || s[p] == '\t')) ++p; }
    bool eat(char c) { ws(); if (p < s.size() && s[p] == c) { ++p; return true; } return false; }
    void expect(char c) { if (!eat(c)) throw std::runtime_error(std::string("json: expected ") + c + " at " + std::to_string(p)); }
    std::string str() {
        expect('"'); std::string out;
        while (p < s.size() && s[p] != '"') {
            char c = s[p++];
            if (c == '\\') {
                char e = s[p++];
                if (e == 'n') out += '\n'; else if (e == 't') out += '\t'; else if (e == 'u') { p += 4; out += '?'; } else out += e;
            } else out += c;
        }
        expect('"'); return out;
    }
    J val() {
        ws(); J v;
        if (p >= s.size()) throw std::runtime_error("json: eof");
        char c = s[p];
        if (c == '{') { v.t = J::OBJ; ++p; if (eat('}')) return v;
            do { std::string k = str(); expect(':'); v.o.emplace_back(k, val()); } while (eat(',')); expect('}'); return v; }
        if (c == '[') { v.t = J::ARR; ++p; if (eat(']')) return v;
            do { v.a.push_back(val()); } while (eat(',')); expect(']'); return v; }
        if (c == '"') { v.t = J::STR; v.s = str(); return v; }
        if (s.compare(p, 4, "true") == 0) { v.t = J::BOOL; v.b = true; p += 4; return v; }
        if (s.compare(p, 5, "false") == 0) { v.t = J::BOOL; v.b = false; p += 5; return v; }
        if (s.compare(p, 4, "null") == 0) { v.t = J::NUL; p += 4; return v; }
        size_t q = p; while (q < s.size() && std::strchr("+-0123456789.eE", s[q])) ++q;
        v.t = J::NUM; v.n = std::strtod(s.substr(p, q - p).c_str(), nullptr); p = q; return v;
    }
};

// ---------------------------------------------------------------- tensors and nodes
struct Tensor {
    std::vector<int> shape; std::vector<int32_t> data;
    size_t numel() const { size_t n = 1; for (int d : shape) n *= (size_t)d; return n; }
};
struct QP { double scale = 1; int zp = 0, qmin = 0, qmax = 255; bool has = false; };
struct Arr { std::string dtype; std::vector<int> shape; const char* ptr = nullptr; size_t nbytes = 0; };
struct Node {
    std::string name, op; std::vector<std::string> inputs; QP q; const J* attrs = nullptr; const J* add_params = nullptr;
    std::map<std::string, Arr> arrays;
    bool has(const char* k) const { return attrs->get(k) != nullptr; }
    long long ai(const char* k, long long def = 0) const { const J* v = attrs->get(k); return v ? v->i() : def; }
    std::vector<int> ail(const char* k) const { std::vector<int> out; for (auto& x : (*attrs)[k].a) out.push_back((int)x.i()); return out; }
    std::string as(const char* k, const char* def = "") const { const J* v = attrs->get(k); return v && v->t == J::STR ? v->s : def; }
    const int32_t* i32(const char* f) const { auto it = arrays.find(f); if (it == arrays.end()) throw std::runtime_error(name + ": missing array " + f); return (const int32_t*)it->second.ptr; }
    const int8_t* i8(const char* f) const { auto it = arrays.find(f); if (it == arrays.end()) throw std::runtime_error(name + ": missing array " + f); return (const int8_t*)it->second.ptr; }
    void clamp(int& lo, int& hi) const { lo = q.qmin; hi = q.qmax; if (const J* c = attrs->get("clamp")) { lo = (int)(*c)[0].i(); hi = (int)(*c)[1].i(); } }
};

static QP parse_q(const J* j) { QP q; if (!j || j->t == J::NUL) return q; q.has = true; q.scale = (*j)["scale"].n; q.zp = (int)(*j)["zero_point"].i(); q.qmin = (int)(*j)["qmin"].i(); q.qmax = (int)(*j)["qmax"].i(); return q; }
static size_t prod(const std::vector<int>& s, size_t a, size_t b) { size_t n = 1; for (size_t i = a; i < b && i < s.size(); ++i) n *= (size_t)s[i]; return n; }
static int32_t clampi(long long v, int lo, int hi) { return (int32_t)(v < lo ? lo : (v > hi ? hi : v)); }

static Tensor broadcast_to(const Tensor& t, const std::vector<int>& shape) {
    if (t.shape == shape) return t;
    if (t.shape.size() != shape.size()) throw std::runtime_error("broadcast: rank mismatch");
    Tensor out; out.shape = shape; out.data.resize(out.numel());
    size_t r = shape.size(); std::vector<size_t> src_stride(r), idx(r, 0);
    size_t st = 1; for (size_t i = r; i-- > 0;) { src_stride[i] = (t.shape[i] == 1) ? 0 : st; st *= (size_t)t.shape[i]; }
    for (size_t o = 0; o < out.data.size(); ++o) {
        size_t src = 0; for (size_t i = 0; i < r; ++i) src += idx[i] * src_stride[i];
        out.data[o] = t.data[src];
        for (size_t i = r; i-- > 0;) { if (++idx[i] < (size_t)shape[i]) break; idx[i] = 0; }
    }
    return out;
}

static Tensor transpose(const Tensor& t, const std::vector<int>& perm) {
    size_t r = t.shape.size(); Tensor out; out.shape.resize(r);
    for (size_t i = 0; i < r; ++i) out.shape[i] = t.shape[perm[i]];
    out.data.resize(t.numel());
    std::vector<size_t> in_stride(r); size_t st = 1; for (size_t i = r; i-- > 0;) { in_stride[i] = st; st *= (size_t)t.shape[i]; }
    std::vector<size_t> idx(r, 0);
    for (size_t o = 0; o < out.data.size(); ++o) {
        size_t src = 0; for (size_t i = 0; i < r; ++i) src += idx[i] * in_stride[perm[i]];
        out.data[o] = t.data[src];
        for (size_t i = r; i-- > 0;) { if (++idx[i] < (size_t)out.shape[i]) break; idx[i] = 0; }
    }
    return out;
}

static int rounding_code(const std::string& r) {
    return r == "tflite" ? R_TFLITE : r == "half_even" ? R_HALF_EVEN : r == "truncate" ? R_TRUNCATE : r == "floor" ? R_FLOOR : R_SINGLE;
}

// ---------------------------------------------------------------- the model file
struct Model {
    std::string buf, header; J h; std::vector<Node> nodes; QP input_q; int rounding = R_TFLITE, acc_bits = 32; const char* raw = nullptr;
    void load(const std::string& path) {
        std::ifstream f(path, std::ios::binary); if (!f) throw std::runtime_error("cannot open " + path);
        buf.assign((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
        if (buf.size() < 12 || buf.compare(0, 8, "NPULOOP1") != 0) throw std::runtime_error("not a .npuloop file");
        uint32_t hlen; std::memcpy(&hlen, buf.data() + 8, 4);
        header = buf.substr(12, hlen); raw = buf.data() + 12 + hlen;
        h = Parser(header).val();
        if (h["version"].i() != 1) throw std::runtime_error("unsupported format version");
        input_q = parse_q(&h["input_q"]);
        rounding = rounding_code(h["requant"]["rounding"].s);
        acc_bits = (int)h["requant"]["acc_bits"].i();
        for (auto& jn : h["nodes"].a) {
            Node n; n.name = jn["name"].s; n.op = jn["op"].s;
            for (auto& x : jn["inputs"].a) n.inputs.push_back(x.s);
            n.q = parse_q(jn.get("out_q")); n.attrs = &jn["attrs"]; n.add_params = jn.get("add_params");
            for (auto& ja : jn["arrays"].a) {
                Arr a; a.dtype = ja["dtype"].s; for (auto& d : ja["shape"].a) a.shape.push_back((int)d.i());
                a.nbytes = (size_t)ja["nbytes"].i(); a.ptr = raw + (size_t)ja["offset"].i();
                n.arrays[ja["field"].s] = a;
            }
            nodes.push_back(n);
        }
    }
};

// ---------------------------------------------------------------- execution
static Tensor run_node(const Model& model, const Node& n, std::map<std::string, Tensor>& vals) {
    auto in = [&](size_t i) -> const Tensor& { auto it = vals.find(n.inputs.at(i)); if (it == vals.end()) throw std::runtime_error(n.name + ": input not computed"); return it->second; };
    auto qof = [&](size_t i) -> QP { for (auto& nd : model.nodes) if (nd.name == n.inputs.at(i)) return nd.q; throw std::runtime_error("unknown input " + n.inputs.at(i)); };
    struct { int rounding; int acc_bits; } m = {n.has("rounding") ? rounding_code(n.as("rounding")) : model.rounding, model.acc_bits};
    Tensor out; int lo, hi; n.clamp(lo, hi);
    if (n.op == "conv") {
        const Tensor& x = in(0); int N = x.shape[0], C = x.shape[1], H = x.shape[2], W = x.shape[3];
        const Arr& wa = n.arrays.at("w_int"); int Cout = wa.shape[0], kh = wa.shape[2], kw = wa.shape[3];
        auto st = n.ail("stride"), pd = n.ail("padding"); int groups = (int)n.ai("groups", 1);
        int Ho = (H + 2 * pd[0] - kh) / st[0] + 1, Wo = (W + 2 * pd[1] - kw) / st[1] + 1;
        out.shape = {N, Cout, Ho, Wo}; out.data.resize(out.numel()); int64_t nsat = 0;
        conv_requant(x.data.data(), N, C, H, W, n.i8("w_int"), Cout, kh, kw, st[0], st[1], pd[0], pd[1], groups, qof(0).zp,
                     n.i32("bias_int"), n.i32("mult"), n.i32("shift"), n.q.zp, lo, hi, m.rounding, m.acc_bits, out.data.data(), Ho, Wo, &nsat);
    } else if (n.op == "linear") {
        const Tensor& x = in(0); int K = x.shape.back(); int rows = (int)(x.numel() / (size_t)K);
        const Arr& wa = n.arrays.at("w_int"); int M = wa.shape[0];
        out.shape = x.shape; out.shape.back() = M; out.data.resize(out.numel());
        linear_requant(x.data.data(), rows, K, n.i8("w_int"), M, qof(0).zp, n.i32("bias_int"), n.i32("mult"), n.i32("shift"),
                       n.q.zp, lo, hi, m.rounding, m.acc_bits, out.data.data());
    } else if (n.op == "add") {
        Tensor a = in(0), b = in(1);
        if (a.shape != b.shape) { std::vector<int> s(a.shape.size()); for (size_t i = 0; i < s.size(); ++i) s[i] = std::max(a.shape[i], b.shape[i]); a = broadcast_to(a, s); b = broadcast_to(b, s); }
        const J& p = *n.add_params; out.shape = a.shape; out.data.resize(a.numel());
        add_requant(a.data.data(), b.data.data(), (int64_t)a.numel(), qof(0).zp, qof(1).zp, (int)p["left_shift"].i(),
                    (int32_t)p["m1"][0].i(), (int)p["m1"][1].i(), (int32_t)p["m2"][0].i(), (int)p["m2"][1].i(), (int32_t)p["mo"][0].i(), (int)p["mo"][1].i(),
                    n.q.zp, lo, hi, m.rounding, out.data.data());
    } else if (n.op == "matmul") {
        const Tensor& a = in(0); const Tensor& b = in(1); size_t r = a.shape.size();
        int M = a.shape[r - 2], K = a.shape[r - 1], N = b.shape[b.shape.size() - 1]; int batch = (int)prod(a.shape, 0, r - 2);
        out.shape = a.shape; out.shape[r - 1] = N; out.data.resize(out.numel()); int64_t nsat = 0;
        matmul_requant(a.data.data(), b.data.data(), batch, M, K, N, qof(0).zp, qof(1).zp, n.i32("mult")[0], n.i32("shift")[0],
                       n.q.zp, lo, hi, m.rounding, m.acc_bits, out.data.data(), &nsat);
    } else if (n.op == "concat") {
        int d = (int)n.ai("dim"); const Tensor& x0 = in(0);
        out.shape = x0.shape; int total_d = 0; for (size_t i = 0; i < n.inputs.size(); ++i) total_d += in(i).shape[d];
        out.shape[d] = total_d; out.data.resize(out.numel());
        size_t outer = prod(x0.shape, 0, d), inner = prod(x0.shape, d + 1, x0.shape.size());
        const int32_t* mult = n.i32("in_mult"); const int32_t* shift = n.i32("in_shift");
        size_t dst_off = 0;
        for (size_t i = 0; i < n.inputs.size(); ++i) {
            const Tensor& xi = in(i); std::vector<int32_t> tmp(xi.numel());
            concat_requant(xi.data.data(), (int64_t)xi.numel(), qof(i).zp, mult[i], (int)shift[i], n.q.zp, n.q.qmin, n.q.qmax, m.rounding, tmp.data());
            size_t chunk = (size_t)xi.shape[d] * inner;
            for (size_t o = 0; o < outer; ++o) std::memcpy(out.data.data() + o * (size_t)total_d * inner + dst_off, tmp.data() + o * chunk, chunk * sizeof(int32_t));
            dst_off += chunk;
        }
    } else if (n.op == "softmax") {
        const Tensor& x = in(0); int d = (int)n.ai("dim"); if (d < 0) d += (int)x.shape.size();
        size_t outer = prod(x.shape, 0, d), inner = prod(x.shape, d + 1, x.shape.size());
        out.shape = x.shape; out.data.resize(x.numel());
        softmax_int(x.data.data(), (int64_t)outer, x.shape[d], (int64_t)inner, n.i32("lut"), (int)n.ai("lut_offset"), n.i32("mult")[0], n.i32("shift")[0],
                    n.q.zp, n.q.qmin, n.q.qmax, m.rounding, out.data.data());
    } else if (n.op == "layernorm") {
        const Tensor& x = in(0); int C = (int)n.ai("channels"); int64_t rows = (int64_t)(x.numel() / (size_t)C);
        out.shape = x.shape; out.data.resize(x.numel());
        layernorm_int(x.data.data(), rows, C, qof(0).zp, (int64_t)n.ai("eps_int"), n.i32("mult"), n.i32("shift"), n.i32("bias_int"),
                      n.q.zp, n.q.qmin, n.q.qmax, m.rounding, out.data.data());
    } else if (n.op == "lut") {
        const Tensor& x = in(0); out.shape = x.shape; out.data.resize(x.numel());
        lut_apply(x.data.data(), (int64_t)x.numel(), n.i32("lut"), (int)n.ai("lut_qmin"), out.data.data());
    } else if (n.op == "pool") {
        const Tensor& x = in(0);
        if (n.as("kind") == "token_mean") {
            int N = x.shape[0], T = x.shape[1], C = x.shape[2];
            out.shape = n.ai("keepdim", 0) ? std::vector<int>{N, 1, C} : std::vector<int>{N, C}; out.data.resize((size_t)N * C);
            token_mean(x.data.data(), N, T, C, n.q.qmin, n.q.qmax, out.data.data());
        } else if (n.as("kind") == "global_avg_requant") {
            int N = x.shape[0], C = x.shape[1], HW = x.shape[2] * x.shape[3];
            out.shape = {N, C, 1, 1}; out.data.resize((size_t)N * C);
            global_avgpool_requant(x.data.data(), N, C, HW, qof(0).zp, n.i32("mult")[0], n.i32("shift")[0], n.q.zp, n.q.qmin, n.q.qmax, m.rounding, out.data.data());
        } else {
            int N = x.shape[0], C = x.shape[1], HW = x.shape[2] * x.shape[3];
            out.shape = {N, C, 1, 1}; out.data.resize((size_t)N * C);
            global_avgpool(x.data.data(), N, C, HW, n.q.qmin, n.q.qmax, out.data.data());
        }
    } else if (n.op == "transpose") {
        out = transpose(in(0), n.ail("perm"));
    } else if (n.op == "reshape") {
        const Tensor& x = in(0); out.data = x.data; out.shape = {x.shape[0]}; for (int d : n.ail("out_shape")) out.shape.push_back(d);
        if (out.numel() != x.numel()) throw std::runtime_error(n.name + ": reshape size mismatch");
    } else if (n.op == "flatten") {
        const Tensor& x = in(0); out.data = x.data; out.shape = {x.shape[0], (int)(x.numel() / (size_t)x.shape[0])};
    } else if (n.op == "const") {
        const Arr& a = n.arrays.at("codes"); out.shape = a.shape; out.data.assign((const int32_t*)a.ptr, (const int32_t*)a.ptr + a.nbytes / 4);
    } else if (n.op == "output") {
        out = in(0);
    } else {
        throw std::runtime_error("unsupported op " + n.op);
    }
    return out;
}

int main(int argc, char** argv) {
    if (argc < 3) { std::fprintf(stderr, "usage: %s model.npuloop input.bin [--float] [--n N] [--out codes.bin] [--dump DIR] [--argmax]\n", argv[0]); return 2; }
    std::string model_path = argv[1], input_path = argv[2], out_path, dump_dir; bool is_float = false, argmax = false; int N = -1;
    for (int i = 3; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--float") is_float = true; else if (a == "--argmax") argmax = true;
        else if (a == "--n" && i + 1 < argc) N = std::atoi(argv[++i]);
        else if (a == "--out" && i + 1 < argc) out_path = argv[++i];
        else if (a == "--dump" && i + 1 < argc) dump_dir = argv[++i];
        else { std::fprintf(stderr, "unknown argument %s\n", a.c_str()); return 2; }
    }
    try {
        Model m; m.load(model_path);
        const Node& input = m.nodes.at(0); if (input.op != "input") throw std::runtime_error("first node is not the input");
        std::vector<int> per_image = input.ail("out_shape"); size_t per = prod(per_image, 0, per_image.size());
        std::ifstream f(input_path, std::ios::binary); if (!f) throw std::runtime_error("cannot open " + input_path);
        std::string raw((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
        size_t elem = is_float ? 4 : 4; size_t total = raw.size() / elem;
        if (N < 0) N = (int)(total / per);
        if ((size_t)N * per > total) throw std::runtime_error("input file too small for N images");
        Tensor x; x.shape = {N}; for (int d : per_image) x.shape.push_back(d); x.data.resize((size_t)N * per);
        if (is_float) {
            const float* fp = (const float*)raw.data();
            for (size_t i = 0; i < x.data.size(); ++i) {
                double c = std::nearbyint((double)fp[i] / m.input_q.scale) + m.input_q.zp;      // half to even, like numpy.rint
                x.data[i] = clampi((long long)c, m.input_q.qmin, m.input_q.qmax);
            }
        } else {
            std::memcpy(x.data.data(), raw.data(), x.data.size() * 4);
        }
        std::map<std::string, Tensor> vals; vals[input.name] = x;
        Tensor result;
        for (size_t i = 1; i < m.nodes.size(); ++i) {
            const Node& n = m.nodes[i];
            Tensor t = run_node(m, n, vals);
            if (n.op == "output") result = t;
            if (!dump_dir.empty()) {
                std::ofstream d(dump_dir + "/" + n.name + ".i32", std::ios::binary);
                d.write((const char*)t.data.data(), t.data.size() * 4);
            }
            vals[n.name] = std::move(t);
        }
        if (!out_path.empty()) { std::ofstream o(out_path, std::ios::binary); o.write((const char*)result.data.data(), result.data.size() * 4); }
        std::printf("ran %zu nodes on %d images; output shape [", m.nodes.size(), N);
        for (size_t i = 0; i < result.shape.size(); ++i) std::printf("%s%d", i ? "," : "", result.shape[i]);
        std::printf("]\n");
        if (argmax && result.shape.size() == 2) {
            for (int i = 0; i < N; ++i) {
                int best = 0; for (int c = 1; c < result.shape[1]; ++c) if (result.data[(size_t)i * result.shape[1] + c] > result.data[(size_t)i * result.shape[1] + best]) best = c;
                std::printf("%d%s", best, i + 1 < N ? " " : "\n");
            }
        }
    } catch (const std::exception& e) {
        std::fprintf(stderr, "int8_runner: %s\n", e.what()); return 1;
    }
    return 0;
}
