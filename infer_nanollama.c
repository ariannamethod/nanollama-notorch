/*
 * infer_nanollama.c — Inference for nanollama 89M on notorch.
 *   13L, dim=576, 9H MHA, FFN=1536, vocab=32000 SentencePiece BPE
 *   No tokenizer here — reads token IDs on stdin, emits IDs on stdout.
 *   Pair with generate.py for SentencePiece encode/decode.
 *
 * Build: cc -O3 -std=c11 -Wall -DUSE_BLAS -DACCELERATE \
 *           -o infer_nanollama infer_nanollama.c notorch.c \
 *           -framework Accelerate -lm
 *
 * Protocol (stdin):
 *   Line 1: "<n_prompt> <n_gen> <temperature> <top_k> <seed>"
 *   Line 2: "<id1> <id2> ... <idN>"   (n_prompt token IDs space-separated)
 *
 * Protocol (stdout):
 *   "<id1> <id2> ... <idM>"           (n_gen generated token IDs)
 *   "DONE\n"
 *
 * Adapted from notorch/examples/infer_llama3_bpe.c.
 */

#include "notorch.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

#ifdef USE_BLAS
  #ifdef ACCELERATE
    #include <Accelerate/Accelerate.h>
  #else
    #include <cblas.h>
  #endif
#endif

#define DIM       576
#define NLAYERS   13
#define NHEADS    9
#define HEAD_DIM  (DIM / NHEADS)   /* 64 */
#define HIDDEN    1536
#define CTX       512
#define VOCAB     32000

typedef struct {
    nt_tensor *wte;
    struct {
        nt_tensor *rms1, *wq, *wk, *wv, *wo, *rms2;
        nt_tensor *w_gate, *w_up, *w_down;
    } L[NLAYERS];
    nt_tensor *rms_f, *head;
} Model;

static int model_n_tensors(void) { return 1 + NLAYERS * 9 + 2; }

static Model* model_new(void) {
    Model* m = (Model*)calloc(1, sizeof(Model));
    m->wte = nt_tensor_new2d(VOCAB, DIM);
    for (int l = 0; l < NLAYERS; l++) {
        m->L[l].rms1 = nt_tensor_new(DIM);
        m->L[l].wq = nt_tensor_new2d(DIM, DIM);
        m->L[l].wk = nt_tensor_new2d(DIM, DIM);
        m->L[l].wv = nt_tensor_new2d(DIM, DIM);
        m->L[l].wo = nt_tensor_new2d(DIM, DIM);
        m->L[l].rms2 = nt_tensor_new(DIM);
        m->L[l].w_gate = nt_tensor_new2d(HIDDEN, DIM);
        m->L[l].w_up = nt_tensor_new2d(HIDDEN, DIM);
        m->L[l].w_down = nt_tensor_new2d(DIM, HIDDEN);
    }
    m->rms_f = nt_tensor_new(DIM);
    m->head = nt_tensor_new2d(VOCAB, DIM);
    return m;
}

static int load_weights(Model* m, const char* path) {
    int n_loaded = 0;
    nt_tensor** loaded = nt_load(path, &n_loaded);
    if (!loaded) return -1;
    int expected = model_n_tensors();
    if (n_loaded != expected) {
        fprintf(stderr, "load: expected %d tensors, got %d\n", expected, n_loaded);
        for (int i = 0; i < n_loaded; i++) nt_tensor_free(loaded[i]);
        free(loaded); return -1;
    }
    nt_tensor** params = (nt_tensor**)malloc(expected * sizeof(nt_tensor*));
    int pi = 0;
    params[pi++] = m->wte;
    for (int l = 0; l < NLAYERS; l++) {
        params[pi++] = m->L[l].rms1;
        params[pi++] = m->L[l].wq;
        params[pi++] = m->L[l].wk;
        params[pi++] = m->L[l].wv;
        params[pi++] = m->L[l].wo;
        params[pi++] = m->L[l].rms2;
        params[pi++] = m->L[l].w_gate;
        params[pi++] = m->L[l].w_up;
        params[pi++] = m->L[l].w_down;
    }
    params[pi++] = m->rms_f;
    params[pi++] = m->head;
    for (int i = 0; i < expected; i++) {
        if (loaded[i]->len != params[i]->len) {
            fprintf(stderr, "load: tensor %d size mismatch (%d vs %d)\n",
                    i, loaded[i]->len, params[i]->len);
            free(params);
            return -1;
        }
        memcpy(params[i]->data, loaded[i]->data, params[i]->len * sizeof(float));
        nt_tensor_free(loaded[i]);
    }
    free(loaded); free(params);
    return 0;
}

/* ── Forward (no tape) ── */

static void rmsnorm(float* out, const float* x, const float* w, int d) {
    float ss = 0;
    for (int i = 0; i < d; i++) ss += x[i] * x[i];
    ss = 1.0f / sqrtf(ss / d + 1e-5f);
    for (int i = 0; i < d; i++) out[i] = x[i] * ss * w[i];
}

static void matmul(float* out, const float* x, const float* w, int out_d, int in_d) {
#ifdef USE_BLAS
    /* y = W @ x, W is [out_d, in_d] row-major */
    cblas_sgemv(CblasRowMajor, CblasNoTrans,
                out_d, in_d, 1.0f, w, in_d, x, 1, 0.0f, out, 1);
#else
    for (int o = 0; o < out_d; o++) {
        float s = 0;
        for (int i = 0; i < in_d; i++) s += w[o * in_d + i] * x[i];
        out[o] = s;
    }
#endif
}

static void rope(float* x, int pos, int dim, int head_dim) {
    for (int h = 0; h < dim / head_dim; h++) {
        for (int i = 0; i < head_dim / 2; i++) {
            float freq = 1.0f / powf(10000.0f, (float)(2 * i) / head_dim);
            float theta = pos * freq;
            float cs = cosf(theta), sn = sinf(theta);
            int idx = h * head_dim + i * 2;
            float x0 = x[idx], x1 = x[idx + 1];
            x[idx]     = x0 * cs - x1 * sn;
            x[idx + 1] = x0 * sn + x1 * cs;
        }
    }
}

/* KV cache (heap — too big for stack at CTX=512, NLAYERS=13, DIM=576) */
static float (*kv_k)[CTX][DIM];   /* [NLAYERS][CTX][DIM] */
static float (*kv_v)[CTX][DIM];

static void forward_pos(Model* m, int token, int pos, float* logits) {
    static float x[DIM], xn[DIM], q[DIM], k[DIM], v[DIM], attn_out[DIM];
    static float gate[HIDDEN], up[HIDDEN], down[DIM];
    static float scores[CTX];

    memcpy(x, m->wte->data + token * DIM, DIM * sizeof(float));

    for (int l = 0; l < NLAYERS; l++) {
        rmsnorm(xn, x, m->L[l].rms1->data, DIM);
        matmul(q, xn, m->L[l].wq->data, DIM, DIM);
        matmul(k, xn, m->L[l].wk->data, DIM, DIM);
        matmul(v, xn, m->L[l].wv->data, DIM, DIM);
        rope(q, pos, DIM, HEAD_DIM);
        rope(k, pos, DIM, HEAD_DIM);

        memcpy(kv_k[l][pos], k, DIM * sizeof(float));
        memcpy(kv_v[l][pos], v, DIM * sizeof(float));

        float scale = 1.0f / sqrtf((float)HEAD_DIM);
        memset(attn_out, 0, DIM * sizeof(float));
        for (int h = 0; h < NHEADS; h++) {
            int ho = h * HEAD_DIM;
            for (int j = 0; j <= pos; j++) {
                float dot = 0;
                for (int d = 0; d < HEAD_DIM; d++) dot += q[ho + d] * kv_k[l][j][ho + d];
                scores[j] = dot * scale;
            }
            float mx = scores[0];
            for (int j = 1; j <= pos; j++) if (scores[j] > mx) mx = scores[j];
            float sm = 0;
            for (int j = 0; j <= pos; j++) { scores[j] = expf(scores[j] - mx); sm += scores[j]; }
            for (int j = 0; j <= pos; j++) scores[j] /= sm;
            for (int j = 0; j <= pos; j++)
                for (int d = 0; d < HEAD_DIM; d++)
                    attn_out[ho + d] += scores[j] * kv_v[l][j][ho + d];
        }

        float proj[DIM];
        matmul(proj, attn_out, m->L[l].wo->data, DIM, DIM);
        for (int i = 0; i < DIM; i++) x[i] += proj[i];

        rmsnorm(xn, x, m->L[l].rms2->data, DIM);
        matmul(gate, xn, m->L[l].w_gate->data, HIDDEN, DIM);
        matmul(up, xn, m->L[l].w_up->data, HIDDEN, DIM);
        for (int i = 0; i < HIDDEN; i++)
            gate[i] = gate[i] / (1.0f + expf(-gate[i])) * up[i];
        matmul(down, gate, m->L[l].w_down->data, DIM, HIDDEN);
        for (int i = 0; i < DIM; i++) x[i] += down[i];
    }

    rmsnorm(xn, x, m->rms_f->data, DIM);
    matmul(logits, xn, m->head->data, VOCAB, DIM);
}

static void softmax(float* x, int n) {
    float mx = x[0]; for (int i = 1; i < n; i++) if (x[i] > mx) mx = x[i];
    float sm = 0; for (int i = 0; i < n; i++) { x[i] = expf(x[i] - mx); sm += x[i]; }
    for (int i = 0; i < n; i++) x[i] /= sm;
}

static int sample(float* logits, float temperature, int top_k) {
    if (temperature <= 0.0f) {
        int mi = 0; float mx = logits[0];
        for (int i = 1; i < VOCAB; i++) if (logits[i] > mx) { mx = logits[i]; mi = i; }
        return mi;
    }
    for (int i = 0; i < VOCAB; i++) logits[i] /= temperature;
    if (top_k > 0 && top_k < VOCAB) {
        static float tmp[VOCAB];
        memcpy(tmp, logits, VOCAB * sizeof(float));
        float threshold = -1e30f;
        for (int k = 0; k < top_k; k++) {
            float mx = -1e30f; int mi = 0;
            for (int i = 0; i < VOCAB; i++) if (tmp[i] > mx) { mx = tmp[i]; mi = i; }
            threshold = mx;
            tmp[mi] = -1e30f;
        }
        for (int i = 0; i < VOCAB; i++) if (logits[i] < threshold) logits[i] = -1e30f;
    }
    softmax(logits, VOCAB);
    float r = (float)rand() / (float)RAND_MAX, cum = 0;
    for (int i = 0; i < VOCAB; i++) { cum += logits[i]; if (cum >= r) return i; }
    return VOCAB - 1;
}

int main(int argc, char** argv) {
    const char* weights_path = argc > 1 ? argv[1] : "nanollama_final.bin";

    /* Allocate KV cache on heap */
    kv_k = (float(*)[CTX][DIM])malloc(sizeof(float) * NLAYERS * CTX * DIM);
    kv_v = (float(*)[CTX][DIM])malloc(sizeof(float) * NLAYERS * CTX * DIM);
    if (!kv_k || !kv_v) { fprintf(stderr, "kv_cache alloc failed\n"); return 1; }

    Model* model = model_new();
    if (load_weights(model, weights_path) < 0) {
        fprintf(stderr, "cannot load %s\n", weights_path);
        return 1;
    }
    fprintf(stderr, "loaded %s (89M, ctx=%d, vocab=%d)\n", weights_path, CTX, VOCAB);

    /* Read header line: n_prompt n_gen temperature top_k seed */
    int n_prompt, n_gen, top_k;
    float temperature;
    unsigned int seed;
    if (scanf("%d %d %f %d %u", &n_prompt, &n_gen, &temperature, &top_k, &seed) != 5) {
        fprintf(stderr, "header parse error\n"); return 1;
    }
    srand(seed ? seed : (unsigned)time(NULL));
    if (n_prompt + n_gen > CTX) {
        fprintf(stderr, "n_prompt+n_gen > CTX (%d)\n", CTX); return 1;
    }

    int* tokens = (int*)malloc(sizeof(int) * (n_prompt + n_gen));
    for (int i = 0; i < n_prompt; i++) {
        if (scanf("%d", &tokens[i]) != 1) {
            fprintf(stderr, "prompt token %d parse error\n", i); return 1;
        }
    }

    fprintf(stderr, "prompt: %d tokens, gen: %d, T=%.2f top_k=%d\n",
            n_prompt, n_gen, temperature, top_k);

    static float logits[VOCAB];

    /* Prefill */
    clock_t t0 = clock();
    for (int i = 0; i < n_prompt; i++) {
        forward_pos(model, tokens[i], i, logits);
    }
    double prefill_s = (double)(clock() - t0) / CLOCKS_PER_SEC;
    fprintf(stderr, "prefill %d tok in %.1fs (%.1f tok/s)\n",
            n_prompt, prefill_s, n_prompt / prefill_s);

    /* Generate */
    int pos = n_prompt;
    t0 = clock();
    for (int s = 0; s < n_gen; s++) {
        int next = sample(logits, temperature, top_k);
        tokens[pos] = next;
        printf("%d ", next);
        fflush(stdout);
        if (s + 1 < n_gen) {
            forward_pos(model, next, pos, logits);
        }
        pos++;
    }
    double gen_s = (double)(clock() - t0) / CLOCKS_PER_SEC;
    fprintf(stderr, "\ngen %d tok in %.1fs (%.2f tok/s)\n",
            n_gen, gen_s, n_gen / gen_s);

    printf("\nDONE\n");
    free(tokens);
    free(kv_k); free(kv_v);
    return 0;
}
