/*
 * VibeASR.cpp - Chunk-level latency baseline for streaming ASR
 *
 * Measures the per-chunk cost of the streaming pipeline against the real-time
 * budget, so we know how much headroom exists before implementing the encoder
 * streaming cache.
 *
 * Budget: one chunk of C latent frames at 7.5 fps must be processed within
 * C / 7.5 seconds. For C = 22 that is 2.933 s.
 *
 *   - without an encoder cache, each chunk re-encodes C + L frames
 *   - with an encoder cache, the lookahead frames are reused, so each chunk
 *     encodes only C new frames
 *
 * Both are reported so the cache's payoff is visible.
 */

#include "vae.h"
#include "llama.h"
#include "ggml.h"

#include "../utils/audio_io.h"
#include "../utils/prompt_builder.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <string>
#include <vector>

struct bench_params {
    std::string vae_model_path;
    std::string lm_model_path;
    std::string audio_path;          // optional; synthetic noise if empty
    int  chunk_frames   = 22;
    int  lookahead      = 4;
    int  n_decode       = 30;        // tokens generated per chunk
    int  kv_prefix      = 0;         // pre-filled KV tokens to emulate history
    int  repeat         = 3;
    int  n_threads      = 4;
    int  n_ctx          = 8192;
    int  n_batch        = 512;
    int  compress_ratio = 3200;
    int  target_sr      = 24000;
};

static void print_usage(const char * prog) {
    fprintf(stderr, "usage: %s --vae-model PATH --lm-model PATH [options]\n\n", prog);
    fprintf(stderr, "options:\n");
    fprintf(stderr, "  --audio PATH        real audio (default: synthetic noise)\n");
    fprintf(stderr, "  --chunk-frames N    latent frames per chunk (default 22)\n");
    fprintf(stderr, "  --lookahead N       lookahead frames (default 4)\n");
    fprintf(stderr, "  --n-decode N        tokens decoded per chunk (default 30)\n");
    fprintf(stderr, "  --kv-prefix N       pre-fill N KV tokens to emulate history (default 0)\n");
    fprintf(stderr, "  --repeat N          repeats per measurement, median kept (default 3)\n");
    fprintf(stderr, "  -t, --threads N     threads (default 4)\n");
}

static bool parse_args(int argc, char ** argv, bench_params & p) {
    for (int i = 1; i < argc; i++) {
        std::string a = argv[i];
        auto next = [&](void) -> std::string { return (i + 1 < argc) ? argv[++i] : ""; };

        if (a == "--vae-model")          p.vae_model_path = next();
        else if (a == "--lm-model")      p.lm_model_path  = next();
        else if (a == "--audio")         p.audio_path     = next();
        else if (a == "--chunk-frames")  p.chunk_frames   = std::stoi(next());
        else if (a == "--lookahead")     p.lookahead      = std::stoi(next());
        else if (a == "--n-decode")      p.n_decode       = std::stoi(next());
        else if (a == "--kv-prefix")     p.kv_prefix      = std::stoi(next());
        else if (a == "--repeat")        p.repeat         = std::stoi(next());
        else if (a == "-t" || a == "--threads") p.n_threads = std::stoi(next());
        else if (a == "-h" || a == "--help") { print_usage(argv[0]); return false; }
        else {
            fprintf(stderr, "unknown argument: %s\n", a.c_str());
            return false;
        }
    }
    if (p.vae_model_path.empty() || p.lm_model_path.empty()) {
        print_usage(argv[0]);
        return false;
    }
    return true;
}

static double get_time_ms(void) {
    return (double)ggml_time_us() / 1000.0;
}

static double median(std::vector<double> v) {
    std::sort(v.begin(), v.end());
    return v[v.size() / 2];
}

// Encode n_frames worth of audio and return the median wall time in ms.
static double bench_encode(
        vae_context_t * ctx,
        vae_model_t * model,
        const std::vector<float> & audio,
        int n_frames,
        int compress_ratio,
        int repeat,
        bool semantic) {

    const int32_t n_samples = n_frames * compress_ratio;
    if ((size_t)n_samples > audio.size()) {
        return -1.0;
    }

    const int dim = semantic ? vae_model_semantic_dim(model) : vae_model_acoustic_dim(model);
    std::vector<float> out((size_t)(n_frames + 8) * dim);

    std::vector<double> times;
    for (int r = 0; r < repeat; r++) {
        const double t0 = get_time_ms();
        const int32_t got = semantic
            ? vae_encode_semantic(ctx, audio.data(), n_samples, out.data())
            : vae_encode_acoustic(ctx, audio.data(), n_samples, out.data());
        const double dt = get_time_ms() - t0;
        if (got < 0) {
            return -1.0;
        }
        times.push_back(dt);
    }
    return median(times);
}

int main(int argc, char ** argv) {
    bench_params params;
    if (!parse_args(argc, argv, params)) {
        return 1;
    }

    ggml_time_init();

    const double chunk_budget_ms = 1000.0 * params.chunk_frames / 7.5;

    fprintf(stderr, "========================================\n");
    fprintf(stderr, " VibeASR.cpp - Chunk Latency Baseline\n");
    fprintf(stderr, "========================================\n\n");
    fprintf(stderr, "  chunk       : %d frames (%.3f s, budget %.0f ms)\n",
            params.chunk_frames, chunk_budget_ms / 1000.0, chunk_budget_ms);
    fprintf(stderr, "  lookahead   : %d frames\n", params.lookahead);
    fprintf(stderr, "  threads     : %d\n", params.n_threads);
    fprintf(stderr, "  repeat      : %d (median)\n\n", params.repeat);

    // ---- audio ----
    std::vector<float> audio;
    const int max_frames = std::max(225, params.chunk_frames + params.lookahead);

    if (!params.audio_path.empty()) {
        audio_io::AudioData a;
        if (!audio_io::load_audio(params.audio_path, params.target_sr, false, a)) {
            fprintf(stderr, "Error: failed to load audio: %s\n", params.audio_path.c_str());
            return 1;
        }
        audio = a.samples;
        fprintf(stderr, "[Audio] %s: %.2f s, %zu samples\n\n", params.audio_path.c_str(),
                a.duration_sec, audio.size());
    } else {
        // Synthetic noise is fine for timing: the graph shape depends on length only.
        std::mt19937 rng(42);
        std::normal_distribution<float> dist(0.0f, 0.1f);
        audio.resize((size_t)max_frames * params.compress_ratio);
        for (auto & s : audio) {
            s = dist(rng);
        }
        fprintf(stderr, "[Audio] synthetic noise: %.2f s, %zu samples\n\n",
                (double)audio.size() / params.target_sr, audio.size());
    }

    // ---- load models ----
    fprintf(stderr, "[Init] loading VAE...\n");
    struct vae_model_params vae_mparams = vae_model_default_params();
    vae_mparams.n_threads = params.n_threads;

    vae_model_t * vae_model = vae_load_model_from_file(params.vae_model_path.c_str(), vae_mparams);
    if (!vae_model) {
        fprintf(stderr, "Error: failed to load VAE model\n");
        return 1;
    }

    struct vae_context_params vae_cparams = vae_context_default_params();
    vae_cparams.n_threads = params.n_threads;

    vae_context_t * vae_ctx = vae_new_context_with_model(vae_model, vae_cparams);
    if (!vae_ctx) {
        fprintf(stderr, "Error: failed to create VAE context\n");
        vae_free_model(vae_model);
        return 1;
    }

    fprintf(stderr, "[Init] loading LM...\n");
    llama_backend_init();
    llama_numa_init(GGML_NUMA_STRATEGY_DISABLED);
    llama_log_set([](enum ggml_log_level, const char *, void *){}, nullptr);

    llama_model_params lm_mparams = llama_model_default_params();
    lm_mparams.n_gpu_layers = 0;

    llama_model * lm_model = llama_load_model_from_file(params.lm_model_path.c_str(), lm_mparams);
    if (!lm_model) {
        fprintf(stderr, "Error: failed to load LM model\n");
        vae_free(vae_ctx);
        vae_free_model(vae_model);
        return 1;
    }

    llama_context_params lm_cparams = llama_context_default_params();
    lm_cparams.n_ctx           = params.n_ctx;
    lm_cparams.n_batch         = params.n_batch;
    lm_cparams.n_threads       = params.n_threads;
    lm_cparams.n_threads_batch = params.n_threads;

    llama_context * lm_ctx = llama_new_context_with_model(lm_model, lm_cparams);
    if (!lm_ctx) {
        fprintf(stderr, "Error: failed to create LM context\n");
        llama_free_model(lm_model);
        vae_free(vae_ctx);
        vae_free_model(vae_model);
        return 1;
    }
    fprintf(stderr, "[Init] done\n\n");

    // ---- VAE encode ----
    const int chunk       = params.chunk_frames;
    const int chunk_plus  = params.chunk_frames + params.lookahead;
    const int full_frames = std::min(max_frames, (int)(audio.size() / params.compress_ratio));

    struct enc_row { const char * label; int frames; double acoustic; double semantic; };
    std::vector<enc_row> rows = {
        { "chunk (cached lookahead)", chunk,       0, 0 },
        { "chunk + lookahead",        chunk_plus,  0, 0 },
        { "full utterance",           full_frames, 0, 0 },
    };

    fprintf(stderr, "[VAE] encoding...\n");
    for (auto & r : rows) {
        r.acoustic = bench_encode(vae_ctx, vae_model, audio, r.frames,
                                  params.compress_ratio, params.repeat, false);
        r.semantic = bench_encode(vae_ctx, vae_model, audio, r.frames,
                                  params.compress_ratio, params.repeat, true);
    }

    // ---- LM prefill + decode ----
    // Emulate streaming history by pre-filling the KV cache, then measure the
    // prefill of one chunk's speech embeddings plus n_decode autoregressive steps.
    fprintf(stderr, "[LM] prefill + decode...\n");

    const int32_t chunk_samples = chunk * params.compress_ratio;
    prompt_builder::PromptTokens prompt = prompt_builder::build_prompt(
        lm_model, chunk_samples, params.compress_ratio,
        (double)chunk_samples / params.target_sr);

    if (prompt.tokens.empty()) {
        fprintf(stderr, "Error: failed to build prompt\n");
        return 1;
    }

    const int acoustic_dim = vae_model_acoustic_dim(vae_model);
    const int semantic_dim = vae_model_semantic_dim(vae_model);
    std::vector<float> af((size_t)(chunk + 8) * acoustic_dim, 0.0f);
    std::vector<float> sf((size_t)(chunk + 8) * semantic_dim, 0.0f);
    vae_encode_acoustic(vae_ctx, audio.data(), chunk_samples, af.data());
    vae_encode_semantic(vae_ctx, audio.data(), chunk_samples, sf.data());

    llama_sampler * smpl = llama_sampler_chain_init(llama_sampler_chain_default_params());
    llama_sampler_chain_add(smpl, llama_sampler_init_greedy());

    std::vector<double> prefill_times, decode_times;
    for (int r = 0; r < params.repeat; r++) {
        llama_kv_cache_clear(lm_ctx);

        // Optional synthetic history.
        int pos = 0;
        if (params.kv_prefix > 0) {
            std::vector<llama_token> filler(params.kv_prefix, prompt.tokens[0]);
            for (int off = 0; off < params.kv_prefix; off += params.n_batch) {
                const int n = std::min(params.n_batch, params.kv_prefix - off);
                llama_batch b = llama_batch_get_one(filler.data() + off, n, pos, 0);
                if (llama_decode(lm_ctx, b) != 0) {
                    fprintf(stderr, "Error: KV prefix decode failed\n");
                    return 1;
                }
                pos += n;
            }
        }

        const double t0 = get_time_ms();
        const int end_pos = prompt_builder::prefill_segmented(
            lm_model, lm_ctx, prompt,
            af.data(), acoustic_dim,
            sf.data(), semantic_dim,
            chunk, params.n_batch);
        prefill_times.push_back(get_time_ms() - t0);

        if (end_pos < 0) {
            fprintf(stderr, "Error: prefill failed\n");
            return 1;
        }

        const double t1 = get_time_ms();
        int cur = end_pos;
        for (int i = 0; i < params.n_decode; i++) {
            llama_token tok = llama_sampler_sample(smpl, lm_ctx, -1);
            llama_sampler_accept(smpl, tok);
            llama_batch b = llama_batch_get_one(&tok, 1, cur++, 0);
            if (llama_decode(lm_ctx, b) != 0) {
                break;
            }
        }
        decode_times.push_back(get_time_ms() - t1);
    }

    const double prefill_ms = median(prefill_times);
    const double decode_ms  = median(decode_times);

    // ---- report ----
    printf("\n");
    printf("=== VAE encode (median of %d, ms) ===\n", params.repeat);
    printf("%-28s %8s %10s %10s %10s\n", "input", "frames", "acoustic", "semantic", "sum");
    for (const auto & r : rows) {
        if (r.acoustic < 0 || r.semantic < 0) {
            printf("%-28s %8d %10s %10s %10s\n", r.label, r.frames, "n/a", "n/a", "n/a");
            continue;
        }
        printf("%-28s %8d %10.1f %10.1f %10.1f\n",
               r.label, r.frames, r.acoustic, r.semantic, r.acoustic + r.semantic);
    }

    printf("\n=== LM (median of %d, ms) ===\n", params.repeat);
    printf("  KV prefix        : %d tokens\n", params.kv_prefix);
    printf("  prefill (%d frm)  : %.1f\n", chunk, prefill_ms);
    printf("  decode (%d tok)   : %.1f   (%.1f ms/tok)\n",
           params.n_decode, decode_ms, decode_ms / std::max(1, params.n_decode));

    const double vae_cached   = rows[0].acoustic + rows[0].semantic;
    const double vae_nocache  = rows[1].acoustic + rows[1].semantic;
    const double lm_total     = prefill_ms + decode_ms;

    printf("\n=== per-chunk total vs %.0f ms budget ===\n", chunk_budget_ms);
    printf("%-34s %10s %8s\n", "configuration", "total ms", "RTF");
    if (vae_cached > 0) {
        printf("%-34s %10.1f %8.3f\n", "with encoder cache (C frames)",
               vae_cached + lm_total, (vae_cached + lm_total) / chunk_budget_ms);
    }
    if (vae_nocache > 0) {
        printf("%-34s %10.1f %8.3f\n", "no cache (C+L frames)",
               vae_nocache + lm_total, (vae_nocache + lm_total) / chunk_budget_ms);
    }
    printf("\nRTF < 1.0 required; leave headroom for IO and scheduling jitter.\n");
    printf("Note: prefill cost grows with KV length, so re-run with --kv-prefix\n");
    printf("      set to a realistic steady-state history (e.g. 2000) as well.\n");

    llama_sampler_free(smpl);
    llama_free(lm_ctx);
    llama_free_model(lm_model);
    llama_backend_free();
    vae_free(vae_ctx);
    vae_free_model(vae_model);
    return 0;
}
