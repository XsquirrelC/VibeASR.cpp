// End-to-end streaming ASR: causal VAE encode + LM with cross-chunk KV retention.
//
// One round = one chunk of C latent frames of new audio. The round does:
//
//   1. vae_stream_encode_block  - encodes C new frames and emits the C+L block
//                                 (L lookahead frames replayed from the cache)
//   2. lm_stream::run_round     - appends <|speech_start|> + C+L embedding rows +
//                                 <|speech_end|> to the retained KV, decodes this
//                                 chunk's text, then appends <|text_chunk_end|>
//
// Nothing is ever recomputed: the encoder keeps per-conv left context plus the
// lookahead latents, and the LM keeps one KV cache for the whole utterance, so
// the prompt is prefilled once and every round only appends.
//
// The real-time budget is C / 7.5 s per round (2.933 s for C = 22). Per-round
// wall time over that budget is the RTF reported below; it must stay under 1 or
// latency accumulates instead of settling at C/2 + L + T.
//
// NOTE ON OUTPUT: <|text_chunk_end|> (151665) only exists in the streaming
// tokenizer's added_tokens. With a non-streaming checkpoint nothing is trained at
// that position, so it is never emitted, every round runs to --max-tokens and the
// transcript is meaningless. Until the streaming checkpoint lands, this program is
// a latency measurement and a plumbing check, not an accuracy check.

#include "vae.h"
#include "llama.h"
#include "ggml.h"

#include "../utils/audio_io.h"
#include "../utils/lm_stream.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

struct stream_params {
    std::string vae_model_path;
    std::string lm_model_path;
    std::string audio_path;

    int chunk_frames = 22;
    int lookahead    = 4;
    int max_rounds   = 0;     // 0 = until the audio runs out
    int max_tokens   = 64;
    int n_ctx        = 16384;
    int n_batch      = 512;
    int n_threads    = 8;
    int target_sr    = 24000;

    std::string show_keys = "speaker, content";
    std::string context_info;
};

static void usage(const char * prog) {
    printf("usage: %s --vae-model PATH --lm-model PATH --audio PATH [options]\n\n", prog);
    printf("options:\n");
    printf("  --chunk-frames N   new latent frames per round (default 22 = 2.933 s)\n");
    printf("  --lookahead N      lookahead frames carried by each window (default 4)\n");
    printf("  --rounds N         stop after N rounds (default 0 = whole file)\n");
    printf("  --max-tokens N     token cap per round (default 64)\n");
    printf("  --n-ctx N          LM context (default 16384)\n");
    printf("  --n-batch N        LM batch (default 512)\n");
    printf("  --show-keys S      prompt keys, must match training (default \"speaker, content\")\n");
    printf("  --context S        extra info appended to the prompt\n");
    printf("  -t, --threads N    threads (default 8)\n");
}

static bool parse_args(int argc, char ** argv, stream_params & p) {
    for (int i = 1; i < argc; i++) {
        std::string a = argv[i];
        auto next = [&]() -> std::string { return (i + 1 < argc) ? argv[++i] : ""; };

        if      (a == "--vae-model")    p.vae_model_path = next();
        else if (a == "--lm-model")     p.lm_model_path  = next();
        else if (a == "--audio")        p.audio_path     = next();
        else if (a == "--chunk-frames") p.chunk_frames   = atoi(next().c_str());
        else if (a == "--lookahead")    p.lookahead      = atoi(next().c_str());
        else if (a == "--rounds")       p.max_rounds     = atoi(next().c_str());
        else if (a == "--max-tokens")   p.max_tokens     = atoi(next().c_str());
        else if (a == "--n-ctx")        p.n_ctx          = atoi(next().c_str());
        else if (a == "--n-batch")      p.n_batch        = atoi(next().c_str());
        else if (a == "--show-keys")    p.show_keys      = next();
        else if (a == "--context")      p.context_info   = next();
        else if (a == "-t" || a == "--threads") p.n_threads = atoi(next().c_str());
        else { usage(argv[0]); return false; }
    }
    if (p.vae_model_path.empty() || p.lm_model_path.empty() || p.audio_path.empty()) {
        usage(argv[0]);
        return false;
    }
    return true;
}

static double median(std::vector<double> v) {
    if (v.empty()) return 0.0;
    std::sort(v.begin(), v.end());
    return v[v.size() / 2];
}

struct round_row {
    int    round;
    int    n_frames;
    double encode_ms;
    double prefill_ms;
    double decode_ms;
    int    n_tokens;
};

int main(int argc, char ** argv) {
    stream_params params;
    if (!parse_args(argc, argv, params)) {
        return 1;
    }

    ggml_time_init();

    const int C = params.chunk_frames;
    const int L = params.lookahead;
    const double budget_ms = 1000.0 * C / 7.5;

    printf("=== VibeASR.cpp streaming ===\n");
    printf("  chunk      : %d frames (%.3f s, budget %.0f ms)\n", C, C / 7.5, budget_ms);
    printf("  lookahead  : %d frames (%.3f s)\n", L, L / 7.5);
    printf("  threads    : %d\n", params.n_threads);

    // ---- audio ----
    audio_io::AudioData a;
    if (!audio_io::load_audio(params.audio_path, params.target_sr, false, a)) {
        fprintf(stderr, "Error: failed to load audio: %s\n", params.audio_path.c_str());
        return 1;
    }
    const std::vector<float> & audio = a.samples;
    printf("  audio      : %s (%.2f s)\n\n", params.audio_path.c_str(), a.duration_sec);

    // ---- VAE ----
    struct vae_model_params vae_mparams = vae_model_default_params();
    vae_mparams.n_threads = params.n_threads;
    vae_model_t * vae_model = vae_load_model_from_file(params.vae_model_path.c_str(), vae_mparams);
    if (!vae_model) {
        fprintf(stderr, "Error: failed to load VAE model\n");
        return 1;
    }

    struct vae_context_params vae_cparams = vae_context_default_params();
    vae_cparams.n_threads = params.n_threads;
    vae_stream_t * vs = vae_stream_init(vae_model, vae_cparams);
    if (!vs) {
        fprintf(stderr, "Error: failed to create VAE streaming state\n");
        vae_free_model(vae_model);
        return 1;
    }
    vae_stream_set_lookahead(vs, L);

    const int acoustic_dim = vae_model_acoustic_dim(vae_model);
    const int semantic_dim = vae_model_semantic_dim(vae_model);

    // ---- LM ----
    llama_backend_init();
    llama_numa_init(GGML_NUMA_STRATEGY_DISABLED);
    llama_log_set([](enum ggml_log_level, const char *, void *){}, nullptr);

    llama_model_params lm_mparams = llama_model_default_params();
    lm_mparams.n_gpu_layers = 0;
    llama_model * lm_model = llama_load_model_from_file(params.lm_model_path.c_str(), lm_mparams);
    if (!lm_model) {
        fprintf(stderr, "Error: failed to load LM model\n");
        vae_stream_free(vs);
        vae_free_model(vae_model);
        return 1;
    }

    llama_context_params lm_cparams = llama_context_default_params();
    lm_cparams.n_ctx           = params.n_ctx;
    lm_cparams.n_batch         = params.n_batch;
    lm_cparams.n_ubatch        = params.n_batch;
    lm_cparams.n_threads       = params.n_threads;
    lm_cparams.n_threads_batch = params.n_threads;

    llama_context * lm_ctx = llama_new_context_with_model(lm_model, lm_cparams);
    if (!lm_ctx) {
        fprintf(stderr, "Error: failed to create LM context\n");
        llama_free_model(lm_model);
        vae_stream_free(vs);
        vae_free_model(vae_model);
        return 1;
    }

    lm_stream::params lmp;
    lmp.n_batch              = params.n_batch;
    lmp.max_tokens_per_chunk = params.max_tokens;
    lmp.show_keys            = params.show_keys;
    lmp.context_info         = params.context_info;

    lm_stream::session sess;
    if (!lm_stream::init(sess, lm_model, lm_ctx, lmp)) {
        fprintf(stderr, "Error: failed to init the streaming LM session\n");
        llama_free(lm_ctx);
        llama_free_model(lm_model);
        vae_stream_free(vs);
        vae_free_model(vae_model);
        return 1;
    }
    printf("  prompt     : %d tokens prefilled once (KV retained across rounds)\n\n",
           sess.n_prompt);

    // ---- rounds ----
    std::vector<float> af((size_t)(C + L) * acoustic_dim);
    std::vector<float> sf((size_t)(C + L) * semantic_dim);
    std::vector<round_row> rows;
    std::string transcript;

    size_t off = 0;   // audio cursor in samples
    int round = 0;
    while (true) {
        // Round 0 has nothing retained, so it must consume C + L frames of audio;
        // every later round consumes C and replays the L cached latents.
        const int want_frames  = (round == 0) ? (C + L) : C;
        const size_t want = (size_t) want_frames * VAE_STREAM_COMPRESS_RATIO;
        if (off + want > audio.size()) {
            break;   // a partial chunk would break the stride alignment
        }
        if (params.max_rounds > 0 && round >= params.max_rounds) {
            break;
        }

        float enc_ms = 0.0f;
        const int32_t n_block = vae_stream_encode_block(
            vs, audio.data() + off, (int32_t) want, af.data(), sf.data(), &enc_ms);
        if (n_block < 0) {
            fprintf(stderr, "Error: streaming encode failed at round %d\n", round);
            break;
        }
        off += want;

        std::string text;
        lm_stream::round_stats st;
        printf("[%2d] ", round);
        fflush(stdout);
        const int n_tok = lm_stream::run_round(
            sess, af.data(), acoustic_dim, sf.data(), semantic_dim, n_block,
            text, &st,
            [](const std::string & piece) { fputs(piece.c_str(), stdout); fflush(stdout); });
        if (n_tok < 0) {
            fprintf(stderr, "\nError: LM round %d failed (context full?)\n", round);
            break;
        }
        transcript += text;

        const double total = enc_ms + st.prefill_ms + st.decode_ms;
        printf("\n     enc %.0f  prefill %.0f  decode %.0f (%d tok)  total %.0f ms  RTF %.3f%s\n",
               enc_ms, st.prefill_ms, st.decode_ms, n_tok, total, total / budget_ms,
               st.hit_cap ? "  [token cap]" : (st.hit_eos ? "  [eos]" : ""));

        rows.push_back({ round, n_block, enc_ms, st.prefill_ms, st.decode_ms, n_tok });
        round++;
    }

    // ---- report ----
    if (!rows.empty()) {
        std::vector<double> enc, pre, dec, tot;
        double sum_tot = 0.0;
        int sum_tok = 0;
        for (const auto & r : rows) {
            const double t = r.encode_ms + r.prefill_ms + r.decode_ms;
            enc.push_back(r.encode_ms);
            pre.push_back(r.prefill_ms);
            dec.push_back(r.decode_ms);
            tot.push_back(t);
            sum_tot += t;
            sum_tok += r.n_tokens;
        }

        const double med_tot = median(tot);
        const double max_tot = *std::max_element(tot.begin(), tot.end());
        const double audio_s = (double) rows.size() * C / 7.5;

        printf("\n=== per-round latency over %zu rounds (ms) ===\n", rows.size());
        printf("  encode  : median %.0f\n", median(enc));
        printf("  prefill : median %.0f  (%d speech rows + 2 specials)\n",
               median(pre), C + L);
        printf("  decode  : median %.0f  (%.1f tok/round)\n",
               median(dec), (double) sum_tok / rows.size());
        printf("  total   : median %.0f   max %.0f   budget %.0f\n",
               med_tot, max_tot, budget_ms);
        printf("\n  RTF (median) : %.3f\n", med_tot / budget_ms);
        printf("  RTF (worst)  : %.3f\n", max_tot / budget_ms);
        printf("  RTF (overall): %.3f   (%.2f s of compute for %.2f s of audio)\n",
               sum_tot / (audio_s * 1000.0), sum_tot / 1000.0, audio_s);
        printf("  latency      : %.2f s average, %.2f s worst-case first packet\n",
               C / 15.0 + L / 7.5 + med_tot / 1000.0,
               (C + L) / 7.5 + med_tot / 1000.0);

        printf("\n=== caches ===\n");
        printf("  conv cache      : %.1f KiB\n", vae_stream_cache_bytes(vs) / 1024.0);
        printf("  lookahead cache : %.1f KiB (%d frames x (%d + %d) floats)\n",
               vae_stream_lookahead_bytes(vs) / 1024.0, L, acoustic_dim, semantic_dim);
        printf("  LM KV           : %d tokens (prompt %d + %zu rounds)\n",
               sess.n_past, sess.n_prompt, rows.size());

        printf("\n=== transcript ===\n%s\n", transcript.c_str());
    }

    lm_stream::free_session(sess);
    llama_free(lm_ctx);
    llama_free_model(lm_model);
    llama_backend_free();
    vae_stream_free(vs);
    vae_free_model(vae_model);
    return 0;
}
