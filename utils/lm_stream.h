/**
 * VibeASR.cpp - streaming LM session (cross-chunk KV retention)
 *
 * The streaming checkpoint is trained on ONE interleaved sequence per utterance:
 *
 *   prompt  X_1 Y_1 <|text_chunk_end|>  X_2 Y_2 <|text_chunk_end|>  ...  [eos]
 *
 * where X_k = <|speech_start|> + (C+L) latent frames + <|speech_end|> and Y_k is
 * chunk k's transcript. See podcast_asr_streaming_loader.py:1357-1375 in the
 * training repo - the audio window carries L lookahead frames, but the text
 * boundary advances by only C, so consecutive X_k overlap by L frames.
 *
 * Inference must therefore keep a single KV cache alive for the whole utterance
 * and only ever append to it: the prompt is prefilled once, and each round
 * appends X_k, decodes Y_k, then appends <|text_chunk_end|> even if generation
 * was cut short - otherwise round k+1 sees a sequence the model never saw in
 * training. Clearing the cache per chunk (what the offline path does) would both
 * throw away all cross-chunk context and re-pay the prompt prefill every round.
 *
 * The prompt is plain text with NO chat template: the streaming layout has no
 * role tags at all, unlike the offline prompt in prompt_builder.h.
 *
 * Reference implementation: VibeVoice-ASR-Streaming's
 * modeling_vibevoice_asr.py:429 streaming_generate().
 */

#ifndef LM_STREAM_H
#define LM_STREAM_H

#include "llama.h"
#include "prompt_builder.h"

#include "time_compat.h"

#include <algorithm>
#include <cstdio>
#include <ctime>
#include <string>
#include <vector>

namespace lm_stream {

static double now_ms() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000.0 + ts.tv_nsec / 1e6;
}

// Canonical HuggingFace token IDs. The GGUF vocab's text->ID mapping can differ
// from the training tokenizer's, but embedding table positions always match, so
// IDs are used directly (same convention as prompt_builder.h).
//
// <|text_chunk_end|> is NOT in the Qwen2.5 base vocabulary - the streaming
// tokenizer appends it, landing at 151665 (see the streaming checkpoint's
// added_tokens.json). A non-streaming checkpoint has nothing trained at that
// position, so it will never be emitted and every round runs to max_tokens.
static const llama_token SPEECH_START_ID   = 151646;  // <|object_ref_start|>
static const llama_token SPEECH_END_ID     = 151647;  // <|object_ref_end|>
static const llama_token TEXT_CHUNK_END_ID = 151665;  // <|text_chunk_end|>
static const llama_token EOS_ID            = 151643;  // <|endoftext|>
static const llama_token IM_END_ID         = 151645;  // <|im_end|>

struct params {
    int   n_batch              = 2048;
    int   max_tokens_per_chunk = 256;
    bool  greedy               = true;
    float temperature          = 0.7f;
    float top_p                = 0.9f;
    int   seed                 = 42;

    llama_token text_chunk_end_id = TEXT_CHUNK_END_ID;

    // Keys the checkpoint was trained to emit, and optional hotwords. Both go
    // into the prompt verbatim, so they must match the training config.
    std::string show_keys    = "speaker, content";
    std::string context_info;
};

struct round_stats {
    double prefill_ms = 0.0;   // X_k: speech_start + frames + speech_end
    double decode_ms  = 0.0;   // Y_k + the trailing <|text_chunk_end|>
    int    n_tokens   = 0;     // text tokens in Y_k (specials excluded)
    bool   hit_cap    = false; // stopped on max_tokens instead of a chunk end
    bool   hit_eos    = false;
};

struct session {
    llama_model   * model = nullptr;
    llama_context * ctx   = nullptr;
    llama_sampler * smpl  = nullptr;

    params cfg;

    int n_embd  = 0;
    int n_ctx   = 0;
    int n_past  = 0;   // KV length; monotonically increasing within an utterance
    int n_prompt = 0;  // tokens the prompt occupies, so reset() can report it
    int n_rounds = 0;

    std::vector<llama_token> prompt_tokens;
};

// ---------------------------------------------------------------- low level --

// Decode a run of token IDs. Logits are produced for the last token only when
// want_logits is set (llama_batch_get_one always asks for the last one, so the
// no-logits case has to build the batch by hand).
static bool decode_tokens(
    llama_context * ctx,
    const llama_token * tokens,
    int n_tokens,
    int & pos,
    int n_batch,
    bool want_logits) {

    int done = 0;
    while (done < n_tokens) {
        const int n = std::min(n_batch, n_tokens - done);
        const bool last_batch = (done + n == n_tokens);

        llama_batch batch = llama_batch_init(n, 0, 1);
        batch.n_tokens = n;
        for (int i = 0; i < n; i++) {
            batch.token[i]    = tokens[done + i];
            batch.pos[i]      = pos + i;
            batch.n_seq_id[i] = 1;
            batch.seq_id[i][0] = 0;
            batch.logits[i]   = (want_logits && last_batch && i == n - 1) ? 1 : 0;
        }
        const int rc = llama_decode(ctx, batch);
        llama_batch_free(batch);
        if (rc != 0) {
            fprintf(stderr, "[lm_stream] token decode failed at pos %d\n", pos);
            return false;
        }
        done += n;
        pos  += n;
    }
    return true;
}

// Decode a run of raw embedding rows (the VAE speech features).
static bool decode_embeddings(
    llama_context * ctx,
    const float * embd,
    int n_embd,
    int n_rows,
    int & pos,
    int n_batch,
    bool want_logits) {

    int done = 0;
    while (done < n_rows) {
        const int n = std::min(n_batch, n_rows - done);
        const bool last_batch = (done + n == n_rows);

        llama_batch batch = llama_batch_init(n, n_embd, 1);
        batch.n_tokens = n;
        batch.token    = nullptr;  // embedding mode

        float * owned = batch.embd;
        batch.embd = const_cast<float *>(embd) + (size_t) done * n_embd;

        for (int i = 0; i < n; i++) {
            batch.pos[i]      = pos + i;
            batch.n_seq_id[i] = 1;
            batch.seq_id[i][0] = 0;
            batch.logits[i]   = (want_logits && last_batch && i == n - 1) ? 1 : 0;
        }
        const int rc = llama_decode(ctx, batch);
        batch.embd = owned;
        llama_batch_free(batch);
        if (rc != 0) {
            fprintf(stderr, "[lm_stream] embedding decode failed at pos %d\n", pos);
            return false;
        }
        done += n;
        pos  += n;
    }
    return true;
}

// -------------------------------------------------------------------- prompt --

// Verbatim copy of the training prompt (podcast_asr_streaming_loader.py:796-798),
// tokenized with no special-token parsing and no chat template.
static std::string build_prompt_text(const params & cfg) {
    std::string s =
        "You are a helpful assistant that transcribes audio input into text output. "
        "Please transcribe the following audios streamingly with these keys: " + cfg.show_keys;
    if (!cfg.context_info.empty()) {
        s += " and extra info: " + cfg.context_info;
    }
    s += "\n";
    return s;
}

// --------------------------------------------------------------- session API --

static void free_session(session & s) {
    if (s.smpl) {
        llama_sampler_free(s.smpl);
        s.smpl = nullptr;
    }
}

// Clear the KV cache and re-prefill the prompt, i.e. start a new utterance.
// Returns false if the prompt could not be prefilled.
static bool reset(session & s) {
    llama_kv_cache_clear(s.ctx);
    s.n_past   = 0;
    s.n_rounds = 0;

    if (!decode_tokens(s.ctx, s.prompt_tokens.data(), (int) s.prompt_tokens.size(),
                       s.n_past, s.cfg.n_batch, /*want_logits =*/ false)) {
        return false;
    }
    s.n_prompt = s.n_past;
    return true;
}

static bool init(session & s, llama_model * model, llama_context * ctx, const params & cfg) {
    s.model  = model;
    s.ctx    = ctx;
    s.cfg    = cfg;
    s.n_embd = llama_n_embd(model);
    s.n_ctx  = (int) llama_n_ctx(ctx);

    const std::string prompt_text = build_prompt_text(cfg);
    s.prompt_tokens = prompt_builder::tokenize(model, prompt_text, false, false);
    if (s.prompt_tokens.empty()) {
        fprintf(stderr, "[lm_stream] failed to tokenize the streaming prompt\n");
        return false;
    }

    s.smpl = llama_sampler_chain_init(llama_sampler_chain_default_params());
    if (cfg.greedy) {
        llama_sampler_chain_add(s.smpl, llama_sampler_init_greedy());
    } else {
        llama_sampler_chain_add(s.smpl, llama_sampler_init_top_k(40));
        llama_sampler_chain_add(s.smpl, llama_sampler_init_top_p(cfg.top_p, 1));
        llama_sampler_chain_add(s.smpl, llama_sampler_init_temp(cfg.temperature));
        llama_sampler_chain_add(s.smpl, llama_sampler_init_dist(cfg.seed));
    }

    if (!reset(s)) {
        free_session(s);
        return false;
    }
    return true;
}

// Upper bound on the KV this round needs: speech_start + frames + speech_end +
// the generated text + text_chunk_end.
static int round_kv_budget(const session & s, int n_frames) {
    return 3 + n_frames + s.cfg.max_tokens_per_chunk;
}

// Run one streaming round on top of the retained KV cache.
//
// acoustic/semantic hold n_frames rows each (C + L frames: the lookahead frames
// replayed from the VAE cache followed by the newly encoded ones).
//
// on_piece, if given, is called with each detokenized piece as it is produced,
// for token-level output. Returns the number of text tokens, or -1 on error.
template <typename PieceFn>
static int run_round(
    session & s,
    const float * acoustic, int acoustic_dim,
    const float * semantic, int semantic_dim,
    int n_frames,
    std::string & out_text,
    round_stats * stats,
    PieceFn on_piece) {

    out_text.clear();
    round_stats st;

    if (s.n_past + round_kv_budget(s, n_frames) > s.n_ctx) {
        // Sliding the window would need a position shift over the whole cache;
        // until that exists, the caller has to start a new utterance.
        fprintf(stderr, "[lm_stream] context full: n_past=%d, need up to %d, n_ctx=%d\n",
                s.n_past, round_kv_budget(s, n_frames), s.n_ctx);
        return -1;
    }

    const double t0 = now_ms();

    // --- X_k: <|speech_start|> + frames + <|speech_end|> ---
    llama_token sp_start = SPEECH_START_ID;
    if (!decode_tokens(s.ctx, &sp_start, 1, s.n_past, s.cfg.n_batch, false)) return -1;

    std::vector<float> speech_emb = prompt_builder::build_speech_embeddings(
        s.n_embd, acoustic, acoustic_dim, semantic, semantic_dim, n_frames);
    if (!decode_embeddings(s.ctx, speech_emb.data(), s.n_embd, n_frames,
                           s.n_past, s.cfg.n_batch, false)) {
        return -1;
    }

    llama_token sp_end = SPEECH_END_ID;
    if (!decode_tokens(s.ctx, &sp_end, 1, s.n_past, s.cfg.n_batch, /*want_logits =*/ true)) {
        return -1;
    }

    const double t1 = now_ms();
    st.prefill_ms = t1 - t0;

    // --- Y_k: decode until <|text_chunk_end|> ---
    std::vector<llama_token> text_tokens;
    while ((int) text_tokens.size() < s.cfg.max_tokens_per_chunk) {
        llama_token tok = llama_sampler_sample(s.smpl, s.ctx, -1);

        if (tok == s.cfg.text_chunk_end_id) {
            break;
        }
        if (tok == EOS_ID || tok == IM_END_ID) {
            st.hit_eos = true;
            break;
        }

        llama_sampler_accept(s.smpl, tok);
        text_tokens.push_back(tok);

        const std::string piece = prompt_builder::token_to_piece(s.model, tok, false);
        if (!piece.empty()) {
            out_text += piece;
            on_piece(piece);
        }

        // Feed the accepted token back so the next sample sees it.
        if (!decode_tokens(s.ctx, &tok, 1, s.n_past, s.cfg.n_batch, /*want_logits =*/ true)) {
            return -1;
        }
    }
    if ((int) text_tokens.size() >= s.cfg.max_tokens_per_chunk) {
        st.hit_cap = true;
    }

    // --- <|text_chunk_end|> goes into the KV unconditionally ---
    // Training always has it between Y_k and X_{k+1}, so it is appended even
    // when the loop stopped on eos or the token cap.
    llama_token tce = s.cfg.text_chunk_end_id;
    if (!decode_tokens(s.ctx, &tce, 1, s.n_past, s.cfg.n_batch, false)) return -1;

    st.decode_ms = now_ms() - t1;
    st.n_tokens  = (int) text_tokens.size();
    if (stats) *stats = st;

    s.n_rounds++;
    return st.n_tokens;
}

// Overload without a per-token callback.
static int run_round(
    session & s,
    const float * acoustic, int acoustic_dim,
    const float * semantic, int semantic_dim,
    int n_frames,
    std::string & out_text,
    round_stats * stats = nullptr) {

    return run_round(s, acoustic, acoustic_dim, semantic, semantic_dim, n_frames,
                     out_text, stats, [](const std::string &) {});
}

}  // namespace lm_stream

#endif // LM_STREAM_H
