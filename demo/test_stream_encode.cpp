// Streaming-vs-batch equivalence check for the VAE encoder.
//
// The encoder is causal, so feeding an utterance one chunk at a time through
// vae_stream_encode must reproduce what vae_encode_* produces for the whole
// utterance in one shot.
//
// F32: exactly reproduces it. max_diff must be 0, or within f32 rounding of a
// reordered reduction - a per-layer cache that is one frame short or misaligned
// shows up as a large diff concentrated at the chunk boundaries, which is why
// this reports the worst frame and its index rather than just a mean.
//
// I8_S: cannot reproduce it bit-exactly. Every I8_S op requantizes its output
// with a per-tensor dynamic absmax, so a 22-frame chunk and a 66-frame utterance
// pick different scales at every layer and the int8 grids do not line up. The
// meaningful question is whether streaming adds error on top of what the I8_S
// quantization already costs, so pass --ref-vae-model with the F32 encoder and
// this compares both I8_S results against that common reference: if
// cos(F32, stream) is on par with cos(F32, batch), streaming is as accurate as
// batch and the diff between them is quantization noise, not a cache bug.

#include "vae.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

static void usage(const char * prog) {
    printf("usage: %s --vae-model PATH [options]\n\n", prog);
    printf("options:\n");
    printf("  --vae-model PATH      encoder under test (F32 or I8_S)\n");
    printf("  --ref-vae-model PATH  F32 encoder to use as the accuracy reference\n");
    printf("  --frames N            total latent frames to encode (default 66)\n");
    printf("  --chunk-frames N      frames per streaming chunk (default 22)\n");
    printf("  --lookahead N         lookahead overlap in frames; exercises the\n");
    printf("                        lookahead embedding cache (default 0 = off)\n");
    printf("  -t, --threads N       threads (default 4)\n");
    printf("  --seed N              RNG seed for the synthetic audio (default 1234)\n");
}

struct stats {
    double max_abs;
    int    worst_frame;
    double rms;
    double absmax;
    double cos;
    double min_frame_cos;
};

static stats compare(const float * a, const float * b, int n_frames, int dim) {
    stats s = { 0.0, -1, 0.0, 0.0, 0.0, 1.0 };
    double sum_sq = 0.0, dot = 0.0, na = 0.0, nb = 0.0;
    for (int f = 0; f < n_frames; f++) {
        double frame_max = 0.0, fdot = 0.0, fna = 0.0, fnb = 0.0;
        for (int d = 0; d < dim; d++) {
            const size_t k = (size_t) f * dim + d;
            const double x = a[k], y = b[k];
            const double diff = fabs(x - y);
            if (diff > frame_max) frame_max = diff;
            sum_sq += diff * diff;
            if (fabs(x) > s.absmax) s.absmax = fabs(x);
            fdot += x * y; fna += x * x; fnb += y * y;
        }
        dot += fdot; na += fna; nb += fnb;
        if (frame_max > s.max_abs) { s.max_abs = frame_max; s.worst_frame = f; }
        if (fna > 0.0 && fnb > 0.0) {
            const double c = fdot / sqrt(fna * fnb);
            if (c < s.min_frame_cos) s.min_frame_cos = c;
        }
    }
    s.rms = sqrt(sum_sq / ((double) n_frames * dim));
    s.cos = (na > 0.0 && nb > 0.0) ? dot / sqrt(na * nb) : 0.0;
    return s;
}

static void report(const char * label, const stats & s) {
    printf("  %-24s max_diff %.3e (frame %d)  rms %.3e  rel %.3e  cos %.6f  min_frame_cos %.6f\n",
           label, s.max_abs, s.worst_frame, s.rms,
           s.absmax > 0.0 ? s.max_abs / s.absmax : 0.0, s.cos, s.min_frame_cos);
}

int main(int argc, char ** argv) {
    std::string model_path;
    std::string ref_path;
    int n_frames = 66;
    int chunk_frames = 22;
    int lookahead = 0;
    int n_threads = 4;
    unsigned seed = 1234;

    for (int i = 1; i < argc; i++) {
        std::string a = argv[i];
        if (a == "--vae-model" && i + 1 < argc) {
            model_path = argv[++i];
        } else if (a == "--ref-vae-model" && i + 1 < argc) {
            ref_path = argv[++i];
        } else if (a == "--frames" && i + 1 < argc) {
            n_frames = atoi(argv[++i]);
        } else if (a == "--chunk-frames" && i + 1 < argc) {
            chunk_frames = atoi(argv[++i]);
        } else if (a == "--lookahead" && i + 1 < argc) {
            lookahead = atoi(argv[++i]);
        } else if ((a == "-t" || a == "--threads") && i + 1 < argc) {
            n_threads = atoi(argv[++i]);
        } else if (a == "--seed" && i + 1 < argc) {
            seed = (unsigned) atoi(argv[++i]);
        } else {
            usage(argv[0]);
            return 1;
        }
    }
    if (model_path.empty() || n_frames <= 0 || chunk_frames <= 0) {
        usage(argv[0]);
        return 1;
    }
    // With lookahead, round 0 consumes C + L frames and every later round C, so
    // N rounds consume N*C + L frames.
    if ((n_frames - lookahead) % chunk_frames != 0 || n_frames <= lookahead) {
        fprintf(stderr, "error: --frames (%d) minus --lookahead (%d) must be a positive multiple "
                        "of --chunk-frames (%d)\n", n_frames, lookahead, chunk_frames);
        return 1;
    }

    const int n_samples = n_frames * VAE_STREAM_COMPRESS_RATIO;
    const int chunk_samples = chunk_frames * VAE_STREAM_COMPRESS_RATIO;

    // Deterministic pseudo-audio: a couple of tones plus noise, so every channel
    // sees signal rather than a constant the caches could hide behind.
    std::vector<float> audio(n_samples);
    unsigned st = seed;
    for (int i = 0; i < n_samples; i++) {
        st = st * 1664525u + 1013904223u;
        const float noise = ((float)(st >> 8) / (float)(1u << 24)) * 2.0f - 1.0f;
        const float t = (float) i / 24000.0f;
        audio[i] = 0.4f * sinf(2.0f * 3.14159265f * 220.0f * t)
                 + 0.2f * sinf(2.0f * 3.14159265f * 1370.0f * t)
                 + 0.1f * noise;
    }

    struct vae_model_params mparams = vae_model_default_params();
    mparams.n_threads = n_threads;
    struct vae_context_params cparams = vae_context_default_params();
    cparams.n_threads = n_threads;

    vae_model_t * model = vae_load_model_from_file(model_path.c_str(), mparams);
    if (!model) {
        fprintf(stderr, "error: failed to load %s\n", model_path.c_str());
        return 1;
    }

    const int ad = vae_model_acoustic_dim(model);
    const int sd = vae_model_semantic_dim(model);
    printf("\nmodel: %s\n", model_path.c_str());
    printf("acoustic_dim=%d semantic_dim=%d\n", ad, sd);
    printf("frames=%d  chunk=%d frames (%d chunks)  threads=%d\n\n",
           n_frames, chunk_frames, n_frames / chunk_frames, n_threads);

    // --- batch ---
    vae_context_t * ctx = vae_new_context_with_model(model, cparams);

    std::vector<float> bat_a((size_t) n_frames * ad);
    std::vector<float> bat_s((size_t) n_frames * sd);
    float t_batch_a = 0.0f, t_batch_s = 0.0f;

    int32_t nf = vae_encode_acoustic_with_timing(ctx, audio.data(), n_samples, bat_a.data(), &t_batch_a);
    if (nf != n_frames) {
        fprintf(stderr, "error: batch acoustic returned %d frames, expected %d\n", nf, n_frames);
        return 1;
    }
    nf = vae_encode_semantic_with_timing(ctx, audio.data(), n_samples, bat_s.data(), &t_batch_s);
    if (nf != n_frames) {
        fprintf(stderr, "error: batch semantic returned %d frames, expected %d\n", nf, n_frames);
        return 1;
    }
    printf("batch    : acoustic %7.1f ms   semantic %7.1f ms   total %7.1f ms\n",
           t_batch_a, t_batch_s, t_batch_a + t_batch_s);

    // --- streaming ---
    vae_stream_t * s = vae_stream_init(model, cparams);
    std::vector<float> str_a((size_t) n_frames * ad);
    std::vector<float> str_s((size_t) n_frames * sd);
    float t_stream = 0.0f, t_worst = 0.0f;

    const int n_chunks = (n_frames - lookahead) / chunk_frames;
    int      block_frames = 0;   // frames the LM sees per round
    double   block_max_diff = 0.0;

    if (lookahead == 0) {
        for (int c = 0; c < n_chunks; c++) {
            const int f0 = c * chunk_frames;
            float t_chunk = 0.0f;
            int32_t got = vae_stream_encode(s,
                audio.data() + (size_t) f0 * VAE_STREAM_COMPRESS_RATIO, chunk_samples,
                str_a.data() + (size_t) f0 * ad,
                str_s.data() + (size_t) f0 * sd,
                &t_chunk);
            if (got != chunk_frames) {
                fprintf(stderr, "error: chunk %d returned %d frames, expected %d\n", c, got, chunk_frames);
                return 1;
            }
            t_stream += t_chunk;
            if (t_chunk > t_worst) t_worst = t_chunk;
        }
        block_frames = chunk_frames;
    } else {
        // Training layout: block k covers frames [C*k, C*k + C + L). Round 0 must
        // therefore be fed C + L frames of audio and every later round C, since
        // the L trailing frames of block k-1 are re-emitted from the cache.
        vae_stream_set_lookahead(s, lookahead);
        block_frames = chunk_frames + lookahead;

        std::vector<float> blk_a((size_t) block_frames * ad);
        std::vector<float> blk_s((size_t) block_frames * sd);

        int audio_f0 = 0;
        for (int c = 0; c < n_chunks; c++) {
            const int feed = (c == 0) ? chunk_frames + lookahead : chunk_frames;
            float t_chunk = 0.0f;
            int32_t got = vae_stream_encode_block(s,
                audio.data() + (size_t) audio_f0 * VAE_STREAM_COMPRESS_RATIO,
                feed * VAE_STREAM_COMPRESS_RATIO,
                blk_a.data(), blk_s.data(), &t_chunk);
            if (got != block_frames) {
                fprintf(stderr, "error: block %d returned %d frames, expected %d\n",
                        c, got, block_frames);
                return 1;
            }
            audio_f0 += feed;
            t_stream += t_chunk;
            if (t_chunk > t_worst) t_worst = t_chunk;

            // Every block, cached frames included, must match the batch rows at
            // its absolute position.
            const int abs0 = c * chunk_frames;
            const stats ba = compare(bat_a.data() + (size_t) abs0 * ad, blk_a.data(), block_frames, ad);
            const stats bs = compare(bat_s.data() + (size_t) abs0 * sd, blk_s.data(), block_frames, sd);
            if (ba.max_abs > block_max_diff) block_max_diff = ba.max_abs;
            if (bs.max_abs > block_max_diff) block_max_diff = bs.max_abs;

            // Fold into the flat arrays for the overall comparison below. The
            // overlap is written twice with the same values by construction.
            memcpy(str_a.data() + (size_t) abs0 * ad, blk_a.data(), (size_t) block_frames * ad * sizeof(float));
            memcpy(str_s.data() + (size_t) abs0 * sd, blk_s.data(), (size_t) block_frames * sd * sizeof(float));
        }
    }

    // The audio a round has to wait for is C frames in steady state (only the
    // first round pays C + L), which is what the lookahead cache buys.
    const double chunk_audio_ms = chunk_frames * 1000.0 * VAE_STREAM_COMPRESS_RATIO / 24000.0;
    printf("streaming: total    %7.1f ms   %.1f ms/chunk (worst %.1f)   conv cache %.1f KiB\n",
           t_stream, t_stream / n_chunks, t_worst, (double) vae_stream_cache_bytes(s) / 1024.0);
    printf("           chunk audio %.0f ms  ->  encoder RTF %.3f (worst %.3f)\n",
           chunk_audio_ms, (t_stream / n_chunks) / chunk_audio_ms, t_worst / chunk_audio_ms);
    if (lookahead > 0) {
        printf("           lookahead=%d: block %d frames, %d encoded/round, "
               "block-vs-batch max_diff %.3e\n",
               lookahead, block_frames, chunk_frames, block_max_diff);
    }
    printf("\n");

    // --- optional F32 reference, for the I8_S accuracy floor ---
    std::vector<float> ref_a, ref_s;
    if (!ref_path.empty()) {
        vae_model_t * rm = vae_load_model_from_file(ref_path.c_str(), mparams);
        if (!rm) {
            fprintf(stderr, "error: failed to load reference %s\n", ref_path.c_str());
            return 1;
        }
        vae_context_t * rc = vae_new_context_with_model(rm, cparams);
        ref_a.resize((size_t) n_frames * ad);
        ref_s.resize((size_t) n_frames * sd);
        vae_encode_acoustic(rc, audio.data(), n_samples, ref_a.data());
        vae_encode_semantic(rc, audio.data(), n_samples, ref_s.data());
        vae_free(rc);
        vae_free_model(rm);
    }

    // --- compare ---
    struct {
        const char * name;
        const float * bat;
        const float * str;
        const std::vector<float> * ref;
        int dim;
    } cmp[2] = {
        { "acoustic", bat_a.data(), str_a.data(), &ref_a, ad },
        { "semantic", bat_s.data(), str_s.data(), &ref_s, sd },
    };

    int rc = 0;
    for (auto & c : cmp) {
        printf("%s\n", c.name);
        const stats sb = compare(c.bat, c.str, n_frames, c.dim);
        report("stream vs batch", sb);

        if (!c.ref->empty()) {
            const stats rb = compare(c.ref->data(), c.bat, n_frames, c.dim);
            const stats rs = compare(c.ref->data(), c.str, n_frames, c.dim);
            report("f32 vs batch", rb);
            report("f32 vs stream", rs);
            // Streaming is acceptable when it is no further from the F32
            // reference than batch already is (1e-3 of slack on cos).
            if (rs.cos < rb.cos - 1e-3) {
                printf("  -> streaming is measurably worse than batch\n");
                rc = 1;
            }
        } else {
            // No reference: demand exactness. Only meaningful for F32 models;
            // an I8_S model will fail here and should be run with
            // --ref-vae-model instead.
            if (sb.absmax > 0.0 && sb.max_abs / sb.absmax > 1e-4) {
                rc = 1;
            }
        }
        printf("\n");
    }

    printf("%s\n", rc == 0 ? "PASS" : "FAIL");

    vae_stream_free(s);
    vae_free(ctx);
    vae_free_model(model);
    return rc;
}
