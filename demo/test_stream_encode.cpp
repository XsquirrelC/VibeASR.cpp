// Streaming-vs-batch equivalence check for the VAE encoder.
//
// The encoder is causal, so feeding an utterance one chunk at a time through
// vae_stream_encode must reproduce what vae_encode_* produces for the whole
// utterance in one shot. On the F32 path the two should agree to within f32
// rounding of a reordered reduction, i.e. max_diff ~ 1e-6 relative, not merely
// "close" - a per-layer cache that is one frame short or misaligned shows up as
// a large diff concentrated at the chunk boundaries, which is why this reports
// the worst frame and its index rather than just a mean.

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
    printf("  --frames N          total latent frames to encode (default 66)\n");
    printf("  --chunk-frames N    frames per streaming chunk (default 22)\n");
    printf("  -t, --threads N     threads (default 4)\n");
    printf("  --seed N            RNG seed for the synthetic audio (default 1234)\n");
}

int main(int argc, char ** argv) {
    std::string model_path;
    int n_frames = 66;
    int chunk_frames = 22;
    int n_threads = 4;
    unsigned seed = 1234;

    for (int i = 1; i < argc; i++) {
        std::string a = argv[i];
        if (a == "--vae-model" && i + 1 < argc) {
            model_path = argv[++i];
        } else if (a == "--frames" && i + 1 < argc) {
            n_frames = atoi(argv[++i]);
        } else if (a == "--chunk-frames" && i + 1 < argc) {
            chunk_frames = atoi(argv[++i]);
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
    if (n_frames % chunk_frames != 0) {
        fprintf(stderr, "error: --frames (%d) must be a multiple of --chunk-frames (%d)\n",
                n_frames, chunk_frames);
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

    // --- batch reference ---
    struct vae_context_params cparams = vae_context_default_params();
    cparams.n_threads = n_threads;
    vae_context_t * ctx = vae_new_context_with_model(model, cparams);

    std::vector<float> ref_a((size_t) n_frames * ad);
    std::vector<float> ref_s((size_t) n_frames * sd);
    float t_batch_a = 0.0f, t_batch_s = 0.0f;

    int32_t nf = vae_encode_acoustic_with_timing(ctx, audio.data(), n_samples, ref_a.data(), &t_batch_a);
    if (nf != n_frames) {
        fprintf(stderr, "error: batch acoustic returned %d frames, expected %d\n", nf, n_frames);
        return 1;
    }
    nf = vae_encode_semantic_with_timing(ctx, audio.data(), n_samples, ref_s.data(), &t_batch_s);
    if (nf != n_frames) {
        fprintf(stderr, "error: batch semantic returned %d frames, expected %d\n", nf, n_frames);
        return 1;
    }
    printf("batch    : acoustic %7.1f ms   semantic %7.1f ms\n", t_batch_a, t_batch_s);

    // --- streaming ---
    vae_stream_t * s = vae_stream_init(model, cparams);
    std::vector<float> str_a((size_t) n_frames * ad);
    std::vector<float> str_s((size_t) n_frames * sd);
    float t_stream = 0.0f;

    for (int c = 0; c * chunk_frames < n_frames; c++) {
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
    }
    printf("streaming: total    %7.1f ms   (%.1f ms/chunk)   cache %.1f KiB\n\n",
           t_stream, t_stream / (float)(n_frames / chunk_frames),
           (double) vae_stream_cache_bytes(s) / 1024.0);

    // --- compare ---
    struct {
        const char * name;
        const float * ref;
        const float * str;
        int dim;
    } cmp[2] = {
        { "acoustic", ref_a.data(), str_a.data(), ad },
        { "semantic", ref_s.data(), str_s.data(), sd },
    };

    int rc = 0;
    for (auto & c : cmp) {
        double max_abs = 0.0, sum_sq = 0.0, ref_absmax = 0.0;
        int worst_frame = -1;
        for (int f = 0; f < n_frames; f++) {
            double frame_max = 0.0;
            for (int d = 0; d < c.dim; d++) {
                const size_t k = (size_t) f * c.dim + d;
                const double diff = fabs((double) c.ref[k] - (double) c.str[k]);
                if (diff > frame_max) frame_max = diff;
                sum_sq += diff * diff;
                if (fabs((double) c.ref[k]) > ref_absmax) ref_absmax = fabs((double) c.ref[k]);
            }
            if (frame_max > max_abs) {
                max_abs = frame_max;
                worst_frame = f;
            }
        }
        const double rms = sqrt(sum_sq / ((double) n_frames * c.dim));
        printf("%-8s  max_diff %.3e (frame %d)  rms %.3e  ref_absmax %.3f  rel %.3e\n",
               c.name, max_abs, worst_frame, rms, ref_absmax,
               ref_absmax > 0.0 ? max_abs / ref_absmax : 0.0);
        // 1e-4 relative is loose enough for reordered f32 reductions and far
        // tighter than any real cache bug would land.
        if (ref_absmax > 0.0 && max_abs / ref_absmax > 1e-4) {
            rc = 1;
        }
    }

    printf("\n%s\n", rc == 0 ? "PASS: streaming matches batch" : "FAIL: streaming diverges from batch");

    vae_stream_free(s);
    vae_free(ctx);
    vae_free_model(model);
    return rc;
}
