#ifndef VAE_H
#define VAE_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// Opaque handle for VAE model
typedef struct vae_model vae_model_t;
typedef struct vae_context vae_context_t;
typedef struct vae_stream vae_stream_t;

// Samples of 24 kHz audio per latent frame (product of the encoder strides).
#define VAE_STREAM_COMPRESS_RATIO 3200

// Model parameters
struct vae_model_params {
    int32_t n_threads;
    bool use_gpu;
};

// Context parameters
struct vae_context_params {
    int32_t n_threads;
};

// Get default model parameters
struct vae_model_params vae_model_default_params(void);

// Get default context parameters
struct vae_context_params vae_context_default_params(void);

// Load VAE model from GGUF file
// Returns NULL on failure
vae_model_t* vae_load_model_from_file(
    const char* model_path,
    struct vae_model_params params);

// Free model
void vae_free_model(vae_model_t* model);

// Create context from model
vae_context_t* vae_new_context_with_model(
    vae_model_t* model,
    struct vae_context_params params);

// Free context
void vae_free(vae_context_t* ctx);

// Get model info
int32_t vae_model_acoustic_dim(const vae_model_t* model);
int32_t vae_model_semantic_dim(const vae_model_t* model);

// Encode audio to acoustic features
// audio: input audio samples [n_samples]
// n_samples: number of audio samples
// output: output features [n_frames * acoustic_dim], caller must allocate
// Returns number of frames on success, -1 on error
int32_t vae_encode_acoustic(
    vae_context_t* ctx,
    const float* audio,
    int32_t n_samples,
    float* output);

// Encode audio to semantic features
// audio: input audio samples [n_samples]
// n_samples: number of audio samples
// output: output features [n_frames * semantic_dim], caller must allocate
// Returns number of frames on success, -1 on error
int32_t vae_encode_semantic(
    vae_context_t* ctx,
    const float* audio,
    int32_t n_samples,
    float* output);

// Encode audio to acoustic features with timing
// audio: input audio samples [n_samples]
// n_samples: number of audio samples
// output: output features [n_frames * acoustic_dim], caller must allocate
// inference_time_ms: output parameter for inference time in milliseconds
// Returns number of frames on success, -1 on error
int32_t vae_encode_acoustic_with_timing(
    vae_context_t* ctx,
    const float* audio,
    int32_t n_samples,
    float* output,
    float* inference_time_ms);

// Encode audio to semantic features with timing
// audio: input audio samples [n_samples]
// n_samples: number of audio samples
// output: output features [n_frames * semantic_dim], caller must allocate
// inference_time_ms: output parameter for inference time in milliseconds
// Returns number of frames on success, -1 on error
int32_t vae_encode_semantic_with_timing(
    vae_context_t* ctx,
    const float* audio,
    int32_t n_samples,
    float* output,
    float* inference_time_ms);

//
// Streaming encode
//
// The encoder is causal (every conv pads left-only), so audio can be fed one
// chunk at a time. Each conv keeps its own left context, which makes a chunk
// cost exactly its own frames instead of its frames plus the 67.6-frame
// receptive field. Cost: ~737 KiB per encoder.
//
// Currently F32 only; see vae_stream_encode_one for why the I8_S path needs a
// requantizing concat first.

// Create streaming state. The model must outlive it.
vae_stream_t* vae_stream_init(
    vae_model_t* model,
    struct vae_context_params params);

// Free streaming state
void vae_stream_free(vae_stream_t* s);

// Drop all history, as if starting a new utterance
void vae_stream_reset(vae_stream_t* s);

// Total bytes held in the per-layer caches (0 before the first encode)
size_t vae_stream_cache_bytes(const vae_stream_t* s);

// Encode the next chunk, continuing from the cached left context.
// n_samples must be a multiple of VAE_STREAM_COMPRESS_RATIO.
// out_acoustic: [n_frames * acoustic_dim], out_semantic: [n_frames * semantic_dim];
// either may be NULL to skip that encoder. Caller allocates.
// inference_time_ms may be NULL.
// Returns n_frames = n_samples / VAE_STREAM_COMPRESS_RATIO, or -1 on error.
int32_t vae_stream_encode(
    vae_stream_t* s,
    const float* audio,
    int32_t n_samples,
    float* out_acoustic,
    float* out_semantic,
    float* inference_time_ms);

#ifdef __cplusplus
}
#endif

#endif // VAE_H
