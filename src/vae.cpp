#include "vae.h"

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-vae-i8_s-mad.h"

#include "time_compat.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <map>
#include <string>
#include <vector>


static struct ggml_tensor* ggml_nn_rms_norm(
    struct ggml_context* ctx,
    struct ggml_tensor* x,
    struct ggml_tensor* gamma) {
    
    if (x->type == GGML_TYPE_I8_S) {
        x = ggml_rms_norm_scaled(ctx, x, gamma, 1e-5f);
    } else {
        x = ggml_rms_norm(ctx, x, 1e-5f);
        x = ggml_mul(ctx, x, gamma);
    }
    
    return x;
}

static struct ggml_tensor* ggml_nn_linear(
    struct ggml_context* ctx,
    struct ggml_tensor* x,
    struct ggml_tensor* w,
    struct ggml_tensor* b) {
    
    int64_t IC = x->ne[0];
    int64_t N = x->ne[1];
    int64_t L = x->ne[2];
    int64_t OC = w->ne[1];
    
    x = ggml_reshape_2d(ctx, x, IC, L * N);
    
    struct ggml_tensor* result;
    
    if (b != NULL && w->type == GGML_TYPE_I8_S) {
        // I8_S fused path
        result = ggml_mul_mat_add(ctx, w, x, b);
    } else {
        result = ggml_mul_mat(ctx, w, x);
        if (b != NULL) {
            result = ggml_add(ctx, result, b);
        }
    }
    
    result = ggml_reshape_3d(ctx, result, OC, L, N);

    return result;
}

static struct ggml_tensor* ggml_nn_linear_relu(
    struct ggml_context* ctx,
    struct ggml_tensor* x,
    struct ggml_tensor* w,
    struct ggml_tensor* b) {

    int64_t IC = x->ne[0];
    int64_t N = x->ne[1];
    int64_t L = x->ne[2];
    int64_t OC = w->ne[1];

    x = ggml_reshape_2d(ctx, x, IC, L * N);

    struct ggml_tensor* result;

    if (b != NULL && w->type == GGML_TYPE_I8_S) {
        result = ggml_mul_mat_add_relu(ctx, w, x, b);
    } else {
        result = ggml_mul_mat(ctx, w, x);
        if (b != NULL) {
            result = ggml_add(ctx, result, b);
        }
        result = ggml_relu(ctx, result);
    }

    result = ggml_reshape_3d(ctx, result, OC, L, N);

    return result;
}

static struct ggml_tensor* ggml_nn_conv_1d(
    struct ggml_context* ctx,
    struct ggml_tensor* x,
    struct ggml_tensor* w,
    struct ggml_tensor* b,
    int stride,
    int padding,
    int dilation) {
    
    struct ggml_tensor* result;
    
    if (b != NULL && w->type == GGML_TYPE_I8_S) {
        struct ggml_tensor* im2col = ggml_im2col_asym(ctx, w, x, stride, 0,
                                                       /*lp0=*/padding, /*rp0=*/0, /*p1=*/0,
                                                       dilation, 0, false, GGML_TYPE_I8_S);
        
        result = ggml_mul_mat_add(ctx,
                ggml_reshape_2d(ctx, w, (w->ne[0] * w->ne[1]), w->ne[2]),
                ggml_reshape_2d(ctx, im2col, im2col->ne[0], (im2col->ne[2] * im2col->ne[1])),
                b);
    } else {
        if (padding > 0) {
            x = ggml_pad_ext(ctx, x, padding, 0, 0, 0, 0, 0, 0, 0);
            padding = 0;
        }
        result = ggml_conv_1d(ctx, w, x, stride, padding, dilation);
        if (b != NULL) {
            result = ggml_add(ctx, result, b);
        }
    }
    
    return result;
}

static struct ggml_tensor* ggml_nn_conv_1d_dw(
    struct ggml_context* ctx,
    struct ggml_tensor* x,
    struct ggml_tensor* w,
    struct ggml_tensor* b,
    int stride,
    int padding,
    int dilation) {
    
    struct ggml_tensor* result;
    
    if (b != NULL && w->type == GGML_TYPE_I8_S) {
        struct ggml_tensor* new_x = ggml_reshape_4d(ctx, x, x->ne[0], 1, x->ne[1], x->ne[2]);
        struct ggml_tensor* im2col = ggml_im2col_asym(ctx, w, new_x, stride, 0,
                                                       padding, 0, 0, dilation, 0, false, GGML_TYPE_I8_S);
        result = ggml_mul_mat_add(ctx, w, im2col, b);
        result = ggml_reshape_3d(ctx, result, result->ne[1], result->ne[2], 1);
        result = ggml_cont(ctx, ggml_permute(ctx, result, 1, 0, 2, 3));
    } else {
        if (padding > 0) {
            x = ggml_pad_ext(ctx, x, padding, 0, 0, 0, 0, 0, 0, 0);
            padding = 0;
        }
        result = ggml_conv_1d_dw(ctx, w, x, stride, padding, dilation);
        if (b != NULL) {
            result = ggml_add(ctx, result, b);
        }
    }
    
    return result;
}

static struct ggml_tensor* ggml_nn_layer_scale(
    struct ggml_context* ctx,
    struct ggml_tensor* x,
    struct ggml_tensor* gamma) {
    return ggml_mul(ctx, x, gamma);
}

//
// Streaming (chunk-by-chunk) encoder state
//
// Every conv in this encoder pads left-only, so the encoder is causal and a
// chunk can be encoded from a bounded amount of history. The history is not
// small though: the kernels compose to a 67.6-latent-frame receptive field, so
// re-encoding the overlap for each 22-frame chunk would cost (68+22)/22 = 4.1x
// the frames and blow the real-time budget. Caching each conv's own left
// context instead makes a chunk cost exactly its own frames, at 737 KiB per
// encoder.
//
// A slot holds the last n_ctx time steps of one conv's input, laid out
// [n_ctx, dim] - the same time-major layout the conv sees - in the same type the
// activations flow in (F32 or I8_S). n_ctx is the causal pad that slot replaces:
// kernel - stride for the strided downsamples, kernel - 1 for the depthwise
// mixers and the head conv. On the I8_S path the slot also remembers the
// per-tensor scale its bytes were quantized with, since I8_S carries one f32
// scale per tensor.

struct vae_stream_slot {
    int    n_ctx;
    int    dim;
    enum ggml_type type;
    size_t offset;  // into vae_stream_enc::buf, in bytes
    size_t nbytes;
    float  scale  = 0.0f;  // I8_S only
    bool   filled = false;
};

struct vae_stream_enc {
    std::vector<vae_stream_slot> slots;
    std::vector<char>            buf;
    size_t                       n_bytes = 0;

    // Per-graph scratch, valid only between one build and its compute.
    int cursor = 0;
    std::vector<struct ggml_tensor*> ext;

    void begin_graph() {
        cursor = 0;
        ext.clear();
    }
};

// Replace a conv's causal left pad with that conv's cached left context.
// Returns the extended input; the tail that becomes the next chunk's cache is
// read straight out of it after compute (see vae_stream_roll). x must be
// contiguous and time-major: [L, C].
//
// Output alignment is exact as long as the stride divides L, which holds for
// every stage whenever the chunk is a whole number of latent frames: with
// n_ctx = kernel - stride prepended and no padding, the conv emits
// (n_ctx + L - kernel)/stride + 1 = L/stride outputs, at the same positions
// batch mode would produce.
static struct ggml_tensor* vae_stream_pad(
    struct ggml_context* ctx,
    vae_stream_enc* st,
    struct ggml_tensor* x,
    int n_ctx) {

    const int64_t C = x->ne[1];
    const bool   i8s = (x->type == GGML_TYPE_I8_S);
    const size_t ts  = i8s ? sizeof(int8_t) : sizeof(float);

    const int idx = st->cursor++;
    if ((int) st->slots.size() == idx) {
        // The first graph fixes the geometry; later graphs must agree, or the
        // slots would be read back into the wrong offsets.
        vae_stream_slot s;
        s.n_ctx  = n_ctx;
        s.dim    = (int) C;
        s.type   = x->type;
        s.offset = st->n_bytes;
        s.nbytes = (size_t) n_ctx * C * ts;
        st->slots.push_back(s);
        st->n_bytes += s.nbytes;
    } else {
        GGML_ASSERT(st->slots[idx].n_ctx == n_ctx);
        GGML_ASSERT(st->slots[idx].dim   == (int) C);
        GGML_ASSERT(st->slots[idx].type  == x->type);
    }
    const vae_stream_slot& slot = st->slots[idx];

    struct ggml_tensor* cache = ggml_new_tensor_2d(ctx, x->type, n_ctx, C);
    if (slot.filled) {
        memcpy(cache->data, st->buf.data() + slot.offset, slot.nbytes);
    } else {
        // First chunk: zero history, which is exactly the zero pad batch mode
        // would have inserted.
        memset(cache->data, 0, slot.nbytes);
    }
    if (i8s) {
        // An unfilled slot is all zeros, and scale 0 is how I8_S encodes that.
        *(float*)((char*) cache->data + (size_t) n_ctx * C) = slot.filled ? slot.scale : 0.0f;
    }

    struct ggml_tensor* ext = i8s
        ? ggml_i8_s_concat(ctx, cache, x)
        : ggml_concat(ctx, cache, x, /*dim =*/ 0);

    st->ext.push_back(ext);

    return ext;
}

// Copy each conv's new left context out of the computed graph. Must run after
// ggml_graph_compute and before the compute context is reset.
static void vae_stream_roll(vae_stream_enc* st) {
    GGML_ASSERT(st->ext.size() == st->slots.size());

    for (size_t i = 0; i < st->slots.size(); i++) {
        vae_stream_slot&    slot = st->slots[i];
        struct ggml_tensor* ext  = st->ext[i];

        const int64_t total = ext->ne[0];          // n_ctx + this chunk's L
        const size_t  ts    = (slot.type == GGML_TYPE_I8_S) ? sizeof(int8_t) : sizeof(float);
        GGML_ASSERT(total >= slot.n_ctx);

        for (int c = 0; c < slot.dim; c++) {
            memcpy(st->buf.data() + slot.offset + (size_t) c * slot.n_ctx * ts,
                   (const char*) ext->data + ((size_t) c * total + (total - slot.n_ctx)) * ts,
                   (size_t) slot.n_ctx * ts);
        }
        if (slot.type == GGML_TYPE_I8_S) {
            slot.scale = *(const float*)((const char*) ext->data + (size_t) total * slot.dim);
        }
        slot.filled = true;
    }
}

//
// ConvNeXt Block
//

struct ConvNeXtBlock {
    // Mixer (Depthwise Conv)
    struct ggml_tensor* mixer_norm_weight;
    struct ggml_tensor* mixer_conv_weight;
    struct ggml_tensor* mixer_conv_bias;
    struct ggml_tensor* mixer_layer_scale;
    
    // FFN
    struct ggml_tensor* ffn_norm_weight;
    struct ggml_tensor* ffn_fc1_weight;
    struct ggml_tensor* ffn_fc1_bias;
    struct ggml_tensor* ffn_fc2_weight;
    struct ggml_tensor* ffn_fc2_bias;
    struct ggml_tensor* ffn_layer_scale;
    
    int dim;
    int kernel_size;
    
    struct ggml_tensor* forward(
        struct ggml_context* ctx,
        struct ggml_tensor* x,
        vae_stream_enc* st = nullptr) {

        struct ggml_tensor* residual = x;
        bool is_i8s = (x->type == GGML_TYPE_I8_S);

        x = ggml_nn_rms_norm(ctx, x, mixer_norm_weight);

        x = ggml_cont(ctx, ggml_permute(ctx, x, 1, 0, 2, 3));

        if (st) {
            x = vae_stream_pad(ctx, st, x, kernel_size-1);
            x = ggml_nn_conv_1d_dw(ctx, x, mixer_conv_weight, mixer_conv_bias,
                                    /*stride=*/1, /*padding=*/0, /*dilation=*/1);
        } else {
            x = ggml_nn_conv_1d_dw(ctx, x, mixer_conv_weight, mixer_conv_bias,
                                    /*stride=*/1, /*padding=*/kernel_size-1, /*dilation=*/1);
        }

        if (is_i8s) {
            x = ggml_add_scaled(ctx, x, residual, mixer_layer_scale);
        } else {
            // F32 path: x = x * layer_scale + residual
            x = ggml_mul(ctx, x, mixer_layer_scale);
            x = ggml_add(ctx, x, residual);
        }
        
        residual = x;
        
        x = ggml_nn_rms_norm(ctx, x, ffn_norm_weight);
        
        if (is_i8s) {
            x = ggml_nn_linear_relu(ctx, x, ffn_fc1_weight, ffn_fc1_bias);
        } else {
            x = ggml_nn_linear(ctx, x, ffn_fc1_weight, ffn_fc1_bias);
            x = ggml_gelu(ctx, x);
        }
        
        x = ggml_nn_linear(ctx, x, ffn_fc2_weight, ffn_fc2_bias);

        if (is_i8s) {
            x = ggml_add_scaled(ctx, x, residual, ffn_layer_scale);
        } else {
            x = ggml_mul(ctx, x, ffn_layer_scale);
            x = ggml_add(ctx, x, residual);
        }
        
        return x;
    }
};

//
// VAE Encoder
//

struct AudioVAEEncoder {
    // Architecture configuration
    static const int n_downsamples = 7;
    static const int downsample_strides[n_downsamples];
    static const int downsample_dims[n_downsamples];
    static const int n_stages = 7;
    static const int stage_depths[n_stages];
    
    // Kernel sizes will be read from actual weights during loading
    int downsample_kernel_sizes[n_downsamples];
    int stage_kernel_sizes[n_stages];
    
    int output_dim;
    int connector_output_dim;
    
    // Downsamples 0-6 (just conv, no norm)
    struct {
        struct ggml_tensor* conv_weight;
        struct ggml_tensor* conv_bias;
    } downsamples[n_downsamples];
    
    // Stages (ConvNeXt blocks)
    std::vector<ConvNeXtBlock> stages[n_stages];
    
    // Head (just conv)
    struct ggml_tensor* head_conv_weight;
    struct ggml_tensor* head_conv_bias;
    
    // Connector (fc1 -> norm -> fc2)
    struct ggml_tensor* connector_fc1_weight;
    struct ggml_tensor* connector_fc1_bias;
    struct ggml_tensor* connector_norm_weight;
    struct ggml_tensor* connector_fc2_weight;
    struct ggml_tensor* connector_fc2_bias;
    
    struct ggml_tensor* forward(
        struct ggml_context* ctx,
        struct ggml_tensor* x,
        vae_stream_enc* st = nullptr) {

        // Downsamples and stages
        for (int i = 0; i < n_stages; i++) {

            const int pad = downsample_kernel_sizes[i] - downsample_strides[i];
            if (st) {
                x = vae_stream_pad(ctx, st, x, pad);
                x = ggml_nn_conv_1d(ctx, x, downsamples[i].conv_weight,
                                     downsamples[i].conv_bias,
                                     downsample_strides[i], 0, 1);
            } else {
                x = ggml_nn_conv_1d(ctx, x, downsamples[i].conv_weight,
                                     downsamples[i].conv_bias,
                                     downsample_strides[i], pad, 1);
            }

            for (int j = 0; j < stage_depths[i]; j++) {
                x = stages[i][j].forward(ctx, x, st);
            }

            x = ggml_cont(ctx, ggml_permute(ctx, x, 1, 0, 2, 3));

        }

        // Head
        if (st) {
            x = vae_stream_pad(ctx, st, x, 8-1);
            x = ggml_nn_conv_1d(ctx, x, head_conv_weight, head_conv_bias, 1, 0, 1);
        } else {
            x = ggml_nn_conv_1d(ctx, x, head_conv_weight, head_conv_bias, 1, 8-1, 1);
        }

        // Connector: fc1 -> norm -> fc2
        x = ggml_nn_linear(ctx, x, connector_fc1_weight, connector_fc1_bias);
        x = ggml_nn_rms_norm(ctx, x, connector_norm_weight);
        x = ggml_nn_linear(ctx, x, connector_fc2_weight, connector_fc2_bias);

        return x;
    }
};

// Static configuration
const int AudioVAEEncoder::downsample_strides[n_downsamples] = {1, 2, 2, 4, 5, 5, 8};
const int AudioVAEEncoder::downsample_dims[n_downsamples] = {32, 64, 128, 256, 512, 1024, 2048};
const int AudioVAEEncoder::stage_depths[n_stages] = {3, 3, 3, 3, 3, 3, 8};

//
// VAE Model
//

struct vae_model {
    struct ggml_context* params_ctx = nullptr;
    ggml_backend_t backend = nullptr;
    ggml_backend_buffer_t params_buffer = nullptr;
    
    AudioVAEEncoder acoustic_encoder;
    AudioVAEEncoder semantic_encoder;
    
    int acoustic_dim = 64;  // Final output dim after connector
    int semantic_dim = 128; // Final output dim after connector
    
    std::map<std::string, struct ggml_tensor*> tensors;
    
    ~vae_model() {
        if (params_buffer) {
            ggml_backend_buffer_free(params_buffer);
        }
        if (params_ctx) {
            ggml_free(params_ctx);
        }
        if (backend) {
            ggml_backend_free(backend);
        }
    }
};

struct vae_context {
    vae_model_t* model = nullptr;
    int n_threads = 4;
    
    struct ggml_context* compute_ctx = nullptr;
    
    ~vae_context() {
        if (compute_ctx) {
            ggml_free(compute_ctx);
        }
    }
};

//
// Helper: Load encoder weights from GGUF
//

static bool load_encoder_weights(
    vae_model_t* model,
    AudioVAEEncoder& encoder,
    const std::string& prefix) {
    
    // Load all downsample layers (0-6)
    for (int i = 0; i < AudioVAEEncoder::n_downsamples; i++) {
        char buf[256];
        
        snprintf(buf, sizeof(buf), "%s.downsample_layers.%d.0.conv.conv.weight", prefix.c_str(), i);
        encoder.downsamples[i].conv_weight = model->tensors[buf];
        
        snprintf(buf, sizeof(buf), "%s.downsample_layers.%d.0.conv.conv.bias", prefix.c_str(), i);
        encoder.downsamples[i].conv_bias = model->tensors[buf];
        
        if (!encoder.downsamples[i].conv_weight || !encoder.downsamples[i].conv_bias) {
            fprintf(stderr, "[VAE] Error: Failed to load downsample %d weights\n", i);
            fprintf(stderr, "[VAE]   Looking for: %s.downsample_layers.%d.0.conv.conv.*\n", prefix.c_str(), i);
            return false;
        }
        
        // Read kernel size from weight tensor shape [out_channels, in_channels, kernel_size]
        // In GGUF, dimensions are reversed, so ne[0] is kernel_size
        encoder.downsample_kernel_sizes[i] = encoder.downsamples[i].conv_weight->ne[0];
    }
    
    // Load stages
    for (int stage = 0; stage < AudioVAEEncoder::n_stages; stage++) {
        int depth = AudioVAEEncoder::stage_depths[stage];
        encoder.stages[stage].resize(depth);
        
        for (int block = 0; block < depth; block++) {
            ConvNeXtBlock& b = encoder.stages[stage][block];
            char buf[256];
            
            // Mixer
            snprintf(buf, sizeof(buf), "%s.stages.%d.%d.norm.weight", prefix.c_str(), stage, block);
            b.mixer_norm_weight = model->tensors[buf];
            
            snprintf(buf, sizeof(buf), "%s.stages.%d.%d.mixer.conv.conv.conv.weight", prefix.c_str(), stage, block);
            b.mixer_conv_weight = model->tensors[buf];
            
            snprintf(buf, sizeof(buf), "%s.stages.%d.%d.mixer.conv.conv.conv.bias", prefix.c_str(), stage, block);
            b.mixer_conv_bias = model->tensors[buf];
            
            snprintf(buf, sizeof(buf), "%s.stages.%d.%d.gamma", prefix.c_str(), stage, block);
            b.mixer_layer_scale = model->tensors[buf];
            
            // FFN
            snprintf(buf, sizeof(buf), "%s.stages.%d.%d.ffn_norm.weight", prefix.c_str(), stage, block);
            b.ffn_norm_weight = model->tensors[buf];
            
            snprintf(buf, sizeof(buf), "%s.stages.%d.%d.ffn.linear1.weight", prefix.c_str(), stage, block);
            b.ffn_fc1_weight = model->tensors[buf];
            
            snprintf(buf, sizeof(buf), "%s.stages.%d.%d.ffn.linear1.bias", prefix.c_str(), stage, block);
            b.ffn_fc1_bias = model->tensors[buf];
            
            snprintf(buf, sizeof(buf), "%s.stages.%d.%d.ffn.linear2.weight", prefix.c_str(), stage, block);
            b.ffn_fc2_weight = model->tensors[buf];
            
            snprintf(buf, sizeof(buf), "%s.stages.%d.%d.ffn.linear2.bias", prefix.c_str(), stage, block);
            b.ffn_fc2_bias = model->tensors[buf];
            
            snprintf(buf, sizeof(buf), "%s.stages.%d.%d.ffn_gamma", prefix.c_str(), stage, block);
            b.ffn_layer_scale = model->tensors[buf];
            
            // Verify all loaded
            if (!b.mixer_norm_weight || !b.mixer_conv_weight || !b.mixer_conv_bias ||
                !b.mixer_layer_scale || !b.ffn_norm_weight || !b.ffn_fc1_weight ||
                !b.ffn_fc1_bias || !b.ffn_fc2_weight || !b.ffn_fc2_bias || !b.ffn_layer_scale) {
                fprintf(stderr, "[VAE] Error: Failed to load stage %d block %d\n", stage, block);
                return false;
            }
            
            // Get dim and kernel_size from mixer conv weight shape [kernel_size, 1, dim]
            // In GGUF, dimensions are reversed, so ne[0] is kernel_size, ne[2] is dim
            b.dim = b.mixer_conv_weight->ne[2];
            b.kernel_size = b.mixer_conv_weight->ne[0];
            
            // Store kernel size at stage level (use first block's kernel size)
            if (block == 0) {
                encoder.stage_kernel_sizes[stage] = b.kernel_size;
            }
        }
    }
    
    // Load head (just conv)
    std::string head_conv_w = prefix + ".head.conv.conv.weight";
    std::string head_conv_b = prefix + ".head.conv.conv.bias";
    
    encoder.head_conv_weight = model->tensors[head_conv_w];
    encoder.head_conv_bias = model->tensors[head_conv_b];
    
    if (!encoder.head_conv_weight || !encoder.head_conv_bias) {
        fprintf(stderr, "[VAE] Error: Failed to load head weights\n");
        return false;
    }
    
    // Get output dim from head conv weight [kernel, in_dim, out_dim]
    encoder.output_dim = encoder.head_conv_weight->ne[2];
    
    // Load connector weights (fc1 -> norm -> fc2)
    std::string connector_fc1_w = prefix + "_connector.fc1.weight";
    std::string connector_fc1_b = prefix + "_connector.fc1.bias";
    std::string connector_norm_w = prefix + "_connector.norm.weight";
    std::string connector_fc2_w = prefix + "_connector.fc2.weight";
    std::string connector_fc2_b = prefix + "_connector.fc2.bias";
    
    encoder.connector_fc1_weight = model->tensors[connector_fc1_w];
    encoder.connector_fc1_bias = model->tensors[connector_fc1_b];
    encoder.connector_norm_weight = model->tensors[connector_norm_w];
    encoder.connector_fc2_weight = model->tensors[connector_fc2_w];
    encoder.connector_fc2_bias = model->tensors[connector_fc2_b];
    
    if (!encoder.connector_fc1_weight || !encoder.connector_fc1_bias ||
        !encoder.connector_norm_weight || !encoder.connector_fc2_weight || 
        !encoder.connector_fc2_bias) {
        fprintf(stderr, "[VAE] Error: Failed to load connector weights\n");
        return false;
    }
    
    // Get connector output dim from fc2 weight [input_dim, output_dim]
    encoder.connector_output_dim = encoder.connector_fc2_weight->ne[1];
    
    return true;
}

//
// Public API Implementation
//

struct vae_model_params vae_model_default_params() {
    struct vae_model_params params;
    params.n_threads = 16;
    params.use_gpu = false;
    return params;
}

struct vae_context_params vae_context_default_params() {
    struct vae_context_params params;
    params.n_threads = 16;
    return params;
}

vae_model_t* vae_load_model_from_file(
    const char* model_path,
    struct vae_model_params params) {
    
    auto model = new vae_model();
    
    // Initialize backend
    model->backend = ggml_backend_cpu_init();
    if (!model->backend) {
        fprintf(stderr, "[VAE] Error: Failed to initialize backend\n");
        delete model;
        return nullptr;
    }
    
    // Load GGUF file
    struct gguf_init_params gguf_params = {
        /*.no_alloc =*/ true,  // Don't allocate memory for tensors yet
        /*.ctx      =*/ &model->params_ctx,
    };
    
    struct gguf_context* gguf_ctx = gguf_init_from_file(model_path, gguf_params);
    if (!gguf_ctx) {
        fprintf(stderr, "[VAE] Error: Failed to load GGUF file\n");
        delete model;
        return nullptr;
    }
    
    // Read metadata
    int n_tensors = gguf_get_n_tensors(gguf_ctx);

    // Map tensors by name
    for (int i = 0; i < n_tensors; i++) {
        const char* name = gguf_get_tensor_name(gguf_ctx, i);
        struct ggml_tensor* tensor = ggml_get_tensor(model->params_ctx, name);
        model->tensors[name] = tensor;
    }
    
    // Allocate backend buffer
    model->params_buffer = ggml_backend_alloc_ctx_tensors(model->params_ctx, model->backend);
    
    // Load tensor data from file
    FILE* f = fopen(model_path, "rb");
    if (!f) {
        fprintf(stderr, "[VAE] Error: Failed to open file for reading\n");
        gguf_free(gguf_ctx);
        delete model;
        return nullptr;
    }
    
    size_t data_offset = gguf_get_data_offset(gguf_ctx);
    for (int i = 0; i < n_tensors; i++) {
        const char* name = gguf_get_tensor_name(gguf_ctx, i);
        struct ggml_tensor* tensor = model->tensors[name];
        size_t offset = data_offset + gguf_get_tensor_offset(gguf_ctx, i);
        
        fseek(f, offset, SEEK_SET);
        
        size_t tensor_size = ggml_nbytes(tensor);
        std::vector<char> buf(tensor_size);
        fread(buf.data(), 1, tensor_size, f);
        
        ggml_backend_tensor_set(tensor, buf.data(), 0, tensor_size);
    }
    
    fclose(f);
    gguf_free(gguf_ctx);
    
    // Load encoder weights
    if (!load_encoder_weights(model, model->acoustic_encoder, "acoustic")) {
        delete model;
        return nullptr;
    }
    
    if (!load_encoder_weights(model, model->semantic_encoder, "semantic")) {
        delete model;
        return nullptr;
    }
    
    model->acoustic_dim = model->acoustic_encoder.connector_output_dim;
    model->semantic_dim = model->semantic_encoder.connector_output_dim;
    
    
    return model;
}

void vae_free_model(vae_model_t* model) {
    delete model;
}

vae_context_t* vae_new_context_with_model(
    vae_model_t* model,
    struct vae_context_params params) {
    
    auto ctx = new vae_context();
    ctx->model = model;
    ctx->n_threads = params.n_threads;
    
    return ctx;
}

void vae_free(vae_context_t* ctx) {
    delete ctx;
}

int32_t vae_model_acoustic_dim(const vae_model_t* model) {
    return model->acoustic_dim;
}

int32_t vae_model_semantic_dim(const vae_model_t* model) {
    return model->semantic_dim;
}

static size_t vae_model_max_nodes(const vae_model_t* model) {
    size_t n_tensors = 552;  // VAE encoder has 552 tensors
    return std::max<size_t>(1024, n_tensors * 3);
}

static int32_t vae_encode_impl(
    vae_context_t* ctx,
    AudioVAEEncoder& encoder,
    const float* audio,
    int32_t n_samples,
    float* output,
    float* inference_time_ms = nullptr) {
    
    if (!ctx || !audio || !output || n_samples <= 0) {
        return -1;
    }
    
    // Start timing if requested
    struct timespec start_time, end_time;
    if (inference_time_ms) {
        clock_gettime(CLOCK_MONOTONIC, &start_time);
    }
    
    // Check if model weights are I8_S — if so, quantize input to I8_S for full INT8 pipeline
    const bool use_i8_s = (encoder.downsamples[0].conv_weight->type == GGML_TYPE_I8_S);

    // Create computation context with sufficient memory.
    // Arena use is linear in the input length, and depends on the weight type:
    // F32 models need more memory than I8_S due to 4x larger intermediate
    // tensors. The I8_S rate is measured at ~8.9 KB per input sample (~214 MB
    // per second of 24 kHz audio), constant across 8 s to 267 s inputs; the F32
    // rate applies the 4x ratio above.
    // A fixed reservation is wrong in both directions. 128 GB is refused
    // outright by Windows (no overcommit) and by Linux heuristic overcommit on
    // any host whose RAM + swap is smaller, aborting in ggml_aligned_malloc
    // before any audio is processed. A small fixed pool starts everywhere but
    // silently caps input length and then segfaults past it. Size the arena
    // from the actual sample count instead, with ~15% headroom.
    // The 4x-of-I8_S estimate for F32 (40960) is too low in practice - a
    // 44-frame F32 encode runs past it and aborts in ggml_new_object. Measured
    // with VAE_DEBUG_MEM=1 the F32 path actually uses 54621 B/sample, so 65536
    // (6.4x the I8_S rate) restores the ~15-20% headroom.
    const size_t bytes_per_sample = use_i8_s ? 10240 : 65536;
    const size_t vae_ctx_mem_size =
        (size_t)n_samples * bytes_per_sample + (size_t)512 * 1024 * 1024;
    struct ggml_init_params ctx_params = {
        /*.mem_size   =*/ vae_ctx_mem_size,
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ false,  // Let ggml allocate tensors
    };
    
    if (ctx->compute_ctx) {
        ggml_free(ctx->compute_ctx);
    }
    ctx->compute_ctx = ggml_init(ctx_params);
    
    struct ggml_tensor* input;
    if (use_i8_s) {
        // Quantize F32 audio to I8_S
        input = ggml_new_tensor_3d(ctx->compute_ctx, GGML_TYPE_I8_S, n_samples, 1, 1);
        ggml_set_name(input, "input_audio_i8s");
        
        // Find max abs value
        float amax = 0.00001f;
        for (int32_t i = 0; i < n_samples; i++) {
            float abs_val = fabsf(audio[i]);
            if (abs_val > amax) amax = abs_val;
        }
        float scale = 127.0f / amax;
        
        // Quantize to int8
        int8_t * dst_i8 = (int8_t *) input->data;
        for (int32_t i = 0; i < n_samples; i++) {
            int v = (int)roundf(audio[i] * scale);
            if (v >  127) v =  127;
            if (v < -128) v = -128;
            dst_i8[i] = (int8_t)v;
        }
        
        // Store scale after int8 data
        float * scale_ptr = (float *)((char *) input->data + n_samples);
        *scale_ptr = scale;
    } else {
        // Use F32 input directly
        input = ggml_new_tensor_3d(ctx->compute_ctx, GGML_TYPE_F32, n_samples, 1, 1);
        ggml_set_name(input, "input_audio");
        memcpy(input->data, audio, n_samples * sizeof(float));
    }
    
    // Build computation graph
    struct ggml_tensor* result = encoder.forward(ctx->compute_ctx, input);
    
    // Build graph with pre-allocated nodes (similar to llama_ref.cpp)
    size_t max_nodes = vae_model_max_nodes(ctx->model);
    struct ggml_cgraph* gf = ggml_new_graph_custom(ctx->compute_ctx, max_nodes, false);
    ggml_build_forward_expand(gf, result);
    
    // Compute
    if (ggml_graph_compute_with_ctx(ctx->compute_ctx, gf, ctx->n_threads) != GGML_STATUS_SUCCESS) {
        fprintf(stderr, "[VAE] Error: Graph computation failed\n");
        return -1;
    }

    if (getenv("VAE_DEBUG_MEM")) {
        fprintf(stderr, "[VAE] batch arena: %.2f / %.2f GiB, %.0f B/sample used\n",
                ggml_used_mem(ctx->compute_ctx) / 1073741824.0,
                vae_ctx_mem_size / 1073741824.0,
                (double) ggml_used_mem(ctx->compute_ctx) / n_samples);
    }
    
    // Get output dimensions
    int64_t batch = result->ne[2];
    int64_t n_frames = result->ne[1];
    int64_t out_dim = result->ne[0];
    
    if (result->type == GGML_TYPE_I8_S) {
        // Dequantize I8_S output to F32
        const int64_t n_total = n_frames * out_dim * batch;
        const int8_t * src_i8 = (const int8_t *) result->data;
        const float * scale_ptr = (const float *)((const char *) result->data + n_total);
        const float dequant = 1.0f / (*scale_ptr);  // max_abs / 127.0
        for (int64_t i = 0; i < n_total; i++) {
            output[i] = (float)src_i8[i] * dequant;
        }
    } else {
        // Copy F32 output directly
        memcpy(output, result->data, n_frames * out_dim * batch * sizeof(float));
    }
    
    // Calculate inference time if requested
    if (inference_time_ms) {
        clock_gettime(CLOCK_MONOTONIC, &end_time);
        double elapsed = (end_time.tv_sec - start_time.tv_sec) * 1000.0 + 
                        (end_time.tv_nsec - start_time.tv_nsec) / 1e6;
        *inference_time_ms = (float)elapsed;
    }
    
    return (int32_t)n_frames;
}

int32_t vae_encode_acoustic(
    vae_context_t* ctx,
    const float* audio,
    int32_t n_samples,
    float* output) {
    
    return vae_encode_impl(ctx, ctx->model->acoustic_encoder, audio, n_samples, output, nullptr);
}

int32_t vae_encode_acoustic_with_timing(
    vae_context_t* ctx,
    const float* audio,
    int32_t n_samples,
    float* output,
    float* inference_time_ms) {
    
    return vae_encode_impl(ctx, ctx->model->acoustic_encoder, audio, n_samples, output, inference_time_ms);
}

int32_t vae_encode_semantic(
    vae_context_t* ctx,
    const float* audio,
    int32_t n_samples,
    float* output) {
    
    return vae_encode_impl(ctx, ctx->model->semantic_encoder, audio, n_samples, output, nullptr);
}

int32_t vae_encode_semantic_with_timing(
    vae_context_t* ctx,
    const float* audio,
    int32_t n_samples,
    float* output,
    float* inference_time_ms) {

    return vae_encode_impl(ctx, ctx->model->semantic_encoder, audio, n_samples, output, inference_time_ms);
}

//
// Streaming encode
//

struct vae_stream {
    vae_model_t* model = nullptr;
    int n_threads = 4;

    vae_stream_enc acoustic;
    vae_stream_enc semantic;

    // Lookahead embedding cache: the last `lookahead` output frames, kept so the
    // next block can re-emit them instead of re-encoding their audio.
    int                lookahead = 0;
    int                la_valid  = 0;
    std::vector<float> la_acoustic;
    std::vector<float> la_semantic;

    int64_t n_samples_seen = 0;

    struct ggml_context* compute_ctx = nullptr;

    ~vae_stream() {
        if (compute_ctx) {
            ggml_free(compute_ctx);
        }
    }
};

static int32_t vae_stream_encode_one(
    vae_stream_t* s,
    AudioVAEEncoder& encoder,
    vae_stream_enc& st,
    const float* audio,
    int32_t n_samples,
    float* output,
    float* inference_time_ms) {

    struct timespec start_time, end_time;
    if (inference_time_ms) {
        clock_gettime(CLOCK_MONOTONIC, &start_time);
    }

    const bool use_i8_s = (encoder.downsamples[0].conv_weight->type == GGML_TYPE_I8_S);

    // Same rates as the batch path: I8_S intermediates are a quarter the size of
    // F32 ones, and the graph is otherwise identical.
    const size_t bytes_per_sample = use_i8_s ? 10240 : 65536;
    const size_t mem_size =
        (size_t) n_samples * bytes_per_sample + (size_t) 512 * 1024 * 1024;
    struct ggml_init_params ctx_params = {
        /*.mem_size   =*/ mem_size,
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ false,
    };

    if (s->compute_ctx) {
        ggml_free(s->compute_ctx);
    }
    s->compute_ctx = ggml_init(ctx_params);

    struct ggml_tensor* input;
    if (use_i8_s) {
        input = ggml_new_tensor_3d(s->compute_ctx, GGML_TYPE_I8_S, n_samples, 1, 1);
        ggml_set_name(input, "input_audio_i8s");

        float amax = 0.00001f;
        for (int32_t i = 0; i < n_samples; i++) {
            const float a = fabsf(audio[i]);
            if (a > amax) amax = a;
        }
        const float scale = 127.0f / amax;
        int8_t* dst_i8 = (int8_t*) input->data;
        for (int32_t i = 0; i < n_samples; i++) {
            int v = (int) roundf(audio[i] * scale);
            if (v >  127) v =  127;
            if (v < -128) v = -128;
            dst_i8[i] = (int8_t) v;
        }
        *(float*)((char*) input->data + n_samples) = scale;
    } else {
        input = ggml_new_tensor_3d(s->compute_ctx, GGML_TYPE_F32, n_samples, 1, 1);
        ggml_set_name(input, "input_audio");
        memcpy(input->data, audio, (size_t) n_samples * sizeof(float));
    }

    st.begin_graph();
    struct ggml_tensor* result = encoder.forward(s->compute_ctx, input, &st);

    // First graph: the slot walk above discovered the geometry, so size the
    // backing buffer now. It stays zero, which is the history the first chunk
    // should see.
    if (st.buf.empty()) {
        st.buf.assign(st.n_bytes, 0);
    }

    size_t max_nodes = vae_model_max_nodes(s->model);
    struct ggml_cgraph* gf = ggml_new_graph_custom(s->compute_ctx, max_nodes, false);
    ggml_build_forward_expand(gf, result);

    if (ggml_graph_compute_with_ctx(s->compute_ctx, gf, s->n_threads) != GGML_STATUS_SUCCESS) {
        fprintf(stderr, "[VAE] Error: streaming graph computation failed\n");
        return -1;
    }

    if (getenv("VAE_DEBUG_MEM")) {
        fprintf(stderr, "[VAE] stream arena: %.2f / %.2f GiB, %.0f B/sample used\n",
                ggml_used_mem(s->compute_ctx) / 1073741824.0,
                mem_size / 1073741824.0,
                (double) ggml_used_mem(s->compute_ctx) / n_samples);
    }

    const int64_t n_frames = result->ne[1];
    const int64_t out_dim  = result->ne[0];
    const int64_t n_total  = n_frames * out_dim * result->ne[2];
    if (result->type == GGML_TYPE_I8_S) {
        const int8_t* src_i8 = (const int8_t*) result->data;
        const float   dequant = 1.0f / *(const float*)((const char*) result->data + n_total);
        for (int64_t i = 0; i < n_total; i++) {
            output[i] = (float) src_i8[i] * dequant;
        }
    } else {
        memcpy(output, result->data, (size_t) n_total * sizeof(float));
    }

    // Roll the caches forward for the next chunk.
    vae_stream_roll(&st);

    if (inference_time_ms) {
        clock_gettime(CLOCK_MONOTONIC, &end_time);
        double elapsed = (end_time.tv_sec - start_time.tv_sec) * 1000.0 +
                        (end_time.tv_nsec - start_time.tv_nsec) / 1e6;
        *inference_time_ms = (float) elapsed;
    }

    return (int32_t) n_frames;
}

vae_stream_t* vae_stream_init(vae_model_t* model, struct vae_context_params params) {
    if (!model) {
        return nullptr;
    }
    auto s = new vae_stream();
    s->model = model;
    s->n_threads = params.n_threads;
    return s;
}

void vae_stream_free(vae_stream_t* s) {
    delete s;
}

void vae_stream_reset(vae_stream_t* s) {
    if (!s) {
        return;
    }
    for (vae_stream_enc* st : { &s->acoustic, &s->semantic }) {
        std::fill(st->buf.begin(), st->buf.end(), 0);
        for (vae_stream_slot& slot : st->slots) {
            slot.filled = false;
            slot.scale  = 0.0f;
        }
    }
    s->la_valid = 0;
    s->n_samples_seen = 0;
}

void vae_stream_set_lookahead(vae_stream_t* s, int32_t n_frames) {
    if (!s || n_frames < 0) {
        return;
    }
    s->lookahead = n_frames;
    s->la_valid  = 0;
    s->la_acoustic.assign((size_t) n_frames * s->model->acoustic_dim, 0.0f);
    s->la_semantic.assign((size_t) n_frames * s->model->semantic_dim, 0.0f);
}

size_t vae_stream_cache_bytes(const vae_stream_t* s) {
    if (!s) {
        return 0;
    }
    return s->acoustic.buf.size() + s->semantic.buf.size();
}

size_t vae_stream_lookahead_bytes(const vae_stream_t* s) {
    if (!s) {
        return 0;
    }
    return (s->la_acoustic.size() + s->la_semantic.size()) * sizeof(float);
}

int32_t vae_stream_encode(
    vae_stream_t* s,
    const float* audio,
    int32_t n_samples,
    float* out_acoustic,
    float* out_semantic,
    float* inference_time_ms) {

    if (!s || !audio || n_samples <= 0) {
        return -1;
    }
    // Whole latent frames only. A partial frame would leave the deepest stage's
    // stride unsatisfied and shift every later chunk's output positions.
    if (n_samples % VAE_STREAM_COMPRESS_RATIO != 0) {
        fprintf(stderr, "[VAE] Error: streaming chunk must be a multiple of %d samples, got %d\n",
                VAE_STREAM_COMPRESS_RATIO, n_samples);
        return -1;
    }

    float t_a = 0.0f, t_s = 0.0f;
    int32_t n_a = -1, n_s = -1;

    if (out_acoustic) {
        n_a = vae_stream_encode_one(s, s->model->acoustic_encoder, s->acoustic,
                                    audio, n_samples, out_acoustic, &t_a);
        if (n_a < 0) return -1;
    }
    if (out_semantic) {
        n_s = vae_stream_encode_one(s, s->model->semantic_encoder, s->semantic,
                                    audio, n_samples, out_semantic, &t_s);
        if (n_s < 0) return -1;
    }
    if (n_a >= 0 && n_s >= 0 && n_a != n_s) {
        fprintf(stderr, "[VAE] Error: encoders disagree on frame count (%d vs %d)\n", n_a, n_s);
        return -1;
    }

    s->n_samples_seen += n_samples;
    if (inference_time_ms) {
        *inference_time_ms = t_a + t_s;
    }
    return n_a >= 0 ? n_a : n_s;
}

int32_t vae_stream_encode_block(
    vae_stream_t* s,
    const float* audio,
    int32_t n_samples,
    float* out_acoustic,
    float* out_semantic,
    float* inference_time_ms) {

    if (!s || !audio || n_samples <= 0) {
        return -1;
    }
    const int ad = s->model->acoustic_dim;
    const int sd = s->model->semantic_dim;
    const int la = s->la_valid;

    // The new frames land after the cached lookahead frames, so the caller sees
    // one contiguous block in time order.
    const int32_t n_new = vae_stream_encode(s, audio, n_samples,
        out_acoustic ? out_acoustic + (size_t) la * ad : nullptr,
        out_semantic ? out_semantic + (size_t) la * sd : nullptr,
        inference_time_ms);
    if (n_new < 0) {
        return -1;
    }

    if (out_acoustic && la > 0) {
        memcpy(out_acoustic, s->la_acoustic.data(), (size_t) la * ad * sizeof(float));
    }
    if (out_semantic && la > 0) {
        memcpy(out_semantic, s->la_semantic.data(), (size_t) la * sd * sizeof(float));
    }

    const int32_t n_out = la + n_new;

    // Keep this block's trailing lookahead frames for the next round.
    if (s->lookahead > 0) {
        if (n_out < s->lookahead) {
            fprintf(stderr, "[VAE] Error: block of %d frames is shorter than the %d-frame lookahead\n",
                    n_out, s->lookahead);
            return -1;
        }
        const int off = n_out - s->lookahead;
        if (out_acoustic) {
            memcpy(s->la_acoustic.data(), out_acoustic + (size_t) off * ad,
                   (size_t) s->lookahead * ad * sizeof(float));
        }
        if (out_semantic) {
            memcpy(s->la_semantic.data(), out_semantic + (size_t) off * sd,
                   (size_t) s->lookahead * sd * sizeof(float));
        }
        s->la_valid = s->lookahead;
    }

    return n_out;
}
