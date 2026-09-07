#include <vector>
#include <type_traits>
#include <assert.h>
#include <cmath>
#include <cstring>
#include "ggml-vae-i8_s-mad.h"
#include "ggml-cpu-impl.h"
#include "lm-config.h"
#include "vae-config.h"

// Every vector path below is AVX2: madd_epi16, cvtepi8_epi16, hadd_epi32 and
// permutevar8x32_epi32 all need it, so a plain -mavx build takes the scalar path.
#if defined(__AVX2__)
#include <immintrin.h>
static inline int hsum_i32_8(const __m256i a) {
    const __m128i sum128 = _mm_add_epi32(_mm256_castsi256_si128(a), _mm256_extractf128_si256(a, 1));
    const __m128i hi64 = _mm_unpackhi_epi64(sum128, sum128);
    const __m128i sum64 = _mm_add_epi32(hi64, sum128);
    const __m128i hi32  = _mm_shuffle_epi32(sum64, _MM_SHUFFLE(2, 3, 0, 1));
    return _mm_cvtsi128_si32(_mm_add_epi32(sum64, hi32));
}

// Pairwise int8 products of two 32-byte vectors, widened to int32: lane k of
// *pl is qx[2k]*qy[2k] + qx[2k+1]*qy[2k+1] over the low 16 bytes, *ph the same
// over the high 16. The widening is what keeps the full [-128,127] domain
// exact: a single product reaches 16384 and a pair 32768, so an int16 result
// would saturate.
static inline void ggml_i8_s_madd_pairs(const __m256i qx, const __m256i qy, __m256i * pl, __m256i * ph) {
    *pl = _mm256_madd_epi16(_mm256_cvtepi8_epi16(_mm256_castsi256_si128(qx)),
                            _mm256_cvtepi8_epi16(_mm256_castsi256_si128(qy)));
    *ph = _mm256_madd_epi16(_mm256_cvtepi8_epi16(_mm256_extracti128_si256(qx, 1)),
                            _mm256_cvtepi8_epi16(_mm256_extracti128_si256(qy, 1)));
}

// Eight 4-byte dot products: lane j is the dot product of bytes 4j..4j+3.
static inline __m256i ggml_i8_s_dot4(const __m256i qx, const __m256i qy) {
#if defined(__AVXVNNIINT8__)
    return _mm256_dpbssd_epi32(_mm256_setzero_si256(), qx, qy);
#else
    __m256i pl, ph;
    ggml_i8_s_madd_pairs(qx, qy, &pl, &ph);
    // hadd interleaves the two 128-bit halves, the permute restores byte order.
    return _mm256_permutevar8x32_epi32(_mm256_hadd_epi32(pl, ph),
                                       _mm256_setr_epi32(0, 1, 4, 5, 2, 3, 6, 7));
#endif
}

// Four 8-byte dot products: lane j is the dot product of bytes 8j..8j+7.
static inline __m128i ggml_i8_s_dot8(const __m256i qx, const __m256i qy) {
    const __m256i d = ggml_i8_s_dot4(qx, qy);
    return _mm256_castsi256_si128(_mm256_permutevar8x32_epi32(
        _mm256_hadd_epi32(d, d), _mm256_setr_epi32(0, 1, 4, 5, 0, 1, 4, 5)));
}

// Fold one 32-byte block into an int32 accumulator that the caller finishes
// with hsum_i32_8.
static inline __m256i ggml_i8_s_acc32(const __m256i accu, const __m256i qx, const __m256i qy) {
#if defined(__AVXVNNIINT8__)
    return _mm256_dpbssd_epi32(accu, qx, qy);
#else
    __m256i pl, ph;
    ggml_i8_s_madd_pairs(qx, qy, &pl, &ph);
    return _mm256_add_epi32(accu, _mm256_add_epi32(pl, ph));
#endif
}
#endif

void ggml_vec_dot_i8_i8_1x1(int n, int32_t * s, size_t bs, const void * vx, size_t bx, const void * vy, size_t by, int nrc) {
#if defined(__AVX2__)
    const int8_t * x = (const int8_t *)vx;
    const int8_t * y = (const int8_t *)vy;

    const int nb   = n / QK_I8_S;
    const int tail = n % QK_I8_S;

    for (int row = 0; row < nrc; row++) {
        const int8_t * px = x + row * bx;
        const int8_t * py = y;
        __m256i accu = _mm256_setzero_si256();

        for (int i = 0; i < nb; i++, px += QK_I8_S, py += QK_I8_S) {
            accu = ggml_i8_s_acc32(accu, _mm256_loadu_si256((const __m256i *)px),
                                         _mm256_loadu_si256((const __m256i *)py));
        }

        int32_t sumi = hsum_i32_8(accu);
        for (int k = 0; k < tail; k++) {
            sumi += (int32_t)px[k] * (int32_t)py[k];
        }
        s[row] = sumi;
    }
#elif defined(__ARM_NEON) && defined(__aarch64__)
    const int8_t * x = (const int8_t *)vx;
    const int8_t * y = (const int8_t *)vy;

    const int nb   = n / QK_I8_S;
    const int tail = n % QK_I8_S;

    for (int row = 0; row < nrc; row++) {
        const int8_t * px = x + row * bx;
        const int8_t * py = y;
        int32x4_t accu = vdupq_n_s32(0);

        for (int i = 0; i < nb; i++, px += QK_I8_S, py += QK_I8_S) {
#if defined(__ARM_FEATURE_DOTPROD)
            accu = vdotq_s32(accu, vld1q_s8(px), vld1q_s8(py));
#else
            const int8x16_t xv = vld1q_s8(px);
            const int8x16_t yv = vld1q_s8(py);
            // One int8*int8 product fits int16, a running sum of them does not,
            // so widen into the int32 accumulator on every block.
            accu = vpadalq_s16(accu, vmull_s8(vget_low_s8(xv), vget_low_s8(yv)));
            accu = vpadalq_s16(accu, vmull_high_s8(xv, yv));
#endif
        }

        int32_t sumi = (int32_t)vaddlvq_s32(accu);
        for (int k = 0; k < tail; k++) {
            sumi += (int32_t)px[k] * (int32_t)py[k];
        }
        s[row] = sumi;
    }
#else
    const int8_t * x = (const int8_t *)vx;
    const int8_t * y = (const int8_t *)vy;
    for (int row = 0; row < nrc; row++) {
        const int8_t * x_row = x + row * bx;
        int32_t sumi = 0;
        for (int k = 0; k < n; k++) {
            sumi += (int32_t)x_row[k] * (int32_t)y[k];
        }
        s[row] = sumi;
    }
#endif
}

void ggml_vec_dot_i8_i8_1xN(int n, int32_t * s, size_t bs, const void * vx, size_t bx, const void * vy, size_t by, int nrc) {
#if defined(__AVX2__)
    const int8_t * x = (const int8_t *)vx;
    const int8_t * y = (const int8_t *)vy;

    const int nb   = n / QK_I8_S;
    const int tail = n % QK_I8_S;

    for (int row = 0; row < nrc; row += VAE_PARALLEL_SIZE) {
        __m256i accu[VAE_PARALLEL_SIZE];
        const int8_t * px[VAE_PARALLEL_SIZE];
        for (int rb = 0; rb < VAE_PARALLEL_SIZE; rb++) {
            accu[rb] = _mm256_setzero_si256();
            px[rb]   = x + (row + rb) * bx;
        }
        const int8_t * py = y;

        for (int i = 0; i < nb; i++, py += QK_I8_S) {
            const __m256i qy = _mm256_loadu_si256((const __m256i *)py);
            for (int rb = 0; rb < VAE_PARALLEL_SIZE; rb++) {
                accu[rb] = ggml_i8_s_acc32(accu[rb],
                    _mm256_loadu_si256((const __m256i *)px[rb]), qy);
                px[rb] += QK_I8_S;
            }
        }

        for (int rb = 0; rb < VAE_PARALLEL_SIZE; rb++) {
            int32_t sumi = hsum_i32_8(accu[rb]);
            for (int k = 0; k < tail; k++) {
                sumi += (int32_t)px[rb][k] * (int32_t)py[k];
            }
            s[row + rb] = sumi;
        }
    }
#elif defined(__ARM_NEON) && defined(__aarch64__)
    const int8_t * x = (const int8_t *)vx;
    const int8_t * y = (const int8_t *)vy;

    const int nb   = n / QK_I8_S;
    const int tail = n % QK_I8_S;

    for (int row = 0; row < nrc; row += VAE_PARALLEL_SIZE) {
        int32x4_t accu[VAE_PARALLEL_SIZE];
        const int8_t * px[VAE_PARALLEL_SIZE];
        for (int rb = 0; rb < VAE_PARALLEL_SIZE; rb++) {
            accu[rb] = vdupq_n_s32(0);
            px[rb]   = x + (row + rb) * bx;
        }
        const int8_t * py = y;

        for (int i = 0; i < nb; i++, py += QK_I8_S) {
            const int8x16_t yv = vld1q_s8(py);
            for (int rb = 0; rb < VAE_PARALLEL_SIZE; rb++) {
#if defined(__ARM_FEATURE_DOTPROD)
                accu[rb] = vdotq_s32(accu[rb], vld1q_s8(px[rb]), yv);
#else
                const int8x16_t xv = vld1q_s8(px[rb]);
                accu[rb] = vpadalq_s16(accu[rb], vmull_s8(vget_low_s8(xv), vget_low_s8(yv)));
                accu[rb] = vpadalq_s16(accu[rb], vmull_high_s8(xv, yv));
#endif
                px[rb] += QK_I8_S;
            }
        }

        for (int rb = 0; rb < VAE_PARALLEL_SIZE; rb++) {
            int32_t sumi = (int32_t)vaddlvq_s32(accu[rb]);
            for (int k = 0; k < tail; k++) {
                sumi += (int32_t)px[rb][k] * (int32_t)py[k];
            }
            s[row + rb] = sumi;
        }
    }
#else
    const int8_t * x = (const int8_t *)vx;
    const int8_t * y = (const int8_t *)vy;
    for (int row = 0; row < nrc; row++) {
        const int8_t * x_row = x + row * bx;
        int32_t sumi = 0;
        for (int k = 0; k < n; k++) {
            sumi += (int32_t)x_row[k] * (int32_t)y[k];
        }
        s[row] = sumi;
    }
#endif
}

void ggml_vec_dot_i8_i8_Nx1(int n, int32_t * s, size_t bs, const void * vx, size_t bx, const void * vy, size_t by, int nrc) {
#if defined(__AVX2__)
    const int8_t * x = (const int8_t *)vx;
    const int8_t * y = (const int8_t *)vy;

    const int nb   = n / QK_I8_S;
    const int tail = n % QK_I8_S;

    for (int col = 0; col < nrc; col += VAE_PARALLEL_SIZE) {
        __m256i accu[VAE_PARALLEL_SIZE];
        const int8_t * py[VAE_PARALLEL_SIZE];
        for (int cb = 0; cb < VAE_PARALLEL_SIZE; cb++) {
            accu[cb] = _mm256_setzero_si256();
            py[cb]   = y + (col + cb) * by;
        }
        const int8_t * px = x;

        for (int i = 0; i < nb; i++, px += QK_I8_S) {
            const __m256i qx = _mm256_loadu_si256((const __m256i *)px);
            for (int cb = 0; cb < VAE_PARALLEL_SIZE; cb++) {
                accu[cb] = ggml_i8_s_acc32(accu[cb], qx,
                    _mm256_loadu_si256((const __m256i *)py[cb]));
                py[cb] += QK_I8_S;
            }
        }

        for (int cb = 0; cb < VAE_PARALLEL_SIZE; cb++) {
            int32_t sumi = hsum_i32_8(accu[cb]);
            for (int k = 0; k < tail; k++) {
                sumi += (int32_t)px[k] * (int32_t)py[cb][k];
            }
            s[(col + cb) * bs] = sumi;
        }
    }
#elif defined(__ARM_NEON) && defined(__aarch64__)
    const int8_t * x = (const int8_t *)vx;
    const int8_t * y = (const int8_t *)vy;

    const int nb   = n / QK_I8_S;
    const int tail = n % QK_I8_S;

    for (int col = 0; col < nrc; col += VAE_PARALLEL_SIZE) {
        int32x4_t accu[VAE_PARALLEL_SIZE];
        const int8_t * py[VAE_PARALLEL_SIZE];
        for (int cb = 0; cb < VAE_PARALLEL_SIZE; cb++) {
            accu[cb] = vdupq_n_s32(0);
            py[cb]   = y + (col + cb) * by;
        }
        const int8_t * px = x;

        for (int i = 0; i < nb; i++, px += QK_I8_S) {
            const int8x16_t xv = vld1q_s8(px);
            for (int cb = 0; cb < VAE_PARALLEL_SIZE; cb++) {
#if defined(__ARM_FEATURE_DOTPROD)
                accu[cb] = vdotq_s32(accu[cb], xv, vld1q_s8(py[cb]));
#else
                const int8x16_t yv = vld1q_s8(py[cb]);
                accu[cb] = vpadalq_s16(accu[cb], vmull_s8(vget_low_s8(xv), vget_low_s8(yv)));
                accu[cb] = vpadalq_s16(accu[cb], vmull_high_s8(xv, yv));
#endif
                py[cb] += QK_I8_S;
            }
        }

        for (int cb = 0; cb < VAE_PARALLEL_SIZE; cb++) {
            int32_t sumi = (int32_t)vaddlvq_s32(accu[cb]);
            for (int k = 0; k < tail; k++) {
                sumi += (int32_t)px[k] * (int32_t)py[cb][k];
            }
            s[(col + cb) * bs] = sumi;
        }
    }
#else
    const int8_t * x = (const int8_t *)vx;
    const int8_t * y = (const int8_t *)vy;
    for (int col = 0; col < nrc; col++) {
        const int8_t * y_col = y + col * by;
        int32_t sumi = 0;
        for (int k = 0; k < n; k++) {
            sumi += (int32_t)x[k] * (int32_t)y_col[k];
        }
        s[col * bs] = sumi;
    }
#endif
}

void ggml_vec_dot_i8_i8(int n, int32_t * s, size_t bs, const void * vx, size_t bx, const void * vy, size_t by, int nrc) {
    if (nrc % VAE_PARALLEL_SIZE == 0) {
#if defined(VAE_ACT_PARALLEL)
        ggml_vec_dot_i8_i8_Nx1(n, s, bs, vx, bx, vy, by, nrc);
#else
        ggml_vec_dot_i8_i8_1xN(n, s, bs, vx, bx, vy, by, nrc);
#endif
    } else {
        ggml_vec_dot_i8_i8_1x1(n, s, bs, vx, bx, vy, by, nrc);
    }
}

// The four kernels below all consume exactly QK_I8_S weight bytes per call,
// i.e. n * COLS == QK_I8_S == one SIMD register, so COLS is QK_I8_S / n and
// varies with the instruction set: 16/8/4/2 columns on AVX2, 8/4/2 on NEON.
// n == QK_I8_S needs no specialization, so the n == 16 kernel is AVX2 only.

void ggml_vec_dot_i8_i8_n2(
    int32_t * s, size_t bs,
    const int8_t * vx,
    const int8_t * vy,
    int nrc) {

#if defined(__AVX2__)
    const __m256i qx = _mm256_loadu_si256((const __m256i *)vx);

    for (int row = 0; row < nrc; row++) {
        int16_t vy_16;
        memcpy(&vy_16, vy + row * 2, sizeof(int16_t));
        const __m256i qy = _mm256_set1_epi16(vy_16);

        // Pair k of the products is already column k, so no reduction at all.
        __m256i pl, ph;
        ggml_i8_s_madd_pairs(qx, qy, &pl, &ph);

        _mm256_storeu_si256((__m256i *)(s + row * bs    ), pl);
        _mm256_storeu_si256((__m256i *)(s + row * bs + 8), ph);
    }
#elif defined(__ARM_NEON) && defined(__aarch64__)
    const int8x16_t w = vld1q_s8(vx);

    for (int row = 0; row < nrc; row++) {
        int16_t vy_16;
        memcpy(&vy_16, vy + row * 2, sizeof(int16_t));
        const int8x16_t yq = vreinterpretq_s8_s16(vdupq_n_s16(vy_16));

        // n == 2 does not match SDOT's 4-byte granularity; vpaddlq_s16 folds
        // adjacent products and widens to int32 in one step.
        vst1q_s32(s + row * bs,     vpaddlq_s16(vmull_s8(vget_low_s8(w), vget_low_s8(yq))));
        vst1q_s32(s + row * bs + 4, vpaddlq_s16(vmull_high_s8(w, yq)));
    }
#else
    for (int row = 0; row < nrc; row++) {
        const int8_t * vy_row = vy + row * 2;
        for (int col = 0; col < QK_I8_S / 2; col++) {
            const int8_t * vx_col = vx + col * 2;
            int32_t sumi = 0;
            for (int k = 0; k < 2; k++) {
                sumi += (int32_t)vx_col[k] * (int32_t)vy_row[k];
            }
            s[row * bs + col] = sumi;
        }
    }
#endif
}

void ggml_vec_dot_i8_i8_n4(
    int32_t * s, size_t bs,
    const int8_t * vx,
    const int8_t * vy,
    int nrc) {

#if defined(__AVX2__)
    const __m256i qx = _mm256_loadu_si256((const __m256i *)vx);

    for (int row = 0; row < nrc; row++) {
        int32_t vy_32;
        memcpy(&vy_32, vy + row * 4, sizeof(int32_t));
        const __m256i qy = _mm256_set1_epi32(vy_32);

        _mm256_storeu_si256((__m256i *)(s + row * bs), ggml_i8_s_dot4(qx, qy));
    }
#elif defined(__ARM_NEON) && defined(__aarch64__)
    const int8x16_t w = vld1q_s8(vx);

    for (int row = 0; row < nrc; row++) {
        int32_t vy_32;
        memcpy(&vy_32, vy + row * 4, sizeof(int32_t));
        const int8x16_t yq = vreinterpretq_s8_s32(vdupq_n_s32(vy_32));

#if defined(__ARM_FEATURE_DOTPROD)
        // SDOT's 4-byte granularity is exactly one column here.
        vst1q_s32(s + row * bs, vdotq_s32(vdupq_n_s32(0), w, yq));
#else
        const int32x4_t lo = vpaddlq_s16(vmull_s8(vget_low_s8(w), vget_low_s8(yq)));
        const int32x4_t hi = vpaddlq_s16(vmull_high_s8(w, yq));
        vst1q_s32(s + row * bs, vpaddq_s32(lo, hi));
#endif
    }
#else
    for (int row = 0; row < nrc; row++) {
        const int8_t * vy_row = vy + row * 4;
        for (int col = 0; col < QK_I8_S / 4; col++) {
            const int8_t * vx_col = vx + col * 4;
            int32_t sumi = 0;
            for (int k = 0; k < 4; k++) {
                sumi += (int32_t)vx_col[k] * (int32_t)vy_row[k];
            }
            s[row * bs + col] = sumi;
        }
    }
#endif
}

void ggml_vec_dot_i8_i8_n8(
    int32_t * s, size_t bs,
    const int8_t * vx,
    const int8_t * vy,
    int nrc) {

#if defined(__AVX2__)
    const __m256i qx = _mm256_loadu_si256((const __m256i *)vx);

    for (int row = 0; row < nrc; row++) {
        int64_t vy_64;
        memcpy(&vy_64, vy + row * 8, sizeof(int64_t));
        const __m256i qy = _mm256_set_epi64x(vy_64, vy_64, vy_64, vy_64);

        _mm_storeu_si128((__m128i *)(s + row * bs), ggml_i8_s_dot8(qx, qy));
    }
#elif defined(__ARM_NEON) && defined(__aarch64__)
    const int8x16_t w = vld1q_s8(vx);

    for (int row = 0; row < nrc; row++) {
        const int8x8_t yv = vld1_s8(vy + row * 8);
#if defined(__ARM_FEATURE_DOTPROD)
        const int32x4_t d = vdotq_s32(vdupq_n_s32(0), w, vcombine_s8(yv, yv));
        vst1_s32(s + row * bs, vpadd_s32(vget_low_s32(d), vget_high_s32(d)));
#else
        s[row * bs + 0] = vaddlvq_s16(vmull_s8(vget_low_s8(w),  yv));
        s[row * bs + 1] = vaddlvq_s16(vmull_s8(vget_high_s8(w), yv));
#endif
    }
#else
    for (int row = 0; row < nrc; row++) {
        const int8_t * vy_row = vy + row * 8;
        for (int col = 0; col < QK_I8_S / 8; col++) {
            const int8_t * vx_col = vx + col * 8;
            int32_t sumi = 0;
            for (int k = 0; k < 8; k++) {
                sumi += (int32_t)vx_col[k] * (int32_t)vy_row[k];
            }
            s[row * bs + col] = sumi;
        }
    }
#endif
}

void ggml_vec_dot_i8_i8_n16(
    int32_t * s, size_t bs,
    const int8_t * vx,
    const int8_t * vy,
    int nrc) {

#if defined(__AVX2__)
    const __m256i qx = _mm256_loadu_si256((const __m256i *)vx);

    for (int row = 0; row < nrc; row++) {
        const __m128i vy_128 = _mm_loadu_si128((const __m128i *)(vy + row * 16));
        const __m256i qy = _mm256_set_m128i(vy_128, vy_128);

        const __m256i d  = ggml_i8_s_dot4(qx, qy);
        const __m256i h1 = _mm256_hadd_epi32(d, d);
        const __m256i h2 = _mm256_hadd_epi32(h1, h1);

        s[row * bs + 0] = _mm256_extract_epi32(h2, 0);
        s[row * bs + 1] = _mm256_extract_epi32(h2, 4);
    }
#else
    for (int row = 0; row < nrc; row++) {
        const int8_t * vy_row = vy + row * 16;
        for (int col = 0; col < QK_I8_S / 16; col++) {
            const int8_t * vx_col = vx + col * 16;
            int32_t sumi = 0;
            for (int k = 0; k < 16; k++) {
                sumi += (int32_t)vx_col[k] * (int32_t)vy_row[k];
            }
            s[row * bs + col] = sumi;
        }
    }
#endif
}

void ggml_vec_dot_i8_i8_depthwise(
    int32_t * dst_data,
    const int8_t * weight_data,
    const int8_t * input_data,
    int64_t ne00,
    int64_t ne02,
    int64_t ne10,
    int64_t ne11) {

#if defined(__AVX2__)
    for (int64_t batch = 0; batch < ne02; batch += 1) {
        const int8_t * weight = weight_data + batch * ne00;
        const int8_t * input  = input_data + batch * ne10 * ne11;
        int32_t * output      = dst_data + batch * ne11;

        int64_t col = 0;

        // The vector path loads a fixed 8 bytes per column; any other reduction
        // length falls through to the scalar loop below.
        if (ne00 == 8) {
            int64_t weight_i64;
            memcpy(&weight_i64, weight, sizeof(int64_t));
            const __m256i w_vec = _mm256_set_epi64x(weight_i64, weight_i64, weight_i64, weight_i64);

            for (; col + 3 < ne11; col += 4) {
                int64_t iv[4];
                for (int c = 0; c < 4; c++) {
                    memcpy(&iv[c], input + (col + c) * ne10, sizeof(int64_t));
                }
                const __m256i i_vec = _mm256_set_epi64x(iv[3], iv[2], iv[1], iv[0]);

                _mm_storeu_si128((__m128i *)(output + col), ggml_i8_s_dot8(i_vec, w_vec));
            }
        }

        for (; col < ne11; col++) {
            int32_t sum = 0;
            for (int k = 0; k < ne00; k++) {
                sum += (int32_t)weight[k] * (int32_t)input[col * ne10 + k];
            }
            output[col] = sum;
        }
    }
#elif defined(__ARM_NEON) && defined(__aarch64__)
    for (int64_t batch = 0; batch < ne02; batch += 1) {
        const int8_t * weight = weight_data + batch * ne00;
        const int8_t * input  = input_data + batch * ne10 * ne11;
        int32_t * output      = dst_data + batch * ne11;

        int64_t col = 0;

        // The vector path loads a fixed 8 bytes per column; any other reduction
        // length falls through to the scalar loop below.
        if (ne00 == 8) {
            const int8x8_t wv = vld1_s8(weight);
#if defined(__ARM_FEATURE_DOTPROD)
            const int8x16_t wq = vcombine_s8(wv, wv);
#endif
            for (; col + 3 < ne11; col += 4) {
                const int8x8_t iv0 = vld1_s8(input + (col + 0) * ne10);
                const int8x8_t iv1 = vld1_s8(input + (col + 1) * ne10);
                const int8x8_t iv2 = vld1_s8(input + (col + 2) * ne10);
                const int8x8_t iv3 = vld1_s8(input + (col + 3) * ne10);
#if defined(__ARM_FEATURE_DOTPROD)
                const int32x4_t d01 = vdotq_s32(vdupq_n_s32(0), vcombine_s8(iv0, iv1), wq);
                const int32x4_t d23 = vdotq_s32(vdupq_n_s32(0), vcombine_s8(iv2, iv3), wq);
                vst1q_s32(output + col, vpaddq_s32(d01, d23));
#else
                output[col + 0] = vaddlvq_s16(vmull_s8(iv0, wv));
                output[col + 1] = vaddlvq_s16(vmull_s8(iv1, wv));
                output[col + 2] = vaddlvq_s16(vmull_s8(iv2, wv));
                output[col + 3] = vaddlvq_s16(vmull_s8(iv3, wv));
#endif
            }
        }

        for (; col < ne11; col++) {
            int32_t sum = 0;
            for (int k = 0; k < ne00; k++) {
                sum += (int32_t)weight[k] * (int32_t)input[col * ne10 + k];
            }
            output[col] = sum;
        }
    }
#else
    for (int64_t batch = 0; batch < ne02; batch += 1) {
        const int8_t * weight = weight_data + batch * ne00;
        const int8_t * input  = input_data + batch * ne10 * ne11;
        int32_t * output      = dst_data + batch * ne11;

        for (int64_t col = 0; col < ne11; col++) {
            int32_t sum = 0;
            for (int k = 0; k < ne00; k++) {
                sum += (int32_t)weight[k] * (int32_t)input[col * ne10 + k];
            }
            output[col] = sum;
        }
    }
#endif
}
