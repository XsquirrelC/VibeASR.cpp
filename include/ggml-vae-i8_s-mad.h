#pragma once

#include "ggml.h"
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif


// INT8 × INT8 vec_dot implementation (output is int32 to avoid overflow)
void ggml_vec_dot_i8_i8(int n, int32_t * s, size_t bs, const void * vx, size_t bx, const void * vy, size_t by, int nrc);

// Specializations for a reduction length n smaller than QK_I8_S. Each call
// consumes exactly QK_I8_S weight bytes, so it covers QK_I8_S / n columns:
// 16/8/4/2 on AVX2, 8/4/2 on NEON. n == 16 equals QK_I8_S on NEON and is
// therefore AVX2 only.
void ggml_vec_dot_i8_i8_n2(
    int32_t * s, size_t bs,
    const int8_t * vx,
    const int8_t * vy,
    int nrc);

void ggml_vec_dot_i8_i8_n4(
    int32_t * s, size_t bs,
    const int8_t * vx,
    const int8_t * vy,
    int nrc);

void ggml_vec_dot_i8_i8_n8(
    int32_t * s, size_t bs,
    const int8_t * vx,
    const int8_t * vy,
    int nrc);

void ggml_vec_dot_i8_i8_n16(
    int32_t * s, size_t bs,
    const int8_t * vx,
    const int8_t * vy,
    int nrc);

// INT8 × INT8 depthwise convolution: one weight row of ne00 taps per channel
// against ne11 input columns of stride ne10. ne00 == 8 takes the vector path,
// every other length falls back to scalar.
void ggml_vec_dot_i8_i8_depthwise(
    int32_t * dst_data,
    const int8_t * weight_data,
    const int8_t * input_data,
    int64_t ne00,
    int64_t ne02,
    int64_t ne10,
    int64_t ne11
    );

#ifdef __cplusplus
}
#endif
