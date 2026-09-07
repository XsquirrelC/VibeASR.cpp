#define VAE_ACT_PARALLEL
#if defined(__AVX__) || defined(__AVX2__) || defined(__AVX512F__) || defined(__SSSE3__)
#define QK_I8_S 32
#if defined(VAE_ACT_PARALLEL)
    #define VAE_ROW_BLOCK_SIZE 16
    #define VAE_COL_BLOCK_SIZE 16
    #define VAE_PARALLEL_SIZE 4
#else
    #define VAE_ROW_BLOCK_SIZE 16
    #define VAE_COL_BLOCK_SIZE 4
    #define VAE_PARALLEL_SIZE 4
#endif
#elif defined(__ARM_NEON)
// Stays at 8 because ggml_gemm_i8_i8 asserts n % QK_I8_S == 0 and only
// specializes n == 2 and n == 4 on ARM, so n == 8 shapes reach the generic tile
// kernel. The dot-product kernels below still work a full 128-bit q-register at
// a time by consuming two adjacent blocks per SDOT.
#define QK_I8_S 8
#if defined(VAE_ACT_PARALLEL)
    #define VAE_ROW_BLOCK_SIZE 16
    #define VAE_COL_BLOCK_SIZE 16
    #define VAE_PARALLEL_SIZE 4
#else
    #define VAE_ROW_BLOCK_SIZE 16
    #define VAE_COL_BLOCK_SIZE 4
    #define VAE_PARALLEL_SIZE 4
#endif
#else
#define QK_I8_S 32
#endif
