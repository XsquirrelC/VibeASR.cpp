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
// One block == one 128-bit q register. Shapes with a smaller n (only n == 8 in
// this model) are intercepted by the specializations in ggml_gemm_i8_i8.
#define QK_I8_S 16
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
