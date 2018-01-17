#ifndef CFASTTEXT_CFASTTEXT_H
#define CFASTTEXT_CFASTTEXT_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif /* __cplusplus */

#ifndef CFASTTEXT_API
#   if defined(_WIN32) || defined(_WIN64)
#       define CFASTTEXT_API __declspec(dllimport)
#   else
#       define CFASTTEXT_API extern
#   endif /* defined(_WIN32) || defined(_WIN64) */
#endif /* CFASTTEXT_API */

#define FASTTEXT_TRUE           (1)
#define CFASTTEXT_FALSE          (0)

typedef void* fasttext_t;
typedef void* fasttext_args_t;

CFASTTEXT_API fasttext_args_t cft_args_parse(int argc, char** argv);
CFASTTEXT_API void cft_args_free(fasttext_args_t handle);

CFASTTEXT_API fasttext_t cft_fasttext_new(void);
CFASTTEXT_API void cft_fasttext_free(fasttext_t handle);
CFASTTEXT_API void cft_fasttext_load_model(fasttext_t handle, const char* filename);
CFASTTEXT_API void cft_fasttext_save_model(fasttext_t handle);
CFASTTEXT_API void cft_fasttext_save_output(fasttext_t handle);
CFASTTEXT_API void cft_fasttext_save_vectors(fasttext_t handle);
CFASTTEXT_API int cft_fasttext_get_dimension(fasttext_t handle);
CFASTTEXT_API int32_t cft_fasttext_get_word_id(fasttext_t handle, const char* word);
CFASTTEXT_API int32_t cft_fasttext_get_subword_id(fasttext_t handle, const char* word);
CFASTTEXT_API bool cft_fasttext_is_quant(fasttext_t handle);
CFASTTEXT_API void cft_fasttext_analogies(fasttext_t handle, int32_t k);
CFASTTEXT_API void cft_fasttext_train_thread(fasttext_t handle, int32_t n);
CFASTTEXT_API void cft_fasttext_load_vectors(fasttext_t handle, const char* filename);
CFASTTEXT_API void cft_fasttext_train(fasttext_t handle, int argc, char** argv);
// CFASTTEXT_API void cft_fasttext_predict(fasttext_t handle, const char* text, int32_t k, float threshold);

// Higher level APIs

CFASTTEXT_API void cft_train(int argc, char** argv);

#ifdef __cplusplus
}
#endif /* __cplusplus */

#endif /* CFASTTEXT_CFASTTEXT_H */
