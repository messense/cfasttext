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

#define CFASTTEXT_TRUE           (1)
#define CFASTTEXT_FALSE          (0)

typedef void* cfasttext_t;
// typedef void* cfasttext_args_t;

// CFASTTEXT_API cfasttext_args_t cfasttext_parse_args(int argc, char** argv);

CFASTTEXT_API cfasttext_t cfasttext_new(void);
CFASTTEXT_API void cfasttext_free(cfasttext_t handle);
CFASTTEXT_API void cfasttext_load_model(cfasttext_t handle, const char* filename);
CFASTTEXT_API void cfasttext_save_model(cfasttext_t handle);
CFASTTEXT_API void cfasttext_save_output(cfasttext_t handle);
CFASTTEXT_API void cfasttext_save_vectors(cfasttext_t handle);
CFASTTEXT_API int cfasttext_get_dimension(cfasttext_t handle);
CFASTTEXT_API int32_t cfasttext_get_word_id(cfasttext_t handle, const char* word);
CFASTTEXT_API int32_t cfasttext_get_subword_id(cfasttext_t handle, const char* word);
CFASTTEXT_API bool cfasttext_is_quant(cfasttext_t handle);
CFASTTEXT_API void cfasttext_analogies(cfasttext_t handle, int32_t k);
CFASTTEXT_API void cfasttext_train_thread(cfasttext_t handle, int32_t n);
CFASTTEXT_API void cfasttext_load_vectors(cfasttext_t handle, const char* filename);
CFASTTEXT_API void cfasttext_train(cfasttext_t handle, int argc, char** argv);
// CFASTTEXT_API void cfasttext_predict(cfasttext_t handle, const char* text, int32_t k, float threshold);

#ifdef __cplusplus
}
#endif /* __cplusplus */

#endif /* CFASTTEXT_CFASTTEXT_H */
