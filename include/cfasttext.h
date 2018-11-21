#ifndef CFASTTEXT_CFASTTEXT_H
#define CFASTTEXT_CFASTTEXT_H

#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>

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

typedef struct fasttext_t fasttext_t;
typedef struct fasttext_args_t fasttext_args_t;
typedef struct {
    float prob;
    char* label;
} fasttext_prediction_t;
typedef struct {
    fasttext_prediction_t* predictions;
    size_t length;
} fasttext_predictions_t;

CFASTTEXT_API fasttext_args_t* cft_args_new(void);
CFASTTEXT_API void cft_args_parse(fasttext_args_t* handle, int argc, char** argv);
CFASTTEXT_API void cft_args_free(fasttext_args_t* handle);
CFASTTEXT_API const char* cft_args_get_input(fasttext_args_t* handle);
CFASTTEXT_API void cft_args_set_input(fasttext_args_t* handle, const char* input);
CFASTTEXT_API const char* cft_args_get_output(fasttext_args_t* handle);
CFASTTEXT_API void cft_args_set_output(fasttext_args_t* handle, const char* output);

CFASTTEXT_API fasttext_t* cft_fasttext_new(void);
CFASTTEXT_API void cft_fasttext_free(fasttext_t* handle);
CFASTTEXT_API void cft_fasttext_load_model(fasttext_t* handle, const char* filename, char** errptr);
CFASTTEXT_API void cft_fasttext_save_model(fasttext_t* handle, char** errptr);
CFASTTEXT_API void cft_fasttext_save_output(fasttext_t* handle, char** errptr);
CFASTTEXT_API void cft_fasttext_save_vectors(fasttext_t* handle, char** errptr);
CFASTTEXT_API int cft_fasttext_get_dimension(fasttext_t* handle);
CFASTTEXT_API int32_t cft_fasttext_get_word_id(fasttext_t* handle, const char* word);
CFASTTEXT_API int32_t cft_fasttext_get_subword_id(fasttext_t* handle, const char* word);
CFASTTEXT_API bool cft_fasttext_is_quant(fasttext_t* handle);
CFASTTEXT_API void cft_fasttext_analogies(fasttext_t* handle, int32_t k);
CFASTTEXT_API void cft_fasttext_train_thread(fasttext_t* handle, int32_t n);
CFASTTEXT_API void cft_fasttext_load_vectors(fasttext_t* handle, const char* filename, char** errptr);
CFASTTEXT_API void cft_fasttext_train(fasttext_t* handle, fasttext_args_t* args, char** errptr);
CFASTTEXT_API fasttext_predictions_t* cft_fasttext_predict(fasttext_t* handle, const char* text, int32_t k, float threshold);
CFASTTEXT_API void cft_fasttext_predictions_free(fasttext_predictions_t* predictions);
CFASTTEXT_API void cft_fasttext_quantize(fasttext_t* handle, fasttext_args_t* args, char** errptr);

/**
 * Get word vector
 * i.e.
 * <pre>
 * <code>
 * fasttext_t* handle = cft_fasttext_new();
 * // ...
 * float *vector = malloc(sizeof(float) * cft_fasttext_get_dimension(handle));
 * cft_get_word_vector(handle, "hello", buf);
 * </code>
 * </pre>
 * @param handle, fasttext handle created with `cft_fasttext_new`
 * @param word, the word to be vectorized
 * @param buf, output buffer to receive word vector, size should not be less than `cft_fasttext_get_dimension(handle)`.
 */
CFASTTEXT_API void cft_fasttext_get_word_vector(fasttext_t* handle, const char* word, float* buf);

#ifdef __cplusplus
}
#endif /* __cplusplus */

#endif /* CFASTTEXT_CFASTTEXT_H */
