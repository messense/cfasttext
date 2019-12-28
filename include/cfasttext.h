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
typedef struct fasttext_autotune_t fasttext_autotune_t;
typedef struct {
    float prob;
    char* label;
} fasttext_prediction_t;
typedef struct {
    fasttext_prediction_t* predictions;
    size_t length;
} fasttext_predictions_t;
typedef struct {
    char** tokens;
    size_t length;
} fasttext_tokens_t;
typedef struct {
    int32_t* words;
    size_t length;
} fasttext_words_t;

typedef enum {
    MODEL_CBOW = 1,
    MODEL_SG,
    MODEL_SUP,
} model_name_t;

typedef enum {
    LOSS_HS = 1,
    LOSS_NS,
    LOSS_SOFTMAX,
    LOSS_OVA,
} loss_name_t;

typedef enum {
    F1_SCORE = 1,
    LABEL_F1_SCORE,
} metric_name_t;

CFASTTEXT_API void cft_str_free(char* s);
/* args APIs */
CFASTTEXT_API fasttext_args_t* cft_args_new(void);
CFASTTEXT_API void cft_args_parse(fasttext_args_t* handle, int argc, char** argv);
CFASTTEXT_API void cft_args_free(fasttext_args_t* handle);
CFASTTEXT_API const char* cft_args_get_input(fasttext_args_t* handle);
CFASTTEXT_API void cft_args_set_input(fasttext_args_t* handle, const char* input);
CFASTTEXT_API const char* cft_args_get_output(fasttext_args_t* handle);
CFASTTEXT_API void cft_args_set_output(fasttext_args_t* handle, const char* output);
CFASTTEXT_API double cft_args_get_lr(fasttext_args_t* handle);
CFASTTEXT_API void cft_args_set_lr(fasttext_args_t* handle, double lr);
CFASTTEXT_API int cft_args_get_lr_update_rate(fasttext_args_t* handle);
CFASTTEXT_API void cft_args_set_lr_update_rate(fasttext_args_t* handle, int rate);
CFASTTEXT_API int cft_args_get_dim(fasttext_args_t* handle);
CFASTTEXT_API void cft_args_set_dim(fasttext_args_t* handle, int dim);
CFASTTEXT_API int cft_args_get_ws(fasttext_args_t* handle);
CFASTTEXT_API void cft_args_set_ws(fasttext_args_t* handle, int ws);
CFASTTEXT_API int cft_args_get_epoch(fasttext_args_t* handle);
CFASTTEXT_API void cft_args_set_epoch(fasttext_args_t* handle, int epoch);
CFASTTEXT_API int cft_args_get_thread(fasttext_args_t* handle);
CFASTTEXT_API void cft_args_set_thread(fasttext_args_t* handle, int thread);
CFASTTEXT_API model_name_t cft_args_get_model(fasttext_args_t* handle);
CFASTTEXT_API void cft_args_set_model(fasttext_args_t* handle, model_name_t model);
CFASTTEXT_API loss_name_t cft_args_get_loss(fasttext_args_t* handle);
CFASTTEXT_API void cft_args_set_loss(fasttext_args_t* handle, loss_name_t loss);
CFASTTEXT_API int cft_args_get_min_count(fasttext_args_t* handle);
CFASTTEXT_API void cft_args_set_min_count(fasttext_args_t* handle, int min_count);
CFASTTEXT_API int cft_args_get_min_count_label(fasttext_args_t* handle);
CFASTTEXT_API void cft_args_set_min_count_label(fasttext_args_t* handle, int min_count);
CFASTTEXT_API int cft_args_get_neg(fasttext_args_t* handle);
CFASTTEXT_API void cft_args_set_neg(fasttext_args_t* handle, int neg);
CFASTTEXT_API int cft_args_get_word_ngrams(fasttext_args_t* handle);
CFASTTEXT_API void cft_args_set_word_ngrams(fasttext_args_t* handle, int ngrams);
CFASTTEXT_API int cft_args_get_bucket(fasttext_args_t* handle);
CFASTTEXT_API void cft_args_set_bucket(fasttext_args_t* handle, int bucket);
CFASTTEXT_API int cft_args_get_minn(fasttext_args_t* handle);
CFASTTEXT_API void cft_args_set_minn(fasttext_args_t* handle, int minn);
CFASTTEXT_API int cft_args_get_maxn(fasttext_args_t* handle);
CFASTTEXT_API void cft_args_set_maxn(fasttext_args_t* handle, int maxn);
CFASTTEXT_API int cft_args_get_t(fasttext_args_t* handle);
CFASTTEXT_API void cft_args_set_t(fasttext_args_t* handle, int t);
CFASTTEXT_API int cft_args_get_verbose(fasttext_args_t* handle);
CFASTTEXT_API void cft_args_set_verbose(fasttext_args_t* handle, int verbose);
CFASTTEXT_API const char* cft_args_get_label(fasttext_args_t* handle);
CFASTTEXT_API void cft_args_set_label(fasttext_args_t* handle, const char* label);
CFASTTEXT_API bool cft_args_get_save_output(fasttext_args_t* handle);
CFASTTEXT_API void cft_args_set_save_output(fasttext_args_t* handle, bool save_output);
CFASTTEXT_API bool cft_args_get_qout(fasttext_args_t* handle);
CFASTTEXT_API void cft_args_set_qout(fasttext_args_t* handle, bool qout);
CFASTTEXT_API bool cft_args_get_retrain(fasttext_args_t* handle);
CFASTTEXT_API void cft_args_set_retrain(fasttext_args_t* handle, bool retrain);
CFASTTEXT_API bool cft_args_get_qnorm(fasttext_args_t* handle);
CFASTTEXT_API void cft_args_set_qnorm(fasttext_args_t* handle, bool qnorm);
CFASTTEXT_API size_t cft_args_get_cutoff(fasttext_args_t* handle);
CFASTTEXT_API void cft_args_set_cutoff(fasttext_args_t* handle, size_t cutoff);
CFASTTEXT_API size_t cft_args_get_dsub(fasttext_args_t* handle);
CFASTTEXT_API void cft_args_set_dsub(fasttext_args_t* handle, size_t dsub);
CFASTTEXT_API const char* cft_args_get_pretrained_vectors(fasttext_args_t* handle);
CFASTTEXT_API void cft_args_set_pretrained_vectors(fasttext_args_t* handle, const char* pretrained_vectors);
CFASTTEXT_API int cft_args_get_seed(fasttext_args_t* handle);
CFASTTEXT_API void cft_args_set_seed(fasttext_args_t* handle, int seed);
CFASTTEXT_API const char* cft_args_get_autotune_validation_file(fasttext_args_t* handle);
CFASTTEXT_API void cft_args_set_autotune_validation_file(fasttext_args_t* handle, const char* autotune_validation_file);
CFASTTEXT_API metric_name_t cft_args_get_autotune_metric(fasttext_args_t* handle);
CFASTTEXT_API const char* cft_args_get_autotune_metric_label(fasttext_args_t *handle);
CFASTTEXT_API int cft_args_get_autotune_predictions(fasttext_args_t* handle);
CFASTTEXT_API void cft_args_set_autotune_predictions(fasttext_args_t* handle, int autotune_predictions);
CFASTTEXT_API int cft_args_get_autotune_duration(fasttext_args_t* handle);
CFASTTEXT_API void cft_args_set_autotune_duration(fasttext_args_t* handle, int autotune_duration);
CFASTTEXT_API int64_t cft_args_get_autotune_model_size(fasttext_args_t* handle);
CFASTTEXT_API bool cft_args_has_autotune(fasttext_args_t* handle);
CFASTTEXT_API void cft_args_print_help(fasttext_args_t* handle);
CFASTTEXT_API void cft_args_print_basic_help(fasttext_args_t* handle);
CFASTTEXT_API void cft_args_print_dictionary_help(fasttext_args_t* handle);
CFASTTEXT_API void cft_args_print_training_help(fasttext_args_t* handle);
CFASTTEXT_API void cft_args_print_quantization_help(fasttext_args_t* handle);
CFASTTEXT_API void cft_args_print_autotune_help(fasttext_args_t* handle);

/* fasttext APIs */
CFASTTEXT_API fasttext_t* cft_fasttext_new(void);
CFASTTEXT_API void cft_fasttext_free(fasttext_t* handle);
CFASTTEXT_API void cft_fasttext_load_model(fasttext_t* handle, const char* filename, char** errptr);
CFASTTEXT_API void cft_fasttext_save_model(fasttext_t* handle, const char* filename, char** errptr);
CFASTTEXT_API void cft_fasttext_save_output(fasttext_t* handle, const char* filename, char** errptr);
CFASTTEXT_API void cft_fasttext_save_vectors(fasttext_t* handle, const char* filename, char** errptr);
CFASTTEXT_API int cft_fasttext_get_dimension(fasttext_t* handle);
CFASTTEXT_API int32_t cft_fasttext_get_word_id(fasttext_t* handle, const char* word);
CFASTTEXT_API int32_t cft_fasttext_get_subword_id(fasttext_t* handle, const char* word);
CFASTTEXT_API bool cft_fasttext_is_quant(fasttext_t* handle);
CFASTTEXT_API void cft_fasttext_train(fasttext_t* handle, fasttext_args_t* args, char** errptr);
CFASTTEXT_API fasttext_predictions_t* cft_fasttext_predict(fasttext_t* handle, const char* text, int32_t k, float threshold, char** errptr);
CFASTTEXT_API fasttext_predictions_t* cft_fasttext_predict_on_words(fasttext_t* handle, fasttext_words_t* words, int32_t k, float threshold, char** errptr);
CFASTTEXT_API void cft_fasttext_predictions_free(fasttext_predictions_t* predictions);
CFASTTEXT_API void cft_fasttext_quantize(fasttext_t* handle, fasttext_args_t* args, char** errptr);
CFASTTEXT_API fasttext_tokens_t* cft_fasttext_tokenize(fasttext_t* handle, const char* text);
CFASTTEXT_API void cft_fasttext_abort(fasttext_t* handle);
CFASTTEXT_API void cft_fasttext_tokens_free(fasttext_tokens_t* tokens);

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
CFASTTEXT_API void cft_fasttext_get_sentence_vector(fasttext_t* handle, const char* sentence, float* buf);

/* autotune APIs */
CFASTTEXT_API fasttext_autotune_t* cft_autotune_new(fasttext_t *handle);
CFASTTEXT_API void cft_autotune_free(fasttext_autotune_t* handle);
CFASTTEXT_API void cft_autotune_train(fasttext_autotune_t* handle, fasttext_args_t* args);

#ifdef __cplusplus
}
#endif /* __cplusplus */

#endif /* CFASTTEXT_CFASTTEXT_H */
