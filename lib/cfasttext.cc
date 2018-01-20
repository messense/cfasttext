#include <iostream>
#include <sstream>

#include <string.h>

#include "args.h"
#include "fasttext.h"

#include "cfasttext.h"

using namespace std;
using namespace fasttext;

#ifdef __cplusplus
extern "C" {
#endif /* __cplusplus */

void cft_str_free(char* s) {
    free(s);
}

fasttext_args_t* cft_args_new(void) {
    return (fasttext_args_t*)(new Args());
}

void cft_args_parse(fasttext_args_t* handle, int argc, char** argv) {
    std::vector<std::string> args(argv, argv + argc);
    ((Args*)handle)->parseArgs(args);
}

void cft_args_free(fasttext_args_t* handle) {
    Args* x = (Args*)handle;
    delete x;
}

const char* cft_args_get_input(fasttext_args_t* handle) {
    return ((Args*)handle)->input.c_str();
}

void cft_args_set_input(fasttext_args_t* handle, const char* input) {
    ((Args*)handle)->input = input;
}

const char* cft_args_get_output(fasttext_args_t* handle) {
    return ((Args*)handle)->output.c_str();
}

void cft_args_set_output(fasttext_args_t* handle, const char* output) {
    ((Args*)handle)->output = output;
}

fasttext_t* cft_fasttext_new(void) {
    return (fasttext_t*)(new FastText());
}

void cft_fasttext_free(fasttext_t* handle) {
    FastText* x = (FastText*)handle;
    delete x;
}

void cft_fasttext_load_model(fasttext_t* handle, const char* filename) {
    ((FastText*)handle)->loadModel(filename);
}

void cft_fasttext_save_model(fasttext_t* handle) {
    ((FastText*)handle)->saveModel();
}

void cft_fasttext_save_output(fasttext_t* handle) {
    ((FastText*)handle)->saveOutput();
}

void cft_fasttext_save_vectors(fasttext_t* handle) {
    ((FastText*)handle)->saveVectors();
}

int cft_fasttext_get_dimension(fasttext_t* handle) {
    return ((FastText*)handle)->getDimension();
}

bool cft_fasttext_is_quant(fasttext_t* handle) {
    return ((FastText*)handle)->isQuant();
}

void cft_fasttext_analogies(fasttext_t* handle, int32_t k) {
    ((FastText*)handle)->analogies(k);
}

void cft_fasttext_train_thread(fasttext_t* handle, int32_t n) {
    ((FastText*)handle)->trainThread(n);
}

void cft_fasttext_load_vectors(fasttext_t* handle, const char* filename) {
    ((FastText*)handle)->loadVectors(filename);
}

int32_t cft_fasttext_get_word_id(fasttext_t* handle, const char* word) {
    return ((FastText*)handle)->getWordId(word);
}

int32_t cft_fasttext_get_subword_id(fasttext_t* handle, const char* word) {
    return ((FastText*)handle)->getSubwordId(word);
}

void cft_fasttext_train(fasttext_t* handle, fasttext_args_t* args) {
    Args* a = (Args*)args;
    ((FastText*)handle)->train(*a);
}

fasttext_predictions_t* cft_fasttext_predict(fasttext_t* handle, const char* text, int32_t k, float threshold) {
    std::vector<std::pair<fasttext::real, std::string>> predictions;
    std::stringstream ioss(text);
    ((FastText*)handle)->predict(ioss, k, predictions, threshold);
    size_t len = predictions.size();
    fasttext_predictions_t* ret = static_cast<fasttext_predictions_t*>(malloc(sizeof(fasttext_predictions_t)));
    ret->length = len;
    fasttext_prediction_t* c_preds = static_cast<fasttext_prediction_t*>(malloc(sizeof(fasttext_prediction_t) * len));
    for (size_t i = 0; i < len; i++) {
        c_preds[i].label = strdup(predictions[i].second.c_str());
        c_preds[i].prob = std::exp(predictions[i].first);
    }
    ret->predictions = c_preds;
    return ret;
}

void cft_fasttext_predictions_free(fasttext_predictions_t* predictions) {
    for (size_t i = 0; i < predictions->length; i++) {
        fasttext_prediction_t pred = predictions->predictions[i];
        free(pred.label);
    }
    free(predictions->predictions);
    free(predictions);
}

void cft_fasttext_quantize(fasttext_t* handle, fasttext_args_t* args) {
    Args* a = (Args*)args;
    ((FastText*)handle)->quantize(*a);
}

#ifdef __cplusplus
}
#endif /* __cplusplus */
