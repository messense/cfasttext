#include <iostream>
#include <sstream>
#include <algorithm>

#include <string.h>

#include "args.h"
#include "fasttext.h"

#include "cfasttext.h"

using namespace std;
using namespace fasttext;

#ifdef __cplusplus
extern "C" {
#endif /* __cplusplus */


static void save_error(char** errptr, const std::exception& e) {
    assert(errptr != nullptr);
    *errptr = strdup(e.what());
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

void cft_fasttext_load_model(fasttext_t* handle, const char* filename, char** errptr) {
    try {
        ((FastText*)handle)->loadModel(filename);
    } catch (const std::invalid_argument& e) {
        save_error(errptr, e);
    }
}

void cft_fasttext_save_model(fasttext_t* handle, char** errptr) {
    try {
        ((FastText*)handle)->saveModel();
    } catch (const std::invalid_argument& e) {
        save_error(errptr, e);
    }
}

void cft_fasttext_save_output(fasttext_t* handle, char** errptr) {
    try {
        ((FastText*)handle)->saveOutput();
    } catch (const std::invalid_argument& e) {
        save_error(errptr, e);
    }
}

void cft_fasttext_save_vectors(fasttext_t* handle, char** errptr) {
    try {
        ((FastText*)handle)->saveVectors();
    } catch (const std::invalid_argument& e) {
        save_error(errptr, e);
    }
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

void cft_fasttext_load_vectors(fasttext_t* handle, const char* filename, char** errptr) {
    try {
        ((FastText*)handle)->loadVectors(filename);
    } catch (const std::invalid_argument& e) {
        save_error(errptr, e);
    }
}

int32_t cft_fasttext_get_word_id(fasttext_t* handle, const char* word) {
    return ((FastText*)handle)->getWordId(word);
}

int32_t cft_fasttext_get_subword_id(fasttext_t* handle, const char* word) {
    return ((FastText*)handle)->getSubwordId(word);
}

void cft_fasttext_train(fasttext_t* handle, fasttext_args_t* args, char** errptr) {
    Args* a = (Args*)args;
    try {
        ((FastText*)handle)->train(*a);
    } catch (const std::invalid_argument& e) {
        save_error(errptr, e);
    }
}

fasttext_predictions_t* cft_fasttext_predict(fasttext_t* handle, const char* text, int32_t k, float threshold) {
    std::vector<std::pair<fasttext::real, int32_t>> predictions;
    std::vector<std::pair<fasttext::real, std::string>> all_predictions;
    std::stringstream ioss(text);
    std::shared_ptr<const fasttext::Dictionary> d = ((FastText*)handle)->getDictionary();
    std::vector<int32_t> words, labels;
    d->getLine(ioss, words, labels);
    ((FastText*)handle)->predict(k, words, predictions, threshold);
    std::transform(
        predictions.begin(),
        predictions.end(),
        std::back_inserter(all_predictions),
        [&d](const std::pair<fasttext::real, int32_t>& prediction) {
            return std::pair<fasttext::real, std::string>(
                std::exp(prediction.first),
                d->getLabel(prediction.second));
        });
    size_t len = all_predictions.size();
    fasttext_predictions_t* ret = static_cast<fasttext_predictions_t*>(malloc(sizeof(fasttext_predictions_t)));
    ret->length = len;
    fasttext_prediction_t* c_preds = static_cast<fasttext_prediction_t*>(malloc(sizeof(fasttext_prediction_t) * len));
    for (size_t i = 0; i < len; i++) {
        c_preds[i].label = strdup(all_predictions[i].second.c_str());
        c_preds[i].prob = all_predictions[i].first;
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

void cft_fasttext_quantize(fasttext_t* handle, fasttext_args_t* args, char** errptr) {
    Args* a = (Args*)args;
    try {
        ((FastText*)handle)->quantize(*a);
    } catch (const std::invalid_argument& e) {
        save_error(errptr, e);
    }
}

void cft_fasttext_get_word_vector(fasttext_t* handle, const char* word, float* buf) {
    Vector vec(((FastText*)handle)->getDimension());
    ((FastText*)handle)->getWordVector(vec, word);
    memcpy(buf, vec.data(), vec.size() * sizeof(real));
}


#ifdef __cplusplus
}
#endif /* __cplusplus */
