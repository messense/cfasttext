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


static void save_error(char** errptr, const std::exception& e) {
    assert(errptr != nullptr);
    *errptr = strdup(e.what());
}

void cft_str_free(char* s) {
    if (s != nullptr) {
        free(s);
    }
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

double cft_args_get_lr(fasttext_args_t* handle) {
    return ((Args*)handle)->lr;
}

void cft_args_set_lr(fasttext_args_t* handle, double lr) {
    ((Args*)handle)->lr = lr;
}

int cft_args_get_lr_update_rate(fasttext_args_t* handle) {
    return ((Args*)handle)->lrUpdateRate;
}

void cft_args_set_lr_update_rate(fasttext_args_t* handle, int rate) {
    ((Args*)handle)->lrUpdateRate = rate;
}

int cft_args_get_dim(fasttext_args_t* handle) {
    return ((Args*)handle)->dim;
}

void cft_args_set_dim(fasttext_args_t* handle, int dim) {
    ((Args*)handle)->dim = dim;
}

int cft_args_get_ws(fasttext_args_t* handle) {
    return ((Args*)handle)->ws;
}

void cft_args_set_ws(fasttext_args_t* handle, int ws) {
    ((Args*)handle)->ws = ws;
}

int cft_args_get_epoch(fasttext_args_t* handle) {
    return ((Args*)handle)->epoch;
}

void cft_args_set_epoch(fasttext_args_t* handle, int epoch) {
    ((Args*)handle)->epoch = epoch;
}

int cft_args_get_thread(fasttext_args_t* handle) {
    return ((Args*)handle)->thread;
}

void cft_args_set_thread(fasttext_args_t* handle, int thread) {
    ((Args*)handle)->thread = thread;
}

model_name_t cft_args_get_model(fasttext_args_t* handle) {
    return static_cast<model_name_t>(static_cast<int>(((Args*)handle)->model));
}

void cft_args_set_model(fasttext_args_t* handle, model_name_t model) {
    ((Args*)handle)->model = static_cast<model_name>(model);
}

loss_name_t cft_args_get_loss(fasttext_args_t* handle) {
    return static_cast<loss_name_t>(static_cast<int>(((Args*)handle)->loss));
}

void cft_args_set_loss(fasttext_args_t* handle, loss_name_t loss) {
    ((Args*)handle)->loss = static_cast<loss_name>(loss);
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

void cft_fasttext_save_model(fasttext_t* handle, const char* filename, char** errptr) {
    try {
        ((FastText*)handle)->saveModel(filename);
    } catch (const std::invalid_argument& e) {
        save_error(errptr, e);
    }
}

void cft_fasttext_save_output(fasttext_t* handle, const char* filename, char** errptr) {
    try {
        ((FastText*)handle)->saveOutput(filename);
    } catch (const std::invalid_argument& e) {
        save_error(errptr, e);
    }
}

void cft_fasttext_save_vectors(fasttext_t* handle, const char* filename, char** errptr) {
    try {
        ((FastText*)handle)->saveVectors(filename);
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
    std::vector<std::pair<fasttext::real, std::string>> predictions;
    std::stringstream ioss(text);
    ((FastText*)handle)->predictLine(ioss, predictions, k, threshold);
    size_t len = predictions.size();
    fasttext_predictions_t* ret = static_cast<fasttext_predictions_t*>(malloc(sizeof(fasttext_predictions_t)));
    ret->length = len;
    fasttext_prediction_t* c_preds = static_cast<fasttext_prediction_t*>(malloc(sizeof(fasttext_prediction_t) * len));
    for (size_t i = 0; i < len; i++) {
        c_preds[i].label = strdup(predictions[i].second.c_str());
        c_preds[i].prob = predictions[i].first;
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

void cft_fasttext_get_sentence_vector(fasttext_t* handle, const char* sentence, float* buf) {
    Vector vec(((FastText*)handle)->getDimension());
    std::stringstream ioss(sentence);
    ((FastText*)handle)->getSentenceVector(ioss, vec);
    memcpy(buf, vec.data(), vec.size() * sizeof(real));
}

fasttext_tokens_t* cft_fasttext_tokenize(fasttext_t* handle, const char* text) {
    std::vector<std::string> text_split;
    std::shared_ptr<const fasttext::Dictionary> d = ((FastText*)handle)->getDictionary();
    std::stringstream ioss(text);
    std::string token;
    while (!ioss.eof()) {
        while (d->readWord(ioss, token)) {
        text_split.push_back(token);
        }
    }
    size_t len = text_split.size();
    fasttext_tokens_t* ret = static_cast<fasttext_tokens_t*>(malloc(sizeof(fasttext_tokens_t)));
    ret->length = len;
    char** tokens = static_cast<char**>(malloc(sizeof(char*) * len));
    for (size_t i = 0; i < len; i++) {
        tokens[i] = strdup(text_split[i].c_str());
    }
    ret->tokens = tokens;
    return ret;
}

void cft_fasttext_tokens_free(fasttext_tokens_t* tokens) {
    for (size_t i = 0; i < tokens->length; i++) {
        free(tokens->tokens[i]);
    }
    free(tokens->tokens);
    free(tokens);
}

#ifdef __cplusplus
}
#endif /* __cplusplus */
