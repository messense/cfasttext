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

// cfasttext_args_t cfasttext_parse_args(int argc, char** argv) {
//     std::vector<std::string> args(argv, argv + argc);
//     Args* handle = new Args();
//     handle->parseArgs(args);
//     return (cfasttext_args_t)handle;
// }

cfasttext_t cfasttext_new(void) {
    return (cfasttext_t)(new FastText());
}

void cfasttext_free(cfasttext_t handle) {
    FastText* x = (FastText*)handle;
    delete x;
}

void cfasttext_load_model(cfasttext_t handle, const char* filename) {
    ((FastText*)handle)->loadModel(filename);
}

void cfasttext_save_model(cfasttext_t handle) {
    ((FastText*)handle)->saveModel();
}

void cfasttext_save_output(cfasttext_t handle) {
    ((FastText*)handle)->saveOutput();
}

void cfasttext_save_vectors(cfasttext_t handle) {
    ((FastText*)handle)->saveVectors();
}

int cfasttext_get_dimension(cfasttext_t handle) {
    return ((FastText*)handle)->getDimension();
}

bool cfasttext_is_quant(cfasttext_t handle) {
    return ((FastText*)handle)->isQuant();
}

void cfasttext_analogies(cfasttext_t handle, int32_t k) {
    ((FastText*)handle)->analogies(k);
}

void cfasttext_train_thread(cfasttext_t handle, int32_t n) {
    ((FastText*)handle)->trainThread(n);
}

void cfasttext_load_vectors(cfasttext_t handle, const char* filename) {
    ((FastText*)handle)->loadVectors(filename);
}

int32_t cfasttext_get_word_id(cfasttext_t handle, const char* word) {
    return ((FastText*)handle)->getWordId(word);
}

int32_t cfasttext_get_subword_id(cfasttext_t handle, const char* word) {
    return ((FastText*)handle)->getSubwordId(word);
}

void cfasttext_train(cfasttext_t handle, int argc, char** argv) {
    std::vector<std::string> args(argv, argv + argc);
    Args a = Args();
    a.parseArgs(args);
    ((FastText*)handle)->train(a);
}

#ifdef __cplusplus
}
#endif /* __cplusplus */
