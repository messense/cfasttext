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

fasttext_args_t cft_args_parse(int argc, char** argv) {
    std::vector<std::string> args(argv, argv + argc);
    Args* handle = new Args();
    handle->parseArgs(args);
    return (fasttext_args_t)handle;
}

void cft_args_free(fasttext_args_t handle) {
    Args* x = (Args*) handle;
    delete x;
}

fasttext_t cft_fasttext_new(void) {
    return (fasttext_t)(new FastText());
}

void cft_fasttext_free(fasttext_t handle) {
    FastText* x = (FastText*)handle;
    delete x;
}

void cft_fasttext_load_model(fasttext_t handle, const char* filename) {
    ((FastText*)handle)->loadModel(filename);
}

void cft_fasttext_save_model(fasttext_t handle) {
    ((FastText*)handle)->saveModel();
}

void cft_fasttext_save_output(fasttext_t handle) {
    ((FastText*)handle)->saveOutput();
}

void cft_fasttext_save_vectors(fasttext_t handle) {
    ((FastText*)handle)->saveVectors();
}

int cft_fasttext_get_dimension(fasttext_t handle) {
    return ((FastText*)handle)->getDimension();
}

bool cft_fasttext_is_quant(fasttext_t handle) {
    return ((FastText*)handle)->isQuant();
}

void cft_fasttext_analogies(fasttext_t handle, int32_t k) {
    ((FastText*)handle)->analogies(k);
}

void cft_fasttext_train_thread(fasttext_t handle, int32_t n) {
    ((FastText*)handle)->trainThread(n);
}

void cft_fasttext_load_vectors(fasttext_t handle, const char* filename) {
    ((FastText*)handle)->loadVectors(filename);
}

int32_t cft_fasttext_get_word_id(fasttext_t handle, const char* word) {
    return ((FastText*)handle)->getWordId(word);
}

int32_t cft_fasttext_get_subword_id(fasttext_t handle, const char* word) {
    return ((FastText*)handle)->getSubwordId(word);
}

void cft_fasttext_train(fasttext_t handle, int argc, char** argv) {
    std::vector<std::string> args(argv, argv + argc);
    Args a = Args();
    a.parseArgs(args);
    ((FastText*)handle)->train(a);
}

#ifdef __cplusplus
}
#endif /* __cplusplus */
