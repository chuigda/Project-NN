#ifndef PROJECT_NN_FNN_H
#define PROJECT_NN_FNN_H

#include "neuron.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {} nn_fnn_t;

nn_fnn_t *nn_fnn_create();

void nn_fnn_destroy(nn_fnn_t *fnn);

_Bool nn_fnn_add_layer(nn_fnn_t *fnn,
                       size_t in_dim,
                       size_t n_cnt,
                       nn_threshold_fn *thres);

void nn_fnn_train(nn_fnn_t *fnn, float *x, float e, float r);

float nn_fnn_test(nn_fnn_t *fnn, float *x);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* PROJECT_NN_FNN_H */
