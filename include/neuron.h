#ifndef PROJECT_NN_NEURON_H
#define PROJECT_NN_NEURON_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef float (nn_threshold_fn)(float value);

typedef struct {
  nn_threshold_fn *thres;
  size_t dim;
  float b;
  float w[];
} nn_neuron_t;

nn_neuron_t *nn_neuron_create(size_t dim, nn_threshold_fn *thres);

void nn_neuron_destroy(nn_neuron_t* n);

void nn_neuron_train(nn_neuron_t *n, float *x, float e, float r);

float nn_neuron_test(nn_neuron_t *n, float *x);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* PROJECT_NN_NEURON_H */
