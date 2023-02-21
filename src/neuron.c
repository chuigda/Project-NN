#include "neuron.h"

#include <assert.h>
#include <stdlib.h>

nn_neuron_t *nn_neuron_create(size_t dim, nn_threshold_fn *thres) {
  nn_neuron_t *r = malloc(sizeof(nn_neuron_t) + dim * sizeof(float));
  if (!r) {
    return NULL;
  }

  r->thres = thres;
  r->dim = dim;
  r->b = 0.0;
  for (size_t i = 0; i < dim; i++) {
    r->w[i] = 0.0;
  }

  return r;
}

void nn_neuron_destroy(nn_neuron_t *n) {
  free(n);
}

void nn_neuron_train(nn_neuron_t *n, float *x, float e, float r) {
  float y = nn_neuron_test(n, x);
  float d = e - y;
  for (size_t i = 0; i < n->dim; i++) {
    n->w[i] += r * d * x[i];
  }
  n->b += r * d;
}

float nn_neuron_test(nn_neuron_t *n, float *x) {
  float activation = n->b;
  for (size_t i = 0; i < n->dim; i++) {
    activation += n->w[i] * x[i];
  }
  return n->thres(activation);
}
