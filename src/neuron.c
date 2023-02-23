#include "neuron.h"

#include <assert.h>
#include <stdlib.h>

struct st_nn_imp_neuron {
  nn_transfer_fn *trans;
  size_t dim;
  float b;
  float w[];
};

nn_neuron_t *nn_neuron_create(size_t dim, nn_transfer_fn *trans) {
  assert(dim && trans);
  if (!(dim && trans)) {
    return NULL;
  }

  nn_neuron_t *r = malloc(sizeof(nn_neuron_t) + dim * sizeof(float));
  if (!r) {
    return NULL;
  }

  r->trans = trans;
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

size_t nn_neuron_dim(nn_neuron_t *n) {
  assert(n);
  if (!n) {
    return 0;
  }

  return n->dim;
}

void nn_neuron_train(nn_neuron_t *n, float *x, float e, float r) {
  assert(n && x && r > 0.0f);
  if (!(n && x && r > 0.0f)) {
    return;
  }

  float y = nn_neuron_test(n, x);
  float d = e - y;

  for (size_t i = 0; i < n->dim; i++) {
    n->w[i] += r * d * x[i];
  }
  n->b += r * d;
}

float nn_neuron_test(nn_neuron_t *n, float *x) {
  assert(n);
  if (!n) {
    return 0.0;
  }

  float activation = n->b;
  for (size_t i = 0; i < n->dim; i++) {
    activation += n->w[i] * x[i];
  }
  return n->trans(activation);
}
