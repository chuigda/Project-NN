#include "neuron.h"

#include <assert.h>
#include <stdlib.h>
#include "impl/util.h"

struct st_nn_imp_neuron {
  nn_transfer_fn *trans;
  size_t dim;
  float b;
  float w[];
};

_Static_assert(offsetof(nn_neuron_t, w) - offsetof(nn_neuron_t, b)
               == sizeof(float));

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

float *nn_neuron_w(nn_neuron_t *n) {
  assert(n);
  if (!n) {
    return NULL;
  }

  return &n->b;
}

void nn_neuron_prewarm(nn_neuron_t *n, float v) {
  assert(n);
  if (!n) {
    return;
  }

  for (size_t i = 0; i < n->dim; i++) {
    n->w[i] = v;
  }
  n->b = v;
}

void nn_neuron_prewarm_rand(nn_neuron_t *n, float l, float r) {
  assert(n && l < r);
  if (!(n && l < r)) {
    return;
  }

  for (size_t i = 0; i < n->dim; i++) {
    n->w[i] = nn_imp_randf(l, r);
  }
  n->b = nn_imp_randf(l, r);
}

void nn_neuron_train(nn_neuron_t *n, float *x, float e, float r) {
  assert(n && x && r > 0.0f);
  if (!(n && x && r > 0.0f)) {
    return;
  }

  float y = nn_neuron_test(n, x, NULL);
  float d = e - y;

  for (size_t i = 0; i < n->dim; i++) {
    n->w[i] += r * d * x[i];
  }
  n->b += r * d;
}

float nn_neuron_test(nn_neuron_t *n, float *x, float *activ) {
  assert(n);
  if (!n) {
    return 0.0;
  }

  float activation = n->b;
  for (size_t i = 0; i < n->dim; i++) {
    activation += n->w[i] * x[i];
  }

  if (activ) {
    *activ = activation;
  }
  return n->trans(activation);
}
