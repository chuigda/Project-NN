#include "neuron.h"

#include <assert.h>
#include <stdlib.h>

typedef struct {
  nn_transfer_fn *trans;
  size_t dim;
  float b;
  float w[];
} nn_neuron_impl_t;

nn_neuron_t *nn_neuron_create(size_t dim, nn_transfer_fn *trans) {
  assert(dim && trans);
  if (!(dim && trans)) {
    return NULL;
  }

  nn_neuron_impl_t *r = malloc(sizeof(nn_neuron_impl_t) + dim * sizeof(float));
  if (!r) {
    return NULL;
  }

  r->trans = trans;
  r->dim = dim;
  r->b = 0.0;
  for (size_t i = 0; i < dim; i++) {
    r->w[i] = 0.0;
  }

  return (nn_neuron_t*)r;
}

void nn_neuron_destroy(nn_neuron_t *n) {
  free(n);
}

size_t nn_neuron_dim(nn_neuron_t *n) {
  assert(n);
  if (!n) {
    return 0;
  }

  nn_neuron_impl_t *impl = (nn_neuron_impl_t*)n;
  return impl->dim;
}

void nn_neuron_train(nn_neuron_t *n, float *x, float e, float r) {
  assert(n && x);
  if (!(n && x)) {
    return;
  }

  float y = nn_neuron_test(n, x);
  float d = e - y;

  nn_neuron_impl_t *impl = (nn_neuron_impl_t*)n;
  for (size_t i = 0; i < impl->dim; i++) {
    impl->w[i] += r * d * x[i];
  }
  impl->b += r * d;
}

float nn_neuron_test(nn_neuron_t *n, float *x) {
  assert(n);
  if (!n) {
    return 0.0;
  }

  nn_neuron_impl_t *impl = (nn_neuron_impl_t*)n;

  float activation = impl->b;
  for (size_t i = 0; i < impl->dim; i++) {
    activation += impl->w[i] * x[i];
  }
  return impl->trans(activation);
}
