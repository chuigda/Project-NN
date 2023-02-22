#include "fnn.h"
#include "neuron.h"

#include <assert.h>
#include <stdlib.h>

typedef struct st_fnn_layer {
    struct st_fnn_layer *prev;
    struct st_fnn_layer *next;

    size_t n_cnt;
    nn_neuron_t *n[];
} nn_fnn_layer_t;

typedef struct {
    nn_fnn_layer_t *layers;
    nn_fnn_layer_t *last;
} nn_fnn_impl_t;

nn_fnn_t *nn_fnn_create() {
    nn_fnn_impl_t *r = malloc(sizeof(nn_fnn_layer_t));
    if (r) {
        r->layers = NULL;
        r->last = NULL;
    }
    return (nn_fnn_t*)r;
}

void nn_fnn_destroy(nn_fnn_t *fnn) {
    if (!fnn) {
        return;
    }

    nn_fnn_impl_t *impl = (nn_fnn_impl_t*)fnn;

    for (nn_fnn_layer_t *layer = impl->layers; layer != NULL; /* nop */) {
        for (size_t i = 0; i < layer->n_cnt; i++) {
            nn_neuron_destroy(layer->n[i]);
        }

        nn_fnn_layer_t *this_layer = layer;
        layer = this_layer->next;
        free(this_layer);
    }
}

_Bool nn_fnn_add_layer(nn_fnn_t *fnn,
                       size_t in_dim,
                       size_t n_cnt,
                       nn_threshold_fn *thres) {
    assert(fnn && in_dim && n_cnt && thres);
    if (!(fnn && in_dim && n_cnt && thres)) {
        return 0;
    }

    nn_fnn_layer_t *layer = 
        malloc(sizeof(nn_fnn_layer_t) + n_cnt * sizeof(nn_neuron_t));
    if (!layer) {
        return 0;
    }

    layer->next = NULL;
    layer->n_cnt = n_cnt;
    for (size_t i = 0; i < n_cnt; i++) {
        nn_neuron_t *n = nn_neuron_create(in_dim, thres);
    }

    return 1;
}
