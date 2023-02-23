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
    
    float *buffer1;
    float *buffer2;
} nn_fnn_impl_t;

nn_fnn_t *nn_fnn_create() {
    nn_fnn_impl_t *r = malloc(sizeof(nn_fnn_impl_t));
    if (r) {
        r->layers = NULL;
        r->last = NULL;
        r->buffer1 = NULL;
        r->buffer2 = NULL;
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

    if (impl->buffer1) {
        free(impl->buffer1);
    }

    if (impl->buffer2) {
        free(impl->buffer2);
    }
}

nn_error_t nn_fnn_add_layer(nn_fnn_t *fnn,
                            size_t in_dim,
                            size_t n_cnt,
                            nn_transfer_fn *trans) {
    assert(fnn && in_dim && n_cnt && trans);
    if (!(fnn && in_dim && n_cnt && trans)) {
        return NN_INVALID_VALUE;
    }

    nn_fnn_impl_t *impl = (nn_fnn_impl_t*)fnn;
    if (impl->last && impl->last->n_cnt != in_dim) {
        return NN_INVALID_VALUE;
    }

    if (impl->buffer1) {
        return NN_INVALID_OPERATION;
    }

    nn_fnn_layer_t *layer = 
        malloc(sizeof(nn_fnn_layer_t) + n_cnt * sizeof(nn_neuron_t*));
    if (!layer) {
        return NN_OUT_OF_MEMORY;
    }

    layer->next = NULL;
    layer->n_cnt = n_cnt;
    for (size_t i = 0; i < n_cnt; i++) {
        nn_neuron_t *n = nn_neuron_create(in_dim, trans);
        if (!n) {
            for (size_t j = 0; j < i; j++) {
                nn_neuron_destroy(layer->n[j]);
            }
            free(layer);
            return NN_OUT_OF_MEMORY;
        }
        layer->n[i] = n;
    }

    if (!impl->layers) {
        layer->prev = NULL;
        impl->layers = layer;
        impl->last = layer;
    } else {
        layer->prev = impl->last;
        impl->last->next = layer;
        impl->last = layer;
    }

    return NN_NO_ERROR;
}

size_t nn_fnn_in_dim(nn_fnn_t *fnn) {
    assert(fnn);
    if (!fnn) {
        return 0;
    }

    nn_fnn_impl_t *impl = (nn_fnn_impl_t*)fnn;
    if (impl->layers) {
        return nn_neuron_dim(impl->layers->n[0]);
    } else {
        return 0;
    }
}

size_t nn_fnn_out_dim(nn_fnn_t *fnn) {
    assert(fnn);
    if (!fnn) {
        return 0;
    }

    nn_fnn_impl_t *impl = (nn_fnn_impl_t*)fnn;
    if (impl->last) {
        return impl->last->n_cnt;
    } else {
        return 0;
    }
}

nn_error_t nn_fnn_fin(nn_fnn_t *fnn) {
    assert(fnn);
    if (!fnn) {
        return NN_INVALID_VALUE;
    }

    nn_fnn_impl_t *impl = (nn_fnn_impl_t*)fnn;
    if (!impl->layers) {
        return NN_INVALID_OPERATION;
    }

    size_t max_dim = 0;
    for (nn_fnn_layer_t *layer = impl->layers;
         layer != NULL;
         layer = layer->next)
    {
        size_t neuron_in_dim = nn_neuron_dim(layer->n[0]);
        if (neuron_in_dim > max_dim) {
            max_dim = neuron_in_dim;
        }

        if (layer->n_cnt > max_dim) {
            max_dim = layer->n_cnt;
        }
    }

    float *buffer1 = malloc(max_dim * sizeof(float));
    float *buffer2 = malloc(max_dim * sizeof(float));
    if (!(buffer1 && buffer2)) {
        free(buffer1);
        free(buffer2);
        return NN_OUT_OF_MEMORY;
    }

    impl->buffer1 = buffer1;
    impl->buffer2 = buffer2;
    return NN_NO_ERROR;
}

/*
nn_error_t nn_fnn_test(nn_fnn_t *fnn, float *x, float *y) {
    assert(fnn && x && y);
    if (!(fnn && x && y)) {
        return NN_INVALID_VALUE;
    }

    nn_fnn_impl_t *impl = (nn_fnn_impl_t*)fnn;
    if (!impl->layers) {
        return NN_INVALID_OPERATION;
    }

    

    return NN_NO_ERROR;
}
*/
