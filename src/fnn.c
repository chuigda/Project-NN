#include "fnn.h"

#include <assert.h>
#include <stdlib.h>
#include <string.h>

#include "deriv.h"
#include "neuron.h"

typedef struct st_fnn_layer {
    struct st_fnn_layer *prev;
    struct st_fnn_layer *next;

    nn_deriv_fn *deriv;

    size_t n_cnt;
    nn_neuron_t **n;
    float *output;
    float *delta;

    _Alignas(uint64_t) uint8_t buffer[];
} nn_fnn_layer_t;

struct st_nn_imp_fnn {
    nn_fnn_layer_t *layers;
    nn_fnn_layer_t *last;
    
    float *buffer;
    float *buf_ptr[4];
};

static void nn_imp_fnn_test(nn_fnn_t *fnn, float *x, float *y);

nn_fnn_t *nn_fnn_create() {
    nn_fnn_t *r = malloc(sizeof(nn_fnn_t));
    if (r) {
        r->layers = NULL;
        r->last = NULL;
        r->buffer = NULL;
        memset(&r->buf_ptr, 0, sizeof(r->buf_ptr));
    }
    return r;
}

void nn_fnn_destroy(nn_fnn_t *fnn) {
    if (!fnn) {
        return;
    }

    for (nn_fnn_layer_t *layer = fnn->layers;
         layer != NULL;
         /* nop */)
    {
        for (size_t i = 0; i < layer->n_cnt; i++) {
            nn_neuron_destroy(layer->n[i]);
        }

        nn_fnn_layer_t *this_layer = layer;
        layer = this_layer->next;
        free(this_layer);
    }

    free(fnn->buffer);
    free(fnn);
}

nn_error_t nn_fnn_add_layer(nn_fnn_t *fnn,
                            size_t in_dim,
                            size_t n_cnt,
                            nn_transfer_fn *trans) 
{
    assert(fnn && in_dim && n_cnt && trans);
    if (!(fnn && in_dim && n_cnt && trans)) {
        return NN_INVALID_VALUE;
    }

    nn_deriv_fn *deriv = nn_deriv(trans);
    if (!deriv) {
        return NN_INVALID_VALUE;
    }

    if (fnn->last && fnn->last->n_cnt != in_dim) {
        return NN_INVALID_VALUE;
    }

    if (fnn->buffer) {
        return NN_INVALID_OPERATION;
    }

    size_t extra_size = 
        n_cnt * (sizeof(nn_neuron_t*) + 2 * sizeof(float));

    nn_fnn_layer_t *layer = malloc(sizeof(nn_fnn_layer_t) + extra_size);
    if (!layer) {
        return NN_OUT_OF_MEMORY;
    }

    size_t output_offset = n_cnt * sizeof(nn_neuron_t*);
    size_t delta_offset = output_offset + n_cnt * sizeof(float);
    layer->next = NULL;
    layer->deriv = deriv;
    layer->n_cnt = n_cnt;
    layer->n = (nn_neuron_t**)layer->buffer;
    layer->output = (float*)(layer->buffer + output_offset);
    layer->delta = (float*)(layer->buffer + delta_offset);
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

    if (!fnn->layers) {
        layer->prev = NULL;
        fnn->layers = layer;
        fnn->last = layer;
    } else {
        layer->prev = fnn->last;
        fnn->last->next = layer;
        fnn->last = layer;
    }

    return NN_NO_ERROR;
}

nn_error_t nn_fnn_prewarm(nn_fnn_t *fnn, float v) {
    assert(fnn);
    if (!fnn) {
        return NN_INVALID_VALUE;
    }

    if (!fnn->last) {
        return NN_INVALID_OPERATION;
    }

    for (size_t i = 0; i < fnn->last->n_cnt; i++) {
        nn_neuron_prewarm(fnn->last->n[i], v);
    }

    return NN_NO_ERROR;
}

nn_error_t nn_fnn_prewarm_rand(nn_fnn_t *fnn, float l, float r) {
    assert(fnn && l < r);
    if (!(fnn && l < r)) {
        return NN_INVALID_VALUE;
    }

    if (!fnn->last) {
        return NN_INVALID_OPERATION;
    }

    for (size_t i = 0; i < fnn->last->n_cnt; i++) {
        nn_neuron_prewarm_rand(fnn->last->n[i], l, r);
    }

    return NN_NO_ERROR;
}

size_t nn_fnn_in_dim(nn_fnn_t *fnn) {
    assert(fnn);
    if (!fnn) {
        return 0;
    }

    if (fnn->layers) {
        return nn_neuron_dim(fnn->layers->n[0]);
    } else {
        return 0;
    }
}

size_t nn_fnn_out_dim(nn_fnn_t *fnn) {
    assert(fnn);
    if (!fnn) {
        return 0;
    }

    if (fnn->last) {
        return fnn->last->n_cnt;
    } else {
        return 0;
    }
}

nn_error_t nn_fnn_fin(nn_fnn_t *fnn) {
    assert(fnn);
    if (!fnn) {
        return NN_INVALID_VALUE;
    }

    if (!fnn->layers) {
        return NN_INVALID_OPERATION;
    }

    size_t max_dim = 0;
    for (nn_fnn_layer_t *layer = fnn->layers;
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

    float *buffer = malloc(max_dim * 8 * sizeof(float));
    if (!buffer) {
        return NN_OUT_OF_MEMORY;
    }

    fnn->buffer = buffer;
    for (size_t i = 0; i < 4; i++) {
        fnn->buf_ptr[i] = buffer + i * max_dim;
    }

    return NN_NO_ERROR;
}

nn_error_t nn_fnn_train(nn_fnn_t *fnn, float *x, float *e, float r) {
    assert(fnn && x && e && r > 0.0f);
    if (!(fnn && x && e && r > 0.0f)) {
        return NN_INVALID_VALUE;
    }

    if (!(fnn->layers && fnn->buffer)) {
        return NN_INVALID_OPERATION;
    }

    return NN_UNIMPLEMENTED;
}

nn_error_t nn_fnn_test(nn_fnn_t *fnn, float *x, float *y) {
    assert(fnn && x && y);
    if (!(fnn && x && y)) {
        return NN_INVALID_VALUE;
    }

    if (!(fnn->layers && fnn->buffer)) {
        return NN_INVALID_OPERATION;
    }

    nn_imp_fnn_test(fnn, x, y);
    return NN_NO_ERROR;
}

static void nn_imp_fnn_test(nn_fnn_t *fnn, float *x, float *y) {
    float *inbuf = x;
    for (nn_fnn_layer_t *layer = fnn->layers;
         layer != NULL;
         layer = layer->next)
    {
        float *outbuf;
        if (layer->next == NULL && y) {
            outbuf = y;
        } else {
            outbuf = layer->output;
        }

        for (size_t i = 0; i < layer->n_cnt; i++) {
            outbuf[i] = nn_neuron_test(layer->n[i], inbuf);
        }
        inbuf = outbuf;
    }
}
