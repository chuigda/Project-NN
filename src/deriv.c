#include "deriv.h"

#include <assert.h>
#include <stdlib.h>
#include <math.h>

#ifndef NN_DERIV_ENTRY_CAP
#define NN_DERIV_ENTRY_CAP 128
#endif

#ifndef NN_NUM_DERIV_STEP
#define NN_NUM_DERIV_STEP 1.0e-5f
#endif

float nn_deriv_linear(nn_transfer_fn *trans, float x, float y) {
    (void)trans;
    (void)x;
    (void)y;

    return 1;
}

float nn_deriv_sigmoid(nn_transfer_fn *trans, float x, float y) {
    (void)trans;
    (void)x;

    return y * (1.0f - y);
}

float nn_deriv_tanh(nn_transfer_fn *trans, float x, float y) {
    (void)trans;
    (void)y;

    float denom = expf(-x) + expf(x);
    denom = denom * denom;

    return 4.0 / denom;
}

float nn_deriv_relu(nn_transfer_fn *trans, float x, float y) {
    (void)trans;
    (void)y;
    if (x <= 0) {
        return 0;
    } else {
        return 1;
    }
}

float nn_deriv_num(nn_transfer_fn *trans, float x, float y) {
    (void)y;

    float step = NN_DERIV_ENTRY_CAP;
    float x_abs = fabs(x);
    if (x_abs >= 1.0) {
        step *= x_abs;
    }

    float l = trans(x - step / 2);
    float r = trans(x + step / 2);
    return (r - l) / step;
}

typedef struct {
    nn_transfer_fn *trans;
    nn_deriv_fn *deriv;
} nn_deriv_entry_t;

static nn_deriv_entry_t g_entries[NN_DERIV_ENTRY_CAP + 5] = {
    { nn_transfer_sigmoid, nn_deriv_sigmoid },
    { nn_transfer_relu, nn_deriv_relu },
    { nn_transfer_tanh, nn_deriv_tanh },
    { nn_transfer_linear, nn_deriv_linear },
    { nn_transfer_thres, NULL }
};
static size_t g_entries_size = 5;

nn_error_t nn_deriv_reg(nn_transfer_fn *f, nn_deriv_fn *deriv) {
    assert(f);
    if (!f) {
        return NN_INVALID_VALUE;
    }

    if (g_entries_size >= NN_DERIV_ENTRY_CAP + 5) {
        return NN_OUT_OF_MEMORY;
    }

    for (size_t i = 0; i < g_entries_size; i++) {
        if (g_entries[i].trans == f) {
            return NN_INVALID_VALUE;
        }
    }
    
    g_entries[g_entries_size] = (nn_deriv_entry_t) { f, deriv };
    g_entries_size += 1;
    return NN_NO_ERROR;
}

nn_deriv_fn *nn_deriv(nn_transfer_fn *f) {
    assert(f);
    if (!f) {
        return NULL;
    }

    for (size_t i = 0; i < g_entries_size; i++) {
        nn_deriv_entry_t *entry = &g_entries[i];
        if (entry->trans == f) {
            return entry->deriv;
        }
    }

    return nn_deriv_num;
}
