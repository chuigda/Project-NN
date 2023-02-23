#include "deriv.h"

#include <stdlib.h>
#include <math.h>

#ifndef NN_DERIV_ENTRY_CAP
#define NN_DERIV_ENTRY_CAP 128
#endif

#ifndef NN_NUM_DERIV_STEP
#define NN_NUM_DERIV_STEP 1.0e-6f
#endif

static float nn_deriv_linear(nn_transfer_fn *trans,
                             float x,
                             float y) 
{
    (void)trans;
    (void)x;
    (void)y;

    return 1;
}

static float nn_deriv_logistic(nn_transfer_fn *trans,
                               float x,
                               float y)
{
    (void)trans;
    (void)x;

    return y * (1.0f - y);
}

static float nn_deriv_tanh(nn_transfer_fn *trans,
                           float x,
                           float y)
{
    (void)trans;
    (void)y;

    float denom = expf(-x) + expf(x);
    denom = denom * denom;

    return 4.0 / denom;
}

static float nn_deriv_relu(nn_transfer_fn *trans,
                           float x,
                           float y)
{
    (void)trans;
    (void)y;
    if (x <= 0) {
        return 0;
    } else {
        return 1;
    }
}

typedef struct {
    nn_transfer_fn *trans;
    nn_deriv_fn *deriv;
} nn_deriv_entry_t;

static nn_deriv_entry_t g_entries[NN_DERIV_ENTRY_CAP + 5] = {
    { nn_transfer_logistic, nn_deriv_logistic },
    { nn_transfer_relu, nn_deriv_relu },
    { nn_transfer_tanh, nn_deriv_tanh },
    { nn_transfer_linear, nn_deriv_linear },
    { nn_transfer_thres, NULL }
};
static size_t g_entries_size = 1;

nn_error_t nn_deriv_register(nn_transfer_fn *f, nn_deriv_fn *deriv) {
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
