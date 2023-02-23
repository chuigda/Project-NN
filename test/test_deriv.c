#include "deriv.h"

#include <assert.h>
#include <stddef.h>

static float test_transfer(float value) {
    return value * 1.5;
}

static float test_deriv(nn_transfer_fn *trans, float x, float y) {
    (void)trans;
    (void)x;
    (void)y;

    return 1.5;
}

int main() {
    assert(nn_deriv(nn_transfer_logistic) == nn_deriv_logistic);
    assert(nn_deriv(nn_transfer_relu) == nn_deriv_relu);
    assert(nn_deriv(nn_transfer_tanh) == nn_deriv_tanh);
    assert(nn_deriv(nn_transfer_linear) == nn_deriv_linear);
    assert(nn_deriv(nn_transfer_thres) == NULL);

    assert(nn_deriv(test_transfer) == nn_deriv_num);
    assert(nn_deriv_reg(test_transfer, test_deriv) == NN_NO_ERROR);
    assert(nn_deriv(test_transfer) == test_deriv);

    assert(nn_deriv_reg(test_transfer, test_deriv)
           == NN_INVALID_VALUE);
    assert(nn_deriv_reg(nn_transfer_logistic, test_deriv) 
           == NN_INVALID_VALUE);
}
