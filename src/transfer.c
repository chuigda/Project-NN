#include "transfer.h"

#include <assert.h>
#include <math.h>

float nn_transfer_linear(float value) {
    return value;
}

float nn_transfer_thres(float value) {
    return value >= 0.0f ? 1.0f : 0.0f;
}

float nn_transfer_sigmoid(float value) {
    return 1.0f / (1.0f + expf(-value));
}

float nn_transfer_tanh(float value) {
    float e1 = expf(value);
    float e2 = expf(-value);
    return (e1 - e2) / (e1 + e2);
}

float nn_transfer_relu(float value) {
    return value > 0.0f ? value : 0.0f;
}

extern float nn_imp_leaky_relu_a;
float nn_imp_leaky_relu_a = 0.1f;

float nn_transfer_leaky_relu(float value) {
    if (value > 0.0f) {
        return value;
    } else {
        return value * nn_imp_leaky_relu_a;
    }
}

void nn_transfer_leaky_relu_set_a(float a) {
    assert(a > 0.0f && a < 1.0f);
    nn_imp_leaky_relu_a = a;
}
