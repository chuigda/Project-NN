#include "transfer.h"

#include <math.h>

float nn_transfer_linear(float value) {
    return value;
}

float nn_transfer_thres(float value) {
    return value >= 0.0f ? 1.0f : 0.0f;
}

float nn_transfer_logistic(float value) {
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
