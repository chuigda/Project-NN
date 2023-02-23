#include "transfer.h"

#include <math.h>

float nn_transfer_thres(float value) {
    return value >= 0.0f ? 1.0f : 0.0f;
}

float nn_transfer_logistic(float value) {
    return (float)(1.0 / (1.0 + exp(-value)));
}
