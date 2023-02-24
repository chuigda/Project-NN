#include "fnn.h"

#include <assert.h>
#include <math.h>
#include <stdio.h>

void test_fnn_xor() {
    static float x[4][2] = {
        { 0.0f, 0.0f },
        { 0.0f, 1.0f },
        { 1.0f, 0.0f },
        { 1.0f, 1.0f }
    };
    static float y[4] = { 0.0f, 1.0f, 1.0f, 0.0f };

    nn_fnn_t *fnn = nn_fnn_create();

    nn_fnn_add_layer(fnn, 2, 2, nn_transfer_logistic);
    nn_fnn_prewarm_rand(fnn, -1.0f, 1.0f);

    nn_fnn_add_layer(fnn, 2, 3, nn_transfer_logistic);
    nn_fnn_prewarm_rand(fnn, -1.0f, 1.0f);

    nn_fnn_add_layer(fnn, 3, 1, nn_transfer_logistic);
    nn_fnn_prewarm_rand(fnn, -1.0f, 1.0f);

    for (size_t i = 0; i < 4096 * 4096; i++) {
        for (size_t j = 0; j < 4; j++) {
            nn_fnn_train(fnn, x[j], &y[j], 0.05, NULL);
        }
    }

    float output;

    nn_fnn_test(fnn, x[0], &output);
    printf("XOR(0, 0) = %g (%g)\n", roundf(output), output);
    nn_fnn_test(fnn, x[1], &output);
    printf("XOR(0, 1) = %g (%g)\n", roundf(output), output);
    nn_fnn_test(fnn, x[2], &output);
    printf("XOR(1, 0) = %g (%g)\n", roundf(output), output);
    nn_fnn_test(fnn, x[3], &output);
    printf("XOR(1, 1) = %g (%g)\n", roundf(output), output);

    nn_fnn_destroy(fnn);
}

int main() {
    test_fnn_xor();
}
