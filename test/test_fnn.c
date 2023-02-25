#include "fnn.h"

#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

static float randf() {
    return (float)rand() / (float)RAND_MAX;
}

static void train_fnn_xor_epoch(nn_fnn_t *fnn,
                                size_t n_epoch, 
                                size_t n_sample,
                                float lr)
{
    for (size_t i = 0; i < n_epoch; i++) {
        float sum_error = 0.0f;
        for (size_t j = 0; j < n_sample; j++) {
            float x1 = randf();
            float x2 = randf();
            
            _Bool b = (x1 > x2) == (x1 > 1 - x2);
            float y = b ? 1.0 : 0.0;

            float x[2] = { x1, x2 };

            float error;
            nn_fnn_train(fnn, x, &y, lr, &error);
            sum_error += error;
        }

        fprintf(stderr, "EPOCH %lu/%lu N_SAMPLE=%lu LR=%g ERROR %g\n",
                i + 1, n_epoch, n_sample, lr, sum_error / n_sample);
    }
}

void test_fnn_xor() {
    nn_fnn_t *fnn = nn_fnn_create();

    nn_fnn_add_layer(fnn, 2, 3, nn_transfer_sigmoid);
    nn_fnn_prewarm_rand(fnn, -0.5f, 0.5f);

    nn_fnn_add_layer(fnn, 3, 1, nn_transfer_sigmoid);
    nn_fnn_prewarm_rand(fnn, -0.5f, 0.5f);

    train_fnn_xor_epoch(fnn, 40, 4096, 0.2);
    train_fnn_xor_epoch(fnn, 40, 4096, 0.1);

    for (size_t i = 0; i < 32; i++) {
        float x1 = randf();
        float x2 = randf();

        _Bool e = (x1 > x2) == (x1 > 1 - x2);

        float input[2] = { x1, x2 };
        float output;
        nn_fnn_test(fnn, input, &output);

        _Bool y = (int)roundf(output);

        fprintf(stderr,
                "TEST %lu/%d IN=(%g,%g) EXP=%d OUT=%d (%g) %s\n",
                i + 1, 32, x1, x2, e, y, output,
                e == y ? "" : "BAD");
    }

    nn_fnn_destroy(fnn);
}

int main() {
    srand(time(NULL));

    fputc('\n', stderr);

    test_fnn_xor();
}
