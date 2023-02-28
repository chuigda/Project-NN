#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "mnist.h"
#include "fnn.h"

static void fnn_train(nn_fnn_t *fnn,
                      size_t image_size,
                      size_t cnt,
                      float *images,
                      uint8_t *labels,
                      float r);

static size_t max_float_idx(float *values, size_t cnt);

int main() {
    srand(time(NULL));

    nn_error_t err = NN_NO_ERROR;
    
    uint8_t *train_images;
    size_t train_image_cnt, train_image_w, train_image_h;

    err = nn_mnist_load_images("./dataset/train-images-idx3-ubyte",
                               &train_images,
                               &train_image_cnt,
                               &train_image_w,
                               &train_image_h);
    assert(err == NN_NO_ERROR);
    size_t image_size = train_image_w * train_image_h;
    size_t elem_cnt = train_image_cnt * image_size;
    float *train_images_f = malloc(sizeof(float) * elem_cnt);
    assert(train_images_f);
    for (size_t i = 0; i < elem_cnt; i++) {
        train_images_f[i] = (float)train_images[i] / 255.0f;
    }
    free(train_images);

    uint8_t *train_labels;
    size_t train_label_cnt;
    err = nn_mnist_load_labels("./dataset/train-labels-idx1-ubyte",
                               &train_labels,
                               &train_label_cnt);
    assert(err == NN_NO_ERROR);
    assert(train_label_cnt == train_image_cnt);

    uint8_t *test_images;
    size_t test_image_cnt, test_image_w, test_image_h;

    err = nn_mnist_load_images("./dataset/t10k-images-idx3-ubyte",
                               &test_images,
                               &test_image_cnt,
                               &test_image_w,
                               &test_image_h);
    assert(err == NN_NO_ERROR);
    assert(test_image_w == train_image_w
           && test_image_h == train_image_h);
    size_t test_elem_cnt = test_image_cnt * image_size;
    float *test_images_f = malloc(sizeof(float) * test_elem_cnt);
    assert(test_images_f);
    for (size_t i = 0; i < test_elem_cnt; i++) {
        test_images_f[i] = (float)test_images[i] / 255.0f;
    }
    free(test_images);

    uint8_t *test_labels;
    size_t test_label_cnt;
    err = nn_mnist_load_labels("./dataset/t10k-labels-idx1-ubyte",
                               &test_labels,
                               &test_label_cnt);
    assert(err == NN_NO_ERROR);
    assert(test_label_cnt == test_image_cnt);

    nn_fnn_t *fnn = nn_fnn_create();

    err = nn_fnn_add_layer(fnn, image_size, 300, nn_transfer_sigmoid);
    assert(err == NN_NO_ERROR);
    err = nn_fnn_prewarm_rand(fnn, -0.5f, 0.5f);
    assert(err == NN_NO_ERROR);

    err = nn_fnn_add_layer(fnn, 300, 100, nn_transfer_sigmoid);
    assert(err == NN_NO_ERROR);
    err = nn_fnn_prewarm_rand(fnn, -0.5f, 0.5f);
    assert(err == NN_NO_ERROR);

    err = nn_fnn_add_layer(fnn, 100, 10, nn_transfer_sigmoid);
    assert(err == NN_NO_ERROR);
    err = nn_fnn_prewarm_rand(fnn, -0.5f, 0.5f);
    assert(err == NN_NO_ERROR);

    for (size_t i = 0; i < 4; i++) {
        printf("RUNNING EPOCH %lu/%d ...\n", i + 1, 4);
        fnn_train(fnn,
                  image_size,
                  train_image_cnt,
                  train_images_f,
                  train_labels,
                  0.2f);
    }

    free(train_images_f);
    free(train_labels);

    printf("TESTING TRAINED NETWORK ...\n");
    size_t error_count = 0;
    for (size_t i = 0; i < test_image_cnt; i++) {
        float *image = test_images_f + image_size * i;
        float y[10];
        uint8_t label = test_labels[i];

        assert(nn_fnn_test(fnn, image, y) == NN_NO_ERROR);
        size_t output = max_float_idx(y, 10);
        
        if (output != label) {
            error_count += 1;
        }
    }
    printf("\tTESTED WITH %lu IMAGES, %lu INCORRECTS (%g%%)\n",
           test_image_cnt,
           error_count,
           (float)error_count / (float)test_image_cnt * 100.0f);

    free(test_images_f);
    free(test_labels);
    nn_fnn_destroy(fnn);
}

static void fnn_train(nn_fnn_t *fnn,
                      size_t image_size,
                      size_t cnt,
                      float *images,
                      uint8_t *labels,
                      float r)
{
    float e[10];
    for (size_t i = 0; i < 10; i++) {
        e[i] = 0.0f;
    }

    float err_sum = 0.0f;
    for (size_t i = 0; i < cnt; i++) {
        float *image = images + image_size * i;
        uint8_t label = labels[i];

        e[label] = 1.0f;
        float err;
        assert(nn_fnn_train(fnn, image, e, r, &err) == NN_NO_ERROR);
        err_sum += err;
        e[label] = 0.0f;

        if (i != 0 && (i + 1) % 5000 == 0) {
            printf("\tPROCESSING IMAGE %lu/%lu (%g%%), REALTIME ERR = %g\n",
                   i + 1,
                   cnt,
                   (float)(i + 1) / (float)cnt * 100.0f,
                   err_sum / i);
        }
    }

    printf("\tINTAKE %lu IMAGES, ERR = %g\n", cnt, err_sum / cnt);
}

static size_t max_float_idx(float *values, size_t cnt) {
    assert(cnt != 0);
    float max = values[0];
    size_t idx = 0;
    for (size_t i = 0; i < cnt; i++) {
        float value = values[i];
        if (value > max) {
            max = value;
            idx = i;
        }
    }
    return idx;
}
