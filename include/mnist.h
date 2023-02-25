#ifndef PROJECT_NN_MNIST_H
#define PROJECT_NN_MNIST_H

#include <stddef.h>
#include <stdint.h>
#include "error.h"

#ifdef __cplusplus
extern "C" {
#endif

nn_error_t nn_mnist_load_images(const char *file_name,
                                uint8_t **p_images,
                                size_t *p_cnt,
                                size_t *p_w,
                                size_t *p_h);
nn_error_t nn_mnist_load_labels(const char *file_name,
                                uint8_t **p_labels,
                                size_t *p_cnt);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* PROJECT_NN_MNIST_H */
