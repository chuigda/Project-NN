/**
 * @file mnist.h
 * @brief This file defines auxiliary function for loading MNIST data
 */

#ifndef PROJECT_NN_MNIST_H
#define PROJECT_NN_MNIST_H

#include <stddef.h>
#include <stdint.h>
#include "error.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Load MNIST images file
 * @param file_name input file name, must not be `NULL`
 * @param p_images images output, must not be `NULL`
 * @param p_cnt image count output, must not be `NULL`
 * @param p_w width output, must not be `NULL`
 * @param p_h height output, must not be `NULL`
 * @return
 *   - `NN_NO_ERROR` on success
 *   - `NN_IO_ERROR` on error reading file
 *   - `NN_BAD_FORMAT` if input file is not valid MNIST images file
 *   - `NN_OUT_OF_MEMORY` on out of memory
 */
nn_error_t nn_mnist_load_images(const char *file_name,
                                uint8_t **p_images,
                                size_t *p_cnt,
                                size_t *p_w,
                                size_t *p_h);

/**
 * @brief Load MNIST labels file
 * @param file_name input file name, must not be `NULL`
 * @param p_labels labels output, must not be `NULL`
 * @param p_cnt label count output, must not be `NULL`
 * @return
 *   - `NN_NO_ERROR` on success
 *   - `NN_IO_ERROR` on error reading file
 *   - `NN_BAD_FORMAT` if input file is not valid MNIST labels file
 *   - `NN_OUT_OF_MEMORY` on out of memory
 */
nn_error_t nn_mnist_load_labels(const char *file_name,
                                uint8_t **p_labels,
                                size_t *p_cnt);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* PROJECT_NN_MNIST_H */
