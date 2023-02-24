/**
 * @file neuron.h
 * @brief This file defines perceptron (artificial neuron) and its
 *        related operations.
 */

#ifndef PROJECT_NN_NEURON_H
#define PROJECT_NN_NEURON_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stddef.h>
#include <stdint.h>

#include "transfer.h"

struct st_nn_imp_neuron;

/**
 * @brief A very basic perceptron (artificial neuron)
 */
typedef struct st_nn_imp_neuron nn_neuron_t;

/**
 * @brief Create a perceptron
 * @param dim input vector dimensions, must not be 0
 * @param trans transfer function, must not be `NULL`
 * @return the created perceptron on success,
 *         `NULL` on out of memory
 *
 * @note When playing with single-layer perceptron, only linear and
 *       step functions are really meaningful.
 */
nn_neuron_t *nn_neuron_create(size_t dim, nn_transfer_fn *trans);

/**
 * @brief Destroy a perceptron, reclaiming its memory resources
 * @param n the perceptron to be destroyed. if it is `NULL`, nothing
 *          would happen
 */
void nn_neuron_destroy(nn_neuron_t* n);

/**
 * @brief Get the input dimension of the perceptron
 * @param n the perceptron, must not be `NULL`
 * @return the input dimension of the perceptron
 */
size_t nn_neuron_dim(nn_neuron_t *n);

/**
 * @brief Prewarms the perceptron by setting weights and bias to fixed
 *        value
 * @param n the perceptron, must not be `NULL`
 * @param v value
 */
void nn_neuron_prewarm(nn_neuron_t *n, float v);

/**
 * @brief Prewarms the perceptron by setting weights and bias to random
 *        values
 * @param n the perceptron, must not be `NULL`
 * @param l lower bound
 * @param r upper bound, must be greater than `l`
 */
void nn_neuron_prewarm_rand(nn_neuron_t *n, float l, float r);

/**
 * @brief Train the perceptron with given input/output at a learning rate
 * @param n the perceptron, must not be `NULL`
 * @param x input vector, must not be `NULL`
 * @param e expected output
 * @param r learning rate, must be greater than `0.0f`
 */
void nn_neuron_train(nn_neuron_t *n, float *x, float e, float r);

/**
 * @brief Feed the perceptron with an input vector, get its output
 * @param n the perceptron, must not be `NULL`
 * @param x input vector, must not be `NULL`
 * @return perceptron's output
 */
float nn_neuron_test(nn_neuron_t *n, float *x);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* PROJECT_NN_NEURON_H */
