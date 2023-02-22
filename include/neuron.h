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

/**
 * @brief Threshold/step function for perceptron
 */
typedef float (nn_threshold_fn)(float value);

/**
 * @brief A very basic perceptron (artificial neuron)
 */
typedef struct {} nn_neuron_t;

/**
 * Creates a perceptron
 * @param dim input vector dimensions, must be greater than 0
 * @param thres threshold function, must not be `NULL`
 * @return the created perceptron on success,
 *         `NULL` on out of memory
 */
nn_neuron_t *nn_neuron_create(size_t dim, nn_threshold_fn *thres);

/**
 * Destroys a perceptron, reclaiming its memory resources
 * @param n the perceptron to be destroyed. if it is `NULL`, nothing
 *          would happen
 */
void nn_neuron_destroy(nn_neuron_t* n);

/**
 * Train the perceptron with an input and expected output
 * @param n the perceptron, must not be `NULL`
 * @param x input vector, must not be `NULL`
 * @param e expected output
 * @param r learning rate
 */
void nn_neuron_train(nn_neuron_t *n, float *x, float e, float r);

/**
 * Feed the perceptron with an input vector, get its output
 * @param n the perceptron, must not be `NULL`
 * @param x input vector, must not be `NULL`
 * @return perceptron's output
 */
float nn_neuron_test(nn_neuron_t *n, float *x);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* PROJECT_NN_NEURON_H */
