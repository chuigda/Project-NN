/**
 * @file transfer.h
 * @brief This file defines common transfer functions
 */

#ifndef PROJECT_NN_TRANSFER_H
#define PROJECT_NN_TRANSFER_H

#ifdef __cplusplus
extern "C" {
#endif 

/**
 * @brief Transfer function for perceptron
 */
typedef float (nn_transfer_fn)(float value);

/**
 * @brief Linear function
 * @param value input value
 * @return `value` itself
 */
float nn_transfer_linear(float value);

/**
 * @brief Threshold/step function
 * @param value input value
 * @return `value >= 0.0f ? 1.0f : 0.0f`
 */
float nn_transfer_thres(float value);

/**
 * @brief Sigmoid function
 * @param value input value
 * @return `1.0f / (1.0f + expf(-value))`
 */
float nn_transfer_sigmoid(float value);

/**
 * @brief Tanh sigmoid function
 * @param value input value
 * @return `(expf(x) - expf(-x)) / (expf(x) + expf(-x))`
 */
float nn_transfer_tanh(float value);

/** 
 * @brief ReLU/Rectifier function
 * @param value input value
 * @return `value > 0.0f ? value : 0.0f`
 */
float nn_transfer_relu(float value);

/** 
 * @brief Leaky ReLU function
 * @param value input value
 * @return `value > 0.0f ? value : a * value`
 */
float nn_transfer_leaky_relu(float value);

/**
 * @brief Set `a` component of the leaky relu function
 * @param a newly set a, must be in range `(0.0f, 1.0f)`
 * @note this function is not thread safe because it interacts with
 *       some kind of global state
 */
void nn_transfer_leaky_relu_set_a(float a);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* PROJECT_NN_TRANSFER_H */
