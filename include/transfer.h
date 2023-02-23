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
 * @brief Threshold/step function
 * @param value input value
 * @return `value >= 0.0f ? 1.0f : 0.0f`
 */
float nn_transfer_thres(float value);

/**
 * @brief Logistic sigmoid function
 * @param value input value
 * @return `(float)(1.0 / (1.0 + exp(-value)))`
 */
 float nn_transfer_logistic(float value);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* PROJECT_NN_TRANSFER_H */