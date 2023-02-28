/**
 * @file deriv.h
 * @brief This file defines derivative function database
 */

#ifndef PROJECT_NN_DERIV_H
#define PROJECT_NN_DERIV_H

#include "transfer.h"
#include "error.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Derivative function of a transfer function
 * @param trans the source transfer function
 * @param x input `x`
 * @param y output of `f(x)`
 * @return output of ```f'(x)```
 */
typedef float (nn_deriv_fn)(nn_transfer_fn *trans, float x, float y);

/**
 * @brief General numeric derivative function
 */
float nn_deriv_num(nn_transfer_fn *trans, float x, float y);

/**
 * @brief Derivative of linear (`nn_transfer_linear`)
 */
float nn_deriv_linear(nn_transfer_fn *trans, float x, float y);

/**
 * @brief Derivative of sigmoid (`nn_transfer_sigmoid`)
 */
float nn_deriv_sigmoid(nn_transfer_fn *trans, float x, float y);

/**
 * @brief Derivative of tanh (`nn_transfer_tanh`)
 */
float nn_deriv_tanh(nn_transfer_fn *trans, float x, float y);

/**
 * @brief Derivative of ReLU (`nn_transfer_relu`)
 */
float nn_deriv_relu(nn_transfer_fn *trans, float x, float y);

/**
 * @brief Register a derivative function to a transfer function
 * @param f the source function, must not be `NULL`
 * @param deriv the derivative function. can be `NULL`, in that case,
 *              the corresponding function `f` will be taken as
 *              not differentiable
 * @return
 *   - `NN_NO_ERROR` if deriv function got successfully registered
 *   - `NN_INVALID_VALUE` if source function already has a registered
 *     derivative function, or you're trying to register derivative
 *     function for a function which is explicitly known as not 
 *     differentiable
 *   - `NN_OUT_OF_MEMORY` if registered function has reached limit
 *     `NN_DERIV_ENTRY_CAP`. the default value of this constant is 128.
 * @note
 *   - this function is not thread safe because it interacts with
 *     some kind of global state
 *   - threshold/step function (`nn_transfer_thres`) is not
 *     differentiable, intuitively
 *   - though ReLU function (`nn_transfer_relu`) is not differentiable
 *     when `x == 0`, it is still very useful in neural networks. thus
 *     ```f'(0) == 0``` was defined for ReLU's "derivative" function
 */
nn_error_t nn_deriv_reg(nn_transfer_fn *f, nn_deriv_fn *deriv);

/**
 * @brief Get the derivative function of input transfer function
 * @param f the source function, must not be `NULL`
 * @return derivative function of the source function, `NULL` if
 *         `f` is explicitly known as not differentiable
 */
nn_deriv_fn *nn_deriv(nn_transfer_fn *f);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* PROJECT_NN_DERIV_H */
