/**
 * @file fnn.h
 * @brief This file defines full-connected feed-forward neural network
 *        (FNN) related facilities
 */

#ifndef PROJECT_NN_FNN_H
#define PROJECT_NN_FNN_H

#include "neuron.h"
#include "error.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief A full-connected feed-forward neural network
 */
typedef struct {} nn_fnn_t;

/**
 * @brief Create a FNN
 * @return the created FNN on success, `NULL` on out of memory
 */
nn_fnn_t *nn_fnn_create();

/**
 * @brief Destroy a FNN, reclaiming its memory resources
 * @param fnn the FNN to be destroyed. if it is `NULL`, nothing
 *            would happen
 */
void nn_fnn_destroy(nn_fnn_t *fnn);

/**
 * @brief Add a new layer to FNN
 * @param fnn the manipulated FNN, must not be `NULL`
 * @param in_dim input dimension of each neuron, must not be 0.
 *               if there's a "prev" layer before this newly added layer,
 *               then `in_dim` must match the output dimension
 *               (`n_cnt`, neuron count) of that "prev" layer
 * @param n_cnt number of neurons in this layer, must not be 0. also,
 *              this would become the output dimension of this layer
 * @param thres threshold/step function, must not be `NULL`
 * @return
 *   - `NN_NO_ERROR` on success
 *   - `NN_OUT_OF_MEMORY` on out of memory
 *   - `NN_INVALID_VALUE` on dimension mismatch
 */
nn_error_t nn_fnn_add_layer(nn_fnn_t *fnn,
                            size_t in_dim,
                            size_t n_cnt,
                            nn_threshold_fn *thres);

/**
 * @brief Get the input dimension of the FNN
 * @param fnn the FNN, must not be `NULL`
 * @return input dimension of the FNN (in effect, the `in_dim` of first
           layer added). if `fnn` has no layer, 0 will be returned
 */
size_t nn_fnn_in_dim(nn_fnn_t *fnn);

/**
 * @brief Get the current output dimension of the FNN
 * @param fnn the FNN, must not be `NULL`
 * @return output dimension of the FNN (in effect, the `n_cnt` of the
           last layer added). if `fnn` has no layer, 0 will be returned
 */
size_t nn_fnn_out_dim(nn_fnn_t *fnn);

nn_error_t nn_fnn_train(nn_fnn_t *fnn, float *x, float *e, float r);

nn_error_t nn_fnn_test(nn_fnn_t *fnn, float *x, float *y);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* PROJECT_NN_FNN_H */
