/**
 * @file error.h
 * @brief This file defines error codes used by this library.
 */

#ifndef PROJECT_NN_ERROR_H
#define PROJECT_NN_ERROR_H

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Error code type
 */
typedef enum {
    /** Misson accomplished successfully */
    NN_NO_ERROR = 0,

    /** Input enum/option not valid */
    NN_INVALID_ENUM = 0x0500,

    /** Input value not valid */
    NN_INVALID_VALUE = 0x0501,

    /** Operation not valid */
    NN_INVALID_OPERATION = 0x0502,

    /** Out of memory */
    NN_OUT_OF_MEMORY = 0x0505,
} nn_error_t;

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* PROJECT_NN_ERROR_H */
