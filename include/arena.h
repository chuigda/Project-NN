/**
 * @file arena.h
 * @brief This file defines a very basic arena for helping memory
 *        management.
 */

#ifndef PROEJCT_NN_ARENA_H
#define PROEJCT_NN_ARENA_H

#include "error.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Generic resource dtor
 */
typedef void (nn_destroy_fn)(void *ptr);

struct st_nn_imp_arena;

/**
 * @brief A very basic arena for helping memory management
 */
typedef struct st_nn_imp_arena nn_arena_t;

/**
 * @brief Create an arena
 * @return the created arena on success, `NULL` on out of memory
 */
nn_arena_t *nn_arena_create();

/**
 * @brief Destroy an arena and all objects it holds
 * @param arena the arena to be destroyed. if it is `NULL`, nothing would
 *              happen
 */
void nn_arena_destroy(nn_arena_t *arena);

/**
 * @brief Add a resource to arena
 * @param arena the arena to use
 * @param ptr the resource to be added
 * @param dtor the custom resoure dtor to be run for destroying resource
 *             held by `ptr`. if it is `NULL`, arena will fallback to use
 *             libc `free`
 * @return `NN_NO_ERROR` on success, `NN_OUT_OF_MEMORY` on out of memory
 */
nn_error_t nn_arena_put(nn_arena_t *arena, void *ptr, nn_destroy_fn *dtor);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* PROJECT_NN_ARENA_H */
