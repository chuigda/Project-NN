#ifndef PROEJCT_NN_ARENA_H
#define PROEJCT_NN_ARENA_H

#ifdef __cplusplus
extern "C" {
#endif

typedef void (nn_destroy_fn)(void *ptr);

typedef struct {} nn_arena_t;

nn_arena_t *nn_arena_create();

void nn_arena_destroy(nn_arena_t *arena);

_Bool nn_arena_put(nn_arena_t *arena, void *ptr, nn_destroy_fn *dtor);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* PROJECT_NN_ARENA_H */
