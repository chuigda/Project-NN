#include "arena.h"

#include <assert.h>
#include <stddef.h>
#include <stdlib.h>
#include <string.h>

#ifndef NN_ARENA_INIT_CAP
#define NN_ARENA_INIT_CAP 16
#endif

typedef struct {
    void *ptr;
    nn_destroy_fn *dtor;
} nn_arena_item_t;

struct st_nn_imp_arena {
    nn_arena_item_t *items;
    size_t size;
    size_t cap;
};

nn_arena_t *nn_arena_create() {
    nn_arena_t *r = malloc(sizeof(nn_arena_t));
    if (!r) {
        return NULL;
    }

    r->items = NULL;
    r->size = 0;
    r->cap = 0;
    return r;
}

void nn_arena_destroy(nn_arena_t *arena) {
    if (!arena) {
        return;
    }

    for (size_t i = 0; i < arena->size; i++) {
        nn_arena_item_t *item = &arena->items[i];
        if (item->dtor) {
            item->dtor(item->ptr);
        } else {
            free(item->ptr);
        }
    }
    free(arena);
}

nn_error_t nn_arena_put(nn_arena_t *arena, void *ptr, nn_destroy_fn *dtor) {
    assert(arena);
    if (!(arena && ptr)) {
        return NN_INVALID_VALUE;
    }

    if (arena->items == NULL) {
        arena->items =
            malloc(NN_ARENA_INIT_CAP * sizeof(nn_arena_item_t));
        if (!arena->items) {
            return NN_OUT_OF_MEMORY;
        }

        arena->cap = NN_ARENA_INIT_CAP;
        arena->size = 1;
        arena->items[0] = (nn_arena_item_t) { ptr, dtor };
        return 1;
    }

    if (arena->size == arena->cap) {
        nn_arena_item_t *new_items = 
            malloc(arena->cap * 2 * sizeof(nn_arena_item_t));
        if (!new_items) {
            return NN_OUT_OF_MEMORY;
        }

        memcpy(new_items,
               arena->items,
               arena->cap * sizeof(nn_arena_item_t));
        free(arena->items);
        arena->items = new_items;
        arena->cap *= 2;
    }

    arena->items[arena->size] = (nn_arena_item_t) { ptr, dtor };
    arena->size += 1;
    return NN_NO_ERROR;
}
