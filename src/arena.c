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

typedef struct {
    nn_arena_item_t *items;
    size_t size;
    size_t cap;
} nn_arena_impl_t;

nn_arena_t *nn_arena_create() {
    nn_arena_impl_t *r = malloc(sizeof(nn_arena_impl_t));
    if (!r) {
        return NULL;
    }

    r->items = NULL;
    r->size = 0;
    r->cap = 0;
    return (nn_arena_t*)r;
}

void nn_arena_destroy(nn_arena_t *arena) {
    if (!arena) {
        return;
    }

    nn_arena_impl_t *impl = (nn_arena_impl_t*)arena;
    for (size_t i = 0; i < impl->size; i++) {
        nn_arena_item_t *item = &impl->items[i];
        if (item->dtor) {
            item->dtor(item->ptr);
        } else {
            free(item->ptr);
        }
    }
    free(impl);
}

nn_error_t nn_arena_put(nn_arena_t *arena, void *ptr, nn_destroy_fn *dtor) {
    assert(arena);
    if (!(arena && ptr)) {
        return NN_INVALID_VALUE;
    }

    nn_arena_impl_t *impl = (nn_arena_impl_t*)arena;
    if (impl->items == NULL) {
        impl->items = 
            malloc(NN_ARENA_INIT_CAP * sizeof(nn_arena_item_t));
        if (!impl->items) {
            return NN_OUT_OF_MEMORY;
        }

        impl->cap = NN_ARENA_INIT_CAP;
        impl->size = 1;
        impl->items[0] = (nn_arena_item_t) { ptr, dtor };
        return 1;
    }

    if (impl->size == impl->cap) {
        nn_arena_item_t *new_items = 
            malloc(impl->cap * 2 * sizeof(nn_arena_impl_t));
        if (!new_items) {
            return NN_OUT_OF_MEMORY;
        }

        memcpy(new_items,
               impl->items,
               impl->cap * sizeof(nn_arena_impl_t));
        free(impl->items);
        impl->items = new_items;
        impl->cap *= 2;
    }

    impl->items[impl->size] = (nn_arena_item_t) { ptr, dtor };
    impl->size += 1;
    return NN_NO_ERROR;
}
