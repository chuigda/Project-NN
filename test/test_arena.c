#include "arena.h"

#include <stdlib.h>

typedef struct {
    void *ptr;
} item_t;

item_t *item_create() {
    item_t *ret = malloc(sizeof(item_t));
    if (!ret) {
        return NULL;
    }

    ret->ptr = malloc(1024);
    return ret;
}

void item_destroy(item_t *item) {
    if (!item) {
        return;
    }

    free(item->ptr);
    free(item);
}

int main() {
    nn_arena_t *arena = nn_arena_create();

    nn_arena_put(arena, item_create(), (nn_destroy_fn*)item_destroy);
    nn_arena_put(arena, item_create(), (nn_destroy_fn*)item_destroy);
    nn_arena_put(arena, item_create(), (nn_destroy_fn*)item_destroy);

    nn_arena_destroy(arena);
}
