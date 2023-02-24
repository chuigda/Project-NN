#include "impl/util.h"

#include <assert.h>
#include <stdlib.h>

size_t nn_imp_veclen(void **vec) {
    if (!vec) {
        return 0;
    }

    size_t r = 0;
    while (vec[r]) {
        r += 1;
    }
    return r;
}

float nn_imp_randf(float l, float r) {
    assert(l < r);
    if (!(l < r)) {
        return l;
    }

    float random = ((float)rand()) / (float)RAND_MAX;
    float d = r - l;
    return l + random * d;
}
