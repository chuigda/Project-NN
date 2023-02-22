#include "impl/util.h"

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
