#ifndef PROJECT_NN_IMPL_UTIL_H
#define PROJECT_NN_IMPL_UTIL_H

#include <stddef.h>

size_t nn_imp_veclen(void **vec);

float nn_imp_randf(float l, float r);

#define NN_IMP_SWAP(T, X, Y) { T t = X; X = Y; Y = t; }

#endif /* PROJECT_NN_IMPL_UTIL_H */
