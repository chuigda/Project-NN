#ifndef PROJECT_NN_DEFS_H
#define PROJECT_NN_DEFS_H

#define PROJECT_NN_DEF_FP_USE_FLOAT       0
#define PROJECT_NN_DEF_FP_USE_DOUBLE      1
#define PROJECT_NN_DEF_FP_USE_LONG_DOUBLE 2

#ifdef PROJECT_NN_CFG_FP
#   if PROJECT_NN_CFG_FP == PROJECT_NN_DEF_FP_USE_FLOAT
typedef float nn_fp_t;
#   elif PROJECT_NN_CFG_FP == PROJECT_NN_DEF_FP_USE_DOUBLE
typedef double nn_fp_t;
#   elif PROJECT_NN_CFG_FP == PROJECT_NN_DEF_FP_USE_LONG_DOUBLE
typedef long double nn_fp_t;
#   endif
#else
typedef float nn_fp_t;
#endif

#endif /* PROJECT_NN_DEFS_H */
