#include "fnn.h"
#include <assert.h>

int main() {
    nn_fnn_t *fnn = nn_fnn_create();

    assert(nn_fnn_add_layer(fnn, 10, 10, nn_transfer_logistic)
           == NN_NO_ERROR);
    assert(nn_fnn_add_layer(fnn, 10, 5, nn_transfer_logistic)
           == NN_NO_ERROR);
    assert(nn_fnn_add_layer(fnn, 5, 1, nn_transfer_thres)
           == NN_NO_ERROR);

    assert(nn_fnn_in_dim(fnn) == 10);
    assert(nn_fnn_out_dim(fnn) == 1);

    assert(nn_fnn_fin(fnn) == NN_NO_ERROR);

    nn_fnn_destroy(fnn);
}
