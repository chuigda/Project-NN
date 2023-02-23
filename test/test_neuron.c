#include "neuron.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

static int random_int(int l, int r) {
  return (rand() % (r - l)) + l;
}

int main() {
  srand(time(NULL));

  nn_neuron_t *n = nn_neuron_create(3, nn_transfer_thres);
  
  for (size_t i = 0; i < 4096 * 4096; i++) {
    float r = (float)random_int(0, 256),
          g = (float)random_int(0, 256),
          b = (float)random_int(0, 256);
    float e = (float)(int)(r > g + b);

    float x[3] = { r, g, b };
    nn_neuron_train(n, x, e, 0.0001);
  }

  int correct = 0, incorrect = 0;
  for (size_t i = 0; i < 4096; i++) {
    float r = (float)random_int(0, 4096),
          g = (float)random_int(0, 4096),
          b = (float)random_int(0, 4096);
    int e = r > g + b;
    
    float x[3] = { r, g, b };
    int y = round(nn_neuron_test(n, x));
    
    if (e == y) {
      correct += 1;
    } else {
      printf("INPUT = (%g, %g, %g), EXPECTED = %d, OUTPUT = %d\n", r, g, b, e, y);
      incorrect += 1;
    }
  }

  printf("POOL = 4096, CORRECT = %d, INCORRECT = %d\n", correct, incorrect);

  nn_neuron_destroy(n);
}
