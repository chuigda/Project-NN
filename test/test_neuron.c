#include "neuron.h"

#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

static int random_int(int l, int r) {
  return (rand() % (r - l)) + l;
}

void test_neuron1() {
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
    int y = round(nn_neuron_test(n, x, NULL));
    
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

void test_neuron2() {
  nn_neuron_t *n = nn_neuron_create(3, nn_transfer_linear);
  
  for (size_t i = 0; i < 4096 * 4096; i++) {
    float r = (float)random_int(0, 256) / 256.0f,
          g = (float)random_int(0, 256) / 256.0f,
          b = (float)random_int(0, 256) / 256.0f;
    float e = r + g + b;

    float x[3] = { r, g, b };
    nn_neuron_train(n, x, e, 0.0001);
  }

  float err_sum = 0.0;
  float err_percent_sum = 0.0;
  for (size_t i = 0; i < 4096; i++) {
    float r = (float)random_int(0, 256) / 256.0f,
          g = (float)random_int(0, 256) / 256.0f,
          b = (float)random_int(0, 256) / 256.0f;
    float e = r + g + b;

    float x[3] = { r, g, b };
    float y = nn_neuron_test(n, x, NULL);
    
    float err = fabs(e - y);
    float err_percent = fabs((e - y) / e);
    printf("EXPECTED = %g, GOT = %g, ERROR = %g (%g %%)\n",
           e,
           y,
           err,
           err_percent * 100);
    err_sum += err;
    err_percent_sum += err_percent;
  }

  printf("AVERAGE(ERROR) = %g (%g %%)\n",
         err_sum / 4096.0f,
         err_percent_sum / 4096.0f * 100.0f);

  nn_neuron_destroy(n);
}

int main(int argc, const char *argv[]) {
  srand(time(NULL));

  assert(argc == 2);
  int test = atoi(argv[1]);

  switch (test) {
    case 1: test_neuron1(); break;
    case 2: test_neuron2(); break;
    default: assert(0 && "invalid test item");
  }

  srand(time(NULL));
}
