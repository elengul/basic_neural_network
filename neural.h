#pragma once

#include <stdlib.h>

struct weights {
    size_t size;
    double *weight;
    double *w_delta;
    double *w_delta_store;
};

struct bias {
    double bias;
    double b_delta;
    double b_delta_store;
};

struct neuron {
    size_t size;
    struct weights *weights;
    struct bias *bias;
    double input;
    double output;
};

struct layer {
    size_t size;
    struct neuron **neuron;
};

struct network {
    size_t size, count;
    struct layer **layer;
};

struct layer *layer_create(size_t this, size_t last, double mu, double var);
struct network *network_create(void);
void network_destroy(struct network **N);
void network_push(struct network *n, struct layer *l);
void network_forward(struct network *n, double (*func)(double));
void network_calculate_errors(struct network *n, int ans, double (*func_der)(double));
void network_apply_errors(struct network *n, int N, double alpha);
