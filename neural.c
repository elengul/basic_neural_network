#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include "neural.h"
#include "rand_gen.h"

struct weights *
weights_create(size_t size, double mu, double var)
{
    struct weights *out = malloc(sizeof(*out));
    out->size          = size;
    out->weight        = malloc(out->size * sizeof(out->weight[0]));
    out->w_delta       = malloc(out->size * sizeof(out->w_delta[0]));
    out->w_delta_store = malloc(out->size * sizeof(out->w_delta_store[0]));
    for (size_t i = 0; i < out->size; i++) {
        out->weight[i]        = rand_normal(mu, var);
        out->w_delta[i]       = 0;
        out->w_delta_store[i] = 0;
    }
    return out;
}

void
weights_destroy(struct weights **W)
{
    struct weights *w = *W;
    free(w->weight);
    free(w->w_delta);
    free(w->w_delta_store);
    free(*W);
    *W = NULL;
}

struct bias *
bias_create(double mu, double var)
{
    struct bias *out = malloc(sizeof(*out));
    out->bias          = rand_normal(mu, var);
    out->b_delta       = 0;
    out->b_delta_store = 0;
    return out;
}

void
bias_destroy(struct bias **B)
{
    free(*B);
    *B = NULL;
}

struct neuron *
neuron_create(size_t size, double mu, double var)
{
    struct neuron *out = malloc(sizeof(*out));
    out->size    = size;
    out->weights = weights_create(size, mu, var);
    out->bias    = bias_create(mu, var);
    out->input   = 0;
    out->output  = 0;
    return out;
}

void
neuron_destroy(struct neuron **n)
{
    weights_destroy(&((*n)->weights));
    bias_destroy(&((*n)->bias));
    free(*n);
    *n = NULL;
}

struct layer *
layer_create(size_t this, size_t last, double mu, double var)
{
    struct layer *out = malloc(sizeof(*out));
    out->size   = this;
    out->neuron = malloc(out->size * sizeof(out->neuron[0]));
    for (size_t i = 0; i < out->size; i++)
        out->neuron[i] = neuron_create(last, mu, var);
    return out;
}

void
layer_destroy(struct layer **L)
{
    struct layer *l = *L;
    for (size_t i = 0; i < l->size; i++)
        neuron_destroy(&(l->neuron[i]));
    free(l->neuron);
    free(*L);
    *L = NULL;
}

struct network *
network_create(void)
{
    struct network *out = malloc(sizeof(*out));
    out->size  = 4;
    out->count = 0;
    out->layer = malloc(out->size * sizeof(out->layer[0]));
    return out;
}

void
network_destroy(struct network **N)
{
    struct network *n = *N;
    for (size_t i = 0; i < n->count; i++)
        layer_destroy(&(n->layer[i]));
    free(n->layer);
    free(*N);
    *N = NULL;
}

void
network_push(struct network *n, struct layer *l)
{
    if (n->count == n->size) {
        n->size *= 2;
        n->layer = realloc(n->layer, n->size * sizeof(n->layer[0]));
    }
    n->layer[n->count] = l;
    n->count++;
}

void
network_reset(struct network *n)
{
    for (size_t i = 0; i < n->count; i++)
        for (size_t j = 0; j < n->layer[i]->size; j++) {
            n->layer[i]->neuron[j]->bias->b_delta_store = 0.0;
            for (size_t k = 0; k < n->layer[i]->neuron[j]->weights->size; k++)
                n->layer[i]->neuron[j]->weights->w_delta_store[i] = 0.0;
        }
}

void
network_forward(struct network *n, double (*func)(double))
{
    for (size_t i = 1; i < n->count; i++) {
        struct layer *this = n->layer[i];
        struct layer *last = n->layer[i-1];
        for (size_t j = 0; j < this->size; j++) {
            struct neuron *neu = this->neuron[j];
            neu->input = neu->bias->bias;
            for (size_t k = 0; k < last->size; k++)
                neu->input += (neu->weights->weight[k] * last->neuron[k]->output);
            neu->output = func(neu->input);
        }
    }
}

double
bias_calc(struct network *n, int lay_idx, int neu_idx, int ans)
{
    if (lay_idx == (n->count - 1))
        return (n->layer[lay_idx]->neuron[neu_idx]->output - ans);
    double out = 0.0;
    for (size_t i = 0; i < n->layer[lay_idx + 1]->size; i++)
        out += n->layer[lay_idx+1]->neuron[i]->bias->b_delta;
    return out;
}

double
weight_calc(struct network *n, int lay_idx, int neu_idx, int k, int ans)
{
    double last_out = n->layer[lay_idx - 1]->neuron[k]->output;
    if (lay_idx == (n->count - 1)) {
        double this_diff = (n->layer[lay_idx]->neuron[neu_idx]->output - ans);
        return this_diff * last_out;
    }
    double out = 0.0;
    for (int i = 0; i < n->layer[lay_idx + 1]->size; i++)
        out += (n->layer[lay_idx+1]->neuron[i]->bias->b_delta *
                n->layer[lay_idx+1]->neuron[i]->weights->weight[k]);
    return out * last_out;
}

void
network_calculate_errors(struct network *n, int ans, double (*func_der)(double))
{
    for (int l = n->count - 1; l >= 0; l--) {
        for (int j = 0; j < n->layer[l]->size; j++) {
            struct neuron *neu = n->layer[l]->neuron[j];
            double da_dz = func_der(neu->input);
            neu->bias->b_delta = da_dz * bias_calc(n, l, j, j == ans);
            neu->bias->b_delta_store += neu->bias->b_delta;
            for (int k = 0; k < neu->size; k++) {
                neu->weights->w_delta[k] = da_dz * weight_calc(n, l, j, k, ans);
                neu->weights->w_delta_store[k] += neu->weights->w_delta[k];
            }
        }
    }
}

void
network_apply_errors(struct network *n, int N, double alpha)
{
    for (size_t i = 0; i < n->count; i++)
        for (size_t j = 0; j < n->layer[i]->size; j++) {
            n->layer[i]->neuron[j]->bias->bias -=
                alpha*(n->layer[i]->neuron[j]->bias->b_delta_store / (double)N);
            for (size_t k = 0; k < n->layer[i]->neuron[j]->size; k++)
                n->layer[i]->neuron[j]->weights->weight[k] -=
                    alpha*(n->layer[i]->neuron[j]->weights->w_delta_store[k] / (double)N);
        }
    network_reset(n);
}
