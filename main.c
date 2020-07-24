#include <math.h>
#include <stdio.h>
#include "neural.h"
#include "rand_gen.h"

#define TR_SZ 1000

double
sigmoid(double in)
{
    return 1.0 / (1.0 + exp(-1.0 * in));
}

double
sig_deriv(double in)
{
    double sig = sigmoid(in);
    return sig*(1 - sig);
}

void
print_weights_and_biases(struct network *n, int gen)
{
    for (size_t i = 1; i < n->count; i++)
        for (size_t j = 0; j < n->layer[i]->size; j++) {
            struct neuron *neu = n->layer[i]->neuron[j];
            printf("Generation %d, Layer %zu, Neuron %zu:\n", gen, i, j);
            printf("  Bias: %11.8f\n", neu->bias->bias);
            for (size_t k = 0; k < neu->weights->size; k++)
                printf("  Weight %zu: %11.8f\n", k, neu->weights->weight[k]);
        }
}

void
training_data_fill(int IMAX, int JMAX, double train[IMAX][JMAX])
{
    for (int i = 0; i < IMAX; i++) {
        for (int j = 0; j < 2; j++)
            train[i][j] = rand_normal(0.99 - 0.98*(i == j), 0.0001);
        train[i][2] = 1.0 - (i % 2);
    }
}

void
training_data_permute(int IMAX, int JMAX, double train[IMAX][JMAX])
{
    for (size_t i = (IMAX-1); i > 0; i--) {
        int j = rand_int_range(0, i);
        double tmp  = train[i][0];
        train[i][0] = train[j][0];
        train[j][0] = tmp;
        tmp         = train[i][1];
        train[i][1] = train[j][1];
        train[j][1] = tmp;
        tmp         = train[i][2];
        train[i][2] = train[j][2];
        train[j][2] = tmp;
    }
}

double
cost(struct network *n, double ans)
{
    return 0.5*pow(ans - n->layer[n->count - 1]->neuron[0]->output, 2);
}

int main(void)
{
    seed();
    struct network *n = network_create();
    struct layer *lay = layer_create(2, 0, 0, 5);
    network_push(n, lay);
    lay = layer_create(1, 2, 0, 5);
    network_push(n, lay);

    double train[TR_SZ][3];
    training_data_fill(TR_SZ, 3, train);
    
    double output[100];
    print_weights_and_biases(n, 0);
    double last_cost = 1.1;
    for (int i = 0; i < 100; i++) {
        output[i] = 0;
        for (int k = 0; k < 10; k++) {
            for (int j = 0; j < (TR_SZ/10); j++) {
                int idx = j + (TR_SZ/10)*k;
                double base_x = train[idx][0];
                double base_y = train[idx][1];
                int ans  = (int)train[idx][2];
                n->layer[0]->neuron[0]->output = base_x;
                n->layer[0]->neuron[1]->output = base_y;
                network_forward(n, sigmoid);
                output[i] += cost(n, 1 - ans);
                network_calculate_errors(n, ans, sig_deriv);
            }
            network_apply_errors(n, TR_SZ/10, 0.01);
        }
        output[i] /= (double)TR_SZ;
        training_data_permute(TR_SZ, 3, train);
        printf("Generation %d, average cost = %10.8f\n", i+1, output[i]);
        if (output[i] > last_cost)
            break;
        last_cost = output[i];
    }

    double test[TR_SZ/10][3];
    training_data_fill(TR_SZ/10, 3, test);
    double cost_sum = 0.0;
    for (int i = 0; i < (TR_SZ/10); i++) {
        n->layer[0]->neuron[0]->output = test[i][0];
        n->layer[0]->neuron[1]->output = test[i][1];
        network_forward(n, sigmoid);
        cost_sum += cost(n, test[i][2]);
    }
    cost_sum /= (double)(TR_SZ/10);
    printf("Test Data, average cost = %10.8f\n", cost_sum);
    print_weights_and_biases(n, 100);
    return 0;
}
