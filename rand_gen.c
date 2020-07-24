#include <inttypes.h>
#include <math.h>
#include <stdio.h>
#include "rand_gen.h"

#define PI 3.14159265358979323846

uint64_t s[4];

static uint64_t
splitmix64(uint64_t *x)
{
    uint64_t z = (*x += 0x9e3779b97f4a7c15);
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9;
    z = (z ^ (z >> 27)) * 0x94d049bb133111eb;
    return z ^ (z >> 31);
}

void
seed(void)
{
    uint64_t init;
    FILE *noise = fopen("/dev/urandom", "r");
    fread(&init, 8, 1, noise);
    fclose(noise);

    s[0] = splitmix64(&init);
    s[1] = splitmix64(&init);
    s[2] = splitmix64(&init);
    s[3] = splitmix64(&init);
}

static inline uint64_t
rotl(const uint64_t x, int k)
{
    return (x << k) | (x >> (64 - k));
}

uint64_t
rand_int64(void)
{
    const uint64_t result = rotl(s[1] * 5, 7) * 9;

    const uint64_t t = s[1] << 17;

    s[2] ^= s[0];
    s[3] ^= s[1];
    s[1] ^= s[2];
    s[0] ^= s[3];

    s[2] ^= t;

    s[3] = rotl(s[3], 45);

    return result;
}

int
rand_int_range(int low, int high)
{
    uint64_t draw = rand_int64();
    int mod = high - low + 1;
    return (low + (draw % mod));
}

double
rand_double(void)
{
    uint64_t r_int = rand_int64();
    double out = r_int / (double)UINT64_MAX;
    return out;
}

double
rand_normal(double mu, double var)
{
    double u1 = rand_double();
    double u2 = rand_double();
    double coeff = sqrt(-2.0 * log(u1));
    double sine = sin(2.0 * PI * u2);
    return (coeff * sine * var) + mu;
}
