#include "neural_net.h"
#include "matrix_opers.h"
#include <cmath>

#include <fcntl.h>

#include <ctime>
#include <cstdio>
#include <cstdlib>

// Declaring these as constants to avoid
// unnecessary runtime allocation
static const int COUNT_TRAINING_EXAMPLES = 750;

static const int N_LAYERS = 4;

static const int SIZE_INPUT_LAYER      = 400;
static const int SIZE_HIDDEN_1_LAYER   = 150;
static const int SIZE_HIDDEN_2_LAYER   = 70;
static const int SIZE_OUTPUT_LAYER     = 10;

// since this is vectorized implementation each
// layer must have enough space to hold the activations
// of each of the training examples. In practice it is
// possible to use just single example (in case of large training set)
// for each pass but the gradient should be updated manually.

static double A1[COUNT_TRAINING_EXAMPLES*(SIZE_INPUT_LAYER+1)];
static double A2[COUNT_TRAINING_EXAMPLES*SIZE_HIDDEN_1_LAYER];
static double A3[COUNT_TRAINING_EXAMPLES*SIZE_HIDDEN_2_LAYER];
static double A4[COUNT_TRAINING_EXAMPLES*SIZE_OUTPUT_LAYER];

static double AB2[COUNT_TRAINING_EXAMPLES*(1+SIZE_HIDDEN_1_LAYER)];
static double AB3[COUNT_TRAINING_EXAMPLES*(1+SIZE_HIDDEN_2_LAYER)];
static double AB4[COUNT_TRAINING_EXAMPLES*(1+SIZE_OUTPUT_LAYER)];

static double Classes[COUNT_TRAINING_EXAMPLES*SIZE_OUTPUT_LAYER];

static const int WEIGHTS_SIZE  =
                (1+SIZE_INPUT_LAYER)*SIZE_HIDDEN_1_LAYER +
                (1+SIZE_HIDDEN_1_LAYER)*SIZE_HIDDEN_2_LAYER +
                (1+SIZE_HIDDEN_2_LAYER)*SIZE_OUTPUT_LAYER;

static double W_all[WEIGHTS_SIZE];
static double G_all[WEIGHTS_SIZE];

void    RandomInit      (double *begin, double *end, double sr, double er);
int     updateWeights   (double *W, double *G, int sz, double alph);
void    load            (const char *fname, double *p, size_t sz);

int main(void)
{
    srand(time(0));

    load("750x401.double", A1, sizeof(A1));
    load("750x10.double", Classes, sizeof(Classes));

    double *A_aggr[3] = { A2, A3, A4 };
    double *AB_aggr[3] = { AB2, AB3, AB4 };

    int offset = 0;
    double *W1 = &W_all[offset];
    offset+=(1+SIZE_INPUT_LAYER)*SIZE_HIDDEN_1_LAYER;
    RandomInit(&W1[0], &W1[offset],
               (-1.0 / sqrt(SIZE_INPUT_LAYER))+(-0.1f), (1.0 / sqrt(SIZE_INPUT_LAYER))+0.1f);

    double *W2 = &W_all[offset];
    offset+=(1+SIZE_HIDDEN_1_LAYER)*SIZE_HIDDEN_2_LAYER;
    RandomInit(&W2[0], &W2[offset],
               (-1.0 / sqrt(SIZE_HIDDEN_1_LAYER))+(-0.1f), (1.0 / sqrt(SIZE_HIDDEN_1_LAYER))+0.1f);

    double *W3 = &W_all[offset];
    offset+=(1+SIZE_HIDDEN_2_LAYER)*SIZE_OUTPUT_LAYER;
    RandomInit(&W3[0], &W3[offset],
               (-1.0 / sqrt(SIZE_HIDDEN_2_LAYER))+(-0.1f), (1.0 / sqrt(SIZE_HIDDEN_2_LAYER))+0.1f);

    double *W[] = { W1, W2, W3 };

    offset = 0;
    double *G1 = &G_all[0];
    offset+=(1+SIZE_INPUT_LAYER)*SIZE_HIDDEN_1_LAYER;
    double *G2 = &G_all[offset];
    offset+=(1+SIZE_HIDDEN_1_LAYER)*SIZE_HIDDEN_2_LAYER;
    double *G3 = &G_all[offset];

    double *G[] = { G1, G2, G3 };

    gnn::nnDef def(4,(const int[]){
                    SIZE_INPUT_LAYER,
                    SIZE_HIDDEN_1_LAYER,
                    SIZE_HIDDEN_2_LAYER,
                    SIZE_OUTPUT_LAYER
                   });

    const double *A_aggrC[] = { A1, A2, A3, A4 };
    const double *AB_aggrC[] = { A1, AB2, AB3, AB4 };

    // So here I'm using gradient descent just for
    // the test, but in practice it is better to use
    // something like conjugate gradient of bfgs
    for(int i = 0 ; i < 250 ; i++) {

        gnn::feedForward(def, A1, (const double **)W,
                         COUNT_TRAINING_EXAMPLES,
                         A_aggr, AB_aggr);

        double c = gnn::cost(def, A4, Classes, COUNT_TRAINING_EXAMPLES);

        printf("%f\n", c);

        gnn::backpropagate(def, A_aggrC, AB_aggrC,
                           Classes, (const double**)W,
                           G, COUNT_TRAINING_EXAMPLES);

        // replace this with blas routine
        updateWeights(W_all, G_all, WEIGHTS_SIZE, 0.5);

    }
}

float RandomFloat(float a, float b) {
    float random = ((float) rand()) / (float) RAND_MAX;
    float diff = b - a;
    float r = random * diff;
    return a + r;
}

void RandomInit(double * begin, double *end,
                double rs, double re)
{
    for(double * i = begin ; i != end ; i++) {
        i[0]=RandomFloat(rs, re);
    }
}

int updateWeights(double *W, double *G, int sz, double alph)
{
    for(int i = 0 ; i < sz ; i++)
    {
        W[i] = W[i] - alph*G[i];
    }
}


void load(const char *fname, double *p, size_t sz)
{
    long long flags = O_RDONLY;
#ifdef __MINGW32__
    flags |= _O_BINARY;
#endif // __MINGW32__
    int fd = open(fname, O_RDONLY, 0);
    read(fd, p, sz);
    close(fd);
}
