#pragma once
#ifndef NEURAL_H
#define NEURAL_H

#define __MINGW_FEATURES__ 1

#include <math.h>
#include <time.h>
#include <stdlib.h>
#include <stdarg.h>
#include "../matrix/matrix.h"
#include "image/image.h"


typedef enum{
	LINEAR, RELU, SIGMOID, TANH,
	//HE, XAVIER,
} FUNCTIONS;

typedef struct layer{
	FUNCTIONS function;
	int inputs;
	int nneurons;
	matrix *weights;
	matrix *bias;
	matrix *neurons;
} layer;

typedef struct net{
	long double learningrate;
	int inputs;
	int outputs;
	int nlayers;
	matrix *input;
	layer **layers;
} net;

typedef long double (*func)(long double);


net *newNet(FUNCTIONS function, long double learningrate, int inputs, int outputs, int nlayers, ...);

void saveNet(net *n, FILE *fp);

net *loadNet(FILE *fp);

matrix *propagate(net *n, matrix *input);

void backPropagate(net *n, matrix *expected);

void feedData(matrix *m, long double array[m->rows][m->cols]);

matrix *imageToInput(image *im);

#endif
