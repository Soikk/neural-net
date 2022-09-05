#pragma once
#ifndef IMAGE_H
#define IMAGE_H

#define __MINGW_FEATURES__ 1

#include <string.h>
#include "../../matrix/matrix.h"

#define MNIST_SIZE 784
#define MAXCHARS 10000


typedef struct image{
	int label;
	matrix *img;
} image;

image *loadCSV(FILE *fp);

void freeImage(image **im);

void printImage(image *im);

#endif
