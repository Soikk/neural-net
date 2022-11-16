#include "neural.h"


void printMatrix(matrix *m){
	printf("%dx%d\n", m->rows, m->cols);
	for(int i = 0; i < m->rows; ++i){
		for(int j = 0; j < m->cols; ++j){
			printf("%.3Lf ", m->data[i][j]);
		}
		printf("\n");
	}
}

void printLayer(layer *l){
	printf("function: %d, inputs: %d, nneurons: %d\n", l->function, l->inputs, l->nneurons);
	printf("Weights\n");
	printMatrix(l->weights);
	printf("Bias\n");
	printMatrix(l->bias);
	printf("Neurons\n");
	printMatrix(l->neurons);
}

void printNet(net *n){
	printf("learningrate: %.3Lf, inputs: %d, outputs, %d, nlayers: %d\n", n->learningrate, n->inputs, n->outputs, n->nlayers);
	printf("input:\n");
	printMatrix(n->input);
	for(int i = 0; i < n->nlayers; ++i){
		printf("Layer %d:\n", i);
		printLayer(n->layers[i]);
	}
}

static long double linear(long double n){ return n; }

static long double derivedLinear(long double n){ return 0.0; }

static long double ReLu(long double n){ return fmaxl(0.0, n); }

static long double derivedReLu(long double n){ return n > 0; }

static long double sigmoid(long double n){ return 1/(1 + expl(-n)); }

static long double derivedSigmoid(long double n){ return n*(1 - n); }

static long double derivedTanhl(long double n){ return 1 - tanhl(n)*tanhl(n); }

static long double he(long double inputs){
	long long int scale = 10000000000;
	int r = rand()%(int)(sqrtl(2.0/inputs)*scale);
	return (long double)(r/scale);
}

static long double xavier(long double inputs){
}


static long double (*functions[])(long double) = {
	linear, ReLu, sigmoid, tanhl,
};

static long double (*derivedFunctions[])(long double) = {
	derivedLinear,derivedReLu, derivedSigmoid, derivedTanhl,
};

static long double placeholder(long double n){
	long double high = 1/sqrtl(n), low = (-1)/sqrtl(n);
	long double difference = high - low; // The difference between the two
	int scale = 10000;
	int scaled_difference = (int)(difference * scale);
	return low + (1.0 * (rand() % scaled_difference) / scale);
}

// Rework
void initializeLayer(layer *l){
	srand(time(NULL));
	// TODO implement different initialization functions (he, xavier)
	for(int i = 0; i < l->weights->rows; ++i){
		for(int j = 0; j < l->weights->cols; ++j){
			l->weights->data[i][j] = placeholder(l->nneurons);
		}
	}
}

static layer *newLayer(FUNCTIONS function, int inputs, int nneurons){
	layer *l = malloc(sizeof(layer));
	l->function = function,
	l->inputs = inputs;
	l->nneurons = nneurons;
	l->weights = newMatrix(inputs, nneurons);
	initializeLayer(l);
	l->bias = newMatrix(1, nneurons);
	fillMatrix(l->bias, 0);
	l->neurons = newMatrix(1, nneurons);
	fillMatrix(l->neurons, 0);
	return l;
}

static void freeLayer(layer **l){
	freeMatrix(&(*l)->weights);
	freeMatrix(&(*l)->bias);
	freeMatrix(&(*l)->neurons);
	free(*l);
	l = NULL;
}

static void saveLayer(layer *l, FILE *fp){
	char header = 'L';
	fwrite(&header, sizeof(char), 1, fp);
	fwrite(&l->function, sizeof(int), 1, fp);
	fwrite(&l->inputs, sizeof(int), 1, fp);
	fwrite(&l->nneurons, sizeof(int), 1, fp);
	saveMatrix(l->weights, fp);
	saveMatrix(l->bias, fp);
	saveMatrix(l->neurons, fp);
	char end = 'E';
	fwrite(&end, sizeof(char), 1, fp);
}

static layer *loadLayer(FILE *fp){
	char header;
	fread(&header, sizeof(char), 1, fp);
	if(header != 'L'){
		fprintf(stderr, "Header is '%c' not 'L'\n", header);
		exit(EXIT_FAILURE);
	}
	FUNCTIONS function;
	int inputs, nneurons;
	fread(&function, sizeof(int), 1, fp);
	fread(&inputs, sizeof(int), 1, fp);
	fread(&nneurons, sizeof(int), 1, fp);
	layer *l = malloc(sizeof(layer));
	l->function = function;
	l->inputs = inputs;
	l->nneurons = nneurons;
	l->weights = loadMatrix(fp);
	l->bias = loadMatrix(fp);
	l->neurons = loadMatrix(fp);
	char end;
	fread(&end, sizeof(char), 1, fp);
	if(end != 'E'){
		fprintf(stderr, "End is '%c' not 'E'\n", end);
		exit(EXIT_FAILURE);
	}
	return l;
}

net *newNet(FUNCTIONS function, long double learningrate, int inputs, int outputs, int nlayers, ...){
	// TODO check if outputs == last layer
	net *n = malloc(sizeof(net));
	n->learningrate = learningrate;
	n->inputs = inputs;
	n->outputs = outputs;
	n->nlayers = nlayers;
	n->input = newMatrix(1, inputs);
	fillMatrix(n->input, 1);
	n->layers = malloc(nlayers*sizeof(layer*));
	va_list layers;
	va_start(layers, nlayers);
	for(int i = 0; i < nlayers; ++i){
		int size = va_arg(layers, int);
		n->layers[i] = newLayer(function, inputs, size);
		inputs = size;
	}
	va_end(layers);
	return n;
}

void freeNet(net **n){
	freeMatrix(&(*n)->input);
	for(int i = 0; i < (*n)->nlayers; ++i){
		freeLayer(&(*n)->layers[i]);
	}
	free((*n)->layers);
	(*n)->layers = NULL;
	free(*n);
	n = NULL;
}

void saveNet(net *n, FILE *fp){
	char header = 'N';
	fwrite(&header, sizeof(char), 1, fp);
	fwrite(&n->learningrate, sizeof(long double), 1, fp);
	fwrite(&n->inputs, sizeof(int), 1, fp);
	fwrite(&n->outputs, sizeof(int), 1, fp);
	fwrite(&n->nlayers, sizeof(int), 1, fp);
	saveMatrix(n->input, fp);
	for(int i = 0; i < n->nlayers; ++i){
		saveLayer(n->layers[i], fp);
	}
	char end = 'E';
	fwrite(&end, sizeof(char), 1, fp);
}

net *loadNet(FILE *fp){
	char header;
	fread(&header, sizeof(char), 1, fp);
	if(header != 'N'){
		fprintf(stderr, "Header is '%c' not 'N'\n", header);
		exit(EXIT_FAILURE);
	}
	long double learningrate;
	int inputs, outputs, nlayers;
	fread(&learningrate, sizeof(long double), 1, fp);
	fread(&inputs, sizeof(int), 1, fp);
	fread(&outputs, sizeof(int), 1, fp);
	fread(&nlayers, sizeof(int), 1, fp);
	net *n = malloc(sizeof(net));
	n->learningrate = learningrate;
	n->inputs = inputs;
	n->outputs = outputs;
	n->nlayers = nlayers;
	n->input = loadMatrix(fp);
	n->layers = malloc(nlayers*sizeof(layer*));
	for(int i = 0; i < nlayers; ++i){
		n->layers[i] = loadLayer(fp);
	}
	char end;
	fread(&end, sizeof(char), 1, fp);
	if(end != 'E'){
		fprintf(stderr, "End is '%c' not 'E'\n", end);
		exit(EXIT_FAILURE);
	}
	return n;
}

static void applyFunction(func function, matrix *m){
	for(int i = 0; i < m->rows; ++i){
		for(int j = 0; j < m->cols; ++j){
			m->data[i][j] = function(m->data[i][j]);
		}
	}
}

static void propagateLayer(layer *l, matrix *inputs){
	matrix *m = multiplyMatrices(inputs, l->weights);
	matrix *a = addMatrices(m, l->bias);
	freeMatrix(&m);
	copyMatrix(l->neurons, a);
	freeMatrix(&a);
	applyFunction(functions[l->function], l->neurons);
}

matrix *propagate(net *n, matrix *input){
	n->input = input;
	for(int i = 0; i < n->nlayers; ++i){
		propagateLayer(n->layers[i], input);
		input = n->layers[i]->neurons;
	}
	return n->layers[n->nlayers-1]->neurons;
}

void backPropagate(net *n, matrix *expected){
	matrix **errors = malloc(n->nlayers*sizeof(matrix*));
	matrix *corrected = subtractMatrices(expected, n->layers[n->nlayers-1]->neurons);

	for(int i = n->nlayers-1; i >= 0; --i){
		matrix *derived = cloneMatrix(n->layers[i]->neurons);
		applyFunction(derivedFunctions[n->layers[i]->function], derived);
		errors[i] = HadamardProduct(corrected, derived);
		freeMatrix(&corrected);
		freeMatrix(&derived);
		
		matrix *transposedWeights = transpose(n->layers[i]->weights);
		corrected = multiplyMatrices(errors[i], transposedWeights);
		freeMatrix(&transposedWeights);
	}
	
	matrix *lastOutput = n->input;
	for(int i = 0; i < n->nlayers; ++i){
		matrix *transposedOutput = transpose(lastOutput);
		multiplyMatrix(transposedOutput, n->learningrate);
		matrix *weightChangeMatrix = multiplyMatrices(transposedOutput, errors[i]);
		freeMatrix(&transposedOutput);
		
		matrix *t = addMatrices(n->layers[i]->weights, weightChangeMatrix);
		freeMatrix(&weightChangeMatrix);
		copyMatrix(n->layers[i]->weights, t);
		freeMatrix(&t);
		
		multiplyMatrix(errors[i], n->learningrate);
		t = addMatrices(n->layers[i]->bias, errors[i]);
		copyMatrix(n->layers[i]->bias, t);
		freeMatrix(&t);
		
		lastOutput = n->layers[i]->neurons;
	}
	lastOutput = NULL;

	for(int i = 0; i < n->nlayers; ++i){
		freeMatrix(&errors[i]);
	}
	free(errors);
	errors = NULL;
	freeMatrix(&corrected);
}

void feedData(matrix *m, long double array[m->rows][m->cols]){
	for(int i = 0; i < m->rows; ++i){
		for(int j = 0; j < m->cols; ++j){
			m->data[i][j] = array[i][j];
		}
	}
}

matrix *imageToInput(image *im){
	matrix *r = newMatrix(1, im->img->rows*im->img->cols);
	int l = 0;
	for(int i = 0; i < im->img->rows; ++i){
		for(int j = 0; j < im->img->cols; ++j){
			r->data[0][l++] = im->img->data[i][j];
		}
	}
	return r;
}

static int maxOutputs(matrix *out){
	long double max = 0;
	int mi = -1;
	for(int i = 0; i < 10; ++i){
		if(out->data[0][i] > max){
			max = out->data[0][i];
			mi = i;
		}
	}
	return mi;
}

int main(){

	matrix **expectedOutputs = malloc(10*sizeof(matrix*));
	for(int i = 0; i < 10; ++i){
		expectedOutputs[i] = newMatrix(1, 10);
		fillMatrix(expectedOutputs[i], 0);
		expectedOutputs[i]->data[0][i] = 1;
	}

	// Training
	net *n = newNet(SIGMOID, 1, 784, 10, 2, 300, 10);
	
	int training = 0;
	FILE *trainFP = fopen("mnist_train.csv", "r");
	char line[MAXCHARS];
	fgets(line, MAXCHARS, trainFP);
	/*while(!feof(trainFP)){
		image *im = loadCSV(trainFP);
		propagate(n, imageToInput(im));
		backPropagate(n, expectedOutputs[im->label]);
		
		if(training%100 == 0){
			printf("\nTrained %d\n", training+1);
			printMatrix(n->layers[n->nlayers-1]->neurons);
			printf("\n");
		}
		
		freeImage(&im);
		++training;
	}
	fclose(trainFP);
	printMatrix(n->input);
	
	FILE *fp = fopen("nn", "wb");
	saveNet(n, fp);
	fclose(fp);*/
	freeNet(&n);
	
	FILE *fp = fopen("nn", "rb");
	n = loadNet(fp);
	fclose(fp);

	// Testing

	int testing = 0;
	FILE *testFP = fopen("mnist_test.csv", "r");
	//char line[MAXCHARS];
	fgets(line, MAXCHARS, testFP);
	
	int count = 0;
	while(!feof(testFP)){
		image *im = loadCSV(testFP);
		matrix *in = imageToInput(im);
		matrix *expectedOutput = expectedOutputs[im->label];
		matrix *output = propagate(n, in);
		if(testing%300 == 0){
			printf("Tested %d so far\n", testing+1);
		}
		/*printImage(im);
		printf("Expected:\n");
		printMatrix(expectedOutput);
		printf("Got:\n");
		printMatrix(output);
		printf("Max output: %d\n", maxOutputs(output));*/
		if(maxOutputs(output) == im->label){
			count++;
		}
		freeMatrix(&in);
		output = NULL;
		freeImage(&im);
		++testing;
		//if(testing == 10)
		//	break;
	}
	fclose(testFP);

	long double success = ((long double)count/testing)*100;
	printf("success rate: %Lf (%d/%d)\n", success, count, testing);
	

	return 0;
}
