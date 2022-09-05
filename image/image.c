#include "image.h"


image *loadCSV(FILE *fp){
	image *im = malloc(sizeof(image));
	char line[MAXCHARS];
	fgets(line, MAXCHARS, fp);
	im->label = atoi(strtok(line, ","));
	matrix *m = newMatrix(28, 28);
	for(int i = 0; i < 28; ++i){
		for(int j = 0; j < 28; ++j){
			m->data[i][j] = strtod(strtok(NULL, ","), NULL)/256;
		}
	}
	im->img = m;
	return im;
}

void freeImage(image **im){
	freeMatrix(&((*im)->img));
	free(*im);
	*im = NULL;
}

void printImage(image *im){
	printf("%d, %d\n", im->img->rows, im->img->cols);
	printf("%d\n", im->label);
	for(int i = 0; i < im->img->rows; ++i){
		for(int j = 0; j < im->img->cols; ++j){
			printf("%c ", (im->img->data[i][j] > 0) ? '1' : 0 );
		}
		printf("\n");
	}
}
/*int main(){

	FILE *fp = fopen("../mnist_train.csv", "r");
	char line[MAXCHARS];
	fgets(line, MAXCHARS, fp);
	image *im = loadMnist_train(fp);
	
	printf("%d, %d\n", im->img->rows, im->img->cols);
	printf("%d\n", im->label);
	for(int i = 0; i < im->img->rows; ++i){
		if(i%28 == 0)
			printf("\n");
		for(int j = 0; j < im->img->cols; ++j){
			printf("%c ", (im->img->data[i][j] > 0) ? '1' : 0 );
		}
	}

	return 0;
}*/
