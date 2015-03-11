/* Host code */
/* Matrices are stored in row-major order: */
/* M(row, col) = *(M.elements + row * M.width + col) */

#include <stdio.h>
#include "../simple-opencl/simpleCL.h"

#define DBG 1
#define BLOCK_SIZE_H 32
#define BLOCK_SIZE_V 4
#define ALPHA 0.0001
#define MATRIX_SIZE 1024
#define DEVICE 32

/* Matrix multiplication - Host code */
/* Matrix dimensions are assumed to be multiples of BLOCK_SIZE */
float doAPoint(int x, int y, float* A, float *B, const int sizeAX, const int sizeBX) {
	/* We assume sizeAX == sizeBY */
	float result=0;
	int posA,posB;
		
	for (int step=0; step<sizeAX; step++){
	
	  posA = (y*sizeAX) + step ;
	  posB = (step*sizeBX) + x;
	
	  result = result + A[posA] *B[posB];
		
	}
	return result;
}

int main() {
	/* This code executes on the OpenCL host */
	int found;

	/* SimpleOpenCL types declaration */
	sclHard* hardware;
	sclSoft software;

	/* Host data */
	float *A = NULL;  // Input array
	float *B = NULL;  // Input array
	float *C = NULL;  // Output array
	float *Ctest = NULL; // Test array
	float diff;
	int count = 0;
	/* Size of the matrix */
	const int elements = MATRIX_SIZE * MATRIX_SIZE;

	/* Compute the size of the data */ 
	size_t datasize = sizeof(float) * elements;
	size_t localblocksize = sizeof(float) * BLOCK_SIZE_H * BLOCK_SIZE_V;

	/* Allocate space for input/output data */
	A = (float *) malloc(datasize);
	B = (float *) malloc(datasize);
	C = (float *) malloc(datasize);
	Ctest = (float *) malloc(datasize);

	/* Initialize the input data */
	for(int i=0; i < elements; i++) {
		A[i] = i*0.00001;
		B[i] = 0.001;
	}

	/* NDRange 2D size initialization*/
	size_t global_size[2];
	size_t local_size[2];

	global_size[0]=MATRIX_SIZE; global_size[1]=MATRIX_SIZE;
	local_size[0]=BLOCK_SIZE_H; local_size[1]=BLOCK_SIZE_V;
	/*local_size[0]=1 and local_size[1]=1 might be necessary for CPU and GPU devices on apple machines*/

	/* Inicialitzar hardware i software */
	found=0;
	hardware = sclGetAllHardware(&found);
	software = sclGetCLSoftware("matmul_kernel.cl","MatMulKernel",hardware[DEVICE]);

	/* Kernel execution */
	cl_event event = sclManageArgsLaunchKernel( hardware[DEVICE], software,
					global_size, local_size,
					"%r %r %w %N %N",
						datasize, (void*) A, 
						datasize, (void*) B, 
						datasize, (void*) C,
						localblocksize,
						localblocksize
						
				);

	cl_ulong time = sclGetEventTime(hardware[DEVICE], event);
	printf("\nAmb OpenCL ha tardat: %f segons\n", time/1000000000.f);

	
	if (DBG) {
		// Check results
		printf("\nCalculant resultats a CPU\n");
		for (int y=0; y<MATRIX_SIZE; y++){
			for (int x=0; x<MATRIX_SIZE; x++) {
				Ctest[(y*MATRIX_SIZE)+x] = doAPoint(x,y,A,B,MATRIX_SIZE,MATRIX_SIZE);
				printf("%f\t",Ctest[(y*MATRIX_SIZE)+x]); 
			}
			printf("\n");
		}

		// Check results
		printf("\nCalculant resultats al dispositiu %d\n", DEVICE);
		for (int y=0; y<MATRIX_SIZE; y++){
			for (int x=0; x<MATRIX_SIZE; x++) {
				printf("%f\t",C[(y*MATRIX_SIZE)+x]); 
			}
			printf("\n");
		}

		// Falta generar resultats per a la matriu C 
		for (int i=0; i<elements; i++){
			diff=C[i]-Ctest[i];

			if (diff>ALPHA){
				count++;
				printf("\nError a la posicio %d de C. Valor de C = %f. Valor de Ctest= %f.",i,C[i],Ctest[i]);
			}
		}
	  
		printf("\n\n");
		if(count > 0)
			printf("Errors %d\n",count);
	}
  
	return 0;
}




