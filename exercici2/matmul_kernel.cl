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

// Matrix multiplication kernel
__kernel void MatMulKernel(
    __global float *A,
    __global float *B,
    __global float *C,
    int size
    ) {
	/* Codi de multiplicacio de les matrius A * B.
	Cada Work Item s'encarrega del resultat d'un punt de la matriu C.*/
	for (int y=0; y<size; y++){
		for (int x=0; x<size; x++) {
			C[(y*size)+x] = doAPoint(x,y,A,B,size,size);
		}
	}
}

