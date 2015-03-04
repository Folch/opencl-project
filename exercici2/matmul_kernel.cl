// Matrix multiplication kernel
__kernel void MatMulKernel(
    __global float *A,
    __global float *B,
    __global float *C,
    int size
    ) {
	/* Codi de multiplicacio de les matrius A * B.
	Cada Work Item s'encarrega del resultat d'un punt de la matriu C.*/
	int posA,posB;
	
	for (int y=0; y<size; y++){
		for (int x=0; x<size; x++) {
			C[(y*size)+x] = 0;
			for (int step=0; step<size; ++step){
				posA = (y*size) + step ;
				posB = (step*size) + x;
				C[(y*size)+x] += A[posA] * B[posB];
				
			}
		}
	}
	barrier(CLK_GLOBAL_MEM_FENCE);
}

