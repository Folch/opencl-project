// Matrix multiplication kernel
__kernel void MatMulKernel(
    __global float *A,
    __global float *B,
    __global float *C
    )
{
	int idX = get_global_id(0);
	int idY = get_global_id(1);
	int size = get_global_size(0);

	/* Codi de multiplicacio de les matrius A * B.
	Cada Work Item s'encarrega del resultat d'un punt de la matriu C.*/
	
	C[idY * size + idX] = 0;
	int posA,posB;
	for (int step=0; step<size; ++step){
		posA = (idY * size) + step ;
		posB = (step * size) + idX;
		C[idY * size + idX] += A[posA] * B[posB];
		
	}
	barrier(CLK_GLOBAL_MEM_FENCE);	
}

