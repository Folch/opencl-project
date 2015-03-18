// Matrix multiplication kernel
__kernel void MatMulKernel(
    __global float *A,
    __global float *B,
    __global float *C
    )
{
	
	
	int posA,posB;
	int idX = get_global_id(0);
	int size = get_global_size(0);
	
	for (int i=0; i<size; ++i){
		C[i * size + idX] = 0;
		for (int step=0; step<size; ++step){
			posA = (i * size) + step ;
			posB = (step * size) + idX;
			C[i * size + idX] += A[posA] * B[posB];
		
		}
	}

	barrier(CLK_GLOBAL_MEM_FENCE);	
}

