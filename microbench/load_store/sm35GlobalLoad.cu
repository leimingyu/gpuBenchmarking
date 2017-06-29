#include <stdio.h>
#include <stdlib.h>                                                             
#include <string.h>                                                             
#include <math.h>                                                               
#include <assert.h>
#include <time.h>                                                               
#include <iostream>
#include <string>


#include <cuda.h>
#include <cuda_runtime.h>
#include <device_functions.h> 

#include "kernel_GlobalLoad.cu"

void global_load()
{
	srand(time(NULL));

	int arraylen = 8;

	float *d_a = NULL;
	cudaMallocManaged(&d_a, sizeof(float) * (arraylen + 1)); 

	// clocks
	uint *d_start = NULL;
	cudaMallocManaged(&d_start, sizeof(uint) * (arraylen + 1)); 

	uint *d_end = NULL;
	cudaMallocManaged(&d_end, sizeof(uint) * (arraylen + 1)); 

	// init
	for(int i=0; i<=arraylen; i++)
	{
		d_a[i] = static_cast<float>(rand());
	}

	cudaDeviceSynchronize();

	float a = 1.1f, b = 1.3f;

	kernel_global_load <<< 1, 1 >>> (d_a, d_start, d_end, arraylen, a, b);

	cudaDeviceSynchronize();

	uint bar1 = d_end[0] - d_start[0];  // 1 mov 
	uint bar2 = d_end[1] - d_start[1];  // 4 mov + LD.E 
	uint bar3 = d_end[2] - d_start[2];  // 1 mov

	printf("\nsm35 : Global Load\n");
	printf("bar1 : %u \t bar2 :%u \t bar3: %u\n", bar1, bar2, bar3);

	// verify mov clocks : 25 on k40c, 30 on k20c
	//assert(bar1 == 25);
	//assert(bar3 == 25);

	uint mov_clk = bar1;

	printf("LD.E (ld.global)\t\t\t\t: %u (clks)\n", bar2 - 4 * mov_clk);

	cudaFree(d_a);
	cudaFree(d_start);
	cudaFree(d_end);

	cudaDeviceReset();
}


int main(int argc, char **argv) {

	int devid=0;
	if(argc == 2)
		devid = atoi(argv[1]);

	cudaSetDevice(devid);

	cudaDeviceProp prop;
	cudaGetDeviceProperties(&prop, devid);
	printf("Device name: %s\n", prop.name);


	//------------------------------------------------------------------------//
	// global 
	//------------------------------------------------------------------------//
	global_load();

	return 0;

}
