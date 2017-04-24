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

#include "kernels.cu"

void load_global()
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

	std::string test_name;

	float a = 1.1f, b = 1.3f;

	kernel_load_global <<< 1, 1 >>> (d_a, d_start, d_end, arraylen, a, b);

	cudaDeviceSynchronize();

	uint bar1 = d_end[0] - d_start[0];  // 3 mov : including store the clock val
	uint bar2 = d_end[1] - d_start[1];  // (2 mov clock) + 2mov + inst + 1 mov + (1mov clock)
	uint bar3 = d_end[2] - d_start[2];  // 3 mov

	printf("\n%s\n", test_name.c_str());
	printf("bar1 : %u \t bar2 :%u \t bar3: %u\n", bar1, bar2, bar3);

	// verify there is 3 mov = 3 * 15 clk / mov
	assert(bar1 == 45);
	assert(bar3 == 45);
	uint mov_clk = bar1 / 3;

	printf("LDG.E (ld.global): %u (clks)\n", bar2 - 6 * mov_clk);

	cudaFree(d_a);
	cudaFree(d_start);
	cudaFree(d_end);

	cudaDeviceReset();

}

void load_shared()
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

	std::string test_name;

	float a = 1.1f, b = 1.3f;

	kernel_load_shared <<< 1, 1 >>> (d_a, d_start, d_end, arraylen, a, b);

	cudaDeviceSynchronize();

	uint bar1 = d_end[0] - d_start[0];  // 3 mov : including store the clock val
	// 22 (s2r/imull32i/mov/iadd) + lds.u.32 = 2 imul + lds.u.32 + 20 other inst
	uint bar2 = d_end[1] - d_start[1];
	uint bar3 = d_end[2] - d_start[2];  // 3 mov

	printf("\n%s\n", test_name.c_str());
	printf("bar1 : %u \t bar2 :%u \t bar3: %u\n", bar1, bar2, bar3);

	// verify there is 3 mov = 3 * 15 clk / mov
	assert(bar1 == 45);
	assert(bar3 == 45);
	uint mov_clk = bar1 / 3;

	// imul is 86 clocks on gtx 950
	printf("LDS (load from shared memory) : %u (clks)\n", bar2 - 20 * mov_clk - 2 * 86);

	cudaFree(d_a);
	cudaFree(d_start);
	cudaFree(d_end);

	cudaDeviceReset();
}

//-----------------------------------------------------------------------------
// global store
//-----------------------------------------------------------------------------
void store_global()
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

	std::string test_name;

	float a = 1.1f, b = 1.3f;

	kernel_store_global <<< 1, 1 >>> (d_a, d_start, d_end, arraylen, a, b);

	cudaDeviceSynchronize();

	uint bar1 = d_end[0] - d_start[0];  // 3 mov : including store the clock val
	uint bar2 = d_end[1] - d_start[1];  // (2 mov clock) + 2 mov + stg + (1mov clock)
	uint bar3 = d_end[2] - d_start[2];  // 3 mov

	printf("\n%s\n", test_name.c_str());
	printf("bar1 : %u \t bar2 :%u \t bar3: %u\n", bar1, bar2, bar3);

	// verify there is 3 mov = 3 * 15 clk / mov
	assert(bar1 == 45);
	assert(bar3 == 45);
	uint mov_clk = bar1 / 3;

	// st.global.f32
	printf("st.global : %u (clks)\n", bar2 - 5 * mov_clk);

	// use st.f32
	//printf("st.global : %u (clks)\n", bar2 - 11 * mov_clk);

	cudaFree(d_a);
	cudaFree(d_start);
	cudaFree(d_end);

	cudaDeviceReset();
}


void store_shared()
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

	std::string test_name;

	float a = 1.1f, b = 1.3f;

	kernel_store_shared <<< 1, 1 >>> (d_a, d_start, d_end, arraylen, a, b);

	cudaDeviceSynchronize();

	uint bar1 = d_end[0] - d_start[0];  // 3 mov : including store the clock val
	uint bar2 = d_end[1] - d_start[1];  // (2 mov clock) + 23 mov + STS + (1mov clock)
	uint bar3 = d_end[2] - d_start[2];  // 3 mov

	printf("\n%s\n", test_name.c_str());
	printf("bar1 : %u \t bar2 :%u \t bar3: %u\n", bar1, bar2, bar3);

	// verify there is 3 mov = 3 * 15 clk / mov
	assert(bar1 == 45);
	assert(bar3 == 45);
	uint mov_clk = bar1 / 3;

	// 2 imul + 21 others
	printf("STS (store to shared memory): %u (clks)\n", bar2 - 21 * mov_clk - 2 * 86);

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
	// load
	//------------------------------------------------------------------------//
	//load_global();
	//load_shared();

	//------------------------------------------------------------------------//
	// store 
	//------------------------------------------------------------------------//
	//store_global();
	store_shared();

	return 0;

}
