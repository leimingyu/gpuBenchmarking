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

	std::string test_name;

	float a = 1.1f, b = 1.3f;

	kernel_global_load <<< 1, 1 >>> (d_a, d_start, d_end, arraylen, a, b);

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

	printf("LDG.E (ld.global)\t\t\t\t: %u (clks)\n", bar2 - 6 * mov_clk);

	cudaFree(d_a);
	cudaFree(d_start);
	cudaFree(d_end);

	cudaDeviceReset();
}


void global_load_v1()
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

	kernel_global_load_v1 <<< 1, 1 >>> (d_a, d_start, d_end, arraylen, a, b);

	cudaDeviceSynchronize();

	uint bar1 = d_end[0] - d_start[0];  // 3 mov : including store the clock val
	uint bar2 = d_end[1] - d_start[1];
	uint bar3 = d_end[2] - d_start[2];  // 3 mov

	printf("\n%s\n", test_name.c_str());
	printf("bar1 : %u \t bar2 :%u \t bar3: %u\n", bar1, bar2, bar3);

	// verify there is 3 mov = 3 * 15 clk / mov
	assert(bar1 == 45);
	assert(bar3 == 45);
	uint mov_clk = bar1 / 3;


	// 3 mov + 23 inst ( 2 imul +  LDG + 2 iadd + 1 fadd + 17 others)
	printf("LDG.E (ld.global): %u (clks)\n\n", bar2 - 3 * mov_clk - 17 * mov_clk - 2 * 15 - 2 * 86 - 1 * 15);

	cudaFree(d_a);
	cudaFree(d_start);
	cudaFree(d_end);

	cudaDeviceReset();
}


//-----------------------------------------------------------------------------
// global store
//-----------------------------------------------------------------------------
void global_store()
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

	kernel_global_store <<< 1, 1 >>> (d_a, d_start, d_end, arraylen, a, b);

	cudaDeviceSynchronize();

	uint bar1 = d_end[0] - d_start[0];  // 3 mov : including store the clock val
	uint bar2 = d_end[1] - d_start[1];  //
	uint bar3 = d_end[2] - d_start[2];  // 3 mov

	printf("\n%s\n", test_name.c_str());
	printf("bar1 : %u \t bar2 :%u \t bar3: %u\n", bar1, bar2, bar3);

	// verify there is 3 mov = 3 * 15 clk / mov
	assert(bar1 == 45);
	assert(bar3 == 45);
	uint mov_clk = bar1 / 3;

	//	(2 mov clock) + 2 mov + stg + (1mov clock)
	printf("STG (Global memory store: register to gm) : %u (clks)\n", bar2 - 5 * mov_clk);

	cudaFree(d_a);
	cudaFree(d_start);
	cudaFree(d_end);

	cudaDeviceReset();
}


//-----------------------------------------------------------------------------
// global store v1
//-----------------------------------------------------------------------------
void global_store_v1()
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

	kernel_global_store_v1 <<< 1, 1 >>> (d_a, d_start, d_end, arraylen, a, b);

	cudaDeviceSynchronize();

	uint bar1 = d_end[0] - d_start[0];  // 3 mov : including store the clock val
	uint bar2 = d_end[1] - d_start[1];  //
	uint bar3 = d_end[2] - d_start[2];  // 3 mov

	printf("\n%s\n", test_name.c_str());
	printf("bar1 : %u \t bar2 :%u \t bar3: %u\n", bar1, bar2, bar3);

	// verify there is 3 mov = 3 * 15 clk / mov
	assert(bar1 == 45);
	assert(bar3 == 45);
	uint mov_clk = bar1 / 3;


	//	(2 mov clock) + 2 mov + stg + (1mov clock)
	printf("STG (Global memory store: register to gm)\t: %u (clks)\n", bar2 - 3 * mov_clk - 20 * mov_clk - 2 * 86);

	cudaFree(d_a);
	cudaFree(d_start);
	cudaFree(d_end);

	cudaDeviceReset();
}



void shared_load()
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

	kernel_shared_load <<< 1, 1 >>> (d_a, d_start, d_end, arraylen, a, b);

	cudaDeviceSynchronize();

	uint bar1 = d_end[0] - d_start[0];  // 3 mov : including store the clock val
	uint bar2 = d_end[1] - d_start[1];
	uint bar3 = d_end[2] - d_start[2];  // 3 mov

	printf("\n%s\n", test_name.c_str());
	printf("bar1 : %u \t bar2 :%u \t bar3: %u\n", bar1, bar2, bar3);

	// verify there is 3 mov = 3 * 15 clk / mov
	assert(bar1 == 45);
	assert(bar3 == 45);
	uint mov_clk = bar1 / 3;

	// imul is 86 clocks on gtx 950
	// 2 mov + 23 ( 2 imul +  LDS + 20 others) + 1 mov
	printf("LDS (load from shared memory)\t\t\t: %u (clks)\n", bar2 - 3 * mov_clk - 20 * mov_clk - 2 * 86);

	cudaFree(d_a);
	cudaFree(d_start);
	cudaFree(d_end);

	cudaDeviceReset();
}



void shared_store()
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

	kernel_shared_store <<< 1, 1 >>> (d_a, d_start, d_end, arraylen, a, b);

	cudaDeviceSynchronize();

	uint bar1 = d_end[0] - d_start[0];  // 3 mov : including store the clock val
	uint bar2 = d_end[1] - d_start[1];  
	uint bar3 = d_end[2] - d_start[2];  // 3 mov

	printf("\n%s\n", test_name.c_str());
	printf("bar1 : %u \t bar2 :%u \t bar3: %u\n", bar1, bar2, bar3);

	// verify there is 3 mov = 3 * 15 clk / mov
	assert(bar1 == 45);
	assert(bar3 == 45);
	uint mov_clk = bar1 / 3;

	//2 mov + 24 (2 imul + STS + 21 others) + 1 mov
	printf("STS (store to shared memory)\t\t\t: %u (clks)\n", bar2 - 3 * mov_clk - 21 * mov_clk - 2 * 86);

	cudaFree(d_a);
	cudaFree(d_start);
	cudaFree(d_end);

	cudaDeviceReset();
}


void const_load()
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

	kernel_const_load<<< 1, 1 >>> (d_a, d_start, d_end, arraylen, a, b);

	cudaDeviceSynchronize();

	uint bar1 = d_end[0] - d_start[0];  // 3 mov : including store the clock val
	uint bar2 = d_end[1] - d_start[1];  // 
	uint bar3 = d_end[2] - d_start[2];  // 3 mov

	printf("\n%s\n", test_name.c_str());
	printf("bar1 : %u \t bar2 :%u \t bar3: %u\n", bar1, bar2, bar3);

	//// verify there is 3 mov = 3 * 15 clk / mov
	assert(bar1 == 45);
	assert(bar3 == 45);
	uint mov_clk = bar1 / 3;

	// 2mov + ( 29 others + 2 iadd + 2 imul )  + 1mov
	printf("LDC (Read from constant memory to register)\t: %u (clks)\n", bar2 - 3 * mov_clk - 29 * mov_clk - 2 * 15 - 2 * 86);

	cudaFree(d_a);
	cudaFree(d_start);
	cudaFree(d_end);

	cudaDeviceReset();
}


void local_load()
{
	srand(time(NULL));

	int arraylen = 32;

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

	kernel_local_load <<< 1, 1 >>> (d_a, d_start, d_end, arraylen, a, b);

	cudaDeviceSynchronize();

	uint bar1 = d_end[0] - d_start[0];  // 3 mov : including store the clock val
	uint bar2 = d_end[1] - d_start[1];  // (2 mov clock) + 5mov + fadd + LDL + (1mov clock)
	uint bar3 = d_end[2] - d_start[2];  // 3 mov

	printf("\n%s\n", test_name.c_str());
	printf("bar1 : %u \t bar2 :%u \t bar3: %u\n", bar1, bar2, bar3);

	////// verify there is 3 mov = 3 * 15 clk / mov
	assert(bar1 == 45);
	assert(bar3 == 45);
	uint mov_clk = bar1 / 3;

	// 8mov +  fadd + LDL
	printf("LDL (Read from local memory to register)\t: %u (clks)\n", bar2 - 8 * mov_clk - 15);

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
	//global_load();
	//global_store_v1();

	//------------------------------------------------------------------------//
	// shared 
	//------------------------------------------------------------------------//
	//shared_load();
	//shared_store();

	//------------------------------------------------------------------------//
	// constant memory read 
 	//------------------------------------------------------------------------//
	//const_load();
 
 	//------------------------------------------------------------------------//
	// local memory 
 	//------------------------------------------------------------------------//
	local_load();

	return 0;

}
