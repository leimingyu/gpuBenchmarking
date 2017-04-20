#include <stdio.h>
#include <stdlib.h>                                                             
#include <string.h>                                                             
#include <assert.h>                                                             
#include <math.h>                                                               
#include <iostream>
#include <string>


#include <cuda.h>
#include <cuda_runtime.h>
#include <device_functions.h> 

#include "repeat.h"

#include "kernel_uint.cu"
#include "kernel_int.cu"
#include "kernel_float.cu"
#include "kernel_double.cu"

#define FMIN(x,y) (((x)<(y))?(x):(y)) 
#define FMAX(x,y) (((x)>(y))?(x):(y)) 


void uint_test(UINT_OP OP_);
void int_test(INT_OP OP_);
void float_test(FLOAT_OP OP_);
void double_test(DOUBLE_OP OP_);

int main(int argc, char **argv) {

	int devid=0;
	if(argc == 2)
		devid = atoi(argv[1]);

	cudaSetDevice(devid);

	cudaDeviceProp prop;                                                        
	cudaGetDeviceProperties(&prop, devid);                                      
	printf("\n---------------------------------------------"
		   "\nRun microbenchmarks on : %s"
		   "\n---------------------------------------------\n\n", prop.name);  

	//----------------------------------------//
	// integer 
	//----------------------------------------//
	int_test(INTADD);
	int_test(INTSUB);
	int_test(INTMIN);
	int_test(INTMAX);
	int_test(INTSAD);
	int_test(INTMUL);
	int_test(INTMAD); // mul + add
	int_test(INTSET);
	int_test(INTSHL);
	int_test(INTSHR);

	// todo
	//int_test(INTDIV);
	//int_test(INTREM);
	//int_test(INTMUL24);
	//int_test(INTMULHI);

	//----------------------------------------//
	// unsigned integer 
	//----------------------------------------//
	//uint_test(UINTADD);
	//uint_test(UINTSUB);
	//uint_test(UINTMAD);
	//uint_test(UINTMUL);
	//uint_test(UINTDIV);
	//uint_test(UINTREM);
	//uint_test(UINTMIN);
	//uint_test(UINTMAX);
	//uint_test(UINTAND);
	//uint_test(UINTOR);
	//uint_test(UINTXOR);
	//uint_test(UINTSHL);
	//uint_test(UINTSHR);
	//uint_test(UINTUMUL24);
	//uint_test(UINTUMULHI);
	//uint_test(UINTUSAD);


	//----------------------------------------//
	// float 
	//----------------------------------------//
	//float_test(FLOATADD);
	//float_test(FLOATMUL);
	//float_test(FLOATMIN);
	//float_test(FLOATMAX);
	//// page 72, https://www.nvidia.com/content/CUDA-ptx_isa_1.4.pdf
	//float_test(FLOATCMP); 	// fset
	//float_test(FLOATFMA);

	// todo
	//float_test(FLOATSUB);
	//float_test(FLOATMAD);
	//float_test(FLOATDIV);
	//float_test(FLOATFADDRN);
	//float_test(FLOATFADDRZ);
	//float_test(FLOATFMULRN);
	//float_test(FLOATFMULRZ);
	//float_test(FLOATFDIVIDEF);


	//----------------------------------------//
	// double
	//----------------------------------------//
	//double_test(DOUBLEADD);
	//double_test(DOUBLEMUL);
	//double_test(DOUBLEMIN);
	//double_test(DOUBLEMAX);
	//double_test(DOUBLEFMA);

	// todo
	//double_test(DOUBLESUB);
	//double_test(DOUBLEMAD);
	//double_test(DOUBLEMUL);
	//double_test(DOUBLEDIV);
	//double_test(DOUBLEMIN);
	//double_test(DOUBLEMAX);
	//double_test(DOUBLEDADDRN);

	return 0;

}

void uint_test(UINT_OP OP_)
{
	uint *d_a = NULL;
	cudaMallocManaged(&d_a, sizeof(uint) * 64);  // todo: change it back to 1  

	uint *d_start = NULL;
	cudaMallocManaged(&d_start, sizeof(uint) * 3); 

	uint *d_end = NULL;
	cudaMallocManaged(&d_end, sizeof(uint) * 3); 

	uint a = 6;
	uint b = 3;

	for(int i=0; i<64; i++)
	{
		d_a[i] = static_cast<uint>(i);
	}

	cudaDeviceSynchronize();

	std::string test_name;

	if(OP_ == UINTADD) {
		test_name = "UINT ADD";
		uint_add <<< 1, 1 >>> (d_a, a, b, d_start, d_end);
	} else if (OP_ == UINTSUB) {
		test_name = "UINT SUB";
		uint_sub <<< 1, 1 >>> (d_a, a, b, d_start, d_end);
	} else if (OP_ == UINTMAD) {
		test_name = "UINT MAD";
		uint_mad <<< 1, 1 >>> (d_a, a, b, d_start, d_end);
	} else if (OP_ == UINTMUL) {
		test_name = "UINT MUL";
		uint_mul <<< 1, 1 >>> (d_a, a, b, d_start, d_end);
	} else if (OP_ == UINTDIV) {
		test_name = "UINT DIV";
		uint_div <<< 1, 1 >>> (d_a, a, b, d_start, d_end);
	} else if (OP_ == UINTREM) {
		test_name = "UINT REM";
		uint_rem <<< 1, 1 >>> (d_a, a, b, d_start, d_end);
	} else if (OP_ == UINTMIN) {
		test_name = "UINT MIN";
		uint_min <<< 1, 1 >>> (d_a, a, b, d_start, d_end);
	} else if (OP_ == UINTMAX) {
		test_name = "UINT MAX";
		uint_max <<< 1, 1 >>> (d_a, a, b, d_start, d_end);
	} else if (OP_ == UINTAND) {
		test_name = "UINT AND";
		uint_and <<< 1, 1 >>> (d_a, a, b, d_start, d_end);
	} else if (OP_ == UINTOR) {
		test_name = "UINT OR";
		uint_or <<< 1, 1 >>> (d_a, a, b, d_start, d_end);
	} else if (OP_ == UINTXOR) {
		test_name = "UINT XOR";
		uint_xor <<< 1, 1 >>> (d_a, a, b, d_start, d_end);
	} else if (OP_ == UINTSHL) {
		test_name = "UINT SHL";
		uint_shl <<< 1, 1 >>> (d_a, a, b, d_start, d_end);
	} else if (OP_ == UINTSHR) {
		test_name = "UINT SHR";
		uint_shr <<< 1, 1 >>> (d_a, a, b, d_start, d_end);
	} else if (OP_ == UINTUMUL24) {
		test_name = "UINT UMUL24";
		uint_umul24 <<< 1, 1 >>> (d_a, a, b, d_start, d_end);
	} else if (OP_ == UINTUMULHI) {
		test_name = "UINT UMULHI";
		uint_umulhi <<< 1, 1 >>> (d_a, a, b, d_start, d_end);
	} else if (OP_ == UINTUSAD) {
		test_name = "UINT USAD";
		uint_usad <<< 1, 1 >>> (d_a, a, b, d_start, d_end);
	}


	cudaDeviceSynchronize();

	uint first_cycles = d_end[0] - d_start[0];
	uint second_cycles = d_end[1] - d_start[1];
	uint inst_cyc = second_cycles - first_cycles;

	printf("%s : %u (clk/warp)\n", test_name.c_str(), inst_cyc);

	cudaFree(d_a);
	cudaFree(d_start);
	cudaFree(d_end);

	cudaDeviceReset();
}

void int_test(INT_OP OP_)
{
	int *d_a = NULL;
	cudaMallocManaged(&d_a, sizeof(int) * 64); 

	uint *d_start = NULL;
	cudaMallocManaged(&d_start, sizeof(uint) * 3); 

	uint *d_end = NULL;
	cudaMallocManaged(&d_end, sizeof(uint) * 3); 

	int a = 6;
	int b = 3;

	for(int i=0; i<64; i++)
	{
		d_a[i] = i;
	}

	cudaDeviceSynchronize();

	std::string test_name;

	if(OP_ == INTADD) {
		test_name = "INT ADD";
		int_add <<< 1, 1 >>> (d_a, a, b, d_start, d_end);
	} else if (OP_ == INTSUB) {
		test_name = "INT SUB";
		int_sub <<< 1, 1 >>> (d_a, a, b, d_start, d_end);
	} else if (OP_ == INTMAD) {
		test_name = "INT MAD";
		int_mad <<< 1, 1 >>> (d_a, a, b, d_start, d_end);
	} else if (OP_ == INTMUL) {
		test_name = "INT MUL";
		int_mul <<< 1, 1 >>> (d_a, a, b, d_start, d_end);
	} else if (OP_ == INTDIV) {
		test_name = "INT DIV";
		int_div <<< 1, 1 >>> (d_a, a, b, d_start, d_end);
	} else if (OP_ == INTREM) {
		test_name = "INT REM";
		int_rem <<< 1, 1 >>> (d_a, a, b, d_start, d_end);
	} else if (OP_ == INTMIN) {
		test_name = "INT MIN";
		int_min <<< 1, 1 >>> (d_a, a, b, d_start, d_end);
	} else if (OP_ == INTMAX) {
		test_name = "INT MAX";
		int_max <<< 1, 1 >>> (d_a, a, b, d_start, d_end);
	} else if (OP_ == INTMUL24) {
		test_name = "INT MUL24";
		int_mul24 <<< 1, 1 >>> (d_a, a, b, d_start, d_end);
	} else if (OP_ == INTMULHI) {
		test_name = "INT MULHI";
		int_mulhi <<< 1, 1 >>> (d_a, a, b, d_start, d_end);
	} else if (OP_ == INTSAD) {
		test_name = "INT SAD";
		int_sad <<< 1, 1 >>> (d_a, a, b, d_start, d_end);
	} else if (OP_ == INTSHL) {
		test_name = "INT SHL";
		int_shl<<< 1, 1 >>> (d_a, a, b, d_start, d_end);
	} else if (OP_ == INTSHR) {
		test_name = "INT SHR";
		int_shr<<< 1, 1 >>> (d_a, a, b, d_start, d_end);
	} else if (OP_ == INTSET) {
		test_name = "INT SET (set equal)";
		int_set<<< 1, 1 >>> (d_a, a, b, d_start, d_end);
	}

	cudaDeviceSynchronize();

	uint bar1 = d_end[0] - d_start[0];  // 3 mov : including store the clock val
	uint bar2 = d_end[1] - d_start[1];  // 2 mov + inst + 2 mov
	uint bar3 = d_end[2] - d_start[2];  // 3 mov

	printf("\n%s\n", test_name.c_str());
	printf("bar1 : %u \t bar2 :%u \t bar3: %u\n", bar1, bar2, bar3);

	// verify there is 3 mov = 3 * 15 clk / mov
	assert(bar1 == 45);
	assert(bar3 == 45);
	uint mov_clk = bar1 / 3;

	if (OP_ == INTMAD) {
		printf("=>Warning: int_mad (2mov + imul + iadd + 3mov).\n");
		printf("%s: %u (clks)\n", test_name.c_str() ,bar2 - 5 * mov_clk);

	} else {
		printf("%s: %u (clks)\n", test_name.c_str() ,bar2 - 4 * mov_clk);
	}


	cudaFree(d_a);
	cudaFree(d_start);
	cudaFree(d_end);

	cudaDeviceReset();
}


void float_test(FLOAT_OP OP_)
{
	float *d_a = NULL;
	cudaMallocManaged(&d_a, sizeof(float) * 64); 

	uint *d_start = NULL;
	cudaMallocManaged(&d_start, sizeof(uint) * 3); 

	uint *d_end = NULL;
	cudaMallocManaged(&d_end, sizeof(uint) * 3); 

	float a = 6;
	float b = 3;

	for(int i=0; i<64; i++)
	{
		d_a[i] = static_cast<float>(i);
	}

	cudaDeviceSynchronize();

	std::string test_name;

	if(OP_ == FLOATADD) {
		test_name = "FLOAT ADD";
		float_add <<< 1, 1 >>> (d_a, a, b, d_start, d_end);
	} else if (OP_ == FLOATSUB) {
		test_name = "FLOAT SUB";
		float_sub <<< 1, 1 >>> (d_a, a, b, d_start, d_end);
	} else if (OP_ == FLOATMAD) {
		test_name = "FLOAT MAD";
		float_mad <<< 1, 1 >>> (d_a, a, b, d_start, d_end);
	} else if (OP_ == FLOATMUL) {
		test_name = "FLOAT MUL";
		float_mul <<< 1, 1 >>> (d_a, a, b, d_start, d_end);
	} else if (OP_ == FLOATDIV) {
		test_name = "FLOAT DIV";
		float_div <<< 1, 1 >>> (d_a, a, b, d_start, d_end);
	} else if (OP_ == FLOATMIN) {
		test_name = "FLOAT MIN";
		float_min <<< 1, 1 >>> (d_a, a, b, d_start, d_end);
	} else if (OP_ == FLOATMAX) {
		test_name = "FLOAT MAX";
		float_max <<< 1, 1 >>> (d_a, a, b, d_start, d_end);
	} else if (OP_ == FLOATFADDRN) {
		test_name = "FLOAT FADDRN";
		float_faddrn <<< 1, 1 >>> (d_a, a, b, d_start, d_end);
	} else if (OP_ == FLOATFADDRZ) {
		test_name = "FLOAT FADDRZ";
		float_faddrz <<< 1, 1 >>> (d_a, a, b, d_start, d_end);
	} else if (OP_ == FLOATFMULRN) {
		test_name = "FLOAT FMULRN";
		float_fmulrn <<< 1, 1 >>> (d_a, a, b, d_start, d_end);
	} else if (OP_ == FLOATFMULRZ) {
		test_name = "FLOAT FMULRZ";
		float_fmulrz <<< 1, 1 >>> (d_a, a, b, d_start, d_end);
	} else if (OP_ == FLOATFDIVIDEF) {
		test_name = "FLOAT FDIVIDEF";
		float_fdividef <<< 1, 1 >>> (d_a, a, b, d_start, d_end);
	} else if (OP_ == FLOATFMA) {
		test_name = "FLOAT FMA";
		float_fma <<< 1, 1 >>> (d_a, a, b, d_start, d_end);
	} else if (OP_ == FLOATCMP) {
		test_name = "FLOAT CMP (SET)";
		float_cmp <<< 1, 1 >>> (d_a, a, b, d_start, d_end);
	}

	cudaDeviceSynchronize();

	uint bar1 = d_end[0] - d_start[0];  // 3 mov : including store the clock val
	uint bar2 = d_end[1] - d_start[1]; // 2 mov + inst + 2 mov
	uint bar3 = d_end[2] - d_start[2]; // 3 mov

	printf("\n%s\n", test_name.c_str());
	printf("bar1 : %u \t bar2 :%u \t bar3: %u\n", bar1, bar2, bar3);

	// verify there is 3 mov = 3 * 15 clk / mov
	assert(bar1 == 45);
	assert(bar3 == 45);
	uint mov_clk = bar1 / 3;

	if (OP_ == FLOATCMP) {
		printf("=>Warning: there are FSET and I2F (consider same cycles) for float_cmp.\n");
		printf("%s: %u (clks)\n", test_name.c_str() , (bar2 - 4 * mov_clk) / 2);
	} else if (OP_ == FLOATFMA) {
		printf("=>Warning: there are MOV32I before FFMA for float_fma.\n");
		printf("%s: %u (clks)\n", test_name.c_str() , bar2 - 5 * mov_clk);
	} else {
		printf("%s: %u (clks)\n", test_name.c_str() ,bar2 - 4 * mov_clk);
	}


	cudaFree(d_a);
	cudaFree(d_start);
	cudaFree(d_end);

	cudaDeviceReset();
}

void double_test(DOUBLE_OP OP_)
{
	double *d_a = NULL;
	cudaMallocManaged(&d_a, sizeof(double) * 64);  // allocate enough space

	uint *d_start = NULL;
	cudaMallocManaged(&d_start, sizeof(uint) * 3); 

	uint *d_end = NULL;
	cudaMallocManaged(&d_end, sizeof(uint) * 3); 

	double a = 3.3;
	double b = 1.1;

	for(int i=0; i<64; i++)
	{
		d_a[i] = static_cast<double>(i);
	}

	cudaDeviceSynchronize();

	std::string test_name;

	if(OP_ == DOUBLEADD) {
		test_name = "DOUBLE ADD";
		double_add <<< 1, 1 >>> (d_a, a, b, d_start, d_end);
	} else if (OP_ == DOUBLESUB) {
		test_name = "DOUBLE SUB";
		double_sub <<< 1, 1 >>> (d_a, a, b, d_start, d_end);
	} else if (OP_ == DOUBLEMAD) {
		test_name = "DOUBLE MAD";
		double_mad <<< 1, 1 >>> (d_a, a, b, d_start, d_end);
	} else if (OP_ == DOUBLEMUL) {
		test_name = "DOUBLE MUL";
		double_mul <<< 1, 1 >>> (d_a, a, b, d_start, d_end);
	} else if (OP_ == DOUBLEDIV) {
		test_name = "DOUBLE DIV";
		double_div <<< 1, 1 >>> (d_a, a, b, d_start, d_end);
	} else if (OP_ == DOUBLEMIN) {
		test_name = "DOUBLE MIN";
		double_min <<< 1, 1 >>> (d_a, a, b, d_start, d_end);
	} else if (OP_ == DOUBLEMAX) {
		test_name = "DOUBLE MAX";
		double_max <<< 1, 1 >>> (d_a, a, b, d_start, d_end);
	} else if (OP_ == DOUBLEDADDRN) {
		test_name = "DOUBLE DADDRN";
		double_daddrn <<< 1, 1 >>> (d_a, a, b, d_start, d_end);
	} else if (OP_ == DOUBLEFMA) {
		test_name = "DOUBLE FMA";
		double_fma <<< 1, 1 >>> (d_a, a, b, d_start, d_end);
	}


	cudaDeviceSynchronize();

	uint bar1= d_end[0] - d_start[0];  // 3 mov : including store the clock val
	uint bar2 = d_end[1] - d_start[1]; // 2 mov + inst + 3 mov
	uint bar3 = d_end[2] - d_start[2]; // 3 mov

	printf("\n%s\n", test_name.c_str());
	printf("bar1 : %u \t bar2 :%u \t bar3: %u\n", bar1, bar2, bar3);

	// verify there is 3 mov = 3 * 15 clk / mov
	assert(bar1 == 45);
	assert(bar3 == 45);
	uint mov_clk = bar1 / 3;

	if (OP_ == DOUBLEFMA) {
		printf("=>Warning: there are 2 more mov for double_fma.\n");
		printf("%s: %u (clks)\n", test_name.c_str() ,bar2 - 7 * mov_clk);

	} else {
		printf("%s: %u (clks)\n", test_name.c_str() ,bar2 - 5 * mov_clk);
	}



	cudaFree(d_a);
	cudaFree(d_start);
	cudaFree(d_end);

	cudaDeviceReset();
}
