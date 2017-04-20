/*
   float data type

   ADD
   add.f32     %r5, %r1, %r2; 

   SUB
   sub.f32     %r5, %r2, %r1;

   MAD
   mul.f32     %f6, %f1, %f2;
   add.f32     %f7, %f6, %f2;

   MUL
   mul.f32     %f6, %f1, %f2;


   DIV
// input : f1, f2
// output: f8

// fd1: convert f1 to double
// f6: round down to float

// fd2: convert f2 to double
// f7: round down to float

// f6 / f7

cvt.f64.f32 %fd1, %f1;
cvt.rn.f32.f64  %f6, %fd1;
cvt.f64.f32 %fd2, %f2;
cvt.rn.f32.f64  %f7, %fd2;
div.rn.f32  %f8, %f6, %f7;

MIN
min.f32     %f6, %f1, %f2;

MAX
max.f32     %r5, %r1, %r2;

FADD_RN
add.rn.f32  %f6, %f1, %f2;

FADD_RZ
add.rz.f32  %f6, %f1, %f2;

FMUL_RN
mul.rn.f32  %f6, %f1, %f2;

FMUL_RZ
mul.rz.f32  %f6, %f1, %f2;

FDIVIDEF
div.approx.f32  %f6, %f1, %f2;

*/

/*
   uint
   WARP, MISC and CONVERSION  are not considered for now
   INTASFLOAT
   POPC
   CLZ
   ALL
   ANY
   SYNC
   */

typedef enum {
	FLOATADD,
	FLOATSUB,
	FLOATMAD,
	FLOATMUL,
	FLOATDIV,
	FLOATMIN,
	FLOATMAX,
	FLOATFADDRN,
	FLOATFADDRZ,
	FLOATFMULRN,
	FLOATFMULRZ,
	FLOATFDIVIDEF,
	FLOATFMA,
	FLOATCMP
} FLOAT_OP;


/*
   ADD                                                                             
   add.f32     %r5, %r1, %r2; 
   */
__global__ void float_add (float *my_array, float a, float b,  uint *start_t, uint *end_t)
{
	unsigned int start_time1;
	unsigned int start_time2;
	unsigned int start_time3;

	unsigned int end_time1;
	unsigned int end_time2;
	unsigned int end_time3;

	float i = a;
	float j = b;
	float k;


	start_time1 = clock();
	end_time1 = clock();

	__syncthreads();

	start_time2 = clock();
	asm volatile (
			"add.f32 %0, %1, %0;\n\t" : "=f"(k) : "f"(i)
			);
	end_time2 = clock();

	__syncthreads();

	start_time3 = clock();
	end_time3 = clock();

	start_t[0] = start_time1;
	start_t[1] = start_time2;
	start_t[2] = start_time3;

	end_t[0] = end_time1;
	end_t[1] = end_time2;
	end_t[2] = end_time3;

	my_array[0] = i; 
	my_array[1] = j; 
	my_array[2] = k; 
}

/*
   SUB                                                                             
   sub.f32     %r5, %r2, %r1;
   */
__global__ void float_sub (float *my_array, float a, float b, uint *start_t, uint *end_t)
{
	unsigned int start_time1;
	unsigned int start_time2;
	unsigned int start_time3;

	unsigned int end_time1;
	unsigned int end_time2;
	unsigned int end_time3;

	float i = a;
	float j = b;
	float k = a * b;

	start_time1 = clock();
	end_time1 = clock();

	__syncthreads();

	start_time2 = clock();
	asm volatile (
			"sub.f32 %0, %1, %0;\n\t" : "=f"(k) : "f"(i)
			);
	end_time2 = clock();

	__syncthreads();

	start_time3 = clock();
	end_time3 = clock();

	start_t[0] = start_time1;
	start_t[1] = start_time2;
	start_t[2] = start_time3;

	end_t[0] = end_time1;
	end_t[1] = end_time2;
	end_t[2] = end_time3;

	my_array[0] = i; 
	my_array[1] = j; 
	my_array[2] = k; 
}

/*
   MAD                                                                             
   mul.f32     %f6, %f1, %f2;
   add.f32     %f7, %f6, %f2;
   */
__global__ void float_mad (float *my_array, float a, float b,  uint *start_t, uint *end_t)
{
	unsigned int start_time1;
	unsigned int start_time2;
	unsigned int start_time3;

	unsigned int end_time1;
	unsigned int end_time2;
	unsigned int end_time3;

	float i = a;
	float j = b;
	float k = i + j;

	__syncthreads();

	start_time1 = clock();
	end_time1 = clock();

	__syncthreads();

	start_time2 = clock();
	asm volatile (
			"mul.f32 %0, %1, %0;\n\t"
			"add.f32 %0, %2, %0;\n\t"
			: "=f"(k) : "f"(i) , "f"(j)
			);
	end_time2 = clock();

	__syncthreads();

	start_time3 = clock();
	end_time3 = clock();

	start_t[0] = start_time1;
	start_t[1] = start_time2;
	start_t[2] = start_time3;

	end_t[0] = end_time1;
	end_t[1] = end_time2;
	end_t[2] = end_time3;

	my_array[0] = i; 
	my_array[1] = j; 
	my_array[2] = k; 
}

/*
   MUL                                                                             
   mul.f32     %f6, %f1, %f2;
   */
__global__ void float_mul (float *my_array, float a, float b,  uint *start_t, uint *end_t)
{
	unsigned int start_time1;
	unsigned int start_time2;
	unsigned int start_time3;

	unsigned int end_time1;
	unsigned int end_time2;
	unsigned int end_time3;

	float i = a;
	float j = b;
	float k;

	__syncthreads();

	start_time1 = clock();
	end_time1 = clock();

	__syncthreads();

	start_time2 = clock();
	asm volatile (
			"mul.f32 %0, %1, %2;\n\t"
			: "=f"(k) : "f"(i) , "f"(j)
			);
	end_time2 = clock();

	__syncthreads();

	start_time3 = clock();
	end_time3 = clock();

	start_t[0] = start_time1;
	start_t[1] = start_time2;
	start_t[2] = start_time3;

	end_t[0] = end_time1;
	end_t[1] = end_time2;
	end_t[2] = end_time3;

	my_array[0] = i; 
	my_array[1] = j; 
	my_array[2] = k; 
}

/*
   DIV                                                                             
   cvt.f64.f32 %fd1, %f1;
   cvt.rn.f32.f64  %f6, %fd1;
   cvt.f64.f32 %fd2, %f2;
   cvt.rn.f32.f64  %f7, %fd2;
   div.rn.f32  %f8, %f6, %f7;
   */
__global__ void float_div (float *my_array, float a, float b,  uint *start_t, uint *end_t)
{
	unsigned int start_time1;
	unsigned int start_time2;
	unsigned int start_time3;

	unsigned int end_time1;
	unsigned int end_time2;
	unsigned int end_time3;

	float i = a;
	float j = b;
	float k;
	double fd1 = 0;
	double fd2 = 0;

	__syncthreads();

	start_time1 = clock();
	end_time1 = clock();

	__syncthreads();

	start_time2 = clock();
	asm volatile (
			"cvt.f64.f32 %3, %1;\n\t"
			"cvt.rn.f32.f64  %0, %3;\n\t"
			"cvt.f64.f32 %4, %2;\n\t"
			"cvt.rn.f32.f64  %2, %4;\n\t"
			"div.rn.f32  %1, %0, %2;\n\t"
			: "=f"(k) : "f"(i) , "f"(j), "d"(fd1), "d"(fd2)
			);
	end_time2 = clock();

	__syncthreads();

	start_time3 = clock();
	end_time3 = clock();

	start_t[0] = start_time1;
	start_t[1] = start_time2;
	start_t[2] = start_time3;

	end_t[0] = end_time1;
	end_t[1] = end_time2;
	end_t[2] = end_time3;

	my_array[0] = i; 
	my_array[1] = j; 
	my_array[2] = k; 
}


/*
   MIN                                                                             
   min.f32     %r5, %r1, %r2;
   */
__global__ void float_min (float *my_array, float a, float b,  uint *start_t, uint *end_t)
{
	unsigned int start_time1;
	unsigned int start_time2;
	unsigned int start_time3;

	unsigned int end_time1;
	unsigned int end_time2;
	unsigned int end_time3;

	float i = a;
	float j = b;

	__syncthreads();

	start_time1 = clock();
	end_time1 = clock();

	__syncthreads();

	start_time2 = clock();
	asm volatile (
			"min.f32 %0, %1, %0;\n\t"
			: "=f"(i) : "f"(j)
			);
	end_time2 = clock();

	__syncthreads();

	start_time3 = clock();
	end_time3 = clock();

	start_t[0] = start_time1;
	start_t[1] = start_time2;
	start_t[2] = start_time3;

	end_t[0] = end_time1;
	end_t[1] = end_time2;
	end_t[2] = end_time3;

	my_array[0] = i + j; 
}

/*
   MAX                                                                             
   max.f32     %r5, %r1, %r2; 
   */
__global__ void float_max (float *my_array, float a, float b,  uint *start_t, uint *end_t)
{
	unsigned int start_time1;
	unsigned int start_time2;
	unsigned int start_time3;

	unsigned int end_time1;
	unsigned int end_time2;
	unsigned int end_time3;

	float i = a;
	float j = b;

	__syncthreads();

	start_time1 = clock();
	end_time1 = clock();

	__syncthreads();

	start_time2 = clock();
	asm volatile (
			"max.f32 %0, %1, %0;\n\t"
			: "=f"(j) : "f"(i)
			);
	end_time2 = clock();

	__syncthreads();

	start_time3 = clock();
	end_time3 = clock();

	start_t[0] = start_time1;
	start_t[1] = start_time2;
	start_t[2] = start_time3;

	end_t[0] = end_time1;
	end_t[1] = end_time2;
	end_t[2] = end_time3;

	my_array[0] = i + j; 
}

/*
   FADD_RN
   add.rn.f32  %f6, %f1, %f2;
   */
__global__ void float_faddrn (float *my_array, float a, float b,  uint *start_t, uint *end_t)
{
	unsigned int start_time1;
	unsigned int start_time2;
	unsigned int start_time3;

	unsigned int end_time1;
	unsigned int end_time2;
	unsigned int end_time3;

	float i = a;
	float j = b;
	float k = 0.f;

	asm volatile (
			repeat128(
				"add.rn.f32 %0, %1, %0;\n\t"
				"add.rn.f32 %0, %2, %0;\n\t"
				)
			: "=f"(k) : "f"(i) , "f"(j)
			);

	__syncthreads();

	start_time1 = clock();
	asm volatile (
			repeat32(
				"add.rn.f32 %0, %1, %0;\n\t"
				"add.rn.f32 %0, %2, %0;\n\t"
				)
			: "=f"(k) : "f"(i) , "f"(j)
			);
	end_time1 = clock();

	__syncthreads();

	start_time2 = clock();
	asm volatile (
			repeat64(
				"add.rn.f32 %0, %1, %0;\n\t"
				"add.rn.f32 %0, %2, %0;\n\t"
				)
			: "=f"(k) : "f"(i) , "f"(j)
			);
	end_time2 = clock();

	__syncthreads();

	start_time3 = clock();
	asm volatile (
			repeat128(
				"add.rn.f32 %0, %1, %0;\n\t"
				"add.rn.f32 %0, %2, %0;\n\t"
				)
			: "=f"(k) : "f"(i) , "f"(j)
			);
	end_time3 = clock();

	start_t[0] = start_time1;
	start_t[1] = start_time2;
	start_t[2] = start_time3;

	end_t[0] = end_time1;
	end_t[1] = end_time2;
	end_t[2] = end_time3;

	my_array[0] = i; 
	my_array[1] = j; 
	my_array[2] = k; 
}


/*
   FADD_RZ
   add.rz.f32  %f6, %f1, %f2;
   */
__global__ void float_faddrz (float *my_array, float a, float b,  uint *start_t, uint *end_t)
{
	unsigned int start_time1;
	unsigned int start_time2;
	unsigned int start_time3;

	unsigned int end_time1;
	unsigned int end_time2;
	unsigned int end_time3;

	float i = a;
	float j = b;
	float k = 0.f;

	asm volatile (
			repeat128(
				"add.rz.f32 %0, %1, %0;\n\t"
				"add.rz.f32 %0, %2, %0;\n\t"
				)
			: "=f"(k) : "f"(i) , "f"(j)
			);

	__syncthreads();

	start_time1 = clock();
	asm volatile (
			repeat32(
				"add.rz.f32 %0, %1, %0;\n\t"
				"add.rz.f32 %0, %2, %0;\n\t"
				)
			: "=f"(k) : "f"(i) , "f"(j)
			);
	end_time1 = clock();

	__syncthreads();

	start_time2 = clock();
	asm volatile (
			repeat64(
				"add.rz.f32 %0, %1, %0;\n\t"
				"add.rz.f32 %0, %2, %0;\n\t"
				)
			: "=f"(k) : "f"(i) , "f"(j)
			);
	end_time2 = clock();

	__syncthreads();

	start_time3 = clock();
	asm volatile (
			repeat128(
				"add.rz.f32 %0, %1, %0;\n\t"
				"add.rz.f32 %0, %2, %0;\n\t"
				)
			: "=f"(k) : "f"(i) , "f"(j)
			);
	end_time3 = clock();

	start_t[0] = start_time1;
	start_t[1] = start_time2;
	start_t[2] = start_time3;

	end_t[0] = end_time1;
	end_t[1] = end_time2;
	end_t[2] = end_time3;

	my_array[0] = i; 
	my_array[1] = j; 
	my_array[2] = k; 
}


/*
   FMUL_RN
   mul.rn.f32  %f6, %f1, %f2;
   */
__global__ void float_fmulrn (float *my_array, float a, float b,  uint *start_t, uint *end_t)
{
	unsigned int start_time1;
	unsigned int start_time2;
	unsigned int start_time3;

	unsigned int end_time1;
	unsigned int end_time2;
	unsigned int end_time3;

	float i = a;
	float j = b;
	float k = 1.f;

	asm volatile (
			repeat128(
				"mul.rn.f32 %0, %1, %0;\n\t"
				"mul.rn.f32 %0, %2, %0;\n\t"
				)
			: "=f"(k) : "f"(i) , "f"(j)
			);

	__syncthreads();

	start_time1 = clock();
	asm volatile (
			repeat32(
				"mul.rn.f32 %0, %1, %0;\n\t"
				"mul.rn.f32 %0, %2, %0;\n\t"
				)
			: "=f"(k) : "f"(i) , "f"(j)
			);
	end_time1 = clock();

	__syncthreads();

	start_time2 = clock();
	asm volatile (
			repeat64(
				"mul.rn.f32 %0, %1, %0;\n\t"
				"mul.rn.f32 %0, %2, %0;\n\t"
				)
			: "=f"(k) : "f"(i) , "f"(j)
			);
	end_time2 = clock();

	__syncthreads();

	start_time3 = clock();
	asm volatile (
			repeat128(
				"mul.rn.f32 %0, %1, %0;\n\t"
				"mul.rn.f32 %0, %2, %0;\n\t"
				)
			: "=f"(k) : "f"(i) , "f"(j)
			);
	end_time3 = clock();

	start_t[0] = start_time1;
	start_t[1] = start_time2;
	start_t[2] = start_time3;

	end_t[0] = end_time1;
	end_t[1] = end_time2;
	end_t[2] = end_time3;

	my_array[0] = i; 
	my_array[1] = j; 
	my_array[2] = k; 
}


/*
   FMUL_RZ
   mul.rz.f32  %f6, %f1, %f2;
   */
__global__ void float_fmulrz (float *my_array, float a, float b,  uint *start_t, uint *end_t)
{
	unsigned int start_time1;
	unsigned int start_time2;
	unsigned int start_time3;

	unsigned int end_time1;
	unsigned int end_time2;
	unsigned int end_time3;

	float i = a;
	float j = b;
	float k = 1.f;

	asm volatile (
			repeat128(
				"mul.rz.f32 %0, %1, %0;\n\t"
				"mul.rz.f32 %0, %2, %0;\n\t"
				)
			: "=f"(k) : "f"(i) , "f"(j)
			);

	__syncthreads();

	start_time1 = clock();
	asm volatile (
			repeat32(
				"mul.rz.f32 %0, %1, %0;\n\t"
				"mul.rz.f32 %0, %2, %0;\n\t"
				)
			: "=f"(k) : "f"(i) , "f"(j)
			);
	end_time1 = clock();

	__syncthreads();

	start_time2 = clock();
	asm volatile (
			repeat64(
				"mul.rz.f32 %0, %1, %0;\n\t"
				"mul.rz.f32 %0, %2, %0;\n\t"
				)
			: "=f"(k) : "f"(i) , "f"(j)
			);
	end_time2 = clock();

	__syncthreads();

	start_time3 = clock();
	asm volatile (
			repeat128(
				"mul.rz.f32 %0, %1, %0;\n\t"
				"mul.rz.f32 %0, %2, %0;\n\t"
				)
			: "=f"(k) : "f"(i) , "f"(j)
			);
	end_time3 = clock();

	start_t[0] = start_time1;
	start_t[1] = start_time2;
	start_t[2] = start_time3;

	end_t[0] = end_time1;
	end_t[1] = end_time2;
	end_t[2] = end_time3;

	my_array[0] = i; 
	my_array[1] = j; 
	my_array[2] = k; 
}


/*
   FDIVIDEF
   div.approx.f32  %f6, %f1, %f2;
   */
__global__ void float_fdividef (float *my_array, float a, float b,  uint *start_t, uint *end_t)
{
	unsigned int start_time1;
	unsigned int start_time2;
	unsigned int start_time3;

	unsigned int end_time1;
	unsigned int end_time2;
	unsigned int end_time3;

	float i = a;
	float j = b;
	float k = 1.f;

	asm volatile (
			repeat128(
				"div.approx.f32 %0, %1, %0;\n\t"
				"div.approx.f32 %0, %2, %0;\n\t"
				)
			: "=f"(k) : "f"(i) , "f"(j)
			);

	__syncthreads();

	start_time1 = clock();
	asm volatile (
			repeat32(
				"div.approx.f32 %0, %1, %0;\n\t"
				"div.approx.f32 %0, %2, %0;\n\t"
				)
			: "=f"(k) : "f"(i) , "f"(j)
			);
	end_time1 = clock();

	__syncthreads();

	start_time2 = clock();
	asm volatile (
			repeat64(
				"div.approx.f32 %0, %1, %0;\n\t"
				"div.approx.f32 %0, %2, %0;\n\t"
				)
			: "=f"(k) : "f"(i) , "f"(j)
			);
	end_time2 = clock();

	__syncthreads();

	start_time3 = clock();
	asm volatile (
			repeat128(
				"div.approx.f32 %0, %1, %0;\n\t"
				"div.approx.f32 %0, %2, %0;\n\t"
				)
			: "=f"(k) : "f"(i) , "f"(j)
			);
	end_time3 = clock();

	start_t[0] = start_time1;
	start_t[1] = start_time2;
	start_t[2] = start_time3;

	end_t[0] = end_time1;
	end_t[1] = end_time2;
	end_t[2] = end_time3;

	my_array[0] = i; 
	my_array[1] = j; 
	my_array[2] = k; 
}

/*
   FMA
   fma.rn.f32     %f6, %f1, %f2;
   */
__global__ void float_fma (float *my_array, float a, float b,  uint *start_t, uint *end_t)
{
	unsigned int start_time1;
	unsigned int start_time2;
	unsigned int start_time3;

	unsigned int end_time1;
	unsigned int end_time2;
	unsigned int end_time3;

	float i = a;
	float j = b;
	float m = 1.f;
	float k;

	__syncthreads();

	start_time1 = clock();
	end_time1 = clock();

	__syncthreads();

	start_time2 = clock();
	asm volatile (
			"fma.rn.f32 %0, %1, %2, %3;\n\t"
			: "=f"(k) : "f"(i) , "f"(j), "f"(m)
			);
	end_time2 = clock();

	__syncthreads();

	start_time3 = clock();
	end_time3 = clock();

	start_t[0] = start_time1;
	start_t[1] = start_time2;
	start_t[2] = start_time3;

	end_t[0] = end_time1;
	end_t[1] = end_time2;
	end_t[2] = end_time3;

	my_array[0] = i; 
	my_array[1] = j; 
	my_array[2] = k; 
	my_array[3] = m;
}

/*
   CMP
   */
__global__ void float_cmp (float *my_array, float a, float b,  uint *start_t, uint *end_t)
{
	unsigned int start_time1;
	unsigned int start_time2;
	unsigned int start_time3;

	unsigned int end_time1;
	unsigned int end_time2;
	unsigned int end_time3;

	float i = a;
	float j = b;
	int k; // output s32 : signed int
	float kk;

	__syncthreads();

	start_time1 = clock();
	end_time1 = clock();

	__syncthreads();

	start_time2 = clock();

	// compare a and b, store the result to k
	asm volatile (
			"set.eq.f32.f32 %0, %1, %2;\n\t"
			: "=r"(k) : "f"(i) , "f"(j)
			);

	end_time2 = clock();

	__syncthreads();

	start_time3 = clock();
	end_time3 = clock();

	start_t[0] = start_time1;
	start_t[1] = start_time2;
	start_t[2] = start_time3;

	end_t[0] = end_time1;
	end_t[1] = end_time2;
	end_t[2] = end_time3;

	if(k) {
		kk = 1.0;	
	}

	my_array[0] = i; 
	my_array[1] = j; 
	my_array[2] = kk; 
}
