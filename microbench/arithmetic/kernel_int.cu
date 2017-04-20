/*
   integer type

   ADD
   add.s32     %r5, %r1, %r2; 
   SUB
   sub.s32     %r5, %r2, %r1;
   MAD
   mul.lo.s32  %r5, %r2, %r1; 
   add.s32     %r6, %r5, %r1;
   MUL
   mul.lo.s32  %r5, %r2, %r1; 

   DIV
   div.u32     %r5, %r1, %r2; 
   REM
   rem.u32     %r5, %r1, %r2;
   MIN
   min.u32     %r5, %r1, %r2;
   MAX
   max.u32     %r5, %r1, %r2;
   AND
   and.b32     %r5, %r1, %r2; 
   OR
   or.b32      %r5, %r1, %r2; 
   XOR
   xor.b32     %r5, %r1, %r2;
   SHL
   shl.b32     %r5, %r1, %r2;
   SHR
   shr.u32     %r5, %r1, %r2;
   UMUL24
   mul24.lo.u32    %r5, %r1, %r2;
   UMULHI
   mul.hi.u32  %r5, %r1, %r2;
   USAD
   sad.u32     %r5, %r1, %r2, %r1;
 */


typedef enum {
	INTADD,
	INTSUB,
	INTMAD,
	INTMUL,
	INTDIV,
	INTREM,
	INTMIN,
	INTMAX,
	INTMUL24,
	INTMULHI,
	INTSAD,
	INTSHL,
	INTSHR,
	INTSET
} INT_OP;


/*
   ADD                                                                             
   add.s32     %r5, %r1, %r2;
 */
__global__ void int_add (int *my_array, int a, uint b,  uint *start_t, uint *end_t)
{
	unsigned int start_time1;
	unsigned int start_time2;
	unsigned int start_time3;

	unsigned int end_time1;
	unsigned int end_time2;
	unsigned int end_time3;

	int i = a;
	int j = b;
	int k;

	__syncthreads();

	start_time1 = clock();
	end_time1 = clock();

	__syncthreads();

	start_time2 = clock();
	asm volatile (
			"add.s32 %0, %1, %2;\n\t"
			: "=r"(k) : "r"(i) , "r"(j)
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
   sub.s32     %r5, %r2, %r1;
 */
__global__ void int_sub (int *my_array, int a, int b,  uint *start_t, uint *end_t)
{
	unsigned int start_time1;
	unsigned int start_time2;
	unsigned int start_time3;

	unsigned int end_time1;
	unsigned int end_time2;
	unsigned int end_time3;

	int i = a;
	int j = b;
	int k;

	__syncthreads();

	start_time1 = clock();
	end_time1 = clock();

	__syncthreads();

	start_time2 = clock();
	asm volatile (
			"sub.s32 %0, %1, %2;\n\t"
			: "=r"(k) : "r"(i) , "r"(j)
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
   mul.lo.s32  %r5, %r2, %r1;                                                    
   add.s32     %r6, %r5, %r1;
 */
__global__ void int_mad (int *my_array, int a, int b,  uint *start_t, uint *end_t)
{
	unsigned int start_time1;
	unsigned int start_time2;
	unsigned int start_time3;

	unsigned int end_time1;
	unsigned int end_time2;
	unsigned int end_time3;

	int i = a;
	int j = b;
	int k = a + b;

	__syncthreads();

	start_time1 = clock();
	end_time1 = clock();

	__syncthreads();

	start_time2 = clock();
	asm volatile (
			"mul.lo.s32 %2, %1, %0;\n\t"
			"add.s32 %1, %2, %0;\n\t"
			: "=r"(k) : "r"(i) , "r"(j)
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
   mul.lo.s32  %r5, %r2, %r1;
 */
__global__ void int_mul (int *my_array, int a, int b,  uint *start_t, uint *end_t)
{
	unsigned int start_time1;
	unsigned int start_time2;
	unsigned int start_time3;

	unsigned int end_time1;
	unsigned int end_time2;
	unsigned int end_time3;

	int i = a;
	int j = b;
	int k;

	__syncthreads();

	start_time1 = clock();
	end_time1 = clock();

	__syncthreads();

	start_time2 = clock();
	asm volatile (
			"mul.lo.s32 %0, %1, %2;\n\t"
			: "=r"(k) : "r"(i) , "r"(j)
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
   div.s32     %r5, %r1, %r2; 
 */
__global__ void int_div (int *my_array, int a, int b,  uint *start_t, uint *end_t)
{
	unsigned int start_time1;
	unsigned int start_time2;
	unsigned int start_time3;

	unsigned int end_time1;
	unsigned int end_time2;
	unsigned int end_time3;

	int i = a;
	int j = b;
	int k;

	__syncthreads();

	start_time1 = clock();
	end_time1 = clock();

	__syncthreads();

	start_time2 = clock();
	asm volatile (
			"div.s32 %0, %1, %2;\n\t"
			: "=r"(k) : "r"(i) , "r"(j)
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
   REM                                                                             
   rem.s32     %r5, %r1, %r2; 
 */
__global__ void int_rem (int *my_array, int a, int b,  uint *start_t, uint *end_t)
{
	unsigned int start_time1;
	unsigned int start_time2;
	unsigned int start_time3;

	unsigned int end_time1;
	unsigned int end_time2;
	unsigned int end_time3;

	int i = a;
	int j = b;

	__syncthreads();

	start_time1 = clock();
	end_time1 = clock();

	__syncthreads();

	start_time2 = clock();
	asm volatile (
			"rem.s32 %0, %1, %0;\n\t"
			: "=r"(i) : "r"(i) 
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
   MIN                                                                             
   min.s32     %r5, %r1, %r2;
 */
__global__ void int_min (int *my_array, int a, int b,  uint *start_t, uint *end_t)
{
	unsigned int start_time1;
	unsigned int start_time2;
	unsigned int start_time3;

	unsigned int end_time1;
	unsigned int end_time2;
	unsigned int end_time3;

	int i = a;
	int j = b;


	__syncthreads();

	start_time1 = clock();
	end_time1 = clock();

	__syncthreads();

	start_time2 = clock();
	asm volatile (
			"min.s32 %0, %1, %0;\n\t"
			: "=r"(i) : "r"(j)
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
   max.s32     %r5, %r1, %r2; 
 */
__global__ void int_max (int *my_array, int a, int b,  uint *start_t, uint *end_t)
{
	unsigned int start_time1;
	unsigned int start_time2;
	unsigned int start_time3;

	unsigned int end_time1;
	unsigned int end_time2;
	unsigned int end_time3;

	int i = a;
	int j = b;

	__syncthreads();

	start_time1 = clock();
	end_time1 = clock();

	__syncthreads();

	start_time2 = clock();
	asm volatile (
			"max.s32 %0, %1, %0;\n\t"
			: "=r"(i) : "r"(j)
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
   MUL24
   mul24.lo.s32    %r5, %r1, %r2;
 */
__global__ void int_mul24 (int *my_array, int a, int b,  uint *start_t, uint *end_t)
{
	unsigned int start_time1;
	unsigned int start_time2;
	unsigned int start_time3;

	unsigned int end_time1;
	unsigned int end_time2;
	unsigned int end_time3;

	int i = a;
	int j = b;

	__syncthreads();

	start_time1 = clock();
	end_time1 = clock();

	__syncthreads();

	start_time2 = clock();
	asm volatile (
			"mul24.lo.s32 %0, %1, %0;\n\t"
			: "=r"(i) : "r"(j)
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
   MULHI                                                                          
   mul.hi.s32  %r5, %r1, %r2;    
 */
__global__ void int_mulhi (int *my_array, int a, int b,  uint *start_t, uint *end_t)
{
	unsigned int start_time1;
	unsigned int start_time2;
	unsigned int start_time3;

	unsigned int end_time1;
	unsigned int end_time2;
	unsigned int end_time3;

	int i = a;
	int j = b;

	__syncthreads();

	start_time1 = clock();
	end_time1 = clock();

	__syncthreads();

	start_time2 = clock();
	asm volatile (
			"mul.hi.s32 %0, %1, %0;\n\t"
			: "=r"(i) : "r"(j)
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
   SAD                                                                            
   sad.s32     %r5, %r1, %r2, %r1;   
 */
__global__ void int_sad (int *my_array, int a, int b,  uint *start_t, uint *end_t)
{
	unsigned int start_time1;
	unsigned int start_time2;
	unsigned int start_time3;

	unsigned int end_time1;
	unsigned int end_time2;
	unsigned int end_time3;

	int i = a;
	int j = b;
	int k = a + b;

	__syncthreads();

	start_time1 = clock();
	end_time1 = clock();

	__syncthreads();

	start_time2 = clock();
	asm volatile (
			"sad.s32 %0, %1, %2, %0;\n\t"
			: "=r"(k) : "r"(i) , "r"(j)
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

	my_array[0] = i + j + k; 
}

/*
	SHL 
	shl.b32     %r0, %r1, %r0
 */
__global__ void int_shl(int *my_array, int a, int b,  uint *start_t, uint *end_t)
{
	unsigned int start_time1;
	unsigned int start_time2;
	unsigned int start_time3;

	unsigned int end_time1;
	unsigned int end_time2;
	unsigned int end_time3;

	int i = a;
	int j = b;

	__syncthreads();

	start_time1 = clock();
	end_time1 = clock();

	__syncthreads();

	start_time2 = clock();
	asm volatile (
			"shl.b32 %0, %1, %0;\n\t"
			: "=r"(j) : "r"(i) , "r"(j)
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
	SHR 
	shr.b32     %r0, %r1, %r0
 */
__global__ void int_shr(int *my_array, int a, int b,  uint *start_t, uint *end_t)
{
	unsigned int start_time1;
	unsigned int start_time2;
	unsigned int start_time3;

	unsigned int end_time1;
	unsigned int end_time2;
	unsigned int end_time3;

	int i = a;
	int j = b;

	__syncthreads();

	start_time1 = clock();
	end_time1 = clock();

	__syncthreads();

	start_time2 = clock();
	asm volatile (
			"shr.b32 %0, %1, %0;\n\t"
			: "=r"(j) : "r"(i) , "r"(j)
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
	SET
	set.eq.u32.u32     %r0, %r1, %r2

http://docs.nvidia.com/cuda/parallel-thread-execution/index.html#comparison-and-selection-instructions
 */
__global__ void int_set(int *my_array, int a, int b,  uint *start_t, uint *end_t)
{
	unsigned int start_time1;
	unsigned int start_time2;
	unsigned int start_time3;

	unsigned int end_time1;
	unsigned int end_time2;
	unsigned int end_time3;

	int i = a;
	int j = b;
	int k;

	__syncthreads();

	start_time1 = clock();
	end_time1 = clock();

	__syncthreads();

	start_time2 = clock();
	asm volatile (
			"set.eq.u32.u32 %0, %1, %2;\n\t"
			: "=r"(k) : "r"(i) , "r"(j)
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

	my_array[0] = i + j + k; 
}
