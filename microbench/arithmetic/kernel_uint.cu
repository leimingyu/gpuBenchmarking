/*
   unsigned int
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
	UINTADD,
	UINTSUB,
	UINTMAD,
	UINTMUL,
	UINTDIV,
	UINTREM,
	UINTMIN,
	UINTMAX,
	UINTAND,
	UINTOR,
	UINTXOR,
	UINTSHL,
	UINTSHR,
	UINTUMUL24,
	UINTUMULHI,
	UINTUSAD,
  UINTCMP
} UINT_OP;


/*
   ADD                                                                             
   add.s32     %r5, %r1, %r2;
 */
__global__ void uint_add (uint *my_array, uint a, uint b,  uint *start_t, uint *end_t)
{
	unsigned int start_time1;
	unsigned int start_time2;
	unsigned int start_time3;

	unsigned int end_time1;
	unsigned int end_time2;
	unsigned int end_time3;

	unsigned int i = a;
	unsigned int j = b;

	__syncthreads();

	start_time1 = clock();
	end_time1 = clock();

	__syncthreads();

	start_time2 = clock();
	asm volatile (
			"add.s32 %0, %1, %0;\n\t"
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
   SUB                                                                             
   sub.s32     %r5, %r2, %r1;
 */
__global__ void uint_sub (uint *my_array, uint a, uint b,  uint *start_t, uint *end_t)
{
	unsigned int start_time1;
	unsigned int start_time2;
	unsigned int start_time3;

	unsigned int end_time1;
	unsigned int end_time2;
	unsigned int end_time3;

	unsigned int i = a;
	unsigned int j = b;

	__syncthreads();

	start_time1 = clock();
	end_time1 = clock();

	__syncthreads();

	start_time2 = clock();
	asm volatile (
			"sub.s32 %0, %1, %0;\n\t"
			: "=r"(j) : "r"(i)
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
   MAD                                                                             
   mul.lo.s32  %r5, %r2, %r1;                                                    
   add.s32     %r6, %r5, %r1;
 */
__global__ void uint_mad (uint *my_array, uint a, uint b,  uint *start_t, uint *end_t)
{
	unsigned int start_time1;
	unsigned int start_time2;
	unsigned int start_time3;

	unsigned int end_time1;
	unsigned int end_time2;
	unsigned int end_time3;

	unsigned int i = a;
	unsigned int j = b;
	unsigned int k = a + b;

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

	my_array[0] = i + j + k; 
}

/*
   MUL                                                                             
   mul.lo.s32  %r5, %r2, %r1;
 */
__global__ void uint_mul (uint *my_array, uint a, uint b,  uint *start_t, uint *end_t)
{
	unsigned int start_time1;
	unsigned int start_time2;
	unsigned int start_time3;

	unsigned int end_time1;
	unsigned int end_time2;
	unsigned int end_time3;

	unsigned int i = a;
	unsigned int j = b;

	__syncthreads();

	start_time1 = clock();
	end_time1 = clock();

	__syncthreads();

	start_time2 = clock();
	asm volatile (
			"mul.lo.s32 %0, %1, %0;\n\t"
			: "=r"(j) : "r"(i)
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
   DIV                                                                             
   div.u32     %r5, %r1, %r2;
 */
__global__ void uint_div (uint *my_array, uint a, uint b,  uint *start_t, uint *end_t)
{
	unsigned int start_time1;
	unsigned int start_time2;
	unsigned int start_time3;

	unsigned int end_time1;
	unsigned int end_time2;
	unsigned int end_time3;

	unsigned int i = a;
	unsigned int j = b;

	__syncthreads();

	start_time1 = clock();
	end_time1 = clock();

	__syncthreads();

	start_time2 = clock();
	asm volatile (
			"div.u32 %0, %1, %0;\n\t"
			: "=r"(j) : "r"(i)
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
   REM                                                                             
   rem.u32     %r5, %r1, %r2; 
 */
__global__ void uint_rem (uint *my_array, uint a, uint b,  uint *start_t, uint *end_t)
{
	unsigned int start_time1;
	unsigned int start_time2;
	unsigned int start_time3;

	unsigned int end_time1;
	unsigned int end_time2;
	unsigned int end_time3;

	unsigned int i = a;
	unsigned int j = b;

	__syncthreads();

	start_time1 = clock();
	end_time1 = clock();

	__syncthreads();

	start_time2 = clock();
	asm volatile (
			"rem.u32 %0, %1, %0;\n\t"
			: "=r"(j) : "r"(i) 
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
   min.u32     %r5, %r1, %r2;
 */
__global__ void uint_min (uint *my_array, uint a, uint b,  uint *start_t, uint *end_t)
{
	unsigned int start_time1;
	unsigned int start_time2;
	unsigned int start_time3;

	unsigned int end_time1;
	unsigned int end_time2;
	unsigned int end_time3;

	unsigned int i = a;
	unsigned int j = b;

	__syncthreads();

	start_time1 = clock();
	end_time1 = clock();

	__syncthreads();

	start_time2 = clock();
	asm volatile (
			"min.u32 %0, %1, %0;\n\t"
			: "=r"(j) : "r"(i)
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
   max.u32     %r5, %r1, %r2; 
 */
__global__ void uint_max (uint *my_array, uint a, uint b,  uint *start_t, uint *end_t)
{
	unsigned int start_time1;
	unsigned int start_time2;
	unsigned int start_time3;

	unsigned int end_time1;
	unsigned int end_time2;
	unsigned int end_time3;

	unsigned int i = a;
	unsigned int j = b;

	__syncthreads();

	start_time1 = clock();
	end_time1 = clock();

	__syncthreads();

	start_time2 = clock();
	asm volatile (
			"max.u32 %0, %1, %0;\n\t"
			: "=r"(j) : "r"(i) 
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
   AND                                                                             
   and.b32     %r5, %r1, %r2;
 */
__global__ void uint_and (uint *my_array, uint a, uint b,  uint *start_t, uint *end_t)
{
	unsigned int start_time1;
	unsigned int start_time2;
	unsigned int start_time3;

	unsigned int end_time1;
	unsigned int end_time2;
	unsigned int end_time3;

	unsigned int i = a;
	unsigned int j = b;

	__syncthreads();

	start_time1 = clock();
	end_time1 = clock();

	__syncthreads();

	start_time2 = clock();
	asm volatile (
			"and.b32 %0, %1, %0;\n\t"
			: "=r"(j) : "r"(i)
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
   OR                                                                              
   or.b32      %r5, %r1, %r2;
 */
__global__ void uint_or (uint *my_array, uint a, uint b,  uint *start_t, uint *end_t)
{
	unsigned int start_time1;
	unsigned int start_time2;
	unsigned int start_time3;

	unsigned int end_time1;
	unsigned int end_time2;
	unsigned int end_time3;

	unsigned int i = a;
	unsigned int j = b;

	__syncthreads();

	start_time1 = clock();
	end_time1 = clock();

	__syncthreads();

	start_time2 = clock();
	asm volatile (
			"or.b32 %0, %1, %0;\n\t"
			: "=r"(j) : "r"(i)
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
   XOR                                                                             
   xor.b32     %r5, %r1, %r2;  
 */
__global__ void uint_xor (uint *my_array, uint a, uint b,  uint *start_t, uint *end_t)
{
	unsigned int start_time1;
	unsigned int start_time2;
	unsigned int start_time3;

	unsigned int end_time1;
	unsigned int end_time2;
	unsigned int end_time3;

	unsigned int i = a;
	unsigned int j = b;
	unsigned int k;

	asm volatile (
			repeat128(
				"xor.b32 %0, %1, %0;\n\t"
				"xor.b32 %0, %2, %0;\n\t"
				)
			: "=r"(k) : "r"(i) , "r"(j)
			);

	__syncthreads();

	start_time1 = clock();
	asm volatile (
			repeat32(
				"xor.b32 %0, %1, %0;\n\t"
				"xor.b32 %0, %2, %0;\n\t"
				)
			: "=r"(k) : "r"(i) , "r"(j)
			);
	end_time1 = clock();

	__syncthreads();

	start_time2 = clock();
	asm volatile (
			repeat64(
				"xor.b32 %0, %1, %0;\n\t"
				"xor.b32 %0, %2, %0;\n\t"
				)
			: "=r"(k) : "r"(i) , "r"(j)
			);
	end_time2 = clock();

	__syncthreads();

	start_time3 = clock();
	asm volatile (
			repeat128(
				"xor.b32 %0, %1, %0;\n\t"
				"xor.b32 %0, %2, %0;\n\t"
				)
			: "=r"(k) : "r"(i) , "r"(j)
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
   SHL                                                                             
   shl.b32     %r5, %r1, %r2;
 */
__global__ void uint_shl (uint *my_array, uint a, uint b,  uint *start_t, uint *end_t)
{
	unsigned int start_time1;
	unsigned int start_time2;
	unsigned int start_time3;

	unsigned int end_time1;
	unsigned int end_time2;
	unsigned int end_time3;

	unsigned int i = a;
	unsigned int j = b;
	unsigned int k;

	asm volatile (
			repeat128(
				"shl.b32 %0, %1, %0;\n\t"
				"shl.b32 %0, %2, %0;\n\t"
				)
			: "=r"(k) : "r"(i) , "r"(j)
			);

	__syncthreads();

	start_time1 = clock();
	asm volatile (
			repeat32(
				"shl.b32 %0, %1, %0;\n\t"
				"shl.b32 %0, %2, %0;\n\t"
				)
			: "=r"(k) : "r"(i) , "r"(j)
			);
	end_time1 = clock();

	__syncthreads();

	start_time2 = clock();
	asm volatile (
			repeat64(
				"shl.b32 %0, %1, %0;\n\t"
				"shl.b32 %0, %2, %0;\n\t"
				)
			: "=r"(k) : "r"(i) , "r"(j)
			);
	end_time2 = clock();

	__syncthreads();

	start_time3 = clock();
	asm volatile (
			repeat128(
				"shl.b32 %0, %1, %0;\n\t"
				"shl.b32 %0, %2, %0;\n\t"
				)
			: "=r"(k) : "r"(i) , "r"(j)
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
   SHR                                                                             
   shr.u32     %r5, %r1, %r2;  
 */
__global__ void uint_shr (uint *my_array, uint a, uint b,  uint *start_t, uint *end_t)
{
	unsigned int start_time1;
	unsigned int start_time2;
	unsigned int start_time3;

	unsigned int end_time1;
	unsigned int end_time2;
	unsigned int end_time3;

	unsigned int i = a;
	unsigned int j = b;
	unsigned int k;

	asm volatile (
			repeat128(
				"shr.u32 %0, %1, %0;\n\t"
				"shr.u32 %0, %2, %0;\n\t"
				)
			: "=r"(k) : "r"(i) , "r"(j)
			);

	__syncthreads();

	start_time1 = clock();
	asm volatile (
			repeat32(
				"shr.u32 %0, %1, %0;\n\t"
				"shr.u32 %0, %2, %0;\n\t"
				)
			: "=r"(k) : "r"(i) , "r"(j)
			);
	end_time1 = clock();

	__syncthreads();

	start_time2 = clock();
	asm volatile (
			repeat64(
				"shr.u32 %0, %1, %0;\n\t"
				"shr.u32 %0, %2, %0;\n\t"
				)
			: "=r"(k) : "r"(i) , "r"(j)
			);
	end_time2 = clock();

	__syncthreads();

	start_time3 = clock();
	asm volatile (
			repeat128(
				"shr.u32 %0, %1, %0;\n\t"
				"shr.u32 %0, %2, %0;\n\t"
				)
			: "=r"(k) : "r"(i) , "r"(j)
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
   UMUL24                                                                          
   mul24.lo.u32    %r5, %r1, %r2; 
 */
__global__ void uint_umul24 (uint *my_array, uint a, uint b,  uint *start_t, uint *end_t)
{
	unsigned int start_time1;
	unsigned int start_time2;
	unsigned int start_time3;

	unsigned int end_time1;
	unsigned int end_time2;
	unsigned int end_time3;

	unsigned int i = a;
	unsigned int j = b;

	__syncthreads();

	start_time1 = clock();
	end_time1 = clock();

	__syncthreads();

	start_time2 = clock();
	asm volatile (
			"mul24.lo.u32 %0, %1, %0;\n\t"
			: "=r"(j) : "r"(i)
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
   UMULHI                                                                          
   mul.hi.u32  %r5, %r1, %r2;    
 */
__global__ void uint_umulhi (uint *my_array, uint a, uint b,  uint *start_t, uint *end_t)
{
	unsigned int start_time1;
	unsigned int start_time2;
	unsigned int start_time3;

	unsigned int end_time1;
	unsigned int end_time2;
	unsigned int end_time3;

	unsigned int i = a;
	unsigned int j = b;

	__syncthreads();

	start_time1 = clock();
	end_time1 = clock();

	__syncthreads();

	start_time2 = clock();
	asm volatile (
			"mul.hi.u32 %0, %1, %0;\n\t"
			: "=r"(j) : "r"(i)
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
   USAD                                                                            
   sad.u32     %r5, %r1, %r2, %r1;   
 */
__global__ void uint_usad (uint *my_array, uint a, uint b,  uint *start_t, uint *end_t)
{
	unsigned int start_time1;
	unsigned int start_time2;
	unsigned int start_time3;

	unsigned int end_time1;
	unsigned int end_time2;
	unsigned int end_time3;

	unsigned int i = a;
	unsigned int j = b;
	unsigned int k = a + b;

	__syncthreads();

	start_time1 = clock();
	end_time1 = clock();

	__syncthreads();

	start_time2 = clock();
	asm volatile (
			"sad.u32 %0, %1, %2, %0;\n\t"
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
   CMP
*/
__global__ void uint_cmp (uint *my_array, uint a, uint b,  uint *start_t, uint *end_t)
{
	unsigned int start_time1;
	unsigned int start_time2;
	unsigned int start_time3;

	unsigned int end_time1;
	unsigned int end_time2;
	unsigned int end_time3;

	uint i = a;
	uint j = b;
	uint k;

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

	my_array[0] = i; 
	my_array[1] = j; 
	my_array[2] = k; 
}

