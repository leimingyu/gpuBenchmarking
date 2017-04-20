/*
   double data type


*/

typedef enum {
	DOUBLEADD,
	DOUBLESUB,
	DOUBLEMAD,
	DOUBLEMUL,
	DOUBLEDIV,
	DOUBLEMIN,
	DOUBLEMAX,
	DOUBLEDADDRN,
	DOUBLEFMA
} DOUBLE_OP;


/*
   ADD
   add.f64     %fd3, %fd1, %fd2;
   */
__global__ void double_add (double *my_array, double a, double b,  uint *start_t, uint *end_t)
{
	unsigned int start_time1;
	unsigned int start_time2;
	unsigned int start_time3;

	unsigned int end_time1;
	unsigned int end_time2;
	unsigned int end_time3;

	double i = a;
	double j = b;
	double k = 0.0;

	start_time1 = clock();
	end_time1 = clock();

	__syncthreads();

	start_time2 = clock();
	asm volatile (
			"add.f64 %0, %1, %0;\n\t" : "=d"(k) : "d"(i), "d"(j)
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
   sub.f64     %fd3, %fd1, %fd2;
   */
__global__ void double_sub (double *my_array, double a, double b,  uint *start_t, uint *end_t)
{
	unsigned int start_time1;
	unsigned int start_time2;
	unsigned int start_time3;

	unsigned int end_time1;
	unsigned int end_time2;
	unsigned int end_time3;

	double i = a;
	double j = b;
	double k = 0.0;

	start_time1 = clock();
	end_time1 = clock();

	__syncthreads();

	start_time2 = clock();
	asm volatile (
			"sub.f64 %0, %1, %0;\n\t" : "=d"(k) : "d"(i), "d"(j)
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
   mul.f64     %fd3, %fd1, %fd2;
   add.f64     %fd4, %fd3, %fd1;
   */
__global__ void double_mad (double *my_array, double a, double b,  uint *start_t, uint *end_t)
{
	unsigned int start_time1;
	unsigned int start_time2;
	unsigned int start_time3;

	unsigned int end_time1;
	unsigned int end_time2;
	unsigned int end_time3;

	double i = a;
	double j = b;
	double k = 1.0;

	asm volatile (
			repeat128(
				"mul.f64 %0, %1, %0;\n\t"
				"add.f64 %0, %2, %0;\n\t"
				"mul.f64 %0, %1, %0;\n\t"
				"add.f64 %0, %2, %0;\n\t")
			: "=d"(k) : "d"(i) , "d"(j)
			);

	__syncthreads();

	start_time1 = clock();
	asm volatile (
			repeat32(
				"mul.f64 %0, %1, %0;\n\t"
				"add.f64 %0, %2, %0;\n\t"
				"mul.f64 %0, %1, %0;\n\t"
				"add.f64 %0, %2, %0;\n\t")
			: "=d"(k) : "d"(i) , "d"(j)
			);
	end_time1 = clock();

	__syncthreads();

	start_time2 = clock();
	asm volatile (
			repeat64(
				"mul.f64 %0, %1, %0;\n\t"
				"add.f64 %0, %2, %0;\n\t"
				"mul.f64 %0, %1, %0;\n\t"
				"add.f64 %0, %2, %0;\n\t")
			: "=d"(k) : "d"(i) , "d"(j)
			);
	end_time2 = clock();

	__syncthreads();

	start_time3 = clock();
	asm volatile (
			repeat128(
				"mul.f64 %0, %1, %0;\n\t"
				"add.f64 %0, %2, %0;\n\t"
				"mul.f64 %0, %1, %0;\n\t"
				"add.f64 %0, %2, %0;\n\t")
			: "=d"(k) : "d"(i) , "d"(j)
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
   MUL
   mul.f64     %fd3, %fd1, %fd2;
   */
__global__ void double_mul (double *my_array, double a, double b,  uint *start_t, uint *end_t)
{
	unsigned int start_time1;
	unsigned int start_time2;
	unsigned int start_time3;

	unsigned int end_time1;
	unsigned int end_time2;
	unsigned int end_time3;

	double i = a;
	double j = b;
	double k = 0.0;

	start_time1 = clock();
	end_time1 = clock();

	__syncthreads();

	start_time2 = clock();
	asm volatile (
			"mul.f64 %0, %1, %0;\n\t" : "=d"(k) : "d"(i), "d"(j)
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
   div.rn.f64     %fd3, %fd1, %fd2;
   */
__global__ void double_div (double *my_array, double a, double b,  uint *start_t, uint *end_t)
{
	unsigned int start_time1;
	unsigned int start_time2;
	unsigned int start_time3;

	unsigned int end_time1;
	unsigned int end_time2;
	unsigned int end_time3;

	double i = a;
	double j = b;
	double k = 1.0;

	asm volatile (
			repeat128(
				"div.rn.f64 %0, %1, %0;\n\t"
				"div.rn.f64 %0, %2, %0;\n\t")
			: "=d"(k) : "d"(i) , "d"(j)
			);

	__syncthreads();

	start_time1 = clock();
	asm volatile (
			repeat32(
				"div.rn.f64 %0, %1, %0;\n\t"
				"div.rn.f64 %0, %2, %0;\n\t")
			: "=d"(k) : "d"(i) , "d"(j)
			);
	end_time1 = clock();

	__syncthreads();

	start_time2 = clock();
	asm volatile (
			repeat64(
				"div.rn.f64 %0, %1, %0;\n\t"
				"div.rn.f64 %0, %2, %0;\n\t")
			: "=d"(k) : "d"(i) , "d"(j)
			);
	end_time2 = clock();

	__syncthreads();

	start_time3 = clock();
	asm volatile (
			repeat128(
				"div.rn.f64 %0, %1, %0;\n\t"
				"div.rn.f64 %0, %2, %0;\n\t")
			: "=d"(k) : "d"(i) , "d"(j)
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
   MIN
   min.f64     %fd3, %fd1, %fd2;
   */
__global__ void double_min (double *my_array, double a, double b,  uint *start_t, uint *end_t)
{
	unsigned int start_time1;
	unsigned int start_time2;
	unsigned int start_time3;

	unsigned int end_time1;
	unsigned int end_time2;
	unsigned int end_time3;

	double i = a;
	double j = b;
	double k = 0.0;

	start_time1 = clock();
	end_time1 = clock();

	__syncthreads();

	start_time2 = clock();
	asm volatile (
			"min.f64 %0, %1, %0;\n\t" : "=d"(k) : "d"(i), "d"(j)
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
   MAX
   max.f64     %fd3, %fd1, %fd2;
   */
__global__ void double_max (double *my_array, double a, double b,  uint *start_t, uint *end_t)
{
	unsigned int start_time1;
	unsigned int start_time2;
	unsigned int start_time3;

	unsigned int end_time1;
	unsigned int end_time2;
	unsigned int end_time3;

	double i = a;
	double j = b;
	double k = 0.0;

	start_time1 = clock();
	end_time1 = clock();

	__syncthreads();

	start_time2 = clock();
	asm volatile (
			"max.f64 %0, %1, %0;\n\t" : "=d"(k) : "d"(i), "d"(j)
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
   DADD_RN
   add.rn.f64  %fd3, %fd1, %fd2;
   */
__global__ void double_daddrn (double *my_array, double a, double b,  uint *start_t, uint *end_t)
{
	unsigned int start_time1;
	unsigned int start_time2;
	unsigned int start_time3;

	unsigned int end_time1;
	unsigned int end_time2;
	unsigned int end_time3;

	double i = a;
	double j = b;
	double k = 1.0;

	asm volatile (
			repeat128(
				"add.rn.f64 %0, %1, %0;\n\t"
				"add.rn.f64 %0, %2, %0;\n\t")
			: "=d"(k) : "d"(i) , "d"(j)
			);

	__syncthreads();

	start_time1 = clock();
	asm volatile (
			repeat32(
				"add.rn.f64 %0, %1, %0;\n\t"
				"add.rn.f64 %0, %2, %0;\n\t")
			: "=d"(k) : "d"(i) , "d"(j)
			);
	end_time1 = clock();

	__syncthreads();

	start_time2 = clock();
	asm volatile (
			repeat64(
				"add.rn.f64 %0, %1, %0;\n\t"
				"add.rn.f64 %0, %2, %0;\n\t")
			: "=d"(k) : "d"(i) , "d"(j)
			);
	end_time2 = clock();

	__syncthreads();

	start_time3 = clock();
	asm volatile (
			repeat128(
				"add.rn.f64 %0, %1, %0;\n\t"
				"add.rn.f64 %0, %2, %0;\n\t")
			: "=d"(k) : "d"(i) , "d"(j)
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
   fma.rn.f64     %f6, %f1, %f2;
   */
__global__ void double_fma (double *my_array, double a, double b,  uint *start_t, uint *end_t)
{
	unsigned int start_time1;
	unsigned int start_time2;
	unsigned int start_time3;

	unsigned int end_time1;
	unsigned int end_time2;
	unsigned int end_time3;

	double i = a;
	double j = b;
	double m = 1.0;
	double k;

	__syncthreads();

	start_time1 = clock();
	end_time1 = clock();

	__syncthreads();

	start_time2 = clock();
	asm volatile (
			"fma.rn.f64 %0, %1, %2, %3;\n\t"
			: "=d"(k) : "d"(i) , "d"(j), "d"(m)
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
