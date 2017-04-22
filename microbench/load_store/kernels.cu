//----------------------------------------------------------------------------//
// http://docs.nvidia.com/cuda/inline-ptx-assembly/index.html#axzz4evAhdxJg
// https://devtalk.nvidia.com/default/topic/527205/problem-about-inline-ptx-code-in-cuda-program/
//----------------------------------------------------------------------------//

__global__ void kernel_load_global(float *my_array, uint *start_t, uint *end_t, 
		int arraylen, float a, float b)
{
	unsigned int start_time1;                                                   
	unsigned int start_time2;                                                   
	unsigned int start_time3;                                                   

	unsigned int end_time1;                                                     
	unsigned int end_time2;                                                     
	unsigned int end_time3;                                                     

	float k;
	float *ptr_global = &my_array[0]; // ptr_global

	__syncthreads();                                                            

	start_time1 = clock();                                                      
	end_time1 = clock();                                                        

	__syncthreads();                                                            

	start_time2 = clock();                                                      
	asm volatile (                                                              
			"ld.global.f32  %0, [%1];\n\t" : "=f"(k) : "l"(ptr_global)
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

	my_array[0] = (float)k;
}


__global__ void kernel_load_shared(float *my_array, uint *start_t, uint *end_t, 
		int arraylen, float a, float b)
{
	unsigned int start_time1;
	unsigned int start_time2;
	unsigned int start_time3;

	unsigned int end_time1;
	unsigned int end_time2;
	unsigned int end_time3;

	__shared__ float sm[1];

	float k;

	// load global to shared memory
	sm[0] = my_array[0];
	__syncthreads();

	float *ptr_sm = &sm[0]; // ptr_global

	start_time1 = clock();                                                      
	end_time1 = clock();                                                        

	__syncthreads();                                                            

	start_time2 = clock();                                                      
	asm volatile (                                                              
			"ld.f32 %0, [%1];\n\t" : "=f"(k) : "l"(ptr_sm)
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

	my_array[0] = (float)k;
}
