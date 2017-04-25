//----------------------------------------------------------------------------//
// http://docs.nvidia.com/cuda/inline-ptx-assembly/index.html#axzz4evAhdxJg
// https://devtalk.nvidia.com/default/topic/527205/problem-about-inline-ptx-code-in-cuda-program/
//----------------------------------------------------------------------------//

__constant__ float constspace[128];

_global__ void kernel_global_load(float *my_array, uint *start_t, uint *end_t, 
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


__global__ void kernel_global_load_v1(float *my_array, uint *start_t, uint *end_t, 
		int arraylen, float a, float b)
{
	unsigned int start_time1;                                                   
	unsigned int start_time2;                                                   
	unsigned int start_time3;                                                   

	unsigned int end_time1;                                                     
	unsigned int end_time2;                                                     
	unsigned int end_time3;                                                     

	float k;

	__syncthreads();                                                            

	start_time1 = clock();                                                      
	end_time1 = clock();                                                        

	__syncthreads();                                                            

	start_time2 = clock();                                                      

	for(int i=0; i<1; i++)
		k += my_array[threadIdx.x + i];

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

	my_array[0] = k;
}



//-----------------------------------------------------------------------------
// global_store
// https://devtalk.nvidia.com/default/topic/545309/ptx-code-quot-st-quot-/
//-----------------------------------------------------------------------------
__global__ void kernel_global_store(float *my_array, uint *start_t, uint *end_t, 
		int arraylen, float a, float b)
{
	unsigned int start_time1;                                                   
	unsigned int start_time2;                                                   
	unsigned int start_time3;                                                   

	unsigned int end_time1;                                                     
	unsigned int end_time2;                                                     
	unsigned int end_time3;                                                     

	float k =  a + b;

	__syncthreads();                                                            

	start_time1 = clock();                                                      
	end_time1 = clock();                                                        

	__syncthreads();                                                            

	start_time2 = clock();                                                      
	asm volatile (
			"st.global.f32  [%0], %1;\n\t" :: "l"(&my_array[0]) , "f"(k)
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

	my_array[0] += (float)k;
}


__global__ void kernel_global_store_v1(float *my_array, uint *start_t, uint *end_t, 
		int arraylen, float a, float b)
{
	unsigned int start_time1;                                                   
	unsigned int start_time2;                                                   
	unsigned int start_time3;                                                   

	unsigned int end_time1;                                                     
	unsigned int end_time2;                                                     
	unsigned int end_time3;                                                     

	float k =  a + b;

	__syncthreads();                                                            

	start_time1 = clock();                                                      
	end_time1 = clock();                                                        

	__syncthreads();                                                            

	start_time2 = clock();                                                      

	for(int i=0; i<1; i++)
		my_array[threadIdx.x] = k;

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

	my_array[0] += k;
}






//-----------------------------------------------------------------------------
//	ld.shared.f32   %f8, [%rd6];
//-----------------------------------------------------------------------------
__global__ void kernel_shared_load(float *my_array, uint *start_t, uint *end_t, 
		int arraylen, float a, float b)
{
	unsigned int start_time1;
	unsigned int start_time2;
	unsigned int start_time3;

	unsigned int end_time1;
	unsigned int end_time2;
	unsigned int end_time3;

	__shared__ float sm[32];

	float k;

	// load global to shared memory
	sm[0] = my_array[0];

	__syncthreads();

	//float *ptr_sm = &sm[0]; // ptr_global

	start_time1 = clock();                                                      
	end_time1 = clock();                                                        

	__syncthreads();                                                            

	start_time2 = clock();                                                      
	//asm volatile (                                                              
	//		"ld.f32 %0, [%1];\n\t" : "=f"(k) : "l"(ptr_sm)
	//		);                                                                  

	//asm volatile(
	//"ld.f32 %0, [%1];\n\t" : "=f"(k) : "l"(&sm[0])
	//);                                                                  

	//k = sm[0];
	k = sm[threadIdx.x];

	end_time2 = clock();                                                        

	__syncthreads();                                                            

	start_time3 = clock();                                                      
	end_time3 = clock();                                                        

	__syncthreads();                                                            


	start_t[0] = start_time1;                                                   
	start_t[1] = start_time2;                                                   
	start_t[2] = start_time3;                                                   

	end_t[0] = end_time1;                                                       
	end_t[1] = end_time2;                                                       
	end_t[2] = end_time3;                                                       

	my_array[0] += k;
}






//-----------------------------------------------------------------------------
// st.shared.f32 : doesn't work, bus error
//-----------------------------------------------------------------------------
__global__ void kernel_shared_store(float *my_array, uint *start_t, uint *end_t, 
		int arraylen, float a, float b)
{
	unsigned int start_time1;                                                   
	unsigned int start_time2;                                                   
	unsigned int start_time3;                                                   

	unsigned int end_time1;                                                     
	unsigned int end_time2;                                                     
	unsigned int end_time3;                                                     

	float k =  a + b;

	__shared__ float sm[32];

	__syncthreads();                                                            

	start_time1 = clock();                                                      
	end_time1 = clock();                                                        

	__syncthreads();                                                            

	start_time2 = clock();                                                      

	//asm volatile (
	//		"st.shared.f32  [%0], %1;\n\t" :: "l"(&sm[0]) , "f"(k)
	//		);

	//asm volatile (
	//		"st.f32  [%0], %1;\n\t" :: "l"(&sm[0]) , "f"(k)
	//		);


	sm[threadIdx.x] = k;

	end_time2 = clock();                                                        

	__syncthreads();                                                            

	start_time3 = clock();
	end_time3 = clock();

	__syncthreads();                                                            

	start_t[0] = start_time1;                                                   
	start_t[1] = start_time2;                                                   
	start_t[2] = start_time3;                                                   

	end_t[0] = end_time1;                                                       
	end_t[1] = end_time2;                                                       
	end_t[2] = end_time3;                                                       

	my_array[0] += sm[threadIdx.x]; 
}



//-----------------------------------------------------------------------------
// Constant memory read (Load) 
// st.shared.f32 : doesn't work, bus error
//-----------------------------------------------------------------------------
__global__ void kernel_const_load(float *my_array, uint *start_t, uint *end_t, 
 		int arraylen, float a, float b)
{
 	unsigned int start_time1;                                                   
 	unsigned int end_time2;                                                     
 	unsigned int end_time3;                                                     
 
	float k;
 
	__syncthreads();                                                            

	start_time1 = clock();                                                      
	end_time1 = clock();                                                        
 
 	__syncthreads();                                                            
 
	start_time2 = clock();

	for(int i=0; i<1; i++)
		k = constspace[threadIdx.x+ i];

	end_time2 = clock();                                                        
 
 	__syncthreads();                                                            
 
	start_time3 = clock();
	end_time3 = clock();
 
 	__syncthreads();                                                            

	start_t[0] = start_time1;                                                   
	start_t[1] = start_time2;                                                   
	start_t[2] = start_time3;                                                   
 
	end_t[0] = end_time1;                                                       
	end_t[1] = end_time2;                                                       
	end_t[2] = end_time3;                                                       
 
	my_array[0] += k; 
}


//-----------------------------------------------------------------------------
// local memory read (Load) 
// 512KB local memory per thread 
// http://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#features-and-technical-specifications
//-----------------------------------------------------------------------------
__global__ void kernel_local_load(float *my_array, uint *start_t, uint *end_t, 
		int arraylen, float a, float b)
{
	unsigned int start_time1;                                                   
	unsigned int start_time2;                                                   
	unsigned int start_time3;                                                   

	unsigned int end_time1;                                                     
	unsigned int end_time2;                                                     
	unsigned int end_time3;                                                     

	float loc_reg[32]; 

	for(int i=0; i<1; i++)
		loc_reg[threadIdx.x] = my_array[i]; 

	float k = 0.0;
 
	__syncthreads();                                                            

	start_time1 = clock();                                                      
	end_time1 = clock();                                                        
 
 	__syncthreads();                                                            
 

	start_time2 = clock();                                                      

	for(int i=0; i<1; i++)
		k += loc_reg[i];

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
 
	my_array[0] += k; 
}
