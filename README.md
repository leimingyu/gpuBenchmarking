# gpuBenchmarking

### Some Results for GTX 950 (maxwell)

| Inst. Type | Opcode | Clocks | Inst. Type | Opcode | Clocks |
|------------|--------|--------|------------|--------|--------|
| Integer | IADD | 15 | Single | FADD | 15 |
|  | ISUB | 15 |  | FMUL | 15 |
|  | IMNMX | 15 |  | FMNMX | 15 |
|  | ISAD | 15 |  | FSET | 15 |
|  | IMUL | 86 |  | FFMA | 15 |
|  | IMAD | 101 | Double | DADD | 48 |
|  | ISET | 15 |  | DMUL | 48 |
|  | SHL | 15 |  | DMNMX | 48 |
|  | SHR | 15 |  | DFMA | 51 |



### Reference
* dissecting gpu memory hierarchy [link](http://www.comp.hkbu.edu.hk/~chxw/gpu_benchmark.html)
* cuda_hand_book [github](https://github.com/ArchaeaSoftware/cudahandbook)
* ptx caching mode [post](http://stackoverflow.com/questions/42889632/making-better-sense-of-the-ptx-store-caching-modes)
