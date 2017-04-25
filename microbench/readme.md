### Requirement
* cuda 8.0 (driver/runtime)
* maxwell gpu

### Doc
* sass instruction list [cuda doc](http://docs.nvidia.com/cuda/cuda-binary-utilities/index.html#instruction-set-ref)

### Some Results for GTX 950 

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

| Memory Access | SASS | Clocks |
|---------------|-------|--------|
| Global Load | LDG | 650 |
| Global Store | STG | 19 |
| Shared Load | LDS | 26 |
| Shared Store | STS | 19 |
| Constant Load | LDC | 799 |
| Local Load | LDL | 360 |

