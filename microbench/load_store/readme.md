# Results 
### GTX 950 (CUDA 8)
* global_load  650 (cycles)
* global_store  64 (cycles)
* shared_load  71 (cycles)
* shared_store  64 (cycles)
* constant memory load  799 (cycles)
* local memory load  360 (cycles)

noted: load directly to register

### load from global
```
CS2R R9, SR_CLOCKLO;           
MOV R9, R9;                    
MOV R9, R9;                    // 2 mov to record clock
                               
I2I.S32.S32 R10, R8;           
SHR R12, R10, 0x1f;            
MOV R13, R12;                  
                               
MOV R12, R10;                  
MOV R12, R12;                  
MOV R13, R13;                  
                               
MOV R10, R12;                  
MOV R12, R13;                  
SHF.L.U64 R12, R10, 0x2, R12;  
                               
SHF.L.U64 R10, RZ, 0x2, R10;   
MOV R13, R2;                   
MOV R14, R3;                   
                               
IADD R10.CC, R13, R10;         
IADD.X R12, R14, R12;          
MOV R13, R12;                  
                               
MOV R12, R10;                  
MOV R12, R12;                  
MOV R13, R13;                  
                               
MOV R14, R12;                  
MOV R12, R13;                  
LEA R10.CC, R14, RZ;           
                               
LEA.HI.X P0, R12, R14, RZ, R12;
MOV R13, R12;                  
MOV R12, R10;                  
                               
LD.E R10, [R12], P0;           
MOV R12, R10;                  
CS2R R10, SR_CLOCKLO;         // 1 mov to recor clock 
```

3 mov (record clocks) + 25 (mov/lea/shf) + ld.e
= 28 * 15 + ld.e = 420 + ld.e = load 1 float to register  

### Global Load (v1)
```
        /*0168*/                   CS2R R10, SR_CLOCKLO;              /* 0x50c800000507000a */
        /*0170*/                   MOV R10, R10;                      /* 0x5c98078000a7000a */
        /*0178*/                   MOV R10, R10;                      /* 0x5c98078000a7000a */
                                                                      /* 0x00643c03fde0190f */
        /*0188*/                   S2R R11, SR_TID.X;                 /* 0xf0c800000217000b */
        /*0190*/                   MOV R11, R11;                      /* 0x5c98078000b7000b */
        /*0198*/                   IMUL32I.U32.U32 R12, R11, 0x4;     /* 0x1f00000000470b0c */
                                                                      /* 0x007fbc03fde0190f */
        /*01a8*/                   IMUL32I.U32.U32.HI R11, R11, 0x4;  /* 0x1f20000000470b0b */
        /*01b0*/                   MOV R12, R12;                      /* 0x5c98078000c7000c */
        /*01b8*/                   MOV R13, R11;                      /* 0x5c98078000b7000d */
                                                                      /* 0x007fbc03fde01fef */
        /*01c8*/                   MOV R11, R2;                       /* 0x5c9807800027000b */
        /*01d0*/                   MOV R14, R3;                       /* 0x5c9807800037000e */
        /*01d8*/                   IADD R11.CC, R11, R12;             /* 0x5c10800000c70b0b */
                                                                      /* 0x007fbc03fde01fef */
        /*01e8*/                   IADD.X R12, R14, R13;              /* 0x5c10080000d70e0c */
        /*01f0*/                   MOV R13, R12;                      /* 0x5c98078000c7000d */
        /*01f8*/                   MOV R12, R11;                      /* 0x5c98078000b7000c */
                                                                      /* 0x00643c03fde01fef */
        /*0208*/                   MOV R12, R12;                      /* 0x5c98078000c7000c */
        /*0210*/                   MOV R13, R13;                      /* 0x5c98078000d7000d */
        /*0218*/                   LDG.E R11, [R12];                  /* 0xeed4200000070c0b */
                                                                      /* 0x007fbc03fde01fef */
        /*0228*/                   FADD R0, R11, R0;                  /* 0x5c58000000070b00 */
        /*0230*/                   MOV R4, R4;                        /* 0x5c98078000470004 */
        /*0238*/                   MOV R5, R5;                        /* 0x5c98078000570005 */
                                                                      /* 0x007fbc03fde01fef */
        /*0248*/                   MOV R0, R0;                        /* 0x5c98078000070000 */
        /*0250*/                   MOV R4, R4;                        /* 0x5c98078000470004 */
        /*0258*/                   MOV R5, R5;                        /* 0x5c98078000570005 */
                                                                      /* 0x007fbc03fde01fef */
        /*0268*/                   MOV R14, R4;                       /* 0x5c9807800047000e */
        /*0270*/                   MOV R15, R5;                       /* 0x5c9807800057000f */
        /*0278*/                   CS2R R4, SR_CLOCKLO;               /* 0x50c8000005070004 */

```

3 mov + 23 inst ( 2 imul +  LDG + 2 iadd + 1 fadd + 17 others)


### load from shared

For a benchmarched sass as below
```
      /*01b8*/                   CS2R R9, SR_CLOCKLO;               /* 0x50c8000005070009 */
                                                                      /* 0x00643c03fde01fef */
        /*01c8*/                   MOV R9, R9;                        /* 0x5c98078000970009 */
        /*01d0*/                   MOV R9, R9;                        /* 0x5c98078000970009 */
        /*01d8*/                   S2R R10, SR_TID.X;                 /* 0xf0c800000217000a */
                                                                      /* 0x00643c0321e01fef */
        /*01e8*/                   MOV R10, R10;                      /* 0x5c98078000a7000a */
        /*01f0*/                   IMUL32I.U32.U32 R11, R10, 0x4;     /* 0x1f00000000470a0b */
        /*01f8*/                   IMUL32I.U32.U32.HI R10, R10, 0x4;  /* 0x1f20000000470a0a */
                                                                      /* 0x00643c03fde01fef */
        /*0208*/                   MOV R13, R11;                      /* 0x5c98078000b7000d */
        /*0210*/                   MOV R14, R10;                      /* 0x5c98078000a7000e */
        /*0218*/                   I2I.U32.U32 R10, RZ;               /* 0x5ce000000ff70a0a */
                                                                      /* 0x007fbc03fde01fef */
        /*0228*/                   MOV R10, R10;                      /* 0x5c98078000a7000a */
        /*0230*/                   MOV R11, RZ;                       /* 0x5c9807800ff7000b */
        /*0238*/                   MOV R10, R10;                      /* 0x5c98078000a7000a */
                                                                      /* 0x007fbc03fde01fef */
        /*0248*/                   MOV R11, R11;                      /* 0x5c98078000b7000b */
        /*0250*/                   MOV R12, R10;                      /* 0x5c98078000a7000c */
        /*0258*/                   MOV R11, R11;                      /* 0x5c98078000b7000b */
                                                                      /* 0x007fbc03fde01fef */
        /*0268*/                   IADD R10.CC, R12, R13;             /* 0x5c10800000d70c0a */
        /*0270*/                   IADD.X R11, R11, R14;              /* 0x5c10080000e70b0b */
        /*0278*/                   MOV R10, R10;                      /* 0x5c98078000a7000a */
                                                                      /* 0x007fbc03fde01fef */
        /*0288*/                   MOV R11, R11;                      /* 0x5c98078000b7000b */
        /*0290*/                   MOV R10, R10;                      /* 0x5c98078000a7000a */
        /*0298*/                   MOV R11, R11;                      /* 0x5c98078000b7000b */
                                                                      /* 0x00643c03fde01fef */
        /*02a8*/                   MOV R12, R10;                      /* 0x5c98078000a7000c */
        /*02b0*/                   MOV R10, R11;                      /* 0x5c98078000b7000a */
        /*02b8*/                   LDS.U.32 R10, [R12];               /* 0xef4c100000070c0a */
                                                                      /* 0x007fbc03fde01fef */
        /*02c8*/                   MOV R10, R10;                      /* 0x5c98078000a7000a */
        /*02d0*/                   CS2R R11, SR_CLOCKLO;              /* 0x50c800000507000b */
```

* (2 mov for start clock)
* 22 (s2r/imull32i/mov/iadd) + lds.u.32 = 2 imul + lds.u.32 + 20 other inst
* (1 mov for end clock)


### global store
```
     /*01e8*/                   CS2R R9, SR_CLOCKLO;             /* 0x50c8000005070009 */
        /*01f0*/                   MOV R9, R9;                      /* 0x5c98078000970009 */
        /*01f8*/                   MOV R12, R9;                     /* 0x5c9807800097000c */
                                                                    /* 0x00643c03fde0190f */
        /*0208*/                   S2R R9, SR_TID.X;                /* 0xf0c8000002170009 */
        /*0210*/                   MOV R9, R9;                      /* 0x5c98078000970009 */
        /*0218*/                   IMUL32I.U32.U32 R10, R9, 0x4;    /* 0x1f0000000047090a */
                                                                    /* 0x007fbc03fde0190f */
        /*0228*/                   IMUL32I.U32.U32.HI R9, R9, 0x4;  /* 0x1f20000000470909 */
        /*0230*/                   MOV R10, R10;                    /* 0x5c98078000a7000a */
        /*0238*/                   MOV R13, R9;                     /* 0x5c9807800097000d */
                                                                    /* 0x007fbc03fde01fef */
        /*0248*/                   MOV R9, R2;                      /* 0x5c98078000270009 */
        /*0250*/                   MOV R14, R3;                     /* 0x5c9807800037000e */
        /*0258*/                   IADD R9.CC, R9, R10;             /* 0x5c10800000a70909 */
                                                                    /* 0x007fbc03fde01fef */
        /*0268*/                   IADD.X R10, R14, R13;            /* 0x5c10080000d70e0a */
        /*0270*/                   MOV R14, R9;                     /* 0x5c9807800097000e */
        /*0278*/                   MOV R15, R10;                    /* 0x5c98078000a7000f */
                                                                    /* 0x0067bc03fde01fef */
        /*0288*/                   MOV R14, R14;                    /* 0x5c98078000e7000e */
        /*0290*/                   MOV R15, R15;                    /* 0x5c98078000f7000f */
        /*0298*/                   STG.E [R14], R6;                 /* 0xeedc200000070e06 */
                                                                    /* 0x007fbc03fde01fef */
        /*02a8*/                   CS2R R9, SR_CLOCKLO;             /* 0x50c8000005070009 */
```



### store to Shared Memory

```
       /*01e8*/                   CS2R R9, SR_CLOCKLO;             /* 0x50c8000005070009 */
        /*01f0*/                   MOV R9, R9;                      /* 0x5c98078000970009 */
        /*01f8*/                   MOV R11, R9;                     /* 0x5c9807800097000b */
                                                                    /* 0x00643c03fde0190f */
        /*0208*/                   S2R R9, SR_TID.X;                /* 0xf0c8000002170009 */
        /*0210*/                   MOV R9, R9;                      /* 0x5c98078000970009 */
        /*0218*/                   IMUL32I.U32.U32 R12, R9, 0x4;    /* 0x1f0000000047090c */
                                                                    /* 0x007fbc03fde0190f */
        /*0228*/                   IMUL32I.U32.U32.HI R9, R9, 0x4;  /* 0x1f20000000470909 */
        /*0230*/                   MOV R14, R12;                    /* 0x5c98078000c7000e */
        /*0238*/                   MOV R15, R9;                     /* 0x5c9807800097000f */
                                                                    /* 0x007fbc03fde0190f */
        /*0248*/                   I2I.U32.U32 R9, RZ;              /* 0x5ce000000ff70a09 */
        /*0250*/                   MOV R12, R9;                     /* 0x5c9807800097000c */
        /*0258*/                   MOV R13, RZ;                     /* 0x5c9807800ff7000d */
                                                                    /* 0x007fbc03fde01fef */
        /*0268*/                   MOV R12, R12;                    /* 0x5c98078000c7000c */
        /*0270*/                   MOV R13, R13;                    /* 0x5c98078000d7000d */
        /*0278*/                   MOV R9, R12;                     /* 0x5c98078000c70009 */
                                                                    /* 0x007fbc03fde01fef */
        /*0288*/                   MOV R12, R13;                    /* 0x5c98078000d7000c */
        /*0290*/                   IADD R9.CC, R9, R14;             /* 0x5c10800000e70909 */
        /*0298*/                   IADD.X R12, R12, R15;            /* 0x5c10080000f70c0c */
                                                                    /* 0x007fbc03fde01fef */
        /*02a8*/                   MOV R13, R12;                    /* 0x5c98078000c7000d */
        /*02b0*/                   MOV R12, R9;                     /* 0x5c9807800097000c */
        /*02b8*/                   MOV R12, R12;                    /* 0x5c98078000c7000c */
                                                                    /* 0x007fbc03fde01fef */
        /*02c8*/                   MOV R13, R13;                    /* 0x5c98078000d7000d */
        /*02d0*/                   MOV R9, R12;                     /* 0x5c98078000c70009 */
        /*02d8*/                   MOV R14, R13;                    /* 0x5c98078000d7000e */
                                                                    /* 0x007fbc03fde019ef */
        /*02e8*/                   STS [R9], R8;                    /* 0xef5c000000070908 */
        /*02f0*/                   MOV R8, R12;                     /* 0x5c98078000c70008 */
        /*02f8*/                   MOV R9, R13;                     /* 0x5c98078000d70009 */
                                                                    /* 0x007fbc03fde01fef */
        /*0308*/                   CS2R R12, SR_CLOCKLO;            /* 0x50c800000507000c */

```


### Constant Memory (Read / Load)

```
      /*0150*/                   CS2R R6, SR_CLOCKLO;               /* 0x50c8000005070006 */
        /*0158*/                   MOV R6, R6;                        /* 0x5c98078000670006 */
                                                                      /* 0x007fbc03fde01fef */
        /*0168*/                   MOV R9, R6;                        /* 0x5c98078000670009 */
        /*0170*/                   MOV R6, R4;                        /* 0x5c98078000470006 */
        /*0178*/                   MOV R7, R5;                        /* 0x5c98078000570007 */
                                                                      /* 0x00643c03fde01fef */
        /*0188*/                   MOV R4, R10;                       /* 0x5c98078000a70004 */
        /*0190*/                   MOV R5, R11;                       /* 0x5c98078000b70005 */
        /*0198*/                   S2R R10, SR_TID.X;                 /* 0xf0c800000217000a */
                                                                      /* 0x00643c0321e01fef */
        /*01a8*/                   MOV R10, R10;                      /* 0x5c98078000a7000a */
        /*01b0*/                   IMUL32I.U32.U32 R11, R10, 0x4;     /* 0x1f00000000470a0b */
        /*01b8*/                   IMUL32I.U32.U32.HI R10, R10, 0x4;  /* 0x1f20000000470a0a */
                                                                      /* 0x00643c03fde01fef */
        /*01c8*/                   MOV R13, R11;                      /* 0x5c98078000b7000d */
        /*01d0*/                   MOV R14, R10;                      /* 0x5c98078000a7000e */
        /*01d8*/                   I2I.U32.U32 R10, RZ;               /* 0x5ce000000ff70a0a */
                                                                      /* 0x007fbc03fde01fef */
        /*01e8*/                   MOV R10, R10;                      /* 0x5c98078000a7000a */
        /*01f0*/                   MOV R11, RZ;                       /* 0x5c9807800ff7000b */
        /*01f8*/                   MOV R10, R10;                      /* 0x5c98078000a7000a */
                                                                      /* 0x007fbc03fde01fef */
        /*0208*/                   MOV R11, R11;                      /* 0x5c98078000b7000b */
        /*0210*/                   MOV R12, R10;                      /* 0x5c98078000a7000c */
        /*0218*/                   MOV R11, R11;                      /* 0x5c98078000b7000b */
                                                                      /* 0x007fbc03fde01fef */
        /*0228*/                   IADD R10.CC, R12, R13;             /* 0x5c10800000d70c0a */
        /*0230*/                   IADD.X R11, R11, R14;              /* 0x5c10080000e70b0b */
        /*0238*/                   MOV R10, R10;                      /* 0x5c98078000a7000a */
                                                                      /* 0x007fbc03fde01fef */
        /*0248*/                   MOV R11, R11;                      /* 0x5c98078000b7000b */
        /*0250*/                   MOV R10, R10;                      /* 0x5c98078000a7000a */
        /*0258*/                   MOV R11, R11;                      /* 0x5c98078000b7000b */
                                                                      /* 0x00643c03fde01fef */
        /*0268*/                   MOV R12, R10;                      /* 0x5c98078000a7000c */
        /*0270*/                   MOV R10, R11;                      /* 0x5c98078000b7000a */
        /*0278*/                   LDC R10, c[0x3][R12];              /* 0xef94003000070c0a */
                                                                      /* 0x007fbc03fde01fef */
        /*0288*/                   MOV R10, R10;                      /* 0x5c98078000a7000a */
        /*0290*/                   MOV R6, R6;                        /* 0x5c98078000670006 */
        /*0298*/                   MOV R7, R7;                        /* 0x5c98078000770007 */
                                                                      /* 0x007fbc03fde01fef */
        /*02a8*/                   MOV R4, R4;                        /* 0x5c98078000470004 */
        /*02b0*/                   MOV R5, R5;                        /* 0x5c98078000570005 */
        /*02b8*/                   MOV R11, R4;                       /* 0x5c9807800047000b */
                                                                      /* 0x007fbc03fde01fef */
        /*02c8*/                   MOV R16, R5;                       /* 0x5c98078000570010 */
        /*02d0*/                   MOV R4, R10;                       /* 0x5c98078000a70004 */
        /*02d8*/                   CS2R R5, SR_CLOCKLO;               /* 0x50c8000005070005 */
```


2mov (start clock) + 34 instructions (ldc  + 2 IADD + 2 imul + 29 others)+ 1mov(end clock)

### Load from Local Memory
```
        /*0370*/                   CS2R R10, SR_CLOCKLO;                 /* 0x50c800000507000a */
        /*0378*/                   MOV R10, R10;                         /* 0x5c98078000a7000a */
                                                                         /* 0x007fbc03fde01fef */
        /*0388*/                   MOV R12, R10;                         /* 0x5c98078000a7000c */
        /*0390*/                   MOV R8, R8;                           /* 0x5c98078000870008 */
        /*0398*/                   MOV R9, R9;                           /* 0x5c98078000970009 */
                                                                         /* 0x00643c03fde01fef */
        /*03a8*/                   MOV R10, R8;                          /* 0x5c9807800087000a */
        /*03b0*/                   MOV R8, R9;                           /* 0x5c98078000970008 */
        /*03b8*/                   LDL R8, [R10];                        /* 0xef44000000070a08 */
                                                                         /* 0x007fbc03fde01fef */
        /*03c8*/                   FADD R8, R8, RZ;                      /* 0x5c5800000ff70808 */
        /*03d0*/                   MOV R8, R8;                           /* 0x5c98078000870008 */
        /*03d8*/                   CS2R R9, SR_CLOCKLO;                  /* 0x50c8000005070009 */
```
