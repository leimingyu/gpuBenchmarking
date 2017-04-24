# Results 
### GTX 950 (CUDA 8)
* global_load  650 (cycles)
* global_store  15 (cycles)
* shared_load  23 (cycles)
* shared_store  15 (cycles)

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
* 22 (s2r/imull32i/mov/iadd) + lds.u 
* (1 mov for end clock)


### global store
using st.global.f32
```
        /*01a8*/                   CS2R R9, SR_CLOCKLO;             /* 0x50c8000005070009 */
        /*01b0*/                   MOV R9, R9;                      /* 0x5c98078000970009 */
        /*01b8*/                   MOV R12, R9;                     /* 0x5c9807800097000c */
                                                                    /* 0x0067bc03fde01fef */
        /*01c8*/                   MOV R2, R2;                      /* 0x5c98078000270002 */
        /*01d0*/                   MOV R3, R3;                      /* 0x5c98078000370003 */
        /*01d8*/                   STG.E [R2], R6;                  /* 0xeedc200000070206 */
                                                                    /* 0x007fbc03fde01fef */
        /*01e8*/                   CS2R R9, SR_CLOCKLO;             /* 0x50c8000005070009 */
        /*01f0*/                   MOV R9, R9;                      /* 0x5c98078000970009 */
```

us st.f32
```
        /*01a8*/                   CS2R R9, SR_CLOCKLO;             /* 0x50c8000005070009 */
        /*01b0*/                   MOV R9, R9;                      /* 0x5c98078000970009 */
        /*01b8*/                   MOV R12, R9;                     /* 0x5c9807800097000c */
                                                                    /* 0x007fbc03fde01fef */
        /*01c8*/                   MOV R2, R2;                      /* 0x5c98078000270002 */
        /*01d0*/                   MOV R3, R3;                      /* 0x5c98078000370003 */
        /*01d8*/                   MOV R10, R2;                     /* 0x5c9807800027000a */
                                                                    /* 0x007fbc03fde01fef */
        /*01e8*/                   MOV R11, R3;                     /* 0x5c9807800037000b */
        /*01f0*/                   LEA R9.CC, R10, RZ;              /* 0x5bd780000ff70a09 */
        /*01f8*/                   LEA.HI.X P0, R10, R10, RZ, R11;  /* 0x5bd805c00ff70a0a */
                                                                    /* 0x0067bc03fde01fef */
        /*0208*/                   MOV R11, R10;                    /* 0x5c98078000a7000b */
        /*0210*/                   MOV R10, R9;                     /* 0x5c9807800097000a */
        /*0218*/                   ST.E [R10], R6, P0;              /* 0xa090000000070a06 */
                                                                    /* 0x007fbc03fde01fef */
        /*0228*/                   CS2R R9, SR_CLOCKLO;             /* 0x50c8000005070009 */

```


### shared store
st.shared.f32 doesn't work, use st.f32 instead!

```
        /*01a8*/                   CS2R R10, SR_CLOCKLO;            /* 0x50c800000507000a */
        /*01b0*/                   MOV R10, R10;                    /* 0x5c98078000a7000a */
        /*01b8*/                   MOV R12, R10;                    /* 0x5c98078000a7000c */
                                                                    /* 0x007fbc03fde0190f */
        /*01c8*/                   I2I.U32.U32 R10, RZ;             /* 0x5ce000000ff70a0a */
        /*01d0*/                   MOV R10, R10;                    /* 0x5c98078000a7000a */
        /*01d8*/                   MOV R11, RZ;                     /* 0x5c9807800ff7000b */
                                                                    /* 0x007fbc03fde01fef */
        /*01e8*/                   MOV R10, R10;                    /* 0x5c98078000a7000a */
        /*01f0*/                   MOV R11, R11;                    /* 0x5c98078000b7000b */
        /*01f8*/                   MOV R13, R10;                    /* 0x5c98078000a7000d */
                                                                    /* 0x007fbc03fde01fef */
        /*0208*/                   MOV R14, R11;                    /* 0x5c98078000b7000e */
        /*0210*/                   MOV R10, c[0x0][0x0];            /* 0x4c9807800007000a */
        /*0218*/                   MOV R11, RZ;                     /* 0x5c9807800ff7000b */
                                                                    /* 0x007fbc03fde01fef */
        /*0228*/                   LOP.OR R10, R13, R10;            /* 0x5c47020000a70d0a */
        /*0230*/                   LOP.OR R11, R14, R11;            /* 0x5c47020000b70e0b */
        /*0238*/                   MOV R10, R10;                    /* 0x5c98078000a7000a */
                                                                    /* 0x007fbc03fde01fef */
        /*0248*/                   MOV R11, R11;                    /* 0x5c98078000b7000b */
        /*0250*/                   MOV R10, R10;                    /* 0x5c98078000a7000a */
        /*0258*/                   MOV R11, R11;                    /* 0x5c98078000b7000b */
                                                                    /* 0x007fbc03fde01fef */
        /*0268*/                   MOV R13, R10;                    /* 0x5c98078000a7000d */
        /*0270*/                   MOV R11, R11;                    /* 0x5c98078000b7000b */
        /*0278*/                   LEA R10.CC, R13, RZ;             /* 0x5bd780000ff70d0a */
                                                                    /* 0x007fbc03fde01fef */
        /*0288*/                   LEA.HI.X P0, R11, R13, RZ, R11;  /* 0x5bd805c00ff70d0b */
        /*0290*/                   MOV R10, R10;                    /* 0x5c98078000a7000a */
        /*0298*/                   MOV R11, R11;                    /* 0x5c98078000b7000b */
                                                                    /* 0x007fbc03fde019ef */
        /*02a8*/                   ST.E [R10], R7, P0;              /* 0xa090000000070a07 */
        /*02b0*/                   CS2R R10, SR_CLOCKLO;            /* 0x50c800000507000a */

```
