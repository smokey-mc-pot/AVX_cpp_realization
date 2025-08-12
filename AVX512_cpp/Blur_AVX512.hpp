#include <immintrin.h>  
#include "oneapi/tbb.h"
#include <iostream>

void Gaussian_Blur_optimized_3x3_AVX512(unsigned char **frame1,unsigned char **filt, unsigned int M,  unsigned int N,  unsigned short int divisor, signed char **filter);
