#include <immintrin.h>  
#include "oneapi/tbb.h"
#include <iostream>

void Gaussian_Blur_optimized_3x3(unsigned char **frame1,unsigned char **filt, unsigned int M,  unsigned int N,  unsigned short int divisor, signed char **filter);
int loop_reminder_3x3(unsigned char **frame1,unsigned char **filt, unsigned int M,  unsigned int N, unsigned int row,  unsigned int col, unsigned int REMINDER_ITERATIONS, unsigned int division_case, unsigned short int divisor,signed char **filter, __m256i c0, __m256i c1, __m256i c0_sh1, __m256i c1_sh1, __m256i c0_sh2, __m256i c1_sh2,  __m256i c0_sh3, __m256i c1_sh3, __m256i f);
int loop_reminder_3x3_first_values(unsigned char **frame1,unsigned char **filt, unsigned int M,  unsigned int N, unsigned int row, unsigned int col, unsigned int REMINDER_ITERATIONS, unsigned int division_case, unsigned short int divisor,signed char **filter, __m256i c0, __m256i c1, __m256i c0_sh1, __m256i c1_sh1, __m256i c0_sh2, __m256i c1_sh2,  __m256i c0_sh3, __m256i c1_sh3, __m256i f);
int loop_reminder_3x3_last_values(unsigned char **frame1,unsigned char **filt, unsigned int M,  unsigned int N,  unsigned int col, unsigned int REMINDER_ITERATIONS, unsigned int division_case, unsigned short int divisor,signed char **filter, __m256i c0, __m256i c1, __m256i c0_sh1, __m256i c1_sh1, __m256i c0_sh2, __m256i c1_sh2,  __m256i c0_sh3, __m256i c1_sh3, __m256i f);
