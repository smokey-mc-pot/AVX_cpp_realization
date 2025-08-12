#include "Blur_AVX512.hpp"

unsigned short int f_vector[32]; 
unsigned int num_of_shift;

unsigned int prepare_for_division(unsigned short int divisor){
  if (divisor == 1)
    return 4;

	num_of_shift = (unsigned int)floorf(log2f(divisor)); 
	float f = powf(2, 16+num_of_shift)/divisor;
	int integer_part_f = (int)f;
	float float_part_f = f - integer_part_f;

	if (float_part_f < 0.0001) 
		return 1; 
	else if (float_part_f < 0.5){
		unsigned short int tmp = (unsigned short int)floorf(f);
		__m512i f_vec = _mm512_set1_epi16(tmp);
		_mm512_store_si512((__m512i*)&f_vector[0], f_vec);
		return 2; 
	}
	else{
		unsigned short int tmp = (unsigned short int)ceilf(f);
		__m512i f_vec = _mm512_set1_epi16(tmp);
		_mm512_store_si512((__m512i*)&f_vector[0], f_vec);
		return 3; 
	}
}

__m512i division(unsigned int division_case, __m512i zmm5, __m512i f){
	__m512i zmm4, zmm6;

	if (division_case == 1)
		return _mm512_srli_epi16(zmm5, num_of_shift); 
	else if (division_case == 2){ 
		zmm5 = _mm512_add_epi16(zmm5, _mm512_set1_epi16(1)); 
		zmm6 = _mm512_mulhi_epu16(zmm5, f);      
		zmm4 = _mm512_sub_epi16(zmm5, zmm6);    
		zmm4 = _mm512_srli_epi16(zmm4, 16);      
		zmm6 = _mm512_add_epi16(zmm6, zmm4);     
		return (_mm512_srli_epi16(zmm6, num_of_shift)); 
	}
	else if (division_case == 3){ 
		zmm6 = _mm512_mulhi_epu16(zmm5, f);    
		zmm4 = _mm512_sub_epi16(zmm5, zmm6);     
		zmm4 = _mm512_srli_epi16(zmm4, 16);   
		zmm6 = _mm512_add_epi16(zmm6, zmm4);     
		return (_mm512_srli_epi16(zmm6, num_of_shift)); 
	}
	else 
		return zmm5;
}

void Gaussian_Blur_optimized_3x3_AVX512(unsigned char** src_image, unsigned char** out_image,  unsigned int img_length,  unsigned int img_height,  unsigned short int divisor, signed char** filter){
	tbb::tick_count t0 = tbb::tick_count::now();
	signed char f00 = filter[0][0]; // 1 
	signed char f01 = filter[0][1]; // 2 
	signed char f02 = filter[0][2]; // 1
	signed char f10 = filter[1][0]; // 2
	signed char f11 = filter[1][1]; // 4
	signed char f12 = filter[1][2]; // 2

	__m512i mask_filter_0        = _mm512_set_epi8(0, f02, f01, f00, 0, f02, f01, f00, 0, f02, f01, f00, 0, f02, f01, f00, 0, f02, f01, f00, 0, f02, f01, f00, 0, f02, f01, f00, 0, f02, f01, f00, 0, f02, f01, f00, 0, f02, f01, f00, 0, f02, f01, f00, 0, f02, f01, f00, 0, f02, f01, f00, 0, f02, f01, f00, 0, f02, f01, f00, 0, f02, f01, f00);
	__m512i mask_filter_0_shift1 = _mm512_set_epi8(f02, f01, f00, 0, f02, f01, f00, 0, f02, f01, f00, 0, f02, f01, f00, 0, f02, f01, f00, 0, f02, f01, f00, 0, f02, f01, f00, 0, f02, f01, f00, 0, f02, f01, f00, 0, f02, f01, f00, 0, f02, f01, f00, 0, f02, f01, f00, 0, f02, f01, f00, 0, f02, f01, f00, 0, f02, f01, f00, 0, f02, f01, f00, 0);
	__m512i mask_filter_0_shift2 = _mm512_set_epi8(0, 0, 0, f02, f01, f00, 0, f02, f01, f00, 0, f02, f01, f00, 0, f02, f01, f00, 0, f02, f01, f00, 0, f02, f01, f00, 0, f02, f01, f00, 0, f02, f01, f00, 0, f02, f01, f00, 0, f02, f01, f00, 0, f02, f01, f00, 0, f02, f01, f00, 0, f02, f01, f00, 0, f02, f01, f00, 0, f02, f01, f00, 0, 0);
	__m512i mask_filter_0_shift3 = _mm512_set_epi8(0, 0, f02, f01, f00, 0, f02, f01, f00, 0, f02, f01, f00, 0, f02, f01, f00, 0, f02, f01, f00, 0, f02, f01, f00, 0, f02, f01, f00, 0, f02, f01, f00, 0, f02, f01, f00, 0, f02, f01, f00, 0, f02, f01, f00, 0, f02, f01, f00, 0, f02, f01, f00, 0, f02, f01, f00, 0, f02, f01, f00, 0, 0, 0);

	__m512i mask_filter_1        = _mm512_set_epi8(0, f12, f11, f10, 0, f12, f11, f10, 0, f12, f11, f10, 0, f12, f11, f10, 0, f12, f11, f10, 0, f12, f11, f10, 0, f12, f11, f10, 0, f12, f11, f10, 0, f12, f11, f10, 0, f12, f11, f10, 0, f12, f11, f10, 0, f12, f11, f10, 0, f12, f11, f10, 0, f12, f11 ,f10, 0, f12, f11, f10, 0, f12, f11, f10);
	__m512i mask_filter_1_shift1 = _mm512_set_epi8(f12, f11, f10, 0, f12, f11, f10, 0, f12, f11, f10, 0, f12, f11, f10, 0, f12, f11, f10, 0, f12, f11, f10, 0, f12, f11, f10, 0, f12, f11, f10, 0, f12, f11, f10, 0, f12, f11, f10, 0, f12, f11, f10, 0, f12, f11, f10, 0, f12, f11, f10, 0, f12, f11, f10, 0, f12, f11, f10, 0, f12, f11, f10, 0);
	__m512i mask_filter_1_shift2 = _mm512_set_epi8(0, 0, 0, f12, f11, f10, 0, f12, f11, f10, 0, f12, f11, f10, 0, f12, f11, f10, 0, f12, f11, f10, 0, f12, f11, f10, 0, f12, f11, f10, 0, f12, f11, f10, 0, f12, f11, f10, 0, f12, f11, f10, 0, f12, f11, f10, 0, f12, f11, f10, 0, f12, f11, f10, 0, f12, f11, f10, 0, f12, f11, f10, 0, 0);
	__m512i mask_filter_1_shift3 = _mm512_set_epi8(0, 0, f12, f11, f10, 0, f12, f11, f10, 0, f12, f11, f10, 0, f12, f11, f10, 0, f12, f11, f10, 0, f12, f11, f10, 0, f12, f11, f10, 0, f12, f11, f10, 0, f12, f11, f10, 0, f12, f11, f10, 0, f12, f11, f10, 0, f12, f11, f10, 0, f12, f11, f10, 0, f12, f11, f10, 0, f12, f11, f10, 0, 0, 0); 

	unsigned int division_case = prepare_for_division(divisor);
	__m512i f = _mm512_load_si512((__m512i*)&f_vector[0]); 

	__m512i mask_for_rem_first = _mm512_set_epi8(255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 0);
	__m512i idx = _mm512_set_epi8(62, 61, 60, 59, 58, 57, 56, 55, 54, 53, 52, 51, 50, 49, 48, 47, 46, 45, 44, 43, 42, 41, 40, 39, 38, 37, 36, 35, 34, 33, 32, 31, 30, 29, 28, 27, 26, 25, 24, 23, 22, 21, 20, 19, 18, 17, 16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0, 0);
	__m512i idx2 = _mm512_set_epi16(0, 0, 30, 29, 28, 27, 26, 25, 24, 23, 22, 21, 20, 19, 18, 17, 16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1);
	__mmask32 k1 = 0x3FFFFFFF;
	__mmask32 k_odd = 0x55555555; 
	__mmask32 k_even = 0xAAAAAAAA; 
	__m512i zmm0, zmm1, zmm2, zmm3, zmm4, zmm5, only_for_even_output, only_for_odd_output;

	for (int row = 0; row <= img_height-3; row++){
		for (int column = 0; column <= img_length-64; column += 62){ 
			if (column == 0){ 
				zmm0 = _mm512_permutexvar_epi8(idx, _mm512_load_si512((__m512i*)&src_image[row][0]));
				zmm0 = _mm512_and_si512(zmm0, mask_for_rem_first);
				zmm1 = _mm512_permutexvar_epi8(idx, _mm512_load_si512((__m512i*)&src_image[row+1][0]));
				zmm1 = _mm512_and_si512(zmm1, mask_for_rem_first);
				zmm2 = _mm512_permutexvar_epi8(idx, _mm512_load_si512((__m512i*)&src_image[row+2][0]));
				zmm2 = _mm512_and_si512(zmm2, mask_for_rem_first);
			}
			else{ 
				zmm0 = _mm512_load_si512((__m512i*)&src_image[row][column]);
				zmm1 = _mm512_load_si512((__m512i*)&src_image[row+1][column]);
				zmm2 = _mm512_load_si512((__m512i*)&src_image[row+2][column]);
			}

			zmm3 = _mm512_maddubs_epi16(zmm0, mask_filter_0);
			zmm4 = _mm512_maddubs_epi16(zmm1, mask_filter_1);
			zmm5 = _mm512_maddubs_epi16(zmm2, mask_filter_0);
			zmm3 = _mm512_add_epi16(zmm3, zmm4);
			zmm3 = _mm512_add_epi16(zmm3, zmm5);
			zmm4 = _mm512_bsrli_epi128(zmm3, 2);
			only_for_even_output = _mm512_maskz_add_epi16(k_odd, zmm4, zmm3); 

			zmm3 = _mm512_maddubs_epi16(zmm0, mask_filter_0_shift1);
			zmm4 = _mm512_maddubs_epi16(zmm1, mask_filter_1_shift1);
			zmm5 = _mm512_maddubs_epi16(zmm2, mask_filter_0_shift1);
			zmm3 = _mm512_add_epi16(zmm3, zmm4);
			zmm3 = _mm512_add_epi16(zmm3, zmm5);
			zmm4 = _mm512_bsrli_epi128(zmm3, 2);
			only_for_odd_output = _mm512_maskz_add_epi16(k_odd, zmm4, zmm3); 

			zmm3 = _mm512_maddubs_epi16(zmm0, mask_filter_0_shift2);
			zmm4 = _mm512_maddubs_epi16(zmm1, mask_filter_1_shift2);
			zmm5 = _mm512_maddubs_epi16(zmm2, mask_filter_0_shift2);
			zmm3 = _mm512_add_epi16(zmm3, zmm4); 
			zmm3 = _mm512_add_epi16(zmm3, zmm5); 
			zmm4 = _mm512_maskz_permutexvar_epi16(k1, idx2, zmm3); 
			zmm5 = _mm512_maskz_add_epi16(k_even, zmm4, zmm3);
			only_for_even_output = _mm512_add_epi16(only_for_even_output, zmm5); 

			zmm3 = _mm512_maddubs_epi16(zmm0, mask_filter_0_shift3);
			zmm4 = _mm512_maddubs_epi16(zmm1, mask_filter_1_shift3);
			zmm5 = _mm512_maddubs_epi16(zmm2, mask_filter_0_shift3);
			zmm3 = _mm512_add_epi16(zmm3, zmm4);
			zmm3 = _mm512_add_epi16(zmm3, zmm5);
			zmm4 = _mm512_maskz_permutexvar_epi16(k1, idx2, zmm3);
			zmm5 = _mm512_maskz_add_epi16(k_even, zmm4, zmm3); 
			only_for_odd_output = _mm512_add_epi16(only_for_odd_output, zmm5); 

			only_for_even_output = division(division_case, only_for_even_output, f); 
			only_for_odd_output = division(division_case, only_for_odd_output, f); 

			only_for_odd_output = _mm512_bslli_epi128(only_for_odd_output, 1); 
			only_for_even_output = _mm512_add_epi8(only_for_even_output, only_for_odd_output);
			_mm512_store_si512((__m512i*)&out_image[row+1][column], only_for_even_output);
		}
	}
	std::cout << "Time: " << (tbb::tick_count::now() - t0).seconds() << " seconds" << std::endl;
}
