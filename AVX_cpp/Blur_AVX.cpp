#include "Blur_AVX.hpp"

// что сделать потом:
// переписать на avx512 (пока только использовал инструкции с масками, но для 256 битной разрядности) +
// понять что делать с константным делением +
// совместить вертикальное и горизонтальное сложение:
//   как минимум: есть ли интринсик или последовательность инструкций для сложения соседних 32 бит и запись в старшую часть
// + сдвигать или использовать _mm256_maskz_permutexvar_epi16 (второе)
// + использовать сложение и логическое и, или использовать сложение c zero writemask k (второе)
// +- обработать хвост с масками, записывается ли только выбранное? (да)
// +- глобальные переменные или каждый раз передавать (передавать и создавать в теле функции)
// написать основной алгоритм и каждый раз вызывать или раскрыть весь в каждом цикле
// разобраться с многопоточностью 
// провести анализ для изображений разного размера (на небольших будет снижение производительности с avx512)
unsigned short int f_vector[16]; 
unsigned int num_of_shift;
unsigned char mask_for_remaining[33][32] = {
		{255,255,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
		{255,255,255,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
		{255,255,255,255,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
		{255,255,255,255,255,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
		{255,255,255,255,255,255,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
		{255,255,255,255,255,255,255,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
		{255,255,255,255,255,255,255,255,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
		{255,255,255,255,255,255,255,255,255,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
		{255,255,255,255,255,255,255,255,255,255,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
		{255,255,255,255,255,255,255,255,255,255,255,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
		{255,255,255,255,255,255,255,255,255,255,255,255,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
		{255,255,255,255,255,255,255,255,255,255,255,255,255,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
		{255,255,255,255,255,255,255,255,255,255,255,255,255,255,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
		{255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
		{255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
		{255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
		{255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
		{255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,0,0,0,0,0,0,0,0,0,0,0,0,0},
		{255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,0,0,0,0,0,0,0,0,0,0,0,0},
		{255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,0,0,0,0,0,0,0,0,0,0,0},
		{255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,0,0,0,0,0,0,0,0,0,0},
		{255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,0,0,0,0,0,0,0,0,0},
		{255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,0,0,0,0,0,0,0,0},
		{255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,0,0,0,0,0,0,0},
		{255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,0,0,0,0,0,0},
		{255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,0,0,0,0,0},
		{255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,0,0,0,0},
		{255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,0,0,0},
		{255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,0,0},
		{255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,0},
		{255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255},
		{255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255},
		{255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255},
} ;

unsigned int prepare_for_division(unsigned short int divisor){
	__m256i f_vec;
	unsigned short int tmp;

  if (divisor == 1)
    return 4;

	//unsigned int w = 16; 
	num_of_shift = (unsigned int)floorf(log2f(divisor));
	float f = powf(2, 16+num_of_shift)/divisor;
	int integer_part_f = (int)f;
	float float_part_f = f - integer_part_f;

	if (float_part_f < 0.0001) // if f == 0.0
		return 1; // case A
	else if (float_part_f < 0.5){
		tmp = (unsigned short int)floorf(f);
		f_vec = _mm256_set1_epi16(tmp);
		_mm256_store_si256((__m256i*)&f_vector[0], f_vec);
		return 2; //case num_of_shift
	}
	else{
		tmp = (unsigned short int)ceilf(f);
		f_vec = _mm256_set1_epi16(tmp);
		_mm256_store_si256((__m256i*)&f_vector[0], f_vec);
		return 3; // case C
	}
}

inline __m256i division(unsigned int division_case, __m256i ymm5, __m256i f){
	__m256i ymm4, m3;

	if (division_case == 1){ // case A потом все запихнуть чисто в return
		ymm5 = _mm256_srli_epi16(ymm5, num_of_shift);
		return ymm5; // логический сдвиг вправо с num_of_shift 
	}
	else if (division_case == 2){ // case num_of_shift
		ymm5 = _mm256_add_epi16(ymm5, _mm256_set1_epi16(1)); // ymm5 = ymm5+1
		m3 = _mm256_mulhi_epu16(ymm5, f);    // умножение верхних unsigned words 
		ymm4 = _mm256_sub_epi16(ymm5, m3);     // вычитание
		ymm4 = _mm256_srli_epi16(ymm4, 16);    // логический сдвиг вправо с w 
		m3 = _mm256_add_epi16(m3, ymm4);     // сложение
		return (_mm256_srli_epi16(m3, num_of_shift)); // логический сдвиг вправо с num_of_shift
	}
	else if (division_case == 3){ // case C
		m3 = _mm256_mulhi_epu16(ymm5, f);    // умножение верхних unsigned words
		ymm4 = _mm256_sub_epi16(ymm5, m3);     // вычитание
		ymm4 = _mm256_srli_epi16(ymm4, 16);    // логический сдвиг вправо с w
		m3 = _mm256_add_epi16(m3, ymm4);     // сложение
		return (_mm256_srli_epi16(m3, num_of_shift)); // логический сдвиг вправо с num_of_shift
	}
	else // деление на 1
		return ymm5;
}

void Gaussian_Blur_optimized_3x3(unsigned char** src_image, unsigned char** out_image,  unsigned int img_length,  unsigned int img_height,  unsigned short int divisor, signed char** filter){
	// filter - передается как (signed char **Mask), загружаются первые 2 строки, т.к третья равна первой
	tbb::tick_count t0 = tbb::tick_count::now();
	signed char f00 = filter[0][0]; // 1 
	signed char f01 = filter[0][1]; // 2 
	signed char f02 = filter[0][2]; // 1
	signed char f10 = filter[1][0]; // 2
	signed char f11 = filter[1][1]; // 4
	signed char f12 = filter[1][2]; // 2

	__m256i mask_filter_0        = _mm256_set_epi8(0, f02, f01, f00, 0, f02, f01, f00, 0, f02, f01, f00, 0, f02, f01, f00, 0, f02, f01, f00, 0, f02, f01, f00, 0, f02, f01, f00, 0, f02, f01, f00);
	__m256i mask_filter_0_shift1 = _mm256_set_epi8(f02, f01, f00, 0, f02, f01, f00, 0, f02, f01, f00, 0, f02, f01, f00, 0, f02, f01, f00, 0, f02, f01, f00, 0, f02, f01, f00, 0, f02, f01, f00, 0);
	__m256i mask_filter_0_shift2 = _mm256_set_epi8(0, 0, 0, f02, f01, f00, 0, f02, f01, f00, 0, f02, f01, f00, 0, f02, f01, f00, 0, f02, f01, f00, 0, f02, f01, f00, 0, f02, f01, f00, 0, 0);
	__m256i mask_filter_0_shift3 = _mm256_set_epi8(0, 0, f02, f01, f00, 0, f02, f01, f00, 0, f02, f01, f00, 0, f02, f01, f00, 0, f02, f01, f00, 0, f02, f01, f00, 0, f02, f01, f00, 0, 0, 0);

	__m256i mask_filter_1        = _mm256_set_epi8(0, f12, f11, f10, 0, f12, f11, f10, 0, f12, f11, f10, 0, f12, f11, f10, 0, f12, f11, f10, 0, f12, f11, f10, 0, f12, f11, f10, 0, f12, f11 ,f10);
	__m256i mask_filter_1_shift1 = _mm256_set_epi8(f12, f11, f10, 0, f12, f11, f10, 0, f12, f11, f10, 0, f12, f11, f10, 0, f12, f11, f10, 0, f12, f11, f10, 0, f12, f11, f10, 0, f12, f11, f10, 0);
	__m256i mask_filter_1_shift2 = _mm256_set_epi8(0, 0, 0, f12, f11, f10, 0, f12, f11, f10, 0, f12, f11, f10, 0, f12, f11, f10, 0, f12, f11, f10, 0, f12, f11, f10, 0, f12, f11, f10, 0, 0);
	__m256i mask_filter_1_shift3 = _mm256_set_epi8(0, 0, f12, f11, f10, 0, f12, f11, f10, 0, f12, f11, f10, 0, f12, f11, f10, 0, f12, f11, f10, 0, f12, f11, f10, 0, f12, f11, f10, 0, 0, 0);

	// не понадобятся если переписать с _mm256_maskz_permutexvar_epi16
	// mask3 - используется в горизонатальном сложении с интринсиком _mm256_and_si256 
	// mask_prelude - используется чтобы занулить все кроме нужного элемента потерянного при сдвиге
	//__m256i mask3        = _mm256_set_epi16(0, 0, 0, 0, 0, 0, 0, 65535, 0, 0, 0, 0, 0, 0, 0, 0);
	//__m256i mask_prelude = _mm256_set_epi8(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 255, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0);

	// output_mask - используется в 1й и 2й итерации
	// output_mask_sh1 - используется в 3й и 4й итерации
	//__m256i output_mask     = _mm256_set_epi16(0, 65535, 0, 65535, 0, 65535, 0, 65535, 0, 65535, 0, 65535, 0, 65535, 0 ,65535);
	//__m256i output_mask_sh1 = _mm256_set_epi16(0, 0, 65535, 0, 65535, 0, 65535, 0, 65535, 0, 65535, 0, 65535, 0, 65535, 0);

	// num_of_remaining_elem_in_row - количество элементов которые останется обработать если в строке осталось меньше 30 элементов
	// передается в loop_reminder'ы, а в них исползуется в mask_for_remaining
	unsigned int num_of_remaining_elem_in_row = (img_length-((((img_length-32)/30)*30)+30)); 

	unsigned int division_case = prepare_for_division(divisor); 
	__m256i f = _mm256_load_si256((__m256i*)&f_vector[0]); 

	// маска для _mm256_maskz_permutexvar_epi8 для загрузки 32 пикселей и смещения на 1 вправо сразу
	__mmask32 k = 0xfffffffe; // 11111111111111111111111111111110
	// индексы сдвига
	__m256i idx = _mm256_set_epi8(30, 29, 28, 27, 26, 25, 24, 23, 22, 21, 20, 19, 18, 17, 16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0, 0);
	// маска для _mm256_maskz_permutexvar_epi16 чтобы не терять элемент при сдвиге и не возвращать его множеством инструкций
	__mmask64 k1 = 0x7FFF; // 0111111111111111
	__m256i idx2 = _mm256_set_epi16(0, 0, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1);
	// маска для выбора нужных элементов для only_for_even_output
	__mmask16 k_odd = 0x5555; // 0101010101010101
	__mmask16 k_even = 0x2AAA; // 0010101010101010

	//#pragma omp parallel{
	// row - строки
	// column - столбцы
	unsigned int row, column;
	// ymm0, ymm1, ymm2 - (вектор-строки img_height, img_height+1, img_height+2), упакованные в 256-битный вектор 
	// (32 8-битных элемента в каждом), которые содержат значения пикселей
	__m256i ymm0, ymm1, ymm2; 
	// ymm3, ymm4, ymm5 - промежуточные результаты для сдвига и хранения суммы
	__m256i	ymm3, ymm4, ymm5;
	__m256i only_for_even_output, only_for_odd_output;

/*---------------------- Gaussian Blur ---------------------------------*/ 
	for (row = 1; row < img_height-1; row++){ // цикл начинается со второй строки до предпоследней
		if (row == 1){ 
			for (column = 0; column <= img_length-32; column += 30){ // обрабатываются элементы столбцов до img_length-32 по 30 штук
				if (column == 0){ // для первого столбца (т.к надо дополнить нулем)
					// загружаем первые 3 строки
					//ymm0 = _mm256_loadu_si256((__m256i*)&src_image[0][0]);
					//ymm1 = _mm256_loadu_si256((__m256i*)&src_image[1][0]);
					//ymm2 = _mm256_loadu_si256((__m256i*)&src_image[2][0]);

					// меньше инструкций
					ymm0 = _mm256_maskz_permutexvar_epi8(k, idx, _mm256_loadu_si256((__m256i*)&src_image[0][0]));
					ymm1 = _mm256_maskz_permutexvar_epi8(k, idx, _mm256_loadu_si256((__m256i*)&src_image[1][0]));
					ymm2 = _mm256_maskz_permutexvar_epi8(k, idx, _mm256_loadu_si256((__m256i*)&src_image[2][0]));

					///////////////////////////////////////////////////////////////////////////////////////////////////
					// Будем считать элементы по индексам (0...15,16...31)
		  		// сместили вправо на 1 элемент (все что в регистре влево на 1 байт) и 0й и 16й стали равны 0
		  		//ymm3 = _mm256_slli_si256(ymm0, 1);  
		  		//ymm4 = _mm256_slli_si256(ymm1, 1); 
		  		//ymm5 = _mm256_slli_si256(ymm2, 1);

		  		// вернем потерянный 16й элемент на место (он находится в ro на 15й)
					// занулили все кроме 15го
		  		//ymm0 = _mm256_and_si256(ymm0, mask_prelude);
					// сдвигаем на 31ю т.к если сдвинем на 1 вправо он занулится
		  		//ymm0 = _mm256_permute2f128_si256(ymm0, ymm0, 1);
					// влево на 16ю
		  		//ymm0 = _mm256_srli_si256(ymm0, 15); 
					// складываем и получаем в ymm0 все элементы смещенные на 1 вправо (ну кроме последнего, он пропал)
		  		//ymm0 = _mm256_add_epi16(ymm3, ymm0);

					// аналогично с ymm1 и ymm2
		  		//ymm1 = _mm256_and_si256(ymm1, mask_prelude);
		  		//ymm1 = _mm256_permute2f128_si256(ymm1, ymm1, 1);
		  		//ymm1 = _mm256_srli_si256(ymm1, 15); 
		  		//ymm1 = _mm256_add_epi16(ymm4, ymm1);

		  		//ymm2 = _mm256_and_si256(ymm2, mask_prelude);
		  		//ymm2 = _mm256_permute2f128_si256(ymm2, ymm2, 1);
		  		//ymm2 = _mm256_srli_si256(ymm2, 15); 
		  		//ymm2 = _mm256_add_epi16(ymm5, ymm2);
					/////////////////////////////////////////////////////////////////////////////////////////////////////
		  	}
		    else{ 
				// загружаем 3 строки
					ymm0 = _mm256_loadu_si256((__m256i*)&src_image[0][column-1]);
					ymm1 = _mm256_loadu_si256((__m256i*)&src_image[1][column-1]);
					ymm2 = _mm256_loadu_si256((__m256i*)&src_image[2][column-1]);
				}

				// алгоритм для 0, 4, 8, 12, 16, 20, 24, 28 элемента в 1й строке 
				// эти индексы будем считать одинаковыми для каждых следующих 30 элементов 
				// (да, во второй 30ке будут 32, 36 и тд для исходной строки, но мы будем обозначать 0, 4 и тд) 
				// умножение и сложение с маской 
				ymm3 = _mm256_maddubs_epi16(ymm0, mask_filter_0);
				ymm4 = _mm256_maddubs_epi16(ymm1, mask_filter_1);
				ymm5 = _mm256_maddubs_epi16(ymm2, mask_filter_0);

				// вертикальное сложение ymm3 = (ymm3 + ymm4 + ymm5)
				ymm3 = _mm256_add_epi16(ymm3, ymm4);
				ymm3 = _mm256_add_epi16(ymm3, ymm5);

				// горизонтальное сложение
				// на 0й и 1й позиции нужные суммы (в ymm3), но нам надо получить 1й элемент чтобы сложить с 0м,
				// поэтому мы сдвигаем ymm3 на 2 байта (т.к уже содержатся суммы (16 элементов по 16 бит)) влево  
				// и записываем это в ymm4 (ymm3 не поменялся), теперь в ymm4 на 0й позиции 1й элемент
				ymm4 = _mm256_srli_si256(ymm3, 2);
				// складываем их
				//ymm5 = _mm256_add_epi16(ymm4, ymm3);
				// оставили только нужное с помощью маски (все что не нужно = 0)
				//only_for_even_output = _mm256_and_si256(ymm5, output_mask);
				only_for_even_output = _mm256_maskz_add_epi16(k_odd, ymm4, ymm3);

				// алгоритм для 1, 5, 9, 13, 17, 21, 25, 29 элемента (0, 4, 8 по индексам) 
				ymm3 = _mm256_maddubs_epi16(ymm0 ,mask_filter_0_shift1);
				ymm4 = _mm256_maddubs_epi16(ymm1, mask_filter_1_shift1);
				ymm5 = _mm256_maddubs_epi16(ymm2, mask_filter_0_shift1);
				ymm3 = _mm256_add_epi16(ymm3, ymm4);
				ymm3 = _mm256_add_epi16(ymm3, ymm5);
				ymm4 = _mm256_srli_si256(ymm3, 2);
				//ymm5 = _mm256_add_epi16(ymm4, ymm3);
				//only_for_odd_output = _mm256_and_si256(ymm5, output_mask);
				only_for_odd_output = _mm256_maskz_add_epi16(k_odd, ymm4, ymm3);

				// алгоритм для 2, 6, 10, 14, 18, 22, 26 (теряем элемент при сдвиге по середине) 
				ymm3 = _mm256_maddubs_epi16(ymm0, mask_filter_0_shift2);
				ymm4 = _mm256_maddubs_epi16(ymm1, mask_filter_1_shift2);
				ymm5 = _mm256_maddubs_epi16(ymm2, mask_filter_0_shift2);
				ymm3 = _mm256_add_epi16(ymm3, ymm4); // с 1 по 12
				ymm3 = _mm256_add_epi16(ymm3, ymm5); // с 1 по 14
				ymm4 = _mm256_maskz_permutexvar_epi16(k1, idx2, ymm3); // с 0 по 13
				ymm5 = _mm256_maskz_add_epi16(k_even, ymm4, ymm3);
				//ymm5 = _mm256_add_epi16(ymm4, ymm3);
				//ymm5 = _mm256_and_si256(ymm5, output_mask_sh1);
				//ymm4 = _mm256_srli_si256(ymm3, 2); // после этого сдвига теряется элемент, переписать это с маской
        //ymm5 = _mm256_add_epi16(ymm4, ymm3);
				//m4 = _mm256_and_si256(ymm3, mask3);
				//m4 = _mm256_permute2f128_si256(m4, m4, 1);
				//m4 = _mm256_slli_si256(m4, 14); 
				//ymm5 = _mm256_add_epi16(ymm5, m4); 
				//ymm5 = _mm256_and_si256(ymm5, output_mask_sh1);
				only_for_even_output = _mm256_add_epi16(only_for_even_output, ymm5);

				// алгоритм для 3, 7, 11, 15, 19, 23, 27
				ymm3 = _mm256_maddubs_epi16(ymm0, mask_filter_0_shift3);
				ymm4 = _mm256_maddubs_epi16(ymm1, mask_filter_1_shift3);
				ymm5 = _mm256_maddubs_epi16(ymm2, mask_filter_0_shift3);
				ymm3 = _mm256_add_epi16(ymm3, ymm4);
				ymm3 = _mm256_add_epi16(ymm3, ymm5);
				ymm4 = _mm256_maskz_permutexvar_epi16(k1, idx2, ymm3);
				ymm5 = _mm256_maskz_add_epi16(k_even, ymm4, ymm3); 
				//ymm5 = _mm256_add_epi16(ymm4, ymm3);
				//ymm5 = _mm256_and_si256(ymm5, output_mask_sh1);
				//ymm4 = _mm256_srli_si256(ymm3, 2); // с 1 по 14
				//ymm5 = _mm256_add_epi16(ymm4, ymm3);
				//m4 = _mm256_and_si256(ymm3, mask3);
				//m4 = _mm256_permute2f128_si256(m4, m4, 1);
				//m4 = _mm256_slli_si256(m4, 14); 
				//ymm5 = _mm256_add_epi16(ymm5, m4);
				//ymm5 = _mm256_and_si256(ymm5, output_mask_sh1);
				only_for_odd_output = _mm256_add_epi16(only_for_odd_output, ymm5);

				only_for_even_output = division(division_case, only_for_even_output, f);
				only_for_odd_output = division(division_case, only_for_odd_output, f);

				only_for_odd_output = _mm256_slli_si256(only_for_odd_output, 1);
				only_for_even_output = _mm256_add_epi8(only_for_even_output, only_for_odd_output);
		    _mm256_storeu_si256((__m256i*)&out_image[1][column], only_for_even_output);

		    // row == 0 (меньше на 1 _maddubs и 1 сложение т.к нет строки выше в изображении)
				ymm3 = _mm256_maddubs_epi16(ymm0, mask_filter_1);
				ymm4 = _mm256_maddubs_epi16(ymm1, mask_filter_0);
				ymm3 = _mm256_add_epi16(ymm3, ymm4);
				ymm4 = _mm256_srli_si256(ymm3, 2);
				//ymm5 = _mm256_add_epi16(ymm4, ymm3);
				//only_for_even_output = _mm256_and_si256(ymm5, output_mask);
				only_for_even_output = _mm256_maskz_add_epi16(k_odd, ymm4, ymm3);

				ymm3 = _mm256_maddubs_epi16(ymm0, mask_filter_1_shift1);
				ymm4 = _mm256_maddubs_epi16(ymm1, mask_filter_0_shift1);
				ymm3 = _mm256_add_epi16(ymm3, ymm4);
				ymm4 = _mm256_srli_si256(ymm3, 2);
				//ymm5 = _mm256_add_epi16(ymm4 ,ymm3);
				//only_for_odd_output = _mm256_and_si256(ymm5, output_mask);
			  only_for_odd_output = _mm256_maskz_add_epi16(k_odd, ymm4, ymm3);

				ymm3 = _mm256_maddubs_epi16(ymm0, mask_filter_1_shift2);
				ymm4 = _mm256_maddubs_epi16(ymm1, mask_filter_0_shift2);
				ymm3 = _mm256_add_epi16(ymm3, ymm4);
				ymm4 = _mm256_maskz_permutexvar_epi16(k1, idx2, ymm3); 
				ymm5 = _mm256_maskz_add_epi16(k_even, ymm4, ymm3);
				//ymm5 = _mm256_add_epi16(ymm3, ymm4);
				//ymm5 = _mm256_and_si256(ymm5, output_mask_sh1);
				//ymm4 = _mm256_srli_si256(ymm3, 2);
				//ymm5 = _mm256_add_epi16(ymm4, ymm3);
				//m4 = _mm256_and_si256(ymm3, mask3);
				//m4 = _mm256_permute2f128_si256(m4, m4, 1);
				//m4 = _mm256_slli_si256(m4, 14); 
				//ymm5 = _mm256_add_epi16(ymm5, m4);
				//ymm5 = _mm256_and_si256(ymm5, output_mask_sh1);
				only_for_even_output = _mm256_add_epi16(only_for_even_output, ymm5);

				ymm3 = _mm256_maddubs_epi16(ymm0, mask_filter_1_shift3);
				ymm4 = _mm256_maddubs_epi16(ymm1, mask_filter_0_shift3);
				ymm3 = _mm256_add_epi16(ymm3, ymm4);
				ymm4 = _mm256_maskz_permutexvar_epi16(k1, idx2, ymm3);
				ymm5 = _mm256_maskz_add_epi16(k_even, ymm4, ymm3);
				//ymm5 = _mm256_add_epi16(ymm3, ymm4);
				//ymm5 = _mm256_and_si256(ymm5, output_mask_sh1);
				//ymm4 = _mm256_srli_si256(ymm3, 2);
				//ymm5 = _mm256_add_epi16(ymm4, ymm3);
				//m4 = _mm256_and_si256(ymm3, mask3);
				//m4 = _mm256_permute2f128_si256(m4, m4, 1);
				//m4 = _mm256_slli_si256(m4, 14); 
				//ymm5 = _mm256_add_epi16(ymm5, m4);
				//ymm5 = _mm256_and_si256(ymm5, output_mask_sh1);
				only_for_odd_output = _mm256_add_epi16(only_for_odd_output, ymm5);

				only_for_even_output = division(division_case, only_for_even_output, f);
				only_for_odd_output = division(division_case, only_for_odd_output, f);

				only_for_odd_output = _mm256_slli_si256(only_for_odd_output, 1);
				only_for_even_output = _mm256_add_epi8(only_for_even_output, only_for_odd_output);
				_mm256_storeu_si256((__m256i*)&out_image[0][column], only_for_even_output);
			}
			loop_reminder_3x3_first_values(src_image, out_image, img_length, img_height, row, column, num_of_remaining_elem_in_row, division_case, divisor, filter, mask_filter_0, mask_filter_1, mask_filter_0_shift1, mask_filter_1_shift1, mask_filter_0_shift2, mask_filter_1_shift2, mask_filter_0_shift3, mask_filter_1_shift3, f);
		}
		else if (row == img_height-2){ // out_image[img_height-2:img_height-1][:]
			for (column = 0; column <= img_length-32; column += 30){
				if (column == 0){
					ymm0 = _mm256_maskz_permutexvar_epi8(k, idx, _mm256_loadu_si256((__m256i*)&src_image[img_height-3][0]));
					ymm1 = _mm256_maskz_permutexvar_epi8(k, idx, _mm256_loadu_si256((__m256i*)&src_image[img_height-2][0]));
					ymm2 = _mm256_maskz_permutexvar_epi8(k, idx, _mm256_loadu_si256((__m256i*)&src_image[img_height-1][0]));

					//ymm0 = _mm256_loadu_si256((__m256i*)&src_image[img_height-3][0]);
					//ymm1 = _mm256_loadu_si256((__m256i*)&src_image[img_height-2][0]);
					//ymm2 = _mm256_loadu_si256((__m256i*)&src_image[img_height-1][0]);
		  	  //ymm3 = _mm256_slli_si256(ymm0, 1); 
		  		//ymm4 = _mm256_slli_si256(ymm1, 1); 
		  		//ymm5 = _mm256_slli_si256(ymm2, 1); 
		  		//ymm0 = _mm256_and_si256(ymm0, mask_prelude);
		  		//ymm0 = _mm256_permute2f128_si256(ymm0, ymm0, 1);
		  		//ymm0 = _mm256_srli_si256(ymm0, 15); 
		  		//ymm0 = _mm256_add_epi16(ymm3, ymm0);
		  		//ymm1 = _mm256_and_si256(ymm1, mask_prelude);
		  		//ymm1 = _mm256_permute2f128_si256(ymm1, ymm1, 1);
		  		//ymm1 = _mm256_srli_si256(ymm1, 15); 
		  		//ymm1 = _mm256_add_epi16(ymm4, ymm1);
		  		//ymm2 = _mm256_and_si256(ymm2, mask_prelude);
		  		//ymm2 = _mm256_permute2f128_si256(ymm2, ymm2, 1);
		  		//ymm2 = _mm256_srli_si256(ymm2, 15); 
		  		//ymm2 = _mm256_add_epi16(ymm5, ymm2);
		  	}
				else{
					ymm0 = _mm256_loadu_si256((__m256i*)&src_image[img_height-3][column-1]);
					ymm1 = _mm256_loadu_si256((__m256i*)&src_image[img_height-2][column-1]);
					ymm2 = _mm256_loadu_si256((__m256i*)&src_image[img_height-1][column-1]);
				}

				// row == img_height-2 предпоследняя
				ymm3 = _mm256_maddubs_epi16(ymm0, mask_filter_0);
				ymm4 = _mm256_maddubs_epi16(ymm1, mask_filter_1);
				ymm5 = _mm256_maddubs_epi16(ymm2, mask_filter_0);
				ymm3 = _mm256_add_epi16(ymm3, ymm4);
				ymm3 = _mm256_add_epi16(ymm3, ymm5);
				ymm4 = _mm256_srli_si256(ymm3, 2);
				//ymm5 = _mm256_add_epi16(ymm4, ymm3);
				//only_for_even_output = _mm256_and_si256(ymm5, output_mask);
				only_for_even_output = _mm256_maskz_add_epi16(k_odd, ymm4, ymm3);

				ymm3 = _mm256_maddubs_epi16(ymm0, mask_filter_0_shift1);
				ymm4 = _mm256_maddubs_epi16(ymm1, mask_filter_1_shift1);
				ymm5 = _mm256_maddubs_epi16(ymm2, mask_filter_0_shift1);
				ymm3 = _mm256_add_epi16(ymm3, ymm4);
				ymm3 = _mm256_add_epi16(ymm3, ymm5);
				ymm4 = _mm256_srli_si256(ymm3, 2);
				//ymm5 = _mm256_add_epi16(ymm4, ymm3);
				//only_for_odd_output = _mm256_and_si256(ymm5, output_mask);
				only_for_odd_output = _mm256_maskz_add_epi16(k_odd, ymm4, ymm3);

				ymm3 = _mm256_maddubs_epi16(ymm0, mask_filter_0_shift2);
				ymm4 = _mm256_maddubs_epi16(ymm1, mask_filter_1_shift2);
				ymm5 = _mm256_maddubs_epi16(ymm2, mask_filter_0_shift2);
				ymm3 = _mm256_add_epi16(ymm3, ymm4);
				ymm3 = _mm256_add_epi16(ymm3, ymm5);
				ymm4 = _mm256_maskz_permutexvar_epi16(k1, idx2, ymm3); 
				ymm5 = _mm256_maskz_add_epi16(k_even, ymm4, ymm3);
				//ymm5 = _mm256_add_epi16(ymm4, ymm3);
				//ymm5 = _mm256_and_si256(ymm5, output_mask_sh1);
				//ymm4 = _mm256_srli_si256(ymm3, 2);
				//ymm5 = _mm256_add_epi16(ymm4, ymm3);
				//m4 = _mm256_and_si256(ymm3, mask3);
				//m4 = _mm256_permute2f128_si256(m4, m4, 1);
				//m4 = _mm256_slli_si256(m4, 14); 
				//ymm5 = _mm256_add_epi16(ymm5, m4);
				//ymm5 = _mm256_and_si256(ymm5, output_mask_sh1);
				only_for_even_output = _mm256_add_epi16(only_for_even_output, ymm5);

				ymm3 = _mm256_maddubs_epi16(ymm0, mask_filter_0_shift3);
				ymm4 = _mm256_maddubs_epi16(ymm1, mask_filter_1_shift3);
				ymm5 = _mm256_maddubs_epi16(ymm2, mask_filter_0_shift3);
				ymm3 = _mm256_add_epi16(ymm3, ymm4);
				ymm3 = _mm256_add_epi16(ymm3, ymm5);
				ymm4 = _mm256_maskz_permutexvar_epi16(k1, idx2, ymm3); 
				ymm5 = _mm256_maskz_add_epi16(k_even, ymm4, ymm3);
				//ymm5 = _mm256_add_epi16(ymm4, ymm3);
				//ymm5 = _mm256_and_si256(ymm5, output_mask_sh1);
				//ymm4 = _mm256_srli_si256(ymm3, 2);
				//ymm5 = _mm256_add_epi16(ymm4, ymm3);
				//m4 = _mm256_and_si256(ymm3, mask3);
				//m4 = _mm256_permute2f128_si256(m4, m4, 1);
				//m4 = _mm256_slli_si256(m4, 14); 
				//ymm5 = _mm256_add_epi16(ymm5, m4);
				//ymm5 = _mm256_and_si256(ymm5, output_mask_sh1);
				only_for_odd_output = _mm256_add_epi16(only_for_odd_output, ymm5);

				only_for_even_output = division(division_case, only_for_even_output, f);
				only_for_odd_output = division(division_case, only_for_odd_output, f);

				only_for_odd_output = _mm256_slli_si256(only_for_odd_output, 1);
				only_for_even_output = _mm256_add_epi8(only_for_even_output, only_for_odd_output);
				_mm256_storeu_si256((__m256i*)&out_image[img_height-2][column], only_for_even_output);

				// row == img_height-1 последняя
				ymm4 = _mm256_maddubs_epi16(ymm1, mask_filter_0);
				ymm5 = _mm256_maddubs_epi16(ymm2, mask_filter_1);
				ymm3 = _mm256_add_epi16(ymm4, ymm5);
				ymm4 = _mm256_srli_si256(ymm3, 2);
				//ymm5 = _mm256_add_epi16(ymm4, ymm3);
				//only_for_even_output = _mm256_and_si256(ymm5, output_mask);
				only_for_even_output = _mm256_maskz_add_epi16(k_odd, ymm4, ymm3);

				ymm4 = _mm256_maddubs_epi16(ymm1, mask_filter_0_shift1);
				ymm5 = _mm256_maddubs_epi16(ymm2, mask_filter_1_shift1);
				ymm3 = _mm256_add_epi16(ymm4, ymm5);
				ymm4 = _mm256_srli_si256(ymm3, 2);
				//ymm5 = _mm256_add_epi16(ymm4, ymm3);
				//only_for_odd_output = _mm256_and_si256(ymm5,output_mask);
				only_for_odd_output = _mm256_maskz_add_epi16(k_odd, ymm4, ymm3);

				ymm4 = _mm256_maddubs_epi16(ymm1, mask_filter_0_shift2);
				ymm5 = _mm256_maddubs_epi16(ymm2, mask_filter_1_shift2);
				ymm3 = _mm256_add_epi16(ymm4, ymm5);
				ymm4 = _mm256_maskz_permutexvar_epi16(k1, idx2, ymm3); 
				ymm5 = _mm256_maskz_add_epi16(k_even, ymm4, ymm3);
				//ymm5 = _mm256_add_epi16(ymm3, ymm4);
				//ymm5 = _mm256_and_si256(ymm5, output_mask_sh1);
				//ymm4 = _mm256_srli_si256(ymm3, 2);
				//ymm5 = _mm256_add_epi16(ymm4, ymm3);
				//m4 = _mm256_and_si256(ymm3, mask3);
				//m4 = _mm256_permute2f128_si256(m4, m4, 1);
				//m4 = _mm256_slli_si256(m4, 14); 
				//ymm5 = _mm256_add_epi16(ymm5, m4);
				//ymm5 = _mm256_and_si256(ymm5, output_mask_sh1);
				only_for_even_output = _mm256_add_epi16(only_for_even_output, ymm5);

				ymm4 = _mm256_maddubs_epi16(ymm1, mask_filter_0_shift3);
				ymm5 = _mm256_maddubs_epi16(ymm2, mask_filter_1_shift3);
				ymm3 = _mm256_add_epi16(ymm4, ymm5);
				ymm4 = _mm256_maskz_permutexvar_epi16(k1, idx2, ymm3); 
				ymm5 = _mm256_maskz_add_epi16(k_even, ymm4, ymm3);
				//ymm5 = _mm256_add_epi16(ymm3, ymm4);
				//ymm5 = _mm256_and_si256(ymm5, output_mask_sh1);
				//ymm4 = _mm256_srli_si256(ymm3, 2);
				//ymm5 = _mm256_add_epi16(ymm4, ymm3);
				//m4 = _mm256_and_si256(ymm3, mask3);
				//m4 = _mm256_permute2f128_si256(m4, m4, 1);
				//m4 = _mm256_slli_si256(m4, 14); 
				//ymm5 = _mm256_add_epi16(ymm5, m4);
				//ymm5 = _mm256_and_si256(ymm5, output_mask_sh1);
				only_for_odd_output = _mm256_add_epi16(only_for_odd_output, ymm5);

				only_for_even_output = division(division_case,only_for_even_output, f);
				only_for_odd_output = division(division_case,only_for_odd_output, f);

				only_for_odd_output = _mm256_slli_si256(only_for_odd_output, 1);
				only_for_even_output = _mm256_add_epi8(only_for_even_output, only_for_odd_output);
				_mm256_storeu_si256((__m256i*)&out_image[img_height-1][column], only_for_even_output);
			}
		  loop_reminder_3x3_last_values(src_image, out_image, img_length, img_height, column, num_of_remaining_elem_in_row, division_case, divisor, filter, mask_filter_0, mask_filter_1, mask_filter_0_shift1, mask_filter_1_shift1, mask_filter_0_shift2, mask_filter_1_shift2, mask_filter_0_shift3, mask_filter_1_shift3, f);
		}
		else{ // главный цикл
		  //tbb::parallel_for(tbb::blocked_range<int>(0, img_length - 32, 30), [&](const tbb::blocked_range<int>& column_range){ 
        //for (int column = column_range.begin(); column < column_range.end(); column += 30) {
			for (column = 0; column <= img_length-32; column += 30){
				if (column == 0){
					ymm0 = _mm256_maskz_permutexvar_epi8(k, idx, _mm256_loadu_si256((__m256i*)&src_image[row-1][0]));
					ymm1 = _mm256_maskz_permutexvar_epi8(k, idx, _mm256_loadu_si256((__m256i*)&src_image[row][0]));
					ymm2 = _mm256_maskz_permutexvar_epi8(k, idx, _mm256_loadu_si256((__m256i*)&src_image[row+1][0]));
					/*ymm0 = _mm256_loadu_si256((__m256i*)&src_image[row-1][0]);
					ymm1 = _mm256_loadu_si256((__m256i*)&src_image[row][0]);
					ymm2 = _mm256_loadu_si256((__m256i*)&src_image[row+1][0]);
		  	  ymm3 = _mm256_slli_si256(ymm0, 1); 
		  		ymm4 = _mm256_slli_si256(ymm1, 1);
		  		ymm5 = _mm256_slli_si256(ymm2, 1); 
		  		ymm0 = _mm256_and_si256(ymm0 ,mask_prelude);
		  		ymm0 = _mm256_permute2f128_si256(ymm0, ymm0, 1);
		  		ymm0 = _mm256_srli_si256(ymm0, 15); 
		  		ymm0 = _mm256_add_epi16(ymm3, ymm0);
		  		ymm1 = _mm256_and_si256(ymm1, mask_prelude);
		  	  ymm1 = _mm256_permute2f128_si256(ymm1, ymm1, 1);
		  		ymm1 = _mm256_srli_si256(ymm1, 15); 
		  		ymm1 = _mm256_add_epi16(ymm4, ymm1);
		  		ymm2 = _mm256_and_si256(ymm2, mask_prelude);
		  		ymm2 = _mm256_permute2f128_si256(ymm2, ymm2, 1);
		  		ymm2 = _mm256_srli_si256(ymm2, 15); 
		  		ymm2 = _mm256_add_epi16(ymm5, ymm2);*/
		  	}
				else{
				  ymm0 = _mm256_loadu_si256((__m256i*)&src_image[row-1][column-1]);
				  ymm1 = _mm256_loadu_si256((__m256i*)&src_image[row][column-1]);
				  ymm2 = _mm256_loadu_si256((__m256i*)&src_image[row+1][column-1]);
				}

				ymm3 = _mm256_maddubs_epi16(ymm0, mask_filter_0);
				ymm4 = _mm256_maddubs_epi16(ymm1, mask_filter_1);
				ymm5 = _mm256_maddubs_epi16(ymm2, mask_filter_0);
				ymm3 = _mm256_add_epi16(ymm3, ymm4);
				ymm3 = _mm256_add_epi16(ymm3, ymm5);
				ymm4 = _mm256_srli_si256(ymm3, 2);
				//ymm5 = _mm256_add_epi16(ymm4, ymm3);
				//only_for_even_output = _mm256_and_si256(ymm5, output_mask);
				only_for_even_output = _mm256_maskz_add_epi16(k_odd, ymm4, ymm3);

				ymm3 = _mm256_maddubs_epi16(ymm0, mask_filter_0_shift1);
				ymm4 = _mm256_maddubs_epi16(ymm1, mask_filter_1_shift1);
				ymm5 = _mm256_maddubs_epi16(ymm2, mask_filter_0_shift1);
				ymm3 = _mm256_add_epi16(ymm3, ymm4);
				ymm3 = _mm256_add_epi16(ymm3, ymm5);
				ymm4 = _mm256_srli_si256(ymm3, 2);
				//ymm5 = _mm256_add_epi16(ymm4, ymm3);
				//only_for_odd_output = _mm256_and_si256(ymm5, output_mask);
				only_for_odd_output = _mm256_maskz_add_epi16(k_odd, ymm4, ymm3);

				ymm3 = _mm256_maddubs_epi16(ymm0, mask_filter_0_shift2);
				ymm4 = _mm256_maddubs_epi16(ymm1, mask_filter_1_shift2);
				ymm5 = _mm256_maddubs_epi16(ymm2, mask_filter_0_shift2);
				ymm3 = _mm256_add_epi16(ymm3, ymm4);
				ymm3 = _mm256_add_epi16(ymm3, ymm5);
				ymm4 = _mm256_maskz_permutexvar_epi16(k1, idx2, ymm3); 
				ymm5 = _mm256_maskz_add_epi16(k_even, ymm4, ymm3);
				//ymm5 = _mm256_add_epi16(ymm4, ymm3);
				//ymm5 = _mm256_and_si256(ymm5, output_mask_sh1);
				//ymm4 = _mm256_srli_si256(ymm3, 2);
				//ymm5 = _mm256_add_epi16(ymm4, ymm3);
				//m4 = _mm256_and_si256(ymm3, mask3);
				//m4 = _mm256_permute2f128_si256(m4, m4, 1);
				//m4 = _mm256_slli_si256(m4, 14); 
				//ymm5 = _mm256_add_epi16(ymm5, m4);
				//ymm5 = _mm256_and_si256(ymm5, output_mask_sh1);
				only_for_even_output = _mm256_add_epi16(only_for_even_output, ymm5);

				ymm3 = _mm256_maddubs_epi16(ymm0, mask_filter_0_shift3);
				ymm4 = _mm256_maddubs_epi16(ymm1, mask_filter_1_shift3);
				ymm5 = _mm256_maddubs_epi16(ymm2, mask_filter_0_shift3);
				ymm3 = _mm256_add_epi16(ymm3, ymm4);
				ymm3 = _mm256_add_epi16(ymm3, ymm5);
				ymm4 = _mm256_maskz_permutexvar_epi16(k1, idx2, ymm3); 
				ymm5 = _mm256_maskz_add_epi16(k_even, ymm4, ymm3);
				//ymm5 = _mm256_add_epi16(ymm4, ymm3);
				//ymm5 = _mm256_and_si256(ymm5, output_mask_sh1);
				//ymm4 = _mm256_srli_si256(ymm3, 2);
				//ymm5 = _mm256_add_epi16(ymm4, ymm3);
				//m4 = _mm256_and_si256(ymm3, mask3);
				//m4 = _mm256_permute2f128_si256(m4, m4, 1);
				//m4 = _mm256_slli_si256(m4, 14); 
				//ymm5 = _mm256_add_epi16(ymm5, m4);
				//ymm5 = _mm256_and_si256(ymm5, output_mask_sh1);
				only_for_odd_output = _mm256_add_epi16(only_for_odd_output, ymm5);

				only_for_even_output = division(division_case, only_for_even_output, f);
				only_for_odd_output = division(division_case, only_for_odd_output, f);

				only_for_odd_output = _mm256_slli_si256(only_for_odd_output, 1);
				only_for_even_output = _mm256_add_epi8(only_for_even_output, only_for_odd_output);
				_mm256_storeu_si256((__m256i*)&out_image[row][column], only_for_even_output);
			}
			loop_reminder_3x3(src_image, out_image, img_length, img_height, row, column, num_of_remaining_elem_in_row, division_case, divisor, filter, mask_filter_0, mask_filter_1, mask_filter_0_shift1, mask_filter_1_shift1, mask_filter_0_shift2, mask_filter_1_shift2, mask_filter_0_shift3, mask_filter_1_shift3, f);
		//});
		}
	}
	std::cout << "Time: " << (tbb::tick_count::now() - t0).seconds() << " seconds" << std::endl;
} 

int loop_reminder_3x3(unsigned char **src_image, unsigned char **out_image, unsigned int img_length, unsigned int img_height, unsigned int row, unsigned int column, unsigned int num_of_remaining_elem_in_row, unsigned int division_case, unsigned short int divisor, signed char **filter, __m256i mask_filter_0, __m256i mask_filter_1, __m256i mask_filter_0_shift1, __m256i mask_filter_1_shift1, __m256i mask_filter_0_shift2, __m256i mask_filter_1_shift2, __m256i mask_filter_0_shift3, __m256i mask_filter_1_shift3, __m256i f){
	__m256i ymm0, ymm1, ymm2, ymm3, ymm4, ymm5, m4, only_for_even_output, only_for_odd_output;
  //__m256i mask3           = _mm256_set_epi16(0, 0, 0, 0, 0, 0, 0, 65535, 0, 0, 0, 0, 0, 0, 0, 0);
	//__m256i output_mask     = _mm256_set_epi16(0, 65535, 0, 65535, 0, 65535, 0, 65535, 0, 65535, 0, 65535, 0, 65535, 0, 65535);
	//__m256i output_mask_sh1 = _mm256_set_epi16(0, 0, 65535, 0, 65535, 0, 65535, 0, 65535, 0, 65535, 0, 65535, 0, 65535, 0);
  __m256i reminder_mask1;


	///////////////////////////////////////////////////////////////////////////////////////////
	// это для эффективной обработки хвоста
	__mmask32 mask_table_for_rem[] = {0x0, 0x1, 0x3, 0x7, 
																		0xF, 0x1F, 0x3F, 0x7F, 
																		0xFF, 0x1FF, 0x3FF, 0x7FF, 
																		0xFFF, 0x1FFF, 0x3FFF, 0x7FFF,
																	  0xFFFF, 0x1FFFF, 0x3FFFF, 0x7FFFF,
																		0xFFFFF, 0x1FFFFF, 0x3FFFFF, 0x7FFFFF,
																		0xFFFFFF, 0x1FFFFFF, 0x3FFFFFF, 0x7FFFFFF,
																		0xFFFFFFF, 0x1FFFFFFF, 0x3FFFFFFF, 0x7FFFFFFF,
	};

	__mmask32 msk_for_rem = mask_table_for_rem[img_length-((((img_length-32)/30)*30)+30)];
	///////////////////////////////////////////////////////////////////////////////////////////

	__mmask16 k_odd = 0x5555; 
	__mmask16 k_even = 0x2AAA; 
	__mmask64 k1 = 0x7FFF; 
	__m256i idx2 = _mm256_set_epi16(0, 0, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1);

	if (num_of_remaining_elem_in_row == 0)
		return 0; 
	reminder_mask1 = _mm256_load_si256((__m256i*)&mask_for_remaining[num_of_remaining_elem_in_row-1][0]);

	ymm0 = _mm256_loadu_si256((__m256i*)&src_image[row-1][column-1]);
	ymm1 = _mm256_loadu_si256((__m256i*)&src_image[row][column-1]);
	ymm2 = _mm256_loadu_si256((__m256i*)&src_image[row+1][column-1]);
	ymm0 = _mm256_and_si256(ymm0, reminder_mask1);
	ymm1 = _mm256_and_si256(ymm1, reminder_mask1);
	ymm2 = _mm256_and_si256(ymm2, reminder_mask1);

	ymm3 = _mm256_maddubs_epi16(ymm0, mask_filter_0);
	ymm4 = _mm256_maddubs_epi16(ymm1, mask_filter_1);
	ymm5 = _mm256_maddubs_epi16(ymm2, mask_filter_0);
	ymm3 = _mm256_add_epi16(ymm3, ymm4);
	ymm3 = _mm256_add_epi16(ymm3, ymm5);
	ymm4 = _mm256_srli_si256(ymm3, 2);
	//ymm5 = _mm256_add_epi16(ymm4, ymm3);
	//only_for_even_output = _mm256_and_si256(ymm5, output_mask);
	only_for_even_output = _mm256_maskz_add_epi16(k_odd, ymm4, ymm3);

	ymm3 = _mm256_maddubs_epi16(ymm0, mask_filter_0_shift1);
	ymm4 = _mm256_maddubs_epi16(ymm1, mask_filter_1_shift1);
	ymm5 = _mm256_maddubs_epi16(ymm2, mask_filter_0_shift1);
	ymm3 = _mm256_add_epi16(ymm3, ymm4);
	ymm3 = _mm256_add_epi16(ymm3, ymm5);
	ymm4 = _mm256_srli_si256(ymm3, 2);
	//ymm5 = _mm256_add_epi16(ymm4, ymm3);
	//only_for_odd_output = _mm256_and_si256(ymm5, output_mask);
	only_for_odd_output = _mm256_maskz_add_epi16(k_odd, ymm4, ymm3);

	ymm3 = _mm256_maddubs_epi16(ymm0, mask_filter_0_shift2);
	ymm4 = _mm256_maddubs_epi16(ymm1, mask_filter_1_shift2);
	ymm5 = _mm256_maddubs_epi16(ymm2, mask_filter_0_shift2);
	ymm3 = _mm256_add_epi16(ymm3, ymm4);
	ymm3 = _mm256_add_epi16(ymm3, ymm5);
	ymm4 = _mm256_maskz_permutexvar_epi16(k1, idx2, ymm3);
	ymm5 = _mm256_maskz_add_epi16(k_even, ymm4, ymm3);
	//ymm5 = _mm256_add_epi16(ymm4, ymm3);
	//ymm5 = _mm256_and_si256(ymm5, output_mask_sh1);
	//ymm4 = _mm256_srli_si256(ymm3, 2);
	//ymm5 = _mm256_add_epi16(ymm4, ymm3);
	//m4 = _mm256_and_si256(ymm3, mask3);
	//m4 = _mm256_permute2f128_si256(m4, m4, 1);
	//m4 = _mm256_slli_si256(m4, 14); 
	//ymm5 = _mm256_add_epi16(ymm5, m4);
	//ymm5 = _mm256_and_si256(ymm5, output_mask_sh1);
	only_for_even_output = _mm256_add_epi16(only_for_even_output, ymm5);

	ymm3 = _mm256_maddubs_epi16(ymm0, mask_filter_0_shift3);
	ymm4 = _mm256_maddubs_epi16(ymm1, mask_filter_1_shift3);
	ymm5 = _mm256_maddubs_epi16(ymm2, mask_filter_0_shift3);
	ymm3 = _mm256_add_epi16(ymm3, ymm4);
	ymm3 = _mm256_add_epi16(ymm3, ymm5);
	ymm4 = _mm256_maskz_permutexvar_epi16(k1, idx2, ymm3); 
	ymm5 = _mm256_maskz_add_epi16(k_even, ymm4, ymm3);
	//ymm5 = _mm256_add_epi16(ymm4, ymm3);
	//ymm5 = _mm256_and_si256(ymm5, output_mask_sh1);
	//ymm4=_mm256_srli_si256(ymm3, 2);
	//ymm5=_mm256_add_epi16(ymm4, ymm3);
	//m4=_mm256_and_si256(ymm3, mask3);
	//m4=_mm256_permute2f128_si256(m4, m4, 1);
	//m4=_mm256_slli_si256(m4, 14); 
	//ymm5=_mm256_add_epi16(ymm5, m4);
	//ymm5 = _mm256_and_si256(ymm5, output_mask_sh1);
	only_for_odd_output = _mm256_add_epi16(only_for_odd_output, ymm5);

	only_for_even_output = division(division_case, only_for_even_output, f);
	only_for_odd_output = division(division_case, only_for_odd_output, f);

	only_for_odd_output = _mm256_slli_si256(only_for_odd_output, 1);
	only_for_even_output = _mm256_add_epi8(only_for_even_output, only_for_odd_output);

	/////////////////////////////////////////////////////////////////////////////////////////////////////////////
	if (num_of_remaining_elem_in_row == 30){
		_mm256_mask_storeu_epi8(&out_image[row][column], msk_for_rem, only_for_even_output);
		{int newPixel = 0;
			newPixel += src_image[row-1][img_length-1-1] * filter[0][0];
			newPixel += src_image[row-1][img_length-1] * filter[0][1];
			newPixel += src_image[row][img_length-1-1] * filter[1][0];
			newPixel += src_image[row][img_length-1] * filter[1][1];
			newPixel += src_image[row+1][img_length-1-1] * filter[2][0];
			newPixel += src_image[row+1][img_length-1] * filter[2][1];
			out_image[row][img_length-1] = (unsigned char)(newPixel/divisor);
		}
	}
	else if (num_of_remaining_elem_in_row == 31){
		_mm256_mask_storeu_epi8(&out_image[row][column], msk_for_rem, only_for_even_output);
		{int newPixel = 0;
			newPixel += src_image[row-1][img_length-2-1] * filter[0][0];
			newPixel += src_image[row-1][img_length-2] * filter[0][1];
			newPixel += src_image[row-1][img_length-2+1] * filter[0][2];
			newPixel += src_image[row][img_length-2-1] * filter[1][0];
			newPixel += src_image[row][img_length-2] * filter[1][1];
			newPixel += src_image[row][img_length-2+1] * filter[1][2];
			newPixel += src_image[row+1][img_length-2-1] * filter[2][0];
			newPixel += src_image[row+1][img_length-2] * filter[2][1];
			newPixel += src_image[row+1][img_length-2+1] * filter[2][2];
			out_image[row][img_length-2] = (unsigned char)(newPixel/divisor);

			newPixel = 0;
			newPixel += src_image[row-1][img_length-1-1] * filter[0][0];
			newPixel += src_image[row-1][img_length-1] * filter[0][1];
			newPixel += src_image[row][img_length-1-1] * filter[1][0];
			newPixel += src_image[row][img_length-1] * filter[1][1];
			newPixel += src_image[row+1][img_length-1-1] * filter[2][0];
			newPixel += src_image[row+1][img_length-1] * filter[2][1];
			out_image[row][img_length-1] = (unsigned char)(newPixel/divisor);
			}
	}
	else{
		_mm256_mask_storeu_epi8(&out_image[row][column], msk_for_rem, only_for_even_output);
	}
	return 0;
}

int loop_reminder_3x3_first_values(unsigned char** src_image, unsigned char** out_image, unsigned int img_length, unsigned int img_height, unsigned int row, unsigned int column, unsigned int num_of_remaining_elem_in_row, unsigned int division_case, unsigned short int divisor, signed char **filter, __m256i mask_filter_0, __m256i mask_filter_1, __m256i mask_filter_0_shift1, __m256i mask_filter_1_shift1, __m256i mask_filter_0_shift2, __m256i mask_filter_1_shift2,  __m256i mask_filter_0_shift3, __m256i mask_filter_1_shift3, __m256i f){
	__m256i ymm0, ymm1, ymm2, ymm3, ymm4, ymm5, m4, only_for_even_output, only_for_odd_output, output_row0, output_row1;
	int newPixel;
	unsigned int i;
	//__m256i mask3           = _mm256_set_epi16(0, 0, 0, 0, 0, 0, 0, 65535, 0, 0, 0, 0, 0, 0, 0, 0);
	//__m256i output_mask     = _mm256_set_epi16(0, 65535, 0, 65535, 0, 65535, 0, 65535, 0, 65535, 0, 65535, 0, 65535, 0, 65535);
	//__m256i output_mask_sh1 = _mm256_set_epi16(0, 0, 65535, 0, 65535, 0, 65535, 0, 65535, 0, 65535, 0, 65535, 0, 65535, 0);


		///////////////////////////////////////////////////////////////////////////////////////////
	// это для эффективной обработки хвоста
	__mmask32 mask_table_for_rem[] = {0x0, 0x1, 0x3, 0x7, 
																		0xF, 0x1F, 0x3F, 0x7F, 
																		0xFF, 0x1FF, 0x3FF, 0x7FF, 
																		0xFFF, 0x1FFF, 0x3FFF, 0x7FFF,
																	  0xFFFF, 0x1FFFF, 0x3FFFF, 0x7FFFF,
																		0xFFFFF, 0x1FFFFF, 0x3FFFFF, 0x7FFFFF,
																		0xFFFFFF, 0x1FFFFFF, 0x3FFFFFF, 0x7FFFFFF,
																		0xFFFFFFF, 0x1FFFFFFF, 0x3FFFFFFF, 0x7FFFFFFF,

	};

	__mmask32 msk_for_rem = mask_table_for_rem[img_length-((((img_length-32)/30)*30)+30)];
	///////////////////////////////////////////////////////////////////////////////////////////


	__mmask16 k_odd = 0x5555; 
	__mmask16 k_even = 0x2AAA;
	__mmask64 k1 = 0x7FFF; 
	__m256i idx2 = _mm256_set_epi16(0, 0, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1);

	if (num_of_remaining_elem_in_row == 0)
		return 0; 
	__m256i reminder_mask1 = _mm256_load_si256((__m256i*)&mask_for_remaining[num_of_remaining_elem_in_row-1][0]);

	ymm0 = _mm256_loadu_si256((__m256i*)&src_image[0][column-1]);
	ymm1 = _mm256_loadu_si256((__m256i*)&src_image[1][column-1]);
	ymm2 = _mm256_loadu_si256((__m256i*)&src_image[2][column-1]);
  // описанные выше операции загрузки выполняются за пределами массива
  // добавляется один дополнительный ноль в конце, чтобы вычислить column = img_height-1
	// Последнее значение column равно img_height-1, а не img_height
	// эти элементы заполняются нулями 
	ymm0 = _mm256_and_si256(ymm0, reminder_mask1);
	ymm1 = _mm256_and_si256(ymm1, reminder_mask1);
	ymm2 = _mm256_and_si256(ymm2, reminder_mask1);

	ymm3 = _mm256_maddubs_epi16(ymm0, mask_filter_0);
	ymm4 = _mm256_maddubs_epi16(ymm1, mask_filter_1);
	ymm5 = _mm256_maddubs_epi16(ymm2, mask_filter_0);
	ymm3 = _mm256_add_epi16(ymm3, ymm4);
	ymm3 = _mm256_add_epi16(ymm3, ymm5);
	ymm4 = _mm256_srli_si256(ymm3, 2);
	//ymm5 = _mm256_add_epi16(ymm4, ymm3);
	//only_for_even_output = _mm256_and_si256(ymm5, output_mask);
	only_for_even_output = _mm256_maskz_add_epi16(k_odd, ymm4, ymm3);

	ymm3 = _mm256_maddubs_epi16(ymm0, mask_filter_0_shift1);
	ymm4 = _mm256_maddubs_epi16(ymm1, mask_filter_1_shift1);
	ymm5 = _mm256_maddubs_epi16(ymm2, mask_filter_0_shift1);
	ymm3 = _mm256_add_epi16(ymm3, ymm4);
	ymm3 = _mm256_add_epi16(ymm3, ymm5);
	ymm4 = _mm256_srli_si256(ymm3, 2);
	//ymm5 = _mm256_add_epi16(ymm4, ymm3);
	//only_for_odd_output = _mm256_and_si256(ymm5, output_mask);
	only_for_odd_output = _mm256_maskz_add_epi16(k_odd, ymm4, ymm3);

	ymm3 = _mm256_maddubs_epi16(ymm0 ,mask_filter_0_shift2);
	ymm4 = _mm256_maddubs_epi16(ymm1, mask_filter_1_shift2);
	ymm5 = _mm256_maddubs_epi16(ymm2, mask_filter_0_shift2);
	ymm3 = _mm256_add_epi16(ymm3, ymm4);
	ymm3 = _mm256_add_epi16(ymm3, ymm5);
	ymm4 = _mm256_maskz_permutexvar_epi16(k1, idx2, ymm3); 
	ymm5 = _mm256_maskz_add_epi16(k_even, ymm4, ymm3);
	//ymm5 = _mm256_add_epi16(ymm4, ymm3);
	//ymm5 = _mm256_and_si256(ymm5, output_mask_sh1);
	//ymm4 = _mm256_srli_si256(ymm3, 2);
	//ymm5 = _mm256_add_epi16(ymm4, ymm3);
	//m4 = _mm256_and_si256(ymm3, mask3);
	//m4 = _mm256_permute2f128_si256(m4, m4, 1);
	//m4 = _mm256_slli_si256(m4, 14);
	//ymm5 = _mm256_add_epi16(ymm5, m4);
	//ymm5 = _mm256_and_si256(ymm5, output_mask_sh1);
	only_for_even_output = _mm256_add_epi16(only_for_even_output, ymm5);

	ymm3 = _mm256_maddubs_epi16(ymm0 ,mask_filter_0_shift3);
	ymm4 = _mm256_maddubs_epi16(ymm1, mask_filter_1_shift3);
	ymm5 = _mm256_maddubs_epi16(ymm2, mask_filter_0_shift3);
	ymm3 = _mm256_add_epi16(ymm3, ymm4);
	ymm3 = _mm256_add_epi16(ymm3, ymm5);
	ymm4 = _mm256_maskz_permutexvar_epi16(k1, idx2, ymm3); 
	ymm5 = _mm256_maskz_add_epi16(k_even, ymm4, ymm3);
	//ymm5 = _mm256_add_epi16(ymm4, ymm3);
	//ymm5 = _mm256_and_si256(ymm5, output_mask_sh1);
	//ymm4 = _mm256_srli_si256(ymm3, 2);
	//ymm5 = _mm256_add_epi16(ymm4, ymm3);
	//m4 = _mm256_and_si256(ymm3, mask3);
	//m4 = _mm256_permute2f128_si256(m4, m4, 1);
	//m4 = _mm256_slli_si256(m4, 14);
	//ymm5 = _mm256_add_epi16(ymm5, m4);
	//ymm5 = _mm256_and_si256(ymm5, output_mask_sh1);
	only_for_odd_output = _mm256_add_epi16(only_for_odd_output, ymm5);

	only_for_even_output = division(division_case, only_for_even_output, f);
	only_for_odd_output = division(division_case, only_for_odd_output, f);

	only_for_odd_output = _mm256_slli_si256(only_for_odd_output, 1);
	only_for_even_output = _mm256_add_epi8(only_for_even_output, only_for_odd_output);
	output_row1 = only_for_even_output;

	// row == 0
	ymm3 = _mm256_maddubs_epi16(ymm0, mask_filter_1);
	ymm4 = _mm256_maddubs_epi16(ymm1, mask_filter_0);
	ymm3 = _mm256_add_epi16(ymm3, ymm4);
	ymm4 = _mm256_srli_si256(ymm3, 2);
	//ymm5 = _mm256_add_epi16(ymm4, ymm3);
	//only_for_even_output = _mm256_and_si256(ymm5, output_mask);
	only_for_even_output = _mm256_maskz_add_epi16(k_odd, ymm4, ymm3);

	ymm3 = _mm256_maddubs_epi16(ymm0, mask_filter_1_shift1);
	ymm4 = _mm256_maddubs_epi16(ymm1, mask_filter_0_shift1);
	ymm3 = _mm256_add_epi16(ymm3, ymm4);
	ymm4 = _mm256_srli_si256(ymm3, 2);
	//ymm5 = _mm256_add_epi16(ymm4, ymm3);
	//only_for_odd_output = _mm256_and_si256(ymm5, output_mask);
	only_for_odd_output = _mm256_maskz_add_epi16(k_odd, ymm4, ymm3);

	ymm3 = _mm256_maddubs_epi16(ymm0, mask_filter_1_shift2);
	ymm4 = _mm256_maddubs_epi16(ymm1, mask_filter_0_shift2);
	ymm3 = _mm256_add_epi16(ymm3, ymm4);
	ymm4 = _mm256_maskz_permutexvar_epi16(k1, idx2, ymm3); 
	ymm5 = _mm256_maskz_add_epi16(k_even, ymm4, ymm3);
	//ymm5 = _mm256_add_epi16(ymm3, ymm4);
	//ymm5 = _mm256_and_si256(ymm5, output_mask_sh1);
	//ymm4 = _mm256_srli_si256(ymm3, 2);
	//ymm5 = _mm256_add_epi16(ymm4, ymm3);
	//m4 = _mm256_and_si256(ymm3, mask3);
	//m4 = _mm256_permute2f128_si256(m4, m4, 1);
	//m4 = _mm256_slli_si256(m4, 14);
	//ymm5 = _mm256_add_epi16(ymm5, m4);
	//ymm5 = _mm256_and_si256(ymm5, output_mask_sh1);
	only_for_even_output = _mm256_add_epi16(only_for_even_output, ymm5);

	ymm3 = _mm256_maddubs_epi16(ymm0, mask_filter_1_shift3);
	ymm4 = _mm256_maddubs_epi16(ymm1, mask_filter_0_shift3);
	ymm3 = _mm256_add_epi16(ymm3, ymm4);
	ymm4 = _mm256_maskz_permutexvar_epi16(k1, idx2, ymm3); 
	ymm5 = _mm256_maskz_add_epi16(k_even, ymm4, ymm3);
	//ymm5 = _mm256_add_epi16(ymm3, ymm4);
	//ymm5 = _mm256_and_si256(ymm5, output_mask_sh1);
	//ymm4 = _mm256_srli_si256(ymm3, 2);
	//ymm5 = _mm256_add_epi16(ymm4, ymm3);
	//m4 = _mm256_and_si256(ymm3, mask3);
	//m4 = _mm256_permute2f128_si256(m4, m4, 1);
	//m4 = _mm256_slli_si256(m4, 14);
	//ymm5 = _mm256_add_epi16(ymm5, m4);
	//ymm5 = _mm256_and_si256(ymm5, output_mask_sh1);
	only_for_odd_output = _mm256_add_epi16(only_for_odd_output, ymm5);

	only_for_even_output = division(division_case, only_for_even_output, f);
	only_for_odd_output = division(division_case, only_for_odd_output, f);

	only_for_odd_output = _mm256_slli_si256(only_for_odd_output, 1);
	only_for_even_output = _mm256_add_epi8(only_for_even_output, only_for_odd_output);
	output_row0 = only_for_even_output;


	/////////////////////////////////////////////////////////////////////////////////////////////////////////////
	if (num_of_remaining_elem_in_row == 30){
		_mm256_mask_storeu_epi8(&out_image[1][column], msk_for_rem, output_row1);
		_mm256_mask_storeu_epi8(&out_image[0][column], msk_for_rem, output_row0);
		{int newPixel = 0;
			newPixel += src_image[row-1][img_length-1-1] * filter[0][0];
			newPixel += src_image[row-1][img_length-1] * filter[0][1];
			newPixel += src_image[row][img_length-1-1] * filter[1][0];
			newPixel += src_image[row][img_length-1] * filter[1][1];
			newPixel += src_image[row+1][img_length-1-1] * filter[2][0];
			newPixel += src_image[row+1][img_length-1] * filter[2][1];
			out_image[row][img_length-1] = (unsigned char)(newPixel/divisor);
			}
	}
	else if (num_of_remaining_elem_in_row == 31){
		_mm256_mask_storeu_epi8(&out_image[1][column], msk_for_rem, output_row1);
		_mm256_mask_storeu_epi8(&out_image[0][column], msk_for_rem, output_row0);
		{int newPixel = 0;
			newPixel += src_image[row-1][img_length-2-1] * filter[0][0];
			newPixel += src_image[row-1][img_length-2] * filter[0][1];
			newPixel += src_image[row-1][img_length-2+1] * filter[0][2];
			newPixel += src_image[row][img_length-2-1] * filter[1][0];
			newPixel += src_image[row][img_length-2] * filter[1][1];
			newPixel += src_image[row][img_length-2+1] * filter[1][2];
			newPixel += src_image[row+1][img_length-2-1] * filter[2][0];
			newPixel += src_image[row+1][img_length-2] * filter[2][1];
			newPixel += src_image[row+1][img_length-2+1] * filter[2][2];
			out_image[row][img_length-2] = (unsigned char)(newPixel/divisor);

			newPixel = 0;
			newPixel += src_image[row-1][img_length-1-1] * filter[0][0];
			newPixel += src_image[row-1][img_length-1] * filter[0][1];
			newPixel += src_image[row][img_length-1-1] * filter[1][0];
			newPixel += src_image[row][img_length-1] * filter[1][1];
			newPixel += src_image[row+1][img_length-1-1] * filter[2][0];
			newPixel += src_image[row+1][img_length-1] * filter[2][1];
			out_image[row][img_length-1] = (unsigned char)(newPixel/divisor);
			}
	}
	else{
		_mm256_mask_storeu_epi8(&out_image[1][column], msk_for_rem, output_row1);
		_mm256_mask_storeu_epi8(&out_image[0][column], msk_for_rem, output_row0);
	}
	return 0;
}

int loop_reminder_3x3_last_values(unsigned char** src_image, unsigned char** out_image, unsigned int img_length, unsigned int img_height, unsigned int column, unsigned int num_of_remaining_elem_in_row, unsigned int division_case, unsigned short int divisor, signed char **filter, __m256i mask_filter_0, __m256i mask_filter_1, __m256i mask_filter_0_shift1, __m256i mask_filter_1_shift1, __m256i mask_filter_0_shift2, __m256i mask_filter_1_shift2, __m256i mask_filter_0_shift3, __m256i mask_filter_1_shift3, __m256i f){
  __m256i ymm0, ymm1, ymm2, ymm3, ymm4, ymm5, m4, only_for_even_output, only_for_odd_output, output_row0, output_row1;
  int newPixel;
  unsigned int i;
  //__m256i mask3           = _mm256_set_epi16(0, 0, 0, 0, 0, 0, 0, 65535, 0, 0, 0, 0, 0, 0, 0, 0);
  //__m256i output_mask     = _mm256_set_epi16(0, 65535, 0, 65535, 0, 65535, 0, 65535, 0, 65535, 0, 65535, 0, 65535, 0, 65535);
  //__m256i output_mask_sh1 = _mm256_set_epi16(0, 0, 65535, 0, 65535, 0, 65535, 0, 65535, 0, 65535, 0, 65535, 0, 65535, 0);
  
	///////////////////////////////////////////////////////////////////////////////////////////
	// это для эффективной обработки хвоста
	__mmask32 mask_table_for_rem[] = {0x0, 0x1, 0x3, 0x7, 
																		0xF, 0x1F, 0x3F, 0x7F, 
																		0xFF, 0x1FF, 0x3FF, 0x7FF, 
																		0xFFF, 0x1FFF, 0x3FFF, 0x7FFF,
																	  0xFFFF, 0x1FFFF, 0x3FFFF, 0x7FFFF,
																		0xFFFFF, 0x1FFFFF, 0x3FFFFF, 0x7FFFFF,
																		0xFFFFFF, 0x1FFFFFF, 0x3FFFFFF, 0x7FFFFFF,
																		0xFFFFFFF, 0x1FFFFFFF, 0x3FFFFFFF, 0x7FFFFFFF,
	};

	__mmask32 msk_for_rem = mask_table_for_rem[img_length-((((img_length-32)/30)*30)+30)];
	///////////////////////////////////////////////////////////////////////////////////////////

	__mmask16 k_odd = 0x5555; 
	__mmask16 k_even = 0x2AAA;
	__mmask64 k1 = 0x7FFF; 
	__m256i idx2 = _mm256_set_epi16(0, 0, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1);

  if (num_of_remaining_elem_in_row == 0)
	  return 0; 
  __m256i reminder_mask1 = _mm256_load_si256((__m256i*)&mask_for_remaining[num_of_remaining_elem_in_row-1][0]);

	ymm0 =_mm256_loadu_si256((__m256i*)&src_image[img_height-3][column-1]);
	ymm1 = _mm256_loadu_si256((__m256i*)&src_image[img_height-2][column-1]);
	ymm2 = _mm256_loadu_si256((__m256i*)&src_image[img_height-1][column-1]);

	ymm0 = _mm256_and_si256(ymm0, reminder_mask1);
	ymm1 = _mm256_and_si256(ymm1, reminder_mask1);
	ymm2 = _mm256_and_si256(ymm2, reminder_mask1);

	ymm3 = _mm256_maddubs_epi16(ymm0, mask_filter_0);
	ymm4 = _mm256_maddubs_epi16(ymm1, mask_filter_1);
	ymm5 = _mm256_maddubs_epi16(ymm2, mask_filter_0);
	ymm3 = _mm256_add_epi16(ymm3, ymm4);
	ymm3 = _mm256_add_epi16(ymm3, ymm5);
	ymm4 = _mm256_srli_si256(ymm3, 2);
	//ymm5 = _mm256_add_epi16(ymm4, ymm3);
  //only_for_even_output = _mm256_and_si256(ymm5, output_mask);
	only_for_even_output = _mm256_maskz_add_epi16(k_odd, ymm4, ymm3);

	ymm3 = _mm256_maddubs_epi16(ymm0, mask_filter_0_shift1);
	ymm4 = _mm256_maddubs_epi16(ymm1, mask_filter_1_shift1);
	ymm5 = _mm256_maddubs_epi16(ymm2, mask_filter_0_shift1);
	ymm3 = _mm256_add_epi16(ymm3, ymm4);
	ymm3 = _mm256_add_epi16(ymm3, ymm5);
	ymm4 = _mm256_srli_si256(ymm3, 2);
	//ymm5 = _mm256_add_epi16(ymm4, ymm3);
	//only_for_odd_output = _mm256_and_si256(ymm5, output_mask);
	only_for_odd_output = _mm256_maskz_add_epi16(k_odd, ymm4, ymm3);

	ymm3 = _mm256_maddubs_epi16(ymm0, mask_filter_0_shift2);
	ymm4 = _mm256_maddubs_epi16(ymm1, mask_filter_1_shift2);
	ymm5 = _mm256_maddubs_epi16(ymm2, mask_filter_0_shift2);
	ymm3 = _mm256_add_epi16(ymm3, ymm4);
	ymm3 = _mm256_add_epi16(ymm3, ymm5);
	ymm4 = _mm256_maskz_permutexvar_epi16(k1, idx2, ymm3); 
	ymm5 = _mm256_maskz_add_epi16(k_even, ymm4, ymm3);
	//ymm5 = _mm256_add_epi16(ymm4, ymm3);
	//ymm5 = _mm256_and_si256(ymm5, output_mask_sh1);
	//ymm4 = _mm256_srli_si256(ymm3, 2);
	//ymm5 = _mm256_add_epi16(ymm4, ymm3);
	//m4 = _mm256_and_si256(ymm3, mask3);
	//m4 = _mm256_permute2f128_si256(m4, m4, 1);
	//m4 = _mm256_slli_si256(m4, 14);
	//ymm5 = _mm256_add_epi16(ymm5, m4);
	//ymm5 = _mm256_and_si256(ymm5, output_mask_sh1);
	only_for_even_output = _mm256_add_epi16(only_for_even_output, ymm5);

	ymm3 = _mm256_maddubs_epi16(ymm0, mask_filter_0_shift3);
	ymm4 = _mm256_maddubs_epi16(ymm1, mask_filter_1_shift3);
	ymm5 = _mm256_maddubs_epi16(ymm2, mask_filter_0_shift3);
	ymm3 = _mm256_add_epi16(ymm3, ymm4);
	ymm3 = _mm256_add_epi16(ymm3, ymm5);
	ymm4 = _mm256_maskz_permutexvar_epi16(k1, idx2, ymm3); 
	ymm5 = _mm256_maskz_add_epi16(k_even, ymm4, ymm3);
	//ymm5 = _mm256_add_epi16(ymm4, ymm3);
	//ymm5 = _mm256_and_si256(ymm5, output_mask_sh1);
	//ymm4 = _mm256_srli_si256(ymm3, 2);
	//ymm5 = _mm256_add_epi16(ymm4, ymm3);
	//m4 = _mm256_and_si256(ymm3, mask3);
	//m4 = _mm256_permute2f128_si256(m4, m4, 1);
	//m4 = _mm256_slli_si256(m4, 14);
	//ymm5 = _mm256_add_epi16(ymm5, m4);
	//ymm5 = _mm256_and_si256(ymm5, output_mask_sh1);
	only_for_odd_output = _mm256_add_epi16(only_for_odd_output, ymm5);

	only_for_even_output = division(division_case, only_for_even_output, f);
	only_for_odd_output = division(division_case, only_for_odd_output, f);

	only_for_odd_output = _mm256_slli_si256(only_for_odd_output, 1);
	only_for_even_output = _mm256_add_epi8(only_for_even_output, only_for_odd_output);
	output_row1 = only_for_even_output;

	// row == img_height-1
	ymm4 = _mm256_maddubs_epi16(ymm1, mask_filter_0);
	ymm5 = _mm256_maddubs_epi16(ymm2, mask_filter_1);
	ymm3 = _mm256_add_epi16(ymm4, ymm5);
	ymm4 = _mm256_srli_si256(ymm3, 2);
	//ymm5 = _mm256_add_epi16(ymm4, ymm3);
	//only_for_even_output = _mm256_and_si256(ymm5, output_mask);
	only_for_even_output = _mm256_maskz_add_epi16(k_odd, ymm4, ymm3);

	ymm4 = _mm256_maddubs_epi16(ymm1, mask_filter_0_shift1);
	ymm5 = _mm256_maddubs_epi16(ymm2, mask_filter_1_shift1);
	ymm3 = _mm256_add_epi16(ymm4, ymm5);
	ymm4 = _mm256_srli_si256(ymm3, 2);
	//ymm5 = _mm256_add_epi16(ymm4, ymm3);
	//only_for_odd_output = _mm256_and_si256(ymm5, output_mask);
	only_for_odd_output = _mm256_maskz_add_epi16(k_odd, ymm4, ymm3);

	ymm4 = _mm256_maddubs_epi16(ymm1, mask_filter_0_shift2);
	ymm5 = _mm256_maddubs_epi16(ymm2, mask_filter_1_shift2);
	ymm3 = _mm256_add_epi16(ymm4, ymm5);
	ymm4 = _mm256_maskz_permutexvar_epi16(k1, idx2, ymm3); 
	ymm5 = _mm256_maskz_add_epi16(k_even, ymm4, ymm3);
	//ymm5 = _mm256_add_epi16(ymm3, ymm4);
	//ymm5 = _mm256_and_si256(ymm5, output_mask_sh1);
	//ymm4 = _mm256_srli_si256(ymm3, 2);
	//ymm5 = _mm256_add_epi16(ymm4, ymm3);
	//m4 = _mm256_and_si256(ymm3, mask3);
	//m4 = _mm256_permute2f128_si256(m4, m4, 1);
	//m4 = _mm256_slli_si256(m4, 14);
	//ymm5 = _mm256_add_epi16(ymm5, m4);
	//ymm5 = _mm256_and_si256(ymm5, output_mask_sh1);
	only_for_even_output = _mm256_add_epi16(only_for_even_output, ymm5);

	ymm4 = _mm256_maddubs_epi16(ymm1, mask_filter_0_shift3);
	ymm5 = _mm256_maddubs_epi16(ymm2, mask_filter_1_shift3);
	ymm3 = _mm256_add_epi16(ymm4, ymm5);
	ymm4 = _mm256_maskz_permutexvar_epi16(k1, idx2, ymm3); 
	ymm5 = _mm256_maskz_add_epi16(k_even, ymm4, ymm3);
	//ymm5 = _mm256_add_epi16(ymm3, ymm4);
	//ymm5 = _mm256_and_si256(ymm5, output_mask_sh1);
	//ymm4 = _mm256_srli_si256(ymm3, 2);
	//ymm5 = _mm256_add_epi16(ymm4, ymm3);
	//m4 = _mm256_and_si256(ymm3, mask3);
	//m4 = _mm256_permute2f128_si256(m4, m4, 1);
	//m4 = _mm256_slli_si256(m4, 14);
	//ymm5 = _mm256_add_epi16(ymm5, m4);
	//ymm5 = _mm256_and_si256(ymm5, output_mask_sh1);
	only_for_odd_output = _mm256_add_epi16(only_for_odd_output, ymm5);

	only_for_even_output = division(division_case, only_for_even_output, f);
	only_for_odd_output = division(division_case, only_for_odd_output, f);

	only_for_odd_output = _mm256_slli_si256(only_for_odd_output, 1);
	only_for_even_output = _mm256_add_epi8(only_for_even_output, only_for_odd_output);
	output_row0 = only_for_even_output;


	/////////////////////////////////////////////////////////////////////////////////////////////////////////////
	if (num_of_remaining_elem_in_row == 30){
		_mm256_mask_storeu_epi8(&out_image[img_height-2][column], msk_for_rem, output_row1);
		_mm256_mask_storeu_epi8(&out_image[img_height-1][column], msk_for_rem, output_row0);
		{newPixel = 0;
		newPixel += src_image[img_height-2-1][img_length-1-1] * filter[0][0];
		newPixel += src_image[img_height-2-1][img_length-1] * filter[0][1];
		newPixel += src_image[img_height-2][img_length-1-1] * filter[1][0];
		newPixel += src_image[img_height-2][img_length-1] * filter[1][1];
		newPixel += src_image[img_height-2+1][img_length-1-1] * filter[2][0];
		newPixel += src_image[img_height-2+1][img_length-1] * filter[2][1];
		out_image[img_height-2][img_length-1] = (unsigned char)(newPixel/divisor);

		newPixel = 0;
		newPixel += src_image[img_height-1-1][img_length-1-1] * filter[0][0];
		newPixel += src_image[img_height-1-1][img_length-1] * filter[0][1];
		newPixel += src_image[img_height-1][img_length-1-1] * filter[1][0];
		newPixel += src_image[img_height-1][img_length-1] * filter[1][1];
		out_image[img_height-1][img_length-1] = (unsigned char)(newPixel/divisor);
		}
	}
	else if (num_of_remaining_elem_in_row == 31){
		_mm256_mask_storeu_epi8(&out_image[img_height-2][column], msk_for_rem, output_row1);
		_mm256_mask_storeu_epi8(&out_image[img_height-1][column], msk_for_rem, output_row0);
		{newPixel = 0;
		newPixel += src_image[img_height-2-1][img_length-2-1] * filter[0][0];
		newPixel += src_image[img_height-2-1][img_length-2] * filter[0][1];
		newPixel += src_image[img_height-2-1][img_length-2+1] * filter[0][2];
		newPixel += src_image[img_height-2][img_length-2-1] * filter[1][0];
	  newPixel += src_image[img_height-2][img_length-2] * filter[1][1];
		newPixel += src_image[img_height-2][img_length-2+1] * filter[1][2];
		newPixel += src_image[img_height-2+1][img_length-2-1] * filter[2][0];
		newPixel += src_image[img_height-2+1][img_length-2] * filter[2][1];
		newPixel += src_image[img_height-2+1][img_length-2+1] * filter[2][2];
		out_image[img_height-2][img_length-2] = (unsigned char)(newPixel/divisor);

		newPixel = 0;
		newPixel += src_image[img_height-2-1][img_length-1-1] * filter[0][0];
		newPixel += src_image[img_height-2-1][img_length-1] * filter[0][1];
		newPixel += src_image[img_height-2][img_length-1-1] * filter[1][0];
		newPixel += src_image[img_height-2][img_length-1] * filter[1][1];
		newPixel += src_image[img_height-2+1][img_length-1-1] * filter[2][0];
		newPixel += src_image[img_height-2+1][img_length-1] * filter[2][1];
		out_image[img_height-2][img_length-1] = (unsigned char)(newPixel/divisor);

		newPixel = 0;
		newPixel += src_image[img_height-1-1][img_length-2-1] * filter[0][0];
		newPixel += src_image[img_height-1-1][img_length-2] * filter[0][1];
		newPixel += src_image[img_height-1-1][img_length-2+1] * filter[0][2];
		newPixel += src_image[img_height-1][img_length-2-1] * filter[1][0];
		newPixel += src_image[img_height-1][img_length-2] * filter[1][1];
		newPixel += src_image[img_height-1][img_length-2+1] * filter[1][2];
		out_image[img_height-1][img_length-2] = (unsigned char)(newPixel/divisor);

		newPixel = 0;
		newPixel += src_image[img_height-1-1][img_length-1-1] * filter[0][0];
		newPixel += src_image[img_height-1-1][img_length-1] * filter[0][1];
		newPixel += src_image[img_height-1][img_length-1-1] * filter[1][0];
		newPixel += src_image[img_height-1][img_length-1] * filter[1][1];
		out_image[img_height-1][img_length-1] = (unsigned char)(newPixel/divisor);
		}
	}
	else{
		_mm256_mask_storeu_epi8(&out_image[img_height-2][column], msk_for_rem, output_row1);
		_mm256_mask_storeu_epi8(&out_image[img_height-1][column], msk_for_rem, output_row0);
	}
	return 0;
}

