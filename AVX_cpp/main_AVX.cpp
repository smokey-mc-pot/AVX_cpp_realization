#include <iostream>
#include <cstdio>
#include <immintrin.h>
#include "Blur_AVX.hpp"

signed char** Mask;         
unsigned short int divisor; 
char In_path[100];
char Out_path[100];
char append[3];
unsigned int M;         // cols
unsigned int N;         // rows
unsigned char** frame1; // input image
unsigned char** filt;   // output image
char header[100];

#define IN "/input_im/15991k_8995k.pgm"
#define OUT "/output_im/15991k_8995k_out.pgm"

int create_IO_arrays(){
	int i, j;
  frame1 = (unsigned char**)_mm_malloc(N * sizeof(unsigned char*), 64);

  for (i = 0; i < N; i++)
    frame1[i] = (unsigned char*)_mm_malloc(M * sizeof(unsigned char), 64);

  filt = (unsigned char**)_mm_malloc(N * sizeof(unsigned char*), 64);

  for (i = 0; i < N; i++)
    filt[i] = (unsigned char*)_mm_malloc(M * sizeof(unsigned char), 64);

  for (i = 0; i < N; i++)
    for (j = 0; j < M; j++)
      filt[i][j] = 0;

  std::cout << "Arrays have been successfully created\n";
  return 0;
}

int getint(FILE *fp){  
  int c, i, firstchar; 
  c = getc(fp);

  while (1){
    if (c == '#'){
      char cmt[256], *sp;
      sp = cmt;  
      firstchar = 1;

      while (1){
        c = getc(fp);

        if (firstchar && c == ' ') 
          firstchar = 0; 
        else{
          if (c == '\n' || c == EOF) 
            break;
          if ((sp-cmt) < 250) 
            *sp++ = c;
        }
      }

      *sp++ = '\n';
      *sp   = '\0';
    }

    if (c == EOF) 
      return 0;
    if (c >= '0' && c <= '9') 
      break; 
    c = getc(fp);
  }

  i = 0;

  while (1){
    i = (i*10) + (c - '0');
    c = getc(fp);
    if (c == EOF) 
      return i;
    if (c < '0' || c > '9') 
      break;
  }
  return i;
}

void openfile(char *filename, FILE** finput){
  int x0, y0, x;
  *finput = fopen(filename, "rb");
  fscanf(*finput, "%s", header);
  x0 = getint(*finput);
  y0 = getint(*finput);
  std::cout << "header is " << header << ", while x = " << x0 << ", y = " << y0 << std::endl;
  M = x0;
  N = y0;
  std::cout << "Image dim are M = " << M << ", N = " << N << std::endl;
  create_IO_arrays();
  x = getint(*finput); 
  std::cout << "range info is " << x << std::endl;
}

void read_image(char* filename){
  int c, i, j, temp;
  FILE *finput;

  std::cout << "Reading " << filename << " image from disk" << std::endl;
  finput = NULL;
  openfile(filename, &finput);

  if ((header[0] == 'P') && (header[1] == '2')){
    for (j = 0; j < N; j++){
      for (i = 0; i < M; i++){
	      if (fscanf(finput, "%d", &temp) == EOF)
		      exit(EXIT_FAILURE);
        frame1[j][i] = (unsigned char)temp;
      }
    }
  }
  else if ((header[0] == 'P') && (header[1] == '5')){
	  for (j = 0; j < N; j++){
	    for (i = 0; i < M; i++){
	      c = getc(finput);
	      frame1[j][i] = (unsigned char)c;
	    }
	  }
	}
  else{
    std::cout << "problem with reading image" << std::endl;
    exit(EXIT_FAILURE);
  }

  fclose(finput);
  std::cout << "image successfully read from disc" << std::endl;
}

void write_image2(char* filename){
  FILE* foutput;
  int i, j;
  std::cout << "Writing result to disk ..." << std::endl << std::endl; 
  foutput = fopen(filename, "wb");
  fprintf(foutput, "P2\n");
  fprintf(foutput, "%d %d\n", M, N);
  fprintf(foutput, "%d\n", 255);

  for (j = 0; j < N; ++j){
    for (i = 0; i < M; ++i){
      fprintf(foutput, "%3d ", filt[j][i]);

      if (i % 32 == 31) 
        fprintf(foutput,"\n");
    }

    if (M % 32 != 0) 
      fprintf(foutput,"\n");
  }
  fclose(foutput);
}

int create_kernel(){
  Mask = (signed char**)_mm_malloc(3 * sizeof(signed char*), 64);

  for (int i = 0; i < 3; i++)
    Mask[i] = (signed char*)_mm_malloc(3 * sizeof(signed char), 64);

	Mask[0][0] = 1; Mask[0][1] = 2;  Mask[0][2] = 1;
  Mask[1][0] = 2; Mask[1][1] = 4;  Mask[1][2] = 2;
  Mask[2][0] = 1; Mask[2][1] = 2;  Mask[2][2] = 1;
  divisor = 16;
  return 0;
}

//void generate_optimized_images(){
//	int i, extension = 0;
//
//	for (i = 0; i < 31; i++){
//		sprintf(append,"%d", extension);
//		strcat(In_path, IN);
//		strcat(In_path, append);
//		strcat(In_path, ".pgm");
//		strcat(Out_path, OUT);
//		strcat(Out_path, append);
//		strcat(Out_path, "_out.pgm");
//
//		read_image(In_path);
//    //tbb::tick_count t0 = tbb::tick_count::now();
//	  Gaussian_Blur_optimized_3x3(frame1, filt, M, N, divisor, Mask);
//    //std::cout << "Time: " << (tbb::tick_count::now() - t0).seconds() << " seconds" << std::endl;
//		write_image2(Out_path);
//
//		extension++;
//		In_path[0] = '\0';
//		Out_path[0] = '\0';
//	}
//}

void generate_optimized_images(){
  strcat_s(In_path, IN);
  strcat_s(Out_path, OUT);
	read_image(In_path);
  //tbb::tick_count t0 = tbb::tick_count::now();
  Gaussian_Blur_optimized_3x3(frame1, filt, M, N, divisor, Mask);
  //std::cout << "Time: " << (tbb::tick_count::now() - t0).seconds() << " seconds" << std::endl;
  write_image2(Out_path);
}

int main(){
  create_kernel();
  //tbb::tick_count t0 = tbb::tick_count::now();
	generate_optimized_images();
  //std::cout << "Time: " << (tbb::tick_count::now() - t0).seconds() << " seconds" << std::endl;
	return 0;
}
