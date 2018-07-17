#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <malloc.h>

#include <opencv2/opencv.hpp>   
#include <opencv2/core/core.hpp> 
#include <opencv2/highgui/highgui.hpp> 

using namespace cv;

#define PI 3.14159265359

#define blockSize 16
#define threshold 6

typedef struct {
	int r, g, b;
}int_rgb;

struct OUTPUT {
	int x_best; //최적의 x
	int y_best; //최적의 y
	int error; //matching error value
};


struct IMG {
	int** image;
	int width;
	int height;
};

int** IntAlloc2(int width, int height)
{
	int** tmp;
	tmp = (int**)calloc(height, sizeof(int*));
	for (int i = 0; i < height; i++)
		tmp[i] = (int*)calloc(width, sizeof(int));
	return(tmp);
}



void IntFree2(int** image, int width, int height)
{
	for (int i = 0; i < height; i++)
		free(image[i]);
	free(image);
}

struct encodingResult { //encoding결과 저장용 구조체
	int x, y;
	int avg;
	double alpha;
	int geo;
	encodingResult* sub;
};

encodingResult** ER_Alloc2(int width, int height) //encoding결과 저장용 구조체 할당
{
	encodingResult** tmp;
	tmp = (encodingResult**)calloc(height, sizeof(encodingResult*));
	for (int i = 0; i < height; i++)
		tmp[i] = (encodingResult*)calloc(width, sizeof(encodingResult));
	return(tmp);
}

void ER_Free2(encodingResult** image, int width, int height) //encoding결과 저장용 구조체 할당해제
{
	for (int i = 0; i < height; i++)
		free(image[i]);
	free(image);
}

int_rgb** IntColorAlloc2(int width, int height)
{
	int_rgb** tmp;
	tmp = (int_rgb**)calloc(height, sizeof(int_rgb*));
	for (int i = 0; i < height; i++)
		tmp[i] = (int_rgb*)calloc(width, sizeof(int_rgb));
	return(tmp);
}


void IntColorFree2(int_rgb** image, int width, int height)
{
	for (int i = 0; i < height; i++)
		free(image[i]);
	free(image);
}



int** ReadImage(char* name, int* width, int* height)
{
	Mat img = imread(name, IMREAD_GRAYSCALE);
	int** image = (int**)IntAlloc2(img.cols, img.rows);
	*width = img.cols;
	*height = img.rows;

	for (int i = 0; i < img.rows; i++)
		for (int j = 0; j < img.cols; j++)
			image[i][j] = img.at<unsigned char>(i, j);
	return(image);
}

void WriteImage(char* name, int** image, int width, int height)
{
	Mat img(height, width, CV_8UC1);
	for (int i = 0; i < height; i++)
		for (int j = 0; j < width; j++)
			img.at<unsigned char>(i, j) = (unsigned char)image[i][j];

	imwrite(name, img);
}


void ImageShow(char* winname, int** image, int width, int height)
{
	Mat img(height, width, CV_8UC1);
	for (int i = 0; i < height; i++)
		for (int j = 0; j < width; j++)
			img.at<unsigned char>(i, j) = (unsigned char)image[i][j];
	imshow(winname, img);
	waitKey(0);
}

int_rgb** ReadColorImage(char* name, int* width, int* height)
{
	Mat img = imread(name, IMREAD_COLOR);
	int_rgb** image = (int_rgb**)IntColorAlloc2(img.cols, img.rows);

	*width = img.cols;
	*height = img.rows;

	for (int i = 0; i < img.rows; i++)
		for (int j = 0; j < img.cols; j++) {
			image[i][j].b = img.at<Vec3b>(i, j)[0];
			image[i][j].g = img.at<Vec3b>(i, j)[1];
			image[i][j].r = img.at<Vec3b>(i, j)[2];
		}
	return(image);
}

void WriteColorImage(char* name, int_rgb** image, int width, int height)
{
	Mat img(height, width, CV_8UC3);
	for (int i = 0; i < height; i++)
		for (int j = 0; j < width; j++) {
			img.at<Vec3b>(i, j)[0] = (unsigned char)image[i][j].b;
			img.at<Vec3b>(i, j)[1] = (unsigned char)image[i][j].g;
			img.at<Vec3b>(i, j)[2] = (unsigned char)image[i][j].r;
		}
	imwrite(name, img);
}

void ColorImageShow(char* winname, int_rgb** image, int width, int height)
{
	Mat img(height, width, CV_8UC3);
	for (int i = 0; i < height; i++)
		for (int j = 0; j < width; j++) {
			img.at<Vec3b>(i, j)[0] = (unsigned char)image[i][j].b;
			img.at<Vec3b>(i, j)[1] = (unsigned char)image[i][j].g;
			img.at<Vec3b>(i, j)[2] = (unsigned char)image[i][j].r;
		}
	imshow(winname, img);
}

template <typename _TP>

void ConnectedComponentLabeling(_TP** seg, int width, int height, int** label, int* no_label)
{
	//Mat bw = threshval < 128 ? (img < threshval) : (img > threshval);
	Mat bw(height, width, CV_8U);

	for (int i = 0; i < height; i++) {
		for (int j = 0; j < width; j++)
			bw.at<unsigned char>(i, j) = (unsigned char)seg[i][j];
	}

	Mat labelImage(bw.size(), CV_32S);
	*no_label = connectedComponents(bw, labelImage, 8); // 0까지 포함된 갯수임
	(*no_label)--;

	for (int i = 0; i < height; i++) {
		for (int j = 0; j < width; j++)
			label[i][j] = labelImage.at<int>(i, j);
	}
}

void Contraction(int** img_in, int width, int height, int** img_out) { //사이즈 줄이기
	for (int y = 0; y < height; y += 2) {
		for (int x = 0; x < width; x += 2) {
			img_out[y / 2][x / 2] = (img_in[y][x] + img_in[y + 1][x]
				+ img_in[y][x + 1] + img_in[y + 1][x + 1]) / 4;
		}
	}
}

void IsoM_0(int** img_in, int width, int height, int**  img_out) {
	for (int y = 0; y < height; y++) {
		for (int x = 0; x < width; x++) {
			img_out[y][x] = img_in[y][x];
		}
	}
}

void IsoM_1(int** img_in, int width, int height, int**  img_out) { //x축 대칭
	for (int y = 0; y < height; y++) {
		for (int x = 0; x < width; x++) {
			img_out[height - 1 - y][x] = img_in[y][x];
		}
	}
}

void IsoM_2(int** img_in, int width, int height, int**  img_out) {  //y축 대칭
	for (int y = 0; y < height; y++) {
		for (int x = 0; x < width; x++) {
			img_out[y][width - 1 - x] = img_in[y][x];
		}
	}
}

void IsoM_3(int** img_in, int width, int height, int**  img_out) {//x,y축 대칭
	for (int y = 0; y < height; y++) {
		for (int x = 0; x < width; x++) {
			img_out[height - 1 - y][width - 1 - x] = img_in[y][x];
		}
	}
}

void IsoM_4(int** img_in, int width, int height, int**  img_out) {
	for (int y = 0; y < height; y++) {
		for (int x = 0; x < width; x++) {
			img_out[width - 1 - x][height - 1 - y] = img_in[y][x];
		}
	}
}

void IsoM_5(int** img_in, int width, int height, int**  img_out) { //+90도회전
	for (int y = 0; y < height; y++) {
		for (int x = 0; x < width; x++) {
			img_out[width - 1 - x][y] = img_in[y][x];
		}
	}
}

void IsoM_6(int** img_in, int width, int height, int**  img_out) { //180도 회전
	for (int y = 0; y < height; y++) {
		for (int x = 0; x < width; x++) {
			img_out[height - 1 - y][x] = img_in[y][x];
		}
	}
}

void IsoM_7(int** img_in, int width, int height, int**  img_out) {
	for (int y = 0; y < height; y++) {
		for (int x = 0; x < width; x++) {
			img_out[x][height - 1 - y] = img_in[y][x];
		}
	}
}

void GeoTransform(int no, int** A, int width, int height, int** B)
{
	switch (no)
	{
	case 0: //복사
		IsoM_0(A, width, height, B);
		break;
	case 1: //y축 대칭
		IsoM_1(A, width, height, B);
		break;
	case 2: //x,y축 대칭
		IsoM_2(A, width, height, B);
		break;
	case 3:
		IsoM_3(A, width, height, B);
		break;
	case 4:
		IsoM_4(A, width, height, B);
		break;
	case 5: // +90도 회전
		IsoM_5(A, width, height, B);
		break;
	case 6: //180도 회전
		IsoM_6(A, width, height, B);
		break;
	case 7: //-90도 회전
		IsoM_7(A, width, height, B);
		break;
	default:
		printf("error!");
		break;
	}
}

int ComputeAVG(int** image, int width, int height)
{
	int avg;
	float temp = 0;
	for (int y = 0; y < height; y++) {
		for (int x = 0; x < width; x++) {
			temp += image[y][x];
		}
	}
	temp /= width*height;
	avg = (int)(temp + 0.5);
	//반올림
	return avg;
}

void ReadBlock(int** image, int x, int y, int dx, int dy, int** block) //x,y는 image의 좌표 //dx,dy는 block크기
{
	for (int j = 0; j < dy; j++) {
		for (int i = 0; i < dx; i++) {
			block[j][i] = image[y + j][x + i];
		}
	}
}

void WriteBlock(int** image, int x, int y, int dx, int dy, int** block) //x,y는 image의 좌표 //dx,dy는 block크기
{
	for (int j = 0; j < dy; j++) {
		for (int i = 0; i < dx; i++) {
			image[y + j][x + i] = block[j][i];
		}
	}
}


void FindBestBlock(int stride, int** block, int size_block, int** image, int width, int height,
	OUTPUT* output)//size_block이 3이면 3x3  //error가장작은 block위치,에러찾기
{ // stride는 jump할 칸

	int error, error_min = INT_MAX;
	int m, n;
	for (int i = 0; i < height; i += stride) {
		if (i + size_block > height) break;
		for (int j = 0; j < width; j += stride) {
			if (j + size_block > width) break;
			error = 0;
			for (int y = 0; y < size_block; y++)
				for (int x = 0; x < size_block; x++)
					error += abs(block[y][x] - image[i + y][j + x]);
			if (error_min > error) {
				error_min = error;
				m = j;
				n = i;
			}
		}
	}

	output->x_best = m;
	output->y_best = n;
	output->error = error_min;
}



int FindBestISO(int** A, //A는 고정되어 있음
	int** B, //B는 geometric transform 함
	int size)
{
	OUTPUT output;
	int opt = INT_MAX;
	int num;
	int** tmp = (int**)IntAlloc2(blockSize, blockSize);
	for (int k = 0; k < 8; k++)
	{
		GeoTransform(k, B, blockSize, blockSize, tmp);
		FindBestBlock(8, A, blockSize, tmp, blockSize, blockSize, &output);
		if (opt > output.error) {
			opt = output.error;
			num = k;
		}
	}
	return num;
}



int blockMinusAVG(int**block, int dx, int dy, int** block_out)

{
	int temp = ComputeAVG(block, dx, dy);
	int tmp = 0;
	for (int i = 0; i < dy; i++)
		for (int j = 0; j < dx; j++) {
			//tmp = block[i][j] - temp;
			//if (tmp < 0) tmp = 0;
			block_out[i][j] = block[i][j] - temp;
		}
	return(temp);
}


void blockPlusValue(int**block, int dx, int dy, int value, int** block_out)
{
	int tmp = 0;
	for (int i = 0; i < blockSize; i++) {
		for (int j = 0; j < blockSize; j++) {
			//tmp = block[i][j] + value;
			//if (tmp > 255) tmp = 255;
			block_out[i][j] = block[i][j] + value;
		}
	}
}



void blockPlusAVG(int**block, int dx, int dy, int** block_out)
{
	int temp = ComputeAVG(block, blockSize, blockSize);
	for (int i = 0; i < blockSize; i++)
		for (int j = 0; j < blockSize; j++)
			block_out[i][j] = block[i][j] + temp;
}



void blockMultiplyAlpha(int** block, int dx, int dy, double alpha, int** block_out) {
	for (int i = 0; i < dy; i++) {
		for (int j = 0; j < dx; j++) {
			block_out[i][j] = (int)(block[i][j] * alpha);
		}
	}
}


void fprintfSub(FILE* fp, encodingResult* A, int* flag)
{
	if (A->sub == NULL){
		*flag = 0;
		return;
	}
	else *flag = 1;

	for (int i = 0; i < 4; i++){
		if (A->sub[i].sub != NULL)
			*flag = 1;
		else *flag = 0;
		fprintf(fp, "%d %d %d %d %f %d\n", A->sub[i].x, A->sub[i].y, A->sub[i].geo, A->sub[i].avg, A->sub[i].alpha, *flag);
		fprintfSub(fp, &(A->sub[i]), flag);
	}
}

bool WirteParameter(char* name, encodingResult** A, int width, int height)
{
	FILE* fp = fopen(name, "w");
	if (fp == NULL) {
		printf("\n Failure in fopen!");
		return (false);
	}
	for (int i = 0; i < height; i++) {
		for (int j = 0; j < width; j++) {
			int flag = 0;
			if (A[i][j].sub != NULL)
				flag = 1;
			fprintf(fp, "%d %d %d %d %f %d\n", A[i][j].x, A[i][j].y, A[i][j].geo, A[i][j].avg, A[i][j].alpha, flag);

			if (A[i][j].sub != NULL){
				fprintfSub(fp, &(A[i][j]), &flag);
			}
		}
	}
	fclose(fp);
	return (true);
}

bool ReadParameter(char* name, encodingResult** A, int width, int height)

{
	FILE* fp = fopen(name, "r");

	if (fp == NULL) {
		printf("\n Failure in fopen!");
		return (false);
	}

	for (int i = 0; i < height; i++) {
		for (int j = 0; j < width; j++) {
			fscanf(fp, "%d%d%d%d%lf", &A[i][j].x, &A[i][j].y, &A[i][j].geo, &A[i][j].avg, &A[i][j].alpha);
		}
	}
	fclose(fp);
	return (true);

}


void TemplateMatchingWithDownSamplingPlusShuffle(int** block, int bx, int by, int** domain, int dx, int dy,//domain은 blocksize의 2배  dx,dy는 도메인 좌표
	int* x, int* y, int* block_mean, int* k, double* alpha, int* error, int n, int m) //돌아가있는 이미지를 받아서 어떻게 돌리면 이미지 평균값 가장작은지 출력, 그 위치,변환 출력
{//x,y는 에러 값 가장작은 위치 출력, num은 그때의 변환이 무엇인지 출력	
	int num;
	int** domain_tmp = (int**)IntAlloc2(bx, by);
	int** block_AC = (int**)IntAlloc2(bx, by);
	int** domain_AC = (int**)IntAlloc2(bx, by);
	int** domain_AC_tmp = (int**)IntAlloc2(bx, by);
	int** domain_AC_tmp2 = (int**)IntAlloc2(bx, by);
	OUTPUT out;
	*block_mean = blockMinusAVG(block, bx, by, block_AC);
	Contraction(domain, 2 * bx, 2 * by, domain_tmp);
	blockMinusAVG(domain_tmp, bx, by, domain_AC_tmp);
	for (double p = 0.3; p <= 1.0; p += 0.1) {
		blockMultiplyAlpha(domain_AC_tmp, bx, by, p, domain_AC_tmp2);
		for (num = 0; num < 8; num++) {
			GeoTransform(num, domain_AC_tmp2, bx, by, domain_AC);
			FindBestBlock(1, domain_AC, bx, block_AC, dx / 2, dy / 2, &out);
			if (*error > out.error) {
				*error = out.error;
				*x = n;
				*y = m;
				*k = num;
				*alpha = p;
			}
		}
	}

	IntFree2(domain_tmp, bx, by);
	IntFree2(domain_AC, bx, by);
	IntFree2(block_AC, bx, by);
	IntFree2(domain_AC_tmp, bx, by);
	IntFree2(domain_AC_tmp2, bx, by);
}

void compareBlockAndDomain(int** img_in, int** block,int bsize, int width, int height, int stride,
	int* x, int* y, int* block_mean, int* geo, double* alpha, int* error)
{
	int** domain = (int**)IntAlloc2(bsize, bsize);

	for (int yy = 0; yy < height - bsize; yy += stride){ //block의 size만큼 이동하면서 도메인 잡고 비교
		for (int xx = 0; xx < width - bsize; xx += stride){
			ReadBlock(img_in, xx, yy, bsize, bsize, domain);
			TemplateMatchingWithDownSamplingPlusShuffle(block, bsize / 2, bsize / 2, domain, bsize, bsize,
				x, y, block_mean, geo, alpha, error, xx, yy);
		}
	}
	IntFree2(domain, bsize, bsize);
}

void fractalCoding(int** image, int width, int height, int** block, int bsize, int error, encodingResult* parameter)
//error 값 크면 한번더자르기 재귀함수 사용
{
	int** block_tmp = (int**)IntAlloc2(bsize / 2, bsize / 2);
	printf("%d\n", error);

	int block_mean, x, y, geo;
	double alpha = 0.0;

	if (bsize < 4){
		parameter->sub = NULL;
		return;
	}
	if (error > threshold) {
		parameter->sub = (encodingResult*)calloc(4, sizeof(encodingResult));
		int i = 0;
		for (int k = 0; k < 2; k++){ //4조각위해 2중 for문
			for (int p = 0; p < 2; p++){
				int error_forCheck = INT_MAX;
				x = 0; y = 0;
				block_mean = 0; geo = 0; alpha = 0.0;
				ReadBlock(block, bsize / 2 * p, bsize / 2 * k, bsize / 2, bsize / 2, block_tmp);
				
				compareBlockAndDomain(image,block_tmp,bsize,width,height,bsize,&x,&y,&block_mean,&geo,&alpha,&error_forCheck);

				parameter->sub[i].x = x;
				parameter->sub[i].y = y;
				parameter->sub[i].avg = block_mean;
				parameter->sub[i].geo = geo;
				parameter->sub[i].alpha = alpha;

				fractalCoding(image, width, height, block_tmp, bsize / 2, error_forCheck / ((bsize / 2) * (bsize / 2)), &(parameter->sub[i]));
				i++;
			}
		}
	}
	else {
		parameter->sub = NULL;
	}
	IntFree2(block_tmp, bsize / 2, bsize / 2);
}

void enCoding(int** img_in, int width, int height)
{
	int** block = (int**)IntAlloc2(blockSize, blockSize);

	int block_mean, x, y, geo;
	double alpha = 0.0;
	int stride = 8;

	encodingResult** B = ER_Alloc2(width / blockSize, height / blockSize);
	bool tmp = 0;
	for (int i = 0; i < height; i += blockSize) {
		for (int j = 0; j < width; j += blockSize) {
			ReadBlock(img_in, j, i, blockSize, blockSize, block);
			int error = INT_MAX;

			compareBlockAndDomain(img_in, block, blockSize*2, width, height, stride, &x, &y, &block_mean, &geo, &alpha, &error);

			B[i / blockSize][j / blockSize].alpha = alpha;
			B[i / blockSize][j / blockSize].avg = block_mean;
			B[i / blockSize][j / blockSize].geo = geo;
			B[i / blockSize][j / blockSize].x = x;
			B[i / blockSize][j / blockSize].y = y;

			if (error / (blockSize*blockSize) > threshold){
				printf("%d\n", error / (blockSize*blockSize));
				fractalCoding(img_in, width, height, block, blockSize, error / (blockSize*blockSize), &B[i / blockSize][j / blockSize]);
			}

			else B[i / blockSize][j / blockSize].sub = NULL;
			printf("block ==>  x : %d y : %d\n", j, i);
		}
	}
	printf("encoding done\n");
	WirteParameter("__encoding_blockSize16_threshold6.txt", B, width / blockSize, height / blockSize);
	IntFree2(block, blockSize, blockSize);
	ER_Free2(B, width / blockSize, height / blockSize);
}


void main()
{
	int width, height;
	int** img_in = ReadImage("lena256x512.bmp", &width, &height);

	enCoding(img_in, width, height); //image_in,image_in의 width,height

	printf("finish\n");
}