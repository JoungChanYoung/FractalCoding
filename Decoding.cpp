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
	for (int i = 0; i<height; i++)
		tmp[i] = (int*)calloc(width, sizeof(int));
	return(tmp);
}

void IntFree2(int** image, int width, int height)
{
	for (int i = 0; i<height; i++)
		free(image[i]);

	free(image);
}

int_rgb** IntColorAlloc2(int width, int height)
{
	int_rgb** tmp;
	tmp = (int_rgb**)calloc(height, sizeof(int_rgb*));
	for (int i = 0; i<height; i++)
		tmp[i] = (int_rgb*)calloc(width, sizeof(int_rgb));
	return(tmp);
}

void IntColorFree2(int_rgb** image, int width, int height)
{
	for (int i = 0; i<height; i++)
		free(image[i]);

	free(image);
}

int** ReadImage(char* name, int* width, int* height)
{
	Mat img = imread(name, IMREAD_GRAYSCALE);
	int** image = (int**)IntAlloc2(img.cols, img.rows);

	*width = img.cols;
	*height = img.rows;

	for (int i = 0; i<img.rows; i++)
		for (int j = 0; j<img.cols; j++)
			image[i][j] = img.at<unsigned char>(i, j);

	return(image);
}

void WriteImage(char* name, int** image, int width, int height)
{
	Mat img(height, width, CV_8UC1);
	for (int i = 0; i<height; i++)
		for (int j = 0; j<width; j++)
			img.at<unsigned char>(i, j) = (unsigned char)image[i][j];

	imwrite(name, img);
}


void ImageShow(char* winname, int** image, int width, int height)
{
	Mat img(height, width, CV_8UC1);
	for (int i = 0; i<height; i++)
		for (int j = 0; j<width; j++)
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

	for (int i = 0; i<img.rows; i++)
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
	for (int i = 0; i<height; i++)
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
	for (int i = 0; i<height; i++)
		for (int j = 0; j<width; j++) {
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

	for (int y = 0; y<height; y++) {
		for (int x = 0; x<width; x++) {
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
	for (int j = 0; j<dy; j++) {
		for (int i = 0; i<dx; i++) {
			block[j][i] = image[y + j][x + i];
		}
	}
}
void WriteBlock(int** image, int x, int y, int dx, int dy, int** block) //x,y는 image의 좌표 //dx,dy는 block크기
{
	for (int j = 0; j<dy; j++) {
		for (int i = 0; i<dx; i++) {
			image[y + j][x + i] = block[j][i];
		}
	}
}



void FindBestBlock(int stride, int** block, int size_block, int** image, int width, int height,
	OUTPUT* output)//size_block이 3이면 3x3  //error가장작은 block위치,에러찾기
{ // stride는 jump할 칸
	int error, error_min = INT_MAX;
	int x, y;
	for (int i = 0; i<height; i += stride) {
		if (i + size_block > height) break;
		for (int j = 0; j<width; j += stride) {
			if (j + size_block > width) break;
			error = 0;
			for (int y = 0; y < size_block; y++)
				for (int x = 0; x < size_block; x++)
					error += abs(block[y][x] - image[i + y][j + x]);
			if (error_min > error) {
				error_min = error;
				x = j;
				y = i;
			}
		}
	}
	output->x_best = x;
	output->y_best = y;
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
			block_out[i][j] = block[i][j] - temp;
		}
	return(temp);
}
void blockPlusValue(int**block, int dx, int dy, int value, int** block_out)
{
	int tmp = 0;
	for (int i = 0; i < dy; i++) {
		for (int j = 0; j < dx; j++) {
			block_out[i][j] = block[i][j] + value;
		}
	}
}

void blockPlusAVG(int**block, int dx, int dy, int** block_out)
{
	int temp = ComputeAVG(block, dx, dy);
	for (int i = 0; i < dy; i++)
		for (int j = 0; j < dx; j++)
			block_out[i][j] = block[i][j] + temp;
}


void blockMultiplyAlpha(int** block, int dx, int dy, double alpha, int** block_out) {
	for (int i = 0; i < dy; i++) {
		for (int j = 0; j < dx; j++) {
			block_out[i][j] = (int)(block[i][j] * alpha);
		}
	}
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
	for (int i = 0; i<height; i++)
		tmp[i] = (encodingResult*)calloc(width, sizeof(encodingResult));
	return(tmp);
}

void ER_Free2(encodingResult** image, int width, int height) //encoding결과 저장용 구조체 할당해제
{
	for (int i = 0; i<height; i++)
		free(image[i]);

	free(image);
}

void Decoding(encodingResult* B, int width, int height, int bsize, int** image_tmp, int** image_dec, int j, int i) {
	int** block = (int**)IntAlloc2(bsize * 2, bsize * 2);
	int** block_tmp = (int**)IntAlloc2(bsize, bsize);
	int** block_tmp2 = (int**)IntAlloc2(bsize, bsize);
	int** block_tmp3 = (int**)IntAlloc2(bsize, bsize);
	int** block_tmp4 = (int**)IntAlloc2(bsize, bsize);
	int** block_tmp5 = (int**)IntAlloc2(bsize, bsize);

	ReadBlock(image_tmp, B->x, B->y, bsize * 2, bsize * 2, block);
	Contraction(block, bsize * 2, bsize * 2, block_tmp);
	blockMinusAVG(block_tmp, bsize, bsize, block_tmp2);

	GeoTransform(B->geo, block_tmp2, bsize, bsize, block_tmp3);
	blockMultiplyAlpha(block_tmp3, bsize, bsize, B->alpha, block_tmp4);
	blockPlusValue(block_tmp4, bsize, bsize, B->avg, block_tmp5);

	WriteBlock(image_dec, j, i, bsize, bsize, block_tmp5);
	
	if (B->sub != NULL){
		int bsize_tmp = bsize / 2;
		int tmp = 0;
		for (int y = 0; y < 2; y++){
			for (int x = 0; x<2; x++){
				if ((B->sub[tmp].x + bsize * 2) > width || (B->sub[tmp].y + bsize * 2) > height)
					continue;
				Decoding(&(B->sub[tmp]), width, height, bsize_tmp, image_tmp, image_dec, j + x*bsize_tmp, i + y*bsize_tmp);
				tmp++;
			}
		}
	}
	
	IntFree2(block, bsize * 2, bsize * 2);
	IntFree2(block_tmp, bsize, bsize);
	IntFree2(block_tmp2, bsize, bsize);
	IntFree2(block_tmp3, bsize, bsize);
	IntFree2(block_tmp4, bsize, bsize);
	IntFree2(block_tmp5, bsize, bsize);
}

void Decoding_block(encodingResult* B, int width, int height, int bsize, int** image_tmp, int** image_dec,Mat image_block, int j, int i) {
	int** block = (int**)IntAlloc2(bsize * 2, bsize * 2);
	int** block_tmp = (int**)IntAlloc2(bsize, bsize);
	int** block_tmp2 = (int**)IntAlloc2(bsize, bsize);
	int** block_tmp3 = (int**)IntAlloc2(bsize, bsize);
	int** block_tmp4 = (int**)IntAlloc2(bsize, bsize);
	int** block_tmp5 = (int**)IntAlloc2(bsize, bsize);
	
	ReadBlock(image_tmp, B->x, B->y, bsize * 2, bsize * 2, block);
	Contraction(block, bsize * 2, bsize * 2, block_tmp);
	blockMinusAVG(block_tmp, bsize, bsize, block_tmp2);

	GeoTransform(B->geo, block_tmp2, bsize, bsize, block_tmp3);
	blockMultiplyAlpha(block_tmp3, bsize, bsize, B->alpha, block_tmp4);
	blockPlusValue(block_tmp4, bsize, bsize, B->avg, block_tmp5);

	WriteBlock(image_dec, j, i, bsize, bsize, block_tmp5);

	if (B->sub != NULL){
		int bsize_tmp = bsize / 2;
		int tmp = 0;
		
		for (int y = 0; y < 2; y++){
			for (int x = 0; x<2; x++){
				line(image_block, Point(j + bsize / 2, i),
					Point(j + bsize / 2,i + bsize), Scalar(255,255,255));
				line(image_block, Point(j, i + bsize/2),
					Point(j + bsize, i + bsize / 2), Scalar(255, 255, 255));
				if ((B->sub[tmp].x + bsize * 2) > width || (B->sub[tmp].y + bsize * 2) > height)
					continue;
				Decoding_block(&(B->sub[tmp]), width, height, bsize_tmp, image_tmp, image_dec,image_block, j + x*bsize_tmp, i + y*bsize_tmp);
				tmp++;
			}
		}
	}

	IntFree2(block, bsize * 2, bsize * 2);
	IntFree2(block_tmp, bsize, bsize);
	IntFree2(block_tmp2, bsize, bsize);
	IntFree2(block_tmp3, bsize, bsize);
	IntFree2(block_tmp4, bsize, bsize);
	IntFree2(block_tmp5, bsize, bsize);
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
			fprintf(fp, "%d %d %d %d %f\n", A[i][j].x, A[i][j].y, A[i][j].geo, A[i][j].avg, A[i][j].alpha);
		}
	}

	fclose(fp);

	return (true);
}

void fscanSub(FILE* fp, int* flag, encodingResult* A)
{
	if (*flag == 0)
		return;

	A->sub = (encodingResult*)calloc(4, sizeof(encodingResult));
	for (int j = 0; j < 4; j++){
		fscanf(fp, "%d%d%d%d%lf%d", &A->sub[j].x, &A->sub[j].y, &A->sub[j].geo, &A->sub[j].avg, &A->sub[j].alpha, flag);
		printf("%d %d %d %d %lf\n", A->sub[j].x, A->sub[j].y, A->sub[j].geo, A->sub[j].avg, A->sub[j].alpha);
		fscanSub(fp, flag,&(A->sub[j]));
	}
}

bool ReadParameter(char* name, encodingResult** A, int width, int height)
{
	FILE* fp = fopen(name, "r");

	int flag = 0;
	if (fp == NULL) {
		printf("\n Failure in fopen!");
		return (false);
	}
	for (int i = 0; i < height; i++) {
		for (int j = 0; j < width; j++) {
			fscanf(fp, "%d%d%d%d%lf%d", &A[i][j].x, &A[i][j].y, &A[i][j].geo, &A[i][j].avg, &A[i][j].alpha, &flag);
			//printf("%d %d %d %d %lf\n", A[i][j].x, A[i][j].y, A[i][j].geo, A[i][j].avg, A[i][j].alpha);
			if (flag != 0){
				fscanSub(fp, &flag, &A[i][j]);
			}
		}
	}
	fclose(fp);
	return (true);
}

double computePSNR(int** A, int** B, int width, int height)
{
	double error = 0.0;
	for (int i = 0; i < height; i++)
	{
		for (int j = 0; j < width; j++) {
			error += (double)(A[i][j] - B[i][j])*(A[i][j] - B[i][j]);
		}
	}
	error = error / (width*height);
	double psnr = 10.0 * log10(255.*255. / error);
	return(psnr);
}

void main() {
	int width, height;

	int** block = (int**)IntAlloc2(blockSize, blockSize);
	int** img_in = ReadImage("lena256x512.bmp", &width, &height);
	int** image_tmp = (int**)IntAlloc2(width, height);
	int** image_dec = (int**)IntAlloc2(width, height);

	encodingResult** C = ER_Alloc2(width / blockSize, height / blockSize);

	Mat img_block(512,256, CV_8UC3,Scalar(0,0,0));
	int** img_block_result = (int**)IntAlloc2(width, height);

	ReadParameter("__encoding_blockSize16_threshold3.txt", C, width / blockSize, height / blockSize);
	printf("Readparameter!!!!");

	for (int i = 0; i < 5; i++) {
		for (int m = 0; m < height / blockSize; m++) {
			for (int n = 0; n < width / blockSize; n++) {

				//block분할 없는 decoding
				//Decoding(&C[m][n], width, height, blockSize, image_tmp, image_dec, n*p, m*p);		

				//block분할을 고려한 decoding
				Decoding_block(&C[m][n], width, height, blockSize, image_tmp, image_dec, img_block, n*blockSize, m*blockSize);
			}
		}
		ImageShow("decoded2", image_dec, width, height);
		GeoTransform(0, image_dec, width, height, image_tmp); //image_dec를 image_tmp로 복사
	}

	for (int y = 0; y < height; y++){
		for (int x = 0; x < width; x++){
			if (img_block.at<Vec3b>(y, x)[0] != 0)
				img_block_result[y][x] = 255;
			else
				img_block_result[y][x] = image_dec[y][x];
		}
	}

	ImageShow("decoded\n", image_dec, width, height);
	ImageShow("img_block_result", img_block_result, width, height);
	WriteImage("dec.bmp", image_dec, width, height);

	double psnr = 0.0;
	psnr = computePSNR(img_in,image_dec,width,height);
	printf("psnr : %lf", psnr);
	//할당 free
	IntFree2(block, blockSize, blockSize);
	IntFree2(img_in, width, height);
	IntFree2(image_tmp, width, height);
	IntFree2(image_dec, width, height);
	ER_Free2(C, width / blockSize, height / blockSize);
}
