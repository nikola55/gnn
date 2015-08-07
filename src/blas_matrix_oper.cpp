#include "matrix_opers.h"
#include <gsl_cblas.h>
#include <cstdio>
#include <cstdlib>
void gnn::mmMultiply(int nRowsA, 
					 int nColsA,
					 const float *A, 
					 bool transA,
					 int nRowsB, 
					 int nColsB,
					 const float *B, 
					 bool transB,
					 float *C,
					 float alpha
					 )
{
	// function pointer for BLAS routine that multiplies 2 T matrices 
	// Order 	Specifies row-major (C) or column-major (Fortran) data ordering.
	// TransA 	Specifies whether to transpose matrix A.
	// TransB 	Specifies whether to transpose matrix B.
	// M 	 	Number of rows in matrices A and C.
	// N 	 	Number of columns in matrices B and C.
	// K 	 	Number of columns in matrix A; number of rows in matrix B.
	// alpha 	Scaling factor for the product of matrices A and B.
	// A 	 	Matrix A.
	// lda 	 	The size of the first dimention of matrix A; if you are passing a matrix A[m][n], the value should be m.
	// B 	 	Matrix B.
	// ldb 	 	The size of the first dimention of matrix B; if you are passing a matrix B[m][n], the value should be m.
	// beta 	Scaling factor for matrix C.
	// C 	 	Matrix C.
	// ldc 	 	The size of the first dimention of matrix C; if you are passing a matrix C[m][n], the value should be m.
	int M = transA ? nColsA : nRowsA;
	int N = transB ? nRowsB : nColsB;
	int K = transA ? nRowsA : nColsA;
	int lda = nColsA;
	int ldb = nColsB;
	int ldc = N;

	cblas_sgemm(CblasRowMajor,
		 transA ? CblasTrans : CblasNoTrans,
		 transB ? CblasTrans : CblasNoTrans,
		 M, N, K, alpha,
		 A, lda, B, ldb,
		 0.0, C, ldc
		);
}

void gnn::mmSubtract(int nRowsA, 
					 int nColsA,
					 const float *A,
					 int nRowsB, 
					 int nColsB,
					 const float *B,
					 float *C
					 )
{
	
	if(nRowsA != nRowsB &&
	   nColsA != nColsB) 
	{
		fprintf(stderr, "Illegal matrix dimention\n");
		exit(0);
	}
	
	int N = nRowsA*nColsA;
	
	// y = x + ( - y )
	// y = (-y) + x
	cblas_scopy (N, A, 1, C, 1);
	cblas_saxpy (N, -1, B, 1, C, 1);
}
					
					