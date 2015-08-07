#ifndef _MATRIXOPER_H_INCLUDED_
#define _MATRIXOPER_H_INCLUDED_
#define DUMP_MATRIX(mat, nrows, ncols) \
	do { \
		int __i; int __j; \
		for(__i = 0 ; __i < (nrows) ; __i++) { \
			for(__j = 0 ; __j < (ncols) ; __j++) {\
				printf("%f ", mat[__i*(ncols)+__j]);\
			}\
			printf("\n");\
		}\
	 } while(0);
namespace gnn {
	
	/*
	* Multiplyes matrix A with matrix B and
	* stores the result in C
	*
	* @param1 number of rows in matrix A
	* @param2 number of cols in matrix A
	* @param3 matrix A
	* @param4 transpose A
	* @param5 number of rows in matrix B
	* @param6 number of cols in matrix B
	* @param7 matrix B
	* @param8 transpose B
	* @param9 result
	* @param10 scale AB
	*/

	void mmMultiply(int nRowsA, 
					int nColsA,
					const float *A, 
					bool transA,
					int nRowsB, 
					int nColsB,
					const float *B, 
					bool transB,
					float *C,
					float alpha
					);

	/*
	* Subtract matrix B from matrix A
	* stores the result in C
	*
	* @param1 number of rows in matrix A
	* @param2 number of cols in matrix A
	* @param3 matrix A
	* @param4 number of rows in matrix B
	* @param5 number of cols in matrix B
	* @param6 matrix B
	* @param7 result
	* 
	*/
					
	void mmSubtract(int nRowsA, 
					int nColsA,
					const float *A,
					int nRowsB, 
					int nColsB,
					const float *B,
					float *C
					);
	
}
#endif