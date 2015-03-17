#ifndef DATA_H
#define DATA_H

#include "constant.h"

typedef struct 
{
  int start, end, tid, pid;
} block_t;

#define POSITION(i,j) ((i) + mSize * (j))
#define array_OFFSET(i,j) (POSITION (i,j) * sizeof(double))
#define source_array_ELEM(i,j) ((double *)source_array)[POSITION (i,j)]
#define work_array_ELEM(i,j) ((double *)work_array)[POSITION (i,j)]
#define target_array_ELEM(i,j) ((double *)target_array)[POSITION (i,j)]


void data_init(int iProc
	       , int nProc
	       , int mStart
	       , int mStop
	       , block_t *block
	       , int *block_num
	       , int tSize
	       , int mSize
	       );

void data_init_tlocal(int mStart
		      , int mStop
		      , block_t *block
		      , int block_num
		      , int *tStart
		      , int *tStop
		      , int tid
		      , double* source_array
		      , double* work_array
		      , double* target_array
		      ,int mSize
		      );

void data_compute(int iProc
		  , int mStart
		  , int mStop
		  , block_t *block
		  , int i
		  , double* work_array
		  , double* target_array
		  , int mSize
		  );

void data_validate(int iProc
		   , int mStart
		   , int mStop
		   , block_t *block
		   , int block_num
		   , int tStart
		   , int tStop
		   , double* source_array
		   , double* work_array
		   , double* target_array
		   , int mSize
		   );

void sort_median(double *begin, double *end);


#endif
