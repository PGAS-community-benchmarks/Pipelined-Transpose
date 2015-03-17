#include <omp.h>

#include "data.h"
#include "assert.h"
#include "math.h"

void data_init(block_t *block
	       , int *block_num
	       , int tSize
	       , int mSize
	       )
{
  int i;
  int num = 0;
  int lid = 0;
  int pid = 0;
  int start = 0;
  int last  = 0;

  for (i = 1; i < M_SZ; ++i)
    {
      if (i % BL_SZ == 0 || i % mSize == 0 || i % tSize == 0)
	{
	  // prev block
	  block[num].end  = i;
	  block[num].start = start;
	  block[num].tid = lid;
	  block[num].pid = pid;

	  if (i % mSize == 0)
	    {	      
	      last = i;
	      pid++;
	    }

	  if (i % tSize == 0)
	    {
	      lid++;
	    }

	  start = i;
	  num++;
	}
    }
  // add last block
  block[num].end  = i;
  block[num].start = start;
  block[num].tid = lid;
  block[num].pid = pid;
  num++;

  *block_num = num;

}


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
		      , int mSize
		      )
{  
  int i, j, k;

  // set thread range 
  for (k = 0; k < block_num; ++k)
    {
      if (block[k].tid == tid)
	{
	  if (*tStart == -1)
	    {
	      *tStart = k;
	    }
	  *tStop = k;
	}
    }

  for (i = *tStart; i <= *tStop; i++)
    {
      for (j = block[i].start; j < block[i].end; j++) 	
	{
	  for (k = 0;  k < mStop - mStart; k++) 	
	    {
	      source_array_ELEM(k,j) = (double) j * M_SZ + k + mStart;
	    }
	}	  
      for (j = block[i].start; j < block[i].end; j++) 	
	{
	  for (k = 0;  k < mStop - mStart; k++) 	
	    {
	      work_array_ELEM(k,j) = (double) -1 ;
	    }
	}
      for (j = block[i].start; j < block[i].end; j++) 	
	{
	  for (k = 0;  k < mStop - mStart; k++) 	
	    {
	      target_array_ELEM(k,j) = (double) -1 ;
	    }
	}
    }
}


void data_compute(int mStart
		  , int mStop
		  , block_t *block
		  , int i
		  , double* work_array
		  , double* target_array
		  , int mSize
		  )
{
  int j, k;
  for (k = 0; k < mStop-mStart; k++) 	
    {
      const int jstart = block[i].pid * mSize;
      const int kstart = block[i].pid * mSize;  
#pragma simd
#pragma vector nontemporal
#pragma vector aligned      
      for (j = block[i].start-jstart; j < block[i].end-jstart; ++j)
	{
	  target_array_ELEM(j,kstart+k) = work_array_ELEM(k, j + jstart);
	}
    }

}


void data_validate(int mSize
		   , int mStart
		   , double* target_array
		   )
{
  int j, k;
  for (j = 0; j < M_SZ; j++)
    {
      for (k = 0;  k < mSize; k++) 	
	{
	  ASSERT(target_array_ELEM(k,j) == (double) (k + mStart) * M_SZ + j);
	}
    }
}

static void swap(double *a, double *b)
{
  double tmp = *a;
  *a = *b;
  *b = tmp;
}

void sort_median(double *begin, double *end)
{
  double *ptr;
  double *split;
  if (end - begin <= 1)
    return;
  ptr = begin;
  split = begin + 1;
  while (++ptr != end) {
    if (*ptr < *begin) {
      swap(ptr, split);
      ++split;
    }
  }
  swap(begin, split - 1);
  sort_median(begin, split - 1);
  sort_median(split, end);
}

