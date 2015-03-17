#include <omp.h>

#include "data.h"
#include "assert.h"
#include "math.h"

void data_init(int iProc
	       , int nProc
	       , int mStart
	       , int mStop
	       , block_t *block
	       , int *block_num
	       , int tSize
	       , int mSize
	       )
{
  int i, j;
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

#ifdef DEBUG

  for (i = 0; i < num; ++i)
    {
      int jstart = block[i].start - block[i].pid * mSize;
      printf("iProc: %d block: id: %d jstart: %d jstart mod CL: %d\n",iProc,i,jstart,jstart % CL);
    }

#endif

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


void data_compute(int iProc
		  , int mStart
		  , int mStop
		  , block_t *block
		  , int i
		  , double* work_array
		  , double* target_array
		  , int mSize
		  )
{
  int j, k, l;
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
		   )
{
  int i, j, k;
  for (j = 0; j < M_SZ; j++)
    {
      for (k = 0;  k < mSize; k++) 	
	{
	  /*
	  printf("iProc: %d k: %d j: %d transpose: %lf - source/work/target: %lf/%lf/%lf\n",
		 iProc,k,j,(double) (k + mStart) * M_SZ + j,source_array_ELEM(k,j),work_array_ELEM(k,j),target_array_ELEM(k,j));
	  */
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

