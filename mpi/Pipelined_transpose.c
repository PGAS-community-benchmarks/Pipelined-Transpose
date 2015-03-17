/*
 * This file is part of a small series of tutorial,
 * which aims to demonstrate key features of the GASPI
 * standard by means of small but expandable examples.
 * Conceptually the tutorial follows a MPI course
 * developed by EPCC and HLRS.
 *
 * Contact point for the MPI tutorial:
 *                 rabenseifner@hlrs.de
 * Contact point for the GASPI tutorial:
 *                 daniel.gruenewald@itwm.fraunhofer.de
 *                 mirko.rahn@itwm.fraunhofer.de
 *                 christian.simmendinger@t-systems.com
 */

#include "assert.h"
#include "constant.h"
#include "data.h"
#include "now.h"
#include "mm_pause.h"
#include "threads.h"

#include <malloc.h>
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

// threadprivate rangelist data
static int tStart = -1;
#pragma omp threadprivate(tStart)
static int tStop  = -2;
#pragma omp threadprivate(tStop)


int main (int argc, char *argv[])
{
  int i;
  int nProc, iProc;
  int provided, required = MPI_THREAD_FUNNELED;
  MPI_Init_thread(&argc, &argv, required, &provided);
  ASSERT(provided == MPI_THREAD_FUNNELED);

  MPI_Comm_rank (MPI_COMM_WORLD, &iProc);
  MPI_Comm_size (MPI_COMM_WORLD, &nProc);

  // num threads
  omp_set_num_threads(nThreads);

  // assignment per proc, i-direction 
  int mSize = M_SZ/nProc;
  ASSERT(M_SZ % nProc == 0); 
  const int mStart = iProc*mSize;
  const int mStop  = (iProc+1)*mSize;

  // allocate segments for source, work, target
  ASSERT(mSize % CL == 0);
  double *source_array = memalign(CL* sizeof (double), mSize * M_SZ * sizeof (double));
  double *work_array = memalign(CL* sizeof (double), mSize * M_SZ * sizeof (double));
  double *target_array = memalign(CL* sizeof (double), mSize * M_SZ * sizeof (double));

  // assignment per thread
  int tSize = M_SZ/nThreads;
  if (M_SZ % nThreads != 0) 
    {
      tSize++;
    }

  // alloc (max) M_SZ blocks per process 
  block_t (*block) = malloc( M_SZ * sizeof(block_t));	  
  int block_num = 0;

  data_init(block
	    , &block_num
	    , tSize
	    , mSize
	    );

  // init thread local data, set thread range (tStart <= row <= tStop)
#pragma omp parallel default (none) shared(block, block_num, \
  	    mSize, source_array, work_array, target_array, stdout, stderr)
  {
    int const tid = omp_get_thread_num();  
    data_init_tlocal(mStart
		     , mStop
		     , block
		     , block_num
		     , &tStart
		     , &tStop
		     , tid
		     , source_array
		     , work_array
		     , target_array
		     , mSize
		     );

  }

  MPI_Barrier(MPI_COMM_WORLD);

  int iter;
  double median[NITER];
  for (iter = 0; iter < NITER; iter++) 
    {
      double time = -now();
      MPI_Barrier(MPI_COMM_WORLD);
#pragma omp parallel default (none) shared(block_num, iProc, nProc, \
	    block, source_array, work_array, target_array, mSize, stdout,stderr)
      {
	int const tid = omp_get_thread_num();  
        if (tid == 0)
	  {
	    MPI_Alltoall(source_array
			 , mSize*mSize
			 , MPI_DOUBLE
			 , work_array
			 , mSize*mSize
			 , MPI_DOUBLE
			 , MPI_COMM_WORLD
			 );
	  }

#ifndef WITHOUT_LOCAL_TRANSPOSE
#pragma omp barrier
	// compute local diagonal
	int l;
	for (l = tStart; l <= tStop; l++) 	
	  {	    
	    // compute
	    data_compute(mStart
			 , mStop
			 , block
			 , l
			 , work_array
			 , target_array
			 , mSize
			 );
	    
	  }
#endif

#pragma omp barrier

      }
      MPI_Barrier(MPI_COMM_WORLD);
      time += now();

      /* iteration time */
      median[iter] = time;
    }


  MPI_Barrier(MPI_COMM_WORLD);

#ifndef WITHOUT_LOCAL_TRANSPOSE
  // validate */ 
  data_validate(mSize
		, mStart
		, target_array
		);
#endif

  
  MPI_Barrier(MPI_COMM_WORLD);

  sort_median(&median[0], &median[NITER-1]);

  printf ("# mpi %s nProc: %d nThreads: %d nBlocks: %d M_SZ: %d niter: %d time: %g\n"
	  , argv[0], nProc, nThreads, block_num, M_SZ, NITER, median[NITER/2]
         );

  MPI_Barrier(MPI_COMM_WORLD);
 
#ifndef WITHOUT_LOCAL_TRANSPOSE
  if (iProc == nProc-1) 
    {
      double res = M_SZ*M_SZ*sizeof(double)*2 / (1024*1024*1024 * median[(NITER-1)/2]);
      printf("\nRate (Transposition Rate): %lf\n",res);
    }
#endif

  MPI_Finalize();


  return EXIT_SUCCESS;

}
