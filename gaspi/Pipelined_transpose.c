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
#include "success_or_die.h"
#include "queue.h"
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
  ASSERT(required == MPI_THREAD_FUNNELED);

  MPI_Comm_rank (MPI_COMM_WORLD, &iProc);
  MPI_Comm_size (MPI_COMM_WORLD, &nProc);

  gaspi_rank_t iProcG, nProcG;
  SUCCESS_OR_DIE (gaspi_proc_init (GASPI_BLOCK));
  SUCCESS_OR_DIE (gaspi_proc_rank (&iProcG));
  SUCCESS_OR_DIE (gaspi_proc_num (&nProcG));
  ASSERT(iProc == iProcG);
  ASSERT(nProc == nProcG);

  // num threads
  omp_set_num_threads(nThreads);

  // assignment per proc, i-direction 
  int mSize = M_SZ/nProc;
  ASSERT(M_SZ % nProc == 0); 
  const int mStart = iProc*mSize;
  const int mStop  = (iProc+1)*mSize;

  // allocate segments for source, work
  gaspi_segment_id_t const source_id = 0;
  SUCCESS_OR_DIE ( gaspi_segment_create
                   ( source_id
                     , mSize * M_SZ * sizeof (double)
                     , GASPI_GROUP_ALL
                     , GASPI_BLOCK
                     , GASPI_MEM_UNINITIALIZED
                     ));
  gaspi_pointer_t source_array;
  SUCCESS_OR_DIE ( gaspi_segment_ptr ( source_id, &source_array) );
  ASSERT (source_array != 0);

  gaspi_segment_id_t const work_id = 1;
  SUCCESS_OR_DIE ( gaspi_segment_create
                   ( work_id
                     , mSize * M_SZ * sizeof (double)
                     , GASPI_GROUP_ALL
                     , GASPI_BLOCK
                     , GASPI_MEM_UNINITIALIZED
                     ));
  gaspi_pointer_t work_array;
  SUCCESS_OR_DIE ( gaspi_segment_ptr ( work_id, &work_array) );
  ASSERT (work_array != 0);

  // allocate target array
  ASSERT(mSize % CL == 0);
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

  // mutual handshake schedule - requires even nProc 
  ASSERT(nProc % 2 == 0);
  gaspi_rank_t to[nProc][nProc-1];
  for (i = 0; i < nProc - 1 ; ++i)
    {
      int k;
      to[i][i] = nProc-1;
      to[nProc-1][i] = i;

      for (k = 1; k < nProc/2 ; ++k)
	{
	  int d1 = (i - k + (nProc-1)) % (nProc-1);	      
	  int d2 = (i + k + (nProc-1)) % (nProc-1);	      
	  to[d1][i] = d2;
	  to[d2][i] = d1;
	}
    }	  

  MPI_Barrier(MPI_COMM_WORLD);

  int iter;
  double median[NITER];

  for (iter = 0; iter < NITER; iter++) 
    {
      double time = -now();
      MPI_Barrier(MPI_COMM_WORLD);
#pragma omp parallel default (none) shared(iter, block_num, iProc, nProc, to, \
	    block, source_array, work_array, target_array, mSize, stdout,stderr)
      {
	// GASPI all-to-all - single comm phase
	if (this_is_the_first_thread())
	  {
	    gaspi_queue_id_t queue_id = 0;
	    int k;
	    for (k = 0; k < nProc - 1 ; ++k)
	      {
		gaspi_rank_t target = to[iProc][k];
		const int len = mSize * mSize * sizeof(double); 
		const int toffset = iProc * mSize;
		const int soffset = target * mSize;
		const gaspi_notification_id_t data_available = iProc;
		wait_for_queue_entries_for_write_notify(&queue_id);
		SUCCESS_OR_DIE ( gaspi_write_notify
				 ( source_id
				   , array_OFFSET (0, soffset)
				   , target
				   , work_id
				   , array_OFFSET (0, toffset)
				   , len
				   , data_available
				   , 1
				   , queue_id
				   , GASPI_BLOCK
				   ));
	      }
	  }

	// multithreaded pipelined local transpose
	int fnl;
	do 
	  {
	    int l;
	    fnl = 0;
	    for (l = tStart; l <= tStop; l++) 	
	      {	    
		if (block[l].stage < iter)
		  {		    
		    if (block[l].pid == iProc)
		      {
			// compute local diagonal 
			data_compute(mStart
				     , mStop
				     , block
				     , l
				     , source_array
				     , target_array
				     , mSize
				     );			    
			block[l].stage++;
		      }
		    else
		      {
			gaspi_notification_id_t data_available = block[l].pid;		
			gaspi_notification_id_t id;
			gaspi_return_t ret;
			if ((ret = gaspi_notify_waitsome (work_id
							  , data_available
							  , 1
							  , &id
							  , GASPI_TEST
							  )) == GASPI_SUCCESS)
			  {
			    ASSERT (id == data_available);
			    
			    // compute off diagonal
			    data_compute(mStart
					 , mStop
					 , block
					 , l
					 , work_array
					 , target_array
					 , mSize
					 );
			    block[l].stage++;
			  }			  
		      }
		  }
		else 
		  {
		    ASSERT(block[l].stage == iter);
		    fnl++;
		  }
	      }
	  }
	while (fnl < tStop-tStart+1);

	// reset all notifications
	if (this_is_the_last_thread())
	  {
	    int k;
	    for (k = 0; k < nProc; ++k)
	      {
		if (k != iProc)
		  {		  
		    const gaspi_notification_id_t data_available = k;
		    gaspi_notification_t value;
		    SUCCESS_OR_DIE (gaspi_notify_reset (work_id
							, data_available
							, &value
							));
		    ASSERT (value == 1);
		  }
	      }
	  }
	
      }
      MPI_Barrier(MPI_COMM_WORLD);
      time += now();
      
      /* iteration time */
      median[iter] = time;
    }
  

  MPI_Barrier(MPI_COMM_WORLD);


  // validate */ 
  data_validate(mSize
		, mStart
		, target_array
		);

  MPI_Barrier(MPI_COMM_WORLD);

  sort_median(&median[0], &median[NITER-1]);

  printf ("# gaspi %s nProc: %d nThreads: %d nBlocks: %d M_SZ: %d niter: %d time: %g\n"
	  , argv[0], nProc, nThreads, block_num, M_SZ, NITER, median[NITER/2]
         );

  MPI_Barrier(MPI_COMM_WORLD);
 
  if (iProc == nProc-1) 
    {
      double res = M_SZ*M_SZ*sizeof(double)*2 / (1024*1024*1024 * median[(NITER-1)/2]);
      printf("\nRate (Transposition Rate): %lf\n",res);
    }

  MPI_Finalize();


  return EXIT_SUCCESS;

}
