

#include "mex.h"
#include "matrix.h"
#include <igraph.h>
#include <string.h>

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
     const mwSize *dmE, *dmW;

     igraph_t graph;
     igraph_vector_t vedg, vw;
     igraph_vector_t result;
     igraph_real_t *edges = mxGetPr(prhs[0]);
     igraph_real_t *ws = mxGetPr(prhs[1]);
     double *exact = mxGetPr(prhs[2]);

     dmE = mxGetDimensions(prhs[0]);
     igraph_vector_view(&vedg, edges, (long int)dmE[0]);

     dmW = mxGetDimensions(prhs[1]);
     igraph_vector_view(&vw, ws, (long int)dmW[0]);

     igraph_create(&graph, &vedg, 0, IGRAPH_DIRECTED);

     igraph_vector_init(&result, 0);
     
     if(*exact == 1.0)
        igraph_feedback_arc_set(&graph, &result, &vw, IGRAPH_FAS_EXACT_IP);
     else
        igraph_feedback_arc_set(&graph, &result, &vw, IGRAPH_FAS_APPROX_EADES);
     
     
     long int result_size = igraph_vector_size(&result);
     
     plhs[0] = mxCreateDoubleMatrix(result_size,1,mxREAL);
     double *output = mxGetPr(plhs[0]);
     
     igraph_vector_copy_to(&result,output);
     

     igraph_vector_destroy(&result);
     igraph_destroy(&graph);
}
