#include "page_rank.h"

#include <stdlib.h>
#include <cmath>
#include <omp.h>
#include <utility>

#include "../common/CycleTimer.h"
#include "../common/graph.h"

// pageRank --
//
// g:           graph to process (see common/graph.h)
// solution:    array of per-vertex vertex scores (length of array is num_nodes(g))
// damping:     page-rank algorithm's damping parameter
// convergence: page-rank algorithm's convergence threshold
//
void pageRank(Graph g, double *solution, double damping, double convergence)
{

  // initialize vertex weights to uniform probability. Double
  // precision scores are used to avoid underflow for large graphs

  int numNodes = num_nodes(g);
  double equal_prob = 1.0 / numNodes;

  #pragma omp parallel for 
  for (int i = 0; i < numNodes; ++i)
  {
    solution[i] = equal_prob;
  }

  /*
     For PP students: Implement the page rank algorithm here.  You
     are expected to parallelize the algorithm using openMP.  Your
     solution may need to allocate (and free) temporary arrays.

     Basic page rank pseudocode is provided below to get you started:

     // initialization: see example code above
     score_old[vi] = 1/numNodes;

     while (!converged) {

       // compute score_new[vi] for all nodes vi:
       score_new[vi] = sum over all nodes vj reachable from incoming edges
                          { score_old[vj] / number of edges leaving vj  }
       score_new[vi] = (damping * score_new[vi]) + (1.0-damping) / numNodes;

       score_new[vi] += sum over all nodes v in graph with no outgoing edges
                          { damping * score_old[v] / numNodes }

       // compute how much per-node scores have changed
       // quit once algorithm has converged

       global_diff = sum over all nodes vi { abs(score_new[vi] - score_old[vi]) };
       converged = (global_diff < convergence)
     }

   */
  double* old_solution = new double[numNodes];
  bool converged = false;

  while (!converged) {
    double global_diff = 0.0f;
    double no_outgoing_edges_sum = 0.0f;
    
    #pragma omp parallel
    {
      #pragma omp for 
      for (int i = 0; i < numNodes; i++) {
        old_solution[i] = solution[i];
      }

      #pragma omp for reduction(+:no_outgoing_edges_sum)
      for (int i = 0; i < numNodes; i++) {
        if (outgoing_size(g, i) == 0) {
          no_outgoing_edges_sum += damping * old_solution[i] / numNodes;
        }
      }

      #pragma omp for reduction(+:global_diff)
      for (int i = 0; i < numNodes; i++) {
        double partial_sum = 0.0f;
        for(const Vertex* v = incoming_begin(g, i); v != incoming_end(g, i); v++) {
          partial_sum += old_solution[*v] / (double)outgoing_size(g, *v);
        }
        partial_sum = damping * partial_sum + (1.0 - damping) / numNodes;
        partial_sum += no_outgoing_edges_sum;
        solution[i] = partial_sum;
        global_diff += fabs(solution[i] - old_solution[i]);
      }
    }
    converged = (global_diff < convergence);
  }

  free(old_solution);
}
