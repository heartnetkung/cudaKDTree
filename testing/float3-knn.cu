// ======================================================================== //
// Copyright 2022-2022 Ingo Wald                                            //
//                                                                          //
// Licensed under the Apache License, Version 2.0 (the "License");          //
// you may not use this file except in compliance with the License.         //
// You may obtain a copy of the License at                                  //
//                                                                          //
//     http://www.apache.org/licenses/LICENSE-2.0                           //
//                                                                          //
// Unless required by applicable law or agreed to in writing, software      //
// distributed under the License is distributed on an "AS IS" BASIS,        //
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. //
// See the License for the specific language governing permissions and      //
// limitations under the License.                                           //
// ======================================================================== //

#include "cukd/builder.h"
// knn = "k-nearest-neighbor" query
#include "cukd/knn.h"
#include <queue>
#include <limits>
#include <iomanip>

float3 *generatePoints(int N)
{
  std::cout << "generating " << N <<  " points" << std::endl;
  float3 *d_points = 0;
  CUKD_CUDA_CALL(MallocManaged((void**)&d_points,N*sizeof(float3)));
  for (int i=0;i<N;i++) {
    d_points[i].x = (float)drand48();
    d_points[i].y = (float)drand48();
    d_points[i].z = (float)drand48();
  }
  return d_points;
}

// ==================================================================
__global__ void d_knn50(float *d_results,
                        float3 *d_queries,
                        int numQueries,
                        float3 *d_nodes,
                        int numNodes,
                        float maxRadius)
{
  int tid = threadIdx.x+blockIdx.x*blockDim.x;
  if (tid >= numQueries) return;

  cukd::HeapCandidateList<50> result(maxRadius);
  float sqrDist
    = cukd::knn
    <cukd::TrivialFloatPointTraits<float3>>
    (result,d_queries[tid],d_nodes,numNodes);
  d_results[tid] = sqrtf(sqrDist);//cukd::knn(result,d_queries[tid],d_nodes,numNodes));
  // d_results[tid] = sqrtf(cukd::knn(result,d_queries[tid],d_nodes,numNodes));
}

void knn50(float *d_results,
           float3 *d_queries,
           int numQueries,
           float3 *d_nodes,
           int numNodes,
           float maxRadius)
{
  int bs = 128;
  int nb = cukd::common::divRoundUp(numQueries,bs);
  d_knn50<<<nb,bs>>>(d_results,d_queries,numQueries,d_nodes,numNodes,maxRadius);
}

// ==================================================================

int main(int ac, const char **av)
{
  using namespace cukd::common;
  
  int nPoints = 173;
  float maxQueryRadius = std::numeric_limits<float>::infinity();
  size_t nQueries = 10*1000*1000;
  int nRepeats = 1;
  for (int i=1;i<ac;i++) {
    std::string arg = av[i];
    if (arg[0] != '-')
      nPoints = std::stoi(arg);
    else if (arg == "-nr")
      nRepeats = atoi(av[++i]);
    else if (arg == "-nq")
      nQueries = atoi(av[++i]);
    else if (arg == "-r")
      maxQueryRadius = std::stof(av[++i]);
    else
      throw std::runtime_error("known cmdline arg "+arg);
  }
  
  float3 *d_points = generatePoints(nPoints);

  {
    double t0 = getCurrentTime();
    std::cout << "calling builder..." << std::endl;
    // float dist2 = cukd::sqrDistance<cukd::TrivialFloatPointTraits<float3>>(queryPoint,points[i]);
    cukd::buildTree<cukd::TrivialFloatPointTraits<float3>>(d_points,nPoints);
    CUKD_CUDA_SYNC_CHECK();
    double t1 = getCurrentTime();
    std::cout << "done building tree, took " << prettyDouble(t1-t0) << "s" << std::endl;
  }

  float3 *d_queries = generatePoints(nQueries);
  float  *d_results;
  CUKD_CUDA_CALL(MallocManaged((void**)&d_results,nQueries*sizeof(float)));

  // ==================================================================
  {
    std::cout << "running " << nRepeats << " sets of knn50 queries..." << std::endl;
    double t0 = getCurrentTime();
    for (int i=0;i<nRepeats;i++)
      knn50(d_results,d_queries,nQueries,d_points,nPoints,maxQueryRadius);
    CUKD_CUDA_SYNC_CHECK();
    double t1 = getCurrentTime();
    std::cout << "done " << nRepeats << " iterations of knn50 query, took " << prettyDouble(t1-t0) << "s" << std::endl;
    std::cout << " that's " << prettyDouble((t1-t0)/nRepeats) << "s per query (avg)..." << std::endl;
    std::cout << " ... or " << prettyDouble(nQueries*nRepeats/(t1-t0)) << " queries/s" << std::endl;
  }

}
