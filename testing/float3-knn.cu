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
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

Float20 *generatePoints(int N)
{
  std::cout << "generating " << N <<  " points" << std::endl;
  Float20 *d_points = 0;
  CUKD_CUDA_CALL(MallocManaged((void**)&d_points,N*sizeof(Float20)));
  for (int i=0;i<N;i++) {
    d_points[i].x = (float)drand48();
    d_points[i].b = (float)drand48();
    d_points[i].c = (float)drand48();
    d_points[i].d = (float)drand48();
    d_points[i].e = (float)drand48();
    //5
    d_points[i].f = (float)drand48();
    d_points[i].g = (float)drand48();
    d_points[i].h = (float)drand48();
    d_points[i].i = (float)drand48();
    d_points[i].j = (float)drand48();
    //10
    d_points[i].k = (float)drand48();
    d_points[i].l = (float)drand48();
    d_points[i].m = (float)drand48();
    d_points[i].n = (float)drand48();
    d_points[i].o = (float)drand48();
    //15
    d_points[i].p = (float)drand48();
    d_points[i].q = (float)drand48();
    d_points[i].r = (float)drand48();
    d_points[i].s = (float)drand48();
    d_points[i].t = (float)drand48();
  }
  return d_points;
}

// ==================================================================
__global__ void d_knn5(uint64_t *d_results,
                        Float20 *d_queries,
                        int numQueries,
                        Float20 *d_nodes,
                        int numNodes,
                        float maxRadius)
{
  int tid = threadIdx.x+blockIdx.x*blockDim.x;
  if (tid >= numQueries) return;

  cukd::FixedCandidateList<5> result(maxRadius);
  cukd::knn
    <cukd::TrivialFloatPointTraits<Float20>>
    (result,d_queries[tid],d_nodes,numNodes);
  int offset = tid*5;
  d_results[offset] = result.entry[0];
  d_results[offset+1] = result.entry[1];
  d_results[offset+2] = result.entry[2];
  d_results[offset+3] = result.entry[3];
  d_results[offset+4] = result.entry[4];
}

void knn5(uint64_t *d_results,
           Float20 *d_queries,
           int numQueries,
           Float20 *d_nodes,
           int numNodes,
           float maxRadius)
{
  int bs = 128;
  int nb = cukd::common::divRoundUp(numQueries,bs);
  d_knn5<<<nb,bs>>>(d_results,d_queries,numQueries,d_nodes,numNodes,maxRadius);
}
// ==================================================================
__global__ void d_knn500(uint64_t *d_results,
                        Float20 *d_queries,
                        int numQueries,
                        Float20 *d_nodes,
                        int numNodes,
                        float maxRadius)
{
  int tid = threadIdx.x+blockIdx.x*blockDim.x;
  if (tid >= numQueries) return;

  cukd::FixedCandidateList<500> result(maxRadius);
  cukd::knn
    <cukd::TrivialFloatPointTraits<Float20>>
    (result,d_queries[tid],d_nodes,numNodes);
  int offset = tid*500;
  d_results[offset] = result.entry[0];
  d_results[offset+1] = result.entry[1];
  d_results[offset+2] = result.entry[2];
  d_results[offset+3] = result.entry[3];
  d_results[offset+4] = result.entry[4];
}

void knn500(uint64_t *d_results,
           Float20 *d_queries,
           int numQueries,
           Float20 *d_nodes,
           int numNodes,
           float maxRadius)
{
  int bs = 128;
  int nb = cukd::common::divRoundUp(numQueries,bs);
  d_knn500<<<nb,bs>>>(d_results,d_queries,numQueries,d_nodes,numNodes,maxRadius);
}
// ==================================================================

Float20 *readPoints(int N)
{
  using namespace cukd::common;
  FILE* stream = fopen("input.txt", "r");
  char line[200];
  Float20 *d_points = 0;
  CUKD_CUDA_CALL(MallocManaged((void**)&d_points,N*sizeof(Float20)));
  int i=0;

  while (fgets(line, 200, stream))
  {
    char* tmp = strdup(line);
    d_points[i].x = (float)atof(strtok(tmp, " "));
    d_points[i].b = (float)atof(strtok(NULL, " "));
    d_points[i].c = (float)atof(strtok(NULL, " "));
    d_points[i].d = (float)atof(strtok(NULL, " "));
    d_points[i].e = (float)atof(strtok(NULL, " "));
    //5
    d_points[i].f = (float)atof(strtok(NULL, " "));
    d_points[i].g = (float)atof(strtok(NULL, " "));
    d_points[i].h = (float)atof(strtok(NULL, " "));
    d_points[i].i = (float)atof(strtok(NULL, " "));
    d_points[i].j = (float)atof(strtok(NULL, " "));
    //10
    d_points[i].k = (float)atof(strtok(NULL, " "));
    d_points[i].l = (float)atof(strtok(NULL, " "));
    d_points[i].m = (float)atof(strtok(NULL, " "));
    d_points[i].n = (float)atof(strtok(NULL, " "));
    d_points[i].o = (float)atof(strtok(NULL, " "));
    //15
    d_points[i].p = (float)atof(strtok(NULL, " "));
    d_points[i].q = (float)atof(strtok(NULL, " "));
    d_points[i].r = (float)atof(strtok(NULL, " "));
    d_points[i].s = (float)atof(strtok(NULL, " "));
    d_points[i].t = (float)atof(strtok(NULL, " "));

    free(tmp);
    i++;
  }

  fclose(stream);

  return d_points;
}

// ==================================================================

int main(int ac, const char **av)
{
  using namespace cukd::common;
  
  int nPoints = 173;
  float maxQueryRadius = std::numeric_limits<float>::infinity();
  int nQueries = 10*1000*1000;
  int nRepeats = 1;
  int isAssigned = 0;

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
    else if (arg == "-t")
      isAssigned = i++;
    else
      throw std::runtime_error("known cmdline arg "+arg);
  }
  
  Float20 *d_points;
  if(isAssigned == 0){
    // TODO change to memcpy
    d_points = generatePoints(nPoints);
  }else{
    d_points = readPoints(nPoints);
  }
  //HNK manual fix
  Float20 *d_queries = readPoints(nPoints);
  nQueries = nPoints;


  {
    double t0 = getCurrentTime();
    std::cout << "calling builder..." << std::endl;
    cukd::buildTree<cukd::TrivialFloatPointTraits<Float20>>(d_points,nPoints);
    CUKD_CUDA_SYNC_CHECK();
    double t1 = getCurrentTime();
    std::cout << "done building tree, took " << prettyDouble(t1-t0) << "s" << std::endl;
  }

  uint64_t  *d_results;
  int nResult;
  if(nPoints<500)
    nResult=5;
  else
    nResult=500;
  CUKD_CUDA_CALL(MallocManaged((void**)&d_results,nQueries*sizeof(uint64_t)*nResult));

  // ==================================================================
  {
    std::cout << "running " << nRepeats << " sets of knn500 queries..." << std::endl;
    double t0 = getCurrentTime();
    for (int i=0;i<nRepeats;i++){
      if(nPoints<500)
        knn5(d_results,d_queries,nQueries,d_points,nPoints,maxQueryRadius);
      else
        knn500(d_results,d_queries,nQueries,d_points,nPoints,maxQueryRadius);
    }
    CUKD_CUDA_SYNC_CHECK();
    for (int i=0;i<nRepeats;i++)
      for(int j=0;j<nQueries;j++){
        std::cout << "j: " << j << " \n";
        for(int k=0;k<nResult;k++)
          std::cout << " closest point is " << d_results[j*nResult+k] << " \n";
      }

    double t1 = getCurrentTime();
    std::cout << "done " << nRepeats << " iterations of knn500 query, took " << prettyDouble(t1-t0) << "s" << std::endl;
    std::cout << " that's " << prettyDouble((t1-t0)/nRepeats) << "s per query (avg)..." << std::endl;
    std::cout << " ... or " << prettyDouble(nQueries*nRepeats/(t1-t0)) << " queries/s" << std::endl;
  }

}
