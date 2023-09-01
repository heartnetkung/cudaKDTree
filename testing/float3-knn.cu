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

// ==================================================================
__global__ void d_knn5(int *d_results,
                        Float20 *d_queries,
                        int numQueries,
                        Float20 *d_nodes,
                        int numNodes,
                        float maxRadius)
{
  int tid = threadIdx.x+blockIdx.x*blockDim.x;
  if (tid >= numQueries) return;

  cukd::HeapCandidateList<5> result(maxRadius);
  cukd::knn
    <cukd::TrivialFloatPointTraits<Float20>>
    (result,d_queries[tid],d_nodes,numNodes);
  int offset = tid*5;
  for(int i=0;i<5;i++)
    d_results[offset+i] = int(result.entry[i]);
}

void knn5(int *d_results,
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
__global__ void d_knn500(int *d_results,
                        Float20 *d_queries,
                        int numQueries,
                        Float20 *d_nodes,
                        int numNodes,
                        float maxRadius)
{
  int tid = threadIdx.x+blockIdx.x*blockDim.x;
  if (tid >= numQueries) return;

  cukd::HeapCandidateList<500> result(maxRadius);
  cukd::knn
    <cukd::TrivialFloatPointTraits<Float20>>
    (result,d_queries[tid],d_nodes,numNodes);
  int offset = tid*500;
  for(int i=0;i<500;i++)
    d_results[offset+i] = int(result.entry[i]);
}

void knn500(int *d_results,
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

void readPoints(int N, Float20 **d_points)
{
  using namespace cukd::common;
  FILE* stream = fopen("input.txt", "r");
  char line[200];
  CUKD_CUDA_CALL(MallocManaged((void**)d_points,N*sizeof(Float20)));
  // CUKD_CUDA_CALL(MallocManaged((void**)d_queries,N*sizeof(Float20)));
  int i=0;
  Float20 point;

  while (fgets(line, 200, stream))
  {
    char* tmp = strdup(line);
    point = *d_points[i];
    point.x = (float)atof(strtok(tmp, " "));
    point.b = (float)atof(strtok(NULL, " "));
    point.c = (float)atof(strtok(NULL, " "));
    point.d = (float)atof(strtok(NULL, " "));
    point.e = (float)atof(strtok(NULL, " "));
    //5
    point.f = (float)atof(strtok(NULL, " "));
    point.g = (float)atof(strtok(NULL, " "));
    point.h = (float)atof(strtok(NULL, " "));
    point.i = (float)atof(strtok(NULL, " "));
    point.j = (float)atof(strtok(NULL, " "));
    //10
    point.k = (float)atof(strtok(NULL, " "));
    point.l = (float)atof(strtok(NULL, " "));
    point.m = (float)atof(strtok(NULL, " "));
    point.n = (float)atof(strtok(NULL, " "));
    point.o = (float)atof(strtok(NULL, " "));
    //15
    point.p = (float)atof(strtok(NULL, " "));
    point.q = (float)atof(strtok(NULL, " "));
    point.r = (float)atof(strtok(NULL, " "));
    point.s = (float)atof(strtok(NULL, " "));
    point.t = (float)atof(strtok(NULL, " "));

    // *d_queries[i] = point;
    free(tmp);
    i++;
  }

  fclose(stream);
}

// ==================================================================

int main(int ac, const char **av)
{
  using namespace cukd::common;
  
  int nPoints = 173;
  float maxQueryRadius = std::numeric_limits<float>::infinity();
  int nQueries = 173;

  for (int i=1;i<ac;i++) {
    std::string arg = av[i];
    if (arg[0] != '-'){
      nPoints = std::stoi(arg);
      nQueries = nPoints;
    }
    else if (arg == "-r")
      maxQueryRadius = std::stof(av[++i]);
    else
      throw std::runtime_error("known cmdline arg "+arg);
  }
  
  Float20 *d_points = 0;
  Float20 *d_queries = 0;
  readPoints(nPoints,&d_points);
  readPoints(nPoints,&d_queries);

  cukd::buildTree<cukd::TrivialFloatPointTraits<Float20>>(d_points,nPoints);
  CUKD_CUDA_SYNC_CHECK();

  int *d_results;
  int nResult;
  if(nPoints<500)
    nResult=5;
  else
    nResult=500;

  CUKD_CUDA_CALL(MallocManaged((void**)&d_results,nQueries*sizeof(int)*nResult));

  // ==================================================================

  if(nPoints<500)
    knn5(d_results,d_queries,nQueries,d_points,nPoints,maxQueryRadius);
  else
    knn500(d_results,d_queries,nQueries,d_points,nPoints,maxQueryRadius);

  CUKD_CUDA_SYNC_CHECK();
  for(int j=0;j<nQueries;j++){
    std::cout << "j: " << j << " \n";
    for(int k=0;k<nResult;k++){
      int index = d_results[j*nResult+k];
      if (index != -1){
        Float20 point = d_points[index];
        std::cout << " closest point is " << point.x << " \n";
      }
    }
  }

}
