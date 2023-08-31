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

using namespace cukd;
using namespace cukd::common;

float4 *generatePoints(int N)
{
  std::cout << "generating " << N <<  " points" << std::endl;
  float4 *d_points = 0;
  CUKD_CUDA_CALL(MallocManaged((void**)&d_points,N*sizeof(float4)));
  for (int i=0;i<N;i++) {
    d_points[i].x = (float)drand48();
    d_points[i].y = (float)drand48();
    d_points[i].z = (float)drand48();
    d_points[i].w = (float)drand48();
  }
  return d_points;
}

float4 *readPoints(int N)
{
  using namespace cukd::common;
  FILE* stream = fopen("hello.txt", "r");
  char line[100];
  float4 *d_points = 0;
  CUKD_CUDA_CALL(MallocManaged((void**)&d_points,N*sizeof(float4)));
  int i=0;

  while (fgets(line, 100, stream))
  {
    char* tmp = strdup(line);
    d_points[i].x = (float)atof(strtok(tmp, " "));
    d_points[i].y = (float)atof(strtok(NULL, " "));
    d_points[i].z = (float)atof(strtok(NULL, " "));
    d_points[i].w = (float)atof(strtok(NULL, " "));
    std::cout << prettyDouble(d_points[i].w) << std::endl;
    free(tmp);
    i++;
  }
  std::cout << prettyDouble(N) << "nline" << prettyDouble(i) << std::endl;

  fclose(stream);

  return d_points;
}

float4 *readPoints2(int N)
{
  using namespace cukd::common;
  FILE* stream = fopen("hello.txt", "r");
  char line[100];
  float4 *d_points = 0;
  CUKD_CUDA_CALL(MallocManaged((void**)&d_points,N*sizeof(float4)));
  int i=0;

  while (fgets(line, 100, stream))
  {
    char* tmp = strdup(line);
    d_points[i].x = (float)atof(strtok(tmp, " "));
    d_points[i].y = (float)atof(strtok(NULL, " "));
    d_points[i].z = (float)atof(strtok(NULL, " "));
    d_points[i].w = (float)atof(strtok(NULL, " "));
    d_points[i].w += 3.0f;
    std::cout << prettyDouble(d_points[i].w) << std::endl;
    free(tmp);
    i++;
  }
  std::cout << prettyDouble(N) << "nline" << prettyDouble(i) << std::endl;

  fclose(stream);

  return d_points;
}

// ==================================================================
__global__ void d_knn50(float *d_results,
                        float4 *d_queries,
                        int numQueries,
                        float4 *d_nodes,
                        int numNodes,
                        float maxRadius)
{
  int tid = threadIdx.x+blockIdx.x*blockDim.x;
  if (tid >= numQueries) return;

  cukd::HeapCandidateList<50> result(maxRadius);
  d_results[tid] = sqrtf(cukd::knn
                         <cukd::TrivialFloatPointTraits<float4>>
                         (result,d_queries[tid],d_nodes,numNodes));
}

void knn50(float *d_results,
           float4 *d_queries,
           int numQueries,
           float4 *d_nodes,
           int numNodes,
           float maxRadius)
{
  int bs = 128;
  int nb = cukd::common::divRoundUp(numQueries,bs);
  d_knn50<<<nb,bs>>>(d_results,d_queries,numQueries,d_nodes,numNodes,maxRadius);
}

// ==================================================================
inline void verifyKNN(int pointID, int k, float maxRadius,
                      float4 *points, int numPoints,
                      float4 queryPoint,
                      float presumedResult)
{
  std::priority_queue<float> closest_k;
  for (int i=0;i<numPoints;i++) {
    float d = sqrtf(cukd::sqrDistance
                    <cukd::TrivialFloatPointTraits<float4>>
                    (queryPoint,points[i]));
    if (d <= maxRadius)
      closest_k.push(d);
    if (closest_k.size() > k)
      closest_k.pop();
  }

  float actualResult = (closest_k.size() == k) ? closest_k.top() : maxRadius;


  // check if the top 21-ish bits are the same; this will allow the
  // compiler to produce slightly different results on host and device
  // (usually caused by it uses madd's on one and separate +/* on
  // t'other...
  bool closeEnough
    =  /* this catches result==inf:*/(actualResult == presumedResult)
    || /* this catches bit errors: */(fabsf(actualResult - presumedResult) <= 1e-6f);
  
  if (!closeEnough) {
    std::cout << "for point #" << pointID << ": "
              << "verify found max dist " << std::setprecision(10) << actualResult
              << " (bits " << (int*)(uint64_t)((uint32_t&)actualResult)
              << "), knn reported " << presumedResult
              << " (bits " << (int*)(uint64_t)((uint32_t&)presumedResult)
              << "), difference is " << (actualResult-presumedResult)
              << std::endl;
    throw std::runtime_error("verification failed");
  }
}

int main(int ac, const char **av)
{
  using namespace cukd::common;
  
  int nPoints = 173;
  bool verify = false;
  float maxQueryRadius = std::numeric_limits<float>::infinity();
  int nRepeats = 1;
  int useFile = 0;

  for (int i=1;i<ac;i++) {
    std::string arg = av[i];
    if (arg[0] != '-')
      nPoints = std::stoi(arg);
    else if (arg == "-v")
      verify = true;
    else if (arg == "-nr")
      nRepeats = atoi(av[++i]);
    else if (arg == "-r")
      maxQueryRadius = std::stof(av[++i]);
    else if (arg == "-t")
      useFile = i++;
    else
      throw std::runtime_error("known cmdline arg "+arg);
  }
  
  float4 *d_points;
  if (useFile == 0)
    d_points = generatePoints(nPoints);
  else
    d_points = readPoints(nPoints);

  {
    double t0 = getCurrentTime();
    std::cout << "calling builder..." << std::endl;
    cukd::buildTree
      <cukd::TrivialFloatPointTraits<float4>>
      (d_points,nPoints);
    CUKD_CUDA_SYNC_CHECK();
    double t1 = getCurrentTime();
    std::cout << "done building tree, took " << prettyDouble(t1-t0) << "s" << std::endl;
  }

  // size_t nQueries = 10*1000*1000;
  // float4 *d_queries = generatePoints(nQueries);
  //HNK manual fix
  float4 *d_queries = readPoints2(nPoints);
  int nQueries = nPoints;

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

    for (int i=0;i<nQueries;i++)
      std::cout << prettyDouble(d_results[i]) << "\n";
    if (verify) {
      std::cout << "verifying result ..." << std::endl;
      for (int i=0;i<nQueries;i++)
        verifyKNN(i,50,maxQueryRadius,d_points,nPoints,d_queries[i],d_results[i]);
      std::cout << "verification passed ... " << std::endl;
    }
  }

}