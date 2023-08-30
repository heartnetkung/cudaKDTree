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
__global__ void d_knn500(float *d_results,
                        Float20 *d_queries,
                        int numQueries,
                        Float20 *d_nodes,
                        int numNodes,
                        float maxRadius)
{
  int tid = threadIdx.x+blockIdx.x*blockDim.x;
  if (tid >= numQueries) return;

  cukd::HeapCandidateList<500> result(maxRadius);
  float sqrDist
    = cukd::knn
    <cukd::TrivialFloatPointTraits<Float20>>
    (result,d_queries[tid],d_nodes,numNodes);
  d_results[tid] = sqrtf(sqrDist);//cukd::knn(result,d_queries[tid],d_nodes,numNodes));
  // d_results[tid] = sqrtf(cukd::knn(result,d_queries[tid],d_nodes,numNodes));
}

void knn500(float *d_results,
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

Float20 *readPoints(const char* file, int N)
{
  std::cout << "hello 1.1";
  using namespace cukd::common;
  FILE* stream = fopen(file, "r");
  char line[100];
  Float20 *d_points = 0;
  CUKD_CUDA_CALL(MallocManaged((void**)&d_points,N*sizeof(Float20)));
  int i=0;

  std::cout << "hello 1.2";
  while (fgets(line, 100, stream))
  {
    char* tmp = strdup(line);
    d_points[i].x = (float)atof(tmp);
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
  std::cout << "hello 1.3";

  fclose(stream);
  std::cout << "my_n_points" << N;
  std::cout << "first coords" << prettyDouble(d_points[0].x);
  std::cout << "second coords" << prettyDouble(d_points[0].b);

  std::cout << "hello 1.4";
  return d_points;
}

// ==================================================================

int main(int ac, const char **av)
{
  using namespace cukd::common;
  
  int nPoints = 173;
  float maxQueryRadius = std::numeric_limits<float>::infinity();
  size_t nQueries = 10*1000*1000;
  int nRepeats = 1;
  char* file = NULL;

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
    else if (arg == "-t"){
      std::cout << "hello 0";
      file = strdup(av[++i].c_str());
      std::cout << "hello 0.1";
    }
    else
      throw std::runtime_error("known cmdline arg "+arg);
  }
  
  std::cout << "hello1";
  Float20 *d_points;
  if(file == NULL){
    d_points = generatePoints(nPoints);
  }else{
    d_points = readPoints(file, nPoints);
  }
  std::cout << "hello2";


  {
    double t0 = getCurrentTime();
    std::cout << "calling builder..." << std::endl;
    cukd::buildTree<cukd::TrivialFloatPointTraits<Float20>>(d_points,nPoints);
    CUKD_CUDA_SYNC_CHECK();
    double t1 = getCurrentTime();
    std::cout << "done building tree, took " << prettyDouble(t1-t0) << "s" << std::endl;
  }

  Float20 *d_queries = generatePoints(nQueries);
  float  *d_results;
  CUKD_CUDA_CALL(MallocManaged((void**)&d_results,nQueries*sizeof(float)));

  // ==================================================================
  {
    std::cout << "running " << nRepeats << " sets of knn500 queries..." << std::endl;
    double t0 = getCurrentTime();
    for (int i=0;i<nRepeats;i++)
      knn500(d_results,d_queries,nQueries,d_points,nPoints,maxQueryRadius);
    CUKD_CUDA_SYNC_CHECK();
    double t1 = getCurrentTime();
    std::cout << "done " << nRepeats << " iterations of knn500 query, took " << prettyDouble(t1-t0) << "s" << std::endl;
    std::cout << " that's " << prettyDouble((t1-t0)/nRepeats) << "s per query (avg)..." << std::endl;
    std::cout << " ... or " << prettyDouble(nQueries*nRepeats/(t1-t0)) << " queries/s" << std::endl;
  }

}
