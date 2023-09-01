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
    if(i==N)
      break;

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

    i++;
    free(tmp);
  }

  fclose(stream);
  return d_points;
}

void writePoints(int nQueries, int nResult, int *d_results, Float20 *d_points)
{
  FILE* stream = fopen("output.txt", "w");
  for(int j=0;j<nQueries;j++){
    for(int k=0;k<nResult;k++){
      int index = d_results[j*nResult+k];
      if (index != -1){
        Float20 point = d_points[index];
        fprintf(stream,"%.0f %.0f %.0f %.0f %.0f %.0f %.0f %.0f %.0f %.0f %.0f %.0f %.0f %.0f %.0f %.0f %.0f %.0f %.0f %.0f %d\n",point.x,point.b,point.c,point.d,point.e,point.f,point.g,point.h,point.i,point.j,point.k,point.l,point.m,point.n,point.o,point.p,point.q,point.r,point.s,point.t,j);
      }
    }
  }
  fclose(stream);
}

int leven1(char* x, int lenX, char* y, int lenY){
  int diff = lenX-lenY;
  if (diff > 1 || diff<-1)
    return 0;

  char* shorter = x;
  char* longer = y;
  int lenShorter = lenX;
  if (diff>0){
    shorter = y;
    longer = x;
    lenShorter = lenY;
  }

  for(int i=0;i<lenShorter;i++){
    if(shorter[i] == longer[i])
      continue;
    if(diff==0){
      for(int j=i+1;j<lenShorter;j++)
        if(shorter[j]!=longer[j])
          return 0;
      return 1;
    }else{
      for(int j=i;j<lenShorter;j++)
        if(shorter[j]!=longer[j+1])
          return 0;
      return 1;
    }
  }
  return 1;
}

// ==================================================================

int main(int ac, const char **av)
{
  char x[6] = "555";
  char y[6] = "55555";
  int lenX = 3;
  int lenY = 5;
  std::cout << leven1(x,lenX,y,lenY) << "leven1\n";

  x[] = "555";
  y[] = "666";
  lenX = 3;
  lenY = 3;
  std::cout << leven1(x,lenX,y,lenY) << "leven2\n";

  x[] = "555";
  y[] = "556";
  lenX = 3;
  lenY = 3;
  std::cout << leven1(x,lenX,y,lenY) << "leven3\n";

  x[] = "555";
  y[] = "55";
  lenX = 3;
  lenY = 2;
  std::cout << leven1(x,lenX,y,lenY) << "leven4\n";

  x[] = "555";
  y[] = "5555";
  lenX = 3;
  lenY = 4;
  std::cout << leven1(x,lenX,y,lenY) << "leven5\n";


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
  
  Float20 *d_points = readPoints(nPoints);
  //TODO read file only once?
  Float20 *d_queries = readPoints(nPoints);

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
  writePoints(nQueries, nResult, d_results, d_points);
  std::cout << "success\n";
  // for(int j=0;j<nQueries;j++){
  //   std::cout << "j: " << j << " \n";
  //   for(int k=0;k<nResult;k++){
  //     int index = d_results[j*nResult+k];
  //     if (index != -1){
  //       Float20 point = d_points[index];
  //       std::cout << " closest point is " << point.x << " \n";
  //     }
  //   }
  // }

}
