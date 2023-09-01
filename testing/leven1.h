#include <stdio.h>
#include <stdlib.h>
#include <string.h>


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

void test_leven(){
  char x[] = "555";
  char y[] = "55555";
  int lenX = 3;
  int lenY = 5;
  std::cout << leven1(x,lenX,y,lenY) << "leven1\n";

  char x2[] = "555";
  char y2[] = "666";
  lenX = 3;
  lenY = 3;
  std::cout << leven1(x2,lenX,y2,lenY) << "leven2\n";

  char x3[] = "555";
  char y3[] = "556";
  lenX = 3;
  lenY = 3;
  std::cout << leven1(x3,lenX,y3,lenY) << "leven3\n";

  char x4[] = "555";
  char y4[] = "55";
  lenX = 3;
  lenY = 2;
  std::cout << leven1(x4,lenX,y4,lenY) << "leven4\n";

  char x5[] = "555";
  char y5[] = "5555";
  lenX = 3;
  lenY = 4;
  std::cout << leven1(x5,lenX,y5,lenY) << "leven5\n";
}