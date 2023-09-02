#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "cukd/common.h"

//ACDEFGHIKLMNPQRSTVWY
void intarr2float(Float20* target, int index, int* i){
	target[index].x = i[0];
	target[index].b = i[1];
	target[index].c = i[2];
	target[index].d = i[3];
	target[index].e = i[4];
	//5
	target[index].f = i[5];
	target[index].g = i[6];
	target[index].h = i[7];
	target[index].i = i[8];
	target[index].j = i[9];
	//10
	target[index].k = i[10];
	target[index].l = i[11];
	target[index].m = i[12];
	target[index].n = i[13];
	target[index].o = i[14];
	//15
	target[index].p = i[15];
	target[index].q = i[16];
	target[index].r = i[17];
	target[index].s = i[18];
	target[index].t = i[19];
}

int* str2intarr(char* str, int str_len){
	int *ans=0;
	ans = (int*)calloc(20, sizeof(int));
	for(int i=0;i<str_len;i++){
		char c = str[i];
		if(c=='A')
			ans[0]++;
		else if(c=='C')
			ans[1]++;
		else if(c=='D')
			ans[2]++;
		else if(c=='E')
			ans[3]++;
		else if(c=='F')
			ans[4]++;
		//5
		else if(c=='G')
			ans[5]++;
		else if(c=='H')
			ans[6]++;
		else if(c=='I')
			ans[7]++;
		else if(c=='K')
			ans[8]++;
		else if(c=='L')
			ans[9]++;
		//10
		else if(c=='M')
			ans[10]++;
		else if(c=='N')
			ans[11]++;
		else if(c=='P')
			ans[12]++;
		else if(c=='Q')
			ans[13]++;
		else if(c=='R')
			ans[14]++;
		//15
		else if(c=='S')
			ans[15]++;
		else if(c=='T')
			ans[16]++;
		else if(c=='V')
			ans[17]++;
		else if(c=='W')
			ans[18]++;
		else
			ans[19]++;
	}
	return ans;
}

void test_util(){
	std::cout << "test_util\n";
	char x[] = "CAAD";
	int* temp = str2intarr(x,4);
	for(int i=0;i<20;i++)
		std::cout << temp[i] << "\n";
	Float20* temp2 = (Float20*)calloc(1, sizeof(Float20));
	intarr2float(temp2,0,temp);
	std::cout << temp2[0].x << "a\n";
	std::cout << temp2[0].b << "a\n";
	std::cout << temp2[0].c << "a\n";
	std::cout << temp2[0].d << "a\n";
}