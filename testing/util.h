#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "cukd/common.h"

//ACDEFGHIKLMNPQRSTVWY
Float20 intarr2float(int* i){
	Float20 ans = {(float)i[0],(float)i[1],(float)i[2],(float)i[3],(float)i[4],
	(float)i[5],(float)i[6],(float)i[7],(float)i[8],(float)i[9],
	(float)i[10],(float)i[11],(float)i[12],(float)i[13],(float)i[14],
	(float)i[15],(float)i[16],(float)i[17],(float)i[18],(float)i[19]};
	return ans;
}

int* str2intarr(char* str, int str_len){
	int ans[20]={0};
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
	int* temp = str2intarr(x);
	for(int i=0;i<20;i++)
		std::cout << temp[i] << "\n";
	Float20 temp2 = intarr2float(temp);
	std::cout << temp2.x << "a\n";
	std::cout << temp2.b << "a\n";
	std::cout << temp2.c << "a\n";
	std::cout << temp2.d << "a\n";
}