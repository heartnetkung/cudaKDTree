#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "cukd/builder.h"
#include "trie.h"
#include "util.h"
#include "leven1.h"

struct FileContent{
	Float20* d_points;
	Float20* d_queries;
	char** str_queries;
	int* str_len_queries;
	TrieNode* str_queries_index;
};

FileContent readContent(int n){
	FileContent ans;
	FILE* stream = fopen("input2.txt", "r");
	char line[200];
	int i = 0;
	int* intarr;
	int len_temp;
	char* tmp;

	ans.str_queries = (char**) malloc(n*sizeof(char*));
	ans.str_len_queries = (int*) malloc(n*sizeof(int));
	CUKD_CUDA_CALL(MallocManaged((void**)&ans.d_points,n*sizeof(Float20)));
	CUKD_CUDA_CALL(MallocManaged((void**)&ans.d_queries,n*sizeof(Float20)));
	ans.str_queries_index = make_trienode();

	while (fgets(line, 200, stream)){
		if(i==n)
			break;

		tmp = strdup(line);
		len_temp = strlen(tmp);
		ans.str_queries[i] = tmp;
		ans.str_len_queries[i] = len_temp;
		intarr = str2intarr(tmp,len_temp);
		intarr2float(ans.d_points,i,intarr);
		intarr2float(ans.d_queries,i,intarr);
		insert_trie(ans.str_queries_index,intarr,i);

		i++;
	}

	fclose(stream);
	return ans;
}

std::vector<int> postprocessing(Float20* d_results, FileContent content, int n, int nResult){
	std::vector<int> ans, indices;
	TrieNode* root = content.str_queries_index;
	char** str_queries = content.str_queries;
	int* str_len_queries = content.str_len_queries;
	TrieNode* result;
	char* str1, str2;
	int strLen1, strLen2, index2;
	Float20* d_points = content.d_points;

	for(int j=0;j<n;j++){
		str1 = str_queries[j];
		strLen1 = str_len_queries[j];

		for(int k=0;k<nResult;k++){
			int index = d_results[j*nResult+k];
			if(index == -1)
				continue;

			result = search_trie(root,float2intarr(d_points[index]));
			indices = result->payload;

			for(int i=0;i<indices.size();i++){
				index2 = indices.at(i);
				if(index2==j)
					continue;

				str2 = str_queries[index2];
				strLen2 = str_len_queries[index2];
				if(leven1(str1,strLen1,str2,strLen2)==1){
					ans.push_back(j);
					ans.push_back(index2);
				}
			}
		}
	}
	return ans;
}

void writeFile(std::vector<int> finalResult){
	FILE* stream = fopen("output.txt", "w");
	for(int i=0;i<finalResult.size();i+=2)
		fprintf(stream,"%d %d",finalResult.at(i),finalResult.at(i+1));
	fclose(stream);
}