#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "cukd/builder.h"
#include "trie.h"
#include "util.h"

struct FileContent{
	Float20* d_points;
	Float20* d_queries;
	char* str_queries;
	int* str_len_queries;
	TrieNode* str_queries_index;
}

FileContent readContent(int n){
	using namespace cukd::common;
	FileContent ans;
	FILE* stream = fopen("input2.txt", "r");
	char line[200];
	int i = 0;
	int* intarr;
	int len_temp;

	char* str_queries = (char*) malloc(n*sizeof(char));
	int* str_len_queries = (int*) malloc(n*sizeof*(int));
	CUKD_CUDA_CALL(MallocManaged((void**)&ans.d_points,N*sizeof(Float20)));
	CUKD_CUDA_CALL(MallocManaged((void**)&ans.d_queries,N*sizeof(Float20)));
	ans.str_queries_index = make_trienode();

	while (fgets(line, 200, stream)){
		if(i==N)
			break;

		char* tmp = strdup(line);
		int len_temp = strlen(tmp);
		str_queries[i] = tmp;
		str_len_queries[i] = len_temp;
		intarr = str2intarr(tmp,len_temp);
		intarr2float(ans.d_points,i,intarr);
		intarr2float(ans.d_queries,i,intarr);
		insert_trie(root,intarr,tmp,len_temp);

		i++;
	}

	fclose(stream);
	return ans;
}


