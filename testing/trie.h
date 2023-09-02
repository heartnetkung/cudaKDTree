/**
    Code for https://journaldev.com article
    Purpose: A Trie Data Structure Implementation in C
    @author: Vijay Ramachandran
    @date: 20-02-2020
*/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <vector>

// The number of children for each node
// We will construct a N-ary tree and make it
// a Trie
// Since we have at most 10 repeated amino acids
// #define N 10

struct TrieNode {
    TrieNode* children[10];
    std::vector<int> payload;
};

TrieNode* make_trienode() {
    // Allocate memory for a TrieNode
    TrieNode* node = (TrieNode*)calloc(1, sizeof(TrieNode));
    for (int i=0; i<10; i++)
        node->children[i] = NULL;
    return node;
}

void insert_trie(TrieNode* root, int* intarr,int payload){
    TrieNode* current = root;
    for(int i=0;i<20;i++){
        int data = intarr[i];
        if(current->children[data]==NULL)
            current->children[data]=make_trienode();
        current = current->children[data];
    }
    current->payload.push_back(payload);
}

TrieNode* search_trie(TrieNode* root, int* intarr){
    TrieNode* current = root;
    for(int i=0;i<20;i++){
        int data = intarr[i];
        if(current->children[data]==NULL)
            return NULL;
        current = current->children[data];
    }
    return current;
}

void printNode(TrieNode* node){
    for(int i=0;i<node->payload.size();i++)
        std::cout << node->payload.at(i) << "\n";
}

void test_trie(){
    std::cout << "test trie \n";
    TrieNode* root = make_trienode();

    int* intarr1 = (int*)calloc(20,sizeof(int));
    intarr1[0]=2;
    intarr1[1]=1;
    int payload1 = 1;
    insert_trie(root,intarr1,payload1);

    int* intarr2 = (int*)calloc(20,sizeof(int));
    intarr2[0]=2;
    intarr2[1]=1;
    int payload2 = 2;
    insert_trie(root,intarr2,payload2);

    int* intarr3 = (int*)calloc(20,sizeof(int));
    intarr3[0]=5;
    int payload3 = 3;
    insert_trie(root,intarr3,payload3);

    std::cout << "test1 \n";
    TrieNode* test1 = search_trie(root,intarr2);
    printNode(test1);

    std::cout << "test2 \n";
    TrieNode* test2 = search_trie(root,intarr3);
    printNode(test2);

    int* intarr4 = (int*)calloc(20,sizeof(int));
    std::cout << "test3 \n";
    TrieNode* test3 = search_trie(root,intarr4);
    printNode(test3);
}
