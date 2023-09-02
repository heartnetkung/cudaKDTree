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

typedef struct TrieNode TrieNode;
struct TrieNode {
    int data;
    TrieNode* children[10];
    int is_leaf;
    std::vector<char*> payload;
    std::vector<int> lengthOfPayload;
};

TrieNode make_trienode(int data) {
    // Allocate memory for a TrieNode
    TrieNode node;
    for (int i=0; i<10; i++)
        node.children[i] = NULL;
    node.is_leaf = 0;
    node.data=data;
    return node;
}

void test_trie(){
    std::cout << "test trie \n";
    TrieNode node = make_trienode(5);
    std::cout << node.data << "c \n";
}
