#pragma once

#include <stdbool.h>

typedef struct BlockHeader {
    struct BlockHeader *prev;
    struct BlockHeader *next;
    unsigned long size;
    bool free;
} BlockHeader;

typedef struct BlockFooter {
    BlockHeader *headerPtr;
} BlockFooter;

typedef struct MemoryPool {
    BlockHeader *freeList;
    BlockHeader *usedList;
    char *memPool;
} MemoryPool;

__host__ void *allocatePools(unsigned int numThreads);

__host__ void freePools(void *ptr);

__device__ void poolinit(void *poolBlockPtr, unsigned int threadInd);

__device__ void *poolcalloc(unsigned long count, unsigned long size);

__device__ void *callocwrapper(unsigned long count, unsigned long size);

__device__ void *poolmalloc(unsigned long size);

__device__ void poolfree(void *ptr);

__device__ void printlayout();

__device__ void printbytes();

__device__ int dataBytes(BlockHeader *head);

__device__ int headerBytes(BlockHeader *head);

__device__ int getThreadIndex();
