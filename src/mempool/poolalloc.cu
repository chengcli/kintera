#include <assert.h>
#include <cstdio>
#include <cstdlib>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include <configure.h>
#include "poolalloc.cuh"

// Alignment
// For now, headers are 16 aligned
// H = Header
// P = Padding
// D = Data
// 0   16  32  48  64
// |P|H|D|D|P|H| | | |
// | | | | | | | | | |

// blockheader type to ensure first one is aligned
// TODO: when switching to cuda, remove malloc() in poolinit function
// using malloc because malloc aligns to 16 on my system
// static BlockHeader memPool[(GPU_MEM_POOL_SIZE/sizeof(BlockHeader))+1];

// TODO: placeholder for now, might be able to reduce memory overhead
//__device__ MemoryPool* g_memoryPools[MEM_MAX_THREADS];
__device__ MemoryPool* g_memoryPools;

__device__ static BlockFooter *getBlockFooter(BlockHeader *header) {
    return (BlockFooter *)((char *)header + sizeof(BlockHeader) + header->size);
}

// NOTE:: Debug functions
__device__ static int debugListSize(BlockHeader *head) {
    int size = 0;
    BlockHeader *current = head;
    while (current) {
        size++;
        current = current->next;
    }
    return size;
}

__device__ static void assertNoSizeOverflow(BlockHeader *head) {
    while (head) {
        assert(head->size < GPU_MEM_POOL_SIZE);
        head = head->next;
    }
}

__device__ static void assertNoCycle(BlockHeader *head) {
    if (head == NULL) {
        return;
    }
    BlockHeader *slow = head;
    BlockHeader *fast = head;
    while (fast && fast->next) {
        slow = slow->next;
        fast = fast->next->next;
        assert(slow != fast);
    }
}

__device__ static void assertListValid(BlockHeader *head) {
    assertNoCycle(head);
    BlockHeader *current = head;
    while (current) {
        if (current->next) {
            assert(current->next->prev == current);
        }
        if (current->prev) {
            assert(current->prev->next == current);
        } else {
            assert(current == head);
        }
        assert(head->size % 16 == 0);
        current = current->next;
    }
}

__device__ static void assertFreeListSorted(BlockHeader *head) {
    while (head && head->next) {
        assert(head->size >= head->next->size);
        assert(head->next->prev == head);
        head = head->next;
    }
}

__device__ static void assertFootersValid(BlockHeader *head) {
    while (head) {
        // BlockFooter *footer =
            // (BlockFooter *)(tmp + sizeof(BlockHeader) + head->size);
        BlockFooter *footer = getBlockFooter(head);
        assert(footer->headerPtr == head);
        head = head->next;
    }
}

__device__ static void listRemove(BlockHeader **head, BlockHeader *target) {
    BlockHeader *front = target->next;
    BlockHeader *back = target->prev;

    if (front) {
        front->prev = back;
    }
    if (back) {
        back->next = front;
    } else {
        *head = front;
    }
}

__device__ static void listPrepend(BlockHeader **head, BlockHeader *added) {
    if (*head == NULL) {
        *head = added;
        added->prev = NULL;
        added->next = NULL;
        return;
    }
    (*head)->prev = added;
    added->next = *head;
    added->prev = NULL;
    *head = added;
}

__device__ static void listSwapHeadSort(BlockHeader **head) {
    if ((*head)->next == NULL) {
        return;
    }
    BlockHeader *front = (*head)->next->next;
    BlockHeader *back = (*head)->next;
    unsigned long targetSize = (*head)->size;

    while (front) {
        unsigned long frontSize = front->size;
        unsigned long backSize = back->size;
        if (frontSize <= targetSize && targetSize <= backSize) {
            front->prev = *head;
            back->next = *head;
            BlockHeader *tmp = (*head)->next;
            (*head)->prev = back;
            (*head)->next = front;
            (*head) = tmp;
            (*head)->prev = NULL;
            return;
        }
        front = front->next;
        back = back->next;
    }
    // not found, insert at end, if statement is for special case where list is
    // length 2
    if (back->size > (*head)->size) {
        BlockHeader *tmp = (*head)->next;
        back->next = *head;
        (*head)->prev = back;
        (*head)->next = NULL;
        *head = tmp;
        (*head)->prev = NULL;
    }
}


__device__ int dataBytes(BlockHeader *head) {
    int tally = 0;
    while (head) {
        tally += head->size;
        head = head->next;
    }
    return tally;
}

__device__ int headerBytes(BlockHeader *head) {
    int tally = 0;
    while (head) {
        tally += sizeof(BlockHeader);
        head = head->next;
    }
    return tally;
}

__device__ static int getFooterAlignedSize() {
    // return sizeof(BlockFooter) + (16 - sizeof(BlockFooter) % 16);
    return ((sizeof(BlockFooter) + 15) / 16) * 16;
}

__device__ static BlockHeader *getNextBlockHeader(BlockHeader *header, unsigned int threadInd) {
    char *memPool = g_memoryPools[threadInd].memPool;
    int headerOffset = (char *)header - (char *)memPool;
    if (headerOffset + sizeof(BlockHeader) + header->size +
            getFooterAlignedSize() >= GPU_MEM_POOL_SIZE) {
        return NULL;
    }

    BlockHeader *nextBlockHeader =
        (BlockHeader *)((char *)header + sizeof(BlockHeader) + header->size +
                        getFooterAlignedSize());
    return nextBlockHeader;
}

__device__ static BlockHeader *getPrevBlockHeader(BlockHeader *header, unsigned int threadInd) {
    char *memPool = g_memoryPools[threadInd].memPool;
    if ((void*)header == (void*)memPool) {
        return NULL;
    }
    BlockFooter *prevBlockFooter =
        (BlockFooter *)((char *)header - getFooterAlignedSize());
    BlockHeader *prevBlockHeader = prevBlockFooter->headerPtr;
    return prevBlockHeader;
}

__host__ void *allocatePools(unsigned int numThreads) {
    // Allocate g_memoryPools array
    MemoryPool* d_arr = nullptr;
    cudaError_t err = cudaMalloc(&d_arr, sizeof(MemoryPool) * numThreads);
    if (err != cudaSuccess) {
        printf("Error allocating memory for g_memoryPools: %s\n", cudaGetErrorString(err));
        return nullptr;
    }
    cudaMemcpyToSymbol(g_memoryPools, &d_arr, sizeof(d_arr));

    // Allocate a memory pool for each thread
    char *allocatedBlock;
    err = cudaMalloc(&allocatedBlock, GPU_MEM_POOL_SIZE * numThreads);
    if (err != cudaSuccess) {
        printf("Error allocating memory pool: %s\n", cudaGetErrorString(err));
        return nullptr;
    }
    return allocatedBlock;
}

__host__ void freePools(void *ptr) {
    // free g_memoryPools array
    MemoryPool* d_arr = nullptr;
    cudaMemcpyFromSymbol(&d_arr, g_memoryPools, sizeof(d_arr));
    cudaFree(d_arr);

    // free the memory pool
    cudaFree(ptr);
}

__device__ void poolinit(void *poolBlockPtr, unsigned int threadInd) {
    g_memoryPools[threadInd].memPool = (char *)poolBlockPtr + (threadInd * GPU_MEM_POOL_SIZE);
    char *memPool = g_memoryPools[threadInd].memPool;
    BlockHeader *&freeList = g_memoryPools[threadInd].freeList;

    BlockHeader *header = (BlockHeader *)memPool;
    // create a block that fills entire pool
    unsigned long dataSize =
        GPU_MEM_POOL_SIZE - sizeof(BlockHeader) -
        getFooterAlignedSize();
    unsigned long dataSizeAligned = dataSize - dataSize % 16;
    header->size = dataSizeAligned;
    header->free = true;
    header->next = NULL;
    header->prev = NULL;

    BlockFooter *footer = getBlockFooter(header);
    footer->headerPtr = header;
    listPrepend(&freeList, header);
}

__device__ void *poolcalloc(unsigned long count, unsigned long size) {
    void *ptr = poolmalloc(count * size);
    memset(ptr, 0, count * size);
    return ptr;
}

__device__ void *callocwrapper(unsigned long count, unsigned long size) {
    void *ptr = malloc(count * size);
    memset(ptr, 0, count * size);
    return ptr;
}

__device__ void *poolmalloc(unsigned long size) {
    // all pointer arithmetic is done on char* for clarity
    // TODO: handle edge case where last alloc takes space of last header
    //unsigned int threadInd = blockIdx.x * blockDim.x + threadIdx.x;
    // global thread id
    unsigned int threadInd = getThreadIndex();
    
    BlockHeader *&freeList = g_memoryPools[threadInd].freeList;
    BlockHeader *&usedList = g_memoryPools[threadInd].usedList;

    if (freeList == NULL || freeList->size < size) {
        return NULL;
    }

    int initFreeListSize = debugListSize(freeList);
    int initUsedListSize = debugListSize(usedList);

    unsigned long oldBlockSize = freeList->size;
    BlockHeader *newAllocatedHeader = freeList;
    // unsigned long dataSizeToAllocate = size + (16 - size % 16);
    unsigned long dataSizeToAllocate = ((size + 15) / 16) * 16;
    unsigned long totalSizeUnaligned = dataSizeToAllocate + sizeof(BlockFooter);
    unsigned long totalSizeAligned =
        // totalSizeUnaligned + (16 - totalSizeUnaligned % 16);
        ((totalSizeUnaligned + 15) / 16) * 16;
    newAllocatedHeader->size = dataSizeToAllocate;
    BlockFooter *newAllocatedFooter = getBlockFooter(newAllocatedHeader);
    newAllocatedFooter->headerPtr = newAllocatedHeader;
    listRemove(&freeList, freeList);
    listPrepend(&usedList, newAllocatedHeader);
    newAllocatedHeader->free = false;
    BlockHeader *newFreeHeader =
        (BlockHeader *)((char *)newAllocatedHeader +
                        (sizeof(BlockHeader) + totalSizeAligned));
    // FIXME: this was wrong before, check this later too (totalSizeAligned vs dataSizeToAllocate)
    newFreeHeader->size =
        oldBlockSize - (totalSizeAligned + sizeof(BlockHeader));
    newFreeHeader->free = true;
    BlockFooter *newFreeFooter = getBlockFooter(newFreeHeader);
    newFreeFooter->headerPtr = newFreeHeader;
    listPrepend(&freeList, newFreeHeader);
    listSwapHeadSort(&freeList);
    void *res = (char *)newAllocatedHeader + (sizeof(BlockHeader));

    assertListValid(freeList);
    assertListValid(usedList);
    assertFootersValid(usedList);
    assertFootersValid(freeList);
    assertNoSizeOverflow(freeList);
    assertNoSizeOverflow(usedList);
    assert(debugListSize(freeList) == initFreeListSize);
    assert(debugListSize(usedList) == initUsedListSize + 1);
    assertFreeListSorted(freeList);
    return res;
}

__device__ void poolfree(void *ptr) {
    //unsigned int threadInd = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int threadInd = getThreadIndex();

    BlockHeader *&freeList = g_memoryPools[threadInd].freeList;
    BlockHeader *&usedList = g_memoryPools[threadInd].usedList;

    int initFreeListSize = debugListSize(freeList);
    int initUsedListSize = debugListSize(usedList);

    if (usedList->next == NULL) {
        freeList = NULL;
        usedList = NULL;
        poolinit(g_memoryPools[0].memPool, threadInd);
        return;
    }

    char *poolPtr = (char *)ptr;
    BlockHeader *freedHeader = (BlockHeader *)(poolPtr - sizeof(BlockHeader));
    listRemove(&usedList, freedHeader);

    bool canCoalesceForwards = false;
    BlockHeader *prevBlockHeader = getPrevBlockHeader(freedHeader, threadInd);
    if (prevBlockHeader == NULL) {
        canCoalesceForwards = false;
    } else if (prevBlockHeader->free) {
        canCoalesceForwards = true;
    } else {
        canCoalesceForwards = false;
    }

    BlockHeader *newFreeHeader = NULL;
    if (canCoalesceForwards) {
        // coalesce forwards
        BlockFooter *freedFooter =
            (BlockFooter *)((char *)freedHeader + freedHeader->size +
                            sizeof(BlockHeader));
        assert(freedFooter->headerPtr == freedHeader);
        prevBlockHeader->size =
            (char *)freedFooter - (char *)prevBlockHeader - sizeof(BlockHeader);
        freedFooter->headerPtr = prevBlockHeader;

        assert(((BlockFooter *)((char *)prevBlockHeader +
                                (sizeof(BlockHeader) + prevBlockHeader->size)))
                   ->headerPtr == prevBlockHeader);

        newFreeHeader = prevBlockHeader;

        listRemove(&freeList, prevBlockHeader);
        listPrepend(&freeList, prevBlockHeader);
        listSwapHeadSort(&freeList);

        assert(debugListSize(freeList) == initFreeListSize);
        assert(debugListSize(usedList) == initUsedListSize - 1);
    } else {
        // no coalesce, add to freelist
        listPrepend(&freeList, freedHeader);
        freedHeader->free = true;
        listSwapHeadSort(&freeList);
        newFreeHeader = freedHeader;

        assert(debugListSize(usedList) == initUsedListSize - 1);
    }
    
    BlockHeader *forwardsBlockHeader = getNextBlockHeader(newFreeHeader, threadInd);
    bool canCoalesceBackwards = false;
    if (forwardsBlockHeader == NULL) {
        canCoalesceBackwards = false;
    } else if (forwardsBlockHeader->free) {
        canCoalesceBackwards = true;
    } else {
        canCoalesceBackwards = false;
    }
    
    if (canCoalesceBackwards) {
        // coalesce backwards
        listRemove(&freeList, forwardsBlockHeader);
        BlockFooter *coalescedFooter = getBlockFooter(forwardsBlockHeader);
        coalescedFooter->headerPtr = newFreeHeader;
        newFreeHeader->size = (char*)coalescedFooter - (char*)newFreeHeader - sizeof(BlockHeader);

        listRemove(&freeList, newFreeHeader);
        listPrepend(&freeList, newFreeHeader);
        listSwapHeadSort(&freeList);

        assert(debugListSize(usedList) == initUsedListSize - 1);
    }

    assertListValid(freeList);
    assertListValid(usedList);
    assertFootersValid(freeList);
    assertFootersValid(usedList);
    assertFreeListSorted(freeList);
}

__device__ int BlockHeaderPtrLess(const void *a, const void *b) {
    BlockHeader **aptr = (BlockHeader **)a;
    BlockHeader **bptr = (BlockHeader **)b;
    return *aptr - *bptr;
}

__device__ void selectionSort(BlockHeader **arr, int n) {
    for (int i = 0; i < n - 1; i++) {
        int min_idx = i;
        for (int j = i + 1; j < n; j++) {
            if (BlockHeaderPtrLess(&arr[j], &arr[min_idx]) < 0) {
                min_idx = j;
            }
        }
        if (min_idx != i) {
            BlockHeader *temp = arr[i];
            arr[i] = arr[min_idx];
            arr[min_idx] = temp;
        }
    }
}

// array based
/*
free: | 1028 |       | 8 \
used: |      | 8 | 8 |
*/

// NOTE: is likely very slow because of single threaded selection sort, but keeping for simplicity for now
__device__ void printlayout() {
    //unsigned int threadInd = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int threadInd = getThreadIndex();

    BlockHeader *&freeList = g_memoryPools[threadInd].freeList;
    BlockHeader *&usedList = g_memoryPools[threadInd].usedList;

    BlockHeader const *headers[MEM_MAX_BLOCKS];
    for (int i = 0; i < MEM_MAX_BLOCKS; i++) {
        headers[i] = NULL;
    }

    int numHeaders = 0;
    BlockHeader *freeListTraverse = freeList;
    BlockHeader *usedListTraverse = usedList;
    while (freeListTraverse) {
        headers[numHeaders] = freeListTraverse;
        numHeaders++;
        freeListTraverse = freeListTraverse->next;
    }
    while (usedListTraverse) {
        headers[numHeaders] = usedListTraverse;
        numHeaders++;
        usedListTraverse = usedListTraverse->next;
    }

    // sorts headers based on address as they are usually sorted by size or
    // recency
    // NOTE: not sure why LSP wants me to cast to BlockHeader** manually
    selectionSort((BlockHeader**)headers, numHeaders);

    printf("THREAD %d:\n", threadInd);
    printf("Memory Layout (total size %lu), size not incl headers:\n",
           GPU_MEM_POOL_SIZE);
    printf("free: |");
    for (int i = 0; i < numHeaders; i++) {
        if (headers[i]->free) {
            printf(" %lu |", headers[i]->size);
        } else {
            int numberLen =
                (int)(log((double)headers[i]->size) / log((double)10)) + 1;
            printf(" ");
            for (int i = 0; i < numberLen; i++) {
                printf(" ");
            }
            printf(" |");
        }
    }
    printf("\n");
    printf("used: |");
    for (int i = 0; i < numHeaders; i++) {
        if (!headers[i]->free) {
            printf(" %lu |", headers[i]->size);
        } else {
            int numberLen =
                (int)(log((double)headers[i]->size) / log((double)10)) + 1;
            printf(" ");
            for (int i = 0; i < numberLen; i++) {
                printf(" ");
            }
            printf(" |");
        }
    }
    printf("\n");
}

__device__ void printbytes() {
    unsigned int threadInd = getThreadIndex();
    char *memPool = g_memoryPools[threadInd].memPool;
    printf("\nBytes:\n");
    for (size_t i = 0; i < GPU_MEM_POOL_SIZE; i++) {
        printf("%02X", memPool[i]);
        if (i < GPU_MEM_POOL_SIZE - 1) {
            printf(" ");
        }
    }
    printf("\n");
}

__device__ int getThreadIndex() {
    int tid_x = blockIdx.x * blockDim.x + threadIdx.x;
    int tid_y = blockIdx.y * blockDim.y + threadIdx.y;
    int tid_z = blockIdx.z * blockDim.z + threadIdx.z;

    int width  = gridDim.x * blockDim.x;
    int height = gridDim.y * blockDim.y;

    return tid_z * width * height + tid_y * width + tid_x;
}
