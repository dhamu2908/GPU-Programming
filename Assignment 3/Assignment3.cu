// GPU Assignment-3 CS24M027 

#include <chrono>
#include <cuda.h>
#include <fstream>
#include <iostream>
#include <math.h>
#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <climits>

using namespace std;

#define MOD  1000000007

int BLOCK_SIZE = 1024;   // Number of threads per block

struct GraphEdge { int u, v, w; };  // Edge structure

// Union to store index and weight
union MinVal {
    struct {
        unsigned int index;
        unsigned int weight;
    };
    unsigned long long combined;
};

//Finding root of the element with path compression
__device__ int findRoot(int roots[], int x) {
    while (roots[x] != x) {
        int parent = roots[x];
        int grandparent = roots[parent];

        //compress path by making parent as root
        if (atomicCAS(&roots[x], parent, grandparent) == parent)
            x = grandparent;
        else x = roots[x];
        __threadfence_block();  // Barrier for threads in the same block
    }
    return x;
}

// Merging
__device__ void mergeTrees(int roots[], int sizes[], int a, int b) {
    a = findRoot(roots, a);
    b = findRoot(roots, b);
    if (a == b) return;   //Already in the same tree

    // Swapping Make sure smaller tree is attached to larger
    if (sizes[a] < sizes[b]) {
        int temp = a;
        a = b;
        b = temp;
    }

    //Update root of smaller tree
    atomicExch(&roots[b], a);

    //Update size of larger tree
    atomicAdd(&sizes[a], sizes[b]);
}

//Kernel to modify edge weights based on the type
__global__ void modifyWeights(GraphEdge* edges, int* types, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    //Using switch case to multiply weight based on edge type
    switch(types[i]) {
        case 1: edges[i].w *= 2; break;  // Green Edge
        case 2: edges[i].w *= 5; break;  // Traffic Edge
        case 3: edges[i].w *= 3; break;  // Dept Edge
    }
}

//Reset minimum values to ULLONG_MAX
__global__ void resetMins(unsigned long long* mins, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) mins[i] = ULLONG_MAX;
}

//Kernel to find minimum weight between two trees
__global__ void findMins(GraphEdge* edges, unsigned long long* mins, int* roots, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    GraphEdge e = edges[i];
    int ru = findRoot(roots, e.u);   //First node root
    int rv = findRoot(roots, e.v);   //Second node root

    //If roots are different update the minimum value
    if (ru != rv) {
        MinVal val;
        val.weight = e.w;
        val.index = i;
        atomicMin(&mins[ru], val.combined);
        atomicMin(&mins[rv], val.combined);
    }
}

// Process the minimum edges and accumulate total weight
__global__ void processMins(unsigned long long* mins, int* used, unsigned long long* total, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    unsigned long long minVal = mins[i];
    if (minVal == ULLONG_MAX) return;

    MinVal mval;
    mval.combined = minVal;
    unsigned idx = mval.index;
    unsigned weight = mval.weight;

    // Attempt to mark this edge as used if not already
    int old = atomicCAS(&used[idx], 0, 1);
    if (old == 0) {
        atomicAdd(total, weight);
    }
}

//Connect components based on minimum weight
__global__ void updateComponents(GraphEdge* edges, unsigned long long* mins,
                                int* roots, int* sizes, bool* flag, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n || roots[i] != i) return;

    unsigned long long minVal = mins[i];
    if (minVal == ULLONG_MAX) return;    //No edges to connect

    MinVal mval;
    mval.combined = minVal;
    GraphEdge e = edges[mval.index];  //Edge with minimum weight
    int r1 = findRoot(roots, e.u);
    int r2 = findRoot(roots, e.v);

    //If roots are different merge the trees
    if (r1 != r2) {
        mergeTrees(roots, sizes, r1, r2);
        *flag = true;
    }
}

// Kernel to count the number of components
__global__ void countComponents(int* roots, int* component_count, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    if (roots[i] == i) {
        atomicAdd(component_count, 1);
    }
}

//Performing modulo operation
__global__ void computeModulo(unsigned long long* total_ptr) {
    *total_ptr = *total_ptr % MOD;
}

int main() {

    //Input reading
    int N, M;
    cin >> N >> M;
    vector<GraphEdge> edges(M);
    vector<int> edgeTypes(M);

    //Edge reading
    for (int i = 0; i < M; ++i) {
        string t;
        int u, v, w;
        cin >> u >> v >> w >> t;
        edges[i].u = u;
        edges[i].v = v;
        edges[i].w = w;
        edgeTypes[i] = (t == "green") ? 1 : (t == "traffic") ? 2 : (t == "dept") ? 3 : 0;
    }

    //Allocating memory in GPU
    GraphEdge* d_edges;
    int *d_roots, *d_sizes, *d_types;
    unsigned long long* d_mins;
    bool* d_flag;
    int* d_used;
    unsigned long long* d_total;
    int* d_component_count;

    cudaMalloc(&d_edges, M*sizeof(GraphEdge));
    cudaMalloc(&d_roots, N*sizeof(int));
    cudaMalloc(&d_sizes, N*sizeof(int));
    cudaMalloc(&d_mins, N*sizeof(unsigned long long));
    cudaMalloc(&d_flag, sizeof(bool));
    cudaMalloc(&d_types, M*sizeof(int));
    cudaMalloc(&d_used, M*sizeof(int));
    cudaMalloc(&d_total, sizeof(unsigned long long));
    cudaMalloc(&d_component_count, sizeof(int));

    auto start = std::chrono::high_resolution_clock::now();

    //Initializing roots and sizes
    vector<int> hostRoots(N);
    for (int i = 0; i < N; ++i) hostRoots[i] = i;
    vector<int> hostSizes(N, 1);

    //Copying data to GPU
    cudaMemcpy(d_edges, edges.data(), M*sizeof(GraphEdge), cudaMemcpyHostToDevice);
    cudaMemcpy(d_roots, hostRoots.data(), N*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_sizes, hostSizes.data(), N*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_types, edgeTypes.data(), M*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemset(d_used, 0, M*sizeof(int));
    cudaMemset(d_total, 0, sizeof(unsigned long long));

    // Modify edge weights
    modifyWeights<<<(M+BLOCK_SIZE-1)/BLOCK_SIZE, BLOCK_SIZE>>>(d_edges, d_types, M);
    cudaDeviceSynchronize();

    int components = N;
    bool changed;

    //do - while loop with  kernel launchs 
    do {

        //Kernel launch to reset min
        resetMins<<<(N+BLOCK_SIZE-1)/BLOCK_SIZE, BLOCK_SIZE>>>(d_mins, N);
        cudaDeviceSynchronize();   //barrier

        //Kernel Launch to find mins
        findMins<<<(M+BLOCK_SIZE-1)/BLOCK_SIZE, BLOCK_SIZE>>>(d_edges, d_mins, d_roots, M);
        cudaDeviceSynchronize();  //barrier

        //Kernel launch to process mins
        processMins<<<(N+BLOCK_SIZE-1)/BLOCK_SIZE, BLOCK_SIZE>>>(d_mins, d_used, d_total, N);
        cudaDeviceSynchronize();

        changed = false;
        cudaMemcpy(d_flag, &changed, sizeof(bool), cudaMemcpyHostToDevice);
        updateComponents<<<(N+BLOCK_SIZE-1)/BLOCK_SIZE, BLOCK_SIZE>>>(d_edges, d_mins, d_roots, d_sizes, d_flag, N);
        cudaDeviceSynchronize();  //barrier

        cudaMemcpy(&changed, d_flag, sizeof(bool), cudaMemcpyDeviceToHost);

        cudaMemset(d_component_count, 0, sizeof(int));
        countComponents<<<(N+BLOCK_SIZE-1)/BLOCK_SIZE, BLOCK_SIZE>>>(d_roots, d_component_count, N);
        cudaDeviceSynchronize();

        //Cuda Mem copy 
        cudaMemcpy(&components, d_component_count, sizeof(int), cudaMemcpyDeviceToHost);

    } while (components > 1 && changed);

    unsigned long long total;
    
    //Kernel Launch to calculate modulo
    computeModulo<<<1, 1>>>(d_total);
    cudaDeviceSynchronize();

    //Cuda mem cpy for total
    cudaMemcpy(&total, d_total, sizeof(unsigned long long), cudaMemcpyDeviceToHost);

    cout << total<< "\n";

    //Ending the timer
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed1 = end - start;
    
    //cout << elapsed1.count() << " s\n";

    return 0;
}
