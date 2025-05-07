#include <stdio.h>
#include <string.h>
#include <cuda_runtime.h>

#define CHARSET "abcdefghijklmnopqrstuvwxyz"
#define CHARSET_LEN 26
#define MIN_LEN 2
#define MAX_LEN 9
#define TILE_SIZE 256
#define PASSWORDS_PER_THREAD 32

__device__ bool is_match(const char* attempt, const char* target, int len) {
    for (int i = 0; i < len; i++) {
        if (attempt[i] != target[i]) return false;
    }
    return target[len] == '\0';
}

// Naive kernel implementation
__global__ void crack_password_naive(const char* target, bool* found, char* result, int len, unsigned long long offset) {
    unsigned long long idx = blockIdx.x * blockDim.x + threadIdx.x + offset;

    if (*found) return;

    char attempt[MAX_LEN + 1];
    attempt[len] = '\0';

    unsigned long long n = idx;
    for (int i = len - 1; i >= 0; i--) {
        attempt[i] = CHARSET[n % CHARSET_LEN];
        n /= CHARSET_LEN;
    }

    if (is_match(attempt, target, len)) {
        *found = true;
        for (int i = 0; i < len; i++) {
            result[i] = attempt[i];
        }
    }
}

// Shared memory tiling kernel implementation
__global__ void crack_password_shared(const char* target, bool* found, char* result, int len, unsigned long long offset) {
    unsigned long long idx = blockIdx.x * blockDim.x + threadIdx.x + offset;
    int tid = threadIdx.x;
    
    // Shared memory for the charset and target
    __shared__ char shared_charset[CHARSET_LEN];
    __shared__ char shared_target[MAX_LEN + 1];
    __shared__ bool shared_found;
    
    // Load charset and target into shared memory
    if (tid < CHARSET_LEN) {
        shared_charset[tid] = CHARSET[tid];
    }
    
    if (tid < len + 1) {
        shared_target[tid] = target[tid];
    }
    
    if (tid == 0) {
        shared_found = *found;
    }
    
    __syncthreads();
    
    if (shared_found) return;
    
    char attempt[MAX_LEN + 1];
    attempt[len] = '\0';
    
    unsigned long long n = idx;
    for (int i = len - 1; i >= 0; i--) {
        attempt[i] = shared_charset[n % CHARSET_LEN];
        n /= CHARSET_LEN;
    }
    
    if (is_match(attempt, shared_target, len)) {
        shared_found = true;
        *found = true;
        for (int i = 0; i < len; i++) {
            result[i] = attempt[i];
        }
    }
}

// Register tiling kernel implementation (your existing code)
__global__ void crack_password_register(const char* target, bool* found, char* result, int len, unsigned long long offset) {
    unsigned long long base_idx = (blockIdx.x * blockDim.x + threadIdx.x) * PASSWORDS_PER_THREAD + offset;
    
    // Early exit if password already found
    if (*found) return;
    
    // Load target into registers for faster access
    char reg_target[MAX_LEN + 1];
    for (int i = 0; i <= len; i++) {
        reg_target[i] = target[i];
    }
    
    // Each thread processes multiple password combinations
    for (int p = 0; p < PASSWORDS_PER_THREAD; p++) {
        unsigned long long idx = base_idx + p;
        
        char attempt[MAX_LEN + 1];
        attempt[len] = '\0';
        
        unsigned long long n = idx;
        for (int i = len - 1; i >= 0; i--) {
            attempt[i] = CHARSET[n % CHARSET_LEN];
            n /= CHARSET_LEN;
        }
        
        if (is_match(attempt, reg_target, len)) {
            *found = true;
            for (int i = 0; i < len; i++) {
                result[i] = attempt[i];
            }
            return;  // Exit early once found
        }
    }
}

// Function to run naive approach
void run_naive_approach(const char* host_target) {
    char* dev_target = 0;
    bool* dev_found = 0;
    char* dev_result = 0;
    bool host_found = false;
    char host_result[MAX_LEN + 1] = "";
    
    cudaMalloc(&dev_target, strlen(host_target) + 1);
    cudaMalloc(&dev_found, sizeof(bool));
    cudaMalloc(&dev_result, MAX_LEN + 1);
    
    cudaMemcpy(dev_target, host_target, strlen(host_target) + 1, cudaMemcpyHostToDevice);
    cudaMemcpy(dev_found, &host_found, sizeof(bool), cudaMemcpyHostToDevice);
    
    int threads = 256;
    
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    cudaEventRecord(start);
    
    for (int len = MIN_LEN; len <= MAX_LEN; len++) {
        unsigned long long max_combos = 1;
        for (int i = 0; i < len; i++) max_combos *= CHARSET_LEN;
        
        unsigned long long total = max_combos;
        unsigned long long offset = 0;
        
        unsigned long long chunk = 10000000;
        while (offset < total && !host_found) {
            unsigned long long remaining = total - offset;
            unsigned long long this_chunk = (remaining > chunk) ? chunk : remaining;
            unsigned long long block_count = (this_chunk + threads - 1) / threads;
            
            crack_password_naive<<<block_count, threads>>>(dev_target, dev_found, dev_result, len, offset);
            cudaDeviceSynchronize();
            
            cudaMemcpy(&host_found, dev_found, sizeof(bool), cudaMemcpyDeviceToHost);
            if (host_found) break;
            
            offset += this_chunk;
        }
        
        if (host_found) break;
    }
    
    if (host_found) {
        cudaMemcpy(host_result, dev_result, strlen(host_target), cudaMemcpyDeviceToHost);
        host_result[strlen(host_target)] = '\0';
        printf("Naive: Password %s found with length %d\n", host_result, strlen(host_result));
    } else {
        printf("Naive: Password not found in 2-9 lowercase chars.\n");
    }
    
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    
    printf("Naive GPU time to crack: %.4f seconds\n", milliseconds / 1000.0f);
    
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    
    cudaFree(dev_target);
    cudaFree(dev_found);
    cudaFree(dev_result);
}

// Function to run shared memory approach
void run_shared_memory_approach(const char* host_target) {
    char* dev_target = 0;
    bool* dev_found = 0;
    char* dev_result = 0;
    bool host_found = false;
    char host_result[MAX_LEN + 1] = "";
    
    cudaMalloc(&dev_target, strlen(host_target) + 1);
    cudaMalloc(&dev_found, sizeof(bool));
    cudaMalloc(&dev_result, MAX_LEN + 1);
    
    cudaMemcpy(dev_target, host_target, strlen(host_target) + 1, cudaMemcpyHostToDevice);
    cudaMemcpy(dev_found, &host_found, sizeof(bool), cudaMemcpyHostToDevice);
    
    int threads = TILE_SIZE;
    
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    cudaEventRecord(start);
    
    for (int len = MIN_LEN; len <= MAX_LEN; len++) {
        unsigned long long max_combos = 1;
        for (int i = 0; i < len; i++) max_combos *= CHARSET_LEN;
        
        unsigned long long total = max_combos;
        unsigned long long offset = 0;
        
        unsigned long long chunk = 10000000;
        while (offset < total && !host_found) {
            unsigned long long remaining = total - offset;
            unsigned long long this_chunk = (remaining > chunk) ? chunk : remaining;
            unsigned long long block_count = (this_chunk + threads - 1) / threads;
            
            crack_password_shared<<<block_count, threads>>>(dev_target, dev_found, dev_result, len, offset);
            cudaDeviceSynchronize();
            
            cudaMemcpy(&host_found, dev_found, sizeof(bool), cudaMemcpyDeviceToHost);
            if (host_found) break;
            
            offset += this_chunk;
        }
        
        if (host_found) break;
    }
    
    if (host_found) {
        cudaMemcpy(host_result, dev_result, strlen(host_target), cudaMemcpyDeviceToHost);
        host_result[strlen(host_target)] = '\0';
        printf("Shared Memory: Password %s found with length %d\n", host_result, strlen(host_result));
    } else {
        printf("Shared Memory: Password not found in 2-9 lowercase chars.\n");
    }
    
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    
    printf("Shared Memory GPU time to crack: %.4f seconds\n", milliseconds / 1000.0f);
    
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    
    cudaFree(dev_target);
    cudaFree(dev_found);
    cudaFree(dev_result);
}

// Function to run register tiling approach (your existing code)
void run_register_tiling_approach(const char* host_target) {
    char* dev_target = 0;
    bool* dev_found = 0;
    char* dev_result = 0;
    bool host_found = false;
    char host_result[MAX_LEN + 1] = "";
    
    cudaMalloc(&dev_target, strlen(host_target) + 1);
    cudaMalloc(&dev_found, sizeof(bool));
    cudaMalloc(&dev_result, MAX_LEN + 1);
    
    cudaMemcpy(dev_target, host_target, strlen(host_target) + 1, cudaMemcpyHostToDevice);
    cudaMemcpy(dev_found, &host_found, sizeof(bool), cudaMemcpyHostToDevice);
    
    int threads = 256;
    
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    cudaEventRecord(start);
    
    for (int len = MIN_LEN; len <= MAX_LEN; len++) {
        unsigned long long max_combos = 1;
        for (int i = 0; i < len; i++) max_combos *= CHARSET_LEN;
        
        unsigned long long total = max_combos;
        unsigned long long offset = 0;
        
        // Adjust chunk size to account for passwords per thread
        unsigned long long chunk = 10000000 * PASSWORDS_PER_THREAD;
        while (offset < total && !host_found) {
            unsigned long long remaining = total - offset;
            unsigned long long this_chunk = (remaining > chunk) ? chunk : remaining;
            unsigned long long effective_chunk = this_chunk / PASSWORDS_PER_THREAD;
            unsigned long long block_count = (effective_chunk + threads - 1) / threads;
            
            crack_password_register<<<block_count, threads>>>(dev_target, dev_found, dev_result, len, offset);
            cudaDeviceSynchronize();
            
            cudaMemcpy(&host_found, dev_found, sizeof(bool), cudaMemcpyDeviceToHost);
            if (host_found) break;
            
            offset += this_chunk;
        }
        
        if (host_found) break;
    }
    
    if (host_found) {
        cudaMemcpy(host_result, dev_result, strlen(host_target), cudaMemcpyDeviceToHost);
        host_result[strlen(host_target)] = '\0';
        printf("Register Tiling: Password %s found with length %d\n", host_result, strlen(host_result));
    } else {
        printf("Register Tiling: Password not found in 2-9 lowercase chars.\n");
    }
    
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    
    printf("Register Tiling GPU time to crack: %.4f seconds\n", milliseconds / 1000.0f);
    
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    
    cudaFree(dev_target);
    cudaFree(dev_found);
    cudaFree(dev_result);
}

int main() {
    const char* host_target = "darrendfd";
    
    printf("Testing password cracking approaches for target: %s with length %d\n\n", host_target, strlen(host_target));
    
    printf("Running naive approach...\n");
    run_naive_approach(host_target);
    
    printf("\nRunning shared memory approach...\n");
    run_shared_memory_approach(host_target);
    
    printf("\nRunning register tiling approach...\n");
    run_register_tiling_approach(host_target);
    
    return 0;
}