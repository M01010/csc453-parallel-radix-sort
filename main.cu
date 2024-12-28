#include <stdio.h>
#include <stdlib.h>
#define N 3
#define MAX_DIGITS 3


__device__ int digit_of(int number, int digit) {
    return number / (int)pow(10, digit - 1) % 10;
}


__global__ void radix_sort(int *idata, int *odata, int size, int d) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    int index_number = digit_of(idata[index], d);

    int before = 0;
    int after = 0;

    int cur_idx = 0;

    while (cur_idx < index) {
        int cur_number = digit_of(idata[cur_idx], d);
        if (cur_number <= index_number)
            before++;
        cur_idx++;
    }
    while (cur_idx < size) {
        int cur_number = digit_of(idata[cur_idx], d);
        if (cur_number < index_number)
            after++;
        cur_idx++;
    }

    odata[before+after] = idata[index];
}


void random_ints(int *data, int size) {
    for (int i = 0; i < size; i++) {
        double random = (double) rand() / RAND_MAX; // Random value in [0, 1]
        int num = (int) (random * pow(10, MAX_DIGITS));
        data[i] = num;
    }
}


void show_array(int *data, int size) {
    printf(" {");
    for (int i = 0; i < size - 1; i++) {
        printf("%d, ", data[i]);
    }
    printf("%d}\n", data[size - 1]);
}


int main() {
    srand(time(NULL));

    int *h_idata, *h_odata;
    int *d_idata, *d_odata;
    size_t size = N * N * sizeof(int);

    h_idata = (int *)malloc(size);
    h_odata = (int *)malloc(size);

    random_ints(h_idata, N * N);

    cudaMalloc((void **)&d_idata, size);
    cudaMalloc((void **)&d_odata, size);

    cudaMemcpy(d_idata, h_idata, size, cudaMemcpyHostToDevice);

    printf("Initial Array:");
    show_array(h_idata, N*N);

    for (int d = 1; d <= MAX_DIGITS; d++) {
        radix_sort<<<N, N>>>(d_idata, d_odata, N*N, d);
        cudaMemcpy(h_odata, d_odata, size, cudaMemcpyDeviceToHost);

        printf("Step %d:", d);
        show_array(h_odata, N*N);

        int *temp = d_idata;
        d_idata = d_odata;
        d_odata = temp;
        cudaDeviceSynchronize();
    }

    if (MAX_DIGITS % 2 == 0)
        cudaMemcpy(h_odata, d_odata, size, cudaMemcpyDeviceToHost);
    else
        cudaMemcpy(h_odata, d_idata, size, cudaMemcpyDeviceToHost);

    printf("Sorted Array:");
    show_array(h_odata, N*N);

    cudaFree(d_idata);
    cudaFree(d_odata);
    free(h_idata);
    free(h_odata);
    return 0;
}