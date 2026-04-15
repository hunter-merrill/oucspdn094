/* Hash Kernel --------------------------------------
*       Generates an array of hash values from nonces.
*/

#define BLOCK_SIZE 1024

__global__
void reduction_kernel(unsigned int* hash_out, unsigned int* nonce_out, unsigned int* hash_array, unsigned int* nonce_array, unsigned int array_size) {

    // Calculate thread index
    __shared__ unsigned int in_s_hash[BLOCK_SIZE];
    __shared__ unsigned int in_s_nonce[BLOCK_SIZE];
    int idx = 2 * blockIdx.x * blockDim.x + threadIdx.x;

    unsigned int hash1 = (idx < array_size) ? hash_array[idx] : UINT_MAX;
    unsigned int hash2 = (idx + BLOCK_SIZE < array_size) ? hash_array[idx + BLOCK_SIZE] : UINT_MAX;

    if (hash1 < hash2) {
        in_s_hash[threadIdx.x] = hash1;
        in_s_nonce[threadIdx.x] = (idx < array_size) ? nonce_array[idx] : UINT_MAX;
    } else {
        in_s_hash[threadIdx.x] = hash2;
        in_s_nonce[threadIdx.x] = (idx + BLOCK_SIZE < array_size) ? nonce_array[idx + BLOCK_SIZE] : UINT_MAX;
    }

    for(int stride = BLOCK_SIZE / 2; stride >= 1; stride = stride / 2) {
        __syncthreads();
        if(threadIdx.x < stride)
            if (in_s_hash[threadIdx.x + stride] < in_s_hash[threadIdx.x]) {
                in_s_hash[threadIdx.x] = in_s_hash[threadIdx.x + stride];
                in_s_nonce[threadIdx.x] = in_s_nonce[threadIdx.x + stride];
            }
    }

    if(threadIdx.x == 0) {
	    hash_out[blockIdx.x] = in_s_hash[0];
        nonce_out[blockIdx.x] = in_s_nonce[0];
    }

} // End Reduction Kernel //
