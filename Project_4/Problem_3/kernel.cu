#define BLUR_SIZE 2

__global__ 
void kernel(int* inputMatrix, int* outputMatrix, int* filterMatrix, int n_row, int n_col)
{
    // Code template from 2.2.3 slide 11

    int Col = blockIdx.x * blockDim.x + threadIdx.x;
    int Row = blockIdx.y * blockDim.y + threadIdx.y;

    // check the boundary condition of the pixel
    if (Col < n_col && Row < n_row) {

        int sum_val = 0;

        // Iterate through all the pixels in the blur window
        // BLUR_SIZE is 2 for a blur window of 5X5
        for (int blurRow = -BLUR_SIZE; blurRow < BLUR_SIZE +1; ++blurRow) {
            for(int blurCol = -BLUR_SIZE; blurCol < BLUR_SIZE+1; ++blurCol) {
                int curRow = Row + blurRow;
                int curCol = Col + blurCol;

                int i_row = blurRow + BLUR_SIZE;
                int i_col = blurCol + BLUR_SIZE;

                if( curRow > -1 && curRow < n_row && curCol > -1 && curCol < n_col)
                {
                    sum_val += inputMatrix[curRow*n_col + curCol]*filterMatrix[i_row*5 + i_col]; 
                }
            }
        }

        outputMatrix[Row*n_col+Col] = sum_val;

    }
}