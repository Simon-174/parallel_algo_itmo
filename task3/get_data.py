import argparse
import itertools
import timeit

from numba import cuda, float32
import numpy as np
import pandas as pd
import tqdm


def similarity_sequential(objs, sim_matrix):
    """
    Arguments
    ---------
        objs        (np.ndarray) : np.ndarray of shape (num_objects, dimensions)
        sim_matrix  (np.ndarray) : empty (or zero) array of shape (num_objects, num_objects)

    Returns
    -------
        sim_matrix  (np.ndarray) : full array of shape (num_objects, num_objects)

    Calculates l2 distance between objects in sequential manner.
    """
    num_objects = objs.shape[0]
    for row in range(num_objects):
        for col in range(num_objects):
            sim_matrix[row, col] = np.sum((objs[row] - objs[col]) ** 2)
    
    return sim_matrix


@cuda.jit()
def kernel_similarity_global(objs, sim_matrix):
    """
    Calculates l2 distance between row and column in numba global.
    """
    row, col = cuda.grid(2)
    if row < sim_matrix.shape[0] and col < sim_matrix.shape[1]:
        current_sum = 0
        for j in range(objs.shape[1]):
            current_sum += (objs[row, j] - objs[col, j]) ** 2
        sim_matrix[row, col] = current_sum
    

def similarity_global(objs, sim_matrix):
    """
    Calculates l2 distance between objects in numba using global memory.
    """
    num_objects = objs.shape[0]
    objs_cuda_global = cuda.to_device(objs)
    sim_matrix_global = cuda.to_device(sim_matrix)

    kernel_similarity_global[(BLOCKS_PER_GRID, BLOCKS_PER_GRID), (NTHREADS, NTHREADS)](objs_cuda_global, sim_matrix_global)
    sim_matrix_global.copy_to_host(sim_matrix)
    
    return sim_matrix


@cuda.jit
def fast_matmul(A, B, C):
    """
    Perform matrix multiplication of C = A * B
    Each thread computes one element of the result matrix C
    """

    # Define an array in the shared memory
    # The size and type of the arrays must be known at compile time
    sA = cuda.shared.array(shape=(TPB, TPB), dtype=float32)
    sB = cuda.shared.array(shape=(TPB, TPB), dtype=float32)

    x, y = cuda.grid(2)
    
    tx = cuda.threadIdx.x
    ty = cuda.threadIdx.y
    
    if x >= C.shape[0] and y >= C.shape[1]:
        # Quit if (x, y) is outside of valid C boundary
        return

    # Each thread computes one element in the result matrix.
    # The dot product is chunked into dot products of TPB-long vectors.
    tmp = 0.
    for i in range(int(A.shape[1] / TPB)):
        # Preload data into shared memory
        sA[tx, ty] = A[x, ty + i * TPB]
        sB[tx, ty] = B[tx + i * TPB, y]

        # Wait until all threads finish preloading
        cuda.syncthreads()

        # Computes partial product on the shared memory
        for j in range(TPB):
            tmp += sA[tx, j] * sB[j, ty]

        # Wait until all threads finish computing
        cuda.syncthreads()

    C[x, y] = tmp


@cuda.jit()
def kernel_similarity_shared(objs, sim_matrix):
    """
    Calculates l2 distance between row and column in numba with shared memory.
    """
    shared_objs_row = cuda.shared.array((NTHREADS, NTHREADS), dtype=float32)
    shared_objs_col = cuda.shared.array((NTHREADS, NTHREADS), dtype=float32)
    row, col = cuda.grid(2) # from 0 to num_objects - 1
    thread_row = cuda.threadIdx.x # from 0 to NTHREADS - 1
    thread_col = cuda.threadIdx.y # from 0 to NTHREADS - 1

    if row < sim_matrix.shape[0] and col < sim_matrix.shape[1]:
        current_sum = 0.0
        for i in range(BLOCKS_PER_GRID):
            shared_objs_row[thread_row, thread_col] = objs[row, thread_col + i * NTHREADS]
            shared_objs_col[thread_row, thread_col] = objs[col, thread_row + i * NTHREADS]
            cuda.syncthreads()
            for j in range(NTHREADS):
                current_sum += (shared_objs_row[thread_row, j] - shared_objs_col[thread_col, j]) ** 2
            cuda.syncthreads()
        sim_matrix[row, col] = current_sum


def similarity_shared(objs, sim_matrix):
    """
    Calculates l2 distance between objects in numba using shared memory.
    """
    num_objects = objs.shape[0]
    objs_cuda_global = cuda.to_device(objs)
    sim_matrix_global = cuda.to_device(sim_matrix)

    kernel_similarity_shared[(BLOCKS_PER_GRID, BLOCKS_PER_GRID), (NTHREADS, NTHREADS)](objs_cuda_global, sim_matrix_global)
    sim_matrix = sim_matrix_global.copy_to_host()

    return sim_matrix


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Gather data for object similarity")
    parser.add_argument("--dims", type=int, nargs="+", default=[2 ** 8, 2 ** 9, 2 ** 10, 2 ** 11], help="Dimensionality to test")
    parser.add_argument("--nthreads", type=int, default=8, help="Number of threads to test")
    args = parser.parse_args()

    print("Check CUDA availability: ", cuda.is_available())

    np.random.seed(42)

    sim_methods = ["similarity_shared", "similarity_global", "similarity_sequential"]
    data = []

    for sim_method in sim_methods:
        for dims in tqdm.tqdm(args.dims, desc=sim_method):
            objs = np.random.random_sample((dims, dims)).astype(np.float32)
            sim_matrix = np.zeros((dims, dims), dtype=np.float32)

            NTHREADS = args.nthreads
            BLOCKS_PER_GRID = int(np.ceil(dims / NTHREADS))

            time = timeit.timeit(stmt=f"{sim_method}(objs, sim_matrix)", globals=globals(), number=1)

            data.append({
                "sim_method": sim_method,
                "dims": dims,
                "nthreads": NTHREADS,
                "time": time,
            })

    data = pd.DataFrame(data)

    data.to_csv(f"task3_{args.nthreads}.csv", index=False)
