#include <mpi.h>
#include <cstdio>

void construct_matrices(int *n_ptr, int *m_ptr, int *l_ptr,
                         int **a_mat_ptr, int **b_mat_ptr) 
{
    int size, rank;
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    // allocates memory and read inputs
    if (rank == 0)
        scanf("%d %d %d", n_ptr, m_ptr, l_ptr);

    MPI_Bcast(n_ptr, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(m_ptr, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(l_ptr, 1, MPI_INT, 0, MPI_COMM_WORLD);
   
    *a_mat_ptr = new int[(*n_ptr) * (*m_ptr)]();
    *b_mat_ptr = new int[(*m_ptr) * (*l_ptr)]();

    if (rank == 0) {
        for (int i = 0; i < *n_ptr; i++) {
            for (int j = 0; j < *m_ptr; j++) {
                scanf("%d", *a_mat_ptr + (i * *(m_ptr) + j));
            }
        }

        for (int i = 0; i < *m_ptr; i++) {
            for (int j = 0; j < *l_ptr; j++) {
                scanf("%d", *b_mat_ptr + (i * *(l_ptr) + j));
            }
        }
    }

    MPI_Bcast(*a_mat_ptr, (*n_ptr) * (*m_ptr), MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(*b_mat_ptr, (*m_ptr) * (*l_ptr), MPI_INT, 0, MPI_COMM_WORLD);
    return;
}

void matrix_multiply(const int n, const int m, const int l,
                     const int *a_mat, const int *b_mat)
{
    int size, rank;
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    // check broadcast success or not
    // printf("N: %d, M: %d, L:%d\n", n, m, l);
    // for (int i = 0; i < n; i++) {
    //     for (int j = 0; j < m; j++) {
    //         printf("%d ", a_mat[i * m + j]);
    //     }
    //     printf("\n");
    // }
    // for (int i = 0; i < m; i++) {
    //     for (int j = 0; j < l; j++) {
    //         printf("%d ", b_mat[i * l + j]);
    //     }
    //     printf("\n");
    // }
    int *local_c_mat = new int[n * l]();
    int *c_mat = new int[n * l]();
    int avg_row = n / size;
    int start_row = rank * avg_row;
    int end_row = (rank == size-1) ? n : start_row + avg_row;
    
    // printf("[Rank: %d] start_row: %d, end_row: %d\n", rank, start_row, end_row);
    for (int i = start_row; i < end_row; i++) {
        for (int j = 0; j < l; j++) {
            for (int k = 0; k < m; k++) {
                local_c_mat[i * l + j] += (a_mat[i * m + k] * b_mat[k * l + j]); 
            }
        }
    }
    MPI_Reduce(local_c_mat, c_mat, n * l, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
    
    if (rank == 0) {
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < l; j++) {
                printf("%d ", c_mat[i * l + j]);
            }
            printf("\n");
        }
    }
    delete[] local_c_mat;
    delete[] c_mat;
    return;
}

void destruct_matrices(int *a_mat, int *b_mat) 
{
    int size, rank;
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    delete[] a_mat;
    delete[] b_mat;
}
