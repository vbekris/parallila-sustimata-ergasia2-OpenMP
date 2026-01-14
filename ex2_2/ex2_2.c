#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <time.h>

// Δομή CSR (Compressed Sparse Row)
// Αποθηκεύει μόνο τα μη-μηδενικά στοιχεία για εξοικονόμηση μνήμης
typedef struct {
    int *values;   // Οι τιμές των μη-μηδενικών στοιχείων
    int *col_ind;  // Οι δείκτες στήλης για κάθε στοιχείο
    int *row_ptr;  // Δείκτες όπου ξεκινάει η κάθε γραμμή (μέγεθος N+1)
    int nnz;       // Συνολικό πλήθος μη-μηδενικών στοιχείων
    int n;         // Διάσταση πίνακα (NxN)
} CSRMatrix;

void free_csr(CSRMatrix *mat) {
    if (mat->values) free(mat->values);
    if (mat->col_ind) free(mat->col_ind);
    if (mat->row_ptr) free(mat->row_ptr);
}

// Αρχικοποίηση δεδομένων (Dense μορφή) για επαλήθευση
void init_dense_and_vector(int *A, int *x, int n, double sparsity) {
    long long total_elements = (long long)n * n;
    
    // Χρήση time(NULL) για seed, η rand() καλείται σειριακά εδώ
    srand(time(NULL));

    for (long long i = 0; i < total_elements; i++) {
        double r = (double)rand() / RAND_MAX;
        if (r < sparsity) {
            A[i] = 0; 
        } else {
            A[i] = (rand() % 10) + 1; // Τυχαίες τιμές 1-10
        }
    }

    for (int i = 0; i < n; i++) {
        x[i] = (rand() % 10) + 1;
    }
}

// Μετατροπή Dense -> CSR (Παράλληλη Υλοποίηση 3 Βημάτων)
double construct_csr_parallel(int *A, int n, CSRMatrix *csr) {
    double start_time = omp_get_wtime();

    csr->n = n;
    csr->row_ptr = (int*)malloc((n + 1) * sizeof(int));
    if (!csr->row_ptr) return -1.0;

    // ΒΗΜΑ 1: Υπολογισμός NNZ ανά γραμμή (Παράλληλα)
    // Κάθε νήμα μετράει τα μη-μηδενικά για τις γραμμές που αναλαμβάνει
    #pragma omp parallel for schedule(dynamic)
    for (int i = 0; i < n; i++) {
        int count = 0;
        for (int j = 0; j < n; j++) {
            long long index = (long long)i * n + j;
            if (A[index] != 0) {
                count++;
            }
        }
        csr->row_ptr[i] = count; 
    }

    // ΒΗΜΑ 2: Prefix Sum / Scan (Σειριακά)
    // Μετατροπή των μετρητών σε δείκτες θέσης. Είναι γρήγορο (O(N)) και δύσκολο να παραλληλιστεί αποδοτικά.
    int total_nnz = 0;
    for (int i = 0; i < n; i++) {
        int count = csr->row_ptr[i];
        csr->row_ptr[i] = total_nnz; 
        total_nnz += count;          
    }
    csr->row_ptr[n] = total_nnz; 
    csr->nnz = total_nnz;

    // Δέσμευση μνήμης βάσει του συνολικού NNZ που βρήκαμε
    csr->values = (int*)malloc(total_nnz * sizeof(int));
    csr->col_ind = (int*)malloc(total_nnz * sizeof(int));
    if (!csr->values || !csr->col_ind) return -1.0;

    // ΒΗΜΑ 3: Γέμισμα πινάκων values και col_ind (Παράλληλα)
    // Τώρα γνωρίζουμε την ακριβή θέση εγγραφής για κάθε γραμμή (row_ptr[i]), οπότε δεν έχουμε race condition.
    #pragma omp parallel for schedule(dynamic)
    for (int i = 0; i < n; i++) {
        int dest_index = csr->row_ptr[i]; 
        
        for (int j = 0; j < n; j++) {
            long long index = (long long)i * n + j;
            if (A[index] != 0) {
                csr->values[dest_index] = A[index];
                csr->col_ind[dest_index] = j;
                dest_index++; 
            }
        }
    }

    return omp_get_wtime() - start_time;
}

// SpMV: Sparse Matrix-Vector Multiplication (y = A * x)
double benchmark_spmv(CSRMatrix *csr, int *x, int *y, int iter) {
    // Αντίγραφο του x για τις επαναλήψεις (ping-pong buffering)
    int *x_curr = (int*)malloc(csr->n * sizeof(int));
    for(int i=0; i<csr->n; i++) x_curr[i] = x[i];

    double start = omp_get_wtime();

    for (int it = 0; it < iter; it++) {
        
        // Κύριος βρόχος SpMV
        // Χρήση dynamic schedule λόγω πιθανής ανισοκατανομής των μη-μηδενικών στοιχείων ανά γραμμή
        #pragma omp parallel for schedule(dynamic)
        for (int i = 0; i < csr->n; i++) {
            int sum = 0;
            int row_start = csr->row_ptr[i];
            int row_end   = csr->row_ptr[i+1];

            for (int k = row_start; k < row_end; k++) {
                // Indirect addressing: x_curr[col_ind[k]]
                // Προκαλεί random memory access pattern (πιθανά cache misses)
                sum += csr->values[k] * x_curr[csr->col_ind[k]];
            }
            y[i] = sum;
        }

        // Ενημέρωση του διανύσματος εισόδου για την επόμενη επανάληψη
        if (iter > 1 && it < iter - 1) {
            #pragma omp parallel for
            for(int i=0; i<csr->n; i++) x_curr[i] = y[i];
        }
    }

    double end = omp_get_wtime();
    free(x_curr);
    return end - start;
}

// Dense Multiplication (y = A * x) - Για σύγκριση απόδοσης
double benchmark_dense(int *A, int *x, int *y, int n, int iter) {
    int *x_curr = (int*)malloc(n * sizeof(int));
    for(int i=0; i<n; i++) x_curr[i] = x[i];

    double start = omp_get_wtime();

    for (int it = 0; it < iter; it++) {
        #pragma omp parallel for schedule(static)
        for (int i = 0; i < n; i++) {
            int sum = 0;
            for (int j = 0; j < n; j++) {
                // Προσπέλαση όλου του πίνακα, ακόμα και των μηδενικών
                long long index = (long long)i * n + j;
                sum += A[index] * x_curr[j]; 
            }
            y[i] = sum;
        }

        if (iter > 1 && it < iter - 1) {
            #pragma omp parallel for
            for(int i=0; i<n; i++) x_curr[i] = y[i];
        }
    }

    double end = omp_get_wtime();
    free(x_curr);
    return end - start;
}

// Σειριακή Κατασκευή (για υπολογισμό Speedup κατασκευής)
double construct_csr_serial(int *A, int n, CSRMatrix *csr) {
    double start = omp_get_wtime();

    csr->n = n;
    csr->row_ptr = (int*)malloc((n + 1) * sizeof(int));
    if (!csr->row_ptr) return -1.0;

    for (int i = 0; i < n; i++) {
        int count = 0;
        for (int j = 0; j < n; j++) {
            long long index = (long long)i * n + j;
            if (A[index] != 0) count++;
        }
        csr->row_ptr[i] = count;
    }

    int total_nnz = 0;
    for (int i = 0; i < n; i++) {
        int count = csr->row_ptr[i];
        csr->row_ptr[i] = total_nnz;
        total_nnz += count;
    }
    csr->row_ptr[n] = total_nnz;
    csr->nnz = total_nnz;

    csr->values = (int*)malloc(total_nnz * sizeof(int));
    csr->col_ind = (int*)malloc(total_nnz * sizeof(int));
    if (!csr->values || !csr->col_ind) return -1.0;

    for (int i = 0; i < n; i++) {
        int dest_index = csr->row_ptr[i];
        for (int j = 0; j < n; j++) {
            long long index = (long long)i * n + j;
            if (A[index] != 0) {
                csr->values[dest_index] = A[index];
                csr->col_ind[dest_index] = j;
                dest_index++;
            }
        }
    }

    return omp_get_wtime() - start;
}

double spmv_serial(CSRMatrix *csr, int *x, int *y, int iter) {
    int *x_curr = (int*)malloc(csr->n * sizeof(int));
    for(int i=0; i<csr->n; i++) x_curr[i] = x[i];

    double start = omp_get_wtime();

    for (int it = 0; it < iter; it++) {
        for (int i = 0; i < csr->n; i++) {
            int sum = 0;
            int row_start = csr->row_ptr[i];
            int row_end   = csr->row_ptr[i+1];

            for (int k = row_start; k < row_end; k++) {
                sum += csr->values[k] * x_curr[csr->col_ind[k]];
            }
            y[i] = sum;
        }
        if (iter > 1 && it < iter - 1) {
            for(int i=0; i<csr->n; i++) x_curr[i] = y[i];
        }
    }

    double end = omp_get_wtime();
    free(x_curr);
    return end - start;
}

int main(int argc, char *argv[]) {
    if (argc != 5) {
        printf("Usage: %s <N> <Sparsity> <Iter> <Threads>\n", argv[0]);
        return 1;
    }

    int n = atoi(argv[1]);
    double sparsity = atof(argv[2]);
    int iter = atoi(argv[3]);
    int threads = atoi(argv[4]);

    if (n <= 0 || iter <= 0 || threads <= 0) return 1;

    omp_set_num_threads(threads);
    printf("--- Sparse Matrix (N=%d, Sparsity=%.2f, Threads=%d) ---\n", n, sparsity, threads);

    long long n_sq = (long long)n * n;
    int *A_dense = (int*)malloc(n_sq * sizeof(int));
    int *x = (int*)malloc(n * sizeof(int));
    int *y_dense = (int*)malloc(n * sizeof(int));
    int *y_csr = (int*)malloc(n * sizeof(int));

    if (!A_dense || !x || !y_dense || !y_csr) {
        fprintf(stderr, "Allocation failed.\n");
        return 1;
    }

    printf("Initializing Data... ");
    init_dense_and_vector(A_dense, x, n, sparsity);
    printf("Done.\n");

    // 1. Κατασκευή CSR (Serial vs Parallel)
    CSRMatrix csr_serial, csr_parallel;
    
    double t_create_ser = construct_csr_serial(A_dense, n, &csr_serial);
    printf("Create CSR Serial: %.6f s\n", t_create_ser);
    
    double t_create_par = construct_csr_parallel(A_dense, n, &csr_parallel);
    printf("Create CSR Parallel: %.6f s (Speedup: %.2fx)\n", t_create_par, t_create_ser / t_create_par);

    // 2. Εκτέλεση SpMV (Serial vs Parallel)
    double t_spmv_ser = spmv_serial(&csr_parallel, x, y_csr, iter);
    printf("SpMV Serial:       %.6f s\n", t_spmv_ser);
    
    double t_spmv_par = benchmark_spmv(&csr_parallel, x, y_csr, iter);
    printf("SpMV Parallel:     %.6f s (Speedup: %.2fx)\n", t_spmv_par, t_spmv_ser / t_spmv_par);

    // 3. Εκτέλεση Dense (μόνο αν το N είναι λογικό)
    if (n <= 10000) {
        double t_dense = benchmark_dense(A_dense, x, y_dense, n, iter);
        printf("Dense Parallel:    %.6f s\n", t_dense);
    }

    // 4. Επαλήθευση
    int errors = 0;
    for (int i = 0; i < n; i++) {
        if (y_dense[i] != y_csr[i]) {
            errors++;
            if(errors == 1) printf("Mismatch at %d: Dense=%d, CSR=%d\n", i, y_dense[i], y_csr[i]);
        }
    }
    printf("Verification: %s\n", errors == 0 ? "SUCCESS" : "FAIL");

    free_csr(&csr_serial);
    free_csr(&csr_parallel);
    free(A_dense); free(x); free(y_dense); free(y_csr);

    return 0;
}