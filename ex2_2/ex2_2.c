#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <time.h>

// Δομή για τον CSR πίνακα
typedef struct {
    int *values;   // Οι μη-μηδενικές τιμές
    int *col_ind;  // Οι δείκτες στηλών
    int *row_ptr;  // Οι δείκτες αρχής γραμμών
    int nnz;       // Πλήθος μη-μηδενικών στοιχείων
    int n;         // Διάσταση πίνακα (N x N)
} CSRMatrix;

// Βοηθητική: Απελευθέρωση μνήμης CSR
void free_csr(CSRMatrix *mat) {
    if (mat->values) free(mat->values);
    if (mat->col_ind) free(mat->col_ind);
    if (mat->row_ptr) free(mat->row_ptr);
}

// Αρχικοποίηση Πυκνού Πίνακα & Διανύσματος
// sparsity: ποσοστό από 0.0 έως 1.0 (π.χ. 0.99 σημαίνει 99% μηδενικά)
void init_dense_and_vector(int *A, int *x, int n, double sparsity) {
    // Χρησιμοποιούμε long long για να αποφύγουμε overflow στον υπολογισμό n*n
    long long total_elements = (long long)n * n;
    
    // ΣΗΜΕΙΩΣΗ: Η rand() δεν είναι thread-safe και είναι αργή.
    // Εδώ την καλούμε σειριακά ή θα μπορούσαμε να χρησιμοποιήσουμε rand_r παράλληλα.
    // Η εκφώνηση λέει "δεν λαμβάνετε υπόψη τον χρόνο αρχικοποίησης", άρα το κάνουμε απλά.

    srand(time(NULL));

    // Γεμίζουμε τον πίνακα A
    for (long long i = 0; i < total_elements; i++) {
        // Παράγουμε έναν τυχαίο float μεταξύ 0 και 1
        double r = (double)rand() / RAND_MAX;
        
        if (r < sparsity) {
            A[i] = 0; // Μηδενικό στοιχείο
        } else {
            // Μη-μηδενικό: τυχαίος ακέραιος 1-10
            A[i] = (rand() % 10) + 1;
        }
    }

    // Γεμίζουμε το διάνυσμα x (πάντα γεμάτο)
    for (int i = 0; i < n; i++) {
        x[i] = (rand() % 10) + 1;
    }
}

// Συνάρτηση Κατασκευής CSR από Dense (Παράλληλη)
// Επιστρέφει τον χρόνο που πήρε (σε δευτερόλεπτα)
double construct_csr_parallel(int *A, int n, CSRMatrix *csr) {
    double start_time = omp_get_wtime();

    csr->n = n;
    // Δέσμευση του row_ptr (N + 1 στοιχεία)
    csr->row_ptr = (int*)malloc((n + 1) * sizeof(int));
    if (!csr->row_ptr) return -1.0;

    // --- ΦΑΣΗ 1: Μέτρημα Μη-Μηδενικών ανά Γραμμή (Parallel) ---
    // Κάθε νήμα αναλαμβάνει κάποιες γραμμές και μετράει τα non-zeros τους.
    #pragma omp parallel for schedule(dynamic)
    for (int i = 0; i < n; i++) {
        int count = 0;
        for (int j = 0; j < n; j++) {
            // Προσοχή στο Indexing: i * n + j μπορεί να υπερβεί το int range
            long long index = (long long)i * n + j;
            if (A[index] != 0) {
                count++;
            }
        }
        csr->row_ptr[i] = count; // Προσωρινή αποθήκευση πλήθους
    }

    // --- ΦΑΣΗ 2: Prefix Sum (Cumulative Sum) (Serial) ---
    // Μετατροπή των Counts σε Start Indices (Row Pointers)
    // Αυτό το κομμάτι είναι σειριακό, αλλά πολύ γρήγορο (O(N)).
    int total_nnz = 0;
    for (int i = 0; i < n; i++) {
        int count = csr->row_ptr[i];
        csr->row_ptr[i] = total_nnz; // Η γραμμή i ξεκινάει εδώ
        total_nnz += count;          // Αυξάνουμε τον μετρητή
    }
    csr->row_ptr[n] = total_nnz; // Το τελευταίο στοιχείο δείχνει το συνολικό μέγεθος
    csr->nnz = total_nnz;

    // --- ΕΝΔΙΑΜΕΣΟ ΒΗΜΑ: Δέσμευση Μνήμης για values & col_ind ---
    csr->values = (int*)malloc(total_nnz * sizeof(int));
    csr->col_ind = (int*)malloc(total_nnz * sizeof(int));

    if (!csr->values || !csr->col_ind) {
        return -1.0; // Memory Error
    }

    // --- ΦΑΣΗ 3: Γέμισμα των values και col_ind (Parallel) ---
    // Τώρα ξέρουμε πού ξεκινάει η κάθε γραμμή (row_ptr[i]).
    // Μπορούμε να γράψουμε παράλληλα χωρίς Race Condition!
    
    #pragma omp parallel for schedule(dynamic)
    for (int i = 0; i < n; i++) {
        int dest_index = csr->row_ptr[i]; // Εδώ ξεκινάμε να γράφουμε για τη γραμμή i
        
        for (int j = 0; j < n; j++) {
            long long index = (long long)i * n + j;
            if (A[index] != 0) {
                csr->values[dest_index] = A[index];
                csr->col_ind[dest_index] = j;
                dest_index++; // Τοπική αύξηση (δεν επηρεάζει άλλες γραμμές)
            }
        }
        // Σημείωση: Στο τέλος του loop, το dest_index θα είναι ίσο με row_ptr[i+1].
        // Αυτό επιβεβαιώνει ότι δεν βγήκαμε εκτός ορίων της γραμμής μας.
    }

    double end_time = omp_get_wtime();
    return end_time - start_time;
}

// Παράλληλος Πολλαπλασιασμός CSR (SpMV)
// Υπολογίζει y = A * x για `iter` επαναλήψεις
double benchmark_spmv(CSRMatrix *csr, int *x, int *y, int iter) {
    // Αρχικοποίηση αποτελέσματος
    // Χρειαζόμαστε έναν προσωρινό buffer αν θέλουμε να κάνουμε πολλές επαναλήψεις
    // όπου το output της μίας είναι input της επόμενης (ping-pong).
    // Για απλότητα εδώ, και επειδή η άσκηση λέει "το διάνυσμα αποτελέσματος... είναι εισόδου της επόμενης",
    // θα κάνουμε swap pointers ή memcpy. Εδώ θα κάνουμε το απλό: x = y στο τέλος.

    // Θέλουμε ένα αντίγραφο του x για να μην χαλάσουμε το αρχικό δεδομένο στα πειράματα
    int *x_curr = (int*)malloc(csr->n * sizeof(int));
    for(int i=0; i<csr->n; i++) x_curr[i] = x[i];

    double start = omp_get_wtime();

    for (int it = 0; it < iter; it++) {
        
        #pragma omp parallel for schedule(dynamic)
        for (int i = 0; i < csr->n; i++) {
            int sum = 0;
            int row_start = csr->row_ptr[i];
            int row_end   = csr->row_ptr[i+1];

            // ΕΔΩ ΕΙΝΑΙ Η ΜΑΓΕΙΑ ΤΟΥ CSR
            for (int k = row_start; k < row_end; k++) {
                // values[k]: η τιμή του στοιχείου
                // col_ind[k]: η στήλη που βρισκόταν (άρα δείκτης στο x)
                sum += csr->values[k] * x_curr[csr->col_ind[k]];
            }
            y[i] = sum;
        }

        // Προετοιμασία για την επόμενη επανάληψη (y γίνεται x)
        // Αντί για memcpy, μπορούμε να κάνουμε pointer swap αν είμαστε προσεκτικοί,
        // αλλά το memcpy είναι πιο ασφαλές για την κατανόηση.
        // Όμως η αντιγραφή είναι σειριακό κόστος. Ας το κάνουμε παράλληλα.
        if (iter > 1 && it < iter - 1) {
            #pragma omp parallel for
            for(int i=0; i<csr->n; i++) x_curr[i] = y[i];
        }
    }

    double end = omp_get_wtime();
    free(x_curr);
    return end - start;
}

// Παράλληλος Πολλαπλασιασμός Dense (Κλασικός)
double benchmark_dense(int *A, int *x, int *y, int n, int iter) {
    int *x_curr = (int*)malloc(n * sizeof(int));
    for(int i=0; i<n; i++) x_curr[i] = x[i];

    double start = omp_get_wtime();

    for (int it = 0; it < iter; it++) {
        
        #pragma omp parallel for schedule(static) // Static εδώ ίσως είναι ΟΚ αν δεν έχουμε sparsity, αλλά dynamic είναι safe
        for (int i = 0; i < n; i++) {
            int sum = 0;
            for (int j = 0; j < n; j++) {
                long long index = (long long)i * n + j;
                // Εδώ κάνουμε πράξεις ακόμα και με τα μηδενικά!
                sum += A[index] * x_curr[j]; 
            }
            y[i] = sum;
        }

        // Swap results
        if (iter > 1 && it < iter - 1) {
            #pragma omp parallel for
            for(int i=0; i<n; i++) x_curr[i] = y[i];
        }
    }

    double end = omp_get_wtime();
    free(x_curr);
    return end - start;
}

// Σειριακή Κατασκευή CSR
double construct_csr_serial(int *A, int n, CSRMatrix *csr) {
    double start = omp_get_wtime();

    csr->n = n;
    csr->row_ptr = (int*)malloc((n + 1) * sizeof(int));
    if (!csr->row_ptr) return -1.0;

    // Βήμα 1: Μέτρημα (Σειριακά)
    for (int i = 0; i < n; i++) {
        int count = 0;
        for (int j = 0; j < n; j++) {
            long long index = (long long)i * n + j;
            if (A[index] != 0) count++;
        }
        csr->row_ptr[i] = count;
    }

    // Βήμα 2: Prefix Sum (Σειριακά - ίδιο με πριν)
    int total_nnz = 0;
    for (int i = 0; i < n; i++) {
        int count = csr->row_ptr[i];
        csr->row_ptr[i] = total_nnz;
        total_nnz += count;
    }
    csr->row_ptr[n] = total_nnz;
    csr->nnz = total_nnz;

    // Δέσμευση
    csr->values = (int*)malloc(total_nnz * sizeof(int));
    csr->col_ind = (int*)malloc(total_nnz * sizeof(int));
    if (!csr->values || !csr->col_ind) return -1.0;

    // Βήμα 3: Γέμισμα (Σειριακά)
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

    double end = omp_get_wtime();
    return end - start;
}

// Σειριακός SpMV
double spmv_serial(CSRMatrix *csr, int *x, int *y, int iter) {
    int *x_curr = (int*)malloc(csr->n * sizeof(int));
    for(int i=0; i<csr->n; i++) x_curr[i] = x[i];

    double start = omp_get_wtime();

    for (int it = 0; it < iter; it++) {
        // Καθαρός σειριακός βρόχος
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
    // 1. Έλεγχος Ορισμάτων
    if (argc != 5) {
        printf("Usage: %s <N> <Sparsity 0.0-1.0> <Iterations> <Threads>\n", argv[0]);
        return 1;
    }

    int n = atoi(argv[1]);
    double sparsity = atof(argv[2]);
    int iter = atoi(argv[3]);
    int threads = atoi(argv[4]);

    if (n <= 0 || iter <= 0 || threads <= 0 || sparsity < 0.0 || sparsity > 1.0) {
        printf("Invalid arguments.\n");
        return 1;
    }

    omp_set_num_threads(threads);
    printf("--- Sparse Matrix (N=%d, Sparsity=%.2f, Threads=%d) ---\n", n, sparsity, threads);

    // 2. Δέσμευση Μνήμης για Πυκνό Πίνακα (Dense)
    // Προσοχή: Για N=10000, αυτό είναι 400MB. Για N=20000 είναι 1.6GB.
    // Ελέγχουμε αν η malloc πέτυχε.
    long long n_sq = (long long)n * n;
    int *A_dense = (int*)malloc(n_sq * sizeof(int));
    int *x = (int*)malloc(n * sizeof(int));
    int *y_dense = (int*)malloc(n * sizeof(int));
    int *y_csr = (int*)malloc(n * sizeof(int));

    if (!A_dense || !x || !y_dense || !y_csr) {
        fprintf(stderr, "Error: Memory allocation failed (Dense).\n");
        return 1;
    }

    // 3. Αρχικοποίηση (εκτός χρονομέτρησης)
    printf("Initializing Dense Data... ");
    init_dense_and_vector(A_dense, x, n, sparsity);
    printf("Done.\n");

    // --- 4. ΚΑΤΑΣΚΕΥΗ CSR ---
    
    // Α. Σειριακή Κατασκευή
    CSRMatrix csr_serial;
    printf("Constructing CSR Serial... ");
    double t_constr_serial = construct_csr_serial(A_dense, n, &csr_serial);
    printf("Done in %.6f sec.\n", t_constr_serial);

    // Β. Παράλληλη Κατασκευή
    CSRMatrix csr_parallel;
    printf("Constructing CSR Parallel... ");
    double t_constr_parallel = construct_csr_parallel(A_dense, n, &csr_parallel);
    printf("Done in %.6f sec.\n", t_constr_parallel);

    printf(">> Construction Speedup: %.2fx\n", t_constr_serial / t_constr_parallel);

    // --- 5. ΠΟΛΛΑΠΛΑΣΙΑΣΜΟΣ SpMV ---

    // Α. Σειριακός SpMV
    printf("Running SpMV Serial (%d iter)... ", iter);
    // Χρησιμοποιούμε τον csr_parallel πίνακα, είναι ίδιος στη δομή
    double t_spmv_serial = spmv_serial(&csr_parallel, x, y_csr, iter);
    printf("Done in %.6f sec.\n", t_spmv_serial);

    // Β. Παράλληλος SpMV
    printf("Running SpMV Parallel (%d iter)... ", iter);
    double t_spmv_parallel = benchmark_spmv(&csr_parallel, x, y_csr, iter);
    printf("Done in %.6f sec.\n", t_spmv_parallel);

    printf(">> SpMV Speedup: %.2fx\n", t_spmv_serial / t_spmv_parallel);

    // --- 6. DENSE MULTIPLICATION (Για αναφορά) ---
    // Μόνο αν το N δεν είναι τεράστιο, αλλιώς θα περιμένουμε αιώνες
    if (n <= 10000) {
        printf("Running Dense Parallel (%d iter)... ", iter);
        double t_dense = benchmark_dense(A_dense, x, y_dense, n, iter);
        printf("Done in %.6f sec.\n", t_dense);
    } else {
        printf("Skipping Dense Mult due to large N.\n");
    }

    // --- 7. Verification (Επαλήθευση) ---
    printf("Verifying results (Dense vs CSR Parallel)... ");
    int errors = 0;
    for (int i = 0; i < n; i++) {
        // Συγκρίνουμε το αποτέλεσμα του Dense (y_dense) με του Parallel CSR (y_csr)
        if (y_dense[i] != y_csr[i]) {
            errors++;
            if (errors < 5) { // Τυπώνουμε μόνο τα 5 πρώτα λάθη
                printf("\nMismatch at index %d: Dense=%d, CSR=%d", i, y_dense[i], y_csr[i]);
            }
        }
    }
    
    if (errors == 0) {
        printf("SUCCESS! Results match.\n");
    } else {
        printf("\nFAILURE! Found %d errors.\n", errors);
    }

    // --- 8. Cleanup (Αποδέσμευση Μνήμης) ---
    
    // 1. Αποδέσμευση των δομών CSR (Serial & Parallel)
    // Χρησιμοποιούμε τη συνάρτηση free_csr που φτιάξαμε
    free_csr(&csr_serial);
    free_csr(&csr_parallel);

    // 2. Αποδέσμευση των απλών πινάκων (Dense)
    // ΠΡΟΣΟΧΗ: Εδώ ήταν το λάθος. Τα ονόματα πρέπει να είναι ίδια με τη δήλωση (malloc).
    free(A_dense); 
    free(x);
    free(y_dense);
    free(y_csr);

    return 0;
}