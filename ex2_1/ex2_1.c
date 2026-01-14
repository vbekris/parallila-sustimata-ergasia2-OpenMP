#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <omp.h> 

#define MIN(a, b) (((a) < (b)) ? (a) : (b))
#define MAX(a, b) (((a) > (b)) ? (a) : (b))

// Εκτύπωση πολυωνύμου (για debugging σε μικρά n)
void print_poly(int *P, int n) {
    if (n > 20) return; 
    for (int i = 0; i <= n; i++) {
        printf("%d ", P[i]);
    }
    printf("\n");
}

// Αρχικοποίηση με τυχαίους ακεραίους [1-10]
void init_poly(int *P, int n) {
    for (int i = 0; i <= n; i++) {
        P[i] = (rand() % 10) + 1; 
    }
}

// Επαλήθευση αποτελεσμάτων (Serial vs Parallel)
int check_result(int *Serial, int *Parallel, int n) {
    int errors = 0;
    for (int i = 0; i <= 2 * n; i++) {
        if (Serial[i] != Parallel[i]) {
            errors++;
            if (errors == 1) {
                printf("Mismatch at index %d: Serial=%d, Parallel=%d\n", 
                       i, Serial[i], Parallel[i]);
            }
        }
    }
    return (errors == 0);
}

// Σειριακή εκτέλεση πολλαπλασιασμού
void serial_mult(int *A, int *B, int *C, long n) {
    for (long k = 0; k <= 2 * n; k++) {
        
        long start_i = MAX(0, k - n);
        long end_i   = MIN(k, n);
        
        int sum = 0;
        // Υπολογισμός συνέλιξης για τον όρο k
        for (long i = start_i; i <= end_i; i++) {
            sum += A[i] * B[k - i];
        }
        C[k] = sum;
    }
}

// Παράλληλη εκτέλεση με OpenMP
void parallel_mult(int *A, int *B, int *C, long n) {
    
    // Επιλογή schedule(dynamic):
    // Ο εσωτερικός βρόχος έχει μεταβλητό μέγεθος (τριγωνική μορφή).
    // Το dynamic εξασφαλίζει σωστή κατανομή φόρτου (load balancing).
    #pragma omp parallel for schedule(dynamic) \
        default(none) \
        shared(A, B, C, n) 
    for (long k = 0; k <= 2 * n; k++) {
        
        // Οι μεταβλητές εντός του parallel for είναι εξ ορισμού private
        long start_i = MAX(0, k - n);
        long end_i   = MIN(k, n);
        
        int sum = 0;
        
        for (long i = start_i; i <= end_i; i++) {
            sum += A[i] * B[k - i];
        }
        C[k] = sum;
    }
}

int main(int argc, char *argv[]) {
    // 1. Έλεγχος ορισμάτων
    if (argc != 3) {
        printf("Usage: %s <degree_n> <num_threads>\n", argv[0]);
        return 1;
    }

    long n = atol(argv[1]); 
    int threads = atoi(argv[2]);

    if (n <= 0 || threads <= 0) {
        printf("Error: Invalid inputs.\n");
        return 1;
    }

    omp_set_num_threads(threads);
    printf("--- Polynomial Multiplication (N=%ld, Threads=%d) ---\n", n, threads);

    // 2. Δέσμευση μνήμης
    int *A = (int*)malloc((n + 1) * sizeof(int));
    int *B = (int*)malloc((n + 1) * sizeof(int));
    int *C_serial = (int*)malloc((2 * n + 1) * sizeof(int));
    int *C_parallel = (int*)malloc((2 * n + 1) * sizeof(int));

    if (!A || !B || !C_serial || !C_parallel) {
        fprintf(stderr, "Allocation failed.\n");
        return 1;
    }

    // 3. Αρχικοποίηση δεδομένων
    // First touch initialization για βελτιστοποίηση NUMA
    #pragma omp parallel for schedule(static)
    for (long i = 0; i <= 2 * n; i++) {
        C_serial[i] = 0;
        C_parallel[i] = 0;
    }

    // Η rand() καλείται σειριακά
    init_poly(A, n);
    init_poly(B, n);

    // 4. Σειριακή Εκτέλεση
    double start = omp_get_wtime();
    serial_mult(A, B, C_serial, n);
    double t_serial = omp_get_wtime() - start;
    printf("Serial Time:       %.6f s\n", t_serial);

    // 5. Παράλληλη Εκτέλεση
    start = omp_get_wtime();
    parallel_mult(A, B, C_parallel, n);
    double t_parallel = omp_get_wtime() - start;
    printf("Parallel Time:     %.6f s\n", t_parallel);

    // 6. Επαλήθευση & Speedup
    printf("Verifying... ");
    if (check_result(C_serial, C_parallel, n)) {
        printf("OK.\n");
        printf("Speedup:           %.2fx\n", t_serial / t_parallel);
    } else {
        printf("FAIL.\n");
    }

    free(A); free(B); free(C_serial); free(C_parallel);
    return 0;
}