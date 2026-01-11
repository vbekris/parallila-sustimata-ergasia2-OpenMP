#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <omp.h> 
// Macros για ταχύτητα (inline expansion)
#define MIN(a, b) (((a) < (b)) ? (a) : (b))
#define MAX(a, b) (((a) > (b)) ? (a) : (b))

// Βοηθητική για εκτύπωση (μόνο για μικρά n, για debugging)
void print_poly(int *P, int n) {
    if (n > 20) return; // Μην τυπώνεις τεράστιους πίνακες
    for (int i = 0; i <= n; i++) {
        printf("%d ", P[i]);
    }
    printf("\n");
}

// Αρχικοποίηση πολυωνύμου με τυχαίους ΜΗ-ΜΗΔΕΝΙΚΟΥΣ ακέραιους
void init_poly(int *P, int n) {
    for (int i = 0; i <= n; i++) {
        // rand() % 10 δίνει 0-9. Προσθέτουμε 1 για να έχουμε 1-10.
        // Έτσι αποφεύγουμε το 0 που απαγορεύει η εκφώνηση.
        P[i] = (rand() % 10) + 1; 
    }
}

// Έλεγχος ορθότητας
// Επιστρέφει 1 αν είναι όλα σωστά, 0 αν βρει λάθος
int check_result(int *Serial, int *Parallel, int n) {
    int errors = 0;
    for (int i = 0; i <= 2 * n; i++) {
        if (Serial[i] != Parallel[i]) {
            errors++;
            // Τυπώνουμε το πρώτο λάθος για να ξέρουμε τι φταίει
            if (errors == 1) {
                printf("Error at index %d: Serial=%d, Parallel=%d\n", 
                       i, Serial[i], Parallel[i]);
            }
        }
    }
    return (errors == 0);
}


void serial_mult(int *A, int *B, int *C, long n) {
    // Διατρέχουμε όλα τα κελιά του αποτελέσματος C
    // Το αποτέλεσμα έχει βαθμό 2*n, άρα δείκτες 0..2n
    for (long k = 0; k <= 2 * n; k++) {
        
        long start_i = MAX(0, k - n);
        long end_i   = MIN(k, n);
        
        int sum = 0;
        
        // Υπολογισμός της συνέλιξης (convolution) για το στοιχείο k
        for (long i = start_i; i <= end_i; i++) {
            sum += A[i] * B[k - i];
        }
        
        C[k] = sum;
    }
}


void parallel_mult(int *A, int *B, int *C, long n) {
    
    // schedule(dynamic): Κρίσιμο για την τριγωνική κατανομή φόρτου.
    // default(none): Μας αναγκάζει να σκεφτούμε το scoping για ΚΑΘΕ μεταβλητή.
    
    #pragma omp parallel for schedule(dynamic) \
        default(none) \
        shared(A, B, C, n) 
    for (long k = 0; k <= 2 * n; k++) {
        
        // Δηλώνοντας τις μεταβλητές ΕΔΩ, είναι αυτόματα private (stack του νήματος)
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
    // --- 1. Λήψη ορισμάτων & Έλεγχοι ---
    if (argc != 3) {
        printf("Usage: %s <degree_n> <num_threads>\n", argv[0]);
        return 1;
    }

    long n = atol(argv[1]); 
    int threads = atoi(argv[2]);

    if (n <= 0 || threads <= 0) {
        printf("Error: n and threads must be positive integers.\n");
        return 1;
    }

    omp_set_num_threads(threads);
    printf("--- Polynomial Multiplication (Degree N=%ld, Threads=%d) ---\n", n, threads);

    // --- 2. Δέσμευση Μνήμης ---
    int *A = (int*)malloc((n + 1) * sizeof(int));
    int *B = (int*)malloc((n + 1) * sizeof(int));
    int *C_serial = (int*)malloc((2 * n + 1) * sizeof(int));
    int *C_parallel = (int*)malloc((2 * n + 1) * sizeof(int));

    if (!A || !B || !C_serial || !C_parallel) {
        fprintf(stderr, "Error: Memory allocation failed!\n");
        return 1;
    }

    // --- 3. Αρχικοποίηση (Μετράμε και αυτόν τον χρόνο όπως ζητήθηκε) ---
    double start_time, end_time; // Μεταβλητές για μέτρηση χρόνου
    
    start_time = omp_get_wtime();
    
    // First Touch Initialization (Σημαντικό για NUMA architectures)
    #pragma omp parallel for schedule(static)
    for (long i = 0; i <= 2 * n; i++) {
        C_serial[i] = 0;
        C_parallel[i] = 0;
    }

    // Γεμίζουμε τα A και B
    // Προσοχή: Η rand() δεν είναι thread-safe, την καλούμε σειριακά εδώ.
    init_poly(A, n);
    init_poly(B, n);
    
    end_time = omp_get_wtime();
    printf("Initialization Time: %.6f seconds\n", end_time - start_time);


    // --- 4. Εκτέλεση & Χρονομέτρηση ΣΕΙΡΙΑΚΟΥ ---
    printf("Running Serial Algorithm...\n");
    start_time = omp_get_wtime();
    
    serial_mult(A, B, C_serial, n);
    
    end_time = omp_get_wtime();
    double serial_duration = end_time - start_time;
    printf("Serial Time:       %.6f seconds\n", serial_duration);


    // --- 5. Εκτέλεση & Χρονομέτρηση ΠΑΡΑΛΛΗΛΟΥ ---
    printf("Running Parallel Algorithm...\n");
    start_time = omp_get_wtime();
    
    parallel_mult(A, B, C_parallel, n);
    
    end_time = omp_get_wtime();
    double parallel_duration = end_time - start_time;
    printf("Parallel Time:     %.6f seconds\n", parallel_duration);


    // --- 6. Επαλήθευση & Speedup ---
    printf("Verifying results... ");
    if (check_result(C_serial, C_parallel, n)) {
        printf("SUCCESS! Results match.\n");
        
        // Υπολογισμός Speedup (S = T_serial / T_parallel)
        double speedup = serial_duration / parallel_duration;
        printf("Speedup:           %.2fx\n", speedup);
        
    } else {
        printf("FAILURE! Results do not match.\n");
    }

    // --- 7. Αποδέσμευση ---
    free(A);
    free(B);
    free(C_serial);
    free(C_parallel);

    return 0;
}