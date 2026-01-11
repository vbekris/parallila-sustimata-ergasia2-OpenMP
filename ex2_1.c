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
    if (argc != 3) {
        printf("Usage: %s <degree_n> <num_threads>\n", argv[0]);
        return 1;
    }

    // 1. Λήψη ορισμάτων
    long n = atol(argv[1]); // Χρήση long για μεγάλα n
    int threads = atoi(argv[2]);

    if (n <= 0 || threads <= 0) {
        printf("Error: n and threads must be positive integers.\n");
        return 1;
    }

    // Ρύθμιση νημάτων OpenMP
    omp_set_num_threads(threads);

    printf("--- Polynomial Multiplication (Degree N=%ld, Threads=%d) ---\n", n, threads);

    // 2. Δυναμική Δέσμευση Μνήμης (Heap Allocation)
    // Προσοχή: Το μέγεθος είναι n+1 για τα A, B και 2*n+1 για τα C
    int *A = (int*)malloc((n + 1) * sizeof(int));
    int *B = (int*)malloc((n + 1) * sizeof(int));
    int *C_serial = (int*)malloc((2 * n + 1) * sizeof(int));
    int *C_parallel = (int*)malloc((2 * n + 1) * sizeof(int));

    // Έλεγχος αν απέτυχε η malloc (Κρίσιμο για μεγάλους πίνακες!)
    if (!A || !B || !C_serial || !C_parallel) {
        fprintf(stderr, "Error: Memory allocation failed!\n");
        return 1;
    }

    // 3. Αρχικοποίηση (Initialization)
    double start_time = omp_get_wtime();
    
    // Αρχικοποίηση με 0 στα αποτελέσματα (σημαντικό για το +=)
    // Χρησιμοποιούμε parallel for για γρήγορη αρχικοποίηση (First Touch Policy)
    #pragma omp parallel for
    for (long i = 0; i <= 2 * n; i++) {
        C_serial[i] = 0;
        C_parallel[i] = 0;
    }

    // Γεμίζουμε τα A και B
    init_poly(A, n);
    init_poly(B, n);
    
    double init_time = omp_get_wtime() - start_time;
    printf("Initialization Time: %f seconds\n", init_time);

    // --- ΕΔΩ ΘΑ ΜΠΟΥΝ ΟΙ ΚΛΗΣΕΙΣ ΣΤΟΥΣ ΑΛΓΟΡΙΘΜΟΥΣ (ΦΑΣΗ 4 - Μέρος 3) ---
    serial_mult(A, B, C_serial, n);
    parallel_mult(A, B, C_parallel, n);
    
    // 4. Αποδέσμευση Μνήμης (Garbage Collection δεν υπάρχει στη C!)
    free(A);
    free(B);
    free(C_serial);
    free(C_parallel);

    return 0;
}