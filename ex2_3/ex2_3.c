#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <time.h>

// Όριο (Granularity) για μετάβαση σε σειριακή εκτέλεση.
// Αποφεύγουμε τη δημιουργία tasks για πολύ μικρά υπο-προβλήματα (overhead reduction).
#define TASK_THRESHOLD 1000 

// Βοηθητική συνάρτηση: Συγχώνευση δύο ταξινομημένων τμημάτων
void merge(int *a, int n, int mid) {
    int *temp = (int *)malloc(n * sizeof(int));
    int i = 0;            
    int j = mid;          
    int k = 0;            

    while (i < mid && j < n) {
        if (a[i] <= a[j]) {
            temp[k++] = a[i++];
        } else {
            temp[k++] = a[j++];
        }
    }
    while (i < mid) temp[k++] = a[i++];
    while (j < n)   temp[k++] = a[j++];

    for (i = 0; i < n; i++) {
        a[i] = temp[i];
    }
    free(temp);
}

// Σειριακή Merge Sort (Κλασική αναδρομική υλοποίηση)
void mergesort_serial(int *a, int n) {
    if (n < 2) return;

    int mid = n / 2;
    
    mergesort_serial(a, mid);
    mergesort_serial(a + mid, n - mid);
    
    merge(a, n, mid);
}

// Παράλληλη Merge Sort με OpenMP Tasks
void mergesort_parallel(int *a, int n) {
    if (n < 2) return;

    // Αν το μέγεθος είναι μικρό, τρέχουμε σειριακά για ταχύτητα
    if (n < TASK_THRESHOLD) {
        mergesort_serial(a, n);
        return;
    }

    int mid = n / 2;

    // Δημιουργία Tasks: Κάθε αναδρομική κλήση γίνεται ανεξάρτητη εργασία.
    // Το 'if' clause αποτρέπει τη δημιουργία task αν το n πέσει κάτω από το όριο.
    #pragma omp task shared(a) if(n > TASK_THRESHOLD)
    mergesort_parallel(a, mid);

    #pragma omp task shared(a) if(n > TASK_THRESHOLD)
    mergesort_parallel(a + mid, n - mid);

    // Συγχρονισμός: Πρέπει να ολοκληρωθούν τα παιδιά πριν γίνει το merge
    #pragma omp taskwait

    merge(a, n, mid);
}

// Έλεγχος ορθότητας ταξινόμησης
void check_sorted(int *a, int n) {
    for (int i = 0; i < n - 1; i++) {
        if (a[i] > a[i+1]) {
            printf("Error: Array not sorted at index %d (%d > %d)\n", i, a[i], a[i+1]);
            return;
        }
    }
    printf("Validation: SUCCESS (Array is sorted)\n");
}

int main(int argc, char *argv[]) {
    if (argc != 4) {
        printf("Usage: %s <N> <Mode 0=Serial/1=Parallel> <Threads>\n", argv[0]);
        return 1;
    }

    int n = atoi(argv[1]);
    int mode = atoi(argv[2]); 
    int threads = atoi(argv[3]);

    int *a = (int *)malloc(n * sizeof(int));
    if (!a) return 1;

    // Σταθερό seed για επαναληψιμότητα των μετρήσεων
    srand(42); 
    for (int i = 0; i < n; i++) {
        a[i] = rand(); 
    }

    printf("--- Sorting N=%d elements (Mode=%s, Threads=%d) ---\n", 
           n, (mode == 0 ? "Serial" : "Parallel"), threads);

    double start_time = omp_get_wtime();

    if (mode == 0) {
        mergesort_serial(a, n);
    } else {
        omp_set_num_threads(threads);
        
        // Ξεκινάμε την παράλληλη περιοχή
        #pragma omp parallel
        {
            // Ένα μόνο νήμα (master) ξεκινάει την αναδρομή.
            // Τα υπόλοιπα νήματα περιμένουν στη "δεξαμενή" (pool) και αναλαμβάνουν tasks.
            #pragma omp single
            {
                mergesort_parallel(a, n);
            }
        }
    }

    double end_time = omp_get_wtime();
    printf("Time: %.6f sec\n", end_time - start_time);
    
    check_sorted(a, n);
    free(a);

    return 0;
}