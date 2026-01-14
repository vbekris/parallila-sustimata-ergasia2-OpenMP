#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <time.h>

// Όριο για να γυρίσουμε σε σειριακό (Granularity)
// Αν ο πίνακας έχει λιγότερα από 1000 στοιχεία, δεν φτιάχνουμε Task.
#define TASK_THRESHOLD 1000 

// Βοηθητική συνάρτηση: Ενώνει δύο ταξινομημένα τμήματα
void merge(int *a, int n, int mid) {
    int *temp = (int *)malloc(n * sizeof(int));
    int i = 0;            // Δείκτης για αριστερό τμήμα
    int j = mid;          // Δείκτης για δεξί τμήμα (a[mid...])
    int k = 0;            // Δείκτης για temp

    // Κλασική διαδικασία Merge
    while (i < mid && j < n) {
        if (a[i] <= a[j]) {
            temp[k++] = a[i++];
        } else {
            temp[k++] = a[j++];
        }
    }
    // Αν περίσσεψαν στοιχεία αριστερά
    while (i < mid) temp[k++] = a[i++];
    // Αν περίσσεψαν στοιχεία δεξιά
    while (j < n)   temp[k++] = a[j++];

    // Αντιγραφή πίσω στον αρχικό πίνακα
    for (i = 0; i < n; i++) {
        a[i] = temp[i];
    }
    free(temp);
}

// Σειριακή Merge Sort (Αναδρομική)
void mergesort_serial(int *a, int n) {
    if (n < 2) return; // Βάση αναδρομής

    int mid = n / 2;
    
    // Αναδρομικές κλήσεις (Σειριακά)
    mergesort_serial(a, mid);
    mergesort_serial(a + mid, n - mid); // Προσοχή στην αριθμητική δεικτών!
    
    merge(a, n, mid);
}

// Παράλληλη Merge Sort (Με Tasks)
void mergesort_parallel(int *a, int n) {
    if (n < 2) return;

    // Αν το πρόβλημα είναι πολύ μικρό, το λύνουμε σειριακά για να αποφύγουμε overhead
    // Αυτό είναι μια έξτρα βελτιστοποίηση.
    if (n < TASK_THRESHOLD) {
        mergesort_serial(a, n);
        return;
    }

    int mid = n / 2;

    // Δημιουργία Tasks για τα δύο μισά
    // Το if clause εδώ ελέγχει αν αξίζει να γίνει task
    #pragma omp task shared(a) if(n > TASK_THRESHOLD)
    mergesort_parallel(a, mid);

    #pragma omp task shared(a) if(n > TASK_THRESHOLD)
    mergesort_parallel(a + mid, n - mid);

    // ΚΡΙΣΙΜΟ: Πρέπει να περιμένουμε τα παιδιά να τελειώσουν!
    #pragma omp taskwait

    // Τώρα μπορούμε να ενώσουμε ασφαλώς
    merge(a, n, mid);
}

// Έλεγχος αν ο πίνακας ταξινομήθηκε σωστά
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
    int mode = atoi(argv[2]); // 0: Serial, 1: Parallel
    int threads = atoi(argv[3]);

    // Δέσμευση και Αρχικοποίηση
    int *a = (int *)malloc(n * sizeof(int));
    srand(42); // Σταθερός σπόρος για δίκαιη σύγκριση
    for (int i = 0; i < n; i++) {
        a[i] = rand(); // Τυχαίοι ακέραιοι
    }

    printf("--- Sorting N=%d elements (Mode=%s, Threads=%d) ---\n", 
           n, (mode == 0 ? "Serial" : "Parallel"), threads);

    double start_time = omp_get_wtime();

    if (mode == 0) {
        // --- SERIAL EXECUTION ---
        mergesort_serial(a, n);
    } else {
        // --- PARALLEL EXECUTION ---
        omp_set_num_threads(threads);
        
        // Ανοίγουμε την παράλληλη περιοχή
        #pragma omp parallel
        {
            // Μόνο ΕΝΑΣ (ο Master) ξεκινάει την αναδρομή
            #pragma omp single
            {
                mergesort_parallel(a, n);
            }
            // Τα υπόλοιπα threads περιμένουν εδώ και "κλέβουν" tasks από το pool
        }
    }

    double end_time = omp_get_wtime();

    printf("Time: %.6f sec\n", end_time - start_time);
    
    // Επαλήθευση
    check_sorted(a, n);

    free(a);
    return 0;
}