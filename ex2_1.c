#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <omp.h> // Απαραίτητο για OpenMP

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