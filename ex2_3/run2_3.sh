#!/bin/bash

# 1. Compile (για να είμαστε σίγουροι)
echo "Compiling..."
make clean
make

# Έλεγχος αν δημιουργήθηκε το εκτελέσιμο
if [ ! -f ./ex2_3 ]; then
    echo "Error: Compilation failed! (Did you create ex2_3.c?)"
    exit 1
fi

echo "=========================================================="
echo "      BENCHMARK: Merge Sort (Serial vs OpenMP Tasks)"
echo "=========================================================="

# Μεγέθη Πινάκων που ζητάει η άσκηση
# 10^7 = 10.000.000 (10M)
# 10^8 = 100.000.000 (100M) - Προσοχή: Θέλει ~800MB RAM
SIZES=(10000000 100000000)

# Αριθμός Νημάτων για κλιμάκωση
THREADS=(1 2 4 8 16)

for n in "${SIZES[@]}"; do
    echo "----------------------------------------------------------"
    echo "Testing Array Size: N = $n"
    echo "----------------------------------------------------------"

    # --- Βήμα 1: Σειριακή Εκτέλεση (Baseline) ---
    echo "-> Running Serial Merge Sort..."
    # Mode 0 = Serial, Threads = 1 (τυπικά)
    ./ex2_3 $n 0 1
    echo ""

    # --- Βήμα 2: Παράλληλη Εκτέλεση ---
    for t in "${THREADS[@]}"; do
        echo "-> Running Parallel Merge Sort with $t Threads..."
        # Mode 1 = Parallel
        ./ex2_3 $n 1 $t
    done
    echo ""
done

echo "=========================================================="
echo "   All experiments completed."
echo "=========================================================="