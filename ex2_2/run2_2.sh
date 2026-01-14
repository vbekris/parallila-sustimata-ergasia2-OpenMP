#!/bin/bash

# 1. Compile (για σιγουριά)
echo "Compiling..."
make clean
make

if [ ! -f ./ex2_2 ]; then
    echo "Error: Compilation failed!"
    exit 1
fi

echo "=================================================================="
echo "   BENCHMARK: Sparse (CSR) vs Dense Matrix-Vector Multiplication"
echo "=================================================================="

# Παράμετροι Πειράματος
# ΠΡΟΣΟΧΗ: Το N=10000 στον Dense απαιτεί πολλή μνήμη και χρόνο.
# Αν το PC σου ζορίζεται, αφαίρεσε το 10000 από τη λίστα SIZES.
SIZES=(1000 5000 10000)

# Sparsity: Ποσοστό μηδενικών (0.99 σημαίνει 99% κενός πίνακας)
# Δοκιμάζουμε: Πυκνό (1% κενά), Αραιό (90% κενά), Πολύ Αραιό (99% κενά)
SPARSITIES=(0.01 0.90 0.99)

# Σταθερός αριθμός επαναλήψεων για να είναι μετρήσιμος ο χρόνος
ITERATIONS=20

# Νήματα
THREADS=(1 2 4 8)

for n in "${SIZES[@]}"; do
    for sp in "${SPARSITIES[@]}"; do
        echo "------------------------------------------------------------"
        echo "PARAMS: N=$n | Sparsity=$sp | Iter=$ITERATIONS"
        echo "------------------------------------------------------------"
        
        for t in "${THREADS[@]}"; do
            echo "-> Running with Threads: $t"
            # Κλήση: ./ex2_2 <N> <Sparsity> <Iter> <Threads>
            ./ex2_2 $n $sp $ITERATIONS $t
            echo ""
        done
    done
done

echo "=================================================================="
echo "   All experiments completed."
echo "=================================================================="