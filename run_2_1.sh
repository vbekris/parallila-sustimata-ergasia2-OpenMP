#!/bin/bash

# Compile first
make clean
make

echo "=========================================="
echo "Running Experiments for Polynomial Multiplication"
echo "=========================================="

# Ορίζουμε τα μεγέθη προβλήματος (Degree N)
SIZES=(10000 50000 100000) 

# Ορίζουμε τα νήματα που θα δοκιμάσουμε
THREADS=(1 2 4 8 16)

for n in "${SIZES[@]}"; do
    echo "------------------------------------------"
    echo "Testing Degree N = $n"
    echo "------------------------------------------"
    
    for t in "${THREADS[@]}"; do
        echo "-> Running with $t threads..."
        ./poly_mult $n $t
        echo ""
    done
done