# Copilot Instructions for parallila-sustimata-ergasia2-OpenMP

## Project Overview
Educational C project implementing parallel polynomial multiplication using OpenMP. Compares serial vs parallel implementations for correctness validation and performance benchmarking.

## Assignment Requirements (Άσκηση 2.1 - 20%)

### Program Arguments
- `argv[1]`: Polynomial degree n
- `argv[2]`: Number of OpenMP threads

### Required Output
1. **Timing**: Initialization time, serial execution time, parallel execution time
2. **Validation**: Confirm parallel results match serial results
3. **No FFT**: Use naive O(n²) polynomial multiplication algorithm

### Benchmarking Requirements
- Test with n = 10⁵ and n = 10⁶
- Test with varying thread counts
- Compare speedup vs serial algorithm
- Compare OpenMP performance vs Pthreads (from Άσκηση 1.1)

### Platform
- Linux Ubuntu (linux01.di.uoa.gr to linux30.di.uoa.gr)
- Include Makefile with submission

## Architecture & Key Patterns

### Polynomial Representation
- Polynomials stored as integer arrays: index `i` = coefficient of $x^i$
- Degree-n polynomial uses `n+1` elements (indices 0 to n)
- Multiplication of degree-n inputs produces degree-2n output (2n+1 elements)
- **Critical**: Coefficients must be non-zero (1-10), enforced via `(rand() % 10) + 1`

### Code Structure in [ex2_1.c](../ex2_1/ex2_1.c)
- `print_poly()`: Debug helper, auto-skips arrays > 20 elements
- `init_poly()`: Initializes with random non-zero coefficients
- `check_result()`: Validates parallel vs serial results, returns 1 (pass) / 0 (fail)

## Development Workflow

### Compilation
```bash
gcc -fopenmp -o ex2_1 ex2_1.c
```

### Debugging
- VS Code debug configuration at [.vscode/launch.json](../.vscode/launch.json)
- Binary output: `build/Debug/outDebug`
- Debugger: GDB with pretty-printing enabled

### Testing Pattern
1. Initialize polynomials via `init_poly()`
2. Run serial implementation
3. Run parallel implementation  
4. Validate with `check_result()` - first mismatch printed on failure

## OpenMP Conventions
- Include `<omp.h>` header
- Use `#pragma omp parallel for` for coefficient computation loops
- Handle race conditions when accumulating to shared result arrays

## Project-Specific Patterns
- Comments in Greek (Ελληνικά) - maintain language consistency
- Loop bounds are **inclusive**: use `i <= n` for degree-n polynomials
- Return convention: 1 = success, 0 = failure (opposite of Unix exit codes)
- Safety guards: `if (n > 20) return` prevents debug output overflow
