# CVXPY Batch Solving

Batch solving allows you to solve multiple optimization problems with the same structure but different parameter values efficiently. This is useful for applications like model predictive control (MPC), parameter sweeps, and Monte Carlo simulations.

## Supported Solvers

| Solver       | Batch Support    | Implementation                          | Notes                                   |
|--------------|------------------|-----------------------------------------|-----------------------------------------|
| **Clarabel** | Yes              | `BatchSolver.solve_batch_parallel()`    | ~5.8x speedup, best for conic programs  |
| **MOSEK**    | Yes              | `env.optimizebatch()`                   | ~2.3x speedup, supports LP/QP/SOCP/SDP  |
| **Gurobi**   | Possible         | `optimizebatch()` via Cluster Manager   | Requires infrastructure setup           |
| **CPLEX**    | No               | Single-problem parallelism only         |                                         |
| **SCS**      | No               | Warm-start only                         |                                         |
| **OSQP**     | No               | Parametric (matrix caching) only        |                                         |
| **HiGHS**    | No               | No batch API                            |                                         |
| **ECOS**     | No               | No batch API                            |                                         |
| **SCIP**     | No               | No batch API                            |                                         |
| **Xpress**   | Limited          | Mosel scenarios only                    |                                         |
| **cuOpt**    | Unclear          | Server API has batch, direct unclear    |                                         |

## Usage

```python
import cvxpy as cp
import numpy as np

# Define problem with parameters
n = 10
x = cp.Variable(n)
x0 = cp.Parameter(n)

prob = cp.Problem(
    cp.Minimize(cp.sum_squares(x - x0)),
    [x >= 0, cp.sum(x) <= 1]
)

# Create batch of parameter values
batch_size = 100
initial_states = np.random.randn(batch_size, n)

# Batch solve all problems
solutions = prob.solve_batch(
    {x0: initial_states},
    solver=cp.CLARABEL,
    num_threads=4
)

# Access results
for i, sol in enumerate(solutions):
    print(f"Problem {i}: status={sol.status}, opt_val={sol.opt_val:.4f}")
```

## Parameter Value Formats

The `param_values` argument accepts multiple formats:

```python
# NumPy array with batch dimension first
prob.solve_batch({param: np.random.randn(100, n)})

# List of values (works for dense and sparse)
prob.solve_batch({param: [val0, val1, val2, ...]})

# Multiple parameters
prob.solve_batch({
    param1: np.random.randn(100, n),
    param2: [sparse_matrix_0, sparse_matrix_1, ...]
})
```

## Solver-Specific Options

### Clarabel

```python
solutions = prob.solve_batch(
    {param: values},
    solver=cp.CLARABEL,
    num_threads=4,  # Number of parallel threads
)
```

### MOSEK

```python
solutions = prob.solve_batch(
    {param: values},
    solver=cp.MOSEK,
    num_threads=4,       # Total thread pool size
    threads_per_task=1,  # Threads per individual problem (default: 1)
)
```

## Requirements

- Problem must be DPP (Disciplined Parametrized Programming) compliant
- All parameters in `param_values` must have the same batch size
- Solver must support batch solving (raises `ValueError` otherwise)

## Performance

Batch solving provides speedups by:
1. **Amortized compilation**: Problem structure is compiled once
2. **Parallel execution**: Multiple problems solved simultaneously
3. **Shared resources**: Single license token (MOSEK), shared factorization patterns (Clarabel)

Typical speedups:
- **Clarabel**: 2-6x depending on problem size and thread count
- **MOSEK**: 1.5-2.5x depending on problem size and thread count
