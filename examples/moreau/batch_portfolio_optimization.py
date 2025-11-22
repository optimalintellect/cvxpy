"""
Batch Portfolio Optimization with Moreau GPU Solver

This example demonstrates how to use the Moreau GPU solver with CVXpy
to solve multiple portfolio optimization problems in parallel.

The Moreau solver is particularly well-suited for batch solving, where
you need to solve many similar problems with different parameters. This
is common in:
- Monte Carlo simulations
- Parameter sweeps
- Real-time trading with multiple scenarios
- Risk analysis under different market conditions
"""

import numpy as np
import cvxpy as cp
import time

def solve_portfolio_batch():
    """
    Solve batch portfolio optimization with different return scenarios.

    Problem:
        minimize    risk (variance)
        subject to  expected return >= target_return
                    sum(weights) == 1
                    weights >= 0 (long-only)
    """
    # Problem setup
    n_assets = 50
    np.random.seed(42)

    # Generate covariance matrix (same for all scenarios)
    A = np.random.randn(n_assets, n_assets)
    Sigma = A.T @ A / n_assets + 0.1 * np.eye(n_assets)

    # Define the optimization problem
    w = cp.Variable(n_assets)  # portfolio weights
    mu = cp.Parameter(n_assets)  # expected returns (parameterized)
    target_return = cp.Parameter()  # target return (parameterized)

    risk = cp.quad_form(w, Sigma)
    constraints = [
        mu @ w >= target_return,  # return constraint
        cp.sum(w) == 1,           # budget constraint
        w >= 0                     # long-only
    ]

    prob = cp.Problem(cp.Minimize(risk), constraints)

    # Generate different return scenarios
    batch_size = 128  # Solve 128 scenarios in parallel
    print(f"\n{'='*70}")
    print(f"BATCH PORTFOLIO OPTIMIZATION - {batch_size} scenarios")
    print(f"{'='*70}")
    print(f"Assets: {n_assets}")
    print(f"Batch size: {batch_size}")

    # Different expected return scenarios (e.g., from Monte Carlo)
    mu_scenarios = np.random.randn(batch_size, n_assets) * 0.05 + 0.10
    target_returns = np.linspace(0.08, 0.12, batch_size)

    # Batch solve with Moreau
    print(f"\nSolving {batch_size} problems with Moreau GPU solver...")
    start = time.time()
    solutions = prob.solve_batch(
        {mu: mu_scenarios, target_return: target_returns},
        solver=cp.MOREAU,
        verbose=False
    )
    moreau_time = time.time() - start

    print(f"Moreau batch solve time: {moreau_time:.3f} seconds")
    print(f"Time per problem: {moreau_time / batch_size * 1000:.2f} ms")
    print(f"Throughput: {batch_size / moreau_time:.1f} problems/second")

    # Check solution quality
    solved = sum(1 for sol in solutions if sol.status == cp.OPTIMAL)
    print(f"\nSuccessfully solved: {solved}/{batch_size} ({100*solved/batch_size:.1f}%)")

    # Display some results
    print(f"\n{'='*70}")
    print("SAMPLE RESULTS")
    print(f"{'='*70}")
    print(f"{'Scenario':<12} {'Target Return':<15} {'Optimal Risk':<15} {'Status':<10}")
    print(f"{'-'*70}")

    for i in [0, batch_size//4, batch_size//2, 3*batch_size//4, batch_size-1]:
        if solutions[i].status == cp.OPTIMAL:
            print(f"{i:<12} {target_returns[i]:<15.4f} {solutions[i].opt_val:<15.6f} {'OPTIMAL':<10}")
        else:
            print(f"{i:<12} {target_returns[i]:<15.4f} {'N/A':<15} {solutions[i].status:<10}")

    # Compare with sequential solving
    print(f"\n{'='*70}")
    print("COMPARISON: Sequential vs Batch Solving")
    print(f"{'='*70}")

    # Solve a few problems sequentially for comparison
    n_sequential = min(10, batch_size)
    print(f"\nSolving {n_sequential} problems sequentially...")
    start = time.time()
    for i in range(n_sequential):
        mu.value = mu_scenarios[i]
        target_return.value = target_returns[i]
        prob.solve(solver=cp.MOREAU, verbose=False)
    sequential_time = time.time() - start

    print(f"Sequential time for {n_sequential} problems: {sequential_time:.3f} seconds")
    print(f"Time per problem: {sequential_time / n_sequential * 1000:.2f} ms")

    # Extrapolate to full batch
    estimated_sequential_time = sequential_time / n_sequential * batch_size
    print(f"\nEstimated time for {batch_size} problems sequentially: {estimated_sequential_time:.3f} seconds")
    print(f"Speedup from batch solving: {estimated_sequential_time / moreau_time:.1f}x")

    print(f"\n{'='*70}")

    return solutions


def solve_portfolio_simple():
    """
    Solve a single portfolio optimization problem.
    Demonstrates basic usage of Moreau solver with CVXpy.
    """
    print(f"\n{'='*70}")
    print("SINGLE PORTFOLIO OPTIMIZATION")
    print(f"{'='*70}")

    # Problem parameters
    n = 20
    np.random.seed(42)

    # Covariance matrix
    A = np.random.randn(n, n)
    Sigma = A.T @ A / n

    # Expected returns
    mu = np.random.randn(n) * 0.05 + 0.10

    # Optimization problem
    w = cp.Variable(n)
    risk = cp.quad_form(w, Sigma)
    expected_return = mu @ w

    prob = cp.Problem(
        cp.Minimize(risk),
        [
            expected_return >= 0.12,
            cp.sum(w) == 1,
            w >= 0
        ]
    )

    # Solve with Moreau
    prob.solve(solver=cp.MOREAU, verbose=True)

    print(f"\nStatus: {prob.status}")
    print(f"Optimal risk: {prob.value:.6f}")
    print(f"Expected return: {expected_return.value:.4f}")
    print(f"\nPortfolio weights (top 5):")
    top_indices = np.argsort(w.value)[-5:][::-1]
    for i in top_indices:
        print(f"  Asset {i}: {w.value[i]:.4f}")

    print(f"\n{'='*70}")


if __name__ == "__main__":
    # Check if Moreau is available
    try:
        import moreau
        print("Moreau GPU solver detected!")
        print(f"Moreau version: {moreau.__version__}")
    except ImportError:
        print("ERROR: Moreau solver not found!")
        print("Please install Moreau first:")
        print("  cd /path/to/moreau")
        print("  pip install -e .")
        exit(1)

    # Run examples
    print("\n" + "="*70)
    print("CVXPY + MOREAU GPU SOLVER EXAMPLES")
    print("="*70)

    # Single problem
    solve_portfolio_simple()

    # Batch problems
    solutions = solve_portfolio_batch()

    print("\n" + "="*70)
    print("DONE")
    print("="*70)
