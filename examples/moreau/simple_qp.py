"""
Simple Quadratic Programming with Moreau GPU Solver

This example demonstrates basic usage of the Moreau GPU solver with CVXpy.
"""

import numpy as np
import cvxpy as cp

def main():
    print("="*60)
    print("MOREAU GPU SOLVER - Simple QP Example")
    print("="*60)

    # Problem data
    n = 10
    np.random.seed(42)

    # Generate a positive definite matrix
    A = np.random.randn(n, n)
    P = A.T @ A + np.eye(n)
    q = np.random.randn(n)

    # Define the optimization problem
    # minimize    (1/2) x'Px + q'x
    # subject to  x >= 0
    #             sum(x) <= 1
    x = cp.Variable(n)

    objective = 0.5 * cp.quad_form(x, P) + q @ x
    constraints = [
        x >= 0,
        cp.sum(x) <= 1
    ]

    prob = cp.Problem(cp.Minimize(objective), constraints)

    # Solve with Moreau
    print("\nSolving with MOREAU...")
    prob.solve(solver=cp.MOREAU, verbose=True)

    # Display results
    print("\n" + "="*60)
    print("RESULTS")
    print("="*60)
    print(f"Status: {prob.status}")
    print(f"Optimal value: {prob.value:.6f}")
    print(f"\nOptimal x:")
    print(x.value)
    print(f"\nConstraint check:")
    print(f"  All x >= 0: {np.all(x.value >= -1e-6)}")
    print(f"  sum(x) = {np.sum(x.value):.6f} <= 1")
    print("="*60)


if __name__ == "__main__":
    try:
        import moreau
    except ImportError:
        print("ERROR: Moreau solver not installed!")
        print("Please install moreau first.")
        exit(1)

    main()
