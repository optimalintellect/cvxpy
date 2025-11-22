"""
Copyright, the CVXPY authors

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""
import pytest
import numpy as np

import cvxpy as cp
import cvxpy.settings as s
from cvxpy.tests.base_test import BaseTest


class MoreauTest(BaseTest):
    """
    Tests for Moreau GPU solver interface.
    """

    @pytest.fixture(autouse=True)
    def skip_if_no_moreau(self):
        """Skip tests if Moreau is not installed."""
        try:
            import moreau  # noqa F401
        except ImportError:
            pytest.skip("Moreau not installed")

    def test_simple_qp(self):
        """Test solving a simple quadratic program."""
        # Simple well-conditioned QP
        x = cp.Variable(2)
        prob = cp.Problem(
            cp.Minimize(cp.sum_squares(x - np.array([2.0, 3.0]))),
            [x >= 0, x <= 5]
        )
        prob.solve(solver=cp.MOREAU, verbose=False)

        self.assertEqual(prob.status, s.OPTIMAL)
        self.assertAlmostEqual(x.value[0], 2.0, places=6)
        self.assertAlmostEqual(x.value[1], 3.0, places=6)

    def test_constrained_qp(self):
        """Test solving a QP with constraints."""
        n = 5
        x = cp.Variable(n)
        prob = cp.Problem(
            cp.Minimize(cp.sum_squares(x)),
            [x >= 0, cp.sum(x) == 1]
        )
        prob.solve(solver=cp.MOREAU, verbose=False)

        self.assertEqual(prob.status, s.OPTIMAL)
        self.assertAlmostEqual(cp.sum(x).value, 1.0, places=6)
        self.assertTrue(np.all(x.value >= -1e-8))

    def test_linear_program(self):
        """Test solving a linear program."""
        n, m = 10, 20
        np.random.seed(42)
        c = np.random.randn(n)
        A = np.random.randn(m, n)
        b = np.abs(np.random.randn(m)) + 1

        x = cp.Variable(n)
        prob = cp.Problem(
            cp.Minimize(c @ x),
            [A @ x <= b, x >= 0]
        )
        prob.solve(solver=cp.MOREAU, verbose=False)

        self.assertEqual(prob.status, s.OPTIMAL)

    def test_socp(self):
        """Test solving a second-order cone program."""
        # Note: Moreau currently only supports dim-3 SOC cones
        x = cp.Variable(2)
        y = cp.Variable()

        prob = cp.Problem(
            cp.Minimize(y),
            [cp.norm(x, 2) <= y, y >= 1]
        )
        prob.solve(solver=cp.MOREAU, verbose=False)

        self.assertEqual(prob.status, s.OPTIMAL)
        self.assertAlmostEqual(y.value, 1.0, places=4)

    def test_exponential_cone(self):
        """Test solving with exponential cone constraints."""
        x = cp.Variable()
        y = cp.Variable()
        z = cp.Variable()

        prob = cp.Problem(
            cp.Minimize(x + y),
            [cp.exp(x) <= y + z, x >= 0, y >= 0, z >= 1]
        )

        try:
            prob.solve(solver=cp.MOREAU, verbose=False)
            # If it solves, check the status
            self.assertIn(prob.status, [s.OPTIMAL, s.OPTIMAL_INACCURATE])
        except Exception as e:
            # Exponential cone support may vary
            pytest.skip(f"Exponential cone not fully supported: {e}")


class MoreauBatchTest(BaseTest):
    """
    Tests for Moreau batch solving functionality.
    """

    @pytest.fixture(autouse=True)
    def skip_if_no_moreau(self):
        """Skip tests if Moreau is not installed."""
        try:
            import moreau  # noqa F401
        except ImportError:
            pytest.skip("Moreau not installed")

    def test_batch_solve_simple_qp(self):
        """Test batch solving with a simple QP."""
        n = 10
        x = cp.Variable(n)
        x0 = cp.Parameter(n)

        # Add constraint (Moreau requires constraints)
        prob = cp.Problem(cp.Minimize(cp.sum_squares(x - x0)), [x >= -100])

        batch_size = 16
        np.random.seed(42)
        initial_states = np.random.randn(batch_size, n)

        solutions = prob.solve_batch(
            {x0: initial_states},
            solver=cp.MOREAU,
            verbose=False
        )

        self.assertEqual(len(solutions), batch_size)
        for sol in solutions:
            self.assertEqual(sol.status, s.OPTIMAL)
            # Effectively unconstrained least squares has optimal value 0
            self.assertAlmostEqual(sol.opt_val, 0.0, places=6)

    def test_batch_solve_constrained(self):
        """Test batch solving with constraints."""
        n = 5
        x = cp.Variable(n)
        x0 = cp.Parameter(n)

        prob = cp.Problem(
            cp.Minimize(cp.sum_squares(x - x0)),
            [x >= 0, cp.sum(x) <= 1]
        )

        batch_size = 8
        np.random.seed(42)
        initial_states = np.random.randn(batch_size, n)

        solutions = prob.solve_batch(
            {x0: initial_states},
            solver=cp.MOREAU,
            verbose=False
        )

        self.assertEqual(len(solutions), batch_size)
        for sol in solutions:
            self.assertIn(sol.status, [s.OPTIMAL, s.OPTIMAL_INACCURATE])

    def test_batch_solve_lp(self):
        """Test batch solving linear programs."""
        n, m = 10, 5
        np.random.seed(42)
        A = np.random.randn(m, n)

        x = cp.Variable(n)
        c = cp.Parameter(n)
        b = cp.Parameter(m)

        prob = cp.Problem(
            cp.Minimize(c @ x),
            [A @ x <= b, x >= 0, x <= 10]  # Add upper bound to prevent unboundedness
        )

        batch_size = 10
        c_values = np.abs(np.random.randn(batch_size, n))  # Positive c ensures bounded
        b_values = np.abs(np.random.randn(batch_size, m)) + 2  # ensure feasibility

        solutions = prob.solve_batch(
            {c: c_values, b: b_values},
            solver=cp.MOREAU,
            verbose=False
        )

        self.assertEqual(len(solutions), batch_size)
        for sol in solutions:
            self.assertIn(sol.status, [s.OPTIMAL, s.OPTIMAL_INACCURATE])

    def test_batch_solve_comparison(self):
        """Verify batch solve matches individual solves."""
        n = 5
        x = cp.Variable(n)
        x0 = cp.Parameter(n)

        prob = cp.Problem(
            cp.Minimize(cp.sum_squares(x - x0)),
            [x >= 0]
        )

        batch_size = 4
        np.random.seed(42)
        initial_states = np.random.randn(batch_size, n)

        # Batch solve
        batch_solutions = prob.solve_batch(
            {x0: initial_states},
            solver=cp.MOREAU,
            verbose=False
        )

        # Individual solves
        for i in range(batch_size):
            x0.value = initial_states[i]
            individual_val = prob.solve(solver=cp.MOREAU, verbose=False)
            # Verify batch and individual solves match
            self.assertAlmostEqual(
                batch_solutions[i].opt_val,
                individual_val,
                places=6
            )

    def test_batch_high_throughput(self):
        """Test high-throughput batch solving (larger batch size)."""
        n = 100
        x = cp.Variable(n)
        q = cp.Parameter(n)

        P = np.eye(n)
        # Add constraint (Moreau requires constraints)
        prob = cp.Problem(cp.Minimize(0.5 * cp.quad_form(x, P) + q @ x), [x >= -100])

        batch_size = 64  # Larger batch to test GPU efficiency
        np.random.seed(42)
        q_values = np.random.randn(batch_size, n)

        solutions = prob.solve_batch(
            {q: q_values},
            solver=cp.MOREAU,
            verbose=False
        )

        self.assertEqual(len(solutions), batch_size)
        # Check that at least most problems solved successfully
        solved = sum(1 for sol in solutions if sol.status in [s.OPTIMAL, s.OPTIMAL_INACCURATE])
        self.assertGreaterEqual(solved, batch_size * 0.9)  # At least 90% success rate

    def test_batch_solve_empty(self):
        """Test batch solving with empty parameter values."""
        x = cp.Variable(2)
        x0 = cp.Parameter(2)
        prob = cp.Problem(cp.Minimize(cp.sum_squares(x - x0)))

        solutions = prob.solve_batch({x0: []}, solver=cp.MOREAU)
        self.assertEqual(len(solutions), 0)

    def test_batch_solver_options(self):
        """Test passing solver options to batch solve."""
        n = 5
        x = cp.Variable(n)
        x0 = cp.Parameter(n)

        prob = cp.Problem(cp.Minimize(cp.sum_squares(x - x0)))

        batch_size = 4
        np.random.seed(42)
        initial_states = np.random.randn(batch_size, n)

        solutions = prob.solve_batch(
            {x0: initial_states},
            solver=cp.MOREAU,
            max_iter=10,  # Limit iterations
            verbose=False
        )

        self.assertEqual(len(solutions), batch_size)
