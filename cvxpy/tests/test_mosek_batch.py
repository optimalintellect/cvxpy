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
import unittest

import numpy as np

import cvxpy as cp
import cvxpy.settings as s

try:
    import mosek
    MOSEK_AVAILABLE = True
except ImportError:
    MOSEK_AVAILABLE = False


@unittest.skipUnless(MOSEK_AVAILABLE, "MOSEK not installed")
class MosekBatchTest(unittest.TestCase):
    """
    Tests for MOSEK batch solving functionality.
    """

    def test_batch_solve_simple_qp(self):
        """Test batch solving with a simple unconstrained QP."""
        n = 5
        x = cp.Variable(n)
        x0 = cp.Parameter(n)

        prob = cp.Problem(cp.Minimize(cp.sum_squares(x - x0)))

        batch_size = 10
        np.random.seed(42)
        initial_states = np.random.randn(batch_size, n)

        solutions = prob.solve_batch(
            {x0: initial_states},
            solver=cp.MOSEK
        )

        self.assertEqual(len(solutions), batch_size)
        for sol in solutions:
            self.assertEqual(sol.status, s.OPTIMAL)
            self.assertAlmostEqual(sol.opt_val, 0.0, places=5)

    def test_batch_solve_constrained_qp(self):
        """Test batch solving with constraints."""
        n = 5
        x = cp.Variable(n)
        x0 = cp.Parameter(n)

        prob = cp.Problem(
            cp.Minimize(cp.sum_squares(x - x0)),
            [x >= 0, cp.sum(x) <= 1]
        )

        batch_size = 5
        np.random.seed(42)
        initial_states = np.random.randn(batch_size, n)

        solutions = prob.solve_batch(
            {x0: initial_states},
            solver=cp.MOSEK
        )

        # Verify against individual solves
        for i in range(batch_size):
            x0.value = initial_states[i]
            individual_val = prob.solve(solver=cp.MOSEK)
            self.assertAlmostEqual(solutions[i].opt_val, individual_val, places=4)

    def test_batch_solve_lp(self):
        """Test batch solving with LP problems."""
        n, m = 10, 20
        x = cp.Variable(n)
        c = cp.Parameter(n)
        b = cp.Parameter(m)
        A = np.random.randn(m, n)

        prob = cp.Problem(
            cp.Minimize(c @ x),
            [A @ x <= b, x >= 0]
        )

        batch_size = 8
        np.random.seed(42)
        c_values = np.random.randn(batch_size, n)
        b_values = np.abs(np.random.randn(batch_size, m)) + 1

        solutions = prob.solve_batch(
            {c: c_values, b: b_values},
            solver=cp.MOSEK
        )

        self.assertEqual(len(solutions), batch_size)
        for sol in solutions:
            self.assertEqual(sol.status, s.OPTIMAL)

    def test_batch_solve_with_threads(self):
        """Test batch solving with multiple threads."""
        n = 10
        x = cp.Variable(n)
        x0 = cp.Parameter(n)

        prob = cp.Problem(
            cp.Minimize(cp.sum_squares(x - x0)),
            [x >= -1, x <= 1]
        )

        batch_size = 20
        np.random.seed(42)
        initial_states = np.random.randn(batch_size, n)

        # Test with different thread configurations
        solutions = prob.solve_batch(
            {x0: initial_states},
            solver=cp.MOSEK,
            num_threads=4,
            threads_per_task=1
        )
        self.assertEqual(len(solutions), batch_size)

    def test_batch_solve_empty(self):
        """Test batch solving with empty parameter values."""
        x = cp.Variable(2)
        x0 = cp.Parameter(2)
        prob = cp.Problem(cp.Minimize(cp.sum_squares(x - x0)))

        solutions = prob.solve_batch({x0: []}, solver=cp.MOSEK)
        self.assertEqual(len(solutions), 0)

    def test_batch_solve_socp(self):
        """Test batch solving with SOCP constraints."""
        n = 5
        x = cp.Variable(n)
        t = cp.Variable()
        x0 = cp.Parameter(n)

        prob = cp.Problem(
            cp.Minimize(t),
            [cp.norm(x - x0) <= t, x >= 0]
        )

        batch_size = 5
        np.random.seed(42)
        initial_states = np.random.randn(batch_size, n)

        solutions = prob.solve_batch(
            {x0: initial_states},
            solver=cp.MOSEK
        )

        self.assertEqual(len(solutions), batch_size)
        for sol in solutions:
            self.assertEqual(sol.status, s.OPTIMAL)


if __name__ == '__main__':
    unittest.main()
