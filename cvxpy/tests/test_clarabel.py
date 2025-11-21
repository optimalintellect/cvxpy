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
import clarabel
import numpy as np

import cvxpy as cp
import cvxpy.settings as s
from cvxpy.expressions.variable import Variable
from cvxpy.problems.problem import Problem
from cvxpy.reductions.solvers.conic_solvers.clarabel_conif import CLARABEL
from cvxpy.reductions.solvers.solver_inverse_data import SolverInverseData
from cvxpy.tests.base_test import BaseTest


class ClarabelTest(BaseTest):
    """
    Tests for Clarabel solver interface.
    """

    def setUp(self):
        """
        Sets up a problem as in test_compile.py function test_quad_form.
        """
        np.random.seed(42)
        P = np.random.randn(3, 3) - 1j*np.random.randn(3, 3)
        P = np.conj(P.T).dot(P)
        b = np.arange(3) + 3j*(np.arange(3) + 10)
        x = Variable(3, complex=True)
        self.value = cp.quad_form(b, P).value
        self.prob = Problem(cp.Minimize(cp.quad_form(x, P)), [x == b])

        # A dummy solution object (clarabel.DefaultSolution has no python constructor).
        self.solution: clarabel.DefaultSolution = type("DefaultSolution", (object,), {
            "x": [5.263500536056257e-12, 0.9999999999975893, 1.999999999997496,],
            "s": [0.0, 0.0, 0.0, ],
            "z": [-491.42397588400473, 279.36833425936913, 191.11806896900944, ],
            "status": CLARABEL.SOLVED,
            "obj_val": 21434.491423910207,
            "obj_val_dual": 21434.491423919164,
            "solve_time": 9.79e-5,
            "iterations": 0,
            "r_prim": 5.5338076549567106e-14,
            "r_dual": 1.5829159225003945e-15,
        })

        return super().setUp()

    def test_invert_when_solved(self):
        """Tests invert when a solution is present and solver status from clarabel is SOLVED."""
        solver = CLARABEL()
        _, _, inverse_data = self.prob.get_problem_data("clarabel")
        solver_inverse_data = SolverInverseData(inverse_data[-1], solver_options={})
        solution = solver.invert(self.solution, solver_inverse_data)
        self.assertEqual(s.OPTIMAL, solution.status)

    def test_invert_when_insufficient_progress_should_fail(self):
        """
        Tests invert when a solution is present and solver status from clarabel is
        InsufficientProgress.
        """
        solver = CLARABEL()
        _, _, inverse_data = self.prob.get_problem_data("clarabel")
        self.solution.status = CLARABEL.INSUFFICIENT_PROGRESS
        solution = solver.invert(self.solution, SolverInverseData(inverse_data[-1], {}))
        self.assertEqual(s.SOLVER_ERROR, solution.status)
        
    def test_invert_when_insufficient_progress_but_accept_unknown(self):
        """
        Tests invert when a solution is present and solver status from clarabel
        is InsufficientProgress but "accept_unknown" solver option was set to true.
        """
        solver = CLARABEL()
        _, _, inverse_data = self.prob.get_problem_data("clarabel")
        self.solution.status = CLARABEL.INSUFFICIENT_PROGRESS
        solver_inverse_data = SolverInverseData(inverse_data[-1],
                                                solver_options={CLARABEL.ACCEPT_UNKNOWN: True})
        solution = solver.invert(self.solution, solver_inverse_data)
        self.assertEqual(s.OPTIMAL_INACCURATE, solution.status)

    def test_invert_when_insufficient_progress_but_accept_unknown_and_no_solution(self):
        """
        Tests invert when a solution is present and solver status from clarabel
        is InsufficientProgress but "accept_unknown" solver option was set to true.
        Nevertheless, clarabel did not return a solution and therefore the resulting
        status should be SolverError.
        """
        solver = CLARABEL()
        _, _, inverse_data = self.prob.get_problem_data("clarabel")
        self.solution.status = CLARABEL.INSUFFICIENT_PROGRESS
        self.solution.x = None
        self.solution.z = None
        solver_inverse_data = SolverInverseData(inverse_data[-1],
                                                solver_options={"accept_unknown": True})
        solution = solver.invert(self.solution, solver_inverse_data)
        self.assertEqual(s.SOLVER_ERROR, solution.status)


class ClarabelBatchTest(BaseTest):
    """
    Tests for Clarabel batch solving functionality.
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
            solver=cp.CLARABEL
        )

        self.assertEqual(len(solutions), batch_size)
        for sol in solutions:
            self.assertEqual(sol.status, s.OPTIMAL)
            # Unconstrained least squares has optimal value 0
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
            solver=cp.CLARABEL
        )

        # Verify against individual solves
        for i in range(batch_size):
            x0.value = initial_states[i]
            individual_val = prob.solve(solver=cp.CLARABEL)
            self.assertAlmostEqual(solutions[i].opt_val, individual_val, places=4)

    def test_batch_solve_multiple_params(self):
        """Test batch solving with multiple parameters."""
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
        b_values = np.abs(np.random.randn(batch_size, m)) + 1  # ensure feasibility

        solutions = prob.solve_batch(
            {c: c_values, b: b_values},
            solver=cp.CLARABEL
        )

        self.assertEqual(len(solutions), batch_size)
        for sol in solutions:
            self.assertEqual(sol.status, s.OPTIMAL)

    def test_batch_solve_list_format(self):
        """Test batch solving with list format for parameters."""
        n = 3
        x = cp.Variable(n)
        x0 = cp.Parameter(n)

        prob = cp.Problem(cp.Minimize(cp.sum_squares(x - x0)))

        # Use list format instead of stacked array
        param_list = [np.array([1, 2, 3]), np.array([4, 5, 6]), np.array([7, 8, 9])]

        solutions = prob.solve_batch(
            {x0: param_list},
            solver=cp.CLARABEL
        )

        self.assertEqual(len(solutions), 3)
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

        # Test with different thread counts
        for num_threads in [1, 2, 4]:
            solutions = prob.solve_batch(
                {x0: initial_states},
                solver=cp.CLARABEL,
                num_threads=num_threads
            )
            self.assertEqual(len(solutions), batch_size)

    def test_batch_solve_empty(self):
        """Test batch solving with empty parameter values."""
        x = cp.Variable(2)
        x0 = cp.Parameter(2)
        prob = cp.Problem(cp.Minimize(cp.sum_squares(x - x0)))

        solutions = prob.solve_batch({x0: []}, solver=cp.CLARABEL)
        self.assertEqual(len(solutions), 0)

    def test_batch_solve_single_item(self):
        """Test batch solving with a single item."""
        x = cp.Variable(2)
        x0 = cp.Parameter(2)
        prob = cp.Problem(cp.Minimize(cp.sum_squares(x - x0)))

        solutions = prob.solve_batch(
            {x0: np.array([[1.0, 2.0]])},
            solver=cp.CLARABEL
        )
        self.assertEqual(len(solutions), 1)
        self.assertEqual(solutions[0].status, s.OPTIMAL)

    def test_batch_solve_inconsistent_batch_size_raises(self):
        """Test that inconsistent batch sizes raise an error."""
        x = cp.Variable(2)
        p1 = cp.Parameter(2)
        p2 = cp.Parameter(2)
        prob = cp.Problem(cp.Minimize(cp.sum_squares(x - p1 - p2)))

        with self.assertRaises(ValueError) as context:
            prob.solve_batch(
                {p1: np.random.randn(5, 2), p2: np.random.randn(3, 2)},
                solver=cp.CLARABEL
            )
        self.assertIn("Inconsistent batch sizes", str(context.exception))

    def test_batch_solve_non_batch_solver_raises(self):
        """Test that non-batch-capable solver raises an error."""
        x = cp.Variable(2)
        x0 = cp.Parameter(2)
        prob = cp.Problem(cp.Minimize(cp.sum_squares(x - x0)))

        # SCS doesn't support batch solving
        with self.assertRaises(ValueError) as context:
            prob.solve_batch(
                {x0: np.random.randn(5, 2)},
                solver=cp.SCS
            )
        self.assertIn("does not support batch solving", str(context.exception))

    def test_batch_solve_preserves_param_values(self):
        """Test that original parameter values are restored after batch solve."""
        x = cp.Variable(2)
        x0 = cp.Parameter(2)
        prob = cp.Problem(cp.Minimize(cp.sum_squares(x - x0)))

        original_value = np.array([100.0, 200.0])
        x0.value = original_value

        prob.solve_batch(
            {x0: np.random.randn(5, 2)},
            solver=cp.CLARABEL
        )

        np.testing.assert_array_equal(x0.value, original_value)

