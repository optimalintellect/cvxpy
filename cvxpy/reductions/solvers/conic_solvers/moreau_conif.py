"""
Copyright 2025, the CVXPY Authors

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
import numpy as np
import scipy.sparse as sp

import cvxpy.settings as s
from cvxpy.constraints import SOC, ExpCone, PowCone3D
from cvxpy.reductions.solution import Solution, failure_solution
from cvxpy.reductions.solvers import utilities
from cvxpy.reductions.solvers.conic_solvers.conic_solver import ConicSolver
from cvxpy.reductions.solvers.solver_inverse_data import SolverInverseData


def dims_to_solver_cones(cone_dims):
    """Convert CVXpy cone dimensions to Moreau cone specification.

    Parameters
    ----------
    cone_dims : ConeDims
        CVXpy cone dimensions object

    Returns
    -------
    moreau.Cones
        Moreau cone specification
    """
    import moreau

    cones = moreau.Cones()

    # Map CVXpy cone dimensions to Moreau cones
    cones.num_zero_cones = cone_dims.zero
    cones.num_nonneg_cones = cone_dims.nonneg

    # SOC cones: Moreau expects number of cones (each of dimension 3)
    # CVXpy provides list of dimensions
    # For now, only support dim-3 SOC cones
    cones.num_soc_cones = len(cone_dims.soc)
    if any(dim != 3 for dim in cone_dims.soc):
        raise ValueError("Moreau currently only supports second-order cones of dimension 3")

    # Exponential cones
    cones.num_exp_cones = cone_dims.exp

    # Power cones
    cones.num_power_cones = len(cone_dims.p3d)
    if cone_dims.p3d:
        cones.power_alphas = list(cone_dims.p3d)

    # Moreau does not support PSD cones yet
    if cone_dims.psd:
        raise ValueError("Moreau does not support PSD cones")

    # Moreau does not support generalized power cones yet
    if cone_dims.pnd:
        raise ValueError("Moreau does not support generalized power cones (PowConeND)")

    return cones


class MOREAU(ConicSolver):
    """An interface for the Moreau GPU solver.

    Moreau is a GPU-accelerated conic optimization solver based on the
    Clarabel interior-point algorithm, designed for high-throughput batch solving.
    """

    # Solver capabilities
    MIP_CAPABLE = False
    BATCH_CAPABLE = True
    REQUIRES_CONSTR = True  # Moreau requires at least one constraint
    SUPPORTED_CONSTRAINTS = ConicSolver.SUPPORTED_CONSTRAINTS + [SOC, ExpCone, PowCone3D]

    # Status messages from Moreau (based on solver/status.hpp)
    SOLVED = "Solved"
    PRIMAL_INFEASIBLE = "PrimalInfeasible"
    DUAL_INFEASIBLE = "DualInfeasible"
    ALMOST_SOLVED = "AlmostSolved"
    ALMOST_PRIMAL_INFEASIBLE = "AlmostPrimalInfeasible"
    ALMOST_DUAL_INFEASIBLE = "AlmostDualInfeasible"
    MAX_ITERATIONS = "MaxIterations"
    MAX_TIME = "MaxTime"
    NUMERICAL_ERROR = "NumericalError"
    INSUFFICIENT_PROGRESS = "InsufficientProgress"
    ACCEPT_UNKNOWN = "accept_unknown"

    STATUS_MAP = {
        SOLVED: s.OPTIMAL,
        PRIMAL_INFEASIBLE: s.INFEASIBLE,
        DUAL_INFEASIBLE: s.UNBOUNDED,
        ALMOST_SOLVED: s.OPTIMAL_INACCURATE,
        ALMOST_PRIMAL_INFEASIBLE: s.INFEASIBLE_INACCURATE,
        ALMOST_DUAL_INFEASIBLE: s.UNBOUNDED_INACCURATE,
        MAX_ITERATIONS: s.USER_LIMIT,
        MAX_TIME: s.USER_LIMIT,
        NUMERICAL_ERROR: s.SOLVER_ERROR,
        INSUFFICIENT_PROGRESS: s.SOLVER_ERROR
    }

    # Order of exponential cone arguments for solver
    EXP_CONE_ORDER = [0, 1, 2]

    def name(self):
        """The name of the solver."""
        return 'MOREAU'

    def import_solver(self) -> None:
        """Imports the solver."""
        import moreau  # noqa F401

    def supports_quad_obj(self) -> bool:
        """Moreau supports quadratic objective with conic constraints."""
        return True

    @staticmethod
    def extract_dual_value(result_vec, offset, constraint):
        """Extracts the dual value for constraint starting at offset."""
        return utilities.extract_dual_value(result_vec, offset, constraint)

    def invert(self, solution, inverse_data):
        """Returns the solution to the original problem given the inverse_data."""
        attr = {}
        status_map = self.STATUS_MAP.copy()

        # Handle accept_unknown option
        if isinstance(inverse_data, SolverInverseData) and \
           MOREAU.ACCEPT_UNKNOWN in inverse_data.solver_options and \
           solution.x is not None and solution.z is not None:
            status_map["InsufficientProgress"] = s.OPTIMAL_INACCURATE

        status = status_map.get(str(solution.status), s.SOLVER_ERROR)
        attr[s.SOLVE_TIME] = solution.solve_time
        attr[s.NUM_ITERS] = solution.iterations

        if status in s.SOLUTION_PRESENT:
            primal_val = solution.obj_val
            opt_val = primal_val + inverse_data[s.OFFSET]
            primal_vars = {
                inverse_data[self.VAR_ID]: solution.x
            }
            eq_dual_vars = utilities.get_dual_values(
                solution.z[:inverse_data[ConicSolver.DIMS].zero],
                self.extract_dual_value,
                inverse_data[self.EQ_CONSTR]
            )
            ineq_dual_vars = utilities.get_dual_values(
                solution.z[inverse_data[ConicSolver.DIMS].zero:],
                self.extract_dual_value,
                inverse_data[self.NEQ_CONSTR]
            )
            dual_vars = {}
            dual_vars.update(eq_dual_vars)
            dual_vars.update(ineq_dual_vars)
            return Solution(status, opt_val, primal_vars, dual_vars, attr)
        else:
            return failure_solution(status, attr)

    @staticmethod
    def parse_solver_opts(verbose, opts, settings=None):
        """Parse solver options and create Moreau settings object.

        Parameters
        ----------
        verbose : bool
            Enable verbose output
        opts : dict
            Solver-specific options
        settings : moreau.Settings, optional
            Existing settings object to update

        Returns
        -------
        moreau.Settings
            Configured settings object
        """
        import moreau

        if settings is None:
            settings = moreau.Settings()

        settings.verbose = verbose

        # Remove CVXpy-specific options
        keys = list(opts.keys())
        if "use_quad_obj" in keys:
            keys.remove("use_quad_obj")
        if MOREAU.ACCEPT_UNKNOWN in keys:
            keys.remove(MOREAU.ACCEPT_UNKNOWN)

        # Map common option names to Moreau settings
        option_map = {
            'max_iter': 'max_iter',
            'time_limit': 'time_limit',
            'tol_gap_abs': 'tol_gap_abs',
            'tol_gap_rel': 'tol_gap_rel',
            'tol_feas': 'tol_feas',
            'tol_infeas_abs': 'tol_infeas_abs',
            'tol_infeas_rel': 'tol_infeas_rel',
            'max_step_fraction': 'max_step_fraction',
        }

        for opt in keys:
            moreau_opt = option_map.get(opt, opt)
            try:
                setattr(settings, moreau_opt, opts[opt])
            except TypeError as e:
                raise TypeError(f"Moreau: Incorrect type for setting '{opt}'.") from e
            except AttributeError as e:
                raise TypeError(f"Moreau: unrecognized solver setting '{opt}'.") from e

        return settings

    def solve_via_data(self, data, warm_start: bool, verbose: bool, solver_opts, solver_cache=None):
        """Returns the result of the call to the solver.

        Parameters
        ----------
        data : dict
            Data generated via an apply call.
        warm_start : bool
            Whether to warm_start Moreau (not currently supported).
        verbose : bool
            Control the verbosity.
        solver_opts : dict
            Moreau-specific solver options.
        solver_cache : dict, optional
            Cache for solver objects (not currently used).

        Returns
        -------
        Solution object from Moreau solver.
        """
        import moreau

        A = data[s.A]
        b = data[s.B]
        q = data[s.C]

        if s.P in data:
            P = data[s.P]
        else:
            nvars = q.size
            # Create empty sparse matrix with proper structure
            P = sp.csr_array((nvars, nvars))

        # Convert to CSR format and take upper triangle
        P = sp.triu(P, format='csr')
        A = A.tocsr()

        # Convert cone dimensions
        cones = dims_to_solver_cones(data[ConicSolver.DIMS])

        # Parse settings
        settings = self.parse_solver_opts(verbose, solver_opts if solver_opts else {})

        # Create solver
        solver = moreau.Solver(
            n=P.shape[0],
            m=A.shape[0],
            batch_size=1,
            P_row_offsets=P.indptr.astype(np.int64),
            P_col_indices=P.indices.astype(np.int64),
            A_row_offsets=A.indptr.astype(np.int64),
            A_col_indices=A.indices.astype(np.int64),
            cones=cones,
            settings=settings
        )

        # Solve
        result = solver.solve(
            P_values=P.data,
            A_values=A.data,
            q=q,
            b=b
        )

        # Convert result to solution object
        import moreau

        # Handle batched vs non-batched results
        # Check if status is an array with shape (need to handle scalar arrays)
        if isinstance(result['status'], np.ndarray) and result['status'].ndim > 0:
            status_val = moreau.SolverStatus(int(result['status'][0]))
        else:
            # Scalar or 0-dimensional array
            status_val = moreau.SolverStatus(int(result['status']))

        class MoreauSolution:
            def __init__(self, result_dict, status):
                self.x = result_dict['x']
                self.s = result_dict['s']
                self.z = result_dict['z']
                self.status = status.name
                self.iterations = result_dict['iterations']
                self.solve_time = result_dict['solve_time']

                # Handle batched vs non-batched objective values
                if isinstance(result_dict['obj_val'], np.ndarray) and result_dict['obj_val'].ndim > 0:
                    self.obj_val = result_dict['obj_val'][0]
                elif isinstance(result_dict['obj_val'], list):
                    self.obj_val = result_dict['obj_val'][0]
                else:
                    # Scalar or 0-dimensional array
                    self.obj_val = float(result_dict['obj_val'])

        return MoreauSolution(result, status_val)

    def solve_batch_via_data(
        self,
        batch_data: list,
        warm_start: bool,
        verbose: bool,
        solver_opts,
        solver_cache=None
    ):
        """Solve a batch of problems with the same structure but different data.

        Parameters
        ----------
        batch_data : list of dict
            List of data dicts, each generated via an apply call.
            All problems must have the same sparsity pattern.
        warm_start : bool
            Not used for batch solving.
        verbose : bool
            Control the verbosity.
        solver_opts : dict
            Moreau-specific solver options.
        solver_cache : dict, optional
            Cache for solver objects.

        Returns
        -------
        list
            List of solution objects from Moreau solver for each problem.
        """
        import moreau

        if not batch_data:
            return []

        # Get dimensions and patterns from first problem
        first = batch_data[0]
        A_pattern = first[s.A].tocsr()
        q_first = first[s.C]
        n = q_first.size
        m = A_pattern.shape[0]

        if s.P in first:
            P_pattern = sp.triu(first[s.P]).tocsr()
        else:
            P_pattern = sp.csr_array((n, n))

        # Convert cone dimensions
        cones = dims_to_solver_cones(first[ConicSolver.DIMS])

        # Parse settings
        settings = self.parse_solver_opts(verbose, solver_opts if solver_opts else {})

        # Determine batch mode based on data patterns
        # Check if all problems share the same P and A matrices
        same_P_A = True
        for data in batch_data[1:]:
            if s.P in data:
                P_data = sp.triu(data[s.P]).tocsr()
                if not np.allclose(P_pattern.data, P_data.data):
                    same_P_A = False
                    break
            A_data = data[s.A].tocsr()
            if not np.allclose(A_pattern.data, A_data.data):
                same_P_A = False
                break

        batch_size = len(batch_data)

        # Create solver
        solver = moreau.Solver(
            n=n,
            m=m,
            batch_size=batch_size,
            P_row_offsets=P_pattern.indptr.astype(np.int64),
            P_col_indices=P_pattern.indices.astype(np.int64),
            A_row_offsets=A_pattern.indptr.astype(np.int64),
            A_col_indices=A_pattern.indices.astype(np.int64),
            cones=cones,
            settings=settings
        )

        # Prepare batched data
        if same_P_A:
            # Use solveBatchRHS: single P/A, batched q/b
            q_batch = np.hstack([data[s.C] for data in batch_data])
            b_batch = np.hstack([data[s.B] for data in batch_data])

            result = solver.solve(
                P_values=P_pattern.data,
                A_values=A_pattern.data,
                q=q_batch,
                b=b_batch
            )
        else:
            # Fully batched: all matrices are different
            P_values_batch = np.hstack([
                (sp.triu(data[s.P]).tocsr().data if s.P in data else np.zeros(P_pattern.nnz))
                for data in batch_data
            ])
            A_values_batch = np.hstack([data[s.A].tocsr().data for data in batch_data])
            q_batch = np.hstack([data[s.C] for data in batch_data])
            b_batch = np.hstack([data[s.B] for data in batch_data])

            result = solver.solve(
                P_values=P_values_batch,
                A_values=A_values_batch,
                q=q_batch,
                b=b_batch
            )

        # Convert results to list of solution objects
        class MoreauSolution:
            def __init__(self, x, s, z, status, iterations, solve_time, obj_val):
                self.x = x
                self.s = s
                self.z = z
                self.status = status
                self.iterations = iterations
                self.solve_time = solve_time
                self.obj_val = obj_val

        results = []
        for i in range(batch_size):
            status_val = moreau.SolverStatus(int(result['status'][i]))
            sol = MoreauSolution(
                x=result['x'][i],
                s=result['s'][i],
                z=result['z'][i],
                status=status_val.name,
                iterations=result['iterations'],
                solve_time=result['solve_time'] / batch_size,  # Approximate per-problem time
                obj_val=result['obj_val'][i] if isinstance(result['obj_val'], (list, np.ndarray)) else result['obj_val']
            )
            results.append(sol)

        return results

    def cite(self, data):
        """Returns bibtex citation for the solver.

        Parameters
        ----------
        data : dict
            Data generated via an apply call.

        Returns
        -------
        str
            BibTeX citation
        """
        return """@misc{moreau2025,
  title={Moreau: GPU-Accelerated Conic Optimization},
  year={2025},
  note={https://github.com/user/moreau}
}"""
