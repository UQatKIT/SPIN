"""This module provides a modern wrapper to Hippylib's Newton-CG solver.

Parametrization is done via data classes, all input and output vectors are numpy arrays.

Classes:
    SolverSettings: Configuration of the Newton-CG solver.
    SolverResult: Data class for storage of solver results
    NewtonCGSolver: Wrapper class for the in-exact Newton-CG solver implementation in Hippylib.
"""

from dataclasses import dataclass
from numbers import Real
from typing import Annotated

import hippylib as hl
import numpy as np
import numpy.typing as npt
from beartype.vale import Is

from spin.fenics import converter as fex_converter


# ==================================================================================================
@dataclass
class SolverSettings:
    """Configuration of the Newton-CG solver.

    All attributes have default values.

    Attributes:
        relative_tolerance: Relative tolerance for the gradient norm (compared to initial guess).
        absolute_tolerance: Absolute tolerance for the gradient norm.
        gradient_projection_tolerance: Tolerance for the inner product (g,dm), where g is the
            current gradient and dm the search direction.
        max_num_newton_iterations: Maximum number of Newton iterations.
        num_gauss_newton_iterations: Number of Gauss-Newton iterations performed initially, before
            switching to full Newton.
        coarsest_tolerance_cg: Termination tolerance for the conjugate gradient linear solver.
        max_num_cg_iterations: Maximum number of conjugate gradient iterations.
        armijo_line_search_constant: Constant for the Armijo line search.
        max_num_line_search_iterations: Maximum number of line search iterations.
        verbose: Whether to print the solver output.
    """

    relative_tolerance: Annotated[Real, Is[lambda x: 0 < x < 1]] = 1e-6
    absolute_tolerance: Annotated[Real, Is[lambda x: 0 < x < 1]] = 1e-12
    gradient_projection_tolerance: Annotated[Real, Is[lambda x: 0 < x < 1]] = 1e-18
    max_num_newton_iterations: Annotated[int, Is[lambda x: x > 0]] = 20
    num_gauss_newton_iterations: Annotated[int, Is[lambda x: x > 0]] = 5
    coarsest_tolerance_cg: Annotated[Real, Is[lambda x: 0 < x < 1]] = 5e-1
    max_num_cg_iterations: Annotated[int, Is[lambda x: x > 0]] = 100
    armijo_line_search_constant: Annotated[Real, Is[lambda x: 0 < x < 1]] = 1e-4
    max_num_line_search_iterations: Annotated[int, Is[lambda x: x > 0]] = 10
    verbose: bool = True


@dataclass
class SolverResult:
    """Data class for storage of solver results.

    Attributes:
        optimal_parameter: Optimal parameter, found by the optimizer.
        forward_solution: Solution of the PDE problem for the optimal parameter.
        adjoint_solution: Solution of the adjoint problem for the optimal parameter.
        converged: Whether the solver has converged.
        num_iterations: Number of Newton iterations.
        termination_reason: Reason for termination.
        final_gradient_norm: Final gradient norm.
    """

    optimal_parameter: npt.NDArray[np.floating]
    forward_solution: npt.NDArray[np.floating]
    adjoint_solution: npt.NDArray[np.floating]
    converged: bool
    num_iterations: Annotated[int, Is[lambda x: x >= 0]]
    termination_reason: str
    final_gradient_norm: Annotated[Real, Is[lambda x: x >= 0]]


# ==================================================================================================
class NewtonCGSolver:
    """Wrapper class for the in-exact Newton-CG solver implementation in Hippylib.

    This class mainly exists to provide a more modern, convenient, and consistent interface to the
    underlying hippylib functionality.
    The incomplite Newton-CG method constitutes a combination of algorithms that efficienly solve
    large-scale optimization problems, with good scalability in terms of the number of degrees of
    freedom. The outer Newton iterations are known to converge independently of the problem size for
    a wide range of applications. The linear system at each Newton step is solved inexactly using
    the conjugate gradient (CG) method. The CG solver is terminated early according to Steihaug and
    Eisenstat-Walker stopping criteria. This ensures termination independent ly of the problem size
    as well. For globalization, armijo line search is utilized.

    For more information on the implementation of the solver in hippylib,
    check out the
    [`NewtonCG`](https://hippylib.readthedocs.io/en/latest/hippylib.algorithms.html#module-hippylib.algorithms.NewtonCG)
    documentation.

    Methods:
        run: Start the solver with an initial guess.
    """

    # ----------------------------------------------------------------------------------------------
    def __init__(self, solver_settings: SolverSettings, inference_model: hl.Model) -> None:
        """Constructor, initializing solver according to settings.

        Args:
            solver_settings (SolverSettings): Solver configuration.
            inference_model (hl.Model): Hippylib inference model defining the optimization problem.
        """
        hippylib_parameterlist = hl.ReducedSpaceNewtonCG_ParameterList()
        hippylib_parameterlist["rel_tolerance"] = solver_settings.relative_tolerance
        hippylib_parameterlist["abs_tolerance"] = solver_settings.absolute_tolerance
        hippylib_parameterlist["gdm_tolerance"] = solver_settings.gradient_projection_tolerance
        hippylib_parameterlist["max_iter"] = solver_settings.max_num_newton_iterations
        hippylib_parameterlist["GN_iter"] = solver_settings.num_gauss_newton_iterations
        hippylib_parameterlist["cg_coarse_tolerance"] = solver_settings.coarsest_tolerance_cg
        hippylib_parameterlist["cg_max_iter"] = solver_settings.max_num_cg_iterations
        if solver_settings.verbose:
            hippylib_parameterlist["print_level"] = 0
        else:
            hippylib_parameterlist["print_level"] = -1
        hippylib_parameterlist["LS"]["c_armijo"] = solver_settings.armijo_line_search_constant
        hippylib_parameterlist["LS"]["max_backtracking_iter"] = (
            solver_settings.max_num_line_search_iterations
        )
        self._hl_newtoncgsolver = hl.ReducedSpaceNewtonCG(inference_model, hippylib_parameterlist)
        self._inference_model = inference_model

    # ----------------------------------------------------------------------------------------------
    def solve(self, initial_guess: npt.NDArray[np.floating]) -> SolverResult:
        """Run the solver, given an initial guess.

        Args:
            initial_guess (npt.NDArray[np.floating]): Initial guess for the optimization problem.

        Raises:
            Checks if the initial guess has the correct size.

        Returns:
            SolverResult: Optimal solution and metadata.
        """
        function_space_variables = self._inference_model.problem.Vh[hl.STATE]
        function_space_parameter = self._inference_model.problem.Vh[hl.PARAMETER]
        if not initial_guess.size == function_space_parameter.dim():
            raise ValueError(
                f"Initial guess has wrong size {initial_guess.size}, "
                f"expected {function_space_parameter.dim()}."
            )

        initial_function = fex_converter.convert_to_dolfin(
            initial_guess.copy(), function_space_parameter
        )
        initial_vector = initial_function.vector()
        forward_solution, optimal_parameter, adjoint_solution = self._hl_newtoncgsolver.solve(
            [None, initial_vector, None]
        )
        optimal_parameter = fex_converter.convert_to_numpy(
            optimal_parameter, function_space_parameter
        )
        forward_solution = fex_converter.convert_to_numpy(
            forward_solution, function_space_variables
        )
        adjoint_solution = fex_converter.convert_to_numpy(
            adjoint_solution, function_space_variables
        )
        solver_result = SolverResult(
            optimal_parameter=optimal_parameter,
            forward_solution=forward_solution,
            adjoint_solution=adjoint_solution,
            converged=self._hl_newtoncgsolver.converged,
            num_iterations=self._hl_newtoncgsolver.it,
            termination_reason=self._hl_newtoncgsolver.termination_reasons[
                self._hl_newtoncgsolver.reason
            ],
            final_gradient_norm=self._hl_newtoncgsolver.final_grad_norm,
        )
        return solver_result
