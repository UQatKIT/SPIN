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
    solution: npt.NDArray[np.floating]
    converged: bool
    num_iterations: Annotated[int, Is[lambda x: x >= 0]]
    termination_reason: str
    final_gradient_norm: Annotated[Real, Is[lambda x: x >= 0]]


# ==================================================================================================
class NewtonCGSolver:
    # ----------------------------------------------------------------------------------------------
    def __init__(self, solver_settings: SolverSettings, inference_model: hl.Model) -> None:
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
        function_space_parameter = self._inference_model.problem.Vh[hl.PARAMETER]
        initial_function = fex_converter.convert_to_dolfin(
            initial_guess.copy(), function_space_parameter
        )
        initial_vector = initial_function.vector()
        _, result_vector, _ = self._hl_newtoncgsolver.solve([None, initial_vector, None])
        result_vector = fex_converter.convert_to_numpy(result_vector, function_space_parameter)

        solver_result = SolverResult(
            solution=result_vector,
            converged=self._hl_newtoncgsolver.converged,
            num_iterations=self._hl_newtoncgsolver.it,
            termination_reason=self._hl_newtoncgsolver.termination_reasons[
                self._hl_newtoncgsolver.reason
            ],
            final_gradient_norm=self._hl_newtoncgsolver.final_grad_norm,
        )
        return solver_result
