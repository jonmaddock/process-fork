"""An adapter for different solvers."""

import importlib
import logging
from abc import ABC, abstractmethod

import cvxpy
import numpy as np
from pyvmcon import (
    AbstractProblem,
    LineSearchConvergenceException,
    QSPSolverException,
    Result,
    VMCONConvergenceException,
    solve,
)
from scipy.optimize import fsolve
import nlopt
import time
from scipy import optimize
import os
import pandas as pd

from process.core.exceptions import ProcessValueError
from process.core.model import DataStructure
from process.core.solver.evaluators import Evaluators
from process.core.solver.iteration_variables import set_scaled_iteration_variable
from scipy.integrate import solve_ivp
from process.core.solver import constraints
from process.models.physics import impurity_radiation
from pathlib import Path
from scipy.optimize import minimize
from termcolor import colored
from process import iteration_variables


logger = logging.getLogger(__name__)
DEBUG_DATAFRAME_OUTPUT = True


class _Solver(ABC):
    """Base class for different solver implementations."""

    def __init__(self, *, data: DataStructure):
        """Initialise a solver."""
        # Exit code for the solver
        self.ifail = 0
        self.data = data
        self.tolerance = self.data.numerics.epsvmc
        self.b: float | None = None
        self.convergence_parameter: float | None = None
        self.maxcal = self.data.globals.maxcal

    def set_evaluators(self, evaluators: Evaluators):
        """Set objective and constraint functions and their gradient evaluators.

        Parameters
        ----------
        evaluators : Evaluators
            objective and constraint evaluators

        """
        self.evaluators = evaluators

    def set_opt_params(self, x_0: np.ndarray):
        """Define the initial optimisation parameters.

        Parameters
        ----------
        x_0 : np.ndarray
            optimisation parameters vector

        """
        self.x_0 = x_0

    def set_bounds(
        self,
        bndl: np.ndarray,
        bndu: np.ndarray,
        ilower: np.ndarray | None = None,
        iupper: np.ndarray | None = None,
    ):
        """Set the bounds on the optimisation parameters.

        Parameters
        ----------
        bndl : np.ndarray
            lower bounds for the optimisation parameters
        bndu : np.ndarray
            upper bounds for the optimisation parameters
        ilower : np.ndarray, optional
            array of 0s and 1s to activate lower bounds on
            optimsation parameters in x
        iupper : np.ndarray, optional
            array of 0s and 1s to activate upper bounds on
            optimsation parameters in x
        """
        self.bndl = bndl
        self.bndu = bndu

        # TODO Remove ilower/iupper and use finite vs. infinite values in bndl/bndu
        # instead to determine defined vs. undefined bounds
        # If lower and upper bounds switch arrays aren't specified, set all bounds
        # to defined
        if ilower is None and iupper is None:
            ilower = np.ones(len(bndl), dtype=int)
            iupper = np.ones(len(bndu), dtype=int)

        self.ilower = ilower
        self.iupper = iupper

    def set_constraints(self, m: int, meq: int):
        """Set the total number of constraints and equality constraints.

        Parameters
        ----------
        m : int
            number of constraint equations
        meq : int
            of the constraint equations, how many are equalities
        """
        self.m = m
        self.meq = meq

    def set_tolerance(self, tolerance: float):
        """Set tolerance for solver termination.

        Parameters
        ----------
        tolerance : float
            tolerance for solver termination
        """
        self.tolerance = tolerance

    def set_b(self, b: float):
        """Set the multiplier for the Hessian approximation.

        Parameters
        ----------
        b : float
            multiplier for an identity matrix as input for the Hessian b(n,n)
        """
        self.b = b

    @abstractmethod
    def solve(self) -> int:
        """Run the optimisation.

        Returns
        -------
        int
            solver error code
        """


class VmconProblem(AbstractProblem):
    def __init__(self, evaluator, nequality, ninequality):
        self._evaluator = evaluator
        self._nequality = nequality
        self._ninequality = ninequality

    def __call__(self, x: np.ndarray) -> Result:
        n = x.shape[0]
        objf, conf = self._evaluator.fcnvmc1(n, self.total_constraints, x, 0)
        fgrd, cnorm = self._evaluator.fcnvmc2(n, self.total_constraints, x, n)

        return Result(
            objf,
            fgrd,
            conf[: self.num_equality],
            cnorm[:, : self.num_equality].T,
            conf[self.num_equality :],
            cnorm[:, self.num_equality :].T,
        )

    @property
    def num_equality(self) -> int:
        return self._nequality

    @property
    def num_inequality(self) -> int:
        return self._ninequality


class Vmcon(_Solver):
    """New VMCON implementation."""

    def solve(self) -> int:
        """Optimise using new VMCON.

        Returns
        -------
        int
            solver error code
        """
        problem = VmconProblem(self.evaluators, self.meq, self.m - self.meq)

        bb = None
        if self.b is not None:
            bb = np.identity(self.data.numerics.nvar) * self.b

        def _solver_callback(i: int, _result, _x, convergence_param: float):
            self.data.numerics.nviter = i + 1
            self.convergence_parameter = convergence_param
            print(
                f"{i + 1} | Convergence Parameter: {convergence_param:.3E}",
                end="\r",
                flush=True,
            )

        def _ineq_cons_satisfied(
            result: Result,
            _x: np.ndarray,
            _delta: np.ndarray,
            _lambda_eq: np.ndarray,
            _lambda_in: np.ndarray,
        ) -> bool:
            """Check that inequality constraints are satisfied.

            This additional convergence criterion ensures that solutions have
            satisfied inequality constraints.

            Parameters
            ----------
            result : Result
                evaluation of current optimisation parameter vector
            _x : np.ndarray
                current optimisation parameter vector
            _delta : np.ndarray
                search direction for line search
            _lambda_eq : np.ndarray
                equality Lagrange multipliers
            _lambda_in : np.ndarray
                inequality Lagrange multipliers

            Returns
            -------
            bool
                True if inequality constraints satisfied

            """
            # negative constraint value = violated
            # Check all ineqs are satisfied to within the tolerance
            # E.g. the relative violations are no more than v=0-tolerance
            return bool(
                np.all(result.ie >= -self.data.numerics.force_vmcon_inequality_tolerance)
            )

        try:
            x, _, _, res = solve(
                problem,
                np.array(self.x_0),
                np.array(self.bndl),
                np.array(self.bndu),
                max_iter=self.maxcal,
                epsilon=self.tolerance,
                qsp_options={"solver": cvxpy.CLARABEL},
                initial_B=bb,
                callback=_solver_callback,
                additional_convergence=_ineq_cons_satisfied
                if self.data.numerics.force_vmcon_inequality_satisfication
                else None,
            )
        except VMCONConvergenceException as e:
            if isinstance(e, LineSearchConvergenceException):
                self.info = 3
            elif isinstance(e, QSPSolverException):
                self.info = 5
            else:
                self.info = 2

            logger.critical(str(e))

            x = e.x
            res = e.result

        except ValueError:
            itervar_name_list = ""
            for count, iter_var in enumerate(
                self.data.numerics.ixc[: self.data.numerics.nvar]
            ):
                itervar_name = self.data.numerics.lablxc[iter_var - 1]
                itervar_name_list += f"{count}: {itervar_name} \n"

            logger.warning(f"Active iteration variables are : \n{itervar_name_list}")
            raise

        else:
            self.info = 1

        # print a blank line because of the carridge return
        # in the callback
        print()

        self.x = x
        self.objf = res.f
        self.conf = np.hstack((res.eq, res.ie))

        return self.info


class VmconBounded(Vmcon):
    """A solver that uses VMCON but checks x is in bounds before running"""

    def set_opt_params(self, x_0: np.ndarray):
        lower_violated = np.less(x_0, self.bndl)
        upper_violated = np.greater(x_0, self.bndu)

        for index, entry in enumerate(lower_violated):
            if entry:
                x_0[index] = self.bndl[index]
        for index, entry in enumerate(upper_violated):
            if entry:
                x_0[index] = self.bndu[index]
        self.x_0 = x_0


# TODO Have to define as functions (not methods on SolveIVP) as terminal event
# requires attribute setting on function: improve!
def detect_steady_state(
    t,
    y,
    self,
):
    # Terminate integration when d/dts below tolerance (crosses 0)
    tol = 1.0e-4
    dx_dt = derivatives(t, y, self)
    return np.sqrt(np.mean(dx_dt**2)) - tol


# Output paths for evaluations and iterations of solvers: debug only
IVP_EVALUATIONS_OUTPUT_PATH = "ivp_evaluations.csv"
IVP_ITERATIONS_OUTPUT_PATH = "ivp_iterations.csv"
RESIDUAL_OPT_EVALUATIONS_OUTPUT_PATH = "res_opt_evaluations.csv"
RESIDUAL_OPT_EVALUATIONS_WITH_OBJECTIVE_OUTPUT_PATH = "res_opt_obj_evaluations.csv"
SLSQP_OUTPUT_PATH = "slsqp_evaluations.csv"


def max_derivatives():
    # Maximum possible Te and ne derivatives
    # Max te derivative
    ni = physics_variables.nd_plasma_ions_total_vol_avg
    ne = physics_variables.nd_plasma_electrons_vol_avg
    vol = physics_variables.vol_plasma
    numerics.dte_dt_max = (
        (2 / 3)
        * (1 / 1.602e-19)
        * ((numerics.ppb_loss_max * 1e6 * vol) / ((ni + ne) * vol))
    ) * 1e-3

    # Max ne derivative
    f_alpha = physics_variables.nd_plasma_alphas_vol_avg / ne
    zimp = calc_zimp()
    numerics.dne_dt_max = (numerics.fe_loss_max / vol) / (
        1 - physics_variables.f_nd_beam_electron - zimp - 2 * f_alpha
    )


def calc_zimp():
    zimp = 0.0
    for imp in range(irm.N_IMPURITIES):
        if irm.impurity_arr_z[imp] > 2:
            zimp += (
                impurity_radiation.zav_of_te(
                    imp,
                    np.array([physics_variables.temp_plasma_electron_vol_avg_kev]),
                ).squeeze()
                * (irm.f_nd_impurity_electron_array[imp])
            )
    return zimp


def derivatives(t, y, self, optimiser=False):
    # Evaluate plasma power balance and fuel equilibrium
    # y is normalised values: fcnvmc1() scales up to real values
    # print(f"Normalised values = {y}")
    try:
        # Model exception may be raised
        _, self.conf = self.evaluators.fcnvmc1(y.shape[0], self.m, y, 0)
    finally:
        # Writes debug df even if models throw exception

        # Need absolute constraint residuals (real values)
        ppb = constraints.constraint_equation_2().constraint_error
        fe = constraints.constraint_equation_93().constraint_error

        ni = self.data.physics.nd_plasma_ions_total_vol_avg
        ne = self.data.physics.nd_plasma_electrons_vol_avg
        te = self.data.physics.temp_plasma_electron_vol_avg_kev
        vol = self.data.physics.vol_plasma
        # dTe/dt keV s^-1
        dte_dt = (
            (2 / 3) * (1 / 1.602e-19) * ((ppb * 1e6 * vol) / ((ni + ne) * vol))
        ) * 1e-3

        zimp = calc_zimp()
        f_alpha = self.data.physics.nd_plasma_alphas_vol_avg / ne
        # dne/dt m^-3 s^-1
        dne_dt = (fe / vol) / (
            1 - self.data.physics.f_nd_beam_electron - zimp - 2 * f_alpha
        )

        # Get real value of t
        if t is None:
            # Residual opt has no time step
            t_unnorm = None
        else:
            # IVP has time step
            t_unnorm = t * self.t0

        if DEBUG_DATAFRAME_OUTPUT:
            # Debugging df including derivatives
            data = {
                "t": [t_unnorm],
                "te": [te],
                "ne": [ne],
                "dte_dt": [dte_dt],
                "dne_dt": [dne_dt],
                "psep": [physics_variables.p_plasma_loss_mw],
                "tau_E": [physics_variables.t_energy_confinement],
            }
            # Write IVP and residual optimiser evaluations to 2 different output files
            # Only write a header when the file is first created
            if optimiser:
                pd.DataFrame(data).to_csv(
                    RESIDUAL_OPT_EVALUATIONS_OUTPUT_PATH,
                    mode="a",
                    header=not os.path.exists(RESIDUAL_OPT_EVALUATIONS_OUTPUT_PATH),
                    index=False,
                    float_format="%.9e",
                )
            else:
                pd.DataFrame(data).to_csv(
                    IVP_EVALUATIONS_OUTPUT_PATH,
                    mode="a",
                    header=not os.path.exists(IVP_EVALUATIONS_OUTPUT_PATH),
                    index=False,
                    float_format="%.9e",
                )

    # If exception thrown (usually from models), will be re-raised here after finally statement
    # Otherwise, return derivatives
    # Scale derivatives back down to normalised (nondimensionalised) values
    # TODO Iteration vars (te, ne) need to be in right order (same as scaling)!
    # TODO Sort out scaling array
    # TODO Check role of t0 here
    return np.array([dte_dt, dne_dt]) * self.t0 * self.scaling[:2]


def residual(x, self):
    # x is normalised vector
    dx_dt_norm = derivatives(None, x, self, optimiser=True)

    # Return sum of squares of normalised derivatives
    # Unnormalise derivatives
    dx_dt = dx_dt_norm / (self.t0 * self.scaling[:2])
    # Normalise using max values (set from previous solution point)
    numerics.dx_dt_normed_max = dx_dt / numerics.dx_dt_norm_max
    res = np.sqrt(np.mean(numerics.dx_dt_normed_max**2))
    if DEBUG_DATAFRAME_OUTPUT:
        # Debug data
        data = {
            "te": [x[0] / self.scaling[0]],
            "ne": [x[1] / self.scaling[1]],
            "dte_dt": [numerics.dx_dt_normed_max[0]],
            "dne_dt": [numerics.dx_dt_normed_max[1]],
            "res": [res],
            "psep": [physics_variables.p_plasma_loss_mw],
            "tau_E": [physics_variables.t_energy_confinement],
        }
        pd.DataFrame(data).to_csv(
            RESIDUAL_OPT_EVALUATIONS_WITH_OBJECTIVE_OUTPUT_PATH,
            mode="a",
            header=not os.path.exists(
                RESIDUAL_OPT_EVALUATIONS_WITH_OBJECTIVE_OUTPUT_PATH
            ),
            index=False,
            float_format="%.9e",
        )
    return res


class SolveIVP(_Solver):
    # Nondimensionalisation
    t0 = 2.0e2

    def handle_converged_sol(self, sol_result):
        # TODO Sort :2 in scaling array and Te, ne ordering requirement
        y_real = sol_result.y / self.scaling[:2, np.newaxis]
        t_real = sol_result.t * self.t0
        # Final point
        x_sol = sol_result.y[:, -1]
        x_sol_real = y_real[:, -1]
        print(f"Equilibrium values = [{x_sol_real[0]}, {x_sol_real[1]}]")
        print(sol_result.message)

        if DEBUG_DATAFRAME_OUTPUT:
            # Debugging df of actual solution timesteps
            data = {"t": t_real[:], "te": y_real[0, :], "ne": y_real[1, :]}
            pd.DataFrame(data).to_csv(
                IVP_ITERATIONS_OUTPUT_PATH,
                mode="a",
                header=not os.path.exists(IVP_ITERATIONS_OUTPUT_PATH),
                index=False,
                float_format="%.9e",
            )
        # Evaluate equality and inequality constraints at equality-satisfying solution
        # (or at last iteration of x if solution not found)
        _, self.conf = self.evaluators.fcnvmc1(x_sol.shape[0], self.m, x_sol, 0)
        # Record converged status and solution vector
        self.info = 1
        self.x = x_sol

    def handle_residual_sol(self, result):
        # Residual minimised: calculate residual to return
        # result.x is normalised vector; get normalised derivatives
        dx_dt_norm = derivatives(None, result.x, self, optimiser=True)
        # Unnormalise derivatives
        dx_dt = dx_dt_norm / (self.t0 * self.scaling[:2])
        # Normalise instead using max values (set from previous solution point)
        numerics.dx_dt_normed_max = dx_dt / numerics.dx_dt_norm_max
        # RMSE
        res = np.sqrt(np.mean(numerics.dx_dt_normed_max**2))
        # Record solution vector
        self.x = result.x
        print(colored("IVP failed, but residual found!", "green"))
        print(f"{dx_dt = }")
        print(f"{numerics.dx_dt_norm_max = }")
        print(f"{numerics.dx_dt_normed_max = }")
        print(f"Residual = {res:.3e}")
        # Set results that will be output in file
        numerics.derivative_rmse = res
        numerics.derivatives = dx_dt

    def solve(self) -> int:
        try:
            Path(RESIDUAL_OPT_EVALUATIONS_OUTPUT_PATH).unlink(missing_ok=True)
            Path(RESIDUAL_OPT_EVALUATIONS_WITH_OBJECTIVE_OUTPUT_PATH).unlink(
                missing_ok=True
            )
            Path(IVP_EVALUATIONS_OUTPUT_PATH).unlink(missing_ok=True)
            Path(IVP_ITERATIONS_OUTPUT_PATH).unlink(missing_ok=True)
            Path(SLSQP_OUTPUT_PATH).unlink(missing_ok=True)
        except:
            pass

        initial_values = self.x_0
        time_span = np.array([0.0, 2.0e3]) / self.t0
        # TODO Have to set attribute on function (scipy)
        detect_steady_state.terminal = True
        self.scaling = np.array(numerics.scale)

        # Radau required due to "stiffness": very different timescales of dte/dt and dne/dt
        sol_result = None
        try:
            sol_result = solve_ivp(
                fun=derivatives,
                args=(self,),
                t_span=time_span,
                y0=initial_values,
                method="Radau",
                events=detect_steady_state,
            )
            print(
                colored(
                    f"IVP: no exceptions. solve_ivp error code {sol_result.status}: {sol_result.message}",
                    "green",
                )
            )
        except ValueError as e:
            # Model error, probably caused by diverging parameter vector
            print(colored(f"IVP exception. Model error: {e}", "red"))

        if sol_result and sol_result.status == 1:
            # IVP converged: termination event
            self.handle_converged_sol(sol_result)
        else:
            # IVP didn't converge: failed to find equilibrium solution
            # Actually may have converged, but didn't trigger termination criterion (didn't cross 0)
            if (
                sol_result
                and sol_result.status == 0
                and detect_steady_state(None, sol_result.y[:, -1], self) < 0
            ):
                # 0: Reached end of time interval without converging, but
                # has actually converged
                self.handle_converged_sol(sol_result)
            else:
                # -1: Solver error
                # Or model exception, probably caused by diverging solution
                # Instead, run optimiser to find minimum residual of derivatives

                # Reset state: try flushing out model errors from likely previous bad IVP
                # state: this works, otherwise get negative znfuel again immediately
                # TODO Clearly needs improvement/justification
                physics_variables.fusden_alpha_total = 0.0
                physics_variables.nd_plasma_alphas_vol_avg = 0.0
                for _ in range(5):
                    try:
                        _, _ = self.evaluators.fcnvmc1(
                            initial_values.shape[0], self.m, initial_values, 0
                        )
                        break
                    except ValueError:
                        pass

                # Model exceptions here are now not caught
                # Residual optimisation will raise exception on model exception or
                # optimiser failure
                # Set derivative normalisation
                numerics.dx_dt_norm_max = np.array([
                    numerics.dte_dt_max,
                    numerics.dne_dt_max,
                ])

                result = minimize(
                    fun=residual,
                    x0=initial_values,
                    # TODO This is clearly not a great way of bounding the minimisation
                    # Bounds required to avoid huge steps and subsequent model errors
                    bounds=((0.5, 1.5), (0.5, 1.5)),
                    args=(self,),
                    jac="3-point",
                )
                if result.success:
                    self.handle_residual_sol(result)
                else:
                    # No model exception, but residual optimiser failed
                    raise Exception(
                        colored(f"Residual optimiser failed: {result.message}", "red")
                    )
                    # TODO Possible other self.info value?

                # IVP didn't converge, but residual optmiser did
                # TODO Check/change this return code: need to consider all solution modes
                self.info = -1

        # No objective function for IVP
        self.objf = None
        return self.info


class FSolve(_Solver):
    """Solve equality constraints to ensure model consistency."""

    global fsolve_con_eval_count
    fsolve_con_eval_count = 0
    FSOLVE_ITERATIONS_PATH = "iterations.csv"

    def evaluate_eq_cons(self, x: np.ndarray) -> np.ndarray:
        """Evaluate equality constraints.

        Parameters
        ----------
        x : np.ndarray
            parameter vector

        Returns
        -------
        np.ndarray
            equality constraint vector
        """
        global fsolve_con_eval_count
        fsolve_con_eval_count += 1
        print(f"{fsolve_con_eval_count = }")
        print(f"fsolve sol vec {x = }")

        # Write iteration parameter vector to CSV
        # Scale opt params up to real values before writing values
        set_scaled_iteration_variable(x, len(x))
        if DEBUG_DATAFRAME_OUTPUT:
            data = {
                "ne": [self.data.physics.nd_plasma_electrons_vol_avg],
                "te": [self.data.physics.temp_plasma_electron_vol_avg_kev],
            }
            # Only write a header when the file is first created
            pd.DataFrame(data).to_csv(
                self.FSOLVE_ITERATIONS_PATH,
                mode="a",
                header=not os.path.exists(self.FSOLVE_ITERATIONS_PATH),
                index=False,
            )

        # Evaluate equality constraints only
        # Calling fcnvmc1 scales the normalised x back up to absolute optimisation
        # parameter values and sets the required variables
        _, conf = self.evaluators.fcnvmc1(x.shape[0], self.meq, x, 0)

        # Required for including 2 iter vars in output, but only solving for the first one!
        # return conf[0]
        return conf

    def solve(self) -> int:
        """Solve equality constraints.

        Returns
        -------
        int
            solver error code
        """
        try:
            Path(self.FSOLVE_ITERATIONS_PATH).unlink()
        except:
            pass

        print("Solving equality constraints using fsolve")
        self.x, _info, err, msg = fsolve(
            self.evaluate_eq_cons, self.x_0, full_output=True, factor=0.1
        )

        # Evaluate equality and inequality constraints at equality-satisfying solution
        # (or at last iteration of x if solution not found)
        _, self.conf = self.evaluators.fcnvmc1(self.x.shape[0], self.m, self.x, 0)

        # err == 1 for successful solve
        if err != 1:
            print(f"fsolve error code {err}: {msg}")
            raise Exception(f"fsolve failed: {msg}")
        self.info = err
        # No objective function
        self.objf = None
        return self.info


class SLSQP(_Solver):
    def obj_func(self, x, grad):
        # Must be passed these args from nlopt requirements
        objf, conf = self.evaluators.fcnvmc1(self.n, self.m, x, self.ifail)
        self.constr_res = np.sqrt(
            np.sum(np.square(conf[: self.meq]))
        )  # only for equality constraints

        if grad.size > 0:
            # Gradient required by solver; modify grad in-place
            fgrd, cnorm = self.evaluators.fcnvmc2(self.n, self.m, x, self.lcnorm)
            grad[...] = fgrd

        self.eval_count += 1
        status = (
            f"Evaluation {self.eval_count}, objective function = {objf:.5}, "
            f"constraint residuals = {self.constr_res:.3e}"
        )
        print(status)
        logger.info(status)
        return objf

    def constraint_eq_vec(self, result, x, grad):
        # i is no. of constraint
        objf, conf = self.evaluators.fcnvmc1(self.n, self.m, x, self.ifail)
        self.constr_res = np.sqrt(
            np.sum(np.square(conf[: self.meq]))
        )  # only for equality constraints

        # Check if constraints are below tolerance yet
        # conf_gt_tol = np.sum((np.abs(conf[:meq]) > CONSTR_TOL))
        # if conf_gt_tol == 0:
        #     logger.info("Constraints are all below tolerance")
        # else:
        #     logger.info(f"Constraints above tolerance: {conf_gt_tol=}")

        if grad.size > 0:
            # Gradient required by solver; modify grad in-place
            fgrd, cnorm = self.evaluators.fcnvmc2(self.n, self.m, x, self.lcnorm)
            if cnorm.ndim == 1:
                # 1 constraint, 1 optimisation parameter (used in tests)
                # TODO Add np.newaxis to fcnvmc2 to avoid this
                grad[...] = -cnorm[:]
            else:
                grad[...] = np.transpose(-cnorm[: x.shape[0], : self.meq])

        # Negative conf and cnorm: nlopt requires opposite formulation of
        # VMCON's (and Process's) constraint form
        result[...] = -conf[: self.meq]

    def constraint_ineq_vec(self, result, x, grad):
        objf, conf = self.evaluators.fcnvmc1(self.n, self.m, x, self.ifail)
        if grad.size > 0:
            # Gradient required by solver; modify grad in-place
            fgrd, cnorm = self.evaluators.fcnvmc2(self.n, self.m, x, self.lcnorm)
            if cnorm.ndim == 1:
                # 1 constraint, 1 optimisation parameter (used in tests)
                # TODO Add np.newaxis to fcnvmc2 to avoid this
                grad[...] = -cnorm[0]
            else:
                grad[...] = np.transpose(-cnorm[: x.shape[0], self.meq : self.m])

        # Negative conf and cnorm: nlopt requires opposite formulation of
        # VMCON's (and Process's) constraint form
        result[...] = -conf[self.meq : self.m]

    def solve(self) -> int:
        """Try running nlopt."""
        print("Using SLSQP")
        self.eval_count = 0
        self.constr_res = 0.0

        self.n = self.x_0.shape[0]
        self.lcnorm = 176  # (ippnvars + 1), but could just be n!

        # Solver tolerances (high)
        MAIN_TOL = 1e-8
        # LOCAL_TOL = 1e-8 # for AUGLAG method only
        CONSTR_TOL = 1e-10

        # Set up optimiser object
        # opt = nlopt.opt(nlopt.AUGLAG, n)
        opt = nlopt.opt(nlopt.LD_SLSQP, self.n)

        # Set subsidiary optimiser
        # For AUGLAG only
        # local_opt = nlopt.opt(nlopt.LD_SLSQP, n)

        opt_name = opt.get_algorithm_name()
        # local_opt_name = local_opt.get_algorithm_name()
        # logger.info(f"{opt_name=}, {local_opt_name=}")
        logger.info(f"{opt_name=}")
        logger.info(f"{MAIN_TOL=:.3e}, {CONSTR_TOL=:.3e}")
        # logger.info(f"{MAIN_TOL=:.3e}, {LOCAL_TOL=:.3e}, {CONSTR_TOL=:.3e}")

        # Define tolerances
        opt.set_ftol_rel(MAIN_TOL)
        # local_opt.set_ftol_rel(LOCAL_TOL)

        # Need to terminate in solver test case 3!
        opt.set_maxeval(1000)

        # opt.set_local_optimizer(local_opt)

        # if self.maximise:
        #     # Maximisation required for test case 5
        #     opt.set_max_objective(self.obj_func)
        # else:
        opt.set_min_objective(self.obj_func)

        # Check bounds are activated for all optimisation parameters (default case)
        # If not, handle it
        if not (np.all(self.ilower) and np.all(self.iupper)):
            # Some bounds are inactive: used e.g. in solver integration tests
            for i in range(self.ilower.shape[0]):
                if self.ilower[i] == 0:
                    # Inactive lower bound: set to -inf
                    self.bndl[i] = -np.inf

            for i in range(self.iupper.shape[0]):
                if self.iupper[i] == 0:
                    # Inactive upper bound: set to +inf
                    self.bndu[i] = np.inf

        opt.set_lower_bounds(self.bndl)
        opt.set_upper_bounds(self.bndu)

        # Kludge initial normalised x into the normalised bounds range if required
        for i in range(self.x_0.shape[0]):
            if self.x_0[i] < self.bndl[i]:
                self.x_0[i] = self.bndl[i]
            elif self.x_0[i] > self.bndu[i]:
                self.x_0[i] = self.bndu[i]

        # Constraints
        if self.meq > 0:
            eq_constr_tols = np.full(self.meq, CONSTR_TOL)
            opt.add_equality_mconstraint(self.constraint_eq_vec, eq_constr_tols)
        if self.meq < self.m:
            ineq_constr_tols = np.full((self.m - self.meq), CONSTR_TOL)
            opt.add_inequality_mconstraint(self.constraint_ineq_vec, ineq_constr_tols)

        start_time = time.time()
        x_opt = opt.optimize(self.x_0)
        end_time = time.time()
        duration = end_time - start_time

        opt_val = opt.last_optimum_value()
        return_value = opt.last_optimize_result()

        # Main opt iterations
        main_opt_evals = opt.get_numevals()

        if return_value < 0:
            raise RuntimeError("nlopt didn't converge")
        elif return_value == nlopt.MAXEVAL_REACHED:
            # For giving up in int test 3
            # TODO Reconcile MAXEVAL with above exception: int case 3 needs to pass
            info = 5
        elif return_value > 0:
            # TODO Might want to be aware of other return values
            info = 1

        print(f"{return_value=}")

        # Recalculate conf at optimum x
        objf, conf = self.evaluators.fcnvmc1(self.n, self.m, x_opt, self.ifail)
        self.constr_res = np.sqrt(
            np.sum(np.square(conf[: self.meq]))
        )  # only for equality constraints

        logger.info(f"Main opt evaluations = {main_opt_evals}")
        logger.info(f"{opt_val=:.3e}, {self.constr_res=:.3e}")
        logger.info(f"{duration=:.1f}\n")
        logger.info(f"{conf[:self.meq]=}")

        print(f"{self.constr_res=:.3e}")

        # Check how many conf elements are above tolerance
        conf_gt_tol = np.sum((np.abs(conf[: self.meq]) > CONSTR_TOL))
        logger.info(f"Constraints above tolerance: {conf_gt_tol}")

        # Store required results on object
        self.objf = objf
        self.conf = conf
        self.x = x_opt

        return info


class Scipy_SLSQP(_Solver):
    """Minimise using scipy's SLSQP."""

    print("Running scipy's SLSQP")
    SOLVER_TOL = 1e-5
    EQ_CONSTRAINT_TOL = 1e-6

    def obj_func(self, x):
        objf, conf = self.evaluators.fcnvmc1(self.n, self.m, x, self.ifail)
        # constr_res = np.sqrt(
        #     np.sum(np.square(conf[:meq]))
        # )  # only for equality constraints
        # logger.debug(f"Constraint residuals: {constr_res:.3e}")

        # print(
        #     f"Evaluation {eval_count}, objective function = {objf:.5}, "
        #     f"constraint residuals = {constr_res:.3e}"
        # )
        return objf

    def constraint_eq_vec(self, x):
        objf, conf = self.evaluators.fcnvmc1(self.n, self.m, x, self.ifail)
        # constr_res = np.sqrt(
        #     np.sum(np.square(conf[:meq]))
        # )  # only to equality constraints

        return conf[: self.meq]

    def constraint_ineq_vec(self, x):
        objf, conf = self.evaluators.fcnvmc1(self.n, self.m, x, self.ifail)
        conf_gt_tol = np.sum((conf[self.meq :] < 0.0))
        logger.info(f"{conf_gt_tol} inequality constraints above 0.0")
        return conf[self.meq : self.m]

    def convergence_progress(self, x_current):
        eqs = self.constraint_eq_vec(x_current)
        ineqs = self.constraint_ineq_vec(x_current)
        cons = np.concatenate((eqs, ineqs))
        print("\nIteration results:")
        ineqs_rms = np.sqrt(np.mean(np.square(ineqs[ineqs < 0.0])))
        print(f"{ineqs_rms = :.3e}")

        # Print constraints sorted by value (most negative (most violated) first)
        sorted_eq_con_indexes = eqs.argsort()
        print("Equality constraints:")
        for i in sorted_eq_con_indexes:
            # Equality constraints first in icc
            print(f"Constraint {numerics.icc[i]} = {eqs[i]:.3e}")

        sorted_ineq_con_indexes = ineqs.argsort()
        print("Violated inequality constraints:")
        for i in sorted_ineq_con_indexes:
            if ineqs[i] < 0.0:
                print(f"Constraint {numerics.icc[len(eqs) + i]} = {ineqs[i]:.3e}")

        if DEBUG_DATAFRAME_OUTPUT:
            # Debugging df including derivatives
            iteration_variables.set_scaled_iteration_variable(x_current, len(x_current))
            data = {
                "te": [physics_variables.temp_plasma_electron_vol_avg_kev],
                "ne": [physics_variables.nd_plasma_electrons_vol_avg],
            }
            # If first run, prepend with initial point
            if not os.path.exists(SLSQP_OUTPUT_PATH):
                # Scale to real values, then back again
                iteration_variables.set_scaled_iteration_variable(
                    self.x_0, len(self.x_0)
                )
                data["te"].insert(0, physics_variables.temp_plasma_electron_vol_avg_kev)
                data["ne"].insert(0, physics_variables.nd_plasma_electrons_vol_avg)

            # Write IVP and residual optimiser evaluations to 2 different output files
            # Only write a header when the file is first created
            pd.DataFrame(data).to_csv(
                SLSQP_OUTPUT_PATH,
                mode="a",
                header=not os.path.exists(SLSQP_OUTPUT_PATH),
                index=False,
                float_format="%.9e",
            )

    def solve(self):
        self.n = self.x_0.shape[0]

        # Check bounds are activated for all optimisation parameters (default case)
        # If not, handle it
        if not (np.all(self.ilower) and np.all(self.iupper)):
            # Some bounds are inactive: used e.g. in solver integration tests
            for i in range(self.ilower.shape[0]):
                if self.ilower[i] == 0:
                    # Inactive lower bound: set to -inf
                    self.bndl[i] = -np.inf

            for i in range(self.iupper.shape[0]):
                if self.iupper[i] == 0:
                    # Inactive upper bound: set to +inf
                    self.bndu[i] = np.inf

        # Kludge initial normalised x into the normalised bounds range if required
        for i in range(self.n):
            if self.x_0[i] < self.bndl[i]:
                self.x_0[i] = self.bndl[i]
            elif self.x_0[i] > self.bndu[i]:
                self.x_0[i] = self.bndu[i]

        bounds = optimize.Bounds(lb=self.bndl, ub=self.bndu)

        constraints = []
        if self.meq > 0:
            eq_constraints = optimize.NonlinearConstraint(
                self.constraint_eq_vec, -self.EQ_CONSTRAINT_TOL, self.EQ_CONSTRAINT_TOL
            )
            constraints.append(eq_constraints)

        if self.meq < self.m:
            ineq_constraints = optimize.NonlinearConstraint(
                self.constraint_ineq_vec,
                0.0,
                np.inf,
            )
            constraints.append(ineq_constraints)

        start_time = time.time()

        result = optimize.minimize(
            self.obj_func,
            self.x_0,
            method="SLSQP",
            jac=None,
            bounds=bounds,
            constraints=constraints,
            tol=self.SOLVER_TOL,
            callback=self.convergence_progress,
            options={"disp": True, "eps": numerics.epsfcn, "maxiter": 20},
        )
        end_time = time.time()
        duration = end_time - start_time

        # Log stuff
        logger.info(f"{self.SOLVER_TOL=}, {self.EQ_CONSTRAINT_TOL=}")
        logger.info(f"Iterations: {result.nit}")
        logger.info(f"Evaluations: {result.nfev}")
        logger.info(f"Duration: {duration:.3}")

        # Recalculate constraints at optimium x
        objf, conf = self.evaluators.fcnvmc1(self.n, self.m, result.x, self.ifail)

        # Check if constraints are all below tolerance
        conf_gt_tol = np.sum((conf[self.meq :] < 0.0))
        logger.info(f"{conf_gt_tol} inequality constraints violated")
        logger.info(f"Constraint residuals: {conf}")

        # constr_res = np.sqrt(
        #     np.sum(np.square(conf[: self.meq]))
        # )  # only for equality constraints
        # logger.info(f"Constraint residuals: {constr_res:.3e}")
        # logger.info(f"{conf=}")

        if result.success:
            info = 1
            # TODO Max derivatives need to be calculated for all solvers: sort out
            max_derivatives()
        else:
            # Want to write error code to MFILE
            # raise RuntimeError("scipy failed to converge")
            info = 2

        self.objf = result.fun
        self.conf = conf
        self.x = result.x

        return info


def get_solver(data: DataStructure, solver_name: str = "vmcon") -> _Solver:
    """Return a solver instance.

    Parameters
    ----------
    solver_name : str, optional
        solver to create, defaults to "vmcon"

    Returns
    -------
    _Solver
        solver to use for optimisation
    """
    solver: _Solver

    if solver_name == "vmcon":
        solver = Vmcon(data=data)
    elif solver_name == "vmcon_bounded":
        solver = VmconBounded(data=data)
    elif solver_name == "fsolve":
        solver = FSolve(data=data)
    elif solver_name == "slsqp":
        solver = SLSQP(data=data)
    elif solver_name == "scipy_slsqp":
        solver = Scipy_SLSQP(data=data)
    elif solver_name == "solve_ivp":
        solver = SolveIVP(data=data)
    else:
        try:
            solver = load_external_solver(solver_name)
        except Exception as e:
            raise ProcessValueError(
                "Solver name is not an inbuilt PROCESS solver or recognised package "
                f'"{solver_name}"'
            ) from e

    return solver


def load_external_solver(package: str):
    """Attempts to load a package of name `package`.

    If a package of the name is available, return the `__process_solver__`
    attribute of that package or raise an `AttributeError`.

    Parameters
    ----------
    package: str :

    """
    module = importlib.import_module(package)

    solver = getattr(module, "__process_solver__", None)

    if solver is None:
        raise AttributeError(
            f"Module {module.__name__} does not have a '__process_solver__' attribute."
        )

    return solver()
