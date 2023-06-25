//
// Created by lurvelly on 25.6.2023.
//

/*
 * To get started, consider the problem of finding the minimum of the function 1/2(10-x)^2
 * This is a trivial problem, whose minimum is located at x = 10, but it is a good place to start to
 * illustrate the basics of solving a problem with the Ceres Solver.
 * */

#include <iostream>
#include <ceres/ceres.h>

/*
 * In some cases, its not possible to define a templated cost functor, for example when the evaluation
 * of the residual involves a call to a library function that you do not have control over. In such a
 * situation, numeric derivatives can be used. The user defines a functor which computes the
 * residual value and construct a NumbercDiffCostFunction using it. e.g., for f(x) = 10 - x the
 * corresponding functor would be:
 * */

class NumericDiffCostFunctor {
public:
    bool operator()(const double* const x, double* residual) const {
        residual[0] = 10.0 - x[0];
        return true;
    }
};

/*
 * Once we have a way of computing the residual function, it is now time to construct a non-linear
 * least squares problem using it and have Ceres solve it.
 * */

int main(int argc, char** argv) {
    // The variable to solve for with its initial value.
    double initial_x = 5.0;
    double x = initial_x;

    // Build the problem
    ceres::Problem problem;

    // Set up the only cost function (also known as residual). This uses
    // numeric differentiation to obtain the derivative (jacobian).
    ceres::CostFunction* cost_function =
            new ceres::NumericDiffCostFunction<NumericDiffCostFunctor, ceres::CENTRAL, 1, 1>(
                    new NumericDiffCostFunctor);
    problem.AddResidualBlock(cost_function, nullptr, &x);

    // Run the solver
    ceres::Solver::Options options;
    options.linear_solver_type = ceres::DENSE_QR;
    options.minimizer_progress_to_stdout = true;

    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);
    std::cout << summary.BriefReport() << std::endl;
    std::cout << "x: " << initial_x << " -> " << x << std::endl;
    return 0;
}