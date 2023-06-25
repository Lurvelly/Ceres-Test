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
 * The first step is to write a functor that will evaluate this function f(x) = 10 - x
 * The important thing to note here is that operator() is a templated method, which assumes that all
 * its inputs and outputs are of some type T. The use of templating here allows Ceres to call
 * CostFunctor::operator<T>(), with T = double when just the value of the residual is needed, and with
 * a special type T = Jet when the Jacobians are needed.
 * */

class CostFunction {
public:
    template<typename T>
    bool operator()(const T *const x, T *residual) const {
        residual[0] = static_cast<T>(10.0) - x[0];
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
    // auto-differentiation to obtain the derivative (jacobian).
    ceres::CostFunction *cost_function =
            new ceres::AutoDiffCostFunction<CostFunction, 1, 1>(new CostFunction);
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