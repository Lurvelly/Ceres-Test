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
 * In some cases, using automatic differentiation is not possible. For example, it may be the case that
 * it is more efficient to compute the derivatives in closed form instead of relying on the chain rule
 * used by the automatic differentiation code.
 *
 * In such cases, it is possible to supply your own residual and jacobian computation code. To do this,
 * define a subclass of CostFunction or SizedCostFunction if you know the sizes of the parameters
 * and residuals at compile time. Here for example is SimpleCostFunction that implements f(x) = 10 - x.
 * */
class QuadraticCostFunction : public ceres::SizedCostFunction<1, 1> {
public:
    virtual ~QuadraticCostFunction() {}
    /*
     * The Evaluate is provided with an input array of parameters, an output array
     * residuals for residuals and an output array jacobians for Jacobians. The jacobians array is
     * optional, Evaluate is expected to check when it is non-null, and if it is the case then fill it with the
     * values off the derivative of the residual function. In this case since the residual function is linear, the
     * Jacobian is constant.
     * */
    virtual bool Evaluate(double const* const* parameters,
                          double* residuals,
                          double** jacobians) const {
        const double x = parameters[0][0];

        // f(x) = 10 - x
        residuals[0] = 10 - x;

        /*
         * f'(x) = -1. Since there's only 1 parameter and that parameter has 1 dimension,
         * there is only 1 element to fill in the jacobian.
         *
         * Since the Evaluate function can be called with the jacobians pointer equal to nullptr,
         * the Evaluate function must check to see if jacobians need to be computed.
         *
         * For this simple problem it is overkill to check if jacobians[0] is nullptr,
         * but in general when writing more complex CostFunctions it is possible that
         * Ceres may only demand the derivatives w.r.t. a subset of the parameter blocks.
         * */

        if (jacobians != nullptr && jacobians[0] != nullptr) {
            jacobians[0][0] = -1;
        }
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
    ceres::CostFunction* cost_function = new QuadraticCostFunction;
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


