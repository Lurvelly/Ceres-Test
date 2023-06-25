//
// Created by lurvelly on 25.6.2023.
//
/*
 * An example program that minimizes Powell's singular function.
 *
 * F = 1/2 (f1^2 + f2^2 + f3^2 + f4^2)
 *
 * f1 = x1 + 10 * x2;
 * f2 = sqrt(5) * (x3 - x4);
 * f3 = (x2 - 2 * x3)^2;
 * f4 = sqrt(10) * (x1 - x4)^2;
 *
 * The starting values are x1 = 3, x2 = -1, x3 = 0, x4 = 1.
 * The minimum is 0 at (x1, x2, x3, x4) = 0.
 * */

#include <iostream>
#include <ceres/ceres.h>

// Number of residuals, or ceres::DYNAMIC.
const int kNumResiduals = 1;
// Number of parameters in each parameter block.
constexpr int Ns(int i) {
    int Ns[4]{1, 1, 1, 1};
    return Ns[i - 1];
}

// The first step is to write four functors

/*
 * For AutoDiffCostFunction
 */
class AutoF1 {
public:
    template<typename T>
    bool operator()(const T* const x1, const T* const x2, T* residual) const {
        // f1 = x1 + 10 * x2;
        residual[0] = x1[0] + 10.0 * x2[0];
        return true;
    }
};

class AutoF2 {
public:
    template<typename T>
    bool operator()(const T* const x3, const T* const x4, T* residual) const {
        // f2 = sqrt(5) * (x3 - x4);
        residual[0] = std::sqrt(5.0) * (x3[0] - x4[0]);
        return true;
    }
};

class AutoF3 {
public:
    template<typename T>
    bool operator()(const T* const x2, const T* const x3, T* residual) const {
        // f3 = (x2 - 2 * x3)^2;
        residual[0] = (x2[0] - 2.0 * x3[0]) * (x2[0] - 2.0 * x3[0]);
        return true;
    }
};

class AutoF4 {
public:
    template<typename T>
    bool operator()(const T* const x1, const T* const x4, T* residual) const {
        // f4 = sqrt(10) * (x1 - x4)^2;
        residual[0] = std::sqrt(10.0) * (x1[0] - x4[0]) * (x1[0] - x4[0]);
        return true;
    }
};

/*
 * For NumericDiffCostFunction
 */
class NumericF1 {
public:
    bool operator()(const double* const x1, const double* const x2, double* residual) const {
        // f1 = x1 + 10 * x2;
        residual[0] = x1[0] + 10.0 * x2[0];
        return true;
    }
};

class NumericF2 {
public:
    bool operator()(const double* const x3, const double* const x4, double* residual) const {
        // f2 = sqrt(5) * (x3 - x4);
        residual[0] = std::sqrt(5.0) * (x3[0] - x4[0]);
        return true;
    }
};

class NumericF3 {
public:
    bool operator()(const double* const x2, const double* const x3, double* residual) const {
        // f3 = (x2 - 2 * x3)^2;
        residual[0] = (x2[0] - 2.0 * x3[0]) * (x2[0] - 2.0 * x3[0]);
        return true;
    }
};

class NumericF4 {
public:
    bool operator()(const double* const x1, const double* const x4, double* residual) const {
        // f4 = sqrt(10) * (x1 - x4)^2;
        residual[0] = std::sqrt(10.0) * (x1[0] - x4[0]) * (x1[0] - x4[0]);
        return true;
    }
};

/*
 * For AnalyticDiffCostFunction
 */
class AnalyticF1 : public ceres::SizedCostFunction<kNumResiduals, 4> {
public:
    ~AnalyticF1() override = default;
    virtual bool Evaluate(double const* const* parameters,
                          double* residuals,
                          double** jacobians) const {
        const double x1 = parameters[0][0];
        const double x2 = parameters[0][1];

        // f1 = x1 + 10 * x2;
        residuals[0] = x1 + 10 * x2;

        if (jacobians != nullptr && jacobians[0] != nullptr) {
            jacobians[0][0] = 1.0;
            jacobians[0][1] = 10.0;
            jacobians[0][2] = 0;
            jacobians[0][3] = 0;
        }
        return true;
    }
};

class AnalyticF2 : public ceres::SizedCostFunction<kNumResiduals, 4> {
public:
    ~AnalyticF2() override = default;
    virtual bool Evaluate(double const* const* parameters,
                          double* residuals,
                          double** jacobians) const {
        const double x3 = parameters[0][2];
        const double x4 = parameters[0][3];

        // f2 = sqrt(5) * (x3 - x4);
        residuals[0] = std::sqrt(5.0) * (x3 - x4);

        if (jacobians != nullptr && jacobians[0] != nullptr) {
            jacobians[0][0] = 0;
            jacobians[0][1] = 0;
            jacobians[0][2] = std::sqrt(5.0);
            jacobians[0][3] = -std::sqrt(5.0);
        }
        return true;
    }
};

class AnalyticF3 : public ceres::SizedCostFunction<kNumResiduals, 4> {
public:
    ~AnalyticF3() override = default;
    virtual bool Evaluate(double const* const* parameters,
                          double* residuals,
                          double** jacobians) const {
        const double x2 = parameters[0][1];
        const double x3 = parameters[0][2];

        // f3 = (x2 - 2 * x3)^2;
        residuals[0] = (x2 - 2.0 * x3) * (x2 - 2.0 * x3);

        if (jacobians != nullptr && jacobians[0] != nullptr) {
            jacobians[0][0] = 0;
            jacobians[0][1] = 2 * (x2 - 2.0 * x3);
            jacobians[0][2] = -4 * (x2 - 2.0 * x3);
            jacobians[0][3] = 0;
        }
        return true;
    }
};

class AnalyticF4 : public ceres::SizedCostFunction<kNumResiduals, 4> {
public:
    ~AnalyticF4() override = default;
    virtual bool Evaluate(double const* const* parameters,
                          double* residuals,
                          double** jacobians) const {
        const double x1 = parameters[0][0];
        const double x4 = parameters[0][3];

        // f4 = sqrt(10) * (x1 - x4)^2;
        residuals[0] = std::sqrt(10.0) * (x1 - x4) * (x1 - x4);

        if (jacobians != nullptr && jacobians[0] != nullptr) {
            jacobians[0][0] = 2 * std::sqrt(10.0) * (x1 - x4);
            jacobians[0][1] = 0;
            jacobians[0][2] = 0;
            jacobians[0][3] = -2 * std::sqrt(10.0) * (x1 - x4);
        }
        return true;
    }
};



template<typename T>
void AutoDiffSolve(const T* const init_x1, const T* const init_x2, const T* const init_x3, const T* const init_x4, ceres::Solver::Options* options, ceres::Solver::Summary* summary) {

    double x1 = init_x1[0];
    double x2 = init_x2[0];
    double x3 = init_x3[0];
    double x4 = init_x4[0];


    ceres::Problem problem;
    // Add residual terms to the problem using the autodiff wrapper to get the derivatives automatically.
    problem.AddResidualBlock(new ceres::AutoDiffCostFunction<AutoF1, kNumResiduals, Ns(1), Ns(2)>(new AutoF1), nullptr, &x1, &x2);
    problem.AddResidualBlock(new ceres::AutoDiffCostFunction<AutoF2, kNumResiduals, Ns(3), Ns(4)>(new AutoF2), nullptr, &x3, &x4);
    problem.AddResidualBlock(new ceres::AutoDiffCostFunction<AutoF3, kNumResiduals, Ns(2), Ns(3)>(new AutoF3), nullptr, &x2, &x3);
    problem.AddResidualBlock(new ceres::AutoDiffCostFunction<AutoF4, kNumResiduals, Ns(1), Ns(4)>(new AutoF4), nullptr, &x1, &x4);

    ceres::Solve(*options, &problem, summary);

    std::cout << __FUNCTION__  << std::endl;
    std::cout << summary->total_time_in_seconds << std::endl;
    std::cout << "Final x1 = " << x1
        << ", x2 = " << x2
        << ", x3 = " << x3
        << ", x4 = " << x4 << std::endl;
}

void NumericDiffSolve(const double* const init_x1, const double* const init_x2,
                        const double* const init_x3, const double* const init_x4,
                        ceres::Solver::Options* options, ceres::Solver::Summary* summary) {
    double x1 = init_x1[0];
    double x2 = init_x2[0];
    double x3 = init_x3[0];
    double x4 = init_x4[0];

    ceres::Problem problem;
    // Add residual terms to the problem using the numeric diff wrapper to get the derivatives automatically.
    problem.AddResidualBlock(new ceres::NumericDiffCostFunction<NumericF1, ceres::CENTRAL, kNumResiduals, Ns(1), Ns(2)>(new NumericF1), nullptr, &x1, &x2);
    problem.AddResidualBlock(new ceres::NumericDiffCostFunction<NumericF2, ceres::CENTRAL, kNumResiduals, Ns(3), Ns(4)>(new NumericF2), nullptr, &x3, &x4);
    problem.AddResidualBlock(new ceres::NumericDiffCostFunction<NumericF3, ceres::CENTRAL, kNumResiduals, Ns(2), Ns(3)>(new NumericF3), nullptr, &x2, &x3);
    problem.AddResidualBlock(new ceres::NumericDiffCostFunction<NumericF4, ceres::CENTRAL, kNumResiduals, Ns(1), Ns(4)>(new NumericF4), nullptr, &x1, &x4);

    ceres::Solve(*options, &problem, summary);

    std::cout << __FUNCTION__  << std::endl;
    std::cout << summary->total_time_in_seconds << std::endl;
    std::cout << "Final x1 = " << x1
              << ", x2 = " << x2
              << ", x3 = " << x3
              << ", x4 = " << x4 << std::endl;
}

void AnalyticDiffSolve(const double* const init_x1, const double* const init_x2,
                       const double* const init_x3, const double* const init_x4,
                       ceres::Solver::Options* options, ceres::Solver::Summary* summary) {
    double x1 = init_x1[0];
    double x2 = init_x2[0];
    double x3 = init_x3[0];
    double x4 = init_x4[0];

    double x[] = {x1, x2, x3, x4};

    ceres::Problem problem;
    // Add residual terms to the problem using the analytic diff wrapper.
    problem.AddResidualBlock(new AnalyticF1, nullptr, x);
    problem.AddResidualBlock(new AnalyticF2, nullptr, x);
    problem.AddResidualBlock(new AnalyticF3, nullptr, x);
    problem.AddResidualBlock(new AnalyticF4, nullptr, x);

    ceres::Solve(*options, &problem, summary);

    std::cout << __FUNCTION__  << std::endl;
    std::cout << summary->total_time_in_seconds << std::endl;
    std::cout << "Final x1 = " << x[0]
              << ", x2 = " << x[1]
              << ", x3 = " << x[2]
              << ", x4 = " << x[3] << std::endl;
}











int main(int argc, char** argv) {

    // The starting values are x1 = 3, x2 = -1, x3 = 0, x4 = 1.
    double x1 = 3.0;
    double x2 = -1.0;
    double x3 = 0.0;
    double x4 = 1.0;

    //
    ceres::Solver::Options options;
    options.max_num_iterations = 100;
    options.linear_solver_type = ceres::DENSE_QR;
    options.minimizer_progress_to_stdout = true;

    ceres::Solver::Summary summary;
    AutoDiffSolve<double>(&x1, &x2, &x3, &x4, &options, &summary);
    NumericDiffSolve(&x1, &x2, &x3, &x4, &options, &summary);
    AnalyticDiffSolve(&x1, &x2, &x3, &x4, &options, &summary);

}





