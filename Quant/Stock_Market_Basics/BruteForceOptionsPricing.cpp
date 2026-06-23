// cppimport
#include <pybind11/pybind11.h>

#include <iostream>
#include <cmath>
#include <string>
#include <algorithm>
#include <vector>
#include <numbers>

namespace py = pybind11;
using namespace std;

const double PI = 3.1415926;


//to simulate the normal cdf we can use this function for it:
//phi (z) = 1/2[1 + erf(x / sqrt(2))] as erf as the error function or gauass normal funtion
class BruteForceOptionPricing{
private:
    double implied_volatility;
    double V; //market price
    double E; // strike price
    double T; // expiry this is actually T - t
    double S; // asset
    double r; // interest rate (risk-neutral)
    double error = 1e-6f; //error value for later Volatility work or NewtonRaphson implementation here
    double volatility;
    double dividents = 0; //this usually just be 0 though so no worry about this

    double get_d1() {
        double d1 = log(S / E) + (r - dividents + 0.5 * volatility * volatility) * T;
        d1 /= volatility * sqrt(T);
        return d1;
    }

    double get_d2() {
        double d1 = get_d1();
        return d1 - volatility * sqrt(T);
    }

    double normal_cdf(double x) { //distribution over N(0,1)
        double z = (x - 0) / 1;
        return 0.5 * (1.0 + erf(z / sqrt(2)));
    }

    double normal_pdf(double x) {
        double z = (x - 0) / 1;
        return (1.0 / sqrt(2 * PI)) * exp(-0.5 * z * z);
    }

public:
    BruteForceOptionPricing(double V, double E, double T, double S, double r) : 
        V(V), E(E), T(T), S(S), r(r) {}
        
    BruteForceOptionPricing(double V, double E, double T, double S, double r, double error) : 
            V(V), E(E), T(T), S(S), r(r), error(error) {}

    BruteForceOptionPricing(double V, double E, double T, double S, double r, 
        double error, double volatility, double dividents) : 
        V(V), E(E), T(T), S(S), r(r), error(error), volatility(volatility), dividents(dividents) {} 

    BruteForceOptionPricing(double E, double T, double S, double r) : 
        E(E), T(T), S(S), r(r) {}
        
    BruteForceOptionPricing(double E, double T, double S, double r, double error) : 
        E(E), T(T), S(S), r(r), error(error) {}

    BruteForceOptionPricing(double E, double T, double S, double r, 
        double error, double volatility, double dividents) : 
        E(E), T(T), S(S), r(r), error(error), volatility(volatility), dividents(dividents) {} 

    double OptionValue(string OptionType) {
        if (OptionType == "Call") {
            return S * exp(-dividents* T) * normal_cdf(get_d1()) - E * exp(-r * T) * normal_cdf(get_d2());
        }

        if (OptionType == "Put") {
            return - S * exp(-dividents* T) * normal_cdf(-1 * get_d1()) +  E * exp(-r * T) * normal_cdf(-1 * get_d2());
        }

        if (OptionType == "Binary Call") {
            return exp(-r * T) * normal_cdf(get_d2());
        }
        
        if (OptionType == "Binary Put") {
            return exp(-r * T) * normal_cdf(-1 * get_d2());
        }
    }

    double DeltaValue(string OptionType) {
        if (OptionType == "Call") {
            return exp(- dividents * T) * normal_cdf(get_d1());
        }

        if (OptionType == "Put") {
            return exp(- dividents * T) * normal_cdf(get_d1());
        }

        if (OptionType == "Binary Call") {
            return (exp(- r * T) * normal_pdf(get_d2())) / (volatility * volatility * S * S * T);
        }
        
        if (OptionType == "Binary Put") {
            return -1 * (exp(- r * T) * normal_pdf(get_d2())) / (volatility * volatility * S * S * T);
        }
    }

        double GammaValue(string OptionType) {
        if (OptionType == "Call" || OptionType == "Put") {
             return (exp(- dividents * T) * normal_pdf(get_d2())) / (volatility * volatility * S * S * T);
        }

        if (OptionType == "Binary Call") {
            return -1 * (exp(- r * T) * get_d1() *normal_pdf(get_d2())) / (volatility * volatility * S * S * T);
        }
        
        if (OptionType == "Binary Put") {
            return (exp(- r * T) * get_d1() *normal_pdf(get_d2())) / (volatility * volatility * S * S * T);
        }
    }

    

    //this will also incorperate on making ImpliedVolatility and Using yfinance data or data i got here to backtest
    //and do option pricing on this actually; also i'll do the MonteCarloOptionPricing version to compare too.
};

PYBIND11_MODULE(BruteForceOptionsPricing, m) {
    m.doc() = "Brute force option pricing wrapper";

    py::class_<BruteForceOptionPricing>(m, "BruteForceOptionPricing")
        .def(py::init<double, double, double, double, double>(),
             py::arg("V"), py::arg("E"), py::arg("T"), py::arg("S"), py::arg("r"))
        .def(py::init<double, double, double, double, double, double>(),
             py::arg("V"), py::arg("E"), py::arg("T"), py::arg("S"), py::arg("r"), py::arg("error"))
        .def(py::init<double, double, double, double, double, double, double, double>(),
             py::arg("V"), py::arg("E"), py::arg("T"), py::arg("S"), py::arg("r"),
             py::arg("error"), py::arg("volatility"), py::arg("dividents"))
        .def("OptionValue", &BruteForceOptionPricing::OptionValue)
        .def("DeltaValue", &BruteForceOptionPricing::DeltaValue)
        .def("GammaValue", &BruteForceOptionPricing::GammaValue);
}
