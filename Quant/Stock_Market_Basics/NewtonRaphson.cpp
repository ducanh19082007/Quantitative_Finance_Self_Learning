#include <iostream>
#include <cmath>
#include <string>
#include <algorithm>
#include <vector>
#include <numbers>
#include <iomanip>

using namespace std;

const double PI = 3.1415926;

// g++ NewtonRaphson.cpp -o NewtonRaphson ./NewtonRaphson

//./NewtonRaphson


//to simulate the normal cdf we can use this function for it:
//phi (z) = 1/2[1 + erf(x / sqrt(2))] as erf as the error function or gauass normal funtion
class NewtonRaphson{
private:
    double implied_volatility;
    double V; //market price
    double E; // strike price
    double T; // expiry
    double S; // asset
    double r; // interest rate (risk-neutral)
    double error; //error value

    double normal_cdf(double x) { //distribution over N(0,1)
        double z = (x - 0) / 1;
        return 0.5 * (1.0 + erf(z / sqrt(2)));
    }
public:
    NewtonRaphson(double V, double E, double T, double S, double r, double error = 1e-6f) : 
                V(V), E(E), T(T), S(S), r(r), error(error) {} 

    void run() {
        double volatility = 1; //random number
        double dv = error + 1;

        while (abs(dv) > error) {
            double d1 = log(S / E) + (r + 0.5 * (volatility * volatility)) * T;
            d1 /= volatility * sqrt(T);
            double d2 = d1 - volatility * sqrt(T);
            
            double PriceErr = S * normal_cdf(d1) - E * exp(-r * T) * normal_cdf(d2) - V;
            double vega = S * sqrt(T / PI / 2) * exp(-0.5 * d1 * d1);

            dv = PriceErr / vega;
            volatility = volatility - dv;
        }

        implied_volatility = volatility;
        return;
    }

    double get_implied_volatility() {
        return implied_volatility;
    }

    void print_implied_volatility() {
        cout << "Calculated IV: " << get_implied_volatility() * 100 << "%\n\n";
    }
};


int main() {
    // Formatting output
    cout << fixed << setprecision(4);
    cout << "=========================================\n";
    cout << "   Implied Volatility Calculator (NR)    \n";
    cout << "=========================================\n\n";

    // --- TEST CASE 1: At-The-Money (ATM) Option ---
    // S = 100, E = 100, T = 1 year, r = 5%, Market Price = 10.45
    // Expected Volatility: ~20% (0.20)
    NewtonRaphson case1(10.4506, 100.0, 1.0, 100.0, 0.05);
    case1.run();
    cout << "Test Case 1 (At-The-Money):\n";
    case1.print_implied_volatility();

    // --- TEST CASE 2: High Volatility Environment ---
    // S = 100, E = 95, T = 0.5 years (6 months), r = 1%, Market Price = 18.20
    // Expected Volatility: ~50% (0.50)
    NewtonRaphson case2(18.2039, 95.0, 0.5, 100.0, 0.01);
    case2.run();
    cout << "Test Case 2 (In-The-Money & High Vol):\n";
    cout << "Calculated IV: " << case2.get_implied_volatility() * 100 << "%\n\n";

    // --- TEST CASE 3: Out-Of-The-Money (OTM) Short-Term Option ---
    // S = 250, E = 270, T = 0.25 years (3 months), r = 3%, Market Price = 4.15
    // Expected Volatility: ~30% (0.30)
    NewtonRaphson case3(4.1524, 270.0, 0.25, 250.0, 0.03);
    case3.run();
    cout << "Test Case 3 (Out-Of-The-Money):\n";
    cout << "Calculated IV: " << case3.get_implied_volatility() * 100 << "%\n";
    cout << "=========================================\n";

    return 0;
}

/*
Test Case 1 (At-The-Money):
Calculated IV: 20.0000%

Test Case 2 (In-The-Money & High Vol):
Calculated IV: 56.1874%

Test Case 3 (Out-Of-The-Money):
Calculated IV: 21.0317%
*/