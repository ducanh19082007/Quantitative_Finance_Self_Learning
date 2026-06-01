#include <iostream>
#include <cmath>
#include <string>
#include <algorithm>
#include <vector>

using namespace std;

float Payoff_call(float S, float K) {
    if (S > K) return S - K;
    return 0.0f;
}

void print_a(const vector<float>& arr) {
    cout << "[";
    int count = 0;
    for (float x : arr) {
        count++;
        cout << x << " ";
        if (count == 10) {
            cout << "...";
            break;
        };
    }
    cout << "]" << endl;
}

float option_price_Europe(float asset, float volatility, float InterestRate,
                          float Strike, float expiry, int NumSteps,
                          string option_type) {
    vector<float> S(NumSteps + 1, 0.0f);
    vector<float> V(NumSteps + 1, 0.0f);

    float time_step = expiry / NumSteps;
    float discount_factor = exp(-InterestRate * time_step);

    float temp = exp(-InterestRate * time_step)
               + exp((InterestRate + volatility * volatility) * time_step);

    float u = 0.5f * (temp + sqrt(temp * temp - 4.0f));
    float v = 1.0f / u;

    float p_ = (exp(InterestRate * time_step) - v) / (u - v);

    S[0] = asset;

    for (int n = 1; n <= NumSteps; n++) {
        for (int j = n; j > 0; j--) {
            S[j] = u * S[j - 1];
        }
        S[0] = v * S[0];
    }

    for (int j = 0; j <= NumSteps; j++) {
        V[j] = Payoff_call(S[j], Strike);
    }

    for (int n = NumSteps; n > 0; n--) {
        for (int j = 0; j < n; j++) {

            if (option_type == "Europe") {
                V[j] = discount_factor *
                       (p_ * V[j + 1] + (1.0f - p_) * V[j]);
            }
            else if (option_type == "American") {
                float temp1 = (V[j + 1] - V[j]) / (u - v);
                float temp2 = discount_factor * (u * V[j] - v * V[j + 1]) / (u - v);

                V[j] = max<float>(temp1 + temp2, Payoff_call(S[j], Strike));
            }
        }
    }

    print_a(S);

    return V[0];
}

int main() {
    float S;
    float r;
    float vol;
    string type;
    float K;
    float T;
    int step;

    //in the book (intro to quant finance by willmott)
    S = 100;
    K = 100;
    r = 0.1;
    vol = 0.2;
    T = 4.0f/12.0f;
    step = 4;

    cout << option_price_Europe(
        S, vol, r, K, T, step, "Europe"
    ) << endl;

    type = "American";

    cout << option_price_Europe(
        S, vol, r, K, T, step, "American"
    ) << endl;

    //Deep in-the-money PUT hence all 0
    S = 50;
    K = 100;
    r = 0.10;
    vol = 0.20;
    T = 1;
    step = 6;

    cout << option_price_Europe(
        S, vol, r, K, T, step, "Europe"
    ) << endl;

    type = "American";

    cout << option_price_Europe(
        S, vol, r, K, T, step, "American"
    ) << endl;

    //High interest rate call
    S = 200;
    K = 50;
    r = 0.3;
    vol = 0.1;
    T = 5;
    step = 6;

    cout << option_price_Europe(
        S, vol, r, K, T, step, "Europe"
    ) << endl;

    type = "American";

    cout << option_price_Europe(
        S, vol, r, K, T, step, "American"
    ) << endl;
}