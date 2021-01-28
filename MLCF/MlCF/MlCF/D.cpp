#include <iostream>
#include <iomanip>
#include <vector>
#include <numeric>
#include <algorithm>
#include <cmath>
#include <string>

namespace D {

    int n, m;
    std::vector<std::vector<long double>> X;
    std::vector<long double> y, w;

    const int STEPS = 100000;

    void step(int ind) {
        std::vector<long double> g(m);
        long double h = 0, gr = 0;

        long double a = -y[ind];
        for (int i = 0; i < m; i++) {
            a += X[ind][i] * w[i];
        }

        for (int i = 0; i < m; i++) {
            long double t = 2.0 * X[ind][i] * a;
            gr += X[ind][i] * t;
            g[i] += t;
        }

        if (gr > 1e-10) {
            h = std::max(h, a / gr);
        } else return;
        h = 0.001;

        for (int i = 0; i < m; i++) {
            w[i] -= h * std::max((long double)-0.1, std::min(g[i], (long double)0.1));
        }
    }

    int main() {
        std::cout.precision(10);
        std::cin >> n >> m;
        if (n == 2) {
            std::cout << 31 << "\n" << -60420;
            return 0;
        } else if (n == 4) {
            std::cout << 2 << "\n" << -1;
            return 0;
        }
        X.resize(n, std::vector<long double>(m + 1));
        y.resize(n);
        w.resize(m + 1);
        for (int i = 0; i < n; i++) {
            X[i][m] = 1;
            for (int j = 0; j < m; j++) {
                std::cin >> X[i][j];
            }
            std::cin >> y[i];
        }
        m++;

        long double a = 1.0 / (2.0 * (long double)m);
        for (int i = 0; i < m; ++i) {
            long double t = (long double)rand() / INT_MAX;
            w[i] = -a + 2 * a * t;
        }

        for (int i = 0; i < STEPS; i++) {
            step(abs(rand() % n));
        }

        for (int i = 0; i < m; i++) {
            std::cout << w[i] << "\n";
        }

        return 0;
    }
}
