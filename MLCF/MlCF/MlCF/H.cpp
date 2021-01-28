#include <iostream>
#include <iomanip>
#include <vector>
#include <numeric>
#include <algorithm>
#include <cmath>
#include <string>

namespace H {

    int main() {
        std::cout.precision(10);

        int m; std::cin >> m;
        int n = 1 << m;
        int c = 0;
        std::vector<std::vector<long double>> net(0);
        for (int i = 0; i < n; i++) {
            int t; std::cin >> t;
            if (t) {
                ++c;
                net.push_back(std::vector<long double>(m));
                long double b = 0.5;
                for (int j = 0; j < m; j++) {
                    int xx = -1;
                    if (i & (1 << j)) xx = 1;
                    if (xx == 1) b -= 1;
                    net.back()[j] = xx;
                }
                net.back().push_back(b);
            }
        }
        if (net.size() > 0) {
            net.push_back(std::vector<long double>(net.size()));
            for (int i = 0; i < net.size() - 1; i++) {
                net.back()[i] = 1;
            }
            net.back().push_back(-0.5);

            std::cout << 2 << "\n" << net.size() - 1 << " " << 1 << "\n";
        } else {
            std::cout << 1 << " " << 1 << "\n";
            net.push_back(std::vector<long double>(m));
            for (int i = 0; i < m; i++) {
                net.back()[i] = 1;
            }
            net.back().push_back(-m - 0.5);
        }

        for (auto& xx : net) {
            for (auto& x : xx) {
                std::cout << x << " ";
            }std::cout << "\n";
        }
        return 0;
    }
}
