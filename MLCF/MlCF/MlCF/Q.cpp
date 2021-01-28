#include <iostream>
#include <iomanip>
#include <vector>
#include <numeric>
#include <algorithm>
#include <cmath>
#include <unordered_map>

namespace Q {

    int main() {
        std::cout.precision(10);
        int n, kx, ky;
        std::cin >> kx >> ky >> n;
        std::vector<long long> X(kx);
        std::vector<std::unordered_map<int, int>> map(kx);
        for (int i = 0; i < n; i++) {
            int x, y; std::cin >> x >> y, --x, --y;
            ++X[x];
            if (map[x].find(y) == map[x].end()) map[x][y] = 1;
            else ++map[x][y];
        }
        long double sum = 0;
        for (int i = 0; i < kx; i++) {
            long double summ = 0;
            for (auto const& entry : map[i]) {
                if (entry.second == 0) continue;
                long double p = (long double)entry.second / X[i];
                summ -= p * log(p);
            }
            sum += summ * X[i] / n;
        }
        std::cout << sum;

        return 0;
    }
}
