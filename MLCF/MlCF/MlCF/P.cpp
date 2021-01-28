#include <iostream>
#include <iomanip>
#include <vector>
#include <numeric>
#include <algorithm>
#include <cmath>
#include <unordered_map>

namespace P {

    int main() {
        std::cout.precision(10);
        int n, k1, k2;
        std::cin >> k1 >> k2 >> n;
        std::vector<long double> cou1(k1), cou2(k2);
        std::vector<std::unordered_map<int, int>> map(k1);
        long long sum1 = 0, sum2 = 0;
        for (int i = 0; i < n; i++) {
            int x, y; std::cin >> x >> y, --x, --y;
            ++cou1[x], ++cou2[y], ++sum1, ++sum2;
            if (map[x].find(y) == map[x].end()) map[x][y] = 1;
            else ++map[x][y];
        }
        long double sum = sum1 * sum2 / n;
        for (int i = 0; i < k1; i++) {
            for (auto const& entry : map[i]) {
                long double e = cou1[i] * cou2[entry.first] / n;
                sum += -e + (entry.second - e) * (entry.second - e) / e;
            }
        }
        std::cout << sum;
        return 0;
    }
}
