#include <iostream>
#include <iomanip>
#include <vector>
#include <numeric>
#include <algorithm>
#include <cmath>

namespace N {

    int main() {
        int n, k;
        std::cin >> k >> n;
        std::vector<long long> X(n);
        std::vector<std::vector<long long>> classes(k + 1, std::vector<long long>());
        for (int i = 0; i < n; i++) {
            std::cin >> X[i];
            long long y; std::cin >> y;
            classes[y - 1].push_back(X[i]);
        }
        std::sort(X.begin(), X.end());
        for (int i = 0; i < k; i++) {
            std::sort(classes[i].begin(), classes[i].end());
        }
        long long all = 0;
        for (long long i = 1; i < n; i++) {
            all += 2 * i * (n - i) * (X[i] - X[i - 1]);
        }
        long long in = 0;
        for (long long class_num = 0; class_num < k; class_num++) {
            long long len = classes[class_num].size();
            for (long long i = 1; i < len; ++i) {
                in += 2 * i * (len - i) * (classes[class_num][i] - classes[class_num][i - 1]);
            }
        }
        std::cout << in << "\n" << all - in;
        return 0;
    }
}
