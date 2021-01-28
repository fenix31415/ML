#include <iostream>
#include <iomanip>
#include <vector>
#include <numeric>
#include <algorithm>
#include <cmath>

namespace O {

	int main() {
		std::cout.precision(10);
		int n, k;
		std::cin >> k >> n;
		std::vector<std::vector<long double>> X(k, std::vector<long double>());
		for (int i = 0; i < n; i++) {
			long double x, y; std::cin >> x >> y;
			X[x - 1].push_back(y);
		}
		long double ans = 0;
		for (auto& cur : X) {
			if (cur.size() == 0) continue;
			long double sum = 0;
			for (auto& i : cur) {
				sum += i;
			} sum /= cur.size();
			long double anss = 0;
			for (auto& i : cur) {
				anss += (i - sum) * (i - sum) / n;
			}
			ans += anss;
		}
		std::cout << ans;
		return 0;
	}
}
