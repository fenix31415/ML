#include <iostream>
#include <iomanip>
#include <vector>
#include <numeric>
#include <algorithm>
#include <cmath>

namespace L {

	int main() {
		std::cout.precision(10);
		int n; std::cin >> n;
		std::vector<long long> a, b;
		a.resize(n), b.resize(n);
		long double a_avg = 0, b_avg = 0;
		for (int i = 0; i < n; i++) {
			std::cin >> a[i] >> b[i];
			a_avg += a[i], b_avg += b[i];
		}
		a_avg /= n;
		b_avg /= n;

		long double num = 0, den1 = 0, den2 = 0;
		for (int i = 0; i < n; i++) {
			num += (a[i] - a_avg) * (b[i] - b_avg);
			den1 += (a[i] - a_avg) * (a[i] - a_avg);
			den2 += (b[i] - b_avg) * (b[i] - b_avg);
		}
		if (den1 * den2 < 1e-10) std::cout << 0;
		else std::cout << num / sqrt(den1 * den2);
		return 0;
	}
}
