#include <iostream>
#include <iomanip>
#include <vector>
#include <numeric>
#include <algorithm>
#include <cmath>

namespace M {

	int main() {
		std::cout.precision(10);
		int n; std::cin >> n;
		std::vector<int> a(n), b(n), inds_a(n), inds_b(n);
		for (int i = 0; i < n; i++) {
			std::cin >> a[i] >> b[i];
			inds_a[i] = i, inds_b[i] = i;
		}

		std::sort(inds_a.begin(), inds_a.end(), [&a](int x, int y) { return a[x] < a[y]; });
		std::sort(inds_b.begin(), inds_b.end(), [&b](int x, int y) { return b[x] < b[y]; });

		std::vector<long double> ra(n), rb(n);
		for (int i = 0; i < n; i++) {
			ra[inds_a[i]] = (long double)i + 1;
			rb[inds_b[i]] = (long double)i + 1;
		}
		long double d = 0;
		for (int i = 0; i < n; i++) {
			d += (ra[i] - rb[i]) * (ra[i] - rb[i]);
		}
		std::cout << 1 - 6 * d / ((long double)n * n * n - n);

		return 0;
	}
}
