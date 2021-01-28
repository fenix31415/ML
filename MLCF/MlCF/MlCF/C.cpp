#include <iostream>
#include <iomanip>
#include <vector>
#include <numeric>
#include <algorithm>
#include <cmath>
#include <string>
#include <cassert>

namespace C {

	typedef std::vector<std::vector<long double>> dataset_t;
	typedef std::vector<long double> dataset_y_t;
	typedef std::vector<long double> object_t;

	std::string ro_type, kernel_type, window_type;
	long double true_h;

	long double get_dist(object_t x, object_t q) {
		long double ans = 0;
		switch (ro_type[0]) {
		case 'e': // euclidean
			for (int i = 0; i < (int)x.size(); ++i) {
				ans += (x[i] - q[i]) * (x[i] - q[i]);
			}
			return sqrt(ans);
		case 'm': // manhattan
			for (int i = 0; i < (int)x.size(); ++i) {
				ans += abs(x[i] - q[i]);
			}
			return ans;
		case 'c': // chebyshev
			for (int i = 0; i < (int)x.size(); ++i) {
				ans += std::max(ans, abs(x[i] - q[i]));
			}
			return ans;
		default:
			return 1.0;
		}
	}

	bool is_inf() {
		return kernel_type == "gaussian" || kernel_type == "logistic" || kernel_type == "sigmoid";
	}

	long double kernel(long double x) {
		if (kernel_type[0] == '_') return x;
		if (abs(x) >= 1.0 && !is_inf()) return 0;

		const long double PI = 3.14159265358979323846;
		switch (kernel_type[0]) {
		case 'u': return 0.5; //uniform
		case 'e': return (1 - x * x) * 3 / 4; // epanechnikov
		case 'q': return ((1 - x * x) * (1 - x * x)) * 15 / 16; // quartic
		case 'g': return exp(-0.5 * x * x) / sqrt(2 * PI); // gaussian
		case 'c': return (PI / 4) * cos(PI * x / 2); // cosine
		case 'l': return 1.0 / (exp(x) + 2 + exp(-x)); // logistic
		case 's': return 2.0 / PI / (exp(x) + exp(-x)); // sigmoid
		case 't':
			if (kernel_type == "triangular") return 1 - abs(x);
			if (kernel_type == "triweight") return ((1 - x * x) * (1 - x * x) * (1 - x * x)) * 35 / 32; // triweight
			if (kernel_type == "tricube") return ((1 - abs(x * x * x)) * (1 - abs(x * x * x)) * (1 - abs(x * x * x))) * 70 / 81; // tricube
		default:
			assert(false);
			return x;
		}
	}

	long double ml_div(long double num, long double den) {
		if (num == 0.0) return 0;
		if (den == 0.0) return 1;
		return num / den;
	}

	long double get_ans(dataset_y_t& Y, std::vector<std::pair<long double, long double>>& dists) {
		long double num = 0, den = 0;
		for (int i = 0; i < (int)Y.size(); i++) {
			long double ro = kernel(ml_div(dists[i].first, true_h));
			num += Y[i] * ro;
			den += ro;
		}
		if (den != 0 || num != 0)
			return ml_div(num, den);
		kernel_type = "_unknown_pokemon_";
		return get_ans(Y, dists);
	}

	int main() {
		std::cout.precision(10);

		int N, M;
		std::cin >> N >> M;
		dataset_t dataset_x(N, std::vector<long double>(M));
		dataset_y_t dataset_y(N);
		for (int i = 0; i < N; i++) {
			for (int j = 0; j < M; j++) {
				std::cin >> dataset_x[i][j];
			}
			std::cin >> dataset_y[i];
		}
		object_t q(M);
		for (int i = 0; i < M; i++) {
			std::cin >> q[i];
		}
		std::cin >> ro_type;
		std::cin >> kernel_type;
		std::cin >> window_type;
		long double param_h;
		std::cin >> param_h;

		std::vector<std::pair<long double, long double>> dists(N);
		for (int i = 0; i < N; i++) {
			dists[i] = std::make_pair(get_dist(dataset_x[i], q), (long double)i);
		}
		std::sort(dists.begin(), dists.end());

		dataset_t sorted_dataset_x;
		dataset_y_t sorted_dataset_y;
		for (auto& pair_dist : dists) {
			sorted_dataset_x.push_back(dataset_x[pair_dist.second]);
			sorted_dataset_y.push_back(dataset_y[pair_dist.second]);
		}

		true_h = param_h;
		if (window_type != "fixed") {
			true_h = dists[param_h].first;
		}

		std::cout << get_ans(sorted_dataset_y, dists);
		return 0;
	}
}
