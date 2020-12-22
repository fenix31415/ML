#include <iostream>
#include <iomanip>
#include <vector>
#include <numeric>

namespace B {

	int get_sum_col(const std::vector<std::vector<int>>& m, int c) {
		int ans = 0;
		for (int i = 0; i < m.size(); i++) {
			ans += m[i][c];
		}
		return ans;
	}

	int get_sum_row(const std::vector<std::vector<int>>& m, int c) {
		return std::accumulate(m[c].begin(), m[c].end(), 0);
	}

	int main() {
		std::cout.precision(10);

		int K, N = 0;
		std::cin >> K;
		std::vector<std::vector<int>> CM(K, std::vector<int>(K));
		for (int c = 0; c < K; c++) {
			for (int t = 0; t < K; t++) {
				std::cin >> CM[t][c];
				N += CM[t][c];
			}
		}

		double mic_f = 0;
		for (int c = 0; c < K; c++) {
			double col = get_sum_col(CM, c);
			double row = get_sum_row(CM, c);
			if (row > 0 && col > 0)
				mic_f += (2.0 * CM[c][c] * col) / (row + col);
		}
		mic_f /= N;

		double prec_w = 0, recall_w = 0;
		for (int c = 0; c < K; c++) {
			double sum_row = get_sum_row(CM, c);
			double sum_col = get_sum_col(CM, c);
			if (sum_row > 0)
				prec_w += CM[c][c] * sum_col / sum_row;
			if (sum_col > 0)
				recall_w += CM[c][c];
		}
		prec_w /= N;
		recall_w /= N;
		double mac_f = 2 * prec_w * recall_w / (prec_w + recall_w);

		std::cout << mac_f << "\n" << mic_f;
		return 0;
	}
}
