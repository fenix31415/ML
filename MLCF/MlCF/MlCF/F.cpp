#include <iostream>
#include <iomanip>
#include <vector>
#include <numeric>
#include <algorithm>
#include <cmath>
#include <string>
#include <set>
#include <unordered_map>

namespace F {

	int k;
	int alpha;
	std::vector<std::pair<std::pair<long double, int>, std::unordered_map<std::string, int>>> data;
	std::vector<int> class_counts;
	std::set<std::string> allwords;

	void get_ans() {
		std::vector<long double> anses(k);
		int len; std::cin >> len;
		std::set<std::string> words;
		for (int i = 0; i < len; i++) {
			std::string word;
			std::cin >> word;
			words.insert(word);
		}

		for (int cl = 0; cl < k; cl++) {
			long double ans = 0;
			if (class_counts[cl] > 0) {
				ans += data[cl].first.first;
				for (auto& word : allwords) {
					//if (allwords.find(word) != allwords.end()) {
					long double den = data[cl].first.second + alpha * 2;
					long double num = alpha + data[cl].second[word];
					if (words.find(word) == words.end()) {
						ans += log(1 - num / den);
					} else {
						ans += log(num / den);
					}
					//}
				}
			}
			anses[cl] = ans;
		}
		long double max = 0;
		for (auto& i : anses) {
			max = std::max(max, i);
		}
		for (int i = 0; i < k; i++) {
			anses[i] -= max;
			if (class_counts[i] > 0) anses[i] = exp(anses[i]);
			else anses[i] = 0;
		}
		long double sum = 0;
		for (auto& i : anses) {
			sum += i;
		}
		for (int i = 0; i < k; i++) {
			anses[i] /= sum;
		}
		for (int i = 0; i < k; i++) {
			std::cout << anses[i] << " ";
		} std::cout << "\n";
	}

	int main() {
		std::cout.precision(10);
		std::cin >> k;
		std::vector<int> lambdas(k);
		int sumlambdas = 0;
		for (int i = 0; i < k; i++) {
			std::cin >> lambdas[i];
			sumlambdas += lambdas[i];
		}
		int n; std::cin >> alpha >> n;
		std::vector<std::pair<int, std::set<std::string>>> train(n);
		for (int m = 0; m < n; m++) {
			int cl, len; std::cin >> cl >> len;
			train[m] = std::make_pair(cl - 1, std::set<std::string>());
			std::string word;
			for (int i = 0; i < len; i++) {
				std::cin >> word;
				train[m].second.insert(word);
			}
		}

		class_counts = std::vector<int>(k);
		std::vector<std::unordered_map<std::string, int>> wordsinmsg_counts;
		wordsinmsg_counts = std::vector<std::unordered_map<std::string, int>>(10, std::unordered_map<std::string, int>());
		allwords = std::set<std::string>();
		for (auto& msg : train) {
			int cl = msg.first;
			++class_counts[cl];
			for (auto& word : msg.second) {
				if (wordsinmsg_counts[cl].find(word) == wordsinmsg_counts[cl].end()) {
					wordsinmsg_counts[cl][word] = 0;
				}
				++wordsinmsg_counts[cl][word];
				allwords.insert(word);
			}
		}

		data = std::vector<std::pair<std::pair<long double, int>, std::unordered_map<std::string, int>>>(10);
		for (int cl = 0; cl < k; cl++) {
			if (class_counts[cl]) {
				long double x = log((long double)class_counts[cl] * lambdas[cl] / (n + sumlambdas - k));
				data[cl] = std::make_pair(std::make_pair(x, class_counts[cl]), wordsinmsg_counts[cl]);
			}
		}

		int m; std::cin >> m;
		for (int i = 0; i < m; i++) {
			get_ans();
		}
		return 0;
	}
}
