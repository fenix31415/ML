#include <iostream>
#include <vector>
#include <unordered_map>

namespace A {

	int main() {
		int M, N, K;
		std::cin >> N >> M >> K;

		std::unordered_map<int, std::vector<int>> classes;

		for (int i = 1; i <= N; i++) {
			int c;
			std::cin >> c;
			classes[c].push_back(i);
		}

		int fold_number = 0;
		std::vector<std::vector<int>> folds(K, std::vector<int>());
		for (auto& clazz : classes) {
			for (auto& obj : clazz.second) {
				folds[fold_number].push_back(obj);
				fold_number = (fold_number + 1) % K;
			}
		}

		for (auto& fold : folds) {
			std::cout << fold.size() << " ";
			for (auto& obj : fold) {
				std::cout << obj << " ";
			}
			std::cout << "\n";
		}
		return 0;
	}
}
