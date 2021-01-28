#include <iostream>
#include <iomanip>
#include <vector>
#include <numeric>
#include <algorithm>
#include <cmath>
#include <string>
#include <unordered_set>
#include <unordered_map>

typedef std::pair<int, long double> Q;

class Node {
public:
    int i;
    int cl;
    bool leaf;
    Q q;
    Node* left = nullptr;
    Node* right = nullptr;

    Node(Q q, Node* l, Node* r) : i(0), cl(-1), leaf(false), q(q), left(l), right(r) {}
    Node(int cl) : i(0), cl(cl), leaf(true) {}
};

int M, K, H, n;
std::vector<std::vector<int>> X;
std::vector<int> Y;

std::pair<int, Node*> build_leaf(const std::vector<int>& left, const std::vector<int>& right) {
    std::vector<int> a(K, 0);
    for (int i = 0; i < left.size(); i++) {
        ++a[Y[left[i]]];
    }
    for (int i = 0; i < right.size(); i++) {
        ++a[Y[right[i]]];
    }

    int cl_max = -1, i_max = -1;
    for (int id : left) {
        if (cl_max < a[Y[id]]) {
            cl_max = a[Y[id]];
            i_max = Y[id];
        }
    }
    for (int id : right) {
        if (cl_max < a[Y[id]]) {
            cl_max = a[Y[id]];
            i_max = Y[id];
        }
    }
    return std::make_pair(1, new Node(i_max + 1));
}

Q split(std::vector<int>& cur, std::vector<int>& left, std::vector<int>& right) {
    double max_gini = -1000;
    double max_v = -2e9;
    int max_i = -1;

    for (int attr_idx = 0; attr_idx < M; ++attr_idx) {
        sort(cur.begin(), cur.end(), [attr_idx](int i1, int i2) {return X[i1][attr_idx] < X[i2][attr_idx]; });

        std::vector<int> left_class_counters(K, 0), right_class_counters(K, 0);
        for (int id : cur) ++right_class_counters[Y[id]];

        double l_sum = 0, r_sum = 0;

        for (int x : right_class_counters) {
            r_sum += pow(x, 2);
        }

        for (size_t i = 0; i < cur.size(); i++) {
            int curr_idx = cur[i];

            r_sum -= pow(right_class_counters[Y[curr_idx]], 2);
            r_sum += pow(--right_class_counters[Y[curr_idx]], 2);
            l_sum -= pow(left_class_counters[Y[curr_idx]], 2);
            l_sum += pow(++left_class_counters[Y[curr_idx]], 2);

            double left_item_count = i + 1;
            double right_item_count = cur.size() - left_item_count;

            double curr_gini = left_item_count ? l_sum / left_item_count : 0;
            if (right_item_count) curr_gini += r_sum / right_item_count;

            if (max_gini < curr_gini) {
                max_gini = curr_gini;
                max_i = attr_idx;
                max_v = right_item_count ? double(X[curr_idx][attr_idx] + X[cur[i + 1]][attr_idx]) / 2 : 2e9;
            }
        }
    }

    for (int xId : cur) {
        if (X[xId][max_i] < max_v) left.push_back(xId);
        else right.push_back(xId);
    }
    return std::make_pair(max_i, max_v);
}

std::pair<int, Node*> build(int d, std::vector<int>& cur) {
    std::vector<int> left, right;
    std::unordered_set<int> cl_set;
    for (auto id : cur) {
        cl_set.insert(Y[id]);
    }
    if (cl_set.size() == 1) {
        return build_leaf(cur, std::vector<int>());
    }

    Q q = split(cur, left, right);

    if (left.empty() || right.empty()) {
        return build_leaf(cur, std::vector<int>());
    } else if (d >= H) {
        return build_leaf(left, right);
    } else {
        auto l = build(d + 1, left), r = build(d + 1, right);
        return std::make_pair(l.first + r.first + 1, new Node(q, l.second, r.second));
    }
}

int num_node = 0;
void init(Node* tree) {
    tree->i = ++num_node;
    if (!tree->leaf) {
        init(tree->left);
        init(tree->right);
    }
}

void print(Node* t) {
    if (t->leaf) {
        std::cout << "C " << t->cl << "\n";
    } else {
        std::cout << "Q " << t->q.first + 1 << " " << t->q.second << " " << t->left->i << " " << t->right->i << "\n";
        print(t->left);
        print(t->right);
    }
}

int main() {
    std::cin >> M >> K >> H >> n;

    X = std::vector<std::vector<int>>(n, std::vector<int>(M));
    Y = std::vector<int>(n);
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < M; j++) {
            std::cin >> X[i][j];
        }
        std::cin >> Y[i]; --Y[i];
    }

    std::vector<int> indexes;
    for (int i = 0; i < n; i++) {
        indexes.push_back(i);
    }

    auto t = build(0, indexes);

    std::cout << t.first << "\n";
    init(t.second);
    print(t.second);

	return 0;
}
