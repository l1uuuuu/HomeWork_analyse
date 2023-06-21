import math
import random
from collections import Counter

class DecisionTree:
    def __init__(self):
        self.tree = None
        self.feature_names = None

    def fit(self, X, y, feature_names):
        data = [X[i] + [y[i]] for i in range(len(X))]
        self.feature_names = feature_names
        self.tree = self.build_tree(data, feature_names)

    def predict(self, X):
        result = []
        for x in X:
            result.append(self.classify(self.tree, x))
        return result

    def build_tree(self, data, feature_names):
        if len(data) == 0:
            return None
        if len(set([d[-1] for d in data])) == 1:
            return data[0][-1]
        if len(data[0]) == 1:
            return Counter([d[-1] for d in data]).most_common(1)[0][0]
        best_feature_index = self.choose_best_feature(data)
        best_feature_name = feature_names[best_feature_index]
        tree = {best_feature_name: {}}
        feature_names = feature_names[:best_feature_index] + feature_names[best_feature_index+1:]
        unique_values = set([d[best_feature_index] for d in data])
        for value in unique_values:
            sub_data = [d[:best_feature_index]+d[best_feature_index+1:] for d in data if d[best_feature_index] == value]
            tree[best_feature_name][value] = self.build_tree(sub_data, feature_names)
        return tree

    def choose_best_feature(self, data):
        num_features = len(data[0]) - 1
        base_entropy = self.calc_entropy(data)
        best_info_gain = 0.0
        best_feature_index = -1
        for i in range(num_features):
            unique_values = set([d[i] for d in data])
            new_entropy = 0.0
            for value in unique_values:
                sub_data = [d for d in data if d[i] == value]
                prob = len(sub_data) / float(len(data))
                new_entropy += prob * self.calc_entropy(sub_data)
            info_gain = base_entropy - new_entropy
            if info_gain > best_info_gain:
                best_info_gain = info_gain
                best_feature_index = i
        return best_feature_index

    def calc_entropy(self, data):
        num_entries = len(data)
        label_counts = {}
        for d in data:
            label = d[-1]
            if label not in label_counts.keys():
                label_counts[label] = 0
            label_counts[label] += 1
        entropy = 0.0
        for key in label_counts:
            prob = float(label_counts[key]) / num_entries
            entropy -= prob * math.log(prob, 2)
        return entropy

    def classify(self, tree, x):
        root_name = list(tree.keys())[0]
        second_dict = tree[root_name]
        index = self.feature_names.index(root_name)
        key = x[index]
        if key not in second_dict:
            return key
        if isinstance(second_dict[key], dict):
            class_label = self.classify(second_dict[key], x)
        else:
            class_label = second_dict[key]
        return class_label


