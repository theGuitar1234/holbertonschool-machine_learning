#!/usr/bin/env python3
"""Random forest module."""

import numpy as np

Decision_Tree = __import__("8-build_decision_tree").Decision_Tree


class Random_Forest:
    """Random forest using multiple randomly-split decision trees."""

    def __init__(self, n_trees=100, max_depth=10, min_pop=1, seed=0):
        self.numpy_predicts = []
        self.target = None
        self.explanatory = None
        self.numpy_preds = None
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.min_pop = min_pop
        self.seed = seed

    def predict(self, explanatory):
        """Predict classes using majority vote across all trees."""
        preds = np.array([pred(explanatory) for pred in self.numpy_preds])

        n_classes = int(preds.max()) + 1
        counts = np.eye(n_classes, dtype=int)[preds]  # (t, n, c)
        votes = counts.sum(axis=0)                    # (n, c)

        return votes.argmax(axis=1)

    def fit(self, explanatory, target, n_trees=100, verbose=0):
        """Train n_trees random decision trees on the same dataset."""
        self.target = target
        self.explanatory = explanatory
        self.numpy_preds = []

        depths = []
        nodes = []
        leaves = []
        accuracies = []

        for i in range(n_trees):
            tree = Decision_Tree(
                max_depth=self.max_depth,
                min_pop=self.min_pop,
                seed=self.seed + i,
            )
            tree.fit(explanatory, target)
            self.numpy_preds.append(tree.predict)

            depths.append(tree.depth())
            nodes.append(tree.count_nodes())
            leaves.append(tree.count_nodes(only_leaves=True))
            accuracies.append(tree.accuracy(tree.explanatory, tree.target))

        if verbose == 1:
            forest_acc = self.accuracy(self.explanatory, self.target)
            print(
                f"""  Training finished.
    - Mean depth                     : {np.array(depths).mean()}
    - Mean number of nodes           : {np.array(nodes).mean()}
    - Mean number of leaves          : {np.array(leaves).mean()}
    - Mean accuracy on training data : {np.array(accuracies).mean()}
    - Accuracy of the forest on td   : {forest_acc}"""
            )

    def accuracy(self, test_explanatory, test_target):
        """Compute accuracy of the forest."""
        preds = self.predict(test_explanatory)
        return np.sum(np.equal(preds, test_target)) / test_target.size
