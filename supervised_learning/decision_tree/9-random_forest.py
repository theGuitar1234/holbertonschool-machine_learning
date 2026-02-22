import numpy as np
Decision_Tree = __import__('8-build_decision_tree').Decision_Tree


class Random_Forest():
    def __init__(self, n_trees=100, max_depth=10, min_pop=1, seed=0):
        self.numpy_predicts = []
        self.target = None
        self.numpy_preds = None
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.min_pop = min_pop
        self.seed = seed

    def predict(self, explanatory):
        """Predict classes using majority vote across all trees."""
        preds = np.array([pred(explanatory) for pred in self.numpy_preds])
        n_classes = int(preds.max()) + 1
        counts = np.eye(n_classes, dtype=int)[preds]
        votes = counts.sum(axis=0)
        return votes.argmax(axis=1)

    def fit(self, explanatory, target, n_trees=100, verbose=0):
        """Docstring."""
        self.target = target
        self.explanatory = explanatory
        self.numpy_preds = []
        depths = []
        nodes = []
        leaves = []
        accuracies = []
        for i in range(n_trees):
            T = Decision_Tree(
                max_depth=self.max_depth,
                min_pop=self.min_pop,
                seed=self.seed + i
            )
            T.fit(explanatory, target)
            self.numpy_preds.append(T.predict)
            depths.append(T.depth())
            nodes.append(T.count_nodes())
            leaves.append(T.count_nodes(only_leaves=True))
            accuracies.append(T.accuracy(T.explanatory, T.target))

        if verbose == 1:
            str = self.accuracy(self.explanatory, self.target)
            print(
                f"""  Training finished.
    - Mean depth                     : {np.array(depths).mean()}
    - Mean number of nodes           : {np.array(nodes).mean()}
    - Mean number of leaves          : {np.array(leaves).mean()}
    - Mean accuracy on training data : {np.array(accuracies).mean()}
    - Accuracy of the forest on td   : {str}"""
            )

    def accuracy(self, test_explanatory, test_target):
        """Docstring."""
        return np.sum(
            np.equal(self.predict(test_explanatory), test_target)
        ) / test_target.size
