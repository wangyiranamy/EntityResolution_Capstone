from .core import Resolver, Evaluator
from .parser import GraphParser, GroundTruthParser


class EntityResolver:

    def __init__(
        self, attr_types, blocking_strategy, raw_blocking=False, alpha=0.5,
        weights=None, attr_strategy=dict(), rel_strategy=None,
        blocking_threshold=3, bootstrap_strategy=None, raw_bootstrap=False,
        edge_match_threshold=1, similarity_threshold=0.8,
        evaluator_strategy='precision-recall', **kwargs
    ):
        self.blocking_strategy = blocking_strategy
        self.raw_blocking = raw_blocking
        self.alpha = alpha
        self.weights = weights
        self.attr_strategy = attr_strategy
        self.rel_strategy = rel_strategy
        self.blocking_threshold = blocking_threshold
        self.bootstrap_strategy = bootstrap_strategy
        self.raw_bootstrap = raw_bootstrap
        self.edge_match_threshold = edge_match_threshold
        self.similarity_threshold = similarity_threshold
        self._kwargs = kwargs
        self._graph_parser = GraphParser(attr_types)
        self._ground_truth_parser = GroundTruthParser()
        self._resolver = Resolver(
            blocking_strategy, raw_blocking=raw_blocking, alpha=alpha,
            weights=weights, attr_strategy=attr_strategy,
            rel_strategy=rel_strategy, blocking_threshold=blocking_threshold,
            bootstrap_strategy=bootstrap_strategy, raw_bootstrap=raw_bootstrap,
            edge_match_threshold=edge_match_threshold,
            similarity_threshold=similarity_threshold, **kwargs
        )
        self._evaluator = Evaluator(strategy=evaluator_strategy)

    def __getattr__(self, name):
        try:
            return self._kwargs[name]
        except KeyError:
            raise AttributeError(f'No attribute named {name}')

    def resolve(self, graph_path):
        graph = self._graph_parser.parse(graph_path)
        return self._resolver.resolve(graph)

    def evaluate(self, ground_truth_path, resolved_mapping):
        ground_truth = self._ground_truth_parser.parse(ground_truth_path)
        return self._evaluator.evaluate(ground_truth, resolved_mapping)

    def resolve_and_eval(self, graph_path, ground_truth_path):
        resolved_mapping = self.resolve(graph_path)
        return self.evaluate(ground_truth_path, resolved_mapping)
