import logging
from matplotlib import pyplot as plt
from .core import Resolver, Evaluator
from .core.utils import WithLogger, logtime
from .parser import GraphParser, GroundTruthParser


class EntityResolver(WithLogger):

    def __init__(
        self, attr_types, blocking_strategy, raw_blocking=False, alpha=0,
        weights=None, attr_strategy=dict(), rel_strategy=None,
        blocking_threshold=3, bootstrap_strategy=None, raw_bootstrap=False,
        edge_match_threshold=1, first_attr=None, first_attr_raw=False,
        second_attr=None, second_attr_raw=False, linkage='max',
        similarity_threshold=0.935, evaluator_strategy='precision-recall',
        seed=None, plot_prc=False, verbose=0, **kwargs
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
        self.first_attr = first_attr
        self.first_attr_raw = first_attr_raw
        self.second_attr = second_attr
        self.second_attr_raw = second_attr_raw
        self.linkage = linkage
        self.similarity_threshold = similarity_threshold
        self.seed = seed
        self.plot_prc = plot_prc
        self._kwargs = kwargs
        self._graph_parser = GraphParser(attr_types, verbose=verbose)
        self._ground_truth_parser = GroundTruthParser(verbose=verbose)
        self._resolver = Resolver(
            blocking_strategy, raw_blocking=raw_blocking, alpha=alpha,
            weights=weights, attr_strategy=attr_strategy,
            rel_strategy=rel_strategy, blocking_threshold=blocking_threshold,
            bootstrap_strategy=bootstrap_strategy, raw_bootstrap=raw_bootstrap,
            edge_match_threshold=edge_match_threshold, first_attr=first_attr,
            first_attr_raw=first_attr_raw, second_attr=second_attr,
            second_attr_raw=second_attr_raw, linkage=linkage,
            similarity_threshold=similarity_threshold, seed=seed,
            plot_prc=plot_prc, verbose=verbose, **kwargs
        )
        self._evaluator = Evaluator(
            strategy=evaluator_strategy, plot_prc=plot_prc,
            verbose=verbose, **kwargs
        )
        super().__init__(verbose=verbose)

    def __getattr__(self, name):
        try:
            return self._kwargs[name]
        except KeyError:
            raise AttributeError(f'No attribute named {name}')

    @logtime('Time taken for the whole resolution process')
    def resolve(self, graph_path):
        if self.plot_prc:
            raise ValueError(
                'Cannot plot precision-recall curve when only resolving'
                'entities. Use resolve_and_eval instead.'
            )
        graph = self._graph_parser.parse(graph_path)
        resolved_mapping = self._resolver.resolve(graph)
        self._resolver.log_time()
        return resolved_mapping

    def evaluate(self, ground_truth_path, resolved_mapping):
        ground_truth = self._parse_ground_truth(ground_truth_path)
        return self._evaluator.evaluate(ground_truth, resolved_mapping)

    def resolve_and_eval(self, graph_path, ground_truth_path):
        ground_truth = self._parse_ground_truth(ground_truth_path)
        graph = self._graph_parser.parse(graph_path)
        resolver_res = self._resolver.resolve(graph, ground_truth)
        self._resolver.log_time()
        if self.plot_prc:
            resolved_mapping, prc_list = resolver_res
            self._plot(prc_list)
            score = self._evaluator.evaluate(ground_truth, resolved_mapping)
            return score, prc_list
        else:
            resolved_mapping = resolver_res
            return self._evaluator.evaluate(ground_truth, resolved_mapping)

    def _parse_ground_truth(self, ground_truth_path):
        return self._ground_truth_parser.parse(ground_truth_path)

    def _plot(self, prc_list):
        precisions, recalls = zip(*prc_list)
        plt.xlabel('precision')
        plt.ylabel('recall')
        plt.title('Precision-Recall Curve')
        plt.plot(list(precisions), list(recalls))
        plt.show()
