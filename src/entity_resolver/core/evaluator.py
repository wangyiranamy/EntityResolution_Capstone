import itertools
from .utils import WithLogger, logtime, ClusteringMetrics


class Evaluator(WithLogger):

    _strategy_funcs = {
        'ami': ClusteringMetrics.ami,
        'v_measure': ClusteringMetrics.v_measure,
        'precision_recall': ClusteringMetrics.precision_recall
    }

    def __init__(
        self, strategy='precision_recall',
        plot_prc=False, verbose=0, **kwargs
    ):
        self._strategy_str = strategy
        self.strategy = self._strategy_funcs[strategy]
        self.plot_prc = plot_prc
        self._kwargs = kwargs
        super().__init__(verbose)

    @logtime('Time taken for evaluation')
    def evaluate(self, labels, preds):
        sorted_preds = list()
        for _, cluster_id in sorted(preds.items(), key=lambda pair: pair[0]):
            sorted_preds.append(cluster_id)
        labels = list(labels.values())
        preds = sorted_preds
        self._logger.debug(f'Number of references in labels: {len(labels)}')
        self._logger.debug(f'Number of clusters in labels: {len(set(labels))}')
        self._logger.debug(f'Number of references in preds: {len(preds)}')
        self._logger.debug(f'Number of clusters in preds:: {len(set(preds))}')
        return self.strategy(labels, preds, **self._kwargs)
