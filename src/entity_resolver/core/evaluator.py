import itertools
from .utils import WithLogger, logtime, ClusteringMetrics


class Evaluator(WithLogger):

    strategy_funcs = {
        'ami': ClusteringMetrics.ami,
        'v-measure': ClusteringMetrics.v_measure,
        'precision-recall': ClusteringMetrics.precision_recall
    }

    def __init__(
        self, strategy='precision-recall',
        plot_prc=False, verbose=0, **kwargs
    ):
        self._strategy_str = strategy
        self.strategy = self.strategy_funcs[strategy]
        self.plot_prc = plot_prc
        self._kwargs = kwargs
        super().__init__(verbose)

    @logtime('Time taken for evaluation')
    def evaluate(self, labels, preds):
        labels, preds = list(labels.values()), list(preds.values())
        self._logger.debug(f'Number of references in labels: {len(labels)}')
        self._logger.debug(f'Number of clusters in labels: {len(set(labels))}')
        self._logger.debug(f'Number of references in preds: {len(preds)}')
        self._logger.debug(f'Number of clusters in preds:: {len(set(preds))}')
        return self.strategy(labels, preds, **self._kwargs)
