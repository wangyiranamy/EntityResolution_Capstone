import itertools
import logging
import numpy as np
from sklearn import metrics
from .utils import WithLogger, logtime


class ClusteringMetrics:

    _logger = logging.getLogger('Evaluator')

    @classmethod
    def precision_recall(cls, labels, preds, log=True, **kwargs):
        cmatrix = metrics.cluster.contingency_matrix(labels, preds)
        row_sum, col_sum = np.sum(cmatrix, axis=1), np.sum(cmatrix, axis=0)
        pairs = cmatrix * (cmatrix-1) // 2
        label_pairs = row_sum * (row_sum-1) // 2
        pred_pairs = col_sum * (col_sum-1) // 2
        tp = np.sum(pairs)
        fp = np.sum(pred_pairs - np.sum(pairs, axis=0))
        fn = np.sum(label_pairs - np.sum(pairs, axis=1))
        precision = tp / (tp+fp)
        recall = tp / (tp+fn)
        f1 = 2*precision*recall / (precision+recall)
        if log:
            cls._logger.debug(f'True positive count: {tp}')
            cls._logger.debug(f'False positive count: {fp}')
            cls._logger.debug(f'False negative count: {fn}')
            cls._logger.info(f'Precision: {precision}')
            cls._logger.info(f'Recall: {recall}')
            cls._logger.info(f'F1 score: {f1}')
        return precision, recall, f1

    @classmethod
    def v_measure(cls, labels, preds, **kwargs):
        score = metrics.v_measure_score(labels, preds)
        cls._logger.info(f'V-measure score: {score}')
        return score

    @classmethod
    def ami(cls, labels, preds, average_method='max', **kwargs):
        cls._logger.debug(f'average_method: {average_method}')
        score = metrics.adjusted_mutual_info_score(
            labels, preds,
            average_method=average_method
        )
        cls._logger.info(f'Adjusted mutual information: {score}')
        return score


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
