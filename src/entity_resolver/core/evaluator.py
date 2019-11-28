import itertools
import logging
from sklearn import metrics
from .utils import WithLogger, logtime


class ClusteringMetrics:

    _logger = logging.getLogger('Evaluator')

    @classmethod
    def precision_recall(cls, labels, preds):
        tp, fp, fn = 0, 0, 0
        for p1, p2 in itertools.combinations(labels, 2):
            if labels[p1] == labels[p2] and preds[p1] == preds[p2]:
                tp += 1
            elif labels[p1] != labels[p2] and preds[p1] == preds[p2]:
                fp += 1
            elif labels[p1] == labels[p2] and preds[p1] != preds[p2]:
                fn += 1
        precision = tp / (tp+fp)
        recall = tp / (tp+fn)
        f1 = 2*precision*recall / (precision+recall)
        cls._logger.info(f'precision: {precision}')
        cls._logger.info(f'recall: {recall}')
        cls._logger.info(f'f1 score: {f1}')
        return precision, recall, f1

    @classmethod
    def v_measure(cls, labels, preds):
        labels = cls._reorder(labels)
        preds = cls._reorder(preds)
        score = metrics.v_measure_score(labels, preds)
        cls._logger.info(f'v-measure score: {score}')
        return score

    @classmethod
    def ami(cls, labels, preds):
        labels = cls._reorder(labels)
        preds = cls._reorder(preds)
        score = metrics.adjusted_mutual_info_score(
            labels, preds,
            average_method='max'
        )
        cls._logger.info(f'adjusted mutual information: {score}')
        return score

    @staticmethod
    def _reorder(labels):
        ordered_pairs = sorted(labels.items(), key=lambda pair: pair[0])
        return [pair[1] for pair in ordered_pairs]


class Evaluator(WithLogger):

    strategy_funcs = {
        'ami': ClusteringMetrics.ami,
        'v-measure': ClusteringMetrics.v_measure,
        'precision-recall': ClusteringMetrics.precision_recall
    }

    def __init__(self, strategy='precision-recall'):
        self._strategy_str = strategy
        self.strategy = self.strategy_funcs[strategy]
        super().__init__()

    @logtime('Time taken for evaluation')
    def evaluate(self, labels, preds):
        num_label_clts = len(set(labels.values()))
        num_pred_clts = len(set(preds.values()))
        self._logger.debug(f'Number of references in labels: {len(labels)}')
        self._logger.debug(f'Number of clusters in labels: {num_label_clts}')
        self._logger.debug(f'Number of references in preds: {len(preds)}')
        self._logger.debug(f'Number of clusters in preds:: {num_pred_clts}')
        return self.strategy(labels, preds)
