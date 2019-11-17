import functools
import itertools
from sklearn import metrics


class ClusteringMetrics:

    @staticmethod
    def precision_recall(labels, preds):
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
        return precision, recall, f1


class Evaluator:

    strategy_funcs = {
        'ami': functools.partial(
            metrics.adjusted_mutual_info_score,
            average_method='max'
        ),
        'v-measure': metrics.v_measure_score,
        'precision-recall': ClusteringMetrics.precision_recall
    }

    def __init__(self, strategy='precision-recall'):
        self._strategy_str = strategy
        self.strategy = self.strategy_funcs[strategy]

    @staticmethod
    def _reorder(labels):
        ordered_pairs = sorted(labels.items(), key=lambda pair: pair[0])
        return [pair[1] for pair in ordered_pairs]

    def evaluate(self, labels, preds):
        if self._strategy_str != 'precision-recall':
            labels = self._reorder(labels)
            preds = self._reorder(preds)
        return self.strategy(labels, preds)
