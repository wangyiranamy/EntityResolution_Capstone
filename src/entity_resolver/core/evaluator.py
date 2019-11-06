import functools
from sklearn import metrics


class Evaluator:

    strategy_funcs = {
        'ami': functools.partial(
            metrics.adjusted_mutual_info_score,
            average_method='max'
        ),
        'v-measure': metrics.v_measure_score
    }

    def __init__(self, strategy='ami'):
        self.strategy = self.strategy_funcs[strategy]

    @staticmethod
    def _reorder(mapping):
        ordered_pairs = sorted(mapping.items(), key=lambda pair: pair[0])
        return [pair[1] for pair in ordered_pairs]

    def evaluate(self, ground_truth, resolved_mapping):
        labels = self._reorder(ground_truth)
        preds = self._reorder(resolved_mapping)
        return self.strategy(labels, preds)
