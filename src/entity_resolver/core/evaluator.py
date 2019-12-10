""" Contains only one evaluator class to be used by the main module

Example:

    >>> from entity_resolver.core import Evaluator
    >>> evaluator = Evaluator(strategy_str='ami')  # AMI score
    >>> score = evaluator.evaluate(labels, preds)
"""

import itertools
import collections
from typing import Mapping, Dict, Any
from .utils import WithLogger, logtime, ClusteringMetrics


class Evaluator(WithLogger):
    """ An evaluator class for entity resolution evaluation.

    Args:
        strategy: The strategy name to be used for evaluation of the entity
            resolution performance. It is the same as ``evaluator_strategy``
            attribute of `~entity_resolver.main.EntityResolver`.
        verbose: Indicate how much information to be logged/printed in the
            console during the program execution. It is the same as ``verbose``
            attribute of `~entity_resolver.main.EntityResolver`.
        **kwargs: Additional keyword arguments that are only used in specific
            cases. Only ``average_method`` is a useful keyword here. Refer to
            the ``kwargs`` parameter of `~entity_resolver.main.EntityResolver`
            for details.

    Attributes:
        strategy (`str`): Same as ``strategy`` in the above parameters
            section.
        kwargs: Same as ``kwargs`` in the above parameters section.
        _strategy_funcs (`~typing.Dict`\ [`str`, `~typing.Callable`\ [[`~typing.Mapping`, `~collections.OrderedDict`], `~typing.Any`]]):
            This is a class attribute registering how each strategy string is
            associated with a evaluation function.
    """

    _strategy_funcs = {
        'ami': ClusteringMetrics.ami,
        'v_measure': ClusteringMetrics.v_measure,
        'precision_recall': ClusteringMetrics.precision_recall
    }

    def __init__(
        self, strategy: str = 'precision_recall',
        verbose: int = 0, **kwargs
    ):
        self.strategy = strategy
        self.kwargs = kwargs
        super().__init__(verbose)

    @property
    def strategy_func(self):
        """ `~typing.Callable`\ [[`~typing.Mapping`, `~collections.OrderedDict`], `~typing.Any`]:
        A strategy function that returns something that represents the
        performance of the entity resolution. After instantiation

        * If ``strategy_str`` is set to ``'ami'``, or ``'v_measure'``,
          the return type is a single `float` representing the score.
        * If ``strategy_str`` is set to ``'precision_recall'``
          (default), the return type is
          `~typing.Tuple`\ [`float`, `float`, `float`] representing
          precision, recall, and f1 scores in the order. Refer to
          :doc:`../advanced_guide` for more details.
        """
        return self._strategy_funcs[self.strategy]

    @logtime('Time taken for evaluation')
    def evaluate(self, labels: Mapping, preds: collections.OrderedDict) -> Any:
        """ Evaluate the performance given predicted results and ground truth.

        Args:
            labels: Mapping reference ids to cluster ids. The cluster ids may
                be different from those in the ground truth.
            preds: Mapping reference ids to cluster ids. The dictionary is
                sorted (key-value pairs are inserted) in ascending order of
                reference ids.

        Returns:
            Something that represents the performance of the entity resolution
            by applying the ``strategy_func`` property. Refer to the attribute
            documentation for possible returns.
        """
        sorted_preds = list()
        for _, cluster_id in sorted(preds.items(), key=lambda pair: pair[0]):
            sorted_preds.append(cluster_id)
        labels = list(labels.values())
        preds = sorted_preds
        self.logger.debug(f'Number of references in labels: {len(labels)}')
        self.logger.debug(f'Number of clusters in labels: {len(set(labels))}')
        self.logger.debug(f'Number of references in preds: {len(preds)}')
        self.logger.debug(f'Number of clusters in preds:: {len(set(preds))}')
        return self.strategy_func(labels, preds, **self.kwargs)
