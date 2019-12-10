""" A collection of utility functions and classes.

This module consists of miscellaneous tools used by various modules. It helps
implement these modules' functionalities in a much cleaner and extensible way.
It includes components for timing, logging, computation, and implementation of
special data structures.
"""

import time
import collections
import inspect
import logging
import argparse
from functools import wraps, partial
from typing import (
    Callable, Any, Union, Mapping, Hashable, Optional,
    Sequence, List, Tuple, Collection, Counter, Iterable
)
import numpy as np
from sklearn import metrics
from py_stringmatching.similarity_measure import (
    jaro_winkler, soft_tfidf, jaro, levenshtein
)


class WithLogger:
    """ A class with a pre-configured logger attribute.

    This class is created as a parent class for any class that needs a built-in
    logger. The logger is pre-configured so that each of the class will have a
    uniform logging style.

    Args:
        verbose: Indicate how much information to be logged/printed in the
            console during the program execution. It is the same as ``verbose``
            attribute of `~entity_resolver.main.EntityResolver`.

    Attributes:
        _verbose (`int`): Same as in the parameters section.
        _logger (`~logging.Logger`): Encapsulated by the ``logger`` property
            for read-only. They are essentially the same.
    """

    def __init__(self, verbose: int = 0):
        self._verbose = verbose
        self._logger = logging.getLogger(self.__class__.__name__)
        self._config_logger()

    @property
    def logger(self):
        """ `~logging.Logger`:
        Configured according to the ``verbose`` attribute. Ready-only.
        """
        return self._logger

    @property
    def verbose(self):
        """ Same as the ``verbose`` argument in the parameters section."""
        return self._verbose

    @verbose.setter
    def verbose(self, value):
        self._verbose = value
        self.logger.setLevel(self._get_level())
        self._set_verbose(value)

    def _set_verbose(self, value):
        """ Remained empty for child classes to extend the verbose setter.

        Args:
            value: The verbose value to be set to.
        """
        pass

    def _get_level(self) -> int:
        """ Convert verbose to logging level

        Returns:
            * If ``verbose <= 0``, then return `logging.WARNING`.
            * If ``verbose == 1``, then return `logging.INFO`.
            * If ``verbose >= 2``, then return `logging.DEBUG`.
        """
        if self.verbose <= 0:
            level = logging.WARNING
        elif self.verbose == 1:
            level = logging.INFO
        else:
            level = logging.DEBUG
        return level

    def _config_logger(self) -> None:
        """ A helper method to initialize logger configuration."""
        level = self._get_level()
        handler = logging.StreamHandler()
        fmt = '[{asctime}] {levelname} {name}: {msg}'
        formatter = logging.Formatter(fmt=fmt, style='{')
        handler.setFormatter(formatter)
        self._logger.addHandler(handler)
        self._logger.setLevel(level)


def timeit(func: Callable) -> Callable:
    """ A decroator to register time spent by a (class) method.

    Args:
        func: A method or a class method to be decorated.

    Important:
        The class or object that the decorated (class) method is attached to
        must have a ``time_dict`` attribute of type:
        `~collections.defaultdict`. It should be instantiated using equivalent
        of the follow:

            >>> collections.defaultdict(lambda: [0, 0])

        This dictionary maps method names to a list of two entries:

        [<number of times called>, <total time spent by all calls>].

    Returns:
        A wrapper of the decorated function which alters the ``time_dict``
        attribute of the class or object the decorated function is attached
        to by incrementing the number of times called by one, and the total
        time spent by the time taken for this method call.
    """
    @wraps(func)
    def timed_func(obj: Any, *args, **kwargs) -> Any:
        start_time = time.time()
        res = func(obj, *args, **kwargs)
        end_time = time.time()
        time_list = obj.time_dict[func.__name__]
        time_list[0] += (end_time - start_time)
        time_list[1] += 1
        return res
    return timed_func


class subparser:
    """ A decorator factory for wrapping a function with a subparser adder.

    Args:
        subcommand: Name of the subcommand to be added.
        subcommand_help: Helper string to explain functionality of the
            subcommand tool to be added.
        *helps: Helper strings that will be associated in the subcommand tool
            with each keyword argument of the decorated function **in the
            order** of their appearances in the function declaration.

    Attributes:
        subcommand (`str`): Same as in the parameters section.
        subcommand_help (`str`): Same as in the parameters section.
        helps (`~typing.Iterable`\ [`str`]): Same as in the parameters
            section.
    """

    def __init__(
        self, subcommand: str, subcommand_help: str, *helps: Iterable[str]
    ):
        self.subcommand = subcommand
        self.subcommand_help = subcommand_help
        self.helps = helps

    def __call__(self, func: Callable) -> Callable:
        """ Decorate the input function.

        Args:
            func: This function should **only** take keyword arguments. It is
                the function to be wrapped into a subcommand tool.

        Returns:
            The returned function takes exactly one special object that is
            returned by `~argparse.ArgumentParser.add_subparsers` and add a
            subparser which performs the same task as the decorated function
            with help strings specified in the object attributes.

            The keyword arguments of the original function correspond to
            optional command line flags with names prefixed by '--', and their
            default values are used as default values of the optional flags as
            well.

            Eg., the function ``f(x=1)`` will results in an optional
            command line argument ``'--x'`` with default value ``1``.
        """
        return wraps(func)(partial(self._create_subparser, func))

    def _create_subparser(
        self,
        func: Callable,
        subparsers: argparse.Action
    ) -> None:
        """ A subparser adder function that wraps the input function.

        This helper method implements a subparser adder function given an input
        function. The core ``__call__`` method uses this function by creating
        a `functools.partial` object that fixes the ``func`` argument with the
        function to be decorated, which is then returned as the desired
        wrapper.

        Args:
            func: The function to be wrapped.
            subparsers: A special action object in the `argparse` module that
                has a single ``add_parser`` method.
        """
        parser = subparsers.add_parser(
            self.subcommand,
            help=self.subcommand_help
        )
        func_info = self._parse_func(func)
        for name, default, arg_help in func_info:
            parser.add_argument(
                f'--{name}', type=type(default),
                default=default, help=arg_help
            )
        parser_function = partial(self._parser_function, func, func_info)
        parser.set_defaults(func=parser_function)

    def _parse_func(self, func: Callable) -> List[Tuple[str, Any, str]]:
        """ Extract information about the input function's keyword arguments.

        Args:
            func: This function's keyword arguments information is to be
                extracted and returned.

        Returns:
            Each item in the returned list is a tuple stroing the information
            about the input function's keyword arguments and the help string
            corresponding to the arguments:

            (<keyword name>, <default value>, <help string>)
        """
        sig = inspect.signature(func)
        func_info = list()
        for arg_help, (name, param) in zip(self.helps, sig.parameters.items()):
            func_info.append((name, param.default, arg_help))
        return func_info

    def _parser_function(
        self,
        func: Callable,
        func_info: List[Tuple[str, Any, str]],
        args: argparse.Namespace
    ) -> Any:
        """ Take arguments from ``args`` and feed into the input function.

        This helper method implements how the input function should parse and
        take arguments from a `~argparse.Namespace` object based on the keyword
        arguments information of this function. Similar to ``_create_parse``,
        it is also used as a partial object with fixed ``func`` and
        ``func_info``, which is then stored in the ``func`` namespace of the
        subparser and called after parsing command line arguments.

        Args:
            func: The function to be wrapped.
            func_info: The keyword arguments information of the ``func``
                argument. It is the returned value of ``_parse_func`` method.
            args: The namespace that the wrapper function should take arguments
                from.

        Returns:
            The same return as that of the ``func`` argument.
        """
        func_args = list()
        for name, _, _ in func_info:
            func_args.append(getattr(args, name))
        return func(*func_args)


class logtime:
    """ A decorator factory for wrapping (class) methods to log time taken.

    Args:
        header: The prefix string attached before the time in the log output.
            Eg., if header is set to ``'time taken'``, then the log for a
            function that takes 1s will be ``'time taken: 1s'``. The colon and
            's' are always added.

    Important:
        The class or object of the decorated method must have the
        ``logger`` attribute of type `~logging.Logger`.

    Note:
        The logged time taken is always rounded up to two decimal places.

    Attributes:
        header (`str`): Same as in the parameters section.
    """

    def __init__(self, header: str):
        self.header = header

    def __call__(self, func: Callable) -> Callable:
        """ Decorate the input (class) method.

        Args:
            func: The method or class method to be decorated.

        Returns:
            A wrapper method that will log the time taken for the input method
            after it finishes.
        """
        return wraps(func)(
            lambda obj, *args, **kwargs:
                self._timed_func(func, obj, *args, **kwargs)
        )

    def _timed_func(self, func: Callable, obj: Any, *args, **kwargs) -> Any:
        """ Execute input (class) method and log its time taken.

        This helper method is used by ``__call__`` to create the wrapper
        method, since fixing the ``func`` parameter of this method with the
        decorated method will just results in a desired wrapper.

        Args:
            func: The method to be executed, whose time taken will also be
                logged.
            obj: The class or object that the input method is attached to.

        Returns:
            The same return as that of the ``func`` argument.
        """
        start_time = time.time()
        res = func(obj, *args, **kwargs)
        end_time = time.time()
        time_taken = round(end_time - start_time, 2)
        obj.logger.info(f'{self.header}: {time_taken}s')
        return res


class SimFuncFactory:
    """ A collection of attribute similarity function producer methods.

    This class consists only of class methods and a helper static method.

    The purpose of this class is to decouple the main clustering algorithm and
    the attribute similarity calculation. It is achieved by registering these
    producer methods in the `~entity_resolver.core.resolver.Resolver` object
    and use a parsing mechansim to preset how the attribute similarity on each
    attribute is supposed to be computed. The producers instead of similarity
    functions themselves are registered because certain similarity functions
    require extra information to be passed and preprocessed before computation
    of similarities.

    For detailed algorithms behind each of these similarity computation, please
    refer to :doc:`../advanced_guide`.
    """

    @classmethod
    def produce_stfidf(
        cls,
        weight: Union[float, int],
        corpus_list: Sequence[Sequence[str]],
        stfidf_threshold: Union[float, int] = 0.5,
        second_sim: str = 'jaro_winkler',
        jw_prefix_weight: Union[float, int] = 0.1,
        **kwargs
    ) -> Callable[[List[str], List[str]], float]:
        """ Return a function to compute soft-tfidf similarity

        Args:
            weight: The weight assigned to a particular reference attribute.
            corpus_list: A list of tokenized strings (a list of a list of
                words).
            stfidf_threshold: Used to filter words involved in computing tfidf
                similarity based on secondary similarity function.
            second_sim: Indicate the secondary similarity function. Valid
                values are ``'jaro_winkler'``, ``'jaro'`` and ``'scaled_lev'``.
            jw_prefix_weight: The prefix weight used to calculate Jaro-Winkler
                similarity. It is used only when ``second_sim`` is set to
                ``'jaro_winkler'``.
            **kwargs: Additional keyword arguments. They may override preceding
                default values.

        Returns:
            A soft-tfidf similarity function. It accepts two tokenized strings
            as inputs and returns a `float` as their similarity score weighted
            by the given weight.
        """
        if second_sim == 'jaro_winkler':
            sim_func = jaro_winkler.JaroWinkler(jw_prefix_weight).get_sim_score
        elif second_sim == 'jaro':
            sim_func = jaro.Jaro().get_sim_score
        elif second_sim == 'scaled_lev':
            sim_func = levenshtein.Levenshtein().get_sim_score
        soft_tfidf_obj = soft_tfidf.SoftTfIdf(
            corpus_list, sim_func,
            stfidf_threshold
        )

        def stfidf_sim(value1, value2):
            score1 = soft_tfidf_obj.get_raw_score(value1, value2)
            score2 = soft_tfidf_obj.get_raw_score(value2, value1)
            # No literature suggests using max here,
            # but if not used then the similarity is asymmetric
            # and is intuitively wrong for clustering
            return weight * max(score1, score2)
        return stfidf_sim

    @classmethod
    def produce_jaro_winkler(
        cls,
        weight: Union[float, int],
        corpus_list: Sequence[Sequence[str]],
        jw_prefix_weight: Union[float, int] = 0.1,
        **kwargs
    ) -> Callable[[str, str], float]:
        """ Returns a function to compute Jaro-Winkler similarity.

        Args:
            weight: The weight assigned to a particular reference attribute.
            corpus_list: A list of tokenized strings (a list of a list of
                words).
            jw_prefix_weight: The prefix weight used to calculate Jaro-Winkler
                similarity. It is used only when ``second_sim`` is set to
                ``'jaro_winkler'``.
            **kwargs: Additional keyword arguments. They may override preceding
                default values.

        Returns:
            A Jaro-Winkler similarity function. It accepts two strings as
            inputs and returns a `float` as their similarity score wighted by
            the given weight.
        """
        jaro_winkler_obj = jaro_winkler.JaroWinkler(jw_prefix_weight)

        def jaro_winkler_sim(value1, value2):
            return weight * jaro_winkler_obj.get_sim_score(value1, value2)
        return jaro_winkler_sim

    @classmethod
    def produce_jaro(
        cls,
        weight: Union[float, int],
        corpus_list: Sequence[Sequence[str]],
        **kwargs
    ) -> Callable[[str, str], float]:
        """ Returns a function to compute Jaro similarity.

        Args:
            weight: The weight assigned to a particular reference attribute.
            corpus_list: A list of tokenized strings (a list of a list of
                words).
            **kwargs: Additional keyword arguments. They may override preceding
                default values.

        Returns:
            A Jaro similarity function. It accepts two strings as inputs and
            returns a `float` as their similarity score wighted by the given
            weight.
        """
        jaro_obj = jaro.Jaro()

        def jaro_sim(value1, value2):
            return weight * jaro_obj.get_sim_score(value1, value2)
        return jaro_sim

    @classmethod
    def produce_jaccard_coef(
        cls, **kwargs
    ) -> Callable[[Collection, Collection, Callable[[Any], float]], float]:
        """ Returns a function to compute Jaccard coefficient.

        Returns:
            A Jaccard coefficient function. Its inputs are two collections of
            ids of neighboring clusters (of the two clusters one intend to
            compute similarity on) and a function which is not used here. It
            returns the similarity score as a `float`
        """
        def jaccard_coef(neighbors1, neighbors2, get_uniqueness):
            set1, set2 = neighbors1, neighbors2
            if type(neighbors1) is not set:
                set1 = set(neighbors1)
            if type(neighbors2) is not set:
                set2 = set(neighbors2)
            intersect = len(set1 & set2)
            union = len(set1 | set2)
            return intersect / union
        return jaccard_coef

    @classmethod
    def produce_jaccard_coef_fr(
        cls, **kwargs
    ) -> Callable[[Collection, Collection, Callable[[Any], float]], float]:
        """ Returns a function to compute Jaccard coefficient using multisets.

        Returns:
            An Jaccard coefficient function which counts neighbor
            multiplicities. Its inputs are two collections of ids of
            neighboring clusters (of the two clusters one intend to compute
            similarity on) and a function which is not used here. It returns
            the similarity score as a `float`
        """
        def jaccard_coef_fr(neighbors1, neighbors2, get_uniqueness):
            counter1 = collections.Counter(neighbors1)
            counter2 = collections.Counter(neighbors2)
            union, intersect = cls._union_intersect_counter(
                counter1, counter2,
                lambda curr, count, key: curr + count
            )
            return intersect / union
        return jaccard_coef_fr

    @classmethod
    def produce_adar_neighbor(
        cls, **kwargs
    ) -> Callable[[Collection, Collection, Callable[[Any], float]], float]:
        """ Returns a function to compute Adar similarity.

        Returns:
            An Adar similarity function. Its inputs are two collections of ids
            of neighboring clusters (of the two clusters one intend to
            compute similarity on) and a function which outputs the uniqueness
            of given a cluster id. It returns the similarity score as a
            `float`.
        """
        def adar_neighbor(neighbors1, neighbors2, get_uniqueness):
            set1, set2 = neighbors1, neighbors2
            if type(neighbors1) is not set:
                set1 = set(neighbors1)
            if type(neighbors2) is not set:
                set2 = set(neighbors2)
            union = set1 | set2
            inters = set1 & set2
            union_uniq = sum(get_uniqueness(cluster) for cluster in union)
            intersect_uniq = sum(get_uniqueness(cluster) for cluster in inters)
            return intersect_uniq / union_uniq
        return adar_neighbor

    @classmethod
    def produce_adar_neighbor_fr(
        cls, **kwargs
    ) -> Callable[[Collection, Collection, Callable[[Any], float]], float]:
        """ Returns a function to compute Adar similarity using multisets.

        Returns:
            An Adar similarity function which counts neighbor multiplicities.
            Its inputs are two collections of ids of neighboring clusters (of
            the two clusters one intend to compute similarity on) and a
            function which outputs the uniqueness of given a cluster id. It
            returns the similarity score as a `float`.
        """
        def adar_neighbor_fr(neighbors1, neighbors2, get_uniqueness):
            counter1 = collections.Counter(neighbors1)
            counter2 = collections.Counter(neighbors2)
            union_uniq, intersect_uniq = cls._union_intersect_counter(
                counter1, counter2,
                lambda curr, count, key: curr + count*get_uniqueness(key)
            )
            return intersect_uniq / union_uniq
        return adar_neighbor_fr

    @classmethod
    def produce_adar_attr(
        cls, **kwargs
    ) -> Callable[[Collection, Collection, Callable[[Any], float]], float]:
        """ Returns a function to compute customized Adar score.

        Returns:
            An Adar similarity function with customized ambiguity estimate
            based on attribute values. Its inputs are two collections of ids of
            neighboring clusters (of the two clusters one intend to compute
            similarity on) and a function which outputs the uniqueness of given
            a cluster id. It returns the similarity score as a `float`.
        """
        def adar_attr(neighbors1, neighbors2, get_uniqueness):
            set1, set2 = neighbors1, neighbors2
            if type(neighbors1) is not set:
                set1 = set(neighbors1)
            if type(neighbors2) is not set:
                set2 = set(neighbors2)
            union = set1 | set2
            inters = set1 & set2
            union_uniq = sum(get_uniqueness(cluster) for cluster in union)
            intersect_uniq = sum(get_uniqueness(cluster) for cluster in inters)
            return intersect_uniq / union_uniq
        return adar_attr

    @classmethod
    def produce_adar_attr_fr(
        cls, **kwargs
    ) -> Callable[[Collection, Collection, Callable[[Any], float]], float]:
        """ Returns a function to compute customized Adar score using multiset.

        Returns:
            A Adar similarity function with customized ambiguity estimate based
            on attribute values which also counts neighbor multiplicities. Its
            inputs are two collections of ids of neighboring clusters (of the
            two clusters one intend to compute similarity on) and a function
            which outputs the uniqueness of given a cluster id. It returns the
            similarity score as a `float`.
        """
        def adar_attr_fr(neighbors1, neighbors2, get_uniqueness):
            counter1 = collections.Counter(neighbors1)
            counter2 = collections.Counter(neighbors2)
            union_uniq, intersect_uniq = cls._union_intersect_counter(
                counter1, counter2,
                lambda curr, count, key: curr + count*get_uniqueness(key)
            )
            return intersect_uniq / union_uniq
        return adar_attr_fr

    @staticmethod
    def _union_intersect_counter(
        counter1: Counter,
        counter2: Counter,
        accumulator: Callable[[float, int, Hashable], float]
    ) -> Tuple[float, float]:
        """ Helper function to compute union and interset of two multisets.

        Args:
            counter1: Represents the multiset of cluster ids.
            counter2: Represents the multiset of cluster ids.
            accumulator: How one should increment the result at each iteration
                when looping through keys in the two counters. Its inputs are
                current accumulated result, current counter value (unioned/
                intersected), and the current counter key.

                Eg., ``lambda curr, count, key: curr + count`` means add the
                unioned/intersected count at each iteration step. Hence the
                final result is just the sum of all the union/intersected
                counts

        Note:
            The initial value for ``accumulator`` to accumulate upon is 0.

        Returns:
            (union result, intersect result) computed based on the rule
            specified by the ``accumulator`` argument.
        """
        union, intersect = 0., 0.
        for key in set(counter1.keys()) | set(counter2.keys()):
            count1 = counter1.get(key, 0)
            count2 = counter2.get(key, 0)
            union_count = max(count1, count2)
            intersect_count = min(count1, count2)
            union = accumulator(union, union_count, key)
            intersect = accumulator(intersect, intersect_count, key)
        return union, intersect


class ClusteringMetrics:
    """ A collection of metrics to measure performance of clustering.

    This class consists only of class methods.

    For detailed algorithms behind each of these clustering metric functions,
    please refer to :doc:`../advanced_guide`.

    Attributes:
        _logger (`~typing.Logger`): Used to log information regarding these
            clustering metric computations.
    """

    _logger = logging.getLogger('Evaluator')

    @classmethod
    def precision_recall(
        cls,
        labels: Mapping,
        preds: collections.OrderedDict,
        log: bool = True,
        **kwargs
    ) -> Tuple[float, float, float]:
        """ Compute the precision, recall, and f1.

        Args:
            labels: Mapping reference ids to cluster ids. The cluster ids may
                be different from those in the ground truth.
            preds: Mapping reference ids to cluster ids. The dictionary is
                sorted (key-value pairs are inserted) in ascending order of
                node id.
            log: Whether to log information when it is called.

        Returns:
            (precision, recall, f1). This is computed using pairwise measure.
        """
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
    def v_measure(
        cls,
        labels: Mapping,
        preds: collections.OrderedDict,
        **kwargs
    ) -> float:
        """ Compute the V-measure score.

        It simply wraps the scikit-learn V-measure function.

        Args:
            labels: Mapping reference ids to cluster ids. The cluster ids may
                be different from those in the ground truth.
            preds: Mapping reference ids to cluster ids. The dictionary is
                sorted (key-value pairs are inserted) in ascending order of
                node id.

        Returns:
            The V-measure score.

        See Also:
            `V-measure explained <https://scikit-learn.org/stable/modules/\
                clustering.html#homogeneity-completeness-and-v-measure>`_ by
            scikit-learn.
        """
        score = metrics.v_measure_score(labels, preds)
        cls._logger.info(f'V-measure score: {score}')
        return score

    @classmethod
    def ami(
        cls,
        labels: Mapping,
        preds: collections.OrderedDict,
        average_method: str = 'max',
        **kwargs
    ) -> float:
        """ Compute the adjusted mutual information.

        It simply wraps the scikit-learn AMI function.

        Args:
            labels: Mapping reference ids to cluster ids. The cluster ids may
                be different from those in the ground truth.
            preds: Mapping reference ids to cluster ids. The dictionary is
                sorted (key-value pairs are inserted) in ascending order of
                node id.
            average_method: Indicate how to compute adjusted mutual
                information. Valid values are ``'min'``, ``'geometric'``,
                ``'arithmetic'``, and ``'max'``.

        Returns:
            The adjusted mutual information value.

        See Also:
            `AMI explained <https://scikit-learn.org/stable/modules/\
                clustering.html#mutual-information-based-scores>`_ by
            scikit-learn.
        """
        cls._logger.debug(f'average_method: {average_method}')
        score = metrics.adjusted_mutual_info_score(
            labels, preds,
            average_method=average_method
        )
        cls._logger.info(f'Adjusted mutual information: {score}')
        return score


class DSU:
    """ Implement the DSU data structure for hashable objects.

    It implements path compression and union by rank to achieve optimal time
    complexity.

    Args:
        items: The list of items to build the DSU data structure from.

    Attributes:
        items (`~typing.Dict`\ [`~typing.Hashable`, `~typing.Hashable`]): Use
            the items in the ``items`` argument as keys and themselves as
            values. Created during object instantiation.
        rank (`~typing.Dict`\ [`~typing.Hashable`, `int`]): Store the rank
            of each disjoint set.
    """

    def __init__(self, items: Iterable[Hashable]):
        self.items = {item: item for item in items}
        self.rank = {item: 1 for item in items}

    def union(self, item1: Hashable, item2: Hashable) -> None:
        """ Merge the sets respectively containing the two items into one.

        Args:
            item1: One of the two items to be merged.
            item2: One of the two items to be merged.
        """
        parent1, parent2 = self.find(item1), self.find(item2)
        if self.rank[parent1] < self.rank[parent2]:
            self.items[parent1] = parent2
            self.rank[parent2] += self.rank[parent1]
        else:
            self.items[parent2] = parent1
            self.rank[parent1] += self.rank[parent2]

    def find(self, item: Hashable) -> Hashable:
        """ Find the item representing the set containing the input ``item``.

        Args:
            item: Intend to find the set containing this item.

        Returns:
            The item representing the set containing the input.
        """
        parent = self.items[item]
        if parent == item:
            return item
        res = self.find(parent)
        self.items[item] = res
        return res


class SimilarityEntry:
    """ Stores information about two clusters and their similarity.

    Comparisons between ``SimilarityEntry`` objects are achieved by
    implementing all comparison magical methods. This helps with implementation
    of the `PriorityQueue` object. Since the `PriorityQueue` class is
    implmented as a min heap while the clustering algorithm needs to pop out
    the entry with the largest similarity first, the comparison operators are
    implemented as the reverse of the similarities comparisons.

    Args:
        cluster1: One of the two clusters between which the similarity is
            computed.
        cluster2: One of the two clusters between which the similarity is
            computed.
        similarity: The similarity between ``cluster1`` and ``cluster2``.

    Attributes:
        clusters (`~typing.Tuple`\ [`~typing.Any`, `~typing.Any`]): The tuple
            (``cluster1``, ``cluster2``) initialized during object
            instantiation.
        similarity (`float`): Same as in the parameters section.
        index (`int`): Store the index of the entry in the `PriorityQueue`
            object. Used to implement its update and remove methods. It is set
            to -1 if not contained in the priority queue.
    """

    def __init__(self, cluster1: Any, cluster2: Any, similarity: float):
        self.cluster1 = cluster1
        self.cluster2 = cluster2
        self.similarity = similarity
        self.index = -1

    def __eq__(self, other: 'SimilarityEntry') -> bool:
        """ Implement the equality operator.

        Args:
            other: Another entry to compare with.

        Returns:
            If this object is equal to ``other``.
        """
        return self.similarity == other.similarity

    def __ne__(self, other: 'SimilarityEntry') -> bool:
        """ Implement the inequality operator.

        Args:
            other: Another entry to compare with.

        Returns:
            If this object is equal to ``other``.
        """
        return self.similarity != other.similarity

    def __lt__(self, other: 'SimilarityEntry') -> bool:
        """ Implement the less than operator.

        Args:
            other: Another entry to compare with.

        Returns:
            If this object is less than to ``other``.
        """
        return self.similarity > other.similarity

    def __le__(self, other: 'SimilarityEntry') -> bool:
        """ Implement the less than or equal to operator.

        Args:
            other: Another entry to compare with.

        Returns:
            If this object is less than or equal to ``other``.
        """
        return self.similarity >= other.similarity

    def __gt__(self, other: 'SimilarityEntry') -> bool:
        """ Implement the greater than operator.

        Args:
            other: Another object to compare with.

        Returns:
            If this object is greater than ``other``.
        """
        return self.similarity < other.similarity

    def __ge__(self, other: 'SimilarityEntry') -> bool:
        """ Implement the greater than or equal to operator.

        Args:
            other: Another object to compare with.

        Returns:
            If this object is greater or equal to than ``other``.
        """
        return self.similarity <= other.similarity


class PriorityQueue:
    """ The priority queue data stucture for `SimilarityEntry` objects.

    Args:
        items: The list of `SimilarityEntry` objects to be put in the priority
            queue.

    Attributes:
        _queue (List[SimilarityEntry]): The inner min heap data structure
            containing all the `SimilarityEntry` objects. It copies the input
            ``items`` argument during object instantiation and hence will never
            modify the original input.

    Important:
        Only use the ``remove`` method if the item is guaranteed to exist in
        the priority queue. Otherwise use the ``discard`` method. Failure to do
        so might result in unexpected behavior.

    Note:
        This class uses a **min** heap to implement the priority queue.
    """

    def __init__(self, items: Iterable[SimilarityEntry] = []):
        self._queue = list(items)
        self._heapify()

    def __len__(self) -> int:
        """ Implement the `len` operator using length of the min heap.

        Returns:
            The length of the priority queue, which is the same as the length
            of the min heap.
        """
        return len(self._queue)

    def push(self, item: SimilarityEntry) -> None:
        """ Push an item into the priority queue.

        Args:
            item: The item to be added to the priority queue.
        """
        self._queue.append(item)
        index = len(self._queue) - 1
        item.index = index
        self._siftdown(0, index)

    def pop(self) -> SimilarityEntry:
        """ Pop the smallest item from the priority queue and return it.

        Returns:
            The smallest item in the priority queue.
        """
        return self.remove(self._queue[0])

    def discard(self, item: SimilarityEntry) -> Optional[SimilarityEntry]:
        """ Remove an item from the priority queue if it is in the queue.

        Returns:
            If the input ``item`` is in the priority queue, this item is
            returned. Otherwise return ``None``.
        """
        if item.index >= 0:
            return self.remove(item)

    def remove(self, item: SimilarityEntry) -> SimilarityEntry:
        """ Remove an item contained in the priority queue from the queue.

        Args:
            item: The item to remove from the priority queue. One must ensure
                that it is contained in the priority queue. Otherwise use the
                ``discard`` method instead.

        Returns:
            The removed item.
        """
        item_index = item.index
        last = self._queue.pop()
        if item_index < len(self._queue):
            self._queue[item_index] = last
            self._queue[item_index].index = item_index
            self._siftdown(0, item_index)
            self._siftup(item_index)
        item.index = -1
        return item

    def update(
        self, item: SimilarityEntry, new_item: SimilarityEntry
    ) -> None:
        """ Replace an item in the priority queue with another.

        Args:
            item: The item in the priority queue to be replaced. One must
                ensure that this item is contained in the priority queue.
            new_item: The item to replace the original item in the priority
                queue.

        Note:
            The method does **not** return anything.
        """
        item_index = item.index
        self._queue[item_index] = new_item
        self._queue[item_index].index = item_index
        item.index = -1
        self._siftdown(0, item_index)
        self._siftup(item_index)

    def _heapify(self) -> None:
        """ Alter the ``_queue`` attribute into a min heap in O(n) time.

        This method is only used once during object instantiation since the
        input is not supposed to be a min heap already.
        """
        length = len(self._queue)
        for i in reversed(range(length)):
            self._queue[i].index = i
            if i < length // 2:
                self._siftup(i)

    def _siftup(self, pos: int) -> None:
        """ Push up the object at the given index in the min heap to leaves.

        Args:
            pos: The index of the object to be pushed up.
        """
        end_pos = len(self._queue)
        start_pos = pos
        new_item = self._queue[pos]
        child_pos = 2*pos + 1
        while child_pos < end_pos:
            right_pos = child_pos + 1
            if (
                right_pos < end_pos
                and not self._queue[child_pos] < self._queue[right_pos]
            ):
                child_pos = right_pos
            self._queue[pos] = self._queue[child_pos]
            self._queue[pos].index = pos
            pos = child_pos
            child_pos = 2*pos + 1
        self._queue[pos] = new_item
        self._queue[pos].index = pos
        self._siftdown(start_pos, pos)

    def _siftdown(self, start_pos: int, pos: int) -> None:
        """ Push down the object at the given index to root till a threshold.

        Args:
            start_pos: The object to be pushed down stops if its index is
                already less than or equal to this value.
            pos: The index of the object to be pushed down.
        """
        new_item = self._queue[pos]
        while pos > start_pos:
            parent_pos = (pos - 1) >> 1
            parent = self._queue[parent_pos]
            if new_item < parent:
                self._queue[pos] = parent
                self._queue[pos].index = pos
                pos = parent_pos
                continue
            break
        self._queue[pos] = new_item
        self._queue[pos].index = pos
