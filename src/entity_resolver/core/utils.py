import time
import collections
import inspect
import logging
from functools import wraps, partial
import numpy as np
from sklearn import metrics
from py_stringmatching.similarity_measure import jaro_winkler, soft_tfidf, jaro


class WithLogger:

    def __init__(self, verbose=0):
        self.verbose = verbose
        self._logger = logging.getLogger(self.__class__.__name__)
        self._config_logger()

    @property
    def logger(self):
        return self._logger

    def _config_logger(self):
        if self.verbose <= 0:
            level = logging.WARNING
        elif self.verbose == 1:
            level = logging.INFO
        else:
            level = logging.DEBUG
        handler = logging.StreamHandler()
        fmt = '[{asctime}] {levelname} {name}: {msg}'
        formatter = logging.Formatter(fmt=fmt, style='{')
        handler.setFormatter(formatter)
        self._logger.addHandler(handler)
        self._logger.setLevel(level)


def timeit(func):
    @wraps(func)
    def timed_func(obj, *args, **kwargs):
        start_time = time.time()
        res = func(obj, *args, **kwargs)
        end_time = time.time()
        time_list = obj.time_dict[func.__name__]
        time_list[0] += (end_time - start_time)
        time_list[1] += 1
        return res
    return timed_func


class subparser:

    def __init__(self, subcommand, subcommand_help, *helps):
        self.subcommand = subcommand
        self.subcommand_help = subcommand_help
        self.helps = helps

    def __call__(self, func):
        return wraps(func)(partial(self._create_subparser, func))

    def _create_subparser(self, func, subparsers):
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

    def _parse_func(self, func):
        sig = inspect.signature(func)
        func_info = list()
        for arg_help, (name, param) in zip(self.helps, sig.parameters.items()):
            func_info.append((name, param.default, arg_help))
        return func_info

    def _parser_function(self, func, func_info, args):
        func_args = list()
        for name, _, _ in func_info:
            func_args.append(getattr(args, name))
        return func(*func_args)


class logtime:

    def __init__(self, header):
        self.header = header

    def __call__(self, func):
        return wraps(func)(
            lambda obj, *args, **kwargs:
                self._timed_func(func, obj, *args, **kwargs)
        )

    def _timed_func(self, func, obj, *args, **kwargs):
        start_time = time.time()
        res = func(obj, *args, **kwargs)
        end_time = time.time()
        time_taken = round(end_time - start_time, 2)
        obj.logger.info(f'{self.header}: {time_taken}s')
        return res


class SimFuncFactory:

    @classmethod
    def produce_stfidf_jaro_winkler(
        cls, weight, corpus_list, soft_tfidf_threshold=0.5,
        jw_prefix_weight=0.15, **kwargs
    ):
        sim_func = jaro_winkler.JaroWinkler().get_sim_score
        soft_tfidf_obj = soft_tfidf.SoftTfIdf(
            corpus_list, sim_func,
            soft_tfidf_threshold
        )

        def stfidf_jaro_winkler_sim(value1, value2):
            score1 = soft_tfidf_obj.get_raw_score(value1, value2)
            score2 = soft_tfidf_obj.get_raw_score(value2, value1)
            return weight * max(score1, score2)
        return stfidf_jaro_winkler_sim

    @classmethod
    def produce_jaro_winkler(
        cls, weight, corpus_list,
        jw_prefix_weight=0.15, **kwargs
    ):
        def jaro_winkler_sim(value1, value2):
            jaro_winkler_obj = jaro_winkler.JaroWinkler(jw_prefix_weight)
            return weight * jaro_winkler_obj.get_sim_score(value1, value2)
        return jaro_winkler_sim

    @classmethod
    def produce_jaro(cls, weight, corpus_list, **kwargs):
        def jaro_sim(value1, value2):
            jaro_obj = jaro.Jaro()
            return weight * jaro_obj.get_sim_score(value1, value2)
        return jaro_sim

    @classmethod
    def produce_jaccard_coef(cls, **kwargs):
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
    def produce_jaccard_coef_fr(cls, **kwargs):
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
    def produce_adar_neighbor(cls, **kwargs):
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
    def produce_adar_neighbor_fr(cls, **kwargs):
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
    def produce_adar_attr(cls, **kwargs):
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
    def produce_adar_attr_fr(cls, **kwargs):
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
    def _union_intersect_counter(counter1, counter2, accumulator, initial=0):
        union, intersect = initial, initial
        for key in set(counter1.keys()) | set(counter2.keys()):
            count1 = counter1.get(key, 0)
            count2 = counter2.get(key, 0)
            union_count = max(count1, count2)
            intersect_count = min(count1, count2)
            union = accumulator(union, union_count, key)
            intersect = accumulator(intersect, intersect_count, key)
        return union, intersect


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


class DSU:

    def __init__(self, items):
        self.items = {item: item for item in items}
        self.rank = {item: 1 for item in items}

    def union(self, item1, item2):
        parent1, parent2 = self.find(item1), self.find(item2)
        if self.rank[parent1] < self.rank[parent2]:
            self.items[parent1] = parent2
            self.rank[parent2] += self.rank[parent1]
        else:
            self.items[parent2] = parent1
            self.rank[parent1] += self.rank[parent2]

    def find(self, item):
        parent = self.items[item]
        if parent == item:
            return item
        res = self.find(parent)
        self.items[item] = res
        return res


class PriorityQueue:

    def __init__(self, items=[]):
        self._queue = list(items)
        self._heapify()

    def __len__(self):
        return len(self._queue)

    def push(self, item):
        self._queue.append(item)
        index = len(self._queue) - 1
        item.index = index
        self._siftdown(0, index)

    def pop(self):
        return self.remove(self._queue[0])

    def discard(self, item):
        if item.index >= 0:
            return self.remove(item)

    def remove(self, item):
        item_index = item.index
        last = self._queue.pop()
        if item_index < len(self._queue):
            self._queue[item_index] = last
            self._queue[item_index].index = item_index
            self._siftdown(0, item_index)
            self._siftup(item_index)
        item.index = -1
        return item

    def update(self, item, new_item):
        item_index = item.index
        self._queue[item_index] = new_item
        self._queue[item_index].index = item_index
        item.index = -1
        self._siftdown(0, item_index)
        self._siftup(item_index)

    def _heapify(self):
        length = len(self._queue)
        for i in reversed(range(length)):
            self._queue[i].index = i
            if i < length // 2:
                self._siftup(i)

    def _siftup(self, pos):
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

    def _siftdown(self, start_pos, pos):
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


class SimilarityEntry:

    def __init__(self, cluster1, cluster2, similarity):
        self.clusters = (cluster1, cluster2)
        self.similarity = similarity
        self.index = -1

    def __eq__(self, other):
        if type(self) is type(other):
            return self.clusters == other.clusters
        return NotImplemented

    def __ne__(self, other):
        if type(self) is type(other):
            return self.clusters != other.clusters
        return NotImplemented

    def __lt__(self, other):
        if type(self) is type(other):
            return self.similarity > other.similarity
        return NotImplemented

    def __le__(self, other):
        if type(self) is type(other):
            return self.similarity >= other.similarity
        return NotImplemented

    def __gt__(self, other):
        if type(self) is type(other):
            return self.similarity < other.similarity
        return NotImplemented

    def __ge__(self, other):
        if type(self) is type(other):
            return self.similarity <= other.similarity
        return NotImplemented
