
import time
import functools
import collections
from py_stringmatching.similarity_measure import jaro_winkler, soft_tfidf, jaro


def timeit(f):
    @functools.wraps(f)
    def timed_f(obj, *args, **kwargs):
        start_time = time.time()
        res = f(obj, *args, **kwargs)
        end_time = time.time()
        time_list = obj._time_dict[f.__name__]
        time_list[0] += (end_time - start_time)
        time_list[1] += 1
        return res
    return timed_f


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
        self._indicies = {item: index for index, item in enumerate(items)}
        self._heapify()

    def __len__(self):
        return len(self._queue)

    def _heapify(self):
        for i in range(len(self._queue) - 1, -1, -1):
            self._shiftdown(i)

    def _shiftup(self, pos):
        while pos > 0:
            parent_pos = ((pos+1) >> 1)-1
            if self._queue[pos] < self._queue[parent_pos]:
                self._swap(pos, parent_pos)
                pos = parent_pos
            else:
                break

    def _shiftdown(self, pos):
        while pos < len(self._queue):
            left_pos = ((pos+1) << 1) - 1
            right_pos = ((pos+1) << 1)
            curr = self._queue[pos]
            left, right = None, None
            if left_pos < len(self._queue):
                left = self._queue[left_pos]
            if right_pos < len(self._queue):
                right = self._queue[right_pos]
            if left is None:
                break
            if right is None and left >= curr:
                break
            if (
                (right is None and left < curr)
                or (left < curr and left <= right)
            ):
                self._swap(pos, left_pos)
                pos = left_pos
            elif right < curr and right <= left:
                self._swap(pos, right_pos)
                pos = right_pos
            else:
                break

    def _swap(self, pos1, pos2):
        item1, item2 = self._queue[pos1], self._queue[pos2]
        self._queue[pos1] = item2
        self._queue[pos2] = item1
        self._indicies[item1] = pos2
        self._indicies[item2] = pos1

    def push(self, item):
        self._queue.append(item)
        index = len(self._queue) - 1
        self._indicies[item] = index
        self._shiftup(index)

    def pop(self):
        return self.remove(self._queue[0])

    def remove(self, item):
        item_index = self._indicies[item]
        self._swap(item_index, len(self._queue) - 1)
        self._queue.pop()
        self._indicies.pop(item)
        if item_index < len(self._queue):
            self._shiftdown(item_index)
            self._shiftup(item_index)
        return item

    def discard(self, item):
        if item in self._indicies:
            return self.remove(item)

    def update(self, item, new_item):
        item_index = self._indicies.pop(item)
        self._queue[item_index] = new_item
        self._indicies[new_item] = item_index
        self._shiftdown(item_index)
        self._shiftup(item_index)


class SimilarityEntry:

    def __init__(self, cluster1, cluster2, similarity):
        self.clusters = (cluster1, cluster2)
        self.similarity = similarity

    def __hash__(self):
        return hash(self.clusters)

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
