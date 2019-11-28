import json
import time
import collections
from ..core.utils import WithLogger


class GroundTruthParser(WithLogger):

    def parse(self, data_path):
        start_time = time.time()
        ground_truth = collections.OrderedDict()
        with open(data_path, 'r') as f:
            data = json.load(f)
            for row in sorted(data, key=lambda row: row['node_id']):
                ground_truth[row['node_id']] = row['cluster_id']
        end_time = time.time()
        time_taken = end_time - start_time
        self._logger.debug(f'Time taken to parse ground truth: {time_taken}s')
        num_refs = len(ground_truth)
        num_clts = len(set(ground_truth.values()))
        self._logger.info(f'Number of references in ground truth: {num_refs}')
        self._logger.info(f'Number of clusters in ground truth: {num_clts}')
        return ground_truth
