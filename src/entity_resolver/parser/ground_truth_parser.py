""" Contains only one parser class to be used by the main module.

Example:

    >>> from entity_resolver.parser import GroundTruthParser
    >>> parser = GroundTruthParser()
    >>> ground_truth = parser.parse('ground truth.json')
"""

import json
import time
import collections
from typing import Dict
from ..core.utils import WithLogger


class GroundTruthParser(WithLogger):
    """ A parser class to parse ground truth data file.

    Currently this class only contains one method and no attributes, so it
    is essentially serving as a dummy wrapper class of its parse method.
    """
    def parse(self, data_path: str) -> collections.OrderedDict:
        """ Parse the ground truth data into a dictionary.

        Args:
            data_path: The path to the ground truth data. The data file has to
                **strictly follow** the format as described in
                :doc:`../quickstart`.

        Returns:
            Mapping reference ids to ground truth cluster ids. The dictionary
            is sorted (key-value pairs are inserted) in ascending order of
            reference ids.
        """
        start_time = time.time()
        ground_truth = collections.OrderedDict()
        with open(data_path, 'r') as f:
            data = json.load(f)
            for row in sorted(data, key=lambda row: row['node_id']):
                ground_truth[row['node_id']] = row['cluster_id']
        end_time = time.time()
        time_taken = end_time - start_time
        self.logger.debug(f'Time taken to parse ground truth: {time_taken}s')
        num_refs = len(ground_truth)
        num_clts = len(set(ground_truth.values()))
        self.logger.info(f'Number of references in ground truth: {num_refs}')
        self.logger.info(f'Number of clusters in ground truth: {num_clts}')
        return ground_truth
