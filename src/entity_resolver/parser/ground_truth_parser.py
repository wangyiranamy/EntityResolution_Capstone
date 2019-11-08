import json


class GroundTruthParser:

    def parse(self, data_path):
        with open(data_path, 'r') as f:
            data = json.load(f)
            ground_truth = {row['node_id']: row['cluster_id'] for row in data}
        return ground_truth
