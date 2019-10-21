import os
import json
from entity_resolver.scripts import run


class TestScripts:

    def test_citeseer(self):
        file_path = 'data/citeseer/citeseer-mrdm05.dat'
        output_path = 'testdata.json'
        run(['norm-citeseer', file_path, output_path])
        with open(output_path, 'r') as f:
            test_dict = json.load(f)

        assert test_dict[0]['attr_dict']['name'] == '  A. Aamodt '
        assert test_dict[0]['node_id'] == 0
        assert test_dict[0]['edge_id'] == 1019

    def test_arxiv(self):
        file_path = 'data/arxiv/arxiv-mrdm05.dat'
        output_path = 'testdata.json'
        run(['norm-citeseer', file_path, output_path])
        with open(output_path, 'r') as f:
            test_dict = json.load(f)
        assert test_dict[0]['attr_dict']['name'] == ' c.itzykson '
        assert test_dict[0]['node_id'] == 0
        assert test_dict[0]['edge_id'] == 2
