import os
from entity_resolver.scripts import run
import json

# class TestArxiv:
#
#     def test_dummy(self):
#         file_path = 'file.txt'
#         run(['norm-arxiv', '', file_path])
#         with open(file_path, 'r') as f:
#             assert f.read() == 'arxiv'
#         os.remove(file_path)


class TestCiteseer:

    # def test_dummy(self):
    #     file_path = 'file.txt'
    #     run(['norm-citeseer', '', file_path])
    #     with open(file_path, 'r') as f:
    #         assert f.read() == 'citeseer'
    #     os.remove(file_path)
    def test_citeseer(self):
        file_path ='/Users/xinxinhuang/Desktop/EntityResolution_Capstone/data/citeseer/citeseer-mrdm05.dat'
        output_path = 'testdata.json'
        run(['norm-citeseer', file_path, output_path])
        with open(output_path, 'r') as f:
            distros_dict = json.load(f)
        print(distros_dict[:100])


