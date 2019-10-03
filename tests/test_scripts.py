import os
from entity_resolver.scripts import run


class TestArxiv:

    def test_dummy(self):
        file_path = 'file.txt'
        run(['norm-arxiv', '', file_path])
        with open(file_path, 'r') as f:
            assert f.read() == 'arxiv'
        os.remove(file_path)


class TestCiteseer:

    def test_dummy(self):
        file_path = 'file.txt'
        run(['norm-citeseer', '', file_path])
        with open(file_path, 'r') as f:
            assert f.read() == 'citeseer'
        os.remove(file_path)
