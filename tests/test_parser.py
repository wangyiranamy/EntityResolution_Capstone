from entity_resolver.parser import GraphParser, GroundTruthParser


class TestGraphParser:

    def test_dummy(self):
        graph_parser = GraphParser({})
        assert 'parse' in dir(graph_parser)


class TestGroundTruthParser:

    def test_dummy(self):
        ground_truth_parser = GroundTruthParser()
        assert 'parse' in dir(ground_truth_parser)
