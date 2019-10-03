from entity_resolver.parser import GraphParser, GroundTruthParser


class TestGraphParser:

    def test_dummy(self):
        graph_parser = GraphParser({'title': 'string', 'name': 'string'})
        assert 'parse' in dir(graph_parser)
        graph = graph_parser.parse('./data/test_data/data.json')
        assert len(graph.nodes) == 4
        assert len(graph.edges) == 2
        assert len(graph.get_neighbors(10)) == 3

class TestGroundTruthParser:

    def test_dummy(self):
        ground_truth_parser = GroundTruthParser()
        assert 'parse' in dir(ground_truth_parser)
