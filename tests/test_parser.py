from entity_resolver.parser import GraphParser, GroundTruthParser


class TestGraphParser:

    def test_graph_parser(self):
        graph_parser = GraphParser({'title': 'text', 'name': 'person_entity'})
        assert 'parse' in dir(graph_parser)
        graph = graph_parser.parse('./data/test_data/data.json')
        print ('parser works')
        print(graph.nodes[0].attrs)
        print(graph.get_attr_names())
        print(graph.get_attr_val())
        # assert len(graph.nodes) == 4
        # assert len(graph.edges) == 2
        # assert len(graph.get_neighbors(graph.nodes[0])) == 3

class TestGroundTruthParser:

    def test_dummy(self):
        ground_truth_parser = GroundTruthParser()
        assert 'parse' in dir(ground_truth_parser)
