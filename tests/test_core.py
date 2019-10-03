from entity_resolver.core import Evaluator, Graph, Resolver


class TestEvaluator:

    def test_dummy(self):
        evaluator = Evaluator()
        assert 'evaluate' in dir(evaluator)


class TestGraph:

    def test_dummy(self):
        graph = Graph([], [])
        assert 'get_neighbors' in dir(graph)


class TestResolver:

    def test_dummy(self):
        resolver = Resolver()
        assert 'resolve' in dir(resolver)
