from .core import Resolver, Evaluator
from .parser import GraphParser, GroundTruthParser


class EntityResolver:

    def __init__(
        self, resolver=None, evaluator=None,
        graph_parser=None, ground_truth_parser=None,
        attr_types={}
    ):
        if resolver is None:
            self.resolver = Resolver()
        else:
            self.resolver = resolver
        if evaluator is None:
            self.evaluator = Evaluator()
        else:
            self.evaluator = evaluator
        if graph_parser is None:
            self.graph_parser = GraphParser(attr_types)
        else:
            self.graph_parser = graph_parser
        if ground_truth_parser is None:
            self.ground_truth_parser = GroundTruthParser()
        else:
            self.ground_truth_parser = ground_truth_parser
        pass

    def resolve(self, graph_path):
        graph = self.graph_parser.parse(graph_path)
        return self.resolver.resolve(graph)

    def evaluate(self, resolved_mapping, ground_truth_path):
        ground_truth = self.ground_truth_parser.parse(ground_truth_path)
        return self.evaluator.evaluate(resolved_mapping, ground_truth)

    def resolve_and_eval(self, graph_path, ground_truth_path):
        resolved_mapping = self.resolve(graph_path)
        return self.evaluate(resolved_mapping, ground_truth_path)
