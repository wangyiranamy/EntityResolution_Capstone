import argparse
import json
from .core.utils import subparser


@subparser(
    'norm-arxiv',
    'Transform the arxiv data into the format expected by EntityResolver',
    'The path of the arxiv data to be transformed',
    'The path of the transformed arxiv graph data',
    'The path of the transformed arxiv ground truth data'
)
def _norm_arxiv(
    data='data/arxiv/arxiv-mrdm05.dat',
    graph='graph.json',
    ground_truth='ground_truth.json'
):
    parse_data(data, graph, ground_truth, 'arxiv')


@subparser(
    'norm-citeseer',
    'Transform the citeseer data into the format expected by EntityResolver',
    'The input file path of the citeseer data to be transformed',
    'The path of the transformed citeseer graph data',
    'The path of the transformed citeseer ground truth data'
)
def _norm_citeseer(
    data='data/citeseer/citeseer-mrdm05.dat',
    graph='graph.json',
    ground_truth='ground_truth.json'
):
    parse_data(data, graph, ground_truth, 'citeseer')


def parse_data(data, graph, ground_truth, name):
    graph_data, ground_truth_data = list(), list()
    with open(data) as dat_file:
        for line in dat_file:
            row = [field for field in line.split('|', 7)]
            # Multiple different rows with id 2716 are present
            if name == 'citeseer' and row[0] == '2716 ':
                continue
            node_id = int(row[0])
            attr = {'name': row[2].strip()}
            cluster_id = int(row[1])
            edge_id = int(row[5])
            graph_row = {
                'node_id': node_id,
                'edge_id': edge_id,
                'attr_dict': attr
            }
            ground_truth_row = {
                'node_id': node_id,
                'cluster_id': cluster_id
            }
            graph_data.append(graph_row)
            ground_truth_data.append(ground_truth_row)

    with open(graph, 'w') as f:
        json.dump(graph_data, f)
    with open(ground_truth, 'w') as f:
        json.dump(ground_truth_data, f)


def run(args=None):
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers()
    _norm_arxiv(subparsers)
    _norm_citeseer(subparsers)
    parsed_args = parser.parse_args(args)
    parsed_args.func(parsed_args)


def main():
    run()
