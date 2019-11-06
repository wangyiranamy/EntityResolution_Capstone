import argparse
import json
from functools import wraps


def _subparser(subcommand, subcommand_help, *args):
    def subparser_dec(f):
        def gen_args(parsed_args):
            for *names, arg_help in args:
                canon_name = names[-1]
                if canon_name[:2] == '--':
                    canon_name = canon_name[2:]
                yield getattr(parsed_args, canon_name)

        @wraps(f)
        def create_subparser(subparsers):
            parser = subparsers.add_parser(subcommand, help=subcommand_help)
            for *names, arg_help in args:
                parser.add_argument(*names, help=arg_help)
            parser.set_defaults(
                func=lambda parsed_args: f(*gen_args(parsed_args))
            )
        return create_subparser
    return subparser_dec


@_subparser(
    'norm-arxiv',
    'Transform the arxiv data into the format expected by EntityResolver',
    ('input', 'The path of the arxiv data to be transformed'),
    ('graph_path', 'The path of the transformed arxiv graph data'),
    (
        'ground_truth_path',
        'The path of the transformed arxiv ground truth data'
    )
)
def _norm_arxiv(input_path, graph_path, ground_truth_path):
    parse_data(input_path, graph_path, ground_truth_path, 'arxiv')


@_subparser(
    'norm-citeseer',
    'Transform the citeseer data into the format expected by EntityResolver',
    ('input', 'The input file path of the citeseer data to be transformed'),
    ('graph_path', 'The path of the transformed citeseer graph data'),
    (
        'ground_truth_path',
        'The path of the transformed citeseer ground truth data'
    )
)
def _norm_citeseer(input_path, graph_path, ground_truth_path):
    parse_data(input_path, graph_path, ground_truth_path, 'citeseer')


def parse_data(input_path, graph_path, ground_truth_path, name):
    graph, ground_truth = list(), list()
    with open(input_path) as dat_file:
        for line in dat_file:
            row = [field for field in line.split('|', 7)]
            # Multiple different rows with id 2716 are present
            if name == 'citeseer' and row[0] != '2716 ':
                node_id = int(row[0])
                attr = {'title': row[7], 'name': row[3]}
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
                graph.append(graph_row)
                ground_truth.append(ground_truth_row)

    with open(graph_path, 'w') as f:
        json.dump(graph, f)
    with open(ground_truth_path, 'w') as f:
        json.dump(ground_truth, f)


def run(args=None):
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers()
    _norm_arxiv(subparsers)
    _norm_citeseer(subparsers)
    parsed_args = parser.parse_args(args)
    parsed_args.func(parsed_args)


def main():
    run()
