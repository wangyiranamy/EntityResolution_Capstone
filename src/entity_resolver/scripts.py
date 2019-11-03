import argparse
import json
from functools import wraps


def _subparser(subcommand, subcommand_help, *args):
    def subparser_dec(f):
        def gen_args(parsed_args):
            for *names, arg_help in args:
                cannon_name = names[-1]
                if cannon_name[:2] == '--':
                    cannon_name = cannon_name[2:]
                yield getattr(parsed_args, cannon_name)

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
    ('output_graph', 'The path of the transformed arxiv graph data'),
    ('output_ground_truth', 'The path of the transformed arxiv ground truth')
)

def _norm_arxiv(input_path, output_path):
    parse_data(input_path, output_path)


@_subparser(
    'norm-citeseer',
    'Transform the citeseer data into the format expected by EntityResolver',
    ('input', 'The input file path of the citeseer data to be transformed'),
    ('output', 'The output file path of the transformed citeseer data')
)
def _norm_citeseer(input_path, output_path):
    parse_data(input_path, output_path)


def parse_data(input_path, output_path):
    with open(input_path) as dat_file:
        data = []
        for line in dat_file:
            row = [field for field in line.split('|')]
            if (len(row) == 8):
                attr = {'title': row[7], 'name': row[3]}
                row_dic = {
                    'node_id': int(row[0]),
                    'edge_id': int(row[5]),
                    'attr_dict': attr
                }
                data.append(row_dic)

    with open(output_path, 'w') as f:
        json.dump(data, f)


def run(args=None):
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers()
    _norm_arxiv(subparsers)
    _norm_citeseer(subparsers)
    parsed_args = parser.parse_args(args)
    parsed_args.func(parsed_args)


def main():
    run()
