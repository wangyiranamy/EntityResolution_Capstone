import argparse
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
    ('output', 'The path of the transformed arxiv data')
)
def _norm_arxiv(input_path, output_path):
    with open(output_path, 'w') as f:
        f.write('arxiv')


@_subparser(
    'norm-citeseer',
    'Transform the citeseer data into the format expected by EntityResolver',
    ('input', 'The input file path of the citeseer data to be transformed'),
    ('output', 'The output file path of the transformed citeseer data')
)
def _norm_citeseer(input_path, output_path):
    with open(output_path, 'w') as f:
        f.write('citeseer')


def run(args=None):
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers()
    _norm_arxiv(subparsers)
    _norm_citeseer(subparsers)
    parsed_args = parser.parse_args(args)
    parsed_args.func(parsed_args)


def main():
    run()
