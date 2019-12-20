""" Implement command line interface to preprocess arxiv and citeseer data.

This is the implementaion file of command line tools
``entity-resolve prep-arxiv`` and ``entity-resolver prep-citeseer``. For usage
of the tools, please refer to :doc:`../tutorial` or use with ``-h`` flag for
help menu. It also includes a python API: the `run` function for such
preprocessing tasks, though it is primarily implemented for testing purpose.

Example:

    >>> from entity_resolver.scripts import run
    >>> cmd_args = [
    ...     'prep-arxiv',
    ...     '--data', 'path/to/input_data',
    ...     '--graph', 'path/to/output_graph',
    ...     '--ground_truth', 'path/to/output_ground_truth'
    ... ]
    >>> run(cmd_args)

The above example is equivalent to the terminal command:

.. code-block:: console

    $ entity-resolver prep-arxiv\\
    > --data path/to/input_data\\
    > --graph path/to/output_graph\\
    > --ground_truth path/to/output_ground_truth
"""

import argparse
import json
from typing import Optional, Sequence
from .core.utils import subparser


@subparser(
    'prep-arxiv',
    'Transform the arxiv data into the format expected by EntityResolver',
    'The path of the arxiv data to be transformed',
    'The path of the transformed arxiv graph data',
    'The path of the transformed arxiv ground truth data'
)
def _norm_arxiv(
    data: str = 'data/arxiv/arxiv-mrdm05.dat',
    graph: str = 'graph.json',
    ground_truth: str = 'ground_truth.json'
) -> None:
    """ Preprocess the arxiv data to the desired format.

    Args:
        data: The path to the input arxiv data.
        graph: The path to the output graph data.
        ground_truth: The path to the output ground truth data.
    """
    parse_data(data, graph, ground_truth, 'arxiv')


@subparser(
    'prep-citeseer',
    'Transform the citeseer data into the format expected by EntityResolver',
    'The input file path of the citeseer data to be transformed',
    'The path of the transformed citeseer graph data',
    'The path of the transformed citeseer ground truth data'
)
def _norm_citeseer(
    data: str = 'data/citeseer/citeseer-mrdm05.dat',
    graph: str = 'graph.json',
    ground_truth: str = 'ground_truth.json'
) -> None:
    """ Preprocess the citeseer data to the desired format.

    Args:
        data: The path to the input citeseer data.
        graph: The path to the output graph data.
        ground_truth: The path to the output ground truth data.
    """
    parse_data(data, graph, ground_truth, 'citeseer')


def parse_data(
    data: str, graph: str, ground_truth: str, name: str
) -> None:
    """ Preprocess either arxiv or citeseer data specified by ``name``.

    Args:
        data: The path to the input data.
        graph: The path to the output graph data.
        ground_truth: The path to the output ground truth data.
        name: Either 'arxiv' or 'citeseer'. Indicate which data is to be
        preprocessed.

    Note:
        * Only the third column named 'normalized_author' is included as
          attribute in the output graph file under the attribute name
          ``'name'``.
        * Also the rows with author id ``2716`` are removed in citeseer data
          during this process because multiple different rows with id 2716 are
          present, which does not make sense.
    """
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


def run(args: Optional[Sequence[str]] = None) -> None:
    """ Run ``prep-arxiv`` or ``prep-citeseer`` with console command input.

    Args:
        args: A list of strings representing a terminal command. The original
            command can be obtained by joinning them with white spaces.
    """
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers()
    _norm_arxiv(subparsers)
    _norm_citeseer(subparsers)
    parsed_args = parser.parse_args(args)
    parsed_args.func(parsed_args)


def main():
    """ The command line script entry point."""
    run()
