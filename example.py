import os
import time
import subprocess
import random
from py_stringmatching.similarity_measure.levenshtein import Levenshtein
from entity_resolver import EntityResolver


def edit_distance(attrs1, attrs2):
    try:
        f_initial1, f_initial2 = attrs1['name'][1][0], attrs2['name'][1][0]
    except IndexError:
        f_initial1, f_initial2 = '', ''
    if f_initial1 != f_initial2:
        return float('inf')
    levenshtein = Levenshtein()
    last_name1 = attrs1['name'][0]
    last_name2 = attrs2['name'][0]
    dist = levenshtein.get_raw_score(last_name1, last_name2)
    return dist


def exact_match(attrs1, attrs2):
    return attrs1['name'] == attrs2['name']


file_path = 'data/citeseer/citeseer-mrdm05.dat'
graph_path = 'graph.json'
ground_truth_path = 'ground_truth.json'
cmd_args = ['entity-resolver', 'norm-citeseer']
subprocess.call(cmd_args)
entity_resolver = EntityResolver(
    attr_types={'name': 'person_entity'}, blocking_strategy=edit_distance,
    raw_blocking=False, alpha=0, attr_strategy={'name': 'jaro_winkler'},
    rel_strategy='jaccard_coef', bootstrap_strategy=exact_match,
    raw_bootstrap=False, first_attr=None, first_attr_raw=False,
    second_attr=None, second_attr_raw=False, linkage='max',
    similarity_threshold=0.8, seed=None, plot_prc=False,
    evaluator_strategy='precision-recall', verbose=1, jw_prefix_weight=0.1,
    soft_tfidf_threshold=0.5, average_method='max'
)
entity_resolver.resolve_and_eval(graph_path, ground_truth_path)
os.remove(graph_path)
os.remove(ground_truth_path)
