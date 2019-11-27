import os
import time
import subprocess
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
    bootstrap_strategy=exact_match, raw_blocking=False, raw_bootstrap=False,
    alpha=0, similarity_threshold=0.8,
    attr_strategy={'name': 'jaro_winkler'},
    jw_prefix_weight=0.1, soft_tfidf_threshold=0.5
)
start_time = time.time()
res = entity_resolver.resolve_and_eval(graph_path, ground_truth_path)
end_time = time.time()
entity_resolver.print_time()
os.remove(graph_path)
os.remove(ground_truth_path)
print(f'precision: {res[0]}')
print(f'recall: {res[1]}')
print(f'f1: {res[2]}')
print(f'total time taken: {end_time - start_time}')
