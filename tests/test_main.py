import pytest
from entity_resolver import EntityResolver


class TestEntityResolver:

    def test_default_kwargs(self):
        entity_resolver = EntityResolver(None, None)
        assert entity_resolver.second_sim == 'jaro_winkler'
        assert entity_resolver.stfidf_threshold == 0.5
        assert entity_resolver.jw_prefix_weight == 0.1
        assert entity_resolver.average_method == 'max'

    def test_set_kwargs(self):
        entity_resolver = EntityResolver(
            None, None,
            second_sim='scaled_lev', stfidf_threshold=0.9,
            jw_prefix_weight=0.2, average_method='arithmetic'
        )
        assert entity_resolver.second_sim == 'scaled_lev'
        assert entity_resolver.stfidf_threshold == 0.9
        assert entity_resolver.jw_prefix_weight == 0.2
        assert entity_resolver.average_method == 'arithmetic'

    def test_set_exception(self):
        entity_resolver = EntityResolver(None, None)
        with pytest.raises(ValueError):
            entity_resolver = EntityResolver(None, None, alpha=-1)
        with pytest.raises(ValueError):
            entity_resolver.alpha = -1
        with pytest.raises(ValueError):
            entity_resolver = EntityResolver(None, None, average_method='foo')
        with pytest.raises(ValueError):
            entity_resolver.average_method = 'foo'
        with pytest.raises(ValueError):
            entity_resolver = EntityResolver(
                None, None, evaluator_strategy='bar'
            )
        with pytest.raises(ValueError):
            entity_resolver.evaluator_strategy = 'bar'
        with pytest.raises(ValueError):
            entity_resolver = EntityResolver(None, None, linkage='boo')
        with pytest.raises(ValueError):
            entity_resolver.linkage = 'boo'
        with pytest.raises(ValueError):
            entity_resolver = EntityResolver(None, None, second_sim='baz')
        with pytest.raises(ValueError):
            entity_resolver.second_sim = 'baz'
        with pytest.raises(ValueError):
            entity_resolver = EntityResolver(
                None, None, weights={'a': 0.5, 'b': 1}
            )
        with pytest.raises(ValueError):
            entity_resolver.weights = {'a': 0.5, 'b': 1}

    def test_setters(self):
        entity_resolver = EntityResolver(None, None)
        attr_list = [
            'attr_types', 'blocking_strategy', 'raw_blocking', 'alpha',
            'weights', 'attr_strategy', 'rel_strategy', 'blocking_threshold',
            'bootstrap_strategy', 'raw_bootstrap', 'edge_match_threshold',
            'first_attr', 'first_attr_raw', 'second_attr', 'second_attr_raw',
            'linkage', 'similarity_threshold', 'evaluator_strategy', 'seed',
            'plot_prc', 'second_sim', 'stfidf_threshold', 'jw_prefix_weight',
            'average_method', 'verbose'
        ]
        for attr in attr_list:
            if attr == 'weights':
                setattr(entity_resolver, attr, {1: 1})
            elif attr == 'linkage':
                setattr(entity_resolver, attr, 'min')
            elif attr == 'evaluator_strategy':
                setattr(entity_resolver, attr, 'ami')
            elif attr == 'second_sim':
                setattr(entity_resolver, attr, 'jaro')
            elif attr == 'average_method':
                setattr(entity_resolver, attr, 'min')
            else:
                setattr(entity_resolver, attr, 0.001)
        for attr in attr_list:
            if attr == 'weights':
                assert getattr(entity_resolver, attr) == {1: 1}
            elif attr == 'linkage':
                assert getattr(entity_resolver, attr) == 'min'
            elif attr == 'evaluator_strategy':
                assert getattr(entity_resolver, attr) == 'ami'
            elif attr == 'second_sim':
                assert getattr(entity_resolver, attr) == 'jaro'
            elif attr == 'average_method':
                assert getattr(entity_resolver, attr) == 'min'
            else:
                assert getattr(entity_resolver, attr) == 0.001
            if attr == 'second_sim':
                assert entity_resolver._resolver.kwargs[attr] == 'jaro'
                assert entity_resolver._evaluator.kwargs[attr] == 'jaro'
            elif attr == 'average_method':
                assert entity_resolver._resolver.kwargs[attr] == 'min'
                assert entity_resolver._evaluator.kwargs[attr] == 'min'
            elif attr in ['stfidf_threshold', 'jw_prefix_weight']:
                assert entity_resolver._resolver.kwargs[attr] == 0.001
                assert entity_resolver._evaluator.kwargs[attr] == 0.001
            elif attr not in [
                'evaluator_strategy', 'attr_types',
                'verbose', 'weights', 'linkage'
            ]:
                assert getattr(entity_resolver._resolver, attr) == 0.001
            assert entity_resolver._evaluator.strategy == 'ami'
            assert entity_resolver._graph_parser.attr_types == 0.001
            assert entity_resolver._graph_parser.verbose == 0.001
            assert entity_resolver._ground_truth_parser.verbose == 0.001
            assert entity_resolver._resolver.verbose == 0.001
            assert entity_resolver._evaluator.verbose == 0.001
            assert entity_resolver._resolver.weights == {1: 1}
            assert entity_resolver._resolver.linkage == 'min'
