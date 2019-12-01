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
            entity_resolver.alpha = -1
        with pytest.raises(ValueError):
            entity_resolver = EntityResolver(None, None, average_method='foo')
            entity_resolver.average_method = 'foo'
        with pytest.raises(ValueError):
            entity_resolver = EntityResolver(
                None, None, evaluator_strategy='bar'
            )
            entity_resolver.evaluator_strategy = 'bar'
        with pytest.raises(ValueError):
            entity_resolver = EntityResolver(None, None, linkage='boo')
            entity_resolver.linkage = 'boo'
        with pytest.raises(ValueError):
            entity_resolver = EntityResolver(None, None, second_sim='baz')
            entity_resolver.second_sim = 'baz'
