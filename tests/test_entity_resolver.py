from entity_resolver import EntityResolver


class TestEntityResolver:

    def test_dummy(self):
        entity_resolver = EntityResolver()
        assert 'resolve' in dir(entity_resolver)
