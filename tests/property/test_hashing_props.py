import random

from hypothesis import given
from hypothesis import strategies as st

from cornerman.pipeline.hashing import hash_obj

json_values = st.recursive(
    st.none() | st.booleans() | st.integers() | st.text(max_size=5),
    lambda inner: (
        st.lists(inner, max_size=3) | st.dictionaries(st.text(max_size=5), inner, max_size=3)
    ),
    max_leaves=10,
)


@given(d=st.dictionaries(st.text(max_size=5), json_values, max_size=6), seed=st.integers())
def test_hash_obj_ignores_dict_insertion_order(d: dict[str, object], seed: int) -> None:
    items = list(d.items())
    random.Random(seed).shuffle(items)
    assert hash_obj(dict(items)) == hash_obj(d)
