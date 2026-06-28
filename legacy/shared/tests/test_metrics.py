import math

from routesense_poc1.metrics import next_token_nll


def test_next_token_nll_basic():
    logits = [
        [[-1.0, 0.0, 1.0], [0.2, 0.3, 0.5]],
        [[0.0, 0.0, 1.0], [0.1, 0.2, 0.7]],
    ]
    input_ids = [[1, 2], [2, 1]]
    nll = next_token_nll(logits, input_ids)
    assert math.isclose(nll, 0.9795253391882157, rel_tol=1e-9)
