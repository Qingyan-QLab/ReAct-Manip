"""Finite-sample, within-scenario pass metrics (never pool different scenarios)."""
from math import comb


def pass_metrics(outcomes, ks):
    outcomes = list(outcomes)
    n, c = len(outcomes), sum(bool(x) for x in outcomes)
    if not n:
        raise ValueError('exam results must not be empty')
    result = {'exam_episodes': n, 'pass_count': c, 'pass_rate': c/n,
              'pass_at_k': {}, 'pass_all_k': {}}
    for k in ks:
        if type(k) is not int or not 1 <= k <= n:
            raise ValueError('K must be an integer in [1, exam_episodes]')
        result['pass_at_k'][str(k)] = 1 - comb(n-c, k)/comb(n, k)
        result['pass_all_k'][str(k)] = comb(c, k)/comb(n, k)
    return result
