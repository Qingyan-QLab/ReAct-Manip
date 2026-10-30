"""Resolved inventory for a fixed conveyor question, independent of reset."""
from copy import deepcopy
import numpy as np


def sample_layout(seed, n, k, example_counts=None):
    # Match the legacy constructor's draw order, including evaluated defaults.
    rng = np.random.default_rng(seed)
    rng.integers(10, 31); rng.integers(2, 5)
    categories = rng.choice(8, k, replace=False)
    rng.uniform(.08, .2); rng.uniform(.4, .8)
    targets = [dict(category=int(categories[i % k]), variant=int(rng.integers(10))) for i in range(n)]
    rng.shuffle(targets)
    sampled_counts = rng.integers(0, 3, k).tolist()
    counts = sampled_counts if example_counts is None else list(example_counts)
    mapping = rng.permutation(categories)
    examples = [dict(category=int(mapping[i]), variant=int(rng.integers(10)), basket=i, slot=s)
                for i, count in enumerate(counts) for s in range(count)]
    return dict(categories=categories.tolist(), targets=targets, examples=examples,
                fixed_categories=[int(c) if count else -1 for c, count in zip(mapping, counts)],
                target_yaws=rng.uniform(-.15, .15, n).tolist())


def validate_layout(layout, n, k):
    if not isinstance(layout, dict) or set(layout) != {'categories', 'targets', 'examples', 'fixed_categories', 'target_yaws'}:
        raise ValueError('invalid conveyor inventory fields')
    cats = layout['categories']
    if not isinstance(cats, list) or len(cats) != k or any(type(c) is not int or not 0 <= c < 8 for c in cats) or len(set(cats)) != k:
        raise ValueError('invalid category selection')
    targets, examples = layout['targets'], layout['examples']
    if not isinstance(targets, list) or len(targets) != n or not isinstance(examples, list):
        raise ValueError('invalid inventory counts')
    for record in targets + examples:
        expected = {'category', 'variant'} | ({'basket', 'slot'} if 'basket' in record else set())
        if not isinstance(record, dict) or set(record) != expected or type(record['category']) is not int or record['category'] not in cats or type(record['variant']) is not int or not 0 <= record['variant'] < 10:
            raise ValueError('invalid asset record')
    if any(set(t) != {'category','variant'} for t in targets):
        raise ValueError('target cannot be an example')
    counts = [sum(t['category'] == c for t in targets) for c in cats]
    if max(counts) - min(counts) > 1:
        raise ValueError('targets must be balanced')
    fixed = layout['fixed_categories']
    if not isinstance(fixed, list) or len(fixed) != k or any(type(c) is not int or c not in [-1, *cats] for c in fixed) or len([c for c in fixed if c >= 0]) != len(set(c for c in fixed if c >= 0)):
        raise ValueError('invalid fixed basket assignment')
    for e in examples:
        if set(e) != {'category','variant','basket','slot'} or type(e['basket']) is not int or not 0 <= e['basket'] < k or type(e['slot']) is not int or e['slot'] not in (0, 1):
            raise ValueError('invalid example slot')
    for i, cat in enumerate(fixed):
        rows = [e for e in examples if e['basket'] == i]
        if (cat < 0 and rows) or (cat >= 0 and not 1 <= len(rows) <= 2) or sorted(e['slot'] for e in rows) != list(range(len(rows))) or any(e['category'] != cat for e in rows):
            raise ValueError('examples disagree with fixed basket assignment')
    yaws = layout['target_yaws']
    if not isinstance(yaws, list) or len(yaws) != n or any(type(y) not in (int,float) or not -.15 <= y <= .15 for y in yaws):
        raise ValueError('invalid target yaw')
    return deepcopy(layout)
