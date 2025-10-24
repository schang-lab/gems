ALLOWED_EDGE_TYPES = [
    ('indiv', 'responds', 'question'),
    ('question', 'responds', 'indiv'),
    ('subgroup', 'contains', 'indiv'),
    ('indiv', 'contains', 'subgroup'),
    ('indiv', 'self', 'indiv'),
    ('question', 'self', 'question'),
    ('subgroup', 'self', 'subgroup'),
]

ALLOWED_NODE_TYPES = [
    'indiv',
    'question',
    'subgroup',
]