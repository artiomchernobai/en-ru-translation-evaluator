"""Validate fixed or adaptive window membership without loading a tokenizer."""


def validate_context(window, center, group_count, radius, config):
    base = list(range(max(0, center-radius), min(group_count, center+radius+1)))
    strategy = config.get('strategy', 'fixed')
    ids = window['group_indices']
    if config.get('radius', radius) != radius:
        raise ValueError('Window radius metadata differs')
    if strategy == 'fixed':
        if ids != base:
            raise ValueError('Invalid fixed COMET window groups')
        return ids
    if strategy != 'adaptive' or radius != 1 or config.get('radius', radius) != radius:
        raise ValueError('Invalid adaptive window configuration')
    extra, limit = config.get('max_extra_groups'), config.get('max_tokens')
    if type(extra) is not int or extra < 0 or type(limit) is not int or limit < 1:
        raise ValueError('Missing adaptive expansion/token limits')
    if (not ids or any(type(i) is not int or not 0 <= i < group_count for i in ids)
            or ids != list(range(ids[0], ids[-1]+1)) or not set(base).issubset(ids)
            or len(ids) > len(base)+extra):
        raise ValueError('Invalid adaptive COMET window groups')
    context = window.get('context', {})
    if context.get('strategy') != strategy or context.get('base_group_indices') != base or context.get('max_tokens') != limit:
        raise ValueError('Adaptive context provenance differs')
    steps = context.get('expansions')
    if not isinstance(steps, list) or len(steps) != len(ids)-len(base):
        raise ValueError('Invalid expansion trace')
    trace = list(base)
    for step in steps:
        direction = step.get('direction')
        expected = trace[0]-1 if direction == 'left' else trace[-1]+1
        if direction not in ('left', 'right') or step.get('group_index') != expected or not step.get('reasons'):
            raise ValueError('Invalid expansion direction/reasons')
        trace = [expected]+trace if direction == 'left' else trace+[expected]
    if trace != ids:
        raise ValueError('Expansion trace differs from window')
    counts = context.get('token_counts', {})
    if set(counts) != {'src', 'mt'} or any(type(n) is not int or n < 0 for n in counts.values()):
        raise ValueError('Missing COMET token counts')
    if max(counts.values()) > limit:
        if ids != base or 'minimum_exceeds_token_limit' not in context.get('stop_reasons', []) or window['verdict']['status'] == 'scored':
            raise ValueError('Over-budget adaptive window was expanded or scored')
    return ids
