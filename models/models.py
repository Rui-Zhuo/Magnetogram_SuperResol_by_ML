import copy
import warnings


models = {}


def register(name):
    def decorator(cls):
        models[name] = cls
        return cls
    return decorator


def make(model_spec, args=None, load_sd=False):
    model_args = copy.deepcopy(model_spec['args'])
    if args is not None:
        model_args.update(args)
    if load_sd and model_spec['name'] == 'lte-warp' and 'position_mode' not in model_args:
        model_args['position_mode'] = 'legacy_query_min'
        warnings.warn('Loading an unversioned legacy LTEW checkpoint with legacy_query_min PM. '
                      'Its tensors are unchanged; patch-center-PM reproduction with these weights is unverified. '
                      'New training configurations use patch_center.', UserWarning, stacklevel=2)
    if load_sd and model_spec['name'] == 'lte-warp':
        phase = model_spec['sd'].get('phase.weight')
        if phase is not None and phase.shape[-1] != 2:
            raise ValueError('This portable decoder accepts two-feature phase checkpoints only; '
                             'use the original ten-feature/Jacobian pipeline for other phase dimensions. '
                             'Checkpoint weights are not reshaped or padded.')
    model = models[model_spec['name']](**model_args)
    if load_sd:
        model.load_state_dict(model_spec['sd'])
    return model
