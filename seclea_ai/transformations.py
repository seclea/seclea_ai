from inspect import signature

TRANFORM_WRAPPER_NAME = '_transform_wrapper'


def tracked_transform(fn):
    def _transform_wrapper(*args, **kwargs):
        # enforce signature to fn declaration e.g kwargs only and returns dict for return argument names.
        fn_sig = signature(fn)
        print(fn_sig)
        return fn(*args, **kwargs)

    return _transform_wrapper
