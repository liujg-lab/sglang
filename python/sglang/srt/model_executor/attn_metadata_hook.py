"""Finish or poison attention metadata after one complete forward."""


def invoke_attn_metadata_hook(backend, name: str) -> None:
    """Call a backend metadata hook when that backend implements it."""
    hook = getattr(backend, name, None)
    if hook is not None:
        hook()


def run_forward_with_metadata_hooks(backend, forward_fn):
    try:
        output = forward_fn()
    except BaseException:
        invoke_attn_metadata_hook(backend, "poison_forward_metadata")
        raise
    invoke_attn_metadata_hook(backend, "finish_forward_metadata")
    return output
