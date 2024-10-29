from tqdm.auto import tqdm
from hoi.utils.logging import logger


def get_pbar(**kwargs):
    """Get progress bar"""
    kwargs["disable"] = logger.getEffectiveLevel() > 20
    kwargs["mininterval"] = 0.016
    kwargs["miniters"] = 1
    kwargs["smoothing"] = 0.05
    kwargs["bar_format"] = (
        "{percentage:3.0f}%|{bar}| {desc} {n_fmt}/{total_fmt} [{elapsed}"
        "<{remaining}, {rate_fmt:>11}{postfix}]"
    )
    return tqdm(**kwargs)
