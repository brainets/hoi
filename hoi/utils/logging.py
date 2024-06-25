"""
"""

import logging
import sys
import re

BLACK, RED, GREEN, YELLOW, BLUE, MAGENTA, CYAN, WHITE = range(8)
RESET_SEQ = "\033[0m"
COLOR_SEQ = "\033[1;%dm"
BOLD_SEQ = "\033[1m"
COLORS = {
    "DEBUG": GREEN,
    "PROFILER": MAGENTA,
    "INFO": WHITE,
    "WARNING": YELLOW,
    "ERROR": RED,
    "CRITICAL": RED,
}
FORMAT = {
    "compact": "$BOLD%(levelname)s | %(message)s",
    "spacy": "$BOLD%(levelname)-19s$RESET | %(message)s",
    "frites": "$BOLD%(name)s-%(levelname)-19s$RESET | %(message)s",
    "print": "%(message)s",
}
_LOGGING_TYPES = dict(
    DEBUG=logging.DEBUG,
    INFO=logging.INFO,
    WARNING=logging.WARNING,
    ERROR=logging.ERROR,
    CRITICAL=logging.CRITICAL,
)


def formatter_message(message, use_color=True):
    """Format the message."""
    return message.replace("$RESET", RESET_SEQ).replace("$BOLD", BOLD_SEQ)


class _Formatter(logging.Formatter):
    """Formatter."""

    def __init__(self, format_type="compact"):
        logging.Formatter.__init__(self, FORMAT[format_type])
        self._format_type = format_type

    def format(self, record):
        name = record.levelname
        msg = record.getMessage()
        # If * in msg, set it in RED :
        if "*" in msg:
            regexp = "\*.*?\*"
            re_search = re.search(regexp, msg).group()
            to_color = (
                COLOR_SEQ % (30 + RED)
                + re_search
                + COLOR_SEQ % (30 + WHITE)
                + RESET_SEQ
            )
            msg_color = re.sub(regexp, to_color, msg)
            msg_color += RESET_SEQ
            record.msg = msg_color
        # Set level color :
        levelname_color = COLOR_SEQ % (30 + COLORS[name]) + name + RESET_SEQ
        record.levelname = levelname_color
        if record.levelno == 20:
            logging.Formatter.__init__(self, FORMAT["print"])
        else:
            logging.Formatter.__init__(self, FORMAT[self._format_type])
        return formatter_message(logging.Formatter.format(self, record))


class _StreamHandler(logging.StreamHandler):
    """Stream handler allowing matching and recording."""

    def __init__(self):
        logging.StreamHandler.__init__(self, sys.stderr)
        self.setFormatter(_lf)
        self._str_pattern = None
        self.emit = self._frites_emit

    def _frites_emit(self, record, *args):
        msg = record.getMessage()
        test = self._match_pattern(record, msg)
        if test:
            record.msg = test
            return logging.StreamHandler.emit(self, record)
        else:
            return

    def _match_pattern(self, record, message):
        if isinstance(self._str_pattern, str):
            if re.search(self._str_pattern, message):
                sub = "*{}*".format(self._str_pattern)
                return re.sub(self._str_pattern, sub, message)
            else:
                return ""
        else:
            return message


logger = logging.getLogger("hoi")
# logger.propagate = True
_lf = _Formatter()
_lh = _StreamHandler()  # needs _lf to exist
logger.addHandler(_lh)
PROFILER_LEVEL_NUM = 1
logging.addLevelName(PROFILER_LEVEL_NUM, "PROFILER")


def profiler_fcn(self, message, *args, **kws):  # noqa
    # Yes, logger takes its '*args' as 'args'.
    if self.isEnabledFor(PROFILER_LEVEL_NUM):
        self._log(PROFILER_LEVEL_NUM, message, args, **kws)


logging.Logger.profiler = profiler_fcn
logger.setLevel(logging.INFO)


def set_log_level(verbose=None, return_old_level=False):
    """Set the logging level.

    Parameters
    ----------
    verbose : bool, str, int, or None
        The verbosity of messages to print. If a str, it can be either DEBUG,
        INFO, WARNING, ERROR, or CRITICAL. Note that these are for
        convenience and are equivalent to passing in logging.DEBUG, etc.
        For bool, True is the same as 'INFO', False is the same as 'WARNING'.
        If None, the environment variable MNE_LOGGING_LEVEL is read, and if
        it doesn't exist, defaults to INFO.
    return_old_level : bool
        If True, return the old verbosity level.

    Returns
    -------
    old_level : int
        The old level. Only returned if ``return_old_level`` is True.
    """
    old_verbose = logger.level

    if verbose is None:
        verbose = "INFO"
    elif isinstance(verbose, bool):
        if verbose is True:
            verbose = "INFO"
        else:
            verbose = "WARNING"
    if isinstance(verbose, str):
        verbose = verbose.upper()
        verbose = _LOGGING_TYPES[verbose]

    if verbose != old_verbose:
        logger.setLevel(verbose)
    return old_verbose if return_old_level else None
