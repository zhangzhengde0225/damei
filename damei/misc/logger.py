import logging
import os
from pathlib import Path

LOGGING_LEVEL = os.environ.get('LOGGING_LEVEL', logging.INFO)
DEFAULT_FORMAT = f"\033[1;35m[%(asctime)s]\033[0m \033[1;32m[%(name)s]\033[0m " \
                 f"\033[1;36m[%(levelname)s]:\033[0m %(message)s"


def _file_logging_disabled():
    value = os.environ.get('DAMEI_DISABLE_FILE_LOG', '')
    return value.lower() in {'1', 'true', 'yes', 'on'}


def _default_log_dir():
    root = os.environ.get('DAMEI_LOG_DIR')
    if root:
        return Path(root)
    return Path.home() / '.damei' / 'logs' / Path(os.getcwd()).name


def _has_file_handler(logger, log_path):
    log_path = os.path.abspath(str(log_path))
    for handler in logger.handlers:
        if isinstance(handler, logging.FileHandler):
            if os.path.abspath(handler.baseFilename) == log_path:
                return True
    return False


def _add_file_handler(logger, log_path, level, format_str):
    if _has_file_handler(logger, log_path):
        return

    try:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(str(log_path), encoding='utf-8')
    except OSError:
        return

    fh.setLevel(level=level)
    fh.setFormatter(logging.Formatter(format_str))
    logger.addHandler(fh)


def get_logger(name=None, **kwargs):
    return getLogger(name, **kwargs)


def getLogger(name=None, **kwargs):
    name = name if name else 'root'
    name_lenth = kwargs.get('name_lenth', 12)
    name = f'{name:<{name_lenth}}'
    logger = logging.getLogger(name)
    level = kwargs.get('level', LOGGING_LEVEL)
    # level = kwargs.get('level', logging.ERROR)
    format_str = kwargs.get('format_str', DEFAULT_FORMAT)
    logging.basicConfig(level=level,
                        format=format_str,
                        # datefmt='%d %b %Y %H:%M:%S'
                        )

    log_dir = kwargs.get('log_dir', None)
    if log_dir is not None:
        log_dir = Path(log_dir)
    else:
        log_dir = _default_log_dir()

    log_to_file = kwargs.get('log_to_file', not _file_logging_disabled())
    if log_to_file:
        log_path = log_dir / f'{Path(os.getcwd()).name}.log'
        _add_file_handler(logger, log_path, level, format_str)

    return logger
