import logging
import os
from pathlib import Path


def get_logger(name=None, **kwargs):
    return getLogger(name, **kwargs)


def getLogger(name=None, **kwargs):
    name = name if name else 'root'
    name_lenth = kwargs.get('name_lenth', 12)
    name = f'{name:<{name_lenth}}'
    logger = logging.getLogger(name)
    level = kwargs.get('level', logging.INFO)
    format_str = f"\033[1;35m[%(asctime)s]\033[0m \033[1;32m[%(name)s]\033[0m " \
                 f"\033[1;36m[%(levelname)s]:\033[0m %(message)s"
    logging.basicConfig(level=level,
                        format=format_str,
                        # datefmt='%d %b %Y %H:%M:%S'
                        )
    logg_dir = f'{Path.home()}/.damei/logs/{Path(os.getcwd()).name}'
    if not os.path.exists(logg_dir):
        os.makedirs(logg_dir)
    fh = logging.FileHandler(f'{logg_dir}/{Path(os.getcwd()).name}.log')
    fh.setLevel(level=level)
    # ch = logging.StreamHandler()
    # ch.setLevel(level=level)
    fh.setFormatter(logging.Formatter(format_str))
    # ch.setFormatter(logging.Formatter(format_str))
    logger.addHandler(fh)
    # logger.addHandler(ch)
    return logger
