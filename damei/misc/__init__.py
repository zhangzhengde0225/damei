from importlib import import_module

from .time import current_time, plus_time, within_time


class _LazyFunc(object):
    def __init__(self, module_name, func_name):
        self._module_name = module_name
        self._func_name = func_name
        self._func = None

    def _load(self):
        if self._func is None:
            self._func = getattr(import_module(self._module_name), self._func_name)
        return self._func

    def __call__(self, *args, **kwargs):
        return self._load()(*args, **kwargs)

    def __repr__(self):
        if self._func is None:
            return f"<lazy function {self._module_name}.{self._func_name}>"
        return repr(self._func)


_MISC_FUNC_NAMES = [
    "list2table",
    "flatten_list",
    "dict2string",
    "dict2info",
    "dict2info_recursive_func",
    "uncomment",
    "rcfile2dict",
    "count_codes",
    "count_lines_and_chars",
]

for _name in _MISC_FUNC_NAMES:
    globals()[_name] = _LazyFunc("damei.misc.misc_func", _name)


__all__ = [
    "current_time",
    "plus_time",
    "within_time",
    *_MISC_FUNC_NAMES,
]
