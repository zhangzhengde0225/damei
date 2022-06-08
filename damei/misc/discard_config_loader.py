import collections
import functools
import os
import re
import sys
from pathlib import Path

from easydict import EasyDict

from . import misc


class PyConfig(collections.UserDict):
    def __init__(self, cfg_file=None, name=None, root_path=None):
        super(PyConfig, self).__init__()
        self.root_path = root_path
        # self.root_path = root_path if root_path else f'{pydir.parent.parent}'  # 上上级路径
        self._name = name if name else f'{cfg_file}'  # 其实是path
        self._items = dict()
        self.init_config(cfg_file)  # 根据配置文件把内容和属性注册到items里

        self.check_items()  # TODO

    def __len__(self):
        return len(self._items)

    def __dict__(self):
        return self._items

    def __getitem__(self, item):
        return self._items[item]

    def __repr__(self):
        format_str = f'<class {self.__class__.__name__}> ' \
                     f'(path="{self._name}", ' \
                     f'items={self.items}'
        # f'keys={self._items.keys()})'

        return format_str

    def __missing__(self, key):
        if isinstance(key, str):
            raise KeyError(key)
        return self[str(key)]

    def __contains__(self, key):
        return str(key) in self.data

    def __setitem__(self, key, item):
        self.data[str(key)] = item

    # def __getattr__(self, key):
    #    return self.demo_for_dm.data[str(key)]

    def info(self):
        """查看当前所有配置信息"""
        # info_str = f'{self.__repr__()}\n'
        info_str = f'Configs:\n'
        info_dict = self.items
        info_str += misc.dict2info(info_dict)
        return info_str

    def init_config(self, cfg_file):
        if cfg_file is None:
            return
        cp = Path(os.path.abspath(cfg_file))  # config path
        # 根据sys.path的环境路径和cp的绝对路径，获取module_dir
        valid_sys_path = [x for x in sys.path if x in f'{cp}']
        if len(valid_sys_path) >= 1:
            rp = valid_sys_path[0]  # root_path
        else:
            sys.path.append(f'{cp.parent}')
            rp = f'{cp.parent}'
        # 提取模块路径：如：modules.detection.seyolov5.config
        module_dir = f'{str(cp.parent).replace(f"{rp}/", "").replace("/", ".")}'
        # print(module_dir)
        # print(f'rp: {rp} \ncp: {cp}\nm dir: {module_dir}')
        code = f'from {module_dir} import {cp.stem}'
        # print(cfg_file, code)
        exec(code)
        cfg = eval(cp.stem)  # 模块名

        # 读取文件获取顶格写并且不以#开头的属性
        with open(cfg_file, 'r') as f:
            data = f.readlines()
        data = [x for x in data if not (x.startswith(' ') or x.startswith('#'))]  # 非顶格的不要，# 开头的不要
        data = '\n'.join(data)
        attrs = re.findall(pattern=r"\s[a-z]\w+ ?= ?", string=data)
        attrs = [x.split('=')[0].replace('\n', '').strip() for x in attrs]

        # 注册属性到_items里，注册属性到self的属性里。
        for attr in attrs:
            if hasattr(cfg, attr):
                exec(f"self._items[attr] = cfg.{attr}")
                attr_value = self._items[attr]
                if isinstance(attr_value, dict):
                    setattr(self, attr, EasyDict(self._items[attr]))
                else:
                    setattr(self, attr, attr_value)
            else:
                pass

    def check_items(self):
        """初始化末，检查所有配置项，处理：缩略语~转为真实路径。
        递归方法，如果v类型是str,判断替换，如果循环完了，结束，如果v的类型是dict，继续循环
        """
        # print(self._items)
        for k, v in self._items.items():
            new_v = self.recursive_func(v)
            self._items[k] = new_v
            # 更新属性
            if hasattr(self, k):
                delattr(self, k)
            setattr(self, k, EasyDict(new_v)) if isinstance(new_v, dict) else setattr(self, k, new_v)

    def recursive_func(self, value):
        """递归函数,如果v类型是str,判断替换，如果循环完了，结束，如果v的类型是dict，继续循环"""
        if value is None:
            return None
        elif isinstance(value, str):
            return value.replace('~', os.environ['HOME'])
        elif isinstance(value, int):
            return value
        elif isinstance(value, float):
            return value
        elif isinstance(value, bool):
            return value
        elif isinstance(value, list):
            new_list = [self.recursive_func(x) for x in value]
            return new_list
        elif isinstance(value, dict):
            new_dict = {}
            for k, v in value.items():
                new_dict[k] = self.recursive_func(v)
            return new_dict
        else:
            raise TypeError(f'Config file only support basic python types but the value type is: {type(value)}.')

    def update_item(self, attrs, value):
        """更新配置条目
        attrs: 长度为0时代表，更新全部value，长度为1时，代表第1个元素是第一层，
        """
        if len(attrs) == 0:
            self.merge(value)
        elif len(attrs) == 1:
            self._items[attrs[0]] = value
        elif len(attrs) == 2:
            self._items[attrs[0]][attrs[1]] = value
        elif len(attrs) == 3:
            self._items[attrs[0]][attrs[1]][attrs[2]] = value
        else:
            self._items[attrs[0]][attrs[1]][attrs[2]][attrs[3]] = value
        self.check_items()

    def merge(self, cfg2):
        """合并另一个配置文件到内部，是inplace的"""
        if cfg2 is None:
            return self
        for k2, v2 in cfg2._items.items():
            merged_v = self.recursive_func2(v2, k2=k2, sub_items=self._items)
            self._items[k2] = merged_v
        # print(self.items)
        self.check_items()
        return self

    def recursive_func2(self, v2, k2=None, sub_items=None):
        """v1是之前的值，v2是现在的值"""
        if k2 and (k2 not in list(sub_items.keys())):
            return v2
        else:
            v1 = sub_items[k2]

            if type(v1) != type(v2):
                return v2
            f = functools.partial(isinstance, v1)
            if f(str) or f(int) or f(float) or f(bool):
                return v2
            elif f(list):
                tmp = [x for x in v2 if x not in v1]
                return v1 + tmp
            elif f(dict):  # v1v2都是dict
                new_dict = v1  # 保留v1
                for kk, vv in v2.items():
                    new_dict[kk] = self.recursive_func2(vv, k2=kk, sub_items=v1)
                return new_dict

    @property
    def items(self):
        return self._items

    def keys(self):
        return self._items.keys()

    def get(self, key, default=None):
        if key in list(self._items.keys()):
            return self._items[key]
        else:
            return default
