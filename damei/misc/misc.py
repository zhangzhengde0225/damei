import os
from pathlib import Path
import collections
import functools

import numpy as np


def list2table(a, float_bit=2, alignment='<'):
    """
    格式化，列表变表格，返回能直接打印的字符串。
    :param a: list, 长度为n，每个元素内由是一个str bool int float None list或dict
        example：
            [['COLUME1', 'COLUME2', 'COLUME2'], [1, True, 3.1415926]]]
    :param float_bit: 浮点数的小数位数
    :param alignment: 表格对齐方式
    :return: format str:
        example:
            COLUME1 COLUME2 COLUME3
            1       True    3.14
    """
    assert isinstance(a, list)

    def convert(x):
        """把所有内部元素转换为list"""
        f = functools.partial(isinstance, x)
        if f(str) or f(int) or f(bool):
            return [f'{x}']
        elif x is None:
            return ['None']
        elif f(float):
            return [f'{x:.{float_bit}f}']
        elif f(list):
            tmp = [convert(xx) for xx in x]
            return flatten_list(tmp)
        elif f(dict):
            return [f"Key('{xx}')" for xx in x.keys()]
        else:
            raise TypeError(f'Only support basic python types. value type: {type(x)}, x: {x}')

    def complection(x, max_lenth):
        """补全内部list，用空补全"""
        tmp = ['-'] * max_lenth
        for i, xx in enumerate(x):
            tmp[i] = xx
        return tmp

    # 处理空值
    a = [convert(x) for x in a]
    max_lenth = np.max([len(x) for x in a])
    a = [complection(x, max_lenth=max_lenth) for x in a]

    # format string
    lenth = np.array([[len(x) for x in xx] for xx in a])  # (n, max_lenth)
    lenth = np.max(lenth, axis=0)  # (max_lenth,)
    format_str = [[f'{x:{alignment}{lenth[i]}}' for i, x in enumerate(xx)] for xx in a]
    format_str = '\n'.join(['  '.join(x) for x in format_str])
    return format_str


def flatten_list(a):
    """展平list"""
    new_list = []
    [new_list.extend(x) for x in a]
    return new_list


def dict2info(info_dict):
    """
    把一个字典的键和值展开为带缩进、带颜色的字符串，易于打印
    :param info_dict: 字典
    :return: str
    """
    # 先递归地展开
    new_info_dict = collections.OrderedDict()
    indents = []
    indent_space = 4
    for k, v in info_dict.items():
        dict2info_recursive_func(k, v, new_info_dict, indents, indent_space=indent_space)
    info_dict = new_info_dict

    indents2color = {0: '32m', 1 * indent_space: '35m', 2 * indent_space: '36m', 3 * indent_space: '33m'}
    lenth = np.max([len(x) for x in info_dict.keys()])
    format_str = ''
    for i, (k, v) in enumerate(info_dict.items()):
        indent = indents[i]
        # print(f'k: {k} {len(k)}')
        k_str = f'{k.strip("*"):>{indent + lenth}}'
        color = indents2color.get(indent, None)
        k_str = f'\033[1;{color}{k_str}\033[0m' if color else k_str  # 上色
        format_str += f'  {k_str} : {v}\n'
    return format_str


def dict2info_recursive_func(k, v, new_info_dict, indents, indent=0, indent_space=4):
    """递归拆分，原来的字典里，某些值可能还是dict，为了方便显示，递归展开，子dict的每个键作为新的顶层的键"""
    if isinstance(v, dict):
        while True:
            if k not in new_info_dict.keys():
                break
            k = '*' + k
        new_info_dict[f'{k}'] = f'({len(v.keys())})'
        indents.append(indent)
        indent += indent_space
        for k2, v2 in v.items():
            # print(f'k2: {k2} indent: {indent}', new_info_dict.keys())
            dict2info_recursive_func(k2, v2, new_info_dict, indents, indent, indent_space=indent_space)

    else:
        while True:
            if k not in new_info_dict.keys():
                break
            k = '*' + k
        new_info_dict[f'{k}'] = v
        indents.append(indent)
    return new_info_dict


def uncomment(string):
    """去除行内注释和换行符"""
    string = string.strip()
    return string.split('#')[0]


def rcfile2dict(rc_file):
    """
    读取.xxxrc文件，保存为dict.
    目前仅支持2层
    """
    data_dict = {}
    with open(rc_file, 'r') as f:
        data = f.readlines()

    data = [x for x in data if x.strip() and not x.startswith('#')]  # 去除空行和注释行
    names = [x for x in data if not x.startswith('\t')]  # 顶格的变量

    for i, name in enumerate(names):
        idx = data.index(name)
        cn = uncomment(name)  # clear_name 去除行内注释
        cn = [x.strip() for x in cn.split(':')]  # 分割变量名和值
        assert len(cn) == 2, f'{name} format error'
        key, value = cn
        if value != '':
            # print(f'{key}: {value}')
            data_dict[key] = value
        else:
            end_idx = data.index(names[i + 1]) if i < len(names) - 1 else len(data)
            content = data[idx + 1:end_idx]  # 包含的内容
            ct_dict = dict()
            for ct in content:
                ct = uncomment(ct)
                ct = [x.strip() for x in ct.split(':')]
                assert len(ct) == 2, f'{ct} format error'
                key2, value2 = ct
                ct_dict[key2] = value2
            data_dict[key] = ct_dict

    return data_dict


def count_codes(root_path, suffix=None, show_detail=False):
    return count_lines_and_chars(root_path, suffix, show_detail)


def count_lines_and_chars(root_path, suffix=None, show_detail=True, ):
    suffix = suffix if suffix else ['.py', '.pyx', '.c', '.cpp', '.h', '.hpp']
    suffix = [suffix] if isinstance(suffix, str) else suffix
    # print(root_path, suffixes)
    assert os.path.isdir(root_path), f'{root_path} is not a directory'

    num_lines = 0
    num_chars = 0
    blank = 30
    for root, dirs, files in os.walk(root_path):
        for file in files:
            # print(len(files), file)
            if f'{Path(file).suffix}' in suffix:
                path = os.path.join(root, file)
                with open(path, 'r') as f:
                    lines = f.readlines()
                lines = [x for x in lines if x.strip()]
                num_char = sum([len(x) for x in lines])
                num_lines += len(lines)
                num_chars += num_char
                if show_detail:
                    rel_path = path.replace(f'{root_path}/', '')
                    blank = max(len(rel_path), blank)
                    print(f'path: {rel_path:<{blank}} {len(lines):<4} lines, {num_char:<5} chars')

    ret = f'Count codes on: {root_path}:\n {num_lines} lines, {num_chars} chars'
    print(ret)
    return (num_lines, num_chars)


if __name__ == '__main__':
    a = [[22, '1', ['xxx', 22]], 'x', {'a': 1, 'b': 3}]
    print(a[2].keys())
    ret = list2table(a)
    print(ret)
