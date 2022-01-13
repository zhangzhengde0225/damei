import functools
import numpy as np


def list2table(a, float_bit=2, alignment='<'):
	"""
	格式化，列表变表格，返回能直接打印的字符串。
	:param a: list, 长度为n，每个元素内由是一个str bool int float None list或dict
	:param float_bit: 浮点数的小数位数
	:param alignment: 表格对齐方式
	:return: None
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


if __name__ == '__main__':
	a = [[22, '1', ['xxx', 22]], 'x', {'a': 1, 'b': 3}]
	print(a[2].keys())
	ret = list2table(a)
	print(ret)
