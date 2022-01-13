import random
import collections


class ColorControl(object):
	def __init__(self, num=None, filt_name=None, random_color=True, rgb=None):
		self.color = self.get_color(
			num=num, filt_name=filt_name, random_color=random_color, rgb=rgb)

	def get_colors_from_inner(self, filt_name=None, rgb=None):
		inner_colors = collections.OrderedDict({
			'dark_blue': ["#003399", (0, 51, 153)],
			'dark_red': ["#F00000", (240, 0, 0)],
			'dark_yellow': ["#CCCC00", (204, 204, 0)],
			'dark_green': ["#009966", (0, 153, 102)],
			'dark_purple': ["#663366", (102, 51, 102)],
			'dark_orange': ["#FF6600", (255, 102, 0)],
			'light_blue:': ["#CCFFFF", (204, 255, 255)],
			'light_red': ["#FF6666", (255, 102, 102)],
			'light_black': ["#CCCCCC", (204, 204, 204)],
			'light_green': ['#CCFF99', (204, 255, 153)],
			'light_pink': ['#FFCCCC', (255, 204, 204)],
			'light_orange': ["#FF9966", (255, 153, 102)]
		})
		# filter
		if filt_name is None:
			colors = list(inner_colors.values())
		else:
			colors = [v for k, v in inner_colors if filt_name in k]
		assert len(colors) >= 1
		if rgb is None:  # 取字符串
			colors = [x[0] for x in colors]
		elif rgb == 'RGB':
			colors = [x[1] for x in colors]
		elif rgb == 'BGR':
			# print(colors[0])
			colors = [(x[1][2], x[1][1], x[1][0]) for x in colors]
		return colors

	def get_color(self, num=None, filt_name=None, random_color=True, rgb=None):
		"""
		方案：
		1.如果随机，直接随机取num个颜色，num为None时取1000个
		2.不随机，根据filt_name确定内置颜色，颜色名字中包含filt_name的颜色被选中。
			同时，不设置num, 全部取回，设置了num，内置颜色大于num, 取回num个，内置颜色小于None, 后面的用random补全，
		:param num:
		:param filt_name: None, 取所有内置颜色
		:return:
		"""
		if random_color:
			num = 1000 if num is None else num
			colors = [[random.randint(0, 255) for _ in range(3)] for __ in range(num)]
		else:
			colors = self.get_colors_from_inner(filt_name=filt_name, rgb=rgb)
			if num is not None:
				# assert len(colors) >= num, f'Number of filtered inner color: {len(colors)} less than num para: {num}.'
				if len(colors) < num:  # 补全
					rand_colors = [[random.randint(0, 255) for _ in range(3)] for __ in range(num - len(colors))]
					colors = colors + rand_colors
				if len(colors) == num:
					pass
				else:  # > num
					colors = colors[:num]
		return colors
