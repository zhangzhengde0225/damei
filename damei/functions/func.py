"""
函数集合
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator


def ReLU(x):
	x = 1 * (x > 0) * x
	return x


def FastPower(x, power=3):
	x = ReLU(x)
	return np.power(x + (1 - x) / 2, power)


def ImDU(x, power=3):
	return ReLU(1 - FastPower(x, power=power))


if __name__ == '__main__':
	#
    ax = plt.gca()
    ax.spines['right'].set_color('none')
    ax.spines['top'].set_color('none')
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')
    ax.spines['bottom'].set_position(('demo_for_dm.data', 0))
    ax.spines['left'].set_position(('demo_for_dm.data', 0))
    ax.xaxis.set_major_locator(MultipleLocator(0.5))
    ax.yaxis.set_major_locator(MultipleLocator(0.5))

    x = np.linspace(0, 2, 100)
    print(x)
    f = FastPower
    # f = ImDU
    y = f(x, power=8)
    y = ImDU(x, 8)
    print(y)
	color = '#EE3B3B'
	# plt.plot((-1, 0), (0, 0), color=color, linewidth=5)
	# plt.plot((0, 0), (0, 1), color=color, linewidth=5)
	plt.plot(x, y, color=color, linewidth=5)
	# plt.plot(x, f(x, power=3))
	plt.xlim(-0.2, 1.2)
	plt.ylim(-0.2, 1.2)

	fontdict = {'size': 18, 'weight': 'bold'}
	# plt.legend(prop=fontdict)
	# plt.ylabel(f'1 (eV)', fontdict=fontdict)
	plt.yticks(size=16, weight='bold')
	plt.xticks(size=16, weight='bold')
	plt.show()
