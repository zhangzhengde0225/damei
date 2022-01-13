import damei as dm
import numpy as np

pts = [[1558, 922], [1588, 834], [2370, 948], [2402, 1084], [1936, 1048], [1564, 996], [1564, 996]]
pts = np.array(pts)
print(pts.shape)

bbox = dm.general.pts2bbox(pts)

print(f'pts: {pts}]')
print(f'bbox: {bbox}')

a = np.min(pts, axis=0)
print(a)

import damei as dm

a = [[22, '1', ['xxx', 22]], 'x', {'a': 1, 'b': 3}]
dm.misc.list2table(a)
print(dm.misc)
