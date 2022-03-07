import damei as dm
import numpy as np
import copy

logger = dm.getLogger('test5')

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

# bbox iou
b1 = np.array([10, 10, 500, 500])
b2 = np.array([12, 12, 450, 520])
dm.general.bbox_iou(b1, b2)

# 时间模块测试
print()
ct = dm.current_time()
et = dm.plus_time(ct, 1 * 24 * 60 * 60)
within = dm.within_time(expiration_time=et)
print(f'ct: {ct}, et: {et} within: {within}')
exit()

# 加密模块测试
logger.info('test dm.rsa')
dm.rsa.gen_rsa(num=2)
raw_plaintext = 'im dacongmei'
plaintext = copy.copy(raw_plaintext)
# 测试1: 加密方案，公钥加密，私钥解密
ciphertext = dm.rsa.encrypt(f'{plaintext}', pubk='pubk1.pem')  # 一端拥有公钥
print(f'ciphertext: {ciphertext} {type(ciphertext)}')
plaintext = dm.rsa.decrypt(ciphertext, privk='privk1.pem')  # 另一端拥有私钥
print(f'plaintext: {plaintext} {type(plaintext)}')
# 测试2: 签密方案，私钥2签名，公钥1加密，私钥1解密，公钥2验签
plaintext = copy.copy(raw_plaintext)
signature = dm.rsa.sign(message=plaintext, privk='privk2.pem', hash_method='MD5')  # 一端拥有公钥1+私钥2
ciphertext = dm.rsa.encrypt(f'{plaintext};{signature}', pubk='pubk1.pem')
print(f'signature: {signature} {type(signature)}')
print(f'ciphertext: {ciphertext} {type(ciphertext)}')
plaintext = dm.rsa.decrypt(ciphertext, privk='privk1.pem')  # 另一端拥有私钥1+公钥2
plain, sign = plaintext.split(';')
print(f'sign: {sign} \nplain: {plain}')
ret = dm.rsa.verify(message=plain, signature=signature, pubk='pubk2.pem')
print(f'is verified: {ret}')
