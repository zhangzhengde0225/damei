damei库是一个深度学习库，包含了常用函数及控制台。

# 1.安装
```python
pip
install
git + https: // github.com / zhangzhengde0225 / damei.git  # 从github安装
pip
install
damei - i
https: // pypi.Python.org / simple  # 从pypi安装
```

# 2.使用

```python
import damei as dm
```

# 3.Features

```python
"""
20200808 v1.0.0 初始版本

20210303 v1.0.15 general.letterbox
				general.plot_one_box_trace_pose_status
				general.xyxy2xywh
				general.xywh2xyxy
				dm.torch_utils.select_device
20211010 v1.0.153 dm.tools.check_COCO dm.tools.check_YOLO
20211201 v1.0.154 数据增强，根据mask的目标的随机位置、随机缩放和随机旋转
20211210 v1.0.155 dm.genral.pts2bbox 
20211223 v1.0.156 dm.video2frames
20211223 v1.0.157 dm.misc.list2table 
                  dm.misc.flatten_list
20220114 v1.0.158 dm.general.torch_distributed_zero_firs
20220117 v1.0.159 dm.getLogger(name=None, name_lenth=12, level=logging.INFO)
20220215 v1.0.160 dm.misc.dict2info(info_dict)  # 字典带颜色转为方便查看的字符串
                  dm.current_system()  # 返回linux windows macos
                  dm.system_lib_suffix()  # 返回.so .dll .dylib
20220304 v1.0.161 支持无torch库时import damei，有torch库时能调用
20220305 v1.0.162 ct = dm.current_time()
                  et = dm.plus_time(ct=None, seconds=1)
                  bool = dm.within_time(et, ct=None)  # 输入到期时间，判断当前时间是否在内 
                  dm.rsa.gen_rsa(length=1024)  # 在随机生成公钥-私钥对，存储在当前路径pubk.txt和privk.txt里
                  dm.rsa.encrypt(plaintext, pubk=None, privk=None, length=1024)  # 用公钥或私钥对铭文加密
                  dm.rsa.decrypt(ciphertext, pubk=None, privk=None, length=1024)  # 用私钥或公钥对密文解密
                  dm.rsa.sign(message, privk, hash_method='MD5')
                  dm.rsa.verify(message, signature, pubk)

# 20220305 v1.1.0 从该版本起，damei不仅支持库使用，还作为一个应用程序使用，命令行输入dm [命令] [选项] [参数]
"""
```




