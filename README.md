[![Stars](https://img.shields.io/github/stars/zhangzhengde0225/damei)](
https://github.com/zhangzhengde0225/damei)
[![Open issue](https://img.shields.io/github/issues/zhangzhengde0225/damei)](
https://github.com/zhangzhengde0225/damei/issues)
[![Source_codes](https://img.shields.io/static/v1?label=Download&message=src&color=orange)](
https://github.com/zhangzhengde0225/damei/archive/refs/heads/master.zip)
[![Source_codes](https://img.shields.io/static/v1?label=Docs&message=Available&color=orange)](
http://47.114.37.111)
# damei

damei(大妹) library。

涉及深度学习和量子力学交叉。

提供常用函数、控制台、解算器等。

# 1.安装

pypi安装:

```
pip install damei  
```

[//]: # (或从源码安装：[Download]&#40;https://github.com/zhangzhengde0225/damei/archive/refs/heads/master.zip&#41;.)
下载源码后：

```
unzip damei-master.zip
cd damei-master
python setup.py install
```

# 2.使用

## [详细API文档](http://47.114.37.111)

```python
import damei as dm
```

[//]: # (### 使用示例[usage.md]&#40;https://github.com/zhangzhengde0225/damei/blob/master/docs/usage.md&#41;.)

# 3.更新日志

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
20220305 v1.0.162 ct = dm.current_time()  # 当前时间
                  et = dm.plus_time(ct=None, seconds=1)  # 未来某个到期时间
                  bool = dm.within_time(et, ct=None)  # 输入到期时间，判断当前时间是否在内 
                  dm.rsa.gen_rsa(length=1024)  # 在随机生成公钥-私钥对，存储在当前路径pubk.pen和privk.pem里
                  dm.rsa.encrypt(plaintext, pubk=None, privk=None, length=1024)  # 用公钥或私钥对明文加密
                  dm.rsa.decrypt(ciphertext, pubk=None, privk=None, length=1024)  # 用私钥或公钥对密文解密
                  dm.rsa.sign(message, privk, hash_method='MD5')  # 签密体制下的签名
                  dm.rsa.verify(message, signature, pubk)  # 签密体制下的验签
20220309 v1.0.163 cfg = dm.PyConfig(cfg_file)
                    print(cfg.info())  # 查看配置
                    cfg.merge(cfg2)  # 合并配置      
20220322 v1.0.164 dm.comm.push_stream(  # 把视频推成流
                source, ip='127.0.0.1', port=1935, stream_type='rtmp', key=None))
20220323 v1.0.165 dmpeg = dm.DmFFMPEG()  # 传入一堆参数
                    for img in imgs:  # 循环中推流
                        dmpeg.push_stream(img)  # 推流
                # 取流：ffplay rtmp://127.0.0.1:1935/live
    
20220408 v1.1.0 # 新增dm.nn
            uaii = dm.nn.api.UAII()
         v1.1.1 # 新增dm.argparse，只读取默认值，不解析命令行参数
            parser = dm.argparse.ArgumentParser()
20220610 v1.1.7 # damei_doc上线
20220621 v1.1.8 # damei.comm.push_stream()  提供API文档
"""
```




