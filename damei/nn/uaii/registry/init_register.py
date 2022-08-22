"""
初始化注册，
在导入nn模块自动注册内部模块时，由于外部调用多次会导致对此初始化注册，仅初始化注册一次
"""
import os, sys


class InitRegister(object):
    def __init__(self, internal_dir=None):
        self.inited = False
        self.internal_dir = internal_dir

    def __call__(self, internal_modules, external_folders):
        """
        初始化注册
        :param internal_modules: internal modules, ex: models/loader/vis_loader
        :param external_folders: external folders, ex: dmapi, repos
        """
        if self.inited:
            pass
        else:
            # print(self.inited)
            ims = internal_modules
            efs = external_folders
            if len(ims) > 0:
                for i, im_path in enumerate(ims):
                    im_package = '.'.join(im_path.split('/'))
                    if self.internal_dir is None:
                        # code = f'from damei.nn.{im_package} import *'
                        code = f'import damei.nn.{im_package}'
                        # code = f'import damei.nn.{im_package}.__init__'
                        exec(code)
                    else:
                        if self.internal_dir not in sys.path:
                            sys.path.append(self.internal_dir)
                        # internal_dir = '.'.join(self.internal_dir.split('/'))
                        # code = f'from {internal_dir} import {im_package}'
                        code = f'import {im_package}'
                        # print(sys.path)
                        # print(code)
                        exec(code)

            if len(efs) > 0:
                dirs = [x for x in os.listdir('.') if os.path.isdir(x)]  # 当前路径文件夹
                for dmp in dirs:
                    if dmp in efs:
                        # code = f'from repos.xxx.dmapi import *'
                        code = f'from {dmp} import *'
                        exec(code)
            self.inited = True
