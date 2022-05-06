"""
用pyinstaller把程序它打包成程序，源码封装成so,索引环境中的所有库，可以脱离配置环境运行。
"""

import os
import shutil

import damei as dm


class Packing(object):
    def __init__(self):
        self.app_name = 'mmlabelme_client'
        self.entrance = './mmlabelme/__main__.py'
        self.resources = []  # 可以为空
        self.resources = ['./mmlabelme/config', './mmlabelme/icons']

    def __call__(self, *args, **kwargs):
        app_name = self.app_name
        entrance = os.path.abspath(self.entrance)

        main_path = os.path.abspath('.')
        software_path = f'{main_path}/software'
        system = dm.current_system()
        software_path += f'/{system}'
        if not os.path.exists(software_path):
            os.makedirs(software_path)

        os.chdir(software_path)
        # os.system('rm -rf *')

        # 打包
        os.system(f'pyinstaller -n {app_name} {entrance}')

        # 删除build和spec文件
        os.system(f'rm -rf ./build')
        os.system(f'rm {app_name}.spec')

        # 创建app索引
        if system == 'linux':
            os.system(f'ln -s {software_path}/dist/{app_name}/{app_name} ./{app_name}')

        # 拷贝资源项
        for i, resource in enumerate(self.resources):
            print(f'\rCopy resources ... [{i + 1}/{len(self.resources)}]', end='')

            resource = resource[2::] if resource.startswith('./') else resource
            sp = f'{main_path}/{resource}'
            tp = f'{software_path}/dist/{app_name}/{resource}'
            if not os.path.exists(tp):
                # print(f'mkdir: {tp}')
                os.makedirs(tp)

            files = [x for x in os.listdir(sp) if not x.endswith('.py')]
            files = [x for x in files if os.path.isfile(f'{sp}/{x}')]
            # print(sp, tp, len(files))
            for j, file in enumerate(files):
                shutil.copy(f'{sp}/{file}', f'{tp}/{file}')


if __name__ == '__main__':
    p = Packing()
    p()
