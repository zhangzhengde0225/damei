#!/home/zzd/anaconda3/envs/yolov5/bin/python
# coding=utf-8
"""
###!/Anaconda3/envs/yolov5/bin/python
damei scp 配置好目标机器，同步文件
usage:
    # 上传
    dmscp siyuan -u xxx.zip --rdir /path/to/remote_server_dir
    dmscp siyuan -d ./xxx.zip -d # 下载 -d --download
    dmscp server list  #
    dmscp config_server
"""
import argparse
import os
from pathlib import Path

import paramiko

import damei as dm

logger = dm.getLogger('dmscp', name_lenth=5)


class RemoteServer(object):
    """远程服务器"""

    def __init__(self, name, host, user, passwd=None, port=22, key_file=None):
        self.name = name
        self.host = host
        self.port = port
        self.user = user
        self.passwd = passwd
        self.key_file = key_file
        self.ssh = None

    def __repr__(self):
        passwd = '*********' if self.passwd else 'No_passwd'
        key_file = self.key_file if self.key_file else ''
        format_str = f'{self.user}@{self.host}:{self.port} {passwd} {key_file}'
        return format_str

    def _init_ssh(self):
        ssh = paramiko.SSHClient()
        ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        self.ssh = ssh

    def exec(self, cmd):
        assert self.ssh is not None
        stdin, stdout, stderr = self.ssh.exec_command(cmd)
        res, err = stdout.read(), stderr.read()
        result = res if res else err
        # print(result.decode('utf-8'))
        return result.decode('utf-8')

    def connect(self):
        """连接远程服务器"""
        if self.ssh is None:
            self._init_ssh()
        try:
            self.ssh.connect(self.host, self.port, self.user, self.passwd)
        except Exception as e:
            logger.error(f'SSH connect to server {self.user}@{self.host} port: {self.port} failed：{e}')
            # print(self.passwd)
            return False
        return True

    def disconnect(self):
        """断开远程服务器"""
        self.ssh.close()


class DmSCP(object):
    def __init__(self):
        self._default_server = None
        self._servers = self._init_server()

    def _init_server(self):
        home = Path.home()
        rc_file = f'{home}/.damei/.dmscprc'
        assert os.path.exists(rc_file), f'{rc_file} not exists'
        data_dict = dm.misc.rcfile2dict(rc_file)  # 读取rcfile

        _servers = {}

        for k, v in data_dict.items():
            if k == 'default':
                self._default_server = v
            else:
                passwd = v.get('passwd', None)
                port = v.get('port', 22)
                key_file = v.get('key_file', None)
                _servers[k] = RemoteServer(k, v['host'], v['user'], passwd=passwd, port=port, key_file=key_file)
        if self._default_server is None:  # 还是None
            self._default_server = list(_servers.keys())[0]  # 默认第一个
        return _servers

    def __call__(self, *args, **kwargs):
        # print(args, type(args))
        print('args:', args, kwargs)
        first_arg = args[0] if len(args) > 0 else None
        if first_arg == 'servers':
            info = dm.misc.dict2info(self.servers)
            print(info)
        elif first_arg == 'server_names':
            print(self.server_names)
        else:  # 默认上传
            # upload = kwargs.get('upload', False)
            # download = kwargs.get('download', False)

            # 第一个参数是上传文件时
            # server_name = first_arg if first_arg else self.default_server
            if first_arg in self.server_names:
                server_name = first_arg
            else:
                assert not upload, f'{first_arg} not in server_names'
                server_name = self.default_server
                # upload = first_arg
                kwargs['upload'] = first_arg

            # print('xxx', first_arg, args, kwargs, upload)
            if kwargs.get('upload', False):
                self.up_down(server_name, **kwargs)
            elif download:
                self.up_down(server_name, **kwargs)
            else:
                logger.info(f'No upload or download, -u (default) or -d to specify')
                # help = dm.popen(f'dmscp {server_name} --help')
            # print(server_name)
        pass

    @property
    def servers(self):
        return self._servers

    @property
    def server_names(self):
        names = list(self._servers.keys())
        return names

    @property
    def default_server(self):
        return self._default_server
        # return list(self._servers.keys())[0]
        # return self._servers['siyuan']

    def up_down(self, server_name, *args, **kwargs):
        """
        上传或下载
        :param server_name:
        :param source: 文件或文件夹
        """
        upload = kwargs.get('upload', None)
        download = kwargs.get('download', None)
        recursive = kwargs.get('recursive', False)
        force = kwargs.get('force', False)
        remote_dir = kwargs.get('remote_dir', None)

        print(upload, download, recursive, force, remote_dir)
        assert not (upload and download), '-u and -d can not be used together'
        assert upload or download, '-u or -d must be used'
        source = upload if upload else download
        up_or_down = 'upload' if upload else 'download'

        # 获取服务器
        s = self.servers[server_name]
        # server.connect()  # 暂时不需要连接

        # 处理source
        src = Path(os.path.abspath(source))
        # 处理递归参数
        recursive = '-r ' if recursive else ''
        # 处理remote dir
        local_home = f'{Path(os.path.expanduser("~"))}'
        local_dir = src.parent
        remote_dir = remote_dir if remote_dir else f'{src.parent}'
        remote_dir = remote_dir.replace(local_home, '~') if remote_dir else None  # 必须是~/xx的形式

        port = s.port
        # code = f'scp {recursive}-P {port} {src} {s.user}@{s.host}:~{rel_dir}/{src.name}'
        if upload:
            code = f'scp {recursive}-P {port} {src} {s.user}@{s.host}:{remote_dir}/{src.name}'
        else:  # download
            code = f'scp {recursive}-P {port} {s.user}@{s.host}:{remote_dir}/{src.name} {src}'

        # print(code)
        logger.info(code)
        # print(src_dir)
        # print(server_name, 'xx')
        # exec(code)
        ret = dm.popen(code)  # 传输成功返回[]，失败可能是：['scp: xxx/misc.py: No such file or directory']

        if len(ret) > 0:
            # logger.warn(f'Upload failed: {ret[0]}')
            if force:
                logger.info(f'Force {up_or_down} ...')
                # 连接远程服务器，创建目录，重新上传
                is_connected = s.connect()
                if is_connected:
                    s.exec(f'mkdir -p {remote_dir}')
                    self.up_down(server_name, *args, **kwargs)
            else:
                logger.error(f'{up_or_down} failed: {ret[0]}, you may use -f to force {up_or_down}.')
                # raise ValueError(f'Upload failed: {ret}')
        else:
            logger.info(f'{up_or_down} {source} success.')

    def download(self, src, dst):
        pass

    def config_server(self, server):
        pass


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='damei scp')

    parser.add_argument('args', nargs='*', default=None, help='servers,  ')  # 或多个参数
    parser.add_argument('-u', '--upload', type=str, default=None, help='upload')
    parser.add_argument('-d', '--download', type=str, default=None, help='download')
    parser.add_argument('-r', '--recursive', action='store_true', default=False, help='recursive copy all sub files')
    parser.add_argument('-f', '--force', action='store_true', default=False,
                        help='force to copy by makedirs in remote if the remote dir not exist')
    parser.add_argument('-rdir', '--remote_dir', type=str, default=None,
                        help='remote dir for save file, default is the same as local dir')

    opt = parser.parse_args()
    print(opt)
    dmscp = DmSCP()

    args = opt.args
    upload = opt.upload
    download = opt.download

    dmscp(*args, upload=upload, download=download, recursive=opt.recursive,
          force=opt.force, remote_dir=opt.remote_dir,
          )
