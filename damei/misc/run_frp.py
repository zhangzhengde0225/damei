import os
from pathlib import Path
# import damei as dm
import time
import threading
import argparse
import logging
import subprocess

# logger = logging.getLogger('run_frp')
pydir = Path(os.path.abspath(__file__)).parent


class StartupFrp(object):
    def __init__(self, frp_dir=None, s_or_c='frpc', cfg_file_name='frpc.ini'):
        self.frp_dir = frp_dir if frp_dir else os.getcwd()
        self.frp_path = f'{self.frp_dir}/{s_or_c}'
        assert os.path.exists(self.frp_path), f'{self.frp_path} not exists'
        self.cfg_file_name = cfg_file_name
        # self.frp_path = frp_path if frp_path else f'{os.getcwd()}/frpc'
        self.host = self.read_host()
        print('Starting Frp ...')
        print(f'Frp_path     : {self.frp_path}')
        print(f'Cfg_file_name: {self.cfg_file_name}')
        print(f'Remote host  : {self.host}')

    def read_host(self, ):
        cfg_file = f'{Path(self.frp_path).parent}/{self.cfg_file_name}'
        with open(cfg_file, 'r') as f:
            lines = f.readlines()
        host = [x for x in lines if 'server_addr' in x]
        host = host[0].split('=')[1].strip()
        return host

    def popen(self, code, need_error=False):
        out = subprocess.Popen(code, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        # print(code)
        if need_error:
            output = out.stdout.read().decode('utf-8').replace('\r', '').split('\n')
            error = out.stderr.read().decode('utf-8').replace('\r', '').split('\n')
            output = output[:-1] if output[-1] == '' else output
            error = error[:-1] if error[-1] == '' else error
            return output, error
        else:
            output = out.stdout.read().decode('utf-8').replace('\r', '').split('\n')
            output = output[:-1] if output[-1] == '' else output
            return output

    def ping(self):
        """判断是否有网络连接"""
        rett = self.popen(f'ping -w 1 -c 1 {self.host}')
        received_bytes = [x for x in rett if '64 bytes from ' in x]
        if len(received_bytes) > 0:
            print(f'Ping {self.host} success')
            return True
        else:
            print(f'Ping {self.host} failed')
            return False

    def run_frp(self, timeout=20, nohup=False):
        cfg_file_name = self.cfg_file_name
        ctime = time.time()
        while True:
            connected = self.ping()  # 直到ping通
            if connected:
                break
            if (time.time() - ctime) > timeout:
                break
            print(f'Waiting for {self.host}, time: {time.time()}')
            time.sleep(1)
        if connected:
            print(f'Start frp')
            if nohup:
                code = f'nohup {self.frp_path} -c {Path(self.frp_path).parent}/{cfg_file_name} &'
            else:
                code = f'{self.frp_path} -c {Path(self.frp_path).parent}/{cfg_file_name}'
            os.system(code)
        else:
            print(f'{self.host} is not connected, do not start frp')

    def __call__(self, run_in_main_thread=False, nohup=False, timeout=20, *args, **kwargs):
        # print(run_in_main_thread, nohup, timeout)
        if run_in_main_thread:
            print('Run in main thread')
            self.run_frp(nohup=nohup, timeout=timeout)
        else:
            print('Run in sub thread')
            thread = threading.Thread(target=self.run_frp, args=(timeout, nohup), daemon=True)
            thread.start()  # 开启线程，父进程结束


if __name__ == '__main__':
    paser = argparse.ArgumentParser()
    paser.add_argument('--frp-dir', type=str, default=None, help='frp dir, default is current dir')
    paser.add_argument('--s-or-c', type=str, default='frpc', help='s: frps, c: frpc, default is frpc')
    paser.add_argument('--cfg-file-name', type=str, default='frpc.ini',
                       help='frpc.ini for frpc or frpc, default is frpc.ini')
    paser.add_argument('--run-in-main-thread', action='store_true', default=False)
    paser.add_argument('--nohup', action='store_true', default=False)
    paser.add_argument('--timeout', type=int, default=20)
    opt = paser.parse_args()
    print(f'opt: {opt}')
    startup = StartupFrp(frp_dir=opt.frp_dir, cfg_file_name=opt.cfg_file_name, s_or_c=opt.s_or_c)
    startup(run_in_main_thread=opt.run_in_main_thread, nohup=opt.nohup, timeout=opt.timeout)
