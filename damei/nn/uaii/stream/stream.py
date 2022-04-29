"""
基础的流类
"""
import sys

import numpy as np
import collections
import copy
import damei as dm
# from ..utils.config_loader import PyConfigLoader
from damei.nn.api.utils import Config

logger = dm.getLogger('stream')


class Stream(object):
    name = 'stream_name'
    status = 'stopped'
    description = 'stream description'

    def __init__(self, parent, id=None, stream_cfg=None, **kwargs):
        self.is_mono = kwargs.pop('is_mono', False)
        self.name = stream_cfg['type']
        self.status = Stream.status
        self.description = stream_cfg.get('description', Stream.description)
        self.id = id
        self._cfg = stream_cfg
        self.modules_list_config = stream_cfg['models']  # list，每个元素是dict，是模块的配置，type和cfg
        # print('初始化', parent)
        self.parent = parent  # uaii
        self._modules, self._module_cfgs = self.init_about_models()  # 未初始化的模块和对应的配置
        self._inited_modules = collections.OrderedDict()  # 已经初始化的模块
        # self.register_attrs()
        # print('xx', self.seyolov5)

    def __repr__(self):
        format_str = f"<class '{self.__class__.__name__}'> " \
                     f"(name='{self.name}', status='{self.status}', " \
                     f"include={self.include(format=list)}, " \
                     f"description='{self.description}')"
        return format_str

    def init_about_models(self):
        """
        init this stream，read configs, DO NOT init modules.
        :return:
            ordered dict of modules: key is module name, value is module class (not inited)
            ordered dict of module configs: key is module name, value is module config: Config object.
        """
        _modules = collections.OrderedDict()
        _module_cfgs = collections.OrderedDict()
        for i, module in enumerate(self.modules_list_config):
            mname = module['type']  # 必须存在
            model_cfg = module.pop('cfg', None)  # 可以为无，默认为None
            module = copy.deepcopy(
                self.parent.get_module(mname, cls='MODULE', module=True))  # copy modeul from registry
            _modules[mname] = module  # 存入模块
            setattr(self, mname, module)  # 存入属性, self.module_name = module
            # cfg = PyConfigLoader(module.default_cfg)  # 这是默认配置
            cfg = Config(module.default_cfg)

            if model_cfg:  # 如果外部设置流的同时指定了配置文件，合并配置
                full_cfg_path = f'{self.parent.root_path}/{model_cfg}'
                # cfg2 = PyConfigLoader(cfg_file=full_cfg_path, root_path=self.parent.root_path)
                cfg2 = Config(cfg_file=full_cfg_path, root_path=self.parent.root_path)
                cfg = cfg.merge(cfg2)
            _module_cfgs[mname] = cfg

        return _modules, _module_cfgs

    def __call__(self, input, *args, **kwargs):
        """单条目运行"""
        # 需要把input封装成mi对象
        self.parent.run_stream(
            self,
            run_in_main_thread=True,
            mtask='infer',
            mi=input,
        )

    def register_attrs(self):
        """把包含的子模块注册为本流的属性"""
        for i, module in enumerate(self.stream_models):
            mname = module['type']
            mtask = module.get('task')
            module = self.parent.get_module(mname, cls='MODULE', module=True)
            setattr(self, mname, module)

    @property
    def cfg(self):
        if self.is_mono:
            # model_names = self.model_names
            return self.get_cfg(addr=f'/{self.model_names[0]}')

        return self._cfg

    @property
    def models(self):
        if self.status == 'stopped':
            return self._modules
        else:
            return self._inited_modules

    @property
    def model_cfgs(self):
        return self._module_cfgs

    @property
    def model_names(self):
        module_names = [x['type'] for x in self.modules_list_config]
        # print(self.models)
        # sys.exit(1)
        return module_names

    def get_cfg(self, addr='/'):
        """
        获取配置
        :param addr: 配置路径, 例如：/带代表stream自身配置，/seyolov5代表流的seyolov5模块的配置，/seyolov5/input_stream代表流的模块的input_stream的配置
        :return: 配置dict
        """
        if addr == '/':  # 全部配置
            return self.cfg
        else:  # 子模块配置
            attrs = [x for x in addr.split('/') if x != '']
            assert len(attrs) >= 1
            mname = attrs.pop(0)  # module name
            cfg = self.model_cfgs[mname]
            """
            m = self.models[mname]
            if m.status == 'stopped':
                cfg = self.model_cfgs[mname]
            else:  # ready running
                cfg = m.cfg  # 可能会merge之后的也是PyConfigLoader对象
            """

            for sub_attr in attrs:
                cfg = cfg[sub_attr]
            return cfg

    def set_cfg(self, addr='/', value=None):
        """
        更新配置
        :param addr:
        :return:
        """
        assert value is not None
        if addr == '/':
            raise NotImplementedError(f"You cannot set config of the stream")
        else:
            attrs = [x for x in addr.split('/') if x != '']
            assert len(attrs) >= 1
            mname = attrs.pop(0)
            m = self.models[mname]
            if m.status != 'stopped':
                logger.warn(f"Please stop the stream: '{self.name}' while configure it's module.")
                return
            mcfg = self.model_cfgs[mname]
            mcfg.update_item(attrs, value)
            self.model_cfgs[mname] = mcfg
            # print(f'newcfg: {mcfg.info()}')

    def ps(self, *args, **kwargs):
        return self.info(*args, **kwargs)

    def info(self, *args, **kwargs):
        format_str = ''
        info = dict()
        info['class'] = f'<class "{self.__class__.__name__}">'
        info['name'] = self.name
        info['description'] = self.description
        info['status'] = self.status
        info['is_mono'] = self.is_mono

        sub_modules = self.include()
        sub_modules_ids = [self.parent.modules.name2id_dict[x] for x in sub_modules]
        info['models'] = dict(zip(sub_modules_ids, sub_modules))

        models = self.models
        model_cfgs = self.model_cfgs
        for i, (mname, model) in enumerate(models.items()):
            # mname = module['type']
            # mtask = module.get('task', None)
            id = self.parent.modules.name2id_dict[mname]
            cfg = self.model_cfgs[mname]

            # m = self.parent.get_module(name=mname)
            # if m.status == 'stopped':
            #    cfg = PyConfigLoader(m.default_cfg)
            # else:  # ready running
            #    cfg = m.cfg  # 可能会merge之后的也是PyConfigLoader对象

            info[f'{mname} config'] = cfg.items

        format_str += self.dict2info(info)
        # format_str += dm.misc.dict2info(info)
        return format_str

    def dict2info(self, info_dict):
        # 先递归地展开
        new_info_dict = collections.OrderedDict()
        indents = []
        indent_space = 4
        for k, v in info_dict.items():
            self.recursive_func(k, v, new_info_dict, indents, indent_space=indent_space)
        info_dict = new_info_dict

        indents2color = {0: '32m', 1 * indent_space: '35m', 2 * indent_space: '36m', 3 * indent_space: '33m'}
        lenth = np.max([len(x) for x in info_dict.keys()])
        format_str = ''
        for i, (k, v) in enumerate(info_dict.items()):
            indent = indents[i]
            # print(f'k: {k} {len(k)}')
            k_str = f'{k.strip("*"):>{indent + lenth}}'
            color = indents2color.get(indent, None)
            k_str = f'\033[1;{color}{k_str}\033[0m' if color else k_str  # 上色
            format_str += f'  {k_str} : {v}\n'
        return format_str

    def recursive_func(self, k, v, new_info_dict, indents, indent=0, indent_space=4):
        """递归拆分，原来的字典里，某些值可能还是dict，为了方便显示，递归展开，子dict的每个键作为新的顶层的键"""
        if isinstance(v, dict):
            while True:
                if k not in new_info_dict.keys():
                    break
                k = '*' + k
            new_info_dict[f'{k}'] = f'({len(v.keys())})'
            indents.append(indent)
            indent += indent_space
            for k2, v2 in v.items():
                # print(f'k2: {k2} indent: {indent}', new_info_dict.keys())
                self.recursive_func(k2, v2, new_info_dict, indents, indent, indent_space=indent_space)
        else:
            while True:
                if k not in new_info_dict.keys():
                    break
                k = '*' + k
            new_info_dict[f'{k}'] = v
            indents.append(indent)
        return new_info_dict

    def include(self, format=list):
        if format == list:
            return [x['type'] for x in self.modules_list_config]
        else:
            return ' '.join([x['type'] for x in self.modules_list_config])

    def init(self, stream_cfg=None, **kwargs):
        """初始化流内模块
        初始化流的意思的：对流内每个模块进行初始化，并配置好模块的I/O
        模块可能的状态有：stopped, ready, running.
        stopped时，直接初始化
        ready或running时，如果force_init=True, 先停止，再初始化
                如果force_init=False, 则不初始化
        """
        # TODO: 通用的monoflow和multiflow的传入stream_cfg的是实现方法
        logger.info(f'Initializing...')
        force_init = stream_cfg is not None

        if self.status != 'stopped':
            if force_init:
                self.stop()
            else:
                logger.warning(f'Stream {self.name} is not stopped, skip initialization')
                return

        if self.status == 'stopped':
            models = self.models
            model_cfgs = self.model_cfgs
            for i, (mname, model) in enumerate(models.items()):
                model_cfg = model_cfgs[mname]

                # print(model.status, force_init, self.status)
                model = model(cfg=model_cfg)

                mi = self.parent.build_io_instance('input', model_cfg)
                mo = self.parent.build_io_instance('output', model_cfg)

                if mi is not None:
                    if hasattr(model, 'mi'):
                        delattr(model, 'mi')
                    setattr(model, 'mi', mi)
                if mo is not None:
                    if hasattr(model, 'mo'):
                        delattr(model, 'mo')
                    setattr(model, 'mo', mo)

                if hasattr(self, mname):
                    delattr(self, mname)
                setattr(self, mname, model)

                if i + 1 == len(models):  # 最后一个模块
                    # print(model.mo)
                    if hasattr(self, 'mo'):
                        delattr(self, 'mo')
                    setattr(self, 'mo', model.mo)

                # self._modules[mname] = model  # 初始化后赋值回去
                self._inited_modules[mname] = model
        self.status = 'ready'

    def pop(self, wait=True, timeout=None):
        return self.mo.pop(wait=wait, timeout=timeout)

    def get_last(self, wait=True, timeout=None):
        return self.mo.get_last(wait=wait, timeout=timeout)

    def scan(self, last=False):
        return self.mo.scan(last=last)

    def run(self, **kwargs):
        """从stopped运行流"""
        return self.parent.run(self, **kwargs)  # run in main thread

    def start(self, **kwargs):
        return self.parent.run_stream(self, run_in_main_thread=False, **kwargs)

    def evaluate(self, **kwargs):
        return self.parent.evaluate(self, run_in_main_thread=True, **kwargs)

    def train(self, **kwargs):
        return self.parent.train(self, run_in_main_thread=True, **kwargs)

    def stop(self):
        if self.status == 'stopped':
            return
        del self._inited_modules  # 引用计数置0，释放内存
        self._inited_modules = collections.OrderedDict()
        self.status = 'stopped'  # 设置成stopped后, self.modules的值就是未出初始化过的模型了
