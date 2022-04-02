import damei as dm

# 1. 测试PyConfig
cfg_file = f'sources/test_cfg.py'
cfg = dm.PyConfig(cfg_file)
print('cfg', cfg)
print(cfg.info())
cfg['im_cfg_test']['ac'] = 2
print(cfg.info())
cfg_file2 = f'sources/test_cfg2.py'
cfg2 = dm.PyConfig(cfg_file2)
cfg.merge(cfg2)
print(cfg.info())
