damei.nn
========

The damei.nn is easy to use neural network library.

Features
---------
+ 提供可见光、红外、雷达三个模态的相关ai算法的统一调用接口，柔性可扩展
+ 提供算法模型的推理、训练、评估、可视化等功能，可实现算法模型自学习和自演进
+ 提供算法模块处理流程自定义、参数配置自定义
+ 提供xai作为python库使用的SDK接口
+ 提供xai作为微服务部署到中心平台的gRPC接口，可实现分布式部署，支持跨系统、跨语言的调用
+ 支持一键功能点测试

+ 快速的模型训练、评估和推理

Usage
------
    >>> import damei as dm
    >>> xai = dm.nn.UAII()


