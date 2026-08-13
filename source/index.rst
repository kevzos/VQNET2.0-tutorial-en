VQNet
=================================

VQNet 核心特性
------------------------

多平台兼容与跨环境支持
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

VQNet 支持用户在多种硬件和操作系统环境下进行量子机器学习的研发。无论是在 CPU 还是 GPU 上进行量子计算模拟，还是通过本源量子云服务调用真实量子芯片，VQNet 都能提供无缝支持。目前，VQNet 兼容 Windows、Linux 和 macOS 系统上的 Python 3.10。

完美的接口设计与易用性
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

VQNet 使用 Python 作为前端语言，提供类似 PyTorch 的函数接口，并可自由选择多种计算后端来实现经典量子机器学习模型的自动微分功能。框架内置了：100 多个常用的 Tensor 计算接口、100 多个量子变分电路计算接口和 50 多个经典神经网络接口。这些接口涵盖了从经典机器学习到量子机器学习的完整开发流程，并将持续更新。

高效的计算性能与扩展能力
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- **真实量子芯片实验支持**：对于需要进行真实量子芯片实验的用户，VQNet 集成了原始 pyQPanda 接口，并结合原始 Sinan 的高效调度能力，实现快速的量子电路模拟计算和真实芯片运行。
- **本地计算优化**：对于本地计算需求，VQNet 提供基于 CPU 或 GPU 的量子机器学习编程接口，并使用自动微分技术进行量子变分电路梯度计算，比参数漂移方法显著更快。详情请参见 :ref:`benchmarks`。
- **分布式计算支持**：VQNet 支持基于 MPI 的分布式计算，可在多个节点上实现大规模混合量子-经典神经网络模型的训练功能。

丰富的应用场景与示例支持
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

VQNet 不仅是一个强大的开发工具，还在公司内部的多个项目中得到广泛应用，包括电力优化、医疗数据分析、图像处理等领域。为了帮助用户快速上手，VQNet 在官方网站和 API 在线文档中提供了从基础教程到高级应用的多种场景。这些资源使用户能够轻松了解如何使用 VQNet 解决实际问题，并快速构建自己的量子机器学习应用。

.. toctree::
    :caption: 安装指南
    :maxdepth: 2

    rst/install.rst

.. toctree::
    :caption: 动手示例
    :maxdepth: 2

    rst/vqc_demo.rst
    rst/qml_demo.rst

.. toctree::
    :caption: 经典神经网络 API
    :maxdepth: 2

    rst/QTensor.rst
    rst/nn.rst
    rst/utils.rst

.. toctree::
    :caption: 集成 pyqpanda 的 QNN API
    :maxdepth: 2

    rst/qnn.rst
    rst/qnn_pq3.rst

.. toctree::
    :caption: 自动微分 QNN API
    :maxdepth: 2

    rst/vqc.rst

.. toctree:: 
    :caption: 其他
    :maxdepth: 2 
    
    rst/torch_api.rst
    rst/vqnet_dist.rst
    rst/FAQ.rst 
    rst/CHANGELOG.rst
