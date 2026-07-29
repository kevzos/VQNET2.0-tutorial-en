

.. _torch_api:

=============================================================
VQNet 使用 torch 进行底层计算
=============================================================

从 2.15.0 版本开始，本软件支持使用 `torch` 作为底层计算的运算后端，并可基于 `torch` 的模型、代码及第三方库进行集成二次开发。

    .. important::

        要使用以下功能，请自行安装 torch>=2.11.0。如果安装 GPU 版本的 torch，需要使用与 CUDA 12.6 兼容的版本，否则由于 NVIDIA CUDA 运行时库的问题，您的 torch 可能无法正常工作。本软件在安装时不会自动安装 torch。

    .. note::

        :ref:`vqc_api` 中的变分量子计算函数（小写命名方式，如 `rx`\ 、`ry`\ 、`rz` 等）以及 :ref:`qtensor_api` 中的 QTensor 基本计算函数，
        在调用 ``pyvqnet.backends.set_backend("torch")`` 后可以接收 `QTensor` 作为输入，其中 `QTensor` 的 `data` 成员从 pyvqnet 的 Tensor 变为 ``torch.Tensor`` 进行计算。

        ``pyvqnet.backends.set_backend("torch")`` 和 ``pyvqnet.backends.set_backend("pyvqnet")`` 会修改全局计算后端。
        在不同后端配置下创建的 ``QTensor`` 对象不能混合用于计算。

基本后端配置
============================================

set_backend
------------------------------------------------

.. py:function:: pyvqnet.backends.set_backend(backend_name)

    设置当前计算和数据存储的后端，默认为 "pyvqnet-ad"，可设置为 "torch"、"torch-native"、"pyvqnet-ad"。

    调用 ``pyvqnet.backends.set_backend("torch")`` 后，接口保持不变。VQNet 的 ``QTensor`` ``data`` 成员变量全部使用 ``torch.Tensor`` 存储数据。
    :ref:`qtensor_api`\ 、:ref:`vqc_api` 和 ``pyvqnet.nn.torch`` 接口接收 ``QTensor`` 作为输入，并输出 ``QTensor``\ 。

    调用 ``pyvqnet.backends.set_backend("torch-native")`` 后，接口保持不变：:ref:`qtensor_api`\ 、:ref:`vqc_api` 和 `pyvqnet.nn.torch` 接口。
    输入可直接接收 ``torch.Tensor`` 或 ``QTensor`` 类型，输出为 ``torch.Tensor``\ ，无需转换为 ``QTensor``\ ，从而减少数据转换。

    调用 ``pyvqnet.backends.set_backend("pyvqnet")`` 后，VQNet 的 ``QTensor`` 的 ``data`` 成员将使用 ``pyvqnet._core.Tensor`` 存储数据，计算将使用 pyvqnet C++ 库。

    调用 ``pyvqnet.backends.set_backend("pyvqnet-ad")`` 后，VQNet 的 ``QTensor`` 的 ``data`` 成员将使用 ``pyvqnet._core.Tensor`` 存储数据，计算将使用 pyvqnet C++ 库，且性能得到提升。


    .. note::

        此函数会修改当前计算后端。在不同后端下创建的 ``QTensor`` 对象不能混合用于计算。

    :param backend_name: 后端的名称，可以是 "pyvqnet" 或 "torch"。

    示例::

        import pyvqnet
        pyvqnet.backends.set_backend("torch")

get_backend
-------------------------------

.. py:function:: pyvqnet.backends.get_backend(t=None)

    如果 `t` 为 None，则获取当前计算后端。
    如果 `t` 是一个 QTensor，则根据其 ``data`` 属性返回创建该 QTensor 时使用的后端。
    如果后端是 "torch"，则返回 pyvqnet torchAPI 后端。
    如果后端是 "pyvqnet"，则直接返回 "pyvqnet"。

    :param t: 当前张量（默认值：None）。
    :return: 后端名称。默认返回 "pyvqnet"。

    示例::

        import pyvqnet
        pyvqnet.backends.set_backend("torch")
        pyvqnet.backends.get_backend()

QTensor 函数
===================

将后端设置为 ``torch`` 后：

.. code-block::

    import pyvqnet
    pyvqnet.backends.set_backend("torch")

:ref:`qtensor_api` 下的所有成员函数、创建函数、数学函数、逻辑函数、矩阵变换等将使用 torch 进行计算。可通过访问 `QTensor.data` 获取 torch 数据。

经典神经网络与变分量子神经网络模块
==========================================================================================

基类
------------------------------------------------

TorchModule
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. py:class:: pyvqnet.nn.torch.TorchModule(*args, **kwargs)

    使用 `torch` 后端时定义模型的基类。此类同时继承自 ``pyvqnet.nn.Module`` 和 ``torch.nn.Module``\ 。
    可作为子模块添加到 torch 模型中。

    .. note::

        此类及其派生类仅适用于 ``pyvqnet.backends.set_backend("torch")``\ 。
        不要与默认的 ``pyvqnet.nn`` 中的 `Module` 混用。

        此类的 ``_buffers`` 中的数据为 ``torch.Tensor`` 类型。
        此类的 ``_parameters`` 中的数据为 ``torch.nn.Parameter`` 类型。

    .. py:method:: pyvqnet.nn.torch.TorchModule.forward(x, *args, **kwargs)

        TorchModule 类的抽象前向计算函数。

        :param x: 输入 QTensor。
        :param args: 非关键字可变参数。
        :param kwargs: 关键字可变参数。

        :return: 输出 QTensor，其内部 `data` 为 ``torch.Tensor``\ 。

        示例::

            import numpy as np
            from pyvqnet.tensor import QTensor
            import pyvqnet
            pyvqnet.backends.set_backend("torch")
            from pyvqnet.nn.torch import Conv2D
            b = 2
            ic = 3
            oc = 2
            test_conv = Conv2D(ic, oc, (3, 3), (2, 2), "valid")
            x0 = QTensor(np.arange(1, b * ic * 5 * 5 + 1).reshape([b, ic, 5, 5]),
                        requires_grad=True,
                        dtype=pyvqnet.kfloat32)
            x = test_conv.forward(x0)
            print(x)

    .. py:method:: pyvqnet.nn.torch.TorchModule.state_dict(destination=None, prefix='')

        返回包含模块完整状态的字典，包括参数和缓冲区的值。
        键为对应参数和缓冲区的名称。

        :param destination: 用于存储模块内部参数的字典。
        :param prefix: 参数和缓冲区名称的前缀。

        :return: 包含模块完整状态的字典。

        示例::

            from pyvqnet.nn.torch import Conv2D
            import pyvqnet
            pyvqnet.backends.set_backend("torch")
            test_conv = Conv2D(2,3,(3,3),(2,2),"same")
            print(test_conv.state_dict().keys())

    .. py:method:: pyvqnet.nn.torch.TorchModule.load_state_dict(state_dict, strict=True)

        将 :attr:`state_dict` 中的参数和缓冲区复制到此模块及其子模块中。

        :param state_dict: 包含参数和持久化缓冲区的字典。
        :param strict: 是否强制 state_dict 中的键与模型的 `state_dict()` 匹配。默认值：True。

        :return: 如果有问题则返回错误信息。

        示例::

            from pyvqnet.nn.torch import TorchModule, Conv2D
            import pyvqnet
            import pyvqnet.utils
            pyvqnet.backends.set_backend("torch")

            class Net(TorchModule):
                def __init__(self):
                    super(Net, self).__init__()
                    self.conv1 = Conv2D(input_channels=1, output_channels=6, kernel_size=(5, 5),
                        stride=(1, 1), padding="valid")

                def forward(self, x):
                    return super().forward(x)

            model = Net()
            pyvqnet.utils.storage.save_parameters(model.state_dict(), "tmp.model")
            model_param = pyvqnet.utils.storage.load_parameters("tmp.model")
            model.load_state_dict(model_param)

    .. py:method:: pyvqnet.nn.torch.TorchModule.toGPU(device: int = DEV_GPU_0)

        将模块及其子模块的参数和缓冲区数据移动到指定的 GPU 设备。

        设备指定内部数据的存储位置。当 device >= DEV_GPU_0 时，数据存储在 GPU 上。
        如果您的计算机有多个 GPU，可以指定不同的设备来存储数据。例如，device = DEV_GPU_1、DEV_GPU_2、DEV_GPU_3……分别对应存储在不同序号的 GPU 上。

        .. note::

            模块不能跨不同的 GPU 进行计算。
            如果您尝试在超过最大验证 GPU ID 的 GPU 上创建 QTensor，将会引发 Cuda 错误。

        :param device: 存储 QTensor 的设备。默认值：DEV_GPU_0。device = pyvqnet.DEV_GPU_0 存储在第一块 GPU，device = DEV_GPU_1 存储在第二块 GPU，依此类推。
        :return: 移动到 GPU 设备的 Module。

        示例::

            from pyvqnet.nn.torch import ConvT2D
            import pyvqnet
            pyvqnet.backends.set_backend("torch")
            test_conv = ConvT2D(3, 2, [4, 4], [2, 2], (0, 0))
            test_conv = test_conv.toGPU()
            print(test_conv.backend)
            #1000

    .. py:method:: pyvqnet.torch.TorchModule.toCPU()

        将模块及其子模块的参数和缓冲区数据移动到特定的 CPU 设备。

        :return: 移动到 CPU 设备的 Module。

        示例::

            from pyvqnet.nn.torch import ConvT2D
            import pyvqnet
            pyvqnet.backends.set_backend("torch")
            test_conv = ConvT2D(3, 2, [4, 4], [2, 2], (0, 0))
            test_conv = test_conv.toCPU()
            print(test_conv.backend)
            #0


TorchModuleList
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. py:class:: pyvqnet.nn.torch.TorchModuleList(modules = None)

    此模块用于在列表中存储子 ``TorchModule`` 实例。TorchModuleList 可以像常规 Python 列表一样被索引，并且其中包含的内部参数可以被保存。

    此类继承自 ``pyvqnet.nn.torch.TorchModule`` 和 ``pyvqnet.nn.ModuleList``\ ，可作为子模块添加到 torch 模型中。

    :param modules: ``pyvqnet.nn.torch.TorchModule`` 的列表

    :return: 一个 TorchModuleList 类

    示例::

        from pyvqnet.tensor import *
        from pyvqnet.nn.torch import TorchModule, Linear, TorchModuleList

        import pyvqnet
        pyvqnet.backends.set_backend("torch")

        class M(TorchModule):
            def __init__(self):
                super(M, self).__init__()
                self.pqc2 = TorchModuleList([Linear(4, 1), Linear(4, 1)])

            def forward(self, x):
                y = self.pqc2[0](x) + self.pqc2[1](x)
                return y

        mm = M()

TorchParameterList
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. py:class:: pyvqnet.nn.torch.TorchParameterList(value=None)

    此模块用于在列表中存储子 ``pyvqnet.nn.Parameter`` 实例。TorchParameterList 可以像常规 Python 列表一样被索引，并且其中包含的内部参数可以被保存。

    此类继承自 ``pyvqnet.nn.torch.TorchModule`` 和 ``pyvqnet.nn.ParameterList``\ ，可作为子模块添加到 torch 模型中。

    :param value: ``nn.Parameter`` 的列表

    :return: 一个 TorchParameterList 类

    示例::

        from pyvqnet.tensor import *
        from pyvqnet.nn.torch import TorchModule, Linear, TorchParameterList
        import pyvqnet.nn as nn
        import pyvqnet
        pyvqnet.backends.set_backend("torch")

        class MyModule(TorchModule):
            def __init__(self):
                super().__init__()
                self.params = TorchParameterList([nn.Parameter((10, 10)) for i in range(10)])

            def forward(self, x):
                # ParameterList 可以像可迭代对象一样使用，也可以用整数索引
                for i, p in enumerate(self.params):
                    x = self.params[i // 2] * x + p * x
                return x

        model = MyModule()
        print(model.state_dict().keys())

TorchSequential
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. py:class:: pyvqnet.nn.torch.TorchSequential(*args)

    该模块按传递的顺序添加模块。或者，您可以传递一个模块的 ``OrderedDict``\ 。``Sequential`` 类的 ``forward()`` 方法接受任何输入并将其转发给其第一个模块。
    然后输出依次链接到每个后续模块的输入，最终输出为最后一个模块的结果。

    此类继承自 ``pyvqnet.nn.torch.TorchModule`` 和 ``pyvqnet.nn.Sequential``\ ，可作为子模块添加到 torch 模型中。

    :param args: 要添加的模块

    :return: 一个 TorchSequential 类

    示例::

        import pyvqnet
        from collections import OrderedDict
        from pyvqnet.tensor import *
        from pyvqnet.nn.torch import TorchModule, Conv2D, ReLu, \
            TorchSequential
        pyvqnet.backends.set_backend("torch")
        model = TorchSequential(
                    Conv2D(1, 20, (5, 5)),
                    ReLu(),
                    Conv2D(20, 64, (5, 5)),
                    ReLu()
                )
        print(model.state_dict().keys())

        model = TorchSequential(OrderedDict([
                    ('conv1', Conv2D(1, 20, (5, 5))),
                    ('relu1', ReLu()),
                    ('conv2', Conv2D(20, 64, (5, 5))),
                    ('relu2', ReLu())
                ]))
        print(model.state_dict().keys())

保存和加载模型参数
--------------------------------------------

您可以使用 :ref:`save_parameters` 中的 ``save_parameters`` 和 ``load_parameters`` 将 ``TorchModule`` 模型的参数以字典形式保存到文件中，值保存为 `numpy.ndarray`\ 。或者，您可以从磁盘加载参数文件。请注意，模型结构不会保存在文件中，您需要手动重建模型结构。您也可以直接使用 ``torch.save`` 和 ``torch.load`` 读取 ``torch`` 模型参数，因为 ``TorchModule`` 的参数以 ``torch.Tensor`` 对象形式存储。


经典神经网络模块
--------------------------------------------

以下经典神经网络模块均继承自 ``pyvqnet.nn.Module`` 和 ``torch.nn.Module``\ ，可作为子模块添加到 torch 模型中。

Linear
^^^^^^^^^^^^^^^^^^^^^^^^

.. py:class:: pyvqnet.nn.torch.Linear(input_channels, output_channels, weight_initializer=None, bias_initializer=None, use_bias=True, dtype=None, name: str = "")

    线性模块（全连接层），:math:`y = x@A.T + b`\ 。
    此类继承自 ``pyvqnet.nn.Module`` 和 ``torch.nn.Module``\ ，可作为 torch 模型的子模块使用。

    类中的 ``_buffers`` 数据为 ``torch.Tensor`` 类型。
    类中的 ``_parameters`` 数据为 ``torch.nn.Parameter`` 类型。

    :param input_channels: `int` - 输入通道数。
    :param output_channels: `int` - 输出通道数。
    :param weight_initializer: `callable` - 权重初始化函数，默认为空，使用 he_uniform。
    :param bias_initializer: `callable` - 偏置初始化函数，默认为空，使用 he_uniform。
    :param use_bias: `bool` - 是否使用偏置项，默认为 True。
    :param dtype: 参数的数据类型，默认为 None，使用默认数据类型 `kfloat32`\ ，表示 32 位浮点数。
    :param name: 线性层的名称，默认为 ""。

    :return: Linear 层的一个实例。

    示例::

        import numpy as np
        import pyvqnet
        from pyvqnet.tensor import QTensor
        from pyvqnet.nn.torch import Linear
        pyvqnet.backends.set_backend("torch")
        c1 = 2
        c2 = 3
        cin = 7
        cout = 5
        n = Linear(cin, cout)
        input = QTensor(np.arange(1, c1 * c2 * cin + 1).reshape((c1, c2, cin)), requires_grad=True, dtype=pyvqnet.kfloat32)
        y = n.forward(input)
        print(y)

Conv1D
^^^^^^^^^^^^^^^^^^^^^^^^

.. py:class:: pyvqnet.nn.torch.Conv1D(input_channels: int, output_channels: int, kernel_size: int, stride: int = 1, padding = "valid", use_bias: bool = True, kernel_initializer = None, bias_initializer = None, dilation_rate: int = 1, group: int = 1, dtype = None, name = "")

    在输入上执行一维卷积。Conv1D 模块的输入形状为 (batch_size, input_channels, in_height)。
    此类继承自 ``pyvqnet.nn.Module`` 和 ``torch.nn.Module``\ ，可作为 torch 模型的子模块使用。

    类中的 ``_buffers`` 数据为 ``torch.Tensor`` 类型。
    类中的 ``_parameters`` 数据为 ``torch.nn.Parameter`` 类型。

    :param input_channels: `int` - 输入通道数。
    :param output_channels: `int` - 输出通道数。
    :param kernel_size: `int` - 卷积核大小。卷积核形状为 [output_channels, input_channels/group, kernel_size, 1]。
    :param stride: `int` - 步长，默认为 1。
    :param padding: `str|int` - 填充选项，可以是字符串 {'valid', 'same'} 或指定应用到输入的填充量的整数。默认为 "valid"。
    :param use_bias: `bool` - 是否使用偏置项，默认为 True。
    :param kernel_initializer: `callable` - 卷积核初始化方法。默认为空，使用 kaiming_uniform。
    :param bias_initializer: `callable` - 偏置初始化方法。默认为空，使用 kaiming_uniform。
    :param dilation_rate: `int` - 膨胀大小，默认为 1。
    :param group: `int` - 分组卷积中的组数。默认为 1。
    :param dtype: 参数的数据类型，默认为 None，使用默认数据类型 `kfloat32`\ ，表示 32 位浮点数。
    :param name: 模块的名称，默认为 ""。

    :return: 一维卷积的一个实例。

    .. note::

        ``padding='valid'`` 不进行填充。

        ``padding='same'`` 对输入进行零填充，输出的 `out_height` 等于 `ceil(in_height / stride)`\ ，不支持 `stride > 1`\ 。

    示例::

        import numpy as np
        import pyvqnet
        from pyvqnet.tensor import QTensor
        from pyvqnet.nn.torch import Conv1D
        pyvqnet.backends.set_backend("torch")
        b = 2
        ic = 3
        oc = 2
        test_conv = Conv1D(ic, oc, 3, 2)
        x0 = QTensor(np.arange(1, b * ic * 5 * 5 + 1).reshape([b, ic, 25]), requires_grad=True, dtype=pyvqnet.kfloat32)
        x = test_conv.forward(x0)
        print(x)

Conv2D
^^^^^^^^^^^^^^^^^^^^^^^^

.. py:class:: pyvqnet.nn.torch.Conv2D(input_channels: int, output_channels: int, kernel_size: tuple, stride: tuple = (1, 1), padding = "valid", use_bias = True, kernel_initializer = None, bias_initializer = None, dilation_rate: int = 1, group: int = 1, dtype = None, name = "")

    在输入上执行二维卷积。Conv2D 模块的输入形状为 (batch_size, input_channels, height, width)。
    此类继承自 ``pyvqnet.nn.Module`` 和 ``torch.nn.Module``\ ，可作为 torch 模型的子模块使用。

    类中的 ``_buffers`` 数据为 ``torch.Tensor`` 类型。
    类中的 ``_parameters`` 数据为 ``torch.nn.Parameter`` 类型。

    :param input_channels: `int` - 输入通道数。
    :param output_channels: `int` - 输出通道数。
    :param kernel_size: `tuple|list` - 卷积核大小。卷积核形状为 [output_channels, input_channels/group, kernel_size, kernel_size]。
    :param stride: `tuple|list` - 步长，默认为 (1, 1)。
    :param padding: `str|tuple` - 填充选项，可以是字符串 {'valid', 'same'} 或指定两侧填充量的元组。默认为 "valid"。
    :param use_bias: `bool` - 是否使用偏置项，默认为 True。
    :param kernel_initializer: `callable` - 卷积核初始化方法。默认为空，使用 kaiming_uniform。
    :param bias_initializer: `callable` - 偏置初始化方法。默认为空，使用 kaiming_uniform。
    :param dilation_rate: `int` - 膨胀大小，默认为 1。
    :param group: `int` - 分组卷积中的组数。默认为 1。
    :param dtype: 参数的数据类型，默认为 None，使用默认数据类型 `kfloat32`\ ，表示 32 位浮点数。
    :param name: 模块的名称，默认为 ""。

    :return: 二维卷积的一个实例。

    .. note::

        ``padding='valid'`` 不进行填充。

        ``padding='same'`` 对输入进行零填充，输出的高度等于 `ceil(in_height / stride)`\ 。

    示例::

        import numpy as np
        import pyvqnet
        from pyvqnet.tensor import QTensor
        from pyvqnet.nn.torch import Conv2D
        pyvqnet.backends.set_backend("torch")
        b = 2
        ic = 3
        oc = 2
        test_conv = Conv2D(ic, oc, (3, 3), (2, 2))
        x0 = QTensor(np.arange(1, b * ic * 5 * 5 + 1).reshape([b, ic, 5, 5]), requires_grad=True, dtype=pyvqnet.kfloat32)
        x = test_conv.forward(x0)
        print(x)

ConvT2D
^^^^^^^^^^^^^^^^^^^^^^^^

.. py:class:: pyvqnet.nn.torch.ConvT2D(input_channels, output_channels, kernel_size, stride=[1, 1], padding=(0, 0), use_bias="True", kernel_initializer=None, bias_initializer=None, dilation_rate: int = 1, out_padding=(0, 0), group=1, dtype=None, name="")

    在输入上执行二维转置卷积。ConvT2D 模块的输入形状为 (batch_size, input_channels, height, width)。
    此类继承自 ``pyvqnet.nn.Module`` 和 ``torch.nn.Module``\ ，可作为 torch 模型的子模块使用。

    类中的 ``_buffers`` 数据为 ``torch.Tensor`` 类型。
    类中的 ``_parameters`` 数据为 ``torch.nn.Parameter`` 类型。

    :param input_channels: `int` - 输入通道数。
    :param output_channels: `int` - 输出通道数。
    :param kernel_size: `tuple|list` - 卷积核大小，卷积核形状 = [input_channels, output_channels/group, kernel_size, kernel_size]。
    :param stride: `tuple|list` - 步长，默认为 (1, 1)。
    :param padding: `tuple` - 填充选项，指定两侧填充量的元组。默认为 (0, 0)。
    :param use_bias: `bool` - 是否使用偏置项，默认为 True。
    :param kernel_initializer: `callable` - 卷积核初始化方法。默认为空，使用 kaiming_uniform。
    :param bias_initializer: `callable` - 偏置初始化方法。默认为空，使用 kaiming_uniform。
    :param dilation_rate: `int` - 膨胀大小，默认为 1。
    :param out_padding: 为每个维度添加到输出形状的额外大小。默认为 (0, 0)。
    :param group: `int` - 分组卷积中的组数。默认为 1。
    :param dtype: 参数的数据类型，默认为 None，使用默认数据类型 `kfloat32`\ ，表示 32 位浮点数。
    :param name: 模块的名称，默认为 ""。

    :return: 二维转置卷积的一个实例。

    .. note::

        ``padding='valid'`` 不进行填充。

        ``padding='same'`` 对输入进行零填充，输出的高度等于 `ceil(height / stride)`\ 。

    示例::

        import numpy as np
        import pyvqnet
        from pyvqnet.tensor import QTensor
        from pyvqnet.nn.torch import ConvT2D
        pyvqnet.backends.set_backend("torch")
        test_conv = ConvT2D(3, 2, (3, 3), (1, 1))
        x = QTensor(np.arange(1, 1 * 3 * 5 * 5 + 1).reshape([1, 3, 5, 5]), requires_grad=True, dtype=pyvqnet.kfloat32)
        y = test_conv.forward(x)
        print(y)

AvgPool1D
^^^^^^^^^^^^^^^^^^^^^^^^

.. py:class:: pyvqnet.nn.torch.AvgPool1D(kernel, stride, padding=0, name = "")

    对一维输入执行平均池化。输入形状为 (batch_size, input_channels, in_height)。
    此类继承自 ``pyvqnet.nn.Module`` 和 ``torch.nn.Module``\ ，可作为子模块添加到 torch 模型中。

    类中的 ``_buffers`` 数据为 ``torch.Tensor`` 类型。
    类中的 ``_parameters`` 数据为 ``torch.nn.Parameter`` 类型。

    :param kernel: 池化窗口的大小。
    :param stride: 窗口移动的步长。
    :param padding: 填充选项，指定填充长度的整数。默认为 0。
    :param name: 模块的名称，默认为 ""。

    :return: 一维平均池化层的一个实例。

    示例::

        import numpy as np
        import pyvqnet
        from pyvqnet.tensor import QTensor
        from pyvqnet.nn.torch import AvgPool1D
        pyvqnet.backends.set_backend("torch")
        test_mp = AvgPool1D([3],[2],0)
        x = QTensor(np.array([0, 1, 0, 4, 5,
                             2, 3, 2, 1, 3,
                             4, 4, 0, 4, 3,
                             2, 5, 2, 6, 4,
                             1, 0, 0, 5, 7], dtype=float).reshape([1, 5, 5]), requires_grad=True)
        y = test_mp.forward(x)
        print(y)

MaxPool1D
^^^^^^^^^^^^^^^^^^^^^^^^

.. py:class:: pyvqnet.nn.torch.MaxPool1D(kernel, stride, padding=0, name="")

    对一维输入执行最大池化。输入形状为 (batch_size, input_channels, in_height)。
    此类继承自 ``pyvqnet.nn.Module`` 和 ``torch.nn.Module``\ ，可作为子模块添加到 torch 模型中。

    类中的 ``_buffers`` 数据为 ``torch.Tensor`` 类型。
    类中的 ``_parameters`` 数据为 ``torch.nn.Parameter`` 类型。

    :param kernel: 池化窗口的大小。
    :param stride: 窗口移动的步长。
    :param padding: 填充选项，指定填充长度的整数。默认为 0。
    :param name: 模块的名称，默认为 ""。

    :return: 一维最大池化层的一个实例。

    示例::

        import numpy as np
        import pyvqnet
        from pyvqnet.tensor import QTensor
        from pyvqnet.nn.torch import MaxPool1D
        pyvqnet.backends.set_backend("torch")
        test_mp = MaxPool1D([3],[2],0)
        x = QTensor(np.array([0, 1, 0, 4, 5,
                             2, 3, 2, 1, 3,
                             4, 4, 0, 4, 3,
                             2, 5, 2, 6, 4,
                             1, 0, 0, 5, 7], dtype=float).reshape([1, 5, 5]), requires_grad=True)
        y = test_mp.forward(x)
        print(y)

AvgPool2D
^^^^^^^^^^^^^^^^^^^^^^^^

.. py:class:: pyvqnet.nn.torch.AvgPool2D(kernel, stride, padding=(0,0), name="")

    对二维输入执行平均池化。输入形状为 (batch_size, input_channels, height, width)。
    此类继承自 ``pyvqnet.nn.Module`` 和 ``torch.nn.Module``\ ，可作为子模块添加到 torch 模型中。

    类中的 ``_buffers`` 数据为 ``torch.Tensor`` 类型。
    类中的 ``_parameters`` 数据为 ``torch.nn.Parameter`` 类型。

    :param kernel: 池化窗口的大小。
    :param stride: 窗口移动的步长。
    :param padding: 填充选项，包含两个整数的元组，指定两个维度的填充。默认为 (0,0)。
    :param name: 模块的名称，默认为 ""。

    :return: 二维平均池化层的一个实例。

    示例::

        import numpy as np
        import pyvqnet
        from pyvqnet.tensor import QTensor
        from pyvqnet.nn.torch import AvgPool2D
        pyvqnet.backends.set_backend("torch")
        test_mp = AvgPool2D([2, 2], [2, 2], 1)
        x = QTensor(np.array([0, 1, 0, 4, 5,
                             2, 3, 2, 1, 3,
                             4, 4, 0, 4, 3,
                             2, 5, 2, 6, 4,
                             1, 0, 0, 5, 7], dtype=float).reshape([1, 1, 5, 5]), requires_grad=True)
        y = test_mp.forward(x)
        print(y)

MaxPool2D
^^^^^^^^^^^^^^^^^^^^^^^^

.. py:class:: pyvqnet.nn.torch.MaxPool2D(kernel, stride, padding=(0,0), name="")

    对二维输入执行最大池化。输入形状为 (batch_size, input_channels, height, width)。
    此类继承自 ``pyvqnet.nn.Module`` 和 ``torch.nn.Module``\ ，可作为子模块添加到 torch 模型中。

    类中的 ``_buffers`` 数据为 ``torch.Tensor`` 类型。
    类中的 ``_parameters`` 数据为 ``torch.nn.Parameter`` 类型。

    :param kernel: 池化窗口的大小。
    :param stride: 窗口移动的步长。
    :param padding: 填充选项，包含两个整数的元组，指定两个维度的填充。默认为 (0,0)。
    :param name: 模块的名称，默认为 ""。

    :return: 二维最大池化层的一个实例。

    示例::

        import numpy as np
        import pyvqnet
        from pyvqnet.tensor import QTensor
        from pyvqnet.nn.torch import MaxPool2D
        pyvqnet.backends.set_backend("torch")
        test_mp = MaxPool2D([2, 2], [2, 2], (0, 0))
        x = QTensor(np.array([0, 1, 0, 4, 5,
                             2, 3, 2, 1, 3,
                             4, 4, 0, 4, 3,
                             2, 5, 2, 6, 4,
                             1, 0, 0, 5, 7], dtype=float).reshape([1, 1, 5, 5]), requires_grad=True)
        y = test_mp.forward(x)
        print(y)

Embedding
^^^^^^^^^^^^^^^^^^^^^^^^

.. py:class:: pyvqnet.nn.torch.Embedding(num_embeddings, embedding_dim, weight_initializer=xavier_normal, dtype=None, name: str = "")

    此模块通常用于存储词嵌入并使用索引检索它们。该模块的输入是索引列表，输出是对应的词嵌入。
    该层的输入应为 `kint64` 类型。
    此类继承自 ``pyvqnet.nn.Module`` 和 ``torch.nn.Module``\ ，可作为子模块添加到 torch 模型中。

    类中的 ``_buffers`` 数据为 ``torch.Tensor`` 类型。
    类中的 ``_parameters`` 数据为 ``torch.nn.Parameter`` 类型。

    :param num_embeddings: `int` - 嵌入字典的大小。
    :param embedding_dim: `int` - 每个嵌入向量的大小。
    :param weight_initializer: `callable` - 权重初始化方法，默认为 Xavier 正态分布。
    :param dtype: 参数的数据类型，默认为 None，使用默认数据类型：`kfloat32`\ （32 位浮点数）。
    :param name: 嵌入层的名称，默认为 ""。

    :return: Embedding 层的一个实例。

    示例::

        import numpy as np
        import pyvqnet
        from pyvqnet.tensor import QTensor
        from pyvqnet.nn.torch import Embedding
        pyvqnet.backends.set_backend("torch")
        vlayer = Embedding(30, 3)
        x = QTensor(np.arange(1, 25).reshape([2, 3, 2, 2]), dtype=pyvqnet.kint64)
        y = vlayer(x)
        print(y)

BatchNorm2d
^^^^^^^^^^^^^^^^^^^^^^^^

.. py:class:: pyvqnet.nn.torch.BatchNorm2d(channel_num:int, momentum:float=0.1, epsilon:float = 1e-5, affine=True, beta_initializer=zeros, gamma_initializer=ones, dtype=None, name="")

    在四维输入 (B, C, H, W) 上应用批归一化。参考论文：
    `Batch Normalization: Accelerating Deep Network Training by Reducing
    Internal Covariate Shift <https://arxiv.org/abs/1502.03167>`__。

    此类继承自 ``pyvqnet.nn.Module`` 和 ``torch.nn.Module``\ ，可作为子模块添加到 torch 模型中。

    类中的 ``_buffers`` 数据为 ``torch.Tensor`` 类型。
    类中的 ``_parameters`` 数据为 ``torch.nn.Parameter`` 类型。

    .. math::
        y = \frac{x - \mathrm{E}[x]}{\sqrt{\mathrm{Var}[x] + \epsilon}} * \gamma + \beta

    其中 :math:`\gamma` 和 :math:`\beta` 是可训练参数。此外，默认情况下，在训练期间，该层会持续估计均值和方差，这些值在评估期间用于归一化。移动平均的动量设置为默认值 0.1。

    :param channel_num: `int` - 输入通道数。
    :param momentum: `float` - 移动平均计算的动量，默认为 0.1。
    :param epsilon: `float` - 数值稳定性的小常数，默认为 1e-5。
    :param affine: `bool` - 是否为每个通道包含可学习的仿射参数。默认为 `True`\ ，将权重初始化为 1，偏置初始化为 0。
    :param beta_initializer: `callable` - beta 的初始化方法，默认为零初始化。
    :param gamma_initializer: `callable` - gamma 的初始化方法，默认为一初始化。
    :param dtype: 参数的数据类型，默认为 None，使用 `kfloat32`\ （32 位浮点数）。
    :param name: 批归一化层的名称，默认为 ""。

    :return: 二维批归一化层的一个实例。

    示例::

        import numpy as np
        import pyvqnet
        from pyvqnet.tensor import QTensor
        from pyvqnet.nn.torch import BatchNorm2d
        pyvqnet.backends.set_backend("torch")
        b = 2
        ic = 2
        test_conv = BatchNorm2d(ic)
        x = QTensor(np.arange(1, 17).reshape([b, ic, 4, 1]), requires_grad=True, dtype=pyvqnet.kfloat32)
        y = test_conv.forward(x)
        print(y)

BatchNorm1d
^^^^^^^^^^^^^^^^^^^^^^^^

.. py:class:: pyvqnet.nn.torch.BatchNorm1d(channel_num:int, momentum:float=0.1, epsilon:float = 1e-5, affine=True, beta_initializer=zeros, gamma_initializer=ones, dtype=None, name="")

    在二维输入 (B, C) 上应用批归一化。参考论文：
    `Batch Normalization: Accelerating Deep Network Training by Reducing
    Internal Covariate Shift <https://arxiv.org/abs/1502.03167>`__。

    .. math::
        y = \frac{x - \mathrm{E}[x]}{\sqrt{\mathrm{Var}[x] + \epsilon}} * \gamma + \beta

    其中 :math:`\gamma` 和 :math:`\beta` 是可训练参数。此外，默认情况下，在训练期间，该层会持续估计均值和方差，这些值在评估期间用于归一化。移动平均的动量设置为默认值 0.1。

    此类继承自 ``pyvqnet.nn.Module`` 和 ``torch.nn.Module``\ ，可作为子模块添加到 torch 模型中。

    类中的 ``_buffers`` 数据为 ``torch.Tensor`` 类型。
    类中的 ``_parameters`` 数据为 ``torch.nn.Parameter`` 类型。

    :param channel_num: `int` - 输入通道数。
    :param momentum: `float` - 移动平均计算的动量，默认为 0.1。
    :param epsilon: `float` - 数值稳定性的小常数，默认为 1e-5。
    :param affine: `bool` - 是否为每个通道包含可学习的仿射参数。默认为 `True`\ ，将权重初始化为 1，偏置初始化为 0。
    :param beta_initializer: `callable` - beta 的初始化方法，默认为零初始化。
    :param gamma_initializer: `callable` - gamma 的初始化方法，默认为一初始化。
    :param dtype: 参数的数据类型，默认为 None，使用 `kfloat32`\ （32 位浮点数）。
    :param name: 批归一化层的名称，默认为 ""。

    :return: 一维批归一化层的一个实例。

    示例::

        import numpy as np
        import pyvqnet
        from pyvqnet.tensor import QTensor
        from pyvqnet.nn.torch import BatchNorm1d
        pyvqnet.backends.set_backend("torch")
        test_conv = BatchNorm1d(4)
        x = QTensor(np.arange(1, 17).reshape([4, 4]), requires_grad=True, dtype=pyvqnet.kfloat32)
        y = test_conv.forward(x)
        print(y)

LayerNormNd
^^^^^^^^^^^^^^^^^^^^^^^^

.. py:class:: pyvqnet.qnn.vqc.torch.LayerNormNd(normalized_shape: list, epsilon: float = 1e-5, affine=True, dtype=None, name="")

    在任何输入的最后 D 个维度上应用层归一化。具体方法在论文中描述：
    `Layer Normalization <https://arxiv.org/abs/1607.06450>`__。

    .. math::
        y = \frac{x - \mathrm{E}[x]}{ \sqrt{\mathrm{Var}[x] + \epsilon}} * \gamma + \beta

    对于如 (B, C, H, W, D) 的输入，``norm_shape`` 可以为 [C, H, W, D]、[H, W, D]、[W, D] 或 [D]。

    此类继承自 ``pyvqnet.nn.Module`` 和 ``torch.nn.Module``\ ，可作为子模块添加到 torch 模型中。

    类中的 ``_buffers`` 数据为 ``torch.Tensor`` 类型。
    类中的 ``_parameters`` 数据为 ``torch.nn.Parameter`` 类型。

    :param norm_shape: `list` - 要进行归一化的形状。
    :param epsilon: `float` - 数值稳定性的小常数，默认为 1e-5。
    :param affine: `bool` - 如果为 `True`\ ，该模块为每个通道具有可学习的仿射参数，初始化为 1（权重）和 0（偏置）。默认为 `True`\ 。
    :param dtype: 参数的数据类型，默认为 None，使用 `kfloat32`\ （32 位浮点数）。
    :param name: 模块的名称，默认为 ""。

    :return: LayerNormNd 类的一个实例。

    示例::

        import numpy as np
        from pyvqnet.tensor import QTensor
        from pyvqnet import kfloat32
        from pyvqnet.nn.torch import LayerNormNd
        import pyvqnet
        pyvqnet.backends.set_backend("torch")
        ic = 4
        test_conv = LayerNormNd([2,2])
        x = QTensor(np.arange(1,17).reshape([2,2,2,2]), requires_grad=True, dtype=kfloat32)
        y = test_conv.forward(x)
        print(y)

LayerNorm2d
^^^^^^^^^^^^^^^^^^^^^^^^

.. py:class:: pyvqnet.qnn.vqc.torch.LayerNorm2d(norm_size:int, epsilon:float = 1e-5, affine=True, dtype=None, name="")

    在四维输入上应用层归一化。具体方法在论文中描述：
    `Layer Normalization <https://arxiv.org/abs/1607.06450>`__。

    .. math::
        y = \frac{x - \mathrm{E}[x]}{ \sqrt{\mathrm{Var}[x] + \epsilon}} * \gamma + \beta

    均值和标准差在除第一个维度外的其余维度上计算。对于如 (B, C, H, W) 的输入，``norm_size`` 应等于 C * H * W。

    此类继承自 ``pyvqnet.nn.Module`` 和 ``torch.nn.Module``\ ，可作为子模块添加到 torch 模型中。

    类中的 ``_buffers`` 数据为 ``torch.Tensor`` 类型。
    类中的 ``_parameters`` 数据为 ``torch.nn.Parameter`` 类型。

    :param norm_size: `int` - 归一化的大小，应等于 C * H * W。
    :param epsilon: `float` - 数值稳定性的小常数，默认为 1e-5。
    :param affine: `bool` - 如果为 `True`\ ，该模块为每个通道具有可学习的仿射参数，初始化为 1（权重）和 0（偏置）。默认为 `True`\ 。
    :param dtype: 参数的数据类型，默认为 None，使用 `kfloat32`\ （32 位浮点数）。
    :param name: 模块的名称，默认为 ""。

    :return: 二维层归一化的一个实例。

    示例::

        import numpy as np
        import pyvqnet
        from pyvqnet.tensor import QTensor
        from pyvqnet.nn.torch import LayerNorm2d
        import pyvqnet
        pyvqnet.backends.set_backend("torch")
        ic = 4
        test_conv = LayerNorm2d(8)
        x = QTensor(np.arange(1,17).reshape([2,2,4,1]), requires_grad=True, dtype=pyvqnet.kfloat32)
        y = test_conv.forward(x)
        print(y)

LayerNorm1d
^^^^^^^^^^^^^^^^^^^^^^^^

.. py:class:: pyvqnet.qnn.vqc.torch.LayerNorm1d(norm_size:int, epsilon:float = 1e-5, affine=True, dtype=None, name="")

    在二维输入上应用层归一化。具体方法在论文中描述：
    `Layer Normalization <https://arxiv.org/abs/1607.06450>`__。

    .. math::
        y = \frac{x - \mathrm{E}[x]}{ \sqrt{\mathrm{Var}[x] + \epsilon}} * \gamma + \beta

    均值和标准差在最后一个维度上计算，其中 ``norm_size`` 是最后一个维度的值。

    此类继承自 ``pyvqnet.nn.Module`` 和 ``torch.nn.Module``\ ，可作为子模块添加到 torch 模型中。

    类中的 ``_buffers`` 数据为 ``torch.Tensor`` 类型。
    类中的 ``_parameters`` 数据为 ``torch.nn.Parameter`` 类型。

    :param norm_size: `int` - 归一化的大小，应等于最后一个维度的大小。
    :param epsilon: `float` - 数值稳定性的小常数，默认为 1e-5。
    :param affine: `bool` - 如果为 `True`\ ，该模块为每个通道具有可学习的仿射参数，初始化为 1（权重）和 0（偏置）。默认为 `True`\ 。
    :param dtype: 参数的数据类型，默认为 None，使用 `kfloat32`\ （32 位浮点数）。
    :param name: 模块的名称，默认为 ""。

    :return: 一维层归一化的一个实例。

    示例::

        import numpy as np
        from pyvqnet.tensor import QTensor
        from pyvqnet.nn.torch import LayerNorm1d
        import pyvqnet
        pyvqnet.backends.set_backend("torch")
        test_conv = LayerNorm1d(4)
        x = QTensor(np.arange(1,17).reshape([4,4]), requires_grad=True, dtype=pyvqnet.kfloat32)
        y = test_conv.forward(x)
        print(y)

GroupNorm
^^^^^^^^^^^^^^^^^^^^^^^^

.. py:class:: pyvqnet.qnn.vqc.torch.GroupNorm(num_groups: int, num_channels: int, epsilon = 1e-5, affine = True, dtype = None, name = "")

    在 mini-batch 输入上应用组归一化。输入：:math:`(N, C, *)`\ ，其中 :math:`C=\mathrm{num\_channels}`\ ，输出：:math:`(N, C, *)`\ 。

    该层实现了论文 `Group Normalization <https://arxiv.org/abs/1803.08494>`__ 中描述的操作。

    .. math::

        y = \frac{x - \mathrm{E}[x]}{ \sqrt{\mathrm{Var}[x] + \epsilon}} * \gamma + \beta

    输入通道被划分为 :attr:`num_groups` 组，每组包含 ``num_channels / num_groups`` 个通道。:attr:`num_channels` 必须能被 :attr:`num_groups` 整除。均值和标准差在每个组内分别计算。如果 :attr:`affine` 为 ``True``\ ，则 :math:`\gamma` 和 :math:`\beta` 是可学习的。每个通道的仿射变换参数是大小为 :attr:`num_channels` 的向量。

    此类继承自 ``pyvqnet.nn.Module`` 和 ``torch.nn.Module``\ ，可作为子模块添加到 torch 模型中。

    类中的 ``_buffers`` 数据为 ``torch.Tensor`` 类型。
    类中的 ``_parameters`` 数据为 ``torch.nn.Parameter`` 类型。

    :param num_groups (int): 要将通道划分的组数。
    :param num_channels (int): 期望的输入通道数。
    :param epsilon: 为数值稳定性添加到分母中的小值。默认为 1e-5。
    :param affine: 布尔值。如果设置为 ``True``\ ，该模块为每个通道具有可学习的仿射参数，初始化为 1（权重）和 0（偏置）。默认为 ``True``\ 。
    :param dtype: 参数的数据类型。默认为 None，使用 `kfloat32`\ （32 位浮点数）。
    :param name: 模块的名称。默认为 ""。

    :return: GroupNorm 类的一个实例。

    示例::

        import numpy as np
        from pyvqnet.tensor import QTensor, kfloat32
        from pyvqnet.nn.torch import GroupNorm
        import pyvqnet
        pyvqnet.backends.set_backend("torch")
        test_conv = GroupNorm(2, 10)
        x = QTensor(np.arange(0, 60*2*5).reshape([2, 10, 3, 2, 5]), requires_grad=True, dtype=kfloat32)
        y = test_conv.forward(x)
        print(y)

Dropout
^^^^^^^^^^^^^^^^^^^^^^^^

.. py:class:: pyvqnet.nn.torch.Dropout(dropout_rate = 0.5)

    Dropout 模块。该模块随机将某些单元的输出置零，同时根据给定的 dropout_rate 概率缩放其余单元。

    此类继承自 ``pyvqnet.nn.Module`` 和 ``torch.nn.Module``\ ，可作为子模块添加到 torch 模型中。

    :param dropout_rate: `float` - 将神经元置零的概率。
    :param name: 模块的名称。默认为 ""。

    :return: Dropout 类的一个实例。

    示例::

        import numpy as np
        from pyvqnet.nn.torch import Dropout
        from pyvqnet.tensor import arange
        import pyvqnet
        pyvqnet.backends.set_backend("torch")
        b = 2
        ic = 2
        x = arange(-1 * ic * 2 * 2.0, (b - 1) * ic * 2 * 2).reshape([b, ic, 2, 2])
        droplayer = Dropout(0.5)
        droplayer.train()
        y = droplayer(x)
        print(y)

DropPath
^^^^^^^^^^^^^^^^^^^^^^^^

.. py:class:: pyvqnet.nn.torch.DropPath(dropout_rate = 0.5, name="")

    DropPath 模块应用随机样本路径 dropout（随机深度）。

    此类继承自 ``pyvqnet.nn.Module`` 和 ``torch.nn.Module``\ ，可作为子模块添加到 torch 模型中。

    :param dropout_rate: `float` - 将神经元置零的概率。
    :param name: 模块的名称。默认为 ""。

    :return: DropPath 类的一个实例。

    示例::

        import pyvqnet.nn.torch as nn
        import pyvqnet.tensor as tensor
        import pyvqnet
        pyvqnet.backends.set_backend("torch")
        x = tensor.randu([4])
        y = nn.DropPath()(x)
        print(y)

Pixel_Shuffle
^^^^^^^^^^^^^^^^^^^^^^^^

.. py:class:: pyvqnet.nn.torch.Pixel_Shuffle(upscale_factors, name="")

    将形状为 (*, C * r^2, H, W) 的张量重新排列为形状 (*, C, H * r, W * r) 的张量，其中 r 是缩放因子。

    此类继承自 ``pyvqnet.nn.Module`` 和 ``torch.nn.Module``\ ，可作为子模块添加到 torch 模型中。

    :param upscale_factors: 变换的缩放因子。
    :param name: 模块的名称。默认为 ""。

    :return: Pixel_Shuffle 模块的一个实例。

    示例::

        from pyvqnet.nn.torch import Pixel_Shuffle
        from pyvqnet.tensor import tensor
        import pyvqnet
        pyvqnet.backends.set_backend("torch")
        ps = Pixel_Shuffle(3)
        inx = tensor.ones([5, 2, 3, 18, 4, 4])
        inx.requires_grad = True
        y = ps(inx)

Pixel_Unshuffle
^^^^^^^^^^^^^^^^^^^^^^^^

.. py:class:: pyvqnet.nn.torch.Pixel_Unshuffle(downscale_factors, name="")

    通过重新排列元素来反转 Pixel_Shuffle 操作。将形状为 (*, C, H * r, W * r) 的张量转换为 (*, C * r^2, H, W)，其中 r 是降采样因子。

    此类继承自 ``pyvqnet.nn.Module`` 和 ``torch.nn.Module``\ ，可作为子模块添加到 torch 模型中。

    :param downscale_factors: 变换的降采样因子。
    :param name: 模块的名称。默认为 ""。

    :return: Pixel_Unshuffle 模块的一个实例。

    示例::

        from pyvqnet.nn.torch import Pixel_Unshuffle
        from pyvqnet.tensor import tensor
        import pyvqnet
        pyvqnet.backends.set_backend("torch")
        ps = Pixel_Unshuffle(3)
        inx = tensor.ones([5, 2, 3, 2, 12, 12])
        inx.requires_grad = True
        y = ps(inx)

GRU
^^^^^^^^^^^^^^^^^^^^^^^^

.. py:class:: pyvqnet.nn.torch.GRU(input_size, hidden_size, num_layers=1, nonlinearity='tanh', batch_first=True, use_bias=True, bidirectional=False, dtype=None, name: str = "")

    门控循环单元（GRU）模块。支持多层堆叠和双向配置。单层单向 GRU 的公式如下：

    .. math::
        \begin{array}{ll}
            r_t = \sigma(W_{ir} x_t + b_{ir} + W_{hr} h_{(t-1)} + b_{hr}) \\
            z_t = \sigma(W_{iz} x_t + b_{iz} + W_{hz} h_{(t-1)} + b_{hz}) \\
            n_t = \tanh(W_{in} x_t + b_{in} + r_t * (W_{hn} h_{(t-1)}+ b_{hn})) \\
            h_t = (1 - z_t) * n_t + z_t * h_{(t-1)}
        \end{array}

    此类继承自 ``pyvqnet.nn.Module`` 和 ``torch.nn.Module``\ ，可作为子模块添加到 torch 模型中。

    类的 ``_buffers`` 包含 ``torch.Tensor`` 数据，类的 ``_parameters`` 包含 ``torch.nn.Parameter`` 数据。

    :param input_size: 输入特征维度。
    :param hidden_size: 隐藏特征维度。
    :param num_layers: 堆叠的 GRU 层数，默认值：1。
    :param batch_first: 如果为 True，输入形状为 [batch_size, seq_len, feature_dim]，如果为 False，形状为 [seq_len, batch_size, feature_dim]，默认值：True。
    :param use_bias: 如果为 False，模块不使用偏置项，默认值：True。
    :param bidirectional: 如果为 True，使 GRU 变为双向，默认值：False。
    :param dtype: 参数的数据类型，默认为 None，使用默认数据类型：kfloat32（32 位浮点数）。
    :param name: 模块的名称，默认值：""。

    :return: GRU 模块的一个实例。

    示例::

        import pyvqnet
        pyvqnet.backends.set_backend("torch")
        from pyvqnet.nn.torch import GRU
        from pyvqnet.tensor import tensor

        rnn2 = GRU(4, 6, 2, batch_first=False, bidirectional=True)

        input = tensor.ones([5, 3, 4])
        h0 = tensor.ones([4, 3, 6])

        output, hn = rnn2(input, h0)



RNN
^^^^^^^^^^^^^^^^^^^^^^^^

.. py:class:: pyvqnet.nn.torch.RNN(input_size, hidden_size, num_layers=1, nonlinearity='tanh', batch_first=True, use_bias=True, bidirectional=False, dtype=None, name: str = "")

    循环神经网络（RNN）模块，使用 :math:`\tanh` 或 :math:`\text{ReLU}` 作为激活函数。支持双向和多层配置。单层单向 RNN 的公式如下：

    .. math::
        h_t = \tanh(W_{ih} x_t + b_{ih} + W_{hh} h_{(t-1)} + b_{hh})

    如果 :attr:`nonlinearity` 为 ``'relu'``\ ，则 :math:`\text{ReLU}` 将替换 :math:`\tanh`\ 。

    此类继承自 ``pyvqnet.nn.Module`` 和 ``torch.nn.Module``\ ，可作为子模块添加到 torch 模型中。

    类的 ``_buffers`` 包含 ``torch.Tensor`` 数据，类的 ``_parameters`` 包含 ``torch.nn.Parameter`` 数据。

    :param input_size: 输入特征维度。
    :param hidden_size: 隐藏特征维度。
    :param num_layers: 堆叠的 RNN 层数，默认值：1。
    :param nonlinearity: 非线性激活函数，默认值：``'tanh'``\ 。
    :param batch_first: 如果为 True，输入形状为 [batch_size, seq_len, feature_dim]，如果为 False，形状为 [seq_len, batch_size, feature_dim]，默认值：True。
    :param use_bias: 如果为 False，模块不使用偏置项，默认值：True。
    :param bidirectional: 如果为 True，使 RNN 变为双向，默认值：False。
    :param dtype: 参数的数据类型，默认为 None，使用默认数据类型：kfloat32（32 位浮点数）。
    :param name: 模块的名称，默认值：""。

    :return: RNN 模块的一个实例。

    示例::

        import pyvqnet
        pyvqnet.backends.set_backend("torch")
        from pyvqnet.nn.torch import RNN
        from pyvqnet.tensor import tensor

        rnn2 = RNN(4, 6, 2, batch_first=False, bidirectional = True)

        input = tensor.ones([5, 3, 4])
        h0 = tensor.ones([4, 3, 6])
        output, hn = rnn2(input, h0)

LSTM
^^^^^^^^^^^^^^^^^^^^^^^^

.. py:class:: pyvqnet.nn.torch.LSTM(input_size, hidden_size, num_layers=1, batch_first=True, use_bias=True, bidirectional=False, dtype=None, name: str = "")

    长短期记忆（LSTM）模块。支持双向 LSTM 和堆叠多层 LSTM 配置。单层单向 LSTM 的公式如下：

    .. math::
        \begin{array}{ll} \\
            i_t = \sigma(W_{ii} x_t + b_{ii} + W_{hi} h_{t-1} + b_{hi}) \\
            f_t = \sigma(W_{if} x_t + b_{if} + W_{hf} h_{t-1} + b_{hf}) \\
            g_t = \tanh(W_{ig} x_t + b_{ig} + W_{hg} h_{t-1} + b_{hg}) \\
            o_t = \sigma(W_{io} x_t + b_{io} + W_{ho} h_{t-1} + b_{ho}) \\
            c_t = f_t \odot c_{t-1} + i_t \odot g_t \\
            h_t = o_t \odot \tanh(c_t) \\
        \end{array}

    此类继承自 ``pyvqnet.nn.Module`` 和 ``torch.nn.Module``\ ，可作为子模块添加到 torch 模型中。

    类的 ``_buffers`` 包含 ``torch.Tensor`` 数据，类的 ``_parameters`` 包含 ``torch.nn.Parameter`` 数据。

    :param input_size: 输入特征维度。
    :param hidden_size: 隐藏特征维度。
    :param num_layers: 堆叠的 LSTM 层数，默认值：1。
    :param batch_first: 如果为 True，输入形状为 [batch_size, seq_len, feature_dim]，如果为 False，形状为 [seq_len, batch_size, feature_dim]，默认值：True。
    :param use_bias: 如果为 False，模块不使用偏置项，默认值：True。
    :param bidirectional: 如果为 True，使 LSTM 变为双向，默认值：False。
    :param dtype: 参数的数据类型，默认为 None，使用默认数据类型：kfloat32（32 位浮点数）。
    :param name: 模块的名称，默认值：""。

    :return: LSTM 模块的一个实例。

    示例::

        import pyvqnet
        pyvqnet.backends.set_backend("torch")
        from pyvqnet.nn.torch import LSTM
        from pyvqnet.tensor import tensor

        rnn2 = LSTM(4, 6, 2, batch_first=False, bidirectional = True)

        input = tensor.ones([5, 3, 4])
        h0 = tensor.ones([4, 3, 6])
        c0 = tensor.ones([4, 3, 6])
        output, (hn, cn) = rnn2(input, (h0, c0))


Dynamic_GRU
^^^^^^^^^^^^^^^^^^^^^^^^

.. py:class:: pyvqnet.nn.torch.Dynamic_GRU(input_size,hidden_size, num_layers=1, batch_first=True, use_bias=True, bidirectional=False, dtype=None, name: str = "")

    对动态长度的输入序列应用多层门控循环单元（GRU）RNN。

    第一个输入应为通过 ``tensor.PackedSequence`` 类定义的可变长度批处理序列输入。

    ``tensor.PackedSequence`` 类可通过依次调用以下函数构造：``pad_sequence``\ 、``pack_pad_sequence``\ 。

    Dynamic_GRU 的第一个输出也是一个 ``tensor.PackedSequence`` 类，
    可以使用 ``tensor.pad_pack_sequence`` 解包为正常的 QTensor。

    对于输入序列中的每个元素，每一层计算以下公式：

    .. math::
        \begin{array}{ll}
        r_t = \sigma(W_{ir} x_t + b_{ir} + W_{hr} h_{(t-1)} + b_{hr}) \\
        z_t = \sigma(W_{iz} x_t + b_{iz} + W_{hz} h_{(t-1)} + b_{hz}) \\
        n_t = \tanh(W_{in} x_t + b_{in} + r_t * (W_{hn} h_{(t-1)}+ b_{hn})) \\
        h_t = (1 - z_t) * n_t + z_t * h_{(t-1)}
        \end{array}

    此类继承自 ``pyvqnet.nn.Module`` 和 ``torch.nn.Module``\ ，可作为子模块添加到 torch 模型中。

    此类的 ``_buffers`` 中的数据为 ``torch.Tensor`` 类型。

    此类的 ``_parameters`` 中的数据为 ``torch.nn.Parameter`` 类型。

    :param input_size: 输入特征维度。
    :param hidden_size: 隐藏特征维度。
    :param num_layers: 循环层数。默认值：1
    :param batch_first: 如果为 True，输入形状为 [batch size, sequence length, feature dimension]。如果为 False，输入形状为 [sequence length, batch size, feature dimension]。默认值：True。
    :param use_bias: 如果为 False，该层不使用偏置权重 b_ih 和 b_hh。默认值：True。
    :param bidirectional: 如果为 True，变为双向 GRU。默认值：False。
    :param dtype: 参数的数据类型，默认值：None，使用默认数据类型：kfloat32，表示 32 位浮点数。
    :param name: 此模块的名称，默认为 ""。

    :return: 一个 Dynamic_GRU 类

    示例::

        import pyvqnet
        pyvqnet.backends.set_backend("torch")
        from pyvqnet.nn.torch import Dynamic_GRU
        from pyvqnet.tensor import tensor
        seq_len = [4,1,2]
        input_size = 4
        batch_size =3
        hidden_size = 2
        ml = 2
        rnn2 = Dynamic_GRU(input_size,
                        hidden_size=2,
                        num_layers=2,
                        batch_first=False,
                        bidirectional=True)

        a = tensor.arange(1, seq_len[0] * input_size + 1).reshape(
            [seq_len[0], input_size])
        b = tensor.arange(1, seq_len[1] * input_size + 1).reshape(
            [seq_len[1], input_size])
        c = tensor.arange(1, seq_len[2] * input_size + 1).reshape(
            [seq_len[2], input_size])

        y = tensor.pad_sequence([a, b, c], False)

        input = tensor.pack_pad_sequence(y,
                                        seq_len,
                                        batch_first=False,
                                        enforce_sorted=False)

        h0 = tensor.ones([ml * 2, batch_size, hidden_size])

        output, hn = rnn2(input, h0)

        seq_unpacked, lens_unpacked = \
        tensor.pad_packed_sequence(output, batch_first=False)

Dynamic_RNN
^^^^^^^^^^^^^^^^^^^^^^^^

.. py:class:: pyvqnet.nn.torch.Dynamic_RNN(input_size, hidden_size, num_layers=1, nonlinearity='tanh', batch_first=True, use_bias=True, bidirectional=False, dtype=None, name: str = "")


    对动态长度输入序列应用循环神经网络（RNN）。

    第一个输入应为通过 ``tensor.PackedSequence`` 类定义的可变长度批处理序列输入。

    ``tensor.PackedSequence`` 类可通过依次调用以下函数构造：``pad_sequence``\ 、``pack_pad_sequence``\ 。

    Dynamic_RNN 的第一个输出也是一个 ``tensor.PackedSequence`` 类，
    可以使用 ``tensor.pad_pack_sequence`` 解包为正常的 QTensor。

    循环神经网络（RNN）模块，使用 :math:`\tanh` 或 :math:`\text{ReLU}` 作为激活函数。支持双向、多层配置。
    单层单向 RNN 的计算公式如下：

    .. math::
        h_t = \tanh(W_{ih} x_t + b_{ih} + W_{hh} h_{(t-1)} + b_{hh})

    如果 :attr:`nonlinearity` 为 ``'relu'``\ ，则 :math:`\text{ReLU}` 将替换 :math:`\tanh`\ 。

    此类继承自 ``pyvqnet.nn.Module`` 和 ``torch.nn.Module``\ ，可作为子模块添加到 torch 模型中。

    此类的 ``_buffers`` 中的数据为 ``torch.Tensor`` 类型。

    此类的 ``_parmeters`` 中的数据为 ``torch.nn.Parameter`` 类型。

    :param input_size: 输入特征维度。
    :param hidden_size: 隐藏特征维度。
    :param num_layers: 堆叠的 RNN 层数，默认值：1。
    :param nonlinearity: 非线性激活函数，默认为 ``'tanh'``\ 。
    :param batch_first: 如果为 True，输入形状为 [batch size, sequence length, feature dimension]，如果为 False，输入形状为 [sequence length, batch size, feature dimension]，默认值为 True。
    :param use_bias: 如果为 False，此模块不应用偏置，默认值：True。
    :param bidirectional: 如果为 True，变为双向 RNN，默认值：False。
    :param dtype: 参数的数据类型，默认值：None，使用默认数据类型：kfloat32，表示 32 位浮点数。
    :param name: 此模块的名称，默认为 ""。

    :return: Dynamic_RNN 实例

    示例::

        import pyvqnet
        pyvqnet.backends.set_backend("torch")
        from pyvqnet.nn.torch import Dynamic_RNN
        from pyvqnet.tensor import tensor
        seq_len = [4,1,2]
        input_size = 4
        batch_size =3
        hidden_size = 2
        ml = 2
        rnn2 = Dynamic_RNN(input_size,
                        hidden_size=2,
                        num_layers=2,
                        batch_first=False,
                        bidirectional=True,
                        nonlinearity='relu')

        a = tensor.arange(1, seq_len[0] * input_size + 1).reshape(
            [seq_len[0], input_size])
        b = tensor.arange(1, seq_len[1] * input_size + 1).reshape(
            [seq_len[1], input_size])
        c = tensor.arange(1, seq_len[2] * input_size + 1).reshape(
            [seq_len[2], input_size])

        y = tensor.pad_sequence([a, b, c], False)

        input = tensor.pack_pad_sequence(y,
                                        seq_len,
                                        batch_first=False,
                                        enforce_sorted=False)

        h0 = tensor.ones([ml * 2, batch_size, hidden_size])

        output, hn = rnn2(input, h0)

        seq_unpacked, lens_unpacked = \
        tensor.pad_packed_sequence(output, batch_first=False)




Dynamic_LSTM
^^^^^^^^^^^^^^^^^^^^^^^^

.. py:class:: pyvqnet.nn.torch.Dynamic_LSTM(input_size, hidden_size, num_layers=1, batch_first=True, use_bias=True, bidirectional=False, dtype=None, name: str = "")


    对动态长度输入序列应用长短期记忆（LSTM）RNN。

    第一个输入应为通过 ``tensor.PackedSequence`` 类定义的可变长度批处理序列输入。

    ``tensor.PackedSequence`` 类可通过依次调用以下函数构造：``pad_sequence``\ 、``pack_pad_sequence``\ 。

    Dynamic_LSTM 的第一个输出也是一个 ``tensor.PackedSequence`` 类，
    可以使用 ``tensor.pad_pack_sequence`` 解包为正常的 QTensor。

    循环神经网络（RNN）模块，使用 :math:`\tanh` 或 :math:`\text{ReLU}` 作为激活函数。支持双向、多层配置。
    单层单向 RNN 的计算公式如下：


    .. math::
        \begin{array}{ll} \\
            i_t = \sigma(W_{ii} x_t + b_{ii} + W_{hi} h_{t-1} + b_{hi}) \\
            f_t = \sigma(W_{if} x_t + b_{if} + W_{hf} h_{t-1} + b_{hf}) \\
            g_t = \tanh(W_{ig} x_t + b_{ig} + W_{hg} h_{t-1} + b_{hg}) \\
            o_t = \sigma(W_{io} x_t + b_{io} + W_{ho} h_{t-1} + b_{ho}) \\
            c_t = f_t \odot c_{t-1} + i_t \odot g_t \\
            h_t = o_t \odot \tanh(c_t) \\
        \end{array}

    此类继承自 ``pyvqnet.nn.Module`` 和 ``torch.nn.Module``\ ，可作为子模块添加到 torch 模型中。

    此类的 ``_buffers`` 中的数据为 ``torch.Tensor`` 类型。

    此类的 ``_parmeters`` 中的数据为 ``torch.nn.Parameter`` 类型。

    :param input_size: 输入特征维度。
    :param hidden_size: 隐藏特征维度。
    :param num_layers: 堆叠的 LSTM 层数，默认值：1。
    :param batch_first: 如果为 True，输入形状为 [batch size, sequence length, feature dimension]，如果为 False，输入形状为 [sequence length, batch size, feature dimension]，默认值为 True。
    :param use_bias: 如果为 False，此模块不应用偏置，默认值：True。
    :param bidirectional: 如果为 True，变为双向 LSTM，默认值：False。
    :param dtype: 参数的数据类型，默认值：None，使用默认数据类型：kfloat32，表示 32 位浮点数。
    :param name: 此模块的名称，默认为 ""。

    :return: Dynamic_LSTM 实例

    示例::

        import pyvqnet
        pyvqnet.backends.set_backend("torch")
        from pyvqnet.nn.torch import Dynamic_LSTM
        from pyvqnet.tensor import tensor

        input_size = 2
        hidden_size = 2
        ml = 2
        seq_len = [3, 4, 1]
        batch_size = 3
        rnn2 = Dynamic_LSTM(input_size,
                            hidden_size=hidden_size,
                            num_layers=ml,
                            batch_first=False,
                            bidirectional=True)

        a = tensor.arange(1, seq_len[0] * input_size + 1).reshape(
            [seq_len[0], input_size])
        b = tensor.arange(1, seq_len[1] * input_size + 1).reshape(
            [seq_len[1], input_size])
        c = tensor.arange(1, seq_len[2] * input_size + 1).reshape(
            [seq_len[2], input_size])
        a.requires_grad = True
        b.requires_grad = True
        c.requires_grad = True
        y = tensor.pad_sequence([a, b, c], False)

        input = tensor.pack_pad_sequence(y,
                                        seq_len,
                                        batch_first=False,
                                        enforce_sorted=False)

        h0 = tensor.ones([ml * 2, batch_size, hidden_size])
        c0 = tensor.ones([ml * 2, batch_size, hidden_size])

        output, (hn, cn) = rnn2(input, (h0, c0))

        seq_unpacked, lens_unpacked = \
        tensor.pad_packed_sequence(output, batch_first=False)

 


Interpolate
^^^^^^^^^^^^^^^^^^^^^^^^

.. py:class:: pyvqnet.nn.torch.Interpolate(size = None, scale_factor = None, mode = "nearest", align_corners = None,  recompute_scale_factor = None, name = "")

    对输入进行下采样/上采样。

    当前仅支持 4D 输入数据。

    输入大小解释为 `B x C x H x W`\ 。

    可用的 `mode` 选项有 ``nearest``\ 、``bilinear``\ 、``bicubic``\ 。

    此类继承自 ``pyvqnet.nn.Module`` 和 ``torch.nn.Module``\ ，可作为子模块添加到 torch 模型中。

    :param size: 输出大小，默认为 None。
    :param scale_factor: 缩放因子，默认为 None。
    :param mode: 用于上采样的算法 ``nearest`` | ``bilinear`` | ``bicubic``\ 。
    :param align_corners: 从几何角度来看，我们将输入和输出的像素视为正方形而不是点。如果设置为 `true`\ ，输入和输出张量将通过其角像素的中心点对齐，角像素的值被保留。如果设置为 `false`\ ，输入和输出张量将通过其角像素的角点对齐，角像素的值被保留，插值将使用边缘值进行填充，此操作独立于输入大小。当 ``scale_factor`` 保持不变时，这仅在 ``mode`` 为 ``bilinear`` 时起作用。
    :param recompute_scale_factor: 重新计算用于插值计算的缩放因子。当 ``scale_factor`` 作为参数传递时，它将用于计算输出大小。
    :param name: 模块名称。

    示例::

        from pyvqnet.nn.torch import Interpolate
        from pyvqnet.tensor import tensor
        import pyvqnet
        pyvqnet.backends.set_backend("torch")
        pyvqnet.utils.set_random_seed(1)

        mode_ = "bilinear"
        size_ = 3

        model = Interpolate(size=size_, mode=mode_)
        input_vqnet = tensor.randu((1, 1, 6, 6),
                                dtype=pyvqnet.kfloat32,
                                requires_grad=True)
        output_vqnet = model(input_vqnet)

SDPA
^^^^^^^^^^^^^^^^^^^^^^^^

.. py:class:: pyvqnet.nn.torch.SDPA(attn_mask=None,dropout_p=0.,scale=None,is_causal=False)

    构造一个计算查询、键和值张量的缩放点积注意力的类。

    此类继承自 ``pyvqnet.nn.Module`` 和 ``torch.nn.Module``\ ，可作为子模块添加到 torch 模型中。

    :param attn_mask: 注意力掩码；默认值：None。形状必须可广播为注意力权重的形状。
    :param dropout_p: Dropout 概率；默认值：0，如果大于 0.0，则应用 dropout。
    :param scale: 在 softmax 之前应用的缩放因子，默认值：None。
    :param is_causal: 默认值：False，如果设置为 true，则在掩码为方阵时，注意力掩码为下三角矩阵。如果同时设置了 attn_mask 和 is_causal，则会引发错误。
    :return: 一个 SDPA 类

    示例::

        from pyvqnet.nn.torch import SDPA
        from pyvqnet import tensor
        import pyvqnet
        pyvqnet.backends.set_backend("torch")
        model = SDPA(tensor.QTensor([1.]))

   .. py:method:: forward(query,key,value)

        执行前向计算。

        :param query: 查询输入 QTensor。
        :param key: 键输入 QTensor。
        :param value: 值输入 QTensor。
        :return: SDPA 计算返回的 QTensor。

        示例::

            from pyvqnet.nn.torch import SDPA
            from pyvqnet import tensor
            import pyvqnet
            pyvqnet.backends.set_backend("torch")

            import numpy as np

            model = SDPA(tensor.QTensor([1.]))

            query_np = np.random.randn(3, 3, 3, 5).astype(np.float32)
            key_np = np.random.randn(3, 3, 3, 5).astype(np.float32)
            value_np = np.random.randn(3, 3, 3, 5).astype(np.float32)

            query_p = tensor.QTensor(query_np, dtype=pyvqnet.kfloat32, requires_grad=True)
            key_p = tensor.QTensor(key_np, dtype=pyvqnet.kfloat32, requires_grad=True)
            value_p = tensor.QTensor(value_np, dtype=pyvqnet.kfloat32, requires_grad=True)

            out_sdpa = model(query_p, key_p, value_p)

            out_sdpa.backward()

损失函数 API
------------------------

MeanSquaredError
^^^^^^^^^^^^^^^^^^^^^^^^

.. py:class:: pyvqnet.nn.torch.MeanSquaredError(name="")

    计算输入 :math:`x` 和目标值 :math:`y` 之间的均方根误差。

    如果平方根误差可以通过以下函数描述：

    .. math::
        \ell(x, y) = L = \{l_1,\dots,l_N\}^\top, \quad
        l_n = \left( x_n - y_n \right)^2,

    :math:`x` 和 :math:`y` 是任意形状的 QTensor，计算总共 :math:`n` 个元素的均方根误差如下：

    .. math::
        \ell(x, y) =
        \operatorname{mean}(L)

    此类继承自 ``pyvqnet.nn.Module`` 和 ``torch.nn.Module``\ ，可作为子模块添加到 torch 模型中。

    :param name: 此模块的名称，默认为 ""。
    :return: 一个均方根误差实例。

    均方根误差前向计算函数所需参数：

        x: :math:`(N, *)` 预测值，其中 :math:`*` 表示任意维度。

        y: :math:`(N, *)`\ ，目标值，与输入相同维度的 QTensor。

    .. note::

        请注意，与 pytorch 等框架不同，在以下 MeanSquaredError 函数的前向函数中，第一个参数是目标值，第二个参数是预测值。


    示例::

        from pyvqnet.tensor import QTensor
        from pyvqnet import kfloat64
        from pyvqnet.nn.torch import MeanSquaredError
        import pyvqnet
        pyvqnet.backends.set_backend("torch")
        y = QTensor([[0, 0, 1, 0, 0, 0, 0, 0, 0, 0]],
                    requires_grad=False,
                    dtype=kfloat64)
        x = QTensor([[0.1, 0.05, 0.7, 0, 0.05, 0.1, 0, 0, 0, 0]],
                    requires_grad=True,
                    dtype=kfloat64)

        loss_result = MeanSquaredError()
        result = loss_result(y, x)
        print(result)



BinaryCrossEntropy
^^^^^^^^^^^^^^^^^^^^^^^^

.. py:class:: pyvqnet.nn.torch.BinaryCrossEntropy(name="")

    衡量目标值和输入之间的平均二分类交叉熵损失。

    未平均的二分类交叉熵如下：

    .. math::
        \ell(x, y) = L = \{l_1,\dots,l_N\}^\top, \quad
        l_n = - w_n \left[ y_n \cdot \log x_n + (1 - y_n) \cdot \log (1 - x_n) \right],

    其中 :math:`N` 是批大小。

    .. math::
        \ell(x, y) = \operatorname{mean}(L)

    此类继承自 ``pyvqnet.nn.Module`` 和 ``torch.nn.Module``\ ，可作为子模块添加到 torch 模型中。

    :param name: 此模块的名称，默认为 ""。
    :return: 一个平均二分类交叉熵实例。

    平均二分类交叉熵误差前向计算函数所需参数：

        x: :math:`(N, *)` 预测值，其中 :math:`*` 表示任意维度。

        y: :math:`(N, *)`\ ，目标值，与输入相同维度的 QTensor。

    .. note::

        请注意，与 pytorch 等框架不同，在 BinaryCrossEntropy 函数的前向函数中，第一个参数是目标值，第二个参数是预测值。

    示例::

        from pyvqnet.tensor import QTensor
        from pyvqnet.nn.torch import BinaryCrossEntropy
        import pyvqnet
        pyvqnet.backends.set_backend("torch")
        x = QTensor([[0.3, 0.7, 0.2], [0.2, 0.3, 0.1]], requires_grad=True)
        y = QTensor([[0.0, 1.0, 0], [0.0, 0, 1]], requires_grad=False)

        loss_result = BinaryCrossEntropy()
        result = loss_result(y, x)
        result.backward()
        print(result)


CategoricalCrossEntropy
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. py:class:: pyvqnet.nn.torch.CategoricalCrossEntropy(name="")

    此损失函数结合了 LogSoftmax 和 NLLLoss 来计算平均分类交叉熵。

    损失函数计算如下，其中 class 是目标值对应的类别标签：

    .. math::
        \text{loss}(x, y) = -\log\left(\frac{\exp(x[class])}{\sum_j \exp(x[j])}\right)
        = -x[class] + \log\left(\sum_j \exp(x[j])\right)

    :param name: 此模块的名称，默认为 ""。
    :return: 平均分类交叉熵实例。

    误差前向计算函数所需参数：

        x: :math:`(N, *)` 预测值，其中 :math:`*` 表示任意维度。

        y: :math:`(N, *)`\ ，目标值，与输入相同维度的 QTensor。必须是 64 位整数，kint64。

    .. note::

        请注意，与 pytorch 等框架不同，在 CategoricalCrossEntropy 函数的前向函数中，第一个参数是目标值，第二个参数是预测值。

        此类继承自 ``pyvqnet.nn.Module`` 和 ``torch.nn.Module``\ ，可作为子模块添加到 torch 模型中。

    示例::

        from pyvqnet.tensor import QTensor
        from pyvqnet import kfloat32,kint64
        from pyvqnet.nn.torch import CategoricalCrossEntropy
        import pyvqnet
        pyvqnet.backends.set_backend("torch")
        x = QTensor([[1, 2, 3, 4, 5],
        [1, 2, 3, 4, 5],
        [1, 2, 3, 4, 5]], requires_grad=True,dtype=kfloat32)
        y = QTensor([[0, 1, 0, 0, 0], [0, 1, 0, 0, 0], [1, 0, 0, 0, 0]], requires_grad=False,dtype=kint64)
        loss_result = CategoricalCrossEntropy()
        result = loss_result(y, x)
        print(result)



SoftmaxCrossEntropy
^^^^^^^^^^^^^^^^^^^^^^^^

.. py:class:: pyvqnet.nn.torch.SoftmaxCrossEntropy(name="")

    此损失函数结合了 LogSoftmax 和 NLLLoss 来计算平均分类交叉熵，并具有更高的数值稳定性。

    损失函数计算如下，其中 class 是目标值对应的分类标签：

    .. math::
        \text{loss}(x, y) = -\log\left(\frac{\exp(x[class])}{\sum_j \exp(x[j])}\right)
        = -x[class] + \log\left(\sum_j \exp(x[j])\right)

    :param name: 此模块的名称，默认为 ""。
    :return: 一个 Softmax 交叉熵损失函数实例

    误差前向计算函数所需参数：

        x: :math:`(N, *)` 预测值，其中 :math:`*` 表示任意维度。

        y: :math:`(N, *)`\ ，目标值，与输入相同维度的 QTensor。必须是 64 位整数，kint64。

    .. note::

        请注意，与 pytorch 等框架不同，在 SoftmaxCrossEntropy 函数的前向函数中，第一个参数是目标值，第二个参数是预测值。

        此类继承自 ``pyvqnet.nn.Module`` 和 ``torch.nn.Module``\ ，可作为子模块添加到 torch 模型中。

    示例::

        from pyvqnet.tensor import QTensor
        from pyvqnet import kfloat32, kint64
        from pyvqnet.nn.torch import SoftmaxCrossEntropy
        import pyvqnet
        pyvqnet.backends.set_backend("torch")
        x = QTensor([[1, 2, 3, 4, 5], [1, 2, 3, 4, 5], [1, 2, 3, 4, 5]],
                    requires_grad=True,
                    dtype=kfloat32)
        y = QTensor([[0, 1, 0, 0, 0], [0, 1, 0, 0, 0], [1, 0, 0, 0, 0]],
                    requires_grad=False,
                    dtype=kint64)
        loss_result = SoftmaxCrossEntropy()
        result = loss_result(y, x)
        result.backward()
        print(result)



NLL_Loss
^^^^^^^^^^^^^^^^^^^^^^^^

.. py:class:: pyvqnet.nn.torch.NLL_Loss(name="")


    平均负对数似然损失。适用于 C 类别的分类问题。

    `x` 是模型给出的概率似然。其形状可以是 :math:`(N, C)` 或 :math:`(N, C, d_1, d_2, ..., d_K)`\ 。`y` 是损失函数的期望真值，包含 :math:`[0, C-1]` 中的类别索引。

    .. math::
        \ell(x, y) = L = \{l_1,\dots,l_N\}^\top, \quad
        l_n = -
        \sum_{n=1}^N \frac{1}{N}x_{n,y_n} \quad

    :param name: 此模块的名称，默认为 ""。
    :return: 一个 NLL_Loss 损失函数实例

    误差前向计算函数所需参数：

        x: :math:`(N, *)`\ ，损失函数的输出预测值，可以是多维变量。

        y: :math:`(N, *)`\ ，损失函数的目标值。必须是 64 位整数，kint64。

    .. note::

        请注意，与 pytorch 等框架不同，在 NLL_Loss 函数的前向函数中，第一个参数是目标值，第二个参数是预测值。

        此类继承自 ``pyvqnet.nn.Module`` 和 ``torch.nn.Module``\ ，可作为子模块添加到 torch 模型中。

    示例::

        from pyvqnet.tensor import QTensor
        from pyvqnet import kint64
        from pyvqnet.nn.torch import NLL_Loss
        import pyvqnet
        pyvqnet.backends.set_backend("torch")
        x = QTensor([
            0.9476322568516703, 0.226547421131723, 0.5944201443911326,
            0.42830868492969476, 0.76414068655387, 0.00286059168094277,
            0.3574236812873617, 0.9096948856639084, 0.4560809854582528,
            0.9818027091583286, 0.8673569904602182, 0.9860275114020933,
            0.9232667066664217, 0.303693313961628, 0.8461034903175555
        ])
        x=x.reshape([1, 3, 1, 5])
        x.requires_grad = True
        y = QTensor([[[2, 1, 0, 0, 2]]], dtype=kint64)

        loss_result = NLL_Loss()
        result = loss_result(y, x)
        print(result)


CrossEntropyLoss
^^^^^^^^^^^^^^^^^^^^^^^^

.. py:class:: pyvqnet.nn.torch.CrossEntropyLoss(name="")

    此函数将 LogSoftmax 和 NLL_Loss 的损失合并计算。

    `x` 包含未归一化的输出。其形状可以是 :math:`(C)`\ 、:math:`(N, C)` 二维或 :math:`(N, C, d_1, d_2, ..., d_K)` 多维。

    损失函数的公式如下，其中 class 是目标值对应的类别标签：

    .. math::
        \text{loss}(x, y) = -\log\left(\frac{\exp(x[class])}{\sum_j \exp(x[j])}\right)
        = -x[class] + \log\left(\sum_j \exp(x[j])\right)

    :param name: 此模块的名称，默认为 ""。
    :return: 一个 CrossEntropyLoss 损失函数实例

    误差前向计算函数所需参数：

        x: :math:`(N, *)`\ ，损失函数的输出，可以是多维变量。

        y: :math:`(N, *)`\ ，损失函数的期望真值。必须是 64 位整数，kint64。

    .. note::

        请注意，与 pytorch 等框架不同，在 CrossEntropyLoss 函数的前向函数中，第一个参数是目标值，第二个参数是预测值。

        此类继承自 ``pyvqnet.nn.Module`` 和 ``torch.nn.Module``\ ，可作为子模块添加到 torch 模型中。

    示例::

        from pyvqnet.tensor import QTensor
        from pyvqnet import kint64
        from pyvqnet.nn.torch import CrossEntropyLoss
        import pyvqnet
        pyvqnet.backends.set_backend("torch")
        x = QTensor([
            0.9476322568516703, 0.226547421131723, 0.5944201443911326,
            0.42830868492969476, 0.76414068655387, 0.00286059168094277,
            0.3574236812873617, 0.9096948856639084, 0.4560809854582528,
            0.9818027091583286, 0.8673569904602182, 0.9860275114020933,
            0.9232667066664217, 0.303693313961628, 0.8461034903175555
        ])
        x=x.reshape([1, 3, 1, 5])
        x.requires_grad = True
        y = QTensor([[[2, 1, 0, 0, 2]]], dtype=kint64)

        loss_result = CrossEntropyLoss()
        result = loss_result(y, x)
        print(result)


激活函数
---------------------

Sigmoid
^^^^^^^^^^^^^^^^^^^^^^^^

.. py:class:: pyvqnet.nn.torch.Sigmoid(name:str="")

    Sigmoid 激活函数层。

    .. math::
        \text{Sigmoid}(x) = \frac{1}{1 + \exp(-x)}

    此类继承自 ``pyvqnet.nn.Module`` 和 ``torch.nn.Module``\ ，可作为子模块添加到 torch 模型中。

    :param name: 激活函数层的名称，默认为 ""。

    :return: 一个 Sigmoid 激活函数层实例。

    示例::

        from pyvqnet.nn.torch import Sigmoid
        from pyvqnet.tensor import QTensor
        import pyvqnet
        pyvqnet.backends.set_backend("torch")
        layer = Sigmoid()
        y = layer(QTensor([1.0, 2.0, 3.0, 4.0]))
        print(y)


Softplus
^^^^^^^^^^^^^^^^^^^^^^^^

.. py:class:: pyvqnet.nn.torch.Softplus(name:str="")

    Softplus

    .. math::
        \text{Softplus}(x) = \log(1 + \exp(x))

    此类继承自 ``pyvqnet.nn.Module`` 和 ``torch.nn.Module``\ ，可作为子模块添加到 torch 模型中。

    :param name: 激活函数层的名称，默认为 ""。

    :return: 一个 Softplus 实例。

    示例::

        from pyvqnet.nn.torch import Softplus
        from pyvqnet.tensor import QTensor
        import pyvqnet
        pyvqnet.backends.set_backend("torch")
        layer = Softplus()
        y = layer(QTensor([1.0, 2.0, 3.0, 4.0]))

Softsign
^^^^^^^^^^^^^^^^^^^^^^^^

.. py:class:: pyvqnet.nn.torch.Softsign(name:str="")

    Softsign。

    .. math::
        \text{SoftSign}(x) = \frac{x}{ 1 + |x|}


    此类继承自 ``pyvqnet.nn.Module`` 和 ``torch.nn.Module``\ ，可作为子模块添加到 torch 模型中。


    :param name: 激活函数层的名称，默认为 ""。

    :return: 一个 SoftSign 实例。

    示例::

        from pyvqnet.nn.torch import Softsign
        from pyvqnet.tensor import QTensor
        import pyvqnet
        pyvqnet.backends.set_backend("torch")
        layer = Softsign()
        y = layer(QTensor([1.0, 2.0, 3.0, 4.0]))



Softmax
^^^^^^^^^^^^^^^^^^^^^^^^

.. py:class:: pyvqnet.nn.torch.Softmax(axis:int = -1,name:str="")

    Softmax

    .. math::
        \text{Softmax}(x_{i}) = \frac{\exp(x_i)}{\sum_j \exp(x_j)}


    此类继承自 ``pyvqnet.nn.Module`` 和 ``torch.nn.Module``\ ，可作为子模块添加到 torch 模型中。


    :param axis: 要计算的维度（最后一个轴为 -1），默认值 = -1。
    :param name: 激活函数层的名称，默认为 ""。

    :return: 一个 Softmax 实例。

    示例::

        from pyvqnet.nn.torch import Softmax
        from pyvqnet.tensor import QTensor
        import pyvqnet
        pyvqnet.backends.set_backend("torch")
        layer = Softmax()
        y = layer(QTensor([1.0, 2.0, 3.0, 4.0]))


HardSigmoid
^^^^^^^^^^^^^^^^^^^^^^^^

.. py:class:: pyvqnet.nn.torch.HardSigmoid(name:str="")

    HardSigmoid

    .. math::
        \text{Hardsigmoid}(x) = \begin{cases}
            0 & \text{ if } x \le -3, \\
            1 & \text{ if } x \ge +3, \\
            x / 6 + 1 / 2 & \text{otherwise}
        \end{cases}


    此类继承自 ``pyvqnet.nn.Module`` 和 ``torch.nn.Module``\ ，可作为子模块添加到 torch 模型中。


    :param name: 激活函数层的名称，默认为 ""。

    :return: HardSigmoid 实例。

    示例::

        from pyvqnet.nn.torch import HardSigmoid
        from pyvqnet.tensor import QTensor
        import pyvqnet
        pyvqnet.backends.set_backend("torch")
        layer = HardSigmoid()
        y = layer(QTensor([1.0, 2.0, 3.0, 4.0]))


ReLu
^^^^^^^^^^^^^^^^^^^^^^^^

.. py:class:: pyvqnet.nn.torch.ReLu(name:str="")

    ReLu。

    .. math::
        \text{ReLu}(x) = \begin{cases}
        x, & \text{ if } x > 0\\
        0, & \text{ if } x \leq 0
        \end{cases}


    此类继承自 ``pyvqnet.nn.Module`` 和 ``torch.nn.Module``\ ，可作为子模块添加到 torch 模型中。


    :param name: 激活函数层的名称，默认为 ""。

    :return: 一个 ReLu 实例。

    示例::

        from pyvqnet.nn.torch import ReLu
        from pyvqnet.tensor import QTensor
        import pyvqnet
        pyvqnet.backends.set_backend("torch")
        layer = ReLu()
        y = layer(QTensor([-1, 2.0, -3, 4.0]))




LeakyReLu
^^^^^^^^^^^^^^^^^^^^^^^^

.. py:class:: pyvqnet.nn.torch.LeakyReLu(alpha:float=0.01,name:str="")

    LeakyReLu

    .. math::
        \text{LeakyRelu}(x) =
        \begin{cases}
        x, & \text{ if } x \geq 0 \\
        \alpha * x, & \text{ otherwise }
        \end{cases}


    此类继承自 ``pyvqnet.nn.Module`` 和 ``torch.nn.Module``\ ，可作为子模块添加到 torch 模型中。


    :param alpha: LeakyRelu 系数，默认值：0.01。
    :param name: 激活函数层的名称，默认为 ""。

    :return: 一个 LeakyReLu 激活实例。

    示例::

        from pyvqnet.nn.torch import LeakyReLu
        from pyvqnet.tensor import QTensor
        import pyvqnet
        pyvqnet.backends.set_backend("torch")
        layer = LeakyReLu()
        y = layer(QTensor([-1, 2.0, -3, 4.0]))



Gelu
^^^^^^^^^^^^^^^^^^^^^^^^

.. py:class:: pyvqnet.nn.torch.Gelu(approximate="tanh", name="")

    Gelu:

    .. math:: \text{GELU}(x) = x * \Phi(x)

    当近似参数为 'tanh' 时，GELU 估计如下：

    .. math:: \text{GELU}(x) = 0.5 * x * (1 + \text{Tanh}(\sqrt{2 / \pi} * (x + 0.044715 * x^3)))


    此类继承自 ``pyvqnet.nn.Module`` 和 ``torch.nn.Module``\ ，可作为子模块添加到 torch 模型中。


    :param approximate: 近似计算方法，默认为 "tanh"。
    :param name: 激活函数层的名称，默认为 ""。

    :return: Gelu 激活实例。

    示例::

        from pyvqnet.tensor import randu, ones_like
        from pyvqnet.nn.torch import Gelu
        import pyvqnet
        pyvqnet.backends.set_backend("torch")
        qa = randu([5,4])
        qb = Gelu()(qa)



ELU
^^^^^^^^^^^^^^^^^^^^^^^^

.. py:class:: pyvqnet.nn.torch.ELU(alpha:float=1,name:str="")

    ELU 指数线性单元激活函数层。

    .. math::
        \text{ELU}(x) = \begin{cases}
        x, & \text{ if } x > 0\\
        \alpha * (\exp(x) - 1), & \text{ if } x \leq 0
        \end{cases}


    此类继承自 ``pyvqnet.nn.Module`` 和 ``torch.nn.Module``\ ，可作为子模块添加到 torch 模型中。



    :param alpha: ELU 系数，默认值：1。
    :param name: 激活函数层的名称，默认为 ""。

    :return: ELU 激活实例。

    示例::

        from pyvqnet.nn.torch import ELU
        from pyvqnet.tensor import QTensor
        import pyvqnet
        pyvqnet.backends.set_backend("torch")
        layer = ELU()
        y = layer(QTensor([-1, 2.0, -3, 4.0]))


Tanh
^^^^^^^^^^^^^^^^^^^^^^^^

.. py:class:: pyvqnet.nn.torch.Tanh(name:str="")

    Tanh 双曲正切激活函数。

    .. math::
        \text{Tanh}(x) = \frac{\exp(x) - \exp(-x)} {\exp(x) + \exp(-x)}


    此类继承自 ``pyvqnet.nn.Module`` 和 ``torch.nn.Module``\ ，可作为子模块添加到 torch 模型中。



    :param name: 激活函数层的名称，默认为 ""。

    :return: Tanh 激活实例。

    示例::

        from pyvqnet.nn.torch import Tanh
        from pyvqnet.tensor import QTensor
        import pyvqnet
        pyvqnet.backends.set_backend("torch")
        layer = Tanh()
        y = layer(QTensor([-1, 2.0, -3, 4.0]))



优化器模块
---------------------------------------------

对于继承自 `TorchModule` 的经典和量子电路模块，参数 `model.paramters()` 可以继续使用 :ref:`Optimizer` 下除 `Rotosolve` 之外的优化器进行优化。



使用 pyqpanda 运行变分量子电路
-------------------------------------------------------------------------

以下是在电路计算中使用 pyqpanda 和 pyqpanda3 的训练变分量子电路接口。

.. warning::

    以下 TorchQpandaQuantumLayer 的量子计算部分使用 pyqpanda2。

    由于 pyqpanda2 和 pyqpanda3 之间存在兼容性问题，您需要自行安装 pyqpnda2，`pip install pyqpanda`

TorchQpandaQuantumLayer
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

如果您更熟悉 pyQPanda2 语法，可以使用接口 TorchQpandaQuantumLayer，在 TorchQpandaQuantumLayer 的参数 ``qprog_with_measure`` 函数中添加自定义量子比特 ``qubits``\ 、经典比特 ``cbits`` 和后端模拟器 ``machine``\ 。

.. py:class:: pyvqnet.qnn.vqc.torch.TorchQpandaQuantumLayer(qprog_with_measure,para_num,diff_method:str = "parameter_shift",delta:float = 0.01,dtype=None,name="")

    变分量子层的抽象计算模块。使用 pyQPanda2 模拟参数化量子电路并获取测量结果。该变分量子层继承了 VQNet 框架的梯度计算模块。可以使用参数漂移方法计算电路参数的梯度，训练变分量子电路模型或将变分量子电路嵌入到混合量子经典模型中。

    :param qprog_with_measure: 使用 pyQPand 构建的量子电路操作和测量函数。
    :param para_num: `int` - 参数数量。
    :param diff_method: 求解量子电路参数梯度的方法，"parameter shift" 或 "finite difference"，默认为 parameter shift。
    :param delta: 通过有限差分计算梯度时的 \delta。
    :param dtype: 参数的数据类型，默认值：None，使用默认数据类型：kfloat32，表示 32 位浮点数。
    :param name: 此模块的名称，默认为 ""。

    :return: 一个可以计算量子电路的模块。

    .. note::

        qprog_with_measure 是在 pyQPanda2 中定义的量子电路函数。

        此函数必须包含以下参数作为函数输入（即使某个参数实际上并未使用），否则在该函数中将无法正常工作。

        与 QuantumLayer 相比，在此接口传入的变分电路运行函数中，用户应手动创建量子比特和模拟器。

        如果 qprog_with_measure 需要量子测量，用户还需要手动创建并分配 cbits。

        量子电路函数 qprog_with_measure (input, param, nqubits, ncbits) 的使用可参考以下示例。

        `input`\ ：输入一维经典数据。如果没有，则输入 None。

        `param`\ ：输入一维待训练的变分量子电路参数。

    示例::

        import pyqpanda as pq
        from pyvqnet.qnn import ProbsMeasure
        import numpy as np
        from pyvqnet.tensor import QTensor
        import pyvqnet
        pyvqnet.backends.set_backend("torch")
        from pyvqnet.qnn.vqc.torch import TorchQpandaQuantumLayer
        def pqctest (input,param):
            num_of_qubits = 4

            m_machine = pq.CPUQVM()# outside
            m_machine.init_qvm()# outside
            qubits = m_machine.qAlloc_many(num_of_qubits)

            circuit = pq.QCircuit()
            circuit.insert(pq.H(qubits[0]))
            circuit.insert(pq.H(qubits[1]))
            circuit.insert(pq.H(qubits[2]))
            circuit.insert(pq.H(qubits[3]))

            circuit.insert(pq.RZ(qubits[0],input[0]))
            circuit.insert(pq.RZ(qubits[1],input[1]))
            circuit.insert(pq.RZ(qubits[2],input[2]))
            circuit.insert(pq.RZ(qubits[3],input[3]))

            circuit.insert(pq.CNOT(qubits[0],qubits[1]))
            circuit.insert(pq.RZ(qubits[1],param[0]))
            circuit.insert(pq.CNOT(qubits[0],qubits[1]))

            circuit.insert(pq.CNOT(qubits[1],qubits[2]))
            circuit.insert(pq.RZ(qubits[2],param[1]))
            circuit.insert(pq.CNOT(qubits[1],qubits[2]))

            circuit.insert(pq.CNOT(qubits[2],qubits[3]))
            circuit.insert(pq.RZ(qubits[3],param[2]))
            circuit.insert(pq.CNOT(qubits[2],qubits[3]))

            prog = pq.QProg()
            prog.insert(circuit)

            rlt_prob = ProbsMeasure([0,2],prog,m_machine,qubits)
            return rlt_prob

        pqc = TorchQpandaQuantumLayer(pqctest,3)

        #classic data as input
        input = QTensor([[1.0,2,3,4],[4,2,2,3],[3,3,2,2]],requires_grad=True)

        #forward circuits
        rlt = pqc(input)

        print(rlt)

        grad =  QTensor(np.ones(rlt.data.shape)*1000)
        #backward circuits
        rlt.backward(grad)

        print(pqc.m_para.grad)
        print(input.grad)


.. warning::

    以下 TorchQcloud3QuantumLayer 和 TorchQpanda3QuantumLayer 接口的量子计算部分使用 pyqpanda3。

    如果您使用此模块下的 QCloud 功能，在代码中导入 pyqpanda2 或使用 pyvqnet 的 pyqpanda2 相关包接口时会出现错误。

TorchQcloud3QuantumLayer
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

当您安装最新版本的 pyqpanda3 时，可以使用此接口定义变分电路并提交到 originqc 的真实芯片上运行。

.. py:class:: pyvqnet.qnn.vqc.torch.TorchQcloud3QuantumLayer(origin_qprog_func, qcloud_token, para_num, pauli_str_dict=None, shots = 1000, initializer=None, dtype=None, name="", diff_method="parameter_shift", submit_kwargs={}, query_kwargs={})

    使用 pyqpanda3 的 originqc 进行真实芯片计算的抽象计算模块。它将参数化量子电路提交到真实芯片并获取测量结果。
    如果 diff_method == "random_coordinate_descent"，该层将随机选择单个参数计算梯度，其他参数保持为零。参考：https://arxiv.org/abs/2311.00088

    .. note::

        qcloud_token 是您从云平台申请的 API 令牌。

        origin_qprog_func 需要返回 pypqanda3.core.QProg 类型的数据。如果未设置 pauli_str_dict，则需要确保已向 QProg 中插入 measure。

        origin_qprog_func 必须采用以下格式：

        origin_qprog_func(input,param )

        `input`\ ：输入 1~2D 经典数据。在 2D 情况下，第一维是批大小。

        `param`\ ：输入一维变分量子电路的待训练参数。

    .. warning::

        此类继承自 ``pyvqnet.nn.Module`` 和 ``torch.nn.Module``\ ，可作为子模块添加到 torch 模型中。

        此类的 ``_buffers`` 中的数据为 ``torch.Tensor`` 类型。

        此类的 ``_parmeters`` 中的数据为 ``torch.nn.Parameter`` 类型。

    :param origin_qprog_func: 由 QPanda 构建的变分量子电路函数，必须返回 QProg。
    :param qcloud_token: `str` - 要执行的量子机类型或云令牌。
    :param para_num: `int` - 参数数量，参数为大小为 [para_num] 的 QTensor。
    :param pauli_str_dict: `dict|list` - 表示量子电路中泡利算符的字典或字典列表。默认为 "None"，表示执行测量操作。如果输入泡利算符字典，则计算单个期望值或多个期望值。
    :param shot: `int` - 测量次数。默认值为 1000。
    :param initializer: 参数值的初始化器。默认值为 "None"，使用 0~2*pi 的正态分布。
    :param dtype: 参数的数据类型。默认值为 None，表示使用默认数据类型 pyvqnet.kfloat32。
    :param name: 模块的名称。默认值为空字符串。
    :param diff_method: 梯度计算的微分方法。默认值为 "parameter_shift"、"random_coordinate_descent"。
    :param submit_kwargs: 提交量子电路的额外关键字参数，默认值：{"if_print_qcloud_log":False,"chip_id":"WK_C180","is_amend":True,"is_mapping":True,"is_optimization":True,"compile_level":3,"default_task_group_size":200,"test_qcloud_fake":True,"":"server_ip_address"}，当 test_qcloud_fake 设置为 True 时，使用本地 CPUQVM 模拟。
    :param query_kwargs: 查询量子结果的额外关键字参数，默认值：{"timeout":2,"print_query_info":True,"sub_circuits_split_size":1}。
    :return: 一个可以计算量子电路的模块。


    示例::

        import pyqpanda3.core as pq
        import pyvqnet
        from pyvqnet.qnn.vqc.torch import TorchQcloud3QuantumLayer

        pyvqnet.backends.set_backend("torch")
        def qfun(input,param):

            m_qlist = range(6)
            cbits = range(6)
            measure_qubits = [0,2]
            m_prog = pq.QProg()
            cir = pq.QCircuit()
            cir<<pq.RZ(m_qlist[0],input[0])
            cir<<pq.CNOT(m_qlist[0],m_qlist[1])
            cir<<pq.RY(m_qlist[1],param[0])
            cir<<pq.CNOT(m_qlist[0],m_qlist[2])
            cir<<pq.RZ(m_qlist[1],input[1])
            cir<<pq.RY(m_qlist[2],param[1])
            cir<<pq.H(m_qlist[2])
            m_prog<<cir

            for idx, ele in enumerate(measure_qubits):
                m_prog << pq.measure(m_qlist[ele], cbits[idx])  # pylint: disable=expression-not-assigned
            return m_prog

        l = TorchQcloud3QuantumLayer(qfun,
                        "3047DE8A59764BEDAC9C3282093B16AF1",
                        2,
                        pauli_str_dict=None,
                        shots = 1000,
                        initializer=None,
                        dtype=None,
                        name="",
                        diff_method="parameter_shift",
                        submit_kwargs={"test_qcloud_fake":True},
                        query_kwargs={})
        x = pyvqnet.tensor.QTensor([[0.56,1.2],[0.56,1.2],[0.56,1.2],[0.56,1.2],[0.56,1.2]],requires_grad= True)
        y = l(x)
        print(y)
        y.backward()
        print(l.m_para.grad)
        print(x.grad)

        def qfun2(input,param ):

            m_qlist = range(6)
            cbits = range(6)
            measure_qubits = [0,2]
            m_prog = pq.QProg()
            cir = pq.QCircuit()
            cir<<pq.RZ(m_qlist[0],input[0])
            cir<<pq.CNOT(m_qlist[0],m_qlist[1])
            cir<<pq.RY(m_qlist[1],param[0])
            cir<<pq.CNOT(m_qlist[0],m_qlist[2])
            cir<<pq.RZ(m_qlist[1],input[1])
            cir<<pq.RY(m_qlist[2],param[1])
            cir<<pq.H(m_qlist[2])
            m_prog<<cir

            return m_prog
        l = TorchQcloud3QuantumLayer(qfun2,
                "3047DE8A59764BEDAC9C3282093B16AF",
                2,

                pauli_str_dict={'Z0 X1':10,'':-0.5,'Y2':-0.543},
                shots = 1000,
                initializer=None,
                dtype=None,
                name="",
                diff_method="parameter_shift",
                submit_kwargs={"test_qcloud_fake":True},
                query_kwargs={})
        x = pyvqnet.tensor.QTensor([[0.56,1.2],[0.56,1.2],[0.56,1.2],[0.56,1.2]],requires_grad= True)
        y = l(x)
        print(y)
        y.backward()
        print(l.m_para.grad)
        print(x.grad)

TorchQpanda3QuantumLayer
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

如果您更熟悉 pyQPanda3 语法，可以使用接口 TorchQpanda3QuantumLayer。

.. py:class:: pyvqnet.qnn.vqc.torch.TorchQpanda3QuantumLayer(qprog_with_measure,para_num,diff_method:str = "parameter_shift",delta:float = 0.01,dtype=None,name="")

    变分量子层的抽象计算模块。使用 pyQPanda3 模拟参数化量子电路并获取测量结果。该变分量子层继承了 VQNet 框架的梯度计算模块。您可以使用参数漂移方法计算电路参数的梯度，训练变分量子电路模型或将变分量子电路嵌入到混合量子经典模型中。

    :param qprog_with_measure: 使用 pyQPand 构建的量子电路操作和测量函数。
    :param para_num: `int` - 参数数量。
    :param diff_method: 求解量子电路参数梯度的方法，"parameter shift" 或 "finite difference"，默认为 parameter shift。
    :param delta: 通过有限差分计算梯度时的 \delta。
    :param dtype: 参数的数据类型，默认值：None，使用默认数据类型：kfloat32，表示 32 位浮点数。
    :param name: 此模块的名称，默认为 ""。

    :return: 一个可以计算量子电路的模块。

    .. note::

        qprog_with_measure 是在 pyQPanda 中定义的量子电路函数。

        此函数必须包含以下参数作为函数输入（即使某个参数实际上并未使用），否则在该函数中将无法正常工作。

        量子电路函数 qprog_with_measure (input,param,nqubits,ncbits) 的使用可参考以下示例。

        `input`\ ：输入一维经典数据。如果没有，则输入 None。

        `param`\ ：输入一维变分量子电路的待训练参数。

    示例::

        import pyqpanda3.core as pq
        from pyvqnet.qnn.pq3 import ProbsMeasure
        import numpy as np
        from pyvqnet.tensor import QTensor
        import pyvqnet
        pyvqnet.backends.set_backend("torch")
        from pyvqnet.qnn.vqc.torch import TorchQpanda3QuantumLayer
        def pqctest (input,param):
            num_of_qubits = 4

            m_machine = pq.CPUQVM()# outside

            qubits =range(num_of_qubits)

            circuit = pq.QCircuit()
            circuit<<pq.H(qubits[0])
            circuit<<pq.H(qubits[1])
            circuit<<pq.H(qubits[2])
            circuit<<pq.H(qubits[3])

            circuit<<pq.RZ(qubits[0],input[0])
            circuit<<pq.RZ(qubits[1],input[1])
            circuit<<pq.RZ(qubits[2],input[2])
            circuit<<pq.RZ(qubits[3],input[3])

            circuit<<pq.CNOT(qubits[0],qubits[1])
            circuit<<pq.RZ(qubits[1],param[0])
            circuit<<pq.CNOT(qubits[0],qubits[1])

            circuit<<pq.CNOT(qubits[1],qubits[2])
            circuit<<pq.RZ(qubits[2],param[1])
            circuit<<pq.CNOT(qubits[1],qubits[2])

            circuit<<pq.CNOT(qubits[2],qubits[3])
            circuit<<pq.RZ(qubits[3],param[2])
            circuit<<pq.CNOT(qubits[2],qubits[3])

            prog = pq.QProg()
            prog<<circuit

            rlt_prob = ProbsMeasure(m_machine,prog,[0,2])
            return rlt_prob

        pqc = TorchQpanda3QuantumLayer(pqctest,3)

        #classic data as input
        input = QTensor([[1.0,2,3,4],[4,2,2,3],[3,3,2,2]],requires_grad=True)

        #forward circuits
        rlt = pqc(input)

        print(rlt)

        grad =  QTensor(np.ones(rlt.data.shape)*1000)
        #backward circuits
        rlt.backward(grad)

        print(pqc.m_para.grad)
        print(input.grad)

基于自动微分的变分量子电路模块与接口
---------------------------------------------------------------------------------------------
基类
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

编写变分量子电路模型需要继承 ``QModule``\ 。

QModule
""""""""""""""""""

.. py:class:: pyvqnet.qnn.vqc.torch.QModule(name="")

    当用户使用 `torch` 后端时，定义量子变分电路模型 `Module` 应继承的基类。
    此类继承自 ``pyvqnet.nn.torch.TorchModule`` 和 ``torch.nn.Module``\ 。

    此类可作为子模块添加到 torch 模型中。

    .. note::

        此类及其派生类仅适用于 ``pyvqnet.backends.set_backend("torch")``\ ，不要与默认 ``pyvqnet.nn`` 下的 ``Module`` 混用。

        此类的 ``_buffers`` 中的数据为 ``torch.Tensor`` 类型。

        此类的 ``_parmeters`` 中的数据为 ``torch.nn.Parameter`` 类型。


QMachine
""""""""""""""""""

.. py:class:: pyvqnet.qnn.vqc.torch.QMachine(num_wires, dtype=pyvqnet.kcomplex64,grad_mode="",save_ir=False)

    变分量子计算的模拟器类，包含状态向量，其 states 属性为量子电路。

    此类继承自 ``pyvqnet.nn.torch.TorchModule`` 和 ``pyvqnet.qnn.QMachine``\ 。

    此类可作为子模块添加到 torch 模型中。

    .. note::

        在每次运行完整量子电路之前，必须使用 `pyvqnet.qnn.vqc.QMachine.reset_states(batchsize)` 重新初始化模拟器中的初始状态，并将其广播到
        (batchsize,*) 维度，以适应批数据训练。

    :param num_wires: 量子比特数量。
    :param dtype: 计算数据的数据类型。默认值为 pyvqnet.kcomplex64，对应的参数精度为 pyvqnet.kfloat32。
    :param grad_mode: 梯度计算模式，可以为 "adjoint"，默认值：""，使用自动微分。
    :param save_ir: 设置为 True 时，将操作保存到 originIR，默认值：False。

    :return: 输出一个 QMachine 对象。

    示例::

        from pyvqnet.qnn.vqc.torch import QMachine
        import pyvqnet
        pyvqnet.backends.set_backend("torch")
        qm = QMachine(4)
        print(qm.states)


   .. py:method:: reset_states(batchsize)

        重新初始化模拟器中的初始状态，并将其广播到
        (batchsize,*) 维度，以适应批数据训练。

        :param batchsize: 批处理维度。

变分量子逻辑门模块
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

以下 ``pyvqnet.qnn.vqc`` 中的函数接口直接支持 ``torch`` 后端的 ``QTensor`` 进行计算。

.. csv-table:: 支持的 pyvqnet.qnn.vqc 接口列表
    :file: ./images/same_apis_from_vqc.csv

以下量子电路模块继承自 ``pyvqnet.qnn.vqc.torch.QModule``\ ，其中计算使用 ``torch.Tensor`` 进行。

.. note::

    此类及其派生类仅适用于 ``pyvqnet.backends.set_backend("torch")``\ ，不要与默认 ``pyvqnet.nn`` 下的 ``Module`` 混用。

    如果这些类有非参数成员变量 ``_buffers``\ ，其中的数据为 ``torch.Tensor`` 类型。
    如果这些类有参数成员变量 ``_parmeters``\ ，其中的数据为 ``torch.nn.Parameter`` 类型。

I
""""""""""""""""""

.. py:class:: pyvqnet.qnn.vqc.torch.I(has_params: bool = False,trainable: bool = False,init_params=None,wires=None,dtype=pyvqnet.kcomplex64,use_dagger=False)

    定义一个 I 量子门。

    此类继承自 ``pyvqnet.qnn.vqc.torch.QModule`` 和 ``torch.nn.Module``\ 。
    此类可作为子模块添加到 torch 模型中。

    :param has_params: 是否有参数，如 RX、RY 等门需要设置为 True，无参数的门需要设置为 False，默认为 False。
    :param trainable: 是否有参数需要训练。如果该层使用外部输入数据构建逻辑门矩阵，则设置为 False。如果需要从该层初始化待训练参数，则为 True，默认为 False。
    :param init_params: 用于编码经典数据 QTensor 的初始化参数，默认为 None。
    :param wires: 线路作用的比特索引，默认为 None。
    :param dtype: 逻辑门内部矩阵的数据精度，可设置为 pyvqnet.kcomplex64 或 pyvqnet.kcomplex128，分别对应 float 输入或 double 输入。
    :param use_dagger: 是否使用门的转置共轭版本，默认为 False。
    :return: 一个 ``pyvqnet.qnn.vqc.torch.QModule`` 实例

    示例::

        from pyvqnet.qnn.vqc.torch import I,QMachine
        import pyvqnet
        pyvqnet.backends.set_backend("torch")
        device = QMachine(4)
        layer = I(wires=0)
        batchsize = 1
        device.reset_states(1)
        layer(q_machine = device)
        print(device.states)


Hadamard
""""""""""""""""""

.. py:class:: pyvqnet.qnn.vqc.torch.Hadamard(has_params: bool = False,trainable: bool = False,init_params=None,wires=None,dtype=pyvqnet.kcomplex64,use_dagger=False)

    定义一个 Hadamard 量子门。

    此类继承自 ``pyvqnet.qnn.vqc.torch.QModule`` 和 ``torch.nn.Module``\ 。
    此类可作为子模块添加到 torch 模型中。

    :param has_params: 是否有参数，如 RX、RY 等门需要设置为 True，无参数的门需要设置为 False，默认为 False。
    :param trainable: 是否有参数需要训练。如果该层使用外部输入数据构建逻辑门矩阵，则设置为 False。如果需要从该层初始化待训练参数，则为 True，默认为 False。
    :param init_params: 用于编码经典数据 QTensor 的初始化参数，默认为 None。
    :param wires: 线路作用的比特索引，默认为 None。
    :param dtype: 逻辑门内部矩阵的数据精度，可设置为 pyvqnet.kcomplex64 或 pyvqnet.kcomplex128，分别对应 float 输入或 double 输入。
    :param use_dagger: 是否使用门的转置共轭版本，默认为 False。
    :return: 一个 ``pyvqnet.qnn.vqc.torch.QModule`` 实例

    示例::

        from pyvqnet.qnn.vqc.torch import Hadamard,QMachine
        import pyvqnet
        pyvqnet.backends.set_backend("torch")
        device = QMachine(4)
        layer = Hadamard(wires=0)
        batchsize = 1
        device.reset_states(1)
        layer(q_machine = device)
        print(device.states)


T
""""""""""""""""""

.. py:class:: pyvqnet.qnn.vqc.torch.T(has_params: bool = False,trainable: bool = False,init_params=None,wires=None,dtype=pyvqnet.kcomplex64,use_dagger=False)

    定义一个 T 量子门。

    此类继承自 ``pyvqnet.qnn.vqc.torch.QModule`` 和 ``torch.nn.Module``\ 。
    此类可作为子模块添加到 torch 模型中。

    :param has_params: 是否有参数，如 RX、RY 等门需要设置为 True，无参数的门需要设置为 False，默认为 False。
    :param trainable: 是否有参数需要训练。如果该层使用外部输入数据构建逻辑门矩阵，则设置为 False。如果需要从该层初始化待训练参数，则为 True，默认为 False。
    :param init_params: 用于编码经典数据 QTensor 的初始化参数，默认为 None。
    :param wires: 线路作用的比特索引，默认为 None。
    :param dtype: 逻辑门内部矩阵的数据精度，可设置为 pyvqnet.kcomplex64 或 pyvqnet.kcomplex128，分别对应 float 输入或 double 输入。
    :param use_dagger: 是否使用门的转置共轭版本，默认为 False。
    :return: 一个 ``pyvqnet.qnn.vqc.torch.QModule`` 实例

    示例::

        from pyvqnet.qnn.vqc.torch import T,QMachine
        import pyvqnet
        pyvqnet.backends.set_backend("torch")
        device = QMachine(4)
        layer = T(wires=0)
        batchsize = 1
        device.reset_states(1)
        layer(q_machine = device)
        print(device.states)



S
""""""""""""""""""

.. py:class:: pyvqnet.qnn.vqc.torch.S(has_params: bool = False,trainable: bool = False,init_params=None,wires=None,dtype=pyvqnet.kcomplex64,use_dagger=False)

    定义一个 S 量子门。

    此类继承自 ``pyvqnet.qnn.vqc.torch.QModule`` 和 ``torch.nn.Module``\ 。
    此类可作为子模块添加到 torch 模型中。

    :param has_params: 是否有参数，如 RX、RY 等门需要设置为 True，无参数的门需要设置为 False，默认为 False。
    :param trainable: 是否有参数需要训练。如果该层使用外部输入数据构建逻辑门矩阵，则设置为 False。如果需要从该层初始化待训练参数，则为 True，默认为 False。
    :param init_params: 用于编码经典数据 QTensor 的初始化参数，默认为 None。
    :param wires: 线路作用的比特索引，默认为 None。
    :param dtype: 逻辑门内部矩阵的数据精度，可设置为 pyvqnet.kcomplex64 或 pyvqnet.kcomplex128，分别对应 float 输入或 double 输入。
    :param use_dagger: 是否使用门的转置共轭版本，默认为 False。
    :return: 一个 ``pyvqnet.qnn.vqc.torch.QModule`` 实例

    示例::

        from pyvqnet.qnn.vqc.torch import S,QMachine
        import pyvqnet
        pyvqnet.backends.set_backend("torch")
        device = QMachine(4)
        layer = S(wires=0)
        batchsize = 1
        device.reset_states(1)
        layer(q_machine = device)
        print(device.states)


PauliX
""""""""""""""""""

.. py:class:: pyvqnet.qnn.vqc.torch.PauliX(has_params: bool = False,trainable: bool = False,init_params=None,wires=None,dtype=pyvqnet.kcomplex64,use_dagger=False)

    定义一个 PauliX 量子门。

    此类继承自 ``pyvqnet.qnn.vqc.torch.QModule`` 和 ``torch.nn.Module``\ 。
    此类可作为子模块添加到 torch 模型中。


    :param has_params: 是否有参数，如 RX、RY 等门需要设置为 True，无参数的门需要设置为 False，默认为 False。
    :param trainable: 是否有参数需要训练。如果该层使用外部输入数据构建逻辑门矩阵，则设置为 False。如果需要从该层初始化待训练参数，则为 True，默认为 False。
    :param init_params: 用于编码经典数据 QTensor 的初始化参数，默认为 None。
    :param wires: 线路作用的比特索引，默认为 None。
    :param dtype: 逻辑门内部矩阵的数据精度，可设置为 pyvqnet.kcomplex64 或 pyvqnet.kcomplex128，分别对应 float 输入或 double 输入。
    :param use_dagger: 是否使用门的转置共轭版本，默认为 False。
    :return: 一个 ``pyvqnet.qnn.vqc.torch.QModule`` 实例

    示例::

        from pyvqnet.qnn.vqc.torch import PauliX,QMachine
        import pyvqnet
        pyvqnet.backends.set_backend("torch")
        device = QMachine(4)
        layer = PauliX(wires=0)
        batchsize = 1
        device.reset_states(1)
        layer(q_machine = device)
        print(device.states)


PauliY
""""""""""""""""""

.. py:class:: pyvqnet.qnn.vqc.torch.PauliY(has_params: bool = False,trainable: bool = False,init_params=None,wires=None,dtype=pyvqnet.kcomplex64,use_dagger=False)

    定义一个 PauliY 量子门。

    此类继承自 ``pyvqnet.qnn.vqc.torch.QModule`` 和 ``torch.nn.Module``\ 。
    此类可作为子模块添加到 torch 模型中。


    :param has_params: 是否有参数，如 RX、RY 等门需要设置为 True，无参数的门需要设置为 False，默认为 False。
    :param trainable: 是否有参数需要训练。如果该层使用外部输入数据构建逻辑门矩阵，则设置为 False。如果需要从该层初始化待训练参数，则为 True，默认为 False。
    :param init_params: 用于编码经典数据 QTensor 的初始化参数，默认为 None。
    :param wires: 线路作用的比特索引，默认为 None。
    :param dtype: 逻辑门内部矩阵的数据精度，可设置为 pyvqnet.kcomplex64 或 pyvqnet.kcomplex128，分别对应 float 输入或 double 输入。
    :param use_dagger: 是否使用门的转置共轭版本，默认为 False。
    :return: 一个 ``pyvqnet.qnn.vqc.torch.QModule`` 实例

    示例::

        from pyvqnet.qnn.vqc.torch import PauliY,QMachine
        import pyvqnet
        pyvqnet.backends.set_backend("torch")
        device = QMachine(4)
        layer = PauliY(wires=0)
        batchsize = 1
        device.reset_states(1)
        layer(q_machine = device)
        print(device.states)



PauliZ
""""""""""""""""""

.. py:class:: pyvqnet.qnn.vqc.torch.PauliZ(has_params: bool = False,trainable: bool = False,init_params=None,wires=None,dtype=pyvqnet.kcomplex64,use_dagger=False)

    定义一个 PauliZ 量子门。

    此类继承自 ``pyvqnet.qnn.vqc.torch.QModule`` 和 ``torch.nn.Module``\ 。
    此类可作为子模块添加到 torch 模型中。


    :param has_params: 是否有参数，如 RX、RY 等门需要设置为 True，无参数的门需要设置为 False，默认为 False。
    :param trainable: 是否有参数需要训练。如果该层使用外部输入数据构建逻辑门矩阵，则设置为 False。如果需要从该层初始化待训练参数，则为 True，默认为 False。
    :param init_params: 用于编码经典数据 QTensor 的初始化参数，默认为 None。
    :param wires: 线路作用的比特索引，默认为 None。
    :param dtype: 逻辑门内部矩阵的数据精度，可设置为 pyvqnet.kcomplex64 或 pyvqnet.kcomplex128，分别对应 float 输入或 double 输入。
    :param use_dagger: 是否使用门的转置共轭版本，默认为 False。
    :return: 一个 ``pyvqnet.qnn.vqc.torch.QModule`` 实例

    示例::

        from pyvqnet.qnn.vqc.torch import PauliZ,QMachine
        import pyvqnet
        pyvqnet.backends.set_backend("torch")
        device = QMachine(4)
        layer = PauliZ(wires=0)
        batchsize = 1
        device.reset_states(1)
        layer(q_machine = device)
        print(device.states)



X1
""""""""""""""""""

.. py:class:: pyvqnet.qnn.vqc.torch.X1(has_params: bool = False,trainable: bool = False,init_params=None,wires=None,dtype=pyvqnet.kcomplex64,use_dagger=False)

    定义一个 X1 量子门。

    此类继承自 ``pyvqnet.qnn.vqc.torch.QModule`` 和 ``torch.nn.Module``\ 。
    此类可作为子模块添加到 torch 模型中。

    :param has_params: 是否有参数，如 RX、RY 等门需要设置为 True，无参数的门需要设置为 False，默认为 False。
    :param trainable: 是否有参数需要训练。如果该层使用外部输入数据构建逻辑门矩阵，则设置为 False。如果需要从该层初始化待训练参数，则为 True，默认为 False。
    :param init_params: 用于编码经典数据 QTensor 的初始化参数，默认为 None。
    :param wires: 线路作用的比特索引，默认为 None。
    :param dtype: 逻辑门内部矩阵的数据精度，可设置为 pyvqnet.kcomplex64 或 pyvqnet.kcomplex128，分别对应 float 输入或 double 输入。
    :param use_dagger: 是否使用门的转置共轭版本，默认为 False。
    :return: 一个 ``pyvqnet.qnn.vqc.torch.QModule`` 实例

    示例::

        from pyvqnet.qnn.vqc.torch import X1,QMachine
        import pyvqnet
        pyvqnet.backends.set_backend("torch")
        device = QMachine(4)
        layer = X1(wires=0)
        batchsize = 1
        device.reset_states(1)
        layer(q_machine = device)
        print(device.states)


RX
""""""""""""""""""

.. py:class:: pyvqnet.qnn.vqc.torch.RX(has_params: bool = False,trainable: bool = False,init_params=None,wires=None,dtype=pyvqnet.kcomplex64,use_dagger=False)

    定义一个 RX 量子门。

    此类继承自 ``pyvqnet.qnn.vqc.torch.QModule`` 和 ``torch.nn.Module``\ 。
    此类可作为子模块添加到 torch 模型中。

    :param has_params: 是否有参数，如 RX、RY 等门需要设置为 True，无参数的门需要设置为 False，默认为 False。
    :param trainable: 是否有参数需要训练。如果该层使用外部输入数据构建逻辑门矩阵，则设置为 False。如果需要从该层初始化待训练参数，则为 True，默认为 False。
    :param init_params: 用于编码经典数据 QTensor 的初始化参数，默认为 None。
    :param wires: 线路作用的比特索引，默认为 None。
    :param dtype: 逻辑门内部矩阵的数据精度，可设置为 pyvqnet.kcomplex64 或 pyvqnet.kcomplex128，分别对应 float 输入或 double 输入。
    :param use_dagger: 是否使用门的转置共轭版本，默认为 False。
    :return: 一个 ``pyvqnet.qnn.vqc.torch.QModule`` 实例

    示例::

        from pyvqnet.qnn.vqc.torch import RX,QMachine
        import pyvqnet
        pyvqnet.backends.set_backend("torch")
        device = QMachine(4)
        layer = RX(has_params= True, trainable= True, wires=0)
        batchsize = 2
        device.reset_states(batchsize)
        layer(q_machine = device)
        print(device.states)



RY
""""""""""""""""""

.. py:class:: pyvqnet.qnn.vqc.torch.RY(has_params: bool = False,trainable: bool = False,init_params=None,wires=None,dtype=pyvqnet.kcomplex64,use_dagger=False)

    定义一个 RY 量子门。

    此类继承自 ``pyvqnet.qnn.vqc.torch.QModule`` 和 ``torch.nn.Module``\ 。
    此类可作为子模块添加到 torch 模型中。

    :param has_params: 是否有参数，如 RX、RY 等门需要设置为 True，无参数的门需要设置为 False，默认为 False。
    :param trainable: 是否有参数需要训练。如果该层使用外部输入数据构建逻辑门矩阵，则设置为 False。如果需要从该层初始化待训练参数，则为 True，默认为 False。
    :param init_params: 用于编码经典数据 QTensor 的初始化参数，默认为 None。
    :param wires: 线路作用的比特索引，默认为 None。
    :param dtype: 逻辑门内部矩阵的数据精度，可设置为 pyvqnet.kcomplex64 或 pyvqnet.kcomplex128，分别对应 float 输入或 double 输入。
    :param use_dagger: 是否使用门的转置共轭版本，默认为 False。
    :return: 一个 ``pyvqnet.qnn.vqc.torch.QModule`` 实例

    示例::

        from pyvqnet.qnn.vqc.torch import RY,QMachine
        import pyvqnet
        pyvqnet.backends.set_backend("torch")
        device = QMachine(4)
        layer = RY(has_params= True, trainable= True, wires=0)
        batchsize = 2
        device.reset_states(batchsize)
        layer(q_machine = device)
        print(device.states)


RZ
""""""""""""""""""

.. py:class:: pyvqnet.qnn.vqc.torch.RZ(has_params: bool = False,trainable: bool = False,init_params=None,wires=None,dtype=pyvqnet.kcomplex64,use_dagger=False)

    定义一个 RZ 量子门。

    此类继承自 ``pyvqnet.qnn.vqc.torch.QModule`` 和 ``torch.nn.Module``\ 。
    此类可作为子模块添加到 torch 模型中。

    :param has_params: 是否有参数，如 RX、RY 等门需要设置为 True，无参数的门需要设置为 False，默认为 False。
    :param trainable: 是否有参数需要训练。如果该层使用外部输入数据构建逻辑门矩阵，则设置为 False。如果需要从该层初始化待训练参数，则为 True，默认为 False。
    :param init_params: 用于编码经典数据 QTensor 的初始化参数，默认为 None。
    :param wires: 线路作用的比特索引，默认为 None。
    :param dtype: 逻辑门内部矩阵的数据精度，可设置为 pyvqnet.kcomplex64 或 pyvqnet.kcomplex128，分别对应 float 输入或 double 输入。
    :param use_dagger: 是否使用门的转置共轭版本，默认为 False。
    :return: 一个 ``pyvqnet.qnn.vqc.torch.QModule`` 实例

    示例::

        from pyvqnet.qnn.vqc.torch import RZ,QMachine
        import pyvqnet
        pyvqnet.backends.set_backend("torch")
        device = QMachine(4)
        layer = RZ(has_params= True, trainable= True, wires=0)
        batchsize = 2
        device.reset_states(batchsize)
        layer(q_machine = device)
        print(device.states)


CRX
""""""""""""""""""

.. py:class:: pyvqnet.qnn.vqc.torch.CRX(has_params: bool = False,trainable: bool = False,init_params=None,wires=None,dtype=pyvqnet.kcomplex64,use_dagger=False)

    定义一个 CRX 量子门。

    此类继承自 ``pyvqnet.qnn.vqc.torch.QModule`` 和 ``torch.nn.Module``\ 。
    此类可作为子模块添加到 torch 模型中。

    :param has_params: 是否有参数，如 RX、RY 等门需要设置为 True，无参数的门需要设置为 False，默认为 False。
    :param trainable: 是否有参数需要训练。如果该层使用外部输入数据构建逻辑门矩阵，则设置为 False。如果需要从该层初始化待训练参数，则为 True，默认为 False。
    :param init_params: 用于编码经典数据 QTensor 的初始化参数，默认为 None。
    :param wires: 线路作用的比特索引，默认为 None。
    :param dtype: 逻辑门内部矩阵的数据精度，可设置为 pyvqnet.kcomplex64 或 pyvqnet.kcomplex128，分别对应 float 输入或 double 输入。
    :param use_dagger: 是否使用门的转置共轭版本，默认为 False。
    :return: 一个 ``pyvqnet.qnn.vqc.torch.QModule`` 实例

    示例::

        from pyvqnet.qnn.vqc.torch import CRX,QMachine
        import pyvqnet
        pyvqnet.backends.set_backend("torch")
        device = QMachine(4)
        layer = CRX(has_params= True, trainable= True, wires=[0,2])
        batchsize = 2
        device.reset_states(batchsize)
        layer(q_machine = device)
        print(device.states)


CRY
""""""""""""""""""


.. py:class:: pyvqnet.qnn.vqc.torch.CRY(has_params: bool = False,trainable: bool = False,init_params=None,wires=None,dtype=pyvqnet.kcomplex64,use_dagger=False)

    定义一个 CRY 量子门。

    此类继承自 ``pyvqnet.qnn.vqc.torch.QModule`` 和 ``torch.nn.Module``\ 。
    此类可作为子模块添加到 torch 模型中。

    :param has_params: 是否有参数，如 RX、RY 等门需要设置为 True，无参数的门需要设置为 False，默认为 False。
    :param trainable: 是否有参数需要训练。如果该层使用外部输入数据构建逻辑门矩阵，则设置为 False。如果需要从该层初始化待训练参数，则为 True，默认为 False。
    :param init_params: 用于编码经典数据 QTensor 的初始化参数，默认为 None。
    :param wires: 线路作用的比特索引，默认为 None。
    :param dtype: 逻辑门内部矩阵的数据精度，可设置为 pyvqnet.kcomplex64 或 pyvqnet.kcomplex128，分别对应 float 输入或 double 输入。
    :param use_dagger: 是否使用门的转置共轭版本，默认为 False。
    :return: 一个 ``pyvqnet.qnn.vqc.torch.QModule`` 实例

    示例::

        from pyvqnet.qnn.vqc.torch import CRY,QMachine
        import pyvqnet
        pyvqnet.backends.set_backend("torch")
        device = QMachine(4)
        layer = CRY(has_params= True, trainable= True, wires=[0,2])
        batchsize = 2
        device.reset_states(batchsize)
        layer(q_machine = device)
        print(device.states)


CRZ
""""""""""""""""""


.. py:class:: pyvqnet.qnn.vqc.torch.CRZ(has_params: bool = False,trainable: bool = False,init_params=None,wires=None,dtype=pyvqnet.kcomplex64,use_dagger=False)

    定义一个 CRZ 量子门。

    此类继承自 ``pyvqnet.qnn.vqc.torch.QModule`` 和 ``torch.nn.Module``\ 。
    此类可作为子模块添加到 torch 模型中。

    :param has_params: 是否有参数，如 RX、RY 等门需要设置为 True，无参数的门需要设置为 False，默认为 False。
    :param trainable: 是否有参数需要训练。如果该层使用外部输入数据构建逻辑门矩阵，则设置为 False。如果需要从该层初始化待训练参数，则为 True，默认为 False。
    :param init_params: 用于编码经典数据 QTensor 的初始化参数，默认为 None。
    :param wires: 线路作用的比特索引，默认为 None。
    :param dtype: 逻辑门内部矩阵的数据精度，可设置为 pyvqnet.kcomplex64 或 pyvqnet.kcomplex128，分别对应 float 输入或 double 输入。
    :param use_dagger: 是否使用门的转置共轭版本，默认为 False。
    :return: 一个 ``pyvqnet.qnn.vqc.torch.QModule`` 实例

    示例::

        from pyvqnet.qnn.vqc.torch import CRZ,QMachine
        import pyvqnet
        pyvqnet.backends.set_backend("torch")
        device = QMachine(4)
        layer = CRZ(has_params= True, trainable= True, wires=[0,2])
        batchsize = 2
        device.reset_states(batchsize)
        layer(q_machine = device)
        print(device.states)



U1
""""""""""""""""""

.. py:class:: pyvqnet.qnn.vqc.torch.U1(has_params: bool = False,trainable: bool = False,init_params=None,wires=None,dtype=pyvqnet.kcomplex64,use_dagger=False)

    定义一个 U1 量子门。

    此类继承自 ``pyvqnet.qnn.vqc.torch.QModule`` 和 ``torch.nn.Module``\ 。
    此类可作为子模块添加到 torch 模型中。

    :param has_params: 是否有参数，如 RX、RY 等门需要设置为 True，无参数的门需要设置为 False，默认为 False。
    :param trainable: 是否有参数需要训练。如果该层使用外部输入数据构建逻辑门矩阵，则设置为 False。如果需要从该层初始化待训练参数，则为 True，默认为 False。
    :param init_params: 用于编码经典数据 QTensor 的初始化参数，默认为 None。
    :param wires: 线路作用的比特索引，默认为 None。
    :param dtype: 逻辑门内部矩阵的数据精度，可设置为 pyvqnet.kcomplex64 或 pyvqnet.kcomplex128，分别对应 float 输入或 double 输入。
    :param use_dagger: 是否使用门的转置共轭版本，默认为 False。
    :return: 一个 ``pyvqnet.qnn.vqc.torch.QModule`` 实例

    示例::

        from pyvqnet.qnn.vqc.torch import U1,QMachine
        import pyvqnet
        pyvqnet.backends.set_backend("torch")
        device = QMachine(4)
        layer = U1(has_params= True, trainable= True, wires=0)
        batchsize = 2
        device.reset_states(batchsize)
        layer(q_machine = device)
        print(device.states)

U2
""""""""""""""""""


.. py:class:: pyvqnet.qnn.vqc.torch.U2(has_params: bool = False,trainable: bool = False,init_params=None,wires=None,dtype=pyvqnet.kcomplex64,use_dagger=False)

    定义一个 U2 量子门。

    此类继承自 ``pyvqnet.qnn.vqc.torch.QModule`` 和 ``torch.nn.Module``\ 。
    此类可作为子模块添加到 torch 模型中。

    :param has_params: 是否有参数，如 RX、RY 等门需要设置为 True，无参数的门需要设置为 False，默认为 False。
    :param trainable: 是否有参数需要训练。如果该层使用外部输入数据构建逻辑门矩阵，则设置为 False。如果需要从该层初始化待训练参数，则为 True，默认为 False。
    :param init_params: 用于编码经典数据 QTensor 的初始化参数，默认为 None。
    :param wires: 线路作用的比特索引，默认为 None。
    :param dtype: 逻辑门内部矩阵的数据精度，可设置为 pyvqnet.kcomplex64 或 pyvqnet.kcomplex128，分别对应 float 输入或 double 输入。
    :param use_dagger: 是否使用门的转置共轭版本，默认为 False。
    :return: 一个 ``pyvqnet.qnn.vqc.torch.QModule`` 实例

    示例::

        from pyvqnet.qnn.vqc.torch import U2,QMachine
        import pyvqnet
        pyvqnet.backends.set_backend("torch")
        device = QMachine(4)
        layer = U2(has_params= True, trainable= True, wires=1)
        batchsize = 2
        device.reset_states(batchsize)
        layer(q_machine = device)
        print(device.states)


U3
""""""""""""""""""


.. py:class:: pyvqnet.qnn.vqc.torch.U3(has_params: bool = False,trainable: bool = False,init_params=None,wires=None,dtype=pyvqnet.kcomplex64,use_dagger=False)

    定义一个 U3 量子门。

    此类继承自 ``pyvqnet.qnn.vqc.torch.QModule`` 和 ``torch.nn.Module``\ 。
    此类可作为子模块添加到 torch 模型中。

    :param has_params: 是否有参数，如 RX、RY 等门需要设置为 True，无参数的门需要设置为 False，默认为 False。
    :param trainable: 是否有参数需要训练。如果该层使用外部输入数据构建逻辑门矩阵，则设置为 False。如果需要从该层初始化待训练参数，则为 True，默认为 False。
    :param init_params: 用于编码经典数据 QTensor 的初始化参数，默认为 None。
    :param wires: 线路作用的比特索引，默认为 None。
    :param dtype: 逻辑门内部矩阵的数据精度，可设置为 pyvqnet.kcomplex64 或 pyvqnet.kcomplex128，分别对应 float 输入或 double 输入。
    :param use_dagger: 是否使用门的转置共轭版本，默认为 False。
    :return: 一个 ``pyvqnet.qnn.vqc.torch.QModule`` 实例

    示例::

        from pyvqnet.qnn.vqc.torch import U3,QMachine
        import pyvqnet
        pyvqnet.backends.set_backend("torch")
        device = QMachine(4)
        layer = U3(has_params= True, trainable= True, wires=1)
        batchsize = 2
        device.reset_states(batchsize)
        layer(q_machine = device)
        print(device.states)



CNOT
""""""""""""""""""

.. py:class:: pyvqnet.qnn.vqc.torch.CNOT(has_params: bool = False,trainable: bool = False,init_params=None,wires=None,dtype=pyvqnet.kcomplex64,use_dagger=False)

    定义一个 CNOT 量子门，别名为 `CX`\ 。

    此类继承自 ``pyvqnet.qnn.vqc.torch.QModule`` 和 ``torch.nn.Module``\ 。
    此类可作为子模块添加到 torch 模型中。

    :param has_params: 是否有参数，如 RX、RY 等门需要设置为 True，无参数的门需要设置为 False，默认为 False。
    :param trainable: 是否有参数需要训练。如果该层使用外部输入数据构建逻辑门矩阵，则设置为 False。如果需要从该层初始化待训练参数，则为 True，默认为 False。
    :param init_params: 用于编码经典数据 QTensor 的初始化参数，默认为 None。
    :param wires: 线路作用的比特索引，默认为 None。
    :param dtype: 逻辑门内部矩阵的数据精度，可设置为 pyvqnet.kcomplex64 或 pyvqnet.kcomplex128，分别对应 float 输入或 double 输入。
    :param use_dagger: 是否使用门的转置共轭版本，默认为 False。
    :return: 一个 ``pyvqnet.qnn.vqc.torch.QModule`` 实例

    示例::

        from pyvqnet.qnn.vqc.torch import CNOT,QMachine
        import pyvqnet
        pyvqnet.backends.set_backend("torch")
        device = QMachine(4)
        layer = CNOT(wires=[0,1])
        batchsize = 2
        device.reset_states(batchsize)
        layer(q_machine = device)
        print(device.states)

CY
""""""""""""""""""

.. py:class:: pyvqnet.qnn.vqc.torch.CY(has_params: bool = False,trainable: bool = False,init_params=None,wires=None,dtype=pyvqnet.kcomplex64,use_dagger=False)

    定义一个 CY 量子门。

    此类继承自 ``pyvqnet.qnn.vqc.torch.QModule`` 和 ``torch.nn.Module``\ 。
    此类可作为子模块添加到 torch 模型中。

    :param has_params: 是否有参数，如 RX、RY 等门需要设置为 True，无参数的门需要设置为 False，默认为 False。
    :param trainable: 是否有参数需要训练。如果该层使用外部输入数据构建逻辑门矩阵，则设置为 False。如果需要从该层初始化待训练参数，则为 True，默认为 False。
    :param init_params: 用于编码经典数据 QTensor 的初始化参数，默认为 None。
    :param wires: 线路作用的比特索引，默认为 None。
    :param dtype: 逻辑门内部矩阵的数据精度，可设置为 pyvqnet.kcomplex64 或 pyvqnet.kcomplex128，分别对应 float 输入或 double 输入。
    :param use_dagger: 是否使用门的转置共轭版本，默认为 False。
    :return: 一个 ``pyvqnet.qnn.vqc.torch.QModule`` 实例

    示例::

        from pyvqnet.qnn.vqc.torch import CY,QMachine
        import pyvqnet
        pyvqnet.backends.set_backend("torch")
        device = QMachine(4)
        layer = CY(wires=[0,1])
        batchsize = 2
        device.reset_states(batchsize)
        layer(q_machine = device)
        print(device.states)


CZ
""""""""""""""""""

.. py:class:: pyvqnet.qnn.vqc.torch.CZ(has_params: bool = False,trainable: bool = False,init_params=None,wires=None,dtype=pyvqnet.kcomplex64,use_dagger=False)

    定义一个 CZ 量子门。

    此类继承自 ``pyvqnet.qnn.vqc.torch.QModule`` 和 ``torch.nn.Module``\ 。
    此类可作为子模块添加到 torch 模型中。

    :param has_params: 是否有参数，如 RX、RY 等门需要设置为 True，无参数的门需要设置为 False，默认为 False。
    :param trainable: 是否有参数需要训练。如果该层使用外部输入数据构建逻辑门矩阵，则设置为 False。如果需要从该层初始化待训练参数，则为 True，默认为 False。
    :param init_params: 用于编码经典数据 QTensor 的初始化参数，默认为 None。
    :param wires: 线路作用的比特索引，默认为 None。
    :param dtype: 逻辑门内部矩阵的数据精度，可设置为 pyvqnet.kcomplex64 或 pyvqnet.kcomplex128，分别对应 float 输入或 double 输入。
    :param use_dagger: 是否使用门的转置共轭版本，默认为 False。
    :return: 一个 ``pyvqnet.qnn.vqc.torch.QModule`` 实例

    示例::

        from pyvqnet.qnn.vqc.torch import CZ,QMachine
        import pyvqnet
        pyvqnet.backends.set_backend("torch")
        device = QMachine(4)
        layer = CZ(wires=[0,1])
        batchsize = 2
        device.reset_states(batchsize)
        layer(q_machine = device)
        print(device.states)




CR
""""""""""""""""""

.. py:class:: pyvqnet.qnn.vqc.torch.CR(has_params: bool = False,trainable: bool = False,init_params=None,wires=None,dtype=pyvqnet.kcomplex64,use_dagger=False)

    定义一个 CR 量子门。

    此类继承自 ``pyvqnet.qnn.vqc.torch.QModule`` 和 ``torch.nn.Module``\ 。
    此类可作为子模块添加到 torch 模型中。

    :param has_params: 是否有参数，如 RX、RY 等门需要设置为 True，无参数的门需要设置为 False，默认为 False。
    :param trainable: 是否有参数需要训练。如果该层使用外部输入数据构建逻辑门矩阵，则设置为 False。如果需要从该层初始化待训练参数，则为 True，默认为 False。
    :param init_params: 用于编码经典数据 QTensor 的初始化参数，默认为 None。
    :param wires: 线路作用的比特索引，默认为 None。
    :param dtype: 逻辑门内部矩阵的数据精度，可设置为 pyvqnet.kcomplex64 或 pyvqnet.kcomplex128，分别对应 float 输入或 double 输入。
    :param use_dagger: 是否使用门的转置共轭版本，默认为 False。
    :return: 一个 ``pyvqnet.qnn.vqc.torch.QModule`` 实例

    示例::

        from pyvqnet.qnn.vqc.torch import CR,QMachine
        import pyvqnet
        pyvqnet.backends.set_backend("torch")
        device = QMachine(4)
        device = QMachine(4)
        layer = CR(has_params= True, trainable= True, wires=[0,2])
        batchsize = 2
        device.reset_states(batchsize)
        layer(q_machine = device)
        print(device.states)



SWAP
""""""""""""""""""


.. py:class:: pyvqnet.qnn.vqc.torch.SWAP(has_params: bool = False,trainable: bool = False,init_params=None,wires=None,dtype=pyvqnet.kcomplex64,use_dagger=False)

    定义一个 SWAP 量子门。

    此类继承自 ``pyvqnet.qnn.vqc.torch.QModule`` 和 ``torch.nn.Module``\ 。
    此类可作为子模块添加到 torch 模型中。

    :param has_params: 是否有参数，如 RX、RY 等门需要设置为 True，无参数的门需要设置为 False，默认为 False。
    :param trainable: 是否有参数需要训练。如果该层使用外部输入数据构建逻辑门矩阵，则设置为 False。如果需要从该层初始化待训练参数，则为 True，默认为 False。
    :param init_params: 用于编码经典数据 QTensor 的初始化参数，默认为 None。
    :param wires: 线路作用的比特索引，默认为 None。
    :param dtype: 逻辑门内部矩阵的数据精度，可设置为 pyvqnet.kcomplex64 或 pyvqnet.kcomplex128，分别对应 float 输入或 double 输入。
    :param use_dagger: 是否使用门的转置共轭版本，默认为 False。
    :return: 一个 ``pyvqnet.qnn.vqc.torch.QModule`` 实例

    示例::

        from pyvqnet.qnn.vqc.torch import SWAP,QMachine
        import pyvqnet
        pyvqnet.backends.set_backend("torch")
        device = QMachine(4)
        layer = SWAP(wires=[0,1])
        batchsize = 2
        device.reset_states(batchsize)
        layer(q_machine = device)
        print(device.states)


CSWAP
""""""""""""""""""

.. py:class:: pyvqnet.qnn.vqc.torch.CSWAP(has_params: bool = False,trainable: bool = False,init_params=None,wires=None,dtype=pyvqnet.kcomplex64,use_dagger=False)

    定义一个 SWAP 量子门。

    .. math:: CSWAP = \begin{bmatrix}
            1 & 0 & 0 & 0 & 0 & 0 & 0 & 0 \\
            0 & 1 & 0 & 0 & 0 & 0 & 0 & 0 \\
            0 & 0 & 1 & 0 & 0 & 0 & 0 & 0 \\
            0 & 0 & 0 & 1 & 0 & 0 & 0 & 0 \\
            0 & 0 & 0 & 0 & 1 & 0 & 0 & 0 \\
            0 & 0 & 0 & 0 & 0 & 0 & 1 & 0 \\
            0 & 0 & 0 & 0 & 0 & 1 & 0 & 0 \\
            0 & 0 & 0 & 0 & 0 & 0 & 0 & 1
        \end{bmatrix}.

    此类继承自 ``pyvqnet.qnn.vqc.torch.QModule`` 和 ``torch.nn.Module``\ 。
    此类可作为子模块添加到 torch 模型中。

    :param has_params: 是否有参数，如 RX、RY 等门需要设置为 True，无参数的门需要设置为 False，默认为 False。
    :param trainable: 是否有参数需要训练。如果该层使用外部输入数据构建逻辑门矩阵，则设置为 False。如果需要从该层初始化待训练参数，则为 True，默认为 False。
    :param init_params: 用于编码经典数据 QTensor 的初始化参数，默认为 None。
    :param wires: 线路作用的比特索引，默认为 None。
    :param dtype: 逻辑门内部矩阵的数据精度，可设置为 pyvqnet.kcomplex64 或 pyvqnet.kcomplex128，分别对应 float 输入或 double 输入。
    :param use_dagger: 是否使用门的转置共轭版本，默认为 False。
    :return: 一个 ``pyvqnet.qnn.vqc.torch.QModule`` 实例

    示例::

        from pyvqnet.qnn.vqc.torch import CSWAP,QMachine
        import pyvqnet
        pyvqnet.backends.set_backend("torch")
        device = QMachine(4)
        layer = CSWAP(wires=[0,1,2])
        batchsize = 2
        device.reset_states(batchsize)
        layer(q_machine = device)
        print(device.states)

RXX
""""""""""""""""""


.. py:class:: pyvqnet.qnn.vqc.torch.RXX(has_params: bool = False,trainable: bool = False,init_params=None,wires=None,dtype=pyvqnet.kcomplex64,use_dagger=False)

    定义一个 RXX 量子门。

    此类继承自 ``pyvqnet.qnn.vqc.torch.QModule`` 和 ``torch.nn.Module``\ 。
    此类可作为子模块添加到 torch 模型中。

    :param has_params: 是否有参数，如 RX、RY 等门需要设置为 True，无参数的门需要设置为 False，默认为 False。
    :param trainable: 是否有参数需要训练。如果该层使用外部输入数据构建逻辑门矩阵，则设置为 False。如果需要从该层初始化待训练参数，则为 True，默认为 False。
    :param init_params: 用于编码经典数据 QTensor 的初始化参数，默认为 None。
    :param wires: 线路作用的比特索引，默认为 None。
    :param dtype: 逻辑门内部矩阵的数据精度，可设置为 pyvqnet.kcomplex64 或 pyvqnet.kcomplex128，分别对应 float 输入或 double 输入。
    :param use_dagger: 是否使用门的转置共轭版本，默认为 False。
    :return: 一个 ``pyvqnet.qnn.vqc.torch.QModule`` 实例

    示例::

        from pyvqnet.qnn.vqc.torch import RXX,QMachine
        import pyvqnet
        pyvqnet.backends.set_backend("torch")
        device = QMachine(4)
        layer = RXX(has_params= True, trainable= True, wires=[0,2])
        batchsize = 2
        device.reset_states(batchsize)
        layer(q_machine = device)
        print(device.states)

RYY
""""""""""""""""""

.. py:class:: pyvqnet.qnn.vqc.torch.RYY(has_params: bool = False,trainable: bool = False,init_params=None,wires=None,dtype=pyvqnet.kcomplex64,use_dagger=False)

    定义一个 RYY 量子门。

    此类继承自 ``pyvqnet.qnn.vqc.torch.QModule`` 和 ``torch.nn.Module``\ 。
    此类可作为子模块添加到 torch 模型中。

    :param has_params: 是否有参数，如 RX、RY 等门需要设置为 True，无参数的门需要设置为 False，默认为 False。
    :param trainable: 是否有参数需要训练。如果该层使用外部输入数据构建逻辑门矩阵，则设置为 False。如果需要从该层初始化待训练参数，则为 True，默认为 False。
    :param init_params: 用于编码经典数据 QTensor 的初始化参数，默认为 None。
    :param wires: 线路作用的比特索引，默认为 None。
    :param dtype: 逻辑门内部矩阵的数据精度，可设置为 pyvqnet.kcomplex64 或 pyvqnet.kcomplex128，分别对应 float 输入或 double 输入。
    :param use_dagger: 是否使用门的转置共轭版本，默认为 False。
    :return: 一个 ``pyvqnet.qnn.vqc.torch.QModule`` 实例

    示例::

        from pyvqnet.qnn.vqc.torch import RYY,QMachine
        import pyvqnet
        pyvqnet.backends.set_backend("torch")
        device = QMachine(4)
        layer = RYY(has_params= True, trainable= True, wires=[0,2])
        batchsize = 2
        device.reset_states(batchsize)
        layer(q_machine = device)
        print(device.states)


RZZ
""""""""""""""""""

.. py:class:: pyvqnet.qnn.vqc.torch.RZZ(has_params: bool = False,trainable: bool = False,init_params=None,wires=None,dtype=pyvqnet.kcomplex64,use_dagger=False)

    定义一个 RZZ 量子门。

    此类继承自 ``pyvqnet.qnn.vqc.torch.QModule`` 和 ``torch.nn.Module``\ 。
    此类可作为子模块添加到 torch 模型中。

    :param has_params: 是否有参数，如 RX、RY 等门需要设置为 True，无参数的门需要设置为 False，默认为 False。
    :param trainable: 是否有参数需要训练。如果该层使用外部输入数据构建逻辑门矩阵，则设置为 False。如果需要从该层初始化待训练参数，则为 True，默认为 False。
    :param init_params: 用于编码经典数据 QTensor 的初始化参数，默认为 None。
    :param wires: 线路作用的比特索引，默认为 None。
    :param dtype: 逻辑门内部矩阵的数据精度，可设置为 pyvqnet.kcomplex64 或 pyvqnet.kcomplex128，分别对应 float 输入或 double 输入。
    :param use_dagger: 是否使用门的转置共轭版本，默认为 False。
    :return: 一个 ``pyvqnet.qnn.vqc.torch.QModule`` 实例

    示例::

        from pyvqnet.qnn.vqc.torch import RZZ,QMachine
        import pyvqnet
        pyvqnet.backends.set_backend("torch")
        device = QMachine(4)
        layer = RZZ(has_params= True, trainable= True, wires=[0,2])
        batchsize = 2
        device.reset_states(batchsize)
        layer(q_machine = device)
        print(device.states)



RZX
""""""""""""""""""

.. py:class:: pyvqnet.qnn.vqc.torch.RZX(has_params: bool = False,trainable: bool = False,init_params=None,wires=None,dtype=pyvqnet.kcomplex64,use_dagger=False)

    定义一个 RZX 量子门。

    此类继承自 ``pyvqnet.qnn.vqc.torch.QModule`` 和 ``torch.nn.Module``\ 。
    此类可作为子模块添加到 torch 模型中。

    :param has_params: 是否有参数，如 RX、RY 等门需要设置为 True，无参数的门需要设置为 False，默认为 False。
    :param trainable: 是否有参数需要训练。如果该层使用外部输入数据构建逻辑门矩阵，则设置为 False。如果需要从该层初始化待训练参数，则为 True，默认为 False。
    :param init_params: 用于编码经典数据 QTensor 的初始化参数，默认为 None。
    :param wires: 线路作用的比特索引，默认为 None。
    :param dtype: 逻辑门内部矩阵的数据精度，可设置为 pyvqnet.kcomplex64 或 pyvqnet.kcomplex128，分别对应 float 输入或 double 输入。
    :param use_dagger: 是否使用门的转置共轭版本，默认为 False。
    :return: 一个 ``pyvqnet.qnn.vqc.torch.QModule`` 实例

    示例::

        from pyvqnet.qnn.vqc.torch import RZX,QMachine
        import pyvqnet
        pyvqnet.backends.set_backend("torch")
        device = QMachine(4)
        layer = RZX(has_params= True, trainable= True, wires=[0,2])
        batchsize = 2
        device.reset_states(batchsize)
        layer(q_machine = device)
        print(device.states)

Toffoli
""""""""""""""""""

.. py:class:: pyvqnet.qnn.vqc.torch.Toffoli(has_params: bool = False,trainable: bool = False,init_params=None,wires=None,dtype=pyvqnet.kcomplex64,use_dagger=False)

    定义一个 Toffoli 量子门。

    此类继承自 ``pyvqnet.qnn.vqc.torch.QModule`` 和 ``torch.nn.Module``\ 。
    此类可作为子模块添加到 torch 模型中。

    :param has_params: 是否有参数，如 RX、RY 等门需要设置为 True，无参数的门需要设置为 False，默认为 False。
    :param trainable: 是否有参数需要训练。如果该层使用外部输入数据构建逻辑门矩阵，则设置为 False。如果需要从该层初始化待训练参数，则为 True，默认为 False。
    :param init_params: 用于编码经典数据 QTensor 的初始化参数，默认为 None。
    :param wires: 线路作用的比特索引，默认为 None。
    :param dtype: 逻辑门内部矩阵的数据精度，可设置为 pyvqnet.kcomplex64 或 pyvqnet.kcomplex128，分别对应 float 输入或 double 输入。
    :param use_dagger: 是否使用门的转置共轭版本，默认为 False。
    :return: 一个 ``pyvqnet.qnn.vqc.torch.QModule`` 实例

    示例::

        from pyvqnet.qnn.vqc.torch import Toffoli,QMachine
        import pyvqnet
        pyvqnet.backends.set_backend("torch")
        device = QMachine(4)
        layer = Toffoli(wires=[0,2,1])
        batchsize = 2
        device.reset_states(batchsize)
        layer(q_machine = device)
        print(device.states)

IsingXX
""""""""""""""""""


.. py:class:: pyvqnet.qnn.vqc.torch.IsingXX(has_params: bool = False,trainable: bool = False,init_params=None,wires=None,dtype=pyvqnet.kcomplex64,use_dagger=False)

    定义一个 IsingXX 量子门。

    此类继承自 ``pyvqnet.qnn.vqc.torch.QModule`` 和 ``torch.nn.Module``\ 。
    此类可作为子模块添加到 torch 模型中。

    :param has_params: 是否有参数，如 RX、RY 等门需要设置为 True，无参数的门需要设置为 False，默认为 False。
    :param trainable: 是否有参数需要训练。如果该层使用外部输入数据构建逻辑门矩阵，则设置为 False。如果需要从该层初始化待训练参数，则为 True，默认为 False。
    :param init_params: 用于编码经典数据 QTensor 的初始化参数，默认为 None。
    :param wires: 线路作用的比特索引，默认为 None。
    :param dtype: 逻辑门内部矩阵的数据精度，可设置为 pyvqnet.kcomplex64 或 pyvqnet.kcomplex128，分别对应 float 输入或 double 输入。
    :param use_dagger: 是否使用门的转置共轭版本，默认为 False。
    :return: 一个 ``pyvqnet.qnn.vqc.torch.QModule`` 实例

    示例::

        from pyvqnet.qnn.vqc.torch import IsingXX,QMachine
        import pyvqnet
        pyvqnet.backends.set_backend("torch")
        device = QMachine(4)
        layer = IsingXX(has_params= True, trainable= True, wires=[0,2])
        batchsize = 2
        device.reset_states(batchsize)
        layer(q_machine = device)
        print(device.states)


IsingYY
""""""""""""""""""


.. py:class:: pyvqnet.qnn.vqc.torch.IsingYY(has_params: bool = False,trainable: bool = False,init_params=None,wires=None,dtype=pyvqnet.kcomplex64,use_dagger=False)

    定义一个 IsingYY 量子门。

    此类继承自 ``pyvqnet.qnn.vqc.torch.QModule`` 和 ``torch.nn.Module``\ 。
    此类可作为子模块添加到 torch 模型中。

    :param has_params: 是否有参数，如 RX、RY 等门需要设置为 True，无参数的门需要设置为 False，默认为 False。
    :param trainable: 是否有参数需要训练。如果该层使用外部输入数据构建逻辑门矩阵，则设置为 False。如果需要从该层初始化待训练参数，则为 True，默认为 False。
    :param init_params: 用于编码经典数据 QTensor 的初始化参数，默认为 None。
    :param wires: 线路作用的比特索引，默认为 None。
    :param dtype: 逻辑门内部矩阵的数据精度，可设置为 pyvqnet.kcomplex64 或 pyvqnet.kcomplex128，分别对应 float 输入或 double 输入。
    :param use_dagger: 是否使用门的转置共轭版本，默认为 False。
    :return: 一个 ``pyvqnet.qnn.vqc.torch.QModule`` 实例

    示例::

        from pyvqnet.qnn.vqc.torch import IsingYY,QMachine
        import pyvqnet
        pyvqnet.backends.set_backend("torch")
        device = QMachine(4)
        layer = IsingYY(has_params= True, trainable= True, wires=[0,2])
        batchsize = 2
        device.reset_states(batchsize)
        layer(q_machine = device)
        print(device.states)


IsingZZ
""""""""""""""""""

.. py:class:: pyvqnet.qnn.vqc.torch.IsingZZ(has_params: bool = False,trainable: bool = False,init_params=None,wires=None,dtype=pyvqnet.kcomplex64,use_dagger=False)

    定义一个 IsingZZ 量子门。

    此类继承自 ``pyvqnet.qnn.vqc.torch.QModule`` 和 ``torch.nn.Module``\ 。
    此类可作为子模块添加到 torch 模型中。

    :param has_params: 是否有参数，如 RX、RY 等门需要设置为 True，无参数的门需要设置为 False，默认为 False。
    :param trainable: 是否有参数需要训练。如果该层使用外部输入数据构建逻辑门矩阵，则设置为 False。如果需要从该层初始化待训练参数，则为 True，默认为 False。
    :param init_params: 用于编码经典数据 QTensor 的初始化参数，默认为 None。
    :param wires: 线路作用的比特索引，默认为 None。
    :param dtype: 逻辑门内部矩阵的数据精度，可设置为 pyvqnet.kcomplex64 或 pyvqnet.kcomplex128，分别对应 float 输入或 double 输入。
    :param use_dagger: 是否使用门的转置共轭版本，默认为 False。
    :return: 一个 ``pyvqnet.qnn.vqc.torch.QModule`` 实例

    示例::

        from pyvqnet.qnn.vqc.torch import IsingZZ,QMachine
        import pyvqnet
        pyvqnet.backends.set_backend("torch")
        device = QMachine(4)
        layer = IsingZZ(has_params= True, trainable= True, wires=[0,2])
        batchsize = 2
        device.reset_states(batchsize)
        layer(q_machine = device)
        print(device.states)


IsingXY
""""""""""""""""""


.. py:class:: pyvqnet.qnn.vqc.torch.IsingXY(has_params: bool = False,trainable: bool = False,init_params=None,wires=None,dtype=pyvqnet.kcomplex64,use_dagger=False)

    定义一个 IsingXY 量子门。

    此类继承自 ``pyvqnet.qnn.vqc.torch.QModule`` 和 ``torch.nn.Module``\ 。
    此类可作为子模块添加到 torch 模型中。

    :param has_params: 是否有参数，如 RX、RY 等门需要设置为 True，无参数的门需要设置为 False，默认为 False。
    :param trainable: 是否有参数需要训练。如果该层使用外部输入数据构建逻辑门矩阵，则设置为 False。如果需要从该层初始化待训练参数，则为 True，默认为 False。
    :param init_params: 用于编码经典数据 QTensor 的初始化参数，默认为 None。
    :param wires: 线路作用的比特索引，默认为 None。
    :param dtype: 逻辑门内部矩阵的数据精度，可设置为 pyvqnet.kcomplex64 或 pyvqnet.kcomplex128，分别对应 float 输入或 double 输入。
    :param use_dagger: 是否使用门的转置共轭版本，默认为 False。
    :return: 一个 ``pyvqnet.qnn.vqc.torch.QModule`` 实例

    示例::

        from pyvqnet.qnn.vqc.torch import IsingXY,QMachine
        import pyvqnet
        pyvqnet.backends.set_backend("torch")
        device = QMachine(4)
        layer = IsingXY(has_params= True, trainable= True, wires=[0,2])
        batchsize = 2
        device.reset_states(batchsize)
        layer(q_machine = device)
        print(device.states)


PhaseShift
""""""""""""""""""


.. py:class:: pyvqnet.qnn.vqc.torch.PhaseShift(has_params: bool = False,trainable: bool = False,init_params=None,wires=None,dtype=pyvqnet.kcomplex64,use_dagger=False)

    定义一个 PhaseShift 量子门。

    此类继承自 ``pyvqnet.qnn.vqc.torch.QModule`` 和 ``torch.nn.Module``\ 。
    此类可作为子模块添加到 torch 模型中。

    :param has_params: 是否有参数，如 RX、RY 等门需要设置为 True，无参数的门需要设置为 False，默认为 False。
    :param trainable: 是否有参数需要训练。如果该层使用外部输入数据构建逻辑门矩阵，则设置为 False。如果需要从该层初始化待训练参数，则为 True，默认为 False。
    :param init_params: 用于编码经典数据 QTensor 的初始化参数，默认为 None。
    :param wires: 线路作用的比特索引，默认为 None。
    :param dtype: 逻辑门内部矩阵的数据精度，可设置为 pyvqnet.kcomplex64 或 pyvqnet.kcomplex128，分别对应 float 输入或 double 输入。
    :param use_dagger: 是否使用门的转置共轭版本，默认为 False。
    :return: 一个 ``pyvqnet.qnn.vqc.torch.QModule`` 实例

    示例::

        from pyvqnet.qnn.vqc.torch import PhaseShift,QMachine
        import pyvqnet
        pyvqnet.backends.set_backend("torch")
        device = QMachine(4)
        layer = PhaseShift(has_params= True, trainable= True, wires=1)
        batchsize = 2
        device.reset_states(batchsize)
        layer(q_machine = device)
        print(device.states)


MultiRZ
""""""""""""""""""

.. py:class:: pyvqnet.qnn.vqc.torch.MultiRZ(has_params: bool = False,trainable: bool = False,init_params=None,wires=None,dtype=pyvqnet.kcomplex64,use_dagger=False)

    定义一个 MultiRZ 量子门。

    此类继承自 ``pyvqnet.qnn.vqc.torch.QModule`` 和 ``torch.nn.Module``\ 。
    此类可作为子模块添加到 torch 模型中。

    :param has_params: 是否有参数，如 RX、RY 等门需要设置为 True，无参数的门需要设置为 False，默认为 False。
    :param trainable: 是否有参数需要训练。如果该层使用外部输入数据构建逻辑门矩阵，则设置为 False。如果需要从该层初始化待训练参数，则为 True，默认为 False。
    :param init_params: 用于编码经典数据 QTensor 的初始化参数，默认为 None。
    :param wires: 线路作用的比特索引，默认为 None。
    :param dtype: 逻辑门内部矩阵的数据精度，可设置为 pyvqnet.kcomplex64 或 pyvqnet.kcomplex128，分别对应 float 输入或 double 输入。
    :param use_dagger: 是否使用门的转置共轭版本，默认为 False。
    :return: 一个 ``pyvqnet.qnn.vqc.torch.QModule`` 实例

    示例::

        from pyvqnet.qnn.vqc.torch import MultiRZ,QMachine
        import pyvqnet
        pyvqnet.backends.set_backend("torch")
        device = QMachine(4)
        layer = MultiRZ(has_params= True, trainable= True, wires=[0,2])
        batchsize = 2
        device.reset_states(batchsize)
        layer(q_machine = device)
        print(device.states)



SDG
""""""""""""""""""


.. py:class:: pyvqnet.qnn.vqc.torch.SDG(has_params: bool = False,trainable: bool = False,init_params=None,wires=None,dtype=pyvqnet.kcomplex64,use_dagger=False)

    定义一个 SDG 量子门。

    此类继承自 ``pyvqnet.qnn.vqc.torch.QModule`` 和 ``torch.nn.Module``\ 。
    此类可作为子模块添加到 torch 模型中。

    :param has_params: 是否有参数，如 RX、RY 等门需要设置为 True，无参数的门需要设置为 False，默认为 False。
    :param trainable: 是否有参数需要训练。如果该层使用外部输入数据构建逻辑门矩阵，则设置为 False。如果需要从该层初始化待训练参数，则为 True，默认为 False。
    :param init_params: 用于编码经典数据 QTensor 的初始化参数，默认为 None。
    :param wires: 线路作用的比特索引，默认为 None。
    :param dtype: 逻辑门内部矩阵的数据精度，可设置为 pyvqnet.kcomplex64 或 pyvqnet.kcomplex128，分别对应 float 输入或 double 输入。
    :param use_dagger: 是否使用门的转置共轭版本，默认为 False。
    :return: 一个 ``pyvqnet.qnn.vqc.torch.QModule`` 实例

    示例::

        from pyvqnet.qnn.vqc.torch import SDG,QMachine
        import pyvqnet
        pyvqnet.backends.set_backend("torch")
        device = QMachine(4)
        layer = SDG(wires=0)
        batchsize = 1
        device.reset_states(1)
        layer(q_machine = device)
        print(device.states)




TDG
""""""""""""""""""

.. py:class:: pyvqnet.qnn.vqc.torch.TDG(has_params: bool = False,trainable: bool = False,init_params=None,wires=None,dtype=pyvqnet.kcomplex64,use_dagger=False)

    定义一个 SDG 量子门。

    此类继承自 ``pyvqnet.qnn.vqc.torch.QModule`` 和 ``torch.nn.Module``\ 。
    此类可作为子模块添加到 torch 模型中。

    :param has_params: 是否有参数，如 RX、RY 等门需要设置为 True，无参数的门需要设置为 False，默认为 False。
    :param trainable: 是否有参数需要训练。如果该层使用外部输入数据构建逻辑门矩阵，则设置为 False。如果需要从该层初始化待训练参数，则为 True，默认为 False。
    :param init_params: 用于编码经典数据 QTensor 的初始化参数，默认为 None。
    :param wires: 线路作用的比特索引，默认为 None。
    :param dtype: 逻辑门内部矩阵的数据精度，可设置为 pyvqnet.kcomplex64 或 pyvqnet.kcomplex128，分别对应 float 输入或 double 输入。
    :param use_dagger: 是否使用门的转置共轭版本，默认为 False。
    :return: 一个 ``pyvqnet.qnn.vqc.torch.QModule`` 实例

    示例::

        from pyvqnet.qnn.vqc.torch import TDG,QMachine
        import pyvqnet
        pyvqnet.backends.set_backend("torch")
        device = QMachine(4)
        layer = TDG(wires=0)
        batchsize = 1
        device.reset_states(1)
        layer(q_machine = device)
        print(device.states)



ControlledPhaseShift
"""""""""""""""""""""""""""""""""""""""""""""


.. py:class:: pyvqnet.qnn.vqc.torch.ControlledPhaseShift(has_params: bool = False,trainable: bool = False,init_params=None,wires=None,dtype=pyvqnet.kcomplex64,use_dagger=False)

    定义一个 ControlledPhaseShift 量子门。

    此类继承自 ``pyvqnet.qnn.vqc.torch.QModule`` 和 ``torch.nn.Module``\ 。
    此类可作为子模块添加到 torch 模型中。

    :param has_params: 是否有参数，如 RX、RY 等门需要设置为 True，无参数的门需要设置为 False，默认为 False。
    :param trainable: 是否有参数需要训练。如果该层使用外部输入数据构建逻辑门矩阵，则设置为 False。如果需要从该层初始化待训练参数，则为 True，默认为 False。
    :param init_params: 用于编码经典数据 QTensor 的初始化参数，默认为 None。
    :param wires: 线路作用的比特索引，默认为 None。
    :param dtype: 逻辑门内部矩阵的数据精度，可设置为 pyvqnet.kcomplex64 或 pyvqnet.kcomplex128，分别对应 float 输入或 double 输入。
    :param use_dagger: 是否使用门的转置共轭版本，默认为 False。
    :return: 一个 ``pyvqnet.qnn.vqc.torch.QModule`` 实例

    示例::

        from pyvqnet.qnn.vqc.torch import ControlledPhaseShift,QMachine
        import pyvqnet
        pyvqnet.backends.set_backend("torch")
        device = QMachine(4)
        layer = ControlledPhaseShift(has_params= True, trainable= True, wires=[0,2])
        batchsize = 2
        device.reset_states(batchsize)
        layer(q_machine = device)
        print(device.states)



MultiControlledX
"""""""""""""""""""""""""""""""""""""""""""""

.. py:class:: pyvqnet.qnn.vqc.torch.MultiControlledX(has_params: bool = False,trainable: bool = False,init_params=None,wires=None,dtype=pyvqnet.kcomplex64,use_dagger=False,control_values=None)

    定义一个 MultiControlledX 量子门。

    此类继承自 ``pyvqnet.qnn.vqc.torch.QModule`` 和 ``torch.nn.Module``\ 。
    此类可作为子模块添加到 torch 模型中。

    :param has_params: 是否有参数，如 RX、RY 等门需要设置为 True，无参数的门需要设置为 False，默认为 False。
    :param trainable: 是否有参数需要训练。如果该层使用外部输入数据构建逻辑门矩阵，则设置为 False。如果需要从该层初始化待训练参数，则为 True，默认为 False。
    :param init_params: 用于编码经典数据 QTensor 的初始化参数，默认为 None。
    :param wires: 线路作用的比特索引，默认为 None。
    :param dtype: 逻辑门内部矩阵的数据精度，可设置为 pyvqnet.kcomplex64 或 pyvqnet.kcomplex128，分别对应 float 输入或 double 输入。
    :param use_dagger: 是否使用门的转置共轭版本，默认为 False。
    :param control_values: 控制值，默认为 None，当比特为 1 时受控。

    :return: 一个 ``pyvqnet.qnn.vqc.torch.QModule`` 实例

    示例::

        import pyvqnet
        pyvqnet.backends.set_backend("torch")
        from pyvqnet.qnn.vqc.torch import QMachine,MultiControlledX
        from pyvqnet.tensor import QTensor,kcomplex64
        qm = QMachine(4,dtype=kcomplex64)
        qm.reset_states(2)
        mcx = MultiControlledX(
                        init_params=None,
                        wires=[2,3,0,1],
                        dtype=kcomplex64,
                        use_dagger=False,control_values=[1,0,0])
        y = mcx(q_machine = qm)
        print(qm.states)


测量 API
^^^^^^^^^^^^^^^^^^^^^^

Probability
"""""""""""""""""""""""""""""


.. py:class:: pyvqnet.qnn.vqc.torch.Probability(wires=None, name="")

    计算量子电路在特定比特上的概率测量结果。

    此类继承自 ``pyvqnet.qnn.vqc.torch.QModule`` 和 ``torch.nn.Module``\ 。
    此类可作为子模块添加到 torch 模型中。

    :param wires: 测量比特的索引，列表、元组或整数。
    :param name: 模块的名称，默认值：""。
    :return: 测量结果，QTensor。

    示例::

        import pyvqnet
        pyvqnet.backends.set_backend("torch")
        from pyvqnet.qnn.vqc.torch import Probability,rx,ry,cnot,QMachine,rz
        from pyvqnet.tensor import QTensor
        from pyvqnet import kfloat64
        x = QTensor([[0.56, 0.1],[0.56, 0.1]],requires_grad=True)
        qm = QMachine(4)
        qm.reset_states(2)
        rz(q_machine=qm,wires=0,params=x[:,[0]])
        rz(q_machine=qm,wires=1,params=x[:,[0]])
        cnot(q_machine=qm,wires=[0,1])
        ry(q_machine=qm,wires=2,params=x[:,[1]])
        cnot(q_machine=qm,wires=[0,2])
        rz(q_machine=qm,wires=3,params=x[:,[1]])
        ma = Probability(wires=1)
        y =ma(q_machine=qm)


MeasureAll
"""""""""""""""""""""""""""""

.. py:class:: pyvqnet.qnn.vqc.torch.MeasureAll(obs=None, name="")

    计算量子电路的测量结果，支持输入 obs 为多个或单个泡利算符或哈密顿量。
    例如：

    {\'X0\': 0.23} 表示对量子比特 0 施加 PauliX 效应，系数为 0.23。

    {\'X1 Z2\': 2.4,\'Y2\': -0.5} 对应观测值 2.4 * X1 @ Z2 - 0.5 * Y2。

    [{\'X1 Z2\': 4,\'Z1 Z0\': 3},{\'X1 Y2 Z0\': 3.5}] 对应两个观测值 4 * X1 @ Z2 + 3 * Z1 @ Z0 和 3.5 * X1 @ Y2 @ Z0。


    此类继承自 ``pyvqnet.qnn.vqc.torch.QModule`` 和 ``torch.nn.Module``\ 。

    此类可作为子模块添加到 torch 模型中。

    :param obs: 可观测量。
    :param name: 模块名称，默认值：""。
    :return: 测量结果，QTensor。

    示例::

        import pyvqnet
        pyvqnet.backends.set_backend("torch")
        from pyvqnet.qnn.vqc.torch import MeasureAll,rx,ry,cnot,QMachine,rz
        from pyvqnet.tensor import QTensor
        from pyvqnet import kfloat64
        x = QTensor([[0.56, 0.1],[0.56, 0.1]],requires_grad=True)
        qm = QMachine(4)
        qm.reset_states(2)
        rz(q_machine=qm,wires=0,params=x[:,[0]])
        rz(q_machine=qm,wires=1,params=x[:,[0]])
        cnot(q_machine=qm,wires=[0,1])
        ry(q_machine=qm,wires=2,params=x[:,[1]])
        cnot(q_machine=qm,wires=[0,2])
        rz(q_machine=qm,wires=3,params=x[:,[1]])
        obs_list = [{
            "Z0 Z1" :2
        }, {
            "X0 Z2" :1
        }]
        ma = MeasureAll(obs = obs_list)
        y = ma(q_machine=qm)
        print(y)



Samples
"""""""""""""""""""""""""""""

.. py:class:: pyvqnet.qnn.vqc.torch.Samples(wires=None, obs=None, shots = 1,name="")

    在特定线路上使用 shot 获取采样结果。

    此类继承自 ``pyvqnet.qnn.vqc.torch.QModule`` 和 ``torch.nn.Module``\ 。
    此类可作为子模块添加到 torch 模型中。


    :param wires: 采样量子比特索引。默认值：None，运行时使用模拟器的所有比特。
    :param obs: 此值只能为 None。
    :param shots: 采样重复次数，默认值：1。
    :param name: 此模块的名称，默认值：""。
    :return: 一个测量方法类

    示例::

        import pyvqnet
        pyvqnet.backends.set_backend("torch")
        from pyvqnet.qnn.vqc.torch import Samples,rx,ry,cnot,QMachine,rz
        from pyvqnet.tensor import QTensor
        from pyvqnet import kfloat64
        x = QTensor([[0.56, 0.1],[0.56, 0.1]],requires_grad=True)

        qm = QMachine(4)
        qm.reset_states(2)
        rz(q_machine=qm,wires=0,params=x[:,[0]])
        rx(q_machine=qm,wires=1,params=x[:,[0]])
        cnot(q_machine=qm,wires=[0,1])

        cnot(q_machine=qm,wires=[0,2])
        ry(q_machine=qm,wires=3,params=x[:,[1]])


        ma = Samples(wires=[0,1,2],shots=3)
        y = ma(q_machine=qm)
        print(y)


HermitianExpval
"""""""""""""""""""""""""""""

.. py:class:: pyvqnet.qnn.vqc.torch.HermitianExpval(obs=None, name="")

    计算量子电路中埃尔米特量的期望值。

    此类继承自 ``pyvqnet.qnn.vqc.torch.QModule`` 和 ``torch.nn.Module``\ 。
    此类可作为子模块添加到 torch 模型中。


    :param obs: 埃尔米特量。
    :param name: 模块名称，默认值：""。
    :return: 期望结果，QTensor。

    示例::

        import pyvqnet
        pyvqnet.backends.set_backend("torch")
        from pyvqnet.qnn.vqc.torch import QMachine, rx,ry,\
            RX, RY, CNOT, PauliX, PauliZ, VQC_RotCircuit,HermitianExpval
        from pyvqnet.tensor import QTensor, tensor
        from pyvqnet.nn import Parameter
        import numpy as np
        bsz = 3
        H = np.array([[8, 4, 0, -6], [4, 0, 4, 0], [0, 4, 8, 0], [-6, 0, 0, 0]])
        class QModel(pyvqnet.nn.Module):
            def __init__(self, num_wires, dtype):
                super(QModel, self).__init__()
                self.rot_param = Parameter((3, ))
                self.rot_param.copy_value_from(tensor.QTensor([-0.5, 1, 2.3]))
                self._num_wires = num_wires
                self._dtype = dtype
                self.qm = QMachine(num_wires, dtype=dtype)
                self.rx_layer1 = VQC_RotCircuit
                self.ry_layer2 = RY(has_params=True,
                                    trainable=True,
                                    wires=0,
                                    init_params=tensor.QTensor([-0.5]))
                self.xlayer = PauliX(wires=0)
                self.cnot = CNOT(wires=[0, 1])
                self.measure = HermitianExpval(obs = {'wires':(1,0),'observables':tensor.to_tensor(H)})

            def forward(self, x, *args, **kwargs):
                self.qm.reset_states(x.shape[0])

                rx(q_machine=self.qm, wires=0, params=x[:, [1]])
                ry(q_machine=self.qm, wires=1, params=x[:, [0]])
                self.xlayer(q_machine=self.qm)
                self.rx_layer1(params=self.rot_param, wire=1, q_machine=self.qm)
                self.ry_layer2(q_machine=self.qm)
                self.cnot(q_machine=self.qm)
                rlt = self.measure(q_machine = self.qm)

                return rlt


        input_x = tensor.arange(1, bsz * 2 + 1,
                                dtype=pyvqnet.kfloat32).reshape([bsz, 2])
        input_x.requires_grad = True

        qunatum_model = QModel(num_wires=2, dtype=pyvqnet.kcomplex64)

        batch_y = qunatum_model(input_x)
        batch_y.backward()

        print(batch_y)

量子电路通用模板
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

VQC_HardwareEfficientAnsatz
""""""""""""""""""""""""""""""""""""""""


.. py:class:: pyvqnet.qnn.vqc.torch.VQC_HardwareEfficientAnsatz(n_qubits,single_rot_gate_list,entangle_gate="CNOT",entangle_rules='linear',depth=1,initial=None,dtype=None)

    论文中介绍的 Hardware Efficient Ansatz 实现：`Hardware-efficient Variational Quantum Eigensolver for Small Molecules <https://arxiv.org/pdf/1704.05018.pdf>`__ 。

    此类继承自 ``pyvqnet.qnn.vqc.torch.QModule`` 和 ``torch.nn.Module``\ 。

    此类可作为子模块添加到 torch 模型中。


    :param n_qubits: 量子比特数量。
    :param single_rot_gate_list: 单一量子比特旋转门列表，由一个或多个作用于每个量子比特的旋转门构建。目前支持 Rx, Ry, Rz。
    :param entangle_gate: 非参数化纠缠门。支持 CNOT, CZ。默认值：CNOT。
    :param entangle_rules: 纠缠门在电路中的使用方式。'linear' 表示纠缠门将作用于每对相邻量子比特。'all' 表示纠缠门将作用于任意两个量子比特。默认值：linear。
    :param depth: ansatz 的深度，默认值：1。
    :param initial: 为所有参数初始化的相同值，默认值：None，此模块将随机初始化参数。
    :param dtype: 参数的数据类型。
    :return: 一个 VQC_HardwareEfficientAnsatz 实例。

    示例::

        from pyvqnet.nn.torch import TorchModule,Linear,TorchModuleList
        from pyvqnet.qnn.vqc.torch.qcircuit import VQC_HardwareEfficientAnsatz,RZZ,RZ
        from pyvqnet.qnn.vqc.torch import Probability,QMachine
        from pyvqnet import tensor
        import pyvqnet
        pyvqnet.backends.set_backend("torch")
        pyvqnet.utils.set_random_seed(25)

        class QM(TorchModule):
            def __init__(self, name=""):
                super().__init__(name)
                self.linearx = Linear(4,2)
                self.ansatz = VQC_HardwareEfficientAnsatz(4, ["rx", "RY", "rz"],
                                            entangle_gate="cnot",
                                            entangle_rules="linear",
                                            depth=2)
                self.encode1 = RZ(wires=0)
                self.encode2 = RZ(wires=1)
                self.measure = Probability(wires=[0,2])
                self.device = QMachine(4)
            def forward(self, x, *args, **kwargs):
                self.device.reset_states(x.shape[0])
                y = self.linearx(x)
                self.encode1(params = y[:, [0]],q_machine = self.device,)
                self.encode2(params = y[:, [1]],q_machine = self.device,)
                self.ansatz(q_machine =self.device)
                return self.measure(q_machine =self.device)

        bz =3
        inputx = tensor.arange(1.0,bz*4+1).reshape([bz,4])
        inputx.requires_grad= True
        qlayer = QM()
        y = qlayer(inputx)
        y.backward()
        print(y)



VQC_BasicEntanglerTemplate
""""""""""""""""""""""""""""""""""""""""

.. py:class:: pyvqnet.qnn.vqc.torch.VQC_BasicEntanglerTemplate(num_layer=1, num_qubits=1, rotation="RX", initial=None, dtype=None)

    该层由每个量子比特上的单参数单量子比特旋转组成，随后在闭合链或环组合中应用多个 CNOT 门。

    CNOT 门的环将每个量子比特连接到其邻居，最后一个量子比特被视为第 a 个量子比特的邻居。

    此类继承自 ``pyvqnet.qnn.vqc.torch.QModule`` 和 ``torch.nn.Module``\ 。

    此类可作为子模块添加到 torch 模型中。

    :param num_layers: 重复层数，默认值：1。
    :param num_qubits: 量子比特数量，默认值：1。
    :param rotation: 使用的单参数单量子比特门，默认值：`RX`
    :param initial: 所有参数的初始化相同值。默认值：None，参数将随机初始化。
    :param dtype: 参数的数据类型，默认值：None，使用 float32。
    :return: 一个 VQC_BasicEntanglerTemplate 实例

    示例::

        import pyvqnet
        pyvqnet.backends.set_backend("torch")
        from pyvqnet.qnn.vqc.torch import QModule,\
            VQC_BasicEntanglerTemplate, Probability, QMachine
        from pyvqnet import tensor


        class QM(QModule):
            def __init__(self, name=""):
                super().__init__(name)

                self.ansatz = VQC_BasicEntanglerTemplate(2,
                                                    4,
                                                    "rz",
                                                    initial=tensor.ones([1, 1]))

                self.measure = Probability(wires=[0, 2])
                self.device = QMachine(4)

            def forward(self,x, *args, **kwargs):

                self.ansatz(q_machine=self.device)
                return self.measure(q_machine=self.device)

        bz = 1
        inputx = tensor.arange(1.0, bz * 4 + 1).reshape([bz, 4])
        qlayer = QM()
        y = qlayer(inputx)
        y.backward()
        print(y)



VQC_StronglyEntanglingTemplate
""""""""""""""""""""""""""""""""""""""""

.. py:class:: pyvqnet.qnn.vqc.torch.VQC_StronglyEntanglingTemplate(num_layers=1, num_qubits=1, ranges=None,initial=None, dtype=None)

    由单量子比特旋转和纠缠器组成的层，如 `circuit-centric classifier design <https://arxiv.org/abs/1804.00633>`__ 中所述。

    此类继承自 ``pyvqnet.qnn.vqc.torch.QModule`` 和 ``torch.nn.Module``\ 。
    此类可作为子模块添加到 torch 模型中。


    :param num_layers: 重复层数，默认值：1。
    :param num_qubits: 量子比特数量，默认值：1。
    :param ranges: 确定每个后续层的范围超参数的序列；默认值：None
                                使用 :math: `r=l \mod M` 对于第 :math:`l` 层和 :math:`M` 个量子比特。
    :param initial: 所有参数的初始值。默认值：None，随机初始化。
    :param dtype: 参数的数据类型，默认值：None，使用 float32。
    :return: 一个 VQC_StronglyEntanglingTemplate 实例。

    示例::

        from pyvqnet.nn.torch import TorchModule,Linear,TorchModuleList
        from pyvqnet.qnn.vqc.torch.qcircuit import VQC_StronglyEntanglingTemplate
        from pyvqnet.qnn.vqc.torch import Probability, QMachine
        from pyvqnet import tensor
        import pyvqnet

        pyvqnet.backends.set_backend("torch")
        pyvqnet.utils.set_random_seed(25)
        class QM(TorchModule):
            def __init__(self, name=""):
                super().__init__(name)

                self.ansatz = VQC_StronglyEntanglingTemplate(2,
                                                    4,
                                                    None,
                                                    initial=tensor.ones([1, 1]))

                self.measure = Probability(wires=[0, 1])
                self.device = QMachine(4)

            def forward(self,x, *args, **kwargs):

                self.ansatz(q_machine=self.device)
                return self.measure(q_machine=self.device)

        bz = 1
        inputx = tensor.arange(1.0, bz * 4 + 1).reshape([bz, 4])
        qlayer = QM()
        y = qlayer(inputx)
        y.backward()
        print(y)



VQC_QuantumEmbedding
""""""""""""""""""""""""""""""""""""""""


.. py:class:: pyvqnet.qnn.vqc.torch.VQC_QuantumEmbedding(qubits, machine, num_repetitions_input, depth_input, num_unitary_layers, num_repetitions,initial = None,dtype = None,name= "")

    使用 RZ,RY,RZ 创建变分量子电路，将经典数据编码为量子态。
    参考 `Quantum embeddings for machine learning <https://arxiv.org/abs/2001.03622>`_。

    此类继承自 ``pyvqnet.qnn.vqc.torch.QModule`` 和 ``torch.nn.Module``\ 。
    此类可作为子模块添加到 torch 模型中。


    :param num_repetitions_input: 在子模块中编码输入的重复次数。
    :param depth_input: 输入维度数量。
    :param num_unitary_layers: 变分量子门的重复次数。
    :param num_repetitions: 子模块的重复次数。
    :param initial: 参数初始化值，默认为 None
    :param dtype: 参数类型，默认为 None，使用 float32。
    :param name: 类名称
    :return: 一个 VQC_QuantumEmbedding 实例。

    示例::

        from pyvqnet.nn.torch import TorchModule
        from pyvqnet.qnn.vqc.torch.qcircuit import VQC_QuantumEmbedding
        from pyvqnet.qnn.vqc.torch import Probability, QMachine, MeasureAll
        from pyvqnet import tensor
        import pyvqnet

        pyvqnet.backends.set_backend("torch")
        pyvqnet.utils.set_random_seed(25)
        depth_input = 2
        num_repetitions = 2
        num_repetitions_input = 2
        num_unitary_layers = 2
        nq = depth_input * num_repetitions_input
        bz = 12

        class QM(TorchModule):
            def __init__(self, name=""):
                super().__init__(name)

                self.ansatz = VQC_QuantumEmbedding(num_repetitions_input, depth_input,
                                                num_unitary_layers,
                                                num_repetitions, initial=tensor.full([1],12.0),dtype=pyvqnet.kfloat32)

                self.measure = MeasureAll(obs={f"Z{nq-1}":1})
                self.device = QMachine(nq)

            def forward(self, x, *args, **kwargs):
                self.device.reset_states(x.shape[0])
                self.ansatz(x,q_machine=self.device)
                return self.measure(q_machine=self.device)

        inputx = tensor.arange(1.0, bz * depth_input + 1).reshape([bz, depth_input])
        qlayer = QM()
        y = qlayer(inputx)
        y.backward()
        print(y)


ExpressiveEntanglingAnsatz
""""""""""""""""""""""""""""""""""""""""

.. py:class:: pyvqnet.qnn.vqc.torch.ExpressiveEntanglingAnsatz(type: int, num_wires: int, depth: int, dtype=None, name: str = "")

    来自论文的 19 种不同 ansatz：`Expressibility and entangling capability of parameterized quantum circuits for hybrid quantum-classical algorithms <https://arxiv.org/pdf/1905.10876.pdf>`_。

    此类继承自 ``pyvqnet.qnn.vqc.torch.QModule`` 和 ``torch.nn.Module``\ 。

    此类可作为子模块添加到 torch 模型中。

    :param type: 电路类型从 1 到 19，共 19 种线路。
    :param num_wires: 量子比特数量。
    :param depth: 电路深度。
    :param dtype: 参数的数据类型，默认值：None，使用 float32。
    :param name: 名称，默认为 ""。

    :return:
        一个 ExpressiveEntanglingAnsatz 实例

    示例::

        from pyvqnet.nn.torch import TorchModule
        from pyvqnet.qnn.vqc.torch.qcircuit import ExpressiveEntanglingAnsatz
        from pyvqnet.qnn.vqc.torch import Probability, QMachine, MeasureAll
        from pyvqnet import tensor
        import pyvqnet

        pyvqnet.backends.set_backend("torch")
        pyvqnet.utils.set_random_seed(25)

        class QModel(TorchModule):
            def __init__(self, num_wires, dtype,grad_mode=""):
                super(QModel, self).__init__()

                self._num_wires = num_wires
                self._dtype = dtype
                self.qm = QMachine(num_wires, dtype=dtype,grad_mode=grad_mode)
                self.c1 = ExpressiveEntanglingAnsatz(1,3,2)
                self.measure = MeasureAll(obs={
                    "Z1":1
                })

            def forward(self, x, *args, **kwargs):
                self.qm.reset_states(x.shape[0])
                self.c1(q_machine = self.qm)
                rlt = self.measure(q_machine=self.qm)
                return rlt


        input_x = tensor.QTensor([[0.1, 0.2, 0.3]])

        qunatum_model = QModel(num_wires=3, dtype=pyvqnet.kcomplex64)

        batch_y = qunatum_model(input_x)
        batch_y.backward()
        print(batch_y)



vqc_basis_embedding
""""""""""""""""""""""""""""""""""""""""

.. py:function:: pyvqnet.qnn.vqc.torch.vqc_basis_embedding(basis_state,q_machine)

    将 n 个二进制特征编码为 ``q_machine`` 的 n 量子比特基态。此函数的别名为 `VQC_BasisEmbedding`\ 。

    例如，对于 ``basis_state=([0, 1, 1])``\ ，量子系统中的基态为 :math:`|011 \rangle`\ 。

    :param basis_state: ``(n)`` 大小的二进制输入。
    :param q_machine: 量子机器设备。


    示例::

        import pyvqnet
        pyvqnet.backends.set_backend("torch")
        from pyvqnet.qnn.vqc.torch import vqc_basis_embedding,QMachine
        qm  = QMachine(3)
        vqc_basis_embedding(basis_state=[1,1,0],q_machine=qm)
        print(qm.states)




vqc_angle_embedding
""""""""""""""""""""""""""""""""""""""""


.. py:function:: pyvqnet.qnn.vqc.torch.vqc_angle_embedding(input_feat, wires, q_machine: pyvqnet.qnn.vqc.torch.QMachine, rotation: str = "X")

    将 :math:`N` 个特征编码为 :math:`n` 个量子比特的旋转角度，其中 :math:`N \leq n`\ 。
    此函数的别名为 `VQC_AngleEmbedding` 。

    旋转方式可选为：'X'、'Y'、'Z'，由 ``rotation`` 参数定义：

    * ``rotation='X'`` 使用特征作为 RX 旋转的角度。

    * ``rotation='Y'`` 使用特征作为 RY 旋转的角度。

    * ``rotation='Z'`` 使用特征作为 RZ 旋转的角度。

    ``wires`` 表示量子比特上旋转门的索引。

    :param input_feat: 表示参数的数组。
    :param wires: 量子比特索引。
    :param q_machine: 量子机器设备。
    :param rotation: 旋转门，默认为 "X"。

    示例::

        import pyvqnet
        pyvqnet.backends.set_backend("torch")
        from pyvqnet.qnn.vqc.torch import vqc_angle_embedding, QMachine
        from pyvqnet.tensor import QTensor
        qm  = QMachine(2)
        vqc_angle_embedding(QTensor([2.2, 1]), [0, 1], q_machine=qm, rotation='X')
        print(qm.states)
        vqc_angle_embedding(QTensor([2.2, 1]), [0, 1], q_machine=qm, rotation='Y')
        print(qm.states)
        vqc_angle_embedding(QTensor([2.2, 1]), [0, 1], q_machine=qm, rotation='Z')
        print(qm.states)



vqc_amplitude_embedding
""""""""""""""""""""""""""""""""""""""""

.. py:function:: pyvqnet.qnn.vqc.torch.vqc_amplitude_embeddingVQC_AmplitudeEmbeddingCircuit(input_feature, q_machine)

    将 :math:`2^n` 个特征编码为 :math:`n` 个量子比特的振幅向量。此函数的别名为 `VQC_AmplitudeEmbedding`\ 。

    :param input_feature: 表示参数的 numpy 数组。
    :param q_machine: 量子机器设备。


    示例::

        import pyvqnet
        pyvqnet.backends.set_backend("torch")
        from pyvqnet.qnn.vqc.torch import vqc_amplitude_embedding, QMachine
        from pyvqnet.tensor import QTensor
        qm  = QMachine(3)
        vqc_amplitude_embedding(QTensor([3.2,-2,-2,0.3,12,0.1,2,-1]), q_machine=qm)
        print(qm.states)



vqc_iqp_embedding
""""""""""""""""""""""""""""""""""""""""

.. py:function:: pyvqnet.qnn.vqc.vqc_iqp_embedding(input_feat, q_machine: pyvqnet.qnn.vqc.torch.QMachine, rep: int = 1)

    使用 IQP 电路的对角门将 :math:`n` 个特征编码为 :math:`n` 个量子比特。别名：``VQC_IQPEmbedding`` 。

    该编码由 `Havlicek et al. (2018) <https://arxiv.org/pdf/1804.11326.pdf>`_ 提出。

    通过指定 ``rep``\ ，可以重复基本的 IQP 电路。

    :param input_feat: 参数的数组。
    :param q_machine: 量子机器。
    :param rep: 重复量子电路块的次数，默认为 1。

    示例::

        import pyvqnet
        pyvqnet.backends.set_backend("torch")
        from pyvqnet.qnn.vqc.torch import vqc_iqp_embedding, QMachine
        from pyvqnet.tensor import QTensor
        qm  = QMachine(3)
        vqc_iqp_embedding(QTensor([3.2,-2,-2]), q_machine=qm)
        print(qm.states)



vqc_rotcircuit
""""""""""""""""""""""""""""""""""""""""

.. py:function:: pyvqnet.qnn.vqc.torch.vqc_rotcircuit(q_machine, wire, params)

    任意单量子比特旋转量子逻辑门组合。此函数别名：``VQC_RotCircuit`` 。

    .. math::
        R(\phi,\theta,\omega) = RZ(\omega)RY(\theta)RZ(\phi)= \begin{bmatrix}
        e^{-i(\phi+\omega)/2}\cos(\theta/2) & -e^{i(\phi-\omega)/2}\sin(\theta/2) \\
        e^{-i(\phi-\omega)/2}\sin(\theta/2) & e^{i(\phi+\omega)/2}\cos(\theta/2)
        \end{bmatrix}.

    :param q_machine: 量子虚拟机设备。
    :param wire: 量子比特索引。
    :param params: 表示参数 :math:`[\phi, \theta, \omega]`\ 。

    示例::

        import pyvqnet
        pyvqnet.backends.set_backend("torch")
        from pyvqnet.qnn.vqc.torch import vqc_rotcircuit, QMachine
        from pyvqnet.tensor import QTensor
        qm  = QMachine(3)
        vqc_rotcircuit(q_machine=qm, wire=[1],params=QTensor([2.0,1.5,2.1]))
        print(qm.states)


vqc_crot_circuit
""""""""""""""""""""""""""""""""""""""""


.. py:function:: pyvqnet.qnn.vqc.torch.vqc_crot_circuit(para,control_qubits,rot_wire,q_machine)

    受控旋转单量子比特旋转的量子逻辑门组合。此函数别名：``VQC_CRotCircuit`` 。

    .. math::
        CR(\phi, \theta, \omega) = \begin{bmatrix}
        1 & 0 & 0 & 0 \\
        0 & 1 & 0 & 0\\
        0 & 0 & e^{-i(\phi+\omega)/2}\cos(\theta/2) & -e^{i(\phi-\omega)/2}\sin(\theta/2)\\
        0 & 0 & e^{-i(\phi-\omega)/2}\sin(\theta/2) & e^{i(\phi+\omega)/2}\cos(\theta/2)
        \end{bmatrix}.

    :param para: 表示参数的数组。
    :param control_qubits: 控制量子比特索引。
    :param rot_wire: 旋转量子比特索引。
    :param q_machine: 量子机器设备。


    示例::

        import pyvqnet
        pyvqnet.backends.set_backend("torch")
        from pyvqnet.tensor import QTensor
        from pyvqnet.qnn.vqc.torch import vqc_crot_circuit,QMachine, MeasureAll
        p = QTensor([2, 3, 4.0])
        qm = QMachine(2)
        vqc_crot_circuit(p, 0, 1, qm)
        m = MeasureAll(obs={"Z0": 1})
        exp = m(q_machine=qm)
        print(exp)




vqc_controlled_hadamard
""""""""""""""""""""""""""""""""""""""""


.. py:function:: pyvqnet.qnn.vqc.torch.vqc_controlled_hadamard(wires, q_machine)

    受控 Hadamard 逻辑门量子电路。此函数别名：``VQC_Controlled_Hadamard`` 。

    .. math::
        CH = \begin{bmatrix}
        1 & 0 & 0 & 0 \\
        0 & 1 & 0 & 0 \\
        0 & 0 & \frac{1}{\sqrt{2}} & \frac{1}{\sqrt{2}} \\
        0 & 0 & \frac{1}{\sqrt{2}} & -\frac{1}{\sqrt{2}}
        \end{bmatrix}.

    :param wires: 量子比特索引列表，第一个为控制比特，列表长度为 2。
    :param q_machine: 量子虚拟机设备。

    示例::

        import pyvqnet
        pyvqnet.backends.set_backend("torch")
        from pyvqnet.tensor import QTensor
        from pyvqnet.qnn.vqc.torch import vqc_controlled_hadamard,\
            QMachine, MeasureAll

        p = QTensor([0.2, 3, 4.0])
        qm = QMachine(3)
        vqc_controlled_hadamard([1, 0], qm)
        m = MeasureAll(obs={"Z0": 1})
        exp = m(q_machine=qm)
        print(exp)



vqc_ccz
""""""""""""""""""""""""""""""""""""""""

.. py:function:: pyvqnet.qnn.vqc.torch.vqc_ccz(wires, q_machine)

    受控-受控-Z 逻辑门。别名：``VQC_CCZ`` 。

    .. math::
        CCZ =
        \begin{pmatrix}
        1 & 0 & 0 & 0 & 0 & 0 & 0 & 0\\
        0 & 1 & 0 & 0 & 0 & 0 & 0 & 0 & 0\\
        0 & 0 & 1 & 0 & 0 & 0 & 0 & 0\\
        0 & 0 & 0 & 1 & 0 & 0 & 0 & 0\\
        0 & 0 & 0 & 0 & 1 & 0 & 0 & 0\\
        0 & 0 & 0 & 0 & 0 & 1 & 0 & 0 & 0\\
        0 & 0 & 0 & 0 & 0 & 1 & 0 & 0\\
        0 & 0 & 0 & 0 & 0 & 0 & 1 & 0 & 0\\
        0 & 0 & 0 & 0 & 0 & 0 & 0 & 1 & 0\\
        0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & -1
        \end{pmatrix}

    :param wires: 量子比特索引列表，第一个为控制比特。列表长度为 3。
    :param q_machine: 量子虚拟机设备。

    示例::

        import pyvqnet
        pyvqnet.backends.set_backend("torch")
        from pyvqnet.tensor import QTensor
        from pyvqnet.qnn.vqc.torch import vqc_ccz,QMachine, MeasureAll
        p = QTensor([0.2, 3, 4.0])

        qm = QMachine(3)

        vqc_ccz([1, 0, 2], qm)
        m = MeasureAll(obs={"Z0": 1})
        exp = m(q_machine=qm)
        print(exp)



vqc_fermionic_single_excitation
""""""""""""""""""""""""""""""""""""""""

.. py:function:: pyvqnet.qnn.vqc.torch.vqc_fermionic_single_excitation(weight, wires, q_machine)

    耦合簇单激发算子的泡利矩阵张量积。矩阵形式如下：

    .. math::
        \hat{U}_{pr}(\theta) = \mathrm{exp} \{ \theta_{pr} (\hat{c}_p^\dagger \hat{c}_r
        -\mathrm{H.c.}) \},

    别名：``VQC_FermionicSingleExcitation`` 。

    :param weight: 量子比特 p 上的参数，仅一个元素。
    :param wires: 区间 [r, p] 中的量子比特索引子集。最小长度必须为 2。第一个索引值解释为 r，最后一个索引值解释为 p。中间索引由 CNOT 门操作，以计算量子比特集的奇偶性。
    :param q_machine: 量子虚拟机设备。



    示例::

        import pyvqnet
        pyvqnet.backends.set_backend("torch")
        from pyvqnet.tensor import QTensor
        from pyvqnet.qnn.vqc.torch import vqc_fermionic_single_excitation,\
            QMachine, MeasureAll
        qm = QMachine(3)
        p0 = QTensor([0.5])

        vqc_fermionic_single_excitation(p0, [1, 0, 2], qm)
        m = MeasureAll(obs={"Z0": 1})
        exp = m(q_machine=qm)
        print(exp)

 


vqc_fermionic_double_excitation
""""""""""""""""""""""""""""""""""""""""


.. py:function:: pyvqnet.qnn.vqc.torch.vqc_fermionic_double_excitation(weight, wires1, wires2, q_machine)

    耦合簇双激发算子的泡利矩阵张量积指数化，矩阵形式如下：

    .. math::
        \hat{U}_{pqrs}(\theta) = \mathrm{exp} \{ \theta (\hat{c}_p^\dagger \hat{c}_q^\dagger
        \hat{c}_r \hat{c}_s - \mathrm{H.c.}) \},

    其中 :math:`\hat{c}` 和 :math:`\hat{c}^\dagger` 是费米子湮灭和产生算符，索引 :math:`r, s` 和 :math:`p, q` 分别位于占据和空分子轨道上。使用 `Jordan-Wigner 变换
    <https://arxiv.org/abs/1208.5986>`_，上述定义的费米子算符可以用泡利矩阵表示（详见
    `arXiv:1805.04340 <https://arxiv.org/abs/1805.04340>`_）

    .. math::
        \hat{U}_{pqrs}(\theta) = \mathrm{exp} \Big\{
        \frac{i\theta}{8} \bigotimes_{b=s+1}^{r-1} \hat{Z}_b \bigotimes_{a=q+1}^{p-1}
        \hat{Z}_a (\hat{X}_s \hat{X}_r \hat{Y}_q \hat{X}_p +
        \hat{Y}_s \hat{X}_r \hat{Y}_q \hat{Y}_p +\\ \hat{X}_s \hat{Y}_r \hat{Y}_q \hat{Y}_p +
        \hat{X}_s \hat{X}_r \hat{X}_q \hat{Y}_p - \mathrm{H.c.} ) \Big\}

    此函数别名为：``VQC_FermionicDoubleExcitation`` 。

    :param weight: 可变参数
    :param wires1: 表示索引列表区间 [s, r] 中的量子比特子集。第一个索引解释为 s，最后一个索引解释为 r。CNOT 门操作中间索引以计算一组量子比特的奇偶性。
    :param wires2: 表示索引列表区间 [q, p] 中的量子比特子集。第一个根索引解释为 q，最后一个索引解释为 p。CNOT 门操作中间索引以计算一组量子比特的奇偶性。
    :param q_machine: 量子虚拟机设备。



    示例::

        import pyvqnet
        pyvqnet.backends.set_backend("torch")
        from pyvqnet.tensor import QTensor
        from pyvqnet.qnn.vqc.torch import vqc_fermionic_double_excitation,\
            QMachine, MeasureAll
        qm = QMachine(5)
        p0 = QTensor([0.5])

        vqc_fermionic_double_excitation(p0, [0, 1], [2, 3], qm)
        m = MeasureAll(obs={"Z0": 1})
        exp = m(q_machine=qm)
        print(exp)
 

vqc_uccsd
""""""""""""""""""""""""""""""""""""""""


.. py:function:: pyvqnet.qnn.vqc.torch.vqc_uccsd(weights, wires, s_wires, d_wires, init_state, q_machine)

    实现酉耦合簇单双激发模拟（UCCSD）。UCCSD 是一种常用于运行量子化学模拟的 VQE 模拟。

    在一阶 Trotter 近似下，UCCSD 酉函数如下：

    .. math::
        \hat{U}(\vec{\theta}) =
        \prod_{p > r} \mathrm{exp} \Big\{\theta_{pr}
        (\hat{c}_p^\dagger \hat{c}_r-\mathrm{H.c.}) \Big\}
        \prod_{p > q > r > s} \mathrm{exp} \Big\{\theta_{pqrs}
        (\hat{c}_p^\dagger \hat{c}_q^\dagger \hat{c}_r \hat{c}_s-\mathrm{H.c.}) \Big\}

    其中 :math:`\hat{c}` 和 :math:`\hat{c}^\dagger` 是费米子湮灭和产生算符，索引 :math:`r, s` 和 :math:`p, q` 分别位于占据和空分子轨道上。（更多详情请参见
    `arXiv:1805.04340 <https://arxiv.org/abs/1805.04340>`_）：

    此函数别名为：``VQC_UCCSD`` 。

    :param weights: 大小为 ``(len(s_wires)+ len(d_wires))`` 的张量，包含输入 Z 旋转 ``FermionicSingleExcitation`` 和 ``FermionicDoubleExcitation`` 的参数 :math:`\theta_{pr}` 和 :math:`\theta_{pqrs}`\ 。
    :param wires: 模板作用的量子比特索引
    :param s_wires: 包含量子比特索引 ``[r,...,p]`` 的列表序列，由单激发 :math:`\vert r, p \rangle = \hat{c}_p^\dagger \hat{c}_r \vert \mathrm{HF} \rangle` 生成，其中 :math:`\vert \mathrm{HF} \rangle` 表示 Hartee-Fock 参考态。
    :param d_wires: 列表序列，每个列表包含两个指定索引 ``[s, ...,r]`` 和 ``[q,..., p]`` 的列表，定义双激发 :math:`\vert s, r, q, p \rangle = \hat{c}_p^\dagger \hat{c}_q^\dagger \hat{c}_r\hat{c}_s \vert \mathrm{HF} \rangle`\ 。
    :param init_state: 长度为 ``len(wires)`` 的占据数向量，表示高频状态。``init_state`` 量子比特的初始化状态。
    :param q_machine: 量子虚拟机设备。

    示例::

        import pyvqnet
        pyvqnet.backends.set_backend("torch")
        from pyvqnet.qnn.vqc.torch import vqc_uccsd, QMachine, MeasureAll
        from pyvqnet.tensor import QTensor
        p0 = QTensor([2, 0.5, -0.2, 0.3, -2, 1, 3, 0])
        s_wires = [[0, 1, 2], [0, 1, 2, 3, 4], [1, 2, 3], [1, 2, 3, 4, 5]]
        d_wires = [[[0, 1], [2, 3]], [[0, 1], [2, 3, 4, 5]], [[0, 1], [3, 4]],
                [[0, 1], [4, 5]]]
        qm = QMachine(6)

        vqc_uccsd(p0, range(6), s_wires, d_wires, QTensor([1.0, 1, 0, 0, 0, 0]), qm)
        m = MeasureAll(obs={"Z1": 1})
        exp = m(q_machine=qm)
        print(exp)

        # [[0.963802]]


vqc_zfeaturemap
""""""""""""""""""""""""""""""""""""""""

.. py:function:: pyvqnet.qnn.vqc.torch.vqc_zfeaturemap(input_feat, q_machine: pyvqnet.qnn.vqc.torch.QMachine, data_map_func=None, rep: int = 2)

    一阶泡利 Z 演化电路。

    对于 3 个量子比特和 2 次重复，电路表示为：

    .. parsed-literal::

        ┌───┐┌──────────────┐┌───┐┌──────────────┐
        ┤ H ├┤ U1(2.0*x[0]) ├┤ H ├┤ U1(2.0*x[0]) ├
        ├───┤├──────────────┤├───┤├──────────────┤
        ┤ H ├┤ U1(2.0*x[1]) ├┤ H ├┤ U1(2.0*x[1]) ├
        ├───┤├──────────────┤├───┤├──────────────┤
        ┤ H ├┤ U1(2.0*x[2]) ├┤ H ├┤ U1(2.0*x[2]) ├
        └───┘└──────────────┘└───┘└──────────────┘

    泡利字符串固定为 ``Z``\ 。因此，一阶展开将是一个不带纠缠门的电路。

    :param input_feat: 表示输入参数的数组。
    :param q_machine: 量子虚拟机。
    :param data_map_func: 参数映射矩阵，一个可调用函数，设计为：``data_map_func = lambda x: x``\ 。
    :param rep: 模块重复的次数。

    示例::

        import pyvqnet
        pyvqnet.backends.set_backend("torch")
        from pyvqnet.qnn.vqc.torch import vqc_zfeaturemap, QMachine, hadamard
        from pyvqnet.tensor import QTensor
        qm = QMachine(3)
        for i in range(3):
            hadamard(q_machine=qm, wires=[i])
        vqc_zfeaturemap(input_feat=QTensor([[0.1,0.2,0.3]]),q_machine = qm)
        print(qm.states)
 

vqc_zzfeaturemap
""""""""""""""""""""""""""""""""""""""""

.. py:function:: pyvqnet.qnn.vqc.torch.vqc_zzfeaturemap(input_feat, q_machine: pyvqnet.qnn.vqc.torch.QMachine, data_map_func=None, entanglement: Union[str, List[List[int]],Callable[[int], List[int]]] = "full",rep: int = 2)

    二阶泡利 Z 演化电路。

    对于 3 个量子比特、1 次重复和线性纠缠，电路表示为：

    .. parsed-literal::


        ┌───┐┌─────────────────┐
        ┤ H ├┤ U1(2.0*φ(x[0])) ├──■────────────────────────────■────────────────────────────────────
        ├───┤├─────────────────┤┌─┴─┐┌──────────────────────┐┌─┴─┐
        ┤ H ├┤ U1(2.0*φ(x[1])) ├┤ X ├┤ U1(2.0*φ(x[0],x[1])) ├┤ X ├──■────────────────────────────■──
        ├───┤├─────────────────┤└───┘└──────────────────────┘└───┘┌─┴─┐┌──────────────────────┐┌─┴─┐
        ┤ H ├┤ U1(2.0*φ(x[2])) ├──────────────────────────────────┤ X ├┤ U1(2.0*φ(x[1],x[2])) ├┤ X ├
        └───┘└─────────────────┘                                  └───┘└──────────────────────┘└───┘

    其中 ``φ`` 是一个经典非线性函数。如果输入两个值，``φ(x,y) = (pi - x)(pi - y)``\ ，如果输入一个值，``φ(x) = x``\ 。使用 ``data_map_func`` 表示如下：

    .. code-block::

        def data_map_func(x):
            coeff = x if x.shape[-1] == 1 else ft.reduce(lambda x, y: (np.pi - x) * (np.pi - y), x)
            return coeff

    :param input_feat: 表示输入参数的数组。
    :param q_machine: 量子虚拟机。
    :param data_map_func: 参数映射矩阵，一个可调用函数。
    :param entanglement: 指定的纠缠结构。
    :param rep: 模块重复次数。

    示例::

        import pyvqnet
        pyvqnet.backends.set_backend("torch")
        from pyvqnet.qnn.vqc.torch import vqc_zzfeaturemap, QMachine
        from pyvqnet.tensor import QTensor

        qm = QMachine(3)
        vqc_zzfeaturemap(q_machine=qm, input_feat=QTensor([[0.1,0.2,0.3]]))
        print(qm.states)


vqc_allsinglesdoubles
""""""""""""""""""""""""""""""""""""""""

.. py:function:: pyvqnet.qnn.vqc.torch.vqc_allsinglesdoubles(weights, q_machine: pyvqnet.qnn.vqc.torch.QMachine, hf_state, wires, singles=None, doubles=None)

    在这种情况下，我们有四个单激发和双激发来保持 Hartree-Fock 态的总自旋投影。

    得到的酉矩阵保持粒子数，并准备 n-量子比特系统处于初始 Hartree-Fock 态和其他编码多激发配置的态的叠加态。

    :param weights: 大小为 ``(len(singles) + len(doubles),)`` 的 QTensor，包含依次进入 vqc.qCircuit.single_excitation 和 vqc.qCircuit.double_excitation 操作的角度
    :param q_machine: 量子机器。
    :param hf_state: 长度为 ``len(wires)`` 的占据数向量，表示 Hartree-Fock 态，``hf_state`` 用于初始化线路。
    :param wires: 要作用的量子比特。
    :param singles: 包含由 single_exitation 操作作用的两个量子比特索引的列表序列。
    :param doubles: 包含由 double_exitation 操作作用的两个量子比特索引的列表序列。

    例如，两个电子和六个量子比特的量子电路如下所示：

    .. image:: ./images/all_singles_doubles.png
        :width: 600 px
        :align: center

    |

    示例::

        import pyvqnet
        pyvqnet.backends.set_backend("torch")
        from pyvqnet.qnn.vqc.torch import vqc_allsinglesdoubles, QMachine

        from pyvqnet.tensor import QTensor
        qubits = 4
        qm = QMachine(qubits)

        vqc_allsinglesdoubles(q_machine=qm, weights=QTensor([0.55, 0.11, 0.53]),
                              hf_state = QTensor([1,1,0,0]), singles=[[0, 2], [1, 3]], doubles=[[0, 1, 2, 3]], wires=[0,1,2,3])
        print(qm.states)

vqc_basisrotation
""""""""""""""""""""""""""""""""""""""""

.. py:function:: pyvqnet.qnn.vqc.torch.vqc_basisrotation(q_machine: pyvqnet.qnn.vqc.torch.QMachine, wires, unitary_matrix: QTensor, check=False)

    实现一个电路，提供可用于执行精确单单元基旋转的集合。该电路源于 `arXiv:1711.04789 <https://arxiv.org/abs/1711.04789>`_ 中给出的单粒子费米子确定的酉变换 :math:`U(u)`

    .. math::
        U(u) = \exp{\left( \sum_{pq} \left[\log u \right]_{pq} (a_p^\dagger a_q - a_q^\dagger a_p) \right)}.

    :math:`U(u)` 通过使用论文 `Optica, 3, 1460 (2016) <https://opg.optica.org/optica/fulltext.cfm?uri=optica-3-12-1460&id=355743>`_ 中给出的方案获得。

    :param q_machine: 量子机器。
    :param wires: 要作用的量子比特。
    :param unitary_matrix: 指定变换基的矩阵。
    :param check: 检查 `unitary_matrix` 是否为酉矩阵。

    示例::

        import pyvqnet

        pyvqnet.backends.set_backend("torch")
        from pyvqnet.qnn.vqc.torch import vqc_basisrotation, QMachine
        from pyvqnet.tensor import QTensor
        import numpy as np

        V = np.array([[0.73678 + 0.27511j, -0.5095 + 0.10704j, -0.06847 + 0.32515j],
                    [0.73678 + 0.27511j, -0.5095 + 0.10704j, -0.06847 + 0.32515j],
                    [-0.21271 + 0.34938j, -0.38853 + 0.36497j, 0.61467 - 0.41317j]])

        eigen_vals, eigen_vecs = np.linalg.eigh(V)
        umat = eigen_vecs.T
        wires = range(len(umat))

        qm = QMachine(len(umat))

        vqc_basisrotation(q_machine=qm,
                        wires=wires,
                        unitary_matrix=QTensor(umat, dtype=qm.state.dtype))

        print(qm.states)



vqc_quantumpooling_circuit
""""""""""""""""""""""""""""""""""""""""

.. py:function:: pyvqnet.qnn.vqc.torch.vqc_quantumpooling_circuit(ignored_wires, sinks_wires, params, q_machine)

    对数据进行下采样的量子电路。

    为了减少电路中的量子比特数，首先在系统中创建量子比特对。初始配对所有量子比特后，对每对量子比特应用一个广义的 2-量子比特酉操作。应用这些两量子比特酉操作后，每对量子比特中有一个量子比特将在神经网络的其余部分被忽略。

    :param sources_wires: 将被忽略的源量子比特索引。
    :param sinks_wires: 将被保留的目标量子比特索引。
    :param params: 输入参数。
    :param q_machine: 量子虚拟机设备。

    示例::

        import pyvqnet

        pyvqnet.backends.set_backend("torch")
        from pyvqnet.qnn.vqc.torch import vqc_quantumpooling_circuit, QMachine, MeasureAll
        from pyvqnet import tensor
        p = tensor.full([6], 0.35)
        qm = QMachine(4)
        vqc_quantumpooling_circuit(q_machine=qm,
                                ignored_wires=[0, 1],
                                sinks_wires=[2, 3],
                                params=p)
        m = MeasureAll(obs={"Z1": 1})
        exp = m(q_machine=qm)
        print(exp)



QuantumLayerAdjoint
""""""""""""""""""""""""""""""""""""""""

.. py:class:: pyvqnet.qnn.vqc.torch.QuantumLayerAdjoint(general_module: pyvqnet.nn.Module, use_qpanda=False, name="")


    一个使用伴随矩阵方法计算梯度的自动微分 QuantumLayer 层，参见 `Efficient calculation of gradients in classical simulations of variational quantum algorithms <https://arxiv.org/abs/2009.02823>`_ 。

    :param general_module: 仅使用 ``pyvqnet.qnn.vqc.torch`` 下的量子电路接口构建的 `pyvqnet.nn.Module` 实例。
    :param use_qpanda: 是否使用 qpanda 线路进行前向传输，默认值：False。
    :param name: 层的名称，默认为 ""。

    .. note::

        general_module 的 QMachine 应将 grad_method 设置为 "adjoint"。

        目前支持以下参数化逻辑门 `RX`\ 、`RY`\ 、`RZ`\ 、`PhaseShift`\ 、`RXX`\ 、`RYY`\ 、`RZZ`\ 、`RZX`\ 、`U1`\ 、`U2`\ 、`U3` 以及由非参数逻辑门组成的其他变分电路。


    示例::

        import pyvqnet
        pyvqnet.backends.set_backend("torch")
        from pyvqnet import tensor
        from pyvqnet.qnn.vqc.torch import QuantumLayerAdjoint, \
            QMachine, RX, RY, CNOT, T, \
                MeasureAll, RZ, VQC_HardwareEfficientAnsatz,\
                    QModule

        class QModel(QModule):
            def __init__(self, num_wires, dtype, grad_mode=""):
                super(QModel, self).__init__()

                self._num_wires = num_wires
                self._dtype = dtype
                self.qm = QMachine(num_wires, dtype=dtype, grad_mode=grad_mode)
                self.rx_layer = RX(has_params=True, trainable=False, wires=0)
                self.ry_layer = RY(has_params=True, trainable=False, wires=1)
                self.rz_layer = RZ(has_params=True, trainable=False, wires=1)
                self.rz_layer2 = RZ(has_params=True, trainable=True, wires=1)

                self.rot = VQC_HardwareEfficientAnsatz(6, ["rx", "RY", "rz"],
                                                    entangle_gate="cnot",
                                                    entangle_rules="linear",
                                                    depth=5)
                self.tlayer = T(wires=1)
                self.cnot = CNOT(wires=[0, 1])
                self.measure = MeasureAll(obs={
                    "X1":1
                })

            def forward(self, x, *args, **kwargs):
                self.qm.reset_states(x.shape[0])

                self.rx_layer(params=x[:, [0]], q_machine=self.qm)
                self.cnot(q_machine=self.qm)
                self.ry_layer(params=x[:, [1]], q_machine=self.qm)
                self.tlayer(q_machine=self.qm)
                self.rz_layer(params=x[:, [2]], q_machine=self.qm)
                self.rz_layer2(q_machine=self.qm)
                self.rot(q_machine=self.qm)
                rlt = self.measure(q_machine=self.qm)

                return rlt


        input_x = tensor.QTensor([[0.1, 0.2, 0.3]])
        input_x = tensor.broadcast_to(input_x, [40, 3])
        input_x.requires_grad = True
        qunatum_model = QModel(num_wires=6,
                            dtype=pyvqnet.kcomplex64,
                            grad_mode="adjoint")
        adjoint_model = QuantumLayerAdjoint(qunatum_model, qunatum_model.qm)
        batch_y = adjoint_model(input_x)
        batch_y.backward()





张量网络后端变分量子电路模块
==========================================================================================

张量网络（TN）通过将复杂张量分解为多个低维张量的网络，显著降低了计算复杂度。

矩阵乘积态（MPS）是张量网络的一种特殊形式。MPS 将量子态表示为一系列矩阵的乘积，从而有效减少了参数数量和计算复杂度。

以下接口基于 ``torch`` 后端，提供在张量网络中构建量子电路的功能支持，包括量子电路基类、量子逻辑门、量子电路和测量的构建，以及通过自动微分模拟而非参数漂移方法计算参数梯度。

以 MPS 方式构建量子线路弥补了对大比特量子线路构建的支持。

.. warning::

        使用本模块中的以下功能需要额外安装 ``tensornetwork`` 和 ``torch``\ 。``pyvqnet`` 的默认安装不包含这两个依赖项。请使用 ``pip install tensornetwork torch`` 进行安装。

.. warning::

        通过 ``TNQMachine`` 中的 ``use_mps`` 参数启用 MPS 构建量子线路，支持大比特（100 及以上）量子线路实现。

.. warning::

        批处理的使用方式与经典模块下不同，基于 vmap 方式，数据和参数构建线路需要低一维输入，如下方示例接口所示，且批处理执行必须同时基于 ``TNQMachine`` 和 ``TNQModule``\ 。

基类
------------------------------------------------

在张量网络上编写变分量子电路模型需要继承 ``TNQModule``\ 。

TNQModule
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. py:class:: pyvqnet.qnn.vqc.tn.TNQModule(use_jit=False, vectorized_argnums=0, name="")

    .. note::

        此类及其派生类仅适用于 ``pyvqnet.backends.set_backend("torch")``\ ，不要与默认 ``pyvqnet.nn`` 下的 ``Module`` 混用。

        此类的 ``_buffers`` 中的数据为 ``torch.Tensor`` 类型。

        此类的 ``_parmeters`` 中的数据为 ``torch.nn.Parameter`` 类型。

    :param use_jit: 控制量子电路 jit 编译功能。
    :param vectorized_argnums: 要向量化的参数，
            这些参数应在第一维共享相同的批处理形状，默认为 0。
    :param name: 模块的名称。

    示例::

        import pyvqnet
        from pyvqnet.nn import Parameter
        pyvqnet.backends.set_backend("torch")
        from pyvqnet.qnn.vqc.tn import TNQModule
        from pyvqnet.qnn.vqc.tn import TNQMachine, RX, RY, CNOT, PauliX, PauliZ,qmeasure,qcircuit,VQC_RotCircuit
        class QModel(TNQModule):
            def __init__(self, num_wires, dtype,batch_size=2):
                super(QModel, self).__init__()

                self._num_wires = num_wires
                self._dtype = dtype
                self.qm = TNQMachine(num_wires, dtype=dtype)

                self.w = Parameter((2,4,3),initializer=pyvqnet.utils.initializer.quantum_uniform)
                self.cnot = CNOT(wires=[0, 1])
                self.batch_size = batch_size
            def forward(self, x, *args, **kwargs):
                self.qm.reset_states(batchsize=self.batch_size)

                def get_cnot(nqubits,qm):
                    for i in range(len(nqubits) - 1):
                        CNOT(wires = [nqubits[i], nqubits[i + 1]])(q_machine = qm)
                    CNOT(wires = [nqubits[len(nqubits) - 1], nqubits[0]])(q_machine = qm)


                def build_circuit(weights, xx, nqubits,qm):
                    def Rot(weights_j, nqubits,qm):#pylint:disable=invalid-name
                        VQC_RotCircuit(qm,nqubits,weights_j)

                    def basisstate(qm,xx, nqubits):
                        for i in nqubits:
                            qcircuit.rz(q_machine=qm, wires=i, params=xx[i])
                            qcircuit.ry(q_machine=qm, wires=i, params=xx[i])
                            qcircuit.rz(q_machine=qm, wires=i, params=xx[i])

                    basisstate(qm,xx,nqubits)

                    for i in range(weights.shape[0]):

                        weights_i = weights[i, :, :]
                        for j in range(len(nqubits)):
                            weights_j = weights_i[j]
                            Rot(weights_j, nqubits[j],qm)
                        get_cnot(nqubits,qm)

                build_circuit(self.w, x,range(4),self.qm)

                y= qmeasure.MeasureAll(obs={'Z0': 1})(self.qm)
                return y


        x= pyvqnet.tensor.QTensor([[1,0,0,1],[1,1,0,1]],dtype=pyvqnet.kfloat32)
        model = QModel(4,pyvqnet.kcomplex64,2)
        y = model(x)
        y.backward()


TNQMachine
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. py:class:: pyvqnet.qnn.vqc.tn.TNQMachine(num_wires, dtype=pyvqnet.kcomplex64, use_mps=False)

    变分量子计算的模拟器类，包含状态向量，其 states 属性为量子电路。

    此类继承自 ``pyvqnet.nn.torch.TorchModule`` 和 ``pyvqnet.qnn.QMachine``\ 。

    此类可作为子模块添加到 torch 模型中。

    .. warning::

        在张量网络的量子电路中，``vmap`` 函数将默认启用，线路上的逻辑门参数中将丢弃批处理维度。
        使用调用参数时，如果维度为 [batch_size, \*]，则丢弃第一个 batch_size 维度，直接使用后续维度，例如，对于输入数据 x[:,1] -> x[1]，对于可训练参数也一样，参见以下示例中 xx、weights 的使用方式。

    .. note::

        在每次运行完整量子电路之前，必须使用 `pyvqnet.qnn.vqc.QMachine.reset_states(batchsize)` 重新初始化模拟器中的初始状态，并将其广播到
        (batchsize,*) 维度，以适应批数据训练。

    :param num_wires: 使用的量子比特数量
    :param dtype: 用于计算的内部数据类型。
    :param use_mps: 为大比特模型打开 MPSCircuit。

    :return: 输出一个 TNQMachine 对象。

    示例::

        import pyvqnet
        from pyvqnet.nn import Parameter
        pyvqnet.backends.set_backend("torch")
        from pyvqnet.qnn.vqc.tn import TNQModule
        from pyvqnet.qnn.vqc.tn import TNQMachine, RX, RY, CNOT, PauliX, PauliZ,qmeasure,qcircuit,VQC_RotCircuit
        class QModel(TNQModule):
            def __init__(self, num_wires, dtype,batch_size=2):
                super(QModel, self).__init__()

                self._num_wires = num_wires
                self._dtype = dtype
                self.qm = TNQMachine(num_wires, dtype=dtype)

                self.w = Parameter((2,4,3),initializer=pyvqnet.utils.initializer.quantum_uniform)
                self.cnot = CNOT(wires=[0, 1])
                self.batch_size = batch_size
            def forward(self, x, *args, **kwargs):
                self.qm.reset_states(batchsize=self.batch_size)

                def get_cnot(nqubits,qm):
                    for i in range(len(nqubits) - 1):
                        CNOT(wires = [nqubits[i], nqubits[i + 1]])(q_machine = qm)
                    CNOT(wires = [nqubits[len(nqubits) - 1], nqubits[0]])(q_machine = qm)


                def build_circuit(weights, xx, nqubits,qm):
                    def Rot(weights_j, nqubits,qm):#pylint:disable=invalid-name
                        VQC_RotCircuit(qm,nqubits,weights_j)

                    def basisstate(qm,xx, nqubits):
                        for i in nqubits:
                            qcircuit.rz(q_machine=qm, wires=i, params=xx[i])
                            qcircuit.ry(q_machine=qm, wires=i, params=xx[i])
                            qcircuit.rz(q_machine=qm, wires=i, params=xx[i])

                    basisstate(qm,xx,nqubits)

                    for i in range(weights.shape[0]):

                        weights_i = weights[i, :, :]
                        for j in range(len(nqubits)):
                            weights_j = weights_i[j]
                            Rot(weights_j, nqubits[j],qm)
                        get_cnot(nqubits,qm)

                build_circuit(self.w, x,range(4),self.qm)

                y= qmeasure.MeasureAll(obs={'Z0': 1})(self.qm)
                return y


        x= pyvqnet.tensor.QTensor([[1,0,0,1],[1,1,0,1]],dtype=pyvqnet.kfloat32)
        model = QModel(4,pyvqnet.kcomplex64,2)
        y = model(x)
        y.backward()

    .. py:method:: get_states()

        获取张量网络量子机器的状态。

变分量子逻辑门模块
------------------------------------------------

以下 ``pyvqnet.qnn.vqc`` 中的函数接口直接支持 ``torch`` 后端的 ``QTensor`` 进行计算，导入路径 ``pyvqnet.qnn.vqc.tn``\ 。

.. csv-table:: 支持的 pyvqnet.qnn.vqc 接口列表
    :file: ./images/same_apis_from_tn.csv

以下量子电路模块继承自 ``pyvqnet.qnn.vqc.tn.TNQModule``\ ，其中计算使用 ``torch.Tensor`` 进行。

.. note::

    此类及其派生类仅适用于 ``pyvqnet.backends.set_backend("torch")``\ ，不要与默认 ``pyvqnet.nn`` 下的 ``Module`` 混用。

    如果这些类有非参数成员变量 ``_buffers``\ ，其中的数据为 ``torch.Tensor`` 类型。
    如果这些类有参数成员变量 ``_parmeters``\ ，其中的数据为 ``torch.nn.Parameter`` 类型。

I
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. py:class:: pyvqnet.qnn.vqc.tn.I(has_params: bool = False,trainable: bool = False,init_params=None,wires=None,dtype=pyvqnet.kcomplex64,use_dagger=False)

    定义一个 I 量子门。

    此类继承自 ``pyvqnet.qnn.vqc.tn.QModule`` 和 ``torch.nn.Module``\ 。
    此类可作为子模块添加到 torch 模型中。

    :param has_params: 是否有参数，如 RX、RY 等门需要设置为 True，无参数的门需要设置为 False，默认为 False。
    :param trainable: 是否有参数需要训练。如果该层使用外部输入数据构建逻辑门矩阵，则设置为 False。如果需要从该层初始化待训练参数，则为 True，默认为 False。
    :param init_params: 用于编码经典数据 QTensor 的初始化参数，默认为 None。
    :param wires: 线路作用的比特索引，默认为 None。
    :param dtype: 逻辑门内部矩阵的数据精度，可设置为 pyvqnet.kcomplex64 或 pyvqnet.kcomplex128，分别对应 float 输入或 double 输入。
    :param use_dagger: 是否使用门的转置共轭版本，默认为 False。
    :return: 一个 ``pyvqnet.qnn.vqc.tn.QModule`` 实例

    示例::

        from pyvqnet.qnn.vqc.tn import I,TNQMachine,TNQModule,MeasureAll, rx
        import pyvqnet
        pyvqnet.backends.set_backend("torch")

        class QModel(TNQModule):

            def __init__(self, num_wires, dtype,batch_size=2):
                super(QModel, self).__init__()
                self.device = TNQMachine(num_wires)
                self.layer = I(wires=0)
                self.batch_size = batch_size
                self.num_wires = num_wires

            def forward(self, x, *args, **kwargs):
                self.device.reset_states(batchsize=self.batch_size)
                for i in range(self.num_wires):
                    rx(self.device, wires=i, params=x[i])
                self.layer(q_machine = self.device)
                y = MeasureAll(obs={'Z0': 1})(self.device)
                return y

        x = pyvqnet.tensor.QTensor([[1,0,0,1],[1,1,0,1]],dtype=pyvqnet.kfloat32,requires_grad=True)
        model = QModel(4,pyvqnet.kcomplex64,2)
        y = model(x)
        print(y)



Hadamard
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. py:class:: pyvqnet.qnn.vqc.tn.Hadamard(has_params: bool = False,trainable: bool = False,init_params=None,wires=None,dtype=pyvqnet.kcomplex64,use_dagger=False)

    定义一个 Hadamard 量子门。

    此类继承自 ``pyvqnet.qnn.vqc.tn.QModule`` 和 ``torch.nn.Module``\ 。
    此类可作为子模块添加到 torch 模型中。

    :param has_params: 是否有参数，如 RX、RY 等门需要设置为 True，无参数的门需要设置为 False，默认为 False。
    :param trainable: 是否有参数需要训练。如果该层使用外部输入数据构建逻辑门矩阵，则设置为 False。如果需要从该层初始化待训练参数，则为 True，默认为 False。
    :param init_params: 用于编码经典数据 QTensor 的初始化参数，默认为 None。
    :param wires: 线路作用的比特索引，默认为 None。
    :param dtype: 逻辑门内部矩阵的数据精度，可设置为 pyvqnet.kcomplex64 或 pyvqnet.kcomplex128，分别对应 float 输入或 double 输入。
    :param use_dagger: 是否使用门的转置共轭版本，默认为 False。
    :return: 一个 ``pyvqnet.qnn.vqc.tn.QModule`` 实例

    示例::

        from pyvqnet.qnn.vqc.tn import Hadamard,TNQMachine,TNQModule,MeasureAll, rx
        import pyvqnet
        pyvqnet.backends.set_backend("torch")

        class QModel(TNQModule):

            def __init__(self, num_wires, dtype,batch_size=2):
                super(QModel, self).__init__()
                self.device = TNQMachine(num_wires)
                self.layer = Hadamard(wires=0)
                self.batch_size = batch_size
                self.num_wires = num_wires

            def forward(self, x, *args, **kwargs):
                self.device.reset_states(batchsize=self.batch_size)
                for i in range(self.num_wires):
                    rx(self.device, wires=i, params=x[i])
                self.layer(q_machine = self.device)
                y = MeasureAll(obs={'Z0': 1})(self.device)
                return y

        x = pyvqnet.tensor.QTensor([[1,0,0,1],[1,1,0,1]],dtype=pyvqnet.kfloat32,requires_grad=True)
        model = QModel(4,pyvqnet.kcomplex64,2)
        y = model(x)
        print(y)


T
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. py:class:: pyvqnet.qnn.vqc.tn.T(has_params: bool = False,trainable: bool = False,init_params=None,wires=None,dtype=pyvqnet.kcomplex64,use_dagger=False)

    定义一个 T 量子门。

    此类继承自 ``pyvqnet.qnn.vqc.tn.QModule`` 和 ``torch.nn.Module``\ 。
    此类可作为子模块添加到 torch 模型中。

    :param has_params: 是否有参数，如 RX、RY 等门需要设置为 True，无参数的门需要设置为 False，默认为 False。
    :param trainable: 是否有参数需要训练。如果该层使用外部输入数据构建逻辑门矩阵，则设置为 False。如果需要从该层初始化待训练参数，则为 True，默认为 False。
    :param init_params: 用于编码经典数据 QTensor 的初始化参数，默认为 None。
    :param wires: 线路作用的比特索引，默认为 None。
    :param dtype: 逻辑门内部矩阵的数据精度，可设置为 pyvqnet.kcomplex64 或 pyvqnet.kcomplex128，分别对应 float 输入或 double 输入。
    :param use_dagger: 是否使用门的转置共轭版本，默认为 False。
    :return: 一个 ``pyvqnet.qnn.vqc.tn.QModule`` 实例

    示例::

        from pyvqnet.qnn.vqc.tn import T,TNQMachine,TNQModule,MeasureAll, rx
        import pyvqnet
        pyvqnet.backends.set_backend("torch")

        class QModel(TNQModule):

            def __init__(self, num_wires, dtype,batch_size=2):
                super(QModel, self).__init__()
                self.device = TNQMachine(num_wires)
                self.layer = T(wires=0)
                self.batch_size = batch_size
                self.num_wires = num_wires

            def forward(self, x, *args, **kwargs):
                self.device.reset_states(batchsize=self.batch_size)
                for i in range(self.num_wires):
                    rx(self.device, wires=i, params=x[i])
                self.layer(q_machine = self.device)
                y = MeasureAll(obs={'Z0': 1})(self.device)
                return y

        x = pyvqnet.tensor.QTensor([[1,0,0,1],[1,1,0,1]],dtype=pyvqnet.kfloat32,requires_grad=True)
        model = QModel(4,pyvqnet.kcomplex64,2)
        y = model(x)
        print(y)



S
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. py:class:: pyvqnet.qnn.vqc.tn.S(has_params: bool = False,trainable: bool = False,init_params=None,wires=None,dtype=pyvqnet.kcomplex64,use_dagger=False)

    定义一个 S 量子门。

    此类继承自 ``pyvqnet.qnn.vqc.tn.QModule`` 和 ``torch.nn.Module``\ 。
    此类可作为子模块添加到 torch 模型中。

    :param has_params: 是否有参数，如 RX、RY 等门需要设置为 True，无参数的门需要设置为 False，默认为 False。
    :param trainable: 是否有参数需要训练。如果该层使用外部输入数据构建逻辑门矩阵，则设置为 False。如果需要从该层初始化待训练参数，则为 True，默认为 False。
    :param init_params: 用于编码经典数据 QTensor 的初始化参数，默认为 None。
    :param wires: 线路作用的比特索引，默认为 None。
    :param dtype: 逻辑门内部矩阵的数据精度，可设置为 pyvqnet.kcomplex64 或 pyvqnet.kcomplex128，分别对应 float 输入或 double 输入。
    :param use_dagger: 是否使用门的转置共轭版本，默认为 False。
    :return: 一个 ``pyvqnet.qnn.vqc.tn.QModule`` 实例

    示例::

        from pyvqnet.qnn.vqc.tn import S,TNQMachine,TNQModule,MeasureAll, rx
        import pyvqnet
        pyvqnet.backends.set_backend("torch")

        class QModel(TNQModule):

            def __init__(self, num_wires, dtype,batch_size=2):
                super(QModel, self).__init__()
                self.device = TNQMachine(num_wires)
                self.layer = S(wires=0)
                self.batch_size = batch_size
                self.num_wires = num_wires

            def forward(self, x, *args, **kwargs):
                self.device.reset_states(batchsize=self.batch_size)
                for i in range(self.num_wires):
                    rx(self.device, wires=i, params=x[i])
                self.layer(q_machine = self.device)
                y = MeasureAll(obs={'Z0': 1})(self.device)
                return y

        x = pyvqnet.tensor.QTensor([[1,0,0,1],[1,1,0,1]],dtype=pyvqnet.kfloat32,requires_grad=True)
        model = QModel(4,pyvqnet.kcomplex64,2)
        y = model(x)
        print(y)


PauliX
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. py:class:: pyvqnet.qnn.vqc.tn.PauliX(has_params: bool = False,trainable: bool = False,init_params=None,wires=None,dtype=pyvqnet.kcomplex64,use_dagger=False)

    定义一个 PauliX 量子门。

    此类继承自 ``pyvqnet.qnn.vqc.tn.QModule`` 和 ``torch.nn.Module``\ 。
    此类可作为子模块添加到 torch 模型中。


    :param has_params: 是否有参数，如 RX、RY 等门需要设置为 True，无参数的门需要设置为 False，默认为 False。
    :param trainable: 是否有参数需要训练。如果该层使用外部输入数据构建逻辑门矩阵，则设置为 False。如果需要从该层初始化待训练参数，则为 True，默认为 False。
    :param init_params: 用于编码经典数据 QTensor 的初始化参数，默认为 None。
    :param wires: 线路作用的比特索引，默认为 None。
    :param dtype: 逻辑门内部矩阵的数据精度，可设置为 pyvqnet.kcomplex64 或 pyvqnet.kcomplex128，分别对应 float 输入或 double 输入。
    :param use_dagger: 是否使用门的转置共轭版本，默认为 False。
    :return: 一个 ``pyvqnet.qnn.vqc.tn.QModule`` 实例

    示例::

        from pyvqnet.qnn.vqc.tn import PauliX,TNQMachine,TNQModule,MeasureAll, rx
        import pyvqnet
        pyvqnet.backends.set_backend("torch")

        class QModel(TNQModule):

            def __init__(self, num_wires, dtype,batch_size=2):
                super(QModel, self).__init__()
                self.device = TNQMachine(num_wires)
                self.layer = PauliX(wires=0)
                self.batch_size = batch_size
                self.num_wires = num_wires

            def forward(self, x, *args, **kwargs):
                self.device.reset_states(batchsize=self.batch_size)
                for i in range(self.num_wires):
                    rx(self.device, wires=i, params=x[i])
                self.layer(q_machine = self.device)
                y = MeasureAll(obs={'Z0': 1})(self.device)
                return y

        x = pyvqnet.tensor.QTensor([[1,0,0,1],[1,1,0,1]],dtype=pyvqnet.kfloat32,requires_grad=True)
        model = QModel(4,pyvqnet.kcomplex64,2)
        y = model(x)
        print(y)



PauliY
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. py:class:: pyvqnet.qnn.vqc.tn.PauliY(has_params: bool = False,trainable: bool = False,init_params=None,wires=None,dtype=pyvqnet.kcomplex64,use_dagger=False)

    定义一个 PauliY 量子门。

    此类继承自 ``pyvqnet.qnn.vqc.tn.QModule`` 和 ``torch.nn.Module``\ 。
    此类可作为子模块添加到 torch 模型中。


    :param has_params: 是否有参数，如 RX、RY 等门需要设置为 True，无参数的门需要设置为 False，默认为 False。
    :param trainable: 是否有参数需要训练。如果该层使用外部输入数据构建逻辑门矩阵，则设置为 False。如果需要从该层初始化待训练参数，则为 True，默认为 False。
    :param init_params: 用于编码经典数据 QTensor 的初始化参数，默认为 None。
    :param wires: 线路作用的比特索引，默认为 None。
    :param dtype: 逻辑门内部矩阵的数据精度，可设置为 pyvqnet.kcomplex64 或 pyvqnet.kcomplex128，分别对应 float 输入或 double 输入。
    :param use_dagger: 是否使用门的转置共轭版本，默认为 False。
    :return: 一个 ``pyvqnet.qnn.vqc.tn.QModule`` 实例

    示例::

        from pyvqnet.qnn.vqc.tn import PauliY,TNQMachine,TNQModule,MeasureAll, rx
        import pyvqnet
        pyvqnet.backends.set_backend("torch")

        class QModel(TNQModule):

            def __init__(self, num_wires, dtype,batch_size=2):
                super(QModel, self).__init__()
                self.device = TNQMachine(num_wires)
                self.layer = PauliY(wires=0)
                self.batch_size = batch_size
                self.num_wires = num_wires

            def forward(self, x, *args, **kwargs):
                self.device.reset_states(batchsize=self.batch_size)
                for i in range(self.num_wires):
                    rx(self.device, wires=i, params=x[i])
                self.layer(q_machine = self.device)
                y = MeasureAll(obs={'Z0': 1})(self.device)
                return y

        x = pyvqnet.tensor.QTensor([[1,0,0,1],[1,1,0,1]],dtype=pyvqnet.kfloat32,requires_grad=True)
        model = QModel(4,pyvqnet.kcomplex64,2)
        y = model(x)
        print(y)


PauliZ
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. py:class:: pyvqnet.qnn.vqc.tn.PauliZ(has_params: bool = False,trainable: bool = False,init_params=None,wires=None,dtype=pyvqnet.kcomplex64,use_dagger=False)

    定义一个 PauliZ 量子门。

    此类继承自 ``pyvqnet.qnn.vqc.tn.QModule`` 和 ``torch.nn.Module``\ 。
    此类可作为子模块添加到 torch 模型中。


    :param has_params: 是否有参数，如 RX、RY 等门需要设置为 True，无参数的门需要设置为 False，默认为 False。
    :param trainable: 是否有参数需要训练。如果该层使用外部输入数据构建逻辑门矩阵，则设置为 False。如果需要从该层初始化待训练参数，则为 True，默认为 False。
    :param init_params: 用于编码经典数据 QTensor 的初始化参数，默认为 None。
    :param wires: 线路作用的比特索引，默认为 None。
    :param dtype: 逻辑门内部矩阵的数据精度，可设置为 pyvqnet.kcomplex64 或 pyvqnet.kcomplex128，分别对应 float 输入或 double 输入。
    :param use_dagger: 是否使用门的转置共轭版本，默认为 False。
    :return: 一个 ``pyvqnet.qnn.vqc.tn.QModule`` 实例

    示例::

        from pyvqnet.qnn.vqc.tn import PauliZ,TNQMachine,TNQModule,MeasureAll, rx
        import pyvqnet
        pyvqnet.backends.set_backend("torch")

        class QModel(TNQModule):

            def __init__(self, num_wires, dtype,batch_size=2):
                super(QModel, self).__init__()
                self.device = TNQMachine(num_wires)
                self.layer = PauliZ(wires=0)
                self.batch_size = batch_size
                self.num_wires = num_wires

            def forward(self, x, *args, **kwargs):
                self.device.reset_states(batchsize=self.batch_size)
                for i in range(self.num_wires):
                    rx(self.device, wires=i, params=x[i])
                self.layer(q_machine = self.device)
                y = MeasureAll(obs={'Z0': 1})(self.device)
                return y

        x = pyvqnet.tensor.QTensor([[1,0,0,1],[1,1,0,1]],dtype=pyvqnet.kfloat32,requires_grad=True)
        model = QModel(4,pyvqnet.kcomplex64,2)
        y = model(x)
        print(y)




X1
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. py:class:: pyvqnet.qnn.vqc.tn.X1(has_params: bool = False,trainable: bool = False,init_params=None,wires=None,dtype=pyvqnet.kcomplex64,use_dagger=False)

    定义一个 X1 量子门。

    此类继承自 ``pyvqnet.qnn.vqc.tn.QModule`` 和 ``torch.nn.Module``\ 。
    此类可作为子模块添加到 torch 模型中。

    :param has_params: 是否有参数，如 RX、RY 等门需要设置为 True，无参数的门需要设置为 False，默认为 False。
    :param trainable: 是否有参数需要训练。如果该层使用外部输入数据构建逻辑门矩阵，则设置为 False。如果需要从该层初始化待训练参数，则为 True，默认为 False。
    :param init_params: 用于编码经典数据 QTensor 的初始化参数，默认为 None。
    :param wires: 线路作用的比特索引，默认为 None。
    :param dtype: 逻辑门内部矩阵的数据精度，可设置为 pyvqnet.kcomplex64 或 pyvqnet.kcomplex128，分别对应 float 输入或 double 输入。
    :param use_dagger: 是否使用门的转置共轭版本，默认为 False。
    :return: 一个 ``pyvqnet.qnn.vqc.tn.QModule`` 实例

    示例::

        from pyvqnet.qnn.vqc.tn import X1,TNQMachine,TNQModule,MeasureAll, rx
        import pyvqnet
        pyvqnet.backends.set_backend("torch")

        class QModel(TNQModule):

            def __init__(self, num_wires, dtype,batch_size=2):
                super(QModel, self).__init__()
                self.device = TNQMachine(num_wires)
                self.layer = X1(wires=0)
                self.batch_size = batch_size
                self.num_wires = num_wires

            def forward(self, x, *args, **kwargs):
                self.device.reset_states(batchsize=self.batch_size)
                for i in range(self.num_wires):
                    rx(self.device, wires=i, params=x[i])
                self.layer(q_machine = self.device)
                y = MeasureAll(obs={'Z0': 1})(self.device)
                return y

        x = pyvqnet.tensor.QTensor([[1,0,0,1],[1,1,0,1]],dtype=pyvqnet.kfloat32,requires_grad=True)
        model = QModel(4,pyvqnet.kcomplex64,2)
        y = model(x)
        print(y)



RX
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. py:class:: pyvqnet.qnn.vqc.tn.RX(has_params: bool = False,trainable: bool = False,init_params=None,wires=None,dtype=pyvqnet.kcomplex64,use_dagger=False)

    定义一个 RX 量子门。

    此类继承自 ``pyvqnet.qnn.vqc.tn.QModule`` 和 ``torch.nn.Module``\ 。
    此类可作为子模块添加到 torch 模型中。

    :param has_params: 是否有参数，如 RX、RY 等门需要设置为 True，无参数的门需要设置为 False，默认为 False。
    :param trainable: 是否有参数需要训练。如果该层使用外部输入数据构建逻辑门矩阵，则设置为 False。如果需要从该层初始化待训练参数，则为 True，默认为 False。
    :param init_params: 用于编码经典数据 QTensor 的初始化参数，默认为 None。
    :param wires: 线路作用的比特索引，默认为 None。
    :param dtype: 逻辑门内部矩阵的数据精度，可设置为 pyvqnet.kcomplex64 或 pyvqnet.kcomplex128，分别对应 float 输入或 double 输入。
    :param use_dagger: 是否使用门的转置共轭版本，默认为 False。
    :return: 一个 ``pyvqnet.qnn.vqc.tn.QModule`` 实例

    示例::

        from pyvqnet.qnn.vqc.tn import RX,TNQMachine,TNQModule,MeasureAll, rx
        import pyvqnet
        pyvqnet.backends.set_backend("torch")

        class QModel(TNQModule):

            def __init__(self, num_wires, dtype,batch_size=2):
                super(QModel, self).__init__()
                self.device = TNQMachine(num_wires)
                self.layer = RX(wires=0,has_params=True)
                self.batch_size = batch_size
                self.num_wires = num_wires

            def forward(self, x, *args, **kwargs):
                self.device.reset_states(batchsize=self.batch_size)
                for i in range(self.num_wires):
                    rx(self.device, wires=i, params=x[i])
                self.layer(q_machine = self.device)
                y = MeasureAll(obs={'Z0': 1})(self.device)
                return y

        x = pyvqnet.tensor.QTensor([[1,0,0,1],[1,1,0,1]],dtype=pyvqnet.kfloat32,requires_grad=True)
        model = QModel(4,pyvqnet.kcomplex64,2)
        y = model(x)
        print(y)





RY
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. py:class:: pyvqnet.qnn.vqc.tn.RY(has_params: bool = False,trainable: bool = False,init_params=None,wires=None,dtype=pyvqnet.kcomplex64,use_dagger=False)

    定义一个 RY 量子门。

    此类继承自 ``pyvqnet.qnn.vqc.tn.QModule`` 和 ``torch.nn.Module``\ 。
    此类可作为子模块添加到 torch 模型中。

    :param has_params: 是否有参数，如 RX、RY 等门需要设置为 True，无参数的门需要设置为 False，默认为 False。
    :param trainable: 是否有参数需要训练。如果该层使用外部输入数据构建逻辑门矩阵，则设置为 False。如果需要从该层初始化待训练参数，则为 True，默认为 False。
    :param init_params: 用于编码经典数据 QTensor 的初始化参数，默认为 None。
    :param wires: 线路作用的比特索引，默认为 None。
    :param dtype: 逻辑门内部矩阵的数据精度，可设置为 pyvqnet.kcomplex64 或 pyvqnet.kcomplex128，分别对应 float 输入或 double 输入。
    :param use_dagger: 是否使用门的转置共轭版本，默认为 False。
    :return: 一个 ``pyvqnet.qnn.vqc.tn.QModule`` 实例

    示例::

        from pyvqnet.qnn.vqc.tn import RY,TNQMachine,TNQModule,MeasureAll, rx
        import pyvqnet
        pyvqnet.backends.set_backend("torch")

        class QModel(TNQModule):

            def __init__(self, num_wires, dtype,batch_size=2):
                super(QModel, self).__init__()
                self.device = TNQMachine(num_wires)
                self.layer = RY(wires=0,has_params=True)
                self.batch_size = batch_size
                self.num_wires = num_wires

            def forward(self, x, *args, **kwargs):
                self.device.reset_states(batchsize=self.batch_size)
                for i in range(self.num_wires):
                    rx(self.device, wires=i, params=x[i])
                self.layer(q_machine = self.device)
                y = MeasureAll(obs={'Z0': 1})(self.device)
                return y

        x = pyvqnet.tensor.QTensor([[1,0,0,1],[1,1,0,1]],dtype=pyvqnet.kfloat32,requires_grad=True)
        model = QModel(4,pyvqnet.kcomplex64,2)
        y = model(x)
        print(y)


RZ
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. py:class:: pyvqnet.qnn.vqc.tn.RZ(has_params: bool = False,trainable: bool = False,init_params=None,wires=None,dtype=pyvqnet.kcomplex64,use_dagger=False)

    定义一个 RZ 量子门。

    此类继承自 ``pyvqnet.qnn.vqc.tn.QModule`` 和 ``torch.nn.Module``\ 。
    此类可作为子模块添加到 torch 模型中。

    :param has_params: 是否有参数，如 RX、RY 等门需要设置为 True，无参数的门需要设置为 False，默认为 False。
    :param trainable: 是否有参数需要训练。如果该层使用外部输入数据构建逻辑门矩阵，则设置为 False。如果需要从该层初始化待训练参数，则为 True，默认为 False。
    :param init_params: 用于编码经典数据 QTensor 的初始化参数，默认为 None。
    :param wires: 线路作用的比特索引，默认为 None。
    :param dtype: 逻辑门内部矩阵的数据精度，可设置为 pyvqnet.kcomplex64 或 pyvqnet.kcomplex128，分别对应 float 输入或 double 输入。
    :param use_dagger: 是否使用门的转置共轭版本，默认为 False。
    :return: 一个 ``pyvqnet.qnn.vqc.tn.QModule`` 实例

    示例::

        from pyvqnet.qnn.vqc.tn import RZ,TNQMachine,TNQModule,MeasureAll, rx
        import pyvqnet
        pyvqnet.backends.set_backend("torch")

        class QModel(TNQModule):

            def __init__(self, num_wires, dtype,batch_size=2):
                super(QModel, self).__init__()
                self.device = TNQMachine(num_wires)
                self.layer = RZ(wires=0,has_params=True)
                self.batch_size = batch_size
                self.num_wires = num_wires

            def forward(self, x, *args, **kwargs):
                self.device.reset_states(batchsize=self.batch_size)
                for i in range(self.num_wires):
                    rx(self.device, wires=i, params=x[i])
                self.layer(q_machine = self.device)
                y = MeasureAll(obs={'Z0': 1})(self.device)
                return y

        x = pyvqnet.tensor.QTensor([[1,0,0,1],[1,1,0,1]],dtype=pyvqnet.kfloat32,requires_grad=True)
        model = QModel(4,pyvqnet.kcomplex64,2)
        y = model(x)
        print(y)


CRX
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. py:class:: pyvqnet.qnn.vqc.tn.CRX(has_params: bool = False,trainable: bool = False,init_params=None,wires=None,dtype=pyvqnet.kcomplex64,use_dagger=False)

    定义一个 CRX 量子门。

    此类继承自 ``pyvqnet.qnn.vqc.tn.QModule`` 和 ``torch.nn.Module``\ 。
    此类可作为子模块添加到 torch 模型中。

    :param has_params: 是否有参数，如 RX、RY 等门需要设置为 True，无参数的门需要设置为 False，默认为 False。
    :param trainable: 是否有参数需要训练。如果该层使用外部输入数据构建逻辑门矩阵，则设置为 False。如果需要从该层初始化待训练参数，则为 True，默认为 False。
    :param init_params: 用于编码经典数据 QTensor 的初始化参数，默认为 None。
    :param wires: 线路作用的比特索引，默认为 None。
    :param dtype: 逻辑门内部矩阵的数据精度，可设置为 pyvqnet.kcomplex64 或 pyvqnet.kcomplex128，分别对应 float 输入或 double 输入。
    :param use_dagger: 是否使用门的转置共轭版本，默认为 False。
    :return: 一个 ``pyvqnet.qnn.vqc.tn.QModule`` 实例

    示例::

        from pyvqnet.qnn.vqc.tn import CRX,TNQMachine,TNQModule,MeasureAll, rx
        import pyvqnet
        pyvqnet.backends.set_backend("torch")

        class QModel(TNQModule):

            def __init__(self, num_wires, dtype,batch_size=2):
                super(QModel, self).__init__()
                self.device = TNQMachine(num_wires)
                self.layer = CRX(has_params= True, trainable= True, wires=[0,2])
                self.batch_size = batch_size
                self.num_wires = num_wires

            def forward(self, x, *args, **kwargs):
                self.device.reset_states(batchsize=self.batch_size)
                for i in range(self.num_wires):
                    rx(self.device, wires=i, params=x[i])
                self.layer(q_machine = self.device)
                y = MeasureAll(obs={'Z0': 1})(self.device)
                return y

        x = pyvqnet.tensor.QTensor([[1,0,0,1],[1,1,0,1]],dtype=pyvqnet.kfloat32,requires_grad=True)
        model = QModel(4,pyvqnet.kcomplex64,2)
        y = model(x)
        print(y)




CRY
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^


.. py:class:: pyvqnet.qnn.vqc.tn.CRY(has_params: bool = False,trainable: bool = False,init_params=None,wires=None,dtype=pyvqnet.kcomplex64,use_dagger=False)

    定义一个 CRY 量子门。

    此类继承自 ``pyvqnet.qnn.vqc.tn.QModule`` 和 ``torch.nn.Module``\ 。
    此类可作为子模块添加到 torch 模型中。

    :param has_params: 是否有参数，如 RX、RY 等门需要设置为 True，无参数的门需要设置为 False，默认为 False。
    :param trainable: 是否有参数需要训练。如果该层使用外部输入数据构建逻辑门矩阵，则设置为 False。如果需要从该层初始化待训练参数，则为 True，默认为 False。
    :param init_params: 用于编码经典数据 QTensor 的初始化参数，默认为 None。
    :param wires: 线路作用的比特索引，默认为 None。
    :param dtype: 逻辑门内部矩阵的数据精度，可设置为 pyvqnet.kcomplex64 或 pyvqnet.kcomplex128，分别对应 float 输入或 double 输入。
    :param use_dagger: 是否使用门的转置共轭版本，默认为 False。
    :return: 一个 ``pyvqnet.qnn.vqc.tn.QModule`` 实例

    示例::

        from pyvqnet.qnn.vqc.tn import CRY,TNQMachine,TNQModule,MeasureAll, rx
        import pyvqnet
        pyvqnet.backends.set_backend("torch")

        class QModel(TNQModule):

            def __init__(self, num_wires, dtype,batch_size=2):
                super(QModel, self).__init__()
                self.device = TNQMachine(num_wires)
                self.layer = CRY(has_params= True, trainable= True, wires=[0,2])
                self.batch_size = batch_size
                self.num_wires = num_wires

            def forward(self, x, *args, **kwargs):
                self.device.reset_states(batchsize=self.batch_size)
                for i in range(self.num_wires):
                    rx(self.device, wires=i, params=x[i])
                self.layer(q_machine = self.device)
                y = MeasureAll(obs={'Z0': 1})(self.device)
                return y

        x = pyvqnet.tensor.QTensor([[1,0,0,1],[1,1,0,1]],dtype=pyvqnet.kfloat32,requires_grad=True)
        model = QModel(4,pyvqnet.kcomplex64,2)
        y = model(x)
        print(y)


CRZ
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^


.. py:class:: pyvqnet.qnn.vqc.tn.CRZ(has_params: bool = False,trainable: bool = False,init_params=None,wires=None,dtype=pyvqnet.kcomplex64,use_dagger=False)

    定义一个 CRZ 量子门。

    此类继承自 ``pyvqnet.qnn.vqc.tn.QModule`` 和 ``torch.nn.Module``\ 。
    此类可作为子模块添加到 torch 模型中。

    :param has_params: 是否有参数，如 RX、RY 等门需要设置为 True，无参数的门需要设置为 False，默认为 False。
    :param trainable: 是否有参数需要训练。如果该层使用外部输入数据构建逻辑门矩阵，则设置为 False。如果需要从该层初始化待训练参数，则为 True，默认为 False。
    :param init_params: 用于编码经典数据 QTensor 的初始化参数，默认为 None。
    :param wires: 线路作用的比特索引，默认为 None。
    :param dtype: 逻辑门内部矩阵的数据精度，可设置为 pyvqnet.kcomplex64 或 pyvqnet.kcomplex128，分别对应 float 输入或 double 输入。
    :param use_dagger: 是否使用门的转置共轭版本，默认为 False。
    :return: 一个 ``pyvqnet.qnn.vqc.tn.QModule`` 实例

    示例::

        from pyvqnet.qnn.vqc.tn import CRZ,TNQMachine,TNQModule,MeasureAll, rx
        import pyvqnet
        pyvqnet.backends.set_backend("torch")

        class QModel(TNQModule):

            def __init__(self, num_wires, dtype,batch_size=2):
                super(QModel, self).__init__()
                self.device = TNQMachine(num_wires)
                self.layer = CRZ(has_params= True, trainable= True, wires=[0,2])
                self.batch_size = batch_size
                self.num_wires = num_wires

            def forward(self, x, *args, **kwargs):
                self.device.reset_states(batchsize=self.batch_size)
                for i in range(self.num_wires):
                    rx(self.device, wires=i, params=x[i])
                self.layer(q_machine = self.device)
                y = MeasureAll(obs={'Z0': 1})(self.device)
                return y

        x = pyvqnet.tensor.QTensor([[1,0,0,1],[1,1,0,1]],dtype=pyvqnet.kfloat32,requires_grad=True)
        model = QModel(4,pyvqnet.kcomplex64,2)
        y = model(x)
        print(y)



U1
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. py:class:: pyvqnet.qnn.vqc.tn.U1(has_params: bool = False,trainable: bool = False,init_params=None,wires=None,dtype=pyvqnet.kcomplex64,use_dagger=False)

    定义一个 U1 量子门。

    此类继承自 ``pyvqnet.qnn.vqc.tn.QModule`` 和 ``torch.nn.Module``\ 。
    此类可作为子模块添加到 torch 模型中。

    :param has_params: 是否有参数，如 RX、RY 等门需要设置为 True，无参数的门需要设置为 False，默认为 False。
    :param trainable: 是否有参数需要训练。如果该层使用外部输入数据构建逻辑门矩阵，则设置为 False。如果需要从该层初始化待训练参数，则为 True，默认为 False。
    :param init_params: 用于编码经典数据 QTensor 的初始化参数，默认为 None。
    :param wires: 线路作用的比特索引，默认为 None。
    :param dtype: 逻辑门内部矩阵的数据精度，可设置为 pyvqnet.kcomplex64 或 pyvqnet.kcomplex128，分别对应 float 输入或 double 输入。
    :param use_dagger: 是否使用门的转置共轭版本，默认为 False。
    :return: 一个 ``pyvqnet.qnn.vqc.tn.QModule`` 实例

    示例::

        from pyvqnet.qnn.vqc.tn import U1,TNQMachine,TNQModule,MeasureAll, rx
        import pyvqnet
        pyvqnet.backends.set_backend("torch")

        class QModel(TNQModule):

            def __init__(self, num_wires, dtype,batch_size=2):
                super(QModel, self).__init__()
                self.device = TNQMachine(num_wires)
                self.layer = U1(has_params= True, trainable= True, wires=0)
                self.batch_size = batch_size
                self.num_wires = num_wires

            def forward(self, x, *args, **kwargs):
                self.device.reset_states(batchsize=self.batch_size)
                for i in range(self.num_wires):
                    rx(self.device, wires=i, params=x[i])
                self.layer(q_machine = self.device)
                y = MeasureAll(obs={'Z0': 1})(self.device)
                return y

        x = pyvqnet.tensor.QTensor([[1,0,0,1],[1,1,0,1]],dtype=pyvqnet.kfloat32,requires_grad=True)
        model = QModel(4,pyvqnet.kcomplex64,2)
        y = model(x)
        print(y)


U2
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^


.. py:class:: pyvqnet.qnn.vqc.tn.U2(has_params: bool = False,trainable: bool = False,init_params=None,wires=None,dtype=pyvqnet.kcomplex64,use_dagger=False)

    定义一个 U2 量子门。

    此类继承自 ``pyvqnet.qnn.vqc.tn.QModule`` 和 ``torch.nn.Module``\ 。
    此类可作为子模块添加到 torch 模型中。

    :param has_params: 是否有参数，如 RX、RY 等门需要设置为 True，无参数的门需要设置为 False，默认为 False。
    :param trainable: 是否有参数需要训练。如果该层使用外部输入数据构建逻辑门矩阵，则设置为 False。如果需要从该层初始化待训练参数，则为 True，默认为 False。
    :param init_params: 用于编码经典数据 QTensor 的初始化参数，默认为 None。
    :param wires: 线路作用的比特索引，默认为 None。
    :param dtype: 逻辑门内部矩阵的数据精度，可设置为 pyvqnet.kcomplex64 或 pyvqnet.kcomplex128，分别对应 float 输入或 double 输入。
    :param use_dagger: 是否使用门的转置共轭版本，默认为 False。
    :return: 一个 ``pyvqnet.qnn.vqc.tn.QModule`` 实例

    示例::

        from pyvqnet.qnn.vqc.tn import U2,TNQMachine,TNQModule,MeasureAll, rx
        import pyvqnet
        pyvqnet.backends.set_backend("torch")

        class QModel(TNQModule):

            def __init__(self, num_wires, dtype,batch_size=2):
                super(QModel, self).__init__()
                self.device = TNQMachine(num_wires)
                self.layer = U2(has_params= True, trainable= True, wires=0)
                self.batch_size = batch_size
                self.num_wires = num_wires

            def forward(self, x, *args, **kwargs):
                self.device.reset_states(batchsize=self.batch_size)
                for i in range(self.num_wires):
                    rx(self.device, wires=i, params=x[i])
                self.layer(q_machine = self.device)
                y = MeasureAll(obs={'Z0': 1})(self.device)
                return y

        x = pyvqnet.tensor.QTensor([[1,0,0,1],[1,1,0,1]],dtype=pyvqnet.kfloat32,requires_grad=True)
        model = QModel(4,pyvqnet.kcomplex64,2)
        y = model(x)
        print(y)


U3
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^


.. py:class:: pyvqnet.qnn.vqc.tn.U3(has_params: bool = False,trainable: bool = False,init_params=None,wires=None,dtype=pyvqnet.kcomplex64,use_dagger=False)

    定义一个 U3 量子门。

    此类继承自 ``pyvqnet.qnn.vqc.tn.QModule`` 和 ``torch.nn.Module``\ 。
    此类可作为子模块添加到 torch 模型中。

    :param has_params: 是否有参数，如 RX、RY 等门需要设置为 True，无参数的门需要设置为 False，默认为 False。
    :param trainable: 是否有参数需要训练。如果该层使用外部输入数据构建逻辑门矩阵，则设置为 False。如果需要从该层初始化待训练参数，则为 True，默认为 False。
    :param init_params: 用于编码经典数据 QTensor 的初始化参数，默认为 None。
    :param wires: 线路作用的比特索引，默认为 None。
    :param dtype: 逻辑门内部矩阵的数据精度，可设置为 pyvqnet.kcomplex64 或 pyvqnet.kcomplex128，分别对应 float 输入或 double 输入。
    :param use_dagger: 是否使用门的转置共轭版本，默认为 False。
    :return: 一个 ``pyvqnet.qnn.vqc.tn.QModule`` 实例

    示例::

        from pyvqnet.qnn.vqc.tn import U3,TNQMachine,TNQModule,MeasureAll, rx
        import pyvqnet
        pyvqnet.backends.set_backend("torch")

        class QModel(TNQModule):

            def __init__(self, num_wires, dtype,batch_size=2):
                super(QModel, self).__init__()
                self.device = TNQMachine(num_wires)
                self.layer = U3(has_params= True, trainable= True, wires=0)
                self.batch_size = batch_size
                self.num_wires = num_wires

            def forward(self, x, *args, **kwargs):
                self.device.reset_states(batchsize=self.batch_size)
                for i in range(self.num_wires):
                    rx(self.device, wires=i, params=x[i])
                self.layer(q_machine = self.device)
                y = MeasureAll(obs={'Z0': 1})(self.device)
                return y

        x = pyvqnet.tensor.QTensor([[1,0,0,1],[1,1,0,1]],dtype=pyvqnet.kfloat32,requires_grad=True)
        model = QModel(4,pyvqnet.kcomplex64,2)
        y = model(x)
        print(y)



CNOT
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. py:class:: pyvqnet.qnn.vqc.tn.CNOT(has_params: bool = False,trainable: bool = False,init_params=None,wires=None,dtype=pyvqnet.kcomplex64,use_dagger=False)

    定义一个 CNOT 量子门，别名为 `CX`\ 。

    此类继承自 ``pyvqnet.qnn.vqc.tn.QModule`` 和 ``torch.nn.Module``\ 。
    此类可作为子模块添加到 torch 模型中。

    :param has_params: 是否有参数，如 RX、RY 等门需要设置为 True，无参数的门需要设置为 False，默认为 False。
    :param trainable: 是否有参数需要训练。如果该层使用外部输入数据构建逻辑门矩阵，则设置为 False。如果需要从该层初始化待训练参数，则为 True，默认为 False。
    :param init_params: 用于编码经典数据 QTensor 的初始化参数，默认为 None。
    :param wires: 线路作用的比特索引，默认为 None。
    :param dtype: 逻辑门内部矩阵的数据精度，可设置为 pyvqnet.kcomplex64 或 pyvqnet.kcomplex128，分别对应 float 输入或 double 输入。
    :param use_dagger: 是否使用门的转置共轭版本，默认为 False。
    :return: 一个 ``pyvqnet.qnn.vqc.tn.QModule`` 实例

    示例::

        from pyvqnet.qnn.vqc.tn import CNOT,TNQMachine,TNQModule,MeasureAll, rx
        import pyvqnet
        pyvqnet.backends.set_backend("torch")

        class QModel(TNQModule):

            def __init__(self, num_wires, dtype,batch_size=2):
                super(QModel, self).__init__()
                self.device = TNQMachine(num_wires)
                self.layer = CNOT(wires=[0,1])
                self.batch_size = batch_size
                self.num_wires = num_wires

            def forward(self, x, *args, **kwargs):
                self.device.reset_states(batchsize=self.batch_size)
                for i in range(self.num_wires):
                    rx(self.device, wires=i, params=x[i])
                self.layer(q_machine = self.device)
                y = MeasureAll(obs={'Z0': 1})(self.device)
                return y

        x = pyvqnet.tensor.QTensor([[1,0,0,1],[1,1,0,1]],dtype=pyvqnet.kfloat32,requires_grad=True)
        model = QModel(4,pyvqnet.kcomplex64,2)
        y = model(x)
        print(y)

CY
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. py:class:: pyvqnet.qnn.vqc.tn.CY(has_params: bool = False,trainable: bool = False,init_params=None,wires=None,dtype=pyvqnet.kcomplex64,use_dagger=False)

    定义一个 CY 量子门。

    此类继承自 ``pyvqnet.qnn.vqc.tn.QModule`` 和 ``torch.nn.Module``\ 。
    此类可作为子模块添加到 torch 模型中。

    :param has_params: 是否有参数，如 RX、RY 等门需要设置为 True，无参数的门需要设置为 False，默认为 False。
    :param trainable: 是否有参数需要训练。如果该层使用外部输入数据构建逻辑门矩阵，则设置为 False。如果需要从该层初始化待训练参数，则为 True，默认为 False。
    :param init_params: 用于编码经典数据 QTensor 的初始化参数，默认为 None。
    :param wires: 线路作用的比特索引，默认为 None。
    :param dtype: 逻辑门内部矩阵的数据精度，可设置为 pyvqnet.kcomplex64 或 pyvqnet.kcomplex128，分别对应 float 输入或 double 输入。
    :param use_dagger: 是否使用门的转置共轭版本，默认为 False。
    :return: 一个 ``pyvqnet.qnn.vqc.tn.QModule`` 实例

    示例::

        from pyvqnet.qnn.vqc.tn import CY,TNQMachine,TNQModule,MeasureAll, rx
        import pyvqnet
        pyvqnet.backends.set_backend("torch")

        class QModel(TNQModule):

            def __init__(self, num_wires, dtype,batch_size=2):
                super(QModel, self).__init__()
                self.device = TNQMachine(num_wires)
                self.layer = CY(wires=[0,1])
                self.batch_size = batch_size
                self.num_wires = num_wires

            def forward(self, x, *args, **kwargs):
                self.device.reset_states(batchsize=self.batch_size)
                for i in range(self.num_wires):
                    rx(self.device, wires=i, params=x[i])
                self.layer(q_machine = self.device)
                y = MeasureAll(obs={'Z0': 1})(self.device)
                return y

        x = pyvqnet.tensor.QTensor([[1,0,0,1],[1,1,0,1]],dtype=pyvqnet.kfloat32,requires_grad=True)
        model = QModel(4,pyvqnet.kcomplex64,2)
        y = model(x)
        print(y)


CZ
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. py:class:: pyvqnet.qnn.vqc.tn.CZ(has_params: bool = False,trainable: bool = False,init_params=None,wires=None,dtype=pyvqnet.kcomplex64,use_dagger=False)

    定义一个 CZ 量子门。

    此类继承自 ``pyvqnet.qnn.vqc.tn.QModule`` 和 ``torch.nn.Module``\ 。
    此类可作为子模块添加到 torch 模型中。

    :param has_params: 是否有参数，如 RX、RY 等门需要设置为 True，无参数的门需要设置为 False，默认为 False。
    :param trainable: 是否有参数需要训练。如果该层使用外部输入数据构建逻辑门矩阵，则设置为 False。如果需要从该层初始化待训练参数，则为 True，默认为 False。
    :param init_params: 用于编码经典数据 QTensor 的初始化参数，默认为 None。
    :param wires: 线路作用的比特索引，默认为 None。
    :param dtype: 逻辑门内部矩阵的数据精度，可设置为 pyvqnet.kcomplex64 或 pyvqnet.kcomplex128，分别对应 float 输入或 double 输入。
    :param use_dagger: 是否使用门的转置共轭版本，默认为 False。
    :return: 一个 ``pyvqnet.qnn.vqc.tn.QModule`` 实例

    示例::

        from pyvqnet.qnn.vqc.tn import CZ,TNQMachine,TNQModule,MeasureAll, rx
        import pyvqnet
        pyvqnet.backends.set_backend("torch")

        class QModel(TNQModule):

            def __init__(self, num_wires, dtype,batch_size=2):
                super(QModel, self).__init__()
                self.device = TNQMachine(num_wires)
                self.layer = CZ(wires=[0,1])
                self.batch_size = batch_size
                self.num_wires = num_wires

            def forward(self, x, *args, **kwargs):
                self.device.reset_states(batchsize=self.batch_size)
                for i in range(self.num_wires):
                    rx(self.device, wires=i, params=x[i])
                self.layer(q_machine = self.device)
                y = MeasureAll(obs={'Z0': 1})(self.device)
                return y

        x = pyvqnet.tensor.QTensor([[1,0,0,1],[1,1,0,1]],dtype=pyvqnet.kfloat32,requires_grad=True)
        model = QModel(4,pyvqnet.kcomplex64,2)
        y = model(x)
        print(y)




CR
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. py:class:: pyvqnet.qnn.vqc.tn.CR(has_params: bool = False,trainable: bool = False,init_params=None,wires=None,dtype=pyvqnet.kcomplex64,use_dagger=False)

    定义一个 CR 量子门。

    此类继承自 ``pyvqnet.qnn.vqc.tn.QModule`` 和 ``torch.nn.Module``\ 。
    此类可作为子模块添加到 torch 模型中。

    :param has_params: 是否有参数，如 RX、RY 等门需要设置为 True，无参数的门需要设置为 False，默认为 False。
    :param trainable: 是否有参数需要训练。如果该层使用外部输入数据构建逻辑门矩阵，则设置为 False。如果需要从该层初始化待训练参数，则为 True，默认为 False。
    :param init_params: 用于编码经典数据 QTensor 的初始化参数，默认为 None。
    :param wires: 线路作用的比特索引，默认为 None。
    :param dtype: 逻辑门内部矩阵的数据精度，可设置为 pyvqnet.kcomplex64 或 pyvqnet.kcomplex128，分别对应 float 输入或 double 输入。
    :param use_dagger: 是否使用门的转置共轭版本，默认为 False。
    :return: 一个 ``pyvqnet.qnn.vqc.tn.QModule`` 实例

    示例::

        from pyvqnet.qnn.vqc.tn import CR,TNQMachine,TNQModule,MeasureAll, rx
        import pyvqnet
        pyvqnet.backends.set_backend("torch")

        class QModel(TNQModule):

            def __init__(self, num_wires, dtype,batch_size=2):
                super(QModel, self).__init__()
                self.device = TNQMachine(num_wires)
                self.layer = CR(has_params= True, trainable= True, wires=[0,2])
                self.batch_size = batch_size
                self.num_wires = num_wires

            def forward(self, x, *args, **kwargs):
                self.device.reset_states(batchsize=self.batch_size)
                for i in range(self.num_wires):
                    rx(self.device, wires=i, params=x[i])
                self.layer(q_machine = self.device)
                y = MeasureAll(obs={'Z0': 1})(self.device)
                return y

        x = pyvqnet.tensor.QTensor([[1,0,0,1],[1,1,0,1]],dtype=pyvqnet.kfloat32,requires_grad=True)
        model = QModel(4,pyvqnet.kcomplex64,2)
        y = model(x)
        print(y)



SWAP
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^


.. py:class:: pyvqnet.qnn.vqc.tn.SWAP(has_params: bool = False,trainable: bool = False,init_params=None,wires=None,dtype=pyvqnet.kcomplex64,use_dagger=False)

    定义一个 SWAP 量子门。

    此类继承自 ``pyvqnet.qnn.vqc.tn.QModule`` 和 ``torch.nn.Module``\ 。
    此类可作为子模块添加到 torch 模型中。

    :param has_params: 是否有参数，如 RX、RY 等门需要设置为 True，无参数的门需要设置为 False，默认为 False。
    :param trainable: 是否有参数需要训练。如果该层使用外部输入数据构建逻辑门矩阵，则设置为 False。如果需要从该层初始化待训练参数，则为 True，默认为 False。
    :param init_params: 用于编码经典数据 QTensor 的初始化参数，默认为 None。
    :param wires: 线路作用的比特索引，默认为 None。
    :param dtype: 逻辑门内部矩阵的数据精度，可设置为 pyvqnet.kcomplex64 或 pyvqnet.kcomplex128，分别对应 float 输入或 double 输入。
    :param use_dagger: 是否使用门的转置共轭版本，默认为 False。
    :return: 一个 ``pyvqnet.qnn.vqc.tn.QModule`` 实例

    示例::

        from pyvqnet.qnn.vqc.tn import SWAP,TNQMachine,TNQModule,MeasureAll, rx
        import pyvqnet
        pyvqnet.backends.set_backend("torch")

        class QModel(TNQModule):

            def __init__(self, num_wires, dtype,batch_size=2):
                super(QModel, self).__init__()
                self.device = TNQMachine(num_wires)
                self.layer = SWAP(wires=[0,1])
                self.batch_size = batch_size
                self.num_wires = num_wires

            def forward(self, x, *args, **kwargs):
                self.device.reset_states(batchsize=self.batch_size)
                for i in range(self.num_wires):
                    rx(self.device, wires=i, params=x[i])
                self.layer(q_machine = self.device)
                y = MeasureAll(obs={'Z0': 1})(self.device)
                return y

        x = pyvqnet.tensor.QTensor([[1,0,0,1],[1,1,0,1]],dtype=pyvqnet.kfloat32,requires_grad=True)
        model = QModel(4,pyvqnet.kcomplex64,2)
        y = model(x)
        print(y)


CSWAP
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. py:class:: pyvqnet.qnn.vqc.tn.CSWAP(has_params: bool = False,trainable: bool = False,init_params=None,wires=None,dtype=pyvqnet.kcomplex64,use_dagger=False)

    定义一个 SWAP 量子门。

    .. math:: CSWAP = \begin{bmatrix}
            1 & 0 & 0 & 0 & 0 & 0 & 0 & 0 \\
            0 & 1 & 0 & 0 & 0 & 0 & 0 & 0 \\
            0 & 0 & 1 & 0 & 0 & 0 & 0 & 0 \\
            0 & 0 & 0 & 1 & 0 & 0 & 0 & 0 \\
            0 & 0 & 0 & 0 & 1 & 0 & 0 & 0 \\
            0 & 0 & 0 & 0 & 0 & 0 & 1 & 0 \\
            0 & 0 & 0 & 0 & 0 & 1 & 0 & 0 \\
            0 & 0 & 0 & 0 & 0 & 0 & 0 & 1
        \end{bmatrix}.

    此类继承自 ``pyvqnet.qnn.vqc.tn.QModule`` 和 ``torch.nn.Module``\ 。
    此类可作为子模块添加到 torch 模型中。

    :param has_params: 是否有参数，如 RX、RY 等门需要设置为 True，无参数的门需要设置为 False，默认为 False。
    :param trainable: 是否有参数需要训练。如果该层使用外部输入数据构建逻辑门矩阵，则设置为 False。如果需要从该层初始化待训练参数，则为 True，默认为 False。
    :param init_params: 用于编码经典数据 QTensor 的初始化参数，默认为 None。
    :param wires: 线路作用的比特索引，默认为 None。
    :param dtype: 逻辑门内部矩阵的数据精度，可设置为 pyvqnet.kcomplex64 或 pyvqnet.kcomplex128，分别对应 float 输入或 double 输入。
    :param use_dagger: 是否使用门的转置共轭版本，默认为 False。
    :return: 一个 ``pyvqnet.qnn.vqc.tn.QModule`` 实例

    示例::

        from pyvqnet.qnn.vqc.tn import CSWAP,TNQMachine,TNQModule,MeasureAll, rx
        import pyvqnet
        pyvqnet.backends.set_backend("torch")

        class QModel(TNQModule):

            def __init__(self, num_wires, dtype,batch_size=2):
                super(QModel, self).__init__()
                self.device = TNQMachine(num_wires)
                self.layer = CSWAP(wires=[0,1,2])
                self.batch_size = batch_size
                self.num_wires = num_wires

            def forward(self, x, *args, **kwargs):
                self.device.reset_states(batchsize=self.batch_size)
                for i in range(self.num_wires):
                    rx(self.device, wires=i, params=x[i])
                self.layer(q_machine = self.device)
                y = MeasureAll(obs={'Z0': 1})(self.device)
                return y

        x = pyvqnet.tensor.QTensor([[1,0,0,1],[1,1,0,1]],dtype=pyvqnet.kfloat32,requires_grad=True)
        model = QModel(4,pyvqnet.kcomplex64,2)
        y = model(x)
        print(y)

RXX
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^


.. py:class:: pyvqnet.qnn.vqc.tn.RXX(has_params: bool = False,trainable: bool = False,init_params=None,wires=None,dtype=pyvqnet.kcomplex64,use_dagger=False)

    定义一个 RXX 量子门。

    此类继承自 ``pyvqnet.qnn.vqc.tn.QModule`` 和 ``torch.nn.Module``\ 。
    此类可作为子模块添加到 torch 模型中。

    :param has_params: 是否有参数，如 RX、RY 等门需要设置为 True，无参数的门需要设置为 False，默认为 False。
    :param trainable: 是否有参数需要训练。如果该层使用外部输入数据构建逻辑门矩阵，则设置为 False。如果需要从该层初始化待训练参数，则为 True，默认为 False。
    :param init_params: 用于编码经典数据 QTensor 的初始化参数，默认为 None。
    :param wires: 线路作用的比特索引，默认为 None。
    :param dtype: 逻辑门内部矩阵的数据精度，可设置为 pyvqnet.kcomplex64 或 pyvqnet.kcomplex128，分别对应 float 输入或 double 输入。
    :param use_dagger: 是否使用门的转置共轭版本，默认为 False。
    :return: 一个 ``pyvqnet.qnn.vqc.tn.QModule`` 实例

    示例::

        from pyvqnet.qnn.vqc.tn import RXX,TNQMachine,TNQModule,MeasureAll, rx
        import pyvqnet
        pyvqnet.backends.set_backend("torch")

        class QModel(TNQModule):

            def __init__(self, num_wires, dtype,batch_size=2):
                super(QModel, self).__init__()
                self.device = TNQMachine(num_wires)
                self.layer = RXX(has_params= True, trainable= True, wires=[0,2])
                self.batch_size = batch_size
                self.num_wires = num_wires

            def forward(self, x, *args, **kwargs):
                self.device.reset_states(batchsize=self.batch_size)
                for i in range(self.num_wires):
                    rx(self.device, wires=i, params=x[i])
                self.layer(q_machine = self.device)
                y = MeasureAll(obs={'Z0': 1})(self.device)
                return y

        x = pyvqnet.tensor.QTensor([[1,0,0,1],[1,1,0,1]],dtype=pyvqnet.kfloat32,requires_grad=True)
        model = QModel(4,pyvqnet.kcomplex64,2)
        y = model(x)
        print(y)

RYY
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. py:class:: pyvqnet.qnn.vqc.tn.RYY(has_params: bool = False,trainable: bool = False,init_params=None,wires=None,dtype=pyvqnet.kcomplex64,use_dagger=False)

    定义一个 RYY 量子门。

    此类继承自 ``pyvqnet.qnn.vqc.tn.QModule`` 和 ``torch.nn.Module``\ 。
    此类可作为子模块添加到 torch 模型中。

    :param has_params: 是否有参数，如 RX、RY 等门需要设置为 True，无参数的门需要设置为 False，默认为 False。
    :param trainable: 是否有参数需要训练。如果该层使用外部输入数据构建逻辑门矩阵，则设置为 False。如果需要从该层初始化待训练参数，则为 True，默认为 False。
    :param init_params: 用于编码经典数据 QTensor 的初始化参数，默认为 None。
    :param wires: 线路作用的比特索引，默认为 None。
    :param dtype: 逻辑门内部矩阵的数据精度，可设置为 pyvqnet.kcomplex64 或 pyvqnet.kcomplex128，分别对应 float 输入或 double 输入。
    :param use_dagger: 是否使用门的转置共轭版本，默认为 False。
    :return: 一个 ``pyvqnet.qnn.vqc.tn.QModule`` 实例

    示例::

        from pyvqnet.qnn.vqc.tn import RYY,TNQMachine,TNQModule,MeasureAll, rx
        import pyvqnet
        pyvqnet.backends.set_backend("torch")

        class QModel(TNQModule):

            def __init__(self, num_wires, dtype,batch_size=2):
                super(QModel, self).__init__()
                self.device = TNQMachine(num_wires)
                self.layer = RYY(has_params= True, trainable= True, wires=[0,2])
                self.batch_size = batch_size
                self.num_wires = num_wires

            def forward(self, x, *args, **kwargs):
                self.device.reset_states(batchsize=self.batch_size)
                for i in range(self.num_wires):
                    rx(self.device, wires=i, params=x[i])
                self.layer(q_machine = self.device)
                y = MeasureAll(obs={'Z0': 1})(self.device)
                return y

        x = pyvqnet.tensor.QTensor([[1,0,0,1],[1,1,0,1]],dtype=pyvqnet.kfloat32,requires_grad=True)
        model = QModel(4,pyvqnet.kcomplex64,2)
        y = model(x)
        print(y)


RZZ
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. py:class:: pyvqnet.qnn.vqc.tn.RZZ(has_params: bool = False,trainable: bool = False,init_params=None,wires=None,dtype=pyvqnet.kcomplex64,use_dagger=False)

    定义一个 RZZ 量子门。

    此类继承自 ``pyvqnet.qnn.vqc.tn.QModule`` 和 ``torch.nn.Module``\ 。
    此类可作为子模块添加到 torch 模型中。

    :param has_params: 是否有参数，如 RX、RY 等门需要设置为 True，无参数的门需要设置为 False，默认为 False。
    :param trainable: 是否有参数需要训练。如果该层使用外部输入数据构建逻辑门矩阵，则设置为 False。如果需要从该层初始化待训练参数，则为 True，默认为 False。
    :param init_params: 用于编码经典数据 QTensor 的初始化参数，默认为 None。
    :param wires: 线路作用的比特索引，默认为 None。
    :param dtype: 逻辑门内部矩阵的数据精度，可设置为 pyvqnet.kcomplex64 或 pyvqnet.kcomplex128，分别对应 float 输入或 double 输入。
    :param use_dagger: 是否使用门的转置共轭版本，默认为 False。
    :return: 一个 ``pyvqnet.qnn.vqc.tn.QModule`` 实例

    示例::

        from pyvqnet.qnn.vqc.tn import RZZ,TNQMachine,TNQModule,MeasureAll, rx
        import pyvqnet
        pyvqnet.backends.set_backend("torch")

        class QModel(TNQModule):

            def __init__(self, num_wires, dtype,batch_size=2):
                super(QModel, self).__init__()
                self.device = TNQMachine(num_wires)
                self.layer = RZZ(has_params= True, trainable= True, wires=[0,2])
                self.batch_size = batch_size
                self.num_wires = num_wires

            def forward(self, x, *args, **kwargs):
                self.device.reset_states(batchsize=self.batch_size)
                for i in range(self.num_wires):
                    rx(self.device, wires=i, params=x[i])
                self.layer(q_machine = self.device)
                y = MeasureAll(obs={'Z0': 1})(self.device)
                return y

        x = pyvqnet.tensor.QTensor([[1,0,0,1],[1,1,0,1]],dtype=pyvqnet.kfloat32,requires_grad=True)
        model = QModel(4,pyvqnet.kcomplex64,2)
        y = model(x)
        print(y)



RZX
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. py:class:: pyvqnet.qnn.vqc.tn.RZX(has_params: bool = False,trainable: bool = False,init_params=None,wires=None,dtype=pyvqnet.kcomplex64,use_dagger=False)

    定义一个 RZX 量子门。

    此类继承自 ``pyvqnet.qnn.vqc.tn.QModule`` 和 ``torch.nn.Module``\ 。
    此类可作为子模块添加到 torch 模型中。

    :param has_params: 是否有参数，如 RX、RY 等门需要设置为 True，无参数的门需要设置为 False，默认为 False。
    :param trainable: 是否有参数需要训练。如果该层使用外部输入数据构建逻辑门矩阵，则设置为 False。如果需要从该层初始化待训练参数，则为 True，默认为 False。
    :param init_params: 用于编码经典数据 QTensor 的初始化参数，默认为 None。
    :param wires: 线路作用的比特索引，默认为 None。
    :param dtype: 逻辑门内部矩阵的数据精度，可设置为 pyvqnet.kcomplex64 或 pyvqnet.kcomplex128，分别对应 float 输入或 double 输入。
    :param use_dagger: 是否使用门的转置共轭版本，默认为 False。
    :return: 一个 ``pyvqnet.qnn.vqc.tn.QModule`` 实例

    示例::

        from pyvqnet.qnn.vqc.tn import RZX,TNQMachine,TNQModule,MeasureAll, rx
        import pyvqnet
        pyvqnet.backends.set_backend("torch")

        class QModel(TNQModule):

            def __init__(self, num_wires, dtype,batch_size=2):
                super(QModel, self).__init__()
                self.device = TNQMachine(num_wires)
                self.layer = RZX(has_params= True, trainable= True, wires=[0,2])
                self.batch_size = batch_size
                self.num_wires = num_wires

            def forward(self, x, *args, **kwargs):
                self.device.reset_states(batchsize=self.batch_size)
                for i in range(self.num_wires):
                    rx(self.device, wires=i, params=x[i])
                self.layer(q_machine = self.device)
                y = MeasureAll(obs={'Z0': 1})(self.device)
                return y

        x = pyvqnet.tensor.QTensor([[1,0,0,1],[1,1,0,1]],dtype=pyvqnet.kfloat32,requires_grad=True)
        model = QModel(4,pyvqnet.kcomplex64,2)
        y = model(x)
        print(y)

Toffoli
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. py:class:: pyvqnet.qnn.vqc.tn.Toffoli(has_params: bool = False,trainable: bool = False,init_params=None,wires=None,dtype=pyvqnet.kcomplex64,use_dagger=False)

    定义一个 Toffoli 量子门。

    此类继承自 ``pyvqnet.qnn.vqc.tn.QModule`` 和 ``torch.nn.Module``\ 。
    此类可作为子模块添加到 torch 模型中。

    :param has_params: 是否有参数，如 RX、RY 等门需要设置为 True，无参数的门需要设置为 False，默认为 False。
    :param trainable: 是否有参数需要训练。如果该层使用外部输入数据构建逻辑门矩阵，则设置为 False。如果需要从该层初始化待训练参数，则为 True，默认为 False。
    :param init_params: 用于编码经典数据 QTensor 的初始化参数，默认为 None。
    :param wires: 线路作用的比特索引，默认为 None。
    :param dtype: 逻辑门内部矩阵的数据精度，可设置为 pyvqnet.kcomplex64 或 pyvqnet.kcomplex128，分别对应 float 输入或 double 输入。
    :param use_dagger: 是否使用门的转置共轭版本，默认为 False。
    :return: 一个 ``pyvqnet.qnn.vqc.tn.QModule`` 实例

    示例::

        from pyvqnet.qnn.vqc.tn import Toffoli,TNQMachine,TNQModule,MeasureAll, rx
        import pyvqnet
        pyvqnet.backends.set_backend("torch")

        class QModel(TNQModule):

            def __init__(self, num_wires, dtype,batch_size=2):
                super(QModel, self).__init__()
                self.device = TNQMachine(num_wires)
                self.layer = Toffoli(wires=[0,2,1])
                self.batch_size = batch_size
                self.num_wires = num_wires

            def forward(self, x, *args, **kwargs):
                self.device.reset_states(batchsize=self.batch_size)
                for i in range(self.num_wires):
                    rx(self.device, wires=i, params=x[i])
                self.layer(q_machine = self.device)
                y = MeasureAll(obs={'Z0': 1})(self.device)
                return y

        x = pyvqnet.tensor.QTensor([[1,0,0,1],[1,1,0,1]],dtype=pyvqnet.kfloat32,requires_grad=True)
        model = QModel(4,pyvqnet.kcomplex64,2)
        y = model(x)
        print(y)

IsingXX
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^


.. py:class:: pyvqnet.qnn.vqc.tn.IsingXX(has_params: bool = False,trainable: bool = False,init_params=None,wires=None,dtype=pyvqnet.kcomplex64,use_dagger=False)

    定义一个 IsingXX 量子门。

    此类继承自 ``pyvqnet.qnn.vqc.tn.QModule`` 和 ``torch.nn.Module``\ 。
    此类可作为子模块添加到 torch 模型中。

    :param has_params: 是否有参数，如 RX、RY 等门需要设置为 True，无参数的门需要设置为 False，默认为 False。
    :param trainable: 是否有参数需要训练。如果该层使用外部输入数据构建逻辑门矩阵，则设置为 False。如果需要从该层初始化待训练参数，则为 True，默认为 False。
    :param init_params: 用于编码经典数据 QTensor 的初始化参数，默认为 None。
    :param wires: 线路作用的比特索引，默认为 None。
    :param dtype: 逻辑门内部矩阵的数据精度，可设置为 pyvqnet.kcomplex64 或 pyvqnet.kcomplex128，分别对应 float 输入或 double 输入。
    :param use_dagger: 是否使用门的转置共轭版本，默认为 False。
    :return: 一个 ``pyvqnet.qnn.vqc.tn.QModule`` 实例

    示例::

        from pyvqnet.qnn.vqc.tn import IsingXX,TNQMachine,TNQModule,MeasureAll, rx
        import pyvqnet
        pyvqnet.backends.set_backend("torch")

        class QModel(TNQModule):

            def __init__(self, num_wires, dtype,batch_size=2):
                super(QModel, self).__init__()
                self.device = TNQMachine(num_wires)
                self.layer = IsingXX(has_params= True, trainable= True, wires=[0,2])
                self.batch_size = batch_size
                self.num_wires = num_wires

            def forward(self, x, *args, **kwargs):
                self.device.reset_states(batchsize=self.batch_size)
                for i in range(self.num_wires):
                    rx(self.device, wires=i, params=x[i])
                self.layer(q_machine = self.device)
                y = MeasureAll(obs={'Z0': 1})(self.device)
                return y

        x = pyvqnet.tensor.QTensor([[1,0,0,1],[1,1,0,1]],dtype=pyvqnet.kfloat32,requires_grad=True)
        model = QModel(4,pyvqnet.kcomplex64,2)
        y = model(x)
        print(y)


IsingYY
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^


.. py:class:: pyvqnet.qnn.vqc.tn.IsingYY(has_params: bool = False,trainable: bool = False,init_params=None,wires=None,dtype=pyvqnet.kcomplex64,use_dagger=False)

    定义一个 IsingYY 量子门。

    此类继承自 ``pyvqnet.qnn.vqc.tn.QModule`` 和 ``torch.nn.Module``\ 。
    此类可作为子模块添加到 torch 模型中。

    :param has_params: 是否有参数，如 RX、RY 等门需要设置为 True，无参数的门需要设置为 False，默认为 False。
    :param trainable: 是否有参数需要训练。如果该层使用外部输入数据构建逻辑门矩阵，则设置为 False。如果需要从该层初始化待训练参数，则为 True，默认为 False。
    :param init_params: 用于编码经典数据 QTensor 的初始化参数，默认为 None。
    :param wires: 线路作用的比特索引，默认为 None。
    :param dtype: 逻辑门内部矩阵的数据精度，可设置为 pyvqnet.kcomplex64 或 pyvqnet.kcomplex128，分别对应 float 输入或 double 输入。
    :param use_dagger: 是否使用门的转置共轭版本，默认为 False。
    :return: 一个 ``pyvqnet.qnn.vqc.tn.QModule`` 实例

    示例::

        from pyvqnet.qnn.vqc.tn import IsingYY,TNQMachine,TNQModule,MeasureAll, rx
        import pyvqnet
        pyvqnet.backends.set_backend("torch")

        class QModel(TNQModule):

            def __init__(self, num_wires, dtype,batch_size=2):
                super(QModel, self).__init__()
                self.device = TNQMachine(num_wires)
                self.layer = IsingYY(has_params= True, trainable= True, wires=[0,2])
                self.batch_size = batch_size
                self.num_wires = num_wires

            def forward(self, x, *args, **kwargs):
                self.device.reset_states(batchsize=self.batch_size)
                for i in range(self.num_wires):
                    rx(self.device, wires=i, params=x[i])
                self.layer(q_machine = self.device)
                y = MeasureAll(obs={'Z0': 1})(self.device)
                return y

        x = pyvqnet.tensor.QTensor([[1,0,0,1],[1,1,0,1]],dtype=pyvqnet.kfloat32,requires_grad=True)
        model = QModel(4,pyvqnet.kcomplex64,2)
        y = model(x)
        print(y)


IsingZZ
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. py:class:: pyvqnet.qnn.vqc.tn.IsingZZ(has_params: bool = False,trainable: bool = False,init_params=None,wires=None,dtype=pyvqnet.kcomplex64,use_dagger=False)

    定义一个 IsingZZ 量子门。

    此类继承自 ``pyvqnet.qnn.vqc.tn.QModule`` 和 ``torch.nn.Module``\ 。
    此类可作为子模块添加到 torch 模型中。

    :param has_params: 是否有参数，如 RX、RY 等门需要设置为 True，无参数的门需要设置为 False，默认为 False。
    :param trainable: 是否有参数需要训练。如果该层使用外部输入数据构建逻辑门矩阵，则设置为 False。如果需要从该层初始化待训练参数，则为 True，默认为 False。
    :param init_params: 用于编码经典数据 QTensor 的初始化参数，默认为 None。
    :param wires: 线路作用的比特索引，默认为 None。
    :param dtype: 逻辑门内部矩阵的数据精度，可设置为 pyvqnet.kcomplex64 或 pyvqnet.kcomplex128，分别对应 float 输入或 double 输入。
    :param use_dagger: 是否使用门的转置共轭版本，默认为 False。
    :return: 一个 ``pyvqnet.qnn.vqc.tn.QModule`` 实例

    示例::

        from pyvqnet.qnn.vqc.tn import IsingZZ,TNQMachine,TNQModule,MeasureAll, rx
        import pyvqnet
        pyvqnet.backends.set_backend("torch")

        class QModel(TNQModule):

            def __init__(self, num_wires, dtype,batch_size=2):
                super(QModel, self).__init__()
                self.device = TNQMachine(num_wires)
                self.layer = IsingZZ(has_params= True, trainable= True, wires=[0,2])
                self.batch_size = batch_size
                self.num_wires = num_wires

            def forward(self, x, *args, **kwargs):
                self.device.reset_states(batchsize=self.batch_size)
                for i in range(self.num_wires):
                    rx(self.device, wires=i, params=x[i])
                self.layer(q_machine = self.device)
                y = MeasureAll(obs={'Z0': 1})(self.device)
                return y

        x = pyvqnet.tensor.QTensor([[1,0,0,1],[1,1,0,1]],dtype=pyvqnet.kfloat32,requires_grad=True)
        model = QModel(4,pyvqnet.kcomplex64,2)
        y = model(x)
        print(y)


IsingXY
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^


.. py:class:: pyvqnet.qnn.vqc.tn.IsingXY(has_params: bool = False,trainable: bool = False,init_params=None,wires=None,dtype=pyvqnet.kcomplex64,use_dagger=False)

    定义一个 IsingXY 量子门。

    此类继承自 ``pyvqnet.qnn.vqc.tn.QModule`` 和 ``torch.nn.Module``\ 。
    此类可作为子模块添加到 torch 模型中。

    :param has_params: 是否有参数，如 RX、RY 等门需要设置为 True，无参数的门需要设置为 False，默认为 False。
    :param trainable: 是否有参数需要训练。如果该层使用外部输入数据构建逻辑门矩阵，则设置为 False。如果需要从该层初始化待训练参数，则为 True，默认为 False。
    :param init_params: 用于编码经典数据 QTensor 的初始化参数，默认为 None。
    :param wires: 线路作用的比特索引，默认为 None。
    :param dtype: 逻辑门内部矩阵的数据精度，可设置为 pyvqnet.kcomplex64 或 pyvqnet.kcomplex128，分别对应 float 输入或 double 输入。
    :param use_dagger: 是否使用门的转置共轭版本，默认为 False。
    :return: 一个 ``pyvqnet.qnn.vqc.tn.QModule`` 实例

    示例::

        from pyvqnet.qnn.vqc.tn import IsingXY,TNQMachine,TNQModule,MeasureAll, rx
        import pyvqnet
        pyvqnet.backends.set_backend("torch")

        class QModel(TNQModule):

            def __init__(self, num_wires, dtype,batch_size=2):
                super(QModel, self).__init__()
                self.device = TNQMachine(num_wires)
                self.layer = IsingXY(has_params= True, trainable= True, wires=[0,2])
                self.batch_size = batch_size
                self.num_wires = num_wires

            def forward(self, x, *args, **kwargs):
                self.device.reset_states(batchsize=self.batch_size)
                for i in range(self.num_wires):
                    rx(self.device, wires=i, params=x[i])
                self.layer(q_machine = self.device)
                y = MeasureAll(obs={'Z0': 1})(self.device)
                return y

        x = pyvqnet.tensor.QTensor([[1,0,0,1],[1,1,0,1]],dtype=pyvqnet.kfloat32,requires_grad=True)
        model = QModel(4,pyvqnet.kcomplex64,2)
        y = model(x)
        print(y)


PhaseShift
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^


.. py:class:: pyvqnet.qnn.vqc.tn.PhaseShift(has_params: bool = False,trainable: bool = False,init_params=None,wires=None,dtype=pyvqnet.kcomplex64,use_dagger=False)

    定义一个 PhaseShift 量子门。

    此类继承自 ``pyvqnet.qnn.vqc.tn.QModule`` 和 ``torch.nn.Module``\ 。
    此类可作为子模块添加到 torch 模型中。

    :param has_params: 是否有参数，如 RX、RY 等门需要设置为 True，无参数的门需要设置为 False，默认为 False。
    :param trainable: 是否有参数需要训练。如果该层使用外部输入数据构建逻辑门矩阵，则设置为 False。如果需要从该层初始化待训练参数，则为 True，默认为 False。
    :param init_params: 用于编码经典数据 QTensor 的初始化参数，默认为 None。
    :param wires: 线路作用的比特索引，默认为 None。
    :param dtype: 逻辑门内部矩阵的数据精度，可设置为 pyvqnet.kcomplex64 或 pyvqnet.kcomplex128，分别对应 float 输入或 double 输入。
    :param use_dagger: 是否使用门的转置共轭版本，默认为 False。
    :return: 一个 ``pyvqnet.qnn.vqc.tn.QModule`` 实例

    示例::

        from pyvqnet.qnn.vqc.tn import PhaseShift,TNQMachine,TNQModule,MeasureAll, rx
        import pyvqnet
        pyvqnet.backends.set_backend("torch")

        class QModel(TNQModule):

            def __init__(self, num_wires, dtype,batch_size=2):
                super(QModel, self).__init__()
                self.device = TNQMachine(num_wires)
                self.layer = PhaseShift(has_params= True, trainable= True, wires=1)
                self.batch_size = batch_size
                self.num_wires = num_wires

            def forward(self, x, *args, **kwargs):
                self.device.reset_states(batchsize=self.batch_size)
                for i in range(self.num_wires):
                    rx(self.device, wires=i, params=x[i])
                self.layer(q_machine = self.device)
                y = MeasureAll(obs={'Z0': 1})(self.device)
                return y

        x = pyvqnet.tensor.QTensor([[1,0,0,1],[1,1,0,1]],dtype=pyvqnet.kfloat32,requires_grad=True)
        model = QModel(4,pyvqnet.kcomplex64,2)
        y = model(x)
        print(y)


MultiRZ
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. py:class:: pyvqnet.qnn.vqc.tn.MultiRZ(has_params: bool = False,trainable: bool = False,init_params=None,wires=None,dtype=pyvqnet.kcomplex64,use_dagger=False)

    定义一个 MultiRZ 量子门。

    此类继承自 ``pyvqnet.qnn.vqc.tn.QModule`` 和 ``torch.nn.Module``\ 。
    此类可作为子模块添加到 torch 模型中。

    :param has_params: 是否有参数，如 RX、RY 等门需要设置为 True，无参数的门需要设置为 False，默认为 False。
    :param trainable: 是否有参数需要训练。如果该层使用外部输入数据构建逻辑门矩阵，则设置为 False。如果需要从该层初始化待训练参数，则为 True，默认为 False。
    :param init_params: 用于编码经典数据 QTensor 的初始化参数，默认为 None。
    :param wires: 线路作用的比特索引，默认为 None。
    :param dtype: 逻辑门内部矩阵的数据精度，可设置为 pyvqnet.kcomplex64 或 pyvqnet.kcomplex128，分别对应 float 输入或 double 输入。
    :param use_dagger: 是否使用门的转置共轭版本，默认为 False。
    :return: 一个 ``pyvqnet.qnn.vqc.tn.QModule`` 实例

    示例::

        from pyvqnet.qnn.vqc.tn import MultiRZ,TNQMachine,TNQModule,MeasureAll, rx
        import pyvqnet
        pyvqnet.backends.set_backend("torch")

        class QModel(TNQModule):

            def __init__(self, num_wires, dtype,batch_size=2):
                super(QModel, self).__init__()
                self.device = TNQMachine(num_wires)
                self.layer = MultiRZ(has_params= True, trainable= True, wires=[0,2])
                self.batch_size = batch_size
                self.num_wires = num_wires

            def forward(self, x, *args, **kwargs):
                self.device.reset_states(batchsize=self.batch_size)
                for i in range(self.num_wires):
                    rx(self.device, wires=i, params=x[i])
                self.layer(q_machine = self.device)
                y = MeasureAll(obs={'Z0': 1})(self.device)
                return y

        x = pyvqnet.tensor.QTensor([[1,0,0,1],[1,1,0,1]],dtype=pyvqnet.kfloat32,requires_grad=True)
        model = QModel(4,pyvqnet.kcomplex64,2)
        y = model(x)
        print(y)



SDG
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^


.. py:class:: pyvqnet.qnn.vqc.tn.SDG(has_params: bool = False,trainable: bool = False,init_params=None,wires=None,dtype=pyvqnet.kcomplex64,use_dagger=False)

    定义一个 SDG 量子门。

    此类继承自 ``pyvqnet.qnn.vqc.tn.QModule`` 和 ``torch.nn.Module``\ 。
    此类可作为子模块添加到 torch 模型中。

    :param has_params: 是否有参数，如 RX、RY 等门需要设置为 True，无参数的门需要设置为 False，默认为 False。
    :param trainable: 是否有参数需要训练。如果该层使用外部输入数据构建逻辑门矩阵，则设置为 False。如果需要从该层初始化待训练参数，则为 True，默认为 False。
    :param init_params: 用于编码经典数据 QTensor 的初始化参数，默认为 None。
    :param wires: 线路作用的比特索引，默认为 None。
    :param dtype: 逻辑门内部矩阵的数据精度，可设置为 pyvqnet.kcomplex64 或 pyvqnet.kcomplex128，分别对应 float 输入或 double 输入。
    :param use_dagger: 是否使用门的转置共轭版本，默认为 False。
    :return: 一个 ``pyvqnet.qnn.vqc.tn.QModule`` 实例

    示例::

        from pyvqnet.qnn.vqc.tn import SDG,TNQMachine,TNQModule,MeasureAll, rx
        import pyvqnet
        pyvqnet.backends.set_backend("torch")

        class QModel(TNQModule):

            def __init__(self, num_wires, dtype,batch_size=2):
                super(QModel, self).__init__()
                self.device = TNQMachine(num_wires)
                self.layer = SDG(wires=0)
                self.batch_size = batch_size
                self.num_wires = num_wires

            def forward(self, x, *args, **kwargs):
                self.device.reset_states(batchsize=self.batch_size)
                for i in range(self.num_wires):
                    rx(self.device, wires=i, params=x[i])
                self.layer(q_machine = self.device)
                y = MeasureAll(obs={'Z0': 1})(self.device)
                return y

        x = pyvqnet.tensor.QTensor([[1,0,0,1],[1,1,0,1]],dtype=pyvqnet.kfloat32,requires_grad=True)
        model = QModel(4,pyvqnet.kcomplex64,2)
        y = model(x)
        print(y)




TDG
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. py:class:: pyvqnet.qnn.vqc.tn.TDG(has_params: bool = False,trainable: bool = False,init_params=None,wires=None,dtype=pyvqnet.kcomplex64,use_dagger=False)

    定义一个 SDG 量子门。

    此类继承自 ``pyvqnet.qnn.vqc.tn.QModule`` 和 ``torch.nn.Module``\ 。
    此类可作为子模块添加到 torch 模型中。

    :param has_params: 是否有参数，如 RX、RY 等门需要设置为 True，无参数的门需要设置为 False，默认为 False。
    :param trainable: 是否有参数需要训练。如果该层使用外部输入数据构建逻辑门矩阵，则设置为 False。如果需要从该层初始化待训练参数，则为 True，默认为 False。
    :param init_params: 用于编码经典数据 QTensor 的初始化参数，默认为 None。
    :param wires: 线路作用的比特索引，默认为 None。
    :param dtype: 逻辑门内部矩阵的数据精度，可设置为 pyvqnet.kcomplex64 或 pyvqnet.kcomplex128，分别对应 float 输入或 double 输入。
    :param use_dagger: 是否使用门的转置共轭版本，默认为 False。
    :return: 一个 ``pyvqnet.qnn.vqc.tn.QModule`` 实例

    示例::

        from pyvqnet.qnn.vqc.tn import TDG,TNQMachine,TNQModule,MeasureAll, rx
        import pyvqnet
        pyvqnet.backends.set_backend("torch")

        class QModel(TNQModule):

            def __init__(self, num_wires, dtype,batch_size=2):
                super(QModel, self).__init__()
                self.device = TNQMachine(num_wires)
                self.layer = TDG(wires=0)
                self.batch_size = batch_size
                self.num_wires = num_wires

            def forward(self, x, *args, **kwargs):
                self.device.reset_states(batchsize=self.batch_size)
                for i in range(self.num_wires):
                    rx(self.device, wires=i, params=x[i])
                self.layer(q_machine = self.device)
                y = MeasureAll(obs={'Z0': 1})(self.device)
                return y

        x = pyvqnet.tensor.QTensor([[1,0,0,1],[1,1,0,1]],dtype=pyvqnet.kfloat32,requires_grad=True)
        model = QModel(4,pyvqnet.kcomplex64,2)
        y = model(x)
        print(y)



ControlledPhaseShift
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^


.. py:class:: pyvqnet.qnn.vqc.tn.ControlledPhaseShift(has_params: bool = False,trainable: bool = False,init_params=None,wires=None,dtype=pyvqnet.kcomplex64,use_dagger=False)

    定义一个 ControlledPhaseShift 量子门。

    此类继承自 ``pyvqnet.qnn.vqc.tn.QModule`` 和 ``torch.nn.Module``\ 。
    此类可作为子模块添加到 torch 模型中。

    :param has_params: 是否有参数，如 RX、RY 等门需要设置为 True，无参数的门需要设置为 False，默认为 False。
    :param trainable: 是否有参数需要训练。如果该层使用外部输入数据构建逻辑门矩阵，则设置为 False。如果需要从该层初始化待训练参数，则为 True，默认为 False。
    :param init_params: 用于编码经典数据 QTensor 的初始化参数，默认为 None。
    :param wires: 线路作用的比特索引，默认为 None。
    :param dtype: 逻辑门内部矩阵的数据精度，可设置为 pyvqnet.kcomplex64 或 pyvqnet.kcomplex128，分别对应 float 输入或 double 输入。
    :param use_dagger: 是否使用门的转置共轭版本，默认为 False。
    :return: 一个 ``pyvqnet.qnn.vqc.tn.QModule`` 实例

    示例::

        from pyvqnet.qnn.vqc.tn import ControlledPhaseShift,TNQMachine,TNQModule,MeasureAll, rx
        import pyvqnet
        pyvqnet.backends.set_backend("torch")

        class QModel(TNQModule):

            def __init__(self, num_wires, dtype,batch_size=2):
                super(QModel, self).__init__()
                self.device = TNQMachine(num_wires)
                self.layer = ControlledPhaseShift(has_params= True, trainable= True, wires=[0,2])
                self.batch_size = batch_size
                self.num_wires = num_wires

            def forward(self, x, *args, **kwargs):
                self.device.reset_states(batchsize=self.batch_size)
                for i in range(self.num_wires):
                    rx(self.device, wires=i, params=x[i])
                self.layer(q_machine = self.device)
                y = MeasureAll(obs={'Z0': 1})(self.device)
                return y

        x = pyvqnet.tensor.QTensor([[1,0,0,1],[1,1,0,1]],dtype=pyvqnet.kfloat32,requires_grad=True)
        model = QModel(4,pyvqnet.kcomplex64,2)
        y = model(x)
        print(y)


测量 API
------------------------------

VQC_Purity
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^


.. py:function:: pyvqnet.qnn.vqc.tn.VQC_Purity(state, qubits_idx, num_wires, use_tn=False)

    从状态向量 ``state`` 计算特定量子比特 ``qubits_idx`` 上的纯度。

    .. math::
        \gamma = \text{Tr}(\rho^2)

    其中 :math:`\rho` 是密度矩阵。归一化量子态的纯度满足 :math:`\frac{1}{d} \leq \gamma \leq 1`\ ，
    其中 :math:`d` 是希尔伯特空间的维度。
    纯态的纯度为 1。

    :param state: 从 TNQMachine.get_states() 获取的量子态
    :param qubits_idx: 要计算纯度的量子比特索引
    :param num_wires: 量子比特索引
    :param use_tn: 使用张量网络需要设置为 True，默认为 False

    :return: 纯度

    .. note::

        batch_size 需要 TNQModule。

    示例::

        import pyvqnet
        from pyvqnet.qnn.vqc.tn import TNQMachine, qcircuit, TNQModule,VQC_Purity
        pyvqnet.backends.set_backend("torch")
        from pyvqnet.tensor import QTensor

        x = QTensor([[0.7, 0.4], [1.7, 2.4]], requires_grad=True).toGPU()

        class QM(TNQModule):
            def __init__(self, name=""):
                super().__init__(name)
                self.device = TNQMachine(3)

            def forward(self, x):
                self.device.reset_states(2)
                qcircuit.rx(q_machine=self.device, wires=0, params=x[0])
                qcircuit.ry(q_machine=self.device, wires=1, params=x[1])
                qcircuit.ry(q_machine=self.device, wires=2, params=x[1])
                qcircuit.cnot(q_machine=self.device, wires=[0, 1])
                qcircuit.cnot(q_machine=self.device, wires=[2, 1])
                return VQC_Purity(self.device.get_states(), [0, 1], num_wires=3, use_tn=True)

        model = QM().toGPU()
        y_tn = model(x)
        x.data.retain_grad()
        y_tn.backward()
        print(y_tn)

VQC_VarMeasure
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. py:function:: pyvqnet.qnn.vqc.tn.VQC_VarMeasure(q_machine, obs)

    返回在 ``q_machine`` 中的状态向量上提供的可观测量 ``obs`` 的测量方差。

    :param q_machine: 从 pyqpanda get_qstate() 获取的量子态
    :param obs: 可观测量

    :return: 方差值

    示例::

        import pyvqnet
        from pyvqnet.qnn.vqc.tn import TNQMachine, qcircuit, VQC_VarMeasure, TNQModule,PauliY
        from pyvqnet.tensor import QTensor
        from pyvqnet import kfloat64
        pyvqnet.backends.set_backend("torch")
        x = QTensor([[0.7, 0.4], [0.6, 0.4]], requires_grad=True).toGPU()

        class QM(TNQModule):
            def __init__(self, name=""):
                super().__init__(name)
                self.device = TNQMachine(3)

            def forward(self, x):
                self.device.reset_states(2)
                qcircuit.rx(q_machine=self.device, wires=0, params=x[0])
                qcircuit.ry(q_machine=self.device, wires=1, params=x[1])
                qcircuit.ry(q_machine=self.device, wires=2, params=x[1])
                qcircuit.cnot(q_machine=self.device, wires=[0, 1])
                qcircuit.cnot(q_machine=self.device, wires=[2, 1])
                return VQC_VarMeasure(q_machine= self.device, obs=PauliY(wires=0))

        model = QM().toGPU()
        y = model(x)
        x.data.retain_grad()
        y.backward()
        print(y)

        # [[0.9370641],
        # [0.9516521]]


VQC_DensityMatrixFromQstate
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^


.. py:function:: pyvqnet.qnn.vqc.tn.VQC_DensityMatrixFromQstate(state, indices, use_tn=False)

    计算量子态 ``state`` 在特定量子比特集 ``indices`` 上的密度矩阵。

    :param state: 状态向量的一维列表。该列表的大小应为 ``(2**N,)``\ 。对于 ``N`` 个量子比特，qstate 应从 000 -> 111 开始。
    :param indices: 所考虑子系统中量子比特索引的列表。
    :param use_tn: 使用张量网络需要设置为 True，默认为 False。

    :return: 大小为 "(b, 2**len(indices), 2**len(indices))" 的密度矩阵。

    示例::

        import pyvqnet
        from pyvqnet.tensor import QTensor
        from pyvqnet.qnn.vqc.tn import TNQMachine, qcircuit, VQC_DensityMatrixFromQstate,TNQModule
        pyvqnet.backends.set_backend("torch")
        x = QTensor([[0.7,0.4],[1.7,2.4]], requires_grad=True).toGPU()
        class QM(TNQModule):
            def __init__(self, name=""):
                super().__init__(name=name, use_jit=True)
                self.device = TNQMachine(3)

            def forward(self, x):
                self.device.reset_states(2)
                qcircuit.rx(q_machine=self.device, wires=0, params=x[0])
                qcircuit.ry(q_machine=self.device, wires=1, params=x[1])
                qcircuit.ry(q_machine=self.device, wires=2, params=x[1])
                qcircuit.cnot(q_machine=self.device, wires=[0, 1])
                qcircuit.cnot(q_machine=self.device, wires=[2, 1])
                return VQC_DensityMatrixFromQstate(self.device.get_states(),[0,1],use_tn=True)

        model = QM().toGPU()
        y = model(x)
        x.data.retain_grad()
        y.backward()
        print(y)

        # [[[0.8155131+0.j        0.1718155+0.j        0.       +0.0627175j
        #   0.       +0.2976855j]
        #  [0.1718155+0.j        0.0669081+0.j        0.       +0.0244234j
        #   0.       +0.0627175j]
        #  [0.       -0.0627175j 0.       -0.0244234j 0.0089152+0.j
        #   0.0228937+0.j       ]
        #  [0.       -0.2976855j 0.       -0.0627175j 0.0228937+0.j
        #   0.1086637+0.j       ]]
        #
        # [[0.3362115+0.j        0.1471083+0.j        0.       +0.1674582j
        #   0.       +0.3827205j]
        #  [0.1471083+0.j        0.0993662+0.j        0.       +0.1131119j
        #   0.       +0.1674582j]
        #  [0.       -0.1674582j 0.       -0.1131119j 0.1287589+0.j
        #   0.1906232+0.j       ]
        #  [0.       -0.3827205j 0.       -0.1674582j 0.1906232+0.j
        #   0.4356633+0.j       ]]]


Probability
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^


.. py:class:: pyvqnet.qnn.vqc.tn.Probability(wires=None, name="")

    计算量子电路在特定比特上的概率测量结果。

    此类继承自 ``pyvqnet.qnn.vqc.tn.QModule`` 和 ``torch.nn.Module``\ 。
    此类可作为子模块添加到 torch 模型中。

    :param wires: 测量比特的索引，列表、元组或整数。
    :param name: 模块的名称，默认值：""。
    :return: 测量结果，QTensor。

    示例::

        import pyvqnet
        pyvqnet.backends.set_backend("torch")
        from pyvqnet.qnn.vqc.tn import Probability,rx,ry,cnot,TNQMachine,rz
        from pyvqnet.tensor import QTensor
        from pyvqnet import kfloat64
        x = QTensor([[0.56, 0.1],[0.56, 0.1]],requires_grad=True)
        qm = TNQMachine(4)
        qm.reset_states(2)
        rz(q_machine=qm,wires=0,params=x[:,[0]])
        rz(q_machine=qm,wires=1,params=x[:,[0]])
        cnot(q_machine=qm,wires=[0,1])
        ry(q_machine=qm,wires=2,params=x[:,[1]])
        cnot(q_machine=qm,wires=[0,2])
        rz(q_machine=qm,wires=3,params=x[:,[1]])
        ma = Probability(wires=1)
        y =ma(q_machine=qm)


MeasureAll
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. py:class:: pyvqnet.qnn.vqc.tn.MeasureAll(obs=None, name="")

    计算量子电路的测量结果，支持输入 obs 为多个或单个泡利算符或哈密顿量。

    例如：

    {\'X0\': 0.23} 表示对量子比特 0 施加 PauliX 效应，系数为 0.23。

    {\'X1 Z2\': 2.4,\'Y2\': -0.5} 对应观测值 2.4 * X1 @ Z2 - 0.5 * Y2。

    [{\'X1 Z2\': 4,\'Z1 Z0\': 3},{\'X1 Y2 Z0\': 3.5}] 对应两个观测值 4 * X1 @ Z2 + 3 * Z1 @ Z0 和 3.5 * X1 @ Y2 @ Z0。

    此类继承自 ``pyvqnet.qnn.vqc.tn.QModule`` 和 ``torch.nn.Module``\ 。

    此类可作为子模块添加到 torch 模型中。

    :param obs: 可观测量。
    :param name: 模块名称，默认值：""。
    :return: 测量结果，QTensor。

    示例::

        import pyvqnet
        pyvqnet.backends.set_backend("torch")
        from pyvqnet.qnn.vqc.tn import MeasureAll,rx,ry,cnot,TNQMachine,rz
        from pyvqnet.tensor import QTensor
        from pyvqnet import kfloat64
        x = QTensor([[0.56, 0.1],[0.56, 0.1]],requires_grad=True)
        qm = TNQMachine(4)
        qm.reset_states(2)
        rz(q_machine=qm,wires=0,params=x[:,[0]])
        rz(q_machine=qm,wires=1,params=x[:,[0]])
        cnot(q_machine=qm,wires=[0,1])
        ry(q_machine=qm,wires=2,params=x[:,[1]])
        cnot(q_machine=qm,wires=[0,2])
        rz(q_machine=qm,wires=3,params=x[:,[1]])
        obs_list = [{
            "Z0 Z1" :2
        }, {
            "X0 Z2" :1
        }]
        ma = MeasureAll(obs = obs_list)
        y = ma(q_machine=qm)
        print(y)



Samples
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. py:class:: pyvqnet.qnn.vqc.tn.Samples(wires=None, obs=None, shots = 1,name="")

    在特定线路上使用 shot 获取采样结果。

    此类继承自 ``pyvqnet.qnn.vqc.tn.QModule`` 和 ``torch.nn.Module``\ 。
    此类可作为子模块添加到 torch 模型中。


    :param wires: 采样量子比特索引。默认值：None，运行时使用模拟器的所有比特。
    :param obs: 此值只能为 None。
    :param shots: 采样重复次数，默认值：1。
    :param name: 此模块的名称，默认值：""。
    :return: 一个测量方法类

    示例::

        import pyvqnet
        pyvqnet.backends.set_backend("torch")
        from pyvqnet.qnn.vqc.tn import Samples,rx,ry,cnot,TNQMachine,rz
        from pyvqnet.tensor import QTensor
        from pyvqnet import kfloat64
        x = QTensor([[0.56, 0.1],[0.56, 0.1]],requires_grad=True)

        qm = TNQMachine(4)
        qm.reset_states(2)
        rz(q_machine=qm,wires=0,params=x[:,[0]])
        rx(q_machine=qm,wires=1,params=x[:,[0]])
        cnot(q_machine=qm,wires=[0,1])

        cnot(q_machine=qm,wires=[0,2])
        ry(q_machine=qm,wires=3,params=x[:,[1]])


        ma = Samples(wires=[0,1,2],shots=3)
        y = ma(q_machine=qm)
        print(y)


HermitianExpval
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. py:class:: pyvqnet.qnn.vqc.tn.HermitianExpval(obs=None, name="")

    计算量子电路中埃尔米特量的期望值。

    此类继承自 ``pyvqnet.qnn.vqc.tn.QModule`` 和 ``torch.nn.Module``\ 。
    此类可作为子模块添加到 torch 模型中。


    :param obs: 埃尔米特量。
    :param name: 模块名称，默认值：""。
    :return: 期望结果，QTensor。

    示例::

        import pyvqnet
        pyvqnet.backends.set_backend("torch")
        from pyvqnet.qnn.vqc.tn import TNQMachine, rx,ry,\
            RX, RY, CNOT, PauliX, PauliZ, VQC_RotCircuit,HermitianExpval, TNQModule
        from pyvqnet.tensor import QTensor, tensor
        from pyvqnet.nn import Parameter
        import numpy as np
        bsz = 3
        H = np.array([[8, 4, 0, -6], [4, 0, 4, 0], [0, 4, 8, 0], [-6, 0, 0, 0]])
        class QModel(TNQModule):
            def __init__(self, num_wires, dtype):
                super(QModel, self).__init__()
                self.rot_param = Parameter((3, ))
                self._num_wires = num_wires
                self._dtype = dtype
                self.qm = TNQMachine(num_wires, dtype=dtype)
                self.rx_layer1 = VQC_RotCircuit
                self.ry_layer2 = RY(has_params=True,
                                    trainable=True,
                                    wires=0,
                                    init_params=tensor.QTensor([-0.5]))
                self.xlayer = PauliX(wires=0)
                self.cnot = CNOT(wires=[0, 1])
                self.measure = HermitianExpval(obs = {'wires':(1,0),'observables':tensor.to_tensor(H)})

            def forward(self, x, *args, **kwargs):
                self.qm.reset_states(bsz)

                rx(q_machine=self.qm, wires=0, params=x[1])
                ry(q_machine=self.qm, wires=1, params=x[0])
                self.xlayer(q_machine=self.qm)
                self.rx_layer1(params=self.rot_param, wire=1, q_machine=self.qm)
                self.ry_layer2(q_machine=self.qm)
                self.cnot(q_machine=self.qm)
                rlt = self.measure(q_machine = self.qm)

                return rlt


        input_x = tensor.arange(1, bsz * 2 + 1,
                                dtype=pyvqnet.kfloat32).reshape([bsz, 2])
        input_x.requires_grad = True

        qunatum_model = QModel(num_wires=2, dtype=pyvqnet.kcomplex64)

        batch_y = qunatum_model(input_x)
        batch_y.backward()

        print(batch_y)

量子电路通用模板
--------------------------------------------

VQC_HardwareEfficientAnsatz
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^


.. py:class:: pyvqnet.qnn.vqc.tn.VQC_HardwareEfficientAnsatz(n_qubits,single_rot_gate_list,entangle_gate="CNOT",entangle_rules='linear',depth=1,initial=None,dtype=None)

    论文中介绍的 Hardware Efficient Ansatz 实现：`Hardware-efficient Variational Quantum Eigensolver for Small Molecules <https://arxiv.org/pdf/1704.05018.pdf>`__ 。

    此类继承自 ``pyvqnet.qnn.vqc.tn.QModule`` 和 ``torch.nn.Module``\ 。

    此类可作为子模块添加到 torch 模型中。


    :param n_qubits: 量子比特数量。
    :param single_rot_gate_list: 单一量子比特旋转门列表，由一个或多个作用于每个量子比特的旋转门构建。目前支持 Rx, Ry, Rz。
    :param entangle_gate: 非参数化纠缠门。支持 CNOT, CZ。默认值：CNOT。
    :param entangle_rules: 纠缠门在电路中的使用方式。'linear' 表示纠缠门将作用于每对相邻量子比特。'all' 表示纠缠门将作用于任意两个量子比特。默认值：linear。
    :param depth: ansatz 的深度，默认值：1。
    :param initial: 为所有参数初始化的相同值，默认值：None，此模块将随机初始化参数。
    :param dtype: 参数的数据类型。
    :return: 一个 VQC_HardwareEfficientAnsatz 实例。

    示例::

        from pyvqnet.nn.torch import Linear
        from pyvqnet.qnn.vqc.tn.qcircuit import VQC_HardwareEfficientAnsatz,RZZ,RZ
        from pyvqnet.qnn.vqc.tn import Probability,TNQMachine, TNQModule
        from pyvqnet import tensor
        import pyvqnet
        pyvqnet.backends.set_backend("torch")
        pyvqnet.utils.set_random_seed(25)

        class QM(TNQModule):
            def __init__(self, name=""):
                super().__init__(name)
                self.linearx = Linear(4,2)
                self.ansatz = VQC_HardwareEfficientAnsatz(4, ["rx", "RY", "rz"],
                                            entangle_gate="cnot",
                                            entangle_rules="linear",
                                            depth=2)
                self.encode1 = RZ(wires=0)
                self.encode2 = RZ(wires=1)
                self.measure = Probability(wires=[0, 2])
                self.device = TNQMachine(4)
            def forward(self, x, *args, **kwargs):
                self.device.reset_states(bz)
                y = self.linearx(x)
                self.encode1(params = y[0],q_machine = self.device,)
                self.encode2(params = y[1],q_machine = self.device,)
                self.ansatz(q_machine =self.device)
                return self.measure(q_machine =self.device)

        bz =3
        inputx = tensor.arange(1.0,bz*4+1).reshape([bz,4])
        inputx.requires_grad= True
        qlayer = QM()
        y = qlayer(inputx)
        y.backward()
        print(y)



VQC_BasicEntanglerTemplate
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. py:class:: pyvqnet.qnn.vqc.tn.VQC_BasicEntanglerTemplate(num_layer=1, num_qubits=1, rotation="RX", initial=None, dtype=None)

    该层由每个量子比特上的单参数单量子比特旋转组成，随后在闭合链或环组合中应用多个 CNOT 门。

    CNOT 门的环将每个量子比特连接到其邻居，最后一个量子比特被视为第 a 个量子比特的邻居。

    此类继承自 ``pyvqnet.qnn.vqc.tn.QModule`` 和 ``torch.nn.Module``\ 。

    此类可作为子模块添加到 torch 模型中。

    :param num_layers: 重复层数，默认值：1。
    :param num_qubits: 量子比特数量，默认值：1。
    :param rotation: 使用的单参数单量子比特门，默认值：`RX`
    :param initial: 所有参数的初始化相同值。默认值：None，参数将随机初始化。
    :param dtype: 参数的数据类型，默认值：None，使用 float32。
    :return: 一个 VQC_BasicEntanglerTemplate 实例

    示例::

        import pyvqnet
        pyvqnet.backends.set_backend("torch")
        from pyvqnet.qnn.vqc.tn import TNQModule,\
            VQC_BasicEntanglerTemplate, Probability, TNQMachine
        from pyvqnet import tensor


        class QM(TNQModule):
            def __init__(self, name=""):
                super().__init__(name)

                self.ansatz = VQC_BasicEntanglerTemplate(2,
                                                    4,
                                                    "rz",
                                                    initial=tensor.ones([1, 1]))

                self.measure = Probability(wires=[0, 2])
                self.device = TNQMachine(4)

            def forward(self,x, *args, **kwargs):

                self.ansatz(q_machine=self.device)
                return self.measure(q_machine=self.device)

        bz = 1
        inputx = tensor.arange(1.0, bz * 4 + 1).reshape([bz, 4])
        qlayer = QM()
        y = qlayer(inputx)
        y.backward()
        print(y)



VQC_StronglyEntanglingTemplate
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. py:class:: pyvqnet.qnn.vqc.tn.VQC_StronglyEntanglingTemplate(num_layers=1, num_qubits=1, ranges=None,initial=None, dtype=None)

    由单量子比特旋转和纠缠器组成的层，如 `circuit-centric classifier design <https://arxiv.org/abs/1804.00633>`__ 中所述。

    此类继承自 ``pyvqnet.qnn.vqc.tn.QModule`` 和 ``torch.nn.Module``\ 。
    此类可作为子模块添加到 torch 模型中。


    :param num_layers: 重复层数，默认值：1。
    :param num_qubits: 量子比特数量，默认值：1。
    :param ranges: 确定每个后续层的范围超参数的序列；默认值：None
                                使用 :math: `r=l \mod M` 对于第 :math:`l` 层和 :math:`M` 个量子比特。
    :param initial: 所有参数的初始值。默认值：None，随机初始化。
    :param dtype: 参数的数据类型，默认值：None，使用 float32。
    :return: 一个 VQC_StronglyEntanglingTemplate 实例。

    示例::

        from pyvqnet.nn.torch import TorchModule,Linear,TorchModuleList
        from pyvqnet.qnn.vqc.tn.qcircuit import VQC_StronglyEntanglingTemplate
        from pyvqnet.qnn.vqc.tn import Probability, TNQMachine, TNQModule
        from pyvqnet import tensor
        import pyvqnet

        pyvqnet.backends.set_backend("torch")
        pyvqnet.utils.set_random_seed(25)
        class QM(TNQModule):
            def __init__(self, name=""):
                super().__init__(name)

                self.ansatz = VQC_StronglyEntanglingTemplate(2,
                                                    4,
                                                    None,
                                                    initial=tensor.ones([1, 1]))

                self.measure = Probability(wires=[0, 1])
                self.device = TNQMachine(4)

            def forward(self,x, *args, **kwargs):

                self.ansatz(q_machine=self.device)
                return self.measure(q_machine=self.device)

        bz = 1
        inputx = tensor.arange(1.0, bz * 4 + 1).reshape([bz, 4])
        qlayer = QM()
        y = qlayer(inputx)
        y.backward()
        print(y)



VQC_QuantumEmbedding
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^


.. py:class:: pyvqnet.qnn.vqc.tn.VQC_QuantumEmbedding(qubits, machine, num_repetitions_input, depth_input, num_unitary_layers, num_repetitions,initial = None,dtype = None,name= "")

    使用 RZ,RY,RZ 创建变分量子电路，将经典数据编码为量子态。
    参考 `Quantum embeddings for machine learning <https://arxiv.org/abs/2001.03622>`_。

    此类继承自 ``pyvqnet.qnn.vqc.tn.QModule`` 和 ``torch.nn.Module``\ 。
    此类可作为子模块添加到 torch 模型中。


    :param num_repetitions_input: 在子模块中编码输入的重复次数。
    :param depth_input: 输入维度数量。
    :param num_unitary_layers: 变分量子门的重复次数。
    :param num_repetitions: 子模块的重复次数。
    :param initial: 参数初始化值，默认为 None
    :param dtype: 参数类型，默认为 None，使用 float32。
    :param name: 类名称
    :return: 一个 VQC_QuantumEmbedding 实例。

    示例::

        from pyvqnet.qnn.vqc.tn.qcircuit import VQC_QuantumEmbedding
        from pyvqnet.qnn.vqc.tn import TNQMachine, MeasureAll, TNQModule
        from pyvqnet import tensor
        import pyvqnet

        pyvqnet.backends.set_backend("torch")
        pyvqnet.utils.set_random_seed(25)
        depth_input = 2
        num_repetitions = 2
        num_repetitions_input = 2
        num_unitary_layers = 2
        nq = depth_input * num_repetitions_input
        bz = 12

        class QM(TNQModule):
            def __init__(self, name=""):
                super().__init__(name)

                self.ansatz = VQC_QuantumEmbedding(num_repetitions_input, depth_input,
                                                num_unitary_layers,
                                                num_repetitions, initial=tensor.full([1],12.0),dtype=pyvqnet.kfloat32)

                self.measure = MeasureAll(obs={f"Z{nq-1}":1})
                self.device = TNQMachine(nq)

            def forward(self, x, *args, **kwargs):
                self.device.reset_states(bz)
                self.ansatz(x,q_machine=self.device)
                return self.measure(q_machine=self.device)

        inputx = tensor.arange(1.0, bz * depth_input + 1).reshape([bz, depth_input])
        qlayer = QM()
        y = qlayer(inputx)
        y.backward()
        print(y)


ExpressiveEntanglingAnsatz
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. py:class:: pyvqnet.qnn.vqc.tn.ExpressiveEntanglingAnsatz(type: int, num_wires: int, depth: int, dtype=None, name: str = "")

    来自论文的 19 种不同 ansatz：`Expressibility and entangling capability of parameterized quantum circuits for hybrid quantum-classical algorithms <https://arxiv.org/pdf/1905.10876.pdf>`_。

    此类继承自 ``pyvqnet.qnn.vqc.tn.QModule`` 和 ``torch.nn.Module``\ 。

    此类可作为子模块添加到 torch 模型中。

    :param type: 电路类型从 1 到 19，共 19 种线路。
    :param num_wires: 量子比特数量。
    :param depth: 电路深度。
    :param dtype: 参数的数据类型，默认值：None，使用 float32。
    :param name: 名称，默认为 ""。

    :return:
        一个 ExpressiveEntanglingAnsatz 实例

    示例::

        from pyvqnet.qnn.vqc.tn.qcircuit import ExpressiveEntanglingAnsatz
        from pyvqnet.qnn.vqc.tn import Probability, TNQMachine, MeasureAll, TNQModule
        from pyvqnet import tensor
        import pyvqnet

        pyvqnet.backends.set_backend("torch")
        pyvqnet.utils.set_random_seed(25)

        class QModel(TNQModule):
            def __init__(self, num_wires, dtype):
                super(QModel, self).__init__()

                self._num_wires = num_wires
                self._dtype = dtype
                self.qm = TNQMachine(num_wires, dtype=dtype)
                self.c1 = ExpressiveEntanglingAnsatz(1,3,2)
                self.measure = MeasureAll(obs={
                    "Z1":1
                })

            def forward(self, x, *args, **kwargs):
                self.qm.reset_states(1)
                self.c1(q_machine = self.qm)
                rlt = self.measure(q_machine=self.qm)
                return rlt


        input_x = tensor.QTensor([[0.1, 0.2, 0.3]])

        qunatum_model = QModel(num_wires=3, dtype=pyvqnet.kcomplex64)

        batch_y = qunatum_model(input_x)
        batch_y.backward()
        print(batch_y)


vqc_basis_embedding
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. py:function:: pyvqnet.qnn.vqc.tn.vqc_basis_embedding(basis_state,q_machine)

    将 n 个二进制特征编码为 ``q_machine`` 的 n 量子比特基态。此函数的别名为 `VQC_BasisEmbedding`\ 。

    例如，对于 ``basis_state=([0, 1, 1])``\ ，量子系统中的基态为 :math:`|011 \rangle`\ 。

    :param basis_state: ``(n)`` 大小的二进制输入。
    :param q_machine: 量子机器设备。


    示例::

        import pyvqnet
        pyvqnet.backends.set_backend("torch")
        from pyvqnet.qnn.vqc.tn import vqc_basis_embedding,TNQMachine
        qm  = TNQMachine(3)
        vqc_basis_embedding(basis_state=[1,1,0],q_machine=qm)
        print(qm.get_states())




vqc_angle_embedding
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^


.. py:function:: pyvqnet.qnn.vqc.tn.vqc_angle_embedding(input_feat, wires, q_machine: pyvqnet.qnn.vqc.tn.TNQMachine, rotation: str = "X")

    将 :math:`N` 个特征编码为 :math:`n` 个量子比特的旋转角度，其中 :math:`N \leq n`\ 。
    此函数的别名为 `VQC_AngleEmbedding` 。

    旋转方式可选为：'X'、'Y'、'Z'，由 ``rotation`` 参数定义：

    * ``rotation='X'`` 使用特征作为 RX 旋转的角度。

    * ``rotation='Y'`` 使用特征作为 RY 旋转的角度。

    * ``rotation='Z'`` 使用特征作为 RZ 旋转的角度。

    ``wires`` 表示量子比特上旋转门的索引。

    :param input_feat: 表示参数的数组。
    :param wires: 量子比特索引。
    :param q_machine: 量子机器设备。
    :param rotation: 旋转门，默认为 "X"。

    示例::

        import pyvqnet
        pyvqnet.backends.set_backend("torch")
        from pyvqnet.qnn.vqc.tn import vqc_angle_embedding, TNQMachine
        from pyvqnet.tensor import QTensor
        qm  = TNQMachine(2)
        vqc_angle_embedding(QTensor([2.2, 1]), [0, 1], q_machine=qm, rotation='X')
        print(qm.get_states())
        vqc_angle_embedding(QTensor([2.2, 1]), [0, 1], q_machine=qm, rotation='Y')
        print(qm.get_states())
        vqc_angle_embedding(QTensor([2.2, 1]), [0, 1], q_machine=qm, rotation='Z')
        print(qm.get_states())



vqc_amplitude_embedding
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. py:function:: pyvqnet.qnn.vqc.tn.vqc_amplitude_embedding(input_feature, q_machine)

    将 :math:`2^n` 个特征编码为 :math:`n` 个量子比特的振幅向量。此函数的别名为 `VQC_AmplitudeEmbedding`\ 。

    :param input_feature: 表示参数的 numpy 数组。
    :param q_machine: 量子机器设备。


    示例::

        import pyvqnet
        pyvqnet.backends.set_backend("torch")
        from pyvqnet.qnn.vqc.tn import vqc_amplitude_embedding, TNQMachine
        from pyvqnet.tensor import QTensor
        qm  = TNQMachine(3)
        vqc_amplitude_embedding(QTensor([3.2,-2,-2,0.3,12,0.1,2,-1]), q_machine=qm)
        print(qm.get_states())



vqc_iqp_embedding
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. py:function:: pyvqnet.qnn.vqc.tn.vqc_iqp_embedding(input_feat, q_machine: pyvqnet.qnn.vqc.tn.TNQMachine, rep: int = 1)

    使用 IQP 电路的对角门将 :math:`n` 个特征编码为 :math:`n` 个量子比特。别名：``VQC_IQPEmbedding`` 。

    该编码由 `Havlicek et al. (2018) <https://arxiv.org/pdf/1804.11326.pdf>`_ 提出。

    通过指定 ``rep``\ ，可以重复基本的 IQP 电路。

    :param input_feat: 参数的数组。
    :param q_machine: 量子机器。
    :param rep: 重复量子电路块的次数，默认为 1。

    示例::

        import pyvqnet
        pyvqnet.backends.set_backend("torch")
        from pyvqnet.qnn.vqc.tn import vqc_iqp_embedding, TNQMachine
        from pyvqnet.tensor import QTensor
        qm  = TNQMachine(3)
        vqc_iqp_embedding(QTensor([3.2,-2,-2]), q_machine=qm)
        print(qm.get_states())



vqc_rotcircuit
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. py:function:: pyvqnet.qnn.vqc.tn.vqc_rotcircuit(q_machine, wire, params)

    任意单量子比特旋转量子逻辑门组合。此函数别名：``VQC_RotCircuit`` 。

    .. math::
        R(\phi,\theta,\omega) = RZ(\omega)RY(\theta)RZ(\phi)= \begin{bmatrix}
        e^{-i(\phi+\omega)/2}\cos(\theta/2) & -e^{i(\phi-\omega)/2}\sin(\theta/2) \\
        e^{-i(\phi-\omega)/2}\sin(\theta/2) & e^{i(\phi+\omega)/2}\cos(\theta/2)
        \end{bmatrix}.

    :param q_machine: 量子虚拟机设备。
    :param wire: 量子比特索引。
    :param params: 表示参数 :math:`[\phi, \theta, \omega]`\ 。

    示例::

        import pyvqnet
        pyvqnet.backends.set_backend("torch")
        from pyvqnet.qnn.vqc.tn import vqc_rotcircuit, TNQMachine
        from pyvqnet.tensor import QTensor
        qm  = TNQMachine(3)
        vqc_rotcircuit(q_machine=qm, wire=[1],params=QTensor([2.0,1.5,2.1]))
        print(qm.get_states())


vqc_crot_circuit
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^


.. py:function:: pyvqnet.qnn.vqc.tn.vqc_crot_circuit(para,control_qubits,rot_wire,q_machine)

    受控旋转单量子比特旋转的量子逻辑门组合。此函数别名：``VQC_CRotCircuit`` 。

    .. math::
        CR(\phi, \theta, \omega) = \begin{bmatrix}
        1 & 0 & 0 & 0 \\
        0 & 1 & 0 & 0\\
        0 & 0 & e^{-i(\phi+\omega)/2}\cos(\theta/2) & -e^{i(\phi-\omega)/2}\sin(\theta/2)\\
        0 & 0 & e^{-i(\phi-\omega)/2}\sin(\theta/2) & e^{i(\phi+\omega)/2}\cos(\theta/2)
        \end{bmatrix}.

    :param para: 表示参数的数组。
    :param control_qubits: 控制量子比特索引。
    :param rot_wire: 旋转量子比特索引。
    :param q_machine: 量子机器设备。


    示例::

        import pyvqnet
        pyvqnet.backends.set_backend("torch")
        from pyvqnet.tensor import QTensor
        from pyvqnet.qnn.vqc.tn import vqc_crot_circuit,TNQMachine, MeasureAll
        p = QTensor([2, 3, 4.0])
        qm = TNQMachine(2)
        vqc_crot_circuit(p, 0, 1, qm)
        m = MeasureAll(obs={"Z0": 1})
        exp = m(q_machine=qm)
        print(exp)




vqc_controlled_hadamard
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^


.. py:function:: pyvqnet.qnn.vqc.tn.vqc_controlled_hadamard(wires, q_machine)

    受控 Hadamard 逻辑门量子电路。此函数别名：``VQC_Controlled_Hadamard`` 。

    .. math::
        CH = \begin{bmatrix}
        1 & 0 & 0 & 0 \\
        0 & 1 & 0 & 0 \\
        0 & 0 & \frac{1}{\sqrt{2}} & \frac{1}{\sqrt{2}} \\
        0 & 0 & \frac{1}{\sqrt{2}} & -\frac{1}{\sqrt{2}}
        \end{bmatrix}.

    :param wires: 量子比特索引列表，第一个为控制比特，列表长度为 2。
    :param q_machine: 量子虚拟机设备。

    示例::

        import pyvqnet
        pyvqnet.backends.set_backend("torch")
        from pyvqnet.tensor import QTensor
        from pyvqnet.qnn.vqc.tn import vqc_controlled_hadamard,\
            TNQMachine, MeasureAll

        p = QTensor([0.2, 3, 4.0])
        qm = TNQMachine(3)
        vqc_controlled_hadamard([1, 0], qm)
        m = MeasureAll(obs={"Z0": 1})
        exp = m(q_machine=qm)
        print(exp)


vqc_ccz
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. py:function:: pyvqnet.qnn.vqc.tn.vqc_ccz(wires, q_machine)

    受控-受控-Z 逻辑门。别名：``VQC_CCZ`` 。

    .. math::
        CCZ =
        \begin{pmatrix}
        1 & 0 & 0 & 0 & 0 & 0 & 0 & 0\\
        0 & 1 & 0 & 0 & 0 & 0 & 0 & 0 & 0\\
        0 & 0 & 1 & 0 & 0 & 0 & 0 & 0\\
        0 & 0 & 0 & 1 & 0 & 0 & 0 & 0\\
        0 & 0 & 0 & 0 & 1 & 0 & 0 & 0\\
        0 & 0 & 0 & 0 & 0 & 1 & 0 & 0 & 0\\
        0 & 0 & 0 & 0 & 0 & 1 & 0 & 0\\
        0 & 0 & 0 & 0 & 0 & 0 & 1 & 0 & 0\\
        0 & 0 & 0 & 0 & 0 & 0 & 0 & 1 & 0\\
        0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & -1
        \end{pmatrix}

    :param wires: 量子比特索引列表，第一个为控制比特。列表长度为 3。
    :param q_machine: 量子虚拟机设备。

    示例::

        import pyvqnet
        pyvqnet.backends.set_backend("torch")
        from pyvqnet.tensor import QTensor
        from pyvqnet.qnn.vqc.tn import vqc_ccz,TNQMachine, MeasureAll
        p = QTensor([0.2, 3, 4.0])

        qm = TNQMachine(3)

        vqc_ccz([1, 0, 2], qm)
        m = MeasureAll(obs={"Z0": 1})
        exp = m(q_machine=qm)
        print(exp)



vqc_fermionic_single_excitation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. py:function:: pyvqnet.qnn.vqc.tn.vqc_fermionic_single_excitation(weight, wires, q_machine)

    耦合簇单激发算子的泡利矩阵张量积。矩阵形式如下：

    .. math::
        \hat{U}_{pr}(\theta) = \mathrm{exp} \{ \theta_{pr} (\hat{c}_p^\dagger \hat{c}_r
        -\mathrm{H.c.}) \},

    别名：``VQC_FermionicSingleExcitation`` 。

    :param weight: 量子比特 p 上的参数，仅一个元素。
    :param wires: 区间 [r, p] 中的量子比特索引子集。最小长度必须为 2。第一个索引值解释为 r，最后一个索引值解释为 p。中间索引由 CNOT 门操作，以计算量子比特集的奇偶性。
    :param q_machine: 量子虚拟机设备。



    示例::

        import pyvqnet
        pyvqnet.backends.set_backend("torch")
        from pyvqnet.tensor import QTensor
        from pyvqnet.qnn.vqc.tn import vqc_fermionic_single_excitation,\
            TNQMachine, MeasureAll
        qm = TNQMachine(3)
        p0 = QTensor([0.5])

        vqc_fermionic_single_excitation(p0, [1, 0, 2], qm)
        m = MeasureAll(obs={"Z0": 1})
        exp = m(q_machine=qm)
        print(exp)



vqc_fermionic_double_excitation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^


.. py:function:: pyvqnet.qnn.vqc.tn.vqc_fermionic_double_excitation(weight, wires1, wires2, q_machine)

    耦合簇双激发算子的泡利矩阵张量积指数化，矩阵形式如下：

    .. math::
        \hat{U}_{pqrs}(\theta) = \mathrm{exp} \{ \theta (\hat{c}_p^\dagger \hat{c}_q^\dagger
        \hat{c}_r \hat{c}_s - \mathrm{H.c.}) \},

    其中 :math:`\hat{c}` 和 :math:`\hat{c}^\dagger` 是费米子湮灭和产生算符，索引 :math:`r, s` 和 :math:`p, q` 分别位于占据和空分子轨道上。使用 `Jordan-Wigner 变换
    <https://arxiv.org/abs/1208.5986>`_，上述定义的费米子算符可以用泡利矩阵表示（详见
    `arXiv:1805.04340 <https://arxiv.org/abs/1805.04340>`_）

    .. math::
        \hat{U}_{pqrs}(\theta) = \mathrm{exp} \Big\{
        \frac{i\theta}{8} \bigotimes_{b=s+1}^{r-1} \hat{Z}_b \bigotimes_{a=q+1}^{p-1}
        \hat{Z}_a (\hat{X}_s \hat{X}_r \hat{Y}_q \hat{X}_p +
        \hat{Y}_s \hat{X}_r \hat{Y}_q \hat{Y}_p +\\ \hat{X}_s \hat{Y}_r \hat{Y}_q \hat{Y}_p +
        \hat{X}_s \hat{X}_r \hat{X}_q \hat{Y}_p - \mathrm{H.c.} ) \Big\}

    此函数别名为：``VQC_FermionicDoubleExcitation`` 。

    :param weight: 可变参数
    :param wires1: 表示索引列表区间 [s, r] 中的量子比特子集。第一个索引解释为 s，最后一个索引解释为 r。CNOT 门操作中间索引以计算一组量子比特的奇偶性。
    :param wires2: 表示索引列表区间 [q, p] 中的量子比特子集。第一个根索引解释为 q，最后一个索引解释为 p。CNOT 门操作中间索引以计算一组量子比特的奇偶性。
    :param q_machine: 量子虚拟机设备。



    示例::

        import pyvqnet
        pyvqnet.backends.set_backend("torch")
        from pyvqnet.tensor import QTensor
        from pyvqnet.qnn.vqc.tn import vqc_fermionic_double_excitation,\
            TNQMachine, MeasureAll
        qm = TNQMachine(5)
        p0 = QTensor([0.5])

        vqc_fermionic_double_excitation(p0, [0, 1], [2, 3], qm)
        m = MeasureAll(obs={"Z0": 1})
        exp = m(q_machine=qm)
        print(exp)


vqc_uccsd
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^


.. py:function:: pyvqnet.qnn.vqc.tn.vqc_uccsd(weights, wires, s_wires, d_wires, init_state, q_machine)

    实现酉耦合簇单双激发模拟（UCCSD）。UCCSD 是一种常用于运行量子化学模拟的 VQE 模拟。

    在一阶 Trotter 近似下，UCCSD 酉函数如下：

    .. math::
        \hat{U}(\vec{\theta}) =
        \prod_{p > r} \mathrm{exp} \Big\{\theta_{pr}
        (\hat{c}_p^\dagger \hat{c}_r-\mathrm{H.c.}) \Big\}
        \prod_{p > q > r > s} \mathrm{exp} \Big\{\theta_{pqrs}
        (\hat{c}_p^\dagger \hat{c}_q^\dagger \hat{c}_r \hat{c}_s-\mathrm{H.c.}) \Big\}

    其中 :math:`\hat{c}` 和 :math:`\hat{c}^\dagger` 是费米子湮灭和产生算符，索引 :math:`r, s` 和 :math:`p, q` 分别位于占据和空分子轨道上。（更多详情请参见
    `arXiv:1805.04340 <https://arxiv.org/abs/1805.04340>`_）：

    此函数别名为：``VQC_UCCSD`` 。

    :param weights: 大小为 ``(len(s_wires)+ len(d_wires))`` 的张量，包含输入 Z 旋转 ``FermionicSingleExcitation`` 和 ``FermionicDoubleExcitation`` 的参数 :math:`\theta_{pr}` 和 :math:`\theta_{pqrs}`\ 。
    :param wires: 模板作用的量子比特索引
    :param s_wires: 包含量子比特索引 ``[r,...,p]`` 的列表序列，由单激发 :math:`\vert r, p \rangle = \hat{c}_p^\dagger \hat{c}_r \vert \mathrm{HF} \rangle` 生成，其中 :math:`\vert \mathrm{HF} \rangle` 表示 Hartee-Fock 参考态。
    :param d_wires: 列表序列，每个列表包含两个指定索引 ``[s, ...,r]`` 和 ``[q,..., p]`` 的列表，定义双激发 :math:`\vert s, r, q, p \rangle = \hat{c}_p^\dagger \hat{c}_q^\dagger \hat{c}_r\hat{c}_s \vert \mathrm{HF} \rangle`\ 。
    :param init_state: 长度为 ``len(wires)`` 的占据数向量，表示高频状态。``init_state`` 量子比特的初始化状态。
    :param q_machine: 量子虚拟机设备。

    示例::

        import pyvqnet
        pyvqnet.backends.set_backend("torch")
        from pyvqnet.qnn.vqc.tn import vqc_uccsd, TNQMachine, MeasureAll
        from pyvqnet.tensor import QTensor
        p0 = QTensor([2, 0.5, -0.2, 0.3, -2, 1, 3, 0])
        s_wires = [[0, 1, 2], [0, 1, 2, 3, 4], [1, 2, 3], [1, 2, 3, 4, 5]]
        d_wires = [[[0, 1], [2, 3]], [[0, 1], [2, 3, 4, 5]], [[0, 1], [3, 4]],
                [[0, 1], [4, 5]]]
        qm = TNQMachine(6)

        vqc_uccsd(p0, range(6), s_wires, d_wires, QTensor([1.0, 1, 0, 0, 0, 0]), qm)
        m = MeasureAll(obs={"Z1": 1})
        exp = m(q_machine=qm)
        print(exp)

        # [[0.963802]]


vqc_zfeaturemap
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. py:function:: pyvqnet.qnn.vqc.tn.vqc_zfeaturemap(input_feat, q_machine: pyvqnet.qnn.vqc.tn.TNQMachine, data_map_func=None, rep: int = 2)

    一阶泡利 Z 演化电路。

    对于 3 个量子比特和 2 次重复，电路表示为：

    .. parsed-literal::

        ┌───┐┌──────────────┐┌───┐┌──────────────┐
        ┤ H ├┤ U1(2.0*x[0]) ├┤ H ├┤ U1(2.0*x[0]) ├
        ├───┤├──────────────┤├───┤├──────────────┤
        ┤ H ├┤ U1(2.0*x[1]) ├┤ H ├┤ U1(2.0*x[1]) ├
        ├───┤├──────────────┤├───┤├──────────────┤
        ┤ H ├┤ U1(2.0*x[2]) ├┤ H ├┤ U1(2.0*x[2]) ├
        └───┘└──────────────┘└───┘└──────────────┘

    泡利字符串固定为 ``Z``\ 。因此，一阶展开将是一个不带纠缠门的电路。

    :param input_feat: 表示输入参数的数组。
    :param q_machine: 量子虚拟机。
    :param data_map_func: 参数映射矩阵，一个可调用函数，设计为：``data_map_func = lambda x: x``\ 。
    :param rep: 模块重复的次数。

    示例::

        import pyvqnet
        pyvqnet.backends.set_backend("torch")
        from pyvqnet.qnn.vqc.tn import vqc_zfeaturemap, TNQMachine, hadamard
        from pyvqnet.tensor import QTensor
        qm = TNQMachine(3)
        for i in range(3):
            hadamard(q_machine=qm, wires=[i])
        vqc_zfeaturemap(input_feat=QTensor([[0.1,0.2,0.3]]),q_machine = qm)
        print(qm.get_states())


vqc_zzfeaturemap
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. py:function:: pyvqnet.qnn.vqc.tn.vqc_zzfeaturemap(input_feat, q_machine: pyvqnet.qnn.vqc.tn.TNQMachine, data_map_func=None, entanglement: Union[str, List[List[int]],Callable[[int], List[int]]] = "full",rep: int = 2)

    二阶泡利 Z 演化电路。

    对于 3 个量子比特、1 次重复和线性纠缠，电路表示为：

    .. parsed-literal::


        ┌───┐┌─────────────────┐
        ┤ H ├┤ U1(2.0*φ(x[0])) ├──■────────────────────────────■────────────────────────────────────
        ├───┤├─────────────────┤┌─┴─┐┌──────────────────────┐┌─┴─┐
        ┤ H ├┤ U1(2.0*φ(x[1])) ├┤ X ├┤ U1(2.0*φ(x[0],x[1])) ├┤ X ├──■────────────────────────────■──
        ├───┤├─────────────────┤└───┘└──────────────────────┘└───┘┌─┴─┐┌──────────────────────┐┌─┴─┐
        ┤ H ├┤ U1(2.0*φ(x[2])) ├──────────────────────────────────┤ X ├┤ U1(2.0*φ(x[1],x[2])) ├┤ X ├
        └───┘└─────────────────┘                                  └───┘└──────────────────────┘└───┘

    其中 ``φ`` 是一个经典非线性函数。如果输入两个值，``φ(x,y) = (pi - x)(pi - y)``\ ，如果输入一个值，``φ(x) = x``\ 。使用 ``data_map_func`` 表示如下：

    .. code-block::

        def data_map_func(x):
            coeff = x if x.shape[-1] == 1 else ft.reduce(lambda x, y: (np.pi - x) * (np.pi - y), x)
            return coeff

    :param input_feat: 表示输入参数的数组。
    :param q_machine: 量子虚拟机。
    :param data_map_func: 参数映射矩阵，一个可调用函数。
    :param entanglement: 指定的纠缠结构。
    :param rep: 模块重复次数。

    示例::

        import pyvqnet
        pyvqnet.backends.set_backend("torch")
        from pyvqnet.qnn.vqc.tn import vqc_zzfeaturemap, TNQMachine
        from pyvqnet.tensor import QTensor

        qm = TNQMachine(3)
        vqc_zzfeaturemap(q_machine=qm, input_feat=QTensor([[0.1,0.2,0.3]]))
        print(qm.get_states())


vqc_allsinglesdoubles
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. py:function:: pyvqnet.qnn.vqc.tn.vqc_allsinglesdoubles(weights, q_machine: pyvqnet.qnn.vqc.tn.TNQMachine, hf_state, wires, singles=None, doubles=None)

    在这种情况下，我们有四个单激发和双激发来保持 Hartree-Fock 态的总自旋投影。

    得到的酉矩阵保持粒子数，并准备 n-量子比特系统处于初始 Hartree-Fock 态和其他编码多激发配置的态的叠加态。

    :param weights: 大小为 ``(len(singles) + len(doubles),)`` 的 QTensor，包含依次进入 vqc.qCircuit.single_excitation 和 vqc.qCircuit.double_excitation 操作的角度
    :param q_machine: 量子机器。
    :param hf_state: 长度为 ``len(wires)`` 的占据数向量，表示 Hartree-Fock 态，``hf_state`` 用于初始化线路。
    :param wires: 要作用的量子比特。
    :param singles: 包含由 single_exitation 操作作用的两个量子比特索引的列表序列。
    :param doubles: 包含由 double_exitation 操作作用的两个量子比特索引的列表序列。

    例如，两个电子和六个量子比特的量子电路如下所示：

    .. image:: ./images/all_singles_doubles.png
        :width: 600 px
        :align: center

    |

    示例::

        import pyvqnet
        pyvqnet.backends.set_backend("torch")
        from pyvqnet.qnn.vqc.tn import vqc_allsinglesdoubles, TNQMachine

        from pyvqnet.tensor import QTensor
        qubits = 4
        qm = TNQMachine(qubits)

        vqc_allsinglesdoubles(q_machine=qm, weights=QTensor([0.55, 0.11, 0.53]),
                              hf_state = QTensor([1,1,0,0]), singles=[[0, 2], [1, 3]], doubles=[[0, 1, 2, 3]], wires=[0,1,2,3])
        print(qm.get_states())

vqc_basisrotation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. py:function:: pyvqnet.qnn.vqc.tn.vqc_basisrotation(q_machine: pyvqnet.qnn.vqc.tn.TNQMachine, wires, unitary_matrix: QTensor, check=False)

    实现一个电路，提供可用于执行精确单单元基旋转的集合。该电路源于 `arXiv:1711.04789 <https://arxiv.org/abs/1711.04789>`_ 中给出的单粒子费米子确定的酉变换 :math:`U(u)`

    .. math::
        U(u) = \exp{\left( \sum_{pq} \left[\log u \right]_{pq} (a_p^\dagger a_q - a_q^\dagger a_p) \right)}.

    :math:`U(u)` 通过使用论文 `Optica, 3, 1460 (2016) <https://opg.optica.org/optica/fulltext.cfm?uri=optica-3-12-1460&id=355743>`_ 中给出的方案获得。

    :param q_machine: 量子机器。
    :param wires: 要作用的量子比特。
    :param unitary_matrix: 指定变换基的矩阵。
    :param check: 检查 `unitary_matrix` 是否为酉矩阵。

    示例::

        import pyvqnet

        pyvqnet.backends.set_backend("torch")
        from pyvqnet.qnn.vqc.tn import vqc_basisrotation, TNQMachine
        from pyvqnet.tensor import QTensor
        import numpy as np

        V = np.array([[0.73678 + 0.27511j, -0.5095 + 0.10704j, -0.06847 + 0.32515j],
                    [0.73678 + 0.27511j, -0.5095 + 0.10704j, -0.06847 + 0.32515j],
                    [-0.21271 + 0.34938j, -0.38853 + 0.36497j, 0.61467 - 0.41317j]])

        eigen_vals, eigen_vecs = np.linalg.eigh(V)
        umat = eigen_vecs.T
        wires = range(len(umat))

        qm = TNQMachine(len(umat))

        vqc_basisrotation(q_machine=qm,
                        wires=wires,
                        unitary_matrix=QTensor(umat, dtype=qm.dtype))

        print(qm.get_states())



分布式接口
================================================

分布式相关函数，当使用 ``torch`` 计算后端时，封装了 torch 的 ``torch.distributed`` 接口。

.. note::

    请参考 `torch distributed <https://pytorch.org/docs/stable/distributed.html>`_ 启动分布式方法。
    当使用 CPU 进行分布式时，请使用 ``gloo`` 而不是 ``mpi``\ 。
    当使用 GPU 进行分布式时，请使用 ``nccl``\ 。

    :ref:`vqnet_dist` VQNet 自己的分布式接口不适用于 ``torch`` 计算后端。

CommController
-------------------------

.. py:class:: pyvqnet.distributed.ControlComm.CommController(backend,rank=None,world_size=None)
    :no-index:

    CommController 用于控制 CPU 和 GPU 下的数据通信控制器。通过设置参数 `backend` 生成 CPU（gloo）和 GPU（nccl）控制器。
    此类将调用 backend、rank、world_size 来初始化 ``torch.distributed.init_process_group(backend, rank, world_size)``\ 。

    :param backend: 用于生成 CPU 或 GPU 数据通信控制器，'gloo' 或 'nccl'。
    :param rank: 当前程序的进程号。
    :param world_size: 所有全局进程的数量。

    :return:
        CommController 实例。

    示例::

        from pyvqnet.distributed import CommController
        import pyvqnet
        pyvqnet.backends.set_backend("torch")
        import os
        import multiprocessing as mp


        def init_process(rank, size):
            """ 初始化分布式环境。 """
            os.environ['MASTER_ADDR'] = '127.0.0.1'
            os.environ['MASTER_PORT'] = '29500'
            os.environ['LOCAL_RANK'] = f"{rank}"
            pp = CommController("gloo", rank=rank, world_size=size)

            local_rank = pp.get_rank()
            print(local_rank)


        if __name__ == "__main__":
            world_size = 2
            processes = []
            mp.set_start_method("spawn")
            for rank in range(world_size):
                p = mp.Process(target=init_process, args=(rank, world_size))
                p.start()
                processes.append(p)

            for p in processes:
                p.join()


    .. py:method:: getRank()
        :no-index:

        用于获取当前进程的进程 ID。

        :return: 返回当前进程的进程 ID。

        示例::

            from pyvqnet.distributed import CommController
            import pyvqnet
            pyvqnet.backends.set_backend("torch")
            import os
            import multiprocessing as mp


            def init_process(rank, size):
                """ 初始化分布式环境。 """
                os.environ['MASTER_ADDR'] = '127.0.0.1'
                os.environ['MASTER_PORT'] = '29500'
                os.environ['LOCAL_RANK'] = f"{rank}"
                pp = CommController("gloo", rank=rank, world_size=size)

                local_rank = pp.getRank()
                print(local_rank)


            if __name__ == "__main__":
                world_size = 2
                processes = []
                mp.set_start_method("spawn")
                for rank in range(world_size):
                    p = mp.Process(target=init_process, args=(rank, world_size))
                    p.start()
                    processes.append(p)

                for p in processes:
                    p.join()



    .. py:method:: getSize()
        :no-index:

        用于获取启动的进程总数。

        :return: 返回进程总数。

        示例::

                        from pyvqnet.distributed import CommController
            import pyvqnet
            pyvqnet.backends.set_backend("torch")
            import os
            import multiprocessing as mp


            def init_process(rank, size):
                """ 初始化分布式环境。 """
                os.environ['MASTER_ADDR'] = '127.0.0.1'
                os.environ['MASTER_PORT'] = '29500'
                os.environ['LOCAL_RANK'] = f"{rank}"
                pp = CommController("gloo", rank=rank, world_size=size)

                local_rank = pp.getSize()
                print(local_rank)


            if __name__ == "__main__":
                world_size = 2
                processes = []
                mp.set_start_method("spawn")
                for rank in range(world_size):
                    p = mp.Process(target=init_process, args=(rank, world_size))
                    p.start()
                    processes.append(p)

                for p in processes:
                    p.join()



    .. py:method:: getLocalRank()
        :no-index:

        在每个进程中，通过 ``os.environ['LOCAL_RANK'] = rank`` 获取每台机器的本地进程号。

        需要预先设置环境变量 `LOCAL_RANK`\ 。

        :return: 当前机器上的当前进程号。

        示例::

            from pyvqnet.distributed import CommController
            import pyvqnet
            pyvqnet.backends.set_backend("torch")
            import os
            import multiprocessing as mp


            def init_process(rank, size):
                """ 初始化分布式环境。 """
                os.environ['MASTER_ADDR'] = '127.0.0.1'
                os.environ['MASTER_PORT'] = '29500'
                os.environ['LOCAL_RANK'] = f"{rank}"
                pp = CommController("gloo", rank=rank, world_size=size)

                local_rank = pp.getLocalRank()
                print(local_rank )


            if __name__ == "__main__":
                world_size = 2
                processes = []
                mp.set_start_method("spawn")
                for rank in range(world_size):
                    p = mp.Process(target=init_process, args=(rank, world_size))
                    p.start()
                    processes.append(p)

                for p in processes:
                    p.join()


    .. py:method:: split_groups(rankL)
        :no-index:

        根据输入参数设置的进程号列表，用于划分多个通信组。

        :param rankL: 进程组列表。
        :return: 包含 ``torch.distributed.ProcessGroup`` 的列表

        示例::

            from pyvqnet.distributed import get_local_rank,CommController
            import pyvqnet
            import numpy as np
            from pyvqnet.tensor import tensor
            pyvqnet.backends.set_backend("torch")
            import os
            import multiprocessing as mp


            def init_process(rank, size):
                """ 初始化分布式环境。 """
                os.environ['MASTER_ADDR'] = '127.0.0.1'
                os.environ['MASTER_PORT'] = '29500'
                os.environ['LOCAL_RANK'] = f"{rank}"
                Comm_OP = CommController("gloo", rank=rank, world_size=size)

                group = Comm_OP.split_groups([[1,3]])

                num = tensor.to_tensor(np.random.rand(1, 5)+get_local_rank()*10)
                print(f"before rank {Comm_OP.getRank()}  {num}\n")

                Comm_OP.reduce_group(num, 1,"sum",group[0])
                print(f"after rank {Comm_OP.getRank()}  {num}\n")


            if __name__ == "__main__":
                world_size = 4
                processes = []
                mp.set_start_method("spawn")
                for rank in range(world_size):
                    p = mp.Process(target=init_process, args=(rank, world_size))
                    p.start()
                    processes.append(p)

                for p in processes:
                    p.join()




    .. py:method:: barrier()
        :no-index:

        不同进程的同步。

        :return: 同步操作。

        示例::

            from pyvqnet.distributed import CommController
            import pyvqnet
            pyvqnet.backends.set_backend("torch")
            import os
            import multiprocessing as mp


            def init_process(rank, size):
                """ 初始化分布式环境。 """
                os.environ['MASTER_ADDR'] = '127.0.0.1'
                os.environ['MASTER_PORT'] = '29500'
                os.environ['LOCAL_RANK'] = f"{rank}"
                pp = CommController("gloo", rank=rank, world_size=size)

                pp.barrier()


            if __name__ == "__main__":
                world_size = 4
                processes = []
                mp.set_start_method("spawn")
                for rank in range(world_size):
                    p = mp.Process(target=init_process, args=(rank, world_size))
                    p.start()
                    processes.append(p)

                for p in processes:
                    p.join()

    .. py:method:: allreduce(tensor, c_op = "avg")
        :no-index:

        支持对数据进行 allreduce 通信。

        :param tensor: 输入数据。
        :param c_op: 计算方法。

        示例::

            from pyvqnet.distributed import get_local_rank,CommController
            import pyvqnet
            import numpy as np
            from pyvqnet.tensor import tensor
            pyvqnet.backends.set_backend("torch")
            import os
            import multiprocessing as mp


            def init_process(rank, size):
                """ 初始化分布式环境。 """
                os.environ['MASTER_ADDR'] = '127.0.0.1'
                os.environ['MASTER_PORT'] = '29500'
                os.environ['LOCAL_RANK'] = f"{rank}"
                Comm_OP = CommController("gloo", rank=rank, world_size=size)

                num = tensor.to_tensor(np.random.rand(1, 5))
                print(f"rank {Comm_OP.getRank()}  {num}")

                Comm_OP.all_reduce(num, "sum")
                print(f"rank {Comm_OP.getRank()}  {num}")

            if __name__ == "__main__":
                world_size = 2
                processes = []
                mp.set_start_method("spawn")
                for rank in range(world_size):
                    p = mp.Process(target=init_process, args=(rank, world_size))
                    p.start()
                    processes.append(p)

                for p in processes:
                    p.join()

    .. py:method:: reduce(tensor, root = 0, c_op = "avg")
        :no-index:

        支持对数据进行 reduce 通信。

        :param tensor: 输入数据。
        :param root: 指定返回数据的节点。
        :param c_op: 计算方法。

        示例::

            from pyvqnet.distributed import get_local_rank,CommController
            import pyvqnet
            import numpy as np
            from pyvqnet.tensor import tensor
            pyvqnet.backends.set_backend("torch")
            import os
            import multiprocessing as mp


            def init_process(rank, size):
                """ 初始化分布式环境。 """
                os.environ['MASTER_ADDR'] = '127.0.0.1'
                os.environ['MASTER_PORT'] = '29500'
                os.environ['LOCAL_RANK'] = f"{rank}"
                Comm_OP = CommController("gloo", rank=rank, world_size=size)

                num = tensor.to_tensor(np.random.rand(1, 5))
                print(f"before rank {Comm_OP.getRank()}  {num}")

                Comm_OP.reduce(num, 1,"sum")
                print(f"after rank {Comm_OP.getRank()}  {num}")


            if __name__ == "__main__":
                world_size = 3
                processes = []
                mp.set_start_method("spawn")
                for rank in range(world_size):
                    p = mp.Process(target=init_process, args=(rank, world_size))
                    p.start()
                    processes.append(p)

                for p in processes:
                    p.join()


    .. py:method:: broadcast(tensor, root = 0)
        :no-index:

        将指定进程 root 上的数据广播到所有进程。

        :param tensor: 输入数据。
        :param root: 指定的节点。

        示例::

            from pyvqnet.distributed import get_local_rank,CommController
            import pyvqnet
            import numpy as np
            from pyvqnet.tensor import tensor
            pyvqnet.backends.set_backend("torch")
            import os
            import multiprocessing as mp


            def init_process(rank, size):
                """ 初始化分布式环境。 """
                os.environ['MASTER_ADDR'] = '127.0.0.1'
                os.environ['MASTER_PORT'] = '29500'
                os.environ['LOCAL_RANK'] = f"{rank}"
                Comm_OP = CommController("gloo", rank=rank, world_size=size)


                num = tensor.to_tensor(np.random.rand(1, 5))+ rank
                print(f"before rank {Comm_OP.getRank()}  {num}")

                Comm_OP.broadcast(num, 1)
                print(f"after rank {Comm_OP.getRank()}  {num}")


            if __name__ == "__main__":
                world_size = 3
                processes = []
                mp.set_start_method("spawn")
                for rank in range(world_size):
                    p = mp.Process(target=init_process, args=(rank, world_size))
                    p.start()
                    processes.append(p)

                for p in processes:
                    p.join()


    .. py:method:: allgather(tensor)
        :no-index:

        将所有进程的所有数据收集在一起。此接口仅支持 nccl 后端。

        :param tensor: 输入数据。

        示例::

            from pyvqnet.distributed import get_local_rank,CommController,get_world_size
            import pyvqnet
            import numpy as np
            from pyvqnet.tensor import tensor
            pyvqnet.backends.set_backend("torch")
            import os
            import multiprocessing as mp


            def init_process(rank, size):
                """ 初始化分布式环境。 """
                os.environ['MASTER_ADDR'] = '127.0.0.1'
                os.environ['MASTER_PORT'] = '29500'
                os.environ['LOCAL_RANK'] = f"{rank}"
                Comm_OP = CommController("nccl", rank=rank, world_size=size)

                num = tensor.QTensor(np.random.rand(5,4),device=pyvqnet.DEV_GPU_0+rank)
                print(f"before rank {Comm_OP.getRank()}  {num}\n")

                num = Comm_OP.all_gather(num)
                print(f"after rank {Comm_OP.getRank()}  {num}\n")


            if __name__ == "__main__":
                world_size = 2
                processes = []
                mp.set_start_method("spawn")
                for rank in range(world_size):
                    p = mp.Process(target=init_process, args=(rank, world_size))
                    p.start()
                    processes.append(p)

                for p in processes:
                    p.join()


    .. py:method:: send(tensor, dest)
        :no-index:

        p2p 通信接口。

        :param tensor: 输入数据。
        :param dest: 目标进程。

        示例::

            from pyvqnet.distributed import get_rank,CommController,get_world_size
            import pyvqnet
            import numpy as np
            from pyvqnet.tensor import tensor
            pyvqnet.backends.set_backend("torch")
            import os
            import multiprocessing as mp


            def init_process(rank, size):
                """ 初始化分布式环境。 """
                os.environ['MASTER_ADDR'] = '127.0.0.1'
                os.environ['MASTER_PORT'] = '29500'
                os.environ['LOCAL_RANK'] = f"{rank}"
                Comm_OP = CommController("gloo", rank=rank, world_size=size)

                num = tensor.to_tensor(np.random.rand(1, 5))
                recv = tensor.zeros_like(num)
                if get_rank() == 0:
                    Comm_OP.send(num, 1)
                elif get_rank() == 1:
                    Comm_OP.recv(recv, 0)
                print(f"before rank {Comm_OP.getRank()}  {num}")
                print(f"after rank {Comm_OP.getRank()}  {recv}")

            if __name__ == "__main__":
                world_size = 2
                processes = []
                mp.set_start_method("spawn")
                for rank in range(world_size):
                    p = mp.Process(target=init_process, args=(rank, world_size))
                    p.start()
                    processes.append(p)

                for p in processes:
                    p.join()


    .. py:method:: recv(tensor, source)
        :no-index:

        p2p 通信接口。

        :param tensor: 输入数据。
        :param source: 接收进程。

        示例::

            from pyvqnet.distributed import get_rank,CommController,get_world_size
            import pyvqnet
            import numpy as np
            from pyvqnet.tensor import tensor
            pyvqnet.backends.set_backend("torch")
            import os
            import multiprocessing as mp


            def init_process(rank, size):
                """ 初始化分布式环境。 """
                os.environ['MASTER_ADDR'] = '127.0.0.1'
                os.environ['MASTER_PORT'] = '29500'
                os.environ['LOCAL_RANK'] = f"{rank}"
                Comm_OP = CommController("gloo", rank=rank, world_size=size)

                num = tensor.to_tensor(np.random.rand(1, 5))
                recv = tensor.zeros_like(num)
                if get_rank() == 0:
                    Comm_OP.send(num, 1)
                elif get_rank() == 1:
                    Comm_OP.recv(recv, 0)
                print(f"before rank {Comm_OP.getRank()}  {num}")
                print(f"after rank {Comm_OP.getRank()}  {recv}")

            if __name__ == "__main__":
                world_size = 2
                processes = []
                mp.set_start_method("spawn")
                for rank in range(world_size):
                    p = mp.Process(target=init_process, args=(rank, world_size))
                    p.start()
                    processes.append(p)

                for p in processes:
                    p.join()


    .. py:method:: allreduce_group(tensor, c_op = "avg", group = None)
        :no-index:

        组内 allreduce 通信接口。

        :param tensor: 输入数据。
        :param c_op: 计算方法。
        :param group: 从 `split_groups` 或 `init_groups` 生成的通信组。

        示例::

            from pyvqnet.distributed import get_local_rank,CommController
            import pyvqnet
            import numpy as np
            from pyvqnet.tensor import tensor
            pyvqnet.backends.set_backend("torch")
            import os
            import multiprocessing as mp


            def init_process(rank, size):
                """ 初始化分布式环境。 """
                os.environ['MASTER_ADDR'] = '127.0.0.1'
                os.environ['MASTER_PORT'] = '29500'
                os.environ['LOCAL_RANK'] = f"{rank}"
                Comm_OP = CommController("gloo", rank=rank, world_size=size)

                rankL = [[0,1],[2,3]]
                groups = Comm_OP.split_groups(rankL)
                num = tensor.to_tensor(np.ones(5)+get_local_rank()*1000)

                print(f"before rank {Comm_OP.getRank()}  {num}")
                if Comm_OP.getRank() in rankL[0]:
                    Comm_OP.all_reduce_group(num, "sum", groups[0])

                    print(f"after rank {Comm_OP.getRank()}  {num}")

                if Comm_OP.getRank() in rankL[1]:
                    Comm_OP.all_reduce_group(num, "sum", groups[1])

                    print(f"after rank {Comm_OP.getRank()}  {num}")

            if __name__ == "__main__":
                world_size = 4
                mp.set_start_method("spawn")
                processes = []
                for rank in range(world_size):
                    p = mp.Process(target=init_process, args=(rank, world_size))
                    p.start()
                    processes.append(p)

                for p in processes:
                    p.join()



    .. py:method:: reduce_group(tensor, root = 0, c_op = "avg", group = None)
        :no-index:

        组内 reduce 通信接口。

        :param tensor: 输入数据。
        :param root: 指定进程号。
        :param c_op: 计算方法。
        :param group: 从 `split_groups` 或 `init_groups` 生成的通信组。

        示例::

            from pyvqnet.distributed import get_local_rank,CommController
            import pyvqnet
            import numpy as np
            from pyvqnet.tensor import tensor
            pyvqnet.backends.set_backend("torch")
            import os
            import multiprocessing as mp


            def init_process(rank, size):
                """ 初始化分布式环境。 """
                os.environ['MASTER_ADDR'] = '127.0.0.1'
                os.environ['MASTER_PORT'] = '29500'
                os.environ['LOCAL_RANK'] = f"{rank}"
                Comm_OP = CommController("gloo", rank=rank, world_size=size)
                rankL = [[1,3],[0,2]]
                group = Comm_OP.split_groups([[1,3],[0,2]])

                num = tensor.to_tensor(np.random.rand(1, 5)+get_local_rank()*10)
                print(f"before rank {Comm_OP.getRank()}  {num}\n")
                if Comm_OP.getRank() in rankL[0]:
                    Comm_OP.reduce_group(num, rankL[0][1],"sum",group[0])
                    print(f"after rank {Comm_OP.getRank()}  {num}\n")
                if Comm_OP.getRank() in rankL[1]:
                    Comm_OP.reduce_group(num, rankL[1][1],"sum",group[1])
                    print(f"after rank {Comm_OP.getRank()}  {num}\n")

            if __name__ == "__main__":
                world_size = 4
                processes = []
                mp.set_start_method("spawn")
                for rank in range(world_size):
                    p = mp.Process(target=init_process, args=(rank, world_size))
                    p.start()
                    processes.append(p)

                for p in processes:
                    p.join()


    .. py:method:: broadcast_group(tensor, root = 0, group = None)
        :no-index:

        组内 broadcast 通信接口。

        :param tensor: 输入数据。
        :param root: 指定进程 ID。
        :param group: 从 `split_groups` 或 `init_groups` 生成的通信组。

        示例::

            from pyvqnet.distributed import get_local_rank,CommController
            import pyvqnet
            import numpy as np
            from pyvqnet.tensor import tensor
            pyvqnet.backends.set_backend("torch")
            import os
            import multiprocessing as mp


            def init_process(rank, size):
                """ 初始化分布式环境。 """
                os.environ['MASTER_ADDR'] = '127.0.0.1'
                os.environ['MASTER_PORT'] = '29500'
                os.environ['LOCAL_RANK'] = f"{rank}"
                Comm_OP = CommController("gloo", rank=rank, world_size=size)

                rankL = [[2,3],[0,1,4]]
                group = Comm_OP.split_groups(rankL)

                num = tensor.to_tensor(np.random.rand(1, 5))+ rank*1000
                print(f"before rank {Comm_OP.getRank()}  {num}")

                if Comm_OP.getRank() in rankL[0]:
                    Comm_OP.broadcast_group(num, rankL[0][0],group[0])
                    print(f"after rank {Comm_OP.getRank()}  {num}")

                if Comm_OP.getRank() in rankL[1]:
                    Comm_OP.broadcast_group(num, rankL[1][1],group[1])
                    print(f"after rank {Comm_OP.getRank()}  {num}")

            if __name__ == "__main__":
                world_size = 5
                processes = []
                mp.set_start_method("spawn")
                for rank in range(world_size):
                    p = mp.Process(target=init_process, args=(rank, world_size))
                    p.start()
                    processes.append(p)

                for p in processes:
                    p.join()


    .. py:method:: allgather_group(tensor, group = None)
        :no-index:

        组内 allgather 通信接口。

        :param tensor: 输入数据。
        :param group: 从 `split_groups` 或 `init_groups` 生成的通信组。

        示例::

            from pyvqnet.distributed import get_local_rank,CommController,get_world_size
            import pyvqnet
            import numpy as np
            from pyvqnet.tensor import tensor
            pyvqnet.backends.set_backend("torch")
            import os
            import multiprocessing as mp


            def init_process(rank, size ):
                """ 初始化分布式环境。 """
                os.environ['MASTER_ADDR'] = '127.0.0.1'
                os.environ['MASTER_PORT'] = '29500'
                os.environ['LOCAL_RANK'] = f"{rank}"
                Comm_OP = CommController("nccl", rank=rank, world_size=size)

                group = Comm_OP.split_groups([[0,1]])
                print(f"get_world_size {get_world_size()}")

                num = tensor.QTensor(np.random.rand(5,4)+get_local_rank()*100,device=pyvqnet.DEV_GPU_0+get_local_rank())
                print(f"before rank {Comm_OP.getRank()}  {num}\n")

                num = Comm_OP.allgather_group(num,group[0])
                print(f"after rank {Comm_OP.getRank()}  {num}\n")


            if __name__ == "__main__":
                world_size = 2
                processes = []
                mp.set_start_method("spawn")
                for rank in range(world_size):
                    p = mp.Process(target=init_process, args=(rank, world_size ))
                    p.start()
                    processes.append(p)

                for p in processes:
                    p.join()
