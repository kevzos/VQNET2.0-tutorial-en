经典神经网络模块
#########################################

以下经典神经网络模块支持自动反向传播计算。在执行前向函数后，可以通过执行反向函数来计算梯度。下面对卷积层给出一个简单示例：

.. code-block::

    from pyvqnet.tensor import arange
    from pyvqnet import kfloat32
    from pyvqnet.nn import Conv2D

    # an image feed into two dimension convolution layer
    b = 2        # batch size
    ic = 2       # input channels
    oc = 2      # output channels
    hw = 4      # input width and heights

    # two dimension convolution layer
    test_conv = Conv2D(ic,oc,(2,2),(2,2),"same")

    # input of shape [b,ic,hw,hw]
    x0 = arange(1,b*ic*hw*hw+1,requires_grad=True,dtype=kfloat32)

    x1 = x0.reshape([b,ic,hw,hw])
    #forward function
    x = test_conv(x1)

    #backward function with autograd
    x.backward()
    print(x0.grad)

    # [
    # [[[0.0958736, 0.3032238, 0.0958736, 0.3032238],
    #  [-0.2665333, 0.1081382, -0.2665333, 0.1081382],
    #  [0.0958736, 0.3032238, 0.0958736, 0.3032238],
    #  [-0.2665333, 0.1081382, -0.2665333, 0.1081382]],
    # [[-0.0068994, 0.0914679, -0.0068994, 0.0914679],
    #  [-0.2820665, 0.3160213, -0.2820665, 0.3160213],
    #  [-0.0068994, 0.0914679, -0.0068994, 0.0914679],
    #  [-0.2820665, 0.3160213, -0.2820665, 0.3160213]]],
    # [[[0.0958736, 0.3032238, 0.0958736, 0.3032238],
    #  [-0.2665333, 0.1081382, -0.2665333, 0.1081382],
    #  [0.0958736, 0.3032238, 0.0958736, 0.3032238],
    #  [-0.2665333, 0.1081382, -0.2665333, 0.1081382]],
    # [[-0.0068994, 0.0914679, -0.0068994, 0.0914679],
    #  [-0.2820665, 0.3160213, -0.2820665, 0.3160213],
    #  [-0.0068994, 0.0914679, -0.0068994, 0.0914679],
    #  [-0.2820665, 0.3160213, -0.2820665, 0.3160213]]]
    # ]

.. currentmodule:: pyvqnet.nn


模块类
********************************************************

抽象计算模块


模块
=================================

.. py:class:: pyvqnet.nn.module.Module

    所有神经网络模块（包括量子模块和经典模块）的基类。
    您的模型也应继承此类以实现自动梯度计算。

    模块也可以包含其他模块，从而允许以树结构嵌套它们。
    您可以将子模块作为常规属性进行赋值::

        class Model(Module):
            def __init__(self):
                super(Model, self).__init__()
                self.conv1 = pyvqnet.nn.Conv2d(1, 20, (5,5))
                self.conv2 = pyvqnet.nn.Conv2d(20, 20, (5,5))

            def forward(self, x):
                x = pyvqnet.nn.activation.relu(self.conv1(x))
                return pyvqnet.nn.activation.relu(self.conv2(x))

    以这种方式赋值的子模块将被注册

前向传播
=================================

.. py:method:: pyvqnet.nn.module.Module.forward(x, *args, **kwargs)

    执行前向传播的抽象方法。

    :param x: 输入QTensor
    :param \*args: 非关键字可变参数
    :param \*\*kwargs: 关键字可变参数
    :return: 模块输出

    Example::

        import numpy as np
        from pyvqnet.tensor import QTensor
        import pyvqnet as vq
        from pyvqnet.nn import Conv2D
        b = 2
        ic = 3
        oc = 2
        test_conv = Conv2D(ic, oc, (3, 3), (2, 2), "same")
        x0 = QTensor(np.arange(1, b * ic * 5 * 5 + 1).reshape([b, ic, 5, 5]),
                    requires_grad=True,
                    dtype=vq.kfloat32)
        x = test_conv.forward(x0)
        print(x)

        # [
        # [[[4.3995643, 3.9317808, -2.0707254],
        #  [20.1951981, 21.6946659, 14.2591858],
        #  [38.4702759, 31.9730244, 24.5977650]],
        # [[-17.0607567, -31.5377998, -7.5618000],
        #  [-22.5664024, -40.3876266, -15.1564388],
        #  [-3.1080279, -18.5986233, -8.0648050]]],
        # [[[6.6493244, -13.4840755, -20.2554188],
        #  [54.4235802, 34.4462433, 26.8171902],
        #  [90.2827682, 62.9092331, 51.6892929]],
        # [[-22.3385429, -45.2448578, 5.7101378],
        #  [-32.9464149, -60.9557228, -10.4994345],
        #  [5.9029331, -20.5480480, -0.9379558]]]
        # ]

状态字典
=================================

.. py:method:: pyvqnet.nn.module.Module.state_dict(destination=None, prefix='')

    返回包含模块完整状态的字典。

    包括参数和持久性缓冲区（例如运行平均值）。
    键是对应的参数和缓冲区名称。

    :param destination: 存储状态的字典
    :param prefix: 用于此模块中参数和缓冲区的前缀
        module

    :return: 包含模块完整状态的字典

    Example::

        from pyvqnet.nn import Conv2D
        test_conv = Conv2D(2,3,(3,3),(2,2),"same")
        print(test_conv.state_dict().keys())
        #odict_keys(['weights', 'bias'])


转至GPU
=================================

.. py:function:: pyvqnet.nn.module.Module.toGPU(device: int = DEV_GPU_0)

    将模块及其子模块的参数和缓冲区数据移动到指定的GPU设备。

    device指定用于存储内部数据的设备。当device >= DEV_GPU_0时，数据存储在GPU上。如果您的计算机有多个GPU，
    您可以指定不同的设备来存储数据。例如，device = DEV_GPU_1 , DEV_GPU_2, DEV_GPU_3, ... 表示存储在不同编号的GPU上。
    
    .. note::
        模块不能在不同GPU上进行计算。如果您尝试在编号超过已验证GPU最大数量的GPU上创建QTensor，将引发Cuda错误。

    :param device: 当前保存QTensor的设备，默认=DEV_GPU_0。device = pyvqnet.DEV_GPU_0，存储在第一块GPU上，device = DEV_GPU_1，存储在第二块GPU上，以此类推。
    :return: 已移动到GPU设备的模块。

    Examples::

        from pyvqnet.nn.conv import ConvT2D 
        test_conv = ConvT2D(3, 2, [4,4], [2, 2], "same")
        test_conv = test_conv.toGPU()
        print(test_conv.backend)
        #1000


转至CPU
=================================

.. py:function:: pyvqnet.nn.module.Module.toCPU()

    将模块及其子模块的参数和缓冲区数据移动到指定的CPU设备。

    :return: 已移动到CPU设备的模块。

    Examples::

        from pyvqnet.nn.conv import ConvT2D 
        test_conv = ConvT2D(3, 2, [4,4], [2, 2], "same")
        test_conv = test_conv.toCPU()
        print(test_conv.backend)
        #0

.. _save_parameters:

保存参数
=================================

.. py:function:: pyvqnet.utils.storage.save_parameters(obj, f)

    将模型参数保存到磁盘文件。

    :param obj: 从 ``state_dict()`` 保存的OrderedDict
    :param f: 包含文件名的字符串或os.PathLike对象
    :return: None

    Example::

        from pyvqnet.nn import Module,Conv2D
        import pyvqnet
        class Net(Module):
            def __init__(self):
                super(Net, self).__init__()
                self.conv1 = Conv2D(input_channels=1, output_channels=6, kernel_size=(5, 5), stride=(1, 1), padding="valid")

            def forward(self, x):
                return super().forward(x)

        model = Net()
        pyvqnet.utils.storage.save_parameters(model.state_dict(),"tmp.model")

加载参数
=================================

.. py:function:: pyvqnet.utils.storage.load_parameters(f)

    从磁盘文件加载模型参数。

    应首先创建模型实例。

    :param f: 包含文件名的字符串或os.PathLike对象
    :return: 用于 ``load_state_dict()`` 的已保存OrderedDict

    Example::

        from pyvqnet.nn import Module,Conv2D
        import pyvqnet

        class Net(Module):
            def __init__(self):
                super(Net, self).__init__()
                self.conv1 = Conv2D(input_channels=1, output_channels=6, kernel_size=(5, 5), stride=(1, 1), padding="valid")

            def forward(self, x):
                return super().forward(x)

        model = Net()
        model1 = Net()  # another Module object
        pyvqnet.utils.storage.save_parameters( model.state_dict(),"tmp.model")
        model_para =  pyvqnet.utils.storage.load_parameters("tmp.model")
        model1.load_state_dict(model_para)



模块列表
**************************************************************************************************************************************************************************

.. py:class:: pyvqnet.nn.module.ModuleList([pyvqnet.nn.module.Module])


    将子模块保存在列表中。ModuleList可以像普通Python列表一样进行索引，并且其包含的Module的内部参数可以被保存。

    :param modules: nn.Module列表

    :return: 模块列表

    Example::

        from pyvqnet.tensor import *
        from pyvqnet.nn import Module,Linear,ModuleList
        from pyvqnet.qnn import ProbsMeasure,QuantumLayer
        import pyqpanda as pq
        def pqctest (input,param,qubits,cbits,m_machine):
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


        class M(Module):
            def __init__(self):
                super(M, self).__init__()
                self.pqc2 = ModuleList([QuantumLayer(pqctest,3,"cpu",4,1), Linear(4,1)
                ])

            def forward(self, x, *args, **kwargs):
                y = self.pqc2[0](x)  + self.pqc2[1](x)
                return y

        mm = M()
        print(mm.state_dict().keys())
        #odict_keys(['pqc2.0.m_para', 'pqc2.1.weights', 'pqc2.1.bias'])



参数列表
*********************************************************
.. py:class:: pyvqnet.nn.module.ParameterList([pyvqnet.nn.module.Module])


    将参数存储在列表中。ParameterList可以像普通Python列表一样进行索引，并且其包含的Parameter的内部参数可以被存储。

    :param modules: nn.Parameter列表。

    :return: 参数列表。

    Example::

        from pyvqnet import nn
        class MyModule(nn.Module):
            def __init__(self):
                super().__init__()
                self.params = nn.ParameterList([nn.Parameter((10, 10)) for i in range(10)])
            def forward(self, x):

                # ParameterList can act as an iterable, or be indexed using ints
                for i, p in enumerate(self.params):
                    x = self.params[i // 2] * x + p * x
                return x

        model = MyModule()
        print(model.state_dict().keys())


Sequential
序列容器
.. py:class:: pyvqnet.nn.module.Sequential([pyvqnet.nn.module.Module])

    模块将按照传入的顺序依次添加。或者可以传入一个 ``OrderedDict`` 的模块。 ``Sequential`` 的 ``forward()`` 方法接受任何输入并将其转发给第一个模块。
    然后将输出依次作为后续每个模块的输入，最后返回最后一个模块的输出。

    :param modules: 要追加的模块。

    :return: Sequential。

    Example::
        
        from pyvqnet import nn
        from collections import OrderedDict

        # Using Sequential to create a small model.
        model = nn.Sequential(
                  nn.Conv2D(1,20,(5, 5)),
                  nn.ReLu(),
                  nn.Conv2D(20,64,(5, 5)),
                  nn.ReLu()
                )
        print(model.state_dict().keys())

        # Using Sequential with OrderedDict. This is functionally the same as the above code
                
        model = nn.Sequential(OrderedDict([
                  ('conv1', nn.Conv2D(1,20,(5, 5))),
                  ('relu1', nn.ReLu()),
                  ('conv2', nn.Conv2D(20,64,(5, 5))),
                  ('relu2', nn.ReLu())
                ]))
        print(model.state_dict().keys())


经典神经网络层
********************************************************

一维卷积
=================================

.. py:class:: pyvqnet.nn.Conv1D(input_channels:int,output_channels:int,kernel_size:int ,stride:int= 1,padding = "valid",use_bias:str = True,kernel_initializer = None,bias_initializer =None, dilation_rate: int = 1, group: int = 1, dtype=None, name='')

    对输入应用一维卷积核。卷积模块的输入形状为 (batch_size, input_channels, height)

    :param input_channels: `int` - 输入通道数
    :param output_channels: `int` - 卷积核数量
    :param kernel_size: `int` - 单个卷积核的大小。kernel shape = [output_channels,input_channels/group,kernel_size,1]
    :param stride: `int` - 步幅，默认为1
    :param padding: `str|int` - 填充选项，可以是字符串{'valid', 'same'}或表示隐式填充量的整数。默认为"valid"。
    :param use_bias: `bool` - 是否使用偏置，默认为True
    :param kernel_initializer: `callable` - 默认为None
    :param bias_initializer: `callable` - 默认为None
    :param dilation_rate: `int` - 膨胀率，默认为1
    :param group: `int` - 分组卷积的组数。默认为1
    :param dtype: 参数的数据类型，默认为None，使用默认数据类型kfloat32，表示32位浮点数。
    :param name: 模块名称，默认为""。
    :return: 一个Conv1D类

    .. note::
        ``padding='valid'`` 等同于无填充。

        ``padding='same'`` 对输入进行填充，使输出与输入具有相同的形状。

    Example::

        import numpy as np
        from pyvqnet.tensor import QTensor
        from pyvqnet.nn import Conv1D
        import pyvqnet
        b= 2
        ic =3
        oc = 2
        test_conv = Conv1D(ic,oc,3,2,"same")
        x0 = QTensor(np.arange(1,b*ic*5*5 +1).reshape([b,ic,25]),requires_grad=True,dtype=pyvqnet.kfloat32)
        x = test_conv.forward(x0)
        print(x)

        # [
        # [[12.4438553, 14.8618164, 15.5595102, 16.2572021, 16.9548950, 17.6525879, 18.3502808, 19.0479736, 19.7456665, 20.4433594, 21.1410522, 21.8387432, 10.5725441],
        #  [-13.7539215, 1.0263026, 1.2747254, 1.5231485, 1.7715728, 2.0199962, 2.2684195, 2.5168428, 2.7652662, 3.0136888, 3.2621140, 3.5105357, 14.0515862]],
        # [[47.4924164, 41.0252953, 41.7229881, 42.4206772, 43.1183739, 43.8160667, 44.5137596, 45.2114487, 45.9091415, 46.6068344, 47.3045311, 48.0022240, 18.3216572],
        #  [-47.2381554, 10.3421783, 10.5906038, 10.8390274, 11.0874519, 11.3358765, 11.5842953, 11.8327246, 12.0811434, 12.3295631, 12.5779924, 12.8264122, 39.4719162]]
        # ]

二维卷积
=================================

.. py:class:: pyvqnet.nn.Conv2D(input_channels:int,output_channels:int,kernel_size:tuple,stride:tuple=(1, 1),padding="valid",use_bias = True,kernel_initializer=None,bias_initializer=None, dilation_rate: int = 1, group: int = 1, dtype = None, name = "")

    对输入应用二维卷积核。卷积模块的输入形状为 (batch_size, input_channels, height, width)

    :param input_channels: `int` - 输入通道数
    :param output_channels: `int` - 卷积核数量
    :param kernel_size: `tuple|list` - 单个卷积核的大小。kernel shape = [output_channels,input_channels/group,kernel_size,kernel_size]
    :param stride: `tuple|list` - 步幅，默认为 (1, 1)|[1,1]
    :param padding: `str|tuple` - 填充选项，可以是字符串{'valid', 'same'}或表示两侧隐式填充量的整数元组。默认为"valid"。
    :param use_bias: `bool` - 是否使用偏置，默认为True
    :param kernel_initializer: `callable` - 默认为None
    :param bias_initializer: `callable` - 默认为None
    :param dilation_rate: `int` - 膨胀率，默认为1
    :param group: `int` - 分组卷积的组数。默认为1。
    :param dtype: 参数的数据类型，默认为None，使用默认数据类型kfloat32，表示32位浮点数。
    :param name: 模块名称，默认为""。

    :return: 一个Conv2D类

    .. note::
        ``padding='valid'`` 等同于无填充。

        ``padding='same'`` 对输入进行填充，使输出与输入具有相同的形状。

    Example::

        import numpy as np
        from pyvqnet.tensor import QTensor
        from pyvqnet.nn import Conv2D
        import pyvqnet
        b= 2
        ic =3
        oc = 2
        test_conv = Conv2D(ic,oc,(3,3),(2,2),"same")
        x0 = QTensor(np.arange(1,b*ic*5*5+1).reshape([b,ic,5,5]),requires_grad=True,dtype=pyvqnet.kfloat32)
        x = test_conv.forward(x0)
        print(x)

        # [
        # [[[-0.1256833, 23.8978596, 26.7449780],
        #  [-7.2959919, 33.4023743, 42.1283913],
        #  [-8.7684336, 25.2698975, 40.4024887]],
        # [[33.0653763, 40.3120155, 27.3781891],
        #  [39.2921371, 45.8685760, 38.1885109],
        #  [23.1873779, 12.0480318, 12.7278290]]],
        # [[[-0.9730744, 61.3967094, 79.0511856],
        #  [-29.3652401, 75.0349350, 112.7325439],
        #  [-26.4682808, 59.0924797, 104.2572098]],
        # [[66.8064194, 96.0953140, 72.9157486],
        #  [90.9154129, 110.7232437, 91.2616043],
        #  [56.8825951, 34.6904907, 30.1957760]]]
        # ]

二维转置卷积
=================================

.. py:class:: pyvqnet.nn.ConvT2D(input_channels,output_channels,kernel_size,stride=[1, 1],padding="valid",use_bias="True", kernel_initializer=None,bias_initializer=None, dilation_rate: int = 1, out_padding=(0,0), group: int = 1, dtype=None, name='')

    对输入应用二维转置卷积核。convT模块的输入形状为 (batch_size, input_channels, height, width)

    :param input_channels: `int` - 输入通道数
    :param output_channels: `int` - 卷积核数量
    :param kernel_size: `tuple|list` - 单个卷积核的大小。kernel shape = [input_channels,output_channels/group,kernel_size,kernel_size]
    :param stride: `tuple|list` - 步幅，默认为 (1, 1)|[1,1]
    :param padding: `str|tuple` - 填充选项，可以是字符串{'valid', 'same'}或表示两侧隐式填充量的整数元组。默认为"valid"。
    :param use_bias: `bool` - 是否使用偏置项。默认为使用
    :param kernel_initializer: `callable` - 默认为None
    :param bias_initializer: `callable` - 默认为None
    :param dilation_rate: `int` - 膨胀率，默认为1。
    :param out_padding: 添加到输出形状每个维度一侧的额外尺寸。默认为(0,0)
    :param group: `int` - 分组卷积的组数。默认为1。
    :param dtype: 参数的数据类型，默认为None，使用默认数据类型kfloat32，表示32位浮点数。
    :param name: 模块名称，默认为""。

    :return: 一个ConvT2D类

    .. note::
        ``padding='valid'`` 等同于无填充。

        ``padding='same'`` 对输入进行填充，使输出与输入具有相同的形状。


    Example::

        import numpy as np
        from pyvqnet.tensor import QTensor
        from pyvqnet.nn import ConvT2D
        import pyvqnet
        test_conv = ConvT2D(3, 2, (3, 3), (1, 1), "valid")
        x = QTensor(np.arange(1, 1 * 3 * 5 * 5+1).reshape([1, 3, 5, 5]), requires_grad=True,dtype=pyvqnet.kfloat32)
        y = test_conv.forward(x)
        print(y)

        # [
        # [[[-3.3675897, 4.8476148, 14.2448473, 14.8897810, 15.5347166, 20.0420666, 10.9831696],
        #  [-14.0110836, -3.2500827, 6.4022207, 6.5149083, 6.6275964, 23.7946320, 12.1828709],
        #  [-22.2661152, -3.5112300, 12.9493723, 13.5486069, 14.1478367, 39.6327629, 18.8349991],
        #  [-24.4063797, -3.0093837, 15.9455290, 16.5447617, 17.1439915, 44.7691879, 21.3293095],
        #  [-26.5466480, -2.5075383, 18.9416828, 19.5409145, 20.1401463, 49.9056053, 23.8236179],
        #  [-24.7624626, -13.7395811, -7.9510674, -7.9967723, -8.0424776, 19.2783546, 7.0562835],
        #  [-3.5170188, 10.2280807, 16.1939259, 16.6804695, 17.1670132, 21.2262039, 6.2889833]],
        # [[-2.0570512, -9.5056667, -25.0429192, -25.9464111, -26.8499031, -24.7305946, -16.9881954],
        #  [-0.7620960, -18.3383904, -49.8948288, -51.2528229, -52.6108208, -52.2179604, -34.3664169],
        #  [-11.7121849, -27.1864738, -62.2154846, -63.6433640, -65.0712280, -52.6787071, -38.4497032],
        #  [-13.3643141, -29.0211792, -69.3548126, -70.7826691, -72.2105408, -58.1659012, -43.7543182],
        #  [-15.0164423, -30.8558884, -76.4941254, -77.9219971, -79.3498535, -63.6530838, -49.0589256],
        #  [-11.6070204, -14.1940546, -35.5471687, -36.0715408, -36.5959129, -23.9147663, -22.8668022],
        #  [-14.4390459, -4.9011412, -6.4719801, -6.5418491, -6.6117167, 9.3329525, -1.7254852]]]
        # ]

一维平均池化
=================================

.. py:class:: pyvqnet.nn.AvgPool1D(kernel, stride, padding='valid', name='')

    此操作对由多个输入平面组成的一维输入信号应用一维平均池化。

    :param kernel: 平均池化窗口的大小
    :param strides: 下采样因子
    :param padding: "valid"、"same"之一或指定填充值的整数，默认为"valid"
    :param name: 输出层名称。

    :return: AvgPool1D层

    .. note::
        ``padding='valid'`` 等同于无填充。

        ``padding='same'`` 对输入进行填充，使输出与输入具有相同的形状。



    Example::

        import numpy as np
        from pyvqnet.tensor import QTensor
        from pyvqnet.nn import AvgPool1D
        test_mp = AvgPool1D([3],[2],"same")
        x= QTensor(np.array([0, 1, 0, 4, 5,
                                    2, 3, 2, 1, 3,
                                    4, 4, 0, 4, 3,
                                    2, 5, 2, 6, 4,
                                    1, 0, 0, 5, 7],dtype=float).reshape([1,5,5]),requires_grad=True)

        y= test_mp.forward(x)
        print(y)
        # [
        # [[0.3333333, 1.6666666, 3],
        #  [1.6666666, 2, 1.3333334],
        #  [2.6666667, 2.6666667, 2.3333333],
        #  [2.3333333, 4.3333335, 3.3333333],
        #  [0.3333333, 1.6666666, 4]]
        # ]

一维最大池化
=================================

.. py:class:: pyvqnet.nn.MaxPool1D(kernel, stride, padding='valid', dtype=None, name='')

    此操作对由多个输入平面组成的一维输入信号应用一维最大池化。

    :param kernel: 最大池化窗口的大小
    :param strides: 下采样因子
    :param padding: "valid"、"same"之一或指定填充值的整数，默认为"valid"
    :param name: 模块名称，默认为""。

    :return: MaxPool1D层

    .. note::

        ``padding='valid'`` 等同于无填充。

        ``padding='same'`` 对输入进行填充，使输出与输入具有相同的形状。


    Example::

        import numpy as np
        from pyvqnet.tensor import QTensor
        from pyvqnet.nn import MaxPool1D
        test_mp = MaxPool1D([3],[2],"same")
        x= QTensor(np.array([0, 1, 0, 4, 5,
                                    2, 3, 2, 1, 3,
                                    4, 4, 0, 4, 3,
                                    2, 5, 2, 6, 4,
                                    1, 0, 0, 5, 7],dtype=float).reshape([1,5,5]),requires_grad=True)

        y= test_mp.forward(x)
        print(y)
        #[[[1. 4. 5.]
        #   [3. 3. 3.]
        #   [4. 4. 4.]
        #   [5. 6. 6.]
        #   [1. 5. 7.]]]

二维平均池化
=================================

.. py:class:: pyvqnet.nn.AvgPool2D(kernel, stride, padding='valid', name='')

    此操作对输入特征应用二维平均池化。

    :param kernel: 平均池化窗口的大小
    :param strides: 下采样因子
    :param padding: "valid"、"same"之一或包含列和行填充值的整数元组，默认为"valid"
    :param name: 输出层名称
    :return: AvgPool2D层

    .. note::
        ``padding='valid'`` 等同于无填充。

        ``padding='same'`` 对输入进行填充，使输出与输入具有相同的形状。


    Example::

        import numpy as np
        from pyvqnet.tensor import QTensor
        from pyvqnet.nn import AvgPool2D
        test_mp = AvgPool2D([2,2],[2,2],"valid")
        x= QTensor(np.array([0, 1, 0, 4, 5,
                                    2, 3, 2, 1, 3,
                                    4, 4, 0, 4, 3,
                                    2, 5, 2, 6, 4,
                                    1, 0, 0, 5, 7],dtype=float).reshape([1,1,5,5]),requires_grad=True)

        y= test_mp.forward(x)
        print(y)
        #[[[[1.5  1.75]
        #    [3.75 3.  ]]]]

二维最大池化
=================================

.. py:class:: pyvqnet.nn.MaxPool2D(kernel, stride, padding='valid', name='')

    此操作对输入特征应用二维最大池化。

    :param kernel: 最大池化窗口的大小
    :param strides: 下采样因子
    :param padding: "valid"、"same"之一或包含列和行填充值的整数元组，默认为"valid"
    :param name: 输出层名称
    :return: MaxPool2D层

    .. note::
        ``padding='valid'`` 等同于无填充。

        ``padding='same'`` 对输入进行填充，使输出与输入具有相同的形状。


    Example::

        import numpy as np
        from pyvqnet.tensor import QTensor
        from pyvqnet.nn import MaxPool2D
        test_mp = MaxPool2D([2,2],[2,2],"valid")
        x= QTensor(np.array([0, 1, 0, 4, 5,
                                    2, 3, 2, 1, 3,
                                    4, 4, 0, 4, 3,
                                    2, 5, 2, 6, 4,
                                    1, 0, 0, 5, 7],dtype=float).reshape([1,1,5,5]),requires_grad=True)

        y= test_mp.forward(x)
        print(y)
        # [[[[3. 4.]
        #    [5. 6.]]]]

嵌入层
=================================

.. py:class:: pyvqnet.nn.embedding.Embedding(num_embeddings, embedding_dim, weight_initializer=<function xavier_normal>,dtype=None, name: str = '')

    此模块通常用于存储词嵌入并通过索引检索它们。
    模块的输入是索引列表，输出是对应的词嵌入。
    word embeddings.

    :param num_embeddings: `int` - 嵌入字典的大小。
    :param embedding_dim: `int` - 每个嵌入向量的维度。
    :param weight_initializer: `callable` - 默认为normal。
    :param dtype: 参数的数据类型，默认为None，使用默认数据类型kfloat32，表示32位浮点数。
    :param name: 输出层名称。

    :return: 一个Embedding类

    Example::

        import numpy as np
        from pyvqnet.tensor import QTensor
        from pyvqnet.nn.embedding import Embedding
        import pyvqnet
        vlayer = Embedding(30,3)
        x = QTensor(np.arange(1,25).reshape([2,3,2,2]),dtype= pyvqnet.kint64)
        y = vlayer(x)
        print(y)

        # [
        # [[[[-0.3168081, 0.0329394, -0.2934906],
        #  [0.1057295, -0.2844988, -0.1687456]],
        # [[-0.2382513, -0.3642318, -0.2257225],
        #  [0.1563180, 0.1567665, 0.3038477]]],
        # [[[-0.4131152, -0.0564500, -0.2804018],
        #  [-0.2955172, -0.0009581, -0.1641144]],
        # [[0.0692555, 0.1094901, 0.4099118],
        #  [0.4348361, 0.0304361, -0.0061203]]],
        # [[[-0.3310401, -0.1836129, 0.1098949],
        #  [-0.1840732, 0.0332474, -0.0261806]],
        # [[-0.1489778, 0.2519453, 0.3299376],
        #  [-0.1942692, -0.1540277, -0.2335350]]]],
        # [[[[-0.2620637, -0.3181309, -0.1857461],
        #  [-0.0878164, -0.4180320, -0.1831555]],
        # [[-0.0738970, -0.1888980, -0.3034399],
        #  [0.1955448, -0.0409723, 0.3023460]]],
        # [[[0.2430045, 0.0880465, 0.4309453],
        #  [-0.1796514, -0.1432367, -0.1253638]],
        # [[-0.5266719, 0.2386262, -0.0329155],
        #  [0.1033449, -0.3442690, -0.0471130]]],
        # [[[-0.5336705, -0.1939755, -0.3000667],
        #  [0.0059001, 0.5567381, 0.1926173]],
        # [[-0.2385869, -0.3910453, 0.2521235],
        #  [-0.0246447, -0.0241158, -0.1402829]]]]
        # ]


二维批量归一化
=================================

.. py:class:: pyvqnet.nn.BatchNorm2d(channel_num:int, momentum:float=0.1, epsilon:float = 1e-5, affine= True, beta_initializer=zeros, gamma_initializer=ones, dtype=None, name="")

    对四维输入 (B,C,H,W) 应用批量归一化，如论文
    `Batch Normalization: Accelerating Deep Network Training by Reducing
    Internal Covariate Shift <https://arxiv.org/abs/1502.03167>`__ 所述。

    .. math::

        y = \frac{x - \mathrm{E}[x]}{\sqrt{\mathrm{Var}[x] + \epsilon}} * \gamma + \beta

    其中 :math:`\gamma` 和 :math:`\beta` 是可学习参数。默认情况下，在训练期间，此层会保持对计算得到的均值和方差的运行估计，并在评估期间用于归一化。
    运行估计的动量默认为0.1。

    :param channel_num: `int` - 输入特征通道数。
    :param momentum: `float` - 计算指数加权平均时的动量，默认为0.1。
    :param epsilon: `float` - 数值稳定性常数，默认为1e-5。
    :param affine: 一个布尔值，当设置为 ``True`` 时，此模块具有可学习的逐通道仿射参数，初始化为1（权重）和0（偏置）。默认为 ``True``。
    :param beta_initializer: `callable` - 默认为zeros。
    :param gamma_initializer: `callable` - 默认为ones。
    :param dtype: 参数的数据类型，默认为None，使用默认数据类型kfloat32，表示32位浮点数。
    :param name: 输出层名称
    :return: 一个BatchNorm2d类

    Example::

        import numpy as np
        from pyvqnet.tensor import QTensor
        from pyvqnet.nn import BatchNorm2d
        import pyvqnet
        b = 2
        ic = 2
        test_conv = BatchNorm2d(ic)

        x = QTensor(np.arange(1, 17).reshape([b, ic, 4, 1]),
                    requires_grad=True,
                    dtype=pyvqnet.kfloat32)
        y = test_conv.forward(x)
        print(y)

        # [
        # [[[-1.3242440],
        #  [-1.0834724],
        #  [-0.8427007],
        #  [-0.6019291]],
        # [[-1.3242440],
        #  [-1.0834724],
        #  [-0.8427007],
        #  [-0.6019291]]],
        # [[[0.6019291],
        #  [0.8427007],
        #  [1.0834724],
        #  [1.3242440]],
        # [[0.6019291],
        #  [0.8427007],
        #  [1.0834724],
        #  [1.3242440]]]
        # ]


一维批量归一化
=================================

.. py:class:: pyvqnet.nn.BatchNorm1d(channel_num:int, momentum:float=0.1, epsilon:float = 1e-5, affine = True, beta_initializer=zeros, gamma_initializer=ones, dtype=None, name="")

    对二维输入 (B,C) 应用批量归一化，如论文
    `Batch Normalization: Accelerating Deep Network Training by Reducing
    Internal Covariate Shift <https://arxiv.org/abs/1502.03167>`__ 所述。

    .. math::

        y = \frac{x - \mathrm{E}[x]}{\sqrt{\mathrm{Var}[x] + \epsilon}} * \gamma + \beta

    其中 :math:`\gamma` 和 :math:`\beta` 是可学习参数。默认情况下，在训练期间，此层会保持对计算得到的均值和方差的运行估计，并在评估期间用于归一化。
    运行估计的动量默认为0.1。


    :param channel_num: `int` - 输入特征通道数。
    :param momentum: `float` - 计算指数加权平均时的动量，默认为0.1
    :param epsilon: `float` - 数值稳定性常数，默认为1e-5。
    :param affine: 一个布尔值，当设置为 ``True`` 时，此模块具有可学习的逐通道仿射参数，初始化为1（权重）和0（偏置）。默认为 ``True``。
    :param beta_initializer: `callable` - 默认为zeros。
    :param gamma_initializer: `callable` - 默认为ones。
    :param dtype: 参数的数据类型，默认为None，使用默认数据类型kfloat32，表示32位浮点数。
    :param name: 输出层名称
    :return: 一个BatchNorm1d类

    Example::

        import numpy as np
        from pyvqnet.tensor import QTensor
        from pyvqnet.nn import BatchNorm1d
        import pyvqnet
        test_conv = BatchNorm1d(4)

        x = QTensor(np.arange(1, 17).reshape([4, 4]),
                    requires_grad=True,
                    dtype=pyvqnet.kfloat32)
        y = test_conv.forward(x)
        print(y)


        # [
        # [-1.3416405, -1.3416405, -1.3416405, -1.3416405],
        # [-0.4472135, -0.4472135, -0.4472135, -0.4472135],
        # [0.4472135, 0.4472135, 0.4472135, 0.4472135],
        # [1.3416405, 1.3416405, 1.3416405, 1.3416405]
        # ]



N维层归一化
=================================

.. py:class:: pyvqnet.nn.layer_norm.LayerNormNd(normalized_shape: list, epsilon: float = 1e-5, affine = True, dtype=None,name="")

    对任意输入的最后几个维度执行层归一化。具体方法如论文所述：
    `Layer Normalization <https://arxiv.org/abs/1607.06450>`__。

    .. math::
        y = \frac{x - \mathrm{E}[x]}{ \sqrt{\mathrm{Var}[x] + \epsilon}} * \gamma + \beta

    对于形如 (B,C,H,W,D) 的输入，``norm_shape`` 可以是 [C,H,W,D]、[H,W,D]、[W,D] 或 [D] 。

    :param norm_shape: `float` - 标准化的形状。
    :param epsilon: `float` - 数值稳定性常数，默认为1e-5。
    :param affine: 一个布尔值，当设置为 ``True`` 时，此模块具有可学习的逐通道仿射参数，初始化为1（权重）和0（偏置）。默认为 ``True``。
    :param dtype: 参数的数据类型，默认为None，使用默认数据类型kfloat32，表示32位浮点数。
    :param name: 输出层名称。

    :return: 一个LayerNormNd类。

    Example::

        import numpy as np
        from pyvqnet.tensor import QTensor
        from pyvqnet import kfloat32
        from pyvqnet.nn.layer_norm import LayerNormNd
        ic = 4
        test_conv = LayerNormNd([2,2])
        x = QTensor(np.arange(1,17).reshape([2,2,2,2]),requires_grad=True,dtype=kfloat32)
        y = test_conv.forward(x)
        print(y)
        # [
        # [[[-1.3416355, -0.4472118],
        #  [0.4472118, 1.3416355]],
        # [[-1.3416355, -0.4472118],
        #  [0.4472118, 1.3416355]]],
        # [[[-1.3416355, -0.4472118],
        #  [0.4472118, 1.3416355]],
        # [[-1.3416355, -0.4472118],
        #  [0.4472118, 1.3416355]]]
        # ]


二维层归一化
=================================

.. py:class:: pyvqnet.nn.layer_norm.LayerNorm2d(norm_size:int, epsilon:float = 1e-5, affine= True, dtype=None, name="")

    对四维输入的小批量应用层归一化，如论文所述：
    `Layer Normalization <https://arxiv.org/abs/1607.06450>`__

    .. math::
        y = \frac{x - \mathrm{E}[x]}{ \sqrt{\mathrm{Var}[x] + \epsilon}} * \gamma + \beta

    均值和标准差是在最后 `D` 维的大小上计算的。

    对于形如 (B,C,H,W) 的输入，``norm_size`` 应等于 C * H * W。

    :param norm_size: `float` - 归一化大小，等于 C * H * W
    :param epsilon: `float` - 数值稳定性常数，默认为1e-5
    :param affine: 一个布尔值，当设置为 ``True`` 时，此模块具有可学习的逐通道仿射参数，初始化为1（权重）和0（偏置）。默认为 ``True``。
    :param name: 输出层名称
    :param dtype: 参数的数据类型，默认为None，使用默认数据类型kfloat32，表示32位浮点数。
    
    :return: 一个LayerNorm2d类

    Example::

        import numpy as np
        import pyvqnet
        from pyvqnet.tensor import QTensor
        from pyvqnet.nn.layer_norm import LayerNorm2d
        ic = 4
        test_conv = LayerNorm2d(8)
        x = QTensor(np.arange(1,17).reshape([2,2,4,1]),requires_grad=True,dtype=pyvqnet.kfloat32)
        y = test_conv.forward(x)
        print(y)

        # [
        # [[[-1.5275238],
        #  [-1.0910884],
        #  [-0.6546531],
        #  [-0.2182177]],
        # [[0.2182177],
        #  [0.6546531],
        #  [1.0910884],
        #  [1.5275238]]],
        # [[[-1.5275238],
        #  [-1.0910884],
        #  [-0.6546531],
        #  [-0.2182177]],
        # [[0.2182177],
        #  [0.6546531],
        #  [1.0910884],
        #  [1.5275238]]]
        # ]

一维层归一化
=================================

.. py:class:: pyvqnet.nn.layer_norm.LayerNorm1d(norm_size:int, epsilon:float = 1e-5, affine= True, dtype=None,name="")

    对二维输入的小批量应用层归一化，如论文所述：
    `Layer Normalization <https://arxiv.org/abs/1607.06450>`__

    .. math::
        y = \frac{x - \mathrm{E}[x]}{ \sqrt{\mathrm{Var}[x] + \epsilon}} * \gamma + \beta

    均值和标准差是在最后一个维度的大小上计算的，其中 ``norm_size`` 是最后一个维度的大小值。
    是最后一个维度的大小值。

    :param norm_size: `float` - 归一化大小，等于最后一个维度
    :param epsilon: `float` - 数值稳定性常数，默认为1e-5
    :param affine: 一个布尔值，当设置为 ``True`` 时，此模块具有可学习的逐通道仿射参数，初始化为1（权重）和0（偏置）。默认为 ``True``。
    :param dtype: 参数的数据类型，默认为None，使用默认数据类型kfloat32，表示32位浮点数。
    :param name: 输出层名称

    :return: 一个LayerNorm1d类

    Example::

        import numpy as np
        import pyvqnet
        from pyvqnet.tensor import QTensor
        from pyvqnet.nn.layer_norm import LayerNorm1d
        test_conv = LayerNorm1d(4)
        x = QTensor(np.arange(1,17).reshape([4,4]),requires_grad=True,dtype=pyvqnet.kfloat32)
        y = test_conv.forward(x)
        print(y)

        # [
        # [-1.3416355, -0.4472118, 0.4472118, 1.3416355],
        # [-1.3416355, -0.4472118, 0.4472118, 1.3416355],
        # [-1.3416355, -0.4472118, 0.4472118, 1.3416355],
        # [-1.3416355, -0.4472118, 0.4472118, 1.3416355]
        # ]


组归一化
=============================================================

.. py:class:: pyvqnet.nn.group_norm.GroupNorm(num_groups: int, num_channels: int, epsilon = 1e-5, affine = True, dtype = None, name = "")

    对输入的小批量应用组归一化。输入：:math:`(N, C, *)` 其中 :math:`C=\mathrm{num\_channels}` ，输出：:math:`(N, C, *)` 。

    此层实现了论文 `Group Normalization <https://arxiv.org/abs/1803.08494>`__ 中描述的操作。

    .. math::

        y = \frac{x - \mathrm{E}[x]}{ \sqrt{\mathrm{Var}[x] + \epsilon}} * \gamma + \beta

    输入通道被分为 :attr:`num_groups` 组，每组包含 ``num_channels / num_groups`` 个通道。 :attr:`num_channels` 必须能被 :attr:`num_groups` 整除。每组分别计算均值和标准差。如果 :attr:`affine` 为 ``True``，则 :math:`\gamma` 和 :math:`\beta` 是可学习的。大小为 :attr:`num_channels` 的逐通道仿射变换参数向量。

    :param num_groups (int): 将通道分成的组数
    :param num_channels (int): 输入中预期的通道数
    :param eps: 添加到分母的数值稳定性常数。默认为1e-5
    :param affine: 一个布尔值，当设置为 ``True`` 时，此模块具有可学习的逐通道仿射参数，初始化为1（权重）和0（偏置）。默认为 ``True``。
    :param dtype: 参数的数据类型，默认为None，使用默认数据类型kfloat32，表示32位浮点数。
    :param name: 输出层名称

    :return: GroupNorm类

    Example::

        import numpy as np
        from pyvqnet.tensor import QTensor
        from pyvqnet import kfloat32
        from pyvqnet.nn import GroupNorm
        test_conv = GroupNorm(2,10)
        x = QTensor(np.arange(0,60*2*5).reshape([2,10,3,2,5]),requires_grad=True,dtype=kfloat32)
        y = test_conv.forward(x)
        print(y)


线性层
=================================

.. py:class:: pyvqnet.nn.Linear(input_channels, output_channels, weight_initializer=None, bias_initializer=None,use_bias=True, dtype=None, name: str = "")

    线性模块（全连接层）。
    :math:`y = x@A.T + b`

    :param input_channels: `int` - 输入特征数
    :param output_channels: `int` - 输出特征数
    :param weight_initializer: `callable` - 默认为normal
    :param bias_initializer: `callable` - 默认为zeros
    :param use_bias: `bool` - 默认为True
    :param dtype: 参数的数据类型，默认为None，使用默认数据类型kfloat32，表示32位浮点数。
    :param name: 输出层名称

    :return: 一个Linear类

    Example::

        import numpy as np
        import pyvqnet
        from pyvqnet.tensor import QTensor
        from pyvqnet.nn import Linear
        c1 =2
        c2 = 3
        cin = 7
        cout = 5
        n = Linear(cin,cout)
        input = QTensor(np.arange(1,c1*c2*cin+1).reshape((c1,c2,cin)),requires_grad=True,dtype=pyvqnet.kfloat32)
        y = n.forward(input)
        print(y)

        # [
        # [[4.3084583, -1.9228780, -0.3428757, 1.2840536, -0.5865945],
        #  [9.8339605, -5.5135884, -3.1228657, 4.3025794, -4.1492314],
        #  [15.3594627, -9.1042995, -5.9028554, 7.3211040, -7.7118683]],
        # [[20.8849659, -12.6950111, -8.6828451, 10.3396301, -11.2745066],
        #  [26.4104652, -16.2857227, -11.4628344, 13.3581581, -14.8371439],
        #  [31.9359703, -19.8764324, -14.2428246, 16.3766804, -18.3997803]]
        # ]


随机丢弃层
=================================

.. py:class:: pyvqnet.nn.dropout.Dropout(dropout_rate = 0.5)

    Dropout模块。Dropout模块随机将某些单元的输出设置为零，同时根据给定的丢弃概率放大其他单元。

    :param dropout_rate: `float` - 神经元被置零的概率
    :return: 一个Dropout类

    Example::

        from pyvqnet.nn.dropout import Dropout
        import numpy as np
        from pyvqnet.tensor import QTensor
        b = 2
        ic = 2
        x = QTensor(np.arange(-1*ic*2*2,(b-1)*ic*2*2.0).reshape([b,ic,2,2]),requires_grad=True)
        droplayer = Dropout(0.5)
        droplayer.train()
        y = droplayer(x)
        print(y)
        # [[[[-16. -14.]
        #    [-12.   0.]]

        #   [[ -8.  -6.]
        #    [ -4.  -2.]]]


        #  [[[  0.   2.]
        #    [  0.   6.]]

        #   [[  0.   0.]
        #    [  0.  14.]]]]

随机深度丢弃层
=================================

.. py:class:: pyvqnet.nn.dropout.DropPath(dropout_rate = 0.5,name="")

    DropPath模块会逐样本随机丢弃路径（随机深度）。

    :param dropout_rate: `float` - 神经元被置零的概率。
    :param name: 此模块的名称，默认为""。

    :return: DropPath实例。

    Example::

        import pyvqnet.nn as nn
        import pyvqnet.tensor as tensor

        x = tensor.randu([4])
        y = nn.DropPath()(x)
        print(y)
        #[0.9074978,0.9350062,0.6896403,0.3541051]


像素混洗层
=================================

.. py:class:: pyvqnet.nn.pixel_shuffle.Pixel_Shuffle(upscale_factors)

    将形状为 (*, C * r^2, H, W) 的张量重排为形状为 (*, C, H * r, W * r) 的张量，其中 r 是缩放因子。

    :param upscale_factors: 放大变换的因子

    :return:
            Pixel_Shuffle模块

    Example::

        from pyvqnet.nn import Pixel_Shuffle
        from pyvqnet.tensor import tensor
        ps = Pixel_Shuffle(3)
        inx = tensor.ones([5,2,3,18,4,4])
        inx.requires_grad=  True
        y = ps(inx)
        print(y.shape)
        #[5, 2, 3, 2, 12, 12]

像素逆混洗层
=================================

.. py:class:: pyvqnet.nn.pixel_shuffle.Pixel_Unshuffle(downscale_factors)

    通过重新排列元素来反转Pixel_Shuffle操作。将形状为 (*, C, H * r, W * r) 的张量重排为 (*, C * r^2, H, W) ，其中 r 是收缩因子。
    
    :param downscale_factors: 缩小变换的因子

    :return:
            Pixel_Unshuffle模块

    Example::

        from pyvqnet.nn import Pixel_Unshuffle
        from pyvqnet.tensor import tensor
        ps = Pixel_Unshuffle(3)
        inx = tensor.ones([5, 2, 3, 2, 12, 12])
        inx.requires_grad = True
        y = ps(inx)
        print(y.shape)
        #[5, 2, 3, 18, 4, 4]


门控循环单元层
=================================

.. py:class:: pyvqnet.nn.gru.GRU(input_size, hidden_size, num_layers=1, nonlinearity='tanh', batch_first=True, use_bias=True, bidirectional=False, dtype=None, name: str = '')


    门控循环单元（GRU）模块。支持多层堆叠、双向配置。
    单层单向GRU的计算公式如下：

    .. math::
        \begin{array}{ll}
            r_t = \sigma(W_{ir} x_t + b_{ir} + W_{hr} h_{(t-1)} + b_{hr}) \\
            z_t = \sigma(W_{iz} x_t + b_{iz} + W_{hz} h_{(t-1)} + b_{hz}) \\
            n_t = \tanh(W_{in} x_t + b_{in} + r_t * (W_{hn} h_{(t-1)}+ b_{hn})) \\
            h_t = (1 - z_t) * n_t + z_t * h_{(t-1)}
        \end{array}

    :param input_size: 输入特征维度。
    :param hidden_size: 隐藏特征维度。
    :param num_layers: 堆叠层数。默认为1。
    :param batch_first: 如果batch_first为True，输入形状应为 [batch_size,seq_len,feature_dim]；
     如果batch_first为False，输入形状应为 [seq_len,batch_size,feature_dim]。默认为True。
    :param use_bias: 如果use_bias为False，此模块将不包含偏置。默认为True。
    :param bidirectional: 如果bidirectional为True，模块将为双向GRU。默认为False。
    :param dtype: 参数的数据类型，默认为None，使用默认数据类型kfloat32，表示32位浮点数。
    :param name: 输出层名称

    :return: 一个GRU模块实例。

    Example::

        from pyvqnet.nn import GRU
        from pyvqnet.tensor import tensor

        rnn2 = GRU(4, 6, 2, batch_first=False, bidirectional=True)

        input = tensor.ones([5, 3, 4])
        h0 = tensor.ones([4, 3, 6])

        output, hn = rnn2(input, h0)
        print(output)
        print(hn)
        # [
        # [[0.2815045, 0.2056844, 0.0750246, 0.5802019, 0.3536537, 0.8136684, -0.0034523, 0.1634004, 0.6099871, 0.8451654, -0.2833570, 0.7294812],
        #  [0.2815045, 0.2056844, 0.0750246, 0.5802019, 0.3536537, 0.8136684, -0.0034523, 0.1634004, 0.6099871, 0.8451654, -0.2833570, 0.7294812],
        #  [0.2815045, 0.2056844, 0.0750246, 0.5802019, 0.3536537, 0.8136684, -0.0034523, 0.1634004, 0.6099871, 0.8451654, -0.2833570, 0.7294812]],
        # [[0.0490867, 0.0115325, -0.2797680, 0.4711050, -0.0687061, 0.7216146, 0.0258964, 0.0619203, 0.6341010, 0.8445141, -0.4164453, 0.7409840],
        #  [0.0490867, 0.0115325, -0.2797680, 0.4711050, -0.0687061, 0.7216146, 0.0258964, 0.0619203, 0.6341010, 0.8445141, -0.4164453, 0.7409840],
        #  [0.0490867, 0.0115325, -0.2797680, 0.4711050, -0.0687061, 0.7216146, 0.0258964, 0.0619203, 0.6341010, 0.8445141, -0.4164453, 0.7409840]],
        # [[0.0182974, -0.0536071, -0.4478674, 0.4315647, -0.2191887, 0.6492687, 0.1572548, 0.0839213, 0.6707115, 0.8444533, -0.3811499, 0.7448123],
        #  [0.0182974, -0.0536071, -0.4478674, 0.4315647, -0.2191887, 0.6492687, 0.1572548, 0.0839213, 0.6707115, 0.8444533, -0.3811499, 0.7448123],
        #  [0.0182974, -0.0536071, -0.4478674, 0.4315647, -0.2191887, 0.6492687, 0.1572548, 0.0839213, 0.6707115, 0.8444533, -0.3811499, 0.7448123]],
        # [[0.0722285, -0.0636698, -0.5457084, 0.3817562, -0.1890205, 0.5696942, 0.3855782, 0.2057217, 0.7370453, 0.8646453, -0.1967214, 0.7630759],
        #  [0.0722285, -0.0636698, -0.5457084, 0.3817562, -0.1890205, 0.5696942, 0.3855782, 0.2057217, 0.7370453, 0.8646453, -0.1967214, 0.7630759],
        #  [0.0722285, -0.0636698, -0.5457084, 0.3817562, -0.1890205, 0.5696942, 0.3855782, 0.2057217, 0.7370453, 0.8646453, -0.1967214, 0.7630759]],
        # [[0.1834545, -0.0489200, -0.6343678, 0.3061281, -0.0449328, 0.4901535, 0.6941375, 0.4570828, 0.8433002, 0.9152645, 0.2342478, 0.8299093],
        #  [0.1834545, -0.0489200, -0.6343678, 0.3061281, -0.0449328, 0.4901535, 0.6941375, 0.4570828, 0.8433002, 0.9152645, 0.2342478, 0.8299093],
        #  [0.1834545, -0.0489200, -0.6343678, 0.3061281, -0.0449328, 0.4901535, 0.6941375, 0.4570828, 0.8433002, 0.9152645, 0.2342478, 0.8299093]]
        # ]
        # [
        # [[-0.8070476, -0.5560303, 0.7575479, -0.2368367, 0.4228620, -0.2573725],
        #  [-0.8070476, -0.5560303, 0.7575479, -0.2368367, 0.4228620, -0.2573725],
        #  [-0.8070476, -0.5560303, 0.7575479, -0.2368367, 0.4228620, -0.2573725]],
        # [[-0.3857390, -0.3195596, 0.0281313, 0.8734715, -0.4499536, 0.2270730],
        #  [-0.3857390, -0.3195596, 0.0281313, 0.8734715, -0.4499536, 0.2270730],
        #  [-0.3857390, -0.3195596, 0.0281313, 0.8734715, -0.4499536, 0.2270730]],
        # [[0.1834545, -0.0489200, -0.6343678, 0.3061281, -0.0449328, 0.4901535],
        #  [0.1834545, -0.0489200, -0.6343678, 0.3061281, -0.0449328, 0.4901535],
        #  [0.1834545, -0.0489200, -0.6343678, 0.3061281, -0.0449328, 0.4901535]],
        # [[-0.0034523, 0.1634004, 0.6099871, 0.8451654, -0.2833570, 0.7294812],
        #  [-0.0034523, 0.1634004, 0.6099871, 0.8451654, -0.2833570, 0.7294812],
        #  [-0.0034523, 0.1634004, 0.6099871, 0.8451654, -0.2833570, 0.7294812]]
        # ]

循环神经网络层
=================================

.. py:class:: pyvqnet.nn.rnn.RNN(input_size, hidden_size, num_layers=1, nonlinearity='tanh', batch_first=True, use_bias=True, bidirectional=False, dtype=None, name: str = '')


    循环神经网络（RNN）模块，使用 :math:`\tanh` 或 :math:`\text{ReLU}` 作为激活函数。
    支持双向RNN和多层RNN。
    单层单向RNN的计算公式如下：

    .. math::
        h_t = \tanh(W_{ih} x_t + b_{ih} + W_{hh} h_{(t-1)} + b_{hh})

    如果 :attr:`nonlinearity` 为 ``'relu'``，则 :math:`\text{ReLU}` 将替换 :math:`\tanh`。

    :param input_size: 输入特征维度。
    :param hidden_size: 隐藏特征维度。
    :param num_layers: 堆叠层数。默认为1。
    :param nonlinearity: 非线性激活函数，默认为 ``'tanh'`` 。
    :param batch_first: 如果batch_first为True，输入形状应为 [batch_size,seq_len,feature_dim]；
     如果batch_first为False，输入形状应为 [seq_len,batch_size,feature_dim]。默认为True。
    :param use_bias: 如果use_bias为False，此模块将不包含偏置。默认为True。
    :param bidirectional: 如果bidirectional为True，模块将为双向RNN。默认为False。
    :param dtype: 参数的数据类型，默认为None，使用默认数据类型kfloat32，表示32位浮点数。
    :param name: 输出层名称

    :return: 一个RNN模块实例。

    Example::

        from pyvqnet.nn import RNN
        from pyvqnet.tensor import tensor

        rnn2 = RNN(4, 6, 2, batch_first=False, bidirectional = True)

        input = tensor.ones([5, 3, 4])
        h0 = tensor.ones([4, 3, 6])
        output, hn = rnn2(input, h0)
        print(output)
        print(hn)
        # [
        # [[-0.4481719, 0.4345263, 0.0284741, 0.6886298, 0.8672314, -0.3574123, 0.8238092, -0.2751125, -0.4704098, 0.7624499, -0.4156595, -0.1646518],
        #  [-0.4481719, 0.4345263, 0.0284741, 0.6886298, 0.8672314, -0.3574123, 0.8238092, -0.2751125, -0.4704098, 0.7624499, -0.4156595, -0.1646518],
        #  [-0.4481719, 0.4345263, 0.0284741, 0.6886298, 0.8672314, -0.3574123, 0.8238092, -0.2751125, -0.4704098, 0.7624499, -0.4156595, -0.1646518]],
        # [[-0.5737326, 0.1401956, -0.6656274, 0.3557707, 0.4083472, 0.3605195, 0.6767184, -0.2054843, -0.2875977, 0.6573941, -0.3289444, -0.1988498],
        #  [-0.5737326, 0.1401956, -0.6656274, 0.3557707, 0.4083472, 0.3605195, 0.6767184, -0.2054843, -0.2875977, 0.6573941, -0.3289444, -0.1988498],
        #  [-0.5737326, 0.1401956, -0.6656274, 0.3557707, 0.4083472, 0.3605195, 0.6767184, -0.2054843, -0.2875977, 0.6573941, -0.3289444, -0.1988498]],
        # [[-0.4233001, 0.1252111, -0.7437832, 0.2092323, 0.5826398, 0.5207447, 0.7403980, -0.0006015, -0.4055642, 0.6553873, -0.0861093, -0.2096289],
        #  [-0.4233001, 0.1252111, -0.7437832, 0.2092323, 0.5826398, 0.5207447, 0.7403980, -0.0006015, -0.4055642, 0.6553873, -0.0861093, -0.2096289],
        #  [-0.4233001, 0.1252111, -0.7437832, 0.2092323, 0.5826398, 0.5207447, 0.7403980, -0.0006015, -0.4055642, 0.6553873, -0.0861093, -0.2096289]],
        # [[-0.3636788, 0.3627384, -0.6542842, 0.0563165, 0.5711210, 0.5174620, 0.4968840, -0.3591014, -0.5738643, 0.7505787, -0.1767489, 0.2954176], [-0.3636788, 0.3627384, -0.6542842, 0.0563165, 0.5711210, 0.5174620, 0.4968840, -0.3591014, -0.5738643, 0.7505787, -0.1767489, 0.2954176], [-0.3636788, 0.3627384, -0.6542842, 0.0563165, 0.5711210, 0.5174620, 0.4968840, -0.3591014, -0.5738643, 0.7505787, -0.1767489, 0.2954176]],
        # [[-0.1619987, 0.3079547, -0.5022690, -0.2989357, 0.2861646, 0.4965633, 0.4618312, -0.4173903, 0.1423969, -0.2332578, -0.4014739, 0.0601179],
        #  [-0.1619987, 0.3079547, -0.5022690, -0.2989357, 0.2861646, 0.4965633, 0.4618312, -0.4173903, 0.1423969, -0.2332578, -0.4014739, 0.0601179],
        #  [-0.1619987, 0.3079547, -0.5022690, -0.2989357, 0.2861646, 0.4965633, 0.4618312, -0.4173903, 0.1423969, -0.2332578, -0.4014739, 0.0601179]]
        # ]
        # [
        # [[-0.1878589, -0.5177042, -0.3672480, 0.1613673, 0.4321197, 0.6168041],
        #  [-0.1878589, -0.5177042, -0.3672480, 0.1613673, 0.4321197, 0.6168041],
        #  [-0.1878589, -0.5177042, -0.3672480, 0.1613673, 0.4321197, 0.6168041]],
        # [[-0.7923757, 0.0184400, -0.2851982, -0.6367047, 0.5933805, -0.6244841],
        #  [-0.7923757, 0.0184400, -0.2851982, -0.6367047, 0.5933805, -0.6244841],
        #  [-0.7923757, 0.0184400, -0.2851982, -0.6367047, 0.5933805, -0.6244841]],
        # [[-0.1619987, 0.3079547, -0.5022690, -0.2989357, 0.2861646, 0.4965633],
        #  [-0.1619987, 0.3079547, -0.5022690, -0.2989357, 0.2861646, 0.4965633],
        #  [-0.1619987, 0.3079547, -0.5022690, -0.2989357, 0.2861646, 0.4965633]],
        # [[0.8238092, -0.2751125, -0.4704098, 0.7624499, -0.4156595, -0.1646518],
        #  [0.8238092, -0.2751125, -0.4704098, 0.7624499, -0.4156595, -0.1646518],
        #  [0.8238092, -0.2751125, -0.4704098, 0.7624499, -0.4156595, -0.1646518]]
        # ]



长短期记忆网络层
=================================

.. py:class:: pyvqnet.nn.lstm.LSTM(input_size, hidden_size, num_layers=1, batch_first=True, use_bias=True, bidirectional=False, dtype=None, name: str = '')

    长短期记忆网络（LSTM）模块。支持双向LSTM、堆叠多层LSTM等配置。
    单层单向LSTM的计算公式如下：

    .. math::
        \begin{array}{ll} \\
            i_t = \sigma(W_{ii} x_t + b_{ii} + W_{hi} h_{t-1} + b_{hi}) \\
            f_t = \sigma(W_{if} x_t + b_{if} + W_{hf} h_{t-1} + b_{hf}) \\
            g_t = \tanh(W_{ig} x_t + b_{ig} + W_{hg} h_{t-1} + b_{hg}) \\
            o_t = \sigma(W_{io} x_t + b_{io} + W_{ho} h_{t-1} + b_{ho}) \\
            c_t = f_t \odot c_{t-1} + i_t \odot g_t \\
            h_t = o_t \odot \tanh(c_t) \\
        \end{array}

    :param input_size: 输入特征维度。
    :param hidden_size: 隐藏特征维度。
    :param num_layers: 堆叠层数。默认为1。
    :param batch_first: 如果batch_first为True，输入形状应为 [batch_size,seq_len,feature_dim]；
     如果batch_first为False，输入形状应为 [seq_len,batch_size,feature_dim]。默认为True。
    :param use_bias: 如果use_bias为False，此模块将不包含偏置。默认为True。
    :param bidirectional: 如果bidirectional为True，模块将为双向LSTM。默认为False。
    :param dtype: 参数的数据类型，默认为None，使用默认数据类型kfloat32，表示32位浮点数。
    :param name: 输出层名称

    :return: 一个LSTM模块实例。

    Example::

        from pyvqnet.nn import LSTM
        from pyvqnet.tensor import tensor

        rnn2 = LSTM(4, 6, 2, batch_first=False, bidirectional = True)

        input = tensor.ones([5, 3, 4])
        h0 = tensor.ones([4, 3, 6])
        c0 = tensor.ones([4, 3, 6])
        output, (hn, cn) = rnn2(input, (h0, c0))

        print(output)
        print(hn)
        print(cn)

        # [
        # [[0.1585344, 0.1758823, 0.4273642, 0.1640685, 0.1030634, 0.1657819, -0.0197110, 0.2073366, 0.0050953, -0.1467141, -0.1413236, -0.1404487], 
        #  [0.1585344, 0.1758823, 0.4273642, 0.1640685, 0.1030634, 0.1657819, -0.0197110, 0.2073366, 0.0050953, -0.1467141, -0.1413236, -0.1404487], 
        #  [0.1585344, 0.1758823, 0.4273642, 0.1640685, 0.1030634, 0.1657819, -0.0197110, 0.2073366, 0.0050953, -0.1467141, -0.1413236, -0.1404487]],[[0.0366294, 0.1421610, 0.2401645, 0.0672358, 0.2205958, 0.1306419, 0.0129892, 0.1626964, 0.0116193, -0.1181969, -0.1101109, -0.0844855],  
        #  [0.0366294, 0.1421610, 0.2401645, 0.0672358, 0.2205958, 0.1306419, 0.0129892, 0.1626964, 0.0116193, -0.1181969, -0.1101109, -0.0844855],  
        #  [0.0366294, 0.1421610, 0.2401645, 0.0672358, 0.2205958, 0.1306419, 0.0129892, 0.1626964, 0.0116193, -0.1181969, -0.1101109, -0.0844855]], 
        # [[0.0169496, 0.1236289, 0.1416115, -0.0382225, 0.2277734, 0.0378894, 0.0252284, 0.1317508, 0.0191879, -0.0379719, -0.0707748, -0.0134158], 
        #  [0.0169496, 0.1236289, 0.1416115, -0.0382225, 0.2277734, 0.0378894, 0.0252284, 0.1317508, 0.0191879, -0.0379719, -0.0707748, -0.0134158], 
        #  [0.0169496, 0.1236289, 0.1416115, -0.0382225, 0.2277734, 0.0378894, 0.0252284, 0.1317508, 0.0191879, -0.0379719, -0.0707748, -0.0134158]],[[0.0223647, 0.1227054, 0.0959055, -0.1043864, 0.2314414, -0.0289589, 0.0346038, 0.1147739, 0.0461321, 0.0998507, 0.0097069, 0.0886721],   
        #  [0.0223647, 0.1227054, 0.0959055, -0.1043864, 0.2314414, -0.0289589, 0.0346038, 0.1147739, 0.0461321, 0.0998507, 0.0097069, 0.0886721],   
        #  [0.0223647, 0.1227054, 0.0959055, -0.1043864, 0.2314414, -0.0289589, 0.0346038, 0.1147739, 0.0461321, 0.0998507, 0.0097069, 0.0886721]],  
        # [[0.0345177, 0.1308527, 0.0884205, -0.1468191, 0.2236451, -0.0705002, 0.0672482, 0.1278620, 0.1676001, 0.2955882, 0.2448514, 0.1802391],   
        #  [0.0345177, 0.1308527, 0.0884205, -0.1468191, 0.2236451, -0.0705002, 0.0672482, 0.1278620, 0.1676001, 0.2955882, 0.2448514, 0.1802391],   
        #  [0.0345177, 0.1308527, 0.0884205, -0.1468191, 0.2236451, -0.0705002, 0.0672482, 0.1278620, 0.1676001, 0.2955882, 0.2448514, 0.1802391]]   
        # ]
        # [
        # [[0.1687095, -0.2087553, 0.0254020, 0.3340017, 0.2515125, 0.2364762],
        #  [0.1687095, -0.2087553, 0.0254020, 0.3340017, 0.2515125, 0.2364762],
        #  [0.1687095, -0.2087553, 0.0254020, 0.3340017, 0.2515125, 0.2364762]],
        # [[0.2621196, 0.2436198, -0.1790378, 0.0883382, -0.0479185, -0.0838870],
        #  [0.2621196, 0.2436198, -0.1790378, 0.0883382, -0.0479185, -0.0838870],
        #  [0.2621196, 0.2436198, -0.1790378, 0.0883382, -0.0479185, -0.0838870]],
        # [[0.0345177, 0.1308527, 0.0884205, -0.1468191, 0.2236451, -0.0705002],
        #  [0.0345177, 0.1308527, 0.0884205, -0.1468191, 0.2236451, -0.0705002],
        #  [0.0345177, 0.1308527, 0.0884205, -0.1468191, 0.2236451, -0.0705002]],
        # [[-0.0197110, 0.2073366, 0.0050953, -0.1467141, -0.1413236, -0.1404487],
        #  [-0.0197110, 0.2073366, 0.0050953, -0.1467141, -0.1413236, -0.1404487],
        #  [-0.0197110, 0.2073366, 0.0050953, -0.1467141, -0.1413236, -0.1404487]]
        # ]
        # [
        # [[0.3588709, -0.3877619, 0.0519047, 0.5984558, 0.7709259, 1.0954115],
        #  [0.3588709, -0.3877619, 0.0519047, 0.5984558, 0.7709259, 1.0954115],
        #  [0.3588709, -0.3877619, 0.0519047, 0.5984558, 0.7709259, 1.0954115]],
        # [[0.4557160, 0.6420789, -0.4407433, 0.1704233, -0.1592798, -0.1966903],
        #  [0.4557160, 0.6420789, -0.4407433, 0.1704233, -0.1592798, -0.1966903],
        #  [0.4557160, 0.6420789, -0.4407433, 0.1704233, -0.1592798, -0.1966903]],
        # [[0.0681112, 0.4060420, 0.1333674, -0.3497016, 0.7122995, -0.1229735],
        #  [0.0681112, 0.4060420, 0.1333674, -0.3497016, 0.7122995, -0.1229735],
        #  [0.0681112, 0.4060420, 0.1333674, -0.3497016, 0.7122995, -0.1229735]],
        # [[-0.0378819, 0.4589431, 0.0142352, -0.3194987, -0.3059436, -0.3285254],
        #  [-0.0378819, 0.4589431, 0.0142352, -0.3194987, -0.3059436, -0.3285254],
        #  [-0.0378819, 0.4589431, 0.0142352, -0.3194987, -0.3059436, -0.3285254]]
        # ]

动态门控循环单元层
=================================

.. py:class:: pyvqnet.nn.gru.Dynamic_GRU(input_size,hidden_size, num_layers=1, batch_first=True, use_bias=True, bidirectional=False, dtype=None, name: str = '')
    
    对动态长度的输入序列应用多层门控循环单元（GRU）RNN。

    第一个输入应是通过 ``tensor.PackedSequence`` 类定义的变长批序列输入。
    ``tensor.PackedSequence`` 类可以通过依次调用以下函数来构造：``pad_sequence``、``pack_pad_sequence``。
    The ``tensor.PackedSequence`` class can be constructed as
    Call the next function in succession: ``pad_sequence``, ``pack_pad_sequence``.

    Dynamic_GRU的第一个输出也是一个 ``tensor.PackedSequence`` 类，
    可以使用 ``tensor.pad_pack_sequence`` 解包为常规的QTensor。

    对于输入序列中的每个元素，每层计算以下公式：

    .. math::
        \begin{array}{ll}
            r_t = \sigma(W_{ir} x_t + b_{ir} + W_{hr} h_{(t-1)} + b_{hr}) \\
            z_t = \sigma(W_{iz} x_t + b_{iz} + W_{hz} h_{(t-1)} + b_{hz}) \\
            n_t = \tanh(W_{in} x_t + b_{in} + r_t * (W_{hn} h_{(t-1)}+ b_{hn})) \\
            h_t = (1 - z_t) * n_t + z_t * h_{(t-1)}
        \end{array}

    :param input_size: 输入特征维度。
    :param hidden_size: 隐藏特征维度。
    :param num_layers: 循环层数。默认为1
    :param batch_first: 如果为True，输入形状为 [batch size, sequence length, feature dimension]；如果为False，输入形状为 [sequence length, batch size, feature dimension]，默认为True。
    :param use_bias: 如果为False，层不使用偏置权重b_ih和b_hh。默认为True。
    :param bidirectional: 如果为True，变为双向GRU。默认为False。
    :param dtype: 参数的数据类型，默认为None，使用默认数据类型kfloat32，表示32位浮点数。
    :param name: 输出层名称

    :return: 一个Dynamic_GRU类

    Example::

        from pyvqnet.nn import Dynamic_GRU
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
        print(seq_unpacked)
        print(lens_unpacked)
        # [
        # [[-0.3918380, 0.0056273, 0.9018179, 0.9006662],
        #  [-0.3715909, 0.0307644, 0.9756137, 0.9705784],
        #  [-0.3917399, 0.0057521, 0.9507942, 0.9456232]],
        # [[-0.6348240, -0.0603764, 0.9014163, 0.8903066],
        #  [0, 0, 0, 0],
        #  [-0.6333261, -0.0592172, 0.9660671, 0.9580816]],
        # [[-0.4571511, 0.0210018, 0.9151242, 0.9011748],
        #  [0, 0, 0, 0],
        #  [0, 0, 0, 0]],
        # [[-0.3585358, 0.0918219, 0.9496037, 0.9391552],
        #  [0, 0, 0, 0],
        #  [0, 0, 0, 0]]
        # ]
        # [4 1 2]

动态循环神经网络层
=================================

.. py:class:: pyvqnet.nn.rnn.Dynamic_RNN(input_size, hidden_size, num_layers=1, nonlinearity='tanh', batch_first=True, use_bias=True, bidirectional=False, dtype=None, name: str = '')
    
    对动态长度输入序列应用循环神经网络（RNN）。

    第一个输入应是通过 ``tensor.PackedSequence`` 类定义的变长批序列输入。
    ``tensor.PackedSequence`` 类可以通过依次调用以下函数来构造：``pad_sequence``、``pack_pad_sequence``。
    The ``tensor.PackedSequence`` class can be constructed as
    Call the next function in succession: ``pad_sequence``, ``pack_pad_sequence``.

    Dynamic_RNN的第一个输出也是一个 ``tensor.PackedSequence`` 类，
    可以使用 ``tensor.pad_pack_sequence`` 解包为常规的QTensor。

    循环神经网络（RNN）模块，使用 :math:`\tanh` 或 :math:`\text{ReLU}` 作为激活函数。支持双向、多层配置。
    单层单向RNN的计算公式如下：

    .. math::
        h_t = \tanh(W_{ih} x_t + b_{ih} + W_{hh} h_{(t-1)} + b_{hh})
    
    如果 :attr:`nonlinearity` 为 ``'relu'``，则 :math:`\text{ReLU}` 将替换 :math:`\tanh`。

    :param input_size: 输入特征维度。
    :param hidden_size: 隐藏特征维度。
    :param num_layers: 堆叠的RNN层数，默认为1。
    :param nonlinearity: 非线性激活函数，默认为 ``'tanh'``。
    :param batch_first: 如果为True，输入形状为 [batch size, sequence length, feature dimension]，
      如果为False，输入形状为 [sequence length, batch size, feature dimension]，默认为True。
    :param use_bias: 如果为False，模块不应用偏置项，默认为True。
    :param bidirectional: 如果为True，变为双向RNN，默认为False。
    :param dtype: 参数的数据类型，默认为None，使用默认数据类型kfloat32，表示32位浮点数。
    :param name: 输出层名称

    :return: Dynamic_RNN实例

    Example::

        from pyvqnet.nn import Dynamic_RNN
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
        print(seq_unpacked)
        print(lens_unpacked)

        # [
        # [[1.2980951, 0, 0, 0],
        #  [1.5040692, 0, 0, 0],
        #  [1.4927036, 0, 0, 0.1065927]],
        # [[2.6561704, 0, 0, 0.2532321],
        #  [0, 0, 0, 0],
        #  [3.1472805, 0, 0, 0]],
        # [[5.1231661, 0, 0, 0.7596353],
        #  [0, 0, 0, 0],
        #  [0, 0, 0, 0]],
        # [[8.4954977, 0, 0, 0.8191229],
        #  [0, 0, 0, 0],
        #  [0, 0, 0, 0]]
        # ]
        # [4 1 2]



动态长短期记忆网络层
=================================

.. py:class:: pyvqnet.nn.lstm.Dynamic_LSTM(input_size, hidden_size, num_layers=1, batch_first=True, use_bias=True, bidirectional=False, dtype=None, name: str = '')
    
    对动态长度输入序列应用长短期记忆网络（LSTM）RNN。

    第一个输入应是通过 ``tensor.PackedSequence`` 类定义的变长批序列输入。
    ``tensor.PackedSequence`` 类可以通过依次调用以下函数来构造：``pad_sequence``、``pack_pad_sequence``。
    The ``tensor.PackedSequence`` class can be constructed as
    Call the next function in succession: ``pad_sequence``, ``pack_pad_sequence``.

    Dynamic_LSTM的第一个输出也是一个 ``tensor.PackedSequence`` 类，
    可以使用 ``tensor.pad_pack_sequence`` 解包为常规的QTensor。

    循环神经网络（RNN）模块，使用 :math:`\tanh` 或 :math:`\text{ReLU}` 作为激活函数。支持双向、多层配置。
    单层单向RNN的计算公式如下：

    .. math::
        \begin{array}{ll} \\
            i_t = \sigma(W_{ii} x_t + b_{ii} + W_{hi} h_{t-1} + b_{hi}) \\
            f_t = \sigma(W_{if} x_t + b_{if} + W_{hf} h_{t-1} + b_{hf}) \\
            g_t = \tanh(W_{ig} x_t + b_{ig} + W_{hg} h_{t-1} + b_{hg}) \\
            o_t = \sigma(W_{io} x_t + b_{io} + W_{ho} h_{t-1} + b_{ho}) \\
            c_t = f_t \odot c_{t-1} + i_t \odot g_t \\
            h_t = o_t \odot \tanh(c_t) \\
        \end{array}

    :param input_size: 输入特征维度。
    :param hidden_size: 隐藏特征维度。
    :param num_layers: 堆叠的LSTM层数，默认为1。
    :param batch_first: 如果为True，输入形状为 [batch size, sequence length, feature dimension]，
      如果为False，输入形状为 [sequence length, batch size, feature dimension]，默认为True。
    :param use_bias: 如果为False，模块不应用偏置项，默认为True。
    :param bidirectional: 如果为True，变为双向LSTM，默认为False。
    :param dtype: 参数的数据类型，默认为None，使用默认数据类型kfloat32，表示32位浮点数。
    :param name: 输出层名称

    :return: Dynamic_LSTM实例

    Example::

        from pyvqnet.nn import Dynamic_LSTM
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

        print(seq_unpacked)
        print(lens_unpacked)

        # [
        # [[0.2038177, 0.1139005, 0.2312966, -0.1140076],
        #  [0.1992285, 0.1221137, 0.2277344, -0.3147154],
        #  [0.2293468, 0.0681745, 0.2426863, 0.2572871]],
        # [[0.1398094, -0.0150359, 0.2513067, 0.0783743],
        #  [0.1328388, -0.0031956, 0.2324090, -0.1962151],
        #  [0, 0, 0, 0]],
        # [[0.0898260, -0.0706460, 0.2396922, 0.2323916],
        #  [0.0817787, -0.0449937, 0.2388873, -0.0000469],
        #  [0, 0, 0, 0]],
        # [[0, 0, 0, 0],
        #  [0.0532839, -0.0870574, 0.2397324, 0.2103822],
        #  [0, 0, 0, 0]]
        # ]
        # [3 4 1]

插值层
=================================
.. py:class:: pyvqnet.nn.Interpolate(size, scale_factor, mode = "nearest", align_corners = None,  recompute_scale_factor = None, name = "")

    对输入进行下采样/上采样。

    目前仅支持四维输入数据。

    输入维度解释为：`B x C x H x W`。

    可用于调整大小的模式有：``nearest``、``bilinear``、``bicubic``。

    :param size: 输出空间大小。
    :param scale_factor: 空间大小的乘数。
    :param mode: 用于上采样的算法 ``nearest`` | ``bilinear`` | ``bicubic``。
    :param align_corners:  从几何角度，我们将输入和输出的像素视为正方形而非点。
            如果设置为 ``True``，输入和输出张量通过它们角像素的中心点对齐，保留角像素处的值。
            如果设置为 ``False``，输入和输出张量通过它们角像素的角点对齐，并且插值对超出边界的值使用边缘值填充，
            使此操作在保持 :attr:`scale_factor` 不变时 *独立于* 输入大小。仅在 :attr:`mode` 为 ``bilinear``、``bicubic`` 时有效。
            If set to ``False``, the input and output tensors are aligned by the corner
            points of their corner pixels, and the interpolation uses edge value padding
            for out-of-boundary values, making this operation *independent* of input size
            when :attr:`scale_factor` is kept the same. This only has an effect when :attr:`mode`
            is ``bilinear``, ``bicubic``.
    :param recompute_scale_factor: 重新计算用于插值计算的scale_factor。
    :param name: 模块名称。

    Example::

        from pyvqnet.nn import Interpolate
        from pyvqnet.tensor import tensor
        import pyvqnet
        pyvqnet.utils.set_random_seed(1)

        import numpy as np
        np.random.seed(0)

        np_ = np.random.randn(36).reshape((1, 1, 6, 6)).astype(np.float32)
        mode_ = "bilinear"
        size_ = 3

        class Model(pyvqnet.nn.Module):

            def __init__(self):
                super().__init__()
                self.inter = Interpolate(size = size_, mode=mode_)
                self.ln = pyvqnet.nn.Linear(9, 1)

            def forward(self, x):
                x = self.inter(x).reshape((1,-1))
                x = self.ln(x)
                return 2 * x

        input_vqnet = tensor.QTensor(np_,  dtype=pyvqnet.kfloat32, requires_grad=True)
        loss_pyvqnet = pyvqnet.nn.MeanSquaredError()
        model_vqnet = Model()
        output_vqnet = model_vqnet(input_vqnet)
        l = loss_pyvqnet(tensor.QTensor([[1.0]]), output_vqnet)
        l.backward()
        print(model_vqnet.parameters()[0].grad)


融合模块
=================================
.. py:class:: pyvqnet.nn.fuse_module(model)

    用于在推理阶段融合模型中对应的相邻模块为一个模块，
    从而减少模型推理阶段的计算量，提高模型推理速度。

    当前支持的模块序列如下：

    conv, bn

    linear, bn


    其他序列保持不变，其中列表中的第一个模块被替换为融合后的模块，其余被替换为 ``Identity``。

    :param input: 包含融合模块的模型。
    :return: 融合后的模型。

    Examples::
    
        from pyvqnet import tensor,kfloat32
        from pyvqnet.nn import Linear
        from pyvqnet.nn import Module, BatchNorm1d, BatchNorm2d, Conv1D, Conv2D

        from pyvqnet.qnn.vqc import *
        from pyvqnet.optim import Adam
        from pyvqnet.nn import Module,BinaryCrossEntropy, Sigmoid
        from pyvqnet.data import data_generator
        import numpy as np
        from pyvqnet.tensor import QTensor

        from time import time
        from pyvqnet.utils import set_random_seed
        from pyvqnet.nn import fuse_module

        def get_accuracy(result, label):
            result = (result > 0.5).astype(4)
            score = tensor.sums(result == label)
            return score.item()
            
        class Model(Module):
            def __init__(self):

                super(Model, self).__init__()

                self.conv1 = Conv2D(1,2,1)
                self.ban = BatchNorm2d(2)

                self.conv2 = Conv2D(2,1,1)
                self.li1 = Linear(64,1)
                self.ac = Sigmoid()
                
            def forward(self, x):
                x = self.conv1(x)
                x = self.ban(x)
                x = self.conv2(x).reshape([-1,64])
                x = self.li1(x)
                x = self.ac(x)

                return x
        X_train = np.random.randn(80, 1, 8, 8)
        y_train = np.random.choice([0,1], size=(80))
        
        model = Model().toGPU()
        optimizer = Adam(model.parameters(), lr = 0.001)
        batch_size = 20
        epoch = 80
        loss = BinaryCrossEntropy()
        print("start training..............")
        model.train()
        
        loss_history = []
        accuracy_history = []
        time2 = time()
        
        for i in range(epoch):
            count = 0
            sum_loss = 0
            accuary = 0
            t = 0
            for data, label in data_generator(X_train, y_train, batch_size, False):
                optimizer.zero_grad()
                data, label = QTensor(data,requires_grad=True).toGPU(), QTensor(label,
                                                    dtype=kfloat32,
                                                    requires_grad=False).toGPU()
                
                result = model(data)
                
                loss_b = loss(label.reshape([-1, 1]), result)
                
                loss_b.backward()
                optimizer._step()

                sum_loss += loss_b.item()
                count += batch_size
                accuary += get_accuracy(result, label.reshape([-1,1]))
                t = t + 1
            
            loss_history.append(sum_loss/count)
            accuracy_history.append(accuary/count)
            print(
                f"epoch:{i}, #### loss:{sum_loss/count} #####accuracy:{accuary/count}"
            )
        print(f"run time {time() - time2}")
        
        
        model.eval()

        input = tensor.randn((20, 1, 8, 8)).toGPU()
        print(list(model.named_children()))
        time_a = time()
        a = model(input)
        print(f"fuse before {time() - time_a}")
        fuse_module(model)
        model.toGPU()
        print(list(model.named_children()))
        time_b = time()
        b = model(input)
        print(f"fuse after {time() - time_b}")
        
        print(tensor.max(tensor.abs(a - b)).item())


缩放点积注意力机制
=================================
.. py:class:: pyvqnet.transformer.e2eqvit.SDPA(attn_mask=None,dropout_p=0.,scale=None,is_causal=False)

    SDPA缩放点积注意力机制。

    :param attn_mask: 注意力掩码；形状必须可广播到注意力权重的形状。
    :param dropout_p: Dropout概率；如果大于0.0，则应用dropout。
    :param scale: softmax之前应用的缩放因子。
    :param is_causal: 如果为true，假定为左上因果注意力掩码，如果同时设置了attn_mask和is_causal则会报错。
    
    Examples::
    
        from pyvqnet.transformer import SDPA
        from pyvqnet import tensor
        import pyvqnet
        from time import time
        import pyvqnet.nn as nn
        import numpy as np

        np.random.seed(42)

        query_np = np.random.randn(3, 3, 3, 5).astype(np.float32) 
        key_np = np.random.randn(3, 3, 3, 5).astype(np.float32)   
        value_np = np.random.randn(3, 3, 3, 5).astype(np.float32) 

        model = SDPA(tensor.QTensor([1.])).toGPU()

        query_p = tensor.QTensor(query_np, dtype=pyvqnet.kfloat32, requires_grad=True).toGPU()
        key_p = tensor.QTensor(key_np, dtype=pyvqnet.kfloat32, requires_grad=True).toGPU()
        value_p = tensor.QTensor(value_np, dtype=pyvqnet.kfloat32, requires_grad=True).toGPU()

        out_sdpa = model(query_p, key_p, value_p)

        out_sdpa.backward()


损失函数层
********************************************************

.. note::

        请注意，与pytorch等框架不同，在以下损失函数的前向函数中，第一个参数是标签，第二个参数是预测值。

均方误差
=================================

.. py:class:: pyvqnet.nn.MeanSquaredError

    创建一个衡量输入 :math:`x` 和目标 :math:`y` 之间均方误差（平方L2范数）的准则。


    未降维的损失可描述为：

    .. math::
        \ell(x, y) = L = \{l_1,\dots,l_N\}^\top, \quad
        l_n = \left( x_n - y_n \right)^2,



    .. math::
        \ell(x, y) =
            \operatorname{mean}(L)


    :math:`x` 和 :math:`y` 是具有各自总共 :math:`n` 个元素的任意形状的QTensor。


    均值操作仍然作用于所有元素，并除以 :math:`n`。

    :param name: 输出层名称

    :return: 一个MeanSquaredError类

    损失前向函数的参数：

        x: :math:`(N, *)` 其中 :math:`*` 表示任意数量的额外维度

        y: :math:`(N, *)`，与输入形状相同

    Example::
    
        from pyvqnet.tensor import QTensor
        from pyvqnet import kfloat64
        from pyvqnet.nn import MeanSquaredError
        y = QTensor([[0, 0, 1, 0, 0, 0, 0, 0, 0, 0]],
                    requires_grad=False,
                    dtype=kfloat64)
        x = QTensor([[0.1, 0.05, 0.7, 0, 0.05, 0.1, 0, 0, 0, 0]],
                    requires_grad=True,
                    dtype=kfloat64)

        loss_result = MeanSquaredError()
        result = loss_result(y, x)
        print(result)

        # [0.0115000]
        

二元交叉熵
=================================

.. py:class:: pyvqnet.nn.BinaryCrossEntropy

    衡量目标和输出之间的二元交叉熵：

    未降维的损失可描述为：

    .. math::
        \ell(x, y) = L = \{l_1,\dots,l_N\}^\top, \quad
        l_n = - w_n \left[ y_n \cdot \log x_n + (1 - y_n) \cdot \log (1 - x_n) \right],

    其中 :math:`N` 是批次大小。

    .. math::
        \ell(x, y) = \operatorname{mean}(L)

    :return: 一个BinaryCrossEntropy类

    损失前向函数的参数：

        x: :math:`(N, *)` 其中 :math:`*` 表示任意数量的额外维度

        y: :math:`(N, *)`，与输入形状相同

    Example::

        import pyvqnet
        from pyvqnet.tensor import QTensor
        x = QTensor([[0.3, 0.7, 0.2], [0.2, 0.3, 0.1]], requires_grad=True)
        y = QTensor([[0, 1.0, 0], [0, 0.0, 1]], requires_grad=True)

        loss_result = pyvqnet.nn.BinaryCrossEntropy()
        result = loss_result(y, x)
        result.backward()
        print(result)

        # [0.6364825]

类别交叉熵
=================================

.. py:class:: pyvqnet.nn.CategoricalCrossEntropy

    该准则将LogSoftmax和NLLLoss合并到一个类中。

    损失可描述如下，其中 `class` 是目标类别的索引：

    .. math::
        \text{loss}(x, class) = -\log\left(\frac{\exp(x[class])}{\sum_j \exp(x[j])}\right)
                       = -x[class] + \log\left(\sum_j \exp(x[j])\right)

    :return: 一个CategoricalCrossEntropy类

    损失前向函数的参数：

        x: :math:`(N, *)` 其中 :math:`*` 表示任意数量的额外维度

        y: :math:`(N, *)`，与输入形状相同，数据类型应为64位整数。

    Example::

        from pyvqnet.tensor import QTensor
        from pyvqnet import kfloat32,kint64
        from pyvqnet.nn import CategoricalCrossEntropy
        x = QTensor([[1, 2, 3, 4, 5],
        [1, 2, 3, 4, 5],
        [1, 2, 3, 4, 5]], requires_grad=True,dtype=kfloat32)
        y = QTensor([[0, 1, 0, 0, 0], [0, 1, 0, 0, 0], [1, 0, 0, 0, 0]], requires_grad=False,dtype=kint64)
        loss_result = CategoricalCrossEntropy()
        result = loss_result(y, x)
        print(result)

        # [3.7852428]

Softmax交叉熵
=================================

.. py:class:: pyvqnet.nn.SoftmaxCrossEntropy

    该准则将LogSoftmax和NLLLoss合并到单个类中，具有更好的数值稳定性。

    损失可描述如下，其中 `class` 是目标类别的索引：

    .. math::
        \text{loss}(x, class) = -\log\left(\frac{\exp(x[class])}{\sum_j \exp(x[j])}\right)
                       = -x[class] + \log\left(\sum_j \exp(x[j])\right)

    :return: 一个SoftmaxCrossEntropy类

    损失前向函数的参数：

        x: :math:`(N, *)` 其中 :math:`*` 表示任意数量的额外维度

        y: :math:`(N, *)`，与输入形状相同，数据类型应为64位整数。

    Example::

        from pyvqnet.tensor import QTensor
        from pyvqnet import kfloat32, kint64
        from pyvqnet.nn import SoftmaxCrossEntropy
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

        # [3.7852478]


负对数似然损失
=================================

.. py:class:: pyvqnet.nn.NLL_Loss()

    平均负对数似然损失。适用于具有 `C` 个类别的分类问题训练。

    通过前向调用传入的 `x` 应包含每个类别的对数概率。`x` 必须是大小为 :math:`(N, C)` 或 :math:`(N, C, d_1, d_2, ..., d_K)` 的张量，其中 :math:`K \geq 1` 对应 `K` 维情况。该损失期望的 `y` 应为范围 :math:`[0, C-1]` 内的类别索引，其中 `C = 类别数`。
     其中 :math:`K \geq 1` 对应 `K` 维情况。该损失期望的 `y` 应为范围 :math:`[0, C-1]` 内的类别索引，其中 `C = 类别数`。

    .. math::

        \ell(x, y) = L = \{l_1,\dots,l_N\}^\top, \quad
        l_n = -
            \sum_{n=1}^N \frac{1}{N}x_{n,y_n}, \quad

    :return: 一个NLL_Loss类

    损失前向函数的参数：

        x: :math:`(N, *)`，损失函数的输出，可以是多维变量。

        y: :math:`(N, *)`，损失函数期望的真实值，数据类型应为64位整数。


    Example::

        from pyvqnet.tensor import QTensor
        from pyvqnet import kfloat32,kint64
        from pyvqnet.nn import NLL_Loss

        x = QTensor([
            0.9476322568516703, 0.226547421131723, 0.5944201443911326,
            0.42830868492969476, 0.76414068655387, 0.00286059168094277,
            0.3574236812873617, 0.9096948856639084, 0.4560809854582528,
            0.9818027091583286, 0.8673569904602182, 0.9860275114020933,
            0.9232667066664217, 0.303693313961628, 0.8461034903175555
        ])
        x= x.reshape([1, 3, 1, 5])
        x.requires_grad = True
        y = QTensor([[[2, 1, 0, 0, 2]]], dtype=kint64)

        loss_result = NLL_Loss()
        result = loss_result(y, x)
        print(result)
        #[-0.6187226]

交叉熵损失
=================================

.. py:class:: pyvqnet.nn.CrossEntropyLoss()

    该准则将LogSoftmax和NLLLoss合并到单个类中。

    `x` 应包含每个类别的原始未归一化分数。`x` 必须是大小为 :math:`(C)`（无批次输入）、:math:`(N, C)` 或 :math:`(N, C, d_1, d_2, ..., d_K)`（其中 :math:`K \geq 1` 对应 `K` 维情况）的张量。

    损失可描述如下，其中 `class` 是目标类别的索引：

    .. math::

        \text{loss}(x, class) = -\log\left(\frac{\exp(x[class])}{\sum_j \exp(x[j])}\right)
                       = -x[class] + \log\left(\sum_j \exp(x[j])\right)

    :return: 一个CrossEntropyLoss类

    损失前向函数的参数：

        x: :math:`(N, *)`，损失函数的输出，可以是多维变量。

        y: :math:`(N, *)`，损失函数期望的真实值，数据类型应为64位整数。


    Example::

        from pyvqnet.tensor import QTensor
        from pyvqnet import kfloat32,kint64
        from pyvqnet.nn import CrossEntropyLoss
        x = QTensor([
            0.9476322568516703, 0.226547421131723, 0.5944201443911326,
            0.42830868492969476, 0.76414068655387, 0.00286059168094277,
            0.3574236812873617, 0.9096948856639084, 0.4560809854582528,
            0.9818027091583286, 0.8673569904602182, 0.9860275114020933,
            0.9232667066664217, 0.303693313961628, 0.8461034903175555
        ])
        x.reshape_([1, 3, 1, 5])
        x.requires_grad = True
        y = QTensor([[[2, 1, 0, 0, 2]]], dtype=kint64)

        loss_result = CrossEntropyLoss()
        result = loss_result(y, x)
        print(result)

        #[1.1508200]



激活函数
********************************************************


激活函数基类
=================================
.. py:class:: pyvqnet.nn.activation.Activation

    激活函数的基类。具体的激活函数继承此类。

Sigmoid激活函数
=================================
.. py:class:: pyvqnet.nn.Sigmoid(name: str = '')

        对给定层应用sigmoid激活函数。

        .. math::
            \text{Sigmoid}(x) = \frac{1}{1 + \exp(-x)}

        :param name: 输出层名称
        :return: sigmoid激活层

        Examples::

            from pyvqnet.nn import Sigmoid
            from pyvqnet.tensor import QTensor
            layer = Sigmoid()
            y = layer(QTensor([1.0, 2.0, 3.0, 4.0]))
            print(y)

            # [0.7310586, 0.8807970, 0.9525741, 0.9820138]

Softplus激活函数
=================================
.. py:class:: pyvqnet.nn.Softplus(name: str = '')

        对给定层应用softplus激活函数。

        .. math::
            \text{Softplus}(x) = \log(1 + \exp(x))

        :param name: 输出层名称
        :return: softplus激活层

    Examples::

        from pyvqnet.nn import Softplus
        from pyvqnet.tensor import QTensor
        layer = Softplus()
        y = layer(QTensor([1.0, 2.0, 3.0, 4.0]))
        print(y)

        # [1.3132616, 2.1269281, 3.0485873, 4.0181499]

Softsign激活函数
=================================
.. py:class:: pyvqnet.nn.Softsign(name: str = '')

        对给定层应用Softsign激活函数。

        .. math::
            \text{SoftSign}(x) = \frac{x}{ 1 + |x|}

        :param name: 输出层名称
        :return: softsign激活层

        Examples::

            from pyvqnet.nn import Softsign
            from pyvqnet.tensor import QTensor
            layer = Softsign()
            y = layer(QTensor([1.0, 2.0, 3.0, 4.0]))
            print(y)

            # [0.5000000, 0.6666667, 0.7500000, 0.8000000]

Softmax激活函数
=================================
.. py:class:: pyvqnet.nn.Softmax(axis: int = - 1, name: str = '')

    对给定层应用softmax激活函数。

    .. math::
        \text{Softmax}(x_{i}) = \frac{\exp(x_i)}{\sum_j \exp(x_j)}


    :param axis: 操作的维度（-1表示最后一个轴），默认为-1
    :param name: 输出层名称
    :return: softmax激活层

    Examples::

        from pyvqnet.nn import Softmax
        from pyvqnet.tensor import QTensor
        layer = Softmax()
        y = layer(QTensor([1.0, 2.0, 3.0, 4.0]))
        print(y)

        # [0.0320586, 0.0871443, 0.2368828, 0.6439142]


HardSigmoid激活函数
=================================
.. py:class:: pyvqnet.nn.HardSigmoid(name: str = '')

    对给定层应用hard sigmoid激活函数。

    .. math::
        \text{Hardsigmoid}(x) = \begin{cases}
            0 & \text{ if } x \le -3, \\
            1 & \text{ if } x \ge +3, \\
            x / 6 + 1 / 2 & \text{otherwise}
        \end{cases}

    :param name: 输出层名称
    :return: hard sigmoid激活层

    Examples::

        from pyvqnet.nn import HardSigmoid
        from pyvqnet.tensor import QTensor
        layer = HardSigmoid()
        y = layer(QTensor([1.0, 2.0, 3.0, 4.0]))
        print(y)

        # [0.6666667, 0.8333334, 1, 1]

ReLu激活函数
=================================
.. py:class:: pyvqnet.nn.ReLu(name: str = '')

    对给定层应用修正线性单元激活函数。

    .. math::
        \text{ReLu}(x) = \begin{cases}
        x, & \text{ if } x > 0\\
        0, & \text{ if } x \leq 0
        \end{cases}


    :param name: 输出层名称
    :return: ReLu激活层

    Examples::

        from pyvqnet.nn import ReLu
        from pyvqnet.tensor import QTensor
        layer = ReLu()
        y = layer(QTensor([-1, 2.0, -3, 4.0]))
        print(y)

        # [0, 2, 0, 4]

LeakyReLu激活函数
=================================
.. py:class:: pyvqnet.nn.LeakyReLu(alpha: float = 0.01, name: str = '')

    对给定层应用泄漏版本的修正线性单元激活函数。
    函数。

    .. math::
        \text{LeakyRelu}(x) =
        \begin{cases}
        x, & \text{ if } x \geq 0 \\
        \alpha * x, & \text{ otherwise }
        \end{cases}

    :param alpha: LeakyRelu系数，默认为0.01
    :param name: 输出层名称
    :return: leaky ReLu激活层

    Examples::

        from pyvqnet.nn import LeakyReLu
        from pyvqnet.tensor import QTensor
        layer = LeakyReLu()
        y = layer(QTensor([-1, 2.0, -3, 4.0]))
        print(y)

        # [-0.0100000, 2, -0.0300000, 4]

Gelu激活函数
=================================
.. py:class:: pyvqnet.nn.Gelu(approximate="tanh", name="")
    
    应用高斯误差线性单元函数：

    .. math:: \text{GELU}(x) = x * \Phi(x)

    当近似参数为'tanh'时，GELU通过以下方式估计：

    .. math:: \text{GELU}(x) = 0.5 * x * (1 + \text{Tanh}(\sqrt{2 / \pi} * (x + 0.044715 * x^3)))

    :param approximate: 近似计算方法，默认为"tanh"。
    :param name: 激活函数层名称，默认为""。

    :return: Gelu激活函数层实例。

    Examples::

        from pyvqnet.tensor import randu, ones_like
        from pyvqnet.nn import Gelu
        qa = randu([5,4])
        qb = Gelu()(qa)
        print(qb)
        # [[0.0292515,0.0668998,0.4036024,0.8369502],
        # [0.1929213,0.1981275,0.2358531,0.7790835],
        # [0.1754935,0.6204091,0.2354677,0.2409406],
        # [0.4238827,0.804715,0.1633414,0.2853],
        # [0.1959854,0.590143,0.553995,0.0008423]]

ELU激活函数
=================================
.. py:class:: pyvqnet.nn.ELU(alpha: float = 1.0, name: str = '')

    对给定层应用指数线性单元激活函数。

    .. math::
        \text{ELU}(x) = \begin{cases}
        x, & \text{ if } x > 0\\
        \alpha * (\exp(x) - 1), & \text{ if } x \leq 0
        \end{cases}

    :param alpha: Elu系数，默认为1.0
    :param name: 输出层名称
    :return: Elu激活层

    Examples::

        from pyvqnet.nn import ELU
        from pyvqnet.tensor import QTensor
        layer = ELU()
        y = layer(QTensor([-1, 2.0, -3, 4.0]))
        print(y)

        # [-0.6321205, 2, -0.9502130, 4]

Tanh激活函数
=================================
.. py:class:: pyvqnet.nn.Tanh(name: str = '')

    对给定层应用双曲正切激活函数。

    .. math::
        \text{Tanh}(x) = \frac{\exp(x) - \exp(-x)} {\exp(x) + \exp(-x)}

    :param name: 输出层名称
    :return: 双曲正切激活层

    Examples::

        from pyvqnet.nn import Tanh
        from pyvqnet.tensor import QTensor
        layer = Tanh()
        y = layer(QTensor([-1, 2.0, -3, 4.0]))
        print(y)

        # [-0.7615942, 0.9640276, -0.9950548, 0.9993293]


.. _Optimizer:

优化器模块
********************************************************


优化器基类
=================================
.. py:class:: pyvqnet.optim.optimizer.Optimizer(params, lr=0.01)

    所有优化器的基类。

    :param params: 需要优化的模型参数
    :param lr: 模型的学习率（默认为0.01）

Adadelta优化器
=================================
.. py:class:: pyvqnet.optim.adadelta.Adadelta(params, lr=0.01, beta=0.99, epsilon=1e-8)

    ADADELTA：自适应学习率方法。参考：(https://arxiv.org/abs/1212.5701)

    .. math::

        E(g_t^2) &= \beta * E(g_{t-1}^2) + (1-\beta) * g^2\\
        Square\_avg &= \sqrt{ ( E(dx_{t-1}^2) + \epsilon ) / ( E(g_t^2) + \epsilon ) }\\
        E(dx_t^2) &= \beta * E(dx_{t-1}^2) + (1-\beta) * (-g*square\_avg)^2 \\
        param\_new &= param - lr * Square\_avg

    :param params: 需要优化的模型参数
    :param lr: 模型的学习率（默认为0.01）
    :param beta: 用于计算平方梯度运行平均值的系数（默认为0.99）
    :param epsilon: 添加到分母以改善数值稳定性的项（默认为1e-8）
    :return: 一个Adadelta优化器

    Example::

        import numpy as np
        from pyvqnet.optim import adadelta
        from pyvqnet.tensor import QTensor
        w = np.arange(24).reshape(1,2,3,4).astype(np.float64)    
        param = QTensor(w)
        param.grad = QTensor(np.arange(24).reshape(1, 2, 3, 4).astype(np.float64))
        params = [param]
        opti = adadelta.Adadelta(params)

        for i in range(1,3):
            opti._step()
            print(param)

        # [
        # [[[0, 0.9999900, 1.9999900, 2.9999900],    
        #  [3.9999900, 4.9999900, 5.9999900, 6.9999900],     
        #  [7.9999900, 8.9999905, 9.9999905, 10.9999905]],   
        # [[11.9999905, 12.9999905, 13.9999905, 14.9999905], 
        #  [15.9999905, 16.9999905, 17.9999905, 18.9999905], 
        #  [19.9999905, 20.9999905, 21.9999905, 22.9999905]]]
        # ]

        # [
        # [[[0, 0.9999800, 1.9999800, 2.9999800],    
        #  [3.9999800, 4.9999800, 5.9999800, 6.9999800],     
        #  [7.9999800, 8.9999800, 9.9999800, 10.9999800]],   
        # [[11.9999800, 12.9999800, 13.9999800, 14.9999800], 
        #  [15.9999800, 16.9999809, 17.9999809, 18.9999809], 
        #  [19.9999809, 20.9999809, 21.9999809, 22.9999809]]]
        # ]

Adagrad优化器
=================================
.. py:class:: pyvqnet.optim.adagrad.Adagrad(params, lr=0.01, epsilon=1e-8 )

    实现Adagrad算法。参考：(https://databricks.com/glossary/adagrad)

    .. math::
        \begin{aligned}
        moment\_new &= moment + g * g\\param\_new
        &= param - \frac{lr * g}{\sqrt{moment\_new} + \epsilon}
        \end{aligned}

    :param params: 需要优化的模型参数
    :param lr: 模型的学习率（默认为0.01）
    :param epsilon: 添加到分母以改善数值稳定性的项（默认为1e-8）
    :return: 一个Adagrad优化器

    Example::

        import numpy as np
        from pyvqnet.optim import adagrad
        from pyvqnet.tensor import QTensor
        w = np.arange(24).reshape(1,2,3,4).astype(np.float64)    
        param = QTensor(w)
        param.grad = QTensor(np.arange(24).reshape(1, 2, 3, 4).astype(np.float64))
        params = [param]
        opti = adagrad.Adagrad(params)

        for i in range(1,3):
            opti._step() 
            print(param)

        # [
        # [[[0, 0.9900000, 1.9900000, 2.9900000],
        #  [3.9900000, 4.9899998, 5.9899998, 6.9899998],
        #  [7.9899998, 8.9899998, 9.9899998, 10.9899998]],
        # [[11.9899998, 12.9899998, 13.9899998, 14.9899998],
        #  [15.9899998, 16.9899998, 17.9899998, 18.9899998],
        #  [19.9899998, 20.9899998, 21.9899998, 22.9899998]]]
        # ]

        # [
        # [[[0, 0.9829289, 1.9829290, 2.9829290],
        #  [3.9829290, 4.9829288, 5.9829288, 6.9829288],
        #  [7.9829288, 8.9829283, 9.9829283, 10.9829283]],
        # [[11.9829283, 12.9829283, 13.9829283, 14.9829283],
        #  [15.9829283, 16.9829292, 17.9829292, 18.9829292],
        #  [19.9829292, 20.9829292, 21.9829292, 22.9829292]]]
        # ]

AdamW优化器
=================================
.. py:class:: pyvqnet.optim.adam.AdamW(params, lr=0.01, beta1=0.9, beta2=0.999, epsilon=1e-8, weight_decay=0.01, amsgrad: bool = False)
    
    实现AdamW算法。

    .. math::
        t=t+1

    .. math::
        param\_new = param - lr*weight\_decay*param
    .. math::
        moment\_1\_new=\beta1*moment\_1+(1−\beta1)g
    .. math::
        moment\_2\_new=\beta2*moment\_2+(1−\beta2)g*g
    .. math::
        lr = lr*\frac{\sqrt{1-\beta2^t}}{1-\beta1^t}
    
    如果参数amsgrad为True

    .. math::
        moment\_2\_max = max(moment\_2\_max,moment\_2)
    .. math::
        param\_new=param\_new-lr*\frac{moment\_1}{\sqrt{moment\_2\_max}+\epsilon}
    
    否则：

    .. math::
        param\_new=param\_new-lr*\frac{moment\_1}{\sqrt{moment\_2}+\epsilon}

    :param params: 需要优化的模型参数。
    :param lr: 学习率（默认为0.01）。
    :param beta1: 用于计算梯度及其平方运行平均值的系数（默认为0.9）。
    :param beta2: 用于计算梯度及其平方运行平均值的系数（默认为0.999）。
    :param epsilon: 添加到分母以改善数值稳定性的常数（默认为1e-8）。
    :param weight_decay: 权重衰减系数，默认为0.01。
    :param amsgrad: 是否使用此算法的AMSGrad变体（默认为False）。
    :return: 一个AdamW优化器。

    Example::

        from pyvqnet.optim import adam
        import numpy as np
        from pyvqnet.tensor import QTensor
        w = np.arange(24).reshape(1,2,3,4).astype(np.float64)
        param = QTensor(w)
        param.grad = QTensor(np.arange(24).reshape(1,2,3,4).astype(np.float64))
        params = [param]
        opti = adam.AdamW(params, lr=0.5)

        for i in range(1,3):
            opti.step()
        print(param)
        # [[[[ 0. ,-0.007475 , 0.98255 , 1.972575 ],
        # [2.9626, 3.952625, 4.9426501, 5.9326751],
        # [6.9227001, 7.9127251, 8.9027501, 9.8927751]],

        # [[10.8828001,11.8728251,12.8628501,13.8528751],
        # [14.8429002,15.8329252,16.8229502,17.8129752],
        # [18.8030002,19.7930252,20.7830502,21.7730752]]]]

Adam优化器
=================================
.. py:class:: pyvqnet.optim.adam.Adam(params, lr=0.01, beta1=0.9, beta2=0.999, epsilon=1e-8,weight_decay = 0, amsgrad: bool = False)

    Adam：一种随机优化方法。参考：(https://arxiv.org/abs/1412.6980)，它利用梯度的一阶矩估计和二阶矩估计动态调整每个参数的学习率。

    .. math::
        t = t + 1
    .. math::
        param  = param - lr*weight\_decay*param
    .. math::
        moment\_1\_new=\beta1∗moment\_1+(1−\beta1)g
    .. math::
        moment\_2\_new=\beta2∗moment\_2+(1−\beta2)g*g
    .. math::
        lr = lr*\frac{\sqrt{1-\beta2^t}}{1-\beta1^t}

    如果amsgrad = True
    
    .. math::
        moment\_2\_max = max(moment\_2\_max,moment\_2)
    .. math::
        param\_new=param-lr*\frac{moment\_1}{\sqrt{moment\_2\_max}+\epsilon} 

    否则

    .. math::
        param\_new=param-lr*\frac{moment\_1}{\sqrt{moment\_2}+\epsilon} 


    :param params: 需要优化的模型参数
    :param lr: 模型的学习率（默认为0.01）
    :param beta1: 用于计算梯度及其平方运行平均值的系数（默认为0.9）
    :param beta2: 用于计算梯度及其平方运行平均值的系数（默认为0.999）
    :param epsilon: 添加到分母以改善数值稳定性的项（默认为1e-8）
    :param weight_decay: 权重衰减系数，默认为0。
    :param amsgrad: 是否使用此算法的AMSGrad变体（默认为False）
    :return: 一个Adam优化器

    Example::

        import numpy as np
        from pyvqnet.optim import adam
        from pyvqnet.tensor import QTensor
        w = np.arange(24).reshape(1,2,3,4).astype(np.float64)    
        param = QTensor(w)
        param.grad = QTensor(np.arange(24).reshape(1, 2, 3, 4).astype(np.float64))
        params = [param]
        opti = adam.Adam(params)
        
        for i in range(1,3):
            opti._step()
            print(param)

        # [
        # [[[0, 0.9900000, 1.9900000, 2.9900000],
        #  [3.9900000, 4.9899998, 5.9899998, 6.9899998],
        #  [7.9899998, 8.9899998, 9.9899998, 10.9899998]],
        # [[11.9899998, 12.9899998, 13.9899998, 14.9899998],
        #  [15.9899998, 16.9899998, 17.9899998, 18.9899998],
        #  [19.9899998, 20.9899998, 21.9899998, 22.9899998]]]
        # ]

        # [
        # [[[0, 0.9800000, 1.9800000, 2.9800000],
        #  [3.9800000, 4.9799995, 5.9799995, 6.9799995],
        #  [7.9799995, 8.9799995, 9.9799995, 10.9799995]],
        # [[11.9799995, 12.9799995, 13.9799995, 14.9799995],
        #  [15.9799995, 16.9799995, 17.9799995, 18.9799995],
        #  [19.9799995, 20.9799995, 21.9799995, 22.9799995]]]
        # ]

Adamax优化器
=================================
.. py:class:: pyvqnet.optim.adamax.Adamax(params, lr=0.01, beta1=0.9, beta2=0.999, epsilon=1e-8)

    实现Adamax算法（基于无穷范数的Adam变体）。参考：(https://arxiv.org/abs/1412.6980)

    .. math::
        \\t = t + 1
    .. math::
        moment\_new=\beta1∗moment+(1−\beta1)g
    .. math::
        norm\_new = \max{(\beta1∗norm+\epsilon, \left|g\right|)}
    .. math::
        lr = \frac{lr}{1-\beta1^t}
    .. math::
        param\_new = param − lr*\frac{moment\_new}{norm\_new}\\

    :param params: 需要优化的模型参数
    :param lr: 模型的学习率（默认为0.01）
    :param beta1: 用于计算梯度及其平方运行平均值的系数（默认为0.9）
    :param beta2: 用于计算梯度及其平方运行平均值的系数（默认为0.999）
    :param epsilon: 添加到分母以改善数值稳定性的项（默认为1e-8）
    :return: 一个Adamax优化器

    Example::

        import numpy as np
        from pyvqnet.optim import adamax
        from pyvqnet.tensor import QTensor
        w = np.arange(24).reshape(1,2,3,4).astype(np.float64)    
        param = QTensor(w)
        param.grad = QTensor(np.arange(24).reshape(1,2,3,4).astype(np.float64))
        params = [param]
        opti = adamax.Adamax(params)
        
        for i in range(1,3):
            opti._step()
            print(param)

        # [
        # [[[0, 0.9900000, 1.9900000, 2.9900000],
        #  [3.9900000, 4.9899998, 5.9899998, 6.9899998],
        #  [7.9899998, 8.9899998, 9.9899998, 10.9899998]],
        # [[11.9899998, 12.9899998, 13.9899998, 14.9899998],
        #  [15.9899998, 16.9899998, 17.9899998, 18.9899998],
        #  [19.9899998, 20.9899998, 21.9899998, 22.9899998]]]
        # ]

        # [
        # [[[0, 0.9800000, 1.9800000, 2.9800000],
        #  [3.9800000, 4.9799995, 5.9799995, 6.9799995],
        #  [7.9799995, 8.9799995, 9.9799995, 10.9799995]],
        # [[11.9799995, 12.9799995, 13.9799995, 14.9799995],
        #  [15.9799995, 16.9799995, 17.9799995, 18.9799995],
        #  [19.9799995, 20.9799995, 21.9799995, 22.9799995]]]
        # ]

RMSProp优化器
=================================
.. py:class:: pyvqnet.optim.rmsprop.RMSProp(params, lr=0.01, beta=0.99, epsilon=1e-8)

    实现RMSprop算法。参考：(https://arxiv.org/pdf/1308.0850v5.pdf)

    .. math::
        s_{t+1} = s_{t} + (1 - \beta)*(g)^2

    .. math::
        param_new = param -  \frac{g}{\sqrt{s_{t+1}} + epsilon}

    :param params: 需要优化的模型参数
    :param lr: 模型的学习率（默认为0.01）
    :param beta: 用于计算梯度及其平方运行平均值的系数（默认为0.99）
    :param epsilon: 添加到分母以改善数值稳定性的项（默认为1e-8）
    :return: 一个RMSProp优化器

    Example::

        import numpy as np
        from pyvqnet.optim import rmsprop
        from pyvqnet.tensor import QTensor
        w = np.arange(24).reshape(1,2,3,4).astype(np.float64)    
        param = QTensor(w)
        param.grad = QTensor(np.arange(24).reshape(1,2,3,4).astype(np.float64))
        params = [param]
        opti = rmsprop.RMSProp(params)
        
        for i in range(1,3):
            opti._step()
            print(param)

        # [
        # [[[0, 0.9000000, 1.9000000, 2.8999999],
        #  [3.8999999, 4.9000001, 5.9000001, 6.9000001],
        #  [7.9000001, 8.8999996, 9.8999996, 10.8999996]],
        # [[11.8999996, 12.8999996, 13.8999996, 14.8999996],
        #  [15.8999996, 16.8999996, 17.8999996, 18.8999996],
        #  [19.8999996, 20.8999996, 21.8999996, 22.8999996]]]
        # ]

        # [
        # [[[0, 0.8291118, 1.8291118, 2.8291118],
        #  [3.8291118, 4.8291121, 5.8291121, 6.8291121],
        #  [7.8291121, 8.8291111, 9.8291111, 10.8291111]],
        # [[11.8291111, 12.8291111, 13.8291111, 14.8291111],
        #  [15.8291111, 16.8291111, 17.8291111, 18.8291111],
        #  [19.8291111, 20.8291111, 21.8291111, 22.8291111]]]
        # ]

SGD优化器
=================================
.. py:class:: pyvqnet.optim.sgd.SGD(params, lr=0.01, momentum=0, nesterov=False)

    实现SGD算法。参考：(https://en.wikipedia.org/wiki/Stochastic_gradient_descent)

    .. math::

        \\param\_new=param-lr*g\\

    :param params: 需要优化的模型参数
    :param lr: 模型的学习率（默认为0.01）
    :param momentum: 动量因子（默认为0）
    :param nesterov: 启用Nesterov动量（默认为False）
    :return: 一个SGD优化器

    Example::

        import numpy as np
        from pyvqnet.optim import sgd
        from pyvqnet.tensor import QTensor
        w = np.arange(24).reshape(1,2,3,4).astype(np.float64)    
        param = QTensor(w)
        param.grad = QTensor(np.arange(24).reshape(1,2,3,4).astype(np.float64))
        params = [param]
        opti = sgd.SGD(params)

        for i in range(1,3):
            opti._step()
            print(param) 

        # [
        # [[[0, 0.9900000, 1.9800000, 2.9700000],
        #  [3.9600000, 4.9499998, 5.9400001, 6.9299998],
        #  [7.9200001, 8.9099998, 9.8999996, 10.8900003]],
        # [[11.8800001, 12.8699999, 13.8599997, 14.8500004],
        #  [15.8400002, 16.8299999, 17.8199997, 18.8099995],
        #  [19.7999992, 20.7900009, 21.7800007, 22.7700005]]]
        # ]

        # [
        # [[[0, 0.9800000, 1.9600000, 2.9400001],
        #  [3.9200001, 4.8999996, 5.8800001, 6.8599997],
        #  [7.8400002, 8.8199997, 9.7999992, 10.7800007]],
        # [[11.7600002, 12.7399998, 13.7199993, 14.7000008],
        #  [15.6800003, 16.6599998, 17.6399994, 18.6199989],
        #  [19.5999985, 20.5800018, 21.5600014, 22.5400009]]]
        # ]


评估指标
********************************************************


MSE（均方误差）
=================================

.. py:class:: pyvqnet.utils.metrics.MSE(y_true_Qtensor, y_pred_Qtensor)

    MSE：均方误差。

    :param y_true_Qtensor: 形状如 (n_samples,) 或 (n_samples, n_outputs) 的QTensor，真实目标值。
    :param y_pred_Qtensor: 形状如 (n_samples,) 或 (n_samples, n_outputs) 的QTensor，估计目标值。
    :return: 返回float类型结果。

    Example::

            import numpy as np
            from pyvqnet.tensor import tensor
            from pyvqnet.utils import metrics as vqnet_metrics
            from pyvqnet import _core
            _vqnet = _core.vqnet

            y_true_Qtensor = tensor.arange(1, 12)
            y_pred_Qtensor = tensor.arange(4, 15)
            result = vqnet_metrics.MSE(y_true_Qtensor, y_pred_Qtensor)
            print(result)
            # 9.0

            y_true_Qtensor = tensor.arange(1, 13).reshape([3, 4])
            y_pred_Qtensor = tensor.arange(4, 16).reshape([3, 4])
            result = vqnet_metrics.MSE(y_true_Qtensor, y_pred_Qtensor)
            print(result)
            # 9.0


RMSE（均方根误差）
=================================

.. py:class:: pyvqnet.utils.metrics.RMSE(y_true_Qtensor, y_pred_Qtensor)

    RMSE：均方根误差。

    :param y_true_Qtensor: 形状如 (n_samples,) 或 (n_samples, n_outputs) 的QTensor，真实目标值。
    :param y_pred_Qtensor: 形状如 (n_samples,) 或 (n_samples, n_outputs) 的QTensor，估计目标值。
    :return: 返回float类型结果。

    Example::

            import numpy as np
            from pyvqnet.tensor import tensor
            from pyvqnet.utils import metrics as vqnet_metrics
            from pyvqnet import _core
            _vqnet = _core.vqnet

            y_true_Qtensor = tensor.arange(1, 12)
            y_pred_Qtensor = tensor.arange(4, 15)
            result = vqnet_metrics.RMSE(y_true_Qtensor, y_pred_Qtensor)
            print(result)
            # 3.0

            y_true_Qtensor = tensor.arange(1, 13).reshape([3, 4])
            y_pred_Qtensor = tensor.arange(4, 16).reshape([3, 4])
            result = vqnet_metrics.RMSE(y_true_Qtensor, y_pred_Qtensor)
            print(result)
            # 3.0



MAE（平均绝对误差）
=================================

.. py:class:: pyvqnet.utils.metrics.MAE(y_true_Qtensor, y_pred_Qtensor)

    MAE：平均绝对误差。

    :param y_true_Qtensor: 形状如 (n_samples,) 或 (n_samples, n_outputs) 的QTensor，真实目标值。
    :param y_pred_Qtensor: 形状如 (n_samples,) 或 (n_samples, n_outputs) 的QTensor，估计目标值。
    :return: 返回float类型结果。

    Example::

            import numpy as np
            from pyvqnet.tensor import tensor
            from pyvqnet.utils import metrics as vqnet_metrics
            from pyvqnet import _core
            _vqnet = _core.vqnet

            y_true_Qtensor = tensor.arange(1, 12)
            y_pred_Qtensor = tensor.arange(4, 15)
            result = vqnet_metrics.MAE(y_true_Qtensor, y_pred_Qtensor)
            print(result)
            # 3.0

            y_true_Qtensor = tensor.arange(1, 13).reshape([3, 4])
            y_pred_Qtensor = tensor.arange(4, 16).reshape([3, 4])
            result = vqnet_metrics.MAE(y_true_Qtensor, y_pred_Qtensor)
            print(result)
            # 3.0


R²决定系数
=================================

.. py:class:: pyvqnet.utils.metrics.R_Square(y_true_Qtensor, y_pred_Qtensor, sample_weight=None)

    R_Square：R²（决定系数）回归得分函数。
    最佳可能得分为1.0，也可能为负数
    （因为模型可以任意恶化）。
    总是预测y的期望值、忽略输入特征的模型，其R²得分为0.0。
    忽略输入特征的模型，其R²得分为0.0。
    
    :param y_true_Qtensor: 形状如 (n_samples,) 或 (n_samples, n_outputs) 的QTensor，真实目标值。
    :param y_pred_Qtensor: 形状如 (n_samples,) 或 (n_samples, n_outputs) 的QTensor，估计目标值。
    :param sample_weight: 形状如 (n_samples,) 的数组，可选的样本权重，默认为None。
    :return: 返回float类型结果。

    Example::

            import numpy as np
            from pyvqnet.tensor import tensor
            from pyvqnet.utils import metrics as vqnet_metrics
            from pyvqnet import _core
            _vqnet = _core.vqnet

            y_true_Qtensor = tensor.arange(1, 12)
            y_pred_Qtensor = tensor.arange(4, 15)
            result = vqnet_metrics.R_Square(y_true_Qtensor, y_pred_Qtensor)
            print(result)
            # 0.09999999999999998

            y_true_Qtensor = tensor.arange(1, 13).reshape([3, 4])
            y_pred_Qtensor = tensor.arange(4, 16).reshape([3, 4])
            result = vqnet_metrics.R_Square(y_true_Qtensor, y_pred_Qtensor)
            print(result)
            # 0.15625


二分类精确率-召回率-F1分数
=================================

.. py:class:: pyvqnet.utils.metrics.precision_recall_f1_2_score(y_true_Qtensor, y_pred_Qtensor)

    计算二分类任务下预测值的精确率、召回率和F1分数。预测值和真实值需要是形状相似的 (n_samples, ) 的QTensor，值为0或1，代表两个类别的标签。
    
    :param y_true_Qtensor: 一维QTensor，真实目标值。
    :param y_pred_Qtensor: 一维QTensor，估计目标值。

    :returns:
        - precision - 精确率结果
        - recall - 召回率结果
        - f1 - F1分数

    Example::

            import numpy as np
            from pyvqnet.tensor import tensor
            from pyvqnet.utils import metrics as vqnet_metrics
            from pyvqnet import _core
            _vqnet = _core.vqnet

            y_true_Qtensor = tensor.QTensor([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
            y_pred_Qtensor = tensor.QTensor([0, 0, 1, 1, 1, 0, 0, 1, 1, 1])

            precision, recall, f1 = vqnet_metrics.precision_recall_f1_2_score(
                y_true_Qtensor, y_pred_Qtensor)
            print(precision, recall, f1)
            # 0.5 0.6 0.5454545454545454


多分类精确率-召回率-F1分数
=================================

.. py:class:: pyvqnet.utils.metrics.precision_recall_f1_N_score(y_true_Qtensor, y_pred_Qtensor, N, average)

    多分类任务的精确率、召回率和F1分数计算。其中预测值和真实值是形状相似的 (n_samples, ) 的QTensor，值为0到N-1的整数，代表N个类别的标签。

    :param y_true_Qtensor: 一维QTensor，真实目标值。
    :param y_pred_Qtensor: 一维QTensor，估计目标值。
    :param N: N个类别（类别数）。
    :param average: 字符串，['micro', 'macro', 'weighted']。
             此参数用于多分类/多标签目标。
             
             ``'micro'``：通过计算总的真正例、假负例和假正例来全局计算指标。
             
             ``'macro'``：计算每个标签的指标并找出其未加权平均值。意味着不考虑标签的平衡性。
             
             ``'weighted'``：计算每个标签的指标并找出它们的平均值（每个标签的真正例数量）。这会改变 ``'macro'`` 以考虑标签不平衡；这可能导致F分数不在精确率和召回率之间。
    
    :returns:
        - precision - 精确率结果
        - recall - 召回率结果
        - f1 - F1分数

    Example::

                import numpy as np
                from pyvqnet.tensor import tensor
                from pyvqnet.utils import metrics as vqnet_metrics
                from pyvqnet import _core
                _vqnet = _core.vqnet

                reference_list = [1, 1, 2, 2, 2, 3, 3, 3, 3, 3]
                prediciton_list = [1, 2, 2, 2, 3, 1, 2, 3, 3, 3]
                y_true_Qtensor = tensor.QTensor(reference_list)
                y_pred_Qtensor = tensor.QTensor(prediciton_list)

                precision_micro, recall_micro, f1_micro = vqnet_metrics.precision_recall_f1_N_score(
                    y_true_Qtensor, y_pred_Qtensor, 3, average='micro')
                print(precision_micro, recall_micro, f1_micro)
                # 0.6 0.6 0.6

                precision_macro, recall_macro, f1_macro = vqnet_metrics.precision_recall_f1_N_score(
                    y_true_Qtensor, y_pred_Qtensor, 3, average='macro')
                print(precision_macro, recall_macro, f1_macro)
                # 0.5833333333333334 0.5888888888888889 0.5793650793650794

                precision_weighted, recall_weighted, f1_weighted = vqnet_metrics.precision_recall_f1_N_score(
                    y_true_Qtensor, y_pred_Qtensor, 3, average='weighted')
                print(precision_weighted, recall_weighted, f1_weighted)
                # 0.625 0.6 0.6047619047619047



多标签精确率-召回率-F1分数
=================================

.. py:class:: pyvqnet.utils.metrics.precision_recall_f1_Multi_score(y_true_Qtensor, y_pred_Qtensor, N, average)

    多分类任务的精确率、召回率和F1分数计算。其中预测值和真实值是形状相似的 (n_samples, N) 的QTensor，值为N维独热编码的标签值。

    :param y_true_Qtensor: 一维QTensor，真实目标值。
    :param y_pred_Qtensor: 一维QTensor，估计目标值。
    :param N: N个类别（类别数）。
    :param average: 字符串，['micro', 'macro', 'weighted']。
             此参数用于多分类/多标签目标。
             
             ``'micro'``：通过计算总的真正例、假负例和假正例来全局计算指标。
             
             ``'macro'``：计算每个标签的指标并找出其未加权平均值。意味着不考虑标签的平衡性。
             
             ``'weighted'``：计算每个标签的指标并找出它们的平均值（每个标签的真正例数量）。这会改变 ``'macro'`` 以考虑标签不平衡；这可能导致F分数不在精确率和召回率之间。
    
    :returns:
        - precision - 精确率结果
        - recall - 召回率结果
        - f1 - F1分数

    Example::


                    import numpy as np
                    from pyvqnet.tensor import tensor
                    from pyvqnet.utils import metrics as vqnet_metrics
                    from pyvqnet import _core
                    _vqnet = _core.vqnet

                    reference_list = [[1, 0], [0, 1], [0, 0], [1, 1], [1, 0]]
                    prediciton_list = [[1, 0], [0, 0], [1, 0], [0, 0], [0, 0]]
                    y_true_Qtensor = tensor.QTensor(reference_list)
                    y_pred_Qtensor = tensor.QTensor(prediciton_list)

                    micro_precision, micro_recall, micro_f1 = vqnet_metrics.precision_recall_f1_Multi_score(y_true_Qtensor,
                                y_pred_Qtensor, 2, average='micro')
                    print(micro_precision, micro_recall, micro_f1)
                    # 0.5 0.2 0.28571428571428575

                    macro_precision, macro_recall, macro_f1 = vqnet_metrics.precision_recall_f1_Multi_score(y_true_Qtensor,
                                y_pred_Qtensor, 2, average='macro')
                    print(macro_precision, macro_recall, macro_f1)
                    # 0.25 0.16666666666666666 0.2

                    weighted_precision, weighted_recall, weighted_f1 = vqnet_metrics.precision_recall_f1_Multi_score(y_true_Qtensor,
                                y_pred_Qtensor, 2, average='weighted')
                    print(weighted_precision, weighted_recall, weighted_f1)
                    # 0.3 0.19999999999999998 0.24

                    reference_list = [[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0], [1, 0, 1]]
                    prediciton_list = [[1, 0, 0], [1, 0, 0], [1, 1, 1], [1, 0, 0], [0, 1, 1]]
                    y_true_Qtensor = tensor.QTensor(reference_list)
                    y_pred_Qtensor = tensor.QTensor(prediciton_list)

                    micro_precision, micro_recall, micro_f1 = vqnet_metrics.precision_recall_f1_Multi_score(y_true_Qtensor,
                                y_pred_Qtensor, 3, average='micro')
                    print(micro_precision, micro_recall, micro_f1) # 0.5 0.5714285714285714 0.5333333333333333

                    macro_precision, macro_recall, macro_f1 = vqnet_metrics.precision_recall_f1_Multi_score(y_true_Qtensor,
                                y_pred_Qtensor, 3, average='macro')
                    print(macro_precision, macro_recall, macro_f1)
                    # 0.5 0.5555555555555555 0.5238095238095238

                    weighted_precision, weighted_recall, weighted_f1 = vqnet_metrics.precision_recall_f1_Multi_score(y_true_Qtensor,
                                y_pred_Qtensor, 3, average='weighted')
                    print(weighted_precision, weighted_recall, weighted_f1)
                    # 0.5 0.5714285714285714 0.5306122448979592



AUC计算
=================================

.. py:class:: pyvqnet.utils.metrics.auc_calculate(y_true_Qtensor, y_pred_Qtensor, pos_label=None, sample_weight=None, drop_intermediate=True)

    计算分类任务的精确率、召回率和f1分数。

    :param y_true_Qtensor: 形状为 [n_samples] 的QTensor。
                             一个真实的二元标签。如果标签不是{-1,1}或{0,1}，应显式给出pos_label。
    :param y_pred_Qtensor: 形状为 [n_samples] 的QTensor。
                             目标分数，可以是正概率估计类、置信值或决策的非阈值度量（由某些分类器上的"decision_function"返回）
    :param pos_label: int或str。正类别的标签。默认为None。
                      当 ``pos_label`` 为None时，如果 ``y_true_Qtensor`` 在{-1,1}或{0,1}中，则 ``pos_label`` 设置为1，否则将引发错误。
    :param sample_weight: 形状为 (n_samples,) 的数组，默认为None。
    :param drop_intermediate: 布尔值，可选（默认为True）。
                     是否丢弃一些不在绘制的ROC曲线上出现的次优阈值。
    :return: 输出float类型结果。

    Example::

                import numpy as np
                from pyvqnet.tensor import tensor
                from pyvqnet.utils import metrics as vqnet_metrics
                from pyvqnet import _core
                _vqnet = _core.vqnet

                y = np.array([1, 1, 1, 1, 0, 1, 0, 0, 0, 0])
                pred = np.array([0.9, 0.8, 0.7, 0.6, 0.6, 0.4, 0.4, 0.3, 0.2, 0.1])
                y_Qtensor = tensor.QTensor(y)
                pred_Qtensor = tensor.QTensor(pred)
                result = vqnet_metrics.auc_calculate(y_Qtensor, pred_Qtensor)
                print("auc:", result)
                # 0.92

                y = np.array([1, 1, 1, 1, 1, 0, 0, 1, 1, 1])
                pred = np.array([1, 0, 1, 1, 1, 1, 0, 1, 1, 0])
                y_Qtensor = tensor.QTensor(y)
                pred_Qtensor = tensor.QTensor(pred)
                result = vqnet_metrics.auc_calculate(y_Qtensor, pred_Qtensor)
                print("auc:", result)
                # 0.625

                y = [1, 2, 1, 1, 1, 0, 0, 1, 1, 1]
                pred = [1, 0, 2, 1, 1, 1, 0, 1, 1, 0]
                y_Qtensor = tensor.QTensor(y)
                pred_Qtensor = tensor.QTensor(pred)
                result = vqnet_metrics.auc_calculate(y_Qtensor, pred_Qtensor, pos_label=2)
                print("auc:", result)
                # 0.1111111111111111


Triton兼容性
*********************************************************

`triton <https://triton-lang.org/main/index.html>`_ 是一种用于编写深度学习高效GPU内核的语言和编译器。
用户编写类似于NumPy的Python代码，然后Triton将其编译为高效的GPU代码（类似于CUDA但更高级）。
Triton依赖于一些PyTorch接口，VQNet实现了类似于PyTorch的接口，允许与使用Triton编写的代码集成，用于模型前向和反向传播。

安装Triton：

.. code-block::

    pip install triton

以下示例修改自Triton官方示例：`layer-norm <https://triton-lang.org/main/getting-started/tutorials/05-layer-norm.html>`_。
此示例需要在带有GPU的Linux上运行，并且需要安装triton和pytorch（用于比较计算正确性）：

.. code-block::

    import torch as real_torch
    import pyvqnet as torch
    import triton
    import triton.language as tl



    DEVICE = torch.DEV_GPU_0

    @triton.jit
    def _layer_norm_fwd_fused(
        X,  # pointer to the input
        Y,  # pointer to the output
        W,  # pointer to the weights
        B,  # pointer to the biases
        Mean,  # pointer to the mean
        Rstd,  # pointer to the 1/std
        stride,  # how much to increase the pointer when moving by 1 row
        N,  # number of columns in X
        eps,  # epsilon to avoid division by zero
        BLOCK_SIZE: tl.constexpr,
    ):
        # Map the program id to the row of X and Y it should compute.
        row = tl.program_id(0)
        Y += row * stride
        X += row * stride
        # Compute mean
        mean = 0
        _mean = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
        for off in range(0, N, BLOCK_SIZE):
            cols = off + tl.arange(0, BLOCK_SIZE)
            a = tl.load(X + cols, mask=cols < N, other=0.).to(tl.float32)
            _mean += a
        mean = tl.sum(_mean, axis=0) / N
        # Compute variance
        _var = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
        for off in range(0, N, BLOCK_SIZE):
            cols = off + tl.arange(0, BLOCK_SIZE)
            x = tl.load(X + cols, mask=cols < N, other=0.).to(tl.float32)
            x = tl.where(cols < N, x - mean, 0.)
            _var += x * x
        var = tl.sum(_var, axis=0) / N
        rstd = 1 / tl.sqrt(var + eps)
        # Write mean / rstd
        tl.store(Mean + row, mean)
        tl.store(Rstd + row, rstd)
        # Normalize and apply linear transformation
        for off in range(0, N, BLOCK_SIZE):
            cols = off + tl.arange(0, BLOCK_SIZE)
            mask = cols < N
            w = tl.load(W + cols, mask=mask)
            b = tl.load(B + cols, mask=mask)
            x = tl.load(X + cols, mask=mask, other=0.).to(tl.float32)
            x_hat = (x - mean) * rstd
            y = x_hat * w + b
            # Write output
            tl.store(Y + cols, y, mask=mask)


    @triton.jit
    def _layer_norm_bwd_dx_fused(DX,  # pointer to the input gradient
                                DY,  # pointer to the output gradient
                                DW,  # pointer to the partial sum of weights gradient
                                DB,  # pointer to the partial sum of biases gradient
                                X,  # pointer to the input
                                W,  # pointer to the weights
                                Mean,  # pointer to the mean
                                Rstd,  # pointer to the 1/std
                                Lock,  # pointer to the lock
                                stride,  # how much to increase the pointer when moving by 1 row
                                N,  # number of columns in X
                                GROUP_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr):
        # Map the program id to the elements of X, DX, and DY it should compute.
        row = tl.program_id(0)
        cols = tl.arange(0, BLOCK_SIZE_N)
        mask = cols < N
        X += row * stride
        DY += row * stride
        DX += row * stride
        # Offset locks and weights/biases gradient pointer for parallel reduction
        lock_id = row % GROUP_SIZE_M
        Lock += lock_id
        Count = Lock + GROUP_SIZE_M
        DW = DW + lock_id * N + cols
        DB = DB + lock_id * N + cols
        # Load data to SRAM

        x = tl.load(X + cols, mask=mask, other=0).to(tl.float32)
        dy = tl.load(DY + cols, mask=mask, other=0).to(tl.float32)
        w = tl.load(W + cols, mask=mask).to(tl.float32)
        mean = tl.load(Mean + row)
        rstd = tl.load(Rstd + row)
        # Compute dx
        xhat = (x - mean) * rstd
        wdy = w * dy
        xhat = tl.where(mask, xhat, 0.)
        wdy = tl.where(mask, wdy, 0.)
        c1 = tl.sum(xhat * wdy, axis=0) / N
        c2 = tl.sum(wdy, axis=0) / N
        dx = (wdy - (xhat * c1 + c2)) * rstd
        # Write dx
        tl.store(DX + cols, dx, mask=mask)

        # Accumulate partial sums for dw/db
        partial_dw = (dy * xhat).to(w.dtype)
        partial_db = (dy).to(w.dtype)
        while tl.atomic_cas(Lock, 0, 1) == 1:
            pass
        count = tl.load(Count)
        # First store doesn't accumulate
        if count == 0:
            tl.atomic_xchg(Count, 1)
        else:
            partial_dw += tl.load(DW, mask=mask)
            partial_db += tl.load(DB, mask=mask)
        tl.store(DW, partial_dw, mask=mask)
        tl.store(DB, partial_db, mask=mask)

        # need a barrier to ensure all threads finished before
        # releasing the lock
        tl.debug_barrier()

        # Release the lock
        tl.atomic_xchg(Lock, 0)


    @triton.jit
    def _layer_norm_bwd_dwdb(DW,  # pointer to the partial sum of weights gradient
                            DB,  # pointer to the partial sum of biases gradient
                            FINAL_DW,  # pointer to the weights gradient
                            FINAL_DB,  # pointer to the biases gradient
                            M,  # GROUP_SIZE_M
                            N,  # number of columns
                            BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr):
        # Map the program id to the elements of DW and DB it should compute.
        pid = tl.program_id(0)
        cols = pid * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
        dw = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
        db = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
        # Iterate through the rows of DW and DB to sum the partial sums.
        for i in range(0, M, BLOCK_SIZE_M):
            rows = i + tl.arange(0, BLOCK_SIZE_M)
            mask = (rows[:, None] < M) & (cols[None, :] < N)
            offs = rows[:, None] * N + cols[None, :]
            dw += tl.load(DW + offs, mask=mask, other=0.)
            db += tl.load(DB + offs, mask=mask, other=0.)
        # Write the final sum to the output.
        sum_dw = tl.sum(dw, axis=0)
        sum_db = tl.sum(db, axis=0)
        tl.store(FINAL_DW + cols, sum_dw, mask=cols < N)
        tl.store(FINAL_DB + cols, sum_db, mask=cols < N)


    class LayerNorm(torch.autograd.Function):

        @staticmethod
        def forward(ctx, x, normalized_shape, weight, bias, eps):
            # allocate output
            y = torch.empty_like(x)
            # reshape input data into 2D tensor
            x_arg = x.reshape([-1, x.shape[-1]])
            M, N = x_arg.shape
            mean = torch.empty((M, ), dtype=torch.float32, device=x.device).data
            rstd = torch.empty((M, ), dtype=torch.float32, device=x.device).data
            # Less than 64KB per feature: enqueue fused kernel
            MAX_FUSED_SIZE = 65536 // x.element_size()
            BLOCK_SIZE = min(MAX_FUSED_SIZE, triton.next_power_of_2(N))
            if N > BLOCK_SIZE:
                raise RuntimeError("This layer norm doesn't support feature dim >= 64KB.");
            # heuristics for number of warps
            num_warps = min(max(BLOCK_SIZE // 256, 1), 8)
            # enqueue kernel
            _layer_norm_fwd_fused[(M, )](  #
                x_arg, y, weight, bias, mean, rstd,  #
                x_arg.stride[0], N, eps,  #
                BLOCK_SIZE=BLOCK_SIZE, num_warps=num_warps, num_ctas=1)
            ctx.save_for_backward(x, weight, bias, mean, rstd)
            ctx.BLOCK_SIZE = BLOCK_SIZE
            ctx.num_warps = num_warps
            ctx.eps = eps
            return y.data

        @staticmethod
        def backward(ctx, dy):
            x, w, b, m, v = ctx.saved_tensors
            # heuristics for amount of parallel reduction stream for DW/DB
            N = w.shape[0]
            GROUP_SIZE_M = 64
            if N <= 8192: GROUP_SIZE_M = 96
            if N <= 4096: GROUP_SIZE_M = 128
            if N <= 1024: GROUP_SIZE_M = 256
            # allocate output
            locks = torch.zeros(2 * GROUP_SIZE_M, dtype=torch.int32, device=w.device).data
            _dw = torch.zeros((GROUP_SIZE_M, N), dtype=x.dtype, device=w.device).data
            _db = torch.zeros((GROUP_SIZE_M, N), dtype=x.dtype, device=w.device).data
            dw = torch.empty((N, ), dtype=w.dtype, device=w.device).data
            db = torch.empty((N, ), dtype=w.dtype, device=w.device).data
            dx = torch.empty_like(dy).data
            # enqueue kernel using forward pass heuristics
            # also compute partial sums for DW and DB
            x_arg = x.reshape([-1, x.shape[-1]])
            M, N = x_arg.shape

            _layer_norm_bwd_dx_fused[(M, )](  #
                dx, dy, _dw, _db, x, w, m, v, locks,  #
                x_arg.stride[0], N,  #
                BLOCK_SIZE_N=ctx.BLOCK_SIZE,  #
                GROUP_SIZE_M=GROUP_SIZE_M,  #
                num_warps=ctx.num_warps)

            grid = lambda meta: (triton.cdiv(N, meta['BLOCK_SIZE_N']), )

            # accumulate partial sums in separate kernel
            _layer_norm_bwd_dwdb[grid](
                _dw, _db, dw, db, min(GROUP_SIZE_M, M), N,  #
                BLOCK_SIZE_M=32,  #
                BLOCK_SIZE_N=128, num_ctas=1)
            return dx, None, dw, db, None

    preprocess = LayerNorm.preprocess
    layer_norm = LayerNorm.apply


    def test_layer_norm(M, N, dtype, eps=1e-5, device=DEVICE):
        # create data
        x_shape = (M, N)
        w_shape = (x_shape[-1], )
        weight = torch.rand(w_shape, dtype=dtype, device=device, requires_grad=True)
        bias = torch.rand(w_shape, dtype=dtype, device=device, requires_grad=True)
        x = -2.3 + 0.5 * torch.randn(x_shape, dtype=dtype, device=device)

        dy = .1 * torch.randn_like(x)
        x.requires_grad = True
        # forward pass
        xc, weightc, biasc = preprocess(x, weight, bias)
        y_tri = layer_norm(xc, w_shape, weightc, biasc, eps)
        y_tri = torch.to_tensor(y_tri)
        numpy_w = weight.detach().cpu().numpy()
        numpy_b = bias.detach().cpu().numpy()
        numpy_x = x.detach().cpu().numpy()
        numpy_dy = dy.detach().cpu().numpy()
        torch_weight = real_torch.tensor(numpy_w,device="cuda:1",requires_grad= True)
        torch_bias = real_torch.tensor(numpy_b,device="cuda:1",requires_grad= True)
        torch_x = real_torch.tensor(numpy_x,device="cuda:1",requires_grad= True)
        torch_dy = real_torch.tensor(numpy_dy,device="cuda:1",requires_grad= True)
        y_ref = real_torch.nn.functional.layer_norm(torch_x, w_shape, torch_weight, torch_bias, eps)
        # backward pass (triton)
        y_tri.backward(dy, retain_graph=True)
        dx_tri, dw_tri, db_tri = [_.grad.detach().cpu().numpy() for _ in [x, weight, bias]]
        x.grad, weight.grad, bias.grad = None, None, None
        # backward pass (torch)
        y_ref.backward(torch_dy, retain_graph=True)
        dx_ref, dw_ref, db_ref = [_.grad.detach().cpu().numpy() for _ in [torch_x, torch_weight, torch_bias]]
        # compare
        import numpy as np
        assert np.allclose(y_tri.detach().cpu().numpy(), y_ref.detach().cpu().numpy(), atol=1e-2, rtol=0)
        assert np.allclose(dx_tri, dx_ref, atol=1e-2, rtol=0)
        assert np.allclose(db_tri, db_ref, atol=1e-2, rtol=0)
        assert np.allclose(dw_tri, dw_ref, atol=1e-2, rtol=0)
        print("same")
    test_layer_norm(12, 32, torch.float32)
