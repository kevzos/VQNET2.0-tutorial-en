VQNet 安装步骤
==================================

VQNet Python 包安装
----------------------------------

我们在 Linux、Windows、macOS 13+ (arm64) 上提供预编译的 Python 包安装方式，支持 **Python 3.10**\ 。

从官网下载相应的压缩包，解压后进入解压目录，运行以下步骤：

.. code-block::

    # 对于 Windows
    ./install.bat
    # 对于 macOS 和 Linux
    ./install.sh
 

对于 Windows 和 Linux 系统，pyvqnet 包包含基于 Nvidia CUDA 的经典神经网络计算内置加速功能，这依赖于特定版本的 NVIDIA CUDA 12.6 运行时库（随包自动安装）。
该包针对以下 CUDA 架构进行了优化：
**sm_80**\ （NVIDIA A100、A30 系列数据中心 GPU）和 **sm_86**\ （NVIDIA GeForce RTX 30 系列消费级 GPU）。请确保您使用的 GPU 支持这些架构，否则程序可能无法正常运行。

    .. important::

        请注意，由于此包不区分 CPU/GPU 版本，在 Windows 和 Linux 下会依赖 NVIDIA CUDA 运行时库（随包自动安装）。这可能会与依赖不同版本 CUDA 的其他软件产生冲突。


验证 VQNet 的安装
----------------------------------

.. code-block::

    import pyvqnet
    from pyvqnet.tensor import *
    a = arange(1,25).reshape([2, 3, 4])
    print(a)

测试 VQNet 的 GPU 功能
----------------------------------

.. code-block::

    from pyvqnet import DEV_GPU_0
    from pyvqnet.tensor import *
    a = ones([4,5],device = DEV_GPU_0)
    print(a)

VQNet 简单示例
--------------------------
这里我们介绍一个包含 VQNet 经典神经网络模块和量子模块的案例，以描述量子机器学习的工作流程。
它参考了 `Data re-uploading for a universal quantum classifier <https://arxiv.org/abs/1907.02085>`_ 。
通常，量子机器学习中的量子计算模块包含以下部分：

(1)编码器（Encoder）：将经典数据编码为量子态；

(2)拟设（Ansatz）：训练参数化量子门中的参数；

(3)测量（Measurement）：测量量子比特的值（量子比特量子态在指定轴上的投影）。

量子计算模块是量子经典神经网络混合模型的理论基础，与经典神经网络模块一样是可微分的。VQNet 支持量子计算模块和经典计算模块组成混合机器学习模型，并提供各种优化算法进行参数优化。（例如卷积层、池化层、全连接层、激活函数等）

.. figure:: ./images/classic-quantum.PNG

在量子计算模块中，VQNet 支持使用高效的量子软件计算包 pyqpanda3 来构建量子模块。
使用 pyqpanda3 提供的各种常用接口，用户可以快速构建量子计算模块。

下面的示例使用 pyqpanda3 构建了一个量子计算模块。通过 VQNet，该量子模块可以直接嵌入到混合机器学习模型中进行量子电路参数训练。
在此示例中，使用 1 个量子比特，多个参数化旋转门 `RZ`\ 、`RY`\ 、`RZ` 用于编码输入 x，并使用 `probs_measure` 函数观测量子比特的概率测量结果作为输出。

.. code-block::

    import pyqpanda3.core as pq
    from pyvqnet.qnn.pq3 import probs_measure
    def qdrl_circuit(input,weights):
        qlist = range(1)
        machine = pq.CPUQVM()
        x1 = input.squeeze()
        param1 = weights.squeeze()
        # 使用 pyqpanda3 接口构建量子电路实例
        circuit = pq.QCircuit()
        # 在第一个量子比特上插入带参数 x1[0] 的 RZ 门
        circuit << pq.RZ(qlist[0], x1[0])
        # 在第一个量子比特上插入带参数 x1[1] 的 RY 门
        circuit << pq.RY(qlist[0], x1[1])
        # 在第一个量子比特上插入带参数 x1[2] 的 RZ 门
        circuit << pq.RZ(qlist[0], x1[2])
        # 在第一个量子比特上插入带参数 param1[0] 的 RZ 门
        circuit << pq.RZ(qlist[0], param1[0])
        # 在第一个量子比特上插入带参数 param1[1] 的 RY 门
        circuit << pq.RY(qlist[0], param1[1])
        # 在第一个量子比特上插入带参数 param1[2] 的 RZ 门
        circuit << pq.RZ(qlist[0], param1[2])
        # 在第一个量子比特上插入带参数 x1[0] 的 RZ 门
        circuit << pq.RZ(qlist[0], x1[0])
        # 在第一个量子比特上插入带参数 x1[1] 的 RY 门
        circuit << pq.RY(qlist[0], x1[1])
        # 在第一个量子比特上插入带参数 x1[2] 的 RZ 门
        circuit << pq.RZ(qlist[0], x1[2])
        # 在第一个量子比特上插入带参数 param1[3] 的 RZ 门
        circuit << pq.RZ(qlist[0], param1[3])
        # 在第一个量子比特上插入带参数 param1[4] 的 RY 门
        circuit << pq.RY(qlist[0], param1[4])
        # 在第一个量子比特上插入带参数 param1[5] 的 RZ 门
        circuit << pq.RZ(qlist[0], param1[5])
        # 在第一个量子比特上插入带参数 x1[0] 的 RZ 门
        circuit << pq.RZ(qlist[0], x1[0])
        # 在第一个量子比特上插入带参数 x1[1] 的 RY 门
        circuit << pq.RY(qlist[0], x1[1])
        # 在第一个量子比特上插入带参数 x1[2] 的 RZ 门
        circuit << pq.RZ(qlist[0], x1[2])
        # 在第一个量子比特上插入带参数 param1[6] 的 RZ 门
        circuit << pq.RZ(qlist[0], param1[6])
        # 在第一个量子比特上插入带参数 param1[7] 的 RY 门
        circuit << pq.RY(qlist[0], param1[7])
        # 在第一个量子比特上插入带参数 param1[8] 的 RZ 门
        circuit << pq.RZ(qlist[0], param1[8])
        # 构建量子程序
        prog = pq.QProg()
        prog << circuit
        # 获取概率测量结果
        prob = probs_measure(machine ,prog,  qlist)

        return prob

我们的任务是使用二分类方法对随机生成的数据进行分类。在此任务中，
圆心在原点，半径 1 以内的点（红色）属于一个类别，蓝色样本属于另一个类别。

.. figure:: ./images/origin_circle.png

训练流程

.. code-block::

    # 导入所需库和函数
    from pyvqnet.qnn.pq3.quantumlayer import QuantumLayer
    from pyvqnet.optim import adam
    from pyvqnet.nn.loss import CategoricalCrossEntropy
    from pyvqnet.tensor import QTensor
    import numpy as np
    from pyvqnet.nn.module import Module


定义模型：``__init__`` 函数定义内部神经网络模块和量子模块，``forward`` 函数定义前向计算。``QuantumLayer`` 是一个抽象类，
封装了量子计算。
VQNet 将自动为 `qdrl_circuit` 计算参数梯度，参数数量为 `param_num`\ 。


.. code-block::

    # 待训练的参数数量
    param_num = 9
    # 量子比特数量
    qbit_num  = 1
    # 定义一个继承自 Module 的模型类
    class Model(Module):
        def __init__(self):
            super(Model, self).__init__()
            # 使用 QuantumLayer 将量子电路嵌入自动微分流程
            self.pqc = QuantumLayer(qdrl_circuit,param_num)
        # 定义前向函数
        def forward(self, x):
            x = self.pqc(x)
            return x

定义训练模型所需的函数

.. code-block::

    # 用于随机生成原始数据的函数
    def circle(samples:int,  rads =  np.sqrt(2/np.pi)) :
        data_x, data_y = [], []
        for i in range(samples):
            x = 2*np.random.rand(2) - 1
            y = [0,1]
            if np.linalg.norm(x) < rads:
                y = [1,0]
            data_x.append(x)
            data_y.append(y)
        return np.array(data_x,dtype=np.float32), np.array(data_y,np.int64)

    # 用于加载数据的函数
    def get_minibatch_data(x_data, label, batch_size):
        for i in range(0,x_data.shape[0]-batch_size+1,batch_size):
            idxs = slice(i, i + batch_size)
            yield x_data[idxs], label[idxs]

    # 用于计算准确率的函数
    def get_score(pred, label):
        pred, label = np.array(pred.data), np.array(label.data)
        pred = np.argmax(pred,axis=1)
        score = np.argmax(label,1)
        score = np.sum(pred == score)
        return score

VQNet 遵循通用的机器学习工作流程：迭代加载数据、前向传播、计算损失函数、反向传播和更新参数。

.. code-block::

    # 实例化模型
    model = Model()
    # 使用 Adam 定义优化器
    optimizer = adam.Adam(model.parameters(),lr =0.6)
    # 使用交叉熵定义损失函数
    Closs = CategoricalCrossEntropy()

训练模型的函数

.. code-block::

    def train():
            
        # 随机生成待训练的数据
        x_train, y_train = circle(500)
        x_train = np.hstack((x_train, np.zeros((x_train.shape[0], 1),dtype=np.float32)))  
        # 定义每批数据的大小
        batch_size = 32
        # 最大训练迭代次数
        epoch = 10
        print("start training...........")
        for i in range(epoch):
            model.train()
            accuracy = 0
            count = 0
            loss = 0
            for data, label in get_minibatch_data(x_train, y_train,batch_size):
                # 清除优化器的梯度
                optimizer.zero_grad()
                # 前向计算
                output = model(data)
                # 计算损失函数
                losss = Closs(label, output)
                # 反向传播
                losss.backward()
                # 更新优化器参数
                optimizer._step()
                # 计算准确率
                accuracy += get_score(output,label)

                loss += losss.item()
                count += batch_size
                
            print(f"epoch:{i}, train_accuracy:{accuracy/count}")
            print(f"epoch:{i}, train_loss:{loss/count}\n")
            
验证模型的函数

.. code-block::

    def test():
        
        batch_size = 1
        model.eval()
        print("start eval...................")
        xtest, y_test = circle(500)
        test_accuracy = 0
        count = 0
        x_test = np.hstack((xtest, np.zeros((xtest.shape[0], 1),dtype=np.float32)))

        for test_data, test_label in get_minibatch_data(x_test,y_test, batch_size):

            test_data, test_label = QTensor(test_data),QTensor(test_label)
            output = model(test_data)
            test_accuracy += get_score(output, test_label)
            count += batch_size

        print(f"test_accuracy:{test_accuracy/count}")

Training and testing results

.. code-block::

    start training...........
    epoch:0, train_accuracy:0.6145833333333334
    epoch:0, train_loss:0.020432369535168013

    epoch:1, train_accuracy:0.6854166666666667
    epoch:1, train_loss:0.01872217481335004

    epoch:2, train_accuracy:0.8104166666666667
    epoch:2, train_loss:0.016634768371780715

    epoch:3, train_accuracy:0.7479166666666667
    epoch:3, train_loss:0.016975031544764835

    epoch:4, train_accuracy:0.7875
    epoch:4, train_loss:0.016502128106852372

    epoch:5, train_accuracy:0.8083333333333333
    epoch:5, train_loss:0.0163204787299037

    epoch:6, train_accuracy:0.8083333333333333
    epoch:6, train_loss:0.01634311651190122

    epoch:7, train_loss:0.016330583145221074

    epoch:8, train_accuracy:0.8125
    epoch:8, train_loss:0.01629052646458149

    epoch:9, train_accuracy:0.8083333333333333
    epoch:9, train_loss:0.016270687493185203

    start eval...................
    test_accuracy:0.826

.. figure:: ./images/qdrl_for_simple.png
