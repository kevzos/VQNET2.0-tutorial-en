常见问题
=============================

**问：VQNet 有哪些特点？**

答：VQNet 是由本源量子基于 pyQPanda 开发的量子机器学习工具集。VQNet 为经典神经网络计算模块提供了丰富的易用接口，便于进行机器学习优化。
模型定义方式与主流机器学习框架一致，降低了用户的学习曲线。
同时，基于本源量子开发的高性能量子模拟器 pyQPanda，VQNet 还支持在个人笔记本电脑上运行大量量子比特。最后，VQNet 还提供了丰富的 :doc:`./qml_demo` 示例供您参考和学习。

**问：如何使用 VQNet 训练量子机器学习模型？**

答：有一类量子机器学习算法基于量子变分电路构建可微分的量子机器学习模型。
VQNet 可以使用梯度下降法来训练这类量子机器学习模型。一般步骤如下：首先，在本地计算机上，用户可以通过 pyQPanda 构建虚拟机，并结合 VQNet 提供的接口构建量子-经典混合模型 ``Module``；其次，调用 ``Module`` 的 ``forward()`` 可以根据用户定义的操作方式进行量子电路模拟和经典神经网络前向计算；
调用 ``Module`` 的 ``backward()`` 时，用户构建的模型可以像 PyTorch 等经典机器学习框架一样进行自动微分，计算量子变分电路和经典计算层中的参数梯度；最后，结合优化器的 ``step()`` 函数来优化参数。

在 VQNet 中，我们使用 `parameter-shift <https://arxiv.org/abs/1803.00745>`_ 来计算量子变分电路的梯度。用户可以使用 VQNet 提供的 :ref:`QuantumLayer_pq3` 接口来封装量子变分电路的自动微分，只需将量子变分电路定义为特定格式的参数即可构建上述类。

在 VQNet 中，我们还可以使用基于自动微分的方法来计算量子变分电路的梯度。用户可以使用 :ref:`vqc_api` 中的接口构建可训练电路。该电路不依赖 pyQPanda，而是将电路中的编码、门操作和测量拆分为可微分算子，从而实现参数梯度的计算功能。

详情请参阅本文档中的相关接口和示例代码。

**问：在 Windows 上安装 VQNet 时遇到错误："importError: DLL load failed while importing _core: The specified module could not be found."**

答：用户可能需要安装 Windows 上的 VC++ 运行时库。
请参考 https://learn.microsoft.com/en-us/cpp/windows/latest-supported-vc-redist?view=msvc-170 安装相应的运行时库。
此外，VQNet 目前仅支持 python3.10 版本，请确认您的 python 版本。

**问：如何调用本源量子云和量子芯片进行计算？**

答：您可以使用本源量子高性能计算集群或真实量子计算机进行量子电路模拟，替代本地的量子电路模拟。
在 VQNet 中，用户可以使用 ``QuantumBatchAsyncQcloudLayer`` 构建变分量子电路模块，输入在本源官网申请的 API KEYS，将任务提交到真机执行。

**问：为什么我定义的模型参数在训练过程中不更新？**

答：构建 VQNet 模型时，需要确保其中使用的所有模块都是可微的。当模型中的某个模块无法计算梯度时，该模块及其之前的模块将无法通过链式法则计算梯度。
如果用户自定义量子变分电路，请使用 VQNet 提供的 :ref:`QuantumLayer_pq3` 接口。对于经典机器学习模块，需要使用 :doc:`./QTensor` 和 :doc:`./nn` 定义的接口。这些接口封装了梯度计算功能，VQNet 可以执行自动微分。

如果用户想在 `Module` 中使用包含多个模块的列表作为子模块，请不要使用 Python 内置列表，而应使用 pyvqnet.nn.module.ModuleList。这样子模块的训练参数才能注册到整个模型，实现自动微分训练。示例如下：

     Example::

         from pyvqnet. tensor import *
         from pyvqnet.nn import Module,Linear,ModuleList
         from pyvqnet.qnn import ProbsMeasure, QuantumLayer
         import pyqpanda as pq
         def pqctest(input, param, qubits, cbits, m_machine):
             circuit = pq. QCircuit()
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
             prog. insert(circuit)

             rlt_prob = ProbsMeasure([0,2],prog,m_machine,qubits)
             return rlt_prob


         class M(Module):
             def __init__(self):
                 super(M, self).__init__()
                 #应使用 ModuleList 构建
                 self.pqc2 = ModuleList([QuantumLayer(pqctest,3,"cpu",4,1), Linear(4,1)
                 ])
                 #直接使用列表无法保存 pqc3 中的参数
                 #self.pqc3 = [QuantumLayer(pqctest,3,"cpu",4,1), Linear(4,1)
                 #]
             def forward(self, x, *args, **kwargs):
                 y = self.pqc2[0](x) + self.pqc2[1](x)
                 return y

         mm = M()
         print(mm. state_dict(). keys())

**问：为什么原有代码在 2.0.7 版本中无法运行？**

答：在 v2.0.7 版本中，我们为 QTensor 添加了不同的数据类型和 dtype 属性，并根据 PyTorch 规范限制了输入格式。例如，Embedding 层输入需要为 kint64 类型，CategoricalCrossEntropy、CrossEntropyLoss、SoftmaxCrossEntropy 和 NLL_Loss 层的标签需要为 kint64 类型。

您可以使用 'astype()' 接口将类型转换为指定的数据类型，或使用相应数据类型的 numpy 数组初始化 QTensor。

**问：VQNet 是否依赖 torch？**

答：VQNet 不依赖 torch，也不会自动安装 torch。

要使用以下功能，您需要自行安装 torch>=2.11.0。自 v2.15.0 起，我们支持使用 `torch >=2.11.0 <https://docs.pytorch.org/docs/stable/index.html>`_ 作为经典神经网络、量子变分电路、分布式计算等的计算后端。
调用 ``pyvqnet.backends.set_backend("torch")`` 后，接口保持不变，但 VQNet 的 ``QTensor`` 的 ``data`` 成员变量均使用 ``torch.Tensor`` 存储数据，
并使用 torch 进行计算。``pyvqnet.nn.torch`` 和 ``pyvqnet.qnn.vqc.torch`` 下的类继承自 ``torch.nn.Module``，可以组成 ``torch`` 模型。
