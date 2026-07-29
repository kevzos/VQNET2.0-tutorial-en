使用QPanda2的量子机器学习API
####################################################


.. warning::

    以下接口的量子计算部分使用pyQPanda2。

    由于pyQPanda2与pyqpanda3之间存在兼容性问题，您需要自行安装pyqpanda2：`pip install pyqpanda`

量子计算层
***********************************

.. _QuantumLayer:

QuantumLayer
=================================

QuantumLayer是支持变分量子电路的autograd模块的一个封装类。您可以定义一个函数作为参数，例如``qprog_with_measure``\ ，该函数需要包含由pyQPanda定义的量子电路：通常包含编码电路（coding-circuit）、演化电路（evolution-circuit）和测量操作（measurement-operation）。
该QuantumLayer类可以嵌入到混合量子经典机器学习模型中，并通过经典梯度下降法最小化混合量子经典模型的目标函数或损失函数。
您可以通过更改``QuantumLayer``\ 中的参数``diff_method``\ 来指定量子电路参数的梯度计算方法。``QuantumLayer``\ 目前支持两种方法，一种是``finite_diff``\ ，另一种是``parameter-shift``\ 方法。

``finite_diff``\ 方法是估计函数梯度最传统、最常用的数值方法之一。其主要思想是用差分代替偏导数：

.. math::

    f^{\prime}(x)=\lim _{h \rightarrow 0} \frac{f(x+h)-f(x)}{h}


对于``parameter-shift``\ 方法，我们使用目标函数，例如：

.. math:: O(\theta)=\left\langle 0\left|U^{\dagger}(\theta) H U(\theta)\right| 0\right\rangle

理论上，可以通过更精确的方法``parameter-shift``\ 计算量子电路中关于哈密顿量的参数梯度。

.. math::

    \nabla O(\theta)=
    \frac{1}{2}\left[O\left(\theta+\frac{\pi}{2}\right)-O\left(\theta-\frac{\pi}{2}\right)\right]

.. py:class:: pyvqnet.qnn.quantumlayer.QuantumLayer(qprog_with_measure,para_num,machine_type_or_cloud_token,num_of_qubits:int,num_of_cbits:int = 1,diff_method:str = "parameter_shift",delta:float = 0.01, dtype=None, name='')

    变分量子电路的抽象计算模块。它模拟一个参数化量子电路并获取测量结果。
    QuantumLayer继承自Module，因此它能够计算电路参数的梯度，训练变分量子电路模型，或将变分量子电路嵌入到混合量子经典模型中。
    
    该类不需要您在``qprog_with_measure``\ 函数中初始化虚拟机。

    :param qprog_with_measure: 可调用的量子电路函数，由pyQPanda2构建
    :param para_num: `int` - 参数数量
    :param machine_type_or_cloud_token: qpanda机器类型或pyQPanda2 QCLOUD令牌
    :param num_of_qubits: 量子比特数量
    :param num_of_cbits: 经典比特数量
    :param diff_method: 'parameter_shift' 或 'finite_diff'
    :param delta: 微分的增量
    :param dtype: 参数的数据类型，默认：None，使用默认数据类型kfloat32（32位浮点数）
    :param name: 输出层的名称

    :return: 一个可以计算量子电路的模块

    .. note::
        qprog_with_measure是一个在pyQPanda2中定义的量子电路函数。

        该函数应包含以下参数，否则无法在QuantumLayer中正常运行。

        qprog_with_measure (input,param,qubits,cbits,m_machine)

            `input`: array_like 输入的一维经典数据

            `param`: array_like 输入的一维量子电路参数

            `qubits`: 由QuantumLayer分配的量子比特

            `cbits`: 由QuantumLayer分配的经典比特。如果您的电路不使用cbits，也请保留此参数

            `m_machine`: 由QuantumLayer创建的模拟器

        使用QuantumLayer的``m_para``\ 属性获取变分量子电路的训练参数。该参数为``QTensor``\ 类，可以使用``to_numpy()``\ 接口转换为numpy数组。

    .. note::

        该类有别名：`QpandaQCircuitVQCLayer` 。

    示例::

        import pyqpanda as pq
        from pyvqnet.qnn.measure import ProbsMeasure
        from pyvqnet.qnn.quantumlayer import QuantumLayer
        import numpy as np
        from pyvqnet.tensor import QTensor
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
            # pauli_dict  = {'Z0 X1':10,'Y2':-0.543}
            rlt_prob = ProbsMeasure([0,2],prog,m_machine,qubits)
            return rlt_prob

        pqc = QuantumLayer(pqctest,3,"cpu",4,1)
        #classic data as input
        input = QTensor([[1,2,3,4],[40,22,2,3],[33,3,25,2.0]] )
        #forward circuits
        rlt = pqc(input)
        grad =  QTensor(np.ones(rlt.data.shape)*1000)
        #backward circuits
        rlt.backward(grad)
        print(rlt)
        # [
        # [0.2500000, 0.2500000, 0.2500000, 0.2500000],
        # [0.2500000, 0.2500000, 0.2500000, 0.2500000],
        # [0.2500000, 0.2500000, 0.2500000, 0.2500000]
        # ]

QuantumLayerV2
=================================

如果您更熟悉pyQPanda2的语法，可以使用QuantumLayerV2类。您可以通过``qubits``\ 、``cbits``\ 和``machine``\ 定义量子电路函数，然后将其作为参数``qprog_with_measure``\ 传递给QuantumLayerV2。

.. py:class:: pyvqnet.qnn.quantumlayer.QuantumLayerV2(qprog_with_measure, para_num, diff_method: str = 'parameter_shift', delta: float = 0.01, dtype=None, name='')

    变分量子电路的抽象计算模块。它模拟一个参数化量子电路并获取测量结果。
    QuantumLayer继承自Module，因此它能够计算电路参数的梯度，训练变分量子电路模型，或将变分量子电路嵌入到混合量子经典模型中。
    
    要使用此模块，您需要创建自己的量子虚拟机并分配量子比特和经典比特。

    :param qprog_with_measure: 可调用的量子电路函数，由pyQPanda2构建
    :param para_num: `int` - 参数数量
    :param diff_method: 'parameter_shift' 或 'finite_diff'
    :param delta: 微分的增量
    :param dtype: 参数的数据类型，默认：None，使用默认数据类型kfloat32（32位浮点数）
    :param name: 输出层的名称
    :return: 一个可以计算量子电路的模块

    .. note::
        qprog_with_measure是一个在pyQPanda中定义的量子电路函数。

        该函数应包含以下参数，否则无法在QuantumLayerV2中正常运行。

        与QuantumLayer相比，您需要自行分配量子比特和模拟器，

        如果qprog_with_measure需要量子测量，您可能还需要分配经典比特。

        qprog_with_measure (input,param)

        `input`: array_like 输入的一维经典数据

        `param`: array_like 输入的一维量子电路参数

    .. note::

        该类有别名：`QpandaQCircuitVQCLayerLite` 。

    示例::

        import pyqpanda as pq
        from pyvqnet.qnn.measure import ProbsMeasure
        from pyvqnet.qnn.quantumlayer import QuantumLayerV2
        import numpy as np
        from pyvqnet.tensor import QTensor
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


        pqc = QuantumLayerV2(pqctest,3)

        #classic data as input
        input = QTensor([[1,2,3,4],[4,2,2,3],[3,3,2,2.0]] )

        #forward circuits
        rlt = pqc(input)

        grad =  QTensor(np.ones(rlt.data.shape)*1000)
        #backward circuits
        rlt.backward(grad)
        print(rlt)

        # [
        # [0.2500000, 0.2500000, 0.2500000, 0.2500000],
        # [0.2500000, 0.2500000, 0.2500000, 0.2500000],
        # [0.2500000, 0.2500000, 0.2500000, 0.2500000]
        # ]

 


NoiseQuantumLayer
=================================

在真实的量子计算机中，由于量子比特的物理特性，计算误差总是不可避免的。为了更好地在量子虚拟机中模拟这种误差，VQNet也支持带有噪声的量子虚拟机。带有噪声的量子虚拟机模拟更接近真实的量子计算机。我们可以自定义支持的逻辑门类型以及逻辑门支持的噪声模型。
现有的支持的量子噪声模型在pyQPanda2的``NoiseQVM``\ 中定义。

我们可以使用``NoiseQuantumLayer``\ 来定义一个量子电路的自动微分类。``NoiseQuantumLayer``\ 支持带有噪声的pyQPanda2量子虚拟机。您可以定义一个函数作为参数``qprog_with_measure``\ ，该函数需要包含由pyQPanda定义的量子电路，同时您还需要传入一个参数``noise_set_config``\ ，通过pyQPanda接口设置噪声模型。

.. py:class:: pyvqnet.qnn.quantumlayer.NoiseQuantumLayer(qprog_with_measure, para_num, machine_type, num_of_qubits: int, num_of_cbits: int = 1, diff_method: str = 'parameter_shift', delta: float = 0.01, noise_set_config=None, dtype=None, name='')

    变分量子电路的抽象计算模块。它模拟一个参数化量子电路并获取测量结果。
    QuantumLayer继承自Module，因此它能够计算电路参数的梯度，训练变分量子电路模型，或将变分量子电路嵌入到混合量子经典模型中。
    
    该模块应通过``noise_set_config``\ 使用噪声模型进行初始化。

    :param qprog_with_measure: 可调用的量子电路函数，由pyQPanda2构建
    :param para_num: `int` - 参数数量
    :param machine_type: pyQPanda2机器类型
    :param num_of_qubits: 量子比特数量
    :param num_of_cbits: 经典比特数量
    :param diff_method: 'parameter_shift' 或 'finite_diff'
    :param delta: 微分的增量
    :param noise_set_config: 噪声设置函数
    :param dtype: 参数的数据类型，默认：None，使用默认数据类型kfloat32（32位浮点数）
    :param name: 输出层的名称
    
    :return: 一个可以计算带噪声模型的量子电路的模块

    .. note::
        qprog_with_measure是一个在pyQPanda中定义的量子电路函数。

        该函数应包含以下参数，否则无法在NoiseQuantumLayer中正常运行。

        qprog_with_measure (input,param,qubits,cbits,m_machine)

        `input`: array_like 输入的一维经典数据

        `param`: array_like 输入的一维量子电路参数

        `qubits`: 由NoiseQuantumLayer分配的量子比特

        `cbits`: 由NoiseQuantumLayer分配的经典比特。如果您的电路不使用cbits，也请保留此参数

        `m_machine`: 由NoiseQuantumLayer创建的模拟器

    示例::

        import pyqpanda as pq
        from pyvqnet.qnn.measure import ProbsMeasure
        from pyvqnet.qnn.quantumlayer import NoiseQuantumLayer
        import numpy as np
        from pyqpanda import *
        from pyvqnet.tensor import QTensor


        def circuit(weights, param, qubits, cbits, machine):

            circuit = pq.QCircuit()

            circuit.insert(pq.H(qubits[0]))
            circuit.insert(pq.RY(qubits[0], weights[0]))
            circuit.insert(pq.RY(qubits[0], param[0]))
            prog = pq.QProg()
            prog.insert(circuit)
            prog << measure_all(qubits, cbits)

            result = machine.run_with_configuration(prog, cbits, 100)

            counts = np.array(list(result.values()))
            states = np.array(list(result.keys())).astype(float)
            # Compute probabilities for each state
            probabilities = counts / 100
            # Get state expectation
            expectation = np.sum(states * probabilities)
            return expectation


        def default_noise_config(qvm, q):

            p = 0.01
            qvm.set_noise_model(NoiseModel.BITFLIP_KRAUS_OPERATOR,
                                GateType.PAULI_X_GATE, p)
            qvm.set_noise_model(NoiseModel.BITFLIP_KRAUS_OPERATOR,
                                GateType.PAULI_Y_GATE, p)
            qvm.set_noise_model(NoiseModel.BITFLIP_KRAUS_OPERATOR,
                                GateType.PAULI_Z_GATE, p)
            qvm.set_noise_model(NoiseModel.BITFLIP_KRAUS_OPERATOR, GateType.RX_GATE, p)
            qvm.set_noise_model(NoiseModel.BITFLIP_KRAUS_OPERATOR, GateType.RY_GATE, p)
            qvm.set_noise_model(NoiseModel.BITFLIP_KRAUS_OPERATOR, GateType.RZ_GATE, p)
            qvm.set_noise_model(NoiseModel.BITFLIP_KRAUS_OPERATOR, GateType.RY_GATE, p)
            qvm.set_noise_model(NoiseModel.BITFLIP_KRAUS_OPERATOR,
                                GateType.HADAMARD_GATE, p)
            qves = []
            for i in range(len(q) - 1):
                qves.append([q[i], q[i + 1]])  #
            qves.append([q[len(q) - 1], q[0]])
            qvm.set_noise_model(NoiseModel.DAMPING_KRAUS_OPERATOR, GateType.CNOT_GATE,
                                p, qves)

            return qvm


        qvc = NoiseQuantumLayer(circuit,
                                24,
                                "noise",
                                1,
                                1,
                                diff_method="parameter_shift",
                                delta=0.01,
                                noise_set_config=default_noise_config)
        input = QTensor([[0., 1., 1., 1.], [0., 0., 1., 1.], [1., 0., 1., 1.]])
        rlt = qvc(input)
        grad = QTensor(np.ones(rlt.data.shape) * 1000)

        rlt.backward(grad)
        print(qvc.m_para.grad)

        #[1195., 105., 70., 0.,
        # 45., -45., 50., 15.,
        # -80., 50., 10., -30.,
        # 10., 60., 75., -110.,
        # 55., 45., 25., 5.,
        # 5., 50., -25., -15.]

以下是一个``noise_set_config``\ 的示例，这里我们为量子门``RX``\ 、``RY``\ 、``RZ``\ 、``X``\ 、``Y``\ 、``Z``\ 、``H``\ 等添加了噪声参数p=0.01的噪声模型BITFLIP_KRAUS_OPERATOR。

.. code-block::

	def noise_set_config(qvm,q):

		p = 0.01
		qvm.set_noise_model(NoiseModel.BITFLIP_KRAUS_OPERATOR, GateType.PAULI_X_GATE, p)
		qvm.set_noise_model(NoiseModel.BITFLIP_KRAUS_OPERATOR, GateType.PAULI_Y_GATE, p)
		qvm.set_noise_model(NoiseModel.BITFLIP_KRAUS_OPERATOR, GateType.PAULI_Z_GATE, p)
		qvm.set_noise_model(NoiseModel.BITFLIP_KRAUS_OPERATOR, GateType.RX_GATE, p)
		qvm.set_noise_model(NoiseModel.BITFLIP_KRAUS_OPERATOR, GateType.RY_GATE, p)
		qvm.set_noise_model(NoiseModel.BITFLIP_KRAUS_OPERATOR, GateType.RZ_GATE, p)
		qvm.set_noise_model(NoiseModel.BITFLIP_KRAUS_OPERATOR, GateType.RY_GATE, p)
		qvm.set_noise_model(NoiseModel.BITFLIP_KRAUS_OPERATOR, GateType.HADAMARD_GATE, p)
		qves =[]
		for i in range(len(q)-1):
			qves.append([q[i],q[i+1]])#
		qves.append([q[len(q)-1],q[0]])
		qvm.set_noise_model(NoiseModel.DAMPING_KRAUS_OPERATOR, GateType.CNOT_GATE, p, qves)

		return qvm



QiskitLayer
=================================

.. py:class:: pyvqnet.qnn.QiskitLayer(qiskit_circuits,para_num)

    在VQNet中使用Qiskit电路实现前向和反向传播的封装层。QISKIT_VQC是一个定义Qiskit量子电路及其运行函数的类。
    以下示例演示了其工作原理。该层仅支持电路输入和权重作为参数。
    
    :param cirq_vqc: 定义Qiskit电路的定义、后端和执行函数的类
    :param para_num: `int` - 参数数量
    :return: 能够运行qiskit量子电路模型的类

    示例::


        """
        

        qiskit                        2.1.1
        qiskit-aer                    0.17.2
        opencv-python
        """
        import sys
        sys.path.insert(0,"../")
        import os
        import os.path
        import urllib
        import gzip
        import numpy as np
        import random
        import sys
        sys.path.insert(0,"../../")
        random.seed(42)
        np.random.seed(42)
        from pyvqnet.nn.module import Module
        from pyvqnet.optim import Adam
        import pyvqnet
        pyvqnet.utils.set_random_seed(42)
        from pyvqnet.nn.loss import MeanSquaredError
        from pyvqnet.qnn.utils import QiskitLayer

        import qiskit
        from qiskit.quantum_info import Statevector
        from qiskit import  QuantumRegister, ClassicalRegister

        from qiskit.quantum_info.operators import  Pauli
        max_parallel_threads = 24
        gpu = False
        method = "statevector"
        backend_options = {
            "method": method,
            "precision": "double",
            "max_parallel_threads": max_parallel_threads,
            "fusion_enable": True,
            "fusion_threshold": 14,
            "fusion_max_qubit": 5,
        }
        from qiskit_aer import StatevectorSimulator
        simulator = StatevectorSimulator()

        simulator.set_options(**backend_options)


        url_base = 'https://ossci-datasets.s3.amazonaws.com/mnist/'
        key_file = {
            'train_img':'train-images-idx3-ubyte.gz',
            'train_label':'train-labels-idx1-ubyte.gz',
            'test_img':'t10k-images-idx3-ubyte.gz',
            'test_label':'t10k-labels-idx1-ubyte.gz'
        }

        def _download(dataset_dir,file_name):
            file_path = dataset_dir + "/" + file_name

            if os.path.exists(file_path):
                with gzip.GzipFile(file_path) as f:
                    file_path_ungz = file_path[:-3].replace('\\', '/')
                    if not os.path.exists(file_path_ungz):
                        open(file_path_ungz,"wb").write(f.read())
                return

            print("Downloading " + file_name + " ... ")
            urllib.request.urlretrieve(url_base + file_name, file_path)
            if os.path.exists(file_path):
                    with gzip.GzipFile(file_path) as f:
                        file_path_ungz = file_path[:-3].replace('\\', '/')
                        file_path_ungz = file_path_ungz.replace('-idx', '.idx')
                        if not os.path.exists(file_path_ungz):
                            open(file_path_ungz,"wb").write(f.read())
            print("Done")

        def download_mnist(dataset_dir):
            for v in key_file.values():
                _download(dataset_dir,v)

        def dataloader(data,label,batch_size, shuffle = True)->np:
            if shuffle:
                for _ in range(len(data)//batch_size):
                    random_index = np.random.randint(0, len(data), (batch_size, 1))
                    yield data[random_index].reshape(batch_size,-1),label[random_index].reshape(batch_size,-1)
            else:
                for i in range(0,len(data)-batch_size+1,batch_size):
                    yield data[i:i+batch_size], label[i:i+batch_size]

        def get_accuracy(result,label):
            result,label = np.array(result.data), np.array(label.data)

            is_correct = (np.abs(result - label) < 0.5)
            is_correct = np.count_nonzero(is_correct)
            acc = is_correct

            return acc

        def load_mnist_4_4(dataset="training_data", digits=np.arange(10),
                        path="."):
            import os, struct
            from array import array as pyarray
            download_mnist(path)
            if dataset == "training_data":
                fname_image = os.path.join(path, 'train-images.idx3-ubyte').replace('\\', '/')
                fname_label = os.path.join(path, 'train-labels.idx1-ubyte').replace('\\', '/')
            elif dataset == "testing_data":
                fname_image = os.path.join(path, 't10k-images.idx3-ubyte').replace('\\', '/')
                fname_label = os.path.join(path, 't10k-labels.idx1-ubyte').replace('\\', '/')
            else:
                raise ValueError("dataset must be 'training_data' or 'testing_data'")

            flbl = open(fname_label, 'rb')
            magic_nr, size = struct.unpack(">II", flbl.read(8))

            lbl = pyarray("b", flbl.read())
            flbl.close()

            fimg = open(fname_image, 'rb')
            magic_nr, size, rows, cols = struct.unpack(">IIII", fimg.read(16))
            img = pyarray("B", fimg.read())
            fimg.close()

            ind = [k for k in range(size) if lbl[k] in digits]
            N = len(ind)
            images = np.zeros((N, rows, cols))
            images_new = []# = np.zeros((N, 4, 4))
            labels = np.zeros((N, 1), dtype=int)
            import cv2
            for i in range(len(ind)):
                tmp1 = np.array(img[ind[i] * rows * cols: (ind[i] + 1) * rows * cols]).reshape((rows, cols))
                tmp1 = tmp1[4:24,4:24]
                tmp = cv2.resize(tmp1,(4,4))

                if np.max(tmp) ==0:
                    continue
                images_new.append(tmp)
                if lbl[ind[i]] ==digits[1]:
                    labels[i] = 1
                else:
                    labels[i] = 0

            return np.array(images_new), labels


        class QISKIT_VQC:

            def __init__(self, n_qubits, backend, shots):
                # --- Circuit definition ---

                qc = ClassicalRegister(1)
                self.qc = qc
                self.n_qubits = n_qubits

                all_qubits = [i for i in range(n_qubits)]
                self.all_qubits= all_qubits

                self.backend = backend
                self.shots = shots

            def run(self,**kwargs):

                x  = kwargs['x']
                weights  = kwargs['w']

                weights = weights.astype(np.float64)
                x = x.astype(np.float64)

                sum_feature = np.power(np.sum([t**2 for t in x]),0.5)
                normalize_feat = x/sum_feature

                self._circuit = qiskit.QuantumCircuit(QuantumRegister(4))

                self.theta = weights.reshape([4,6])
                self._circuit.initialize(normalize_feat, [0,1,2,3])


                for i in range(self.n_qubits):
                    self._circuit.rz(self.theta[i,0], i)
                    self._circuit.ry(self.theta[i,1], i)
                    self._circuit.rz(self.theta[i,2], i)

                for d in range(3, 6):

                    for i in range(self.n_qubits-1):
                        self._circuit.cx(i, i + 1)
                    self._circuit.cx(self.n_qubits-1, 0)

                    for i in range(self.n_qubits):
                        self._circuit.ry(self.theta[i,d], i)

                statevec = Statevector(self._circuit)
                Expectation = np.real(statevec.expectation_value(Pauli('ZIII')))
                return Expectation

        #define qiskit circuits class
        circuit = QISKIT_VQC(4, simulator, 1000)

        class Model_qiskit(Module):
            def __init__(self):
                super(Model_qiskit, self).__init__()
                self.qvc = QiskitLayer(circuit,24)

            def forward(self, x):

                return self.qvc(x)*0.5 + 0.5

        def Run_qiskit():

            x_train, y_train = load_mnist_4_4("training_data",digits=[3,6])

            y_train = y_train.reshape(-1, 1)

            x_test, y_test = load_mnist_4_4("testing_data",digits=[3,6])

            x_train = x_train.astype(np.float32)
            x_test = x_test.astype(np.float32)
            y_train = y_train.astype(np.float32)
            y_test = y_test.astype(np.float32)
            x_train = x_train *np.pi / 255
            x_test = x_test *np.pi / 255
            x_train = x_train[:100]
            y_train = y_train[:100]

            x_test = x_test[:50]
            y_test = y_test[:50]

            model = Model_qiskit()

            optimizer = Adam(model.parameters(),lr =0.01)
            batch_size = 10
            epoch = 2

            loss = MeanSquaredError()
            print("start training..............")
            model.train()

            TL=[]

            TA=[]

            for i in range(epoch):
                count=0
                sum_loss = 0
                accuracy = 0
                t = 0
                model.train()
                for data,label in dataloader(x_train,y_train,batch_size,True):

                    optimizer.zero_grad()

                    result = model(data)

                    loss_b = loss(label,result)

                    loss_b.backward()
                    optimizer._step()
                    sum_loss += loss_b.item()
                    count+=batch_size
                    accuracy += get_accuracy(result,label)
                    t = t + 1

                    print(f"epoch:{i}, iter{t} #### loss:{sum_loss*batch_size/count} #####accuracy:{accuracy/count}")
                TL.append(sum_loss*batch_size/count)
                TA.append(accuracy/count)
            print(f"qiskit epoch {epoch}, accuracy {TA[-1]}")

        if __name__=="__main__":

            Run_qiskit()


CirqLayer
=================================

.. py:class:: pyvqnet.qnn.CirqLayer(cirq_vqc,para_num)

    一个Cirq电路封装层，用于在vqnet中实现前向和反向传播。CIRQ_VQC是一个需要用户定义Cirq量子电路及其`run`\ 函数的类。以下示例演示了其工作原理。
    该层仅支持电路输入和权重作为参数。

    :param cirq_vqc: 定义Cirq电路的定义、后端和运行函数的类
    :param para_num: `int` - 参数数量
    :return: 能够运行Cirq量子电路模型的类


    .. note::

        以下示例代码需要`cirq==1.5.0, numpy <2`\ 。

    示例::

        import numpy as np
        import random
        random.seed(42)
        np.random.seed(42)
        from pyvqnet.nn.module import Module
        import pyvqnet
        pyvqnet.utils.set_random_seed(42)
        from pyvqnet.optim import Adam

        from pyvqnet.nn.loss import MeanSquaredError
        from pyvqnet.qnn.utils import CirqLayer


        import cirq
        import sympy
        from pyvqnet.utils.utils import get_circuit_symbols


        def dataloader(data,label,batch_size, shuffle = True)->np:
            if shuffle:
                for _ in range(len(data)//batch_size):
                    random_index = np.random.randint(0, len(data), (batch_size, 1))
                    yield data[random_index].reshape(batch_size,-1),label[random_index].reshape(batch_size,-1)
            else:
                for i in range(0,len(data)-batch_size+1,batch_size):
                    yield data[i:i+batch_size].reshape(batch_size,-1), label[i:i+batch_size].reshape(batch_size,-1)

        def get_accuracy(result,label):
            result,label = np.array(result.data), np.array(label.data)

            is_correct = (np.abs(result - label) < 0.5)
            is_correct = np.count_nonzero(is_correct)
            acc = is_correct

            return acc

        def load_mnist_4_4(dataset="training_data", digits=np.arange(10), 
                        path=".",encoding = "raw" ):
            import os, struct
            from array import array as pyarray
            if dataset == "training_data":
                fname_image = os.path.join(path, 'train-images.idx3-ubyte').replace('\\', '/')
                fname_label = os.path.join(path, 'train-labels.idx1-ubyte').replace('\\', '/')
            elif dataset == "testing_data":
                fname_image = os.path.join(path, 't10k-images.idx3-ubyte').replace('\\', '/')
                fname_label = os.path.join(path, 't10k-labels.idx1-ubyte').replace('\\', '/')
            else:
                raise ValueError("dataset must be 'training_data' or 'testing_data'")

            flbl = open(fname_label, 'rb')
            magic_nr, size = struct.unpack(">II", flbl.read(8))

            lbl = pyarray("b", flbl.read())
            flbl.close()

            fimg = open(fname_image, 'rb')
            magic_nr, size, rows, cols = struct.unpack(">IIII", fimg.read(16))
            img = pyarray("B", fimg.read())
            fimg.close()

            ind = [k for k in range(size) if lbl[k] in digits]
            N = len(ind)

            images_new = []# = np.zeros((N, 4, 4))
            labels = np.zeros((N, 1), dtype=int)
            import cv2
            for i in range(len(ind)):
                tmp1 = np.array(img[ind[i] * rows * cols: (ind[i] + 1) * rows * cols]).reshape((rows, cols))
                tmp1 = tmp1[4:24,4:24]
                tmp = cv2.resize(tmp1,(4,4))

                if np.max(tmp) ==0:
                    continue
                if encoding == "normalized":
                    sum_feature = np.power(np.sum([t**2 for t in tmp.flatten()]),0.5)
                        
                    normalize_feat = tmp/sum_feature
                images_new.append(normalize_feat)
                if lbl[ind[i]] ==digits[1]:
                    labels[i] = 1
                else:
                    labels[i] = 0 

            return np.array(images_new), labels



        class CIRQ_VQC:

            def __init__(self,simulator = cirq.Simulator ()):

                self._circuit = cirq.Circuit()
                n_qubits =4
                ###define qubits
                q0 = cirq.NamedQubit ('q0')
                q1 = cirq.NamedQubit ('q1')
                q2 = cirq.NamedQubit ('q2')
                q3 = cirq.NamedQubit ('q3')
                qubits = [q0,q1,q2,q3]
                self.qubits = [q0,q1,q2,q3]
                ###define varational parameters
                param = sympy.symbols(f'theta(0:24)')
                self.theta = np.asarray(param).reshape((4,6))

                ###define circuits
                circuit = cirq.Circuit()

                for i ,q in enumerate(qubits):
                    circuit.append(cirq.rz(self.theta[i][0])(q))
                    circuit.append(cirq.ry(self.theta[i][1])(q))
                    circuit.append(cirq.rz(self.theta[i][2])(q))
                
                for d in range(3, 6):
                    for i in range(n_qubits-1):
                        circuit.append(cirq.CNOT(qubits[i], qubits[i + 1]))
                    circuit.append(cirq.CNOT(qubits[n_qubits-1], qubits[0]))

                    for i ,q in enumerate(qubits):
                        circuit.append(cirq.ry(self.theta[i][d])(q))

                self._circuit = circuit
                
                ###define backend
                self._backend = simulator

                self._param_symbols_list,self._input_symbols_list = get_circuit_symbols(self._circuit)


            def run(self,resolver,init_state):

                rlt = self._backend.simulate(self._circuit,resolver,initial_state=init_state).final_state_vector
                z0 = cirq.Z(self.qubits[0])

                qubit_map={self.qubits[0]: 0}
                
                expectation = z0.expectation_from_state_vector(rlt, qubit_map).real

                return expectation

        #define cirq circuits class
        circuit = CIRQ_VQC()

        class Model_cirq(Module):
            def __init__(self):
                super(Model_cirq, self).__init__()
                self.qvc = CirqLayer(circuit,24)

            def forward(self, x):

                y = self.qvc(x)*0.5 + 0.5
                return y.astype(x.dtype)


        def run_cirq():

            x_train, y_train = load_mnist_4_4("training_data",digits=[3,6],encoding="normalized")
            y_train = y_train.reshape(-1, 1) 

            x_test, y_test = load_mnist_4_4("testing_data",digits=[3,6],encoding="normalized")

            x_train = x_train.astype(np.float32)
            x_test = x_test.astype(np.float32)
            y_test = y_test.astype(np.float32)
            y_train = y_train.astype(np.float32)
            x_train = x_train[:100] 

            y_train = y_train[:100] 

            x_test = x_test[:50]

            y_test = y_test[:50]  

            model = Model_cirq()

            optimizer = Adam(model.parameters(),lr =0.01)
            batch_size = 10
            epoch = 5

            loss = MeanSquaredError()
            print("start training..............")
            model.train()

            TL=[]
            TA=[]

            for i in range(epoch):
                count=0
                sum_loss = 0
                accuracy = 0
                t = 0
                for data,label in dataloader(x_train,y_train,batch_size,False):

                    optimizer.zero_grad()
                    result = model(data)
                    loss_b = loss(label,result)

                    loss_b.backward()
                    optimizer._step()
                    sum_loss += loss_b.item()
                    count+=batch_size
                    accuracy += get_accuracy(result,label)
                    t = t + 1

                    print(f"epoch:{i},  #### loss:{sum_loss*batch_size/count} #####accuracy:{accuracy/count}")
                TL.append(sum_loss*batch_size/count)
                TA.append(accuracy/count)
            print(f"cirq epoch {epoch}, final accuracy {TA[-1]}")

        if __name__=="__main__":
        
            run_cirq()

量子门
***********************************

处理量子比特的方式称为量子门。通过使用量子门，我们可以有意识地演化量子态。量子门是量子算法的基础。

基本量子门
=================================

在VQNet中，我们使用本源量子开发的pyQPanda的每个逻辑门来构建量子电路并进行量子模拟。
目前pyQPanda支持的量子门可以在pyQPanda的量子门部分定义。
此外，VQNet还封装了一些量子机器学习中常用的量子门组合。



AmplitudeEmbeddingCircuit
=================================

.. py:function:: pyvqnet.qnn.template.AmplitudeEmbeddingCircuit(input_feat, qubits)

    将:math:`2^n`\ 个特征编码到:math:`n`\ 个量子比特的振幅向量中。
    为了表示有效的量子态向量，``features``\ 的L2范数必须为1。

    :param input_feat: 表示参数的numpy数组
    :param qubits: 由pyQPanda分配的量子比特
    :return: 量子电路

    示例::

        import numpy as np
        import pyqpanda as pq
        from pyvqnet.qnn.template import AmplitudeEmbeddingCircuit
        input_feat = np.array([2.2, 1, 4.5, 3.7])
        m_machine = pq.init_quantum_machine(pq.QMachineType.CPU)
        m_qlist = m_machine.qAlloc_many(2)
        m_clist = m_machine.cAlloc_many(2)
        m_prog = pq.QProg()
        cir = AmplitudeEmbeddingCircuit(input_feat,m_qlist)
        print(cir)
        pq.destroy_quantum_machine(m_machine)

        #                              ┌────────────┐     ┌────────────┐
        # q_0:  |0>─────────────── ─── ┤RY(0.853255)├ ─── ┤RY(1.376290)├
        #           ┌────────────┐ ┌─┐ └──────┬─────┘ ┌─┐ └──────┬─────┘
        # q_1:  |0>─┤RY(2.355174)├ ┤X├ ───────■────── ┤X├ ───────■──────
        #           └────────────┘ └─┘                └─┘




使用pyQPanda2的量子机器学习API
***************************************************************************

用于学习和加载随机分布的量子生成对抗网络
==================================================================================================

量子生成对抗网络（`QGAN <https://www.nature.com/articles/s41534-019-0223-2>`_）算法使用纯量子变分电路来制备具有特定随机分布的量子态，这可以减少生成特定量子态所需的逻辑门数量，并降低量子电路的复杂度。它采用了经典的GAN模型结构，包含两个子模型：生成器和判别器。生成器为量子电路生成特定分布，而判别器则区分生成器生成的样本与真实随机分布的训练数据样本。
以下是根据Christa Zoufal的论文`Quantum Generative Adversarial Networks for learning and loading random distributions <https://www.nature.com/articles/s41534-019-0223-2>`_，VQNet实现QGAN学习和加载随机分布的示例。

.. image:: ./images/qgan-arch.PNG
   :width: 600 px
   :align: center

|

为了实现VQNet中量子生成对抗网络``QGANAPI``\ 类的构建，使用量子生成器制备真实分布数据的初始状态。量子比特数量为3，量子生成器内部参数化电路模块的重复次数为1。同时，使用KL作为QGAN加载随机分布的度量指标。

.. code-block::

    import pickle
    import os
    import pyqpanda as pq
    from pyvqnet.qnn.qgan.qgan_utils import QGANAPI
    import numpy as np

    num_of_qubits = 3  # paper config
    rep = 1
    number_of_data = 10000
    # Load data samples from different distributions
    mu = 1
    sigma = 1
    real_data = np.random.lognormal(mean=mu, sigma=sigma, size=number_of_data)


    # intial
    save_dir = None
    qgan_model = QGANAPI(
        real_data,
        # numpy generated data distribution, 1 - dim.
        num_of_qubits,
        batch_size=2000,
        num_epochs=2000,
        q_g_cir=None,
        bounds = [0.0,2**num_of_qubits -1],
        reps=rep,
        metric="kl",
        tol_rel_ent=0.01,
        if_save_param_dir=save_dir
    )

以下是QGAN的``train``\ 模块。

.. code-block::

    # train
    qgan_model.train()  # train qgan


QGAN的``eval``\ 模块用于绘制QGAN制备的随机分布与真实分布之间的损失函数曲线和概率分布图。

.. code-block::

    # show probability distribution function of generated distribution and real distribution
    qgan_model.eval(real_data)  #draw pdf

QGAN的``get_trained_quantum_parameters``\ 模块用于获取训练参数并以numpy数组形式输出。如果``save_DIR``\ 不为空，则将训练参数保存到文件。QGAN的``Load_param_and_eval``\ 模块用于加载训练参数，``get_circuits_with_trained_param``\ 模块用于获取训练后由量子生成器生成的pyQPanda电路。

.. code-block::

    # get trained quantum parameters
    param = qgan_model.get_trained_quantum_parameters()
    print(f" trained param {param}")

    #load saved parameters files
    if save_dir is not None:
        path = os.path.join(
            save_dir, qgan_model._start_time + "trained_qgan_param.pickle")
        with open(path, "rb") as file:
            t3 = pickle.load(file)
        param = t3["quantum_parameters"]
        print(f" trained param {param}")

    #show probability distribution function of generated distribution and real distribution
    qgan_model.load_param_and_eval(param)

    #calculate metric
    print(qgan_model.eval_metric(param, "kl"))

    #get generator quantum circuit
    m_machine = pq.CPUQVM()
    m_machine.init_qvm()
    qubits = m_machine.qAlloc_many(num_of_qubits)
    qpanda_cir = qgan_model.get_circuits_with_trained_param(qubits)
    print(qpanda_cir)

通常，QGAN学习和加载随机分布需要多次使用不同的随机种子训练模型才能获得预期结果。例如，以下是通过QGAN实现的对数正态分布与真实对数正态分布之间的概率分布函数图，以及QGAN生成器和判别器之间的损失函数曲线。

.. image:: ./images/qgan-loss.png
   :width: 600 px
   :align: center

|

.. image:: ./images/qgan-pdf.png
   :width: 600 px
   :align: center

|


量子核SVM
=================================

在机器学习任务中，数据通常无法在原始空间中通过超平面进行分离。寻找此类超平面的常用技术是对数据应用非线性变换函数。
该函数称为特征映射，通过它我们可以计算机器学习分类任务中数据点在新特征空间中的接近程度。

本示例参考论文：`Supervised learning with quantum enhanced feature spaces <https://arxiv.org/pdf/1804.11326.pdf>`_ 。
第一种方法构建用于数据分类任务的变分电路。

``gen_vqc_qsvm_data``\ 是生成该示例所需数据的函数。``vqc_qsvm``\ 是一个用于对输入数据进行分类的变分子电路类。
``vqc_qsvm.plot()``\ 函数用于可视化数据分布。

.. image:: ./images/VQC-SVM.png
   :width: 600 px
   :align: center

|

    .. code-block::

        """
        VQC QSVM
        """
        from pyvqnet.qnn.svm import vqc_qsvm, gen_vqc_qsvm_data
        import matplotlib.pyplot as plt
        import numpy as np

        batch_size = 40
        maxiter = 40
        training_size = 20
        test_size = 10
        gap = 0.3
        #sub-circuits repeat times
        rep = 3

        #defines QSVM class
        VQC_QSVM = vqc_qsvm(batch_size, maxiter, rep)
        #randomly generates data from thesis.
        train_features, test_features, train_labels, test_labels, samples = \
            gen_vqc_qsvm_data(training_size=training_size, test_size=test_size, gap=gap)
        VQC_QSVM.plot(train_features, test_features, train_labels, test_labels, samples)
        #train
        VQC_QSVM.train(train_features, train_labels)
        #test
        rlt, acc_1 = VQC_QSVM.predict(test_features, test_labels)
        print(f"testing_accuracy {acc_1}")


除了上述直接使用变分量子电路将经典数据特征映射到量子特征空间的方法外，论文`Supervised learning with quantum enhanced feature spaces <https://arxiv.org/pdf/1804.11326.pdf>`_
还介绍了使用量子电路直接估计核函数并使用经典支持向量机进行分类的方法。
类比经典SVM中的各种核函数:math:`K(i,j)`\ ，使用量子核函数定义经典数据在量子特征空间:math:`\phi(\mathbf{x}_i)`\ 中的内积：

.. math::
    |\langle \phi(\mathbf{x}_j) | \phi(\mathbf{x}_i) \rangle |^2 =  |\langle 0 | U^\dagger(\mathbf{x}_j) U(\mathbf{x}_i) | 0 \rangle |^2

使用VQNet和pyQPanda，我们定义一个``QuantumKernel_VQNet``\ 来生成量子核函数，并使用``sklearn``\ 进行分类：

.. image:: ./images/qsvm-kernel.png
   :width: 600 px
   :align: center

|

.. code-block::

    import numpy as np
    import pyqpanda as pq
    from sklearn.svm import SVC
    from pyqpanda import *
    from pyqpanda.Visualization.circuit_draw import *
    from pyvqnet.qnn.svm import QuantumKernel_VQNet, gen_vqc_qsvm_data
    import matplotlib
    try:
        matplotlib.use('TkAgg')
    except:
        pass
    import matplotlib.pyplot as plt

    train_features, test_features,train_labels, test_labels, samples = gen_vqc_qsvm_data(20,5,0.3)
    quantum_kernel = QuantumKernel_VQNet(n_qbits=2)
    quantum_svc = SVC(kernel=quantum_kernel.evaluate)
    quantum_svc.fit(train_features, train_labels)
    score = quantum_svc.score(test_features, test_labels)
    print(f"quantum kernel classification test score: {score}")


同时扰动随机逼近优化器
=================================================================


.. py:function:: pyvqnet.qnn.SPSA(maxiter: int = 1000, save_steps: int = 1, last_avg: int = 1, c0: float = _C0, c1: float = 0.2, c2: float = 0.602, c3: float = 0.101, c4: float = 0, init_para=None, model=None, calibrate_flag=False)
    

    同时扰动随机逼近（SPSA）优化器。

    SPSA提供了一种用于逼近多变量可微代价函数梯度的随机方法。
    为了实现这一点，代价函数使用扰动的参数向量进行两次评估：原始参数向量的每个分量同时被一个随机生成的值扰动。
    更多信息请访问`SPSA网站 <http://www.jhuapl.edu/SPSA>`__。

    :param maxiter: 最大迭代次数。默认值：1000
    :param save_steps: 每隔save_steps步保存一次中间信息。默认值：1
    :param last_avg: 最后last_avg次迭代的平均参数。
        如果last_avg = 1，则仅考虑最后一次迭代。默认值：1
    :param c0: 初始a。更新参数的步长。默认值：0.2*pi
    :param c1: 初始c。用于近似梯度的步长。默认值：0.1
    :param c2: 论文中的alpha，用于在每次迭代中调整a(c0)。默认值：0.602
    :param c3: 论文中的gamma，用于在每次迭代中调整c(c1)。默认值：0.101
    :param c4: 也用于控制a的参数。默认值：0
    :param init_para: 初始化参数。默认值：None
    :param model: 参数化模型：model。默认值：None
    :param calibrate_flag: 是否校准超参数a和c，默认值：False

    :return: 一个SPSA优化器实例


    .. warning::

        SPSA仅支持一维参数。

    示例::

        import numpy as np
        import pyqpanda as pq

        import sys
        sys.path.insert(0, "../")
        import pyvqnet

        from pyvqnet.nn.module import Module
        from pyvqnet.qnn import SPSA
        from pyvqnet.tensor.tensor import QTensor
        from pyvqnet.qnn import AngleEmbeddingCircuit, expval, QuantumLayerV2, expval
        from pyvqnet.qnn.template import BasicEntanglerTemplate

        class Model_spsa(Module):
            def __init__(self):
                super(Model_spsa, self).__init__()
                self.qvc = QuantumLayerV2(layer_fn_spsa_pq, 3)

            def forward(self, x):
                y = self.qvc(x)
                return y


        def layer_fn_spsa_pq(input, weights):
            num_of_qubits = 1

            machine = pq.CPUQVM()
            machine.init_qvm()
            qubits = machine.qAlloc_many(num_of_qubits)
            c1 = AngleEmbeddingCircuit(input, qubits)
            weights =weights.reshape([4,1])
            bc_class = BasicEntanglerTemplate(weights, 1)
            c2 = bc_class.create_circuit(qubits)
            m_prog = pq.QProg()
            m_prog.insert(c1)
            m_prog.insert(c2)
            pauli_dict = {'Z0': 1}
            exp2 = expval(machine, m_prog, pauli_dict, qubits)

            return exp2

        model = Model_spsa()

        optimizer = SPSA(maxiter=20,
            init_para=model.parameters(),
            model=model,
        )


.. py:function:: pyvqnet.qnn.SPSA._step(input_data)

    使用SPSA优化输入数据。

    :param input_data: 输入数据
    :return:

        train_para: 最终参数

        theta_best: 最后`last_avg`\ 次迭代后的平均参数

    示例::

        import numpy as np
        import pyqpanda as pq

        import sys
        sys.path.insert(0, "../")
        import pyvqnet

        from pyvqnet.nn.module import Module
        from pyvqnet.qnn import SPSA
        from pyvqnet.tensor.tensor import QTensor
        from pyvqnet.qnn import AngleEmbeddingCircuit, expval, QuantumLayerV2, expval
        from pyvqnet.qnn.template import BasicEntanglerTemplate


        class Model_spsa(Module):
            def __init__(self):
                super(Model_spsa, self).__init__()
                self.qvc = QuantumLayerV2(layer_fn_spsa_pq, 3)

            def forward(self, x):
                y = self.qvc(x)
                return y


        def layer_fn_spsa_pq(input, weights):
            num_of_qubits = 1

            machine = pq.CPUQVM()
            machine.init_qvm()
            qubits = machine.qAlloc_many(num_of_qubits)
            c1 = AngleEmbeddingCircuit(input, qubits)
            weights =weights.reshape([4,1])
            bc_class = BasicEntanglerTemplate(weights, 1)
            c2 = bc_class.create_circuit(qubits)
            m_prog = pq.QProg()
            m_prog.insert(c1)
            m_prog.insert(c2)
            pauli_dict = {'Z0': 1}
            exp2 = expval(machine, m_prog, pauli_dict, qubits)

            return exp2

        model = Model_spsa()

        optimizer = SPSA(maxiter=20,
            init_para=model.parameters(),
            model=model,
        )

        data = QTensor(np.array([[0.27507603]]))
        p = model.parameters()
        p[0].data = pyvqnet._core.Tensor( np.array([3.97507603, 3.12950603, 1.00854038,
                        1.25907603]))

        optimizer._step(input_data=data)


        y = model(data)
        print(y)

量子Fisher信息计算矩阵
========================================================

.. py:class:: pyvqnet.qnn.opt.quantum_fisher(py_qpanda_config, params, target_gate_type_lists,target_gate_bits_lists, qcir_lists, wires)
    
    返回量子电路的量子Fisher信息矩阵。

    .. math::

        \mathrm{QFIM}_{i, j}=4 \operatorname{Re}\left[\left\langle\partial_i \psi(\boldsymbol{\theta}) \mid \partial_j \psi(\boldsymbol{\theta})\right\rangle-\left\langle\partial_i \psi(\boldsymbol{\theta}) \mid \psi(\boldsymbol{\theta})\right\rangle\left\langle\psi(\boldsymbol{\theta}) \mid \partial_j \psi(\boldsymbol{\theta})\right\rangle\right]

    简写为 :math::math:`\left|\partial_j \psi(\boldsymbol{\theta})\right\rangle:=\frac{\partial}{\partial \theta_j}|\psi(\boldsymbol{\theta})\rangle`\ 。

    .. note::

        目前仅支持RX、RY、RZ。

    :param params: 电路中的可变参数
    :param target_gate_type_lists: 支持"RX"、"RY"、"RZ"或列表
    :param target_gate_bits_lists: 参数化门作用的量子比特或比特列表
    :param qcir_lists: 目标参数化门之前的量子电路列表，用于计算度量张量，请参见以下示例
    :param wires: 量子电路的全局量子比特索引

    示例::
    
        import pyqpanda as pq

        from pyvqnet import *
        from pyvqnet.qnn.opt import pyqpanda_config_wrapper, insert_pauli_for_mt, quantum_fisher
        from pyvqnet.qnn import ProbsMeasure
        import numpy as np
        import pennylane as qml
        import pennylane.numpy as pnp

        n_wires = 4
        def layer_subcircuit_new(config: pyqpanda_config_wrapper, params):
            qcir = pq.QCircuit()
            qcir.insert(pq.RX(config._qubits[0], params[0]))
            qcir.insert(pq.RY(config._qubits[1], params[1]))
            
            qcir.insert(pq.CNOT(config._qubits[0], config._qubits[1]))
            
            qcir.insert(pq.RZ(config._qubits[2], params[2]))
            qcir.insert(pq.RZ(config._qubits[3], params[3]))
            return qcir


        def get_p1_diagonal_new(config, params, target_gate_type, target_gate_bits,
                            wires):
            qcir = layer_subcircuit_new(config, params)
            qcir2 = insert_pauli_for_mt(config._qubits, target_gate_type,
                                        target_gate_bits)
            qcir3 = pq.QCircuit()
            qcir3.insert(qcir)
            qcir3.insert(qcir2)
            
            m_prog = pq.QProg()
            m_prog.insert(qcir3)
            return ProbsMeasure(wires, m_prog, config._machine, config._qubits)

        config = pyqpanda_config_wrapper(n_wires)
        qcir = []
        
        qcir.append(get_p1_diagonal_new)

        params2 = QTensor([0.5, 0.5, 0.5, 0.25], requires_grad=True)

        mt = quantum_fisher(config, params2, [['RX', 'RY', 'RZ', 'RZ']],
                                [[0, 1, 2, 3]], qcir, [0, 1, 2, 3])

        # The above example shows that there are no identical gates in the same layer, 
        # but in the same layer you need to modify the logic gates according to the following example.
        
        n_wires = 3
        def layer_subcircuit_01(config: pyqpanda_config_wrapper, params):
            qcir = pq.QCircuit()
            qcir.insert(pq.RX(config._qubits[0], params[0]))
            qcir.insert(pq.RY(config._qubits[1], params[1]))
            qcir.insert(pq.CNOT(config._qubits[0], config._qubits[1]))
            
            return qcir

        def layer_subcircuit_02(config: pyqpanda_config_wrapper, params):
            qcir = pq.QCircuit()
            qcir.insert(pq.RX(config._qubits[0], params[0]))
            qcir.insert(pq.RY(config._qubits[1], params[1]))
            qcir.insert(pq.CNOT(config._qubits[0], config._qubits[1]))
            qcir.insert(pq.RZ(config._qubits[1], params[2]))
            return qcir

        def layer_subcircuit_03(config: pyqpanda_config_wrapper, params):
            qcir = pq.QCircuit()
            qcir.insert(pq.RX(config._qubits[0], params[0]))
            qcir.insert(pq.RY(config._qubits[1], params[1]))
            qcir.insert(pq.CNOT(config._qubits[0], config._qubits[1])) #  01 part
            
            qcir.insert(pq.RZ(config._qubits[1], params[2]))  #  02 part
            
            qcir.insert(pq.RZ(config._qubits[1], params[3]))
            return qcir

        def get_p1_diagonal_01(config, params, target_gate_type, target_gate_bits,
                            wires):
            qcir = layer_subcircuit_01(config, params)
            qcir2 = insert_pauli_for_mt(config._qubits, target_gate_type,
                                        target_gate_bits)
            qcir3 = pq.QCircuit()
            qcir3.insert(qcir)
            qcir3.insert(qcir2)
            
            m_prog = pq.QProg()
            m_prog.insert(qcir3)
            return ProbsMeasure(wires, m_prog, config._machine, config._qubits)
        
        def get_p1_diagonal_02(config, params, target_gate_type, target_gate_bits,
                            wires):
            qcir = layer_subcircuit_02(config, params)
            qcir2 = insert_pauli_for_mt(config._qubits, target_gate_type,
                                        target_gate_bits)
            qcir3 = pq.QCircuit()
            qcir3.insert(qcir)
            qcir3.insert(qcir2)
            
            m_prog = pq.QProg()
            m_prog.insert(qcir3)
            return ProbsMeasure(wires, m_prog, config._machine, config._qubits)
        
        def get_p1_diagonal_03(config, params, target_gate_type, target_gate_bits,
                            wires):
            qcir = layer_subcircuit_03(config, params)
            qcir2 = insert_pauli_for_mt(config._qubits, target_gate_type,
                                        target_gate_bits)
            qcir3 = pq.QCircuit()
            qcir3.insert(qcir)
            qcir3.insert(qcir2)
            
            m_prog = pq.QProg()
            m_prog.insert(qcir3)
            return ProbsMeasure(wires, m_prog, config._machine, config._qubits)
        
        config = pyqpanda_config_wrapper(n_wires)
        qcir = []
        
        qcir.append(get_p1_diagonal_01)
        qcir.append(get_p1_diagonal_02)
        qcir.append(get_p1_diagonal_03)
        
        params2 = QTensor([0.5, 0.5, 0.5, 0.25], requires_grad=True)

        mt = quantum_fisher(config, params2, [['RX', 'RY'], ['RZ'], ['RZ']], # rx,ry counts as layer one, first rz as layer two, second rz as layer three.
                                [[0, 1], [1], [1]], qcir, [0, 1])
