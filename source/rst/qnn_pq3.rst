使用 pyQPanda3 量子机器学习模块
#########################################################

.. warning::

    以下接口的量子计算部分使用了 pyqpanda3。

    如果您在此模块下使用 QCloud 功能，则在代码中导入 pyqpanda2 或使用 pyvqnet 的 pyqpanda2 相关包接口时会出现错误。

量子计算层
***********************************

.. _QuantumLayer_pq3:

QuantumLayer
============================

如果您熟悉 pyQPanda3 语法，可以使用 QuantumLayer 接口自定义 pyqpanda3 模拟器进行计算。

.. py:class:: pyvqnet.qnn.pq3.quantumlayer.QuantumLayer(qprog_with_measure,para_num,diff_method:str = "parameter_shift",delta:float = 0.01,dtype=None,name="")

    变分量子层的抽象计算模块。使用 pyQPanda3 模拟参数化量子电路并获取测量结果。该变分量
    子层继承了 VQNet 框架的梯度计算模块，可使用参数偏移法计算电路参数的梯度，训练变分量
    子电路模型，或将变分量子电路嵌入到混合量子-经典模型中。

    :param qprog_with_measure: 使用 pyQPanda 构建的量子电路运算与测量函数。
    :param para_num: `int` - 参数数量。
    :param diff_method: 量子电路参数梯度的求解方法，"parameter shift"（参数偏移）或 "finite difference"（有限差分），默认为参数偏移。
    :param delta: 通过有限差分计算梯度时的 \delta 值。
    :param dtype: 参数的数据类型，默认值：None，使用默认数据类型：kfloat32，表示 32 位浮点数。
    :param name: 该模块的名称，默认为 ""。

    :return: 一个能够计算量子电路的模块。

    .. note::

        qprog_with_measure 是在 pyQPanda3 中定义的量子电路函数。

        该函数必须包含两个参数 input 和 parameter 作为函数输入（即使某个参数实际未使用），输出为电路的测量结果或期望值（需要是 np.ndarray 或包含值的列表），否则在 QpandaQCircuitVQCLayerLite 中将无法正常运行。

        量子电路函数 qprog_with_measure (input, param) 的使用方法可参考下方示例。

        `input`：输入一维经典数据。如果没有输入，则传入 None

        `param`：输入待训练的一维变分量子电路参数

    .. note::

        该类有别名 `QuantumLayerV2`、`QpandaQCircuitVQCLayerLite`。

    Example::

        from pyvqnet.qnn.pq3.measure import ProbsMeasure
        from pyvqnet.qnn.pq3.quantumlayer import QuantumLayer
        from pyvqnet.tensor import QTensor,ones
        import pyqpanda3.core as pq
        def pqctest (input,param):
            num_of_qubits = 4

            m_machine = pq.CPUQVM()

            qubits = range(num_of_qubits)

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
        pqc = QuantumLayer(pqctest,3)

        #classic data as input
        input = QTensor([[1,2,3,4],[4,2,2,3],[3.0,3,2,2]] )

        #forward circuits
        rlt = pqc(input)

        print(rlt)

        grad = ones(rlt.data.shape)*1000
        #backward circuits
        rlt.backward(grad)

        print(pqc.m_para.grad)

QpandaQProgVQCLayer
=============================

.. py:class:: pyvqnet.qnn.pq3.quantumlayer.QuantumLayerV3(origin_qprog_func,para_num,qvm_type="cpu", pauli_str_dict=None, shots=1000, initializer=None,dtype=None,name="")

    将参数化量子电路提交至本地 QPanda3 全振幅模拟器进行计算，并训练电路中的参数。
    支持批量数据，并使用参数偏移规则来估计参数的梯度。
    对于 CRX、CRY、CRZ，该层使用 https://iopscience.iop.org/article/10.1088/1367-2630/ac2cb3 中的公式，其余逻辑门使用默认的参数偏移法计算梯度。

    :param origin_qprog_func: 使用 QPanda 构建的可调用量子电路函数。
    :param para_num: `int` - 参数数量；参数为一维。
    :param qvm_type: `str` - 使用的 pyqpanda3 模拟器类型，`cpu` 或 `gpu`，默认为 `cpu`。
    :param pauli_str_dict: `dict|list` - 表示量子电路中泡利算符的字典或字典列表。默认值为 None。
    :param shots: `int` - 测量次数。默认值为 1000。
    :param initializer: 参数值的初始化器。默认值为 None。
    :param dtype: 参数的数据类型。默认值为 None，表示使用默认数据类型。
    :param name: 模块的名称。默认为空字符串。

    :return: 返回一个 QpandaQProgVQCLayer 类

    .. note::

        origin_qprog_func 是用户使用 pyQPanda3 定义的量子电路函数。

        该函数必须包含两个参数 input 和 parameter 作为函数输入（即使某个参数实际未使用），输出为 pyqpanda3.core.QProg 类型数据，否则在 QuantumLayerV3 中将无法正常运行。

        origin_qprog_func (input,param )

        `input`：用户自定义数组类输入的一维经典数据

        `param`：array_like 输入用户自定义的一维量子电路参数

    .. note::

        该类有别名 `QuantumLayerV3`。

    Example::

        import pyqpanda3.core as pq
        import pyvqnet
        from pyvqnet.qnn.pq3.quantumlayer import  QuantumLayerV3


        def qfun(input, param ):
            m_qlist = range(3)
            cbits = range(3)
            measure_qubits = [0,1, 2]
            m_prog = pq.QProg()
            cir = pq.QCircuit(3)

            cir<<pq.RZ(m_qlist[0], input[0])
            cir<<pq.RX(m_qlist[2], input[2])
            
            qcir = pq.RX(m_qlist[1], param[1]).control(m_qlist[0])
        
            cir<<qcir

            qcir = pq.RY(m_qlist[0], param[2]).control(m_qlist[1])
        
            cir<<qcir

            cir<<pq.RY(m_qlist[0], input[1])

            qcir = pq.RZ(m_qlist[0], param[3]).control(m_qlist[1])
        
            cir<<qcir
            m_prog<<cir

            for idx, ele in enumerate(measure_qubits):
                m_prog << pq.measure(m_qlist[ele], cbits[idx])  # pylint: disable=expression-not-assigned
            return m_prog

        from pyvqnet.utils.initializer import ones
        l = QuantumLayerV3(qfun,
                        4,
                        "cpu",
                        pauli_str_dict=None,
                        shots=1000,
                        initializer=ones,
                        name="")
        x = pyvqnet.tensor.QTensor(
            [[2.56, 1.2,-3]],
            requires_grad=True)
        y = l(x)

        y.backward()
        print(l.m_para.grad.to_numpy())
        print(x.grad.to_numpy())

QuantumBatchAsyncQcloudLayer
==================================

当您安装最新版本的 pyqpanda3 时，可以使用此接口定义变分电路并提交至 originqc 真实芯片运行。

.. py:class:: pyvqnet.qnn.pq3.quantumlayer.QuantumBatchAsyncQcloudLayer(origin_qprog_func, qcloud_token, para_num, pauli_str_dict=None, shots = 1000, initializer=None, dtype=None, name="", diff_method="parameter_shift", submit_kwargs={}, query_kwargs={})
    
    一个使用 pyqpanda3 QCLOUD 在 originqc 真实芯片上进行计算的抽象计算模块。它将参数化量子电路提交至真实芯片并获取测量结果。
    如果 diff_method == "random_coordinate_descent"，该层将随机选择单个参数计算梯度，其他参数保持为零。参考：https://arxiv.org/abs/2311.00088

    .. note::

        qcloud_token 是您从云平台申请的 API 令牌。

        origin_qprog_func 需要返回 pyqpanda3.core.QProg 类型的数据。如果未设置 pauli_str_dict，则需要确保已在 QProg 中插入测量操作。

        origin_qprog_func 必须采用以下格式：

        origin_qprog_func(input,param)

            `input`：输入 1~2 维经典数据。若为 2 维，第一维为批大小

            `param`：输入待训练的 1 维变分量子电路参数

    .. note::

        在当前版本中，单次电路提交至 QCloud 的默认总超时时间为 60 秒。如果因 QCloud 繁忙导致超时，可以在 ``query_kwargs`` 中设置 ``total_timeout`` 键的值为所需的等待秒数。

    :param origin_qprog_func: 使用 QPanda 构建的变分量子电路函数，必须返回 QProg。
    :param qcloud_token: `str` - 量子计算机类型或用于执行的云令牌。
    :param para_num: `int` - 参数数量，参数为大小为 [para_num] 的 QTensor。
    :param pauli_str_dict: `dict|list` - 表示量子电路中泡利算符的字典或字典列表。默认为 "None"，表示执行测量操作。如果输入泡利算符字典，将计算单个期望值或多个期望值。
    :param shots: `int` - 测量次数。默认值为 1000。
    :param initializer: 参数值的初始化器。默认值为 "None"，使用 0~2*pi 正态分布。
    :param dtype: 参数的数据类型。默认值为 None，表示使用默认数据类型 pyvqnet.kfloat32。
    :param name: 模块的名称。默认为空字符串。
    :param diff_method: 梯度计算的微分方法。默认值为 "parameter_shift"，可选 "random_coordinate_descent"。
    :param submit_kwargs: 提交量子电路的附加关键字参数，默认值：{"if_print_qcloud_log":False,"chip_id":"WK_C180","is_amend":True,"is_mapping":True,"is_optimization":True,"compile_level":3,"default_task_group_size":200,"test_qcloud_fake":False,"":"server_ip_address"}，当 test_qcloud_fake 设置为 True 时，使用本地 CPUQVM 模拟。
    :param query_kwargs: 查询量子结果的附加关键字参数，默认值：{"timeout":1,"total_timeout":60,"print_query_info":True,"sub_circuits_split_size":1}。
    :return: 一个能够计算量子电路的模块。

    Example::

        import pyqpanda3.core as pq
        import pyvqnet
        from pyvqnet.qnn.pq3.quantumlayer import QuantumBatchAsyncQcloudLayer

        def qfun(input,param):
            measure_qubits = [0,2]
            m_qlist = range(6)
            cir = pq.QCircuit(6)
            cir << (pq.RZ(m_qlist[0],input[0]))
            cir << pq.CNOT(m_qlist[0],m_qlist[1])
            cir << pq.RY(m_qlist[1],param[0])
            cir << pq.CNOT(m_qlist[0],m_qlist[2])
            cir << pq.RZ(m_qlist[1],input[1])
            cir << pq.RY(m_qlist[2],param[1])
            cir << pq.H(m_qlist[2])
            m_prog = pq.QProg(cir)


            for idx, ele in enumerate(measure_qubits):
                m_prog << pq.measure(m_qlist[ele], m_qlist[idx])  # pylint: disable=expression-not-assigned
            return m_prog

        l = QuantumBatchAsyncQcloudLayer(qfun,
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
            cir = pq.QCircuit(6)
            cir<<pq.RZ(m_qlist[0],input[0])
            cir<<pq.CNOT(m_qlist[0],m_qlist[1])
            cir<<pq.RY(m_qlist[1],param[0])
            cir<<pq.CNOT(m_qlist[0],m_qlist[2])
            cir<<pq.RZ(m_qlist[1],input[1])
            cir<<pq.RY(m_qlist[2],param[1])
            cir<<pq.H(m_qlist[2])
            m_prog = pq.QProg(cir)

        
            
            return m_prog
        l = QuantumBatchAsyncQcloudLayer(qfun2,
                "3047DE8A59764BEDAC9C3282093B16AF",
                2,

                pauli_str_dict={'Z0 X1':10,'':-0.5,'Y2':-0.543,"":3333},
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


QuantumLayerAdjoint
============================

.. py:class:: pyvqnet.qnn.pq3.quantumlayer.QuantumLayerAdjoint(pq3_vqc_circuit, param_num, pauli_dicts, dtype = None, name="")

    该类使用 pyqpanda3 的 VQCircuit 接口，通过伴随方法计算量子电路中参数相对于哈密顿量的梯度。

    该类支持批量输入和多个哈密顿量输出。

    .. note::

        使用此接口时，必须使用 VQCircuit 的逻辑门来构建电路。

        目前支持的逻辑门有限，使用不支持的逻辑门将抛出异常。

        ``pq3_vqc_circuit`` 输入参数只能包含两个参数 `x` 和 `param`，这两个参数必须是一维数组或列表。

        在 ``pq3_vqc_circuit`` 函数中，用户必须使用 ``pyqpanda3.vqcircuit.VQCircuit().set_Param`` 来自定义输入和参数的处理方式。

        此外，用户必须在 ``param_num`` 中预先输入参数数量。该接口将初始化一个长度为 ``param_num`` 的参数 ``m_para``。

        请参考下方示例。

    :param pq3_vqc_circuit: 自定义的 pyqpanda3 VQCircuit 电路。
    :param param_num: 参数数量。:param pauli_dicts: 期望观测值，可以是列表。
    :param dtype: 参数类型，kfloat32 或 kfloat64，默认值：None，使用 kfloat32。
    :param name: 该接口的名称。
    :return: 返回一个 QuantumLayerAdjoint 实例


    Example::

        from pyvqnet.qnn.pq3 import QuantumLayerAdjoint
        from pyvqnet import tensor

        from pyqpanda3.vqcircuit import VQCircuit
        import pyqpanda3 as pq3

        l = 3
        n = 7
        def pqctest(x,param):
            vqc = VQCircuit()
            vqc.set_Param([len(param) +len(x)])
            w_offset = len(x)
            for j in range(len(x)):
                vqc << pq3.core.RX(j, vqc.Param([j  ]))
            for j in range(l):
                for i in range(n - 1):
                    vqc << pq3.core.CNOT(i, i + 1)
                for i in range(n):
                    vqc << pq3.core.RX(i, vqc.Param([w_offset + 3 * n * j + i]))
                        
                    vqc << pq3.core.RZ(i, vqc.Param([w_offset + 3 * n * j + i + n]))
                    vqc << pq3.core.RY(i, vqc.Param([w_offset + 3 * n * j + i + 2 * n]))
            
            return vqc

        Xn_string = ' '.join([f'X{i}' for i in range(n)])
        pauli_dict  = {Xn_string:1.}

        layer = QuantumLayerAdjoint(pqctest,3*l*n,pauli_dict)

        x = tensor.randn([2,5])
        x.requires_grad = True
        y = layer(x)
        y.backward()
        print(layer.m_para.grad)
        print(x.grad)

        Xn_string = ' '.join([f'X{i}' for i in range(n)])
        Zn_string = ' '.join([f'Z{i}' for i in range(n)])
        pauli_dict  = {Xn_string:1.,Zn_string:0.5}

        layer = QuantumLayerAdjoint(pqctest,3*l*n,pauli_dict)

        x = tensor.randn([2,5])
        x.requires_grad = True
        y = layer(x)
        y.backward()
        print(layer.m_para.grad)
        print(x.grad)

        Xn_string = ' '.join([f'X{i}' for i in range(n)])
        Zn_string = ' '.join([f'Z{i}' for i in range(n)])
        pauli_dict  = {Xn_string:1.,Zn_string:0.5}

        layer = QuantumLayerAdjoint(pqctest,3*l*n,pauli_dict)

        x = tensor.randn([1,5])
        x.requires_grad = True
        y = layer(x)
        y.backward()
        print(layer.m_para.grad)
        print(x.grad)

        Xn_string = ' '.join([f'X{i}' for i in range(n)])
        Zn_string = ' '.join([f'Z{i}' for i in range(n)])
        pauli_dict  = [{Xn_string:1.,Zn_string:0.5},{Xn_string:1.,Zn_string:0.5}]

        layer = QuantumLayerAdjoint(pqctest,3*l*n,pauli_dict)

        x = tensor.randn([1,5])
        x.requires_grad = True
        y = layer(x)
        y.backward()
        print(layer.m_para.grad)
        print(x.grad)

        Xn_string = ' '.join([f'X{i}' for i in range(n)])
        Zn_string = ' '.join([f'Z{i}' for i in range(n)])
        pauli_dict  = [{Xn_string:1.,Zn_string:0.5},{Xn_string:1.,Zn_string:0.5}]

        layer = QuantumLayerAdjoint(pqctest,3*l*n,pauli_dict)

        x = tensor.randn([2,5])
        x.requires_grad = True
        y = layer(x)
        y.backward()
        print(layer.m_para.grad)
        print(x.grad)
        """
        [-0.1086438, 0.1805159, 0.2619071,..., 0.1508062, 0.0329617,-0.0043367]
        <QTensor [63] DEV_CPU kfloat32>

        [[-0.0425088, 0.0187212,-0.0326243, 0.1314874,-0.0729216],
        [-0.0972663,-0.0371378,-0.0455299,-0.0170686,-0.0328533]]
        <QTensor [2, 5] DEV_CPU kfloat32>

        [ 0.0706403,-0.1070583, 0.0547093,...,-0.0183769,-0.0742296, 0.0026942]
        <QTensor [63] DEV_CPU kfloat32>

        [[-0.07577  ,-0.1364278, 0.0220043, 0.0690343, 0.0281384],
        [ 0.0075356,-0.1627405,-0.0381604, 0.1185545, 0.1409108]]
        <QTensor [2, 5] DEV_CPU kfloat32>

        [-0.0634308,-0.0128268, 0.0396237,...,-0.0350691,-0.116307 , 0.0164972]
        <QTensor [63] DEV_CPU kfloat32>

        [[-0.0823639,-0.0418629, 0.0105356, 0.0699336, 0.041226 ]]
        <QTensor [1, 5] DEV_CPU kfloat32>

        [-0.1281752, 0.0852512, 0.0678721,...,-0.080481 , 0.0202518,-0.0348869]
        <QTensor [63] DEV_CPU kfloat32>

        [[-0.0339751,-0.0330053,-0.0651799, 0.2171837,-0.1267595]]
        <QTensor [1, 5] DEV_CPU kfloat32>

        [ 0.305574 , 0.2730191, 0.0605986,...,-0.2138517,-0.2475468, 0.174026 ]
        <QTensor [63] DEV_CPU kfloat32>

        [[ 0.1867954,-0.0704528,-0.0603823,-0.0123921,-0.0938597],
        [-0.041001 ,-0.2520995, 0.0683114,-0.0986969, 0.1000023]]
        <QTensor [2, 5] DEV_CPU kfloat32>
        """

VQCQCloudLayer
============================

.. py:class:: pyvqnet.qnn.pq3.vqc_qcloud_layer.VQCQCloudLayer(vqc_module, qcloud_token, pauli_str_dict=None, shots=1000, name="", submit_kwargs={}, query_kwargs={})

    将 VQC 模块提交至 QCloud 真实芯片或 pyqpanda3 本地模拟器执行。

    前向传播：不执行 VQNet 的量子变分电路计算，而是调用量子真实芯片或 qpanda 本地模拟器进行计算。

    反向传播：使用 parameter_shift 规则计算梯度。对于每个输入维度及 VQC 中的每个可训练参数，
    生成 +/- pi/2 偏移的电路并提交计算，获取结果后计算雅可比矩阵。梯度被设置到输入张量和 VQC 的可训练参数上。

    .. note::

        单次电路提交至 QCloud 的默认总超时时间为 60 秒。如果因 QCloud 繁忙导致超时，可以在 ``query_kwargs`` 中设置 ``total_timeout`` 键来指定等待秒数。

    .. note::

        不能在 ``vqc_module`` 中定义测量函数（如 ``MeasureAll``）。应通过 ``pauli_str_dict`` 参数指定可观测量。
        例如：``VQCQCloudLayer(vqc_module, token, pauli_str_dict={'Z0': 1, 'Z1': 1})``。

    :param vqc_module: VQNet VQC 模块，必须包含一个 save_ir=True 的 QMachine。
    :param qcloud_token: QCloud API 令牌。如果使用本地模拟器，则传入空字符串。
    :param pauli_str_dict: 用于期望值计算的泡利算符字典。默认值为 None，表示执行测量操作。
    :param shots: 测量次数。默认值为 1000。
    :param name: 模块名称。默认为空字符串。
    :param submit_kwargs: 提交量子电路的附加关键字参数，默认值：{"if_print_qcloud_log":False,"chip_id":"WK_C180","is_amend":True,"is_mapping":True,"is_optimization":True,"compile_level":3,"default_task_group_size":200,"test_qcloud_fake":False,"":"server_ip_address"}，当 test_qcloud_fake 设置为 True 时，使用本地 CPUQVM 模拟。
    :param query_kwargs: 查询量子结果的附加关键字参数。默认值：{"timeout":1,"total_timeout":60, "print_query_info":True,"sub_circuits_split_size":1}。

    Example::

        from pyvqnet.qnn.vqc import *
        from pyvqnet.qnn.pq3 import VQCQCloudLayer
        from pyvqnet.nn import Module
        import pyvqnet

        class QModel(Module):
            def __init__(self, num_wires, dtype):
                super(QModel, self).__init__()
                self._num_wires = num_wires
                self._dtype = dtype
                self.qm = QMachine(num_wires, dtype=dtype, save_ir=True)
                self.rx_layer = RX(has_params=True, trainable=False, wires=0)
                self.u1 = U1(has_params=True, trainable=True, wires=[1])
                self.cnot = CNOT(wires=[0, 1])

            def forward(self, x, *args, **kwargs):
                self.qm.reset_states(x.shape[0])
                self.rx_layer(params=x[:, [0]], q_machine=self.qm)
                self.cnot(q_machine=self.qm)
                self.u1(q_machine=self.qm)
                return x

        qmodel = QModel(num_wires=2, dtype=pyvqnet.kcomplex64)
        layer = VQCQCloudLayer(
            qmodel,
            "your_qcloud_token",
            pauli_str_dict={'Z0': 1, 'Z1': 1},
            shots=1000,
            submit_kwargs={"test_qcloud_fake": True},
        )
        x = pyvqnet.tensor.QTensor([[0.5, 0.3], [0.5, 0.3]], requires_grad=True)
        y = layer(x)
        y.backward()
        print(x.grad)
        print(qmodel.u1.params.grad)

grad
===============
.. py:function:: pyvqnet.qnn.pq3.quantumlayer.grad(quantum_prog_func, input_params, *args)

    grad 函数提供了一个计算用户设计的参数化量子电路参数梯度的接口。
    用户可以按照下方示例使用 pyQPanda3 设计电路运行函数 ``quantum_prog_func``，并将其作为参数传递给 grad 函数。
    grad 函数的第二个参数是您想要计算梯度位置的量子逻辑门参数坐标。
    返回值的形状为 [参数数量, 输出数量]。

    :param quantum_prog_func: 由 pyQPanda3 设计的量子电路运行函数。
    :param input_params: 需要计算梯度的参数。
    :param \*args: 传递给 quantum_prog_func 函数的其他参数。
    :return:
        参数的梯度


    Examples::

        from pyvqnet.qnn.pq3 import grad, ProbsMeasure
        import pyqpanda3.core as pq

        def pqctest(param):
            machine = pq.CPUQVM()
        
            qubits = range(2)
            circuit = pq.QCircuit(2)

            circuit<<pq.RX(qubits[0], param[0])

            circuit<<pq.RY(qubits[1], param[1])
            circuit<<pq.CNOT(qubits[0], qubits[1])

            circuit<<pq.RX(qubits[1], param[2])

            prog = pq.QProg()
            prog<<circuit

            EXP = ProbsMeasure(machine,prog,[1])
            return EXP


        g = grad(pqctest, [0.1,0.2, 0.3])
        print(g)
        exp = pqctest([0.1,0.2, 0.3])
        print(exp)







QLinear
==============

QLinear 实现了一种量子全连接算法。首先将数据编码为量子态，然后通过量子电路进行演化操作和测量，最终得到全连接结果。

.. image:: ./images/qlinear_cir.png

.. py:class:: pyvqnet.qnn.qlinear.QLinear(input_channels,output_channels,machine: str = "CPU")

    量子全连接模块。全连接模块的输入形状为（输入通道数, 输出通道数）。注意该层不包含变分量子参数。

    :param input_channels: `int` - 输入通道数。
    :param output_channels: `int` - 输出通道数。
    :param machine: `str` - 使用的虚拟机，默认使用 CPU 模拟。
    :return: 量子全连接层。

    Example::

        from pyvqnet.tensor import QTensor
        from pyvqnet.qnn.qlinear import QLinear
        params = [[0.37454012, 0.95071431, 0.73199394, 0.59865848, 0.15601864, 0.15599452], 
        [1.37454012, 0.95071431, 0.73199394, 0.59865848, 0.15601864, 0.15599452],
        [1.37454012, 1.95071431, 0.73199394, 0.59865848, 0.15601864, 0.15599452],
        [1.37454012, 1.95071431, 1.73199394, 1.59865848, 0.15601864, 0.15599452]]
        m = QLinear(6, 2)
        input = QTensor(params, requires_grad=True)
        output = m(input)
        output.backward()
        print(output)

        #[
        #[0.0568473, 0.1264389],
        #[0.1524036, 0.1264389],
        #[0.1524036, 0.1442845],
        #[0.1524036, 0.1442845]
        #]



Qconv
==========================

    Qconv 是一个量子卷积算法接口。
    量子卷积操作利用量子电路对经典数据执行卷积运算。它不需要计算乘法和加法操作，只需将数据编码为量子态，
    然后通过量子电路进行演化操作和测量，最终获得卷积结果。
    根据卷积核范围内的输入数据数量申请相同数量的量子比特，然后构建量子电路进行计算。

    .. image:: ./images/qcnn.png

    量子电路的编码方式为：首先在每个量子比特上插入 :math:`RY` 和 :math:`RZ` 门，
    然后在任意两个量子比特上使用 :math:`Z` 和 :math:`U3` 门进行纠缠和信息交换。以下是 4 量子比特的示例：

    .. image:: ./images/qcnn_cir.png

.. py:class:: pyvqnet.qnn.qcnn.qconv.QConv(input_channels,output_channels,quantum_number,stride=(1, 1),padding=(0, 0),kernel_initializer=normal,machine:str = "CPU",dtype=None, name ="")

    量子卷积模块。用量子电路替代 Conv2D 卷积核。卷积模块的输入形状为 (批大小, 输入通道数, 高度, 宽度) `Samuel et al. (2020) <https://arxiv.org/abs/2012.12177>`_ 。

        :param input_channels: `int` - 输入通道数。
        :param output_channels: `int` - 输出通道数。
        :param quantum_number: `int` - 单个卷积核的大小。
        :param stride: `tuple` - 步长，默认为 (1,1)。
        :param padding: `tuple` - 填充，默认为 (0,0)。
        :param kernel_initializer: `callable` - 默认为正态分布。
        :param machine: `str` - 使用的虚拟机，默认为 CPU 模拟。
        :param dtype: 参数的数据类型，默认值：None，使用默认数据类型：kfloat32，表示 32 位浮点数。
        :param name: 该模块的名称，默认为 ""。

        :return: 量子卷积层。


        Example::

            from pyvqnet.tensor import tensor
            from pyvqnet.qnn.qcnn.qconv import QConv
            x = tensor.ones([1,3,4,4])
            layer = QConv(input_channels=3, output_channels=2, quantum_number=4, stride=(2, 2))
            y = layer(x)
            print(y)

            # [
            # [[[-0.0889078, -0.0889078],
            #  [-0.0889078, -0.0889078]],
            # [[0.7992646, 0.7992646],
            #  [0.7992646, 0.7992646]]]
            # ]

量子逻辑门
************************************

处理量子比特的方式是量子逻辑门。通过使用量子逻辑门，我们有意识地演化量子态。量子逻辑门是量子算法的基础。

基本量子逻辑门
=============================

在本节中，我们使用本源量子开发的 pyqpanda3 的各种逻辑门来构建量子电路并进行量子模拟。
pyQPanda3 当前支持的逻辑门可参考 pyQPanda3 量子逻辑门的定义。
此外，VQNet 还封装了一些量子机器学习中常用的量子逻辑门组合：


BasicEmbeddingCircuit
============================

.. py:function:: pyvqnet.qnn.pq3.template.BasicEmbeddingCircuit(input_feat,qlist)
    
    将 n 个二进制特征编码到 n 个量子比特的基态中。

    例如，对于 ``features=([0, 1, 1])``，在量子系统中的基态为 :math:`|011 \rangle`。

    :param input_feat: 大小为 ``(n)`` 的二进制输入。
    :param qlist: 用于构建模板电路的量子比特。
    :return: 量子电路。


    Example::

            from pyvqnet.qnn.pq3.template import BasicEmbeddingCircuit
            import pyqpanda3.core as pq
            from pyvqnet import tensor
            input_feat = tensor.QTensor([1,1,0])
            
            qlist = range(3)
            circuit = BasicEmbeddingCircuit(input_feat,qlist)
            print(circuit)


AngleEmbeddingCircuit
============================

.. py:function:: pyvqnet.qnn.pq3.template.AngleEmbeddingCircuit(input_feat,qubits,rotation:str='X')

    将 :math:`N` 个特征编码到 :math:`n` 个量子比特的旋转角度中，其中 :math:`N \leq n`。

    旋转方式可选择：'X'、'Y'、'Z'，由 ``rotation`` 参数定义：

    * ``rotation='X'`` 将特征作为 RX 旋转的角度。

    * ``rotation='Y'`` 将特征作为 RY 旋转的角度。

    * ``rotation='Z'`` 将特征作为 RZ 旋转的角度。

    ``features`` 的长度必须小于或等于量子比特数。如果 ``features`` 的长度小于量子比特数，则电路不会施加剩余的旋转门。

    :param input_feat: 表示参数的 numpy 数组。
    :param qubits: 量子比特索引。
    :param rotation: 使用的旋转方式，默认为 "X"。
    :return: 量子电路。

    Example::

        from pyvqnet.qnn.pq3.template import AngleEmbeddingCircuit
        import numpy as np 
        m_qlist = range(2)

        input_feat = np.array([2.2, 1])
        C = AngleEmbeddingCircuit(input_feat,m_qlist,'X')
        print(C)
        C = AngleEmbeddingCircuit(input_feat,m_qlist,'Y')
        print(C)
        C = AngleEmbeddingCircuit(input_feat,m_qlist,'Z')
        print(C)

IQPEmbeddingCircuits
============================

.. py:function:: pyvqnet.qnn.pq3.template.IQPEmbeddingCircuits(input_feat,qubits,rep:int = 1)
    
    使用 IQP 电路的对角门将 :math:`n` 个特征编码到 :math:`n` 个量子比特中。

    该编码方法由 `Havlicek et al. (2018) <https://arxiv.org/pdf/1804.11326.pdf>`_ 提出。

    基础 IQP 电路可以通过指定 ``n_repeats`` 进行重复。

    :param input_feat: 表示参数的 numpy 数组。
    :param qubits: 量子比特索引列表。
    :param rep: 量子电路块的重复次数，默认次数为 1。
    :return: 量子电路。

    Example::

        import numpy as np
        from pyvqnet.qnn.pq3.template import IQPEmbeddingCircuits
        input_feat = np.arange(1,100)
        qlist = range(3)
        circuit = IQPEmbeddingCircuits(input_feat,qlist,rep = 3)
        print(circuit)


RotCircuit
============================

.. py:function:: pyvqnet.qnn.pq3.template.RotCircuit(para,qubits)

    任意单量子比特旋转。qlists 的数量应为 1，参数数量应为 3。

    .. math::

        R(\phi,\theta,\omega) = RZ(\omega)RY(\theta)RZ(\phi)= \begin{bmatrix}
        e^{-i(\phi+\omega)/2}\cos(\theta/2) & -e^{i(\phi-\omega)/2}\sin(\theta/2) \\
        e^{-i(\phi-\omega)/2}\sin(\theta/2) & e^{i(\phi+\omega)/2}\cos(\theta/2)
        \end{bmatrix}.

    :param para: 表示参数 :math:`[\phi, \theta, \omega]` 的 numpy 数组。
    :param qubits: 量子比特索引，仅接受单量子比特。
    :return: 量子电路。

    Example::

        from pyvqnet.qnn.pq3.template import RotCircuit
        import pyqpanda3.core as pq
        from pyvqnet import tensor

        m_qlist = 1

        param =tensor.QTensor([3,4,5])
        c = RotCircuit(param,m_qlist)
        print(c)

CRotCircuit
============================

.. py:function:: pyvqnet.qnn.pq3.template.CRotCircuit(para,control_qubits,rot_qubits)

    受控旋转算子。

    .. math:: 
        
        CR(\phi, \theta, \omega) = \begin{bmatrix}
        1 & 0 & 0 & 0 \\
        0 & 1 & 0 & 0\\
        0 & 0 & e^{-i(\phi+\omega)/2}\cos(\theta/2) & -e^{i(\phi-\omega)/2}\sin(\theta/2)\\
        0 & 0 & e^{-i(\phi-\omega)/2}\sin(\theta/2) & e^{i(\phi+\omega)/2}\cos(\theta/2)
        \end{bmatrix}.

    :param para: 表示参数 :math:`[\phi, \theta, \omega]` 的 numpy 数组。
    :param control_qubits: 控制量子比特索引，数量应为 1。
    :param rot_qubits: 旋转量子比特索引，数量应为 1。
    :return: 量子电路。

    Example::

        from pyvqnet.qnn.pq3.template import CRotCircuit
        import pyqpanda3.core as pq
        import numpy as np
        m_qlist = range(1)
        control_qlist = [1]
        param = np.array([3,4,5])
        cir = CRotCircuit(param,control_qlist,m_qlist)

        print(cir)


CSWAPcircuit
============================

.. py:function:: pyvqnet.qnn.pq3.template.CSWAPcircuit(qubits)

    受控 SWAP 电路。

    .. math:: 
        
        CSWAP = \begin{bmatrix}
        1 & 0 & 0 & 0 & 0 & 0 & 0 & 0 \\
        0 & 1 & 0 & 0 & 0 & 0 & 0 & 0 \\
        0 & 0 & 1 & 0 & 0 & 0 & 0 & 0 \\
        0 & 0 & 0 & 1 & 0 & 0 & 0 & 0 \\
        0 & 0 & 0 & 0 & 1 & 0 & 0 & 0 \\
        0 & 0 & 0 & 0 & 0 & 0 & 1 & 0 \\
        0 & 0 & 0 & 0 & 0 & 1 & 0 & 0 \\
        0 & 0 & 0 & 0 & 0 & 0 & 0 & 1
        \end{bmatrix}.

    .. note:: 
        
        提供的第一个量子比特对应 **控制量子比特**。

    :param qubits: 量子比特索引列表。第一个量子比特为控制量子比特。qlist 的长度必须为 3。
    :return: 量子电路。

    Example::

        from pyvqnet.qnn.pq3 import CSWAPcircuit
        import pyqpanda3.core as pq
        m_machine = pq.CPUQVM()

        m_qlist = range(3)

        c =CSWAPcircuit([m_qlist[1],m_qlist[2],m_qlist[0]])
        print(c)


Controlled_Hadamard
=======================

.. py:function:: pyvqnet.qnn.pq3.template.Controlled_Hadamard(qubits)
    
    受控 Hadamard 逻辑门

    .. math:: 
        CH = \begin{bmatrix}
        1 & 0 & 0 & 0 \\
        0 & 1 & 0 & 0 \\
        0 & 0 & \frac{1}{\sqrt{2}} & \frac{1}{\sqrt{2}} \\
        0 & 0 & \frac{1}{\sqrt{2}} & -\frac{1}{\sqrt{2}}
        \end{bmatrix}.

    :param qubits: 量子比特索引。

    Examples::

        import pyqpanda3.core as pq

        machine = pq.CPUQVM()
        
        qubits =range(2)
        from pyvqnet.qnn.pq3 import Controlled_Hadamard

        cir = Controlled_Hadamard(qubits)
        print(cir)

CCZ
==============

.. py:function:: pyvqnet.qnn.pq3.template.CCZ(qubits)

    受控-受控-Z 逻辑门。

    .. math::

        CCZ =
        \begin{pmatrix}
        1 & 0 & 0 & 0 & 0 & 0 & 0 & 0\\
        0 & 1 & 0 & 0 & 0 & 0 & 0 & 0 & 0\\
        0 & 0 & 1 & 0 & 0 & 0 & 0 & 0 & 0\\
        0 & 0 & 0 & 1 & 0 & 0 & 0 & 0\\
        0 & 0 & 0 & 0 & 0 & 1 & 0 & 0 & 0\\
        0 & 0 & 0 & 0 & 0 & 0 & 1 & 0 & 0\\
        0 & 0 & 0 & 0 & 0 & 0 & 0 & 1 & 0 & 0\\
        0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & -1
        \end{pmatrix}

    :param qubits: 量子比特索引。

    :return:
        pyQPanda3 QCircuit

    Example::

        import pyqpanda3.core as pq

        machine = pq.CPUQVM()

        qubits = range(3)

        from pyvqnet.qnn.pq3 import CCZ

        cir = CCZ(qubits)


FermionicSingleExcitation
===========================

.. py:function:: pyvqnet.qnn.pq3.template.FermionicSingleExcitation(weight, wires, qubits)

    耦合簇单激发算符，用于泡利矩阵张量积的指数化。矩阵形式如下：

    .. math::

        \hat{U}_{pr}(\theta) = \mathrm{exp} \{ \theta_{pr} (\hat{c}_p^\dagger \hat{c}_r
        -\mathrm{H.c.}) \},

    :param weight: 量子比特 p 上的可变参数。
    :param wires: 表示区间 [r, p] 内量子比特索引的子集。最小长度必须为 2。第一个索引值解释为 r，最后一个索引值解释为 p。
        中间索引通过 CNOT 门计算量子比特集合的奇偶性。
    :param qubits: 量子比特索引。

    :return:
        pyQPanda3 QCircuit

    Examples::

        from pyvqnet.qnn.pq3 import FermionicSingleExcitation, expval

        weight=0.5
        import pyqpanda3.core as pq
        machine = pq.CPUQVM()

        qlists = range(3)

        cir = FermionicSingleExcitation(weight, [1, 0, 2], qlists)


FermionicDoubleExcitation
============================

.. py:function:: pyvqnet.qnn.pq3.template.FermionicDoubleExcitation(weight, wires1, wires2, qubits)

    耦合簇双激发算符，用于泡利矩阵的张量积，矩阵形式如下：

    .. math::

        \hat{U}_{pqrs}(\theta) = \mathrm{exp} \{ \theta (\hat{c}_p^\dagger \hat{c}_q^\dagger
        \hat{c}_r \hat{c}_s - \mathrm{H.c.}) \},

    其中 :math:`\hat{c}` 和 :math:`\hat{c}^\dagger` 是费米子湮灭和
    产生算符，索引 :math:`r, s` 和 :math:`p, q` 分别对应占据和
    空分子轨道。通过 `Jordan-Wigner 变换
    <https://arxiv.org/abs/1208.5986>`_，上述定义的费米子算符可以
    用泡利矩阵表示（详见
    `arXiv:1805.04340 <https://arxiv.org/abs/1805.04340>`_）

    .. math::

        \hat{U}_{pqrs}(\theta) = \mathrm{exp} \Big\{
        \frac{i\theta}{8} \bigotimes_{b=s+1}^{r-1} \hat{Z}_b \bigotimes_{a=q+1}^{p-1}
        \hat{Z}_a (\hat{X}_s \hat{X}_r \hat{Y}_q \hat{X}_p +
        \hat{Y}_s \hat{X}_r \hat{Y}_q \hat{Y}_p +\\ \hat{X}_s \hat{Y}_r \hat{Y}_q \hat{Y}_p +
        \hat{X}_s \hat{X}_r \hat{X}_q \hat{Y}_p - \mathrm{H.c.} ) \Big\}

    :param weight: 可变参数。
    :param wires1: 表示量子比特索引列表中区间 [s, r] 内的量子比特子集。第一个索引解释为 s，最后一个索引解释为 r。CNOT 门对中间索引进行操作以计算一组量子比特的奇偶性。
    :param wires2: 表示量子比特索引列表中区间 [q, p] 内的量子比特子集。第一个根索引解释为 q，最后一个索引解释为 p。CNOT 门对中间索引进行操作以计算一组量子比特的奇偶性。
    :param qubits: 量子比特索引。

    :return:
        pyQPanda3 QCircuit

    Examples::

        import pyqpanda3.core as pq
        from pyvqnet.qnn.pq3 import FermionicDoubleExcitation, expval
        machine = pq.CPUQVM()
        
        qlists = range(5)
        weight = 1.5
        cir = FermionicDoubleExcitation(weight,
                                        wires1=[0, 1],
                                        wires2=[2, 3, 4],
                                        qubits=qlists)


UCCSD
===================

.. py:function:: pyvqnet.qnn.pq3.template.UCCSD(weights, wires, s_wires, d_wires, init_state, qubits)

    实现酉耦合簇单双激发模拟（UCCSD）。UCCSD 是一种 VQE 模拟，常用于运行量子化学模拟。

    在一阶 Trotter 近似下，UCCSD 酉函数如下：

    .. math::

        \hat{U}(\vec{\theta}) =
        \prod_{p > r} \mathrm{exp} \Big\{\theta_{pr}
        (\hat{c}_p^\dagger \hat{c}_r-\mathrm{H.c.}) \Big\}
        \prod_{p > q > r > s} \mathrm{exp} \Big\{\theta_{pqrs}
        (\hat{c}_p^\dagger \hat{c}_q^\dagger \hat{c}_r \hat{c}_s-\mathrm{H.c.}) \Big\}
    
    其中 :math:`\hat{c}` 和 :math:`\hat{c}^\dagger` 是费米子湮灭和

    产生算符，索引 :math:`r, s` 和 :math:`p, q` 分别对应占据和
    空分子轨道。（更多细节请参见
    `arXiv:1805.04340 <https://arxiv.org/abs/1805.04340>`_）：

    :param weights: 大小为 ``(len(s_wires)+ len(d_wires))`` 的张量，包含输入到 Z 旋转 ``FermionicSingleExcitation`` 和 ``FermionicDoubleExcitation`` 的参数 :math:`\theta_{pr}` 和 :math:`\theta_{pqrs}`。
    :param wires: 待模板化的量子比特索引。
    :param s_wires: 包含量子比特索引 ``[r,...,p]`` 的列表序列，由单激发 :math:`\vert r, p \rangle = \hat{c}_p^\dagger \hat{c}_r \vert \mathrm{HF} \rangle` 生成，其中 :math:`\vert \mathrm{HF} \rangle` 表示 Hartee-Fock 参考态。
    :param d_wires: 列表的序列，每个包含两个列表，指定索引 ``[s, ...,r]`` 和 ``[q,..., p]``。定义双激发 :math:`\vert s, r, q, p \rangle = \hat{c}_p^\dagger \hat{c}_q^\dagger \hat{c}_r\hat{c}_s \vert \mathrm{HF} \rangle`。
    :param init_state: 长度为 ``len(wires)`` 的占据数向量，表示高频状态。``init_state`` 初始化量子比特状态。
    :param qubits: 量子比特索引。

    Examples::
        
        import pyqpanda3.core as pq
        from pyvqnet.tensor import tensor
        from pyvqnet.qnn.pq3 import UCCSD, expval
        machine = pq.CPUQVM()
        
        qlists = range(6)
        weight = tensor.zeros([8])
        cir = UCCSD(weight,wires = [0,1,2,3,4,5,6],
                                        s_wires=[[0, 1, 2], [0, 1, 2, 3, 4], [1, 2, 3], [1, 2, 3, 4, 5]],
                                        d_wires=[[[0, 1], [2, 3]], [[0, 1], [2, 3, 4, 5]], [[0, 1], [3, 4]], [[0, 1], [4, 5]]],
                                        init_state=[1, 1, 0, 0, 0, 0],
                                        qubits=qlists)

QuantumPoolingCircuit
============================

.. py:function:: pyvqnet.qnn.pq3.template.QuantumPoolingCircuit(sources_wires, sinks_wires, params,qubits)

    用于降采样的量子电路。

    为减少电路中的量子比特数量，首先在系统中创建量子比特对。将所有量子比特初始配对后，对每个量子比特对应用广义的 2 量子比特酉门。在应用这些二量子比特酉门后，对于神经网络的其余部分，每对量子比特中忽略一个量子比特。

    :param sources_wires: 被忽略的源量子比特索引。
    :param sinks_wires: 保留的目标量子比特索引。
    :param params: 输入参数。
    :param qubits: 量子比特索引。

    :return:
        pyQPanda3 QCircuit

    Examples::

        from pyvqnet.qnn.pq3.template import QuantumPoolingCircuit
        import pyqpanda3.core as pq
        from pyvqnet import tensor

        qlists = range(4)
        p = tensor.full([6], 0.35)
        cir = QuantumPoolingCircuit([0, 1], [2, 3], p, qlists)
        print(cir)

常用量子电路组合
***********************************************************
VQNet 提供了一些量子机器学习研究中常用的量子电路

HardwareEfficientAnsatz
=============================

.. py:class:: pyvqnet.qnn.pq3.ansatz.HardwareEfficientAnsatz(qubits,single_rot_gate_list,entangle_gate="CNOT",entangle_rules='linear',depth=1)

    硬件高效拟设的实现，出自论文：`Hardware-efficient Variational Quantum Eigensolver for Small Molecules <https://arxiv.org/pdf/1704.05018.pdf>`__。

    :param qubits: 量子比特索引。
    :param single_rot_gate_list: 单量子比特旋转门列表，由作用于每个量子比特的一个或多个旋转门组成。当前支持 Rx、Ry、Rz。
    :param entangle_gate: 非参数化纠缠门。支持 CNOT、CZ。默认值：CNOT。
    :param entangle_rules: 纠缠门在电路中的使用方式。``linear`` 表示纠缠门将作用于每个相邻量子比特。``all`` 表示纠缠门将作用于任意两个量子比特。默认值：``linear``。
    :param depth: 拟设深度，默认值：1。

    :return:
        一个 HardwareEfficientAnsatz 实例

    Example::

        import pyqpanda3.core as pq
        from pyvqnet.tensor import QTensor,tensor
        from pyvqnet.qnn.pq3.ansatz import HardwareEfficientAnsatz
        machine = pq.CPUQVM()

        qlist = range(4)
        c = HardwareEfficientAnsatz(qlist,["rx", "RY", "rz"],
                                entangle_gate="cnot",
                                entangle_rules="linear",
                                depth=1)
        w = tensor.ones([c.get_para_num()])

        cir = c.create_ansatz(w)
        print(cir)

BasicEntanglerTemplate
============================

.. py:class:: pyvqnet.qnn.pq3.template.BasicEntanglerTemplate(weights=None, num_qubits=1, rotation=pyqpanda3.RX)
    
    该层由每个量子比特上的单参数单量子比特旋转门组成，后接以闭环或环状组合的多个 CNOT 门。
    CNOT 门的环将每个量子比特与其相邻量子比特连接，最后一个量子比特被视为第一个量子比特的邻居。
    层数 :math:`L` 由参数 ``weights`` 的第一维确定。

    :param weights: 形状为 `(L, len(qubits))` 的权重张量。每个权重用作量子参数化门中的参数。默认值为：``None``，此时使用 `(1,1)` 正态分布随机数作为权重。
    :param num_qubits: 量子比特数，默认为 1。
    :param rotation: 使用的单参数单量子比特门，默认值为 ``pyqpanda3.RX``。
    :return:
        一个 BasicEntanglerTemplate 实例

    Example::

        import pyqpanda3.core as pq
        import numpy as np
        from pyvqnet.qnn.pq3 import BasicEntanglerTemplate
        np.random.seed(42)
        num_qubits = 5
        shape = [1, num_qubits]
        weights = np.random.random(size=shape)

        machine = pq.CPUQVM()

        qubits = range(num_qubits)

        circuit = BasicEntanglerTemplate(weights=weights, num_qubits=num_qubits, rotation=pq.RZ)
        result = circuit.compute_circuit()
        circuit.print_circuit(qubits)


StronglyEntanglingTemplate
============================

.. py:class:: pyvqnet.qnn.pq3.template.StronglyEntanglingTemplate(weights=None, num_qubits=1, ranges=None)

    由单量子比特旋转和纠缠器组成的层，如 `circuit-centric classifier design <https://arxiv.org/abs/1804.00633>`__ 中所述。
    参数 ``weights`` 包含每一层的权重。因此层数 :math:`L` 等于 ``weights`` 的第一维。
    它包含作用于 :math:`M` 个量子比特上的 2 量子比特 CNOT 门，:math:`i = 1,...,M`。每个门的第二个量子比特编号由公式 :math:`(i+r)\mod M` 给出，其中 :math:`r` 是一个称为 ``range`` 的超参数，且 :math:`0 < r < M`。

    :param weights: 形状为 ``(L, M, 3)`` 的权重张量，默认值：None，使用形状为 ``(1,1,3)`` 的随机张量。
    :param num_qubits: 量子比特数，默认值：1。
    :param ranges: 确定每个后续层的范围超参数的序列；默认值：None，使用 :math:`r=l \mod M` 作为 ranges 的值。
    :return:
        一个 StronglyEntanglingTemplate 实例

    Example::

        from pyvqnet.qnn.pq3 import StronglyEntanglingTemplate
        import pyqpanda3.core as pq
        from pyvqnet.tensor import *
        import numpy as np
        np.random.seed(42)
        num_qubits = 3
        shape = [2, num_qubits, 3]
        weights = np.random.random(size=shape)

        machine = pq.CPUQVM()

        qubits = range(num_qubits)

        circuit = StronglyEntanglingTemplate(weights, num_qubits=num_qubits )
        result = circuit.compute_circuit()
        print(result)
        circuit.print_circuit(qubits)


ComplexEntanglingTemplate
============================

.. py:class:: pyvqnet.qnn.pq3.ComplexEntanglingTemplate(weights,num_qubits,depth)

    由 U3 门和 CNOT 门组成的强纠缠层。
    该电路模板来自以下论文：https://arxiv.org/abs/1804.00633。

    :param weights: 参数，形状为 [depth, num_qubits, 3]。
    :param num_qubits: 量子比特数。
    :param depth: 子电路的深度。
    :return:
        一个 ComplexEntanglingTemplate 实例

    Example::

        from pyvqnet.qnn.pq3 import ComplexEntanglingTemplate
        import pyqpanda3.core as pq
        from pyvqnet.tensor import *
        depth=3
        num_qubits = 8
        shape = [depth, num_qubits, 3]
        weights = tensor.randn(shape)

        machine = pq.CPUQVM()

        qubits = range(num_qubits)

        circuit = ComplexEntanglingTemplate(weights, num_qubits=num_qubits,depth=depth)
        result = circuit.create_circuit(qubits)
        circuit.print_circuit(qubits)


Quantum_Embedding
============================

.. py:class:: pyvqnet.qnn.pq3.Quantum_Embedding(qubits, machine, num_repetitions_input, depth_input, num_unitary_layers, num_repetitions)

    使用 RZ、RY、RZ 创建变分量子电路，将经典数据编码为量子态。
    参考 `Quantum embeddings for machine learning <https://arxiv.org/abs/2001.03622>`_。

    初始化该类后，其成员函数 ``compute_circuit`` 是运行函数，可作为参数输入到 ``QuantumLayerV2`` 类中，构成量子机器学习模型的一个层。

    :param qubits: pyQPanda3 申请的量子比特。
    :param machine: pyQPanda3 申请的量子虚拟机。
    :param num_repetitions_input: 子模块中输入编码的重复次数。
    :param depth_input: 输入数据的特征维度。
    :param num_unitary_layers: 每个子模块中变分量子门的重复次数。
    :param num_repetitions: 子模块的重复次数。
    :return:
        一个 Quantum_Embedding 实例

    Example::

        from pyvqnet.qnn.pq3 import QuantumLayerV2,Quantum_Embedding
        from pyvqnet.tensor import tensor
        import pyqpanda3.core as pq
        depth_input = 2
        num_repetitions = 2
        num_repetitions_input = 2
        num_unitary_layers = 2

        loacl_machine = pq.CPUQVM()

        nq = depth_input * num_repetitions_input
        qubits = range(nq)
        cubes = range(nq)

        data_in = tensor.ones([12, depth_input])
        data_in.requires_grad = True

        qe = Quantum_Embedding(nq, loacl_machine, num_repetitions_input,
        depth_input, num_unitary_layers, num_repetitions)
        qlayer = QuantumLayerV2(qe.compute_circuit,
        qe.param_num)

        data_in.requires_grad = True
        y = qlayer.forward(data_in)
        y.backward()
        print(data_in.grad)


测量量子电路
************************************

expval
============================

.. py:function:: pyvqnet.qnn.pq3.measure.expval(machine,prog,pauli_str_dict,shots=1000,noise_model=None)

    提供的哈密顿量观测量的期望值。

    如果观测量为 :math:`0.7Z\otimes X\otimes I+0.2I\otimes Z\otimes I`，
    则哈密顿量字典为 ``{{'Z0, X1':0.7} ,{'Z1':0.2}}``。

    expval API 目前支持 pyQPanda3 模拟器。

    :param machine: pyQPanda3 创建的量子机器。
    :param prog: pyQPanda3 创建的量子程序。
    :param pauli_str_dict: 哈密顿量观测值。
    :param shots: 测量次数，默认为 1000。
    :param noise_model: 应用的噪声模型，默认为 None（理想模拟）。

    :return: 期望值。

    Example::

        import pyqpanda3.core as pq
        from pyvqnet.qnn.pq3.measure import expval
        input = [0.56, 0.1]
        m_machine = pq.CPUQVM()

        m_qlist = range(3)
        cir = pq.QCircuit(3)
        cir<<pq.RZ(m_qlist[0],input[0])
        cir<<pq.CNOT(m_qlist[0],m_qlist[1])
        cir<<pq.RY(m_qlist[1],input[1])
        cir<<pq.CNOT(m_qlist[0],m_qlist[2])
        m_prog = pq.QProg(cir)

        pauli_dict = {'Z0 X1':10,'Y2':-0.543}
        exp2 = expval(m_machine,m_prog,pauli_dict)
        print(exp2)
 


QuantumMeasure
=============================

.. py:function:: pyvqnet.qnn.pq3.measure.QuantumMeasure(machine,prog,measure_qubits:list,shots:int = 1000, qcloud_option="",noise_model=None)

    计算量子电路测量结果。返回通过蒙特卡洛方法获得的测量值。

    有关测量的更多详细信息，请参考 pyQPanda3 文档。

    QuantumMeasure API 目前仅支持 pyQPanda3 的 ``CPUQVM`` 或 ``QCloud``。

    :param machine: pyQPanda3 分配的量子虚拟机。
    :param prog: pyQPanda3 创建的量子程序。
    :param measure_qubits: 包含测量比特索引的列表。
    :param shots: 测量次数，默认值为 1000。
    :param qcloud_option: 设置 qcloud 配置，默认值为 ""，可以传入一个 QCloudOptions 类，仅在使用 qcloud 时有效。
    :param noise_model: 应用的噪声模型，默认为 None（理想模拟）。
    :return: 返回通过蒙特卡洛方法获得的测量结果。

    Example::

        from pyqpanda3.core import *
        circuit = QCircuit(3)
        circuit << H(0)
        circuit << P(2, 0.2)
        circuit << RX(1, 0.9)
        circuit << RX(0, 0.6)
        circuit << RX(1, 0.3)
        circuit << RY(1, 0.3)
        circuit << RY(2, 2.7)
        circuit << RX(0, 1.5)

        prog = QProg()
        prog.append(circuit)

        machine = CPUQVM()


        from pyvqnet.qnn.pq3.measure import probs_measure,quantum_measure

        measure_result = quantum_measure(machine,prog,[2,0])
        print(measure_result)


ProbsMeasure
============================

.. py:function:: pyvqnet.qnn.pq3.measure.ProbsMeasure(machine,prog,measure_qubits:list,shots=1,noise_model=None)

    计算电路概率测量结果。

    有关概率测量的更多详细信息，请参考 pyQPanda3 文档。

    ProbsMeasure API 目前仅支持 pyQPanda 的 ``CPUQVM`` 或 ``QCloud``。

    :param measure_qubits: 包含测量量子比特索引的列表。
    :param prog: qpanda 创建的量子程序。
    :param machine: pyQPanda 分配的量子虚拟机。
    :param shots: 测量次数，默认为 1，表示计算理论值。
    :param noise_model: 应用的噪声模型，默认为 None（理想模拟）。
    :return: 按字典序排列的测量量子比特。


    Example::

        from pyqpanda3.core import *
        from pyvqnet.qnn.pq3.measure import probs_measure
        circuit = QCircuit(3)
        circuit << H(0)
        circuit << P(2, 0.2)
        circuit << RX(1, 0.9)
        circuit << RX(0, 0.6)
        circuit << RX(1, 0.3)
        circuit << RY(1, 0.3)
        circuit << RY(2, 2.7)
        circuit << RX(0, 1.5)

        prog = QProg()
        prog.append(circuit)
        prog.append(measure(0, 0))
        prog.append(measure(1, 1))
        prog.append(measure(2, 2))

        machine = CPUQVM()

        measure_result = probs_measure(machine,prog,[2,0])
        print(measure_result)
        #[0.04796392899146941, 0, 0.4760180355042653, 0.4760180355042653]


DensityMatrixFromQstate
===========================
.. py:function:: pyvqnet.qnn.pq3.measure.DensityMatrixFromQstate(state, indices)
    
    计算量子态在特定量子比特集合上的密度矩阵。

    :param state: 状态向量的一维列表。列表大小应为 ``(2**N,)``，对于量子比特数 ``N``，qstate 应从 000 到 111。
    :param indices: 所考虑子系统中量子比特索引的列表。
    :return: 
        大小为 ``(2**len(indices), 2**len(indices))`` 的密度矩阵。

    Example::

        from pyvqnet.qnn.pq3.measure import DensityMatrixFromQstate
        qstate = [(0.9306699299765968+0j), (0.18865613455240968+0j), (0.1886561345524097+0j), (0.03824249173404786+0j), -0.048171819846746615j, -0.00976491131165138j, -0.23763904794287155j, -0.048171819846746615j]
        print(DensityMatrixFromQstate(qstate,[0,1]))
        # [[0.86846704+0.j 0.1870241 +0.j 0.17604699+0.j 0.03791166+0.j]
        # [0.1870241 +0.j 0.09206345+0.j 0.03791166+0.j 0.01866219+0.j]
        # [0.17604699+0.j 0.03791166+0.j 0.03568649+0.j 0.00768507+0.j]
        # [0.03791166+0.j 0.01866219+0.j 0.00768507+0.j 0.00378301+0.j]]


VN_Entropy
==============
.. py:function:: pyvqnet.qnn.pq3.measure.VN_Entropy(state, indices, base=None)
    
    计算给定量子比特列表上状态向量的冯·诺依曼熵。

    .. math::

        S( \rho ) = -\text{Tr}( \rho \log ( \rho ))

    :param state: 状态向量的一维列表。列表大小应为 ``(2**N,)``，对于量子比特数 ``N``，qstate 应从 000 到 111。
    :param indices: 所考虑子系统中量子比特索引的列表。
    :param base: 对数的底数。如果为 None，则使用自然对数。

    :return: 冯·诺依曼熵的浮点值。

    Example::

        from pyvqnet.qnn.pq3.measure import VN_Entropy
        qstate = [(0.9022961387408862 + 0j), -0.06676534788028633j,
        (0.18290448232350312 + 0j), -0.3293638014158896j,
        (0.03707657410649268 + 0j), -0.06676534788028635j,
        (0.18290448232350312 + 0j), -0.013534006039561714j] 
        print(VN_Entropy(qstate, [0, 1]))
        #0.14592917648464448

Mutal_Info
==============
.. py:function:: pyvqnet.qnn.pq3.measure.Mutal_Info(state, indices0, indices1, base=None)

    计算两个子量子比特列表上状态向量的互信息。

    .. math::

        I(A, B) = S(\rho^A) + S(\rho^B) - S(\rho^{AB})
        其中 :math:`S` 是冯·诺依曼熵。

    互信息是衡量两个子系统之间相关性的指标。更具体地说，它量化了一个系统通过测量另一个系统可以获取的信息量。

    每个状态都可以作为计算基下的状态向量给出。

    :param state: 状态向量的一维列表。列表大小应为 ``(2**N,)``，对于量子比特数 ``N``，qstate 应从 000 到 111。
    :param indices0: 第一个子系统中量子比特索引的列表。
    :param indices1: 第二个子系统中量子比特索引的列表。
    :param base: 对数的底数。如果为 None，则使用自然对数。默认为 None。

    :return: 子系统之间的互信息

    Example::

        from pyvqnet.qnn.pq3.measure import Mutal_Info
        qstate = [(0.9022961387408862 + 0j), -0.06676534788028633j,
        (0.18290448232350312 + 0j), -0.3293638014158896j,
        (0.03707657410649268 + 0j), -0.06676534788028635j,
        (0.18290448232350312 + 0j), -0.013534006039561714j]
        print(Mutal_Info(qstate, [0], [2], 2))
        #0.13763425302805887

Purity
=========================

.. py:function:: pyvqnet.qnn.pq3.measure.Purity(state, qubits_idx)

    从状态向量计算特定量子比特的纯度。

    .. math::

        \gamma = \text{Tr}(\rho^2)
        
    其中 :math:`\rho` 是密度矩阵。归一化量子态的纯度满足 :math:`\frac{1}{d} \leq \gamma \leq 1`，
    其中 :math:`d` 是希尔伯特空间的维度。
    纯态的纯度为 1。

    :param state: 从 pyqpanda3 获得的量子态。
    :param qubits_idx: 要计算纯度的量子比特索引。

    :return:
        纯度

    Examples::

        from pyvqnet.qnn.pq3.measure import Purity
        qstate = [(0.9306699299765968 + 0j), (0.18865613455240968 + 0j),
        (0.1886561345524097 + 0j), (0.03824249173404786 + 0j),
        -0.048171819846746615j, -0.00976491131165138j, -0.23763904794287155j,
        -0.048171819846746615j]
        pp = Purity(qstate, [1])
        print(pp)
        #0.902503479761881
