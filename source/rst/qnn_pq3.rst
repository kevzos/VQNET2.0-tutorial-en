Use o módulo de aprendizado de máquina quântico pyQPanda3
#########################################################

.. warning::

    A parte de computação quântica da seguinte interface usa pyqpanda3.

    Se você usar a função QCloud neste módulo, haverá erros ao importar pyqpanda2 no código ou ao usar a interface de pacote relacionada ao pyqpanda2 do pyvqnet.

Camada de computação quântica
***********************************

.. _QuantumLayer_pq3:

QuantumLayer
============================

Se você está familiarizado com a sintaxe do pyQPanda3, pode usar a interface QuantumLayer para personalizar o simulador pyqpanda3 para cálculo.

.. py:class:: pyvqnet.qnn.pq3.quantumlayer.QuantumLayer(qprog_with_measure,para_num,diff_method:str = "parameter_shift",delta:float = 0.01,dtype=None,name="")

    Módulo de computação abstrata de camada quântica variacional. Use pyQPanda3 para simular um circuito quântico parametrizado e obter os resultados de medição. Esta camada quântica variacional herda o módulo de cálculo de gradiente do framework VQNet. Ela pode usar o método de desvio de parâmetro para calcular o gradiente dos parâmetros do circuito, treinar modelos de circuito quântico variacional ou incorporar circuitos quânticos variacionais em modelos híbridos quânticos e clássicos.

    :param qprog_with_measure: Função de operação e medição de circuito quântico construída com pyQPanda.
    :param para_num: ``int`` - número de parâmetros.
    :param diff_method: Método para resolver gradientes de parâmetros do circuito quântico, "parameter shift" ou "finite difference", padrão é deslocamento de parâmetro.
    :param delta: \delta ao calcular gradientes por diferença finita.
    :param dtype: tipo de dado do parâmetro, padrão: None, usa o tipo de dado padrão: kfloat32, representando números de ponto flutuante de 32 bits.
    :param name: o nome deste módulo, padrão é "".

    :return: um módulo que pode calcular circuitos quânticos.

    .. note::

        qprog_with_measure é uma função de circuito quântico definida no pyQPanda3.

        Esta função deve conter dois parâmetros, input e parameter, como entrada da função (mesmo que um parâmetro não seja realmente usado), e a saída é o resultado da medição ou valor esperado do circuito (precisa ser np.ndarray ou uma lista contendo valores), caso contrário não funcionará corretamente no QpandaQCircuitVQCLayerLite.

        O uso da função de circuito quântico qprog_with_measure (input, param) pode ser consultado no exemplo abaixo.

        ``input``: Dado clássico unidimensional de entrada. Se não houver, insira None

        ``param``: Parâmetros do circuito quântico variacional unidimensional a serem treinados

    .. note::

        Esta classe tem os apelidos ``QuantumLayerV2``, ``QpandaQCircuitVQCLayerLite``.

    Exemplo::

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

        #dados clássicos como entrada
        input = QTensor([[1,2,3,4],[4,2,2,3],[3.0,3,2,2]] )

        #circuitos forward
        rlt = pqc(input)

        print(rlt)

        grad = ones(rlt.data.shape)*1000
        #circuitos backward
        rlt.backward(grad)

        print(pqc.m_para.grad)

QpandaQProgVQCLayer
=============================

.. py:class:: pyvqnet.qnn.pq3.quantumlayer.QuantumLayerV3(origin_qprog_func,para_num,qvm_type="cpu", pauli_str_dict=None, shots=1000, initializer=None,dtype=None,name="")

    Envia o circuito quântico parametrizado para o simulador de amplitude total local QPanda3 para cálculo e treina os parâmetros no circuito.
    Suporta dados em lote e usa a regra de deslocamento de parâmetro para estimar o gradiente dos parâmetros.
    Para CRX, CRY, CRZ, esta camada usa a fórmula em https://iopscience.iop.org/article/10.1088/1367-2630/ac2cb3, e o restante das portas lógicas usa o método de desvio de parâmetro padrão para calcular o gradiente.

    :param origin_qprog_func: A função de circuito quântico chamável construída por QPanda.
    :param para_num: ``int`` - Número de parâmetros; os parâmetros são unidimensionais.
    :param qvm_type: ``str`` - Tipo de simulador pyqpanda3 a ser usado, ``cpu`` ou ``gpu``, padrão ``cpu``.
    :param pauli_str_dict: ``dict|list`` - Dicionário ou lista de dicionários representando os operadores de Pauli no circuito quântico. O padrão é None.
    :param shots: ``int`` - Número de medições. O padrão é 1000.
    :param initializer: Inicializador para valores de parâmetros. O padrão é None.
    :param dtype: Tipo de dado do parâmetro. O padrão é None, o que significa usar o tipo de dado padrão.
    :param name: Nome do módulo. O padrão é a string vazia.

    :return: Retorna uma classe QpandaQProgVQCLayer

    .. note::

        origin_qprog_func é uma função de circuito quântico definida pelo usuário usando pyQPanda3.

        Esta função deve conter dois parâmetros, input e parameter, como entrada da função (mesmo que um parâmetro não seja realmente usado), e a saída é dados do tipo pyqpanda3.core.QProg, caso contrário não pode ser executada corretamente no QuantumLayerV3.

        origin_qprog_func (input,param )

        ``input``: classe de array definida pelo usuário para entrada de dados clássicos unidimensionais

        ``param``: entrada array_like de parâmetros de circuito quântico unidimensional definidos pelo usuário

    .. note::

        Esta classe tem um apelido ``QuantumLayerV3`` .

    Exemplo::

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

Quando você instala a versão mais recente do pyqpanda3, pode usar esta interface para definir um circuito variacional e enviá-lo ao chip real originqc para operação.

.. py:class:: pyvqnet.qnn.pq3.quantumlayer.QuantumBatchAsyncQcloudLayer(origin_qprog_func, qcloud_token, para_num, pauli_str_dict=None, shots = 1000, initializer=None, dtype=None, name="", diff_method="parameter_shift", submit_kwargs={}, query_kwargs={})
    
    Um módulo de computação abstrata para o chip real originqc usando pyqpanda3 QCLOUD. Ele envia circuitos quânticos parametrizados para o chip real e obtém resultados de medição.
    Se diff_method == "random_coordinate_descent", a camada selecionará aleatoriamente um único parâmetro para calcular o gradiente, e outros parâmetros permanecerão zero. Referência: https://arxiv.org/abs/2311.00088

    .. note::

        qcloud_token é o token de api que você solicitou da plataforma em nuvem.

        origin_qprog_func precisa retornar dados do tipo pyqpanda3.core.QProg. Se pauli_str_dict não for definido, é necessário garantir que a medição foi inserida no QProg.

        origin_qprog_func deve estar no seguinte formato:

        origin_qprog_func(input,param)

            ``input``: Dados clássicos de entrada 1~2D. No caso de 2D, a primeira dimensão é o tamanho do lote

            ``param``: Parâmetros a serem treinados para o circuito quântico variacional 1D

    .. note::

        Na versão atual, o tempo limite total padrão para o envio de um único circuito ao QCloud é de 60 segundos. Se ocorrer um tempo limite devido ao QCloud estar ocupado, você pode definir o valor da chave ``total_timeout`` em ``query_kwargs`` para o número desejado de segundos de espera.

    :param origin_qprog_func: A função de circuito quântico variacional construída por QPanda, que deve retornar um QProg.
    :param qcloud_token: ``str`` - O tipo de máquina quântica ou o token de nuvem usado para execução.
    :param para_num: ``int`` - O número de parâmetros, o parâmetro é um QTensor de tamanho [para_num].
    :param pauli_str_dict: ``dict|list`` - Um dicionário ou lista de dicionários representando os operadores de Pauli no circuito quântico. O padrão é "None", que realiza operações de medição. Se um dicionário de operadores de Pauli for inserido, uma expectativa única ou múltiplas expectativas serão calculadas.
    :param shots: ``int`` - O número de medições. O valor padrão é 1000.
    :param initializer: Inicializador para valores de parâmetros. O padrão é "None", que usa uma distribuição normal 0~2*pi.
    :param dtype: O tipo de dado do parâmetro. O valor padrão é None, o que significa usar o tipo de dado padrão pyvqnet.kfloat32.
    :param name: O nome do módulo. O padrão é uma string vazia.
    :param diff_method: Método de diferenciação para cálculo do gradiente. O padrão é "parameter_shift", "random_coordinate_descent".
    :param submit_kwargs: Parâmetros de palavra-chave adicionais para envio de circuitos quânticos, padrão: {"if_print_qcloud_log":False,"chip_id":"WK_C180","is_amend":True,"is_mapping":True,"is_optimization":True,"compile_level":3,"default_task_group_size":200,"test_qcloud_fake":False,"":"server_ip_address"}, quando test_qcloud_fake é definido como True, simulação local CPUQVM.
    :param query_kwargs: Parâmetros de palavra-chave adicionais para consultar resultados quânticos, padrão: {"timeout":1,"total_timeout":60,"print_query_info":True,"sub_circuits_split_size":1}.
    :return: Um módulo que pode calcular circuitos quânticos.

    Exemplo::

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

    Esta classe usa a interface VQCircuit do pyqpanda3 para calcular os gradientes dos parâmetros em um circuito quântico em relação ao Hamiltoniano usando o método adjunto.

    Esta classe suporta entrada em lote e múltiplas saídas Hamiltonianas.

    .. note::

        Ao usar esta interface, você deve construir o circuito usando portas lógicas do VQCircuit.

        Atualmente, portas lógicas limitadas são suportadas; uma exceção será lançada se não for suportada.

        O parâmetro de entrada ``pq3_vqc_circuit`` pode conter apenas dois parâmetros, ``x`` e ``param``, que devem ser um array ou lista unidimensional.

        Na função ``pq3_vqc_circuit``, os usuários devem usar ``pyqpanda3.vqcircuit.VQCircuit().set_Param`` para personalizar como entradas e parâmetros são processados.

        Além disso, os usuários devem pré-inserir o número de parâmetros em ``param_num``. Esta interface inicializará um parâmetro ``m_para`` com um comprimento de ``param_num``.

        Veja o exemplo abaixo.

    :param pq3_vqc_circuit: Personaliza o circuito pyqpanda3 VQCircuit.
    :param param_num: Número de parâmetros.
    :param pauli_dicts: Observações esperadas, pode ser uma lista.
    :param dtype: Tipo de parâmetro, kfloat32 ou kfloat64, padrão: None, use kfloat32.
    :param name: O nome desta interface.
    :return: Retorna uma instância de QuantumLayerAdjoint


    Exemplo::

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

    Envia o Módulo VQC para o chip real QCloud ou para o simulador local pyqpanda3 para execução.

    Propagação forward: Em vez de executar o cálculo do circuito quântico variacional do VQNet, chama o chip quântico real ou o simulador local qpanda para o cálculo.

    Propagação backward: Usa a regra parameter_shift para calcular gradientes. Para cada dimensão de entrada e cada parâmetro treinável no VQC,
    gera circuitos deslocados em +/- pi/2 e os envia para cálculo, recupera resultados para calcular o jacobiano. Os gradientes são definidos no tensor de entrada e nos Parâmetros treináveis do VQC.

    .. note::

        O tempo limite total padrão para o envio de um único circuito ao QCloud é de 60 segundos. Se ocorrer um tempo limite devido ao QCloud estar ocupado, você pode definir a chave ``total_timeout`` em ``query_kwargs`` para especificar os segundos de espera.

    .. note::

        Você não pode definir uma função de medição (como ``MeasureAll``) em ``vqc_module``. A medição deve ser especificada através do parâmetro ``pauli_str_dict`` para indicar observáveis.
        Por exemplo: ``VQCQCloudLayer(vqc_module, token, pauli_str_dict={'Z0': 1, 'Z1': 1})``.

    :param vqc_module: Módulo VQC do VQNet, deve incluir uma QMachine com save_ir=True.
    :param qcloud_token: Token da API QCloud. Passe uma string vazia se estiver usando um simulador local.
    :param pauli_str_dict: Dicionário de operadores de Pauli para cálculo do valor esperado. O padrão é None, que realiza operação de medição.
    :param shots: Número de medições. O padrão é 1000.
    :param name: Nome do módulo. O padrão é string vazia.
    :param submit_kwargs: Parâmetros de palavra-chave adicionais para envio de circuitos quânticos, padrão: {"if_print_qcloud_log":False,"chip_id":"WK_C180","is_amend":True,"is_mapping":True,"is_optimization":True,"compile_level":3,"default_task_group_size":200,"test_qcloud_fake":False,"":"server_ip_address"}, quando test_qcloud_fake é definido como True, simulação local CPUQVM.
    :param query_kwargs: Argumentos de palavra-chave adicionais para consultar resultados quânticos. Padrão: {"timeout":1,"total_timeout":60, "print_query_info":True,"sub_circuits_split_size":1}.

    Exemplo::

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

    A função grad fornece uma interface para calcular o gradiente dos parâmetros do circuito quântico com parâmetros projetado pelo usuário.
    Os usuários podem usar pyQPanda3 para projetar a função de execução do circuito ``quantum_prog_func`` como mostrado abaixo, e passá-la como um parâmetro para a função grad.
    O segundo parâmetro da função grad são as coordenadas do gradiente do parâmetro da porta lógica quântica que você deseja calcular.
    A forma do valor de retorno é [número de parâmetros, número de saídas].

    :param quantum_prog_func: função de execução do circuito quântico projetada por pyQPanda3.
    :param input_params: parâmetros para os quais o gradiente será calculado.
    :param \*args: outros parâmetros de entrada para a função quantum_prog_func.
    :return:
        Gradiente dos parâmetros


    Exemplos::

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

QLinear implementa um algoritmo quântico de conexão completa. Primeiro, os dados são codificados em um estado quântico, e então a operação de evolução e medição são realizadas através de circuitos quânticos para obter o resultado final de conexão completa.

.. image:: ./images/qlinear_cir.png

.. py:class:: pyvqnet.qnn.qlinear.QLinear(input_channels,output_channels,machine: str = "CPU")

    Módulo quântico totalmente conectado. A entrada do módulo totalmente conectado tem a forma (canais de entrada, canais de saída). Note que esta camada não recebe parâmetros quânticos variacionais.

    :param input_channels: ``int`` - Número de canais de entrada.
    :param output_channels: ``int`` - Número de canais de saída.
    :param machine: ``str`` - A máquina virtual a ser usada, simulação CPU é usada por padrão.
    :return: Camada quântica totalmente conectada.

    Exemplo::

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

    Qconv é uma interface de algoritmo de convolução quântica.
    A operação de convolução quântica usa circuitos quânticos para realizar operações de convolução em dados clássicos. Ela não precisa calcular operações de multiplicação e adição. Basta codificar os dados em estados quânticos e, em seguida, realizar operações de evolução e medições através de circuitos quânticos para obter os resultados finais da convolução.
    Aplica o mesmo número de bits quânticos de acordo com o número de dados de entrada dentro do intervalo do kernel de convolução e, em seguida, constrói circuitos quânticos para o cálculo.

    .. image:: ./images/qcnn.png

    O circuito quântico é codificado inserindo primeiro portas :math:`RY` e :math:`RZ` em cada qubit, e depois usando :math:`Z` e :math:`U3` em quaisquer dois qubits para entrelaçar e trocar informações. Abaixo está um exemplo de 4 qubits

    .. image:: ./images/qcnn_cir.png

.. py:class:: pyvqnet.qnn.qcnn.qconv.QConv(input_channels,output_channels,quantum_number,stride=(1, 1),padding=(0, 0),kernel_initializer=normal,machine:str = "CPU", dtype=None, name ="")

    Módulo de convolução quântica. Substitui o kernel Conv2D por um circuito quântico. A entrada do módulo conv tem a forma (tamanho do lote, canais de entrada, altura, largura) `Samuel et al. (2020) <https://arxiv.org/abs/2012.12177>`_ .

        :param input_channels: ``int`` - Número de canais de entrada.
        :param output_channels: ``int`` - Número de canais de saída.
        :param quantum_number: ``int`` - O tamanho de um único kernel.
        :param stride: ``tuple`` - O passo, padrão é (1,1).
        :param padding: ``tuple`` - Preenchimento, padrão é (0,0).
        :param kernel_initializer: ``callable`` - Padrão é distribuição normal.
        :param machine: ``str`` - A máquina virtual a ser usada, padrão é simulação CPU.
        :param dtype: O tipo de dado do parâmetro, padrão: None, usa o tipo de dado padrão: kfloat32, representando números de ponto flutuante de 32 bits.
        :param name: O nome deste módulo, padrão é "".

        :return: Camada de convolução quântica.


        Exemplo::

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

Portas lógicas quânticas
************************************

A maneira de processar bits quânticos é através de portas lógicas quânticas. Usando portas lógicas quânticas, evoluímos conscientemente os estados quânticos. A porta lógica quântica é a base do algoritmo quântico.

Porta lógica quântica básica
=============================

Nesta seção, usamos as várias portas lógicas do pyqpanda3 desenvolvidas pela Origin Quantum para construir circuitos quânticos e realizar simulações quânticas.
As portas lógicas atualmente suportadas pelo pyQPanda3 podem ser consultadas na definição de portas lógicas quânticas do pyQPanda3.
Além disso, o VQNet também encapsula algumas combinações de portas lógicas quânticas comumente usadas em aprendizado de máquina quântico:


BasicEmbeddingCircuit
============================

.. py:function:: pyvqnet.qnn.pq3.template.BasicEmbeddingCircuit(input_feat,qlist)
    
    Codifica n características binárias no estado fundamental de n qubits.

    Por exemplo, para ``features=([0, 1, 1])``, o estado fundamental é :math:`|011 \rangle` em um sistema quântico.

    :param input_feat: entrada binária de tamanho ``(n)``.
    :param qlist: qubits para construir o circuito template.
    :return: circuito quântico.


    Exemplo::

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

    Codifica :math:`N` características no ângulo de rotação de :math:`n` qubits, onde :math:`N \leq n`.

    A rotação pode ser escolhida como: 'X', 'Y', 'Z', conforme definido pelo parâmetro ``rotation``:

    * ``rotation='X'`` Usa a característica como o ângulo da rotação RX.

    * ``rotation='Y'`` Usa a característica como o ângulo da rotação RY.

    * ``rotation='Z'`` Usa a característica como o ângulo da rotação RZ.

    O comprimento de ``features`` deve ser menor ou igual ao número de qubits. Se o comprimento de ``features`` for menor que o número de qubits, o circuito não aplica as portas de rotação restantes.

    :param input_feat: array numpy representando os parâmetros.
    :param qubits: índices dos qubits.
    :param rotation: qual rotação usar, padrão é "X".
    :return: circuito quântico.

    Exemplo::

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
    
    Codifica :math:`n` características em :math:`n` qubits usando portas diagonais de um circuito IQP.

    A codificação foi proposta por `Havlicek et al. (2018) <https://arxiv.org/pdf/1804.11326.pdf>`_.

    O circuito IQP básico pode ser repetido especificando ``n_repeats``.

    :param input_feat: array numpy representando os parâmetros.
    :param qubits: lista de índices dos qubits.
    :param rep: Repetir o bloco do circuito quântico, o número padrão de vezes é 1.
    :return: circuito quântico.

    Exemplo::

        import numpy as np
        from pyvqnet.qnn.pq3.template import IQPEmbeddingCircuits
        input_feat = np.arange(1,100)
        qlist = range(3)
        circuit = IQPEmbeddingCircuits(input_feat,qlist,rep = 3)
        print(circuit)


RotCircuit
============================

.. py:function:: pyvqnet.qnn.pq3.template.RotCircuit(para,qubits)

    Rotação arbitrária de um único qubit. O número de qlists deve ser 1, e o número de parâmetros deve ser 3.

    .. math::

        R(\phi,\theta,\omega) = RZ(\omega)RY(\theta)RZ(\phi)= \begin{bmatrix}
        e^{-i(\phi+\omega)/2}\cos(\theta/2) & -e^{i(\phi-\omega)/2}\sin(\theta/2) \\
        e^{-i(\phi-\omega)/2}\sin(\theta/2) & e^{i(\phi+\omega)/2}\cos(\theta/2)
        \end{bmatrix}.

    :param para: array numpy representando os parâmetros :math:`[\phi, \theta, \omega]`.
    :param qubits: índice do qubit, apenas qubits únicos são aceitos.
    :return: circuito quântico.

    Exemplo::

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

    Operador Rot controlado.

    .. math:: 
        
        CR(\phi, \theta, \omega) = \begin{bmatrix}
        1 & 0 & 0 & 0 \\
        0 & 1 & 0 & 0\\
        0 & 0 & e^{-i(\phi+\omega)/2}\cos(\theta/2) & -e^{i(\phi-\omega)/2}\sin(\theta/2)\\
        0 & 0 & e^{-i(\phi-\omega)/2}\sin(\theta/2) & e^{i(\phi+\omega)/2}\cos(\theta/2)
        \end{bmatrix}.

    :param para: Um array numpy representando os parâmetros :math:`[\phi, \theta, \omega]`.
    :param control_qubits: índice do qubit de controle, o número de qubits deve ser 1.
    :param rot_qubits: índice do qubit de rotação, o número de qubits deve ser 1.
    :return: circuito quântico.

    Exemplo::

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

    Circuito SWAP controlado.

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
        
        O primeiro qubit fornecido corresponde ao **qubit de controle** .

    :param qubits: lista de índices dos qubits. O primeiro qubit é o qubit de controle. O comprimento de qlist deve ser 3.
    :return: O circuito quântico.

    Exemplo::

        from pyvqnet.qnn.pq3 import CSWAPcircuit
        import pyqpanda3.core as pq
        m_machine = pq.CPUQVM()

        m_qlist = range(3)

        c =CSWAPcircuit([m_qlist[1],m_qlist[2],m_qlist[0]])
        print(c)


Controlled_Hadamard
=======================

.. py:function:: pyvqnet.qnn.pq3.template.Controlled_Hadamard(qubits)
    
    Porta lógica Hadamard controlada

    .. math:: 
        CH = \begin{bmatrix}
        1 & 0 & 0 & 0 \\
        0 & 1 & 0 & 0 \\
        0 & 0 & \frac{1}{\sqrt{2}} & \frac{1}{\sqrt{2}} \\
        0 & 0 & \frac{1}{\sqrt{2}} & -\frac{1}{\sqrt{2}}
        \end{bmatrix}.

    :param qubits: índice do qubit.

    Exemplos::

        import pyqpanda3.core as pq

        machine = pq.CPUQVM()
        
        qubits =range(2)
        from pyvqnet.qnn.pq3 import Controlled_Hadamard

        cir = Controlled_Hadamard(qubits)
        print(cir)

CCZ
==============

.. py:function:: pyvqnet.qnn.pq3.template.CCZ(qubits)

    Porta lógica Z controlada-controlada.

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

    :param qubits: índice do qubit.

    :return:
        pyQPanda3 QCircuit

    Exemplo::

        import pyqpanda3.core as pq

        machine = pq.CPUQVM()

        qubits = range(3)

        from pyvqnet.qnn.pq3 import CCZ

        cir = CCZ(qubits)


FermionicSingleExcitation
===========================

.. py:function:: pyvqnet.qnn.pq3.template.FermionicSingleExcitation(weight, wires, qubits)

    Operador de excitação única de cluster acoplado para exponenciação de produtos tensoriais de matrizes de Pauli. A forma matricial é dada por:

    .. math::

        \hat{U}_{pr}(\theta) = \mathrm{exp} \{ \theta_{pr} (\hat{c}_p^\dagger \hat{c}_r
        -\mathrm{H.c.}) \},

    :param weight: parâmetro variável no qubit p.
    :param wires: representa um subconjunto de índices de qubits no intervalo [r, p]. O comprimento mínimo deve ser 2. O primeiro valor do índice é interpretado como r e o último valor do índice é interpretado como p.
        Os índices intermediários são operados por portas CNOT para calcular a paridade do conjunto de qubits.
    :param qubits: índices dos qubits.

    :return:
        pyQPanda3 QCircuit

    Exemplos::

        from pyvqnet.qnn.pq3 import FermionicSingleExcitation, expval

        weight=0.5
        import pyqpanda3.core as pq
        machine = pq.CPUQVM()

        qlists = range(3)

        cir = FermionicSingleExcitation(weight, [1, 0, 2], qlists)


FermionicDoubleExcitation
============================

.. py:function:: pyvqnet.qnn.pq3.template.FermionicDoubleExcitation(weight, wires1, wires2, qubits)

    Operador de dupla excitação de cluster acoplado para o produto tensorial de matrizes de Pauli, a forma matricial é dada por:

    .. math::

        \hat{U}_{pqrs}(\theta) = \mathrm{exp} \{ \theta (\hat{c}_p^\dagger \hat{c}_q^\dagger
        \hat{c}_r \hat{c}_s - \mathrm{H.c.}) \},

    onde :math:`\hat{c}` e :math:`\hat{c}^\dagger` são os operadores de aniquilação e
    criação de férmions e os índices :math:`r, s` e :math:`p, q` em orbitais moleculares
    ocupados e vazios respectivamente. Usando a `transformação de Jordan-Wigner
    <https://arxiv.org/abs/1208.5986>`_ o operador férmion definido acima pode ser escrito
    em termos da matriz de Pauli (veja
    `arXiv:1805.04340 <https://arxiv.org/abs/1805.04340>`_ para mais detalhes)

    .. math::

        \hat{U}_{pqrs}(\theta) = \mathrm{exp} \Big\{
        \frac{i\theta}{8} \bigotimes_{b=s+1}^{r-1} \hat{Z}_b \bigotimes_{a=q+1}^{p-1}
        \hat{Z}_a (\hat{X}_s \hat{X}_r \hat{Y}_q \hat{X}_p +
        \hat{Y}_s \hat{X}_r \hat{Y}_q \hat{Y}_p +\\ \hat{X}_s \hat{Y}_r \hat{Y}_q \hat{Y}_p +
        \hat{X}_s \hat{X}_r \hat{X}_q \hat{Y}_p - \mathrm{H.c.} ) \Big\}

    :param weight: parâmetro variável
    :param wires1: representa o subconjunto de qubits no intervalo [s, r] ocupado pela lista de índices dos qubits. O primeiro índice é interpretado como s e o último índice é interpretado como r. A porta CNOT opera nos índices intermediários para calcular a paridade de um conjunto de qubits.
    :param wires2: representa o subconjunto de qubits no intervalo [q, p] ocupado pela lista de índices dos qubits. O primeiro índice raiz é interpretado como q e o último índice é interpretado como p. A porta CNOT opera no índice intermediário para calcular a paridade de um conjunto de qubits.
    :param qubits: índices dos qubits.

    :return:
        pyQPanda3 QCircuit

    Exemplos::

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

    Implementa a Simulação de Excitação Única e Dupla de Cluster Acoplado Unitário (UCCSD). UCCSD é uma simulação VQE, comumente usada para executar simulações de química quântica.

    Na aproximação de Trotter de primeira ordem, a função unitária UCCSD é dada por:

    .. math::

        \hat{U}(\vec{\theta}) =
        \prod_{p > r} \mathrm{exp} \Big\{\theta_{pr}
        (\hat{c}_p^\dagger \hat{c}_r-\mathrm{H.c.}) \Big\}
        \prod_{p > q > r > s} \mathrm{exp} \Big\{\theta_{pqrs}
        (\hat{c}_p^\dagger \hat{c}_q^\dagger \hat{c}_r \hat{c}_s-\mathrm{H.c.}) \Big\}
    
    onde :math:`\hat{c}` e :math:`\hat{c}^\dagger` são os operadores de aniquilação e

    criação de férmions e os índices :math:`r, s` e :math:`p, q` são os orbitais moleculares
    ocupados e vazios, respectivamente. (Para mais detalhes veja
    `arXiv:1805.04340 <https://arxiv.org/abs/1805.04340>`_):

    :param weights: tensor de tamanho ``(len(s_wires)+ len(d_wires))`` contendo os parâmetros
     :math:`\theta_{pr}` e :math:`\theta_{pqrs}` de entrada para as rotações Z ``FermionicSingleExcitation`` e ``FermionicDoubleExcitation`` .
    :param wires: índices dos qubits a serem transformados em template
    :param s_wires: sequência de lista contendo índices de qubits ``[r,...,p]`` gerados por uma excitação única
     :math:`| r, p \rangle = \hat{c}_p^\dagger \hat{c}_r | \mathrm{HF} \rangle`, onde :math:`| \mathrm{HF} \rangle` denota o estado de referência de Hartree-Fock.
    :param d_wires: sequência de listas, cada uma contendo duas listas. Especifica os índices ``[s, ...,r]`` e ``[q,..., p]`` . Define a excitação dupla :math:`| s, r, q, p \rangle = \hat{c}_p^\dagger \hat{c}_q^\dagger \hat{c}_r \hat{c}_s | \mathrm{HF} \rangle` .
    :param init_state: vetor de número de ocupação de comprimento ``len(wires)`` representando o estado de alta frequência. ``init_state`` Inicializa o estado do qubit.
    :param qubits: Índice do qubit.

    Exemplos::
        
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

    Circuito quântico que reduz a amostragem de dados.

    Para reduzir o número de qubits no circuito, primeiro crie pares de qubits no sistema. Após emparelhar inicialmente todos os qubits, aplique a unitária generalizada de 2 qubits a cada par de qubits. E após aplicar estas duas unitárias de qubit, ignore um qubit em cada par de qubits para o resto da rede neural.

    :param sources_wires: Índices dos qubits de origem a serem ignorados.
    :param sinks_wires: Índices dos qubits de destino a serem mantidos.
    :param params: Parâmetros de entrada.
    :param qubits: Índices dos qubits.

    :return:
        pyQPanda3 QCircuit

    Exemplos::

        from pyvqnet.qnn.pq3.template import QuantumPoolingCircuit
        import pyqpanda3.core as pq
        from pyvqnet import tensor

        qlists = range(4)
        p = tensor.full([6], 0.35)
        cir = QuantumPoolingCircuit([0, 1], [2, 3], p, qlists)
        print(cir)

Combinações de circuitos quânticos comumente usadas
***********************************************************
O VQNet fornece alguns circuitos quânticos comumente usados em pesquisa de aprendizado de máquina quântico

HardwareEfficientAnsatz
=============================

.. py:class:: pyvqnet.qnn.pq3.ansatz.HardwareEfficientAnsatz(qubits,single_rot_gate_list,entangle_gate="CNOT",entangle_rules='linear',depth=1)

    Implementação do Hardware Efficient Ansatz introduzido no artigo: `Hardware-efficient Variational Quantum Eigensolver for Small Molecules <https://arxiv.org/pdf/1704.05018.pdf>`__ .

    :param qubits: índice do qubit.
    :param single_rot_gate_list: Lista de portas de rotação de qubit único consistindo em uma ou mais portas de rotação atuando em cada qubit. Atualmente suportadas são Rx, Ry, Rz.
    :param entangle_gate: Porta de entrelaçamento não paramétrica. Suporta CNOT, CZ. Padrão: CNOT.
    :param entangle_rules: Como a porta de entrelaçamento é usada no circuito. ``linear`` significa que a porta de entrelaçamento atuará em cada qubit adjacente. ``all`` significa que a porta de entrelaçamento atuará em quaisquer dois qubits. Padrão: ``linear``.
    :param depth: Profundidade no ansatz, padrão: 1.

    :return:
        Uma instância de HardwareEfficientAnsatz

    Exemplo::

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
    
    Uma camada consistindo de rotações de qubit único com um único parâmetro em cada qubit, seguidas por múltiplas portas CNOT combinadas em uma cadeia fechada ou anel.
    O anel de portas CNOT conecta cada qubit aos seus vizinhos, com o último qubit considerado vizinho do primeiro.
    O número de camadas :math:`L` é determinado pela primeira dimensão do parâmetro ``weights``.

    :param weights: Um tensor de pesos de forma `(L, len(qubits))`. Cada peso é usado como um parâmetro em uma porta paramétrica quântica. O valor padrão é: ``None``, então números aleatórios normalmente distribuídos ``(1,1)`` são usados como pesos.
    :param num_qubits: O número de qubits, padrão é 1.
    :param rotation: Use uma porta de qubit único com um parâmetro, ``pyqpanda3.RX`` é usado como valor padrão.
    :return:
        Uma instância de BasicEntanglerTemplate

    Exemplo::

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

    Camadas consistindo de rotações de qubit único e entrelaçadores, como no `design de classificador centrado em circuito <https://arxiv.org/abs/1804.00633>`__ .
    O parâmetro ``weights`` contém os pesos de cada camada. Então o número de camadas :math:`L` é igual à primeira dimensão de ``weights``.
    Ele contém portas CNOT de 2 qubits atuando em :math:`M` qubits, :math:`i = 1,...,M`. O segundo número de qubit de cada porta é dado pela fórmula :math:`(i+r)\mod M`, onde :math:`r` é um hiperparâmetro chamado ``range``, e :math:`0 < r < M`.

    :param weights: Tensor de pesos de forma ``(L, M, 3)``, valor padrão: None, usa um tensor aleatório de forma ``(1,1,3)``.
    :param num_qubits: Número de qubits, valor padrão: 1.
    :param ranges: Sequência que determina os hiperparâmetros de intervalo para cada camada subsequente; valor padrão: None, usa :math:`r=l \mod M` como o valor de ranges.
    :return:
        Uma instância de StronglyEntanglingTemplate

    Exemplo::

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

    Camada fortemente entrelaçada consistindo de portas U3 e portas CNOT.
    Este modelo de circuito é do seguinte artigo: https://arxiv.org/abs/1804.00633.

    :param weights: parâmetros, forma [depth,num_qubits,3]
    :param num_qubits: número de qubits.
    :param depth: profundidade do subcircuito.
    :return:
        Uma instância de ComplexEntanglingTemplate

    Exemplo::

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

    Use RZ, RY, RZ para criar um circuito quântico variacional para codificar dados clássicos em estados quânticos.
    Referência `Quantum embeddings for machine learning <https://arxiv.org/abs/2001.03622>`_.

    Após inicializar a classe, sua função membro ``compute_circuit`` é a função de execução, que pode ser usada como um parâmetro para a classe ``QuantumLayerV2`` para formar uma camada do modelo de aprendizado de máquina quântico.

    :param qubits: Os bits quânticos solicitados pelo pyQPanda3.
    :param machine: Máquina virtual quântica aplicada pelo pyQPanda3.
    :param num_repetitions_input: O número de repetições de codificação da entrada no submódulo.
    :param depth_input: A dimensão da característica dos dados de entrada.
    :param num_unitary_layers: O número de repetições da porta quântica variacional em cada submódulo.
    :param num_repetitions: O número de repetições do submódulo.
    :return:
        Uma instância de Quantum_Embedding

    Exemplo::

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


Medir circuitos quânticos
************************************

expval
============================

.. py:function:: pyvqnet.qnn.pq3.measure.expval(machine,prog,pauli_str_dict,shots=1000,noise_model=None)

    O valor esperado da observação Hamiltoniana fornecida.

    Se a observação for :math:`0.7Z\otimes X\otimes I+0.2I\otimes Z\otimes I`,
    então o dicionário Hamiltoniano será ``{{'Z0, X1':0.7} ,{'Z1':0.2}}``.

    A API expval agora suporta o simulador pyQPanda3.

    :param machine: A máquina quântica criada pelo pyQPanda3.
    :param prog: O programa quântico criado pelo pyQPanda3.
    :param pauli_str_dict: Valor observado Hamiltoniano.
    :param shots: Número de medições, padrão é 1000.
    :param noise_model: Modelo de ruído a ser aplicado, padrão é None (simulação ideal).

    :return: valor esperado.

    Exemplo::

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

    Calcula medições de circuitos quânticos. Retorna medições obtidas por métodos de Monte Carlo.

    Para mais detalhes sobre medição, consulte a documentação do pyQPanda3.

    A API QuantumMeasure atualmente suporta apenas ``CPUQVM`` ou ``QCloud`` do pyQPanda3.

    :param machine: A máquina virtual quântica alocada pelo pyQPanda3.
    :param prog: O programa quântico criado pelo pyQPanda3.
    :param measure_qubits: Lista contendo os índices dos bits de medição.
    :param shots: O número de medições, o valor padrão é 1000.
    :param qcloud_option: Define a configuração qcloud, o valor padrão é "", você pode passar uma classe QCloudOptions, que é útil apenas ao usar qcloud.
    :param noise_model: Modelo de ruído a ser aplicado, padrão é None (simulação ideal).
    :return: Retorna os resultados de medição obtidos pelo método de Monte Carlo.

    Exemplo::

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

    Calcula medições de probabilidade do circuito.

    Para mais detalhes, consulte a documentação do pyQPanda3 sobre medição de probabilidade.

    A API ProbsMeasure atualmente suporta apenas ``CPUQVM`` ou ``QCloud`` do pyQPanda.

    :param measure_qubits: Lista contendo os índices dos qubits de medição.
    :param prog: O programa quântico criado pelo qpanda.
    :param machine: A máquina virtual quântica alocada pelo pyQPanda.
    :param shots: Número de medições, padrão é 1, que calcula o valor teórico.
    :param noise_model: Modelo de ruído a ser aplicado, padrão é None (simulação ideal).
    :return: Mede qubits em ordem lexicográfica.


    Exemplo::

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
    
    Calcula a matriz densidade de um estado quântico sobre um conjunto específico de qubits.

    :param state: Lista 1D de vetores de estado. O tamanho desta lista deve ser ``(2**N,)``. Para um número de qubits ``N``, qstate deve começar de 000 -> 111.
    :param indices: Lista de índices de qubits no subsistema considerado.
    :return: 
        Matriz densidade de tamanho ``(2**len(indices), 2**len(indices))`` .

    Exemplo::

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
    
    Calcula a entropia de Von Neumann dado um vetor de estado em uma lista de qubits.

    .. math::

        S( \rho ) = -\text{Tr}( \rho \log ( \rho ))

    :param state: Lista 1D de vetores de estado. O tamanho desta lista deve ser ``(2**N,)``. Para um número de qubits ``N``, qstate deve começar de 000 -> 111.
    :param indices: Lista de índices de qubits no subsistema em consideração.
    :param base: Base do logaritmo. Se None, o logaritmo natural é usado.

    :return: valor de ponto flutuante da entropia de von Neumann.

    Exemplo::

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

    Calcula a informação mútua dado o vetor de estado em duas listas de sub-qubits.

    .. math::

        I(A, B) = S(\rho^A) + S(\rho^B) - S(\rho^{AB})
        onde :math:`S` é a entropia de von Neumann.

    Informação mútua é uma medida da correlação entre dois sub-sistemas. Mais especificamente, ela quantifica a quantidade de informação que um sistema pode obter ao medir o outro.

    Cada estado pode ser dado como um vetor de estado na base de computação.

    :param state: Lista 1D de vetores de estado. O tamanho desta lista deve ser ``(2**N,)``. Para um número de qubits ``N``, qstate deve começar de 000 -> 111.
    :param indices0: Lista de índices de qubits no primeiro subsistema.
    :param indices1: Lista de índices de qubits no segundo subsistema.
    :param base: Base dos logaritmos. Se None, logaritmos naturais são usados. Padrão é None.

    :return: Informação mútua entre subsistemas

    Exemplo::

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

    Calcula a pureza de um qubit específico a partir do vetor de estado.

    .. math::

        \gamma = \text{Tr}(\rho^2)
        
    onde :math:`\rho` é a matriz densidade. A pureza de um estado quântico normalizado satisfaz :math:`\frac{1}{d} \leq \gamma \leq 1` ,
    onde :math:`d` é a dimensão do espaço de Hilbert.
    A pureza de um estado puro é 1.

    :param state: estado quântico obtido do pyqpanda3
    :param qubits_idx: índice do qubit para o qual a pureza deve ser calculada

    :return:
        pureza

    Exemplos::

        from pyvqnet.qnn.pq3.measure import Purity
        qstate = [(0.9306699299765968 + 0j), (0.18865613455240968 + 0j),
        (0.1886561345524097 + 0j), (0.03824249173404786 + 0j),
        -0.048171819846746615j, -0.00976491131165138j, -0.23763904794287155j,
        -0.048171819846746615j]
        pp = Purity(qstate, [1])
        print(pp)
        #0.902503479761881
