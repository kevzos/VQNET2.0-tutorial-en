API de Machine Learning Quântico usando QPanda2
####################################################


.. warning::

    A parte de computação quântica da interface a seguir usa pyQPanda2.

    Devido a problemas de compatibilidade entre pyQPanda2 e pyqpanda3, você precisa instalar o pyqpnda2 por conta própria, `pip install pyqpanda`

Camada de Computação Quântica
***********************************

.. _QuantumLayer:

QuantumLayer
=================================

QuantumLayer é uma classe de pacote do módulo autograd que suporta circuitos quânticos variacionais. Você pode definir uma função como argumento, como ``qprog_with_measure``. Esta função precisa conter o circuito quântico definido por pyQPanda: geralmente contém circuito de codificação, circuito de evolução e operação de medição.
Esta classe QuantumLayer pode ser incorporada ao modelo híbrido de aprendizado de máquina quântico-clássico e minimizar a função objetivo ou função de perda do modelo híbrido quântico-clássico através do método clássico de descida de gradiente.
Você pode especificar o método de cálculo do gradiente dos parâmetros do circuito quântico em ``QuantumLayer`` alterando o parâmetro ``diff_method``. ``QuantumLayer`` atualmente suporta dois métodos: ``finite_diff`` e ``parameter-shift``.

O método ``finite_diff`` é um dos métodos numéricos mais tradicionais e comuns para estimar o gradiente de uma função. A ideia principal é substituir derivadas parciais por diferenças:

.. math::

    f^{\prime}(x)=\lim _{h \rightarrow 0} \frac{f(x+h)-f(x)}{h}


Para o método ``parameter-shift`` usamos a função objetivo, como:

.. math:: O(\theta)=\left\langle 0\left|U^{\dagger}(\theta) H U(\theta)\right| 0\right\rangle

É teoricamente possível calcular o gradiente dos parâmetros em relação ao Hamiltoniano em um circuito quântico pelo método mais preciso: ``parameter-shift``.

.. math::

    \nabla O(\theta)=
    \frac{1}{2}\left[O\left(\theta+\frac{\pi}{2}\right)-O\left(\theta-\frac{\pi}{2}\right)\right]

.. py:class:: pyvqnet.qnn.quantumlayer.QuantumLayer(qprog_with_measure,para_num,machine_type_or_cloud_token,num_of_qubits:int,num_of_cbits:int = 1,diff_method:str = "parameter_shift",delta:float = 0.01, dtype=None, name='')

    Módulo de cálculo abstrato para circuitos quânticos variacionais. Simula um circuito quântico parametrizado e obtém o resultado da medição.
    QuantumLayer herda de Module, para que possa calcular gradientes dos parâmetros do circuito e treinar modelos de circuitos quânticos variacionais ou incorporar circuitos quânticos variacionais em modelos híbridos quântico-clássicos.
    
    Esta classe não requer que você inicialize a máquina virtual na função ``qprog_with_measure``.

    :param qprog_with_measure: funções de circuitos quânticos chamáveis, construídas por pyQPanda2
    :param para_num: `int` - Número de parâmetros
    :param machine_type_or_cloud_token: tipo de máquina qpanda ou token QCLOUD do pyQPanda2
    :param num_of_qubits: número de qubits
    :param num_of_cbits: número de bits clássicos
    :param diff_method: 'parameter_shift' ou 'finite_diff'
    :param delta: delta para a diferença
    :param dtype: O tipo de dado do parâmetro, padrão: None, usa o tipo de dado padrão kfloat32, que representa um número de ponto flutuante de 32 bits.
    :param name: nome da camada de saída

    :return: um módulo capaz de calcular circuitos quânticos.

    .. note::
        qprog_with_measure é uma função de circuito quântico definida em pyQPanda2.

        Esta função deve conter os seguintes parâmetros, caso contrário não funcionará corretamente no QuantumLayer.

        qprog_with_measure (input,param,qubits,cbits,m_machine)

            `input`: array_like entrada 1-dim de dados clássicos

            `param`: array_like entrada 1-dim dos parâmetros do circuito quântico

            `qubits`: qubits alocados pelo QuantumLayer

            `cbits`: cbits alocados pelo QuantumLayer. Se seu circuito não usar cbits, você também deve reservar este parâmetro.

            `m_machine`: simulador criado pelo QuantumLayer

        Use o atributo ``m_para`` do QuantumLayer para obter os parâmetros de treinamento do circuito quântico variacional. O parâmetro é da classe ``QTensor``, que pode ser convertido em um array numpy usando a interface ``to_numpy()``.

    .. note::

        A classe possui um alias: `QpandaQCircuitVQCLayer` .

    Exemplo::

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
        #dados clássicos como entrada
        input = QTensor([[1,2,3,4],[40,22,2,3],[33,3,25,2.0]] )
        #circuito forward
        rlt = pqc(input)
        grad =  QTensor(np.ones(rlt.data.shape)*1000)
        #circuito backward
        rlt.backward(grad)
        print(rlt)
        # [
        # [0.2500000, 0.2500000, 0.2500000, 0.2500000],
        # [0.2500000, 0.2500000, 0.2500000, 0.2500000],
        # [0.2500000, 0.2500000, 0.2500000, 0.2500000]
        # ]

QuantumLayerV2
=================================

Se você está mais familiarizado com a sintaxe do pyQPanda2, use a classe QuantumLayerV2. Você pode definir a função de circuito quântico usando ``qubits``, ``cbits`` e ``machine``, e então passá-la como argumento ``qprog_with_measure`` do QuantumLayerV2.

.. py:class:: pyvqnet.qnn.quantumlayer.QuantumLayerV2(qprog_with_measure, para_num, diff_method: str = 'parameter_shift', delta: float = 0.01, dtype=None, name='')

    Módulo de cálculo abstrato para circuitos quânticos variacionais. Simula um circuito quântico parametrizado e obtém o resultado da medição.
    QuantumLayerV2 herda de Module, para que possa calcular gradientes dos parâmetros do circuito e treinar modelos de circuitos quânticos variacionais ou incorporar circuitos quânticos variacionais em modelos híbridos quântico-clássicos.
    
    Para usar este módulo, você precisa criar sua máquina virtual quântica e alocar qubits e cbits.

    :param qprog_with_measure: funções de circuitos quânticos chamáveis, construídas por pyQPanda2
    :param para_num: `int` - Número de parâmetros
    :param diff_method: 'parameter_shift' ou 'finite_diff'
    :param delta: delta para a diferença
    :param dtype: O tipo de dado do parâmetro, padrão: None, usa o tipo de dado padrão kfloat32, que representa um número de ponto flutuante de 32 bits.
    :param name: nome da camada de saída
    :return: um módulo capaz de calcular circuitos quânticos.

    .. note::
        qprog_with_measure é uma função de circuito quântico definida em pyQPanda.

        Esta função deve conter os seguintes parâmetros, caso contrário não funcionará corretamente no QuantumLayerV2.

        Comparado ao QuantumLayer, você deve alocar qubits e o simulador por conta própria,

        e você também pode precisar alocar cbits se qprog_with_measure precisar de medição quântica.

        qprog_with_measure (input,param)

        `input`: array_like entrada 1-dim de dados clássicos

        `param`: array_like entrada 1-dim dos parâmetros do circuito quântico

    .. note::

        A classe possui um alias: `QpandaQCircuitVQCLayerLite` .

    Exemplo::

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

        #dados clássicos como entrada
        input = QTensor([[1,2,3,4],[4,2,2,3],[3,3,2,2.0]] )

        #circuito forward
        rlt = pqc(input)

        grad =  QTensor(np.ones(rlt.data.shape)*1000)
        #circuito backward
        rlt.backward(grad)
        print(rlt)

        # [
        # [0.2500000, 0.2500000, 0.2500000, 0.2500000],
        # [0.2500000, 0.2500000, 0.2500000, 0.2500000],
        # [0.2500000, 0.2500000, 0.2500000, 0.2500000]
        # ]

 



NoiseQuantumLayer
=================================

Em um computador quântico real, devido às características físicas do bit quântico, sempre há erros de cálculo inevitáveis. Para simular melhor esse erro em uma máquina virtual quântica, o VQNet também suporta máquina virtual quântica com ruído. A simulação da máquina virtual quântica com ruído é mais próxima do computador quântico real. Podemos personalizar o tipo de porta lógica suportada e o modelo de ruído suportado pela porta lógica.
O modelo de ruído quântico existente suportado é definido no ``NoiseQVM`` do pyQPanda2.

Podemos usar ``NoiseQuantumLayer`` para definir uma miclassificação automática de circuitos quânticos. ``NoiseQuantumLayer`` suporta máquina virtual quântica pyQPanda2 com ruído. Você pode definir uma função como argumento ``qprog_with_measure``. Esta função precisa conter o circuito quântico definido por pyQPanda, e você também precisa passar um argumento ``noise_set_config``, usando a interface pyQPanda para configurar o modelo de ruído.

.. py:class:: pyvqnet.qnn.quantumlayer.NoiseQuantumLayer(qprog_with_measure, para_num, machine_type, num_of_qubits: int, num_of_cbits: int = 1, diff_method: str = 'parameter_shift', delta: float = 0.01, noise_set_config=None, dtype=None, name='')

    Módulo de cálculo abstrato para circuitos quânticos variacionais. Simula um circuito quântico parametrizado e obtém o resultado da medição.
    NoiseQuantumLayer herda de Module, para que possa calcular gradientes dos parâmetros do circuito e treinar modelos de circuitos quânticos variacionais ou incorporar circuitos quânticos variacionais em modelos híbridos quântico-clássicos.
    
    Este módulo deve ser inicializado com modelo de ruído através de ``noise_set_config``.

    :param qprog_with_measure: funções de circuitos quânticos chamáveis, construídas por pyQPanda2
    :param para_num: `int` - Número de parâmetros
    :param machine_type: tipo de máquina pyQPanda2
    :param num_of_qubits: número de qubits
    :param num_of_cbits: número de cbits
    :param diff_method: 'parameter_shift' ou 'finite_diff'
    :param delta: delta para a diferença
    :param noise_set_config: função de configuração de ruído
    :param dtype: O tipo de dado do parâmetro, padrão: None, usa o tipo de dado padrão kfloat32, que representa um número de ponto flutuante de 32 bits.
    :param name: nome da camada de saída
    
    :return: um módulo capaz de calcular circuitos quânticos com modelo de ruído.

    .. note::
        qprog_with_measure é uma função de circuito quântico definida em pyQPanda.

        Esta função deve conter os seguintes parâmetros, caso contrário não funcionará corretamente no NoiseQuantumLayer.

        qprog_with_measure (input,param,qubits,cbits,m_machine)

        `input`: array_like entrada 1-dim de dados clássicos

        `param`: array_like entrada 1-dim dos parâmetros do circuito quântico

        `qubits`: qubits alocados pelo NoiseQuantumLayer

        `cbits`: cbits alocados pelo NoiseQuantumLayer. Se seu circuito não usar cbits, você também deve reservar este parâmetro.

        `m_machine`: simulador criado pelo NoiseQuantumLayer

    Exemplo::

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
            # Calcula probabilidades para cada estado
            probabilities = counts / 100
            # Obtém a expectativa do estado
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

Aqui está um exemplo de ``noise_set_config``, onde adicionamos o modelo de ruído BITFLIP_KRAUS_OPERATOR com argumento de ruído p=0.01 às portas quânticas ``RX`` , ``RY`` , ``RZ`` , ``X`` , ``Y`` , ``Z`` , ``H``, etc.

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

    Uma camada wrapper para implementar propagação forward e backward com circuitos Qiskit no VQNet. QISKIT_VQC é uma classe que define um circuito quântico Qiskit e sua função de execução.
    O exemplo a seguir demonstra como funciona. Esta camada suporta apenas entradas de circuito e pesos como parâmetros.
    
    :param cirq_vqc: Uma classe que define a definição, o backend e as funções de execução de um circuito Qiskit.
    :param para_num: `int` - O número de parâmetros.
    :return: Uma classe capaz de executar modelos de circuitos quânticos qiskit.

    Example::


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
                # --- Definição do circuito ---

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

        #define a classe de circuitos qiskit
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
            print("iniciando treinamento..............")
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

    Uma camada de encapsulamento de circuito cirq para implementar propagação forward e backward no vqnet. CIRQ_VQC é uma classe que exige que os usuários definam um circuito quântico cirq e sua função `run`. O exemplo a seguir demonstra seu princípio de funcionamento.
    Esta camada suporta apenas entradas de circuito e pesos como parâmetros.

    :param cirq_vqc: Uma classe que define a definição, o backend e as funções de execução de um circuito Cirq.
    :param para_num: `int` - O número de parâmetros.
    :return: Uma classe capaz de executar o modelo de circuito quântico Cirq.


    .. note::

        O código de exemplo a seguir requer `cirq==1.5.0, numpy <2`.

    Example::

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
                ###define parâmetros variacionais
                param = sympy.symbols(f'theta(0:24)')
                self.theta = np.asarray(param).reshape((4,6))

                ###define circuitos
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

        #define a classe de circuitos cirq
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
            print("iniciando treinamento..............")
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

Portas Quânticas
***********************************

A forma de lidar com qubits é chamada de portas quânticas. Usando portas quânticas, evoluímos conscientemente os estados quânticos. Portas quânticas são a base dos algoritmos quânticos.

Portas quânticas básicas
=================================

No VQNet, usamos cada porta lógica do pyQPanda desenvolvido pela Origin Quantum para construir circuitos quânticos e realizar simulações quânticas.
As portas atualmente suportadas pelo pyQPanda podem ser definidas na seção de portas quânticas do pyQPanda.
Além disso, o VQNet também encapsula algumas combinações de portas quânticas comumente usadas em aprendizado de máquina quântico.



AmplitudeEmbeddingCircuit
=================================

.. py:function:: pyvqnet.qnn.template.AmplitudeEmbeddingCircuit(input_feat, qubits)

    Codifica :math:`2^n` características no vetor de amplitude de :math:`n` qubits.
    Para representar um vetor de estado quântico válido, a norma L2 de ``features`` deve ser um.

    :param input_feat: array numpy que representa os parâmetros
    :param qubits: qubits alocados por pyQPanda
    :return: circuitos quânticos

    Example::

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




APIs de Machine Learning Quântico usando pyQPanda2
***************************************************************************

Redes Adversariais Generativas Quânticas para aprendizado e carregamento de distribuições aleatórias
====================================================================================================

O algoritmo de Redes Adversariais Generativas Quânticas (`QGAN <https://www.nature.com/articles/s41534-019-0223-2>`_ ) usa circuitos quânticos variacionais puros para preparar os estados quânticos gerados com distribuição aleatória específica, o que pode reduzir as portas lógicas necessárias para gerar estados quânticos específicos e reduzir a complexidade dos circuitos quânticos. Ele usa a estrutura clássica do modelo GAN, que possui dois submodelos: Generator e Discriminator. O Generator gera uma distribuição específica para o circuito quântico. E o Discriminator discrimina as amostras de dados geradas pelo Generator e as amostras de dados de treinamento reais distribuídas aleatoriamente.
Aqui está um exemplo do VQNet implementando o aprendizado e carregamento de distribuições aleatórias do QGAN baseado no artigo `Quantum Generative Adversarial Networks for learning and loading random distributions <https://www.nature.com/articles/s41534-019-0223-2>`_ de Christa Zoufal.

.. image:: ./images/qgan-arch.PNG
   :width: 600 px
   :align: center

|

Para realizar a construção da classe ``QGANAPI`` de rede adversarial generativa quântica pelo VQNet, o gerador quântico é usado para preparar o estado inicial dos dados distribuídos reais. O número de bits quânticos é 3, e o número de repetições do módulo de circuito paramétrico interno do gerador quântico é 1. Enquanto isso, KL é usado como métrica para o carregamento de distribuição aleatória do QGAN.

.. code-block::

    import pickle
    import os
    import pyqpanda as pq
    from pyvqnet.qnn.qgan.qgan_utils import QGANAPI
    import numpy as np

    num_of_qubits = 3  # configuração do artigo
    rep = 1
    number_of_data = 10000
    # Carrega amostras de dados de diferentes distribuições
    mu = 1
    sigma = 1
    real_data = np.random.lognormal(mean=mu, sigma=sigma, size=number_of_data)


    # inicializa
    save_dir = None
    qgan_model = QGANAPI(
        real_data,
        # distribuição de dados gerada por numpy, 1 - dim.
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

A seguir está o módulo ``train`` do QGAN.

.. code-block::

    # treina
    qgan_model.train()  # treina qgan


O módulo ``eval`` do QGAN foi projetado para desenhar a curva da função de perda e o diagrama de distribuição de probabilidade entre a distribuição aleatória preparada pelo QGAN e a distribuição real.

.. code-block::

    # mostra a função de distribuição de probabilidade da distribuição gerada e da distribuição real
    qgan_model.eval(real_data)  #desenha pdf

O módulo ``get_trained_quantum_parameters`` do QGAN é usado para obter os parâmetros de treinamento e gerá-los como um array numpy. Se ``save_DIR`` não estiver vazio, os parâmetros de treinamento são salvos em um arquivo. O módulo ``Load_param_and_eval`` do QGAN carrega os parâmetros de treinamento, e o módulo ``get_circuits_with_trained_param`` obtém o circuito pyQPanda gerado pelo gerador quântico após o treinamento.

.. code-block::

    # obtém parâmetros quânticos treinados
    param = qgan_model.get_trained_quantum_parameters()
    print(f" trained param {param}")

    #carrega arquivos de parâmetros salvos
    if save_dir is not None:
        path = os.path.join(
            save_dir, qgan_model._start_time + "trained_qgan_param.pickle")
        with open(path, "rb") as file:
            t3 = pickle.load(file)
        param = t3["quantum_parameters"]
        print(f" trained param {param}")

    #mostra a função de distribuição de probabilidade da distribuição gerada e da distribuição real
    qgan_model.load_param_and_eval(param)

    #calcula métrica
    print(qgan_model.eval_metric(param, "kl"))

    #obtém o circuito quântico do gerador
    m_machine = pq.CPUQVM()
    m_machine.init_qvm()
    qubits = m_machine.qAlloc_many(num_of_qubits)
    qpanda_cir = qgan_model.get_circuits_with_trained_param(qubits)
    print(qpanda_cir)

Em geral, o aprendizado e carregamento de distribuição aleatória do QGAN requer múltiplos modelos de treinamento com diferentes sementes aleatórias para obter os resultados esperados. Por exemplo, a seguir está o gráfico da função de distribuição de probabilidade entre a distribuição lognormal implementada pelo QGAN e a distribuição lognormal real, e a curva da função de perda entre o gerador e o discriminador do QGAN.

.. image:: ./images/qgan-loss.png
   :width: 600 px
   :align: center

|

.. image:: ./images/qgan-pdf.png
   :width: 600 px
   :align: center

|


SVM com kernel quântico
=================================

Em tarefas de aprendizado de máquina, os dados frequentemente não podem ser separados por um hiperplano no espaço original. Uma técnica comum para encontrar tais hiperplanos é aplicar uma função de transformação não linear aos dados.
Esta função é chamada de mapa de características, através do qual podemos calcular quão próximos os pontos de dados estão neste novo espaço de características para a tarefa de classificação do aprendizado de máquina.

Este exemplo refere-se à tese: `Supervised learning with quantum enhanced feature spaces <https://arxiv.org/pdf/1804.11326.pdf>`_ .
O primeiro método constrói circuitos variacionais para tarefas de classificação de dados.

``gen_vqc_qsvm_data`` são os dados necessários para gerar este exemplo. ``vqc_qsvm`` é uma classe de subcircuito variável usada para classificar os dados de entrada.
A função ``vqc_qsvm.plot()`` visualiza a distribuição dos dados.

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
        #vezes de repetição dos subcircuitos
        rep = 3

        #define a classe QSVM
        VQC_QSVM = vqc_qsvm(batch_size, maxiter, rep)
        #gera dados aleatoriamente a partir da tese.
        train_features, test_features, train_labels, test_labels, samples = \
            gen_vqc_qsvm_data(training_size=training_size, test_size=test_size, gap=gap)
        VQC_QSVM.plot(train_features, test_features, train_labels, test_labels, samples)
        #treina
        VQC_QSVM.train(train_features, train_labels)
        #testa
        rlt, acc_1 = VQC_QSVM.predict(test_features, test_labels)
        print(f"testing_accuracy {acc_1}")


Além do uso direto de circuitos quânticos variacionais mencionado acima para mapear características de dados clássicos para espaços de características quânticas, no artigo `Supervised learning with quantum enhanced feature spaces <https://arxiv.org/pdf/1804.11326.pdf>`_,
também é introduzido o método de estimar diretamente funções kernel usando circuitos quânticos e classificá-los usando máquinas de vetor de suporte clássicas.
Analogia a várias funções kernel no SVM clássico :math:`K(i,j)` , use a função kernel quântica para definir o produto interno de dados clássicos no espaço de características quânticas :math:`\phi(\mathbf{x}_i)` :

.. math::
    |\langle \phi(\mathbf{x}_j) | \phi(\mathbf{x}_i) \rangle |^2 =  |\langle 0 | U^\dagger(\mathbf{x}_j) U(\mathbf{x}_i) | 0 \rangle |^2

Usando VQNet e pyQPanda, definimos um ``QuantumKernel_VQNet`` para gerar uma função kernel quântica e usar ``sklearn`` para classificação:

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


Otimizador de Aproximação Estocástica por Perturbação Simultânea
=================================================================


.. py:function:: pyvqnet.qnn.SPSA(maxiter: int = 1000, save_steps: int = 1, last_avg: int = 1, c0: float = _C0, c1: float = 0.2, c2: float = 0.602, c3: float = 0.101, c4: float = 0, init_para=None, model=None, calibrate_flag=False)
    

    Otimizador de Aproximação Estocástica por Perturbação Simultânea (SPSA).

    O SPSA fornece um método estocástico para aproximar o gradiente de uma função de custo diferenciável multivariada.
    Para conseguir isso, a função de custo é avaliada duas vezes usando um vetor de parâmetros perturbado: cada componente do vetor de parâmetros original é deslocado simultaneamente por um valor gerado aleatoriamente.
    Mais informações estão disponíveis no `site do SPSA <http://www.jhuapl.edu/SPSA>`__.

    :param maxiter: O número máximo de iterações a serem realizadas. Valor padrão: 1000.
    :param save_steps: Salva as informações intermediárias de cada save_steps passos. Valor padrão: 1.
    :param last_avg: Parâmetro de média para as últimas last_avg iterações.
        Se last_avg = 1, apenas a última iteração é considerada. Valor padrão: 1.
    :param c0: a inicial. Tamanho do passo para atualizar parâmetros. Valor padrão: 0.2*pi
    :param c1: c inicial. O tamanho do passo usado para aproximar o gradiente. Padrão: 0.1.
    :param c2: alpha do artigo, usado para ajustar a(c0) a cada iteração. Valor padrão: 0.602.
    :param c3: gamma do artigo, usado para ajustar c(c1) a cada iteração. Valor padrão: 0.101.
    :param c4: Também usado para controlar os parâmetros de a. Valor padrão: 0.
    :param init_para: Parâmetros de inicialização. Padrão: None.
    :param model: Modelo paramétrico: model. Padrão: None.
    :param calibrate_flag: se deve calibrar os hiperparâmetros a e c, valor padrão: False.

    :return: uma instância do otimizador SPSA


    .. warning::

        SPSA suporta apenas parâmetros 1-dim.

    Exemplo::

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

    usa SPSA para otimizar dados de entrada.

    :param input_data: dados de entrada
    :return:

        train_para: parâmetro final

        theta_best: A média dos parâmetros após o último `last_avg`.

    Exemplo::

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

Matriz de informação de Fisher quântica
========================================================

.. py:class:: pyvqnet.qnn.opt.quantum_fisher(py_qpanda_config, params, target_gate_type_lists,target_gate_bits_lists, qcir_lists, wires)
    
    Retorna a matriz de informação de Fisher quântica para um circuito quântico.

    .. math::

        \mathrm{QFIM}_{i, j}=4 \operatorname{Re}\left[\left\langle\partial_i \psi(\boldsymbol{\theta}) \mid \partial_j \psi(\boldsymbol{\theta})\right\rangle-\left\langle\partial_i \psi(\boldsymbol{\theta}) \mid \psi(\boldsymbol{\theta})\right\rangle\left\langle\psi(\boldsymbol{\theta}) \mid \partial_j \psi(\boldsymbol{\theta})\right\rangle\right]

    A versão resumida é :math::math:`\left|\partial_j \psi(\boldsymbol{\theta})\right\rangle:=\frac{\partial}{\partial \theta_j}|\psi(\boldsymbol{\theta})\rangle`.

    .. note::

        Atualmente apenas RX, RY, RZ são suportados.

    :param params: Parâmetros variáveis nos circuitos.
    :param target_gate_type_lists: Suporta "RX", "RY", "RZ" ou listas.
    :param target_gate_bits_lists:  Em qual bit ou bits quânticos a porta parametrizada atua.
    :param qcir_lists: A lista de circuitos quânticos antes da porta parametrizada alvo para calcular o tensor métrico, veja o exemplo a seguir.
    :param wires: Índice total de bits quânticos para circuitos quânticos.

    Example::
    
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

        # O exemplo acima mostra que não há portas idênticas na mesma camada, 
        # mas na mesma camada você precisa modificar as portas lógicas de acordo com o exemplo a seguir.
        
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
            qcir.insert(pq.CNOT(config._qubits[0], config._qubits[1])) #  parte 01
            
            qcir.insert(pq.RZ(config._qubits[1], params[2]))  #  parte 02
            
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

        mt = quantum_fisher(config, params2, [['RX', 'RY'], ['RZ'], ['RZ']], # rx,ry conta como camada um, primeiro rz como camada dois, segundo rz como camada três.
                                [[0, 1], [1], [1]], qcir, [0, 1])

