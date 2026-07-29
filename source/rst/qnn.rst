API de Aprendizaje Automático Cuántico con QPanda2
####################################################


.. warning::

    La parte de computación cuántica de la siguiente interfaz utiliza pyQPanda2.

    Debido a problemas de compatibilidad entre pyQPanda2 y pyqpanda3, es necesario instalar pyqpanda2 manualmente: `pip install pyqpanda`

Capa de Computación Cuántica
***********************************

.. _QuantumLayer:

QuantumLayer
=================================

QuantumLayer es una clase envolvente del módulo autograd que admite circuitos cuánticos variacionales. Puede definir una función como argumento, por ejemplo ``qprog_with_measure``. Esta función debe contener el circuito cuántico definido por pyQPanda: generalmente incluye un circuito de codificación, un circuito de evolución y una operación de medición.
Esta clase QuantumLayer puede integrarse en un modelo híbrido clásico-cuántico de aprendizaje automático y minimizar la función objetivo o función de pérdida del modelo híbrido clásico-cuántico mediante el método clásico de descenso por gradiente.
Puede especificar el método de cálculo del gradiente de los parámetros del circuito cuántico en ``QuantumLayer`` cambiando el parámetro ``diff_method``. ``QuantumLayer`` admite actualmente dos métodos: ``finite_diff`` y ``parameter-shift``.

El método ``finite_diff`` es uno de los métodos numéricos más tradicionales y comunes para estimar el gradiente de una función. La idea principal consiste en reemplazar las derivadas parciales por diferencias:

.. math::

    f^{\prime}(x)=\lim _{h \rightarrow 0} \frac{f(x+h)-f(x)}{h}


Para el método ``parameter-shift`` utilizamos la función objetivo, como por ejemplo:

.. math:: O(\theta)=\left\langle 0\left|U^{\dagger}(\theta) H U(\theta)\right| 0\right\rangle

Teóricamente es posible calcular el gradiente de los parámetros con respecto al Hamiltoniano en un circuito cuántico mediante el método más preciso: ``parameter-shift``.

.. math::

    \nabla O(\theta)=
    \frac{1}{2}\left[O\left(\theta+\frac{\pi}{2}\right)-O\left(\theta-\frac{\pi}{2}\right)\right]

.. py:class:: pyvqnet.qnn.quantumlayer.QuantumLayer(qprog_with_measure,para_num,machine_type_or_cloud_token,num_of_qubits:int,num_of_cbits:int = 1,diff_method:str = "parameter_shift",delta:float = 0.01, dtype=None, name='')

    Módulo de cálculo abstracto para circuitos cuánticos variacionales. Simula un circuito cuántico parametrizado y obtiene el resultado de la medición.
    QuantumLayer hereda de Module, por lo que puede calcular gradientes de los parámetros del circuito y entrenar modelos de circuitos cuánticos variacionales o integrar circuitos cuánticos variacionales en modelos híbridos cuántico-clásicos.
    
    Esta clase no requiere que inicialice la máquina virtual en la función ``qprog_with_measure``.

    :param qprog_with_measure: funciones de circuitos cuánticos invocables, construidas con pyQPanda2
    :param para_num: `int` - Número de parámetros
    :param machine_type_or_cloud_token: tipo de máquina qpanda o token QCLOUD de pyQPanda2
    :param num_of_qubits: número de qubits
    :param num_of_cbits: número de bits clásicos
    :param diff_method: 'parameter_shift' o 'finite_diff'
    :param delta: delta para la diferenciación
    :param dtype: Tipo de dato del parámetro, valor predeterminado: None, usa el tipo de dato predeterminado kfloat32, que representa un número de punto flotante de 32 bits.
    :param name: nombre de la capa de salida

    :return: un módulo que puede calcular circuitos cuánticos.

    .. note::
        qprog_with_measure es una función de circuito cuántico definida en pyQPanda2.

        Esta función debe contener los siguientes parámetros, de lo contrario no podrá ejecutarse correctamente en QuantumLayer.

        qprog_with_measure (input,param,qubits,cbits,m_machine)

            `input`: array_like, datos clásicos unidimensionales de entrada

            `param`: array_like, parámetros unidimensionales del circuito cuántico de entrada

            `qubits`: qubits asignados por QuantumLayer

            `cbits`: cbits asignados por QuantumLayer. Si su circuito no utiliza cbits, debe reservar este parámetro de todas formas.

            `m_machine`: simulador creado por QuantumLayer

        Use el atributo ``m_para`` de QuantumLayer para obtener los parámetros de entrenamiento del circuito cuántico variable. El parámetro es de la clase ``QTensor``, que puede convertirse en un array de numpy mediante la interfaz ``to_numpy()``.

    .. note::

        La clase tiene un alias: `QpandaQCircuitVQCLayer` .

    Example::

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
        #datos clásicos como entrada
        input = QTensor([[1,2,3,4],[40,22,2,3],[33,3,25,2.0]] )
        #circuito hacia adelante
        rlt = pqc(input)
        grad =  QTensor(np.ones(rlt.data.shape)*1000)
        #circuito hacia atrás
        rlt.backward(grad)
        print(rlt)
        # [
        # [0.2500000, 0.2500000, 0.2500000, 0.2500000],
        # [0.2500000, 0.2500000, 0.2500000, 0.2500000],
        # [0.2500000, 0.2500000, 0.2500000, 0.2500000]
        # ]

QuantumLayerV2
================================

Si está más familiarizado con la sintaxis de pyQPanda2, puede usar la clase QuantumLayerV2. Puede definir la función de circuito cuántico usando ``qubits``, ``cbits`` y ``machine``, y luego pasarla como argumento ``qprog_with_measure`` de QuantumLayerV2.

.. py:class:: pyvqnet.qnn.quantumlayer.QuantumLayerV2(qprog_with_measure, para_num, diff_method: str = 'parameter_shift', delta: float = 0.01, dtype=None, name='')

    Módulo de cálculo abstracto para circuitos cuánticos variacionales. Simula un circuito cuántico parametrizado y obtiene el resultado de la medición.
    QuantumLayer hereda de Module, por lo que puede calcular gradientes de los parámetros del circuito y entrenar modelos de circuitos cuánticos variacionales o integrar circuitos cuánticos variacionales en modelos híbridos cuántico-clásicos.
    
    Para usar este módulo, debe crear su propia máquina virtual cuántica y asignar qubits y cbits.

    :param qprog_with_measure: funciones de circuitos cuánticos invocables, construidas con pyQPanda2
    :param para_num: `int` - Número de parámetros
    :param diff_method: 'parameter_shift' o 'finite_diff'
    :param delta: delta para la diferenciación
    :param dtype: Tipo de dato del parámetro, valor predeterminado: None, usa el tipo de dato predeterminado kfloat32, que representa un número de punto flotante de 32 bits.
    :param name: nombre de la capa de salida
    :return: un módulo que puede calcular circuitos cuánticos.

    .. note::
        qprog_with_measure es una función de circuito cuántico definida en pyQPanda.

        Esta función debe contener los siguientes parámetros, de lo contrario no podrá ejecutarse correctamente en QuantumLayerV2.

        En comparación con QuantumLayer, debe asignar qubits y el simulador usted mismo,

        y también puede necesitar asignar cbits si qprog_with_measure requiere medición cuántica.

        qprog_with_measure (input,param)

        `input`: array_like, datos clásicos unidimensionales de entrada

        `param`: array_like, parámetros unidimensionales del circuito cuántico de entrada

    .. note::

        La clase tiene un alias: `QpandaQCircuitVQCLayerLite` .

    Example::

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

        #datos clásicos como entrada
        input = QTensor([[1,2,3,4],[4,2,2,3],[3,3,2,2.0]] )

        #circuito hacia adelante
        rlt = pqc(input)

        grad =  QTensor(np.ones(rlt.data.shape)*1000)
        #circuito hacia atrás
        rlt.backward(grad)
        print(rlt)

        # [
        # [0.2500000, 0.2500000, 0.2500000, 0.2500000],
        # [0.2500000, 0.2500000, 0.2500000, 0.2500000],
        # [0.2500000, 0.2500000, 0.2500000, 0.2500000]
        # ]

 


NoiseQuantumLayer
================================

En un ordenador cuántico real, debido a las características físicas del bit cuántico, siempre existe un error de cálculo inevitable. Para simular mejor este error en una máquina virtual cuántica, VQNet también admite una máquina virtual cuántica con ruido. La simulación con una máquina virtual cuántica con ruido se aproxima más a un ordenador cuántico real. Podemos personalizar el tipo de puerta lógica admitida y el modelo de ruido asociado a cada puerta lógica.
El modelo de ruido cuántico actualmente compatible está definido en ``NoiseQVM`` de pyQPanda2.

Podemos usar ``NoiseQuantumLayer`` para definir una capa de diferenciación automática de circuitos cuánticos. ``NoiseQuantumLayer`` admite la máquina virtual cuántica con ruido de pyQPanda2. Puede definir una función como argumento ``qprog_with_measure``. Esta función debe contener el circuito cuántico definido por pyQPanda, y también debe pasar un argumento ``noise_set_config``, utilizando la interfaz de pyQPanda para configurar el modelo de ruido.

.. py:class:: pyvqnet.qnn.quantumlayer.NoiseQuantumLayer(qprog_with_measure, para_num, machine_type, num_of_qubits: int, num_of_cbits: int = 1, diff_method: str = 'parameter_shift', delta: float = 0.01, noise_set_config=None, dtype=None, name='')

    Módulo de cálculo abstracto para circuitos cuánticos variacionales. Simula un circuito cuántico parametrizado y obtiene el resultado de la medición.
    QuantumLayer hereda de Module, por lo que puede calcular gradientes de los parámetros del circuito y entrenar modelos de circuitos cuánticos variacionales o integrar circuitos cuánticos variacionales en modelos híbridos cuántico-clásicos.
    
    Este módulo debe inicializarse con un modelo de ruido mediante ``noise_set_config``.

    :param qprog_with_measure: funciones de circuitos cuánticos invocables, construidas con pyQPanda2
    :param para_num: `int` - Número de parámetros
    :param machine_type: tipo de máquina pyQPanda2
    :param num_of_qubits: número de qubits
    :param num_of_cbits: número de cbits
    :param diff_method: 'parameter_shift' o 'finite_diff'
    :param delta: delta para la diferenciación
    :param noise_set_config: función de configuración de ruido
    :param dtype: Tipo de dato del parámetro, valor predeterminado: None, usa el tipo de dato predeterminado kfloat32, que representa un número de punto flotante de 32 bits.
    :param name: nombre de la capa de salida
    
    :return: un módulo que puede calcular circuitos cuánticos con modelo de ruido.

    .. note::
        qprog_with_measure es una función de circuito cuántico definida en pyQPanda.

        Esta función debe contener los siguientes parámetros, de lo contrario no podrá ejecutarse correctamente en NoiseQuantumLayer.

        qprog_with_measure (input,param,qubits,cbits,m_machine)

        `input`: array_like, datos clásicos unidimensionales de entrada

        `param`: array_like, parámetros unidimensionales del circuito cuántico de entrada

        `qubits`: qubits asignados por NoiseQuantumLayer

        `cbits`: cbits asignados por NoiseQuantumLayer. Si su circuito no utiliza cbits, debe reservar este parámetro de todas formas.

        `m_machine`: simulador creado por NoiseQuantumLayer

    Example::

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
            # Calcular probabilidades para cada estado
            probabilities = counts / 100
            # Obtener el valor esperado del estado
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

A continuación se muestra un ejemplo de ``noise_set_config``, donde añadimos el modelo de ruido BITFLIP_KRAUS_OPERATOR con el argumento de ruido p=0.01 a las puertas cuánticas ``RX``, ``RY``, ``RZ``, ``X``, ``Y``, ``Z``, ``H``, etc.

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
================================

.. py:class:: pyvqnet.qnn.QiskitLayer(qiskit_circuits,para_num)

    Una capa envolvente para implementar la propagación hacia adelante y hacia atrás con circuitos de Qiskit en VQNet. QISKIT_VQC es una clase que define un circuito cuántico de Qiskit y su función de ejecución.
    El siguiente ejemplo demuestra su funcionamiento. Esta capa solo admite entradas de circuito y pesos como parámetros.
    
    :param cirq_vqc: Una clase que define la definición, el backend y las funciones de ejecución de un circuito de Qiskit.
    :param para_num: `int` - El número de parámetros.
    :return: Una clase capaz de ejecutar modelos de circuitos cuánticos de Qiskit.

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
                # --- Definición del circuito ---

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

        #definir clase de circuitos qiskit
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
================================

.. py:class:: pyvqnet.qnn.CirqLayer(cirq_vqc,para_num)

    Una capa envolvente para circuitos Cirq que implementa la propagación hacia adelante y hacia atrás en VQNet. CIRQ_VQC es una clase que requiere que el usuario defina un circuito cuántico de Cirq y su función `run`. El siguiente ejemplo demuestra su principio de funcionamiento.
    Esta capa solo admite entradas de circuito y pesos como parámetros.

    :param cirq_vqc: Una clase que define la definición, el backend y las funciones de ejecución de un circuito de Cirq.
    :param para_num: `int` - El número de parámetros.
    :return: Una clase capaz de ejecutar el modelo de circuito cuántico de Cirq.


    .. note::

        El código de ejemplo siguiente requiere `cirq==1.5.0, numpy <2`.

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
                ###definir qubits
                q0 = cirq.NamedQubit ('q0')
                q1 = cirq.NamedQubit ('q1')
                q2 = cirq.NamedQubit ('q2')
                q3 = cirq.NamedQubit ('q3')
                qubits = [q0,q1,q2,q3]
                self.qubits = [q0,q1,q2,q3]
                ###definir parámetros variacionales
                param = sympy.symbols(f'theta(0:24)')
                self.theta = np.asarray(param).reshape((4,6))

                ###definir circuitos
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
                
                ###definir backend
                self._backend = simulator

                self._param_symbols_list,self._input_symbols_list = get_circuit_symbols(self._circuit)


            def run(self,resolver,init_state):

                rlt = self._backend.simulate(self._circuit,resolver,initial_state=init_state).final_state_vector
                z0 = cirq.Z(self.qubits[0])

                qubit_map={self.qubits[0]: 0}
                
                expectation = z0.expectation_from_state_vector(rlt, qubit_map).real

                return expectation

        #definir clase de circuitos cirq
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

Puertas Cuánticas
***********************************

La forma de manipular los qubits se denomina puertas cuánticas. Mediante las puertas cuánticas, evolucionamos conscientemente los estados cuánticos. Las puertas cuánticas son la base de los algoritmos cuánticos.

Puertas cuánticas básicas
================================

En VQNet, utilizamos las puertas lógicas de pyQPanda desarrolladas por Origin Quantum para construir circuitos cuánticos y realizar simulaciones cuánticas.
Las puertas actualmente compatibles con pyQPanda pueden definirse en la sección de puertas cuánticas de pyQPanda.
Además, VQNet también encapsula algunas combinaciones de puertas cuánticas comúnmente utilizadas en aprendizaje automático cuántico.



AmplitudeEmbeddingCircuit
================================

.. py:function:: pyvqnet.qnn.template.AmplitudeEmbeddingCircuit(input_feat, qubits)

    Codifica :math:`2^n` características en el vector de amplitudes de :math:`n` qubits.
    Para representar un vector de estado cuántico válido, la norma L2 de ``features`` debe ser uno.

    :param input_feat: array numpy que representa los parámetros
    :param qubits: qubits asignados por pyQPanda
    :return: circuitos cuánticos

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




APIs de Aprendizaje Automático Cuántico con pyQPanda2
***************************************************************************

Redes Generativas Antagónicas Cuánticas para aprender y cargar distribuciones aleatorias
==================================================================================================

El algoritmo de Redes Generativas Antagónicas Cuánticas (`QGAN <https://www.nature.com/articles/s41534-019-0223-2>`_ ) utiliza circuitos cuánticos puramente variacionales para preparar estados cuánticos generados con una distribución aleatoria específica, lo que puede reducir las puertas lógicas necesarias para generar estados cuánticos específicos y disminuir la complejidad de los circuitos cuánticos. Utiliza la estructura clásica del modelo GAN, que tiene dos submodelos: Generador y Discriminador. El Generador produce una distribución específica para el circuito cuántico. Y el Discriminador discrimina entre las muestras de datos generadas por el Generador y las muestras de datos de entrenamiento de distribución aleatoria reales.
A continuación se muestra un ejemplo de VQNet implementando QGAN para aprender y cargar distribuciones aleatorias basado en el artículo `Quantum Generative Adversarial Networks for learning and loading random distributions <https://www.nature.com/articles/s41534-019-0223-2>`_ de Christa Zoufal.

.. image:: ./images/qgan-arch.PNG
   :width: 600 px
   :align: center

|

Con el fin de realizar la construcción de la clase ``QGANAPI`` de la red generativa antagónica cuántica mediante VQNet, el generador cuántico se utiliza para preparar el estado inicial de los datos con distribución real. El número de bits cuánticos es 3, y el número de repeticiones del módulo de circuito paramétrico interno del generador cuántico es 1. Además, se utiliza KL como métrica para la carga de la distribución aleatoria en QGAN.

.. code-block::

    import pickle
    import os
    import pyqpanda as pq
    from pyvqnet.qnn.qgan.qgan_utils import QGANAPI
    import numpy as np

    num_of_qubits = 3  # configuración del artículo
    rep = 1
    number_of_data = 10000
    # Cargar muestras de datos de diferentes distribuciones
    mu = 1
    sigma = 1
    real_data = np.random.lognormal(mean=mu, sigma=sigma, size=number_of_data)


    # inicial
    save_dir = None
    qgan_model = QGANAPI(
        real_data,
        # distribución de datos generada por numpy, 1 - dim.
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

A continuación se muestra el módulo ``train`` de QGAN.

.. code-block::

    # entrenamiento
    qgan_model.train()  # entrenar qgan


El módulo ``eval`` de QGAN está diseñado para dibujar la curva de la función de pérdida y el diagrama de distribución de probabilidad entre la distribución aleatoria preparada por QGAN y la distribución real.

.. code-block::

    # mostrar la función de distribución de probabilidad de la distribución generada y la real
    qgan_model.eval(real_data)  #dibujar pdf

El módulo ``get_trained_quantum_parameters`` de QGAN se utiliza para obtener los parámetros de entrenamiento y mostrarlos como un array numpy. Si ``save_DIR`` no está vacío, los parámetros de entrenamiento se guardan en un archivo. El módulo ``Load_param_and_eval`` de QGAN carga los parámetros de entrenamiento, y el módulo ``get_circuits_with_trained_param`` obtiene el circuito pyQPanda generado por el generador cuántico después del entrenamiento.

.. code-block::

    # obtener parámetros cuánticos entrenados
    param = qgan_model.get_trained_quantum_parameters()
    print(f" trained param {param}")

    #cargar archivos de parámetros guardados
    if save_dir is not None:
        path = os.path.join(
            save_dir, qgan_model._start_time + "trained_qgan_param.pickle")
        with open(path, "rb") as file:
            t3 = pickle.load(file)
        param = t3["quantum_parameters"]
        print(f" trained param {param}")

    #mostrar la función de distribución de probabilidad de la distribución generada y la real
    qgan_model.load_param_and_eval(param)

    #calcular métrica
    print(qgan_model.eval_metric(param, "kl"))

    #obtener circuito cuántico del generador
    m_machine = pq.CPUQVM()
    m_machine.init_qvm()
    qubits = m_machine.qAlloc_many(num_of_qubits)
    qpanda_cir = qgan_model.get_circuits_with_trained_param(qubits)
    print(qpanda_cir)

En general, el aprendizaje y la carga de distribuciones aleatorias con QGAN requiere entrenar múltiples modelos con diferentes semillas aleatorias para obtener los resultados esperados. Por ejemplo, a continuación se muestra el gráfico de la función de distribución de probabilidad entre la distribución lognormal implementada por QGAN y la distribución lognormal real, y la curva de la función de pérdida entre el generador y el discriminador de QGAN.

.. image:: ./images/qgan-loss.png
   :width: 600 px
   :align: center

|

.. image:: ./images/qgan-pdf.png
   :width: 600 px
   :align: center

|


Máquina de vectores de soporte con kernel cuántico
==================================================

En tareas de aprendizaje automático, los datos a menudo no pueden separarse mediante un hiperplano en el espacio original. Una técnica común para encontrar dichos hiperplanos es aplicar una función de transformación no lineal a los datos.
Esta función se denomina mapa de características, a través del cual podemos calcular qué tan cercanos están los puntos de datos en este nuevo espacio de características para la tarea de clasificación del aprendizaje automático.

Este ejemplo se basa en la tesis: `Supervised learning with quantum enhanced feature spaces <https://arxiv.org/pdf/1804.11326.pdf>`_ .
El primer método construye circuitos variacionales para tareas de clasificación de datos.

``gen_vqc_qsvm_data`` son los datos necesarios para generar este ejemplo. ``vqc_qsvm`` es una clase de subcircuito variable utilizada para clasificar los datos de entrada.
La función ``vqc_qsvm.plot()`` visualiza la distribución de los datos.

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
        #veces de repetición de subcircuitos
        rep = 3

        #define la clase QSVM
        VQC_QSVM = vqc_qsvm(batch_size, maxiter, rep)
        #genera datos aleatorios de la tesis.
        train_features, test_features, train_labels, test_labels, samples = \
            gen_vqc_qsvm_data(training_size=training_size, test_size=test_size, gap=gap)
        VQC_QSVM.plot(train_features, test_features, train_labels, test_labels, samples)
        #entrenar
        VQC_QSVM.train(train_features, train_labels)
        #prueba
        rlt, acc_1 = VQC_QSVM.predict(test_features, test_labels)
        print(f"testing_accuracy {acc_1}")


Además del uso mencionado anteriormente de circuitos cuánticos variacionales para mapear características de datos clásicos a espacios de características cuánticos, en el artículo `Supervised learning with quantum enhanced feature spaces <https://arxiv.org/pdf/1804.11326.pdf>`_,
también se introduce el método de estimar directamente funciones kernel utilizando circuitos cuánticos y clasificarlos mediante máquinas de vectores de soporte clásicas.
De forma análoga a varias funciones kernel en SVM clásica :math:`K(i,j)`, se utiliza una función kernel cuántica para definir el producto interno de datos clásicos en el espacio de características cuántico :math:`\phi(\mathbf{x}_i)` :

.. math::
    |\langle \phi(\mathbf{x}_j) | \phi(\mathbf{x}_i) \rangle |^2 =  |\langle 0 | U^\dagger(\mathbf{x}_j) U(\mathbf{x}_i) | 0 \rangle |^2

Usando VQNet y pyQPanda, definimos un ``QuantumKernel_VQNet`` para generar una función kernel cuántica y utilizamos ``sklearn`` para la clasificación:

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


Optimizadores por Aproximación Estocástica de Perturbación Simultánea
=====================================================================


.. py:function:: pyvqnet.qnn.SPSA(maxiter: int = 1000, save_steps: int = 1, last_avg: int = 1, c0: float = _C0, c1: float = 0.2, c2: float = 0.602, c3: float = 0.101, c4: float = 0, init_para=None, model=None, calibrate_flag=False)
    

    Optimizador por Aproximación Estocástica de Perturbación Simultánea (SPSA).

    SPSA proporciona un método estocástico para aproximar el gradiente de una función de coste diferenciable multivariante.
    Para lograrlo, la función de coste se evalúa dos veces utilizando un vector de parámetros perturbado: cada componente del vector de parámetros original se desplaza simultáneamente por un valor generado aleatoriamente.
    Más información disponible en el `sitio web de SPSA <http://www.jhuapl.edu/SPSA>`__.

    :param maxiter: Número máximo de iteraciones a realizar. Valor predeterminado: 1000.
    :param save_steps: Guardar la información intermedia de cada paso save_steps. Valor predeterminado: 1.
    :param last_avg: Parámetro de promediado para las últimas last_avg iteraciones.
        Si last_avg = 1, solo se considera la última iteración. Valor predeterminado: 1.
    :param c0: a inicial. Tamaño de paso para actualizar parámetros. Valor predeterminado: 0.2*pi
    :param c1: c inicial. Tamaño de paso utilizado para aproximar el gradiente. Valor predeterminado: 0.1.
    :param c2: alpha del artículo, utilizado para ajustar a(c0) en cada iteración. Valor predeterminado: 0.602.
    :param c3: gamma del artículo, utilizado para ajustar c(c1) en cada iteración. Valor predeterminado: 0.101.
    :param c4: También se utiliza para controlar los parámetros de a. Valor predeterminado: 0.
    :param init_para: Parámetros de inicialización. Valor predeterminado: None.
    :param model: Modelo paramétrico: model. Valor predeterminado: None.
    :param calibrate_flag: si se deben calibrar los hiperparámetros a y c. Valor predeterminado: False.

    :return: una instancia del optimizador SPSA


    .. warning::

        SPSA solo admite parámetros unidimensionales.

    Example::

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

    utiliza SPSA para optimizar los datos de entrada.

    :param input_data: datos de entrada
    :return:

        train_para: parámetro final

        theta_best: Los parámetros promedio después de las últimas `last_avg` iteraciones.

    Example::

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

Matriz de información de Fisher cuántica
========================================================

.. py:class:: pyvqnet.qnn.opt.quantum_fisher(py_qpanda_config, params, target_gate_type_lists,target_gate_bits_lists, qcir_lists, wires)
    
    Devuelve la matriz de información de Fisher cuántica para un circuito cuántico.

    .. math::

        \mathrm{QFIM}_{i, j}=4 \operatorname{Re}\left[\left\langle\partial_i \psi(\boldsymbol{\theta}) \mid \partial_j \psi(\boldsymbol{\theta})\right\rangle-\left\langle\partial_i \psi(\boldsymbol{\theta}) \mid \psi(\boldsymbol{\theta})\right\rangle\left\langle\psi(\boldsymbol{\theta}) \mid \partial_j \psi(\boldsymbol{\theta})\right\rangle\right]

    La versión corta es :math::math:`\left|\partial_j \psi(\boldsymbol{\theta})\right\rangle:=\frac{\partial}{\partial \theta_j}|\psi(\boldsymbol{\theta})\rangle`.

    .. note::

        Actualmente solo se admiten RX, RY, RZ.

    :param params: Parámetros variables en los circuitos.
    :param target_gate_type_lists: Soporta "RX", "RY", "RZ" o listas.
    :param target_gate_bits_lists: El bit o bits cuánticos sobre los que actúa la puerta parametrizada.
    :param qcir_lists: La lista de circuitos cuánticos antes de la puerta parametrizada objetivo para calcular el tensor métrico, consulte el siguiente ejemplo.
    :param wires: Índice total de bits cuánticos para los circuitos cuánticos.

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

        # El ejemplo anterior muestra que no hay puertas idénticas en la misma capa,
        # pero en la misma capa debe modificar las puertas lógicas según el siguiente ejemplo.
        
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

        mt = quantum_fisher(config, params2, [['RX', 'RY'], ['RZ'], ['RZ']], # rx,ry cuenta como capa uno, primer rz como capa dos, segundo rz como capa tres.
                                [[0, 1], [1], [1]], qcir, [0, 1])

