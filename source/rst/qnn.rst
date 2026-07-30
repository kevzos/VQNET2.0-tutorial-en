Quantum Machine Learning API mit QPanda2
####################################################


.. warning::

    Der Quantencomputing-Teil der folgenden Schnittstelle verwendet pyQPanda2.

    Aufgrund von Kompatibilitätsproblemen zwischen pyQPanda2 und pyqpanda3 müssen Sie pyqpanda2 selbst installieren: `pip install pyqpanda`

Quantencomputing-Schicht
***********************************

.. _QuantumLayer:

QuantumLayer
=================================

QuantumLayer ist eine Paketklasse des Autograd-Moduls, die variationelle Quantenschaltungen unterstützt. Sie können eine Funktion als Argument definieren, z.B. ``qprog_with_measure``. Diese Funktion muss die von pyQPanda definierte Quantenschaltung enthalten: Sie enthält in der Regel eine Codier-Schaltung, eine Evolutions-Schaltung und eine Mess-Operation.
Diese QuantumLayer-Klasse kann in das hybride quanten-klassische maschinelle Lernmodell eingebettet werden und die Zielfunktion oder Verlustfunktion des hybriden quanten-klassischen Modells mittels der klassischen Gradientenabstiegsmethode minimieren.
Sie können die Gradientenberechnungsmethode der Quantenschaltungsparameter in ``QuantumLayer`` durch Ändern des Parameters ``diff_method`` festlegen. ``QuantumLayer`` unterstützt derzeit zwei Methoden: ``finite_diff`` und ``parameter-shift``.

Die ``finite_diff``-Methode ist eine der traditionellsten und gebräuchlichsten numerischen Methoden zur Schätzung des Funktionsgradienten. Die Grundidee besteht darin, partielle Ableitungen durch Differenzen zu ersetzen:

.. math::

    f^{\prime}(x)=\lim _{h \rightarrow 0} \frac{f(x+h)-f(x)}{h}


Für die ``parameter-shift``-Methode verwenden wir die Zielfunktion, wie zum Beispiel:

.. math:: O(\theta)=\left\langle 0\left|U^{\dagger}(\theta) H U(\theta)\right| 0\right\rangle

Theoretisch ist es möglich, den Gradienten der Parameter bezüglich des Hamiltonians in einer Quantenschaltung mit der präziseren Methode ``parameter-shift`` zu berechnen.

.. math::

    \nabla O(\theta)=
    \frac{1}{2}\left[O\left(\theta+\frac{\pi}{2}\right)-O\left(\theta-\frac{\pi}{2}\right)\right]

.. py:class:: pyvqnet.qnn.quantumlayer.QuantumLayer(qprog_with_measure,para_num,machine_type_or_cloud_token,num_of_qubits:int,num_of_cbits:int = 1,diff_method:str = "parameter_shift",delta:float = 0.01, dtype=None, name='')

    Abstraktes Berechnungsmodul für variationelle Quantenschaltungen. Es simuliert eine parametrisierte Quantenschaltung und liefert das Messergebnis.
    QuantumLayer erbt von Module, sodass es Gradienten der Schaltungsparameter berechnen, variationelle Quantenschaltungsmodelle trainieren oder variationelle Quantenschaltungen in hybride quanten-klassische Modelle einbetten kann.
    
    Diese Klasse erfordert keine Initialisierung der virtuellen Maschine in der ``qprog_with_measure``-Funktion.

    :param qprog_with_measure: aufrufbare Quantenschaltungsfunktionen, erstellt mit pyQPanda2
    :param para_num: `int` - Anzahl der Parameter
    :param machine_type_or_cloud_token: QPanda-Maschinentyp oder pyQPanda2 QCLOUD-Token
    :param num_of_qubits: Anzahl der Qubits
    :param num_of_cbits: Anzahl der klassischen Bits
    :param diff_method: 'parameter_shift' oder 'finite_diff'
    :param delta: Delta für die Differenz
    :param dtype: Der Datentyp des Parameters, Standardwert: None, verwendet den Standard-Datentyp kfloat32, der eine 32-Bit-Gleitkommazahl darstellt.
    :param name: Name der Ausgabeschicht

    :return: ein Modul, das Quantenschaltungen berechnen kann.

    .. note::
        qprog_with_measure ist eine Quantenschaltungsfunktion, die in pyQPanda2 definiert ist.

        Diese Funktion sollte die folgenden Parameter enthalten, sonst kann sie in QuantumLayer nicht richtig ausgeführt werden.

        qprog_with_measure (input,param,qubits,cbits,m_machine)

            `input`: array_like Eingabe 1-dim klassische Daten

            `param`: array_like Eingabe 1-dim Parameter der Quantenschaltung

            `qubits`: von QuantumLayer zugewiesene Qubits

            `cbits`: von QuantumLayer zugewiesene klassische Bits. Falls Ihre Schaltung keine cbits verwendet, sollten Sie diesen Parameter dennoch reservieren.

            `m_machine`: von QuantumLayer erstellter Simulator

        Verwenden Sie das Attribut ``m_para`` von QuantumLayer, um die Trainingsparameter der variablen Quantenschaltung zu erhalten. Der Parameter ist eine ``QTensor``-Klasse, die mit der ``to_numpy()``-Schnittstelle in ein numpy-Array konvertiert werden kann.

    .. note::

        Die Klasse hat den Alias: `QpandaQCircuitVQCLayer` .

    Beispiel::

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
        #klassische Daten als Eingabe
        input = QTensor([[1,2,3,4],[40,22,2,3],[33,3,25,2.0]] )
        #Vorwärtsschaltungen
        rlt = pqc(input)
        grad =  QTensor(np.ones(rlt.data.shape)*1000)
        #Rückwärtsschaltungen
        rlt.backward(grad)
        print(rlt)
        # [
        # [0.2500000, 0.2500000, 0.2500000, 0.2500000],
        # [0.2500000, 0.2500000, 0.2500000, 0.2500000],
        # [0.2500000, 0.2500000, 0.2500000, 0.2500000]
        # ]

QuantumLayerV2
=================================

Wenn Sie mit der pyQPanda2-Syntax vertrauter sind, verwenden Sie bitte die QuantumLayerV2-Klasse. Sie können die Quantenschaltungsfunktion mit ``qubits``, ``cbits`` und ``machine`` definieren und diese dann als Argument ``qprog_with_measure`` von QuantumLayerV2 übergeben.

.. py:class:: pyvqnet.qnn.quantumlayer.QuantumLayerV2(qprog_with_measure, para_num, diff_method: str = 'parameter_shift', delta: float = 0.01, dtype=None, name='')

    Abstraktes Berechnungsmodul für variationelle Quantenschaltungen. Es simuliert eine parametrisierte Quantenschaltung und liefert das Messergebnis.
    QuantumLayer erbt von Module, sodass es Gradienten der Schaltungsparameter berechnen, variationelle Quantenschaltungsmodelle trainieren oder variationelle Quantenschaltungen in hybride quanten-klassische Modelle einbetten kann.
    
    Um dieses Modul zu verwenden, müssen Sie Ihre eigene Quanten-Virtuellmaschine erstellen und Qubits sowie klassische Bits zuweisen.

    :param qprog_with_measure: aufrufbare Quantenschaltungsfunktionen, erstellt mit pyQPanda2
    :param para_num: `int` - Anzahl der Parameter
    :param diff_method: 'parameter_shift' oder 'finite_diff'
    :param delta: Delta für die Differenz
    :param dtype: Der Datentyp des Parameters, Standardwert: None, verwendet den Standard-Datentyp kfloat32, der eine 32-Bit-Gleitkommazahl darstellt.
    :param name: Name der Ausgabeschicht
    :return: ein Modul, das Quantenschaltungen berechnen kann.

    .. note::
        qprog_with_measure ist eine Quantenschaltungsfunktion, die in pyQPanda definiert ist.

        Diese Funktion sollte die folgenden Parameter enthalten, sonst kann sie in QuantumLayerV2 nicht richtig ausgeführt werden.

        Im Vergleich zu QuantumLayer müssen Sie Qubits und Simulator selbst zuweisen,

        und Sie müssen möglicherweise auch cbits zuweisen, wenn qprog_with_measure eine Quantenmessung benötigt.

        qprog_with_measure (input,param)

        `input`: array_like Eingabe 1-dim klassische Daten

        `param`: array_like Eingabe 1-dim Parameter der Quantenschaltung

    .. note::

        Die Klasse hat den Alias: `QpandaQCircuitVQCLayerLite` .

    Beispiel::

        import pyqpanda as pq
        from pyvqnet.qnn.measure import ProbsMeasure
        from pyvqnet.qnn.quantumlayer import QuantumLayerV2
        import numpy as np
        from pyvqnet.tensor import QTensor
        def pqctest (input,param):
            num_of_qubits = 4

            m_machine = pq.CPUQVM()# außerhalb
            m_machine.init_qvm()# außerhalb
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

        #klassische Daten als Eingabe
        input = QTensor([[1,2,3,4],[4,2,2,3],[3,3,2,2.0]] )

        #Vorwärtsschaltungen
        rlt = pqc(input)

        grad =  QTensor(np.ones(rlt.data.shape)*1000)
        #Rückwärtsschaltungen
        rlt.backward(grad)
        print(rlt)

        # [
        # [0.2500000, 0.2500000, 0.2500000, 0.2500000],
        # [0.2500000, 0.2500000, 0.2500000, 0.2500000],
        # [0.2500000, 0.2500000, 0.2500000, 0.2500000]
        # ]

 



NoiseQuantumLayer
=================================

In einem echten Quantencomputer treten aufgrund der physikalischen Eigenschaften des Qubits stets unvermeidliche Berechnungsfehler auf. Um diesen Fehler in der Quanten-Virtuellmaschine besser zu simulieren, unterstützt VQNet auch Quanten-Virtuellmaschinen mit Rauschen. Die Simulation mit Rauschen kommt dem echten Quantencomputer näher. Wir können den unterstützten Logikgattertyp und das vom Logikgatter unterstützte Rauschmodell anpassen.
Das derzeit unterstützte Quantenrauschmodell ist in pyQPanda2s ``NoiseQVM`` definiert.

Wir können ``NoiseQuantumLayer`` verwenden, um eine automatische Mikroklassifikation von Quantenschaltungen zu definieren. ``NoiseQuantumLayer`` unterstützt pyQPanda2-Quanten-Virtuellmaschinen mit Rauschen. Sie können eine Funktion als Argument ``qprog_with_measure`` definieren. Diese Funktion muss die von pyQPanda definierte Quantenschaltung enthalten, und Sie müssen auch ein Argument ``noise_set_config`` übergeben, um das Rauschmodell über die pyQPanda-Schnittstelle einzustellen.

.. py:class:: pyvqnet.qnn.quantumlayer.NoiseQuantumLayer(qprog_with_measure, para_num, machine_type, num_of_qubits: int, num_of_cbits: int = 1, diff_method: str = 'parameter_shift', delta: float = 0.01, noise_set_config=None, dtype=None, name='')

    Abstraktes Berechnungsmodul für variationelle Quantenschaltungen. Es simuliert eine parametrisierte Quantenschaltung und liefert das Messergebnis.
    QuantumLayer erbt von Module, sodass es Gradienten der Schaltungsparameter berechnen, variationelle Quantenschaltungsmodelle trainieren oder variationelle Quantenschaltungen in hybride quanten-klassische Modelle einbetten kann.
    
    Dieses Modul sollte mit einem Rauschmodell über ``noise_set_config`` initialisiert werden.

    :param qprog_with_measure: aufrufbare Quantenschaltungsfunktionen, erstellt mit pyQPanda2
    :param para_num: `int` - Anzahl der Parameter
    :param machine_type: pyQPanda2 Maschinentyp
    :param num_of_qubits: Anzahl der Qubits
    :param num_of_cbits: Anzahl der klassischen Bits
    :param diff_method: 'parameter_shift' oder 'finite_diff'
    :param delta: Delta für die Differenz
    :param noise_set_config: Rauschkonfigurationsfunktion
    :param dtype: Der Datentyp des Parameters, Standardwert: None, verwendet den Standard-Datentyp kfloat32, der eine 32-Bit-Gleitkommazahl darstellt.
    :param name: Name der Ausgabeschicht
    
    :return: ein Modul, das Quantenschaltungen mit Rauschmodell berechnen kann.

    .. note::
        qprog_with_measure ist eine Quantenschaltungsfunktion, die in pyQPanda definiert ist.

        Diese Funktion sollte die folgenden Parameter enthalten, sonst kann sie in NoiseQuantumLayer nicht richtig ausgeführt werden.

        qprog_with_measure (input,param,qubits,cbits,m_machine)

        `input`: array_like Eingabe 1-dim klassische Daten

        `param`: array_like Eingabe 1-dim Parameter der Quantenschaltung

        `qubits`: von NoiseQuantumLayer zugewiesene Qubits

        `cbits`: von NoiseQuantumLayer zugewiesene klassische Bits. Falls Ihre Schaltung keine cbits verwendet, sollten Sie diesen Parameter dennoch reservieren.

        `m_machine`: von NoiseQuantumLayer erstellter Simulator

    Beispiel::

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
            # Berechne Wahrscheinlichkeiten für jeden Zustand
            probabilities = counts / 100
            # Berechne Zustandserwartungswert
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

Hier ist ein Beispiel für ``noise_set_config``. Hier fügen wir das Rauschmodell BITFLIP_KRAUS_OPERATOR mit dem Rauschparameter p=0.01 zu den Quantengattern ``RX``, ``RY``, ``RZ``, ``X``, ``Y``, ``Z``, ``H`` usw. hinzu.

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

    Eine Wrapper-Schicht zur Implementierung von Vorwärts- und Rückwärtspropagation mit Qiskit-Schaltungen in VQNet. QISKIT_VQC ist eine Klasse, die eine Qiskit-Quantenschaltung und ihre Ausführungsfunktion definiert.
    Das folgende Beispiel zeigt, wie es funktioniert. Diese Schicht unterstützt nur Schaltungseingaben und Gewichte als Parameter.
    
    :param cirq_vqc: Eine Klasse, die die Definition, das Backend und die Ausführungsfunktionen einer Qiskit-Schaltung definiert.
    :param para_num: `int` - Anzahl der Parameter.
    :return: Eine Klasse, die Qiskit-Quantenschaltungsmodelle ausführen kann.

    Beispiel::


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
                # --- Schaltungsdefinition ---

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

        #Qiskit-Schaltungen Klasse definieren
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

    Eine Cirq-Schaltungs-Kapselungsschicht zur Implementierung von Vorwärts- und Rückwärtspropagation in VQNet. CIRQ_VQC ist eine Klasse, die vom Benutzer erfordert, eine Cirq-Quantenschaltung und ihre `run`-Funktion zu definieren. Das folgende Beispiel zeigt das Funktionsprinzip.
    Diese Schicht unterstützt nur Schaltungseingaben und Gewichte als Parameter.

    :param cirq_vqc: Eine Klasse, die die Definition, das Backend und die Ausführungsfunktionen einer Cirq-Schaltung definiert.
    :param para_num: `int` - Anzahl der Parameter.
    :return: Eine Klasse, die das Cirq-Quantenschaltungsmodell ausführen kann.


    .. note::

        Der folgende Beispielcode erfordert `cirq==1.5.0, numpy <2`.

    Beispiel::

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
                ###Qubits definieren
                q0 = cirq.NamedQubit ('q0')
                q1 = cirq.NamedQubit ('q1')
                q2 = cirq.NamedQubit ('q2')
                q3 = cirq.NamedQubit ('q3')
                qubits = [q0,q1,q2,q3]
                self.qubits = [q0,q1,q2,q3]
                ###Variationsparameter definieren
                param = sympy.symbols(f'theta(0:24)')
                self.theta = np.asarray(param).reshape((4,6))

                ###Schaltungen definieren
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
                
                ###Backend definieren
                self._backend = simulator

                self._param_symbols_list,self._input_symbols_list = get_circuit_symbols(self._circuit)


            def run(self,resolver,init_state):

                rlt = self._backend.simulate(self._circuit,resolver,initial_state=init_state).final_state_vector
                z0 = cirq.Z(self.qubits[0])

                qubit_map={self.qubits[0]: 0}
                
                expectation = z0.expectation_from_state_vector(rlt, qubit_map).real

                return expectation

        #Cirq-Schaltungen Klasse definieren
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

Quantengatter
***********************************

Die Methode zur Verarbeitung von Qubits wird als Quantengatter bezeichnet. Mit Quantengattern entwickeln wir Quantenzustände gezielt weiter. Quantengatter sind die Grundlage von Quantenalgorithmen.

Basis-Quantengatter
=================================

In VQNet verwenden wir die von Origin Quantum entwickelten Logikgatter von pyQPanda, um Quantenschaltungen zu erstellen und Quantensimulationen durchzuführen.
Die derzeit von pyQPanda unterstützten Gatter können im Abschnitt über Quantengatter von pyQPanda definiert werden.
Darüber hinaus kapselt VQNet auch einige häufig verwendete Quantengatter-Kombinationen im Bereich des maschinellen Lernens mit Quanten.



AmplitudeEmbeddingCircuit
=================================

.. py:function:: pyvqnet.qnn.template.AmplitudeEmbeddingCircuit(input_feat, qubits)

    Kodiert :math:`2^n` Merkmale in den Amplitudenvektor von :math:`n` Qubits.
    Um einen gültigen Quantenzustandsvektor darzustellen, muss die L2-Norm von ``features`` eins sein.

    :param input_feat: numpy-Array, das die Parameter darstellt
    :param qubits: von pyQPanda zugewiesene Qubits
    :return: Quantenschaltungen

    Beispiel::

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




Quantum Machine Learning APIs mit pyQPanda2
***************************************************************************

Quanten Generative Adversarial Networks zum Lernen und Laden von Zufallsverteilungen
==================================================================================================

Der Quantum Generative Adversarial Networks (`QGAN <https://www.nature.com/articles/s41534-019-0223-2>`_ )-Algorithmus verwendet reine variationelle Quantenschaltungen, um die erzeugten Quantenzustände mit einer bestimmten Zufallsverteilung zu präparieren. Dies reduziert die Anzahl der Logikgatter, die zur Erzeugung bestimmter Quantenzustände erforderlich sind, und verringert die Komplexität der Quantenschaltungen. Er verwendet die klassische GAN-Modellstruktur mit zwei Untermodellen: Generator und Diskriminator. Der Generator erzeugt eine bestimmte Verteilung für die Quantenschaltung. Und der Diskriminator unterscheidet die vom Generator erzeugten Datenstichproben von den echten, zufallsverteilten Trainingsdatenstichproben.
Hier ist ein Beispiel, wie VQNet das QGAN-Lernen und Laden von Zufallsverteilungen basierend auf dem Artikel `Quantum Generative Adversarial Networks for learning and loading random distributions <https://www.nature.com/articles/s41534-019-0223-2>`_ von Christa Zoufal implementiert.

.. image:: ./images/qgan-arch.PNG
   :width: 600 px
   :align: center

|

Um die Konstruktion der ``QGANAPI``-Klasse des quanten-generativen adversarialen Netzwerks durch VQNet zu realisieren, wird der Quantengenerator verwendet, um den Anfangszustand der real verteilten Daten zu präparieren. Die Anzahl der Qubits beträgt 3, und die Wiederholungsanzahl des internen parametrischen Schaltungsmoduls des Quantengenerators ist 1. Gleichzeitig wird KL als Metrik für das Laden der Zufallsverteilung durch QGAN verwendet.

.. code-block::

    import pickle
    import os
    import pyqpanda as pq
    from pyvqnet.qnn.qgan.qgan_utils import QGANAPI
    import numpy as np

    num_of_qubits = 3  # Konfiguration aus dem Paper
    rep = 1
    number_of_data = 10000
    # Datenstichproben aus verschiedenen Verteilungen laden
    mu = 1
    sigma = 1
    real_data = np.random.lognormal(mean=mu, sigma=sigma, size=number_of_data)


    # initialisieren
    save_dir = None
    qgan_model = QGANAPI(
        real_data,
        # numpy-generierte Datenverteilung, 1-dim.
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

Das folgende ist das ``train``-Modul von QGAN.

.. code-block::

    # trainieren
    qgan_model.train()  # QGAN trainieren


Das ``eval``-Modul von QGAN wird verwendet, um die Verlustfunktionskurve und das Wahrscheinlichkeitsverteilungsdiagramm zwischen der von QGAN vorbereiteten Zufallsverteilung und der realen Verteilung zu zeichnen.

.. code-block::

    # Wahrscheinlichkeitsverteilungsfunktion der generierten und der realen Verteilung anzeigen
    qgan_model.eval(real_data)  # PDF zeichnen

Das ``get_trained_quantum_parameters``-Modul von QGAN wird verwendet, um Trainingsparameter abzurufen und als numpy-Array auszugeben. Wenn ``save_DIR`` nicht leer ist, werden die Trainingsparameter in einer Datei gespeichert. Das ``Load_param_and_eval``-Modul von QGAN lädt Trainingsparameter, und das ``get_circuits_with_trained_param``-Modul ruft die nach dem Training vom Quantengenerator erzeugte pyQPanda-Schaltung ab.

.. code-block::

    # trainierte Quantenparameter abrufen
    param = qgan_model.get_trained_quantum_parameters()
    print(f" trainierte Parameter {param}")

    # gespeicherte Parameterdateien laden
    if save_dir is not None:
        path = os.path.join(
            save_dir, qgan_model._start_time + "trained_qgan_param.pickle")
        with open(path, "rb") as file:
            t3 = pickle.load(file)
        param = t3["quantum_parameters"]
        print(f" trainierte Parameter {param}")

    # Wahrscheinlichkeitsverteilungsfunktion der generierten und der realen Verteilung anzeigen
    qgan_model.load_param_and_eval(param)

    # Metrik berechnen
    print(qgan_model.eval_metric(param, "kl"))

    # Generator-Quantenschaltung abrufen
    m_machine = pq.CPUQVM()
    m_machine.init_qvm()
    qubits = m_machine.qAlloc_many(num_of_qubits)
    qpanda_cir = qgan_model.get_circuits_with_trained_param(qubits)
    print(qpanda_cir)

Im Allgemeinen erfordert das QGAN-Lernen und Laden von Zufallsverteilungen mehrere Trainingsdurchläufe mit verschiedenen Zufallskeimen, um die erwarteten Ergebnisse zu erzielen. Das folgende Beispiel zeigt die Wahrscheinlichkeitsverteilungsfunktion zwischen der von QGAN implementierten Lognormalverteilung und der realen Lognormalverteilung sowie die Verlustfunktionskurve zwischen QGANs Generator und Diskriminator.

.. image:: ./images/qgan-loss.png
   :width: 600 px
   :align: center

|

.. image:: ./images/qgan-pdf.png
   :width: 600 px
   :align: center

|


Quanten-Kernel-SVM
=================================

Bei maschinellen Lernaufgaben können Daten im ursprünglichen Raum oft nicht durch eine Hyperebene getrennt werden. Eine gängige Technik zum Finden solcher Hyperebenen besteht darin, eine nichtlineare Transformationsfunktion auf die Daten anzuwenden.
Diese Funktion wird als Merkmalsabbildung bezeichnet, mit der wir berechnen können, wie nahe sich Datenpunkte in diesem neuen Merkmalsraum für die Klassifikationsaufgabe des maschinellen Lernens stehen.

Dieses Beispiel bezieht sich auf die Arbeit: `Supervised learning with quantum enhanced feature spaces <https://arxiv.org/pdf/1804.11326.pdf>`_ .
Die erste Methode konstruiert variationelle Schaltungen für Datenklassifikationsaufgaben.

``gen_vqc_qsvm_data`` sind die Daten, die für dieses Beispiel benötigt werden. ``vqc_qsvm`` ist eine variable Unterschaltungsklasse zur Klassifikation der Eingabedaten.
Die Funktion ``vqc_qsvm.plot()`` visualisiert die Verteilung der Daten.

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
        #Wiederholungsanzahl der Unterschaltungen
        rep = 3

        #definiert die QSVM-Klasse
        VQC_QSVM = vqc_qsvm(batch_size, maxiter, rep)
        #erzeugt zufällig Daten aus der Arbeit.
        train_features, test_features, train_labels, test_labels, samples = \
            gen_vqc_qsvm_data(training_size=training_size, test_size=test_size, gap=gap)
        VQC_QSVM.plot(train_features, test_features, train_labels, test_labels, samples)
        #trainieren
        VQC_QSVM.train(train_features, train_labels)
        #testen
        rlt, acc_1 = VQC_QSVM.predict(test_features, test_labels)
        print(f"Testgenauigkeit {acc_1}")


Zusätzlich zu der oben erwähnten direkten Verwendung von variationellen Quantenschaltungen zur Abbildung klassischer Datenmerkmale auf Quanten-Merkmalsräume wird in der Arbeit `Supervised learning with quantum enhanced feature spaces <https://arxiv.org/pdf/1804.11326.pdf>`_
auch die Methode eingeführt, Kernel-Funktionen direkt mit Quantenschaltungen zu schätzen und sie mit klassischen Support-Vektor-Maschinen zu klassifizieren.
Analog zu verschiedenen Kernel-Funktionen in klassischen SVMs :math:`K(i,j)` wird die Quanten-Kernel-Funktion verwendet, um das innere Produkt klassischer Daten im Quanten-Merkmalsraum :math:`\phi(\mathbf{x}_i)` zu definieren:

.. math::
    |\langle \phi(\mathbf{x}_j) | \phi(\mathbf{x}_i) \rangle |^2 =  |\langle 0 | U^\dagger(\mathbf{x}_j) U(\mathbf{x}_i) | 0 \rangle |^2

Mit VQNet und pyQPanda definieren wir einen ``QuantumKernel_VQNet``, um eine Quanten-Kernel-Funktion zu erzeugen, und verwenden ``sklearn`` zur Klassifikation:

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
    print(f"Quanten-Kernel-Klassifikationstestergebnis: {score}")


Simultaneous Perturbation Stochastic Approximation Optimierer
=================================================================


.. py:function:: pyvqnet.qnn.SPSA(maxiter: int = 1000, save_steps: int = 1, last_avg: int = 1, c0: float = _C0, c1: float = 0.2, c2: float = 0.602, c3: float = 0.101, c4: float = 0, init_para=None, model=None, calibrate_flag=False)
    

    Simultaneous Perturbation Stochastic Approximation (SPSA) Optimierer.

    SPSA bietet eine stochastische Methode zur Approximation des Gradienten einer mehrdimensionalen differenzierbaren Kostenfunktion.
    Dabei wird die Kostenfunktion zweimal mit einem gestörten Parametervektor ausgewertet: Jede Komponente des ursprünglichen Parametervektors wird gleichzeitig um einen zufällig generierten Wert verschoben.
    Weitere Informationen finden Sie auf der `SPSA-Website <http://www.jhuapl.edu/SPSA>`__.

    :param maxiter: Die maximale Anzahl der durchzuführenden Iterationen. Standardwert: 1000.
    :param save_steps: Speichert die Zwischeninformationen nach jeweils save_steps Schritten. Standardwert: 1.
    :param last_avg: Mittelungsparameter für die letzten last_avg Iterationen.
        Wenn last_avg = 1, wird nur die letzte Iteration berücksichtigt. Standardwert: 1.
    :param c0: initial a. Schrittweite für die Aktualisierung der Parameter. Standardwert: 0.2*pi
    :param c1: initial c. Schrittweite zur Approximation des Gradienten. Standardwert: 0.1.
    :param c2: alpha aus dem Paper, verwendet zur Anpassung von a(c0) bei jeder Iteration. Standardwert: 0.602.
    :param c3: gamma aus dem Paper, verwendet zur Anpassung von c(c1) bei jeder Iteration. Standardwert: 0.101.
    :param c4: Wird ebenfalls zur Steuerung der Parameter von a verwendet. Standardwert: 0.
    :param init_para: Initialisierungsparameter. Standardwert: None.
    :param model: Parametrisches Modell: model. Standardwert: None.
    :param calibrate_flag: Ob die Hyperparameter a und c kalibriert werden sollen. Standardwert: False.

    :return: eine SPSA-Optimierer-Instanz


    .. warning::

        SPSA unterstützt nur 1-dimensionale Parameter.

    Beispiel::

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

    Verwendet SPSA zur Optimierung der Eingabedaten.

    :param input_data: Eingabedaten
    :return:

        train_para: endgültiger Parameter

        theta_best: Die gemittelten Parameter nach den letzten `last_avg` Iterationen.

    Beispiel::

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

Quantum Fisher Informationsberechnungsmatrix
========================================================

.. py:class:: pyvqnet.qnn.opt.quantum_fisher(py_qpanda_config, params, target_gate_type_lists,target_gate_bits_lists, qcir_lists, wires)
    
    Gibt die Quantum-Fisher-Informationsmatrix für eine Quantenschaltung zurück.

    .. math::

        \mathrm{QFIM}_{i, j}=4 \operatorname{Re}\left[\left\langle\partial_i \psi(\boldsymbol{\theta}) \mid \partial_j \psi(\boldsymbol{\theta})\right\rangle-\left\langle\partial_i \psi(\boldsymbol{\theta}) \mid \psi(\boldsymbol{\theta})\right\rangle\left\langle\psi(\boldsymbol{\theta}) \mid \partial_j \psi(\boldsymbol{\theta})\right\rangle\right]

    Die Kurzform ist :math::math:`\left|\partial_j \psi(\boldsymbol{\theta})\right\rangle:=\frac{\partial}{\partial \theta_j}|\psi(\boldsymbol{\theta})\rangle`.

    .. note::

        Derzeit werden nur RX, RY, RZ unterstützt.

    :param params: Variable Parameter in den Schaltungen.
    :param target_gate_type_lists: Unterstützt "RX", "RY", "RZ" oder Listen.
    :param target_gate_bits_lists: Auf welches Qubit oder welche Qubits das parametrisierte Gatter wirkt.
    :param qcir_lists: Die Liste der Quantenschaltungen vor dem Ziel-Parametrisierten-Gatter zur Berechnung des metrischen Tensors, siehe folgendes Beispiel.
    :param wires: Gesamter Qubit-Index für Quantenschaltungen.

    Beispiel::
    
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

        # Das obige Beispiel zeigt, dass es keine identischen Gatter in derselben Schicht gibt,
        # aber in derselben Schicht müssen Sie die Logikgatter gemäß dem folgenden Beispiel anpassen.
        
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
            qcir.insert(pq.CNOT(config._qubits[0], config._qubits[1])) #  01-Teil
            
            qcir.insert(pq.RZ(config._qubits[1], params[2]))  #  02-Teil
            
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

        mt = quantum_fisher(config, params2, [['RX', 'RY'], ['RZ'], ['RZ']], # rx,ry zählen als Schicht eins, erstes rz als Schicht zwei, zweites rz als Schicht drei.
                                [[0, 1], [1], [1]], qcir, [0, 1])
