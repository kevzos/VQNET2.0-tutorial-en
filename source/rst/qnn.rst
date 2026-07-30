Quantum Machine Learning API con QPanda2
####################################################


.. warning::

    La parte di calcolo quantistico della seguente interfaccia utilizza pyQPanda2.

    A causa di problemi di compatibilità tra pyQPanda2 e pyqpanda3, è necessario installare pyqpnda2 manualmente, `pip install pyqpanda`

Layer di Calcolo Quantistico
***********************************

.. _QuantumLayer:

QuantumLayer
=================================

QuantumLayer è una classe wrapper del modulo autograd che supporta circuiti quantistici variazionali. È possibile definire una funzione come argomento, ad esempio ``qprog_with_measure``. Questa funzione deve contenere il circuito quantistico definito da pyQPanda: generalmente contiene un circuito di codifica, un circuito di evoluzione e un'operazione di misura.
Questa classe QuantumLayer può essere incorporata in un modello ibrido quantistico-classico di machine learning e minimizzare la funzione obiettivo o la funzione di loss del modello ibrido quantistico-classico tramite il metodo classico di discesa del gradiente.
È possibile specificare il metodo di calcolo del gradiente dei parametri del circuito quantistico in ``QuantumLayer`` modificando il parametro ``diff_method``. ``QuantumLayer`` supporta attualmente due metodi: ``finite_diff`` e ``parameter-shift``.

Il metodo ``finite_diff`` è uno dei metodi numerici più tradizionali e comuni per stimare il gradiente di una funzione. L'idea principale è sostituire le derivate parziali con differenze:

.. math::

    f^{\prime}(x)=\lim _{h 

ightarrow 0} rac{f(x+h)-f(x)}{h}


Per il metodo ``parameter-shift`` utilizziamo la funzione obiettivo, ad esempio:


.. math:: O(\theta)=\left\langle 0\left|U^{\dagger}(\theta) H U(\theta)\right| 0\right\rangle

È teoricamente possibile calcolare il gradiente dei parametri rispetto all'Hamiltoniana in un circuito quantistico con il metodo più preciso: ``parameter-shift``.

.. math::

    \nabla O(\theta)=
    \frac{1}{2}\left[O\left(\theta+\frac{\pi}{2}\right)-O\left(\theta-\frac{\pi}{2}\right)\right]

.. py:class:: pyvqnet.qnn.quantumlayer.QuantumLayer(qprog_with_measure,para_num,machine_type_or_cloud_token,num_of_qubits:int,num_of_cbits:int = 1,diff_method:str = "parameter_shift",delta:float = 0.01, dtype=None, name='')

    Modulo di calcolo astratto per circuiti quantistici variazionali. Simula un circuito quantistico parametrizzato e ottiene il risultato della misura.
    QuantumLayer eredita da Module, quindi può calcolare i gradienti dei parametri del circuito e addestrare modelli di circuiti quantistici variazionali o incorporare circuiti quantistici variazionali in modelli ibridi quantistico-classici.

    Questa classe non richiede di inizializzare la macchina virtuale nella funzione ``qprog_with_measure``.

    :param qprog_with_measure: funzioni di circuito quantistico chiamabili, costruite con pyQPanda2
    :param para_num: ``int`` - Numero di parametri
    :param machine_type_or_cloud_token: tipo di macchina qpanda o token QCLOUD di pyQPanda2
    :param num_of_qubits: numero di qubit
    :param num_of_cbits: numero di bit classici
    :param diff_method: ``'parameter_shift'`` o ``'finite_diff'``
    :param delta: delta per la differenziazione
    :param dtype: Il tipo di dato del parametro, default: None, utilizza il tipo di dato predefinito kfloat32, che rappresenta un numero a virgola mobile a 32 bit.
    :param name: nome del layer di output

    :return: un modulo in grado di calcolare circuiti quantistici.

    .. note::
        qprog_with_measure è una funzione di circuito quantistico definita in pyQPanda2.

        Questa funzione deve contenere i seguenti parametri, altrimenti non può funzionare correttamente in QuantumLayer.

        qprog_with_measure (input,param,qubits,cbits,m_machine)

            `input`: array_like dati classici 1-dim in ingresso

            `param`: array_like parametri del circuito quantistico 1-dim in ingresso

            `qubits`: qubit allocati da QuantumLayer

            `cbits`: cbit allocati da QuantumLayer. Se il tuo circuito non utilizza cbit, dovresti comunque riservare questo parametro.

            `m_machine`: simulatore creato da QuantumLayer

        Utilizza l'attributo ``m_para`` di QuantumLayer per ottenere i parametri di addestramento del circuito quantistico variabile. Il parametro è una classe ``QTensor``, che può essere convertita in un array numpy usando l'interfaccia ``to_numpy()``.

    .. note::

        La classe ha un alias: `QpandaQCircuitVQCLayer` .

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
        # dati classici in ingresso
        input = QTensor([[1,2,3,4],[40,22,2,3],[33,3,25,2.0]] )
        # propagazione in avanti (forward)
        rlt = pqc(input)
        grad =  QTensor(np.ones(rlt.data.shape)*1000)
        # retropropagazione (backward)
        rlt.backward(grad)
        print(rlt)
        # [
        # [0.2500000, 0.2500000, 0.2500000, 0.2500000],
        # [0.2500000, 0.2500000, 0.2500000, 0.2500000],
        # [0.2500000, 0.2500000, 0.2500000, 0.2500000]
        # ]

QuantumLayerV2
=================================

Se si ha maggiore familiarità con la sintassi di pyQPanda2, si può utilizzare la classe QuantumLayerV2. È possibile definire la funzione del circuito quantistico usando ``qubits``, ``cbits`` e ``machine``, quindi passarla come argomento ``qprog_with_measure`` di QuantumLayerV2.

.. py:class:: pyvqnet.qnn.quantumlayer.QuantumLayerV2(qprog_with_measure, para_num, diff_method: str = 'parameter_shift', delta: float = 0.01, dtype=None, name='')

    Modulo di calcolo astratto per circuiti quantistici variazionali. Simula un circuito quantistico parametrizzato e ottiene il risultato della misura.
    QuantumLayer eredita da Module, quindi può calcolare i gradienti dei parametri del circuito e addestrare modelli di circuiti quantistici variazionali o incorporare circuiti quantistici variazionali in modelli ibridi quantistico-classici.

    Per utilizzare questo modulo, è necessario creare la propria macchina virtuale quantistica e allocare qubit e cbit.

    :param qprog_with_measure: funzioni di circuito quantistico chiamabili, costruite con pyQPanda2
    :param para_num: ``int`` - Numero di parametri
    :param diff_method: ``'parameter_shift'`` o ``'finite_diff'``
    :param delta: delta per la differenziazione
    :param dtype: Il tipo di dato del parametro, default: None, utilizza il tipo di dato predefinito kfloat32, che rappresenta un numero a virgola mobile a 32 bit.
    :param name: nome del layer di output
    :return: un modulo in grado di calcolare circuiti quantistici.

    .. note::
        qprog_with_measure è una funzione di circuito quantistico definita in pyQPanda.

        Questa funzione deve contenere i seguenti parametri, altrimenti non può funzionare correttamente in QuantumLayerV2.

        Rispetto a QuantumLayer, è necessario allocare qubit e simulatore manualmente,

        e potrebbe anche essere necessario allocare cbit se qprog_with_measure richiede la misura quantistica.

        qprog_with_measure (input,param)

        `input`: array_like dati classici 1-dim in ingresso

        `param`: array_like parametri del circuito quantistico 1-dim in ingresso

    .. note::

        La classe ha un alias: `QpandaQCircuitVQCLayerLite` .

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

        # dati classici in ingresso
        input = QTensor([[1,2,3,4],[4,2,2,3],[3,3,2,2.0]] )

        # propagazione in avanti (forward)
        rlt = pqc(input)

        grad =  QTensor(np.ones(rlt.data.shape)*1000)
        # retropropagazione (backward)
        rlt.backward(grad)
        print(rlt)

        # [
        # [0.2500000, 0.2500000, 0.2500000, 0.2500000],
        # [0.2500000, 0.2500000, 0.2500000, 0.2500000],
        # [0.2500000, 0.2500000, 0.2500000, 0.2500000]
        # ]

 


NoiseQuantumLayer
=================================

In un computer quantistico reale, a causa delle caratteristiche fisiche del bit quantistico, c'è sempre un inevitabile errore di calcolo. Per simulare meglio questo errore in una macchina virtuale quantistica, VQNet supporta anche una macchina virtuale quantistica con rumore. La simulazione di una macchina virtuale quantistica con rumore è più vicina al computer quantistico reale. È possibile personalizzare il tipo di porta logica supportata e il modello di rumore supportato dalla porta logica.
Il modello di rumore quantistico attualmente supportato è definito in ``NoiseQVM`` di pyQPanda2.

È possibile utilizzare ``NoiseQuantumLayer`` per definire un modulo di differenziazione automatica per circuiti quantistici. ``NoiseQuantumLayer`` supporta la macchina virtuale quantistica di pyQPanda2 con rumore. È possibile definire una funzione come argomento ``qprog_with_measure``. Questa funzione deve contenere il circuito quantistico definito da pyQPanda, ed è inoltre necessario passare un argomento ``noise_set_config``, utilizzando l'interfaccia pyQPanda per configurare il modello di rumore.

.. py:class:: pyvqnet.qnn.quantumlayer.NoiseQuantumLayer(qprog_with_measure, para_num, machine_type, num_of_qubits: int, num_of_cbits: int = 1, diff_method: str = 'parameter_shift', delta: float = 0.01, noise_set_config=None, dtype=None, name='')

    Modulo di calcolo astratto per circuiti quantistici variazionali. Simula un circuito quantistico parametrizzato e ottiene il risultato della misura.
    QuantumLayer eredita da Module, quindi può calcolare i gradienti dei parametri del circuito e addestrare modelli di circuiti quantistici variazionali o incorporare circuiti quantistici variazionali in modelli ibridi quantistico-classici.

    Questo modulo deve essere inizializzato con il modello di rumore tramite ``noise_set_config``.

    :param qprog_with_measure: funzioni di circuito quantistico chiamabili, costruite con pyQPanda2
    :param para_num: ``int`` - Numero di parametri
    :param machine_type: tipo di macchina pyQPanda2
    :param num_of_qubits: numero di qubit
    :param num_of_cbits: numero di cbit
    :param diff_method: ``'parameter_shift'`` o ``'finite_diff'``
    :param delta: delta per la differenziazione
    :param noise_set_config: funzione di configurazione del rumore
    :param dtype: Il tipo di dato del parametro, default: None, utilizza il tipo di dato predefinito kfloat32, che rappresenta un numero a virgola mobile a 32 bit.
    :param name: nome del layer di output

    :return: un modulo in grado di calcolare circuiti quantistici con modello di rumore.

    .. note::
        qprog_with_measure è una funzione di circuito quantistico definita in pyQPanda.

        Questa funzione deve contenere i seguenti parametri, altrimenti non può funzionare correttamente in NoiseQuantumLayer.

        qprog_with_measure (input,param,qubits,cbits,m_machine)

        `input`: array_like dati classici 1-dim in ingresso

        `param`: array_like parametri del circuito quantistico 1-dim in ingresso

        `qubits`: qubit allocati da NoiseQuantumLayer

        `cbits`: cbit allocati da NoiseQuantumLayer. Se il tuo circuito non utilizza cbit, dovresti comunque riservare questo parametro.

        `m_machine`: simulatore creato da NoiseQuantumLayer

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
            # Calcola le probabilità per ogni stato
            probabilities = counts / 100
            # Ottieni il valore atteso dello stato
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

Ecco un esempio di ``noise_set_config``: qui aggiungiamo il modello di rumore BITFLIP_KRAUS_OPERATOR con argomento di rumore p=0.01 alle porte quantistiche ``RX``, ``RY``, ``RZ``, ``X``, ``Y``, ``Z``, ``H``, ecc.

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

    Un layer wrapper per implementare la propagazione in avanti e all'indietro con circuiti Qiskit in VQNet. QISKIT_VQC è una classe che definisce un circuito quantistico Qiskit e la sua funzione di esecuzione.
    L'esempio seguente mostra il suo funzionamento. Questo layer supporta solo ingressi del circuito e pesi come parametri.

    :param cirq_vqc: Una classe che definisce la definizione, il backend e le funzioni di esecuzione di un circuito Qiskit.
    :param para_num: ``int`` - Il numero di parametri.
    :return: Una classe in grado di eseguire modelli di circuiti quantistici qiskit.

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
                    file_path_ungz = file_path[:-3].replace('\', '/')
                    if not os.path.exists(file_path_ungz):
                        open(file_path_ungz,"wb").write(f.read())
                return

            print("Downloading " + file_name + " ... ")
            urllib.request.urlretrieve(url_base + file_name, file_path)
            if os.path.exists(file_path):
                    with gzip.GzipFile(file_path) as f:
                        file_path_ungz = file_path[:-3].replace('\', '/')
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
                fname_image = os.path.join(path, 'train-images.idx3-ubyte').replace('\', '/')
                fname_label = os.path.join(path, 'train-labels.idx1-ubyte').replace('\', '/')
            elif dataset == "testing_data":
                fname_image = os.path.join(path, 't10k-images.idx3-ubyte').replace('\', '/')
                fname_label = os.path.join(path, 't10k-labels.idx1-ubyte').replace('\', '/')
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
                # --- Definizione del circuito ---

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

        # definisci la classe del circuito qiskit
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

    Un layer di incapsulamento per circuiti Cirq per implementare la propagazione in avanti e all'indietro in VQNet. CIRQ_VQC è una classe che richiede all'utente di definire un circuito quantistico Cirq e la sua funzione ``run``. L'esempio seguente dimostra il suo principio di funzionamento.
    Questo layer supporta solo ingressi del circuito e pesi come parametri.

    :param cirq_vqc: Una classe che definisce la definizione, il backend e le funzioni di esecuzione di un circuito Cirq.
    :param para_num: ``int`` - Il numero di parametri.
    :return: Una classe in grado di eseguire il modello di circuito quantistico Cirq.


    .. note::

        Il codice di esempio seguente richiede ``cirq==1.5.0, numpy <2``.

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
                fname_image = os.path.join(path, 'train-images.idx3-ubyte').replace('\', '/')
                fname_label = os.path.join(path, 'train-labels.idx1-ubyte').replace('\', '/')
            elif dataset == "testing_data":
                fname_image = os.path.join(path, 't10k-images.idx3-ubyte').replace('\', '/')
                fname_label = os.path.join(path, 't10k-labels.idx1-ubyte').replace('\', '/')
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
                ### definisci qubit
                q0 = cirq.NamedQubit ('q0')
                q1 = cirq.NamedQubit ('q1')
                q2 = cirq.NamedQubit ('q2')
                q3 = cirq.NamedQubit ('q3')
                qubits = [q0,q1,q2,q3]
                self.qubits = [q0,q1,q2,q3]
                ### definisci parametri variazionali
                param = sympy.symbols(f'theta(0:24)')
                self.theta = np.asarray(param).reshape((4,6))

                ### definisci circuiti
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
                
                ### definisci backend
                self._backend = simulator

                self._param_symbols_list,self._input_symbols_list = get_circuit_symbols(self._circuit)


            def run(self,resolver,init_state):

                rlt = self._backend.simulate(self._circuit,resolver,initial_state=init_state).final_state_vector
                z0 = cirq.Z(self.qubits[0])

                qubit_map={self.qubits[0]: 0}
                
                expectation = z0.expectation_from_state_vector(rlt, qubit_map).real

                return expectation

        # definisci la classe del circuito cirq
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

Porte Quantistiche
***********************************

Il modo di operare sui qubit è chiamato porte quantistiche. Utilizzando le porte quantistiche, evolviamo consapevolmente gli stati quantistici. Le porte quantistiche sono la base degli algoritmi quantistici.

Porte quantistiche di base
=================================

In VQNet, utilizziamo le porte logiche di pyQPanda sviluppate da Origin Quantum per costruire circuiti quantistici e condurre simulazioni quantistiche.
Le porte attualmente supportate da pyQPanda possono essere definite nella sezione delle porte quantistiche di pyQPanda.
Inoltre, VQNet incapsula anche alcune combinazioni di porte quantistiche comunemente usate nel quantum machine learning.



AmplitudeEmbeddingCircuit
=================================

.. py:function:: pyvqnet.qnn.template.AmplitudeEmbeddingCircuit(input_feat, qubits)

    Codifica :math:`2^n` feature nel vettore di ampiezza di :math:`n` qubit.
    Per rappresentare un vettore di stato quantistico valido, la norma L2 di ``features`` deve essere pari a uno.

    :param input_feat: array numpy che rappresenta i parametri
    :param qubits: qubit allocati da pyQPanda
    :return: circuiti quantistici

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
        # q_0:  |0>───────────── ─── └RY(0.853255)├ ─── └RY(1.376290)├
        #           ┌────────────┐ ┌─┐ └──────┬─────┘ ┌─┐ └──────┬─────┘
        # q_1:  |0>─┤RY(2.355174)├ ┤X┬ ───────┬────── ┤X┬ ───────┬──────
        #           └────────────┘ └─┬┘                └─┬┘               




Quantum Machine Learning APIs con pyQPanda2
***************************************************************************

Reti Generative Avversarie Quantistiche per l'apprendimento e il caricamento di distribuzioni casuali
=====================================================================================================

L'algoritmo Quantum Generative Adversarial Networks (`QGAN <https://www.nature.com/articles/s41534-019-0223-2>`_) utilizza circuiti variazionali puramente quantistici per preparare stati quantistici generati con una distribuzione casuale specifica, riducendo le porte logiche necessarie per generare stati quantistici specifici e la complessità dei circuiti quantistici. Utilizza la struttura classica del modello GAN, che ha due sottomodelli: Generatore e Discriminatore. Il Generatore produce una distribuzione specifica per il circuito quantistico. Il Discriminatore distingue tra i campioni di dati generati dal Generatore e i campioni di dati di addestramento reali distribuiti casualmente.
Ecco un esempio di VQNet che implementa l'apprendimento e il caricamento di distribuzioni casuali con QGAN basato sull'articolo `Quantum Generative Adversarial Networks for learning and loading random distributions <https://www.nature.com/articles/s41534-019-0223-2>`_ di Christa Zoufal.

.. image:: ./images/qgan-arch.PNG
   :width: 600 px
   :align: center

|

Per realizzare la costruzione della classe ``QGANAPI`` della rete generativa avversaria quantistica tramite VQNet, il generatore quantistico viene utilizzato per preparare lo stato iniziale dei dati reali distribuiti. Il numero di bit quantistici è 3 e il numero di ripetizioni del modulo circuitale parametrico interno del generatore quantistico è 1. Nel frattempo, KL viene utilizzata come metrica per il caricamento della distribuzione casuale di QGAN.

.. code-block::

    import pickle
    import os
    import pyqpanda as pq
    from pyvqnet.qnn.qgan.qgan_utils import QGANAPI
    import numpy as np

    num_of_qubits = 3  # paper config
    rep = 1
    number_of_data = 10000
    # Carica campioni di dati da diverse distribuzioni
    mu = 1
    sigma = 1
    real_data = np.random.lognormal(mean=mu, sigma=sigma, size=number_of_data)


    # inizializzazione
    save_dir = None
    qgan_model = QGANAPI(
        real_data,
        # distribuzione dati generata con numpy, 1-dim.
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

Di seguito è riportato il modulo ``train`` di QGAN.

.. code-block::

    # train
    qgan_model.train()  # addestra qgan


Il modulo ``eval`` di QGAN è progettato per tracciare la curva della funzione di loss e il diagramma della distribuzione di probabilità tra la distribuzione casuale preparata da QGAN e la distribuzione reale.

.. code-block::

    # mostra la funzione di distribuzione di probabilità della distribuzione generata e di quella reale
    qgan_model.eval(real_data)  # disegna pdf

Il modulo ``get_trained_quantum_parameters`` di QGAN viene utilizzato per ottenere i parametri di addestramento e restituirli come array numpy. Se ``save_DIR`` non è vuoto, i parametri di addestramento vengono salvati in un file. Il modulo ``Load_param_and_eval`` di QGAN carica i parametri di addestramento, e il modulo ``get_circuits_with_trained_param`` ottiene il circuito pyQPanda generato dal generatore quantistico dopo l'addestramento.

.. code-block::

    # ottieni parametri quantistici addestrati
    param = qgan_model.get_trained_quantum_parameters()
    print(f" trained param {param}")

    # carica file di parametri salvati
    if save_dir is not None:
        path = os.path.join(
            save_dir, qgan_model._start_time + "trained_qgan_param.pickle")
        with open(path, "rb") as file:
            t3 = pickle.load(file)
        param = t3["quantum_parameters"]
        print(f" trained param {param}")

    # mostra la funzione di distribuzione di probabilità della distribuzione generata e di quella reale
    qgan_model.load_param_and_eval(param)

    # calcola metrica
    print(qgan_model.eval_metric(param, "kl"))

    # ottieni circuito quantistico del generatore
    m_machine = pq.CPUQVM()
    m_machine.init_qvm()
    qubits = m_machine.qAlloc_many(num_of_qubits)
    qpanda_cir = qgan_model.get_circuits_with_trained_param(qubits)
    print(qpanda_cir)

In generale, l'apprendimento e il caricamento di distribuzioni casuali con QGAN richiede molteplici addestramenti del modello con diversi seed casuali per ottenere i risultati attesi. Ad esempio, ecco il grafico della funzione di distribuzione di probabilità tra la distribuzione lognormale implementata da QGAN e la distribuzione lognormale reale, e la curva della funzione di loss tra il generatore e il discriminatore di QGAN.

.. image:: ./images/qgan-loss.png
   :width: 600 px
   :align: center

|

.. image:: ./images/qgan-pdf.png
   :width: 600 px
   :align: center

|


SVM con kernel quantistico
=================================

Nelle attività di machine learning, i dati spesso non possono essere separati da un iperpiano nello spazio originale. Una tecnica comune per trovare tali iperpiani consiste nell'applicare una funzione di trasformazione non lineare ai dati.
Questa funzione è chiamata feature map, attraverso la quale possiamo calcolare quanto siano vicini i punti dati in questo nuovo spazio delle feature per il compito di classificazione del machine learning.

Questo esempio si riferisce alla tesi: `Supervised learning with quantum enhanced feature spaces <https://arxiv.org/pdf/1804.11326.pdf>`_.
Il primo metodo costruisce circuiti variazionali per compiti di classificazione dei dati.

``gen_vqc_qsvm_data`` sono i dati necessari per generare questo esempio. ``vqc_qsvm`` è una classe di sottocircuito variabile utilizzata per classificare i dati in ingresso.
La funzione ``vqc_qsvm.plot()`` visualizza la distribuzione dei dati.

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
        # numero di ripetizioni dei sottocircuiti
        rep = 3

        # definisce la classe QSVM
        VQC_QSVM = vqc_qsvm(batch_size, maxiter, rep)
        # genera casualmente i dati dalla tesi.
        train_features, test_features, train_labels, test_labels, samples = \
            gen_vqc_qsvm_data(training_size=training_size, test_size=test_size, gap=gap)
        VQC_QSVM.plot(train_features, test_features, train_labels, test_labels, samples)
        # addestramento
        VQC_QSVM.train(train_features, train_labels)
        # test
        rlt, acc_1 = VQC_QSVM.predict(test_features, test_labels)
        print(f"testing_accuracy {acc_1}")


Oltre al suddetto uso diretto di circuiti quantistici variazionali per mappare le feature dei dati classici negli spazi delle feature quantistiche, nell'articolo `Supervised learning with quantum enhanced feature spaces <https://arxiv.org/pdf/1804.11326.pdf>`_,
viene introdotto anche il metodo di stima diretta delle funzioni kernel utilizzando circuiti quantistici e la loro classificazione tramite macchine a vettori di supporto classiche.
In analogia con le varie funzioni kernel nella SVM classica :math:`K(i,j)`, si utilizza la funzione kernel quantistica per definire il prodotto interno dei dati classici nello spazio delle feature quantistiche :math:`\phi(\mathbf{x}_i)` :

.. math::
    |\langle \phi(\mathbf{x}_j) | \phi(\mathbf{x}_i) \rangle |^2 =  |\langle 0 | U^\dagger(\mathbf{x}_j) U(\mathbf{x}_i) | 0 \rangle |^2

Utilizzando VQNet e pyQPanda, definiamo un ``QuantumKernel_VQNet`` per generare una funzione kernel quantistica e utilizziamo ``sklearn`` per la classificazione:

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


Ottimizzatori Simultaneous Perturbation Stochastic Approximation
=================================================================


.. py:function:: pyvqnet.qnn.SPSA(maxiter: int = 1000, save_steps: int = 1, last_avg: int = 1, c0: float = _C0, c1: float = 0.2, c2: float = 0.602, c3: float = 0.101, c4: float = 0, init_para=None, model=None, calibrate_flag=False)
    

    Ottimizzatore Simultaneous Perturbation Stochastic Approximation (SPSA).

    SPSA fornisce un metodo stocastico per approssimare il gradiente di una funzione di costo differenziabile multivariata.
    Per ottenere ciò, la funzione di costo viene valutata due volte utilizzando un vettore di parametri perturbato: ogni componente del vettore di parametri originale viene spostata simultaneamente di un valore generato casualmente.
    Ulteriori informazioni sono disponibili sul `sito web di SPSA <http://www.jhuapl.edu/SPSA>`__.

    :param maxiter: Il numero massimo di iterazioni da eseguire. Valore predefinito: 1000.
    :param save_steps: Salva le informazioni intermedie di ogni passo save_steps. Valore predefinito: 1.
    :param last_avg: Parametro di media per le ultime last_avg iterazioni.
        Se last_avg = 1, viene considerata solo l'ultima iterazione. Valore predefinito: 1.
    :param c0: a iniziale. Dimensione del passo per l'aggiornamento dei parametri. Valore predefinito: 0.2*pi
    :param c1: c iniziale. Dimensione del passo utilizzata per approssimare il gradiente. Valore predefinito: 0.1.
    :param c2: alpha dall'articolo, utilizzato per regolare a(c0) a ogni iterazione. Valore predefinito: 0.602.
    :param c3: gamma dell'articolo, utilizzato per regolare c(c1) a ogni iterazione. Valore predefinito: 0.101.
    :param c4: Utilizzato anche per controllare i parametri di a. Valore predefinito: 0.
    :param init_para: Parametri di inizializzazione. Predefinito: None.
    :param model: Modello parametrico: model. Predefinito: None.
    :param calibrate_flag: se calibrare gli iperparametri a e c, valore predefinito: False.

    :return: un'istanza dell'ottimizzatore SPSA


    .. warning::

        SPSA supporta solo parametri 1-dim.

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

    utilizza SPSA per ottimizzare i dati in ingresso.

    :param input_data: dati in ingresso
    :return:

        train_para: parametro finale

        theta_best: La media dei parametri dopo le ultime ``last_avg`` iterazioni.

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


Matrice di calcolo dell'informazione quantistica di Fisher
===========================================================

.. py:class:: pyvqnet.qnn.opt.quantum_fisher(py_qpanda_config, params, target_gate_type_lists,target_gate_bits_lists, qcir_lists, wires)
    
    Restituisce la matrice di informazione quantistica di Fisher per un circuito quantistico.

    .. math::

        \mathrm{QFIM}_{i, j}=4 \operatorname{Re}\left[\left\langle\partial_i \psi(\boldsymbol{\theta}) \mid \partial_j \psi(\boldsymbol{\theta})\right\rangle-\left\langle\partial_i \psi(\boldsymbol{\theta}) \mid \psi(\boldsymbol{\theta})\right\rangle\left\langle\psi(\boldsymbol{\theta}) \mid \partial_j \psi(\boldsymbol{\theta})\right\rangle\right]

    La versione abbreviata è :math::math:`\left|\partial_j \psi(\boldsymbol{\theta})\right\rangle:=\frac{\partial}{\partial \theta_j}|\psi(\boldsymbol{\theta})\rangle`.

    .. note::

        Attualmente sono supportati solo RX, RY, RZ.

    :param params: Parametri variabili nei circuiti.
    :param target_gate_type_lists: Supporta ``"RX"``, ``"RY"``, ``"RZ"`` o liste.
    :param target_gate_bits_lists: Quale bit quantistico o quali bit quantistici su cui agisce la porta parametrizzata.
    :param qcir_lists: La lista dei circuiti quantistici prima della porta parametrizzata target per calcolare il tensore metrico, vedere l'esempio seguente.
    :param wires: Indice totale dei bit quantistici per i circuiti quantistici.

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

        # L'esempio precedente mostra che non ci sono porte identiche nello stesso layer,
        # ma nello stesso layer è necessario modificare le porte logiche secondo l'esempio seguente.
        
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

        mt = quantum_fisher(config, params2, [['RX', 'RY'], ['RZ'], ['RZ']], # rx,ry contano come layer uno, primo rz come layer due, secondo rz come layer tre.
                                [[0, 1], [1], [1]], qcir, [0, 1])
