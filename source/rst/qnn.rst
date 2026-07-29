API d'apprentissage automatique quantique avec QPanda2
#########################################################


.. warning::

    La partie calcul quantique de l'interface suivante utilise pyQPanda2.

    En raison des problèmes de compatibilité entre pyQPanda2 et pyqpanda3, vous devez installer pyqpnda2 vous-même, `pip install pyqpanda`

Couche de calcul quantique
***********************************

.. _QuantumLayer:

QuantumLayer
=================================

QuantumLayer est une classe d'encapsulation du module autograd qui prend en charge les circuits quantiques variationnels. Vous pouvez definir une fonction comme argument, telle que ``qprog_with_measure``. Cette fonction doit contenir le circuit quantique defini par pyQPanda : elle contient generalement un circuit de codage, un circuit d'evolution et une operation de mesure.
Cette classe QuantumLayer peut etre integree dans un modele d'apprentissage automatique hybride quantique-classique et minimiser la fonction objectif ou la fonction de perte du modele hybride quantique-classique grace a la methode classique de descente de gradient.
Vous pouvez specifier la methode de calcul du gradient des parametres du circuit quantique dans ``QuantumLayer`` en modifiant le parametre ``diff_method``. ``QuantumLayer`` prend actuellement en charge deux methodes : ``finite_diff`` et ``parameter-shift``.

La methode ``finite_diff`` est l'une des methodes numeriques les plus traditionnelles et les plus courantes pour estimer le gradient d'une fonction. L'idee principale est de remplacer les derivees partielles par des differences :

.. math::

    f^{\prime}(x)=\lim _{h \rightarrow 0} \frac{f(x+h)-f(x)}{h}


Pour la methode ``parameter-shift``, nous utilisons la fonction objectif, telle que :

.. math:: O(\theta)=\left\langle 0\left|U^{\dagger}(\theta) H U(\theta)\right| 0\right\rangle

Il est theoriquement possible de calculer le gradient des parametres par rapport au Hamiltonien dans un circuit quantique en utilisant la methode plus precise : ``parameter-shift``.

.. math::

    \nabla O(\theta)=
    \frac{1}{2}\left[O\left(\theta+\frac{\pi}{2}\right)-O\left(\theta-\frac{\pi}{2}\right)\right]

.. py:class:: pyvqnet.qnn.quantumlayer.QuantumLayer(qprog_with_measure,para_num,machine_type_or_cloud_token,num_of_qubits:int,num_of_cbits:int = 1,diff_method:str = "parameter_shift",delta:float = 0.01, dtype=None, name='')

    Module de calcul abstrait pour les circuits quantiques variationnels. Il simule un circuit quantique parametre et obtient le resultat de mesure.
    QuantumLayer herite de Module, ce qui lui permet de calculer les gradients des parametres du circuit, d'entrainer un modele de circuits quantiques variationnels ou d'integrer des circuits quantiques variationnels dans un modele hybride quantique-classique.
    
    Cette classe ne necessite pas d'initialiser la machine virtuelle dans la fonction ``qprog_with_measure``.

    :param qprog_with_measure: fonctions de circuits quantiques appelables, construites avec pyQPanda2
    :param para_num: `int` - Nombre de parametres
    :param machine_type_or_cloud_token: type de machine qpanda ou jeton QCLOUD pyQPanda2
    :param num_of_qubits: nombre de qubits
    :param num_of_cbits: nombre de bits classiques
    :param diff_method: 'parameter_shift' ou 'finite_diff'
    :param delta:  delta pour la difference
    :param dtype: Type de donnees du parametre, defaut: None, utilise le type par defaut kfloat32, qui represente un nombre a virgule flottante 32 bits.
    :param name: nom de la couche de sortie

    :return: un module capable de calculer des circuits quantiques.

    .. note::
        qprog_with_measure est une fonction de circuit quantique definie dans pyQPanda2.

        Cette fonction doit contenir les parametres suivants, sinon elle ne peut pas fonctionner correctement dans QuantumLayer.

        qprog_with_measure (input,param,qubits,cbits,m_machine)

            `input`: donnees classiques d'entree 1-dimensionnelles de type array_like

            `param`: parametres du circuit quantique 1-dimensionnels de type array_like

            `qubits`: qubits alloues par QuantumLayer

            `cbits`: bits classiques alloues par QuantumLayer. Si votre circuit n'utilise pas de cbits, vous devez tout de meme reserver ce parametre.

            `m_machine`: simulateur cree par QuantumLayer

        Utilisez l'attribut ``m_para`` de QuantumLayer pour obtenir les parametres d'entrainement du circuit quantique variable. Le parametre est de classe ``QTensor``, qui peut etre converti en tableau numpy en utilisant l'interface ``to_numpy()``.

    .. note::

        Cette classe a un alias : `QpandaQCircuitVQCLayer` .

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
            # pauli_dict = {'Z0 X1':10,'Y2':-0.543}
            rlt_prob = ProbsMeasure([0,2],prog,m_machine,qubits)
            return rlt_prob

        pqc = QuantumLayer(pqctest,3,"cpu",4,1)
        # donnees classiques en entree
        input = QTensor([[1,2,3,4],[40,22,2,3],[33,3,25,2.0]] )
        # propagation avant du circuit
        rlt = pqc(input)
        grad =  QTensor(np.ones(rlt.data.shape)*1000)
        # propagation arriere du circuit
        rlt.backward(grad)
        print(rlt)
        # [
        # [0.2500000, 0.2500000, 0.2500000, 0.2500000],
        # [0.2500000, 0.2500000, 0.2500000, 0.2500000],
        # [0.2500000, 0.2500000, 0.2500000, 0.2500000]
        # ]

QuantumLayerV2
=================================

Si vous etes plus familier avec la syntaxe de pyQPanda2, veuillez utiliser la classe QuantumLayerV2. Vous pouvez definir la fonction de circuit quantique en utilisant ``qubits``, ``cbits`` et ``machine``, puis la passer comme argument ``qprog_with_measure`` de QuantumLayerV2.

.. py:class:: pyvqnet.qnn.quantumlayer.QuantumLayerV2(qprog_with_measure, para_num, diff_method: str = 'parameter_shift', delta: float = 0.01, dtype=None, name='')

    Module de calcul abstrait pour les circuits quantiques variationnels. Il simule un circuit quantique parametre et obtient le resultat de mesure.
    QuantumLayer herite de Module, ce qui lui permet de calculer les gradients des parametres du circuit, d'entrainer un modele de circuits quantiques variationnels ou d'integrer des circuits quantiques variationnels dans un modele hybride quantique-classique.
    
    Pour utiliser ce module, vous devez creer votre machine virtuelle quantique et allouer des qubits et des cbits.

    :param qprog_with_measure: fonctions de circuits quantiques appelables, construites avec pyQPanda2
    :param para_num: `int` - Nombre de parametres
    :param diff_method: 'parameter_shift' ou 'finite_diff'
    :param delta:  delta pour la difference
    :param dtype: Type de donnees du parametre, defaut: None, utilise le type par defaut kfloat32, qui represente un nombre a virgule flottante 32 bits.
    :param name: nom de la couche de sortie
    :return: un module capable de calculer des circuits quantiques.

    .. note::
        qprog_with_measure est une fonction de circuit quantique definie dans pyQPanda.

        Cette fonction doit contenir les parametres suivants, sinon elle ne peut pas fonctionner correctement dans QuantumLayerV2.

        Par rapport a QuantumLayer, vous devez allouer vous-meme les qubits et le simulateur,

        et vous devrez peut-etre aussi allouer des cbits si qprog_with_measure necessite une mesure quantique.

        qprog_with_measure (input,param)

        `input`: donnees classiques d'entree 1-dimensionnelles de type array_like

        `param`: parametres du circuit quantique 1-dimensionnels de type array_like

    .. note::

        Cette classe a un alias : `QpandaQCircuitVQCLayerLite` .

    Example::

        import pyqpanda as pq
        from pyvqnet.qnn.measure import ProbsMeasure
        from pyvqnet.qnn.quantumlayer import QuantumLayerV2
        import numpy as np
        from pyvqnet.tensor import QTensor
        def pqctest (input,param):
            num_of_qubits = 4

            m_machine = pq.CPUQVM()# exterieur
            m_machine.init_qvm()# exterieur
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

        # donnees classiques en entree
        input = QTensor([[1,2,3,4],[4,2,2,3],[3,3,2,2.0]] )

        # propagation avant du circuit
        rlt = pqc(input)

        grad =  QTensor(np.ones(rlt.data.shape)*1000)
        # propagation arriere du circuit
        rlt.backward(grad)
        print(rlt)

        # [
        # [0.2500000, 0.2500000, 0.2500000, 0.2500000],
        # [0.2500000, 0.2500000, 0.2500000, 0.2500000],
        # [0.2500000, 0.2500000, 0.2500000, 0.2500000]
        # ]

 


NoiseQuantumLayer
=================================

Dans un ordinateur quantique reel, en raison des caracteristiques physiques du bit quantique, il existe toujours des erreurs de calcul inevitables. Afin de mieux simuler cette erreur dans la machine virtuelle quantique, VQNet prend egalement en charge la machine virtuelle quantique avec bruit. La simulation de la machine virtuelle quantique avec bruit est plus proche de l'ordinateur quantique reel. Nous pouvons personnaliser le type de porte logique prise en charge et le modele de bruit associe.
Le modele de bruit quantique actuellement pris en charge est defini dans ``NoiseQVM`` de pyQPanda2.

Nous pouvons utiliser ``NoiseQuantumLayer`` pour definir une differenciation automatique des circuits quantiques. ``NoiseQuantumLayer`` prend en charge la machine virtuelle quantique pyQPanda2 avec bruit. Vous pouvez definir une fonction comme argument ``qprog_with_measure``. Cette fonction doit contenir le circuit quantique defini par pyQPanda, et vous devez egalement passer un argument ``noise_set_config`` pour configurer le modele de bruit via l'interface pyQPanda.

.. py:class:: pyvqnet.qnn.quantumlayer.NoiseQuantumLayer(qprog_with_measure, para_num, machine_type, num_of_qubits: int, num_of_cbits: int = 1, diff_method: str = 'parameter_shift', delta: float = 0.01, noise_set_config=None, dtype=None, name='')

    Module de calcul abstrait pour les circuits quantiques variationnels. Il simule un circuit quantique parametre et obtient le resultat de mesure.
    QuantumLayer herite de Module, ce qui lui permet de calculer les gradients des parametres du circuit, d'entrainer un modele de circuits quantiques variationnels ou d'integrer des circuits quantiques variationnels dans un modele hybride quantique-classique.
    
    Ce module doit etre initialise avec un modele de bruit via ``noise_set_config``.

    :param qprog_with_measure: fonctions de circuits quantiques appelables, construites avec pyQPanda2
    :param para_num: `int` - Nombre de parametres
    :param machine_type: type de machine pyQPanda2
    :param num_of_qubits: nombre de qubits
    :param num_of_cbits: nombre de bits classiques
    :param diff_method: 'parameter_shift' ou 'finite_diff'
    :param delta:  delta pour la difference
    :param noise_set_config: fonction de configuration du bruit
    :param dtype: Type de donnees du parametre, defaut: None, utilise le type par defaut kfloat32, qui represente un nombre a virgule flottante 32 bits.
    :param name: nom de la couche de sortie
    
    :return: un module capable de calculer des circuits quantiques avec un modele de bruit.

    .. note::
        qprog_with_measure est une fonction de circuit quantique definie dans pyQPanda.

        Cette fonction doit contenir les parametres suivants, sinon elle ne peut pas fonctionner correctement dans NoiseQuantumLayer.

        qprog_with_measure (input,param,qubits,cbits,m_machine)

        `input`: donnees classiques d'entree 1-dimensionnelles de type array_like

        `param`: parametres du circuit quantique 1-dimensionnels de type array_like

        `qubits`: qubits alloues par NoiseQuantumLayer

        `cbits`: bits classiques alloues par NoiseQuantumLayer. Si votre circuit n'utilise pas de cbits, vous devez tout de meme reserver ce parametre.

        `m_machine`: simulateur cree par NoiseQuantumLayer

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
            # Calculer les probabilites pour chaque etat
            probabilities = counts / 100
            # Obtenir l'esperance de l'etat
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

Voici un exemple de ``noise_set_config``. Nous ajoutons ici le modele de bruit BITFLIP_KRAUS_OPERATOR avec l'argument de bruit p=0.01 aux portes quantiques ``RX``, ``RY``, ``RZ``, ``X``, ``Y``, ``Z``, ``H``, etc.

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

    Une couche d'encapsulation pour implementer la propagation avant et arriere avec des circuits Qiskit dans VQNet. QISKIT_VQC est une classe qui definit un circuit quantique Qiskit et sa fonction d'execution.
    L'exemple suivant montre son fonctionnement. Cette couche ne prend en charge que les entrees du circuit et les poids comme parametres.
    
    :param cirq_vqc: Une classe qui definit la definition, le backend et les fonctions d'execution d'un circuit Qiskit.
    :param para_num: `int` - Le nombre de parametres.
    :return: Une classe capable d'executer des modeles de circuits quantiques qiskit.

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
                # --- Definition du circuit ---

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

        # definir la classe de circuits qiskit
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

    Une couche d'encapsulation de circuit Cirq pour implementer la propagation avant et arriere dans vqnet. CIRQ_VQC est une classe qui necessite que les utilisateurs definissent un circuit quantique Cirq et sa fonction `run`. L'exemple suivant montre son principe de fonctionnement.
    Cette couche ne prend en charge que les entrees du circuit et les poids comme parametres.

    :param cirq_vqc: Une classe definissant la definition, le backend et les fonctions d'execution d'un circuit Cirq.
    :param para_num: `int` - Le nombre de parametres.
    :return: Une classe capable d'executer le modele de circuit quantique Cirq.


    .. note::

        Le code d'exemple suivant necessite `cirq==1.5.0, numpy <2`.

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
                # definir les qubits
                q0 = cirq.NamedQubit ('q0')
                q1 = cirq.NamedQubit ('q1')
                q2 = cirq.NamedQubit ('q2')
                q3 = cirq.NamedQubit ('q3')
                qubits = [q0,q1,q2,q3]
                self.qubits = [q0,q1,q2,q3]
                ### definir les parametres variationnels
                param = sympy.symbols(f'theta(0:24)')
                self.theta = np.asarray(param).reshape((4,6))

                ### definir les circuits
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
                
                ### definir le backend
                self._backend = simulator

                self._param_symbols_list,self._input_symbols_list = get_circuit_symbols(self._circuit)


            def run(self,resolver,init_state):

                rlt = self._backend.simulate(self._circuit,resolver,initial_state=init_state).final_state_vector
                z0 = cirq.Z(self.qubits[0])

                qubit_map={self.qubits[0]: 0}
                
                expectation = z0.expectation_from_state_vector(rlt, qubit_map).real

                return expectation

        # definir la classe de circuits cirq
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

Portes quantiques
***********************************

La maniere de traiter les qubits est appelee portes quantiques. En utilisant des portes quantiques, nous faisons evoluer les etats quantiques de maniere controlee. Les portes quantiques sont la base des algorithmes quantiques.

Portes quantiques de base
=================================

Dans VQNet, nous utilisons chaque porte logique de pyQPanda developpee par Origin Quantum pour construire des circuits quantiques et effectuer des simulations quantiques.
Les portes actuellement prises en charge par pyQPanda peuvent etre definies dans la section des portes quantiques de pyQPanda.
De plus, VQNet encapsule egalement certaines combinaisons de portes quantiques couramment utilisees en apprentissage automatique quantique.



AmplitudeEmbeddingCircuit
=================================

.. py:function:: pyvqnet.qnn.template.AmplitudeEmbeddingCircuit(input_feat, qubits)

    Encode :math:`2^n` caracteristiques dans le vecteur d'amplitude de :math:`n` qubits.
    Pour representer un vecteur d'etat quantique valide, la norme L2 de ``features`` doit etre egale a un.

    :param input_feat: tableau numpy representant les parametres
    :param qubits: qubits alloues par pyQPanda
    :return: circuits quantiques

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




API d'apprentissage automatique quantique avec pyQPanda2
***************************************************************************

Reseaux antagonistes generatifs quantiques pour l'apprentissage et le chargement de distributions aleatoires
====================================================================================================================

L'algorithme des reseaux antagonistes generatifs quantiques (`QGAN <https://www.nature.com/articles/s41534-019-0223-2>`_) utilise des circuits quantiques variationnels purs pour preparer les etats quantiques generes avec une distribution aleatoire specifique, ce qui peut reduire le nombre de portes logiques necessaires pour generer des etats quantiques specifiques et reduire la complexite des circuits quantiques. Il utilise la structure classique du modele GAN, qui comporte deux sous-modeles : le Generateur et le Discriminateur. Le Generateur produit une distribution specifique pour le circuit quantique. Et le Discriminateur distingue les echantillons de donnees generes par le Generateur des echantillons reels de donnees d'entrainement distribues aleatoirement.
Voici un exemple de VQNet implementant l'apprentissage et le chargement de distributions aleatoires par QGAN, base sur l'article `Quantum Generative Adversarial Networks for learning and loading random distributions <https://www.nature.com/articles/s41534-019-0223-2>`_ de Christa Zoufal.

.. image:: ./images/qgan-arch.PNG
   :width: 600 px
   :align: center

|

Afin de realiser la construction de la classe ``QGANAPI`` du reseau antagoniste generatif quantique par VQNet, le generateur quantique est utilise pour preparer l'etat initial des donnees distribuees reellement. Le nombre de bits quantiques est de 3, et le nombre de repetitions du module de circuit parametrique interne du generateur quantique est de 1. Par ailleurs, la divergence KL est utilisee comme metrique pour le chargement de la distribution aleatoire par QGAN.

.. code-block::

    import pickle
    import os
    import pyqpanda as pq
    from pyvqnet.qnn.qgan.qgan_utils import QGANAPI
    import numpy as np

    num_of_qubits = 3  # configuration de l'article
    rep = 1
    number_of_data = 10000
    # Charger des echantillons de donnees provenant de differentes distributions
    mu = 1
    sigma = 1
    real_data = np.random.lognormal(mean=mu, sigma=sigma, size=number_of_data)


    # initialisation
    save_dir = None
    qgan_model = QGANAPI(
        real_data,
        # distribution de donnees generees par numpy, 1-dimension.
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

Voici le module ``train`` de QGAN.

.. code-block::

    # entrainement
    qgan_model.train()  # entrainer le qgan


Le module ``eval`` de QGAN est concu pour tracer la courbe de la fonction de perte et le diagramme de distribution de probabilite entre la distribution aleatoire preparee par QGAN et la distribution reelle.

.. code-block::

    # afficher la fonction de distribution de probabilite de la distribution generee et de la distribution reelle
    qgan_model.eval(real_data)  # dessiner la pdf

Le module ``get_trained_quantum_parameters`` de QGAN est utilise pour obtenir les parametres d'entrainement et les sortir sous forme de tableau numpy. Si ``save_DIR`` n'est pas vide, les parametres d'entrainement sont sauvegardes dans un fichier. Le module ``Load_param_and_eval`` de QGAN charge les parametres d'entrainement, et le module ``get_circuits_with_trained_param`` obtient le circuit pyQPanda genere par le generateur quantique apres l'entrainement.

.. code-block::

    # obtenir les parametres quantiques entrainés
    param = qgan_model.get_trained_quantum_parameters()
    print(f" trained param {param}")

    # charger les fichiers de parametres sauvegardes
    if save_dir is not None:
        path = os.path.join(
            save_dir, qgan_model._start_time + "trained_qgan_param.pickle")
        with open(path, "rb") as file:
            t3 = pickle.load(file)
        param = t3["quantum_parameters"]
        print(f" trained param {param}")

    # afficher la fonction de distribution de probabilite de la distribution generee et de la distribution reelle
    qgan_model.load_param_and_eval(param)

    # calculer la metrique
    print(qgan_model.eval_metric(param, "kl"))

    # obtenir le circuit quantique du generateur
    m_machine = pq.CPUQVM()
    m_machine.init_qvm()
    qubits = m_machine.qAlloc_many(num_of_qubits)
    qpanda_cir = qgan_model.get_circuits_with_trained_param(qubits)
    print(qpanda_cir)

En general, l'apprentissage et le chargement de distribution aleatoire par QGAN necessite plusieurs entrainements de modeles avec differentes graines aleatoires pour obtenir les resultats attendus. Par exemple, voici le graphique de la fonction de distribution de probabilite entre la distribution lognormale implementee par QGAN et la distribution lognormale reelle, ainsi que la courbe de la fonction de perte entre le generateur et le discriminateur de QGAN.

.. image:: ./images/qgan-loss.png
   :width: 600 px
   :align: center

|

.. image:: ./images/qgan-pdf.png
   :width: 600 px
   :align: center

|


SVM a noyau quantique
=================================

Dans les tâches d'apprentissage automatique, les donnees ne peuvent souvent pas etre separees par un hyperplan dans l'espace d'origine. Une technique courante pour trouver de tels hyperplans consiste a appliquer une fonction de transformation non lineaire aux donnees.
Cette fonction est appelee carte de caracteristiques, grace a laquelle nous pouvons calculer la proximite des points de donnees dans ce nouvel espace de caracteristiques pour la tâche de classification de l'apprentissage automatique.

Cet exemple fait reference a l'article : `Supervised learning with quantum enhanced feature spaces <https://arxiv.org/pdf/1804.11326.pdf>`_.
La premiere methode construit des circuits variationnels pour les tâches de classification de donnees.

``gen_vqc_qsvm_data`` genere les donnees necessaires pour cet exemple. ``vqc_qsvm`` est une classe de sous-circuit variable utilisee pour classer les donnees d'entree.
La fonction ``vqc_qsvm.plot()`` visualise la distribution des donnees.

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
        # nombre de repetitions des sous-circuits
        rep = 3

        # definit la classe QSVM
        VQC_QSVM = vqc_qsvm(batch_size, maxiter, rep)
        # genere aleatoirement les donnees de l'article
        train_features, test_features, train_labels, test_labels, samples = \
            gen_vqc_qsvm_data(training_size=training_size, test_size=test_size, gap=gap)
        VQC_QSVM.plot(train_features, test_features, train_labels, test_labels, samples)
        # entrainement
        VQC_QSVM.train(train_features, train_labels)
        # test
        rlt, acc_1 = VQC_QSVM.predict(test_features, test_labels)
        print(f"testing_accuracy {acc_1}")


En plus de l'utilisation directe mentionnee ci-dessus de circuits quantiques variationnels pour mapper les caracteristiques de donnees classiques vers des espaces de caracteristiques quantiques, l'article `Supervised learning with quantum enhanced feature spaces <https://arxiv.org/pdf/1804.11326.pdf>`_,
introduit egalement la methode d'estimation directe des fonctions de noyau a l'aide de circuits quantiques et de classification a l'aide de machines a vecteurs de support classiques.
Par analogie avec diverses fonctions de noyau dans le SVM classique :math:`K(i,j)`, utilisez la fonction de noyau quantique pour definir le produit scalaire des donnees classiques dans l'espace de caracteristiques quantique :math:`\phi(\mathbf{x}_i)` :

.. math::
    |\langle \phi(\mathbf{x}_j) | \phi(\mathbf{x}_i) \rangle |^2 =  |\langle 0 | U^\dagger(\mathbf{x}_j) U(\mathbf{x}_i) | 0 \rangle |^2

En utilisant VQNet et pyQPanda, nous definissons ``QuantumKernel_VQNet`` pour generer une fonction de noyau quantique et utilisons ``sklearn`` pour la classification :

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


Optimiseur par approximation stochastique par perturbation simultanee (SPSA)
=============================================================================


.. py:function:: pyvqnet.qnn.SPSA(maxiter: int = 1000, save_steps: int = 1, last_avg: int = 1, c0: float = _C0, c1: float = 0.2, c2: float = 0.602, c3: float = 0.101, c4: float = 0, init_para=None, model=None, calibrate_flag=False)
    

    Optimiseur par approximation stochastique par perturbation simultanee (SPSA).

    SPSA fournit une methode stochastique pour approximer le gradient d'une fonction de cout differentiable multivariee.
    Pour ce faire, la fonction de cout est evaluee deux fois en utilisant un vecteur de parametres perturbe : chaque composante du vecteur de parametres original est simultanement decalee par une valeur generee aleatoirement.
    Des informations supplementaires sont disponibles sur le `site web SPSA <http://www.jhuapl.edu/SPSA>`__.

    :param maxiter: Le nombre maximal d'iterations a effectuer. Valeur par defaut : 1000.
    :param save_steps: Sauvegarder les informations intermediaires toutes les save_steps etapes. Valeur par defaut : 1.
    :param last_avg: Parametre de moyennage pour les last_avg dernieres iterations.
        Si last_avg = 1, seule la derniere iteration est prise en compte. Valeur par defaut : 1.
    :param c0: a initial. Taille du pas pour la mise a jour des parametres. Valeur par defaut : 0.2*pi
    :param c1: c initial. Taille du pas utilisee pour approximer le gradient. Valeur par defaut : 0.1.
    :param c2: alpha de l'article, utilise pour ajuster a(c0) a chaque iteration. Valeur par defaut : 0.602.
    :param c3: gamma de l'article, utilise pour ajuster c(c1) a chaque iteration. Valeur par defaut : 0.101.
    :param c4: Egalement utilise pour controler les parametres de a. Valeur par defaut : 0.
    :param init_para: Parametres d'initialisation. Valeur par defaut : None.
    :param model: Modele parametrique : modele. Valeur par defaut : None.
    :param calibrate_flag: Indique s'il faut calibrer les hyperparametres a et c. Valeur par defaut : False.

    :return: une instance d'optimiseur SPSA


    .. warning::

        SPSA ne prend en charge que les parametres 1-dimensionnels.

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

    utilise SPSA pour optimiser les donnees d'entree.

    :param input_data: donnees d'entree
    :return:

        train_para: parametre final

        theta_best: La moyenne des parametres apres les dernieres `last_avg` iterations.

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

Matrice de calcul de l'information de Fisher quantique
========================================================

.. py:class:: pyvqnet.qnn.opt.quantum_fisher(py_qpanda_config, params, target_gate_type_lists,target_gate_bits_lists, qcir_lists, wires)
    
    Retourne la matrice d'information de Fisher quantique pour un circuit quantique.

    .. math::

        \mathrm{QFIM}_{i, j}=4 \operatorname{Re}\left[\left\langle\partial_i \psi(\boldsymbol{\theta}) \mid \partial_j \psi(\boldsymbol{\theta})\right\rangle-\left\langle\partial_i \psi(\boldsymbol{\theta}) \mid \psi(\boldsymbol{\theta})\right\rangle\left\langle\psi(\boldsymbol{\theta}) \mid \partial_j \psi(\boldsymbol{\theta})\right\rangle\right]

    La version courte est :math::math:`\left|\partial_j \psi(\boldsymbol{\theta})\right\rangle:=\frac{\partial}{\partial \theta_j}|\psi(\boldsymbol{\theta})\rangle`.

    .. note::

        Actuellement, seuls RX, RY, RZ sont pris en charge.

    :param params: Parametres variables dans les circuits.
    :param target_gate_type_lists: Prend en charge "RX", "RY", "RZ" ou des listes.
    :param target_gate_bits_lists: Sur quel bit quantique ou quels bits la porte parametree agit.
    :param qcir_lists: La liste des circuits quantiques avant la porte parametree cible pour calculer le tenseur metrique, voir l'exemple suivant.
    :param wires: Index total des bits quantiques pour les circuits quantiques.

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

        # L'exemple ci-dessus montre qu'il n'y a pas de portes identiques dans la meme couche,
        # mais dans la meme couche, vous devez modifier les portes logiques selon l'exemple suivant.
        
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
            qcir.insert(pq.CNOT(config._qubits[0], config._qubits[1])) #  partie 01
            
            qcir.insert(pq.RZ(config._qubits[1], params[2]))  #  partie 02
            
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

        mt = quantum_fisher(config, params2, [['RX', 'RY'], ['RZ'], ['RZ']], # rx,ry comptent comme couche une, premier rz comme couche deux, deuxieme rz comme couche trois.
                                [[0, 1], [1], [1]], qcir, [0, 1])

