Schritte zur VQNet-Installation
==================================

Installation des Python-Pakets VQNet
-------------------------------------

Wir bieten vorkompilierte Python-Pakete für die Installation unter Linux, Windows, macOS 13+ (arm64) an, die **Python 3.10** unterstützen.

Laden Sie das entsprechende Archiv von der offiziellen Website herunter, extrahieren Sie es, navigieren Sie zum extrahierten Verzeichnis und führen Sie die folgenden Schritte aus:

.. code-block::

    # Für Windows
    ./install.bat
    # Für macOS und Linux
    ./install.sh
 


Für Windows- und Linux-Systeme enthält das pyvqnet-Paket integrierte Beschleunigungsfunktionen für klassische neuronale Netzberechnungen basierend auf Nvidia CUDA, die von der spezifischen Version der NVIDIA CUDA 12.6-Laufzeitbibliotheken abhängen (automatisch mit dem Paket installiert).
Das Paket ist für die folgenden CUDA-Architekturen optimiert:
**sm_80** (NVIDIA A100, A30 Serie Rechenzentrums-GPUs) und **sm_86** (NVIDIA GeForce RTX 30 Serie Verbraucher-GPUs). Bitte stellen Sie sicher, dass Sie eine GPU verwenden, die diese Architekturen unterstützt; andernfalls funktioniert das Programm möglicherweise nicht richtig.

    .. important::

        Bitte beachten Sie, dass dieses Paket nicht zwischen CPU-/GPU-Versionen unterscheidet und daher unter Windows und Linux von den NVIDIA CUDA-Laufzeitbibliotheken abhängt, die automatisch mit dem Paket installiert werden. Dies kann zu Konflikten mit anderer Software führen, die von anderen Versionen abhängt.


VQNet-Installation validieren
----------------------------------

.. code-block::

    import pyvqnet
    from pyvqnet.tensor import *
    a = arange(1,25).reshape([2, 3, 4])
    print(a)

Testen der GPU-Funktionalität in VQNet
----------------------------------------

.. code-block::

    from pyvqnet import DEV_GPU_0
    from pyvqnet.tensor import *
    a = ones([4,5],device = DEV_GPU_0)
    print(a)

Ein einfaches Beispiel mit VQNet
--------------------------------
Hier stellen wir ein Beispiel vor, das aus klassischen neuronalen Netzwerkmodulen und Quantenmodulen von VQNet besteht, um den Arbeitsablauf des quantenmaschinellen Lernens zu beschreiben.
Es bezieht sich auf `Data re-uploading for a universal quantum classifier <https://arxiv.org/abs/1907.02085>`_ .
Im Allgemeinen gibt es die folgenden Teile des Quantenberechnungsmoduls im quantenmaschinellen Lernen:

(1) Encoder: Codierung klassischer Daten in einen Quantenzustand;

(2) Ansatz: Training der Parameter in parametrisierten Quantengattern;

(3) Messung: Messung des Wertes eines Qubits (Projektion des Quantenzustands des Qubits auf eine bestimmte Achse).

Das Quantenberechnungsmodul ist die theoretische Grundlage des hybriden Modells aus quantenmechanischem und klassischem neuronalem Netzwerk, das ebenfalls differenzierbar ist wie klassische neuronale Netzwerkmodule. VQNet unterstützt Quantenberechnungsmodule und klassische Berechnungsmodule, um ein hybrides Modell für maschinelles Lernen zu bilden, und bietet verschiedene Optimierungsalgorithmen zur Parameteroptimierung (z. B. Faltungsschicht, Pooling-Schicht, vollständig verbundene Schicht, Aktivierungsfunktion usw.).

.. figure:: ./images/classic-quantum.PNG

Im Quantenberechnungsmodul unterstützt VQNet die Verwendung des effizienten Quantensoftware-Pakets pyqpanda3 zum Erstellen von Quantenmodulen.
Mit den verschiedenen häufig verwendeten Schnittstellen von pyqpanda3 können Benutzer schnell Quantenberechnungsmodule erstellen.

Das folgende Beispiel verwendet pyqpanda3, um ein Quantenberechnungsmodul zu erstellen. Über VQNet kann dieses Quantenmodul direkt in ein hybrides Modell für maschinelles Lernen zum Training von Quantenschaltkreisparametern eingebettet werden.
In diesem Beispiel wird 1 Qubit verwendet, mehrere parametrisierte Rotationsgatter `RZ`, `RY`, `RZ` werden zur Codierung der Eingabe x verwendet, und die Funktion `probs_measure` wird verwendet, um das Wahrscheinlichkeitsmessungsergebnis des Qubits als Ausgabe zu beobachten.

.. code-block::

    import pyqpanda3.core as pq
    from pyvqnet.qnn.pq3 import probs_measure
    def qdrl_circuit(input,weights):
        qlist = range(1)
        machine = pq.CPUQVM()
        x1 = input.squeeze()
        param1 = weights.squeeze()
        # Build quantum circuit instance using pyqpanda3 interface
        circuit = pq.QCircuit()
        # Insert RZ gate on the first qubit with parameter x1[0]
        circuit << pq.RZ(qlist[0], x1[0])
        # Insert RY gate on the first qubit with parameter x1[1]
        circuit << pq.RY(qlist[0], x1[1])
        # Insert RZ gate on the first qubit with parameter x1[2]
        circuit << pq.RZ(qlist[0], x1[2])
        # Insert RZ gate on the first qubit with parameter param1[0]
        circuit << pq.RZ(qlist[0], param1[0])
        # Insert RY gate on the first qubit with parameter param1[1]
        circuit << pq.RY(qlist[0], param1[1])
        # Insert RZ gate on the first qubit with parameter param1[2]
        circuit << pq.RZ(qlist[0], param1[2])
        # Insert RZ gate on the first qubit with parameter x1[0]
        circuit << pq.RZ(qlist[0], x1[0])
        # Insert RY gate on the first qubit with parameter x1[1]
        circuit << pq.RY(qlist[0], x1[1])
        # Insert RZ gate on the first qubit with parameter x1[2]
        circuit << pq.RZ(qlist[0], x1[2])
        # Insert RZ gate on the first qubit with parameter param1[3]
        circuit << pq.RZ(qlist[0], param1[3])
        # Insert RY gate on the first qubit with parameter param1[4]
        circuit << pq.RY(qlist[0], param1[4])
        # Insert RZ gate on the first qubit with parameter param1[5]
        circuit << pq.RZ(qlist[0], param1[5])
        # Insert RZ gate on the first qubit with parameter x1[0]
        circuit << pq.RZ(qlist[0], x1[0])
        # Insert RY gate on the first qubit with parameter x1[1]
        circuit << pq.RY(qlist[0], x1[1])
        # Insert RZ gate on the first qubit with parameter x1[2]
        circuit << pq.RZ(qlist[0], x1[2])
        # Insert RZ gate on the first qubit with parameter param1[6]
        circuit << pq.RZ(qlist[0], param1[6])
        # Insert RY gate on the first qubit with parameter param1[7]
        circuit << pq.RY(qlist[0], param1[7])
        # Insert RZ gate on the first qubit with parameter param1[8]
        circuit << pq.RZ(qlist[0], param1[8])
        # Build quantum program
        prog = pq.QProg()
        prog << circuit
        # Get probability measurement
        prob = probs_measure(machine ,prog,  qlist)

        return prob

Unsere Aufgabe ist es, diese zufällig generierten Daten mit einem binären Klassifikationsansatz zu klassifizieren. In dieser Aufgabe
befindet sich der Mittelpunkt eines Kreises im Ursprung, Punkte innerhalb des Radius 1, die rot gefärbt sind, gehören zu einer Kategorie, und die blau gefärbten Proben gehören zu einer anderen Kategorie.

.. figure:: ./images/origin_circle.png

Der Ablauf des Trainingsprozesses

.. code-block::

    # import required libraries and functions
    from pyvqnet.qnn.pq3.quantumlayer import QuantumLayer
    from pyvqnet.optim import adam
    from pyvqnet.nn.loss import CategoricalCrossEntropy
    from pyvqnet.tensor import QTensor
    import numpy as np
    from pyvqnet.nn.module import Module


Defining a model: the ``__init__`` function defines the internal neural network modules and quantum modules, and the ``forward`` function defines the forward computation. ``QuantumLayer`` is an abstract class
that encapsulates quantum computing.
VQNet will automatically calculate the parameter gradients for `qdrl_circuit` with `param_num`.


.. code-block::

    # number of parameters to be trained.
    param_num = 9
    # qubit number.
    qbit_num  = 1
    #define a model class inherits from Module.
    class Model(Module):
        def __init__(self):
            super(Model, self).__init__()
            #use QuantumLayer to embed quantum circuit into autodiff pipeline.
            self.pqc = QuantumLayer(qdrl_circuit,param_num)
        #define the forward function
        def forward(self, x):
            x = self.pqc(x)
            return x

Defining some functions for training the model 

.. code-block::

    # a function to generate the raw data randomly
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

    # a function to load data
    def get_minibatch_data(x_data, label, batch_size):
        for i in range(0,x_data.shape[0]-batch_size+1,batch_size):
            idxs = slice(i, i + batch_size)
            yield x_data[idxs], label[idxs]

    # a function to compute the accuracy
    def get_score(pred, label):
        pred, label = np.array(pred.data), np.array(label.data)
        pred = np.argmax(pred,axis=1)
        score = np.argmax(label,1)
        score = np.sum(pred == score)
        return score

VQNet follows the general workflow of machine learning: loading the data iteratively, forward propagation, calculating the loss function, back propagation, and updating the parameters.

.. code-block::

    # instantiating a model
    model = Model()
    # using Adam to define a optimizer
    optimizer = adam.Adam(model.parameters(),lr =0.6)
    # using cross-entropy to define a loss function
    Closs = CategoricalCrossEntropy()

A function to train the model

.. code-block::

    def train():
            
        #  generate data to be trained randomly   
        x_train, y_train = circle(500)
        x_train = np.hstack((x_train, np.zeros((x_train.shape[0], 1),dtype=np.float32)))  
        # define the number of data about each batch
        batch_size = 32
        # Maximum of training iteration times
        epoch = 10
        print("start training...........")
        for i in range(epoch):
            model.train()
            accuracy = 0
            count = 0
            loss = 0
            for data, label in get_minibatch_data(x_train, y_train,batch_size):
                # clear the gradients of optimizer
                optimizer.zero_grad()
                # forward computing
                output = model(data)
                # calculating loss function
                losss = Closs(label, output)
                # anti-propagation
                losss.backward()
                # update the optimizer parameters
                optimizer._step()
                # calculate the accuracy
                accuracy += get_score(output,label)

                loss += losss.item()
                count += batch_size
                
            print(f"epoch:{i}, train_accuracy:{accuracy/count}")
            print(f"epoch:{i}, train_loss:{loss/count}\n")
            
A function to validate the model

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







