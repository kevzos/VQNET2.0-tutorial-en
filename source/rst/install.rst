Passaggi per l'installazione di VQNet
=====================================

Installazione del pacchetto Python VQNet
-----------------------------------------

Forniamo pacchetti Python precompilati per l'installazione su Linux, Windows, macOS 13+ (arm64), compatibili con **Python 3.10**.

Scaricare l'archivio corrispondente dal sito web ufficiale, estrarlo, navigare nella directory estratta ed eseguire i seguenti passaggi:

.. code-block::

    # Per Windows
    ./install.bat
    # Per macOS e Linux
    ./install.sh
 


Per i sistemi Windows e Linux, il pacchetto pyvqnet include funzionalità di accelerazione integrate per calcoli di reti neurali classiche basate su Nvidia CUDA, che dipendono dalla versione specifica delle librerie runtime NVIDIA CUDA 12.6 (installate automaticamente con il pacchetto).
Il pacchetto è ottimizzato per le seguenti architetture CUDA:
**sm_80** (GPU per data center NVIDIA A100, serie A30) e **sm_86** (GPU consumer NVIDIA GeForce serie RTX 30). Assicurarsi di utilizzare una GPU che supporti queste architetture; in caso contrario, il programma potrebbe non funzionare correttamente.

    .. important::

        Si noti che, poiché questo pacchetto non distingue tra versioni CPU/GPU, dipende dalle librerie runtime NVIDIA CUDA in Windows e Linux, che vengono installate automaticamente con il pacchetto. Ciò potrebbe causare conflitti con altri software che dipendono da versioni diverse.


Validazione dell'installazione di VQNet
----------------------------------------

.. code-block::

    import pyvqnet
    from pyvqnet.tensor import *
    a = arange(1,25).reshape([2, 3, 4])
    print(a)

Test della funzionalità GPU in VQNet
----------------------------------------

.. code-block::

    from pyvqnet import DEV_GPU_0
    from pyvqnet.tensor import *
    a = ones([4,5],device = DEV_GPU_0)
    print(a)

Un caso semplice con VQNet
--------------------------
Qui presentiamo un caso composto da moduli di reti neurali classiche e moduli quantistici di VQNet per descrivere il flusso di lavoro dell'apprendimento automatico quantistico.
Si riferisce a `Data re-uploading for a universal quantum classifier <https://arxiv.org/abs/1907.02085>`_ .
Generalmente, le seguenti parti del modulo di calcolo quantistico sono presenti nell'apprendimento automatico quantistico:

(1) Encoder: codifica dei dati classici in stato quantistico;

(2) Ansatz: addestramento dei parametri nelle porte quantistiche parametrizzate;

(3) Misurazione: misurazione del valore di un qubit (proiezione dello stato quantistico del qubit su un asse specificato).

Il modulo di calcolo quantistico è la base teorica del modello ibrido di rete neurale quantistico-classica, anch'esso differenziabile come i moduli di reti neurali classiche. VQNet supporta moduli di calcolo quantistico e moduli di calcolo classico per formare un modello ibrido di apprendimento automatico, e fornisce vari algoritmi di ottimizzazione per l'ottimizzazione dei parametri (ad es. strato di convoluzione, strato di pooling, strato completamente connesso, funzione di attivazione, ecc.).

.. figure:: ./images/classic-quantum.PNG

Nel modulo di calcolo quantistico, VQNet supporta l'uso del pacchetto software di calcolo quantistico efficiente pyqpanda3 per costruire moduli quantistici.
Utilizzando le varie interfacce di uso comune fornite da pyqpanda3, gli utenti possono costruire rapidamente moduli di calcolo quantistico.

Il seguente esempio utilizza pyqpanda3 per costruire un modulo di calcolo quantistico. Attraverso VQNet, questo modulo quantistico può essere direttamente incorporato in un modello ibrido di apprendimento automatico per l'addestramento dei parametri del circuito quantistico.
In questo esempio, viene utilizzato 1 qubit, più porte di rotazione parametrizzate `RZ`, `RY`, `RZ` sono usate per codificare l'input x, e la funzione `probs_measure` è usata per osservare il risultato della misurazione di probabilità del qubit come output.

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

Il nostro compito è classificare questi dati generati casualmente utilizzando un approccio di classificazione binaria. In questo compito,
il centro di un cerchio si trova nell'origine, i punti entro il raggio 1 colorati in rosso appartengono a una categoria, e i campioni colorati in blu appartengono a un'altra categoria.

.. figure:: ./images/origin_circle.png

La pipeline del processo di addestramento

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







