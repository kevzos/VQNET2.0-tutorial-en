Quantenmaschinelles Lernen mit Qpanda
###############################################

Wir verwenden VQNet und pyqpanda2 oder pyqpanda3, um mehrere Beispiele für quantenmaschinelles Lernen zu implementieren.

.. warning::

    Der Quantencomputing-Teil der folgenden Schnittstelle verwendet möglicherweise pyqpanda2.

    Sie müssen pyqpanda zusätzlich installieren, `pip install pyqpanda`


Anwendung parametrisierter Quantenschaltungen bei Klassifikationsaufgaben
*****************************************************************************************************

1. QVC-Demo
========================================

Dieses Beispiel verwendet VQNet, um den Algorithmus aus der Arbeit `Circuit-centric quantum classifiers <https://arxiv.org/pdf/1804.00633.pdf>`_ zu implementieren.
Dieses Beispiel dient dazu, zu bestimmen, ob eine Binärzahl gerade oder ungerade ist. Durch Kodierung der Binärzahl auf das Qubit und Optimierung der variablen Parameter in der Schaltung
kann die Beobachtung in z-Richtung der Schaltung anzeigen, ob die Eingabe gerade oder ungerade ist.

Quantenschaltung
-------------------------------
Die variable Komponente definiert üblicherweise eine Unterschaltung, die eine grundlegende Schaltungsarchitektur darstellt, und komplexe Variationsschaltungen können durch wiederholte Schichten aufgebaut werden.
Unsere Schaltungsschicht besteht aus mehreren rotierenden Quantenlogikgattern und ``CNOT``-Quantenlogikgattern, die jedes Qubit mit seinen benachbarten Qubits verschränken.
Wir benötigen auch eine Schaltung, um klassische Daten in einen Quantenzustand zu kodieren, sodass der Ausgang der Schaltungsmessung mit der Eingabe zusammenhängt.
In diesem Beispiel kodieren wir die binäre Eingabe in der entsprechenden Reihenfolge auf die Qubits. Zum Beispiel wird der Eingabedatensatz 1101 in 4 Qubits kodiert.

.. math::

    x = 0101 \rightarrow|\psi\rangle=|0101\rangle

.. figure:: ./images/qvc_circuit.png
   :width: 600 px
   :align: center

|

Dieses Beispiel verwendet pyqpanda3.

.. code-block::

    import pyqpanda3.core as pq
    from pyvqnet.nn.module import Module
    from pyvqnet.optim.sgd import SGD
    from pyvqnet.nn.loss import CategoricalCrossEntropy
    from pyvqnet.tensor.tensor import QTensor
    from pyvqnet.data import data_generator as dataloader

    from pyvqnet.qnn.pq3.quantumlayer import QuantumLayer
    from pyvqnet.qnn.pq3.measure import probs_measure
    qnum = 4
    def qvc_circuits(input,weights):

        qlist = range(qnum)
        machine =pq.CPUQVM()
        def get_cnot(nqubits):
            cir = pq.QCircuit()
            for i in range(len(nqubits)-1):
                cir << pq.CNOT(nqubits[i],nqubits[i+1])
            cir << pq.CNOT(nqubits[len(nqubits)-1],nqubits[0])
            return cir

        def build_circuit(weights, xx, nqubits):

            def Rot(weights_j, qubits):
                circuit = pq.QCircuit()
                circuit << pq.RZ(qubits, weights_j[0])
                circuit << pq.RY(qubits, weights_j[1])
                circuit << pq.RZ(qubits, weights_j[2])
                return circuit
            def basisstate():
                circuit = pq.QCircuit()
                for i in range(len(nqubits)):
                    if xx[i] == 1:
                        circuit << pq.X(nqubits[i])
                return circuit

            circuit = pq.QCircuit()
            circuit << basisstate()

            for i in range(weights.shape[0]):

                weights_i = weights[i,:,:]
                for j in range(len(nqubits)):
                    weights_j = weights_i[j]
                    circuit << Rot(weights_j,nqubits[j])
                cnots = get_cnot(nqubits)
                circuit << cnots

            circuit << pq.Z(nqubits[0])

            prog = pq.QProg()
            prog << circuit
            return prog

        weights = weights.reshape([2,4,3])
        prog = build_circuit(weights,input,qlist)
        prob = probs_measure(machine,prog,qlist[0])

        return prob


Modellaufbau
-----------------------
Wir haben variable Quantenschaltungen ``qvc_circuits`` definiert.
Wir möchten sie in VQNets automatischem Differenzierungsframework verwenden,
um die Optimierungsfunktionen von VQNet für das Modelltraining zu nutzen.
Wir definieren eine Model-Klasse, die von der abstrakten Klasse ``Module`` erbt.
Das Modell verwendet die Klasse ``pyvqnet.qnn.pq3.QuantumLayer``, eine automatisch differenzierbare Quantenberechnungsschicht.
``qvc_circuits`` ist die Quantenschaltung, die wir ausführen möchten,
24 ist die Anzahl aller trainierbaren Quantenschaltungsparameter.

.. code-block::

     class Model(Module):
        def __init__(self):
            super(Model, self).__init__()
            self.qvc = QuantumLayer(qvc_circuits,24)

        def forward(self, x):
            return self.qvc(x)


Modelltraining und Test
------------------------------------
Wir verwenden vorab generierte Zufallsbinärzahlen und deren Gerade/Ungerade-Labels.
Die Daten sind wie folgt.

.. code-block::

    import numpy as np
    import os
    qvc_train_data = [0,1,0,0,1,
    0, 1, 0, 1, 0,
    0, 1, 1, 0, 0,
    0, 1, 1, 1, 1,
    1, 0, 0, 0, 1,
    1, 0, 0, 1, 0,
    1, 0, 1, 0, 0,
    1, 0, 1, 1, 1,
    1, 1, 0, 0, 0,
    1, 1, 0, 1, 1,
    1, 1, 1, 0, 1,
    1, 1, 1, 1, 0]
    qvc_test_data= [0, 0, 0, 0, 0,
    0, 0, 0, 1, 1,
    0, 0, 1, 0, 1,
    0, 0, 1, 1, 0]

    def get_data(dataset_str):
        if dataset_str == "train":
            datasets = np.array(qvc_train_data)

        else:
            datasets = np.array(qvc_test_data)

        datasets = datasets.reshape([-1,5])
        data = datasets[:,:-1]
        label = datasets[:,-1].astype(int)
        label = np.eye(2)[label].reshape(-1,2)
        return data, label

Modellvorwärtsberechnung, Verlustfunktionsberechnung,
Rückwärtsberechnung und Optimiererberechnung können wie das allgemeine
neuronale Netzwerk-Training erfolgen, bis die Anzahl der Iterationen den vorgegebenen Wert erreicht.
Die verwendeten Trainingsdaten wurden oben generiert, die Testdaten sind qvc_test_data und die Trainingsdaten sind qvc_train_data.

.. code-block::

    def get_accuracy(result,label):
        result,label = np.array(result.data), np.array(label.data)
        score = np.sum(np.argmax(result,axis=1)==np.argmax(label,1))
        return score

    model = Model()

    optimizer = SGD(model.parameters(),lr =0.1)

    batch_size = 3

    epoch = 20

    loss = CategoricalCrossEntropy()

    model.train()
    datas,labels = get_data("train")

    for i in range(epoch):
        count=0
        sum_loss = 0
        accuary = 0
        t = 0
        for data,label in dataloader(datas,labels,batch_size,False):
            optimizer.zero_grad()
            data,label = QTensor(data), QTensor(label)

            result = model(data)

            loss_b = loss(label,result)
            loss_b.backward()
            optimizer._step()
            sum_loss += loss_b.item()
            count+=batch_size
            accuary += get_accuracy(result,label)
            t = t + 1

        print(f"epoch:{i}, #### loss:{sum_loss/count} #####accuracy:{accuary/count}")

    model.eval()
    count = 0
    test_data,test_label = get_data("test")
    test_batch_size = 1
    accuary = 0
    sum_loss = 0
    for testd,testl in dataloader(test_data,test_label,test_batch_size):
        testd = QTensor(testd)
        test_result = model(testd)
        test_loss = loss(testl,test_result)
        sum_loss += test_loss
        count+=test_batch_size
        accuary += get_accuracy(test_result,testl)
    print(f"test:--------------->loss:{sum_loss/count} #####accuracy:{accuary/count}")

.. code-block::

    epoch:0, #### loss:0.20194714764753977 #####accuracy:0.6666666666666666
    epoch:1, #### loss:0.19724808633327484 #####accuracy:0.8333333333333334
    epoch:2, #### loss:0.19266503552595773 #####accuracy:1.0
    epoch:3, #### loss:0.18812804917494455 #####accuracy:1.0
    epoch:4, #### loss:0.1835678368806839 #####accuracy:1.0
    epoch:5, #### loss:0.1789149840672811 #####accuracy:1.0
    epoch:6, #### loss:0.17410411685705185 #####accuracy:1.0
    epoch:7, #### loss:0.16908332953850427 #####accuracy:1.0
    epoch:8, #### loss:0.16382796317338943 #####accuracy:1.0
    epoch:9, #### loss:0.15835540741682053 #####accuracy:1.0
    epoch:10, #### loss:0.15273457020521164 #####accuracy:1.0
    epoch:11, #### loss:0.14708336691061655 #####accuracy:1.0
    epoch:12, #### loss:0.14155150949954987 #####accuracy:1.0
    epoch:13, #### loss:0.1362930883963903 #####accuracy:1.0
    epoch:14, #### loss:0.1314386005202929 #####accuracy:1.0
    epoch:15, #### loss:0.12707658857107162 #####accuracy:1.0
    epoch:16, #### loss:0.123248390853405 #####accuracy:1.0
    epoch:17, #### loss:0.11995399743318558 #####accuracy:1.0
    epoch:18, #### loss:0.1171633576353391 #####accuracy:1.0
    epoch:19, #### loss:0.11482855677604675 #####accuracy:1.0
    [0.3412148654]
    test:--------------->loss:QTensor(0.3412148654, requires_grad=True) #####accuracy:1.0

Die folgende Abbildung zeigt die Kurve der Modellgenauigkeit:

.. figure:: ./images/qvc_accuracy.png
   :width: 600 px
   :align: center

|

2. Data-Re-Uploading-Algorithmus
=======================================
In einem neuronalen Netzwerk empfängt jedes Neuron Informationen von allen Neuronen der vorherigen Schicht (Abbildung a).
Im Gegensatz dazu akzeptiert der Ein-Bit-Quantenklassifikator die vorherige Informationsverarbeitungseinheit und die Eingabe (Abbildung b).
Bei traditionellen Quantenschaltungen kann das Ergebnis nach dem Hochladen der Daten direkt durch mehrere unitäre
Transformationen :math:`U(\theta_1,\theta_2,\theta_3)` ermittelt werden. Bei der Quantum Data Re-Uploading (QDRL)-Aufgabe müssen die Daten jedoch vor jeder unitären Transformation erneut hochgeladen werden.

                                                                .. centered:: Vergleich von QDRL und klassischem neuronalen Netzwerk-Schema

.. figure:: ./images/qdrl.png
   :width: 600 px
   :align: center

|

.. code-block::

    import sys
    sys.path.insert(0, "../")
    import numpy as np
    from pyvqnet.nn.linear import Linear
    from pyvqnet.qnn.qdrl.vqnet_model import vmodel
    from pyvqnet.optim import sgd
    from pyvqnet.nn.loss import CategoricalCrossEntropy
    from pyvqnet.tensor.tensor import QTensor
    from pyvqnet.nn.module import Module
    import matplotlib.pyplot as plt
    import matplotlib
    from pyvqnet.data import data_generator as get_minibatch_data
    try:
        matplotlib.use("TkAgg")
    except:  
        print("Can not use matplot TkAgg")
        pass

    np.random.seed(42)

    num_layers = 3
    params = np.random.uniform(size=(num_layers, 3))


    class Model(Module):
        def __init__(self):

            super(Model, self).__init__()
            self.pqc = vmodel(params.shape)
            self.fc2 = Linear(2, 2)

        def forward(self, x):
            x = self.pqc(x)
            return x


    def circle(samples: int, reps=np.sqrt(1 / 2)):
        data_x, data_y = [], []
        for _ in range(samples):
            x = np.random.rand(2)
            y = [0, 1]
            if np.linalg.norm(x) < reps:
                y = [1, 0]
            data_x.append(x)
            data_y.append(y)
        return np.array(data_x), np.array(data_y)


    def plot_data(x, y, fig=None, ax=None):

        if fig is None:
            fig, ax = plt.subplots(1, 1, figsize=(5, 5))
        reds = y == 0
        blues = y == 1
        ax.scatter(x[reds, 0], x[reds, 1], c="red", s=20, edgecolor="k")
        ax.scatter(x[blues, 0], x[blues, 1], c="blue", s=20, edgecolor="k")
        ax.set_xlabel("$x_1$")
        ax.set_ylabel("$x_2$")


    def get_score(pred, label):
        pred, label = np.array(pred.data), np.array(label.data)
        score = np.sum(np.argmax(pred, axis=1) == np.argmax(label, 1))
        return score


    model = Model()
    optimizer = sgd.SGD(model.parameters(), lr=1)


    def train():
        """
        Hauptfunktion zum Training des QDRL-Modells
        """
        batch_size = 5
        model.train()
        x_train, y_train = circle(500)
        x_train = np.hstack((x_train, np.ones((x_train.shape[0], 1))))  # 500*3

        epoch = 10
        print("starte Training...........")
        for i in range(epoch):
            accuracy = 0
            count = 0
            loss = 0
            for data, label in get_minibatch_data(x_train, y_train, batch_size):
                optimizer.zero_grad()

                data, label = QTensor(data), QTensor(label)

                output = model(data)

                loss_fun = CategoricalCrossEntropy()
                losss = loss_fun(label, output)

                losss.backward()

                optimizer._step()
                accuracy += get_score(output, label)

                loss += losss.item()

                count += batch_size

            print(f"epoch:{i}, train_accuracy_for_each_batch:{accuracy/count}")
            print(f"epoch:{i}, train_loss_for_each_batch:{loss/count}")


    def test():
        batch_size = 5
        model.eval()
        print("starte Auswertung...................")
        x_test, y_test = circle(500)
        test_accuracy = 0
        count = 0
        x_test = np.hstack((x_test, np.ones((x_test.shape[0], 1))))

        for test_data, test_label in get_minibatch_data(x_test, y_test,
                                                        batch_size):

            test_data, test_label = QTensor(test_data), QTensor(test_label)
            output = model(test_data)
            test_accuracy += get_score(output, test_label)
            count += batch_size
        print(f"test_accuracy:{test_accuracy/count}")


    if __name__ == "__main__":
        train()
        test()

Die folgende Abbildung zeigt die Kurve der Modellgenauigkeit:

.. figure:: ./images/qdrl_accuracy.png
   :width: 600 px
   :align: center

|

3. VSQL: Variational Shadow Quantum Learning für Klassifikationsmodelle
================================================================================================================================================================
Verwendung variabler Quantenschaltungen zur Konstruktion eines Zweiklassen-Klassifikationsmodells,
Vergleich der Klassifikationsgenauigkeit mit einem neuronalen Netz mit ähnlicher Parameteranzahl.
Die Genauigkeit beider ist ähnlich. Die Anzahl der Parameter von Quantenschaltungen ist viel geringer als die von klassischen neuronalen Netzen.
Der Algorithmus basiert auf der Arbeit `Variational Shadow Quantum Learning for Classification Model <https://arxiv.org/abs/2012.08288>`_ und wurde
reproduziert.

Die folgende Abbildung zeigt die Architektur des VSQL-Algorithmus:

.. figure:: ./images/vsql_model.PNG
   :width: 600 px
   :align: center

|

Die folgenden Abbildungen zeigen die lokalen Quantenschaltungsstrukturen auf jedem Qubit:

.. figure:: ./images/vsql_0.png
.. figure:: ./images/vsql_1.png
.. figure:: ./images/vsql_2.png
.. figure:: ./images/vsql_3.png
.. figure:: ./images/vsql_4.png
.. figure:: ./images/vsql_5.png
.. figure:: ./images/vsql_6.png
.. figure:: ./images/vsql_7.png
.. figure:: ./images/vsql_8.png

.. code-block::

    import sys
    sys.path.insert(0, "../")
    import os
    import os.path
    import struct
    import gzip
    from pyvqnet.nn.module import Module
    from pyvqnet.nn.loss import CategoricalCrossEntropy
    from pyvqnet.optim.adam import Adam
    from pyvqnet.data.data import data_generator
    from pyvqnet.tensor import tensor
    from pyvqnet.qnn.measure import expval
    from pyvqnet.qnn.quantumlayer import QuantumLayer
    from pyvqnet.qnn.template import AmplitudeEmbeddingCircuit
    from pyvqnet.nn.linear import Linear
    import numpy as np
    import pyqpanda as pq
    import matplotlib.pyplot as plt
    import matplotlib
    try:
        matplotlib.use("TkAgg")
    except:  
        print("Can not use matplot TkAgg")
        pass

    try:
        import urllib.request
    except ImportError:
        raise ImportError("You should use Python 3.x")

    url_base = 'https://ossci-datasets.s3.amazonaws.com/mnist/'
    key_file = {
        "train_img": "train-images-idx3-ubyte.gz",
        "train_label": "train-labels-idx1-ubyte.gz",
        "test_img": "t10k-images-idx3-ubyte.gz",
        "test_label": "t10k-labels-idx1-ubyte.gz"
    }


    def _download(dataset_dir, file_name):
        """
        Download-Funktion für die MNIST-Datensatzdatei
        """
        file_path = dataset_dir + "/" + file_name

        if os.path.exists(file_path):
            with gzip.GzipFile(file_path) as file:
                file_path_ungz = file_path[:-3].replace("\\", "/")
                if not os.path.exists(file_path_ungz):
                    open(file_path_ungz, "wb").write(file.read())
            return

        print("Downloading " + file_name + " ... ")
        urllib.request.urlretrieve(url_base + file_name, file_path)
        if os.path.exists(file_path):
            with gzip.GzipFile(file_path) as file:
                file_path_ungz = file_path[:-3].replace("\\", "/")
                file_path_ungz = file_path_ungz.replace("-idx", ".idx")
                if not os.path.exists(file_path_ungz):
                    open(file_path_ungz, "wb").write(file.read())
        print("Done")


    def download_mnist(dataset_dir):
        for v in key_file.values():
            _download(dataset_dir, v)


    if not os.path.exists("./result"):
        os.makedirs("./result")
    else:
        pass


    def circuits_of_vsql(x, weights, qlist, clist, machine):  
        """
        VSQL-Modell der Quantenschaltungen
        """
        weights = weights.reshape([depth + 1, 3, n_qsc])

        def subcir(weights, qlist, depth, n_qsc, n_start):  
            cir = pq.QCircuit()

            for i in range(n_qsc):
                cir.insert(pq.RX(qlist[n_start + i], weights[0][0][i]))
                cir.insert(pq.RY(qlist[n_start + i], weights[0][1][i]))
                cir.insert(pq.RX(qlist[n_start + i], weights[0][2][i]))
            for repeat in range(1, depth + 1):
                for i in range(n_qsc - 1):
                    cir.insert(pq.CNOT(qlist[n_start + i], qlist[n_start + i + 1]))
                cir.insert(pq.CNOT(qlist[n_start + n_qsc - 1], qlist[n_start]))
                for i in range(n_qsc):
                    cir.insert(pq.RY(qlist[n_start + i], weights[repeat][1][i]))

            return cir

        def get_pauli_str(n_start, n_qsc):  
            pauli_str = ",".join("X" + str(i)
                                for i in range(n_start, n_start + n_qsc))
            return {pauli_str: 1.0}

        f_i = []
        origin_in = AmplitudeEmbeddingCircuit(x, qlist)
        for st in range(n - n_qsc + 1):
            psd = get_pauli_str(st, n_qsc)
            cir = pq.QCircuit()
            cir.insert(origin_in)
            cir.insert(subcir(weights, qlist, depth, n_qsc, st))
            prog = pq.QProg()
            prog.insert(cir)

            f_ij = expval(machine, prog, psd, qlist)
            f_i.append(f_ij)
        f_i = np.array(f_i)
        return f_i


    #GLOBALE VARIABLEN
    n = 10
    n_qsc = 2
    depth = 1


    class QModel(Module):
        """
        Modell von VSQL
        """
        def __init__(self):
            super().__init__()
            self.vq = QuantumLayer(circuits_of_vsql, (depth + 1) * 3 * n_qsc,
                                "cpu", 10)
            self.fc = Linear(n - n_qsc + 1, 2)

        def forward(self, x):
            x = self.vq(x)
            x = self.fc(x)

            return x


    class Model(Module):
        def __init__(self):
            super().__init__()
            self.fc1 = Linear(input_channels=28 * 28, output_channels=2)

        def forward(self, x):

            x = tensor.flatten(x, 1)
            x = self.fc1(x)
            return x


    def load_mnist(dataset="training_data", digits=np.arange(2), path="./"):
        """
        Lädt MNIST-Daten
        """
        from array import array as pyarray
        download_mnist(path)
        if dataset == "training_data":
            fname_image = os.path.join(path, "train-images.idx3-ubyte").replace(
                "\\", "/")
            fname_label = os.path.join(path, "train-labels.idx1-ubyte").replace(
                "\\", "/")
        elif dataset == "testing_data":
            fname_image = os.path.join(path, "t10k-images.idx3-ubyte").replace(
                "\\", "/")
            fname_label = os.path.join(path, "t10k-labels.idx1-ubyte").replace(
                "\\", "/")
        else:
            raise ValueError("dataset must be 'training_data' or 'testing_data'")

        flbl = open(fname_label, "rb")
        _, size = struct.unpack(">II", flbl.read(8))

        lbl = pyarray("b", flbl.read())
        flbl.close()

        fimg = open(fname_image, "rb")
        _, size, rows, cols = struct.unpack(">IIII", fimg.read(16))
        img = pyarray("B", fimg.read())
        fimg.close()

        ind = [k for k in range(size) if lbl[k] in digits]
        num = len(ind)
        images = np.zeros((num, rows, cols), dtype=np.float32)

        labels = np.zeros((num, 1), dtype=int)
        for i in range(len(ind)):
            images[i] = np.array(img[ind[i] * rows * cols:(ind[i] + 1) * rows *
                                    cols]).reshape((rows, cols))
            labels[i] = lbl[ind[i]]

        return images, labels


    def run_vsql():
        """
        VSQL-MODELL
        """
        digits = [0, 1]
        x_train, y_train = load_mnist("training_data", digits)
        x_train = x_train / 255
        y_train = y_train.reshape(-1, 1)
        y_train = np.eye(len(digits))[y_train].reshape(-1, len(digits)).astype(
            np.int64)
        x_test, y_test = load_mnist("testing_data", digits)
        x_test = x_test / 255
        y_test = y_test.reshape(-1, 1)
        y_test = np.eye(len(digits))[y_test].reshape(-1,
                                                    len(digits)).astype(np.int64)

        x_train_list = []
        x_test_list = []
        for i in range(x_train.shape[0]):
            x_train_list.append(
                np.pad(x_train[i, :, :].flatten(), (0, 240),
                    constant_values=(0, 0)))
        x_train = np.array(x_train_list)

        for i in range(x_test.shape[0]):
            x_test_list.append(
                np.pad(x_test[i, :, :].flatten(), (0, 240),
                    constant_values=(0, 0)))

        x_test = np.array(x_test_list)

        x_train = x_train[:500]
        y_train = y_train[:500]

        x_test = x_test[:100]
        y_test = y_test[:100]
        print("model start")
        model = QModel()

        optimizer = Adam(model.parameters(), lr=0.1)

        model.train()
        result_file = open("./result/vqslrlt.txt", "w")
        for epoch in range(1, 3):

            model.train()
            full_loss = 0
            n_loss = 0
            n_eval = 0
            batch_size = 1
            correct = 0
            for x, y in data_generator(x_train,
                                    y_train,
                                    batch_size=batch_size,
                                    shuffle=True):
                optimizer.zero_grad()
                try:
                    x = x.reshape(batch_size, 1024)
                except:  
                    x = x.reshape(-1, 1024)

                output = model(x)
                cceloss = CategoricalCrossEntropy()
                loss = cceloss(y, output)
                loss.backward()
                optimizer._step()

                full_loss += loss.item()
                n_loss += batch_size
                np_output = np.array(output.data, copy=False)
                mask = np_output.argmax(1) == y.argmax(1)
                correct += sum(mask)
                print(f" n_loss {n_loss} Train Accuracy: {correct/n_loss} ")
            print(f"Train Accuracy: {correct/n_loss} ")
            print(f"Epoch: {epoch}, Loss: {full_loss / n_loss}")
            result_file.write(f"{epoch}\t{full_loss / n_loss}\t{correct/n_loss}\t")

            # Auswertung
            model.eval()
            print("eval")
            correct = 0
            full_loss = 0
            n_loss = 0
            n_eval = 0
            batch_size = 1
            for x, y in data_generator(x_test,
                                    y_test,
                                    batch_size=batch_size,
                                    shuffle=True):
                x = x.reshape(1, 1024)
                output = model(x)

                cceloss = CategoricalCrossEntropy()
                loss = cceloss(y, output)
                full_loss += loss.item()

                np_output = np.array(output.data, copy=False)
                mask = np_output.argmax(1) == y.argmax(1)
                correct += sum(mask)
                n_eval += 1
                n_loss += 1

            print(f"Eval Accuracy: {correct/n_eval}")
            result_file.write(f"{full_loss / n_loss}\t{correct/n_eval}\n")

        result_file.close()
        del model
        print("\ndone vqsl\n")


    if __name__ == "__main__":

        run_vsql()

Die folgende Grafik zeigt die Kurve der Modellgenauigkeit und des Verlusts:

.. figure:: ./images/vsql_cacc.PNG
   :width: 600 px
   :align: center

.. figure:: ./images/vsql_closs.PNG
   :width: 600 px
   :align: center

.. figure:: ./images/vsql_qacc.PNG
   :width: 600 px
   :align: center

.. figure:: ./images/vsql_qloss.PNG
   :width: 600 px
   :align: center

|

4. Quanvolution für die Bildklassifikation
=======================================================================================================================

In diesem Beispiel implementieren wir ein Quanten-Convolutional-Neuronales-Netz, eine Methode, die ursprünglich in der Arbeit `Quanvolutional Neural Networks: Powering Image Recognition with Quantum Circuits <https://arxiv.org/abs/1904.04767>`_ vorgestellt wurde.

Ähnlich wie bei der klassischen Faltung umfasst die Quanvolution die folgenden Schritte:
Eine kleine Region des Eingabebildes, in unserem Fall ein 2x2-Quadrat klassischer Daten, wird in die Quantenschaltung eingebettet.
In diesem Beispiel wird dies durch Anwendung parametrisierter rotierender Logikgatter auf Qubits erreicht, die im Grundzustand initialisiert wurden. Der Faltungskern erzeugt hier Variationsschaltungen aus stochastischen Schaltungen, wie in der Referenz vorgeschlagen.
Schließlich wird das Quantensystem gemessen, um eine Liste klassischer Erwartungswerte zu erhalten.
Ähnlich wie bei einer klassischen Faltungsschicht wird jeder Erwartungswert auf einen anderen Kanal eines einzelnen Ausgabepixels abgebildet.
Durch Wiederholen desselben Prozesses über verschiedene Regionen kann das gesamte Eingabebild gescannt werden, wodurch ein Ausgabeobjekt entsteht, das als mehrkanaliges Bild konstruiert wird.
Um Klassifikationsaufgaben durchzuführen, verwendet dieses Beispiel die klassische vollständig verbundene Schicht ``Linear``, nachdem die Quanvolution die Messwerte erhalten hat.
Der Hauptunterschied zur klassischen Faltung besteht darin, dass Quanvolution hochkomplexe Kernel erzeugen kann, deren Berechnung zumindest im Prinzip klassisch nicht durchführbar ist.

.. image:: ./images/quanvo.png
   :width: 600 px
   :align: center

|

Definition des MNIST-Datensatzes

.. code-block::

    import os
    import os.path
    import struct
    import gzip
    import sys
    sys.path.insert(0, "../")
    from pyvqnet.nn.module import Module
    from pyvqnet.nn.loss import NLL_Loss
    from pyvqnet.optim.adam import Adam
    from pyvqnet.data.data import data_generator
    from pyvqnet.tensor import tensor
    from pyvqnet.qnn.measure import expval
    from pyvqnet.nn.linear import Linear
    import numpy as np
    from pyvqnet.qnn.qcnn import Quanvolution
    import matplotlib.pyplot as plt
    import matplotlib
    try:
        matplotlib.use("TkAgg")
    except:  
        print("Can not use matplot TkAgg")
        pass

    try:
        import urllib.request
    except ImportError:
        raise ImportError("You should use Python 3.x")

    url_base = 'https://ossci-datasets.s3.amazonaws.com/mnist/'
    key_file = {
        "train_img": "train-images-idx3-ubyte.gz",
        "train_label": "train-labels-idx1-ubyte.gz",
        "test_img": "t10k-images-idx3-ubyte.gz",
        "test_label": "t10k-labels-idx1-ubyte.gz"
    }


    def _download(dataset_dir, file_name):
        """
        Download-Funktion für die MNIST-Datensatzdatei
        """
        file_path = dataset_dir + "/" + file_name

        if os.path.exists(file_path):
            with gzip.GzipFile(file_path) as file:
                file_path_ungz = file_path[:-3].replace("\\", "/")
                if not os.path.exists(file_path_ungz):
                    open(file_path_ungz, "wb").write(file.read())
            return

        print("Downloading " + file_name + " ... ")
        urllib.request.urlretrieve(url_base + file_name, file_path)
        if os.path.exists(file_path):
            with gzip.GzipFile(file_path) as file:
                file_path_ungz = file_path[:-3].replace("\\", "/")
                file_path_ungz = file_path_ungz.replace("-idx", ".idx")
                if not os.path.exists(file_path_ungz):
                    open(file_path_ungz, "wb").write(file.read())
        print("Done")


    def download_mnist(dataset_dir):
        for v in key_file.values():
            _download(dataset_dir, v)


    if not os.path.exists("./result"):
        os.makedirs("./result")
    else:
        pass


    def load_mnist(dataset="training_data", digits=np.arange(10), path="./"):
        """
        Lädt MNIST-Daten
        """
        from array import array as pyarray
        download_mnist(path)
        if dataset == "training_data":
            fname_image = os.path.join(path, "train-images.idx3-ubyte").replace(
                "\\", "/")
            fname_label = os.path.join(path, "train-labels.idx1-ubyte").replace(
                "\\", "/")
        elif dataset == "testing_data":
            fname_image = os.path.join(path, "t10k-images.idx3-ubyte").replace(
                "\\", "/")
            fname_label = os.path.join(path, "t10k-labels.idx1-ubyte").replace(
                "\\", "/")
        else:
            raise ValueError("dataset must be 'training_data' or 'testing_data'")

        flbl = open(fname_label, "rb")
        _, size = struct.unpack(">II", flbl.read(8))

        lbl = pyarray("b", flbl.read())
        flbl.close()

        fimg = open(fname_image, "rb")
        _, size, rows, cols = struct.unpack(">IIII", fimg.read(16))
        img = pyarray("B", fimg.read())
        fimg.close()

        ind = [k for k in range(size) if lbl[k] in digits]
        num = len(ind)
        images = np.zeros((num, rows, cols))

        labels = np.zeros((num, 1), dtype=int)
        for i in range(len(ind)):
            images[i] = np.array(img[ind[i] * rows * cols:(ind[i] + 1) * rows *
                                    cols]).reshape((rows, cols))
            labels[i] = lbl[ind[i]]

        return images, labels
    def _download(dataset_dir, file_name):
        """
        Download-Funktion fuer die MNIST-Datensatzdatei
        """
        file_path = dataset_dir + "/" + file_name

        if os.path.exists(file_path):
            with gzip.GzipFile(file_path) as file:
                file_path_ungz = file_path[:-3].replace("\\", "/")
                if not os.path.exists(file_path_ungz):
                    open(file_path_ungz, "wb").write(file.read())
            return

        print("Downloading " + file_name + " ... ")
        urllib.request.urlretrieve(url_base + file_name, file_path)
        if os.path.exists(file_path):
            with gzip.GzipFile(file_path) as file:
                file_path_ungz = file_path[:-3].replace("\\", "/")
                file_path_ungz = file_path_ungz.replace("-idx", ".idx")
                if not os.path.exists(file_path_ungz):
                    open(file_path_ungz, "wb").write(file.read())
        print("Done")


    def download_mnist(dataset_dir):
        for v in key_file.values():
            _download(dataset_dir, v)


    if not os.path.exists("./result"):
        os.makedirs("./result")
    else:
        pass


    def load_mnist(dataset="training_data", digits=np.arange(10), path="./"):
        """
        Laedt MNIST-Daten
        """
        from array import array as pyarray
        download_mnist(path)
        if dataset == "training_data":
            fname_image = os.path.join(path, "train-images.idx3-ubyte").replace(
                "\\", "/")
            fname_label = os.path.join(path, "train-labels.idx1-ubyte").replace(
                "\\", "/")
        elif dataset == "testing_data":
            fname_image = os.path.join(path, "t10k-images.idx3-ubyte").replace(
                "\\", "/")
            fname_label = os.path.join(path, "t10k-labels.idx1-ubyte").replace(
                "\\", "/")
        else:
            raise ValueError("dataset must be 'training_data' or 'testing_data'")

        flbl = open(fname_label, "rb")
        _, size = struct.unpack(">II", flbl.read(8))

        lbl = pyarray("b", flbl.read())
        flbl.close()

        fimg = open(fname_image, "rb")
        _, size, rows, cols = struct.unpack(">IIII", fimg.read(16))
        img = pyarray("B", fimg.read())
        fimg.close()

        ind = [k for k in range(size) if lbl[k] in digits]
        num = len(ind)
        images = np.zeros((num, rows, cols))

        labels = np.zeros((num, 1), dtype=int)
        for i in range(len(ind)):
            images[i] = np.array(img[ind[i] * rows * cols:(ind[i] + 1) * rows *
                                    cols]).reshape((rows, cols))
            labels[i] = lbl[ind[i]]

        return images, labels

Definition des Moduls und der Prozessfunktion

.. code-block::

    class QModel(Module):

        def __init__(self):
            super().__init__()
            self.vq = Quanvolution([4, 2], (2, 2))
            self.fc = Linear(4 * 14 * 14, 10)

        def forward(self, x):
            x = self.vq(x)
            x = tensor.flatten(x, 1)
            x = self.fc(x)
            x = tensor.log_softmax(x)
            return x



    def run_quanvolution():

        digit = 10
        x_train, y_train = load_mnist("training_data", digits=np.arange(digit))
        x_train = x_train / 255

        y_train = y_train.flatten()

        x_test, y_test = load_mnist("testing_data", digits=np.arange(digit))

        x_test = x_test / 255
        y_test = y_test.flatten()

        x_train = x_train[:500]
        y_train = y_train[:500]

        x_test = x_test[:100]
        y_test = y_test[:100]

        print("model start")
        model = QModel()

        optimizer = Adam(model.parameters(), lr=5e-3)

        model.train()
        result_file = open("quanvolution.txt", "w")

        cceloss = NLL_Loss()
        N_EPOCH = 15

        for epoch in range(1, N_EPOCH):

            model.train()
            full_loss = 0
            n_loss = 0
            n_eval = 0
            batch_size = 10
            correct = 0
            for x, y in data_generator(x_train,
                                    y_train,
                                    batch_size=batch_size,
                                    shuffle=True):
                optimizer.zero_grad()
                try:
                    x = x.reshape(batch_size, 1, 28, 28)
                except:  
                    x = x.reshape(-1, 1, 28, 28)

                output = model(x)

                loss = cceloss(y, output)
                print(f"loss {loss}")
                loss.backward()
                optimizer._step()

                full_loss += loss.item()
                n_loss += batch_size
                np_output = np.array(output.data, copy=False)
                mask = np_output.argmax(1) == y

                correct += sum(mask)
                print(f"correct {correct}")
            print(f"Train Accuracy: {correct/n_loss}%")
            print(f"Epoch: {epoch}, Loss: {full_loss / n_loss}")
            result_file.write(f"{epoch}\t{full_loss / n_loss}\t{correct/n_loss}\t")

            # Auswertung
            model.eval()
            print("eval")
            correct = 0
            full_loss = 0
            n_loss = 0
            n_eval = 0
            batch_size = 1
            for x, y in data_generator(x_test,
                                    y_test,
                                    batch_size=batch_size,
                                    shuffle=True):
                x = x.reshape(-1, 1, 28, 28)
                output = model(x)

                loss = cceloss(y, output)
                full_loss += loss.item()

                np_output = np.array(output.data, copy=False)
                mask = np_output.argmax(1) == y
                correct += sum(mask)
                n_eval += 1
                n_loss += 1

            print(f"Eval Accuracy: {correct/n_eval}")
            result_file.write(f"{full_loss / n_loss}\t{correct/n_eval}\n")

        result_file.close()
        del model
        print("\ndone\n")


    if __name__ == "__main__":

        run_quanvolution()

Trainings- und Validierungsverlust sowie Klassifikationsgenauigkeit in Abhaengigkeit von der Epoche.

.. code-block::

    # Epoche Trainingsverlust Trainingsgenauigkeit Validierungsverlust Validierungsgenauigkeit
    # 1	0.2488900272846222	0.232	1.7297331787645818	0.39
    # 2	0.12281704187393189	0.646	1.201728610806167	0.61
    # 3	0.08001763761043548	0.772	0.8947569639235735	0.73
    # 4	0.06211201059818268	0.83	0.777864265316166	0.74
    # 5	0.052190632969141004	0.858	0.7291000287979841	0.76
    # 6	0.04542196464538574	0.87	0.6764470228599384	0.8
    # 7	0.04029472427070141	0.896	0.6153804161818698	0.79
    # 8	0.03600500610470772	0.902	0.5644993982824963	0.81
    # 9	0.03230033944547176	0.916	0.528938240573043	0.81
    # 10	0.02912954458594322	0.93	0.5058713140769396	0.83
    # 11	0.026443827204406262	0.936	0.49064547760412097	0.83
    # 12	0.024144304402172564	0.942	0.4800815625616815	0.82
    # 13	0.022141477409750223	0.952	0.4724775951183983	0.83
    # 14	0.020372112181037665	0.956	0.46692863543197743	0.83


Quantum AutoEncoder Demo
*******************************

1. Quantum AutoEncoder
=======================================

Der klassische Autoencoder ist ein neuronales Netz, das hocheffiziente niedrigdimensionale Darstellungen von Daten in einem hochdimensionalen Raum erlernen kann.
Die Aufgabe des Autoencoders besteht darin, bei einer gegebenen Eingabe x diese auf einen niedrigdimensionalen Punkt y abzubilden, sodass x aus y wiederhergestellt werden kann.
Die Struktur des zugrunde liegenden Autoencoder-Netzwerks kann so gewaehlt werden, dass die Daten in einer kleineren Dimension dargestellt werden, wodurch die Eingabe effektiv komprimiert wird.
Inspiriert von dieser Idee wird das Modell des Quanten-Autoencoders verwendet, um aehnliche Aufgaben mit Quantendaten durchzufuehren.
Quanten-Autoencoder werden trainiert, um bestimmte Datensaetze von Quantenzustaenden zu komprimieren, wobei klassische Kompressionsalgorithmen nicht verwendet werden koennen.
Die Parameter des Quanten-Autoencoders werden mit klassischen Optimierungsalgorithmen trainiert.
Wir zeigen ein Beispiel einer einfachen programmierbaren Schaltung, die als effizienter Autoencoder trainiert werden kann.
Wir wenden unser Modell im Kontext der Quantensimulation an, um das Hubbard-Modell und den Grundzustand des Hamilton-Operators zu komprimieren.
Dieser Algorithmus basiert auf `Quantum autoencoders for efficient compression of quantum data <https://arxiv.org/pdf/1612.02806.pdf>`_ .


QAE-Quantenschaltungen:

.. figure:: ./images/QAE_Quantum_Cir.png
   :width: 600 px
   :align: center

|

.. code-block::

    import os
    import sys
    sys.path.insert(0,'../')
    import numpy as np
    from pyvqnet.nn.module import Module
    from pyvqnet.nn.loss import  fidelityLoss
    from pyvqnet.optim.adam import Adam
    from pyvqnet.data.data import data_generator
    from pyvqnet.qnn.qae.qae import QAElayer
    import matplotlib.pyplot as plt
    import matplotlib
    try:
        matplotlib.use('TkAgg')
    except:
        pass
    try:
        import urllib.request
    except ImportError:
        raise ImportError('You should use Python 3.x')
    import os.path
    import gzip

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


    class Model(Module):

        def __init__(self, trash_num: int = 2, total_num: int = 7):
            super().__init__()
            self.pqc = QAElayer(trash_num, total_num)

        def forward(self, x):

            x = self.pqc(x)
            return x

    def load_mnist(dataset="training_data", digits=np.arange(2), path="./"):         
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
        labels = np.zeros((N, 1), dtype=int)
        for i in range(len(ind)):
            images[i] = np.array(img[ind[i] * rows * cols: (ind[i] + 1) * rows * cols]).reshape((rows, cols))
            labels[i] = lbl[ind[i]]

        return images, labels

    def run2():
        ##Datensatz laden

        x_train, y_train = load_mnist("training_data")                       
        x_train = x_train / 255                                             

        x_test, y_test = load_mnist("testing_data")

        x_test = x_test / 255

        x_train = x_train.reshape([-1, 1, 28, 28])
        x_test = x_test.reshape([-1, 1, 28, 28])
        x_train = x_train[:100, :, :, :]
        x_train = np.resize(x_train, [x_train.shape[0], 1, 2, 2])

        x_test = x_test[:10, :, :, :]
        x_test = np.resize(x_test, [x_test.shape[0], 1, 2, 2])
        encode_qubits = 4
        latent_qubits = 2
        trash_qubits = encode_qubits - latent_qubits
        total_qubits = 1 + trash_qubits + encode_qubits
        print("model start")
        model = Model(trash_qubits, total_qubits)

        optimizer = Adam(model.parameters(), lr=0.005)
        model.train()
        F1 = open("rlt.txt", "w")
        loss_list = []
        loss_list_test = []
        fidelity_train = []
        fidelity_val = []

        for epoch in range(1, 10):
            running_fidelity_train = 0
            running_fidelity_val = 0
            print(f"epoch {epoch}")
            model.train()
            full_loss = 0
            n_loss = 0
            n_eval = 0
            batch_size = 1
            correct = 0
            iter = 0
            if epoch %5 ==1:
                optimizer.lr  = optimizer.lr *0.5
            for x, y in data_generator(x_train, y_train, batch_size=batch_size, shuffle=True): #Daten mischen, nicht Batches

                x = x.reshape((-1, encode_qubits))
                x = np.concatenate((np.zeros([batch_size, 1 + trash_qubits]), x), 1)
                optimizer.zero_grad()
                output = model(x)
                iter += 1
                np_out = np.array(output.data)
                floss = fidelityLoss()
                loss = floss(output)
                loss_data = np.array(loss.data)
                loss.backward()

                running_fidelity_train += np_out[0]
                optimizer._step()
                full_loss += loss_data[0]
                n_loss += batch_size
                np_output = np.array(output.data, copy=False)
                mask = np_output.argmax(1) == y.argmax(1)

                correct += sum(mask)

            loss_output = full_loss / n_loss
            print(f"Epoch: {epoch}, Loss: {loss_output}")
            loss_list.append(loss_output)


            # Auswertung
            model.eval()
            correct = 0
            full_loss = 0
            n_loss = 0
            n_eval = 0
            batch_size = 1
            for x, y in data_generator(x_test, y_test, batch_size=batch_size, shuffle=True):
                x = x.reshape((-1, encode_qubits))
                x = np.concatenate((np.zeros([batch_size, 1 + trash_qubits]),x),1)
                output = model(x)

                floss = fidelityLoss()
                loss = floss(output)
                loss_data = np.array(loss.data)
                full_loss += loss_data[0]
                running_fidelity_val += np.array(output.data)[0]

                n_eval += 1
                n_loss += 1

            loss_output = full_loss / n_loss
            print(f"Epoch: {epoch}, Loss: {loss_output}")
            loss_list_test.append(loss_output)

            fidelity_train.append(running_fidelity_train / 64)
            fidelity_val.append(running_fidelity_val / 64)

        figure_path = os.path.join(os.getcwd(), 'QAE-rate1.png')
        plt.plot(loss_list, color="blue", label="train")
        plt.plot(loss_list_test, color="red", label="validation")
        plt.title('QAE')
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.legend(loc="upper right")
        plt.savefig(figure_path)
        plt.show()

        F1.write(f"done\n")
        F1.close()
        del model

    if __name__ == '__main__':
        run2()

Der durch Ausfuehren des obigen Codes erhaltene QAE-Fehlerwert, der Verlust ist 1/Fidelity, wobei ein Wert nahe 1 bedeutet, dass die Fidelity nahe 1 liegt.

.. figure:: ./images/qae_train_loss.png
   :width: 600 px
   :align: center

|

Demo zum Lernen von Quantenschaltungsstrukturen
***********************************************

1. Lernen von Quantenschaltungsstrukturen
===============================================================================

In der Quantenschaltungsstruktur sind die am haeufigsten verwendeten parametrisierten Quantengatter `RZ`, `RY` und `RX`. Welches Gatter unter welchen Umstaenden verwendet werden sollte, ist eine lohnende Forschungsfrage. Eine Methode ist die zufaellige Auswahl, aber in diesem Fall werden wahrscheinlich nicht die besten Ergebnisse erzielt.
Das Kernziel der Aufgabe zum Lernen von Quantenschaltungsstrukturen ist es, die optimale Kombination von parametrisierten Quantengattern zu finden.
Der hier verfolgte Ansatz ist, dass diese Menge optimaler Quantenlogikgatter die Verlustfunktion minimieren sollte.


.. code-block::

    """
    Demo zum Lernen von Quantenschaltungsstrukturen

    """
    import sys
    sys.path.insert(0,"../")

    import copy
    import pyqpanda as pq
    from pyvqnet.tensor.tensor import QTensor
    from pyvqnet.qnn.measure import expval
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib
    try:
        matplotlib.use("TkAgg")
    except:  
        print("Can not use matplot TkAgg")
        pass

    machine = pq.CPUQVM()
    machine.init_qvm()
    nqbits = machine.qAlloc_many(2)

    def gen(param, generators, qbits, circuit):
        if generators == "X":
            circuit.insert(pq.RX(qbits, param))
        elif generators == "Y":
            circuit.insert(pq.RY(qbits, param))
        else:
            circuit.insert(pq.RZ(qbits, param))

    def circuits(params, generators, circuit):
        gen(params[0], generators[0], nqbits[0], circuit)
        gen(params[1], generators[1], nqbits[1], circuit)
        circuit.insert(pq.CNOT(nqbits[0], nqbits[1]))
        prog = pq.QProg()
        prog.insert(circuit)
        return prog

    def ansatz1(params: QTensor, generators):
        circuit = pq.QCircuit()
        params = params.to_numpy()
        prog = circuits(params, generators, circuit)
        return expval(machine, prog, {"Z0": 1},
                    nqbits), expval(machine, prog, {"Y1": 1}, nqbits)


    def ansatz2(params: QTensor, generators):
        circuit = pq.QCircuit()
        params = params.to_numpy()
        prog = circuits(params, generators, circuit)
        return expval(machine, prog, {"X0": 1}, nqbits)


    def loss(params, generators):
        z, y = ansatz1(params, generators)
        x = ansatz2(params, generators)
        return 0.5 * y + 0.8 * z - 0.2 * x


    def rotosolve(d, params, generators, cost, M_0):
        """
        Implementierung des Rotosolve-Algorithmus
        """
        params[d] = np.pi / 2.0
        m0_plus = cost(QTensor(params), generators)
        params[d] = -np.pi / 2.0
        m0_minus = cost(QTensor(params), generators)
        a = np.arctan2(2.0 * M_0 - m0_plus - m0_minus,
                    m0_plus - m0_minus)  # gibt Wert in (-pi,pi] zurueck
        params[d] = -np.pi / 2.0 - a
        if params[d] <= -np.pi:
            params[d] += 2 * np.pi
        return cost(QTensor(params), generators)


    def optimal_theta_and_gen_helper(index, params, generators):
        """
        Findet optimale Variablen
        """
        params[index] = 0.
        m0 = loss(QTensor(params), generators)  #Startwert
        for kind in ["X", "Y", "Z"]:
            generators[index] = kind
            params_cost = rotosolve(index, params, generators, loss, m0)
            if kind == "X" or params_cost <= params_opt_cost:
                params_opt_d = params[index]
                params_opt_cost = params_cost
                generators_opt_d = kind
        return params_opt_d, generators_opt_d


    def rotoselect_cycle(params: np, generators):
        for index in range(params.shape[0]):
            params[index], generators[index] = optimal_theta_and_gen_helper(
                index, params, generators)
        return params, generators


    params = QTensor(np.array([0.3, 0.25]))
    params = params.to_numpy()
    generator = ["X", "Y"]
    generators = copy.deepcopy(generator)
    epoch = 20
    state_save = []
    for i in range(epoch):
        state_save.append(loss(QTensor(params), generators))
        params, generators = rotoselect_cycle(params, generators)

    print("Optimal generators are: {}".format(generators))
    print("Optimal params are: {}".format(params))
    steps = np.arange(0, epoch)


    plt.plot(steps, state_save, "o-")
    plt.title("rotoselect")
    plt.xlabel("cycles")
    plt.ylabel("cost")
    plt.yticks(np.arange(-1.25, 0.80, 0.25))
    plt.tight_layout()
    plt.show()

Die durch Ausfuehren des obigen Codes erhaltene Quantenschaltungsstruktur enthaelt :math:`RX` und ein :math:`RY`:

.. figure:: ./images/final_quantum_circuit.png
   :width: 600 px
   :align: center

|

Und mit den Parameterwerten :math:`\theta_1`, :math:`\theta_2` in den Quantengattern hat die Verlustfunktion unterschiedliche Werte.

.. figure:: ./images/loss3d.png
   :width: 600 px
   :align: center

|

Demo eines hybriden quantenklassischen neuronalen Netzes
********************************************************

1. Hybrides quantenklassisches neuronales Netzwerkmodell
==============================================================================

Maschinelles Lernen (ML) hat sich zu einem erfolgreichen interdisziplinaeren Bereich entwickelt, der darauf abzielt, mathematisch verallgemeinerbare Informationen aus Daten zu extrahieren.
Quantenmaschinelles Lernen versucht, die Prinzipien der Quantenmechanik zu nutzen, um maschinelles Lernen zu verbessern, und umgekehrt.
Ob Ihr Ziel darin besteht, klassische ML-Algorithmen zu verbessern, indem schwierige Berechnungen an Quantencomputer ausgelagert werden,
oder klassische ML-Architekturen zur Optimierung von Quantenalgorithmen zu verwenden, beides faellt in die Kategorie des quantenmaschinellen Lernens (QML).
In diesem Kapitel werden wir untersuchen, wie man klassische neuronale Netze teilweise quantisiert, um hybride quantenklassische neuronale Netze zu erstellen.
Quantenschaltungen bestehen aus Quantenlogikgattern, und die von diesen Logikgattern implementierten Quantenberechnungen
erweisen sich als differenzierbar, wie in der Arbeit `Quantum Circuit Learning <https://arxiv.org/abs/1803.00745>`_ gezeigt.
Daher versuchen Forscher, Quantenschaltungen und klassische neuronale Netzwerkmodule zusammenzufuehren, um sie in hybriden quantenklassischen maschinellen Lernaufgaben zu trainieren.
Wir werden ein einfaches Beispiel schreiben, um eine neuronale Netzwerkmodell-Trainingsaufgabe mit VQNet zu implementieren.
Der Zweck dieses Beispiels ist es, die Einfachheit von VQNet zu demonstrieren und ML-Praktiker zu ermutigen, die Moeglichkeiten des Quantencomputings zu erkunden.


Datenvorbereitung
-----------------------

Wir werden `MNIST-Datensaetze <https://ossci-datasets.s3.amazonaws.com/mnist/>`_, die grundlegendste neuronale Netzwerk-Handschriftdatenbank, als Klassifikationsdaten verwenden.
Wir laden zunaechst MNIST und filtern Datenproben, die 0 und 1 enthalten.
Diese Proben werden in Trainingsdaten (training_data) und Testdaten (testing_data) aufgeteilt, jeweils mit einer Dimension von 1*784.

.. code-block::

    import time
    import os
    import struct
    import gzip
    from pyvqnet.nn.module import Module
    from pyvqnet.nn.linear import Linear
    from pyvqnet.nn.conv import Conv2D

    from pyvqnet.nn import activation as F
    from pyvqnet.nn.pooling import MaxPool2D
    from pyvqnet.nn.loss import CategoricalCrossEntropy
    from pyvqnet.optim.adam import Adam
    from pyvqnet.data.data import data_generator
    from pyvqnet.tensor import tensor
    from pyvqnet.tensor import QTensor
    import pyqpanda as pq
    import pyvqnet
    pyvqnet.utils.set_random_seed(142)
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib
    try:
        matplotlib.use("TkAgg")
    except:  
        print("Can not use matplot TkAgg")
        pass

    try:
        import urllib.request
    except ImportError:
        raise ImportError("You should use Python 3.x")

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

    def load_mnist(dataset="training_data", digits=np.arange(2), path="./"):         
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
        labels = np.zeros((N, 1), dtype=int)
        for i in range(len(ind)):
            images[i] = np.array(img[ind[i] * rows * cols: (ind[i] + 1) * rows * cols]).reshape((rows, cols))
            labels[i] = lbl[ind[i]]

        return images, labels

    def data_select(train_num, test_num):
        x_train, y_train = load_mnist("training_data")
        x_test, y_test = load_mnist("testing_data")
        # Behalten nur Labels 0 und 1
        idx_train = np.append(np.where(y_train == 0)[0][:train_num],
                        np.where(y_train == 1)[0][:train_num])
        x_train = x_train[idx_train]
        y_train = y_train[idx_train]
        x_train = x_train / 255
        y_train = np.eye(2)[y_train].reshape(-1, 2)
        # Test: Behalten nur Labels 0 und 1
        idx_test = np.append(np.where(y_test == 0)[0][:test_num],
                        np.where(y_test == 1)[0][:test_num])
        x_test = x_test[idx_test]
        y_test = y_test[idx_test]
        x_test = x_test / 255
        y_test = np.eye(2)[y_test].reshape(-1, 2)
        return x_train, y_train, x_test, y_test

    n_samples_show = 6

    x_train, y_train, x_test, y_test = data_select(100, 50)
    fig, axes = plt.subplots(nrows=1, ncols=n_samples_show, figsize=(10, 3))

    for img ,targets in zip(x_test,y_test):
        if n_samples_show <= 3:
            break

        if targets[0] == 1:
            axes[n_samples_show - 1].set_title("Labeled: 0")
            axes[n_samples_show - 1].imshow(img.squeeze(), cmap='gray')
            axes[n_samples_show - 1].set_xticks([])
            axes[n_samples_show - 1].set_yticks([])
            n_samples_show -= 1

    for img ,targets in zip(x_test,y_test):
        if n_samples_show <= 0:
            break

        if targets[0] == 0:
            axes[n_samples_show - 1].set_title("Labeled: 1")
            axes[n_samples_show - 1].imshow(img.squeeze(), cmap='gray')
            axes[n_samples_show - 1].set_xticks([])
            axes[n_samples_show - 1].set_yticks([])
            n_samples_show -= 1

    plt.show()

.. figure:: ./images/mnsit_data_examples.png
   :width: 600 px
   :align: center

|

Aufbau von Quantenschaltungen
------------------------------------

In diesem Beispiel verwenden wir pyQPanda2, eine einfache Quantenschaltung mit 1 Qubit wird definiert. Die Schaltung nimmt die Ausgabe der klassischen neuronalen Netzwerkschicht als Eingabe, kodiert Quantendaten durch ``H``- und ``RY``-Quantenlogikgatter und berechnet den Erwartungswert des Hamilton-Operators in z-Richtung als Ausgabe.

.. code-block::

    from pyqpanda import *
    import pyqpanda as pq
    import numpy as np
    def circuit(x ,weights):
        num_qubits = 1
        #Verwendung von pyQPanda2 zur Erstellung eines Simulators
        machine = pq.CPUQVM()
        machine.init_qvm()
        #Verwendung von pyQPanda2 zur Zuweisung von Qubits
        qubits = machine.qAlloc_many(num_qubits)
        #Verwendung von pyQPanda2 zur Zuweisung klassischer Bits
        cbits = machine.cAlloc_many(num_qubits)
        #Aufbau der Schaltung
        circuit = pq.QCircuit()
        circuit.insert(pq.H(qubits[0]))
        circuit.insert(pq.RY(qubits[0], x[0]))
        #Aufbau des Quantenprogramms
        prog = pq.QProg()
        prog.insert(circuit)
        #Definition der Messung
        prog << measure_all(qubits, cbits)
        shots = 1000
        #Ausfuehren der Quantenmessung
        result = machine.run_with_configuration(prog, cbits,shots)
        machine.finalize()
        #Auffuellen beider Ergebnisse: Alle Schuesse koennten auf einen einzigen Basiszustand kollabieren
        count0 = result.get("0", 0)
        count1 = result.get("1", 0)
        expectation = (count0 - count1) / shots   # <Z> = P(0) - P(1)
        return expectation
        

.. figure:: ./images/hqcnn_quantum_cir.png
   :width: 600 px
   :align: center

|

Erstellung des hybriden Modells
-------------------------------

Da Quantenschaltungen automatische Differenzierungsberechnungen zusammen mit klassischen neuronalen Netzen durchfuehren koennen,
koennen wir daher die Faltungsschicht ``Conv2D``, die Pooling-Schicht ``MaxPool2D``, die vollstaendig verbundene Schicht ``Linear`` von VQNet und
die soeben definierte Quantenschaltung verwenden, um ein Modell zu erstellen.
Die Definition der Klassen `Net` und `Hybrid` erbt vom VQNet-Autodiff-Modul ``Module``
und die Definition der Vorwaertsberechnung erfolgt in der Funktion ``forward()``.
Ein automatisches Differenzierungsmodell aus Faltung, Quantenkodierung und Messung der MNIST-Daten wird konstruiert, um die endgueltigen Merkmale zu erhalten, die fuer die Klassifikationsaufgabe erforderlich sind.

.. code-block::

    #Definition des Vorwaertspasses der Quantenberechnungsschicht und der Gradientenberechnungsfunktion, die von der abstrakten Klasse Module erben muss
    from pyvqnet.qnn import QuantumLayerV2
    #Modelldefinition
    class Net(Module):
        def __init__(self):
            super(Net, self).__init__()
            self.conv1 = Conv2D(input_channels=1, output_channels=6, kernel_size=(5, 5), stride=(1, 1), padding="valid")
            self.maxpool1 = MaxPool2D([2, 2], [2, 2], padding="valid")
            self.conv2 = Conv2D(input_channels=6, output_channels=16, kernel_size=(5, 5), stride=(1, 1), padding="valid")
            self.maxpool2 = MaxPool2D([2, 2], [2, 2], padding="valid")
            self.fc1 = Linear(input_channels=256, output_channels=64)
            self.fc2 = Linear(input_channels=64, output_channels=1)
            self.hybrid = QuantumLayerV2(circuit,0)
            self.fc3 = Linear(input_channels=1, output_channels=2)

        def forward(self, x):
            x = F.ReLu()(self.conv1(x))  # 1 6 24 24
            x = self.maxpool1(x)
            x = F.ReLu()(self.conv2(x))  # 1 16 8 8
            x = self.maxpool2(x)
            x = tensor.flatten(x, 1)   # 1 256
            x = F.ReLu()(self.fc1(x))  # 1 64
            x = self.fc2(x)    # 1 1
            x = self.hybrid(x)
            x = self.fc3(x)
            return x

.. figure:: ./images/hqcnnmodel.PNG
   :width: 600 px
   :align: center

|

Training und Test
-----------------------

Fuer das hybride neuronale Netzwerkmodell, wie in der folgenden Abbildung gezeigt, berechnen wir die Verlustfunktion durch iteratives Einspeisen von Daten in das Modell,
und VQNet berechnet automatisch den Gradienten jedes Parameters in der Rueckwaertsberechnung
und verwendet den Optimierer, um die Parameter zu optimieren, bis die Anzahl der Iterationen den vorgegebenen Wert erreicht.

.. figure:: ./images/hqcnnarch.PNG
   :width: 600 px
   :align: center

|

.. code-block::

    x_train, y_train, x_test, y_test = data_select(1000, 100)

    #Modell erstellen
    model = Net() 
    #Adam-Optimierer verwenden
    optimizer = Adam(model.parameters(), lr=0.005)
    #Kreuzentropieverlust verwenden
    loss_func = CategoricalCrossEntropy()

    #Trainingsepochen   
    epochs = 10
    train_loss_list = []
    val_loss_list = []
    train_acc_list =[]
    val_acc_list = []


    for epoch in range(1, epochs):
        total_loss = []
        model.train()
        batch_size = 250
        correct = 0
        n_train = 0
        for x, y in data_generator(x_train, y_train, batch_size=batch_size, shuffle=True):

            x = x.reshape(-1, 1, 28, 28)

            optimizer.zero_grad()
            output = model(x)       
            loss = loss_func(y, output)  
            loss_np = np.array(loss.data)
            
            np_output = np.array(output.data, copy=False)
            mask = (np_output.argmax(1) == y.argmax(1))
            correct += np.sum(np.array(mask))
            n_train += batch_size

            loss.backward()
            optimizer._step()

            total_loss.append(loss_np)

        train_loss_list.append(np.sum(total_loss) / len(total_loss))
        train_acc_list.append(np.sum(correct) / n_train)
        print("{:.0f} loss is : {:.10f}".format(epoch, train_loss_list[-1]))


        model.eval()
        correct = 0
        n_eval = 0

        for x, y in data_generator(x_test, y_test, batch_size=1, shuffle=True):
            x = x.reshape(-1, 1, 28, 28)
            output = model(x)
            loss = loss_func(y, output)
            loss_np = np.array(loss.data)
            np_output = np.array(output.data, copy=False)
            mask = (np_output.argmax(1) == y.argmax(1))
            correct += np.sum(np.array(mask))
            n_eval += 1
            
            total_loss.append(loss_np)
        print(f"Eval Accuracy: {correct / n_eval}")
        val_loss_list.append(np.sum(total_loss) / len(total_loss))
        val_acc_list.append(np.sum(correct) / n_eval)

Visualisierung
---------------------

Die Visualisierungskurve des Datenverlusts und der Genauigkeit auf Trainings- und Testdaten.

.. code-block::

    import os
    plt.figure()
    xrange = range(1,len(train_loss_list)+1)
    figure_path = os.path.join(os.getcwd(), 'HQCNN LOSS.png')
    plt.plot(xrange,train_loss_list, color="blue", label="train")
    plt.plot(xrange,val_loss_list, color="red", label="validation")
    plt.title('HQCNN')
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.xticks(np.arange(1, epochs +1,step = 2))
    plt.legend(loc="upper right")
    plt.savefig(figure_path)
    plt.show()

    plt.figure()
    figure_path = os.path.join(os.getcwd(), 'HQCNN Accuracy.png')
    plt.plot(xrange,train_acc_list, color="blue", label="train")
    plt.plot(xrange,val_acc_list, color="red", label="validation")
    plt.title('HQCNN')
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.xticks(np.arange(1, epochs +1,step = 2))
    plt.legend(loc="lower right")
    plt.savefig(figure_path)
    plt.show()


.. figure:: ./images/HQCNNLOSS.png
   :width: 600 px
   :align: center

.. figure:: ./images/HQCNNAccuracy.png
   :width: 600 px
   :align: center

|

.. code-block::

    n_samples_show = 6
    count = 0
    fig, axes = plt.subplots(nrows=1, ncols=n_samples_show, figsize=(10, 3))
    model.eval()
    for x, y in data_generator(x_test, y_test, batch_size=1, shuffle=True):
        if count == n_samples_show:
            break
        x = x.reshape(-1, 1, 28, 28)
        output = model(x)
        pred = QTensor.argmax(output, [1],False)
        axes[count].imshow(x[0].squeeze(), cmap='gray')
        axes[count].set_xticks([])
        axes[count].set_yticks([])
        axes[count].set_title('Predicted {}'.format(np.array(pred.data)))
        count += 1
    plt.show()

.. figure:: ./images/eval_test.png
   :width: 600 px
   :align: center

|

2. Hybrides quantenklassisches Transfer-Learning-Modell
=======================================================================================================================
Wir wenden eine maschinelle Lernmethode namens Transfer Learning auf einen Bildklassifikator an, der auf einem hybriden klassischen Quanten-Netzwerk basiert.
Wir werden ein einfaches Beispiel zur Integration von pyQPanda2 mit VQNet schreiben. Transfer Learning basiert auf der allgemeinen Intuition,
dass ein vortrainiertes Netzwerk, wenn es gut darin ist, ein bestimmtes Problem zu loesen, auch zur Loesung eines anderen,
aber verwandten Problems verwendet werden kann, wobei nur zusaetzliches Training erforderlich ist.

Das Diagramm der Quantenschaltungsteile ist unten dargestellt:

.. figure:: ./images/QTransferLearning_cir.png
   :width: 600 px
   :align: center

|

.. code-block::

    """
    Demo zum Transfer-Learning mit quantenklassischem neuronalen Netz

    """

    import os
    import sys
    sys.path.insert(0,'../')
    import numpy as np
    import matplotlib.pyplot as plt

    from pyvqnet.nn.module import Module
    from pyvqnet.nn.linear import Linear
    from pyvqnet.nn.conv import Conv2D
    from pyvqnet.utils.storage import load_parameters, save_parameters

    from pyvqnet.nn import activation as F
    from pyvqnet.nn.pooling import MaxPool2D

    from pyvqnet.nn.batch_norm import BatchNorm2d
    from pyvqnet.nn.loss import SoftmaxCrossEntropy

    from pyvqnet.optim.sgd import SGD
    from pyvqnet.optim.adam import Adam
    from pyvqnet.data.data import data_generator
    from pyvqnet.tensor import tensor
    from pyvqnet.tensor.tensor import QTensor
    import pyqpanda as pq
    from pyqpanda import *
    import matplotlib
    from pyvqnet.nn.module import *
    from pyvqnet.utils.initializer import *
    from pyvqnet.qnn.quantumlayer import QuantumLayer

    try:
        matplotlib.use('TkAgg')
    except:
        pass

    try:
        import urllib.request
    except ImportError:
        raise ImportError('You should use Python 3.x')
    import os.path
    import gzip

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


    if not os.path.exists("./result"):
        os.makedirs("./result")
    else:
        pass
    # klassisches CNN
    class CNN(Module):
        def __init__(self):
            super(CNN, self).__init__()

            self.conv1 = Conv2D(input_channels=1, output_channels=16, kernel_size=(3, 3), stride=(1, 1), padding="valid")
            self.BatchNorm2d1 = BatchNorm2d(16)
            self.Relu1 = F.ReLu()

            self.conv2 = Conv2D(input_channels=16, output_channels=32, kernel_size=(3, 3), stride=(1, 1), padding="valid")
            self.BatchNorm2d2 = BatchNorm2d(32)
            self.Relu2 = F.ReLu()
            self.maxpool2 = MaxPool2D([2, 2], [2, 2], padding="valid")

            self.conv3 = Conv2D(input_channels=32, output_channels=64, kernel_size=(3, 3), stride=(1, 1), padding="valid")
            self.BatchNorm2d3 = BatchNorm2d(64)
            self.Relu3 = F.ReLu()

            self.conv4 = Conv2D(input_channels=64, output_channels=128, kernel_size=(3, 3), stride=(1, 1), padding="valid")
            self.BatchNorm2d4 = BatchNorm2d(128)
            self.Relu4 = F.ReLu()
            self.maxpool4 = MaxPool2D([2, 2], [2, 2], padding="valid")

            self.fc1 = Linear(input_channels=128 * 4 * 4, output_channels=1024)
            self.fc2 = Linear(input_channels=1024, output_channels=128)
            self.fc3 = Linear(input_channels=128, output_channels=10)

        def forward(self, x):

            x = self.Relu1(self.conv1(x))
            x = self.maxpool2(self.Relu2(self.conv2(x)))
            x = self.Relu3(self.conv3(x))
            x = self.maxpool4(self.Relu4(self.conv4(x)))
            x = tensor.flatten(x, 1)
            x = F.ReLu()(self.fc1(x))  # 1 64
            x = F.ReLu()(self.fc2(x))  # 1 64
            x = self.fc3(x)  # 1 1
            return x

    def load_mnist(dataset="training_data", digits=np.arange(2), path="./"):         
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
        labels = np.zeros((N, 1), dtype=int)
        for i in range(len(ind)):
            images[i] = np.array(img[ind[i] * rows * cols: (ind[i] + 1) * rows * cols]).reshape((rows, cols))
            labels[i] = lbl[ind[i]]

        return images, labels


    """
    Abrufen der CNN-Modellparameter fuer Transfer Learning
    """

    train_size = 10000
    eval_size = 1000
    EPOCHES = 100
    def classcal_cnn_model_making():
        # Trainingsdaten laden
        x_train, y_train = load_mnist("training_data", digits=np.arange(10))
        x_test, y_test = load_mnist("testing_data", digits=np.arange(10))

        x_train = x_train[:train_size]
        y_train = y_train[:train_size]
        x_test = x_test[:eval_size]
        y_test = y_test[:eval_size]

        x_train = x_train / 255
        x_test = x_test / 255
        y_train = np.eye(10)[y_train].reshape(-1, 10)
        y_test = np.eye(10)[y_test].reshape(-1, 10)

        model = CNN()

        optimizer = SGD(model.parameters(), lr=0.005)
        loss_func = SoftmaxCrossEntropy()

        epochs = EPOCHES
        loss_list = []
        model.train()

        SAVE_FLAG = True
        temp_loss = 0
        for epoch in range(1, epochs):
            total_loss = []
            for x, y in data_generator(x_train, y_train, batch_size=4, shuffle=True):

                x = x.reshape(-1, 1, 28, 28)
                optimizer.zero_grad()
                # Vorwaertspass
                output = model(x)

                # Verlustberechnung
                loss = loss_func(y, output)  # Zielausgabe
                loss_np = np.array(loss.data)
                # Rueckwaertspass
                loss.backward()
                # Gewichte optimieren
                optimizer._step()

                total_loss.append(loss_np)

            loss_list.append(np.sum(total_loss) / len(total_loss))
            print("{:.0f} loss is : {:.10f}".format(epoch, loss_list[-1]))

            if SAVE_FLAG:
                temp_loss = loss_list[-1]
                save_parameters(model.state_dict(), "./result/QCNN_TL_1.model")
                SAVE_FLAG = False
            else:
                if temp_loss > loss_list[-1]:
                    temp_loss = loss_list[-1]
                    save_parameters(model.state_dict(), "./result/QCNN_TL_1.model")


        model.eval()
        correct = 0
        n_eval = 0

        for x, y in data_generator(x_test, y_test, batch_size=4, shuffle=True):
            x = x.reshape(-1, 1, 28, 28)
            output = model(x)
            loss = loss_func(y, output)
            np_output = np.array(output.data, copy=False)
            mask = (np_output.argmax(1) == y.argmax(1))
            correct += np.sum(np.array(mask))
            n_eval += 1
        print(f"Eval Accuracy: {correct / n_eval}")

        n_samples_show = 6
        count = 0
        fig, axes = plt.subplots(nrows=1, ncols=n_samples_show, figsize=(10, 3))
        model.eval()
        for x, y in data_generator(x_test, y_test, batch_size=1, shuffle=True):
            if count == n_samples_show:
                break
            x = x.reshape(-1, 1, 28, 28)
            output = model(x)
            pred = QTensor.argmax(output, [1],False)
            axes[count].imshow(x[0].squeeze(), cmap='gray')
            axes[count].set_xticks([])
            axes[count].set_yticks([])
            axes[count].set_title('Predicted {}'.format(np.array(pred.data)))
            count += 1
        plt.show()

    def classical_cnn_TransferLearning_predict():
        x_test, y_test = load_mnist("testing_data", digits=np.arange(10))

        x_test = x_test[:eval_size]
        y_test = y_test[:eval_size]
        x_test = x_test / 255
        y_test = np.eye(10)[y_test].reshape(-1, 10)
        model = CNN()

        model_parameter = load_parameters("./result/QCNN_TL_1.model")
        model.load_state_dict(model_parameter)
        model.eval()
        correct = 0
        n_eval = 0

        for x, y in data_generator(x_test, y_test, batch_size=1, shuffle=True):
            x = x.reshape(-1, 1, 28, 28)
            output = model(x)

            np_output = np.array(output.data, copy=False)
            mask = (np_output.argmax(1) == y.argmax(1))
            correct += np.sum(np.array(mask))
            n_eval += 1

        print(f"Eval Accuracy: {correct / n_eval}")

        n_samples_show = 6
        count = 0
        fig, axes = plt.subplots(nrows=1, ncols=n_samples_show, figsize=(10, 3))
        model.eval()
        for x, y in data_generator(x_test, y_test, batch_size=1, shuffle=True):
            if count == n_samples_show:
                break
            x = x.reshape(-1, 1, 28, 28)
            output = model(x)
            pred = QTensor.argmax(output, [1],False)
            axes[count].imshow(x[0].squeeze(), cmap='gray')
            axes[count].set_xticks([])
            axes[count].set_yticks([])
            axes[count].set_title('Predicted {}'.format(np.array(pred.data)))
            count += 1
        plt.show()

    def quantum_cnn_TransferLearning():

        n_qubits = 4  # Anzahl der Qubits
        q_depth = 6  # Tiefe der Quantenschaltung (Anzahl der Variationsschichten)

        def Q_H_layer(qubits, nqubits):
            """Schicht von Einzel-Qubit-Hadamard-Gattern.
            """
            circuit = pq.QCircuit()
            for idx in range(nqubits):
                circuit.insert(pq.H(qubits[idx]))
            return circuit

        def Q_RY_layer(qubits, w):
            """Schicht parametrisierter Qubit-Rotationen um die y-Achse.
            """
            circuit = pq.QCircuit()
            for idx, element in enumerate(w):
                circuit.insert(pq.RY(qubits[idx], element))
            return circuit

        def Q_entangling_layer(qubits, nqubits):
            """Schicht von CNOTs gefolgt von einer weiteren verschobenen CNOT-Schicht.
            """
            # Mit anderen Worten, sie sollte etwa Folgendes anwenden:
            # CNOT  CNOT  CNOT  CNOT...  CNOT
            #   CNOT  CNOT  CNOT...  CNOT
            circuit = pq.QCircuit()
            for i in range(0, nqubits - 1, 2):  # Iteration ueber gerade Indizes: i=0,2,...N-2
                circuit.insert(pq.CNOT(qubits[i], qubits[i + 1]))
            for i in range(1, nqubits - 1, 2):  # Iteration ueber ungerade Indizes: i=1,3,...N-3
                circuit.insert(pq.CNOT(qubits[i], qubits[i + 1]))
            return circuit

        def Q_quantum_net(q_input_features, q_weights_flat, qubits, cbits, machine):
            """
            Die variationsquantenschaltung.
            """
            machine = pq.CPUQVM()
            machine.init_qvm()
            qubits = machine.qAlloc_many(n_qubits)
            circuit = pq.QCircuit()

            # Gewichte umformen
            q_weights = q_weights_flat.reshape([q_depth, n_qubits])

            # Start vom Zustand |+>, unverzerrt bzgl. |0> und |1>
            circuit.insert(Q_H_layer(qubits, n_qubits))

            # Merkmale in den Quantenknoten einbetten
            circuit.insert(Q_RY_layer(qubits, q_input_features))

            # Sequenz von trainierbaren Variationsschichten
            for k in range(q_depth):
                circuit.insert(Q_entangling_layer(qubits, n_qubits))
                circuit.insert(Q_RY_layer(qubits, q_weights[k]))

            # Erwartungswerte in der Z-Basis
            prog = pq.QProg()
            prog.insert(circuit)

            exp_vals = []
            for position in range(n_qubits):
                pauli_str = "Z" + str(position)
                pauli_map = pq.PauliOperator(pauli_str, 1)
                hamiltion = pauli_map.toHamiltonian(True)
                exp = machine.get_expectation(prog, hamiltion, qubits)
                exp_vals.append(exp)

            return exp_vals

        class Q_DressedQuantumNet(Module):

            def __init__(self):
                """
                Definition des *verkleideten* Layouts.
                """

                super().__init__()
                self.pre_net = Linear(128, n_qubits)
                self.post_net = Linear(n_qubits, 10)
                self.temp_Q = QuantumLayer(Q_quantum_net, q_depth * n_qubits, "cpu", n_qubits, n_qubits)

            def forward(self, input_features):
                """
                Definition, wie Tensoren durch das *verkleidete* Quanten-Netzwerk
                fliessen sollen.
                """

                # Eingabemerkmale fuer die Quantenschaltung durch Reduzierung
                # der Merkmalsdimension von 512 auf 4 erhalten
                pre_out = self.pre_net(input_features)
                q_in = tensor.tanh(pre_out) * np.pi / 2.0
                q_out_elem = self.temp_Q(q_in)

                result = q_out_elem
                # Rueckgabe der zweidimensionalen Vorhersage aus der Nachbearbeitungsschicht
                return self.post_net(result)

        x_train, y_train = load_mnist("training_data", digits=np.arange(10))
        x_test, y_test = load_mnist("testing_data", digits=np.arange(10))
        x_train = x_train[:train_size]
        y_train = y_train[:train_size]
        x_test = x_test[:eval_size]
        y_test = y_test[:eval_size]

        x_train = x_train / 255
        x_test = x_test / 255
        y_train = np.eye(10)[y_train].reshape(-1, 10)
        y_test = np.eye(10)[y_test].reshape(-1, 10)

        model = CNN()
        model_param = load_parameters("./result/QCNN_TL_1.model")
        model.load_state_dict(model_param)

        loss_func = SoftmaxCrossEntropy()

        epochs = EPOCHES
        loss_list = []
        eval_losses = []

        model_hybrid = model
        print(model_hybrid)

        for param in model_hybrid.parameters():
            param.requires_grad = False
        model_hybrid.fc3 = Q_DressedQuantumNet()
        optimizer_hybrid = Adam(model_hybrid.fc3.parameters(), lr=0.001)
        model_hybrid.train()

        SAVE_FLAG = True
        temp_loss = 0
        for epoch in range(1, epochs):
            total_loss = []
            for x, y in data_generator(x_train, y_train, batch_size=4, shuffle=True):
                x = x.reshape(-1, 1, 28, 28)
                optimizer_hybrid.zero_grad()
                # Vorwaertspass
                output = model_hybrid(x)

                loss = loss_func(y, output)  # Zielausgabe
                loss_np = np.array(loss.data)
                # Rueckwaertspass
                loss.backward()
                # Gewichte optimieren
                optimizer_hybrid._step()
                total_loss.append(loss_np)

            loss_list.append(np.sum(total_loss) / len(total_loss))
            print("{:.0f} loss is : {:.10f}".format(epoch, loss_list[-1]))
            if SAVE_FLAG:
                temp_loss = loss_list[-1]
                save_parameters(model_hybrid.fc3.state_dict(), "./result/QCNN_TL_FC3.model")
                save_parameters(model_hybrid.state_dict(), "./result/QCNN_TL_ALL.model")
                SAVE_FLAG = False
            else:
                if temp_loss > loss_list[-1]:
                    temp_loss = loss_list[-1]
                    save_parameters(model_hybrid.fc3.state_dict(), "./result/QCNN_TL_FC3.model")
                    save_parameters(model_hybrid.state_dict(), "./result/QCNN_TL_ALL.model")

            correct = 0
            n_eval = 0
            loss_temp =[]
            for x1, y1 in data_generator(x_test, y_test, batch_size=4, shuffle=True):
                x1 = x1.reshape(-1, 1, 28, 28)
                output = model_hybrid(x1)
                loss = loss_func(y1, output)
                np_loss = np.array(loss.data)
                np_output = np.array(output.data, copy=False)
                mask = (np_output.argmax(1) == y1.argmax(1))
                correct += np.sum(np.array(mask))
                n_eval += 1
                loss_temp.append(np_loss)
            eval_losses.append(np.sum(loss_temp) / n_eval)
            print("{:.0f} eval loss is : {:.10f}".format(epoch, eval_losses[-1]))


        plt.title('model loss')
        plt.plot(loss_list, color='green', label='train_losses')
        plt.plot(eval_losses, color='red', label='eval_losses')
        plt.ylabel('loss')
        plt.legend(["train_losses", "eval_losses"])
        plt.savefig("qcnn_transfer_learning_classical")
        plt.show()
        plt.close()

        n_samples_show = 6
        count = 0
        fig, axes = plt.subplots(nrows=1, ncols=n_samples_show, figsize=(10, 3))
        model_hybrid.eval()
        for x, y in data_generator(x_test, y_test, batch_size=1, shuffle=True):
            if count == n_samples_show:
                break
            x = x.reshape(-1, 1, 28, 28)
            output = model_hybrid(x)
            pred = QTensor.argmax(output, [1],False)
            axes[count].imshow(x[0].squeeze(), cmap='gray')
            axes[count].set_xticks([])
            axes[count].set_yticks([])
            axes[count].set_title('Predicted {}'.format(np.array(pred.data)))
            count += 1
        plt.show()

    def quantum_cnn_TransferLearning_predict():

        n_qubits = 4  # Anzahl der Qubits
        q_depth = 6  # Tiefe der Quantenschaltung (Anzahl der Variationsschichten)

        def Q_H_layer(qubits, nqubits):
            """Schicht von Einzel-Qubit-Hadamard-Gattern.
            """
            circuit = pq.QCircuit()
            for idx in range(nqubits):
                circuit.insert(pq.H(qubits[idx]))
            return circuit

        def Q_RY_layer(qubits, w):
            """Schicht parametrisierter Qubit-Rotationen um die y-Achse.
            """
            circuit = pq.QCircuit()
            for idx, element in enumerate(w):
                circuit.insert(pq.RY(qubits[idx], element))
            return circuit

        def Q_entangling_layer(qubits, nqubits):
            """Schicht von CNOTs gefolgt von einer weiteren verschobenen CNOT-Schicht.
            """
            # Mit anderen Worten, sie sollte etwa Folgendes anwenden:
            # CNOT  CNOT  CNOT  CNOT...  CNOT
            #   CNOT  CNOT  CNOT...  CNOT
            circuit = pq.QCircuit()
            for i in range(0, nqubits - 1, 2):  # Iteration ueber gerade Indizes: i=0,2,...N-2
                circuit.insert(pq.CNOT(qubits[i], qubits[i + 1]))
            for i in range(1, nqubits - 1, 2):  # Iteration ueber ungerade Indizes: i=1,3,...N-3
                circuit.insert(pq.CNOT(qubits[i], qubits[i + 1]))
            return circuit

        def Q_quantum_net(q_input_features, q_weights_flat, qubits, cbits, machine):
            """
            Die variationsquantenschaltung.
            """
            machine = pq.CPUQVM()
            machine.init_qvm()
            qubits = machine.qAlloc_many(n_qubits)
            circuit = pq.QCircuit()

            # Gewichte umformen
            q_weights = q_weights_flat.reshape([q_depth, n_qubits])

            # Start vom Zustand |+>, unverzerrt bzgl. |0> und |1>
            circuit.insert(Q_H_layer(qubits, n_qubits))

            # Merkmale in den Quantenknoten einbetten
            circuit.insert(Q_RY_layer(qubits, q_input_features))

            # Sequenz von trainierbaren Variationsschichten
            for k in range(q_depth):
                circuit.insert(Q_entangling_layer(qubits, n_qubits))
                circuit.insert(Q_RY_layer(qubits, q_weights[k]))

            # Erwartungswerte in der Z-Basis
            prog = pq.QProg()
            prog.insert(circuit)
            exp_vals = []
            for position in range(n_qubits):
                pauli_str = "Z" + str(position)
                pauli_map = pq.PauliOperator(pauli_str, 1)
                hamiltion = pauli_map.toHamiltonian(True)
                exp = machine.get_expectation(prog, hamiltion, qubits)
                exp_vals.append(exp)

            return exp_vals

        class Q_DressedQuantumNet(Module):

            def __init__(self):
                """
                Definition des *verkleideten* Layouts.
                """

                super().__init__()
                self.pre_net = Linear(128, n_qubits)
                self.post_net = Linear(n_qubits, 10)
                self.temp_Q = QuantumLayer(Q_quantum_net, q_depth * n_qubits, "cpu", n_qubits, n_qubits)

            def forward(self, input_features):
                """
                Definition, wie Tensoren durch das *verkleidete* Quanten-Netzwerk
                fliessen sollen.
                """

                # Eingabemerkmale fuer die Quantenschaltung durch Reduzierung
                # der Merkmalsdimension von 512 auf 4 erhalten
                pre_out = self.pre_net(input_features)
                q_in = tensor.tanh(pre_out) * np.pi / 2.0
                q_out_elem = self.temp_Q(q_in)

                result = q_out_elem
                # Rueckgabe der zweidimensionalen Vorhersage aus der Nachbearbeitungsschicht
                return self.post_net(result)

        x_train, y_train = load_mnist("training_data", digits=np.arange(10))
        x_test, y_test = load_mnist("testing_data", digits=np.arange(10))
        x_train = x_train[:2000]
        y_train = y_train[:2000]
        x_test = x_test[:500]
        y_test = y_test[:500]

        x_train = x_train / 255
        x_test = x_test / 255
        y_train = np.eye(10)[y_train].reshape(-1, 10)
        y_test = np.eye(10)[y_test].reshape(-1, 10)

        model = CNN()
        model_hybrid = model
        model_hybrid.fc3 = Q_DressedQuantumNet()
        for param in model_hybrid.parameters():
            param.requires_grad = False
        model_param_quantum = load_parameters("./result/QCNN_TL_ALL.model")

        model_hybrid.load_state_dict(model_param_quantum)
        model_hybrid.eval()

        loss_func = SoftmaxCrossEntropy()
        eval_losses = []

        correct = 0
        n_eval = 0
        loss_temp =[]
        eval_batch_size = 4
        for x1, y1 in data_generator(x_test, y_test, batch_size=eval_batch_size, shuffle=True):
            x1 = x1.reshape(-1, 1, 28, 28)
            output = model_hybrid(x1)
            loss = loss_func(y1, output)
            np_loss = np.array(loss.data)
            np_output = np.array(output.data, copy=False)
            mask = (np_output.argmax(1) == y1.argmax(1))
            correct += np.sum(np.array(mask))

            n_eval += 1
            loss_temp.append(np_loss)

        eval_losses.append(np.sum(loss_temp) / n_eval)
        print(f"Eval Accuracy: {correct / (eval_batch_size*n_eval)}")

        n_samples_show = 6
        count = 0
        fig, axes = plt.subplots(nrows=1, ncols=n_samples_show, figsize=(10, 3))
        model_hybrid.eval()
        for x, y in data_generator(x_test, y_test, batch_size=1, shuffle=True):
            if count == n_samples_show:
                break
            x = x.reshape(-1, 1, 28, 28)
            output = model_hybrid(x)
            pred = QTensor.argmax(output, [1],False)
            axes[count].imshow(x[0].squeeze(), cmap='gray')
            axes[count].set_xticks([])
            axes[count].set_yticks([])
            axes[count].set_title('Predicted {}'.format(np.array(pred.data)))
            count += 1
        plt.show()

    if __name__ == "__main__":
        # klassische Modellparameter speichern
        if not os.path.exists('./result/QCNN_TL_1.model'):
            classcal_cnn_model_making()
            classical_cnn_TransferLearning_predict()
        #Quantenschaltungen trainieren.
        print("Verwende vorhandene CNN-Modellparameter zum Trainieren der Quantenparameter.")
        quantum_cnn_TransferLearning()
        #Quantenschaltungen auswerten.
        quantum_cnn_TransferLearning_predict()


Verlust auf dem Trainingsdatensatz

.. figure:: ./images/qcnn_transfer_learning_classical.png
   :width: 600 px
   :align: center

|

Klassifikation auf dem Testdatensatz ausfuehren

.. figure:: ./images/qcnn_transfer_learning_predict.png
   :width: 600 px
   :align: center

|

3. Hybrides quantenklassisches Unet-Netzwerkmodell
==============================================================================

Bildsegmentierung ist ein klassisches Problem in der Forschung des Computersehens und hat sich zu einem Brennpunkt
im Bereich der Bildverarbeitung entwickelt. Bildsegmentierung ist ein wichtiger Bestandteil der Bildverarbeitung und eines der schwierigsten Probleme in der Bildverarbeitung.
Die sogenannte Bildsegmentierung bezieht sich auf die Segmentierung basierend auf Grauwert, Farbe und raeumlicher Textur. Das Bild
wird durch Merkmale wie Theorie und Geometrie in mehrere disjunkte Regionen unterteilt, sodass diese Merkmale in derselben Region
Konsistenz oder Aehnlichkeit und zwischen verschiedenen Regionen deutliche Unterschiede aufweisen. Kurz gesagt,
es geht darum, ein Bild zu nehmen und jedes Pixel auf dem Bild zu klassifizieren. Die Pixelbereiche, die zu
verschiedenen Objekten gehoeren, werden getrennt. `Unet <https://arxiv.org/abs/1505.04597>`_ ist ein klassischer Bildsegmentierungsalgorithmus.

Hier untersuchen wir, wie das klassische neuronale Netzwerk teilweise quantisiert werden kann, um ein hybrides quantenklassisches
`QUnet`-neuronales Netzwerk zu erstellen. Wir werden ein einfaches Beispiel zur Integration von pyQPanda2 mit VQNet schreiben.
QUnet wird hauptsaechlich zur Loesung der Bildsegmentierungstechnologie verwendet.



Datenvorbereitung
-------------------------

Wir werden die Daten der offiziellen Bibliothek `VOC2012 <http://host.robots.ox.ac.uk/pascal/VOC/voc2012/#devkit>`_ als Bildsegmentierungsdaten verwenden. Diese Proben werden
in Trainingsdaten (training_data) und Testdaten (testing_data) aufgeteilt.

.. figure:: ./images/Unet_data_imshow.png
   :width: 600 px
   :align: center

|

Aufbau von Quantenschaltungen
-------------------------------------------
In diesem Beispiel definieren wir eine Quantenschaltung mit pyqpanda2. Die eingegebenen 3-Kanal-Farbbilddaten
werden in ein einkanaliges Graustufenbild komprimiert und gespeichert, und dann werden die Merkmale der Daten
extrahiert und durch Quantenfaltungsoperationen dimensionsreduziert.


.. figure:: ./images/qunet_cir.png
   :width: 600 px
   :align: center

|

Importieren der erforderlichen Bibliotheken und Funktionen

.. code-block::

    import os
    import numpy as np
    from pyvqnet.nn.module import Module
    from pyvqnet.nn.conv import Conv2D, ConvT2D
    from pyvqnet.nn import activation as F
    from pyvqnet.nn.batch_norm import BatchNorm2d
    from pyvqnet.nn.loss import BinaryCrossEntropy
    from pyvqnet.optim.adam import Adam
    from pyvqnet.dtype import *
    from pyvqnet.tensor import tensor,kfloat32
    from pyvqnet.tensor.tensor import QTensor
    import pyqpanda as pq
    from pyqpanda import *
    from pyvqnet.utils.storage import load_parameters, save_parameters

    import matplotlib
    try:
        matplotlib.use('TkAgg')
    except:
        pass
    import matplotlib.pyplot as plt

    import cv2

Datenvorverarbeitung

.. code-block::

    # Datenvorverarbeitung
    class PreprocessingData:
        def __init__(self, path):
            self.path = path
            self.x_data = []
            self.y_label = []


        def processing(self):
            list_path = os.listdir((self.path+"/images"))
            for i in range(len(list_path)):

                temp_data = cv2.imread(self.path+"/images" + '/' + list_path[i], cv2.IMREAD_COLOR)
                temp_data = cv2.resize(temp_data, (128, 128))
                grayimg = cv2.cvtColor(temp_data, cv2.COLOR_BGR2GRAY)
                temp_data = grayimg.reshape(temp_data.shape[0], temp_data.shape[0], 1).astype(np.float32)
                self.x_data.append(temp_data)

                label_data = cv2.imread(self.path+"/labels" + '/' +list_path[i].split(".")[0] + ".png", cv2.IMREAD_COLOR)
                label_data = cv2.resize(label_data, (128, 128))

                label_data = cv2.cvtColor(label_data, cv2.COLOR_BGR2GRAY)
                label_data = label_data.reshape(label_data.shape[0], label_data.shape[0], 1).astype(np.int64)
                self.y_label.append(label_data)

            return self.x_data, self.y_label

        def read(self):
            self.x_data, self.y_label = self.processing()
            x_data = np.array(self.x_data)
            y_label = np.array(self.y_label)

            return x_data, y_label

    # Quanten-Kodierungsschaltung
    class QCNN_:
        def __init__(self, image):
            self.image = image

        def encode_cir(self, qlist, pixels):
            cir = pq.QCircuit()
            for i, pix in enumerate(pixels):
                theta = np.arctan(pix)
                phi = np.arctan(pix**2)
                cir.insert(pq.RY(qlist[i], theta))
                cir.insert(pq.RZ(qlist[i], phi))
            return cir

        def entangle_cir(self, qlist):
            k_size = len(qlist)
            cir = pq.QCircuit()
            for i in range(k_size):
                ctr = i
                ctred = i+1
                if ctred == k_size:
                    ctred = 0
                cir.insert(pq.CNOT(qlist[ctr], qlist[ctred]))
            return cir

        def qcnn_circuit(self, pixels):
            k_size = len(pixels)
            machine = pq.MPSQVM()
            machine.init_qvm()
            qlist = machine.qAlloc_many(k_size)
            cir = pq.QProg()

            cir.insert(self.encode_cir(qlist, np.array(pixels) * np.pi / 2))
            cir.insert(self.entangle_cir(qlist))

            result0 = machine.prob_run_list(cir, [qlist[0]], -1)
            result1 = machine.prob_run_list(cir, [qlist[1]], -1)
            result2 = machine.prob_run_list(cir, [qlist[2]], -1)
            result3 = machine.prob_run_list(cir, [qlist[3]], -1)

            result = [result0[-1]+result1[-1]+result2[-1]+result3[-1]]
            machine.finalize()
            return result

    def quanconv_(image):
        """Faltet das Eingabebild mit mehreren Anwendungen derselben Quantenschaltung."""
        out = np.zeros((64, 64, 1))
        
        for j in range(0, 128, 2):
            for k in range(0, 128, 2):
                # Verarbeitet einen quadratischen 2x2-Bereich des Bildes mit einer Quantenschaltung
                q_results = QCNN_(image).qcnn_circuit(
                    [
                        image[j, k, 0],
                        image[j, k + 1, 0],
                        image[j + 1, k, 0],
                        image[j + 1, k + 1, 0]
                    ]
                )
                
                for c in range(1):
                    out[j // 2, k // 2, c] = q_results[c]
        return out

    def quantum_data_preprocessing(images):
        quantum_images = []
        for _, img in enumerate(images):
            quantum_images.append(quanconv_(img))
        quantum_images = np.asarray(quantum_images)
        return quantum_images

Aufbau eines hybriden quantenklassischen neuronalen Netzes
----------------------------------------------------------

Entsprechend dem Unet-Netzwerkframework verwenden wir das `VQNet`-Framework, um den klassischen Netzwerkteil zu erstellen.
Die Downsampling-Neuronale-Netzwerkschicht wird zur Dimensionsreduktion und Merkmalsextraktion verwendet;
Die Upsampling-Neuronale-Netzwerkschicht wird zur Dimensionswiederherstellung verwendet; Die Auf- und Abtastschichten
sind durch Concatenate zur Merkmalsfusion verbunden.


.. figure:: ./images/Unet.png
   :width: 600 px
   :align: center

|

.. code-block::

    # Definition der Downsampling-Neuronalen-Netzwerkschicht
    class DownsampleLayer(Module):
        def __init__(self, in_ch, out_ch):
            super(DownsampleLayer, self).__init__()
            self.conv1 = Conv2D(input_channels=in_ch, output_channels=out_ch, kernel_size=(3, 3), stride=(1, 1),
                                padding="same")
            self.BatchNorm2d1 = BatchNorm2d(out_ch)
            self.Relu1 = F.ReLu()
            self.conv2 = Conv2D(input_channels=out_ch, output_channels=out_ch, kernel_size=(3, 3), stride=(1, 1),
                                padding="same")
            self.BatchNorm2d2 = BatchNorm2d(out_ch)
            self.Relu2 = F.ReLu()
            self.conv3 = Conv2D(input_channels=out_ch, output_channels=out_ch, kernel_size=(3, 3), stride=(2, 2),
                                padding=(1,1))
            self.BatchNorm2d3 = BatchNorm2d(out_ch)
            self.Relu3 = F.ReLu()

        def forward(self, x):
            """
            :param x:
            :return: out(Ausgabe an tiefere Schicht), out_2(Eingabe fuer naechste Ebene),
            """
            x1 = self.conv1(x)
            x2 = self.BatchNorm2d1(x1)
            x3 = self.Relu1(x2)
            x4 = self.conv2(x3)
            x5 = self.BatchNorm2d2(x4)
            out = self.Relu2(x5)
            x6 = self.conv3(out)
            x7 = self.BatchNorm2d3(x6)
            out_2 = self.Relu3(x7)
            return out, out_2

    # Definition der Upsampling-Neuronalen-Netzwerkschicht
    class UpSampleLayer(Module):
        def __init__(self, in_ch, out_ch):
            super(UpSampleLayer, self).__init__()

            self.conv1 = Conv2D(input_channels=in_ch, output_channels=out_ch * 2, kernel_size=(3, 3), stride=(1, 1),
                                padding="same")
            self.BatchNorm2d1 = BatchNorm2d(out_ch * 2)
            self.Relu1 = F.ReLu()
            self.conv2 = Conv2D(input_channels=out_ch * 2, output_channels=out_ch * 2, kernel_size=(3, 3), stride=(1, 1),
                                padding="same")
            self.BatchNorm2d2 = BatchNorm2d(out_ch * 2)
            self.Relu2 = F.ReLu()

            self.conv3 = ConvT2D(input_channels=out_ch * 2, output_channels=out_ch, kernel_size=(3, 3), stride=(2, 2),
                                 padding=(1,1))
            self.BatchNorm2d3 = BatchNorm2d(out_ch)
            self.Relu3 = F.ReLu()

        def forward(self, x):
            '''
            :param x: Eingabe-Faltungsschicht
            :param out: Verbindung mit UpSampleLayer
            :return:
            '''
            x = self.conv1(x)
            x = self.BatchNorm2d1(x)
            x = self.Relu1(x)
            x = self.conv2(x)
            x = self.BatchNorm2d2(x)
            x = self.Relu2(x)
            x = self.conv3(x)
            x = self.BatchNorm2d3(x)
            x_out = self.Relu3(x)
            return x_out

    # Unet-Gesamtnetzwerkarchitektur
    class UNet(Module):
        def __init__(self):
            super(UNet, self).__init__()
            out_channels = [2 ** (i + 4) for i in range(5)]

            # DownSampleLayer
            self.d1 = DownsampleLayer(1, out_channels[0])  # 3-64
            self.d2 = DownsampleLayer(out_channels[0], out_channels[1])  # 64-128
            self.d3 = DownsampleLayer(out_channels[1], out_channels[2])  # 128-256
            self.d4 = DownsampleLayer(out_channels[2], out_channels[3])  # 256-512
            # UpSampleLayer
            self.u1 = UpSampleLayer(out_channels[3], out_channels[3])  # 512-1024-512
            self.u2 = UpSampleLayer(out_channels[4], out_channels[2])  # 1024-512-256
            self.u3 = UpSampleLayer(out_channels[3], out_channels[1])  # 512-256-128
            self.u4 = UpSampleLayer(out_channels[2], out_channels[0])  # 256-128-64
            # Ausgabe
            self.conv1 = Conv2D(input_channels=out_channels[1], output_channels=out_channels[0], kernel_size=(3, 3),
                                stride=(1, 1), padding="same")
            self.BatchNorm2d1 = BatchNorm2d(out_channels[0])
            self.Relu1 = F.ReLu()
            self.conv2 = Conv2D(input_channels=out_channels[0], output_channels=out_channels[0], kernel_size=(3, 3),
                                stride=(1, 1), padding="same")
            self.BatchNorm2d2 = BatchNorm2d(out_channels[0])
            self.Relu2 = F.ReLu()
            self.conv3 = Conv2D(input_channels=out_channels[0], output_channels=1, kernel_size=(3, 3),
                                stride=(1, 1), padding="same")
            self.Sigmoid = F.Sigmoid()

        def forward(self, x):
            out_1, out1 = self.d1(x)
            out_2, out2 = self.d2(out1)
            out_3, out3 = self.d3(out2)
            out_4, out4 = self.d4(out3)

            out5 = self.u1(out4)
            out5_pad_out4 = tensor.pad2d(out5, (1, 0, 1, 0), 0)
            cat_out5 = tensor.concatenate([out5_pad_out4, out_4], axis=1)

            out6 = self.u2(cat_out5)
            out6_pad_out_3 = tensor.pad2d(out6, (1, 0, 1, 0), 0)
            cat_out6 = tensor.concatenate([out6_pad_out_3, out_3], axis=1)

            out7 = self.u3(cat_out6)
            out7_pad_out_2 = tensor.pad2d(out7, (1, 0, 1, 0), 0)
            cat_out7 = tensor.concatenate([out7_pad_out_2, out_2], axis=1)

            out8 = self.u4(cat_out7)
            out8_pad_out_1 = tensor.pad2d(out8, (1, 0, 1, 0), 0)
            cat_out8 = tensor.concatenate([out8_pad_out_1, out_1], axis=1)
            out = self.conv1(cat_out8)
            out = self.BatchNorm2d1(out)
            out = self.Relu1(out)
            out = self.conv2(out)
            out = self.BatchNorm2d2(out)
            out = self.Relu2(out)
            out = self.conv3(out)
            out = self.Sigmoid(out)
            return out

Training und Modellspeicherung
-------------------------------

Aehnlich wie beim Training eines klassischen neuronalen Netzwerkmodells
muessen wir auch das Modell instanziieren, die Verlustfunktion und den Optimierer definieren und den gesamten Trainings- und
Testprozess definieren. Fuer das hybride neuronale Netzwerkmodell, wie in der folgenden Abbildung gezeigt, berechnen wir den Verlustwert in der
Vorwaertsfunktion, den Gradienten jedes Parameters in der
Rueckwaertsberechnung automatisch, und verwenden den Optimierer, um die Parameter zu optimieren, bis die Anzahl der
Iterationen den vorgegebenen Wert erreicht. Wenn ``PREPROCESS`` auf False gesetzt ist, wird die Quantendatenvorverarbeitung uebersprungen.

.. code-block::

    PREPROCESS = True

    class MyDataset():
        def __init__(self, x_data, x_label):
            self.x_set = x_data
            self.label = x_label

        def __getitem__(self, item):
            img, target = self.x_set[item], self.label[item]
            img_np = np.uint8(img).transpose(2, 0, 1)
            target_np = np.uint8(target).transpose(2, 0, 1)

            img = img_np
            target = target_np
            return img, target

        def __len__(self):
            return len(self.x_set)

    if not os.path.exists("./result"):
        os.makedirs("./result")
    else:
        pass
    if not os.path.exists("./Intermediate_results"):
        os.makedirs("./Intermediate_results")
    else:
        pass

    # Trainings-/Testdaten und Labels vorbereiten
    path0 = 'training_data'
    path1 = 'testing_data'
    train_images, train_labels = PreprocessingData(path0).read()
    test_images, test_labels = PreprocessingData(path1).read()

    print('train: ', train_images.shape, '\ntest: ', test_images.shape)
    print('train: ', train_labels.shape, '\ntest: ', test_labels.shape)
    train_images = train_images / 255
    test_images = test_images / 255

    # Quantenencoder zur Datenvorverarbeitung verwenden

    if PREPROCESS == True:
        print("Quanten-Vorverarbeitung der Trainingsbilder:")
        q_train_images = quantum_data_preprocessing(train_images)
        q_test_images = quantum_data_preprocessing(test_images)
        q_train_label = quantum_data_preprocessing(train_labels)
        q_test_label = quantum_data_preprocessing(test_labels)

        # Vorverarbeitete Bilder speichern
        print('Quantendaten werden gespeichert...')
        np.save("./result/q_train.npy", q_train_images)
        np.save("./result/q_test.npy", q_test_images)
        np.save("./result/q_train_label.npy", q_train_label)
        np.save("./result/q_test_label.npy", q_test_label)
        print('Quantendaten speichern abgeschlossen!')

    # Quantendaten laden
    SAVE_PATH = "./result/"
    train_x = np.load(SAVE_PATH + "q_train.npy")
    train_labels = np.load(SAVE_PATH + "q_train_label.npy")
    test_x = np.load(SAVE_PATH + "q_test.npy")
    test_labels = np.load(SAVE_PATH + "q_test_label.npy")

    train_x = train_x.astype(np.uint8)
    test_x = test_x.astype(np.uint8)
    train_labels = train_labels.astype(np.uint8)
    test_labels = test_labels.astype(np.uint8)
    train_y = train_labels
    test_y = test_labels

    trainset = MyDataset(train_x, train_y)
    testset = MyDataset(test_x, test_y)
    
    x_train = []
    y_label = []
    model = UNet()
    optimizer = Adam(model.parameters(), lr=0.01)
    loss_func = BinaryCrossEntropy()
    epochs = 200

    loss_list = []
    SAVE_FLAG = True
    temp_loss = 0
    file = open("./result/result.txt", 'w').close()
    for epoch in range(1, epochs):
        total_loss = []
        model.train()
        for i, (x, y) in enumerate(trainset):
            x_img = QTensor(x, dtype=kfloat32)
            x_img_Qtensor = tensor.unsqueeze(x_img, 0)
            y_img = QTensor(y, dtype=kfloat32)
            y_img_Qtensor = tensor.unsqueeze(y_img, 0)
            optimizer.zero_grad()
            img_out = model(x_img_Qtensor)

            print(f"=========={epoch}==================")
            loss = loss_func(y_img_Qtensor, img_out)  # Zielausgabe
            if i == 1:
                plt.figure()
                plt.subplot(1, 2, 1)
                plt.title("predict")
                img_out_tensor = tensor.squeeze(img_out, 0)

                if matplotlib.__version__ >= '3.4.2':
                    plt.imshow(np.array(img_out_tensor.data).transpose([1, 2, 0]))
                else:
                    plt.imshow(np.array(img_out_tensor.data).transpose([1, 2, 0]).squeeze(2))
                plt.subplot(1, 2, 2)
                plt.title("label")
                y_img_tensor = tensor.squeeze(y_img_Qtensor, 0)
                if matplotlib.__version__ >= '3.4.2':
                    plt.imshow(np.array(y_img_tensor.data).transpose([1, 2, 0]))
                else:
                    plt.imshow(np.array(y_img_tensor.data).transpose([1, 2, 0]).squeeze(2))

                plt.savefig("./Intermediate_results/" + str(epoch) + "_" + str(i) + ".jpg")

            loss_data = np.array(loss.data)
            print("{} - {} loss_data: {}".format(epoch, i, loss_data))
            loss.backward()
            optimizer._step()
            total_loss.append(loss_data)

        loss_list.append(np.sum(total_loss) / len(total_loss))
        out_read = open("./result/result.txt", 'a')
        out_read.write(str(loss_list[-1]))
        out_read.write(str("\n"))
        out_read.close()
        print("{:.0f} loss is : {:.10f}".format(epoch, loss_list[-1]))
        if SAVE_FLAG:
            temp_loss = loss_list[-1]
            save_parameters(model.state_dict(), "./result/Q-Unet_End.model")
            SAVE_FLAG = False
        else:
            if temp_loss > loss_list[-1]:
                temp_loss = loss_list[-1]
                save_parameters(model.state_dict(), "./result/Q-Unet_End.model")


Datenvisualisierung
-----------------------

Die Verlustfunktionskurve der Trainingsdaten wird angezeigt und gespeichert, und die Testergebnisse werden gespeichert.

.. code-block::

    out_read = open("./result/result.txt", 'r')
    plt.figure()
    lines_read = out_read.readlines()
    data_read = []
    for line in lines_read:
        float_line = float(line)
        data_read.append(float_line)
    out_read.close()
    plt.plot(data_read)
    plt.title('Unet Training')
    plt.xlabel('Training Iterations')
    plt.ylabel('Loss')
    plt.savefig("./result/traing_loss.jpg")

    modela = load_parameters("./result/Q-Unet_End.model")
    print("----------------PREDICT-------------")
    model.load_state_dict(modela)
    model.eval()

    for i, (x1, y1) in enumerate(testset):
        x_img = QTensor(x1, dtype=kfloat32)
        x_img_Qtensor = tensor.unsqueeze(x_img, 0)
        y_img = QTensor(y1, dtype=kfloat32)
        y_img_Qtensor = tensor.unsqueeze(y_img, 0)
        img_out = model(x_img_Qtensor)
        loss = loss_func(y_img_Qtensor, img_out)
        loss_data = np.array(loss.data)
        print("{} loss_eval: {}".format(i, loss_data))
        plt.figure()
        plt.subplot(1, 2, 1)
        plt.title("predict")
        img_out_tensor = tensor.squeeze(img_out, 0)
        if matplotlib.__version__ >= '3.4.2':
            plt.imshow(np.array(img_out_tensor.data).transpose([1, 2, 0]))
        else:
            plt.imshow(np.array(img_out_tensor.data).transpose([1, 2, 0]).squeeze(2))
        plt.subplot(1, 2, 2)
        plt.title("label")
        y_img_tensor = tensor.squeeze(y_img_Qtensor, 0)
        if matplotlib.__version__ >= '3.4.2':
            plt.imshow(np.array(y_img_tensor.data).transpose([1, 2, 0]))
        else:
            plt.imshow(np.array(y_img_tensor.data).transpose([1, 2, 0]).squeeze(2))
        plt.savefig("./result/" + str(i) + "_1" + ".jpg")
    print("end!")

Verlust auf dem Trainingsdatensatz

.. figure:: ./images/qunet_train_loss.png
   :width: 600 px
   :align: center

|

Klassifikation auf dem Testdatensatz ausfuehren

.. figure:: ./images/qunet_eval_1.jpg
   :width: 600 px
   :align: center

.. figure:: ./images/qunet_eval_2.jpg
   :width: 600 px
   :align: center

.. figure:: ./images/qunet_eval_3.jpg
   :width: 600 px
   :align: center

|

4. Hybrides quantenklassisches QMLP-Netzwerkmodell
===============================================================================

Wir stellen eine vorgeschlagene Quanten-Multilayer-Perzeptron (QMLP)-Architektur vor und analysieren sie, die sich durch fehlertolerante Eingabeeinbettungen, reichhaltige Nichtlinearitaeten und erweiterte Variationsschaltungssimulationen mit parametrisierten Zwei-Qubit-Verschraenkungsgattern auszeichnet.
`QMLP: An Error-Tolerant Nonlinear Quantum MLP Architecture using Parameterized Two-Qubit Gates <https://arxiv.org/pdf/2206.01345.pdf>`_ .
Wir werden ein einfaches Beispiel zur Integration von pyQPanda2 mit VQNet schreiben.

Aufbau hybrider quantenklassischer neuronaler Netze
----------------------------------------------------------

.. code-block::

    import os
    import gzip
    import struct
    import numpy as np
    import pyqpanda as pq
    from pyvqnet.nn.module import Module
    from pyvqnet.nn.loss import MeanSquaredError, CrossEntropyLoss
    from pyvqnet.optim.adam import Adam
    from pyvqnet.tensor.tensor import QTensor
    from pyvqnet.qnn.measure import expval
    from pyvqnet.qnn.quantumlayer import QuantumLayer
    from pyvqnet.nn.pooling import AvgPool2D
    from pyvqnet.nn.linear import Linear
    from pyvqnet.data.data import data_generator
    from pyvqnet.tensor import tensor
    import matplotlib
    from matplotlib import pyplot as plt
    try:
        matplotlib.use("TkAgg")
    except:
        print("Can not use matplot TkAgg")

    try:
        import urllib.request
    except ImportError:
        raise ImportError("You should use Python 3.x")

    url_base = "https://ossci-datasets.s3.amazonaws.com/mnist/"
    key_file = {
        "train_img": "train-images-idx3-ubyte.gz",
        "train_label": "train-labels-idx1-ubyte.gz",
        "test_img": "t10k-images-idx3-ubyte.gz",
        "test_label": "t10k-labels-idx1-ubyte.gz"
    }


    def _download(dataset_dir, file_name):
        """
        Laedt MNIST-Daten bei Bedarf herunter.
        """
        file_path = dataset_dir + "/" + file_name

        if os.path.exists(file_path):
            with gzip.GzipFile(file_path) as file:
                file_path_ungz = file_path[:-3].replace("\\", "/")
                if not os.path.exists(file_path_ungz):
                    open(file_path_ungz, "wb").write(file.read())
            return

        print("Downloading " + file_name + " ... ")
        urllib.request.urlretrieve(url_base + file_name, file_path)
        if os.path.exists(file_path):
            with gzip.GzipFile(file_path) as file:
                file_path_ungz = file_path[:-3].replace("\\", "/")
                file_path_ungz = file_path_ungz.replace("-idx", ".idx")
                if not os.path.exists(file_path_ungz):
                    open(file_path_ungz, "wb").write(file.read())
        print("Done")

    def download_mnist(dataset_dir):
        for v in key_file.values():
            _download(dataset_dir, v)

    def load_mnist(dataset="training_data", digits=np.arange(2), path="./"):
        """
        Laedt MNIST-Daten
        """
        from array import array as pyarray
        download_mnist(path)
        if dataset == "training_data":
            fname_image = os.path.join(path, "train-images.idx3-ubyte").replace(
                "\\", "/")
            fname_label = os.path.join(path, "train-labels.idx1-ubyte").replace(
                "\\", "/")
        elif dataset == "testing_data":
            fname_image = os.path.join(path, "t10k-images.idx3-ubyte").replace(
                "\\", "/")
            fname_label = os.path.join(path, "t10k-labels.idx1-ubyte").replace(
                "\\", "/")
        else:
            raise ValueError("dataset must be 'training_data' or 'testing_data'")

        flbl = open(fname_label, "rb")
        _, size = struct.unpack(">II", flbl.read(8))
        lbl = pyarray("b", flbl.read())
        flbl.close()

        fimg = open(fname_image, "rb")
        _, size, rows, cols = struct.unpack(">IIII", fimg.read(16))
        img = pyarray("B", fimg.read())
        fimg.close()

        ind = [k for k in range(size) if lbl[k] in digits]
        num = len(ind)
        images = np.zeros((num, rows, cols))
        labels = np.zeros((num, 1), dtype=int)
        for i in range(len(ind)):
            images[i] = np.array(img[ind[i] * rows * cols:(ind[i] + 1) * rows *
                                     cols]).reshape((rows, cols))
            labels[i] = lbl[ind[i]]

        return images, labels

    def data_select(train_num, test_num):
        """
        Waehlt Daten aus dem MNIST-Datensatz aus.
        """
        x_train, y_train = load_mnist("training_data")  
        x_test, y_test = load_mnist("testing_data")  
        idx_train = np.append(
            np.where(y_train == 0)[0][:train_num],
            np.where(y_train == 1)[0][:train_num])

        x_train = x_train[idx_train]
        y_train = y_train[idx_train]

        x_train = x_train / 255
        y_train = np.eye(2)[y_train].reshape(-1, 2)

        # Test: Behalten nur Labels 0 und 1
        idx_test = np.append(
            np.where(y_test == 0)[0][:test_num],
            np.where(y_test == 1)[0][:test_num])

        x_test = x_test[idx_test]
        y_test = y_test[idx_test]
        x_test = x_test / 255
        y_test = np.eye(2)[y_test].reshape(-1, 2)

        return x_train, y_train, x_test, y_test

    def RotCircuit(para, qlist):

        if isinstance(para, QTensor):
            para = QTensor._to_numpy(para)
        if para.ndim > 1:
            raise ValueError(" dim of paramters in Rot should be 1")
        if para.shape[0] != 3:
            raise ValueError(" numbers of paramters in Rot should be 3")

        cir = pq.QCircuit()
        cir.insert(pq.RZ(qlist, para[2]))
        cir.insert(pq.RY(qlist, para[1]))
        cir.insert(pq.RZ(qlist, para[0]))

        return cir

    def build_RotCircuit(qubits, weights):
        cir = pq.QCircuit()
        cir.insert(RotCircuit(weights[0:3], qubits[0]))
        cir.insert(RotCircuit(weights[3:6], qubits[1]))
        cir.insert(RotCircuit(weights[6:9], qubits[2]))
        cir.insert(RotCircuit(weights[9:12], qubits[3]))
        cir.insert(RotCircuit(weights[12:15], qubits[4]))
        cir.insert(RotCircuit(weights[15:18], qubits[5]))
        cir.insert(RotCircuit(weights[18:21], qubits[6]))
        cir.insert(RotCircuit(weights[21:24], qubits[7]))
        cir.insert(RotCircuit(weights[24:27], qubits[8]))
        cir.insert(RotCircuit(weights[27:30], qubits[9]))
        cir.insert(RotCircuit(weights[30:33], qubits[10]))
        cir.insert(RotCircuit(weights[33:36], qubits[11]))
        cir.insert(RotCircuit(weights[36:39], qubits[12]))
        cir.insert(RotCircuit(weights[39:42], qubits[13]))
        cir.insert(RotCircuit(weights[42:45], qubits[14]))
        cir.insert(RotCircuit(weights[45:48], qubits[15]))

        return cir

    def CRXCircuit(para, control_qlists, rot_qlists):
        cir = pq.QCircuit()
        cir.insert(pq.RX(rot_qlists, para))
        cir.set_control(control_qlists)
        return cir

    def build_CRotCircuit(qubits, weights):
        cir = pq.QCircuit()
        cir.insert(CRXCircuit(weights[0], qubits[0], qubits[1]))
        cir.insert(CRXCircuit(weights[1], qubits[1], qubits[2]))
        cir.insert(CRXCircuit(weights[2], qubits[2], qubits[3]))
        cir.insert(CRXCircuit(weights[3], qubits[3], qubits[4]))
        cir.insert(CRXCircuit(weights[4], qubits[4], qubits[5]))
        cir.insert(CRXCircuit(weights[5], qubits[5], qubits[6]))
        cir.insert(CRXCircuit(weights[6], qubits[6], qubits[7]))
        cir.insert(CRXCircuit(weights[7], qubits[7], qubits[8]))
        cir.insert(CRXCircuit(weights[8], qubits[8], qubits[9]))
        cir.insert(CRXCircuit(weights[9], qubits[9], qubits[10]))
        cir.insert(CRXCircuit(weights[10], qubits[10], qubits[11]))
        cir.insert(CRXCircuit(weights[11], qubits[11], qubits[12]))
        cir.insert(CRXCircuit(weights[12], qubits[12], qubits[13]))
        cir.insert(CRXCircuit(weights[13], qubits[13], qubits[14]))
        cir.insert(CRXCircuit(weights[14], qubits[14], qubits[15]))
        cir.insert(CRXCircuit(weights[15], qubits[15], qubits[0]))

        return cir


    def build_qmlp_circuit(x, weights, qubits, clist, machine):
        cir = pq.QCircuit()
        num_qubits = len(qubits)
        for i in range(num_qubits):
            cir.insert(pq.RX(qubits[i], x[i]))

        cir.insert(build_RotCircuit(qubits, weights[0:48]))
        cir.insert(build_CRotCircuit(qubits, weights[48:64]))

        for i in range(num_qubits):
            cir.insert(pq.RX(qubits[i], x[i]))

        cir.insert(build_RotCircuit(qubits, weights[64:112]))
        cir.insert(build_CRotCircuit(qubits, weights[112:128]))

        prog = pq.QProg()
        prog.insert(cir)

        exp_vals = []
        for position in range(num_qubits):
            pauli_str = {"Z" + str(position): 1.0}
            exp2 = expval(machine, prog, pauli_str, qubits)
            exp_vals.append(exp2)

        return exp_vals

    

    class QMLPModel(Module):
        def __init__(self):
            super(QMLPModel, self).__init__()
            self.ave_pool2d = AvgPool2D([7, 7], [7, 7], "valid")
            self.quantum_circuit = QuantumLayer(build_qmlp_circuit, 128, "CPU", 16, diff_method="finite_diff")
            
            self.linear = Linear(16, 10)

        def forward(self, x):
            bsz = x.shape[0]
            x = self.ave_pool2d(x)
            input_data = x.reshape([bsz, 16])
            quanutum_result = self.quantum_circuit(input_data)
            result = self.linear(quanutum_result)
            return result

    def vqnet_test_QMLPModel():
        # train num=1000, test_num=100

        train_size = 1000
        eval_size = 100
        x_train, y_train = load_mnist("training_data", digits=np.arange(10))
        x_test, y_test = load_mnist("testing_data", digits=np.arange(10))

        x_train = x_train[:train_size]
        y_train = y_train[:train_size]
        x_test = x_test[:eval_size]
        y_test = y_test[:eval_size]

        x_train = x_train / 255
        x_test = x_test / 255
        y_train = np.eye(10)[y_train].reshape(-1, 10)
        y_test = np.eye(10)[y_test].reshape(-1, 10)

        model = QMLPModel()
        optimizer = Adam(model.parameters(), lr=0.005)
        loss_func = CrossEntropyLoss()
        loss_list = []
        epochs = 30
        for epoch in range(1, epochs):
            total_loss = []

            correct = 0
            n_train = 0
            for x, y in data_generator(x_train,
                                       y_train,
                                       batch_size=16,
                                       shuffle=True):

                x = x.reshape(-1, 1, 28, 28)
                optimizer.zero_grad()
                # Forward pass
                output = model(x)
                # Calculating loss
                loss = loss_func(y, output)
                loss_np = np.array(loss.data)
                print("loss: ", loss_np)
                np_output = np.array(output.data, copy=False)

                temp_out = np_output.argmax(axis=1)
                temp_output = np.zeros((temp_out.size, 10))
                temp_output[np.arange(temp_out.size), temp_out] = 1
                temp_maks = (temp_output == y)

                correct += np.sum(np.array(temp_maks))
                n_train += 160

                # Backward pass
                loss.backward()
                # Optimize the weights
                optimizer._step()
                total_loss.append(loss_np)
            print("##########################")
            print(f"Train Accuracy: {correct / n_train}")
            loss_list.append(np.sum(total_loss) / len(total_loss))
            print("epoch: ", epoch)
            print("{:.0f} loss is : {:.10f}".format(epoch, loss_list[-1]))

    if __name__ == "__main__":

        vqnet_test_QMLPModel()



data result
--------------------

Die Verlustfunktionskurve der Trainingsdaten wird angezeigt und gespeichert.
Verlust auf dem Trainingsdatensatz.

.. image:: ./images/QMLP.png
   :width: 600 px
   :align: center

|


.. _QDRL_DEMO:

5. Hybrides quantenklassisches QDRL-Netzwerkmodell
===============================================================================

Wir stellen ein vorgeschlagenes Quanten-Reinforcement-Learning-Netzwerk (QDRL) vor und analysieren es, dessen Merkmale klassische Deep-Reinforcement-Learning-Algorithmen wie Experience Replay und Target Networks in Darstellungen von Variationsquantenschaltungen umformen.
Darueber hinaus verwenden wir ein Quanteninformationskodierungsschema, um die Anzahl der Modellparameter im Vergleich zu klassischen neuronalen Netzen zu reduzieren. `QDRL: Variational Quantum Circuits for Deep Reinforcement Learning <https://arxiv.org/pdf/1907.00397.pdf>`_.
Wir werden ein einfaches Beispiel zur Integration von pyQPanda2 mit VQNet schreiben.



Aufbau hybrider quantenklassischer neuronaler Netze
----------------------------------------------------------

Erfordert ``gym`` == 0.23.0 , ``pygame`` == 2.1.2 .

.. code-block::

    import numpy as np
    import random
    import gym
    import time
    from matplotlib import animation
    import pyqpanda as pq
    from pyvqnet.nn.module import Module
    from pyvqnet.nn.loss import MeanSquaredError
    from pyvqnet.optim.adam import Adam
    from pyvqnet.tensor.tensor import QTensor
    from pyvqnet import kfloat32
    from pyvqnet.qnn.quantumlayer import QuantumLayer
    from pyvqnet.tensor import tensor
    from pyvqnet.qnn.measure import expval
    from pyvqnet._core import Tensor as CoreTensor
    import matplotlib
    from matplotlib import pyplot as plt
    try:
        matplotlib.use("TkAgg")
    except:
        print("Can not use matplot TkAgg")
    def display_frames_as_gif(frames, c_index):
        patch = plt.imshow(frames[0])
        plt.axis('off')
        def animate(i):
            patch.set_data(frames[i])
        anim = animation.FuncAnimation(plt.gcf(), animate, frames=len(frames), interval=5)
        name_result = "./result_"+str(c_index)+".gif"
        anim.save(name_result, writer='pillow', fps=10)
    CIRCUIT_SIZE = 4
    MAX_ITERATIONS = 50
    MAX_STEPS = 250
    BATCHSIZE = 5
    TARGET_MAX = 20
    GAMMA = 0.99
    STATE_T = 0
    ACTION = 1
    REWARD = 2
    STATE_NT = 3
    DONE = 4
    def RotCircuit(para, qlist):
        
        if isinstance(para, QTensor):
            para = QTensor._to_numpy(para)
        if para.ndim > 1:
            raise ValueError(" dim of paramters in Rot should be 1")
        if para.shape[0] != 3:
            raise ValueError(" numbers of paramters in Rot should be 3")
        cir = pq.QCircuit()
        cir.insert(pq.RZ(qlist, para[2]))
        cir.insert(pq.RY(qlist, para[1]))
        cir.insert(pq.RZ(qlist, para[0]))
        return cir
    def layer_circuit(qubits, weights):
        cir = pq.QCircuit()
        #Verschraenkungsblock
        cir.insert(pq.CNOT(qubits[0], qubits[1]))
        cir.insert(pq.CNOT(qubits[1], qubits[2]))
        cir.insert(pq.CNOT(qubits[2], qubits[3]))
        #u3-Gatter
        cir.insert(RotCircuit(weights[0], qubits[0]))
        cir.insert(RotCircuit(weights[1], qubits[1]))
        cir.insert(RotCircuit(weights[2], qubits[2]))
        cir.insert(RotCircuit(weights[3], qubits[3]))
        return cir
    def encoder(encodings):
        encodings = int(encodings[0])
        return [i for i, b in enumerate(f'{encodings:0{CIRCUIT_SIZE}b}') if b == '1']

    def build_qc(x, weights, qubits, cbits ,machine):


        cir = pq.QCircuit()
        if x:
            wires = encoder(x)
            for wire in wires:
                cir.insert(pq.RX(qubits[wire], np.pi))
                cir.insert(pq.RZ(qubits[wire], np.pi))
        #Parameteranzahl = 24
        weights = weights.reshape([2, 4, 3])
        #schichtweise
        for w in weights:
            cir.insert(layer_circuit(qubits, w))
        prog = pq.QProg()
        prog.insert(cir)
        exp_vals = []
        for position in range(n_qubits):
            pauli_str = {"Z" + str(position): 1.0}
            exp2 = expval(machine, prog, pauli_str, qubits)
            exp_vals.append(exp2)
        return exp_vals
    class DRLModel(Module):
        def __init__(self):
            super(DRLModel, self).__init__()
            self.quantum_circuit = QuantumLayer(build_qc, 24, "CPU", 4, diff_method="finite_diff")

        def forward(self, x):
            quanutum_result = self.quantum_circuit(x)
            return quanutum_result
    env = gym.make("FrozenLake-v1", is_slippery = False, map_name = '4x4')
    state = env.reset()
    n_layers = 2
    n_qubits = 4
    targ_counter = 0
    sampled_vs = []
    memory = {}
    param = QTensor(0.01 * np.random.randn(n_layers, n_qubits, 3))
    bias = QTensor([[0.0, 0.0, 0.0, 0.0]])
    param_targ = param.copy().reshape([1, -1]).pdata[0]
    bias_targ = bias.copy()
    loss_func = MeanSquaredError()
    model = DRLModel()
    opt = Adam(model.parameters(), lr=5)
    for i in range(MAX_ITERATIONS):
        start = time.time()
        state_t = env.reset()
        a_init = env.action_space.sample()
        total_reward = 0
        done = False
        frames = []
        for t in range(MAX_STEPS):
            frames.append(env.render(mode='rgb_array'))
            time.sleep(0.1)
            input_x = QTensor([[state_t]],dtype=kfloat32)
            acts = model(input_x) + bias

            act_t = tensor.QTensor.argmax(acts)

            act_t_np = int(act_t.pdata[0])
            print(f'Episode: {i}, Steps: {t}, act: {act_t_np}')
            state_nt, reward, done, info = env.step(action=act_t_np)
            targ_counter += 1
            input_state_nt = QTensor([[state_nt]],dtype=kfloat32)
            act_nt = QTensor.argmax(model(input_state_nt)+bias)
            act_nt_np = int(act_nt.pdata[0])
            memory[i, t] = (state_t, act_t, reward, state_nt, done)
            if len(memory) >= BATCHSIZE:
                sampled_vs = [memory[k] for k in random.sample(list(memory), BATCHSIZE)]
                target_temp = []
                for s in sampled_vs:
                    if s[DONE]:
                        target_temp.append(QTensor(s[REWARD]).reshape([1, -1]))
                    else:
                        input_s = QTensor([[s[STATE_NT]]],dtype=kfloat32)
                        out_temp = s[REWARD] + GAMMA * tensor.max(model(input_s) + bias_targ)
                        out_temp = out_temp.reshape([1, -1])
                        target_temp.append(out_temp)
                target_out = []
                for b in sampled_vs:
                    input_b = QTensor([[b[STATE_T]]], requires_grad=True,dtype=kfloat32)
                    out_result = model(input_b) + bias
                    index = int(b[ACTION].pdata[0])
                    out_result_temp = out_result[0][index].reshape([1, -1])
                    target_out.append(out_result_temp)
                opt.zero_grad()
                target_label = tensor.concatenate(target_temp, 1)
                output = tensor.concatenate(target_out, 1)
                loss = loss_func(target_label, output)
                loss.backward()
                opt.step()
            #Parameter in der Zielschaltung aktualisieren
            if targ_counter == TARGET_MAX:
                param_targ = param.copy().reshape([1, -1]).pdata[0]
                bias_targ = bias.copy()
                targ_counter = 0
            state_t, act_t_np = state_nt, act_nt_np
            if done:
                print("reward", reward)
                if reward == 1.0:
                    frames.append(env.render(mode='rgb_array'))
                    display_frames_as_gif(frames, i)
                    exit()
                break
        end = time.time()


Unsupervised learning
****************************

1 Quantum Kmeans
=======================================

1.1 Einfuehrung
-----------------------

Der Clustering-Algorithmus ist ein typischer unueberwachter Lernalgorithmus, der hauptsaechlich dazu dient, aehnliche Proben automatisch in eine Klasse einzuordnen. Im Clustering-Algorithmus werden Proben nach der Aehnlichkeit zwischen ihnen in verschiedene Kategorien eingeteilt. Fuer unterschiedliche Aehnlichkeitsberechnungsmethoden werden unterschiedliche Clusterergebnisse erzielt. Die gebraeuchlichste Aehnlichkeitsberechnungsmethode ist die euklidische Distanz. Was wir zeigen moechten, ist der Quantum-k-Means-Algorithmus. Der k-Means-Algorithmus ist ein distanzbasierter Clustering-Algorithmus. Er verwendet die Distanz als Bewertungsindex der Aehnlichkeit, d.h. je kleiner die Distanz zwischen zwei Objekten, desto groesser die Aehnlichkeit. Der Algorithmus geht davon aus, dass Cluster aus nahe beieinander liegenden Objekten bestehen, sodass kompakte und unabhaengige Cluster das ultimative Ziel sind.

Quantum-k-Means-Quantenmaschinelles-Lernmodelle koennen auch in VQNet entwickelt werden. Im Folgenden wird ein Beispiel fuer die Quantum-k-Means-Clusteringaufgabe gegeben. Durch die Quantenschaltung koennen wir eine Messung konstruieren, die positiv mit der euklidischen Distanz der Variablen des klassischen maschinellen Lernens korreliert, um so das Ziel der Suche nach dem naechsten Nachbarn zu erreichen.


1.2 Einfuehrung in das Algorithmusprinzip
------------------------------------------------

Die Implementierung des Quantum-k-Means-Algorithmus verwendet hauptsaechlich den Swap-Test, um die Distanz zwischen Eingabedatenpunkten zu vergleichen. Waehlen Sie zufaellig k Punkte aus N Datenpunkten als Zentroide aus, messen Sie die Distanz von jedem Punkt zu jedem Zentroid, weisen Sie ihn der naechsten Zentroidklasse zu, berechnen Sie den Zentroid jeder Klasse neu und iterieren Sie die Schritte 2 bis 3, bis der neue Zentroid gleich oder kleiner als der angegebene Schwellenwert ist. In unserem Beispiel waehlen wir 100 Datenpunkte und 2 Zentroide aus und verwenden die cswap-Schaltung zur Berechnung der Distanz.
Schliesslich erhalten wir zwei Datenpunktcluster. :math:`|0\rangle` ist ein Hilfsbit, durch das H-Logikgatter wird das Qubit zu :math:`\frac{1}{\sqrt{2}}(|0\rangle + |1\rangle)`. Unter der Kontrolle des :math:`|1\rangle`-Qubits
wird die Quantenschaltung :math:`|x\rangle` und :math:`|y\rangle` vertauschen. Schliesslich erhalten wir das Ergebnis:

.. math::

    |0_{anc}\rangle |x\rangle |y\rangle \rightarrow \frac{1}{2}|0_{anc}\rangle(|xy\rangle + |yx\rangle) + \frac{1}{2}|1_{anc}\rangle(|xy\rangle - |yx\rangle)

Wenn wir das Hilfsqubit separat messen, ist die Wahrscheinlichkeit des Endzustands des Grundzustands :math:`|1\rangle`:

.. math::

    P(|1_{anc}\rangle) = \frac{1}{2} - \frac{1}{2}|\langle x | y \rangle|^2

Die euklidische Distanz zwischen zwei Quantenzustaenden ist wie folgt:

.. math::

    Euclidean \ distance = \sqrt{(2 - 2|\langle x | y \rangle|)}

Die messbare Wahrscheinlichkeit des :math:`|1\rangle`-Qubits ist positiv mit der euklidischen Distanz korreliert. Die Quantenschaltung dieses Algorithmus ist wie folgt:

.. figure:: ./images/Kmeans.jpg
   :width: 600 px
   :align: center

|

1.3 VQNet-Implementierung
---------------------------------

1.3.1 Umgebungsvorbereitung
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Die Umgebung verwendet Python 3.8. Es wird empfohlen, CONDA fuer die Umgebungskonfiguration zu verwenden. Es enthaelt numpy, SciPy, Matplotlib, sklearn und andere Toolkits zur einfachen Verwendung. Bei Verwendung der Python-Umgebung muessen relevante Pakete installiert werden, und die folgende Umgebung pyvqnet muss vorbereitet werden.


1.3.2 Datenvorbereitung
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Die Daten werden zufaellig von make_blobs unter SciPy generiert, und die Funktion wird definiert, um Gaußsche Verteilungsdaten zu generieren.


.. code-block::

    import sys
    sys.path.insert(0, "../")
    import math
    import numpy as np
    from pyvqnet.tensor.tensor import QTensor, zeros
    import pyvqnet.tensor.tensor as tensor
    import pyqpanda as pq
    from sklearn.datasets import make_blobs
    import matplotlib.pyplot as plt
    import matplotlib
    try:
        matplotlib.use("TkAgg")
    except:  
        print("Can not use matplot TkAgg")
        pass

    def get_data(n, k, std):
        data = make_blobs(n_samples=n,
                        n_features=2,
                        centers=k,
                        cluster_std=std,
                        random_state=100)
        points = data[0]
        centers = data[1]
        return points, centers

1.3.3 Quantenschaltung
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Aufbau von Quantenschaltungen mit VQNet

.. code-block::

    # Der Eingabe-Quantengatter-Drehwinkel wird basierend auf dem Eingabekoordinatenpunkt d (x, y) berechnet
    def get_theta(d):
        x = d[0]
        y = d[1]
        theta = 2 * math.acos((x.item() + y.item()) / 2.0)
        return theta

    # Die Quantenschaltung wird basierend auf den Eingabe-Quantendatenpunkten konstruiert
    def qkmeans_circuits(x, y):

        theta_1 = get_theta(x)
        theta_2 = get_theta(y)

        num_qubits = 3
        machine = pq.CPUQVM()
        machine.init_qvm()
        qubits = machine.qAlloc_many(num_qubits)
        cbits = machine.cAlloc_many(num_qubits)
        circuit = pq.QCircuit()

        circuit.insert(pq.H(qubits[0]))
        circuit.insert(pq.H(qubits[1]))
        circuit.insert(pq.H(qubits[2]))

        circuit.insert(pq.U3(qubits[1], theta_1, np.pi, np.pi))
        circuit.insert(pq.U3(qubits[2], theta_2, np.pi, np.pi))

        circuit.insert(pq.SWAP(qubits[1], qubits[2]).control([qubits[0]]))

        circuit.insert(pq.H(qubits[0]))

        prog = pq.QProg()
        prog.insert(circuit)
        prog << pq.Measure(qubits[0], cbits[0])
        prog.insert(pq.Reset(qubits[0]))
        prog.insert(pq.Reset(qubits[1]))
        prog.insert(pq.Reset(qubits[2]))

        result = machine.run_with_configuration(prog, cbits, 1024)

        data = result

        if len(data) == 1:
            return 0.0
        else:
            return data['001'] / 1024.0

1.3.4 Datenvisualisierung
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Visuelle Berechnung relevanter Clustering-Daten

.. code-block::

    # Visualisierung von Streupunkten und Clusterzentren
    def draw_plot(points, centers, label=True):
        points = np.array(points)
        centers = np.array(centers)
        if label==False:
            plt.scatter(points[:,0], points[:,1])
        else:
            plt.scatter(points[:,0], points[:,1], c=centers, cmap='viridis')
        plt.xlim(0, 1)
        plt.ylim(0, 1)
        plt.show()

1.3.5 Clusterberechnung
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Berechnung des Clusterzentrums relevanter Clusterdaten

.. code-block::

    # Zufaellige Generierung von Clusterzentrumspunkten
    def initialize_centers(points,k):
        return points[np.random.randint(points.shape[0],size=k),:]


    def find_nearest_neighbour(points, centroids):
        n = points.shape[0]
        k = centroids.shape[0]

        centers = zeros([n], dtype=points.dtype)

        for i in range(n):
            min_dis = 10000
            ind = 0
            for j in range(k):

                temp_dis = qkmeans_circuits(points[i, :], centroids[j, :])

                if temp_dis < min_dis:
                    min_dis = temp_dis
                    ind = j
            centers[i] = ind

        return centers

    def find_centroids(points, centers):

        k = int(tensor.max(centers).item()) + 1
        centroids = tensor.zeros([k, 2], dtype=points.dtype)
        for i in range(k):

            cur_i = centers == i

            x = points[:,0]
            x = x[cur_i]
            y = points[:,1]
            y = y[cur_i]
            centroids[i, 0] = tensor.mean(x)
            centroids[i, 1] = tensor.mean(y)

        return centroids

    def preprocess(points):
        n = len(points)
        x = 30.0 * np.sqrt(2)
        for i in range(n):
            points[i, :] += 15
            points[i, :] /= x

        return points


    def qkmean_run():
        n = 100  # Anzahl der Datenpunkte
        k = 3  # Anzahl der Zentren
        std = 2  # Std der Datenpunkte

        points, o_centers = get_data(n, k, std)  # Datensatz

        points = preprocess(points)  # Datensatz normalisieren

        centroids = initialize_centers(points, k)  # Zentroide initialisieren

        epoch = 9
        points = QTensor(points)
        centroids = QTensor(centroids)
        plt.figure()
        draw_plot(points.data, o_centers,label=False)

        for i in range(epoch):
                centers = find_nearest_neighbour(points, centroids)  # naechste Zentren finden
                centroids = find_centroids(points, centers)  # Zentroide finden

        plt.figure()
        draw_plot(points.data, centers.data)

    # Programmeintritt
    if __name__ == "__main__":
        qkmean_run()


1.3.6 Datenverteilung vor dem Clustering
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. figure:: ./images/ep_1.png
   :width: 600 px
   :align: center

|

1.3.7 Datenverteilung nach dem Clustering
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. figure:: ./images/ep_9.png
   :width: 600 px
   :align: center

|

Quantum Machine Learning Research
*********************************************

Quantenmodelle als Fourier-Reihen
================================================================================

Quantencomputer koennen fuer ueberwachtes Lernen verwendet werden, indem parametrisierte Quantenschaltungen als Modelle behandelt werden, die Dateneingaenge
auf Vorhersagen abbilden. Waehrend viel Arbeit geleistet wurde, um die praktischen Auswirkungen dieses Ansatzes zu untersuchen, bleiben viele wichtige
theoretische Eigenschaften dieser Modelle unbekannt. Hier untersuchen wir, wie die Strategie, mit der Daten in das Modell kodiert werden,
die Ausdruckskraft parametrisierter Quantenschaltungen als Funktionenapproximatoren beeinflusst.


Die Arbeit `The effect of data encoding on the expressive power of variational quantum machine learning models <https://arxiv.org/pdf/2008.08605.pdf>`_ verknuepft allgemeine Quantenmaschinelles-Lernen-Modelle, die fuer kurzfristige Quantencomputer entwickelt wurden, mit Fourier-Reihen


1.1 Anpassung von Fourier-Reihen mit serieller Pauli-Rotationskodierung
-----------------------------------------------------------------------

Zunaechst zeigen wir, wie Quantenmodelle, die Pauli-Rotationen als Datenkodierungsgatter verwenden, nur Fourier-Reihen bis zu einem bestimmten Grad anpassen koennen. Der Einfachheit halber betrachten wir nur Ein-Qubit-Schaltungen:

.. image:: ./images/single_qubit_model.png
   :width: 600 px
   :align: center

|

Eingabedaten erstellen, parallele Quantenmodelle definieren und keine Modelltrainingsergebnisse durchfuehren.

.. code-block::

    """
    Quantum Fourier Series
    """
    import numpy as np
    import pyqpanda as pq
    from pyvqnet.qnn.measure import expval
    import matplotlib.pyplot as plt
    import matplotlib
    try:
        matplotlib.use("TkAgg")
    except:
        print("Can not use matplot TkAgg")
        pass

    np.random.seed(42)


    degree = 1  # Grad der Zielfunktion
    scaling = 1  # Skalierung der Daten
    coeffs = [0.15 + 0.15j]*degree  # Koeffizienten der Nicht-Null-Frequenzen
    coeff0 = 0.1  # Koeffizient der Null-Frequenz

    def target_function(x):
        """Generiert eine abgeschnittene Fourier-Reihe, wobei die Daten neu skaliert werden."""
        res = coeff0
        for idx, coeff in enumerate(coeffs):
            exponent = np.complex128(scaling * (idx+1) * x * 1j)
            conj_coeff = np.conjugate(coeff)
            res += coeff * np.exp(exponent) + conj_coeff * np.exp(-exponent)
        return np.real(res)

    x = np.linspace(-6, 6, 70)
    target_y = np.array([target_function(x_) for x_ in x])

    plt.plot(x, target_y, c='black')
    plt.scatter(x, target_y, facecolor='white', edgecolor='black')
    plt.ylim(-1, 1)
    plt.show()


    def S(scaling, x, qubits):
        cir = pq.QCircuit()
        cir.insert(pq.RX(qubits[0], scaling * x))
        return cir

    def W(theta, qubits):
        cir = pq.QCircuit()
        cir.insert(pq.RZ(qubits[0], theta[0]))
        cir.insert(pq.RY(qubits[0], theta[1]))
        cir.insert(pq.RZ(qubits[0], theta[2]))
        return cir

    def serial_quantum_model(weights, x, num_qubits, scaling):
        cir = pq.QCircuit()
        machine = pq.CPUQVM()
        machine.init_qvm()
        qubits = machine.qAlloc_many(num_qubits)

        for theta in weights[:-1]:
            cir.insert(W(theta, qubits))
            cir.insert(S(scaling, x, qubits))

        cir.insert(W(weights[-1], qubits))
        prog = pq.QProg()
        prog.insert(cir)

        exp_vals = []
        for position in range(num_qubits):
            pauli_str = {"Z" + str(position): 1.0}
            exp2 = expval(machine, prog, pauli_str, qubits)
            exp_vals.append(exp2)

        return exp_vals

    r = 1
    weights = 2 * np.pi * np.random.random(size=(r+1, 3))

    x = np.linspace(-6, 6, 70)
    random_quantum_model_y = [serial_quantum_model(weights, x_, 1, 1) for x_ in x]

    plt.plot(x, target_y, c='black', label="true")
    plt.scatter(x, target_y, facecolor='white', edgecolor='black')
    plt.plot(x, random_quantum_model_y, c='blue', label="predict")
    plt.ylim(-1, 1)
    plt.legend(loc="upper right")
    plt.show()

Das Ergebnis der Ausfuehrung der Quantenschaltung ohne Training ist:

.. image:: ./images/single_qubit_model_result_no_train.png
   :width: 600 px
   :align: center

|

Eingabedaten erstellen, das serielle Quantenmodell definieren und das Trainingsmodell in Kombination mit der QuantumLayer des VQNet-Frameworks erstellen.

.. code-block::

    """
    Quantum Fourier Series Serial
    """
    import numpy as np
    from pyvqnet.nn.module import Module
    from pyvqnet.nn.loss import MeanSquaredError
    from pyvqnet.optim.adam import Adam
    from pyvqnet.tensor.tensor import QTensor
    import pyqpanda as pq
    from pyvqnet.qnn.measure import expval
    from pyvqnet.qnn.quantumlayer import QuantumLayer
    import matplotlib.pyplot as plt
    import matplotlib
    try:
        matplotlib.use("TkAgg")
    except:
        print("Can not use matplot TkAgg")
        pass

    np.random.seed(42)

    degree = 1
    scaling = 1
    coeffs = [0.15 + 0.15j]*degree
    coeff0 = 0.1

    def target_function(x):
        """Generiert eine abgeschnittene Fourier-Reihe."""
        res = coeff0
        for idx, coeff in enumerate(coeffs):
            exponent = np.complex128(scaling * (idx+1) * x * 1j)
            conj_coeff = np.conjugate(coeff)
            res += coeff * np.exp(exponent) + conj_coeff * np.exp(-exponent)
        return np.real(res)

    x = np.linspace(-6, 6, 70)
    target_y = np.array([target_function(xx) for xx in x])

    plt.plot(x, target_y, c='black')
    plt.scatter(x, target_y, facecolor='white', edgecolor='black')
    plt.ylim(-1, 1)
    plt.show()


    def S(x, qubits):
        cir = pq.QCircuit()
        cir.insert(pq.RX(qubits[0], x))
        return cir

    def W(theta, qubits):
        cir = pq.QCircuit()
        cir.insert(pq.RZ(qubits[0], theta[0]))
        cir.insert(pq.RY(qubits[0], theta[1]))
        cir.insert(pq.RZ(qubits[0], theta[2]))
        return cir


    r = 1
    weights = 2 * np.pi * np.random.random(size=(r+1, 3))

    x = np.linspace(-6, 6, 70)


    def q_circuits_loop(x, weights, qubits, clist, machine):

        result = []
        for xx in x:
            cir = pq.QCircuit()
            weights = weights.reshape([2, 3])

            for theta in weights[:-1]:
                cir.insert(W(theta, qubits))
                cir.insert(S(xx, qubits))

            cir.insert(W(weights[-1], qubits))
            prog = pq.QProg()
            prog.insert(cir)

            exp_vals = []
            for position in range(1):
                pauli_str = {"Z" + str(position): 1.0}
                exp2 = expval(machine, prog, pauli_str, qubits)
                exp_vals.append(exp2)
                result.append(exp2)
        return result

    class Model(Module):
        def __init__(self):
            super(Model, self).__init__()
            self.q_fourier_series = QuantumLayer(q_circuits_loop, 6, "CPU", 1)

        def forward(self, x):
            return self.q_fourier_series(x)

    def run():
        model = Model()

        optimizer = Adam(model.parameters(), lr=0.5)
        batch_size = 2
        epoch = 5
        loss = MeanSquaredError()
        print("start training..............")
        model.train()
        max_steps = 50
        for i in range(epoch):
            sum_loss = 0
            count = 0
            for step in range(max_steps):
                optimizer.zero_grad()
                batch_index = np.random.randint(0, len(x), (batch_size,))
                x_batch = x[batch_index]
                y_batch = target_y[batch_index]
                data, label = QTensor([x_batch]), QTensor([y_batch])
                result = model(data)
                loss_b = loss(label, result)
                loss_b.backward()
                optimizer._step()
                sum_loss += loss_b.item()
                count += batch_size
            print(f"epoch:{i}, #### loss:{sum_loss/count} ")

            model.eval()
            predictions = []
            for xx in x:
                data = QTensor([[xx]])
                result = model(data)
                predictions.append(result.pdata[0])

            plt.plot(x, target_y, c='black', label="true")
            plt.scatter(x, target_y, facecolor='white', edgecolor='black')
            plt.plot(x, predictions, c='blue', label="predict")
            plt.ylim(-1, 1)
            plt.legend(loc="upper right")
            plt.show()

    if __name__ == "__main__":
        run()

Das Quantenmodell ist:

.. image:: ./images/single_qubit_model_circuit.png
   :width: 600 px
   :align: center

|

Die Netzwerktrainingsergebnisse sind:

.. image:: ./images/single_qubit_model_result.png
   :width: 600 px
   :align: center

|

Der Netzwerktrainingsverlust ist:

.. code-block::

    start training..............
    epoch:0, #### loss:0.04852807720773853
    epoch:1, #### loss:0.012945819365559146
    epoch:2, #### loss:0.0009359727291666786
    epoch:3, #### loss:0.00015995280153333625
    epoch:4, #### loss:3.988249877352246e-05


1.2 Anpassung von Fourier-Reihen mit paralleler Pauli-Rotationskodierung
------------------------------------------------------------------------

Wie in der Arbeit gezeigt, erwarten wir aehnliche Ergebnisse wie beim seriellen Modell: Eine Fourier-Reihe der Ordnung r kann nur angepasst werden, wenn das kodierte Gatter mindestens r Wiederholungen im Quantenmodell hat. Quantenschaltung:

.. image:: ./images/parallel_model.png
   :width: 600 px
   :align: center

|

Eingabedaten erstellen, parallele Quantenmodelle definieren und keine Modelltrainingsergebnisse durchfuehren.

.. code-block::

    """
    Quantum Fourier Series
    """
    import numpy as np
    import pyqpanda as pq
    from pyvqnet.qnn.measure import expval
    import matplotlib.pyplot as plt
    import matplotlib
    try:
        matplotlib.use("TkAgg")
    except:
        print("Can not use matplot TkAgg")
        pass

    np.random.seed(42)

    degree = 1
    scaling = 1
    coeffs = [0.15 + 0.15j] * degree
    coeff0 = 0.1

    def target_function(x):
        """Generiert eine abgeschnittene Fourier-Reihe."""
        res = coeff0
        for idx, coeff in enumerate(coeffs):
            exponent = np.complex128(scaling * (idx + 1) * x * 1j)
            conj_coeff = np.conjugate(coeff)
            res += coeff * np.exp(exponent) + conj_coeff * np.exp(-exponent)
        return np.real(res)

    x = np.linspace(-6, 6, 70)
    target_y = np.array([target_function(xx) for xx in x])


    def S1(x, qubits):
        cir = pq.QCircuit()
        for q in qubits:
            cir.insert(pq.RX(q, x))
        return cir

    def W1(theta, qubits):
        cir = pq.QCircuit()
        for i in range(len(qubits)):
            cir.insert(pq.RZ(qubits[i], theta[0][i][0]))
            cir.insert(pq.RY(qubits[i], theta[0][i][1]))
            cir.insert(pq.RZ(qubits[i], theta[0][i][2]))

        for i in range(len(qubits) - 1):
            cir.insert(pq.CNOT(qubits[i], qubits[i + 1]))
        cir.insert(pq.CNOT(qubits[len(qubits) - 1], qubits[0]))

        for i in range(len(qubits)):
            cir.insert(pq.RZ(qubits[i], theta[1][i][0]))
            cir.insert(pq.RY(qubits[i], theta[1][i][1]))
            cir.insert(pq.RZ(qubits[i], theta[1][i][2]))

        cir.insert(pq.CNOT(qubits[0], qubits[len(qubits) - 1]))
        for i in range(len(qubits) - 1):
            cir.insert(pq.CNOT(qubits[i + 1], qubits[i]))

        for i in range(len(qubits)):
            cir.insert(pq.RZ(qubits[i], theta[2][i][0]))
            cir.insert(pq.RY(qubits[i], theta[2][i][1]))
            cir.insert(pq.RZ(qubits[i], theta[2][i][2]))

        for i in range(len(qubits) - 1):
            cir.insert(pq.CNOT(qubits[i], qubits[i + 1]))
        cir.insert(pq.CNOT(qubits[len(qubits) - 1], qubits[0]))

        return cir

    def parallel_quantum_model(weights, x, num_qubits):
        cir = pq.QCircuit()
        machine = pq.CPUQVM()
        machine.init_qvm()
        qubits = machine.qAlloc_many(num_qubits)

        cir.insert(W1(weights[0], qubits))
        cir.insert(S1(x, qubits))

        cir.insert(W1(weights[1], qubits))
        prog = pq.QProg()
        prog.insert(cir)

        exp_vals = []
        for position in range(1):
            pauli_str = {"Z" + str(position): 1.0}
            exp2 = expval(machine, prog, pauli_str, qubits)
            exp_vals.append(exp2)

        return exp_vals

    r = 3

    trainable_block_layers = 3
    weights = 2 * np.pi * np.random.random(size=(2, trainable_block_layers, r, 3))
    x = np.linspace(-6, 6, 70)
    random_quantum_model_y = [parallel_quantum_model(weights, xx, r) for xx in x]

    plt.plot(x, target_y, c='black', label="true")
    plt.scatter(x, target_y, facecolor='white', edgecolor='black')
    plt.plot(x, random_quantum_model_y, c='blue', label="predict")
    plt.ylim(-1, 1)
    plt.legend(loc="upper right")
    plt.show()

Das Ergebnis der Ausfuehrung der Quantenschaltung ohne Training ist:

.. image:: ./images/parallel_model_result_no_train.png
   :width: 600 px
   :align: center

|

Eingabedaten erstellen, das parallele Quantenmodell definieren und das Trainingsmodell in Kombination mit der QuantumLayer des VQNet-Frameworks erstellen.

.. code-block::

    """
    Quantum Fourier Series
    """
    import numpy as np

    from pyvqnet.nn.module import Module
    from pyvqnet.nn.loss import MeanSquaredError
    from pyvqnet.optim.adam import Adam
    from pyvqnet.tensor.tensor import QTensor
    import pyqpanda as pq
    from pyvqnet.qnn.measure import expval
    from pyvqnet.qnn.quantumlayer import QuantumLayer
    import matplotlib.pyplot as plt
    import matplotlib
    try:
        matplotlib.use("TkAgg")
    except:
        print("Can not use matplot TkAgg")
        pass

    np.random.seed(42)

    degree = 1
    scaling = 1
    coeffs = [0.15 + 0.15j] * degree
    coeff0 = 0.1

    def target_function(x):
        """Generiert eine abgeschnittene Fourier-Reihe."""
        res = coeff0
        for idx, coeff in enumerate(coeffs):
            exponent = np.complex128(scaling * (idx + 1) * x * 1j)
            conj_coeff = np.conjugate(coeff)
            res += coeff * np.exp(exponent) + conj_coeff * np.exp(-exponent)
        return np.real(res)

    x = np.linspace(-6, 6, 70)
    target_y = np.array([target_function(xx) for xx in x])

    plt.plot(x, target_y, c='black')
    plt.scatter(x, target_y, facecolor='white', edgecolor='black')
    plt.ylim(-1, 1)
    plt.show()

    def S1(x, qubits):
        cir = pq.QCircuit()
        for q in qubits:
            cir.insert(pq.RX(q, x))
        return cir

    def W1(theta, qubits):
        cir = pq.QCircuit()
        for i in range(len(qubits)):
            cir.insert(pq.RZ(qubits[i], theta[0][i][0]))
            cir.insert(pq.RY(qubits[i], theta[0][i][1]))
            cir.insert(pq.RZ(qubits[i], theta[0][i][2]))

        for i in range(len(qubits) - 1):
            cir.insert(pq.CNOT(qubits[i], qubits[i + 1]))
        cir.insert(pq.CNOT(qubits[len(qubits) - 1], qubits[0]))

        for i in range(len(qubits)):
            cir.insert(pq.RZ(qubits[i], theta[1][i][0]))
            cir.insert(pq.RY(qubits[i], theta[1][i][1]))
            cir.insert(pq.RZ(qubits[i], theta[1][i][2]))

        cir.insert(pq.CNOT(qubits[0], qubits[len(qubits) - 1]))
        for i in range(len(qubits) - 1):
            cir.insert(pq.CNOT(qubits[i + 1], qubits[i]))

        for i in range(len(qubits)):
            cir.insert(pq.RZ(qubits[i], theta[2][i][0]))
            cir.insert(pq.RY(qubits[i], theta[2][i][1]))
            cir.insert(pq.RZ(qubits[i], theta[2][i][2]))

        for i in range(len(qubits) - 1):
            cir.insert(pq.CNOT(qubits[i], qubits[i + 1]))
        cir.insert(pq.CNOT(qubits[len(qubits) - 1], qubits[0]))

        return cir

    def q_circuits_loop(x, weights, qubits, clist, machine):

        result = []
        for xx in x:
            cir = pq.QCircuit()
            weights = weights.reshape([2, 3, 3, 3])

            cir.insert(W1(weights[0], qubits))
            cir.insert(S1(xx, qubits))

            cir.insert(W1(weights[1], qubits))
            prog = pq.QProg()
            prog.insert(cir)

            exp_vals = []
            for position in range(1):
                pauli_str = {"Z" + str(position): 1.0}
                exp2 = expval(machine, prog, pauli_str, qubits)
                exp_vals.append(exp2)
                result.append(exp2)
        return result


    class Model(Module):
        def __init__(self):
            super(Model, self).__init__()

            self.q_fourier_series = QuantumLayer(q_circuits_loop, 2 * 3 * 3 * 3, "CPU", 3)

        def forward(self, x):
            return self.q_fourier_series(x)

    def run():
        model = Model()

        optimizer = Adam(model.parameters(), lr=0.01)
        batch_size = 2
        epoch = 5
        loss = MeanSquaredError()
        print("start training..............")
        model.train()
        max_steps = 50
        for i in range(epoch):
            sum_loss = 0
            count = 0
            for step in range(max_steps):
                optimizer.zero_grad()
                batch_index = np.random.randint(0, len(x), (batch_size,))
                x_batch = x[batch_index]
                y_batch = target_y[batch_index]
                data, label = QTensor([x_batch]), QTensor([y_batch])
                result = model(data)
                loss_b = loss(label, result)
                loss_b.backward()
                optimizer._step()
                sum_loss += loss_b.item()
                count += batch_size

            loss_cout = sum_loss / count
            print(f"epoch:{i}, #### loss:{loss_cout} ")

            if loss_cout < 0.002:
                model.eval()
                predictions = []
                for xx in x:
                    data = QTensor([[xx]])
                    result = model(data)
                    predictions.append(result.pdata[0])

                plt.plot(x, target_y, c='black', label="true")
                plt.scatter(x, target_y, facecolor='white', edgecolor='black')
                plt.plot(x, predictions, c='blue', label="predict")
                plt.ylim(-1, 1)
                plt.legend(loc="upper right")
                plt.show()



    if __name__ == "__main__":
        run()


Das Quantenmodell ist:

.. image:: ./images/parallel_model_circuit.png
   :width: 600 px
   :align: center

|

Die Netzwerktrainingsergebnisse sind:

.. image:: ./images/parallel_model_result.png
   :width: 600 px
   :align: center

|

Der Netzwerktrainingsverlust ist:

.. code-block::

    start training..............
    epoch:0, #### loss:0.0037272341538482578
    epoch:1, #### loss:5.271130586635309e-05
    epoch:2, #### loss:4.714951917250687e-07
    epoch:3, #### loss:1.0968826371082763e-08
    epoch:4, #### loss:2.1258629738507562e-10

Ausdruckskraft von Quantenschaltungen
===============================================================================

In der Arbeit `Expressibility and entangling capability of parameterized quantum circuits for hybrid quantum-classical algorithms <https://arxiv.org/abs/1905.10876>`_,
schlagen die Autoren eine Methode zur Quantifizierung der Ausdruckskraft basierend auf der Fidelity-Wahrscheinlichkeitsverteilung zwischen den Ausgangszustaenden neuronaler Netze vor.
Fuer ein beliebiges Quantenneuronales Netz :math:`U(\vec{\theta})` werden die Parameter des neuronalen Netzes zweimal abgetastet (gesetzt auf :math:`\vec{\phi}` und :math:`\vec{\psi }` ).
Dann gehorcht die Fidelity zwischen den Ausgangszustaenden zweier Quantenschaltungen :math:`F=|\langle0|U(\vec{\phi})^\dagger U(\vec{\psi})|0\rangle|^ 2` einer Wahrscheinlichkeitsverteilung:

.. math::

    F\sim{P}(f)

Die Literatur weist darauf hin, dass, wenn das Quantenneuronale Netz :math:`U` gleichmassig auf allen unitaeren Matrizen verteilt sein kann (in diesem Fall heisst es, :math:`U` gehorcht der Haar-Verteilung), die Wahrscheinlichkeitsverteilung der Fidelity :math:`P_\text{Haar }(f)` erfuellt:

.. math::

    P_\text{Haar}(f)=(2^{n}-1)(1-f)^{2^n-2}

Die K-L-Divergenz (auch relative Entropie genannt) in der Statistik misst den Unterschied zwischen zwei Wahrscheinlichkeitsverteilungen. Die K-L-Divergenz zwischen zwei diskreten Wahrscheinlichkeitsverteilungen :math:`P,Q` ist definiert als:

.. math::

    D_{KL}(P||Q)=\sum_jP(j)\ln\frac{P(j)}{Q(j)}

Wenn die Fidelity-Verteilung des Quantenneuronalen Netzausgangs als :math:`P_\text{QNN}(f)` bezeichnet wird, dann ist die Ausdruckskraft des Quantenneuronalen Netzes definiert als die K-L-Divergenz zwischen :math:`P_\text{QNN}(f)` und :math:`P_\text{Haar}(f)`:

.. math::

    \text{Expr}_\text{QNN}=D_{KL}(P_\text{QNN}(f)||P_\text{Haar}(f))

Daher, wenn :math:`P_\text{QNN}(f)` naeher an :math:`P_\text{Haar}(f)` liegt, wird :math:`\text{Expr}` kleiner sein (mehr gegen 0 tendierend),
und die Ausdruckskraft des Quantenneuronalen Netzes ist staerker; umgekehrt, je groesser :math:`\text{Expr}` ist, desto schwaecher ist die Ausdruckskraft des Quantenneuronalen Netzes.
Wir koennen direkt die Ausdruckskraft von Ein-Bit-Quantenneuronalen Netzen nach dieser Definition berechnen: :math:`R_Y(\theta)` , :math:`R_Y(\theta_1)R_Z(\theta_2)` und
:math:`R_Y(\theta_1)R_Z(\theta_2)R_Y(\theta_3)`:

Im Folgenden wird mit VQNet die Quantenschaltungsausdruckskraft von `HardwareEfficientAnsatz <https://arxiv.org/abs/1704.05018>`_ in verschiedenen Tiefen (1, 2, 3) demonstriert.


.. code-block::

    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter
    from scipy import integrate
    from scipy.linalg import sqrtm
    from scipy.stats import entropy
    import pyqpanda as pq
    import numpy as np
    from pyvqnet.qnn.ansatz import HardwareEfficientAnsatz
    from pyvqnet.tensor import tensor
    from pyvqnet.qnn.quantum_expressibility.quantum_express import fidelity_of_cir, fidelity_harr_sample
    num_qubit = 1
    num_sample = 2000
    outputs_y = list()
    def plot_hist(data, num_bin, title_str):
        def to_percent(y, position):
            return str(np.around(y * 100, decimals=2)) + '%'
        plt.hist(data,
                weights=[1. / len(data)] * len(data),
                bins=np.linspace(0, 1, num=num_bin),
                facecolor="blue",
                edgecolor="black",
                alpha=0.7)
        plt.xlabel("Fidelity")
        plt.ylabel("Haeufigkeit")
        plt.title(title_str)
        formatter = FuncFormatter(to_percent)
        plt.gca().yaxis.set_major_formatter(formatter)
        plt.show()
    def cir(num_qubits, depth):
        machine = pq.CPUQVM()
        machine.init_qvm()
        qlist = machine.qAlloc_many(num_qubits)
        az = HardwareEfficientAnsatz(num_qubits, ["rx", "RY", "rz"],
                                    qlist,
                                    entangle_gate="cnot",
                                    entangle_rules="linear",
                                    depth=depth)
        w = tensor.QTensor(
            np.random.uniform(size=[az.get_para_num()], low=0, high=2 * np.pi))
        cir1 = az.create_ansatz(w)
        return cir1, machine, qlist
.. image:: ./images/haar-fidelity.png
   :width: 600 px
   :align: center

|

.. code-block::

    # Schaltungsbreite und maximale Tiefe festlegen
    num_qubit = 4
    max_depth = 3
    # Fidelity-Verteilung entsprechend der Haar-Abtastung berechnen

    flist, p_haar, theory_haar = fidelity_harr_sample(num_qubit, num_sample)
    title_str = "haar, %d qubit(s)" % num_qubit
    plot_hist(flist, 50, title_str)
    Expr_cel = list()
    # Ausdruckskraft von neuronalen Netzen mit unterschiedlichen Tiefen berechnen
    for DEPTH in range(1, max_depth + 1):
        print("Abtastung der Schaltung bei Tiefe %d..." % DEPTH)
        f_list, p_cel = fidelity_of_cir(HardwareEfficientAnsatz, num_qubit, DEPTH,
                                        num_sample)
        title_str = f"HardwareEfficientAnsatz, {num_qubit} qubit(s) {DEPTH} layer(s)"
        plot_hist(f_list, 50, title_str)
        expr = entropy(p_cel, theory_haar)
        Expr_cel.append(expr)
    # Ausdruckskraft von neuronalen Netzen unterschiedlicher Tiefen vergleichen
    print(
        f"Die Ausdruckskraft von neuronalen Netzen mit Tiefen 1, 2 und 3 betraegt { np.around(Expr_cel, decimals=4)}, je kleiner desto besser.", )
    plt.plot(range(1, max_depth + 1), Expr_cel, marker='>')
    plt.xlabel("Tiefe")
    plt.yscale('log')
    plt.ylabel("Expr.")
    plt.xticks(range(1, max_depth + 1))
    plt.title("Ausdruckskraft vs. Schaltungstiefe")
    plt.show()

.. image:: ./images/f1.png
   :width: 600 px
   :align: center

|

.. image:: ./images/f2.png
   :width: 600 px
   :align: center

|

.. image:: ./images/f3.png
   :width: 600 px
   :align: center

|

.. image:: ./images/express.png
   :width: 600 px
   :align: center

|



Quantum Perceptron
=======================================

Kuenstliche neuronale Netze sind das Herz von Algorithmen des maschinellen Lernens und von Protokollen der kuenstlichen Intelligenz. Historisch gesehen geht die einfachste Implementierung eines kuenstlichen Neurons auf das klassische Rosenblatt-Perzeptron zurueck, aber seine langfristigen praktischen Anwendungen koennten durch die schnell wachsende Rechenkomplexitaet behindert werden, die besonders fuer das Training mehrschichtiger Perzeptronnetze relevant ist.
Hier beziehen wir uns auf die Arbeit `An Artificial Neuron Implemented on an Actual Quantum Processor <https://arxiv.org/abs/1811.02266>`_, die einen quanteninformationsbasierten Algorithmus zur Implementierung der Quantencomputerversion eines Perzeptrons vorstellt, der einen exponentiellen Vorteil bei der Kodierungsressourcen gegenueber alternativen Realisierungen zeigt.

Fuer dieses Quantenperzeptron sind die verarbeiteten Daten ein String aus 0-1-Binaerbits. Das Ziel ist es, Muster zu identifizieren, die wie ein W-Kreuz geformt sind, wie in der folgenden Abbildung dargestellt.

.. image:: ./images/QP-data.png
   :width: 600 px
   :align: center

|

Es wird mit einem Binaerbit-String kodiert, wobei Schwarz fuer 0 und Weiss fuer 1 steht, sodass w als (1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 0, 0, 1, 1, 0, 1) kodiert wird. Insgesamt 16-Bit-Strings koennen in das Vorzeichen der Amplitude des 4-Bit-Quantenzustands kodiert werden. Das Vorzeichen ist 0 fuer negative Zahlen und 1 fuer positive Zahlen. Durch die obige Kodierungsmethode wird unsere Algorithmuseingabe in einen 16-Bit-Binaerstring umgewandelt. Solche nicht-wiederholenden Binaerstrings koennen jeweils einem bestimmten Eingabe :math:`U_i` entsprechen.

Die Schaltungsstruktur des in dieser Arbeit vorgeschlagenen Quantenperzeptrons ist wie folgt:

.. image:: ./images/QP-cir.png
   :width: 600 px
   :align: center

|

Die Kodierungsschaltung :math:`U_i` wird auf Bits 0~3 aufgebaut und enthaelt mehrfach gesteuerte :math:`CZ`-, :math:`CNOT`-Gatter und :math:`H`-Gatter; die Gewichtstransformationsschaltung :math:`U_w` wird unmittelbar nach :math:`U_i` konstruiert und besteht ebenfalls aus gesteuerten Gattern und :math:`H`-Gattern. :math:`U_i` kann verwendet werden, um unitaere Matrixtransformationen durchzufuehren, um Daten in Quantenzustaende zu kodieren:

.. math::
    U_i|0\rangle^{\otimes N}=\left|\psi_i\right\rangle

Verwenden Sie die unitaere Matrixtransformation :math:`U_w`, um das innere Produkt zwischen Eingabe und Gewichten zu berechnen:

.. math::
    U_w\left|\psi_i\right\rangle=\sum_{j=0}^{m-1} c_j|j\rangle \equiv\left|\phi_{i, w}\right\rangle

Die normalisierten Aktivierungswahrscheinlichkeitswerte fuer :math:`U_i` und :math:`U_w` koennen durch Verwendung eines mehrfach gesteuerten NOT-Gatters mit Zielbits auf Hilfsbits und einigen nachfolgenden :math:`H`-, :math:`X`- und :math:`CX`-Gattern als Aktivierungsfunktionen erhalten werden:

.. math::
    \left|\phi_{i, w}\right\rangle|0\rangle_a \rightarrow \sum_{j=0}^{m-2} c_j|j\rangle|0\rangle_a+c_{m-1}|m-1\rangle|1\rangle_a

Wenn der Binaerstring der Eingabe i genau mit w uebereinstimmt, sollte der normalisierte Wahrscheinlichkeitswert am groessten sein.

VQNet stellt das ``QuantumNeuron``-Modul zur Implementierung dieses Algorithmus bereit. Initialisieren Sie zunaechst ein Quantenperzeptron ``QuantumNeuron``.

.. code-block::

    perceptron = QuantumNeuron()

Verwenden Sie die ``gen_4bitstring_data``-Schnittstelle, um verschiedene Daten aus der Arbeit und deren Kategorielabels zu generieren.

.. code-block::

    training_label, test_label = perceptron.gen_4bitstring_data()

Durch die Verwendung der ``train``-Schnittstelle zum Durchlaufen aller Daten erhalten Sie die letzte trainierte Quantenperzeptronschaltung :math:`U_w`.

.. code-block::

    trained_para = perceptron.train(training_label, test_label)

.. image:: ./images/QP-pic.png
   :width: 600 px
   :align: center

|

Auf den Testdaten koennen die Genauigkeitsergebnisse erzielt werden

.. image:: ./images/QP-acc.png
   :width: 600 px
   :align: center

|



Doubly Stochastic Gradient Descent
===============================================================================

In variativen Quantenalgorithmen werden parametrisierte Quantenschaltungen durch
klassischen Gradientenabstieg optimiert, um den erwarteten Funktionswert zu minimieren.
Obwohl der Erwartungswert im klassischen Simulator analytisch berechnet werden kann,
ist das Programm auf Quantenhardware auf die Abtastung des Erwartungswerts beschraenkt;
mit zunehmender Anzahl von Abtastungen und Schusszahlen wird
der auf diese Weise erhaltene Erwartungswert zum theoretischen Erwartungswert konvergieren,
koennte aber immer ein genauer Wert sein.
Sweke et al. fanden eine doppelt stochastische Gradientenabstiegsmethode in `der Arbeit <https://arxiv.org/abs/1910.01155>`_.
In dieser Arbeit zeigen sie, dass der Quanten-Gradientenabstieg, der eine endliche Anzahl von Messabtastungen
(oder Schusszahlen) zur Schaetzung von Gradienten verwendet, eine Form des stochastischen Gradientenabstiegs ist.
Darueber hinaus, wenn die Optimierung eine Linearkombination von
Erwartungswerten (wie VQE) beinhaltet, kann die Abtastung aus den Termen dieser
Linearkombination die erforderliche Zeitkomplexitaet weiter reduzieren.

VQNet implementiert ein Beispiel dieses Algorithmus: Loesung der Grundzustandsenergie des Ziel-Hamiltonoperators mit VQE. Beachten Sie, dass wir hier die Anzahl der Schusszahlen fuer Quantenschaltungsbeobachtungen auf nur 1 setzen.

.. math::

    H = \begin{bmatrix}
          8 & 4 & 0 & -6\\
          4 & 0 & 4 & 0\\
          0 & 4 & 8 & 0\\
          -6 & 0 & 0 & 0
        \end{bmatrix}.

.. code-block::

    import numpy as np
    import pyqpanda as pq
    from pyvqnet.qnn.template import StronglyEntanglingTemplate
    from pyvqnet.qnn.measure import Hermitian_expval
    from pyvqnet.qnn import QuantumLayerV2
    from pyvqnet.optim import SGD
    import pyvqnet._core as _core
    from pyvqnet.tensor import QTensor
    from matplotlib import pyplot as plt
    num_layers = 2
    num_wires = 2
    eta = 0.01
    steps = 200
    n = 1
    param_shape = [2, 2, 3]
    shots = 1
    H = np.array([[8, 4, 0, -6], [4, 0, 4, 0], [0, 4, 8, 0], [-6, 0, 0, 0]])
    init_params = np.random.uniform(low=0,
                                    high=2 * np.pi,
                                    size=param_shape).astype(np.float32)
    # einige grundlegende Pauli-Matrizen
    I = np.eye(2)
    X = np.array([[0, 1], [1, 0]])
    Y = np.array([[0, -1j], [1j, 0]])
    Z = np.array([[1, 0], [0, -1]])

    def pq_circuit(params):
        params = params.reshape(param_shape)
        num_qubits = 2
        machine = pq.CPUQVM()
        machine.init_qvm()
        qubits = machine.qAlloc_many(num_qubits)
        circuit = StronglyEntanglingTemplate(params, num_qubits=num_qubits)
        qcir = circuit.create_circuit(qubits)
        prog = pq.QProg()
        prog.insert(qcir)
        machine.directly_run(prog)
        result = machine.get_qstate()
        return result

Der Hamiltonoperator in diesem Beispiel ist eine hermitesche Matrix,
die wir immer als Summe von Pauli-Matrizen darstellen koennen.

.. math::

    H = \sum_{i,j=0,1,2,3} a_{i,j} (\sigma_i\otimes \sigma_j)

und

.. math::

    a_{i,j} = \frac{1}{4}\text{tr}[(\sigma_i\otimes \sigma_j )H], ~~ \sigma = \{I, X, Y, Z\}.

Durch Einsetzen in die obige Formel koennen wir sehen, dass

.. math::

    H = 4  + 2I\otimes X + 4I \otimes Z - X\otimes X + 5 Y\otimes Y + 2Z\otimes X.

Um "doppelt stochastischen" Gradientenabstieg durchzufuehren, wenden wir einfach die stochastische Gradientenabstiegsmethode an, nehmen aber zusaetzlich bei jedem Optimierungsschritt eine gleichmassige Teilmenge der Hamilton-Erwartungswerte.
Die Funktion vqe_func_analytic() verwendet Parametershift zur Berechnung theoretischer Gradienten,
und vqe_func_shots() verwendet zufaellig abgetastete Werte und zufaellig abgetastete Hamilton-Erwartungswerte
fuer "doppelt stochastische" Gradientenberechnungen.

.. code-block::

    terms = np.array([
        2 * np.kron(I, X),
        4 * np.kron(I, Z),
        -np.kron(X, X),
        5 * np.kron(Y, Y),
        2 * np.kron(Z, X),
    ])
    def vqe_func_analytic(input, init_params):
        qstate = pq_circuit(init_params)
        expval = Hermitian_expval(H, qstate, [0, 1], 2)
        return  expval
    def vqe_func_shots(input, init_params):
        qstate = pq_circuit(init_params)
        idx = np.random.choice(np.arange(5), size=n, replace=False)
        A = np.sum(terms[idx], axis=0)
        expval = Hermitian_expval(A, qstate, [0, 1], 2, shots)
        return 4 + (5 / 1) * expval


Verwenden Sie VQNet zur Parameteroptimierung und vergleichen Sie die Kurve der Verlustfunktion.
Da die doppelt stochastische Gradientenabstiegsmethode jedes Mal nur die partielle Pauli-Operatoreinsumme von H berechnet,
kann daher der Durchschnittswert verwendet werden, um das erwartete Ergebnis der endgueltigen
Beobachtung darzustellen. Hier wird der gleitende Durchschnitt moving_average() zur Berechnung verwendet.

.. code-block::


    ##############################################################################
    # Optimierung der Schaltung mittels Gradientenabstieg durch die Parameter-Shift-Regel:
    qlayer_ana = QuantumLayerV2(vqe_func_analytic, 2*2*3 )
    qlayer_shots = QuantumLayerV2(vqe_func_shots, 2*2*3 )
    cost_sgd = []
    cost_dsgd = []
    temp = _core.Tensor(init_params)
    _core.vqnet.copyTensor(temp, qlayer_ana.m_para.data)
    opti_ana = SGD(qlayer_ana.parameters())

    _core.vqnet.copyTensor(temp, qlayer_shots.m_para.data)
    opti_shots = SGD(qlayer_shots.parameters())
    
    for i in range(steps):
        opti_ana.zero_grad()
        loss = qlayer_ana(QTensor([[1.0]]))
        loss.backward()
        cost_sgd.append(loss.item())
        opti_ana._step()
    for i in range(steps+50):
        opti_shots.zero_grad()
        loss = qlayer_shots(QTensor([[1.0]]))
        loss.backward()
        cost_dsgd.append(loss.item())
        opti_shots._step()
    def moving_average(data, n=3):
        ret = np.cumsum(data, dtype=np.float64)
        ret[n:] = ret[n:] - ret[:-n]
        return ret[n - 1:] / n
    ta = moving_average(np.array(cost_dsgd), n=50)
    ta = ta[:-26]
    average = np.vstack([np.arange(25, 200),ta ])
    final_param = qlayer_shots.parameters()[0].to_numpy()
    print("Doubly stochastic gradient descent min energy = ", vqe_func_analytic(QTensor([1]),final_param))
    final_param  = qlayer_ana.parameters()[0].to_numpy()
    print("stochastic gradient descent min energy = ", vqe_func_analytic(QTensor([1]),final_param))
    plt.plot(cost_sgd, label="Vanilla gradient descent")
    plt.plot(cost_dsgd, ".", label="Doubly QSGD")
    plt.plot(average[0], average[1], "--", label="Doubly QSGD (moving average)")
    plt.ylabel("Cost function value")
    plt.xlabel("Optimization steps")
    plt.xlim(-2, 200)
    plt.legend()
    plt.show()

.. image:: ./images/dsgd.png
   :width: 600 px
   :align: center


Barren plateaus
===============================================================================


Im Training klassischer neuronaler Netze begegnen gradientenbasierte Optimierungsmethoden nicht nur dem Problem lokaler Minima,
sondern auch geometrischen Strukturen wie Sattelpunkten, an denen der Gradient nahe Null ist.
Entsprechend existiert der **Barren-Plateau-Effekt** (Plateaus) auch im Quantenneuronalen Netz.
Dieses eigentuemliche Phaenomen wurde erstmals 2018 von McClean et al. entdeckt: `Barren plateaus in quantum neural network training landscapes <https://arxiv.org/abs/1803.11173>`_.
Einfach ausgedrueckt wird die Optimierungslandschaft sehr flach, wenn man eine zufaellige Schaltungsstruktur waehlt, die ein bestimmtes Komplexitaetsniveau erfuellt,
wodurch gradientenbasierte Optimierungsmethoden Schwierigkeiten haben, das globale Minimum zu finden.
Fuer die meisten variativen Quantenalgorithmen (VQE usw.) bedeutet dieses Phaenomen, dass bei zunehmender Anzahl von Qubits
Schaltungen mit zufaelligen Strukturen moeglicherweise nicht gut funktionieren.
Dies wird die Optimierungsoberflaeche, die der gut entworfenen Verlustfunktion entspricht, in eine riesige Plattform verwandeln,
was das Training von Quantenneuronalen Netzen erschwert.
Der vom Modell zufaellig gefundene Anfangswert kann dieser Plattform nur schwer entkommen, und die Konvergenzgeschwindigkeit des Gradientenabstiegs wird sehr langsam sein.


Dieser Fall verwendet hauptsaechlich VQNet, um das Barren-Plateau-Phaenomen anzuzeigen, und verwendet die Gradientenanalysefunktion, um den Parametergradienten im benutzerdefinierten Quantenneuronalen Netz zu analysieren.

Der folgende Code erstellt die folgende zufaellige Schaltung nach der aehnlichen Methode, die in der Originalarbeit erwahnt wird:

Zuerst wird auf alle Qubits mit einer Rotation um die Y-Achse der Bloch-Kugel :math:`\pi/4` angewendet.

Der Rest der Strukturen addiert sich zu einem Modul (Block), jedes Modul ist in zwei Schichten unterteilt:

- Die erste Schicht erstellt eine zufaellige Drehtuer, wobei :math:`R \in \{R_x, R_y, R_z\}`.
- Die zweite Schicht besteht aus CZ-Gattern, die auf jedes benachbarte Qubit-Paar wirken.

Der Schaltungscode ist in der Funktion rand_circuit_pq gezeigt.

Nachdem wir die Struktur der Schaltung bestimmt haben, muessen wir auch eine Verlustfunktion definieren, um die Optimierungsoberflaeche zu bestimmen.
Wie in der Originalarbeit erwahnt, verwenden wir die im VQE-Algorithmus uebliche Verlustfunktion:

.. math::

    \mathcal{L}(\boldsymbol{\theta})= \langle0| U^{\dagger}(\boldsymbol{\theta})H U(\boldsymbol{\theta}) |0\rangle

Die unitaere Matrix :math:`U(\boldsymbol{\theta})` ist das Quantenneuronale Netz mit zufaelliger Struktur, das wir im vorherigen Teil erstellt haben.
Hamiltonian :math:`H = |00\cdots 0\rangle\langle00\cdots 0|`.
In diesem Fall wird der obige VQE-Algorithmus auf verschiedenen Qubit-Anzahlen konstruiert, und es werden 200 Sets zufaelliger Netzwerkstrukturen und verschiedener zufaelliger Anfangsparameter generiert.
Der Gradient der Parameter in der Leitung mit Parametern wird nach dem Parameter-Shift-Algorithmus berechnet.
Dann werden der Durchschnitt und die Varianz der erhaltenen 200 Gradienten der Variationsparameter gezaehlt.

Das folgende Beispiel analysiert den letzten der variablen Quantenparameter, und die Leser koennen ihn auch auf andere sinnvolle Werte aendern.
Durch die Ausfuehrung werden die Leser feststellen, dass mit zunehmender Anzahl von Qubits die Varianz des Gradienten der Quantenparameter immer kleiner wird und der Mittelwert naeher an 0 liegt.

.. code-block::


        """
        barren plateau
        """
        import pyqpanda as pq
        import numpy as np
        import matplotlib.pyplot as plt

        from pyvqnet.qnn import Hermitian_expval, grad

        param_idx = -1
        gate_set = [pq.RX, pq.RY, pq.RZ]


        def rand_circuit_pq(params, num_qubits):
            cir = pq.QCircuit()
            machine = pq.CPUQVM()
            machine.init_qvm()
            qlist = machine.qAlloc_many(num_qubits)

            for i in range(num_qubits):
                cir << pq.RY(
                    qlist[i],
                    np.pi / 4,
                )

            random_gate_sequence = {
                i: np.random.choice(gate_set)
                for i in range(num_qubits)
            }
            for i in range(num_qubits):
                cir << random_gate_sequence[i](qlist[i], params[i])

            for i in range(num_qubits - 1):
                cir << pq.CZ(qlist[i], qlist[i + 1])

            prog = pq.QProg()
            prog.insert(cir)
            machine.directly_run(prog)
            result = machine.get_qstate()

            H = np.zeros((2**num_qubits, 2**num_qubits))
            H[0, 0] = 1
            expval = Hermitian_expval(H, result, [i for i in range(num_qubits)],
                                    num_qubits)

            return expval


        qubits = [2, 3, 4, 5, 6]
        variances = []
        num_samples = 200
        means = []
        for num_qubits in qubits:
            grad_vals = []
            for i in range(num_samples):
                params = np.random.uniform(0, np.pi, size=num_qubits)
                g = grad(rand_circuit_pq, params, num_qubits)

                grad_vals.append(g[-1])
            variances.append(np.var(grad_vals))
            means.append(np.mean(grad_vals))
        variances = np.array(variances)
        means = np.array(means)
        qubits = np.array(qubits)


        plt.figure()

        plt.plot(qubits, variances, "v-")

        plt.xlabel(r"N Qubits")
        plt.ylabel(r"Varianz")
        plt.show()


        plt.figure()
        plt.plot(qubits, means, "v-")

        plt.xlabel(r"N Qubits")
        plt.ylabel(r"Mittelwert")

        plt.show()


Die folgende Abbildung zeigt, wie der Mittelwert des Parametergradienten mit der Anzahl der Qubits variiert. Mit zunehmender Anzahl von Qubits naehert sich der Parametergradient 0.

.. image:: ./images/Barren_Plateau_mean.png
   :width: 600 px
   :align: center

|

Die folgende Abbildung zeigt, wie die Varianz des Parametergradienten mit der Anzahl der Qubits variiert, und der Parametergradient aendert sich bei zunehmender Anzahl von Qubits kaum.
Es ist absehbar, dass die durch beliebige parametrische Logikgatter aufgebaute Quantenschaltung bei beliebiger Parameterinitialisierung bei steigender Qubit-Anzahl nur schwer zu aktualisieren ist.

.. image:: ./images/Barren_Plateau_variance.png
   :width: 600 px
   :align: center

|


Gradient based pruning
===============================================================================

Das folgende Beispiel implementiert den Algorithmus aus der Arbeit `Towards Efficient On-Chip Training of Quantum Neural Networks <https://openreview.net/forum?id=vKefw-zKOft>`_.
Durch sorgfaeltiges Studium des Prozesses von Parametern in der Quantenvariationsschaltung beobachteten die Forscher, dass kleine Gradienten unter Quantenrauschen oft grosse relative Aenderungen oder sogar falsche Richtungen aufweisen.
Auch sind nicht alle Gradientenberechnungen fuer den Trainingsprozess notwendig, insbesondere bei Gradienten mit kleiner Groesse.
Inspiriert davon schlagen die Forscher eine probabilistische Gradienten-Beschneidungsmethode vor, um nur Gradienten mit hoher Zuverlaessigkeit vorherzusagen und zu berechnen.
Der Ansatz reduziert Rauscheffekte und spart auch die Anzahl der Schaltungen, die auf einer echten Quantenmaschine ausgefuehrt werden muessen.

Im gradientenbasierten Beschneidungsalgorithmus werden fuer den Optimierungsprozess von Parametern zwei Phasen unterteilt: Akkumulationsfenster und Beschneidungsfenster, und alle Trainingsperioden werden in eine wiederholte Abfolge von Akkumulationsfenster und dann Beschneidungsfenster unterteilt. Es gibt drei wichtige Hyperparameter in der probabilistischen Gradienten-Beschneidungsmethode:

     * Akkumulationsfensterbreite :math:`\omega_a` , 
     * Bescheidungsverhaeltnis :math:`r` ,
     * Bescheidungsfensterbreite :math:`\omega_p` .

Im Akkumulationsfenster sammeln die Forscher die Gradienteninformationen in jedem Trainingsschritt. Bei jedem Schritt des Beschneidungsfensters
ueberspringt der Algorithmus basierend auf den aus dem Akkumulationsfenster gesammelten Informationen und dem Bescheidungsverhaeltnis
probabilistisch einige Gradientenberechnungen.

.. image:: ./images/gbp_arch.png
   :width: 600 px
   :align: center

|

Das Bescheidungsverhaeltnis :math:`r`, die Akkumulationsfensterbreite :math:`\omega_a` und die Bescheidungsfensterbreite :math:`\omega_p` bestimmen jeweils die Zuverlaessigkeit der Gradiententrendbewertung.
Somit betraegt die prozentuale Zeitersparnis unserer probabilistischen Gradienten-Beschneidungsmethode :math:`r\tfrac{\omega_p}{\omega_a +\omega_p}\times 100\%`.
Im Folgenden wird die Anwendung des QVC-Klassifikationsbeispiels mit dem Gradienten-Beschneidungsalgorithmus gezeigt.

.. code-block::

    import random
    import numpy as np
    import pyqpanda as pq

    from pyvqnet.data import data_generator as dataloader
    from pyvqnet.nn.module import Module
    from pyvqnet.optim import sgd
    from pyvqnet.qnn.quantumlayer import QuantumLayer
    from pyvqnet.nn.loss import CategoricalCrossEntropy
    from pyvqnet.tensor.tensor import QTensor
    from pyvqnet.qnn import Gradient_Prune_Instance
    random.seed(1234)

    qvc_train_data = [
        0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 0, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 1,
        1, 0, 0, 1, 0, 1, 0, 1, 0, 0, 1, 0, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 0, 1, 1,
        1, 1, 1, 0, 1, 1, 1, 1, 1, 0
    ]
    qvc_test_data = [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 0, 1, 0, 0, 1, 1, 0]


    def qvc_circuits(x, weights, qlist, clist, machine):
        """
        Funktion zum Ausfuehren von Quantenschaltungen
        """
        def get_cnot(nqubits):
            cir = pq.QCircuit()
            for i in range(len(nqubits) - 1):
                cir.insert(pq.CNOT(nqubits[i], nqubits[i + 1]))
            cir.insert(pq.CNOT(nqubits[len(nqubits) - 1], nqubits[0]))
            return cir

        def build_circuit(weights, xx, nqubits):
            def Rot(weights_j, qubits):
                circuit = pq.QCircuit()
                circuit.insert(pq.RZ(qubits, weights_j[0]))
                circuit.insert(pq.RY(qubits, weights_j[1]))
                circuit.insert(pq.RZ(qubits, weights_j[2]))
                return circuit

            def basisstate():
                circuit = pq.QCircuit()
                for i in range(len(nqubits)):
                    if xx[i] == 1:
                        circuit.insert(pq.X(nqubits[i]))
                return circuit

            circuit = pq.QCircuit()
            circuit.insert(basisstate())

            for i in range(weights.shape[0]):

                weights_i = weights[i, :, :]
                for j in range(len(nqubits)):
                    weights_j = weights_i[j]
                    circuit.insert(Rot(weights_j, nqubits[j]))
                cnots = get_cnot(nqubits)
                circuit.insert(cnots)

            circuit.insert(pq.Z(nqubits[0]))

            prog = pq.QProg()

            prog.insert(circuit)
            return prog

        weights = weights.reshape([2, 4, 3])
        prog = build_circuit(weights, x, qlist)
        prob = machine.prob_run_dict(prog, qlist[0], -1)
        prob = list(prob.values())

        return prob


    def qvc_circuits2(x, weights, qlist, clist, machine):
        """
        Funktion zum Ausfuehren von Quantenschaltungen
        """
        prog = pq.QProg()
        circuit = pq.QCircuit()
        circuit.insert(pq.RZ(qlist[0], x[0]))
        circuit.insert(pq.RZ(qlist[1], x[1]))
        circuit.insert(pq.CNOT(qlist[0], qlist[1]))
        circuit.insert(pq.CNOT(qlist[1], qlist[2]))
        circuit.insert(pq.CNOT(qlist[2], qlist[3]))
        circuit.insert(pq.RY(qlist[0], weights[0]))
        circuit.insert(pq.RY(qlist[1], weights[1]))
        circuit.insert(pq.RY(qlist[2], weights[2]))

        circuit.insert(pq.CNOT(qlist[0], qlist[1]))
        circuit.insert(pq.CNOT(qlist[1], qlist[2]))
        circuit.insert(pq.CNOT(qlist[2], qlist[3]))
        prog.insert(circuit)
        prob = machine.prob_run_dict(prog, qlist[0], -1)
        prob = list(prob.values())

        return prob

    class Model(Module):
        def __init__(self):
            super(Model, self).__init__()
            self.qvc = QuantumLayer(qvc_circuits, 24, "cpu", 4)

        def forward(self, x):
            y = self.qvc(x)
            return y


    def get_data(dataset_str):
        """
        Wandelt Daten in gueltige Form um
        """
        if dataset_str == "train":
            datasets = np.array(qvc_train_data)

        else:
            datasets = np.array(qvc_test_data)

        datasets = datasets.reshape([-1, 5])
        data = datasets[:, :-1]
        label = datasets[:, -1].astype(int)
        label = np.eye(2)[label].reshape(-1, 2)
        return data, label




    def get_accuracy(result, label):
        result, label = np.array(result.data), np.array(label.data)
        score = np.sum(np.argmax(result, axis=1) == np.argmax(label, 1))
        return score

Wir verwenden die Klasse ``Gradient_Prune_Instance`` und geben `24` als Anzahl der Parameter `param_num` ein.
Das Bescheidungsverhaeltnis `prune_ratio` betraegt 0.5,
die Akkumulationsfenstergroesse `accumulation_window_size` betraegt 4,
und das Bescheidungsfenster `pruning_window_size` betraegt 2.
Bei jedem Durchlauf des Rueckwaertsteils des Codes, vor dem Optimierer ``step``,
fuehren Sie die ``step``-Funktion von ``Gradient_Prune_Instance`` aus.

.. code-block::

    def run():
        """
        Hauptausfuehrungsfunktion
        """
        model = Model()

        optimizer = sgd.SGD(model.parameters(), lr=0.5)
        batch_size = 3
        epoch = 10
        loss = CategoricalCrossEntropy()
        print("start training..............")
        model.train()

        datas, labels = get_data("train")
        print(datas)
        print(labels)
        print(datas.shape)


        GBP_HELPER = Gradient_Prune_Instance(param_num = 24,prune_ratio=0.5,accumulation_window_size=4,pruning_window_size=2)
        for i in range(epoch):
            count = 0
            sum_loss = 0
            accuary = 0
            t = 0
            for data, label in dataloader(datas, labels, batch_size, False):
                optimizer.zero_grad()
                data, label = QTensor(data), QTensor(label)

                result = model(data)

                loss_b = loss(label, result)

                loss_b.backward()
                
                GBP_HELPER.step(model.parameters())
                optimizer._step()
                sum_loss += loss_b.item()
                count += batch_size
                accuary += get_accuracy(result, label)
                t = t + 1

            print(
                f"epoch:{i}, #### loss:{sum_loss/count} #####accuracy:{accuary/count}"
            )
        print("start testing..............")
        model.eval()
        count = 0

        test_data, test_label = get_data("test")
        test_batch_size = 1
        accuary = 0
        sum_loss = 0
        for testd, testl in dataloader(test_data, test_label, test_batch_size):
            testd = QTensor(testd)
            test_result = model(testd)
            test_loss = loss(testl, test_result)
            sum_loss += test_loss
            count += test_batch_size
            accuary += get_accuracy(test_result, testl)
        print(
            f"test:--------------->loss:{sum_loss/count} #####accuracy:{accuary/count}"
        )


    if __name__ == "__main__":

        run()


Modelltraining mit der Quantenberechnungsschicht in VQNet
*******************************************************************

Im Folgenden finden Sie einige Beispiele fuer die Verwendung der VQNet-Schnittstelle fuer quantenmaschinelles Lernen ``QuantumLayer`` , ``NoiseQuantumLayer`` .

Modelltraining mit QuantumLayer in VQNet
===============================================================================

.. code-block::

    import sys,os
    from pyvqnet.nn.module import Module
    from pyvqnet.optim import sgd
    import numpy as np

    from pyvqnet.data import data_generator as dataloader
    from pyvqnet.nn.linear import Linear
    from pyvqnet.nn.loss import CategoricalCrossEntropy

    from pyvqnet.tensor.tensor import QTensor
    import random
    import pyqpanda as pq
    from pyvqnet.qnn.quantumlayer import QuantumLayer
    from pyqpanda import *
    random.seed(1234)

    qvc_train_data = [0,1,0,0,1,
    0, 1, 0, 1, 0,
    0, 1, 1, 0, 0,
    0, 1, 1, 1, 1,
    1, 0, 0, 0, 1,
    1, 0, 0, 1, 0,
    1, 0, 1, 0, 0,
    1, 0, 1, 1, 1,
    1, 1, 0, 0, 0,
    1, 1, 0, 1, 1,
    1, 1, 1, 0, 1,
    1, 1, 1, 1, 0]
    qvc_test_data= [0, 0, 0, 0, 0,
    0, 0, 0, 1, 1,
    0, 0, 1, 0, 1,
    0, 0, 1, 1, 0]

    def qvc_circuits(input,weights,qlist,clist,machine):

        def get_cnot(nqubits):
            cir = pq.QCircuit()
            for i in range(len(nqubits)-1):
                cir.insert(pq.CNOT(nqubits[i],nqubits[i+1]))
            cir.insert(pq.CNOT(nqubits[len(nqubits)-1],nqubits[0]))
            return cir

        def build_circuit(weights, xx, nqubits):

            def Rot(weights_j, qubits):
                circuit = pq.QCircuit()
                circuit.insert(pq.RZ(qubits, weights_j[0]))
                circuit.insert(pq.RY(qubits, weights_j[1]))
                circuit.insert(pq.RZ(qubits, weights_j[2]))
                return circuit
            def basisstate():
                circuit = pq.QCircuit()
                for i in range(len(nqubits)):
                    if xx[i]==1:
                        circuit.insert(pq.X(nqubits[i]))
                return circuit

            circuit = pq.QCircuit()
            circuit.insert(basisstate())

            for i in range(weights.shape[0]):

                weights_i = weights[i,:,:]
                for j in range(len(nqubits)):
                    weights_j = weights_i[j]
                    circuit.insert(Rot(weights_j,nqubits[j]))
                cnots = get_cnot(nqubits)
                circuit.insert(cnots)

            circuit.insert(pq.Z(nqubits[0]))

            prog = pq.QProg()

            prog.insert(circuit)
            return prog

        weights = weights.reshape([2,4,3])
        prog = build_circuit(weights,input,qlist)
        prob = machine.prob_run_dict(prog, qlist[0], -1)
        prob = list(prob.values())

        return prob

    class Model(Module):
        def __init__(self):
            super(Model, self).__init__()
            self.qvc = QuantumLayer(qvc_circuits,24,"cpu",4)

        def forward(self, x):
            return self.qvc(x)

    def get_data(dataset_str):
        if dataset_str == "train":
            datasets = np.array(qvc_train_data)

        else:
            datasets = np.array(qvc_test_data)

        datasets = datasets.reshape([-1,5])
        data = datasets[:,:-1]
        label = datasets[:,-1].astype(int)
        label = np.eye(2)[label].reshape(-1,2)
        return data, label

    def get_accuracy(result,label):
        result,label = np.array(result.data), np.array(label.data)
        score = np.sum(np.argmax(result,axis=1)==np.argmax(label,1))
        return score

    def Run():

        model = Model()

        optimizer = sgd.SGD(model.parameters(),lr =0.5)
        batch_size = 3
        epoch = 10
        loss = CategoricalCrossEntropy()
        print("start training..............")
        model.train()

        datas,labels = get_data("train")
        print(datas)
        print(labels)
        print(datas.shape)
        for i in range(epoch):
            count=0
            sum_loss = 0
            accuary = 0
            t = 0
            for data,label in dataloader(datas,labels,batch_size,False):
                optimizer.zero_grad()
                data,label = QTensor(data), QTensor(label)

                result = model(data)

                loss_b = loss(label,result)
                loss_b.backward()
                optimizer._step()
                sum_loss += loss_b.item()
                count+=batch_size
                accuary += get_accuracy(result,label)
                t = t + 1

            print(f"epoch:{i}, #### loss:{sum_loss/count} #####accuracy:{accuary/count}")
        print("start testing..............")
        model.eval()
        count = 0
        test_data, test_label = get_data("test")
        test_batch_size = 1
        accuary = 0
        sum_loss = 0
        for testd,testl in dataloader(test_data,test_label,test_batch_size):
            testd = QTensor(testd)
            test_result = model(testd)
            test_loss = loss(testl,test_result)
            sum_loss += test_loss
            count+=test_batch_size
            accuary += get_accuracy(test_result,testl)
        print(f"test:--------------->loss:{sum_loss/count} #####accuracy:{accuary/count}")

    if __name__=="__main__":

        Run()

Verlust- und Genauigkeitsergebnisse der Ausfuehrung:

.. code-block::
	
	start training..............
	epoch:0, #### loss:[0.20585182] #####accuracy:0.6
	epoch:1, #### loss:[0.17479989] #####accuracy:1.0
	epoch:2, #### loss:[0.12679021] #####accuracy:1.0
	epoch:3, #### loss:[0.11088503] #####accuracy:1.0
	epoch:4, #### loss:[0.10598478] #####accuracy:1.0
	epoch:5, #### loss:[0.10482856] #####accuracy:1.0
	epoch:6, #### loss:[0.10453037] #####accuracy:1.0
	epoch:7, #### loss:[0.10445572] #####accuracy:1.0
	epoch:8, #### loss:[0.10442699] #####accuracy:1.0
	epoch:9, #### loss:[0.10442187] #####accuracy:1.0
	epoch:10, #### loss:[0.10442089] #####accuracy:1.0
	epoch:11, #### loss:[0.10442062] #####accuracy:1.0
	epoch:12, #### loss:[0.10442055] #####accuracy:1.0
	epoch:13, #### loss:[0.10442055] #####accuracy:1.0
	epoch:14, #### loss:[0.10442055] #####accuracy:1.0
	epoch:15, #### loss:[0.10442055] #####accuracy:1.0
	epoch:16, #### loss:[0.10442055] #####accuracy:1.0

	start testing..............
	[0.3132616580]
	test:--------------->loss:QTensor(0.3132616580, requires_grad=True) #####accuracy:1.0


Modelltraining mit NoiseQuantumLayer in VQNet
==============================================================================

Verwendung von ``NoiseQuantumLayer`` zum Erstellen und Trainieren von verrauschten Quantenschaltungen mit der Rausch-VM von pyQPanda2.

Ein Beispiel fuer ein vollstaendiges verrauschtes Quantenmaschinelles-Lernen-Modell ist wie folgt:

.. code-block::

    import os
    import numpy as np

    from pyvqnet.nn.module import Module
    from pyvqnet.nn.linear import Linear
    from pyvqnet.nn.conv import Conv2D

    from pyvqnet.nn import activation as F
    from pyvqnet.nn.pooling import MaxPool2D
    from pyvqnet.nn.loss import CategoricalCrossEntropy
    from pyvqnet.optim.adam import Adam
    from pyvqnet.data.data import data_generator
    from pyvqnet.tensor import tensor

    import pyqpanda as pq
    from pyqpanda import *
    from pyvqnet.qnn.quantumlayer import NoiseQuantumLayer
    import matplotlib
    try:
        matplotlib.use('TkAgg')
    except:
        pass
    import time
    try:
        matplotlib.use('TkAgg')
    except:
        pass

    try:
        import urllib.request
    except ImportError:
        raise ImportError('You should use Python 3.x')
    import os.path
    import gzip

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

    #Verwendung von qpanda zur Erstellung von Quantenschaltungen
    def circuit(weights,param,qubits,cbits,machine):

        circuit = pq.QCircuit()
        circuit.insert(pq.H(qubits[0]))
        circuit.insert(pq.RY(qubits[0], weights[0]))
        prog = pq.QProg()
        prog.insert(circuit)
        prog << measure_all(qubits, cbits)
        result = machine.run_with_configuration(prog, cbits, 100)
        counts = np.array(list(result.values()))
        states = np.array(list(result.keys())).astype(float)
        # Berechnung der Wahrscheinlichkeiten fuer jeden Zustand
        probabilities = counts / 100
        # Erwartungswert des Zustands ermitteln
        expectation = np.sum(states * probabilities)
        return expectation



    class Net(Module):
        def __init__(self):
            super(Net, self).__init__()
            self.conv1 = Conv2D(input_channels=1, output_channels=6, kernel_size=(5, 5), stride=(1, 1), padding="valid")
            self.maxpool1 = MaxPool2D([2, 2], [2, 2], padding="valid")
            self.conv2 = Conv2D(input_channels=6, output_channels=16, kernel_size=(5, 5), stride=(1, 1), padding="valid")
            self.maxpool2 = MaxPool2D([2, 2], [2, 2], padding="valid")
            self.fc1 = Linear(input_channels=256, output_channels=64)
            self.fc2 = Linear(input_channels=64, output_channels=1)

            self.hybrid = NoiseQuantumLayer(circuit,1,"noise",1)
            self.fc3 = Linear(input_channels=1, output_channels=2)


        def forward(self, x):
            x = F.ReLu()(self.conv1(x))
            x = self.maxpool1(x)
            x = F.ReLu()(self.conv2(x))
            x = self.maxpool2(x)
            x = tensor.flatten(x, 1)
            x = F.ReLu()(self.fc1(x))
            x = self.fc2(x)
            x = self.hybrid(x)
            x = self.fc3(x)

            return x

Das Modell ist ein hybrides Quantenschaltungs- und klassisches Netzwerkmodell, wobei der Quantenschaltungsteil
``NoiseQuantumLayer`` verwendet, um die Quantenschaltung plus Rauschmodell zu simulieren. Dieses Modell
wird zur Klassifikation der handgeschriebenen Ziffern 0 und 1 in der MNIST-Datenbank verwendet.

.. code-block::

    def load_mnist(dataset="training_data", digits=np.arange(2), path="./"):         
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
        _, size = struct.unpack(">II", flbl.read(8))
        lbl = pyarray("b", flbl.read())
        flbl.close()

        fimg = open(fname_image, 'rb')
        _, size, rows, cols = struct.unpack(">IIII", fimg.read(16))
        img = pyarray("B", fimg.read())
        fimg.close()

        ind = [k for k in range(size) if lbl[k] in digits]
        N = len(ind)
        images = np.zeros((N, rows, cols))
        labels = np.zeros((N, 1), dtype=int)
        for i in range(len(ind)):
            images[i] = np.array(img[ind[i] * rows * cols: (ind[i] + 1) * rows * cols]).reshape((rows, cols))
            labels[i] = lbl[ind[i]]

        return images, labels

    def data_select(train_num, test_num):
        x_train, y_train = load_mnist("training_data")  
        x_test, y_test = load_mnist("testing_data")
        idx_train = np.append(np.where(y_train == 0)[0][:train_num],
                        np.where(y_train == 1)[0][:train_num])

        x_train = x_train[idx_train]
        y_train = y_train[idx_train]
        
        x_train = x_train / 255
        y_train = np.eye(2)[y_train].reshape(-1, 2)

        # Test: Behalten nur Labels 0 und 1
        idx_test = np.append(np.where(y_test == 0)[0][:test_num],
                        np.where(y_test == 1)[0][:test_num])

        x_test = x_test[idx_test]
        y_test = y_test[idx_test]
        x_test = x_test / 255
        y_test = np.eye(2)[y_test].reshape(-1, 2)
        
        return x_train, y_train, x_test, y_test

    if __name__=="__main__":
        x_train, y_train, x_test, y_test = data_select(100, 50)
        
        model = Net()
        optimizer = Adam(model.parameters(), lr=0.005)
        loss_func = CategoricalCrossEntropy()

        epochs = 10
        loss_list = []
        eval_loss_list = []
        train_acc_list = []
        eval_acc_list = []
        model.train()
        if not os.path.exists("./result"):
            os.makedirs("./result")
        else:
            pass
        eval_time = []
        F1 = open("./result/hqcnn_train_rlt.txt","w")
        F2 = open("./result/hqcnn_eval_rlt.txt","w")
        for epoch in range(1, epochs):
            total_loss = []
            iter  = 0
            correct = 0
            n_train = 0
            for x, y in data_generator(x_train, y_train, batch_size=1, shuffle=True):
                iter +=1
                start_time = time.time()
                x = x.reshape(-1, 1, 28, 28)
                optimizer.zero_grad()
                # Forward pass
                output = model(x)
                # Calculating loss
                loss = loss_func(y, output) 
                loss_np = np.array(loss.data)
                np_output = np.array(output.data, copy=False)
                mask = (np_output.argmax(1) == y.argmax(1))
                correct += np.sum(np.array(mask))
                n_train += 1
                
                # Backward pass
                loss.backward()
                # Optimize the weights
                optimizer._step()
                total_loss.append(loss_np)
            print("##########################")
            print(f"Train Accuracy: {correct / n_train}")
            loss_list.append(np.sum(total_loss) / len(total_loss))
            train_acc_list.append(correct/n_train)
            print("epoch: ", epoch)
            print(100. * (epoch + 1) / epochs)
            print("{:.0f} loss is : {:.10f}".format(epoch, loss_list[-1]))
            F1.writelines(f"{epoch},{loss_list[-1]},{correct/n_train}\n")

            model.eval()
            correct = 0
            total_eval_loss = []
            n_eval = 0
            
            for x, y in data_generator(x_test, y_test, batch_size=1, shuffle=True):
                start_time1 = time.time()
                x = x.reshape(-1, 1, 28, 28)
                output = model(x)
                loss = loss_func(y, output)

                np_output = np.array(output.data, copy=False)
                mask = (np_output.argmax(1) == y.argmax(1))
                correct += np.sum(np.array(mask))
                n_eval += 1
                
                loss_np = np.array(loss.data)
                total_eval_loss.append(loss_np)
                eval_acc_list.append(correct/n_eval)
            print(f"Eval Accuracy: {correct / n_eval}")
            F2.writelines(f"{epoch},{np.sum(total_eval_loss) / len(total_eval_loss)},{correct/n_eval}\n")
        F1.close()
        F2.close()
		

Vergleich der Klassifikationsergebnisse von Modellen des maschinellen Lernens mit verrauschten Quantenschaltungen und idealen Quantenschaltungen.
Das Verlust-Aenderungsprotokoll und das Genauigkeits-Aenderungsprotokoll sind wie folgt:

.. code-block::

    Train Accuracy: 0.715
    epoch:  1
    1 loss is : 0.6519572449
    Eval Accuracy: 0.99
    ##########################
    Train Accuracy: 1.0
    epoch:  2
    2 loss is : 0.4458528900
    Eval Accuracy: 1.0
    ##########################
    Train Accuracy: 1.0     
    epoch:  3
    3 loss is : 0.3142367172
    Eval Accuracy: 1.0
    ##########################
    Train Accuracy: 1.0     
    epoch:  4
    4 loss is : 0.2259583092
    Eval Accuracy: 1.0
    ##########################
    Train Accuracy: 1.0     
    epoch:  5
    5 loss is : 0.1661866951
    Eval Accuracy: 1.0
    ##########################
    Train Accuracy: 1.0     
    epoch:  6
    6 loss is : 0.1306252861
    Eval Accuracy: 1.0
    ##########################
    Train Accuracy: 1.0
    epoch:  7
    7 loss is : 0.0996847820
    Eval Accuracy: 1.0
    ##########################
    Train Accuracy: 1.0
    epoch:  8
    8 loss is : 0.0801456261
    Eval Accuracy: 1.0
    ##########################
    Train Accuracy: 1.0
    epoch:  9
    9 loss is : 0.0649107647
    Eval Accuracy: 1.0

|
