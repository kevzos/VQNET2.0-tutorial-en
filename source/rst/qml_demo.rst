Aprendizado de M\u00e1quina Qu\u00e2ntico Usando Qpanda
#######################################################

Usamos VQNet e pyqpanda2 ou pyqpanda3 para implementar v\u00e1rios exemplos de aprendizado de m\u00e1quina qu\u00e2ntico.

.. warning::

    A parte de computa\u00e7\u00e3o qu\u00e2ntica da interface a seguir pode usar pyqpanda2.

    Voc\u00ea precisa instalar pyqpanda adicionalmente, `pip install pyqpanda`


Aplica\u00e7\u00e3o de Circuito Qu\u00e2ntico Parametrizado em Tarefa de Classifica\u00e7\u00e3o
*****************************************************************************************************

1. Demonstra\u00e7\u00e3o QVC
========================================

Este exemplo usa VQNet para implementar o algoritmo da tese: `Circuit-centric quantum classifiers <https://arxiv.org/pdf/1804.00633.pdf>`_  .
Este exemplo \u00e9 usado para determinar se um n\u00famero bin\u00e1rio \u00e9 \u00edmpar ou par. Codificando o n\u00famero bin\u00e1rio no qubit e otimizando os par\u00e2metros vari\u00e1veis no circuito, 
a observa\u00e7\u00e3o na dire\u00e7\u00e3o z do circuito pode indicar se a entrada \u00e9 \u00edmpar ou par.

Circuito qu\u00e2ntico
-------------------------------
O subcircuito de componente vari\u00e1vel geralmente define um subcircuito, que \u00e9 uma arquitetura b\u00e1sica de circuito, e circuitos variacionais complexos podem ser constru\u00eddos repetindo camadas.
Nossa camada de circuito consiste em m\u00faltiplas portas l\u00f3gicas qu\u00e2nticas de rota\u00e7\u00e3o e portas l\u00f3gicas qu\u00e2nticas ``CNOT`` que entrela\u00e7am cada qubit com seus qubits vizinhos.
Tamb\u00e9m precisamos de um circuito para codificar dados cl\u00e1ssicos em um estado qu\u00e2ntico, para que a sa\u00edda da medi\u00e7\u00e3o do circuito esteja relacionada \u00e0 entrada.
Neste exemplo, codificamos a entrada bin\u00e1ria nos qubits na ordem correspondente. Por exemplo, o dado de entrada 1101 \u00e9 codificado em 4 qubits.

.. math::

    x = 0101 \rightarrow|\psi\rangle=|0101\rangle

.. figure:: ./images/qvc_circuit.png
   :width: 600 px
   :align: center

|

Este exemplo usa pyqpanda3.

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


Constru\u00e7\u00e3o do modelo
------------------------------
Definimos circuitos qu\u00e2nticos vari\u00e1veis ``qvc_circuits`` .
Esperamos us\u00e1-los no framework de diferencia\u00e7\u00e3o autom\u00e1tica do VQNet,
para aproveitar as fun\u00e7\u00f5es de otimiza\u00e7\u00e3o do VQNet para treinamento do modelo.
Definimos uma classe Model, que herda da classe abstrata ``Module``.
O modelo usa a classe ``pyvqnet.qnn.pq3.QuantumLayer``, que \u00e9 uma camada de computa\u00e7\u00e3o qu\u00e2ntica que pode ser diferenciada automaticamente.
``qvc_circuits`` \u00e9 o circuito qu\u00e2ntico que queremos executar,
24 \u00e9 o n\u00famero de todos os par\u00e2metros do circuito qu\u00e2ntico que precisam ser treinados.

.. code-block::

     class Model(Module):
        def __init__(self):
            super(Model, self).__init__()
            self.qvc = QuantumLayer(qvc_circuits,24)

        def forward(self, x):
            return self.qvc(x)


Treinamento e teste do modelo
------------------------------------
Usamos n\u00fameros bin\u00e1rios aleat\u00f3rios pr\u00e9-gerados e seus r\u00f3tulos de \u00edmpar e par.
Os dados s\u00e3o os seguintes.

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

O encaminhamento do modelo, c\u00e1lculo da fun\u00e7\u00e3o de perda,
c\u00e1lculo reverso e c\u00e1lculo do otimizador podem ser realizados como o modo geral 
de treinamento de rede neural, at\u00e9 que o n\u00famero de itera\u00e7\u00f5es atinja o valor predefinido.
Os dados de treinamento usados s\u00e3o gerados acima, os dados de teste s\u00e3o qvc_test_data e os dados de treinamento s\u00e3o qvc_train_data.

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

A figura a seguir ilustra a curva de precis\u00e3o do modelo:

.. figure:: ./images/qvc_accuracy.png
   :width: 600 px
   :align: center

|

2. algoritmo de reenvio de dados
=======================================
Em uma rede neural, cada neur\u00f4nio recebe informa\u00e7\u00f5es de todos os neur\u00f4nios da camada superior (Figura a). 
Em contraste, o classificador qu\u00e2ntico de \u00fanico bit aceita a unidade de processamento de informa\u00e7\u00e3o anterior e a entrada (Figura b).
Para circuitos qu\u00e2nticos tradicionais, quando os dados s\u00e3o enviados, o resultado pode ser obtido diretamente atrav\u00e9s de v\u00e1rias transforma\u00e7\u00f5es 
unit\u00e1rias :math:`U(\theta_1,\theta_2,\theta_3)`. No entanto, na tarefa de Reenvio de Dados Qu\u00e2nticos (QDRL), os dados precisam ser reenviados antes de cada transforma\u00e7\u00e3o unit\u00e1ria.

                                                                .. centered:: Compara\u00e7\u00e3o dos esquemas QDRL e rede neural cl\u00e1ssica

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
        \"\"\"
        Fun\u00e7\u00e3o principal para treinar o modelo qdrl
        \"\"\"
        batch_size = 5
        model.train()
        x_train, y_train = circle(500)
        x_train = np.hstack((x_train, np.ones((x_train.shape[0], 1))))  # 500*3

        epoch = 10
        print("start training...........")
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
        print("start eval...................")
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

A figura a seguir ilustra a curva de precis\u00e3o do modelo: 

.. figure:: ./images/qdrl_accuracy.png
   :width: 600 px
   :align: center

|

3. VSQL: Variational Shadow Quantum Learning for Classification Model (VSQL: Aprendizado Qu\u00e2ntico Sombra Variacional para Modelo de Classifica\u00e7\u00e3o)
=================================================================================================================================================================
Usando circuitos qu\u00e2nticos vari\u00e1veis para construir um modelo de classifica\u00e7\u00e3o bin\u00e1ria, 
comparando a precis\u00e3o de classifica\u00e7\u00e3o com uma rede neural com precis\u00e3o de par\u00e2metros similar, 
a precis\u00e3o dos dois \u00e9 semelhante. A quantidade de par\u00e2metros dos circuitos qu\u00e2nticos \u00e9 muito menor que a das redes neurais cl\u00e1ssicas.
O algoritmo \u00e9 baseado no artigo: `Variational Shadow Quantum Learning for Classification Model <https://arxiv.org/abs/2012.08288>`_ para 
reproduzir.

A figura a seguir mostra a arquitetura do algoritmo VSQL:

.. figure:: ./images/vsql_model.PNG
   :width: 600 px
   :align: center

|

As figuras a seguir mostram a estrutura dos circuitos qu\u00e2nticos locais em cada qubit:

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
        \"\"\"
        Fun\u00e7\u00e3o de download para arquivo do dataset mnist
        \"\"\"
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
        \"\"\"
        Modelo VSQL de circuitos qu\u00e2nticos
        \"\"\"
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


    #GLOBAL VAR
    n = 10
    n_qsc = 2
    depth = 1


    class QModel(Module):
        """
        Model of VSQL
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
        \"\"\"
        carregar dados mnist
        \"\"\"
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
        VQSL MODEL
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

            # Evaluation
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

O seguinte mostra a curva de precis\u00e3o e perda do modelo: 

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

4. Quanvolution para classifica\u00e7\u00e3o de imagens
=======================================================================================================================

Neste exemplo, implementamos uma Rede Neural Convolucional Qu\u00e2ntica, um m\u00e9todo originalmente introduzido no artigo `Quanvolutional Neural Networks: Powering Image Recognition with Quantum Circuits <https://arxiv.org/abs/1904.04767>`_ .

Similar \u00e0 convolu\u00e7\u00e3o cl\u00e1ssica, a Quanvolution tem os seguintes passos:
Uma pequena regi\u00e3o da imagem de entrada, no nosso caso um quadrado 2\u00d72 de dados cl\u00e1ssicos, \u00e9 incorporada ao circuito qu\u00e2ntico.
Neste exemplo, isso \u00e9 alcan\u00e7ado aplicando portas l\u00f3gicas de rota\u00e7\u00e3o parametrizadas a qubits inicializados no estado fundamental. O kernel convolucional aqui gera circuitos variacionais a partir de circuitos estoc\u00e1sticos propostos na refer\u00eancia.
Finalmente, o sistema qu\u00e2ntico \u00e9 medido para obter uma lista de valores esperados cl\u00e1ssicos.
Similar a uma camada convolucional cl\u00e1ssica, cada valor esperado \u00e9 mapeado para um canal diferente de um \u00fanico pixel de sa\u00edda.
Repetindo o mesmo processo em diferentes regi\u00f5es, a imagem de entrada completa pode ser escaneada, produzindo um objeto de sa\u00edda que ser\u00e1 constru\u00eddo como uma imagem multicanal.
Para realizar tarefas de classifica\u00e7\u00e3o, este exemplo usa a camada totalmente conectada cl\u00e1ssica ``Linear`` para realizar tarefas de classifica\u00e7\u00e3o ap\u00f3s a Quanvolution obter os valores de medi\u00e7\u00e3o.
A principal diferen\u00e7a da convolu\u00e7\u00e3o cl\u00e1ssica \u00e9 que a Quanvolution pode gerar kernels altamente complexos cujo c\u00e1lculo \u00e9, ao menos em princ\u00edpio, classicamente intrat\u00e1vel.

.. image:: ./images/quanvo.png
   :width: 600 px
   :align: center

|

Defini\u00e7\u00e3o do dataset MNIST

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
        \"\"\"
        Fun\u00e7\u00e3o de download para arquivo do dataset mnist
        \"\"\"
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
        \"\"\"
        carregar dados mnist
        \"\"\"
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

Defini\u00e7\u00e3o do m\u00f3dulo e defini\u00e7\u00e3o da fun\u00e7\u00e3o de processo

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

            # Evaluation
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

Perda do conjunto de treinamento, perda do conjunto de valida\u00e7\u00e3o, precis\u00e3o de classifica\u00e7\u00e3o do conjunto de treinamento e valida\u00e7\u00e3o com a transforma\u00e7\u00e3o das \u00e9pocas.

.. code-block::

    # epoch train_loss      train_accuracy eval_loss    eval_accuracy
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


Demonstra\u00e7\u00e3o do Quantum AutoEncoder
*********************************************

1. Quantum AutoEncoder
=======================================

O autoencoder cl\u00e1ssico \u00e9 uma rede neural que pode aprender representa\u00e7\u00f5es de baixa dimens\u00e3o de alta efici\u00eancia de dados em um espa\u00e7o de alta dimens\u00e3o. 
A tarefa do autoencoder \u00e9 mapear x para um ponto de baixa dimens\u00e3o y dada uma entrada x, de modo que x possa ser recuperado a partir de y.
A estrutura da rede autoencoder subjacente pode ser selecionada para representar os dados em uma dimens\u00e3o menor, comprimindo efetivamente a entrada. 
Inspirado por esta ideia, o modelo de autoencoder qu\u00e2ntico \u00e9 usado para realizar tarefas semelhantes em dados qu\u00e2nticos.
Autoencoders qu\u00e2nticos s\u00e3o treinados para comprimir conjuntos de dados espec\u00edficos de estados qu\u00e2nticos, e algoritmos de compress\u00e3o cl\u00e1ssicos n\u00e3o podem ser usados. 
Os par\u00e2metros do autoencoder qu\u00e2ntico s\u00e3o treinados usando algoritmos de otimiza\u00e7\u00e3o cl\u00e1ssicos.
Mostramos um exemplo de um circuito program\u00e1vel simples, que pode ser treinado como um autoencoder eficiente. 
Aplicamos nosso modelo no contexto de simula\u00e7\u00e3o qu\u00e2ntica para comprimir o modelo Hubbard e o estado fundamental do Hamiltoniano.
Este algoritmo \u00e9 baseado em `Quantum autoencoders for efficient compression of quantum data <https://arxiv.org/pdf/1612.02806.pdf>`_ .


Circuitos qu\u00e2nticos QAE:

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
        ##load dataset

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
            for x, y in data_generator(x_train, y_train, batch_size=batch_size, shuffle=True): #shuffle batch rather than data

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


            # Evaluation
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

O valor de erro QAE obtido executando o c\u00f3digo acima. A perda \u00e9 1/fidelidade, tendendo a 1 significa que a fidelidade est\u00e1 pr\u00f3xima de 1.

.. figure:: ./images/qae_train_loss.png
   :width: 600 px
   :align: center

|

Demonstra\u00e7\u00e3o de Aprendizado de Estrutura de Circuitos Qu\u00e2nticos
******************************************************************************

1. Aprendizado de estrutura de circuitos qu\u00e2nticos
===============================================================================

Na estrutura do circuito qu\u00e2ntico, as portas qu\u00e2nticas com par\u00e2metros mais frequentemente usadas s\u00e3o as portas `RZ`, `RY` e `RX`, mas qual porta usar em quais circunst\u00e2ncias \u00e9 uma quest\u00e3o que vale a pena estudar. Um m\u00e9todo \u00e9 a sele\u00e7\u00e3o aleat\u00f3ria, mas neste caso \u00e9 muito prov\u00e1vel que os melhores resultados n\u00e3o sejam alcan\u00e7ados.
O objetivo central da tarefa de aprendizado de estrutura de circuitos qu\u00e2nticos \u00e9 encontrar a combina\u00e7\u00e3o \u00f3tima de portas qu\u00e2nticas com par\u00e2metros.
A abordagem aqui \u00e9 que este conjunto de portas l\u00f3gicas qu\u00e2nticas \u00f3timas deve fazer com que a fun\u00e7\u00e3o de perda seja minimizada.


.. code-block::

    \"\"\"
    Demonstra\u00e7\u00e3o de Aprendizado de Estrutura de Circuitos Qu\u00e2nticos

    \"\"\"
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
        \"\"\"
        implementa\u00e7\u00e3o do algoritmo rotosolve
        \"\"\"
        params[d] = np.pi / 2.0
        m0_plus = cost(QTensor(params), generators)
        params[d] = -np.pi / 2.0
        m0_minus = cost(QTensor(params), generators)
        a = np.arctan2(2.0 * M_0 - m0_plus - m0_minus,
                    m0_plus - m0_minus)  # returns value in (-pi,pi]
        params[d] = -np.pi / 2.0 - a
        if params[d] <= -np.pi:
            params[d] += 2 * np.pi
        return cost(QTensor(params), generators)


    def optimal_theta_and_gen_helper(index, params, generators):
        \"\"\"
        encontrar vari\u00e1veis \u00f3timas
        \"\"\"
        params[index] = 0.
        m0 = loss(QTensor(params), generators)  #init value
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

O circuito qu\u00e2ntico obtido executando o c\u00f3digo acima cont\u00e9m :math:`RX`, um :math:`RY`

.. figure:: ./images/final_quantum_circuit.png
   :width: 600 px
   :align: center

|

E com os par\u00e2metros na porta qu\u00e2ntica :math:`\theta_1`, :math:`\theta_2` mudando, a fun\u00e7\u00e3o de perda tem valores diferentes.

.. figure:: ./images/loss3d.png
   :width: 600 px
   :align: center

|

Demonstra\u00e7\u00e3o de Rede Neural Cl\u00e1ssica Qu\u00e2ntica H\u00edbrida
******************************************************************************

1. Modelo de Rede Neural Cl\u00e1ssica Qu\u00e2ntica H\u00edbrida
==============================================================================

Aprendizado de m\u00e1quina (ML) tornou-se um campo interdisciplinar de sucesso que visa extrair informa\u00e7\u00f5es generaliz\u00e1veis dos dados matematicamente. 
O aprendizado de m\u00e1quina qu\u00e2ntico busca usar os princ\u00edpios da mec\u00e2nica qu\u00e2ntica para melhorar o aprendizado de m\u00e1quina, e vice-versa.
Se seu objetivo \u00e9 melhorar algoritmos cl\u00e1ssicos de ML terceirizando c\u00e1lculos dif\u00edceis para computadores qu\u00e2nticos, 
ou usar arquiteturas cl\u00e1ssicas de ML para otimizar algoritmos qu\u00e2nticos ambos se enquadram na categoria de aprendizado de m\u00e1quina qu\u00e2ntico (QML).
Neste cap\u00edtulo, exploraremos como quantizar parcialmente redes neurais cl\u00e1ssicas para criar redes neurais qu\u00e2nticas cl\u00e1ssicas h\u00edbridas. 
Circuitos qu\u00e2nticos s\u00e3o compostos por portas l\u00f3gicas qu\u00e2nticas, e os c\u00e1lculos qu\u00e2nticos implementados por 
essas portas l\u00f3gicas s\u00e3o comprovadamente diferenci\u00e1veis pelo artigo `Quantum Circuit Learning <https://arxiv.org/abs/1803.00745>`_. 
Portanto, pesquisadores tentam colocar circuitos qu\u00e2nticos e m\u00f3dulos de rede neural cl\u00e1ssica juntos para treinamento em tarefas de aprendizado de m\u00e1quina qu\u00e2ntico cl\u00e1ssico h\u00edbrido.
Escreveremos um exemplo simples para implementar uma tarefa de treinamento de modelo de rede neural usando VQNet. 
O prop\u00f3sito deste exemplo \u00e9 demonstrar a simplicidade do VQNet e encorajar profissionais de ML a explorar as possibilidades da computa\u00e7\u00e3o qu\u00e2ntica.


Prepara\u00e7\u00e3o dos Dados
------------------------------

Usaremos os `datasets MNIST <https://ossci-datasets.s3.amazonaws.com/mnist/>`_, o banco de dados de d\u00edgitos manuscritos de rede neural mais b\u00e1sico como dados de classifica\u00e7\u00e3o.
Primeiro carregamos MNIST e filtramos as amostras de dados contendo 0 e 1.
Essas amostras s\u00e3o divididas em dados de treinamento training_data e dados de teste testing_data, cada um com dimens\u00e3o 1*784.

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
        # Train Leaving only labels 0 and 1
        idx_train = np.append(np.where(y_train == 0)[0][:train_num],
                        np.where(y_train == 1)[0][:train_num])
        x_train = x_train[idx_train]
        y_train = y_train[idx_train]
        x_train = x_train / 255
        y_train = np.eye(2)[y_train].reshape(-1, 2)
        # Test Leaving only labels 0 and 1
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

Construir Circuitos Qu\u00e2nticos
----------------------------------

Neste exemplo, usamos pyQPanda2, um circuito qu\u00e2ntico simples de 1 qubit \u00e9 definido. O circuito recebe a sa\u00edda da camada de rede neural cl\u00e1ssica como entrada, codifica dados qu\u00e2nticos atrav\u00e9s das portas l\u00f3gicas qu\u00e2nticas ``H``, ``RY`` e calcula o valor esperado do Hamiltoniano na dire\u00e7\u00e3o z como sa\u00edda.

.. code-block::

    from pyqpanda import *
    import pyqpanda as pq
    import numpy as np
    def circuit(x ,weights):
        num_qubits = 1
        #Use pyQPanda2 para criar um simulador 
        machine = pq.CPUQVM()
        machine.init_qvm()
        #Use pyQPanda2 para alocar qubits
        qubits = machine.qAlloc_many(num_qubits)
        #Use pyQPanda2 para alocar bits cl\u00e1ssicos
        cbits = machine.cAlloc_many(num_qubits)
        #Construir circuitos
        circuit = pq.QCircuit()
        circuit.insert(pq.H(qubits[0]))
        circuit.insert(pq.RY(qubits[0], x[0]))
        #Construir programa qu\u00e2ntico
        prog = pq.QProg()
        prog.insert(circuit)
        #Define medi\u00e7\u00e3o
        prog << measure_all(qubits, cbits)
        shots = 1000
        #executa qu\u00e2ntico com medi\u00e7\u00f5es qu\u00e2nticas
        result = machine.run_with_configuration(prog, cbits,shots)
        machine.finalize()
        # Pad both outcomes: all shots could collapse to a single basis state
        count0 = result.get("0", 0)
        count1 = result.get("1", 0)
        expectation = (count0 - count1) / shots   # ⟨Z⟩ = P(0) - P(1)
        return expectation
        

.. figure:: ./images/hqcnn_quantum_cir.png
   :width: 600 px
   :align: center

|

Criar Modelo H\u00edbrido
--------------------------

Como os circuitos qu\u00e2nticos podem realizar c\u00e1lculos de diferencia\u00e7\u00e3o autom\u00e1tica juntamente com redes neurais cl\u00e1ssicas,
podemos usar a camada convolucional ``Conv2D`` do VQNet, a camada de pooling ``MaxPool2D``, a camada totalmente conectada ``Linear`` e
o circuito qu\u00e2ntico para construir o modelo agora.
A defini\u00e7\u00e3o das classes `Net` e `Hybrid` herdam do m\u00f3dulo de diferencia\u00e7\u00e3o autom\u00e1tica do VQNet ``Module`` 
e a defini\u00e7\u00e3o do c\u00e1lculo forward \u00e9 definida na fun\u00e7\u00e3o forward ``forward()``,
Um Modelo de diferencia\u00e7\u00e3o autom\u00e1tica de convolu\u00e7\u00e3o, codifica\u00e7\u00e3o qu\u00e2ntica e medi\u00e7\u00e3o dos dados MNIST \u00e9 constru\u00eddo para obter as caracter\u00edsticas finais necess\u00e1rias para a tarefa de classifica\u00e7\u00e3o.

.. code-block::

    #Camada de computa\u00e7\u00e3o qu\u00e2ntica forward e defini\u00e7\u00e3o da fun\u00e7\u00e3o de c\u00e1lculo do gradiente, que precisa herdar da classe abstrata Module
    from pyvqnet.qnn import QuantumLayerV2
    #Model definition
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

Treinamento e teste
-----------------------

Para o modelo de rede neural h\u00edbrida como mostrado na figura abaixo, calculamos a fun\u00e7\u00e3o de perda alimentando dados no modelo iterativamente, 
e o VQNet calcular\u00e1 o gradiente de cada par\u00e2metro no c\u00e1lculo reverso automaticamente, 
e usar\u00e1 o otimizador para otimizar os par\u00e2metros at\u00e9 que o n\u00famero de itera\u00e7\u00f5es atenda ao valor predefinido.

.. figure:: ./images/hqcnnarch.PNG
   :width: 600 px
   :align: center

|

.. code-block::

    x_train, y_train, x_test, y_test = data_select(1000, 100)

    #Criar um modelo
    model = Net() 
    #Use otimizador adam
    optimizer = Adam(model.parameters(), lr=0.005)
    #Use entropia cruzada como fun\u00e7\u00e3o de perda
    loss_func = CategoricalCrossEntropy()

    #train epoches   
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

Visualiza\u00e7\u00e3o
----------------------

A curva de visualiza\u00e7\u00e3o da fun\u00e7\u00e3o de perda e precis\u00e3o nos dados de treinamento e teste.

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

2. Modelo de aprendizado por transfer\u00eancia qu\u00e2ntico cl\u00e1ssico h\u00edbrido
=======================================================================================================================
Aplicamos um m\u00e9todo de aprendizado de m\u00e1quina chamado aprendizado por transfer\u00eancia em classificador de imagens baseado em rede cl\u00e1ssica qu\u00e2ntica
h\u00edbrida. Escreveremos um exemplo simples de integra\u00e7\u00e3o do pyQPanda2 com o VQNet. O aprendizado por transfer\u00eancia \u00e9 baseado na intui\u00e7\u00e3o geral
de que, se a rede pr\u00e9-treinada \u00e9 boa em resolver um determinado problema, ela tamb\u00e9m pode ser usada para resolver um problema diferente,
mas relacionado, com apenas algum treinamento adicional.

Diagrama parcial do circuito qu\u00e2ntico ilustrado abaixo:

.. figure:: ./images/QTransferLearning_cir.png
   :width: 600 px
   :align: center

|

.. code-block::

    \"\"\"
    Demonstra\u00e7\u00e3o de aprendizado por transfer\u00eancia em rede neural cl\u00e1ssica qu\u00e2ntica

    \"\"\"

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
    # CNN cl\u00e1ssica
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


    \"\"\"
    para obter par\u00e2metros do modelo CNN para aprendizado por transfer\u00eancia
    \"\"\"

    train_size = 10000
    eval_size = 1000
    EPOCHES = 100
    def classcal_cnn_model_making():
        # load train data
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
                # Forward pass
                output = model(x)

                # Calculating loss
                loss = loss_func(y, output)  # target output
                loss_np = np.array(loss.data)
                # Backward pass
                loss.backward()
                # Optimize the weights
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

        n_qubits = 4  # N\u00famero de qubits
        q_depth = 6  # Profundidade do circuito qu\u00e2ntico (n\u00famero de camadas variacionais)

        def Q_H_layer(qubits, nqubits):
            """Camada de portas Hadamard de \u00fanico qubit.
            """
            circuit = pq.QCircuit()
            for idx in range(nqubits):
                circuit.insert(pq.H(qubits[idx]))
            return circuit

        def Q_RY_layer(qubits, w):
            """Camada de rota\u00e7\u00f5es parametrizadas de qubit em torno do eixo y.
            """
            circuit = pq.QCircuit()
            for idx, element in enumerate(w):
                circuit.insert(pq.RY(qubits[idx], element))
            return circuit

        def Q_entangling_layer(qubits, nqubits):
            """Camada de CNOTs seguida por outra camada deslocada de CNOT.
            """
            # Em outras palavras, deve aplicar algo como :
            # CNOT  CNOT  CNOT  CNOT...  CNOT
            #   CNOT  CNOT  CNOT...  CNOT
            circuit = pq.QCircuit()
            for i in range(0, nqubits - 1, 2):  # Loop over even indices: i=0,2,...N-2
                circuit.insert(pq.CNOT(qubits[i], qubits[i + 1]))
            for i in range(1, nqubits - 1, 2):  # Loop over odd indices:  i=1,3,...N-3
                circuit.insert(pq.CNOT(qubits[i], qubits[i + 1]))
            return circuit

        def Q_quantum_net(q_input_features, q_weights_flat, qubits, cbits, machine):
            \"\"\"
            O circuito qu\u00e2ntico variacional.
            \"\"\"
            machine = pq.CPUQVM()
            machine.init_qvm()
            qubits = machine.qAlloc_many(n_qubits)
            circuit = pq.QCircuit()

            # Redimensionar pesos
            q_weights = q_weights_flat.reshape([q_depth, n_qubits])

            # Come\u00e7ar do estado |+> , imparcial em rela\u00e7\u00e3o a |0> e |1>
            circuit.insert(Q_H_layer(qubits, n_qubits))

            # Incorporar caracter\u00edsticas no n\u00f3 qu\u00e2ntico
            circuit.insert(Q_RY_layer(qubits, q_input_features))

            # Sequ\u00eancia de camadas variacionais trein\u00e1veis
            for k in range(q_depth):
                circuit.insert(Q_entangling_layer(qubits, n_qubits))
                circuit.insert(Q_RY_layer(qubits, q_weights[k]))

            # Valores esperados na base Z
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
                \"\"\"
                Defini\u00e7\u00e3o da arquitetura *dressed*.
                \"\"\"

                super().__init__()
                self.pre_net = Linear(128, n_qubits)
                self.post_net = Linear(n_qubits, 10)
                self.temp_Q = QuantumLayer(Q_quantum_net, q_depth * n_qubits, "cpu", n_qubits, n_qubits)

            def forward(self, input_features):
                \"\"\"
                Define como os tensores devem se mover atrav\u00e9s da rede qu\u00e2ntica
                *dressed*.
                \"\"\"

                # obter as caracter\u00edsticas de entrada para o circuito qu\u00e2ntico
                # reduzindo a dimens\u00e3o da caracter\u00edstica de 512 para 4
                pre_out = self.pre_net(input_features)
                q_in = tensor.tanh(pre_out) * np.pi / 2.0
                q_out_elem = self.temp_Q(q_in)

                result = q_out_elem
                # retornar a previs\u00e3o bidimensional da camada de p\u00f3s-processamento
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
                # Forward pass
                output = model_hybrid(x)

                loss = loss_func(y, output)  # target output
                loss_np = np.array(loss.data)
                # Backward pass
                loss.backward()
                # Optimize the weights
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

        n_qubits = 4  # N\u00famero de qubits
        q_depth = 6  # Profundidade do circuito qu\u00e2ntico (n\u00famero de camadas variacionais)

        def Q_H_layer(qubits, nqubits):
            """Camada de portas Hadamard de \u00fanico qubit.
            """
            circuit = pq.QCircuit()
            for idx in range(nqubits):
                circuit.insert(pq.H(qubits[idx]))
            return circuit

        def Q_RY_layer(qubits, w):
            """Camada de rota\u00e7\u00f5es parametrizadas de qubit em torno do eixo y.
            """
            circuit = pq.QCircuit()
            for idx, element in enumerate(w):
                circuit.insert(pq.RY(qubits[idx], element))
            return circuit

        def Q_entangling_layer(qubits, nqubits):
            """Camada de CNOTs seguida por outra camada deslocada de CNOT.
            """
            # Em outras palavras, deve aplicar algo como :
            # CNOT  CNOT  CNOT  CNOT...  CNOT
            #   CNOT  CNOT  CNOT...  CNOT
            circuit = pq.QCircuit()
            for i in range(0, nqubits - 1, 2):  # Loop over even indices: i=0,2,...N-2
                circuit.insert(pq.CNOT(qubits[i], qubits[i + 1]))
            for i in range(1, nqubits - 1, 2):  # Loop over odd indices:  i=1,3,...N-3
                circuit.insert(pq.CNOT(qubits[i], qubits[i + 1]))
            return circuit

        def Q_quantum_net(q_input_features, q_weights_flat, qubits, cbits, machine):
            \"\"\"
            O circuito qu\u00e2ntico variacional.
            \"\"\"
            machine = pq.CPUQVM()
            machine.init_qvm()
            qubits = machine.qAlloc_many(n_qubits)
            circuit = pq.QCircuit()

            # Redimensionar pesos
            q_weights = q_weights_flat.reshape([q_depth, n_qubits])

            # Come\u00e7ar do estado |+>, imparcial em rela\u00e7\u00e3o a |0> e |1>
            circuit.insert(Q_H_layer(qubits, n_qubits))

            # Incorporar caracter\u00edsticas no n\u00f3 qu\u00e2ntico
            circuit.insert(Q_RY_layer(qubits, q_input_features))

            # Sequ\u00eancia de camadas variacionais trein\u00e1veis
            for k in range(q_depth):
                circuit.insert(Q_entangling_layer(qubits, n_qubits))
                circuit.insert(Q_RY_layer(qubits, q_weights[k]))

            # Valores esperados na base Z
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
                \"\"\"
                Defini\u00e7\u00e3o da arquitetura *dressed*.
                \"\"\"

                super().__init__()
                self.pre_net = Linear(128, n_qubits)
                self.post_net = Linear(n_qubits, 10)
                self.temp_Q = QuantumLayer(Q_quantum_net, q_depth * n_qubits, "cpu", n_qubits, n_qubits)

            def forward(self, input_features):
                \"\"\"
                Define como os tensores devem se mover atrav\u00e9s da rede qu\u00e2ntica
                *dressed*.
                \"\"\"

                # obter as caracter\u00edsticas de entrada para o circuito qu\u00e2ntico
                # reduzindo a dimens\u00e3o da caracter\u00edstica de 512 para 4
                pre_out = self.pre_net(input_features)
                q_in = tensor.tanh(pre_out) * np.pi / 2.0
                q_out_elem = self.temp_Q(q_in)

                result = q_out_elem
                # retornar a previs\u00e3o bidimensional da camada de p\u00f3s-processamento
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
        # salvar par\u00e2metros do modelo cl\u00e1ssico
        if not os.path.exists('./result/QCNN_TL_1.model'):
            classcal_cnn_model_making()
            classical_cnn_TransferLearning_predict()
        #treinar circuitos qu\u00e2nticos.
        print("usar par\u00e2metros do modelo CNN existente para treinar par\u00e2metros qu\u00e2nticos.")
        quantum_cnn_TransferLearning()
        #avaliar circuitos qu\u00e2nticos.
        quantum_cnn_TransferLearning_predict()


Perda no conjunto de treinamento

.. figure:: ./images/qcnn_transfer_learning_classical.png
   :width: 600 px
   :align: center

|

Executar classifica\u00e7\u00e3o no conjunto de teste

.. figure:: ./images/qcnn_transfer_learning_predict.png
   :width: 600 px
   :align: center

|

3. Modelo de rede neural Unet qu\u00e2ntico cl\u00e1ssico h\u00edbrido
==============================================================================

Segmenta\u00e7\u00e3o de imagens \u00e9 um problema cl\u00e1ssico na pesquisa de vis\u00e3o computacional e se tornou um ponto
quente no campo de entendimento de imagens. A segmenta\u00e7\u00e3o de imagens \u00e9 uma parte importante do entendimento de imagens, e um dos problemas mais dif\u00edceis no processamento de imagens.
A chamada segmenta\u00e7\u00e3o de imagens refere-se \u00e0 segmenta\u00e7\u00e3o baseada em cinza, cor e textura espacial. A imagem
\u00e9 dividida em v\u00e1rias regi\u00f5es disjuntas por caracter\u00edsticas como teoria e geometria, de modo que essas caracter\u00edsticas mostrem
consist\u00eancia ou similaridade na mesma regi\u00e3o e diferen\u00e7as \u00f3bvias entre diferentes regi\u00f5es. Em suma,
\u00e9 dar uma imagem e classificar cada pixel na imagem. Separar as regi\u00f5es de pixel pertencentes
a diferentes objetos. `Unet <https://arxiv.org/abs/1505.04597>`_ \u00e9 um algoritmo cl\u00e1ssico de segmenta\u00e7\u00e3o de imagens.

Aqui, exploramos como quantizar parcialmente a rede neural cl\u00e1ssica para criar uma rede neural
`QUnet` qu\u00e2ntica cl\u00e1ssica h\u00edbrida. Escreveremos um exemplo simples de integra\u00e7\u00e3o do pyQPanda2 com o VQNet.
QUnet \u00e9 usado principalmente para resolver a t\u00e9cnica de segmenta\u00e7\u00e3o de imagens.



Prepara\u00e7\u00e3o dos dados
------------------------------

Usaremos os dados da biblioteca oficial `VOC2012 <http://host.robots.ox.ac.uk/pascal/VOC/voc2012/#devkit>`_ como dados de segmenta\u00e7\u00e3o de imagens. Essas amostras s\u00e3o divididas
em dados de treinamento training_data e dados de teste testing_data.

.. figure:: ./images/Unet_data_imshow.png
   :width: 600 px
   :align: center

|

Construindo circuitos qu\u00e2nticos
-------------------------------------------
Neste exemplo, definimos um circuito qu\u00e2ntico usando pyqpanda2. Os dados da imagem colorida de
3 canais de entrada s\u00e3o comprimidos em uma imagem cinza de \u00fanico canal e armazenados, e ent\u00e3o a caracter\u00edstica dos dados \u00e9
extra\u00edda e a dimensionalidade reduzida pela opera\u00e7\u00e3o de convolu\u00e7\u00e3o qu\u00e2ntica.


.. figure:: ./images/qunet_cir.png
   :width: 600 px
   :align: center

|

Importar bibliotecas e fun\u00e7\u00f5es necess\u00e1rias

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

Pr\u00e9-processamento de dados

.. code-block::

    # Dados de pr\u00e9-processamento
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

    # Circuito de codifica\u00e7\u00e3o qu\u00e2ntica
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
        \"\"\"Convolui a imagem de entrada com m\u00faltiplas aplica\u00e7\u00f5es do mesmo circuito qu\u00e2ntico.\"\"\"
        out = np.zeros((64, 64, 1))
        
        for j in range(0, 128, 2):
            for k in range(0, 128, 2):
                # Process a squared 2x2 region of the image with a quantum circuit
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

Construindo rede neural qu\u00e2ntica cl\u00e1ssica h\u00edbrida
----------------------------------------------------------------

De acordo com o framework de rede Unet, usamos o framework `VQNet` para construir a parte cl\u00e1ssica da rede.
A camada de rede neural de down-sampling \u00e9 usada para reduzir a dimens\u00e3o e extrair caracter\u00edsticas;
A camada de rede neural de up-sampling \u00e9 usada para restaurar a dimens\u00e3o; As camadas de up e down sampling
s\u00e3o conectadas atrav\u00e9s de concatenate para fus\u00e3o de caracter\u00edsticas.


.. figure:: ./images/Unet.png
   :width: 600 px
   :align: center

|

.. code-block::

    # Definition of down sampling neural network layer
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
            \"\"\"
            :param x:
            :return: out(Sa\u00edda para profundo), out_2(entrada para pr\u00f3ximo n\u00edvel),
            \"\"\"
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

    # Definition of up sampling neural network layer
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
            \'\'\'
            :param x: camada de entrada conv
            :param out: conectar com UpsampleLayer
            :return:
            \'\'\'
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

    # Unet overall network architecture
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
            # output
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

Treinamento e salvamento do modelo
----------------------------------

Similar ao treinamento de modelo de rede neural cl\u00e1ssica,
tamb\u00e9m precisamos instanciar o modelo, definir a fun\u00e7\u00e3o de perda e o otimizador, e definir todo o processo de treinamento e
teste. Para o modelo de rede neural h\u00edbrida como mostrado na figura abaixo, calculamos o valor da perda na
fun\u00e7\u00e3o forward, o gradiente de cada par\u00e2metro no
c\u00e1lculo reverso automaticamente, e usamos o otimizador para otimizar os par\u00e2metros at\u00e9 que o n\u00famero de
itera\u00e7\u00f5es atenda ao valor predefinido. Se ``PREPROCESS`` for False, o c\u00f3digo pular\u00e1 o pr\u00e9-processamento qu\u00e2ntico de dados.

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

    # preparar dados/r\u00f3tulos de treino e teste
    path0 = 'training_data'
    path1 = 'testing_data'
    train_images, train_labels = PreprocessingData(path0).read()
    test_images, test_labels = PreprocessingData(path1).read()

    print('train: ', train_images.shape, '\ntest: ', test_images.shape)
    print('train: ', train_labels.shape, '\ntest: ', test_labels.shape)
    train_images = train_images / 255
    test_images = test_images / 255

    # usar codificador qu\u00e2ntico para pr\u00e9-processar dados

    if PREPROCESS == True:
        print("Quantum pre-processing of train images:")
        q_train_images = quantum_data_preprocessing(train_images)
        q_test_images = quantum_data_preprocessing(test_images)
        q_train_label = quantum_data_preprocessing(train_labels)
        q_test_label = quantum_data_preprocessing(test_labels)

        # Save pre-processed images
        print('Quantum Data Saving...')
        np.save("./result/q_train.npy", q_train_images)
        np.save("./result/q_test.npy", q_test_images)
        np.save("./result/q_train_label.npy", q_train_label)
        np.save("./result/q_test_label.npy", q_test_label)
        print('Quantum Data Saving Over!')

    # loading quantum data
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
            loss = loss_func(y_img_Qtensor, img_out)  # target output
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


Visualiza\u00e7\u00e3o de dados

A curva da fun\u00e7\u00e3o de perda dos dados de treinamento \u00e9 exibida e salva, e os resultados dos dados de teste s\u00e3o salvos.

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

Perda no conjunto de treinamento

.. figure:: ./images/qunet_train_loss.png
   :width: 600 px
   :align: center

|

Executar classifica\u00e7\u00e3o no conjunto de teste

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


4. Modelo de rede neural QMLP qu\u00e2ntico-cl\u00e1ssico h\u00edbrido
===============================================================================

Apresentamos e analisamos uma arquitetura proposta de perceptron multicamadas qu\u00e2ntico (QMLP) com embeddings de entrada tolerantes a falhas, n\u00e3o linearidades ricas e simula\u00e7\u00f5es de circuitos variacionais aprimorados com portas de emaranhamento de dois qubits parametrizadas.
`QMLP: An Error-Tolerant Nonlinear Quantum MLP Architecture using Parameterized Two-Qubit Gates <https://arxiv.org/pdf/2206.01345.pdf>`_ .
Escreveremos um exemplo simples de integra\u00e7\u00e3o do pyQPanda2 com o VQNet.


Construindo Redes Neurais Qu\u00e2ntico-Cl\u00e1ssicas H\u00edbridas
--------------------------------------------------------------------

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
    except:  # pylint:disable=bare-except
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
        \"\"\"
        Baixar dados mnist se necess\u00e1rio.
        \"\"\"
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
        \"\"\"
        carregar dados mnist
        \"\"\"
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
        \"\"\"
        Selecionar dados do dataset mnist.
        \"\"\"
        x_train, y_train = load_mnist("training_data")  
        x_test, y_test = load_mnist("testing_data")  
        idx_train = np.append(
            np.where(y_train == 0)[0][:train_num],
            np.where(y_train == 1)[0][:train_num])

        x_train = x_train[idx_train]
        y_train = y_train[idx_train]

        x_train = x_train / 255
        y_train = np.eye(2)[y_train].reshape(-1, 2)

        # Test Leaving only labels 0 and 1
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
        # print(prog)
        # exit()

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
        # x_train, y_train, x_test, y_test = data_select(1000, 100)

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
            # train_acc_list.append(correct / n_train)
            print("epoch: ", epoch)
            # print(100. * (epoch + 1) / epochs)
            print("{:.0f} loss is : {:.10f}".format(epoch, loss_list[-1]))

    if __name__ == "__main__":

        vqnet_test_QMLPModel()



resultado dos dados
--------------------

A curva da fun\u00e7\u00e3o de perda dos dados de treinamento \u00e9 exibida e salva, e os resultados dos dados de teste s\u00e3o salvos.
Situa\u00e7\u00e3o da perda no conjunto de treinamento.

.. image:: ./images/QMLP.png
   :width: 600 px
   :align: center

|


.. _QDRL_DEMO:

5. Modelo de rede neural QDRL qu\u00e2ntico-cl\u00e1ssico h\u00edbrido
===============================================================================

Apresentamos e analisamos uma proposta de rede de aprendizado por refor\u00e7o qu\u00e2ntico (QDRL), cujas caracter\u00edsticas reformulam algoritmos cl\u00e1ssicos de aprendizado por refor\u00e7o profundo, como replay de experi\u00eancia e redes alvo, em representa\u00e7\u00f5es de circuitos qu\u00e2nticos variacionais.
Al\u00e9m disso, usamos um esquema de codifica\u00e7\u00e3o de informa\u00e7\u00e3o qu\u00e2ntica para reduzir o n\u00famero de par\u00e2metros do modelo em compara\u00e7\u00e3o com redes neurais cl\u00e1ssicas. `QDRL: Variational Quantum Circuits for Deep Reinforcement Learning <https://arxiv.org/pdf/1907.00397.pdf>`_.
Escreveremos um exemplo simples de integra\u00e7\u00e3o do pyQPanda2 com o VQNet.


Construindo Redes Neurais Qu\u00e2ntico-Cl\u00e1ssicas H\u00edbridas
--------------------------------------------------------------------

Requer ``gym`` == 0.23.0 , ``pygame`` == 2.1.2 .

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
    except:  # pylint:disable=bare-except
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
        # Bloco de emaranhamento
        cir.insert(pq.CNOT(qubits[0], qubits[1]))
        cir.insert(pq.CNOT(qubits[1], qubits[2]))
        cir.insert(pq.CNOT(qubits[2], qubits[3]))
        # u3 gate
        cir.insert(RotCircuit(weights[0], qubits[0]))  # weights shape = [4, 3]
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
        # parameter number = 24
        weights = weights.reshape([2, 4, 3])
        # layer wise
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
                # print('Optimizing...')
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
            # update parameters in target circuit
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



Aprendizado n\u00e3o supervisionado
***********************************

1. Quantum Kmeans
=======================================

1.1 Introdu\u00e7\u00e3o
------------------------

O algoritmo de agrupamento \u00e9 um algoritmo t\u00edpico de aprendizado n\u00e3o supervisionado, que \u00e9 usado principalmente para classificar automaticamente amostras semelhantes em uma classe. No algoritmo de agrupamento, as amostras s\u00e3o divididas em diferentes categorias de acordo com a similaridade entre as amostras. Para diferentes m\u00e9todos de c\u00e1lculo de similaridade, diferentes resultados de agrupamento ser\u00e3o obtidos. O m\u00e9todo de c\u00e1lculo de similaridade comum \u00e9 o m\u00e9todo da dist\u00e2ncia euclidiana. O que queremos mostrar \u00e9 o algoritmo k-means qu\u00e2ntico. O algoritmo K-means \u00e9 um algoritmo de agrupamento baseado em dist\u00e2ncia. Ele usa a dist\u00e2ncia como o \u00edndice de avalia\u00e7\u00e3o de similaridade, ou seja, quanto mais pr\u00f3xima a dist\u00e2ncia entre dois objetos, maior a similaridade. O algoritmo considera que clusters s\u00e3o compostos por objetos pr\u00f3ximos uns dos outros, portanto clusters compactos e independentes s\u00e3o o objetivo final.

O modelo de aprendizado de m\u00e1quina qu\u00e2ntico kmeans qu\u00e2ntico tamb\u00e9m pode ser desenvolvido no VQNet. Um exemplo da tarefa de agrupamento kmeans qu\u00e2ntico \u00e9 fornecido abaixo. Atrav\u00e9s do circuito qu\u00e2ntico, podemos construir uma medi\u00e7\u00e3o que \u00e9 positivamente correlacionada com a dist\u00e2ncia euclidiana das vari\u00e1veis de aprendizado de m\u00e1quina cl\u00e1ssico, de modo a alcan\u00e7ar o objetivo de encontrar o vizinho mais pr\u00f3ximo.


1.2 Introdu\u00e7\u00e3o ao princ\u00edpio do algoritmo
-------------------------------------------------------

A implementa\u00e7\u00e3o do algoritmo k-means qu\u00e2ntico usa principalmente swap test para comparar a dist\u00e2ncia entre pontos de dados de entrada. Selecione aleatoriamente k pontos de N pontos de dados como centr\u00f3ides, me\u00e7a a dist\u00e2ncia de cada ponto a cada centr\u00f3ide, atribua ao centr\u00f3ide de classe mais pr\u00f3ximo, recalcule o centr\u00f3ide de cada classe e itere 2 a 3 passos at\u00e9 que o novo centr\u00f3ide seja igual ou menor que o limite especificado. Em nosso exemplo, selecionamos 100 pontos de dados e 2 centr\u00f3ides, e usamos o circuito cswap para calcular a dist\u00e2ncia.
Finalmente, obtivemos dois clusters de pontos de dados. :math:`|0\rangle` \u00e9 um bit auxiliar, atrav\u00e9s da porta l\u00f3gica H, o qubit se tornar\u00e1 :math:`\frac{1}{\sqrt{2}}(|0\rangle + |1\rangle)`. Sob o controle do qubit :math:`|1\rangle`,
O circuito qu\u00e2ntico ir\u00e1 inverter :math:`|x\rangle` e :math:`|y\rangle​` . Finalmente obtemos o resultado:

.. math::

    |0_{anc}\rangle |x\rangle |y\rangle \rightarrow \frac{1}{2}|0_{anc}\rangle(|xy\rangle + |yx\rangle) + \frac{1}{2}|1_{anc}\rangle(|xy\rangle - |yx\rangle)

Se medirmos o qubit auxiliar separadamente, a probabilidade do estado final do estado fundamental :math:`|1\rangle` \u00e9:

.. math::

    P(|1_{anc}\rangle) = \frac{1}{2} - \frac{1}{2}|\langle x | y \rangle|^2

A dist\u00e2ncia euclidiana entre dois estados qu\u00e2nticos \u00e9 a seguinte:

.. math::

    Euclidean \ distance = \sqrt{(2 - 2|\langle x | y \rangle|)}

Visivelmente, medir o qubit :math:`|1\rangle` \u00e9 positivamente correlacionado com a dist\u00e2ncia euclidiana. O circuito qu\u00e2ntico deste algoritmo \u00e9 o seguinte:

.. figure:: ./images/Kmeans.jpg
   :width: 600 px
   :align: center

|

1.3 Implementa\u00e7\u00e3o VQNet
---------------------------------

1.3.1 Preparo do ambiente
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

O ambiente adota Python 3.8. Recomenda-se usar CONDA para configura\u00e7\u00e3o do ambiente. Ele vem com numpy, SciPy, Matplotlib, sklearn e outros kits de ferramentas para f\u00e1cil uso. Se o ambiente python for adotado, pacotes relevantes precisam ser instalados, e o seguinte ambiente pyvqnet precisa ser preparado


1.3.2 Preparo dos dados
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Os dados s\u00e3o gerados aleatoriamente por make_blobs no SciPy, e a fun\u00e7\u00e3o \u00e9 definida para gerar dados de distribui\u00e7\u00e3o gaussiana.


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

1.3.3 Circuito qu\u00e2ntico
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Construindo circuitos qu\u00e2nticos usando VQNet

.. code-block::

    # O \u00e2ngulo de rota\u00e7\u00e3o da porta qu\u00e2ntica de entrada \u00e9 calculado de acordo com o ponto de coordenada de entrada d (x, y)
    def get_theta(d):
        x = d[0]
        y = d[1]
        theta = 2 * math.acos((x.item() + y.item()) / 2.0)
        return theta

    # O circuito qu\u00e2ntico \u00e9 constru\u00eddo de acordo com os pontos de dados qu\u00e2nticos de entrada
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

1.3.4 Visualiza\u00e7\u00e3o de dados
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

C\u00e1lculo visual dos dados de agrupamento relevantes

.. code-block::

    # Visualiza\u00e7\u00e3o de pontos dispersos e centr\u00f3ides dos clusters
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

1.3.5 C\u00e1lculo do agrupamento
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Calcular o centr\u00f3ide do cluster dos dados de agrupamento relevantes

.. code-block::

    # Gerar pontos centrais do cluster aleatoriamente
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
        n = 100  # number of data points
        k = 3  # Number of centers
        std = 2  # std of datapoints

        points, o_centers = get_data(n, k, std)  # dataset

        points = preprocess(points)  # Normalize dataset

        centroids = initialize_centers(points, k)  # Intialize centroids

        epoch = 9
        points = QTensor(points)
        centroids = QTensor(centroids)
        plt.figure()
        draw_plot(points.data, o_centers,label=False)

        for i in range(epoch):
                centers = find_nearest_neighbour(points, centroids)  # find nearest centers
                centroids = find_centroids(points, centers)  # find centroids

        plt.figure()
        draw_plot(points.data, centers.data)

    # Run program entry
    if __name__ == "__main__":
        qkmean_run()


1.3.6 Distribui\u00e7\u00e3o dos dados antes do agrupamento
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. figure:: ./images/ep_1.png
   :width: 600 px
   :align: center

|

1.3.7 Distribui\u00e7\u00e3o dos dados ap\u00f3s o agrupamento
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. figure:: ./images/ep_9.png
   :width: 600 px
   :align: center

|

Pesquisa em Aprendizado de M\u00e1quina Qu\u00e2ntico
*****************************************************

Modelos qu\u00e2nticos como s\u00e9ries de Fourier
================================================================================

Computadores qu\u00e2nticos podem ser usados para aprendizado supervisionado tratando circuitos qu\u00e2nticos parametrizados como modelos que mapeiam dados
de entrada para previs\u00f5es. Embora muito trabalho tenha sido feito para investigar as implica\u00e7\u00f5es pr\u00e1ticas desta abordagem, muitas propriedades
te\u00f3ricas importantes desses modelos permanecem desconhecidas. Aqui investigamos como a estrat\u00e9gia com a qual os dados s\u00e3o codificados no
modelo influencia o poder expressivo de circuitos qu\u00e2nticos parametrizados como aproximadores de fun\u00e7\u00e3o.


O artigo `The effect of data encoding on the expressive power of variational quantum machine learning models <https://arxiv.org/pdf/2008.08605.pdf>`_ relaciona modelos comuns de aprendizado de m\u00e1quina qu\u00e2ntico projetados para computadores qu\u00e2nticos de curto prazo a s\u00e9ries de Fourier


1.1 Ajuste de s\u00e9rie de Fourier com codifica\u00e7\u00e3o serial de rota\u00e7\u00e3o Pauli
-----------------------------------------------------------------------------------------------

Primeiro mostramos como modelos qu\u00e2nticos que usam rota\u00e7\u00f5es Pauli como portas de codifica\u00e7\u00e3o de dados s\u00f3 podem ajustar s\u00e9ries de Fourier at\u00e9 um certo grau. Para simplificar, veremos apenas circuitos de \u00fanico qubit:

.. image:: ./images/single_qubit_model.png
   :width: 600 px
   :align: center

|

Fazer dados de entrada, definir modelos qu\u00e2nticos seriais e n\u00e3o realizar resultados de treinamento do modelo.

.. code-block::

    \"""
    S\u00e9rie de Fourier Qu\u00e2ntica
    \"\"\"
    import numpy as np
    import pyqpanda as pq
    from pyvqnet.qnn.measure import expval
    import matplotlib.pyplot as plt
    import matplotlib
    try:
        matplotlib.use(\"TkAgg\")
    except:  # pylint:disable=bare-except
        print(\"Can not use matplot TkAgg\")
        pass

    np.random.seed(42)


    degree = 1  # grau da fun\u00e7\u00e3o alvo
    scaling = 1  # escala dos dados
    coeffs = [0.15 + 0.15j]*degree  # coeficientes de frequ\u00eancias n\u00e3o nulas
    coeff0 = 0.1  # coeficiente de frequ\u00eancia zero

    def target_function(x):
        \"\"\"Gerar uma s\u00e9rie de Fourier truncada, onde os dados s\u00e3o reescalados.\"\"\"
        res = coeff0
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
        machine = pq.CPUQVM()  # outside
        machine.init_qvm()  # outside
        qubits = machine.qAlloc_many(num_qubits)

        for theta in weights[:-1]:
            cir.insert(W(theta, qubits))
            cir.insert(S(scaling, x, qubits))

        # (L+1)'th unitary
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
    weights = 2 * np.pi * np.random.random(size=(r+1, 3))  # some random initial weights

    x = np.linspace(-6, 6, 70)
    random_quantum_model_y = [serial_quantum_model(weights, x_, 1, 1) for x_ in x]

    plt.plot(x, target_y, c='black', label="true")
    plt.scatter(x, target_y, facecolor='white', edgecolor='black')
    plt.plot(x, random_quantum_model_y, c='blue', label="predict")
    plt.ylim(-1, 1)
    plt.legend(loc="upper right")
    plt.show()


O resultado da execu\u00e7\u00e3o do circuito qu\u00e2ntico sem treinamento \u00e9:

.. image:: ./images/single_qubit_model_result_no_train.png
   :width: 600 px
   :align: center

|


Fazer os dados de entrada, definir o modelo qu\u00e2ntico serial e construir o modelo de treinamento em combina\u00e7\u00e3o com a QuantumLayer do framework VQNet.

.. code-block::

    \"""
    S\u00e9rie de Fourier Qu\u00e2ntica Serial
    \"""
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
        matplotlib.use(\"TkAgg\")
    except:  # pylint:disable=bare-except
        print(\"Can not use matplot TkAgg\")
        pass

    np.random.seed(42)

    degree = 1  # grau da fun\u00e7\u00e3o alvo
    scaling = 1  # escala dos dados
    coeffs = [0.15 + 0.15j]*degree  # coeficientes de frequ\u00eancias n\u00e3o nulas
    coeff0 = 0.1  # coeficiente de frequ\u00eancia zero

    def target_function(x):
        \"\"\"Gerar uma s\u00e9rie de Fourier truncada, onde os dados s\u00e3o reescalados.\"\"\"
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
    weights = 2 * np.pi * np.random.random(size=(r+1, 3))  # some random initial weights

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
                # Select batch of data
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

O modelo qu\u00e2ntico \u00e9:

.. image:: ./images/single_qubit_model_circuit.png
   :width: 600 px
   :align: center

|

Os resultados do treinamento da rede s\u00e3o:

.. image:: ./images/single_qubit_model_result.png
   :width: 600 px
   :align: center

|

A perda do treinamento da rede \u00e9:

.. code-block::

    start training..............
    epoch:0, #### loss:0.04852807720773853
    epoch:1, #### loss:0.012945819365559146
    epoch:2, #### loss:0.0009359727291666786
    epoch:3, #### loss:0.00015995280153333625
    epoch:4, #### loss:3.988249877352246e-05


1.2 Ajuste de s\u00e9rie de Fourier com codifica\u00e7\u00e3o paralela de rota\u00e7\u00e3o Pauli
-------------------------------------------------------------------------------------------------

Como mostrado no artigo, esperamos resultados semelhantes ao modelo serial: uma s\u00e9rie de Fourier de ordem r s\u00f3 pode ser ajustada se a porta codificada tiver pelo menos r repeti\u00e7\u00f5es no modelo qu\u00e2ntico. Circuito qu\u00e2ntico:

.. image:: ./images/parallel_model.png
   :width: 600 px
   :align: center

|

Fazer dados de entrada, definir modelos qu\u00e2nticos paralelos e n\u00e3o realizar resultados de treinamento do modelo.

.. code-block::

    \"""
    S\u00e9rie de Fourier Qu\u00e2ntica
    \"""
    import numpy as np
    import pyqpanda as pq
    from pyvqnet.qnn.measure import expval
    import matplotlib.pyplot as plt
    import matplotlib
    try:
        matplotlib.use(\"TkAgg\")
    except:  # pylint:disable=bare-except
        print(\"Can not use matplot TkAgg\")
        pass

    np.random.seed(42)

    degree = 1  # grau da fun\u00e7\u00e3o alvo
    scaling = 1  # escala dos dados
    coeffs = [0.15 + 0.15j] * degree  # coeficientes de frequ\u00eancias n\u00e3o nulas
    coeff0 = 0.1  # coeficiente de frequ\u00eancia zero

    def target_function(x):
        \"\"\"Gerar uma s\u00e9rie de Fourier truncada, onde os dados s\u00e3o reescalados.\"\"\"
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
        machine = pq.CPUQVM()  # outside
        machine.init_qvm()  # outside
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
    # print(weights)
    x = np.linspace(-6, 6, 70)
    random_quantum_model_y = [parallel_quantum_model(weights, xx, r) for xx in x]

    plt.plot(x, target_y, c='black', label="true")
    plt.scatter(x, target_y, facecolor='white', edgecolor='black')
    plt.plot(x, random_quantum_model_y, c='blue', label="predict")
    plt.ylim(-1, 1)
    plt.legend(loc="upper right")
    plt.show()

O resultado da execu\u00e7\u00e3o do circuito qu\u00e2ntico sem treinamento \u00e9:

.. image:: ./images/parallel_model_result_no_train.png
   :width: 600 px
   :align: center

|


Fazer os dados de entrada, definir o modelo qu\u00e2ntico paralelo e construir o modelo de treinamento em combina\u00e7\u00e3o com a camada QuantumLayer do framework VQNet.

.. code-block::

    \"""
    S\u00e9rie de Fourier Qu\u00e2ntica
    \"\"\"
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
        matplotlib.use(\"TkAgg\")
    except:  # pylint:disable=bare-except
        print(\"Can not use matplot TkAgg\")
        pass

    np.random.seed(42)

    degree = 1  # grau da fun\u00e7\u00e3o alvo
    scaling = 1  # escala dos dados
    coeffs = [0.15 + 0.15j] * degree  # coeficientes de frequ\u00eancias n\u00e3o nulas
    coeff0 = 0.1  # coeficiente de frequ\u00eancia zero

    def target_function(x):
        \"\"\"Gerar uma s\u00e9rie de Fourier truncada, onde os dados s\u00e3o reescalados.\"\"\"
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
                # Select batch of data
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


O modelo qu\u00e2ntico \u00e9:

.. image:: ./images/parallel_model_circuit.png
   :width: 600 px
   :align: center

|

Os resultados do treinamento da rede s\u00e3o:

.. image:: ./images/parallel_model_result.png
   :width: 600 px
   :align: center

|

A perda do treinamento da rede \u00e9:

.. code-block::

    start training..............
    epoch:0, #### loss:0.0037272341538482578
    epoch:1, #### loss:5.271130586635309e-05
    epoch:2, #### loss:4.714951917250687e-07
    epoch:3, #### loss:1.0968826371082763e-08
    epoch:4, #### loss:2.1258629738507562e-10

Poder expressivo de circuitos qu\u00e2nticos
===============================================================================

No artigo `Expressibility and entangling capability of parameterized quantum circuits for hybrid quantum-classical algorithms <https://arxiv.org/abs/1905.10876>`_,
os autores prop\u00f5em um m\u00e9todo para quantifica\u00e7\u00e3o da expressividade baseado na distribui\u00e7\u00e3o de probabilidade de fidelidade entre estados de sa\u00edda da rede neural.
Para qualquer rede neural qu\u00e2ntica :math:`U(\vec{\theta})`, amostre os par\u00e2metros da rede neural duas vezes (definidos como :math:`\vec{\phi}` e :math:`\vec{\psi }` ),
Ent\u00e3o a fidelidade entre os estados de sa\u00edda de dois circuitos qu\u00e2nticos :math:`F=|\langle0|U(\vec{\phi})^\dagger U(\vec{\psi})|0\rangle|^ 2` obedece a uma distribui\u00e7\u00e3o de probabilidade:

.. math::

    F\sim{P}(f)

A literatura aponta que quando a rede neural qu\u00e2ntica :math:`U` pode ser uniformemente distribu\u00edda em todas as matrizes unit\u00e1rias (neste caso, \u00e9 chamada de :math:`U` obedece \u00e0 distribui\u00e7\u00e3o Haar), a distribui\u00e7\u00e3o de probabilidade da fidelidade :math:`P_\text{Haar }(f)` satisfaz:

.. math::

    P_\text{Haar}(f)=(2^{n}-1)(1-f)^{2^n-2}

A diverg\u00eancia K-L (tamb\u00e9m conhecida como entropia relativa) em matem\u00e1tica estat\u00edstica mede a diferen\u00e7a entre duas distribui\u00e7\u00f5es de probabilidade. A diverg\u00eancia K-L entre duas distribui\u00e7\u00f5es de probabilidade discretas :math:`P,Q` \u00e9 definida como:

.. math::

    D_{KL}(P||Q)=\sum_jP(j)\ln\frac{P(j)}{Q(j)}

Se a distribui\u00e7\u00e3o de fidelidade da sa\u00edda da rede neural qu\u00e2ntica for registrada como :math:`P_\text{QNN}(f)`, ent\u00e3o a capacidade expressiva da rede neural qu\u00e2ntica \u00e9 definida como a diverg\u00eancia K-L entre :math:`P_\text{QNN}(f) ` e :math:`P_\text{Haar}(f)`:

.. math::

    \text{Expr}_\text{QNN}=D_{KL}(P_\text{QNN}(f)||P_\text{Haar}(f))

Portanto, quando :math:`P_\text{QNN}(f)` estiver mais pr\u00f3ximo de :math:`P_\text{Haar}(f)`, :math:`\text{Expr}` ser\u00e1 menor (mais tendendo a 0),
A capacidade expressiva da rede neural qu\u00e2ntica tamb\u00e9m ser\u00e1 mais forte; inversamente, quanto maior :math:`\text{Expr}`, mais fraca ser\u00e1 a capacidade expressiva da rede neural qu\u00e2ntica.
Podemos calcular diretamente a expressividade de redes neurais qu\u00e2nticas de \u00fanico bit de acordo com esta defini\u00e7\u00e3o :math:`R_Y(\theta)` , :math:`R_Y(\theta_1)R_Z(\theta_2)` e
:math:`R_Y(\theta_1)R_Z(\theta_2)R_Y(\theta_3)`:

O seguinte usa VQNet para demonstrar as capacidades de express\u00e3o de circuito qu\u00e2ntico de `HardwareEfficientAnsatz <https://arxiv.org/abs/1704.05018>`_ em diferentes profundidades (1, 2, 3).


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
    num_qubit = 1  # o n\u00famero de qubits
    num_sample = 2000  # o n\u00famero de amostras
    outputs_y = list()  # salvar sa\u00eddas QNN
    # plotar histograma
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
        plt.ylabel("frequency")
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

    # Definir largura do circuito e profundidade m\u00e1xima
    num_qubit = 4
    max_depth = 3
    # Calcular a distribui\u00e7\u00e3o de fidelidade correspondente \u00e0 amostragem Haar

    flist, p_haar, theory_haar = fidelity_harr_sample(num_qubit, num_sample)
    title_str = "haar, %d qubit(s)" % num_qubit
    plot_hist(flist, 50, title_str)
    Expr_cel = list()
    # Calcular o poder expressivo de redes neurais com diferentes profundidades
    for DEPTH in range(1, max_depth + 1):
        print("Amostrando circuito na profundidade %d..." % DEPTH)
        f_list, p_cel = fidelity_of_cir(HardwareEfficientAnsatz, num_qubit, DEPTH,
                                        num_sample)
        title_str = f"HardwareEfficientAnsatz, {num_qubit} qubit(s) {DEPTH} layer(s)"
        plot_hist(f_list, 50, title_str)
        expr = entropy(p_cel, theory_haar)
        Expr_cel.append(expr)
    # Comparar o poder expressivo de redes neurais de diferentes profundidades
    print(
        f"O poder expressivo de redes neurais com profundidades 1, 2 e 3 \u00e9 { np.around(Expr_cel, decimals=4)}, quanto menor melhor.", )
    plt.plot(range(1, max_depth + 1), Expr_cel, marker='>')
    plt.xlabel("depth")
    plt.yscale('log')
    plt.ylabel("Expr.")
    plt.xticks(range(1, max_depth + 1))
    plt.title("Expressibility vs Circuit Depth")
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



Perceptron Qu\u00e2ntico
=======================================

Redes neurais artificiais s\u00e3o o cora\u00e7\u00e3o dos algoritmos de aprendizado de m\u00e1quina e protocolos de intelig\u00eancia artificial. Historicamente, a implementa\u00e7\u00e3o mais simples de um neur\u00f4nio artificial remonta ao perceptron cl\u00e1ssico de Rosenblatt, mas suas aplica\u00e7\u00f5es pr\u00e1ticas de longo prazo podem ser prejudicadas pelo r\u00e1pido aumento da complexidade computacional, especialmente relevante para o treinamento de redes de perceptron multicamadas.
Aqui nos referimos ao artigo `An Artificial Neuron Implemented on an Actual Quantum Processor <https://arxiv.org/abs/1811.02266>`__ que introduz um algoritmo baseado em informa\u00e7\u00e3o qu\u00e2ntica implementando a vers\u00e3o de computador qu\u00e2ntico de um perceptron, que mostra vantagem exponencial em recursos de codifica\u00e7\u00e3o sobre realiza\u00e7\u00f5es alternativas.

Para este perceptron qu\u00e2ntico, os dados processados s\u00e3o uma sequ\u00eancia de bits bin\u00e1rios 0 1. O objetivo \u00e9 identificar padr\u00f5es que t\u00eam a forma de um w cruzado como mostrado na figura abaixo.

.. image:: ./images/QP-data.png
   :width: 600 px
   :align: center

|

\u00c9 codificado usando uma sequ\u00eancia de bits bin\u00e1ria, onde preto \u00e9 0 e branco \u00e9 1, de modo que w \u00e9 codificado como (1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 0, 0, 1, 1, 0, 1). Um total de 16 sequ\u00eancias de bits pode ser codificado no sinal da amplitude do estado qu\u00e2ntico de 4 bits. O sinal \u00e9 0 para n\u00fameros negativos e 1 para n\u00fameros positivos. Atrav\u00e9s do m\u00e9todo de codifica\u00e7\u00e3o acima, nossa entrada de algoritmo \u00e9 convertida em uma sequ\u00eancia bin\u00e1ria de 16 bits. Essas sequ\u00eancias bin\u00e1rias n\u00e3o repetitivas podem corresponder respectivamente a uma entrada espec\u00edfica :math:`U_i` .

A estrutura do circuito do perceptron qu\u00e2ntico proposta neste artigo \u00e9 a seguinte:

.. image:: ./images/QP-cir.png
   :width: 600 px
   :align: center

|

O circuito de codifica\u00e7\u00e3o :math:`U_i` \u00e9 constru\u00eddo nos bits 0~3, incluindo m\u00faltiplas portas controladas :math:`CZ` , :math:`CNOT` e portas :math:`H`; o circuito de transforma\u00e7\u00e3o de pesos :math:`U_w` \u00e9 constru\u00eddo imediatamente ap\u00f3s :math:`U_i` , que tamb\u00e9m \u00e9 composto por portas controladas e portas :math:`H`. :math:`U_i` pode ser usado para realizar transforma\u00e7\u00f5es de matriz unit\u00e1ria para codificar dados em estados qu\u00e2nticos:

.. math::
    U_i|0\rangle^{\otimes N}=\left|\psi_i\right\rangle

Use a transforma\u00e7\u00e3o de matriz unit\u00e1ria :math:`U_w` para calcular o produto interno entre a entrada e os pesos:

.. math::
    U_w\left|\psi_i\right\rangle=\sum_{j=0}^{m-1} c_j|j\rangle \equiv\left|\phi_{i, w}\right\rangle

Os valores de probabilidade de ativa\u00e7\u00e3o normalizados para :math:`U_i` e :math:`U_w` podem ser obtidos usando uma porta NOT multicontrolada com bits alvo em bits auxiliares, e usando algumas portas :math:`H` subsequentes, portas :math:`X` e portas :math:`CX` como fun\u00e7\u00f5es de ativa\u00e7\u00e3o:

.. math::
    \left|\phi_{i, w}\right\rangle|0\rangle_a \rightarrow \sum_{j=0}^{m-2} c_j|j\rangle|0\rangle_a+c_{m-1}|m-1\rangle|1\rangle_a

Quando a sequ\u00eancia bin\u00e1ria da entrada i \u00e9 exatamente igual a w, o valor de probabilidade normalizado deve ser o maior.

O VQNet fornece o m\u00f3dulo ``QuantumNeuron`` para implementar este algoritmo. Primeiro inicialize um perceptron qu\u00e2ntico ``QuantumNeuron``.

.. code-block::

    perceptron = QuantumNeuron()

Use a interface ``gen_4bitstring_data`` para gerar v\u00e1rios dados no artigo e seus r\u00f3tulos de categoria.

.. code-block::

    training_label, test_label = perceptron.gen_4bitstring_data()

Usando a interface ``train`` para percorrer todos os dados, voc\u00ea pode obter o \u00faltimo circuito perceptron qu\u00e2ntico treinado :math:`U_w`.

.. code-block::

    trained_para = perceptron.train(training_label, test_label)

.. image:: ./images/QP-pic.png
   :width: 600 px
   :align: center

|

Nos dados de teste, os resultados de precis\u00e3o nos dados de teste podem ser obtidos

.. image:: ./images/QP-acc.png
   :width: 600 px
   :align: center

|



Gradiente Descendente Duplamente Estoc\u00e1stico
===============================================================================

Em algoritmos qu\u00e2nticos variacionais, circuitos qu\u00e2nticos parametrizados s\u00e3o otimizados por 
gradiente descendente cl\u00e1ssico para minimizar o valor esperado da fun\u00e7\u00e3o.
Embora o valor esperado possa ser calculado analiticamente em um simulador cl\u00e1ssico,
no hardware qu\u00e2ntico o programa \u00e9 limitado a amostrar do valor esperado;
\u00e0 medida que o n\u00famero de amostras e o n\u00famero de shots aumentam, 
o valor esperado obtido desta forma convergir\u00e1 para o valor esperado te\u00f3rico,
mas pode sempre ser um valor aproximado.
Sweke et al. encontrou um m\u00e9todo de gradiente descendente duplamente estoc\u00e1stico em `the paper <https://arxiv.org/abs/1910.01155>`_.
Neste artigo, eles mostram que o gradiente descendente qu\u00e2ntico, que usa um n\u00famero finito de amostras 
de medi\u00e7\u00e3o (ou shots) para estimar gradientes, \u00e9 uma forma de gradiente descendente estoc\u00e1stico.
Al\u00e9m disso, se a otimiza\u00e7\u00e3o envolver uma combina\u00e7\u00e3o linear de 
valores esperados (como VQE), amostrar os termos nessa 
combina\u00e7\u00e3o linear pode reduzir ainda mais a complexidade de tempo necess\u00e1ria.

O VQNet implementa um exemplo deste algoritmo: resolvendo a energia do estado fundamental do Hamiltoniano alvo usando VQE. Note que aqui definimos o n\u00famero de shots para observa\u00e7\u00f5es do circuito qu\u00e2ntico como apenas 1.

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
    # some basic Pauli matrices
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

O Hamiltoniano neste exemplo \u00e9 uma matriz Hermitiana,
que podemos sempre representar como uma soma de matrizes de Pauli.

.. math::

    H = \sum_{i,j=0,1,2,3} a_{i,j} (\sigma_i\otimes \sigma_j)

and 

.. math::

    a_{i,j} = \frac{1}{4}\text{tr}[(\sigma_i\otimes \sigma_j )H], ~~ \sigma = \{I, X, Y, Z\}.

Substituindo na f\u00f3rmula acima, podemos ver que

.. math::

    H = 4  + 2I\otimes X + 4I \otimes Z - X\otimes X + 5 Y\otimes Y + 2Z\otimes X.

Para realizar o gradiente descendente "duplamente estoc\u00e1stico", simplesmente aplicamos o m\u00e9todo de gradiente descendente estoc\u00e1stico, mas adicionalmente amostramos uniformemente um subconjunto da expectativa do Hamiltoniano a cada passo de otimiza\u00e7\u00e3o.
A fun\u00e7\u00e3o vqe_func_analytic() usa deslocamento de par\u00e2metros para calcular gradientes te\u00f3ricos, 
e vqe_func_shots() usa valores amostrados aleatoriamente e subconjuntos de expectativa do Hamiltoniano 
amostrados aleatoriamente para c\u00e1lculos de gradiente "duplamente estoc\u00e1stico".

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


Use VQNet para otimiza\u00e7\u00e3o de par\u00e2metros e compare a curva da fun\u00e7\u00e3o de perda.
Como o m\u00e9todo de gradiente descendente duplamente estoc\u00e1stico calcula apenas a soma parcial do operador 
de Pauli de H a cada vez,
portanto, o valor m\u00e9dio pode ser usado para representar o resultado esperado da 
observa\u00e7\u00e3o final. Aqui, a m\u00e9dia m\u00f3vel moving_average() \u00e9 usada para c\u00e1lculo.

.. code-block::


    ##############################################################################
    # Optimizing the circuit using gradient descent via the parameter-shift rule:
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
    #Doubly stochastic gradient descent min energy =  -4.337801834749975
    #stochastic gradient descent min energy =  -4.531484333030544
.. image:: ./images/dsgd.png
   :width: 600 px
   :align: center


Plat\u00f4s est\u00e9reis (Barren plateaus)
===============================================================================


No treinamento de redes neurais cl\u00e1ssicas, m\u00e9todos de otimiza\u00e7\u00e3o baseados em gradiente n\u00e3o apenas encontram o problema de m\u00ednimos locais,
mas tamb\u00e9m encontram estruturas geom\u00e9tricas como pontos de sela onde o gradiente est\u00e1 pr\u00f3ximo de zero.
Correspondentemente, o **efeito de plat\u00f4 est\u00e9ril** (barren plateau) tamb\u00e9m existe na rede neural qu\u00e2ntica.
Este fen\u00f4meno peculiar foi descoberto pela primeira vez por McClean et al. em 2018 `Barren plateaus in quantum neural network training landscapes <https://arxiv.org/abs/1803.11173>`_.
Simplificando, a paisagem de otimiza\u00e7\u00e3o se torna muito plana quando voc\u00ea escolhe uma estrutura de circuito aleat\u00f3ria que satisfaz um certo n\u00edvel de complexidade,
Isso torna dif\u00edcil para m\u00e9todos de otimiza\u00e7\u00e3o baseados em gradiente descendente encontrar o m\u00ednimo global.
Para a maioria dos algoritmos qu\u00e2nticos variacionais (VQE, etc.), este fen\u00f4meno significa que quando o n\u00famero de qubits aumenta,
circuitos com estruturas aleat\u00f3rias podem n\u00e3o funcionar bem.
Isso transformar\u00e1 a superf\u00edcie de otimiza\u00e7\u00e3o correspondente \u00e0 fun\u00e7\u00e3o de perda bem projetada em uma plataforma enorme,
tornando o treinamento de redes neurais qu\u00e2nticas mais dif\u00edcil.
O valor inicial encontrado aleatoriamente pelo modelo \u00e9 dif\u00edcil de escapar desta plataforma, e a velocidade de converg\u00eancia do gradiente descendente ser\u00e1 muito lenta.


Este caso usa principalmente VQNet para exibir o fen\u00f4meno de plat\u00f4 est\u00e9ril, e usa a fun\u00e7\u00e3o de an\u00e1lise de gradiente para analisar o gradiente dos par\u00e2metros na rede neural qu\u00e2ntica definida pelo usu\u00e1rio.

O c\u00f3digo a seguir constr\u00f3i o seguinte circuito aleat\u00f3rio de acordo com o m\u00e9todo similar mencionado no artigo original:

Primeiro, atue em todos os qubits com uma rota\u00e7\u00e3o em torno do eixo Y da esfera de Bloch :math:`\pi/4`.

O resto das estruturas se somam para formar um m\u00f3dulo (Block), cada m\u00f3dulo \u00e9 dividido em duas camadas:

- A primeira camada constr\u00f3i uma porta girat\u00f3ria aleat\u00f3ria, onde :math:`R \in \{R_x, R_y, R_z\}`.
- A segunda camada consiste em portas CZ atuando em cada par de qubits adjacentes.

O c\u00f3digo do circuito \u00e9 mostrado na fun\u00e7\u00e3o rand_circuit_pq.

Depois de determinarmos a estrutura do circuito, tamb\u00e9m precisamos definir uma fun\u00e7\u00e3o de perda (loss function) para determinar a superf\u00edcie de otimiza\u00e7\u00e3o.
Como mencionado no artigo original, usamos a fun\u00e7\u00e3o de perda comumente usada no algoritmo VQE:

.. math::

    \mathcal{L}(\boldsymbol{\theta})= \langle0| U^{\dagger}(\boldsymbol{\theta})H U(\boldsymbol{\theta}) |0\rangle

A matriz unit\u00e1ria :math:`U(\boldsymbol{\theta})` \u00e9 a rede neural qu\u00e2ntica com estrutura aleat\u00f3ria que constru\u00edmos na parte anterior.
Hamiltoniano :math:`H = |00\cdots 0\rangle\langle00\cdots 0|`.
Neste caso, o algoritmo VQE acima \u00e9 constru\u00eddo em diferentes n\u00fameros de qubits, e 200 conjuntos de estruturas de rede aleat\u00f3rias e diferentes par\u00e2metros iniciais aleat\u00f3rios s\u00e3o gerados.
O gradiente dos par\u00e2metros na linha com par\u00e2metros \u00e9 calculado de acordo com o algoritmo paramter-shift.
Em seguida, calcule a m\u00e9dia e a vari\u00e2ncia dos 200 gradientes dos par\u00e2metros variacionais obtidos.

O exemplo a seguir analisa o \u00faltimo dos par\u00e2metros qu\u00e2nticos vari\u00e1veis, e os leitores tamb\u00e9m podem modific\u00e1-lo para outros valores razo\u00e1veis.
Atrav\u00e9s da opera\u00e7\u00e3o, n\u00e3o \u00e9 dif\u00edcil para os leitores descobrirem que, \u00e0 medida que o n\u00famero de qubits aumenta, a vari\u00e2ncia do gradiente dos par\u00e2metros qu\u00e2nticos se torna cada vez menor, e o valor m\u00e9dio se aproxima de 0.

.. code-block::


        \"\"\"
        plateau est\u00e9ril
        \"\"\"
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
        plt.ylabel(r"variance")
        plt.show()


        plt.figure()
        # Plot the straight line fit to the semilog
        plt.plot(qubits, means, "v-")

        plt.xlabel(r"N Qubits")
        plt.ylabel(r"means")

        plt.show()


A figura abaixo mostra como o valor m\u00e9dio do gradiente do par\u00e2metro varia com o n\u00famero de qubits. \u00c0 medida que o n\u00famero de qubits aumenta, o gradiente do par\u00e2metro se aproxima de 0.

.. image:: ./images/Barren_Plateau_mean.png
   :width: 600 px
   :align: center

|

A figura abaixo mostra como a vari\u00e2ncia do gradiente do par\u00e2metro varia com o n\u00famero de qubits, e o gradiente do par\u00e2metro quase n\u00e3o muda \u00e0 medida que o n\u00famero de qubits aumenta.
Pode-se prever que o circuito qu\u00e2ntico constru\u00eddo por qualquer porta l\u00f3gica param\u00e9trica ser\u00e1 dif\u00edcil de atualizar no caso de inicializa\u00e7\u00e3o de par\u00e2metros arbitr\u00e1ria quando os qubits forem aumentados.

.. image:: ./images/Barren_Plateau_variance.png
   :width: 600 px
   :align: center

|


Poda baseada em gradiente
===============================================================================

O exemplo a seguir implementa o algoritmo no artigo `Towards Efficient On-Chip Training of Quantum Neural Networks <https://openreview.net/forum?id=vKefw-zKOft>`_.
Ao estudar cuidadosamente o processo dos par\u00e2metros no circuito variacional qu\u00e2ntico, os pesquisadores observaram que gradientes pequenos frequentemente t\u00eam grandes mudan\u00e7as relativas ou mesmo dire\u00e7\u00f5es erradas sob ru\u00eddo qu\u00e2ntico.
Al\u00e9m disso, nem todos os c\u00e1lculos de gradiente s\u00e3o necess\u00e1rios para o processo de treinamento, especialmente para gradientes de pequena magnitude.
Inspirados por isso, os pesquisadores prop\u00f5em um m\u00e9todo de poda de gradiente probabil\u00edstico para prever e calcular apenas gradientes com alta confiabilidade.
A abordagem reduz os efeitos do ru\u00eddo e tamb\u00e9m economiza o n\u00famero de circuitos necess\u00e1rios para executar em uma m\u00e1quina qu\u00e2ntica real.

No algoritmo de poda baseada em gradiente, para o processo de otimiza\u00e7\u00e3o dos par\u00e2metros, duas etapas de janela de acumula\u00e7\u00e3o e janela de poda s\u00e3o divididas, e todos os per\u00edodos de treinamento s\u00e3o divididos em uma janela de acumula\u00e7\u00e3o repetida e depois uma janela de poda. Existem tr\u00eas hiperpar\u00e2metros importantes no m\u00e9todo de poda de gradiente probabil\u00edstico:

     * Largura da janela de acumula\u00e7\u00e3o :math:`\omega_a` , 
     * Taxa de poda :math:`r` ,
     * Largura da janela de poda :math:`\omega_p` .

Na janela de acumula\u00e7\u00e3o, os pesquisadores coletam a informa\u00e7\u00e3o do gradiente em cada etapa de treinamento. Em cada etapa da janela de poda,
com base nas informa\u00e7\u00f5es reunidas da janela de acumula\u00e7\u00e3o e na taxa de poda, o algoritmo
probabilisticamente pula alguns c\u00e1lculos de gradiente.

.. image:: ./images/gbp_arch.png
   :width: 600 px
   :align: center

|

A taxa de poda :math:`r` , a largura da janela de acumula\u00e7\u00e3o :math:`\omega_a` e a largura da janela de poda :math:`\omega_p` determinam respectivamente a confiabilidade da avalia\u00e7\u00e3o da tend\u00eancia do gradiente.
Assim, a porcentagem de tempo economizada pelo nosso m\u00e9todo de poda de gradiente probabil\u00edstico \u00e9 :math:`r\tfrac{\omega_p}{\omega_a +\omega_p}\times 100\%`.
A seguir, a aplica\u00e7\u00e3o do exemplo de classifica\u00e7\u00e3o QVC usando o algoritmo de poda de gradiente.

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
        \"\"\"
        Fun\u00e7\u00e3o de execu\u00e7\u00e3o de circuitos qu\u00e2nticos
        \"\"\"
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
        \"\"\"
        Fun\u00e7\u00e3o de execu\u00e7\u00e3o de circuitos qu\u00e2nticos
        \"\"\"
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
            #y = self.qvc2(y)
            return y


    def get_data(dataset_str):
        \"\"\"
        Transformar dados para formato v\u00e1lido
        \"\"\"
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

Usamos a classe ``Gradient_Prune_Instance``, e inserimos `24` como o n\u00famero de par\u00e2metros `param_num`.
A taxa de poda `prune_ratio` \u00e9 0.5,
o tamanho da janela de acumula\u00e7\u00e3o `accumulation_window_size` \u00e9 4,
e a janela de poda `pruning_window_size` \u00e9 2.
Ao fazer backpropagation de parte do c\u00f3digo a cada execu\u00e7\u00e3o, antes do otimizador ``step``,
Execute a fun\u00e7\u00e3o ``step`` de ``Gradient_Prune_Instance``.

.. code-block::

    def run():
        \"\"\"
        Fun\u00e7\u00e3o principal de execu\u00e7\u00e3o
        \"\"\"
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

    # epoch:0, #### loss:0.2255942871173223 #####accuracy:0.5833333333333334
    # epoch:1, #### loss:0.1989427705605825 #####accuracy:1.0
    # epoch:2, #### loss:0.16489211718241373 #####accuracy:1.0
    # epoch:3, #### loss:0.13245886812607446 #####accuracy:1.0
    # epoch:4, #### loss:0.11463981121778488 #####accuracy:1.0
    # epoch:5, #### loss:0.1078591321905454 #####accuracy:1.0
    # epoch:6, #### loss:0.10561319688955943 #####accuracy:1.0
    # epoch:7, #### loss:0.10483601937691371 #####accuracy:1.0
    # epoch:8, #### loss:0.10457512239615123 #####accuracy:1.0
    # epoch:9, #### loss:0.10448987782001495 #####accuracy:1.0
    # start testing..............
    # test:--------------->loss:[0.3134713] #####accuracy:1.0



Treinamento de modelo usando camada de computa\u00e7\u00e3o qu\u00e2ntica no VQNet
**********************************************************************************

A seguir, alguns exemplos de uso da interface VQNet para aprendizado de m\u00e1quina qu\u00e2ntico ``QuantumLayer`` , ``NoiseQuantumLayer`` .

Treinamento de modelo usando quantumlayer no VQNet
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

Resultados de perda e precis\u00e3o da execu\u00e7\u00e3o:

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


Treinamento de modelo usando NoiseQuantumLayer no VQNet
==============================================================================

Usando ``NoiseQuantumLayer`` para construir e treinar circuitos qu\u00e2nticos ruidosos usando a m\u00e1quina virtual de ru\u00eddo do pyQPanda2.

Um exemplo completo de modelo de aprendizado de m\u00e1quina qu\u00e2ntico ruidoso \u00e9 o seguinte:

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

    #usar qpanda para criar circuitos qu\u00e2nticos
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
        # Compute probabilities for each state
        probabilities = counts / 100
        # Get state expectation
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

O modelo \u00e9 um modelo h\u00edbrido de circuito qu\u00e2ntico e rede cl\u00e1ssica, onde a parte do circuito 
qu\u00e2ntico usa ``NoiseQuantumLayer`` para simular o circuito qu\u00e2ntico mais modelo de ru\u00eddo. Este modelo 
\u00e9 usado para classificar d\u00edgitos manuscritos 0 e 1 no banco de dados MNIST.

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

        # Teste deixando apenas r\u00f3tulos 0 e 1
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
		
Comparando os resultados de classifica\u00e7\u00e3o de modelos de aprendizado de m\u00e1quina de circuitos qu\u00e2nticos ruidosos e circuitos qu\u00e2nticos ideais, 
o log de mudan\u00e7a de perda e o log de mudan\u00e7a de acur\u00e1cia s\u00e3o os seguintes:

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
 