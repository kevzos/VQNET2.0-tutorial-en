Pasos de instalación de VQNet
==================================

Instalación del paquete Python de VQNet
-----------------------------------------

Proporcionamos paquetes Python precompilados para instalación en Linux, Windows, macOS 13+ (arm64), compatibles con **Python 3.10**.

Descargue el archivo correspondiente desde el sitio web oficial, extráigalo, navegue al directorio extraído y ejecute los siguientes pasos:

.. code-block::

    # Para Windows
    ./install.bat
    # Para macOS y Linux
    ./install.sh
 


Para los sistemas Windows y Linux, el paquete pyvqnet incluye funciones de aceleración integradas para cálculos de redes neuronales clásicas basadas en Nvidia CUDA, que dependen de la versión específica de las bibliotecas de tiempo de ejecución de NVIDIA CUDA 12.6 (instaladas automáticamente con el paquete).
El paquete está optimizado para las siguientes arquitecturas CUDA:
**sm_80** (GPU de centro de datos NVIDIA A100, serie A30) y **sm_86** (GPU de consumo NVIDIA GeForce serie RTX 30). Asegúrese de estar utilizando una GPU compatible con estas arquitecturas; de lo contrario, el programa podría no funcionar correctamente.

    .. important::

        Tenga en cuenta que, dado que este paquete no distingue entre versiones CPU/GPU, depende de las bibliotecas de tiempo de ejecución de NVIDIA CUDA en Windows y Linux, que se instalan automáticamente con el paquete. Esto puede causar conflictos con otro software que dependa de versiones diferentes.


Validar la instalación de VQNet
----------------------------------

.. code-block::

    import pyvqnet
    from pyvqnet.tensor import *
    a = arange(1,25).reshape([2, 3, 4])
    print(a)

Prueba de la funcionalidad GPU en VQNet
-----------------------------------------

.. code-block::

    from pyvqnet import DEV_GPU_0
    from pyvqnet.tensor import *
    a = ones([4,5],device = DEV_GPU_0)
    print(a)

Un caso simple de VQNet
--------------------------
Aquí presentamos un caso que consiste en módulos de redes neuronales clásicas y módulos cuánticos de VQNet para describir el flujo de trabajo del aprendizaje automático cuántico.
Se refiere a `Data re-uploading for a universal quantum classifier <https://arxiv.org/abs/1907.02085>`_ .
Generalmente, las siguientes partes del módulo de computación cuántica están presentes en el aprendizaje automático cuántico:

(1) Codificador: codificación de datos clásicos en estado cuántico;

(2) Ansatz: entrenamiento de los parámetros en puertas cuánticas parametrizadas;

(3) Medición: medición del valor de un qubit (proyección del estado cuántico del qubit en un eje especificado).

El módulo de computación cuántica es la base teórica del modelo híbrido de red neuronal cuántico-clásica, que también es diferenciable como los módulos de redes neuronales clásicas. VQNet admite módulos de computación cuántica y módulos de computación clásica para formar un modelo de aprendizaje automático híbrido, y proporciona varios algoritmos de optimización para la optimización de parámetros (por ejemplo, capa de convolución, capa de pooling, capa totalmente conectada, función de activación, etc.).

.. figure:: ./images/classic-quantum.PNG

En el módulo de computación cuántica, VQNet admite el uso del paquete de software de computación cuántica eficiente pyqpanda3 para construir módulos cuánticos.
Utilizando las diversas interfaces comúnmente utilizadas proporcionadas por pyqpanda3, los usuarios pueden construir rápidamente módulos de computación cuántica.

El siguiente ejemplo utiliza pyqpanda3 para construir un módulo de computación cuántica. A través de VQNet, este módulo cuántico se puede integrar directamente en un modelo de aprendizaje automático híbrido para el entrenamiento de parámetros del circuito cuántico.
En este ejemplo, se utiliza 1 qubit, se utilizan múltiples puertas de rotación parametrizadas `RZ`, `RY`, `RZ` para codificar la entrada x, y la función `probs_measure` se utiliza para observar el resultado de medición de probabilidad del qubit como salida.

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

Nuestra tarea es clasificar estos datos generados aleatoriamente utilizando un enfoque de clasificación binaria. En esta tarea,
el centro de un círculo está en el origen, los puntos dentro del radio 1 coloreados en rojo pertenecen a una categoría, y las muestras coloreadas en azul pertenecen a otra categoría.

.. figure:: ./images/origin_circle.png

El pipeline del proceso de entrenamiento

.. code-block::

    # importar las bibliotecas y funciones requeridas
    from pyvqnet.qnn.pq3.quantumlayer import QuantumLayer
    from pyvqnet.optim import adam
    from pyvqnet.nn.loss import CategoricalCrossEntropy
    from pyvqnet.tensor import QTensor
    import numpy as np
    from pyvqnet.nn.module import Module


Definiendo un modelo: la función ``__init__`` define los módulos de red neuronal internos y los módulos cuánticos, y la función ``forward`` define el cálculo forward. ``QuantumLayer`` es una clase abstracta
que encapsula la computación cuántica.
VQNet calculará automáticamente los gradientes de los parámetros para `qdrl_circuit` con `param_num`.


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

Definiendo algunas funciones para entrenar el modelo 

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

VQNet sigue el flujo de trabajo general del aprendizaje automático: carga iterativa de datos, propagación hacia adelante, cálculo de la función de pérdida, retropropagación y actualización de los parámetros.

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







