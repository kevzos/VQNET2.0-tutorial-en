Étapes d'installation de VQNet
==================================

Installation du package Python VQNet
-------------------------------------

Nous fournissons des packages Python précompilés pour une installation sur Linux, Windows, macOS 13+ (arm64), prenant en charge **Python 3.10**.

Téléchargez l'archive correspondante depuis le site officiel, extrayez-la, naviguez vers le répertoire extrait et exécutez les étapes suivantes :

.. code-block::

    # Pour Windows
    ./install.bat
    # Pour macOS et Linux
    ./install.sh
 


Pour les systèmes Windows et Linux, le package pyvqnet inclut des fonctionnalités d'accélération intégrées pour les calculs de réseaux neuronaux classiques basés sur Nvidia CUDA, qui dépendent de la version spécifique des bibliothèques d'exécution NVIDIA CUDA 12.6 (installées automatiquement avec le package).
Le package est optimisé pour les architectures CUDA suivantes :
**sm_80** (GPU de centre de données NVIDIA A100, série A30) et **sm_86** (GPU grand public NVIDIA GeForce série RTX 30). Veuillez vous assurer que vous utilisez un GPU prenant en charge ces architectures ; sinon, le programme pourrait ne pas fonctionner correctement.

    .. important::

        Veuillez noter que ce package ne faisant pas de distinction entre les versions CPU/GPU, il dépend des bibliothèques d'exécution NVIDIA CUDA sous Windows et Linux, qui sont installées automatiquement avec le package. Cela peut entraîner des conflits avec d'autres logiciels qui dépendent de versions différentes.


Validation de l'installation de VQNet
----------------------------------------

.. code-block::

    import pyvqnet
    from pyvqnet.tensor import *
    a = arange(1,25).reshape([2, 3, 4])
    print(a)

Test des fonctionnalités GPU dans VQNet
-----------------------------------------

.. code-block::

    from pyvqnet import DEV_GPU_0
    from pyvqnet.tensor import *
    a = ones([4,5],device = DEV_GPU_0)
    print(a)

Un cas simple avec VQNet
---------------------------
Nous présentons ici un cas composé de modules de réseaux neuronaux classiques et de modules quantiques de VQNet pour décrire le flux de travail de l'apprentissage automatique quantique.
Il fait référence à `Data re-uploading for a universal quantum classifier <https://arxiv.org/abs/1907.02085>`_ .
Généralement, les parties suivantes du module de calcul quantique sont présentes dans l'apprentissage automatique quantique :

(1) Codeur : encodage des données classiques en état quantique ;

(2) Ansatz : entraînement des paramètres dans les portes quantiques paramétrées ;

(3) Mesure : mesure de la valeur d'un qubit (projection de l'état quantique du qubit sur un axe spécifié).

Le module de calcul quantique est la base théorique du modèle hybride de réseau neuronal quantique-classique, qui est également différenciable comme les modules de réseaux neuronaux classiques. VQNet prend en charge les modules de calcul quantique et les modules de calcul classique pour former un modèle d'apprentissage automatique hybride, et fournit divers algorithmes d'optimisation pour l'optimisation des paramètres (par exemple, couche de convolution, couche de pooling, couche entièrement connectée, fonction d'activation, etc.).

.. figure:: ./images/classic-quantum.PNG

Dans le module de calcul quantique, VQNet prend en charge l'utilisation du package logiciel de calcul quantique efficace pyqpanda3 pour construire des modules quantiques.
En utilisant les diverses interfaces couramment utilisées fournies par pyqpanda3, les utilisateurs peuvent rapidement construire des modules de calcul quantique.

L'exemple suivant utilise pyqpanda3 pour construire un module de calcul quantique. Via VQNet, ce module quantique peut être directement intégré dans un modèle d'apprentissage automatique hybride pour l'entraînement des paramètres du circuit quantique.
Dans cet exemple, 1 qubit est utilisé, plusieurs portes de rotation paramétrées `RZ`, `RY`, `RZ` sont utilisées pour encoder l'entrée x, et la fonction `probs_measure` est utilisée pour observer le résultat de mesure de probabilité du qubit en sortie.

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

Notre tâche est de classer ces données générées aléatoirement en utilisant une approche de classification binaire. Dans cette tâche,
le centre d'un cercle est à l'origine, les points de rayon 1 colorés en rouge appartiennent à une catégorie, et les échantillons colorés en bleu appartiennent à une autre catégorie.

.. figure:: ./images/origin_circle.png

Le pipeline du processus d'entraînement

.. code-block::

    # import required libraries and functions
    from pyvqnet.qnn.pq3.quantumlayer import QuantumLayer
    from pyvqnet.optim import adam
    from pyvqnet.nn.loss import CategoricalCrossEntropy
    from pyvqnet.tensor import QTensor
    import numpy as np
    from pyvqnet.nn.module import Module


Définition d'un modèle : la fonction ``__init__`` définit les modules de réseaux neuronaux internes et les modules quantiques, et la fonction ``forward`` définit le calcul direct. ``QuantumLayer`` est une classe abstraite
qui encapsule le calcul quantique.
VQNet calculera automatiquement les gradients des paramètres pour `qdrl_circuit` avec `param_num`.


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

Définition de quelques fonctions pour l'entraînement du modèle 

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

VQNet suit le flux de travail général de l'apprentissage automatique : chargement itératif des données, propagation avant, calcul de la fonction de perte, rétropropagation et mise à jour des paramètres.

.. code-block::

    # instantiating a model
    model = Model()
    # using Adam to define a optimizer
    optimizer = adam.Adam(model.parameters(),lr =0.6)
    # using cross-entropy to define a loss function
    Closs = CategoricalCrossEntropy()

Une fonction pour entraîner le modèle

.. code-block::

    def train():
            
        #  générer des données à entraîner aléatoirement
        x_train, y_train = circle(500)
        x_train = np.hstack((x_train, np.zeros((x_train.shape[0], 1),dtype=np.float32)))  
        # définir le nombre de données par lot
        batch_size = 32
        # Nombre maximum d'itérations d'entraînement
        epoch = 10
        print("start training...........")
        for i in range(epoch):
            model.train()
            accuracy = 0
            count = 0
            loss = 0
            for data, label in get_minibatch_data(x_train, y_train,batch_size):
                # effacer les gradients de l'optimiseur
                optimizer.zero_grad()
                # calcul avant
                output = model(data)
                # calcul de la fonction de perte
                losss = Closs(label, output)
                # rétropropagation
                losss.backward()
                # mettre à jour les paramètres de l'optimiseur
                optimizer._step()
                # calculer la précision
                accuracy += get_score(output,label)

                loss += losss.item()
                count += batch_size
                
            print(f"epoch:{i}, train_accuracy:{accuracy/count}")
            print(f"epoch:{i}, train_loss:{loss/count}\n")
            
Une fonction pour valider le modèle

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

Résultats de l'entraînement et du test

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







