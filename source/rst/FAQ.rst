Foire Aux Questions
=============================

**Q : Quelles sont les fonctionnalités de VQNet ?**

R : VQNet est un ensemble d'outils d'apprentissage automatique quantique développé par Origin Quantum à partir de pyQPanda. VQNet fournit un ensemble riche d'interfaces faciles à utiliser pour les modules de calcul de réseaux neuronaux classiques, permettant une optimisation pratique de l'apprentissage automatique.
La méthode de définition du modèle est cohérente avec les frameworks d'apprentissage automatique grand public, ce qui réduit la courbe d'apprentissage pour les utilisateurs.
Parallèlement, basé sur pyQPanda, un simulateur quantique haute performance développé par Origin Quantum, VQNet peut également prendre en charge le fonctionnement d'un grand nombre de bits quantiques sur des ordinateurs portables personnels. Enfin, VQNet fournit également des exemples riches :doc:`./qml_demo` pour votre référence et votre apprentissage.

**Q : Comment utiliser VQNet pour entraîner des modèles d'apprentissage automatique quantique ?**

R : Il existe un type d'algorithme d'apprentissage automatique quantique qui construit des modèles différenciables basés sur des circuits variationnels quantiques.
VQNet peut utiliser la méthode de descente de gradient pour entraîner ce type de modèle. Les étapes générales sont les suivantes : Premièrement, sur l'ordinateur local, les utilisateurs peuvent construire une machine virtuelle via pyQPanda et combiner les interfaces fournies dans VQNet pour construire un modèle hybride quantique-classique ``Module`` ; deuxièmement, l'appel de ``forward()`` du ``Module`` peut effectuer la simulation du circuit quantique et le calcul direct du réseau neuronal classique selon le mode de fonctionnement défini par l'utilisateur ;
Lors de l'appel de ``backward()`` du ``Module``, le modèle construit par l'utilisateur peut être automatiquement différencié comme dans les frameworks d'apprentissage automatique classiques tels que PyTorch, et calculer les gradients des paramètres dans les circuits variationnels quantiques et les couches de calcul classiques ; enfin, combinez la fonction ``step()`` de l'optimiseur pour optimiser les paramètres.

Dans VQNet, nous utilisons `parameter-shift <https://arxiv.org/abs/1803.00745>`_ pour calculer le gradient des circuits variationnels quantiques. Les utilisateurs peuvent utiliser l'interface sous :ref:`QuantumLayer_pq3` fournie par VQNet pour encapsuler la différenciation automatique des circuits variationnels quantiques. Les utilisateurs n'ont qu'à définir les circuits variationnels quantiques comme paramètres dans un certain format pour construire les classes ci-dessus.

Dans VQNet, nous pouvons également utiliser la méthode basée sur la différenciation automatique pour calculer le gradient des circuits variationnels quantiques. Les utilisateurs peuvent utiliser l'interface dans :ref:`vqc_api` pour construire un circuit entraînable. Ce circuit ne dépend pas de pyQPanda, mais divise le codage, les opérations de porte et la mesure dans le circuit en opérateurs différenciables, afin de réaliser la fonction de calcul du gradient des paramètres.

Pour plus de détails, veuillez vous référer aux interfaces pertinentes et aux exemples de code dans ce document.

**Q : Sous Windows, j'ai rencontré une erreur lors de l'installation de VQNet : "importError: DLL load failed while importing _core: The specified module could not be found."**

R : Les utilisateurs peuvent avoir besoin d'installer la bibliothèque d'exécution VC++ sur Windows.
Référez-vous à https://learn.microsoft.com/en-us/cpp/windows/latest-supported-vc-redist?view=msvc-170 pour installer la bibliothèque d'exécution appropriée.
De plus, VQNet ne prend actuellement en charge que la version python3.10, veuillez donc confirmer votre version de python.

**Q : Comment appeler le cloud quantique d'origine et la puce quantique pour le calcul ?**

R : Vous pouvez utiliser le cluster de calcul haute performance d'Origin Quantum ou de véritables ordinateurs quantiques pour la simulation de circuits quantiques, remplaçant ainsi la simulation locale de circuits quantiques par le cloud computing.
Dans VQNet, les utilisateurs peuvent utiliser ``QuantumBatchAsyncQcloudLayer`` pour construire un module de circuit variationnel quantique, saisir les CLÉS API demandées sur le site officiel d'Origin et soumettre la tâche à la machine réelle pour exécution.

**Q : Pourquoi les paramètres du modèle que j'ai définis ne sont-ils pas mis à jour pendant l'entraînement ?**

R : Pour construire un modèle VQNet, il est nécessaire de s'assurer que tous les modules utilisés sont différenciables. Lorsqu'un module du modèle ne peut pas calculer le gradient, ce module et les modules précédents ne pourront pas calculer le gradient en utilisant la règle de la chaîne.
Si l'utilisateur personnalise un circuit variationnel quantique, veuillez utiliser l'interface sous :ref:`QuantumLayer_pq3` fournie par VQNet. Pour les modules d'apprentissage automatique classiques, vous devez utiliser les interfaces définies par :doc:`./QTensor` et :doc:`./nn`. Ces interfaces encapsulent les fonctions de calcul de gradient, et VQNet peut effectuer une différenciation automatique.

Si l'utilisateur souhaite utiliser une liste contenant plusieurs modules comme sous-module dans `Module`, veuillez ne pas utiliser la liste Python intégrée. Utilisez plutôt pyvqnet.nn.module.ModuleList. Ainsi, les paramètres d'entraînement des sous-modules peuvent être enregistrés dans l'ensemble du modèle, permettant un entraînement par différenciation automatique. Voici un exemple :

     Example::

         from pyvqnet. tensor import *
         from pyvqnet.nn import Module,Linear,ModuleList
         from pyvqnet.qnn import ProbsMeasure, QuantumLayer
         import pyqpanda as pq
         def pqctest(input, param, qubits, cbits, m_machine):
             circuit = pq. QCircuit()
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
             prog. insert(circuit)

             rlt_prob = ProbsMeasure([0,2],prog,m_machine,qubits)
             return rlt_prob


         class M(Module):
             def __init__(self):
                 super(M, self).__init__()
                 #Should be built using ModuleList
                 self.pqc2 = ModuleList([QuantumLayer(pqctest,3,"cpu",4,1), Linear(4,1)
                 ])
                 #Direct use of list cannot save the parameters in pqc3.
                 #self.pqc3 = [QuantumLayer(pqctest,3,"cpu",4,1), Linear(4,1)
                 #]
             def forward(self, x, *args, **kwargs):
                 y = self.pqc2[0](x) + self.pqc2[1](x)
                 return y

         mm = M()
         print(mm. state_dict(). keys())

**Q : Pourquoi le code d'origine ne fonctionnait-il pas dans la version 2.0.7 ?**

R : Dans la version v2.0.7, nous avons ajouté différents types de données et attributs dtype à QTensor, et restreint les formats d'entrée selon les conventions PyTorch. Par exemple, l'entrée de la couche Embedding doit être de type kint64, et les étiquettes pour les couches CategoricalCrossEntropy, CrossEntropyLoss, SoftmaxCrossEntropy et NLL_Loss doivent être de type kint64.

Vous pouvez utiliser l'interface 'astype()' pour convertir le type vers le type de données spécifié, ou initialiser le QTensor en utilisant un tableau numpy du type de données correspondant.

**Q : VQNet dépend-il de torch ?**

R : VQNet ne dépend pas de torch et n'installe pas torch automatiquement.

Pour utiliser les fonctionnalités suivantes, vous devez installer torch>=2.11.0 vous-même. Depuis la v2.15.0, nous supportons l'utilisation de `torch >=2.11.0 <https://docs.pytorch.org/docs/stable/index.html>`_ comme moteur de calcul pour les réseaux neuronaux classiques, les circuits variationnels quantiques, le calcul distribué, etc.
Après avoir appelé ``pyvqnet.backends.set_backend("torch")``, l'interface reste inchangée, mais les variables membres ``data`` du ``QTensor`` de VQNet utilisent toutes ``torch.Tensor`` pour stocker les données,
et utilisent torch pour le calcul. Les classes sous ``pyvqnet.nn.torch`` et ``pyvqnet.qnn.vqc.torch`` héritent de ``torch.nn.Module`` et peuvent former des modèles ``torch``.
