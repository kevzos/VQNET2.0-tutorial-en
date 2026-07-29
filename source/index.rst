.. VQNet documentation master file, created by
   sphinx-quickstart on Tue Jul 27 15:25:07 2021.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

VQNet
=================================

Fonctionnalités principales de VQNet
--------------------------------------

Compatibilité multi-plateforme et support multi-environnement
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

VQNet permet aux utilisateurs de mener des recherches et développements en apprentissage automatique quantique dans divers environnements matériels et systèmes d'exploitation. Que ce soit en utilisant un CPU ou un GPU pour la simulation de calcul quantique, ou en appelant des puces quantiques réelles via le service cloud Benyuan Quantum, VQNet offre un support transparent. Actuellement, VQNet est compatible avec Python 3.10 sur les systèmes Windows, Linux et macOS.

Conception d'interface parfaite et facilité d'utilisation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

VQNet utilise Python comme langage frontal, offre une interface similaire à PyTorch et permet de choisir librement parmi plusieurs moteurs de calcul pour implémenter la fonction de différenciation automatique des modèles d'apprentissage automatique classiques et quantiques. Le framework intègre : plus de 100 interfaces de calcul Tensor couramment utilisées, plus de 100 interfaces de calcul de circuits variationnels quantiques et plus de 50 interfaces de réseaux neuronaux classiques. Ces interfaces couvrent l'ensemble du processus de développement, de l'apprentissage automatique classique à l'apprentissage automatique quantique, et seront continuellement mises à jour.

Performances de calcul efficaces et capacités d'extension
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- **Support des expériences sur puces quantiques réelles** : Pour les utilisateurs ayant besoin d'expériences sur de vraies puces quantiques, VQNet intègre l'interface pyQPanda d'origine et combine les capacités de planification efficaces de Sinan pour réaliser des calculs de simulation rapides de circuits quantiques et le fonctionnement sur puce réelle.
- **Optimisation du calcul local** : Pour les besoins de calcul local, VQNet fournit une interface de programmation d'apprentissage automatique quantique basée sur CPU ou GPU, et utilise la technologie de différenciation automatique pour effectuer les calculs de gradient des circuits variationnels quantiques, ce qui est nettement plus rapide que les méthodes de décalage de paramètres. Les détails se trouvent dans :ref:`benchmarks`.
- **Support du calcul distribué** : VQNet prend en charge le calcul distribué basé sur MPI, ce qui permet d'entraîner des modèles de réseaux neuronaux hybrides quantiques-classiques à grande échelle sur plusieurs nœuds.

Scénarios d'application riches et support d'exemples
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

VQNet n'est pas seulement un outil de développement puissant, mais il est également largement utilisé dans de nombreux projets internes, notamment dans les domaines de l'optimisation énergétique, de l'analyse de données médicales, du traitement d'images, etc. Pour aider les utilisateurs à démarrer rapidement, VQNet propose une variété de scénarios allant des tutoriels de base aux applications avancées sur le site officiel et la documentation en ligne de l'API. Ces ressources permettent aux utilisateurs de comprendre facilement comment utiliser VQNet pour résoudre des problèmes concrets et construire rapidement leurs propres applications d'apprentissage automatique quantique.

.. toctree::
    :caption: Installation Guide
    :maxdepth: 2

    rst/install.rst

.. toctree::
    :caption: Hands-on Examples
    :maxdepth: 2

    rst/vqc_demo.rst
    rst/qml_demo.rst

.. toctree::
    :caption: Classic neural network API
    :maxdepth: 2

    rst/QTensor.rst
    rst/nn.rst
    rst/utils.rst

.. toctree::
    :caption: QNN API integrated with pyqpanda
    :maxdepth: 2

    rst/qnn.rst
    rst/qnn_pq3.rst

.. toctree::
    :caption: Autograd QNN API
    :maxdepth: 2

    rst/vqc.rst

.. toctree:: 
    :caption: Others 
    :maxdepth: 2 
    
    rst/torch_api.rst
    rst/vqnet_dist.rst
    rst/FAQ.rst 
    rst/CHANGELOG.rst




