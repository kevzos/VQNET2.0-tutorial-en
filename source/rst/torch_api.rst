

.. _torch_api:

=============================================================
VQNet utilise torch pour les calculs de bas niveau
=============================================================

À partir de la version 2.15.0, ce logiciel supporte l'utilisation de ``torch`` comme backend de calcul pour les opérations de bas niveau et peut être intégré avec des modèles, du code et des bibliothèques tierces basés sur ``torch`` pour le développement secondaire.

    .. important::

        Pour utiliser les fonctionnalités suivantes, veuillez installer torch>=2.11.0 vous-même. Si vous installez une version GPU de torch, vous devez utiliser une version compatible avec CUDA 12.6, sinon votre torch pourrait ne pas fonctionner en raison de problèmes de bibliothèque d'exécution NVIDIA CUDA. Ce logiciel n'installe pas automatiquement torch lors de l'installation.

    .. note::

        Les fonctions de calcul quantique variationnel (avec des noms en minuscules, comme ``rx``, ``ry``, ``rz``, etc.) dans :ref:`vqc_api`, ainsi que les fonctions de calcul de base de QTensor dans :ref:`qtensor_api`,
        peuvent prendre un ``QTensor`` en entrée après avoir appelé ``pyvqnet.backends.set_backend("torch")``, le membre ``data`` du ``QTensor`` passant du Tensor de pyvqnet à ``torch.Tensor`` pour le calcul.

        ``pyvqnet.backends.set_backend("torch")`` et ``pyvqnet.backends.set_backend("pyvqnet")`` modifient le backend de calcul global.
        Les objets ``QTensor`` créés sous différentes configurations de backend ne peuvent pas être mélangés dans les calculs.

Configuration de base du backend
============================================

set_backend
------------------------------------------------

.. py:function:: pyvqnet.backends.set_backend(backend_name)

    Définit le backend pour les calculs et le stockage de données en cours. La valeur par défaut est "pyvqnet-ad", mais peut être définie sur "torch", "torch-native", "pyvqnet-ad".
    
    Après avoir appelé ``pyvqnet.backends.set_backend("torch")``, l'interface reste inchangée. Le membre ``data`` du ``QTensor`` de VQNet utilise tous ``torch.Tensor`` pour stocker les données.
    :ref:`qtensor_api`, :ref:`vqc_api` et les interfaces ``pyvqnet.nn.torch`` acceptent ``QTensor`` en entrée et ``QTensor`` en sortie.

    Après avoir appelé ``pyvqnet.backends.set_backend("torch-native")``, les interfaces restent inchangées : :ref:`qtensor_api`, :ref:`vqc_api` et l'interface `pyvqnet.nn.torch`.
    Les entrées peuvent accepter directement les types ``torch.Tensor`` ou ``QTensor``, et les sorties sont ``torch.Tensor``, éliminant ainsi le besoin de conversion en ``QTensor``, réduisant ainsi la conversion de données.
    
    Après avoir appelé ``pyvqnet.backends.set_backend("pyvqnet")``, le membre ``data`` du ``QTensor`` de VQNet stockera les données en utilisant ``pyvqnet._core.Tensor``, et les calculs utiliseront la bibliothèque C++ pyvqnet.

    Après avoir appelé ``pyvqnet.backends.set_backend("pyvqnet-ad")``, le membre ``data`` du ``QTensor`` de VQNet stockera les données en utilisant ``pyvqnet._core.Tensor``, et les calculs utiliseront la bibliothèque C++ pyvqnet avec des performances améliorées.


    .. note::

        Cette fonction modifie le backend de calcul actuel. Les objets ``QTensor`` créés sous différents backends ne peuvent pas être utilisés ensemble dans les calculs.

    :param backend_name: Nom du backend, peut être "pyvqnet" ou "torch".

    Example::

        import pyvqnet
        pyvqnet.backends.set_backend("torch")

get_backend
-------------------------------

.. py:function:: pyvqnet.backends.get_backend(t=None)

    Si `t` est None, récupère le backend de calcul actuel.
    Si `t` est un QTensor, retourne le backend utilisé pour créer le QTensor en fonction de sa propriété ``data``.
    Si "torch" est le backend, retourne le backend torchAPI de pyvqnet.
    Si "pyvqnet" est le backend, retourne simplement "pyvqnet".
    
    :param t: Le tenseur actuel (par défaut : None).
    :return: Le backend. Par défaut, retourne "pyvqnet".

    Example::

        import pyvqnet
        pyvqnet.backends.set_backend("torch")
        pyvqnet.backends.get_backend()

Fonctions QTensor
===================

Après avoir défini le backend sur ``torch`` :

.. code-block::

    import pyvqnet
    pyvqnet.backends.set_backend("torch")

Toutes les fonctions membres, fonctions de création, fonctions mathématiques, fonctions logiques, transformations matricielles, etc., sous :ref:`qtensor_api` utiliseront torch pour le calcul. Le `QTensor.data` peut être consulté pour récupérer les données torch.

Modules de réseau neuronal classique et de réseau neuronal quantique variationnel
==========================================================================================

Classe de base
------------------------------------------------

TorchModule
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. py:class:: pyvqnet.nn.torch.TorchModule(*args, **kwargs)

    La classe de base qui définit les modèles lors de l'utilisation du backend ``torch``. Cette classe hérite à la fois de ``pyvqnet.nn.Module`` et de ``torch.nn.Module``.
    Elle peut être ajoutée comme sous-module à un modèle torch.

    .. note::

        Cette classe et ses classes dérivées ne sont adaptées qu'à une utilisation avec ``pyvqnet.backends.set_backend("torch")``.
        Ne pas mélanger avec le ``Module`` par défaut de ``pyvqnet.nn``.
    
        Les données dans les ``_buffers`` de cette classe sont de type ``torch.Tensor``.
        Les données dans les ``_parameters`` de cette classe sont de type ``torch.nn.Parameter``.

    .. py:method:: pyvqnet.nn.torch.TorchModule.forward(x, *args, **kwargs)

        Fonction de calcul forward abstraite pour la classe TorchModule.

        :param x: QTensor d'entrée.
        :param args: Arguments variables non nommés.
        :param kwargs: Arguments variables nommés.

        :return: QTensor de sortie, dont le `data` interne est un ``torch.Tensor``.

        Example::

            import numpy as np
            from pyvqnet.tensor import QTensor
            import pyvqnet
            pyvqnet.backends.set_backend("torch")
            from pyvqnet.nn.torch import Conv2D
            b = 2
            ic = 3
            oc = 2
            test_conv = Conv2D(ic, oc, (3, 3), (2, 2), "valid")
            x0 = QTensor(np.arange(1, b * ic * 5 * 5 + 1).reshape([b, ic, 5, 5]),
                        requires_grad=True,
                        dtype=pyvqnet.kfloat32)
            x = test_conv.forward(x0)
            print(x)

    .. py:method:: pyvqnet.nn.torch.TorchModule.state_dict(destination=None, prefix='')

        Retourne un dictionnaire contenant l'état complet du module, y compris les paramètres et les valeurs des buffers.
        Les clés sont les noms des paramètres et buffers correspondants.

        :param destination: Le dictionnaire dans lequel stocker les paramètres internes du module.
        :param prefix: Un préfixe utilisé pour les noms des paramètres et buffers.

        :return: Un dictionnaire contenant l'état complet du module.

        Example::

            from pyvqnet.nn.torch import Conv2D
            import pyvqnet
            pyvqnet.backends.set_backend("torch")
            test_conv = Conv2D(2,3,(3,3),(2,2),"same")
            print(test_conv.state_dict().keys())

    .. py:method:: pyvqnet.nn.torch.TorchModule.load_state_dict(state_dict, strict=True)

        Copie les paramètres et buffers depuis :attr:`state_dict` dans ce module et ses sous-modules.

        :param state_dict: Un dictionnaire contenant les paramètres et les buffers persistants.
        :param strict: Si True, vérifie que les clés du state_dict correspondent exactement au `state_dict()` du modèle. Par défaut : True.

        :return: Un message d'erreur en cas de problème.

        Examples::

            from pyvqnet.nn.torch import TorchModule, Conv2D
            import pyvqnet
            import pyvqnet.utils
            pyvqnet.backends.set_backend("torch")

            class Net(TorchModule):
                def __init__(self):
                    super(Net, self).__init__()
                    self.conv1 = Conv2D(input_channels=1, output_channels=6, kernel_size=(5, 5),
                        stride=(1, 1), padding="valid")

                def forward(self, x):
                    return super().forward(x)

            model = Net()
            pyvqnet.utils.storage.save_parameters(model.state_dict(), "tmp.model")
            model_param = pyvqnet.utils.storage.load_parameters("tmp.model")
            model.load_state_dict(model_param)

    .. py:method:: pyvqnet.nn.torch.TorchModule.toGPU(device: int = DEV_GPU_0)

        Déplace le module et ses sous-modules (paramètres et données des buffers) vers le périphérique GPU spécifié.

        Le périphérique spécifie où les données internes sont stockées. Lorsque device >= DEV_GPU_0, les données sont stockées sur le GPU.
        Si votre ordinateur possède plusieurs GPU, vous pouvez spécifier différents périphériques pour stocker les données. Par exemple, device = DEV_GPU_1, DEV_GPU_2, DEV_GPU_3, ... fait référence au stockage sur des GPU avec différents numéros de série.
        
        .. note::

            Les modules ne peuvent pas effectuer de calculs sur différents GPU.
            Si vous tentez de créer un QTensor sur un ID GPU dépassant le maximum autorisé pour la validation, une erreur Cuda sera générée.

        :param device: Le périphérique sur lequel stocker le QTensor. Par défaut : DEV_GPU_0. device = pyvqnet.DEV_GPU_0 stocke sur le premier GPU, device = DEV_GPU_1 stocke sur le deuxième GPU, etc.
        :return: Le module déplacé sur le périphérique GPU.

        Examples::

            from pyvqnet.nn.torch import ConvT2D
            import pyvqnet
            pyvqnet.backends.set_backend("torch")
            test_conv = ConvT2D(3, 2, [4, 4], [2, 2], (0, 0))
            test_conv = test_conv.toGPU()
            print(test_conv.backend)
            #1000

    .. py:method:: pyvqnet.torch.TorchModule.toCPU()

        Déplace le module et ses sous-modules (paramètres et données des buffers) vers un périphérique CPU spécifique.

        :return: Le module déplacé sur le périphérique CPU.

        Examples::

            from pyvqnet.nn.torch import ConvT2D
            import pyvqnet
            pyvqnet.backends.set_backend("torch")
            test_conv = ConvT2D(3, 2, [4, 4], [2, 2], (0, 0))
            test_conv = test_conv.toCPU()
            print(test_conv.backend)
            #0


TorchModuleList
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. py:class:: pyvqnet.nn.torch.TorchModuleList(modules = None)

    Ce module est utilisé pour stocker des instances enfants ``TorchModule`` dans une liste. TorchModuleList peut être indexé comme une liste Python classique, et les paramètres internes qu'il contient peuvent être sauvegardés.
    
    Cette classe hérite de ``pyvqnet.nn.torch.TorchModule`` et ``pyvqnet.nn.ModuleList``, et peut être ajoutée comme sous-module à un modèle torch.

    :param modules: Une liste de ``pyvqnet.nn.torch.TorchModule``

    :return: Une classe TorchModuleList

    Example::

        from pyvqnet.tensor import *
        from pyvqnet.nn.torch import TorchModule, Linear, TorchModuleList

        import pyvqnet
        pyvqnet.backends.set_backend("torch")

        class M(TorchModule):
            def __init__(self):
                super(M, self).__init__()
                self.pqc2 = TorchModuleList([Linear(4, 1), Linear(4, 1)])

            def forward(self, x):
                y = self.pqc2[0](x) + self.pqc2[1](x)
                return y

        mm = M()

TorchParameterList
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. py:class:: pyvqnet.nn.torch.TorchParameterList(value=None)

    Ce module est utilisé pour stocker des instances enfants ``pyvqnet.nn.Parameter`` dans une liste. TorchParameterList peut être indexé comme une liste Python classique, et les paramètres internes qu'il contient peuvent être sauvegardés.
    
    Cette classe hérite de ``pyvqnet.nn.torch.TorchModule`` et ``pyvqnet.nn.ParameterList``, et peut être ajoutée comme sous-module à un modèle torch.

    :param value: Une liste de ``nn.Parameter``

    :return: Une classe TorchParameterList

    Example::

        from pyvqnet.tensor import *
        from pyvqnet.nn.torch import TorchModule, Linear, TorchParameterList
        import pyvqnet.nn as nn
        import pyvqnet
        pyvqnet.backends.set_backend("torch")

        class MyModule(TorchModule):
            def __init__(self):
                super().__init__()
                self.params = TorchParameterList([nn.Parameter((10, 10)) for i in range(10)])

            def forward(self, x):
                # ParameterList peut agir comme un itérable, ou être indexé avec des entiers
                for i, p in enumerate(self.params):
                    x = self.params[i // 2] * x + p * x
                return x

        model = MyModule()
        print(model.state_dict().keys())

TorchSequential
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. py:class:: pyvqnet.nn.torch.TorchSequential(*args)

    Le module ajoute les modules dans l'ordre dans lequel ils sont passés. Vous pouvez également passer un ``OrderedDict`` de modules. La méthode ``forward()`` de la classe ``Sequential`` accepte n'importe quelle entrée et la transmet à son premier module.
    La sortie est ensuite séquentiellement liée à l'entrée de chaque module suivant, la sortie finale étant le résultat du dernier module.

    Cette classe hérite de ``pyvqnet.nn.torch.TorchModule`` et ``pyvqnet.nn.Sequential``, et peut être ajoutée comme sous-module à un modèle torch.

    :param args: Modules à ajouter

    :return: Une classe TorchSequential

    Example::

        import pyvqnet
        from collections import OrderedDict
        from pyvqnet.tensor import *
        from pyvqnet.nn.torch import TorchModule, Conv2D, ReLu, \
            TorchSequential
        pyvqnet.backends.set_backend("torch")
        model = TorchSequential(
                    Conv2D(1, 20, (5, 5)),
                    ReLu(),
                    Conv2D(20, 64, (5, 5)),
                    ReLu()
                )
        print(model.state_dict().keys())

        model = TorchSequential(OrderedDict([
                    ('conv1', Conv2D(1, 20, (5, 5))),
                    ('relu1', ReLu()),
                    ('conv2', Conv2D(20, 64, (5, 5))),
                    ('relu2', ReLu())
                ]))
        print(model.state_dict().keys())

Sauvegarde et chargement des paramètres du modèle
---------------------------------------------------

Vous pouvez utiliser :ref:`save_parameters` ``save_parameters`` et ``load_parameters`` pour sauvegarder les paramètres d'un modèle ``TorchModule`` sous forme de dictionnaire dans un fichier, les valeurs étant sauvegardées sous forme de `numpy.ndarray`. Vous pouvez également charger le fichier de paramètres depuis le disque. Notez que la structure du modèle n'est pas sauvegardée dans le fichier, et vous devrez reconstruire manuellement la structure du modèle. Vous pouvez également utiliser directement ``torch.save`` et ``torch.load`` pour lire les paramètres du modèle ``torch``, car les paramètres de ``TorchModule`` sont stockés sous forme d'objets ``torch.Tensor``.


Modules de réseau neuronal classique
--------------------------------------------

Les modules de réseau neuronal classique suivants héritent tous de ``pyvqnet.nn.Module`` et ``torch.nn.Module``, et peuvent être ajoutés comme sous-modules à un modèle torch.

Linear
^^^^^^^^^^^^^^^^^^^^^^^^

.. py:class:: pyvqnet.nn.torch.Linear(input_channels, output_channels, weight_initializer=None, bias_initializer=None, use_bias=True, dtype=None, name: str = "")

    Un module linéaire (couche entièrement connectée), :math:`y = x@A.T + b`.
    This class inherits from ``pyvqnet.nn.Module`` and ``torch.nn.Module`` and can be used as a submodule of a torchmodel.

    Les données dans les ``_buffers`` de la classe sont de type ``torch.Tensor``.
    Les données dans les ``_parameters`` de la classe sont de type ``torch.nn.Parameter``.

    :param input_channels: `int` - Le nombre de canaux d'entrée.
    :param output_channels: `int` - Le nombre de canaux de sortie.
    :param weight_initializer: `callable` - Fonction d'initialisation des poids, par défaut vide, utilise he_uniform.
    :param bias_initializer: `callable` - Fonction d'initialisation du biais, par défaut vide, utilise he_uniform.
    :param use_bias: `bool` - Utiliser ou non le terme de biais, par défaut True.
    :param dtype: Type de données pour les paramètres, par défaut None, utilise le type de données par défaut ``kfloat32``, qui représente des nombres à virgule flottante 32 bits.
    :param name: Le nom de la couche linéaire, par défaut "".

    :return: Une instance de la couche Linear.

    Example::

        import numpy as np
        import pyvqnet
        from pyvqnet.tensor import QTensor
        from pyvqnet.nn.torch import Linear
        pyvqnet.backends.set_backend("torch")
        c1 = 2
        c2 = 3
        cin = 7
        cout = 5
        n = Linear(cin, cout)
        input = QTensor(np.arange(1, c1 * c2 * cin + 1).reshape((c1, c2, cin)), requires_grad=True, dtype=pyvqnet.kfloat32)
        y = n.forward(input)
        print(y)

Conv1D
^^^^^^^^^^^^^^^^^^^^^^^^

.. py:class:: pyvqnet.nn.torch.Conv1D(input_channels: int, output_channels: int, kernel_size: int, stride: int = 1, padding = "valid", use_bias: bool = True, kernel_initializer = None, bias_initializer = None, dilation_rate: int = 1, group: int = 1, dtype = None, name = "")

    Effectue une convolution 1D sur l'entrée. L'entrée du module Conv1D a la forme (batch_size, input_channels, in_height).
    This class inherits from ``pyvqnet.nn.Module`` and ``torch.nn.Module``, and can be used as a submodule of a torchmodel.

    Les données dans les ``_buffers`` de la classe sont de type ``torch.Tensor``.
    Les données dans les ``_parameters`` de la classe sont de type ``torch.nn.Parameter``.

    :param input_channels: `int` - Le nombre de canaux d'entrée.
    :param output_channels: `int` - The number of output channels.
    :param kernel_size: `int` - La taille du noyau de convolution. La forme du noyau est [output_channels, input_channels/group, kernel_size, 1].
    :param stride: `int` - Le pas, par défaut 1.
    :param padding: `str|int` - Options de padding, peut être une chaîne {'valid', 'same'} ou un entier spécifiant la quantité de padding à appliquer à l'entrée. Par défaut "valid".
    :param use_bias: `bool` - Utiliser ou non le terme de biais, par défaut True.
    :param kernel_initializer: `callable` - La méthode d'initialisation du noyau de convolution. Par défaut vide, utilise kaiming_uniform.
    :param bias_initializer: `callable` - La méthode d'initialisation du biais. Par défaut vide, utilise kaiming_uniform.
    :param dilation_rate: `int` - La taille de dilatation, par défaut 1.
    :param group: `int` - Le nombre de groupes dans la convolution groupée. Par défaut 1.
    :param dtype: Type de données pour les paramètres, par défaut None, utilise le type de données par défaut ``kfloat32``, qui représente des nombres à virgule flottante 32 bits.
    :param name: Le nom du module, par défaut "".

    :return: Une instance de la convolution 1D.

    .. note::

        ``padding='valid'`` n'applique pas de padding.

        ``padding='same'`` applique un padding nul à l'entrée, avec ``out_height`` égal à ``ceil(in_height / stride)``, et ne supporte pas ``stride > 1``.

    Example::

        import numpy as np
        import pyvqnet
        from pyvqnet.tensor import QTensor
        from pyvqnet.nn.torch import Conv1D
        pyvqnet.backends.set_backend("torch")
        b = 2
        ic = 3
        oc = 2
        test_conv = Conv1D(ic, oc, 3, 2)
        x0 = QTensor(np.arange(1, b * ic * 5 * 5 + 1).reshape([b, ic, 25]), requires_grad=True, dtype=pyvqnet.kfloat32)
        x = test_conv.forward(x0)
        print(x)

Conv2D
^^^^^^^^^^^^^^^^^^^^^^^^

.. py:class:: pyvqnet.nn.torch.Conv2D(input_channels: int, output_channels: int, kernel_size: tuple, stride: tuple = (1, 1), padding = "valid", use_bias = True, kernel_initializer = None, bias_initializer = None, dilation_rate: int = 1, group: int = 1, dtype = None, name = "")

    Effectue une convolution 2D sur l'entrée. L'entrée du module Conv2D a la forme (batch_size, input_channels, height, width).
    This class inherits from ``pyvqnet.nn.Module`` and ``torch.nn.Module``, and can be used as a submodule of a torchmodel.

    Les données dans les ``_buffers`` de la classe sont de type ``torch.Tensor``.
    Les données dans les ``_parameters`` de la classe sont de type ``torch.nn.Parameter``.

    :param input_channels: `int` - The number of input channels.
    :param output_channels: `int` - The number of output channels.
    :param kernel_size: `tuple|list` - La taille du noyau de convolution. La forme du noyau est [output_channels, input_channels/group, kernel_size, kernel_size].
    :param stride: `tuple|list` - Le pas, par défaut (1, 1).
    :param padding: `str|tuple` - Options de padding, peut être une chaîne {'valid', 'same'} ou un tuple spécifiant le padding à appliquer aux deux côtés. Par défaut "valid".
    :param use_bias: `bool` - Whether to use the bias term, default is True.
    :param kernel_initializer: `callable` - The convolution kernel initialization method. Default is empty, using kaiming_uniform.
    :param bias_initializer: `callable` - The bias initialization method. Default is empty, using kaiming_uniform.
    :param dilation_rate: `int` - The dilation size, default is 1.
    :param group: `int` - The number of groups in the grouped convolution. Default is 1.
    :param dtype: Data type for the parameters, defaults to None, uses the default data type `kfloat32`, which represents 32-bit floating point numbers.
    :param name: The name of the module, default is "".

    :return: Une instance de la convolution 2D.

    .. note::

        ``padding='valid'`` does not apply padding.

        ``padding='same'`` applique un padding nul à l'entrée, avec la hauteur de sortie égale à ``ceil(in_height / stride)``.

    Example::

        import numpy as np
        import pyvqnet
        from pyvqnet.tensor import QTensor
        from pyvqnet.nn.torch import Conv2D
        pyvqnet.backends.set_backend("torch")
        b = 2
        ic = 3
        oc = 2
        test_conv = Conv2D(ic, oc, (3, 3), (2, 2))
        x0 = QTensor(np.arange(1, b * ic * 5 * 5 + 1).reshape([b, ic, 5, 5]), requires_grad=True, dtype=pyvqnet.kfloat32)
        x = test_conv.forward(x0)
        print(x)

ConvT2D
^^^^^^^^^^^^^^^^^^^^^^^^

.. py:class:: pyvqnet.nn.torch.ConvT2D(input_channels, output_channels, kernel_size, stride=[1, 1], padding=(0, 0), use_bias="True", kernel_initializer=None, bias_initializer=None, dilation_rate: int = 1, out_padding=(0, 0), group=1, dtype=None, name="")

    Effectue une convolution transposée 2D sur l'entrée. L'entrée du module ConvT2D a la forme (batch_size, input_channels, height, width).
    This class inherits from ``pyvqnet.nn.Module`` and ``torch.nn.Module``, and can be used as a submodule of a torchmodel.

    Les données dans les ``_buffers`` de la classe sont de type ``torch.Tensor``.
    Les données dans les ``_parameters`` de la classe sont de type ``torch.nn.Parameter``.

    :param input_channels: `int` - The number of input channels.
    :param output_channels: `int` - The number of output channels.
    :param kernel_size: `tuple|list` - La taille du noyau de convolution, avec kernel shape = [input_channels, output_channels/group, kernel_size, kernel_size].
    :param stride: `tuple|list` - The stride, default is (1, 1).
    :param padding: `tuple` - Options de padding, un tuple spécifiant le padding à appliquer aux deux côtés. Par défaut (0, 0).
    :param use_bias: `bool` - Whether to use the bias term, default is True.
    :param kernel_initializer: `callable` - The convolution kernel initialization method. Default is empty, using kaiming_uniform.
    :param bias_initializer: `callable` - The bias initialization method. Default is empty, using kaiming_uniform.
    :param dilation_rate: `int` - The dilation size, default is 1.
    :param out_padding: Taille supplémentaire ajoutée à la forme de la sortie pour chaque dimension. Par défaut (0, 0).
    :param group: `int` - The number of groups in the grouped convolution. Default is 1.
    :param dtype: Data type for the parameters, defaults to None, uses the default data type `kfloat32`, which represents 32-bit floating point numbers.
    :param name: The name of the module, default is "".

    :return: Une instance de la convolution transposée 2D.

    .. note::

        ``padding='valid'`` does not apply padding.

        ``padding='same'`` applique un padding nul à l'entrée, avec la hauteur de sortie égale à ``ceil(height / stride)``.

    Example::

        import numpy as np
        import pyvqnet
        from pyvqnet.tensor import QTensor
        from pyvqnet.nn.torch import ConvT2D
        pyvqnet.backends.set_backend("torch")
        test_conv = ConvT2D(3, 2, (3, 3), (1, 1))
        x = QTensor(np.arange(1, 1 * 3 * 5 * 5 + 1).reshape([1, 3, 5, 5]), requires_grad=True, dtype=pyvqnet.kfloat32)
        y = test_conv.forward(x)
        print(y)

AvgPool1D
^^^^^^^^^^^^^^^^^^^^^^^^

.. py:class:: pyvqnet.nn.torch.AvgPool1D(kernel, stride, padding=0, name = "")

    Effectue un pooling moyen sur une entrée 1D. L'entrée a la forme (batch_size, input_channels, in_height).
    This class inherits from ``pyvqnet.nn.Module`` and ``torch.nn.Module``, and can be added as a submodule to a torchmodel.

    Les données dans les ``_buffers`` de la classe sont de type ``torch.Tensor``.
    Les données dans les ``_parameters`` de la classe sont de type ``torch.nn.Parameter``.

    :param kernel: La taille de la fenêtre de pooling.
    :param stride: La taille du pas pour le déplacement de la fenêtre.
    :param padding: Option de padding, un entier spécifiant la longueur du padding. Par défaut 0.
    :param name: The name of the module, default is "".

    :return: Une instance de la couche de pooling moyen 1D.

    Example::

        import numpy as np
        import pyvqnet
        from pyvqnet.tensor import QTensor
        from pyvqnet.nn.torch import AvgPool1D
        pyvqnet.backends.set_backend("torch")
        test_mp = AvgPool1D([3],[2],0)
        x = QTensor(np.array([0, 1, 0, 4, 5,
                             2, 3, 2, 1, 3,
                             4, 4, 0, 4, 3,
                             2, 5, 2, 6, 4,
                             1, 0, 0, 5, 7], dtype=float).reshape([1, 5, 5]), requires_grad=True)
        y = test_mp.forward(x)
        print(y)

MaxPool1D
^^^^^^^^^^^^^^^^^^^^^^^^

.. py:class:: pyvqnet.nn.torch.MaxPool1D(kernel, stride, padding=0, name="")

    Effectue un pooling maximal sur une entrée 1D. The input has the shape (batch_size, input_channels, in_height).
    This class inherits from ``pyvqnet.nn.Module`` and ``torch.nn.Module``, and can be added as a submodule to a torchmodel.

    Les données dans les ``_buffers`` de la classe sont de type ``torch.Tensor``.
    Les données dans les ``_parameters`` de la classe sont de type ``torch.nn.Parameter``.

    :param kernel: The size of the pooling window.
    :param stride: The step size for moving the window.
    :param padding: Padding option, an integer specifying the padding length. Default is 0.
    :param name: The name of the module, default is "".

    :return: Une instance de la couche de pooling maximal 1D.

    Example::

        import numpy as np
        import pyvqnet
        from pyvqnet.tensor import QTensor
        from pyvqnet.nn.torch import MaxPool1D
        pyvqnet.backends.set_backend("torch")
        test_mp = MaxPool1D([3],[2],0)
        x = QTensor(np.array([0, 1, 0, 4, 5,
                             2, 3, 2, 1, 3,
                             4, 4, 0, 4, 3,
                             2, 5, 2, 6, 4,
                             1, 0, 0, 5, 7], dtype=float).reshape([1, 5, 5]), requires_grad=True)
        y = test_mp.forward(x)
        print(y)

AvgPool2D
^^^^^^^^^^^^^^^^^^^^^^^^

.. py:class:: pyvqnet.nn.torch.AvgPool2D(kernel, stride, padding=(0,0), name="")

    Effectue un pooling moyen sur une entrée 2D. L'entrée a la forme (batch_size, input_channels, height, width).
    This class inherits from ``pyvqnet.nn.Module`` and ``torch.nn.Module``, and can be added as a submodule to a torchmodel.

    Les données dans les ``_buffers`` de la classe sont de type ``torch.Tensor``.
    Les données dans les ``_parameters`` de la classe sont de type ``torch.nn.Parameter``.

    :param kernel: The size of the pooling window.
    :param stride: The step size for moving the window.
    :param padding: Option de padding, un tuple contenant deux entiers spécifiant le padding pour les deux dimensions. Par défaut (0,0).
    :param name: The name of the module, default is "".

    :return: Une instance de la couche de pooling moyen 2D.

    Example::

        import numpy as np
        import pyvqnet
        from pyvqnet.tensor import QTensor
        from pyvqnet.nn.torch import AvgPool2D
        pyvqnet.backends.set_backend("torch")
        test_mp = AvgPool2D([2, 2], [2, 2], 1)
        x = QTensor(np.array([0, 1, 0, 4, 5,
                             2, 3, 2, 1, 3,
                             4, 4, 0, 4, 3,
                             2, 5, 2, 6, 4,
                             1, 0, 0, 5, 7], dtype=float).reshape([1, 1, 5, 5]), requires_grad=True)
        y = test_mp.forward(x)
        print(y)

MaxPool2D
^^^^^^^^^^^^^^^^^^^^^^^^

.. py:class:: pyvqnet.nn.torch.MaxPool2D(kernel, stride, padding=(0,0), name="")

    Effectue un pooling maximal sur une entrée 2D. The input has the shape (batch_size, input_channels, height, width).
    This class inherits from ``pyvqnet.nn.Module`` and ``torch.nn.Module``, and can be added as a submodule to a torchmodel.

    Les données dans les ``_buffers`` de la classe sont de type ``torch.Tensor``.
    Les données dans les ``_parameters`` de la classe sont de type ``torch.nn.Parameter``.

    :param kernel: The size of the pooling window.
    :param stride: The step size for moving the window.
    :param padding: Padding option, a tuple containing two integers specifying padding for both dimensions. Default is (0,0).
    :param name: The name of the module, default is "".

    :return: Une instance de la couche de pooling maximal 2D.

    Example::

        import numpy as np
        import pyvqnet
        from pyvqnet.tensor import QTensor
        from pyvqnet.nn.torch import MaxPool2D
        pyvqnet.backends.set_backend("torch")
        test_mp = MaxPool2D([2, 2], [2, 2], (0, 0))
        x = QTensor(np.array([0, 1, 0, 4, 5,
                             2, 3, 2, 1, 3,
                             4, 4, 0, 4, 3,
                             2, 5, 2, 6, 4,
                             1, 0, 0, 5, 7], dtype=float).reshape([1, 1, 5, 5]), requires_grad=True)
        y = test_mp.forward(x)
        print(y)

Embedding
^^^^^^^^^^^^^^^^^^^^^^^^

.. py:class:: pyvqnet.nn.torch.Embedding(num_embeddings, embedding_dim, weight_initializer=xavier_normal, dtype=None, name: str = "")

    Ce module est généralement utilisé pour stocker des plongements de mots et les récupérer à l'aide d'indices. L'entrée du module est une liste d'indices, et la sortie correspond aux plongements de mots correspondants.
    L'entrée de cette couche doit être de type ``kint64``. 
    This class inherits from ``pyvqnet.nn.Module`` and ``torch.nn.Module``, and can be added as a submodule to a torchmodel.

    Les données dans les ``_buffers`` de la classe sont de type ``torch.Tensor``.
    Les données dans les ``_parameters`` de la classe sont de type ``torch.nn.Parameter``.

    :param num_embeddings: `int` - La taille du dictionnaire de plongement.
    :param embedding_dim: `int` - La taille de chaque vecteur de plongement.
    :param weight_initializer: `callable` - La méthode d'initialisation des poids, par défaut Xavier Normal.
    :param dtype: Le type de données pour les paramètres, par défaut None, utilise le type de données par défaut : ``kfloat32`` (virgule flottante 32 bits).
    :param name: Le nom de la couche de plongement, par défaut "".

    :return: Une instance de la couche Embedding.

    Example::

        import numpy as np
        import pyvqnet
        from pyvqnet.tensor import QTensor
        from pyvqnet.nn.torch import Embedding
        pyvqnet.backends.set_backend("torch")
        vlayer = Embedding(30, 3)
        x = QTensor(np.arange(1, 25).reshape([2, 3, 2, 2]), dtype=pyvqnet.kint64)
        y = vlayer(x)
        print(y)

BatchNorm2d
^^^^^^^^^^^^^^^^^^^^^^^^

.. py:class:: pyvqnet.nn.torch.BatchNorm2d(channel_num:int, momentum:float=0.1, epsilon:float = 1e-5, affine=True, beta_initializer=zeros, gamma_initializer=ones, dtype=None, name="")

    Applique la normalisation par lot sur une entrée 4D (B, C, H, W). Voir l'article
    `Batch Normalization: Accelerating Deep Network Training by Reducing
    Internal Covariate Shift <https://arxiv.org/abs/1502.03167>`__.

    This class inherits from ``pyvqnet.nn.Module`` and ``torch.nn.Module``, and can be added as a submodule to a torchmodel.

    Les données dans les ``_buffers`` de la classe sont de type ``torch.Tensor``.
    Les données dans les ``_parameters`` de la classe sont de type ``torch.nn.Parameter``.

    .. math::
        y = \frac{x - \mathrm{E}[x]}{\sqrt{\mathrm{Var}[x] + \epsilon}} * \gamma + \beta

    où :math:`\gamma` et :math:`\beta` sont des paramètres entraînables. De plus, par défaut, pendant l'entraînement, la couche continue d'estimer la moyenne et la variance, qui sont ensuite utilisées pour la normalisation lors de l'évaluation. La valeur du momentum pour les moyennes mobiles est définie par défaut sur 0.1.

    :param channel_num: `int` - The number of input channels.
    :param momentum: `float` - Momentum pour le calcul de la moyenne mobile, par défaut 0.1.
    :param epsilon: `float` - Une petite constante pour la stabilité numérique, par défaut 1e-5.
    :param affine: `bool` - Si True, inclut des paramètres affines entraînables pour chaque canal. Par défaut ``True``, initialise les paramètres à 1 pour les poids et 0 pour les biais.
    :param beta_initializer: `callable` - La méthode d'initialisation pour bêta, par défaut initialisation à zéro.
    :param gamma_initializer: `callable` - La méthode d'initialisation pour gamma, par défaut initialisation à un.
    :param dtype: Le type de données pour les paramètres, par défaut None, utilise ``kfloat32`` (virgule flottante 32 bits).
    :param name: Le nom de la couche de normalisation par lot, par défaut "".

    :return: Une instance de la couche de normalisation par lot 2D.

    Example::

        import numpy as np
        import pyvqnet
        from pyvqnet.tensor import QTensor
        from pyvqnet.nn.torch import BatchNorm2d
        pyvqnet.backends.set_backend("torch")
        b = 2
        ic = 2
        test_conv = BatchNorm2d(ic)
        x = QTensor(np.arange(1, 17).reshape([b, ic, 4, 1]), requires_grad=True, dtype=pyvqnet.kfloat32)
        y = test_conv.forward(x)
        print(y)

BatchNorm1d
^^^^^^^^^^^^^^^^^^^^^^^^

.. py:class:: pyvqnet.nn.torch.BatchNorm1d(channel_num:int, momentum:float=0.1, epsilon:float = 1e-5, affine=True, beta_initializer=zeros, gamma_initializer=ones, dtype=None, name="")

    Applique la normalisation par lot sur une entrée 2D (B, C). Voir l'article
    `Batch Normalization: Accelerating Deep Network Training by Reducing
    Internal Covariate Shift <https://arxiv.org/abs/1502.03167>`__.

    .. math::
        y = \frac{x - \mathrm{E}[x]}{\sqrt{\mathrm{Var}[x] + \epsilon}} * \gamma + \beta

    where :math:`\gamma` and :math:`\beta` are trainable parameters. Additionally, by default, during training, the layer continues to estimate the mean and variance, which are then used for normalization during evaluation. The momentum for the moving averages is set to the default value of 0.1.

    This class inherits from ``pyvqnet.nn.Module`` and ``torch.nn.Module``, and can be added as a submodule to a torchmodel.

    Les données dans les ``_buffers`` de la classe sont de type ``torch.Tensor``.
    Les données dans les ``_parameters`` de la classe sont de type ``torch.nn.Parameter``.

    :param channel_num: `int` - The number of input channels.
    :param momentum: `float` - Momentum for the moving average calculation, default is 0.1.
    :param epsilon: `float` - A small constant for numerical stability, default is 1e-5.
    :param affine: `bool` - Whether to include learnable affine parameters for each channel. Default is `True`, which initializes the parameters as 1 for weights and 0 for biases.
    :param beta_initializer: `callable` - The initialization method for beta, default is zero initialization.
    :param gamma_initializer: `callable` - The initialization method for gamma, default is one initialization.
    :param dtype: The data type for the parameters, defaults to None, using `kfloat32` (32-bit floating point).
    :param name: The name of the batch normalization layer, default is "".

    :return: Une instance de la couche de normalisation par lot 1D.

    Example::

        import numpy as np
        import pyvqnet
        from pyvqnet.tensor import QTensor
        from pyvqnet.nn.torch import BatchNorm1d
        pyvqnet.backends.set_backend("torch")
        test_conv = BatchNorm1d(4)
        x = QTensor(np.arange(1, 17).reshape([4, 4]), requires_grad=True, dtype=pyvqnet.kfloat32)
        y = test_conv.forward(x)
        print(y)

LayerNormNd
^^^^^^^^^^^^^^^^^^^^^^^^

.. py:class:: pyvqnet.nn.torch.LayerNormNd(normalized_shape: list, epsilon: float = 1e-5, affine=True, dtype=None, name="")

    Applique la normalisation de couche sur les D dernières dimensions de n'importe quelle entrée. La méthode spécifique est décrite dans l'article :
    `Layer Normalization <https://arxiv.org/abs/1607.06450>`__.

    .. math::
        y = \frac{x - \mathrm{E}[x]}{ \sqrt{\mathrm{Var}[x] + \epsilon}} * \gamma + \beta

    Pour des entrées comme (B, C, H, W, D), le ``norm_shape`` peut être [C, H, W, D], [H, W, D], [W, D] ou [D].

    This class inherits from ``pyvqnet.nn.Module`` and ``torch.nn.Module``, and can be added as a submodule to a torchmodel.

    Les données dans les ``_buffers`` de la classe sont de type ``torch.Tensor``.
    Les données dans les ``_parameters`` de la classe sont de type ``torch.nn.Parameter``.

    :param norm_shape: `list` - La forme à normaliser.
    :param epsilon: `float` - A small constant for numerical stability, default is 1e-5.
    :param affine: `bool` - Si ``True``, ce module a des paramètres affines entraînables pour chaque canal, initialisés à 1 (pour les poids) et 0 (pour les biais). Par défaut ``True``.
    :param dtype: The data type for the parameters, defaults to None, using `kfloat32` (32-bit floating point).
    :param name: The name of the module, default is "".

    :return: Une instance de la classe LayerNormNd.

    Example::

        import numpy as np
        from pyvqnet.tensor import QTensor
        from pyvqnet import kfloat32
        from pyvqnet.nn.torch import LayerNormNd
        import pyvqnet
        pyvqnet.backends.set_backend("torch")
        ic = 4
        test_conv = LayerNormNd([2,2])
        x = QTensor(np.arange(1,17).reshape([2,2,2,2]), requires_grad=True, dtype=kfloat32)
        y = test_conv.forward(x)
        print(y)

LayerNorm2d
^^^^^^^^^^^^^^^^^^^^^^^^

.. py:class:: pyvqnet.nn.torch.LayerNorm2d(norm_size:int, epsilon:float = 1e-5, affine=True, dtype=None, name="")

    Applique la normalisation de couche sur des entrées 4D. The specific method is described in the paper:
    `Layer Normalization <https://arxiv.org/abs/1607.06450>`__.

    .. math::
        y = \frac{x - \mathrm{E}[x]}{ \sqrt{\mathrm{Var}[x] + \epsilon}} * \gamma + \beta

    La moyenne et l'écart type sont calculés sur les dimensions restantes, excluant la première. Pour des entrées comme (B, C, H, W), ``norm_size`` doit être égal à C * H * W.

    This class inherits from ``pyvqnet.nn.Module`` and ``torch.nn.Module``, and can be added as a submodule to a torchmodel.

    Les données dans les ``_buffers`` de la classe sont de type ``torch.Tensor``.
    Les données dans les ``_parameters`` de la classe sont de type ``torch.nn.Parameter``.

    :param norm_size: `int` - La taille de la normalisation, doit être égale à C * H * W.
    :param epsilon: `float` - A small constant for numerical stability, default is 1e-5.
    :param affine: `bool` - If `True`, this module has learnable affine parameters for each channel, initialized to 1 (for weights) and 0 (for biases). Default is `True`.
    :param dtype: The data type for the parameters, defaults to None, using `kfloat32` (32-bit floating point).
    :param name: The name of the module, default is "".

    :return: Une instance de la normalisation de couche 2D.

    Example::

        import numpy as np
        import pyvqnet
        from pyvqnet.tensor import QTensor
        from pyvqnet.nn.torch import LayerNorm2d
        import pyvqnet
        pyvqnet.backends.set_backend("torch")
        ic = 4
        test_conv = LayerNorm2d(8)
        x = QTensor(np.arange(1,17).reshape([2,2,4,1]), requires_grad=True, dtype=pyvqnet.kfloat32)
        y = test_conv.forward(x)
        print(y)

LayerNorm1d
^^^^^^^^^^^^^^^^^^^^^^^^

.. py:class:: pyvqnet.nn.torch.LayerNorm1d(norm_size:int, epsilon:float = 1e-5, affine=True, dtype=None, name="")

    Applique la normalisation de couche sur des entrées 2D. The specific method is described in the paper:
    `Layer Normalization <https://arxiv.org/abs/1607.06450>`__.

    .. math::
        y = \frac{x - \mathrm{E}[x]}{ \sqrt{\mathrm{Var}[x] + \epsilon}} * \gamma + \beta

    La moyenne et l'écart type sont calculés sur la taille de la dernière dimension, où ``norm_size`` est la valeur de la dernière dimension.

    This class inherits from ``pyvqnet.nn.Module`` and ``torch.nn.Module``, and can be added as a submodule to a torchmodel.

    Les données dans les ``_buffers`` de la classe sont de type ``torch.Tensor``.
    Les données dans les ``_parameters`` de la classe sont de type ``torch.nn.Parameter``.

    :param norm_size: `int` - La taille de la normalisation, doit être égale à la taille de la dernière dimension.
    :param epsilon: `float` - A small constant for numerical stability, default is 1e-5.
    :param affine: `bool` - If `True`, this module has learnable affine parameters for each channel, initialized to 1 (for weights) and 0 (for biases). Default is `True`.
    :param dtype: The data type for the parameters, defaults to None, using `kfloat32` (32-bit floating point).
    :param name: The name of the module, default is "".

    :return: Une instance de la normalisation de couche 1D.

    Example::

        import numpy as np
        from pyvqnet.tensor import QTensor
        from pyvqnet.nn.torch import LayerNorm1d
        import pyvqnet
        pyvqnet.backends.set_backend("torch")
        test_conv = LayerNorm1d(4)
        x = QTensor(np.arange(1,17).reshape([4,4]), requires_grad=True, dtype=pyvqnet.kfloat32)
        y = test_conv.forward(x)
        print(y)

GroupNorm
^^^^^^^^^^^^^^^^^^^^^^^^

.. py:class:: pyvqnet.nn.torch.GroupNorm(num_groups: int, num_channels: int, epsilon = 1e-5, affine = True, dtype = None, name = "")

    Applique la normalisation de groupe sur des entrées de mini-lots. Entrée : :math:`(N, C, *)` où :math:`C=\mathrm{num\_channels}`, Sortie : :math:`(N, C, *)`.

    Cette couche implémente l'opération décrite dans l'article `Group Normalization <https://arxiv.org/abs/1803.08494>`__.

    .. math::
        
        y = \frac{x - \mathrm{E}[x]}{ \sqrt{\mathrm{Var}[x] + \epsilon}} * \gamma + \beta

    Les canaux d'entrée sont divisés en :attr:`num_groups` groupes, chacun contenant ``num_channels / num_groups`` canaux. :attr:`num_channels` doit être divisible par :attr:`num_groups`. La moyenne et l'écart type sont calculés séparément dans chaque groupe. Si :attr:`affine` est ``True``, alors :math:`\gamma` et :math:`\beta` sont entraînables. Les paramètres de transformation affine pour chaque canal sont des vecteurs de taille :attr:`num_channels`.

    This class inherits from ``pyvqnet.nn.Module`` and ``torch.nn.Module``, and can be added as a submodule to a torchmodel.

    Les données dans les ``_buffers`` de la classe sont de type ``torch.Tensor``.
    Les données dans les ``_parameters`` de la classe sont de type ``torch.nn.Parameter``.

    :param num_groups (int): Le nombre de groupes dans lequel diviser les canaux.
    :param num_channels (int): Le nombre de canaux d'entrée attendus.
    :param epsilon: Une petite valeur ajoutée au dénominateur pour la stabilité numérique. Par défaut 1e-5.
    :param affine: Une valeur booléenne. Si ``True``, ce module a des paramètres affines entraînables pour chaque canal, initialisés à 1 (pour les poids) et 0 (pour les biais). Par défaut ``True``.
    :param dtype: The data type for the parameters. Defaults to None, using `kfloat32` (32-bit floating point).
    :param name: The name of the module. Default is "".

    :return: An instance of the GroupNorm class.

    Example::

        import numpy as np
        from pyvqnet.tensor import QTensor, kfloat32
        from pyvqnet.nn.torch import GroupNorm
        import pyvqnet
        pyvqnet.backends.set_backend("torch")
        test_conv = GroupNorm(2, 10)
        x = QTensor(np.arange(0, 60*2*5).reshape([2, 10, 3, 2, 5]), requires_grad=True, dtype=kfloat32)
        y = test_conv.forward(x)
        print(y)

Dropout
^^^^^^^^^^^^^^^^^^^^^^^^

.. py:class:: pyvqnet.nn.torch.Dropout(dropout_rate = 0.5)

    Module Dropout. Le module Dropout met aléatoirement à zéro la sortie de certaines unités, tout en mettant à l'échelle les unités restantes selon la probabilité dropout_rate donnée.

    This class inherits from ``pyvqnet.nn.Module`` and ``torch.nn.Module``, and can be added as a submodule to a torchmodel.

    :param dropout_rate: `float` - La probabilité de mettre les neurones à zéro.
    :param name: The name of the module. Default is "".

    :return: Une instance de la classe Dropout.

    Example::

        import numpy as np
        from pyvqnet.nn.torch import Dropout
        from pyvqnet.tensor import arange
        import pyvqnet
        pyvqnet.backends.set_backend("torch")
        b = 2
        ic = 2
        x = arange(-1 * ic * 2 * 2.0, (b - 1) * ic * 2 * 2).reshape([b, ic, 2, 2])
        droplayer = Dropout(0.5)
        droplayer.train()
        y = droplayer(x)
        print(y)

DropPath
^^^^^^^^^^^^^^^^^^^^^^^^

.. py:class:: pyvqnet.nn.torch.DropPath(dropout_rate = 0.5, name="")

    Le module DropPath applique un dropout aléatoire de chemin (profondeur aléatoire).

    This class inherits from ``pyvqnet.nn.Module`` and ``torch.nn.Module``, and can be added as a submodule to a torchmodel.

    :param dropout_rate: `float` - The probability of setting neurons to zero.
    :param name: The name of the module. Default is "".

    :return: Une instance de la classe DropPath.

    Example::

        import pyvqnet.nn.torch as nn
        import pyvqnet.tensor as tensor
        import pyvqnet
        pyvqnet.backends.set_backend("torch")
        x = tensor.randu([4])
        y = nn.DropPath()(x)
        print(y)

Pixel_Shuffle
^^^^^^^^^^^^^^^^^^^^^^^^

.. py:class:: pyvqnet.nn.torch.Pixel_Shuffle(upscale_factors, name="")

    Réorganise un tenseur de forme : (*, C * r^2, H, W) en un tenseur de forme (*, C, H * r, W * r), où r est le facteur d'échelle.

    This class inherits from ``pyvqnet.nn.Module`` and ``torch.nn.Module``, and can be added as a submodule to a torchmodel.

    :param upscale_factors: Le facteur d'échelle pour la transformation.
    :param name: The name of the module. Default is "".

    :return: Une instance du module Pixel_Shuffle.

    Example::

        from pyvqnet.nn.torch import Pixel_Shuffle
        from pyvqnet.tensor import tensor
        import pyvqnet
        pyvqnet.backends.set_backend("torch")
        ps = Pixel_Shuffle(3)
        inx = tensor.ones([5, 2, 3, 18, 4, 4])
        inx.requires_grad = True
        y = ps(inx)

Pixel_Unshuffle
^^^^^^^^^^^^^^^^^^^^^^^^

.. py:class:: pyvqnet.nn.torch.Pixel_Unshuffle(downscale_factors, name="")

    Inverse l'opération Pixel_Shuffle en réorganisant les éléments. Transforme un tenseur de forme (*, C, H * r, W * r) en (*, C * r^2, H, W), où r est le facteur de réduction.

    This class inherits from ``pyvqnet.nn.Module`` and ``torch.nn.Module``, and can be added as a submodule to a torchmodel.

    :param downscale_factors: Le facteur de réduction pour la transformation.
    :param name: The name of the module. Default is "".

    :return: Une instance du module Pixel_Unshuffle.

    Example::

        from pyvqnet.nn.torch import Pixel_Unshuffle
        from pyvqnet.tensor import tensor
        import pyvqnet
        pyvqnet.backends.set_backend("torch")
        ps = Pixel_Unshuffle(3)
        inx = tensor.ones([5, 2, 3, 2, 12, 12])
        inx.requires_grad = True
        y = ps(inx)

GRU
^^^^^^^^^^^^^^^^^^^^^^^^

.. py:class:: pyvqnet.nn.torch.GRU(input_size, hidden_size, num_layers=1, nonlinearity='tanh', batch_first=True, use_bias=True, bidirectional=False, dtype=None, name: str = "")

    Module d'unité récurrente à portes (GRU). Supporte l'empilement multicouche et la configuration bidirectionnelle. La formule pour un GRU unidirectionnel monocouche est :

    .. math::
        \begin{array}{ll}
            r_t = \sigma(W_{ir} x_t + b_{ir} + W_{hr} h_{(t-1)} + b_{hr}) \\
            z_t = \sigma(W_{iz} x_t + b_{iz} + W_{hz} h_{(t-1)} + b_{hz}) \\
            n_t = \tanh(W_{in} x_t + b_{in} + r_t * (W_{hn} h_{(t-1)}+ b_{hn})) \\
            h_t = (1 - z_t) * n_t + z_t * h_{(t-1)}
        \end{array}

    This class inherits from ``pyvqnet.nn.Module`` and ``torch.nn.Module``, and can be added as a submodule to a torchmodel.

    Les ``_buffers`` de la classe contiennent des données ``torch.Tensor``, et les ``_parameters`` de la classe contiennent des données ``torch.nn.Parameter``.

    :param input_size: La dimension des caractéristiques d'entrée.
    :param hidden_size: La dimension des caractéristiques cachées.
    :param num_layers: Le nombre de couches GRU empilées, par défaut : 1.
    :param batch_first: Si True, la forme d'entrée est [batch_size, seq_len, feature_dim], si False, la forme est [seq_len, batch_size, feature_dim], par défaut : True.
    :param use_bias: Si False, le module n'utilise pas de termes de biais, par défaut : True.
    :param bidirectional: Si True, rend le GRU bidirectionnel, par défaut : False.
    :param dtype: Le type de données des paramètres, par défaut None, utilise le type de données par défaut : ``kfloat32`` (flottant 32 bits).
    :param name: Le nom du module, par défaut : "".

    :return: Une instance du module GRU.

    Example::

        import pyvqnet
        pyvqnet.backends.set_backend("torch")
        from pyvqnet.nn.torch import GRU
        from pyvqnet.tensor import tensor

        rnn2 = GRU(4, 6, 2, batch_first=False, bidirectional=True)

        input = tensor.ones([5, 3, 4])
        h0 = tensor.ones([4, 3, 6])

        output, hn = rnn2(input, h0)



RNN
^^^^^^^^^^^^^^^^^^^^^^^^

.. py:class:: pyvqnet.nn.torch.RNN(input_size, hidden_size, num_layers=1, nonlinearity='tanh', batch_first=True, use_bias=True, bidirectional=False, dtype=None, name: str = "")

    Module de réseau neuronal récurrent (RNN), utilisant :math:`\tanh` ou :math:`\text{ReLU}` comme fonction d'activation. Supporte les configurations bidirectionnelles et multicouches. La formule pour un RNN unidirectionnel monocouche est :

    .. math::
        h_t = \tanh(W_{ih} x_t + b_{ih} + W_{hh} h_{(t-1)} + b_{hh})

    Si :attr:`nonlinearity` est ``'relu'``, alors :math:`\text{ReLU}` remplacera :math:`\tanh`.

    This class inherits from ``pyvqnet.nn.Module`` and ``torch.nn.Module``, and can be added as a submodule to a torchmodel.

    Les ``_buffers`` de la classe contiennent des données ``torch.Tensor``, et les ``_parameters`` de la classe contiennent des données ``torch.nn.Parameter``.

    :param input_size: The input feature dimension.
    :param hidden_size: The hidden feature dimension.
    :param num_layers: Le nombre de couches RNN empilées, par défaut : 1.
    :param nonlinearity: La fonction d'activation non linéaire, par défaut : ``'tanh'``.
    :param batch_first: If True, the input shape is [batch_size, seq_len, feature_dim], if False, the shape is [seq_len, batch_size, feature_dim], default: True.
    :param use_bias: If False, the module does not use bias terms, default: True.
    :param bidirectional: Si True, rend le RNN bidirectionnel, par défaut : False.
    :param dtype: The data type of the parameters, defaults to None, using the default data type: kfloat32 (32-bit float).
    :param name: Le nom du module, par défaut : "".

    :return: Une instance du module RNN.

    Example::

        import pyvqnet
        pyvqnet.backends.set_backend("torch")
        from pyvqnet.nn.torch import RNN
        from pyvqnet.tensor import tensor

        rnn2 = RNN(4, 6, 2, batch_first=False, bidirectional = True)

        input = tensor.ones([5, 3, 4])
        h0 = tensor.ones([4, 3, 6])
        output, hn = rnn2(input, h0)

LSTM
^^^^^^^^^^^^^^^^^^^^^^^^

.. py:class:: pyvqnet.nn.torch.LSTM(input_size, hidden_size, num_layers=1, batch_first=True, use_bias=True, bidirectional=False, dtype=None, name: str = "")

    Module de mémoire à long et court terme (LSTM). Supporte les configurations LSTM bidirectionnelles et LSTM multicouche empilée. La formule pour un LSTM unidirectionnel monocouche est la suivante :

    .. math::
        \begin{array}{ll} \\
            i_t = \sigma(W_{ii} x_t + b_{ii} + W_{hi} h_{t-1} + b_{hi}) \\
            f_t = \sigma(W_{if} x_t + b_{if} + W_{hf} h_{t-1} + b_{hf}) \\
            g_t = \tanh(W_{ig} x_t + b_{ig} + W_{hg} h_{t-1} + b_{hg}) \\
            o_t = \sigma(W_{io} x_t + b_{io} + W_{ho} h_{t-1} + b_{ho}) \\
            c_t = f_t \odot c_{t-1} + i_t \odot g_t \\
            h_t = o_t \odot \tanh(c_t) \\
        \end{array}

    This class inherits from ``pyvqnet.nn.Module`` and ``torch.nn.Module``, and can be added as a submodule to a torchmodel.

    Les ``_buffers`` de la classe contiennent des données ``torch.Tensor``, et les ``_parameters`` de la classe contiennent des données ``torch.nn.Parameter``.

    :param input_size: The input feature dimension.
    :param hidden_size: The hidden feature dimension.
    :param num_layers: Le nombre de couches LSTM empilées, par défaut : 1.
    :param batch_first: If True, the input shape is [batch_size, seq_len, feature_dim], if False, the shape is [seq_len, batch_size, feature_dim], default: True.
    :param use_bias: If False, the module does not use bias terms, default: True.
    :param bidirectional: Si True, rend le LSTM bidirectionnel, par défaut : False.
    :param dtype: The data type of the parameters, defaults to None, using the default data type: kfloat32 (32-bit float).
    :param name: The name of the module, default: "".

    :return: Une instance du module LSTM.

    Example::

        import pyvqnet
        pyvqnet.backends.set_backend("torch")
        from pyvqnet.nn.torch import LSTM
        from pyvqnet.tensor import tensor

        rnn2 = LSTM(4, 6, 2, batch_first=False, bidirectional = True)

        input = tensor.ones([5, 3, 4])
        h0 = tensor.ones([4, 3, 6])
        c0 = tensor.ones([4, 3, 6])
        output, (hn, cn) = rnn2(input, (h0, c0))


Dynamic_GRU
^^^^^^^^^^^^^^^^^^^^^^^^

.. py:class:: pyvqnet.nn.torch.Dynamic_GRU(input_size,hidden_size, num_layers=1, batch_first=True, use_bias=True, bidirectional=False, dtype=None, name: str = "")

    Applique un réseau neuronal récurrent GRU multicouche à des séquences d'entrée de longueur dynamique.

    La première entrée doit être une entrée de séquence par lot avec une longueur variable définie
    via une classe ``tensor.PackedSequence``.

    La classe ``tensor.PackedSequence`` peut être construite en
    appelant les fonctions suivantes consécutivement : ``pad_sequence``, ``pack_pad_sequence``.

    La première sortie de Dynamic_GRU est également une classe ``tensor.PackedSequence``,
    qui peut être décompressée en un QTensor normal en utilisant ``tensor.pad_pack_sequence``.

    Pour chaque élément de la séquence d'entrée, chaque couche calcule la formule suivante :

    .. math::
        \begin{array}{ll}
        r_t = \sigma(W_{ir} x_t + b_{ir} + W_{hr} h_{(t-1)} + b_{hr}) \\
        z_t = \sigma(W_{iz} x_t + b_{iz} + W_{hz} h_{(t-1)} + b_{hz}) \\
        n_t = \tanh(W_{in} x_t + b_{in} + r_t * (W_{hn} h_{(t-1)}+ b_{hn})) \\
        h_t = (1 - z_t) * n_t + z_t * h_{(t-1)}
        \end{array}

    Cette classe hérite de ``pyvqnet.nn.Module`` et ``torch.nn.Module`` et peut être ajoutée au modèle torch comme sous-module de ``torch.nn.Module``.

    Les données dans ``_buffers`` de cette classe sont de type ``torch.Tensor``.

    Les données dans ``_parameters`` de cette classe sont de type ``torch.nn.Parameter``.

    :param input_size: Dimension des caractéristiques d'entrée.
    :param hidden_size: Dimension des caractéristiques cachées.
    :param num_layers: Nombre de couches récurrentes. Valeur par défaut : 1
    :param batch_first: Si True, la forme d'entrée est [taille du lot, longueur de la séquence, dimension des caractéristiques]. Si False, la forme d'entrée est [longueur de la séquence, taille du lot, dimension des caractéristiques]. Valeur par défaut : True.
    :param use_bias: Si False, les poids de biais b_ih et b_hh ne sont pas utilisés pour cette couche. Valeur par défaut : True.
    :param bidirectional: Si True, devient un GRU bidirectionnel. Valeur par défaut : False.
    :param dtype: Le type de données du paramètre, par défaut : None, utilise le type de données par défaut : ``kfloat32``, représentant des nombres à virgule flottante 32 bits.
    :param name: Le nom de ce module, par défaut "".

    :return: Une classe Dynamic_GRU

    Example::

        import pyvqnet
        pyvqnet.backends.set_backend("torch")
        from pyvqnet.nn.torch import Dynamic_GRU
        from pyvqnet.tensor import tensor
        seq_len = [4,1,2]
        input_size = 4
        batch_size =3
        hidden_size = 2
        ml = 2
        rnn2 = Dynamic_GRU(input_size,
                        hidden_size=2,
                        num_layers=2,
                        batch_first=False,
                        bidirectional=True)

        a = tensor.arange(1, seq_len[0] * input_size + 1).reshape(
            [seq_len[0], input_size])
        b = tensor.arange(1, seq_len[1] * input_size + 1).reshape(
            [seq_len[1], input_size])
        c = tensor.arange(1, seq_len[2] * input_size + 1).reshape(
            [seq_len[2], input_size])

        y = tensor.pad_sequence([a, b, c], False)

        input = tensor.pack_pad_sequence(y,
                                        seq_len,
                                        batch_first=False,
                                        enforce_sorted=False)

        h0 = tensor.ones([ml * 2, batch_size, hidden_size])

        output, hn = rnn2(input, h0)

        seq_unpacked, lens_unpacked = \
        tensor.pad_packed_sequence(output, batch_first=False)

Dynamic_RNN 
^^^^^^^^^^^^^^^^^^^^^^^^

.. py:class:: pyvqnet.nn.torch.Dynamic_RNN(input_size, hidden_size, num_layers=1, nonlinearity='tanh', batch_first=True, use_bias=True, bidirectional=False, dtype=None, name: str = "")


    Applique un réseau neuronal récurrent (RNN) à une séquence d'entrée de longueur dynamique.

    La première entrée doit être une entrée de séquence par lot avec une longueur variable définie
    via la classe ``tensor.PackedSequence``.

    La classe ``tensor.PackedSequence`` peut être construite en
    appelant la fonction suivante successivement : ``pad_sequence``, ``pack_pad_sequence``.

    La première sortie de Dynamic_RNN est également une classe ``tensor.PackedSequence``,
    qui peut être décompressée en un QTensor normal en utilisant ``tensor.pad_pack_sequence``.

    Module de réseau neuronal récurrent (RNN), utilisant :math:`\tanh` ou :math:`\text{ReLU}` comme fonction d'activation. Supporte les configurations bidirectionnelles et multicouches.
    La formule de calcul du RNN unidirectionnel monocouche est la suivante :

    .. math::
        h_t = \tanh(W_{ih} x_t + b_{ih} + W_{hh} h_{(t-1)} + b_{hh})

    If :attr:`nonlinearity` is ``'relu'``, then :math:`\text{ReLU}` will replace :math:`\tanh`.

    Cette classe hérite de ``pyvqnet.nn.Module`` et ``torch.nn.Module``, et peut être ajoutée au modèle torch comme sous-module de ``torch.nn.Module``.

    Les données dans ``_buffers`` de cette classe sont de type ``torch.Tensor``.

    Les données dans ``_parmeters`` de cette classe sont de type ``torch.nn.Parameter``.

    :param input_size: Input feature dimension.
    :param hidden_size: Hidden feature dimension.
    :param num_layers: Nombre de couches RNN empilées, par défaut : 1.
    :param nonlinearity: Fonction d'activation non linéaire, par défaut ``'tanh'``.
    :param batch_first: Si True, la forme d'entrée est [taille du lot, longueur de la séquence, dimension des caractéristiques], si False, la forme d'entrée est [longueur de la séquence, taille du lot, dimension des caractéristiques], par défaut True.
    :param use_bias: Si False, ce module n'applique pas de biais, par défaut : True.
    :param bidirectional: Si True, devient un RNN bidirectionnel, par défaut : False.
    :param dtype: The data type of the parameter, default: None, use the default data type: kfloat32, representing 32-bit floating point numbers.
    :param name: Le nom de ce module, par défaut "".

    :return: Instance Dynamic_RNN

    Example::

        import pyvqnet
        pyvqnet.backends.set_backend("torch")
        from pyvqnet.nn.torch import Dynamic_RNN
        from pyvqnet.tensor import tensor
        seq_len = [4,1,2]
        input_size = 4
        batch_size =3
        hidden_size = 2
        ml = 2
        rnn2 = Dynamic_RNN(input_size,
                        hidden_size=2,
                        num_layers=2,
                        batch_first=False,
                        bidirectional=True,
                        nonlinearity='relu')

        a = tensor.arange(1, seq_len[0] * input_size + 1).reshape(
            [seq_len[0], input_size])
        b = tensor.arange(1, seq_len[1] * input_size + 1).reshape(
            [seq_len[1], input_size])
        c = tensor.arange(1, seq_len[2] * input_size + 1).reshape(
            [seq_len[2], input_size])

        y = tensor.pad_sequence([a, b, c], False)

        input = tensor.pack_pad_sequence(y,
                                        seq_len,
                                        batch_first=False,
                                        enforce_sorted=False)

        h0 = tensor.ones([ml * 2, batch_size, hidden_size])

        output, hn = rnn2(input, h0)

        seq_unpacked, lens_unpacked = \
        tensor.pad_packed_sequence(output, batch_first=False)




Dynamic_LSTM
^^^^^^^^^^^^^^^^^^^^^^^^

.. py:class:: pyvqnet.nn.torch.Dynamic_LSTM(input_size, hidden_size, num_layers=1, batch_first=True, use_bias=True, bidirectional=False, dtype=None, name: str = "")


    Applique un réseau neuronal récurrent LSTM (mémoire à long et court terme) à des séquences d'entrée de longueur dynamique.

    La première entrée doit être une entrée de séquence par lot avec une longueur variable définie
    via une classe ``tensor.PackedSequence``.

    La classe ``tensor.PackedSequence`` peut être construite en
    appelant les fonctions suivantes successivement : ``pad_sequence``, ``pack_pad_sequence``.

    La première sortie de Dynamic_LSTM est également une classe ``tensor.PackedSequence``,
    qui peut être décompressée en un QTensor normal en utilisant ``tensor.pad_pack_sequence``.

    Module de réseau neuronal récurrent (RNN), utilisant :math:`\tanh` ou :math:`\text{ReLU}` comme fonction d'activation. Supporte les configurations bidirectionnelles et multicouches.
    La formule de calcul du RNN unidirectionnel monocouche est la suivante : 
    
    
    .. math::
        \begin{array}{ll} \\
            i_t = \sigma(W_{ii} x_t + b_{ii} + W_{hi} h_{t-1} + b_{hi}) \\
            f_t = \sigma(W_{if} x_t + b_{if} + W_{hf} h_{t-1} + b_{hf}) \\
            g_t = \tanh(W_{ig} x_t + b_{ig} + W_{hg} h_{t-1} + b_{hg}) \\
            o_t = \sigma(W_{io} x_t + b_{io} + W_{ho} h_{t-1} + b_{ho}) \\
            c_t = f_t \odot c_{t-1} + i_t \odot g_t \\
            h_t = o_t \odot \tanh(c_t) \\
        \end{array}

    This class inherits from ``pyvqnet.nn.Module`` and ``torch.nn.Module``, and can be added to the torch model as a submodule of ``torch.nn.Module``.

    Les données dans ``_buffers`` de cette classe sont de type ``torch.Tensor``.

    Les données dans ``_parmeters`` de cette classe sont de type ``torch.nn.Parameter``.

    :param input_size: Input feature dimension.
    :param hidden_size: Hidden feature dimension.
    :param num_layers: Nombre de couches LSTM empilées, par défaut : 1.
    :param batch_first: Si True, la forme d'entrée est [taille du lot, longueur de la séquence, dimension des caractéristiques], si False, la forme d'entrée est [longueur de la séquence, taille du lot, dimension des caractéristiques], par défaut True.
    :param use_bias: If False, this module does not apply bias, default: True.
    :param bidirectional: Si True, devient un LSTM bidirectionnel, par défaut : False.
    :param dtype: The data type of the parameter, default: None, use the default data type: kfloat32, representing 32-bit floating point numbers.
    :param name: Le nom de ce module, par défaut "".

    :return: Instance Dynamic_LSTM

    Example::

        import pyvqnet
        pyvqnet.backends.set_backend("torch")
        from pyvqnet.nn.torch import Dynamic_LSTM
        from pyvqnet.tensor import tensor

        input_size = 2
        hidden_size = 2
        ml = 2
        seq_len = [3, 4, 1]
        batch_size = 3
        rnn2 = Dynamic_LSTM(input_size,
                            hidden_size=hidden_size,
                            num_layers=ml,
                            batch_first=False,
                            bidirectional=True)

        a = tensor.arange(1, seq_len[0] * input_size + 1).reshape(
            [seq_len[0], input_size])
        b = tensor.arange(1, seq_len[1] * input_size + 1).reshape(
            [seq_len[1], input_size])
        c = tensor.arange(1, seq_len[2] * input_size + 1).reshape(
            [seq_len[2], input_size])
        a.requires_grad = True
        b.requires_grad = True
        c.requires_grad = True
        y = tensor.pad_sequence([a, b, c], False)

        input = tensor.pack_pad_sequence(y,
                                        seq_len,
                                        batch_first=False,
                                        enforce_sorted=False)

        h0 = tensor.ones([ml * 2, batch_size, hidden_size])
        c0 = tensor.ones([ml * 2, batch_size, hidden_size])

        output, (hn, cn) = rnn2(input, (h0, c0))

        seq_unpacked, lens_unpacked = \
        tensor.pad_packed_sequence(output, batch_first=False)

 


Interpolate
^^^^^^^^^^^^^^^^^^^^^^^^

.. py:class:: pyvqnet.nn.torch.Interpolate(size = None, scale_factor = None, mode = "nearest", align_corners = None,  recompute_scale_factor = None, name = "")

    Sous-échantillonne ou sur-échantillonne l'entrée.

    Supporte actuellement uniquement les données d'entrée 4D.

    La taille d'entrée est interprétée comme `B x C x H x W`.

    Les options `mode` disponibles sont ``nearest``, ``bilinear``, ``bicubic``.

    This class inherits from ``pyvqnet.nn.Module`` and ``torch.nn.Module`` and can be added to the torch model as a submodule of ``torch.nn.Module``.

    :param size: Taille de sortie, par défaut None.
    :param scale_factor: Facteur d'échelle, par défaut None.
    :param mode: Algorithme utilisé pour le sur-échantillonnage ``nearest`` | ``bilinear`` | ``bicubic``.
    :param align_corners: From a geometric point of view, we treat the pixels of the input and output as squares instead of points. The pixels of the input and output are treated as squares instead of points.If set to `true`, the input and output tensors will be aligned by the center points of their corner pixels. Corner pixel center points are aligned, and the values ​​of the corner pixels are preserved.If set to `false`, the input and output tensors will be aligned by the corner points of their corner pixels, and the values ​​of the corner pixels are preserved. Corner pixel corner points are aligned, and interpolation will use edge values ​​for padding.Values ​​outside the boundaries are padded, making this operation independent of the input size.When ``scale_factor`` remains unchanged. This only works when ``mode`` is ``bilinear``.
    :param recompute_scale_factor: Recompute the scale factor for use in the interpolation calculation. When ``scale_factor`` is passed as an argument, it will be used to calculate the output size.
    :param name: Nom du module.

    Example::

        from pyvqnet.nn.torch import Interpolate
        from pyvqnet.tensor import tensor
        import pyvqnet
        pyvqnet.backends.set_backend("torch")
        pyvqnet.utils.set_random_seed(1)

        mode_ = "bilinear"
        size_ = 3

        model = Interpolate(size=size_, mode=mode_)
        input_vqnet = tensor.randu((1, 1, 6, 6),
                                dtype=pyvqnet.kfloat32,
                                requires_grad=True)
        output_vqnet = model(input_vqnet)

SDPA
^^^^^^^^^^^^^^^^^^^^^^^^

.. py:class:: pyvqnet.nn.torch.SDPA(attn_mask=None,dropout_p=0.,scale=None,is_causal=False)

    Construit une classe qui calcule l'attention produit scalaire pondérée pour les tenseurs de requête, clé et valeur.

    This class inherits from ``pyvqnet.nn.Module`` and ``torch.nn.Module``, and can be added to the torch model as a submodule of ``torch.nn.Module``.

    :param attn_mask: Masque d'attention ; valeur par défaut : None. La forme doit pouvoir être diffusée à la forme des poids d'attention.
    :param dropout_p: Probabilité de dropout ; valeur par défaut : 0, si supérieure à 0.0, un dropout est appliqué.
    :param scale: Facteur d'échelle appliqué avant softmax, valeur par défaut : None.
    :param is_causal: valeur par défaut : False, si défini sur true, le masque d'attention est une matrice triangulaire inférieure lorsque le masque est une matrice carrée. Si attn_mask et is_causal sont tous deux définis, une erreur est levée.
    :return: Une classe SDPA

    Examples::
    
        from pyvqnet.nn.torch import SDPA
        from pyvqnet import tensor
        import pyvqnet
        pyvqnet.backends.set_backend("torch")
        model = SDPA(tensor.QTensor([1.]))

   .. py:method:: forward(query,key,value)

        Effectue le calcul forward.

        :param query: Le QTensor d'entrée de la requête.
        :param key: Le QTensor d'entrée de la clé.
        :param value: Le QTensor d'entrée de la valeur.
        :return: Le QTensor retourné par le calcul SDPA.
        
        Examples::
        
            from pyvqnet.nn.torch import SDPA
            from pyvqnet import tensor
            import pyvqnet
            pyvqnet.backends.set_backend("torch")

            import numpy as np

            model = SDPA(tensor.QTensor([1.]))

            query_np = np.random.randn(3, 3, 3, 5).astype(np.float32) 
            key_np = np.random.randn(3, 3, 3, 5).astype(np.float32)   
            value_np = np.random.randn(3, 3, 3, 5).astype(np.float32) 

            query_p = tensor.QTensor(query_np, dtype=pyvqnet.kfloat32, requires_grad=True)
            key_p = tensor.QTensor(key_np, dtype=pyvqnet.kfloat32, requires_grad=True)
            value_p = tensor.QTensor(value_np, dtype=pyvqnet.kfloat32, requires_grad=True)

            out_sdpa = model(query_p, key_p, value_p)

            out_sdpa.backward()

API des fonctions de perte
--------------------------

MeanSquaredError
^^^^^^^^^^^^^^^^^^^^^^^^

.. py:class:: pyvqnet.nn.torch.MeanSquaredError(name="")

    Calcule l'erreur quadratique moyenne entre l'entrée :math:`x` et la valeur cible :math:`y`.

    Si l'erreur quadratique peut être décrite par la fonction suivante :

    .. math::
        \ell(x, y) = L = \{l_1,\dots,l_N\}^\top, \quad
        l_n = \left( x_n - y_n \right)^2,

    :math:`x` et :math:`y` sont des QTensor de formes quelconques, et l'erreur quadratique moyenne des :math:`n` éléments totaux est calculée comme suit.

    .. math::
        \ell(x, y) =
        \operatorname{mean}(L)

    This class inherits from ``pyvqnet.nn.Module`` and ``torch.nn.Module``, and can be added to the torch model as a submodule of ``torch.nn.Module``.

    :param name: The name of this module, defaults to "".
    :return: Une instance d'erreur quadratique moyenne.

    Paramètres requis pour la fonction de calcul forward de l'erreur quadratique moyenne :

        x: :math:`(N, *)` valeur prédite, où :math:`*` représente n'importe quelle dimension.

        y: :math:`(N, *)`, valeur cible, un QTensor de même dimension que l'entrée.

    .. note::

        Veuillez noter que contrairement à des frameworks comme pytorch, dans la fonction forward de la fonction MeanSquaredError suivante, le premier paramètre est la valeur cible et le second paramètre est la valeur prédite.


    Example::

        from pyvqnet.tensor import QTensor
        from pyvqnet import kfloat64
        from pyvqnet.nn.torch import MeanSquaredError
        import pyvqnet
        pyvqnet.backends.set_backend("torch")
        y = QTensor([[0, 0, 1, 0, 0, 0, 0, 0, 0, 0]],
                    requires_grad=False,
                    dtype=kfloat64)
        x = QTensor([[0.1, 0.05, 0.7, 0, 0.05, 0.1, 0, 0, 0, 0]],
                    requires_grad=True,
                    dtype=kfloat64)

        loss_result = MeanSquaredError()
        result = loss_result(y, x)
        print(result)



BinaryCrossEntropy
^^^^^^^^^^^^^^^^^^^^^^^^

.. py:class:: pyvqnet.nn.torch.BinaryCrossEntropy(name="")

    Mesure la perte d'entropie croisée binaire moyenne entre la cible et l'entrée.

    L'entropie croisée binaire sans moyenne est la suivante :

    .. math::
        \ell(x, y) = L = \{l_1,\dots,l_N\}^\top, \quad
        l_n = - w_n \left[ y_n \cdot \log x_n + (1 - y_n) \cdot \log (1 - x_n) \right],

    where :math:`N` is the batch size.

    .. math::
        \ell(x, y) = \operatorname{mean}(L)

    This class inherits from ``pyvqnet.nn.Module`` and ``torch.nn.Module`` and can be added to torch models as a submodule of ``torch.nn.Module``.

    :param name: The name of this module, defaults to "".
    :return: Une instance d'entropie croisée binaire moyenne.

    Paramètres requis pour la fonction de calcul forward de l'erreur d'entropie croisée binaire moyenne :

        x: :math:`(N, *)` predicted value, where :math:`*` represents any dimension.

        y: :math:`(N, *)`, target value, a QTensor of the same dimension as the input.

    .. note::

        Veuillez noter que contrairement à des frameworks comme pytorch, dans la fonction forward de la fonction BinaryCrossEntropy, le premier paramètre est la valeur cible et le second paramètre est la valeur prédite.
        
    Example::

        from pyvqnet.tensor import QTensor
        from pyvqnet.nn.torch import BinaryCrossEntropy
        import pyvqnet
        pyvqnet.backends.set_backend("torch")
        x = QTensor([[0.3, 0.7, 0.2], [0.2, 0.3, 0.1]], requires_grad=True)
        y = QTensor([[0.0, 1.0, 0], [0.0, 0, 1]], requires_grad=False)

        loss_result = BinaryCrossEntropy()
        result = loss_result(y, x)
        result.backward()
        print(result)


CategoricalCrossEntropy
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. py:class:: pyvqnet.nn.torch.CategoricalCrossEntropy(name="")

    Cette fonction de perte combine LogSoftmax et NLLLoss pour calculer l'entropie croisée catégorielle moyenne.

    La fonction de perte est calculée comme suit, où class est l'étiquette de catégorie correspondante de la valeur cible :

    .. math::
        \text{loss}(x, y) = -\log\left(\frac{\exp(x[class])}{\sum_j \exp(x[j])}\right)
        = -x[class] + \log\left(\sum_j \exp(x[j])\right)

    :param name: The name of this module, defaults to "".
    :return: L'instance d'entropie croisée catégorielle moyenne.

    Paramètres requis pour la fonction de calcul forward de l'erreur :

        x: :math:`(N, *)` Valeur prédite, où :math:`*` indique n'importe quelle dimension.

        y: :math:`(N, *)`, valeur cible, un QTensor de même dimension que l'entrée. Doit être un entier 64 bits, ``kint64``.

    .. note::

        Veuillez noter que contrairement à pytorch et autres frameworks, dans la fonction forward de la fonction CategoricalCrossEntropy, le premier paramètre est la valeur cible et le second paramètre est la valeur prédite.

        This class inherits from ``pyvqnet.nn.Module`` and ``torch.nn.Module``, and can be added to the torch model as a submodule of ``torch.nn.Module``.

    Example::

        from pyvqnet.tensor import QTensor
        from pyvqnet import kfloat32,kint64
        from pyvqnet.nn.torch import CategoricalCrossEntropy
        import pyvqnet
        pyvqnet.backends.set_backend("torch")
        x = QTensor([[1, 2, 3, 4, 5],
        [1, 2, 3, 4, 5],
        [1, 2, 3, 4, 5]], requires_grad=True,dtype=kfloat32)
        y = QTensor([[0, 1, 0, 0, 0], [0, 1, 0, 0, 0], [1, 0, 0, 0, 0]], requires_grad=False,dtype=kint64)
        loss_result = CategoricalCrossEntropy()
        result = loss_result(y, x)
        print(result)



SoftmaxCrossEntropy
^^^^^^^^^^^^^^^^^^^^^^^^

.. py:class:: pyvqnet.nn.torch.SoftmaxCrossEntropy(name="")

    Cette fonction de perte combine LogSoftmax et NLLLoss pour calculer l'entropie croisée de classification moyenne, et a une stabilité numérique plus élevée.

    La fonction de perte est calculée comme suit, où class est l'étiquette de classification correspondante de la valeur cible :

    .. math::
        \text{loss}(x, y) = -\log\left(\frac{\exp(x[class])}{\sum_j \exp(x[j])}\right)
        = -x[class] + \log\left(\sum_j \exp(x[j])\right)

    :param name: The name of this module, defaults to "".
    :return: Une instance de fonction de perte d'entropie croisée softmax

    Required parameters for the error forward calculation function:

        x: :math:`(N, *)` valeur prédite, où :math:`*` indique n'importe quelle dimension.

        y: :math:`(N, *)`, target value, a QTensor of the same dimension as the input. Must be a 64-bit integer, kint64.

    .. note::

        Veuillez noter que contrairement à pytorch et autres frameworks, dans la fonction forward de la fonction SoftmaxCrossEntropy, le premier paramètre est la valeur cible et le second paramètre est la valeur prédite.

        This class inherits from ``pyvqnet.nn.Module`` and ``torch.nn.Module``, and can be added to the torch model as a submodule of ``torch.nn.Module``.
        
    Example::

        from pyvqnet.tensor import QTensor
        from pyvqnet import kfloat32, kint64
        from pyvqnet.nn.torch import SoftmaxCrossEntropy
        import pyvqnet
        pyvqnet.backends.set_backend("torch")
        x = QTensor([[1, 2, 3, 4, 5], [1, 2, 3, 4, 5], [1, 2, 3, 4, 5]],
                    requires_grad=True,
                    dtype=kfloat32)
        y = QTensor([[0, 1, 0, 0, 0], [0, 1, 0, 0, 0], [1, 0, 0, 0, 0]],
                    requires_grad=False,
                    dtype=kint64)
        loss_result = SoftmaxCrossEntropy()
        result = loss_result(y, x)
        result.backward()
        print(result)



NLL_Loss
^^^^^^^^^^^^^^^^^^^^^^^^

.. py:class:: pyvqnet.nn.torch.NLL_Loss(name="")

    
    Perte de log-vraisemblance négative moyenne. Utile pour les problèmes de classification avec C classes.

    `x` est la vraisemblance de probabilité donnée par le modèle. Sa forme peut être :math:`(N, C)` ou :math:`(N, C, d_1, d_2, ..., d_K)`. `y` est la valeur réelle attendue de la fonction de perte, contenant les indices de classe dans :math:`[0, C-1]`.

    .. math::
        \ell(x, y) = L = \{l_1,\dots,l_N\}^\top, \quad
        l_n = -
        \sum_{n=1}^N \frac{1}{N}x_{n,y_n} \quad

    :param name: The name of this module, defaults to "".
    :return: Une instance de fonction de perte NLL_Loss

    Required parameters for the error forward calculation function:

        x: :math:`(N, *)`, la valeur de prédiction de sortie de la fonction de perte, qui peut être une variable multidimensionnelle.

        y: :math:`(N, *)`, la valeur cible de la fonction de perte. Doit être un entier 64 bits, ``kint64``.

    .. note::

        Veuillez noter que contrairement à des frameworks comme pytorch, dans la fonction forward de la fonction NLL_Loss, le premier paramètre est la valeur cible et le second paramètre est la valeur prédite.

        This class inherits from ``pyvqnet.nn.Module`` and ``torch.nn.Module``, and can be added to the torch model as a submodule of ``torch.nn.Module``.
            
    Example::

        from pyvqnet.tensor import QTensor
        from pyvqnet import kint64
        from pyvqnet.nn.torch import NLL_Loss
        import pyvqnet
        pyvqnet.backends.set_backend("torch")
        x = QTensor([
            0.9476322568516703, 0.226547421131723, 0.5944201443911326,
            0.42830868492969476, 0.76414068655387, 0.00286059168094277,
            0.3574236812873617, 0.9096948856639084, 0.4560809854582528,
            0.9818027091583286, 0.8673569904602182, 0.9860275114020933,
            0.9232667066664217, 0.303693313961628, 0.8461034903175555
        ])
        x=x.reshape([1, 3, 1, 5])
        x.requires_grad = True
        y = QTensor([[[2, 1, 0, 0, 2]]], dtype=kint64)

        loss_result = NLL_Loss()
        result = loss_result(y, x)
        print(result)


CrossEntropyLoss
^^^^^^^^^^^^^^^^^^^^^^^^

.. py:class:: pyvqnet.nn.torch.CrossEntropyLoss(name="")

    Cette fonction calcule la perte combinée de LogSoftmax et NLL_Loss.

    `x` contient la sortie non normalisée. Sa forme peut être :math:`(C)`, :math:`(N, C)` bidimensionnelle ou :math:`(N, C, d_1, d_2, ..., d_K)` multidimensionnelle.

    La formule de la fonction de perte est la suivante, où class est l'étiquette de classe correspondante de la valeur cible :

    .. math::
        \text{loss}(x, y) = -\log\left(\frac{\exp(x[class])}{\sum_j \exp(x[j])}\right)
        = -x[class] + \log\left(\sum_j \exp(x[j])\right)

    :param name: The name of this module, default is "".
    :return: Une instance de fonction de perte CrossEntropyLoss

    Required parameters for the error forward calculation function:

        x: :math:`(N, *)`, la sortie de la fonction de perte, qui peut être une variable multidimensionnelle.

        y: :math:`(N, *)`, la valeur réelle attendue de la fonction de perte. Doit être un entier 64 bits, ``kint64``.

    .. note::

        Veuillez noter que contrairement à des frameworks comme pytorch, dans la fonction forward de la fonction CrossEntropyLoss, le premier paramètre est la valeur cible et le second paramètre est la valeur prédite.

        This class inherits from ``pyvqnet.nn.Module`` and ``torch.nn.Module``, and can be added to the torch model as a submodule of ``torch.nn.Module``.

    Example::

        from pyvqnet.tensor import QTensor
        from pyvqnet import kint64
        from pyvqnet.nn.torch import CrossEntropyLoss
        import pyvqnet
        pyvqnet.backends.set_backend("torch")
        x = QTensor([
            0.9476322568516703, 0.226547421131723, 0.5944201443911326,
            0.42830868492969476, 0.76414068655387, 0.00286059168094277,
            0.3574236812873617, 0.9096948856639084, 0.4560809854582528,
            0.9818027091583286, 0.8673569904602182, 0.9860275114020933,
            0.9232667066664217, 0.303693313961628, 0.8461034903175555
        ])
        x=x.reshape([1, 3, 1, 5])
        x.requires_grad = True
        y = QTensor([[[2, 1, 0, 0, 2]]], dtype=kint64)

        loss_result = CrossEntropyLoss()
        result = loss_result(y, x)
        print(result)


Fonctions d'activation
-----------------------

Sigmoid
^^^^^^^^^^^^^^^^^^^^^^^^

.. py:class:: pyvqnet.nn.torch.Sigmoid(name:str="")

    Couche de fonction d'activation Sigmoid.

    .. math::
        \text{Sigmoid}(x) = \frac{1}{1 + \exp(-x)}

    This class inherits from ``pyvqnet.nn.Module`` and ``torch.nn.Module``, and can be added to the torch model as a submodule of ``torch.nn.Module``.

    :param name: Le nom de la couche de fonction d'activation, par défaut "".

    :return: Une instance de couche de fonction d'activation Sigmoid.

    Examples::

        from pyvqnet.nn.torch import Sigmoid
        from pyvqnet.tensor import QTensor
        import pyvqnet
        pyvqnet.backends.set_backend("torch")
        layer = Sigmoid()
        y = layer(QTensor([1.0, 2.0, 3.0, 4.0]))
        print(y)


Softplus
^^^^^^^^^^^^^^^^^^^^^^^^

.. py:class:: pyvqnet.nn.torch.Softplus(name:str="")

    Softplus 

    .. math::
        \text{Softplus}(x) = \log(1 + \exp(x))

    This class inherits from ``pyvqnet.nn.Module`` and ``torch.nn.Module``, and can be added to the torch model as a submodule of ``torch.nn.Module``.

    :param name: The name of the activation function layer, default is "".

    :return: Une instance Softplus.

    Examples::

        from pyvqnet.nn.torch import Softplus
        from pyvqnet.tensor import QTensor
        import pyvqnet
        pyvqnet.backends.set_backend("torch")
        layer = Softplus()
        y = layer(QTensor([1.0, 2.0, 3.0, 4.0]))

Softsign
^^^^^^^^^^^^^^^^^^^^^^^^

.. py:class:: pyvqnet.nn.torch.Softsign(name:str="")

    Softsign.

    .. math::
        \text{SoftSign}(x) = \frac{x}{ 1 + |x|}


    This class inherits from ``pyvqnet.nn.Module`` and ``torch.nn.Module``, and can be added to the torch model as a submodule of ``torch.nn.Module``.


    :param name: The name of the activation function layer, default is "".

    :return: Une instance SoftSign.

    Examples::

        from pyvqnet.nn.torch import Softsign
        from pyvqnet.tensor import QTensor
        import pyvqnet
        pyvqnet.backends.set_backend("torch")
        layer = Softsign()
        y = layer(QTensor([1.0, 2.0, 3.0, 4.0]))



Softmax
^^^^^^^^^^^^^^^^^^^^^^^^

.. py:class:: pyvqnet.nn.torch.Softmax(axis:int = -1,name:str="")

    Softmax 

    .. math::
        \text{Softmax}(x_{i}) = \frac{\exp(x_i)}{\sum_j \exp(x_j)}


    This class inherits from ``pyvqnet.nn.Module`` and ``torch.nn.Module``, and can be added to the torch model as a submodule of ``torch.nn.Module``.


    :param axis: la dimension sur laquelle calculer (le dernier axe est -1), valeur par défaut = -1.
    :param name: The name of the activation function layer, default is "".

    :return: Une instance Softmax.

    Examples::

        from pyvqnet.nn.torch import Softmax
        from pyvqnet.tensor import QTensor
        import pyvqnet
        pyvqnet.backends.set_backend("torch")
        layer = Softmax()
        y = layer(QTensor([1.0, 2.0, 3.0, 4.0]))


HardSigmoid
^^^^^^^^^^^^^^^^^^^^^^^^

.. py:class:: pyvqnet.nn.torch.HardSigmoid(name:str="")

    HardSigmoid 

    .. math::
        \text{Hardsigmoid}(x) = \begin{cases}
            0 & \text{ if } x \le -3, \\
            1 & \text{ if } x \ge +3, \\
            x / 6 + 1 / 2 & \text{otherwise}
        \end{cases}


    This class inherits from ``pyvqnet.nn.Module`` and ``torch.nn.Module``, and can be added to the torch model as a submodule of ``torch.nn.Module``.


    :param name: The name of the activation function layer, default is "".

    :return: Instance HardSigmoid.

    Examples::

        from pyvqnet.nn.torch import HardSigmoid
        from pyvqnet.tensor import QTensor
        import pyvqnet
        pyvqnet.backends.set_backend("torch")
        layer = HardSigmoid()
        y = layer(QTensor([1.0, 2.0, 3.0, 4.0]))


ReLu
^^^^^^^^^^^^^^^^^^^^^^^^

.. py:class:: pyvqnet.nn.torch.ReLu(name:str="")

    ReLu.

    .. math::
        \text{ReLu}(x) = \begin{cases}
        x, & \text{ if } x > 0\\
        0, & \text{ if } x \leq 0
        \end{cases}


    This class inherits from ``pyvqnet.nn.Module`` and ``torch.nn.Module``, and can be added to the torch model as a submodule of ``torch.nn.Module``.


    :param name: The name of the activation function layer, default is "".

    :return: Une instance ReLu.

    Examples::

        from pyvqnet.nn.torch import ReLu
        from pyvqnet.tensor import QTensor
        import pyvqnet
        pyvqnet.backends.set_backend("torch")
        layer = ReLu()
        y = layer(QTensor([-1, 2.0, -3, 4.0]))

        


LeakyReLu
^^^^^^^^^^^^^^^^^^^^^^^^

.. py:class:: pyvqnet.nn.torch.LeakyReLu(alpha:float=0.01,name:str="")

    LeakyReLu  

    .. math::
        \text{LeakyRelu}(x) =
        \begin{cases}
        x, & \text{ if } x \geq 0 \\
        \alpha * x, & \text{ otherwise }
        \end{cases}


    This class inherits from ``pyvqnet.nn.Module`` and ``torch.nn.Module``, and can be added to the torch model as a submodule of ``torch.nn.Module``.


    :param alpha: Coefficient LeakyRelu, par défaut : 0.01.
    :param name: The name of the activation function layer, default is "".

    :return: Une instance d'activation LeakyReLu.

    Examples::

        from pyvqnet.nn.torch import LeakyReLu
        from pyvqnet.tensor import QTensor
        import pyvqnet
        pyvqnet.backends.set_backend("torch")
        layer = LeakyReLu()
        y = layer(QTensor([-1, 2.0, -3, 4.0]))



Gelu
^^^^^^^^^^^^^^^^^^^^^^^^

.. py:class:: pyvqnet.nn.torch.Gelu(approximate="tanh", name="")
    
    Gelu:

    .. math:: \text{GELU}(x) = x * \Phi(x)

    Lorsque le paramètre d'approximation est 'tanh', GELU est estimé comme suit :

    .. math:: \text{GELU}(x) = 0.5 * x * (1 + \text{Tanh}(\sqrt{2 / \pi} * (x + 0.044715 * x^3)))


    This class inherits from ``pyvqnet.nn.Module`` and ``torch.nn.Module``, and can be added to the torch model as a submodule of ``torch.nn.Module``.


    :param approximate: Méthode de calcul approximatif, par défaut "tanh".
    :param name: The name of the activation function layer, default is "".

    :return: Instance d'activation Gelu.

    Examples::

        from pyvqnet.tensor import randu, ones_like
        from pyvqnet.nn.torch import Gelu
        import pyvqnet
        pyvqnet.backends.set_backend("torch")
        qa = randu([5,4])
        qb = Gelu()(qa)



ELU
^^^^^^^^^^^^^^^^^^^^^^^^

.. py:class:: pyvqnet.nn.torch.ELU(alpha:float=1,name:str="")

    Couche de fonction d'activation ELU (Exponential Linear Unit).

    .. math::
        \text{ELU}(x) = \begin{cases}
        x, & \text{ if } x > 0\\
        \alpha * (\exp(x) - 1), & \text{ if } x \leq 0
        \end{cases}


    This class inherits from ``pyvqnet.nn.Module`` and ``torch.nn.Module``, and can be added to the torch model as a submodule of ``torch.nn.Module``.



    :param alpha: Coefficient ELU, par défaut : 1.
    :param name: The name of the activation function layer, default is "".

    :return: Instance d'activation ELU.

    Examples::

        from pyvqnet.nn.torch import ELU
        from pyvqnet.tensor import QTensor
        import pyvqnet
        pyvqnet.backends.set_backend("torch")
        layer = ELU()
        y = layer(QTensor([-1, 2.0, -3, 4.0]))


Tanh
^^^^^^^^^^^^^^^^^^^^^^^^

.. py:class:: pyvqnet.nn.torch.Tanh(name:str="")

    Fonction d'activation tangente hyperbolique Tanh.

    .. math::
        \text{Tanh}(x) = \frac{\exp(x) - \exp(-x)} {\exp(x) + \exp(-x)}


    This class inherits from ``pyvqnet.nn.Module`` and ``torch.nn.Module``, and can be added to the torch model as a submodule of ``torch.nn.Module``.



    :param name: The name of the activation function layer, default is "".

    :return: Instance d'activation Tanh.

    Examples::

        from pyvqnet.nn.torch import Tanh
        from pyvqnet.tensor import QTensor
        import pyvqnet
        pyvqnet.backends.set_backend("torch")
        layer = Tanh()
        y = layer(QTensor([-1, 2.0, -3, 4.0]))



Module d'optimisation
---------------------------------------------

Pour les modules de circuits classiques et quantiques héritant de `TorchModule`, les paramètres `model.paramters()` peuvent continuer à être optimisés en utilisant des optimiseurs autres que `Rotosolve` sous :ref:`Optimizer`.



Utilisation de pyqpanda pour exécuter des circuits quantiques variationnels
---------------------------------------------------------------------------

Ce qui suit est l'interface de circuit quantique variationnel d'entraînement pour le calcul de circuits utilisant pyqpanda et pyqpanda3.

.. warning::

    La partie de calcul quantique de TorchQpandaQuantumLayer suivante utilise pyqpanda2.

    En raison des problèmes de compatibilité entre pyqpanda2 et pyqpanda3, vous devez installer pyqpnda2 vous-même, `pip install pyqpanda`

TorchQpandaQuantumLayer
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Si vous êtes plus familier avec la syntaxe pyQPanda2, vous pouvez utiliser l'interface TorchQpandaQuantumLayer, en ajoutant des bits quantiques personnalisés ``qubits``, des bits classiques ``cbits`` et le simulateur backend ``machine`` au paramètre de la fonction ``qprog_with_measure`` de TorchQpandaQuantumLayer.

.. py:class:: pyvqnet.qnn.vqc.torch.TorchQpandaQuantumLayer(qprog_with_measure,para_num,diff_method:str = "parameter_shift",delta:float = 0.01,dtype=None,name="")

    Module de calcul abstrait de la couche quantique variationnelle. Utilise pyQPanda2 pour simuler un circuit quantique paramétré et obtenir les résultats de mesure. Cette couche quantique variationnelle hérite du module de calcul de gradient du framework VQNet. Elle peut utiliser la méthode de dérive des paramètres pour calculer le gradient des paramètres du circuit, entraîner des modèles de circuit quantique variationnel ou intégrer des circuits quantiques variationnels dans des modèles hybrides quantiques et classiques.

    :param qprog_with_measure: Fonctions d'opération et de mesure de circuit quantique construites avec pyQPand.
    :param para_num: `int` - nombre de paramètres.
    :param diff_method: Méthode pour résoudre les gradients des paramètres du circuit quantique, "parameter shift" ou "finite difference", par défaut parameter shift.
    :param delta: \delta lors du calcul des gradients par différences finies.
    :param dtype: Type de données du paramètre, par défaut : None, utilise le type de données par défaut : ``kfloat32``, représentant des nombres à virgule flottante 32 bits.
    :param name: The name of this module, default is "".

    :return: Un module qui peut calculer des circuits quantiques.

    .. note::

        qprog_with_measure est une fonction de circuit quantique définie dans pyQPanda2.

        Cette fonction doit contenir les paramètres suivants comme entrée de fonction (même si un paramètre n'est pas réellement utilisé), sinon elle ne fonctionnera pas correctement dans cette fonction.

        Comparé à QuantumLayer, dans la fonction d'exécution de circuit variationnel passée par cette interface, l'utilisateur doit créer manuellement les bits quantiques et les simulateurs.

        Si qprog_with_measure nécessite une mesure quantique, l'utilisateur doit également créer et allouer manuellement les cbits.

        L'utilisation de la fonction de circuit quantique qprog_with_measure (input, param, nqubits, ncbits) peut être consultée dans l'exemple suivant.

        `input` : Entrée de données classiques unidimensionnelles. Si aucune, entrer None.

        `param` : Entrée des paramètres du circuit quantique variationnel unidimensionnel à entraîner.

    Example::

        import pyqpanda as pq
        from pyvqnet.qnn import ProbsMeasure
        import numpy as np
        from pyvqnet.tensor import QTensor
        import pyvqnet
        pyvqnet.backends.set_backend("torch")
        from pyvqnet.qnn.vqc.torch import TorchQpandaQuantumLayer
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

        pqc = TorchQpandaQuantumLayer(pqctest,3)

        # données classiques comme entrée
        input = QTensor([[1.0,2,3,4],[4,2,2,3],[3,3,2,2]],requires_grad=True)

        # circuit forward
        rlt = pqc(input)

        print(rlt)

        grad =  QTensor(np.ones(rlt.data.shape)*1000)
        # circuit backward
        rlt.backward(grad)

        print(pqc.m_para.grad)
        print(input.grad)


.. warning::

    La partie de calcul quantique des interfaces TorchQcloud3QuantumLayer et TorchQpanda3QuantumLayer suivantes utilise pyqpanda3.

    Si vous utilisez la fonction QCloud sous ce module, il y aura des erreurs lors de l'importation de pyqpanda2 dans le code ou de l'utilisation des interfaces de paquets liées à pyqpanda2 de pyvqnet.

TorchQcloud3QuantumLayer
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Lorsque vous installez la dernière version de pyqpanda3, vous pouvez utiliser cette interface pour définir un circuit variationnel et le soumettre à la puce réelle d'originqc pour exécution.

.. py:class:: pyvqnet.qnn.vqc.torch.TorchQcloud3QuantumLayer(origin_qprog_func, qcloud_token, para_num, pauli_str_dict=None, shots = 1000, initializer=None, dtype=None, name="", diff_method="parameter_shift", submit_kwargs={}, query_kwargs={})
    
    Un module de calcul abstrait pour les puces réelles utilisant originqc de pyqpanda3. Il soumet des circuits quantiques paramétrés à des puces réelles et obtient les résultats de mesure.
    Si diff_method == "random_coordinate_descent", la couche sélectionnera aléatoirement un seul paramètre pour calculer le gradient, et les autres paramètres resteront à zéro. Référence : https://arxiv.org/abs/2311.00088

    .. note::

        qcloud_token est le jeton API que vous avez demandé auprès de la plateforme cloud.

        origin_qprog_func doit retourner des données de type pypqanda3.core.QProg. Si pauli_str_dict n'est pas défini, il est nécessaire de s'assurer que la mesure a été insérée dans le QProg.

        origin_qprog_func doit être au format suivant :

        origin_qprog_func(input,param )

        `input` : Entrée de données classiques 1~2D. Dans le cas 2D, la première dimension est la taille du lot.

        `param` : Entrée des paramètres à entraîner du circuit quantique variationnel 1D.

    .. warning::

        Cette classe hérite de ``pyvqnet.nn.Module`` et ``torch.nn.Module``, et peut être ajoutée au modèle torch comme sous-module de ``torch.nn.Module``.

        Les données dans ``_buffers`` de cette classe sont de type ``torch.Tensor``.

        Les données dans ``_parmeters`` de cette classe sont de type ``torch.nn.Parameter``.

    :param origin_qprog_func: La fonction de circuit quantique variationnel construite par QPanda, qui doit retourner QProg.
    :param qcloud_token: `str` - Le type de machine quantique ou le jeton cloud pour l'exécution.
    :param para_num: `int` - Le nombre de paramètres, le paramètre est un QTensor de taille [para_num].
    :param pauli_str_dict: `dict|list` - Dictionnaire ou liste de dictionnaires représentant les opérateurs de Pauli dans les circuits quantiques. Par défaut "None", ce qui signifie que les opérations de mesure sont effectuées. Si un dictionnaire d'opérateurs de Pauli est saisi, une espérance unique ou des espérances multiples sont calculées.
    :param shot: `int` - Nombre de mesures. La valeur par défaut est 1000.
    :param initializer: Initialisateur pour les valeurs des paramètres. La valeur par défaut est "None", utilisant une distribution normale 0~2*pi.
    :param dtype: Type de données du paramètre. La valeur par défaut est None, ce qui signifie utiliser le type de données par défaut pyvqnet.kfloat32.
    :param name: Le nom du module. La valeur par défaut est une chaîne vide.
    :param diff_method: Méthode de différenciation pour le calcul du gradient. La valeur par défaut est "parameter_shift", "random_coordinate_descent".
    :param submit_kwargs: Paramètres supplémentaires pour la soumission de circuits quantiques, par défaut : {"if_print_qcloud_log":False,"chip_id":"WK_C180","is_amend":True,"is_mapping":True,"is_optimization":True,"compile_level":3,"default_task_group_size":200,"test_qcloud_fake":False,"":"server_ip_address"}, lorsque test_qcloud_fake est défini sur True, simulation CPUQVM locale.
    :param query_kwargs: Paramètres supplémentaires pour interroger les résultats quantiques, par défaut : {"timeout":2,"print_query_info":True,"sub_circuits_split_size":1}.
    :return: A module that can calculate quantum circuits.


    Example::

        import pyqpanda3.core as pq
        import pyvqnet
        from pyvqnet.qnn.vqc.torch import TorchQcloud3QuantumLayer

        pyvqnet.backends.set_backend("torch")
        def qfun(input,param):

            m_qlist = range(6)
            cbits = range(6)
            measure_qubits = [0,2]
            m_prog = pq.QProg()
            cir = pq.QCircuit()
            cir<<pq.RZ(m_qlist[0],input[0])
            cir<<pq.CNOT(m_qlist[0],m_qlist[1])
            cir<<pq.RY(m_qlist[1],param[0])
            cir<<pq.CNOT(m_qlist[0],m_qlist[2])
            cir<<pq.RZ(m_qlist[1],input[1])
            cir<<pq.RY(m_qlist[2],param[1])
            cir<<pq.H(m_qlist[2])
            m_prog<<cir

            for idx, ele in enumerate(measure_qubits):
                m_prog << pq.measure(m_qlist[ele], cbits[idx])  # pylint: disable=expression-not-assigned
            return m_prog

        l = TorchQcloud3QuantumLayer(qfun,
                        "3047DE8A59764BEDAC9C3282093B16AF1",
                        2,
                        pauli_str_dict=None,
                        shots = 1000,
                        initializer=None,
                        dtype=None,
                        name="",
                        diff_method="parameter_shift",
                        submit_kwargs={"test_qcloud_fake":True},
                        query_kwargs={})
        x = pyvqnet.tensor.QTensor([[0.56,1.2],[0.56,1.2],[0.56,1.2],[0.56,1.2],[0.56,1.2]],requires_grad= True)
        y = l(x)
        print(y)
        y.backward()
        print(l.m_para.grad)
        print(x.grad)

        def qfun2(input,param ):

            m_qlist = range(6)
            cbits = range(6)
            measure_qubits = [0,2]
            m_prog = pq.QProg()
            cir = pq.QCircuit()
            cir<<pq.RZ(m_qlist[0],input[0])
            cir<<pq.CNOT(m_qlist[0],m_qlist[1])
            cir<<pq.RY(m_qlist[1],param[0])
            cir<<pq.CNOT(m_qlist[0],m_qlist[2])
            cir<<pq.RZ(m_qlist[1],input[1])
            cir<<pq.RY(m_qlist[2],param[1])
            cir<<pq.H(m_qlist[2])
            m_prog<<cir

            return m_prog
        l = TorchQcloud3QuantumLayer(qfun2,
                "3047DE8A59764BEDAC9C3282093B16AF",
                2,

                pauli_str_dict={'Z0 X1':10,'':-0.5,'Y2':-0.543},
                shots = 1000,
                initializer=None,
                dtype=None,
                name="",
                diff_method="parameter_shift",
                submit_kwargs={"test_qcloud_fake":True},
                query_kwargs={})
        x = pyvqnet.tensor.QTensor([[0.56,1.2],[0.56,1.2],[0.56,1.2],[0.56,1.2]],requires_grad= True)
        y = l(x)
        print(y)
        y.backward()
        print(l.m_para.grad)
        print(x.grad)

TorchQpanda3QuantumLayer
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Si vous êtes plus familier avec la syntaxe pyQPanda3, vous pouvez utiliser l'interface TorchQpanda3QuantumLayer.

.. py:class:: pyvqnet.qnn.vqc.torch.TorchQpanda3QuantumLayer(qprog_with_measure,para_num,diff_method:str = "parameter_shift",delta:float = 0.01,dtype=None,name="")

    Module de calcul abstrait de la couche quantique variationnelle. Utilise pyQPanda3 pour simuler un circuit quantique paramétré et obtenir les résultats de mesure. Cette couche quantique variationnelle hérite du module de calcul de gradient du framework VQNet. Vous pouvez utiliser la méthode de dérive des paramètres pour calculer le gradient des paramètres du circuit, entraîner le modèle de circuit quantique variationnel ou intégrer le circuit quantique variationnel dans un modèle hybride quantique et classique.

    :param qprog_with_measure: Quantum circuit operation and measurement functions built with pyQPand.
    :param para_num: `int` - number of parameters.
    :param diff_method: méthode pour résoudre les gradients des paramètres du circuit quantique, "parameter shift" ou "finite difference", par défaut parameter shift.
    :param delta: \delta when calculating gradients by finite difference.
    :param dtype: type de données du paramètre, par défaut : None, utilise le type de données par défaut : ``kfloat32``, représentant des nombres à virgule flottante 32 bits.
    :param name: le nom de ce module, par défaut "".

    :return: un module qui peut calculer des circuits quantiques.

    .. note::

        qprog_with_measure est une fonction de circuit quantique définie dans pyQPanda.

        Cette fonction doit inclure les paramètres suivants comme entrées de fonction (même si un paramètre n'est pas réellement utilisé), sinon elle ne fonctionnera pas correctement dans cette fonction.

        L'utilisation de la fonction de circuit quantique qprog_with_measure (input,param,nqubits,ncbits) peut être consultée dans l'exemple suivant.

        `input` : Entrée de données classiques unidimensionnelles. Si aucune, entrer None.

        `param` : Entrée des paramètres à entraîner pour le circuit quantique variationnel unidimensionnel.

    Example::

        import pyqpanda3.core as pq
        from pyvqnet.qnn.pq3 import ProbsMeasure
        import numpy as np
        from pyvqnet.tensor import QTensor
        import pyvqnet
        pyvqnet.backends.set_backend("torch")
        from pyvqnet.qnn.vqc.torch import TorchQpanda3QuantumLayer
        def pqctest (input,param):
            num_of_qubits = 4

            m_machine = pq.CPUQVM()# outside
        
            qubits =range(num_of_qubits)

            circuit = pq.QCircuit()
            circuit<<pq.H(qubits[0])
            circuit<<pq.H(qubits[1])
            circuit<<pq.H(qubits[2])
            circuit<<pq.H(qubits[3])

            circuit<<pq.RZ(qubits[0],input[0])
            circuit<<pq.RZ(qubits[1],input[1])
            circuit<<pq.RZ(qubits[2],input[2])
            circuit<<pq.RZ(qubits[3],input[3])

            circuit<<pq.CNOT(qubits[0],qubits[1])
            circuit<<pq.RZ(qubits[1],param[0])
            circuit<<pq.CNOT(qubits[0],qubits[1])

            circuit<<pq.CNOT(qubits[1],qubits[2])
            circuit<<pq.RZ(qubits[2],param[1])
            circuit<<pq.CNOT(qubits[1],qubits[2])

            circuit<<pq.CNOT(qubits[2],qubits[3])
            circuit<<pq.RZ(qubits[3],param[2])
            circuit<<pq.CNOT(qubits[2],qubits[3])

            prog = pq.QProg()
            prog<<circuit

            rlt_prob = ProbsMeasure(m_machine,prog,[0,2])
            return rlt_prob

        pqc = TorchQpanda3QuantumLayer(pqctest,3)

        # données classiques comme entrée
        input = QTensor([[1.0,2,3,4],[4,2,2,3],[3,3,2,2]],requires_grad=True)

        # circuit forward
        rlt = pqc(input)

        print(rlt)

        grad =  QTensor(np.ones(rlt.data.shape)*1000)
        # circuit backward
        rlt.backward(grad)

        print(pqc.m_para.grad)
        print(input.grad)

Module de circuit quantique variationnel et interface basés sur la différenciation automatique
-----------------------------------------------------------------------------------------------
Classe de base
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

L'écriture d'un modèle de circuit quantique variationnel nécessite d'hériter de ``QModule``.

QModule
""""""""""""""""""

.. py:class:: pyvqnet.qnn.vqc.torch.QModule(name="")

    Lorsque l'utilisateur utilise le backend ``torch``, définit la classe de base que le modèle de circuit quantique variationnel ``Module`` doit hériter.
    Cette classe hérite de ``pyvqnet.nn.torch.TorchModule`` et ``torch.nn.Module``.

    Cette classe peut être ajoutée au modèle torch comme sous-module de ``torch.nn.Module``.

    .. note::

        Cette classe et ses classes dérivées ne sont applicables qu'à ``pyvqnet.backends.set_backend("torch")``, ne pas mélanger avec le ``Module`` sous le ``pyvqnet.nn`` par défaut.

        Les données dans ``_buffers`` de cette classe sont de type ``torch.Tensor``.

        Les données dans ``_parmeters`` de cette classe sont de type ``torch.nn.Parameter``.


QMachine
""""""""""""""""""

.. py:class:: pyvqnet.qnn.vqc.torch.QMachine(num_wires, dtype=pyvqnet.kcomplex64,grad_mode="",save_ir=False)

    Classe de simulateur pour le calcul quantique variationnel, incluant des vecteurs d'état dont l'attribut states est constitué de circuits quantiques.

    Cette classe hérite de ``pyvqnet.nn.torch.TorchModule`` et ``pyvqnet.qnn.QMachine``.

    Cette classe peut être ajoutée au modèle torch comme sous-module de ``torch.nn.Module``.

    .. note::

        Avant chaque exécution du circuit quantique complet, vous devez utiliser `pyvqnet.qnn.vqc.QMachine.reset_states(batchsize)` pour réinitialiser l'état initial dans le simulateur et le diffuser aux dimensions (batchsize,*) pour s'adapter à l'entraînement par lots.

    :param num_wires: Le nombre de bits quantiques.
    :param dtype: Le type de données des données calculées. La valeur par défaut est pyvqnet.kcomplex64, et la précision des paramètres correspondante est pyvqnet.kfloat32.
    :param grad_mode: Le mode de calcul du gradient, qui peut être "adjoint", la valeur par défaut : "", utilise la différenciation automatique.
    :param save_ir: Lorsqu'il est défini sur True, sauvegarde l'opération dans originIR, la valeur par défaut : False.

    :return: Retourne un objet QMachine.

    Example::
        
        from pyvqnet.qnn.vqc.torch import QMachine
        import pyvqnet
        pyvqnet.backends.set_backend("torch")
        qm = QMachine(4)
        print(qm.states)


   .. py:method:: reset_states(batchsize)

        Réinitialise l'état initial dans le simulateur et le diffuse aux dimensions (batchsize,*) pour s'adapter à l'entraînement par lots.

        :param batchsize: Dimension de traitement par lots.

Module de porte logique quantique variationnelle
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Les interfaces de fonctions suivantes dans ``pyvqnet.qnn.vqc`` supportent directement le ``QTensor`` du backend ``torch`` pour le calcul.

.. csv-table:: List of supported pyvqnet.qnn.vqc interfaces
    :file: ./images/same_apis_from_vqc.csv

Les modules de circuit quantique suivants héritent de ``pyvqnet.qnn.vqc.torch.QModule``, où les calculs sont effectués en utilisant ``torch.Tensor``.

.. note::

    Cette classe et ses classes dérivées ne sont applicables qu'à ``pyvqnet.backends.set_backend("torch")``, ne pas mélanger avec ``Module`` sous le ``pyvqnet.nn`` par défaut.

    Si ces classes ont des variables membres non paramétrées ``_buffers``, les données qu'elles contiennent sont de type ``torch.Tensor``.
    Si ces classes ont des variables membres paramétrées ``_parmeters``, les données qu'elles contiennent sont de type ``torch.nn.Parameter``.

I
""""""""""""""""""

.. py:class:: pyvqnet.qnn.vqc.torch.I(has_params: bool = False,trainable: bool = False,init_params=None,wires=None,dtype=pyvqnet.kcomplex64,use_dagger=False)
    
    définit une porte quantique I.

    This class inherits from ``pyvqnet.qnn.vqc.torch.QModule`` and ``torch.nn.Module``.
    Cette classe peut être ajoutée au modèle torch comme sous-module de ``torch.nn.Module``.

    :param has_params: indique s'il a des paramètres, comme les portes RX, RY, etc. qui doivent être définies sur True, et celles sans paramètres doivent être définies sur False, la valeur par défaut est False.
    :param trainable: indique s'il a des paramètres à entraîner. Si la couche utilise des données d'entrée externes pour construire la matrice de la porte logique, définir sur False. Si les paramètres à entraîner doivent être initialisés à partir de cette couche, c'est True, la valeur par défaut est False.
    :param init_params: Paramètres d'initialisation utilisés pour encoder les données classiques QTensor, la valeur par défaut est None.
    :param wires: Index du bit de l'effet de ligne, la valeur par défaut est None.
    :param dtype: La précision des données de la matrice interne de la porte logique peut être définie sur pyvqnet.kcomplex64 ou pyvqnet.kcomplex128, correspondant respectivement à une entrée float ou double.
    :param use_dagger: indique s'il faut utiliser la version conjuguée transposée de la porte, la valeur par défaut est False.
    :return: une instance ``pyvqnet.qnn.vqc.torch.QModule``

    Example::
        
        from pyvqnet.qnn.vqc.torch import I,QMachine
        import pyvqnet
        pyvqnet.backends.set_backend("torch")
        device = QMachine(4)
        layer = I(wires=0)
        batchsize = 1
        device.reset_states(1)
        layer(q_machine = device)
        print(device.states)


Hadamard
""""""""""""""""""

.. py:class:: pyvqnet.qnn.vqc.torch.Hadamard(has_params: bool = False,trainable: bool = False,init_params=None,wires=None,dtype=pyvqnet.kcomplex64,use_dagger=False)
    
    définit une porte quantique Hadamard.

    This class inherits from ``pyvqnet.qnn.vqc.torch.QModule`` and ``torch.nn.Module``.
    Cette classe peut être ajoutée au modèle torch comme sous-module de ``torch.nn.Module``.

    :param has_params: indique s'il a des paramètres, comme les portes RX, RY, etc. qui doivent être définies sur True, et celles sans paramètres doivent être définies sur False, la valeur par défaut est False.
    :param trainable: indique s'il a des paramètres à entraîner. Si la couche utilise des données d'entrée externes pour construire la matrice de la porte logique, définir sur False. Si les paramètres à entraîner doivent être initialisés à partir de cette couche, c'est True, la valeur par défaut est False.
    :param init_params: Paramètres d'initialisation utilisés pour encoder les données classiques QTensor, la valeur par défaut est None.
    :param wires: Index du bit de l'effet de ligne, la valeur par défaut est None.
    :param dtype: La précision des données de la matrice interne de la porte logique peut être définie sur pyvqnet.kcomplex64 ou pyvqnet.kcomplex128, correspondant respectivement à une entrée float ou double.
    :param use_dagger: indique s'il faut utiliser la version conjuguée transposée de la porte, la valeur par défaut est False.
    :return: une instance ``pyvqnet.qnn.vqc.torch.QModule``

    Example::
        
        from pyvqnet.qnn.vqc.torch import Hadamard,QMachine
        import pyvqnet
        pyvqnet.backends.set_backend("torch")
        device = QMachine(4)
        layer = Hadamard(wires=0)
        batchsize = 1
        device.reset_states(1)
        layer(q_machine = device)
        print(device.states)


T
""""""""""""""""""

.. py:class:: pyvqnet.qnn.vqc.torch.T(has_params: bool = False,trainable: bool = False,init_params=None,wires=None,dtype=pyvqnet.kcomplex64,use_dagger=False)
    
    définit une porte quantique T.

    This class inherits from ``pyvqnet.qnn.vqc.torch.QModule`` and ``torch.nn.Module``.
    Cette classe peut être ajoutée au modèle torch comme sous-module de ``torch.nn.Module``.

    :param has_params: indique s'il a des paramètres, comme les portes RX, RY, etc. qui doivent être définies sur True, et celles sans paramètres doivent être définies sur False, la valeur par défaut est False.
    :param trainable: indique s'il a des paramètres à entraîner. Si la couche utilise des données d'entrée externes pour construire la matrice de la porte logique, définir sur False. Si les paramètres à entraîner doivent être initialisés à partir de cette couche, c'est True, la valeur par défaut est False.
    :param init_params: Paramètres d'initialisation utilisés pour encoder les données classiques QTensor, la valeur par défaut est None.
    :param wires: Index du bit de l'effet de ligne, la valeur par défaut est None.
    :param dtype: La précision des données de la matrice interne de la porte logique peut être définie sur pyvqnet.kcomplex64 ou pyvqnet.kcomplex128, correspondant respectivement à une entrée float ou double.
    :param use_dagger: indique s'il faut utiliser la version conjuguée transposée de la porte, la valeur par défaut est False.
    :return: une instance ``pyvqnet.qnn.vqc.torch.QModule``

    Example::
        
        from pyvqnet.qnn.vqc.torch import T,QMachine
        import pyvqnet
        pyvqnet.backends.set_backend("torch")
        device = QMachine(4)
        layer = T(wires=0)
        batchsize = 1
        device.reset_states(1)
        layer(q_machine = device)
        print(device.states)



S
""""""""""""""""""

.. py:class:: pyvqnet.qnn.vqc.torch.S(has_params: bool = False,trainable: bool = False,init_params=None,wires=None,dtype=pyvqnet.kcomplex64,use_dagger=False)
    
    définit une porte quantique S.

    This class inherits from ``pyvqnet.qnn.vqc.torch.QModule`` and ``torch.nn.Module``.
    Cette classe peut être ajoutée au modèle torch comme sous-module de ``torch.nn.Module``.

    :param has_params: indique s'il a des paramètres, comme les portes RX, RY, etc. qui doivent être définies sur True, et celles sans paramètres doivent être définies sur False, la valeur par défaut est False.
    :param trainable: indique s'il a des paramètres à entraîner. Si la couche utilise des données d'entrée externes pour construire la matrice de la porte logique, définir sur False. Si les paramètres à entraîner doivent être initialisés à partir de cette couche, c'est True, la valeur par défaut est False.
    :param init_params: Paramètres d'initialisation utilisés pour encoder les données classiques QTensor, la valeur par défaut est None.
    :param wires: Index du bit de l'effet de ligne, la valeur par défaut est None.
    :param dtype: La précision des données de la matrice interne de la porte logique peut être définie sur pyvqnet.kcomplex64 ou pyvqnet.kcomplex128, correspondant respectivement à une entrée float ou double.
    :param use_dagger: indique s'il faut utiliser la version conjuguée transposée de la porte, la valeur par défaut est False.
    :return: une instance ``pyvqnet.qnn.vqc.torch.QModule``

    Example::
        
        from pyvqnet.qnn.vqc.torch import S,QMachine
        import pyvqnet
        pyvqnet.backends.set_backend("torch")
        device = QMachine(4)
        layer = S(wires=0)
        batchsize = 1
        device.reset_states(1)
        layer(q_machine = device)
        print(device.states)


PauliX
""""""""""""""""""

.. py:class:: pyvqnet.qnn.vqc.torch.PauliX(has_params: bool = False,trainable: bool = False,init_params=None,wires=None,dtype=pyvqnet.kcomplex64,use_dagger=False)
    
    définit une porte quantique PauliX.

    This class inherits from ``pyvqnet.qnn.vqc.torch.QModule`` and ``torch.nn.Module``.
    Cette classe peut être ajoutée au modèle torch comme sous-module de ``torch.nn.Module``.


    :param has_params: indique s'il a des paramètres, comme les portes RX, RY, etc. qui doivent être définies sur True, et celles sans paramètres doivent être définies sur False, la valeur par défaut est False.
    :param trainable: indique s'il a des paramètres à entraîner. Si la couche utilise des données d'entrée externes pour construire la matrice de la porte logique, définir sur False. Si les paramètres à entraîner doivent être initialisés à partir de cette couche, c'est True, la valeur par défaut est False.
    :param init_params: Paramètres d'initialisation utilisés pour encoder les données classiques QTensor, la valeur par défaut est None.
    :param wires: Index du bit de l'effet de ligne, la valeur par défaut est None.
    :param dtype: La précision des données de la matrice interne de la porte logique peut être définie sur pyvqnet.kcomplex64 ou pyvqnet.kcomplex128, correspondant respectivement à une entrée float ou double.
    :param use_dagger: indique s'il faut utiliser la version conjuguée transposée de la porte, la valeur par défaut est False.
    :return: une instance ``pyvqnet.qnn.vqc.torch.QModule``

    Example::
        
        from pyvqnet.qnn.vqc.torch import PauliX,QMachine
        import pyvqnet
        pyvqnet.backends.set_backend("torch")
        device = QMachine(4)
        layer = PauliX(wires=0)
        batchsize = 1
        device.reset_states(1)
        layer(q_machine = device)
        print(device.states)


PauliY
""""""""""""""""""

.. py:class:: pyvqnet.qnn.vqc.torch.PauliY(has_params: bool = False,trainable: bool = False,init_params=None,wires=None,dtype=pyvqnet.kcomplex64,use_dagger=False)
    
    définit une porte quantique PauliY.

    This class inherits from ``pyvqnet.qnn.vqc.torch.QModule`` and ``torch.nn.Module``.
    Cette classe peut être ajoutée au modèle torch comme sous-module de ``torch.nn.Module``.


    :param has_params: indique s'il a des paramètres, comme les portes RX, RY, etc. qui doivent être définies sur True, et celles sans paramètres doivent être définies sur False, la valeur par défaut est False.
    :param trainable: indique s'il a des paramètres à entraîner. Si la couche utilise des données d'entrée externes pour construire la matrice de la porte logique, définir sur False. Si les paramètres à entraîner doivent être initialisés à partir de cette couche, c'est True, la valeur par défaut est False.
    :param init_params: Paramètres d'initialisation utilisés pour encoder les données classiques QTensor, la valeur par défaut est None.
    :param wires: Index du bit de l'effet de ligne, la valeur par défaut est None.
    :param dtype: La précision des données de la matrice interne de la porte logique peut être définie sur pyvqnet.kcomplex64 ou pyvqnet.kcomplex128, correspondant respectivement à une entrée float ou double.
    :param use_dagger: indique s'il faut utiliser la version conjuguée transposée de la porte, la valeur par défaut est False.
    :return: une instance ``pyvqnet.qnn.vqc.torch.QModule``

    Example::
        
        from pyvqnet.qnn.vqc.torch import PauliY,QMachine
        import pyvqnet
        pyvqnet.backends.set_backend("torch")
        device = QMachine(4)
        layer = PauliY(wires=0)
        batchsize = 1
        device.reset_states(1)
        layer(q_machine = device)
        print(device.states)



PauliZ
""""""""""""""""""

.. py:class:: pyvqnet.qnn.vqc.torch.PauliZ(has_params: bool = False,trainable: bool = False,init_params=None,wires=None,dtype=pyvqnet.kcomplex64,use_dagger=False)
    
    définit une porte quantique PauliZ.

    This class inherits from ``pyvqnet.qnn.vqc.torch.QModule`` and ``torch.nn.Module``.
    Cette classe peut être ajoutée au modèle torch comme sous-module de ``torch.nn.Module``.


    :param has_params: indique s'il a des paramètres, comme les portes RX, RY, etc. qui doivent être définies sur True, et celles sans paramètres doivent être définies sur False, la valeur par défaut est False.
    :param trainable: indique s'il a des paramètres à entraîner. Si la couche utilise des données d'entrée externes pour construire la matrice de la porte logique, définir sur False. Si les paramètres à entraîner doivent être initialisés à partir de cette couche, c'est True, la valeur par défaut est False.
    :param init_params: Paramètres d'initialisation utilisés pour encoder les données classiques QTensor, la valeur par défaut est None.
    :param wires: Index du bit de l'effet de ligne, la valeur par défaut est None.
    :param dtype: La précision des données de la matrice interne de la porte logique peut être définie sur pyvqnet.kcomplex64 ou pyvqnet.kcomplex128, correspondant respectivement à une entrée float ou double.
    :param use_dagger: indique s'il faut utiliser la version conjuguée transposée de la porte, la valeur par défaut est False.
    :return: une instance ``pyvqnet.qnn.vqc.torch.QModule``

    Example::
        
        from pyvqnet.qnn.vqc.torch import PauliZ,QMachine
        import pyvqnet
        pyvqnet.backends.set_backend("torch")
        device = QMachine(4)
        layer = PauliZ(wires=0)
        batchsize = 1
        device.reset_states(1)
        layer(q_machine = device)
        print(device.states)



X1
""""""""""""""""""

.. py:class:: pyvqnet.qnn.vqc.torch.X1(has_params: bool = False,trainable: bool = False,init_params=None,wires=None,dtype=pyvqnet.kcomplex64,use_dagger=False)
    
    définit une porte quantique X1.

    This class inherits from ``pyvqnet.qnn.vqc.torch.QModule`` and ``torch.nn.Module``.
    Cette classe peut être ajoutée au modèle torch comme sous-module de ``torch.nn.Module``.

    :param has_params: indique s'il a des paramètres, comme les portes RX, RY, etc. qui doivent être définies sur True, et celles sans paramètres doivent être définies sur False, la valeur par défaut est False.
    :param trainable: indique s'il a des paramètres à entraîner. Si la couche utilise des données d'entrée externes pour construire la matrice de la porte logique, définir sur False. Si les paramètres à entraîner doivent être initialisés à partir de cette couche, c'est True, la valeur par défaut est False.
    :param init_params: Paramètres d'initialisation utilisés pour encoder les données classiques QTensor, la valeur par défaut est None.
    :param wires: Index du bit de l'effet de ligne, la valeur par défaut est None.
    :param dtype: La précision des données de la matrice interne de la porte logique peut être définie sur pyvqnet.kcomplex64 ou pyvqnet.kcomplex128, correspondant respectivement à une entrée float ou double.
    :param use_dagger: indique s'il faut utiliser la version conjuguée transposée de la porte, la valeur par défaut est False.
    :return: une instance ``pyvqnet.qnn.vqc.torch.QModule``

    Example::
        
        from pyvqnet.qnn.vqc.torch import X1,QMachine
        import pyvqnet
        pyvqnet.backends.set_backend("torch")
        device = QMachine(4)
        layer = X1(wires=0)
        batchsize = 1
        device.reset_states(1)
        layer(q_machine = device)
        print(device.states)


RX
""""""""""""""""""

.. py:class:: pyvqnet.qnn.vqc.torch.RX(has_params: bool = False,trainable: bool = False,init_params=None,wires=None,dtype=pyvqnet.kcomplex64,use_dagger=False)
    
    définit une porte quantique RX.

    This class inherits from ``pyvqnet.qnn.vqc.torch.QModule`` and ``torch.nn.Module``.
    Cette classe peut être ajoutée au modèle torch comme sous-module de ``torch.nn.Module``.

    :param has_params: indique s'il a des paramètres, comme les portes RX, RY, etc. qui doivent être définies sur True, et celles sans paramètres doivent être définies sur False, la valeur par défaut est False.
    :param trainable: indique s'il a des paramètres à entraîner. Si la couche utilise des données d'entrée externes pour construire la matrice de la porte logique, définir sur False. Si les paramètres à entraîner doivent être initialisés à partir de cette couche, c'est True, la valeur par défaut est False.
    :param init_params: Paramètres d'initialisation utilisés pour encoder les données classiques QTensor, la valeur par défaut est None.
    :param wires: Index du bit de l'effet de ligne, la valeur par défaut est None.
    :param dtype: La précision des données de la matrice interne de la porte logique peut être définie sur pyvqnet.kcomplex64 ou pyvqnet.kcomplex128, correspondant respectivement à une entrée float ou double.
    :param use_dagger: indique s'il faut utiliser la version conjuguée transposée de la porte, la valeur par défaut est False.
    :return: une instance ``pyvqnet.qnn.vqc.torch.QModule``

    Example::

        from pyvqnet.qnn.vqc.torch import RX,QMachine
        import pyvqnet
        pyvqnet.backends.set_backend("torch")
        device = QMachine(4)
        layer = RX(has_params= True, trainable= True, wires=0)
        batchsize = 2
        device.reset_states(batchsize)
        layer(q_machine = device)
        print(device.states)



RY
""""""""""""""""""

.. py:class:: pyvqnet.qnn.vqc.torch.RY(has_params: bool = False,trainable: bool = False,init_params=None,wires=None,dtype=pyvqnet.kcomplex64,use_dagger=False)
    
    définit une porte quantique RY.

    This class inherits from ``pyvqnet.qnn.vqc.torch.QModule`` and ``torch.nn.Module``.
    Cette classe peut être ajoutée au modèle torch comme sous-module de ``torch.nn.Module``.

    :param has_params: indique s'il a des paramètres, comme les portes RX, RY, etc. qui doivent être définies sur True, et celles sans paramètres doivent être définies sur False, la valeur par défaut est False.
    :param trainable: indique s'il a des paramètres à entraîner. Si la couche utilise des données d'entrée externes pour construire la matrice de la porte logique, définir sur False. Si les paramètres à entraîner doivent être initialisés à partir de cette couche, c'est True, la valeur par défaut est False.
    :param init_params: Paramètres d'initialisation utilisés pour encoder les données classiques QTensor, la valeur par défaut est None.
    :param wires: Index du bit de l'effet de ligne, la valeur par défaut est None.
    :param dtype: La précision des données de la matrice interne de la porte logique peut être définie sur pyvqnet.kcomplex64 ou pyvqnet.kcomplex128, correspondant respectivement à une entrée float ou double.
    :param use_dagger: indique s'il faut utiliser la version conjuguée transposée de la porte, la valeur par défaut est False.
    :return: une instance ``pyvqnet.qnn.vqc.torch.QModule``

    Example::

        from pyvqnet.qnn.vqc.torch import RY,QMachine
        import pyvqnet
        pyvqnet.backends.set_backend("torch")
        device = QMachine(4)
        layer = RY(has_params= True, trainable= True, wires=0)
        batchsize = 2
        device.reset_states(batchsize)
        layer(q_machine = device)
        print(device.states)


RZ
""""""""""""""""""

.. py:class:: pyvqnet.qnn.vqc.torch.RZ(has_params: bool = False,trainable: bool = False,init_params=None,wires=None,dtype=pyvqnet.kcomplex64,use_dagger=False)
    
    définit une porte quantique RZ.

    This class inherits from ``pyvqnet.qnn.vqc.torch.QModule`` and ``torch.nn.Module``.
    Cette classe peut être ajoutée au modèle torch comme sous-module de ``torch.nn.Module``.

    :param has_params: indique s'il a des paramètres, comme les portes RX, RY, etc. qui doivent être définies sur True, et celles sans paramètres doivent être définies sur False, la valeur par défaut est False.
    :param trainable: indique s'il a des paramètres à entraîner. Si la couche utilise des données d'entrée externes pour construire la matrice de la porte logique, définir sur False. Si les paramètres à entraîner doivent être initialisés à partir de cette couche, c'est True, la valeur par défaut est False.
    :param init_params: Paramètres d'initialisation utilisés pour encoder les données classiques QTensor, la valeur par défaut est None.
    :param wires: Index du bit de l'effet de ligne, la valeur par défaut est None.
    :param dtype: La précision des données de la matrice interne de la porte logique peut être définie sur pyvqnet.kcomplex64 ou pyvqnet.kcomplex128, correspondant respectivement à une entrée float ou double.
    :param use_dagger: indique s'il faut utiliser la version conjuguée transposée de la porte, la valeur par défaut est False.
    :return: une instance ``pyvqnet.qnn.vqc.torch.QModule``

    Example::

        from pyvqnet.qnn.vqc.torch import RZ,QMachine
        import pyvqnet
        pyvqnet.backends.set_backend("torch")
        device = QMachine(4)
        layer = RZ(has_params= True, trainable= True, wires=0)
        batchsize = 2
        device.reset_states(batchsize)
        layer(q_machine = device)
        print(device.states)


CRX
""""""""""""""""""

.. py:class:: pyvqnet.qnn.vqc.torch.CRX(has_params: bool = False,trainable: bool = False,init_params=None,wires=None,dtype=pyvqnet.kcomplex64,use_dagger=False)
    
    définit une porte quantique CRX.

    This class inherits from ``pyvqnet.qnn.vqc.torch.QModule`` and ``torch.nn.Module``.
    Cette classe peut être ajoutée au modèle torch comme sous-module de ``torch.nn.Module``.

    :param has_params: indique s'il a des paramètres, comme les portes RX, RY, etc. qui doivent être définies sur True, et celles sans paramètres doivent être définies sur False, la valeur par défaut est False.
    :param trainable: indique s'il a des paramètres à entraîner. Si la couche utilise des données d'entrée externes pour construire la matrice de la porte logique, définir sur False. Si les paramètres à entraîner doivent être initialisés à partir de cette couche, c'est True, la valeur par défaut est False.
    :param init_params: Paramètres d'initialisation utilisés pour encoder les données classiques QTensor, la valeur par défaut est None.
    :param wires: Index du bit de l'effet de ligne, la valeur par défaut est None.
    :param dtype: La précision des données de la matrice interne de la porte logique peut être définie sur pyvqnet.kcomplex64 ou pyvqnet.kcomplex128, correspondant respectivement à une entrée float ou double.
    :param use_dagger: indique s'il faut utiliser la version conjuguée transposée de la porte, la valeur par défaut est False.
    :return: une instance ``pyvqnet.qnn.vqc.torch.QModule``

    Example::

        from pyvqnet.qnn.vqc.torch import CRX,QMachine
        import pyvqnet
        pyvqnet.backends.set_backend("torch")
        device = QMachine(4)
        layer = CRX(has_params= True, trainable= True, wires=[0,2])
        batchsize = 2
        device.reset_states(batchsize)
        layer(q_machine = device)
        print(device.states)


CRY
""""""""""""""""""


.. py:class:: pyvqnet.qnn.vqc.torch.CRY(has_params: bool = False,trainable: bool = False,init_params=None,wires=None,dtype=pyvqnet.kcomplex64,use_dagger=False)
    
    définit une porte quantique CRY.

    This class inherits from ``pyvqnet.qnn.vqc.torch.QModule`` and ``torch.nn.Module``.
    Cette classe peut être ajoutée au modèle torch comme sous-module de ``torch.nn.Module``.

    :param has_params: indique s'il a des paramètres, comme les portes RX, RY, etc. qui doivent être définies sur True, et celles sans paramètres doivent être définies sur False, la valeur par défaut est False.
    :param trainable: indique s'il a des paramètres à entraîner. Si la couche utilise des données d'entrée externes pour construire la matrice de la porte logique, définir sur False. Si les paramètres à entraîner doivent être initialisés à partir de cette couche, c'est True, la valeur par défaut est False.
    :param init_params: Paramètres d'initialisation utilisés pour encoder les données classiques QTensor, la valeur par défaut est None.
    :param wires: Index du bit de l'effet de ligne, la valeur par défaut est None.
    :param dtype: La précision des données de la matrice interne de la porte logique peut être définie sur pyvqnet.kcomplex64 ou pyvqnet.kcomplex128, correspondant respectivement à une entrée float ou double.
    :param use_dagger: indique s'il faut utiliser la version conjuguée transposée de la porte, la valeur par défaut est False.
    :return: une instance ``pyvqnet.qnn.vqc.torch.QModule``

    Example::

        from pyvqnet.qnn.vqc.torch import CRY,QMachine
        import pyvqnet
        pyvqnet.backends.set_backend("torch")
        device = QMachine(4)
        layer = CRY(has_params= True, trainable= True, wires=[0,2])
        batchsize = 2
        device.reset_states(batchsize)
        layer(q_machine = device)
        print(device.states)


CRZ
""""""""""""""""""


.. py:class:: pyvqnet.qnn.vqc.torch.CRZ(has_params: bool = False,trainable: bool = False,init_params=None,wires=None,dtype=pyvqnet.kcomplex64,use_dagger=False)
    
    définit une porte quantique CRZ.

    This class inherits from ``pyvqnet.qnn.vqc.torch.QModule`` and ``torch.nn.Module``.
    Cette classe peut être ajoutée au modèle torch comme sous-module de ``torch.nn.Module``.

    :param has_params: indique s'il a des paramètres, comme les portes RX, RY, etc. qui doivent être définies sur True, et celles sans paramètres doivent être définies sur False, la valeur par défaut est False.
    :param trainable: indique s'il a des paramètres à entraîner. Si la couche utilise des données d'entrée externes pour construire la matrice de la porte logique, définir sur False. Si les paramètres à entraîner doivent être initialisés à partir de cette couche, c'est True, la valeur par défaut est False.
    :param init_params: Paramètres d'initialisation utilisés pour encoder les données classiques QTensor, la valeur par défaut est None.
    :param wires: Index du bit de l'effet de ligne, la valeur par défaut est None.
    :param dtype: La précision des données de la matrice interne de la porte logique peut être définie sur pyvqnet.kcomplex64 ou pyvqnet.kcomplex128, correspondant respectivement à une entrée float ou double.
    :param use_dagger: indique s'il faut utiliser la version conjuguée transposée de la porte, la valeur par défaut est False.
    :return: une instance ``pyvqnet.qnn.vqc.torch.QModule``

    Example::

        from pyvqnet.qnn.vqc.torch import CRZ,QMachine
        import pyvqnet
        pyvqnet.backends.set_backend("torch")
        device = QMachine(4)
        layer = CRZ(has_params= True, trainable= True, wires=[0,2])
        batchsize = 2
        device.reset_states(batchsize)
        layer(q_machine = device)
        print(device.states)



U1
""""""""""""""""""

.. py:class:: pyvqnet.qnn.vqc.torch.U1(has_params: bool = False,trainable: bool = False,init_params=None,wires=None,dtype=pyvqnet.kcomplex64,use_dagger=False)
    
    définit une porte quantique U1.

    This class inherits from ``pyvqnet.qnn.vqc.torch.QModule`` and ``torch.nn.Module``.
    Cette classe peut être ajoutée au modèle torch comme sous-module de ``torch.nn.Module``.

    :param has_params: indique s'il a des paramètres, comme les portes RX, RY, etc. qui doivent être définies sur True, et celles sans paramètres doivent être définies sur False, la valeur par défaut est False.
    :param trainable: indique s'il a des paramètres à entraîner. Si la couche utilise des données d'entrée externes pour construire la matrice de la porte logique, définir sur False. Si les paramètres à entraîner doivent être initialisés à partir de cette couche, c'est True, la valeur par défaut est False.
    :param init_params: Paramètres d'initialisation utilisés pour encoder les données classiques QTensor, la valeur par défaut est None.
    :param wires: Index du bit de l'effet de ligne, la valeur par défaut est None.
    :param dtype: La précision des données de la matrice interne de la porte logique peut être définie sur pyvqnet.kcomplex64 ou pyvqnet.kcomplex128, correspondant respectivement à une entrée float ou double.
    :param use_dagger: indique s'il faut utiliser la version conjuguée transposée de la porte, la valeur par défaut est False.
    :return: une instance ``pyvqnet.qnn.vqc.torch.QModule``

    Example::

        from pyvqnet.qnn.vqc.torch import U1,QMachine
        import pyvqnet
        pyvqnet.backends.set_backend("torch")
        device = QMachine(4)
        layer = U1(has_params= True, trainable= True, wires=0)
        batchsize = 2
        device.reset_states(batchsize)
        layer(q_machine = device)
        print(device.states)

U2
""""""""""""""""""


.. py:class:: pyvqnet.qnn.vqc.torch.U2(has_params: bool = False,trainable: bool = False,init_params=None,wires=None,dtype=pyvqnet.kcomplex64,use_dagger=False)
    
    définit une porte quantique U2.

    This class inherits from ``pyvqnet.qnn.vqc.torch.QModule`` and ``torch.nn.Module``.
    Cette classe peut être ajoutée au modèle torch comme sous-module de ``torch.nn.Module``.

    :param has_params: indique s'il a des paramètres, comme les portes RX, RY, etc. qui doivent être définies sur True, et celles sans paramètres doivent être définies sur False, la valeur par défaut est False.
    :param trainable: indique s'il a des paramètres à entraîner. Si la couche utilise des données d'entrée externes pour construire la matrice de la porte logique, définir sur False. Si les paramètres à entraîner doivent être initialisés à partir de cette couche, c'est True, la valeur par défaut est False.
    :param init_params: Paramètres d'initialisation utilisés pour encoder les données classiques QTensor, la valeur par défaut est None.
    :param wires: Index du bit de l'effet de ligne, la valeur par défaut est None.
    :param dtype: La précision des données de la matrice interne de la porte logique peut être définie sur pyvqnet.kcomplex64 ou pyvqnet.kcomplex128, correspondant respectivement à une entrée float ou double.
    :param use_dagger: indique s'il faut utiliser la version conjuguée transposée de la porte, la valeur par défaut est False.
    :return: une instance ``pyvqnet.qnn.vqc.torch.QModule``

    Example::

        from pyvqnet.qnn.vqc.torch import U2,QMachine
        import pyvqnet
        pyvqnet.backends.set_backend("torch")
        device = QMachine(4)
        layer = U2(has_params= True, trainable= True, wires=1)
        batchsize = 2
        device.reset_states(batchsize)
        layer(q_machine = device)
        print(device.states)


U3
""""""""""""""""""


.. py:class:: pyvqnet.qnn.vqc.torch.U3(has_params: bool = False,trainable: bool = False,init_params=None,wires=None,dtype=pyvqnet.kcomplex64,use_dagger=False)
    
    définit une porte quantique U3.

    This class inherits from ``pyvqnet.qnn.vqc.torch.QModule`` and ``torch.nn.Module``.
    Cette classe peut être ajoutée au modèle torch comme sous-module de ``torch.nn.Module``.

    :param has_params: indique s'il a des paramètres, comme les portes RX, RY, etc. qui doivent être définies sur True, et celles sans paramètres doivent être définies sur False, la valeur par défaut est False.
    :param trainable: indique s'il a des paramètres à entraîner. Si la couche utilise des données d'entrée externes pour construire la matrice de la porte logique, définir sur False. Si les paramètres à entraîner doivent être initialisés à partir de cette couche, c'est True, la valeur par défaut est False.
    :param init_params: Paramètres d'initialisation utilisés pour encoder les données classiques QTensor, la valeur par défaut est None.
    :param wires: Index du bit de l'effet de ligne, la valeur par défaut est None.
    :param dtype: La précision des données de la matrice interne de la porte logique peut être définie sur pyvqnet.kcomplex64 ou pyvqnet.kcomplex128, correspondant respectivement à une entrée float ou double.
    :param use_dagger: indique s'il faut utiliser la version conjuguée transposée de la porte, la valeur par défaut est False.
    :return: une instance ``pyvqnet.qnn.vqc.torch.QModule``

    Example::

        from pyvqnet.qnn.vqc.torch import U3,QMachine
        import pyvqnet
        pyvqnet.backends.set_backend("torch")
        device = QMachine(4)
        layer = U3(has_params= True, trainable= True, wires=1)
        batchsize = 2
        device.reset_states(batchsize)
        layer(q_machine = device)
        print(device.states)



CNOT
""""""""""""""""""

.. py:class:: pyvqnet.qnn.vqc.torch.CNOT(has_params: bool = False,trainable: bool = False,init_params=None,wires=None,dtype=pyvqnet.kcomplex64,use_dagger=False)
    
    définit une porte quantique CNOT, alias `CX`.

    This class inherits from ``pyvqnet.qnn.vqc.torch.QModule`` and ``torch.nn.Module``.
    Cette classe peut être ajoutée au modèle torch comme sous-module de ``torch.nn.Module``.

    :param has_params: indique s'il a des paramètres, comme les portes RX, RY, etc. qui doivent être définies sur True, et celles sans paramètres doivent être définies sur False, la valeur par défaut est False.
    :param trainable: indique s'il a des paramètres à entraîner. Si la couche utilise des données d'entrée externes pour construire la matrice de la porte logique, définir sur False. Si les paramètres à entraîner doivent être initialisés à partir de cette couche, c'est True, la valeur par défaut est False.
    :param init_params: Paramètres d'initialisation utilisés pour encoder les données classiques QTensor, la valeur par défaut est None.
    :param wires: Index du bit de l'effet de ligne, la valeur par défaut est None.
    :param dtype: La précision des données de la matrice interne de la porte logique peut être définie sur pyvqnet.kcomplex64 ou pyvqnet.kcomplex128, correspondant respectivement à une entrée float ou double.
    :param use_dagger: indique s'il faut utiliser la version conjuguée transposée de la porte, la valeur par défaut est False.
    :return: une instance ``pyvqnet.qnn.vqc.torch.QModule``

    Example::

        from pyvqnet.qnn.vqc.torch import CNOT,QMachine
        import pyvqnet
        pyvqnet.backends.set_backend("torch")
        device = QMachine(4)
        layer = CNOT(wires=[0,1])
        batchsize = 2
        device.reset_states(batchsize)
        layer(q_machine = device)
        print(device.states)

CY
""""""""""""""""""

.. py:class:: pyvqnet.qnn.vqc.torch.CY(has_params: bool = False,trainable: bool = False,init_params=None,wires=None,dtype=pyvqnet.kcomplex64,use_dagger=False)
    
    définit une porte quantique CY.

    This class inherits from ``pyvqnet.qnn.vqc.torch.QModule`` and ``torch.nn.Module``.
    Cette classe peut être ajoutée au modèle torch comme sous-module de ``torch.nn.Module``.

    :param has_params: indique s'il a des paramètres, comme les portes RX, RY, etc. qui doivent être définies sur True, et celles sans paramètres doivent être définies sur False, la valeur par défaut est False.
    :param trainable: indique s'il a des paramètres à entraîner. Si la couche utilise des données d'entrée externes pour construire la matrice de la porte logique, définir sur False. Si les paramètres à entraîner doivent être initialisés à partir de cette couche, c'est True, la valeur par défaut est False.
    :param init_params: Paramètres d'initialisation utilisés pour encoder les données classiques QTensor, la valeur par défaut est None.
    :param wires: Index du bit de l'effet de ligne, la valeur par défaut est None.
    :param dtype: La précision des données de la matrice interne de la porte logique peut être définie sur pyvqnet.kcomplex64 ou pyvqnet.kcomplex128, correspondant respectivement à une entrée float ou double.
    :param use_dagger: indique s'il faut utiliser la version conjuguée transposée de la porte, la valeur par défaut est False.
    :return: une instance ``pyvqnet.qnn.vqc.torch.QModule``

    Example::

        from pyvqnet.qnn.vqc.torch import CY,QMachine
        import pyvqnet
        pyvqnet.backends.set_backend("torch")
        device = QMachine(4)
        layer = CY(wires=[0,1])
        batchsize = 2
        device.reset_states(batchsize)
        layer(q_machine = device)
        print(device.states)


CZ
""""""""""""""""""

.. py:class:: pyvqnet.qnn.vqc.torch.CZ(has_params: bool = False,trainable: bool = False,init_params=None,wires=None,dtype=pyvqnet.kcomplex64,use_dagger=False)
    
    définit une porte quantique CZ.

    This class inherits from ``pyvqnet.qnn.vqc.torch.QModule`` and ``torch.nn.Module``.
    Cette classe peut être ajoutée au modèle torch comme sous-module de ``torch.nn.Module``.

    :param has_params: indique s'il a des paramètres, comme les portes RX, RY, etc. qui doivent être définies sur True, et celles sans paramètres doivent être définies sur False, la valeur par défaut est False.
    :param trainable: indique s'il a des paramètres à entraîner. Si la couche utilise des données d'entrée externes pour construire la matrice de la porte logique, définir sur False. Si les paramètres à entraîner doivent être initialisés à partir de cette couche, c'est True, la valeur par défaut est False.
    :param init_params: Paramètres d'initialisation utilisés pour encoder les données classiques QTensor, la valeur par défaut est None.
    :param wires: Index du bit de l'effet de ligne, la valeur par défaut est None.
    :param dtype: La précision des données de la matrice interne de la porte logique peut être définie sur pyvqnet.kcomplex64 ou pyvqnet.kcomplex128, correspondant respectivement à une entrée float ou double.
    :param use_dagger: indique s'il faut utiliser la version conjuguée transposée de la porte, la valeur par défaut est False.
    :return: une instance ``pyvqnet.qnn.vqc.torch.QModule``

    Example::

        from pyvqnet.qnn.vqc.torch import CZ,QMachine
        import pyvqnet
        pyvqnet.backends.set_backend("torch")
        device = QMachine(4)
        layer = CZ(wires=[0,1])
        batchsize = 2
        device.reset_states(batchsize)
        layer(q_machine = device)
        print(device.states)




CR
""""""""""""""""""

.. py:class:: pyvqnet.qnn.vqc.torch.CR(has_params: bool = False,trainable: bool = False,init_params=None,wires=None,dtype=pyvqnet.kcomplex64,use_dagger=False)
    
    définit une porte quantique CR.

    This class inherits from ``pyvqnet.qnn.vqc.torch.QModule`` and ``torch.nn.Module``.
    Cette classe peut être ajoutée au modèle torch comme sous-module de ``torch.nn.Module``.

    :param has_params: indique s'il a des paramètres, comme les portes RX, RY, etc. qui doivent être définies sur True, et celles sans paramètres doivent être définies sur False, la valeur par défaut est False.
    :param trainable: indique s'il a des paramètres à entraîner. Si la couche utilise des données d'entrée externes pour construire la matrice de la porte logique, définir sur False. Si les paramètres à entraîner doivent être initialisés à partir de cette couche, c'est True, la valeur par défaut est False.
    :param init_params: Paramètres d'initialisation utilisés pour encoder les données classiques QTensor, la valeur par défaut est None.
    :param wires: Index du bit de l'effet de ligne, la valeur par défaut est None.
    :param dtype: La précision des données de la matrice interne de la porte logique peut être définie sur pyvqnet.kcomplex64 ou pyvqnet.kcomplex128, correspondant respectivement à une entrée float ou double.
    :param use_dagger: indique s'il faut utiliser la version conjuguée transposée de la porte, la valeur par défaut est False.
    :return: une instance ``pyvqnet.qnn.vqc.torch.QModule``

    Example::

        from pyvqnet.qnn.vqc.torch import CR,QMachine
        import pyvqnet
        pyvqnet.backends.set_backend("torch")
        device = QMachine(4)
        device = QMachine(4)
        layer = CR(has_params= True, trainable= True, wires=[0,2])
        batchsize = 2
        device.reset_states(batchsize)
        layer(q_machine = device)
        print(device.states)



SWAP
""""""""""""""""""


.. py:class:: pyvqnet.qnn.vqc.torch.SWAP(has_params: bool = False,trainable: bool = False,init_params=None,wires=None,dtype=pyvqnet.kcomplex64,use_dagger=False)
    
    définit une porte quantique CSWAP.

    This class inherits from ``pyvqnet.qnn.vqc.torch.QModule`` and ``torch.nn.Module``.
    Cette classe peut être ajoutée au modèle torch comme sous-module de ``torch.nn.Module``.

    :param has_params: indique s'il a des paramètres, comme les portes RX, RY, etc. qui doivent être définies sur True, et celles sans paramètres doivent être définies sur False, la valeur par défaut est False.
    :param trainable: indique s'il a des paramètres à entraîner. Si la couche utilise des données d'entrée externes pour construire la matrice de la porte logique, définir sur False. Si les paramètres à entraîner doivent être initialisés à partir de cette couche, c'est True, la valeur par défaut est False.
    :param init_params: Paramètres d'initialisation utilisés pour encoder les données classiques QTensor, la valeur par défaut est None.
    :param wires: Index du bit de l'effet de ligne, la valeur par défaut est None.
    :param dtype: La précision des données de la matrice interne de la porte logique peut être définie sur pyvqnet.kcomplex64 ou pyvqnet.kcomplex128, correspondant respectivement à une entrée float ou double.
    :param use_dagger: indique s'il faut utiliser la version conjuguée transposée de la porte, la valeur par défaut est False.
    :return: une instance ``pyvqnet.qnn.vqc.torch.QModule``

    Example::

        from pyvqnet.qnn.vqc.torch import SWAP,QMachine
        import pyvqnet
        pyvqnet.backends.set_backend("torch")
        device = QMachine(4)
        layer = SWAP(wires=[0,1])
        batchsize = 2
        device.reset_states(batchsize)
        layer(q_machine = device)
        print(device.states)


CSWAP
""""""""""""""""""

.. py:class:: pyvqnet.qnn.vqc.torch.CSWAP(has_params: bool = False,trainable: bool = False,init_params=None,wires=None,dtype=pyvqnet.kcomplex64,use_dagger=False)
    
    définit une porte quantique CSWAP.

    .. math:: CSWAP = \begin{bmatrix}
            1 & 0 & 0 & 0 & 0 & 0 & 0 & 0 \\
            0 & 1 & 0 & 0 & 0 & 0 & 0 & 0 \\
            0 & 0 & 1 & 0 & 0 & 0 & 0 & 0 \\
            0 & 0 & 0 & 1 & 0 & 0 & 0 & 0 \\
            0 & 0 & 0 & 0 & 1 & 0 & 0 & 0 \\
            0 & 0 & 0 & 0 & 0 & 0 & 1 & 0 \\
            0 & 0 & 0 & 0 & 0 & 1 & 0 & 0 \\
            0 & 0 & 0 & 0 & 0 & 0 & 0 & 1
        \end{bmatrix}.

    This class inherits from ``pyvqnet.qnn.vqc.torch.QModule`` and ``torch.nn.Module``.
    Cette classe peut être ajoutée au modèle torch comme sous-module de ``torch.nn.Module``.

    :param has_params: indique s'il a des paramètres, comme les portes RX, RY, etc. qui doivent être définies sur True, et celles sans paramètres doivent être définies sur False, la valeur par défaut est False.
    :param trainable: indique s'il a des paramètres à entraîner. Si la couche utilise des données d'entrée externes pour construire la matrice de la porte logique, définir sur False. Si les paramètres à entraîner doivent être initialisés à partir de cette couche, c'est True, la valeur par défaut est False.
    :param init_params: Paramètres d'initialisation utilisés pour encoder les données classiques QTensor, la valeur par défaut est None.
    :param wires: Index du bit de l'effet de ligne, la valeur par défaut est None.
    :param dtype: La précision des données de la matrice interne de la porte logique peut être définie sur pyvqnet.kcomplex64 ou pyvqnet.kcomplex128, correspondant respectivement à une entrée float ou double.
    :param use_dagger: indique s'il faut utiliser la version conjuguée transposée de la porte, la valeur par défaut est False.
    :return: une instance ``pyvqnet.qnn.vqc.torch.QModule``

    Example::

        from pyvqnet.qnn.vqc.torch import CSWAP,QMachine
        import pyvqnet
        pyvqnet.backends.set_backend("torch")
        device = QMachine(4)
        layer = CSWAP(wires=[0,1,2])
        batchsize = 2
        device.reset_states(batchsize)
        layer(q_machine = device)
        print(device.states)

RXX
""""""""""""""""""


.. py:class:: pyvqnet.qnn.vqc.torch.RXX(has_params: bool = False,trainable: bool = False,init_params=None,wires=None,dtype=pyvqnet.kcomplex64,use_dagger=False)
    
    définit une porte quantique RXX.

    This class inherits from ``pyvqnet.qnn.vqc.torch.QModule`` and ``torch.nn.Module``.
    Cette classe peut être ajoutée au modèle torch comme sous-module de ``torch.nn.Module``.

    :param has_params: indique s'il a des paramètres, comme les portes RX, RY, etc. qui doivent être définies sur True, et celles sans paramètres doivent être définies sur False, la valeur par défaut est False.
    :param trainable: indique s'il a des paramètres à entraîner. Si la couche utilise des données d'entrée externes pour construire la matrice de la porte logique, définir sur False. Si les paramètres à entraîner doivent être initialisés à partir de cette couche, c'est True, la valeur par défaut est False.
    :param init_params: Paramètres d'initialisation utilisés pour encoder les données classiques QTensor, la valeur par défaut est None.
    :param wires: Index du bit de l'effet de ligne, la valeur par défaut est None.
    :param dtype: La précision des données de la matrice interne de la porte logique peut être définie sur pyvqnet.kcomplex64 ou pyvqnet.kcomplex128, correspondant respectivement à une entrée float ou double.
    :param use_dagger: indique s'il faut utiliser la version conjuguée transposée de la porte, la valeur par défaut est False.
    :return: une instance ``pyvqnet.qnn.vqc.torch.QModule``

    Example::

        from pyvqnet.qnn.vqc.torch import RXX,QMachine
        import pyvqnet
        pyvqnet.backends.set_backend("torch")
        device = QMachine(4)
        layer = RXX(has_params= True, trainable= True, wires=[0,2])
        batchsize = 2
        device.reset_states(batchsize)
        layer(q_machine = device)
        print(device.states)

RYY
""""""""""""""""""

.. py:class:: pyvqnet.qnn.vqc.torch.RYY(has_params: bool = False,trainable: bool = False,init_params=None,wires=None,dtype=pyvqnet.kcomplex64,use_dagger=False)
    
    définit une porte quantique RYY.

    This class inherits from ``pyvqnet.qnn.vqc.torch.QModule`` and ``torch.nn.Module``.
    Cette classe peut être ajoutée au modèle torch comme sous-module de ``torch.nn.Module``.

    :param has_params: indique s'il a des paramètres, comme les portes RX, RY, etc. qui doivent être définies sur True, et celles sans paramètres doivent être définies sur False, la valeur par défaut est False.
    :param trainable: indique s'il a des paramètres à entraîner. Si la couche utilise des données d'entrée externes pour construire la matrice de la porte logique, définir sur False. Si les paramètres à entraîner doivent être initialisés à partir de cette couche, c'est True, la valeur par défaut est False.
    :param init_params: Paramètres d'initialisation utilisés pour encoder les données classiques QTensor, la valeur par défaut est None.
    :param wires: Index du bit de l'effet de ligne, la valeur par défaut est None.
    :param dtype: La précision des données de la matrice interne de la porte logique peut être définie sur pyvqnet.kcomplex64 ou pyvqnet.kcomplex128, correspondant respectivement à une entrée float ou double.
    :param use_dagger: indique s'il faut utiliser la version conjuguée transposée de la porte, la valeur par défaut est False.
    :return: une instance ``pyvqnet.qnn.vqc.torch.QModule``

    Example::

        from pyvqnet.qnn.vqc.torch import RYY,QMachine
        import pyvqnet
        pyvqnet.backends.set_backend("torch")
        device = QMachine(4)
        layer = RYY(has_params= True, trainable= True, wires=[0,2])
        batchsize = 2
        device.reset_states(batchsize)
        layer(q_machine = device)
        print(device.states)


RZZ
""""""""""""""""""

.. py:class:: pyvqnet.qnn.vqc.torch.RZZ(has_params: bool = False,trainable: bool = False,init_params=None,wires=None,dtype=pyvqnet.kcomplex64,use_dagger=False)
    
    définit une porte quantique RZZ.

    This class inherits from ``pyvqnet.qnn.vqc.torch.QModule`` and ``torch.nn.Module``.
    Cette classe peut être ajoutée au modèle torch comme sous-module de ``torch.nn.Module``.

    :param has_params: indique s'il a des paramètres, comme les portes RX, RY, etc. qui doivent être définies sur True, et celles sans paramètres doivent être définies sur False, la valeur par défaut est False.
    :param trainable: indique s'il a des paramètres à entraîner. Si la couche utilise des données d'entrée externes pour construire la matrice de la porte logique, définir sur False. Si les paramètres à entraîner doivent être initialisés à partir de cette couche, c'est True, la valeur par défaut est False.
    :param init_params: Paramètres d'initialisation utilisés pour encoder les données classiques QTensor, la valeur par défaut est None.
    :param wires: Index du bit de l'effet de ligne, la valeur par défaut est None.
    :param dtype: La précision des données de la matrice interne de la porte logique peut être définie sur pyvqnet.kcomplex64 ou pyvqnet.kcomplex128, correspondant respectivement à une entrée float ou double.
    :param use_dagger: indique s'il faut utiliser la version conjuguée transposée de la porte, la valeur par défaut est False.
    :return: une instance ``pyvqnet.qnn.vqc.torch.QModule``

    Example::

        from pyvqnet.qnn.vqc.torch import RZZ,QMachine
        import pyvqnet
        pyvqnet.backends.set_backend("torch")
        device = QMachine(4)
        layer = RZZ(has_params= True, trainable= True, wires=[0,2])
        batchsize = 2
        device.reset_states(batchsize)
        layer(q_machine = device)
        print(device.states)



RZX
""""""""""""""""""

.. py:class:: pyvqnet.qnn.vqc.torch.RZX(has_params: bool = False,trainable: bool = False,init_params=None,wires=None,dtype=pyvqnet.kcomplex64,use_dagger=False)
    
    définit une porte quantique RZX.

    This class inherits from ``pyvqnet.qnn.vqc.torch.QModule`` and ``torch.nn.Module``.
    Cette classe peut être ajoutée au modèle torch comme sous-module de ``torch.nn.Module``.

    :param has_params: indique s'il a des paramètres, comme les portes RX, RY, etc. qui doivent être définies sur True, et celles sans paramètres doivent être définies sur False, la valeur par défaut est False.
    :param trainable: indique s'il a des paramètres à entraîner. Si la couche utilise des données d'entrée externes pour construire la matrice de la porte logique, définir sur False. Si les paramètres à entraîner doivent être initialisés à partir de cette couche, c'est True, la valeur par défaut est False.
    :param init_params: Paramètres d'initialisation utilisés pour encoder les données classiques QTensor, la valeur par défaut est None.
    :param wires: Index du bit de l'effet de ligne, la valeur par défaut est None.
    :param dtype: La précision des données de la matrice interne de la porte logique peut être définie sur pyvqnet.kcomplex64 ou pyvqnet.kcomplex128, correspondant respectivement à une entrée float ou double.
    :param use_dagger: indique s'il faut utiliser la version conjuguée transposée de la porte, la valeur par défaut est False.
    :return: une instance ``pyvqnet.qnn.vqc.torch.QModule``

    Example::

        from pyvqnet.qnn.vqc.torch import RZX,QMachine
        import pyvqnet
        pyvqnet.backends.set_backend("torch")
        device = QMachine(4)
        layer = RZX(has_params= True, trainable= True, wires=[0,2])
        batchsize = 2
        device.reset_states(batchsize)
        layer(q_machine = device)
        print(device.states)

Toffoli
""""""""""""""""""

.. py:class:: pyvqnet.qnn.vqc.torch.Toffoli(has_params: bool = False,trainable: bool = False,init_params=None,wires=None,dtype=pyvqnet.kcomplex64,use_dagger=False)
    
    définit une porte quantique Toffoli.

    This class inherits from ``pyvqnet.qnn.vqc.torch.QModule`` and ``torch.nn.Module``.
    Cette classe peut être ajoutée au modèle torch comme sous-module de ``torch.nn.Module``.

    :param has_params: indique s'il a des paramètres, comme les portes RX, RY, etc. qui doivent être définies sur True, et celles sans paramètres doivent être définies sur False, la valeur par défaut est False.
    :param trainable: indique s'il a des paramètres à entraîner. Si la couche utilise des données d'entrée externes pour construire la matrice de la porte logique, définir sur False. Si les paramètres à entraîner doivent être initialisés à partir de cette couche, c'est True, la valeur par défaut est False.
    :param init_params: Paramètres d'initialisation utilisés pour encoder les données classiques QTensor, la valeur par défaut est None.
    :param wires: Index du bit de l'effet de ligne, la valeur par défaut est None.
    :param dtype: La précision des données de la matrice interne de la porte logique peut être définie sur pyvqnet.kcomplex64 ou pyvqnet.kcomplex128, correspondant respectivement à une entrée float ou double.
    :param use_dagger: indique s'il faut utiliser la version conjuguée transposée de la porte, la valeur par défaut est False.
    :return: une instance ``pyvqnet.qnn.vqc.torch.QModule``

    Example::

        from pyvqnet.qnn.vqc.torch import Toffoli,QMachine
        import pyvqnet
        pyvqnet.backends.set_backend("torch")
        device = QMachine(4)
        layer = Toffoli(wires=[0,2,1])
        batchsize = 2
        device.reset_states(batchsize)
        layer(q_machine = device)
        print(device.states)

IsingXX
""""""""""""""""""


.. py:class:: pyvqnet.qnn.vqc.torch.IsingXX(has_params: bool = False,trainable: bool = False,init_params=None,wires=None,dtype=pyvqnet.kcomplex64,use_dagger=False)
    
    définit une porte quantique IsingXX.

    This class inherits from ``pyvqnet.qnn.vqc.torch.QModule`` and ``torch.nn.Module``.
    Cette classe peut être ajoutée au modèle torch comme sous-module de ``torch.nn.Module``.

    :param has_params: indique s'il a des paramètres, comme les portes RX, RY, etc. qui doivent être définies sur True, et celles sans paramètres doivent être définies sur False, la valeur par défaut est False.
    :param trainable: indique s'il a des paramètres à entraîner. Si la couche utilise des données d'entrée externes pour construire la matrice de la porte logique, définir sur False. Si les paramètres à entraîner doivent être initialisés à partir de cette couche, c'est True, la valeur par défaut est False.
    :param init_params: Paramètres d'initialisation utilisés pour encoder les données classiques QTensor, la valeur par défaut est None.
    :param wires: Index du bit de l'effet de ligne, la valeur par défaut est None.
    :param dtype: La précision des données de la matrice interne de la porte logique peut être définie sur pyvqnet.kcomplex64 ou pyvqnet.kcomplex128, correspondant respectivement à une entrée float ou double.
    :param use_dagger: indique s'il faut utiliser la version conjuguée transposée de la porte, la valeur par défaut est False.
    :return: une instance ``pyvqnet.qnn.vqc.torch.QModule``

    Example::

        from pyvqnet.qnn.vqc.torch import IsingXX,QMachine
        import pyvqnet
        pyvqnet.backends.set_backend("torch")
        device = QMachine(4)
        layer = IsingXX(has_params= True, trainable= True, wires=[0,2])
        batchsize = 2
        device.reset_states(batchsize)
        layer(q_machine = device)
        print(device.states)


IsingYY
""""""""""""""""""


.. py:class:: pyvqnet.qnn.vqc.torch.IsingYY(has_params: bool = False,trainable: bool = False,init_params=None,wires=None,dtype=pyvqnet.kcomplex64,use_dagger=False)
    
    définit une porte quantique IsingYY.

    This class inherits from ``pyvqnet.qnn.vqc.torch.QModule`` and ``torch.nn.Module``.
    Cette classe peut être ajoutée au modèle torch comme sous-module de ``torch.nn.Module``.

    :param has_params: indique s'il a des paramètres, comme les portes RX, RY, etc. qui doivent être définies sur True, et celles sans paramètres doivent être définies sur False, la valeur par défaut est False.
    :param trainable: indique s'il a des paramètres à entraîner. Si la couche utilise des données d'entrée externes pour construire la matrice de la porte logique, définir sur False. Si les paramètres à entraîner doivent être initialisés à partir de cette couche, c'est True, la valeur par défaut est False.
    :param init_params: Paramètres d'initialisation utilisés pour encoder les données classiques QTensor, la valeur par défaut est None.
    :param wires: Index du bit de l'effet de ligne, la valeur par défaut est None.
    :param dtype: La précision des données de la matrice interne de la porte logique peut être définie sur pyvqnet.kcomplex64 ou pyvqnet.kcomplex128, correspondant respectivement à une entrée float ou double.
    :param use_dagger: indique s'il faut utiliser la version conjuguée transposée de la porte, la valeur par défaut est False.
    :return: une instance ``pyvqnet.qnn.vqc.torch.QModule``

    Example::

        from pyvqnet.qnn.vqc.torch import IsingYY,QMachine
        import pyvqnet
        pyvqnet.backends.set_backend("torch")
        device = QMachine(4)
        layer = IsingYY(has_params= True, trainable= True, wires=[0,2])
        batchsize = 2
        device.reset_states(batchsize)
        layer(q_machine = device)
        print(device.states)


IsingZZ
""""""""""""""""""

.. py:class:: pyvqnet.qnn.vqc.torch.IsingZZ(has_params: bool = False,trainable: bool = False,init_params=None,wires=None,dtype=pyvqnet.kcomplex64,use_dagger=False)
    
    définit une porte quantique IsingZZ.

    This class inherits from ``pyvqnet.qnn.vqc.torch.QModule`` and ``torch.nn.Module``.
    Cette classe peut être ajoutée au modèle torch comme sous-module de ``torch.nn.Module``.

    :param has_params: indique s'il a des paramètres, comme les portes RX, RY, etc. qui doivent être définies sur True, et celles sans paramètres doivent être définies sur False, la valeur par défaut est False.
    :param trainable: indique s'il a des paramètres à entraîner. Si la couche utilise des données d'entrée externes pour construire la matrice de la porte logique, définir sur False. Si les paramètres à entraîner doivent être initialisés à partir de cette couche, c'est True, la valeur par défaut est False.
    :param init_params: Paramètres d'initialisation utilisés pour encoder les données classiques QTensor, la valeur par défaut est None.
    :param wires: Index du bit de l'effet de ligne, la valeur par défaut est None.
    :param dtype: La précision des données de la matrice interne de la porte logique peut être définie sur pyvqnet.kcomplex64 ou pyvqnet.kcomplex128, correspondant respectivement à une entrée float ou double.
    :param use_dagger: indique s'il faut utiliser la version conjuguée transposée de la porte, la valeur par défaut est False.
    :return: une instance ``pyvqnet.qnn.vqc.torch.QModule``

    Example::

        from pyvqnet.qnn.vqc.torch import IsingZZ,QMachine
        import pyvqnet
        pyvqnet.backends.set_backend("torch")
        device = QMachine(4)
        layer = IsingZZ(has_params= True, trainable= True, wires=[0,2])
        batchsize = 2
        device.reset_states(batchsize)
        layer(q_machine = device)
        print(device.states)


IsingXY
""""""""""""""""""


.. py:class:: pyvqnet.qnn.vqc.torch.IsingXY(has_params: bool = False,trainable: bool = False,init_params=None,wires=None,dtype=pyvqnet.kcomplex64,use_dagger=False)
    
    définit une porte quantique IsingXY.

    This class inherits from ``pyvqnet.qnn.vqc.torch.QModule`` and ``torch.nn.Module``.
    Cette classe peut être ajoutée au modèle torch comme sous-module de ``torch.nn.Module``.

    :param has_params: indique s'il a des paramètres, comme les portes RX, RY, etc. qui doivent être définies sur True, et celles sans paramètres doivent être définies sur False, la valeur par défaut est False.
    :param trainable: indique s'il a des paramètres à entraîner. Si la couche utilise des données d'entrée externes pour construire la matrice de la porte logique, définir sur False. Si les paramètres à entraîner doivent être initialisés à partir de cette couche, c'est True, la valeur par défaut est False.
    :param init_params: Paramètres d'initialisation utilisés pour encoder les données classiques QTensor, la valeur par défaut est None.
    :param wires: Index du bit de l'effet de ligne, la valeur par défaut est None.
    :param dtype: La précision des données de la matrice interne de la porte logique peut être définie sur pyvqnet.kcomplex64 ou pyvqnet.kcomplex128, correspondant respectivement à une entrée float ou double.
    :param use_dagger: indique s'il faut utiliser la version conjuguée transposée de la porte, la valeur par défaut est False.
    :return: une instance ``pyvqnet.qnn.vqc.torch.QModule``

    Example::

        from pyvqnet.qnn.vqc.torch import IsingXY,QMachine
        import pyvqnet
        pyvqnet.backends.set_backend("torch")
        device = QMachine(4)
        layer = IsingXY(has_params= True, trainable= True, wires=[0,2])
        batchsize = 2
        device.reset_states(batchsize)
        layer(q_machine = device)
        print(device.states)


PhaseShift
""""""""""""""""""


.. py:class:: pyvqnet.qnn.vqc.torch.PhaseShift(has_params: bool = False,trainable: bool = False,init_params=None,wires=None,dtype=pyvqnet.kcomplex64,use_dagger=False)
    
    définit une porte quantique PhaseShift.

    This class inherits from ``pyvqnet.qnn.vqc.torch.QModule`` and ``torch.nn.Module``.
    Cette classe peut être ajoutée au modèle torch comme sous-module de ``torch.nn.Module``.

    :param has_params: indique s'il a des paramètres, comme les portes RX, RY, etc. qui doivent être définies sur True, et celles sans paramètres doivent être définies sur False, la valeur par défaut est False.
    :param trainable: indique s'il a des paramètres à entraîner. Si la couche utilise des données d'entrée externes pour construire la matrice de la porte logique, définir sur False. Si les paramètres à entraîner doivent être initialisés à partir de cette couche, c'est True, la valeur par défaut est False.
    :param init_params: Paramètres d'initialisation utilisés pour encoder les données classiques QTensor, la valeur par défaut est None.
    :param wires: Index du bit de l'effet de ligne, la valeur par défaut est None.
    :param dtype: La précision des données de la matrice interne de la porte logique peut être définie sur pyvqnet.kcomplex64 ou pyvqnet.kcomplex128, correspondant respectivement à une entrée float ou double.
    :param use_dagger: indique s'il faut utiliser la version conjuguée transposée de la porte, la valeur par défaut est False.
    :return: une instance ``pyvqnet.qnn.vqc.torch.QModule``

    Example::

        from pyvqnet.qnn.vqc.torch import PhaseShift,QMachine
        import pyvqnet
        pyvqnet.backends.set_backend("torch")
        device = QMachine(4)
        layer = PhaseShift(has_params= True, trainable= True, wires=1)
        batchsize = 2
        device.reset_states(batchsize)
        layer(q_machine = device)
        print(device.states)


MultiRZ
""""""""""""""""""

.. py:class:: pyvqnet.qnn.vqc.torch.MultiRZ(has_params: bool = False,trainable: bool = False,init_params=None,wires=None,dtype=pyvqnet.kcomplex64,use_dagger=False)
    
    définit une porte quantique MultiRZ.

    This class inherits from ``pyvqnet.qnn.vqc.torch.QModule`` and ``torch.nn.Module``.
    Cette classe peut être ajoutée au modèle torch comme sous-module de ``torch.nn.Module``.

    :param has_params: indique s'il a des paramètres, comme les portes RX, RY, etc. qui doivent être définies sur True, et celles sans paramètres doivent être définies sur False, la valeur par défaut est False.
    :param trainable: indique s'il a des paramètres à entraîner. Si la couche utilise des données d'entrée externes pour construire la matrice de la porte logique, définir sur False. Si les paramètres à entraîner doivent être initialisés à partir de cette couche, c'est True, la valeur par défaut est False.
    :param init_params: Paramètres d'initialisation utilisés pour encoder les données classiques QTensor, la valeur par défaut est None.
    :param wires: Index du bit de l'effet de ligne, la valeur par défaut est None.
    :param dtype: La précision des données de la matrice interne de la porte logique peut être définie sur pyvqnet.kcomplex64 ou pyvqnet.kcomplex128, correspondant respectivement à une entrée float ou double.
    :param use_dagger: indique s'il faut utiliser la version conjuguée transposée de la porte, la valeur par défaut est False.
    :return: une instance ``pyvqnet.qnn.vqc.torch.QModule``

    Example::

        from pyvqnet.qnn.vqc.torch import MultiRZ,QMachine
        import pyvqnet
        pyvqnet.backends.set_backend("torch")
        device = QMachine(4)
        layer = MultiRZ(has_params= True, trainable= True, wires=[0,2])
        batchsize = 2
        device.reset_states(batchsize)
        layer(q_machine = device)
        print(device.states)



SDG
""""""""""""""""""


.. py:class:: pyvqnet.qnn.vqc.torch.SDG(has_params: bool = False,trainable: bool = False,init_params=None,wires=None,dtype=pyvqnet.kcomplex64,use_dagger=False)
    
    définit une porte quantique TDG.

    This class inherits from ``pyvqnet.qnn.vqc.torch.QModule`` and ``torch.nn.Module``.
    Cette classe peut être ajoutée au modèle torch comme sous-module de ``torch.nn.Module``.

    :param has_params: indique s'il a des paramètres, comme les portes RX, RY, etc. qui doivent être définies sur True, et celles sans paramètres doivent être définies sur False, la valeur par défaut est False.
    :param trainable: indique s'il a des paramètres à entraîner. Si la couche utilise des données d'entrée externes pour construire la matrice de la porte logique, définir sur False. Si les paramètres à entraîner doivent être initialisés à partir de cette couche, c'est True, la valeur par défaut est False.
    :param init_params: Paramètres d'initialisation utilisés pour encoder les données classiques QTensor, la valeur par défaut est None.
    :param wires: Index du bit de l'effet de ligne, la valeur par défaut est None.
    :param dtype: La précision des données de la matrice interne de la porte logique peut être définie sur pyvqnet.kcomplex64 ou pyvqnet.kcomplex128, correspondant respectivement à une entrée float ou double.
    :param use_dagger: indique s'il faut utiliser la version conjuguée transposée de la porte, la valeur par défaut est False.
    :return: une instance ``pyvqnet.qnn.vqc.torch.QModule``

    Example::
        
        from pyvqnet.qnn.vqc.torch import SDG,QMachine
        import pyvqnet
        pyvqnet.backends.set_backend("torch")
        device = QMachine(4)
        layer = SDG(wires=0)
        batchsize = 1
        device.reset_states(1)
        layer(q_machine = device)
        print(device.states)




TDG
""""""""""""""""""

.. py:class:: pyvqnet.qnn.vqc.torch.TDG(has_params: bool = False,trainable: bool = False,init_params=None,wires=None,dtype=pyvqnet.kcomplex64,use_dagger=False)
    
    définit une porte quantique TDG.

    This class inherits from ``pyvqnet.qnn.vqc.torch.QModule`` and ``torch.nn.Module``.
    Cette classe peut être ajoutée au modèle torch comme sous-module de ``torch.nn.Module``.

    :param has_params: indique s'il a des paramètres, comme les portes RX, RY, etc. qui doivent être définies sur True, et celles sans paramètres doivent être définies sur False, la valeur par défaut est False.
    :param trainable: indique s'il a des paramètres à entraîner. Si la couche utilise des données d'entrée externes pour construire la matrice de la porte logique, définir sur False. Si les paramètres à entraîner doivent être initialisés à partir de cette couche, c'est True, la valeur par défaut est False.
    :param init_params: Paramètres d'initialisation utilisés pour encoder les données classiques QTensor, la valeur par défaut est None.
    :param wires: Index du bit de l'effet de ligne, la valeur par défaut est None.
    :param dtype: La précision des données de la matrice interne de la porte logique peut être définie sur pyvqnet.kcomplex64 ou pyvqnet.kcomplex128, correspondant respectivement à une entrée float ou double.
    :param use_dagger: indique s'il faut utiliser la version conjuguée transposée de la porte, la valeur par défaut est False.
    :return: une instance ``pyvqnet.qnn.vqc.torch.QModule``

    Example::
        
        from pyvqnet.qnn.vqc.torch import TDG,QMachine
        import pyvqnet
        pyvqnet.backends.set_backend("torch")
        device = QMachine(4)
        layer = TDG(wires=0)
        batchsize = 1
        device.reset_states(1)
        layer(q_machine = device)
        print(device.states)



ControlledPhaseShift
"""""""""""""""""""""""""""""""""""""""""""""


.. py:class:: pyvqnet.qnn.vqc.torch.ControlledPhaseShift(has_params: bool = False,trainable: bool = False,init_params=None,wires=None,dtype=pyvqnet.kcomplex64,use_dagger=False)
    
    définit une porte quantique ControlledPhaseShift.

    This class inherits from ``pyvqnet.qnn.vqc.torch.QModule`` and ``torch.nn.Module``.
    Cette classe peut être ajoutée au modèle torch comme sous-module de ``torch.nn.Module``.

    :param has_params: indique s'il a des paramètres, comme les portes RX, RY, etc. qui doivent être définies sur True, et celles sans paramètres doivent être définies sur False, la valeur par défaut est False.
    :param trainable: indique s'il a des paramètres à entraîner. Si la couche utilise des données d'entrée externes pour construire la matrice de la porte logique, définir sur False. Si les paramètres à entraîner doivent être initialisés à partir de cette couche, c'est True, la valeur par défaut est False.
    :param init_params: Paramètres d'initialisation utilisés pour encoder les données classiques QTensor, la valeur par défaut est None.
    :param wires: Index du bit de l'effet de ligne, la valeur par défaut est None.
    :param dtype: La précision des données de la matrice interne de la porte logique peut être définie sur pyvqnet.kcomplex64 ou pyvqnet.kcomplex128, correspondant respectivement à une entrée float ou double.
    :param use_dagger: indique s'il faut utiliser la version conjuguée transposée de la porte, la valeur par défaut est False.
    :return: une instance ``pyvqnet.qnn.vqc.torch.QModule``

    Example::

        from pyvqnet.qnn.vqc.torch import ControlledPhaseShift,QMachine
        import pyvqnet
        pyvqnet.backends.set_backend("torch")
        device = QMachine(4)
        layer = ControlledPhaseShift(has_params= True, trainable= True, wires=[0,2])
        batchsize = 2
        device.reset_states(batchsize)
        layer(q_machine = device)
        print(device.states)



MultiControlledX
"""""""""""""""""""""""""""""""""""""""""""""

.. py:class:: pyvqnet.qnn.vqc.torch.MultiControlledX(has_params: bool = False,trainable: bool = False,init_params=None,wires=None,dtype=pyvqnet.kcomplex64,use_dagger=False,control_values=None)
    
    définit une porte quantique MultiControlledX.

    This class inherits from ``pyvqnet.qnn.vqc.torch.QModule`` and ``torch.nn.Module``.
    Cette classe peut être ajoutée au modèle torch comme sous-module de ``torch.nn.Module``.
    
    :param has_params: indique s'il a des paramètres, comme les portes RX, RY, etc. qui doivent être définies sur True, et celles sans paramètres doivent être définies sur False, la valeur par défaut est False.
    :param trainable: indique s'il a des paramètres à entraîner. Si la couche utilise des données d'entrée externes pour construire la matrice de la porte logique, définir sur False. Si les paramètres à entraîner doivent être initialisés à partir de cette couche, c'est True, la valeur par défaut est False.
    :param init_params: Paramètres d'initialisation utilisés pour encoder les données classiques QTensor, la valeur par défaut est None.
    :param wires: Index du bit de l'effet de ligne, la valeur par défaut est None.
    :param dtype: La précision des données de la matrice interne de la porte logique peut être définie sur pyvqnet.kcomplex64 ou pyvqnet.kcomplex128, correspondant respectivement à une entrée float ou double.
    :param use_dagger: indique s'il faut utiliser la version conjuguée transposée de la porte, la valeur par défaut est False.
    :param control_values: Control value, the default is None, when the bit is 1, it is controlled.

    :return: une instance ``pyvqnet.qnn.vqc.torch.QModule``

    Example::

        import pyvqnet
        pyvqnet.backends.set_backend("torch")
        from pyvqnet.qnn.vqc.torch import QMachine,MultiControlledX
        from pyvqnet.tensor import QTensor,kcomplex64
        qm = QMachine(4,dtype=kcomplex64)
        qm.reset_states(2)
        mcx = MultiControlledX( 
                        init_params=None,
                        wires=[2,3,0,1],
                        dtype=kcomplex64,
                        use_dagger=False,control_values=[1,0,0])
        y = mcx(q_machine = qm)
        print(qm.states)


API de mesures
^^^^^^^^^^^^^^^^^^^^^^

Probability
"""""""""""""""""""""""""""""


.. py:class:: pyvqnet.qnn.vqc.torch.Probability(wires=None, name="")

    Calcule le résultat de mesure de probabilité du circuit quantique sur un bit spécifique.

    This class inherits from ``pyvqnet.qnn.vqc.torch.QModule`` and ``torch.nn.Module``.
    Cette classe peut être ajoutée au modèle torch comme sous-module de ``torch.nn.Module``.

    :param wires: L'indice du bit de mesure, liste, tuple ou entier.
    :param name: The name of the module, default: "".
    :return: Le résultat de la mesure, QTensor.

    Example::

        import pyvqnet
        pyvqnet.backends.set_backend("torch")
        from pyvqnet.qnn.vqc.torch import Probability,rx,ry,cnot,QMachine,rz
        from pyvqnet.tensor import QTensor
        from pyvqnet import kfloat64
        x = QTensor([[0.56, 0.1],[0.56, 0.1]],requires_grad=True)
        qm = QMachine(4)
        qm.reset_states(2)
        rz(q_machine=qm,wires=0,params=x[:,[0]])
        rz(q_machine=qm,wires=1,params=x[:,[0]])
        cnot(q_machine=qm,wires=[0,1])
        ry(q_machine=qm,wires=2,params=x[:,[1]])
        cnot(q_machine=qm,wires=[0,2])
        rz(q_machine=qm,wires=3,params=x[:,[1]])
        ma = Probability(wires=1)
        y =ma(q_machine=qm)


MeasureAll
"""""""""""""""""""""""""""""

.. py:class:: pyvqnet.qnn.vqc.torch.MeasureAll(obs=None, name="")

    Calcule les résultats de mesure des circuits quantiques, supporte l'entrée obs comme plusieurs ou un seul opérateur de Pauli ou Hamiltonien.
    For example:

    {\'X0\': 0.23} indique un effet PauliX sur le qubit 0, avec un coefficient de 0.23.

    {\'X1 Z2\': 2.4,\'Y2\': -0.5} correspond à la valeur observée 2.4 * X1 @ Z2 - 0.5 * Y2.

    [{\'X1 Z2\': 4,\'Z1 Z0\': 3},{\'X1 Y2 Z0\': 3.5}] correspond aux deux valeurs observées 4 * X1 @ Z2 + 3 * Z1 @ Z0 et 3.5 * X1 @ Y2 @ Z0.
        
        
    This class inherits from ``pyvqnet.qnn.vqc.torch.QModule`` and ``torch.nn.Module``.

    Cette classe peut être ajoutée au modèle torch comme sous-module de ``torch.nn.Module``.

    :param obs: observable.
    :param name: nom du module, par défaut : "".
    :return: résultat de la mesure, QTensor.

    Example::

        import pyvqnet
        pyvqnet.backends.set_backend("torch")
        from pyvqnet.qnn.vqc.torch import MeasureAll,rx,ry,cnot,QMachine,rz
        from pyvqnet.tensor import QTensor
        from pyvqnet import kfloat64
        x = QTensor([[0.56, 0.1],[0.56, 0.1]],requires_grad=True)
        qm = QMachine(4)
        qm.reset_states(2)
        rz(q_machine=qm,wires=0,params=x[:,[0]])
        rz(q_machine=qm,wires=1,params=x[:,[0]])
        cnot(q_machine=qm,wires=[0,1])
        ry(q_machine=qm,wires=2,params=x[:,[1]])
        cnot(q_machine=qm,wires=[0,2])
        rz(q_machine=qm,wires=3,params=x[:,[1]])
        obs_list = [{
            "Z0 Z1" :2
        }, {
            "X0 Z2" :1
        }]
        ma = MeasureAll(obs = obs_list)
        y = ma(q_machine=qm)
        print(y)



Samples
"""""""""""""""""""""""""""""

.. py:class:: pyvqnet.qnn.vqc.torch.Samples(wires=None, obs=None, shots = 1,name="")

    Obtient les résultats d'échantillonnage avec des tirs sur des fils spécifiques.

    This class inherits from ``pyvqnet.qnn.vqc.torch.QModule`` and ``torch.nn.Module``.
    Cette classe peut être ajoutée au modèle torch comme sous-module de ``torch.nn.Module``.


    :param wires: Indice du qubit d'échantillonnage. Valeur par défaut : None, utilise tous les bits du simulateur au moment de l'exécution.
    :param obs: Cette valeur ne peut être que None.
    :param shots: Nombre de répétitions d'échantillonnage, valeur par défaut : 1.
    :param name: Le nom de ce module, valeur par défaut : "".
    :return: une classe de méthode de mesure

    Example::

        import pyvqnet
        pyvqnet.backends.set_backend("torch")
        from pyvqnet.qnn.vqc.torch import Samples,rx,ry,cnot,QMachine,rz
        from pyvqnet.tensor import QTensor
        from pyvqnet import kfloat64
        x = QTensor([[0.56, 0.1],[0.56, 0.1]],requires_grad=True)

        qm = QMachine(4)
        qm.reset_states(2)
        rz(q_machine=qm,wires=0,params=x[:,[0]])
        rx(q_machine=qm,wires=1,params=x[:,[0]])
        cnot(q_machine=qm,wires=[0,1])

        cnot(q_machine=qm,wires=[0,2])
        ry(q_machine=qm,wires=3,params=x[:,[1]])


        ma = Samples(wires=[0,1,2],shots=3)
        y = ma(q_machine=qm)
        print(y)


HermitianExpval
"""""""""""""""""""""""""""""

.. py:class:: pyvqnet.qnn.vqc.torch.HermitianExpval(obs=None, name="")

    Calcule l'espérance d'une quantité hermitienne dans un circuit quantique.

    This class inherits from ``pyvqnet.qnn.vqc.torch.QModule`` and ``torch.nn.Module``.
    Cette classe peut être ajoutée au modèle torch comme sous-module de ``torch.nn.Module``.


    :param obs: Quantité hermitienne.
    :param name: module name, default: "".
    :return: résultat attendu, QTensor.

    Example::

        import pyvqnet
        pyvqnet.backends.set_backend("torch")
        from pyvqnet.qnn.vqc.torch import QMachine, rx,ry,\
            RX, RY, CNOT, PauliX, PauliZ, VQC_RotCircuit,HermitianExpval
        from pyvqnet.tensor import QTensor, tensor
        from pyvqnet.nn import Parameter
        import numpy as np
        bsz = 3
        H = np.array([[8, 4, 0, -6], [4, 0, 4, 0], [0, 4, 8, 0], [-6, 0, 0, 0]])
        class QModel(pyvqnet.nn.Module):
            def __init__(self, num_wires, dtype):
                super(QModel, self).__init__()
                self.rot_param = Parameter((3, ))
                self.rot_param.copy_value_from(tensor.QTensor([-0.5, 1, 2.3]))
                self._num_wires = num_wires
                self._dtype = dtype
                self.qm = QMachine(num_wires, dtype=dtype)
                self.rx_layer1 = VQC_RotCircuit
                self.ry_layer2 = RY(has_params=True,
                                    trainable=True,
                                    wires=0,
                                    init_params=tensor.QTensor([-0.5]))
                self.xlayer = PauliX(wires=0)
                self.cnot = CNOT(wires=[0, 1])
                self.measure = HermitianExpval(obs = {'wires':(1,0),'observables':tensor.to_tensor(H)})

            def forward(self, x, *args, **kwargs):
                self.qm.reset_states(x.shape[0])

                rx(q_machine=self.qm, wires=0, params=x[:, [1]])
                ry(q_machine=self.qm, wires=1, params=x[:, [0]])
                self.xlayer(q_machine=self.qm)
                self.rx_layer1(params=self.rot_param, wire=1, q_machine=self.qm)
                self.ry_layer2(q_machine=self.qm)
                self.cnot(q_machine=self.qm)
                rlt = self.measure(q_machine = self.qm)

                return rlt


        input_x = tensor.arange(1, bsz * 2 + 1,
                                dtype=pyvqnet.kfloat32).reshape([bsz, 2])
        input_x.requires_grad = True

        qunatum_model = QModel(num_wires=2, dtype=pyvqnet.kcomplex64)

        batch_y = qunatum_model(input_x)
        batch_y.backward()

        print(batch_y)

Gabarits courants pour les circuits quantiques
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

VQC_HardwareEfficientAnsatz
""""""""""""""""""""""""""""""""""""""""


.. py:class:: pyvqnet.qnn.vqc.torch.VQC_HardwareEfficientAnsatz(n_qubits,single_rot_gate_list,entangle_gate="CNOT",entangle_rules='linear',depth=1,initial=None,dtype=None)

    Implémentation de Hardware Efficient Ansatz introduit dans l'article : `Hardware-efficient Variational Quantum Eigensolver for Small Molecules <https://arxiv.org/pdf/1704.05018.pdf>`__.

    This class inherits from ``pyvqnet.qnn.vqc.torch.QModule`` and ``torch.nn.Module``.

    Cette classe peut être ajoutée au modèle torch comme sous-module de ``torch.nn.Module``.


    :param n_qubits: Nombre de qubits.
    :param single_rot_gate_list: Une liste de portes de rotation à un qubit est construite par une ou plusieurs portes de rotation qui agissent sur chaque qubit. Supporte actuellement Rx, Ry, Rz.
    :param entangle_gate: La porte d'intrication non paramétrée. CNOT, CZ sont supportés. Défaut : CNOT.
    :param entangle_rules: Comment la porte d'intrication est utilisée dans le circuit. 'linear' signifie que la porte d'intrication agira sur chaque paire de qubits voisins. 'all' signifie que la porte d'intrication agira sur deux qubits quelconques. Défaut : linear.
    :param depth: La profondeur de l'ansatz, défaut : 1.
    :param initial: initie une même valeur pour tous les paramètres, défaut : None, ce module initialisera les paramètres aléatoirement.
    :param dtype: type de données des paramètres.
    :return: une instance VQC_HardwareEfficientAnsatz.

    Example::

        from pyvqnet.nn.torch import TorchModule,Linear,TorchModuleList
        from pyvqnet.qnn.vqc.torch.qcircuit import VQC_HardwareEfficientAnsatz,RZZ,RZ
        from pyvqnet.qnn.vqc.torch import Probability,QMachine
        from pyvqnet import tensor
        import pyvqnet
        pyvqnet.backends.set_backend("torch")
        pyvqnet.utils.set_random_seed(25)

        class QM(TorchModule):
            def __init__(self, name=""):
                super().__init__(name)
                self.linearx = Linear(4,2)
                self.ansatz = VQC_HardwareEfficientAnsatz(4, ["rx", "RY", "rz"],
                                            entangle_gate="cnot",
                                            entangle_rules="linear",
                                            depth=2)
                self.encode1 = RZ(wires=0)
                self.encode2 = RZ(wires=1)
                self.measure = Probability(wires=[0,2])
                self.device = QMachine(4)
            def forward(self, x, *args, **kwargs):
                self.device.reset_states(x.shape[0])
                y = self.linearx(x)
                self.encode1(params = y[:, [0]],q_machine = self.device,)
                self.encode2(params = y[:, [1]],q_machine = self.device,)
                self.ansatz(q_machine =self.device)
                return self.measure(q_machine =self.device)

        bz =3
        inputx = tensor.arange(1.0,bz*4+1).reshape([bz,4])
        inputx.requires_grad= True
        qlayer = QM()
        y = qlayer(inputx)
        y.backward()
        print(y)



VQC_BasicEntanglerTemplate
""""""""""""""""""""""""""""""""""""""""

.. py:class:: pyvqnet.qnn.vqc.torch.VQC_BasicEntanglerTemplate(num_layer=1, num_qubits=1, rotation="RX", initial=None, dtype=None)

    Une couche consistant en une rotation à un qubit et un paramètre sur chaque qubit, suivie de plusieurs portes CNOT dans une combinaison de chaîne fermée ou d'anneau.

    Un anneau de portes CNOT connecte chaque qubit à ses voisins, et finalement le a-ème qubit est considéré comme le voisin du a-ème qubit.

    This class inherits from ``pyvqnet.qnn.vqc.torch.QModule`` and ``torch.nn.Module``.

    Cette classe peut être ajoutée au modèle torch comme sous-module de ``torch.nn.Module``.

    :param num_layers: nombre de couches de répétition, défaut : 1.
    :param num_qubits: nombre de qubits, défaut : 1.
    :param rotation: porte à un qubit et un paramètre à utiliser, défaut : ``RX``
    :param initial: valeur initialisée identique pour tous les paramètres. défaut : None, les paramètres seront initialisés aléatoirement.
    :param dtype: type de données du paramètre, défaut : None, utilise float32.
    :return: Une instance VQC_BasicEntanglerTemplate

    Example::

        import pyvqnet
        pyvqnet.backends.set_backend("torch")
        from pyvqnet.qnn.vqc.torch import QModule,\
            VQC_BasicEntanglerTemplate, Probability, QMachine
        from pyvqnet import tensor


        class QM(QModule):
            def __init__(self, name=""):
                super().__init__(name)

                self.ansatz = VQC_BasicEntanglerTemplate(2,
                                                    4,
                                                    "rz",
                                                    initial=tensor.ones([1, 1]))

                self.measure = Probability(wires=[0, 2])
                self.device = QMachine(4)

            def forward(self,x, *args, **kwargs):

                self.ansatz(q_machine=self.device)
                return self.measure(q_machine=self.device)

        bz = 1
        inputx = tensor.arange(1.0, bz * 4 + 1).reshape([bz, 4])
        qlayer = QM()
        y = qlayer(inputx)
        y.backward()
        print(y)



VQC_StronglyEntanglingTemplate
""""""""""""""""""""""""""""""""""""""""

.. py:class:: pyvqnet.qnn.vqc.torch.VQC_StronglyEntanglingTemplate(num_layers=1, num_qubits=1, ranges=None,initial=None, dtype=None)

    Couches constituées de rotations à un qubit et d'intrications, comme dans `circuit-centric classifier design <https://arxiv.org/abs/1804.00633>`__.

    This class inherits from ``pyvqnet.qnn.vqc.torch.QModule`` and ``torch.nn.Module``.
    Cette classe peut être ajoutée au modèle torch comme sous-module de ``torch.nn.Module``.


    :param num_layers: number of repeat layers, default: 1.
    :param num_qubits: number of qubits, default: 1.
    :param ranges: séquence déterminant l'hyperparamètre de plage pour chaque couche suivante ; défaut : None
                                en utilisant :math:`r=l \mod M` pour la :math:`l`-ième couche et :math:`M` qubits.
    :param initial: valeur initiale pour tous les paramètres. défaut : None, initialisé aléatoirement.
    :param dtype: type de données du paramètre, défaut : None, utilise float32.
    :return: Une instance VQC_StronglyEntanglingTemplate.

    Example::

        from pyvqnet.nn.torch import TorchModule,Linear,TorchModuleList
        from pyvqnet.qnn.vqc.torch.qcircuit import VQC_StronglyEntanglingTemplate
        from pyvqnet.qnn.vqc.torch import Probability, QMachine
        from pyvqnet import tensor
        import pyvqnet

        pyvqnet.backends.set_backend("torch")
        pyvqnet.utils.set_random_seed(25)
        class QM(TorchModule):
            def __init__(self, name=""):
                super().__init__(name)

                self.ansatz = VQC_StronglyEntanglingTemplate(2,
                                                    4,
                                                    None,
                                                    initial=tensor.ones([1, 1]))

                self.measure = Probability(wires=[0, 1])
                self.device = QMachine(4)

            def forward(self,x, *args, **kwargs):

                self.ansatz(q_machine=self.device)
                return self.measure(q_machine=self.device)

        bz = 1
        inputx = tensor.arange(1.0, bz * 4 + 1).reshape([bz, 4])
        qlayer = QM()
        y = qlayer(inputx)
        y.backward()
        print(y)



VQC_QuantumEmbedding
""""""""""""""""""""""""""""""""""""""""


.. py:class:: pyvqnet.qnn.vqc.torch.VQC_QuantumEmbedding(qubits, machine, num_repetitions_input, depth_input, num_unitary_layers, num_repetitions,initial = None,dtype = None,name= "")
    
    Utilise RZ, RY, RZ pour créer des circuits quantiques variationnels afin d'encoder des données classiques dans des états quantiques.
    Référence `Quantum embeddings for machine learning <https://arxiv.org/abs/2001.03622>`_.

    This class inherits from ``pyvqnet.qnn.vqc.torch.QModule`` and ``torch.nn.Module``.
    Cette classe peut être ajoutée au modèle torch comme sous-module de ``torch.nn.Module``.

 
    :param num_repetitions_input: nombre de répétitions pour encoder l'entrée dans un sous-module.
    :param depth_input: nombre de dimensions d'entrée.
    :param num_unitary_layers: nombre de répétitions des portes quantiques variationnelles.
    :param num_repetitions: nombre de répétitions du sous-module.
    :param initial: valeur d'initialisation des paramètres, défaut est None
    :param dtype: type de paramètre, défaut est None, utilise float32.
    :param name: nom de la classe
    :return: Une instance VQC_QuantumEmbedding.

    Example::

        from pyvqnet.nn.torch import TorchModule
        from pyvqnet.qnn.vqc.torch.qcircuit import VQC_QuantumEmbedding
        from pyvqnet.qnn.vqc.torch import Probability, QMachine, MeasureAll
        from pyvqnet import tensor
        import pyvqnet

        pyvqnet.backends.set_backend("torch")
        pyvqnet.utils.set_random_seed(25)
        depth_input = 2
        num_repetitions = 2
        num_repetitions_input = 2
        num_unitary_layers = 2
        nq = depth_input * num_repetitions_input
        bz = 12

        class QM(TorchModule):
            def __init__(self, name=""):
                super().__init__(name)

                self.ansatz = VQC_QuantumEmbedding(num_repetitions_input, depth_input,
                                                num_unitary_layers,
                                                num_repetitions, initial=tensor.full([1],12.0),dtype=pyvqnet.kfloat32)

                self.measure = MeasureAll(obs={f"Z{nq-1}":1})
                self.device = QMachine(nq)

            def forward(self, x, *args, **kwargs):
                self.device.reset_states(x.shape[0])
                self.ansatz(x,q_machine=self.device)
                return self.measure(q_machine=self.device)

        inputx = tensor.arange(1.0, bz * depth_input + 1).reshape([bz, depth_input])
        qlayer = QM()
        y = qlayer(inputx)
        y.backward()
        print(y)


ExpressiveEntanglingAnsatz
""""""""""""""""""""""""""""""""""""""""

.. py:class:: pyvqnet.qnn.vqc.torch.ExpressiveEntanglingAnsatz(type: int, num_wires: int, depth: int, dtype=None, name: str = "")

    19 ansatz différents de l'article `Expressibility and entangling capability of parameterized quantum circuits for hybrid quantum-classical algorithms <https://arxiv.org/pdf/1905.10876.pdf>`_.

    This class inherits from ``pyvqnet.qnn.vqc.torch.QModule`` and ``torch.nn.Module``.

    Cette classe peut être ajoutée au modèle torch comme sous-module de ``torch.nn.Module``.

    :param type: Type de circuit de 1 à 19, un total de 19 types.
    :param num_wires: Number of qubits.
    :param depth: Profondeur du circuit.
    :param dtype: data type of parameter, default:None,use float32.
    :param name: Nom, défaut "".

    :return:
        une instance ExpressiveEntanglingAnsatz

    Example::

        from pyvqnet.nn.torch import TorchModule
        from pyvqnet.qnn.vqc.torch.qcircuit import ExpressiveEntanglingAnsatz
        from pyvqnet.qnn.vqc.torch import Probability, QMachine, MeasureAll
        from pyvqnet import tensor
        import pyvqnet

        pyvqnet.backends.set_backend("torch")
        pyvqnet.utils.set_random_seed(25)

        class QModel(TorchModule):
            def __init__(self, num_wires, dtype,grad_mode=""):
                super(QModel, self).__init__()

                self._num_wires = num_wires
                self._dtype = dtype
                self.qm = QMachine(num_wires, dtype=dtype,grad_mode=grad_mode)
                self.c1 = ExpressiveEntanglingAnsatz(1,3,2)
                self.measure = MeasureAll(obs={
                    "Z1":1
                })

            def forward(self, x, *args, **kwargs):
                self.qm.reset_states(x.shape[0])
                self.c1(q_machine = self.qm)
                rlt = self.measure(q_machine=self.qm)
                return rlt
            

        input_x = tensor.QTensor([[0.1, 0.2, 0.3]])

        qunatum_model = QModel(num_wires=3, dtype=pyvqnet.kcomplex64)

        batch_y = qunatum_model(input_x)
        batch_y.backward()
        print(batch_y)



vqc_basis_embedding
""""""""""""""""""""""""""""""""""""""""

.. py:function:: pyvqnet.qnn.vqc.torch.vqc_basis_embedding(basis_state,q_machine)

    Encode n caractéristiques binaires dans l'état de base à n qubits de ``q_machine``. Cette fonction est aliasée sous `VQC_BasisEmbedding`.

    For example, for ``basis_state=([0, 1, 1])``, the basis state in the quantum system is :math:`|011 \rangle`.

    :param basis_state: Entrée binaire de taille ``(n)``.
    :param q_machine: périphérique de machine quantique.
    

    Example::

        import pyvqnet
        pyvqnet.backends.set_backend("torch")
        from pyvqnet.qnn.vqc.torch import vqc_basis_embedding,QMachine
        qm  = QMachine(3)
        vqc_basis_embedding(basis_state=[1,1,0],q_machine=qm)
        print(qm.states)




vqc_angle_embedding
""""""""""""""""""""""""""""""""""""""""


.. py:function:: pyvqnet.qnn.vqc.torch.vqc_angle_embedding(input_feat, wires, q_machine: pyvqnet.qnn.vqc.torch.QMachine, rotation: str = "X")

    Encode :math:`N` caractéristiques dans l'angle de rotation de :math:`n` qubits, où :math:`N \leq n`.
    Cette fonction est aliasée sous `VQC_AngleEmbedding`.

    La rotation peut être sélectionnée comme : 'X', 'Y', 'Z', comme défini par le paramètre ``rotation`` :

    * ``rotation='X'`` Utilise la caractéristique comme angle de rotation RX.

    * ``rotation='Y'`` Utilise la caractéristique comme angle de rotation RY.

    * ``rotation='Z'`` Utilise la caractéristique comme angle de rotation RZ.

    ``wires`` représente l'idx de la porte de rotation sur le qubit.

    :param input_feat: Tableau représentant les paramètres.
    :param wires: Idx du qubit.
    :param q_machine: Quantum machine device.
    :param rotation: Porte de rotation, défaut est "X".

    Example::

        import pyvqnet
        pyvqnet.backends.set_backend("torch")
        from pyvqnet.qnn.vqc.torch import vqc_angle_embedding, QMachine
        from pyvqnet.tensor import QTensor
        qm  = QMachine(2)
        vqc_angle_embedding(QTensor([2.2, 1]), [0, 1], q_machine=qm, rotation='X')
        print(qm.states)
        vqc_angle_embedding(QTensor([2.2, 1]), [0, 1], q_machine=qm, rotation='Y')
        print(qm.states)
        vqc_angle_embedding(QTensor([2.2, 1]), [0, 1], q_machine=qm, rotation='Z')
        print(qm.states)



vqc_amplitude_embedding
""""""""""""""""""""""""""""""""""""""""

.. py:function:: pyvqnet.qnn.vqc.torch.vqc_amplitude_embeddingVQC_AmplitudeEmbeddingCircuit(input_feature, q_machine)

    Encode une caractéristique de :math:`2^n` dans un vecteur d'amplitude de :math:`n` qubits. Cette fonction est aliasée sous `VQC_AmplitudeEmbedding`.

    :param input_feature: tableau numpy représentant le paramètre.
    :param q_machine: quantum machine device.
    

    Example::

        import pyvqnet
        pyvqnet.backends.set_backend("torch")
        from pyvqnet.qnn.vqc.torch import vqc_amplitude_embedding, QMachine
        from pyvqnet.tensor import QTensor
        qm  = QMachine(3)
        vqc_amplitude_embedding(QTensor([3.2,-2,-2,0.3,12,0.1,2,-1]), q_machine=qm)
        print(qm.states)



vqc_iqp_embedding
""""""""""""""""""""""""""""""""""""""""

.. py:function:: pyvqnet.qnn.vqc.vqc_iqp_embedding(input_feat, q_machine: pyvqnet.qnn.vqc.torch.QMachine, rep: int = 1)

    Encode :math:`n` caractéristiques dans :math:`n` qubits en utilisant des portes diagonales d'un circuit IQP. Alias : ``VQC_IQPEmbedding``.

    L'encodage est proposé par `Havlicek et al. (2018) <https://arxiv.org/pdf/1804.11326.pdf>`_.

    En spécifiant ``rep``, le circuit IQP de base peut être répété.

    :param input_feat: Tableau de paramètres.
    :param q_machine: Machine quantique.
    :param rep: Nombre de répétitions du bloc de circuit quantique, défaut est 1.

    Example::

        import pyvqnet
        pyvqnet.backends.set_backend("torch")
        from pyvqnet.qnn.vqc.torch import vqc_iqp_embedding, QMachine
        from pyvqnet.tensor import QTensor
        qm  = QMachine(3)
        vqc_iqp_embedding(QTensor([3.2,-2,-2]), q_machine=qm)
        print(qm.states)        



vqc_rotcircuit
""""""""""""""""""""""""""""""""""""""""

.. py:function:: pyvqnet.qnn.vqc.torch.vqc_rotcircuit(q_machine, wire, params)

    Combinaison de portes logiques quantiques de rotation à un seul bit quantique arbitraire. Cette fonction alias : ``VQC_RotCircuit``.

    .. math::
        R(\phi,\theta,\omega) = RZ(\omega)RY(\theta)RZ(\phi)= \begin{bmatrix}
        e^{-i(\phi+\omega)/2}\cos(\theta/2) & -e^{i(\phi-\omega)/2}\sin(\theta/2) \\
        e^{-i(\phi-\omega)/2}\sin(\theta/2) & e^{i(\phi+\omega)/2}\cos(\theta/2)
        \end{bmatrix}.

    :param q_machine: périphérique de machine virtuelle quantique.
    :param wire: indice de bit quantique.
    :param params: représente les paramètres :math:`[\phi, \theta, \omega]`.

    Example::

        import pyvqnet
        pyvqnet.backends.set_backend("torch")
        from pyvqnet.qnn.vqc.torch import vqc_rotcircuit, QMachine
        from pyvqnet.tensor import QTensor
        qm  = QMachine(3)
        vqc_rotcircuit(q_machine=qm, wire=[1],params=QTensor([2.0,1.5,2.1]))
        print(qm.states)


vqc_crot_circuit
""""""""""""""""""""""""""""""""""""""""


.. py:function:: pyvqnet.qnn.vqc.torch.vqc_crot_circuit(para,control_qubits,rot_wire,q_machine)

    Combinaison de portes logiques quantiques de rotation Rot contrôlée à un seul bit quantique. Cette fonction alias : ``VQC_CRotCircuit``.

    .. math:: 
        CR(\phi, \theta, \omega) = \begin{bmatrix}
        1 & 0 & 0 & 0 \\
        0 & 1 & 0 & 0\\
        0 & 0 & e^{-i(\phi+\omega)/2}\cos(\theta/2) & -e^{i(\phi-\omega)/2}\sin(\theta/2)\\
        0 & 0 & e^{-i(\phi-\omega)/2}\sin(\theta/2) & e^{i(\phi+\omega)/2}\cos(\theta/2)
        \end{bmatrix}.

    :param para: représente le tableau de paramètres.
    :param control_qubits: Indice du qubit de contrôle.
    :param rot_wire: Indice du qubit de rotation.
    :param q_machine: Quantum machine device.
    

    Example::

        import pyvqnet
        pyvqnet.backends.set_backend("torch")
        from pyvqnet.tensor import QTensor
        from pyvqnet.qnn.vqc.torch import vqc_crot_circuit,QMachine, MeasureAll
        p = QTensor([2, 3, 4.0])
        qm = QMachine(2)
        vqc_crot_circuit(p, 0, 1, qm)
        m = MeasureAll(obs={"Z0": 1})
        exp = m(q_machine=qm)
        print(exp)




vqc_controlled_hadamard
""""""""""""""""""""""""""""""""""""""""


.. py:function:: pyvqnet.qnn.vqc.torch.vqc_controlled_hadamard(wires, q_machine)

    Circuit quantique de porte logique Hadamard contrôlée. Cette fonction alias : ``VQC_Controlled_Hadamard``.

    .. math:: 
        CH = \begin{bmatrix}
        1 & 0 & 0 & 0 \\
        0 & 1 & 0 & 0 \\
        0 & 0 & \frac{1}{\sqrt{2}} & \frac{1}{\sqrt{2}} \\
        0 & 0 & \frac{1}{\sqrt{2}} & -\frac{1}{\sqrt{2}}
        \end{bmatrix}.

    :param wires: liste d'indices de bits quantiques, le premier est le bit de contrôle, la longueur de la liste est 2.
    :param q_machine: quantum virtual machine device.

    Examples::

        import pyvqnet
        pyvqnet.backends.set_backend("torch")
        from pyvqnet.tensor import QTensor
        from pyvqnet.qnn.vqc.torch import vqc_controlled_hadamard,\
            QMachine, MeasureAll

        p = QTensor([0.2, 3, 4.0])
        qm = QMachine(3)
        vqc_controlled_hadamard([1, 0], qm)
        m = MeasureAll(obs={"Z0": 1})
        exp = m(q_machine=qm)
        print(exp)



vqc_ccz
""""""""""""""""""""""""""""""""""""""""

.. py:function:: pyvqnet.qnn.vqc.torch.vqc_ccz(wires, q_machine)

    Porte logique Controlled-controlled-Z. Alias : ``VQC_CCZ``.

    .. math::
        CCZ =
        \begin{pmatrix}
        1 & 0 & 0 & 0 & 0 & 0 & 0 & 0\\
        0 & 1 & 0 & 0 & 0 & 0 & 0 & 0 & 0\\
        0 & 0 & 1 & 0 & 0 & 0 & 0 & 0\\
        0 & 0 & 0 & 1 & 0 & 0 & 0 & 0\\
        0 & 0 & 0 & 0 & 1 & 0 & 0 & 0\\
        0 & 0 & 0 & 0 & 0 & 1 & 0 & 0 & 0\\
        0 & 0 & 0 & 0 & 0 & 1 & 0 & 0\\
        0 & 0 & 0 & 0 & 0 & 0 & 1 & 0 & 0\\
        0 & 0 & 0 & 0 & 0 & 0 & 0 & 1 & 0\\
        0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & -1
        \end{pmatrix}

    :param wires: liste d'indices de bits quantiques, le premier est le bit de contrôle. La longueur de la liste est 3.
    :param q_machine: quantum virtual machine device.

    Example::

        import pyvqnet
        pyvqnet.backends.set_backend("torch")
        from pyvqnet.tensor import QTensor
        from pyvqnet.qnn.vqc.torch import vqc_ccz,QMachine, MeasureAll
        p = QTensor([0.2, 3, 4.0])

        qm = QMachine(3)

        vqc_ccz([1, 0, 2], qm)
        m = MeasureAll(obs={"Z0": 1})
        exp = m(q_machine=qm)
        print(exp)



vqc_fermionic_single_excitation
""""""""""""""""""""""""""""""""""""""""

.. py:function:: pyvqnet.qnn.vqc.torch.vqc_fermionic_single_excitation(weight, wires, q_machine)

    Opérateur d'excitation simple de cluster couplé pour le produit tensoriel de matrices de Pauli. La forme matricielle est donnée par :

    .. math::
        \hat{U}_{pr}(\theta) = \mathrm{exp} \{ \theta_{pr} (\hat{c}_p^\dagger \hat{c}_r
        -\mathrm{H.c.}) \},

    Alias: ``VQC_FermionicSingleExcitation`` .

    :param weight: Paramètre sur le qubit p, seulement a éléments.
    :param wires: Un sous-ensemble d'indices de qubits dans l'intervalle [r, p]. La longueur minimale doit être 2. La première valeur d'indice est interprétée comme r, et la dernière valeur d'indice est interprétée comme p. Les indices intermédiaires sont soumis à des portes CNOT pour calculer la parité de l'ensemble de qubits.
    :param q_machine: Périphérique de machine virtuelle quantique.

    

    Examples::

        import pyvqnet
        pyvqnet.backends.set_backend("torch")
        from pyvqnet.tensor import QTensor
        from pyvqnet.qnn.vqc.torch import vqc_fermionic_single_excitation,\
            QMachine, MeasureAll
        qm = QMachine(3)
        p0 = QTensor([0.5])

        vqc_fermionic_single_excitation(p0, [1, 0, 2], qm)
        m = MeasureAll(obs={"Z0": 1})
        exp = m(q_machine=qm)
        print(exp)

 


vqc_fermionic_double_excitation
""""""""""""""""""""""""""""""""""""""""


.. py:function:: pyvqnet.qnn.vqc.torch.vqc_fermionic_double_excitation(weight, wires1, wires2, q_machine)

    Opérateur de biexcitation de cluster couplé pour le produit tensoriel de matrices de Pauli exponentiel, la forme matricielle est donnée par :

    .. math::
        \hat{U}_{pqrs}(\theta) = \mathrm{exp} \{ \theta (\hat{c}_p^\dagger \hat{c}_q^\dagger
        \hat{c}_r \hat{c}_s - \mathrm{H.c.}) \},

    where :math:`\hat{c}` and :math:`\hat{c}^\dagger` are fermion annihilation and
    operators are created and indexed :math:`r, s` and :math:`p, q` on occupied and
    empty molecular orbitals respectively. Use `Jordan-Wigner transformation
    <https://arxiv.org/abs/1208.5986>`_ The fermion operator defined above can be written as
    in terms of the Pauli matrix (see
    `arXiv:1805.04340 <https://arxiv.org/abs/1805.04340>`_ for more details)

    .. math::
        \hat{U}_{pqrs}(\theta) = \mathrm{exp} \Big\{
        \frac{i\theta}{8} \bigotimes_{b=s+1}^{r-1} \hat{Z}_b \bigotimes_{a=q+1}^{p-1}
        \hat{Z}_a (\hat{X}_s \hat{X}_r \hat{Y}_q \hat{X}_p +
        \hat{Y}_s \hat{X}_r \hat{Y}_q \hat{Y}_p +\\ \hat{X}_s \hat{Y}_r \hat{Y}_q \hat{Y}_p +
        \hat{X}_s \hat{X}_r \hat{X}_q \hat{Y}_p - \mathrm{H.c.} ) \Big\}

    This function is aliased as: ``VQC_FermionicDoubleExcitation`` .

    :param weight: paramètre variable
    :param wires1: représente le sous-ensemble de qubits dans l'intervalle de liste d'indices [s, r]. Le a-ème indice est interprété comme s et le dernier indice est interprété comme r. La porte CNOT opère sur les index du milieu pour calculer la parité d'un groupe de qubits.
    :param wires2: représente le sous-ensemble de qubits dans l'intervalle de liste d'indices [q, p]. Le premier indice racine est interprété comme q et le dernier indice est interprété comme p. La porte CNOT opère sur les index du milieu pour calculer la parité d'un groupe de qubits.
    :param q_machine: Quantum virtual machine device.

    

    Examples::

        import pyvqnet
        pyvqnet.backends.set_backend("torch")
        from pyvqnet.tensor import QTensor
        from pyvqnet.qnn.vqc.torch import vqc_fermionic_double_excitation,\
            QMachine, MeasureAll
        qm = QMachine(5)
        p0 = QTensor([0.5])

        vqc_fermionic_double_excitation(p0, [0, 1], [2, 3], qm)
        m = MeasureAll(obs={"Z0": 1})
        exp = m(q_machine=qm)
        print(exp)
 

vqc_uccsd
""""""""""""""""""""""""""""""""""""""""


.. py:function:: pyvqnet.qnn.vqc.torch.vqc_uccsd(weights, wires, s_wires, d_wires, init_state, q_machine)

    Implémente la simulation UCCSD (Unitary Coupled Cluster Single and Double Excitations). UCCSD est une simulation VQE couramment utilisée pour exécuter des simulations de chimie quantique.

    Dans l'approximation de Trotter de premier ordre, la fonction unitaire UCCSD est donnée par :

    .. math::
        \hat{U}(\vec{\theta}) =
        \prod_{p > r} \mathrm{exp} \Big\{\theta_{pr}
        (\hat{c}_p^\dagger \hat{c}_r-\mathrm{H.c.}) \Big\}
        \prod_{p > q > r > s} \mathrm{exp} \Big\{\theta_{pqrs}
        (\hat{c}_p^\dagger \hat{c}_q^\dagger \hat{c}_r \hat{c}_s-\mathrm{H.c.}) \Big\}

    where :math:`\hat{c}` and :math:`\hat{c}^\dagger` are fermion annihilation and
    creation operators and index :math:`r, s` and :math:`p, q` on occupied and
    empty molecular orbitals respectively. (For more details see
    `arXiv:1805.04340 <https://arxiv.org/abs/1805.04340>`_):

    This function is aliased as: ``VQC_UCCSD`` .

    :param weights: tenseur de taille ``(len(s_wires)+ len(d_wires))`` contenant les paramètres :math:`\theta_{pr}` et :math:`\theta_{pqrs}` rotations Z d'entrée ``FermionicSingleExcitation`` et ``FermionicDoubleExcitation``.
    :param wires: indices de qubits pour l'action du gabarit
    :param s_wires: séquence de listes contenant les indices de qubits ``[r,...,p]`` générés par une excitation simple :math:`\vert r, p \rangle = \hat{c}_p^\dagger \hat{c}_r \vert \mathrm{HF} \rangle`, où :math:`\vert \mathrm{HF} \rangle` désigne l'état de référence Hartree-Fock.
    :param d_wires: séquence de listes, chacune contenant deux listes spécifiant les indices ``[s, ...,r]`` et ``[q,..., p]`` définissant la double excitation :math:`\vert s, r, q, p \rangle = \hat{c}_p^\dagger \hat{c}_q^\dagger \hat{c}_r\hat{c}_s \vert \mathrm{HF} \rangle`.
    :param init_state: vecteur de nombre d'occupation de longueur ``len(wires)`` représentant l'état haute-fréquence. ``init_state`` État d'initialisation du qubit.
    :param q_machine: Quantum virtual machine device.

    Examples::

        import pyvqnet
        pyvqnet.backends.set_backend("torch")
        from pyvqnet.qnn.vqc.torch import vqc_uccsd, QMachine, MeasureAll
        from pyvqnet.tensor import QTensor
        p0 = QTensor([2, 0.5, -0.2, 0.3, -2, 1, 3, 0])
        s_wires = [[0, 1, 2], [0, 1, 2, 3, 4], [1, 2, 3], [1, 2, 3, 4, 5]]
        d_wires = [[[0, 1], [2, 3]], [[0, 1], [2, 3, 4, 5]], [[0, 1], [3, 4]],
                [[0, 1], [4, 5]]]
        qm = QMachine(6)

        vqc_uccsd(p0, range(6), s_wires, d_wires, QTensor([1.0, 1, 0, 0, 0, 0]), qm)
        m = MeasureAll(obs={"Z1": 1})
        exp = m(q_machine=qm)
        print(exp)

        # [[0.963802]]


vqc_zfeaturemap
""""""""""""""""""""""""""""""""""""""""

.. py:function:: pyvqnet.qnn.vqc.torch.vqc_zfeaturemap(input_feat, q_machine: pyvqnet.qnn.vqc.torch.QMachine, data_map_func=None, rep: int = 2)

    Circuit d'évolution Pauli Z de premier ordre.

    For 3 qubits and 2 repetitions, the circuit is represented as:

    .. parsed-literal::

        ┌───┐┌──────────────┐┌───┐┌──────────────┐
        ┤ H ├┤ U1(2.0*x[0]) ├┤ H ├┤ U1(2.0*x[0]) ├
        ├───┤├──────────────┤├───┤├──────────────┤
        ┤ H ├┤ U1(2.0*x[1]) ├┤ H ├┤ U1(2.0*x[1]) ├
        ├───┤├──────────────┤├───┤├──────────────┤
        ┤ H ├┤ U1(2.0*x[2]) ├┤ H ├┤ U1(2.0*x[2]) ├
        └───┘└──────────────┘└───┘└──────────────┘

    La chaîne de Pauli est fixée à ``Z``. Par conséquent, l'expansion de premier ordre sera un circuit sans portes d'intrication.

    :param input_feat: Tableau représentant les paramètres d'entrée.
    :param q_machine: Quantum virtual machine.
    :param data_map_func: Matrice de mappage de paramètres, une fonction appelable, conçue comme : ``data_map_func = lambda x: x``.
    :param rep: Nombre de répétitions du module.

    Example::

        import pyvqnet
        pyvqnet.backends.set_backend("torch")
        from pyvqnet.qnn.vqc.torch import vqc_zfeaturemap, QMachine, hadamard
        from pyvqnet.tensor import QTensor
        qm = QMachine(3)
        for i in range(3):
            hadamard(q_machine=qm, wires=[i])
        vqc_zfeaturemap(input_feat=QTensor([[0.1,0.2,0.3]]),q_machine = qm)
        print(qm.states)
 

vqc_zzfeaturemap
""""""""""""""""""""""""""""""""""""""""

.. py:function:: pyvqnet.qnn.vqc.torch.vqc_zzfeaturemap(input_feat, q_machine: pyvqnet.qnn.vqc.torch.QMachine, data_map_func=None, entanglement: Union[str, List[List[int]],Callable[[int], List[int]]] = "full",rep: int = 2)

    Circuit d'évolution Pauli-Z de second ordre.

    For 3 qubits, 1 repeat, and linear entanglement, the circuit is represented as:

    .. parsed-literal::


        ┌───┐┌─────────────────┐
        ┤ H ├┤ U1(2.0*φ(x[0])) ├──■────────────────────────────■────────────────────────────────────
        ├───┤├─────────────────┤┌─┴─┐┌──────────────────────┐┌─┴─┐
        ┤ H ├┤ U1(2.0*φ(x[1])) ├┤ X ├┤ U1(2.0*φ(x[0],x[1])) ├┤ X ├──■────────────────────────────■──
        ├───┤├─────────────────┤└───┘└──────────────────────┘└───┘┌─┴─┐┌──────────────────────┐┌─┴─┐
        ┤ H ├┤ U1(2.0*φ(x[2])) ├──────────────────────────────────┤ X ├┤ U1(2.0*φ(x[1],x[2])) ├┤ X ├
        └───┘└─────────────────┘                                  └───┘└──────────────────────┘└───┘
    
    Where ``φ`` is a classic nonlinear function. If two values ​​are input, ``φ(x,y) = (pi - x)(pi - y)``, and if a is input, ``φ(x) = x``. It is expressed as follows using ``data_map_func``:

    .. code-block::

        def data_map_func(x):
            coeff = x if x.shape[-1] == 1 else ft.reduce(lambda x, y: (np.pi - x) * (np.pi - y), x)
            return coeff

    :param input_feat: Array representing input parameters.
    :param q_machine: Quantum virtual machine.
    :param data_map_func: matrice de mappage de paramètres, une fonction appelable.
    :param entanglement: structure d'intrication spécifiée.
    :param rep: nombre de répétitions du module.
    
    Example::

        import pyvqnet
        pyvqnet.backends.set_backend("torch")
        from pyvqnet.qnn.vqc.torch import vqc_zzfeaturemap, QMachine
        from pyvqnet.tensor import QTensor

        qm = QMachine(3)
        vqc_zzfeaturemap(q_machine=qm, input_feat=QTensor([[0.1,0.2,0.3]]))
        print(qm.states)


vqc_allsinglesdoubles
""""""""""""""""""""""""""""""""""""""""

.. py:function:: pyvqnet.qnn.vqc.torch.vqc_allsinglesdoubles(weights, q_machine: pyvqnet.qnn.vqc.torch.QMachine, hf_state, wires, singles=None, doubles=None)

    Dans ce cas, nous avons quatre excitations simples et doubles pour préserver la projection de spin totale de l'état de Hartree-Fock.

    La matrice unitaire résultante préserve la population de particules et prépare le système à n qubits dans une superposition de l'état initial de Hartree-Fock et d'autres états encodant la configuration multi-excitation.

    :param weights: Un QTensor de taille ``(len(singles) + len(doubles),)`` contenant les angles qui entrent dans les opérations vqc.qCircuit.single_excitation et vqc.qCircuit.double_excitation en séquence
    :param q_machine: The quantum machine.
    :param hf_state: Un vecteur de nombres d'occupation de longueur ``len(wires)`` représentant l'état de Hartree-Fock, ``hf_state`` utilisé pour initialiser les fils.
    :param wires: Les qubits sur lesquels agir.
    :param singles: Une séquence de listes avec les indices des deux qubits sur lesquels agit l'opération single_exitation.
    :param doubles: Séquence de listes avec les indices des deux qubits sur lesquels agit l'opération double_exitation.

    For example, the quantum circuit for two electrons and six qubits is shown below:

    .. image:: ./images/all_singles_doubles.png
        :width: 600 px
        :align: center

    |

    Example::

        import pyvqnet
        pyvqnet.backends.set_backend("torch")
        from pyvqnet.qnn.vqc.torch import vqc_allsinglesdoubles, QMachine

        from pyvqnet.tensor import QTensor
        qubits = 4
        qm = QMachine(qubits)

        vqc_allsinglesdoubles(q_machine=qm, weights=QTensor([0.55, 0.11, 0.53]), 
                              hf_state = QTensor([1,1,0,0]), singles=[[0, 2], [1, 3]], doubles=[[0, 1, 2, 3]], wires=[0,1,2,3])
        print(qm.states)

vqc_basisrotation
""""""""""""""""""""""""""""""""""""""""

.. py:function:: pyvqnet.qnn.vqc.torch.vqc_basisrotation(q_machine: pyvqnet.qnn.vqc.torch.QMachine, wires, unitary_matrix: QTensor, check=False)

    Implémente un circuit qui fournit un ensemble pouvant être utilisé pour effectuer des rotations de base à unité unique précises. Le circuit est dérivé de la transformation unitaire déterminée par un fermion à une particule :math:`U(u)` donnée dans `arXiv:1711.04789 <https://arxiv.org/abs/1711.04789>`_\ 
    
    .. math::
        U(u) = \exp{\left( \sum_{pq} \left[\log u \right]_{pq} (a_p^\dagger a_q - a_q^\dagger a_p) \right)}.

    :math:`U(u)` is obtained by using the scheme given in the paper `Optica, 3, 1460 (2016) <https://opg.optica.org/optica/fulltext.cfm?uri=optica-3-12-1460&id=355743>`_\ .

    :param q_machine: quantum machine.
    :param wires: qubits to act on.
    :param unitary_matrix: matrice spécifiant la base pour la transformation.
    :param check: vérifie si `unitary_matrix` est une matrice unitaire.

    Example::

        import pyvqnet

        pyvqnet.backends.set_backend("torch")
        from pyvqnet.qnn.vqc.torch import vqc_basisrotation, QMachine
        from pyvqnet.tensor import QTensor
        import numpy as np

        V = np.array([[0.73678 + 0.27511j, -0.5095 + 0.10704j, -0.06847 + 0.32515j],
                    [0.73678 + 0.27511j, -0.5095 + 0.10704j, -0.06847 + 0.32515j],
                    [-0.21271 + 0.34938j, -0.38853 + 0.36497j, 0.61467 - 0.41317j]])

        eigen_vals, eigen_vecs = np.linalg.eigh(V)
        umat = eigen_vecs.T
        wires = range(len(umat))

        qm = QMachine(len(umat))

        vqc_basisrotation(q_machine=qm,
                        wires=wires,
                        unitary_matrix=QTensor(umat, dtype=qm.state.dtype))

        print(qm.states)



vqc_quantumpooling_circuit
""""""""""""""""""""""""""""""""""""""""

.. py:function:: pyvqnet.qnn.vqc.torch.vqc_quantumpooling_circuit(ignored_wires, sinks_wires, params, q_machine)

    Circuit quantique qui sous-échantillonne les données.

    Pour réduire le nombre de qubits dans le circuit, des paires de qubits sont d'abord créées dans le système. Après avoir initialement apparié tous les qubits, une unitaire généralisée à 2 qubits est appliquée à chaque paire de qubits. Après l'application de ces unitaires à deux qubits, un qubit de chaque paire de qubits est ignoré pour le reste du réseau neuronal.

    :param sources_wires: Indices des qubits sources qui seront ignorés.
    :param sinks_wires: Indices des qubits cibles qui seront conservés.
    :param params: Paramètres d'entrée.
    :param q_machine: Quantum virtual machine device.

    Examples:: 

        import pyvqnet

        pyvqnet.backends.set_backend("torch")
        from pyvqnet.qnn.vqc.torch import vqc_quantumpooling_circuit, QMachine, MeasureAll
        from pyvqnet import tensor
        p = tensor.full([6], 0.35)
        qm = QMachine(4)
        vqc_quantumpooling_circuit(q_machine=qm,
                                ignored_wires=[0, 1],
                                sinks_wires=[2, 3],
                                params=p)
        m = MeasureAll(obs={"Z1": 1})
        exp = m(q_machine=qm)
        print(exp)



QuantumLayerAdjoint
""""""""""""""""""""""""""""""""""""""""

.. py:class:: pyvqnet.qnn.vqc.torch.QuantumLayerAdjoint(general_module: pyvqnet.nn.Module, use_qpanda=False, name="")


    Une couche QuantumLayer automatiquement différentiable qui utilise l'approche de matrice adjointe pour calculer les gradients, voir `Efficient calculation of gradients in classical simulations of variational quantum algorithms <https://arxiv.org/abs/2009.02823>`_.

    :param general_module: une instance `pyvqnet.nn.Module` construite en utilisant uniquement l'interface de circuit quantique sous ``pyvqnet.qnn.vqc.torch``.
    :param use_qpanda: Utiliser ou non la ligne qpanda pour la transmission forward, défaut : False.
    :param name: Le nom de la couche, par défaut "".

    .. note::

        Le QMachine de general_module doit définir grad_method = "adjoint".

        Supporte actuellement les portes logiques paramétrées suivantes `RX`, `RY`, `RZ`, `PhaseShift`, `RXX`, `RYY`, `RZZ`, `RZX`, `U1`, `U2`, `U3` et autres circuits variationnels constitués de portes logiques non paramétrées.


    Example::

        import pyvqnet
        pyvqnet.backends.set_backend("torch")
        from pyvqnet import tensor
        from pyvqnet.qnn.vqc.torch import QuantumLayerAdjoint, \
            QMachine, RX, RY, CNOT, T, \
                MeasureAll, RZ, VQC_HardwareEfficientAnsatz,\
                    QModule

        class QModel(QModule):
            def __init__(self, num_wires, dtype, grad_mode=""):
                super(QModel, self).__init__()

                self._num_wires = num_wires
                self._dtype = dtype
                self.qm = QMachine(num_wires, dtype=dtype, grad_mode=grad_mode)
                self.rx_layer = RX(has_params=True, trainable=False, wires=0)
                self.ry_layer = RY(has_params=True, trainable=False, wires=1)
                self.rz_layer = RZ(has_params=True, trainable=False, wires=1)
                self.rz_layer2 = RZ(has_params=True, trainable=True, wires=1)

                self.rot = VQC_HardwareEfficientAnsatz(6, ["rx", "RY", "rz"],
                                                    entangle_gate="cnot",
                                                    entangle_rules="linear",
                                                    depth=5)
                self.tlayer = T(wires=1)
                self.cnot = CNOT(wires=[0, 1])
                self.measure = MeasureAll(obs={
                    "X1":1
                })

            def forward(self, x, *args, **kwargs):
                self.qm.reset_states(x.shape[0])

                self.rx_layer(params=x[:, [0]], q_machine=self.qm)
                self.cnot(q_machine=self.qm)
                self.ry_layer(params=x[:, [1]], q_machine=self.qm)
                self.tlayer(q_machine=self.qm)
                self.rz_layer(params=x[:, [2]], q_machine=self.qm)
                self.rz_layer2(q_machine=self.qm)
                self.rot(q_machine=self.qm)
                rlt = self.measure(q_machine=self.qm)

                return rlt


        input_x = tensor.QTensor([[0.1, 0.2, 0.3]])
        input_x = tensor.broadcast_to(input_x, [40, 3])
        input_x.requires_grad = True
        qunatum_model = QModel(num_wires=6,
                            dtype=pyvqnet.kcomplex64,
                            grad_mode="adjoint")
        adjoint_model = QuantumLayerAdjoint(qunatum_model, qunatum_model.qm)
        batch_y = adjoint_model(input_x)
        batch_y.backward()





Module de circuit quantique variationnel backend réseau de tenseurs
==========================================================================================

Le réseau de tenseurs (TN) réduit considérablement la complexité de calcul en décomposant un tenseur complexe en un réseau de multiples tenseurs de faible dimension.

L'état de produit matriciel (MPS) est une forme spéciale de réseau de tenseurs. MPS représente un état quantique comme le produit d'une série de matrices, réduisant ainsi efficacement le nombre de paramètres et la complexité de calcul.

L'interface suivante est basée sur le backend ``torch``, qui fournit un support fonctionnel pour la construction de circuits quantiques dans les réseaux de tenseurs, y compris la construction de classes de base de circuits quantiques, de portes logiques quantiques, de circuits quantiques et de mesures, ainsi que le calcul des gradients de paramètres par simulation différentielle automatique au lieu de la méthode de dérive des paramètres.

La construction de lignes quantiques de manière MPS compense le support pour la construction de lignes quantiques à grands bits.

.. warning::

        L'utilisation des fonctionnalités suivantes dans ce module nécessite l'installation supplémentaire de ``tensornetwork`` et ``torch``. L'installation par défaut de ``pyvqnet`` n'inclut pas ces deux dépendances. Veuillez les installer en utilisant ``pip install tensornetwork torch``.

.. warning::

        Permet à MPS de construire des lignes quantiques via le paramètre ``use_mps`` dans ``TNQMachine``, qui supporte les implémentations de lignes quantiques à grands bits (100 et plus).

.. warning::
        
        Le batching est utilisé différemment que sous les modules classiques, basé sur l'approche vmap, où les lignes de construction des données et des paramètres doivent être entrées avec une dimension en moins, comme montré dans l'exemple d'interface ci-dessous, et l'exécution par lots doit être basée à la fois sur ``TNQMachine`` et ``TNQModule``.

Classe de base
------------------------------------------------

L'écriture d'un modèle de circuit quantique variationnel sur tensornetwork nécessite d'hériter de ``TNQModule``.

TNQModule
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. py:class:: pyvqnet.qnn.vqc.tn.TNQModule(use_jit=False, vectorized_argnums=0, name="")

    .. note::

        Cette classe et ses classes dérivées ne sont applicables qu'à ``pyvqnet.backends.set_backend("torch")``, ne pas mélanger avec le ``Module`` sous le ``pyvqnet.nn`` par défaut.

        Les données dans ``_buffers`` de cette classe sont de type ``torch.Tensor``.

        Les données dans ``_parmeters`` de cette classe sont de type ``torch.nn.Parameter``.

    :param use_jit: contrôle la fonctionnalité de compilation jit du circuit quantique.
    :param vectorized_argnums: les arguments à vectoriser,
            ces arguments doivent partager la même forme de lot dans la première dimension, par défaut 0.
    :param name: nom du Module.

    Example::

        import pyvqnet
        from pyvqnet.nn import Parameter
        pyvqnet.backends.set_backend("torch")
        from pyvqnet.qnn.vqc.tn import TNQModule
        from pyvqnet.qnn.vqc.tn import TNQMachine, RX, RY, CNOT, PauliX, PauliZ,qmeasure,qcircuit,VQC_RotCircuit
        class QModel(TNQModule):
            def __init__(self, num_wires, dtype,batch_size=2):
                super(QModel, self).__init__()

                self._num_wires = num_wires
                self._dtype = dtype
                self.qm = TNQMachine(num_wires, dtype=dtype)

                self.w = Parameter((2,4,3),initializer=pyvqnet.utils.initializer.quantum_uniform)
                self.cnot = CNOT(wires=[0, 1])
                self.batch_size = batch_size
            def forward(self, x, *args, **kwargs):
                self.qm.reset_states(batchsize=self.batch_size)

                def get_cnot(nqubits,qm):
                    for i in range(len(nqubits) - 1):
                        CNOT(wires = [nqubits[i], nqubits[i + 1]])(q_machine = qm)
                    CNOT(wires = [nqubits[len(nqubits) - 1], nqubits[0]])(q_machine = qm)


                def build_circuit(weights, xx, nqubits,qm):
                    def Rot(weights_j, nqubits,qm):#pylint:disable=invalid-name
                        VQC_RotCircuit(qm,nqubits,weights_j)

                    def basisstate(qm,xx, nqubits):
                        for i in nqubits:
                            qcircuit.rz(q_machine=qm, wires=i, params=xx[i])
                            qcircuit.ry(q_machine=qm, wires=i, params=xx[i])
                            qcircuit.rz(q_machine=qm, wires=i, params=xx[i])

                    basisstate(qm,xx,nqubits)

                    for i in range(weights.shape[0]):

                        weights_i = weights[i, :, :]
                        for j in range(len(nqubits)):
                            weights_j = weights_i[j]
                            Rot(weights_j, nqubits[j],qm)
                        get_cnot(nqubits,qm)

                build_circuit(self.w, x,range(4),self.qm)

                y= qmeasure.MeasureAll(obs={'Z0': 1})(self.qm)
                return y


        x= pyvqnet.tensor.QTensor([[1,0,0,1],[1,1,0,1]],dtype=pyvqnet.kfloat32)
        model = QModel(4,pyvqnet.kcomplex64,2)
        y = model(x)
        y.backward()


TNQMachine
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. py:class:: pyvqnet.qnn.vqc.tn.TNQMachine(num_wires, dtype=pyvqnet.kcomplex64, use_mps=False)

    Simulator class for variational quantum computing, including statevectors whose states attribute is quantum circuits.

    This class inherits from ``pyvqnet.nn.torch.TorchModule`` and ``pyvqnet.qnn.QMachine``.

    Cette classe peut être ajoutée au modèle torch comme sous-module de ``torch.nn.Module``.

    .. warning::
        
        Dans le circuit quantique du réseau de tenseurs, la fonction ``vmap`` sera activée par défaut, et la dimension du lot sera supprimée dans les paramètres de la porte logique sur la ligne.
        Lors de l'utilisation du paramètre d'appel, si la dimension est [batch_size, \*], la première dimension batch_size est supprimée, et les dimensions suivantes sont utilisées directement, par exemple, pour les données d'entrée x[:,1] -> x[1], et pour le paramètre entraînable également, voir l'exemple suivant pour l'utilisation de xx, weights.

    .. note::

        Avant chaque exécution du circuit quantique complet, vous devez utiliser `pyvqnet.qnn.vqc.QMachine.reset_states(batchsize)` pour réinitialiser l'état initial dans le simulateur et le diffuser aux dimensions (batchsize,*) pour s'adapter à l'entraînement par lots.

    :param num_wires: nombre de qubits à utiliser
    :param dtype: type de données interne utilisé pour le calcul.
    :param use_mps: ouvre MPSCircuit pour les modèles à grands bits.

    :return: Retourne un objet TNQMachine.

    Example::
        
        import pyvqnet
        from pyvqnet.nn import Parameter
        pyvqnet.backends.set_backend("torch")
        from pyvqnet.qnn.vqc.tn import TNQModule
        from pyvqnet.qnn.vqc.tn import TNQMachine, RX, RY, CNOT, PauliX, PauliZ,qmeasure,qcircuit,VQC_RotCircuit
        class QModel(TNQModule):
            def __init__(self, num_wires, dtype,batch_size=2):
                super(QModel, self).__init__()

                self._num_wires = num_wires
                self._dtype = dtype
                self.qm = TNQMachine(num_wires, dtype=dtype)

                self.w = Parameter((2,4,3),initializer=pyvqnet.utils.initializer.quantum_uniform)
                self.cnot = CNOT(wires=[0, 1])
                self.batch_size = batch_size
            def forward(self, x, *args, **kwargs):
                self.qm.reset_states(batchsize=self.batch_size)

                def get_cnot(nqubits,qm):
                    for i in range(len(nqubits) - 1):
                        CNOT(wires = [nqubits[i], nqubits[i + 1]])(q_machine = qm)
                    CNOT(wires = [nqubits[len(nqubits) - 1], nqubits[0]])(q_machine = qm)


                def build_circuit(weights, xx, nqubits,qm):
                    def Rot(weights_j, nqubits,qm):#pylint:disable=invalid-name
                        VQC_RotCircuit(qm,nqubits,weights_j)

                    def basisstate(qm,xx, nqubits):
                        for i in nqubits:
                            qcircuit.rz(q_machine=qm, wires=i, params=xx[i])
                            qcircuit.ry(q_machine=qm, wires=i, params=xx[i])
                            qcircuit.rz(q_machine=qm, wires=i, params=xx[i])

                    basisstate(qm,xx,nqubits)

                    for i in range(weights.shape[0]):

                        weights_i = weights[i, :, :]
                        for j in range(len(nqubits)):
                            weights_j = weights_i[j]
                            Rot(weights_j, nqubits[j],qm)
                        get_cnot(nqubits,qm)

                build_circuit(self.w, x,range(4),self.qm)

                y= qmeasure.MeasureAll(obs={'Z0': 1})(self.qm)
                return y


        x= pyvqnet.tensor.QTensor([[1,0,0,1],[1,1,0,1]],dtype=pyvqnet.kfloat32)
        model = QModel(4,pyvqnet.kcomplex64,2)
        y = model(x)
        y.backward()

    .. py:method:: get_states()

        obtient les états qmachine du réseau de tenseurs.

Variational quantum logic gate module
------------------------------------------------

Les interfaces de fonctions suivantes dans ``pyvqnet.qnn.vqc`` supportent directement le ``QTensor`` du backend ``torch`` pour le calcul, chemin d'import ``pyvqnet.qnn.vqc.tn``.

.. csv-table:: List of supported pyvqnet.qnn.vqc interfaces
    :file: ./images/same_apis_from_tn.csv

Les modules de circuit quantique suivants héritent de ``pyvqnet.qnn.vqc.tn.TNQModule``, où les calculs sont effectués en utilisant ``torch.Tensor``.

.. note::

    This class and its derived classes are only applicable to ``pyvqnet.backends.set_backend("torch")``, do not mix with ``Module`` under the default ``pyvqnet.nn``.

    If these classes have non-parameter member variables ``_buffers``, the data in them is of type ``torch.Tensor``.
    If these classes have parameter member variables ``_parmeters``, the data in them is of type ``torch.nn.Parameter``.

I
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. py:class:: pyvqnet.qnn.vqc.tn.I(has_params: bool = False,trainable: bool = False,init_params=None,wires=None,dtype=pyvqnet.kcomplex64,use_dagger=False)
    
    définit une porte quantique I.

    Cette classe hérite de ``pyvqnet.qnn.vqc.tn.QModule`` et ``torch.nn.Module``.
    Cette classe peut être ajoutée au modèle torch comme sous-module de ``torch.nn.Module``.

    :param has_params: indique s'il a des paramètres, comme les portes RX, RY, etc. qui doivent être définies sur True, et celles sans paramètres doivent être définies sur False, la valeur par défaut est False.
    :param trainable: indique s'il a des paramètres à entraîner. Si la couche utilise des données d'entrée externes pour construire la matrice de la porte logique, définir sur False. Si les paramètres à entraîner doivent être initialisés à partir de cette couche, c'est True, la valeur par défaut est False.
    :param init_params: Paramètres d'initialisation utilisés pour encoder les données classiques QTensor, la valeur par défaut est None.
    :param wires: Index du bit de l'effet de ligne, la valeur par défaut est None.
    :param dtype: La précision des données de la matrice interne de la porte logique peut être définie sur pyvqnet.kcomplex64 ou pyvqnet.kcomplex128, correspondant respectivement à une entrée float ou double.
    :param use_dagger: indique s'il faut utiliser la version conjuguée transposée de la porte, la valeur par défaut est False.
    :return: une instance ``pyvqnet.qnn.vqc.tn.QModule``

    Example::
        
        from pyvqnet.qnn.vqc.tn import I,TNQMachine,TNQModule,MeasureAll, rx
        import pyvqnet
        pyvqnet.backends.set_backend("torch")

        class QModel(TNQModule):
            
            def __init__(self, num_wires, dtype,batch_size=2):
                super(QModel, self).__init__()
                self.device = TNQMachine(num_wires)
                self.layer = I(wires=0)
                self.batch_size = batch_size
                self.num_wires = num_wires
                
            def forward(self, x, *args, **kwargs):
                self.device.reset_states(batchsize=self.batch_size)
                for i in range(self.num_wires):
                    rx(self.device, wires=i, params=x[i])
                self.layer(q_machine = self.device)
                y = MeasureAll(obs={'Z0': 1})(self.device)
                return y

        x = pyvqnet.tensor.QTensor([[1,0,0,1],[1,1,0,1]],dtype=pyvqnet.kfloat32,requires_grad=True)
        model = QModel(4,pyvqnet.kcomplex64,2)
        y = model(x)
        print(y)



Hadamard
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. py:class:: pyvqnet.qnn.vqc.tn.Hadamard(has_params: bool = False,trainable: bool = False,init_params=None,wires=None,dtype=pyvqnet.kcomplex64,use_dagger=False)
    
    définit une porte quantique Hadamard.

    Cette classe hérite de ``pyvqnet.qnn.vqc.tn.QModule`` et ``torch.nn.Module``.
    Cette classe peut être ajoutée au modèle torch comme sous-module de ``torch.nn.Module``.

    :param has_params: indique s'il a des paramètres, comme les portes RX, RY, etc. qui doivent être définies sur True, et celles sans paramètres doivent être définies sur False, la valeur par défaut est False.
    :param trainable: indique s'il a des paramètres à entraîner. Si la couche utilise des données d'entrée externes pour construire la matrice de la porte logique, définir sur False. Si les paramètres à entraîner doivent être initialisés à partir de cette couche, c'est True, la valeur par défaut est False.
    :param init_params: Paramètres d'initialisation utilisés pour encoder les données classiques QTensor, la valeur par défaut est None.
    :param wires: Index du bit de l'effet de ligne, la valeur par défaut est None.
    :param dtype: La précision des données de la matrice interne de la porte logique peut être définie sur pyvqnet.kcomplex64 ou pyvqnet.kcomplex128, correspondant respectivement à une entrée float ou double.
    :param use_dagger: indique s'il faut utiliser la version conjuguée transposée de la porte, la valeur par défaut est False.
    :return: une instance ``pyvqnet.qnn.vqc.tn.QModule``

    Example::
        
        from pyvqnet.qnn.vqc.tn import Hadamard,TNQMachine,TNQModule,MeasureAll, rx
        import pyvqnet
        pyvqnet.backends.set_backend("torch")

        class QModel(TNQModule):
            
            def __init__(self, num_wires, dtype,batch_size=2):
                super(QModel, self).__init__()
                self.device = TNQMachine(num_wires)
                self.layer = Hadamard(wires=0)
                self.batch_size = batch_size
                self.num_wires = num_wires
                
            def forward(self, x, *args, **kwargs):
                self.device.reset_states(batchsize=self.batch_size)
                for i in range(self.num_wires):
                    rx(self.device, wires=i, params=x[i])
                self.layer(q_machine = self.device)
                y = MeasureAll(obs={'Z0': 1})(self.device)
                return y

        x = pyvqnet.tensor.QTensor([[1,0,0,1],[1,1,0,1]],dtype=pyvqnet.kfloat32,requires_grad=True)
        model = QModel(4,pyvqnet.kcomplex64,2)
        y = model(x)
        print(y)


T
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. py:class:: pyvqnet.qnn.vqc.tn.T(has_params: bool = False,trainable: bool = False,init_params=None,wires=None,dtype=pyvqnet.kcomplex64,use_dagger=False)
    
    définit une porte quantique T.

    Cette classe hérite de ``pyvqnet.qnn.vqc.tn.QModule`` et ``torch.nn.Module``.
    Cette classe peut être ajoutée au modèle torch comme sous-module de ``torch.nn.Module``.

    :param has_params: indique s'il a des paramètres, comme les portes RX, RY, etc. qui doivent être définies sur True, et celles sans paramètres doivent être définies sur False, la valeur par défaut est False.
    :param trainable: indique s'il a des paramètres à entraîner. Si la couche utilise des données d'entrée externes pour construire la matrice de la porte logique, définir sur False. Si les paramètres à entraîner doivent être initialisés à partir de cette couche, c'est True, la valeur par défaut est False.
    :param init_params: Paramètres d'initialisation utilisés pour encoder les données classiques QTensor, la valeur par défaut est None.
    :param wires: Index du bit de l'effet de ligne, la valeur par défaut est None.
    :param dtype: La précision des données de la matrice interne de la porte logique peut être définie sur pyvqnet.kcomplex64 ou pyvqnet.kcomplex128, correspondant respectivement à une entrée float ou double.
    :param use_dagger: indique s'il faut utiliser la version conjuguée transposée de la porte, la valeur par défaut est False.
    :return: une instance ``pyvqnet.qnn.vqc.tn.QModule``

    Example::
        
        from pyvqnet.qnn.vqc.tn import T,TNQMachine,TNQModule,MeasureAll, rx
        import pyvqnet
        pyvqnet.backends.set_backend("torch")

        class QModel(TNQModule):
            
            def __init__(self, num_wires, dtype,batch_size=2):
                super(QModel, self).__init__()
                self.device = TNQMachine(num_wires)
                self.layer = T(wires=0)
                self.batch_size = batch_size
                self.num_wires = num_wires
                
            def forward(self, x, *args, **kwargs):
                self.device.reset_states(batchsize=self.batch_size)
                for i in range(self.num_wires):
                    rx(self.device, wires=i, params=x[i])
                self.layer(q_machine = self.device)
                y = MeasureAll(obs={'Z0': 1})(self.device)
                return y

        x = pyvqnet.tensor.QTensor([[1,0,0,1],[1,1,0,1]],dtype=pyvqnet.kfloat32,requires_grad=True)
        model = QModel(4,pyvqnet.kcomplex64,2)
        y = model(x)
        print(y)



S
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. py:class:: pyvqnet.qnn.vqc.tn.S(has_params: bool = False,trainable: bool = False,init_params=None,wires=None,dtype=pyvqnet.kcomplex64,use_dagger=False)
    
    définit une porte quantique S.

    Cette classe hérite de ``pyvqnet.qnn.vqc.tn.QModule`` et ``torch.nn.Module``.
    Cette classe peut être ajoutée au modèle torch comme sous-module de ``torch.nn.Module``.

    :param has_params: indique s'il a des paramètres, comme les portes RX, RY, etc. qui doivent être définies sur True, et celles sans paramètres doivent être définies sur False, la valeur par défaut est False.
    :param trainable: indique s'il a des paramètres à entraîner. Si la couche utilise des données d'entrée externes pour construire la matrice de la porte logique, définir sur False. Si les paramètres à entraîner doivent être initialisés à partir de cette couche, c'est True, la valeur par défaut est False.
    :param init_params: Paramètres d'initialisation utilisés pour encoder les données classiques QTensor, la valeur par défaut est None.
    :param wires: Index du bit de l'effet de ligne, la valeur par défaut est None.
    :param dtype: La précision des données de la matrice interne de la porte logique peut être définie sur pyvqnet.kcomplex64 ou pyvqnet.kcomplex128, correspondant respectivement à une entrée float ou double.
    :param use_dagger: indique s'il faut utiliser la version conjuguée transposée de la porte, la valeur par défaut est False.
    :return: une instance ``pyvqnet.qnn.vqc.tn.QModule``

    Example::
        
        from pyvqnet.qnn.vqc.tn import S,TNQMachine,TNQModule,MeasureAll, rx
        import pyvqnet
        pyvqnet.backends.set_backend("torch")

        class QModel(TNQModule):
            
            def __init__(self, num_wires, dtype,batch_size=2):
                super(QModel, self).__init__()
                self.device = TNQMachine(num_wires)
                self.layer = S(wires=0)
                self.batch_size = batch_size
                self.num_wires = num_wires
                
            def forward(self, x, *args, **kwargs):
                self.device.reset_states(batchsize=self.batch_size)
                for i in range(self.num_wires):
                    rx(self.device, wires=i, params=x[i])
                self.layer(q_machine = self.device)
                y = MeasureAll(obs={'Z0': 1})(self.device)
                return y

        x = pyvqnet.tensor.QTensor([[1,0,0,1],[1,1,0,1]],dtype=pyvqnet.kfloat32,requires_grad=True)
        model = QModel(4,pyvqnet.kcomplex64,2)
        y = model(x)
        print(y)


PauliX
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. py:class:: pyvqnet.qnn.vqc.tn.PauliX(has_params: bool = False,trainable: bool = False,init_params=None,wires=None,dtype=pyvqnet.kcomplex64,use_dagger=False)
    
    définit une porte quantique PauliX.

    Cette classe hérite de ``pyvqnet.qnn.vqc.tn.QModule`` et ``torch.nn.Module``.
    Cette classe peut être ajoutée au modèle torch comme sous-module de ``torch.nn.Module``.


    :param has_params: indique s'il a des paramètres, comme les portes RX, RY, etc. qui doivent être définies sur True, et celles sans paramètres doivent être définies sur False, la valeur par défaut est False.
    :param trainable: indique s'il a des paramètres à entraîner. Si la couche utilise des données d'entrée externes pour construire la matrice de la porte logique, définir sur False. Si les paramètres à entraîner doivent être initialisés à partir de cette couche, c'est True, la valeur par défaut est False.
    :param init_params: Paramètres d'initialisation utilisés pour encoder les données classiques QTensor, la valeur par défaut est None.
    :param wires: Index du bit de l'effet de ligne, la valeur par défaut est None.
    :param dtype: La précision des données de la matrice interne de la porte logique peut être définie sur pyvqnet.kcomplex64 ou pyvqnet.kcomplex128, correspondant respectivement à une entrée float ou double.
    :param use_dagger: indique s'il faut utiliser la version conjuguée transposée de la porte, la valeur par défaut est False.
    :return: une instance ``pyvqnet.qnn.vqc.tn.QModule``

    Example::
        
        from pyvqnet.qnn.vqc.tn import PauliX,TNQMachine,TNQModule,MeasureAll, rx
        import pyvqnet
        pyvqnet.backends.set_backend("torch")

        class QModel(TNQModule):
            
            def __init__(self, num_wires, dtype,batch_size=2):
                super(QModel, self).__init__()
                self.device = TNQMachine(num_wires)
                self.layer = PauliX(wires=0)
                self.batch_size = batch_size
                self.num_wires = num_wires
                
            def forward(self, x, *args, **kwargs):
                self.device.reset_states(batchsize=self.batch_size)
                for i in range(self.num_wires):
                    rx(self.device, wires=i, params=x[i])
                self.layer(q_machine = self.device)
                y = MeasureAll(obs={'Z0': 1})(self.device)
                return y

        x = pyvqnet.tensor.QTensor([[1,0,0,1],[1,1,0,1]],dtype=pyvqnet.kfloat32,requires_grad=True)
        model = QModel(4,pyvqnet.kcomplex64,2)
        y = model(x)
        print(y)



PauliY
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. py:class:: pyvqnet.qnn.vqc.tn.PauliY(has_params: bool = False,trainable: bool = False,init_params=None,wires=None,dtype=pyvqnet.kcomplex64,use_dagger=False)
    
    définit une porte quantique PauliY.

    Cette classe hérite de ``pyvqnet.qnn.vqc.tn.QModule`` et ``torch.nn.Module``.
    Cette classe peut être ajoutée au modèle torch comme sous-module de ``torch.nn.Module``.


    :param has_params: indique s'il a des paramètres, comme les portes RX, RY, etc. qui doivent être définies sur True, et celles sans paramètres doivent être définies sur False, la valeur par défaut est False.
    :param trainable: indique s'il a des paramètres à entraîner. Si la couche utilise des données d'entrée externes pour construire la matrice de la porte logique, définir sur False. Si les paramètres à entraîner doivent être initialisés à partir de cette couche, c'est True, la valeur par défaut est False.
    :param init_params: Paramètres d'initialisation utilisés pour encoder les données classiques QTensor, la valeur par défaut est None.
    :param wires: Index du bit de l'effet de ligne, la valeur par défaut est None.
    :param dtype: La précision des données de la matrice interne de la porte logique peut être définie sur pyvqnet.kcomplex64 ou pyvqnet.kcomplex128, correspondant respectivement à une entrée float ou double.
    :param use_dagger: indique s'il faut utiliser la version conjuguée transposée de la porte, la valeur par défaut est False.
    :return: une instance ``pyvqnet.qnn.vqc.tn.QModule``

    Example::
        
        from pyvqnet.qnn.vqc.tn import PauliY,TNQMachine,TNQModule,MeasureAll, rx
        import pyvqnet
        pyvqnet.backends.set_backend("torch")

        class QModel(TNQModule):
            
            def __init__(self, num_wires, dtype,batch_size=2):
                super(QModel, self).__init__()
                self.device = TNQMachine(num_wires)
                self.layer = PauliY(wires=0)
                self.batch_size = batch_size
                self.num_wires = num_wires
                
            def forward(self, x, *args, **kwargs):
                self.device.reset_states(batchsize=self.batch_size)
                for i in range(self.num_wires):
                    rx(self.device, wires=i, params=x[i])
                self.layer(q_machine = self.device)
                y = MeasureAll(obs={'Z0': 1})(self.device)
                return y

        x = pyvqnet.tensor.QTensor([[1,0,0,1],[1,1,0,1]],dtype=pyvqnet.kfloat32,requires_grad=True)
        model = QModel(4,pyvqnet.kcomplex64,2)
        y = model(x)
        print(y)


PauliZ
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. py:class:: pyvqnet.qnn.vqc.tn.PauliZ(has_params: bool = False,trainable: bool = False,init_params=None,wires=None,dtype=pyvqnet.kcomplex64,use_dagger=False)
    
    définit une porte quantique PauliZ.

    Cette classe hérite de ``pyvqnet.qnn.vqc.tn.QModule`` et ``torch.nn.Module``.
    Cette classe peut être ajoutée au modèle torch comme sous-module de ``torch.nn.Module``.


    :param has_params: indique s'il a des paramètres, comme les portes RX, RY, etc. qui doivent être définies sur True, et celles sans paramètres doivent être définies sur False, la valeur par défaut est False.
    :param trainable: indique s'il a des paramètres à entraîner. Si la couche utilise des données d'entrée externes pour construire la matrice de la porte logique, définir sur False. Si les paramètres à entraîner doivent être initialisés à partir de cette couche, c'est True, la valeur par défaut est False.
    :param init_params: Paramètres d'initialisation utilisés pour encoder les données classiques QTensor, la valeur par défaut est None.
    :param wires: Index du bit de l'effet de ligne, la valeur par défaut est None.
    :param dtype: La précision des données de la matrice interne de la porte logique peut être définie sur pyvqnet.kcomplex64 ou pyvqnet.kcomplex128, correspondant respectivement à une entrée float ou double.
    :param use_dagger: indique s'il faut utiliser la version conjuguée transposée de la porte, la valeur par défaut est False.
    :return: une instance ``pyvqnet.qnn.vqc.tn.QModule``

    Example::
        
        from pyvqnet.qnn.vqc.tn import PauliZ,TNQMachine,TNQModule,MeasureAll, rx
        import pyvqnet
        pyvqnet.backends.set_backend("torch")

        class QModel(TNQModule):
            
            def __init__(self, num_wires, dtype,batch_size=2):
                super(QModel, self).__init__()
                self.device = TNQMachine(num_wires)
                self.layer = PauliZ(wires=0)
                self.batch_size = batch_size
                self.num_wires = num_wires
                
            def forward(self, x, *args, **kwargs):
                self.device.reset_states(batchsize=self.batch_size)
                for i in range(self.num_wires):
                    rx(self.device, wires=i, params=x[i])
                self.layer(q_machine = self.device)
                y = MeasureAll(obs={'Z0': 1})(self.device)
                return y

        x = pyvqnet.tensor.QTensor([[1,0,0,1],[1,1,0,1]],dtype=pyvqnet.kfloat32,requires_grad=True)
        model = QModel(4,pyvqnet.kcomplex64,2)
        y = model(x)
        print(y)




X1
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. py:class:: pyvqnet.qnn.vqc.tn.X1(has_params: bool = False,trainable: bool = False,init_params=None,wires=None,dtype=pyvqnet.kcomplex64,use_dagger=False)
    
    définit une porte quantique X1.

    Cette classe hérite de ``pyvqnet.qnn.vqc.tn.QModule`` et ``torch.nn.Module``.
    Cette classe peut être ajoutée au modèle torch comme sous-module de ``torch.nn.Module``.

    :param has_params: indique s'il a des paramètres, comme les portes RX, RY, etc. qui doivent être définies sur True, et celles sans paramètres doivent être définies sur False, la valeur par défaut est False.
    :param trainable: indique s'il a des paramètres à entraîner. Si la couche utilise des données d'entrée externes pour construire la matrice de la porte logique, définir sur False. Si les paramètres à entraîner doivent être initialisés à partir de cette couche, c'est True, la valeur par défaut est False.
    :param init_params: Paramètres d'initialisation utilisés pour encoder les données classiques QTensor, la valeur par défaut est None.
    :param wires: Index du bit de l'effet de ligne, la valeur par défaut est None.
    :param dtype: La précision des données de la matrice interne de la porte logique peut être définie sur pyvqnet.kcomplex64 ou pyvqnet.kcomplex128, correspondant respectivement à une entrée float ou double.
    :param use_dagger: indique s'il faut utiliser la version conjuguée transposée de la porte, la valeur par défaut est False.
    :return: une instance ``pyvqnet.qnn.vqc.tn.QModule``

    Example::
        
        from pyvqnet.qnn.vqc.tn import X1,TNQMachine,TNQModule,MeasureAll, rx
        import pyvqnet
        pyvqnet.backends.set_backend("torch")

        class QModel(TNQModule):
            
            def __init__(self, num_wires, dtype,batch_size=2):
                super(QModel, self).__init__()
                self.device = TNQMachine(num_wires)
                self.layer = X1(wires=0)
                self.batch_size = batch_size
                self.num_wires = num_wires
                
            def forward(self, x, *args, **kwargs):
                self.device.reset_states(batchsize=self.batch_size)
                for i in range(self.num_wires):
                    rx(self.device, wires=i, params=x[i])
                self.layer(q_machine = self.device)
                y = MeasureAll(obs={'Z0': 1})(self.device)
                return y

        x = pyvqnet.tensor.QTensor([[1,0,0,1],[1,1,0,1]],dtype=pyvqnet.kfloat32,requires_grad=True)
        model = QModel(4,pyvqnet.kcomplex64,2)
        y = model(x)
        print(y)



RX
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. py:class:: pyvqnet.qnn.vqc.tn.RX(has_params: bool = False,trainable: bool = False,init_params=None,wires=None,dtype=pyvqnet.kcomplex64,use_dagger=False)
    
    définit une porte quantique RX.

    Cette classe hérite de ``pyvqnet.qnn.vqc.tn.QModule`` et ``torch.nn.Module``.
    Cette classe peut être ajoutée au modèle torch comme sous-module de ``torch.nn.Module``.

    :param has_params: indique s'il a des paramètres, comme les portes RX, RY, etc. qui doivent être définies sur True, et celles sans paramètres doivent être définies sur False, la valeur par défaut est False.
    :param trainable: indique s'il a des paramètres à entraîner. Si la couche utilise des données d'entrée externes pour construire la matrice de la porte logique, définir sur False. Si les paramètres à entraîner doivent être initialisés à partir de cette couche, c'est True, la valeur par défaut est False.
    :param init_params: Paramètres d'initialisation utilisés pour encoder les données classiques QTensor, la valeur par défaut est None.
    :param wires: Index du bit de l'effet de ligne, la valeur par défaut est None.
    :param dtype: La précision des données de la matrice interne de la porte logique peut être définie sur pyvqnet.kcomplex64 ou pyvqnet.kcomplex128, correspondant respectivement à une entrée float ou double.
    :param use_dagger: indique s'il faut utiliser la version conjuguée transposée de la porte, la valeur par défaut est False.
    :return: une instance ``pyvqnet.qnn.vqc.tn.QModule``

    Example::

        from pyvqnet.qnn.vqc.tn import RX,TNQMachine,TNQModule,MeasureAll, rx
        import pyvqnet
        pyvqnet.backends.set_backend("torch")

        class QModel(TNQModule):
            
            def __init__(self, num_wires, dtype,batch_size=2):
                super(QModel, self).__init__()
                self.device = TNQMachine(num_wires)
                self.layer = RX(wires=0,has_params=True)
                self.batch_size = batch_size
                self.num_wires = num_wires
                
            def forward(self, x, *args, **kwargs):
                self.device.reset_states(batchsize=self.batch_size)
                for i in range(self.num_wires):
                    rx(self.device, wires=i, params=x[i])
                self.layer(q_machine = self.device)
                y = MeasureAll(obs={'Z0': 1})(self.device)
                return y

        x = pyvqnet.tensor.QTensor([[1,0,0,1],[1,1,0,1]],dtype=pyvqnet.kfloat32,requires_grad=True)
        model = QModel(4,pyvqnet.kcomplex64,2)
        y = model(x)
        print(y)





RY
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. py:class:: pyvqnet.qnn.vqc.tn.RY(has_params: bool = False,trainable: bool = False,init_params=None,wires=None,dtype=pyvqnet.kcomplex64,use_dagger=False)
    
    définit une porte quantique RY.

    Cette classe hérite de ``pyvqnet.qnn.vqc.tn.QModule`` et ``torch.nn.Module``.
    Cette classe peut être ajoutée au modèle torch comme sous-module de ``torch.nn.Module``.

    :param has_params: indique s'il a des paramètres, comme les portes RX, RY, etc. qui doivent être définies sur True, et celles sans paramètres doivent être définies sur False, la valeur par défaut est False.
    :param trainable: indique s'il a des paramètres à entraîner. Si la couche utilise des données d'entrée externes pour construire la matrice de la porte logique, définir sur False. Si les paramètres à entraîner doivent être initialisés à partir de cette couche, c'est True, la valeur par défaut est False.
    :param init_params: Paramètres d'initialisation utilisés pour encoder les données classiques QTensor, la valeur par défaut est None.
    :param wires: Index du bit de l'effet de ligne, la valeur par défaut est None.
    :param dtype: La précision des données de la matrice interne de la porte logique peut être définie sur pyvqnet.kcomplex64 ou pyvqnet.kcomplex128, correspondant respectivement à une entrée float ou double.
    :param use_dagger: indique s'il faut utiliser la version conjuguée transposée de la porte, la valeur par défaut est False.
    :return: une instance ``pyvqnet.qnn.vqc.tn.QModule``

    Example::

        from pyvqnet.qnn.vqc.tn import RY,TNQMachine,TNQModule,MeasureAll, rx
        import pyvqnet
        pyvqnet.backends.set_backend("torch")

        class QModel(TNQModule):
            
            def __init__(self, num_wires, dtype,batch_size=2):
                super(QModel, self).__init__()
                self.device = TNQMachine(num_wires)
                self.layer = RY(wires=0,has_params=True)
                self.batch_size = batch_size
                self.num_wires = num_wires
                
            def forward(self, x, *args, **kwargs):
                self.device.reset_states(batchsize=self.batch_size)
                for i in range(self.num_wires):
                    rx(self.device, wires=i, params=x[i])
                self.layer(q_machine = self.device)
                y = MeasureAll(obs={'Z0': 1})(self.device)
                return y

        x = pyvqnet.tensor.QTensor([[1,0,0,1],[1,1,0,1]],dtype=pyvqnet.kfloat32,requires_grad=True)
        model = QModel(4,pyvqnet.kcomplex64,2)
        y = model(x)
        print(y)


RZ
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. py:class:: pyvqnet.qnn.vqc.tn.RZ(has_params: bool = False,trainable: bool = False,init_params=None,wires=None,dtype=pyvqnet.kcomplex64,use_dagger=False)
    
    définit une porte quantique RZ.

    Cette classe hérite de ``pyvqnet.qnn.vqc.tn.QModule`` et ``torch.nn.Module``.
    Cette classe peut être ajoutée au modèle torch comme sous-module de ``torch.nn.Module``.

    :param has_params: indique s'il a des paramètres, comme les portes RX, RY, etc. qui doivent être définies sur True, et celles sans paramètres doivent être définies sur False, la valeur par défaut est False.
    :param trainable: indique s'il a des paramètres à entraîner. Si la couche utilise des données d'entrée externes pour construire la matrice de la porte logique, définir sur False. Si les paramètres à entraîner doivent être initialisés à partir de cette couche, c'est True, la valeur par défaut est False.
    :param init_params: Paramètres d'initialisation utilisés pour encoder les données classiques QTensor, la valeur par défaut est None.
    :param wires: Index du bit de l'effet de ligne, la valeur par défaut est None.
    :param dtype: La précision des données de la matrice interne de la porte logique peut être définie sur pyvqnet.kcomplex64 ou pyvqnet.kcomplex128, correspondant respectivement à une entrée float ou double.
    :param use_dagger: indique s'il faut utiliser la version conjuguée transposée de la porte, la valeur par défaut est False.
    :return: une instance ``pyvqnet.qnn.vqc.tn.QModule``

    Example::

        from pyvqnet.qnn.vqc.tn import RZ,TNQMachine,TNQModule,MeasureAll, rx
        import pyvqnet
        pyvqnet.backends.set_backend("torch")

        class QModel(TNQModule):
            
            def __init__(self, num_wires, dtype,batch_size=2):
                super(QModel, self).__init__()
                self.device = TNQMachine(num_wires)
                self.layer = RZ(wires=0,has_params=True)
                self.batch_size = batch_size
                self.num_wires = num_wires
                
            def forward(self, x, *args, **kwargs):
                self.device.reset_states(batchsize=self.batch_size)
                for i in range(self.num_wires):
                    rx(self.device, wires=i, params=x[i])
                self.layer(q_machine = self.device)
                y = MeasureAll(obs={'Z0': 1})(self.device)
                return y

        x = pyvqnet.tensor.QTensor([[1,0,0,1],[1,1,0,1]],dtype=pyvqnet.kfloat32,requires_grad=True)
        model = QModel(4,pyvqnet.kcomplex64,2)
        y = model(x)
        print(y)


CRX
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. py:class:: pyvqnet.qnn.vqc.tn.CRX(has_params: bool = False,trainable: bool = False,init_params=None,wires=None,dtype=pyvqnet.kcomplex64,use_dagger=False)
    
    définit une porte quantique CRX.

    Cette classe hérite de ``pyvqnet.qnn.vqc.tn.QModule`` et ``torch.nn.Module``.
    Cette classe peut être ajoutée au modèle torch comme sous-module de ``torch.nn.Module``.

    :param has_params: indique s'il a des paramètres, comme les portes RX, RY, etc. qui doivent être définies sur True, et celles sans paramètres doivent être définies sur False, la valeur par défaut est False.
    :param trainable: indique s'il a des paramètres à entraîner. Si la couche utilise des données d'entrée externes pour construire la matrice de la porte logique, définir sur False. Si les paramètres à entraîner doivent être initialisés à partir de cette couche, c'est True, la valeur par défaut est False.
    :param init_params: Paramètres d'initialisation utilisés pour encoder les données classiques QTensor, la valeur par défaut est None.
    :param wires: Index du bit de l'effet de ligne, la valeur par défaut est None.
    :param dtype: La précision des données de la matrice interne de la porte logique peut être définie sur pyvqnet.kcomplex64 ou pyvqnet.kcomplex128, correspondant respectivement à une entrée float ou double.
    :param use_dagger: indique s'il faut utiliser la version conjuguée transposée de la porte, la valeur par défaut est False.
    :return: une instance ``pyvqnet.qnn.vqc.tn.QModule``

    Example::

        from pyvqnet.qnn.vqc.tn import CRX,TNQMachine,TNQModule,MeasureAll, rx
        import pyvqnet
        pyvqnet.backends.set_backend("torch")

        class QModel(TNQModule):
            
            def __init__(self, num_wires, dtype,batch_size=2):
                super(QModel, self).__init__()
                self.device = TNQMachine(num_wires)
                self.layer = CRX(has_params= True, trainable= True, wires=[0,2])
                self.batch_size = batch_size
                self.num_wires = num_wires
                
            def forward(self, x, *args, **kwargs):
                self.device.reset_states(batchsize=self.batch_size)
                for i in range(self.num_wires):
                    rx(self.device, wires=i, params=x[i])
                self.layer(q_machine = self.device)
                y = MeasureAll(obs={'Z0': 1})(self.device)
                return y

        x = pyvqnet.tensor.QTensor([[1,0,0,1],[1,1,0,1]],dtype=pyvqnet.kfloat32,requires_grad=True)
        model = QModel(4,pyvqnet.kcomplex64,2)
        y = model(x)
        print(y)




CRY
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^


.. py:class:: pyvqnet.qnn.vqc.tn.CRY(has_params: bool = False,trainable: bool = False,init_params=None,wires=None,dtype=pyvqnet.kcomplex64,use_dagger=False)
    
    définit une porte quantique CRY.

    Cette classe hérite de ``pyvqnet.qnn.vqc.tn.QModule`` et ``torch.nn.Module``.
    Cette classe peut être ajoutée au modèle torch comme sous-module de ``torch.nn.Module``.

    :param has_params: indique s'il a des paramètres, comme les portes RX, RY, etc. qui doivent être définies sur True, et celles sans paramètres doivent être définies sur False, la valeur par défaut est False.
    :param trainable: indique s'il a des paramètres à entraîner. Si la couche utilise des données d'entrée externes pour construire la matrice de la porte logique, définir sur False. Si les paramètres à entraîner doivent être initialisés à partir de cette couche, c'est True, la valeur par défaut est False.
    :param init_params: Paramètres d'initialisation utilisés pour encoder les données classiques QTensor, la valeur par défaut est None.
    :param wires: Index du bit de l'effet de ligne, la valeur par défaut est None.
    :param dtype: La précision des données de la matrice interne de la porte logique peut être définie sur pyvqnet.kcomplex64 ou pyvqnet.kcomplex128, correspondant respectivement à une entrée float ou double.
    :param use_dagger: indique s'il faut utiliser la version conjuguée transposée de la porte, la valeur par défaut est False.
    :return: une instance ``pyvqnet.qnn.vqc.tn.QModule``

    Example::

        from pyvqnet.qnn.vqc.tn import CRY,TNQMachine,TNQModule,MeasureAll, rx
        import pyvqnet
        pyvqnet.backends.set_backend("torch")

        class QModel(TNQModule):
            
            def __init__(self, num_wires, dtype,batch_size=2):
                super(QModel, self).__init__()
                self.device = TNQMachine(num_wires)
                self.layer = CRY(has_params= True, trainable= True, wires=[0,2])
                self.batch_size = batch_size
                self.num_wires = num_wires
                
            def forward(self, x, *args, **kwargs):
                self.device.reset_states(batchsize=self.batch_size)
                for i in range(self.num_wires):
                    rx(self.device, wires=i, params=x[i])
                self.layer(q_machine = self.device)
                y = MeasureAll(obs={'Z0': 1})(self.device)
                return y

        x = pyvqnet.tensor.QTensor([[1,0,0,1],[1,1,0,1]],dtype=pyvqnet.kfloat32,requires_grad=True)
        model = QModel(4,pyvqnet.kcomplex64,2)
        y = model(x)
        print(y)


CRZ
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^


.. py:class:: pyvqnet.qnn.vqc.tn.CRZ(has_params: bool = False,trainable: bool = False,init_params=None,wires=None,dtype=pyvqnet.kcomplex64,use_dagger=False)
    
    définit une porte quantique CRZ.

    Cette classe hérite de ``pyvqnet.qnn.vqc.tn.QModule`` et ``torch.nn.Module``.
    Cette classe peut être ajoutée au modèle torch comme sous-module de ``torch.nn.Module``.

    :param has_params: indique s'il a des paramètres, comme les portes RX, RY, etc. qui doivent être définies sur True, et celles sans paramètres doivent être définies sur False, la valeur par défaut est False.
    :param trainable: indique s'il a des paramètres à entraîner. Si la couche utilise des données d'entrée externes pour construire la matrice de la porte logique, définir sur False. Si les paramètres à entraîner doivent être initialisés à partir de cette couche, c'est True, la valeur par défaut est False.
    :param init_params: Paramètres d'initialisation utilisés pour encoder les données classiques QTensor, la valeur par défaut est None.
    :param wires: Index du bit de l'effet de ligne, la valeur par défaut est None.
    :param dtype: La précision des données de la matrice interne de la porte logique peut être définie sur pyvqnet.kcomplex64 ou pyvqnet.kcomplex128, correspondant respectivement à une entrée float ou double.
    :param use_dagger: indique s'il faut utiliser la version conjuguée transposée de la porte, la valeur par défaut est False.
    :return: une instance ``pyvqnet.qnn.vqc.tn.QModule``

    Example::

        from pyvqnet.qnn.vqc.tn import CRZ,TNQMachine,TNQModule,MeasureAll, rx
        import pyvqnet
        pyvqnet.backends.set_backend("torch")

        class QModel(TNQModule):
            
            def __init__(self, num_wires, dtype,batch_size=2):
                super(QModel, self).__init__()
                self.device = TNQMachine(num_wires)
                self.layer = CRZ(has_params= True, trainable= True, wires=[0,2])
                self.batch_size = batch_size
                self.num_wires = num_wires
                
            def forward(self, x, *args, **kwargs):
                self.device.reset_states(batchsize=self.batch_size)
                for i in range(self.num_wires):
                    rx(self.device, wires=i, params=x[i])
                self.layer(q_machine = self.device)
                y = MeasureAll(obs={'Z0': 1})(self.device)
                return y

        x = pyvqnet.tensor.QTensor([[1,0,0,1],[1,1,0,1]],dtype=pyvqnet.kfloat32,requires_grad=True)
        model = QModel(4,pyvqnet.kcomplex64,2)
        y = model(x)
        print(y)



U1
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. py:class:: pyvqnet.qnn.vqc.tn.U1(has_params: bool = False,trainable: bool = False,init_params=None,wires=None,dtype=pyvqnet.kcomplex64,use_dagger=False)
    
    définit une porte quantique U1.

    Cette classe hérite de ``pyvqnet.qnn.vqc.tn.QModule`` et ``torch.nn.Module``.
    Cette classe peut être ajoutée au modèle torch comme sous-module de ``torch.nn.Module``.

    :param has_params: indique s'il a des paramètres, comme les portes RX, RY, etc. qui doivent être définies sur True, et celles sans paramètres doivent être définies sur False, la valeur par défaut est False.
    :param trainable: indique s'il a des paramètres à entraîner. Si la couche utilise des données d'entrée externes pour construire la matrice de la porte logique, définir sur False. Si les paramètres à entraîner doivent être initialisés à partir de cette couche, c'est True, la valeur par défaut est False.
    :param init_params: Paramètres d'initialisation utilisés pour encoder les données classiques QTensor, la valeur par défaut est None.
    :param wires: Index du bit de l'effet de ligne, la valeur par défaut est None.
    :param dtype: La précision des données de la matrice interne de la porte logique peut être définie sur pyvqnet.kcomplex64 ou pyvqnet.kcomplex128, correspondant respectivement à une entrée float ou double.
    :param use_dagger: indique s'il faut utiliser la version conjuguée transposée de la porte, la valeur par défaut est False.
    :return: une instance ``pyvqnet.qnn.vqc.tn.QModule``

    Example::

        from pyvqnet.qnn.vqc.tn import U1,TNQMachine,TNQModule,MeasureAll, rx
        import pyvqnet
        pyvqnet.backends.set_backend("torch")

        class QModel(TNQModule):
            
            def __init__(self, num_wires, dtype,batch_size=2):
                super(QModel, self).__init__()
                self.device = TNQMachine(num_wires)
                self.layer = U1(has_params= True, trainable= True, wires=0)
                self.batch_size = batch_size
                self.num_wires = num_wires
                
            def forward(self, x, *args, **kwargs):
                self.device.reset_states(batchsize=self.batch_size)
                for i in range(self.num_wires):
                    rx(self.device, wires=i, params=x[i])
                self.layer(q_machine = self.device)
                y = MeasureAll(obs={'Z0': 1})(self.device)
                return y

        x = pyvqnet.tensor.QTensor([[1,0,0,1],[1,1,0,1]],dtype=pyvqnet.kfloat32,requires_grad=True)
        model = QModel(4,pyvqnet.kcomplex64,2)
        y = model(x)
        print(y)


U2
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^


.. py:class:: pyvqnet.qnn.vqc.tn.U2(has_params: bool = False,trainable: bool = False,init_params=None,wires=None,dtype=pyvqnet.kcomplex64,use_dagger=False)
    
    définit une porte quantique U2.

    Cette classe hérite de ``pyvqnet.qnn.vqc.tn.QModule`` et ``torch.nn.Module``.
    Cette classe peut être ajoutée au modèle torch comme sous-module de ``torch.nn.Module``.

    :param has_params: indique s'il a des paramètres, comme les portes RX, RY, etc. qui doivent être définies sur True, et celles sans paramètres doivent être définies sur False, la valeur par défaut est False.
    :param trainable: indique s'il a des paramètres à entraîner. Si la couche utilise des données d'entrée externes pour construire la matrice de la porte logique, définir sur False. Si les paramètres à entraîner doivent être initialisés à partir de cette couche, c'est True, la valeur par défaut est False.
    :param init_params: Paramètres d'initialisation utilisés pour encoder les données classiques QTensor, la valeur par défaut est None.
    :param wires: Index du bit de l'effet de ligne, la valeur par défaut est None.
    :param dtype: La précision des données de la matrice interne de la porte logique peut être définie sur pyvqnet.kcomplex64 ou pyvqnet.kcomplex128, correspondant respectivement à une entrée float ou double.
    :param use_dagger: indique s'il faut utiliser la version conjuguée transposée de la porte, la valeur par défaut est False.
    :return: une instance ``pyvqnet.qnn.vqc.tn.QModule``

    Example::

        from pyvqnet.qnn.vqc.tn import U2,TNQMachine,TNQModule,MeasureAll, rx
        import pyvqnet
        pyvqnet.backends.set_backend("torch")

        class QModel(TNQModule):
            
            def __init__(self, num_wires, dtype,batch_size=2):
                super(QModel, self).__init__()
                self.device = TNQMachine(num_wires)
                self.layer = U2(has_params= True, trainable= True, wires=0)
                self.batch_size = batch_size
                self.num_wires = num_wires
                
            def forward(self, x, *args, **kwargs):
                self.device.reset_states(batchsize=self.batch_size)
                for i in range(self.num_wires):
                    rx(self.device, wires=i, params=x[i])
                self.layer(q_machine = self.device)
                y = MeasureAll(obs={'Z0': 1})(self.device)
                return y

        x = pyvqnet.tensor.QTensor([[1,0,0,1],[1,1,0,1]],dtype=pyvqnet.kfloat32,requires_grad=True)
        model = QModel(4,pyvqnet.kcomplex64,2)
        y = model(x)
        print(y)


U3
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^


.. py:class:: pyvqnet.qnn.vqc.tn.U3(has_params: bool = False,trainable: bool = False,init_params=None,wires=None,dtype=pyvqnet.kcomplex64,use_dagger=False)
    
    définit une porte quantique U3.

    Cette classe hérite de ``pyvqnet.qnn.vqc.tn.QModule`` et ``torch.nn.Module``.
    Cette classe peut être ajoutée au modèle torch comme sous-module de ``torch.nn.Module``.

    :param has_params: indique s'il a des paramètres, comme les portes RX, RY, etc. qui doivent être définies sur True, et celles sans paramètres doivent être définies sur False, la valeur par défaut est False.
    :param trainable: indique s'il a des paramètres à entraîner. Si la couche utilise des données d'entrée externes pour construire la matrice de la porte logique, définir sur False. Si les paramètres à entraîner doivent être initialisés à partir de cette couche, c'est True, la valeur par défaut est False.
    :param init_params: Paramètres d'initialisation utilisés pour encoder les données classiques QTensor, la valeur par défaut est None.
    :param wires: Index du bit de l'effet de ligne, la valeur par défaut est None.
    :param dtype: La précision des données de la matrice interne de la porte logique peut être définie sur pyvqnet.kcomplex64 ou pyvqnet.kcomplex128, correspondant respectivement à une entrée float ou double.
    :param use_dagger: indique s'il faut utiliser la version conjuguée transposée de la porte, la valeur par défaut est False.
    :return: une instance ``pyvqnet.qnn.vqc.tn.QModule``

    Example::

        from pyvqnet.qnn.vqc.tn import U3,TNQMachine,TNQModule,MeasureAll, rx
        import pyvqnet
        pyvqnet.backends.set_backend("torch")

        class QModel(TNQModule):
            
            def __init__(self, num_wires, dtype,batch_size=2):
                super(QModel, self).__init__()
                self.device = TNQMachine(num_wires)
                self.layer = U3(has_params= True, trainable= True, wires=0)
                self.batch_size = batch_size
                self.num_wires = num_wires
                
            def forward(self, x, *args, **kwargs):
                self.device.reset_states(batchsize=self.batch_size)
                for i in range(self.num_wires):
                    rx(self.device, wires=i, params=x[i])
                self.layer(q_machine = self.device)
                y = MeasureAll(obs={'Z0': 1})(self.device)
                return y

        x = pyvqnet.tensor.QTensor([[1,0,0,1],[1,1,0,1]],dtype=pyvqnet.kfloat32,requires_grad=True)
        model = QModel(4,pyvqnet.kcomplex64,2)
        y = model(x)
        print(y)



CNOT
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. py:class:: pyvqnet.qnn.vqc.tn.CNOT(has_params: bool = False,trainable: bool = False,init_params=None,wires=None,dtype=pyvqnet.kcomplex64,use_dagger=False)
    
    définit une porte quantique CNOT, alias `CX`.

    Cette classe hérite de ``pyvqnet.qnn.vqc.tn.QModule`` et ``torch.nn.Module``.
    Cette classe peut être ajoutée au modèle torch comme sous-module de ``torch.nn.Module``.

    :param has_params: indique s'il a des paramètres, comme les portes RX, RY, etc. qui doivent être définies sur True, et celles sans paramètres doivent être définies sur False, la valeur par défaut est False.
    :param trainable: indique s'il a des paramètres à entraîner. Si la couche utilise des données d'entrée externes pour construire la matrice de la porte logique, définir sur False. Si les paramètres à entraîner doivent être initialisés à partir de cette couche, c'est True, la valeur par défaut est False.
    :param init_params: Paramètres d'initialisation utilisés pour encoder les données classiques QTensor, la valeur par défaut est None.
    :param wires: Index du bit de l'effet de ligne, la valeur par défaut est None.
    :param dtype: La précision des données de la matrice interne de la porte logique peut être définie sur pyvqnet.kcomplex64 ou pyvqnet.kcomplex128, correspondant respectivement à une entrée float ou double.
    :param use_dagger: indique s'il faut utiliser la version conjuguée transposée de la porte, la valeur par défaut est False.
    :return: une instance ``pyvqnet.qnn.vqc.tn.QModule``

    Example::

        from pyvqnet.qnn.vqc.tn import CNOT,TNQMachine,TNQModule,MeasureAll, rx
        import pyvqnet
        pyvqnet.backends.set_backend("torch")

        class QModel(TNQModule):
            
            def __init__(self, num_wires, dtype,batch_size=2):
                super(QModel, self).__init__()
                self.device = TNQMachine(num_wires)
                self.layer = CNOT(wires=[0,1])
                self.batch_size = batch_size
                self.num_wires = num_wires
                
            def forward(self, x, *args, **kwargs):
                self.device.reset_states(batchsize=self.batch_size)
                for i in range(self.num_wires):
                    rx(self.device, wires=i, params=x[i])
                self.layer(q_machine = self.device)
                y = MeasureAll(obs={'Z0': 1})(self.device)
                return y

        x = pyvqnet.tensor.QTensor([[1,0,0,1],[1,1,0,1]],dtype=pyvqnet.kfloat32,requires_grad=True)
        model = QModel(4,pyvqnet.kcomplex64,2)
        y = model(x)
        print(y)

CY
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. py:class:: pyvqnet.qnn.vqc.tn.CY(has_params: bool = False,trainable: bool = False,init_params=None,wires=None,dtype=pyvqnet.kcomplex64,use_dagger=False)
    
    définit une porte quantique CY.

    Cette classe hérite de ``pyvqnet.qnn.vqc.tn.QModule`` et ``torch.nn.Module``.
    Cette classe peut être ajoutée au modèle torch comme sous-module de ``torch.nn.Module``.

    :param has_params: indique s'il a des paramètres, comme les portes RX, RY, etc. qui doivent être définies sur True, et celles sans paramètres doivent être définies sur False, la valeur par défaut est False.
    :param trainable: indique s'il a des paramètres à entraîner. Si la couche utilise des données d'entrée externes pour construire la matrice de la porte logique, définir sur False. Si les paramètres à entraîner doivent être initialisés à partir de cette couche, c'est True, la valeur par défaut est False.
    :param init_params: Paramètres d'initialisation utilisés pour encoder les données classiques QTensor, la valeur par défaut est None.
    :param wires: Index du bit de l'effet de ligne, la valeur par défaut est None.
    :param dtype: La précision des données de la matrice interne de la porte logique peut être définie sur pyvqnet.kcomplex64 ou pyvqnet.kcomplex128, correspondant respectivement à une entrée float ou double.
    :param use_dagger: indique s'il faut utiliser la version conjuguée transposée de la porte, la valeur par défaut est False.
    :return: une instance ``pyvqnet.qnn.vqc.tn.QModule``

    Example::

        from pyvqnet.qnn.vqc.tn import CY,TNQMachine,TNQModule,MeasureAll, rx
        import pyvqnet
        pyvqnet.backends.set_backend("torch")

        class QModel(TNQModule):
            
            def __init__(self, num_wires, dtype,batch_size=2):
                super(QModel, self).__init__()
                self.device = TNQMachine(num_wires)
                self.layer = CY(wires=[0,1])
                self.batch_size = batch_size
                self.num_wires = num_wires
                
            def forward(self, x, *args, **kwargs):
                self.device.reset_states(batchsize=self.batch_size)
                for i in range(self.num_wires):
                    rx(self.device, wires=i, params=x[i])
                self.layer(q_machine = self.device)
                y = MeasureAll(obs={'Z0': 1})(self.device)
                return y

        x = pyvqnet.tensor.QTensor([[1,0,0,1],[1,1,0,1]],dtype=pyvqnet.kfloat32,requires_grad=True)
        model = QModel(4,pyvqnet.kcomplex64,2)
        y = model(x)
        print(y)


CZ
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. py:class:: pyvqnet.qnn.vqc.tn.CZ(has_params: bool = False,trainable: bool = False,init_params=None,wires=None,dtype=pyvqnet.kcomplex64,use_dagger=False)
    
    définit une porte quantique CZ.

    Cette classe hérite de ``pyvqnet.qnn.vqc.tn.QModule`` et ``torch.nn.Module``.
    Cette classe peut être ajoutée au modèle torch comme sous-module de ``torch.nn.Module``.

    :param has_params: indique s'il a des paramètres, comme les portes RX, RY, etc. qui doivent être définies sur True, et celles sans paramètres doivent être définies sur False, la valeur par défaut est False.
    :param trainable: indique s'il a des paramètres à entraîner. Si la couche utilise des données d'entrée externes pour construire la matrice de la porte logique, définir sur False. Si les paramètres à entraîner doivent être initialisés à partir de cette couche, c'est True, la valeur par défaut est False.
    :param init_params: Paramètres d'initialisation utilisés pour encoder les données classiques QTensor, la valeur par défaut est None.
    :param wires: Index du bit de l'effet de ligne, la valeur par défaut est None.
    :param dtype: La précision des données de la matrice interne de la porte logique peut être définie sur pyvqnet.kcomplex64 ou pyvqnet.kcomplex128, correspondant respectivement à une entrée float ou double.
    :param use_dagger: indique s'il faut utiliser la version conjuguée transposée de la porte, la valeur par défaut est False.
    :return: une instance ``pyvqnet.qnn.vqc.tn.QModule``

    Example::

        from pyvqnet.qnn.vqc.tn import CZ,TNQMachine,TNQModule,MeasureAll, rx
        import pyvqnet
        pyvqnet.backends.set_backend("torch")

        class QModel(TNQModule):
            
            def __init__(self, num_wires, dtype,batch_size=2):
                super(QModel, self).__init__()
                self.device = TNQMachine(num_wires)
                self.layer = CZ(wires=[0,1])
                self.batch_size = batch_size
                self.num_wires = num_wires
                
            def forward(self, x, *args, **kwargs):
                self.device.reset_states(batchsize=self.batch_size)
                for i in range(self.num_wires):
                    rx(self.device, wires=i, params=x[i])
                self.layer(q_machine = self.device)
                y = MeasureAll(obs={'Z0': 1})(self.device)
                return y

        x = pyvqnet.tensor.QTensor([[1,0,0,1],[1,1,0,1]],dtype=pyvqnet.kfloat32,requires_grad=True)
        model = QModel(4,pyvqnet.kcomplex64,2)
        y = model(x)
        print(y)





CR
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. py:class:: pyvqnet.qnn.vqc.tn.CR(has_params: bool = False,trainable: bool = False,init_params=None,wires=None,dtype=pyvqnet.kcomplex64,use_dagger=False)
    
    définit une porte quantique CR.

    Cette classe hérite de ``pyvqnet.qnn.vqc.tn.QModule`` et ``torch.nn.Module``.
    Cette classe peut être ajoutée au modèle torch comme sous-module de ``torch.nn.Module``.

    :param has_params: indique s'il a des paramètres, comme les portes RX, RY, etc. qui doivent être définies sur True, et celles sans paramètres doivent être définies sur False, la valeur par défaut est False.
    :param trainable: indique s'il a des paramètres à entraîner. Si la couche utilise des données d'entrée externes pour construire la matrice de la porte logique, définir sur False. Si les paramètres à entraîner doivent être initialisés à partir de cette couche, c'est True, la valeur par défaut est False.
    :param init_params: Paramètres d'initialisation utilisés pour encoder les données classiques QTensor, la valeur par défaut est None.
    :param wires: Index du bit de l'effet de ligne, la valeur par défaut est None.
    :param dtype: La précision des données de la matrice interne de la porte logique peut être définie sur pyvqnet.kcomplex64 ou pyvqnet.kcomplex128, correspondant respectivement à une entrée float ou double.
    :param use_dagger: indique s'il faut utiliser la version conjuguée transposée de la porte, la valeur par défaut est False.
    :return: une instance ``pyvqnet.qnn.vqc.tn.QModule``

    Example::

        from pyvqnet.qnn.vqc.tn import CR,TNQMachine,TNQModule,MeasureAll, rx
        import pyvqnet
        pyvqnet.backends.set_backend("torch")

        class QModel(TNQModule):
            
            def __init__(self, num_wires, dtype,batch_size=2):
                super(QModel, self).__init__()
                self.device = TNQMachine(num_wires)
                self.layer = CR(has_params= True, trainable= True, wires=[0,2])
                self.batch_size = batch_size
                self.num_wires = num_wires
                
            def forward(self, x, *args, **kwargs):
                self.device.reset_states(batchsize=self.batch_size)
                for i in range(self.num_wires):
                    rx(self.device, wires=i, params=x[i])
                self.layer(q_machine = self.device)
                y = MeasureAll(obs={'Z0': 1})(self.device)
                return y

        x = pyvqnet.tensor.QTensor([[1,0,0,1],[1,1,0,1]],dtype=pyvqnet.kfloat32,requires_grad=True)
        model = QModel(4,pyvqnet.kcomplex64,2)
        y = model(x)
        print(y)



SWAP
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^


.. py:class:: pyvqnet.qnn.vqc.tn.SWAP(has_params: bool = False,trainable: bool = False,init_params=None,wires=None,dtype=pyvqnet.kcomplex64,use_dagger=False)
    
    définit une porte quantique CSWAP.

    Cette classe hérite de ``pyvqnet.qnn.vqc.tn.QModule`` et ``torch.nn.Module``.
    Cette classe peut être ajoutée au modèle torch comme sous-module de ``torch.nn.Module``.

    :param has_params: indique s'il a des paramètres, comme les portes RX, RY, etc. qui doivent être définies sur True, et celles sans paramètres doivent être définies sur False, la valeur par défaut est False.
    :param trainable: indique s'il a des paramètres à entraîner. Si la couche utilise des données d'entrée externes pour construire la matrice de la porte logique, définir sur False. Si les paramètres à entraîner doivent être initialisés à partir de cette couche, c'est True, la valeur par défaut est False.
    :param init_params: Paramètres d'initialisation utilisés pour encoder les données classiques QTensor, la valeur par défaut est None.
    :param wires: Index du bit de l'effet de ligne, la valeur par défaut est None.
    :param dtype: La précision des données de la matrice interne de la porte logique peut être définie sur pyvqnet.kcomplex64 ou pyvqnet.kcomplex128, correspondant respectivement à une entrée float ou double.
    :param use_dagger: indique s'il faut utiliser la version conjuguée transposée de la porte, la valeur par défaut est False.
    :return: une instance ``pyvqnet.qnn.vqc.tn.QModule``

    Example::

        from pyvqnet.qnn.vqc.tn import SWAP,TNQMachine,TNQModule,MeasureAll, rx
        import pyvqnet
        pyvqnet.backends.set_backend("torch")

        class QModel(TNQModule):
            
            def __init__(self, num_wires, dtype,batch_size=2):
                super(QModel, self).__init__()
                self.device = TNQMachine(num_wires)
                self.layer = SWAP(wires=[0,1])
                self.batch_size = batch_size
                self.num_wires = num_wires
                
            def forward(self, x, *args, **kwargs):
                self.device.reset_states(batchsize=self.batch_size)
                for i in range(self.num_wires):
                    rx(self.device, wires=i, params=x[i])
                self.layer(q_machine = self.device)
                y = MeasureAll(obs={'Z0': 1})(self.device)
                return y

        x = pyvqnet.tensor.QTensor([[1,0,0,1],[1,1,0,1]],dtype=pyvqnet.kfloat32,requires_grad=True)
        model = QModel(4,pyvqnet.kcomplex64,2)
        y = model(x)
        print(y)


CSWAP
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. py:class:: pyvqnet.qnn.vqc.tn.CSWAP(has_params: bool = False,trainable: bool = False,init_params=None,wires=None,dtype=pyvqnet.kcomplex64,use_dagger=False)
    
    définit une porte quantique CSWAP.

    .. math:: CSWAP = \begin{bmatrix}
            1 & 0 & 0 & 0 & 0 & 0 & 0 & 0 \\
            0 & 1 & 0 & 0 & 0 & 0 & 0 & 0 \\
            0 & 0 & 1 & 0 & 0 & 0 & 0 & 0 \\
            0 & 0 & 0 & 1 & 0 & 0 & 0 & 0 \\
            0 & 0 & 0 & 0 & 1 & 0 & 0 & 0 \\
            0 & 0 & 0 & 0 & 0 & 0 & 1 & 0 \\
            0 & 0 & 0 & 0 & 0 & 1 & 0 & 0 \\
            0 & 0 & 0 & 0 & 0 & 0 & 0 & 1
        \end{bmatrix}.

    Cette classe hérite de ``pyvqnet.qnn.vqc.tn.QModule`` et ``torch.nn.Module``.
    Cette classe peut être ajoutée au modèle torch comme sous-module de ``torch.nn.Module``.

    :param has_params: indique s'il a des paramètres, comme les portes RX, RY, etc. qui doivent être définies sur True, et celles sans paramètres doivent être définies sur False, la valeur par défaut est False.
    :param trainable: indique s'il a des paramètres à entraîner. Si la couche utilise des données d'entrée externes pour construire la matrice de la porte logique, définir sur False. Si les paramètres à entraîner doivent être initialisés à partir de cette couche, c'est True, la valeur par défaut est False.
    :param init_params: Paramètres d'initialisation utilisés pour encoder les données classiques QTensor, la valeur par défaut est None.
    :param wires: Index du bit de l'effet de ligne, la valeur par défaut est None.
    :param dtype: La précision des données de la matrice interne de la porte logique peut être définie sur pyvqnet.kcomplex64 ou pyvqnet.kcomplex128, correspondant respectivement à une entrée float ou double.
    :param use_dagger: indique s'il faut utiliser la version conjuguée transposée de la porte, la valeur par défaut est False.
    :return: une instance ``pyvqnet.qnn.vqc.tn.QModule``

    Example::

        from pyvqnet.qnn.vqc.tn import CSWAP,TNQMachine,TNQModule,MeasureAll, rx
        import pyvqnet
        pyvqnet.backends.set_backend("torch")

        class QModel(TNQModule):
            
            def __init__(self, num_wires, dtype,batch_size=2):
                super(QModel, self).__init__()
                self.device = TNQMachine(num_wires)
                self.layer = CSWAP(wires=[0,1,2])
                self.batch_size = batch_size
                self.num_wires = num_wires
                
            def forward(self, x, *args, **kwargs):
                self.device.reset_states(batchsize=self.batch_size)
                for i in range(self.num_wires):
                    rx(self.device, wires=i, params=x[i])
                self.layer(q_machine = self.device)
                y = MeasureAll(obs={'Z0': 1})(self.device)
                return y

        x = pyvqnet.tensor.QTensor([[1,0,0,1],[1,1,0,1]],dtype=pyvqnet.kfloat32,requires_grad=True)
        model = QModel(4,pyvqnet.kcomplex64,2)
        y = model(x)
        print(y)

RXX
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^


.. py:class:: pyvqnet.qnn.vqc.tn.RXX(has_params: bool = False,trainable: bool = False,init_params=None,wires=None,dtype=pyvqnet.kcomplex64,use_dagger=False)
    
    définit une porte quantique RXX.

    Cette classe hérite de ``pyvqnet.qnn.vqc.tn.QModule`` et ``torch.nn.Module``.
    Cette classe peut être ajoutée au modèle torch comme sous-module de ``torch.nn.Module``.

    :param has_params: indique s'il a des paramètres, comme les portes RX, RY, etc. qui doivent être définies sur True, et celles sans paramètres doivent être définies sur False, la valeur par défaut est False.
    :param trainable: indique s'il a des paramètres à entraîner. Si la couche utilise des données d'entrée externes pour construire la matrice de la porte logique, définir sur False. Si les paramètres à entraîner doivent être initialisés à partir de cette couche, c'est True, la valeur par défaut est False.
    :param init_params: Paramètres d'initialisation utilisés pour encoder les données classiques QTensor, la valeur par défaut est None.
    :param wires: Index du bit de l'effet de ligne, la valeur par défaut est None.
    :param dtype: La précision des données de la matrice interne de la porte logique peut être définie sur pyvqnet.kcomplex64 ou pyvqnet.kcomplex128, correspondant respectivement à une entrée float ou double.
    :param use_dagger: indique s'il faut utiliser la version conjuguée transposée de la porte, la valeur par défaut est False.
    :return: une instance ``pyvqnet.qnn.vqc.tn.QModule``

    Example::

        from pyvqnet.qnn.vqc.tn import RXX,TNQMachine,TNQModule,MeasureAll, rx
        import pyvqnet
        pyvqnet.backends.set_backend("torch")

        class QModel(TNQModule):
            
            def __init__(self, num_wires, dtype,batch_size=2):
                super(QModel, self).__init__()
                self.device = TNQMachine(num_wires)
                self.layer = RXX(has_params= True, trainable= True, wires=[0,2])
                self.batch_size = batch_size
                self.num_wires = num_wires
                
            def forward(self, x, *args, **kwargs):
                self.device.reset_states(batchsize=self.batch_size)
                for i in range(self.num_wires):
                    rx(self.device, wires=i, params=x[i])
                self.layer(q_machine = self.device)
                y = MeasureAll(obs={'Z0': 1})(self.device)
                return y

        x = pyvqnet.tensor.QTensor([[1,0,0,1],[1,1,0,1]],dtype=pyvqnet.kfloat32,requires_grad=True)
        model = QModel(4,pyvqnet.kcomplex64,2)
        y = model(x)
        print(y)

RYY
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. py:class:: pyvqnet.qnn.vqc.tn.RYY(has_params: bool = False,trainable: bool = False,init_params=None,wires=None,dtype=pyvqnet.kcomplex64,use_dagger=False)
    
    définit une porte quantique RYY.

    Cette classe hérite de ``pyvqnet.qnn.vqc.tn.QModule`` et ``torch.nn.Module``.
    Cette classe peut être ajoutée au modèle torch comme sous-module de ``torch.nn.Module``.

    :param has_params: indique s'il a des paramètres, comme les portes RX, RY, etc. qui doivent être définies sur True, et celles sans paramètres doivent être définies sur False, la valeur par défaut est False.
    :param trainable: indique s'il a des paramètres à entraîner. Si la couche utilise des données d'entrée externes pour construire la matrice de la porte logique, définir sur False. Si les paramètres à entraîner doivent être initialisés à partir de cette couche, c'est True, la valeur par défaut est False.
    :param init_params: Paramètres d'initialisation utilisés pour encoder les données classiques QTensor, la valeur par défaut est None.
    :param wires: Index du bit de l'effet de ligne, la valeur par défaut est None.
    :param dtype: La précision des données de la matrice interne de la porte logique peut être définie sur pyvqnet.kcomplex64 ou pyvqnet.kcomplex128, correspondant respectivement à une entrée float ou double.
    :param use_dagger: indique s'il faut utiliser la version conjuguée transposée de la porte, la valeur par défaut est False.
    :return: une instance ``pyvqnet.qnn.vqc.tn.QModule``

    Example::

        from pyvqnet.qnn.vqc.tn import RYY,TNQMachine,TNQModule,MeasureAll, rx
        import pyvqnet
        pyvqnet.backends.set_backend("torch")

        class QModel(TNQModule):
            
            def __init__(self, num_wires, dtype,batch_size=2):
                super(QModel, self).__init__()
                self.device = TNQMachine(num_wires)
                self.layer = RYY(has_params= True, trainable= True, wires=[0,2])
                self.batch_size = batch_size
                self.num_wires = num_wires
                
            def forward(self, x, *args, **kwargs):
                self.device.reset_states(batchsize=self.batch_size)
                for i in range(self.num_wires):
                    rx(self.device, wires=i, params=x[i])
                self.layer(q_machine = self.device)
                y = MeasureAll(obs={'Z0': 1})(self.device)
                return y

        x = pyvqnet.tensor.QTensor([[1,0,0,1],[1,1,0,1]],dtype=pyvqnet.kfloat32,requires_grad=True)
        model = QModel(4,pyvqnet.kcomplex64,2)
        y = model(x)
        print(y)


RZZ
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. py:class:: pyvqnet.qnn.vqc.tn.RZZ(has_params: bool = False,trainable: bool = False,init_params=None,wires=None,dtype=pyvqnet.kcomplex64,use_dagger=False)
    
    définit une porte quantique RZZ.

    Cette classe hérite de ``pyvqnet.qnn.vqc.tn.QModule`` et ``torch.nn.Module``.
    Cette classe peut être ajoutée au modèle torch comme sous-module de ``torch.nn.Module``.

    :param has_params: indique s'il a des paramètres, comme les portes RX, RY, etc. qui doivent être définies sur True, et celles sans paramètres doivent être définies sur False, la valeur par défaut est False.
    :param trainable: indique s'il a des paramètres à entraîner. Si la couche utilise des données d'entrée externes pour construire la matrice de la porte logique, définir sur False. Si les paramètres à entraîner doivent être initialisés à partir de cette couche, c'est True, la valeur par défaut est False.
    :param init_params: Paramètres d'initialisation utilisés pour encoder les données classiques QTensor, la valeur par défaut est None.
    :param wires: Index du bit de l'effet de ligne, la valeur par défaut est None.
    :param dtype: La précision des données de la matrice interne de la porte logique peut être définie sur pyvqnet.kcomplex64 ou pyvqnet.kcomplex128, correspondant respectivement à une entrée float ou double.
    :param use_dagger: indique s'il faut utiliser la version conjuguée transposée de la porte, la valeur par défaut est False.
    :return: une instance ``pyvqnet.qnn.vqc.tn.QModule``

    Example::

        from pyvqnet.qnn.vqc.tn import RZZ,TNQMachine,TNQModule,MeasureAll, rx
        import pyvqnet
        pyvqnet.backends.set_backend("torch")

        class QModel(TNQModule):
            
            def __init__(self, num_wires, dtype,batch_size=2):
                super(QModel, self).__init__()
                self.device = TNQMachine(num_wires)
                self.layer = RZZ(has_params= True, trainable= True, wires=[0,2])
                self.batch_size = batch_size
                self.num_wires = num_wires
                
            def forward(self, x, *args, **kwargs):
                self.device.reset_states(batchsize=self.batch_size)
                for i in range(self.num_wires):
                    rx(self.device, wires=i, params=x[i])
                self.layer(q_machine = self.device)
                y = MeasureAll(obs={'Z0': 1})(self.device)
                return y

        x = pyvqnet.tensor.QTensor([[1,0,0,1],[1,1,0,1]],dtype=pyvqnet.kfloat32,requires_grad=True)
        model = QModel(4,pyvqnet.kcomplex64,2)
        y = model(x)
        print(y)



RZX
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. py:class:: pyvqnet.qnn.vqc.tn.RZX(has_params: bool = False,trainable: bool = False,init_params=None,wires=None,dtype=pyvqnet.kcomplex64,use_dagger=False)
    
    définit une porte quantique RZX.

    Cette classe hérite de ``pyvqnet.qnn.vqc.tn.QModule`` et ``torch.nn.Module``.
    Cette classe peut être ajoutée au modèle torch comme sous-module de ``torch.nn.Module``.

    :param has_params: indique s'il a des paramètres, comme les portes RX, RY, etc. qui doivent être définies sur True, et celles sans paramètres doivent être définies sur False, la valeur par défaut est False.
    :param trainable: indique s'il a des paramètres à entraîner. Si la couche utilise des données d'entrée externes pour construire la matrice de la porte logique, définir sur False. Si les paramètres à entraîner doivent être initialisés à partir de cette couche, c'est True, la valeur par défaut est False.
    :param init_params: Paramètres d'initialisation utilisés pour encoder les données classiques QTensor, la valeur par défaut est None.
    :param wires: Index du bit de l'effet de ligne, la valeur par défaut est None.
    :param dtype: La précision des données de la matrice interne de la porte logique peut être définie sur pyvqnet.kcomplex64 ou pyvqnet.kcomplex128, correspondant respectivement à une entrée float ou double.
    :param use_dagger: indique s'il faut utiliser la version conjuguée transposée de la porte, la valeur par défaut est False.
    :return: une instance ``pyvqnet.qnn.vqc.tn.QModule``

    Example::

        from pyvqnet.qnn.vqc.tn import RZX,TNQMachine,TNQModule,MeasureAll, rx
        import pyvqnet
        pyvqnet.backends.set_backend("torch")

        class QModel(TNQModule):
            
            def __init__(self, num_wires, dtype,batch_size=2):
                super(QModel, self).__init__()
                self.device = TNQMachine(num_wires)
                self.layer = RZX(has_params= True, trainable= True, wires=[0,2])
                self.batch_size = batch_size
                self.num_wires = num_wires
                
            def forward(self, x, *args, **kwargs):
                self.device.reset_states(batchsize=self.batch_size)
                for i in range(self.num_wires):
                    rx(self.device, wires=i, params=x[i])
                self.layer(q_machine = self.device)
                y = MeasureAll(obs={'Z0': 1})(self.device)
                return y

        x = pyvqnet.tensor.QTensor([[1,0,0,1],[1,1,0,1]],dtype=pyvqnet.kfloat32,requires_grad=True)
        model = QModel(4,pyvqnet.kcomplex64,2)
        y = model(x)
        print(y)

Toffoli
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. py:class:: pyvqnet.qnn.vqc.tn.Toffoli(has_params: bool = False,trainable: bool = False,init_params=None,wires=None,dtype=pyvqnet.kcomplex64,use_dagger=False)
    
    définit une porte quantique Toffoli.

    Cette classe hérite de ``pyvqnet.qnn.vqc.tn.QModule`` et ``torch.nn.Module``.
    Cette classe peut être ajoutée au modèle torch comme sous-module de ``torch.nn.Module``.

    :param has_params: indique s'il a des paramètres, comme les portes RX, RY, etc. qui doivent être définies sur True, et celles sans paramètres doivent être définies sur False, la valeur par défaut est False.
    :param trainable: indique s'il a des paramètres à entraîner. Si la couche utilise des données d'entrée externes pour construire la matrice de la porte logique, définir sur False. Si les paramètres à entraîner doivent être initialisés à partir de cette couche, c'est True, la valeur par défaut est False.
    :param init_params: Paramètres d'initialisation utilisés pour encoder les données classiques QTensor, la valeur par défaut est None.
    :param wires: Index du bit de l'effet de ligne, la valeur par défaut est None.
    :param dtype: La précision des données de la matrice interne de la porte logique peut être définie sur pyvqnet.kcomplex64 ou pyvqnet.kcomplex128, correspondant respectivement à une entrée float ou double.
    :param use_dagger: indique s'il faut utiliser la version conjuguée transposée de la porte, la valeur par défaut est False.
    :return: une instance ``pyvqnet.qnn.vqc.tn.QModule``

    Example::

        from pyvqnet.qnn.vqc.tn import Toffoli,TNQMachine,TNQModule,MeasureAll, rx
        import pyvqnet
        pyvqnet.backends.set_backend("torch")

        class QModel(TNQModule):
            
            def __init__(self, num_wires, dtype,batch_size=2):
                super(QModel, self).__init__()
                self.device = TNQMachine(num_wires)
                self.layer = Toffoli(wires=[0,2,1])
                self.batch_size = batch_size
                self.num_wires = num_wires
                
            def forward(self, x, *args, **kwargs):
                self.device.reset_states(batchsize=self.batch_size)
                for i in range(self.num_wires):
                    rx(self.device, wires=i, params=x[i])
                self.layer(q_machine = self.device)
                y = MeasureAll(obs={'Z0': 1})(self.device)
                return y

        x = pyvqnet.tensor.QTensor([[1,0,0,1],[1,1,0,1]],dtype=pyvqnet.kfloat32,requires_grad=True)
        model = QModel(4,pyvqnet.kcomplex64,2)
        y = model(x)
        print(y)

IsingXX
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^


.. py:class:: pyvqnet.qnn.vqc.tn.IsingXX(has_params: bool = False,trainable: bool = False,init_params=None,wires=None,dtype=pyvqnet.kcomplex64,use_dagger=False)
    
    définit une porte quantique IsingXX.

    Cette classe hérite de ``pyvqnet.qnn.vqc.tn.QModule`` et ``torch.nn.Module``.
    Cette classe peut être ajoutée au modèle torch comme sous-module de ``torch.nn.Module``.

    :param has_params: indique s'il a des paramètres, comme les portes RX, RY, etc. qui doivent être définies sur True, et celles sans paramètres doivent être définies sur False, la valeur par défaut est False.
    :param trainable: indique s'il a des paramètres à entraîner. Si la couche utilise des données d'entrée externes pour construire la matrice de la porte logique, définir sur False. Si les paramètres à entraîner doivent être initialisés à partir de cette couche, c'est True, la valeur par défaut est False.
    :param init_params: Paramètres d'initialisation utilisés pour encoder les données classiques QTensor, la valeur par défaut est None.
    :param wires: Index du bit de l'effet de ligne, la valeur par défaut est None.
    :param dtype: La précision des données de la matrice interne de la porte logique peut être définie sur pyvqnet.kcomplex64 ou pyvqnet.kcomplex128, correspondant respectivement à une entrée float ou double.
    :param use_dagger: indique s'il faut utiliser la version conjuguée transposée de la porte, la valeur par défaut est False.
    :return: une instance ``pyvqnet.qnn.vqc.tn.QModule``

    Example::

        from pyvqnet.qnn.vqc.tn import IsingXX,TNQMachine,TNQModule,MeasureAll, rx
        import pyvqnet
        pyvqnet.backends.set_backend("torch")

        class QModel(TNQModule):
            
            def __init__(self, num_wires, dtype,batch_size=2):
                super(QModel, self).__init__()
                self.device = TNQMachine(num_wires)
                self.layer = IsingXX(has_params= True, trainable= True, wires=[0,2])
                self.batch_size = batch_size
                self.num_wires = num_wires
                
            def forward(self, x, *args, **kwargs):
                self.device.reset_states(batchsize=self.batch_size)
                for i in range(self.num_wires):
                    rx(self.device, wires=i, params=x[i])
                self.layer(q_machine = self.device)
                y = MeasureAll(obs={'Z0': 1})(self.device)
                return y

        x = pyvqnet.tensor.QTensor([[1,0,0,1],[1,1,0,1]],dtype=pyvqnet.kfloat32,requires_grad=True)
        model = QModel(4,pyvqnet.kcomplex64,2)
        y = model(x)
        print(y)


IsingYY
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^


.. py:class:: pyvqnet.qnn.vqc.tn.IsingYY(has_params: bool = False,trainable: bool = False,init_params=None,wires=None,dtype=pyvqnet.kcomplex64,use_dagger=False)
    
    définit une porte quantique IsingYY.

    Cette classe hérite de ``pyvqnet.qnn.vqc.tn.QModule`` et ``torch.nn.Module``.
    Cette classe peut être ajoutée au modèle torch comme sous-module de ``torch.nn.Module``.

    :param has_params: indique s'il a des paramètres, comme les portes RX, RY, etc. qui doivent être définies sur True, et celles sans paramètres doivent être définies sur False, la valeur par défaut est False.
    :param trainable: indique s'il a des paramètres à entraîner. Si la couche utilise des données d'entrée externes pour construire la matrice de la porte logique, définir sur False. Si les paramètres à entraîner doivent être initialisés à partir de cette couche, c'est True, la valeur par défaut est False.
    :param init_params: Paramètres d'initialisation utilisés pour encoder les données classiques QTensor, la valeur par défaut est None.
    :param wires: Index du bit de l'effet de ligne, la valeur par défaut est None.
    :param dtype: La précision des données de la matrice interne de la porte logique peut être définie sur pyvqnet.kcomplex64 ou pyvqnet.kcomplex128, correspondant respectivement à une entrée float ou double.
    :param use_dagger: indique s'il faut utiliser la version conjuguée transposée de la porte, la valeur par défaut est False.
    :return: une instance ``pyvqnet.qnn.vqc.tn.QModule``

    Example::

        from pyvqnet.qnn.vqc.tn import IsingYY,TNQMachine,TNQModule,MeasureAll, rx
        import pyvqnet
        pyvqnet.backends.set_backend("torch")

        class QModel(TNQModule):
            
            def __init__(self, num_wires, dtype,batch_size=2):
                super(QModel, self).__init__()
                self.device = TNQMachine(num_wires)
                self.layer = IsingYY(has_params= True, trainable= True, wires=[0,2])
                self.batch_size = batch_size
                self.num_wires = num_wires
                
            def forward(self, x, *args, **kwargs):
                self.device.reset_states(batchsize=self.batch_size)
                for i in range(self.num_wires):
                    rx(self.device, wires=i, params=x[i])
                self.layer(q_machine = self.device)
                y = MeasureAll(obs={'Z0': 1})(self.device)
                return y

        x = pyvqnet.tensor.QTensor([[1,0,0,1],[1,1,0,1]],dtype=pyvqnet.kfloat32,requires_grad=True)
        model = QModel(4,pyvqnet.kcomplex64,2)
        y = model(x)
        print(y)


IsingZZ
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. py:class:: pyvqnet.qnn.vqc.tn.IsingZZ(has_params: bool = False,trainable: bool = False,init_params=None,wires=None,dtype=pyvqnet.kcomplex64,use_dagger=False)
    
    définit une porte quantique IsingZZ.

    Cette classe hérite de ``pyvqnet.qnn.vqc.tn.QModule`` et ``torch.nn.Module``.
    Cette classe peut être ajoutée au modèle torch comme sous-module de ``torch.nn.Module``.

    :param has_params: indique s'il a des paramètres, comme les portes RX, RY, etc. qui doivent être définies sur True, et celles sans paramètres doivent être définies sur False, la valeur par défaut est False.
    :param trainable: indique s'il a des paramètres à entraîner. Si la couche utilise des données d'entrée externes pour construire la matrice de la porte logique, définir sur False. Si les paramètres à entraîner doivent être initialisés à partir de cette couche, c'est True, la valeur par défaut est False.
    :param init_params: Paramètres d'initialisation utilisés pour encoder les données classiques QTensor, la valeur par défaut est None.
    :param wires: Index du bit de l'effet de ligne, la valeur par défaut est None.
    :param dtype: La précision des données de la matrice interne de la porte logique peut être définie sur pyvqnet.kcomplex64 ou pyvqnet.kcomplex128, correspondant respectivement à une entrée float ou double.
    :param use_dagger: indique s'il faut utiliser la version conjuguée transposée de la porte, la valeur par défaut est False.
    :return: une instance ``pyvqnet.qnn.vqc.tn.QModule``

    Example::

        from pyvqnet.qnn.vqc.tn import IsingZZ,TNQMachine,TNQModule,MeasureAll, rx
        import pyvqnet
        pyvqnet.backends.set_backend("torch")

        class QModel(TNQModule):
            
            def __init__(self, num_wires, dtype,batch_size=2):
                super(QModel, self).__init__()
                self.device = TNQMachine(num_wires)
                self.layer = IsingZZ(has_params= True, trainable= True, wires=[0,2])
                self.batch_size = batch_size
                self.num_wires = num_wires
                
            def forward(self, x, *args, **kwargs):
                self.device.reset_states(batchsize=self.batch_size)
                for i in range(self.num_wires):
                    rx(self.device, wires=i, params=x[i])
                self.layer(q_machine = self.device)
                y = MeasureAll(obs={'Z0': 1})(self.device)
                return y

        x = pyvqnet.tensor.QTensor([[1,0,0,1],[1,1,0,1]],dtype=pyvqnet.kfloat32,requires_grad=True)
        model = QModel(4,pyvqnet.kcomplex64,2)
        y = model(x)
        print(y)


IsingXY
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^


.. py:class:: pyvqnet.qnn.vqc.tn.IsingXY(has_params: bool = False,trainable: bool = False,init_params=None,wires=None,dtype=pyvqnet.kcomplex64,use_dagger=False)
    
    définit une porte quantique IsingXY.

    Cette classe hérite de ``pyvqnet.qnn.vqc.tn.QModule`` et ``torch.nn.Module``.
    Cette classe peut être ajoutée au modèle torch comme sous-module de ``torch.nn.Module``.

    :param has_params: indique s'il a des paramètres, comme les portes RX, RY, etc. qui doivent être définies sur True, et celles sans paramètres doivent être définies sur False, la valeur par défaut est False.
    :param trainable: indique s'il a des paramètres à entraîner. Si la couche utilise des données d'entrée externes pour construire la matrice de la porte logique, définir sur False. Si les paramètres à entraîner doivent être initialisés à partir de cette couche, c'est True, la valeur par défaut est False.
    :param init_params: Paramètres d'initialisation utilisés pour encoder les données classiques QTensor, la valeur par défaut est None.
    :param wires: Index du bit de l'effet de ligne, la valeur par défaut est None.
    :param dtype: La précision des données de la matrice interne de la porte logique peut être définie sur pyvqnet.kcomplex64 ou pyvqnet.kcomplex128, correspondant respectivement à une entrée float ou double.
    :param use_dagger: indique s'il faut utiliser la version conjuguée transposée de la porte, la valeur par défaut est False.
    :return: une instance ``pyvqnet.qnn.vqc.tn.QModule``

    Example::

        from pyvqnet.qnn.vqc.tn import IsingXY,TNQMachine,TNQModule,MeasureAll, rx
        import pyvqnet
        pyvqnet.backends.set_backend("torch")

        class QModel(TNQModule):
            
            def __init__(self, num_wires, dtype,batch_size=2):
                super(QModel, self).__init__()
                self.device = TNQMachine(num_wires)
                self.layer = IsingXY(has_params= True, trainable= True, wires=[0,2])
                self.batch_size = batch_size
                self.num_wires = num_wires
                
            def forward(self, x, *args, **kwargs):
                self.device.reset_states(batchsize=self.batch_size)
                for i in range(self.num_wires):
                    rx(self.device, wires=i, params=x[i])
                self.layer(q_machine = self.device)
                y = MeasureAll(obs={'Z0': 1})(self.device)
                return y

        x = pyvqnet.tensor.QTensor([[1,0,0,1],[1,1,0,1]],dtype=pyvqnet.kfloat32,requires_grad=True)
        model = QModel(4,pyvqnet.kcomplex64,2)
        y = model(x)
        print(y)


PhaseShift
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^


.. py:class:: pyvqnet.qnn.vqc.tn.PhaseShift(has_params: bool = False,trainable: bool = False,init_params=None,wires=None,dtype=pyvqnet.kcomplex64,use_dagger=False)
    
    définit une porte quantique PhaseShift.

    Cette classe hérite de ``pyvqnet.qnn.vqc.tn.QModule`` et ``torch.nn.Module``.
    Cette classe peut être ajoutée au modèle torch comme sous-module de ``torch.nn.Module``.

    :param has_params: indique s'il a des paramètres, comme les portes RX, RY, etc. qui doivent être définies sur True, et celles sans paramètres doivent être définies sur False, la valeur par défaut est False.
    :param trainable: indique s'il a des paramètres à entraîner. Si la couche utilise des données d'entrée externes pour construire la matrice de la porte logique, définir sur False. Si les paramètres à entraîner doivent être initialisés à partir de cette couche, c'est True, la valeur par défaut est False.
    :param init_params: Paramètres d'initialisation utilisés pour encoder les données classiques QTensor, la valeur par défaut est None.
    :param wires: Index du bit de l'effet de ligne, la valeur par défaut est None.
    :param dtype: La précision des données de la matrice interne de la porte logique peut être définie sur pyvqnet.kcomplex64 ou pyvqnet.kcomplex128, correspondant respectivement à une entrée float ou double.
    :param use_dagger: indique s'il faut utiliser la version conjuguée transposée de la porte, la valeur par défaut est False.
    :return: une instance ``pyvqnet.qnn.vqc.tn.QModule``

    Example::

        from pyvqnet.qnn.vqc.tn import PhaseShift,TNQMachine,TNQModule,MeasureAll, rx
        import pyvqnet
        pyvqnet.backends.set_backend("torch")

        class QModel(TNQModule):
            
            def __init__(self, num_wires, dtype,batch_size=2):
                super(QModel, self).__init__()
                self.device = TNQMachine(num_wires)
                self.layer = PhaseShift(has_params= True, trainable= True, wires=1)
                self.batch_size = batch_size
                self.num_wires = num_wires
                
            def forward(self, x, *args, **kwargs):
                self.device.reset_states(batchsize=self.batch_size)
                for i in range(self.num_wires):
                    rx(self.device, wires=i, params=x[i])
                self.layer(q_machine = self.device)
                y = MeasureAll(obs={'Z0': 1})(self.device)
                return y

        x = pyvqnet.tensor.QTensor([[1,0,0,1],[1,1,0,1]],dtype=pyvqnet.kfloat32,requires_grad=True)
        model = QModel(4,pyvqnet.kcomplex64,2)
        y = model(x)
        print(y)


MultiRZ
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. py:class:: pyvqnet.qnn.vqc.tn.MultiRZ(has_params: bool = False,trainable: bool = False,init_params=None,wires=None,dtype=pyvqnet.kcomplex64,use_dagger=False)
    
    définit une porte quantique MultiRZ.

    Cette classe hérite de ``pyvqnet.qnn.vqc.tn.QModule`` et ``torch.nn.Module``.
    Cette classe peut être ajoutée au modèle torch comme sous-module de ``torch.nn.Module``.

    :param has_params: indique s'il a des paramètres, comme les portes RX, RY, etc. qui doivent être définies sur True, et celles sans paramètres doivent être définies sur False, la valeur par défaut est False.
    :param trainable: indique s'il a des paramètres à entraîner. Si la couche utilise des données d'entrée externes pour construire la matrice de la porte logique, définir sur False. Si les paramètres à entraîner doivent être initialisés à partir de cette couche, c'est True, la valeur par défaut est False.
    :param init_params: Paramètres d'initialisation utilisés pour encoder les données classiques QTensor, la valeur par défaut est None.
    :param wires: Index du bit de l'effet de ligne, la valeur par défaut est None.
    :param dtype: La précision des données de la matrice interne de la porte logique peut être définie sur pyvqnet.kcomplex64 ou pyvqnet.kcomplex128, correspondant respectivement à une entrée float ou double.
    :param use_dagger: indique s'il faut utiliser la version conjuguée transposée de la porte, la valeur par défaut est False.
    :return: une instance ``pyvqnet.qnn.vqc.tn.QModule``

    Example::

        from pyvqnet.qnn.vqc.tn import MultiRZ,TNQMachine,TNQModule,MeasureAll, rx
        import pyvqnet
        pyvqnet.backends.set_backend("torch")

        class QModel(TNQModule):
            
            def __init__(self, num_wires, dtype,batch_size=2):
                super(QModel, self).__init__()
                self.device = TNQMachine(num_wires)
                self.layer = MultiRZ(has_params= True, trainable= True, wires=[0,2])
                self.batch_size = batch_size
                self.num_wires = num_wires
                
            def forward(self, x, *args, **kwargs):
                self.device.reset_states(batchsize=self.batch_size)
                for i in range(self.num_wires):
                    rx(self.device, wires=i, params=x[i])
                self.layer(q_machine = self.device)
                y = MeasureAll(obs={'Z0': 1})(self.device)
                return y

        x = pyvqnet.tensor.QTensor([[1,0,0,1],[1,1,0,1]],dtype=pyvqnet.kfloat32,requires_grad=True)
        model = QModel(4,pyvqnet.kcomplex64,2)
        y = model(x)
        print(y)



SDG
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^


.. py:class:: pyvqnet.qnn.vqc.tn.SDG(has_params: bool = False,trainable: bool = False,init_params=None,wires=None,dtype=pyvqnet.kcomplex64,use_dagger=False)
    
    définit une porte quantique TDG.

    Cette classe hérite de ``pyvqnet.qnn.vqc.tn.QModule`` et ``torch.nn.Module``.
    Cette classe peut être ajoutée au modèle torch comme sous-module de ``torch.nn.Module``.

    :param has_params: indique s'il a des paramètres, comme les portes RX, RY, etc. qui doivent être définies sur True, et celles sans paramètres doivent être définies sur False, la valeur par défaut est False.
    :param trainable: indique s'il a des paramètres à entraîner. Si la couche utilise des données d'entrée externes pour construire la matrice de la porte logique, définir sur False. Si les paramètres à entraîner doivent être initialisés à partir de cette couche, c'est True, la valeur par défaut est False.
    :param init_params: Paramètres d'initialisation utilisés pour encoder les données classiques QTensor, la valeur par défaut est None.
    :param wires: Index du bit de l'effet de ligne, la valeur par défaut est None.
    :param dtype: La précision des données de la matrice interne de la porte logique peut être définie sur pyvqnet.kcomplex64 ou pyvqnet.kcomplex128, correspondant respectivement à une entrée float ou double.
    :param use_dagger: indique s'il faut utiliser la version conjuguée transposée de la porte, la valeur par défaut est False.
    :return: une instance ``pyvqnet.qnn.vqc.tn.QModule``

    Example::
        
        from pyvqnet.qnn.vqc.tn import SDG,TNQMachine,TNQModule,MeasureAll, rx
        import pyvqnet
        pyvqnet.backends.set_backend("torch")

        class QModel(TNQModule):
            
            def __init__(self, num_wires, dtype,batch_size=2):
                super(QModel, self).__init__()
                self.device = TNQMachine(num_wires)
                self.layer = SDG(wires=0)
                self.batch_size = batch_size
                self.num_wires = num_wires
                
            def forward(self, x, *args, **kwargs):
                self.device.reset_states(batchsize=self.batch_size)
                for i in range(self.num_wires):
                    rx(self.device, wires=i, params=x[i])
                self.layer(q_machine = self.device)
                y = MeasureAll(obs={'Z0': 1})(self.device)
                return y

        x = pyvqnet.tensor.QTensor([[1,0,0,1],[1,1,0,1]],dtype=pyvqnet.kfloat32,requires_grad=True)
        model = QModel(4,pyvqnet.kcomplex64,2)
        y = model(x)
        print(y)




TDG
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. py:class:: pyvqnet.qnn.vqc.tn.TDG(has_params: bool = False,trainable: bool = False,init_params=None,wires=None,dtype=pyvqnet.kcomplex64,use_dagger=False)
    
    définit une porte quantique TDG.

    Cette classe hérite de ``pyvqnet.qnn.vqc.tn.QModule`` et ``torch.nn.Module``.
    Cette classe peut être ajoutée au modèle torch comme sous-module de ``torch.nn.Module``.

    :param has_params: indique s'il a des paramètres, comme les portes RX, RY, etc. qui doivent être définies sur True, et celles sans paramètres doivent être définies sur False, la valeur par défaut est False.
    :param trainable: indique s'il a des paramètres à entraîner. Si la couche utilise des données d'entrée externes pour construire la matrice de la porte logique, définir sur False. Si les paramètres à entraîner doivent être initialisés à partir de cette couche, c'est True, la valeur par défaut est False.
    :param init_params: Paramètres d'initialisation utilisés pour encoder les données classiques QTensor, la valeur par défaut est None.
    :param wires: Index du bit de l'effet de ligne, la valeur par défaut est None.
    :param dtype: La précision des données de la matrice interne de la porte logique peut être définie sur pyvqnet.kcomplex64 ou pyvqnet.kcomplex128, correspondant respectivement à une entrée float ou double.
    :param use_dagger: indique s'il faut utiliser la version conjuguée transposée de la porte, la valeur par défaut est False.
    :return: une instance ``pyvqnet.qnn.vqc.tn.QModule``

    Example::
        
        from pyvqnet.qnn.vqc.tn import TDG,TNQMachine,TNQModule,MeasureAll, rx
        import pyvqnet
        pyvqnet.backends.set_backend("torch")

        class QModel(TNQModule):
            
            def __init__(self, num_wires, dtype,batch_size=2):
                super(QModel, self).__init__()
                self.device = TNQMachine(num_wires)
                self.layer = TDG(wires=0)
                self.batch_size = batch_size
                self.num_wires = num_wires
                
            def forward(self, x, *args, **kwargs):
                self.device.reset_states(batchsize=self.batch_size)
                for i in range(self.num_wires):
                    rx(self.device, wires=i, params=x[i])
                self.layer(q_machine = self.device)
                y = MeasureAll(obs={'Z0': 1})(self.device)
                return y

        x = pyvqnet.tensor.QTensor([[1,0,0,1],[1,1,0,1]],dtype=pyvqnet.kfloat32,requires_grad=True)
        model = QModel(4,pyvqnet.kcomplex64,2)
        y = model(x)
        print(y)



ControlledPhaseShift
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^


.. py:class:: pyvqnet.qnn.vqc.tn.ControlledPhaseShift(has_params: bool = False,trainable: bool = False,init_params=None,wires=None,dtype=pyvqnet.kcomplex64,use_dagger=False)
    
    définit une porte quantique ControlledPhaseShift.

    Cette classe hérite de ``pyvqnet.qnn.vqc.tn.QModule`` et ``torch.nn.Module``.
    Cette classe peut être ajoutée au modèle torch comme sous-module de ``torch.nn.Module``.

    :param has_params: indique s'il a des paramètres, comme les portes RX, RY, etc. qui doivent être définies sur True, et celles sans paramètres doivent être définies sur False, la valeur par défaut est False.
    :param trainable: indique s'il a des paramètres à entraîner. Si la couche utilise des données d'entrée externes pour construire la matrice de la porte logique, définir sur False. Si les paramètres à entraîner doivent être initialisés à partir de cette couche, c'est True, la valeur par défaut est False.
    :param init_params: Paramètres d'initialisation utilisés pour encoder les données classiques QTensor, la valeur par défaut est None.
    :param wires: Index du bit de l'effet de ligne, la valeur par défaut est None.
    :param dtype: La précision des données de la matrice interne de la porte logique peut être définie sur pyvqnet.kcomplex64 ou pyvqnet.kcomplex128, correspondant respectivement à une entrée float ou double.
    :param use_dagger: indique s'il faut utiliser la version conjuguée transposée de la porte, la valeur par défaut est False.
    :return: une instance ``pyvqnet.qnn.vqc.tn.QModule``

    Example::

        from pyvqnet.qnn.vqc.tn import ControlledPhaseShift,TNQMachine,TNQModule,MeasureAll, rx
        import pyvqnet
        pyvqnet.backends.set_backend("torch")

        class QModel(TNQModule):
            
            def __init__(self, num_wires, dtype,batch_size=2):
                super(QModel, self).__init__()
                self.device = TNQMachine(num_wires)
                self.layer = ControlledPhaseShift(has_params= True, trainable= True, wires=[0,2])
                self.batch_size = batch_size
                self.num_wires = num_wires
                
            def forward(self, x, *args, **kwargs):
                self.device.reset_states(batchsize=self.batch_size)
                for i in range(self.num_wires):
                    rx(self.device, wires=i, params=x[i])
                self.layer(q_machine = self.device)
                y = MeasureAll(obs={'Z0': 1})(self.device)
                return y

        x = pyvqnet.tensor.QTensor([[1,0,0,1],[1,1,0,1]],dtype=pyvqnet.kfloat32,requires_grad=True)
        model = QModel(4,pyvqnet.kcomplex64,2)
        y = model(x)
        print(y)


API de mesures
------------------------------

VQC_Purity
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^


.. py:function:: pyvqnet.qnn.vqc.tn.VQC_Purity(state, qubits_idx, num_wires, use_tn=False)

    Calcule la pureté sur un qubit particulier ``qubits_idx`` à partir du vecteur d'état ``state``.

    .. math::
        \gamma = \text{Tr}(\rho^2)

    where :math:`\rho` is a density matrix. The pureté of a normalized quantum state satisfies :math:`\frac{1}{d} \leq \gamma \leq 1` ,
    where :math:`d` is the dimension of the Hilbert space.
    La pureté de l'état pur est 1.

    :param state: État quantique obtenu à partir de TNQMachine.get_states()
    :param qubits_idx: Indice du qubit pour lequel calculer la pureté
    :param num_wires: Idx du qubit
    :param use_tn: utiliser tensornetwork doit être défini sur True, défaut False

    :return: purity

    .. note::
        
        batch_size nécessite TNQModule.

    Example::

        import pyvqnet
        from pyvqnet.qnn.vqc.tn import TNQMachine, qcircuit, TNQModule,VQC_Purity
        pyvqnet.backends.set_backend("torch")
        from pyvqnet.tensor import QTensor

        x = QTensor([[0.7, 0.4], [1.7, 2.4]], requires_grad=True).toGPU()

        class QM(TNQModule):
            def __init__(self, name=""):
                super().__init__(name)
                self.device = TNQMachine(3)
                
            def forward(self, x):
                self.device.reset_states(2)
                qcircuit.rx(q_machine=self.device, wires=0, params=x[0])
                qcircuit.ry(q_machine=self.device, wires=1, params=x[1])
                qcircuit.ry(q_machine=self.device, wires=2, params=x[1])
                qcircuit.cnot(q_machine=self.device, wires=[0, 1])
                qcircuit.cnot(q_machine=self.device, wires=[2, 1])
                return VQC_Purity(self.device.get_states(), [0, 1], num_wires=3, use_tn=True)

        model = QM().toGPU()
        y_tn = model(x)
        x.data.retain_grad()
        y_tn.backward()
        print(y_tn)

VQC_VarMeasure
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. py:function:: pyvqnet.qnn.vqc.tn.VQC_VarMeasure(q_machine, obs)

    Retourne la variance de mesure de l'observable fournie ``obs`` dans les vecteurs d'état de ``q_machine``.

    :param q_machine: État quantique obtenu à partir de pyqpanda get_qstate()
    :param obs: observables

    :return: valeur de variance

    Example::

        import pyvqnet
        from pyvqnet.qnn.vqc.tn import TNQMachine, qcircuit, VQC_VarMeasure, TNQModule,PauliY
        from pyvqnet.tensor import QTensor
        from pyvqnet import kfloat64
        pyvqnet.backends.set_backend("torch")
        x = QTensor([[0.7, 0.4], [0.6, 0.4]], requires_grad=True).toGPU()

        class QM(TNQModule):
            def __init__(self, name=""):
                super().__init__(name)
                self.device = TNQMachine(3)
                
            def forward(self, x):
                self.device.reset_states(2)
                qcircuit.rx(q_machine=self.device, wires=0, params=x[0])
                qcircuit.ry(q_machine=self.device, wires=1, params=x[1])
                qcircuit.ry(q_machine=self.device, wires=2, params=x[1])
                qcircuit.cnot(q_machine=self.device, wires=[0, 1])
                qcircuit.cnot(q_machine=self.device, wires=[2, 1])
                return VQC_VarMeasure(q_machine= self.device, obs=PauliY(wires=0))
            
        model = QM().toGPU()
        y = model(x)
        x.data.retain_grad()
        y.backward()
        print(y)

        # [[0.9370641],
        # [0.9516521]]


VQC_DensityMatrixFromQstate
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^


.. py:function:: pyvqnet.qnn.vqc.tn.VQC_DensityMatrixFromQstate(state, indices, use_tn=False)

    Calcule la matrice de densité des états quantiques ``state`` sur un ensemble spécifique de qubits ``indices``.

    :param state: Une liste 1D de vecteurs d'état. La taille de cette liste doit être ``(2**N,)`` Pour le nombre de qubits ``N``, qstate doit commencer de 000 -> 111.
    :param indices: Une liste d'indices de qubits dans le sous-système considéré.
    :param use_tn: utiliser tensornetwork doit être défini sur True, défaut False.

    :return: Une matrice de densité de taille "(b, 2**len(indices), 2**len(indices))".

    Example::

        import pyvqnet
        from pyvqnet.tensor import QTensor
        from pyvqnet.qnn.vqc.tn import TNQMachine, qcircuit, VQC_DensityMatrixFromQstate,TNQModule
        pyvqnet.backends.set_backend("torch")
        x = QTensor([[0.7,0.4],[1.7,2.4]], requires_grad=True).toGPU()
        class QM(TNQModule):
            def __init__(self, name=""):
                super().__init__(name=name, use_jit=True)
                self.device = TNQMachine(3)
                
            def forward(self, x):
                self.device.reset_states(2)
                qcircuit.rx(q_machine=self.device, wires=0, params=x[0])
                qcircuit.ry(q_machine=self.device, wires=1, params=x[1])
                qcircuit.ry(q_machine=self.device, wires=2, params=x[1])
                qcircuit.cnot(q_machine=self.device, wires=[0, 1])
                qcircuit.cnot(q_machine=self.device, wires=[2, 1])
                return VQC_DensityMatrixFromQstate(self.device.get_states(),[0,1],use_tn=True)
            
        model = QM().toGPU()
        y = model(x)
        x.data.retain_grad()
        y.backward()
        print(y)

        # [[[0.8155131+0.j        0.1718155+0.j        0.       +0.0627175j
        #   0.       +0.2976855j]
        #  [0.1718155+0.j        0.0669081+0.j        0.       +0.0244234j
        #   0.       +0.0627175j]
        #  [0.       -0.0627175j 0.       -0.0244234j 0.0089152+0.j
        #   0.0228937+0.j       ]
        #  [0.       -0.2976855j 0.       -0.0627175j 0.0228937+0.j
        #   0.1086637+0.j       ]]
        # 
        # [[0.3362115+0.j        0.1471083+0.j        0.       +0.1674582j
        #   0.       +0.3827205j]
        #  [0.1471083+0.j        0.0993662+0.j        0.       +0.1131119j
        #   0.       +0.1674582j]
        #  [0.       -0.1674582j 0.       -0.1131119j 0.1287589+0.j
        #   0.1906232+0.j       ]
        #  [0.       -0.3827205j 0.       -0.1674582j 0.1906232+0.j
        #   0.4356633+0.j       ]]]   


Probability
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^


.. py:class:: pyvqnet.qnn.vqc.tn.Probability(wires=None, name="")

    Calculate the probability measurement result of the quantum circuit on a specific bit.

    Cette classe hérite de ``pyvqnet.qnn.vqc.tn.QModule`` et ``torch.nn.Module``.
    Cette classe peut être ajoutée au modèle torch comme sous-module de ``torch.nn.Module``.

    :param wires: The index of the measurement bit, list, tuple or integer.
    :param name: The name of the module, default: "".
    :return: The measurement result, QTensor.

    Example::

        import pyvqnet
        pyvqnet.backends.set_backend("torch")
        from pyvqnet.qnn.vqc.tn import Probability,rx,ry,cnot,TNQMachine,rz
        from pyvqnet.tensor import QTensor
        from pyvqnet import kfloat64
        x = QTensor([[0.56, 0.1],[0.56, 0.1]],requires_grad=True)
        qm = TNQMachine(4)
        qm.reset_states(2)
        rz(q_machine=qm,wires=0,params=x[:,[0]])
        rz(q_machine=qm,wires=1,params=x[:,[0]])
        cnot(q_machine=qm,wires=[0,1])
        ry(q_machine=qm,wires=2,params=x[:,[1]])
        cnot(q_machine=qm,wires=[0,2])
        rz(q_machine=qm,wires=3,params=x[:,[1]])
        ma = Probability(wires=1)
        y =ma(q_machine=qm)


MeasureAll
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. py:class:: pyvqnet.qnn.vqc.tn.MeasureAll(obs=None, name="")

    Calculate the measurement results of quantum circuits, support input obs as multiple or single Pauli operators or Hamiltonians.
    
    For example:

    {\'X0\': 0.23} indicates a PauliX effect on qubit 0, with a coefficient of 0.23.

    {\'X1 Z2\': 2.4,\'Y2\': -0.5} corresponds to the observed value 2.4 * X1 @ Z2 - 0.5 * Y2.

    [{\'X1 Z2\': 4,\'Z1 Z0\': 3},{\'X1 Y2 Z0\': 3.5}] corresponds to the two observed values 4 * X1 @ Z2 + 3 * Z1 @ Z0 and 3.5 * X1 @ Y2 @ Z0.
        
    Cette classe hérite de ``pyvqnet.qnn.vqc.tn.QModule`` et ``torch.nn.Module``.

    Cette classe peut être ajoutée au modèle torch comme sous-module de ``torch.nn.Module``.

    :param obs: observable.
    :param name: module name, default: "".
    :return: measurement result, QTensor.

    Example::

        import pyvqnet
        pyvqnet.backends.set_backend("torch")
        from pyvqnet.qnn.vqc.tn import MeasureAll,rx,ry,cnot,TNQMachine,rz
        from pyvqnet.tensor import QTensor
        from pyvqnet import kfloat64
        x = QTensor([[0.56, 0.1],[0.56, 0.1]],requires_grad=True)
        qm = TNQMachine(4)
        qm.reset_states(2)
        rz(q_machine=qm,wires=0,params=x[:,[0]])
        rz(q_machine=qm,wires=1,params=x[:,[0]])
        cnot(q_machine=qm,wires=[0,1])
        ry(q_machine=qm,wires=2,params=x[:,[1]])
        cnot(q_machine=qm,wires=[0,2])
        rz(q_machine=qm,wires=3,params=x[:,[1]])
        obs_list = [{
            "Z0 Z1" :2
        }, {
            "X0 Z2" :1
        }]
        ma = MeasureAll(obs = obs_list)
        y = ma(q_machine=qm)
        print(y)



Samples
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. py:class:: pyvqnet.qnn.vqc.tn.Samples(wires=None, obs=None, shots = 1,name="")

    Get sample results with shot on  specific wires.

    Cette classe hérite de ``pyvqnet.qnn.vqc.tn.QModule`` et ``torch.nn.Module``.
    Cette classe peut être ajoutée au modèle torch comme sous-module de ``torch.nn.Module``.


    :param wires: Sample qubit index. Default value: None, use all bits of the simulator at runtime.
    :param obs: This value can only be None.
    :param shots: Sample repetition count, default value: 1.
    :param name: The name of this module, default value: "".
    :return: a measurement method class

    Example::

        import pyvqnet
        pyvqnet.backends.set_backend("torch")
        from pyvqnet.qnn.vqc.tn import Samples,rx,ry,cnot,TNQMachine,rz
        from pyvqnet.tensor import QTensor
        from pyvqnet import kfloat64
        x = QTensor([[0.56, 0.1],[0.56, 0.1]],requires_grad=True)

        qm = TNQMachine(4)
        qm.reset_states(2)
        rz(q_machine=qm,wires=0,params=x[:,[0]])
        rx(q_machine=qm,wires=1,params=x[:,[0]])
        cnot(q_machine=qm,wires=[0,1])

        cnot(q_machine=qm,wires=[0,2])
        ry(q_machine=qm,wires=3,params=x[:,[1]])


        ma = Samples(wires=[0,1,2],shots=3)
        y = ma(q_machine=qm)
        print(y)


HermitianExpval
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. py:class:: pyvqnet.qnn.vqc.tn.HermitianExpval(obs=None, name="")

    Compute the expectation of a Hermitian quantity in a quantum circuit.

    Cette classe hérite de ``pyvqnet.qnn.vqc.tn.QModule`` et ``torch.nn.Module``.
    Cette classe peut être ajoutée au modèle torch comme sous-module de ``torch.nn.Module``.


    :param obs: Hermitian quantity.
    :param name: module name, default: "".
    :return: expected result, QTensor.

    Example::

        import pyvqnet
        pyvqnet.backends.set_backend("torch")
        from pyvqnet.qnn.vqc.tn import TNQMachine, rx,ry,\
            RX, RY, CNOT, PauliX, PauliZ, VQC_RotCircuit,HermitianExpval, TNQModule
        from pyvqnet.tensor import QTensor, tensor
        from pyvqnet.nn import Parameter
        import numpy as np
        bsz = 3
        H = np.array([[8, 4, 0, -6], [4, 0, 4, 0], [0, 4, 8, 0], [-6, 0, 0, 0]])
        class QModel(TNQModule):
            def __init__(self, num_wires, dtype):
                super(QModel, self).__init__()
                self.rot_param = Parameter((3, ))
                self._num_wires = num_wires
                self._dtype = dtype
                self.qm = TNQMachine(num_wires, dtype=dtype)
                self.rx_layer1 = VQC_RotCircuit
                self.ry_layer2 = RY(has_params=True,
                                    trainable=True,
                                    wires=0,
                                    init_params=tensor.QTensor([-0.5]))
                self.xlayer = PauliX(wires=0)
                self.cnot = CNOT(wires=[0, 1])
                self.measure = HermitianExpval(obs = {'wires':(1,0),'observables':tensor.to_tensor(H)})

            def forward(self, x, *args, **kwargs):
                self.qm.reset_states(bsz)

                rx(q_machine=self.qm, wires=0, params=x[1])
                ry(q_machine=self.qm, wires=1, params=x[0])
                self.xlayer(q_machine=self.qm)
                self.rx_layer1(params=self.rot_param, wire=1, q_machine=self.qm)
                self.ry_layer2(q_machine=self.qm)
                self.cnot(q_machine=self.qm)
                rlt = self.measure(q_machine = self.qm)

                return rlt


        input_x = tensor.arange(1, bsz * 2 + 1,
                                dtype=pyvqnet.kfloat32).reshape([bsz, 2])
        input_x.requires_grad = True

        qunatum_model = QModel(num_wires=2, dtype=pyvqnet.kcomplex64)

        batch_y = qunatum_model(input_x)
        batch_y.backward()

        print(batch_y)

Gabarits courants pour les circuits quantiques
-----------------------------------------------

VQC_HardwareEfficientAnsatz
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^


.. py:class:: pyvqnet.qnn.vqc.tn.VQC_HardwareEfficientAnsatz(n_qubits,single_rot_gate_list,entangle_gate="CNOT",entangle_rules='linear',depth=1,initial=None,dtype=None)

    Implementation of Hardware Efficient Ansatz introduced in the paper: `Hardware-efficient Variational Quantum Eigensolver for Small Molecules <https://arxiv.org/pdf/1704.05018.pdf>`__ .

    Cette classe hérite de ``pyvqnet.qnn.vqc.tn.QModule`` et ``torch.nn.Module``.

    Cette classe peut être ajoutée au modèle torch comme sous-module de ``torch.nn.Module``.


    :param n_qubits: Number of qubits.
    :param single_rot_gate_list: A single qubit rotation gate list is constructed by one or several rotation gate that act on every qubit.Currently support Rx, Ry, Rz.
    :param entangle_gate: The non parameterized entanglement gate.CNOT,CZ is supported.default:CNOT.
    :param entangle_rules: How entanglement gate is used in the circuit. 'linear' means the entanglement gate will be act on every neighboring qubits. 'all' means the entanglment gate will be act on any two qbuits. Default:linear.
    :param depth: The depth of ansatz, default:1.
    :param initial: initial one same value for paramaters,default:None,this module will initialize parameters randomly.
    :param dtype: data dtype of parameters.
    :return: a VQC_HardwareEfficientAnsatz instance.

    Example::

        from pyvqnet.nn.torch import Linear
        from pyvqnet.qnn.vqc.tn.qcircuit import VQC_HardwareEfficientAnsatz,RZZ,RZ
        from pyvqnet.qnn.vqc.tn import Probability,TNQMachine, TNQModule
        from pyvqnet import tensor
        import pyvqnet
        pyvqnet.backends.set_backend("torch")
        pyvqnet.utils.set_random_seed(25)

        class QM(TNQModule):
            def __init__(self, name=""):
                super().__init__(name)
                self.linearx = Linear(4,2)
                self.ansatz = VQC_HardwareEfficientAnsatz(4, ["rx", "RY", "rz"],
                                            entangle_gate="cnot",
                                            entangle_rules="linear",
                                            depth=2)
                self.encode1 = RZ(wires=0)
                self.encode2 = RZ(wires=1)
                self.measure = Probability(wires=[0, 2])
                self.device = TNQMachine(4)
            def forward(self, x, *args, **kwargs):
                self.device.reset_states(bz)
                y = self.linearx(x)
                self.encode1(params = y[0],q_machine = self.device,)
                self.encode2(params = y[1],q_machine = self.device,)
                self.ansatz(q_machine =self.device)
                return self.measure(q_machine =self.device)

        bz =3
        inputx = tensor.arange(1.0,bz*4+1).reshape([bz,4])
        inputx.requires_grad= True
        qlayer = QM()
        y = qlayer(inputx)
        y.backward()
        print(y)



VQC_BasicEntanglerTemplate
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. py:class:: pyvqnet.qnn.vqc.tn.VQC_BasicEntanglerTemplate(num_layer=1, num_qubits=1, rotation="RX", initial=None, dtype=None)

    A layer consisting of a single-parameter single-qubit rotation on each qubit, followed by multiple CNOT gates in a closed chain or ring combination.

    A ring of CNOT gates connects each qubit to its neighbors, and finally the a qubit is considered to be the neighbor of the a th qubit.

    Cette classe hérite de ``pyvqnet.qnn.vqc.tn.QModule`` et ``torch.nn.Module``.

    Cette classe peut être ajoutée au modèle torch comme sous-module de ``torch.nn.Module``.

    :param num_layers: number of repeat layers, default: 1.
    :param num_qubits: number of qubits, default: 1.
    :param rotation: one-parameter single-qubit gate to use, default: `RX`
    :param initial: initialized same value for all paramters. default:None,parameters will be initialized randomly.
    :param dtype: data type of parameter, default:None,use float32.
    :return: A VQC_BasicEntanglerTemplate instance

    Example::

        import pyvqnet
        pyvqnet.backends.set_backend("torch")
        from pyvqnet.qnn.vqc.tn import TNQModule,\
            VQC_BasicEntanglerTemplate, Probability, TNQMachine
        from pyvqnet import tensor


        class QM(TNQModule):
            def __init__(self, name=""):
                super().__init__(name)

                self.ansatz = VQC_BasicEntanglerTemplate(2,
                                                    4,
                                                    "rz",
                                                    initial=tensor.ones([1, 1]))

                self.measure = Probability(wires=[0, 2])
                self.device = TNQMachine(4)

            def forward(self,x, *args, **kwargs):

                self.ansatz(q_machine=self.device)
                return self.measure(q_machine=self.device)

        bz = 1
        inputx = tensor.arange(1.0, bz * 4 + 1).reshape([bz, 4])
        qlayer = QM()
        y = qlayer(inputx)
        y.backward()
        print(y)



VQC_StronglyEntanglingTemplate
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. py:class:: pyvqnet.qnn.vqc.tn.VQC_StronglyEntanglingTemplate(num_layers=1, num_qubits=1, ranges=None,initial=None, dtype=None)

    Layers consisting of single qubit rotations and entanglers, as in `circuit-centric classifier design <https://arxiv.org/abs/1804.00633>`__ .

    Cette classe hérite de ``pyvqnet.qnn.vqc.tn.QModule`` et ``torch.nn.Module``.
    Cette classe peut être ajoutée au modèle torch comme sous-module de ``torch.nn.Module``.


    :param num_layers: number of repeat layers, default: 1.
    :param num_qubits: number of qubits, default: 1.
    :param ranges: sequence determining the range hyperparameter for each subsequent layer; default: None
                                using :math: `r=l \mod M` for the :math:`l` th layer and :math:`M` qubits.
    :param initial: initial value for all parameters.default: None,initialized randomly.
    :param dtype: data type of parameter, default:None,use float32.
    :return: A VQC_StronglyEntanglingTemplate instance.

    Example::

        from pyvqnet.nn.torch import TorchModule,Linear,TorchModuleList
        from pyvqnet.qnn.vqc.tn.qcircuit import VQC_StronglyEntanglingTemplate
        from pyvqnet.qnn.vqc.tn import Probability, TNQMachine, TNQModule
        from pyvqnet import tensor
        import pyvqnet

        pyvqnet.backends.set_backend("torch")
        pyvqnet.utils.set_random_seed(25)
        class QM(TNQModule):
            def __init__(self, name=""):
                super().__init__(name)

                self.ansatz = VQC_StronglyEntanglingTemplate(2,
                                                    4,
                                                    None,
                                                    initial=tensor.ones([1, 1]))

                self.measure = Probability(wires=[0, 1])
                self.device = TNQMachine(4)

            def forward(self,x, *args, **kwargs):

                self.ansatz(q_machine=self.device)
                return self.measure(q_machine=self.device)

        bz = 1
        inputx = tensor.arange(1.0, bz * 4 + 1).reshape([bz, 4])
        qlayer = QM()
        y = qlayer(inputx)
        y.backward()
        print(y)



VQC_QuantumEmbedding
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^


.. py:class:: pyvqnet.qnn.vqc.tn.VQC_QuantumEmbedding(qubits, machine, num_repetitions_input, depth_input, num_unitary_layers, num_repetitions,initial = None,dtype = None,name= "")
    
    Use RZ,RY,RZ to create variational quantum circuits to encode classical data into quantum states.
    Reference `Quantum embeddings for machine learning <https://arxiv.org/abs/2001.03622>`_.

    Cette classe hérite de ``pyvqnet.qnn.vqc.tn.QModule`` et ``torch.nn.Module``.
    Cette classe peut être ajoutée au modèle torch comme sous-module de ``torch.nn.Module``.

 
    :param num_repetitions_input: number of repeat times to encode input in a submodule.
    :param depth_input: number of input dimension .
    :param num_unitary_layers: number of repeat times of variational quantum gates.
    :param num_repetitions: number of repeat times of submodule.
    :param initial: parameter initialization value, default is None
    :param dtype: parameter type, default is None, use float32.
    :param name: class name
    :return: A VQC_QuantumEmbedding instance.

    Example::

        from pyvqnet.qnn.vqc.tn.qcircuit import VQC_QuantumEmbedding
        from pyvqnet.qnn.vqc.tn import TNQMachine, MeasureAll, TNQModule
        from pyvqnet import tensor
        import pyvqnet

        pyvqnet.backends.set_backend("torch")
        pyvqnet.utils.set_random_seed(25)
        depth_input = 2
        num_repetitions = 2
        num_repetitions_input = 2
        num_unitary_layers = 2
        nq = depth_input * num_repetitions_input
        bz = 12

        class QM(TNQModule):
            def __init__(self, name=""):
                super().__init__(name)

                self.ansatz = VQC_QuantumEmbedding(num_repetitions_input, depth_input,
                                                num_unitary_layers,
                                                num_repetitions, initial=tensor.full([1],12.0),dtype=pyvqnet.kfloat32)

                self.measure = MeasureAll(obs={f"Z{nq-1}":1})
                self.device = TNQMachine(nq)

            def forward(self, x, *args, **kwargs):
                self.device.reset_states(bz)
                self.ansatz(x,q_machine=self.device)
                return self.measure(q_machine=self.device)

        inputx = tensor.arange(1.0, bz * depth_input + 1).reshape([bz, depth_input])
        qlayer = QM()
        y = qlayer(inputx)
        y.backward()
        print(y)


ExpressiveEntanglingAnsatz
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. py:class:: pyvqnet.qnn.vqc.tn.ExpressiveEntanglingAnsatz(type: int, num_wires: int, depth: int, dtype=None, name: str = "")

    19 different ansatz from the paper `Expressibility and entangling capability of parameterized quantum circuits for hybrid quantum-classical algorithms <https://arxiv.org/pdf/1905.10876.pdf>`_.

    Cette classe hérite de ``pyvqnet.qnn.vqc.tn.QModule`` et ``torch.nn.Module``.

    Cette classe peut être ajoutée au modèle torch comme sous-module de ``torch.nn.Module``.

    :param type: Circuit type from 1 to 19, a total of 19 lines.
    :param num_wires: Number of qubits.
    :param depth: Circuit depth.
    :param dtype: data type of parameter, default:None,use float32.
    :param name: Name, default "".

    :return:
        a ExpressiveEntanglingAnsatz instance

    Example::

        from pyvqnet.qnn.vqc.tn.qcircuit import ExpressiveEntanglingAnsatz
        from pyvqnet.qnn.vqc.tn import Probability, TNQMachine, MeasureAll, TNQModule
        from pyvqnet import tensor
        import pyvqnet

        pyvqnet.backends.set_backend("torch")
        pyvqnet.utils.set_random_seed(25)

        class QModel(TNQModule):
            def __init__(self, num_wires, dtype):
                super(QModel, self).__init__()

                self._num_wires = num_wires
                self._dtype = dtype
                self.qm = TNQMachine(num_wires, dtype=dtype)
                self.c1 = ExpressiveEntanglingAnsatz(1,3,2)
                self.measure = MeasureAll(obs={
                    "Z1":1
                })

            def forward(self, x, *args, **kwargs):
                self.qm.reset_states(1)
                self.c1(q_machine = self.qm)
                rlt = self.measure(q_machine=self.qm)
                return rlt
            

        input_x = tensor.QTensor([[0.1, 0.2, 0.3]])

        qunatum_model = QModel(num_wires=3, dtype=pyvqnet.kcomplex64)

        batch_y = qunatum_model(input_x)
        batch_y.backward()
        print(batch_y)


vqc_basis_embedding
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. py:function:: pyvqnet.qnn.vqc.tn.vqc_basis_embedding(basis_state,q_machine)

    Encode n binary features into the n-qubit basis state of ``q_machine``. This function is aliased as `VQC_BasisEmbedding`.

    For example, for ``basis_state=([0, 1, 1])``, the basis state in the quantum system is :math:`|011 \rangle`.

    :param basis_state: ``(n)`` size binary input.
    :param q_machine: quantum machine device.
    

    Example::

        import pyvqnet
        pyvqnet.backends.set_backend("torch")
        from pyvqnet.qnn.vqc.tn import vqc_basis_embedding,TNQMachine
        qm  = TNQMachine(3)
        vqc_basis_embedding(basis_state=[1,1,0],q_machine=qm)
        print(qm.get_states())




vqc_angle_embedding
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^


.. py:function:: pyvqnet.qnn.vqc.tn.vqc_angle_embedding(input_feat, wires, q_machine: pyvqnet.qnn.vqc.tn.TNQMachine, rotation: str = "X")

    Encodes :math:`N` features into the rotation angle of :math:`n` qubits, where :math:`N \leq n`.
    This function is aliased as `VQC_AngleEmbedding` .

    La rotation peut être sélectionnée comme : 'X', 'Y', 'Z', comme défini par le paramètre ``rotation`` :

    * ``rotation='X'`` Use the feature as the angle of RX rotation.

    * ``rotation='Y'`` Use the feature as the angle of RY rotation.

    * ``rotation='Z'`` Use the feature as the angle of RZ rotation.

    ``wires`` represents the idx of the rotation gate on the qubit.

    :param input_feat: Array representing parameters.
    :param wires: Qubit idx.
    :param q_machine: Quantum machine device.
    :param rotation: Rotation gate, default is "X".

    Example::

        import pyvqnet
        pyvqnet.backends.set_backend("torch")
        from pyvqnet.qnn.vqc.tn import vqc_angle_embedding, TNQMachine
        from pyvqnet.tensor import QTensor
        qm  = TNQMachine(2)
        vqc_angle_embedding(QTensor([2.2, 1]), [0, 1], q_machine=qm, rotation='X')
        print(qm.get_states())
        vqc_angle_embedding(QTensor([2.2, 1]), [0, 1], q_machine=qm, rotation='Y')
        print(qm.get_states())
        vqc_angle_embedding(QTensor([2.2, 1]), [0, 1], q_machine=qm, rotation='Z')
        print(qm.get_states())



vqc_amplitude_embedding
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. py:function:: pyvqnet.qnn.vqc.tn.vqc_amplitude_embedding(input_feature, q_machine)

    Encodes a :math:`2^n` feature into an amplitude vector of :math:`n` qubits. This function is aliased as `VQC_AmplitudeEmbedding`.

    :param input_feature: numpy array representing the parameter.
    :param q_machine: quantum machine device.
    

    Example::

        import pyvqnet
        pyvqnet.backends.set_backend("torch")
        from pyvqnet.qnn.vqc.tn import vqc_amplitude_embedding, TNQMachine
        from pyvqnet.tensor import QTensor
        qm  = TNQMachine(3)
        vqc_amplitude_embedding(QTensor([3.2,-2,-2,0.3,12,0.1,2,-1]), q_machine=qm)
        print(qm.get_states())



vqc_iqp_embedding
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. py:function:: pyvqnet.qnn.vqc.tn.vqc_iqp_embedding(input_feat, q_machine: pyvqnet.qnn.vqc.tn.TNQMachine, rep: int = 1)

    Encode :math:`n` features into :math:`n` qubits using diagonal gates of an IQP circuit. Alias: ``VQC_IQPEmbedding`` .

    L'encodage est proposé par `Havlicek et al. (2018) <https://arxiv.org/pdf/1804.11326.pdf>`_.

    By specifying ``rep`` , the basic IQP circuit can be repeated.

    :param input_feat: Array of parameters.
    :param q_machine: Quantum machine machine.
    :param rep: Number of times to repeat the quantum circuit block, default is 1.

    Example::

        import pyvqnet
        pyvqnet.backends.set_backend("torch")
        from pyvqnet.qnn.vqc.tn import vqc_iqp_embedding, TNQMachine
        from pyvqnet.tensor import QTensor
        qm  = TNQMachine(3)
        vqc_iqp_embedding(QTensor([3.2,-2,-2]), q_machine=qm)
        print(qm.get_states())        



vqc_rotcircuit
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. py:function:: pyvqnet.qnn.vqc.tn.vqc_rotcircuit(q_machine, wire, params)

    Arbitrary single quantum bit rotation quantum logic gate combination. This function alias: ``VQC_RotCircuit`` .

    .. math::
        R(\phi,\theta,\omega) = RZ(\omega)RY(\theta)RZ(\phi)= \begin{bmatrix}
        e^{-i(\phi+\omega)/2}\cos(\theta/2) & -e^{i(\phi-\omega)/2}\sin(\theta/2) \\
        e^{-i(\phi-\omega)/2}\sin(\theta/2) & e^{i(\phi+\omega)/2}\cos(\theta/2)
        \end{bmatrix}.

    :param q_machine: quantum virtual machine device.
    :param wire: quantum bit index.
    :param params: represents parameters :math:`[\phi, \theta, \omega]`.

    Example::

        import pyvqnet
        pyvqnet.backends.set_backend("torch")
        from pyvqnet.qnn.vqc.tn import vqc_rotcircuit, TNQMachine
        from pyvqnet.tensor import QTensor
        qm  = TNQMachine(3)
        vqc_rotcircuit(q_machine=qm, wire=[1],params=QTensor([2.0,1.5,2.1]))
        print(qm.get_states())


vqc_crot_circuit
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^


.. py:function:: pyvqnet.qnn.vqc.tn.vqc_crot_circuit(para,control_qubits,rot_wire,q_machine)

    Quantum logic gate combination of controlled Rot single quantum bit rotation. This function alias: ``VQC_CRotCircuit`` .

    .. math:: 
        CR(\phi, \theta, \omega) = \begin{bmatrix}
        1 & 0 & 0 & 0 \\
        0 & 1 & 0 & 0\\
        0 & 0 & e^{-i(\phi+\omega)/2}\cos(\theta/2) & -e^{i(\phi-\omega)/2}\sin(\theta/2)\\
        0 & 0 & e^{-i(\phi-\omega)/2}\sin(\theta/2) & e^{i(\phi+\omega)/2}\cos(\theta/2)
        \end{bmatrix}.

    :param para: represents the array of parameters.
    :param control_qubits: Control qubit index.
    :param rot_wire: Rot qubit index.
    :param q_machine: Quantum machine device.
    

    Example::

        import pyvqnet
        pyvqnet.backends.set_backend("torch")
        from pyvqnet.tensor import QTensor
        from pyvqnet.qnn.vqc.tn import vqc_crot_circuit,TNQMachine, MeasureAll
        p = QTensor([2, 3, 4.0])
        qm = TNQMachine(2)
        vqc_crot_circuit(p, 0, 1, qm)
        m = MeasureAll(obs={"Z0": 1})
        exp = m(q_machine=qm)
        print(exp)




vqc_controlled_hadamard
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^


.. py:function:: pyvqnet.qnn.vqc.tn.vqc_controlled_hadamard(wires, q_machine)

    Controlled Hadamard logic gate quantum circuit. This function alias: ``VQC_Controlled_Hadamard`` .

    .. math:: 
        CH = \begin{bmatrix}
        1 & 0 & 0 & 0 \\
        0 & 1 & 0 & 0 \\
        0 & 0 & \frac{1}{\sqrt{2}} & \frac{1}{\sqrt{2}} \\
        0 & 0 & \frac{1}{\sqrt{2}} & -\frac{1}{\sqrt{2}}
        \end{bmatrix}.

    :param wires: quantum bit index list, the first one is the control bit, the list length is 2.
    :param q_machine: quantum virtual machine device.

    Examples::

        import pyvqnet
        pyvqnet.backends.set_backend("torch")
        from pyvqnet.tensor import QTensor
        from pyvqnet.qnn.vqc.tn import vqc_controlled_hadamard,\
            TNQMachine, MeasureAll

        p = QTensor([0.2, 3, 4.0])
        qm = TNQMachine(3)
        vqc_controlled_hadamard([1, 0], qm)
        m = MeasureAll(obs={"Z0": 1})
        exp = m(q_machine=qm)
        print(exp)


vqc_ccz
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. py:function:: pyvqnet.qnn.vqc.tn.vqc_ccz(wires, q_machine)

    Controlled-controlled-Z logic gate. Alias: ``VQC_CCZ`` .

    .. math::
        CCZ =
        \begin{pmatrix}
        1 & 0 & 0 & 0 & 0 & 0 & 0 & 0\\
        0 & 1 & 0 & 0 & 0 & 0 & 0 & 0 & 0\\
        0 & 0 & 1 & 0 & 0 & 0 & 0 & 0\\
        0 & 0 & 0 & 1 & 0 & 0 & 0 & 0\\
        0 & 0 & 0 & 0 & 1 & 0 & 0 & 0\\
        0 & 0 & 0 & 0 & 0 & 1 & 0 & 0 & 0\\
        0 & 0 & 0 & 0 & 0 & 1 & 0 & 0\\
        0 & 0 & 0 & 0 & 0 & 0 & 1 & 0 & 0\\
        0 & 0 & 0 & 0 & 0 & 0 & 0 & 1 & 0\\
        0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & -1
        \end{pmatrix}

    :param wires: quantum bit index list, the first one is the control bit. The list length is 3.
    :param q_machine: quantum virtual machine device.

    Example::

        import pyvqnet
        pyvqnet.backends.set_backend("torch")
        from pyvqnet.tensor import QTensor
        from pyvqnet.qnn.vqc.tn import vqc_ccz,TNQMachine, MeasureAll
        p = QTensor([0.2, 3, 4.0])

        qm = TNQMachine(3)

        vqc_ccz([1, 0, 2], qm)
        m = MeasureAll(obs={"Z0": 1})
        exp = m(q_machine=qm)
        print(exp)



vqc_fermionic_single_excitation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. py:function:: pyvqnet.qnn.vqc.tn.vqc_fermionic_single_excitation(weight, wires, q_machine)

    Coupled cluster single excitation operator for tensor product of Pauli matrices. Matrix form is given by:

    .. math::
        \hat{U}_{pr}(\theta) = \mathrm{exp} \{ \theta_{pr} (\hat{c}_p^\dagger \hat{c}_r
        -\mathrm{H.c.}) \},

    Alias: ``VQC_FermionicSingleExcitation`` .

    :param weight: Parameter on qubit p, only a elements.
    :param wires: A subset of qubit indices in the interval [r, p]. Minimum length must be 2. The first index value is interpreted as r, and the last a index value is interpreted as p.The intermediate indices are acted upon by CNOT gates to compute the parity of the qubit set.
    :param q_machine: Quantum virtual machine device.

    

    Examples::

        import pyvqnet
        pyvqnet.backends.set_backend("torch")
        from pyvqnet.tensor import QTensor
        from pyvqnet.qnn.vqc.tn import vqc_fermionic_single_excitation,\
            TNQMachine, MeasureAll
        qm = TNQMachine(3)
        p0 = QTensor([0.5])

        vqc_fermionic_single_excitation(p0, [1, 0, 2], qm)
        m = MeasureAll(obs={"Z0": 1})
        exp = m(q_machine=qm)
        print(exp)

 


vqc_fermionic_double_excitation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^


.. py:function:: pyvqnet.qnn.vqc.tn.vqc_fermionic_double_excitation(weight, wires1, wires2, q_machine)

    Coupled clustered biexcitation operator for tensor product of Pauli matrices exponentiated, matrix form given by:

    .. math::
        \hat{U}_{pqrs}(\theta) = \mathrm{exp} \{ \theta (\hat{c}_p^\dagger \hat{c}_q^\dagger
        \hat{c}_r \hat{c}_s - \mathrm{H.c.}) \},

    where :math:`\hat{c}` and :math:`\hat{c}^\dagger` are fermion annihilation and
    operators are created and indexed :math:`r, s` and :math:`p, q` on occupied and
    empty molecular orbitals respectively. Use `Jordan-Wigner transformation
    <https://arxiv.org/abs/1208.5986>`_ The fermion operator defined above can be written as
    in terms of the Pauli matrix (see
    `arXiv:1805.04340 <https://arxiv.org/abs/1805.04340>`_ for more details)

    .. math::
        \hat{U}_{pqrs}(\theta) = \mathrm{exp} \Big\{
        \frac{i\theta}{8} \bigotimes_{b=s+1}^{r-1} \hat{Z}_b \bigotimes_{a=q+1}^{p-1}
        \hat{Z}_a (\hat{X}_s \hat{X}_r \hat{Y}_q \hat{X}_p +
        \hat{Y}_s \hat{X}_r \hat{Y}_q \hat{Y}_p +\\ \hat{X}_s \hat{Y}_r \hat{Y}_q \hat{Y}_p +
        \hat{X}_s \hat{X}_r \hat{X}_q \hat{Y}_p - \mathrm{H.c.} ) \Big\}

    This function is aliased as: ``VQC_FermionicDoubleExcitation`` .

    :param weight: variable parameter
    :param wires1: represents the subset of qubits in the index list interval [s, r]. The ath index is interpreted as s and the last index is interpreted as r. The CNOT gate operates on the middle indexes to calculate the parity of a group of qubits.
    :param wires2: represents the subset of qubits in the index list interval [q, p]. The first root index is interpreted as q and the last index is interpreted as p. The CNOT gate operates on the middle indexes to calculate the parity of a group of qubits.
    :param q_machine: Quantum virtual machine device.

    

    Examples::

        import pyvqnet
        pyvqnet.backends.set_backend("torch")
        from pyvqnet.tensor import QTensor
        from pyvqnet.qnn.vqc.tn import vqc_fermionic_double_excitation,\
            TNQMachine, MeasureAll
        qm = TNQMachine(5)
        p0 = QTensor([0.5])

        vqc_fermionic_double_excitation(p0, [0, 1], [2, 3], qm)
        m = MeasureAll(obs={"Z0": 1})
        exp = m(q_machine=qm)
        print(exp)
 

vqc_uccsd
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^


.. py:function:: pyvqnet.qnn.vqc.tn.vqc_uccsd(weights, wires, s_wires, d_wires, init_state, q_machine)

    Implements the Unitary Coupled Cluster Single and Double Excitations Simulation (UCCSD). UCCSD is a VQE simulation commonly used to run quantum chemistry simulations.

    Within the first-order Trotter approximation, the UCCSD unitary function is given by:

    .. math::
        \hat{U}(\vec{\theta}) =
        \prod_{p > r} \mathrm{exp} \Big\{\theta_{pr}
        (\hat{c}_p^\dagger \hat{c}_r-\mathrm{H.c.}) \Big\}
        \prod_{p > q > r > s} \mathrm{exp} \Big\{\theta_{pqrs}
        (\hat{c}_p^\dagger \hat{c}_q^\dagger \hat{c}_r \hat{c}_s-\mathrm{H.c.}) \Big\}

    where :math:`\hat{c}` and :math:`\hat{c}^\dagger` are fermion annihilation and
    creation operators and index :math:`r, s` and :math:`p, q` on occupied and
    empty molecular orbitals respectively. (For more details see
    `arXiv:1805.04340 <https://arxiv.org/abs/1805.04340>`_):

    This function is aliased as: ``VQC_UCCSD`` .

    :param weights: tensor of size ``(len(s_wires)+ len(d_wires))`` containing the parameters :math:`\theta_{pr}` and :math:`\theta_{pqrs}` input Z rotations ``FermionicSingleExcitation`` and ``FermionicDoubleExcitation`` .
    :param wires: qubit indices for template action
    :param s_wires: sequence of lists containing qubit indices ``[r,...,p]`` generated by a single excitation :math:`\vert r, p \rangle = \hat{c}_p^\dagger \hat{c}_r \vert \mathrm{HF} \rangle`,where :math:`\vert \mathrm{HF} \rangle` denotes the Hartee-Fock reference state.
    :param d_wires: sequence of lists, each containing two lists specifying indices ``[s, ...,r]`` and ``[q,..., p]`` defining double excitation :math:`\vert s, r, q, p \rangle = \hat{c}_p^\dagger \hat{c}_q^\dagger \hat{c}_r\hat{c}_s \vert \mathrm{HF} \rangle` .
    :param init_state: occupation-number vector of length ``len(wires)`` representing the high-frequency state. ``init_state`` Initialization state of the qubit.
    :param q_machine: Quantum virtual machine device.

    Examples::

        import pyvqnet
        pyvqnet.backends.set_backend("torch")
        from pyvqnet.qnn.vqc.tn import vqc_uccsd, TNQMachine, MeasureAll
        from pyvqnet.tensor import QTensor
        p0 = QTensor([2, 0.5, -0.2, 0.3, -2, 1, 3, 0])
        s_wires = [[0, 1, 2], [0, 1, 2, 3, 4], [1, 2, 3], [1, 2, 3, 4, 5]]
        d_wires = [[[0, 1], [2, 3]], [[0, 1], [2, 3, 4, 5]], [[0, 1], [3, 4]],
                [[0, 1], [4, 5]]]
        qm = TNQMachine(6)

        vqc_uccsd(p0, range(6), s_wires, d_wires, QTensor([1.0, 1, 0, 0, 0, 0]), qm)
        m = MeasureAll(obs={"Z1": 1})
        exp = m(q_machine=qm)
        print(exp)

        # [[0.963802]]


vqc_zfeaturemap
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. py:function:: pyvqnet.qnn.vqc.tn.vqc_zfeaturemap(input_feat, q_machine: pyvqnet.qnn.vqc.tn.TNQMachine, data_map_func=None, rep: int = 2)

    First-order Pauli Z-evolution circuit.

    For 3 qubits and 2 repetitions, the circuit is represented as:

    .. parsed-literal::

        ┌───┐┌──────────────┐┌───┐┌──────────────┐
        ┤ H ├┤ U1(2.0*x[0]) ├┤ H ├┤ U1(2.0*x[0]) ├
        ├───┤├──────────────┤├───┤├──────────────┤
        ┤ H ├┤ U1(2.0*x[1]) ├┤ H ├┤ U1(2.0*x[1]) ├
        ├───┤├──────────────┤├───┤├──────────────┤
        ┤ H ├┤ U1(2.0*x[2]) ├┤ H ├┤ U1(2.0*x[2]) ├
        └───┘└──────────────┘└───┘└──────────────┘

    La chaîne de Pauli est fixée à ``Z``. Par conséquent, l'expansion de premier ordre sera un circuit sans portes d'intrication.

    :param input_feat: Array representing input parameters.
    :param q_machine: Quantum virtual machine.
    :param data_map_func: Parameter mapping matrix, a callable function, designed as: ``data_map_func = lambda x: x``.
    :param rep: Number of times the module is repeated.

    Example::

        import pyvqnet
        pyvqnet.backends.set_backend("torch")
        from pyvqnet.qnn.vqc.tn import vqc_zfeaturemap, TNQMachine, hadamard
        from pyvqnet.tensor import QTensor
        qm = TNQMachine(3)
        for i in range(3):
            hadamard(q_machine=qm, wires=[i])
        vqc_zfeaturemap(input_feat=QTensor([[0.1,0.2,0.3]]),q_machine = qm)
        print(qm.get_states())
 

vqc_zzfeaturemap
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. py:function:: pyvqnet.qnn.vqc.tn.vqc_zzfeaturemap(input_feat, q_machine: pyvqnet.qnn.vqc.tn.TNQMachine, data_map_func=None, entanglement: Union[str, List[List[int]],Callable[[int], List[int]]] = "full",rep: int = 2)

    Second-order Pauli-Z evolution circuit.

    For 3 qubits, 1 repeat, and linear entanglement, the circuit is represented as:

    .. parsed-literal::


        ┌───┐┌─────────────────┐
        ┤ H ├┤ U1(2.0*φ(x[0])) ├──■────────────────────────────■────────────────────────────────────
        ├───┤├─────────────────┤┌─┴─┐┌──────────────────────┐┌─┴─┐
        ┤ H ├┤ U1(2.0*φ(x[1])) ├┤ X ├┤ U1(2.0*φ(x[0],x[1])) ├┤ X ├──■────────────────────────────■──
        ├───┤├─────────────────┤└───┘└──────────────────────┘└───┘┌─┴─┐┌──────────────────────┐┌─┴─┐
        ┤ H ├┤ U1(2.0*φ(x[2])) ├──────────────────────────────────┤ X ├┤ U1(2.0*φ(x[1],x[2])) ├┤ X ├
        └───┘└─────────────────┘                                  └───┘└──────────────────────┘└───┘
    
    Where ``φ`` is a classic nonlinear function. If two values ​​are input, ``φ(x,y) = (pi - x)(pi - y)``, and if a is input, ``φ(x) = x``. It is expressed as follows using ``data_map_func``:

    .. code-block::

        def data_map_func(x):
            coeff = x if x.shape[-1] == 1 else ft.reduce(lambda x, y: (np.pi - x) * (np.pi - y), x)
            return coeff

    :param input_feat: Array representing input parameters.
    :param q_machine: Quantum virtual machine.
    :param data_map_func: parameter mapping matrix, a callable function.
    :param entanglement: specified entanglement structure.
    :param rep: module repetition times.
    
    Example::

        import pyvqnet
        pyvqnet.backends.set_backend("torch")
        from pyvqnet.qnn.vqc.tn import vqc_zzfeaturemap, TNQMachine
        from pyvqnet.tensor import QTensor

        qm = TNQMachine(3)
        vqc_zzfeaturemap(q_machine=qm, input_feat=QTensor([[0.1,0.2,0.3]]))
        print(qm.get_states())


vqc_allsinglesdoubles
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. py:function:: pyvqnet.qnn.vqc.tn.vqc_allsinglesdoubles(weights, q_machine: pyvqnet.qnn.vqc.tn.TNQMachine, hf_state, wires, singles=None, doubles=None)

    In this case, we have four single excitations and double excitations to preserve the total spin projection of the Hartree-Fock state.

    La matrice unitaire résultante préserve la population de particules et prépare le système à n qubits dans une superposition de l'état initial de Hartree-Fock et d'autres états encodant la configuration multi-excitation.

    :param weights: A QTensor of size ``(len(singles) + len(doubles),)`` containing the angles that enter the vqc.qCircuit.single_excitation and vqc.qCircuit.double_excitation operations in sequence
    :param q_machine: The quantum machine.
    :param hf_state: A vector of length ``len(wires)`` occupancy numbers representing the Hartree-Fock state, ``hf_state`` used to initialize the wires.
    :param wires: The qubits to act on.
    :param singles: A sequence of lists with the indices of the two qubits acted on by the single_exitation operation.
    :param doubles: List sequence with the indices of the two qubits acted on by the double_exitation operation.

    For example, the quantum circuit for two electrons and six qubits is shown below:

    .. image:: ./images/all_singles_doubles.png
        :width: 600 px
        :align: center

    |

    Example::

        import pyvqnet
        pyvqnet.backends.set_backend("torch")
        from pyvqnet.qnn.vqc.tn import vqc_allsinglesdoubles, TNQMachine

        from pyvqnet.tensor import QTensor
        qubits = 4
        qm = TNQMachine(qubits)

        vqc_allsinglesdoubles(q_machine=qm, weights=QTensor([0.55, 0.11, 0.53]), 
                              hf_state = QTensor([1,1,0,0]), singles=[[0, 2], [1, 3]], doubles=[[0, 1, 2, 3]], wires=[0,1,2,3])
        print(qm.get_states())

vqc_basisrotation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. py:function:: pyvqnet.qnn.vqc.tn.vqc_basisrotation(q_machine: pyvqnet.qnn.vqc.tn.TNQMachine, wires, unitary_matrix: QTensor, check=False)

    Implement a circuit that provides an ensemble that can be used to perform accurate single-unit basis rotations. The circuit is derived from the single-particle fermion-determined unitary transformation :math:`U(u)` given in `arXiv:1711.04789 <https://arxiv.org/abs/1711.04789>`_\ 
    
    .. math::
        U(u) = \exp{\left( \sum_{pq} \left[\log u \right]_{pq} (a_p^\dagger a_q - a_q^\dagger a_p) \right)}.

    :math:`U(u)` is obtained by using the scheme given in the paper `Optica, 3, 1460 (2016) <https://opg.optica.org/optica/fulltext.cfm?uri=optica-3-12-1460&id=355743>`_\ .

    :param q_machine: quantum machine.
    :param wires: qubits to act on.
    :param unitary_matrix: matrix specifying the basis for the transformation.
    :param check: check if `unitary_matrix` is a unitary matrix.

    Example::

        import pyvqnet

        pyvqnet.backends.set_backend("torch")
        from pyvqnet.qnn.vqc.tn import vqc_basisrotation, TNQMachine
        from pyvqnet.tensor import QTensor
        import numpy as np

        V = np.array([[0.73678 + 0.27511j, -0.5095 + 0.10704j, -0.06847 + 0.32515j],
                    [0.73678 + 0.27511j, -0.5095 + 0.10704j, -0.06847 + 0.32515j],
                    [-0.21271 + 0.34938j, -0.38853 + 0.36497j, 0.61467 - 0.41317j]])

        eigen_vals, eigen_vecs = np.linalg.eigh(V)
        umat = eigen_vecs.T
        wires = range(len(umat))

        qm = TNQMachine(len(umat))

        vqc_basisrotation(q_machine=qm,
                        wires=wires,
                        unitary_matrix=QTensor(umat, dtype=qm.dtype))

        print(qm.get_states())



Interface distribuée
================================================

Fonctions liées à la distribution, lors de l'utilisation du backend de calcul ``torch``, encapsulent l'interface ``torch.distributed`` de torch,

.. note::

    Veuillez vous référer à `torch distributed <https://pytorch.org/docs/stable/distributed.html>`_ pour démarrer la méthode distribuée.
    Lors de l'utilisation du CPU pour la distribution, veuillez utiliser ``gloo`` au lieu de ``mpi``.
    Lors de l'utilisation du GPU pour la distribution, veuillez utiliser ``nccl``.

    :ref:`vqnet_dist` L'interface distribuée propre à VQNet n'est pas applicable au backend de calcul ``torch``.

CommController
-------------------------

.. py:class:: pyvqnet.distributed.ControlComm.CommController(backend,rank=None,world_size=None)
    :no-index:
    
    CommController est utilisé pour contrôler le contrôleur de communication de données sous cpu et gpu. Il génère des contrôleurs cpu (gloo) et gpu (nccl) en définissant le paramètre `backend`.
    Cette classe appellera backend, rank, world_size pour initialiser ``torch.distributed.init_process_group(backend, rank, world_size)``.

    :param backend: utilisé pour générer le contrôleur de communication de données cpu ou gpu, 'gloo' ou 'nccl'.
    :param rank: le numéro de processus du programme actuel.
    :param world_size: le nombre de tous les processus globaux.

    :return:
        Instance CommController.

    Examples::

        from pyvqnet.distributed import CommController
        import pyvqnet
        pyvqnet.backends.set_backend("torch")
        import os
        import multiprocessing as mp


        def init_process(rank, size):
            """ Initialize the distributed environment. """
            os.environ['MASTER_ADDR'] = '127.0.0.1'
            os.environ['MASTER_PORT'] = '29500'
            os.environ['LOCAL_RANK'] = f"{rank}"
            pp = CommController("gloo", rank=rank, world_size=size)
            
            local_rank = pp.get_rank()
            print(local_rank)


        if __name__ == "__main__":
            world_size = 2
            processes = []
            mp.set_start_method("spawn")
            for rank in range(world_size):
                p = mp.Process(target=init_process, args=(rank, world_size))
                p.start()
                processes.append(p)

            for p in processes:
                p.join()
 

    .. py:method:: getRank()
        :no-index:

        Utilisé pour obtenir l'ID du processus actuel.

        :return: Retourne l'ID du processus actuel.

        Examples::

            from pyvqnet.distributed import CommController
            import pyvqnet
            pyvqnet.backends.set_backend("torch")
            import os
            import multiprocessing as mp


            def init_process(rank, size):
                """ Initialize the distributed environment. """
                os.environ['MASTER_ADDR'] = '127.0.0.1'
                os.environ['MASTER_PORT'] = '29500'
                os.environ['LOCAL_RANK'] = f"{rank}"
                pp = CommController("gloo", rank=rank, world_size=size)
                
                local_rank = pp.getRank()
                print(local_rank)


            if __name__ == "__main__":
                world_size = 2
                processes = []
                mp.set_start_method("spawn")
                for rank in range(world_size):
                    p = mp.Process(target=init_process, args=(rank, world_size))
                    p.start()
                    processes.append(p)

                for p in processes:
                    p.join()
            


    .. py:method:: getSize()
        :no-index:

        Utilisé pour obtenir le nombre total de processus démarrés.

        :return: Retourne le nombre total de processus.

        Examples::

                        from pyvqnet.distributed import CommController
            import pyvqnet
            pyvqnet.backends.set_backend("torch")
            import os
            import multiprocessing as mp


            def init_process(rank, size):
                """ Initialize the distributed environment. """
                os.environ['MASTER_ADDR'] = '127.0.0.1'
                os.environ['MASTER_PORT'] = '29500'
                os.environ['LOCAL_RANK'] = f"{rank}"
                pp = CommController("gloo", rank=rank, world_size=size)
                
                local_rank = pp.getSize()
                print(local_rank)


            if __name__ == "__main__":
                world_size = 2
                processes = []
                mp.set_start_method("spawn")
                for rank in range(world_size):
                    p = mp.Process(target=init_process, args=(rank, world_size))
                    p.start()
                    processes.append(p)

                for p in processes:
                    p.join()
            


    .. py:method:: getLocalRank()
        :no-index:

        Dans chaque processus, obtient le numéro de processus local de chaque machine via ``os.environ['LOCAL_RANK'] = rank``.

        La variable d'environnement `LOCAL_RANK` doit être définie à l'avance.

        :return: Le numéro de processus actuel sur la machine actuelle.

        Examples::

            from pyvqnet.distributed import CommController
            import pyvqnet
            pyvqnet.backends.set_backend("torch")
            import os
            import multiprocessing as mp


            def init_process(rank, size):
                """ Initialize the distributed environment. """
                os.environ['MASTER_ADDR'] = '127.0.0.1'
                os.environ['MASTER_PORT'] = '29500'
                os.environ['LOCAL_RANK'] = f"{rank}"
                pp = CommController("gloo", rank=rank, world_size=size)
                
                local_rank = pp.getLocalRank()
                print(local_rank )


            if __name__ == "__main__":
                world_size = 2
                processes = []
                mp.set_start_method("spawn")
                for rank in range(world_size):
                    p = mp.Process(target=init_process, args=(rank, world_size))
                    p.start()
                    processes.append(p)

                for p in processes:
                    p.join()

 
    .. py:method:: split_groups(rankL)
        :no-index:

        La liste de numéros de processus définie selon le paramètre d'entrée est utilisée pour diviser plusieurs groupes de communication.

        :param rankL: liste de groupes de processus.
        :return: une liste contenant ``torch.distributed.ProcessGroup``

        Examples::

            from pyvqnet.distributed import get_local_rank,CommController
            import pyvqnet
            import numpy as np
            from pyvqnet.tensor import tensor
            pyvqnet.backends.set_backend("torch")
            import os
            import multiprocessing as mp


            def init_process(rank, size):
                """ Initialize the distributed environment. """
                os.environ['MASTER_ADDR'] = '127.0.0.1'
                os.environ['MASTER_PORT'] = '29500'
                os.environ['LOCAL_RANK'] = f"{rank}"
                Comm_OP = CommController("gloo", rank=rank, world_size=size)

                group = Comm_OP.split_groups([[1,3]])

                num = tensor.to_tensor(np.random.rand(1, 5)+get_local_rank()*10)
                print(f"before rank {Comm_OP.getRank()}  {num}\n")
                
                Comm_OP.reduce_group(num, 1,"sum",group[0])
                print(f"after rank {Comm_OP.getRank()}  {num}\n")
                

            if __name__ == "__main__":
                world_size = 4
                processes = []
                mp.set_start_method("spawn")
                for rank in range(world_size):
                    p = mp.Process(target=init_process, args=(rank, world_size))
                    p.start()
                    processes.append(p)

                for p in processes:
                    p.join()
 


 
    .. py:method:: barrier()
        :no-index:

        Synchronisation des différents processus.

        :return: Opération de synchronisation.

        Examples::

            from pyvqnet.distributed import CommController
            import pyvqnet
            pyvqnet.backends.set_backend("torch")
            import os
            import multiprocessing as mp


            def init_process(rank, size):
                """ Initialize the distributed environment. """
                os.environ['MASTER_ADDR'] = '127.0.0.1'
                os.environ['MASTER_PORT'] = '29500'
                os.environ['LOCAL_RANK'] = f"{rank}"
                pp = CommController("gloo", rank=rank, world_size=size)
                
                pp.barrier()


            if __name__ == "__main__":
                world_size = 4
                processes = []
                mp.set_start_method("spawn")
                for rank in range(world_size):
                    p = mp.Process(target=init_process, args=(rank, world_size))
                    p.start()
                    processes.append(p)

                for p in processes:
                    p.join()

    .. py:method:: allreduce(tensor, c_op = "avg")
        :no-index:

        Supporte la communication allreduce sur les données.

        :param tensor: Données d'entrée.
        :param c_op: Méthode de calcul.

        Examples::

            from pyvqnet.distributed import get_local_rank,CommController
            import pyvqnet
            import numpy as np
            from pyvqnet.tensor import tensor
            pyvqnet.backends.set_backend("torch")
            import os
            import multiprocessing as mp


            def init_process(rank, size):
                """ Initialize the distributed environment. """
                os.environ['MASTER_ADDR'] = '127.0.0.1'
                os.environ['MASTER_PORT'] = '29500'
                os.environ['LOCAL_RANK'] = f"{rank}"
                Comm_OP = CommController("gloo", rank=rank, world_size=size)

                num = tensor.to_tensor(np.random.rand(1, 5))
                print(f"rank {Comm_OP.getRank()}  {num}")

                Comm_OP.all_reduce(num, "sum")
                print(f"rank {Comm_OP.getRank()}  {num}")

            if __name__ == "__main__":
                world_size = 2
                processes = []
                mp.set_start_method("spawn")
                for rank in range(world_size):
                    p = mp.Process(target=init_process, args=(rank, world_size))
                    p.start()
                    processes.append(p)

                for p in processes:
                    p.join()

    .. py:method:: reduce(tensor, root = 0, c_op = "avg")
        :no-index:

        Supporte la communication reduce sur les données.

        :param tensor: Input data.
        :param root: Spécifie le nœud où les données sont retournées.
        :param c_op: Calculation method.

        Examples::

            from pyvqnet.distributed import get_local_rank,CommController
            import pyvqnet
            import numpy as np
            from pyvqnet.tensor import tensor
            pyvqnet.backends.set_backend("torch")
            import os
            import multiprocessing as mp


            def init_process(rank, size):
                """ Initialize the distributed environment. """
                os.environ['MASTER_ADDR'] = '127.0.0.1'
                os.environ['MASTER_PORT'] = '29500'
                os.environ['LOCAL_RANK'] = f"{rank}"
                Comm_OP = CommController("gloo", rank=rank, world_size=size)

                num = tensor.to_tensor(np.random.rand(1, 5))
                print(f"before rank {Comm_OP.getRank()}  {num}")
                
                Comm_OP.reduce(num, 1,"sum")
                print(f"after rank {Comm_OP.getRank()}  {num}")
                

            if __name__ == "__main__":
                world_size = 3
                processes = []
                mp.set_start_method("spawn")
                for rank in range(world_size):
                    p = mp.Process(target=init_process, args=(rank, world_size))
                    p.start()
                    processes.append(p)

                for p in processes:
                    p.join()
 
 
    .. py:method:: broadcast(tensor, root = 0)
        :no-index:

        Diffuse les données sur le root de processus spécifié à tous les processus.

        :param tensor: Input data.
        :param root: Le nœud spécifié.

        Examples::

            from pyvqnet.distributed import get_local_rank,CommController
            import pyvqnet
            import numpy as np
            from pyvqnet.tensor import tensor
            pyvqnet.backends.set_backend("torch")
            import os
            import multiprocessing as mp


            def init_process(rank, size):
                """ Initialize the distributed environment. """
                os.environ['MASTER_ADDR'] = '127.0.0.1'
                os.environ['MASTER_PORT'] = '29500'
                os.environ['LOCAL_RANK'] = f"{rank}"
                Comm_OP = CommController("gloo", rank=rank, world_size=size)


                num = tensor.to_tensor(np.random.rand(1, 5))+ rank
                print(f"before rank {Comm_OP.getRank()}  {num}")
                
                Comm_OP.broadcast(num, 1)
                print(f"after rank {Comm_OP.getRank()}  {num}")
                

            if __name__ == "__main__":
                world_size = 3
                processes = []
                mp.set_start_method("spawn")
                for rank in range(world_size):
                    p = mp.Process(target=init_process, args=(rank, world_size))
                    p.start()
                    processes.append(p)

                for p in processes:
                    p.join()


    .. py:method:: allgather(tensor)
        :no-index:

        Rassemble toutes les données de tous les processus ensemble. Cette interface supporte uniquement le backend nccl.

        :param tensor: Input data.

        Examples::

            from pyvqnet.distributed import get_local_rank,CommController,get_world_size
            import pyvqnet
            import numpy as np
            from pyvqnet.tensor import tensor
            pyvqnet.backends.set_backend("torch")
            import os
            import multiprocessing as mp


            def init_process(rank, size):
                """ Initialize the distributed environment. """
                os.environ['MASTER_ADDR'] = '127.0.0.1'
                os.environ['MASTER_PORT'] = '29500'
                os.environ['LOCAL_RANK'] = f"{rank}"
                Comm_OP = CommController("nccl", rank=rank, world_size=size)

                num = tensor.QTensor(np.random.rand(5,4),device=pyvqnet.DEV_GPU_0+rank)
                print(f"before rank {Comm_OP.getRank()}  {num}\n")

                num = Comm_OP.all_gather(num)
                print(f"after rank {Comm_OP.getRank()}  {num}\n")


            if __name__ == "__main__":
                world_size = 2
                processes = []
                mp.set_start_method("spawn")
                for rank in range(world_size):
                    p = mp.Process(target=init_process, args=(rank, world_size))
                    p.start()
                    processes.append(p)

                for p in processes:
                    p.join()


    .. py:method:: send(tensor, dest)
        :no-index:

        Interface de communication p2p.

        :param tensor: input data.
        :param dest: processus de destination.

        Examples::

            from pyvqnet.distributed import get_rank,CommController,get_world_size
            import pyvqnet
            import numpy as np
            from pyvqnet.tensor import tensor
            pyvqnet.backends.set_backend("torch")
            import os
            import multiprocessing as mp


            def init_process(rank, size):
                """ Initialize the distributed environment. """
                os.environ['MASTER_ADDR'] = '127.0.0.1'
                os.environ['MASTER_PORT'] = '29500'
                os.environ['LOCAL_RANK'] = f"{rank}"
                Comm_OP = CommController("gloo", rank=rank, world_size=size)

                num = tensor.to_tensor(np.random.rand(1, 5))
                recv = tensor.zeros_like(num)
                if get_rank() == 0:
                    Comm_OP.send(num, 1)
                elif get_rank() == 1:
                    Comm_OP.recv(recv, 0)
                print(f"before rank {Comm_OP.getRank()}  {num}")
                print(f"after rank {Comm_OP.getRank()}  {recv}")

            if __name__ == "__main__":
                world_size = 2
                processes = []
                mp.set_start_method("spawn")
                for rank in range(world_size):
                    p = mp.Process(target=init_process, args=(rank, world_size))
                    p.start()
                    processes.append(p)

                for p in processes:
                    p.join()
 
 
    .. py:method:: recv(tensor, source)
        :no-index:

        p2p communication interface.

        :param tensor: input data.
        :param source: processus de réception.

        Examples::

            from pyvqnet.distributed import get_rank,CommController,get_world_size
            import pyvqnet
            import numpy as np
            from pyvqnet.tensor import tensor
            pyvqnet.backends.set_backend("torch")
            import os
            import multiprocessing as mp


            def init_process(rank, size):
                """ Initialize the distributed environment. """
                os.environ['MASTER_ADDR'] = '127.0.0.1'
                os.environ['MASTER_PORT'] = '29500'
                os.environ['LOCAL_RANK'] = f"{rank}"
                Comm_OP = CommController("gloo", rank=rank, world_size=size)

                num = tensor.to_tensor(np.random.rand(1, 5))
                recv = tensor.zeros_like(num)
                if get_rank() == 0:
                    Comm_OP.send(num, 1)
                elif get_rank() == 1:
                    Comm_OP.recv(recv, 0)
                print(f"before rank {Comm_OP.getRank()}  {num}")
                print(f"after rank {Comm_OP.getRank()}  {recv}")

            if __name__ == "__main__":
                world_size = 2
                processes = []
                mp.set_start_method("spawn")
                for rank in range(world_size):
                    p = mp.Process(target=init_process, args=(rank, world_size))
                    p.start()
                    processes.append(p)

                for p in processes:
                    p.join()
            

    .. py:method:: allreduce_group(tensor, c_op = "avg", group = None)
        :no-index:

        Interface de communication allreduce intra-groupe.

        :param tensor: Input data.
        :param c_op: Calculation method.
        :param group: Groupe de communication généré à partir de `split_groups` ou `init_groups`.

        Examples::

            from pyvqnet.distributed import get_local_rank,CommController
            import pyvqnet
            import numpy as np
            from pyvqnet.tensor import tensor
            pyvqnet.backends.set_backend("torch")
            import os
            import multiprocessing as mp


            def init_process(rank, size):
                """ Initialize the distributed environment. """
                os.environ['MASTER_ADDR'] = '127.0.0.1'
                os.environ['MASTER_PORT'] = '29500'
                os.environ['LOCAL_RANK'] = f"{rank}"
                Comm_OP = CommController("gloo", rank=rank, world_size=size)

                rankL = [[0,1],[2,3]]
                groups = Comm_OP.split_groups(rankL)
                num = tensor.to_tensor(np.ones(5)+get_local_rank()*1000)

                print(f"before rank {Comm_OP.getRank()}  {num}")
                if Comm_OP.getRank() in rankL[0]:
                    Comm_OP.all_reduce_group(num, "sum", groups[0])

                    print(f"after rank {Comm_OP.getRank()}  {num}")

                if Comm_OP.getRank() in rankL[1]:
                    Comm_OP.all_reduce_group(num, "sum", groups[1])

                    print(f"after rank {Comm_OP.getRank()}  {num}")

            if __name__ == "__main__":
                world_size = 4
                mp.set_start_method("spawn")
                processes = []
                for rank in range(world_size):
                    p = mp.Process(target=init_process, args=(rank, world_size))
                    p.start()
                    processes.append(p)

                for p in processes:
                    p.join()



    .. py:method:: reduce_group(tensor, root = 0, c_op = "avg", group = None)
        :no-index:

        Interface de communication reduce intra-groupe.

        :param tensor: Input data.
        :param root: Spécifie le numéro de processus.
        :param c_op: Calculation method.
        :param group: Communication group generated from `split_groups` or `init_groups` .

        Examples::

            from pyvqnet.distributed import get_local_rank,CommController
            import pyvqnet
            import numpy as np
            from pyvqnet.tensor import tensor
            pyvqnet.backends.set_backend("torch")
            import os
            import multiprocessing as mp


            def init_process(rank, size):
                """ Initialize the distributed environment. """
                os.environ['MASTER_ADDR'] = '127.0.0.1'
                os.environ['MASTER_PORT'] = '29500'
                os.environ['LOCAL_RANK'] = f"{rank}"
                Comm_OP = CommController("gloo", rank=rank, world_size=size)
                rankL = [[1,3],[0,2]]
                group = Comm_OP.split_groups([[1,3],[0,2]])

                num = tensor.to_tensor(np.random.rand(1, 5)+get_local_rank()*10)
                print(f"before rank {Comm_OP.getRank()}  {num}\n")
                if Comm_OP.getRank() in rankL[0]:
                    Comm_OP.reduce_group(num, rankL[0][1],"sum",group[0])
                    print(f"after rank {Comm_OP.getRank()}  {num}\n")
                if Comm_OP.getRank() in rankL[1]:
                    Comm_OP.reduce_group(num, rankL[1][1],"sum",group[1])
                    print(f"after rank {Comm_OP.getRank()}  {num}\n")

            if __name__ == "__main__":
                world_size = 4
                processes = []
                mp.set_start_method("spawn")
                for rank in range(world_size):
                    p = mp.Process(target=init_process, args=(rank, world_size))
                    p.start()
                    processes.append(p)

                for p in processes:
                    p.join()

 
    .. py:method:: broadcast_group(tensor, root = 0, group = None)
        :no-index:

        Interface de communication broadcast intra-groupe.

        :param tensor: Input data.
        :param root: Spécifie l'ID du processus.
        :param group: Communication group generated from `split_groups` or `init_groups` .

        Examples::
            
            from pyvqnet.distributed import get_local_rank,CommController
            import pyvqnet
            import numpy as np
            from pyvqnet.tensor import tensor
            pyvqnet.backends.set_backend("torch")
            import os
            import multiprocessing as mp


            def init_process(rank, size):
                """ Initialize the distributed environment. """
                os.environ['MASTER_ADDR'] = '127.0.0.1'
                os.environ['MASTER_PORT'] = '29500'
                os.environ['LOCAL_RANK'] = f"{rank}"
                Comm_OP = CommController("gloo", rank=rank, world_size=size)

                rankL = [[2,3],[0,1,4]]
                group = Comm_OP.split_groups(rankL)

                num = tensor.to_tensor(np.random.rand(1, 5))+ rank*1000
                print(f"before rank {Comm_OP.getRank()}  {num}")
                
                if Comm_OP.getRank() in rankL[0]:
                    Comm_OP.broadcast_group(num, rankL[0][0],group[0])
                    print(f"after rank {Comm_OP.getRank()}  {num}")

                if Comm_OP.getRank() in rankL[1]:
                    Comm_OP.broadcast_group(num, rankL[1][1],group[1])
                    print(f"after rank {Comm_OP.getRank()}  {num}")

            if __name__ == "__main__":
                world_size = 5
                processes = []
                mp.set_start_method("spawn")
                for rank in range(world_size):
                    p = mp.Process(target=init_process, args=(rank, world_size))
                    p.start()
                    processes.append(p)

                for p in processes:
                    p.join()

 
    .. py:method:: allgather_group(tensor, group = None)
        :no-index:
        
        Interface de communication Allgather au sein du groupe.

        :param tensor: input data.
        :param group: Communication group generated from `split_groups` or `init_groups` .

        Examples::
            
            from pyvqnet.distributed import get_local_rank,CommController,get_world_size
            import pyvqnet
            import numpy as np
            from pyvqnet.tensor import tensor
            pyvqnet.backends.set_backend("torch")
            import os
            import multiprocessing as mp
            

            def init_process(rank, size ):
                """ Initialize the distributed environment. """
                os.environ['MASTER_ADDR'] = '127.0.0.1'
                os.environ['MASTER_PORT'] = '29500'
                os.environ['LOCAL_RANK'] = f"{rank}"
                Comm_OP = CommController("nccl", rank=rank, world_size=size)

                group = Comm_OP.split_groups([[0,1]])
                print(f"get_world_size {get_world_size()}")

                num = tensor.QTensor(np.random.rand(5,4)+get_local_rank()*100,device=pyvqnet.DEV_GPU_0+get_local_rank())
                print(f"before rank {Comm_OP.getRank()}  {num}\n")

                num = Comm_OP.allgather_group(num,group[0])
                print(f"after rank {Comm_OP.getRank()}  {num}\n")


            if __name__ == "__main__":
                world_size = 2
                processes = []
                mp.set_start_method("spawn")
                for rank in range(world_size):
                    p = mp.Process(target=init_process, args=(rank, world_size ))
                    p.start()
                    processes.append(p)

                for p in processes:
                    p.join()

