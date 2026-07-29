
.. _vqnet_dist:

Module de calcul distribué natif VQNet
*********************************************************

Déploiement de l'environnement
=================================

Cette section décrit comment déployer l'environnement VQNet sous Linux pour le calcul distribué CPU et GPU.

Installation de MPI
^^^^^^^^^^^^^^^^^^^^^^

MPI est une bibliothèque courante pour la communication inter-CPU, et la fonction de calcul distribué du CPU dans VQNet est implémentée sur la base de MPI.
La section suivante décrit comment installer MPI sur un système Linux (actuellement, la fonction de calcul distribué basée sur CPU n'est supportée que sous Linux).

Vérifiez si les compilateurs gcc et gfortran sont installés.

.. code-block::
        
    which gcc 
    which gfortran

Lorsque les chemins vers gcc et gfortran s'affichent, vous pouvez passer à l'étape suivante de l'installation. Si vous ne disposez pas des compilateurs correspondants,
veuillez d'abord les installer. Une fois les compilateurs vérifiés, utilisez la commande wget pour télécharger mpich.

.. code-block::
        
    wget http://www.mpich.org/static/downloads/3.3.2/mpich-3.3.2.tar.gz 
    tar -zxvf mpich-3.3.2.tar.gz 
    cd mpich-3.3.2 
    ./configure --prefix=/usr/local/mpich
    make 
    make install 

Après avoir compilé et installé mpich, configurez ses variables d'environnement.

.. code-block::
        
    vim ~/.bashrc

    # En bas du document, ajoutez
    export PATH="/usr/local/mpich/bin:$PATH"

Après avoir sauvegardé et quitté, utilisez source pour appliquer les modifications.

.. code-block::

    source ~/.bashrc

Utilisez which pour vérifier que les variables d'environnement sont correctement configurées. Si le chemin s'affiche, l'installation est terminée avec succès.

De plus, vous pouvez installer mpi4py via pip install. Si vous rencontrez l'erreur suivante :

.. image:: ./images/mpi_bug.png
    :align: center

|

Pour résoudre l'incompatibilité entre mpi4py et la version de Python, vous pouvez effectuer les opérations suivantes :

.. code-block::

    # Sauvegardez le compilateur de l'environnement Python actuel
    pushd /root/anaconda3/envs/$CONDA_DEFAULT_ENV/compiler_compat && mv ld ld.bak && popd

    # Réinstallez mpi4py
    pip install mpi4py

    # Restaurez le compilateur d'origine
    pushd /root/anaconda3/envs/$CONDA_DEFAULT_ENV/compiler_compat && mv ld.bak ld && popd

Installation de NCCL
^^^^^^^^^^^^^^^^^^^^^^

NCCL est une bibliothèque courante pour la communication GPU, et la fonction de calcul distribué des GPU dans VQNet est implémentée sur la base de NCCL.
Ce logiciel installe la bibliothèque de liens dynamiques NCCL par défaut lors de l'installation, il n'est donc généralement pas nécessaire d'installer NCCL séparément.
Cette section nécessite le support MPI, donc l'environnement MPI doit également être déployé.

Lancement distribué
=================================
 
En utilisant l'interface de calcul distribué lancée par la commande ``vqnetrun``, les paramètres de ``vqnetrun`` sont décrits ci-dessous.

n, np
^^^^^^^^^^^^^^^^^^^^^^

L'interface ``vqnetrun`` permet de contrôler le nombre de processus lancés avec les paramètres ``-n``, ``-np``, comme illustré dans l'exemple suivant.

    Exemple ::

        from pyvqnet.distributed import CommController
        Comm_OP = CommController("mpi") # initialise le contrôleur mpi
        
        rank = Comm_OP.getRank()
        size = Comm_OP.getSize()
        print(f"rank: {rank}, size {size}")

        # vqnetrun -n 2 python test.py
        # vqnetrun -np 2 python test.py

backend
^^^^^^^^^^^^^^^^^^^^^^

L'interface ``vqnetrun`` permet de sélectionner le backend distribué avec le paramètre ``--backend``, supportant les modes ``mpi`` (par défaut) et ``nccl``.
Le mode MPI est destiné au calcul distribué sur CPU, et le mode NCCL est destiné au calcul distribué sur GPU.

    Exemple ::

        from pyvqnet.distributed import CommController
        Comm_OP = CommController("nccl")

        rank = Comm_OP.getRank()
        size = Comm_OP.getSize()
        print(f"rank: {rank}, size {size}")

        # vqnetrun --backend nccl --nproc_per_node 2 python test.py

nproc_per_node
^^^^^^^^^^^^^^^^^^^^^^

L'interface ``vqnetrun`` permet de contrôler le nombre de processus lancés sur chaque nœud avec le paramètre ``--nproc_per_node``, disponible uniquement en mode NCCL.

    Exemple ::

        from pyvqnet.distributed import CommController
        Comm_OP = CommController("nccl")

        rank = Comm_OP.getRank()
        size = Comm_OP.getSize()
        print(f"rank: {rank}, size {size}")

        # vqnetrun --backend nccl --nproc_per_node 4 python test.py

output-filename
^^^^^^^^^^^^^^^^^^^^^^

L'interface ``vqnetrun`` permet d'enregistrer la sortie dans un fichier spécifié avec le paramètre de ligne de commande ``--output-filename``.

Un exemple d'implémentation est le suivant :

    Exemple ::

        from pyvqnet.distributed import CommController, get_host_name
        Comm_OP = CommController("mpi") # initialise le contrôleur mpi
        
        rank = Comm_OP.getRank()
        size = Comm_OP.getSize()
        print(f"rank: {rank}, size {size}")
        print(f"LocalRank {Comm_OP.getLocalRank()} hosts name {get_host_name()}")

        # vqnetrun -np 4 --hostfile hosts --output-filename output  python test.py


verbose
^^^^^^^^^^^^^^^^^^^^^^

L'interface ``vqnetrun`` peut être utilisée avec le paramètre de ligne de commande ``--verbose`` pour activer la journalisation détaillée des communications et afficher les résultats.

Un exemple d'implémentation est le suivant :

    Exemple ::

        from pyvqnet.distributed import CommController, get_host_name
        Comm_OP = CommController("mpi") # initialise le contrôleur mpi
        
        rank = Comm_OP.getRank()
        size = Comm_OP.getSize()
        print(f"rank: {rank}, size {size}")
        print(f"LocalRank {Comm_OP.getLocalRank()} hosts name {get_host_name()}")

        # vqnetrun -np 4 --hostfile hosts --verbose python test.py


start-timeout
^^^^^^^^^^^^^^^^^^^^^^

L'interface ``vqnetrun`` peut être utilisée avec le paramètre de ligne de commande ``-start-timeout`` pour spécifier que toutes les vérifications sont effectuées et que le processus est démarré avant le délai d'attente. La valeur par défaut est de 30 secondes.

Un exemple d'implémentation est le suivant :

    Exemple ::

        from pyvqnet.distributed import CommController, get_host_name
        Comm_OP = CommController("mpi") # initialise le contrôleur mpi
        
        rank = Comm_OP.getRank()
        size = Comm_OP.getSize()
        print(f"rank: {rank}, size {size}")
        print(f"LocalRank {Comm_OP.getLocalRank()} hosts name {get_host_name()}")

        # vqnetrun -np 4 --start-timeout 10 python test.py


CommController
=================================

    Le calcul distribué est utilisé pour contrôler la communication des données entre différents processus sous CPU et GPU. Il génère différents contrôleurs pour CPU (MPI) et GPU (NCCL), et appelle les méthodes de communication pour effectuer la communication et la synchronisation des données entre différents processus.

__init__
^^^^^^^^^^^^^^^^^^^^^^
.. py:class:: pyvqnet.distributed.ControlComm.CommController(backend,rank=None,world_size=None)
    
    CommController est utilisé pour contrôler la communication des données sous CPU et GPU. En définissant le paramètre `backend`, il génère le contrôleur pour CPU (MPI) ou GPU (NCCL). (Actuellement, la fonction de calcul distribué n'est supportée que sous Linux.)

    :param backend: utilisé pour générer le contrôleur de communication de données pour cpu ou gpu.
    :param rank: Ce paramètre n'est utile que dans les backends non pyvqnet, la valeur par défaut est : None.
    :param world_size: Ce paramètre n'est utile que dans les backends non pyvqnet, la valeur par défaut est : None.
        
    :return:
        Instance de CommController.

    Exemples ::

        from pyvqnet.distributed import CommController
        Comm_OP = CommController("nccl") # initialise le contrôleur nccl

        # Comm_OP = CommController("mpi") # initialise le contrôleur mpi

 
    .. py:method:: getRank()
        
        Utilisé pour obtenir le numéro de processus du processus actuel.

        :return: Renvoie le numéro de processus du processus actuel.

        Exemples ::

            from pyvqnet.distributed import CommController
            Comm_OP = CommController("nccl") # initialise le contrôleur nccl
            
            Comm_OP.getRank()

 
    .. py:method:: getSize()
    
        Utilisé pour obtenir le nombre total de processus lancés.


        :return: Renvoie le nombre total de processus.

        Exemples ::

            from pyvqnet.distributed import CommController
            Comm_OP = CommController("nccl") # initialise le contrôleur nccl
            
            Comm_OP.getSize()
            # vqnetrun --backend nccl --nproc_per_node 2 python test.py



    .. py:method:: getLocalRank()
    
        Utilisé pour obtenir le numéro de processus actuel sur la machine courante.


        :return: Le numéro de processus actuel sur la machine courante.

        Exemples ::

            from pyvqnet.distributed import CommController
            Comm_OP = CommController("nccl") # initialise le contrôleur nccl
            
            Comm_OP.getLocalRank()
            # vqnetrun --backend nccl --nproc_per_node 2 python test.py


    .. py:method:: split_groups(rankL)
        
        Divise plusieurs groupes de communication selon la liste des numéros de processus définie par le paramètre d'entrée.

        :param rankL: Une liste de rangs de groupes de processus.

        :return: Lorsque le backend est `nccl`, un tuple de rangs de groupes de processus est renvoyé.
                 Lorsque le backend est `mpi`, renvoie une liste dont la longueur est égale au nombre de groupes ; chaque élément est un tuple (comm, rank), où comm est le communicateur MPI du groupe et rank est le numéro de séquence au sein du groupe.

        Exemples ::
            
            from pyvqnet.distributed import CommController,get_rank,get_local_rank
            from pyvqnet.tensor import tensor
            import numpy as np
            Comm_OP = CommController("mpi")

            groups = Comm_OP.split_groups([[0, 1],[2,3]])
            print(groups)
            #[[<mpi4py.MPI.Intracomm object at 0x7f53691f3230>, [0, 3]], [<mpi4py.MPI.Intracomm object at 0x7f53691f3010>, [2, 1]]]

 
    .. py:method:: barrier()
    
        Synchronisation.

        :return: Synchronisation.

        Exemples ::

            from pyvqnet.distributed import CommController
            Comm_OP = CommController("nccl")
            
            Comm_OP.barrier()

 
    .. py:method:: get_device_num()
    
        Utilisé pour obtenir le nombre de cartes graphiques sur le nœud actuel (supporté uniquement sur gpu).

        :return: Renvoie le nombre de cartes graphiques sur le nœud actuel.

        Exemples ::

            from pyvqnet.distributed import CommController
            Comm_OP = CommController("nccl")
            
            Comm_OP.get_device_num()
            # python test.py


 
    .. py:method:: allreduce(tensor, c_op = "avg")
    
        Prend en charge la communication allreduce des données.

        :param tensor: Données d'entrée.
        :param c_op: Calcul.

        Exemples ::

            from pyvqnet.distributed import CommController
            from pyvqnet.tensor import tensor
            import numpy as np
            Comm_OP = CommController("mpi")

            num = tensor.to_tensor(np.random.rand(1, 5))
            print(f"rank {Comm_OP.getRank()}  {num}")

            Comm_OP.allreduce(num, "sum")
            print(f"rank {Comm_OP.getRank()}  {num}")
            # vqnetrun -n 2 python test.py

 
    .. py:method:: reduce(tensor, root = 0, c_op = "avg")
    
        Prend en charge la communication reduce des données.

        :param tensor: entrée.
        :param root: Spécifie le nœud vers lequel les données sont renvoyées.
        :param c_op: Calcul.

        Exemples ::

            from pyvqnet.distributed import CommController
            from pyvqnet.tensor import tensor
            import numpy as np
            Comm_OP = CommController("mpi")

            num = tensor.to_tensor(np.random.rand(1, 5))
            print(f"rank {Comm_OP.getRank()}  {num}")
            
            Comm_OP.reduce(num, 1)
            print(f"rank {Comm_OP.getRank()}  {num}")
            # vqnetrun -n 2 python test.py


 
    .. py:method:: broadcast(tensor, root = 0)
    
        Diffuse les données sur le processus racine spécifié vers tous les processus.

        :param tensor: entrée.
        :param root: Spécifie le nœud.

        Exemples ::

            from pyvqnet.distributed import CommController
            from pyvqnet.tensor import tensor
            import numpy as np
            Comm_OP = CommController("mpi")

            num = tensor.to_tensor(np.random.rand(1, 5))
            print(f"rank {Comm_OP.getRank()}  {num}")
            
            Comm_OP.broadcast(num, 1)
            print(f"rank {Comm_OP.getRank()}  {num}")
            # vqnetrun -n 2 python test.py

 
    .. py:method:: allgather(tensor)
    
        Rassemble les données de tous les processus.

        :param tensor: entrée.

        Exemples ::

            from pyvqnet.distributed import CommController
            from pyvqnet.tensor import tensor
            import numpy as np
            Comm_OP = CommController("mpi")

            num = tensor.to_tensor(np.random.rand(1, 5))
            print(f"rank {Comm_OP.getRank()}  {num}")

            num = Comm_OP.allgather(num)
            print(f"rank {Comm_OP.getRank()}  {num}")
            # vqnetrun -n 2 python test.py


    .. py:method:: send(tensor, dest)
    
        Interface de communication p2p.

        :param tensor: entrée.
        :param dest: Processus de destination.

        Exemples ::

            from pyvqnet.distributed import CommController,get_rank
            from pyvqnet.tensor import tensor
            import numpy as np
            Comm_OP = CommController("mpi")

            num = tensor.to_tensor(np.random.rand(1, 5))
            recv = tensor.zeros_like(num)

            if get_rank() == 0:
                Comm_OP.send(num, 1)
            elif get_rank() == 1:
                Comm_OP.recv(recv, 0)
            print(f"rank {Comm_OP.getRank()}  {num}")
            print(f"rank {Comm_OP.getRank()}  {recv}")
            
            # vqnetrun -n 2 python test.py

 
    .. py:method:: recv(tensor, source)
    
        Interface de communication p2p.

        :param tensor: entrée.
        :param source: Processus de réception.

        Exemples ::

            from pyvqnet.distributed import CommController,get_rank
            from pyvqnet.tensor import tensor
            import numpy as np
            Comm_OP = CommController("mpi")

            num = tensor.to_tensor(np.random.rand(1, 5))
            recv = tensor.zeros_like(num)

            if get_rank() == 0:
                Comm_OP.send(num, 1)
            elif get_rank() == 1:
                Comm_OP.recv(recv, 0)
            print(f"rank {Comm_OP.getRank()}  {num}")
            print(f"rank {Comm_OP.getRank()}  {recv}")
            
            # vqnetrun -n 2 python test.py


    .. py:method:: allreduce_group(tensor, c_op = "avg", group = None)
    
        Interface de communication allreduce de groupe.

        :param tensor: entrée.
        :param c_op: Calcul.
        :param group: Groupe de communication. Lors de l'utilisation du backend mpi, entrez le groupe généré par `init_groups` ou `split_groups` correspondant au groupe de communication. Lors de l'utilisation du backend nccl, entrez le numéro de groupe généré par `split_groups`.


        Exemples ::

            from pyvqnet.distributed import CommController,get_rank,get_local_rank
            from pyvqnet.tensor import tensor
            import numpy as np
            Comm_OP = CommController("nccl")

            groups = Comm_OP.split_groups([[0, 1]])

            complex_data = tensor.QTensor([3+1j, 2, 1 + get_rank()],dtype=8).reshape((3,1)).toGPU(1000+ get_local_rank())

            print(f"allreduce_group before rank {get_rank()}: {complex_data}")

            Comm_OP.allreduce_group(complex_data, c_op="sum",group = groups[0])
            print(f"allreduce_group after rank {get_rank()}: {complex_data}")
            # vqnetrun --backend nccl --nproc_per_node 2 python test.py


    .. py:method:: reduce_group(tensor, root = 0, c_op = "avg", group = None)
    
        Interface de communication reduce intra-groupe.

        :param tensor: Entrée.
        :param root: Spécifie le numéro de processus.
        :param c_op: Calcul.
        :param group: Groupe de communication. Lors de l'utilisation du backend mpi, entrez le groupe généré par `init_groups` ou `split_groups` correspondant au groupe de communication. Lors de l'utilisation du backend nccl, entrez le numéro de groupe généré par `split_groups`.

        Exemples ::
            
            from pyvqnet.distributed import CommController,get_rank,get_local_rank
            from pyvqnet.tensor import tensor
            import numpy as np
            Comm_OP = CommController("nccl")

            groups = Comm_OP.split_groups([[0, 1]])

            complex_data = tensor.QTensor([3+1j, 2, 1 + get_rank()],dtype=8).reshape((3,1)).toGPU(1000+ get_local_rank())

            print(f"reduce_group before rank {get_rank()}: {complex_data}")

            Comm_OP.reduce_group(complex_data, c_op="sum",group = groups[0])
            print(f"reduce_group after rank {get_rank()}: {complex_data}")
            # vqnetrun --backend nccl --nproc_per_node 2 python test.py


 
    .. py:method:: broadcast_group(tensor, root = 0, group = None)
    
        Interface de communication broadcast intra-groupe.

        :param tensor: Entrée.
        :param root: Spécifie le numéro de processus à partir duquel diffuser, par défaut : 0.
        :param group: Groupe de communication. Lors de l'utilisation du backend mpi, entrez le groupe généré par `init_groups` ou `split_groups` correspondant au groupe de communication. Lors de l'utilisation du backend nccl, entrez le numéro de groupe généré par `split_groups`.

        Exemples ::
            
            from pyvqnet.distributed import CommController,get_rank,get_local_rank
            from pyvqnet.tensor import tensor
            import numpy as np
            Comm_OP = CommController("nccl")

            groups = Comm_OP.split_groups([[0, 1]])

            complex_data = tensor.QTensor([3+1j, 2, 1 + get_rank()],dtype=8).reshape((3,1)).toGPU(1000+ get_local_rank())

            print(f"broadcast_group before rank {get_rank()}: {complex_data}")

            Comm_OP.broadcast_group(complex_data,group = groups[0])
            Comm_OP.barrier()
            print(f"broadcast_group after rank {get_rank()}: {complex_data}")
            # vqnetrun --backend nccl --nproc_per_node 2 python test.py


 
    .. py:method:: allgather_group(tensor, group = None)
    
        Interface de communication allgather de groupe.

        :param tensor: Entrée.
        :param group: Groupe de communication. Lors de l'utilisation du backend mpi, entrez le groupe généré par `init_groups` ou `split_groups` correspondant au groupe de communication. Lors de l'utilisation du backend nccl, entrez le numéro de groupe généré par `split_groups`.

        Exemples ::
            
            from pyvqnet.distributed import CommController,get_rank,get_local_rank
            from pyvqnet.tensor import tensor
            import numpy as np
            Comm_OP = CommController("nccl")

            groups = Comm_OP.split_groups([[0, 1]])

            complex_data = tensor.QTensor([3+1j, 2, 1 + get_rank()],dtype=8).reshape((3,1)).toGPU(1000+ get_local_rank())

            print(f"allgather_group before rank {get_rank()}: {complex_data}")

            complex_data = Comm_OP.allgather_group(complex_data,group = groups[0])
            print(f"allgather_group after rank {get_rank()}: {complex_data}")
            # vqnetrun --backend nccl --nproc_per_node 2 python test.py


 
    .. py:method:: grad_allreduce(optimizer)
    
        Met à jour le gradient des paramètres dans l'optimiseur avec allreduce.

        :param optimizer: optimiseur.

        Exemples ::
            
            from pyvqnet.distributed import CommController,get_rank,get_local_rank
            from pyvqnet.tensor import tensor
            from pyvqnet.nn.module import Module
            from pyvqnet.nn.linear import Linear
            from pyvqnet.nn.loss import MeanSquaredError
            from pyvqnet.optim import Adam
            from pyvqnet.nn import activation as F
            import pyvqnet
            import numpy as np
            Comm_OP = CommController("nccl")

            class Net(Module):
                def __init__(self):
                    super(Net, self).__init__()
                    self.fc = Linear(input_channels=5, output_channels=1)
                def forward(self, x):
                    x = F.ReLu()(self.fc(x))
                    return x
                
            model = Net().toGPU(1000+ get_local_rank())
            opti = Adam(model.parameters(), lr=0.01)
            actual = tensor.QTensor([1,1,1,1,1,0,0,0,0,0],dtype=pyvqnet.kfloat32).reshape((10,1)).toGPU(1000+ get_local_rank())
            x = tensor.randn((10, 5)).toGPU(1000+ get_local_rank())
            for i in range(10):
                opti.zero_grad()
                model.train()
                result = model(x)
                loss = MeanSquaredError()(actual, result)
                loss.backward()
                # print(f"rank {get_rank()} grad is {model.parameters()[0].grad} para {model.parameters()[0]}")
                Comm_OP.grad_allreduce(opti)
                # print(Comm_OP._allgather(model.parameters()[0]))
                if get_rank() == 0 :
                    print(f"rank {get_rank()} grad is {model.parameters()[0].grad} para {model.parameters()[0]} after")
                opti.step()
            Comm_OP.destroy()
            # vqnetrun --backend nccl --nproc_per_node 2 python test.py


 
    .. py:method:: param_allreduce(model)
    
        Met à jour les paramètres du modèle de manière allreduce.

        :param model: Modèle.

        Exemples ::
        
            from pyvqnet.distributed import CommController,get_rank,get_local_rank
            from pyvqnet.tensor import tensor
            from pyvqnet.nn.module import Module
            from pyvqnet.nn.linear import Linear
            from pyvqnet.nn import activation as F
            import numpy as np
            Comm_OP = CommController("nccl")

            class Net(Module):
                def __init__(self):
                    super(Net, self).__init__()
                    self.fc = Linear(input_channels=5, output_channels=1)
                def forward(self, x):
                    x = F.ReLu()(self.fc(x))
                    return x
                
            model = Net().toGPU(1000+ get_local_rank())
            print(f"rank {get_rank()} parameters is {model.parameters()}")
            Comm_OP.param_allreduce(model)
                
            if get_rank() == 0:
                print(model.parameters())


    .. py:method:: broadcast_model_params(model, root = 0)
    
        Diffuse les paramètres du modèle sur le numéro de processus spécifié.

        :param model: Modèles.
        :param root: Spécifie le numéro de processus.

        Exemples ::
        
            from pyvqnet.distributed import CommController,get_rank,get_local_rank
            from pyvqnet.tensor import tensor
            from pyvqnet.nn.module import Module
            from pyvqnet.nn.linear import Linear
            from pyvqnet.nn import activation as F
            import numpy as np
            Comm_OP = CommController("nccl")

            class Net(Module):
                def __init__(self):
                    super(Net, self).__init__()
                    self.fc = Linear(input_channels=5, output_channels=1)
                def forward(self, x):
                    x = F.ReLu()(self.fc(x))
                    return x
                
            model = Net().toGPU(1000+ get_local_rank())
            print(f"bcast before rank {get_rank()}:{model.parameters()}")
            Comm_OP.broadcast_model_params(model, 0)
            # model = model
            print(f"bcast after rank {get_rank()}: {model.parameters()}")


    .. py:method:: nccl_async_all_gather( output, input, group=None, async_op=False):

        Utilise NCCL pour un all_gather asynchrone ou synchrone sur les données GPU.

        :param output: QTensor - Le QTensor cible pour le résultat du all_gather.
        :param input: QTensor - Le QTensor à rassembler.
        :param group: Groupe de processus de communication, group est un tuple contenant les indices de groupe. Par défaut : None, aucun groupe n'est utilisé.
        :param async_op: Si cette opération est asynchrone, par défaut : False.
        :return: Work - Un gestionnaire de communication asynchrone. Utilisez wait() pour attendre la fin de cette opération.

        Exemples ::

            from pyvqnet import tensor
            import pyvqnet
            from pyvqnet.distributed import CommController,get_rank,get_local_rank
            Comm_OP = CommController("nccl")

            complex_data = tensor.QTensor([3+1j, 2, 1 + get_rank()],dtype=8).reshape((3,1)).toGPU(1000+ get_local_rank())

            out_data = tensor.empty([2,3,1],dtype=pyvqnet.kcomplex64).toGPU(pyvqnet.DEV_GPU_0+ get_local_rank())
            work = Comm_OP.nccl_async_all_gather(out_data, complex_data, group = None,async_op=True)
            work.wait()

    .. py:method:: nccl_async_all_reduce(tensor, c_op="avg",group=None, async_op=False):

        Utilise NCCL pour un allreduce asynchrone ou synchrone sur les données GPU.

        :param tensor: QTensor - Le QTensor qui doit être réduit.
        :param c_op: Méthode de calcul, peut être "sum" ou "avg", la valeur par défaut est "avg".
        :param group: Groupe de processus de communication, group est un tuple contenant les indices de groupe. Par défaut : None, aucun groupe n'est utilisé.
        :param async_op: Si cette opération est asynchrone, par défaut : False.
        :return: Work - Un gestionnaire de communication asynchrone. Utilisez wait() pour attendre la fin de cette opération.

        Exemples ::

            import pyvqnet
            from pyvqnet.tensor import tensor
            from pyvqnet.distributed.ControlComm import CommController,get_local_rank

            Comm_OP = CommController("nccl")
            complex_data = tensor.ones([500,500],dtype=pyvqnet.kcomplex64).toGPU(pyvqnet.DEV_GPU_0 + get_local_rank())
            work = Comm_OP.nccl_async_all_reduce(complex_data, "sum",group = None,async_op = True)
            work.wait()

    .. py:method:: nccl_async_reduce( tensor_, dest, c_op="avg", group=None, async_op=False ):

        Utilise NCCL pour un reduce asynchrone ou synchrone sur les données GPU.

        :param tensor_: QTensor - Le QTensor qui doit être réduit.
        :param dest: Le rang de destination du QTensor réduit.
        :param c_op: Méthode de calcul, peut être "sum" ou "avg", la valeur par défaut est "avg".
        :param group: Groupe de processus de communication, group est un tuple contenant les indices de groupe. Par défaut : None, aucun groupe n'est utilisé.
        :param async_op: Si cette opération est asynchrone, par défaut : False.
        :return: Work - Un gestionnaire de communication asynchrone. Utilisez wait() pour attendre la fin de cette opération.

        Exemples ::

            from pyvqnet import tensor
            import pyvqnet
            from pyvqnet.distributed import CommController,get_rank,get_local_rank
            Comm_OP = CommController("nccl")

            complex_data = tensor.ones([500,500],dtype=pyvqnet.kcomplex64).toGPU(pyvqnet.DEV_GPU_0 + get_local_rank())
            work = Comm_OP.nccl_async_reduce(complex_data, 0,"sum",group = None,async_op = True)
            work.wait()

    .. py:method:: nccl_async_broadcast(tensor, src, group=None, async_op=False )

        Utilise NCCL pour un broadcast asynchrone ou synchrone sur les données GPU.

        :param tensor: QTensor - Le QTensor qui doit être diffusé.
        :param src: Le rang source du QTensor diffusé.
        :param group: Groupe de processus de communication, group est un tuple contenant les indices de groupe. Par défaut : None, aucun groupe n'est utilisé.
        :param async_op: Si cette opération est asynchrone, par défaut : False.
        :return: Work - Un gestionnaire de communication asynchrone. Utilisez wait() pour attendre la fin de cette opération.

        Exemples ::

            import pyvqnet
            from pyvqnet.tensor import tensor
            from pyvqnet.distributed.ControlComm import CommController,get_local_rank
            Comm_OP = CommController("nccl")
            if get_local_rank() == 1:
                complex_data = tensor.ones([5,5],dtype=pyvqnet.kcomplex64).toGPU(pyvqnet.DEV_GPU_0+ get_local_rank())
            else:
                complex_data = tensor.zeros([5,5],dtype=pyvqnet.kcomplex64).toGPU(pyvqnet.DEV_GPU_0+ get_local_rank())
            work = Comm_OP.nccl_async_broadcast(complex_data, 1, group = None,async_op=True)

            work.wait()


    .. py:method:: nccl_async_send( t, dest, async_op=False ):

        Utilise NCCL pour un envoi P2P asynchrone ou synchrone sur les données GPU.

        :param t: QTensor - Le QTensor qui doit être envoyé.
        :param dest: Le rang de destination vers lequel envoyer le QTensor.
        :param async_op: Si cette opération est asynchrone, par défaut : False.
        :return: Work - Un gestionnaire de communication asynchrone. Utilisez wait() pour attendre la fin de cette opération.

        Exemples ::

            import pyvqnet
            from pyvqnet.tensor import tensor
            from pyvqnet.distributed.ControlComm import CommController,get_local_rank
            Comm_OP = CommController("nccl")

            if get_local_rank() == 0:
                complex_data = tensor.ones([5000,5000],dtype=pyvqnet.kcomplex64).toGPU(pyvqnet.DEV_GPU_0+ get_local_rank())*10
            else:
                complex_data = tensor.ones([5000,5000],dtype=pyvqnet.kcomplex64).toGPU(pyvqnet.DEV_GPU_0+ get_local_rank())

            if get_local_rank() == 0:
                work = Comm_OP.nccl_async_send(complex_data, 1 ,True)
            else:
                work = Comm_OP.nccl_async_recv(complex_data, 0 ,True)
            work.wait()

    .. py:method:: nccl_async_recv( t, src, async_op=False ):

        Utilise NCCL pour une réception P2P asynchrone ou synchrone sur les données GPU.

        :param t: QTensor - Le QTensor qui reçoit les données.
        :param src: Le rang source du QTensor reçu.
        :param async_op: Si cette opération est asynchrone, par défaut : False.
        :return: Work - Un gestionnaire de communication asynchrone. Utilisez wait() pour attendre la fin de cette opération.

        Exemples ::

            import pyvqnet
            from pyvqnet.tensor import tensor
            from pyvqnet.distributed.ControlComm import CommController,get_local_rank
            Comm_OP = CommController("nccl")

            if get_local_rank() == 0:
                complex_data = tensor.ones([5000,5000],dtype=pyvqnet.kcomplex64).toGPU(pyvqnet.DEV_GPU_0+ get_local_rank())*10
            else:
                complex_data = tensor.ones([5000,5000],dtype=pyvqnet.kcomplex64).toGPU(pyvqnet.DEV_GPU_0+ get_local_rank())

            if get_local_rank() == 0:
                work = Comm_OP.nccl_async_send(complex_data, 1 ,True)
            else:
                work = Comm_OP.nccl_async_recv(complex_data, 0 ,True)
            work.wait()


split_data
~~~~~~~~~~~~~~~~~~~~~~~~~~

Dans un contexte multi-processus, utilisez ``split_data`` pour découper les données en fonction du nombre de processus et renvoyer les données sur le processus correspondant.

.. py:function:: pyvqnet.distributed.datasplit.split_data(x_train, y_train, shuffle=False)

Définit les paramètres pour le calcul distribué.

    :param x_train: `np.array` - données d'entraînement.
    :param y_train: `np.array` - étiquettes des données d'entraînement.
    :param shuffle: `bool` - Indique s'il faut mélanger puis découper, par défaut est False.

    :return: données d'entraînement et étiquettes découpées.

    Exemple ::

        from pyvqnet.distributed import split_data
        import numpy as np

        x_train = np.random.randint(255, size = (100, 5))
        y_train = np.random.randint(2, size = (100, 1))

        x_train, y_train= split_data(x_train, y_train)

get_local_rank
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Utilisez ``get_local_rank`` pour obtenir le numéro de processus sur la machine courante.

.. py:function:: pyvqnet.distributed.ControlComm.get_local_rank()

    Utilisé pour obtenir le numéro de processus actuel sur la machine courante.

    :return: numéro de processus actuel sur la machine courante.

    Exemple ::

        from pyvqnet.distributed.ControlComm import get_local_rank

        print(get_local_rank())
        # vqnetrun -n 2 python test.py

get_rank
~~~~~~~~~~~~~~~~~~~~~~~~~~~
Utilisez ``get_rank`` pour obtenir le numéro de processus sur la machine courante.

.. py:function:: pyvqnet.distributed.ControlComm.get_rank()

    Utilisé pour obtenir le numéro de processus du processus actuel.

    :return: le numéro de processus du processus actuel.

    Exemple ::

        from pyvqnet.distributed.ControlComm import get_rank

        print(get_rank())
        # vqnetrun -n 2 python test.py

init_groups
~~~~~~~~~~~~~~~~~~~~~~~~~~~
Utilisez ``init_groups`` pour initialiser les groupes de processus basés sur cpu à partir de la liste de numéros de processus donnée.

.. py:function:: pyvqnet.distributed.ControlComm.init_groups(rank_lists)

    Utilisé pour initialiser le groupe de communication de processus pour le backend `mpi`.

    :param rank_lists: Liste des groupes de processus de communication.
    :return: Une liste de groupes de processus initialisés.

    Exemple ::

        from pyvqnet.distributed import *
        import numpy as np
        Comm_OP = CommController("mpi")
        num = tensor.to_tensor(np.random.rand(1, 5))
        print(f"rank {Comm_OP.getRank()}  {num} before allreduce")

        group_l = init_groups([[0,2], [1]])

        for comm_ in group_l:
            if Comm_OP.getRank() in comm_:
                Comm_OP.allreduce_group(num, "sum", group = comm_)
                print(f"rank {Comm_OP.getRank()}  {num} after allreduce")

        # vqnetrun -n 3 python test.py
        

PipelineParallelTrainingWrapper
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. py:class:: pyvqnet.distributed.pp.PipelineParallelTrainingWrapper(args,join_layers,trainset)
    
    Pipeline Parallel Training Wrapper implémente l'entraînement 1F1B. Disponible uniquement sur les plateformes Linux avec GPU.
    Plus de détails sur l'algorithme sont disponibles à l'adresse (https://www.deepspeed.ai/tutorials/pipeline/).

    :param args: Dictionnaire de paramètres. Voir les exemples.
    :param join_layers: Liste de modules Sequential.
    :param trainset: Jeu de données.

    :return:
        Instance de PipelineParallelTrainingWrapper.

    L'exemple suivant utilise le jeu de données CIFAR10 ``CIFAR10_Dataset`` pour entraîner la tâche de classification sur AlexNet sur 2 GPU.
    Dans cet exemple, il est divisé en deux processus pipeline parallèle ``pipeline_parallel_size`` = 2.
    La taille de lot est ``train_batch_size`` = 64, sur un seul GPU elle est ``train_micro_batch_size_per_gpu`` = 32.
    Les autres paramètres de configuration peuvent être trouvés dans ``args``.
    De plus, chaque processus doit configurer la variable d'environnement ``LOCAL_RANK`` dans la fonction ``__main__``.

    Exemples ::

        import os
        import pyvqnet

        from pyvqnet.nn import Module,Sequential,CrossEntropyLoss
        from pyvqnet.nn import Linear
        from pyvqnet.nn import Conv2D
        from pyvqnet.nn import activation as F
        from pyvqnet.nn import MaxPool2D
        from pyvqnet.nn import CrossEntropyLoss

        from pyvqnet.tensor import tensor
        from pyvqnet.distributed.pp import PipelineParallelTrainingWrapper
        from pyvqnet.distributed.configs import comm as dist
        from pyvqnet.distributed import *


        pipeline_parallel_size = 2

        num_steps = 1000

        def cifar_trainset_vqnet(local_rank, dl_path='./cifar10-data'):
            transform = pyvqnet.data.TransformCompose([
                pyvqnet.data.TransformResize(256),
                pyvqnet.data.TransformCenterCrop(224),
                pyvqnet.data.TransformToTensor(),
                pyvqnet.data.TransformNormalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])

            trainset = pyvqnet.data.CIFAR10_Dataset(root=dl_path,
                                                    mode="train",
                                                    transform=transform,layout="HWC")

            return trainset

        class Model(Module):
            def __init__(self):
                super(Model, self).__init__()
                self.features = Sequential( 
                Conv2D(input_channels=3, output_channels=8, kernel_size=(3, 3), stride=(1, 1), padding='same'),
                F.ReLu(),
                MaxPool2D([2, 2], [2, 2]),

                Conv2D(input_channels=8, output_channels=16, kernel_size=(3, 3), stride=(1, 1), padding='same'),
                F.ReLu(),
                MaxPool2D([2, 2], [2, 2]),

                Conv2D(input_channels=16, output_channels=32, kernel_size=(3, 3), stride=(1, 1), padding='same'),
                F.ReLu(),

                Conv2D(input_channels=32, output_channels=64, kernel_size=(3, 3), stride=(1, 1), padding='same'),
                F.ReLu(),

                Conv2D(input_channels=64, output_channels=64, kernel_size=(3, 3), stride=(1, 1), padding='same'),
                F.ReLu(),
                MaxPool2D([3, 3], [2, 2]),)
                
                self.cls = Sequential( 
                Linear(64 * 27 * 27, 512),
                F.ReLu(),

                Linear(512, 256),
                F.ReLu(),
                Linear(256, 10) )

            def forward(self, x):
                x = self.features(x)
                x = tensor.flatten(x,1)
                x = self.cls(x)

                return x
            
        def join_layers(vision_model):
            layers = [
                *vision_model.features,
                lambda x: tensor.flatten(x, 1),
                *vision_model.cls,
            ]
            return layers


        if __name__ == "__main__":


            args = {
            "backend":'nccl',  
            "train_batch_size" : 64,
            "train_micro_batch_size_per_gpu" : 32,
            "epochs":5,
        "optimizer": {
            "type": "Adam",
            "params": {
            "lr": 0.001
            }}, 
            "local_rank":dist.get_local_rank(), 
            "pipeline_parallel_size":pipeline_parallel_size, "seed":42, "steps":num_steps,
            "loss":CrossEntropyLoss(),
            }
            os.environ["LOCAL_RANK"] = str(dist.get_local_rank())
            trainset = cifar_trainset_vqnet(args["local_rank"])
            w = PipelineParallelTrainingWrapper(args,join_layers(Model()),trainset)

            all_loss = {}

            for i in range(args["epochs"]):
                w.train_batch()
                
            all_loss = w.loss_dict


ZeroModelInitial
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. py:class:: pyvqnet.distributed.ZeroModelInitial(args,model,optimizer)
    
    Interface de l'API Zero1, actuellement uniquement pour plateforme Linux basée sur le calcul parallèle GPU.

    :param args: dictionnaire de paramètres.
    :param model: Module.
    :param optimizer: Optimiseur.

    :return:
        Moteur Zero1.

L'exemple suivant utilise le jeu de données MNIST pour entraîner une tâche de classification sur un modèle MLP sur 2 GPU.

    La taille de lot est ``train_batch_size`` = 64, et l'étape ``stage`` de ``zero_optimization`` est définie à 1.
    Si Optimizer est None, la configuration de ``optimizer`` dans ``args`` est utilisée. Les autres paramètres de configuration peuvent être trouvés dans ``args``.
    
    De plus, chaque processus doit être configuré avec la variable d'environnement ``LOCAL_RANK``.
    
    .. code-block::

        os.environ["LOCAL_RANK"] = str(dist.get_local_rank())

    Exemples ::

        from pyvqnet.distributed import *
        from pyvqnet import *
        from time import time
        import pyvqnet.optim as optim
        import pyvqnet.nn as nn
        import pyvqnet
        import sys
        import pyvqnet 
        import numpy as np
        import os
        import struct

        def load_mnist(dataset="training_data",
                    digits=np.arange(2),
                    path="./"):
            """
            charge les données mnist
            """
            from array import array as pyarray
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
            images = np.zeros((num, rows, cols),dtype=np.float32)

            labels = np.zeros((num, 1), dtype=int)
            for i in range(len(ind)):
                images[i] = np.array(img[ind[i] * rows * cols:(ind[i] + 1) * rows *
                                        cols]).reshape((rows, cols))
                labels[i] = lbl[ind[i]]

            return images, labels


        train_images_np, train_labels_np = load_mnist(dataset="training_data", digits=np.arange(10),path="../data/MNIST_data/")
        train_images_np = train_images_np / 255.

        test_images_np, test_labels_np = load_mnist(dataset="testing_data", digits=np.arange(10),path="../data/MNIST_data/")
        test_images_np = test_images_np / 255.

        local_rank = pyvqnet.distributed.get_rank()

        from pyvqnet.distributed import ZeroModelInitial

        class MNISTClassifier(nn.Module):
            
            def __init__(self):
                super(MNISTClassifier, self).__init__()
                self.fc1 = nn.Linear(28*28, 512)
                self.fc2 = nn.Linear(512, 256)
                self.fc3 = nn.Linear(256, 128)
                self.fc4 = nn.Linear(128, 64)
                self.fc5 = nn.Linear(64, 10)
                self.ac = nn.activation.ReLu()
                
            def forward(self, x:pyvqnet.QTensor):
                
                x = x.reshape([-1, 28*28])  
                x = self.ac(self.fc1(x))
                x = self.fc2(x)
                x = self.fc3(x)
                x = self.fc4(x)
                x = self.fc5(x)
                return x
        
        model = MNISTClassifier()

        model.to(local_rank + 1000)
            
        Comm_op = CommController("nccl")
        Comm_op.broadcast_model_params(model, 0)

        batch_size = 64

        criterion = nn.CrossEntropyLoss()  
        optimizer = optim.Adam(model.parameters(), lr=0.001) 

        args_ = {
                "train_batch_size": batch_size, 
                "optimizer": {
                    "type": "adam",
                    "params": {
                    "lr": 0.001,
                    }
                },
                "zero_optimization": {
                    "stage": 1, 
                }    
            }

        os.environ["LOCAL_RANK"] = str(get_local_rank())
        model = ZeroModelInitial(args=args_, model=model, optimizer=optimizer) 

        def compute_acc(outputs, labels, correct, total):
            predicted = pyvqnet.tensor.argmax(outputs, dim=1, keepdims=True)
            total += labels.size
            correct += pyvqnet.tensor.sums(predicted == labels).item()
            return correct, total

        train_acc = 0
        test_acc = 0
        epochs = 5
        loss = 0
        time1 = time()

        for epoch in range(epochs):
            model.train()
            total = 0
            correct = 0
            step = 0
            
            num_batches = (train_images_np.shape[0] + batch_size - 1) // batch_size
            
            for i in range(num_batches):
                
                data_ = tensor.QTensor(train_images_np[i*batch_size: (i+1) * batch_size,:], dtype = kfloat32)
                labels = tensor.QTensor(train_labels_np[i*batch_size: (i+1) * batch_size,:], dtype = kint64)
                    
                data_ = data_.to(local_rank + 1000)
                labels = labels.to(local_rank + 1000)
                
                outputs = model(data_)
                loss = criterion(labels, outputs)
                
                model.backward(loss) 
                model.step() 

                correct, total = compute_acc(outputs, labels, correct, total)
                step += 1
                if step % 50 == 0:
                    print(f"Train : rank {get_rank()} Epoch [{epoch+1}/{epochs}], step {step} Loss: {loss.item():.4f} acc {100 * correct / total}")
                    sys.stdout.flush()
                    
            train_acc = 100 * correct / total
            
        time2 = time()
        print(f'Accuracy of the model on the 10000 Train images: {train_acc}% time cost {time2 - time1}')


ColumnParallelLinear
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. py:class:: pyvqnet.distributed.ColumnParallelLinear(input_size,output_size,weight_initializer,bias_initializer,use_bias,dtype,name,tp_comm)
    
    Calcul tensor-parallèle avec couche linéaire parallèle en colonnes
    
    La couche linéaire est définie comme Y = XA + b. Ses lignes parallèles 2D sont A = [A_1, ... , A_p].

    :param input_size: première dimension de la matrice A.
    :param output_size: deuxième dimension de la matrice A.
    :param weight_initializer: `callable` - par défaut normal.
    :param bias_initializer: `callable` - par défaut zeros.
    :param use_bias: `bool` - par défaut True.
    :param dtype: par défaut : None, utilise le type de données par défaut.
    :param name: nom du module, par défaut : "".
    :param tp_comm: Contrôleur de communication.

    L'exemple suivant utilise le jeu de données MNIST pour entraîner une tâche de classification sur un modèle MLP sur 2 GPU.

    L'utilisation est similaire à celle de la couche Linear classique.

    Utilisation multi-processus basée sur `# vqnetrun --backend nccl --nproc_per_node 2 python test.py`.

    Exemples ::

        import pyvqnet.distributed
        import pyvqnet.optim as optim
        import pyvqnet.nn as nn
        import pyvqnet
        import sys
        from pyvqnet.distributed.tensor_parallel import ColumnParallelLinear, RowParallelLinear
        from pyvqnet.distributed import *
        from time import time

        import pyvqnet 
        import numpy as np
        import os
        from pyvqnet import *
        import pytest

        Comm_OP = CommController("nccl")

        import struct
        def load_mnist(dataset="training_data",
                    digits=np.arange(2),
                    path="./"):
            """
            charge les données mnist
            """
            from array import array as pyarray
            # download_mnist(path)
            if dataset == "training_data":
                fname_image = os.path.join(path, "train-images-idx3-ubyte").replace(
                    "\\", "/")
                fname_label = os.path.join(path, "train-labels-idx1-ubyte").replace(
                    "\\", "/")
            elif dataset == "testing_data":
                fname_image = os.path.join(path, "t10k-images-idx3-ubyte").replace(
                    "\\", "/")
                fname_label = os.path.join(path, "t10k-labels-idx1-ubyte").replace(
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
            images = np.zeros((num, rows, cols),dtype=np.float32)

            labels = np.zeros((num, 1), dtype=int)
            for i in range(len(ind)):
                images[i] = np.array(img[ind[i] * rows * cols:(ind[i] + 1) * rows *
                                        cols]).reshape((rows, cols))
                labels[i] = lbl[ind[i]]

            return images, labels

        train_images_np, train_labels_np = load_mnist(dataset="training_data", digits=np.arange(10),path="./data/MNIST/raw/")
        train_images_np = train_images_np / 255.

        test_images_np, test_labels_np = load_mnist(dataset="testing_data", digits=np.arange(10),path="./data/MNIST/raw/")
        test_images_np = test_images_np / 255.

        local_rank = pyvqnet.distributed.get_rank()

        class MNISTClassifier(nn.Module):
            def __init__(self):
                super(MNISTClassifier, self).__init__()
                self.fc1 = RowParallelLinear(28*28, 512, tp_comm = Comm_OP)
                self.fc2 = ColumnParallelLinear(512, 256, tp_comm = Comm_OP)
                self.fc3 = RowParallelLinear(256, 128, tp_comm = Comm_OP)
                self.fc4 = ColumnParallelLinear(128, 64, tp_comm = Comm_OP)
                self.fc5 = RowParallelLinear(64, 10, tp_comm = Comm_OP)  
                self.ac = nn.activation.ReLu()
                
            def forward(self, x:pyvqnet.QTensor):
                
                x = x.reshape([-1, 28*28])  
                x = self.ac(self.fc1(x))
                x = self.fc2(x)
                x = self.fc3(x)
                x = self.fc4(x)
                x = self.fc5(x)
                return x
            
        
        model = MNISTClassifier()
        total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        model.to(local_rank + 1000)

        Comm_OP.broadcast_model_params(model, 0)

        criterion = nn.CrossEntropyLoss() 
        optimizer = optim.Adam(model.parameters(), lr=0.001)

        def compute_acc(outputs, labels, correct, total):
            predicted = pyvqnet.tensor.argmax(outputs, dim=1, keepdims=True)
            total += labels.size
            correct += pyvqnet.tensor.sums(predicted == labels).item()
            return correct, total

        train_acc = 0
        test_acc = 0
        epochs = 5
        loss = 0

        time1 = time()
        for epoch in range(epochs):
            model.train()
            total = 0
            correct = 0
            step = 0
            
            batch_size = 64
            num_batches = (train_images_np.shape[0] + batch_size - 1) // batch_size
            
            for i in range(num_batches):
                data_ = tensor.QTensor(train_images_np[i*batch_size: (i+1) * batch_size,:], dtype = kfloat32)
                labels = tensor.QTensor(train_labels_np[i*batch_size: (i+1) * batch_size,:], dtype = kint64)

                data_ = data_.to(local_rank + 1000)
                labels = labels.to(local_rank + 1000)

                optimizer.zero_grad()

                outputs = model(data_)
                loss = criterion(labels, outputs)

                loss.backward()
                optimizer.step()

                correct, total = compute_acc(outputs, labels, correct, total)
                step += 1
                if step % 50 == 0:
                    print(f"Train : rank {get_rank()} Epoch [{epoch+1}/{epochs}], step {step} Loss: {loss.item():.4f} acc {100 * correct / total}")
                    sys.stdout.flush()

            train_acc = 100 * correct / total
        time2 = time()

        print(f'Accuracy of the model on the 10000 Train images: {train_acc}% time cost {time2 - time1}')


RowParallelLinear
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. py:class:: pyvqnet.distributed.RowParallelLinear(input_size,output_size,weight_initializer,bias_initializer,use_bias,dtype,name,tp_comm)
    
    Calcul tensor-parallèle avec couche linéaire parallèle en lignes.

    La couche linéaire est définie comme Y = XA + b. A est parallélisé le long de sa première dimension et X le long de sa deuxième dimension.
    A = transpose([A_1 .. A_p]) X = [X_1, ..., X_p].

    :param input_size: première dimension de la matrice A.
    :param output_size: deuxième dimension de la matrice A.
    :param weight_initializer: `callable` - par défaut normal.
    :param bias_initializer: `callable` - par défaut zeros.
    :param use_bias: `bool` - par défaut True.
    :param dtype: par défaut : None, utilise le type de données par défaut.
    :param name: nom du module, par défaut : "".
    :param tp_comm: Contrôleur de communication.

    L'exemple suivant utilise le jeu de données MNIST pour entraîner une tâche de classification sur un modèle MLP sur 2 GPU.

    L'utilisation est similaire à celle de la couche Linear classique.

    Utilisation multi-processus basée sur `# vqnetrun --backend nccl --nproc_per_node 2 python test.py`.

    Exemples ::

        import pyvqnet.distributed
        import pyvqnet.optim as optim
        import pyvqnet.nn as nn
        import pyvqnet
        import sys
        from pyvqnet.distributed.tensor_parallel import ColumnParallelLinear, RowParallelLinear
        from pyvqnet.distributed import *
        from time import time

        import pyvqnet 
        import numpy as np
        import os
        from pyvqnet import *
        import pytest

        Comm_OP = CommController("nccl")

        import struct
        def load_mnist(dataset="training_data",
                    digits=np.arange(2),
                    path="./"):
            """
            charge les données mnist
            """
            from array import array as pyarray
            # download_mnist(path)
            if dataset == "training_data":
                fname_image = os.path.join(path, "train-images-idx3-ubyte").replace(
                    "\\", "/")
                fname_label = os.path.join(path, "train-labels-idx1-ubyte").replace(
                    "\\", "/")
            elif dataset == "testing_data":
                fname_image = os.path.join(path, "t10k-images-idx3-ubyte").replace(
                    "\\", "/")
                fname_label = os.path.join(path, "t10k-labels-idx1-ubyte").replace(
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
            images = np.zeros((num, rows, cols),dtype=np.float32)

            labels = np.zeros((num, 1), dtype=int)
            for i in range(len(ind)):
                images[i] = np.array(img[ind[i] * rows * cols:(ind[i] + 1) * rows *
                                        cols]).reshape((rows, cols))
                labels[i] = lbl[ind[i]]

            return images, labels

        train_images_np, train_labels_np = load_mnist(dataset="training_data", digits=np.arange(10),path="./data/MNIST/raw/")
        train_images_np = train_images_np / 255.

        test_images_np, test_labels_np = load_mnist(dataset="testing_data", digits=np.arange(10),path="./data/MNIST/raw/")
        test_images_np = test_images_np / 255.

        local_rank = pyvqnet.distributed.get_rank()

        class MNISTClassifier(nn.Module):
            def __init__(self):
                super(MNISTClassifier, self).__init__()
                self.fc1 = RowParallelLinear(28*28, 512, tp_comm = Comm_OP)
                self.fc2 = ColumnParallelLinear(512, 256, tp_comm = Comm_OP)
                self.fc3 = RowParallelLinear(256, 128, tp_comm = Comm_OP)
                self.fc4 = ColumnParallelLinear(128, 64, tp_comm = Comm_OP)
                self.fc5 = RowParallelLinear(64, 10, tp_comm = Comm_OP)  
                self.ac = nn.activation.ReLu()
                
                
            def forward(self, x:pyvqnet.QTensor):
                
                x = x.reshape([-1, 28*28])  
                x = self.ac(self.fc1(x))
                x = self.fc2(x)
                x = self.fc3(x)
                x = self.fc4(x)
                x = self.fc5(x)
                return x
            
        model = MNISTClassifier()
        total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        model.to(local_rank + 1000)
        Comm_OP.broadcast_model_params(model, 0)

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)

        def compute_acc(outputs, labels, correct, total):
            predicted = pyvqnet.tensor.argmax(outputs, dim=1, keepdims=True)
            total += labels.size
            correct += pyvqnet.tensor.sums(predicted == labels).item()
            return correct, total

        train_acc = 0
        test_acc = 0
        epochs = 5
        loss = 0

        time1 = time()
        for epoch in range(epochs):
            model.train()
            total = 0
            correct = 0
            step = 0
            
            batch_size = 64
            num_batches = (train_images_np.shape[0] + batch_size - 1) // batch_size
            
            for i in range(num_batches):
                data_ = tensor.QTensor(train_images_np[i*batch_size: (i+1) * batch_size,:], dtype = kfloat32)
                labels = tensor.QTensor(train_labels_np[i*batch_size: (i+1) * batch_size,:], dtype = kint64)

                data_ = data_.to(local_rank + 1000)
                labels = labels.to(local_rank + 1000)

                optimizer.zero_grad()

                outputs = model(data_)
                loss = criterion(labels, outputs)

                loss.backward()
                optimizer.step()

                correct, total = compute_acc(outputs, labels, correct, total)
                step += 1
                if step % 50 == 0:
                    print(f"Train : rank {get_rank()} Epoch [{epoch+1}/{epochs}], step {step} Loss: {loss.item():.4f} acc {100 * correct / total}")
                    sys.stdout.flush()

            train_acc = 100 * correct / total
        time2 = time()

        print(f'Accuracy of the model on the 10000 Train images: {train_acc}% time cost {time2 - time1}')


Réorganisation des bits
=================================

La réorganisation des qubits est une technique de parallélisme de bits. Son objectif principal est de réduire le nombre de transformations de bits requises par le parallélisme de bits en modifiant l'ordre des portes logiques quantiques. Les modules suivants sont nécessaires pour construire des circuits quantiques à grand nombre de bits basés sur le parallélisme de bits. Référez-vous à l'article `Lazy Qubit Reordering for Accelerating Parallel State-Vector-based Quantum Circuit Simulation <https://export.arxiv.org/abs/2410.04252>`__.

Les interfaces suivantes nécessitent `mpi` pour lancer plusieurs processus de calcul.

DistributeQMachine
^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. py:class:: pyvqnet.distributed.qubits_reorder.DistributeQMachine(num_wires, dtype, grad_mode)
    
    Une classe pour simuler des calculs quantiques variationnels en parallélisme de bits, incluant les états quantiques sur un sous-ensemble de bits sur chaque nœud. Chaque nœud demande une simulation de circuit quantique variationnel distribué via MPI. La valeur de N doit être égale à une puissance de 2 élevée au nombre de bits parallèles distribués, `global_qubit`, et peut être configurée via `set_qr_config`.

    :param num_wires: Le nombre de bits dans le circuit quantique complet.
    :param dtype: Le type de données des données de calcul. La valeur par défaut est pyvqnet.kcomplex64, correspondant à la précision des paramètres de pyvqnet.kfloat32.
    :param grad_mode: Définir sur adjoint lors de la rétropropagation ``DistQuantumLayerAdjoint``.

    .. note::

        Le nombre de bits saisi est le nombre de bits requis pour l'ensemble du circuit quantique. DistributeQMachine construira un simulateur quantique basé sur le nombre global de bits, qui est ``num_wires - global_qubit``.

        La rétropropagation doit être basée sur ``DistQuantumLayerAdjoint``.

    .. warning::

        Cette interface ne fonctionne que sous Linux ;

        Les paramètres de parallélisme de bits dans ``DistributeQMachine`` doivent être configurés, comme illustré dans l'exemple, notamment :

        .. code-block::

            qm.set_just_defined(True)
            qm.set_save_op_history_flag(True) # active la sauvegarde des opérations
            qm.set_qr_config({'qubit': nombre total de qubits, 'global_qubit': nombre de qubits distribués})
    
    
    Exemples ::

        from pyvqnet.distributed import get_rank
        from pyvqnet import tensor
        from pyvqnet.qnn.vqc import rx, ry, cnot, MeasureAll,rz
        import pyvqnet
        from pyvqnet.distributed.qubits_reorder import DistributeQMachine,DistQuantumLayerAdjoint
        pyvqnet.utils.set_random_seed(123)


        qubit = 10
        batch_size = 5

        class QModel(pyvqnet.nn.Module):
            def __init__(self, num_wires, dtype, grad_mode=""):
                super(QModel, self).__init__()

                self._num_wires = num_wires
                self._dtype = dtype
                self.qm = DistributeQMachine(num_wires, dtype=dtype, grad_mode=grad_mode)
                
                self.qm.set_just_defined(True)
                self.qm.set_save_op_history_flag(True) # active la sauvegarde des opérations
                self.qm.set_qr_config({"qubit": num_wires, # active la réorganisation des qubits, configure les paramètres
                                        "global_qubit": 1}) # global_qubit == log2(nproc)
                
                self.params = pyvqnet.nn.Parameter([qubit])

                self.measure = MeasureAll(obs={
                    "X5":1.0
                })

            def forward(self, x, *args, **kwargs):
                self.qm.reset_states(x.shape[0])

                for i in range(qubit):
                    rx(q_machine=self.qm, params=self.params[i], wires=i)
                    ry(q_machine=self.qm, params=self.params[3], wires=i)
                    rz(q_machine=self.qm, params=self.params[4], wires=i)
                cnot(q_machine=self.qm,  wires=[0, 1])
                rlt = self.measure(q_machine=self.qm)
                return rlt


        input_x = tensor.QTensor([[0.1, 0.2, 0.3]], requires_grad=True).toGPU(pyvqnet.DEV_GPU_0+get_rank())

        input_x = tensor.broadcast_to(input_x, [2, 3])

        input_x.requires_grad = True

        quantum_model = QModel(num_wires=qubit,
                            dtype=pyvqnet.kcomplex64,
                            grad_mode="adjoint").toGPU(pyvqnet.DEV_GPU_0+get_rank())

        adjoint_model = DistQuantumLayerAdjoint(quantum_model)
        adjoint_model.train()

        batch_y = adjoint_model(input_x)
        batch_y.backward()

        print(batch_y)
        # mpirun -n 2 python test.py

DistQuantumLayerAdjoint
^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. py:class:: pyvqnet.distributed.qubits_reorder.DistQuantumLayerAdjoint(vqc_module,name)
    
    Une couche DistQuantumLayer qui calcule les gradients pour les paramètres dans les calculs en parallélisme de bits en utilisant l'approche de la matrice adjointe.

    :param vqc_module: Le module ``DistributeQMachine`` implicite d'entrée.
    :param name: Le nom du module.

    .. note::

        Le module vqc_module d'entrée doit contenir ``DistributeQMachine``. Les calculs de gradient par rétropropagation adjointe sont effectués sur la base de ``DistributeQMachine`` dans les calculs en parallélisme de bits.

    .. warning::

        Cette interface n'est supportée que sous Linux ;


    Exemples ::

        from pyvqnet.distributed import get_rank
        from pyvqnet import tensor
        from pyvqnet.qnn.vqc import rx, ry, cnot, MeasureAll,rz
        import pyvqnet
        from pyvqnet.distributed.qubits_reorder import DistributeQMachine,DistQuantumLayerAdjoint
        pyvqnet.utils.set_random_seed(123)


        qubit = 10
        batch_size = 5

        class QModel(pyvqnet.nn.Module):
            def __init__(self, num_wires, dtype, grad_mode=""):
                super(QModel, self).__init__()

                self._num_wires = num_wires
                self._dtype = dtype
                self.qm = DistributeQMachine(num_wires, dtype=dtype, grad_mode=grad_mode)
                
                self.qm.set_just_defined(True)
                self.qm.set_save_op_history_flag(True) # active la sauvegarde des opérations
                self.qm.set_qr_config({"qubit": num_wires, # active la réorganisation des qubits, configure les paramètres
                                            "global_qubit": 1}) # global_qubit == log2(nproc)
                
                self.params = pyvqnet.nn.Parameter([qubit])

                self.measure = MeasureAll(obs={
                    "X5":1.0
                })

            def forward(self, x, *args, **kwargs):
                self.qm.reset_states(x.shape[0])

                for i in range(qubit):
                    rx(q_machine=self.qm, params=self.params[i], wires=i)
                    ry(q_machine=self.qm, params=self.params[3], wires=i)
                    rz(q_machine=self.qm, params=self.params[4], wires=i)
                cnot(q_machine=self.qm,  wires=[0, 1])
                rlt = self.measure(q_machine=self.qm)
                return rlt


        input_x = tensor.QTensor([[0.1, 0.2, 0.3]], requires_grad=True).toGPU(pyvqnet.DEV_GPU_0+get_rank())

        input_x = tensor.broadcast_to(input_x, [2, 3])

        input_x.requires_grad = True

        quantum_model = QModel(num_wires=qubit,
                            dtype=pyvqnet.kcomplex64,
                            grad_mode="adjoint").toGPU(pyvqnet.DEV_GPU_0+get_rank())

        adjoint_model = DistQuantumLayerAdjoint(quantum_model)
        adjoint_model.train()

        batch_y = adjoint_model(input_x)
        batch_y.backward()

        print(batch_y)
        # mpirun -n 2 python test.py
