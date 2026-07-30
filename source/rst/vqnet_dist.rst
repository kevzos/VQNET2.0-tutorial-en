
.. _vqnet_dist:

Modulo di Calcolo Distribuito VQNet
*********************************************************

Configurazione dell'Ambiente
=================================

Questa sezione descrive come configurare l'ambiente VQNet su Linux per il calcolo distribuito su CPU e GPU.

Installazione di MPI
^^^^^^^^^^^^^^^^^^^^^^

MPI e' una libreria comune per la comunicazione inter-CPU e la funzione di calcolo distribuito della CPU in VQNet e' implementata su MPI. 
La seguente sezione descrive come installare MPI su un sistema Linux (al momento, la funzione di calcolo distribuito basata su CPU e' supportata solo su Linux).

Verificare se i compilatori gcc e gfortran sono installati.

.. code-block::
        
    which gcc 
    which gfortran

Quando vengono mostrati i percorsi di gcc e gfortran, si puo' procedere al passo successivo dell'installazione. Se non si dispone dei compilatori corrispondenti, 
installarli prima. Una volta verificati i compilatori, usare il comando wget per scaricare mpich.

.. code-block::
        
    wget http://www.mpich.org/static/downloads/3.3.2/mpich-3.3.2.tar.gz 
    tar -zxvf mpich-3.3.2.tar.gz 
    cd mpich-3.3.2 
    ./configure --prefix=/usr/local/mpich
    make 
    make install 

Dopo aver compilato e installato mpich, configurare le sue variabili d'ambiente.

.. code-block::
        
    vim ~/.bashrc

    # Alla fine del documento, aggiungere
    export PATH="/usr/local/mpich/bin:$PATH"

Dopo aver salvato e chiuso, usare source per applicare le modifiche.

.. code-block::

    source ~/.bashrc

Usare which per verificare che le variabili d'ambiente siano configurate correttamente. Se il percorso viene visualizzato, l'installazione e' stata completata con successo.

Inoltre, e' possibile installare mpi4py tramite pip install. Se si incontra il seguente errore:

.. image:: ./images/mpi_bug.png
    :align: center

|

Per risolvere l'incompatibilita' tra mpi4py e la versione di Python, si puo' procedere come segue:

.. code-block::

    # Eseguire il backup del compilatore per l'ambiente Python corrente
    pushd /root/anaconda3/envs/$CONDA_DEFAULT_ENV/compiler_compat && mv ld ld.bak && popd

    # Re-installare mpi4py
    pip install mpi4py

    # Ripristinare il compilatore originale
    pushd /root/anaconda3/envs/$CONDA_DEFAULT_ENV/compiler_compat && mv ld.bak ld && popd

Installazione di NCCL
^^^^^^^^^^^^^^^^^^^^^^

NCCL e' una libreria comune per la comunicazione GPU e la funzione di calcolo distribuito delle GPU in VQNet e' implementata su NCCL.
Questo software installa la libreria a collegamento dinamico di NCCL per impostazione predefinita al momento dell'installazione, quindi generalmente non e' necessario installare NCCL separatamente.
Questa sezione richiede il supporto MPI, quindi e' necessario configurare anche l'ambiente MPI.

Avvio distribuito
=================================
 
Utilizzando l'interfaccia di calcolo distribuito avviata dal comando ``vqnetrun``, i parametri di ``vqnetrun`` sono descritti di seguito.

n, np
^^^^^^^^^^^^^^^^^^^^^^

L'interfaccia ``vqnetrun`` consente di controllare il numero di processi avviati con i parametri ``-n``, ``-np``, come mostrato nell'esempio seguente.

    Example::

        from pyvqnet.distributed import CommController
        Comm_OP = CommController("mpi") # inizializza il controllore mpi
        
        rank = Comm_OP.getRank()
        size = Comm_OP.getSize()
        print(f"rank: {rank}, size {size}")

        # vqnetrun -n 2 python test.py
        # vqnetrun -np 2 python test.py

backend
^^^^^^^^^^^^^^^^^^^^^^

L'interfaccia ``vqnetrun`` consente di selezionare il backend distribuito con il parametro ``--backend``, supportando le modalita' ``mpi`` (predefinita) e ``nccl``.
La modalita' MPI e' per il calcolo distribuito su CPU, mentre la modalita' NCCL e' per il calcolo distribuito su GPU.

    Example::

        from pyvqnet.distributed import CommController
        Comm_OP = CommController("nccl")

        rank = Comm_OP.getRank()
        size = Comm_OP.getSize()
        print(f"rank: {rank}, size {size}")

        # vqnetrun --backend nccl --nproc_per_node 2 python test.py

nproc_per_node
^^^^^^^^^^^^^^^^^^^^^^

L'interfaccia ``vqnetrun`` consente di controllare il numero di processi avviati su ogni nodo con il parametro ``--nproc_per_node``, disponibile solo in modalita' NCCL.

    Example::

        from pyvqnet.distributed import CommController
        Comm_OP = CommController("nccl")

        rank = Comm_OP.getRank()
        size = Comm_OP.getSize()
        print(f"rank: {rank}, size {size}")

        # vqnetrun --backend nccl --nproc_per_node 4 python test.py

output-filename
^^^^^^^^^^^^^^^^^^^^^^

L'interfaccia ``vqnetrun`` consente di salvare l'output in un file specificato con il parametro da riga di comando ``--output-filename``.

Un esempio di implementazione e' il seguente:

    Example::

        from pyvqnet.distributed import CommController, get_host_name
        Comm_OP = CommController("mpi") # inizializza il controllore mpi
        
        rank = Comm_OP.getRank()
        size = Comm_OP.getSize()
        print(f"rank: {rank}, size {size}")
        print(f"LocalRank {Comm_OP.getLocalRank()} hosts name {get_host_name()}")

        # vqnetrun -np 4 --hostfile hosts --output-filename output  python test.py


verbose
^^^^^^^^^^^^^^^^^^^^^^

L'interfaccia ``vqnetrun`` puo' essere usata con il parametro da riga di comando ``--verbose`` per abilitare la registrazione dettagliata della comunicazione e visualizzare i risultati.

Un esempio di implementazione e' il seguente

    Example::

        from pyvqnet.distributed import CommController, get_host_name
        Comm_OP = CommController("mpi") # inizializza il controllore mpi
        
        rank = Comm_OP.getRank()
        size = Comm_OP.getSize()
        print(f"rank: {rank}, size {size}")
        print(f"LocalRank {Comm_OP.getLocalRank()} hosts name {get_host_name()}")

        # vqnetrun -np 4 --hostfile hosts --verbose python test.py


start-timeout
^^^^^^^^^^^^^^^^^^^^^^

L'interfaccia ``vqnetrun`` puo' essere usata con il parametro da riga di comando ``-start-timeout`` per specificare che tutti i controlli vengano eseguiti e il processo venga avviato prima del timeout. Il valore predefinito e' 30 secondi.

Un esempio di implementazione e' il seguente

    Example::

        from pyvqnet.distributed import CommController, get_host_name
        Comm_OP = CommController("mpi") # inizializza il controllore mpi
        
        rank = Comm_OP.getRank()
        size = Comm_OP.getSize()
        print(f"rank: {rank}, size {size}")
        print(f"LocalRank {Comm_OP.getLocalRank()} hosts name {get_host_name()}")

        # vqnetrun -np 4 --start-timeout 10 python test.py


CommController
=================================

    Il calcolo distribuito serve a controllare la comunicazione dei dati tra diversi processi su CPU e GPU. Genera diversi controllori per CPU (MPI) e GPU (NCCL) e chiama i metodi di comunicazione per completare la comunicazione e la sincronizzazione dei dati tra diversi processi.

__init__
^^^^^^^^^^^^^^^^^^^^^^
.. py:class:: pyvqnet.distributed.ControlComm.CommController(backend,rank=None,world_size=None)
    
    CommController viene usato per controllare la comunicazione dei dati su CPU e GPU. Impostando il parametro `backend`, genera il controllore per CPU (MPI) o GPU (NCCL). (Attualmente, la funzione di calcolo distribuito supporta solo Linux.)

    :param backend: utilizzato per generare il controllore della comunicazione dati per cpu o gpu.
    :param rank: Questo parametro e' utile solo nei backend non pyvqnet, il valore predefinito e': None.
    :param world_size: Questo parametro e' utile solo nei backend non pyvqnet, il valore predefinito e': None.
        
    :return:
        Istanza di CommController.

    Examples::

        from pyvqnet.distributed import CommController
        Comm_OP = CommController("nccl") # inizializza il controllore nccl

        # Comm_OP = CommController("mpi") # inizializza il controllore mpi

 
    .. py:method:: getRank()
        
        Utilizzato per ottenere il numero di processo del processo corrente.

        :return: Restituisce il numero di processo del processo corrente.

        Examples::

            from pyvqnet.distributed import CommController
            Comm_OP = CommController("nccl") # inizializza il controllore nccl
            
            Comm_OP.getRank()

 
    .. py:method:: getSize()
    
        Utilizzato per ottenere il numero totale di processi avviati.


        :return: Restituisce il numero totale di processi.

        Examples::

            from pyvqnet.distributed import CommController
            Comm_OP = CommController("nccl") # inizializza il controllore nccl
            
            Comm_OP.getSize()
            # vqnetrun --backend nccl --nproc_per_node 2 python test.py




    .. py:method:: getLocalRank()
    
        Utilizzato per ottenere il numero di processo corrente sulla macchina corrente.


        :return: Il numero di processo corrente sulla macchina corrente.

        Examples::

            from pyvqnet.distributed import CommController
            Comm_OP = CommController("nccl") # inizializza il controllore nccl
            
            Comm_OP.getLocalRank()
            # vqnetrun --backend nccl --nproc_per_node 2 python test.py


    .. py:method:: split_groups(rankL)
        
        Divide piu' gruppi di comunicazione in base alla lista di numeri di processo impostata dal parametro di input.

        :param rankL: Una lista di rank dei gruppi di processo.

        :return: Quando il backend e' `nccl`, viene restituita una tupla di rank dei gruppi di processo. 
                 Quando il backend e' `mpi`, restituisce una lista la cui lunghezza e' uguale al numero di gruppi; ogni elemento e' una tupla (comm, rank), dove comm e' il comunicatore MPI del gruppo e rank e' il numero di sequenza all'interno del gruppo.

        Examples::
            
            from pyvqnet.distributed import CommController,get_rank,get_local_rank
            from pyvqnet.tensor import tensor
            import numpy as np
            Comm_OP = CommController("mpi")

            groups = Comm_OP.split_groups([[0, 1],[2,3]])
            print(groups)
            #[[<mpi4py.MPI.Intracomm object at 0x7f53691f3230>, [0, 3]], [<mpi4py.MPI.Intracomm object at 0x7f53691f3010>, [2, 1]]]

 
    .. py:method:: barrier()
    
        Sincronizzazione.

        :return: Sincronizzazione.

        Examples::

            from pyvqnet.distributed import CommController
            Comm_OP = CommController("nccl")
            
            Comm_OP.barrier()

 
    .. py:method:: get_device_num()
    
        Utilizzato per ottenere il numero di schede grafiche sul nodo corrente (supportato solo su GPU).

        :return: Restituisce il numero di schede grafiche sul nodo corrente.

        Examples::

            from pyvqnet.distributed import CommController
            Comm_OP = CommController("nccl")
            
            Comm_OP.get_device_num()
            # python test.py


 
    .. py:method:: allreduce(tensor, c_op = "avg")
    
        Supporta la comunicazione allreduce dei dati.

        :param tensor: Dati di input.
        :param c_op: Operazione di calcolo.

        Examples::

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
    
        Supporta la comunicazione reduce dei dati.

        :param tensor: input.
        :param root: Specifica il nodo a cui vengono restituiti i dati.
        :param c_op: Operazione di calcolo.

        Examples::

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
    
        Trasmette i dati dal processo root specificato a tutti i processi.

        :param tensor: input.
        :param root: Specifica il nodo.

        Examples::

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
    
        Esegue allgather dei dati su tutti i processi.

        :param tensor: input.

        Examples::

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
    
        Interfaccia di comunicazione p2p.

        :param tensor: input.
        :param dest: Processo di destinazione.

        Examples::

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
    
        Interfaccia di comunicazione p2p.

        :param tensor: input.
        :param source: Processo di ricezione.

        Examples::

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
    
        Interfaccia di comunicazione allreduce di gruppo.

        :param tensor: input.
        :param c_op: Operazione di calcolo.
        :param group: Gruppo di comunicazione. Quando si usa il backend mpi, inserire il gruppo generato da `init_groups` o `split_groups` corrispondente al gruppo di comunicazione. Quando si usa il backend nccl, inserire il numero di gruppo generato da `split_groups`.


        Examples::

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
    
        Interfaccia di comunicazione reduce all'interno del gruppo.

        :param tensor: Input.
        :param root: Specifica il numero di processo.
        :param c_op: Operazione di calcolo.
        :param group: Gruppo di comunicazione. Quando si usa il backend mpi, inserire il gruppo generato da `init_groups` o `split_groups` corrispondente al gruppo di comunicazione. Quando si usa il backend nccl, inserire il numero di gruppo generato da `split_groups`.

        Examples::
            
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
    
        Interfaccia di comunicazione broadcast all'interno del gruppo.

        :param tensor: Input.
        :param root: Specifica il numero di processo da cui trasmettere, default: 0.
        :param group: Gruppo di comunicazione. Quando si usa il backend mpi, inserire il gruppo generato da `init_groups` o `split_groups` corrispondente al gruppo di comunicazione. Quando si usa il backend nccl, inserire il numero di gruppo generato da `split_groups`.

        Examples::
            
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
    
        Interfaccia di comunicazione allgather di gruppo.

        :param tensor: Input.
        :param group: Gruppo di comunicazione. Quando si usa il backend mpi, inserire il gruppo generato da `init_groups` o `split_groups` corrispondente al gruppo di comunicazione. Quando si usa il backend nccl, inserire il numero di gruppo generato da `split_groups`.

        Examples::
            
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
    
        Aggiorna il gradiente dei parametri nell'ottimizzatore con allreduce.

        :param optimizer: ottimizzatore.

        Examples::
            
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
    
        Aggiorna i parametri del modello tramite allreduce.

        :param model: Modello.

        Examples::
        
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
    
        Trasmette i parametri del modello sul numero di processo specificato.

        :param model: Modello.
        :param root: Specifica il numero di processo.

        Examples::
        
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

        Usa NCCL per all_gather asincrono o sincrono sui dati GPU.

        :param output: QTensor - Il QTensor di destinazione per il risultato di all_gather.
        :param input: QTensor - Il QTensor da raccogliere.
        :param group: Gruppo di processi di comunicazione, group e' una tupla contenente gli indici del gruppo. Default: None, nessun gruppo utilizzato.
        :param async_op: Indica se questa operazione e' asincrona, default: False.
        :return: Work - Un handle di comunicazione asincrona. Usare wait() per attendere il completamento di questa operazione.

        Examples::

            from pyvqnet import tensor
            import pyvqnet
            from pyvqnet.distributed import CommController,get_rank,get_local_rank
            Comm_OP = CommController("nccl")

            complex_data = tensor.QTensor([3+1j, 2, 1 + get_rank()],dtype=8).reshape((3,1)).toGPU(1000+ get_local_rank())

            out_data = tensor.empty([2,3,1],dtype=pyvqnet.kcomplex64).toGPU(pyvqnet.DEV_GPU_0+ get_local_rank())
            work = Comm_OP.nccl_async_all_gather(out_data, complex_data, group = None,async_op=True)
            work.wait()

    .. py:method:: nccl_async_all_reduce(tensor, c_op="avg",group=None, async_op=False):

        Usa NCCL per allreduce asincrono o sincrono sui dati GPU.

        :param tensor: QTensor - Il QTensor che deve essere ridotto.
        :param c_op: Metodo di calcolo, puo' essere "sum" o "avg", il valore predefinito e' "avg".
        :param group: Gruppo di processi di comunicazione, group e' una tupla contenente gli indici del gruppo. Default: None, nessun gruppo utilizzato.
        :param async_op: Indica se questa operazione e' asincrona, default: False.
        :return: Work - Un handle di comunicazione asincrona. Usare wait() per attendere il completamento di questa operazione.

        Examples::

            import pyvqnet
            from pyvqnet.tensor import tensor
            from pyvqnet.distributed.ControlComm import CommController,get_local_rank

            Comm_OP = CommController("nccl")
            complex_data = tensor.ones([500,500],dtype=pyvqnet.kcomplex64).toGPU(pyvqnet.DEV_GPU_0 + get_local_rank())
            work = Comm_OP.nccl_async_all_reduce(complex_data, "sum",group = None,async_op = True)
            work.wait()

    .. py:method:: nccl_async_reduce( tensor_, dest, c_op="avg", group=None, async_op=False ):

        Usa NCCL per reduce asincrono o sincrono sui dati GPU.

        :param tensor_: QTensor - Il QTensor che deve essere ridotto.
        :param dest: Il rank di destinazione del QTensor ridotto.
        :param c_op: Metodo di calcolo, puo' essere "sum" o "avg", il valore predefinito e' "avg".
        :param group: Gruppo di processi di comunicazione, group e' una tupla contenente gli indici del gruppo. Default: None, nessun gruppo utilizzato.
        :param async_op: Indica se questa operazione e' asincrona, default: False.
        :return: Work - Un handle di comunicazione asincrona. Usare wait() per attendere il completamento di questa operazione.

        Examples::

            from pyvqnet import tensor
            import pyvqnet
            from pyvqnet.distributed import CommController,get_rank,get_local_rank
            Comm_OP = CommController("nccl")

            complex_data = tensor.ones([500,500],dtype=pyvqnet.kcomplex64).toGPU(pyvqnet.DEV_GPU_0 + get_local_rank())
            work = Comm_OP.nccl_async_reduce(complex_data, 0,"sum",group = None,async_op = True)
            work.wait()

    .. py:method:: nccl_async_broadcast(tensor, src, group=None, async_op=False )

        Usa NCCL per broadcast asincrono o sincrono sui dati GPU.

        :param tensor: QTensor - Il QTensor che deve essere trasmesso.
        :param src: Il rank di origine del QTensor da trasmettere.
        :param group: Gruppo di processi di comunicazione, group e' una tupla contenente gli indici del gruppo. Default: None, nessun gruppo utilizzato.
        :param async_op: Indica se questa operazione e' asincrona, default: False.
        :return: Work - Un handle di comunicazione asincrona. Usare wait() per attendere il completamento di questa operazione.

        Examples::

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

        Usa NCCL per l'invio P2P asincrono o sincrono sui dati GPU.

        :param t: QTensor - Il QTensor che deve essere inviato.
        :param dest: Il rank di destinazione a cui inviare il QTensor.
        :param async_op: Indica se questa operazione e' asincrona, default: False.
        :return: Work - Un handle di comunicazione asincrona. Usare wait() per attendere il completamento di questa operazione.

        Examples::

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

        Usa NCCL per la ricezione P2P asincrona o sincrona sui dati GPU.

        :param t: QTensor - Il QTensor che riceve i dati.
        :param src: Il rank di origine del QTensor ricevuto.
        :param async_op: Indica se questa operazione e' asincrona, default: False.
        :return: Work - Un handle di comunicazione asincrona. Usare wait() per attendere il completamento di questa operazione.

        Examples::

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

In un ambiente multi-processo, usare ``split_data`` per suddividere i dati in base al numero di processi e restituire i dati sul processo corrispondente.

.. py:function:: pyvqnet.distributed.datasplit.split_data(x_train, y_train, shuffle=False)

Imposta i parametri per il calcolo distribuito.

    :param x_train: `np.array` - dati di addestramento.
    :param y_train: `np.array` - etichette dei dati di addestramento.
    :param shuffle: `bool` - Indica se mescolare e poi suddividere, il valore predefinito e' False.

    :return: dati di addestramento suddivisi e relative etichette.

    Example::

        from pyvqnet.distributed import split_data
        import numpy as np

        x_train = np.random.randint(255, size = (100, 5))
        y_train = np.random.randint(2, size = (100, 1))

        x_train, y_train= split_data(x_train, y_train)

get_local_rank
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Usare ``get_local_rank`` per ottenere il numero di processo sulla macchina corrente.

.. py:function:: pyvqnet.distributed.ControlComm.get_local_rank()

    Utilizzato per ottenere il numero di processo corrente sulla macchina corrente.

    :return: numero di processo corrente sulla macchina corrente.

    Example::

        from pyvqnet.distributed.ControlComm import get_local_rank

        print(get_local_rank())
        # vqnetrun -n 2 python test.py

get_rank
~~~~~~~~~~~~~~~~~~~~~~~~~~~
Usare ``get_rank`` per ottenere il numero di processo sulla macchina corrente.

.. py:function:: pyvqnet.distributed.ControlComm.get_rank()

    Utilizzato per ottenere il numero di processo del processo corrente.

    :return: il numero di processo del processo corrente.

    Example::

        from pyvqnet.distributed.ControlComm import get_rank

        print(get_rank())
        # vqnetrun -n 2 python test.py

init_groups
~~~~~~~~~~~~~~~~~~~~~~~~~~~
Usare ``init_groups`` per inizializzare gruppi di processi basati su CPU a partire dalla lista data di numeri di processo.

.. py:function:: pyvqnet.distributed.ControlComm.init_groups(rank_lists)

    Utilizzato per inizializzare il gruppo di comunicazione dei processi per il backend `mpi`.

    :param rank_lists: Lista di gruppi di processi di comunicazione.
    :return: Una lista di gruppi di processi inizializzati.

    Example::

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
    
    Pipeline Parallel Training Wrapper implementa l'addestramento 1F1B. Disponibile solo su piattaforme Linux con GPU.
    Maggiori dettagli sull'algoritmo si trovano su (https://www.deepspeed.ai/tutorials/pipeline/).

    :param args: Dizionario dei parametri. Vedere gli esempi.
    :param join_layers: Lista di moduli Sequential.
    :param trainset: Dataset.

    :return:
        Istanza di PipelineParallelTrainingWrapper.

    L'esempio seguente usa il dataset CIFAR10 `CIFAR10_Dataset` per addestrare il task di classificazione su AlexNet su 2 GPU.
    In questo esempio, e' suddiviso in due processi pipeline paralleli `pipeline_parallel_size` = 2.
    La dimensione del batch e' `train_batch_size` = 64, su una singola GPU e' `train_micro_batch_size_per_gpu` = 32.
    Altri parametri di configurazione si trovano in `args`.
    Inoltre, ogni processo deve configurare la variabile d'ambiente `LOCAL_RANK` nella funzione `__main__`.

    Examples::

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
    
    Interfaccia API Zero1, attualmente solo per piattaforma Linux basata su calcolo parallelo GPU.

    :param args: dizionario dei parametri.
    :param model: Modulo.
    :param optimizer: Ottimizzatore.

    :return:
        Zero1 Engine.

L'esempio seguente usa il dataset MNIST per addestrare un task di classificazione su un modello MLP su 2 GPU.

    La dimensione del batch e' `train_batch_size` = 64, e lo stage `stage` di `zero_optimization` e' impostato a 1. 
    Se Optimizer e' None, viene usata l'impostazione di `optimizer` in `args`. Altri parametri di configurazione si trovano in `args`. 
    
    Inoltre, ogni processo deve essere configurato con la variabile d'ambiente `LOCAL_RANK`.
    
    .. code-block::

        os.environ["LOCAL_RANK"] = str(dist.get_local_rank())

    Examples::

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
            load mnist data
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
    
    Calcolo tensore-parallelo con layer lineare parallelo per colonne
    
    Il layer lineare e' definito come Y = XA + b. Le sue righe parallele 2D sono A = [A_1, ..., A_p].

    :param input_size: prima dimensione della matrice A.
    :param output_size: seconda dimensione della matrice A.
    :param weight_initializer: `callable` - default: normal.
    :param bias_initializer: `callable` - default: zeros.
    :param use_bias: `bool` - default: True.
    :param dtype: default: None, usa il tipo di dato predefinito.
    :param name: nome del modulo, default: "".
    :param tp_comm: Comm Controller.

    L'esempio seguente usa il dataset MNIST per addestrare un task di classificazione su un modello MLP su 2 GPU.

    L'uso e' simile a quello del classico layer Linear.

    Utilizzo multi-processo basato su `# vqnetrun --backend nccl --nproc_per_node 2 python test.py`.

    Examples::

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
            load mnist data
            """
            from array import array as pyarray
            # scarica i dati mnist
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
    
    Calcolo tensore-parallelo con layer lineare parallelo per righe.

    Il layer lineare e' definito come Y = XA + b. A e' parallelizzato lungo la sua prima dimensione e X lungo la sua seconda dimensione.
    A = transpose([A_1 .. A_p]) X = [X_1, ..., X_p].

    :param input_size: prima dimensione della matrice A.
    :param output_size: seconda dimensione della matrice A.
    :param weight_initializer: `callable` - default: normal.
    :param bias_initializer: `callable` - default: zeros.
    :param use_bias: `bool` - default: True.
    :param dtype: default: None, usa il tipo di dato predefinito.
    :param name: nome del modulo, default: "".
    :param tp_comm: Comm Controller.

    L'esempio seguente usa il dataset MNIST per addestrare un task di classificazione su un modello MLP su 2 GPU.

    L'uso e' simile a quello del classico layer Linear.

    Utilizzo multi-processo basato su `# vqnetrun --backend nccl --nproc_per_node 2 python test.py`.

    Examples::

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
            load mnist data
            """
            from array import array as pyarray
            # scarica i dati mnist
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


Riordinamento dei Qubit
=================================

Il riordinamento dei qubit e' una tecnica nel parallelismo di bit. Il suo obiettivo principale e' ridurre il numero di trasformazioni di bit richieste dal parallelismo di bit modificando l'ordine delle porte logiche quantistiche. I seguenti moduli sono necessari per costruire circuiti quantistici a molti bit basati sul parallelismo di bit. Fare riferimento all'articolo `Lazy Qubit Reordering for Accelerating Parallel State-Vector-based Quantum Circuit Simulation <https://export.arxiv.org/abs/2410.04252>`__.

Le seguenti interfacce richiedono `mpi` per avviare piu' processi per il calcolo.

DistributeQMachine
^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. py:class:: pyvqnet.distributed.qubits_reorder.DistributeQMachine(num_wires, dtype, grad_mode)
    
    Una classe per simulare computazioni quantistiche variazionali bit-parallele, inclusi stati quantistici su un sottoinsieme di bit su ogni nodo. Ogni nodo richiede una simulazione di circuito quantistico variazionale distribuito tramite MPI. Il valore N deve essere uguale a una potenza di 2 elevata al numero di bit paralleli distribuiti, `global_qubit`, e puo' essere configurato tramite `set_qr_config`.

    :param num_wires: Il numero di bit nel circuito quantistico completo.
    :param dtype: Il tipo di dato dei dati di calcolo. Il default e' pyvqnet.kcomplex64, corrispondente alla precisione dei parametri di pyvqnet.kfloat32.
    :param grad_mode: Impostare su adjoint per la retropropagazione di ``DistQuantumLayerAdjoint``.

    .. note::

        Il numero di bit in input e' il numero di bit richiesti per l'intero circuito quantistico. DistributeQMachine costruira' un simulatore quantistico basato sul numero globale di bit, che e' ``num_wires - global_qubit``.

        La retropropagazione deve basarsi su ``DistQuantumLayerAdjoint``.

    .. warning::

        Questa interfaccia supporta solo l'esecuzione su Linux;

        I parametri bit-paralleli in ``DistributeQMachine`` devono essere configurati, come mostrato nell'esempio, inclusi:

        .. code-block::

            qm.set_just_defined(True)
            qm.set_save_op_history_flag(True) # abilita salvataggio operazioni
            qm.set_qr_config({'qubit': total qubits number, 'global_qubit':  distributed qubits number})
    
    
    Examples::

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
                self.qm.set_save_op_history_flag(True) # abilita salvataggio operazioni
                self.qm.set_qr_config({"qubit": num_wires, # abilita riordinamento qubit, imposta configurazione
                                        "global_qubit": 1}) # global_qubit == log2(nproc), numero di processi
                
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
    
    Un layer DistQuantumLayer che calcola i gradienti per i parametri nelle computazioni bit-parallele usando l'approccio della matrice aggiunta.

    :param vqc_module: Il modulo ``DistributeQMachine`` implicito in input.
    :param name: Il nome del modulo.

    .. note::

        Il modulo vqc_module in input deve contenere ``DistributeQMachine``. I calcoli del gradiente di retropropagazione aggiunti vengono eseguiti basandosi su ``DistributeQMachine`` nelle computazioni bit-parallele.

    .. warning::

        Questa interfaccia e' supportata solo su Linux;


    Examples::

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
                self.qm.set_save_op_history_flag(True) # abilita salvataggio operazioni
                self.qm.set_qr_config({"qubit": num_wires, # abilita riordinamento qubit, imposta configurazione
                                            "global_qubit": 1}) # global_qubit == log2(nproc), numero di processi
                
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