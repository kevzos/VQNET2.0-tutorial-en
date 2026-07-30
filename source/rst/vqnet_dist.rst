
.. _vqnet_dist:

VQNet Einfaches Verteiltes Rechnen-Modul
*********************************************************

Umgebungseinrichtung
=================================

Dieser Abschnitt beschreibt, wie die VQNet-Umgebung unter Linux für verteiltes Rechnen mit CPU und GPU eingerichtet wird.

MPI-Installation
^^^^^^^^^^^^^^^^^^^^^^

MPI ist eine verbreitete Bibliothek für die Kommunikation zwischen CPUs. Die verteilte Rechnen-Funktion der CPU in VQNet basiert auf MPI.
Im folgenden Abschnitt wird beschrieben, wie MPI unter Linux installiert wird (derzeit wird die verteilte Rechnen-Funktion basierend auf der CPU nur unter Linux unterstützt).

Prüfen, ob die Compiler gcc und gfortran installiert sind.

.. code-block::
        
    which gcc 
    which gfortran

Wenn die Pfade zu gcc und gfortran angezeigt werden, können Sie mit dem nächsten Installationsschritt fortfahren. Falls die entsprechenden Compiler nicht vorhanden sind,
installieren Sie diese bitte zuerst. Nachdem die Compiler überprüft wurden, laden Sie mpich mit dem Befehl wget herunter.

.. code-block::
        
    wget http://www.mpich.org/static/downloads/3.3.2/mpich-3.3.2.tar.gz 
    tar -zxvf mpich-3.3.2.tar.gz 
    cd mpich-3.3.2 
    ./configure --prefix=/usr/local/mpich
    make 
    make install 

Nach dem Kompilieren und Installieren von mpich konfigurieren Sie die Umgebungsvariablen.

.. code-block::
        
    vim ~/.bashrc

    # Am Ende der Datei folgende Zeile hinzufügen
    export PATH="/usr/local/mpich/bin:$PATH"

Nach dem Speichern und Schließen wenden Sie die Änderungen mit source an.

.. code-block::

    source ~/.bashrc

Verwenden Sie which, um zu überprüfen, ob die Umgebungsvariablen korrekt konfiguriert sind. Wenn der Pfad angezeigt wird, ist die Installation erfolgreich abgeschlossen.

Zusätzlich können Sie mpi4py mit pip installieren. Sollte der folgende Fehler auftreten:

.. image:: ./images/mpi_bug.png
    :align: center

|

Um die Inkompatibilität zwischen mpi4py und der Python-Version zu beheben, gehen Sie wie folgt vor:

.. code-block::

    # Compiler der aktuellen Python-Umgebung sichern
    pushd /root/anaconda3/envs/$CONDA_DEFAULT_ENV/compiler_compat && mv ld ld.bak && popd

    # mpi4py neu installieren
    pip install mpi4py

    # Original-Compiler wiederherstellen
    pushd /root/anaconda3/envs/$CONDA_DEFAULT_ENV/compiler_compat && mv ld.bak ld && popd

NCCL-Installation
^^^^^^^^^^^^^^^^^^^^^^

NCCL ist eine verbreitete Bibliothek für die GPU-Kommunikation. Die verteilte Rechnen-Funktion der GPUs in VQNet basiert auf NCCL.
Diese Software installiert die dynamische NCCL-Bibliothek standardmäßig bei der Installation, daher ist es in der Regel nicht erforderlich, NCCL separat zu installieren.
Dieser Abschnitt setzt MPI-Unterstützung voraus, daher muss auch die MPI-Umgebung eingerichtet werden.

Verteilter Start
=================================
 
Im Folgenden werden die Parameter des ``vqnetrun``-Befehls beschrieben, der die Schnittstelle für verteiltes Rechnen startet.

n, np
^^^^^^^^^^^^^^^^^^^^^^

Die ``vqnetrun``-Schnittstelle ermöglicht die Steuerung der Anzahl gestarteter Prozesse mit den Parametern ``-n`` und ``-np``, wie im folgenden Beispiel gezeigt.

    Example::

        from pyvqnet.distributed import CommController
        Comm_OP = CommController("mpi") # MPI-Controller initialisieren
        
        rank = Comm_OP.getRank()
        size = Comm_OP.getSize()
        print(f"rank: {rank}, size {size}")

        # vqnetrun -n 2 python test.py
        # vqnetrun -np 2 python test.py

backend
^^^^^^^^^^^^^^^^^^^^^^

Die ``vqnetrun``-Schnittstelle ermöglicht die Auswahl des verteilten Backends mit dem Parameter ``--backend`` und unterstützt die Modi ``mpi`` (Standard) und ``nccl``.
Der MPI-Modus ist für verteiltes Rechnen auf der CPU vorgesehen, der NCCL-Modus für verteiltes Rechnen auf der GPU.

    Example::

        from pyvqnet.distributed import CommController
        Comm_OP = CommController("nccl")

        rank = Comm_OP.getRank()
        size = Comm_OP.getSize()
        print(f"rank: {rank}, size {size}")

        # vqnetrun --backend nccl --nproc_per_node 2 python test.py

nproc_per_node
^^^^^^^^^^^^^^^^^^^^^^

Die ``vqnetrun``-Schnittstelle ermöglicht die Steuerung der Anzahl der Prozesse pro Knoten mit dem Parameter ``--nproc_per_node``. Dieser ist nur im NCCL-Modus verfügbar.

    Example::

        from pyvqnet.distributed import CommController
        Comm_OP = CommController("nccl")

        rank = Comm_OP.getRank()
        size = Comm_OP.getSize()
        print(f"rank: {rank}, size {size}")

        # vqnetrun --backend nccl --nproc_per_node 4 python test.py

output-filename
^^^^^^^^^^^^^^^^^^^^^^

Die ``vqnetrun``-Schnittstelle ermöglicht das Speichern der Ausgabe in einer angegebenen Datei mit dem Befehlszeilenparameter ``--output-filename``.

Eine beispielhafte Implementierung sieht wie folgt aus:

    Example::

        from pyvqnet.distributed import CommController, get_host_name
        Comm_OP = CommController("mpi") # MPI-Controller initialisieren
        
        rank = Comm_OP.getRank()
        size = Comm_OP.getSize()
        print(f"rank: {rank}, size {size}")
        print(f"LocalRank {Comm_OP.getLocalRank()} hosts name {get_host_name()}")

        # vqnetrun -np 4 --hostfile hosts --output-filename output  python test.py


verbose
^^^^^^^^^^^^^^^^^^^^^^

Die ``vqnetrun``-Schnittstelle kann mit dem Befehlszeilenparameter ``--verbose`` verwendet werden, um eine detaillierte Kommunikationsprotokollierung zu aktivieren und die Ergebnisse auszugeben.

Eine beispielhafte Implementierung sieht wie folgt aus:

    Example::

        from pyvqnet.distributed import CommController, get_host_name
        Comm_OP = CommController("mpi") # MPI-Controller initialisieren
        
        rank = Comm_OP.getRank()
        size = Comm_OP.getSize()
        print(f"rank: {rank}, size {size}")
        print(f"LocalRank {Comm_OP.getLocalRank()} hosts name {get_host_name()}")

        # vqnetrun -np 4 --hostfile hosts --verbose python test.py


start-timeout
^^^^^^^^^^^^^^^^^^^^^^

Die ``vqnetrun``-Schnittstelle kann mit dem Befehlszeilenparameter ``-start-timeout`` verwendet werden, um festzulegen, dass alle Prüfungen durchgeführt und der Prozess vor dem Timeout gestartet werden. Der Standardwert beträgt 30 Sekunden.

Eine beispielhafte Implementierung sieht wie folgt aus:

    Example::

        from pyvqnet.distributed import CommController, get_host_name
        Comm_OP = CommController("mpi") # MPI-Controller initialisieren
        
        rank = Comm_OP.getRank()
        size = Comm_OP.getSize()
        print(f"rank: {rank}, size {size}")
        print(f"LocalRank {Comm_OP.getLocalRank()} hosts name {get_host_name()}")

        # vqnetrun -np 4 --start-timeout 10 python test.py


CommController
=================================

    Verteiltes Rechnen dient der Steuerung der Datenkommunikation zwischen verschiedenen Prozessen unter CPU und GPU. Es erzeugt unterschiedliche Controller für CPU (MPI) und GPU (NCCL) und ruft die Kommunikationsmethoden auf, um die Datenkommunikation und Synchronisation zwischen verschiedenen Prozessen durchzuführen.

__init__
^^^^^^^^^^^^^^^^^^^^^^
.. py:class:: pyvqnet.distributed.ControlComm.CommController(backend,rank=None,world_size=None)
    
    CommController dient der Steuerung der Datenkommunikation unter CPU und GPU. Durch die Angabe des Parameters `backend` wird der Controller für CPU (MPI) oder GPU (NCCL) erzeugt. (Derzeit wird die verteilte Rechnen-Funktion nur unter Linux unterstützt.)

    :param backend: Wird verwendet, um den Datenkommunikations-Controller für CPU oder GPU zu erzeugen.
    :param rank: Dieser Parameter ist nur in Nicht-pyvqnet-Backends nützlich, der Standardwert ist: None.
    :param world_size: Dieser Parameter ist nur in Nicht-pyvqnet-Backends nützlich, der Standardwert ist: None.
        
    :return:
        CommController-Instanz.

    Examples::

        from pyvqnet.distributed import CommController
        Comm_OP = CommController("nccl") # NCCL-Controller initialisieren

        # Comm_OP = CommController("mpi") # MPI-Controller initialisieren

 
    .. py:method:: getRank()
        
        Wird verwendet, um die Prozessnummer des aktuellen Prozesses zu erhalten.

        :return: Gibt die Prozessnummer des aktuellen Prozesses zurück.

        Examples::

            from pyvqnet.distributed import CommController
            Comm_OP = CommController("nccl") # NCCL-Controller initialisieren
            
            Comm_OP.getRank()

 
    .. py:method:: getSize()
    
        Wird verwendet, um die Gesamtzahl der gestarteten Prozesse zu erhalten.


        :return: Gibt die Gesamtzahl der Prozesse zurück.

        Examples::

            from pyvqnet.distributed import CommController
            Comm_OP = CommController("nccl") # NCCL-Controller initialisieren
            
            Comm_OP.getSize()
            # vqnetrun --backend nccl --nproc_per_node 2 python test.py




    .. py:method:: getLocalRank()
    
        Wird verwendet, um die aktuelle Prozessnummer auf der aktuellen Maschine zu erhalten.


        :return: Die aktuelle Prozessnummer auf der aktuellen Maschine.

        Examples::

            from pyvqnet.distributed import CommController
            Comm_OP = CommController("nccl") # NCCL-Controller initialisieren
            
            Comm_OP.getLocalRank()
            # vqnetrun --backend nccl --nproc_per_node 2 python test.py


    .. py:method:: split_groups(rankL)
        
        Teilt mehrere Kommunikationsgruppen entsprechend der durch den Eingabeparameter festgelegten Prozessnummernliste auf.

        :param rankL: Eine Liste von Prozessgruppen-Ranks.

        :return: Wenn das Backend `nccl` ist, wird ein Tupel von Prozessgruppen-Ranks zurückgegeben.
                 Wenn das Backend `mpi` ist, wird eine Liste zurückgegeben, deren Länge der Anzahl der Gruppen entspricht; jedes Element ist ein Tupel (comm, rank), wobei comm der MPI-Kommunikator der Gruppe und rank die Sequenznummer innerhalb der Gruppe ist.

        Examples::
            
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

        Examples::

            from pyvqnet.distributed import CommController
            Comm_OP = CommController("nccl")
            
            Comm_OP.barrier()

 
    .. py:method:: get_device_num()
    
        Wird verwendet, um die Anzahl der Grafikkarten auf dem aktuellen Knoten zu ermitteln (nur auf GPU unterstützt).

        :return: Gibt die Anzahl der Grafikkarten auf dem aktuellen Knoten zurück.

        Examples::

            from pyvqnet.distributed import CommController
            Comm_OP = CommController("nccl")
            
            Comm_OP.get_device_num()
            # python test.py


 
    .. py:method:: allreduce(tensor, c_op = "avg")
    
        Unterstützt die Allreduce-Kommunikation von Daten.

        :param tensor: Eingabedaten.
        :param c_op: Berechnungsoperation.

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
    
        Unterstützt die Reduce-Kommunikation von Daten.

        :param tensor: Eingabe.
        :param root: Gibt den Knoten an, an den die Daten zurückgegeben werden.
        :param c_op: Berechnungsoperation.

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
    
        Sendet Daten vom angegebenen Prozess (root) an alle Prozesse.

        :param tensor: Eingabe.
        :param root: Gibt den Knoten an.

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
    
        Sammelt die Daten aller Prozesse mit Allgather zusammen.

        :param tensor: Eingabe.

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
    
        P2P-Kommunikationsschnittstelle.

        :param tensor: Eingabe.
        :param dest: Zielprozess.

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
    
        P2P-Kommunikationsschnittstelle.

        :param tensor: Eingabe.
        :param source: Quellprozess.

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
    
        Die Gruppen-Allreduce-Kommunikationsschnittstelle.

        :param tensor: Eingabe.
        :param c_op: Berechnungsoperation.
        :param group: Kommunikationsgruppe. Bei Verwendung des MPI-Backends geben Sie die von `init_groups` oder `split_groups` erzeugte Gruppe ein, die der Kommunikationsgruppe entspricht. Bei Verwendung des NCCL-Backends geben Sie die von `split_groups` erzeugte Gruppennummer ein.


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
    
        Gruppeninterne Reduce-Kommunikationsschnittstelle.

        :param tensor: Eingabe.
        :param root: Gibt die Prozessnummer an.
        :param c_op: Berechnungsoperation.
        :param group: Kommunikationsgruppe. Bei Verwendung des MPI-Backends geben Sie die von `init_groups` oder `split_groups` erzeugte Gruppe ein, die der Kommunikationsgruppe entspricht. Bei Verwendung des NCCL-Backends geben Sie die von `split_groups` erzeugte Gruppennummer ein.

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
    
        Gruppeninterne Broadcast-Kommunikationsschnittstelle.

        :param tensor: Eingabe.
        :param root: Gibt die Prozessnummer an, von der aus gesendet wird, Standard: 0.
        :param group: Kommunikationsgruppe. Bei Verwendung des MPI-Backends geben Sie die von `init_groups` oder `split_groups` erzeugte Gruppe ein, die der Kommunikationsgruppe entspricht. Bei Verwendung des NCCL-Backends geben Sie die von `split_groups` erzeugte Gruppennummer ein.

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
    
        Die Gruppen-Allgather-Kommunikationsschnittstelle.

        :param tensor: Eingabe.
        :param group: Kommunikationsgruppe. Bei Verwendung des MPI-Backends geben Sie die von `init_groups` oder `split_groups` erzeugte Gruppe ein, die der Kommunikationsgruppe entspricht. Bei Verwendung des NCCL-Backends geben Sie die von `split_groups` erzeugte Gruppennummer ein.

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
    
        Aktualisiert die Gradienten der Parameter im Optimierer mit Allreduce.

        :param optimizer: Optimierer.

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
    
        Aktualisiert die Parameter im Modell mittels Allreduce.

        :param model: Modell.

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
    
        Sendet die Modellparameter auf der angegebenen Prozessnummer an alle Prozesse.

        :param model: Modell.
        :param root: Gibt die Prozessnummer an.

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

        Verwendet NCCL für asynchrones oder synchrones Allgather auf GPU-Daten.

        :param output: QTensor - Der Ziel-QTensor für das Allgather-Ergebnis.
        :param input: QTensor - Der QTensor, der eingesammelt werden soll.
        :param group: Kommunikationsprozessgruppe, group ist ein Tupel mit Gruppenindizes. Standard: None, es wird keine Gruppe verwendet.
        :param async_op: Gibt an, ob dieser Vorgang asynchron ist, Standard: False.
        :return: Work - Ein asynchrones Kommunikations-Handle. Verwenden Sie wait(), um auf die Fertigstellung dieses Vorgangs zu warten.

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

        Verwendet NCCL für asynchrones oder synchrones Allreduce auf GPU-Daten.

        :param tensor: QTensor - Der QTensor, der reduziert werden soll.
        :param c_op: Berechnungsmethode, kann "sum" oder "avg" sein, Standardwert ist "avg".
        :param group: Kommunikationsprozessgruppe, group ist ein Tupel mit Gruppenindizes. Standard: None, es wird keine Gruppe verwendet.
        :param async_op: Gibt an, ob dieser Vorgang asynchron ist, Standard: False.
        :return: Work - Ein asynchrones Kommunikations-Handle. Verwenden Sie wait(), um auf die Fertigstellung dieses Vorgangs zu warten.

        Examples::

            import pyvqnet
            from pyvqnet.tensor import tensor
            from pyvqnet.distributed.ControlComm import CommController,get_local_rank

            Comm_OP = CommController("nccl")
            complex_data = tensor.ones([500,500],dtype=pyvqnet.kcomplex64).toGPU(pyvqnet.DEV_GPU_0 + get_local_rank())
            work = Comm_OP.nccl_async_all_reduce(complex_data, "sum",group = None,async_op = True)
            work.wait()

    .. py:method:: nccl_async_reduce( tensor_, dest, c_op="avg", group=None, async_op=False ):

        Verwendet NCCL für asynchrones oder synchrones Reduce auf GPU-Daten.

        :param tensor_: QTensor - Der QTensor, der reduziert werden soll.
        :param dest: Der Ziel-Rank des reduzierten QTensor.
        :param c_op: Berechnungsmethode, kann "sum" oder "avg" sein, Standardwert ist "avg".
        :param group: Kommunikationsprozessgruppe, group ist ein Tupel mit Gruppenindizes. Standard: None, es wird keine Gruppe verwendet.
        :param async_op: Gibt an, ob dieser Vorgang asynchron ist, Standard: False.
        :return: Work - Ein asynchrones Kommunikations-Handle. Verwenden Sie wait(), um auf die Fertigstellung dieses Vorgangs zu warten.

        Examples::

            from pyvqnet import tensor
            import pyvqnet
            from pyvqnet.distributed import CommController,get_rank,get_local_rank
            Comm_OP = CommController("nccl")

            complex_data = tensor.ones([500,500],dtype=pyvqnet.kcomplex64).toGPU(pyvqnet.DEV_GPU_0 + get_local_rank())
            work = Comm_OP.nccl_async_reduce(complex_data, 0,"sum",group = None,async_op = True)
            work.wait()

    .. py:method:: nccl_async_broadcast(tensor, src, group=None, async_op=False )

        Verwendet NCCL für asynchrones oder synchrones Broadcast auf GPU-Daten.

        :param tensor: QTensor - Der QTensor, der gesendet werden soll.
        :param src: Der Quell-Rank des zu sendenden QTensor.
        :param group: Kommunikationsprozessgruppe, group ist ein Tupel mit Gruppenindizes. Standard: None, es wird keine Gruppe verwendet.
        :param async_op: Gibt an, ob dieser Vorgang asynchron ist, Standard: False.
        :return: Work - Ein asynchrones Kommunikations-Handle. Verwenden Sie wait(), um auf die Fertigstellung dieses Vorgangs zu warten.

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

        Verwendet NCCL für asynchrones oder synchrones P2P-Senden auf GPU-Daten.

        :param t: QTensor - Der QTensor, der gesendet werden soll.
        :param dest: Der Ziel-Rank, an den der QTensor gesendet werden soll.
        :param async_op: Gibt an, ob dieser Vorgang asynchron ist, Standard: False.
        :return: Work - Ein asynchrones Kommunikations-Handle. Verwenden Sie wait(), um auf die Fertigstellung dieses Vorgangs zu warten.

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

        Verwendet NCCL für asynchrones oder synchrones P2P-Empfangen auf GPU-Daten.

        :param t: QTensor - Der QTensor, der die Daten empfängt.
        :param src: Der Quell-Rank des empfangenen QTensor.
        :param async_op: Gibt an, ob dieser Vorgang asynchron ist, Standard: False.
        :return: Work - Ein asynchrones Kommunikations-Handle. Verwenden Sie wait(), um auf die Fertigstellung dieses Vorgangs zu warten.

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

Verwendet ``split_data``, um die Daten in Mehrprozess-Umgebungen entsprechend der Anzahl der Prozesse aufzuteilen und die Daten auf dem entsprechenden Prozess zurückzugeben.

.. py:function:: pyvqnet.distributed.datasplit.split_data(x_train, y_train, shuffle=False)

Legt Parameter für verteiltes Rechnen fest.

    :param x_train: `np.array` - Trainingsdaten.
    :param y_train: `np.array` - Labels der Trainingsdaten.
    :param shuffle: `bool` - Gibt an, ob vor dem Aufteilen gemischt werden soll, Standard ist False.

    :return: Aufgeteilte Trainingsdaten und Labels.

    Example::

        from pyvqnet.distributed import split_data
        import numpy as np

        x_train = np.random.randint(255, size = (100, 5))
        y_train = np.random.randint(2, size = (100, 1))

        x_train, y_train= split_data(x_train, y_train)

get_local_rank
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Verwenden Sie ``get_local_rank``, um die Prozessnummer auf der aktuellen Maschine zu erhalten.

.. py:function:: pyvqnet.distributed.ControlComm.get_local_rank()

    Wird verwendet, um die aktuelle Prozessnummer auf der aktuellen Maschine zu erhalten.

    :return: Aktuelle Prozessnummer auf der aktuellen Maschine.

    Example::

        from pyvqnet.distributed.ControlComm import get_local_rank

        print(get_local_rank())
        # vqnetrun -n 2 python test.py

get_rank
~~~~~~~~~~~~~~~~~~~~~~~~~~~
Verwenden Sie ``get_rank``, um die Prozessnummer auf der aktuellen Maschine zu erhalten.

.. py:function:: pyvqnet.distributed.ControlComm.get_rank()

    Wird verwendet, um die Prozessnummer des aktuellen Prozesses zu erhalten.

    :return: Die Prozessnummer des aktuellen Prozesses.

    Example::

        from pyvqnet.distributed.ControlComm import get_rank

        print(get_rank())
        # vqnetrun -n 2 python test.py

init_groups
~~~~~~~~~~~~~~~~~~~~~~~~~~~
Verwenden Sie ``init_groups``, um CPU-basierte Prozessgruppen basierend auf der angegebenen Liste von Prozessnummern zu initialisieren.

.. py:function:: pyvqnet.distributed.ControlComm.init_groups(rank_lists)

    Wird verwendet, um die Prozesskommunikationsgruppe für das `mpi`-Backend zu initialisieren.

    :param rank_lists: Liste der Kommunikationsprozessgruppen.
    :return: Eine Liste der initialisierten Prozessgruppen.

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
    
    Der Pipeline-Parallel-Training-Wrapper implementiert das 1F1B-Training. Nur auf Linux-Plattformen mit GPU verfügbar.
    Weitere algorithmische Details finden Sie unter (https://www.deepspeed.ai/tutorials/pipeline/).

    :param args: Parameterwörterbuch. Siehe Beispiele.
    :param join_layers: Liste von Sequential-Modulen.
    :param trainset: Datensatz.

    :return:
        PipelineParallelTrainingWrapper-Instanz.

    Im Folgenden wird der CIFAR10-Datensatz `CIFAR10_Dataset` verwendet, um die Klassifikationsaufgabe auf AlexNet auf 2 GPUs zu trainieren.
    In diesem Beispiel wird in zwei Pipeline-Parallel-Prozesse `pipeline_parallel_size` = 2 aufgeteilt.
    Die Batchgröße beträgt `train_batch_size` = 64, auf einer einzelnen GPU beträgt sie `train_micro_batch_size_per_gpu` = 32.
    Weitere Konfigurationsparameter finden Sie in `args`.
    Zusätzlich muss jeder Prozess die Umgebungsvariable `LOCAL_RANK` in der `__main__`-Funktion konfigurieren.

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
    
    Zero1-API-Schnittstelle, derzeit nur für Linux-Plattformen mit GPU-Parallelverarbeitung.

    :param args: Parameterwörterbuch.
    :param model: Modul.
    :param optimizer: Optimierer.

    :return:
        Zero1-Engine.

Im Folgenden wird der MNIST-Datensatz verwendet, um eine Klassifikationsaufgabe auf einem MLP-Modell auf 2 GPUs zu trainieren.

    Die Batchgröße beträgt `train_batch_size` = 64, und die Stufe `stage` von `zero_optimization` ist auf 1 gesetzt.
    Wenn der Optimierer None ist, wird die Einstellung von `optimizer` in `args` verwendet. Weitere Konfigurationsparameter finden Sie in `args`.
    
    Zusätzlich muss jeder Prozess mit der Umgebungsvariable `LOCAL_RANK` konfiguriert werden.
    
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
    
    Tensor-parallele Berechnung mit spaltenparalleler linearer Schicht
    
    Die lineare Schicht ist definiert als Y = XA + b. Ihre 2D-parallelen Zeilen sind A = [A_1, ... , A_p].

    :param input_size: Erste Dimension der Matrix A.
    :param output_size: Zweite Dimension der Matrix A.
    :param weight_initializer: `callable` - Standard ist normal.
    :param bias_initializer: `callable` - Standard ist zeros.
    :param use_bias: `bool` - Standard ist True.
    :param dtype: Standard: None, verwendet den Standard-Datentyp.
    :param name: Name des Moduls, Standard: "".
    :param tp_comm: Comm Controller.

    Im Folgenden wird der MNIST-Datensatz verwendet, um eine Klassifikationsaufgabe auf einem MLP-Modell auf 2 GPUs zu trainieren.

    Die Verwendung ist ähnlich wie bei der klassischen Linear-Schicht.

    Mehrprozess-Nutzung basierend auf `# vqnetrun --backend nccl --nproc_per_node 2 python test.py`.

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
    
    Tensor-parallele Berechnung mit zeilenparalleler linearer Schicht.

    Die lineare Schicht ist definiert als Y = XA + b. A wird entlang seiner ersten Dimension und X entlang seiner zweiten Dimension parallelisiert.
    A = transpose([A_1 .. A_p]) X = [X_1, ..., X_p].

    :param input_size: Erste Dimension der Matrix A.
    :param output_size: Zweite Dimension der Matrix A.
    :param weight_initializer: `callable` - Standard ist normal.
    :param bias_initializer: `callable` - Standard ist zeros.
    :param use_bias: `bool` - Standard ist True.
    :param dtype: Standard: None, verwendet den Standard-Datentyp.
    :param name: Name des Moduls, Standard: "".
    :param tp_comm: Comm Controller.

    Im Folgenden wird der MNIST-Datensatz verwendet, um eine Klassifikationsaufgabe auf einem MLP-Modell auf 2 GPUs zu trainieren.

    Die Verwendung ist ähnlich wie bei der klassischen Linear-Schicht.

    Mehrprozess-Nutzung basierend auf `# vqnetrun --backend nccl --nproc_per_node 2 python test.py`.

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


Bit-Umordnung
=================================

Bei der Qubit-Umordnung handelt es sich um eine Technik des Bit-Parallelismus. Ihr Kernziel ist es, die Anzahl der benötigten Bit-Transformationen im Bit-Parallelismus zu reduzieren, indem die Reihenfolge der Quantenlogikgatter geändert wird. Die folgenden Module werden für den Aufbau großer Bit-Quantenschaltungen auf Basis des Bit-Parallelismus benötigt. Siehe auch das Paper `Lazy Qubit Reordering for Accelerating Parallel State-Vector-based Quantum Circuit Simulation <https://export.arxiv.org/abs/2410.04252>`__.

Die folgenden Schnittstellen erfordern `mpi`, um mehrere Prozesse für die Berechnung zu starten.

DistributeQMachine
^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. py:class:: pyvqnet.distributed.qubits_reorder.DistributeQMachine(num_wires, dtype, grad_mode)
    
    Eine Klasse zur Simulation bit-paralleler varianter Quantenberechnungen, einschließlich Quantenzuständen auf einer Teilmenge von Bits auf jedem Knoten. Jeder Knoten beantragt eine verteilte Quanten-Variationsschaltungssimulation via MPI. Der Wert von N muss gleich einer Potenz von 2 hoch der Anzahl verteilter paralleler Bits, `global_qubit`, sein und kann über `set_qr_config` konfiguriert werden.

    :param num_wires: Die Anzahl der Bits in der vollständigen Quantenschaltung.
    :param dtype: Der Datentyp der Berechnungsdaten. Der Standardwert ist pyvqnet.kcomplex64, entsprechend der Parameterpräzision von pyvqnet.kfloat32.
    :param grad_mode: Auf adjoint setzen bei der Rückwärtspropagation von ``DistQuantumLayerAdjoint``.

    .. note::

        Die eingegebene Anzahl von Bits ist die Anzahl der Bits, die für die gesamte Quantenschaltung benötigt werden. DistributeQMachine erstellt einen Quantensimulator basierend auf der globalen Anzahl von Bits, die ``num_wires - global_qubit`` beträgt.

        Die Rückwärtspropagation muss auf ``DistQuantumLayerAdjoint`` basieren.

    .. warning::

        Diese Schnittstelle wird nur unter Linux unterstützt;

        Die Bit-Parallelismus-Parameter in ``DistributeQMachine`` müssen konfiguriert werden, wie im Beispiel gezeigt, einschließlich:

        .. code-block::

            qm.set_just_defined(True)
            qm.set_save_op_history_flag(True) # Speicheroperation aktivieren
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
                self.qm.set_save_op_history_flag(True) # Speicheroperation aktivieren
                self.qm.set_qr_config({"qubit": num_wires, # Qubit-Umordnung aktivieren, Konfiguration setzen
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
    
    Eine DistQuantumLayer-Schicht, die Gradienten für Parameter in bit-parallelen Berechnungen mit dem Ansatz der adjungierten Matrix berechnet.

    :param vqc_module: Das eingegebene implizite ``DistributeQMachine``-Modul.
    :param name: Der Modulname.

    .. note::

        Das eingegebene vqc_module-Modul muss ``DistributeQMachine`` enthalten. Die adjungierte Rückwärtspropagations-Gradientenberechnung wird basierend auf ``DistributeQMachine`` in bit-parallelen Berechnungen durchgeführt.

    .. warning::

        Diese Schnittstelle wird nur unter Linux unterstützt;


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
                self.qm.set_save_op_history_flag(True) # Speicheroperation aktivieren
                self.qm.set_qr_config({"qubit": num_wires, # Qubit-Umordnung aktivieren, Konfiguration setzen
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