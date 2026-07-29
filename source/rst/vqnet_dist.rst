
.. _vqnet_dist:

Módulo de Computación Distribuida Naive de VQNet
*********************************************************

Despliegue del Entorno
=================================

Esta sección describe cómo desplegar el entorno VQNet en Linux para computación distribuida con CPU y GPU.

Instalación de MPI
^^^^^^^^^^^^^^^^^^^^^^

MPI es una biblioteca común para la comunicación entre CPUs, y la función de computación distribuida de la CPU en VQNet se implementa basándose en MPI.
La siguiente sección describe cómo instalar MPI en un sistema Linux (actualmente, la función de computación distribuida basada en CPU solo es compatible con Linux).

Detecte si los compiladores gcc y gfortran están instalados.

.. code-block::
        
    which gcc 
    which gfortran

Cuando se muestren las rutas a gcc y gfortran, puede continuar con el siguiente paso de la instalación. Si no tiene los compiladores correspondientes,
instálelos primero. Una vez que los compiladores hayan sido verificados, use el comando wget para descargar mpich.

.. code-block::
        
    wget http://www.mpich.org/static/downloads/3.3.2/mpich-3.3.2.tar.gz 
    tar -zxvf mpich-3.3.2.tar.gz 
    cd mpich-3.3.2 
    ./configure --prefix=/usr/local/mpich
    make 
    make install 

Después de compilar e instalar mpich, configure sus variables de entorno.

.. code-block::
        
    vim ~/.bashrc

    # Al final del documento, añada
    export PATH="/usr/local/mpich/bin:$PATH"

Después de guardar y salir, use source para aplicar los cambios.

.. code-block::

    source ~/.bashrc

Use which para verificar que las variables de entorno se hayan configurado correctamente. Si se muestra la ruta, la instalación se ha completado con éxito.

Además, puede instalar mpi4py mediante pip install. Si encuentra el siguiente error:

.. image:: ./images/mpi_bug.png
    :align: center

|

Para resolver la incompatibilidad de versiones entre mpi4py y Python, puede hacer lo siguiente:

.. code-block::

    # Haga una copia de seguridad del compilador del entorno Python actual
    pushd /root/anaconda3/envs/$CONDA_DEFAULT_ENV/compiler_compat && mv ld ld.bak && popd

    # Re-instale mpi4py
    pip install mpi4py

    # Restaure el compilador original
    pushd /root/anaconda3/envs/$CONDA_DEFAULT_ENV/compiler_compat && mv ld.bak ld && popd

Instalación de NCCL
^^^^^^^^^^^^^^^^^^^^^^

NCCL es una biblioteca común para la comunicación entre GPUs, y la función de computación distribuida de las GPUs en VQNet se implementa basándose en NCCL.
Este software instala la biblioteca de enlace dinámico de NCCL por defecto durante la instalación, por lo que generalmente no es necesario instalar NCCL por separado.
Esta sección requiere soporte de MPI, por lo que también es necesario desplegar el entorno MPI.

Lanzamiento Distribuido
=================================
 
Usando la interfaz de computación distribuida iniciada por el comando ``vqnetrun``, los parámetros de ``vqnetrun`` se describen a continuación.

n, np
^^^^^^^^^^^^^^^^^^^^^^

La interfaz ``vqnetrun`` permite controlar el número de procesos iniciados con los parámetros ``-n``, ``-np``, como se muestra en el siguiente ejemplo.

    Ejemplo::

        from pyvqnet.distributed import CommController
        Comm_OP = CommController("mpi") # inicializa el controlador mpi
        
        rank = Comm_OP.getRank()
        size = Comm_OP.getSize()
        print(f"rank: {rank}, size {size}")

        # vqnetrun -n 2 python test.py
        # vqnetrun -np 2 python test.py

backend
^^^^^^^^^^^^^^^^^^^^^^

La interfaz ``vqnetrun`` permite seleccionar el backend distribuido con el parámetro ``--backend``, compatible con los modos ``mpi`` (predeterminado) y ``nccl``.
El modo MPI es para computación distribuida basada en CPU, y el modo NCCL es para computación distribuida basada en GPU.

    Ejemplo::

        from pyvqnet.distributed import CommController
        Comm_OP = CommController("nccl")

        rank = Comm_OP.getRank()
        size = Comm_OP.getSize()
        print(f"rank: {rank}, size {size}")

        # vqnetrun --backend nccl --nproc_per_node 2 python test.py

nproc_per_node
^^^^^^^^^^^^^^^^^^^^^^

La interfaz ``vqnetrun`` permite controlar el número de procesos iniciados en cada nodo con el parámetro ``--nproc_per_node``, solo disponible en modo NCCL.

    Ejemplo::

        from pyvqnet.distributed import CommController
        Comm_OP = CommController("nccl")

        rank = Comm_OP.getRank()
        size = Comm_OP.getSize()
        print(f"rank: {rank}, size {size}")

        # vqnetrun --backend nccl --nproc_per_node 4 python test.py

output-filename
^^^^^^^^^^^^^^^^^^^^^^

La interfaz ``vqnetrun`` permite guardar la salida en un archivo específico con el parámetro de línea de comandos ``--output-filename``.

Un ejemplo de implementación es el siguiente:

    Ejemplo::

        from pyvqnet.distributed import CommController, get_host_name
        Comm_OP = CommController("mpi") # inicializa el controlador mpi
        
        rank = Comm_OP.getRank()
        size = Comm_OP.getSize()
        print(f"rank: {rank}, size {size}")
        print(f"LocalRank {Comm_OP.getLocalRank()} hosts name {get_host_name()}")

        # vqnetrun -np 4 --hostfile hosts --output-filename output  python test.py


verbose
^^^^^^^^^^^^^^^^^^^^^^

La interfaz ``vqnetrun`` se puede utilizar con el parámetro de línea de comandos ``--verbose`` para habilitar el registro detallado de la comunicación y mostrar los resultados.

Un ejemplo de implementación es el siguiente

    Ejemplo::

        from pyvqnet.distributed import CommController, get_host_name
        Comm_OP = CommController("mpi") # inicializa el controlador mpi
        
        rank = Comm_OP.getRank()
        size = Comm_OP.getSize()
        print(f"rank: {rank}, size {size}")
        print(f"LocalRank {Comm_OP.getLocalRank()} hosts name {get_host_name()}")

        # vqnetrun -np 4 --hostfile hosts --verbose python test.py


start-timeout
^^^^^^^^^^^^^^^^^^^^^^

La interfaz ``vqnetrun`` se puede utilizar con el parámetro de línea de comandos ``-start-timeout`` para especificar que se realicen todas las comprobaciones y se inicie el proceso antes de que expire el tiempo de espera. El valor predeterminado es 30 segundos.

Un ejemplo de implementación es el siguiente

    Ejemplo::

        from pyvqnet.distributed import CommController, get_host_name
        Comm_OP = CommController("mpi") # inicializa el controlador mpi
        
        rank = Comm_OP.getRank()
        size = Comm_OP.getSize()
        print(f"rank: {rank}, size {size}")
        print(f"LocalRank {Comm_OP.getLocalRank()} hosts name {get_host_name()}")

        # vqnetrun -np 4 --start-timeout 10 python test.py


CommController
=================================

    La computación distribuida se utiliza para controlar la comunicación de datos entre diferentes procesos en CPU y GPU. Genera diferentes controladores para CPU (MPI) y GPU (NCCL), y llama a los métodos de comunicación para completar la comunicación y sincronización de datos entre diferentes procesos.

__init__
^^^^^^^^^^^^^^^^^^^^^^
.. py:class:: pyvqnet.distributed.ControlComm.CommController(backend,rank=None,world_size=None)
    
    CommController se utiliza para controlar la comunicación de datos en CPU y GPU. Configurando el parámetro `backend`, genera el controlador para CPU (MPI) o GPU (NCCL). (Actualmente, la función de computación distribuida solo es compatible con Linux.)

    :param backend: se utiliza para generar el controlador de comunicación de datos para cpu o gpu.
    :param rank: Este parámetro solo es útil en backends que no son pyvqnet, el valor predeterminado es: None.
    :param world_size: Este parámetro solo es útil en backends que no son pyvqnet, el valor predeterminado es: None.
        
    :return:
        Instancia de CommController.

    Ejemplos::

        from pyvqnet.distributed import CommController
        Comm_OP = CommController("nccl") # inicializa el controlador nccl

        # Comm_OP = CommController("mpi") # inicializa el controlador mpi

 
    .. py:method:: getRank()
        
        Se utiliza para obtener el número de proceso del proceso actual.

        :return: Devuelve el número de proceso del proceso actual.

        Ejemplos::

            from pyvqnet.distributed import CommController
            Comm_OP = CommController("nccl") # inicializa el controlador nccl
            
            Comm_OP.getRank()

 
    .. py:method:: getSize()
    
        Se utiliza para obtener el número total de procesos iniciados.


        :return: Devuelve el número total de procesos.

        Ejemplos::

            from pyvqnet.distributed import CommController
            Comm_OP = CommController("nccl") # inicializa el controlador nccl
            
            Comm_OP.getSize()
            # vqnetrun --backend nccl --nproc_per_node 2 python test.py




    .. py:method:: getLocalRank()
    
        Se utiliza para obtener el número de proceso actual en la máquina actual.


        :return: El número de proceso actual en la máquina actual.

        Ejemplos::

            from pyvqnet.distributed import CommController
            Comm_OP = CommController("nccl") # inicializa el controlador nccl
            
            Comm_OP.getLocalRank()
            # vqnetrun --backend nccl --nproc_per_node 2 python test.py


    .. py:method:: split_groups(rankL)
        
        Divide múltiples grupos de comunicación según la lista de números de proceso establecida por el parámetro de entrada.

        :param rankL: Una lista de rangos de grupos de proceso.

        :return: Cuando el backend es `nccl`, se devuelve una tupla de rangos de grupos de proceso.
                 Cuando el backend es `mpi`, devuelve una lista cuya longitud es igual al número de grupos; cada elemento es una tupla (comm, rank), donde comm es el comunicador MPI del grupo y rank es el número de secuencia dentro del grupo.

        Ejemplos::
            
            from pyvqnet.distributed import CommController,get_rank,get_local_rank
            from pyvqnet.tensor import tensor
            import numpy as np
            Comm_OP = CommController("mpi")

            groups = Comm_OP.split_groups([[0, 1],[2,3]])
            print(groups)
            #[[<mpi4py.MPI.Intracomm object at 0x7f53691f3230>, [0, 3]], [<mpi4py.MPI.Intracomm object at 0x7f53691f3010>, [2, 1]]]

 
    .. py:method:: barrier()
    
        Sincronización.

        :return: Sincronización.

        Ejemplos::

            from pyvqnet.distributed import CommController
            Comm_OP = CommController("nccl")
            
            Comm_OP.barrier()

 
    .. py:method:: get_device_num()
    
        Se utiliza para obtener el número de tarjetas gráficas en el nodo actual, (solo compatible con gpu).

        :return: Devuelve el número de tarjetas gráficas en el nodo actual.

        Ejemplos::

            from pyvqnet.distributed import CommController
            Comm_OP = CommController("nccl")
            
            Comm_OP.get_device_num()
            # python test.py


 
    .. py:method:: allreduce(tensor, c_op = "avg")
    
        Admite la comunicación allreduce de datos.

        :param tensor: Datos de entrada.
        :param c_op: Cálculo.

        Ejemplos::

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
    
        Admite la comunicación reduce de datos.

        :param tensor: entrada.
        :param root: Especifica el nodo al que se devuelven los datos.
        :param c_op: Cálculo.

        Ejemplos::

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
    
        Difunde los datos en el proceso raíz especificado a todos los procesos.

        :param tensor: entrada.
        :param root: Especifica el nodo.

        Ejemplos::

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
    
        Reúne (allgather) los datos de todos los procesos.

        :param tensor: entrada.

        Ejemplos::

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
    
        Interfaz de comunicación p2p.

        :param tensor: entrada.
        :param dest: Proceso de destino.

        Ejemplos::

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
    
        Interfaz de comunicación p2p.

        :param tensor: entrada.
        :param source: Proceso de aceptación.

        Ejemplos::

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
    
        La interfaz de comunicación allreduce de grupo.

        :param tensor: entrada.
        :param c_op: Cálculo.
        :param group: Grupo de comunicación. Cuando se usa el backend mpi, ingrese el grupo generado por `init_groups` o `split_groups` correspondiente al grupo de comunicación. Cuando se usa el backend nccl, ingrese el número de grupo generado por `split_groups`.


        Ejemplos::

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
    
        Interfaz de comunicación reduce dentro del grupo.

        :param tensor: Entrada.
        :param root: Especifique el número de proceso.
        :param c_op: Cálculo.
        :param group: Grupo de comunicación. Cuando se usa el backend mpi, ingrese el grupo generado por `init_groups` o `split_groups` correspondiente al grupo de comunicación. Cuando se usa el backend nccl, ingrese el número de grupo generado por `split_groups`.

        Ejemplos::
            
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
    
        Interfaz de comunicación broadcast dentro del grupo.

        :param tensor: Entrada.
        :param root: Especifique el número de proceso desde el cual difundir, predeterminado: 0.
        :param group: Grupo de comunicación. Cuando se usa el backend mpi, ingrese el grupo generado por `init_groups` o `split_groups` correspondiente al grupo de comunicación. Cuando se usa el backend nccl, ingrese el número de grupo generado por `split_groups`.

        Ejemplos::
            
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
    
        La interfaz de comunicación allgather de grupo.

        :param tensor: Entrada.
        :param group: Grupo de comunicación. Cuando se usa el backend mpi, ingrese el grupo generado por `init_groups` o `split_groups` correspondiente al grupo de comunicación. Cuando se usa el backend nccl, ingrese el número de grupo generado por `split_groups`.

        Ejemplos::
            
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
    
        Actualiza el gradiente de los parámetros en el optimizador con allreduce.

        :param optimizer: optimizador.

        Ejemplos::
            
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
    
        Actualiza los parámetros del modelo mediante allreduce.

        :param model: Modelo.

        Ejemplos::
        
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
    
        Difunde los parámetros del modelo en el número de proceso especificado.

        :param model: Modelo.
        :param root: Especifique el número de proceso.

        Ejemplos::
        
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

        Usa NCCL para all_gather asíncrono o síncrono en datos de GPU.

        :param output: QTensor - El QTensor de destino para el resultado de all_gather.
        :param input: QTensor - El QTensor que se va a reunir.
        :param group: Grupo de proceso de comunicación, group es una tupla que contiene índices de grupo. Predeterminado: None, no se usa ningún grupo.
        :param async_op: Si esta operación es asíncrona, predeterminado: False.
        :return: Work - Un manejador de comunicación asíncrona. Use wait() para esperar a que esta operación se complete.

        Ejemplos::

            from pyvqnet import tensor
            import pyvqnet
            from pyvqnet.distributed import CommController,get_rank,get_local_rank
            Comm_OP = CommController("nccl")

            complex_data = tensor.QTensor([3+1j, 2, 1 + get_rank()],dtype=8).reshape((3,1)).toGPU(1000+ get_local_rank())

            out_data = tensor.empty([2,3,1],dtype=pyvqnet.kcomplex64).toGPU(pyvqnet.DEV_GPU_0+ get_local_rank())
            work = Comm_OP.nccl_async_all_gather(out_data, complex_data, group = None,async_op=True)
            work.wait()

    .. py:method:: nccl_async_all_reduce(tensor, c_op="avg",group=None, async_op=False):

        Usa NCCL para allreduce asíncrono o síncrono en datos de GPU.

        :param tensor: QTensor - El QTensor que necesita ser reducido.
        :param c_op: Método de cálculo, puede ser "sum" o "avg", el valor predeterminado es "avg".
        :param group: Grupo de proceso de comunicación, group es una tupla que contiene índices de grupo. Predeterminado: None, no se usa ningún grupo.
        :param async_op: Si esta operación es asíncrona, predeterminado: False.
        :return: Work - Un manejador de comunicación asíncrona. Use wait() para esperar a que esta operación se complete.

        Ejemplos::

            import pyvqnet
            from pyvqnet.tensor import tensor
            from pyvqnet.distributed.ControlComm import CommController,get_local_rank

            Comm_OP = CommController("nccl")
            complex_data = tensor.ones([500,500],dtype=pyvqnet.kcomplex64).toGPU(pyvqnet.DEV_GPU_0 + get_local_rank())
            work = Comm_OP.nccl_async_all_reduce(complex_data, "sum",group = None,async_op = True)
            work.wait()

    .. py:method:: nccl_async_reduce( tensor_, dest, c_op="avg", group=None, async_op=False ):

        Usa NCCL para reduce asíncrono o síncrono en datos de GPU.

        :param tensor_: QTensor - El QTensor que necesita ser reducido.
        :param dest: El rango de destino del QTensor reducido.
        :param c_op: Método de cálculo, puede ser "sum" o "avg", el valor predeterminado es "avg".
        :param group: Grupo de proceso de comunicación, group es una tupla que contiene índices de grupo. Predeterminado: None, no se usa ningún grupo.
        :param async_op: Si esta operación es asíncrona, predeterminado: False.
        :return: Work - Un manejador de comunicación asíncrona. Use wait() para esperar a que esta operación se complete.

        Ejemplos::

            from pyvqnet import tensor
            import pyvqnet
            from pyvqnet.distributed import CommController,get_rank,get_local_rank
            Comm_OP = CommController("nccl")

            complex_data = tensor.ones([500,500],dtype=pyvqnet.kcomplex64).toGPU(pyvqnet.DEV_GPU_0 + get_local_rank())
            work = Comm_OP.nccl_async_reduce(complex_data, 0,"sum",group = None,async_op = True)
            work.wait()

    .. py:method:: nccl_async_broadcast(tensor, src, group=None, async_op=False )

        Usa NCCL para broadcast asíncrono o síncrono en datos de GPU.

        :param tensor: QTensor - El QTensor que necesita ser difundido.
        :param src: El rango de origen del QTensor a difundir.
        :param group: Grupo de proceso de comunicación, group es una tupla que contiene índices de grupo. Predeterminado: None, no se usa ningún grupo.
        :param async_op: Si esta operación es asíncrona, predeterminado: False.
        :return: Work - Un manejador de comunicación asíncrona. Use wait() para esperar a que esta operación se complete.

        Ejemplos::

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

        Usa NCCL para envío P2P asíncrono o síncrono en datos de GPU.

        :param t: QTensor - El QTensor que necesita ser enviado.
        :param dest: El rango de destino al que enviar el QTensor.
        :param async_op: Si esta operación es asíncrona, predeterminado: False.
        :return: Work - Un manejador de comunicación asíncrona. Use wait() para esperar a que esta operación se complete.

        Ejemplos::

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

        Usa NCCL para recepción P2P asíncrona o síncrona en datos de GPU.

        :param t: QTensor - El QTensor que recibe los datos.
        :param src: El rango de origen del QTensor recibido.
        :param async_op: Si esta operación es asíncrona, predeterminado: False.
        :return: Work - Un manejador de comunicación asíncrona. Use wait() para esperar a que esta operación se complete.

        Ejemplos::

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

En múltiples procesos, use ``split_data`` para dividir los datos según el número de procesos y devolver los datos en el proceso correspondiente.

.. py:function:: pyvqnet.distributed.datasplit.split_data(x_train, y_train, shuffle=False)

Establece parámetros para la computación distribuida.

    :param x_train: `np.array` - datos de entrenamiento.
    :param y_train: `np.array` - Etiquetas de datos de entrenamiento.
    :param shuffle: `bool` - Si se debe mezclar y luego dividir, el valor predeterminado es False.

    :return: datos de entrenamiento y etiquetas divididos.

    Ejemplo::

        from pyvqnet.distributed import split_data
        import numpy as np

        x_train = np.random.randint(255, size = (100, 5))
        y_train = np.random.randint(2, size = (100, 1))

        x_train, y_train= split_data(x_train, y_train)

get_local_rank
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Use ``get_local_rank`` para obtener el número de proceso en la máquina actual.

.. py:function:: pyvqnet.distributed.ControlComm.get_local_rank()

    Se utiliza para obtener el número de proceso actual en la máquina actual.

    :return: número de proceso actual en la máquina actual.

    Ejemplo::

        from pyvqnet.distributed.ControlComm import get_local_rank

        print(get_local_rank())
        # vqnetrun -n 2 python test.py

get_rank
~~~~~~~~~~~~~~~~~~~~~~~~~~~
Use ``get_rank`` para obtener el número de proceso en la máquina actual.

.. py:function:: pyvqnet.distributed.ControlComm.get_rank()

    Se utiliza para obtener el número de proceso del proceso actual.

    :return: el número de proceso del proceso actual.

    Ejemplo::

        from pyvqnet.distributed.ControlComm import get_rank

        print(get_rank())
        # vqnetrun -n 2 python test.py

init_groups
~~~~~~~~~~~~~~~~~~~~~~~~~~~
Use ``init_groups`` para inicializar grupos de procesos basados en CPU según la lista dada de números de proceso.

.. py:function:: pyvqnet.distributed.ControlComm.init_groups(rank_lists)

    Se utiliza para inicializar el grupo de comunicación de procesos para el backend `mpi`.

    :param rank_lists: Lista de grupos de comunicación de procesos.
    :return: Una lista de grupos de proceso inicializados.

    Ejemplo::

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
    
    Pipeline Parallel Training Wrapper implementa el entrenamiento 1F1B. Solo disponible en plataformas Linux con GPU.
    Más detalles del algoritmo se pueden encontrar en (https://www.deepspeed.ai/tutorials/pipeline/).

    :param args: Diccionario de parámetros. Ver ejemplos.
    :param join_layers: Lista de módulos Sequential.
    :param trainset: Conjunto de datos.

    :return:
        Instancia de PipelineParallelTrainingWrapper.

    El siguiente ejemplo utiliza el conjunto de datos CIFAR10 `CIFAR10_Dataset` para entrenar la tarea de clasificación en AlexNet en 2 GPUs.
    En este ejemplo, se divide en dos procesos de pipeline paralelo `pipeline_parallel_size` = 2.
    El tamaño del lote es `train_batch_size` = 64, en una sola GPU es `train_micro_batch_size_per_gpu` = 32.
    Otros parámetros de configuración se pueden encontrar en `args`.
    Además, cada proceso necesita configurar la variable de entorno `LOCAL_RANK` en la función `__main__`.

    Ejemplos::

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
    
    Interfaz de la api Zero1, actualmente solo para plataforma Linux basada en computación paralela con GPU.

    :param args: diccionario de parámetros.
    :param model: Module.
    :param optimizer: Optimizador.

    :return:
        Motor Zero1.

El siguiente ejemplo utiliza el conjunto de datos MNIST para entrenar una tarea de clasificación en un modelo MLP en 2 GPUs.

    El tamaño del lote es `train_batch_size` = 64, y la etapa `stage` de `zero_optimization` se establece en 1.
    Si Optimizer es None, se utiliza la configuración de `optimizer` en `args`. Otros parámetros de configuración se pueden encontrar en `args`.
    
    Además, cada proceso necesita configurar la variable de entorno `LOCAL_RANK`.
    
    .. code-block::

        os.environ["LOCAL_RANK"] = str(dist.get_local_rank())

    Ejemplos::

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
            carga datos mnist
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
    
    Computación de tensor paralelo con capa lineal de columnas paralelas
    
    La capa lineal se define como Y = XA + b. Sus filas paralelas 2D son A = [A_1, ... , A_p].

    :param input_size: primera dimensión de la matriz A.
    :param output_size: segunda dimensión de la matriz A.
    :param weight_initializer: `callable` - predeterminado normal.
    :param bias_initializer: `callable` - predeterminado ceros.
    :param use_bias: `bool` - predeterminado True.
    :param dtype: predeterminado: None, usa el tipo de dato predeterminado.
    :param name: nombre del módulo, predeterminado: "".
    :param tp_comm: Controlador de comunicación (Comm Controller).

    El siguiente ejemplo utiliza el conjunto de datos MNIST para entrenar una tarea de clasificación en un modelo MLP en 2 GPUs.

    El uso es similar al de la capa Linear clásica.

    Uso multiproceso basado en `# vqnetrun --backend nccl --nproc_per_node 2 python test.py`.

    Ejemplos::

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
            carga datos mnist
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
    
    Computación de tensor paralelo con capa lineal de filas paralelas.

    La capa lineal se define como Y = XA + b. A se paraleliza a lo largo de su primera dimensión y X a lo largo de su segunda dimensión.
    A = transpose([A_1 .. A_p]) X = [X_1, ..., X_p].

    :param input_size: primera dimensión de la matriz A.
    :param output_size: segunda dimensión de la matriz A.
    :param weight_initializer: `callable` - predeterminado normal.
    :param bias_initializer: `callable` - predeterminado ceros.
    :param use_bias: `bool` - predeterminado True.
    :param dtype: predeterminado: None, usa el tipo de dato predeterminado.
    :param name: nombre del módulo, predeterminado: "".
    :param tp_comm: Controlador de comunicación (Comm Controller).

    El siguiente ejemplo utiliza el conjunto de datos MNIST para entrenar una tarea de clasificación en un modelo MLP en 2 GPUs.

    El uso es similar al de la capa Linear clásica.

    Uso multiproceso basado en `# vqnetrun --backend nccl --nproc_per_node 2 python test.py`.

    Ejemplos::

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
            carga datos mnist
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


Reordenamiento de Bits
=================================

El reordenamiento de qubits es una técnica en el paralelismo de bits. Su objetivo principal es reducir el número de transformaciones de bits requeridas por el paralelismo de bits cambiando el orden de las compuertas lógicas cuánticas. Los siguientes módulos son necesarios para construir circuitos cuánticos de bits grandes basados en paralelismo de bits. Consulte el artículo `Lazy Qubit Reordering for Accelerating Parallel State-Vector-based Quantum Circuit Simulation <https://export.arxiv.org/abs/2410.04252>`__.

Las siguientes interfaces requieren que `mpi` lance múltiples procesos para la computación.

DistributeQMachine
^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. py:class:: pyvqnet.distributed.qubits_reorder.DistributeQMachine(num_wires, dtype, grad_mode)
    
    Una clase para simular cómputos cuánticos variacionales de bits paralelos, incluyendo estados cuánticos en un subconjunto de bits en cada nodo. Cada nodo solicita una simulación de circuito cuántico variacional distribuido a través de MPI. El valor de N debe ser igual a una potencia de 2 elevada al número de bits paralelos distribuidos, `global_qubit`, y puede configurarse mediante `set_qr_config`.

    :param num_wires: El número de bits en el circuito cuántico completo.
    :param dtype: El tipo de datos de los datos de cómputo. El valor predeterminado es pyvqnet.kcomplex64, correspondiente a la precisión de parámetros de pyvqnet.kfloat32.
    :param grad_mode: Establecer en adjoint al retropropagar ``DistQuantumLayerAdjoint``.

    .. note::

        El número de bits de entrada es el número de bits requeridos para todo el circuito cuántico. DistributeQMachine construirá un simulador cuántico basado en el número global de bits, que es ``num_wires - global_qubit``.

        La retropropagación debe basarse en ``DistQuantumLayerAdjoint``.

    .. warning::

        Esta interfaz solo es compatible con Linux;

        Los parámetros de bits paralelos en ``DistributeQMachine`` deben configurarse, como se muestra en el ejemplo, incluyendo:

        .. code-block::

            qm.set_just_defined(True)
            qm.set_save_op_history_flag(True) # activa guardar op
            qm.set_qr_config({'qubit': número total de qubits, 'global_qubit': número de qubits distribuidos})
    
    
    Ejemplos::

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
                self.qm.set_save_op_history_flag(True) # activa guardar op
                self.qm.set_qr_config({"qubit": num_wires, # activa reordenamiento de qubits, establece configuración
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
    
    Una capa DistQuantumLayer que calcula los gradientes de los parámetros en cómputos de bits paralelos utilizando el enfoque de matriz adjunta.

    :param vqc_module: El módulo ``DistributeQMachine`` implícito de entrada.
    :param name: El nombre del módulo.

    .. note::

        El módulo vqc_module de entrada debe contener ``DistributeQMachine``. Los cálculos de gradiente de retropropagación adjunta se realizan basándose en ``DistributeQMachine`` en cómputos de bits paralelos.

    .. warning::

        Esta interfaz solo es compatible con Linux;


    Ejemplos::

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
                self.qm.set_save_op_history_flag(True) # activa guardar op
                self.qm.set_qr_config({"qubit": num_wires, # activa reordenamiento de qubits, establece configuración
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
