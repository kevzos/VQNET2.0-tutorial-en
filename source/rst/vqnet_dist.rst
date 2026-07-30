
.. _vqnet_dist:

Módulo de Computação Distribuída Ingênua do VQNet
*********************************************************

Implantação do Ambiente
=================================

Esta seção descreve como implantar o ambiente VQNet no Linux para computação distribuída em CPU e GPU.

Instalação do MPI
^^^^^^^^^^^^^^^^^^^^^^

MPI é uma biblioteca comum para comunicação entre CPUs, e a função de computação distribuída da CPU no VQNet é implementada com base no MPI.
A seção a seguir descreve como instalar o MPI em um sistema Linux (atualmente, a função de computação distribuída baseada em CPU é suportada apenas no Linux).

Detecte se os compiladores gcc e gfortran estão instalados.

.. code-block::
        
    which gcc 
    which gfortran

Quando os caminhos para gcc e gfortran forem exibidos, você pode prosseguir para a próxima etapa da instalação. Se você não tiver os compiladores correspondentes,
instale-os primeiro. Após verificar os compiladores, use o comando wget para baixar o mpich.

.. code-block::
        
    wget http://www.mpich.org/static/downloads/3.3.2/mpich-3.3.2.tar.gz 
    tar -zxvf mpich-3.3.2.tar.gz 
    cd mpich-3.3.2 
    ./configure --prefix=/usr/local/mpich
    make 
    make install 

Após compilar e instalar o mpich, configure suas variáveis de ambiente.

.. code-block::
        
    vim ~/.bashrc

    # No final do documento, adicione
    export PATH="/usr/local/mpich/bin:$PATH"

Após salvar e sair, use source para aplicar as alterações.

.. code-block::

    source ~/.bashrc

Use which para verificar se as variáveis de ambiente estão configuradas corretamente. Se o caminho for exibido, a instalação foi concluída com sucesso.

Além disso, você pode instalar o mpi4py via pip install. Se encontrar o seguinte erro:

.. image:: ./images/mpi_bug.png
    :align: center

|

Para resolver a incompatibilidade entre mpi4py e a versão do Python, você pode fazer o seguinte:

.. code-block::

    # Faça backup do compilador do ambiente Python atual
    pushd /root/anaconda3/envs/$CONDA_DEFAULT_ENV/compiler_compat && mv ld ld.bak && popd

    # Reinstale o mpi4py
    pip install mpi4py

    # Restaure o compilador original
    pushd /root/anaconda3/envs/$CONDA_DEFAULT_ENV/compiler_compat && mv ld.bak ld && popd

Instalação do NCCL
^^^^^^^^^^^^^^^^^^^^^^

NCCL é uma biblioteca comum para comunicação em GPU, e a função de computação distribuída das GPUs no VQNet é implementada com base no NCCL.
Este software instala a biblioteca de vínculo dinâmico do NCCL por padrão durante a instalação, portanto, geralmente não é necessário instalar o NCCL separadamente.
Esta seção requer suporte MPI, portanto, o ambiente MPI também precisa ser implantado.

Inicialização Distribuída
=================================
 
Usando a Interface de Computação Distribuída iniciada pelo comando ``vqnetrun``, os parâmetros do ``vqnetrun`` são descritos abaixo.

n, np
^^^^^^^^^^^^^^^^^^^^^^

A interface ``vqnetrun`` permite controlar o número de processos iniciados com os parâmetros ``-n``, ``-np``, conforme mostrado no exemplo a seguir.

    Exemplo::

        from pyvqnet.distributed import CommController
        Comm_OP = CommController("mpi") # inicializa o controlador mpi
        
        rank = Comm_OP.getRank()
        size = Comm_OP.getSize()
        print(f"rank: {rank}, size {size}")

        # vqnetrun -n 2 python test.py
        # vqnetrun -np 2 python test.py

backend
^^^^^^^^^^^^^^^^^^^^^^

A interface ``vqnetrun`` permite selecionar o backend distribuído com o parâmetro ``--backend``, suportando os modos ``mpi`` (padrão) e ``nccl``.
O modo MPI é para computação distribuída baseada em CPU, e o modo NCCL é para computação distribuída baseada em GPU.

    Exemplo::

        from pyvqnet.distributed import CommController
        Comm_OP = CommController("nccl")

        rank = Comm_OP.getRank()
        size = Comm_OP.getSize()
        print(f"rank: {rank}, size {size}")

        # vqnetrun --backend nccl --nproc_per_node 2 python test.py

nproc_per_node
^^^^^^^^^^^^^^^^^^^^^^

A interface ``vqnetrun`` permite controlar o número de processos iniciados em cada nó com o parâmetro ``--nproc_per_node``, disponível apenas no modo NCCL.

    Exemplo::

        from pyvqnet.distributed import CommController
        Comm_OP = CommController("nccl")

        rank = Comm_OP.getRank()
        size = Comm_OP.getSize()
        print(f"rank: {rank}, size {size}")

        # vqnetrun --backend nccl --nproc_per_node 4 python test.py

output-filename
^^^^^^^^^^^^^^^^^^^^^^

A interface ``vqnetrun`` permite salvar a saída em um arquivo especificado com o parâmetro de linha de comando ``--output-filename``.

Um exemplo de implementação é o seguinte:

    Exemplo::

        from pyvqnet.distributed import CommController, get_host_name
        Comm_OP = CommController("mpi") # inicializa o controlador mpi
        
        rank = Comm_OP.getRank()
        size = Comm_OP.getSize()
        print(f"rank: {rank}, size {size}")
        print(f"LocalRank {Comm_OP.getLocalRank()} hosts name {get_host_name()}")

        # vqnetrun -np 4 --hostfile hosts --output-filename output  python test.py


verbose
^^^^^^^^^^^^^^^^^^^^^^

A interface ``vqnetrun`` pode ser usada com o parâmetro de linha de comando ``--verbose`` para ativar o registro detalhado de comunicação e exibir os resultados.

Um exemplo de implementação é o seguinte:

    Exemplo::

        from pyvqnet.distributed import CommController, get_host_name
        Comm_OP = CommController("mpi") # inicializa o controlador mpi
        
        rank = Comm_OP.getRank()
        size = Comm_OP.getSize()
        print(f"rank: {rank}, size {size}")
        print(f"LocalRank {Comm_OP.getLocalRank()} hosts name {get_host_name()}")

        # vqnetrun -np 4 --hostfile hosts --verbose python test.py


start-timeout
^^^^^^^^^^^^^^^^^^^^^^

A interface ``vqnetrun`` pode ser usada com o parâmetro de linha de comando ``-start-timeout`` para especificar que todas as verificações sejam realizadas e o processo seja iniciado antes do tempo limite. O valor padrão é 30 segundos.

Um exemplo de implementação é o seguinte:

    Exemplo::

        from pyvqnet.distributed import CommController, get_host_name
        Comm_OP = CommController("mpi") # inicializa o controlador mpi
        
        rank = Comm_OP.getRank()
        size = Comm_OP.getSize()
        print(f"rank: {rank}, size {size}")
        print(f"LocalRank {Comm_OP.getLocalRank()} hosts name {get_host_name()}")

        # vqnetrun -np 4 --start-timeout 10 python test.py


CommController
=================================

    A computação distribuída é usada para controlar a comunicação de dados entre diferentes processos em CPU e GPU. Ela gera diferentes controladores para CPU (MPI) e GPU (NCCL) e chama os métodos de comunicação para realizar a comunicação e sincronização de dados entre diferentes processos.

__init__
^^^^^^^^^^^^^^^^^^^^^^
.. py:class:: pyvqnet.distributed.ControlComm.CommController(backend,rank=None,world_size=None)
    
    CommController é usado para controlar a comunicação de dados em CPU e GPU. Ao definir o parâmetro `backend`, ele gera o controlador para CPU (MPI) ou GPU (NCCL). (Atualmente, a função de computação distribuída suporta apenas Linux.)

    :param backend: usado para gerar o controlador de comunicação de dados para cpu ou gpu.
    :param rank: Este parâmetro é útil apenas em backends não pyvqnet. O valor padrão é: None.
    :param world_size: Este parâmetro é útil apenas em backends não pyvqnet. O valor padrão é: None.
        
    :return:
        Instância de CommController.

    Examples::

        from pyvqnet.distributed import CommController
        Comm_OP = CommController("nccl") # inicializa o controlador nccl

        # Comm_OP = CommController("mpi") # inicializa o controlador mpi


    .. py:method:: getRank()
        
        Usado para obter o número do processo atual.

        :return: Retorna o número do processo atual.

        Examples::

            from pyvqnet.distributed import CommController
            Comm_OP = CommController("nccl") # inicializa o controlador nccl
            
            Comm_OP.getRank()

 
    .. py:method:: getSize()
    
        Usado para obter o número total de processos iniciados.


        :return: Retorna o número total de processos.

        Examples::

            from pyvqnet.distributed import CommController
            Comm_OP = CommController("nccl") # inicializa o controlador nccl
            
            Comm_OP.getSize()
            # vqnetrun --backend nccl --nproc_per_node 2 python test.py




    .. py:method:: getLocalRank()
    
        Usado para obter o número do processo atual na máquina atual.


        :return: O número do processo atual na máquina atual.

        Examples::

            from pyvqnet.distributed import CommController
            Comm_OP = CommController("nccl") # inicializa o controlador nccl
            
            Comm_OP.getLocalRank()
            # vqnetrun --backend nccl --nproc_per_node 2 python test.py


    .. py:method:: split_groups(rankL)
        
        Divide múltiplos grupos de comunicação de acordo com a lista de números de processos definida pelo parâmetro de entrada.

        :param rankL: Uma lista de ranks dos grupos de processos.

        :return: Quando o backend é `nccl`, uma tupla de ranks dos grupos de processos é retornada.
                 Quando o backend é `mpi`, retorna uma lista cujo comprimento é igual ao número de grupos; cada elemento é uma tupla (comm, rank), onde comm é o comunicador MPI do grupo e rank é o número de sequência dentro do grupo.

        Examples::
            
            from pyvqnet.distributed import CommController,get_rank,get_local_rank
            from pyvqnet.tensor import tensor
            import numpy as np
            Comm_OP = CommController("mpi")

            groups = Comm_OP.split_groups([[0, 1],[2,3]])
            print(groups)
            #[[<mpi4py.MPI.Intracomm object at 0x7f53691f3230>, [0, 3]], [<mpi4py.MPI.Intracomm object at 0x7f53691f3010>, [2, 1]]]

 
    .. py:method:: barrier()
    
        Sincronização.

        :return: Sincronização.

        Examples::

            from pyvqnet.distributed import CommController
            Comm_OP = CommController("nccl")
            
            Comm_OP.barrier()

 
    .. py:method:: get_device_num()
    
        Usado para obter o número de placas gráficas no nó atual (suportado apenas em gpu).

        :return: Retorna o número de placas gráficas no nó atual.

        Examples::

            from pyvqnet.distributed import CommController
            Comm_OP = CommController("nccl")
            
            Comm_OP.get_device_num()
            # python test.py


 
    .. py:method:: allreduce(tensor, c_op = "avg")
    
        Suporta comunicação allreduce de dados.

        :param tensor: Dados de entrada.
        :param c_op: Cálculo.

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
    
        Suporta comunicação reduce de dados.

        :param tensor: entrada.
        :param root: Especifica o nó para o qual os dados são retornados.
        :param c_op: Cálculo.

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
    
        Transmite dados do processo raiz especificado para todos os processos.

        :param tensor: entrada.
        :param root: Especifica o nó.

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
    
        Reúne (allgather) os dados de todos os processos.

        :param tensor: entrada.

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
    
        Interface de comunicação p2p.

        :param tensor: entrada.
        :param dest: Processo de destino.

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
    
        Interface de comunicação p2p.

        :param tensor: entrada.
        :param source: Processo de aceitação.

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
    
        A interface de comunicação allreduce em grupo.

        :param tensor: entrada.
        :param c_op: Cálculo.
        :param group: Grupo de comunicação. Ao usar o backend mpi, insira o grupo gerado por `init_groups` ou `split_groups` correspondente ao grupo de comunicação. Ao usar o backend nccl, insira o número do grupo gerado por `split_groups`.


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
    
        Interface de comunicação reduce intra-grupo.

        :param tensor: Entrada.
        :param root: Especifica o número do processo.
        :param c_op: Cálculo.
        :param group: Grupo de comunicação. Ao usar o backend mpi, insira o grupo gerado por `init_groups` ou `split_groups` correspondente ao grupo de comunicação. Ao usar o backend nccl, insira o número do grupo gerado por `split_groups`.

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
    
        Interface de comunicação broadcast intra-grupo.

        :param tensor: Entrada.
        :param root: Especifica o número do processo de origem da transmissão, padrão: 0.
        :param group: Grupo de comunicação. Ao usar o backend mpi, insira o grupo gerado por `init_groups` ou `split_groups` correspondente ao grupo de comunicação. Ao usar o backend nccl, insira o número do grupo gerado por `split_groups`.

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
    
        A interface de comunicação allgather em grupo.

        :param tensor: Entrada.
        :param group: Grupo de comunicação. Ao usar o backend mpi, insira o grupo gerado por `init_groups` ou `split_groups` correspondente ao grupo de comunicação. Ao usar o backend nccl, insira o número do grupo gerado por `split_groups`.

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
    
        Atualiza o gradiente dos parâmetros no otimizador com allreduce.

        :param optimizer: otimizador.

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
    
        Atualiza os parâmetros no modelo de forma allreduce.

        :param model: Modelo.

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
    
        Transmite os parâmetros do modelo no número de processo especificado.

        :param model: Modelo.
        :param root: Especifica o número do processo.

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

        Usa NCCL para all_gather assíncrono ou síncrono em dados de GPU.

        :param output: QTensor - O QTensor de destino para o resultado all_gather.
        :param input: QTensor - O QTensor a ser reunido.
        :param group: Grupo de processo de comunicação. group é uma tupla contendo índices de grupo. Padrão: None, nenhum grupo é usado.
        :param async_op: Se esta operação é assíncrona, padrão: False.
        :return: Work - Um manipulador de comunicação assíncrona. Use wait() para aguardar a conclusão desta operação.

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

        Usa NCCL para allreduce assíncrono ou síncrono em dados de GPU.

        :param tensor: QTensor - O QTensor que precisa ser reduzido.
        :param c_op: Método de computação, pode ser "sum" ou "avg", valor padrão é "avg".
        :param group: Grupo de processo de comunicação. group é uma tupla contendo índices de grupo. Padrão: None, nenhum grupo é usado.
        :param async_op: Se esta operação é assíncrona, padrão: False.
        :return: Work - Um manipulador de comunicação assíncrona. Use wait() para aguardar a conclusão desta operação.

        Examples::

            import pyvqnet
            from pyvqnet.tensor import tensor
            from pyvqnet.distributed.ControlComm import CommController,get_local_rank

            Comm_OP = CommController("nccl")
            complex_data = tensor.ones([500,500],dtype=pyvqnet.kcomplex64).toGPU(pyvqnet.DEV_GPU_0 + get_local_rank())
            work = Comm_OP.nccl_async_all_reduce(complex_data, "sum",group = None,async_op = True)
            work.wait()

    .. py:method:: nccl_async_reduce( tensor_, dest, c_op="avg", group=None, async_op=False ):

        Usa NCCL para reduce assíncrono ou síncrono em dados de GPU.

        :param tensor_: QTensor - O QTensor que precisa ser reduzido.
        :param dest: O rank de destino do QTensor reduzido.
        :param c_op: Método de computação, pode ser "sum" ou "avg", valor padrão é "avg".
        :param group: Grupo de processo de comunicação. group é uma tupla contendo índices de grupo. Padrão: None, nenhum grupo é usado.
        :param async_op: Se esta operação é assíncrona, padrão: False.
        :return: Work - Um manipulador de comunicação assíncrona. Use wait() para aguardar a conclusão desta operação.

        Examples::

            from pyvqnet import tensor
            import pyvqnet
            from pyvqnet.distributed import CommController,get_rank,get_local_rank
            Comm_OP = CommController("nccl")

            complex_data = tensor.ones([500,500],dtype=pyvqnet.kcomplex64).toGPU(pyvqnet.DEV_GPU_0 + get_local_rank())
            work = Comm_OP.nccl_async_reduce(complex_data, 0,"sum",group = None,async_op = True)
            work.wait()

    .. py:method:: nccl_async_broadcast(tensor, src, group=None, async_op=False )

        Usa NCCL para broadcast assíncrono ou síncrono em dados de GPU.

        :param tensor: QTensor - O QTensor que precisa ser transmitido.
        :param src: O rank de origem do QTensor transmitido.
        :param group: Grupo de processo de comunicação. group é uma tupla contendo índices de grupo. Padrão: None, nenhum grupo é usado.
        :param async_op: Se esta operação é assíncrona, padrão: False.
        :return: Work - Um manipulador de comunicação assíncrona. Use wait() para aguardar a conclusão desta operação.

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

        Usa NCCL para envio P2P assíncrono ou síncrono em dados de GPU.

        :param t: QTensor - O QTensor que precisa ser enviado.
        :param dest: O rank de destino para onde enviar o QTensor.
        :param async_op: Se esta operação é assíncrona, padrão: False.
        :return: Work - Um manipulador de comunicação assíncrona. Use wait() para aguardar a conclusão desta operação.

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

        Usa NCCL para recebimento P2P assíncrono ou síncrono em dados de GPU.

        :param t: QTensor - O QTensor que recebe os dados.
        :param src: O rank de origem do QTensor recebido.
        :param async_op: Se esta operação é assíncrona, padrão: False.
        :return: Work - Um manipulador de comunicação assíncrona. Use wait() para aguardar a conclusão desta operação.

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

Em multiprocesso, use ``split_data`` para fatiar os dados de acordo com o número de processos e retornar os dados no processo correspondente.

.. py:function:: pyvqnet.distributed.datasplit.split_data(x_train, y_train, shuffle=False)

Define parâmetros para computação distribuída.

    :param x_train: `np.array` - dados de treinamento.
    :param y_train: `np.array` - Rótulos dos dados de treinamento.
    :param shuffle: `bool` - Se deve embaralhar e depois fatiar, o padrão é False.

    :return: dados de treinamento e rótulos fatiados.

    Example::

        from pyvqnet.distributed import split_data
        import numpy as np

        x_train = np.random.randint(255, size = (100, 5))
        y_train = np.random.randint(2, size = (100, 1))

        x_train, y_train= split_data(x_train, y_train)

get_local_rank
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Use ``get_local_rank`` para obter o número do processo na máquina atual.

.. py:function:: pyvqnet.distributed.ControlComm.get_local_rank()

    Usado para obter o número do processo atual na máquina atual.

    :return: número do processo atual na máquina atual.

    Example::

        from pyvqnet.distributed.ControlComm import get_local_rank

        print(get_local_rank())
        # vqnetrun -n 2 python test.py

get_rank
~~~~~~~~~~~~~~~~~~~~~~~~~~~
Use ``get_rank`` para obter o número do processo na máquina atual.

.. py:function:: pyvqnet.distributed.ControlComm.get_rank()

    Usado para obter o número do processo atual.

    :return: o número do processo atual.

    Example::

        from pyvqnet.distributed.ControlComm import get_rank

        print(get_rank())
        # vqnetrun -n 2 python test.py

init_groups
~~~~~~~~~~~~~~~~~~~~~~~~~~~
Use ``init_groups`` para inicializar grupos de processos baseados em CPU de acordo com a lista fornecida de números de processos.

.. py:function:: pyvqnet.distributed.ControlComm.init_groups(rank_lists)

    Usado para inicializar o grupo de comunicação de processos para o backend `mpi`.

    :param rank_lists: Lista de grupos de comunicação de processos.
    :return: Uma lista de grupos de processos inicializados.

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
    
    PipelineParallelTrainingWrapper implementa o treinamento 1F1B. Disponível apenas em plataformas Linux com GPU.
    Mais detalhes do algoritmo podem ser encontrados em (https://www.deepspeed.ai/tutorials/pipeline/).

    :param args: Dicionário de parâmetros. Veja os exemplos.
    :param join_layers: Lista de módulos Sequential.
    :param trainset: Conjunto de dados.

    :return:
        Instância de PipelineParallelTrainingWrapper.

    O exemplo a seguir usa o conjunto de dados CIFAR10 `CIFAR10_Dataset` para treinar a tarefa de classificação no AlexNet em 2 GPUs.
    Neste exemplo, ele é dividido em dois processos de pipeline paralelo `pipeline_parallel_size` = 2.
    O tamanho do lote é `train_batch_size` = 64, em uma única GPU é `train_micro_batch_size_per_gpu` = 32.
    Outros parâmetros de configuração podem ser encontrados em `args`.
    Além disso, cada processo precisa configurar a variável de ambiente `LOCAL_RANK` na função `__main__`.

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
    
    Interface da api Zero1, atualmente apenas para plataforma Linux baseada em computação paralela em GPU.

    :param args: dicionário de parâmetros.
    :param model: Module.
    :param optimizer: Optimizer.

    :return:
        Zero1 Engine.

O exemplo a seguir usa o conjunto de dados MNIST para treinar uma tarefa de classificação em um modelo MLP em 2 GPUs.

    O tamanho do lote é `train_batch_size` = 64, e o estágio `stage` de `zero_optimization` é definido como 1.
    Se Optimizer for None, a configuração de `optimizer` em `args` é usada. Outros parâmetros de configuração podem ser encontrados em `args`.
    
    Além disso, cada processo precisa ser configurado com a variável de ambiente `LOCAL_RANK`.
    
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
    
    Computação tensor-paralela com camada linear em colunas paralelas
    
    A camada linear é definida como Y = XA + b. Suas linhas paralelas 2D são A = [A_1, ... , A_p].

    :param input_size: primeira dimensão da matriz A.
    :param output_size: segunda dimensão da matriz A.
    :param weight_initializer: `callable` - padrão é normal.
    :param bias_initializer: `callable` - padrão é zeros.
    :param use_bias: `bool` - padrão é True.
    :param dtype: padrão: None, usa o tipo de dado padrão.
    :param name: nome do módulo, padrão: "".
    :param tp_comm: Comm Controller.

    O exemplo a seguir usa o conjunto de dados MNIST para treinar uma tarefa de classificação em um modelo MLP em 2 GPUs.

    O uso é semelhante ao da camada Linear clássica.

    Uso multiprocesso baseado em `# vqnetrun --backend nccl --nproc_per_node 2 python test.py`.

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
    
    Computação tensor-paralela com camada linear em linhas paralelas.

    A camada linear é definida como Y = XA + b. A é paralelizado ao longo de sua primeira dimensão e X ao longo de sua segunda dimensão.
    A = transpose([A_1 .. A_p]) X = [X_1, ..., X_p].

    :param input_size: primeira dimensão da matriz A.
    :param output_size: segunda dimensão da matriz A.
    :param weight_initializer: `callable` - padrão é normal.
    :param bias_initializer: `callable` - padrão é zeros.
    :param use_bias: `bool` - padrão é True.
    :param dtype: padrão: None, usa o tipo de dado padrão.
    :param name: nome do módulo, padrão: "".
    :param tp_comm: Comm Controller.

    O exemplo a seguir usa o conjunto de dados MNIST para treinar uma tarefa de classificação em um modelo MLP em 2 GPUs.

    O uso é semelhante ao da camada Linear clássica.

    Uso multiprocesso baseado em `# vqnetrun --backend nccl --nproc_per_node 2 python test.py`.

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


Reordenação de Qubits
=================================

A reordenação de qubits é uma técnica em paralelismo de bits. Seu objetivo principal é reduzir o número de transformações de bits necessárias no paralelismo de bits, alterando a ordem das portas lógicas quânticas. Os seguintes módulos são necessários para construir circuitos quânticos de grandes bits baseados em paralelismo de bits. Consulte o artigo `Lazy Qubit Reordering for Accelerating Parallel State-Vector-based Quantum Circuit Simulation <https://export.arxiv.org/abs/2410.04252>`__.

As interfaces a seguir exigem `mpi` para iniciar múltiplos processos para computação.

DistributeQMachine
^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. py:class:: pyvqnet.distributed.qubits_reorder.DistributeQMachine(num_wires, dtype, grad_mode)
    
    Uma classe para simular computações quânticas variacionais com paralelismo de bits, incluindo estados quânticos em um subconjunto de bits em cada nó. Cada nó solicita uma simulação de circuito quântico variacional distribuída via MPI. O valor de N deve ser igual a uma potência de 2 elevada ao número de bits paralelos distribuídos, `global_qubit`, e pode ser configurado via `set_qr_config`.

    :param num_wires: O número de bits no circuito quântico completo.
    :param dtype: O tipo de dado dos dados de computação. O padrão é pyvqnet.kcomplex64, correspondente à precisão do parâmetro pyvqnet.kfloat32.
    :param grad_mode: Definido como adjoint ao retropropagar ``DistQuantumLayerAdjoint``.

    .. note::

        O número de bits de entrada é o número de bits necessários para todo o circuito quântico. DistributeQMachine construirá um simulador quântico baseado no número global de bits, que é ``num_wires - global_qubit``.

        A retropropagação deve ser baseada em ``DistQuantumLayerAdjoint``.

    .. warning::

        Esta interface só suporta execução no Linux;

        Os parâmetros de paralelismo de bits em ``DistributeQMachine`` devem ser configurados, conforme mostrado no exemplo, incluindo:

        .. code-block::

            qm.set_just_defined(True)
            qm.set_save_op_history_flag(True) # ativa salvamento de operações
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
                self.qm.set_save_op_history_flag(True) # ativa salvamento de operações
                self.qm.set_qr_config({"qubit": num_wires, # ativa reordenação de qubits, define config
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
    
    Uma camada DistQuantumLayer que calcula gradientes para parâmetros em computações com paralelismo de bits usando a abordagem de matriz adjunta.

    :param vqc_module: O módulo ``DistributeQMachine`` implícito de entrada.
    :param name: O nome do módulo.

    .. note::

        O módulo vqc_module de entrada deve conter ``DistributeQMachine``. Os cálculos de gradiente de retropropagação adjunta são realizados com base em ``DistributeQMachine`` em computações com paralelismo de bits.

    .. warning::

        Esta interface é suportada apenas no Linux;


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
                self.qm.set_save_op_history_flag(True) # ativa salvamento de operações
                self.qm.set_qr_config({"qubit": num_wires, # ativa reordenação de qubits, define config
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