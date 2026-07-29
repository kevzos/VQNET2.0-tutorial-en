
.. _vqnet_dist:

VQNet 基础分布式计算模块
*********************************************************

环境部署
=================================

本节介绍如何在 Linux 上为 CPU 和 GPU 分布式计算部署 VQNet 环境。

MPI 安装
^^^^^^^^^^^^^^^^^^^^^^

MPI 是 CPU 间通信的常用库，VQNet 中 CPU 的分布式计算功能基于 MPI 实现。
以下部分介绍如何在 Linux 系统上安装 MPI（目前，基于 CPU 的分布式计算功能仅在 Linux 上受支持）。

检测是否已安装 gcc、gfortran 编译器。

.. code-block::
        
    which gcc 
    which gfortran

当显示 gcc 和 gfortran 的路径时，即可继续进行下一步安装。如果没有相应的编译器，
请先安装它们。确认编译器就绪后，使用 wget 命令下载 mpich。

.. code-block::
        
    wget http://www.mpich.org/static/downloads/3.3.2/mpich-3.3.2.tar.gz 
    tar -zxvf mpich-3.3.2.tar.gz 
    cd mpich-3.3.2 
    ./configure --prefix=/usr/local/mpich
    make 
    make install 

编译安装 mpich 后，配置其环境变量。

.. code-block::
        
    vim ~/.bashrc

    # 在文档末尾，添加
    export PATH="/usr/local/mpich/bin:$PATH"

保存退出后，使用 source 命令使更改生效。

.. code-block::

    source ~/.bashrc

使用 which 验证环境变量配置是否正确。如果显示路径，则表示安装成功。

此外，可以通过 pip install 安装 mpi4py。如果遇到以下错误：

.. image:: ./images/mpi_bug.png
    :align: center

|

要解决 mpi4py 与 Python 版本不兼容的问题，可以执行以下操作：

.. code-block::

    # 备份当前 Python 环境的编译器
    pushd /root/anaconda3/envs/$CONDA_DEFAULT_ENV/compiler_compat && mv ld ld.bak && popd

    # 重新安装 mpi4py
    pip install mpi4py

    # 恢复原始编译器
    pushd /root/anaconda3/envs/$CONDA_DEFAULT_ENV/compiler_compat && mv ld.bak ld && popd

NCCL 安装
^^^^^^^^^^^^^^^^^^^^^^

NCCL 是 GPU 通信的常用库，VQNet 中 GPU 的分布式计算功能基于 NCCL 实现。
该软件在安装时会默认安装 NCCL 动态链接库，因此一般不需要单独安装 NCCL。
本部分需要 MPI 支持，因此也需要部署 MPI 环境。

分布式启动
=================================
 
使用 ``vqnetrun`` 命令启动的分布式计算接口，``vqnetrun`` 的参数说明如下。

n, np
^^^^^^^^^^^^^^^^^^^^^^

``vqnetrun`` 接口允许您使用 ``-n``、``-np`` 参数控制启动的进程数量，如下例所示。

    示例::

        from pyvqnet.distributed import CommController
        Comm_OP = CommController("mpi") # 初始化 mpi 控制器
        
        rank = Comm_OP.getRank()
        size = Comm_OP.getSize()
        print(f"rank: {rank}, size {size}")

        # vqnetrun -n 2 python test.py
        # vqnetrun -np 2 python test.py

backend
^^^^^^^^^^^^^^^^^^^^^^

``vqnetrun`` 接口允许您使用 ``--backend`` 参数选择分布式后端，支持 ``mpi``（默认）和 ``nccl`` 两种模式。
MPI 模式用于基于 CPU 的分布式计算，NCCL 模式用于基于 GPU 的分布式计算。

    示例::

        from pyvqnet.distributed import CommController
        Comm_OP = CommController("nccl")

        rank = Comm_OP.getRank()
        size = Comm_OP.getSize()
        print(f"rank: {rank}, size {size}")

        # vqnetrun --backend nccl --nproc_per_node 2 python test.py

nproc_per_node
^^^^^^^^^^^^^^^^^^^^^^

``vqnetrun`` 接口允许您使用 ``--nproc_per_node`` 参数控制每个节点上启动的进程数量，仅在 NCCL 模式下可用。

    示例::

        from pyvqnet.distributed import CommController
        Comm_OP = CommController("nccl")

        rank = Comm_OP.getRank()
        size = Comm_OP.getSize()
        print(f"rank: {rank}, size {size}")

        # vqnetrun --backend nccl --nproc_per_node 4 python test.py

output-filename
^^^^^^^^^^^^^^^^^^^^^^

``vqnetrun`` 接口允许您使用命令行参数 ``--output-filename`` 将输出保存到指定文件。

示例实现如下：

    示例::

        from pyvqnet.distributed import CommController, get_host_name
        Comm_OP = CommController("mpi") # 初始化 mpi 控制器
        
        rank = Comm_OP.getRank()
        size = Comm_OP.getSize()
        print(f"rank: {rank}, size {size}")
        print(f"LocalRank {Comm_OP.getLocalRank()} hosts name {get_host_name()}")

        # vqnetrun -np 4 --hostfile hosts --output-filename output  python test.py


verbose
^^^^^^^^^^^^^^^^^^^^^^

``vqnetrun`` 接口可以使用命令行参数 ``--verbose`` 来启用详细的通信日志并输出结果。

示例实现如下：

    示例::

        from pyvqnet.distributed import CommController, get_host_name
        Comm_OP = CommController("mpi") # 初始化 mpi 控制器
        
        rank = Comm_OP.getRank()
        size = Comm_OP.getSize()
        print(f"rank: {rank}, size {size}")
        print(f"LocalRank {Comm_OP.getLocalRank()} hosts name {get_host_name()}")

        # vqnetrun -np 4 --hostfile hosts --verbose python test.py


start-timeout
^^^^^^^^^^^^^^^^^^^^^^

``vqnetrun`` 接口可以使用命令行参数 ``--start-timeout`` 指定在所有检查执行完毕且进程启动前的超时时间。默认值为 30 秒。

示例实现如下：

    示例::

        from pyvqnet.distributed import CommController, get_host_name
        Comm_OP = CommController("mpi") # 初始化 mpi 控制器
        
        rank = Comm_OP.getRank()
        size = Comm_OP.getSize()
        print(f"rank: {rank}, size {size}")
        print(f"LocalRank {Comm_OP.getLocalRank()} hosts name {get_host_name()}")

        # vqnetrun -np 4 --start-timeout 10 python test.py


CommController
=================================

    分布式计算用于控制 CPU 和 GPU 下不同进程之间的数据通信。它会为 CPU（MPI）和 GPU（NCCL）生成不同的控制器，并调用通信方法完成不同进程之间的数据通信与同步。

__init__
^^^^^^^^^^^^^^^^^^^^^^
.. py:class:: pyvqnet.distributed.ControlComm.CommController(backend,rank=None,world_size=None)
    
    CommController 用于控制 CPU 和 GPU 下的数据通信。通过设置参数 `backend`，生成 CPU（MPI）或 GPU（NCCL）的控制器。（目前，分布式计算功能仅支持 Linux。）

    :param backend: 用于生成 cpu 或 gpu 的数据通信控制器。
    :param rank: 该参数仅在非 pyvqnet 后端中有用，默认值为：None。
    :param world_size: 该参数仅在非 pyvqnet 后端中有用，默认值为：None。
        
    :return:
        CommController 实例。

    Examples::

        from pyvqnet.distributed import CommController
        Comm_OP = CommController("nccl") # 初始化 nccl 控制器

        # Comm_OP = CommController("mpi") # 初始化 mpi 控制器

 
    .. py:method:: getRank()
        
        用于获取当前进程的进程号。

        :return: 返回当前进程的进程号。

        Examples::

            from pyvqnet.distributed import CommController
            Comm_OP = CommController("nccl") # 初始化 nccl 控制器
            
            Comm_OP.getRank()

 
    .. py:method:: getSize()
    
        用于获取启动的进程总数。


        :return: 返回进程总数。

        Examples::

            from pyvqnet.distributed import CommController
            Comm_OP = CommController("nccl") # 初始化 nccl 控制器
            
            Comm_OP.getSize()
            # vqnetrun --backend nccl --nproc_per_node 2 python test.py




    .. py:method:: getLocalRank()
    
        用于获取当前机器上的当前进程号。


        :return: 当前机器上的当前进程号。

        Examples::

            from pyvqnet.distributed import CommController
            Comm_OP = CommController("nccl") # 初始化 nccl 控制器
            
            Comm_OP.getLocalRank()
            # vqnetrun --backend nccl --nproc_per_node 2 python test.py


    .. py:method:: split_groups(rankL)
        
        根据输入参数设置的进程号列表来划分多个通信组。

        :param rankL: 进程组排名的列表。

        :return: 当后端为 `nccl` 时，返回一个进程组排名的元组。
                 当后端为 `mpi` 时，返回一个列表，其长度等于组数；每个元素是一个元组 (comm, rank)，其中 comm 是该组的 MPI 通信器，rank 是该组内的序号。

        Examples::
            
            from pyvqnet.distributed import CommController,get_rank,get_local_rank
            from pyvqnet.tensor import tensor
            import numpy as np
            Comm_OP = CommController("mpi")

            groups = Comm_OP.split_groups([[0, 1],[2,3]])
            print(groups)
            #[[<mpi4py.MPI.Intracomm object at 0x7f53691f3230>, [0, 3]], [<mpi4py.MPI.Intracomm object at 0x7f53691f3010>, [2, 1]]]

 
    .. py:method:: barrier()
    
        同步。

        :return: 同步。

        Examples::

            from pyvqnet.distributed import CommController
            Comm_OP = CommController("nccl")
            
            Comm_OP.barrier()

 
    .. py:method:: get_device_num()
    
        用于获取当前节点上的显卡数量（仅在 gpu 上支持）。

        :return: 返回当前节点上的显卡数量。

        Examples::

            from pyvqnet.distributed import CommController
            Comm_OP = CommController("nccl")
            
            Comm_OP.get_device_num()
            # python test.py


 
    .. py:method:: allreduce(tensor, c_op = "avg")
    
        支持数据的 allreduce 通信。

        :param tensor: 输入数据。
        :param c_op: 计算方式。

        Examples::

            from pyvqnet.distributed import CommController
            from pyvqnet.tensor import tensor
            import numpy as np
            Comm_OP = CommController("mpi")

            num = tensor.to_tensor(np.random.rand(1, 5))
            print(f"rank {Comm_OP.getRank()}  {num}")

            Comm_OP.allreduce(num, "sum")
            print(f"rank {Comm_OP.getRank()}  {num}")
            # vqnetrun -n 2 python test.py"

 
    .. py:method:: reduce(tensor, root = 0, c_op = "avg")
    
        支持数据的 reduce 通信。

        :param tensor: 输入。
        :param root: 指定数据返回的目标节点。
        :param c_op: 计算方式。

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
    
        将指定进程 root 上的数据广播到所有进程。

        :param tensor: 输入。
        :param root: 指定节点。

        Examples::

            from pyvqnet.distributed import CommController
            from pyvqnet.tensor import tensor
            import numpy as np
            Comm_OP = CommController("mpi")

            num = tensor.to_tensor(np.random.rand(1, 5))
            print(f"rank {Comm_OP.getRank()}  {num}")
            
            Comm_OP.broadcast(num, 1)
            print(f"rank {Comm_OP.getRank()}  {num}")
            # vqnetrun -n 2 python test.py"


    .. py:method:: allgather(tensor)
    
        将所有进程上的数据收集到一起。

        :param tensor: 输入。

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
    
        p2p 通信接口。

        :param tensor: 输入。
        :param dest: 目标进程。

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
    
        p2p 通信接口。

        :param tensor: 输入。
        :param source: 接收进程。

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
    
        组内 allreduce 通信接口。

        :param tensor: 输入。
        :param c_op: 计算方式。
        :param group: 通信组。使用 mpi 后端时，传入 `init_groups` 或 `split_groups` 生成的与通信组对应的组。使用 nccl 后端时，传入 `split_groups` 生成的组号。


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
    
        组内 reduce 通信接口。

        :param tensor: 输入。
        :param root: 指定进程号。
        :param c_op: 计算方式。
        :param group: 通信组。使用 mpi 后端时，传入 `init_groups` 或 `split_groups` 生成的与通信组对应的组。使用 nccl 后端时，传入 `split_groups` 生成的组号。

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
    
        组内 broadcast 通信接口。

        :param tensor: 输入。
        :param root: 指定广播来源的进程号，默认值：0。
        :param group: 通信组。使用 mpi 后端时，传入 `init_groups` 或 `split_groups` 生成的与通信组对应的组。使用 nccl 后端时，传入 `split_groups` 生成的组号。

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
    
        组内 allgather 通信接口。

        :param tensor: 输入。
        :param group: 通信组。使用 mpi 后端时，传入 `init_groups` 或 `split_groups` 生成的与通信组对应的组。使用 nccl 后端时，传入 `split_groups` 生成的组号。

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
    
        使用 allreduce 更新优化器中参数的梯度。

        :param optimizer: 优化器。

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
    
        以 allreduce 方式更新模型中的参数。

        :param model: 模型。

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
    
        广播指定进程号上的模型参数。

        :param model: 模型。
        :param root: 指定进程号。

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

        使用 NCCL 对 GPU 数据进行异步或同步的 all_gather 操作。

        :param output: QTensor - all_gather 结果的目标 QTensor。
        :param input: QTensor - 待收集的 QTensor。
        :param group: 通信进程组，group 是包含组索引的元组。默认值：None，不使用组。
        :param async_op: 此操作是否为异步，默认值：False。
        :return: Work - 异步通信句柄。使用 wait() 等待此操作完成。

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

        使用 NCCL 对 GPU 数据进行异步或同步的 allreduce 操作。

        :param tensor: QTensor - 需要归约的 QTensor。
        :param c_op: 计算方式，可选 "sum" 或 "avg"，默认值为 "avg"。
        :param group: 通信进程组，group 是包含组索引的元组。默认值：None，不使用组。
        :param async_op: 此操作是否为异步，默认值：False。
        :return: Work - 异步通信句柄。使用 wait() 等待此操作完成。

        Examples::

            import pyvqnet
            from pyvqnet.tensor import tensor
            from pyvqnet.distributed.ControlComm import CommController,get_local_rank

            Comm_OP = CommController("nccl")
            complex_data = tensor.ones([500,500],dtype=pyvqnet.kcomplex64).toGPU(pyvqnet.DEV_GPU_0 + get_local_rank())
            work = Comm_OP.nccl_async_all_reduce(complex_data, "sum",group = None,async_op = True)
            work.wait()

    .. py:method:: nccl_async_reduce( tensor_, dest, c_op="avg", group=None, async_op=False ):

        使用 NCCL 对 GPU 数据进行异步或同步的 reduce 操作。

        :param tensor_: QTensor - 需要归约的 QTensor。
        :param dest: 归约后 QTensor 的目标进程号。
        :param c_op: 计算方式，可选 "sum" 或 "avg"，默认值为 "avg"。
        :param group: 通信进程组，group 是包含组索引的元组。默认值：None，不使用组。
        :param async_op: 此操作是否为异步，默认值：False。
        :return: Work - 异步通信句柄。使用 wait() 等待此操作完成。

        Examples::

            from pyvqnet import tensor
            import pyvqnet
            from pyvqnet.distributed import CommController,get_rank,get_local_rank
            Comm_OP = CommController("nccl")

            complex_data = tensor.ones([500,500],dtype=pyvqnet.kcomplex64).toGPU(pyvqnet.DEV_GPU_0 + get_local_rank())
            work = Comm_OP.nccl_async_reduce(complex_data, 0,"sum",group = None,async_op = True)
            work.wait()

    .. py:method:: nccl_async_broadcast(tensor, src, group=None, async_op=False )

        使用 NCCL 对 GPU 数据进行异步或同步的 broadcast 操作。

        :param tensor: QTensor - 需要广播的 QTensor。
        :param src: 广播 QTensor 的源进程号。
        :param group: 通信进程组，group 是包含组索引的元组。默认值：None，不使用组。
        :param async_op: 此操作是否为异步，默认值：False。
        :return: Work - 异步通信句柄。使用 wait() 等待此操作完成。

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

        使用 NCCL 对 GPU 数据进行异步或同步的 P2P 发送操作。

        :param t: QTensor - 需要发送的 QTensor。
        :param dest: 发送 QTensor 的目标进程号。
        :param async_op: 此操作是否为异步，默认值：False。
        :return: Work - 异步通信句柄。使用 wait() 等待此操作完成。

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

        使用 NCCL 对 GPU 数据进行异步或同步的 P2P 接收操作。

        :param t: QTensor - 接收数据的 QTensor。
        :param src: 接收 QTensor 的源进程号。
        :param async_op: 此操作是否为异步，默认值：False。
        :return: Work - 异步通信句柄。使用 wait() 等待此操作完成。

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

在多进程中，使用 ``split_data`` 根据进程数量对数据进行切片，并返回对应进程上的数据。

.. py:function:: pyvqnet.distributed.datasplit.split_data(x_train, y_train, shuffle=False)

设置分布式计算的参数。

    :param x_train: `np.array` - 训练数据。
    :param y_train: `np.array` - 训练数据标签。
    :param shuffle: `bool` - 是否先打乱再切片，默认值为 False。

    :return: 切片后的训练数据和标签。

    Example::

        from pyvqnet.distributed import split_data
        import numpy as np

        x_train = np.random.randint(255, size = (100, 5))
        y_train = np.random.randint(2, size = (100, 1))

        x_train, y_train= split_data(x_train, y_train)

get_local_rank
~~~~~~~~~~~~~~~~~~~~~~~~~~~

使用 ``get_local_rank`` 获取当前机器上的进程号。

.. py:function:: pyvqnet.distributed.ControlComm.get_local_rank()

    用于获取当前机器上的当前进程号。

    :return: 当前机器上的当前进程号。

    Example::

        from pyvqnet.distributed.ControlComm import get_local_rank

        print(get_local_rank())
        # vqnetrun -n 2 python test.py

get_rank
~~~~~~~~~~~~~~~~~~~~~~~~~~~
使用 ``get_rank`` 获取当前机器上的进程号。

.. py:function:: pyvqnet.distributed.ControlComm.get_rank()

    用于获取当前进程的进程号。

    :return: 当前进程的进程号。

    Example::

        from pyvqnet.distributed.ControlComm import get_rank

        print(get_rank())
        # vqnetrun -n 2 python test.py

init_groups
~~~~~~~~~~~~~~~~~~~~~~~~~~~
使用 ``init_groups`` 根据给定的进程号列表初始化基于 CPU 的进程组。

.. py:function:: pyvqnet.distributed.ControlComm.init_groups(rank_lists)

    用于初始化 `mpi` 后端的进程通信组。

    :param rank_lists: 通信进程组列表。
    :return: 初始化后的进程组列表。

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
    
    流水线并行训练包装器实现了 1F1B 训练。仅适用于 Linux 平台且需配备 GPU。
    更多算法详情请参见 (https://www.deepspeed.ai/tutorials/pipeline/)。

    :param args: 参数字典。参见示例。
    :param join_layers: Sequential 模块列表。
    :param trainset: 数据集。

    :return:
        PipelineParallelTrainingWrapper 实例。

    以下使用 CIFAR10 数据集 `CIFAR10_Dataset` 在 2 个 GPU 上对 AlexNet 进行分类任务训练。
    在此示例中，流水线并行进程数 `pipeline_parallel_size` = 2。
    批大小为 `train_batch_size` = 64，单个 GPU 上为 `train_micro_batch_size_per_gpu` = 32。
    其他配置参数可在 `args` 中找到。
    此外，每个进程需要在 `__main__` 函数中配置环境变量 `LOCAL_RANK`。

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
    
    Zero1 API 接口，目前仅适用于基于 GPU 并行计算的 Linux 平台。

    :param args: 参数字典。
    :param model: 模型（Module）。
    :param optimizer: 优化器。

    :return:
        Zero1 引擎。

以下使用 MNIST 数据集在 2 个 GPU 上对 MLP 模型进行分类任务训练。

    批大小为 `train_batch_size` = 64，`zero_optimization` 的阶段 `stage` 设置为 1。
    如果 Optimizer 为 None，则使用 `args` 中 `optimizer` 的设置。其他配置参数可在 `args` 中找到。
    
    此外，每个进程需要配置环境变量 `LOCAL_RANK`。
    
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
            加载 mnist 数据
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
    
    张量并行计算，列并行线性层。
    
    线性层定义为 Y = XA + b。其 2D 并行行形式为 A = [A_1, ... , A_p]。

    :param input_size: 矩阵 A 的第一维。
    :param output_size: 矩阵 A 的第二维。
    :param weight_initializer: `callable` - 默认为 normal。
    :param bias_initializer: `callable` - 默认为 zeros。
    :param use_bias: `bool` - 默认为 True。
    :param dtype: 默认：None，使用默认数据类型。
    :param name: 模块名称，默认：""。
    :param tp_comm: 通信控制器（Comm Controller）。

    以下使用 MNIST 数据集在 2 个 GPU 上对 MLP 模型进行分类任务训练。

    使用方法与经典 Linear 层类似。

    基于 `# vqnetrun --backend nccl --nproc_per_node 2 python test.py` 的多进程使用方式。

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
            加载 mnist 数据
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
    
    张量并行计算，行并行线性层。

    线性层定义为 Y = XA + b。A 沿其第一维并行化，X 沿其第二维并行化。
    A = transpose([A_1 .. A_p]) X = [X_1, ..., X_p]。

    :param input_size: 矩阵 A 的第一维。
    :param output_size: 矩阵 A 的第二维。
    :param weight_initializer: `callable` - 默认为 normal。
    :param bias_initializer: `callable` - 默认为 zeros。
    :param use_bias: `bool` - 默认为 True。
    :param dtype: 默认：None，使用默认数据类型。
    :param name: 模块名称，默认：""。
    :param tp_comm: 通信控制器（Comm Controller）。

    以下使用 MNIST 数据集在 2 个 GPU 上对 MLP 模型进行分类任务训练。

    使用方法与经典 Linear 层类似。

    基于 `# vqnetrun --backend nccl --nproc_per_node 2 python test.py` 的多进程使用方式。

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


Bit Reordering
=================================

量子位重排序是比特并行中的一项技术。其核心目标是通过改变量子逻辑门的顺序来减少比特并行所需的比特变换次数。构建基于比特并行的大位量子电路需要以下模块。参考论文 `Lazy Qubit Reordering for Accelerating Parallel State-Vector-based Quantum Circuit Simulation <https://export.arxiv.org/abs/2410.04252>`__。

以下接口需要 `mpi` 启动多个进程进行计算。

DistributeQMachine
^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. py:class:: pyvqnet.distributed.qubits_reorder.DistributeQMachine(num_wires, dtype, grad_mode)
    
    用于模拟比特并行变分量子计算的类，每个节点上包含量子比特子集上的量子态。每个节点通过 MPI 申请分布式量子变分电路模拟。N 的值必须等于 2 的（分布式并行比特数）次幂，即 `global_qubit`，可以通过 `set_qr_config` 配置。

    :param num_wires: 完整量子电路中的比特数。
    :param dtype: 计算数据的数据类型。默认为 pyvqnet.kcomplex64，对应 pyvqnet.kfloat32 的参数精度。
    :param grad_mode: 当反向传播 ``DistQuantumLayerAdjoint`` 时设置为 adjoint。

    .. note::

        输入的比特数是整个量子电路所需的比特数。DistributeQMachine 将基于全局比特数构建量子模拟器，即 ``num_wires - global_qubit``。

        反向传播必须基于 ``DistQuantumLayerAdjoint``。

    .. warning::

        该接口仅支持在 Linux 下运行；

        ``DistributeQMachine`` 中的比特并行参数必须配置，如示例所示，包括：

        .. code-block::

            qm.set_just_defined(True)
            qm.set_save_op_history_flag(True) # 开启保存操作
            qm.set_qr_config({'qubit': 总量子比特数, 'global_qubit': 分布式量子比特数})
    
    
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
                self.qm.set_save_op_history_flag(True) # 开启保存操作
                self.qm.set_qr_config({"qubit": num_wires, # 开启比特重排序，设置配置
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
    
    一个 DistQuantumLayer 层，使用伴随矩阵方法计算比特并行计算中参数的梯度。

    :param vqc_module: 输入的隐式 ``DistributeQMachine`` 模块。
    :param name: 模块名称。

    .. note::

        输入的 vqc_module 模块必须包含 ``DistributeQMachine``。在比特并行计算中，基于 ``DistributeQMachine`` 执行伴随反向传播梯度计算。

    .. warning::

        该接口仅在 Linux 上受支持；


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
                self.qm.set_save_op_history_flag(True) # 开启保存操作
                self.qm.set_qr_config({"qubit": num_wires, # 开启比特重排序，设置配置
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
