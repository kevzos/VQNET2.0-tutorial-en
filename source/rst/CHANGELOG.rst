VQNet Changelog
###############################


[v2.18.0] - 2026-04-22
***************************

Added
===================
- ``vqnetrun`` 现在支持 ``--backend nccl`` 模式，可通过 ``--nproc_per_node``\ 、``--nccl_socket_ifname`` 参数控制分布式启动。
- 新增 ``VQCQCloudLayer`` 接口，用于将 VQC Module 提交到 QCloud 真芯片或 pyqpanda3 本地模拟器，支持 parameter_shift 反向传播。
- ``CommController`` 新增 ``destroy()`` 方法，用于 NCCL 通信资源清理。

Changed
===================
- 默认后端改为 ``pyvqnet-ad``\ 。
- 移除了已弃用的 ``QuantumLayerMultiProcess``\ 、``DataParallelHybirdVQCQpandaQVMLayer`` 和 ``HybirdVQCQpanda3QVMLayer`` 接口。
- ``split_group`` 重命名为 ``split_groups``\ 。
- 依赖于 CUDA 12.6 的 NVIDIA 运行时。
- "chip_id" 默认值改为 "WK_C180"。
- ``ComplexEntangelingTemplate`` 已重命名为 ``ComplexEntanglingTemplate``\ 。
- ``vqc.rst``\ ：新增 "Test 2: 10-Qubit VQC Gradient Comparison" 基准测试章节，对比 VQNet / TorchQuantum / DeepQuantum / Pennylane / MindQuantum。
- 更新了基准测试规格表，使用 CUDA 12.6、torchquantum 0.2.0 和 mindquantum 0.12.0。

Fixed
===================
- 修复了 ``roll`` CUDA 内核中的整数溢出问题。
- 修复了 ``cuda_masked_fill`` 对 complex64/complex128 类型的支持。
- 修复了 ``log_softmax`` 在 ``bfloat16`` 下前向计算产生错误 +inf 值的问题。
- 修复了跨 GPU 内存访问时缺少 ``CUDAGuard`` 导致的设备错误。
- 修复了多处拼写错误。


[v2.17.3] - 2026-03-31
***************************

Added
===================
- 新增 bfloat16 数据类型。
- 新增异步 NCCL 通信接口：``nccl_async_all_gather``\ 、``nccl_async_all_reduce``\ 、``nccl_async_reduce``\ 、``nccl_async_broadcast``\ 、``nccl_async_send`` 和 ``nccl_async_recv``\ 。
- 新增对芯片 ID 为 ``WK_C180`` 的最新本源量子芯片的支持。
- 新增 ``data_ptr`` 等接口，实验性支持 `triton <https://triton-lang.org/main/index.html>`_。

Changed
===================

- 默认后端改为 "pyvqnet-ad"。
- MacOS 上的计算基于 arm neon 指令实现。
- ``matmul`` 接口支持超过 4 维的数据。
- 减少了安装 whl 包时对一些 cuda 运行时库的依赖。
- pyvqnet 中 QTensor 的数据类型不再是整数，而是具体的数据类型。
- 修改了 QTensor 的 pickle 逻辑，不再 pickle grad。
- 移除了 is_dense。
- 移除了 pq2 的 ``QuantumBatchAsyncQcloudLayer``\ 。
- 更新了 pq3 ``QuantumBatchAsyncQcloudLayer`` 的文档。

Fixed
===================
- 修复了 ``Linear`` 在 ``use_bias=False`` 时的错误。
- 修复了 ``MAX_GPUS`` 问题；当前最大 GPU 数量为 16。
- 修复了 Windows jupyter 上的导入错误。


[v2.17.2] -2025-11-18
***********************************


Added
===================

- 新增对基于量子自然梯度 QNG 接口的 `torch` 后端的支持。
- 新增 `pyvqnet-ad` 后端，使用类似 torch 的 C++ 自动微分后端。数据结构仍然使用原有的 ``_core.Tensor``\ ，支持绝大多数现有接口。
- 新增 "Benchmarking of Variational Quantum Circuit's Gradients for Batch Data" 文档。

Changed
===================

- 删除 `HybirdVQCQpanda3QVMLayer`\ 、`QuantumLayerMultiProcess`\ 、`TorchHybirdVQCQpanda3QVMLayer`\ ；
- 删除 `is_csr`\ 、`csr_members`\ 、`SparseHamiltonian`\ 、`csr_to_dense`\ 、`dense_to_csr`\ ；
- 新增 `QiskitLayer` 和 `CirqLayer` 接口。
- 为 `QuantumBatchAsyncQcloudLayer` 层新增 `if_print_qcloud_log` 支持，用于打印 qcloud 日志。
- 安装命令改为 `pip install pyvqnet --upgrade`\ 。
- 支持的 Python 版本为 `Python 3.10`\ 。
- 修改了指定的 mpicxx 安装命令。


Fixed
===================

- 支持最新版本 pyqpanda2QCloud 的返回值；
- 为点积接口添加输入数据设备检查；
- 修复了 `TorchModule` 中的 bug；



[v2.17.1] - 2025-8-22
***************************

Added
===================

- 新增量子自然梯度 SPSA 算法（qnspsa）接口、量子电路生机器（QBM）、带动量的量子自然梯度接口以及纯量子 QGRU 示例。
- 新增 ``torch_native`` 后端。
- 新增比特并行接口以支持比特并行量子电路，并新增比特重排序功能以减少比特交换次数。
- 新增 ``split_groups`` 方法。

Changed
==================
- Linear 层实现从 `:math:`y = Ax + b`` 改为 `:math:`y = x@A.T + b``
- 修改了 `MeasureAll` 接口中的 ``obs`` 参数。
- 移除了 ``QuantumLayerES`` 接口。将 `allgather_group`\ 、`allreduce_group`\ 、`reduce_group`\ 、`broadcast_group` 等接口中的参数名从 ControllComm 改为 ControlComm。
- 移除了 `ncclsplitGroup` 接口。

Fixed
===================
- 解决了分布式通信接口的同步延迟问题。
- 修改了分布式通信接口定义。
- 解决了伴随梯度计算不支持 ``PauliX``\ 、``PauliY`` 和 ``PauliZ`` 的问题。


[v2.17.0] - 2025-4-22
***************************

Added
===================

- 新增量子电路模块的张量网络后端实现，包括对基本逻辑门、测量和复杂量子电路的支持。
- 新增用于构建大量子比特量子电路的张量网络后端实现。
- 新增 QTensor.swapaxes 接口，别名为 swapaxis。

Changed
===================
- 矩阵运算使用 openblas。
- 使用 sleef 进行 CPU SIMD 运算。
- 移除 qnn.MeasurePauliSum。
- 当 torch 版本低于 2.4 时，使用 torch 后端计算会发出警告。

Fixed
====================
- 解决了保存模型时 QMachine 状态的问题。
- 解决了 layernorm 和 groupnorm 在 ``affine=False`` 时的问题。
- 解决了 ``QuantumLayerAdjoint`` 在 eval 模式下的问题。

[v2.16.0] - 2025-1-15
***************************

Added
===================

- 新增使用 pyqpanda3 进行量子电路计算的接口。
- MeasureAll 接口支持复合 Pauli 算符。
- 新增 DataParallelVQCAdjointLayer 和 DataParallelVQCLayer 接口。

Changed
===================

- 移除了过时的 ONNX 函数和大多数集成了 pyqpanda 的接口，同时保留了示例代码中使用的一些接口。
- 修改了 ``VQC_QuantumEmbedding`` 接口。
- 安装此包时不再安装 pyqpanda，而是安装 pyqpanda3。
- VQC 接口支持使用 `x[:,:2]` 作为输入参数，原先仅支持 `x[:,[2]]` 格式。
- 本软件支持 Python 3.10。

Fixed
====================
- 解决了内存泄漏问题。
- 解决了 GPU 随机数问题。
- 对于 reduce 相关操作，最大支持的数组维度已从 8 增加到 30。
- 优化了代码，在某些情况下提升了 Python 代码运行速度。


[v2.15.0] - 2024-11-19
***************************

Added
===================

- 新增 `pyvqnet.backends.set_backend()` 接口。当用户安装 `torch` 后，可使用 `torch` 进行 QTensor 的矩阵计算和变分量子电路计算。详情请参见文档 :ref:`torch_api`\ 。
- 新增 `pyvqnet.nn.torch`\ ，继承 `torch.nn.Module` 的神经网络接口和变分量子电路神经接口。详情请参见文档 :ref:`torch_api`\ 。

Changed
===================
- 修改了 diag 接口。
- 修改了 all_gather 实现，与 torch.distributed.all_gather 保持一致。
- 修改 `QTensor` 以支持最多 30 维数据。
- 分布式功能所需的 `mpi4py` 要求版本 4.0.1 及以上。

Fixed
===================
- 由于 OpenMP，部分随机数实现无法固定种子。
- 修复了分布式启动中的一些 bug。


[v2.14.0] - 2024-09-30
***************************

Added
===================

- 新增块编码算法：``VQC_LCU``\ 、``VQC_FABLE``\ 、``VQC_QSVT``\ ，以及 qpanda 算法实现 ``QPANDA_QSVT``\ 、``QPANDA_LCU``\ 、``QPANDA_FABLE``\ 。
- 新增量子比特上的整数加法 ``vqc_qft_add_to_register``\ 、两个量子比特上的数字加法 ``vqc_qft_add_two_register``\ ，以及两个量子比特上的数字乘法 ``vqc_qft_mul``\ 。
- 新增混合 qpanda 和 vqc 训练模块 ``HybirdVQCQpandaQVMLayer``\ 。
- 新增 ``einsum``\ 、``moveaxis``\ 、``eigh``\ 、``diagonal`` 等接口实现。
- 分布式计算中新增张量并行计算函数：``ColumnParallelLinear``\ 、``RowParallelLinear``\ 。
- 分布式计算中新增 Zero stage-1 功能：``ZeroModelInitial``\ 。
- ``QuantumBatchAsyncQcloudLayer``\ ：当 ``diff_method == "random_coordinate_descent"`` 时，使用随机参数选择进行梯度计算，替代 PSR。

Changed
====================
- 删除了 xtensor 部分。
- API 文档进行了部分重构。区分了基于自动微分和基于 qpanda 的量子机器学习示例，以及基于自动微分和基于 qpanda 的量子机器学习接口。
- `matmul` 支持 1d@1d、2d@1d、1d@2d。
- 新增一些量子计算层别名：``QpandaQCircuitVQCLayer`` = ``QuantumLayer``\ 、``QpandaQCircuitVQCLayerLite`` = ``QuantumLayerV2``\ 、``QpandaQProgVQCLayer`` = ``QuantumLayerV3``\ 。

Fixed
====================
- 修改了分布式计算功能中的底层通信接口 ``allreduce``\ 、``allgather``\ 、``reduce``\ 、``broadcast``\ ，并增加了对 ``core.Tensor`` 数据通信的支持。
- 解决了随机数生成中的 bug。
- 解决了将 VQC 的 ``RXX``\ 、``RYY``\ 、``RZZ``\ 、``RZX`` 转换为 originIR 时的错误。


[v2.13.0] - 2024-07-30
***************************

Added
==================

- 新增 `no_grad`\ 、`GroupNorm`\ 、`Interpolate`\ 、`contiguous`\ 、`QuantumLayerV3`\ 、`fuse_model`\ 、`SDPA` 接口。
- 新增量子 Dropout 方法以防止过拟合。

Changed
===================

- 为 `BatchNorm`\ 、`LayerNorm`\ 、`GroupNorm` 添加了 affine 接口。
- `diag` 接口现在对 2d 输入返回对角线的 1d 输出，与 torch 一致。
- slice 和 permute 等操作将尝试使用 view 方法返回共享内存中的 QTensor。
- 所有接口支持非连续输入。
- `Adam` 支持 weight_decay 参数。

Fixed
====================
- 修改了 VQC 部分逻辑门分解函数的错误。
- 修复了部分函数的内存泄漏问题。
- 修复了 `QuantumLayerMultiProcess` 不支持 GPU 输入的问题。
- 修改了 `Linear` 的默认参数初始化方法。


[v2.12.0] - 2024-05-01
***************************

Added
===================

- 新增 PipelineParallelTrainingWrapper 接口。
- 新增 `Gelu`\ 、`DropPath`\ 、`binomial`\ 、`adamW` 接口。
- 新增 `QuantumBatchAsyncQcloudLayer`\ ，支持 pyqpanda 的本地虚拟机模拟计算。
- 添加 xtensor 的 `QuantumBatchAsyncQcloudLayer`\ ，支持 pyqpanda 的本地虚拟机模拟计算和真机计算。
- 使 QTensor 支持 deepcopy 和 pickle。
- 新增分布式计算启动命令 `vqnetrun`\ ，在使用分布式计算接口时使用。
- 新增 ES 梯度计算方法真机接口 `QuantumBatchAsyncQcloudLayerES`\ ，支持 pyqpanda 的本地虚拟机模拟计算和真机计算。
- 新增分布式计算中支持 QTensor 的数据通信接口 `allreduce`\ 、`reduce`\ 、`broadcast`\ 、`allgather`\ 、`send`\ 、`recv` 等。

Changed
===================

- 安装包新增依赖 "Pillow" 和 "hjson"，Linux 系统新增依赖 "psutil" 和 "cloudpickle"。
- 优化了 GPU 下的 softmax 和 transpose 运行速度。
- 使用 cuda11.8 编译。
- 基于 cpu 和 gpu 的分布式计算接口集成。

Fixed
===================
- 减少了启动 Linux-GPU 版本时的内存消耗。
- 修复了 select 和 power 函数的内存泄漏问题。
- 移除了基于 reduce 方法的 cpu、gpu 模型参数和梯度更新方法 `nccl_average_parameters_reduce`\ 、`nccl_average_grad_reduce`\ 。

[v2.11.0] - 2024-03-01
***************************

Added
===================

- 新增 `QNG`\ （量子自然梯度）API 和示例。
- 新增量子电路优化，如 `wrapper_single_qubit_op_fuse`\ 、`wrapper_commute_controlled`\ 、`wrapper_merge_rotations` API 和示例。
- 新增 `CY`\ 、`SparseHamiltonian`\ 、`HermitianExpval`\ 。
- 新增 `is_csr`\ 、`is_dense`\ 、`dense_to_csr`\ 、`csr_to_dense`\ 。
- 新增 `QuantumBatchAsyncQcloudLayer`\ ，支持 pyqpanda 的 QCloud 真芯片计算，`expval_qcloud`\ 。
- 新增基于 NCCL 的接口实现，用于单节点多 GPU 分布式计算数据的并行模型训练：`nccl_average_parameters_allreduce`\ 、`nccl_average_parameters_reduce`\ 、`nccl_average_grad_allreduce`\ 、`nccl_average_grad_reduce`\ ，以及控制 NCCL 初始化和相关操作的类 `NCCL_api`\ 。
- 新增量子线路演化策略梯度计算接口 `QuantumLayerES`\ 。

Changed
===================

- 将 `VQC_CSWAP` 电路重构为 `CSWAP`\ 。
- 删除旧版 QNG 文档。
- 移除了 `pyvqnet.qnn.vqc` 中函数和类的无用 `num_wires` 参数。
- 重构 `MeasureAll`\ 、`Probability` API。
- 为 `QuantumMeasure` 添加 qtype 参数。

Fixed
===================
- 将 `QuantumMeasure` 的 slots 改为 shots。

[v2.10.0] - 2023-12-30
***************************

Added
===========
- 在 pyvqnet.qnn.vqc 下新增接口：IsingXX、IsingXY、IsingYY、IsingZZ、SDG、TDG、PhaseShift、MultiRZ、MultiCnot、MultixCnot、ControlledPhaseShift、SingleExcitation、DoubleExcitation、VQC_AllSinglesDoubles、ExpressiveEntanglingAnsatz 等；
- 新增支持伴随梯度计算的 pyvqnet.qnn.vqc.QuantumLayerAdjoint 接口；
- 支持 originIR 与 VQC 之间的相互转换功能；
- 支持统计 VQC 模型中的经典和量子模块信息；
- 在量子经典神经网络混合模型下新增两个案例：基于小样本的量子卷积神经网络模型和用于手写数字识别的量子核函数模型。


[v2.9.0] - 2023-09-08
***************************

Added
===================
- 新增 xtensor 接口定义，支持自动算子并行和多种 CPU/GPU 后端。包括超过 150 个常用的数学、逻辑和矩阵计算接口，用于多维数组，以及常见的经典神经网络层和优化器。

Changed
===================
- 版本号从 v2.0.8 提升至 v2.9.0。
- 包上传到公司的 PyPI 仓库，使用 ``pip install pyvqnet --index-url <pypi_url>``\ 。


[v2.0.8] - 2023-07-26
***************************

Added
===================
- 新增现有接口对 complex128、complex64、double、float、uint8、int8、bool、int16、int32、int64 等类型计算的支持（gpu）。
- 基于 vqc 的基本逻辑门：Hadamard、CNOT、I、RX、RY、PauliZ、PauliX、PauliY、S、RZ、RXX、RYY、RZZ、RZX、X1、Y1、Z1、U1、U2、U3、T、SWAP、P、TOFFOLI、CZ、CR、ISWAP。
- 基于 vqc 的组合量子电路：VQC_HardwareEfficientAnsatz、VQC_BasicEntanglerTemplate、VQC_StronglyEntanglingTemplate、VQC_QuantumEmbedding、VQC_RotCircuit、VQC_CRotCircuit、VQC_CSWAPcircuit、VQC_Controlled_Hadamard、VQC_CCZ、VQC_FermionicSingleExcitation、VQC_FermionicDoubleExcitation、VQC_UCCSD、VQC_QuantumPoolingCircuit、VQC_BasisEmbedding、VQC_AngleEmbedding、VQC_AmplitudeEmbedding、VQC_IQPEmbedding。
- 基于 vqc 的测量方法：VQC_Purity、VQC_VarMeasure、VQC_DensityMatrixFromQstate、Probability、MeasureAll。


[v2.0.7] - 2023-07-03
***************************

Added
===================
- 经典神经网络新增 kron、gather、scatter、broadcast_to 接口。
- 新增对不同数据精度的支持：dtype 支持 kbool、kuint8、kint8、kint16、kint32、kint64、kfloat32、kfloat64、kcomplex64、kcomplex128，分别对应 bool、uint8_t、int8_t、int16_t、int32_t、int64_t、float、double、complex<float>、complex<double>。
- 支持 python 3.8、3.9、3.10。

Changed
===================
- QTenor 和 Module 类的 init 函数新增 `dtype` 参数。限制了部分神经网络层的 QTenor 索引和输入的类型。
- 量子神经网络：由于 MacOS 兼容性问题，移除了 Mnist_Dataset 和 CIFAR10_Dataset 接口。

[v2.0.6] - 2023-02-22
***************************


Added
===================

- 经典神经网络新增接口：multinomial、pixel_shuffle、pixel_unshuffle、为 QTensor 新增 numel、新增 CPU 动态内存池功能、为 Parameter 新增 init_from_tensor 接口。
- 经典神经网络新增接口：Dynamic_LSTM、Dynamic_RNN、Dynamic_GRU。
- 经典神经网络新增接口：pad_sequence、pad_packed_sequence、pack_pad_sequence。
- 量子神经网络新增接口：CCZ、Controlled_Hadamard、FermionicSingleExcitation、UCCSD、QuantumPoolingCircuit。
- 量子神经网络新增接口：Quantum_Embedding、Mnist_Dataset、CIFAR10_Dataset、grad、Purity。
- 量子神经网络新增示例：基于梯度裁剪、quanvolution、量子电路表达能力、贫瘠高原（barren plateau）和量子强化学习 QDRL。

Changed
===================

- API 文档重构内容结构，新增 "quantum machine learning research" 模块，将 "VQNet2ONNX module" 改为 "Other Utility Functions"。


fixed
===================

- 经典神经网络：解决了相同随机种子在不同平台产生不同正态分布的问题。
- 量子神经网络：使 expval、ProbMeasure、QuantumMeasure 支持 QPanda GPU 虚拟机。


[v2.0.5] - 2022-12-25
***************************


Added
===================

- 经典神经网络新增 log_softmax 实现，新增模型到 ONNX 的 export_model 函数接口。
- 经典神经网络支持将大部分现有经典神经网络模块转换为 ONNX。详情参见 API 文档 "VQNet2ONNX module"。
- 量子神经网络新增 VarMeasure、MeasurePauliSum、Quantum_Embedding、SPSA 等接口。
- 量子神经网络新增 LinearGNN、ConvGNN、QMLP、量子自然梯度、量子随机参数漂移算法、DoublySGD 算法等。


Changed
===================

- 经典神经网络：为 BN1d、BN2d 接口增加了维度检查。

fixed
==================

- 解决了 maxpooling 参数检查的 bug。
- 解决了 [::-1] 切片的 bug。


[v2.0.4] - 2022-09-20
***************************


Added
==================

- 经典神经网络新增 LayernormNd 实现，支持多维数据 layernorm 计算。
- 经典神经网络新增 CrossEntropyLoss 和 NLL_Loss 损失函数计算接口，支持 1 维到 N 维输入。
- 量子神经网络新增常用电路模板：HardwareEfficientAnsatz、StronglyEntanglingTemplate、BasicEntanglerTemplate。
- 量子神经网络新增计算量子比特子系统互信息的 Mutal_info 接口、冯·诺依曼熵 VB_Entropy 和密度矩阵 DensityMatrixFromQstate。
- 量子神经网络新增量子感知器算法示例 QuantumNeuron，新增量子傅立叶级数算法示例。
- 量子神经网络新增支持多进程加速量子电路运行的接口 QuantumLayerMultiProcess。

Changed
==================

- 经典神经网络支持将分组卷积参数 group、扩张卷积的 dilation_rate 以及任意值 padding 作为一维卷积 Conv1d、二维卷积 Conv2d 和反卷积 ConvT2d 的参数。
- 对同维度的数据跳过广播操作，减少不必要的运行逻辑。

fixed
==================

- 解决了 stack 函数在某些参数下计算错误的问题。


[v2.0.3] - 2022-07-15
***************************


Added
==================

- 新增对 stack、双向循环神经网络接口 RNN、LSTM、GRU 的支持。
- 新增常用计算性能指标接口：MSE、RMSE、MAE、R_Square、precision_recall_f1_2_score、precision_recall_f1_Multi_score、precision_recall_f1_N_score、auc_calculate。
- 新增量子核 SVM 算法示例。

Changed
==================

- 加快 QTensor 数据过多时的打印速度。
- 在 Windows 和 Linux 下使用 openmp 加速计算。

fixed
==================

- 解决了部分 Python import 方法导入失败的问题。
- 解决了批归一化（BN）层重复计算的问题。
- 修复了 ``QTensor.reshape`` 和 ``transpose`` 接口无法计算梯度的 bug。
- 为 ``tensor.power`` 接口添加了输入参数形状验证。

[v2.0.2] - 2022-05-15
***************************


Added
==================

- 新增 topK、argtopK。
- 新增 cumsum。
- 新增 masked_fill。
- 新增 triu、tril。
- 新增 QGAN 生成随机分布的示例。

Changed
==================

- 支持高级切片索引和普通切片索引。
- matmul 支持 3D、4D 张量运算。
- 修改 HardSigmoid 函数实现。

fixed
==================

- 解决了卷积、批归一化、反卷积、池化等层在多次反向传播时因未缓存内部变量导致梯度计算错误的问题。
- 修复了 QLinear 层的实现和示例。
- 解决了 macOS 上 conda 环境中导入 VQNet 时 Image 加载失败的问题。




[v2.0.1] - 2022-03-30
**************************


Added
==================

- 新增超过 100 个基本 QTensor 数据结构接口，包括创建函数、逻辑函数、数学函数和矩阵运算。
- 新增 14 个基本神经网络函数，包括卷积、反卷积、池化等。
- 新增 4 个损失函数，包括 MSE、BCE、CCE、SCE 等。
- 新增 10 个激活函数，包括 ReLu、Sigmoid、ELU 等。
- 新增 6 个优化器，包括 SGD、RMSPROP、ADAM 等。
- 新增机器学习示例：QVC、QDRL、Q-KMEANS、QUnet、HQCNN、VSQL、Quantum Autoencoder。
- 新增量子机器学习层：QuantumLayer、NoiseQuantumLayer。
