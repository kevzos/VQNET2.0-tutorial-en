Registro de alterações do VQNet
###############################


[v2.18.0] - 2026-04-22
***************************

Adicionado
===================
- ``vqnetrun`` agora suporta o modo ``--backend nccl``, com inicialização distribuída controlável pelos parâmetros ``--nproc_per_node``, ``--nccl_socket_ifname``.
- Adicionada a interface ``VQCQCloudLayer`` para submeter módulos VQC a chips reais QCloud ou simuladores locais pyqpanda3, com suporte a retropropagação parameter_shift.
- ``CommController`` adiciona um método ``destroy()`` para limpeza de recursos de comunicação NCCL.

Alterado
===================
- O backend padrão foi alterado para ``pyvqnet-ad``.
- Removidas as interfaces obsoletas ``QuantumLayerMultiProcess``, ``DataParallelHybirdVQCQpandaQVMLayer`` e ``HybirdVQCQpanda3QVMLayer``.
- ``split_group`` renomeado para ``split_groups``.
- Depende do runtime NVIDIA para CUDA 12.6.
- O padrão de "chip_id" foi alterado para "WK_C180".
- ``ComplexEntangelingTemplate`` foi renomeado para ``ComplexEntanglingTemplate``.
- ``vqc.rst``: adicionada a seção de benchmark "Test 2: 10-Qubit VQC Gradient Comparison" comparando VQNet / TorchQuantum / DeepQuantum / Pennylane / MindQuantum.
- Tabela de especificações de benchmark atualizada com CUDA 12.6, torchquantum 0.2.0 e mindquantum 0.12.0.

Corrigido
===================
- Corrigido problema de estouro de inteiro no kernel CUDA ``roll``.
- Corrigido suporte de ``cuda_masked_fill`` para tipos complex64/complex128.
- Corrigida a computação forward de ``log_softmax`` que produzia valores +inf incorretos sob ``bfloat16``.
- Corrigidos erros de dispositivo causados pela falta de ``CUDAGuard`` durante acesso à memória entre GPUs.
- Corrigidos muitos erros de digitação.


[v2.17.3] - 2026-03-31
***************************

Adicionado
===================
- Adicionado o tipo de dado bfloat16.
- Adicionadas interfaces de comunicação NCCL assíncronas: ``nccl_async_all_gather``, ``nccl_async_all_reduce``, ``nccl_async_reduce``, ``nccl_async_broadcast``, ``nccl_async_send`` e ``nccl_async_recv``.
- Adicionado suporte ao mais recente chip Origin Quantum com ID de chip ``WK_C180``.
- Adicionados ``data_ptr`` e outras interfaces; suporte a `triton <https://triton-lang.org/main/index.html>`_ adicionado experimentalmente.

Alterado
===================

- O backend padrão foi alterado para "pyvqnet-ad".
- A computação no MacOS é implementada com base nas instruções arm neon.
- A interface ``matmul`` suporta dados com mais de 4 dimensões.
- Reduzida a dependência de algumas bibliotecas de runtime CUDA ao instalar o pacote whl.
- O tipo de dado QTensor no pyvqnet não é mais um inteiro, mas um tipo de dado específico.
- Modificada a lógica de pickle do QTensor; grad não é mais preservado durante pickle.
- Removido is_dense.
- Removido ``QuantumBatchAsyncQcloudLayer`` do pq2.
- Documentação atualizada para ``QuantumBatchAsyncQcloudLayer`` do pq3.

Corrigido
===================
- Corrigido o erro de ``Linear`` quando ``use_bias=False``.
- Corrigido o problema de ``MAX_GPUS``; o número máximo atual de GPUs agora é 16.
- Corrigido erro de importação no Windows Jupyter.


[v2.17.2] -2025-11-18
***********************************


Adicionado
===================

- Adicionado suporte ao backend ``torch`` através da interface de gradiente natural quântico QNG.
- Adicionado o backend ``pyvqnet-ad``, que usa um backend de diferenciação automática em C++ similar ao torch. A estrutura de dados ainda usa o ``_core.Tensor`` original e suporta a grande maioria das interfaces existentes.
- Adicionada a documentação "Benchmarking of Variational Quantum Circuit's Gradients for Batch Data".

Alterado
===================

- Removidos `HybirdVQCQpanda3QVMLayer`, `QuantumLayerMultiProcess`, `TorchHybirdVQCQpanda3QVMLayer`;
- Removidos `is_csr`, `csr_members`, `SparseHamiltonian`, `csr_to_dense`, `dense_to_csr`;
- Adicionadas as interfaces `QiskitLayer` e `CirqLayer`.
- Adicionado suporte a `if_print_qcloud_log` na camada `QuantumBatchAsyncQcloudLayer` para exibir logs do qcloud.
- O comando de instalação foi alterado para `pip install pyvqnet --upgrade`.
- A versão suportada do Python é `Python 3.10`.
- Modificado o comando de instalação do mpicxx especificado.



Corrigido
===================

- Suporte aos valores de retorno da versão mais recente do pyqpanda2QCloud;
- Adicionada verificação do dispositivo dos dados de entrada para a interface de produto interno;
- Corrigido um bug em `TorchModule`;



[v2.17.1] - 2025-8-22
***************************

Adicionado
===================

- Adicionadas a interface do algoritmo SPSA de gradiente natural quântico (qnspsa), a máquina de Born de circuito quântico (QBM), a interface de gradiente natural quântico com momentum e um exemplo de QGRU puramente quântico.
- Adicionado o backend ``torch_native``.
- Adicionada uma interface de paralelismo de bits para suportar circuitos quânticos com paralelismo de bits, e adicionada uma função de reordenação de bits para reduzir o número de trocas de bits.
- Adicionado o método ``split_groups``.

Alterado
==================
- A implementação da camada Linear foi alterada de `:math:`y = Ax + b`` para `:math:`y = x@A.T + b``
- Modificado o parâmetro ``obs`` na interface `MeasureAll`.
- Removida a interface ``QuantumLayerES``. Nomes de parâmetros alterados de ControllComm para ControlComm em `allgather_group`, `allreduce_group`, `reduce_group`, `broadcast_group` e outras interfaces.
- Removida a interface `ncclsplitGroup`.

Corrigido
===================
- Resolvidos problemas de atraso de sincronização com interfaces de comunicação distribuída.
- Modificadas as definições de interfaces de comunicação distribuída.
- Resolvido um problema em que os cálculos de gradiente adjunto não suportavam ``PauliX``, ``PauliY`` e ``PauliZ``.


[v2.17.0] - 2025-4-22
***************************

Adicionado
===================

- Adicionada implementação de backend de rede de tensores para módulos de circuito quântico, incluindo suporte a portas lógicas básicas, medição e circuitos quânticos complexos.
- Adicionada implementação de backend de rede de tensores para construção de circuitos quânticos de muitos qubits.
- Adicionada interface QTensor.swapaxes, também conhecida como swapaxis.

Alterado
===================
- Operações matriciais usando openblas.
- Uso de sleef para operações SIMD em CPU.
- Removido qnn.MeasurePauliSum.
- Exibição de aviso ao usar cálculos com backend torch quando o torch está abaixo da versão 2.4.

Corrigido
===================
- Resolvido o problema dos estados do QMachine ao salvar modelos.
- Resolvido o problema com layernorm e groupnorm quando ``affine=False``.
- Resolvido o problema com ``QuantumLayerAdjoint`` no modo eval.

[v2.16.0] - 2025-1-15
***************************

Adicionado
===================

- Adicionada uma interface para cálculo de circuito quântico usando pyqpanda3.
- A interface MeasureAll suporta operadores de Pauli compostos.
- Adicionadas as interfaces DataParallelVQCAdjointLayer e DataParallelVQCLayer.

Alterado
===================

- Removidas funções ONNX desatualizadas e a maioria das interfaces que integravam pyqpanda, mantendo algumas interfaces usadas no código de exemplo.
- Modificada a interface ``VQC_QuantumEmbedding``.
- Ao instalar este pacote, o pyqpanda não é mais instalado; em vez disso, o pyqpanda3 é instalado.
- A interface VQC suporta o uso de `x[:,:2]` como parâmetros de entrada, que originalmente só suportava o formato `x[:,[2]]`.
- Este software suporta Python 3.10.

Corrigido
===================
- Resolvido o problema de vazamento de memória.
- Resolvido o problema de números aleatórios em GPU.
- Para operações relacionadas a reduce, a dimensão máxima suportada de array foi aumentada de 8 para 30.
- Código otimizado e velocidade de execução do Python melhorada em alguns casos.
  
  
[v2.15.0] - 2024-11-19
***************************

Adicionado
===================

- Adicionada a interface `pyvqnet.backends.set_backend()`. Quando os usuários instalam `torch`, o `torch` pode ser usado para realizar cálculos matriciais e cálculos de circuito quântico variacional do QTensor. Para detalhes, consulte o documento :ref:`torch_api`.
- Adicionado `pyvqnet.nn.torch` para herdar a interface de rede neural e a interface neural de circuito quântico variacional de `torch.nn.Module`. Para detalhes, consulte o documento :ref:`torch_api`.

Alterado
===================
- Interface diag modificada.
- Implementação de all_gather modificada para ser consistente com torch.distributed.all_gather.
- Modificado `QTensor` para suportar dados de até 30 dimensões.
- Modificado `mpi4py` exigido para funções distribuídas para requerer versão 4.0.1 ou superior.

Corrigido
===================
- Algumas implementações de números aleatórios não conseguiam fixar a semente devido ao OpenMP.
- Corrigidos alguns bugs na inicialização distribuída.


[v2.14.0] - 2024-09-30
***************************

Adicionado
===================

- Adicionados algoritmos de block-encoding: ``VQC_LCU``, ``VQC_FABLE``, ``VQC_QSVT``, e implementações de algoritmo qpanda ``QPANDA_QSVT``, ``QPANDA_LCU``, ``QPANDA_FABLE``.
- Adicionada adição de inteiros em bits quânticos ``vqc_qft_add_to_register``, adição de números em dois bits quânticos ``vqc_qft_add_two_register`` e multiplicação de números em dois bits quânticos ``vqc_qft_mul``.
- Adicionado módulo de treinamento híbrido qpanda e vqc ``HybirdVQCQpandaQVMLayer``.
- Adicionadas implementações das interfaces ``einsum``, ``moveaxis``, ``eigh``, ``diagonal`` entre outras.
- Adicionadas funções de computação paralela de tensores em computação distribuída: ``ColumnParallelLinear``, ``RowParallelLinear``.
- Adicionada função Zero stage-1 em computação distribuída: ``ZeroModelInitial``.
- ``QuantumBatchAsyncQcloudLayer``: quando ``diff_method == "random_coordinate_descent"``, usa seleção aleatória de parâmetros para cálculo de gradiente em vez de PSR.

Alterado
===================
- Deletada a parte xtensor.
- A documentação da API foi parcialmente reestruturada. Distinguidos entre exemplos de aprendizado de máquina quântico baseados em diferenciação automática e aqueles baseados em qpanda, e distinguidos entre interfaces de aprendizado de máquina quântico baseadas em diferenciação automática e interfaces de exemplo baseadas em qpanda.
- `matmul` suporta 1d@1d, 2d@1d, 1d@2d.
- Adicionados alguns aliases de camada de computação quântica: ``QpandaQCircuitVQCLayer`` = ``QuantumLayer``, ``QpandaQCircuitVQCLayerLite`` = ``QuantumLayerV2``, ``QpandaQProgVQCLayer`` = ``QuantumLayerV3``.

Corrigido
===================
- Modificadas as interfaces de comunicação subjacentes ``allreduce``, ``allgather``, ``reduce``, ``broadcast`` na função de computação distribuída, e adicionado suporte para comunicação de dados ``core.Tensor``.
- Resolvido o bug na geração de números aleatórios.
- Resolvido o erro na conversão de ``RXX``, ``RYY``, ``RZZ``, ``RZX`` do VQC para originIR.


[v2.13.0] - 2024-07-30
***************************

Adicionado
==================

- Adicionadas interfaces `no_grad`, `GroupNorm`, `Interpolate`, `contiguous`, `QuantumLayerV3`, `fuse_model`, `SDPA`.
- Adicionado método Quantum Dropout para evitar overfitting.

Alterado
===================

- Adicionada interface affine a `BatchNorm`, `LayerNorm`, `GroupNorm`.
- A interface `diag` agora retorna saída 1d na diagonal para entrada 2d, consistente com torch.
- Operações como slice e permute tentarão usar o método view para retornar um QTensor em memória compartilhada.
- Todas as interfaces suportam entrada não contígua.
- `Adam` suporta o parâmetro weight_decay.

Corrigido
===================
- Modificado o erro de algumas funções de decomposição de portas lógicas do VQC.
- Corrigido o problema de vazamento de memória de algumas funções.
- Corrigido o problema de `QuantumLayerMultiProcess` não suportar entrada em GPU.
- Modificado o método de inicialização de parâmetros padrão de `Linear`.


[v2.12.0] - 2024-05-01
***************************

Adicionado
===================

- Adicionada interface PipelineParallelTrainingWrapper.
- Adicionadas interfaces `Gelu`, `DropPath`, `binomial`, `adamW`.
- Adicionado `QuantumBatchAsyncQcloudLayer` para suportar cálculo de simulação em máquina virtual local do pyqpanda.
- Adicionado `QuantumBatchAsyncQcloudLayer` do xtensor para suportar cálculo de simulação em máquina virtual local do pyqpanda e cálculo em máquina real.
- Habilita QTensor a ser deepcopy e pickle.
- Adicionado comando de inicialização de computação distribuída `vqnetrun`, usado ao utilizar a interface de computação distribuída.
- Adicionada interface de máquina real para método de cálculo de gradiente ES `QuantumBatchAsyncQcloudLayerES` para suportar simulação em VM local, bem como cálculos em máquina real para pyqpanda.
- Adicionadas interfaces de comunicação de dados `allreduce`, `reduce`, `broadcast`, `allgather`, `send`, `recv`, etc., que suportam QTensor em computação distribuída.

Alterado
===================

- Adicionadas novas dependências "Pillow" e "hjson" ao pacote de instalação; adicionadas novas dependências "psutil" e "cloudpickle" em sistemas Linux.
- Otimizadas as velocidades de execução de softmax e transpose em GPU.
- Compilado usando cuda11.8.
- Integração de interfaces de computação distribuída baseadas em CPU e GPU.

Corrigido
===================
- Reduzido o consumo de memória ao iniciar a versão Linux-GPU.
- Corrigido o problema de vazamento de memória das funções select e power.
- Removidos os métodos de atualização de parâmetros e gradientes do modelo `nccl_average_parameters_reduce`, `nccl_average_grad_reduce` baseados no método reduce para CPU e GPU.

[v2.11.0] - 2024-03-01
***************************

Adicionado
===================

- Adicionada nova API `QNG` (Quantum Natural Gradient) e demonstração.
- Adicionada otimização de circuito quântico, como as APIs e demonstrações `wrapper_single_qubit_op_fuse`, `wrapper_commute_controlled`, `wrapper_merge_rotations`.
- Adicionados `CY`, `SparseHamiltonian`, `HermitianExpval`.
- Adicionados `is_csr`, `is_dense`, `dense_to_csr`, `csr_to_dense`.
- Adicionado `QuantumBatchAsyncQcloudLayer` para suportar cálculo em chip real QCloud do pyqpanda, `expval_qcloud`.
- Adicionadas implementações de interface baseadas em NCCL para treinamento paralelo de modelo de dados de computação distribuída multi-GPU em um único nó: `nccl_average_parameters_allreduce`, `nccl_average_parameters_reduce`, `nccl_average_grad_allreduce`, `nccl_average_grad_reduce`, e classes para controlar a inicialização do NCCL e operações relacionadas `NCCL_api`.
- Adicionada interface de cálculo de gradiente de estratégia de evolução de linha quântica `QuantumLayerES`.

Alterado
===================

- Circuito `VQC_CSWAP` refatorado para `CSWAP`.
- Documentos QNG antigos deletados.
- Parâmetro `num_wires` inútil removido de funções e classes em `pyvqnet.qnn.vqc`.
- APIs `MeasureAll`, `Probability` refatoradas.
- Parâmetro qtype adicionado a `QuantumMeasure`.

Corrigido
===================
- `QuantumMeasure` alterado de slots para shots.

[v2.10.0] - 2023-12-30
***************************

Adicionado
===========
- Adicionadas novas interfaces em pyvqnet.qnn.vqc: IsingXX, IsingXY, IsingYY, IsingZZ, SDG, TDG, PhaseShift, MultiRZ, MultiCnot, MultixCnot, ControlledPhaseShift, SingleExcitation, DoubleExcitation, VQC_AllSinglesDoubles, ExpressiveEntanglingAnsatz, etc.;
- Adicionada interface pyvqnet.qnn.vqc.QuantumLayerAdjoint que suporta cálculo de gradiente adjunto;
- Suportada a função de conversão mútua entre originIR e VQC;
- Suportadas informações de módulos clássicos e quânticos em modelos VQC estatísticos;
- Adicionados dois casos sob o modelo híbrido de rede neural clássico-quântica: modelo de rede neural convolucional quântica baseado em pequenas amostras e modelo de função kernel quântica para reconhecimento de dígitos manuscritos.


[v2.9.0] - 2023-09-08
***************************

Adicionado
===================
- A definição da interface xtensor foi adicionada para suportar paralelismo automático de operadores e múltiplos backends CPU/GPU. Inclui mais de 150 interfaces para matemática comum, lógica e cálculos matriciais para arrays multidimensionais, bem como camadas de rede neural clássicas comuns e otimizadores.

Alterado
===================
- Versão de v2.0.8 para v2.9.0.
- Pacotes são enviados no repositório PyPI da empresa, use ``pip install pyvqnet --index-url <pypi_url>``.


[v2.0.8] - 2023-07-26
***************************

Adicionado
===================
- Adicionadas interfaces existentes para suportar computação (gpu) dos tipos complex128, complex64, double, float, uint8, int8, bool, int16, int32, int64 e outros.
- Portas lógicas básicas baseadas em vqc: Hadamard, CNOT, I, RX, RY, PauliZ, PauliX, PauliY, S, RZ, RXX, RYY, RZZ, RZX, X1, Y1, Z1, U1, U2, U3, T, SWAP, P, TOFFOLI, CZ, CR, ISWAP.
- Circuito quântico combinado baseado em vqc: VQC_HardwareEfficientAnsatz, VQC_BasicEntanglerTemplate, VQC_StronglyEntanglingTemplate, VQC_QuantumEmbedding, VQC_RotCircuit, VQC_CRotCircuit, VQC_CSWAPcircuit, VQC_Controlled_Hadamard, VQC_CCZ, VQC_FermionicSingleExcitation, VQC_FermionicDoubleExcitation, VQC_UCCSD, VQC_QuantumPoolingCircuit, VQC_BasisEmbedding, VQC_AngleEmbedding, VQC_AmplitudeEmbedding, VQC_IQPEmbedding.
- Métodos de medição baseados em vqc: VQC_Purity, VQC_VarMeasure, VQC_DensityMatrixFromQstate, Probability, MeasureAll.


[v2.0.7] - 2023-07-03
***************************

Adicionado
===================
- Para rede neural clássica, adicionadas interfaces kron, gather, scatter, broadcast_to.
- Adicionado suporte a diferentes precisões de dados: o tipo de dado dtype suporta kbool, kuint8, kint8, kint16, kint32, kint64, kfloat32, kfloat64, kcomplex64, kcomplex128, que representam respectivamente bool, uint8_t, int8_t, int16_t, int32_t, int64_t, float, double, complex<float>, complex<double>.
- Suporte a Python 3.8, 3.9, 3.10.

Alterado
===================
- A função init das classes QTenor e Module adiciona o parâmetro `dtype`. Os tipos do índice QTenor e da entrada de algumas camadas de rede neural são restritos.
- Rede neural quântica: devido a problemas de compatibilidade com MacOS, as interfaces Mnist_Dataset e CIFAR10_Dataset foram removidas.

[v2.0.6] - 2023-02-22
***************************


Adicionado
===================

- Rede neural clássica: adicionadas interfaces multinomial, pixel_shuffle, pixel_unshuffle; adicionado numel para QTensor; adicionada função de pool de memória dinâmica em CPU; adicionada interface init_from_tensor para Parameter.
- Rede neural clássica: adicionadas interfaces Dynamic_LSTM, Dynamic_RNN, Dynamic_GRU.
- Rede neural clássica: adicionadas interfaces pad_sequence, pad_packed_sequence, pack_pad_sequence.
- Rede neural quântica: adicionadas interfaces CCZ, Controlled_Hadamard, FermionicSingleExcitation, UCCSD, QuantumPoolingCircuit.
- Rede neural quântica: adicionadas interfaces Quantum_Embedding, Mnist_Dataset, CIFAR10_Dataset, grad, Purity.
- Rede neural quântica: adicionados exemplos baseados em recorte de gradiente, quanvolution, expressividade de circuito quântico, planalto estéril (barren plateau) e aprendizado por reforço quântico QDRL.

Alterado
===================

- Documentação da API: reestruturado o conteúdo; adicionado o módulo "quantum machine learning research"; "VQNet2ONNX module" alterado para "Other Utility Functions".



Corrigido
===================

- Rede neural clássica: resolvido o problema de que a mesma semente aleatória produzia diferentes distribuições normais entre plataformas.
- Rede neural quântica: implementados expval, ProbMeasure, QuantumMeasure com suporte para máquina virtual GPU do QPanda.


[v2.0.5] - 2022-12-25
***************************


Adicionado
===================

- Rede neural clássica: adicionada implementação de log_softmax; adicionada função export_model do modelo para ONNX.
- Rede neural clássica: suporta a conversão da maioria dos módulos de rede neural clássica existentes para ONNX. Para detalhes, consulte o documento da API "VQNet2ONNX module".
- Rede neural quântica: adicionadas interfaces VarMeasure, MeasurePauliSum, Quantum_Embedding, SPSA e outras.
- Rede neural quântica: adicionados LinearGNN, ConvGNN, QMLP, gradiente natural quântico, algoritmo de deslocamento de parâmetro quântico aleatório, algoritmo DoublySGD, etc.


Alterado
===================

- Redes neurais clássicas: adicionadas verificações de dimensionalidade para interfaces BN1d, BN2d.

Corrigido
==================

- Resolvido o bug de verificação de parâmetros de maxpooling.
- Resolvido o bug de slice [::-1].


[v2.0.4] - 2022-09-20
***************************


Adicionado
==================

- Rede neural clássica: adicionada implementação de LayernormNd, suportando cálculo de layernorm para dados multidimensionais.
- Rede neural clássica: adicionadas interfaces de função de perda CrossEntropyLoss e NLL_Loss, suportando entrada de 1 a N dimensões.
- Rede neural quântica: adicionados templates de circuito comuns: HardwareEfficientAnsatz, StronglyEntanglingTemplate, BasicEntanglerTemplate.
- Rede neural quântica: adicionada interface Mutal_info para calcular informação mútua de subsistemas de qubits, entropia de Von Neumann VB_Entropy e matriz densidade DensityMatrixFromQstate.
- Rede neural quântica: adicionado exemplo de algoritmo perceptron quântico QuantumNeuron e exemplo de algoritmo de série de Fourier quântica.
- Rede neural quântica: adicionada interface QuantumLayerMultiProcess que suporta operação acelerada por múltiplos processos de circuitos quânticos.

Alterado
==================

- Rede neural clássica: suporta grupo de convolução (group), taxa de dilatação (dilation_rate) de convolução dilatada e preenchimento de valor arbitrário como parâmetros para convolução unidimensional Conv1d, convolução bidimensional Conv2d e deconvolução ConvT2d.
- Operação de broadcast ignorada para dados na mesma dimensão, reduzindo lógica de execução desnecessária.

Corrigido
==================

- Resolvido o problema em que a função stack calculava incorretamente sob certos parâmetros.


[v2.0.3] - 2022-07-15
***************************


Adicionado
==================

- Adicionado suporte a stack e interfaces de rede neural recorrente bidirecional: RNN, LSTM, GRU.
- Adicionadas interfaces para indicadores comuns de desempenho de cálculo: MSE, RMSE, MAE, R_Square, precision_recall_f1_2_score, precision_recall_f1_Multi_score, precision_recall_f1_N_score, auc_calculate.
- Adicionado exemplo de algoritmo de SVM kernel quântico.

Alterado
==================

- Velocidade de impressão acelerada quando há muitos dados QTensor.
- Uso de openmp para acelerar cálculos no Windows e Linux.

Corrigido
==================

- Resolvido o problema em que alguns métodos de importação do Python falhavam ao importar.
- Resolvido o problema de cálculos repetidos da camada de normalização em lote (BN).
- Resolvido o bug em que as interfaces ``QTensor.reshape`` e ``transpose`` não conseguiam calcular gradientes.
- Adicionada validação de forma dos parâmetros de entrada para a interface ``tensor.power``.

[v2.0.2] - 2022-05-15
***************************


Adicionado
==================

- Adicionados topK, argtoK.
- Adicionado cumsum.
- Adicionado masked_fill.
- Adicionados triu, tril.
- Adicionados exemplos de distribuição aleatória gerada por QGAN.

Alterado
==================

- Suporte a índice de slice avançado e índice de slice comum.
- matmul suporta operações de tensor 3D e 4D.
- Modificada a implementação da função HardSigmoid.

Corrigido
==================

- Resolvido o problema em que camadas de convolução, normalização em lote, deconvolução, pooling e outras não armazenavam em cache variáveis internas, causando problemas de cálculo de gradiente durante múltiplas retropropagações após uma única propagação forward.
- Corrigida implementação e exemplo da camada QLinear.
- Resolvido o problema em que Image falhava ao carregar ao importar VQNet em um ambiente conda no macOS.




[v2.0.1] - 2022-03-30
**************************


Adicionado
==================

- Mais de 100 interfaces básicas de estrutura de dados QTensor foram adicionadas, incluindo funções de criação, funções lógicas, funções matemáticas e operações matriciais.
- Adicionadas 14 funções básicas de rede neural, incluindo convolução, deconvolução, pooling, etc.
- Adicionadas 4 funções de perda, incluindo MSE, BCE, CCE, SCE, etc.
- Adicionadas 10 funções de ativação, incluindo ReLu, Sigmoid, ELU, etc.
- Adicionados 6 otimizadores, incluindo SGD, RMSPROP, ADAM, etc.
- Adicionados exemplos de aprendizado de máquina: QVC, QDRL, Q-KMEANS, QUnet, HQCNN, VSQL, Quantum Autoencoder.
- Adicionadas camadas de aprendizado de máquina quântico: QuantumLayer, NoiseQuantumLayer.
