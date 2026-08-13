Registro delle modifiche di VQNet
#################################


[v2.18.0] - 2026-04-22
***************************

Aggiunto
===================
- ``vqnetrun`` ora supporta la modalità ``--backend nccl``, con avvio distribuito controllabile tramite i parametri ``--nproc_per_node``, ``--nccl_socket_ifname``.
- Aggiunta l'interfaccia ``VQCQCloudLayer`` per inviare il modulo VQC a chip reali QCloud o simulatori locali pyqpanda3, con supporto della backpropagation parameter_shift.
- ``CommController`` aggiunge un metodo ``destroy()`` per la pulizia delle risorse di comunicazione NCCL.

Modificato
===================
- Il backend predefinito è stato cambiato in ``pyvqnet-ad``.
- Rimosse le interfacce deprecate ``QuantumLayerMultiProcess``, ``DataParallelHybirdVQCQpandaQVMLayer`` e ``HybirdVQCQpanda3QVMLayer``.
- ``split_group`` rinominato in ``split_groups``.
- Dipende dal runtime NVIDIA per CUDA 12.6.
- Il valore predefinito di "chip_id" è cambiato in "WK_C180".
- ``ComplexEntangelingTemplate`` è stato rinominato in ``ComplexEntanglingTemplate``.
- ``vqc.rst``: aggiunta la sezione benchmark "Test 2: 10-Qubit VQC Gradient Comparison" che confronta VQNet / TorchQuantum / DeepQuantum / Pennylane / MindQuantum.
- Aggiornata la tabella delle specifiche benchmark con CUDA 12.6, torchquantum 0.2.0 e mindquantum 0.12.0.

Corretto
===================
- Risolto il problema di integer overflow nel kernel CUDA ``roll``.
- Risolto il supporto di ``cuda_masked_fill`` per i tipi complex64/complex128.
- Risolta la computazione forward di ``log_softmax`` che produceva valori +inf errati con ``bfloat16``.
- Risolti errori di dispositivo causati dalla mancanza di ``CUDAGuard`` durante l'accesso alla memoria cross-GPU.
- Corretti molti errori di battitura.


[v2.17.3] - 2026-03-31
***************************

Aggiunto
===================
- Aggiunto il tipo di dato bfloat16.
- Aggiunte le interfacce di comunicazione NCCL asincrone: ``nccl_async_all_gather``, ``nccl_async_all_reduce``, ``nccl_async_reduce``, ``nccl_async_broadcast``, ``nccl_async_send`` e ``nccl_async_recv``.
- Aggiunto il supporto per il più recente chip Origin Quantum con chip ID ``WK_C180``.
- Aggiunte ``data_ptr`` e altre interfacce, aggiunto sperimentalmente il supporto per `triton <https://triton-lang.org/main/index.html>`_.

Modificato
===================

- Il backend predefinito è stato cambiato in "pyvqnet-ad".
- La computazione su MacOS è implementata basandosi sulle istruzioni arm neon.
- L'interfaccia ``matmul`` supporta dati con più di 4 dimensioni.
- Ridotta la dipendenza da alcune librerie runtime cuda durante l'installazione del pacchetto whl.
- Il tipo di dato QTensor in pyvqnet non è più un intero, ma un tipo di dato specifico.
- Modificata la logica di pickle di QTensor, non serializza più grad.
- Rimossa is_dense.
- Rimossa pq2 ``QuantumBatchAsyncQcloudLayer``.
- Aggiornata la documentazione per pq3 ``QuantumBatchAsyncQcloudLayer``.

Corretto
===================
- Risolto l'errore di ``Linear`` quando ``use_bias=False``.
- Risolto il problema di ``MAX_GPUS``; il numero massimo attuale di GPU è ora 16.
- Risolto l'errore di importazione su Windows jupyter.


[v2.17.2] -2025-11-18
***********************************


Aggiunto
===================

- Aggiunto il supporto per il backend ``torch`` tramite l'interfaccia quantum natural gradient QNG.
- Aggiunto il backend ``pyvqnet-ad``, che utilizza un backend di differenziazione automatica in C++ simile a torch. La struttura dati utilizza ancora il ``_core.Tensor`` originale e supporta la stragrande maggioranza delle interfacce esistenti.
- Aggiunta la documentazione "Benchmarking of Variational Quantum Circuit's Gradients for Batch Data".

Modificato
===================

- Rimosse ``HybirdVQCQpanda3QVMLayer``, ``QuantumLayerMultiProcess``, ``TorchHybirdVQCQpanda3QVMLayer``;
- Rimossi ``is_csr``, ``csr_members``, ``SparseHamiltonian``, ``csr_to_dense``, ``dense_to_csr``;
- Aggiunte le interfacce ``QiskitLayer`` e ``CirqLayer``.
- Aggiunto il supporto ``if_print_qcloud_log`` al layer ``QuantumBatchAsyncQcloudLayer`` per la stampa dei log qcloud.
- Modificato il comando di installazione in ``pip install pyvqnet --upgrade``.
- La versione python supportata è ``Python 3.10``.
- Modificato il comando di installazione mpicxx specificato.



Corretto
===================

- Supporto per i valori restituiti dell'ultima versione di pyqpanda2QCloud;
- Aggiunti controlli sul dispositivo dei dati di input per l'interfaccia del prodotto scalare;
- Risolto un bug in ``TorchModule``;



[v2.17.1] - 2025-8-22
***************************

Aggiunto
===================

- Aggiunte l'interfaccia dell'algoritmo quantum natural gradient SPSA (qnspsa), la quantum circuit born machine (QBM), l'interfaccia quantum natural gradient con momentum e un esempio di QGRU puramente quantistico.
- Aggiunto il backend ``torch_native``.
- Aggiunta un'interfaccia bit-parallel per supportare circuiti quantistici bit-parallel, e aggiunta una funzione di riordinamento dei bit per ridurre il numero di scambi di bit.
- Aggiunto il metodo ``split_groups``.

Modificato
==================
- Modificata l'implementazione del layer Linear da `:math:`y = Ax + b` a `:math:`y = x@A.T + b`
- Modificato il parametro ``obs`` nell'interfaccia ``MeasureAll``.
- Rimossa l'interfaccia ``QuantumLayerES``. Modificati i nomi dei parametri da ControllComm a ControlComm in ``allgather_group``, ``allreduce_group``, ``reduce_group``, ``broadcast_group`` e altre interfacce.
- Rimossa l'interfaccia ``ncclsplitGroup``.

Corretto
===================
- Risolti i problemi di ritardo di sincronizzazione con le interfacce di comunicazione distribuita.
- Modificate le definizioni delle interfacce di comunicazione distribuita.
- Risolto un problema per cui i calcoli del gradiente aggiunto non supportavano ``PauliX``, ``PauliY`` e ``PauliZ``.


[v2.17.0] - 2025-4-22
***************************

Aggiunto
===================

- Aggiunta l'implementazione del backend tensor network per i moduli di circuito quantistico, inclusi supporto per porte logiche di base, misurazione e circuiti quantistici complessi.
- Aggiunta l'implementazione del backend tensor network per la costruzione di circuiti quantistici a molti qubit.
- Aggiunta l'interfaccia QTensor.swapaxes, altro nome swapaxis.

Modificato
===================
- Operazioni sulle matrici che utilizzano openblas.
- Utilizzo di sleef per operazioni SIMD su CPU.
- Rimossa qnn.MeasurePauliSum.
- Visualizzazione di un avviso quando si utilizzano calcoli con backend torch con torch di versione inferiore alla 2.4.

Corretto
====================
- Risolto il problema degli stati QMachine durante il salvataggio dei modelli.
- Risolto il problema con layernorm e groupnorm quando ``affine=False``.
- Risolto il problema con ``QuantumLayerAdjoint`` in modalità eval.

[v2.16.0] - 2025-1-15
***************************

Aggiunto
===================

- Aggiunta un'interfaccia per il calcolo di circuiti quantistici utilizzando pyqpanda3.
- L'interfaccia MeasureAll supporta operatori di Pauli composti.
- Aggiunte le interfacce DataParallelVQCAdjointLayer e DataParallelVQCLayer.

Modificato
===================

- Rimosse le funzioni ONNX obsolete e la maggior parte delle interfacce che integravano pyqpanda, mantenendo alcune interfacce utilizzate nel codice di esempio.
- Modificata l'interfaccia ``VQC_QuantumEmbedding``.
- Durante l'installazione di questo pacchetto, pyqpanda non viene più installato; viene invece installato pyqpanda3.
- L'interfaccia VQC supporta l'uso di `x[:,:2]` come parametri di input, che in precedenza supportava solo il formato `x[:,[2]]`.
- Questo software supporta Python 3.10.

Corretto
====================
- Risolto il problema di memory leak.
- Risolto il problema dei numeri casuali su GPU.
- Per le operazioni correlate a reduce, la dimensione massima supportata per gli array è stata aumentata da 8 a 30.
- Ottimizzato il codice e migliorata la velocità di esecuzione del codice Python in alcuni casi.
   
   
[v2.15.0] - 2024-11-19
***************************

Aggiunto
===================

- Aggiunta l'interfaccia ``pyvqnet.backends.set_backend()``. Quando gli utenti installano ``torch``, è possibile utilizzare ``torch`` per eseguire calcoli matriciali e calcoli di circuiti quantistici variazionali di QTensor. Per i dettagli, vedere il documento :ref:`torch_api`.
- Aggiunto ``pyvqnet.nn.torch`` per ereditare l'interfaccia della rete neurale e l'interfaccia neurale del circuito quantistico variazionale di ``torch.nn.Module``. Per i dettagli, vedere il documento :ref:`torch_api`.

Modificato
===================
- Modificata l'interfaccia diag.
- Modificata l'implementazione di all_gather per essere coerente con torch.distributed.all_gather.
- Modifica di ``QTensor`` per supportare fino a 30 dimensioni.
- Modifica del requisito ``mpi4py`` per le funzioni distribuite alla versione 4.0.1 o superiore.

Corretto
===================
- Alcune implementazioni di numeri casuali non riuscivano a fissare il seed a causa di OpenMP.
- Risolti alcuni bug nell'avvio distribuito.


[v2.14.0] - 2024-09-30
***************************

Aggiunto
===================

- Aggiunti algoritmi di block-encoding: ``VQC_LCU``, ``VQC_FABLE``, ``VQC_QSVT`` e implementazioni degli algoritmi qpanda ``QPANDA_QSVT``, ``QPANDA_LCU``, ``QPANDA_FABLE``.
- Aggiunta l'addizione intera su bit quantistici ``vqc_qft_add_to_register``, addizione di numeri su due bit quantistici ``vqc_qft_add_two_register`` e moltiplicazione di numeri su due bit quantistici ``vqc_qft_mul``.
- Aggiunto il modulo di training ibrido qpanda e vqc ``HybirdVQCQpandaQVMLayer``.
- Aggiunte le implementazioni delle interfacce ``einsum``, ``moveaxis``, ``eigh``, ``diagonal`` e altre.
- Aggiunte funzioni di calcolo tensoriale parallelo nel calcolo distribuito: ``ColumnParallelLinear``, ``RowParallelLinear``.
- Aggiunta la funzione Zero stage-1 nel calcolo distribuito: ``ZeroModelInitial``.
- ``QuantumBatchAsyncQcloudLayer``: quando ``diff_method == "random_coordinate_descent"``, utilizza la selezione casuale dei parametri per il calcolo del gradiente invece di PSR.

Modificato
====================
- Eliminata la parte xtensor.
- La documentazione API è stata parzialmente ristrutturata. Distinti tra esempi di machine learning quantistico basati su differenziazione automatica e quelli basati su qpanda, e distinti tra interfacce di machine learning quantistico basate su differenziazione automatica e interfacce di esempio basate su qpanda.
- `matmul` supporta 1d@1d, 2d@1d, 1d@2d.
- Aggiunti alcuni alias per i layer di calcolo quantistico: ``QpandaQCircuitVQCLayer`` = ``QuantumLayer``, ``QpandaQCircuitVQCLayerLite`` = ``QuantumLayerV2``, ``QpandaQProgVQCLayer`` = ``QuantumLayerV3``.

Corretto
====================
- Modificate le interfacce di comunicazione sottostanti ``allreduce``, ``allgather``, ``reduce``, ``broadcast`` nella funzione di calcolo distribuito, e aggiunto il supporto per la comunicazione dati ``core.Tensor``.
- Risolto il bug nella generazione di numeri casuali.
- Risolto l'errore nella conversione di ``RXX``, ``RYY``, ``RZZ``, ``RZX`` di VQC in originIR.


[v2.13.0] - 2024-07-30
***************************

Aggiunto
==================

- Aggiunte le interfacce `no_grad`, `GroupNorm`, `Interpolate`, `contiguous`, `QuantumLayerV3`, `fuse_model`, `SDPA`.
- Aggiunto il metodo Quantum Dropout per evitare l'overfitting.

Modificato
===================

- Aggiunta l'interfaccia affine a `BatchNorm`, `LayerNorm`, `GroupNorm`.
- L'interfaccia `diag` ora restituisce output 1d sulla diagonale per input 2d, coerente con torch.
- Operazioni come slice e permute cercheranno di utilizzare il metodo view per restituire un QTensor in memoria condivisa.
- Tutte le interfacce supportano input non contigui.
- `Adam` supporta il parametro weight_decay.

Corretto
====================
- Modificato l'errore di alcune funzioni di decomposizione delle porte logiche di VQC.
- Risolto il problema di memory leak di alcune funzioni.
- Risolto il problema per cui ``QuantumLayerMultiProcess`` non supportava input GPU.
- Modificato il metodo di inizializzazione predefinito dei parametri di `Linear`


[v2.12.0] - 2024-05-01
***************************

Aggiunto
===================

- Aggiunta l'interfaccia PipelineParallelTrainingWrapper.
- Aggiunte le interfacce `Gelu`, `DropPath`, `binomial`, `adamW`.
- Aggiunto `QuantumBatchAsyncQcloudLayer` per supportare il calcolo di simulazione su macchina virtuale locale di pyqpanda.
- Aggiunto xtensor `QuantumBatchAsyncQcloudLayer` per supportare il calcolo di simulazione su macchina virtuale locale e il calcolo su macchina reale di pyqpanda.
- Abilita QTensor a essere deepcopy e pickle.
- Aggiunto il comando di avvio per il calcolo distribuito `vqnetrun`, utilizzato quando si utilizza l'interfaccia di calcolo distribuito.
- Aggiunta l'interfaccia per macchina reale del metodo di calcolo del gradiente ES `QuantumBatchAsyncQcloudLayerES` per supportare simulazioni su macchina virtuale locale e calcoli su macchina reale per pyqpanda.
- Aggiunte le interfacce di comunicazione dati `allreduce`, `reduce`, `broadcast`, `allgather`, `send`, `recv`, ecc. che supportano QTensor nel calcolo distribuito.

Modificato
===================

- Aggiunte nuove dipendenze "Pillow" e "hjson" al pacchetto di installazione, aggiunte nuove dipendenze "psutil" e "cloudpickle" sui sistemi linux.
- Ottimizzata la velocità di esecuzione di softmax e transpose su GPU.
- Compilato utilizzando cuda11.8.
- Integrazione delle interfacce di calcolo distribuito basate su cpu e gpu.

Corretto
===================
- Ridotto il consumo di memoria all'avvio della versione Linux-GPU.
- Risolto il problema di memory leak delle funzioni select e power.
- Rimossi i metodi di aggiornamento dei parametri e dei gradienti del modello `nccl_average_parameters_reduce`, `nccl_average_grad_reduce` basati sul metodo reduce per cpu, gpu.

[v2.11.0] - 2024-03-01
***************************

Aggiunto
===================

- Aggiunte la nuova API `QNG` (Quantum Natural Gradient) e la relativa demo.
- Aggiunta l'ottimizzazione di circuiti quantistici, come le api `wrapper_single_qubit_op_fuse`, `wrapper_commute_controlled`, `wrapper_merge_rotations` e la relativa demo.
- Aggiunti `CY`, `SparseHamiltonian`, `HermitianExpval`.
- Aggiunti `is_csr`, `is_dense`, `dense_to_csr`, `csr_to_dense`.
- Aggiunto `QuantumBatchAsyncQcloudLayer` per supportare il calcolo su chip reale QCloud di pyqpanda, `expval_qcloud`.
- Aggiunte implementazioni di interfacce basate su NCCL per l'addestramento parallelo di modelli su dati distribuiti multi-GPU su singolo nodo `nccl_average_parameters_allreduce`, `nccl_average_parameters_reduce`, `nccl_average_grad_allreduce`, `nccl_average_grad_reduce` e classi per controllare l'inizializzazione NCCL e le operazioni correlate `NCCL_api`.
- Aggiunta l'interfaccia di calcolo del gradiente basata sull'evoluzione delle linee quantistiche `QuantumLayerES`.

Modificato
===================

- Rifattorizzato il circuito `VQC_CSWAP` in `CSWAP`.
- Eliminati i vecchi documenti QNG.
- Rimosso il parametro inutile `num_wires` da `pyvqnet.qnn.vqc` per funzioni e classi.
- Rifattorizzate le api `MeasureAll`, `Probability`.
- Aggiunto il parametro qtype a `QuantumMeasure`.

Corretto
===================
- Modificati gli slot di `QuantumMeasure` in shots.

[v2.10.0] - 2023-12-30
***************************

Aggiunto
===========
- Aggiunte nuove interfacce sotto pyvqnet.qnn.vqc: IsingXX, IsingXY, IsingYY, IsingZZ, SDG, TDG, PhaseShift, MultiRZ, MultiCnot, MultixCnot, ControlledPhaseShift, SingleExcitation, DoubleExcitation, VQC_AllSinglesDoubles, ExpressiveEntanglingAnsatz, ecc.;
- Aggiunta l'interfaccia pyvqnet.qnn.vqc.QuantumLayerAdjoint che supporta il calcolo del gradiente aggiunto;
- Supportata la funzione di conversione reciproca tra originIR e VQC;
- Supportate le informazioni sui moduli classici e quantistici nei modelli VQC statistici;
- Aggiunti due casi nel modello ibrido rete neurale classico-quantistica: modello di rete neurale convoluzionale quantistica basato su piccoli campioni e modello di funzione kernel quantistica per il riconoscimento di cifre scritte a mano.


[v2.9.0] - 2023-09-08
***************************

Aggiunto
===================
- Aggiunta la definizione dell'interfaccia xtensor per supportare il parallelismo automatico degli operatori e molteplici backend CPU/GPU. Include più di 150 interfacce per matematica comune, logica e calcoli matriciali per array multidimensionali, nonché comuni layer di reti neurali classiche e ottimizzatori.

Modificato
===================
- Versione da v2.0.8 portata a v2.9.0.
- I pacchetti sono caricati nel repository PyPI dell'azienda, utilizzare ``pip install pyvqnet --index-url <pypi_url>``.


[v2.0.8] - 2023-07-26
***************************

Aggiunto
===================
- Aggiunte interfacce esistenti per supportare il calcolo (gpu) con tipi complex128, complex64, double, float, uint8, int8, bool, int16, int32, int64 e altri.
- Porte logiche di base basate su vqc: Hadamard, CNOT, I, RX, RY, PauliZ, PauliX, PauliY, S, RZ, RXX, RYY, RZZ, RZX, X1, Y1, Z1, U1, U2, U3, T, SWAP, P, TOFFOLI, CZ, CR, ISWAP.
- Circuito quantistico combinato basato su vqc: VQC_HardwareEfficientAnsatz, VQC_BasicEntanglerTemplate, VQC_StronglyEntanglingTemplate, VQC_QuantumEmbedding, VQC_RotCircuit, VQC_CRotCircuit, VQC_CSWAPcircuit, VQC_Controlled_Hadamard, VQC_CCZ, VQC_FermionicSingleExcitation, VQC_FermionicDoubleExcitation, VQC_UCCSD, VQC_QuantumPoolingCircuit, VQC_BasisEmbedding, VQC_AngleEmbedding, VQC_AmplitudeEmbedding, VQC_IQPEmbedding.
- Metodi di misurazione basati su vqc: VQC_Purity, VQC_VarMeasure, VQC_DensityMatrixFromQstate, Probability, MeasureAll.


[v2.0.7] - 2023-07-03
***************************

Aggiunto
===================
- Per la rete neurale classica, aggiunte le interfacce kron, gather, scatter, broadcast_to.
- Aggiunto il supporto per diverse precisioni di dati: il tipo di dato dtype supporta kbool, kuint8, kint8, kint16, kint32, kint64, kfloat32, kfloat64, kcomplex64, kcomplex128, che rappresentano rispettivamente bool, uint8_t, int8_t, int16_t, int32_t, int64_t, float, double, complex<float>, complex<double>.
- Supporto Python 3.8, 3.9, 3.10.

Modificato
===================
- La funzione init di QTensor e della classe Module aggiunge il parametro `dtype`. I tipi dell'indice di QTensor e dell'input di alcuni layer di rete neurale sono limitati.
- Rete neurale quantistica, a causa di problemi di compatibilità con MacOS, le interfacce Mnist_Dataset e CIFAR10_Dataset sono state rimosse.

[v2.0.6] - 2023-02-22
***************************


Aggiunto
===================

- Rete neurale classica, aggiunte interfacce: multinomial, pixel_shuffle, pixel_unshuffle, aggiunto numel per QTensor, aggiunta funzione di pool di memoria dinamica per CPU, aggiunta interfaccia init_from_tensor per Parameter.
- Rete neurale classica, aggiunte interfacce: Dynamic_LSTM, Dynamic_RNN, Dynamic_GRU.
- Rete neurale classica, aggiunte interfacce: pad_sequence, pad_packed_sequence, pack_pad_sequence.
- Rete neurale quantistica, aggiunte interfacce: CCZ, Controlled_Hadamard, FermionicSingleExcitation, UCCSD, QuantumPoolingCircuit.
- Rete neurale quantistica, aggiunte interfacce: Quantum_Embedding, Mnist_Dataset, CIFAR10_Dataset, grad, Purity.
- Rete neurale quantistica, aggiunti esempi: basati su gradient clipping, quanvolution, espressività del circuito quantistico, barren plateau e apprendimento per rinforzo quantistico QDRL.

Modificato
===================

- Documentazione API, ristrutturazione del contenuto, aggiunto il modulo "quantum machine learning research", cambiato "VQNet2ONNX module" in "Other Utility Functions".



corretto
===================

- Rete neurale classica, risolto il problema per cui lo stesso seed casuale produce distribuzioni normali diverse su piattaforme diverse.
- Rete neurale quantistica, implementato il supporto expval, ProbMeasure, QuantumMeasure per la macchina virtuale GPU di QPanda.


[v2.0.5] - 2022-12-25
***************************


Aggiunto
===================

- Rete neurale classica, aggiunta l'implementazione di log_softmax, aggiunta la funzione export_model del modello verso ONNX.
- Rete neurale classica, che supporta la conversione della maggior parte dei moduli di rete neurale classica esistenti in ONNX. Per i dettagli, fare riferimento al documento API "VQNet2ONNX module".
- Rete neurale quantistica, aggiunte le interfacce VarMeasure, MeasurePauliSum, Quantum_Embedding, SPSA e altre.
- Rete neurale quantistica, aggiunti LinearGNN, ConvGNN, QMLP, gradiente naturale quantistico, algoritmo quantum random parameter-shift, algoritmo DoublySGD, ecc.


Modificato
===================

- Reti neurali classiche, aggiunti controlli dimensionali per le interfacce BN1d, BN2d.

corretto
==================

- Risolto il bug del controllo dei parametri di maxpooling.
- Risolto il bug di slice [::-1].


[v2.0.4] - 2022-09-20
***************************


Aggiunto
==================

- Rete neurale classica, aggiunta l'implementazione di LayernormNd, che supporta il calcolo layernorm su dati multidimensionali.
- Rete neurale classica, aggiunte le interfacce di calcolo delle funzioni di perdita CrossEntropyLoss e NLL_Loss, con supporto per input da 1 a N dimensioni.
- Rete neurale quantistica, aggiunti template circuitali comuni: HardwareEfficientAnsatz, StronglyEntanglingTemplate, BasicEntanglerTemplate.
- Rete neurale quantistica, aggiunta l'interfaccia Mutal_info per il calcolo dell'informazione mutua di sottosistemi di qubit, entropia di Von Neumann VB_Entropy e matrice di densità DensityMatrixFromQstate.
- Rete neurale quantistica, aggiunto l'esempio di algoritmo percettrone quantistico QuantumNeuron, aggiunto l'esempio di algoritmo serie di Fourier quantistica.
- Rete neurale quantistica, aggiunta l'interfaccia QuantumLayerMultiProcess che supporta l'operazione accelerata multi-processo dei circuiti quantistici.

Modificato
==================

- Rete neurale classica, supporta il parametro group per la convoluzione a gruppi, dilation_rate per la convoluzione dilatata e padding con valori arbitrari come parametri per la convoluzione unidimensionale Conv1d, la convoluzione bidimensionale Conv2d e la deconvoluzione ConvT2d.
- Saltata l'operazione di broadcast per dati nella stessa dimensione, riducendo la logica di esecuzione non necessaria.

corretto
==================

- Risolto il problema per cui la funzione stack calcolava in modo errato con determinati parametri.


[v2.0.3] - 2022-07-15
***************************


Aggiunto
==================

- Aggiunto il supporto per stack, interfacce di rete neurale ricorrente bidirezionale: RNN, LSTM, GRU
- Aggiunte interfacce per indicatori di performance di calcolo comuni: MSE, RMSE, MAE, R_Square, precision_recall_f1_2_score, precision_recall_f1_Multi_score, precision_recall_f1_N_score, auc_calculate
- Aggiunto l'esempio di algoritmo quantum kernel SVM

Modificato
==================

- Accelerata la velocità di stampa quando ci sono troppi dati QTensor
- Utilizzo di openmp per accelerare i calcoli su Windows e Linux.

corretto
==================

- Risolto il problema per cui alcuni metodi di import Python non riuscivano a importare.
- Risolto il problema dei calcoli ripetuti del layer di batch normalization (BN).
- Risolto il bug per cui le interfacce ``QTensor.reshape`` e ``transpose`` non riuscivano a calcolare i gradienti.
- Aggiunta la validazione della forma dei parametri di input per l'interfaccia ``tensor.power``.

[v2.0.2] - 2022-05-15
***************************


Aggiunto
==================

- Aggiunti topK, argtoK
- Aggiunto cumsum
- Aggiunto masked_fill
- Aggiunti triu, tril
- Aggiunti esempi di distribuzione casuale generata da QGAN

Modificato
==================

- Supporto per indici slice avanzati e indici slice comuni
- matmul supporta operazioni tensoriali 3D, 4D
- Modificata l'implementazione della funzione HardSigmoid

corretto
==================

- Risolto il problema per cui convoluzione, batch normalization, deconvoluzione, layer di pooling e altri layer non memorizzavano nella cache le variabili interne, causando problemi di calcolo del gradiente durante piu' passaggi all'indietro dopo un passaggio in avanti.
- Risolta l'implementazione e l'esempio del layer QLinear
- Risolto il problema per cui Image non riusciva a caricare durante l'importazione di VQNet in un ambiente conda su macOS.



[v2.0.1] - 2022-03-30
**************************


Aggiunto
==================

- Aggiunte piu' di 100 interfacce di base della struttura dati QTensor, incluse funzioni di creazione, funzioni logiche, funzioni matematiche e operazioni matriciali.
- Aggiunte 14 funzioni di base della rete neurale, tra cui convoluzione, deconvoluzione, pooling, ecc.
- Aggiunte 4 funzioni di perdita, tra cui MSE, BCE, CCE, SCE, ecc.
- Aggiunte 10 funzioni di attivazione, tra cui ReLu, Sigmoid, ELU, ecc.
- Aggiunti 6 ottimizzatori, tra cui SGD, RMSPROP, ADAM, ecc.
- Aggiunti esempi di machine learning: QVC, QDRL, Q-KMEANS, QUnet, HQCNN, VSQL, Quantum Autoencoder.
- Aggiunti layer di machine learning quantistico: QuantumLayer, NoiseQuantumLayer.
