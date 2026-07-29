Registro de cambios de VQNet
###############################


[v2.18.0] - 2026-04-22
***************************

Agregado
===================
- ``vqnetrun`` ahora admite el modo ``--backend nccl``, con inicio distribuido controlable mediante los parámetros ``--nproc_per_node``, ``--nccl_socket_ifname``.
- Se agregó la interfaz ``VQCQCloudLayer`` para enviar el Módulo VQC a chips reales de QCloud o simuladores locales de pyqpanda3, compatible con retropropagación parameter_shift.
- ``CommController`` agrega un método ``destroy()`` para la limpieza de recursos de comunicación NCCL.

Cambiado
===================
- El backend predeterminado cambió a ``pyvqnet-ad``.
- Se eliminaron las interfaces obsoletas ``QuantumLayerMultiProcess``, ``DataParallelHybirdVQCQpandaQVMLayer`` e ``HybirdVQCQpanda3QVMLayer``.
- ``split_group`` renombrado a ``split_groups``.
- Depende del runtime de NVIDIA para CUDA 12.6.
- El valor predeterminado de "chip_id" cambió a "WK_C180".
- ``ComplexEntangelingTemplate`` ha sido renombrado a ``ComplexEntanglingTemplate``.
- ``vqc.rst``: se agregó la sección de referencia "Test 2: 10-Qubit VQC Gradient Comparison" que compara VQNet / TorchQuantum / DeepQuantum / Pennylane / MindQuantum.
- Se actualizó la tabla de especificaciones de referencia con CUDA 12.6, torchquantum 0.2.0 y mindquantum 0.12.0.

Corregido
===================
- Se corrigió el problema de desbordamiento de enteros en el kernel CUDA ``roll``.
- Se corrigió el soporte de ``cuda_masked_fill`` para tipos complex64/complex128.
- Se corrigió la computación forward de ``log_softmax`` que producía valores +inf incorrectos bajo ``bfloat16``.
- Se corrigieron errores de dispositivo causados por la falta de ``CUDAGuard`` durante el acceso a memoria entre GPUs.
- Se corrigieron muchos errores tipográficos.


[v2.17.3] - 2026-03-31
***************************

Agregado
===================
- Se agregó el tipo de dato bfloat16.
- Se agregaron las interfaces de comunicación asíncrona NCCL: ``nccl_async_all_gather``, ``nccl_async_all_reduce``, ``nccl_async_reduce``, ``nccl_async_broadcast``, ``nccl_async_send`` y ``nccl_async_recv``.
- Se agregó soporte para el chip más reciente de Origin Quantum con ID de chip ``WK_C180``.
- Se agregaron ``data_ptr`` y otras interfaces; se agregó soporte experimental para `triton <https://triton-lang.org/main/index.html>`_.

Cambiado
===================

- El backend predeterminado cambió a "pyvqnet-ad".
- La computación en MacOS se implementa basada en instrucciones arm neon.
- La interfaz ``matmul`` soporta datos con más de 4 dimensiones.
- Se redujo la dependencia de algunas librerías de runtime cuda al instalar el paquete whl.
- El tipo de dato QTensor en pyvqnet ya no es un entero, sino un tipo de dato específico.
- Se modificó la lógica de pickle de QTensor, ya no se serializa grad.
- Se eliminó is_dense.
- Se eliminó pq2 ``QuantumBatchAsyncQcloudLayer``.
- Se actualizó la documentación de pq3 ``QuantumBatchAsyncQcloudLayer``.

Corregido
===================
- Se corrigió el error de ``Linear`` cuando ``use_bias=False``.
- Se corrigió el problema de ``MAX_GPUS``; el número máximo actual de GPUs es ahora 16.
- Se corrigió el error de importación en windows jupyter.


[v2.17.2] -2025-11-18
***********************************


Agregado
===================

- Se agregó soporte para el backend ``torch`` a través de la interfaz de gradiente natural cuántico QNG.
- Se agregó el backend ``pyvqnet-ad``, que utiliza un backend diferencial automático en C++ similar a torch. La estructura de datos aún usa ``_core.Tensor`` original y soporta la gran mayoría de las interfaces existentes.
- Se agregó la documentación "Benchmarking of Variational Quantum Circuit's Gradients for Batch Data".

Cambiado
===================

- Se eliminaron `HybirdVQCQpanda3QVMLayer`, `QuantumLayerMultiProcess`, `TorchHybirdVQCQpanda3QVMLayer`;
- Se eliminaron `is_csr`, `csr_members`, `SparseHamiltonian`, `csr_to_dense`, `dense_to_csr`;
- Se agregaron las interfaces `QiskitLayer` y `CirqLayer`.
- Se agregó soporte de `if_print_qcloud_log` a la capa `QuantumBatchAsyncQcloudLayer` para imprimir registros de qcloud.
- Se cambió el comando de instalación a `pip install pyvqnet --upgrade`.
- La versión de python compatible es `Python 3.10`.
- Se modificó el comando de instalación de mpicxx especificado.



Corregido
===================

- Soporta los valores de retorno de la versión más reciente de pyqpanda2QCloud;
- Agrega verificación del dispositivo de datos de entrada para la interfaz de producto punto;
- Se corrigió un error en `TorchModule`;



[v2.17.1] - 2025-8-22
***************************

Agregado
===================

- Se agregaron la interfaz del algoritmo SPSA de gradiente natural cuántico (qnspsa), la máquina de nacimiento de circuitos cuánticos (QBM), la interfaz de gradiente natural cuántico con momentum y un ejemplo puramente cuántico de QGRU.
- Se agregó el backend ``torch_native``.
- Se agregó una interfaz de bit-paralelo para soportar circuitos cuánticos bit-paralelos, y se agregó una función de reordenamiento de bits para reducir el número de intercambios de bits.
- Se agregó el método ``split_groups``.

Cambiado
==================
- Se cambió la implementación de la capa Linear de `:math:`y = Ax + b` a `:math:`y = x@A.T + b`
- Se modificó el parámetro ``obs`` en la interfaz `MeasureAll`.
- Se eliminó la interfaz ``QuantumLayerES``. Se cambiaron los nombres de parámetros de ControllComm a ControlComm en `allgather_group`, `allreduce_group`, `reduce_group`, `broadcast_group` y otras interfaces.
- Se eliminó la interfaz `ncclsplitGroup`.

Corregido
===================
- Se resolvieron problemas de retardo de sincronización con las interfaces de comunicación distribuidas.
- Se modificaron las definiciones de las interfaces de comunicación distribuidas.
- Se resolvió un problema donde los cálculos de gradiente adjunto no soportaban ``PauliX``, ``PauliY`` y ``PauliZ``.


[v2.17.0] - 2025-4-22
***************************

Agregado
===================

- Se agregó la implementación de backend de red tensorial para módulos de circuitos cuánticos, incluyendo soporte para compuertas lógicas básicas, medición y circuitos cuánticos complejos.
- Se agregó la implementación de backend de red tensorial para construir circuitos cuánticos de muchos qubits.
- Se agregó la interfaz QTensor.swapaxes, otro nombre es swapaxis.

Cambiado
===================
- Operaciones matriciales usando openblas.
- Se usa sleef para operaciones SIMD en CPU.
- Se eliminó qnn.MeasurePauliSum.
- Se muestra una advertencia al usar cálculos con backend torch cuando torch está por debajo de la versión 2.4.

Corregido
====================
- Se resolvió el problema de los estados de QMachine al guardar modelos.
- Se resolvió el problema con layernorm y groupnorm cuando ``affine=False``.
- Se resolvió el problema con ``QuantumLayerAdjoint`` en modo eval.

[v2.16.0] - 2025-1-15
***************************

Agregado
===================

- Se agregó una interfaz para el cálculo de circuitos cuánticos usando pyqpanda3.
- La interfaz MeasureAll soporta operadores de Pauli compuestos.
- Se agregaron las interfaces DataParallelVQCAdjointLayer y DataParallelVQCLayer.

Cambiado
===================

- Se eliminaron funciones ONNX obsoletas y la mayoría de las interfaces que integraban pyqpanda, manteniendo algunas interfaces utilizadas en el código de ejemplo.
- Se modificó la interfaz ``VQC_QuantumEmbedding``.
- Al instalar este paquete, ya no se instala pyqpanda; en su lugar, se instala pyqpanda3.
- La interfaz VQC soporta el uso de `x[:,:2]` como parámetros de entrada, que originalmente solo soportaba el formato `x[:,[2]]`.
- Este software soporta Python 3.10.

Corregido
====================
- Se resolvió el problema de fuga de memoria.
- Se resolvió el problema de números aleatorios en GPU.
- Para operaciones relacionadas con reduce, la dimensión máxima soportada de arreglos se incrementó de 8 a 30.
- Se optimizó el código y se mejoró la velocidad de ejecución del código Python en algunos casos.
   
   
[v2.15.0] - 2024-11-19
***************************

Agregado
===================

- Se agregó la interfaz `pyvqnet.backends.set_backend()`. Cuando los usuarios instalan `torch`, se puede usar torch para realizar cálculos matriciales y cálculos de circuitos cuánticos variacionales de QTensor. Para más detalles, consulte el documento :ref:`torch_api`.
- Se agregó `pyvqnet.nn.torch` para heredar la interfaz de red neuronal y la interfaz neuronal de circuitos cuánticos variacionales de `torch.nn.Module`. Para más detalles, consulte el documento :ref:`torch_api`.

Cambiado
===================
- Se modificó la interfaz diag.
- Se modificó la implementación de all_gather para que sea consistente con torch.distributed.all_gather.
- Se modificó `QTensor` para soportar hasta 30 dimensiones de datos.
- Se modificó `mpi4py` requerido para funciones distribuidas para requerir la versión 4.0.1 o superior.

Corregido
===================
- Algunas implementaciones de números aleatorios no podían fijar la semilla debido a OpenMP.
- Se corrigieron algunos errores en el inicio distribuido.


[v2.14.0] - 2024-09-30
***************************

Agregado
===================

- Se agregaron algoritmos de block-encoding: ``VQC_LCU``, ``VQC_FABLE``, ``VQC_QSVT``, y las implementaciones de algoritmo qpanda ``QPANDA_QSVT``, ``QPANDA_LCU``, ``QPANDA_FABLE``.
- Se agregó suma de enteros en bits cuánticos ``vqc_qft_add_to_register``, suma de números en dos bits cuánticos ``vqc_qft_add_two_register`` y multiplicación de números en dos bits cuánticos ``vqc_qft_mul``.
- Se agregó el módulo de entrenamiento híbrido qpanda y vqc ``HybirdVQCQpandaQVMLayer``.
- Se agregaron implementaciones de las interfaces ``einsum``, ``moveaxis``, ``eigh``, ``diagonal`` y otras.
- Se agregaron funciones de cómputo tensorial paralelo en computación distribuida: ``ColumnParallelLinear``, ``RowParallelLinear``.
- Se agregó la función Zero stage-1 en computación distribuida: ``ZeroModelInitial``.
- ``QuantumBatchAsyncQcloudLayer``: cuando ``diff_method == "random_coordinate_descent"``, usa selección aleatoria de parámetros para el cálculo del gradiente en lugar de PSR.

Cambiado
====================
- Se eliminó la parte xtensor.
- La documentación de la API ha sido parcialmente reestructurada. Se distinguieron entre ejemplos de aprendizaje automático cuántico basados en diferenciación automática y aquellos basados en qpanda, y se distinguieron entre interfaces de aprendizaje automático cuántico basadas en diferenciación automática e interfaces de ejemplo basadas en qpanda.
- `matmul` soporta 1d@1d, 2d@1d, 1d@2d.
- Se agregaron algunos alias de capas de computación cuántica: ``QpandaQCircuitVQCLayer`` = ``QuantumLayer``, ``QpandaQCircuitVQCLayerLite`` = ``QuantumLayerV2``, ``QpandaQProgVQCLayer`` = ``QuantumLayerV3``.

Corregido
====================
- Se modificaron las interfaces de comunicación subyacentes ``allreduce``, ``allgather``, ``reduce``, ``broadcast`` en la función de computación distribuida, y se agregó soporte para comunicación de datos ``core.Tensor``.
- Se resolvió el error en la generación de números aleatorios.
- Se resolvió el error en la conversión de ``RXX``, ``RYY``, ``RZZ``, ``RZX`` de VQC a originIR.


[v2.13.0] - 2024-07-30
***************************

Agregado
==================

- Se agregaron las interfaces `no_grad`, `GroupNorm`, `Interpolate`, `contiguous`, `QuantumLayerV3`, `fuse_model`, `SDPA`.
- Se agregó el método Quantum Dropout para evitar el sobreajuste.

Cambiado
===================

- Se agregó la interfaz affine a `BatchNorm`, `LayerNorm`, `GroupNorm`.
- La interfaz `diag` ahora devuelve una salida 1d en la diagonal para entrada 2d, consistente con torch.
- Operaciones como slice y permute intentarán usar el método view para devolver un QTensor en memoria compartida.
- Todas las interfaces soportan entrada no contigua.
- `Adam` soporta el parámetro weight_decay.

Corregido
====================
- Se modificó el error de algunas funciones de descomposición de compuertas lógicas de VQC.
- Se corrigió el problema de fuga de memoria de algunas funciones.
- Se corrigió el problema de que `QuantumLayerMultiProcess` no soportaba entrada GPU.
- Se modificó el método de inicialización de parámetros predeterminado de `Linear`.


[v2.12.0] - 2024-05-01
***************************

Agregado
===================

- Se agregó la interfaz PipelineParallelTrainingWrapper.
- Se agregaron las interfaces `Gelu`, `DropPath`, `binomial`, `adamW`.
- Se agregó `QuantumBatchAsyncQcloudLayer` para soportar el cálculo de simulación de máquina virtual local de pyqpanda.
- Se agregó `QuantumBatchAsyncQcloudLayer` de xtensor para soportar el cálculo de simulación de máquina virtual local de pyqpanda y el cálculo en máquina real.
- Permite que QTensor sea deepcopy y pickle.
- Se agregó el comando de inicio de computación distribuida `vqnetrun`, utilizado al usar la interfaz de computación distribuida.
- Se agregó la interfaz de máquina real del método de cálculo de gradiente ES `QuantumBatchAsyncQcloudLayerES` para soportar cálculos de simulación de VM local, así como cálculos en máquina real para pyqpanda.
- Se agregaron las interfaces de comunicación de datos `allreduce`, `reduce`, `broadcast`, `allgather`, `send`, `recv`, etc., que soportan QTensor en computación distribuida.

Cambiado
===================

- Se agregaron nuevas dependencias "Pillow" y "hjson" al paquete de instalación; se agregaron nuevas dependencias "psutil" y "cloudpickle" en sistemas linux.
- Se optimizó la velocidad de ejecución de softmax y transpose bajo GPU.
- Compilado usando cuda11.8.
- Integración de interfaces de computación distribuida bajo cpu y gpu.

Corregido
===================
- Se redujo el consumo de memoria al iniciar la versión Linux-GPU.
- Se corrigió el problema de fuga de memoria de las funciones select y power.
- Se eliminaron los métodos de actualización de parámetros y gradientes del modelo `nccl_average_parameters_reduce`, `nccl_average_grad_reduce` basados en el método reduce para cpu y gpu.

[v2.11.0] - 2024-03-01
***************************

Agregado
===================

- Se agregaron la nueva API `QNG` (Quantum Natural Gradient) y su demo.
- Se agregaron optimizaciones de circuitos cuánticos, como las API `wrapper_single_qubit_op_fuse`, `wrapper_commute_controlled`, `wrapper_merge_rotations` y sus demos.
- Se agregaron `CY`, `SparseHamiltonian`, `HermitianExpval`.
- Se agregaron `is_csr`, `is_dense`, `dense_to_csr`, `csr_to_dense`.
- Se agregó `QuantumBatchAsyncQcloudLayer` para soportar el cálculo en chips reales de QCloud de pyqpanda, `expval_qcloud`.
- Se agregaron implementaciones de interfaz basadas en NCCL para el entrenamiento de modelos paralelos de datos de computación distribuida multi-GPU en un solo nodo: `nccl_average_parameters_allreduce`, `nccl_average_parameters_reduce`, `nccl_average_grad_allreduce`, `nccl_average_grad_reduce`, y clases para controlar la inicialización de NCCL y operaciones relacionadas `NCCL_api`.
- Se agregó la interfaz de cálculo de gradiente de estrategia de evolución de líneas cuánticas `QuantumLayerES`.

Cambiado
===================

- Se reestructuró el circuito `VQC_CSWAP` a `CSWAP`.
- Se eliminaron documentos antiguos de QNG.
- Se eliminó el parámetro `num_wires` innecesario de `pyvqnet.qnn.vqc` para funciones y clases.
- Se reestructuraron las API `MeasureAll`, `Probability`.
- Se agregó el parámetro qtype a `QuantumMeasure`.

Corregido
===================
- Se cambiaron los slots de `QuantumMeasure` a shots.

[v2.10.0] - 2023-12-30
***************************

Agregado
===========
- Se agregaron nuevas interfaces en pyvqnet.qnn.vqc: IsingXX, IsingXY, IsingYY, IsingZZ, SDG, TDG, PhaseShift, MultiRZ, MultiCnot, MultixCnot, ControlledPhaseShift, SingleExcitation, DoubleExcitation, VQC_AllSinglesDoubles, ExpressiveEntanglingAnsatz, etc.;
- Se agregó la interfaz pyvqnet.qnn.vqc.QuantumLayerAdjoint que soporta el cálculo de gradiente adjunto;
- Se soportó la función de conversión mutua entre originIR y VQC;
- Se soportó información de módulos clásicos y cuánticos en modelos VQC estadísticos;
- Se agregaron dos casos bajo el modelo híbrido de red neuronal clásico-cuántica: modelo de red neuronal convolucional cuántica basado en muestras pequeñas, y modelo de función kernel cuántica para reconocimiento de dígitos manuscritos.


[v2.9.0] - 2023-09-08
***************************

Agregado
===================
- Se agregó la definición de la interfaz xtensor para soportar paralelismo automático de operadores y múltiples backends CPU/GPU. Incluye más de 150 interfaces para matemáticas comunes, lógica y cálculos matriciales para arreglos multidimensionales, así como capas de red neuronal clásicas comunes y optimizadores.

Cambiado
===================
- La versión pasó de v2.0.8 a v2.9.0.
- Los paquetes se cargan en el repositorio PyPI de la compañía, use ``pip install pyvqnet --index-url <pypi_url>``.


[v2.0.8] - 2023-07-26
***************************

Agregado
===================
- Se agregaron interfaces existentes para soportar tipos de cálculo complex128, complex64, double, float, uint8, int8, bool, int16, int32, int64 y otros (gpu).
- Compuertas lógicas básicas basadas en vqc: Hadamard, CNOT, I, RX, RY, PauliZ, PauliX, PauliY, S, RZ, RXX, RYY, RZZ, RZX, X1, Y1, Z1, U1, U2, U3, T, SWAP, P, TOFFOLI, CZ, CR, ISWAP.
- Circuitos cuánticos combinados basados en vqc: VQC_HardwareEfficientAnsatz, VQC_BasicEntanglerTemplate, VQC_StronglyEntanglingTemplate, VQC_QuantumEmbedding, VQC_RotCircuit, VQC_CRotCircuit, VQC_CSWAPcircuit, VQC_Controlled_Hadamard, VQC_CCZ, VQC_FermionicSingleExcitation, VQC_FermionicDoubleExcitation, VQC_UCCSD, VQC_QuantumPoolingCircuit, VQC_BasisEmbedding, VQC_AngleEmbedding, VQC_AmplitudeEmbedding, VQC_IQPEmbedding.
- Métodos de medición basados en vqc: VQC_Purity, VQC_VarMeasure, VQC_DensityMatrixFromQstate, Probability, MeasureAll.


[v2.0.7] - 2023-07-03
***************************

Agregado
===================
- Para redes neuronales clásicas, se agregaron las interfaces kron, gather, scatter, broadcast_to.
- Se agregó soporte para diferentes precisiones de datos: el tipo de dato dtype soporta kbool, kuint8, kint8, kint16, kint32, kint64, kfloat32, kfloat64, kcomplex64, kcomplex128, que representan respectivamente bool, uint8_t, int8_t, int16_t, int32_t, int64_t, float, double, complex<float>, complex<double>.
- Soporta python 3.8, 3.9, 3.10.

Cambiado
===================
- La función init de QTenor y la clase Module agregaron el parámetro `dtype`. Se restringieron los tipos de índice de QTenor y entrada de algunas capas de redes neuronales.
- Red neuronal cuántica: debido a problemas de compatibilidad con MacOS, se eliminaron las interfaces Mnist_Dataset y CIFAR10_Dataset.

[v2.0.6] - 2023-02-22
***************************


Agregado
===================

- Red neuronal clásica, se agregaron las interfaces: multinomial, pixel_shuffle, pixel_unshuffle, se agregó numel para QTensor, se agregó la función de grupo de memoria dinámica de CPU, se agregó la interfaz init_from_tensor para Parameter.
- Red neuronal clásica, se agregaron las interfaces: Dynamic_LSTM, Dynamic_RNN, Dynamic_GRU.
- Red neuronal clásica, se agregaron las interfaces: pad_sequence, pad_packed_sequence, pack_pad_sequence.
- Red neuronal cuántica, se agregaron las interfaces: CCZ, Controlled_Hadamard, FermionicSingleExcitation, UCCSD, QuantumPoolingCircuit.
- Red neuronal cuántica, se agregaron las interfaces: Quantum_Embedding, Mnist_Dataset, CIFAR10_Dataset, grad, Purity.
- Red neuronal cuántica, se agregaron ejemplos: basados en recorte de gradiente, quanvolution, expresividad de circuitos cuánticos, barren plateau y aprendizaje por refuerzo cuántico QDRL.

Cambiado
===================

- Documentación de la API, se reestructuró la estructura de contenido, se agregó el módulo "quantum machine learning research", se cambió "VQNet2ONNX module" a "Other Utility Functions".



corregido
===================

- Red neuronal clásica: se resolvió el problema de que la misma semilla aleatoria produce diferentes distribuciones normales entre plataformas.
- Red neuronal cuántica: se implementó soporte de expval, ProbMeasure, QuantumMeasure para la máquina virtual GPU de QPanda.


[v2.0.5] - 2022-12-25
***************************


Agregado
===================

- Red neuronal clásica, se agregó la implementación de log_softmax, se agregó la función export_model del modelo a ONNX.
- Red neuronal clásica, que soporta la conversión de la mayoría de los módulos de redes neuronales clásicas existentes a ONNX. Para más detalles, consulte el documento de la API "VQNet2ONNX module".
- Red neuronal cuántica, se agregaron interfaces VarMeasure, MeasurePauliSum, Quantum_Embedding, SPSA y otras.
- Red neuronal cuántica, se agregaron LinearGNN, ConvGNN, ConvGNN, QMLP, gradiente natural cuántico, algoritmo cuántico de cambio de parámetros aleatorios, algoritmo DoublySGD, etc.


Cambiado
===================

- Redes neuronales clásicas, se agregaron verificaciones de dimensionalidad para las interfaces BN1d, BN2d.

corregido
==================

- Se resolvió el error de verificación de parámetros de maxpooling.
- Se resolvió el error de slice [::-1].


[v2.0.4] - 2022-09-20
***************************


Agregado
==================

- Red neuronal clásica, se agregó la implementación de LayernormNd, soportando cálculo de layernorm para datos multidimensionales.
- Red neuronal clásica, se agregaron las interfaces de cálculo de función de pérdida CrossEntropyLoss y NLL_Loss, soportando entrada de 1 a N dimensiones.
- Red neuronal cuántica, se agregaron plantillas de circuitos comunes: HardwareEfficientAnsatz, StronglyEntanglingTemplate, BasicEntanglerTemplate.
- Red neuronal cuántica, se agregó la interfaz Mutal_info para calcular la información mutua de subsistemas de qubits, la entropía de Von Neumann VB_Entropy y la matriz de densidad DensityMatrixFromQstate.
- Red neuronal cuántica, se agregó el ejemplo de algoritmo de perceptrón cuántico QuantumNeuron, se agregó el ejemplo de algoritmo de serie de Fourier cuántica.
- Red neuronal cuántica, se agregó la interfaz QuantumLayerMultiProcess que soporta operación acelerada de circuitos cuánticos con múltiples procesos.

Cambiado
==================

- Red neuronal clásica, soporta el parámetro de grupo de convolución group, dilation_rate de convolución dilatada y padding de valor arbitrario como parámetros para la convolución unidimensional Conv1d, la convolución bidimensional Conv2d y la deconvolución ConvT2d.
- Se omitió la operación de broadcast para datos en la misma dimensión, reduciendo la lógica de ejecución innecesaria.

corregido
==================

- Se resolvió el problema donde la función stack calculaba incorrectamente bajo ciertos parámetros.


[v2.0.3] - 2022-07-15
***************************


Agregado
==================

- Se agregó soporte para stack, interfaz de red neuronal recurrente bidireccional: RNN, LSTM, GRU.
- Se agregaron interfaces para indicadores de rendimiento de cálculo comunes: MSE, RMSE, MAE, R_Square, precision_recall_f1_2_score, precision_recall_f1_Multi_score, precision_recall_f1_N_score, auc_calculate.
- Se agregó el ejemplo de algoritmo de kernel SVM cuántico.

Cambiado
==================

- Se aceleró la velocidad de impresión cuando hay demasiados datos en QTensor.
- Se usó openmp para acelerar cálculos en Windows y Linux.

corregido
==================

- Se resolvió el problema donde algunos métodos de importación de Python fallaban al importar.
- Se resolvió el problema de cálculos repetidos de la capa de normalización por lotes (BN).
- Se resolvió el error donde las interfaces ``QTensor.reshape`` y ``transpose`` no podían calcular gradientes.
- Se agregó validación de forma de los parámetros de entrada para la interfaz ``tensor.power``.

[v2.0.2] - 2022-05-15
***************************


Agregado
==================

- Se agregó topK, argtoK.
- Se agregó cumsum.
- Se agregó masked_fill.
- Se agregó triu, tril.
- Se agregaron ejemplos de distribución aleatoria generada por QGAN.

Cambiado
==================

- Soporta índice slice avanzado e índice slice común.
- matmul soporta operaciones tensoriales 3D y 4D.
- Se modificó la implementación de la función HardSigmoid.

corregido
==================

- Se resolvió el problema donde la convolución, la normalización por lotes, la deconvolución, las capas de pooling y otras capas no almacenaban en caché las variables internas, causando problemas de cálculo de gradiente durante múltiples retropropagaciones después de una propagación forward.
- Se corrigió la implementación y el ejemplo de la capa QLinear.
- Se resolvió el problema donde Image fallaba al cargar al importar VQNet en un entorno conda en macOS.



[v2.0.1] - 2022-03-30
**************************


Agregado
==================

- Se agregaron más de 100 interfaces básicas de estructura de datos QTensor, incluyendo funciones de creación, funciones lógicas, funciones matemáticas y operaciones matriciales.
- Se agregaron 14 funciones básicas de redes neuronales, incluyendo convolución, deconvolución, pooling, etc.
- Se agregaron 4 funciones de pérdida, incluyendo MSE, BCE, CCE, SCE, etc.
- Se agregaron 10 funciones de activación, incluyendo ReLu, Sigmoid, ELU, etc.
- Se agregaron 6 optimizadores, incluyendo SGD, RMSPROP, ADAM, etc.
- Se agregaron ejemplos de aprendizaje automático: QVC, QDRL, Q-KMEANS, QUnet, HQCNN, VSQL, Quantum Autoencoder.
- Se agregaron capas de aprendizaje automático cuántico: QuantumLayer, NoiseQuantumLayer.
