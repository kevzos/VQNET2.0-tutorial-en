VQNet Änderungsprotokoll
###############################


[v2.18.0] - 2026-04-22
***************************

Hinzugefügt
===================
- ``vqnetrun`` unterstützt jetzt den ``--backend nccl`` Modus, mit verteiltem Start über die Parameter ``--nproc_per_node``, ``--nccl_socket_ifname``.
- ``VQCQCloudLayer``-Schnittstelle hinzugefügt zum Übermitteln von VQC-Modulen an echte QCloud-Chips oder lokale pyqpanda3-Simulatoren, mit Unterstützung für parameter_shift-Rückpropagation.
- ``CommController`` fügt eine ``destroy()``-Methode zur Bereinigung von NCCL-Kommunikationsressourcen hinzu.

Geändert
===================
- Standard-Backend auf ``pyvqnet-ad`` geändert.
- Veraltete Schnittstellen ``QuantumLayerMultiProcess``, ``DataParallelHybirdVQCQpandaQVMLayer`` und ``HybirdVQCQpanda3QVMLayer`` entfernt.
- ``split_group`` in ``split_groups`` umbenannt.
- Abhängigkeit von NVIDIA-Laufzeitumgebung für CUDA 12.6.
- Standardwert von "chip_id" auf "WK_C180" geändert.
- ``ComplexEntangelingTemplate`` wurde in ``ComplexEntanglingTemplate`` umbenannt.
- ``vqc.rst``: Benchmark-Abschnitt "Test 2: 10-Qubit VQC Gradient Comparison" hinzugefügt, der VQNet / TorchQuantum / DeepQuantum / Pennylane / MindQuantum vergleicht.
- Benchmark-Spezifikationstabelle mit CUDA 12.6, torchquantum 0.2.0 und mindquantum 0.12.0 aktualisiert.

Behoben
===================
- Integer-Überlaufproblem im ``roll`` CUDA-Kernel behoben.
- Unterstützung von ``cuda_masked_fill`` für complex64/complex128-Typen behoben.
- Fehler in der ``log_softmax``-Vorwärtsberechnung behoben, die unter ``bfloat16`` falsche +inf-Werte erzeugte.
- Gerätefehler aufgrund fehlender ``CUDAGuard`` bei GPU-übergreifendem Speicherzugriff behoben.
- Viele Tippfehler behoben.


[v2.17.3] - 2026-03-31
***************************

Hinzugefügt
===================
- bfloat16-Datentyp hinzugefügt.
- Asynchrone NCCL-Kommunikationsschnittstellen hinzugefügt: ``nccl_async_all_gather``, ``nccl_async_all_reduce``, ``nccl_async_reduce``, ``nccl_async_broadcast``, ``nccl_async_send`` und ``nccl_async_recv``.
- Unterstützung für den neuesten Origin-Quantum-Chip mit Chip-ID ``WK_C180`` hinzugefügt.
- ``data_ptr`` und weitere Schnittstellen hinzugefügt, experimentelle Unterstützung für `triton <https://triton-lang.org/main/index.html>`_ hinzugefügt.

Geändert
===================

- Standard-Backend auf "pyvqnet-ad" geändert.
- Berechnung auf MacOS basiert auf arm-neon-Instruktionen.
- Die ``matmul``-Schnittstelle unterstützt Daten mit mehr als 4 Dimensionen.
- Reduzierte Abhängigkeit von einigen CUDA-Laufzeitbibliotheken bei Installation des whl-Pakets.
- Der QTensor-Datentyp in pyvqnet ist kein Integer mehr, sondern ein spezifischer Datentyp.
- Pickle-Logik von QTensor geändert, pickelt keine Gradienten mehr.
- is_dense entfernt.
- ``QuantumBatchAsyncQcloudLayer`` von pq2 entfernt.
- Dokumentation für ``QuantumBatchAsyncQcloudLayer`` von pq3 aktualisiert.

Behoben
===================
- Fehler von ``Linear`` bei ``use_bias=False`` behoben.
- ``MAX_GPUS``-Problem behoben; die maximale Anzahl von GPUs beträgt jetzt 16.
- Importfehler auf Windows Jupyter behoben.


[v2.17.2] -2025-11-18
***********************************


Hinzugefügt
===================

- Unterstützung für das `torch`-Backend über das Quanten-Natural-Gradient-QNG-Interface hinzugefügt.
- Das `pyvqnet-ad`-Backend hinzugefügt, das ein C++-Backend für automatische Differentiation ähnlich wie torch verwendet. Die Datenstruktur verwendet weiterhin den ursprünglichen ``_core.Tensor`` und unterstützt die überwiegende Mehrheit der vorhandenen Schnittstellen.
- Dokumentation "Benchmarking of Variational Quantum Circuit's Gradients for Batch Data" hinzugefügt.

Geändert
===================

- `HybirdVQCQpanda3QVMLayer`, `QuantumLayerMultiProcess`, `TorchHybirdVQCQpanda3QVMLayer` entfernt;
- `is_csr`, `csr_members`, `SparseHamiltonian`, `csr_to_dense`, `dense_to_csr` entfernt;
- Die Schnittstellen `QiskitLayer` und `CirqLayer` hinzugefügt.
- Unterstützung für `if_print_qcloud_log` in der Schicht `QuantumBatchAsyncQcloudLayer` zum Drucken von QCloud-Protokollen hinzugefügt.
- Installationsbefehl auf `pip install pyvqnet --upgrade` geändert.
- Die unterstützte Python-Version ist `Python 3.10`.
- Installationsbefehl für mpicxx angepasst.



Behoben
===================

- Rückgabewerte der neuesten Version von pyqpanda2QCloud unterstützt;
- Eingabedaten-Geräteprüfungen für die Punktproduktschnittstelle hinzugefügt;
- Fehler in `TorchModule` behoben;



[v2.17.1] - 2025-8-22
***************************

Hinzugefügt
===================

- Schnittstellen für den Quanten-Natural-Gradient-SPSA-Algorithmus (qnspsa), Quantenschaltungs-Born-Maschine (QBM), Quanten-Natural-Gradient mit Impuls und ein reines Quanten-QGRU-Beispiel hinzugefügt.
- Das ``torch_native``-Backend hinzugefügt.
- Eine Bit-parallele Schnittstelle zur Unterstützung bit-paralleler Quantenschaltungen hinzugefügt, sowie eine Bit-Neuordnungsfunktion zur Reduzierung der Anzahl von Bit-Vertauschungen.
- Die ``split_groups``-Methode hinzugefügt.

Geändert
==================
- Die Implementierung der Linear-Schicht von `:math:`y = Ax + b`` auf `:math:`y = x@A.T + b`` geändert.
- Den ``obs``-Parameter in der `MeasureAll`-Schnittstelle geändert.
- Die ``QuantumLayerES``-Schnittstelle entfernt. Parameternamen in `allgather_group`, `allreduce_group`, `reduce_group`, `broadcast_group` und anderen Schnittstellen von ControllComm auf ControlComm geändert.
- Die `ncclsplitGroup`-Schnittstelle entfernt.

Behoben
===================
- Synchronisationsverzögerungsprobleme bei verteilten Kommunikationsschnittstellen behoben.
- Definitionen verteilter Kommunikationsschnittstellen angepasst.
- Problem behoben, bei dem adjungierte Gradientenberechnungen ``PauliX``, ``PauliY`` und ``PauliZ`` nicht unterstützten.


[v2.17.0] - 2025-4-22
***************************

Hinzugefügt
===================

- Tensor-Netzwerk-Backend-Implementierung für Quantenschaltungsmodule hinzugefügt, einschließlich Unterstützung für basische Logikgatter, Messung und komplexe Quantenschaltungen.
- Tensor-Netzwerk-Backend-Implementierung zur Konstruktion von Quantenschaltungen mit vielen Qubits hinzugefügt.
- QTensor.swapaxes-Schnittstelle hinzugefügt, alternativer Name ist swapaxis.

Geändert
===================
- Matrixoperationen verwenden openblas.
- Verwendung von sleef für CPU-SIMD-Operationen.
- qnn.MeasurePauliSum entfernt.
- Warnung ausgeben bei Verwendung des torch-Backends mit torch-Version unter 2.4.

Behoben
====================
- Problem mit QMachine-Zuständen beim Speichern von Modellen behoben.
- Problem mit Layernorm und Groupnorm bei ``affine=False`` behoben.
- Problem mit ``QuantumLayerAdjoint`` im Evaluierungsmodus behoben.

[v2.16.0] - 2025-1-15
***************************

Hinzugefügt
===================

- Schnittstelle zur Quantenschaltungsberechnung mit pyqpanda3 hinzugefügt.
- Die MeasureAll-Schnittstelle unterstützt zusammengesetzte Pauli-Operatoren.
- DataParallelVQCAdjointLayer- und DataParallelVQCLayer-Schnittstellen hinzugefügt.

Geändert
===================

- Veraltete ONNX-Funktionen und die meisten Schnittstellen, die pyqpanda integrieren, entfernt, während einige im Beispielcode verwendete Schnittstellen beibehalten wurden.
- Die ``VQC_QuantumEmbedding``-Schnittstelle geändert.
- Bei Installation dieses Pakets wird nicht mehr pyqpanda, sondern pyqpanda3 installiert.
- Die VQC-Schnittstelle unterstützt die Verwendung von `x[:,:2]` als Eingabeparameter, die ursprünglich nur das Format `x[:,[2]]` unterstützte.
- Diese Software unterstützt Python 3.10.

Behoben
====================
- Speicherleckproblem behoben.
- GPU-Zufallszahlenproblem behoben.
- Die maximale unterstützte Array-Dimension für Reduktionsoperationen wurde von 8 auf 30 erhöht.
- Code optimiert und Python-Code-Ausführungsgeschwindigkeit in einigen Fällen verbessert.

  
[v2.15.0] - 2024-11-19
***************************

Hinzugefügt
===================

- `pyvqnet.backends.set_backend()`-Schnittstelle hinzugefügt. Wenn Benutzer `torch` installieren, kann torch zur Durchführung von Matrixberechnungen und Variations-Quantenschaltungsberechnungen von QTensor verwendet werden. Siehe Dokument :ref:`torch_api`.
- `pyvqnet.nn.torch` hinzugefügt, um die neuronalen Netzwerkschnittstellen und die Variations-Quantenschaltungs-Neuronalnetz-Schnittstelle von `torch.nn.Module` zu erben. Siehe Dokument :ref:`torch_api`.

Geändert
===================
- diag-Schnittstelle geändert.
- all_gather-Implementierung geändert, um mit torch.distributed.all_gather konsistent zu sein.
- `QTensor` geändert, um bis zu 30-dimensionale Daten zu unterstützen.
- Für verteilte Funktionen benötigtes `mpi4py` geändert, erfordert jetzt Version 4.0.1 oder höher.

Behoben
===================
- Einige Zufallszahlenimplementierungen konnten aufgrund von OpenMP den Seed nicht fixieren.
- Einige Fehler beim verteilten Start behoben.


[v2.14.0] - 2024-09-30
***************************

Hinzugefügt
===================

- Block-Encoding-Algorithmen hinzugefügt: ``VQC_LCU``, ``VQC_FABLE``, ``VQC_QSVT`` und qpanda-Algorithmusimplementierungen ``QPANDA_QSVT``, ``QPANDA_LCU``, ``QPANDA_FABLE``.
- Ganzzahladdition auf Quantenbits ``vqc_qft_add_to_register``, Addition von Zahlen auf zwei Quantenbits ``vqc_qft_add_two_register`` und Multiplikation von Zahlen auf zwei Quantenbits ``vqc_qft_mul`` hinzugefügt.
- Hybrides qpanda- und vqc-Trainingsmodul ``HybirdVQCQpandaQVMLayer`` hinzugefügt.
- ``einsum``, ``moveaxis``, ``eigh``, ``diagonal`` und weitere Schnittstellenimplementierungen hinzugefügt.
- Tensor-parallele Rechenfunktionen in verteiltem Rechnen hinzugefügt: ``ColumnParallelLinear``, ``RowParallelLinear``.
- Zero-Stage-1-Funktion in verteiltem Rechnen hinzugefügt: ``ZeroModelInitial``.
- ``QuantumBatchAsyncQcloudLayer``: verwendet bei ``diff_method == "random_coordinate_descent"`` zufällige Parameterauswahl für die Gradientenberechnung anstelle von PSR.

Geändert
====================
- Der xtensor-Teil wurde gelöscht.
- Die API-Dokumentation wurde teilweise umstrukturiert. Es wurde zwischen Quantenmaschinenlernbeispielen basierend auf automatischer Differentiation und solchen basierend auf qpanda unterschieden, sowie zwischen Quantenmaschinenlern-Schnittstellen basierend auf automatischer Differentiation und Beispielschnittstellen basierend auf qpanda.
- `matmul` unterstützt 1d@1d, 2d@1d, 1d@2d.
- Einige Quantenberechnungsschicht-Aliasnamen hinzugefügt: ``QpandaQCircuitVQCLayer`` = ``QuantumLayer``, ``QpandaQCircuitVQCLayerLite`` = ``QuantumLayerV2``, ``QpandaQProgVQCLayer`` = ``QuantumLayerV3``.

Behoben
====================
- Die zugrundeliegenden Kommunikationsschnittstellen ``allreduce``, ``allgather``, ``reduce``, ``broadcast`` in der verteilten Rechenfunktion geändert und Unterstützung für ``core.Tensor``-Datenkommunikation hinzugefügt.
- Fehler bei der Zufallszahlengenerierung behoben.
- Fehler bei der Konvertierung von VQCs ``RXX``, ``RYY``, ``RZZ``, ``RZX`` in originIR behoben.


[v2.13.0] - 2024-07-30
***************************

Hinzugefügt
==================

- `no_grad`, `GroupNorm`, `Interpolate`, `contiguous`, `QuantumLayerV3`, `fuse_model`, `SDPA`-Schnittstellen hinzugefügt.
- Quantum-Dropout-Methode zur Vermeidung von Überanpassung hinzugefügt.

Geändert
===================

- Affine-Schnittstelle zu `BatchNorm`, `LayerNorm`, `GroupNorm` hinzugefügt.
- Die `diag`-Schnittstelle gibt jetzt für 2D-Eingaben eine 1D-Ausgabe auf der Diagonalen zurück, konsistent mit torch.
- Operationen wie slice und permute versuchen jetzt, die View-Methode zu verwenden, um einen QTensor im gemeinsamen Speicher zurückzugeben.
- Alle Schnittstellen unterstützen nicht-kontiguierliche Eingaben.
- `Adam` unterstützt den weight_decay-Parameter.

Behoben
====================
- Fehler bei einigen Logikgatter-Zerlegungsfunktionen von VQC behoben.
- Speicherleckproblem einiger Funktionen behoben.
- Problem behoben, dass `QuantumLayerMultiProcess` keine GPU-Eingaben unterstützt.
- Standard-Parameterinitialisierungsmethode von `Linear` geändert.


[v2.12.0] - 2024-05-01
***************************

Hinzugefügt
===================

- PipelineParallelTrainingWrapper-Schnittstelle hinzugefügt.
- `Gelu`, `DropPath`, `binomial`, `adamW`-Schnittstellen hinzugefügt.
- `QuantumBatchAsyncQcloudLayer` hinzugefügt zur Unterstützung lokaler virtueller Maschinen-Simulationsberechnungen von pyqpanda.
- xtensors `QuantumBatchAsyncQcloudLayer` hinzugefügt zur Unterstützung lokaler virtueller Maschinen-Simulationsberechnungen und echter Maschinenberechnungen von pyqpanda.
- Ermöglicht Deepcopy und Pickle von QTensor.
- Verteilten Startbefehl `vqnetrun` hinzugefügt, der bei Verwendung der verteilten Rechensschnittstelle verwendet wird.
- ES-Gradientenberechnungsmethode für echte Maschinenschnittstelle `QuantumBatchAsyncQcloudLayerES` hinzugefügt, die sowohl lokale VM-Simulationsberechnungen als auch echte Maschinenberechnungen für pyqpanda unterstützt.
- Datenkommunikationsschnittstellen `allreduce`, `reduce`, `broadcast`, `allgather`, `send`, `recv` usw. hinzugefügt, die QTensor in verteiltem Rechnen unterstützen.

Geändert
===================

- Neue Abhängigkeiten "Pillow" und "hjson" zum Installationspaket hinzugefügt, neue Abhängigkeiten "psutil" und "cloudpickle" auf Linux-Systemen hinzugefügt.
- Softmax- und Transpose-Ausführungsgeschwindigkeit unter GPU optimiert.
- Kompiliert mit CUDA 11.8.
- Integration verteilter Rechensschnittstellen unter CPU und GPU.

Behoben
===================
- Speicherverbrauch beim Starten der Linux-GPU-Version reduziert.
- Speicherleckproblem der select- und power-Funktionen behoben.
- Modellparameter- und Gradientenaktualisierungsmethoden `nccl_average_parameters_reduce`, `nccl_average_grad_reduce` basierend auf der reduce-Methode für CPU und GPU entfernt.

[v2.11.0] - 2024-03-01
***************************

Hinzugefügt
===================

- Neue `QNG`-API (Quantum Natural Gradient) und Demo hinzugefügt.
- Quantenschaltungsoptimierung hinzugefügt, wie `wrapper_single_qubit_op_fuse`, `wrapper_commute_controlled`, `wrapper_merge_rotations`-API und Demo.
- `CY`, `SparseHamiltonian`, `HermitianExpval` hinzugefügt.
- `is_csr`, `is_dense`, `dense_to_csr`, `csr_to_dense` hinzugefügt.
- `QuantumBatchAsyncQcloudLayer` zur Unterstützung von pyqpandas QCloud-Echtchip-Berechnung hinzugefügt, `expval_qcloud`.
- NCCL-basierte Schnittstellenimplementierungen für paralleles Modelltraining von Multi-GPU-verteilten Rechendaten auf einem einzelnen Knoten hinzugefügt: `nccl_average_parameters_allreduce`, `nccl_average_parameters_reduce`, `nccl_average_grad_allreduce`, `nccl_average_grad_reduce`, sowie Klassen zur Steuerung der NCCL-Initialisierung und zugehöriger Operationen `NCCL_api`.
- Evolutionsstrategie-Gradientenberechnungsschnittstelle für Quantenschaltungen `QuantumLayerES` hinzugefügt.

Geändert
===================

- `VQC_CSWAP`-Schaltung zu `CSWAP` umgestaltet.
- Alte QNG-Dokumente gelöscht.
- Unnötigen `num_wires`-Parameter aus `pyvqnet.qnn.vqc` für Funktionen und Klassen entfernt.
- `MeasureAll`, `Probability`-API umgestaltet.
- qtype-Parameter zu `QuantumMeasure` hinzugefügt.

Behoben
===================
- `QuantumMeasure`'s slots in shots geändert.

[v2.10.0] - 2023-12-30
***************************

Hinzugefügt
===========
- Neue Schnittstellen unter pyvqnet.qnn.vqc hinzugefügt: IsingXX, IsingXY, IsingYY, IsingZZ, SDG, TDG, PhaseShift, MultiRZ, MultiCnot, MultixCnot, ControlledPhaseShift, SingleExcitation, DoubleExcitation, VQC_AllSinglesDoubles, ExpressiveEntanglingAnsatz usw.;
- pyvqnet.qnn.vqc.QuantumLayerAdjoint-Schnittstelle hinzugefügt, die adjungierte Gradientenberechnung unterstützt;
- Funktion zur gegenseitigen Konvertierung zwischen originIR und VQC unterstützt;
- Klassische und Quantenmodulinformationen in statistischen VQC-Modellen unterstützt;
- Zwei Fälle unter dem hybriden quanten-klassischen neuronalen Netzwerkmodell hinzugefügt: quantenfaltungsbasiertes neuronales Netzwerkmodell basierend auf kleinen Stichproben und Quantenkernfunktionsmodell für handschriftliche Ziffernerkennung.


[v2.9.0] - 2023-09-08
***************************

Hinzugefügt
===================
- Die xtensor-Schnittstellendefinition wurde hinzugefügt, um automatischen Operator-Parallelismus und mehrere CPU/GPU-Backends zu unterstützen. Sie umfasst mehr als 150 Schnittstellen für häufig verwendete mathematische, logische und Matrixberechnungen für mehrdimensionale Arrays sowie gängige klassische neuronale Netzwerkschichten und Optimierer.

Geändert
===================
- Version von v2.0.8 auf v2.9.0 erhöht.
- Pakete werden im unternehmenseigenen PyPI-Repository hochgeladen, Verwendung von ``pip install pyvqnet --index-url <pypi_url>``.


[v2.0.8] - 2023-07-26
***************************

Hinzugefügt
===================
- Vorhandene Schnittstellen zur Unterstützung von Berechnungen (GPU) mit complex128, complex64, double, float, uint8, int8, bool, int16, int32, int64 und anderen Typen hinzugefügt.
- Basislogikgatter basierend auf vqc: Hadamard, CNOT, I, RX, RY, PauliZ, PauliX, PauliY, S, RZ, RXX, RYY, RZZ, RZX, X1, Y1, Z1, U1, U2, U3, T, SWAP, P, TOFFOLI, CZ, CR, ISWAP.
- Kombinierte Quantenschaltungen basierend auf vqc: VQC_HardwareEfficientAnsatz, VQC_BasicEntanglerTemplate, VQC_StronglyEntanglingTemplate, VQC_QuantumEmbedding, VQC_RotCircuit, VQC_CRotCircuit, VQC_CSWAPcircuit, VQC_Controlled_Hadamard, VQC_CCZ, VQC_FermionicSingleExcitation, VQC_FermionicDoubleExcitation, VQC_UCCSD, VQC_QuantumPoolingCircuit, VQC_BasisEmbedding, VQC_AngleEmbedding, VQC_AmplitudeEmbedding, VQC_IQPEmbedding.
- Messmethoden basierend auf vqc: VQC_Purity, VQC_VarMeasure, VQC_DensityMatrixFromQstate, Probability, MeasureAll.


[v2.0.7] - 2023-07-03
***************************

Hinzugefügt
===================
- Für klassische neuronale Netzwerke wurden kron, gather, scatter, broadcast_to-Schnittstellen hinzugefügt.
- Unterstützung für verschiedene Datenpräzisionen hinzugefügt: Datentyp dtype unterstützt kbool, kuint8, kint8, kint16, kint32, kint64, kfloat32, kfloat64, kcomplex64, kcomplex128, die jeweils bool, uint8_t, int8_t, int16_t, int32_t, int64_t, float, double, complex<float>, complex<double> darstellen.
- Unterstützt Python 3.8, 3.9, 3.10.

Geändert
===================
- Die init-Funktion der QTensor- und Module-Klasse fügt den `dtype`-Parameter hinzu. Die Typen des QTensor-Index und der Eingabe einiger neuronaler Netzwerkschichten sind eingeschränkt.
- Quanten-neuronales Netzwerk: Aufgrund von MacOS-Kompatibilitätsproblemen wurden die Mnist_Dataset- und CIFAR10_Dataset-Schnittstellen entfernt.

[v2.0.6] - 2023-02-22
***************************


Hinzugefügt
===================

- Klassisches neuronales Netzwerk, Schnittstellen hinzugefügt: multinomial, pixel_shuffle, pixel_unshuffle, numel für QTensor hinzugefügt, CPU-Dynamikspeicher-Pool-Funktion hinzugefügt, init_from_tensor-Schnittstelle für Parameter hinzugefügt.
- Klassisches neuronales Netzwerk, Schnittstellen hinzugefügt: Dynamic_LSTM, Dynamic_RNN, Dynamic_GRU.
- Klassisches neuronales Netzwerk, Schnittstellen hinzugefügt: pad_sequence, pad_packed_sequence, pack_pad_sequence.
- Quanten-neuronales Netzwerk, Schnittstellen hinzugefügt: CCZ, Controlled_Hadamard, FermionicSingleExcitation, UCCSD, QuantumPoolingCircuit.
- Quanten-neuronales Netzwerk, Schnittstellen hinzugefügt: Quantum_Embedding, Mnist_Dataset, CIFAR10_Dataset, grad, Purity.
- Quanten-neuronales Netzwerk, Beispiele hinzugefügt: basierend auf Gradienten-Clipping, Quanvolution, Quantenschaltungs-Ausdrucksfähigkeit, Barren-Plateau und quantenverstärkendem Lernen QDRL.

Geändert
===================

- API-Dokumentation, Inhaltsstruktur umgestaltet, Modul "Quantum Machine Learning Research" hinzugefügt, "VQNet2ONNX module" in "Other Utility Functions" geändert.



behoben
===================

- Klassisches neuronales Netzwerk: Problem behoben, dass derselbe Zufallsseed auf verschiedenen Plattformen unterschiedliche Normalverteilungen erzeugt.
- Quanten-neuronales Netzwerk: expval, ProbMeasure, QuantumMeasure-Unterstützung für QPanda-GPU-VM implementiert.


[v2.0.5] - 2022-12-25
***************************


Hinzugefügt
===================

- Klassisches neuronales Netzwerk, log_softmax-Implementierung hinzugefügt, die export_model-Funktion des Modells zu ONNX hinzugefügt.
- Klassisches neuronales Netzwerk, das die Konvertierung der meisten vorhandenen klassischen neuronalen Netzwerkmodule zu ONNX unterstützt. Details siehe API-Dokument "VQNet2ONNX module".
- Quanten-neuronales Netzwerk, VarMeasure, MeasurePauliSum, Quantum_Embedding, SPSA und andere Schnittstellen hinzugefügt.
- Quanten-neuronales Netzwerk, LinearGNN, ConvGNN, QMLP, Quanten-Natural-Gradient, Quanten-Zufalls-Parameter-Shift-Algorithmus, DoublySGD-Algorithmus usw. hinzugefügt.


Geändert
===================

- Klassische neuronale Netzwerke, Dimensionsprüfungen für BN1d-, BN2d-Schnittstellen hinzugefügt.

behoben
==================

- Fehler bei der Maxpooling-Parameterprüfung behoben.
- [::-1]-Slice-Fehler behoben.


[v2.0.4] - 2022-09-20
***************************


Hinzugefügt
==================

- Klassisches neuronales Netzwerk, LayernormNd-Implementierung hinzugefügt, unterstützt mehrdimensionale Daten-Layernorm-Berechnung.
- Klassisches neuronales Netzwerk, CrossEntropyLoss- und NLL_Loss-Verlustfunktionsberechnungsschnittstellen hinzugefügt, unterstützt 1-dimensionale bis N-dimensionale Eingabe.
- Quanten-neuronales Netzwerk, gängige Schaltungsvorlagen hinzugefügt: HardwareEfficientAnsatz, StronglyEntanglingTemplate, BasicEntanglerTemplate.
- Quanten-neuronales Netzwerk, Mutal_info-Schnittstelle zur Berechnung der gegenseitigen Information von Qubit-Subsystemen, Von-Neumann-Entropie VB_Entropy und Dichtematrix DensityMatrixFromQstate hinzugefügt.
- Quanten-neuronales Netzwerk, Quantenperzeptron-Algorithmusbeispiel QuantumNeuron hinzugefügt, Quanten-Fourier-Reihen-Algorithmusbeispiel hinzugefügt.
- Quanten-neuronales Netzwerk, Schnittstelle QuantumLayerMultiProcess hinzugefügt, die Mehrprozess-beschleunigte Operation von Quantenschaltungen unterstützt.

Geändert
==================

- Klassisches neuronales Netzwerk, unterstützt Gruppenfaltungsparameter group, Dilationsrate der dilatierten Faltung und beliebige Auffüllung als Parameter für eindimensionale Faltung Conv1d, zweidimensionale Faltung Conv2d und Entfaltung ConvT2d.
- Broadcast-Operation für Daten in derselben Dimension übersprungen, wodurch unnötige Ausführungslogik reduziert wird.

behoben
==================

- Problem behoben, bei dem die stack-Funktion unter bestimmten Parametern falsch berechnete.


[v2.0.3] - 2022-07-15
***************************


Hinzugefügt
==================

- Unterstützung für stack, bidirektionale rekurrente neuronale Netzwerkschnittstellen hinzugefügt: RNN, LSTM, GRU.
- Schnittstellen für gängige Berechnungsleistungsindikatoren hinzugefügt: MSE, RMSE, MAE, R_Square, precision_recall_f1_2_score, precision_recall_f1_Multi_score, precision_recall_f1_N_score, auc_calculate.
- Algorithmusbeispiel für Quanten-Kernel-SVM hinzugefügt.

Geändert
==================

- Druckgeschwindigkeit bei zu vielen QTensor-Daten beschleunigt.
- Openmp zur Beschleunigung von Berechnungen unter Windows und Linux verwendet.

behoben
==================

- Problem behoben, bei dem einige Python-Importmethoden nicht importiert werden konnten.
- Problem wiederholter Batch-Normalisierungs(BN)-Schichtberechnungen behoben.
- Fehler behoben, bei dem die ``QTensor.reshape``- und ``transpose``-Schnittstellen keine Gradienten berechnen konnten.
- Eingabeparameter-Formvalidierung für die ``tensor.power``-Schnittstelle hinzugefügt.

[v2.0.2] - 2022-05-15
***************************


Hinzugefügt
==================

- topK, argtoK hinzugefügt.
- cumsum hinzugefügt.
- masked_fill hinzugefügt.
- triu, tril hinzugefügt.
- Beispiele für von QGAN erzeugte Zufallsverteilung hinzugefügt.

Geändert
==================

- Unterstützt erweiterte Slice-Indizes und gängige Slice-Indizes.
- matmul unterstützt 3D-, 4D-Tensor-Operationen.
- HardSigmoid-Funktionsimplementierung geändert.

behoben
==================

- Problem behoben, bei dem Faltungs-, Batch-Normalisierungs-, Entfaltungs-, Pooling-Schichten und andere Schichten keine internen Variablen zwischenspeicherten, was zu Gradientenberechnungsproblemen bei mehreren Rückwärtsdurchläufen nach einem Vorwärtsdurchlauf führte.
- Implementierung und Beispiel der QLinear-Schicht behoben.
- Problem behoben, bei dem das Laden von Bildern beim Importieren von VQNet in einer Conda-Umgebung auf macOS fehlschlug.




[v2.0.1] - 2022-03-30
**************************


Hinzugefügt
==================

- Mehr als 100 grundlegende QTensor-Datenstruktur-Schnittstellen hinzugefügt, darunter Erstellungsfunktionen, logische Funktionen, mathematische Funktionen und Matrixoperationen.
- 14 grundlegende neuronale Netzwerkfunktionen hinzugefügt, darunter Faltung, Entfaltung, Pooling usw.
- 4 Verlustfunktionen hinzugefügt, darunter MSE, BCE, CCE, SCE usw.
- 10 Aktivierungsfunktionen hinzugefügt, darunter ReLu, Sigmoid, ELU usw.
- 6 Optimierer hinzugefügt, darunter SGD, RMSPROP, ADAM usw.
- Beispiele für maschinelles Lernen hinzugefügt: QVC, QDRL, Q-KMEANS, QUnet, HQCNN, VSQL, Quantum Autoencoder.
- Quanten-Maschinenlern-Schichten hinzugefügt: QuantumLayer, NoiseQuantumLayer.
