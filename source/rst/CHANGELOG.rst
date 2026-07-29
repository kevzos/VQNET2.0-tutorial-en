Journal des modifications de VQNet
############################################


[v2.18.0] - 2026-04-22
***************************

Ajouté
===================
- ``vqnetrun`` prend désormais en charge le mode ``--backend nccl``, avec un démarrage distribué contrôlable via les paramètres ``--nproc_per_node``, ``--nccl_socket_ifname``.
- Ajout de l'interface ``VQCQCloudLayer`` pour soumettre un module VQC aux puces réelles QCloud ou aux simulateurs locaux pyqpanda3, avec prise en charge de la rétropropagation parameter_shift.
- ``CommController`` ajoute une méthode ``destroy()`` pour le nettoyage des ressources de communication NCCL.

Modifié
===================
- Le backend par défaut est passé à ``pyvqnet-ad``.
- Suppression des interfaces obsolètes ``QuantumLayerMultiProcess``, ``DataParallelHybirdVQCQpandaQVMLayer`` et ``HybirdVQCQpanda3QVMLayer``.
- ``split_group`` renommé en ``split_groups``.
- Dépend de l'environnement d'exécution NVIDIA pour CUDA 12.6.
- La valeur par défaut de "chip_id" est passée à "WK_C180".
- ``ComplexEntangelingTemplate`` a été renommé en ``ComplexEntanglingTemplate``.
- ``vqc.rst`` : ajout de la section de benchmark "Test 2 : Comparaison des gradients VQC à 10 qubits" comparant VQNet / TorchQuantum / DeepQuantum / Pennylane / MindQuantum.
- Mise à jour du tableau de spécifications de benchmark avec CUDA 12.6, torchquantum 0.2.0 et mindquantum 0.12.0.

Corrigé
===================
- Correction d'un problème de débordement d'entier dans le noyau CUDA ``roll``.
- Correction de la prise en charge de ``cuda_masked_fill`` pour les types complex64/complex128.
- Correction du calcul direct de ``log_softmax`` produisant des valeurs +inf incorrectes sous ``bfloat16``.
- Correction des erreurs de périphérique causées par l'absence de ``CUDAGuard`` lors de l'accès mémoire entre GPU.
- Correction de nombreuses fautes de frappe.


[v2.17.3] - 2026-03-31
***************************

Ajouté
===================
- Ajout du type de données bfloat16.
- Ajout des interfaces de communication NCCL asynchrones : ``nccl_async_all_gather``, ``nccl_async_all_reduce``, ``nccl_async_reduce``, ``nccl_async_broadcast``, ``nccl_async_send`` et ``nccl_async_recv``.
- Ajout de la prise en charge de la dernière puce Origin Quantum avec l'ID de puce ``WK_C180``.
- Ajout de ``data_ptr`` et d'autres interfaces, ajout expérimental de la prise en charge de `triton <https://triton-lang.org/main/index.html>`_.

Modifié
===================

- Le backend par défaut est passé à "pyvqnet-ad".
- Le calcul sur MacOS est implémenté avec les instructions arm neon.
- L'interface ``matmul`` prend en charge les données de plus de 4 dimensions.
- Réduction de la dépendance à certaines bibliothèques d'exécution cuda lors de l'installation du package whl.
- Le type de données QTensor dans pyvqnet n'est plus un entier, mais un type de données spécifique.
- Modification de la logique de pickling de QTensor, ne pickling plus le gradient.
- Suppression de is_dense.
- Suppression de pq2 ``QuantumBatchAsyncQcloudLayer``.
- Mise à jour de la documentation pour pq3 ``QuantumBatchAsyncQcloudLayer``.

Corrigé
===================
- Correction de l'erreur de ``Linear`` lorsque ``use_bias=False``.
- Correction du problème ``MAX_GPUS`` ; le nombre maximal actuel de GPU est désormais de 16.
- Correction d'une erreur d'importation sur Windows jupyter.


[v2.17.2] -2025-11-18
***********************************


Ajouté
===================

- Ajout de la prise en charge du backend `torch` via l'interface de gradient naturel quantique QNG.
- Ajout du backend `pyvqnet-ad`, qui utilise un backend de différenciation automatique C++ similaire à torch. La structure de données utilise toujours le ``_core.Tensor`` d'origine et prend en charge la grande majorité des interfaces existantes.
- Ajout de la documentation "Benchmarking des gradients de circuits variationnels quantiques pour les données par lots".

Modifié
===================

- Suppression de `HybirdVQCQpanda3QVMLayer`, `QuantumLayerMultiProcess`, `TorchHybirdVQCQpanda3QVMLayer` ;
- Suppression de `is_csr`, `csr_members`, `SparseHamiltonian`, `csr_to_dense`, `dense_to_csr` ;
- Ajout des interfaces `QiskitLayer` et `CirqLayer`.
- Ajout de la prise en charge de `if_print_qcloud_log` dans la couche `QuantumBatchAsyncQcloudLayer` pour l'affichage des logs qcloud.
- Modification de la commande d'installation en `pip install pyvqnet --upgrade`.
- La version python prise en charge est `Python 3.10`.
- Modification de la commande d'installation mpicxx spécifiée.



Corrigé
===================

- Prise en charge des valeurs de retour de la dernière version de pyqpanda2QCloud ;
- Ajout de vérifications du périphérique de données d'entrée pour l'interface de produit scalaire ;
- Correction d'un bogue dans `TorchModule` ;



[v2.17.1] - 2025-8-22
***************************

Ajouté
===================

- Ajout de l'interface de l'algorithme SPSA à gradient naturel quantique (qnspsa), de la machine Born à circuit quantique (QBM), de l'interface de gradient naturel quantique avec momentum et d'un exemple QGRU purement quantique.
- Ajout du backend ``torch_native``.
- Ajout d'une interface parallèle par bit pour prendre en charge les circuits quantiques parallèles, et ajout d'une fonction de réorganisation des bits pour réduire le nombre d'échanges de bits.
- Ajout de la méthode ``split_groups``.

Modifié
==================
- Modification de l'implémentation de la couche Linear, passant de `:math:`y = Ax + b` à `:math:`y = x@A.T + b`.
- Modification du paramètre ``obs`` dans l'interface `MeasureAll`.
- Suppression de l'interface ``QuantumLayerES``. Modification des noms de paramètres de ControllComm à ControlComm dans `allgather_group`, `allreduce_group`, `reduce_group`, `broadcast_group` et autres interfaces.
- Suppression de l'interface `ncclsplitGroup`.

Corrigé
==================
- Résolution des problèmes de délai de synchronisation avec les interfaces de communication distribuée.
- Modification des définitions des interfaces de communication distribuée.
- Résolution d'un problème où les calculs de gradient adjoint ne prenaient pas en charge ``PauliX``, ``PauliY`` et ``PauliZ``.


[v2.17.0] - 2025-4-22
***************************

Ajouté
===================

- Ajout de l'implémentation du backend de réseau tensoriel pour les modules de circuits quantiques, incluant la prise en charge des portes logiques de base, de la mesure et des circuits quantiques complexes.
- Ajout de l'implémentation du backend de réseau tensoriel pour la construction de circuits quantiques à grand nombre de qubits.
- Ajout de l'interface QTensor.swapaxes, également appelée swapaxis.

Modifié
==================
- Opérations matricielles utilisant openblas.
- Utilisation de sleef pour les opérations SIMD CPU.
- Suppression de qnn.MeasurePauliSum.
- Affichage d'un avertissement lors de l'utilisation du backend torch lorsque torch est inférieur à la version 2.4.

Corrigé
===================
- Résolution du problème des états QMachine lors de la sauvegarde des modèles.
- Résolution du problème avec layernorm et groupnorm lorsque ``affine=False``.
- Résolution du problème avec ``QuantumLayerAdjoint`` en mode eval.

[v2.16.0] - 2025-1-15
***************************

Ajouté
===================

- Ajout d'une interface pour le calcul de circuits quantiques utilisant pyqpanda3.
- L'interface MeasureAll prend désormais en charge les opérateurs de Pauli composés.
- Ajout des interfaces DataParallelVQCAdjointLayer et DataParallelVQCLayer.

Modifié
===================

- Suppression des fonctions ONNX obsolètes et de la plupart des interfaces intégrant pyqpanda, tout en conservant certaines interfaces utilisées dans les exemples de code.
- Modification de l'interface ``VQC_QuantumEmbedding``.
- Lors de l'installation de ce package, pyqpanda n'est plus installé ; à la place, pyqpanda3 est installé.
- L'interface VQC prend en charge l'utilisation de `x[:,:2]` comme paramètres d'entrée, alors qu'elle ne supportait auparavant que le format `x[:,[2]]`.
- Ce logiciel prend en charge Python 3.10.

Corrigé
===================
- Résolution du problème de fuite de mémoire.
- Résolution du problème de nombres aléatoires sur GPU.
- Pour les opérations liées à reduce, la dimension maximale de tableau prise en charge a été augmentée de 8 à 30.
- Optimisation du code et amélioration de la vitesse d'exécution du code Python dans certains cas.
  
  
[v2.15.0] - 2024-11-19
***************************

Ajouté
===================

- Ajout de l'interface `pyvqnet.backends.set_backend()`. Lorsque les utilisateurs installent `torch`, torch peut être utilisé pour effectuer les calculs matriciels et les calculs de circuits variationnels quantiques de QTensor. Pour plus de détails, consultez le document :ref:`torch_api`.
- Ajout de `pyvqnet.nn.torch` pour hériter de l'interface réseau neuronal et de l'interface neuronale de circuit variationnel quantique de `torch.nn.Module`. Pour plus de détails, consultez le document :ref:`torch_api`.

Modifié
==================
- Modification de l'interface diag.
- Modification de l'implémentation d'all_gather pour être cohérente avec torch.distributed.all_gather.
- Modification de `QTensor` pour prendre en charge jusqu'à 30 dimensions de données.
- Modification de la version requise de `mpi4py` pour les fonctions distribuées, nécessitant désormais la version 4.0.1 ou supérieure.

Corrigé
==================
- Certaines implémentations de nombres aléatoires ne pouvaient pas fixer la graine en raison d'OpenMP.
- Correction de certains bogues dans le démarrage distribué.


[v2.14.0] - 2024-09-30
***************************

Ajouté
===================

- Ajout d'algorithmes de codage par blocs : ``VQC_LCU``, ``VQC_FABLE``, ``VQC_QSVT``, et implémentations d'algorithmes qpanda ``QPANDA_QSVT``, ``QPANDA_LCU``, ``QPANDA_FABLE``.
- Ajout de l'addition d'entiers sur bits quantiques ``vqc_qft_add_to_register``, de l'addition de nombres sur deux bits quantiques ``vqc_qft_add_two_register`` et de la multiplication de nombres sur deux bits quantiques ``vqc_qft_mul``.
- Ajout du module d'entraînement hybride qpanda et vqc ``HybirdVQCQpandaQVMLayer``.
- Ajout des implémentations d'interfaces ``einsum``, ``moveaxis``, ``eigh``, ``diagonal``, etc.
- Ajout des fonctions de calcul parallèle tensoriel dans le calcul distribué : ``ColumnParallelLinear``, ``RowParallelLinear``.
- Ajout de la fonction Zero stage-1 dans le calcul distribué : ``ZeroModelInitial``.
- ``QuantumBatchAsyncQcloudLayer`` : lorsque ``diff_method == "random_coordinate_descent"``, utilise une sélection aléatoire de paramètres pour le calcul du gradient au lieu de PSR.

Modifié
===================
- Suppression de la partie xtensor.
- Restructuration partielle de la documentation API. Distinction entre les exemples d'apprentissage automatique quantique basés sur la différenciation automatique et ceux basés sur qpanda, et distinction entre les interfaces d'apprentissage automatique quantique basées sur la différenciation automatique et les exemples d'interfaces basés sur qpanda.
- `matmul` prend en charge 1d@1d, 2d@1d, 1d@2d.
- Ajout d'alias pour certaines couches de calcul quantique : ``QpandaQCircuitVQCLayer`` = ``QuantumLayer``, ``QpandaQCircuitVQCLayerLite`` = ``QuantumLayerV2``, ``QpandaQProgVQCLayer`` = ``QuantumLayerV3``.

Corrigé
===================
- Modification des interfaces de communication sous-jacentes ``allreduce``, ``allgather``, ``reduce``, ``broadcast`` dans la fonction de calcul distribué, et ajout de la prise en charge de la communication de données ``core.Tensor``.
- Résolution du bogue dans la génération de nombres aléatoires.
- Résolution de l'erreur de conversion des ``RXX``, ``RYY``, ``RZZ``, ``RZX`` de VQC vers originIR.


[v2.13.0] - 2024-07-30
***************************

Ajouté
=================

- Ajout des interfaces `no_grad`, `GroupNorm`, `Interpolate`, `contiguous`, `QuantumLayerV3`, `fuse_model`, `SDPA`.
- Ajout de la méthode Quantum Dropout pour éviter le surapprentissage.

Modifié
==================

- Ajout de l'interface affine à `BatchNorm`, `LayerNorm`, `GroupNorm`.
- L'interface `diag` retourne désormais une sortie 1d sur la diagonale pour une entrée 2d, cohérente avec torch.
- Les opérations telles que slice et permute essaient désormais d'utiliser la méthode view pour retourner un QTensor en mémoire partagée.
- Toutes les interfaces prennent en charge les entrées non contiguës.
- `Adam` prend désormais en charge le paramètre weight_decay.

Corrigé
===================
- Correction de l'erreur de certaines fonctions de décomposition de portes logiques de VQC.
- Correction du problème de fuite mémoire de certaines fonctions.
- Correction du problème où `QuantumLayerMultiProcess` ne supportait pas l'entrée GPU.
- Modification de la méthode d'initialisation par défaut des paramètres de `Linear`.


[v2.12.0] - 2024-05-01
***************************

Ajouté
===================

- Ajout de l'interface PipelineParallelTrainingWrapper.
- Ajout des interfaces `Gelu`, `DropPath`, `binomial`, `adamW`.
- Ajout de `QuantumBatchAsyncQcloudLayer` pour prendre en charge la simulation de machine virtuelle locale de pyqpanda.
- Ajout de `QuantumBatchAsyncQcloudLayer` basé sur xtensor pour prendre en charge la simulation de machine virtuelle locale et le calcul sur machine réelle de pyqpanda.
- Activation de la copie profonde (deepcopy) et du pickling pour QTensor.
- Ajout de la commande de démarrage de calcul distribué `vqnetrun`, utilisée lors de l'utilisation de l'interface de calcul distribué.
- Ajout de l'interface de calcul de gradient par méthode ES pour machine réelle `QuantumBatchAsyncQcloudLayerES` prenant en charge les simulations de VM locales ainsi que les calculs sur machine réelle pour pyqpanda.
- Ajout des interfaces de communication de données `allreduce`, `reduce`, `broadcast`, `allgather`, `send`, `recv`, etc. prenant en charge QTensor dans le calcul distribué.

Modifié
==================

- Ajout des nouvelles dépendances "Pillow" et "hjson" au package d'installation, ajout des nouvelles dépendances "psutil" et "cloudpickle" sur les systèmes Linux.
- Optimisation de la vitesse d'exécution de softmax et transpose sous GPU.
- Compilation avec cuda11.8.
- Intégration des interfaces de calcul distribué basées sur CPU et GPU.

Corrigé
==================
- Réduction de la consommation mémoire au démarrage de la version Linux-GPU.
- Correction du problème de fuite mémoire des fonctions select et power.
- Suppression des méthodes de mise à jour des paramètres et gradients du modèle `nccl_average_parameters_reduce`, `nccl_average_grad_reduce` basées sur la méthode reduce pour cpu, gpu.

[v2.11.0] - 2024-03-01
***************************

Ajouté
===================

- Ajout de la nouvelle API `QNG` (Quantum Natural Gradient) et de sa démonstration.
- Ajout de l'optimisation de circuits quantiques, comme les API `wrapper_single_qubit_op_fuse`, `wrapper_commute_controlled`, `wrapper_merge_rotations` et leurs démonstrations.
- Ajout de `CY`, `SparseHamiltonian`, `HermitianExpval`.
- Ajout de `is_csr`, `is_dense`, `dense_to_csr`, `csr_to_dense`.
- Ajout de `QuantumBatchAsyncQcloudLayer` pour prendre en charge le calcul sur puce réelle QCloud de pyqpanda, `expval_qcloud`.
- Ajout d'implémentations d'interfaces basées sur NCCL pour l'entraînement parallèle de modèles de données de calcul distribué multi-GPU sur un seul nœud : `nccl_average_parameters_allreduce`, `nccl_average_parameters_reduce`, `nccl_average_grad_allreduce`, `nccl_average_grad_reduce`, et classes pour contrôler l'initialisation NCCL et les opérations associées `NCCL_api`.
- Ajout de l'interface de calcul de gradient par stratégie d'évolution de circuit quantique `QuantumLayerES`.

Modifié
==================

- Refonte du circuit `VQC_CSWAP` en `CSWAP`.
- Suppression des anciens documents QNG.
- Suppression du paramètre inutile `num_wires` de `pyvqnet.qnn.vqc` pour les fonctions et classes.
- Refonte des API `MeasureAll`, `Probability`.
- Ajout du paramètre qtype à `QuantumMeasure`.

Corrigé
==================
- Modification de `QuantumMeasure` : slots remplacés par shots.

[v2.10.0] - 2023-12-30
***************************

Ajouté
===========
- Ajout de nouvelles interfaces sous pyvqnet.qnn.vqc : IsingXX, IsingXY, IsingYY, IsingZZ, SDG, TDG, PhaseShift, MultiRZ, MultiCnot, MultixCnot, ControlledPhaseShift, SingleExcitation, DoubleExcitation, VQC_AllSinglesDoubles, ExpressiveEntanglingAnsatz, etc. ;
- Ajout de l'interface pyvqnet.qnn.vqc.QuantumLayerAdjoint prenant en charge le calcul de gradient adjoint ;
- Prise en charge de la fonction de conversion mutuelle entre originIR et VQC ;
- Prise en charge des informations de modules classiques et quantiques dans les modèles VQC statistiques ;
- Ajout de deux cas sous le modèle hybride réseau neuronal classique-quantique : modèle de réseau neuronal convolutif quantique basé sur de petits échantillons, et modèle de fonction kernel quantique pour la reconnaissance de chiffres manuscrits.

[v2.9.0] - 2023-09-08
***************************

Ajouté
==================
- Ajout de la définition d'interface xtensor pour prendre en charge le parallélisme automatique des opérateurs et plusieurs backends CPU/GPU. Elle comprend plus de 150 interfaces pour les calculs mathématiques, logiques et matriciels courants pour les tableaux multidimensionnels, ainsi que des couches de réseaux neuronaux classiques et des optimiseurs courants.

Modifié
==================
- Passage de la version v2.0.8 à v2.9.0.
- Les packages sont téléchargés dans le dépôt PyPI de l'entreprise, utilisez ``pip install pyvqnet --index-url <pypi_url>``.

[v2.0.8] - 2023-07-26
***************************

Ajouté
==================
- Ajout de la prise en charge par les interfaces existantes des types complex128, complex64, double, float, uint8, int8, bool, int16, int32, int64 et autres (gpu).
- Portes logiques de base basées sur vqc : Hadamard, CNOT, I, RX, RY, PauliZ, PauliX, PauliY, S, RZ, RXX, RYY, RZZ, RZX, X1, Y1, Z1, U1, U2, U3, T, SWAP, P, TOFFOLI, CZ, CR, ISWAP.
- Circuits quantiques combinés basés sur vqc : VQC_HardwareEfficientAnsatz, VQC_BasicEntanglerTemplate, VQC_StronglyEntanglingTemplate, VQC_QuantumEmbedding, VQC_RotCircuit, VQC_CRotCircuit, VQC_CSWAPcircuit, VQC_Controlled_Hadamard, VQC_CCZ, VQC_FermionicSingleExcitation, VQC_FermionicDoubleExcitation, VQC_UCCSD, VQC_QuantumPoolingCircuit, VQC_BasisEmbedding, VQC_AngleEmbedding, VQC_AmplitudeEmbedding, VQC_IQPEmbedding.
- Méthodes de mesure basées sur vqc : VQC_Purity, VQC_VarMeasure, VQC_DensityMatrixFromQstate, Probability, MeasureAll.


[v2.0.7] - 2023-07-03
***************************

Ajouté
==================
- Pour les réseaux neuronaux classiques, ajout des interfaces kron, gather, scatter, broadcast_to.
- Ajout de la prise en charge de différentes précisions de données : le type de données dtype prend en charge kbool, kuint8, kint8, kint16, kint32, kint64, kfloat32, kfloat64, kcomplex64, kcomplex128, qui représentent respectivement bool, uint8_t, int8_t, int16_t, int32_t, int64_t, float, double, complex<float>, complex<double>.
- Prise en charge de Python 3.8, 3.9, 3.10.

Modifié
==================
- La fonction init de QTensor et de la classe Module ajoute le paramètre `dtype`. Les types d'index QTensor et d'entrée de certaines couches de réseaux neuronaux sont restreints.
- Réseau neuronal quantique : pour des raisons de compatibilité MacOS, les interfaces Mnist_Dataset et CIFAR10_Dataset ont été supprimées.

[v2.0.6] - 2023-02-22
***************************


Ajouté
===================

- Réseau neuronal classique, ajout des interfaces : multinomial, pixel_shuffle, pixel_unshuffle, ajout de numel pour QTensor, ajout de la fonction de pool mémoire CPU dynamique, ajout de l'interface init_from_tensor pour Parameter.
- Réseau neuronal classique, ajout des interfaces : Dynamic_LSTM, Dynamic_RNN, Dynamic_GRU.
- Réseau neuronal classique, ajout des interfaces : pad_sequence, pad_packed_sequence, pack_pad_sequence.
- Réseau neuronal quantique, ajout des interfaces : CCZ, Controlled_Hadamard, FermionicSingleExcitation, UCCSD, QuantumPoolingCircuit.
- Réseau neuronal quantique, ajout des interfaces : Quantum_Embedding, Mnist_Dataset, CIFAR10_Dataset, grad, Purity.
- Réseau neuronal quantique, ajout d'exemples : basés sur le clipping de gradient, quanvolution, l'expressivité des circuits quantiques, le plateau stérile et l'apprentissage par renforcement quantique QDRL.

Modifié
==================

- Documentation API, restructuration de la structure du contenu, ajout du module "recherche en apprentissage automatique quantique", changement du module "VQNet2ONNX" en "Autres fonctions utilitaires".

Corrigé
==================

- Réseau neuronal classique : résolution du problème où la même graine aléatoire produisait des distributions normales différentes selon les plateformes.
- Réseau neuronal quantique : implémentation de la prise en charge d'expval, ProbMeasure, QuantumMeasure pour la machine virtuelle GPU QPanda.

[v2.0.5] - 2022-12-25
***************************


Ajouté
==================

- Réseau neuronal classique : ajout de l'implémentation de log_softmax, ajout de la fonction export_model du modèle vers ONNX.
- Réseau neuronal classique : prise en charge de la conversion de la plupart des modules de réseaux neuronaux classiques existants vers ONNX. Pour plus de détails, consultez le document API "Module VQNet2ONNX".
- Réseau neuronal quantique : ajout des interfaces VarMeasure, MeasurePauliSum, Quantum_Embedding, SPSA, etc.
- Réseau neuronal quantique : ajout de LinearGNN, ConvGNN, QMLP, gradient naturel quantique, algorithme de décalage de paramètre aléatoire quantique, algorithme DoublySGD, etc.

Modifié
==================

- Réseaux neuronaux classiques : ajout de vérifications de dimensionnalité pour les interfaces BN1d, BN2d.

Corrigé
=================

- Correction du bogue de vérification des paramètres de maxpooling.
- Correction du bogue de découpage [::-1].


[v2.0.4] - 2022-09-20
***************************


Ajouté
=================

- Réseau neuronal classique : ajout de l'implémentation de LayernormNd, prenant en charge le calcul layernorm pour données multidimensionnelles.
- Réseau neuronal classique : ajout des interfaces de calcul de fonction de perte CrossEntropyLoss et NLL_Loss, prenant en charge les entrées de 1 à N dimensions.
- Réseau neuronal quantique : ajout de modèles de circuits courants : HardwareEfficientAnsatz, StronglyEntanglingTemplate, BasicEntanglerTemplate.
- Réseau neuronal quantique : ajout de l'interface Mutal_info pour le calcul de l'information mutuelle des sous-systèmes de qubits, de l'entropie de Von Neumann VB_Entropy et de la matrice de densité DensityMatrixFromQstate.
- Réseau neuronal quantique : ajout de l'exemple d'algorithme de perceptron quantique QuantumNeuron, ajout de l'exemple d'algorithme de série de Fourier quantique.
- Réseau neuronal quantique : ajout de l'interface QuantumLayerMultiProcess prenant en charge l'accélération multi-processus des circuits quantiques.

Modifié
=================

- Réseau neuronal classique : prise en charge du paramètre de groupe pour la convolution groupée, du taux de dilatation pour la convolution dilatée et du padding de valeur arbitraire comme paramètres pour la convolution 1D Conv1d, la convolution 2D Conv2d et la déconvolution ConvT2d.
- Saut de l'opération de broadcast pour les données dans la même dimension, réduisant la logique d'exécution inutile.

Corrigé
=================

- Résolution du problème où la fonction stack calculait incorrectement sous certains paramètres.

[v2.0.3] - 2022-07-15
***************************


Ajouté
=================

- Ajout de la prise en charge de stack, des interfaces de réseau neuronal récurrent bidirectionnel : RNN, LSTM, GRU.
- Ajout d'interfaces pour les indicateurs de performance de calcul courants : MSE, RMSE, MAE, R_Square, precision_recall_f1_2_score, precision_recall_f1_Multi_score, precision_recall_f1_N_score, auc_calculate.
- Ajout de l'exemple d'algorithme de SVM à kernel quantique.

Modifié
=================

- Accélération de la vitesse d'impression lorsqu'il y a trop de données QTensor.
- Utilisation d'openmp pour accélérer les calculs sous Windows et Linux.

Corrigé
=================

- Résolution du problème où certaines méthodes d'importation Python échouaient.
- Résolution du problème de calculs répétés des couches de normalisation par lots (BN).
- Résolution du bogue où les interfaces ``QTensor.reshape`` et ``transpose`` ne pouvaient pas calculer les gradients.
- Ajout de la validation de forme des paramètres d'entrée pour l'interface ``tensor.power``.

[v2.0.2] - 2022-05-15
***************************


Ajouté
=================

- Ajout de topK, argtopK.
- Ajout de cumsum.
- Ajout de masked_fill.
- Ajout de triu, tril.
- Ajout d'exemples de distribution aléatoire générée par QGAN.

Modifié
=================

- Prise en charge des index de découpage avancés et des index de découpage courants.
- matmul prend en charge les opérations tensorielles 3D et 4D.
- Modification de l'implémentation de la fonction HardSigmoid.

Corrigé
=================

- Résolution du problème où la convolution, la normalisation par lots, la déconvolution, les couches de pooling, etc. ne mettaient pas en cache les variables internes, ce qui causait des problèmes de calcul de gradient lors de plusieurs passages arrière après un passage avant.
- Correction de l'implémentation et de l'exemple de la couche QLinear.
- Résolution du problème où le chargement d'Image échouait lors de l'importation de VQNet dans un environnement conda sur macOS.

[v2.0.1] - 2022-03-30
**************************


Ajouté
=================

- Plus de 100 interfaces de base de la structure de données QTensor ont été ajoutées, incluant des fonctions de création, des fonctions logiques, des fonctions mathématiques et des opérations matricielles.
- Ajout de 14 fonctions de base de réseaux neuronaux, incluant convolution, déconvolution, pooling, etc.
- Ajout de 4 fonctions de perte, incluant MSE, BCE, CCE, SCE, etc.
- Ajout de 10 fonctions d'activation, incluant ReLu, Sigmoid, ELU, etc.
- Ajout de 6 optimiseurs, incluant SGD, RMSPROP, ADAM, etc.
- Ajout d'exemples d'apprentissage automatique : QVC, QDRL, Q-KMEANS, QUnet, HQCNN, VSQL, Quantum Autoencoder.
- Ajout de couches d'apprentissage automatique quantique : QuantumLayer, NoiseQuantumLayer.