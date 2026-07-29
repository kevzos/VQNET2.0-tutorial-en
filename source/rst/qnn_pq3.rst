Utiliser le module d'apprentissage automatique quantique pyQPanda3
##################################################################

.. warning::

    La partie informatique quantique de l'interface suivante utilise pyqpanda3.

    Si vous utilisez la fonction QCloud sous ce module, des erreurs se produiront lors de l'importation de pyqpanda2 dans le code ou lors de l'utilisation de l'interface du paquet lié à pyqpanda2 de pyvqnet.

Couche d'informatique quantique
*******************************

.. _QuantumLayer_pq3:

QuantumLayer
============

Si vous connaissez la syntaxe pyQPanda3, vous pouvez utiliser l'interface QuantumLayer pour personnaliser le simulateur pyqpanda3 pour le calcul.

.. py:class:: pyvqnet.qnn.pq3.quantumlayer.QuantumLayer(qprog_with_measure,para_num,diff_method:str = "parameter_shift",delta:float = 0.01,dtype=None,name="")

    Module de calcul abstrait de la couche quantique variationnelle. Utilise pyQPanda3 pour simuler un circuit quantique paramétré et obtenir les résultats de mesure. Cette couche quantique variationnelle hérite du module de calcul de gradient du framework VQNet. Elle peut utiliser la méthode de décalage de paramètre pour calculer le gradient des paramètres du circuit, entraîner des modèles de circuits quantiques variationnels ou intégrer des circuits quantiques variationnels dans des modèles hybrides quantiques et classiques.

    :param qprog_with_measure: Fonctions d'opération et de mesure du circuit quantique construites avec pyQPanda.
    :param para_num: ``int`` - nombre de paramètres.
    :param diff_method: Méthode de calcul des gradients des paramètres du circuit quantique, « parameter shift » ou « finite difference », par défaut décalage de paramètre.
    :param delta: \delta lors du calcul des gradients par différences finies.
    :param dtype: type de données du paramètre, par défaut : None, utilise le type de données par défaut : kfloat32, représentant des nombres à virgule flottante 32 bits.
    :param name: le nom de ce module, par défaut "".

    :return: un module qui peut calculer des circuits quantiques.

    .. note::

        qprog_with_measure est une fonction de circuit quantique définie dans pyQPanda3.

        Cette fonction doit contenir deux paramètres, input et param, en entrée de la fonction (même si un paramètre n'est pas réellement utilisé), et la sortie est le résultat de mesure ou la valeur attendue du circuit (doit être np.ndarray ou une liste contenant des valeurs), sinon elle ne fonctionnera pas correctement dans QpandaQCircuitVQCLayerLite.

        L'utilisation de la fonction de circuit quantique qprog_with_measure (input, param) peut être consultée dans l'exemple ci-dessous.

        ``input`` : Entrée de données classiques unidimensionnelles. Si non, entrer None.

        ``param`` : Entrée des paramètres unidimensionnels du circuit quantique variationnel à entraîner.

    .. note::

        Cette classe a pour alias `QuantumLayerV2`, `QpandaQCircuitVQCLayerLite`.

    Example::

        from pyvqnet.qnn.pq3.measure import ProbsMeasure
        from pyvqnet.qnn.pq3.quantumlayer import QuantumLayer
        from pyvqnet.tensor import QTensor,ones
        import pyqpanda3.core as pq
        def pqctest (input,param):
            num_of_qubits = 4

            m_machine = pq.CPUQVM()

            qubits = range(num_of_qubits)

            circuit = pq.QCircuit()
            circuit<<pq.H(qubits[0])
            circuit<<pq.H(qubits[1])
            circuit<<pq.H(qubits[2])
            circuit<<pq.H(qubits[3])

            circuit<<pq.RZ(qubits[0],input[0])
            circuit<<pq.RZ(qubits[1],input[1])
            circuit<<pq.RZ(qubits[2],input[2])
            circuit<<pq.RZ(qubits[3],input[3])

            circuit<<pq.CNOT(qubits[0],qubits[1])
            circuit<<pq.RZ(qubits[1],param[0])
            circuit<<pq.CNOT(qubits[0],qubits[1])

            circuit<<pq.CNOT(qubits[1],qubits[2])
            circuit<<pq.RZ(qubits[2],param[1])
            circuit<<pq.CNOT(qubits[1],qubits[2])

            circuit<<pq.CNOT(qubits[2],qubits[3])
            circuit<<pq.RZ(qubits[3],param[2])
            circuit<<pq.CNOT(qubits[2],qubits[3])

            prog = pq.QProg()
            prog<<circuit

            rlt_prob = ProbsMeasure(m_machine,prog,[0,2])
            return rlt_prob
        pqc = QuantumLayer(pqctest,3)

        # donnees classiques en entree
        input = QTensor([[1,2,3,4],[4,2,2,3],[3.0,3,2,2]] )

        # circuits de propagation avant
        rlt = pqc(input)

        print(rlt)

        grad = ones(rlt.data.shape)*1000
        # circuits de retropropagation
        rlt.backward(grad)

        print(pqc.m_para.grad)

QpandaQProgVQCLayer
====================

.. py:class:: pyvqnet.qnn.pq3.quantumlayer.QuantumLayerV3(origin_qprog_func,para_num,qvm_type="cpu", pauli_str_dict=None, shots=1000, initializer=None,dtype=None,name="")

    Il soumet le circuit quantique paramétré au simulateur local pleine amplitude QPanda3 pour le calcul et entraîne les paramètres dans le circuit.
    Il prend en charge les données par lots et utilise la règle du décalage de paramètre pour estimer le gradient des paramètres.
    Pour CRX, CRY, CRZ, cette couche utilise la formule de https://iopscience.iop.org/article/10.1088/1367-2630/ac2cb3, et les autres portes logiques utilisent la méthode de décalage de paramètre par défaut pour calculer le gradient.

    :param origin_qprog_func: La fonction de circuit quantique appelable construite par QPanda.
    :param para_num: ``int`` - Nombre de paramètres ; les paramètres sont unidimensionnels.
    :param qvm_type: ``str`` - Type de simulateur pyqpanda3 à utiliser, « cpu » ou « gpu », par défaut « cpu ».
    :param pauli_str_dict: ``dict|list`` - Dictionnaire ou liste de dictionnaires représentant les opérateurs de Pauli dans le circuit quantique. Par défaut, None.
    :param shots: ``int`` - Nombre de mesures. Par défaut, 1000.
    :param initializer: Initialisateur des valeurs des paramètres. Par défaut, None.
    :param dtype: Type de données du paramètre. Par défaut, None, ce qui signifie utiliser le type de données par défaut.
    :param name: Nom du module. Par défaut, la chaîne vide.

    :return: Renvoie une classe QpandaQProgVQCLayer

    .. note::

        origin_qprog_func est une fonction de circuit quantique définie par l'utilisateur utilisant pyQPanda3.

        Cette fonction doit contenir deux paramètres, input et param, en entrée de la fonction (même si un paramètre n'est pas réellement utilisé), et la sortie est des données de type pyqpanda3.core.QProg, sinon elle ne peut pas fonctionner correctement dans QuantumLayerV3.

        origin_qprog_func (input, param)

        ``input`` : entrée de tableau défini par l'utilisateur, données classiques unidimensionnelles

        ``param`` : entrée de type tableau, paramètres unidimensionnels du circuit quantique définis par l'utilisateur

    .. note::

        Cette classe a pour alias `QuantumLayerV3`.

    Example::

        import pyqpanda3.core as pq
        import pyvqnet
        from pyvqnet.qnn.pq3.quantumlayer import  QuantumLayerV3


        def qfun(input, param ):
            m_qlist = range(3)
            cbits = range(3)
            measure_qubits = [0,1, 2]
            m_prog = pq.QProg()
            cir = pq.QCircuit(3)

            cir<<pq.RZ(m_qlist[0], input[0])
            cir<<pq.RX(m_qlist[2], input[2])
            
            qcir = pq.RX(m_qlist[1], param[1]).control(m_qlist[0])
        
            cir<<qcir

            qcir = pq.RY(m_qlist[0], param[2]).control(m_qlist[1])
        
            cir<<qcir

            cir<<pq.RY(m_qlist[0], input[1])

            qcir = pq.RZ(m_qlist[0], param[3]).control(m_qlist[1])
        
            cir<<qcir
            m_prog<<cir

            for idx, ele in enumerate(measure_qubits):
                m_prog << pq.measure(m_qlist[ele], cbits[idx])  # pylint: disable=expression-not-assigned
            return m_prog

        from pyvqnet.utils.initializer import ones
        l = QuantumLayerV3(qfun,
                        4,
                        "cpu",
                        pauli_str_dict=None,
                        shots=1000,
                        initializer=ones,
                        name="")
        x = pyvqnet.tensor.QTensor(
            [[2.56, 1.2,-3]],
            requires_grad=True)
        y = l(x)

        y.backward()
        print(l.m_para.grad.to_numpy())
        print(x.grad.to_numpy())

QuantumBatchAsyncQcloudLayer
=============================

Lorsque vous installez la dernière version de pyqpanda3, vous pouvez utiliser cette interface pour définir un circuit variationnel et le soumettre à la puce réelle originqc pour exécution.

.. py:class:: pyvqnet.qnn.pq3.quantumlayer.QuantumBatchAsyncQcloudLayer(origin_qprog_func, qcloud_token, para_num, pauli_str_dict=None, shots = 1000, initializer=None, dtype=None, name="", diff_method="parameter_shift", submit_kwargs={}, query_kwargs={})
    
    Un module de calcul abstrait pour la puce réelle originqc utilisant pyqpanda3 QCLOUD. Il soumet des circuits quantiques paramétrés à la puce réelle et obtient les résultats de mesure.
    Si diff_method == "random_coordinate_descent", la couche sélectionnera aléatoirement un seul paramètre pour calculer le gradient, et les autres paramètres resteront à zéro. Référence : https://arxiv.org/abs/2311.00088

    .. note::

        qcloud_token est le jeton API que vous avez demandé auprès de la plateforme cloud.

        origin_qprog_func doit renvoyer des données de type pyqpanda3.core.QProg. Si pauli_str_dict n'est pas défini, il est nécessaire de s'assurer que la mesure a été insérée dans le QProg.

        origin_qprog_func doit être au format suivant :

        origin_qprog_func(input, param)

            ``input`` : Entrée de données classiques 1D à 2D. Dans le cas 2D, la première dimension est la taille du lot

            ``param`` : Entrée des paramètres à entraîner pour le circuit quantique variationnel 1D

    .. note::

        Dans la version actuelle, le délai d'attente total par défaut pour la soumission d'un seul circuit au QCloud est de 60 secondes. Si un délai d'attente se produit en raison de la saturation du QCloud, vous pouvez définir la valeur de la clé ``total_timeout`` dans ``query_kwargs`` au nombre de secondes d'attente souhaité.

    :param origin_qprog_func: La fonction de circuit quantique variationnel construite par QPanda, qui doit renvoyer un QProg.
    :param qcloud_token: ``str`` - Le type de machine quantique ou le jeton cloud utilisé pour l'exécution.
    :param para_num: ``int`` - Le nombre de paramètres, le paramètre est un QTensor de taille [para_num].
    :param pauli_str_dict: ``dict|list`` - Un dictionnaire ou une liste de dictionnaires représentant les opérateurs de Pauli dans le circuit quantique. La valeur par défaut est « None », qui effectue des opérations de mesure. Si un dictionnaire d'opérateurs de Pauli est saisi, une espérance unique ou plusieurs espérances seront calculées.
    :param shots: ``int`` - Le nombre de mesures. La valeur par défaut est 1000.
    :param initializer: Initialisateur des valeurs des paramètres. La valeur par défaut est « None », qui utilise une distribution normale 0~2*pi.
    :param dtype: Le type de données du paramètre. La valeur par défaut est None, ce qui signifie utiliser le type de données par défaut pyvqnet.kfloat32.
    :param name: Le nom du module. La valeur par défaut est une chaîne vide.
    :param diff_method: Méthode de différenciation pour le calcul du gradient. La valeur par défaut est « parameter_shift », « random_coordinate_descent ».
    :param submit_kwargs: Paramètres de mots-clés supplémentaires pour la soumission des circuits quantiques, par défaut : {"if_print_qcloud_log":False,"chip_id":"WK_C180","is_amend":True,"is_mapping":True,"is_optimization":True,"compile_level":3,"default_task_group_size":200,"test_qcloud_fake":False,"":"server_ip_address"}, lorsque test_qcloud_fake est défini sur True, simulation CPUQVM locale.
    :param query_kwargs: Paramètres de mots-clés supplémentaires pour interroger les résultats quantiques, par défaut : {"timeout":1,"total_timeout":60,"print_query_info":True,"sub_circuits_split_size":1}.
    :return: Un module qui peut calculer des circuits quantiques.

    Example::

        import pyqpanda3.core as pq
        import pyvqnet
        from pyvqnet.qnn.pq3.quantumlayer import QuantumBatchAsyncQcloudLayer

        def qfun(input,param):
            measure_qubits = [0,2]
            m_qlist = range(6)
            cir = pq.QCircuit(6)
            cir << (pq.RZ(m_qlist[0],input[0]))
            cir << pq.CNOT(m_qlist[0],m_qlist[1])
            cir << pq.RY(m_qlist[1],param[0])
            cir << pq.CNOT(m_qlist[0],m_qlist[2])
            cir << pq.RZ(m_qlist[1],input[1])
            cir << pq.RY(m_qlist[2],param[1])
            cir << pq.H(m_qlist[2])
            m_prog = pq.QProg(cir)


            for idx, ele in enumerate(measure_qubits):
                m_prog << pq.measure(m_qlist[ele], m_qlist[idx])  # pylint: disable=expression-not-assigned
            return m_prog

        l = QuantumBatchAsyncQcloudLayer(qfun,
                        "3047DE8A59764BEDAC9C3282093B16AF1",
                        2,

                        pauli_str_dict=None,
                        shots = 1000,
                        initializer=None,
                        dtype=None,
                        name="",
                        diff_method="parameter_shift",
                        submit_kwargs={"test_qcloud_fake":True},
                        query_kwargs={})
        x = pyvqnet.tensor.QTensor([[0.56,1.2],[0.56,1.2],[0.56,1.2],[0.56,1.2],[0.56,1.2]],requires_grad= True)
        y = l(x)
        print(y)
        y.backward()
        print(l.m_para.grad)
        print(x.grad)

        def qfun2(input,param ):
            
            m_qlist = range(6)
            cir = pq.QCircuit(6)
            cir<<pq.RZ(m_qlist[0],input[0])
            cir<<pq.CNOT(m_qlist[0],m_qlist[1])
            cir<<pq.RY(m_qlist[1],param[0])
            cir<<pq.CNOT(m_qlist[0],m_qlist[2])
            cir<<pq.RZ(m_qlist[1],input[1])
            cir<<pq.RY(m_qlist[2],param[1])
            cir<<pq.H(m_qlist[2])
            m_prog = pq.QProg(cir)

        
            
            return m_prog
        l = QuantumBatchAsyncQcloudLayer(qfun2,
                "3047DE8A59764BEDAC9C3282093B16AF",
                2,

                pauli_str_dict={'Z0 X1':10,'':-0.5,'Y2':-0.543,"":3333},
                shots = 1000,
                initializer=None,
                dtype=None,
                name="",
                diff_method="parameter_shift",
                submit_kwargs={"test_qcloud_fake":True},
                query_kwargs={})
        x = pyvqnet.tensor.QTensor([[0.56,1.2],[0.56,1.2],[0.56,1.2],[0.56,1.2]],requires_grad= True)
        y = l(x)
        print(y)
        y.backward()
        print(l.m_para.grad)
        print(x.grad)


QuantumLayerAdjoint
===================

.. py:class:: pyvqnet.qnn.pq3.quantumlayer.QuantumLayerAdjoint(pq3_vqc_circuit, param_num, pauli_dicts, dtype = None, name="")

    Cette classe utilise l'interface VQCircuit de pyqpanda3 pour calculer les gradients des paramètres d'un circuit quantique par rapport à l'hamiltonien en utilisant la méthode adjointe.

    Cette classe prend en charge l'entrée par lots et les sorties multi-hamiltoniennes.

    .. note::

        Lors de l'utilisation de cette interface, vous devez construire le circuit en utilisant des portes logiques de VQCircuit.

        Actuellement, un nombre limité de portes logiques sont prises en charge ; une exception sera levée si une porte non prise en charge est utilisée.

        Le paramètre d'entrée ``pq3_vqc_circuit`` ne peut contenir que deux paramètres, `x` et `param`, qui doivent être un tableau ou une liste unidimensionnelle.

        Dans la fonction ``pq3_vqc_circuit``, les utilisateurs doivent utiliser ``pyqpanda3.vqcircuit.VQCircuit().set_Param`` pour personnaliser la gestion des entrées et des paramètres.

        De plus, les utilisateurs doivent pré-saisir le nombre de paramètres dans ``param_num``. Cette interface initialisera un paramètre ``m_para`` avec une longueur de ``param_num``.

        Voir l'exemple ci-dessous.

    :param pq3_vqc_circuit: Personnalise le circuit VQCircuit de pyqpanda3.
    :param param_num: Nombre de paramètres. :param pauli_dicts: Observations attendues, peut être une liste.
    :param dtype: Type de paramètre, kfloat32 ou kfloat64, par défaut : None, utilise kfloat32.
    :param name: Le nom de cette interface.
    :return: Renvoie une instance de QuantumLayerAdjoint


    Example::

        from pyvqnet.qnn.pq3 import QuantumLayerAdjoint
        from pyvqnet import tensor

        from pyqpanda3.vqcircuit import VQCircuit
        import pyqpanda3 as pq3

        l = 3
        n = 7
        def pqctest(x,param):
            vqc = VQCircuit()
            vqc.set_Param([len(param) +len(x)])
            w_offset = len(x)
            for j in range(len(x)):
                vqc << pq3.core.RX(j, vqc.Param([j  ]))
            for j in range(l):
                for i in range(n - 1):
                    vqc << pq3.core.CNOT(i, i + 1)
                for i in range(n):
                    vqc << pq3.core.RX(i, vqc.Param([w_offset + 3 * n * j + i]))
                        
                    vqc << pq3.core.RZ(i, vqc.Param([w_offset + 3 * n * j + i + n]))
                    vqc << pq3.core.RY(i, vqc.Param([w_offset + 3 * n * j + i + 2 * n]))
            
            return vqc

        Xn_string = ' '.join([f'X{i}' for i in range(n)])
        pauli_dict  = {Xn_string:1.}

        layer = QuantumLayerAdjoint(pqctest,3*l*n,pauli_dict)

        x = tensor.randn([2,5])
        x.requires_grad = True
        y = layer(x)
        y.backward()
        print(layer.m_para.grad)
        print(x.grad)

        Xn_string = ' '.join([f'X{i}' for i in range(n)])
        Zn_string = ' '.join([f'Z{i}' for i in range(n)])
        pauli_dict  = {Xn_string:1.,Zn_string:0.5}

        layer = QuantumLayerAdjoint(pqctest,3*l*n,pauli_dict)

        x = tensor.randn([2,5])
        x.requires_grad = True
        y = layer(x)
        y.backward()
        print(layer.m_para.grad)
        print(x.grad)

        Xn_string = ' '.join([f'X{i}' for i in range(n)])
        Zn_string = ' '.join([f'Z{i}' for i in range(n)])
        pauli_dict  = {Xn_string:1.,Zn_string:0.5}

        layer = QuantumLayerAdjoint(pqctest,3*l*n,pauli_dict)

        x = tensor.randn([1,5])
        x.requires_grad = True
        y = layer(x)
        y.backward()
        print(layer.m_para.grad)
        print(x.grad)

        Xn_string = ' '.join([f'X{i}' for i in range(n)])
        Zn_string = ' '.join([f'Z{i}' for i in range(n)])
        pauli_dict  = [{Xn_string:1.,Zn_string:0.5},{Xn_string:1.,Zn_string:0.5}]

        layer = QuantumLayerAdjoint(pqctest,3*l*n,pauli_dict)

        x = tensor.randn([1,5])
        x.requires_grad = True
        y = layer(x)
        y.backward()
        print(layer.m_para.grad)
        print(x.grad)

        Xn_string = ' '.join([f'X{i}' for i in range(n)])
        Zn_string = ' '.join([f'Z{i}' for i in range(n)])
        pauli_dict  = [{Xn_string:1.,Zn_string:0.5},{Xn_string:1.,Zn_string:0.5}]

        layer = QuantumLayerAdjoint(pqctest,3*l*n,pauli_dict)

        x = tensor.randn([2,5])
        x.requires_grad = True
        y = layer(x)
        y.backward()
        print(layer.m_para.grad)
        print(x.grad)
        """
        [-0.1086438, 0.1805159, 0.2619071,..., 0.1508062, 0.0329617,-0.0043367]
        <QTensor [63] DEV_CPU kfloat32>

        [[-0.0425088, 0.0187212,-0.0326243, 0.1314874,-0.0729216],
        [-0.0972663,-0.0371378,-0.0455299,-0.0170686,-0.0328533]]
        <QTensor [2, 5] DEV_CPU kfloat32>

        [ 0.0706403,-0.1070583, 0.0547093,...,-0.0183769,-0.0742296, 0.0026942]
        <QTensor [63] DEV_CPU kfloat32>

        [[-0.07577  ,-0.1364278, 0.0220043, 0.0690343, 0.0281384],
        [ 0.0075356,-0.1627405,-0.0381604, 0.1185545, 0.1409108]]
        <QTensor [2, 5] DEV_CPU kfloat32>

        [-0.0634308,-0.0128268, 0.0396237,...,-0.0350691,-0.116307 , 0.0164972]
        <QTensor [63] DEV_CPU kfloat32>

        [[-0.0823639,-0.0418629, 0.0105356, 0.0699336, 0.041226 ]]
        <QTensor [1, 5] DEV_CPU kfloat32>

        [-0.1281752, 0.0852512, 0.0678721,...,-0.080481 , 0.0202518,-0.0348869]
        <QTensor [63] DEV_CPU kfloat32>

        [[-0.0339751,-0.0330053,-0.0651799, 0.2171837,-0.1267595]]
        <QTensor [1, 5] DEV_CPU kfloat32>

        [ 0.305574 , 0.2730191, 0.0605986,...,-0.2138517,-0.2475468, 0.174026 ]
        <QTensor [63] DEV_CPU kfloat32>

        [[ 0.1867954,-0.0704528,-0.0603823,-0.0123921,-0.0938597],
        [-0.041001 ,-0.2520995, 0.0683114,-0.0986969, 0.1000023]]
        <QTensor [2, 5] DEV_CPU kfloat32>
        """

VQCQCloudLayer
==============

.. py:class:: pyvqnet.qnn.pq3.vqc_qcloud_layer.VQCQCloudLayer(vqc_module, qcloud_token, pauli_str_dict=None, shots=1000, name="", submit_kwargs={}, query_kwargs={})

    Soumet le module VQC à la puce réelle QCloud ou au simulateur local pyqpanda3 pour exécution.

    Propagation avant : au lieu d'exécuter le calcul de circuit quantique variationnel de VQNet, il appelle la puce réelle quantique ou le simulateur local qpanda pour le calcul.

    Propagation arrière : utilise la règle du décalage de paramètre pour calculer les gradients. Pour chaque dimension d'entrée et chaque paramètre entraînable du VQC,
    génère des circuits décalés de +/- pi/2 et les soumet pour le calcul, récupère les résultats pour calculer le jacobien. Les gradients sont définis sur le tenseur d'entrée et les paramètres entraînables du VQC.

    .. note::

        Le délai d'attente total par défaut pour la soumission d'un seul circuit au QCloud est de 60 secondes. Si un délai d'attente se produit en raison de la saturation du QCloud, vous pouvez définir la clé ``total_timeout`` dans ``query_kwargs`` pour spécifier les secondes d'attente.

    .. note::

        Vous ne pouvez pas définir de fonction de mesure (comme ``MeasureAll``) dans ``vqc_module``. La mesure doit être spécifiée via le paramètre ``pauli_str_dict`` pour indiquer les observables.
        Par exemple : ``VQCQCloudLayer(vqc_module, token, pauli_str_dict={'Z0': 1, 'Z1': 1})``.

    :param vqc_module: Module VQC VQNet, doit inclure une QMachine avec save_ir=True.
    :param qcloud_token: Jeton API QCloud. Passez une chaîne vide si vous utilisez un simulateur local.
    :param pauli_str_dict: Dictionnaire d'opérateurs de Pauli pour le calcul de la valeur d'espérance. Par défaut, None, ce qui effectue l'opération de mesure.
    :param shots: Nombre de mesures. Par défaut, 1000.
    :param name: Nom du module. Par défaut, chaîne vide.
    :param submit_kwargs: Paramètres de mots-clés supplémentaires pour la soumission des circuits quantiques, par défaut : {"if_print_qcloud_log":False,"chip_id":"WK_C180","is_amend":True,"is_mapping":True,"is_optimization":True,"compile_level":3,"default_task_group_size":200,"test_qcloud_fake":False,"":"server_ip_address"}, lorsque test_qcloud_fake est défini sur True, simulation CPUQVM locale.
    :param query_kwargs: Paramètres de mots-clés supplémentaires pour interroger les résultats quantiques. Par défaut : {"timeout":1,"total_timeout":60, "print_query_info":True,"sub_circuits_split_size":1}.

    Example::

        from pyvqnet.qnn.vqc import *
        from pyvqnet.qnn.pq3 import VQCQCloudLayer
        from pyvqnet.nn import Module
        import pyvqnet

        class QModel(Module):
            def __init__(self, num_wires, dtype):
                super(QModel, self).__init__()
                self._num_wires = num_wires
                self._dtype = dtype
                self.qm = QMachine(num_wires, dtype=dtype, save_ir=True)
                self.rx_layer = RX(has_params=True, trainable=False, wires=0)
                self.u1 = U1(has_params=True, trainable=True, wires=[1])
                self.cnot = CNOT(wires=[0, 1])

            def forward(self, x, *args, **kwargs):
                self.qm.reset_states(x.shape[0])
                self.rx_layer(params=x[:, [0]], q_machine=self.qm)
                self.cnot(q_machine=self.qm)
                self.u1(q_machine=self.qm)
                return x

        qmodel = QModel(num_wires=2, dtype=pyvqnet.kcomplex64)
        layer = VQCQCloudLayer(
            qmodel,
            "your_qcloud_token",
            pauli_str_dict={'Z0': 1, 'Z1': 1},
            shots=1000,
            submit_kwargs={"test_qcloud_fake": True},
        )
        x = pyvqnet.tensor.QTensor([[0.5, 0.3], [0.5, 0.3]], requires_grad=True)
        y = layer(x)
        y.backward()
        print(x.grad)
        print(qmodel.u1.params.grad)

grad
====
.. py:function:: pyvqnet.qnn.pq3.quantumlayer.grad(quantum_prog_func, input_params, *args)

    La fonction grad fournit une interface pour calculer le gradient des paramètres du circuit quantique paramétré conçu par l'utilisateur.
    Les utilisateurs peuvent utiliser pyQPanda3 pour concevoir la fonction d'exécution du circuit ``quantum_prog_func`` comme indiqué ci-dessous, et la passer comme paramètre à la fonction grad.
    Le deuxième paramètre de la fonction grad correspond aux coordonnées du gradient du paramètre de la porte logique quantique que vous souhaitez calculer.
    La forme de la valeur de retour est [nombre de paramètres, nombre de sorties].

    :param quantum_prog_func: fonction d'exécution du circuit quantique conçue par pyQPanda3.
    :param input_params: paramètres pour lesquels le gradient doit être calculé.
    :param \*args: autres paramètres d'entrée de la fonction quantum_prog_func.
    :return:
        Gradient des paramètres


    Examples::

        from pyvqnet.qnn.pq3 import grad, ProbsMeasure
        import pyqpanda3.core as pq

        def pqctest(param):
            machine = pq.CPUQVM()
        
            qubits = range(2)
            circuit = pq.QCircuit(2)

            circuit<<pq.RX(qubits[0], param[0])

            circuit<<pq.RY(qubits[1], param[1])
            circuit<<pq.CNOT(qubits[0], qubits[1])

            circuit<<pq.RX(qubits[1], param[2])

            prog = pq.QProg()
            prog<<circuit

            EXP = ProbsMeasure(machine,prog,[1])
            return EXP


        g = grad(pqctest, [0.1,0.2, 0.3])
        print(g)
        exp = pqctest([0.1,0.2, 0.3])
        print(exp)






QLinear
=======

QLinear implémente un algorithme de connexion totale quantique. D'abord, les données sont encodées dans un état quantique, puis l'opération d'évolution et la mesure sont effectuées via des circuits quantiques pour obtenir le résultat final de connexion totale.

.. image:: ./images/qlinear_cir.png

.. py:class:: pyvqnet.qnn.qlinear.QLinear(input_channels,output_channels,machine: str = "CPU")

    Module quantique entièrement connecté. L'entrée du module entièrement connecté a la forme (canaux d'entrée, canaux de sortie). Notez que cette couche ne prend pas de paramètres quantiques variationnels.

    :param input_channels: ``int`` - Nombre de canaux d'entrée.
    :param output_channels: ``int`` - Nombre de canaux de sortie.
    :param machine: ``str`` - La machine virtuelle à utiliser, la simulation CPU est utilisée par défaut.
    :return: Couche quantique entièrement connectée.

    Example::

        from pyvqnet.tensor import QTensor
        from pyvqnet.qnn.qlinear import QLinear
        params = [[0.37454012, 0.95071431, 0.73199394, 0.59865848, 0.15601864, 0.15599452], 
        [1.37454012, 0.95071431, 0.73199394, 0.59865848, 0.15601864, 0.15599452],
        [1.37454012, 1.95071431, 0.73199394, 0.59865848, 0.15601864, 0.15599452],
        [1.37454012, 1.95071431, 1.73199394, 1.59865848, 0.15601864, 0.15599452]]
        m = QLinear(6, 2)
        input = QTensor(params, requires_grad=True)
        output = m(input)
        output.backward()
        print(output)

        #[
        #[0.0568473, 0.1264389],
        #[0.1524036, 0.1264389],
        #[0.1524036, 0.1442845],
        #[0.1524036, 0.1442845]
        #]



QConv
=====

    QConv est une interface d'algorithme de convolution quantique.
    L'opération de convolution quantique utilise des circuits quantiques pour effectuer des opérations de convolution sur des données classiques. Elle n'a pas besoin de calculer des opérations de multiplication et d'addition. Il suffit d'encoder les données dans des états quantiques, puis d'effectuer des opérations d'évolution et des mesures via des circuits quantiques pour obtenir les résultats de convolution finaux.
    Le même nombre de bits quantiques est alloué en fonction du nombre de données d'entrée dans la plage du noyau de convolution, puis des circuits quantiques sont construits pour le calcul.

    .. image:: ./images/qcnn.png

    Le circuit quantique est encodé en insérant d'abord des portes :math:`RY` et :math:`RZ` sur chaque qubit, puis en utilisant :math:`Z` et :math:`U3` sur deux qubits quelconques pour les intriquer et échanger des informations. Voici un exemple avec 4 qubits

    .. image:: ./images/qcnn_cir.png

.. py:class:: pyvqnet.qnn.qcnn.qconv.QConv(input_channels,output_channels,quantum_number,stride=(1, 1),padding=(0, 0),kernel_initializer=normal,machine:str = "CPU", dtype=None, name ="")

    Module de convolution quantique. Remplace le noyau Conv2D par un circuit quantique. L'entrée du module de convolution a la forme (taille du lot, canaux d'entrée, hauteur, largeur) `Samuel et al. (2020) <https://arxiv.org/abs/2012.12177>`_ .

        :param input_channels: ``int`` - Nombre de canaux d'entrée.
        :param output_channels: ``int`` - Nombre de canaux de sortie.
        :param quantum_number: ``int`` - La taille d'un noyau unique.
        :param stride: ``tuple`` - Le pas, par défaut (1,1).
        :param padding: ``tuple`` - Le bourrage, par défaut (0,0).
        :param kernel_initializer: ``callable`` - Par défaut, distribution normale.
        :param machine: ``str`` - La machine virtuelle à utiliser, par défaut simulation CPU.
        :param dtype: Le type de données du paramètre, par défaut : None, utilise le type de données par défaut : kfloat32, représentant des nombres à virgule flottante 32 bits.
        :param name: Le nom de ce module, par défaut "".

        :return: Couche de convolution quantique.


        Example::

            from pyvqnet.tensor import tensor
            from pyvqnet.qnn.qcnn.qconv import QConv
            x = tensor.ones([1,3,4,4])
            layer = QConv(input_channels=3, output_channels=2, quantum_number=4, stride=(2, 2))
            y = layer(x)
            print(y)

            # [
            # [[[-0.0889078, -0.0889078],
            #  [-0.0889078, -0.0889078]],
            # [[0.7992646, 0.7992646],
            #  [0.7992646, 0.7992646]]]
            # ]

Portes logiques quantiques
**************************

La manière de traiter les bits quantiques est la porte logique quantique. En utilisant une porte logique quantique, nous faisons évoluer consciemment les états quantiques. La porte logique quantique est la base de l'algorithme quantique.

Porte logique quantique de base
===============================

Dans cette section, nous utilisons les différentes portes logiques de pyqpanda3 développées par Origin Quantum pour construire des circuits quantiques et effectuer des simulations quantiques.
Les portes logiques actuellement prises en charge par pyQPanda3 peuvent être consultées dans la définition des portes logiques quantiques de pyQPanda3.
De plus, VQNet encapsule également certaines combinaisons de portes logiques quantiques couramment utilisées en apprentissage automatique quantique :


BasicEmbeddingCircuit
=====================

.. py:function:: pyvqnet.qnn.pq3.template.BasicEmbeddingCircuit(input_feat,qlist)
    
    Encode n caractéristiques binaires dans l'état fondamental de n qubits.

    Par exemple, pour ``features=([0, 1, 1])``, l'état fondamental est :math:`|011 \rangle` dans un système quantique.

    :param input_feat: ``(n)`` entrée binaire de taille n.
    :param qlist: qubits pour construire le circuit modèle.
    :return: circuit quantique.


    Example::

            from pyvqnet.qnn.pq3.template import BasicEmbeddingCircuit
            import pyqpanda3.core as pq
            from pyvqnet import tensor
            input_feat = tensor.QTensor([1,1,0])
            
            qlist = range(3)
            circuit = BasicEmbeddingCircuit(input_feat,qlist)
            print(circuit)


AngleEmbeddingCircuit
=====================

.. py:function:: pyvqnet.qnn.pq3.template.AngleEmbeddingCircuit(input_feat,qubits,rotation:str='X')

    Encode :math:`N` caractéristiques dans l'angle de rotation de :math:`n` qubits, où :math:`N \leq n`.

    La rotation peut être choisie : 'X', 'Y', 'Z', comme défini par le paramètre ``rotation`` :

    * ``rotation='X'`` Utilise la caractéristique comme angle de la rotation RX.

    * ``rotation='Y'`` Utilise la caractéristique comme angle de la rotation RY.

    * ``rotation='Z'`` Utilise la caractéristique comme angle de la rotation RZ.

    La longueur de ``features`` doit être inférieure ou égale au nombre de qubits. Si la longueur de ``features`` est inférieure au nombre de qubits, le circuit n'applique pas les portes de rotation restantes.

    :param input_feat: tableau numpy représentant les paramètres.
    :param qubits: indices des qubits.
    :param rotation: quelle rotation utiliser, par défaut « X ».
    :return: circuit quantique.

    Example::

        from pyvqnet.qnn.pq3.template import AngleEmbeddingCircuit
        import numpy as np 
        m_qlist = range(2)

        input_feat = np.array([2.2, 1])
        C = AngleEmbeddingCircuit(input_feat,m_qlist,'X')
        print(C)
        C = AngleEmbeddingCircuit(input_feat,m_qlist,'Y')
        print(C)
        C = AngleEmbeddingCircuit(input_feat,m_qlist,'Z')
        print(C)

IQPEmbeddingCircuits
====================

.. py:function:: pyvqnet.qnn.pq3.template.IQPEmbeddingCircuits(input_feat,qubits,rep:int = 1)
    
    Encode :math:`n` caractéristiques dans :math:`n` qubits en utilisant des portes diagonales d'un circuit IQP.

    L'encodage est proposé par `Havlicek et al. (2018) <https://arxiv.org/pdf/1804.11326.pdf>`_.

    Le circuit IQP de base peut être répété en spécifiant ``n_repeats``.

    :param input_feat: tableau numpy représentant les paramètres.
    :param qubits: liste des indices des qubits.
    :param rep: Répète le bloc de circuit quantique, le nombre de répétitions par défaut est 1.
    :return: circuit quantique.

    Example::

        import numpy as np
        from pyvqnet.qnn.pq3.template import IQPEmbeddingCircuits
        input_feat = np.arange(1,100)
        qlist = range(3)
        circuit = IQPEmbeddingCircuits(input_feat,qlist,rep = 3)
        print(circuit)


RotCircuit
==========

.. py:function:: pyvqnet.qnn.pq3.template.RotCircuit(para,qubits)

    Rotation arbitraire d'un seul qubit. Le nombre de qlist doit être 1, et le nombre de paramètres doit être 3.

    .. math::

        R(\phi,\theta,\omega) = RZ(\omega)RY(\theta)RZ(\phi)= \begin{bmatrix}
        e^{-i(\phi+\omega)/2}\cos(\theta/2) & -e^{i(\phi-\omega)/2}\sin(\theta/2) \\
        e^{-i(\phi-\omega)/2}\sin(\theta/2) & e^{i(\phi+\omega)/2}\cos(\theta/2)
        \end{bmatrix}.

    :param para: tableau numpy représentant les paramètres :math:`[\phi, \theta, \omega]`.
    :param qubits: indice du qubit, seuls les qubits uniques sont acceptés.
    :return: circuit quantique.

    Example::

        from pyvqnet.qnn.pq3.template import RotCircuit
        import pyqpanda3.core as pq
        from pyvqnet import tensor

        m_qlist = 1

        param =tensor.QTensor([3,4,5])
        c = RotCircuit(param,m_qlist)
        print(c)

CRotCircuit
===========

.. py:function:: pyvqnet.qnn.pq3.template.CRotCircuit(para,control_qubits,rot_qubits)

    Opérateur Rot contrôlé.

    .. math:: 
        
        CR(\phi, \theta, \omega) = \begin{bmatrix}
        1 & 0 & 0 & 0 \\
        0 & 1 & 0 & 0\\
        0 & 0 & e^{-i(\phi+\omega)/2}\cos(\theta/2) & -e^{i(\phi-\omega)/2}\sin(\theta/2)\\
        0 & 0 & e^{-i(\phi-\omega)/2}\sin(\theta/2) & e^{i(\phi+\omega)/2}\cos(\theta/2)
        \end{bmatrix}.

    :param para: Un tableau numpy représentant les paramètres :math:`[\phi, \theta, \omega]`.
    :param control_qubits: indice du qubit de contrôle, le nombre de qubits doit être 1.
    :param rot_qubits: indice du qubit de rotation, le nombre de qubits doit être 1.
    :return: circuit quantique.

    Example::

        from pyvqnet.qnn.pq3.template import CRotCircuit
        import pyqpanda3.core as pq
        import numpy as np
        m_qlist = range(1)
        control_qlist = [1]
        param = np.array([3,4,5])
        cir = CRotCircuit(param,control_qlist,m_qlist)

        print(cir)


CSWAPcircuit
============

.. py:function:: pyvqnet.qnn.pq3.template.CSWAPcircuit(qubits)

    Circuit SWAP contrôlé.

    .. math:: 
        
        CSWAP = \begin{bmatrix}
        1 & 0 & 0 & 0 & 0 & 0 & 0 & 0 \\
        0 & 1 & 0 & 0 & 0 & 0 & 0 & 0 \\
        0 & 0 & 1 & 0 & 0 & 0 & 0 & 0 \\
        0 & 0 & 0 & 1 & 0 & 0 & 0 & 0 \\
        0 & 0 & 0 & 0 & 1 & 0 & 0 & 0 \\
        0 & 0 & 0 & 0 & 0 & 0 & 1 & 0 \\
        0 & 0 & 0 & 0 & 0 & 1 & 0 & 0 \\
        0 & 0 & 0 & 0 & 0 & 0 & 0 & 1
        \end{bmatrix}.

    .. note:: 
        
        Le premier qubit fourni correspond au **qubit de contrôle**.

    :param qubits: liste des indices des qubits. Le premier qubit est le qubit de contrôle. La longueur de qlist doit être 3.
    :return: Le circuit quantique.

    Example::

        from pyvqnet.qnn.pq3 import CSWAPcircuit
        import pyqpanda3.core as pq
        m_machine = pq.CPUQVM()

        m_qlist = range(3)

        c =CSWAPcircuit([m_qlist[1],m_qlist[2],m_qlist[0]])
        print(c)


Controlled_Hadamard
===================

.. py:function:: pyvqnet.qnn.pq3.template.Controlled_Hadamard(qubits)
    
    Porte logique Hadamard contrôlée

    .. math:: 
        CH = \begin{bmatrix}
        1 & 0 & 0 & 0 \\
        0 & 1 & 0 & 0 \\
        0 & 0 & \frac{1}{\sqrt{2}} & \frac{1}{\sqrt{2}} \\
        0 & 0 & \frac{1}{\sqrt{2}} & -\frac{1}{\sqrt{2}}
        \end{bmatrix}.

    :param qubits: indice du qubit.

    Examples::

        import pyqpanda3.core as pq

        machine = pq.CPUQVM()
        
        qubits =range(2)
        from pyvqnet.qnn.pq3 import Controlled_Hadamard

        cir = Controlled_Hadamard(qubits)
        print(cir)

CCZ
===

.. py:function:: pyvqnet.qnn.pq3.template.CCZ(qubits)

    Porte logique Z contrôlée-contrôlée.

    .. math::

        CCZ =
        \begin{pmatrix}
        1 & 0 & 0 & 0 & 0 & 0 & 0 & 0\\
        0 & 1 & 0 & 0 & 0 & 0 & 0 & 0 & 0\\
        0 & 0 & 1 & 0 & 0 & 0 & 0 & 0 & 0\\
        0 & 0 & 0 & 1 & 0 & 0 & 0 & 0\\
        0 & 0 & 0 & 0 & 0 & 1 & 0 & 0 & 0\\
        0 & 0 & 0 & 0 & 0 & 0 & 1 & 0 & 0\\
        0 & 0 & 0 & 0 & 0 & 0 & 0 & 1 & 0 & 0\\
        0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & -1
        \end{pmatrix}

    :param qubits: indice du qubit.

    :return:
        QCircuit pyQPanda3

    Example::

        import pyqpanda3.core as pq

        machine = pq.CPUQVM()

        qubits = range(3)

        from pyvqnet.qnn.pq3 import CCZ

        cir = CCZ(qubits)


FermionicSingleExcitation
=========================

.. py:function:: pyvqnet.qnn.pq3.template.FermionicSingleExcitation(weight, wires, qubits)

    Opérateur d'excitation simple de cluster couplé pour l'exponentiation de produits tensoriels de matrices de Pauli. La forme matricielle est donnée par :

    .. math::

        \hat{U}_{pr}(\theta) = \mathrm{exp} \{ \theta_{pr} (\hat{c}_p^\dagger \hat{c}_r
        -\mathrm{H.c.}) \},

    :param weight: paramètre variable sur le qubit p.
    :param wires: représente un sous-ensemble d'indices de qubits dans l'intervalle [r, p]. La longueur minimale doit être 2. La première valeur d'indice est interprétée comme r et la dernière comme p.
        Les indices intermédiaires sont traités par des portes CNOT pour calculer la parité de l'ensemble de qubits.
    :param qubits: indices des qubits.

    :return:
        QCircuit pyQPanda3

    Examples::

        from pyvqnet.qnn.pq3 import FermionicSingleExcitation, expval

        weight=0.5
        import pyqpanda3.core as pq
        machine = pq.CPUQVM()

        qlists = range(3)

        cir = FermionicSingleExcitation(weight, [1, 0, 2], qlists)


FermionicDoubleExcitation
=========================

.. py:function:: pyvqnet.qnn.pq3.template.FermionicDoubleExcitation(weight, wires1, wires2, qubits)

    Opérateur d'excitation double de cluster couplé pour le produit tensoriel de matrices de Pauli, la forme matricielle est donnée par :

    .. math::

        \hat{U}_{pqrs}(\theta) = \mathrm{exp} \{ \theta (\hat{c}_p^\dagger \hat{c}_q^\dagger
        \hat{c}_r \hat{c}_s - \mathrm{H.c.}) \},

    où :math:`\hat{c}` et :math:`\hat{c}^\dagger` sont les opérateurs d'annihilation et
    de création de fermions et les indices :math:`r, s` et :math:`p, q` sur les orbitales moléculaires
    occupées et vides respectivement. En utilisant la `transformation de Jordan-Wigner
    <https://arxiv.org/abs/1208.5986>`_, l'opérateur fermionique défini ci-dessus peut être écrit
    en termes de matrices de Pauli (voir
    `arXiv:1805.04340 <https://arxiv.org/abs/1805.04340>`_ pour plus de détails)

    .. math::

        \hat{U}_{pqrs}(\theta) = \mathrm{exp} \Big\{
        \frac{i\theta}{8} \bigotimes_{b=s+1}^{r-1} \hat{Z}_b \bigotimes_{a=q+1}^{p-1}
        \hat{Z}_a (\hat{X}_s \hat{X}_r \hat{Y}_q \hat{X}_p +
        \hat{Y}_s \hat{X}_r \hat{Y}_q \hat{Y}_p +\\ \hat{X}_s \hat{Y}_r \hat{Y}_q \hat{Y}_p +
        \hat{X}_s \hat{X}_r \hat{X}_q \hat{Y}_p - \mathrm{H.c.} ) \Big\}

    :param weight: paramètre variable
    :param wires1: représente le sous-ensemble de qubits dans l'intervalle [s, r] occupé par la liste d'indices de qubits. Le premier indice est interprété comme s et le dernier comme r. La porte CNOT opère sur les indices intermédiaires pour calculer la parité d'un ensemble de qubits.
    :param wires2: représente le sous-ensemble de qubits dans l'intervalle [q, p] occupé par la liste d'indices de qubits. Le premier indice racine est interprété comme q et le dernier comme p. La porte CNOT opère sur l'indice intermédiaire pour calculer la parité d'un ensemble de qubits.
    :param qubits: indices des qubits.

    :return:
        QCircuit pyQPanda3

    Examples::

        import pyqpanda3.core as pq
        from pyvqnet.qnn.pq3 import FermionicDoubleExcitation, expval
        machine = pq.CPUQVM()
        
        qlists = range(5)
        weight = 1.5
        cir = FermionicDoubleExcitation(weight,
                                        wires1=[0, 1],
                                        wires2=[2, 3, 4],
                                        qubits=qlists)


UCCSD
=====

.. py:function:: pyvqnet.qnn.pq3.template.UCCSD(weights, wires, s_wires, d_wires, init_state, qubits)

    Implémente la simulation d'excitation simple et double de cluster couplé unitaire (UCCSD). UCCSD est une simulation VQE, couramment utilisée pour exécuter des simulations de chimie quantique.

    Dans l'approximation de Trotter au premier ordre, la fonction unitaire UCCSD est donnée par :

    .. math::

        \hat{U}(\vec{\theta}) =
        \prod_{p > r} \mathrm{exp} \Big\{\theta_{pr}
        (\hat{c}_p^\dagger \hat{c}_r-\mathrm{H.c.}) \Big\}
        \prod_{p > q > r > s} \mathrm{exp} \Big\{\theta_{pqrs}
        (\hat{c}_p^\dagger \hat{c}_q^\dagger \hat{c}_r \hat{c}_s-\mathrm{H.c.}) \Big\}
    
    où :math:`\hat{c}` et :math:`\hat{c}^\dagger` sont les opérateurs d'annihilation et

    de création de fermions et les indices :math:`r, s` et :math:`p, q` sont les orbitales moléculaires
    occupées et vides, respectivement. (Pour plus de détails, voir
    `arXiv:1805.04340 <https://arxiv.org/abs/1805.04340>`_) :

    :param weights: tenseur de taille ``(len(s_wires)+ len(d_wires))`` contenant les paramètres
     :math:`\theta_{pr}` et :math:`\theta_{pqrs}` en entrée des rotations Z ``FermionicSingleExcitation`` et ``FermionicDoubleExcitation``.
    :param wires: indices des qubits à modéliser
    :param s_wires: séquence de listes contenant les indices des qubits ``[r,...,p]`` générés par une excitation simple
     :math:`\vert r, p \rangle = \hat{c}_p^\dagger \hat{c}_r \vert \mathrm{HF} \rangle`, où :math:`\vert \mathrm{HF} \rangle` désigne l'état de référence Hartree-Fock.
    :param d_wires: séquence de listes, chacune contenant deux listes. Spécifie les indices ``[s, ...,r]`` et ``[q,..., p]``. Définit l'excitation double :math:`\vert s, r, q, p \rangle = \hat{c}_p^\dagger \hat{c}_q^\dagger \hat{c}_r\hat{c}_s \vert \mathrm{HF} \rangle`.
    :param init_state: vecteur de nombre d'occupation de longueur ``len(wires)`` représentant l'état haute fréquence. ``init_state`` Initialise l'état du qubit.
    :param qubits: Indice du qubit.

    Examples::
        
        import pyqpanda3.core as pq
        from pyvqnet.tensor import tensor
        from pyvqnet.qnn.pq3 import UCCSD, expval
        machine = pq.CPUQVM()
        
        qlists = range(6)
        weight = tensor.zeros([8])
        cir = UCCSD(weight,wires = [0,1,2,3,4,5,6],
                                        s_wires=[[0, 1, 2], [0, 1, 2, 3, 4], [1, 2, 3], [1, 2, 3, 4, 5]],
                                        d_wires=[[[0, 1], [2, 3]], [[0, 1], [2, 3, 4, 5]], [[0, 1], [3, 4]], [[0, 1], [4, 5]]],
                                        init_state=[1, 1, 0, 0, 0, 0],
                                        qubits=qlists)

QuantumPoolingCircuit
=====================

.. py:function:: pyvqnet.qnn.pq3.template.QuantumPoolingCircuit(sources_wires, sinks_wires, params,qubits)

    Circuit quantique qui sous-échantillonne les données.

    Pour réduire le nombre de qubits dans le circuit, créez d'abord des paires de qubits dans le système. Après avoir initialement apparié tous les qubits, appliquez l'opérateur unitaire généralisé à 2 qubits à chaque paire de qubits. Après avoir appliqué ces opérateurs unitaires à deux qubits, ignorez un qubit dans chaque paire pour le reste du réseau neuronal.

    :param sources_wires: Indices des qubits source à ignorer.
    :param sinks_wires: Indices des qubits cibles à conserver.
    :param params: Paramètres d'entrée.
    :param qubits: Indices des qubits.

    :return:
        QCircuit pyQPanda3

    Examples::

        from pyvqnet.qnn.pq3.template import QuantumPoolingCircuit
        import pyqpanda3.core as pq
        from pyvqnet import tensor

        qlists = range(4)
        p = tensor.full([6], 0.35)
        cir = QuantumPoolingCircuit([0, 1], [2, 3], p, qlists)
        print(cir)

Combinaisons de circuits quantiques courantes
*********************************************
VQNet fournit certains circuits quantiques couramment utilisés dans la recherche en apprentissage automatique quantique

HardwareEfficientAnsatz
=======================

.. py:class:: pyvqnet.qnn.pq3.ansatz.HardwareEfficientAnsatz(qubits,single_rot_gate_list,entangle_gate="CNOT",entangle_rules='linear',depth=1)

    Implémentation de l'Ansatz Efficace pour le Matériel présenté dans l'article : `Hardware-efficient Variational Quantum Eigensolver for Small Molecules <https://arxiv.org/pdf/1704.05018.pdf>`__ .

    :param qubits: indice du qubit.
    :param single_rot_gate_list: Liste de portes de rotation à qubit unique comprenant une ou plusieurs portes de rotation agissant sur chaque qubit. Actuellement prises en charge : Rx, Ry, Rz.
    :param entangle_gate: Porte d'intrication non paramétrique. Prend en charge CNOT, CZ. Par défaut : CNOT.
    :param entangle_rules: Comment la porte d'intrication est utilisée dans le circuit. ``linear`` signifie que la porte d'intrication agira sur chaque qubit adjacent. ``all`` signifie que la porte d'intrication agira sur deux qubits quelconques. Par défaut : ``linear``.
    :param depth: Profondeur de l'ansatz, par défaut : 1.

    :return:
        Une instance de HardwareEfficientAnsatz

    Example::

        import pyqpanda3.core as pq
        from pyvqnet.tensor import QTensor,tensor
        from pyvqnet.qnn.pq3.ansatz import HardwareEfficientAnsatz
        machine = pq.CPUQVM()

        qlist = range(4)
        c = HardwareEfficientAnsatz(qlist,["rx", "RY", "rz"],
                                entangle_gate="cnot",
                                entangle_rules="linear",
                                depth=1)
        w = tensor.ones([c.get_para_num()])

        cir = c.create_ansatz(w)
        print(cir)

BasicEntanglerTemplate
======================

.. py:class:: pyvqnet.qnn.pq3.template.BasicEntanglerTemplate(weights=None, num_qubits=1, rotation=pyqpanda3.RX)
    
    Une couche composée de rotations à paramètre unique sur chaque qubit, suivie de plusieurs portes CNOT combinées en une chaîne fermée ou un anneau.
    L'anneau de portes CNOT connecte chaque qubit à ses voisins, le dernier qubit étant considéré comme un voisin du premier.
    Le nombre de couches :math:`L` est déterminé par la première dimension du paramètre ``weights``.

    :param weights: Un tenseur de poids de forme ``(L, len(qubits))``. Chaque poids est utilisé comme paramètre dans une porte quantique paramétrique. La valeur par défaut est : ``None``, alors des nombres aléatoires distribués normalement ``(1,1)`` sont utilisés comme poids.
    :param num_qubits: Le nombre de qubits, par défaut 1.
    :param rotation: Utilise une porte à qubit unique à paramètre unique, ``pyqpanda3.RX`` est utilisé comme valeur par défaut.
    :return:
        Une instance de BasicEntanglerTemplate

    Example::

        import pyqpanda3.core as pq
        import numpy as np
        from pyvqnet.qnn.pq3 import BasicEntanglerTemplate
        np.random.seed(42)
        num_qubits = 5
        shape = [1, num_qubits]
        weights = np.random.random(size=shape)

        machine = pq.CPUQVM()

        qubits = range(num_qubits)

        circuit = BasicEntanglerTemplate(weights=weights, num_qubits=num_qubits, rotation=pq.RZ)
        result = circuit.compute_circuit()
        circuit.print_circuit(qubits)


StronglyEntanglingTemplate
==========================

.. py:class:: pyvqnet.qnn.pq3.template.StronglyEntanglingTemplate(weights=None, num_qubits=1, ranges=None)

    Couches composées de rotations à qubit unique et d'intricateurs, comme dans `circuit-centric classifier design <https://arxiv.org/abs/1804.00633>`__ .
    Le paramètre ``weights`` contient les poids de chaque couche. Ainsi, le nombre de couches :math:`L` est égal à la première dimension de ``weights``.
    Il contient des portes CNOT à 2 qubits agissant sur :math:`M` qubits, :math:`i = 1,...,M`. Le deuxième numéro de qubit de chaque porte est donné par la formule :math:`(i+r)\mod M`, où :math:`r` est un hyperparamètre appelé ``range``, et :math:`0 < r < M`.

    :param weights: Tenseur de poids de forme ``(L, M, 3)``, valeur par défaut : None, utilise un tenseur aléatoire de forme ``(1,1,3)``.
    :param num_qubits: Nombre de qubits, valeur par défaut : 1.
    :param ranges: Séquence qui détermine les hyperparamètres de plage pour chaque couche suivante ; valeur par défaut : None, utilise :math:`r=l \mod M` comme valeur de ranges.
    :return:
        Une instance de StronglyEntanglingTemplate

    Example::

        from pyvqnet.qnn.pq3 import StronglyEntanglingTemplate
        import pyqpanda3.core as pq
        from pyvqnet.tensor import *
        import numpy as np
        np.random.seed(42)
        num_qubits = 3
        shape = [2, num_qubits, 3]
        weights = np.random.random(size=shape)

        machine = pq.CPUQVM()

        qubits = range(num_qubits)

        circuit = StronglyEntanglingTemplate(weights, num_qubits=num_qubits )
        result = circuit.compute_circuit()
        print(result)
        circuit.print_circuit(qubits)


ComplexEntanglingTemplate
=========================

.. py:class:: pyvqnet.qnn.pq3.ComplexEntanglingTemplate(weights,num_qubits,depth)

    Couche fortement intriguée composée de portes U3 et de portes CNOT.
    Ce modèle de circuit provient de l'article suivant : https://arxiv.org/abs/1804.00633.

    :param weights: paramètres, forme [depth, num_qubits, 3]
    :param num_qubits: nombre de qubits.
    :param depth: profondeur du sous-circuit.
    :return:
        Une instance de ComplexEntanglingTemplate

    Example::

        from pyvqnet.qnn.pq3 import ComplexEntanglingTemplate
        import pyqpanda3.core as pq
        from pyvqnet.tensor import *
        depth=3
        num_qubits = 8
        shape = [depth, num_qubits, 3]
        weights = tensor.randn(shape)

        machine = pq.CPUQVM()

        qubits = range(num_qubits)

        circuit = ComplexEntanglingTemplate(weights, num_qubits=num_qubits,depth=depth)
        result = circuit.create_circuit(qubits)
        circuit.print_circuit(qubits)

Quantum_Embedding
=================

.. py:class:: pyvqnet.qnn.pq3.Quantum_Embedding(qubits, machine, num_repetitions_input, depth_input, num_unitary_layers, num_repetitions)

    Utilise RZ, RY, RZ pour créer un circuit quantique variationnel afin d'encoder des données classiques dans des états quantiques.
    Référence `Quantum embeddings for machine learning <https://arxiv.org/abs/2001.03622>`_.

    Après avoir initialisé la classe, sa fonction membre ``compute_circuit`` est la fonction d'exécution, qui peut être utilisée comme paramètre d'entrée de la classe ``QuantumLayerV2`` pour former une couche du modèle d'apprentissage automatique quantique.

    :param qubits: Les bits quantiques demandés par pyQPanda3.
    :param machine: Machine virtuelle quantique appliquée par pyQPanda3.
    :param num_repetitions_input: Le nombre de répétitions de l'encodage de l'entrée dans le sous-module.
    :param depth_input: La dimension des caractéristiques des données d'entrée.
    :param num_unitary_layers: Le nombre de répétitions de la porte quantique variationnelle dans chaque sous-module.
    :param num_repetitions: Le nombre de répétitions du sous-module.
    :return:
        Une instance de Quantum_Embedding

    Example::

        from pyvqnet.qnn.pq3 import QuantumLayerV2,Quantum_Embedding
        from pyvqnet.tensor import tensor
        import pyqpanda3.core as pq
        depth_input = 2
        num_repetitions = 2
        num_repetitions_input = 2
        num_unitary_layers = 2

        loacl_machine = pq.CPUQVM()

        nq = depth_input * num_repetitions_input
        qubits = range(nq)
        cubes = range(nq)

        data_in = tensor.ones([12, depth_input])
        data_in.requires_grad = True

        qe = Quantum_Embedding(nq, loacl_machine, num_repetitions_input,
        depth_input, num_unitary_layers, num_repetitions)
        qlayer = QuantumLayerV2(qe.compute_circuit,
        qe.param_num)

        data_in.requires_grad = True
        y = qlayer.forward(data_in)
        y.backward()
        print(data_in.grad)


Mesure des circuits quantiques
*******************************

expval
======

.. py:function:: pyvqnet.qnn.pq3.measure.expval(machine,prog,pauli_str_dict,shots=1000,noise_model=None)

    La valeur attendue de l'observation hamiltonienne fournie.

    Si l'observation est :math:`0.7Z\otimes X\otimes I+0.2I\otimes Z\otimes I`,
    alors le dictionnaire hamiltonien sera ``{{'Z0, X1':0.7} ,{'Z1':0.2}}``.

    L'API expval prend désormais en charge le simulateur pyQPanda3.

    :param machine: La machine quantique créée par pyQPanda3.
    :param prog: Le programme quantique créé par pyQPanda3.
    :param pauli_str_dict: Valeur observée de l'hamiltonien.
    :param shots: Nombre de mesures, par défaut 1000.
    :param noise_model: Modèle de bruit à appliquer, par défaut None (simulation idéale).

    :return: valeur attendue.

    Example::

        import pyqpanda3.core as pq
        from pyvqnet.qnn.pq3.measure import expval
        input = [0.56, 0.1]
        m_machine = pq.CPUQVM()

        m_qlist = range(3)
        cir = pq.QCircuit(3)
        cir<<pq.RZ(m_qlist[0],input[0])
        cir<<pq.CNOT(m_qlist[0],m_qlist[1])
        cir<<pq.RY(m_qlist[1],input[1])
        cir<<pq.CNOT(m_qlist[0],m_qlist[2])
        m_prog = pq.QProg(cir)

        pauli_dict = {'Z0 X1':10,'Y2':-0.543}
        exp2 = expval(m_machine,m_prog,pauli_dict)
        print(exp2)
 


QuantumMeasure
==============

.. py:function:: pyvqnet.qnn.pq3.measure.QuantumMeasure(machine,prog,measure_qubits:list,shots:int = 1000, qcloud_option="",noise_model=None)

    Calcule les mesures du circuit quantique. Renvoie les mesures obtenues par des méthodes de Monte Carlo.

    Pour plus de détails sur la mesure, veuillez vous référer à la documentation pyQPanda3.

    L'API QuantumMeasure ne prend actuellement en charge que ``CPUQVM`` ou ``QCloud`` de pyQPanda3.

    :param machine: La machine virtuelle quantique allouée par pyQPanda3.
    :param prog: Le programme quantique créé par pyQPanda3.
    :param measure_qubits: Liste contenant les indices des bits de mesure.
    :param shots: Le nombre de mesures, la valeur par défaut est 1000.
    :param qcloud_option: Définit la configuration qcloud, la valeur par défaut est "", vous pouvez passer une classe QCloudOptions, qui n'est utile que lors de l'utilisation de qcloud.
    :param noise_model: Modèle de bruit à appliquer, par défaut None (simulation idéale).
    :return: Renvoie les résultats de mesure obtenus par la méthode de Monte Carlo.

    Example::

        from pyqpanda3.core import *
        circuit = QCircuit(3)
        circuit << H(0)
        circuit << P(2, 0.2)
        circuit << RX(1, 0.9)
        circuit << RX(0, 0.6)
        circuit << RX(1, 0.3)
        circuit << RY(1, 0.3)
        circuit << RY(2, 2.7)
        circuit << RX(0, 1.5)

        prog = QProg()
        prog.append(circuit)

        machine = CPUQVM()


        from pyvqnet.qnn.pq3.measure import probs_measure,quantum_measure

        measure_result = quantum_measure(machine,prog,[2,0])
        print(measure_result)


ProbsMeasure
============

.. py:function:: pyvqnet.qnn.pq3.measure.ProbsMeasure(machine,prog,measure_qubits:list,shots=1,noise_model=None)

    Calcule les mesures de probabilité du circuit.

    Pour plus de détails, veuillez vous référer à la documentation pyQPanda3 sur la mesure de probabilité.

    L'API ProbsMeasure ne prend actuellement en charge que ``CPUQVM`` ou ``QCloud`` de pyQPanda3.

    :param measure_qubits: Liste contenant les indices des qubits de mesure.
    :param prog: Le programme quantique créé par qpanda.
    :param machine: La machine virtuelle quantique allouée par pyQPanda.
    :param shots: Nombre de mesures, par défaut 1, ce qui calcule la valeur théorique.
    :param noise_model: Modèle de bruit à appliquer, par défaut None (simulation idéale).
    :return: Mesure les qubits dans l'ordre lexicographique.


    Example::

        from pyqpanda3.core import *
        from pyvqnet.qnn.pq3.measure import probs_measure
        circuit = QCircuit(3)
        circuit << H(0)
        circuit << P(2, 0.2)
        circuit << RX(1, 0.9)
        circuit << RX(0, 0.6)
        circuit << RX(1, 0.3)
        circuit << RY(1, 0.3)
        circuit << RY(2, 2.7)
        circuit << RX(0, 1.5)

        prog = QProg()
        prog.append(circuit)
        prog.append(measure(0, 0))
        prog.append(measure(1, 1))
        prog.append(measure(2, 2))

        machine = CPUQVM()

        measure_result = probs_measure(machine,prog,[2,0])
        print(measure_result)
        #[0.04796392899146941, 0, 0.4760180355042653, 0.4760180355042653]


DensityMatrixFromQstate
=======================

.. py:function:: pyvqnet.qnn.pq3.measure.DensityMatrixFromQstate(state, indices)
    
    Calcule la matrice de densité d'un état quantique sur un ensemble spécifique de qubits.

    :param state: Liste 1D de vecteurs d'état. La taille de cette liste doit être ``(2**N,)``. Pour un nombre de qubits ``N``, qstate doit commencer de 000 -> 111.
    :param indices: Liste des indices des qubits dans le sous-système considéré.
    :return: 
        Matrice de densité de taille ``(2**len(indices), 2**len(indices))``.

    Example::

        from pyvqnet.qnn.pq3.measure import DensityMatrixFromQstate
        qstate = [(0.9306699299765968+0j), (0.18865613455240968+0j), (0.1886561345524097+0j), (0.03824249173404786+0j), -0.048171819846746615j, -0.00976491131165138j, -0.23763904794287155j, -0.048171819846746615j]
        print(DensityMatrixFromQstate(qstate,[0,1]))
        # [[0.86846704+0.j 0.1870241 +0.j 0.17604699+0.j 0.03791166+0.j]
        # [0.1870241 +0.j 0.09206345+0.j 0.03791166+0.j 0.01866219+0.j]
        # [0.17604699+0.j 0.03791166+0.j 0.03568649+0.j 0.00768507+0.j]
        # [0.03791166+0.j 0.01866219+0.j 0.00768507+0.j 0.00378301+0.j]]


VN_Entropy
==========

.. py:function:: pyvqnet.qnn.pq3.measure.VN_Entropy(state, indices, base=None)
    
    Calcule l'entropie de Von Neumann à partir d'un vecteur d'état sur une liste donnée de qubits.

    .. math::

        S( \rho ) = -\text{Tr}( \rho \log ( \rho ))

    :param state: Liste 1D de vecteurs d'état. La taille de cette liste doit être ``(2**N,)``. Pour un nombre de qubits ``N``, qstate doit commencer de 000 -> 111.
    :param indices: Liste des indices des qubits dans le sous-système considéré.
    :param base: Base du logarithme. Si None, le logarithme népérien est utilisé.

    :return: valeur flottante de l'entropie de von Neumann.

    Example::

        from pyvqnet.qnn.pq3.measure import VN_Entropy
        qstate = [(0.9022961387408862 + 0j), -0.06676534788028633j,
        (0.18290448232350312 + 0j), -0.3293638014158896j,
        (0.03707657410649268 + 0j), -0.06676534788028635j,
        (0.18290448232350312 + 0j), -0.013534006039561714j] 
        print(VN_Entropy(qstate, [0, 1]))
        #0.14592917648464448

Mutal_Info
==========

.. py:function:: pyvqnet.qnn.pq3.measure.Mutal_Info(state, indices0, indices1, base=None)

    Calcule l'information mutuelle à partir du vecteur d'état sur deux listes de sous-qubits.

    .. math::

        I(A, B) = S(\rho^A) + S(\rho^B) - S(\rho^{AB})
        où :math:`S` est l'entropie de von Neumann.

    L'information mutuelle est une mesure de la corrélation entre deux sous-systèmes. Plus précisément, elle quantifie la quantité d'information qu'un système peut acquérir en mesurant l'autre.

    Chaque état peut être donné comme un vecteur d'état dans la base de calcul.

    :param state: Liste 1D de vecteurs d'état. La taille de cette liste doit être ``(2**N,)``. Pour un nombre de qubits ``N``, qstate doit commencer de 000 -> 111.
    :param indices0: Liste des indices des qubits dans le premier sous-système.
    :param indices1: Liste des indices des qubits dans le second sous-système.
    :param base: Base des logarithmes. Si None, les logarithmes népériens sont utilisés. Par défaut, None.

    :return: Information mutuelle entre les sous-systèmes

    Example::

        from pyvqnet.qnn.pq3.measure import Mutal_Info
        qstate = [(0.9022961387408862 + 0j), -0.06676534788028633j,
        (0.18290448232350312 + 0j), -0.3293638014158896j,
        (0.03707657410649268 + 0j), -0.06676534788028635j,
        (0.18290448232350312 + 0j), -0.013534006039561714j]
        print(Mutal_Info(qstate, [0], [2], 2))
        #0.13763425302805887

Purity
======

.. py:function:: pyvqnet.qnn.pq3.measure.Purity(state, qubits_idx)

    Calcule la pureté d'un qubit particulier à partir du vecteur d'état.

    .. math::

        \gamma = \text{Tr}(\rho^2)
        
    où :math:`\rho` est la matrice de densité. La pureté d'un état quantique normalisé satisfait :math:`\frac{1}{d} \leq \gamma \leq 1`,
    où :math:`d` est la dimension de l'espace de Hilbert.
    La pureté d'un état pur est 1.

    :param state: état quantique obtenu depuis pyqpanda3
    :param qubits_idx: indice du qubit pour lequel la pureté doit être calculée

    :return:
        pureté

    Examples::

        from pyvqnet.qnn.pq3.measure import Purity
        qstate = [(0.9306699299765968 + 0j), (0.18865613455240968 + 0j),
        (0.1886561345524097 + 0j), (0.03824249173404786 + 0j),
        -0.048171819846746615j, -0.00976491131165138j, -0.23763904794287155j,
        -0.048171819846746615j]
        pp = Purity(qstate, [1])
        print(pp)
        #0.902503479761881