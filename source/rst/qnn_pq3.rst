Usare il modulo di apprendimento automatico quantistico pyQPanda3
#################################################################

.. warning::

    La parte di calcolo quantistico della seguente interfaccia utilizza pyqpanda3.

    Se si utilizza la funzione QCloud in questo modulo, si verificheranno errori durante l'importazione di pyqpanda2 nel codice o l'utilizzo dell'interfaccia del pacchetto pyqpanda2 di pyvqnet.

Strato di calcolo quantistico
***********************************

.. _QuantumLayer_pq3:

QuantumLayer
============================

Se si ha familiarità con la sintassi di pyQPanda3, si può utilizzare l'interfaccia QuantumLayer per personalizzare il simulatore pyqpanda3 per il calcolo.

.. py:class:: pyvqnet.qnn.pq3.quantumlayer.QuantumLayer(qprog_with_measure,para_num,diff_method:str = "parameter_shift",delta:float = 0.01,dtype=None,name="")

    Modulo di calcolo astratto del layer variazionale quantistico. Utilizza pyQPanda3 per simulare un circuito quantistico parametrizzato e ottenere i risultati di misurazione. Questo layer variazionale quantistico eredita il modulo di calcolo del gradiente del framework VQNet. Può utilizzare il metodo di spostamento dei parametri per calcolare il gradiente dei parametri del circuito, addestrare modelli di circuiti quantistici variazionali o incorporare circuiti quantistici variazionali in modelli quantistici e classici ibridi.

    :param qprog_with_measure: Funzione di operazione del circuito quantistico e misurazione costruita con pyQPanda.
    :param para_num: ``int`` - Numero di parametri.
    :param diff_method: Metodo per risolvere i gradienti dei parametri del circuito quantistico, "parameter_shift" o "finite_difference", default spostamento dei parametri.
    :param delta: \delta durante il calcolo dei gradienti per differenze finite.
    :param dtype: Tipo di dato del parametro, default: None, usa il tipo di dato predefinito: kfloat32, che rappresenta numeri a virgola mobile a 32 bit.
    :param name: Nome di questo modulo, default "".

    :return: Un modulo in grado di calcolare circuiti quantistici.

    .. note::

        qprog_with_measure è una funzione di circuito quantistico definita in pyQPanda3.

        Questa funzione deve contenere due parametri, input e param, come input della funzione (anche se un parametro non viene effettivamente utilizzato), e l'output è il risultato di misurazione o il valore atteso del circuito (deve essere np.ndarray o una lista contenente valori), altrimenti non funzionerà correttamente in QpandaQCircuitVQCLayerLite.

        L'uso della funzione di circuito quantistico qprog_with_measure (input, param) può essere consultato nell'esempio seguente.

        ``input``: Dati classici unidimensionali in input. Se non presenti, inserire None

        ``param``: Parametri del circuito quantistico variazionale unidimensionali da addestrare

    .. note::

        Questa classe ha alias ``QuantumLayerV2``, ``QpandaQCircuitVQCLayerLite``.

    Esempio::

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

        #dati classici come input
        input = QTensor([[1,2,3,4],[4,2,2,3],[3.0,3,2,2]] )

        #circuiti forward
        rlt = pqc(input)

        print(rlt)

        grad = ones(rlt.data.shape)*1000
        #circuiti backward
        rlt.backward(grad)

        print(pqc.m_para.grad)

QpandaQProgVQCLayer
=============================

.. py:class:: pyvqnet.qnn.pq3.quantumlayer.QuantumLayerV3(origin_qprog_func,para_num,qvm_type="cpu", pauli_str_dict=None, shots=1000, initializer=None,dtype=None,name="")

    Invia il circuito quantistico parametrizzato al simulatore a piena ampiezza locale QPanda3 per il calcolo e addestra i parametri nel circuito.
    Supporta dati in batch e utilizza la regola dello spostamento dei parametri per stimare il gradiente dei parametri.
    Per CRX, CRY, CRZ, questo layer utilizza la formula in https://iopscience.iop.org/article/10.1088/1367-2630/ac2cb3, e il resto delle porte logiche utilizza il metodo predefinito di spostamento dei parametri per calcolare il gradiente.

    :param origin_qprog_func: La funzione di circuito quantistico chiamabile costruita da QPanda.
    :param para_num: ``int`` - Numero di parametri; i parametri sono unidimensionali.
    :param qvm_type: ``str`` - Tipo di simulatore pyqpanda3 da utilizzare, ``cpu`` o ``gpu``, default ``cpu``.
    :param pauli_str_dict: ``dict|list`` - Dizionario o lista di dizionari che rappresentano gli operatori di Pauli nel circuito quantistico. Default è None.
    :param shots: ``int`` - Numero di misurazioni. Default è 1000.
    :param initializer: Inizializzatore per i valori dei parametri. Default è None.
    :param dtype: Tipo di dato del parametro. Default è None, che significa utilizzare il tipo di dato predefinito.
    :param name: Nome del modulo. Default è la stringa vuota.

    :return: Restituisce una classe QpandaQProgVQCLayer

    .. note::

        origin_qprog_func è una funzione di circuito quantistico definita dall'utente utilizzando pyQPanda3.

        Questa funzione deve contenere due parametri, input e param, come input della funzione (anche se un parametro non viene effettivamente utilizzato), e l'output è di tipo pyqpanda3.core.QProg, altrimenti non può funzionare correttamente in QuantumLayerV3.

        origin_qprog_func (input,param )

        ``input``: dati classici unidimensionali in input definiti dall'utente

        ``param``: parametri del circuito quantistico unidimensionali definiti dall'utente

    .. note::

        Questa classe ha un alias ``QuantumLayerV3``.

    Esempio::

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
==================================

Quando si installa l'ultima versione di pyqpanda3, si può utilizzare questa interfaccia per definire un circuito variazionale e inviarlo al chip reale originqc per l'esecuzione.

.. py:class:: pyvqnet.qnn.pq3.quantumlayer.QuantumBatchAsyncQcloudLayer(origin_qprog_func, qcloud_token, para_num, pauli_str_dict=None, shots = 1000, initializer=None, dtype=None, name="", diff_method="parameter_shift", submit_kwargs={}, query_kwargs={})
    
    Un modulo di calcolo astratto per il chip reale originqc che utilizza pyQPanda3 QCLOUD. Invia circuiti quantistici parametrizzati al chip reale e ottiene i risultati di misurazione.
    Se diff_method == "random_coordinate_descent", il layer selezionerà casualmente un singolo parametro per calcolare il gradiente, e gli altri parametri rimarranno zero. Riferimento: https://arxiv.org/abs/2311.00088

    .. note::

        qcloud_token è il token API ottenuto dalla piattaforma cloud.

        origin_qprog_func deve restituire dati di tipo pyqpanda3.core.QProg. Se pauli_str_dict non è impostato, è necessario assicurarsi che la misura sia stata inserita nel QProg.

        origin_qprog_func deve essere nel formato seguente:

        origin_qprog_func(input,param)

            ``input``: Input di dati classici 1D~2D. Nel caso 2D, la prima dimensione è la dimensione del batch

            ``param``: Input dei parametri da addestrare per il circuito quantistico variazionale 1D

    .. note::

        Nella versione corrente, il timeout totale predefinito per l'invio di un singolo circuito al QCloud è di 60 secondi. Se si verifica un timeout a causa della occupazione di QCloud, è possibile impostare il valore della chiave ``total_timeout`` in ``query_kwargs`` al numero desiderato di secondi di attesa.

    :param origin_qprog_func: La funzione di circuito quantistico variazionale costruita da QPanda, che deve restituire un QProg.
    :param qcloud_token: ``str`` - Il tipo di macchina quantistica o il token cloud utilizzato per l'esecuzione.
    :param para_num: ``int`` - Il numero di parametri, il parametro è un QTensor di dimensione [para_num].
    :param pauli_str_dict: ``dict|list`` - Un dizionario o lista di dizionari che rappresentano gli operatori di Pauli nel circuito quantistico. Il default è "None", che esegue operazioni di misurazione. Se viene inserito un dizionario di operatori di Pauli, verrà calcolato un singolo valore atteso o valori attesi multipli.
    :param shots: ``int`` - Il numero di misurazioni. Il valore predefinito è 1000.
    :param initializer: Inizializzatore per i valori dei parametri. Il default è "None", che utilizza una distribuzione normale 0~2*pi.
    :param dtype: Il tipo di dato del parametro. Il valore predefinito è None, che significa utilizzare il tipo di dato predefinito pyvqnet.kfloat32.
    :param name: Il nome del modulo. Il default è una stringa vuota.
    :param diff_method: Metodo di differenziazione per il calcolo del gradiente. Il default è "parameter_shift", "random_coordinate_descent".
    :param submit_kwargs: Parametri aggiuntivi per l'invio di circuiti quantistici, default: {"if_print_qcloud_log":False,"chip_id":"WK_C180","is_amend":True,"is_mapping":True,"is_optimization":True,"compile_level":3,"default_task_group_size":200,"test_qcloud_fake":False,"":"server_ip_address"}, quando test_qcloud_fake è impostato su True, simulazione locale CPUQVM.
    :param query_kwargs: Parametri aggiuntivi per interrogare i risultati quantistici, default: {"timeout":1,"total_timeout":60,"print_query_info":True,"sub_circuits_split_size":1}.
    :return: Un modulo in grado di calcolare circuiti quantistici.

    Esempio::

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
============================

.. py:class:: pyvqnet.qnn.pq3.quantumlayer.QuantumLayerAdjoint(pq3_vqc_circuit, param_num, pauli_dicts, dtype = None, name="")

    Questa classe utilizza l'interfaccia VQCircuit di pyqpanda3 per calcolare i gradienti dei parametri in un circuito quantistico rispetto all'hamiltoniana utilizzando il metodo aggiunto.

    Questa classe supporta input batch e output hamiltoniani multipli.

    .. note::

        Quando si utilizza questa interfaccia, è necessario costruire il circuito utilizzando porte logiche di VQCircuit.

        Attualmente, solo un numero limitato di porte logiche è supportato; verrà sollevata un'eccezione per quelle non supportate.

        Il parametro di input ``pq3_vqc_circuit`` può contenere solo due parametri, ``x`` e ``param``, che devono essere un array o una lista unidimensionale.

        Nella funzione ``pq3_vqc_circuit``, gli utenti devono utilizzare ``pyqpanda3.vqcircuit.VQCircuit().set_Param`` per personalizzare la gestione degli input e dei parametri.

        Inoltre, gli utenti devono inserire preventivamente il numero di parametri in ``param_num``. Questa interfaccia inizializzerà un parametro ``m_para`` con una lunghezza di ``param_num``.

        Vedere l'esempio seguente.

    :param pq3_vqc_circuit: Personalizza il circuito VQCircuit di pyqpanda3.
    :param param_num: Numero di parametri. :param pauli_dicts: Osservazioni attese, può essere una lista.
    :param dtype: Tipo del parametro, kfloat32 o kfloat64, default: None, usa kfloat32.
    :param name: Il nome di questa interfaccia.
    :return: Restituisce un'istanza di QuantumLayerAdjoint


    Esempio::

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
============================

.. py:class:: pyvqnet.qnn.pq3.vqc_qcloud_layer.VQCQCloudLayer(vqc_module, qcloud_token, pauli_str_dict=None, shots=1000, name="", submit_kwargs={}, query_kwargs={})

    Invia un modulo VQC al chip reale QCloud o al simulatore locale pyqpanda3 per l'esecuzione.

    Propagazione forward: Invece di eseguire il calcolo del circuito quantistico variazionale di VQNet, chiama il chip quantistico reale o il simulatore locale qpanda per il calcolo.

    Propagazione backward: Utilizza la regola dello spostamento dei parametri per calcolare i gradienti. Per ogni dimensione di input e ogni parametro addestrabile nel VQC,
    genera circuiti spostati di +/- pi/2 e li invia per il calcolo, recupera i risultati per calcolare lo jacobiano. I gradienti sono impostati sul tensore di input e sui parametri addestrabili del VQC.

    .. note::

        Il timeout totale predefinito per l'invio di un singolo circuito al QCloud è di 60 secondi. Se si verifica un timeout a causa della occupazione di QCloud, è possibile impostare la chiave ``total_timeout`` in ``query_kwargs`` per specificare i secondi di attesa.

    .. note::

        Non è possibile definire una funzione di misurazione (come ``MeasureAll``) in ``vqc_module``. La misurazione deve essere specificata tramite il parametro ``pauli_str_dict`` per indicare gli osservabili.
        Ad esempio: ``VQCQCloudLayer(vqc_module, token, pauli_str_dict={'Z0': 1, 'Z1': 1})``.

    :param vqc_module: Modulo VQC di VQNet, deve includere una QMachine con save_ir=True.
    :param qcloud_token: Token API QCloud. Passare una stringa vuota se si utilizza un simulatore locale.
    :param pauli_str_dict: Dizionario degli operatori di Pauli per il calcolo del valore atteso. Default è None, che esegue l'operazione di misurazione.
    :param shots: Numero di misurazioni. Default è 1000.
    :param name: Nome del modulo. Default è stringa vuota.
    :param submit_kwargs: Parametri aggiuntivi per l'invio di circuiti quantistici, default: {"if_print_qcloud_log":False,"chip_id":"WK_C180","is_amend":True,"is_mapping":True,"is_optimization":True,"compile_level":3,"default_task_group_size":200,"test_qcloud_fake":False,"":"server_ip_address"}, quando test_qcloud_fake è impostato su True, simulazione locale CPUQVM.
    :param query_kwargs: Parametri aggiuntivi per interrogare i risultati quantistici. Default: {"timeout":1,"total_timeout":60, "print_query_info":True,"sub_circuits_split_size":1}.

    Esempio::

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
===============
.. py:function:: pyvqnet.qnn.pq3.quantumlayer.grad(quantum_prog_func, input_params, *args)

    La funzione grad fornisce un'interfaccia per calcolare il gradiente dei parametri del circuito quantistico con parametri progettato dall'utente.
    Gli utenti possono utilizzare pyQPanda3 per progettare la funzione di esecuzione del circuito ``quantum_prog_func`` come mostrato sotto, e passarla come parametro alla funzione grad.
    Il secondo parametro della funzione grad sono le coordinate del gradiente del parametro della porta logica quantistica che si desidera calcolare.
    La forma del valore restituito è [num di parametri, num di output].

    :param quantum_prog_func: funzione di esecuzione del circuito quantistico progettata da pyQPanda3.
    :param input_params: parametri di cui calcolare il gradiente.
    :param \*args: altri parametri di input per la funzione quantum_prog_func.
    :return:
        Gradiente dei parametri


    Esempi::

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
==============

QLinear implementa un algoritmo di connessione completa quantistico. Prima i dati vengono codificati in uno stato quantistico, poi vengono eseguite l'evoluzione e la misurazione attraverso circuiti quantistici per ottenere il risultato finale di connessione completa.

.. image:: ./images/qlinear_cir.png

.. py:class:: pyvqnet.qnn.qlinear.QLinear(input_channels,output_channels,machine: str = "CPU")

    Modulo completamente connesso quantistico. L'input del modulo completamente connesso ha forma (canali di input, canali di output). Si noti che questo layer non accetta parametri quantistici variazionali.

    :param input_channels: ``int`` - Numero di canali di input.
    :param output_channels: ``int`` - Numero di canali di output.
    :param machine: ``str`` - La macchina virtuale da utilizzare, per default viene utilizzata la simulazione CPU.
    :return: Layer completamente connesso quantistico.

    Esempio::

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



Qconv
==========================

    Qconv è un'interfaccia per l'algoritmo di convoluzione quantistica.
    L'operazione di convoluzione quantistica utilizza circuiti quantistici per eseguire operazioni di convoluzione su dati classici. Non necessita di calcolare moltiplicazioni e addizioni. Necessita solo di codificare i dati in stati quantistici, quindi eseguire operazioni di evoluzione e misurazione attraverso circuiti quantistici per ottenere i risultati finali di convoluzione.
    Applica lo stesso numero di bit quantistici in base al numero di dati di input nell'intervallo del kernel di convoluzione, quindi costruisce circuiti quantistici per il calcolo.

    .. image:: ./images/qcnn.png

    Il circuito quantistico viene codificato inserendo prima porte :math:`RY` e :math:`RZ` su ogni qubit, quindi utilizzando :math:`Z` e :math:`U3` su due qubit qualsiasi per entangle e scambiare informazioni. Di seguito è riportato un esempio con 4 qubit

    .. image:: ./images/qcnn_cir.png

.. py:class:: pyvqnet.qnn.qcnn.qconv.QConv(input_channels,output_channels,quantum_number,stride=(1, 1),padding=(0, 0),kernel_initializer=normal,machine:str = "CPU",dtype=None, name ="")

    Modulo di convoluzione quantistica. Sostituisce il kernel Conv2D con un circuito quantistico. L'input del modulo di convoluzione ha forma (dimensione batch, canali di input, altezza, larghezza) `Samuel et al. (2020) <https://arxiv.org/abs/2012.12177>`_ .

        :param input_channels: ``int`` - Numero di canali di input.
        :param output_channels: ``int`` - Numero di canali di output.
        :param quantum_number: ``int`` - La dimensione di un singolo kernel.
        :param stride: ``tuple`` - Il passo, default (1,1).
        :param padding: ``tuple`` - Padding, default (0,0).
        :param kernel_initializer: ``callable`` - Default distribuzione normale.
        :param machine: ``str`` - La macchina virtuale da utilizzare, default simulazione CPU.
        :param dtype: Il tipo di dato del parametro, default: None, usa il tipo di dato predefinito: kfloat32, che rappresenta numeri a virgola mobile a 32 bit.
        :param name: Il nome di questo modulo, default "".

        :return: Layer di convoluzione quantistica.


        Esempio::

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

Porte logiche quantistiche
************************************

Il modo di elaborare i bit quantistici è la porta logica quantistica. Utilizzando le porte logiche quantistiche, si evolvono consapevolmente gli stati quantistici. La porta logica quantistica è la base degli algoritmi quantistici.

Porta logica quantistica di base
================================

In questa sezione, utilizziamo le varie porte logiche di pyqpanda3 sviluppate da Origin Quantum per costruire circuiti quantistici e eseguire simulazioni quantistiche.
Le porte logiche attualmente supportate da pyQPanda3 possono essere consultate nella definizione delle porte logiche quantistiche di pyQPanda3.
Inoltre, VQNet incapsula anche alcune combinazioni di porte logiche quantistiche comunemente utilizzate nell'apprendimento automatico quantistico:


BasicEmbeddingCircuit
============================

.. py:function:: pyvqnet.qnn.pq3.template.BasicEmbeddingCircuit(input_feat,qlist)
    
    Codifica n caratteristiche binarie nello stato fondamentale di n qubit.

    Ad esempio, per ``features=([0, 1, 1])``, lo stato fondamentale è :math:`|011 \rangle` in un sistema quantistico.

    :param input_feat: Input binario di dimensione ``(n)``.
    :param qlist: qubit per costruire il circuito template.
    :return: circuito quantistico.


    Esempio::

            from pyvqnet.qnn.pq3.template import BasicEmbeddingCircuit
            import pyqpanda3.core as pq
            from pyvqnet import tensor
            input_feat = tensor.QTensor([1,1,0])
            
            qlist = range(3)
            circuit = BasicEmbeddingCircuit(input_feat,qlist)
            print(circuit)


AngleEmbeddingCircuit
============================

.. py:function:: pyvqnet.qnn.pq3.template.AngleEmbeddingCircuit(input_feat,qubits,rotation:str='X')

    Codifica :math:`N` caratteristiche nell'angolo di rotazione di :math:`n` qubit, dove :math:`N \leq n`.

    La rotazione può essere scelta come: 'X', 'Y', 'Z', come definito dal parametro ``rotation``:

    * ``rotation='X'`` Utilizza la caratteristica come angolo della rotazione RX.

    * ``rotation='Y'`` Utilizza la caratteristica come angolo della rotazione RY.

    * ``rotation='Z'`` Utilizza la caratteristica come angolo della rotazione RZ.

    La lunghezza di ``features`` deve essere minore o uguale al numero di qubit. Se la lunghezza di ``features`` è inferiore al numero di qubit, il circuito non applica le rimanenti porte di rotazione.

    :param input_feat: array numpy che rappresenta i parametri.
    :param qubits: indici dei qubit.
    :param rotation: quale rotazione utilizzare, default "X".
    :return: circuito quantistico.

    Esempio::

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
============================

.. py:function:: pyvqnet.qnn.pq3.template.IQPEmbeddingCircuits(input_feat,qubits,rep:int = 1)
    
    Codifica :math:`n` caratteristiche in :math:`n` qubit utilizzando porte diagonali di un circuito IQP.

    La codifica è proposta da `Havlicek et al. (2018) <https://arxiv.org/pdf/1804.11326.pdf>`_.

    Il circuito IQP di base può essere ripetuto specificando ``n_repeats``.

    :param input_feat: array numpy che rappresenta i parametri.
    :param qubits: lista di indici dei qubit.
    :param rep: Ripete il blocco del circuito quantistico, il numero predefinito di volte è 1.
    :return: circuito quantistico.

    Esempio::

        import numpy as np
        from pyvqnet.qnn.pq3.template import IQPEmbeddingCircuits
        input_feat = np.arange(1,100)
        qlist = range(3)
        circuit = IQPEmbeddingCircuits(input_feat,qlist,rep = 3)
        print(circuit)


RotCircuit
============================

.. py:function:: pyvqnet.qnn.pq3.template.RotCircuit(para,qubits)

    Rotazione arbitraria di un singolo qubit. Il numero di qlist deve essere 1 e il numero di parametri deve essere 3.

    .. math::

        R(\phi,\theta,\omega) = RZ(\omega)RY(\theta)RZ(\phi)= \begin{bmatrix}
        e^{-i(\phi+\omega)/2}\cos(\theta/2) & -e^{i(\phi-\omega)/2}\sin(\theta/2) \\
        e^{-i(\phi-\omega)/2}\sin(\theta/2) & e^{i(\phi+\omega)/2}\cos(\theta/2)
        \end{bmatrix}.

    :param para: array numpy che rappresenta i parametri :math:`[\phi, \theta, \omega]`.
    :param qubits: indice del qubit, vengono accettati solo qubit singoli.
    :return: circuito quantistico.

    Esempio::

        from pyvqnet.qnn.pq3.template import RotCircuit
        import pyqpanda3.core as pq
        from pyvqnet import tensor

        m_qlist = 1

        param =tensor.QTensor([3,4,5])
        c = RotCircuit(param,m_qlist)
        print(c)

CRotCircuit
============================

.. py:function:: pyvqnet.qnn.pq3.template.CRotCircuit(para,control_qubits,rot_qubits)

    Operatore Rot controllato.

    .. math:: 
        
        CR(\phi, \theta, \omega) = \begin{bmatrix}
        1 & 0 & 0 & 0 \\
        0 & 1 & 0 & 0\\
        0 & 0 & e^{-i(\phi+\omega)/2}\cos(\theta/2) & -e^{i(\phi-\omega)/2}\sin(\theta/2)\\
        0 & 0 & e^{-i(\phi-\omega)/2}\sin(\theta/2) & e^{i(\phi+\omega)/2}\cos(\theta/2)
        \end{bmatrix}.

    :param para: Un array numpy che rappresenta i parametri :math:`[\phi, \theta, \omega]`.
    :param control_qubits: indice del qubit di controllo, il numero di qubit deve essere 1.
    :param rot_qubits: indice del qubit di rotazione, il numero di qubit deve essere 1.
    :return: circuito quantistico.

    Esempio::

        from pyvqnet.qnn.pq3.template import CRotCircuit
        import pyqpanda3.core as pq
        import numpy as np
        m_qlist = range(1)
        control_qlist = [1]
        param = np.array([3,4,5])
        cir = CRotCircuit(param,control_qlist,m_qlist)

        print(cir)


CSWAPcircuit
============================

.. py:function:: pyvqnet.qnn.pq3.template.CSWAPcircuit(qubits)

    Circuito SWAP controllato.

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
        
        Il primo qubit fornito corrisponde al **qubit di controllo**.

    :param qubits: lista di indici dei qubit. Il primo qubit è il qubit di controllo. La lunghezza di qlist deve essere 3.
    :return: Il circuito quantistico.

    Esempio::

        from pyvqnet.qnn.pq3 import CSWAPcircuit
        import pyqpanda3.core as pq
        m_machine = pq.CPUQVM()

        m_qlist = range(3)

        c =CSWAPcircuit([m_qlist[1],m_qlist[2],m_qlist[0]])
        print(c)


Controlled_Hadamard
=======================

.. py:function:: pyvqnet.qnn.pq3.template.Controlled_Hadamard(qubits)
    
    Porta logica Hadamard controllata

    .. math:: 
        CH = \begin{bmatrix}
        1 & 0 & 0 & 0 \\
        0 & 1 & 0 & 0 \\
        0 & 0 & \frac{1}{\sqrt{2}} & \frac{1}{\sqrt{2}} \\
        0 & 0 & \frac{1}{\sqrt{2}} & -\frac{1}{\sqrt{2}}
        \end{bmatrix}.

    :param qubits: indice del qubit.

    Esempi::

        import pyqpanda3.core as pq

        machine = pq.CPUQVM()
        
        qubits =range(2)
        from pyvqnet.qnn.pq3 import Controlled_Hadamard

        cir = Controlled_Hadamard(qubits)
        print(cir)

CCZ
==============

.. py:function:: pyvqnet.qnn.pq3.template.CCZ(qubits)

    Porta logica Z controllata-controllata.

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

    :param qubits: indice del qubit.

    :return:
        pyQPanda3 QCircuit

    Esempio::

        import pyqpanda3.core as pq

        machine = pq.CPUQVM()

        qubits = range(3)

        from pyvqnet.qnn.pq3 import CCZ

        cir = CCZ(qubits)


FermionicSingleExcitation
===========================

.. py:function:: pyvqnet.qnn.pq3.template.FermionicSingleExcitation(weight, wires, qubits)

    Operatore di eccitazione singola del coupled cluster per l'esponenziazione di prodotti tensoriali di matrici di Pauli. La forma matriciale è data da:

    .. math::

        \hat{U}_{pr}(\theta) = \mathrm{exp} \{ \theta_{pr} (\hat{c}_p^\dagger \hat{c}_r
        -\mathrm{H.c.}) \},

    :param weight: parametro variabile sul qubit p.
    :param wires: rappresenta un sottoinsieme di indici di qubit nell'intervallo [r, p]. La lunghezza minima deve essere 2. Il primo valore dell'indice è interpretato come r e l'ultimo come p.
        Gli indici intermedi sono soggetti a porte CNOT per calcolare la parità del set di qubit.
    :param qubits: indici dei qubit.

    :return:
        pyQPanda3 QCircuit

    Esempi::

        from pyvqnet.qnn.pq3 import FermionicSingleExcitation, expval

        weight=0.5
        import pyqpanda3.core as pq
        machine = pq.CPUQVM()

        qlists = range(3)

        cir = FermionicSingleExcitation(weight, [1, 0, 2], qlists)


FermionicDoubleExcitation
============================

.. py:function:: pyvqnet.qnn.pq3.template.FermionicDoubleExcitation(weight, wires1, wires2, qubits)

    Operatore di doppia eccitazione del coupled cluster per il prodotto tensoriale di matrici di Pauli, la forma matriciale è data da:

    .. math::

        \hat{U}_{pqrs}(\theta) = \mathrm{exp} \{ \theta (\hat{c}_p^\dagger \hat{c}_q^\dagger
        \hat{c}_r \hat{c}_s - \mathrm{H.c.}) \},

    dove :math:`\hat{c}` e :math:`\hat{c}^\dagger` sono gli operatori fermionici di annichilazione e
    creazione, e gli indici :math:`r, s` e :math:`p, q` si riferiscono rispettivamente agli orbitali molecolari
    occupati e vuoti. Utilizzando la `trasformazione di Jordan-Wigner
    <https://arxiv.org/abs/1208.5986>`_, l'operatore fermionico definito sopra può essere scritto
    in termini di matrici di Pauli (vedere
    `arXiv:1805.04340 <https://arxiv.org/abs/1805.04340>`_ per maggiori dettagli)

    .. math::

        \hat{U}_{pqrs}(\theta) = \mathrm{exp} \Big\{
        \frac{i\theta}{8} \bigotimes_{b=s+1}^{r-1} \hat{Z}_b \bigotimes_{a=q+1}^{p-1}
        \hat{Z}_a (\hat{X}_s \hat{X}_r \hat{Y}_q \hat{X}_p +
        \hat{Y}_s \hat{X}_r \hat{Y}_q \hat{Y}_p +\\ \hat{X}_s \hat{Y}_r \hat{Y}_q \hat{Y}_p +
        \hat{X}_s \hat{X}_r \hat{X}_q \hat{Y}_p - \mathrm{H.c.} ) \Big\}

    :param weight: parametro variabile
    :param wires1: rappresenta il sottoinsieme di qubit nell'intervallo [s, r] occupato dalla lista di indici dei qubit. Il primo indice è interpretato come s e l'ultimo come r. La porta CNOT opera sugli indici intermedi per calcolare la parità di un set di qubit.
    :param wires2: rappresenta il sottoinsieme di qubit nell'intervallo [q, p] occupato dalla lista di indici dei qubit. Il primo indice è interpretato come q e l'ultimo come p. La porta CNOT opera sull'indice intermedio per calcolare la parità di un set di qubit.
    :param qubits: indici dei qubit.

    :return:
        pyQPanda3 QCircuit

    Esempi::

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
===================

.. py:function:: pyvqnet.qnn.pq3.template.UCCSD(weights, wires, s_wires, d_wires, init_state, qubits)

    Implementa la simulazione UCCSD (Unitary Coupled Cluster Single and Double Excitation). UCCSD è una simulazione VQE, comunemente utilizzata per eseguire simulazioni di chimica quantistica.

    Nell'approssimazione di primo ordine di Trotter, la funzione unitaria UCCSD è data da:

    .. math::

        \hat{U}(\vec{\theta}) =
        \prod_{p > r} \mathrm{exp} \Big\{\theta_{pr}
        (\hat{c}_p^\dagger \hat{c}_r-\mathrm{H.c.}) \Big\}
        \prod_{p > q > r > s} \mathrm{exp} \Big\{\theta_{pqrs}
        (\hat{c}_p^\dagger \hat{c}_q^\dagger \hat{c}_r \hat{c}_s-\mathrm{H.c.}) \Big\}
    
    dove :math:`\hat{c}` e :math:`\hat{c}^\dagger` sono gli operatori fermionici di annichilazione e

    creazione, e gli indici :math:`r, s` e :math:`p, q` sono rispettivamente gli orbitali molecolari
    occupati e vuoti. (Per maggiori dettagli vedere
    `arXiv:1805.04340 <https://arxiv.org/abs/1805.04340>`_):

    :param weights: tensore di dimensione ``(len(s_wires)+ len(d_wires))`` contenente i parametri
     :math:`\theta_{pr}` e :math:`\theta_{pqrs}` in input alle rotazioni Z ``FermionicSingleExcitation`` e ``FermionicDoubleExcitation``.
    :param wires: indici dei qubit da templatizzare
    :param s_wires: sequenza di liste contenente gli indici dei qubit ``[r,...,p]`` generati da una singola eccitazione
     :math:`\vert r, p \rangle = \hat{c}_p^\dagger \hat{c}_r \vert \mathrm{HF} \rangle`, dove :math:`\vert \mathrm{HF} \rangle` denota lo stato di riferimento di Hartree-Fock.
    :param d_wires: sequenza di liste, ciascuna contenente due liste che specificano gli indici ``[s, ...,r]`` e ``[q,..., p]``. Definisce la doppia eccitazione :math:`\vert s, r, q, p \rangle = \hat{c}_p^\dagger \hat{c}_q^\dagger \hat{c}_r\hat{c}_s \vert \mathrm{HF} \rangle`.
    :param init_state: vettore di occupazione di lunghezza ``len(wires)`` che rappresenta lo stato ad alta frequenza. ``init_state`` Inizializza lo stato del qubit.
    :param qubits: Indice del qubit.

    Esempi::
        
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
============================

.. py:function:: pyvqnet.qnn.pq3.template.QuantumPoolingCircuit(sources_wires, sinks_wires, params,qubits)

    Circuito quantistico che esegue il downsampling dei dati.

    Per ridurre il numero di qubit nel circuito, creare prima coppie di qubit nel sistema. Dopo aver inizialmente accoppiato tutti i qubit, applicare l'unitario generalizzato a 2 qubit a ogni coppia di qubit. Dopo aver applicato questi unitari a due qubit, ignorare un qubit in ogni coppia per il resto della rete neurale.

    :param sources_wires: Indici dei qubit sorgente da ignorare.
    :param sinks_wires: Indici dei qubit di destinazione da mantenere.
    :param params: Parametri di input.
    :param qubits: Indici dei qubit.

    :return:
        pyQPanda3 QCircuit

    Esempi::

        from pyvqnet.qnn.pq3.template import QuantumPoolingCircuit
        import pyqpanda3.core as pq
        from pyvqnet import tensor

        qlists = range(4)
        p = tensor.full([6], 0.35)
        cir = QuantumPoolingCircuit([0, 1], [2, 3], p, qlists)
        print(cir)

Combinazioni di circuiti quantistici comunemente utilizzate
***********************************************************
VQNet fornisce alcuni circuiti quantistici comunemente utilizzati nella ricerca sull'apprendimento automatico quantistico

HardwareEfficientAnsatz
=============================

.. py:class:: pyvqnet.qnn.pq3.ansatz.HardwareEfficientAnsatz(qubits,single_rot_gate_list,entangle_gate="CNOT",entangle_rules='linear',depth=1)

    Implementazione dell'Hardware Efficient Ansatz introdotto nell'articolo: `Hardware-efficient Variational Quantum Eigensolver for Small Molecules <https://arxiv.org/pdf/1704.05018.pdf>`__ .

    :param qubits: indice del qubit.
    :param single_rot_gate_list: Lista di porte di rotazione a singolo qubit composta da una o più porte di rotazione che agiscono su ogni qubit. Attualmente supporta Rx, Ry, Rz.
    :param entangle_gate: Porta di entanglement non parametrica. Supporta CNOT, CZ. Default: CNOT.
    :param entangle_rules: Modalità di utilizzo della porta di entanglement nel circuito. ``linear`` significa che la porta di entanglement agirà su ogni qubit adiacente. ``all`` significa che la porta di entanglement agirà su qualsiasi coppia di qubit. Default: ``linear``.
    :param depth: Profondità dell'ansatz, default: 1.

    :return:
        Un'istanza di HardwareEfficientAnsatz

    Esempio::

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
============================

.. py:class:: pyvqnet.qnn.pq3.template.BasicEntanglerTemplate(weights=None, num_qubits=1, rotation=pyqpanda3.RX)
    
    Un layer composto da rotazioni a singolo qubit con un singolo parametro su ogni qubit, seguito da porte CNOT multiple combinate in una catena chiusa o anello.
    L'anello di porte CNOT collega ogni qubit ai suoi vicini, con l'ultimo qubit considerato vicino del primo.
    Il numero di layer :math:`L` è determinato dalla prima dimensione del parametro ``weights``.

    :param weights: Un tensore di pesi di forma `(L, len(qubits))`. Ogni peso è usato come parametro in una porta parametrica quantistica. Il valore predefinito è: ``None``, in tal caso vengono utilizzati numeri casuali distribuiti normalmente `(1,1)` come pesi.
    :param num_qubits: Il numero di qubit, default è 1.
    :param rotation: Utilizza una porta a singolo qubit con un singolo parametro, ``pyqpanda3.RX`` è usato come valore predefinito.
    :return:
        Un'istanza di BasicEntanglerTemplate

    Esempio::

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
============================

.. py:class:: pyvqnet.qnn.pq3.template.StronglyEntanglingTemplate(weights=None, num_qubits=1, ranges=None)

    Layer composti da rotazioni a singolo qubit e entangler, come in `circuit-centric classifier design <https://arxiv.org/abs/1804.00633>`__ .
    Il parametro ``weights`` contiene i pesi di ogni layer. Quindi il numero di layer :math:`L` è uguale alla prima dimensione di ``weights``.
    Contiene porte CNOT a 2 qubit che agiscono su :math:`M` qubit, :math:`i = 1,...,M`. Il secondo qubit di ogni porta è dato dalla formula :math:`(i+r)\mod M`, dove :math:`r` è un iperparametro chiamato ``range``, e :math:`0 < r < M`.

    :param weights: Tensore di pesi di forma ``(L, M, 3)``, valore predefinito: None, usa un tensore casuale di forma ``(1,1,3)``.
    :param num_qubits: Numero di qubit, valore predefinito: 1.
    :param ranges: Sequenza che determina gli iperparametri di range per ogni layer successivo; valore predefinito: None, usa :math:`r=l \mod M` come valore di ranges.
    :return:
        Un'istanza di StronglyEntanglingTemplate

    Esempio::

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
============================

.. py:class:: pyvqnet.qnn.pq3.ComplexEntanglingTemplate(weights,num_qubits,depth)

    Layer fortemente entangled composto da porte U3 e porte CNOT.
    Questo template di circuito proviene dal seguente articolo: https://arxiv.org/abs/1804.00633.

    :param weights: parametri, forma [depth,num_qubits,3]
    :param num_qubits: numero di qubit.
    :param depth: profondità del sottocircuito.
    :return:
        Un'istanza di ComplexEntanglingTemplate

    Esempio::

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
============================

.. py:class:: pyvqnet.qnn.pq3.Quantum_Embedding(qubits, machine, num_repetitions_input, depth_input, num_unitary_layers, num_repetitions)

    Utilizza RZ, RY, RZ per creare un circuito quantistico variazionale per codificare dati classici in stati quantistici.
    Riferimento `Quantum embeddings for machine learning <https://arxiv.org/abs/2001.03622>`_.

    Dopo aver inizializzato la classe, la sua funzione membro ``compute_circuit`` è la funzione di esecuzione, che può essere utilizzata come parametro per l'input della classe ``QuantumLayerV2`` per formare un layer del modello di apprendimento automatico quantistico.

    :param qubits: I bit quantistici richiesti da pyQPanda3.
    :param machine: Macchina virtuale quantistica richiesta da pyQPanda3.
    :param num_repetitions_input: Il numero di ripetizioni della codifica dell'input nel sottomodulo.
    :param depth_input: La dimensione della caratteristica dei dati di input.
    :param num_unitary_layers: Il numero di ripetizioni della porta quantistica variazionale in ogni sottomodulo.
    :param num_repetitions: Il numero di ripetizioni del sottomodulo.
    :return:
        Un'istanza di Quantum_Embedding

    Esempio::

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


Misurare circuiti quantistici
************************************

expval
============================

.. py:function:: pyvqnet.qnn.pq3.measure.expval(machine,prog,pauli_str_dict,shots=1000,noise_model=None)

    Il valore atteso dell'osservazione hamiltoniana fornita.

    Se l'osservazione è :math:`0.7Z\otimes X\otimes I+0.2I\otimes Z\otimes I`,
    allora il dizionario hamiltoniano sarà ``{{'Z0, X1':0.7} ,{'Z1':0.2}}``.

    L'API expval supporta ora il simulatore pyQPanda3.

    :param machine: La macchina quantistica creata da pyQPanda3.
    :param prog: Il programma quantistico creato da pyQPanda3.
    :param pauli_str_dict: Valore osservato hamiltoniano.
    :param shots: Numero di misurazioni, default è 1000.
    :param noise_model: Modello di rumore da applicare, default è None (simulazione ideale).

    :return: valore atteso.

    Esempio::

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
=============================

.. py:function:: pyvqnet.qnn.pq3.measure.QuantumMeasure(machine,prog,measure_qubits:list,shots:int = 1000, qcloud_option="",noise_model=None)

    Calcola le misurazioni del circuito quantistico. Restituisce le misurazioni ottenute con il metodo Monte Carlo.

    Per maggiori dettagli sulla misurazione, si prega di consultare la documentazione di pyQPanda3.

    L'API QuantumMeasure attualmente supporta solo ``CPUQVM`` o ``QCloud`` di pyQPanda3.

    :param machine: La macchina virtuale quantistica allocata da pyQPanda3.
    :param prog: Il programma quantistico creato da pyQPanda3.
    :param measure_qubits: Lista contenente gli indici dei bit di misurazione.
    :param shots: Il numero di misurazioni, il valore predefinito è 1000.
    :param qcloud_option: Imposta la configurazione qcloud, il valore predefinito è "", è possibile passare una classe QCloudOptions, utile solo quando si utilizza qcloud.
    :param noise_model: Modello di rumore da applicare, default è None (simulazione ideale).
    :return: Restituisce i risultati di misurazione ottenuti con il metodo Monte Carlo.

    Esempio::

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
============================

.. py:function:: pyvqnet.qnn.pq3.measure.ProbsMeasure(machine,prog,measure_qubits:list,shots=1,noise_model=None)

    Calcola le misurazioni di probabilità del circuito.

    Per maggiori dettagli, si prega di consultare la documentazione di pyQPanda3 sulla misurazione di probabilità.

    L'API ProbsMeasure attualmente supporta solo ``CPUQVM`` o ``QCloud`` di pyQPanda.

    :param measure_qubits: Lista contenente gli indici dei qubit di misurazione.
    :param prog: Il programma quantistico creato da qpanda.
    :param machine: La macchina virtuale quantistica allocata da pyQPanda.
    :param shots: Numero di misurazioni, default è 1, che calcola il valore teorico.
    :param noise_model: Modello di rumore da applicare, default è None (simulazione ideale).
    :return: Misura i qubit in ordine lessicografico.


    Esempio::

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
===========================
.. py:function:: pyvqnet.qnn.pq3.measure.DensityMatrixFromQstate(state, indices)
    
    Calcola la matrice di densità di uno stato quantistico su un insieme specifico di qubit.

    :param state: Lista 1D di vettori di stato. La dimensione di questa lista dovrebbe essere ``(2**N,)`` Per un numero di qubit ``N``, qstate dovrebbe iniziare da 000 -> 111.
    :param indices: Lista di indici dei qubit nel sottosistema considerato.
    :return: 
        Matrice di densità di dimensione ``(2**len(indices), 2**len(indices))``.

    Esempio::

        from pyvqnet.qnn.pq3.measure import DensityMatrixFromQstate
        qstate = [(0.9306699299765968+0j), (0.18865613455240968+0j), (0.1886561345524097+0j), (0.03824249173404786+0j), -0.048171819846746615j, -0.00976491131165138j, -0.23763904794287155j, -0.048171819846746615j]
        print(DensityMatrixFromQstate(qstate,[0,1]))
        # [[0.86846704+0.j 0.1870241 +0.j 0.17604699+0.j 0.03791166+0.j]
        # [0.1870241 +0.j 0.09206345+0.j 0.03791166+0.j 0.01866219+0.j]
        # [0.17604699+0.j 0.03791166+0.j 0.03568649+0.j 0.00768507+0.j]
        # [0.03791166+0.j 0.01866219+0.j 0.00768507+0.j 0.00378301+0.j]]


VN_Entropy
==============
.. py:function:: pyvqnet.qnn.pq3.measure.VN_Entropy(state, indices, base=None)
    
    Calcola l'entropia di Von Neumann dato un vettore di stato su una data lista di qubit.

    .. math::

        S( \rho ) = -\text{Tr}( \rho \log ( \rho ))

    :param state: Lista 1D di vettori di stato. La dimensione di questa lista dovrebbe essere ``(2**N,)`` Per un numero di qubit ``N``, qstate dovrebbe iniziare da 000 -> 111.
    :param indices: Lista di indici dei qubit nel sottosistema in esame.
    :param base: Base del logaritmo. Se None, viene utilizzato il logaritmo naturale.

    :return: valore a virgola mobile dell'entropia di von Neumann.

    Esempio::

        from pyvqnet.qnn.pq3.measure import VN_Entropy
        qstate = [(0.9022961387408862 + 0j), -0.06676534788028633j,
        (0.18290448232350312 + 0j), -0.3293638014158896j,
        (0.03707657410649268 + 0j), -0.06676534788028635j,
        (0.18290448232350312 + 0j), -0.013534006039561714j] 
        print(VN_Entropy(qstate, [0, 1]))
        #0.14592917648464448

Mutal_Info
==============
.. py:function:: pyvqnet.qnn.pq3.measure.Mutal_Info(state, indices0, indices1, base=None)

    Calcola l'informazione mutua dato il vettore di stato su due liste di sotto-qubit.

    .. math::

        I(A, B) = S(\rho^A) + S(\rho^B) - S(\rho^{AB})
        dove :math:`S` è l'entropia di von Neumann.

    L'informazione mutua è una misura della correlazione tra due sottosistemi. Più specificamente, quantifica la quantità di informazioni che un sistema può ottenere misurando l'altro.

    Ogni stato può essere dato come un vettore di stato nella base di calcolo.

    :param state: Lista 1D di vettori di stato. La dimensione di questa lista dovrebbe essere ``(2**N,)`` Per il numero di qubit ``N``, qstate dovrebbe iniziare da 000 -> 111.
    :param indices0: Lista di indici dei qubit nel primo sottosistema.
    :param indices1: Lista di indici dei qubit nel secondo sottosistema.
    :param base: Base dei logaritmi. Se None, vengono utilizzati i logaritmi naturali. Default è None.

    :return: Informazione mutua tra i sottosistemi

    Esempio::

        from pyvqnet.qnn.pq3.measure import Mutal_Info
        qstate = [(0.9022961387408862 + 0j), -0.06676534788028633j,
        (0.18290448232350312 + 0j), -0.3293638014158896j,
        (0.03707657410649268 + 0j), -0.06676534788028635j,
        (0.18290448232350312 + 0j), -0.013534006039561714j]
        print(Mutal_Info(qstate, [0], [2], 2))
        #0.13763425302805887

Purity
=========================

.. py:function:: pyvqnet.qnn.pq3.measure.Purity(state, qubits_idx)

    Calcola la purezza di un particolare qubit dal vettore di stato.

    .. math::

        \gamma = \text{Tr}(\rho^2)
        
    dove :math:`\rho` è la matrice di densità. La purezza di uno stato quantistico normalizzato soddisfa :math:`\frac{1}{d} \leq \gamma \leq 1`,
    dove :math:`d` è la dimensione dello spazio di Hilbert.
    La purezza di uno stato puro è 1.

    :param state: stato quantistico ottenuto da pyqpanda3
    :param qubits_idx: indice del qubit per cui calcolare la purezza

    :return:
        purezza

    Esempi::

        from pyvqnet.qnn.pq3.measure import Purity
        qstate = [(0.9306699299765968 + 0j), (0.18865613455240968 + 0j),
        (0.1886561345524097 + 0j), (0.03824249173404786 + 0j),
        -0.048171819846746615j, -0.00976491131165138j, -0.23763904794287155j,
        -0.048171819846746615j]
        pp = Purity(qstate, [1])
        print(pp)
        #0.902503479761881
