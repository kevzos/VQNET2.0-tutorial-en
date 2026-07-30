Verwendung des pyQPanda3 Quantum Machine Learning Moduls
#########################################################

.. warning::

    Der Quantencomputing-Teil der folgenden Schnittstelle verwendet pyqpanda3.

    Wenn Sie die QCloud-Funktion unter diesem Modul verwenden, treten Fehler beim Import von pyqpanda2 im Code oder bei der Verwendung der pyvqnet-Schnittstellenpakete für pyqpanda2 auf.

Quantencomputing-Schicht
***********************************

.. _QuantumLayer_pq3:

QuantumLayer
============================

Wenn Sie mit der pyQPanda3-Syntax vertraut sind, können Sie die Schnittstelle QuantumLayer verwenden, um den pyqpanda3-Simulator für Berechnungen anzupassen.

.. py:class:: pyvqnet.qnn.pq3.quantumlayer.QuantumLayer(qprog_with_measure,para_num,diff_method:str = "parameter_shift",delta:float = 0.01,dtype=None,name="")

    Abstraktes Berechnungsmodul der variationellen Quantenschicht. Verwendet pyQPanda3, um einen parametrisierten Quantenschaltkreis zu simulieren und die Messergebnisse zu erhalten. Diese variationelle Quantenschicht erbt das Gradientenberechnungsmodul des VQNet-Frameworks. Sie kann die Parameterdrift-Methode zur Berechnung des Gradienten der SchaltkreispParameter verwenden, variationelle Quantenschaltkreismodelle trainieren oder variationelle Quantenschaltkreise in hybride Quanten-Klassik-Modelle einbetten.

    :param qprog_with_measure: Quantenschaltkreisoperationen und Messfunktionen, erstellt mit pyQPanda.
    :param para_num: ``int`` - Anzahl der Parameter.
    :param diff_method: Methode zur Lösung der Gradienten der Quantenschaltkreisparameter, „parameter shift“ oder „finite difference“, Standard: Parameter-Offset.
    :param delta: :math:`\delta` bei der Berechnung von Gradienten durch finite Differenzen.
    :param dtype: Datentyp des Parameters, Standard: None, verwendet den Standard-Datentyp: kfloat32, 32-Bit-Gleitkommazahlen.
    :param name: Der Name dieses Moduls, Standard: „“.

    :return: Ein Modul, das Quantenschaltkreise berechnen kann.

    .. note::

        qprog_with_measure ist eine in pyQPanda3 definierte Quantenschaltkreisfunktion.

        Diese Funktion muss zwei Parameter enthalten, input und parameter, als Funktionseingabe (auch wenn ein Parameter nicht tatsächlich verwendet wird), und die Ausgabe ist das Messergebnis oder der Erwartungswert des Schaltkreises (muss ein np.ndarray oder eine Liste mit Werten sein), sonst funktioniert es nicht richtig in QpandaQCircuitVQCLayerLite.

        Die Verwendung der Quantenschaltkreisfunktion qprog_with_measure (input, param) kann dem folgenden Beispiel entnommen werden.

        ``input``: Eingabe eindimensionaler klassischer Daten. Wenn nicht vorhanden, geben Sie None ein.

        ``param``: Eingabe der eindimensionalen variationellen Quantenschaltkreisparameter, die trainiert werden sollen.

    .. note::

        Diese Klasse hat die Aliase ``QuantumLayerV2``, ``QpandaQCircuitVQCLayerLite``.

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

        # klassische Daten als Eingabe
        input = QTensor([[1,2,3,4],[4,2,2,3],[3.0,3,2,2]] )

        # Vorwärtsdurchlauf der Schaltkreise
        rlt = pqc(input)

        print(rlt)

        grad = ones(rlt.data.shape)*1000
        # Rückwärtsdurchlauf der Schaltkreise
        rlt.backward(grad)

        print(pqc.m_para.grad)

QpandaQProgVQCLayer
=============================

.. py:class:: pyvqnet.qnn.pq3.quantumlayer.QuantumLayerV3(origin_qprog_func,para_num,qvm_type="cpu", pauli_str_dict=None, shots=1000, initializer=None,dtype=None,name="")

    Es übermittelt den parametrisierten Quantenschaltkreis an den lokalen QPanda3-Vollamplituden-Simulator zur Berechnung und trainiert die Parameter im Schaltkreis.
    Es unterstützt Batch-Daten und verwendet die Parameter-Shift-Regel zur Schätzung des Gradienten der Parameter.
    Für CRX, CRY, CRZ verwendet diese Schicht die Formel aus https://iopscience.iop.org/article/10.1088/1367-2630/ac2cb3, und die restlichen Logikgatter verwenden die Standard-Parameterdrift-Methode zur Berechnung des Gradienten.

    :param origin_qprog_func: Die aufrufbare Quantenschaltkreisfunktion, erstellt mit QPanda.
    :param para_num: ``int`` - Anzahl der Parameter; Parameter sind eindimensional.
    :param qvm_type: ``str`` - Typ des zu verwendenden pyqpanda3-Simulators, ``cpu`` oder ``gpu``, Standard ``cpu``.
    :param pauli_str_dict: ``dict|list`` - Wörterbuch oder Liste von Wörterbüchern, die die Pauli-Operatoren im Quantenschaltkreis darstellen. Standard ist None.
    :param shots: ``int`` - Anzahl der Messdurchläufe. Standard ist 1000.
    :param initializer: Initialisierer für die Parameterwerte. Standard ist None.
    :param dtype: Datentyp des Parameters. Standard ist None, was den Standard-Datentyp bedeutet.
    :param name: Name des Moduls. Standard ist der leere String.

    :return: Gibt eine QpandaQProgVQCLayer-Klasse zurück.

    .. note::

        origin_qprog_func ist eine vom Benutzer mit pyQPanda3 definierte Quantenschaltkreisfunktion.

        Diese Funktion muss zwei Parameter enthalten, input und parameter, als Funktionseingabe (auch wenn ein Parameter nicht tatsächlich verwendet wird), und die Ausgabe muss vom Typ pyqpanda3.core.QProg sein, sonst kann sie nicht richtig in QuantumLayerV3 ausgeführt werden.

        origin_qprog_func (input,param )

        ``input``: benutzerdefinierte Array-Klasse, Eingabe eindimensionaler klassischer Daten

        ``param``: array_artige Eingabe, benutzerdefinierte eindimensionale Quantenschaltkreisparameter

    .. note::

        Diese Klasse hat den Alias ``QuantumLayerV3``.

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
==================================

Wenn Sie die neueste Version von pyqpanda3 installiert haben, können Sie diese Schnittstelle verwenden, um einen variationellen Schaltkreis zu definieren und ihn an den originqc-Realchip zur Ausführung zu übermitteln.

.. py:class:: pyvqnet.qnn.pq3.quantumlayer.QuantumBatchAsyncQcloudLayer(origin_qprog_func, qcloud_token, para_num, pauli_str_dict=None, shots = 1000, initializer=None, dtype=None, name="", diff_method="parameter_shift", submit_kwargs={}, query_kwargs={})
    
    Ein abstraktes Berechnungsmodul für den originqc-Realchip unter Verwendung von pyqpanda3 QCLOUD. Es übermittelt parametrisierte Quantenschaltkreise an den Realchip und erhält Messergebnisse.
    Wenn diff_method == "random_coordinate_descent", wählt die Schicht zufällig einen einzelnen Parameter zur Berechnung des Gradienten aus, während andere Parameter auf Null bleiben. Referenz: https://arxiv.org/abs/2311.00088

    .. note::

        qcloud_token ist der API-Token, den Sie von der Cloud-Plattform beantragt haben.

        origin_qprog_func muss Daten vom Typ pyqpanda3.core.QProg zurückgeben. Wenn pauli_str_dict nicht gesetzt ist, muss sichergestellt werden, dass die Messung in das QProg eingefügt wurde.

        origin_qprog_func muss das folgende Format haben:

        origin_qprog_func(input,param)

            ``input``: Eingabe 1~2D klassischer Daten. Im 2D-Fall ist die erste Dimension die Batch-Größe

            ``param``: Eingabe der zu trainierenden Parameter für den 1D-variationellen Quantenschaltkreis

    .. note::

        In der aktuellen Version beträgt das standardmäßige Gesamt-Timeout für die Übermittlung eines einzelnen Schaltkreises an die QCloud 60 Sekunden. Falls ein Timeout aufgrund einer ausgelasteten QCloud auftritt, können Sie den Wert des Schlüssels ``total_timeout`` in ``query_kwargs`` auf die gewünschte Anzahl von Wartesekunden setzen.

    :param origin_qprog_func: Die mit QPanda erstellte variationelle Quantenschaltkreisfunktion, die ein QProg zurückgeben muss.
    :param qcloud_token: ``str`` - Der Typ der Quantenmaschine oder der Cloud-Token, der zur Ausführung verwendet wird.
    :param para_num: ``int`` - Die Anzahl der Parameter, der Parameter ist ein QTensor der Größe [para_num].
    :param pauli_str_dict: ``dict|list`` - Ein Wörterbuch oder eine Liste von Wörterbüchern, die die Pauli-Operatoren im Quantenschaltkreis darstellen. Der Standardwert ist „None“, was Messoperationen durchführt. Wenn ein Wörterbuch von Pauli-Operatoren eingegeben wird, wird ein einzelner Erwartungswert oder mehrere Erwartungswerte berechnet.
    :param shots: ``int`` - Die Anzahl der Messungen. Der Standardwert ist 1000.
    :param initializer: Initialisierer für die Parameterwerte. Der Standardwert ist „None“, was eine Normalverteilung von 0~2*pi verwendet.
    :param dtype: Der Datentyp des Parameters. Der Standardwert ist None, was den Standard-Datentyp pyvqnet.kfloat32 bedeutet.
    :param name: Der Name des Moduls. Der Standardwert ist ein leerer String.
    :param diff_method: Differenzierungsmethode für die Gradientenberechnung. Der Standardwert ist „parameter_shift“, „random_coordinate_descent“.
    :param submit_kwargs: Zusätzliche Schlüsselwortparameter für die Übermittlung von Quantenschaltkreisen, Standard: {"if_print_qcloud_log":False,"chip_id":"WK_C180","is_amend":True,"is_mapping":True,"is_optimization":True,"compile_level":3,"default_task_group_size":200,"test_qcloud_fake":False,"":"server_ip_address"}, wenn test_qcloud_fake auf True gesetzt ist, lokale CPUQVM-Simulation.
    :param query_kwargs: Zusätzliche Schlüsselwortparameter für die Abfrage von Quantenergebnissen, Standard: {"timeout":1,"total_timeout":60,"print_query_info":True,"sub_circuits_split_size":1}.
    :return: Ein Modul, das Quantenschaltkreise berechnen kann.

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
============================

.. py:class:: pyvqnet.qnn.pq3.quantumlayer.QuantumLayerAdjoint(pq3_vqc_circuit, param_num, pauli_dicts, dtype = None, name="")

    Diese Klasse verwendet die pyqpanda3 VQCircuit-Schnittstelle, um die Gradienten der Parameter in einem Quantenschaltkreis bezüglich des Hamiltonians mit der adjungierten Methode zu berechnen.

    Diese Klasse unterstützt Batch-Eingabe und mehrere Hamiltonian-Ausgaben.

    .. note::

        Bei Verwendung dieser Schnittstelle muss der Schaltkreis mit Logikgattern aus VQCircuit konstruiert werden.

        Derzeit werden nur begrenzte Logikgatter unterstützt; bei nicht unterstützten wird eine Ausnahme ausgelöst.

        Der Eingabeparameter ``pq3_vqc_circuit`` kann nur zwei Parameter enthalten, ``x`` und ``param``, die ein eindimensionales Array oder eine Liste sein müssen.

        In der Funktion ``pq3_vqc_circuit`` müssen Benutzer ``pyqpanda3.vqcircuit.VQCircuit().set_Param`` verwenden, um anzupassen, wie Eingaben und Parameter verarbeitet werden.

        Zusätzlich müssen Benutzer die Anzahl der Parameter in ``param_num`` vorab eingeben. Diese Schnittstelle initialisiert einen Parameter ``m_para`` mit einer Länge von ``param_num``.

        Siehe das folgende Beispiel.

    :param pq3_vqc_circuit: Anpassung des pyqpanda3 VQCircuit-Schaltkreises.
    :param param_num: Anzahl der Parameter.
    :param pauli_dicts: Erwartete Observablen, kann eine Liste sein.
    :param dtype: Parametertyp, kfloat32 oder kfloat64, Standard: None, verwendet kfloat32.
    :param name: Der Name dieser Schnittstelle.
    :return: Gibt eine QuantumLayerAdjoint-Instanz zurück.


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
============================

.. py:class:: pyvqnet.qnn.pq3.vqc_qcloud_layer.VQCQCloudLayer(vqc_module, qcloud_token, pauli_str_dict=None, shots=1000, name="", submit_kwargs={}, query_kwargs={})

    Übermittelt das VQC-Modul an den QCloud-Realchip oder den lokalen pyqpanda3-Simulator zur Ausführung.

    Vorwärtspropagierung: Statt der Ausführung der VQNet-Quantenvariationsschaltkreisberechnung wird der Quanten-Realchip oder der qpanda-lokale Simulator zur Berechnung aufgerufen.

    Rückwärtspropagierung: Verwendet die Parameter-Shift-Regel zur Berechnung von Gradienten. Für jede Eingabedimension und jeden trainierbaren Parameter im VQC werden um +/- pi/2 verschobene Schaltkreise generiert und zur Berechnung übermittelt. Die Ergebnisse werden abgerufen, um die Jacobi-Matrix zu berechnen. Gradienten werden auf dem Eingabe-Tensor und den trainierbaren Parametern des VQC gesetzt.

    .. note::

        Das Standard-Timeout für die Übermittlung eines einzelnen Schaltkreises an die QCloud beträgt 60 Sekunden. Falls ein Timeout aufgrund einer ausgelasteten QCloud auftritt, können Sie den Schlüssel ``total_timeout`` in ``query_kwargs`` setzen, um die Wartesekunden anzugeben.

    .. note::

        Sie können keine Messfunktion (wie ``MeasureAll``) in ``vqc_module`` definieren. Die Messung sollte über den Parameter ``pauli_str_dict`` angegeben werden, um Observablen zu kennzeichnen.
        Zum Beispiel: ``VQCQCloudLayer(vqc_module, token, pauli_str_dict={'Z0': 1, 'Z1': 1})``.

    :param vqc_module: VQNet VQC-Modul, muss eine QMachine mit save_ir=True enthalten.
    :param qcloud_token: QCloud-API-Token. Übergeben Sie einen leeren String, wenn Sie einen lokalen Simulator verwenden.
    :param pauli_str_dict: Pauli-Operator-Wörterbuch für die Erwartungswertberechnung. Standard ist None, was eine Messoperation durchführt.
    :param shots: Anzahl der Messungen. Standard ist 1000.
    :param name: Modulname. Standard ist leerer String.
    :param submit_kwargs: Zusätzliche Schlüsselwortparameter für die Übermittlung von Quantenschaltkreisen, Standard: {"if_print_qcloud_log":False,"chip_id":"WK_C180","is_amend":True,"is_mapping":True,"is_optimization":True,"compile_level":3,"default_task_group_size":200,"test_qcloud_fake":False,"":"server_ip_address"}, wenn test_qcloud_fake auf True gesetzt ist, lokale CPUQVM-Simulation.
    :param query_kwargs: Zusätzliche Schlüsselwortargumente für die Abfrage von Quantenergebnissen. Standard: {"timeout":1,"total_timeout":60, "print_query_info":True,"sub_circuits_split_size":1}.

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
===============
.. py:function:: pyvqnet.qnn.pq3.quantumlayer.grad(quantum_prog_func, input_params, *args)

    Die grad-Funktion bietet eine Schnittstelle zur Berechnung des Gradienten der Parameter eines benutzerdefinierten parametrisierten Quantenschaltkreises.
    Benutzer können mit pyQPanda3 die Schaltkreisausführungsfunktion ``quantum_prog_func`` wie unten gezeigt entwerfen und sie als Parameter an die grad-Funktion übergeben.
    Der zweite Parameter der grad-Funktion sind die Koordinaten des Quantenlogikgatter-Parametergradienten, den Sie berechnen möchten.
    Die Form des Rückgabewerts ist [Anzahl der Parameter, Anzahl der Ausgaben].

    :param quantum_prog_func: Mit pyQPanda3 entworfene Quantenschaltkreis-Ausführungsfunktion.
    :param input_params: Parameter, für die der Gradient berechnet werden soll.
    :param \*args: Andere Parameter, die an die Funktion quantum_prog_func übergeben werden.
    :return:
        Gradient der Parameter


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
==============

QLinear implementiert einen Quanten-Vollverbindungsalgorithmus. Zunächst werden die Daten in einen Quantenzustand kodiert, dann werden durch Quantenschaltkreise die Evolutionsoperation und Messung durchgeführt, um das endgültige Vollverbindungsergebnis zu erhalten.

.. image:: ./images/qlinear_cir.png

.. py:class:: pyvqnet.qnn.qlinear.QLinear(input_channels,output_channels,machine: str = "CPU")

    Quanten-Vollverbindungsmodul. Die Eingabe des Vollverbindungsmoduls hat die Form (Eingabekanäle, Ausgabekanäle). Beachten Sie, dass diese Schicht keine variationellen Quantenparameter verwendet.

    :param input_channels: ``int`` - Anzahl der Eingabekanäle.
    :param output_channels: ``int`` - Anzahl der Ausgabekanäle.
    :param machine: ``str`` - Die zu verwendende virtuelle Maschine, standardmäßig wird CPU-Simulation verwendet.
    :return: Quanten-Vollverbindungsschicht.

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



Qconv
==========================

    Qconv ist eine Quantenfaltungsalgorithmus-Schnittstelle.
    Die Quantenfaltungsoperation verwendet Quantenschaltkreise, um Faltungsoperationen auf klassischen Daten durchzuführen. Sie benötigt keine Multiplikations- und Additionsoperationen, sondern kodiert die Daten lediglich in Quantenzustände und führt dann Evolutionsoperationen und Messungen über Quantenschaltkreise durch, um die endgültigen Faltungsergebnisse zu erhalten.
    Es werden entsprechend der Anzahl der Eingabedaten im Bereich des Faltungskerns die gleiche Anzahl von Quantenbits angefordert und dann Quantenschaltkreise für die Berechnung erstellt.

    .. image:: ./images/qcnn.png

    Der Quantenschaltkreis wird kodiert, indem zunächst :math:`RY`- und :math:`RZ`-Gatter auf jedes Qubit gesetzt werden, und dann :math:`Z` und :math:`U3` auf beliebigen zwei Qubits verwendet werden, um zu verschränken und Informationen auszutauschen. Nachfolgend ein Beispiel mit 4 Qubits:

    .. image:: ./images/qcnn_cir.png

.. py:class:: pyvqnet.qnn.qcnn.qconv.QConv(input_channels,output_channels,quantum_number,stride=(1, 1),padding=(0, 0),kernel_initializer=normal,machine:str = "CPU", dtype=None, name ="")

    Quanten-Faltungsmodul. Ersetzt den Conv2D-Kernel durch einen Quantenschaltkreis. Die Eingabe des Faltungsmoduls hat die Form (Batch-Größe, Eingabekanäle, Höhe, Breite) `Samuel et al. (2020) <https://arxiv.org/abs/2012.12177>`_ .

        :param input_channels: ``int`` - Anzahl der Eingabekanäle.
        :param output_channels: ``int`` - Anzahl der Ausgabekanäle.
        :param quantum_number: ``int`` - Die Größe eines einzelnen Kernels.
        :param stride: ``tuple`` - Der Schritt, Standardwert (1,1).
        :param padding: ``tuple`` - Padding, Standardwert (0,0).
        :param kernel_initializer: ``callable`` - Standardmäßig Normalverteilung.
        :param machine: ``str`` - Die zu verwendende virtuelle Maschine, standardmäßig CPU-Simulation.
        :param dtype: Der Datentyp des Parameters, Standard: None, verwendet den Standard-Datentyp: kfloat32, 32-Bit-Gleitkommazahlen.
        :param name: Der Name dieses Moduls, Standard: „“.

        :return: Quanten-Faltungsschicht.


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

Quantenlogikgatter
************************************

Die Art und Weise, Quantenbits zu verarbeiten, sind Quantenlogikgatter. Mit Quantenlogikgattern entwickeln wir bewusst Quantenzustände weiter. Quantenlogikgatter sind die Grundlage von Quantenalgorithmen.

Basis-Quantenlogikgatter
=============================

In diesem Abschnitt verwenden wir die verschiedenen Logikgatter von pyqpanda3, die von Origin Quantum entwickelt wurden, um Quantenschaltkreise zu erstellen und Quantensimulationen durchzuführen.
Die derzeit von pyQPanda3 unterstützten Logikgatter finden Sie in der Definition der pyQPanda3-Quantenlogikgatter.
Darüber hinaus kapselt VQNet einige häufig verwendete Quantenlogikgatter-Kombinationen im Quantenmaschinellen Lernen:


BasicEmbeddingCircuit
============================

.. py:function:: pyvqnet.qnn.pq3.template.BasicEmbeddingCircuit(input_feat,qlist)
    
    Kodiert n binäre Merkmale in den Grundzustand von n Qubits.

    Zum Beispiel für ``features=([0, 1, 1])`` ist der Grundzustand :math:`|011 \rangle` in einem Quantensystem.

    :param input_feat: Binäre Eingabe der Größe ``(n)``.
    :param qlist: Qubits zum Erstellen des Vorlagenschaltkreises.
    :return: Quantenschaltkreis.


    Example::

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

    Kodiert :math:`N` Merkmale in den Rotationswinkel von :math:`n` Qubits, wobei :math:`N \leq n`.

    Die Rotation kann gewählt werden als: 'X', 'Y', 'Z', wie durch den Parameter ``rotation`` definiert:

    * ``rotation='X'`` Verwendet das Merkmal als Winkel der RX-Rotation.

    * ``rotation='Y'`` Verwendet das Merkmal als Winkel der RY-Rotation.

    * ``rotation='Z'`` Verwendet das Merkmal als Winkel der RZ-Rotation.

    Die Länge von ``features`` muss kleiner oder gleich der Anzahl der Qubits sein. Wenn die Länge von ``features`` kleiner als die Anzahl der Qubits ist, wendet der Schaltkreis die verbleibenden Rotationsgatter nicht an.

    :param input_feat: numpy-Array, das die Parameter darstellt.
    :param qubits: Qubit-Indizes.
    :param rotation: Welche Rotation verwendet werden soll, Standard ist „X“.
    :return: Quantenschaltkreis.

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
============================

.. py:function:: pyvqnet.qnn.pq3.template.IQPEmbeddingCircuits(input_feat,qubits,rep:int = 1)
    
    Kodiert :math:`n` Merkmale in :math:`n` Qubits unter Verwendung diagonaler Gatter eines IQP-Schaltkreises.

    Die Kodierung wurde von `Havlicek et al. (2018) <https://arxiv.org/pdf/1804.11326.pdf>`_ vorgeschlagen.

    Der grundlegende IQP-Schaltkreis kann durch Angabe von ``n_repeats`` wiederholt werden.

    :param input_feat: numpy-Array, das die Parameter darstellt.
    :param qubits: Liste der Qubit-Indizes.
    :param rep: Wiederholung des Quantenschaltkreisblocks, die Standardanzahl ist 1.
    :return: Quantenschaltkreis.

    Example::

        import numpy as np
        from pyvqnet.qnn.pq3.template import IQPEmbeddingCircuits
        input_feat = np.arange(1,100)
        qlist = range(3)
        circuit = IQPEmbeddingCircuits(input_feat,qlist,rep = 3)
        print(circuit)


RotCircuit
============================

.. py:function:: pyvqnet.qnn.pq3.template.RotCircuit(para,qubits)

    Beliebige Einzel-Qubit-Rotation. Die Anzahl der qlists sollte 1 sein, und die Anzahl der Parameter sollte 3 sein.

    .. math::

        R(\phi,\theta,\omega) = RZ(\omega)RY(\theta)RZ(\phi)= \begin{bmatrix}
        e^{-i(\phi+\omega)/2}\cos(\theta/2) & -e^{i(\phi-\omega)/2}\sin(\theta/2) \\
        e^{-i(\phi-\omega)/2}\sin(\theta/2) & e^{i(\phi+\omega)/2}\cos(\theta/2)
        \end{bmatrix}.

    :param para: numpy-Array, das die Parameter :math:`[\phi, \theta, \omega]` darstellt.
    :param qubits: Qubit-Index, es werden nur einzelne Qubits akzeptiert.
    :return: Quantenschaltkreis.

    Example::

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

    Kontrollierter Rot-Operator.

    .. math:: 
        
        CR(\phi, \theta, \omega) = \begin{bmatrix}
        1 & 0 & 0 & 0 \\
        0 & 1 & 0 & 0\\
        0 & 0 & e^{-i(\phi+\omega)/2}\cos(\theta/2) & -e^{i(\phi-\omega)/2}\sin(\theta/2)\\
        0 & 0 & e^{-i(\phi-\omega)/2}\sin(\theta/2) & e^{i(\phi+\omega)/2}\cos(\theta/2)
        \end{bmatrix}.

    :param para: Ein numpy-Array, das die Parameter :math:`[\phi, \theta, \omega]` darstellt.
    :param control_qubits: Qubit-Index, die Anzahl der Qubits sollte 1 sein.
    :param rot_qubits: Rotations-Qubit-Index, die Anzahl der Qubits sollte 1 sein.
    :return: Quantenschaltkreis.

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
============================

.. py:function:: pyvqnet.qnn.pq3.template.CSWAPcircuit(qubits)

    Kontrollierte SWAP-Schaltung.

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
        
        Das erste bereitgestellte Qubit entspricht dem **Kontroll-Qubit**.

    :param qubits: Liste der Qubit-Indizes. Das erste Qubit ist das Kontroll-Qubit. Die Länge von qlist muss 3 sein.
    :return: Der Quantenschaltkreis.

    Example::

        from pyvqnet.qnn.pq3 import CSWAPcircuit
        import pyqpanda3.core as pq
        m_machine = pq.CPUQVM()

        m_qlist = range(3)

        c =CSWAPcircuit([m_qlist[1],m_qlist[2],m_qlist[0]])
        print(c)


Controlled_Hadamard
=======================

.. py:function:: pyvqnet.qnn.pq3.template.Controlled_Hadamard(qubits)
    
    Kontrolliertes Hadamard-Logikgatter

    .. math:: 
        CH = \begin{bmatrix}
        1 & 0 & 0 & 0 \\
        0 & 1 & 0 & 0 \\
        0 & 0 & \frac{1}{\sqrt{2}} & \frac{1}{\sqrt{2}} \\
        0 & 0 & \frac{1}{\sqrt{2}} & -\frac{1}{\sqrt{2}}
        \end{bmatrix}.

    :param qubits: Qubit-Index.

    Examples::

        import pyqpanda3.core as pq

        machine = pq.CPUQVM()
        
        qubits =range(2)
        from pyvqnet.qnn.pq3 import Controlled_Hadamard

        cir = Controlled_Hadamard(qubits)
        print(cir)

CCZ
==============

.. py:function:: pyvqnet.qnn.pq3.template.CCZ(qubits)

    Kontrolliert-kontrolliertes Z-Logikgatter.

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

    :param qubits: Qubit-Index.

    :return:
        pyQPanda3 QCircuit

    Example::

        import pyqpanda3.core as pq

        machine = pq.CPUQVM()

        qubits = range(3)

        from pyvqnet.qnn.pq3 import CCZ

        cir = CCZ(qubits)


FermionicSingleExcitation
===========================

.. py:function:: pyvqnet.qnn.pq3.template.FermionicSingleExcitation(weight, wires, qubits)

    Gekoppelter Cluster-Single-Excitation-Operator für die Exponentiation von Tensorprodukten von Pauli-Matrizen. Die Matrixform ist gegeben durch:

    .. math::

        \hat{U}_{pr}(\theta) = \mathrm{exp} \{ \theta_{pr} (\hat{c}_p^\dagger \hat{c}_r
        -\mathrm{H.c.}) \},

    :param weight: Variabler Parameter auf Qubit p.
    :param wires: Stellt eine Teilmenge von Qubit-Indizes im Intervall [r, p] dar. Die Mindestlänge muss 2 sein. Der erste Indexwert wird als r interpretiert und der letzte Indexwert als p.
        Die dazwischenliegenden Indizes werden von CNOT-Gattern verwendet, um die Parität der Qubit-Menge zu berechnen.
    :param qubits: Qubit-Indizes.

    :return:
        pyQPanda3 QCircuit

    Examples::

        from pyvqnet.qnn.pq3 import FermionicSingleExcitation, expval

        weight=0.5
        import pyqpanda3.core as pq
        machine = pq.CPUQVM()

        qlists = range(3)

        cir = FermionicSingleExcitation(weight, [1, 0, 2], qlists)


FermionicDoubleExcitation
===========================

.. py:function:: pyvqnet.qnn.pq3.template.FermionicDoubleExcitation(weight, wires1, wires2, qubits)

    Gekoppelter Cluster-Double-Excitation-Operator für das Tensorprodukt von Pauli-Matrizen. Die Matrixform ist gegeben durch:

    .. math::

        \hat{U}_{pqrs}(\theta) = \mathrm{exp} \{ \theta (\hat{c}_p^\dagger \hat{c}_q^\dagger
        \hat{c}_r \hat{c}_s - \mathrm{H.c.}) \},

    wobei :math:`\hat{c}` und :math:`\hat{c}^\dagger` die Fermionen-Vernichtungs- und
    Erzeugungsoperatoren sind und die Indizes :math:`r, s` und :math:`p, q` auf besetzten bzw.
    leeren Molekülorbitalen liegen. Unter Verwendung der `Jordan-Wigner-Transformation
    <https://arxiv.org/abs/1208.5986>`_ kann der oben definierte Fermionenoperator in
    Form der Pauli-Matrix geschrieben werden (siehe
    `arXiv:1805.04340 <https://arxiv.org/abs/1805.04340>`_ für weitere Details):

    .. math::

        \hat{U}_{pqrs}(\theta) = \mathrm{exp} \Big\{
        \frac{i\theta}{8} \bigotimes_{b=s+1}^{r-1} \hat{Z}_b \bigotimes_{a=q+1}^{p-1}
        \hat{Z}_a (\hat{X}_s \hat{X}_r \hat{Y}_q \hat{X}_p +
        \hat{Y}_s \hat{X}_r \hat{Y}_q \hat{Y}_p +\\ \hat{X}_s \hat{Y}_r \hat{Y}_q \hat{Y}_p +
        \hat{X}_s \hat{X}_r \hat{X}_q \hat{Y}_p - \mathrm{H.c.} ) \Big\}

    :param weight: Variabler Parameter.
    :param wires1: Stellt die Teilmenge der Qubits im Intervall [s, r] dar, die von der Indexliste der Qubits belegt wird. Der erste Index wird als s und der letzte Index als r interpretiert. Das CNOT-Gatter operiert auf den mittleren Indizes, um die Parität einer Menge von Qubits zu berechnen.
    :param wires2: Stellt die Teilmenge der Qubits im Intervall [q, p] dar, die von der Indexliste der Qubits belegt wird. Der erste Wurzelindex wird als q und der letzte Index als p interpretiert. Das CNOT-Gatter operiert auf dem mittleren Index, um die Parität einer Menge von Qubits zu berechnen.
    :param qubits: Qubit-Indizes.

    :return:
        pyQPanda3 QCircuit

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
===================

.. py:function:: pyvqnet.qnn.pq3.template.UCCSD(weights, wires, s_wires, d_wires, init_state, qubits)

    Implementiert die Unitär Gekoppelte Cluster Single und Double Excitation Simulation (UCCSD). UCCSD ist eine VQE-Simulation, die häufig für quantenchemische Simulationen verwendet wird.

    In der Trotter-Näherung erster Ordnung ist die unitäre UCCSD-Funktion gegeben durch:

    .. math::

        \hat{U}(\vec{\theta}) =
        \prod_{p > r} \mathrm{exp} \Big\{\theta_{pr}
        (\hat{c}_p^\dagger \hat{c}_r-\mathrm{H.c.}) \Big\}
        \prod_{p > q > r > s} \mathrm{exp} \Big\{\theta_{pqrs}
        (\hat{c}_p^\dagger \hat{c}_q^\dagger \hat{c}_r \hat{c}_s-\mathrm{H.c.}) \Big\}
    
    wobei :math:`\hat{c}` und :math:`\hat{c}^\dagger` die Fermionen-Vernichtungs- und

    Erzeugungsoperatoren sind und die Indizes :math:`r, s` und :math:`p, q` die besetzten bzw.
    leeren Molekülorbitale sind. (Für weitere Details siehe
    `arXiv:1805.04340 <https://arxiv.org/abs/1805.04340>`_):

    :param weights: Tensor der Größe ``(len(s_wires)+ len(d_wires))``, der die Parameter
     :math:`\theta_{pr}` und :math:`\theta_{pqrs}` enthält, die den Z-Rotationen ``FermionicSingleExcitation`` und ``FermionicDoubleExcitation`` eingegeben werden.
    :param wires: Qubit-Indizes, die als Vorlage verwendet werden sollen.
    :param s_wires: Listensequenz mit Qubit-Indizes ``[r,...,p]``, die durch eine einzelne Anregung erzeugt wird
     :math:`\vert r, p \rangle = \hat{c}_p^\dagger \hat{c}_r \vert \mathrm{HF} \rangle`, wobei :math:`\vert \mathrm{HF} \rangle` den Hartree-Fock-Referenzzustand bezeichnet.
    :param d_wires: Sequenz von Listen, die jeweils zwei Listen enthalten. Gibt Indizes ``[s, ...,r]`` und ``[q,..., p]`` an. Definiert die doppelte Anregung :math:`\vert s, r, q, p \rangle = \hat{c}_p^\dagger \hat{c}_q^\dagger \hat{c}_r\hat{c}_s \vert \mathrm{HF} \rangle`.
    :param init_state: Besetzungszahl-Vektor der Länge ``len(wires)``, der den HF-Zustand darstellt. ``init_state`` initialisiert den Qubit-Zustand.
    :param qubits: Qubit-Index.

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
============================

.. py:function:: pyvqnet.qnn.pq3.template.QuantumPoolingCircuit(sources_wires, sinks_wires, params,qubits)

    Quantenschaltkreis, der Daten herunterabtastet.

    Um die Anzahl der Qubits im Schaltkreis zu reduzieren, werden zunächst Paare von Qubits im System erstellt. Nach dem anfänglichen Paaren aller Qubits wird die verallgemeinerte 2-Qubit-Unitäre auf jedes Qubit-Paar angewendet. Nach Anwendung dieser zwei Qubit-Unitären wird ein Qubit in jedem Qubit-Paar für den Rest des neuronalen Netzwerks ignoriert.

    :param sources_wires: Quell-Qubit-Indizes, die ignoriert werden sollen.
    :param sinks_wires: Ziel-Qubit-Indizes, die beibehalten werden sollen.
    :param params: Eingabeparameter.
    :param qubits: Qubit-Indizes.

    :return:
        pyQPanda3 QCircuit

    Examples::

        from pyvqnet.qnn.pq3.template import QuantumPoolingCircuit
        import pyqpanda3.core as pq
        from pyvqnet import tensor

        qlists = range(4)
        p = tensor.full([6], 0.35)
        cir = QuantumPoolingCircuit([0, 1], [2, 3], p, qlists)
        print(cir)

Häufig verwendete Quantenschaltkreis-Kombinationen
***********************************************************
VQNet stellt einige Quantenschaltkreise bereit, die in der Forschung zum quantenmaschinellen Lernen häufig verwendet werden.

HardwareEfficientAnsatz
=============================

.. py:class:: pyvqnet.qnn.pq3.ansatz.HardwareEfficientAnsatz(qubits,single_rot_gate_list,entangle_gate="CNOT",entangle_rules='linear',depth=1)

    Implementierung des Hardwareeffizienten Ansatzes, eingeführt in der Arbeit: `Hardware-efficient Variational Quantum Eigensolver for Small Molecules <https://arxiv.org/pdf/1704.05018.pdf>`__ .

    :param qubits: Qubit-Index.
    :param single_rot_gate_list: Liste von Einzel-Qubit-Rotationsgattern, bestehend aus einem oder mehreren Rotationsgattern, die auf jedes Qubit wirken. Derzeit unterstützt werden Rx, Ry, Rz.
    :param entangle_gate: Nicht-parametrisches Verschränkungsgatter. Unterstützt CNOT, CZ. Standard: CNOT.
    :param entangle_rules: Wie das Verschränkungsgatter im Schaltkreis verwendet wird. ``linear`` bedeutet, dass das Verschränkungsgatter auf jedes benachbarte Qubit wirkt. ``all`` bedeutet, dass das Verschränkungsgatter auf beliebige zwei Qubits wirkt. Standard: ``linear``.
    :param depth: Tiefe des Ansatzes, Standard: 1.

    :return:
        Eine HardwareEfficientAnsatz-Instanz

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
============================

.. py:class:: pyvqnet.qnn.pq3.template.BasicEntanglerTemplate(weights=None, num_qubits=1, rotation=pyqpanda3.RX)
    
    Eine Schicht bestehend aus Einzelparameter-Einzel-Qubit-Rotationen auf jedem Qubit, gefolgt von mehreren CNOT-Gattern, die in einer geschlossenen Kette oder einem Ring kombiniert sind.
    Der Ring von CNOT-Gattern verbindet jedes Qubit mit seinen Nachbarn, wobei das letzte Qubit als Nachbar des ersten betrachtet wird.
    Die Anzahl der Schichten :math:`L` wird durch die erste Dimension des Parameters ``weights`` bestimmt.

    :param weights: Ein Gewichtstensor der Form `(L, len(qubits))`. Jedes Gewicht wird als Parameter in einem quantenparametrischen Gatter verwendet. Der Standardwert ist: ``None``, dann werden `(1,1)` normalverteilte Zufallszahlen als Gewichte verwendet.
    :param num_qubits: Die Anzahl der Qubits, Standard ist 1.
    :param rotation: Verwendet ein Einzelparameter-Einzel-Qubit-Gatter, ``pyqpanda3.RX`` wird als Standardwert verwendet.
    :return:
        Eine BasicEntanglerTemplate-Instanz

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
============================

.. py:class:: pyvqnet.qnn.pq3.template.StronglyEntanglingTemplate(weights=None, num_qubits=1, ranges=None)

    Schichten bestehend aus Einzel-Qubit-Rotationen und Entanglern, wie im `Circuit-Centric Classifier Design <https://arxiv.org/abs/1804.00633>`__ .
    Der Parameter ``weights`` enthält die Gewichte jeder Schicht. Die Anzahl der Schichten :math:`L` ist gleich der ersten Dimension von ``weights``.
    Es enthält 2-Qubit-CNOT-Gatter, die auf :math:`M` Qubits wirken, :math:`i = 1,...,M`. Die zweite Qubit-Nummer jedes Gatters ist durch die Formel :math:`(i+r)\mod M` gegeben, wobei :math:`r` ein Hyperparameter namens ``range`` ist und :math:`0 < r < M`.

    :param weights: Gewichtstensor der Form ``(L, M, 3)``, Standardwert: None, verwendet einen Zufallstensor der Form ``(1,1,3)``.
    :param num_qubits: Anzahl der Qubits, Standardwert: 1.
    :param ranges: Sequenz, die den Bereichshyperparameter für jede nachfolgende Schicht bestimmt; Standardwert: None, verwendet :math:`r=l \mod M` als Wert von ranges.
    :return:
        Eine StronglyEntanglingTemplate-Instanz

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
============================

.. py:class:: pyvqnet.qnn.pq3.ComplexEntanglingTemplate(weights,num_qubits,depth)

    Stark verschränkte Schicht bestehend aus U3-Gattern und CNOT-Gattern.
    Diese Schaltkreisvorlage stammt aus der folgenden Arbeit: https://arxiv.org/abs/1804.00633.

    :param weights: Parameter, Form [depth, num_qubits, 3].
    :param num_qubits: Anzahl der Qubits.
    :param depth: Tiefe des Unterschaltkreises.
    :return:
        Eine ComplexEntanglingTemplate-Instanz

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
============================

.. py:class:: pyvqnet.qnn.pq3.Quantum_Embedding(qubits, machine, num_repetitions_input, depth_input, num_unitary_layers, num_repetitions)

    Verwendet RZ, RY, RZ, um einen variationellen Quantenschaltkreis zu erstellen, der klassische Daten in Quantenzustände kodiert.
    Referenz: `Quantum embeddings for machine learning <https://arxiv.org/abs/2001.03622>`_.

    Nach der Initialisierung der Klasse ist ihre Memberfunktion ``compute_circuit`` die Ausführungsfunktion, die als Parameter für die ``QuantumLayerV2``-Klasse verwendet werden kann, um eine Schicht des Quantenmaschinenlernmodells zu bilden.

    :param qubits: Die von pyQPanda3 angeforderten Quantenbits.
    :param machine: Von pyQPanda3 angeforderte Quantenvirtuelle Maschine.
    :param num_repetitions_input: Die Anzahl der Wiederholungen der Kodierung der Eingabe im Untermodul.
    :param depth_input: Die Merkmalsdimension der Eingabedaten.
    :param num_unitary_layers: Die Anzahl der Wiederholungen des variationellen Quantengatters in jedem Untermodul.
    :param num_repetitions: Die Anzahl der Wiederholungen des Untermoduls.
    :return:
        Eine Quantum_Embedding-Instanz

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


Messung von Quantenschaltkreisen
************************************

expval
============================

.. py:function:: pyvqnet.qnn.pq3.measure.expval(machine,prog,pauli_str_dict,shots=1000,noise_model=None)

    Der Erwartungswert der bereitgestellten Hamiltonian-Beobachtung.

    Wenn die Observablen :math:`0.7Z\otimes X\otimes I+0.2I\otimes Z\otimes I` ist,
    dann ist das Hamiltonian-Wörterbuch ``{{'Z0, X1':0.7} ,{'Z1':0.2}}``.

    Die expval-API unterstützt jetzt den pyQPanda3-Simulator.

    :param machine: Die von pyQPanda3 erstellte Quantenmaschine.
    :param prog: Das von pyQPanda3 erstellte Quantenprogramm.
    :param pauli_str_dict: Hamiltonian-Beobachtungswert.
    :param shots: Anzahl der Messungen, Standard ist 1000.
    :param noise_model: Anzuwendendes Rauschmodell, Standard ist None (ideale Simulation).

    :return: Erwartungswert.

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
=============================

.. py:function:: pyvqnet.qnn.pq3.measure.QuantumMeasure(machine,prog,measure_qubits:list,shots:int = 1000, qcloud_option="",noise_model=None)

    Berechnet Quantenschaltkreis-Messungen. Gibt Messungen zurück, die mit Monte-Carlo-Methoden erhalten wurden.

    Weitere Einzelheiten zur Messung finden Sie in der pyQPanda3-Dokumentation.

    Die QuantumMeasure-API unterstützt derzeit nur pyQPanda3 ``CPUQVM`` oder ``QCloud``.

    :param machine: Die von pyQPanda3 zugewiesene Quantenvirtuelle Maschine.
    :param prog: Das von pyQPanda3 erstellte Quantenprogramm.
    :param measure_qubits: Liste mit den Messbit-Indizes.
    :param shots: Die Anzahl der Messungen, der Standardwert ist 1000.
    :param qcloud_option: Setzt die qcloud-Konfiguration, der Standardwert ist „“, Sie können eine QCloudOptions-Klasse übergeben, die nur bei Verwendung von qcloud nützlich ist.
    :param noise_model: Anzuwendendes Rauschmodell, Standard ist None (ideale Simulation).
    :return: Gibt die mit der Monte-Carlo-Methode erhaltenen Messergebnisse zurück.

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
============================

.. py:function:: pyvqnet.qnn.pq3.measure.ProbsMeasure(machine,prog,measure_qubits:list,shots=1,noise_model=None)

    Berechnet Wahrscheinlichkeitsmessungen des Schaltkreises.

    Weitere Einzelheiten finden Sie in der pyQPanda3-Dokumentation zur Wahrscheinlichkeitsmessung.

    Die ProbsMeasure-API unterstützt derzeit nur pyQPanda ``CPUQVM`` oder ``QCloud``.

    :param measure_qubits: Liste mit den Mess-Qubit-Indizes.
    :param prog: Das von qpanda erstellte Quantenprogramm.
    :param machine: Die von pyQPanda zugewiesene Quantenvirtuelle Maschine.
    :param shots: Anzahl der Messungen, Standard ist 1, was den theoretischen Wert berechnet.
    :param noise_model: Anzuwendendes Rauschmodell, Standard ist None (ideale Simulation).
    :return: Misst Qubits in lexikografischer Reihenfolge.


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
===========================
.. py:function:: pyvqnet.qnn.pq3.measure.DensityMatrixFromQstate(state, indices)
    
    Berechnet die Dichtematrix eines Quantenzustands über einer bestimmten Menge von Qubits.

    :param state: 1D-Liste der Zustandsvektoren. Die Größe dieser Liste sollte ``(2**N,)`` sein. Für eine Anzahl von Qubits ``N`` sollte qstate von 000 -> 111 beginnen.
    :param indices: Liste der Qubit-Indizes im betrachteten Subsystem.
    :return: 
        Dichtematrix der Größe ``(2**len(indices), 2**len(indices))``.

    Example::

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
    
    Berechnet die Von-Neumann-Entropie für einen gegebenen Zustandsvektor auf einer gegebenen Liste von Qubits.

    .. math::

        S( \rho ) = -\text{Tr}( \rho \log ( \rho ))

    :param state: 1D-Liste der Zustandsvektoren. Die Größe dieser Liste sollte ``(2**N,)`` sein. Für eine Anzahl von Qubits ``N`` sollte qstate von 000 -> 111 beginnen.
    :param indices: Liste der Qubit-Indizes im betrachteten Subsystem.
    :param base: Basis des Logarithmus. Wenn None, wird der natürliche Logarithmus verwendet.

    :return: Gleitkommawert der Von-Neumann-Entropie.

    Example::

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

    Berechnet die gegenseitige Information für einen gegebenen Zustandsvektor auf zwei Listen von Sub-Qubits.

    .. math::

        I(A, B) = S(\rho^A) + S(\rho^B) - S(\rho^{AB})
        where :math:`S` is the von Neumann entropy.

    Die gegenseitige Information ist ein Maß für die Korrelation zwischen zwei Subsystemen. Genauer gesagt quantifiziert sie die Informationsmenge, die ein System durch Messung des anderen gewinnen kann.

    Jeder Zustand kann als Zustandsvektor in der Berechnungsbasis angegeben werden.

    :param state: 1D-Liste der Zustandsvektoren. Die Größe dieser Liste sollte ``(2**N,)`` sein. Für eine Anzahl von Qubits ``N`` sollte qstate von 000 -> 111 beginnen.
    :param indices0: Liste der Qubit-Indizes im ersten Subsystem.
    :param indices1: Liste der Qubit-Indizes im zweiten Subsystem.
    :param base: Basis der Logarithmen. Wenn None, werden natürliche Logarithmen verwendet. Standard ist None.

    :return: Gegenseitige Information zwischen Subsystemen.

    Example::

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

    Berechnet die Reinheit eines bestimmten Qubits aus dem Zustandsvektor.

    .. math::

        \gamma = \text{Tr}(\rho^2)
        
    wobei :math:`\rho` die Dichtematrix ist. Die Reinheit eines normalisierten Quantenzustands erfüllt :math:`\frac{1}{d} \leq \gamma \leq 1`,
    wobei :math:`d` die Dimension des Hilbertraums ist.
    Die Reinheit eines reinen Zustands ist 1.

    :param state: Von pyqpanda3 erhaltener Quantenzustand.
    :param qubits_idx: Qubit-Index, für den die Reinheit berechnet werden soll.

    :return:
        Reinheit

    Examples::

        from pyvqnet.qnn.pq3.measure import Purity
        qstate = [(0.9306699299765968 + 0j), (0.18865613455240968 + 0j),
        (0.1886561345524097 + 0j), (0.03824249173404786 + 0j),
        -0.048171819846746615j, -0.00976491131165138j, -0.23763904794287155j,
        -0.048171819846746615j]
        pp = Purity(qstate, [1])
        print(pp)
        #0.902503479761881
