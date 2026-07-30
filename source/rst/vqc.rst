.. _vqc_api:

API dei Circuiti Quantistici Variazionali ad Autograd
******************************************************************************

VQNet si basa sulla costruzione di operatori a differenziazione automatica e su alcuni dei piu comuni gate logici quantistici, circuiti quantistici e metodi di misurazione. La differenziazione automatica puo essere utilizzata per calcolare i gradienti al posto del metodo di parameter-shift dei circuiti quantistici.
Possiamo usare gli operatori VQC per formare reti neurali complesse come altri `Modules`. La macchina virtuale `QMachine` deve essere definita in `Module`, e gli `states` nella macchina devono essere sottoposti a reset_states in base al batchsize di input. Si veda il seguente esempio per i dettagli.

.. code-block::

    from pyvqnet.nn import Module,Linear,ModuleList
    from pyvqnet.qnn.vqc.qcircuit import VQC_HardwareEfficientAnsatz,RZZ,RZ
    from pyvqnet.qnn.vqc import Probability,QMachine
    from pyvqnet import tensor

    class QM(Module):
        def __init__(self, name=""):
            super().__init__(name)
            self.linearx = Linear(4,2)
            self.ansatz = VQC_HardwareEfficientAnsatz(4, ["rx", "RY", "rz"],
                                        entangle_gate="cnot",
                                        entangle_rules="linear",
                                        depth=2)
            #VQC basato su RZ sul bit 0
            self.encode1 = RZ(wires=0)
            #VQC basato su RZ sul bit 1
            self.encode2 = RZ(wires=1)
            #Misura di probabilita basata su VQC sui bit 0, 2
            self.measure = Probability(wires=[0,2])
            #Dispositivo quantistico QMachine, utilizza 4 bit.
            self.device = QMachine(4)
        def forward(self, x, *args, **kwargs):
            #Gli stati devono essere resettati allo stesso batchsize dell'input.
            self.device.reset_states(x.shape[0])
            y = self.linearx(x)
            #Codifica l'input nel gate RZ. Nota: l'input deve avere forma [batchsize,1]
            self.encode1(params = y[:, [0]],q_machine = self.device,)
            #Codifica l'input nel gate RZ. Nota: l'input deve avere forma [batchsize,1]
            self.encode2(params = y[:, [1]],q_machine = self.device,)
            self.ansatz(q_machine =self.device)
            return self.measure(q_machine =self.device)

    bz=3
    inputx = tensor.arange(1.0,bz*4+1).reshape([bz,4])
    inputx.requires_grad= True
    #Definito come altri Modules
    qlayer = QM()
    #Pre-esecuzione
    y = qlayer(inputx)

    y.backward()
    print(y)


L'esempio seguente mostra il calcolo quantistico variazionale su GPU (inclusa la codifica dei dati e i circuiti variazionali parametrizzati):

.. code-block::

    from pyvqnet.nn import Module,Linear,ModuleList
    from pyvqnet.qnn.vqc.qcircuit import VQC_HardwareEfficientAnsatz,RZZ,RZ,rz,ry,cnot
    from pyvqnet.qnn.vqc import Probability,QMachine
    from pyvqnet import tensor
    from pyvqnet import DEV_GPU
    class QM(Module):
        def __init__(self, name=""):
            super().__init__(name)
            self.linearx = Linear(4,2)
            self.ansatz = VQC_HardwareEfficientAnsatz(4, ["rx", "RY", "rz"],
                                        entangle_gate="cnot",
                                        entangle_rules="linear",
                                        depth=2)
            #VQC basato su RZ sul bit 0
            self.encode1 = RZ(wires=0)
            #VQC basato su RZ sul bit 1
            self.encode2 = RZ(wires=1)
            #RZ con parametri addestrabili has_params = True, trainable = True
            self.vqc = RZ(has_params = True,trainable = True,wires=1)
            #Misura di probabilita basata su VQC sui bit 0, 2
            self.measure = Probability(wires=[0,2])
            #Dispositivo quantistico QMachine, utilizza 4 bit.
            self.device = QMachine(4)
        def forward(self, x, *args, **kwargs):
            #Gli stati devono essere resettati allo stesso batchsize dell'input.
            self.device.reset_states(x.shape[0])
            y = self.linearx(x)
            #Codifica l'input nel gate RZ. Nota: l'input deve avere forma [batchsize,1]
            self.encode1(params = y[:, 0],q_machine = self.device,)
            #Codifica l'input nel gate RZ. Nota: l'input deve avere forma [batchsize,1]
            self.encode2(params = y[:, 1],q_machine = self.device,)
            #Circuito variazionale composto da gate RZ, sara incluso nell'addestramento.
            self.vqc(q_machine =self.device)
            self.ansatz(q_machine =self.device)
            return self.measure(q_machine =self.device)

    bz =3
    #crea tensor su GPU
    inputx = tensor.arange(1.0,bz*4+1,device=DEV_GPU).reshape([bz,4])
    inputx.requires_grad= True
    #Definito come altri Modules
    qlayer = QM()
    #sposta il modulo su GPU
    qlayer = qlayer.to(DEV_GPU)
    #Forward
    y = qlayer(inputx)
    #Backward
    y.backward()
    print(y)



Simulatore
=======================================

QMachine
-------------------------------

.. py:class:: pyvqnet.qnn.vqc.QMachine(num_wires, dtype=pyvqnet.kcomplex64)

    Classe simulatore per il calcolo quantistico variazionale, che include vettori di stato il cui attributo states e un circuito quantistico.

    :param num_wires: numero di qubit.
    :param dtype: tipo di dato dei dati calcolati, il default e pyvqnet.kcomplex64, e la precisione del parametro corrispondente e pyvqnet.kfloat32

    :return: Output QMachine.

    Example::
        
        from pyvqnet.qnn.vqc import QMachine
        qm  = QMachine(4)

        print(qm.states)

        # [[[[[1.+0.j 0.+0.j]
        #     [0.+0.j 0.+0.j]]

        #    [[0.+0.j 0.+0.j]
        #     [0.+0.j 0.+0.j]]]


        #   [[[0.+0.j 0.+0.j]
        #     [0.+0.j 0.+0.j]]

        #    [[0.+0.j 0.+0.j]
        #     [0.+0.j 0.+0.j]]]]]


    .. py:method:: reset_states(batchsize)
    
        Reinizializza lo stato iniziale nel simulatore e lo trasmette alle
        dimensioni (batchsize,[2]**num_qubits) per adattarsi all'addestramento su dati batch.

        :param batchsize: dimensione del batch.


Gate Quantistici e Operazioni sui Gate Quantistici
==================================================


i
-------------------------------

.. py:function:: pyvqnet.qnn.vqc.i(q_machine, wires, params=None,  use_dagger=False)

    Applica il gate logico quantistico I ai vettori di stato in ``q_machine``.

    :param q_machine: dispositivo macchina virtuale quantistica.
    :param wires: indice del qubit.
    :param params: matrice dei parametri, default None. Per una funzione operativa di gate logico con p parametri, la dimensione del parametro di input deve essere [1,p] o [p].
    :param use_dagger: se utilizzare la trasposta coniugata, default False.


    Example::
        
        from pyvqnet.qnn.vqc import i,QMachine
        qm  = QMachine(4)
        i(q_machine=qm, wires=1)
        print(qm.states)
        # [[[[[1.+0.j 0.+0.j]
        #     [0.+0.j 0.+0.j]]

        #    [[0.+0.j 0.+0.j]
        #     [0.+0.j 0.+0.j]]]


        #   [[[0.+0.j 0.+0.j]
        #     [0.+0.j 0.+0.j]]

        #    [[0.+0.j 0.+0.j]
        #     [0.+0.j 0.+0.j]]]]]

I
---------------------------------------------------------------

.. py:class:: pyvqnet.qnn.vqc.I(has_params: bool = False,trainable: bool = False,init_params=None,wires=None,dtype=pyvqnet.kcomplex64,use_dagger=False)
    
    Definisce una classe di gate logico I.

    :param has_params: Indica se ci sono parametri. Gate come RX, RY e altri devono essere impostati a True, quelli senza parametri devono essere impostati a False, default False.
    :param trainable: Indica se contiene parametri da addestrare. Se il layer usa dati di input esterni per costruire una matrice di gate logico, impostare a False. Se contiene parametri da addestrare, e True. Default False.
    :param init_params: Parametri di inizializzazione, usati per codificare dati classici QTensor, default None. Se e un gate logico con parametri con p parametri, la dimensione del dato di input deve essere [1,p] o [p].
    :param wires: Indice del bit su cui agisce il gate, default None.
    :param dtype: La precisione dei dati della matrice interna del gate logico puo essere impostata a pyvqnet.kcomplex64 o pyvqnet.kcomplex128, corrispondenti rispettivamente a parametri di input float o double.
    :param use_dagger: Se utilizzare la versione trasposta coniugata di questo gate, default False.
    :return: Un Module che puo essere usato per addestrare il modello.

    Example::
    
        from pyvqnet.qnn.vqc import I,QMachine
        device = QMachine(4)
        layer = I(wires=0)
        batchsize=1
        device.reset_states(1)
        layer(q_machine = device)
        print(device.states)

hadamard
---------------------------------------------------------------

.. py:function:: pyvqnet.qnn.vqc.hadamard(q_machine, wires, params=None,  use_dagger=False)
    
    Applica il gate logico quantistico hadamard ai vettori di stato in ``q_machine``.

    :param q_machine: dispositivo macchina virtuale quantistica.
    :param wires: indice del qubit.
    :param params: matrice dei parametri, default None. Per una funzione operativa di gate logico con p parametri, la dimensione del parametro di input deve essere [1,p] o [p].
    :param use_dagger: se utilizzare la trasposta coniugata, default False.


    Example::
        
        from pyvqnet.qnn.vqc import hadamard,QMachine
        qm  = QMachine(4)
        hadamard(q_machine=qm, wires=1)
        print(qm.states)
        # [[[[[0.7071068+0.j 0.       +0.j]
        #     [0.       +0.j 0.       +0.j]]
        # 
        #    [[0.7071068+0.j 0.       +0.j]
        #     [0.       +0.j 0.       +0.j]]]
        # 
        # 
        #   [[[0.       +0.j 0.       +0.j]
        #     [0.       +0.j 0.       +0.j]]
        # 
        #    [[0.       +0.j 0.       +0.j]
        #     [0.       +0.j 0.       +0.j]]]]]

Hadamard
---------------------------------------------------------------

.. py:class:: pyvqnet.qnn.vqc.Hadamard(has_params: bool = False,trainable: bool = False,init_params=None,wires=None,dtype=pyvqnet.kcomplex64,use_dagger=False)
    
    Definisce una classe di gate logico Hadamard.

    :param has_params: Indica se ci sono parametri. Gate come RX, RY e altri devono essere impostati a True, quelli senza parametri devono essere impostati a False, default False.
    :param trainable: Indica se contiene parametri da addestrare. Se il layer usa dati di input esterni per costruire una matrice di gate logico, impostare a False. Se contiene parametri da addestrare, e True. Default False.
    :param init_params: Parametri di inizializzazione, usati per codificare dati classici QTensor, default None. Se e un gate logico con parametri con p parametri, la dimensione del dato di input deve essere [1,p] o [p].
    :param wires: Indice del bit su cui agisce il gate, default None.
    :param dtype: La precisione dei dati della matrice interna del gate logico puo essere impostata a pyvqnet.kcomplex64 o pyvqnet.kcomplex128, corrispondenti rispettivamente a parametri di input float o double.
    :param use_dagger: Se utilizzare la versione trasposta coniugata di questo gate, default False.
    :return: Un Module che puo essere usato per addestrare il modello.

    Example::
    
        from pyvqnet.qnn.vqc import Hadamard,QMachine
        device = QMachine(4)
        layer = Hadamard(wires=0)
        batchsize=1
        device.reset_states(1)
        layer(q_machine = device)
        print(device.states)

t
----------------

.. py:function:: pyvqnet.qnn.vqc.t(q_machine, wires, params=None,  use_dagger=False)
    
    Applica il gate logico quantistico t ai vettori di stato in ``q_machine``.

    :param q_machine: dispositivo macchina virtuale quantistica.
    :param wires: indice del qubit.
    :param params: matrice dei parametri, default None. Per una funzione operativa di gate logico con p parametri, la dimensione del parametro di input deve essere [1,p] o [p].
    :param use_dagger: se utilizzare la trasposta coniugata, default False.


    Example::
        
        from pyvqnet.qnn.vqc import t,QMachine
        qm  = QMachine(4)
        t(q_machine=qm, wires=1)
        print(qm.states)
        # [[[[[1.+0.j 0.+0.j]
        #     [0.+0.j 0.+0.j]]
        # 
        #    [[0.+0.j 0.+0.j]
        #     [0.+0.j 0.+0.j]]]
        # 
        # 
        #   [[[0.+0.j 0.+0.j]
        #     [0.+0.j 0.+0.j]]
        # 
        #    [[0.+0.j 0.+0.j]
        #     [0.+0.j 0.+0.j]]]]]

T
---------------------------------------------------------------

.. py:class:: pyvqnet.qnn.vqc.T(has_params: bool = False,trainable: bool = False,init_params=None,wires=None,dtype=pyvqnet.kcomplex64,use_dagger=False)
    
    Definisce una classe di gate logico T.

    :param has_params: Indica se ci sono parametri. Gate come RX, RY e altri devono essere impostati a True, quelli senza parametri devono essere impostati a False, default False.
    :param trainable: Indica se contiene parametri da addestrare. Se il layer usa dati di input esterni per costruire una matrice di gate logico, impostare a False. Se contiene parametri da addestrare, e True. Default False.
    :param init_params: Parametri di inizializzazione, usati per codificare dati classici QTensor, default None. Se e un gate logico con parametri con p parametri, la dimensione del dato di input deve essere [1,p] o [p].
    :param wires: Indice del bit su cui agisce il gate, default None.
    :param dtype: La precisione dei dati della matrice interna del gate logico puo essere impostata a pyvqnet.kcomplex64 o pyvqnet.kcomplex128, corrispondenti rispettivamente a parametri di input float o double.
    :param use_dagger: Se utilizzare la versione trasposta coniugata di questo gate, default False.
    :return: Un Module che puo essere usato per addestrare il modello.

    Example::
    
        from pyvqnet.qnn.vqc import T,QMachine
        device = QMachine(4)
        layer = T(wires=0)
        batchsize=1
        device.reset_states(1)
        layer(q_machine = device)
        print(device.states)

s
------

.. py:function:: pyvqnet.qnn.vqc.s(q_machine, wires, params=None,  use_dagger=False)
    
    Applica il gate logico quantistico s ai vettori di stato in ``q_machine``.

    :param q_machine: dispositivo macchina virtuale quantistica.
    :param wires: indice del qubit.
    :param params: matrice dei parametri, default None. Per una funzione operativa di gate logico con p parametri, la dimensione del parametro di input deve essere [1,p] o [p].
    :param use_dagger: se utilizzare la trasposta coniugata, default False.


    Example::
        
        from pyvqnet.qnn.vqc import s,QMachine
        qm  = QMachine(4)
        s(q_machine=qm, wires=1)
        print(qm.states)
        # [[[[[1.+0.j 0.+0.j]       
        #     [0.+0.j 0.+0.j]]
        # 
        #    [[0.+0.j 0.+0.j]
        #     [0.+0.j 0.+0.j]]]
        # 
        # 
        #   [[[0.+0.j 0.+0.j]
        #     [0.+0.j 0.+0.j]]
        # 
        #    [[0.+0.j 0.+0.j]
        #     [0.+0.j 0.+0.j]]]]]

S
--------------------------------------------------------------

.. py:class:: pyvqnet.qnn.vqc.S(has_params: bool = False,trainable: bool = False,init_params=None,wires=None,dtype=pyvqnet.kcomplex64,use_dagger=False)
    
    Definisce una classe di gate logico S.

    :param has_params: Indica se ci sono parametri. Gate come RX, RY e altri devono essere impostati a True, quelli senza parametri devono essere impostati a False, default False.
    :param trainable: Indica se contiene parametri da addestrare. Se il layer usa dati di input esterni per costruire una matrice di gate logico, impostare a False. Se contiene parametri da addestrare, e True. Default False.
    :param init_params: Parametri di inizializzazione, usati per codificare dati classici QTensor, default None. Se e un gate logico con parametri con p parametri, la dimensione del dato di input deve essere [1,p] o [p].
    :param wires: Indice del bit su cui agisce il gate, default None.
    :param dtype: La precisione dei dati della matrice interna del gate logico puo essere impostata a pyvqnet.kcomplex64 o pyvqnet.kcomplex128, corrispondenti rispettivamente a parametri di input float o double.
    :param use_dagger: Se utilizzare la versione trasposta coniugata di questo gate, default False.
    :return: Un Module che puo essere usato per addestrare il modello.

    Example::
    
        from pyvqnet.qnn.vqc import S,QMachine
        device = QMachine(4)
        layer = S(wires=0)
        batchsize=1
        device.reset_states(1)
        layer(q_machine = device)
        print(device.states)

paulix
---------------

.. py:function:: pyvqnet.qnn.vqc.paulix(q_machine, wires, params=None,  use_dagger=False)
    
    Applica il gate logico quantistico paulix ai vettori di stato in ``q_machine``.

    :param q_machine: dispositivo macchina virtuale quantistica.
    :param wires: indice del qubit.
    :param params: matrice dei parametri, default None. Per una funzione operativa di gate logico con p parametri, la dimensione del parametro di input deve essere [1,p] o [p].
    :param use_dagger: se utilizzare la trasposta coniugata, default False.


    Example::
        
        from pyvqnet.qnn.vqc import paulix,QMachine
        qm  = QMachine(4)
        paulix(q_machine=qm, wires=1)
        print(qm.states)
        # [[[[[0.+0.j 0.+0.j]
        #     [0.+0.j 0.+0.j]]
        # 
        #    [[1.+0.j 0.+0.j]
        #     [0.+0.j 0.+0.j]]]
        # 
        # 
        #   [[[0.+0.j 0.+0.j]
        #     [0.+0.j 0.+0.j]]
        # 
        #    [[0.+0.j 0.+0.j]
        #     [0.+0.j 0.+0.j]]]]]

PauliX
---------------------------------------------------------------

.. py:class:: pyvqnet.qnn.vqc.PauliX(has_params: bool = False,trainable: bool = False,init_params=None,wires=None,dtype=pyvqnet.kcomplex64,use_dagger=False)
    
    Definisce una classe di gate logico PauliX.

    :param has_params: Indica se ci sono parametri. Gate come RX, RY e altri devono essere impostati a True, quelli senza parametri devono essere impostati a False, default False.
    :param trainable: Indica se contiene parametri da addestrare. Se il layer usa dati di input esterni per costruire una matrice di gate logico, impostare a False. Se contiene parametri da addestrare, e True. Default False.
    :param init_params: Parametri di inizializzazione, usati per codificare dati classici QTensor, default None. Se e un gate logico con parametri con p parametri, la dimensione del dato di input deve essere [1,p] o [p].
    :param wires: Indice del bit su cui agisce il gate, default None.
    :param dtype: La precisione dei dati della matrice interna del gate logico puo essere impostata a pyvqnet.kcomplex64 o pyvqnet.kcomplex128, corrispondenti rispettivamente a parametri di input float o double.
    :param use_dagger: Se utilizzare la versione trasposta coniugata di questo gate, default False.
    :return: Un Module che puo essere usato per addestrare il modello.

    Example::
    
        from pyvqnet.qnn.vqc import PauliX,QMachine
        device = QMachine(4)
        layer = PauliX(wires=0)
        batchsize=1
        device.reset_states(1)
        layer(q_machine = device)
        print(device.states)

pauliy
----------------

.. py:function:: pyvqnet.qnn.vqc.pauliy(q_machine, wires, params=None,  use_dagger=False)
    
    Applica il gate logico quantistico pauliy ai vettori di stato in ``q_machine``.

    :param q_machine: dispositivo macchina virtuale quantistica.
    :param wires: indice del qubit.
    :param params: matrice dei parametri, default None. Per una funzione operativa di gate logico con p parametri, la dimensione del parametro di input deve essere [1,p] o [p].
    :param use_dagger: se utilizzare la trasposta coniugata, default False.


    Example::
        
        from pyvqnet.qnn.vqc import pauliy,QMachine
        qm  = QMachine(4)
        pauliy(q_machine=qm, wires=1)
        print(qm.states)
        # [[[[[0.+0.j 0.+0.j]
        #     [0.+0.j 0.+0.j]]
        # 
        #    [[0.+1.j 0.+0.j]
        #     [0.+0.j 0.+0.j]]]
        # 
        # 
        #   [[[0.+0.j 0.+0.j]
        #     [0.+0.j 0.+0.j]]
        # 
        #    [[0.+0.j 0.+0.j]
        #     [0.+0.j 0.+0.j]]]]]

PauliY
---------------------------------------------------------------

.. py:class:: pyvqnet.qnn.vqc.PauliY(has_params: bool = False,trainable: bool = False,init_params=None,wires=None,dtype=pyvqnet.kcomplex64,use_dagger=False)
    
    Definisce una classe di gate logico PauliY.

    :param has_params: Indica se ci sono parametri. Gate come RX, RY e altri devono essere impostati a True, quelli senza parametri devono essere impostati a False, default False.
    :param trainable: Indica se contiene parametri da addestrare. Se il layer usa dati di input esterni per costruire una matrice di gate logico, impostare a False. Se contiene parametri da addestrare, e True. Default False.
    :param init_params: Parametri di inizializzazione, usati per codificare dati classici QTensor, default None. Se e un gate logico con parametri con p parametri, la dimensione del dato di input deve essere [1,p] o [p].
    :param wires: Indice del bit su cui agisce il gate, default None.
    :param dtype: La precisione dei dati della matrice interna del gate logico puo essere impostata a pyvqnet.kcomplex64 o pyvqnet.kcomplex128, corrispondenti rispettivamente a parametri di input float o double.
    :param use_dagger: Se utilizzare la versione trasposta coniugata di questo gate, default False.
    :return: Un Module che puo essere usato per addestrare il modello.

    Example::
    
        from pyvqnet.qnn.vqc import PauliY,QMachine
        device = QMachine(4)
        layer = PauliY(wires=0)
        batchsize=1
        device.reset_states(1)
        layer(q_machine = device)
        print(device.states)


pauliz
-----------------

.. py:function:: pyvqnet.qnn.vqc.pauliz(q_machine, wires, params=None,  use_dagger=False)
    
    Applica il gate logico quantistico pauliz ai vettori di stato in ``q_machine``.

    :param q_machine: dispositivo macchina virtuale quantistica.
    :param wires: indice del qubit.
    :param params: matrice dei parametri, default None. Per una funzione operativa di gate logico con p parametri, la dimensione del parametro di input deve essere [1,p] o [p].
    :param use_dagger: se utilizzare la trasposta coniugata, default False.


    Example::
        
        from pyvqnet.qnn.vqc import pauliz,QMachine
        qm  = QMachine(4)
        pauliz(q_machine=qm, wires=1)
        print(qm.states)
        # [[[[[1.+0.j 0.+0.j]
        #     [0.+0.j 0.+0.j]]
        # 
        #    [[0.+0.j 0.+0.j]
        #     [0.+0.j 0.+0.j]]]
        # 
        # 
        #   [[[0.+0.j 0.+0.j]
        #     [0.+0.j 0.+0.j]]
        # 
        #    [[0.+0.j 0.+0.j]
        #     [0.+0.j 0.+0.j]]]]]

PauliZ
---------------------------------------------------------------

.. py:class:: pyvqnet.qnn.vqc.PauliZ(has_params: bool = False,trainable: bool = False,init_params=None,wires=None,dtype=pyvqnet.kcomplex64,use_dagger=False)
    
    Definisce una classe di gate logico PauliZ.

    :param has_params: Indica se ci sono parametri. Gate come RX, RY e altri devono essere impostati a True, quelli senza parametri devono essere impostati a False, default False.
    :param trainable: Indica se contiene parametri da addestrare. Se il layer usa dati di input esterni per costruire una matrice di gate logico, impostare a False. Se contiene parametri da addestrare, e True. Default False.
    :param init_params: Parametri di inizializzazione, usati per codificare dati classici QTensor, default None. Se e un gate logico con parametri con p parametri, la dimensione del dato di input deve essere [1,p] o [p].
    :param wires: Indice del bit su cui agisce il gate, default None.
    :param dtype: La precisione dei dati della matrice interna del gate logico puo essere impostata a pyvqnet.kcomplex64 o pyvqnet.kcomplex128, corrispondenti rispettivamente a parametri di input float o double.
    :param use_dagger: Se utilizzare la versione trasposta coniugata di questo gate, default False.
    :return: Un Module che puo essere usato per addestrare il modello.

    Example::
    
        from pyvqnet.qnn.vqc import PauliZ,QMachine
        device = QMachine(4)
        layer = PauliZ(wires=0)
        batchsize=1
        device.reset_states(1)
        layer(q_machine = device)
        print(device.states)

x1
--------

.. py:function:: pyvqnet.qnn.vqc.x1(q_machine, wires, params=None,  use_dagger=False)
    
    Applica il gate logico quantistico x1 ai vettori di stato in ``q_machine``.

    :param q_machine: dispositivo macchina virtuale quantistica.
    :param wires: indice del qubit.
    :param params: matrice dei parametri, default None. Per una funzione operativa di gate logico con p parametri, la dimensione del parametro di input deve essere [1,p] o [p].
    :param use_dagger: se utilizzare la trasposta coniugata, default False.


    Example::
        
        from pyvqnet.qnn.vqc import x1,QMachine
        qm  = QMachine(4)
        x1(q_machine=qm, wires=1)
        print(qm.states)
        # [[[[[0.7071068+0.j        0.       +0.j       ]
        #     [0.       +0.j        0.       +0.j       ]]
        # 
        #    [[0.       -0.7071068j 0.       +0.j       ]
        #     [0.       +0.j        0.       +0.j       ]]]
        # 
        # 
        #   [[[0.       +0.j        0.       +0.j       ]
        #     [0.       +0.j        0.       +0.j       ]]
        # 
        #    [[0.       +0.j        0.       +0.j       ]
        #     [0.       +0.j        0.       +0.j       ]]]]]

X1
---------------------------------------------------------------

.. py:class:: pyvqnet.qnn.vqc.X1(has_params: bool = False,trainable: bool = False,init_params=None,wires=None,dtype=pyvqnet.kcomplex64,use_dagger=False)
    
    Definisce una classe di gate logico X1.

    :param has_params: Indica se ci sono parametri. Gate come RX, RY e altri devono essere impostati a True, quelli senza parametri devono essere impostati a False, default False.
    :param trainable: Indica se contiene parametri da addestrare. Se il layer usa dati di input esterni per costruire una matrice di gate logico, impostare a False. Se contiene parametri da addestrare, e True. Default False.
    :param init_params: Parametri di inizializzazione, usati per codificare dati classici QTensor, default None. Se e un gate logico con parametri con p parametri, la dimensione del dato di input deve essere [1,p] o [p].
    :param wires: Indice del bit su cui agisce il gate, default None.
    :param dtype: La precisione dei dati della matrice interna del gate logico puo essere impostata a pyvqnet.kcomplex64 o pyvqnet.kcomplex128, corrispondenti rispettivamente a parametri di input float o double.
    :param use_dagger: Se utilizzare la versione trasposta coniugata di questo gate, default False.
    :return: Un Module che puo essere usato per addestrare il modello.

    Example::
    
        from pyvqnet.qnn.vqc import X1,QMachine
        device = QMachine(4)
        layer = X1(wires=0)
        batchsize=1
        device.reset_states(1)
        layer(q_machine = device)
        print(device.states)

y1
-----------------

.. py:function:: pyvqnet.qnn.vqc.y1(q_machine, wires, params=None,  use_dagger=False)
    
    Applica il gate logico quantistico y1 ai vettori di stato in ``q_machine``.

    :param q_machine: dispositivo macchina virtuale quantistica.
    :param wires: indice del qubit.
    :param params: matrice dei parametri, default None. Per una funzione operativa di gate logico con p parametri, la dimensione del parametro di input deve essere [1,p] o [p].
    :param use_dagger: se utilizzare la trasposta coniugata, default False.



    Example::
        
        from pyvqnet.qnn.vqc import y1,QMachine
        qm  = QMachine(4)
        y1(q_machine=qm, wires=1)
        print(qm.states)
        # [[[[[0.7071068+0.j 0.       +0.j]
        #     [0.       +0.j 0.       +0.j]]
        # 
        #    [[0.7071068+0.j 0.       +0.j]
        #     [0.       +0.j 0.       +0.j]]]
        # 
        # 
        #   [[[0.       +0.j 0.       +0.j]
        #     [0.       +0.j 0.       +0.j]]
        # 
        #    [[0.       +0.j 0.       +0.j]
        #     [0.       +0.j 0.       +0.j]]]]]

Y1
---------------------------------------------------------------

.. py:class:: pyvqnet.qnn.vqc.Y1(has_params: bool = False,trainable: bool = False,init_params=None,wires=None,dtype=pyvqnet.kcomplex64,use_dagger=False)
    
    Definisce una classe di gate logico Y1.

    :param has_params: Indica se ci sono parametri. Gate come RX, RY e altri devono essere impostati a True, quelli senza parametri devono essere impostati a False, default False.
    :param trainable: Indica se contiene parametri da addestrare. Se il layer usa dati di input esterni per costruire una matrice di gate logico, impostare a False. Se contiene parametri da addestrare, e True. Default False.
    :param init_params: Parametri di inizializzazione, usati per codificare dati classici QTensor, default None. Se e un gate logico con parametri con p parametri, la dimensione del dato di input deve essere [1,p] o [p].
    :param wires: Indice del bit su cui agisce il gate, default None.
    :param dtype: La precisione dei dati della matrice interna del gate logico puo essere impostata a pyvqnet.kcomplex64 o pyvqnet.kcomplex128, corrispondenti rispettivamente a parametri di input float o double.
    :param use_dagger: Se utilizzare la versione trasposta coniugata di questo gate, default False.
    :return: Un Module che puo essere usato per addestrare il modello.

    Example::
    
        from pyvqnet.qnn.vqc import Y1,QMachine
        device = QMachine(4)
        layer = Y1(wires=0)
        batchsize=1
        device.reset_states(1)
        layer(q_machine = device)
        print(device.states)

z1
---------------------------

.. py:function:: pyvqnet.qnn.vqc.z1(q_machine, wires, params=None,  use_dagger=False)
    
    Applica il gate logico quantistico z1 ai vettori di stato in ``q_machine``.

    :param q_machine: dispositivo macchina virtuale quantistica.
    :param wires: indice del qubit.
    :param params: matrice dei parametri, default None. Per una funzione operativa di gate logico con p parametri, la dimensione del parametro di input deve essere [1,p] o [p].
    :param use_dagger: se utilizzare la trasposta coniugata, default False.


    Example::
        
        from pyvqnet.qnn.vqc import z1,QMachine
        qm  = QMachine(4)
        z1(q_machine=qm, wires=1)
        print(qm.states)
        # [[[[[0.7071068-0.7071068j 0.       +0.j       ]
        #     [0.       +0.j        0.       +0.j       ]]
        # 
        #    [[0.       +0.j        0.       +0.j       ]
        #     [0.       +0.j        0.       +0.j       ]]]
        # 
        # 
        #   [[[0.       +0.j        0.       +0.j       ]
        #     [0.       +0.j        0.       +0.j       ]]
        # 
        #    [[0.       +0.j        0.       +0.j       ]
        #     [0.       +0.j        0.       +0.j       ]]]]]

Z1
---------------------------------------------------------------

.. py:class:: pyvqnet.qnn.vqc.Z1(has_params: bool = False,trainable: bool = False,init_params=None,wires=None,dtype=pyvqnet.kcomplex64,use_dagger=False)
    
    Definisce una classe di gate logico Z1.

    :param has_params: Indica se ci sono parametri. Gate come RX, RY e altri devono essere impostati a True, quelli senza parametri devono essere impostati a False, default False.
    :param trainable: Indica se contiene parametri da addestrare. Se il layer usa dati di input esterni per costruire una matrice di gate logico, impostare a False. Se contiene parametri da addestrare, e True. Default False.
    :param init_params: Parametri di inizializzazione, usati per codificare dati classici QTensor, default None. Se e un gate logico con parametri con p parametri, la dimensione del dato di input deve essere [1,p] o [p].
    :param wires: Indice del bit su cui agisce il gate, default None.
    :param dtype: La precisione dei dati della matrice interna del gate logico puo essere impostata a pyvqnet.kcomplex64 o pyvqnet.kcomplex128, corrispondenti rispettivamente a parametri di input float o double.
    :param use_dagger: Se utilizzare la versione trasposta coniugata di questo gate, default False.
    :return: Un Module che puo essere usato per addestrare il modello.

    Example::
    
        from pyvqnet.qnn.vqc import Z1,QMachine
        device = QMachine(4)
        layer = Z1(wires=0)
        batchsize=1
        device.reset_states(1)
        layer(q_machine = device)
        print(device.states)

rx
----

.. py:function:: pyvqnet.qnn.vqc.rx(q_machine, wires, params=None,  use_dagger=False)
    
    Applica il gate logico quantistico rx ai vettori di stato in ``q_machine``.

    :param q_machine: dispositivo macchina virtuale quantistica.
    :param wires: indice del qubit.
    :param params: matrice dei parametri, default None. Per una funzione operativa di gate logico con p parametri, la dimensione del parametro di input deve essere [1,p] o [p].
    :param use_dagger: se utilizzare la trasposta coniugata, default False.


    Example::
        
        from pyvqnet.qnn.vqc import rx,QMachine
        from pyvqnet.tensor import QTensor
        qm  = QMachine(4)
        rx(q_machine=qm, wires=1,params=QTensor([0.5]))
        print(qm.states)
        # [[[[[0.9689124+0.j       0.       +0.j      ]
        #     [0.       +0.j       0.       +0.j      ]]
        # 
        #    [[0.       -0.247404j 0.       +0.j      ]
        #     [0.       +0.j       0.       +0.j      ]]]
        # 
        # 
        #   [[[0.       +0.j       0.       +0.j      ]
        #     [0.       +0.j       0.       +0.j      ]]
        # 
        #    [[0.       +0.j       0.       +0.j      ]
        #     [0.       +0.j       0.       +0.j      ]]]]]

RX
---------------------------------------------------------------


.. py:class:: pyvqnet.qnn.vqc.RX(has_params: bool = False,trainable: bool = False,init_params=None,wires=None,dtype=pyvqnet.kcomplex64,use_dagger=False)
    
    Definisce una classe di gate logico RX.

    :param has_params: Indica se ci sono parametri. Gate come RX, RY e altri devono essere impostati a True, quelli senza parametri devono essere impostati a False, default False.
    :param trainable: Indica se contiene parametri da addestrare. Se il layer usa dati di input esterni per costruire una matrice di gate logico, impostare a False. Se contiene parametri da addestrare, e True. Default False.
    :param init_params: Parametri di inizializzazione, usati per codificare dati classici QTensor, default None. Se e un gate logico con parametri con p parametri, la dimensione del dato di input deve essere [1,p] o [p].
    :param wires: Indice del bit su cui agisce il gate, default None.
    :param dtype: La precisione dei dati della matrice interna del gate logico puo essere impostata a pyvqnet.kcomplex64 o pyvqnet.kcomplex128, corrispondenti rispettivamente a parametri di input float o double.
    :param use_dagger: Se utilizzare la versione trasposta coniugata di questo gate, default False.
    :return: Un Module che puo essere usato per addestrare il modello.

    Example::

        from pyvqnet.qnn.vqc import RX,QMachine
        device = QMachine(4)
        layer = RX(has_params= True, trainable= True, wires=0)
        batchsize = 2
        device.reset_states(batchsize)
        layer(q_machine = device)
        print(device.states)

ry
------------

.. py:function:: pyvqnet.qnn.vqc.ry(q_machine, wires, params=None,  use_dagger=False)
    
    Applica il gate logico quantistico ry ai vettori di stato in ``q_machine``.

    :param q_machine: dispositivo macchina virtuale quantistica.
    :param wires: indice del qubit.
    :param params: matrice dei parametri, default None. Per una funzione operativa di gate logico con p parametri, la dimensione del parametro di input deve essere [1,p] o [p].
    :param use_dagger: se utilizzare la trasposta coniugata, default False.


    Example::
        
        from pyvqnet.qnn.vqc import ry,QMachine
        from pyvqnet.tensor import QTensor
        qm  = QMachine(4)
        ry(q_machine=qm, wires=1,params=QTensor([0.5]))
        print(qm.states)
        # [[[[[0.9689124+0.j 0.       +0.j]
        #     [0.       +0.j 0.       +0.j]]
        # 
        #    [[0.247404 +0.j 0.       +0.j]
        #     [0.       +0.j 0.       +0.j]]]
        # 
        # 
        #   [[[0.       +0.j 0.       +0.j]
        #     [0.       +0.j 0.       +0.j]]
        # 
        #    [[0.       +0.j 0.       +0.j]
        #     [0.       +0.j 0.       +0.j]]]]]

RY
---------------------------------------------------------------


.. py:class:: pyvqnet.qnn.vqc.RY(has_params: bool = False,trainable: bool = False,init_params=None,wires=None,dtype=pyvqnet.kcomplex64,use_dagger=False)
    
    Definisce una classe di gate logico RY.

    :param has_params: Indica se ci sono parametri. Gate come RX, RY e altri devono essere impostati a True, quelli senza parametri devono essere impostati a False, default False.
    :param trainable: Indica se contiene parametri da addestrare. Se il layer usa dati di input esterni per costruire una matrice di gate logico, impostare a False. Se contiene parametri da addestrare, e True. Default False.
    :param init_params: Parametri di inizializzazione, usati per codificare dati classici QTensor, default None. Se e un gate logico con parametri con p parametri, la dimensione del dato di input deve essere [1,p] o [p].
    :param wires: Indice del bit su cui agisce il gate, default None.
    :param dtype: La precisione dei dati della matrice interna del gate logico puo essere impostata a pyvqnet.kcomplex64 o pyvqnet.kcomplex128, corrispondenti rispettivamente a parametri di input float o double.
    :param use_dagger: Se utilizzare la versione trasposta coniugata di questo gate, default False.
    :return: Un Module che puo essere usato per addestrare il modello.

    Example::

        from pyvqnet.qnn.vqc import RY,QMachine
        device = QMachine(4)
        layer = RY(has_params= True, trainable= True, wires=0)
        batchsize = 2
        device.reset_states(batchsize)
        layer(q_machine = device)
        print(device.states)

rz
-----

.. py:function:: pyvqnet.qnn.vqc.rz(q_machine, wires, params=None,  use_dagger=False)
    
    Applica il gate logico quantistico rz ai vettori di stato in ``q_machine``.

    :param q_machine: dispositivo macchina virtuale quantistica.
    :param wires: indice del qubit.
    :param params: matrice dei parametri, default None. Per una funzione operativa di gate logico con p parametri, la dimensione del parametro di input deve essere [1,p] o [p].
    :param use_dagger: se utilizzare la trasposta coniugata, default False.


    Example::
        
        from pyvqnet.qnn.vqc import rz,QMachine
        from pyvqnet.tensor import QTensor
        qm  = QMachine(4)
        rz(q_machine=qm, wires=1,params=QTensor([0.5]))
        print(qm.states)
        # [[[[[0.9689124-0.247404j 0.       +0.j      ]
        #     [0.       +0.j       0.       +0.j      ]]
        # 
        #    [[0.       +0.j       0.       +0.j      ]
        #     [0.       +0.j       0.       +0.j      ]]]
        # 
        # 
        #   [[[0.       +0.j       0.       +0.j      ]
        #     [0.       +0.j       0.       +0.j      ]]
        # 
        #    [[0.       +0.j       0.       +0.j      ]
        #     [0.       +0.j       0.       +0.j      ]]]]]


RZ
---------------------------------------------------------------


.. py:class:: pyvqnet.qnn.vqc.RZ(has_params: bool = False,trainable: bool = False,init_params=None,wires=None,dtype=pyvqnet.kcomplex64,use_dagger=False)
    
    Definisce una classe di gate logico RZ.

    :param has_params: Indica se ci sono parametri. Gate come RX, RY e altri devono essere impostati a True, quelli senza parametri devono essere impostati a False, default False.
    :param trainable: Indica se contiene parametri da addestrare. Se il layer usa dati di input esterni per costruire una matrice di gate logico, impostare a False. Se contiene parametri da addestrare, e True. Default False.
    :param init_params: Parametri di inizializzazione, usati per codificare dati classici QTensor, default None. Se e un gate logico con parametri con p parametri, la dimensione del dato di input deve essere [1,p] o [p].
    :param wires: Indice del bit su cui agisce il gate, default None.
    :param dtype: La precisione dei dati della matrice interna del gate logico puo essere impostata a pyvqnet.kcomplex64 o pyvqnet.kcomplex128, corrispondenti rispettivamente a parametri di input float o double.
    :param use_dagger: Se utilizzare la versione trasposta coniugata di questo gate, default False.
    :return: Un Module che puo essere usato per addestrare il modello.

    Example::

        from pyvqnet.qnn.vqc import RZ,QMachine
        device = QMachine(4)
        layer = RZ(has_params= True, trainable= True, wires=0)
        batchsize = 2
        device.reset_states(batchsize)
        layer(q_machine = device)
        print(device.states)

crx
-------------

.. py:function:: pyvqnet.qnn.vqc.crx(q_machine, wires, params=None,  use_dagger=False)
    
    Applica il gate logico quantistico crx ai vettori di stato in ``q_machine``.

    :param q_machine: dispositivo macchina virtuale quantistica.
    :param wires: indice del qubit.
    :param params: matrice dei parametri, default None. Per una funzione operativa di gate logico con p parametri, la dimensione del parametro di input deve essere [1,p] o [p].
    :param use_dagger: se utilizzare la trasposta coniugata, default False.



    Example::

        from pyvqnet.qnn.vqc import QMachine
        import pyvqnet.qnn.vqc as vqc
        from pyvqnet.tensor import QTensor
        qm = QMachine(4)
        vqc.crx(q_machine=qm,wires=[0,2], params=QTensor([0.5]))
        print(qm.states)

        # [[[[[1.+0.j 0.+0.j]
        #     [0.+0.j 0.+0.j]]
        # 
        #    [[0.+0.j 0.+0.j]
        #     [0.+0.j 0.+0.j]]]
        # 
        # 
        #   [[[0.+0.j 0.+0.j]
        #     [0.+0.j 0.+0.j]]
        # 
        #    [[0.+0.j 0.+0.j]
        #     [0.+0.j 0.+0.j]]]]]


CRX
---------------------------------------------------------------


.. py:class:: pyvqnet.qnn.vqc.CRX(has_params: bool = False,trainable: bool = False,init_params=None,wires=None,dtype=pyvqnet.kcomplex64,use_dagger=False)
    
     Definisce una classe di gate logico CRX.

    :param has_params: Indica se ci sono parametri. Gate come RX, RY e altri devono essere impostati a True, quelli senza parametri devono essere impostati a False, default False.
    :param trainable: Indica se contiene parametri da addestrare. Se il layer usa dati di input esterni per costruire una matrice di gate logico, impostare a False. Se contiene parametri da addestrare, e True. Default False.
    :param init_params: Parametri di inizializzazione, usati per codificare dati classici QTensor, default None. Se e un gate logico con parametri con p parametri, la dimensione del dato di input deve essere [1,p] o [p].
    :param wires: Indice del bit su cui agisce il gate, default None.
    :param dtype: La precisione dei dati della matrice interna del gate logico puo essere impostata a pyvqnet.kcomplex64 o pyvqnet.kcomplex128, corrispondenti rispettivamente a parametri di input float o double.
    :param use_dagger: Se utilizzare la versione trasposta coniugata di questo gate, default False.
    :return: Un Module che puo essere usato per addestrare il modello.

    Example::

        from pyvqnet.qnn.vqc import CRX,QMachine
        device = QMachine(4)
        layer = CRX(has_params= True, trainable= True, wires=[0,2])
        batchsize = 2
        device.reset_states(batchsize)
        layer(q_machine = device)
        print(device.states)

cry
-----------------

.. py:function:: pyvqnet.qnn.vqc.cry(q_machine, wires, params=None,  use_dagger=False)
    
    Applica il gate logico quantistico cry ai vettori di stato in ``q_machine``.

    :param q_machine: dispositivo macchina virtuale quantistica.
    :param wires: indice del qubit.
    :param params: matrice dei parametri, default None. Per una funzione operativa di gate logico con p parametri, la dimensione del parametro di input deve essere [1,p] o [p].
    :param use_dagger: se utilizzare la trasposta coniugata, default False.


    Example::

        from pyvqnet.qnn.vqc import QMachine
        import pyvqnet.qnn.vqc as vqc
        from pyvqnet.tensor import QTensor
        qm = QMachine(4)
        vqc.cry(q_machine=qm,wires=[0,2], params=QTensor([0.5]))
        print(qm.states)

        # [[[[[1.+0.j 0.+0.j]
        #     [0.+0.j 0.+0.j]]
        # 
        #    [[0.+0.j 0.+0.j]
        #     [0.+0.j 0.+0.j]]]
        # 
        # 
        #   [[[0.+0.j 0.+0.j]
        #     [0.+0.j 0.+0.j]]
        # 
        #    [[0.+0.j 0.+0.j]
        #     [0.+0.j 0.+0.j]]]]]

CRY
---------------------------------------------------------------


.. py:class:: pyvqnet.qnn.vqc.CRY(has_params: bool = False,trainable: bool = False,init_params=None,wires=None,dtype=pyvqnet.kcomplex64,use_dagger=False)
    
     Definisce una classe di gate logico CRY.

    :param has_params: Indica se ci sono parametri. Gate come RX, RY e altri devono essere impostati a True, quelli senza parametri devono essere impostati a False, default False.
    :param trainable: Indica se contiene parametri da addestrare. Se il layer usa dati di input esterni per costruire una matrice di gate logico, impostare a False. Se contiene parametri da addestrare, e True. Default False.
    :param init_params: Parametri di inizializzazione, usati per codificare dati classici QTensor, default None. Se e un gate logico con parametri con p parametri, la dimensione del dato di input deve essere [1,p] o [p].
    :param wires: Indice del bit su cui agisce il gate, default None.
    :param dtype: La precisione dei dati della matrice interna del gate logico puo essere impostata a pyvqnet.kcomplex64 o pyvqnet.kcomplex128, corrispondenti rispettivamente a parametri di input float o double.
    :param use_dagger: Se utilizzare la versione trasposta coniugata di questo gate, default False.
    :return: Un Module che puo essere usato per addestrare il modello.

    Example::

        from pyvqnet.qnn.vqc import CRY,QMachine
        device = QMachine(4)
        layer = CRY(has_params= True, trainable= True, wires=[0,2])
        batchsize = 2
        device.reset_states(batchsize)
        layer(q_machine = device)
        print(device.states)

crz
------------

.. py:function:: pyvqnet.qnn.vqc.crz(q_machine, wires, params=None,  use_dagger=False)
    
    Applica il gate logico quantistico crz ai vettori di stato in ``q_machine``.

    :param q_machine: dispositivo macchina virtuale quantistica.
    :param wires: indice del qubit.
    :param params: matrice dei parametri, default None. Per una funzione operativa di gate logico con p parametri, la dimensione del parametro di input deve essere [1,p] o [p].
    :param use_dagger: se utilizzare la trasposta coniugata, default False.


    Example::

        from pyvqnet.qnn.vqc import QMachine
        import pyvqnet.qnn.vqc as vqc
        from pyvqnet.tensor import QTensor
        qm = QMachine(4)
        vqc.crz(q_machine=qm,wires=[0,2], params=QTensor([0.5]))
        print(qm.states)
        
        # [[[[[1.+0.j 0.+0.j]
        #     [0.+0.j 0.+0.j]]
        # 
        #    [[0.+0.j 0.+0.j]
        #     [0.+0.j 0.+0.j]]]
        # 
        # 
        #   [[[0.+0.j 0.+0.j]
        #     [0.+0.j 0.+0.j]]
        # 
        #    [[0.+0.j 0.+0.j]
        #     [0.+0.j 0.+0.j]]]]]

CRZ
---------------------------------------------------------------


.. py:class:: pyvqnet.qnn.vqc.CRZ(has_params: bool = False,trainable: bool = False,init_params=None,wires=None,dtype=pyvqnet.kcomplex64,use_dagger=False)
    
    Definisce una classe di gate logico CRZ.

    :param has_params: Indica se ci sono parametri. Gate come RX, RY e altri devono essere impostati a True, quelli senza parametri devono essere impostati a False, default False.
    :param trainable: Indica se contiene parametri da addestrare. Se il layer usa dati di input esterni per costruire una matrice di gate logico, impostare a False. Se contiene parametri da addestrare, e True. Default False.
    :param init_params: Parametri di inizializzazione, usati per codificare dati classici QTensor, default None. Se e un gate logico con parametri con p parametri, la dimensione del dato di input deve essere [1,p] o [p].
    :param wires: Indice del bit su cui agisce il gate, default None.
    :param dtype: La precisione dei dati della matrice interna del gate logico puo essere impostata a pyvqnet.kcomplex64 o pyvqnet.kcomplex128, corrispondenti rispettivamente a parametri di input float o double.
    :param use_dagger: Se utilizzare la versione trasposta coniugata di questo gate, default False.
    :return: Un Module che puo essere usato per addestrare il modello.

    Example::

        from pyvqnet.qnn.vqc import CRZ,QMachine
        device = QMachine(4)
        layer = CRZ(has_params= True, trainable= True, wires=[0,2])
        batchsize = 2
        device.reset_states(batchsize)
        layer(q_machine = device)
        print(device.states)
 

u1
-------------------------------

.. py:function:: pyvqnet.qnn.vqc.u1(q_machine, wires, params=None,  use_dagger=False)
    
    Applica il gate logico quantistico u1 ai vettori di stato in ``q_machine``.

    :param q_machine: dispositivo macchina virtuale quantistica.
    :param wires: indice del qubit.
    :param params: matrice dei parametri, default None. Per una funzione operativa di gate logico con p parametri, la dimensione del parametro di input deve essere [1,p] o [p].
    :param use_dagger: se utilizzare la trasposta coniugata, default False.


    Example::
        
        from pyvqnet.qnn.vqc import u1,QMachine
        from pyvqnet.tensor import QTensor
        qm  = QMachine(4)
        u1(q_machine=qm, wires=1,params=QTensor([24.0]))
        print(qm.states)
        # [[[[1.+0.j 0.+0.j]
        #     [0.+0.j 0.+0.j]]
        # 
        #    [[0.+0.j 0.+0.j]
        #     [0.+0.j 0.+0.j]]]
        # 
        # 
        #   [[[0.+0.j 0.+0.j]
        #     [0.+0.j 0.+0.j]]
        # 
        #    [[0.+0.j 0.+0.j]
        #     [0.+0.j 0.+0.j]]]]]

U1
--------------------------------------


.. py:class:: pyvqnet.qnn.vqc.U1(has_params: bool = False,trainable: bool = False,init_params=None,wires=None,dtype=pyvqnet.kcomplex64,use_dagger=False)
    
    Definisce una classe di gate logico U1.

    :param has_params: Indica se ci sono parametri. Gate come RX, RY e altri devono essere impostati a True, quelli senza parametri devono essere impostati a False, default False.
    :param trainable: Indica se contiene parametri da addestrare. Se il layer usa dati di input esterni per costruire una matrice di gate logico, impostare a False. Se contiene parametri da addestrare, e True. Default False.
    :param init_params: Parametri di inizializzazione, usati per codificare dati classici QTensor, default None. Se e un gate logico con parametri con p parametri, la dimensione del dato di input deve essere [1,p] o [p].
    :param wires: Indice del bit su cui agisce il gate, default None.
    :param dtype: La precisione dei dati della matrice interna del gate logico puo essere impostata a pyvqnet.kcomplex64 o pyvqnet.kcomplex128, corrispondenti rispettivamente a parametri di input float o double.
    :param use_dagger: Se utilizzare la versione trasposta coniugata di questo gate, default False.
    :return: Un Module che puo essere usato per addestrare il modello.

    Example::

        from pyvqnet.qnn.vqc import U1,QMachine
        device = QMachine(4)
        layer = U1(has_params= True, trainable= True, wires=0)
        batchsize = 2
        device.reset_states(batchsize)
        layer(q_machine = device)
        print(device.states)

u2
------------------

.. py:function:: pyvqnet.qnn.vqc.u2(q_machine, wires, params=None,  use_dagger=False)
    
    Applica il gate logico quantistico u2 ai vettori di stato in ``q_machine``.

    :param q_machine: dispositivo macchina virtuale quantistica.
    :param wires: indice del qubit.
    :param params: matrice dei parametri, default None. Per una funzione operativa di gate logico con p parametri, la dimensione del parametro di input deve essere [1,p] o [p].
    :param use_dagger: se utilizzare la trasposta coniugata, default False.


    Example::
        
        from pyvqnet.qnn.vqc import u2,QMachine
        from pyvqnet.tensor import QTensor
        qm  = QMachine(4)
        u2(q_machine=qm, wires=1,params=QTensor([[24.0,-3]]))
        print(qm.states)
        # [[[[[0.7071068+0.j        0.       +0.j       ]
        #     [0.       +0.j        0.       +0.j       ]]
        # 
        #    [[0.2999398-0.6403406j 0.       +0.j       ]
        #     [0.       +0.j        0.       +0.j       ]]]
        # 
        # 
        #   [[[0.       +0.j        0.       +0.j       ]
        #     [0.       +0.j        0.       +0.j       ]]
        # 
        #    [[0.       +0.j        0.       +0.j       ]
        #     [0.       +0.j        0.       +0.j       ]]]]]

U2
-----------------------------------------------


.. py:class:: pyvqnet.qnn.vqc.U2(has_params: bool = False,trainable: bool = False,init_params=None,wires=None,dtype=pyvqnet.kcomplex64,use_dagger=False)
    
    Definisce una classe di gate logico U2.

    :param has_params: Indica se ci sono parametri. Gate come RX, RY e altri devono essere impostati a True, quelli senza parametri devono essere impostati a False, default False.
    :param trainable: Indica se contiene parametri da addestrare. Se il layer usa dati di input esterni per costruire una matrice di gate logico, impostare a False. Se contiene parametri da addestrare, e True. Default False.
    :param init_params: Parametri di inizializzazione, usati per codificare dati classici QTensor, default None. Se e un gate logico con parametri con p parametri, la dimensione del dato di input deve essere [1,p] o [p].
    :param wires: Indice del bit su cui agisce il gate, default None.
    :param dtype: La precisione dei dati della matrice interna del gate logico puo essere impostata a pyvqnet.kcomplex64 o pyvqnet.kcomplex128, corrispondenti rispettivamente a parametri di input float o double.
    :param use_dagger: Se utilizzare la versione trasposta coniugata di questo gate, default False.
    :return: Un Module che puo essere usato per addestrare il modello.

    Example::

        from pyvqnet.qnn.vqc import U2,QMachine
        device = QMachine(4)
        layer = U2(has_params= True, trainable= True, wires=0)
        batchsize = 2
        device.reset_states(batchsize)
        layer(q_machine = device)
        print(device.states)

u3
------

.. py:function:: pyvqnet.qnn.vqc.u3(q_machine, wires, params=None,  use_dagger=False)
    
    Applica il gate logico quantistico u3 ai vettori di stato in ``q_machine``.

    :param q_machine: dispositivo macchina virtuale quantistica.
    :param wires: indice del qubit.
    :param params: matrice dei parametri, default None. Per una funzione operativa di gate logico con p parametri, la dimensione del parametro di input deve essere [1,p] o [p].
    :param use_dagger: se utilizzare la trasposta coniugata, default False.


    Example::
        
        from pyvqnet.qnn.vqc import u3,QMachine
        from pyvqnet.tensor import QTensor
        qm  = QMachine(4)
        u3(q_machine=qm, wires=1,params=QTensor([[24.0,-3,1]]))
        print(qm.states)
        # [[[[[0.843854 +0.j        0.       +0.j       ]
        #     [0.       +0.j        0.       +0.j       ]]
        # 
        #    [[0.5312032+0.0757212j 0.       +0.j       ]
        #     [0.       +0.j        0.       +0.j       ]]]
        # 
        # 
        #   [[[0.       +0.j        0.       +0.j       ]
        #     [0.       +0.j        0.       +0.j       ]]
        # 
        #    [[0.       +0.j        0.       +0.j       ]
        #     [0.       +0.j        0.       +0.j       ]]]]]

U3
-----------------------------------------


.. py:class:: pyvqnet.qnn.vqc.U3(has_params: bool = False,trainable: bool = False,init_params=None,wires=None,dtype=pyvqnet.kcomplex64,use_dagger=False)
    
    Definisce una classe di gate logico U3.

    :param has_params: Indica se ci sono parametri. Gate come RX, RY e altri devono essere impostati a True, quelli senza parametri devono essere impostati a False, default False.
    :param trainable: Indica se contiene parametri da addestrare. Se il layer usa dati di input esterni per costruire una matrice di gate logico, impostare a False. Se contiene parametri da addestrare, e True. Default False.
    :param init_params: Parametri di inizializzazione, usati per codificare dati classici QTensor, default None. Se e un gate logico con parametri con p parametri, la dimensione del dato di input deve essere [1,p] o [p].
    :param wires: Indice del bit su cui agisce il gate, default None.
    :param dtype: La precisione dei dati della matrice interna del gate logico puo essere impostata a pyvqnet.kcomplex64 o pyvqnet.kcomplex128, corrispondenti rispettivamente a parametri di input float o double.
    :param use_dagger: Se utilizzare la versione trasposta coniugata di questo gate, default False.
    :return: Un Module che puo essere usato per addestrare il modello.

    Example::

        from pyvqnet.qnn.vqc import U3,QMachine
        device = QMachine(4)
        layer = U3(has_params= True, trainable= True, wires=0)
        batchsize = 2
        device.reset_states(batchsize)
        layer(q_machine = device)
        print(device.states)

cy
---------------------------------------------------------------

.. py:function:: pyvqnet.qnn.vqc.cy(q_machine, wires, params=None, use_dagger=False)

    Applica il gate logico quantistico cy ai vettori di stato in ``q_machine``.

    :param q_machine: Dispositivo macchina virtuale quantistica.
    :param wires: Indice del qubit.
    :param params: Matrice dei parametri, default None.
    :param use_dagger: Se utilizzare la trasposta coniugata, default False.

    Example::

        from pyvqnet.qnn.vqc import cy,QMachine
        qm = QMachine(4)
        cy(q_machine=qm,wires=(1,0))
        print(qm.states)
        # [[[[[1.+0.j,0.+0.j],
        #     [0.+0.j,0.+0.j]],

        #    [[0.+0.j,0.+0.j],
        #     [0.+0.j,0.+0.j]]],


        #   [[[0.+0.j,0.+0.j],
        #     [0.+0.j,0.+0.j]],

        #    [[0.+0.j,0.+0.j],
        #     [0.+0.j,0.+0.j]]]]]


CY
---------------------------------------------------------------

.. py:class:: pyvqnet.qnn.vqc.CY(has_params: bool = False,trainable: bool = False,init_params=None,wires=None,dtype=pyvqnet.kcomplex64,use_dagger=False)

    Definisce una classe di gate logico CY.

    :param has_params: Indica se ci sono parametri. Gate come RX, RY e altri devono essere impostati a True, quelli senza parametri devono essere impostati a False, default False.
    :param trainable: Indica se include parametri da addestrare. Se questo layer usa dati di input esterni per costruire una matrice di gate logico, impostare a False. Se i parametri da addestrare devono essere inizializzati da questo layer, sara True. Default False.
    :param init_params: Parametri di inizializzazione, usati per codificare dati classici QTensor, default None. Se e un gate logico con parametri con p parametri, la dimensione del dato di input deve essere [1,p] o [p].
    :param wires: Indice del bit su cui agisce il gate, default None.
    :param dtype: La precisione dei dati della matrice interna del gate logico puo essere impostata a pyvqnet.kcomplex64 o pyvqnet.kcomplex128, corrispondenti rispettivamente a parametri di input float o double.
    :param use_dagger: Se utilizzare la versione trasposta coniugata di questo gate, default False.
    :return: Un Module che puo essere usato per addestrare il modello.

    Example::

            from pyvqnet.qnn.vqc import CY,QMachine
            device = QMachine(4)
            layer = CY(wires=[0,1])
            batchsize = 2
            device.reset_states(batchsize)
            layer(q_machine = device)
            print(device.states)


cnot
-------------------

.. py:function:: pyvqnet.qnn.vqc.cnot(q_machine, wires, params=None,  use_dagger=False)
    
    Applica il gate logico quantistico cnot ai vettori di stato in ``q_machine``.

    :param q_machine: dispositivo macchina virtuale quantistica.
    :param wires: indice del qubit.
    :param params: matrice dei parametri, default None. Per una funzione operativa di gate logico con p parametri, la dimensione del parametro di input deve essere [1,p] o [p].
    :param use_dagger: se utilizzare la trasposta coniugata, default False.


    Example::
        
        from pyvqnet.qnn.vqc import cnot,QMachine
        qm  = QMachine(4)
        cnot(q_machine=qm,wires=[1,0])
        print(qm.states)
        # [[[[[1.+0.j 0.+0.j]
        #     [0.+0.j 0.+0.j]]
        # 
        #    [[0.+0.j 0.+0.j]
        #     [0.+0.j 0.+0.j]]]
        # 
        # 
        #   [[[0.+0.j 0.+0.j]
        #     [0.+0.j 0.+0.j]]
        # 
        #    [[0.+0.j 0.+0.j]
        #     [0.+0.j 0.+0.j]]]]]


CNOT
-------------------------------------------


.. py:class:: pyvqnet.qnn.vqc.CNOT(has_params: bool = False,trainable: bool = False,init_params=None,wires=None,dtype=pyvqnet.kcomplex64,use_dagger=False)
    
    Definisce una classe di gate logico CNOT.

    :param has_params: Indica se ci sono parametri. Gate come RX, RY e altri devono essere impostati a True, quelli senza parametri devono essere impostati a False, default False.
    :param trainable: Indica se contiene parametri da addestrare. Se il layer usa dati di input esterni per costruire una matrice di gate logico, impostare a False. Se contiene parametri da addestrare, e True. Default False.
    :param init_params: Parametri di inizializzazione, usati per codificare dati classici QTensor, default None. Se e un gate logico con parametri con p parametri, la dimensione del dato di input deve essere [1,p] o [p].
    :param wires: Indice del bit su cui agisce il gate, default None.
    :param dtype: La precisione dei dati della matrice interna del gate logico puo essere impostata a pyvqnet.kcomplex64 o pyvqnet.kcomplex128, corrispondenti rispettivamente a parametri di input float o double.
    :param use_dagger: Se utilizzare la versione trasposta coniugata di questo gate, default False.
    :return: Un Module che puo essere usato per addestrare il modello.

    Example::

        from pyvqnet.qnn.vqc import CNOT,QMachine
        device = QMachine(4)
        layer = CNOT(wires=[0,1])
        batchsize = 2
        device.reset_states(batchsize)
        layer(q_machine = device)
        print(device.states)

cr
-------------------

.. py:function:: pyvqnet.qnn.vqc.cr(q_machine, wires, params=None,  use_dagger=False)
    
    Applica il gate logico quantistico cr ai vettori di stato in ``q_machine``.

    :param q_machine: dispositivo macchina virtuale quantistica.
    :param wires: indice del qubit.
    :param params: matrice dei parametri, default None. Per una funzione operativa di gate logico con p parametri, la dimensione del parametro di input deve essere [1,p] o [p].
    :param use_dagger: se utilizzare la trasposta coniugata, default False.


    Example::
        
        from pyvqnet.qnn.vqc import cr,QMachine
        from pyvqnet.tensor import QTensor
        qm  = QMachine(4)
        cr(q_machine=qm,wires=[1,0],params=QTensor([0.5]))
        print(qm.states)
        # [[[[[1.+0.j 0.+0.j]
        #     [0.+0.j 0.+0.j]]
        # 
        #    [[0.+0.j 0.+0.j]
        #     [0.+0.j 0.+0.j]]]
        # 
        # 
        #   [[[0.+0.j 0.+0.j]
        #     [0.+0.j 0.+0.j]]
        # 
        #    [[0.+0.j 0.+0.j]
        #     [0.+0.j 0.+0.j]]]]]

CR
---------------------------------------------------------------


.. py:class:: pyvqnet.qnn.vqc.CR(has_params: bool = False,trainable: bool = False,init_params=None,wires=None,dtype=pyvqnet.kcomplex64,use_dagger=False)
    
    Definisce un gate logico CR.

    :param has_params: Indica se ci sono parametri. Gate come RX, RY e altri devono essere impostati a True, quelli senza parametri devono essere impostati a False, default False.
    :param trainable: Indica se contiene parametri da addestrare. Se il layer usa dati di input esterni per costruire una matrice di gate logico, impostare a False. Se contiene parametri da addestrare, e True. Default False.
    :param init_params: Parametri di inizializzazione, usati per codificare dati classici QTensor, default None. Se e un gate logico con parametri con p parametri, la dimensione del dato di input deve essere [1,p] o [p].
    :param wires: Indice del bit su cui agisce il gate, default None.
    :param dtype: La precisione dei dati della matrice interna del gate logico puo essere impostata a pyvqnet.kcomplex64 o pyvqnet.kcomplex128, corrispondenti rispettivamente a parametri di input float o double.
    :param use_dagger: Se utilizzare la versione trasposta coniugata di questo gate, default False.
    :return: Un Module che puo essere usato per addestrare il modello.

    Example::

        from pyvqnet.qnn.vqc import CR,QMachine
        device = QMachine(4)
        layer = CR(has_params= True, trainable= True, wires=[0,2])
        batchsize = 2
        device.reset_states(batchsize)
        layer(q_machine = device)
        print(device.states)

iswap
---------------

.. py:function:: pyvqnet.qnn.vqc.iswap(q_machine, wires, params=None,  use_dagger=False)
    
    Applica il gate logico quantistico iswap ai vettori di stato in ``q_machine``.

    :param q_machine: dispositivo macchina virtuale quantistica.
    :param wires: indice del qubit.
    :param params: matrice dei parametri, default None. Per una funzione operativa di gate logico con p parametri, la dimensione del parametro di input deve essere [1,p] o [p].
    :param use_dagger: se utilizzare la trasposta coniugata, default False.


    Example::
        
        from pyvqnet.qnn.vqc import iswap,QMachine
        from pyvqnet.tensor import QTensor
        qm  = QMachine(4)
        iswap(q_machine=qm,wires=[1,0] )
        print(qm.states)
        # [[[[[1.+0.j 0.+0.j]
        #     [0.+0.j 0.+0.j]]
        # 
        #    [[0.+0.j 0.+0.j]
        #     [0.+0.j 0.+0.j]]]
        # 
        # 
        #   [[[0.+0.j 0.+0.j]
        #     [0.+0.j 0.+0.j]]
        # 
        #    [[0.+0.j 0.+0.j]
        #     [0.+0.j 0.+0.j]]]]]


swap
-------------------

.. py:function:: pyvqnet.qnn.vqc.swap(q_machine, wires, params=None,  use_dagger=False)
    
    Applica il gate logico quantistico swap ai vettori di stato in ``q_machine``.

    :param q_machine: dispositivo macchina virtuale quantistica.
    :param wires: indice del qubit.
    :param params: matrice dei parametri, default None. Per una funzione operativa di gate logico con p parametri, la dimensione del parametro di input deve essere [1,p] o [p].
    :param use_dagger: se utilizzare la trasposta coniugata, default False.


    Example::
        
        from pyvqnet.qnn.vqc import swap,QMachine
        qm  = QMachine(4)
        swap(q_machine=qm,wires=[1,0])
        print(qm.states)
        # [[[[[1.+0.j 0.+0.j]
        #     [0.+0.j 0.+0.j]]
        # 
        #    [[0.+0.j 0.+0.j]
        #     [0.+0.j 0.+0.j]]]
        # 
        # 
        #   [[[0.+0.j 0.+0.j]
        #     [0.+0.j 0.+0.j]]
        # 
        #    [[0.+0.j 0.+0.j]
        #     [0.+0.j 0.+0.j]]]]]

SWAP
----------------------------------------


.. py:class:: pyvqnet.qnn.vqc.SWAP(has_params: bool = False,trainable: bool = False,init_params=None,wires=None,dtype=pyvqnet.kcomplex64,use_dagger=False)
    
    Define a SWAP logic gate class.

    :param has_params: Indica se ci sono parametri. Gate come RX, RY e altri devono essere impostati a True, quelli senza parametri devono essere impostati a False, default False.
    :param trainable: Indica se contiene parametri da addestrare. Se il layer usa dati di input esterni per costruire una matrice di gate logico, impostare a False. Se contiene parametri da addestrare, e True. Default False.
    :param init_params: Parametri di inizializzazione, usati per codificare dati classici QTensor, default None. Se e un gate logico con parametri con p parametri, la dimensione del dato di input deve essere [1,p] o [p].
    :param wires: Indice del bit su cui agisce il gate, default None.
    :param dtype: La precisione dei dati della matrice interna del gate logico puo essere impostata a pyvqnet.kcomplex64 o pyvqnet.kcomplex128, corrispondenti rispettivamente a parametri di input float o double.
    :param use_dagger: Se utilizzare la versione trasposta coniugata di questo gate, default False.
    :return: Un Module che puo essere usato per addestrare il modello.

    Example::

        from pyvqnet.qnn.vqc import SWAP,QMachine
        device = QMachine(4)
        layer = SWAP(wires=[0,1])
        batchsize = 2
        device.reset_states(batchsize)
        layer(q_machine = device)
        print(device.states)


cswap
---------------------------------------------------------------

.. py:function:: pyvqnet.qnn.vqc.cswap(q_machine, wires, params=None, use_dagger=False)

    Applica il gate logico quantistico cswap ai vettori di stato in ``q_machine``.

    :param q_machine: Dispositivo macchina virtuale quantistica.
    :param wires: Indice del qubit.
    :param params: Matrice dei parametri, default None.
    :param use_dagger: Se utilizzare la trasposta coniugata, default False.
    :return: Output QTensor.

    Example::

        from pyvqnet.qnn.vqc import cswap,QMachine
        qm = QMachine(4)
        cswap(q_machine=qm,wires=[1,0,3],)
        print(qm.states)
        # [[[[[1.+0.j,0.+0.j],
        # [0.+0.j,0.+0.j]],

        # [[0.+0.j,0.+0.j],
        # [0.+0.j,0.+0.j]]],


        # [[[0.+0.j,0.+0.j],
        # [0.+0.j,0.+0.j]],

        # [[0.+0.j,0.+0.j],
        # [0.+0.j,0.+0.j]]]]]


CSWAP
-------------------------------------------------

.. py:class:: pyvqnet.qnn.vqc.CSWAP(has_params: bool = False, trainable: bool = False, init_params=None, wires=None, dtype=pyvqnet.kcomplex64, use_dagger=False)
    
    Definisce una classe di gate logico CSWAP.

    .. math:: CSWAP = \begin{bmatrix}
            1 & 0 & 0 & 0 & 0 & 0 & 0 & 0 \\
            0 & 1 & 0 & 0 & 0 & 0 & 0 & 0 \\
            0 & 0 & 1 & 0 & 0 & 0 & 0 & 0 \\
            0 & 0 & 0 & 1 & 0 & 0 & 0 & 0 \\
            0 & 0 & 0 & 0 & 1 & 0 & 0 & 0 \\
            0 & 0 & 0 & 0 & 0 & 0 & 1 & 0 \\
            0 & 0 & 0 & 0 & 0 & 1 & 0 & 0 \\
            0 & 0 & 0 & 0 & 0 & 0 & 0 & 1
        \end{bmatrix}.

    :param has_params: Indica se ci sono parametri. Gate come RX, RY e altri devono essere impostati a True, quelli senza parametri devono essere impostati a False, default False.
    :param trainable: Indica se include parametri da addestrare. Se questo layer usa dati di input esterni per costruire una matrice di gate logico, impostare a False. Se i parametri da addestrare devono essere inizializzati da questo layer, sara True. Default False.
    :param init_params: Parametri di inizializzazione, usati per codificare dati classici QTensor, default None. Se e un gate logico con parametri con p parametri, la dimensione del dato di input deve essere [1,p] o [p].
    :param wires: Indice del bit su cui agisce il gate, default None.
    :param dtype: La precisione dei dati della matrice interna del gate logico puo essere impostata a pyvqnet.kcomplex64 o pyvqnet.kcomplex128, corrispondenti rispettivamente a parametri di input float o double.
    :param use_dagger: Se utilizzare la versione trasposta coniugata di questo gate, default False.
    :return: Un Module che puo essere usato per addestrare il modello.

    Example::

        from pyvqnet.qnn.vqc import CSWAP,QMachine
        device = QMachine(4)
        layer = CSWAP(wires=[0,1,2])
        batchsize = 2
        device.reset_states(batchsize)
        layer(q_machine = device)
        print(device.states)

        # [[[[[1.+0.j,0.+0.j],
        #     [0.+0.j,0.+0.j]],

        #    [[0.+0.j,0.+0.j],
        #     [0.+0.j,0.+0.j]]],


        #   [[[0.+0.j,0.+0.j],
        #     [0.+0.j,0.+0.j]],

        #    [[0.+0.j,0.+0.j],
        #     [0.+0.j,0.+0.j]]]],



        #  [[[[1.+0.j,0.+0.j],
        #     [0.+0.j,0.+0.j]],

        #    [[0.+0.j,0.+0.j],
        #     [0.+0.j,0.+0.j]]],


        #   [[[0.+0.j,0.+0.j],
        #     [0.+0.j,0.+0.j]],

        #    [[0.+0.j,0.+0.j],
        #     [0.+0.j,0.+0.j]]]]]


cz
-----------

.. py:function:: pyvqnet.qnn.vqc.cz(q_machine, wires, params=None,  use_dagger=False)
    
    Applica il gate logico quantistico cz ai vettori di stato in ``q_machine``.

    :param q_machine: dispositivo macchina virtuale quantistica.
    :param wires: indice del qubit.
    :param params: matrice dei parametri, default None. Per una funzione operativa di gate logico con p parametri, la dimensione del parametro di input deve essere [1,p] o [p].
    :param use_dagger: se utilizzare la trasposta coniugata, default False.


    Example::
        
        from pyvqnet.qnn.vqc import cz,QMachine
        qm  = QMachine(4)
        cz(q_machine=qm,wires=[1,0])
        print(qm.states)
        # [[[[[1.+0.j 0.+0.j]
        #     [0.+0.j 0.+0.j]]
        # 
        #    [[0.+0.j 0.+0.j]
        #     [0.+0.j 0.+0.j]]]
        # 
        # 
        #   [[[0.+0.j 0.+0.j]
        #     [0.+0.j 0.+0.j]]
        # 
        #    [[0.+0.j 0.+0.j]
        #     [0.+0.j 0.+0.j]]]]]

CZ
--------------------------------------------

.. py:class:: pyvqnet.qnn.vqc.CZ(has_params: bool = False,trainable: bool = False,init_params=None,wires=None,dtype=pyvqnet.kcomplex64,use_dagger=False)
    
    Definisce una classe di gate logico CZ.

    :param has_params: Indica se ci sono parametri. Gate come RX, RY e altri devono essere impostati a True, quelli senza parametri devono essere impostati a False, default False.
    :param trainable: Indica se contiene parametri da addestrare. Se il layer usa dati di input esterni per costruire una matrice di gate logico, impostare a False. Se contiene parametri da addestrare, e True. Default False.
    :param init_params: Parametri di inizializzazione, usati per codificare dati classici QTensor, default None. Se e un gate logico con parametri con p parametri, la dimensione del dato di input deve essere [1,p] o [p].
    :param wires: Indice del bit su cui agisce il gate, default None.
    :param dtype: La precisione dei dati della matrice interna del gate logico puo essere impostata a pyvqnet.kcomplex64 o pyvqnet.kcomplex128, corrispondenti rispettivamente a parametri di input float o double.
    :param use_dagger: Se utilizzare la versione trasposta coniugata di questo gate, default False.
    :return: Un Module che puo essere usato per addestrare il modello.

    Example::

        from pyvqnet.qnn.vqc import CZ,QMachine
        device = QMachine(4)
        layer = CZ(wires=[0,1])
        batchsize = 2
        device.reset_states(batchsize)
        layer(q_machine = device)
        print(device.states)
        
rxx
----------------

.. py:function:: pyvqnet.qnn.vqc.rxx(q_machine, wires, params=None,  use_dagger=False)
    
    Applica il gate logico quantistico rxx ai vettori di stato in ``q_machine``.

    :param q_machine: dispositivo macchina virtuale quantistica.
    :param wires: indice del qubit.
    :param params: matrice dei parametri, default None. Per una funzione operativa di gate logico con p parametri, la dimensione del parametro di input deve essere [1,p] o [p].
    :param use_dagger: se utilizzare la trasposta coniugata, default False.


    Example::
        
        from pyvqnet.qnn.vqc import rxx,QMachine
        from pyvqnet.tensor import QTensor
        qm  = QMachine(4)
        rxx(q_machine=qm,wires=[1,0],params=QTensor([0.2]))
        print(qm.states)
        # [[[[[0.9950042+0.j        0.       +0.j       ]
        #     [0.       +0.j        0.       +0.j       ]]
        # 
        #    [[0.       +0.j        0.       +0.j       ]
        #     [0.       +0.j        0.       +0.j       ]]]
        # 
        # 
        #   [[[0.       +0.j        0.       +0.j       ]
        #     [0.       +0.j        0.       +0.j       ]]
        # 
        #    [[0.       -0.0998334j 0.       +0.j       ]
        #     [0.       +0.j        0.       +0.j       ]]]]]

RXX
------------------------------------------


.. py:class:: pyvqnet.qnn.vqc.RXX(has_params: bool = False,trainable: bool = False,init_params=None,wires=None,dtype=pyvqnet.kcomplex64,use_dagger=False)
    
    Definisce una classe di gate logico RXX.

    :param has_params: Indica se ci sono parametri. Gate come RX, RY e altri devono essere impostati a True, quelli senza parametri devono essere impostati a False, default False.
    :param trainable: Indica se contiene parametri da addestrare. Se il layer usa dati di input esterni per costruire una matrice di gate logico, impostare a False. Se contiene parametri da addestrare, e True. Default False.
    :param init_params: Parametri di inizializzazione, usati per codificare dati classici QTensor, default None. Se e un gate logico con parametri con p parametri, la dimensione del dato di input deve essere [1,p] o [p].
    :param wires: Indice del bit su cui agisce il gate, default None.
    :param dtype: La precisione dei dati della matrice interna del gate logico puo essere impostata a pyvqnet.kcomplex64 o pyvqnet.kcomplex128, corrispondenti rispettivamente a parametri di input float o double.
    :param use_dagger: Se utilizzare la versione trasposta coniugata di questo gate, default False.
    :return: Un Module che puo essere usato per addestrare il modello.

    Example::

        from pyvqnet.qnn.vqc import RXX,QMachine
        device = QMachine(4)
        layer = RXX(has_params= True, trainable= True, wires=[0,2])
        batchsize = 2
        device.reset_states(batchsize)
        layer(q_machine = device)
        print(device.states)

ryy
---------------

.. py:function:: pyvqnet.qnn.vqc.ryy(q_machine, wires, params=None,  use_dagger=False)
    
    Applica il gate logico quantistico ryy ai vettori di stato in ``q_machine``.

    :param q_machine: dispositivo macchina virtuale quantistica.
    :param wires: indice del qubit.
    :param params: matrice dei parametri, default None. Per una funzione operativa di gate logico con p parametri, la dimensione del parametro di input deve essere [1,p] o [p].
    :param use_dagger: se utilizzare la trasposta coniugata, default False.


    Example::
        
        from pyvqnet.qnn.vqc import ryy,QMachine
        from pyvqnet.tensor import QTensor
        qm  = QMachine(4)
        ryy(q_machine=qm,wires=[1,0],params=QTensor([0.2]))
        print(qm.states)
        # [[[[[0.9950042+0.j        0.       +0.j       ]
        #     [0.       +0.j        0.       +0.j       ]]
        # 
        #    [[0.       +0.j        0.       +0.j       ]
        #     [0.       +0.j        0.       +0.j       ]]]
        # 
        # 
        #   [[[0.       +0.j        0.       +0.j       ]
        #     [0.       +0.j        0.       +0.j       ]]
        # 
        #    [[0.       +0.0998334j 0.       +0.j       ]
        #     [0.       +0.j        0.       +0.j       ]]]]]

RYY
------------------------------------------


.. py:class:: pyvqnet.qnn.vqc.RYY(has_params: bool = False,trainable: bool = False,init_params=None,wires=None,dtype=pyvqnet.kcomplex64,use_dagger=False)
    
    Definisce una classe di gate logico RYY.

    :param has_params: Indica se ci sono parametri. Gate come RX, RY e altri devono essere impostati a True, quelli senza parametri devono essere impostati a False, default False.
    :param trainable: Indica se contiene parametri da addestrare. Se il layer usa dati di input esterni per costruire una matrice di gate logico, impostare a False. Se contiene parametri da addestrare, e True. Default False.
    :param init_params: Parametri di inizializzazione, usati per codificare dati classici QTensor, default None. Se e un gate logico con parametri con p parametri, la dimensione del dato di input deve essere [1,p] o [p].
    :param wires: Indice del bit su cui agisce il gate, default None.
    :param dtype: La precisione dei dati della matrice interna del gate logico puo essere impostata a pyvqnet.kcomplex64 o pyvqnet.kcomplex128, corrispondenti rispettivamente a parametri di input float o double.
    :param use_dagger: Se utilizzare la versione trasposta coniugata di questo gate, default False.
    :return: Un Module che puo essere usato per addestrare il modello.

    Example::

        from pyvqnet.qnn.vqc import RYY,QMachine
        device = QMachine(4)
        layer = RYY(has_params= True, trainable= True, wires=[0,2])
        batchsize = 2
        device.reset_states(batchsize)
        layer(q_machine = device)
        print(device.states)


rzz
---------------

.. py:function:: pyvqnet.qnn.vqc.rzz(q_machine, wires, params=None,  use_dagger=False)
    
    Applica il gate logico quantistico rzz ai vettori di stato in ``q_machine``.

    :param q_machine: dispositivo macchina virtuale quantistica.
    :param wires: indice del qubit.
    :param params: matrice dei parametri, default None. Per una funzione operativa di gate logico con p parametri, la dimensione del parametro di input deve essere [1,p] o [p].
    :param use_dagger: se utilizzare la trasposta coniugata, default False.


    Example::
        
        from pyvqnet.qnn.vqc import rzz,QMachine
        from pyvqnet.tensor import QTensor
        qm  = QMachine(4)
        rzz(q_machine=qm,wires=[1,0],params=QTensor([0.2]))
        print(qm.states)
        # [[[[[0.9950042-0.0998334j 0.       +0.j       ]
        #     [0.       +0.j        0.       +0.j       ]]
        # 
        #    [[0.       +0.j        0.       +0.j       ]
        #     [0.       +0.j        0.       +0.j       ]]]
        # 
        # 
        #   [[[0.       +0.j        0.       +0.j       ]
        #     [0.       +0.j        0.       +0.j       ]]
        # 
        #    [[0.       +0.j        0.       +0.j       ]
        #     [0.       +0.j        0.       +0.j       ]]]]]

RZZ
------------------------------------------


.. py:class:: pyvqnet.qnn.vqc.RZZ(has_params: bool = False,trainable: bool = False,init_params=None,wires=None,dtype=pyvqnet.kcomplex64,use_dagger=False)
    
    Definisce una classe di gate logico RZZ.

    :param has_params: Indica se ci sono parametri. Gate come RX, RY e altri devono essere impostati a True, quelli senza parametri devono essere impostati a False, default False.
    :param trainable: Indica se contiene parametri da addestrare. Se il layer usa dati di input esterni per costruire una matrice di gate logico, impostare a False. Se contiene parametri da addestrare, e True. Default False.
    :param init_params: Parametri di inizializzazione, usati per codificare dati classici QTensor, default None. Se e un gate logico con parametri con p parametri, la dimensione del dato di input deve essere [1,p] o [p].
    :param wires: Indice del bit su cui agisce il gate, default None.
    :param dtype: La precisione dei dati della matrice interna del gate logico puo essere impostata a pyvqnet.kcomplex64 o pyvqnet.kcomplex128, corrispondenti rispettivamente a parametri di input float o double.
    :param use_dagger: Se utilizzare la versione trasposta coniugata di questo gate, default False.
    :return: Un Module che puo essere usato per addestrare il modello.

    Example::

        from pyvqnet.qnn.vqc import RZZ,QMachine
        device = QMachine(4)
        layer = RZZ(has_params= True, trainable= True, wires=[0,2])
        batchsize = 2
        device.reset_states(batchsize)
        layer(q_machine = device)
        print(device.states)

rzx
-------------

.. py:function:: pyvqnet.qnn.vqc.rzx(q_machine, wires, params=None,  use_dagger=False)
    
    Applica il gate logico quantistico rzx ai vettori di stato in ``q_machine``.

    :param q_machine: dispositivo macchina virtuale quantistica.
    :param wires: indice del qubit.
    :param params: matrice dei parametri, default None. Per una funzione operativa di gate logico con p parametri, la dimensione del parametro di input deve essere [1,p] o [p].
    :param use_dagger: se utilizzare la trasposta coniugata, default False.


    Example::
        
        from pyvqnet.qnn.vqc import rzx,QMachine
        from pyvqnet.tensor import QTensor
        qm  = QMachine(4)
        rzx(q_machine=qm,wires=[1,0],params=QTensor([0.2]))
        print(qm.states)
        # [[[[[0.9950042+0.j        0.       +0.j       ]
        #     [0.       +0.j        0.       +0.j       ]]
        # 
        #    [[0.       +0.j        0.       +0.j       ]
        #     [0.       +0.j        0.       +0.j       ]]]
        # 
        # 
        #   [[[0.       -0.0998334j 0.       +0.j       ]
        #     [0.       +0.j        0.       +0.j       ]]
        # 
        #    [[0.       +0.j        0.       +0.j       ]
        #     [0.       +0.j        0.       +0.j       ]]]]]

RZX
------------------------------------------


.. py:class:: pyvqnet.qnn.vqc.RZX(has_params: bool = False,trainable: bool = False,init_params=None,wires=None,dtype=pyvqnet.kcomplex64,use_dagger=False)
    
    Definisce una classe di gate logico RZX.

    :param has_params: Indica se ci sono parametri. Gate come RX, RY e altri devono essere impostati a True, quelli senza parametri devono essere impostati a False, default False.
    :param trainable: Indica se contiene parametri da addestrare. Se il layer usa dati di input esterni per costruire una matrice di gate logico, impostare a False. Se contiene parametri da addestrare, e True. Default False.
    :param init_params: Parametri di inizializzazione, usati per codificare dati classici QTensor, default None. Se e un gate logico con parametri con p parametri, la dimensione del dato di input deve essere [1,p] o [p].
    :param wires: Indice del bit su cui agisce il gate, default None.
    :param dtype: La precisione dei dati della matrice interna del gate logico puo essere impostata a pyvqnet.kcomplex64 o pyvqnet.kcomplex128, corrispondenti rispettivamente a parametri di input float o double.
    :param use_dagger: Se utilizzare la versione trasposta coniugata di questo gate, default False.
    :return: Un Module che puo essere usato per addestrare il modello.

    Example::

        from pyvqnet.qnn.vqc import RZX,QMachine
        device = QMachine(4)
        layer = RZX(has_params= True, trainable= True, wires=[0,2])
        batchsize = 2
        device.reset_states(batchsize)
        layer(q_machine = device)
        print(device.states)

toffoli
--------------------------

.. py:function:: pyvqnet.qnn.vqc.toffoli(q_machine, wires, params=None,  use_dagger=False)
    
    Applica il gate logico quantistico toffoli ai vettori di stato in ``q_machine``.

    :param q_machine: dispositivo macchina virtuale quantistica.
    :param wires: indice del qubit.
    :param params: matrice dei parametri, default None. Per una funzione operativa di gate logico con p parametri, la dimensione del parametro di input deve essere [1,p] o [p].
    :param use_dagger: se utilizzare la trasposta coniugata, default False.
    :return: Output QTensor.

    Example::
        
        from pyvqnet.qnn.vqc import toffoli,QMachine
        qm  = QMachine(4)
        toffoli(q_machine=qm,wires=[0,1,2])
        print(qm.states)
        # [[[[[1.+0.j 0.+0.j]
        #     [0.+0.j 0.+0.j]]
        # 
        #    [[0.+0.j 0.+0.j]
        #     [0.+0.j 0.+0.j]]]
        # 
        # 
        #   [[[0.+0.j 0.+0.j]
        #     [0.+0.j 0.+0.j]]
        # 
        #    [[0.+0.j 0.+0.j]
        #     [0.+0.j 0.+0.j]]]]]

Toffoli
-----------------------------------


.. py:class:: pyvqnet.qnn.vqc.Toffoli(has_params: bool = False,trainable: bool = False,init_params=None,wires=None,dtype=pyvqnet.kcomplex64,use_dagger=False)
    
    Definisce una classe di gate logico Toffoli.

    :param has_params: Indica se ci sono parametri. Gate come RX, RY e altri devono essere impostati a True, quelli senza parametri devono essere impostati a False, default False.
    :param trainable: Indica se contiene parametri da addestrare. Se il layer usa dati di input esterni per costruire una matrice di gate logico, impostare a False. Se contiene parametri da addestrare, e True. Default False.
    :param init_params: Parametri di inizializzazione, usati per codificare dati classici QTensor, default None. Se e un gate logico con parametri con p parametri, la dimensione del dato di input deve essere [1,p] o [p].
    :param wires: Indice del bit su cui agisce il gate, default None.
    :param dtype: La precisione dei dati della matrice interna del gate logico puo essere impostata a pyvqnet.kcomplex64 o pyvqnet.kcomplex128, corrispondenti rispettivamente a parametri di input float o double.
    :param use_dagger: Se utilizzare la versione trasposta coniugata di questo gate, default False.
    :return: Un Module che puo essere usato per addestrare il modello.

    Example::

        from pyvqnet.qnn.vqc import Toffoli,QMachine
        device = QMachine(4)
        layer = Toffoli( wires=[0,2,1])
        batchsize = 2
        device.reset_states(batchsize)
        layer(q_machine = device)
        print(device.states)


isingxx
----------------------

.. py:function:: pyvqnet.qnn.vqc.isingxx(q_machine, wires, params=None,  use_dagger=False)
    
    Applica il gate logico quantistico isingxx ai vettori di stato in ``q_machine``.

    :param q_machine: dispositivo macchina virtuale quantistica.
    :param wires: indice del qubit.
    :param params: matrice dei parametri, default None. Per una funzione operativa di gate logico con p parametri, la dimensione del parametro di input deve essere [1,p] o [p].
    :param use_dagger: se utilizzare la trasposta coniugata, default False.
    :return: Output QTensor.

    Example::
    
        from pyvqnet.qnn.vqc import QMachine
        import pyvqnet.qnn.vqc as vqc
        from pyvqnet.tensor import QTensor
        qm = QMachine(4)
        vqc.isingxx(q_machine=qm,wires=[0,1], params = QTensor([0.5]))
        print(qm.states)

        # [[[[[0.9689124+0.j       0.       +0.j      ]
        #     [0.       +0.j       0.       +0.j      ]]
        # 
        #    [[0.       +0.j       0.       +0.j      ]
        #     [0.       +0.j       0.       +0.j      ]]]
        # 
        # 
        #   [[[0.       +0.j       0.       +0.j      ]
        #     [0.       +0.j       0.       +0.j      ]]
        # 
        #    [[0.       -0.247404j 0.       +0.j      ]
        #     [0.       +0.j       0.       +0.j      ]]]]]

IsingXX
---------------------------------------


.. py:class:: pyvqnet.qnn.vqc.IsingXX(has_params: bool = False,trainable: bool = False,init_params=None,wires=None,dtype=pyvqnet.kcomplex64,use_dagger=False)
    
    Definisce una classe di gate logico IsingXX.

    :param has_params: Indica se ci sono parametri. Gate come RX, RY e altri devono essere impostati a True, quelli senza parametri devono essere impostati a False, default False.
    :param trainable: Indica se contiene parametri da addestrare. Se il layer usa dati di input esterni per costruire una matrice di gate logico, impostare a False. Se contiene parametri da addestrare, e True. Default False.
    :param init_params: Parametri di inizializzazione, usati per codificare dati classici QTensor, default None. Se e un gate logico con parametri con p parametri, la dimensione del dato di input deve essere [1,p] o [p].
    :param wires: Indice del bit su cui agisce il gate, default None.
    :param dtype: La precisione dei dati della matrice interna del gate logico puo essere impostata a pyvqnet.kcomplex64 o pyvqnet.kcomplex128, corrispondenti rispettivamente a parametri di input float o double.
    :param use_dagger: Se utilizzare la versione trasposta coniugata di questo gate, default False.
    :return: Un Module che puo essere usato per addestrare il modello.

    Example::

        from pyvqnet.qnn.vqc import IsingXX,QMachine
        device = QMachine(4)
        layer = IsingXX(has_params= True, trainable= True, wires=[0,2])
        batchsize = 2
        device.reset_states(batchsize)
        layer(q_machine = device)
        print(device.states)

isingyy
-------------------

.. py:function:: pyvqnet.qnn.vqc.isingyy(q_machine, wires, params=None,  use_dagger=False)
    
    Applica il gate logico quantistico isingyy ai vettori di stato in ``q_machine``.

    :param q_machine: dispositivo macchina virtuale quantistica.
    :param wires: indice del qubit.
    :param params: matrice dei parametri, default None. Per una funzione operativa di gate logico con p parametri, la dimensione del parametro di input deve essere [1,p] o [p].
    :param use_dagger: se utilizzare la trasposta coniugata, default False.
    :return: Output QTensor.

    Example::
    
        from pyvqnet.qnn.vqc import QMachine
        import pyvqnet.qnn.vqc as vqc
        from pyvqnet.tensor import QTensor
        qm = QMachine(4)
        vqc.isingyy(q_machine=qm,wires=[0,1], params = QTensor([0.5]))
        print(qm.states)

        # [[[[[0.9689124+0.j       0.       +0.j      ]
        #     [0.       +0.j       0.       +0.j      ]]
        # 
        #    [[0.       +0.j       0.       +0.j      ]
        #     [0.       +0.j       0.       +0.j      ]]]
        # 
        # 
        #   [[[0.       +0.j       0.       +0.j      ]
        #     [0.       +0.j       0.       +0.j      ]]
        # 
        #    [[0.       +0.247404j 0.       +0.j      ]
        #     [0.       +0.j       0.       +0.j      ]]]]]

IsingYY
---------------------------------------


.. py:class:: pyvqnet.qnn.vqc.IsingYY(has_params: bool = False,trainable: bool = False,init_params=None,wires=None,dtype=pyvqnet.kcomplex64,use_dagger=False)
    
    Definisce una classe di gate logico IsingYY.

    :param has_params: Indica se ci sono parametri. Gate come RX, RY e altri devono essere impostati a True, quelli senza parametri devono essere impostati a False, default False.
    :param trainable: Indica se contiene parametri da addestrare. Se il layer usa dati di input esterni per costruire una matrice di gate logico, impostare a False. Se contiene parametri da addestrare, e True. Default False.
    :param init_params: Parametri di inizializzazione, usati per codificare dati classici QTensor, default None. Se e un gate logico con parametri con p parametri, la dimensione del dato di input deve essere [1,p] o [p].
    :param wires: Indice del bit su cui agisce il gate, default None.
    :param dtype: La precisione dei dati della matrice interna del gate logico puo essere impostata a pyvqnet.kcomplex64 o pyvqnet.kcomplex128, corrispondenti rispettivamente a parametri di input float o double.
    :param use_dagger: Se utilizzare la versione trasposta coniugata di questo gate, default False.
    :return: Un Module che puo essere usato per addestrare il modello.

    Example::

        from pyvqnet.qnn.vqc import IsingYY,QMachine
        device = QMachine(4)
        layer = IsingYY(has_params= True, trainable= True, wires=[0,2])
        batchsize = 2
        device.reset_states(batchsize)
        layer(q_machine = device)
        print(device.states)

isingzz
---------------------

.. py:function:: pyvqnet.qnn.vqc.isingzz(q_machine, wires, params=None,  use_dagger=False)
    
    Applica il gate logico quantistico isingzz ai vettori di stato in ``q_machine``.

    :param q_machine: dispositivo macchina virtuale quantistica.
    :param wires: indice del qubit.
    :param params: matrice dei parametri, default None. Per una funzione operativa di gate logico con p parametri, la dimensione del parametro di input deve essere [1,p] o [p].
    :param use_dagger: se utilizzare la trasposta coniugata, default False.
    :return: Output QTensor.

    Example::
    
        from pyvqnet.qnn.vqc import QMachine
        import pyvqnet.qnn.vqc as vqc
        from pyvqnet.tensor import QTensor
        qm = QMachine(4)
        vqc.isingzz(q_machine=qm,wires=[0,1], params = QTensor([0.5]))
        print(qm.states)

        # [[[[[0.9689124-0.247404j 0.       +0.j      ]
        #     [0.       +0.j       0.       +0.j      ]]
        # 
        #    [[0.       +0.j       0.       +0.j      ]
        #     [0.       +0.j       0.       +0.j      ]]]
        # 
        # 
        #   [[[0.       +0.j       0.       +0.j      ]
        #     [0.       +0.j       0.       +0.j      ]]
        # 
        #    [[0.       +0.j       0.       +0.j      ]
        #     [0.       +0.j       0.       +0.j      ]]]]]


IsingZZ
---------------------------------------


.. py:class:: pyvqnet.qnn.vqc.IsingZZ(has_params: bool = False,trainable: bool = False,init_params=None,wires=None,dtype=pyvqnet.kcomplex64,use_dagger=False)
    
    Definisce una classe di gate logico IsingZZ.

    :param has_params: Indica se ci sono parametri. Gate come RX, RY e altri devono essere impostati a True, quelli senza parametri devono essere impostati a False, default False.
    :param trainable: Indica se contiene parametri da addestrare. Se il layer usa dati di input esterni per costruire una matrice di gate logico, impostare a False. Se contiene parametri da addestrare, e True. Default False.
    :param init_params: Parametri di inizializzazione, usati per codificare dati classici QTensor, default None. Se e un gate logico con parametri con p parametri, la dimensione del dato di input deve essere [1,p] o [p].
    :param wires: Indice del bit su cui agisce il gate, default None.
    :param dtype: La precisione dei dati della matrice interna del gate logico puo essere impostata a pyvqnet.kcomplex64 o pyvqnet.kcomplex128, corrispondenti rispettivamente a parametri di input float o double.
    :param use_dagger: Se utilizzare la versione trasposta coniugata di questo gate, default False.
    :return: Un Module che puo essere usato per addestrare il modello.

    Example::

        from pyvqnet.qnn.vqc import IsingZZ,QMachine
        device = QMachine(4)
        layer = IsingZZ(has_params= True, trainable= True, wires=[0,2])
        batchsize = 2
        device.reset_states(batchsize)
        layer(q_machine = device)
        print(device.states)

isingxy
---------------------

.. py:function:: pyvqnet.qnn.vqc.isingxy(q_machine, wires, params=None,  use_dagger=False)
    
    Applica il gate logico quantistico isingxy ai vettori di stato in ``q_machine``.

    :param q_machine: dispositivo macchina virtuale quantistica.
    :param wires: indice del qubit.
    :param params: matrice dei parametri, default None. Per una funzione operativa di gate logico con p parametri, la dimensione del parametro di input deve essere [1,p] o [p].
    :param use_dagger: se utilizzare la trasposta coniugata, default False.
    :return: Output QTensor.

    Example::
    
        from pyvqnet.qnn.vqc import QMachine
        import pyvqnet.qnn.vqc as vqc
        from pyvqnet.tensor import QTensor
        qm = QMachine(4)
        vqc.isingxy(q_machine=qm,wires=[0,1], params = QTensor([0.5]))
        print(qm.states)

        # [[[[[1.+0.j 0.+0.j]
        #     [0.+0.j 0.+0.j]]
        # 
        #    [[0.+0.j 0.+0.j]
        #     [0.+0.j 0.+0.j]]]
        # 
        # 
        #   [[[0.+0.j 0.+0.j]
        #     [0.+0.j 0.+0.j]]
        # 
        #    [[0.+0.j 0.+0.j]
        #     [0.+0.j 0.+0.j]]]]]

IsingXY
---------------------------------------


.. py:class:: pyvqnet.qnn.vqc.IsingXY(has_params: bool = False,trainable: bool = False,init_params=None,wires=None,dtype=pyvqnet.kcomplex64,use_dagger=False)
    
    Definisce una classe di gate logico IsingXY.

    :param has_params: Indica se ci sono parametri. Gate come RX, RY e altri devono essere impostati a True, quelli senza parametri devono essere impostati a False, default False.
    :param trainable: Indica se contiene parametri da addestrare. Se il layer usa dati di input esterni per costruire una matrice di gate logico, impostare a False. Se contiene parametri da addestrare, e True. Default False.
    :param init_params: Parametri di inizializzazione, usati per codificare dati classici QTensor, default None. Se e un gate logico con parametri con p parametri, la dimensione del dato di input deve essere [1,p] o [p].
    :param wires: Indice del bit su cui agisce il gate, default None.
    :param dtype: La precisione dei dati della matrice interna del gate logico puo essere impostata a pyvqnet.kcomplex64 o pyvqnet.kcomplex128, corrispondenti rispettivamente a parametri di input float o double.
    :param use_dagger: Se utilizzare la versione trasposta coniugata di questo gate, default False.
    :return: Un Module che puo essere usato per addestrare il modello.

     Example::

         from pyvqnet.qnn.vqc import IsingXY,QMachine
         device = QMachine(4)
         layer = IsingXY(has_params= True, trainable= True, wires=[0,2])
         batchsize = 2
         device.reset_states(batchsize)
         layer(q_machine = device)
         print(device.states)

phaseshift
---------------

.. py:function:: pyvqnet.qnn.vqc.phaseshift(q_machine, wires, params=None,  use_dagger=False)
    
    Applica il gate logico quantistico phaseshift ai vettori di stato in ``q_machine``.

    :param q_machine: dispositivo macchina virtuale quantistica.
    :param wires: indice del qubit.
    :param params: matrice dei parametri, default None. Per una funzione operativa di gate logico con p parametri, la dimensione del parametro di input deve essere [1,p] o [p].
    :param use_dagger: se utilizzare la trasposta coniugata, default False.
    :return: Output QTensor.

    Example::
    
        from pyvqnet.qnn.vqc import QMachine
        import pyvqnet.qnn.vqc as vqc
        from pyvqnet.tensor import QTensor
        qm = QMachine(4)
        vqc.phaseshift(q_machine=qm,wires=[0], params = QTensor([0.5]))
        print(qm.states)

        # [[[[[1.+0.j 0.+0.j]
        #     [0.+0.j 0.+0.j]]
        # 
        #    [[0.+0.j 0.+0.j]
        #     [0.+0.j 0.+0.j]]]
        # 
        # 
        #   [[[0.+0.j 0.+0.j]
        #     [0.+0.j 0.+0.j]]
        # 
        #    [[0.+0.j 0.+0.j]
        #     [0.+0.j 0.+0.j]]]]]

PhaseShift
-----------------------------------------------


.. py:class:: pyvqnet.qnn.vqc.PhaseShift(has_params: bool = False,trainable: bool = False,init_params=None,wires=None,dtype=pyvqnet.kcomplex64,use_dagger=False)
    
    Definisce una classe di gate logico PhaseShift.

    :param has_params: Indica se ci sono parametri. Gate come RX, RY e altri devono essere impostati a True, quelli senza parametri devono essere impostati a False, default False.
    :param trainable: Indica se contiene parametri da addestrare. Se il layer usa dati di input esterni per costruire una matrice di gate logico, impostare a False. Se contiene parametri da addestrare, e True. Default False.
    :param init_params: Parametri di inizializzazione, usati per codificare dati classici QTensor, default None. Se e un gate logico con parametri con p parametri, la dimensione del dato di input deve essere [1,p] o [p].
    :param wires: Indice del bit su cui agisce il gate, default None.
    :param dtype: La precisione dei dati della matrice interna del gate logico puo essere impostata a pyvqnet.kcomplex64 o pyvqnet.kcomplex128, corrispondenti rispettivamente a parametri di input float o double.
    :param use_dagger: Se utilizzare la versione trasposta coniugata di questo gate, default False.
    :return: Un Module che puo essere usato per addestrare il modello.

    Example::

        from pyvqnet.qnn.vqc import PhaseShift,QMachine
        device = QMachine(4)
        layer = PhaseShift(has_params= True, trainable= True, wires=1)
        batchsize = 2
        device.reset_states(batchsize)
        layer(q_machine = device)
        print(device.states)

multirz
--------------------

.. py:function:: pyvqnet.qnn.vqc.multirz(q_machine, wires, params=None,  use_dagger=False)
    
    Applica il gate logico quantistico multirz ai vettori di stato in ``q_machine``.

    :param q_machine: dispositivo macchina virtuale quantistica.
    :param wires: indice del qubit.
    :param params: matrice dei parametri, default None. Per una funzione operativa di gate logico con p parametri, la dimensione del parametro di input deve essere [1,p] o [p].
    :param use_dagger: se utilizzare la trasposta coniugata, default False.
    :return: Output QTensor.

    Example::
    
        from pyvqnet.qnn.vqc import QMachine
        import pyvqnet.qnn.vqc as vqc
        from pyvqnet.tensor import QTensor
        qm = QMachine(4)
        vqc.multirz(q_machine=qm,wires=[0, 1], params = QTensor([0.5]))
        print(qm.states)

        # [[[[[0.9689124-0.247404j 0.       +0.j      ]
        #     [0.       +0.j       0.       +0.j      ]]
        # 
        #    [[0.       +0.j       0.       +0.j      ]
        #     [0.       +0.j       0.       +0.j      ]]]
        # 
        # 
        #   [[[0.       +0.j       0.       +0.j      ]
        #     [0.       +0.j       0.       +0.j      ]]
        # 
        #    [[0.       +0.j       0.       +0.j      ]
        #     [0.       +0.j       0.       +0.j      ]]]]]


MultiRZ
-------------------------------------------------


.. py:class:: pyvqnet.qnn.vqc.MultiRZ(has_params: bool = False,trainable: bool = False,init_params=None,wires=None,dtype=pyvqnet.kcomplex64,use_dagger=False)
    
    Definisce una classe di gate logico MultiRZ.

    :param has_params: Indica se ci sono parametri. Gate come RX, RY e altri devono essere impostati a True, quelli senza parametri devono essere impostati a False, default False.
    :param trainable: Indica se contiene parametri da addestrare. Se il layer usa dati di input esterni per costruire una matrice di gate logico, impostare a False. Se contiene parametri da addestrare, e True. Default False.
    :param init_params: Parametri di inizializzazione, usati per codificare dati classici QTensor, default None. Se e un gate logico con parametri con p parametri, la dimensione del dato di input deve essere [1,p] o [p].
    :param wires: Indice del bit su cui agisce il gate, default None.
    :param dtype: La precisione dei dati della matrice interna del gate logico puo essere impostata a pyvqnet.kcomplex64 o pyvqnet.kcomplex128, corrispondenti rispettivamente a parametri di input float o double.
    :param use_dagger: Se utilizzare la versione trasposta coniugata di questo gate, default False.
    :return: Un Module che puo essere usato per addestrare il modello.

    Example::

        from pyvqnet.qnn.vqc import MultiRZ,QMachine
        device = QMachine(4)
        layer = MultiRZ(has_params= True, trainable= True, wires=[0,2])
        batchsize = 2
        device.reset_states(batchsize)
        layer(q_machine = device)
        print(device.states)

sdg
--------------

.. py:function:: pyvqnet.qnn.vqc.sdg(q_machine, wires, params=None,  use_dagger=False)
    
    Applica il gate logico quantistico sdg ai vettori di stato in ``q_machine``.

    :param q_machine: dispositivo macchina virtuale quantistica.
    :param wires: indice del qubit.
    :param params: matrice dei parametri, default None. Per una funzione operativa di gate logico con p parametri, la dimensione del parametro di input deve essere [1,p] o [p].
    :param use_dagger: se utilizzare la trasposta coniugata, default False.


    Example::
    
        from pyvqnet.qnn.vqc import QMachine
        import pyvqnet.qnn.vqc as vqc
        from pyvqnet.tensor import QTensor
        qm = QMachine(4)
        vqc.sdg(q_machine=qm,wires=[0])
        print(qm.states)

        # [[[[[1.+0.j 0.+0.j]
        #     [0.+0.j 0.+0.j]]
        # 
        #    [[0.+0.j 0.+0.j]
        #     [0.+0.j 0.+0.j]]]
        # 
        # 
        #   [[[0.+0.j 0.+0.j]
        #     [0.+0.j 0.+0.j]]
        # 
        #    [[0.+0.j 0.+0.j]
        #     [0.+0.j 0.+0.j]]]]]

SDG
----------------------------------------


.. py:class:: pyvqnet.qnn.vqc.SDG(has_params: bool = False,trainable: bool = False,init_params=None,wires=None,dtype=pyvqnet.kcomplex64,use_dagger=False)
    
    Definisce una classe di gate logico SDG.

    :param has_params: Indica se ci sono parametri. Gate come RX, RY e altri devono essere impostati a True, quelli senza parametri devono essere impostati a False, default False.
    :param trainable: Indica se contiene parametri da addestrare. Se il layer usa dati di input esterni per costruire una matrice di gate logico, impostare a False. Se contiene parametri da addestrare, e True. Default False.
    :param init_params: Parametri di inizializzazione, usati per codificare dati classici QTensor, default None. Se e un gate logico con parametri con p parametri, la dimensione del dato di input deve essere [1,p] o [p].
    :param wires: Indice del bit su cui agisce il gate, default None.
    :param dtype: La precisione dei dati della matrice interna del gate logico puo essere impostata a pyvqnet.kcomplex64 o pyvqnet.kcomplex128, corrispondenti rispettivamente a parametri di input float o double.
    :param use_dagger: Se utilizzare la versione trasposta coniugata di questo gate, default False.
    :return: Un Module che puo essere usato per addestrare il modello.

    Example::
    
        from pyvqnet.qnn.vqc import SDG,QMachine
        device = QMachine(4)
        layer = SDG(wires=0)
        batchsize=1
        device.reset_states(1)
        layer(q_machine = device)
        print(device.states)

tdg
------------------

.. py:function:: pyvqnet.qnn.vqc.tdg(q_machine, wires, params=None,  use_dagger=False)
    
    Applica il gate logico quantistico tdg ai vettori di stato in ``q_machine``.

    :param q_machine: dispositivo macchina virtuale quantistica.
    :param wires: indice del qubit.
    :param params: matrice dei parametri, default None. Per una funzione operativa di gate logico con p parametri, la dimensione del parametro di input deve essere [1,p] o [p].
    :param use_dagger: se utilizzare la trasposta coniugata, default False.


    Example::
    
        from pyvqnet.qnn.vqc import QMachine
        import pyvqnet.qnn.vqc as vqc
        from pyvqnet.tensor import QTensor
        qm = QMachine(4)
        vqc.tdg(q_machine=qm,wires=[0])
        print(qm.states)

        # [[[[[1.+0.j 0.+0.j]
        #     [0.+0.j 0.+0.j]]
        # 
        #    [[0.+0.j 0.+0.j]
        #     [0.+0.j 0.+0.j]]]
        # 
        # 
        #   [[[0.+0.j 0.+0.j]
        #     [0.+0.j 0.+0.j]]
        # 
        #    [[0.+0.j 0.+0.j]
        #     [0.+0.j 0.+0.j]]]]]

TDG
---------------------------------

.. py:class:: pyvqnet.qnn.vqc.TDG(has_params: bool = False,trainable: bool = False,init_params=None,wires=None,dtype=pyvqnet.kcomplex64,use_dagger=False)
    
    Definisce una classe di gate logico TDG.

    :param has_params: Indica se ci sono parametri. Gate come RX, RY e altri devono essere impostati a True, quelli senza parametri devono essere impostati a False, default False.
    :param trainable: Indica se contiene parametri da addestrare. Se il layer usa dati di input esterni per costruire una matrice di gate logico, impostare a False. Se contiene parametri da addestrare, e True. Default False.
    :param init_params: Parametri di inizializzazione, usati per codificare dati classici QTensor, default None. Se e un gate logico con parametri con p parametri, la dimensione del dato di input deve essere [1,p] o [p].
    :param wires: Indice del bit su cui agisce il gate, default None.
    :param dtype: La precisione dei dati della matrice interna del gate logico puo essere impostata a pyvqnet.kcomplex64 o pyvqnet.kcomplex128, corrispondenti rispettivamente a parametri di input float o double.
    :param use_dagger: Se utilizzare la versione trasposta coniugata di questo gate, default False.
    :return: Un Module che puo essere usato per addestrare il modello.

    Example::
    
        from pyvqnet.qnn.vqc import TDG,QMachine
        device = QMachine(4)
        layer = TDG(wires=0)
        batchsize=1
        device.reset_states(1)
        layer(q_machine = device)
        print(device.states)


controlledphaseshift
-----------------------------

.. py:function:: pyvqnet.qnn.vqc.controlledphaseshift(q_machine, wires, params=None,  use_dagger=False)
    
    Applica il gate logico quantistico controlledphaseshift ai vettori di stato in ``q_machine``.

    :param q_machine: dispositivo macchina virtuale quantistica.
    :param wires: indice del qubit.
    :param params: matrice dei parametri, default None. Per una funzione operativa di gate logico con p parametri, la dimensione del parametro di input deve essere [1,p] o [p].
    :param use_dagger: se utilizzare la trasposta coniugata, default False.


    Example::
    
        from pyvqnet.qnn.vqc import QMachine
        import pyvqnet.qnn.vqc as vqc
        from pyvqnet.tensor import QTensor
        qm = QMachine(4)
        for i in range(4):
            vqc.hadamard(q_machine=qm, wires=i)
        vqc.controlledphaseshift(q_machine=qm,params=QTensor([0.5]),wires=[0,1])
        print(qm.states)

        # [[[[[0.25     +0.j        0.25     +0.j       ]
        #     [0.25     +0.j        0.25     +0.j       ]]
        # 
        #    [[0.25     +0.j        0.25     +0.j       ]
        #     [0.25     +0.j        0.25     +0.j       ]]]
        # 
        # 
        #   [[[0.25     +0.j        0.25     +0.j       ]
        #     [0.25     +0.j        0.25     +0.j       ]]
        # 
        #    [[0.2193956+0.1198564j 0.2193956+0.1198564j]
        #     [0.2193956+0.1198564j 0.2193956+0.1198564j]]]]]

ControlledPhaseShift
----------------------------------------


.. py:class:: pyvqnet.qnn.vqc.ControlledPhaseShift(has_params: bool = False,trainable: bool = False,init_params=None,wires=None,dtype=pyvqnet.kcomplex64,use_dagger=False)
    
    Definisce una classe di gate logico ControlledPhaseShift.

    :param has_params: Indica se ci sono parametri. Gate come RX, RY e altri devono essere impostati a True, quelli senza parametri devono essere impostati a False, default False.
    :param trainable: Indica se contiene parametri da addestrare. Se il layer usa dati di input esterni per costruire una matrice di gate logico, impostare a False. Se contiene parametri da addestrare, e True. Default False.
    :param init_params: Parametri di inizializzazione, usati per codificare dati classici QTensor, default None. Se e un gate logico con parametri con p parametri, la dimensione del dato di input deve essere [1,p] o [p].
    :param wires: Indice del bit su cui agisce il gate, default None.
    :param dtype: La precisione dei dati della matrice interna del gate logico puo essere impostata a pyvqnet.kcomplex64 o pyvqnet.kcomplex128, corrispondenti rispettivamente a parametri di input float o double.
    :param use_dagger: Se utilizzare la versione trasposta coniugata di questo gate, default False.
    :return: Un Module che puo essere usato per addestrare il modello.

    Example::

        from pyvqnet.qnn.vqc import ControlledPhaseShift,QMachine
        device = QMachine(4)
        layer = ControlledPhaseShift(has_params= True, trainable= True, wires=[0,2])
        batchsize = 2
        device.reset_states(batchsize)
        layer(q_machine = device)
        print(device.states)

multicontrolledx
---------------------------------------------------------------

.. py:function:: pyvqnet.qnn.vqc.multicontrolledx(q_machine, wires, params=None, use_dagger=False,control_values=None)
    
    Applica il gate logico quantistico multicontrolledx ai vettori di stato in ``q_machine``.

    :param q_machine: dispositivo macchina virtuale quantistica.
    :param wires: qubit index.
    :param params: parameter matrix, default is None.
    :param use_dagger: whether to conjugate transpose, default is False.
    :param control_values: valore di controllo, default None, controlla quando il bit e 1.


    Example::
 


        from pyvqnet.qnn.vqc import QMachine
        import pyvqnet.qnn.vqc as vqc
        from pyvqnet.tensor import QTensor
        qm = QMachine(4)
        for i in range(4):
            vqc.hadamard(q_machine=qm, wires=i)
        vqc.phaseshift(q_machine=qm,wires=[0], params = QTensor([0.5]))
        vqc.phaseshift(q_machine=qm,wires=[1], params = QTensor([2]))
        vqc.phaseshift(q_machine=qm,wires=[3], params = QTensor([3]))
        vqc.multicontrolledx(qm, wires=[0, 1, 3, 2])
        print(qm.states)

        # [[[[[ 0.25     +0.j       ,-0.2474981+0.03528j  ],
        #     [ 0.25     +0.j       ,-0.2474981+0.03528j  ]],

        #    [[-0.1040367+0.2273243j, 0.0709155-0.239731j ],
        #     [-0.1040367+0.2273243j, 0.0709155-0.239731j ]]],


        #   [[[ 0.2193956+0.1198564j,-0.2341141-0.0876958j],
        #     [ 0.2193956+0.1198564j,-0.2341141-0.0876958j]],

        #    [[-0.2002859+0.149618j , 0.1771674-0.176385j ],
        #     [-0.2002859+0.149618j , 0.1771674-0.176385j ]]]]]


MultiControlledX
---------------------------------------------------------------

.. py:class:: pyvqnet.qnn.vqc.MultiControlledX(has_params: bool = False,trainable: bool = False,init_params=None,wires=None,dtype=pyvqnet.kcomplex64,use_dagger=False,control_values=None)
    
    Definisce una classe di gate logico MultiControlledX.

    :param has_params: Indica se ha parametri. Gate come RX, RY e altri devono essere impostati a True, quelli senza parametri devono essere impostati a False, default False.
    :param trainable: Indica se ha parametri da addestrare. Se il layer usa dati di input esterni per costruire la matrice del gate logico, impostare a False. Se i parametri da addestrare devono essere inizializzati da questo layer, e True, default False.
    :param init_params: Parametri di inizializzazione usati per codificare dati classici QTensor, default None.
    :param wires: Bit index of the wire action, the default is None.
    :param dtype: La precisione dei dati della matrice interna del gate logico, puo essere impostata a pyvqnet.kcomplex64 o pyvqnet.kcomplex128, corrispondenti rispettivamente a parametri di input float o double.
    :param use_dagger: Whether to use the transposed conjugate version of the gate, the default is False.
    :param control_values: valore di controllo, default None, controlla quando il bit e 1.

    :return: Un Module che puo essere usato per addestrare il modello.

    Example::

        from pyvqnet.qnn.vqc import QMachine
        import pyvqnet.qnn.vqc as vqc
        from pyvqnet.tensor import QTensor
        from pyvqnet import kcomplex64

        qm = QMachine(4,dtype=kcomplex64)
        qm.reset_states(2)
        for i in range(4):
            vqc.hadamard(q_machine=qm, wires=i)
        vqc.isingzz(q_machine=qm, params=QTensor([0.25]), wires=[1,0])
        vqc.double_excitation(q_machine=qm, params=QTensor([0.55]), wires=[0,1,2,3])

        mcx = vqc.MultiControlledX( 
                        init_params=None,
                        wires=[2,3,0,1],
                        dtype=kcomplex64,
                        use_dagger=False,control_values=[1,0,0])
        y = mcx(q_machine = qm)
        print(qm.states)
        """
        [[[[[0.2480494-0.0311687j,0.2480494-0.0311687j],
            [0.2480494+0.0311687j,0.1713719-0.0215338j]],

        [[0.2480494+0.0311687j,0.2480494+0.0311687j],
            [0.2480494-0.0311687j,0.2480494+0.0311687j]]],


        [[[0.2480494+0.0311687j,0.2480494+0.0311687j],
            [0.2480494+0.0311687j,0.2480494+0.0311687j]],

        [[0.306086 -0.0384613j,0.2480494-0.0311687j],
            [0.2480494-0.0311687j,0.2480494-0.0311687j]]]],



        [[[[0.2480494-0.0311687j,0.2480494-0.0311687j],
            [0.2480494+0.0311687j,0.1713719-0.0215338j]],

        [[0.2480494+0.0311687j,0.2480494+0.0311687j],
            [0.2480494-0.0311687j,0.2480494+0.0311687j]]],


        [[[0.2480494+0.0311687j,0.2480494+0.0311687j],
            [0.2480494+0.0311687j,0.2480494+0.0311687j]],

        [[0.306086 -0.0384613j,0.2480494-0.0311687j],
            [0.2480494-0.0311687j,0.2480494-0.0311687j]]]]]
        """


single_excitation
-----------------------------

.. py:function:: pyvqnet.qnn.vqc.single_excitation(q_machine, wires, params=None,  use_dagger=False)
    
    Applica il gate logico quantistico single_excitation ai vettori di stato in ``q_machine``.

    :param q_machine: dispositivo macchina virtuale quantistica.
    :param wires: indice del qubit.
    :param params: matrice dei parametri, default None. Per una funzione operativa di gate logico con p parametri, la dimensione del parametro di input deve essere [1,p] o [p].
    :param use_dagger: se utilizzare la trasposta coniugata, default False.


    Example::
    
        from pyvqnet.qnn.vqc import QMachine
        import pyvqnet.qnn.vqc as vqc
        from pyvqnet.tensor import QTensor
        qm = QMachine(4)
        vqc.single_excitation(q_machine=qm, wires=[0, 1],params=QTensor([0.5]))
        print(qm.states)

        # [[[[[1.+0.j 0.+0.j]
        #     [0.+0.j 0.+0.j]]
        # 
        #    [[0.+0.j 0.+0.j]
        #     [0.+0.j 0.+0.j]]]
        # 
        # 
        #   [[[0.+0.j 0.+0.j]
        #     [0.+0.j 0.+0.j]]
        # 
        #    [[0.+0.j 0.+0.j]
        #     [0.+0.j 0.+0.j]]]]]

double_excitation
--------------------------

.. py:function:: pyvqnet.qnn.vqc.double_excitation(q_machine, wires, params=None,  use_dagger=False)
    
    Applica il gate logico quantistico double_excitation ai vettori di stato in ``q_machine``.

    :param q_machine: dispositivo macchina virtuale quantistica.
    :param wires: indice del qubit.
    :param params: matrice dei parametri, default None. Per una funzione operativa di gate logico con p parametri, la dimensione del parametro di input deve essere [1,p] o [p].
    :param use_dagger: se utilizzare la trasposta coniugata, default False.


    Example::
    
        from pyvqnet.qnn.vqc import QMachine
        import pyvqnet.qnn.vqc as vqc
        from pyvqnet.tensor import QTensor
        qm = QMachine(4)
        for i in range(4):
            vqc.hadamard(q_machine=qm, wires=i)
        vqc.isingzz(q_machine=qm, params=QTensor([0.55]), wires=[1,0])
        vqc.double_excitation(q_machine=qm, params=QTensor([0.55]), wires=[0,1,2,3])
        print(qm.states)

        # [[[[[0.2406063-0.0678867j 0.2406063-0.0678867j]
        #     [0.2406063-0.0678867j 0.1662296-0.0469015j]]
        # 
        #    [[0.2406063+0.0678867j 0.2406063+0.0678867j]
        #     [0.2406063+0.0678867j 0.2406063+0.0678867j]]]
        # 
        # 
        #   [[[0.2406063+0.0678867j 0.2406063+0.0678867j]
        #     [0.2406063+0.0678867j 0.2406063+0.0678867j]]
        # 
        #    [[0.2969014-0.0837703j 0.2406063-0.0678867j]
        #     [0.2406063-0.0678867j 0.2406063-0.0678867j]]]]]  

VQC_BasisEmbedding
--------------------------

.. py:function:: pyvqnet.qnn.vqc.VQC_BasisEmbedding(basis_state,q_machine)

    Codifica le caratteristiche binarie ``basis_state`` nello stato fondamentale di n qubit in ``q_machine``.

    Ad esempio, per ``basis_state=([0, 1, 1])``, lo stato fondamentale del sistema quantistico e :math:`|011 \rangle`.

    :param basis_state: input binario di dimensione ``(n)``.
    :param q_machine: dispositivo macchina virtuale quantistica.


    Example::
        
        from pyvqnet.qnn.vqc import VQC_BasisEmbedding,QMachine
        qm  = QMachine(3)
        VQC_BasisEmbedding(basis_state=[1,1,0],q_machine=qm)
        print(qm.states)
        # [[[[0.+0.j 0.+0.j]
        #    [0.+0.j 0.+0.j]]
        # 
        #   [[0.+0.j 0.+0.j]
        #    [1.+0.j 0.+0.j]]]]


VQC_AngleEmbedding
--------------------------

.. py:function:: pyvqnet.qnn.vqc.VQC_AngleEmbedding(input_feat, wires, q_machine: pyvqnet.qnn.vqc.QMachine, rotation: str = "X")

    Codifica le :math:`N` caratteristiche nell'angolo di rotazione del :math:`n`-esimo qubit, dove :math:`N \leq n` in ``q_machine`` .

    La rotazione puo essere selezionata come: 'X', 'Y', 'Z', come definito dal parametro ``rotation``:

    * ``rotation='X'`` Usa la caratteristica come angolo per la rotazione RX.

    * ``rotation='Y'`` Usa la caratteristica come angolo per la rotazione RY.

    * ``rotation='Z'`` Usa la caratteristica come angolo per la rotazione RZ.

     ``wires`` indica l'indice dei gate di rotazione sui qubit.

    :param input_feat: array che rappresenta i parametri.
    :param wires: indice del qubit.
    :param q_machine: Dispositivo macchina virtuale quantistica.
    :param rotation: Gate di rotazione, default "X".


    Example::

        from pyvqnet.qnn.vqc import VQC_AngleEmbedding, QMachine
        from pyvqnet.tensor import QTensor
        qm  = QMachine(2)
        VQC_AngleEmbedding(QTensor([2.2, 1]), [0, 1], q_machine=qm, rotation='X')
        print(qm.states)
        # [[[ 0.398068 +0.j         0.       -0.2174655j]
        #   [ 0.       -0.7821081j -0.4272676+0.j       ]]]

        VQC_AngleEmbedding(QTensor([2.2, 1]), [0, 1], q_machine=qm, rotation='Y')

        print(qm.states)
        # [[[-0.0240995+0.6589843j  0.4207355+0.2476033j]
        #   [ 0.4042482-0.2184162j  0.       -0.3401631j]]]

        VQC_AngleEmbedding(QTensor([2.2, 1]), [0, 1], q_machine=qm, rotation='Z')

        print(qm.states)

        # [[[0.659407 +0.0048471j 0.4870554-0.0332093j]
        #   [0.4569675+0.047989j  0.340018 +0.0099326j]]]

VQC_AmplitudeEmbedding
--------------------------

.. py:function:: pyvqnet.qnn.vqc.VQC_AmplitudeEmbeddingCircuit(input_feature, q_machine)

    Codifica una caratteristica di :math:`2^n` in un vettore di ampiezza di :math:`n` qubit in ``q_machine`` .

    :param input_feature: Un array numpy che rappresenta i parametri.
    :param q_machine: Dispositivo macchina virtuale quantistica.


    Example::

        from pyvqnet.qnn.vqc import VQC_AmplitudeEmbedding, QMachine
        from pyvqnet.tensor import QTensor
        qm  = QMachine(3)
        VQC_AmplitudeEmbedding(QTensor([3.2,-2,-2,0.3,12,0.1,2,-1]), q_machine=qm)
        print(qm.states)

        # [[[[ 0.2473717+0.j -0.1546073+0.j]
        #    [-0.1546073+0.j  0.0231911+0.j]]
        # 
        #   [[ 0.9276441+0.j  0.0077304+0.j]
        #    [ 0.1546073+0.j -0.0773037+0.j]]]]

VQC_IQPEmbedding
--------------------------

.. py:function:: pyvqnet.qnn.vqc.VQC_IQPEmbedding(input_feat, q_machine: pyvqnet.qnn.vqc.QMachine, rep: int = 1)

    Applica gate diagonali usando linee IQP per codificare :math:`n` caratteristiche in :math:`n` qubit di ``q_machine``.

    La codifica e stata proposta da `Havlicek et al. (2018) <https://arxiv.org/pdf/1804.11326.pdf>`_.

    Specificando ``rep``, le linee IQP di base possono essere ripetute.

    :param input_feat: Un array numpy che rappresenta i parametri.
    :param q_machine: Dispositivo macchina virtuale quantistica.
    :param rep: Numero di ripetizioni del blocco di circuito quantistico, il valore predefinito e 1.


    Example::

        from pyvqnet.qnn.vqc import VQC_IQPEmbedding, QMachine
        from pyvqnet.tensor import QTensor
        qm  = QMachine(3)
        VQC_IQPEmbedding(QTensor([3.2,-2,-2]), q_machine=qm)
        print(qm.states)        
        
        # [[[[ 0.0309356-0.3521973j  0.3256442+0.1376801j]
        #    [ 0.3256442+0.1376801j  0.2983474+0.1897071j]]
        # 
        #   [[ 0.0309356+0.3521973j -0.3170519-0.1564546j]
        #    [-0.3170519-0.1564546j -0.2310978-0.2675701j]]]]


VQC_RotCircuit
--------------------------

.. py:function:: pyvqnet.qnn.vqc.VQC_RotCircuit(q_machine, wire, params)

    Applica rotazioni arbitrarie a singolo qubit nei vettori di stato di ``q_machine``.

    .. math::

        R(\phi,\theta,\omega) = RZ(\omega)RY(\theta)RZ(\phi)= \begin{bmatrix}
        e^{-i(\phi+\omega)/2}\cos(\theta/2) & -e^{i(\phi-\omega)/2}\sin(\theta/2) \\
        e^{-i(\phi-\omega)/2}\sin(\theta/2) & e^{i(\phi+\omega)/2}\cos(\theta/2)
        \end{bmatrix}.


    :param q_machine: Dispositivo macchina virtuale quantistica.
    :param wire: Indice del qubit.
    :param params: Parametri :math:`[\phi, \theta, \omega]`.
    :return: Output QTensor.

    Example::

        from pyvqnet.qnn.vqc import VQC_RotCircuit, QMachine
        from pyvqnet.tensor import QTensor
        qm  = QMachine(3)
        VQC_RotCircuit(q_machine=qm, wire=[1],params=QTensor([2.0,1.5,2.1]))
        print(qm.states)

        # [[[[-0.3373617-0.6492732j  0.       +0.j       ]
        #    [ 0.6807868-0.0340677j  0.       +0.j       ]]
        # 
        #   [[ 0.       +0.j         0.       +0.j       ]
        #    [ 0.       +0.j         0.       +0.j       ]]]]

VQC_CRotCircuit
--------------------------

.. py:function:: pyvqnet.qnn.vqc.VQC_CRotCircuit(para,control_qubits,rot_wire,q_machine)

	Circuito Rot controllato.

    .. math:: CR(\phi, \theta, \omega) = \begin{bmatrix}
            1 & 0 & 0 & 0 \\
            0 & 1 & 0 & 0\\
            0 & 0 & e^{-i(\phi+\omega)/2}\cos(\theta/2) & -e^{i(\phi-\omega)/2}\sin(\theta/2)\\
            0 & 0 & e^{-i(\phi-\omega)/2}\sin(\theta/2) & e^{i(\phi+\omega)/2}\cos(\theta/2)
        \end{bmatrix}.
    
    :param para: array numpy che rappresenta i parametri.
    :param control_qubits: Indice dei bit di controllo.
    :param rot_wire: Indice del bit di rotazione.
    :param q_machine: Dispositivo macchina virtuale quantistica.
    :return: Output QTensor.

    Example::

        from pyvqnet.tensor import QTensor
        from pyvqnet.qnn.vqc.qcircuit import VQC_CRotCircuit
        from pyvqnet.qnn.vqc import QMachine, MeasureAll
        p = QTensor([2, 3, 4.0])
        qm = QMachine(2)
        VQC_CRotCircuit(p, 0, 1, qm)
        m = MeasureAll(obs={"Z0": 1})
        exp = m(q_machine=qm)
        print(exp)

        # [[0.9999999]]


VQC_Controlled_Hadamard
--------------------------

.. py:function:: pyvqnet.qnn.vqc.VQC_Controlled_Hadamard(wires, q_machine)

    Applica l'operazione Hadamard controllata in ``q_machine``.

    .. math:: CH = \begin{bmatrix}
            1 & 0 & 0 & 0 \\
            0 & 1 & 0 & 0 \\
            0 & 0 & \frac{1}{\sqrt{2}} & \frac{1}{\sqrt{2}} \\
            0 & 0 & \frac{1}{\sqrt{2}} & -\frac{1}{\sqrt{2}}
        \end{bmatrix}.

    :param wires: Indice del qubit, il primo e il bit di controllo, la lunghezza della lista e 2.
    :param q_machine: Dispositivo macchina virtuale quantistica.

    Examples::

        from pyvqnet.tensor import QTensor
        from pyvqnet.qnn.vqc.qcircuit import VQC_Controlled_Hadamard
        from pyvqnet.qnn.vqc import QMachine, MeasureAll
        p = QTensor([0.2, 3, 4.0])

        qm = QMachine(3)

        VQC_Controlled_Hadamard([1, 0], qm)
        m = MeasureAll(obs={"Z0": 1})
        exp = m(q_machine=qm)
        print(exp)

        # [[1.]]

VQC_CCZ
--------------------------

.. py:function:: pyvqnet.qnn.vqc.VQC_CCZ(wires, q_machine)

    Applica la logica Controlled-controlled-Z in ``q_machine``.

    .. math::

        CCZ =
        \begin{pmatrix}
        1 & 0 & 0 & 0 & 0 & 0 & 0 & 0\\
        0 & 1 & 0 & 0 & 0 & 0 & 0 & 0\\
        0 & 0 & 1 & 0 & 0 & 0 & 0 & 0\\
        0 & 0 & 0 & 1 & 0 & 0 & 0 & 0\\
        0 & 0 & 0 & 0 & 1 & 0 & 0 & 0\\
        0 & 0 & 0 & 0 & 0 & 1 & 0 & 0\\
        0 & 0 & 0 & 0 & 0 & 0 & 1 & 0\\
        0 & 0 & 0 & 0 & 0 & 0 & 0 & -1
        \end{pmatrix}
    
    :param wires: Lista di indici di qubit, il primo bit e il bit di controllo. La lunghezza della lista e 3.
    :param q_machine: Dispositivo macchina virtuale quantistica.

    :return:
            pyqpanda QCircuit 

    Example::

        from pyvqnet.tensor import QTensor
        from pyvqnet.qnn.vqc.qcircuit import VQC_CCZ
        from pyvqnet.qnn.vqc import QMachine, MeasureAll
        p = QTensor([0.2, 3, 4.0])

        qm = QMachine(3)

        VQC_CCZ([1, 0, 2], qm)
        m = MeasureAll(obs={"Z0": 1})
        exp = m(q_machine=qm)
        print(exp)

        # [[0.9999999]]


VQC_FermionicSingleExcitation
-----------------------------------------

.. py:function:: pyvqnet.qnn.vqc.VQC_FermionicSingleExcitation(weight, wires, q_machine)

    Un operatore coupled cluster a singola eccitazione per esponenziare il prodotto tensoriale di una matrice di Pauli. La forma matriciale e data da:

    .. math::

        \hat{U}_{pr}(\theta) = \mathrm{exp} \{ \theta_{pr} (\hat{c}_p^\dagger \hat{c}_r
        -\mathrm{H.c.}) \},

    :param weight: Il parametro sul qubit p ha un solo elemento.
    :param wires: Denota un sottoinsieme di indici di qubit nell'intervallo [r, p]. La lunghezza minima deve essere 2. Il primo indice viene interpretato come r e l'ultimo come p.
                 L'indice intermedio viene elaborato dal gate CNOT per calcolare la parita del insieme di qubit.
    :param q_machine: Dispositivo macchina virtuale quantistica.

    :return:
            pyqpanda QCircuit

    Examples::

        from pyvqnet.tensor import QTensor
        from pyvqnet.qnn.vqc.qcircuit import VQC_FermionicSingleExcitation
        from pyvqnet.qnn.vqc import QMachine, MeasureAll
        qm = QMachine(3)
        p0 = QTensor([0.5])

        VQC_FermionicSingleExcitation(p0, [1, 0, 2], qm)
        m = MeasureAll(obs={"Z0": 1})
        exp = m(q_machine=qm)
        print(exp)

        # [[0.9999998]]


VQC_FermionicDoubleExcitation
-----------------------------------------

.. py:function:: pyvqnet.qnn.vqc.VQC_FermionicDoubleExcitation(weight, wires1, wires2, q_machine)

    L'operatore coupled cluster a doppia eccitazione che esponenzia il prodotto tensoriale della matrice di Pauli. La forma matriciale e data da:

    .. math::

        \hat{U}_{pqrs}(\theta) = \mathrm{exp} \{ \theta (\hat{c}_p^\dagger \hat{c}_q^\dagger
        \hat{c}_r \hat{c}_s - \mathrm{H.c.}) \},

    dove :math:`\hat{c}` e :math:`\hat{c}^\dagger` sono gli operatori di annichilazione e creazione fermionici e gli indici :math:`r, s` e :math:`p, q` si riferiscono rispettivamente agli orbitali molecolari occupati e
    vuoti. Usando la `trasformazione di Jordan-Wigner <https://arxiv.org/abs/1208.5986>`_, l'operatore fermionico definito sopra puo essere scritto come
    matrice di Pauli (per maggiori dettagli, vedere `arXiv:1805.04340 <https://arxiv.org/abs/1805.04340>`_)

    .. math::

        \hat{U}_{pqrs}(\theta) = \mathrm{exp} \Big\{
        \frac{i\theta}{8} \bigotimes_{b=s+1}^{r-1} \hat{Z}_b \bigotimes_{a=q+1}^{p-1}
        \hat{Z}_a (\hat{X}_s \hat{X}_r \hat{Y}_q \hat{X}_p +
        \hat{Y}_s \hat{X}_r \hat{Y}_q \hat{Y}_p +\\ \hat{X}_s \hat{Y}_r \hat{Y}_q \hat{Y}_p +
        \hat{X}_s \hat{X}_r \hat{X}_q \hat{Y}_p - \mathrm{H.c.}  ) \Big\}

    :param weight: Parametro variabile.
    :param wires1: Lista di indici di qubit che rappresenta il sottoinsieme di qubit occupati nell'intervallo [s, r]. Il primo indice viene interpretato come s, l'ultimo come r.
     I gate CNOT operano sugli indici intermedi per calcolare la parita di un insieme di qubit.
    :param wires2: Lista di indici di qubit che rappresenta il sottoinsieme di qubit occupati nell'intervallo [q, p]. Il primo indice viene interpretato come q, l'ultimo come p.
     I gate CNOT operano sugli indici intermedi per calcolare la parita di un insieme di qubit.
    :param q_machine: Dispositivo macchina virtuale quantistica.

    :return:
        pyqpanda QCircuit

    Examples::

        from pyvqnet.tensor import QTensor
        from pyvqnet.qnn.vqc.qcircuit import VQC_FermionicDoubleExcitation
        from pyvqnet.qnn.vqc import QMachine, MeasureAll
        qm = QMachine(5)
        p0 = QTensor([0.5])

        VQC_FermionicDoubleExcitation(p0, [0, 1], [2, 3], qm)
        m = MeasureAll(obs={"Z0": 1})
        exp = m(q_machine=qm)
        print(exp)
        
        # [[0.9999998]]

VQC_UCCSD
-----------------------------------------

.. py:function:: pyvqnet.qnn.vqc.VQC_UCCSD(weights, wires, s_wires, d_wires, init_state, q_machine)

    Realizza il progetto unitary coupled cluster a singola e doppia eccitazione (UCCSD). UCCSD e il progetto VQE proposto, comunemente usato per eseguire simulazioni di chimica quantistica.

    Con l'approssimazione di Trotter del primo ordine, la funzione unitaria UCCSD e data da:

    .. math::

        \hat{U}(\vec{\theta}) =
        \prod_{p > r} \mathrm{exp} \Big\{\theta_{pr}
        (\hat{c}_p^\dagger \hat{c}_r-\mathrm{H.c.}) \Big\}
        \prod_{p > q > r > s} \mathrm{exp} \Big\{\theta_{pqrs}
        (\hat{c}_p^\dagger \hat{c}_q^\dagger \hat{c}_r \hat{c}_s-\mathrm{H.c.}) \Big\}

    dove :math:`\hat{c}` e :math:`\hat{c}^\dagger` sono gli operatori di annichilazione e
    creazione fermionici e gli indici :math:`r, s` e :math:`p, q` si riferiscono rispettivamente agli orbitali molecolari occupati e
    vuoti. (Per maggiori dettagli vedere `arXiv:1805.04340 <https://arxiv.org/abs/1805.04340>`_):


    :param weights: Un tensore ``(len(s_wires)+ len(d_wires))`` contenente i parametri
         :math:`\theta_{pr}` e :math:`\theta_{pqrs}` di rotazione Z in input
         ``FermionicSingleExcitation`` e ``FermionicDoubleExcitation``.
    :param wires: Indicizzazione dei qubit degli effetti template
    :param s_wires: Una sequenza di liste ``[r,...,p]`` contenente indici di qubit
         prodotta da una singola eccitazione
         :math:`\vert r, p \rangle = \hat{c}_p^\dagger \hat{c}_r \vert \mathrm{HF} \rangle`,
         dove :math:`\vert \mathrm{HF} \rangle` rappresenta lo stato di riferimento di Hartree-Fock.
    :param d_wires: sequenza di liste, ogni lista contenente due liste
         specifica gli indici ``[s, ...,r]`` e ``[q,...,p]``
         Definisce la doppia eccitazione: math:`\vert s, r, q, p \rangle = \hat{c}_p^\dagger \hat{c}_q^\dagger \hat{c}_r\hat{c}_s \ vert \mathrm{HF} \rangle`.
    :param init_state: vettore di rappresentazione del numero di occupazione di lunghezza ``len(wires)``
         stato ad alta frequenza. ``init_state`` e lo stato di inizializzazione del qubit.
    :param q_machine: Dispositivo macchina virtuale quantistica.

    Examples::

        from pyvqnet.qnn.vqc import VQC_UCCSD, QMachine, MeasureAll
        from pyvqnet.tensor import QTensor
        p0 = QTensor([2, 0.5, -0.2, 0.3, -2, 1, 3, 0])
        s_wires = [[0, 1, 2], [0, 1, 2, 3, 4], [1, 2, 3], [1, 2, 3, 4, 5]]
        d_wires = [[[0, 1], [2, 3]], [[0, 1], [2, 3, 4, 5]], [[0, 1], [3, 4]],
                [[0, 1], [4, 5]]]
        qm = QMachine(6)

        VQC_UCCSD(p0, range(6), s_wires, d_wires, QTensor([1.0, 1, 0, 0, 0, 0]), qm)
        m = MeasureAll(obs={"Z1": 1})
        exp = m(q_machine=qm)
        print(exp)

        # [[0.963802]]

VQC_ZFeatureMap
-----------------------------------------

.. py:function:: pyvqnet.qnn.vqc.VQC_ZFeatureMap(input_feat, q_machine: pyvqnet.qnn.vqc.QMachine, data_map_func=None, rep: int = 2)

    Circuito di evoluzione Z del primo ordine.

    For 3 quantum bits and 2 repetitions, the circuit is represented as:

    .. parsed-literal::

        ┌───┐┌──────────────┐┌───┐┌──────────────┐
        ┤ H ├┤ U1(2.0*x[0]) ├┤ H ├┤ U1(2.0*x[0]) ├
        ├───┤├──────────────┤├───┤├──────────────┤
        ┤ H ├┤ U1(2.0*x[1]) ├┤ H ├┤ U1(2.0*x[1]) ├
        ├───┤├──────────────┤├───┤├──────────────┤
        ┤ H ├┤ U1(2.0*x[2]) ├┤ H ├┤ U1(2.0*x[2]) ├
        └───┘└──────────────┘└───┘└──────────────┘
    
    La stringa di Pauli e fissata a ``Z``. Quindi, l'espansione del primo ordine sara un circuito senza gate di entanglement.

    :param input_feat: Un array che rappresenta i parametri di input.
    :param q_machine: Macchina quantistica.
    :param data_map_func: Matrice di mappatura dei parametri, progettata come ``data_map = lambda x: x``.
    :param rep: Numero di ripetizioni del modulo.
    
    Example::

        from pyvqnet.qnn.vqc import VQC_ZFeatureMap, QMachine, hadamard
        from pyvqnet.tensor import QTensor
        qm = QMachine(3)
        for i in range(3):
            hadamard(q_machine=qm, wires=[i])
        VQC_ZFeatureMap(input_feat=QTensor([[0.1,0.2,0.3]]),q_machine = qm)
        print(qm.states)
        
        # [[[[0.3535534+0.j        0.2918002+0.1996312j]
        #    [0.3256442+0.1376801j 0.1910257+0.2975049j]]
        # 
        #   [[0.3465058+0.0702402j 0.246323 +0.2536236j]
        #    [0.2918002+0.1996312j 0.1281128+0.3295255j]]]]

VQC_ZZFeatureMap
-----------------------------------------

.. py:function:: pyvqnet.qnn.vqc.VQC_ZZFeatureMap(input_feat, q_machine: pyvqnet.qnn.vqc.QMachine, data_map_func=None, entanglement: Union[str, List[List[int]],Callable[[int], List[int]]] = "full",rep: int = 2)

    Circuiti di evoluzione Pauli-Z del secondo ordine.

    For 3 quantum bits, 1 repetition and linear entanglement, the circuit is represented as:

    .. parsed-literal::

        ┌───┐┌─────────────────┐
        ┤ H ├┤ U1(2.0*φ(x[0])) ├──■────────────────────────────■────────────────────────────────────
        ├───┤├─────────────────┤┌─┴─┐┌──────────────────────┐┌─┴─┐
        ┤ H ├┤ U1(2.0*φ(x[1])) ├┤ X ├┤ U1(2.0*φ(x[0],x[1])) ├┤ X ├──■────────────────────────────■──
        ├───┤├─────────────────┤└───┘└──────────────────────┘└───┘┌─┴─┐┌──────────────────────┐┌─┴─┐
        ┤ H ├┤ U1(2.0*φ(x[2])) ├──────────────────────────────────┤ X ├┤ U1(2.0*φ(x[1],x[2])) ├┤ X ├
        └───┘└─────────────────┘                                  └───┘└──────────────────────┘└───┘
    
    dove ``φ`` e la funzione non lineare classica che per default e ``φ(x) = x`` se e ``φ(x,y) = (pi - x)(pi - y)``, progettata come:
    
    .. code-block::
        
        def data_map_func(x):
            coeff = x if x.shape[-1] == 1 else ft.reduce(lambda x, y: (np.pi - x) * (np.pi - y), x)
            return coeff

    :param input_feat: Un array che rappresenta i parametri di input.
    :param q_machine: Macchina quantistica.
    :param data_map_func: Matrice di mappatura dei parametri.
    :param entanglement: Struttura di entanglement specificata.
    :param rep: Numero di ripetizioni del modulo.
    
    Example::

        from pyvqnet.qnn.vqc import VQC_ZZFeatureMap, QMachine
        from pyvqnet.tensor import QTensor
        qm = QMachine(3)
        VQC_ZZFeatureMap(q_machine=qm, input_feat=QTensor([[0.1,0.2,0.3]]))
        print(qm.states)

        # [[[[-0.4234843-0.0480578j -0.144067 +0.1220178j]
        #    [-0.0800646+0.0484439j -0.5512857-0.2947832j]]
        # 
        #   [[ 0.0084012-0.0050071j -0.2593993-0.2717131j]
        #    [-0.1961917-0.3470543j  0.2786197+0.0732045j]]]]

VQC_AllSinglesDoubles
-----------------------------------------

.. py:function:: pyvqnet.qnn.vqc.VQC_AllSinglesDoubles(weights, q_machine: pyvqnet.qnn.vqc.QMachine, hf_state, wires, singles=None, doubles=None)

    Applica tutte le operazioni ``SingleExcitation`` e ``DoubleExcitation`` su ``q_machine`` allo stato iniziale di Hartree-Fock, preparando lo stato di associazione molecolare.

    :param weights: QTensor di dimensione ``(len(singles) + len(doubles),)`` contenente gli angoli che entrano nelle operazioni vqc.qCircuit.single_excitation e vqc.qCircuit.double_excitation in sequenza
    :param q_machine: Macchina quantistica.
    :param hf_state: Rappresenta la lunghezza dello stato di Hartree-Fock ``len(wires)`` Vettore del conteggio di occupazione, ``hf_state`` e usato per inizializzare i wire.
    :param wires: Qubit su cui agire.
    :param singles: Sequenza di liste con i due indici di bit quantistici su cui agisce l'operazione single_excitation.
    :param doubles: Sequenza di liste con i due indici di bit quantistici su cui agisce l'operazione double_excitation.

    Ad esempio, il circuito quantistico per il caso di due elettroni e sei bit quantistici e mostrato di seguito:
    
.. image:: ./images/all_singles_doubles.png
    :width: 600 px
    :align: center

|

    Example::

        from pyvqnet.qnn.vqc import VQC_AllSinglesDoubles, QMachine
        from pyvqnet.tensor import QTensor
        qubits = 4
        qm = QMachine(qubits)

        VQC_AllSinglesDoubles(q_machine=qm, weights=QTensor([0.55, 0.11, 0.53]), 
                              hf_state = QTensor([1,1,0,0]), singles=[[0, 2], [1, 3]], doubles=[[0, 1, 2, 3]], wires=[0,1,2,3])
        print(qm.states)
        
        # [ 0.        +0.j  0.        +0.j  0.        +0.j -0.23728043+0.j
        #   0.        +0.j  0.        +0.j -0.27552837+0.j  0.        +0.j
        #   0.        +0.j -0.12207296+0.j  0.        +0.j  0.        +0.j
        #   0.9235152 +0.j  0.        +0.j  0.        +0.j  0.        +0.j]


VQC_BasisRotation
-----------------------------------------

.. py:function:: pyvqnet.qnn.vqc.VQC_BasisRotation(q_machine: pyvqnet.qnn.vqc.QMachine, wires, unitary_matrix: QTensor, check=False)

    Implementa un circuito che fornisce un insieme utilizzabile per eseguire rotazioni di base monolitiche precise.

    :class:`~.vqc.qCircuit.VQC_BasisRotation` Esegue la seguente trasformata You determinata da fermioni a singola particella data in `arXiv:1711.04789 <https://arxiv.org/abs/1711.04789>`_\ :math:`U(u)`
    
    .. math::

        U(u) = \exp{\left( \sum_{pq} \left[\log u \right]_{pq} (a_p^\dagger a_q - a_q^\dagger a_p) \right)}.
    
    :math:`U(u)` utilizzando lo schema fornito nell'articolo `Optica, 3, 1460 (2016) <https://opg.optica.org/optica/fulltext.cfm?uri=optica-3-12-1460&id=355743>`_\.
    La decomposizione della matrice You di input e implementata efficientemente da una serie di gate :class:`~vqc.qCircuit.phaseshift` e :class:`~vqc.qCircuit.single_exitation`.
    

    :param q_machine: Dispositivo macchina virtuale quantistica.
    :param wires: Qubit su cui agire.
    :param unitary_matrix: Specifica la matrice della trasformazione di base.
    :param check: Verifica se `unitary_matrix` e una matrice You.

    Example::

        from pyvqnet.qnn.vqc import VQC_BasisRotation, QMachine, hadamard, isingzz
        from pyvqnet.tensor import QTensor
        import numpy as np
        V = np.array([[0.73678+0.27511j, -0.5095 +0.10704j, -0.06847+0.32515j],
                      [0.73678+0.27511j, -0.5095 +0.10704j, -0.06847+0.32515j],
                      [-0.21271+0.34938j, -0.38853+0.36497j,  0.61467-0.41317j]])

        eigen_vals, eigen_vecs = np.linalg.eigh(V)
        umat = eigen_vecs.T
        wires = range(len(umat))
        
        qm = QMachine(len(umat))

        for i in range(len(umat)):
            hadamard(q_machine=qm, wires=i)
        isingzz(q_machine=qm, params=QTensor([0.55]), wires=[0,2])
        VQC_BasisRotation(q_machine=qm, wires=wires,unitary_matrix=QTensor(umat,dtype=qm.state.dtype))
        
        print(qm.states)
        
        # [[[[ 0.3402686-0.0960063j  0.4140436-0.3069579j]
        #    [ 0.1206574+0.1982292j  0.5662895-0.0949503j]]
        # 
        #   [[-0.1715559-0.1614315j  0.1624039-0.0598041j]
        #    [ 0.0608986-0.1078906j -0.305845 +0.1773662j]]]]

VQC_QuantumPoolingCircuit
-----------------------------------------

.. py:function:: pyvqnet.qnn.vqc.VQC_QuantumPoolingCircuit(ignored_wires, sinks_wires, params, q_machine)

    Un circuito quantistico che esegue il downsampling dei dati.

    Per ridurre il numero di qubit in un circuito, le coppie di qubit vengono prima create nel sistema. Dopo aver inizialmente accoppiato tutti i qubit, una unitaria generalizzata a 2 qubit viene applicata a ciascuna coppia di qubit.
    E dopo aver applicato l'unitaria a due qubit, un qubit in ciascuna coppia viene ignorato nel resto della rete neurale.

    :param sources_wires: Indice del qubit sorgente che verra ignorato.
    :param sinks_wires: Indice del qubit di destinazione da mantenere.
    :param params: Parametri di input.
    :param q_machine: Dispositivo macchina virtuale quantistica.

    :return:
        pyqpanda QCircuit

    Examples:: 

        from pyvqnet.qnn.vqc import VQC_QuantumPoolingCircuit, QMachine, MeasureAll
        import pyqpanda as pq
        from pyvqnet import tensor
        machine = pq.CPUQVM()
        machine.init_qvm()
        qlists = machine.qAlloc_many(4)
        p = tensor.full([6], 0.35)
        qm = QMachine(4)
        VQC_QuantumPoolingCircuit(q_machine=qm,
                                ignored_wires=[0, 1],
                                sinks_wires=[2, 3],
                                params=p)
        m = MeasureAll(obs={"Z1": 1})
        exp = m(q_machine=qm)
        print(exp)



ExpressiveEntanglingAnsatz
---------------------------------------------------------------

.. py:class:: pyvqnet.qnn.vqc.ExpressiveEntanglingAnsatz(type: int, num_wires: int, depth: int, name: str = "")

    19 diversi ansatz tratti dall'articolo `Expressibility and entangling capability of parameterized quantum circuits for hybrid quantum-classical algorithms <https://arxiv.org/pdf/1905.10876.pdf>`_.

    :param type: Tipo di circuito da 1 a 19, per un totale di 19 tipi.
    :param num_wires: Numero di qubit.
    :param depth: Profondita del circuito.
    :param name: Nome, default "".

    :return:
        Un'istanza di ExpressiveEntanglingAnsatz

    Example::

        from pyvqnet.qnn.vqc  import *
        import pyvqnet
        pyvqnet.utils.set_random_seed(42)
        from pyvqnet.nn import Module
        class QModel(Module):
            def __init__(self, num_wires, dtype,grad_mode=""):
                super(QModel, self).__init__()

                self._num_wires = num_wires
                self._dtype = dtype
                self.qm = QMachine(num_wires, dtype=dtype,grad_mode=grad_mode,save_ir=True)
                self.c1 = ExpressiveEntanglingAnsatz(13,3,2)
                self.measure = MeasureAll(obs = {
                    "Z1":1
                })

            def forward(self, x, *args, **kwargs):
                self.qm.reset_states(x.shape[0])
                self.c1(q_machine = self.qm)
                rlt = self.measure(q_machine=self.qm)
                return rlt
            

        input_x = tensor.QTensor([[0.1, 0.2, 0.3]])


        input_x.requires_grad = True

        qunatum_model = QModel(num_wires=3, dtype=pyvqnet.kcomplex64)

        batch_y = qunatum_model(input_x)
        z = vqc_to_originir_list(qunatum_model)
        for zi in z:
            print(zi)
        batch_y.backward()
        print(batch_y)




vqc_qft_add_to_register
-------------------------------------

.. py:function:: pyvqnet.qnn.vqc.vqc_qft_add_to_register(q_machine, m, k)

    Codifica un intero senza segno `m` in un qubit e poi aggiunge `k` ad esso.

    .. math:: \text{Sum(k)}\vert m \rangle = \vert m + k \rangle.

    Questa operazione unitaria viene implementata come segue:

    (1). Convertire lo stato dalla base computazionale alla base di Fourier applicando la QFT allo stato :math:`\vert m \rangle`.

    (2). Usare il gate :math:`R_Z` per ruotare il :math:`j`-esimo qubit dell'angolo :math:`\frac{2k\pi}{2^{j}}`, ottenendo la nuova fase :math:`\frac{2(m + k)\pi}{2^{j}}`.

    (3). Applicare la QFT inversa per tornare alla base computazionale e ottenere :math:`m+k`.

    :param q_machine: La macchina quantistica da simulare.
    :param m: L'intero classico da incorporare nel registro.
    :param k: L'intero classico da aggiungere al registro.

    :retrun: Restituisce la rappresentazione binaria della somma target.

    .. note::

        Si noti che il numero di bit usati da ``q_machine`` deve essere sufficiente per codificare il valore binario della somma risultante usando lo stato base X.

    Example::

        import numpy as np
        from pyvqnet.qnn.vqc import QMachine,Samples, vqc_qft_add_to_register
        dev = QMachine(4)
        vqc_qft_add_to_register(dev,3, 7)
        ma = Samples()
        y = ma(q_machine=dev)
        print(y)
        #[[1,0,1,0]]


vqc_qft_add_two_register
-------------------------------------

.. py:function:: vqc_qft_add_two_register(q_machine, m, k, wires_m, wires_k, wires_solution)

    Somma gli interi senza segno codificati in due qubit.

    .. math:: \text{Sum}_2\vert m \rangle \vert k \rangle \vert 0 \rangle = \vert m \rangle \vert k \rangle \vert m+k \rangle

    In questo caso, possiamo interpretare il terzo registro (inizialmente a :math:`0`) come un contatore che contera le unita sommate di :math:`m` e :math:`k`. La fattorizzazione binaria rendera questo semplice. Se abbiamo :math:`\vert m \rangle = \vert \overline{q_0q_1q_2} \rangle`, allora se :math:`q_2 = 1`, dobbiamo aggiungere :math:`1` al contatore, altrimenti non aggiungere nulla. In generale, se l':math:`i`-esimo qubit e nello stato :math:`\vert 1 \rangle`, dobbiamo aggiungere :math:`2^{n-i-1}` unita, altrimenti aggiungere 0.

    :param q_machine: La macchina quantistica da simulare.
    :param m: L'intero classico incorporato nel registro come lhs.
    :param k: L'intero classico incorporato nel registro come rhs.
    :param wires_m: L'indice del qubit per codificare m.
    :param wires_k: L'indice del qubit per codificare k.
    :param wires_solution: L'indice del qubit per codificare la soluzione.

    :retrun: Restituisce la rappresentazione binaria della somma target.

    .. note::

        Il numero di bit usati in ``wires_m`` deve essere sufficiente per codificare il valore binario di `m` usando lo stato base X.
        ``wires_k`` usa abbastanza bit per codificare il valore binario di `k` usando lo stato base X.
        ``wires_solution`` usa abbastanza bit per codificare il valore binario del risultato usando lo stato base X.

    Example::

        import numpy as np
        from pyvqnet.qnn.vqc import QMachine,Samples, vqc_qft_add_two_register


        wires_m = [0, 1, 2]             # qubit necessari per codificare m
        wires_k = [3, 4, 5]             # qubit necessari per codificare k
        wires_solution = [6, 7, 8, 9]   # qubit necessari per codificare la soluzione
        dev = QMachine(len(wires_m) + len(wires_k) + len(wires_solution))

        vqc_qft_add_two_register(dev,3, 7, wires_m, wires_k, wires_solution)

        ma = Samples(wires=wires_solution)
        y = ma(q_machine=dev)
        print(y)


vqc_qft_mul
-------------------------------------

.. py:function:: vqc_qft_mul(q_machine, m, k, wires_m, wires_k, wires_solution)

    Moltiplica i valori codificati in due qubit.

    .. math:: \text{Mul}\vert m \rangle \vert k \rangle \vert 0 \rangle = \vert m \rangle \vert k \rangle \vert m\cdot k \rangle

    :param q_machine: La macchina quantistica da simulare.
    :param m: L'intero classico incorporato in un registro come lhs.
    :param k: L'intero classico incorporato in un registro come rhs.
    :param wires_m: L'indice del qubit per codificare m.
    :param wires_k: L'indice del qubit per codificare k.
    :param wires_solution: L'indice del qubit per codificare la soluzione.

    :retrun: Restituisce la rappresentazione binaria del prodotto target.

    .. note::

        ``wires_m`` deve usare abbastanza bit per codificare il valore binario di `m` usando lo stato base X.
        ``wires_k`` usa abbastanza bit per codificare il valore binario di `k` usando lo stato base X.
        ``wires_solution`` usa abbastanza bit per codificare il valore binario del risultato usando lo stato base X.

    Example::

        import numpy as np
        from pyvqnet.qnn.vqc import QMachine,Samples, vqc_qft_mul
        wires_m = [0, 1, 2]           # qubit necessari per codificare m
        wires_k = [3, 4, 5]           # qubit necessari per codificare k
        wires_solution = [6, 7, 8, 9, 10]  # qubit necessari per codificare la soluzione
        
        dev = QMachine(len(wires_m) + len(wires_k) + len(wires_solution))

        vqc_qft_mul(dev,3, 7, wires_m, wires_k, wires_solution)


        ma = Samples(wires=wires_solution)
        y = ma(q_machine=dev)
        print(y)
        #[[1,0,1,0,1]]

VQC_FABLE
--------------------

.. py:class:: pyvqnet.qnn.vqc.VQC_FABLE(wires)

    Costruisce un QCircuit basato su VQC utilizzando un metodo di codifica a blocchi approssimata veloce. Per matrici di determinate strutture [`arXiv:2205.00081 <https://arxiv.org/abs/2205.00081>`_], il metodo FABLE puo semplificare il circuito di codifica a blocchi senza perdere precisione.

    :param wires: L'indice qlist su cui agisce l'operatore.

    :return: Restituisce un'istanza della classe FABLE basata su VQC.

    Examples::

        from pyvqnet.qnn.vqc import VQC_FABLE
        from pyvqnet.qnn.vqc import QMachine
        from pyvqnet.dtype import float_dtype_to_complex_dtype
        import numpy as np
        from pyvqnet import QTensor
        
        A = QTensor(np.array([[0.1, 0.2 ], [0.3, 0.4 ]]) )
        qf = VQC_FABLE(list(range(3)))
        qm = QMachine(3,dtype=float_dtype_to_complex_dtype(A.dtype))
        qm.reset_states(1)
        z1 = qf(qm,A,0.001)
 
        """
        [[[[0.05     +0.j,0.15     +0.j],
        [0.05     +0.j,0.15     +0.j]],

        [[0.4974937+0.j,0.4769696+0.j],
        [0.4974937+0.j,0.4769696+0.j]]]]
        """


VQC_LCU
--------------------

.. py:class:: pyvqnet.qnn.vqc.VQC_LCU(wires)

    Costruisce un QCircuit basato su VQC usando Linear Combination Unit (LCU), `Hamiltonian Simulation via Qubitization <https://arxiv.org/abs/1610.06546>`_.
    Il dtype di input puo essere kfloat32, kfloat64, kcomplex64, kcomplex128
    L'input deve essere Hermitiano.

    :param wires: Indice qlist su cui agisce l'operatore, puo richiedere qubit ausiliari.
    :param check_hermitian: Verifica se l'input e Hermitiano, default: True.

    Examples::

        from pyvqnet.qnn.vqc import VQC_LCU
        from pyvqnet.qnn.vqc import QMachine
        from pyvqnet.dtype import float_dtype_to_complex_dtype,kfloat64

        from pyvqnet import QTensor

        A = QTensor([[0.25,0,0,0.75],[0,-0.25,0.75,0],[0,0.75,0.25,0],[0.75,0,0,-0.25]],device=1001,dtype=kfloat64)
        qf = VQC_LCU(list(range(3)))
        qm = QMachine(3,dtype=float_dtype_to_complex_dtype(A.dtype))
        qm.reset_states(2)
        z1 = qf(qm,A)
        print(z1)
        """
        [[[[ 0.25     +0.j, 0.       +0.j],
        [ 0.       +0.j, 0.75     +0.j]],

        [[-0.4330127+0.j, 0.       +0.j],
        [ 0.       +0.j, 0.4330127+0.j]]],


        [[[ 0.25     +0.j, 0.       +0.j],
        [ 0.       +0.j, 0.75     +0.j]],

        [[-0.4330127+0.j, 0.       +0.j],
        [ 0.       +0.j, 0.4330127+0.j]]]]
        <QTensor [2, 2, 2, 2] DEV_CPU kcomplex128>
        """


VQC_QSVT
--------------------

.. py:class:: pyvqnet.qnn.vqc.VQC_QSVT(A, angles, wires)

    Implementa il circuito di
    `trasformazione dei valori singolari quantistici <https://arxiv.org/abs/1806.01838>`__ (QSVT).

    :param A: La matrice generale :math:`(n \times m)` da codificare.
    :param angles: La lista di angoli da spostare per ottenere il polinomio desiderato.
    :param wires: Gli indici dei qubit su cui agisce A.

    Example::

        from pyvqnet import DEV_GPU
        from pyvqnet.qnn.vqc import QMachine,VQC_QSVT
        from pyvqnet.dtype import float_dtype_to_complex_dtype,kfloat64
        import numpy as np
        from pyvqnet import QTensor

        A = QTensor([[0.1, 0.2], [0.3, 0.4]])
        angles = QTensor([0.1, 0.2, 0.3])
        qm = QMachine(4,dtype=float_dtype_to_complex_dtype(A.dtype))
        qm.reset_states(1)
        qf = VQC_QSVT(A,angles,wires=[2,1,3])
        z1 = qf(qm)
        print(z1)
        """
        [[[[[ 0.9645935+0.2352667j,-0.0216623+0.0512362j],
        [-0.0062613+0.0308878j,-0.0199871+0.0985996j]],

        [[ 0.       +0.j       , 0.       +0.j       ],
            [ 0.       +0.j       , 0.       +0.j       ]]],


        [[[ 0.       +0.j       , 0.       +0.j       ],
            [ 0.       +0.j       , 0.       +0.j       ]],

        [[ 0.       +0.j       , 0.       +0.j       ],
            [ 0.       +0.j       , 0.       +0.j       ]]]]]
        """

Misure Quantistiche
=============================================

VQC_Purity
----------------------

.. py:function:: pyvqnet.qnn.vqc.VQC_Purity(state, qubits_idx, num_wires)

    Calcola la purezza su un particolare qubit ``qubits_idx`` dal vettore di stato ``state``.

    .. math::
        \gamma = \text{Tr}(\rho^2)

    dove :math:`\rho` e una matrice densita. La purezza di uno stato quantistico normalizzato soddisfa :math:`\frac{1}{d} \leq \gamma \leq 1` ,
    dove :math:`d` e la dimensione dello spazio di Hilbert.
    La purezza dello stato puro e 1.

    :param state: Stato quantistico ottenuto da pyqpanda get_qstate()
    :param qubits_idx: Indice del qubit per cui calcolare la purezza
    :param num_wires: Indice del qubit

    :return: purezza

    Example::

        from pyvqnet.qnn.vqc import VQC_Purity, rx, ry, cnot, QMachine
        from pyvqnet.tensor import QTensor
        from pyvqnet import kfloat64
        x = QTensor([[0.7, 0.4], [1.7, 2.4]], requires_grad=True)
        qm = QMachine(3)
        qm.reset_states(2)
        rx(q_machine=qm, wires=0, params=x[:, [0]])
        ry(q_machine=qm, wires=1, params=x[:, [1]])
        ry(q_machine=qm, wires=2, params=x[:, [1]])
        cnot(q_machine=qm, wires=[0, 1])
        cnot(q_machine=qm, wires=[2, 1])
        y = VQC_Purity(qm.states, [0, 1], num_wires=3)
        y.backward()
        print(y)

        # [0.9356751 0.875957]

VQC_VarMeasure
-------------------------------------

.. py:function:: pyvqnet.qnn.vqc.VQC_VarMeasure(q_machine, obs)

    Restituisce la varianza di misura dell'osservabile fornito ``obs`` nei vettori di stato in ``q_machine``.

    :param q_machine: Stato quantistico ottenuto da pyqpanda get_qstate()
    :param obs: osservabili

    :return: valore di varianza

    Example::

        from pyvqnet.tensor import QTensor
        from pyvqnet.qnn.vqc import VQC_VarMeasure, rx, cnot, hadamard, QMachine,PauliY
        x = QTensor([[0.5]], requires_grad=True)
        qm = QMachine(3)
        rx(q_machine=qm, wires=0, params=x)
        var_result = VQC_VarMeasure(q_machine= qm, obs=PauliY(wires=0))
        var_result.backward()
        print(var_result)

        # [[0.7701511]]

VQC_DensityMatrixFromQstate
-------------------------------------

.. py:function:: pyvqnet.qnn.vqc.VQC_DensityMatrixFromQstate(state, indices)

    Calcola la matrice densita degli stati quantistici ``state`` su un insieme specifico di qubit ``indices``.

    :param state: Una lista 1D di vettori di stato. La dimensione di questa lista deve essere ``(2**N,)`` Per il numero di qubit ``N``, qstate deve iniziare da 000 -> 111.
    :param indices: Una lista di indici di qubit nel sottosistema considerato.

    :return: Una matrice densita di dimensione "(b, 2**len(indices), 2**len(indices))".

    Example::

        from pyvqnet.qnn.vqc import VQC_DensityMatrixFromQstate,rx,ry,cnot,QMachine
        from pyvqnet.tensor import QTensor
        from pyvqnet import kfloat64
        x = QTensor([[0.7,0.4],[1.7,2.4]],requires_grad=True)

        qm = QMachine(3)
        qm.reset_states(2)
        rx(q_machine=qm,wires=0,params=x[:,[0]])
        ry(q_machine=qm,wires=1,params=x[:,[1]])
        ry(q_machine=qm,wires=2,params=x[:,[1]])
        cnot(q_machine=qm,wires=[0,1])
        cnot(q_machine=qm,wires=[2, 1])
        y = VQC_DensityMatrixFromQstate(qm.states,[0,1])
        print(y)

        # [[[0.8155131+0.j        0.1718155+0.j        0.       +0.0627175j
        #   0.       +0.2976855j]
        #  [0.1718155+0.j        0.0669081+0.j        0.       +0.0244234j
        #   0.       +0.0627175j]
        #  [0.       -0.0627175j 0.       -0.0244234j 0.0089152+0.j
        #   0.0228937+0.j       ]
        #  [0.       -0.2976855j 0.       -0.0627175j 0.0228937+0.j
        #   0.1086637+0.j       ]]
        # 
        # [[0.3362115+0.j        0.1471083+0.j        0.       +0.1674582j
        #   0.       +0.3827205j]
        #  [0.1471083+0.j        0.0993662+0.j        0.       +0.1131119j
        #   0.       +0.1674582j]
        #  [0.       -0.1674582j 0.       -0.1131119j 0.1287589+0.j
        #   0.1906232+0.j       ]
        #  [0.       -0.3827205j 0.       -0.1674582j 0.1906232+0.j
        #   0.4356633+0.j       ]]]   


Probability
--------------------

.. py:class:: pyvqnet.qnn.vqc.Probability(wires, name="")

    Calcola le misure di probabilita dei circuiti quantistici su bit specifici

    :param wires: Indice del qubit da misurare.
    :param name: nome del modulo

    .. py:method:: forward(q_machine)

        Esegue il calcolo della misura di probabilita

        :param q_machine: simulatore di vettori di stato quantistici
        :return: risultati della misura di probabilita

    .. note::

        I risultati della misura di probabilita calcolati usando questa classe sono generalmente [b, len(wires)], dove b e il numero di batch di q_machine.reset_states(b).

 

    Example::

        from pyvqnet.qnn.vqc import Probability,rx,ry,cnot,QMachine,rz
        from pyvqnet.tensor import QTensor
        from pyvqnet import kfloat64
        x = QTensor([[0.56, 0.1],[0.56, 0.1]],requires_grad=True)
        qm = QMachine(4)
        qm.reset_states(2)
        rz(q_machine=qm,wires=0,params=x[:,[0]])
        rz(q_machine=qm,wires=1,params=x[:,[0]])
        cnot(q_machine=qm,wires=[0,1])
        ry(q_machine=qm,wires=2,params=x[:,[1]])
        cnot(q_machine=qm,wires=[0,2])
        rz(q_machine=qm,wires=3,params=x[:,[1]])
        ma = Probability(wires=1)
        y =ma(q_machine=qm)

        # [[1.0000002 0.       ]
        #  [1.0000002 0.       ]]        

MeasureAll
--------------------

.. py:class:: pyvqnet.qnn.vqc.MeasureAll(obs,name="")

    Calcola i risultati di misura di un circuito quantistico. Supporta l'input di osservabili ``obs``. Questo puo essere in formato dizionario, che rappresenta un osservabile composto da piu operatori di Pauli, o in formato lista, che rappresenta una lista di osservabili con piu valori attesi.
    Ad esempio:

        {\'X0\': 0.23} indica un effetto PauliX sul qubit 0, con un coefficiente di 0.23.

        {\'X1 Z2\': 2.4,\'Y2\': -0.5} corrisponde al valore osservato 2.4 * X1 @ Z2 - 0.5 * Y2.

        [{\'X1 Z2\': 4,\'Z1 Z0\': 3},{\'X1 Y2 Z0\': 3.5}] corrisponde ai due valori osservati 4 * X1 @ Z2 + 3 * Z1 @ Z0 e 3.5 * X1 @ Y2 @ Z0.

    :param obs: osservabili, stringa del dizionario degli operatori di Pauli.

    .. note::

        Se ``obs`` e una lista, il risultato della misura calcolato usando questa classe e generalmente [b, lunghezza lista obs], dove b e il numero di batch di q_machine.reset_states(b).

        Se ``obs`` e un dizionario, il risultato della misura calcolato usando questa classe e generalmente [b,1], dove b e il numero di batch di q_machine.reset_states(b).

    .. py:method:: forward(q_machine)

        Esegue l'operazione di misura

        :param q_machine: simulatore di vettori di stato quantistici
        :return: risultato della misura, QTensor.



    Example::

        from pyvqnet.qnn.vqc import MeasureAll,rx,ry,cnot,QMachine,rz
        from pyvqnet.tensor import QTensor
        from pyvqnet import kfloat64
        x = QTensor([[0.56, 0.1],[0.56, 0.1]],requires_grad=True)
        qm = QMachine(4)
        qm.reset_states(2)
        rz(q_machine=qm,wires=0,params=x[:,[0]])
        rz(q_machine=qm,wires=1,params=x[:,[0]])
        cnot(q_machine=qm,wires=[0,1])
        ry(q_machine=qm,wires=2,params=x[:,[1]])
        cnot(q_machine=qm,wires=[0,2])
        rz(q_machine=qm,wires=3,params=x[:,[1]])
        obs_list = [{
            "Z0 X2":1
        }, {
            "Z0 Y2":1
        }]
        ma = MeasureAll(obs=obs_list)
        y =ma(q_machine=qm)
        print(y)
 

Samples
----------------------------

.. py:class:: pyvqnet.qnn.vqc.Samples(wires=None, obs=None, shots = 1,name="")
    
    Ottiene il risultato dell'osservazione ``obs`` con ``shots`` sui wire specificati ``wires``.

    .. py:method:: forward(q_machine)

        Esegue operazioni di campionamento.

        :param q_machine: Il simulatore di vettori di stato quantistici in uso
        :return: Risultati della misura, QTensor.

    .. note::

        I risultati della misura calcolati usando questa classe sono generalmente [b, shots, len(wires)], dove b e il numero di batch di q_machine.reset_states(b).

    :param wires: Indice del qubit da campionare. Valore predefinito: None, usa tutti i bit del simulatore al runtime.
    :param obs: Questo valore puo essere solo None.
    :param shots: Numero di ripetizioni del campionamento, valore predefinito: 1.
    :param name: Nome di questo modulo, valore predefinito: "".
    :return: Una classe di metodo di misura

    Example::

        from pyvqnet.qnn.vqc import Samples,rx,ry,cnot,QMachine,rz
        from pyvqnet.tensor import QTensor
        from pyvqnet import kfloat64
        x = QTensor([[0.56, 0.1],[0.56, 0.1]],requires_grad=True)

        qm = QMachine(4)
        qm.reset_states(2)
        rz(q_machine=qm,wires=0,params=x[:,[0]])
        rx(q_machine=qm,wires=1,params=x[:,[0]])
        cnot(q_machine=qm,wires=[0,1])

        cnot(q_machine=qm,wires=[0,2])
        ry(q_machine=qm,wires=3,params=x[:,[1]])


        ma = Samples(wires=[0,1,2],shots=3)
        y = ma(q_machine=qm)
        print(y)
        """
        [[[0,0,0],
        [0,1,0],
        [0,0,0]],

        [[0,1,0],
        [0,0,0],
        [0,1,0]]]
        """




HermitianExpval
---------------------------------------------------------------

.. py:class:: pyvqnet.qnn.vqc.HermitianExpval(obs, name="")

    Calcola il valore atteso di un osservabile Hermitiano ``obs`` di un circuito quantistico.
    
    :param obs: Quantita Hermitiana.
    :param name: Il nome del modulo, default: "".
    :return: Un'istanza di HermitianExpval.

    .. py:method:: forward(q_machine)

        Esegue la misura Hermitiana.

        :param q_machine: simulatore di vettori di stato quantistici
        :return: risultato della misura, QTensor.

    .. note::

        Il risultato della misura calcolato usando questa classe e generalmente [b,1], dove b e il numero di batch di q_machine.reset_states(b).

    Example::

        from pyvqnet.qnn.vqc import qcircuit
        from pyvqnet.qnn.vqc import QMachine, RX, RY, CNOT, PauliX, qmatrix, PauliZ, VQC_RotCircuit,HermitianExpval
        from pyvqnet.tensor import QTensor, tensor
        import pyvqnet
        from pyvqnet.nn import Parameter
        import numpy as np
        bsz = 3
        H = np.array([[8, 4, 0, -6], [4, 0, 4, 0], [0, 4, 8, 0], [-6, 0, 0, 0]])
        class QModel(pyvqnet.nn.Module):
            def __init__(self, num_wires, dtype):
                super(QModel, self).__init__()
                self.rot_param = Parameter((3, ))
                self.rot_param.copy_value_from(tensor.QTensor([-0.5, 1, 2.3]))
                self._num_wires = num_wires
                self._dtype = dtype
                self.qm = QMachine(num_wires, dtype=dtype)
                self.rx_layer1 = VQC_RotCircuit
                self.ry_layer2 = RY(has_params=True,
                                    trainable=True,
                                    wires=0,
                                    init_params=tensor.QTensor([-0.5]))
                self.xlayer = PauliX(wires=0)
                self.cnot = CNOT(wires=[0, 1])
                self.measure = HermitianExpval(obs = {'wires':(1,0),'observables':tensor.to_tensor(H)})

            def forward(self, x, *args, **kwargs):
                self.qm.reset_states(x.shape[0])

                qcircuit.rx(q_machine=self.qm, wires=0, params=x[:, [1]])
                qcircuit.ry(q_machine=self.qm, wires=1, params=x[:, [0]])
                self.xlayer(q_machine=self.qm)
                self.rx_layer1(params=self.rot_param, wire=1, q_machine=self.qm)
                self.ry_layer2(q_machine=self.qm)
                self.cnot(q_machine=self.qm)
                rlt = self.measure(q_machine = self.qm)

                return rlt


        input_x = tensor.arange(1, bsz * 2 + 1,
                                dtype=pyvqnet.kfloat32).reshape([bsz, 2])
        input_x.requires_grad = True

        qunatum_model = QModel(num_wires=2, dtype=pyvqnet.kcomplex64)

        batch_y = qunatum_model(input_x)
        batch_y.backward()

        print(batch_y)


        # [[5.3798223],
        #  [7.1294155],
        #  [0.7028297]]


Template di circuiti variazionali quantistici di uso comune
===========================================================

VQC_HardwareEfficientAnsatz
-----------------------------------------

.. py:class:: pyvqnet.qnn.vqc.VQC_HardwareEfficientAnsatz(n_qubits,single_rot_gate_list,entangle_gate="CNOT",entangle_rules='linear',depth=1,initial=None,dtype=None)

    Implementazione dell'Hardware Efficient Ansatz introdotto nell'articolo: `Hardware-efficient Variational Quantum Eigensolver for Small Molecules <https://arxiv.org/pdf/1704.05018.pdf>`__.

    :param n_qubits: Numero di qubit.
    :param single_rot_gate_list: Una lista di gate di rotazione a singolo qubit costruita da uno o piu gate di rotazione che agiscono su ogni qubit. Attualmente supporta Rx, Ry, Rz.
    :param entangle_gate: Il gate di entanglement non parametrizzato. Sono supportati CNOT, CZ. Default: CNOT.
    :param entangle_rules: Come il gate di entanglement viene usato nel circuito. 'linear' significa che il gate di entanglement agira su ogni coppia di qubit vicini. 'all' significa che il gate di entanglement agira su qualsiasi coppia di qubit. Default: linear.
    :param depth: La profondita dell'ansatz, default: 1.
    :param initial: Valore iniziale uguale per tutti i parametri, default: None, questo modulo inizializzera i parametri casualmente.
    :param dtype: Tipo di dato dei parametri.
    :return: Un'istanza di VQC_HardwareEfficientAnsatz.

    Example::

        from pyvqnet.nn import Module,Linear,ModuleList
        from pyvqnet.qnn.vqc.qcircuit import VQC_HardwareEfficientAnsatz,RZZ,RZ
        from pyvqnet.qnn.vqc import Probability,QMachine
        from pyvqnet import tensor

        class QM(Module):
            def __init__(self, name=""):
                super().__init__(name)
                self.linearx = Linear(4,2)
                self.ansatz = VQC_HardwareEfficientAnsatz(4, ["rx", "RY", "rz"],
                                            entangle_gate="cnot",
                                            entangle_rules="linear",
                                            depth=2)
                self.encode1 = RZ(wires=0)
                self.encode2 = RZ(wires=1)
                self.measure = Probability(wires=[0,2])
                self.device = QMachine(4)
            def forward(self, x, *args, **kwargs):
                self.device.reset_states(x.shape[0])
                y = self.linearx(x)
                self.encode1(params = y[:, [0]],q_machine = self.device,)
                self.encode2(params = y[:, [1]],q_machine = self.device,)
                self.ansatz(q_machine =self.device)
                return self.measure(q_machine =self.device)

        bz =3
        inputx = tensor.arange(1.0,bz*4+1).reshape([bz,4])
        inputx.requires_grad= True
        qlayer = QM()
        y = qlayer(inputx)
        y.backward()
        print(y)
        # [[0.3075959 0.2315064 0.2491432 0.2117545]
        #  [0.3075958 0.2315062 0.2491433 0.2117546]
        #  [0.3075958 0.2315062 0.2491432 0.2117545]]

VQC_BasicEntanglerTemplate
-------------------------------

.. py:class:: pyvqnet.qnn.vqc.VQC_BasicEntanglerTemplate(num_layer=1, num_qubits=1, rotation="RX", initial=None, dtype=None)

    Un layer costituito da una rotazione a singolo qubit con un singolo parametro su ogni qubit, seguita da una catena chiusa o combinazione ad anello di piu gate CNOT.

    Un anello di gate CNOT collega ogni qubit ai suoi vicini, con l'ultimo qubit considerato vicino del primo qubit.

    :param num_layers: numero di layer di ripetizione, default: 1.
    :param num_qubits: numero di qubit, default: 1.
    :param rotation: gate a singolo qubit con un parametro da usare, default: `RX`
    :param initial: valore iniziale uguale per tutti i parametri. default: None, i parametri verranno inizializzati casualmente.
    :param dtype: tipo di dato del parametro, default: None, usa float32.
    :return: Un'istanza di VQC_BasicEntanglerTemplate

    Example::

        from pyvqnet.nn import Module, Linear, ModuleList
        from pyvqnet.qnn.vqc.qcircuit import VQC_BasicEntanglerTemplate, RZZ, RZ
        from pyvqnet.qnn.vqc import Probability, QMachine
        from pyvqnet import tensor


        class QM(Module):
            def __init__(self, name=""):
                super().__init__(name)

                self.ansatz = VQC_BasicEntanglerTemplate(2,
                                                    4,
                                                    "rz",
                                                    initial=tensor.ones([1, 1]))

                self.measure = Probability(wires=[0, 2])
                self.device = QMachine(4)

            def forward(self,x, *args, **kwargs):

                self.ansatz(q_machine=self.device)
                return self.measure(q_machine=self.device)

        bz = 1
        inputx = tensor.arange(1.0, bz * 4 + 1).reshape([bz, 4])
        qlayer = QM()
        y = qlayer(inputx)
        y.backward()
        print(y)

        # [[1.0000002 0.        0.        0.       ]]


VQC_StronglyEntanglingTemplate
------------------------------------

.. py:class:: pyvqnet.qnn.vqc.VQC_StronglyEntanglingTemplate(num_layers=1, num_qubits=1, ranges=None,initial=None, dtype=None)

    Un layer costituito da una rotazione a singolo qubit e un entangler, vedere `circuit-centric classifier design <https://arxiv.org/abs/1804.00633>`__.

    :param num_layers: numero di layer di ripetizione, default: 1.
    :param num_qubits: numero di qubit, default: 1.
    :param ranges: sequenza che determina l'iperparametro di intervallo per ogni layer successivo; default: None
                                usando :math: `r=l \mod M` per l':math:`l`-esimo layer e :math:`M` qubit.
    :param initial: valore iniziale per tutti i parametri. default: None, inizializzato casualmente.
    :param dtype: tipo di dato del parametro, default: None, usa float32.
    :return: Un'istanza di VQC_StronglyEntanglingTemplate.

    Example::

        from pyvqnet.nn import Module
        from pyvqnet.qnn.vqc.qcircuit import VQC_StronglyEntanglingTemplate
        from pyvqnet.qnn.vqc import Probability, QMachine
        from pyvqnet import tensor


        class QM(Module):
            def __init__(self, name=""):
                super().__init__(name)

                self.ansatz = VQC_StronglyEntanglingTemplate(2,
                                                    4,
                                                    None,
                                                    initial=tensor.ones([1, 1]))

                self.measure = Probability(wires=[0, 1])
                self.device = QMachine(4)

            def forward(self,x, *args, **kwargs):

                self.ansatz(q_machine=self.device)
                return self.measure(q_machine=self.device)

        bz = 1
        inputx = tensor.arange(1.0, bz * 4 + 1).reshape([bz, 4])
        qlayer = QM()
        y = qlayer(inputx)
        y.backward()
        print(y)

        # [[0.3745951 0.154298  0.059156  0.4119509]]


VQC_QuantumEmbedding
--------------------------

.. py:class:: pyvqnet.qnn.vqc.VQC_QuantumEmbedding( num_repetitions_input, depth_input, num_unitary_layers, num_repetitions,initial = None,dtype = None,name= "")

    Usa RZ, RY, RZ per creare circuiti quantistici variazionali che codificano dati classici in stati quantistici.
    Riferimento `Quantum embeddings for machine learning <https://arxiv.org/abs/2001.03622>`_.
    Dopo che la classe e inizializzata, la sua funzione membro ``compute_circuit`` e una funzione eseguibile, che puo essere inserita come parametro.
    La classe ``QuantumLayerV2`` costituisce un layer del modello di machine learning quantistico.

    :param num_repetitions_input: numero di ripetizioni per codificare l'input in un sottomodulo.
    :param depth_input: numero di dimensioni di input.
    :param num_unitary_layers: numero di ripetizioni dei gate quantistici variazionali.
    :param num_repetitions: numero di ripetizioni del sottomodulo.
    :param initial: valore di inizializzazione dei parametri, default None
    :param dtype: tipo di parametro, default None, usa float32.
    :param name: nome della classe
    :return: Un'istanza di VQC_QuantumEmbedding.

    Example::

        from pyvqnet.nn import Module
        from pyvqnet.qnn.vqc.qcircuit import VQC_QuantumEmbedding
        from pyvqnet.qnn.vqc import  QMachine,MeasureAll
        from pyvqnet import tensor
        import pyvqnet
        depth_input = 2
        num_repetitions = 2
        num_repetitions_input = 2
        num_unitary_layers = 2
        nq = depth_input * num_repetitions_input
        bz = 12

        class QM(Module):
            def __init__(self, name=""):
                super().__init__(name)

                self.ansatz = VQC_QuantumEmbedding(num_repetitions_input, depth_input,
                                                num_unitary_layers,
                                                num_repetitions, initial=tensor.full([1],12.0),dtype=pyvqnet.kfloat64)
                self.measure = MeasureAll(obs={f"Z{nq-1}":1})
                self.device = QMachine(nq,dtype=pyvqnet.kcomplex128)

            def forward(self, x, *args, **kwargs):
                self.device.reset_states(x.shape[0])
                self.ansatz(x,q_machine=self.device)
                return self.measure(q_machine=self.device)

        inputx = tensor.arange(1.0, bz * depth_input + 1,
                                dtype=pyvqnet.kfloat64).reshape([bz, depth_input])
        qlayer = QM()
        y = qlayer(inputx)
        y.backward()
        print(y)
        # [[-0.2539548]
        #  [-0.1604787]
        #  [ 0.1492931]
        #  [-0.1711956]
        #  [-0.1577133]
        #  [ 0.1396999]
        #  [ 0.016864 ]
        #  [-0.0893069]
        #  [ 0.1897014]
        #  [ 0.0941301]
        #  [ 0.0550722]
        #  [ 0.2408579]]


Interfaccia del modello di machine learning quantistico
=======================================================

Quanvolution
---------------------------------------------------------------

.. py:class:: pyvqnet.qnn.qcnn.Quanvolution(params_shape, stride=(1, 1), kernel_initializer=quantum_uniform, machine_type_or_cloud_token: str = "cpu")

    Basato sulla convoluzione quantistica implementata in "Quanvolutional Neural Networks: Powering Image Recognition with Quantum Circuits" (https://arxiv.org/abs/1904.04767), il filtro di convoluzione classico viene sostituito da un circuito quantistico variazionale per ottenere una rete neurale convoluzionale quantistica con un filtro di convoluzione quantistico.

    :param params_shape: La forma dei parametri, che dovrebbe essere bidimensionale.
    :param stride: La dimensione del passo della finestra di scorrimento, default (1,1).
    :param kernel_initializer: Parametri dell'inizializzatore del kernel di convoluzione.
    :param machine_type_or_cloud_token: Stringa del tipo di macchina o token Qcloud, default "cpu".
    :return: Un'istanza di Quanvolution.

    Examples::

        from pyvqnet.qnn.qcnn import Quanvolution
        import pyvqnet.tensor as tensor
        qlayer = Quanvolution([4,2],(3,3))

        x = tensor.arange(1,25*25*3+1).reshape([3,1,25,25])

        y = qlayer(x)

        print(y.shape)

        y.backward()

        print(qlayer.m_para)
        print(qlayer.m_para.grad)
        #[3, 4, 8, 8]

        #[4.0270405,4.3587413,2.4935627,2.8155506,0.3314773,0.8889271,3.7357519, 0.9196261]
        #<Parameter [8] DEV_CPU kfloat32>

        #[ -0.2364242, -0.6942478, -8.445061 , -0.0558891, -0.       ,-49.498577 ,40.339344 , 40.339344 ]
        #<QTensor [8] DEV_CPU kfloat32>

QDRL
---------------------------------------------------------------

.. py:class:: pyvqnet.qnn.qdrl_vqc.QDRL(nq)

    L'algoritmo Quantum Data Re-uploading (QDRL) basato su "Data re-uploading for a universal quantum classifier" (https://arxiv.org/abs/1907.02085) e un modello di ricarica dati quantistici che combina circuiti quantistici con reti neurali classiche.

    :param nq: Il numero di bit quantistici (qubit) usati nel circuito quantistico. Questo determina la scala del sistema quantistico che il modello gestira.
    :return: Un'istanza di QDRL.

    Example::

        import numpy as np
        from pyvqnet.dtype import kfloat32
        from pyvqnet.qnn.qdrl_vqc import QDRL
        import pyvqnet.tensor as tensor

        # Imposta il numero di bit quantistici (qubit)
        nq = 1

        # Inizializza il modello
        model = QDRL(nq)

        # Create an example input (assume the input is a (batch_size, 3) shaped data)
        # Supponiamo di avere un batch_size di 4 e ogni input ha 3 caratteristiche
        x_input = tensor.QTensor(np.random.randn(4, 3), dtype=kfloat32)

        # Passa l'input attraverso il modello
        output = model(x_input)

        output.backward()

        # Produce il risultato
        print("Model output:")
        print(output)


QGRU
------------------------------------------------------------

.. py:class:: pyvqnet.qnn.qgru.QGRU(para_num, num_of_qubits,input_size,hidden_size,batch_first=True)

    GRU (Gated Recurrent Unit) basato su circuiti variazionali quantistici, che utilizza circuiti quantistici per gli aggiornamenti di stato e la memorizzazione.

    :param para_num: Il numero di parametri nel circuito quantistico.
    :param num_of_qubits: Il numero di qubit.
    :param input_size: La dimensione delle caratteristiche dei dati di input.
    :param hidden_size: La dimensione dell'unita nascosta.
    :param batch_first: Se la prima dimensione dell'input e la dimensione del batch.
    :return: Un'istanza di QGRU.

    Example::

        import numpy as np
        from pyvqnet.tensor import tensor
        from pyvqnet.qnn.qgru import QGRU
        from pyvqnet.dtype import kfloat32
        # Esempio di utilizzo
 
        # Imposta i parametri
        para_num = 8
        num_of_qubits = 8
        input_size = 4
        hidden_size = 4
        batch_size = 1
        seq_length = 1
        # Crea il modello QGRU
        qgru = QGRU(para_num, num_of_qubits, input_size, hidden_size, batch_first=True)

        # Crea i dati di input
        x = tensor.QTensor(np.random.randn(batch_size, seq_length, input_size), dtype=kfloat32)

        # Chiama il modello
        output, h_t = qgru(x)
        output.backward()

        print("Output shape:", output.shape)  # Forma dell'output
        print("h_t shape:", h_t.shape)  # Forma dello stato nascosto finale

QLSTM
-------------------------------------------------------------

.. py:class:: pyvqnet.qnn.qlstm.QLSTM(para_num, num_of_qubits,input_size, hidden_size,batch_first=True)
    
    QLSTM (Quantum Long Short-Term Memory) e un modello ibrido che combina il calcolo quantistico e la LSTM classica, progettato per utilizzare il parallelismo del calcolo quantistico e la capacita di memoria della LSTM classica per elaborare dati sequenziali.

    :param para_num: Il numero di parametri nel circuito quantistico.
    :param num_of_qubits: Il numero di bit quantistici.
    :param input_size: La dimensione delle caratteristiche dei dati di input.
    :param hidden_size: La dimensione dell'unita nascosta.
    :param batch_first: Se la prima dimensione dell'input e la dimensione del batch.
    :return: Un'istanza di QLSTM.

    Example::

        import numpy as np
        from pyvqnet.tensor import tensor
        from pyvqnet.qnn.qlstm import QLSTM
        from pyvqnet.dtype import *
 
        para_num = 4
        num_of_qubits = 4
        input_size = 4
        hidden_size = 20
        batch_size = 3
        seq_length = 5
        qlstm = QLSTM(para_num, num_of_qubits, input_size, hidden_size, batch_first=True)
        x = tensor.QTensor(np.random.randn(batch_size, seq_length, input_size), dtype=kfloat32)

        output, (h_t, c_t) = qlstm(x)

        print("Output shape:", output.shape)  # Forma dell'output
        print("h_t shape:", h_t.shape)  # Forma dello stato nascosto h_t
        print("c_t shape:", c_t.shape)  # Forma dello stato della cella c_t



QMLPModel
--------------------------------------------------------------

.. py:class:: pyvqnet.qnn.qmlp.qmlp.QMLPModel(input_channels: int,output_channels: int,num_qubits: int, kernel: _size_type,stride: _size_type,padding: _padding_type = "valid",weight_initializer: Union[Callable, None] = None,bias_initializer: Union[Callable, None] = None,use_bias: bool = True,dtype: Union[int, None] = None)
    
    QMLPModel e una rete neurale di ispirazione quantistica basata su QMLP: An Error-Tolerant Nonlinear Quantum MLP Architecture using Parameterized Two-Qubit Gates (https://arxiv.org/abs/2206.01345). QMLPModel combina circuiti quantistici con operazioni classiche di rete neurale come pooling e layer completamente connessi. E progettato per elaborare dati quantistici ed estrarre caratteristiche rilevanti attraverso operazioni quantistiche e layer classici.

    :param input_channels: Il numero di caratteristiche di input.
    :param output_channels: Il numero di caratteristiche di output.
    :param num_qubits: Il numero di qubit.
    :param kernel: La dimensione della finestra di average pooling.
    :param stride: Il fattore di passo per il downsampling.
    :param padding: Il metodo di padding, opzioni "valid" o "same".
    :param weight_initializer: Inizializzatore dei pesi, default distribuzione normale.
    :param bias_initializer: Inizializzatore del bias, default inizializzazione zero.
    :param use_bias: Se usare il bias, default True.
    :param dtype: Default None, usa il tipo di dato predefinito.
    :return: Un'istanza di QMLPModel.

    Example::

        import numpy as np
        from pyvqnet.tensor import tensor
        from pyvqnet.qnn.qmlp.qmlp import QMLPModel
        from pyvqnet.dtype import *

        input_channels = 16
        output_channels = 10
        num_qubits = 4
        kernel = (2, 2)
        stride = (2, 2)
        padding = "valid"
        batch_size = 8

        model = QMLPModel(input_channels=num_qubits,
        output_channels=output_channels,
        num_qubits=num_qubits,
        kernel=kernel,
        stride=stride,
        padding=padding)

        x = tensor.QTensor(np.random.randn(batch_size, input_channels, 32, 32),dtype=kfloat32)

        output = model(x)

        print("Output shape:", output.shape)



QRLModel
-------------------------------------------------------------

.. py:class:: pyvqnet.qnn.qrl.QRLModel(num_qubits, n_layers)

    Modello di deep reinforcement learning quantistico che utilizza circuiti quantistici variazionali in :ref:`QDRL_DEMO`.

    :param num_qubits: Il numero di qubit usati nel circuito quantistico.
    :param n_layers: Il numero di layer nel circuito quantistico variazionale.
    :return: Un'istanza di QRLModel.

    Example::

        from pyvqnet.tensor import tensor, QTensor
        from pyvqnet.qnn.qrl import QRLModel

        num_qubits = 4
        model = QRLModel(num_qubits=num_qubits, n_layers=2)

        batch_size = 3
        x = QTensor([[1.1, 0.3, 1.2, 0.6], [0.2, 1.1, 0, 1.1], [1.3, 1.3, 0.3, 0.3]])
        output = model(x)
        output.backward()

        print("Model output:", output)


QRNN
------------------------------------------------------------------

.. py:class:: pyvqnet.qnn.qrnn.QRNN(para_num, num_of_qubits=4,input_size=100,hidden_size=100,batch_first=True)

    QRNN (Quantum Recurrent Neural Network) e una rete neurale ricorrente quantistica progettata per elaborare dati sequenziali e catturare dipendenze a lungo termine nella sequenza.

    :param para_num: Il numero di parametri nel circuito quantistico.
    :param num_of_qubits: Il numero di bit quantistici.
    :param input_size: La dimensione delle caratteristiche dei dati di input.
    :param hidden_size: La dimensione dell'unita nascosta.
    :param batch_first: Se la prima dimensione dell'input e la dimensione del batch, default True.
    :return: Un'istanza di QRNN.

    Example::

        from pyvqnet.dtype import kfloat32
        from pyvqnet.qnn.qrnn import QRNN
        from pyvqnet.tensor import tensor, QTensor
        import numpy as np

 
        para_num = 8
        num_of_qubits = 8
        input_size = 4
        hidden_size = 4
        batch_size = 1
        seq_length = 1
        qrnn = QRNN(para_num, num_of_qubits, input_size, hidden_size, batch_first=True)

        x = tensor.QTensor(np.random.randn(batch_size, seq_length, input_size), dtype=kfloat32)

        output, h_t = qrnn(x)

        print("Output shape:", output.shape)
        print("h_t shape:", h_t.shape)


TTOLayer
----------------------------------------------------------------

.. py:class:: pyvqnet.qnn.ttolayer.TTOLayer(inp_modes,out_modes,mat_ranks,biases_initializer=tensor.zeros)

    TTOLayer, basato su "Compressing deep neural networks by matrix product operators" (https://arxiv.org/abs/1904.06194), decompone il tensore di input per ottenere una rappresentazione efficiente di dati ad alta dimensionalita. Questo layer permette di apprendere la decomposizione tensoriale sotto vincoli di rango, che puo ridurre la complessita computazionale e l'utilizzo di memoria rispetto ai tradizionali layer completamente connessi.

    :param inp_modes: Le dimensioni del tensore di input.
    :param out_modes: Le dimensioni del tensore di output.
    :param mat_ranks: Il rango del kernel tensoriale (rango di decomposizione) nella decomposizione tensoriale.
    :param biases_initializer: La funzione di inizializzazione del bias.
    :return: Un'istanza di TTOLayer.

    Example::

        from pyvqnet.tensor import tensor
        import numpy as np
        from pyvqnet.qnn.ttolayer import TTOLayer
        from pyvqnet.dtype import kfloat32

        inp_modes = [4, 5]
        out_modes = [4, 5]
        mat_ranks = [1, 3, 1]
        tto_layer = TTOLayer(inp_modes, out_modes, mat_ranks)

        batch_size = 2
        len = 4
        embed_size = 5
        inp = tensor.QTensor(np.random.randn(batch_size, len, embed_size), dtype=kfloat32)

        output = tto_layer(inp)

        print("Input shape:", inp.shape)
        print("Output shape:", output.shape)


Altre funzioni
=====================



QuantumLayerAdjoint
-----------------------------------------
.. py:class:: pyvqnet.qnn.vqc.QuantumLayerAdjoint(general_module: pyvqnet.nn.Module, use_qpanda=False,name="")


    Un layer QuantumLayer a differenziazione automatica che utilizza il metodo della matrice aggiunta per il calcolo del gradiente, fare riferimento a `Efficient calculation of gradients in classical simulations of variational quantum algorithms <https://arxiv.org/abs/2009.02823>`_.

    :param general_module: Un'istanza di `pyvqnet.nn.Module` costruita usando solo l'interfaccia circuitale quantistica inferiore di `pyvqnet.qnn.vqc`.
    :param use_qpanda: Se usare la linea qpanda per la trasmissione forward, default: False.
    :param name: Il nome di questo layer, default "".

    .. note::

        QMachine per general_module deve impostare grad_method = "adjoint".
        Attualmente, sono supportati i seguenti gate logici parametrici: `RX`, `RY`, `RZ`, `PhaseShift`, `RXX`, `RYY`, `RZZ`, `RZX`, `U1`, `U2`, `U3` e altri circuiti variazionali senza gate logici parametrici.

    Example::

        from pyvqnet import tensor
        from pyvqnet.qnn.vqc import QuantumLayerAdjoint, QMachine, RX, RY, CNOT, PauliX, qmatrix, PauliZ, T, MeasureAll, RZ, VQC_RotCircuit, VQC_HardwareEfficientAnsatz
        import pyvqnet
        from pyvqnet.utils import set_random_seed

        set_random_seed(42)
        class QModel(pyvqnet.nn.Module):
            def __init__(self, num_wires, dtype, grad_mode=""):
                super(QModel, self).__init__()

                self._num_wires = num_wires
                self._dtype = dtype
                self.qm = QMachine(num_wires, dtype=dtype, grad_mode=grad_mode)
                self.rx_layer = RX(has_params=True, trainable=False, wires=0)
                self.ry_layer = RY(has_params=True, trainable=False, wires=1)
                self.rz_layer = RZ(has_params=True, trainable=False, wires=1)
                self.rz_layer2 = RZ(has_params=True, trainable=True, wires=1)

                self.rot = VQC_HardwareEfficientAnsatz(6, ["rx", "RY", "rz"],
                                                    entangle_gate="cnot",
                                                    entangle_rules="linear",
                                                    depth=5)
                self.tlayer = T(wires=1)
                self.cnot = CNOT(wires=[0, 1])
                self.measure = MeasureAll(obs = {
                    "X1":1
                })

            def forward(self, x, *args, **kwargs):
                self.qm.reset_states(x.shape[0])

                self.rx_layer(params=x[:, [0]], q_machine=self.qm)
                self.cnot(q_machine=self.qm)
                self.ry_layer(params=x[:, [1]], q_machine=self.qm)
                self.tlayer(q_machine=self.qm)
                self.rz_layer(params=x[:, [2]], q_machine=self.qm)
                self.rz_layer2(q_machine=self.qm)
                self.rot(q_machine=self.qm)
                rlt = self.measure(q_machine=self.qm)

                return rlt


        input_x = tensor.QTensor([[0.1, 0.2, 0.3]])

        input_x = tensor.broadcast_to(input_x, [4, 3])

        input_x.requires_grad = True

        qunatum_model = QModel(num_wires=6,
                            dtype=pyvqnet.kcomplex64,
                            grad_mode="adjoint")

        adjoint_model = QuantumLayerAdjoint(qunatum_model)
        adjoint_model.train()
        batch_y = adjoint_model(input_x)
        batch_y.backward()
        print(batch_y)
        # [[-0.0778451],
        #  [-0.0778451],
        #  [-0.0778451],
        #  [-0.0778451]]


DataParallelVQCAdjointLayer
---------------------------------------------------------------

.. py:class:: pyvqnet.distributed.DataParallelVQCAdjointLayer(Comm_OP, vqc_module, name="")

Crea un VQC con parallelismo dei dati per la dimensione del batch utilizzando il calcolo del gradiente aggiunto. Dove ``vqc_module`` deve essere un modulo VQC di tipo ``QuantumLayerAdjoint``.

Se usiamo N nodi per eseguire questo modulo,
In ogni nodo, i dati `batch_size/N` vengono elaborati in forward per eseguire il circuito quantistico variazionale e calcolare i gradienti.

:param Comm_OP: Imposta il controller di comunicazione per l'ambiente distribuito.
:param vqc_module: Un modulo VQC di tipo QuantumLayerAdjoint con forward(), assicurarsi che qmachine sia configurato correttamente.
:param name: Il nome del modulo. Il valore predefinito e una stringa vuota.
:return: Un modulo che puo calcolare circuiti quantistici.

Example::

    #mpirun -n 2 python test.py

    import sys
    sys.path.insert(0,"../../")
    from pyvqnet.distributed import CommController,DataParallelVQCAdjointLayer,\
    get_local_rank

    from pyvqnet.qnn import *
    from pyvqnet.qnn.vqc import *
    import pyvqnet
    from pyvqnet.nn import Module, Linear
    from pyvqnet.device import DEV_GPU_0

    bsize = 100


    class QModel(Module):
        def __init__(self, num_wires, dtype, grad_mode="adjoint"):
            super(QModel, self).__init__()

            self._num_wires = num_wires
            self._dtype = dtype
            self.qm = QMachine(num_wires, dtype=dtype, grad_mode=grad_mode)
            self.rx_layer = RX(has_params=True, trainable=False, wires=0)
            self.ry_layer = RY(has_params=True, trainable=False, wires=1)
            self.rz_layer = RZ(has_params=True, trainable=False, wires=1)
            self.u1 = U1(has_params=True, trainable=True, wires=[2])
            self.u2 = U2(has_params=True, trainable=True, wires=[3])
            self.u3 = U3(has_params=True, trainable=True, wires=[1])
            self.i = I(wires=[3])
            self.s = S(wires=[3])
            self.x1 = X1(wires=[3])
            self.y1 = Y1(wires=[3])
            self.z1 = Z1(wires=[3])
            self.x = PauliX(wires=[3])
            self.y = PauliY(wires=[3])
            self.z = PauliZ(wires=[3])
            self.swap = SWAP(wires=[2, 3])
            self.cz = CZ(wires=[2, 3])
            self.cr = CR(has_params=True, trainable=True, wires=[2, 3])
            self.rxx = RXX(has_params=True, trainable=True, wires=[2, 3])
            self.rzz = RYY(has_params=True, trainable=True, wires=[2, 3])
            self.ryy = RZZ(has_params=True, trainable=True, wires=[2, 3])
            self.rzx = RZX(has_params=True, trainable=False, wires=[2, 3])
            self.toffoli = Toffoli(wires=[2, 3, 4], use_dagger=True)

            self.h = Hadamard(wires=[1])

            self.iSWAP = iSWAP(wires=[0, 2])
            self.tlayer = T(wires=1)
            self.cnot = CNOT(wires=[0, 1])
            self.measure = MeasureAll(obs={'Z0': 2})

        def forward(self, x, *args, **kwargs):
            self.qm.reset_states(x.shape[0])
            self.i(q_machine=self.qm)
            self.s(q_machine=self.qm)
            self.swap(q_machine=self.qm)
            self.cz(q_machine=self.qm)
            self.x(q_machine=self.qm)
            self.x1(q_machine=self.qm)
            self.y(q_machine=self.qm)
            self.y1(q_machine=self.qm)
            self.z(q_machine=self.qm)
            self.z1(q_machine=self.qm)
            self.ryy(q_machine=self.qm)
            self.rxx(q_machine=self.qm)
            self.rzz(q_machine=self.qm)
            self.rzx(q_machine=self.qm, params=x[:, [1]])

            self.u1(q_machine=self.qm)
            self.u2(q_machine=self.qm)
            self.u3(q_machine=self.qm)
            self.rx_layer(params=x[:, [0]], q_machine=self.qm)
            self.cnot(q_machine=self.qm)
            self.h(q_machine=self.qm)
            self.iSWAP(q_machine=self.qm)
            self.ry_layer(params=x[:, [1]], q_machine=self.qm)
            self.tlayer(q_machine=self.qm)
            self.rz_layer(params=x[:, [2]], q_machine=self.qm)
            self.toffoli(q_machine=self.qm)
            rlt = self.measure(q_machine=self.qm)

            return rlt


    pyvqnet.utils.set_random_seed(42)

    Comm_OP = CommController("mpi")

    input_x = tensor.QTensor([[0.1, 0.2, 0.3]])
    input_x = tensor.broadcast_to(input_x, [bsize, 3])
    input_x.requires_grad = True

    qunatum_model = QModel(num_wires=6, dtype=pyvqnet.kcomplex64)

    l = DataParallelVQCAdjointLayer(
        Comm_OP,
        qunatum_model,
    )
    l.train()
    y = l(input_x)

    y.backward()


DataParallelVQCLayer
---------------------------------------------------------------

.. py:class:: pyvqnet.distributed.DataParallelVQCLayer(Comm_OP, vqc_module, name="")

    Crea un VQC con parallelismo dei dati per la dimensione del batch utilizzando il calcolo a differenziazione automatica.
    Se usiamo N nodi per eseguire questo modulo,
    In ogni nodo, i dati `batch_size/N` vengono elaborati in forward attraverso il circuito quantistico variazionale per calcolare i gradienti.

    :param Comm_OP: Imposta il controller di comunicazione per l'ambiente distribuito.
    :param vqc_module: Modulo VQC con forward(), assicurarsi che qmachine sia configurato correttamente.
    :param name: Nome del modulo. Il valore predefinito e una stringa vuota.
    :return: Modulo che puo calcolare circuiti quantistici.

    Example::

        #mpirun -n 2 python xxx.py

        import pyvqnet.backends

        from pyvqnet.qnn.vqc import QMachine, cnot, rx, rz, ry, MeasureAll
        from pyvqnet.tensor import tensor

        from pyvqnet.distributed import CommController, DataParallelVQCLayer

        from pyvqnet.qnn import *
        from pyvqnet.qnn.vqc import *
        import pyvqnet
        from pyvqnet.nn import Module, Linear
        from pyvqnet.device import DEV_GPU_0


        class QModel(Module):

            def __init__(self, num_wires, num_layer, dtype, grad_mode=""):
                super(QModel, self).__init__()

                self._num_wires = num_wires
                self._dtype = dtype
                self.qm = QMachine(num_wires, dtype=dtype, grad_mode=grad_mode)

                self.measure = MeasureAll(obs=PauliX)
                self.n = num_wires
                self.l = num_layer

            def forward(self, param, *args, **kwargs):
                n = self.n
                l = self.l
                qm = self.qm
                qm.reset_states(param.shape[0])
                j = 0

                for j in range(l):
                    cnot(qm, wires=[j, (j + 1) % l])
                    for i in range(n):
                        rx(qm, i, param[:, 3 * n * j + i])
                    for i in range(n):
                        rz(qm, i, param[:, 3 * n * j + i + n], i)
                    for i in range(n):
                        rx(qm, i, param[:, 3 * n * j + i + 2 * n], i)

                y = self.measure(qm)
                return y


        n = 4
        b = 4
        l = 2

        input = tensor.ones([b, 3 * n * l])

        Comm = CommController("mpi")
        
        input.requires_grad = True
        qunatum_model = QModel(num_wires=n, num_layer=l, dtype=pyvqnet.kcomplex64)
        
        layer = qunatum_model

        layer = DataParallelVQCLayer(
            Comm,
            qunatum_model,
        )
        y = layer(input)
        y.backward()


vqc_to_originir_list
-------------------------------------

.. py:function:: pyvqnet.qnn.vqc.vqc_to_originir_list(vqc_model: pyvqnet.nn.Module)

    Converte il modulo VQNet vqc in originIR.

    vqc_model deve eseguire la funzione forward prima di questa funzione per ottenere i dati di input.
    Se i dati di input sono dati batch. Per ogni input restituira piu stringhe IR.

    :param vqc_model: Modulo VQNet vqc, che deve prima eseguire forward.

    :return: stringa originIR o lista di stringhe originIR.

    Example::

        import pyvqnet
        import pyvqnet.tensor as tensor
        from pyvqnet.qnn.vqc import *
        from pyvqnet.nn import Module
        from pyvqnet.utils import set_random_seed
        set_random_seed(42)
        class QModel(Module):
            def __init__(self, num_wires, dtype,grad_mode=""):
                super(QModel, self).__init__()

                self._num_wires = num_wires
                self._dtype = dtype
                self.qm = QMachine(num_wires, dtype=dtype,grad_mode=grad_mode,save_ir=True)
                self.rx_layer = RX(has_params=True, trainable=False, wires=0)
                self.ry_layer = RY(has_params=True, trainable=False, wires=1)
                self.rz_layer = RZ(has_params=True, trainable=False, wires=1)
                self.u1 = U1(has_params=True,trainable=True,wires=[2])
                self.u2 = U2(has_params=True,trainable=True,wires=[3])
                self.u3 = U3(has_params=True,trainable=True,wires=[1])
                self.i = I(wires=[3])
                self.s = S(wires=[3])
                self.x1 = X1(wires=[3])
                self.y1 = Y1(wires=[3])
                self.z1 = Z1(wires=[3])
                self.x = PauliX(wires=[3])
                self.y = PauliY(wires=[3])
                self.z = PauliZ(wires=[3])
                self.swap = SWAP(wires=[2,3])
                self.cz = CZ(wires=[2,3])
                self.cr = CR(has_params=True,trainable=True,wires=[2,3])
                self.rxx = RXX(has_params=True,trainable=True,wires=[2,3])
                self.rzz = RYY(has_params=True,trainable=True,wires=[2,3])
                self.ryy = RZZ(has_params=True,trainable=True,wires=[2,3])
                self.rzx = RZX(has_params=True,trainable=False, wires=[2,3])
                self.toffoli = Toffoli(wires=[2,3,4],use_dagger=True)
                self.h =Hadamard(wires=[1])
                self.rot = VQC_HardwareEfficientAnsatz(6, ["rx", "RY", "rz"],
                                                    entangle_gate="cnot",
                                                    entangle_rules="linear",
                                                    depth=5)

                self.iSWAP = iSWAP( wires=[0,2])
                self.tlayer = T(wires=1)
                self.cnot = CNOT(wires=[0, 1])
                self.measure = MeasureAll(obs = {
                    "X1":1
                })

            def forward(self, x, *args, **kwargs):
                self.qm.reset_states(x.shape[0])
                self.i(q_machine=self.qm)
                self.s(q_machine=self.qm)
                self.swap(q_machine=self.qm)
                self.cz(q_machine=self.qm)
                self.x(q_machine=self.qm)
                self.x1(q_machine=self.qm)
                self.y(q_machine=self.qm)
                self.y1(q_machine=self.qm)
                self.z(q_machine=self.qm)
                self.z1(q_machine=self.qm)
                self.ryy(q_machine=self.qm)
                self.rxx(q_machine=self.qm)
                self.rzz(q_machine=self.qm)
                self.rzx(q_machine=self.qm,params = x[:,[1]])
                self.cr(q_machine=self.qm)
                self.u1(q_machine=self.qm)
                self.u2(q_machine=self.qm)
                self.u3(q_machine=self.qm)
                self.rx_layer(params = x[:,[0]], q_machine=self.qm)
                self.cnot(q_machine=self.qm)
                self.h(q_machine=self.qm)
                self.iSWAP(q_machine=self.qm)
                self.ry_layer(params = x[:,[1]], q_machine=self.qm)
                self.tlayer(q_machine=self.qm)
                self.rz_layer(params = x[:,[2]], q_machine=self.qm)
                self.toffoli(q_machine=self.qm)
                rlt = self.measure(q_machine=self.qm)

                return rlt
            

        input_x = tensor.QTensor([[0.1, 0.2, 0.3]])

        input_x = tensor.broadcast_to(input_x,[2,3])

        input_x.requires_grad = True

        qunatum_model = QModel(num_wires=6, dtype=pyvqnet.kcomplex64)

        batch_y = qunatum_model(input_x)
        batch_y.backward()
        ll = vqc_to_originir_list(qunatum_model)
        from pyqpanda import CPUQVM,convert_originir_str_to_qprog,convert_qprog_to_originir
        for l in ll :
            print(l)

            machine = CPUQVM()
            machine.init_qvm()
            prog, qv, cv = convert_originir_str_to_qprog(l, machine)
            print(machine.prob_run_dict(prog,qv))

        # QINIT 6
        # CREG 6
        # I q[3]
        # S q[3]
        # SWAP q[2],q[3]
        # CZ q[2],q[3]
        # X q[3]
        # X1 q[3]
        # Y q[3]
        # Y1 q[3]
        # Z q[3]
        # Z1 q[3]
        # RZZ q[2],q[3],(4.484121322631836)
        # RXX q[2],q[3],(5.302337169647217)
        # RYY q[2],q[3],(3.470323085784912)
        # RZX q[2],q[3],(0.20000000298023224)
        # CR q[2],q[3],(5.467088222503662)
        # U1 q[2],(6.254805088043213)
        # U2 q[3],(1.261604905128479,0.9901542067527771)
        # U3 q[1],(5.290454387664795,6.182775020599365,1.1797741651535034)
        # RX q[0],(0.10000000149011612)
        # CNOT q[0],q[1]
        # H q[1]
        # ISWAP q[0],q[2]
        # RY q[1],(0.20000000298023224)
        # T q[1]
        # RZ q[1],(0.30000001192092896)
        # DAGGER
        # TOFFOLI q[2],q[3],q[4]
        # ENDDAGGER

        # {'000000': 0.006448949346548678, '000001': 0.004089870964118778, '000010': 0.1660891289303212, '000011': 0.08520414851665635, '000100': 0.0048503036661063, '000101': 8.679196482917438e-05, '000110': 0.14379026566368325, '000111': 0.0005079553597106437, '001000': 0.0023774056959510325, '001001': 0.008241263544544148, '001010': 0.06122877075562884, '001011': 0.1984226195587807, '001100': 0.0, '001101': 0.0, '001110': 0.0, '001111': 0.0, '010000': 0.0, '010001': 0.0, '010010': 0.0, '010011': 0.0, '010100': 0.0, '010101': 0.0, '010110': 0.0, '010111': 0.0, '011000': 0.0, '011001': 0.0, '011010': 0.0, '011011': 0.0, '011100': 0.011362100696548312, '011101': 0.00019143557058348747, '011110': 0.3059886012103368, '011111': 0.0011203885556518832, '100000': 0.0, '100001': 0.0, '100010': 0.0, '100011': 0.0, '100100': 0.0, '100101': 0.0, '100110': 0.0, '100111': 0.0, '101000': 0.0, '101001': 0.0, '101010': 0.0, '101011': 0.0, '101100': 0.0, '101101': 0.0, '101110': 0.0, '101111': 0.0, '110000': 0.0, '110001': 0.0, '110010': 0.0, '110011': 0.0, '110100': 0.0, '110101': 0.0, '110110': 0.0, '110111': 0.0, '111000': 0.0, '111001': 0.0, '111010': 0.0, '111011': 0.0, '111100': 0.0, '111101': 0.0, '111110': 0.0, '111111': 0.0}
        # QINIT 6
        # CREG 6
        # I q[3]
        # S q[3]
        # SWAP q[2],q[3]
        # CZ q[2],q[3]
        # X q[3]
        # X1 q[3]
        # Y q[3]
        # Y1 q[3]
        # Z q[3]
        # Z1 q[3]
        # RZZ q[2],q[3],(4.484121322631836)
        # RXX q[2],q[3],(5.302337169647217)
        # RYY q[2],q[3],(3.470323085784912)
        # RZX q[2],q[3],(0.20000000298023224)
        # CR q[2],q[3],(5.467088222503662)
        # U1 q[2],(6.254805088043213)
        # U2 q[3],(1.261604905128479,0.9901542067527771)
        # U3 q[1],(5.290454387664795,6.182775020599365,1.1797741651535034)
        # RX q[0],(0.10000000149011612)
        # CNOT q[0],q[1]
        # H q[1]
        # ISWAP q[0],q[2]
        # RY q[1],(0.20000000298023224)
        # T q[1]
        # RZ q[1],(0.30000001192092896)
        # DAGGER
        # TOFFOLI q[2],q[3],q[4]
        # ENDDAGGER

        # {'000000': 0.006448949346548678, '000001': 0.004089870964118778, '000010': 0.1660891289303212, '000011': 0.08520414851665635, '000100': 0.0048503036661063, '000101': 8.679196482917438e-05, '000110': 0.14379026566368325, '000111': 0.0005079553597106437, '001000': 0.0023774056959510325, '001001': 0.008241263544544148, '001010': 0.06122877075562884, '001011': 0.1984226195587807, '001100': 0.0, '001101': 0.0, '001110': 0.0, '001111': 0.0, '010000': 0.0, '010001': 0.0, '010010': 0.0, '010011': 0.0, '010100': 0.0, '010101': 0.0, '010110': 0.0, '010111': 0.0, '011000': 0.0, '011001': 0.0, '011010': 0.0, '011011': 0.0, '011100': 0.011362100696548312, '011101': 0.00019143557058348747, '011110': 0.3059886012103368, '011111': 0.0011203885556518832, '100000': 0.0, '100001': 0.0, '100010': 0.0, '100011': 0.0, '100100': 0.0, '100101': 0.0, '100110': 0.0, '100111': 0.0, '101000': 0.0, '101001': 0.0, '101010': 0.0, '101011': 0.0, '101100': 0.0, '101101': 0.0, '101110': 0.0, '101111': 0.0, '110000': 0.0, '110001': 0.0, '110010': 0.0, '110011': 0.0, '110100': 0.0, '110101': 0.0, '110110': 0.0, '110111': 0.0, '111000': 0.0, '111001': 0.0, '111010': 0.0, '111011': 0.0, '111100': 0.0, '111101': 0.0, '111110': 0.0, '111111': 0.0}

originir_to_vqc
------------------------------------------

.. py:function:: pyvqnet.qnn.vqc.originir_to_vqc(originir, tmp="code_tmp.py", verbose=False)

    Analizza originIR in codice modello vqc.
    Il codice crea un circuito quantistico variazionale `pyvqnet.nn.Module` senza `Measure`, e restituisce la forma del vettore di stato dello stato quantistico, come [b,2,...,2].
    Questa funzione generera un file di codice che definisce il corrispondente modello VQNet in "./origin_ir_gen_code/" + tmp + ".py".

    :param originir: IR originale.
    :param tmp: nome del file di codice, default ``code_tmp.py``.
    :param verbose: Se visualizzare il codice generato, default = False
    :return:
        Genera codice eseguibile.

    Example::

        from pyvqnet.qnn.vqc import originir_to_vqc
        ss = "QINIT 3\nCREG 3\nH q[1]"
    
        Z = originir_to_vqc(ss,verbose=True)

        exec(Z)
        m =Exported_Model()
        print(m(2))

        # from pyvqnet.nn import Module
        # from pyvqnet.tensor import QTensor
        # from pyvqnet.qnn.vqc import *
        # class Exported_Model(Module):
        # def __init__(self, name=""):
        # super().__init__(name)

        # self.q_machine = QMachine(num_wires=3)
        # self.H_0 = Hadamard(wires=1, use_dagger = False)

        # def forward(self, x, *args, **kwargs):
        # x = self.H_0(q_machine=self.q_machine)
        # return self.q_machine.states

        # [[[[0.7071068+0.j 0. +0.j]
        # [0.7071068+0.j 0. +0.j]]

        # [[0. +0.j 0. +0.j]
        # [0. +0.j 0. +0.j]]]]


model_summary
-----------------------------------------------

.. py:function:: pyvqnet.model_summary(vqc_module)

    Stampa le informazioni sui layer classici e gli operatori di gate quantistici registrati in vqc_module.

    :param vqc_module: modulo vqc
    :return:
        stringa di riepilogo


    Example::

        import pyvqnet
        from pyvqnet.qnn.vqc import QMachine, RX, RY, CNOT, PauliX, qmatrix, PauliZ,MeasureAll
        from pyvqnet.tensor import QTensor, tensor
        from pyvqnet import kcomplex64
        
        from pyvqnet.nn import LSTM,Linear
        from pyvqnet import model_summary
        class QModel(pyvqnet.nn.Module):
            def __init__(self, num_wires, dtype):
                super(QModel, self).__init__()

                self._num_wires = num_wires
                self._dtype = dtype
                self.qm = QMachine(num_wires, dtype=dtype)
                self.rx_layer1 = RX(has_params=True,
                                    trainable=True,
                                    wires=1,
                                    init_params=tensor.QTensor([0.5]))
                self.ry_layer2 = RY(has_params=True,
                                    trainable=True,
                                    wires=0,
                                    init_params=tensor.QTensor([-0.5]))
                self.xlayer = PauliX(wires=0)
                self.cnot = CNOT(wires=[0, 1])
                self.measure = MeasureAll(obs = PauliZ)
                self.linear = Linear(24,2)
                self.lstm =LSTM(23,5)
            def forward(self, x, *args, **kwargs):
                return super().forward(x, *args, **kwargs)
        Z = QModel(4,kcomplex64)

        print(model_summary(Z))
        # ###################QModel Summary#######################

        # classic layers: {'Linear': 1, 'LSTM': 1}
        # total classic parameters: 650

        # =========================================
        # qubits num: 4
        # gates: {'RX': 1, 'RY': 1, 'PauliX': 1, 'CNOT': 1}
        # total quantum gates: 4
        # total quantum parameter gates: 2
        # total quantum parameters: 2
        # #########################################################


QNG
---------------------------------------------------------------

.. py:class:: pyvqnet.qnn.vqc.qng.QNG(qmodel, stepsize=0.01, momentum=0)

    I modelli di machine learning quantistico usano generalmente il metodo della discesa del gradiente per ottimizzare i parametri nei circuiti logici quantistici variazionali. La formula del classico metodo di discesa del gradiente e la seguente:

    .. math:: \theta_{t+1} = \theta_t -\eta \nabla \mathcal{L}(\theta),

    Essenzialmente, ad ogni iterazione, calcoliamo la direzione di massima discesa del gradiente nello spazio dei parametri come direzione di cambiamento dei parametri.
    In qualsiasi direzione nello spazio, la velocita di discesa nell'intervallo locale non e veloce quanto quella della direzione del gradiente negativo.
    In spazi diversi, la derivazione della direzione di massima discesa dipende dalla norma della differenziazione dei parametri, la metrica di distanza. La metrica di distanza gioca un ruolo centrale qui,
    Metriche diverse portano a diverse direzioni di massima discesa. Per lo spazio euclideo in cui si trovano i parametri nel problema di ottimizzazione classico, la direzione di massima discesa e la direzione del gradiente negativo.
    Anche cosi, ad ogni passo dell'ottimizzazione dei parametri, mentre la funzione di perdita cambia con i parametri, il suo spazio dei parametri viene trasformato. Rende possibile trovare un'altra norma di distanza migliore.

    Il `metodo del gradiente naturale quantistico <https://arxiv.org/abs/1909.02108>`_ attinge a concetti del `metodo del gradiente naturale classico di Amari <https://www.mitpressjournals.org/doi/abs/10.1162/089976698300017746>`__ ,
    Consideriamo invece il problema di ottimizzazione come una distribuzione di probabilita dei possibili valori di output per un dato input (cioe, stima di massima verosimiglianza), un approccio migliore consiste nell'eseguire la
    discesa del gradiente nello spazio delle distribuzioni, che e adimensionale e invariante rispetto alla parametrizzazione. Pertanto, indipendentemente dalla parametrizzazione, ogni passo di ottimizzazione scegliera sempre la dimensione del passo ottimale per ogni parametro.
    Nei compiti di machine learning quantistico, lo spazio degli stati quantistici ha un tensore metrico invariante unico chiamato tensore metrico di Fubini-Study :math:`g_{ij}`.
    Questo tensore converte la massima discesa nello spazio dei parametri del circuito quantistico nella massima discesa nello spazio delle distribuzioni.
    La formula per il gradiente naturale quantistico e la seguente:

    .. math:: \theta_{t+1} = \theta_t + momentum(x^{(t)} - x^{(t-1)}) - \eta g^{+}(\theta_t)\nabla \mathcal{L}(\theta)

    dove :math:`g^{+}` e la pseudo-inversa.

    `wrapper_calculate_qng` e un decoratore che deve essere aggiunto alla funzione forward del modello per il calcolo del gradiente naturale quantistico. Vengono ottimizzati solo i parametri di tipo `Parameter` registrati con il modello.

    :param qmodel: Modello di circuito quantistico variazionale, e necessario usare `wrapper_calculate_qng` come decoratore della funzione forward.
    :param stepsize: La dimensione del passo del metodo di discesa del gradiente, default 0.01.
    :param momentum: Momento, default 0.

    .. note::

        Testato solo su dati non batch.
        Sono supportati solo circuiti quantistici puramente variazionali.
        step() aggiornera i gradienti dell'input e dei parametri.
        step() aggiorna solo i valori numerici dei parametri del modello.


    Example::

        from pyvqnet.qnn.vqc import QMachine, RX, RY, RZ, CNOT, rz, PauliX, qmatrix, PauliZ, Probability, rx, ry, MeasureAll, U2
        from pyvqnet.tensor import QTensor, tensor
        import pyvqnet
        import numpy as np

        from pyvqnet.qnn.vqc import wrapper_calculate_qng

        class QModel(pyvqnet.nn.Module):
            def __init__(self, num_wires, dtype):
                super(QModel, self).__init__()

                self._num_wires = num_wires
                self._dtype = dtype
                self.qm = QMachine(num_wires, dtype=dtype)
                self.rz_layer1 = RZ(has_params=True, trainable=False, wires=0)
                self.rz_layer2 = RZ(has_params=True, trainable=False, wires=1)
                self.u2_layer1 = U2(has_params=True, trainable=False, wires=0)
                self.l_train1 = RY(has_params=True, trainable=True, wires=1)
                self.l_train1.params.init_from_tensor(
                    QTensor([333], dtype=pyvqnet.kfloat32))
                self.l_train2 = RX(has_params=True, trainable=True, wires=2)
                self.l_train2.params.init_from_tensor(
                    QTensor([4444], dtype=pyvqnet.kfloat32))
                self.xlayer = PauliX(wires=0)
                self.cnot01 = CNOT(wires=[0, 1])
                self.cnot12 = CNOT(wires=[1, 2])
                self.measure = MeasureAll(obs={'Y0': 1})

            @wrapper_calculate_qng
            def forward(self, x, *args, **kwargs):
                self.qm.reset_states(x.shape[0])

                ry(q_machine=self.qm, wires=0, params=np.pi / 4)
                ry(q_machine=self.qm, wires=1, params=np.pi / 3)
                ry(q_machine=self.qm, wires=2, params=np.pi / 7)
                self.rz_layer1(q_machine=self.qm, params=x[:, [0]])
                self.rz_layer2(q_machine=self.qm, params=x[:, [1]])

                self.u2_layer1(q_machine=self.qm, params=x[:, [4, 5]])  #

                self.cnot01(q_machine=self.qm)
                self.cnot12(q_machine=self.qm)
                ry(q_machine=self.qm, wires=0, params=np.pi / 7)
                ry(q_machine=self.qm, wires=1, params=x[:, [2]])
                rx(q_machine=self.qm, wires=2, params=x[:, [3]])
                rz(q_machine=self.qm, wires=1, params=x[:, [3]])
                ry(q_machine=self.qm, wires=0, params=np.pi / 7)
                rz(q_machine=self.qm, wires=1, params=x[:, [3]])

                self.cnot01(q_machine=self.qm)
                self.cnot12(q_machine=self.qm)
                rlt = self.measure(q_machine=self.qm)
                return rlt

        qmodel = QModel(3, pyvqnet.kcomplex64)

        x = QTensor([[1111.0, 2222, 333,444, 55, 666]], requires_grad=True)

        qng = pyvqnet.qnn.vqc.QNG(qmodel,0.01)
        qng.step(x)

        print(x)
        print(x.grad)


wrapper_single_qubit_op_fuse
---------------------------------------------------------------

.. py:function:: pyvqnet.qnn.vqc.wrapper_single_qubit_op_fuse(f)

    Un decoratore per fondere operazioni a singolo bit in operazioni Rot.

    .. note::

        f e la funzione forward del modulo, e la funzione forward del modello deve essere eseguita una volta per avere effetto.
        Il modello definito qui eredita da `pyvqnet.qnn.vqc.QModule`, che e una sottoclasse di `pyvqnet.nn.Module`.


    Example::

        from pyvqnet import tensor
        from pyvqnet import kcomplex128
        from pyvqnet.tensor import adjoint
        import numpy as np
        from pyvqnet.qnn.vqc import single_qubit_ops_fuse, wrapper_single_qubit_op_fuse, QModule,op_history_summary
        from pyvqnet.qnn.vqc import QMachine, RX, RY, CNOT, PauliX, qmatrix, PauliZ, T, MeasureAll, RZ
        from pyvqnet.tensor import QTensor, tensor
        import pyvqnet
        import numpy as np
        from pyvqnet.utils import set_random_seed


        set_random_seed(42)

        class QModel(QModule):
            def __init__(self, num_wires, dtype):
                super(QModel, self).__init__()

                self._num_wires = num_wires
                self._dtype = dtype
                self.qm = QMachine(num_wires, dtype=dtype)
                self.rx_layer = RX(has_params=True, trainable=False, wires=0, dtype=dtype)
                self.ry_layer = RY(has_params=True, trainable=False, wires=1, dtype=dtype)
                self.rz_layer = RZ(has_params=True, trainable=False, wires=1, dtype=dtype)
                self.rz_layer2 = RZ(has_params=True, trainable=False, wires=1, dtype=dtype)
                self.tlayer = T(wires=1)
                self.cnot = CNOT(wires=[0, 1])
                self.measure = MeasureAll(obs={
                    "X1":1
                })

            @wrapper_single_qubit_op_fuse
            def forward(self, x, *args, **kwargs):
                self.qm.reset_states(x.shape[0])

                self.rx_layer(params=x[:, [0]], q_machine=self.qm)
                self.cnot(q_machine=self.qm)
                self.ry_layer(params=x[:, [1]], q_machine=self.qm)
                self.tlayer(q_machine=self.qm)
                self.rz_layer(params=x[:, [2]], q_machine=self.qm)
                self.rz_layer2(params=x[:, [3]], q_machine=self.qm)
                rlt = self.measure(q_machine=self.qm)

                return rlt

        input_x = tensor.QTensor([[0.1, 0.2, 0.3, 0.4], [0.1, 0.2, 0.3, 0.4]],
                                dtype=pyvqnet.kfloat64)

        input_xt = tensor.tile(input_x, (100, 1))
        input_xt.requires_grad = True

        qunatum_model = QModel(num_wires=2, dtype=pyvqnet.kcomplex128)
        batch_y = qunatum_model(input_xt)
        print(op_history_summary(qunatum_model.qm.op_history))


        # ###################Summary#######################
        # qubits num: 2
        # gates: {'rot': 2, 'cnot': 1}
        # total gates: 3
        # total parameter gates: 2
        # total parameters: 6
        # #################################################


wrapper_commute_controlled
---------------------------------------------------------------

.. py:function:: pyvqnet.qnn.vqc.wrapper_commute_controlled(f, direction = "right")

    Decoratore per lo scambio di gate controllati
    Questa e una trasformazione quantistica usata per spostare i gate scambiabili davanti ai bit di controllo e target dell'operazione controllata.
    I gate diagonali su entrambi i lati del bit di controllo non influenzano il risultato del gate controllato; pertanto, possiamo spingere tutti i gate a singolo bit che agiscono sul primo bit insieme verso destra (e fonderli se necessario).
    Allo stesso modo, i gate X sono intercambiabili con i bit target di CNOT e Toffoli (come lo sono PauliY e CRY).
    Possiamo usare questa trasformazione per spingere i gate a singolo bit il piu possibile in profondita nell'operazione controllata.

    .. note::

        f e la funzione forward del modulo, e la funzione forward del modello deve essere eseguita una volta per avere effetto.
        Il modello definito qui eredita da `pyvqnet.qnn.vqc.QModule`, che e una sottoclasse di `pyvqnet.nn.Module`.

    :param f: funzione forward.
    :param direction: La direzione in cui spostare il gate a singolo bit, i valori opzionali sono "left" o "right", default "right".



    Example::

        from pyvqnet import tensor
        from pyvqnet.qnn.vqc import QMachine
        from pyvqnet import kcomplex128
        from pyvqnet.tensor import adjoint
        import numpy as np
        from pyvqnet.qnn.vqc import wrapper_commute_controlled, pauliy, QModule,op_history_summary

        from pyvqnet.qnn.vqc import QMachine, RX, RY, CNOT, S, CRY, PauliZ, PauliX, T, MeasureAll, RZ, CZ, PhaseShift, Toffoli, cnot, cry, toffoli
        from pyvqnet.tensor import QTensor, tensor
        import pyvqnet
        import numpy as np
        from pyvqnet.utils import set_random_seed
        from pyvqnet.qnn import expval, QuantumLayerV2
        import time
        from functools import partial
        set_random_seed(42)

        class QModel(QModule):
            def __init__(self, num_wires, dtype):
                super(QModel, self).__init__()

                self._num_wires = num_wires
                self._dtype = dtype
                self.qm = QMachine(num_wires, dtype=dtype)

                self.cz = CZ(wires=[0, 2])
                self.paulix = PauliX(wires=2)
                self.s = S(wires=0)
                self.ps = PhaseShift(has_params=True, trainable= True, wires=0, dtype=dtype)
                self.t = T(wires=0)
                self.rz = RZ(has_params=True, wires=1, dtype=dtype)
                self.measure = MeasureAll(obs={
                                "Z0":1
                })

            @partial(wrapper_commute_controlled, direction="left")
            def forward(self, x, *args, **kwargs):
                self.qm.reset_states(x.shape[0])
                self.cz(q_machine=self.qm)
                self.paulix(q_machine=self.qm)
                self.s(q_machine=self.qm)
                cnot(q_machine=self.qm, wires=[0, 1])
                pauliy(q_machine=self.qm, wires=1)
                cry(q_machine=self.qm, params=1 / 2, wires=[0, 1])
                self.ps(q_machine=self.qm)
                toffoli(q_machine=self.qm, wires=[0, 1, 2])
                self.t(q_machine=self.qm)
                self.rz(q_machine=self.qm)
                rlt = self.measure(q_machine=self.qm)

                return rlt

        import pyvqnet
        import pyvqnet.tensor as tensor
        input_x = tensor.QTensor([[0.1, 0.2, 0.3, 0.4], [0.1, 0.2, 0.3, 0.4]],
                                    dtype=pyvqnet.kfloat64)

        input_xt = tensor.tile(input_x, (100, 1))
        input_xt.requires_grad = True

        qunatum_model = QModel(num_wires=3, dtype=pyvqnet.kcomplex128)

        batch_y = qunatum_model(input_xt)
        for d in qunatum_model.qm.op_history:
            name = d["name"]
            wires = d["wires"]
            p = d["params"]
            print(f"name: {name} wires: {wires}, params = {p}")


        # name: s wires: (0,), params = None
        # name: phaseshift wires: (0,), params = [[4.744782]]
        # name: t wires: (0,), params = None
        # name: cz wires: (0, 2), params = None
        # name: paulix wires: (2,), params = None
        # name: cnot wires: (0, 1), params = None
        # name: pauliy wires: (1,), params = None
        # name: cry wires: (0, 1), params = [[0.5]]
        # name: rz wires: (1,), params = [[4.7447823]]
        # name: toffoli wires: (0, 1, 2), params = None
        # name: MeasureAll wires: [0], params = None


wrapper_merge_rotations
---------------------------------------------------------------

.. py:function:: pyvqnet.qnn.vqc.wrapper_merge_rotations(f)

    Decoratore per fondere cancelli dello stesso tipo, inclusi "rx", "ry", "rz", "phaseshift", "crx", "cry", "crz", "controlledphaseshift", "isingxx",
        "isingyy", "isingzz", "rot".

    .. note::

        f e la funzione forward del modulo, e la funzione forward del modello deve essere eseguita una volta per avere effetto.
        Il modello definito qui eredita da `pyvqnet.qnn.vqc.QModule`, che e una sottoclasse di `pyvqnet.nn.Module`.

    :param f: funzione forward.


    Example::

        import pyvqnet
        from pyvqnet.tensor import tensor

        from pyvqnet import tensor
        from pyvqnet.qnn.vqc import QMachine,op_history_summary
        from pyvqnet import kcomplex128
        from pyvqnet.tensor import adjoint
        import numpy as np


        from pyvqnet.qnn.vqc import *
        from pyvqnet.qnn.vqc import QModule
        from pyvqnet.tensor import QTensor, tensor
        import pyvqnet
        import numpy as np
        from pyvqnet.utils import set_random_seed

        set_random_seed(42)

        class QModel(QModule):
            def __init__(self, num_wires, dtype):
                super(QModel, self).__init__()

                self._num_wires = num_wires
                self._dtype = dtype
                self.qm = QMachine(num_wires, dtype=dtype)

                self.measure = MeasureAll(obs={
                                "Z0":1
                })

            @wrapper_merge_rotations
            def forward(self, x, *args, **kwargs):

                self.qm.reset_states(x.shape[0])
                
                rx(q_machine=self.qm, params=x[:, [1]], wires=(0, ))
                rx(q_machine=self.qm, params=x[:, [1]], wires=(0, ))
                rx(q_machine=self.qm, params=x[:, [1]], wires=(0, ))
                rot(q_machine=self.qm, params=x, wires=(1, ), use_dagger=True)
                rot(q_machine=self.qm, params=x, wires=(1, ), use_dagger=True)
                isingxy(q_machine=self.qm, params=x[:, [2]], wires=(0, 1))
                isingxy(q_machine=self.qm, params=x[:, [0]], wires=(0, 1))
                cnot(q_machine=self.qm, wires=[1, 2])
                ry(q_machine=self.qm, params=x[:, [1]], wires=(1, ))
                hadamard(q_machine=self.qm, wires=(2, ))
                crz(q_machine=self.qm, params=x[:, [2]], wires=(2, 0))
                ry(q_machine=self.qm, params=-x[:, [1]], wires=1)
                return self.measure(q_machine=self.qm)


        input_x = tensor.QTensor([[1, 2, 3], [1, 2, 3]], dtype=pyvqnet.kfloat64)

        input_x.requires_grad = True

        qunatum_model = QModel(num_wires=3, dtype=pyvqnet.kcomplex128)
        qunatum_model.use_merge_rotations = True
        batch_y = qunatum_model(input_x)
        print(op_history_summary(qunatum_model.qm.op_history))
        # ###################Summary#######################
        # qubits num: 3
        # gates: {'rx': 1, 'rot': 1, 'isingxy': 2, 'cnot': 1, 'hadamard': 1, 'crz': 1}
        # total gates: 7
        # total parameter gates: 5
        # total parameters: 7
        # #################################################



wrapper_compile
---------------------------------------------------------------

.. py:function:: pyvqnet.qnn.vqc.wrapper_compile(f,compile_rules=[commute_controlled_right, merge_rotations, single_qubit_ops_fuse])

    Usa regole di compilazione per ottimizzare i circuiti di QModule.

    .. note::

        f e la funzione forward del modulo, e la funzione forward del modello deve essere eseguita una volta per avere effetto.
        Il modello definito qui eredita da `pyvqnet.qnn.vqc.QModule`, che e una sottoclasse di `pyvqnet.nn.Module`.

    :param f: funzione forward.


    Example::

        from functools import partial

        from pyvqnet.qnn.vqc import op_history_summary
        from pyvqnet.qnn.vqc import QModule
        from pyvqnet import tensor
        from pyvqnet.qnn.vqc import QMachine, wrapper_compile

        from pyvqnet.qnn.vqc import pauliy

        from pyvqnet.qnn.vqc import QMachine, ry,rz, ControlledPhaseShift, \
            rx, S, rot, isingxy,CSWAP, PauliX, T, MeasureAll, RZ, CZ, PhaseShift, u3, cnot, cry, toffoli, cy
        from pyvqnet.tensor import QTensor, tensor
        import pyvqnet

        class QModel_before(QModule):
            def __init__(self, num_wires, dtype):
                super(QModel_before, self).__init__()

                self._num_wires = num_wires
                self._dtype = dtype
                self.qm = QMachine(num_wires, dtype=dtype)
                self.qm.set_save_op_history_flag(True)
                self.cswap = CSWAP(wires=(0, 2, 1))
                self.cz = CZ(wires=[0, 2])

                self.paulix = PauliX(wires=2)

                self.s = S(wires=0)

                self.ps = PhaseShift(has_params=True,
                                        trainable=True,
                                        wires=0,
                                        dtype=dtype)

                self.cps = ControlledPhaseShift(has_params=True,
                                                trainable=True,
                                                wires=(1, 0),
                                                dtype=dtype)
                self.t = T(wires=0)
                self.rz = RZ(has_params=True, wires=1, dtype=dtype)

                self.measure = MeasureAll(obs={
                                "Z0":1
                })

            def forward(self, x, *args, **kwargs):
                self.qm.reset_states(x.shape[0])
                self.cz(q_machine=self.qm)
                self.paulix(q_machine=self.qm)
                rx(q_machine=self.qm,wires=1,params = x[:,[0]])
                ry(q_machine=self.qm,wires=1,params = x[:,[1]])
                rz(q_machine=self.qm,wires=1,params = x[:,[2]])
                rot(q_machine=self.qm, params=x[:, 0:3], wires=(1, ), use_dagger=True)
                rot(q_machine=self.qm, params=x[:, 1:4], wires=(1, ), use_dagger=True)
                isingxy(q_machine=self.qm, params=x[:, [2]], wires=(0, 1))
                u3(q_machine=self.qm, params=x[:, 0:3], wires=1)
                self.s(q_machine=self.qm)
                self.cswap(q_machine=self.qm)
                cnot(q_machine=self.qm, wires=[0, 1])
                ry(q_machine=self.qm,wires=2,params = x[:,[1]])
                pauliy(q_machine=self.qm, wires=1)
                cry(q_machine=self.qm, params=1 / 2, wires=[0, 1])
                self.ps(q_machine=self.qm)
                self.cps(q_machine=self.qm)
                ry(q_machine=self.qm,wires=2,params = x[:,[1]])
                rz(q_machine=self.qm,wires=2,params = x[:,[2]])
                toffoli(q_machine=self.qm, wires=[0, 1, 2])
                self.t(q_machine=self.qm)

                cy(q_machine=self.qm, wires=(2, 1))
                ry(q_machine=self.qm,wires=1,params = x[:,[1]])
                self.rz(q_machine=self.qm)

                rlt = self.measure(q_machine=self.qm)

                return rlt
        class QModel(QModule):
            def __init__(self, num_wires, dtype):
                super(QModel, self).__init__()

                self._num_wires = num_wires
                self._dtype = dtype
                self.qm = QMachine(num_wires, dtype=dtype)

                self.cswap = CSWAP(wires=(0, 2, 1))
                self.cz = CZ(wires=[0, 2])

                self.paulix = PauliX(wires=2)

                self.s = S(wires=0)

                self.ps = PhaseShift(has_params=True,
                                        trainable=True,
                                        wires=0,
                                        dtype=dtype)

                self.cps = ControlledPhaseShift(has_params=True,
                                                trainable=True,
                                                wires=(1, 0),
                                                dtype=dtype)
                self.t = T(wires=0)
                self.rz = RZ(has_params=True, wires=1, dtype=dtype)

                self.measure = MeasureAll(obs={
                                "Z0":1
                })

            @partial(wrapper_compile)
            def forward(self, x, *args, **kwargs):
                self.qm.reset_states(x.shape[0])
                self.cz(q_machine=self.qm)
                self.paulix(q_machine=self.qm)
                rx(q_machine=self.qm,wires=1,params = x[:,[0]])
                ry(q_machine=self.qm,wires=1,params = x[:,[1]])
                rz(q_machine=self.qm,wires=1,params = x[:,[2]])
                rot(q_machine=self.qm, params=x[:, 0:3], wires=(1, ), use_dagger=True)
                rot(q_machine=self.qm, params=x[:, 1:4], wires=(1, ), use_dagger=True)
                isingxy(q_machine=self.qm, params=x[:, [2]], wires=(0, 1))
                u3(q_machine=self.qm, params=x[:, 0:3], wires=1)
                self.s(q_machine=self.qm)
                self.cswap(q_machine=self.qm)
                cnot(q_machine=self.qm, wires=[0, 1])
                ry(q_machine=self.qm,wires=2,params = x[:,[1]])
                pauliy(q_machine=self.qm, wires=1)
                cry(q_machine=self.qm, params=1 / 2, wires=[0, 1])
                self.ps(q_machine=self.qm)
                self.cps(q_machine=self.qm)
                ry(q_machine=self.qm,wires=2,params = x[:,[1]])
                rz(q_machine=self.qm,wires=2,params = x[:,[2]])
                toffoli(q_machine=self.qm, wires=[0, 1, 2])
                self.t(q_machine=self.qm)

                cy(q_machine=self.qm, wires=(2, 1))
                ry(q_machine=self.qm,wires=1,params = x[:,[1]])
                self.rz(q_machine=self.qm)

                rlt = self.measure(q_machine=self.qm)

                return rlt

        import pyvqnet
        import pyvqnet.tensor as tensor
        input_x = tensor.QTensor([[0.1, 0.2, 0.3, 0.4], [0.1, 0.2, 0.3, 0.4]],
                                    dtype=pyvqnet.kfloat64)

        input_x.requires_grad = True
        num_wires = 3
        qunatum_model = QModel(num_wires=num_wires, dtype=pyvqnet.kcomplex128)
        qunatum_model_before = QModel_before(num_wires=num_wires, dtype=pyvqnet.kcomplex128)

        batch_y = qunatum_model(input_x)
        batch_y = qunatum_model_before(input_x)

        flatten_oph_names = []

        print("before")

        print(op_history_summary(qunatum_model_before.qm.op_history))
        flatten_oph_names = []
        for d in qunatum_model.compiled_op_historys:
                if "compile" in d.keys():
                    oph = d["op_history"]
                    for i in oph:
                        n = i["name"]
                        w = i["wires"]
                        p = i["params"]
                        flatten_oph_names.append({"name":n,"wires":w, "params": p})
        print("after")
        print(op_history_summary(qunatum_model.qm.op_history))


        # ###################Summary#######################
        # qubits num: 3
        # gates: {'cz': 1, 'paulix': 1, 'rx': 1, 'ry': 4, 'rz': 3, 'rot': 2, 'isingxy': 1, 'u3': 1, 's': 1, 'cswap': 1, 'cnot': 1, 'pauliy': 1, 'cry': 1, 'phaseshift': 1, 'controlledphaseshift': 1, 'toffoli': 1, 't': 1, 'cy': 1}
        # total gates: 24
        # total parameter gates: 15
        # total parameters: 21
        # #################################################
            
        # after


        # ###################Summary#######################
        # qubits num: 3
        # gates: {'cz': 1, 'rot': 7, 'isingxy': 1, 'u3': 1, 'cswap': 1, 'cnot': 1, 'cry': 1, 'controlledphaseshift': 1, 'toffoli': 1, 'cy': 1}
        # total gates: 16
        # total parameter gates: 11
        # total parameters: 27
        # #################################################



QNSPSAOptimizer
---------------------------------------------------------------

.. py:class:: pyvqnet.qnn.vqc.qng.QNSPSAOptimizer(stepsize=1e-3,regularization=1e-3,finite_diff_step=1e-2,resamplings=1,blocking=True,history_length=5,seed=None)

    L'ottimizzatore Quantum Natural SPSA (QNSPSA) e un ottimizzatore stocastico del secondo ordine per circuiti quantistici che combina la discesa del gradiente con le informazioni del tensore metrico di Fubini-Study.
    Stima del gradiente utilizzando perturbazioni simmetriche (simile a SPSA):

    .. math::
        \widehat{\nabla f}(\mathbf{x}) \approx \frac{f(\mathbf{x}+\epsilon \mathbf{h})-f(\mathbf{x}-\epsilon \mathbf{h})}{2\epsilon}

    Calcola la metrica di Fubini-Study dalla misura di sovrapposizione degli stati:

    .. math::
        \widehat{\mathbf{g}}(\mathbf{x}) \approx \frac{\delta F}{8\epsilon^2}(\mathbf{h}_1\mathbf{h}_2^\intercal + \mathbf{h}_2\mathbf{h}_1^\intercal)
    .. math::
        \delta F = F(\mathbf{x}+\epsilon\mathbf{h}_1+\epsilon\mathbf{h}_2) - F(\mathbf{x}+\epsilon\mathbf{h}_1) - F(\mathbf{x}-\epsilon\mathbf{h}_1+\epsilon\mathbf{h}_2) + F(\mathbf{x}-\epsilon\mathbf{h}_1)

    dove δF misura le differenze di sovrapposizione delle valutazioni dei quattro circuiti.

    Regola di aggiornamento:

    .. math::
        \mathbf{x}^{(t+1)} = \mathbf{x}^{(t)} - \eta \widehat{\mathbf{g}}^{-1}(\mathbf{x}^{(t)})\widehat{\nabla f}(\mathbf{x}^{(t)})
    
    :param stepsize: Iperparametro del tasso di apprendimento definito dall'utente :math:`\eta` (default: 1e-3)
    :param regularization: Termine di regolarizzazione :math:`\beta` usato per il tensore metrico di Fubini-Study, per stabilita numerica (default: 1e-3)
    :param finite_diff_step: Dimensione del passo :math:`\epsilon` usata per calcolare i gradienti a differenze finite e il tensore metrico di Fubini-Study (default: 1e-2)
    :param resamplings: Numero medio di campioni per aggiornamento dei parametri (default: 1)
    :param blocking: Quando True, accetta solo aggiornamenti che risultano in una perdita non maggiore della somma dei valori di perdita precedenti (per agevolare la convergenza) (default: True)
    :param history_length: Quando ``blocking`` e True, la tolleranza e impostata alla media dei valori di costo ``history_length`` precedenti (default: 5)
    :param seed: seme per il campionamento casuale (default: None)

    .. py:method:: step(qmodel, *args, **kwargs)
        Aggiorna i parametri addestrabili una volta usando l'ottimizzatore.

        :param qmodel: Modello quantistico addestrabile
        :param args: QTensor addestrabile a lunghezza variabile per qmodel.
        :param kwargs: Argomenti con parole chiave a lunghezza variabile per qmodel.

        :return: Parametri aggiornati.

        Examples::

            from pyvqnet.tensor import QTensor,ones,randu
            from pyvqnet.qnn.vqc import rx,cry,QMachine,MeasureAll,QModule

            num_qubits = 2
            class QModuleDemo(QModule):
                def __init__(self, name=""):
                    super().__init__(name)
                    self.qm = QMachine(num_qubits)
                    self.ma = MeasureAll({"Z1 Z0":1})
                def forward(self,params):
                    qm = self.qm
                    qm.reset_states(1)
                    rx(qm, 0, params[0])
                    cry(qm, [0, 1], params[1])
                    return self.ma(qm)

            qmd = QModuleDemo()

            from pyvqnet.qnn.vqc.qnspsa import QNSPSAOptimizer
            params = QTensor([0.37454012, 0.95071431])

            params.requires_grad = True
            opt =  QNSPSAOptimizer(stepsize=5e-2,seed=1)
            for i in range(51):
                params = opt.step(qmd, params)
                loss =qmd(params)
                if i % 10 == 0:
                    print(f"Step {i}: cost = {loss}")


.. _benchmarks:

Benchmark di Addestramento su Dati Batch per Machine Learning Quantistico
==================================================================================

Test 1: Confronto del Gradiente su Dati Batch (VQNet / DeepQuantum / Pennylane)
------------------------------------------------------------------------------------

Nel machine learning quantistico, il calcolo del gradiente e un fattore chiave che influenza l'efficienza dei circuiti quantistici variazionali. Per valutare le prestazioni del calcolo del gradiente quantistico in diversi framework, questo documento ha condotto test benchmark su VQNet, Deepquantum e Pennylane sotto sistema Linux utilizzando GPU. I test sono stati eseguiti con diverse dimensioni di dati batch (batch size 16, 32), profondita del circuito (layer 2, 4) e numero di qubit (qubit 4, 8, 12, 16). La struttura del circuito era layer di codifica CNOT + RX + RZ + RX. Il tempo medio di esecuzione di ciascun framework su 10 esecuzioni e stato registrato. Deepquantum e Pennylane sono implementati basandosi sul backend GPU di Torch, mentre VQNet utilizza uno schema di accelerazione GPU sviluppato internamente.

.. image:: ./images/grad-benchmarks.png
   :width: 600 px
   :align: center

|



.. code-block::

    import time
    import json
    from functools import reduce
    import numpy as np
    import pennylane as qml

    import matplotlib.pyplot as plt
    def benchmark(f, *args, trials=10, sync_fn=None):
        time0 = time.time()
        r = f(*args)
        if sync_fn:
            sync_fn(r)
        time1 = time.time()
        for _ in range(trials):
            r = f(*args)
        if sync_fn:
            sync_fn(r)
        time2 = time.time()
        if trials > 0:
            time21 = (time2 - time1) / trials
        else:
            time21 = 0
        ts = (time1 - time0, time21)

        print('staging time: %.6f s' % ts[0])
        if trials > 0:
            print('running time: %.6f s' % ts[1])
        return r, ts[1]

    import torch
    import deepquantum as dq
    def grad_dq(b, n, l, trials=10):
        def get_grad_dq(params):
            if params.grad != None:
                params.grad.zero_()
            cir = dq.QubitCircuit(n)
            for j in range(l):

                for i in range(n - 1):
                    cir.cnot(i, i + 1)
                cir.rxlayer(encode=True)
                cir.rzlayer(encode=True)
                cir.rxlayer(encode=True)
            for w in range(n):
                cir.observable(basis='z',wires=w)
            cir.to("cuda:0")
            cir(data=params)
            exp = cir.expectation()
            exp.backward(torch.ones_like(exp))
            return params.grad


        return benchmark(get_grad_dq, torch.ones([b,3 * n * l], requires_grad=True,device="cuda:0"),
                         sync_fn=lambda _: torch.cuda.synchronize())


    def grad_pl_torchlayer(b, n, l,t):

        dev = qml.device("default.qubit", wires=n)

        @qml.qnode(dev, interface="torch")
        def circuit(inputs,weights):
            params = inputs

            for j in range(l):
                for i in range(n - 1):
                    qml.CNOT(wires=[i, i + 1])
                for i in range(n):
                    qml.RX(params[:,3 * n * j + i], i)
                for i in range(n):
                    qml.RZ(params[:,3 * n * j + i + n], i)
                for i in range(n):
                    qml.RX(params[:,3 * n * j + i + 2 * n], i)

            obs = reduce(lambda x, y: x @ y, [qml.PauliZ(i) for i in range(n)])
            y = qml.expval(obs)
            return y

        def get_grad_pl(params):
            params.grad = None
            weight_shapes = {"weights": 1}

            qlayer = qml.qnn.TorchLayer(circuit, weight_shapes = weight_shapes)
            qlayer.to("cuda:0")
            y = qlayer(params)

            y.backward(torch.ones_like(y))
            return params.grad
        return benchmark(get_grad_pl, torch.ones([b,3 * n * l],device="cuda:0", requires_grad=True),trials=t,
                         sync_fn=lambda _: torch.cuda.synchronize())

    def grad_pyvqnet_vqc(b, n, l, t):
        from pyvqnet.qnn.vqc import QMachine,cnot,rx,rz,ry,MeasureAll
        from pyvqnet.tensor import tensor
        import pyvqnet
        pyvqnet.backends.set_backend("pyvqnet-ad")
        def pqctest(qm,param):
            param.zero_grad()
            qm.reset_states(param.shape[0])
            for j in range(l):
                for i in range(n - 1):
                    cnot(qm,[i, i + 1])
                for i in range(n):
                    rx(qm, i, param[:,3 * n * j + i])
                for i in range(n):
                    rz(qm, i, param[:,3 * n * j + i + n])
                for i in range(n):
                    rx(qm, i, param[:,3 * n * j + i + 2 * n])
            pauli_str =""
            for position in range(n):
                pauli_str += "Z" + str(position)+" "
            p_dict = {pauli_str:1}
            ma = MeasureAll(obs=p_dict)

            y = ma(qm)
            y.backward()

            return param.grad

        def get_grad(qm,values):
            r = pqctest(qm,values)

            return r

        input = tensor.ones([b,3 * n * l],device=pyvqnet.DEV_GPU)
        input.requires_grad = True
        qm = QMachine(n)
        qm.toGPU(pyvqnet.DEV_GPU)
        return benchmark(get_grad, qm, input, trials=t,
                         sync_fn=lambda r: r.numpy())



    N_LIST = [ 4,8,12,16 ]
    L_LIST =[2,4]
    B_LIST =[16, 32]
    def test_1():
        results ={}
        config_key = []
        for n in N_LIST:
            for l in L_LIST:
                for b in B_LIST:
                    for t in [10,]:
                        config_key.append(str(b) + '-' + str(n) + '-' + str(l))

                        dqr, ts1 = grad_pl_torchlayer(b,n, l, t)
                        results[str(b) + '-' + str(n) + '-' + str(l) + '-' + 'grad' + '-pl'] = ts1
                        print(f'PL batch={b} qubits={n} layers={l} trials={t}, grad avg={ts1:.4f}s')

                        dqr, ts3 = grad_dq(b, n, l, t)
                        results[str(b) + '-' + str(n) + '-' + str(l) + '-' + 'grad' + '-dq'] = ts3
                        print(f'DQ batch={b} qubits={n} layers={l} trials={t}, grad avg={ts3:.4f}s')


                        result, ts2 = grad_pyvqnet_vqc(b, n, l, t)
                        results[str(b) + '-' + str(n) + '-' + str(l) + '-' + 'grad' + '-pyvqnet'] = ts2
                        print(f'pyVQNet batch={b} qubits={n} layers={l} trials={t}, grad avg={ts2:.4f}s')


        with open('gradient_results.data', 'w') as f:
            json.dump(results, f)

        with open('gradient_results.data', 'r') as f:
            results = json.load(f)

        data = results

        sub_w  = 2
        sub_l = int(len(N_LIST)/2)
        assert len(N_LIST)%2==0
        fig, axes = plt.subplots(sub_w, sub_l, figsize=(12, 10))
        ax_i=0
        for n in N_LIST:
            config_key = []
            for l in L_LIST:
                for b in B_LIST:
                    for t in [10,]:
                        config_key.append(str(b) + '-' + str(n) + '-' + str(l))
            groups = config_key
            pl_times = [data[f'{group}-grad-pl'] for group in groups]
            dq_times = [data[f'{group}-grad-dq'] for group in groups]
            pyvqnet_times = [data[f'{group}-grad-pyvqnet'] for group in groups]


            x = np.arange(len(groups))
            width = 0.15

            ax = axes[int(ax_i/sub_w), ax_i %sub_w]
            ax_i +=1
            #fig, ax = plt.subplots(figsize=(10, 6))


            rects2 = ax.bar(x , pyvqnet_times, width, label='pyvqnet')
            rects3 = ax.bar(x + width, dq_times, width, label='deepquantum')
            rects1 = ax.bar(x - width, pl_times, width, label='pennylane')

            ax.set_ylabel('cost time (sec)')
            ax.set_title(f'Gradient benchmarks of {n}-Qubit Variational Quantum Circuit.')
            ax.set_xticks(x)
            ax.set_xlabel('batchsize-qubits-layer_depth')
            ax.set_xticklabels(groups)
            ax.legend()
            ax.grid(axis='y', linestyle='--', alpha=0.7)

            def autolabel(rects):

                for rect in rects:
                    height = rect.get_height()
                    ax.annotate(f'{height:.2f}',
                                xy=(rect.get_x() + rect.get_width() / 2, height),
                                xytext=(0, 3),  # 3 points vertical offset
                                textcoords="offset points",
                                ha='center', va='bottom')

            autolabel(rects1)
            autolabel(rects2)
            autolabel(rects3)
        fig.tight_layout()
        plt.savefig(f"grad-benchmarks.png")

    test_1()



Test 2: Confronto del Gradiente VQC a 10 Qubit (con TorchQuantum)
------------------------------------------------------------------------------------

Questo test e basato sul circuito VQC a 10 qubit e 10 layer utilizzato nell'articolo del modello large di Origin Quantum. Confronta le prestazioni di calcolo del gradiente di cinque framework: VQNet, TorchQuantum (TQ), DeepQuantum (DQ), Pennylane (PL) e MindQuantum (MQ). La struttura del circuito e:

  RY(data) -> [RY(param) -> CRZ(param) -> RY(param) -> CRZ(param)] x L

Ogni layer contiene 40 parametri (4 gruppi x 10 qubit), per un totale di 400 parametri. Le dimensioni del batch vanno da 1 a 1024. Conteggio prove: VQNet / TQ / DQ eseguono 20 prove ciascuno, PL / MQ eseguono 2 prove ciascuno (gli ultimi due sono piu lenti con dati batch e limitati a 2 prove per risparmiare tempo).

.. image:: ./images/grad_benchmarks_10q_ry_crz.png
   :width: 600 px
   :align: center

|

.. code-block:: python

    from pyvqnet.tensor import tensor
    from pyvqnet.qnn.vqc import RX, RY, RZ, crz, PauliX, PauliY, PauliZ, paulix, pauliy, pauliz, rx, ry, rz, MeasureAll, fused_multi_crz
    from pyvqnet.nn import ParameterDict, Parameter
    from pyvqnet.qnn.vqc import QModule, QMachine
    import numpy as np
    import pyvqnet
    import time

    QuantumDevice = QMachine
    class Encoder(QModule):

        def __init__(self):
            super().__init__()
            pass

        def forward(self, x, qdev):
            raise NotImplementedError

    op_name_dict = {
        "x": PauliX,
        "y": PauliY,
        "z": PauliZ,
        "rx": RX,
        "ry": RY,
        "rz": RZ
    }

    func_name_dict = {
        "x": paulix,
        "y": pauliy,
        "z": pauliz,
        "rx": rx,
        "ry": ry,
        "rz": rz
    }

    class GeneralEncoder(Encoder):
        """func_list list of dict

        """

        def __init__(self, func_list):
            super().__init__()
            self.func_list = func_list

        def forward(self, x, qdev):
            for info in self.func_list:
                if op_name_dict[info["func"]].num_params > 0:
                    params = x[:, info["input_idx"]]
                else:
                    params = None

                func_name_dict[info["func"]](qdev,
                                             wires=info["wires"],
                                             params=params)

        def __call__(self, *args, **kwargs):
            return self.forward(*args, **kwargs)


    class VQC_new(QModule):
        """VQC using fused_multi_crz - one parameter vector per layer"""

        def __init__(self, n_wires: int = 4, n_qlayers: int = 1):
            super().__init__()
            self.n_wires = n_wires
            self.n_qlayers = n_qlayers
            self.dev = QuantumDevice(self.n_wires)
            enc_cnt = list()
            for i in range(self.n_wires):
                cnt = {'input_idx': [i], 'func': 'ry', 'wires': [i]}
                enc_cnt.append(cnt)

            self.encoder = GeneralEncoder(enc_cnt)
            self._use_vqnet = True

            self.params_ry1_dct = ParameterDict()
            self.params_ry2_dct = ParameterDict()
            self.params_crx1_dct = ParameterDict()
            self.params_crx2_dct = ParameterDict()

            for k in range(self.n_qlayers):
                self.params_crx1_dct[str(k)] = Parameter([self.n_wires])
                self.params_crx2_dct[str(k)] = Parameter([self.n_wires])
                for i in range(self.n_wires):
                    self.params_ry1_dct[str(i + k * self.n_wires)] = Parameter([1])
                    self.params_ry2_dct[str(i + k * self.n_wires)] = Parameter([1])

            self.use_pq3 = False

            obs_list = []
            for i in range(self.n_wires):
                obs_list.append({f"Z{i}": 1})

            self.measure = MeasureAll(obs=obs_list)

        def forward(self, x):
            q_device = self.dev
            q_device.reset_states(x.shape[0])

            self.encoder(x, q_device)

            for k in range(self.n_qlayers):
                for i in range(self.n_wires):
                    ry(q_machine=q_device, wires=i, params=self.params_ry1_dct[str(i + k * self.n_wires)])

                obj_qubits = [(i + 1) % self.n_wires for i in range(self.n_wires - 1, -1, -1)]
                ctrls = list(range(self.n_wires - 1, -1, -1))
                fused_multi_crz(
                    q_machine=q_device,
                    params=self.params_crx1_dct[str(k)],
                    obj_qubits=obj_qubits,
                    ctrls=ctrls)

                for i in range(self.n_wires):
                    ry(q_machine=q_device, params=self.params_ry2_dct[str(i + k * self.n_wires)], wires=i)

                obj_qubits = [(i - 1) % self.n_wires for i in [self.n_wires - 1] + list(range(self.n_wires - 1))]
                ctrls = [self.n_wires - 1] + list(range(self.n_wires - 1))
                fused_multi_crz(
                    q_machine=q_device,
                    params=self.params_crx2_dct[str(k)],
                    obj_qubits=obj_qubits,
                    ctrls=ctrls)

            if self.use_pq3:
                return x
            else:
                return self.measure(q_device)


    class VQC(QModule):
        """VQC using individual crz gates - one parameter per gate"""

        def __init__(self, n_wires: int = 4, n_qlayers: int = 1):
            super().__init__()
            self.n_wires = n_wires
            self.n_qlayers = n_qlayers
            self.dev = QuantumDevice(self.n_wires)
            enc_cnt = list()
            for i in range(self.n_wires):
                cnt = {'input_idx': [i], 'func': 'ry', 'wires': [i]}
                enc_cnt.append(cnt)

            self.encoder = GeneralEncoder(enc_cnt)
            self._use_vqnet = True

            self.params_ry1_dct = ParameterDict()
            self.params_ry2_dct = ParameterDict()
            self.params_crx1_dct = ParameterDict()
            self.params_crx2_dct = ParameterDict()

            for k in range(self.n_qlayers):
                for i in range(self.n_wires):
                    self.params_crx1_dct[str(i + k * self.n_wires)] = Parameter([1])
                    self.params_crx2_dct[str(i + k * self.n_wires)] = Parameter([1])
                    self.params_ry1_dct[str(i + k * self.n_wires)] = Parameter([1])
                    self.params_ry2_dct[str(i + k * self.n_wires)] = Parameter([1])

            self.use_pq3 = False

            obs_list = []
            for i in range(self.n_wires):
                obs_list.append({f"Z{i}": 1})

            self.measure = MeasureAll(obs=obs_list)

        def forward(self, x):
            q_device = self.dev
            q_device.reset_states(x.shape[0])

            self.encoder(x, q_device)

            for k in range(self.n_qlayers):
                for i in range(self.n_wires):
                    ry(q_machine=q_device, wires=i, params=self.params_ry1_dct[str(i + k * self.n_wires)])

                for i in range(self.n_wires - 1, -1, -1):
                    crz(
                        q_machine=q_device,
                        params=self.params_crx1_dct[str(i + k * self.n_wires)],
                        wires=[i, (i + 1) % self.n_wires])

                for i in range(self.n_wires):
                    ry(q_machine=q_device, params=self.params_ry2_dct[str(i + k * self.n_wires)], wires=i)

                for i in [self.n_wires - 1] + list(range(self.n_wires - 1)):
                    crz(
                        q_machine=q_device,
                        params=self.params_crx2_dct[str(i + k * self.n_wires)],
                        wires=[i, (i - 1) % self.n_wires])

            if self.use_pq3:
                return x
            else:
                return self.measure(q_device)


    def benchmark(f, *args, trials=10, sync_fn=None):
        time0 = time.time()
        r = f(*args)
        if sync_fn:
            sync_fn(r)
        time1 = time.time()
        for _ in range(trials):
            r = f(*args)
        if sync_fn:
            sync_fn(r)
        time2 = time.time()
        if trials > 0:
            time21 = (time2 - time1) / trials
        else:
            time21 = 0
        ts = (time1 - time0, time21)
        print('staging time: %.6f s' % ts[0])
        if trials > 0:
            print('running time: %.6f s' % ts[1])
        return r, ts


    def grad_pyvqnet_vqc_new(b, n, l, trials=10):
        """Test VQC_new (fused_multi_crz)"""
        pyvqnet.utils.set_random_seed(42)
        layer = VQC_new(n, l)
        layer.toGPU(pyvqnet.DEV_GPU)

        def get_grad(values):
            r = layer(values)
            r.backward()
            return values.grad

        input = tensor.ones([b, n], device=pyvqnet.DEV_GPU)
        input.requires_grad = True
        return benchmark(get_grad, input, trials=trials,
                         sync_fn=lambda r: r.numpy())


    def grad_pyvqnet_vqc(b, n, l, trials=10):
        pyvqnet.utils.set_random_seed(42)
        layer = VQC(n, l)
        layer.toGPU(pyvqnet.DEV_GPU)

        def get_grad(values):
            r = layer(values)
            r.backward()
            return values.grad

        input = tensor.ones([b, n], device=pyvqnet.DEV_GPU)
        input.requires_grad = True
        return benchmark(get_grad, input, trials=trials,
                         sync_fn=lambda r: r.numpy())


    def grad_tq_vqc_with_params(b, n, l, trials=10):
        """TorchQuantum VQC benchmark"""
        import torchquantum as tq
        import torch
        import torch.cuda

        class VQC_TQ(tq.QuantumModule):
            """TorchQuantum VQC matching VQNet's VQC structure"""

            def __init__(self, n_wires: int = 4, n_qlayers: int = 1):
                super().__init__()
                self.n_wires = n_wires
                self.n_qlayers = n_qlayers

                enc_cnt = list()
                for i in range(self.n_wires):
                    cnt = {'input_idx': [i], 'func': 'ry', 'wires': [i]}
                    enc_cnt.append(cnt)
                self.encoder = tq.GeneralEncoder(enc_cnt)

                self.params_ry1_dct = tq.QuantumModuleDict()
                self.params_ry2_dct = tq.QuantumModuleDict()
                self.params_crx1_dct = tq.QuantumModuleDict()
                self.params_crx2_dct = tq.QuantumModuleDict()

                for k in range(self.n_qlayers):
                    for i in range(self.n_wires):
                        self.params_ry1_dct[str(i + k * self.n_wires)] = tq.RY(has_params=True, trainable=True)
                        self.params_crx1_dct[str(i + k * self.n_wires)] = tq.CRZ(has_params=True, trainable=True)
                        self.params_ry2_dct[str(i + k * self.n_wires)] = tq.RY(has_params=True, trainable=True)
                        self.params_crx2_dct[str(i + k * self.n_wires)] = tq.CRZ(has_params=True, trainable=True)

                self.measure = tq.MeasureMultipleTimes([{'wires': range(self.n_wires), 'observables': ['z'] * self.n_wires}])

                from torchquantum import QuantumDevice as TQQuantumDevice
                self.dev = TQQuantumDevice(self.n_wires)

            def forward(self, x: torch.Tensor):
                q_device = self.dev
                q_device.reset_states(x.shape[0])
                self.encoder(q_device, x)

                for k in range(self.n_qlayers):
                    for i in range(self.n_wires):
                        self.params_ry1_dct[str(i + k * self.n_wires)](q_device, wires=i)

                    for i in range(self.n_wires - 1, -1, -1):
                        self.params_crx1_dct[str(i + k * self.n_wires)](q_device, wires=[i, (i + 1) % self.n_wires])

                    for i in range(self.n_wires):
                        self.params_ry2_dct[str(i + k * self.n_wires)](q_device, wires=i)

                    for i in [self.n_wires - 1] + list(range(self.n_wires - 1)):
                        self.params_crx2_dct[str(i + k * self.n_wires)](q_device, wires=[i, (i - 1) % self.n_wires])

                return self.measure(q_device)

        torch.manual_seed(42)
        layer = VQC_TQ(n, l)
        layer.to("cuda:0")

        def get_grad(values):
            r = layer(values)
            r.backward(torch.ones_like(r))
            return values.grad

        input = torch.ones([b, n], device="cuda:0")
        input.requires_grad = True
        return benchmark(get_grad, input, trials=trials,
                         sync_fn=lambda _: torch.cuda.synchronize())

    # ──────────────────────────────────────────────
    # PennyLane benchmark
    # ──────────────────────────────────────────────

    def grad_pl_vqc(b, n, l, trials=1):
        """PennyLane VQC (default.qubit) matching VQNet VQC structure."""
        import pennylane as qml
        from functools import reduce
        import torch
        dev = qml.device("default.qubit", wires=n)

        @qml.qnode(dev, interface="torch")
        def circuit(inputs, weights_ry1, weights_crz1, weights_ry2, weights_crz2):
            for j in range(l):
                for i in range(n):
                    qml.RY(inputs[:, i], wires=i)
                for i in range(n):
                    qml.RY(weights_ry1[j, i], wires=i)
                for i in range(n - 1, -1, -1):
                    qml.CRZ(weights_crz1[j, i], wires=[i, (i + 1) % n])
                for i in range(n):
                    qml.RY(weights_ry2[j, i], wires=i)
                for i in [n - 1] + list(range(n - 1)):
                    qml.CRZ(weights_crz2[j, i], wires=[i, (i - 1) % n])

            obs = reduce(lambda x, y: x @ y, [qml.PauliZ(i) for i in range(n)])
            return qml.expval(obs)

        weight_shapes = {
            "weights_ry1": (l, n),
            "weights_crz1": (l, n),
            "weights_ry2": (l, n),
            "weights_crz2": (l, n),
        }

        def get_grad_pl(inputs):
            torch.manual_seed(42)
            qlayer = qml.qnn.TorchLayer(circuit, weight_shapes=weight_shapes)
            qlayer.to("cuda:0")
            y = qlayer(inputs)
            y.backward(torch.ones_like(y))
            return inputs.grad

        params = torch.ones([b, n], device="cuda:0", requires_grad=True)
        return benchmark(get_grad_pl, params, trials=trials)


    # ──────────────────────────────────────────────
    # DeepQuantum benchmark
    # ──────────────────────────────────────────────

    def grad_dq_vqc(b, n, l, trials=10):
        """DeepQuantum VQC matching VQNet VQC structure."""
        import deepquantum as dq
        import torch
        import torch.cuda
        def get_grad_dq(input_data):
            cir = dq.QubitCircuit(n, reupload=True)
            for j in range(l):
                for i in range(n):
                    cir.ry(wires=i, encode=True)
                for i in range(n):
                    cir.ry(wires=i)
                for i in range(n - 1, -1, -1):
                    cir.crz(control=i, target=(i + 1) % n)
                for i in range(n):
                    cir.ry(wires=i)
                for i in [n - 1] + list(range(n - 1)):
                    cir.crz(control=i, target=(i - 1) % n)
            for w in range(n):
                cir.observable(basis='z', wires=w)
            cir.to("cuda:0")
            cir(data=input_data)
            exp = cir.expectation()
            exp.backward(torch.ones_like(exp))
            return input_data.grad

        params = torch.ones([b, n], device="cuda:0", requires_grad=True)
        return benchmark(get_grad_dq, params, trials=trials,
                         sync_fn=lambda _: torch.cuda.synchronize())


    # ──────────────────────────────────────────────
    # MindQuantum benchmark
    # ──────────────────────────────────────────────

    def grad_mq_vqc(b, n, l, trials=1):
        """MindQuantum VQC with mqvector_gpu (cuQuantum) backend."""
        from mindquantum.core.circuit import Circuit
        from mindquantum.core.gates import RY, RZ, X
        from mindquantum.core.operators import Hamiltonian, QubitOperator
        from mindquantum.simulator import Simulator

        total_circuit = Circuit()
        for j in range(l):
            layer_enc = Circuit()
            for i in range(n):
                layer_enc += RY(f'enc_{j}_{i}').on(i)
            layer_enc.as_encoder()

            layer_ans = Circuit()
            for i in range(n):
                layer_ans += RY(f'ry1_{j}_{i}').on(i)
            for i in range(n - 1, -1, -1):
                tgt = (i + 1) % n
                ctrl = i
                p = f'crz1_{j}_{i}'
                layer_ans += RZ({p: 0.5}).on(tgt)
                layer_ans += X.on(tgt, ctrl)
                layer_ans += RZ({p: -0.5}).on(tgt)
                layer_ans += X.on(tgt, ctrl)
            for i in range(n):
                layer_ans += RY(f'ry2_{j}_{i}').on(i)
            for i in [n - 1] + list(range(n - 1)):
                tgt = (i - 1) % n
                ctrl = i
                p = f'crz2_{j}_{i}'
                layer_ans += RZ({p: 0.5}).on(tgt)
                layer_ans += X.on(tgt, ctrl)
                layer_ans += RZ({p: -0.5}).on(tgt)
                layer_ans += X.on(tgt, ctrl)
            layer_ans.as_ansatz()

            total_circuit += layer_enc + layer_ans

        obs = ' '.join(f'Z{i}' for i in range(n))
        ham = Hamiltonian(QubitOperator(obs))
        sim = Simulator('mqvector_gpu', n)
        grad_ops = sim.get_expectation_with_grad(ham, total_circuit)
        n_ansatz_params = 4 * n * l
        ansatz_data = np.ones(n_ansatz_params, dtype=np.float32)

        def get_grad_mq(input_data):
            encoder_data = np.tile(input_data, (1, l)).astype(np.float32)
            _, g_enc, _ = grad_ops(encoder_data, ansatz_data)
            return g_enc

        def sync_mq(g_enc):
            g_enc_real = np.asarray(g_enc.real, dtype=np.float32)
            g = np.zeros((b, n), dtype=np.float32)
            for j in range(l):
                g += g_enc_real[:, 0, j * n : (j + 1) * n]
            return g

        input_data = np.ones((b, n), dtype=np.float32)
        return benchmark(get_grad_mq, input_data, trials=trials,
                         sync_fn=sync_mq)


    # ──────────────────────────────────────────────
    # Plotting
    # ──────────────────────────────────────────────

    def _parse_trials(results, fw):
        """Extract trials count from results keys like '*-grad-{fw}-t{N}'."""
        for k in results:
            if f'-grad-{fw}-t' in k:
                return k.split('-t')[-1]
        return '?'


    def plot_results(results, output_path="grad_benchmarks_10q_ry_crz.png"):
        """
        Line chart: x = batch size, y = running time (log scale).
        Keys format: "{batch}-{n}-{l}-grad-{framework}-t{trials}".
        """
        import matplotlib.pyplot as plt

        frameworks = ['pyvqnet', 'new', 'torchquantum', 'pl', 'dq', 'mq']
        labels_base = {
            'pyvqnet': 'pyVQNet',
            'new': 'pyVQNet (fused CRZ)',
            'torchquantum': 'TorchQuantum',
            'pl': 'PennyLane',
            'dq': 'DeepQuantum',
            'mq': 'MindQuantum',
        }
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']

        batch_sizes = sorted({int(k.split('-')[0]) for k in results})
        n_qubits = sorted({k.split('-')[1] for k in results})
        n_layers = sorted({k.split('-')[2] for k in results})
        n_q = int(n_qubits[0]) if n_qubits else 10
        n_l = int(n_layers[0]) if n_layers else 10

        fig, ax = plt.subplots(figsize=(10, 6))

        for fw, color in zip(frameworks, colors):
            trials_str = _parse_trials(results, fw)
            label = f"{labels_base[fw]} (t={trials_str})"
            times = []
            for bs in batch_sizes:
                candidates = [k for k in results if k.startswith(f'{bs}-{n_q}-{n_l}-grad-{fw}')]
                if candidates:
                    times.append(results[candidates[0]][1])
                else:
                    times.append(None)
            valid = [(bs, t) for bs, t in zip(batch_sizes, times) if t is not None]
            if valid:
                xs, ys = zip(*valid)
                ax.plot(xs, ys, marker='o', label=label, color=color, linewidth=2)

        ax.set_xlabel('Batch Size', fontsize=13)
        ax.set_ylabel('Running Time (s) — log scale', fontsize=13)
        ax.set_title(f'VQC Gradient Benchmark (n_qubits={n_q}, n_layers={n_l})', fontsize=14)
        ax.set_xticks(batch_sizes)
        ax.set_yscale('log')
        ax.legend(fontsize=11, loc='upper left', bbox_to_anchor=(0.02, 0.98))
        ax.grid(True, alpha=0.3)

        fig.tight_layout()
        fig.savefig(output_path, dpi=150)

    def test_3():
        """
        Run all benchmarks across multiple configs.
        Fast frameworks (pyvqnet, TQ, DQ): trials=100
        Slow frameworks (PL, MQ):         trials=2
        JSON keys include trials count so results are distinguishable.
        """
        import os
        import json
        json_path = "compare_grad_calc_results.json"
        if os.path.exists(json_path):
            print(f"{json_path} already exists, loading and plotting directly...")
            with open(json_path) as f:
                results = json.load(f)
            print()
            print("=== Loaded Results ===")
            for key, ts in results.items():
                print(f"{key}: staging={ts[0]:.4f}s, running={ts[1]:.4f}s")
            plot_results(results)
            return

        results = {}
        n_list = [10,]
        l_list = [10, ]
        b_list = [1024,512,256,128,64]
        t_fast, t_slow = 20, 2

        for n in n_list:
            for l in l_list:
                for b in b_list:
                    print(str(b) + '-' + str(n) + '-' + str(l) + '-' + 'grad')
                    print("grad_pyvqnet_vqc")
                    pyvqnet.utils.set_random_seed(42)
                    result2, ts = grad_pyvqnet_vqc(b, n, l, trials=t_fast)
                    results[str(b) + '-' + str(n) + '-' + str(l) + '-' + 'grad' + f'-pyvqnet-t{t_fast}'] = ts

                    print(str(b) + '-' + str(n) + '-' + str(l) + '-' + 'grad')
                    print("grad_pyvqnet_vqc_new")
                    pyvqnet.utils.set_random_seed(42)
                    result1, ts = grad_pyvqnet_vqc_new(b, n, l, trials=t_fast)
                    results[str(b) + '-' + str(n) + '-' + str(l) + '-' + 'grad' + f'-new-t{t_fast}'] = ts

                    print(str(b) + '-' + str(n) + '-' + str(l) + '-' + 'grad')
                    print("grad_torchquantum_vqc")
                    result_tq, ts = grad_tq_vqc_with_params(b, n, l, trials=t_fast)
                    results[str(b) + '-' + str(n) + '-' + str(l) + '-' + 'grad' + f'-torchquantum-t{t_fast}'] = ts

                    print(str(b) + '-' + str(n) + '-' + str(l) + '-' + 'grad')
                    print("grad_pennylane_vqc")
                    _, ts = grad_pl_vqc(b, n, l, trials=t_slow)
                    results[f'{b}-{n}-{l}-grad-pl-t{t_slow}'] = ts

                    print(str(b) + '-' + str(n) + '-' + str(l) + '-' + 'grad')
                    print("grad_deepquantum_vqc")
                    _, ts = grad_dq_vqc(b, n, l, trials=t_fast)
                    results[f'{b}-{n}-{l}-grad-dq-t{t_fast}'] = ts

                    print(str(b) + '-' + str(n) + '-' + str(l) + '-' + 'grad')
                    print("grad_mindquantum_vqc")
                    _, ts = grad_mq_vqc(b, n, l, trials=t_slow)
                    results[f'{b}-{n}-{l}-grad-mq-t{t_slow}'] = ts

        print("\n=== All Results ===")
        for key, ts in results.items():
            print(f"{key}: staging={ts[0]:.4f}s, running={ts[1]:.4f}s")

        with open('compare_grad_calc_results.json', 'w') as f:
            json.dump(results, f)
        print("Results saved to compare_grad_calc_results.json")

        # Also generate plot
        plot_results(results)


    test_3()


+------------------+----------------+
| Project          | Specification  |
+==================+================+
| CPU              | i9-9900K       |
+------------------+----------------+
| GPU              | GTX3090        |
+------------------+----------------+
| CUDA             | 12.6           |
+------------------+----------------+
| RAM              | 64GB           |
+------------------+----------------+
| deepquantum      | 4.5.0          |
+------------------+----------------+
| mindquantum      | 0.12.0         |
+------------------+----------------+
| pennylane        | 0.42.3         |
+------------------+----------------+
| pyvqnet          | 2.18.0         |
+------------------+----------------+
| torch            | 2.12.0         |
+------------------+----------------+
| torchquantum     | 0.2.0          |
+------------------+----------------+
