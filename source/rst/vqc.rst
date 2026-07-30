.. _vqc_api:

API de Circuitos Quânticos Variacionais com Autograd
******************************************************************************

O VQNet é baseado na construção de operadores diferenciais automáticos e algumas portas lógicas quânticas comuns, circuitos quânticos e métodos de medição. A diferenciação automática pode ser usada para calcular gradientes em vez do método de deslocamento de parâmetros do circuito quântico.
Podemos usar operadores VQC para formar redes neurais complexas como outros `Modules`. A máquina virtual `QMachine` precisa ser definida em `Module`, e os `states` na máquina precisam ser redefinidos com reset_states baseado no batchsize de entrada. Veja o exemplo a seguir para detalhes.

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
            #VQC baseado em RZ no bit 0
            self.encode1 = RZ(wires=0)
            #VQC baseado em RZ no bit 1
            self.encode2 = RZ(wires=1)
            #Medição de probabilidade baseada em VQC nos bits 0, 2
            self.measure = Probability(wires=[0,2])
            #Dispositivo quântico QMachine, usa 4 bits.
            self.device = QMachine(4)
        def forward(self, x, *args, **kwargs):
            #Os estados devem ser redefinidos com o mesmo batchsize da entrada.
            self.device.reset_states(x.shape[0])
            y = self.linearx(x)
            #Codifica a entrada para a porta RZ. Note que a entrada deve ter shape [batchsize,1]
            self.encode1(params = y[:, [0]],q_machine = self.device,)
            #Codifica a entrada para a porta RZ. Note que a entrada deve ter shape [batchsize,1]
            self.encode2(params = y[:, [1]],q_machine = self.device,)
            self.ansatz(q_machine =self.device)
            return self.measure(q_machine =self.device)

    bz=3
    inputx = tensor.arange(1.0,bz*4+1).reshape([bz,4])
    inputx.requires_grad= True
    #Define como outros Modules
    qlayer = QM()
    #Prequel
    y = qlayer(inputx)

    y.backward()
    print(y)


O exemplo a seguir demonstra a computação quântica variacional em GPU (incluindo codificação de dados e circuitos variacionais parametrizados):

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
            #VQC baseado em RZ no bit 0
            self.encode1 = RZ(wires=0)
            #VQC baseado em RZ no bit 1
            self.encode2 = RZ(wires=1)
            #RZ com parâmetros treináveis has_params = True, trainable = True
            self.vqc = RZ(has_params = True,trainable = True,wires=1)
            #Medição de probabilidade baseada em VQC nos bits 0, 2
            self.measure = Probability(wires=[0,2])
            #Dispositivo quântico QMachine, usa 4 bits.
            self.device = QMachine(4)
        def forward(self, x, *args, **kwargs):
            #Os estados devem ser redefinidos com o mesmo batchsize da entrada.
            self.device.reset_states(x.shape[0])
            y = self.linearx(x)
            #Codifica a entrada para a porta RZ. Note que a entrada deve ter shape [batchsize,1]
            self.encode1(params = y[:, 0],q_machine = self.device,)
            #Codifica a entrada para a porta RZ. Note que a entrada deve ter shape [batchsize,1]
            self.encode2(params = y[:, 1],q_machine = self.device,)
            #Circuito variacional composto por portas RZ, será incluído no treinamento.
            self.vqc(q_machine =self.device)
            self.ansatz(q_machine =self.device)
            return self.measure(q_machine =self.device)

    bz =3
    #cria tensor na GPU
    inputx = tensor.arange(1.0,bz*4+1,device=DEV_GPU).reshape([bz,4])
    inputx.requires_grad= True
    #Define como outros Modules
    qlayer = QM()
    #move o módulo para GPU
    qlayer = qlayer.to(DEV_GPU)
    #Forward
    y = qlayer(inputx)
    #Backward
    y.backward()
    print(y)



Simulador
=======================================

QMachine
-------------------------------

.. py:class:: pyvqnet.qnn.vqc.QMachine(num_wires, dtype=pyvqnet.kcomplex64)

    Uma classe simuladora para computação quântica variacional, incluindo vetores de estado cujo atributo states é um circuito quântico.

    :param num_wires: número de qubits.
    :param dtype: o tipo de dado dos dados calculados, o padrão é pyvqnet.kcomplex64, e a precisão do parâmetro correspondente é pyvqnet.kfloat32

    :return: Saída QMachine.

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
    
        Reinicializa o estado inicial no simulador e o transmite para
        as dimensões (batchsize,[2]**num_qubits) para se adaptar ao treinamento com dados em lote.

        :param batchsize: tamanho do lote de processamento.


Portas Quânticas e Operação de Portas Quânticas
===============================================


i
-------------------------------

.. py:function:: pyvqnet.qnn.vqc.i(q_machine, wires, params=None,  use_dagger=False)

    Aplica a porta lógica quântica I aos vetores de estado em ``q_machine``.

    :param q_machine: dispositivo de máquina virtual quântica.
    :param wires: índice do qubit.
    :param params: matriz de parâmetros, o padrão é None. Para uma função de operação de porta lógica com p parâmetros, a dimensão do parâmetro de entrada precisa ser [1,p] ou [p].
    :param use_dagger: se deve usar a transposta conjugada, o padrão é False.


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
    
    Define uma classe de porta lógica I.

    :param has_params: Se possui parâmetros. Portas como RX, RY precisam ser configuradas como True; as sem parâmetros precisam ser False. O padrão é False.
    :param trainable: Se contém parâmetros a serem treinados. Se a camada usa dados de entrada externos para construir uma matriz de porta lógica, defina como False. Se contém parâmetros a serem treinados, defina como True. O padrão é False.
    :param init_params: Parâmetros de inicialização, usados para codificar dados clássicos QTensor. O padrão é None. Se for uma porta lógica com p parâmetros, a dimensão dos dados de entrada precisa ser [1,p] ou [p].
    :param wires: Índice do bit de atuação do fio, o padrão é None.
    :param dtype: A precisão dos dados da matriz interna da porta lógica pode ser definida como pyvqnet.kcomplex64 ou pyvqnet.kcomplex128, correspondendo respectivamente a parâmetros de entrada float ou double.
    :param use_dagger: Se deve usar a versão transposta conjugada desta porta, o padrão é False.
    :return: Um Module que pode ser usado para treinar o modelo.

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
    
    Aplica a porta lógica quântica hadamard aos vetores de estado em ``q_machine``.

    :param q_machine: dispositivo de máquina virtual quântica.
    :param wires: índice do qubit.
    :param params: matriz de parâmetros, o padrão é None. Para uma função de operação de porta lógica com p parâmetros, a dimensão do parâmetro de entrada precisa ser [1,p] ou [p].
    :param use_dagger: se deve usar a transposta conjugada, o padrão é False.


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
    
    Define uma classe de porta lógica Hadamard.

    :param has_params: Se possui parâmetros. Portas como RX, RY precisam ser configuradas como True; as sem parâmetros precisam ser False. O padrão é False.
    :param trainable: Se contém parâmetros a serem treinados. Se a camada usa dados de entrada externos para construir uma matriz de porta lógica, defina como False. Se contém parâmetros a serem treinados, defina como True. O padrão é False.
    :param init_params: Parâmetros de inicialização, usados para codificar dados clássicos QTensor. O padrão é None. Se for uma porta lógica com p parâmetros, a dimensão dos dados de entrada precisa ser [1,p] ou [p].
    :param wires: Índice do bit de atuação do fio, o padrão é None.
    :param dtype: A precisão dos dados da matriz interna da porta lógica pode ser definida como pyvqnet.kcomplex64 ou pyvqnet.kcomplex128, correspondendo respectivamente a parâmetros de entrada float ou double.
    :param use_dagger: Se deve usar a versão transposta conjugada desta porta, o padrão é False.
    :return: Um Module que pode ser usado para treinar o modelo.

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
    
    Aplica a porta lógica quântica t aos vetores de estado em ``q_machine``.

    :param q_machine: dispositivo de máquina virtual quântica.
    :param wires: índice do qubit.
    :param params: matriz de parâmetros, o padrão é None. Para uma função de operação de porta lógica com p parâmetros, a dimensão do parâmetro de entrada precisa ser [1,p] ou [p].
    :param use_dagger: se deve usar a transposta conjugada, o padrão é False.


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
    
    Define uma classe de porta lógica T.

    :param has_params: Se possui parâmetros. Portas como RX, RY precisam ser configuradas como True; as sem parâmetros precisam ser False. O padrão é False.
    :param trainable: Se contém parâmetros a serem treinados. Se a camada usa dados de entrada externos para construir uma matriz de porta lógica, defina como False. Se contém parâmetros a serem treinados, defina como True. O padrão é False.
    :param init_params: Parâmetros de inicialização, usados para codificar dados clássicos QTensor. O padrão é None. Se for uma porta lógica com p parâmetros, a dimensão dos dados de entrada precisa ser [1,p] ou [p].
    :param wires: Índice do bit de atuação do fio, o padrão é None.
    :param dtype: A precisão dos dados da matriz interna da porta lógica pode ser definida como pyvqnet.kcomplex64 ou pyvqnet.kcomplex128, correspondendo respectivamente a parâmetros de entrada float ou double.
    :param use_dagger: Se deve usar a versão transposta conjugada desta porta, o padrão é False.
    :return: Um Module que pode ser usado para treinar o modelo.

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
    
    Aplica a porta lógica quântica s aos vetores de estado em ``q_machine``.

    :param q_machine: dispositivo de máquina virtual quântica.
    :param wires: índice do qubit.
    :param params: matriz de parâmetros, o padrão é None. Para uma função de operação de porta lógica com p parâmetros, a dimensão do parâmetro de entrada precisa ser [1,p] ou [p].
    :param use_dagger: se deve usar a transposta conjugada, o padrão é False.


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
    
    Define uma classe de porta lógica S.

    :param has_params: Se possui parâmetros. Portas como RX, RY precisam ser configuradas como True; as sem parâmetros precisam ser False. O padrão é False.
    :param trainable: Se contém parâmetros a serem treinados. Se a camada usa dados de entrada externos para construir uma matriz de porta lógica, defina como False. Se contém parâmetros a serem treinados, defina como True. O padrão é False.
    :param init_params: Parâmetros de inicialização, usados para codificar dados clássicos QTensor. O padrão é None. Se for uma porta lógica com p parâmetros, a dimensão dos dados de entrada precisa ser [1,p] ou [p].
    :param wires: Índice do bit de atuação do fio, o padrão é None.
    :param dtype: A precisão dos dados da matriz interna da porta lógica pode ser definida como pyvqnet.kcomplex64 ou pyvqnet.kcomplex128, correspondendo respectivamente a parâmetros de entrada float ou double.
    :param use_dagger: Se deve usar a versão transposta conjugada desta porta, o padrão é False.
    :return: Um Module que pode ser usado para treinar o modelo.

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
    
    Aplica a porta lógica quântica paulix aos vetores de estado em ``q_machine``.

    :param q_machine: dispositivo de máquina virtual quântica.
    :param wires: índice do qubit.
    :param params: matriz de parâmetros, o padrão é None. Para uma função de operação de porta lógica com p parâmetros, a dimensão do parâmetro de entrada precisa ser [1,p] ou [p].
    :param use_dagger: se deve usar a transposta conjugada, o padrão é False.


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
    
    Define uma classe de porta lógica PauliX.

    :param has_params: Se possui parâmetros. Portas como RX, RY precisam ser configuradas como True; as sem parâmetros precisam ser False. O padrão é False.
    :param trainable: Se contém parâmetros a serem treinados. Se a camada usa dados de entrada externos para construir uma matriz de porta lógica, defina como False. Se contém parâmetros a serem treinados, defina como True. O padrão é False.
    :param init_params: Parâmetros de inicialização, usados para codificar dados clássicos QTensor. O padrão é None. Se for uma porta lógica com p parâmetros, a dimensão dos dados de entrada precisa ser [1,p] ou [p].
    :param wires: Índice do bit de atuação do fio, o padrão é None.
    :param dtype: A precisão dos dados da matriz interna da porta lógica pode ser definida como pyvqnet.kcomplex64 ou pyvqnet.kcomplex128, correspondendo respectivamente a parâmetros de entrada float ou double.
    :param use_dagger: Se deve usar a versão transposta conjugada desta porta, o padrão é False.
    :return: Um Module que pode ser usado para treinar o modelo.

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
    
    Aplica a porta lógica quântica pauliy aos vetores de estado em ``q_machine``.

    :param q_machine: dispositivo de máquina virtual quântica.
    :param wires: índice do qubit.
    :param params: matriz de parâmetros, o padrão é None. Para uma função de operação de porta lógica com p parâmetros, a dimensão do parâmetro de entrada precisa ser [1,p] ou [p].
    :param use_dagger: se deve usar a transposta conjugada, o padrão é False.


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
    
    Define uma classe de porta lógica PauliY.

    :param has_params: Se possui parâmetros. Portas como RX, RY precisam ser configuradas como True; as sem parâmetros precisam ser False. O padrão é False.
    :param trainable: Se contém parâmetros a serem treinados. Se a camada usa dados de entrada externos para construir uma matriz de porta lógica, defina como False. Se contém parâmetros a serem treinados, defina como True. O padrão é False.
    :param init_params: Parâmetros de inicialização, usados para codificar dados clássicos QTensor. O padrão é None. Se for uma porta lógica com p parâmetros, a dimensão dos dados de entrada precisa ser [1,p] ou [p].
    :param wires: Índice do bit de atuação do fio, o padrão é None.
    :param dtype: A precisão dos dados da matriz interna da porta lógica pode ser definida como pyvqnet.kcomplex64 ou pyvqnet.kcomplex128, correspondendo respectivamente a parâmetros de entrada float ou double.
    :param use_dagger: Se deve usar a versão transposta conjugada desta porta, o padrão é False.
    :return: Um Module que pode ser usado para treinar o modelo.

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
    
    Aplica a porta lógica quântica pauliz aos vetores de estado em ``q_machine``.

    :param q_machine: dispositivo de máquina virtual quântica.
    :param wires: índice do qubit.
    :param params: matriz de parâmetros, o padrão é None. Para uma função de operação de porta lógica com p parâmetros, a dimensão do parâmetro de entrada precisa ser [1,p] ou [p].
    :param use_dagger: se deve usar a transposta conjugada, o padrão é False.


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
    
    Define uma classe de porta lógica PauliZ.

    :param has_params: Se possui parâmetros. Portas como RX, RY precisam ser configuradas como True; as sem parâmetros precisam ser False. O padrão é False.
    :param trainable: Se contém parâmetros a serem treinados. Se a camada usa dados de entrada externos para construir uma matriz de porta lógica, defina como False. Se contém parâmetros a serem treinados, defina como True. O padrão é False.
    :param init_params: Parâmetros de inicialização, usados para codificar dados clássicos QTensor. O padrão é None. Se for uma porta lógica com p parâmetros, a dimensão dos dados de entrada precisa ser [1,p] ou [p].
    :param wires: Índice do bit de atuação do fio, o padrão é None.
    :param dtype: A precisão dos dados da matriz interna da porta lógica pode ser definida como pyvqnet.kcomplex64 ou pyvqnet.kcomplex128, correspondendo respectivamente a parâmetros de entrada float ou double.
    :param use_dagger: Se deve usar a versão transposta conjugada desta porta, o padrão é False.
    :return: Um Module que pode ser usado para treinar o modelo.

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
    
    Aplica a porta lógica quântica x1 aos vetores de estado em ``q_machine``.

    :param q_machine: dispositivo de máquina virtual quântica.
    :param wires: índice do qubit.
    :param params: matriz de parâmetros, o padrão é None. Para uma função de operação de porta lógica com p parâmetros, a dimensão do parâmetro de entrada precisa ser [1,p] ou [p].
    :param use_dagger: se deve usar a transposta conjugada, o padrão é False.


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
    
    Define uma classe de porta lógica X1.

    :param has_params: Se possui parâmetros. Portas como RX, RY precisam ser configuradas como True; as sem parâmetros precisam ser False. O padrão é False.
    :param trainable: Se contém parâmetros a serem treinados. Se a camada usa dados de entrada externos para construir uma matriz de porta lógica, defina como False. Se contém parâmetros a serem treinados, defina como True. O padrão é False.
    :param init_params: Parâmetros de inicialização, usados para codificar dados clássicos QTensor. O padrão é None. Se for uma porta lógica com p parâmetros, a dimensão dos dados de entrada precisa ser [1,p] ou [p].
    :param wires: Índice do bit de atuação do fio, o padrão é None.
    :param dtype: A precisão dos dados da matriz interna da porta lógica pode ser definida como pyvqnet.kcomplex64 ou pyvqnet.kcomplex128, correspondendo respectivamente a parâmetros de entrada float ou double.
    :param use_dagger: Se deve usar a versão transposta conjugada desta porta, o padrão é False.
    :return: Um Module que pode ser usado para treinar o modelo.

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
    
    Aplica a porta lógica quântica y1 aos vetores de estado em ``q_machine``.

    :param q_machine: dispositivo de máquina virtual quântica.
    :param wires: índice do qubit.
    :param params: matriz de parâmetros, o padrão é None. Para uma função de operação de porta lógica com p parâmetros, a dimensão do parâmetro de entrada precisa ser [1,p] ou [p].
    :param use_dagger: se deve usar a transposta conjugada, o padrão é False.



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
    
    Define uma classe de porta lógica Y1.

    :param has_params: Se possui parâmetros. Portas como RX, RY precisam ser configuradas como True; as sem parâmetros precisam ser False. O padrão é False.
    :param trainable: Se contém parâmetros a serem treinados. Se a camada usa dados de entrada externos para construir uma matriz de porta lógica, defina como False. Se contém parâmetros a serem treinados, defina como True. O padrão é False.
    :param init_params: Parâmetros de inicialização, usados para codificar dados clássicos QTensor. O padrão é None. Se for uma porta lógica com p parâmetros, a dimensão dos dados de entrada precisa ser [1,p] ou [p].
    :param wires: Índice do bit de atuação do fio, o padrão é None.
    :param dtype: A precisão dos dados da matriz interna da porta lógica pode ser definida como pyvqnet.kcomplex64 ou pyvqnet.kcomplex128, correspondendo respectivamente a parâmetros de entrada float ou double.
    :param use_dagger: Se deve usar a versão transposta conjugada desta porta, o padrão é False.
    :return: Um Module que pode ser usado para treinar o modelo.

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
    
    Aplica a porta lógica quântica z1 aos vetores de estado em ``q_machine``.

    :param q_machine: dispositivo de máquina virtual quântica.
    :param wires: índice do qubit.
    :param params: matriz de parâmetros, o padrão é None. Para uma função de operação de porta lógica com p parâmetros, a dimensão do parâmetro de entrada precisa ser [1,p] ou [p].
    :param use_dagger: se deve usar a transposta conjugada, o padrão é False.


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
    
    Define uma classe de porta lógica Z1.

    :param has_params: Se possui parâmetros. Portas como RX, RY precisam ser configuradas como True; as sem parâmetros precisam ser False. O padrão é False.
    :param trainable: Se contém parâmetros a serem treinados. Se a camada usa dados de entrada externos para construir uma matriz de porta lógica, defina como False. Se contém parâmetros a serem treinados, defina como True. O padrão é False.
    :param init_params: Parâmetros de inicialização, usados para codificar dados clássicos QTensor. O padrão é None. Se for uma porta lógica com p parâmetros, a dimensão dos dados de entrada precisa ser [1,p] ou [p].
    :param wires: Índice do bit de atuação do fio, o padrão é None.
    :param dtype: A precisão dos dados da matriz interna da porta lógica pode ser definida como pyvqnet.kcomplex64 ou pyvqnet.kcomplex128, correspondendo respectivamente a parâmetros de entrada float ou double.
    :param use_dagger: Se deve usar a versão transposta conjugada desta porta, o padrão é False.
    :return: Um Module que pode ser usado para treinar o modelo.

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
    
    Aplica a porta lógica quântica rx aos vetores de estado em ``q_machine``.

    :param q_machine: dispositivo de máquina virtual quântica.
    :param wires: índice do qubit.
    :param params: matriz de parâmetros, o padrão é None. Para uma função de operação de porta lógica com p parâmetros, a dimensão do parâmetro de entrada precisa ser [1,p] ou [p].
    :param use_dagger: se deve usar a transposta conjugada, o padrão é False.


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
    
    Define uma classe de porta lógica RX.

    :param has_params: Se possui parâmetros. Portas como RX, RY precisam ser configuradas como True; as sem parâmetros precisam ser False. O padrão é False.
    :param trainable: Se contém parâmetros a serem treinados. Se a camada usa dados de entrada externos para construir uma matriz de porta lógica, defina como False. Se contém parâmetros a serem treinados, defina como True. O padrão é False.
    :param init_params: Parâmetros de inicialização, usados para codificar dados clássicos QTensor. O padrão é None. Se for uma porta lógica com p parâmetros, a dimensão dos dados de entrada precisa ser [1,p] ou [p].
    :param wires: Índice do bit de atuação do fio, o padrão é None.
    :param dtype: A precisão dos dados da matriz interna da porta lógica pode ser definida como pyvqnet.kcomplex64 ou pyvqnet.kcomplex128, correspondendo respectivamente a parâmetros de entrada float ou double.
    :param use_dagger: Se deve usar a versão transposta conjugada desta porta, o padrão é False.
    :return: Um Module que pode ser usado para treinar o modelo.

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
    
    Aplica a porta lógica quântica ry aos vetores de estado em ``q_machine``.

    :param q_machine: dispositivo de máquina virtual quântica.
    :param wires: índice do qubit.
    :param params: matriz de parâmetros, o padrão é None. Para uma função de operação de porta lógica com p parâmetros, a dimensão do parâmetro de entrada precisa ser [1,p] ou [p].
    :param use_dagger: se deve usar a transposta conjugada, o padrão é False.


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
    
    Define uma classe de porta lógica RY.

    :param has_params: Se possui parâmetros. Portas como RX, RY precisam ser configuradas como True; as sem parâmetros precisam ser False. O padrão é False.
    :param trainable: Se contém parâmetros a serem treinados. Se a camada usa dados de entrada externos para construir uma matriz de porta lógica, defina como False. Se contém parâmetros a serem treinados, defina como True. O padrão é False.
    :param init_params: Parâmetros de inicialização, usados para codificar dados clássicos QTensor. O padrão é None. Se for uma porta lógica com p parâmetros, a dimensão dos dados de entrada precisa ser [1,p] ou [p].
    :param wires: Índice do bit de atuação do fio, o padrão é None.
    :param dtype: A precisão dos dados da matriz interna da porta lógica pode ser definida como pyvqnet.kcomplex64 ou pyvqnet.kcomplex128, correspondendo respectivamente a parâmetros de entrada float ou double.
    :param use_dagger: Se deve usar a versão transposta conjugada desta porta, o padrão é False.
    :return: Um Module que pode ser usado para treinar o modelo.

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
    
    Aplica a porta lógica quântica rz aos vetores de estado em ``q_machine``.

    :param q_machine: dispositivo de máquina virtual quântica.
    :param wires: índice do qubit.
    :param params: matriz de parâmetros, o padrão é None. Para uma função de operação de porta lógica com p parâmetros, a dimensão do parâmetro de entrada precisa ser [1,p] ou [p].
    :param use_dagger: se deve usar a transposta conjugada, o padrão é False.


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
    
    Define uma classe de porta lógica RZ.

    :param has_params: Se possui parâmetros. Portas como RX, RY precisam ser configuradas como True; as sem parâmetros precisam ser False. O padrão é False.
    :param trainable: Se contém parâmetros a serem treinados. Se a camada usa dados de entrada externos para construir uma matriz de porta lógica, defina como False. Se contém parâmetros a serem treinados, defina como True. O padrão é False.
    :param init_params: Parâmetros de inicialização, usados para codificar dados clássicos QTensor. O padrão é None. Se for uma porta lógica com p parâmetros, a dimensão dos dados de entrada precisa ser [1,p] ou [p].
    :param wires: Índice do bit de atuação do fio, o padrão é None.
    :param dtype: A precisão dos dados da matriz interna da porta lógica pode ser definida como pyvqnet.kcomplex64 ou pyvqnet.kcomplex128, correspondendo respectivamente a parâmetros de entrada float ou double.
    :param use_dagger: Se deve usar a versão transposta conjugada desta porta, o padrão é False.
    :return: Um Module que pode ser usado para treinar o modelo.

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
    
    Aplica a porta lógica quântica crx aos vetores de estado em ``q_machine``.

    :param q_machine: dispositivo de máquina virtual quântica.
    :param wires: índice do qubit.
    :param params: matriz de parâmetros, o padrão é None. Para uma função de operação de porta lógica com p parâmetros, a dimensão do parâmetro de entrada precisa ser [1,p] ou [p].
    :param use_dagger: se deve usar a transposta conjugada, o padrão é False.



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
    
     Define uma classe de porta lógica CRX.

    :param has_params: Se possui parâmetros. Portas como RX, RY precisam ser configuradas como True; as sem parâmetros precisam ser False. O padrão é False.
    :param trainable: Se contém parâmetros a serem treinados. Se a camada usa dados de entrada externos para construir uma matriz de porta lógica, defina como False. Se contém parâmetros a serem treinados, defina como True. O padrão é False.
    :param init_params: Parâmetros de inicialização, usados para codificar dados clássicos QTensor. O padrão é None. Se for uma porta lógica com p parâmetros, a dimensão dos dados de entrada precisa ser [1,p] ou [p].
    :param wires: Índice do bit de atuação do fio, o padrão é None.
    :param dtype: A precisão dos dados da matriz interna da porta lógica pode ser definida como pyvqnet.kcomplex64 ou pyvqnet.kcomplex128, correspondendo respectivamente a parâmetros de entrada float ou double.
    :param use_dagger: Se deve usar a versão transposta conjugada desta porta, o padrão é False.
    :return: Um Module que pode ser usado para treinar o modelo.

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
    
    Aplica a porta lógica quântica cry aos vetores de estado em ``q_machine``.

    :param q_machine: dispositivo de máquina virtual quântica.
    :param wires: índice do qubit.
    :param params: matriz de parâmetros, o padrão é None. Para uma função de operação de porta lógica com p parâmetros, a dimensão do parâmetro de entrada precisa ser [1,p] ou [p].
    :param use_dagger: se deve usar a transposta conjugada, o padrão é False.


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
    
     Define uma classe de porta lógica CRY.

    :param has_params: Se possui parâmetros. Portas como RX, RY precisam ser configuradas como True; as sem parâmetros precisam ser False. O padrão é False.
    :param trainable: Se contém parâmetros a serem treinados. Se a camada usa dados de entrada externos para construir uma matriz de porta lógica, defina como False. Se contém parâmetros a serem treinados, defina como True. O padrão é False.
    :param init_params: Parâmetros de inicialização, usados para codificar dados clássicos QTensor. O padrão é None. Se for uma porta lógica com p parâmetros, a dimensão dos dados de entrada precisa ser [1,p] ou [p].
    :param wires: Índice do bit de atuação do fio, o padrão é None.
    :param dtype: A precisão dos dados da matriz interna da porta lógica pode ser definida como pyvqnet.kcomplex64 ou pyvqnet.kcomplex128, correspondendo respectivamente a parâmetros de entrada float ou double.
    :param use_dagger: Se deve usar a versão transposta conjugada desta porta, o padrão é False.
    :return: Um Module que pode ser usado para treinar o modelo.

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
    
    Aplica a porta lógica quântica crz aos vetores de estado em ``q_machine``.

    :param q_machine: dispositivo de máquina virtual quântica.
    :param wires: índice do qubit.
    :param params: matriz de parâmetros, o padrão é None. Para uma função de operação de porta lógica com p parâmetros, a dimensão do parâmetro de entrada precisa ser [1,p] ou [p].
    :param use_dagger: se deve usar a transposta conjugada, o padrão é False.


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
    
    Define uma classe de porta lógica CRZ.

    :param has_params: Se possui parâmetros. Portas como RX, RY precisam ser configuradas como True; as sem parâmetros precisam ser False. O padrão é False.
    :param trainable: Se contém parâmetros a serem treinados. Se a camada usa dados de entrada externos para construir uma matriz de porta lógica, defina como False. Se contém parâmetros a serem treinados, defina como True. O padrão é False.
    :param init_params: Parâmetros de inicialização, usados para codificar dados clássicos QTensor. O padrão é None. Se for uma porta lógica com p parâmetros, a dimensão dos dados de entrada precisa ser [1,p] ou [p].
    :param wires: Índice do bit de atuação do fio, o padrão é None.
    :param dtype: A precisão dos dados da matriz interna da porta lógica pode ser definida como pyvqnet.kcomplex64 ou pyvqnet.kcomplex128, correspondendo respectivamente a parâmetros de entrada float ou double.
    :param use_dagger: Se deve usar a versão transposta conjugada desta porta, o padrão é False.
    :return: Um Module que pode ser usado para treinar o modelo.

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
    
    Aplica a porta lógica quântica u1 aos vetores de estado em ``q_machine``.

    :param q_machine: dispositivo de máquina virtual quântica.
    :param wires: índice do qubit.
    :param params: matriz de parâmetros, o padrão é None. Para uma função de operação de porta lógica com p parâmetros, a dimensão do parâmetro de entrada precisa ser [1,p] ou [p].
    :param use_dagger: se deve usar a transposta conjugada, o padrão é False.


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
    
    Define uma classe de porta lógica U1.

    :param has_params: Se possui parâmetros. Portas como RX, RY precisam ser configuradas como True; as sem parâmetros precisam ser False. O padrão é False.
    :param trainable: Se contém parâmetros a serem treinados. Se a camada usa dados de entrada externos para construir uma matriz de porta lógica, defina como False. Se contém parâmetros a serem treinados, defina como True. O padrão é False.
    :param init_params: Parâmetros de inicialização, usados para codificar dados clássicos QTensor. O padrão é None. Se for uma porta lógica com p parâmetros, a dimensão dos dados de entrada precisa ser [1,p] ou [p].
    :param wires: Índice do bit de atuação do fio, o padrão é None.
    :param dtype: A precisão dos dados da matriz interna da porta lógica pode ser definida como pyvqnet.kcomplex64 ou pyvqnet.kcomplex128, correspondendo respectivamente a parâmetros de entrada float ou double.
    :param use_dagger: Se deve usar a versão transposta conjugada desta porta, o padrão é False.
    :return: Um Module que pode ser usado para treinar o modelo.

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
    
    Aplica a porta lógica quântica u2 aos vetores de estado em ``q_machine``.

    :param q_machine: dispositivo de máquina virtual quântica.
    :param wires: índice do qubit.
    :param params: matriz de parâmetros, o padrão é None. Para uma função de operação de porta lógica com p parâmetros, a dimensão do parâmetro de entrada precisa ser [1,p] ou [p].
    :param use_dagger: se deve usar a transposta conjugada, o padrão é False.


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
    
    Define uma classe de porta lógica U2.

    :param has_params: Se possui parâmetros. Portas como RX, RY precisam ser configuradas como True; as sem parâmetros precisam ser False. O padrão é False.
    :param trainable: Se contém parâmetros a serem treinados. Se a camada usa dados de entrada externos para construir uma matriz de porta lógica, defina como False. Se contém parâmetros a serem treinados, defina como True. O padrão é False.
    :param init_params: Parâmetros de inicialização, usados para codificar dados clássicos QTensor. O padrão é None. Se for uma porta lógica com p parâmetros, a dimensão dos dados de entrada precisa ser [1,p] ou [p].
    :param wires: Índice do bit de atuação do fio, o padrão é None.
    :param dtype: A precisão dos dados da matriz interna da porta lógica pode ser definida como pyvqnet.kcomplex64 ou pyvqnet.kcomplex128, correspondendo respectivamente a parâmetros de entrada float ou double.
    :param use_dagger: Se deve usar a versão transposta conjugada desta porta, o padrão é False.
    :return: Um Module que pode ser usado para treinar o modelo.

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
    
    Aplica a porta lógica quântica u3 aos vetores de estado em ``q_machine``.

    :param q_machine: dispositivo de máquina virtual quântica.
    :param wires: índice do qubit.
    :param params: matriz de parâmetros, o padrão é None. Para uma função de operação de porta lógica com p parâmetros, a dimensão do parâmetro de entrada precisa ser [1,p] ou [p].
    :param use_dagger: se deve usar a transposta conjugada, o padrão é False.


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
    
    Define uma classe de porta lógica U3.

    :param has_params: Se possui parâmetros. Portas como RX, RY precisam ser configuradas como True; as sem parâmetros precisam ser False. O padrão é False.
    :param trainable: Se contém parâmetros a serem treinados. Se a camada usa dados de entrada externos para construir uma matriz de porta lógica, defina como False. Se contém parâmetros a serem treinados, defina como True. O padrão é False.
    :param init_params: Parâmetros de inicialização, usados para codificar dados clássicos QTensor. O padrão é None. Se for uma porta lógica com p parâmetros, a dimensão dos dados de entrada precisa ser [1,p] ou [p].
    :param wires: Índice do bit de atuação do fio, o padrão é None.
    :param dtype: A precisão dos dados da matriz interna da porta lógica pode ser definida como pyvqnet.kcomplex64 ou pyvqnet.kcomplex128, correspondendo respectivamente a parâmetros de entrada float ou double.
    :param use_dagger: Se deve usar a versão transposta conjugada desta porta, o padrão é False.
    :return: Um Module que pode ser usado para treinar o modelo.

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

    Aplica a porta lógica quântica cy aos vetores de estado em ``q_machine``.

    :param q_machine: Dispositivo de máquina virtual quântica.
    :param wires: Índice do qubit.
    :param params: Matriz de parâmetros, o padrão é None.
    :param use_dagger: Se deve usar a transposta conjugada, o padrão é False.

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

    Define uma classe de porta lógica CY.

    :param has_params: Se possui parâmetros. Portas como RX, RY precisam ser configuradas como True; as sem parâmetros precisam ser False. O padrão é False.
    :param trainable: Se contém parâmetros a serem treinados. Se esta camada usa dados de entrada externos para construir uma matriz de porta lógica, defina como False. Se os parâmetros a serem treinados precisam ser inicializados a partir desta camada, defina como True. O padrão é False.
    :param init_params: Parâmetros de inicialização, usados para codificar dados clássicos QTensor, o padrão é None. Se for uma porta lógica com p parâmetros, a dimensão dos dados de entrada precisa ser [1,p] ou [p].
    :param wires: Índice do bit de atuação do fio, o padrão é None.
    :param dtype: A precisão dos dados da matriz interna da porta lógica pode ser definida como pyvqnet.kcomplex64 ou pyvqnet.kcomplex128, correspondendo respectivamente a parâmetros de entrada float ou double.
    :param use_dagger: Se deve usar a versão transposta conjugada desta porta, o padrão é False.
    :return: Um Module que pode ser usado para treinar o modelo.

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
    
    Aplica a porta lógica quântica cnot aos vetores de estado em ``q_machine``.

    :param q_machine: dispositivo de máquina virtual quântica.
    :param wires: índice do qubit.
    :param params: matriz de parâmetros, o padrão é None. Para uma função de operação de porta lógica com p parâmetros, a dimensão do parâmetro de entrada precisa ser [1,p] ou [p].
    :param use_dagger: se deve usar a transposta conjugada, o padrão é False.


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
    
    Define uma classe de porta lógica CNOT.

    :param has_params: Se possui parâmetros. Portas como RX, RY precisam ser configuradas como True; as sem parâmetros precisam ser False. O padrão é False.
    :param trainable: Se contém parâmetros a serem treinados. Se a camada usa dados de entrada externos para construir uma matriz de porta lógica, defina como False. Se contém parâmetros a serem treinados, defina como True. O padrão é False.
    :param init_params: Parâmetros de inicialização, usados para codificar dados clássicos QTensor. O padrão é None. Se for uma porta lógica com p parâmetros, a dimensão dos dados de entrada precisa ser [1,p] ou [p].
    :param wires: Índice do bit de atuação do fio, o padrão é None.
    :param dtype: A precisão dos dados da matriz interna da porta lógica pode ser definida como pyvqnet.kcomplex64 ou pyvqnet.kcomplex128, correspondendo respectivamente a parâmetros de entrada float ou double.
    :param use_dagger: Se deve usar a versão transposta conjugada desta porta, o padrão é False.
    :return: Um Module que pode ser usado para treinar o modelo.

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
    
    Aplica a porta lógica quântica cr aos vetores de estado em ``q_machine``.

    :param q_machine: dispositivo de máquina virtual quântica.
    :param wires: índice do qubit.
    :param params: matriz de parâmetros, o padrão é None. Para uma função de operação de porta lógica com p parâmetros, a dimensão do parâmetro de entrada precisa ser [1,p] ou [p].
    :param use_dagger: se deve usar a transposta conjugada, o padrão é False.


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
    
    Define uma porta lógica CR.

    :param has_params: Se possui parâmetros. Portas como RX, RY precisam ser configuradas como True; as sem parâmetros precisam ser False. O padrão é False.
    :param trainable: Se contém parâmetros a serem treinados. Se a camada usa dados de entrada externos para construir uma matriz de porta lógica, defina como False. Se contém parâmetros a serem treinados, defina como True. O padrão é False.
    :param init_params: Parâmetros de inicialização, usados para codificar dados clássicos QTensor. O padrão é None. Se for uma porta lógica com p parâmetros, a dimensão dos dados de entrada precisa ser [1,p] ou [p].
    :param wires: Índice do bit de atuação do fio, o padrão é None.
    :param dtype: A precisão dos dados da matriz interna da porta lógica pode ser definida como pyvqnet.kcomplex64 ou pyvqnet.kcomplex128, correspondendo respectivamente a parâmetros de entrada float ou double.
    :param use_dagger: Se deve usar a versão transposta conjugada desta porta, o padrão é False.
    :return: Um Module que pode ser usado para treinar o modelo.

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
    
    Aplica a porta lógica quântica iswap aos vetores de estado em ``q_machine``.

    :param q_machine: dispositivo de máquina virtual quântica.
    :param wires: índice do qubit.
    :param params: matriz de parâmetros, o padrão é None. Para uma função de operação de porta lógica com p parâmetros, a dimensão do parâmetro de entrada precisa ser [1,p] ou [p].
    :param use_dagger: se deve usar a transposta conjugada, o padrão é False.


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
    
    Aplica a porta lógica quântica swap aos vetores de estado em ``q_machine``.

    :param q_machine: dispositivo de máquina virtual quântica.
    :param wires: índice do qubit.
    :param params: matriz de parâmetros, o padrão é None. Para uma função de operação de porta lógica com p parâmetros, a dimensão do parâmetro de entrada precisa ser [1,p] ou [p].
    :param use_dagger: se deve usar a transposta conjugada, o padrão é False.


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
    
    Define uma classe de porta lógica SWAP.

    :param has_params: Se possui parâmetros. Portas como RX, RY precisam ser configuradas como True; as sem parâmetros precisam ser False. O padrão é False.
    :param trainable: Se contém parâmetros a serem treinados. Se a camada usa dados de entrada externos para construir uma matriz de porta lógica, defina como False. Se contém parâmetros a serem treinados, defina como True. O padrão é False.
    :param init_params: Parâmetros de inicialização, usados para codificar dados clássicos QTensor. O padrão é None. Se for uma porta lógica com p parâmetros, a dimensão dos dados de entrada precisa ser [1,p] ou [p].
    :param wires: Índice do bit de atuação do fio, o padrão é None.
    :param dtype: A precisão dos dados da matriz interna da porta lógica pode ser definida como pyvqnet.kcomplex64 ou pyvqnet.kcomplex128, correspondendo respectivamente a parâmetros de entrada float ou double.
    :param use_dagger: Se deve usar a versão transposta conjugada desta porta, o padrão é False.
    :return: Um Module que pode ser usado para treinar o modelo.

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

    Aplica a porta lógica quântica cswap aos vetores de estado em ``q_machine``.

    :param q_machine: Dispositivo de máquina virtual quântica.
    :param wires: Índice do qubit.
    :param params: Matriz de parâmetros, o padrão é None.
    :param use_dagger: Se deve usar a transposta conjugada, o padrão é False.
    :return: Saída QTensor.

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
    
    Define uma classe de porta lógica SWAP.

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

    :param has_params: Se possui parâmetros. Portas como RX, RY precisam ser configuradas como True; as sem parâmetros precisam ser False. O padrão é False.
    :param trainable: Se contém parâmetros a serem treinados. Se esta camada usa dados de entrada externos para construir uma matriz de porta lógica, defina como False. Se os parâmetros a serem treinados precisam ser inicializados a partir desta camada, defina como True. O padrão é False.
    :param init_params: Parâmetros de inicialização, usados para codificar dados clássicos QTensor. O padrão é None. Se for uma porta lógica com p parâmetros, a dimensão dos dados de entrada precisa ser [1,p] ou [p].
    :param wires: Índice do bit de atuação do fio, o padrão é None.
    :param dtype: A precisão dos dados da matriz interna da porta lógica pode ser definida como pyvqnet.kcomplex64 ou pyvqnet.kcomplex128, correspondendo respectivamente a parâmetros de entrada float ou double.
    :param use_dagger: Se deve usar a versão transposta conjugada desta porta, o padrão é False.
    :return: Um Module que pode ser usado para treinar o modelo.

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
    
    Aplica a porta lógica quântica cz aos vetores de estado em ``q_machine``.

    :param q_machine: dispositivo de máquina virtual quântica.
    :param wires: índice do qubit.
    :param params: matriz de parâmetros, o padrão é None. Para uma função de operação de porta lógica com p parâmetros, a dimensão do parâmetro de entrada precisa ser [1,p] ou [p].
    :param use_dagger: se deve usar a transposta conjugada, o padrão é False.


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
    
    Define uma classe de porta lógica CZ.

    :param has_params: Se possui parâmetros. Portas como RX, RY precisam ser configuradas como True; as sem parâmetros precisam ser False. O padrão é False.
    :param trainable: Se contém parâmetros a serem treinados. Se a camada usa dados de entrada externos para construir uma matriz de porta lógica, defina como False. Se contém parâmetros a serem treinados, defina como True. O padrão é False.
    :param init_params: Parâmetros de inicialização, usados para codificar dados clássicos QTensor. O padrão é None. Se for uma porta lógica com p parâmetros, a dimensão dos dados de entrada precisa ser [1,p] ou [p].
    :param wires: Índice do bit de atuação do fio, o padrão é None.
    :param dtype: A precisão dos dados da matriz interna da porta lógica pode ser definida como pyvqnet.kcomplex64 ou pyvqnet.kcomplex128, correspondendo respectivamente a parâmetros de entrada float ou double.
    :param use_dagger: Se deve usar a versão transposta conjugada desta porta, o padrão é False.
    :return: Um Module que pode ser usado para treinar o modelo.

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
    
    Aplica a porta lógica quântica rxx aos vetores de estado em ``q_machine``.

    :param q_machine: dispositivo de máquina virtual quântica.
    :param wires: índice do qubit.
    :param params: matriz de parâmetros, o padrão é None. Para uma função de operação de porta lógica com p parâmetros, a dimensão do parâmetro de entrada precisa ser [1,p] ou [p].
    :param use_dagger: se deve usar a transposta conjugada, o padrão é False.


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
    
    Define uma classe de porta lógica RXX.

    :param has_params: Se possui parâmetros. Portas como RX, RY precisam ser configuradas como True; as sem parâmetros precisam ser False. O padrão é False.
    :param trainable: Se contém parâmetros a serem treinados. Se a camada usa dados de entrada externos para construir uma matriz de porta lógica, defina como False. Se contém parâmetros a serem treinados, defina como True. O padrão é False.
    :param init_params: Parâmetros de inicialização, usados para codificar dados clássicos QTensor. O padrão é None. Se for uma porta lógica com p parâmetros, a dimensão dos dados de entrada precisa ser [1,p] ou [p].
    :param wires: Índice do bit de atuação do fio, o padrão é None.
    :param dtype: A precisão dos dados da matriz interna da porta lógica pode ser definida como pyvqnet.kcomplex64 ou pyvqnet.kcomplex128, correspondendo respectivamente a parâmetros de entrada float ou double.
    :param use_dagger: Se deve usar a versão transposta conjugada desta porta, o padrão é False.
    :return: Um Module que pode ser usado para treinar o modelo.

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
    
    Aplica a porta lógica quântica ryy aos vetores de estado em ``q_machine``.

    :param q_machine: dispositivo de máquina virtual quântica.
    :param wires: índice do qubit.
    :param params: matriz de parâmetros, o padrão é None. Para uma função de operação de porta lógica com p parâmetros, a dimensão do parâmetro de entrada precisa ser [1,p] ou [p].
    :param use_dagger: se deve usar a transposta conjugada, o padrão é False.


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
    
    Define uma classe de porta lógica RYY.

    :param has_params: Se possui parâmetros. Portas como RX, RY precisam ser configuradas como True; as sem parâmetros precisam ser False. O padrão é False.
    :param trainable: Se contém parâmetros a serem treinados. Se a camada usa dados de entrada externos para construir uma matriz de porta lógica, defina como False. Se contém parâmetros a serem treinados, defina como True. O padrão é False.
    :param init_params: Parâmetros de inicialização, usados para codificar dados clássicos QTensor. O padrão é None. Se for uma porta lógica com p parâmetros, a dimensão dos dados de entrada precisa ser [1,p] ou [p].
    :param wires: Índice do bit de atuação do fio, o padrão é None.
    :param dtype: A precisão dos dados da matriz interna da porta lógica pode ser definida como pyvqnet.kcomplex64 ou pyvqnet.kcomplex128, correspondendo respectivamente a parâmetros de entrada float ou double.
    :param use_dagger: Se deve usar a versão transposta conjugada desta porta, o padrão é False.
    :return: Um Module que pode ser usado para treinar o modelo.

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
    
    Aplica a porta lógica quântica rzz aos vetores de estado em ``q_machine``.

    :param q_machine: dispositivo de máquina virtual quântica.
    :param wires: índice do qubit.
    :param params: matriz de parâmetros, o padrão é None. Para uma função de operação de porta lógica com p parâmetros, a dimensão do parâmetro de entrada precisa ser [1,p] ou [p].
    :param use_dagger: se deve usar a transposta conjugada, o padrão é False.


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
    
    Define uma classe de porta lógica RZZ.

    :param has_params: Se possui parâmetros. Portas como RX, RY precisam ser configuradas como True; as sem parâmetros precisam ser False. O padrão é False.
    :param trainable: Se contém parâmetros a serem treinados. Se a camada usa dados de entrada externos para construir uma matriz de porta lógica, defina como False. Se contém parâmetros a serem treinados, defina como True. O padrão é False.
    :param init_params: Parâmetros de inicialização, usados para codificar dados clássicos QTensor. O padrão é None. Se for uma porta lógica com p parâmetros, a dimensão dos dados de entrada precisa ser [1,p] ou [p].
    :param wires: Índice do bit de atuação do fio, o padrão é None.
    :param dtype: A precisão dos dados da matriz interna da porta lógica pode ser definida como pyvqnet.kcomplex64 ou pyvqnet.kcomplex128, correspondendo respectivamente a parâmetros de entrada float ou double.
    :param use_dagger: Se deve usar a versão transposta conjugada desta porta, o padrão é False.
    :return: Um Module que pode ser usado para treinar o modelo.

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
    
    Aplica a porta lógica quântica rzx aos vetores de estado em ``q_machine``.

    :param q_machine: dispositivo de máquina virtual quântica.
    :param wires: índice do qubit.
    :param params: matriz de parâmetros, o padrão é None. Para uma função de operação de porta lógica com p parâmetros, a dimensão do parâmetro de entrada precisa ser [1,p] ou [p].
    :param use_dagger: se deve usar a transposta conjugada, o padrão é False.


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
        #   [[0.       -0.0998334j 0.       +0.j       ]
        #     [0.       +0.j        0.       +0.j       ]]
        # 
        #    [[0.       +0.j        0.       +0.j       ]
        #     [0.       +0.j        0.       +0.j       ]]]]]

RZX
------------------------------------------


.. py:class:: pyvqnet.qnn.vqc.RZX(has_params: bool = False,trainable: bool = False,init_params=None,wires=None,dtype=pyvqnet.kcomplex64,use_dagger=False)
    
    Define uma classe de porta lógica RZX.

    :param has_params: Se possui parâmetros. Portas como RX, RY precisam ser configuradas como True; as sem parâmetros precisam ser False. O padrão é False.
    :param trainable: Se contém parâmetros a serem treinados. Se a camada usa dados de entrada externos para construir uma matriz de porta lógica, defina como False. Se contém parâmetros a serem treinados, defina como True. O padrão é False.
    :param init_params: Parâmetros de inicialização, usados para codificar dados clássicos QTensor. O padrão é None. Se for uma porta lógica com p parâmetros, a dimensão dos dados de entrada precisa ser [1,p] ou [p].
    :param wires: Índice do bit de atuação do fio, o padrão é None.
    :param dtype: A precisão dos dados da matriz interna da porta lógica pode ser definida como pyvqnet.kcomplex64 ou pyvqnet.kcomplex128, correspondendo respectivamente a parâmetros de entrada float ou double.
    :param use_dagger: Se deve usar a versão transposta conjugada desta porta, o padrão é False.
    :return: Um Module que pode ser usado para treinar o modelo.

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
    
    Aplica a porta lógica quântica toffoli aos vetores de estado em ``q_machine``.

    :param q_machine: dispositivo de máquina virtual quântica.
    :param wires: índice do qubit.
    :param params: matriz de parâmetros, o padrão é None. Para uma função de operação de porta lógica com p parâmetros, a dimensão do parâmetro de entrada precisa ser [1,p] ou [p].
    :param use_dagger: se deve usar a transposta conjugada, o padrão é False.
    :return: Saída QTensor.

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
    
    Define uma classe de porta lógica Toffoli.

    :param has_params: Se possui parâmetros. Portas como RX, RY precisam ser configuradas como True; as sem parâmetros precisam ser False. O padrão é False.
    :param trainable: Se contém parâmetros a serem treinados. Se a camada usa dados de entrada externos para construir uma matriz de porta lógica, defina como False. Se contém parâmetros a serem treinados, defina como True. O padrão é False.
    :param init_params: Parâmetros de inicialização, usados para codificar dados clássicos QTensor. O padrão é None. Se for uma porta lógica com p parâmetros, a dimensão dos dados de entrada precisa ser [1,p] ou [p].
    :param wires: Índice do bit de atuação do fio, o padrão é None.
    :param dtype: A precisão dos dados da matriz interna da porta lógica pode ser definida como pyvqnet.kcomplex64 ou pyvqnet.kcomplex128, correspondendo respectivamente a parâmetros de entrada float ou double.
    :param use_dagger: Se deve usar a versão transposta conjugada desta porta, o padrão é False.
    :return: Um Module que pode ser usado para treinar o modelo.

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
    
    Aplica a porta lógica quântica isingxx aos vetores de estado em ``q_machine``.

    :param q_machine: dispositivo de máquina virtual quântica.
    :param wires: índice do qubit.
    :param params: matriz de parâmetros, o padrão é None. Para uma função de operação de porta lógica com p parâmetros, a dimensão do parâmetro de entrada precisa ser [1,p] ou [p].
    :param use_dagger: se deve usar a transposta conjugada, o padrão é False.
    :return: Saída QTensor.

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
    
    Define uma classe de porta lógica IsingXX.

    :param has_params: Se possui parâmetros. Portas como RX, RY precisam ser configuradas como True; as sem parâmetros precisam ser False. O padrão é False.
    :param trainable: Se contém parâmetros a serem treinados. Se a camada usa dados de entrada externos para construir uma matriz de porta lógica, defina como False. Se contém parâmetros a serem treinados, defina como True. O padrão é False.
    :param init_params: Parâmetros de inicialização, usados para codificar dados clássicos QTensor. O padrão é None. Se for uma porta lógica com p parâmetros, a dimensão dos dados de entrada precisa ser [1,p] ou [p].
    :param wires: Índice do bit de atuação do fio, o padrão é None.
    :param dtype: A precisão dos dados da matriz interna da porta lógica pode ser definida como pyvqnet.kcomplex64 ou pyvqnet.kcomplex128, correspondendo respectivamente a parâmetros de entrada float ou double.
    :param use_dagger: Se deve usar a versão transposta conjugada desta porta, o padrão é False.
    :return: Um Module que pode ser usado para treinar o modelo.

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
    
    Aplica a porta lógica quântica isingyy aos vetores de estado em ``q_machine``.

    :param q_machine: dispositivo de máquina virtual quântica.
    :param wires: índice do qubit.
    :param params: matriz de parâmetros, o padrão é None. Para uma função de operação de porta lógica com p parâmetros, a dimensão do parâmetro de entrada precisa ser [1,p] ou [p].
    :param use_dagger: se deve usar a transposta conjugada, o padrão é False.
    :return: Saída QTensor.

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
    
    Define uma classe de porta lógica IsingYY.

    :param has_params: Se possui parâmetros. Portas como RX, RY precisam ser configuradas como True; as sem parâmetros precisam ser False. O padrão é False.
    :param trainable: Se contém parâmetros a serem treinados. Se a camada usa dados de entrada externos para construir uma matriz de porta lógica, defina como False. Se contém parâmetros a serem treinados, defina como True. O padrão é False.
    :param init_params: Parâmetros de inicialização, usados para codificar dados clássicos QTensor. O padrão é None. Se for uma porta lógica com p parâmetros, a dimensão dos dados de entrada precisa ser [1,p] ou [p].
    :param wires: Índice do bit de atuação do fio, o padrão é None.
    :param dtype: A precisão dos dados da matriz interna da porta lógica pode ser definida como pyvqnet.kcomplex64 ou pyvqnet.kcomplex128, correspondendo respectivamente a parâmetros de entrada float ou double.
    :param use_dagger: Se deve usar a versão transposta conjugada desta porta, o padrão é False.
    :return: Um Module que pode ser usado para treinar o modelo.

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
    
    Aplica a porta lógica quântica isingzz aos vetores de estado em ``q_machine``.

    :param q_machine: dispositivo de máquina virtual quântica.
    :param wires: índice do qubit.
    :param params: matriz de parâmetros, o padrão é None. Para uma função de operação de porta lógica com p parâmetros, a dimensão do parâmetro de entrada precisa ser [1,p] ou [p].
    :param use_dagger: se deve usar a transposta conjugada, o padrão é False.
    :return: Saída QTensor.

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
    
    Define uma classe de porta lógica IsingZZ.

    :param has_params: Se possui parâmetros. Portas como RX, RY precisam ser configuradas como True; as sem parâmetros precisam ser False. O padrão é False.
    :param trainable: Se contém parâmetros a serem treinados. Se a camada usa dados de entrada externos para construir uma matriz de porta lógica, defina como False. Se contém parâmetros a serem treinados, defina como True. O padrão é False.
    :param init_params: Parâmetros de inicialização, usados para codificar dados clássicos QTensor. O padrão é None. Se for uma porta lógica com p parâmetros, a dimensão dos dados de entrada precisa ser [1,p] ou [p].
    :param wires: Índice do bit de atuação do fio, o padrão é None.
    :param dtype: A precisão dos dados da matriz interna da porta lógica pode ser definida como pyvqnet.kcomplex64 ou pyvqnet.kcomplex128, correspondendo respectivamente a parâmetros de entrada float ou double.
    :param use_dagger: Se deve usar a versão transposta conjugada desta porta, o padrão é False.
    :return: Um Module que pode ser usado para treinar o modelo.

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
    
    Aplica a porta lógica quântica isingxy aos vetores de estado em ``q_machine``.

    :param q_machine: dispositivo de máquina virtual quântica.
    :param wires: índice do qubit.
    :param params: matriz de parâmetros, o padrão é None. Para uma função de operação de porta lógica com p parâmetros, a dimensão do parâmetro de entrada precisa ser [1,p] ou [p].
    :param use_dagger: se deve usar a transposta conjugada, o padrão é False.
    :return: Saída QTensor.

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
    
    Define uma classe de porta lógica IsingXY.

    :param has_params: Se possui parâmetros. Portas como RX, RY precisam ser configuradas como True; as sem parâmetros precisam ser False. O padrão é False.
    :param trainable: Se contém parâmetros a serem treinados. Se a camada usa dados de entrada externos para construir uma matriz de porta lógica, defina como False. Se contém parâmetros a serem treinados, defina como True. O padrão é False.
    :param init_params: Parâmetros de inicialização, usados para codificar dados clássicos QTensor. O padrão é None. Se for uma porta lógica com p parâmetros, a dimensão dos dados de entrada precisa ser [1,p] ou [p].
    :param wires: Índice do bit de atuação do fio, o padrão é None.
    :param dtype: A precisão dos dados da matriz interna da porta lógica pode ser definida como pyvqnet.kcomplex64 ou pyvqnet.kcomplex128, correspondendo respectivamente a parâmetros de entrada float ou double.
    :param use_dagger: Se deve usar a versão transposta conjugada desta porta, o padrão é False.
    :return: Um Module que pode ser usado para treinar o modelo.

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
    
    Aplica a porta lógica quântica phaseshift aos vetores de estado em ``q_machine``.

    :param q_machine: dispositivo de máquina virtual quântica.
    :param wires: índice do qubit.
    :param params: matriz de parâmetros, o padrão é None. Para uma função de operação de porta lógica com p parâmetros, a dimensão do parâmetro de entrada precisa ser [1,p] ou [p].
    :param use_dagger: se deve usar a transposta conjugada, o padrão é False.
    :return: Saída QTensor.

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
    
    Define uma classe de porta lógica PhaseShift.

    :param has_params: Se possui parâmetros. Portas como RX, RY precisam ser configuradas como True; as sem parâmetros precisam ser False. O padrão é False.
    :param trainable: Se contém parâmetros a serem treinados. Se a camada usa dados de entrada externos para construir uma matriz de porta lógica, defina como False. Se contém parâmetros a serem treinados, defina como True. O padrão é False.
    :param init_params: Parâmetros de inicialização, usados para codificar dados clássicos QTensor. O padrão é None. Se for uma porta lógica com p parâmetros, a dimensão dos dados de entrada precisa ser [1,p] ou [p].
    :param wires: Índice do bit de atuação do fio, o padrão é None.
    :param dtype: A precisão dos dados da matriz interna da porta lógica pode ser definida como pyvqnet.kcomplex64 ou pyvqnet.kcomplex128, correspondendo respectivamente a parâmetros de entrada float ou double.
    :param use_dagger: Se deve usar a versão transposta conjugada desta porta, o padrão é False.
    :return: Um Module que pode ser usado para treinar o modelo.

    Example::

        from pyvqnet.qnn.vqc import PhaseShift,QMachine
        device = QMachine(4)
        layer = PhaseShift(has_params= True, trainable= True, wires=1)
        batchsize = 2
        device.reset_states(batchsize)
        layer(q_machine = device)
        print(device.states)

sdg
--------------

.. py:function:: pyvqnet.qnn.vqc.sdg(q_machine, wires, params=None,  use_dagger=False)
    
    Aplica a porta lógica quântica sdg aos vetores de estado em ``q_machine``.

    :param q_machine: dispositivo de máquina virtual quântica.
    :param wires: índice do qubit.
    :param params: matriz de parâmetros, o padrão é None. Para uma função de operação de porta lógica com p parâmetros, a dimensão do parâmetro de entrada precisa ser [1,p] ou [p].
    :param use_dagger: se deve usar a transposta conjugada, o padrão é False.


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
    
    Define uma classe de porta lógica SDG.

    :param has_params: Se possui parâmetros. Portas como RX, RY precisam ser configuradas como True; as sem parâmetros precisam ser False. O padrão é False.
    :param trainable: Se contém parâmetros a serem treinados. Se a camada usa dados de entrada externos para construir uma matriz de porta lógica, defina como False. Se contém parâmetros a serem treinados, defina como True. O padrão é False.
    :param init_params: Parâmetros de inicialização, usados para codificar dados clássicos QTensor. O padrão é None. Se for uma porta lógica com p parâmetros, a dimensão dos dados de entrada precisa ser [1,p] ou [p].
    :param wires: Índice do bit de atuação do fio, o padrão é None.
    :param dtype: A precisão dos dados da matriz interna da porta lógica pode ser definida como pyvqnet.kcomplex64 ou pyvqnet.kcomplex128, correspondendo respectivamente a parâmetros de entrada float ou double.
    :param use_dagger: Se deve usar a versão transposta conjugada desta porta, o padrão é False.
    :return: Um Module que pode ser usado para treinar o modelo.

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
    
    Aplica a porta lógica quântica tdg aos vetores de estado em ``q_machine``.

    :param q_machine: dispositivo de máquina virtual quântica.
    :param wires: índice do qubit.
    :param params: matriz de parâmetros, o padrão é None. Para uma função de operação de porta lógica com p parâmetros, a dimensão do parâmetro de entrada precisa ser [1,p] ou [p].
    :param use_dagger: se deve usar a transposta conjugada, o padrão é False.


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
    
    Define uma classe de porta lógica TDG.

    :param has_params: Se possui parâmetros. Portas como RX, RY precisam ser configuradas como True; as sem parâmetros precisam ser False. O padrão é False.
    :param trainable: Se contém parâmetros a serem treinados. Se a camada usa dados de entrada externos para construir uma matriz de porta lógica, defina como False. Se contém parâmetros a serem treinados, defina como True. O padrão é False.
    :param init_params: Parâmetros de inicialização, usados para codificar dados clássicos QTensor. O padrão é None. Se for uma porta lógica com p parâmetros, a dimensão dos dados de entrada precisa ser [1,p] ou [p].
    :param wires: Índice do bit de atuação do fio, o padrão é None.
    :param dtype: A precisão dos dados da matriz interna da porta lógica pode ser definida como pyvqnet.kcomplex64 ou pyvqnet.kcomplex128, correspondendo respectivamente a parâmetros de entrada float ou double.
    :param use_dagger: Se deve usar a versão transposta conjugada desta porta, o padrão é False.
    :return: Um Module que pode ser usado para treinar o modelo.

    Example::
    
        from pyvqnet.qnn.vqc import TDG,QMachine
        device = QMachine(4)
        layer = TDG(wires=0)
        batchsize=1
        device.reset_states(1)
        layer(q_machine = device)
        print(device.states)

MultiRZ
-------------------------------------------------


.. py:class:: pyvqnet.qnn.vqc.MultiRZ(has_params: bool = False,trainable: bool = False,init_params=None,wires=None,dtype=pyvqnet.kcomplex64,use_dagger=False)
    
    Define uma classe de porta lógica MultiRZ.

    :param has_params: Se possui parâmetros. Portas como RX, RY precisam ser configuradas como True; as sem parâmetros precisam ser False. O padrão é False.
    :param trainable: Se contém parâmetros a serem treinados. Se a camada usa dados de entrada externos para construir uma matriz de porta lógica, defina como False. Se contém parâmetros a serem treinados, defina como True. O padrão é False.
    :param init_params: Parâmetros de inicialização, usados para codificar dados clássicos QTensor. O padrão é None. Se for uma porta lógica com p parâmetros, a dimensão dos dados de entrada precisa ser [1,p] ou [p].
    :param wires: Índice do bit de atuação do fio, o padrão é None.
    :param dtype: A precisão dos dados da matriz interna da porta lógica pode ser definida como pyvqnet.kcomplex64 ou pyvqnet.kcomplex128, correspondendo respectivamente a parâmetros de entrada float ou double.
    :param use_dagger: Se deve usar a versão transposta conjugada desta porta, o padrão é False.
    :return: Um Module que pode ser usado para treinar o modelo.

    Example::

        from pyvqnet.qnn.vqc import MultiRZ,QMachine
        device = QMachine(4)
        layer = MultiRZ(has_params= True, trainable= True, wires=[0,2])
        batchsize = 2
        device.reset_states(batchsize)
        layer(q_machine = device)
        print(device.states)

controlledphaseshift
-----------------------------

.. py:function:: pyvqnet.qnn.vqc.controlledphaseshift(q_machine, wires, params=None,  use_dagger=False)
    
    Aplica a porta lógica quântica controlledphaseshift aos vetores de estado em ``q_machine``.

    :param q_machine: dispositivo de máquina virtual quântica.
    :param wires: índice do qubit.
    :param params: matriz de parâmetros, o padrão é None. Para uma função de operação de porta lógica com p parâmetros, a dimensão do parâmetro de entrada precisa ser [1,p] ou [p].
    :param use_dagger: se deve usar a transposta conjugada, o padrão é False.


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
    
    Define uma classe de porta lógica ControlledPhaseShift.

    :param has_params: Se possui parâmetros. Portas como RX, RY precisam ser configuradas como True; as sem parâmetros precisam ser False. O padrão é False.
    :param trainable: Se contém parâmetros a serem treinados. Se a camada usa dados de entrada externos para construir uma matriz de porta lógica, defina como False. Se contém parâmetros a serem treinados, defina como True. O padrão é False.
    :param init_params: Parâmetros de inicialização, usados para codificar dados clássicos QTensor. O padrão é None. Se for uma porta lógica com p parâmetros, a dimensão dos dados de entrada precisa ser [1,p] ou [p].
    :param wires: Índice do bit de atuação do fio, o padrão é None.
    :param dtype: A precisão dos dados da matriz interna da porta lógica pode ser definida como pyvqnet.kcomplex64 ou pyvqnet.kcomplex128, correspondendo respectivamente a parâmetros de entrada float ou double.
    :param use_dagger: Se deve usar a versão transposta conjugada desta porta, o padrão é False.
    :return: Um Module que pode ser usado para treinar o modelo.

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
    
    Aplica a porta lógica quântica multicontrolledx aos vetores de estado em ``q_machine``.

    :param q_machine: dispositivo de máquina virtual quântica.
    :param wires: índice do qubit.
    :param params: matriz de parâmetros, o padrão é None.
    :param use_dagger: se deve usar a transposta conjugada, o padrão é False.
    :param control_values: valor de controle, o padrão é None, controla quando o bit é 1.


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
    
    Define uma classe de porta lógica MultiControlledX.

    :param has_params: Se possui parâmetros. Portas como RX, RY precisam ser configuradas como True; as sem parâmetros precisam ser False. O padrão é False.
    :param trainable: Se possui parâmetros a serem treinados. Se a camada usa dados de entrada externos para construir a matriz da porta lógica, defina como False. Se os parâmetros a serem treinados precisam ser inicializados a partir desta camada, defina como True. O padrão é False.
    :param init_params: Parâmetros de inicialização usados para codificar dados clássicos QTensor, o padrão é None.
    :param wires: Índice do bit de atuação do fio, o padrão é None.
    :param dtype: A precisão dos dados da matriz interna da porta lógica, que pode ser definida como pyvqnet.kcomplex64 ou pyvqnet.kcomplex128, correspondendo respectivamente a float ou double.
    :param use_dagger: Se deve usar a versão transposta conjugada da porta, o padrão é False.
    :param control_values: valor de controle, o padrão é None, controla quando o bit é 1.

    :return: Um Module que pode ser usado para treinar o modelo.

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
    
    Atua com portas lógicas quânticas nos vetores de estado em q_machine single_excitation.

    :param q_machine: dispositivo de máquina virtual quântica.
    :param wires: índice do qubit.
    :param params: matriz de parâmetros, o padrão é None. Para uma função de operação de porta lógica com p parâmetros, a dimensão do parâmetro de entrada precisa ser [1,p] ou [p].
    :param use_dagger: se deve usar a transposta conjugada, o padrão é False.


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
    
    Atua com portas lógicas quânticas nos vetores de estado em q_machine double_excitation.

    :param q_machine: dispositivo de máquina virtual quântica.
    :param wires: índice do qubit.
    :param params: matriz de parâmetros, o padrão é None. Para uma função de operação de porta lógica com p parâmetros, a dimensão do parâmetro de entrada precisa ser [1,p] ou [p].
    :param use_dagger: se deve usar a transposta conjugada, o padrão é False.


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

    Codifica características binárias ``basis_state`` no estado fundamental de n qubits em ``q_machine``.

    Por exemplo, para ``basis_state=([0, 1, 1])``, o estado fundamental do sistema quântico é :math:`|011 \rangle`.

    :param basis_state: entrada binária de tamanho ``(n)``.
    :param q_machine: dispositivo de máquina virtual quântica.


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

    Codifica as :math:`N` características no ângulo de rotação de :math:`n` qubits, onde :math:`N \leq n` em ``q_machine``.

    A rotação pode ser selecionada como: 'X', 'Y', 'Z', conforme a definição do parâmetro ``rotation``:

    * ``rotation='X'`` Usa a característica como ângulo para rotação RX.

    * ``rotation='Y'`` Usa a característica como ângulo para rotação RY.

    * ``rotation='Z'`` Usa a característica como ângulo para rotação RZ.

     ``wires`` denota o índice das portas de rotação nos qubits.

    :param input_feat: array representando os parâmetros.
    :param wires: índice do qubit.
    :param q_machine: Dispositivo de máquina virtual quântica.
    :param rotation: Porta de rotação, o padrão é "X".


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

    Codifica uma característica de :math:`2^n` em um vetor de amplitude de :math:`n` qubits em ``q_machine``.

    :param input_feature: Um array numpy representando os parâmetros.
    :param q_machine: Dispositivo de máquina virtual quântica.


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

    Aplica portas diagonais usando linhas IQP para codificar :math:`n` características em :math:`n` qubits de ``q_machine``.

    A codificação foi proposta por `Havlicek et al. (2018) <https://arxiv.org/pdf/1804.11326.pdf>`_.

    Especificando ``rep``, as linhas IQP básicas podem ser repetidas.

    :param input_feat: Um array numpy representando os parâmetros.
    :param q_machine: Dispositivo de máquina virtual quântica.
    :param rep: O número de vezes para repetir o bloco do circuito quântico, o número padrão é 1.


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

    Aplica rotações arbitrárias de um único qubit nos vetores de estado de ``q_machine``.

    .. math::

        R(\phi,\theta,\omega) = RZ(\omega)RY(\theta)RZ(\phi)= \begin{bmatrix}
        e^{-i(\phi+\omega)/2}\cos(\theta/2) & -e^{i(\phi-\omega)/2}\sin(\theta/2) \\
        e^{-i(\phi-\omega)/2}\sin(\theta/2) & e^{i(\phi+\omega)/2}\cos(\theta/2)
        \end{bmatrix}.


    :param q_machine: Dispositivo de máquina virtual quântica.
    :param wire: Índice do qubit.
    :param params: Parâmetros :math:`[\phi, \theta, \omega]`.
    :return: Saída QTensor.

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

	Circuito Rot controlado.

    .. math:: CR(\phi, \theta, \omega) = \begin{bmatrix}
            1 & 0 & 0 & 0 \\
            0 & 1 & 0 & 0\\
            0 & 0 & e^{-i(\phi+\omega)/2}\cos(\theta/2) & -e^{i(\phi-\omega)/2}\sin(\theta/2)\\
            0 & 0 & e^{-i(\phi-\omega)/2}\sin(\theta/2) & e^{i(\phi+\omega)/2}\cos(\theta/2)
        \end{bmatrix}.
    
    :param para: array numpy representando os parâmetros.
    :param control_qubits: Índice dos bits de controle.
    :param rot_wire: Índice dos bits de rotação.
    :param q_machine: Dispositivo de máquina virtual quântica.
    :return: Saída QTensor.

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

    Aplica a operação Hadamard Controlada em ``q_machine``.

    .. math:: CH = \begin{bmatrix}
            1 & 0 & 0 & 0 \\
            0 & 1 & 0 & 0 \\
            0 & 0 & \frac{1}{\sqrt{2}} & \frac{1}{\sqrt{2}} \\
            0 & 0 & \frac{1}{\sqrt{2}} & -\frac{1}{\sqrt{2}}
        \end{bmatrix}.

    :param wires: Índice do qubit, o primeiro é o bit de controle, e o comprimento da lista é 2.
    :param q_machine: Dispositivo de máquina virtual quântica.

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

    Aplica a lógica Controlled-controlled-Z em ``q_machine``.

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
    
    :param wires: Lista de índices dos qubits, o primeiro bit é o bit de controle. O comprimento da lista é 3.
    :param q_machine: Dispositivo de máquina virtual quântica.

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

    Um operador de excitação única de cluster acoplado para exponenciar o produto tensorial de uma matriz de Pauli. A forma matricial é dada por:

    .. math::

        \hat{U}_{pr}(\theta) = \mathrm{exp} \{ \theta_{pr} (\hat{c}_p^\dagger \hat{c}_r
        -\mathrm{H.c.}) \},

    :param weight: O parâmetro no qubit p tem apenas um elemento.
    :param wires: Denota um subconjunto de índices de qubit no intervalo [r, p]. O comprimento mínimo deve ser 2. O primeiro valor do índice é interpretado como r e o último como p.
                 O índice intermediário sofre ação da porta CNOT para calcular a paridade do conjunto de qubits.
    :param q_machine: Dispositivo de máquina virtual quântica.

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

    O operador de dupla excitação de cluster acoplado que exponencia o produto tensorial da matriz de Pauli. A forma matricial é dada por:

    .. math::

        \hat{U}_{pqrs}(\theta) = \mathrm{exp} \{ \theta (\hat{c}_p^\dagger \hat{c}_q^\dagger
        \hat{c}_r \hat{c}_s - \mathrm{H.c.}) \},

    onde :math:`\hat{c}` e :math:`\hat{c}^\dagger` são os operadores de aniquilação e criação de férmions e os índices :math:`r, s` e :math:`p, q` nos orbitais moleculares ocupados e
    vazios, respectivamente. Usando a `transformação de Jordan-Wigner <https://arxiv.org/abs/1208.5986>`_, o operador fermiônico definido acima pode ser escrito
    de acordo com a matriz de Pauli (para mais detalhes, veja `arXiv:1805.04340 <https://arxiv.org/abs/1805.04340>`_)

    .. math::

        \hat{U}_{pqrs}(\theta) = \mathrm{exp} \Big\{
        \frac{i\theta}{8} \bigotimes_{b=s+1}^{r-1} \hat{Z}_b \bigotimes_{a=q+1}^{p-1}
        \hat{Z}_a (\hat{X}_s \hat{X}_r \hat{Y}_q \hat{X}_p +
        \hat{Y}_s \hat{X}_r \hat{Y}_q \hat{Y}_p +\\ \hat{X}_s \hat{Y}_r \hat{Y}_q \hat{Y}_p +
        \hat{X}_s \hat{X}_r \hat{X}_q \hat{Y}_p - \mathrm{H.c.}  ) \Big\}

    :param weight: Parâmetro variável.
    :param wires1: A lista de índices de qubits representando o subconjunto de qubits ocupados no intervalo [s, r]. O primeiro índice é interpretado como s, o último como r.
     Portas CNOT operam em índices intermediários para calcular a paridade de um conjunto de qubits.
    :param wires2: A lista de índices de qubits representando o subconjunto de qubits ocupados no intervalo [q, p]. O primeiro índice é interpretado como q, o último como p. 
     Portas CNOT operam em índices intermediários para calcular a paridade de um conjunto de qubits.
    :param q_machine: Dispositivo de máquina virtual quântica.

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

    Realiza o projeto de excitação única e dupla de cluster unitário acoplado (UCCSD). UCCSD é o projeto VQE proposto, comumente usado para executar simulações de química quântica.

    Dentro da aproximação de Trotter de primeira ordem, a função unitária UCCSD é dada por:

    .. math::

        \hat{U}(\vec{\theta}) =
        \prod_{p > r} \mathrm{exp} \Big\{\theta_{pr}
        (\hat{c}_p^\dagger \hat{c}_r-\mathrm{H.c.}) \Big\}
        \prod_{p > q > r > s} \mathrm{exp} \Big\{\theta_{pqrs}
        (\hat{c}_p^\dagger \hat{c}_q^\dagger \hat{c}_r \hat{c}_s-\mathrm{H.c.}) \Big\}

    onde :math:`\hat{c}` e :math:`\hat{c}^\dagger` são os operadores de aniquilação e
    criação de férmions e os índices :math:`r, s` e :math:`p, q` nos orbitais moleculares ocupados e
    vazios, respectivamente. (Para mais detalhes, veja `arXiv:1805.04340 <https://arxiv.org/abs/1805.04340>`_):


    :param weights: Um tensor ``(len(s_wires)+ len(d_wires))`` contendo os parâmetros
         :math:`\theta_{pr}` e :math:`\theta_{pqrs}` de rotação Z de entrada
         ``FermionicSingleExcitation`` e ``FermionicDoubleExcitation``.
    :param wires: Indexação de qubits dos efeitos do template
    :param s_wires: Uma sequência de listas ``[r,...,p]`` contendo índices de qubit
         produzidos por uma excitação única
         :math:`\vert r, p \rangle = \hat{c}_p^\dagger \hat{c}_r \vert \mathrm{HF} \rangle`,
         onde :math:`\vert \mathrm{HF} \rangle` representa o estado de referência de Hartree-Fock.
    :param d_wires: sequência de listas, cada lista contendo duas listas
         especificando índices ``[s, ...,r]`` e ``[q,...,p]``
         Define dupla excitação: :math:`\vert s, r, q, p \rangle = \hat{c}_p^\dagger \hat{c}_q^\dagger \hat{c}_r\hat{c}_s \vert \mathrm{HF} \rangle`.
    :param init_state: comprimento ``len(wires)`` vetor de representação de número de ocupação
         estado de alta frequência. ``init_state`` é o estado de inicialização do qubit.
    :param q_machine: Dispositivo de máquina virtual quântica.

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

    Circuito de evolução-Z de primeira ordem.

    Para 3 qubits e 2 repetições, o circuito é representado como:

    .. parsed-literal::

        ┌───┐┌──────────────┐┌───┐┌──────────────┐
        ┤ H ├┤ U1(2.0*x[0]) ├┤ H ├┤ U1(2.0*x[0]) ├
        ├───┤├──────────────┤├───┤├──────────────┤
        ┤ H ├┤ U1(2.0*x[1]) ├┤ H ├┤ U1(2.0*x[1]) ├
        ├───┤├──────────────┤├───┤├──────────────┤
        ┤ H ├┤ U1(2.0*x[2]) ├┤ H ├┤ U1(2.0*x[2]) ├
        └───┘└──────────────┘└───┘└──────────────┘
    
    A string de Pauli é fixada como ``Z``. Assim, a expansão de primeira ordem será um circuito sem portas de emaranhamento.

    :param input_feat: Um array representando os parâmetros de entrada.
    :param q_machine: Máquina quântica.
    :param data_map_func: Matriz de mapeamento de parâmetros, projetada como ``data_map = lambda x: x``.
    :param rep: Número de repetições do módulo.
    
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

    Circuitos de evolução Pauli-Z de segunda ordem.

    Para 3 qubits, 1 repetição e emaranhamento linear, o circuito é representado como:

    .. parsed-literal::

        ┌───┐┌─────────────────┐
        ┤ H ├┤ U1(2.0*φ(x[0])) ├──■────────────────────────────■────────────────────────────────────
        ├───┤├─────────────────┤┌─┴─┐┌──────────────────────┐┌─┴─┐
        ┤ H ├┤ U1(2.0*φ(x[1])) ├┤ X ├┤ U1(2.0*φ(x[0],x[1])) ├┤ X ├──■────────────────────────────■──
        ├───┤├─────────────────┤└───┘└──────────────────────┘└───┘┌─┴─┐┌──────────────────────┐┌─┴─┐
        ┤ H ├┤ U1(2.0*φ(x[2])) ├──────────────────────────────────┤ X ├┤ U1(2.0*φ(x[1],x[2])) ├┤ X ├
        └───┘└─────────────────┘                                  └───┘└──────────────────────┘└───┘
    
    onde ``φ`` é a função não linear clássica que padrão é ``φ(x) = x`` e ``φ(x,y) = (pi - x)(pi - y)``, projetada como:
    
    .. code-block::
        
        def data_map_func(x):
            coeff = x if x.shape[-1] == 1 else ft.reduce(lambda x, y: (np.pi - x) * (np.pi - y), x)
            return coeff

    :param input_feat: Um array representando os parâmetros de entrada.
    :param q_machine: Máquina quântica.
    :param data_map_func: Matriz de mapeamento de parâmetros.
    :param entanglement: estrutura de emaranhamento especificada.
    :param rep: Número de repetições do módulo.
    
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

    Aplica todas as operações ``SingleExcitation`` e ``DoubleExcitation`` no ``q_machine`` ao estado inicial de Hartree-Fock, preparando o estado de associação molecular.

    :param weights: QTensor de tamanho ``(len(singles) + len(doubles),)`` contendo ângulos que entram nas operações vqc.qCircuit.single_excitation e vqc.qCircuit.double_excitation sequencialmente
    :param q_machine: Máquina quântica.
    :param hf_state: Representa o comprimento do estado de Hartree-Fock ``len(wires)`` Vetor de contagem de ocupação, ``hf_state`` é usado para inicializar os fios.
    :param wires: Qubits sobre os quais atuar.
    :param singles: Sequência de listas com os dois índices de qubit nos quais a operação single_excitation atua.
    :param doubles: Sequência de listas com os dois índices de qubit nos quais a operação double_excitation atua.

    Por exemplo, o circuito quântico para o caso de dois elétrons e seis qubits é mostrado abaixo:
    
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

    Implementa um circuito que fornece um todo que pode ser usado para realizar rotações de base monolíticas precisas.

    :class:`~.vqc.qCircuit.VQC_BasisRotation` realiza a seguinte transformação determinada por férmions de partícula única dada em `arXiv:1711.04789 <https://arxiv.org/abs/1711.04789>`_\ :math:`U(u)`
    
    .. math::

        U(u) = \exp{\left( \sum_{pq} \left[\log u \right]_{pq} (a_p^\dagger a_q - a_q^\dagger a_p) \right)}.
    
    :math:`U(u)` usando o esquema dado no artigo `Optica, 3, 1460 (2016) <https://opg.optica.org/optica/fulltext.cfm?uri=optica-3-12-1460&id=355743>`_\.
    A decomposição da matriz de entrada é eficientemente implementada por uma série de portas :class:`~vqc.qCircuit.phaseshift` e :class:`~vqc.qCircuit.single_exitation`.
    

    :param q_machine: Máquina quântica.
    :param wires: Qubits sobre os quais atuar.
    :param unitary_matrix: Especifica a matriz da transformação de base.
    :param check: Testa se `unitary_matrix` é uma matriz.

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

    Um circuito quântico que reduz a amostragem dos dados.

    Para reduzir o número de qubits em um circuito, pares de qubits são criados primeiro no sistema. Após emparelhar inicialmente todos os qubits, uma unitária generalizada de 2 qubits é aplicada a cada par de qubits.
    E após aplicar a unitária de dois qubits, um qubit em cada par de qubits é ignorado no restante da rede neural.

    :param sources_wires: O índice do qubit de origem que será ignorado.
    :param sinks_wires: O índice do qubit de destino a ser mantido.
    :param params: Parâmetros de entrada.
    :param q_machine: Dispositivo de máquina virtual quântica.

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

    19 ansätze diferentes do artigo `Expressibility and entangling capability of parameterized quantum circuits for hybrid quantum-classical algorithms <https://arxiv.org/pdf/1905.10876.pdf>`_.

    :param type: Tipo de circuito de 1 a 19, total de 19 fios.
    :param num_wires: Número de qubits.
    :param depth: Profundidade do circuito.
    :param name: Nome, padrão "".

    :return:
        Uma instância de ExpressiveEntanglingAnsatz

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

    Codifica um inteiro sem sinal `m` em um qubit e então adiciona `k` a ele.

    .. math:: \text{Sum(k)}\vert m \rangle = \vert m + k \rangle.

    Esta operação unitária é implementada da seguinte forma:

    (1). Converte o estado da base computacional para a base de Fourier aplicando a QFT ao estado :math:`\vert m \rangle`.

    (2). Usa a porta :math:`R_Z` para girar o :math:`j`-ésimo qubit pelo ângulo :math:`\frac{2k\pi}{2^{j}}`, resultando na nova fase :math:`\frac{2(m + k)\pi}{2^{j}}`.

    (3). Aplica a QFT inversa de volta à base computacional e obtém :math:`m+k`.

    :param q_machine: A máquina quântica para simular.
    :param m: O inteiro clássico a ser incorporado no registrador.
    :param k: O inteiro clássico a ser adicionado ao registrador.

    :retrun: Retorna a representação binária da soma alvo.

    .. note::

        Observe que o número de bits usados por ``q_machine`` precisa ser suficiente para codificar o valor binário da soma resultante usando o estado base X.

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

    Soma os inteiros sem sinal codificados nos dois qubits.

    .. math:: \text{Sum}_2\vert m \rangle \vert k \rangle \vert 0 \rangle = \vert m \rangle \vert k \rangle \vert m+k \rangle

    Neste caso, podemos entender o terceiro registrador (inicialmente em :math:`0`) como um contador que contará o número de unidades que :math:`m` e :math:`k` somam. A fatoração binária tornará isso fácil. Se tivermos :math:`\vert m \rangle = \vert \overline{q_0q_1q_2} \rangle`, então se :math:`q_2 = 1`, devemos adicionar :math:`1` ao contador, caso contrário, não adicionar nada. Em geral, se o :math:`i`-ésimo qubit estiver no estado :math:`\vert 1 \rangle`, devemos adicionar :math:`2^{n-i-1}` unidades, caso contrário, adicionar 0.

    :param q_machine: A máquina quântica para simular.
    :param m: O inteiro clássico incorporado no registrador como lhs.
    :param k: O inteiro clássico incorporado no registrador como rhs.
    :param wires_m: O índice do qubit para codificar m.
    :param wires_k: O índice do qubit para codificar k.
    :param wires_solution: O índice do qubit para codificar a solução.

    :retrun: Retorna a representação binária da soma alvo.

    .. note::

        O número de bits usados em ``wires_m`` precisa ser suficiente para codificar o valor binário de `m` usando o estado base X.
        ``wires_k`` usa bits suficientes para codificar o valor binário de `k` usando o estado base X.
        ``wires_solution`` usa bits suficientes para codificar o valor binário do resultado usando o estado base X.

    Example::

        import numpy as np
        from pyvqnet.qnn.vqc import QMachine,Samples, vqc_qft_add_two_register


        wires_m = [0, 1, 2]             # qubits necessários para codificar m
        wires_k = [3, 4, 5]             # qubits necessários para codificar k
        wires_solution = [6, 7, 8, 9]   # qubits necessários para codificar a solução
        dev = QMachine(len(wires_m) + len(wires_k) + len(wires_solution))

        vqc_qft_add_two_register(dev,3, 7, wires_m, wires_k, wires_solution)

        ma = Samples(wires=wires_solution)
        y = ma(q_machine=dev)
        print(y)


vqc_qft_mul
-------------------------------------

.. py:function:: vqc_qft_mul(q_machine, m, k, wires_m, wires_k, wires_solution)

    Multiplica os valores codificados em dois qubits.

    .. math:: \text{Mul}\vert m \rangle \vert k \rangle \vert 0 \rangle = \vert m \rangle \vert k \rangle \vert m\cdot k \rangle

    :param q_machine: A máquina quântica para simular.
    :param m: O inteiro clássico incorporado em um registrador como o lado esquerdo.
    :param k: O inteiro clássico incorporado em um registrador como o lado direito.
    :param wires_m: O índice do qubit para codificar m.
    :param wires_k: O índice do qubit para codificar k.
    :param wires_solution: O índice do qubit para codificar a solução.

    :retrun: Retorna a representação binária do produto alvo.

    .. note::

        ``wires_m`` precisa usar bits suficientes para codificar o valor binário de `m` usando o estado base X.
        ``wires_k`` usa bits suficientes para codificar o valor binário de `k` usando o estado base X.
        ``wires_solution`` usa bits suficientes para codificar o valor binário do resultado usando o estado base X.

    Example::

        import numpy as np
        from pyvqnet.qnn.vqc import QMachine,Samples, vqc_qft_mul
        wires_m = [0, 1, 2]           # qubits necessários para codificar m
        wires_k = [3, 4, 5]           # qubits necessários para codificar k
        wires_solution = [6, 7, 8, 9, 10]  # qubits necessários para codificar a solução
        
        dev = QMachine(len(wires_m) + len(wires_k) + len(wires_solution))

        vqc_qft_mul(dev,3, 7, wires_m, wires_k, wires_solution)


        ma = Samples(wires=wires_solution)
        y = ma(q_machine=dev)
        print(y)
        #[[1,0,1,0,1]]

VQC_FABLE
--------------------

.. py:class:: pyvqnet.qnn.vqc.VQC_FABLE(wires)

    Constrói um QCircuit baseado em VQC usando um método rápido de codificação aproximada de blocos. Para matrizes de certas estruturas [`arXiv:2205.00081 <https://arxiv.org/abs/2205.00081>`_], o método FABLE pode simplificar o circuito de codificação de blocos sem perder precisão.

    :param wires: O índice qlist ao qual o operador atua.

    :return: Retorna uma instância da classe FABLE baseada em VQC.

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

    Constrói um QCircuit baseado em VQC usando Combinação Linear de Unidades (LCU), `Hamiltonian Simulation via Qubitization <https://arxiv.org/abs/1610.06546>`_.
    O dtype de entrada pode ser kfloat32, kfloat64, kcomplex64, kcomplex128.
    A entrada deve ser Hermitiana.

    :param wires: Índice qlist no qual o operador atua, pode exigir qubits auxiliares.
    :param check_hermitian: Verifica se a entrada é Hermitiana, padrão: True.

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

    Implementa o circuito de `transformação de valor singular quântica <https://arxiv.org/abs/1806.01838>`__ (QSVT).

    :param A: A matriz geral :math:`(n \times m)` a ser codificada.
    :param angles: A lista de ângulos para deslocar e obter o polinômio desejado.
    :param wires: Os índices de qubit nos quais A atua.

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

Medições Quânticas
=============================================

VQC_Purity
----------------------

.. py:function:: pyvqnet.qnn.vqc.VQC_Purity(state, qubits_idx, num_wires)

    Calcula a pureza em um qubit específico ``qubits_idx`` a partir do vetor de estado ``state``.

    .. math::
        \gamma = \text{Tr}(\rho^2)

    onde :math:`\rho` é uma matriz densidade. A pureza de um estado quântico normalizado satisfaz :math:`\frac{1}{d} \leq \gamma \leq 1`,
    onde :math:`d` é a dimensão do espaço de Hilbert.
    A pureza do estado puro é 1.

    :param state: Estado quântico obtido de pyqpanda get_qstate()
    :param qubits_idx: Índice do qubit para o qual calcular a pureza
    :param num_wires: Índice do qubit

    :return: pureza

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

    Retorna a variância da medição do observável fornecido ``obs`` nos vetores de estado em ``q_machine``.

    :param q_machine: Estado quântico obtido de pyqpanda get_qstate()
    :param obs: observáveis

    :return: valor da variância

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

    Calcula a matriz densidade dos estados quânticos ``state`` sobre um conjunto específico de qubits ``indices``.

    :param state: Uma lista 1D de vetores de estado. O tamanho desta lista deve ser ``(2**N,)`` Para o número de qubits ``N``, qstate deve começar de 000 -> 111.
    :param indices: Uma lista de índices de qubit no subsistema considerado.

    :return: Uma matriz densidade de tamanho "(b, 2**len(indices), 2**len(indices))".

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

    Calcula as medições de probabilidade de circuitos quânticos em bits específicos

    :param wires: Índice do qubit de medição.
    :param name: nome do módulo

    .. py:method:: forward(q_machine)

        Realiza cálculos de medição de probabilidade

        :param q_machine: simulador de vetor de estado quântico
        :return: resultados da medição de probabilidade

    .. note::

        Os resultados da medição de probabilidade calculados usando esta classe são geralmente [b, len(wires)], onde b é o número do lote b de q_machine.reset_states(b).

 

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

    Calcula os resultados de medição de um circuito quântico. Suporta entrada de observáveis ``obs``. Pode ser no formato de dicionário, representando um observável composto por múltiplos operadores de Pauli, ou no formato de lista, representando uma lista de observáveis com múltiplos valores esperados.
    Por exemplo:

        {\'X0\': 0.23} indica um efeito PauliX no qubit 0, com um coeficiente de 0.23.

        {\'X1 Z2\': 2.4,\'Y2\': -0.5} corresponde ao valor observado 2.4 * X1 @ Z2 - 0.5 * Y2.

        [{\'X1 Z2\': 4,\'Z1 Z0\': 3},{\'X1 Y2 Z0\': 3.5}] corresponde aos dois valores observados 4 * X1 @ Z2 + 3 * Z1 @ Z0 e 3.5 * X1 @ Y2 @ Z0.

    :param obs: observáveis string dict do operador pauli.

    .. note::

        Se ``obs`` for uma lista, o resultado da medição calculado usando esta classe é geralmente [b, comprimento da lista de obs], onde b é o número do lote b de q_machine.reset_states(b).

        Se ``obs`` for um dicionário, o resultado da medição calculado usando esta classe é geralmente [b,1], onde b é o número do lote b de q_machine.reset_states(b).

    .. py:method:: forward(q_machine)

        Realiza a operação de medição

        :param q_machine: simulador de vetor de estado quântico
        :return: resultado da medição, QTensor.



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
    
    Obtém o resultado da observação ``obs`` com ``shots`` nos fios especificados ``wires``.

    .. py:method:: forward(q_machine)

        Realiza operações de amostragem.

        :param q_machine: O simulador de vetor de estado quântico em uso
        :return: Resultados da medição, QTensor.

    .. note::

        Os resultados da medição calculados usando esta classe são geralmente [b, shots, len(wires)], onde b é o número do lote b de q_machine.reset_states(b).

    :param wires: Índice do qubit de amostragem. Valor padrão: None, usa todos os bits do simulador em tempo de execução.
    :param obs: Este valor só pode ser None.
    :param shots: Número de repetições de amostragem, valor padrão: 1.
    :param name: Nome deste módulo, valor padrão: "".
    :return: Uma classe de método de medição

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

    Calcula a expectativa de um observável Hermitiano ``obs`` de um circuito quântico.
    
    :param obs: Quantidade Hermitiana.
    :param name: O nome do módulo, padrão: "".
    :return: Uma instância de HermitianExpval.

    .. py:method:: forward(q_machine)

        Realiza a medição Hermitiana.

        :param q_machine: simulador de vetor de estado quântico
        :return: resultado da medição, QTensor.

    .. note::

        O resultado da medição calculado usando esta classe é geralmente [b,1], onde b é o número do lote b de q_machine.reset_states(b).

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


Modelos de circuito de variação quântica comumente usados
=========================================================

VQC_HardwareEfficientAnsatz
-----------------------------------------

.. py:class:: pyvqnet.qnn.vqc.VQC_HardwareEfficientAnsatz(n_qubits,single_rot_gate_list,entangle_gate="CNOT",entangle_rules='linear',depth=1,initial=None,dtype=None)

    A implementação do Hardware Efficient Ansatz introduzido no artigo:`Hardware-efficient Variational Quantum Eigensolver for Small Molecules <https://arxiv.org/pdf/1704.05018.pdf>`__.

    :param n_qubits: Número de qubits.
    :param single_rot_gate_list: Uma lista de portas de rotação de qubit único é construída por uma ou várias portas de rotação que atuam em cada qubit. Atualmente suporta Rx, Ry, Rz.
    :param entangle_gate: A porta de emaranhamento não parametrizada. CNOT, CZ são suportados. Padrão: CNOT.
    :param entangle_rules: Como a porta de emaranhamento é usada no circuito. 'linear' significa que a porta de emaranhamento atuará em cada par de qubits vizinhos. 'all' significa que a porta de emaranhamento atuará em quaisquer dois qubits. Padrão: linear.
    :param depth: A profundidade do ansatz, padrão: 1.
    :param initial: valor inicial único para parâmetros, padrão: None, este módulo inicializará os parâmetros aleatoriamente.
    :param dtype: tipo de dado dos parâmetros.
    :return: uma instância de VQC_HardwareEfficientAnsatz.

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

    Uma camada que consiste em uma rotação de qubit único com um parâmetro em cada qubit, seguida por uma cadeia fechada ou combinação em anel de múltiplas portas CNOT.

    Um anel de porta CNOT conecta cada qubit aos seus vizinhos, com o último qubit sendo considerado vizinho do primeiro qubit.

    :param num_layers: número de camadas de repetição, padrão: 1.
    :param num_qubits: número de qubits, padrão: 1.
    :param rotation: porta de qubit único com um parâmetro a ser usada, padrão: `RX`
    :param initial: valor inicializado para todos os parâmetros. padrão: None, os parâmetros serão inicializados aleatoriamente.
    :param dtype: tipo de dado do parâmetro, padrão: None, usa float32.
    :return: Uma instância de VQC_BasicEntanglerTemplate

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

    Uma camada que consiste em uma rotação de qubit único e um emaranhador, veja `circuit-centric classifier design <https://arxiv.org/abs/1804.00633>`__.

    :param num_layers: número de camadas de repetição, padrão: 1.
    :param num_qubits: número de qubits, padrão: 1.
    :param ranges: sequência que determina o parâmetro de alcance para cada camada subsequente; padrão: None
                                usando :math: `r=l \mod M` para a :math:`l`-ésima camada e :math:`M` qubits.
    :param initial: valor inicial para todos os parâmetros. padrão: None, inicializado aleatoriamente.
    :param dtype: tipo de dado do parâmetro, padrão: None, usa float32.
    :return: Uma instância de VQC_StronglyEntanglingTemplate.

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

    Usa RZ, RY, RZ para criar circuitos quânticos variacionais que codificam dados clássicos em estados quânticos.
    Referência `Quantum embeddings for machine learning <https://arxiv.org/abs/2001.03622>`_.
    Após a classe ser inicializada, sua função membro ``compute_circuit`` é uma função de execução, que pode ser inserida como parâmetro.
    A classe ``QuantumLayerV2`` constitui uma camada do modelo de aprendizado de máquina quântico.

    :param num_repetitions_input: número de repetições para codificar a entrada em um submódulo.
    :param depth_input: número da dimensão de entrada.
    :param num_unitary_layers: número de repetições de portas quânticas variacionais.
    :param num_repetitions: número de repetições do submódulo.
    :param initial: valor de inicialização do parâmetro, padrão é None
    :param dtype: tipo do parâmetro, padrão é None, usa float32.
    :param name: nome da classe
    :return: Uma instância de VQC_QuantumEmbedding.

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


Interface do modelo de aprendizado de máquina quântico
======================================================

Quanvolution
---------------------------------------------------------------

.. py:class:: pyvqnet.qnn.qcnn.Quanvolution(params_shape, stride=(1, 1), kernel_initializer=quantum_uniform, machine_type_or_cloud_token: str = "cpu")

    Baseado na convolução quântica implementada em "Quanvolutional Neural Networks: Powering Image Recognition with Quantum Circuits" (https://arxiv.org/abs/1904.04767), o filtro de convolução clássico é substituído por um circuito quântico variacional para obter uma rede neural convolutional quântica com um filtro de convolução quântico.

    :param params_shape: A forma dos parâmetros, que deve ser bidimensional.
    :param stride: O tamanho do passo da janela de corte, o padrão é (1,1).
    :param kernel_initializer: Parâmetros do inicializador do kernel de convolução.
    :param machine_type_or_cloud_token: String do tipo de máquina ou token Qcloud, padrão é "cpu".
    :return: Uma instância de Quanvolution.

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

    O algoritmo Quantum Data Re-uploading (QDRL) baseado em "Data re-uploading for a universal quantum classifier" (https://arxiv.org/abs/1907.02085) é um modelo de reenvio de dados quânticos que combina circuitos quânticos com redes neurais clássicas.

    :param nq: O número de bits quânticos (qubits) usados no circuito quântico. Isso determina a escala do sistema quântico que o modelo irá manipular.
    :return: Uma instância de QDRL.

    Example::

        import numpy as np
        from pyvqnet.dtype import kfloat32
        from pyvqnet.qnn.qdrl_vqc import QDRL
        import pyvqnet.tensor as tensor

        # Define o número de bits quânticos (qubits)
        nq = 1

        # Inicializa o modelo
        model = QDRL(nq)

        # Cria um exemplo de entrada (assume que a entrada são dados de formato (batch_size, 3))
        # Suponha que temos um batch_size de 4 e cada entrada tem 3 características
        x_input = tensor.QTensor(np.random.randn(4, 3), dtype=kfloat32)

        # Passa a entrada pelo modelo
        output = model(x_input)

        output.backward()

        # Saída do resultado
        print("Saída do modelo:")
        print(output)


QGRU
------------------------------------------------------------

.. py:class:: pyvqnet.qnn.qgru.QGRU(para_num, num_of_qubits,input_size,hidden_size,batch_first=True)

    GRU (Gated Recurrent Unit) baseado em circuitos quânticos variacionais, usando circuitos quânticos para atualização de estado e retenção de memória.

    :param para_num: O número de parâmetros no circuito quântico.
    :param num_of_qubits: O número de qubits.
    :param input_size: A dimensão das características dos dados de entrada.
    :param hidden_size: A dimensão da unidade oculta.
    :param batch_first: Se a primeira dimensão da entrada é o tamanho do lote.
    :return: Uma instância de QGRU.

    Example::

        import numpy as np
        from pyvqnet.tensor import tensor
        from pyvqnet.qnn.qgru import QGRU
        from pyvqnet.dtype import kfloat32
        # Exemplo de uso
 
        # Define parâmetros
        para_num = 8
        num_of_qubits = 8
        input_size = 4
        hidden_size = 4
        batch_size = 1
        seq_length = 1
        # Cria modelo QGRU
        qgru = QGRU(para_num, num_of_qubits, input_size, hidden_size, batch_first=True)

        # Cria dados de entrada
        x = tensor.QTensor(np.random.randn(batch_size, seq_length, input_size), dtype=kfloat32)

        # Chama o modelo
        output, h_t = qgru(x)
        output.backward()

        print("Forma da saída:", output.shape)  # Forma da saída
        print("Forma de h_t:", h_t.shape)  # Forma do estado oculto final

QLSTM
-------------------------------------------------------------

.. py:class:: pyvqnet.qnn.qlstm.QLSTM(para_num, num_of_qubits,input_size, hidden_size,batch_first=True)
    
    QLSTM (Quantum Long Short-Term Memory) é um modelo híbrido que combina computação quântica e LSTM clássica, projetado para usar o paralelismo da computação quântica e a capacidade de memória da LSTM clássica para processar dados sequenciais.

    :param para_num: O número de parâmetros no circuito quântico.
    :param num_of_qubits: O número de bits quânticos.
    :param input_size: A dimensão das características dos dados de entrada.
    :param hidden_size: A dimensão da unidade oculta.
    :param batch_first: Se a primeira dimensão da entrada é o tamanho do lote.
    :return: Uma instância de QLSTM.

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

        print("Forma da saída:", output.shape)
        print("Forma de h_t:", h_t.shape)
        print("Forma de c_t:", c_t.shape)

QMLPModel
--------------------------------------------------------------

.. py:class:: pyvqnet.qnn.qmlp.qmlp.QMLPModel(input_channels: int,output_channels: int,num_qubits: int, kernel: _size_type,stride: _size_type,padding: _padding_type = "valid",weight_initializer: Union[Callable, None] = None,bias_initializer: Union[Callable, None] = None,use_bias: bool = True,dtype: Union[int, None] = None)
    
    QMLPModel é uma rede neural inspirada em quântica baseada em QMLP: An Error-Tolerant Nonlinear Quantum MLP Architecture using Parameterized Two-Qubit Gates (https://arxiv.org/abs/2206.01345). QMLPModel combina circuitos quânticos com operações clássicas de redes neurais, como pooling e camadas totalmente conectadas. É projetado para processar dados quânticos e extrair características relevantes através de operações quânticas e camadas clássicas.

    :param input_channels: O número de características de entrada.
    :param output_channels: O número de características de saída.
    :param num_qubits: O número de qubits.
    :param kernel: O tamanho da janela de pooling média.
    :param stride: O fator de tamanho do passo para redução de amostragem.
    :param padding: O método de preenchimento, opcional "valid" ou "same".
    :param weight_initializer: Inicializador de pesos, padrão é distribuição normal.
    :param bias_initializer: Inicializador de viés, padrão é inicialização zero.
    :param use_bias: Se deve usar viés, padrão é True.
    :param dtype: Padrão é None, usa o tipo de dado padrão.
    :return: Uma instância de QMLPModel.

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

        print("Forma da saída:", output.shape)



QRLModel
-------------------------------------------------------------

.. py:class:: pyvqnet.qnn.qrl.QRLModel(num_qubits, n_layers)

    Modelo de aprendizado por reforço quântico profundo usando circuitos quânticos variacionais em :ref:`QDRL_DEMO`.

    :param num_qubits: O número de qubits usados no circuito quântico.
    :param n_layers: O número de camadas no circuito quântico variacional.
    :return: Uma instância de QRLModel.

    Example::

        from pyvqnet.tensor import tensor, QTensor
        from pyvqnet.qnn.qrl import QRLModel

        num_qubits = 4
        model = QRLModel(num_qubits=num_qubits, n_layers=2)

        batch_size = 3
        x = QTensor([[1.1, 0.3, 1.2, 0.6], [0.2, 1.1, 0, 1.1], [1.3, 1.3, 0.3, 0.3]])
        output = model(x)
        output.backward()

        print("Saída do modelo:", output)


QRNN
------------------------------------------------------------------

.. py:class:: pyvqnet.qnn.qrnn.QRNN(para_num, num_of_qubits=4,input_size=100,hidden_size=100,batch_first=True)

    QRNN (Quantum Recurrent Neural Network) é uma rede neural recorrente quântica projetada para processar dados sequenciais e capturar dependências de longo prazo na sequência.

    :param para_num: O número de parâmetros no circuito quântico.
    :param num_of_qubits: O número de bits quânticos.
    :param input_size: A dimensão das características dos dados de entrada.
    :param hidden_size: A dimensão da unidade oculta.
    :param batch_first: Se a primeira dimensão da entrada é o tamanho do lote, o padrão é True.
    :return: Uma instância de QRNN.

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

        print("Forma da saída:", output.shape)
        print("Forma de h_t:", h_t.shape)


TTOLayer
----------------------------------------------------------------

.. py:class:: pyvqnet.qnn.ttolayer.TTOLayer(inp_modes,out_modes,mat_ranks,biases_initializer=tensor.zeros)

    TTOLayer, baseado em "Compressing deep neural networks by matrix product operators" (https://arxiv.org/abs/1904.06194), decompõe o tensor de entrada para alcançar uma representação eficiente de dados de alta dimensão. Esta camada permite aprender a decomposição tensorial sob restrições de posto, o que pode reduzir a complexidade computacional e o uso de memória em comparação com camadas totalmente conectadas tradicionais.

    :param inp_modes: As dimensões do tensor de entrada.
    :param out_modes: As dimensões do tensor de saída.
    :param mat_ranks: O posto do kernel tensorial (posto de decomposição) na decomposição tensorial.
    :param biases_initializer: A função de inicialização do viés.
    :return: Uma instância de TTOLayer.

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

        print("Forma da entrada:", inp.shape)
        print("Forma da saída:", output.shape)


Outras funções
=====================



QuantumLayerAdjoint
-----------------------------------------
.. py:class:: pyvqnet.qnn.vqc.QuantumLayerAdjoint(general_module: pyvqnet.nn.Module, use_qpanda=False,name="")


    Uma camada QuantumLayer diferenciada automaticamente que usa o método de matriz adjunta para cálculo de gradiente, consulte `Efficient calculation of gradients in classical simulations of variational quantum algorithms <https://arxiv.org/abs/2009.02823>`_.

    :param general_module: Uma instância de `pyvqnet.nn.Module` construída usando apenas a interface de circuito quântico inferior `pyvqnet.qnn.vqc`.
    :param use_qpanda: Se deve usar a linha qpanda para transmissão direta, padrão: False.
    :param name: O nome desta camada, o padrão é "".

    .. note::

        QMachine para general_module deve definir grad_method = "adjoint".
        Atualmente, as seguintes portas lógicas paramétricas são suportadas: `RX`, `RY`, `RZ`, `PhaseShift`, `RXX`, `RYY`, `RZZ`, `RZX`, `U1`, `U2`, `U3` e outros circuitos variacionais sem portas lógicas paramétricas.

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

Cria vqc com paralelismo de dados para tamanho de lote de dados usando cálculo de gradiente adjunto. Onde ``vqc_module`` deve ser um módulo VQC do tipo ``QuantumLayerAdjoint``.

Se usarmos N nós para executar este módulo,
Em cada nó, os dados `batch_size/N` são encaminhados para executar o circuito quântico variacional para calcular gradientes.

:param Comm_OP: Define o controlador de comunicação para o ambiente distribuído.
:param vqc_module: Um módulo VQC do tipo QuantumLayerAdjoint com forward(), certifique-se de que qmachine esteja configurado corretamente.
:param name: O nome do módulo. O valor padrão é uma string vazia.
:return: Um módulo que pode calcular circuitos quânticos.

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

    Cria vqc com paralelismo de dados para tamanho de lote de dados usando cálculo de diferenciação automática.
    Se usarmos N nós para executar este módulo,
    Em cada nó, os dados `batch_size/N` são processados em diante através do circuito quântico variacional para calcular gradientes.

    :param Comm_OP: Define o controlador de comunicação para o ambiente distribuído.
    :param vqc_module: Módulo VQC com forward(), certifique-se de que qmachine esteja configurado corretamente.
    :param name: Nome do módulo. O padrão é uma string vazia.
    :return: Módulo que pode calcular circuitos quânticos.

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

    Converte o módulo vqc do VQNet para originIR.

    vqc_model deve executar a função forward antes desta função para obter os dados de entrada.
    Se os dados de entrada forem dados em lote. Para cada entrada, retornará múltiplas strings IR.

    :param vqc_model: Módulo vqc do VQNet, que deve executar forward primeiro.

    :return: string originIR ou lista de strings originIR.

originir_to_vqc
------------------------------------------

.. py:function:: pyvqnet.qnn.vqc.originir_to_vqc(originir, tmp="code_tmp.py", verbose=False)

    Analisa originIR em código de modelo vqc.
    O código cria um circuito quântico variacional `pyvqnet.nn.Module` sem `Measure`, e retorna a forma de vetor de estado do estado quântico, como [b,2,...,2].
    Esta função gerará um arquivo de código definindo o modelo VQNet correspondente em "./origin_ir_gen_code/" + tmp + ".py".

    :param originir: IR original.
    :param tmp: nome do arquivo de código, padrão ``code_tmp.py``.
    :param verbose: Se deve exibir o código gerado, padrão = False
    :return:
        Gera código executável.

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

    Imprime informações sobre a camada clássica e operadores de porta quântica registrados em vqc_module.

    :param vqc_module: módulo vqc
    :return:
        string de resumo


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

    Modelos de aprendizado de máquina quântica geralmente usam o método de descida de gradiente para otimizar parâmetros em circuitos lógicos quânticos variacionais. A fórmula do método clássico de descida de gradiente é a seguinte:

    .. math:: \theta_{t+1} = \theta_t -\eta \nabla \mathcal{L}(\theta),

    Essencialmente, a cada iteração, calcularemos a direção da descida mais íngreme do gradiente no espaço de parâmetros como a direção da mudança de parâmetro.
    Em qualquer direção no espaço, a velocidade de descida no intervalo local não é tão rápida quanto a da direção do gradiente negativo.
    Em espaços diferentes, a derivação da direção de descida mais íngreme depende da norma da diferenciação do parâmetro - a métrica de distância. A métrica de distância desempenha um papel central aqui,
    Métricas diferentes resultam em direções de descida mais íngreme diferentes. Para o espaço euclidiano onde os parâmetros no problema de otimização clássico estão localizados, a direção da descida mais íngreme é a direção do gradiente negativo.
    Mesmo assim, a cada passo da otimização de parâmetros, à medida que a função de perda muda com os parâmetros, seu espaço de parâmetros é transformado. Isso possibilita encontrar outra norma de distância melhor.

    `Quantum natural gradient method <https://arxiv.org/abs/1909.02108>`_ se baseia em conceitos do `classical natural gradient method Amari <https://www.mitpressjournals.org/doi/abs/10.1162/089976698300017746>`__,
    Em vez disso, vemos o problema de otimização como uma distribuição de probabilidade de possíveis valores de saída para uma dada entrada (ou seja, estimativa de máxima verossimilhança), uma abordagem melhor é realizar a descida de gradiente no espaço de distribuição, que é adimensional e invariante em relação à parametrização. Portanto, independentemente da parametrização, cada passo de otimização sempre escolherá o tamanho do passo ideal para cada parâmetro.
    Em tarefas de aprendizado de máquina quântico, o espaço de estado quântico tem um tensor métrico invariante único chamado tensor métrico de Fubini-Study :math:`g_{ij}`.
    Este tensor converte a descida mais íngreme no espaço de parâmetros do circuito quântico para a descida mais íngreme no espaço de distribuição.
    A fórmula para o gradiente natural quântico é a seguinte:

    .. math:: \theta_{t+1} = \theta_t + momentum(x^{(t)} - x^{(t-1)}) - \eta g^{+}(\theta_t)\nabla \mathcal{L}(\theta)

    onde :math:`g^{+}` é a pseudo-inversa.

    `wrapper_calculate_qng` é um decorador que precisa ser adicionado à função forward do modelo a ser calculado para o gradiente natural quântico. Apenas parâmetros do tipo `Parameter` registrados no modelo são otimizados.

    :param qmodel: Modelo de circuito quântico variacional, você precisa usar `wrapper_calculate_qng` como o decorador da função forward.
    :param stepsize: O tamanho do passo do método de descida de gradiente, o padrão é 0.01.
    :param momentum: Momentum, padrão é 0.

    .. note::

        Testado apenas em dados não agrupados em lote.
        Apenas circuitos quânticos puramente variacionais são suportados.
        step() atualizará os gradientes da entrada e dos parâmetros.
        step() atualiza apenas os valores numéricos dos parâmetros do modelo.

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

    Um decorador para fundir operações de bit único em operações Rot.

    .. note::

        f é a função forward do módulo, e a função forward do modelo precisa ser executada uma vez para ter efeito.
        O modelo definido aqui herda de `pyvqnet.qnn.vqc.QModule`, que é uma subclasse de `pyvqnet.nn.Module`.


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

    Decoradores para troca de porta controlada
    Esta é uma transformação quântica usada para mover portas comutáveis para frente dos bits de controle e alvo da operação controlada.
    As portas diagonais em ambos os lados do bit de controle não afetam o resultado da porta controlada; portanto, podemos empurrar todas as portas de bit único atuando no primeiro bit juntas para a direita (e fundi-las se necessário).
    Da mesma forma, portas X são intercambiáveis com os bits alvo de CNOT e Toffoli (assim como PauliY e CRY).
    Podemos usar esta transformação para empurrar portas de bit único o mais fundo possível na operação controlada.

    .. note::

        f é a função forward do módulo, e a função forward do modelo precisa ser executada uma vez para ter efeito.
        O modelo definido aqui herda de `pyvqnet.qnn.vqc.QModule`, que é uma subclasse de `pyvqnet.nn.Module`.

    :param f: função forward.
    :param direction: A direção para mover a porta de bit único, o valor opcional é "left" ou "right", o padrão é "right".

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

wrapper_merge_rotations
---------------------------------------------------------------

.. py:function:: pyvqnet.qnn.vqc.wrapper_merge_rotations(f)

    Decorador para mesclar rotações do mesmo tipo, incluindo "rx", "ry", "rz", "phaseshift", "crx", "cry", "crz", "controlledphaseshift", "isingxx",
        "isingyy", "isingzz", "rot".

    .. note::

        f é a função forward do módulo, e a função forward do modelo precisa ser executada uma vez para ter efeito.
        O modelo definido aqui herda de `pyvqnet.qnn.vqc.QModule`, que é uma subclasse de `pyvqnet.nn.Module`.

    :param f: função forward.


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

    Usa regras de compilação para otimizar circuitos do QModule.

    .. note::

        f é a função forward do módulo, e a função forward do modelo precisa ser executada uma vez para ter efeito.
        O modelo definido aqui herda de `pyvqnet.qnn.vqc.QModule`, que é uma subclasse de `pyvqnet.nn.Module`.

    :param f: função forward.


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

    Quantum Natural SPSA (QNSPSA) Optimizer é um otimizador estocástico de segunda ordem para circuitos quânticos que combina descida de gradiente com informações do tensor métrico de Fubini-Study.
    Estimativa de gradiente usando perturbações simétricas (similar ao SPSA):

    .. math::
        \widehat{\nabla f}(\mathbf{x}) \approx \frac{f(\mathbf{x}+\epsilon \mathbf{h})-f(\mathbf{x}-\epsilon \mathbf{h})}{2\epsilon}

    Calcula a métrica de Fubini-Study a partir da medida de sobreposição de estado:

    .. math::
        \widehat{\mathbf{g}}(\mathbf{x}) \approx \frac{\delta F}{8\epsilon^2}(\mathbf{h}_1\mathbf{h}_2^\intercal + \mathbf{h}_2\mathbf{h}_1^\intercal)
    .. math::
        \delta F = F(\mathbf{x}+\epsilon\mathbf{h}_1+\epsilon\mathbf{h}_2) - F(\mathbf{x}+\epsilon\mathbf{h}_1) - F(\mathbf{x}-\epsilon\mathbf{h}_1+\epsilon\mathbf{h}_2) + F(\mathbf{x}-\epsilon\mathbf{h}_1)

    onde δF mede as diferenças de avaliação de sobreposição dos quatro circuitos.

    Regra de atualização:

    .. math::
        \mathbf{x}^{(t+1)} = \mathbf{x}^{(t)} - \eta \widehat{\mathbf{g}}^{-1}(\mathbf{x}^{(t)})\widehat{\nabla f}(\mathbf{x}^{(t)})
    
    :param stepsize: Hiperparâmetro da taxa de aprendizado definida pelo usuário :math:`\eta` (padrão: 1e-3)
    :param regularization: Termo de regularização :math:`\beta` usado para o tensor métrico de Fubini-Study, para estabilidade numérica (padrão: 1e-3)
    :param finite_diff_step: Tamanho do passo :math:`\epsilon` usado para calcular gradientes de diferença finita e o tensor métrico de Fubini-Study (padrão: 1e-2)
    :param resamplings: Número médio de amostras por atualização de parâmetro (padrão: 1)
    :param blocking: Quando True, apenas aceita atualizações que resultam em uma perda não maior que a soma dos valores de perda anteriores (para ajudar na convergência) (padrão: True)
    :param history_length: Quando ``blocking`` é True, a tolerância é definida como a média dos valores de custo anteriores ``history_length`` (padrão: 5)
    :param seed: semente para amostragem aleatória (padrão: None)

    .. py:method:: step(qmodel, *args, **kwargs)
        Atualiza parâmetros treináveis uma vez usando o otimizador.

        :param qmodel: Modelo quântico treinável
        :param args: QTensor treinável de comprimento variável para qmodel.
        :param kwargs: Argumentos de palavra-chave de comprimento variável para qmodel.

        :return: Parâmetros atualizados.

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

Benchmarks de Treinamento de Dados em Lote de Aprendizado de Máquina Quântico
==================================================================================

Teste 1: Comparação de Gradiente de Dados em Lote (VQNet / DeepQuantum / Pennylane)
------------------------------------------------------------------------------------

Em aprendizado de máquina quântico, o cálculo de gradiente é um fator chave que afeta a eficiência dos circuitos quânticos variacionais. Para avaliar o desempenho do cálculo de gradiente quântico em diferentes frameworks, este artigo realizou testes de benchmark no VQNet, Deepquantum e Pennylane no sistema Linux usando GPU. Os testes foram realizados sob diferentes tamanhos de dados em lote (batch size 16, 32), profundidades de circuito (layer 2, 4) e número de qubits (qubit 4, 8, 12, 16). A estrutura do circuito foi CNOT + RX + RZ + RX encoding layers. O tempo médio de execução de cada framework em 10 execuções foi registrado. Deepquantum e Pennylane são implementados com base no backend GPU do Torch, enquanto o VQNet usa um esquema de aceleração GPU desenvolvido internamente.

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

Teste 2: Comparação de Gradiente VQC de 10 Qubits (com TorchQuantum)
------------------------------------------------------------------------------------

Este teste é baseado no circuito VQC de 10 qubits e 10 camadas usado no artigo do modelo grande Origin Quantum. Ele compara o desempenho do cálculo de gradiente de cinco frameworks: VQNet, TorchQuantum (TQ), DeepQuantum (DQ), Pennylane (PL) e MindQuantum (MQ). A estrutura do circuito é:

  RY(data) -> [RY(param) -> CRZ(param) -> RY(param) -> CRZ(param)] x L

Cada camada contém 40 parâmetros (4 grupos x 10 qubits), totalizando 400 parâmetros. Os tamanhos de lote variam de 1 a 1024. Contagens de tentativas: VQNet / TQ / DQ executam 20 tentativas cada, PL / MQ executam 2 tentativas cada (os dois últimos são mais lentos com dados em lote e limitados a 2 tentativas para economizar tempo).

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
