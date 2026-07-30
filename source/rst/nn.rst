Modulo di Rete Neurale Classica
#########################################

I seguenti moduli di rete neurale classica supportano il calcolo automatico della retropropagazione. Dopo aver eseguito la funzione forward, puoi calcolare il gradiente eseguendo la funzione backward. Un semplice esempio del livello di convoluzione è il seguente:

.. code-block::

    from pyvqnet.tensor import arange
    from pyvqnet import kfloat32
    from pyvqnet.nn import Conv2D

    # un'immagine viene inserita in un livello di convoluzione bidimensionale
    b = 2        # dimensione del batch
    ic = 2       # canali di input
    oc = 2      # canali di output
    hw = 4      # larghezza e altezza dell'input

    # livello di convoluzione bidimensionale
    test_conv = Conv2D(ic,oc,(2,2),(2,2),"same")

    # input di forma [b,ic,hw,hw]
    x0 = arange(1,b*ic*hw*hw+1,requires_grad=True,dtype=kfloat32)

    x1 = x0.reshape([b,ic,hw,hw])
    # funzione forward
    x = test_conv(x1)

    # funzione backward con autograd
    x.backward()
    print(x0.grad)

    # [
    # [[[0.0958736, 0.3032238, 0.0958736, 0.3032238],
    #  [-0.2665333, 0.1081382, -0.2665333, 0.1081382],
    #  [0.0958736, 0.3032238, 0.0958736, 0.3032238],
    #  [-0.2665333, 0.1081382, -0.2665333, 0.1081382]],
    # [[-0.0068994, 0.0914679, -0.0068994, 0.0914679],
    #  [-0.2820665, 0.3160213, -0.2820665, 0.3160213],
    #  [-0.0068994, 0.0914679, -0.0068994, 0.0914679],
    #  [-0.2820665, 0.3160213, -0.2820665, 0.3160213]]],
    # [[[0.0958736, 0.3032238, 0.0958736, 0.3032238],
    #  [-0.2665333, 0.1081382, -0.2665333, 0.1081382],
    #  [0.0958736, 0.3032238, 0.0958736, 0.3032238],
    #  [-0.2665333, 0.1081382, -0.2665333, 0.1081382]],
    # [[-0.0068994, 0.0914679, -0.0068994, 0.0914679],
    #  [-0.2820665, 0.3160213, -0.2820665, 0.3160213],
    #  [-0.0068994, 0.0914679, -0.0068994, 0.0914679],
    #  [-0.2820665, 0.3160213, -0.2820665, 0.3160213]]]
    # ]

.. currentmodule:: pyvqnet.nn


Classe Module
********************************************************

modulo di calcolo astratto


Module
=================================

.. py:class:: pyvqnet.nn.module.Module

    Classe base per tutti i moduli di rete neurale, inclusi i moduli quantistici e quelli classici.
    I tuoi modelli dovrebbero essere anch'essi sottoclassi di questa classe per il calcolo dell'autograd.

    I moduli possono anche contenere altri moduli, permettendo di annidarli
    in una struttura ad albero. Puoi assegnare i sottomoduli come attributi regolari::

        class Model(Module):
            def __init__(self):
                super(Model, self).__init__()
                self.conv1 = pyvqnet.nn.Conv2d(1, 20, (5,5))
                self.conv2 = pyvqnet.nn.Conv2d(20, 20, (5,5))

            def forward(self, x):
                x = pyvqnet.nn.activation.relu(self.conv1(x))
                return pyvqnet.nn.activation.relu(self.conv2(x))

    I sottomoduli assegnati in questo modo verranno registrati

forward
=================================

.. py:method:: pyvqnet.nn.module.Module.forward(x, *args, **kwargs)

    Metodo astratto che esegue il passaggio forward.

    :param x: QTensor di input
    :param \*args: Un parametro variabile non nominativo
    :param \*\*kwargs: Un parametro variabile nominativo
    :return: output del modulo

    Esempio::

        import numpy as np
        from pyvqnet.tensor import QTensor
        import pyvqnet as vq
        from pyvqnet.nn import Conv2D
        b = 2
        ic = 3
        oc = 2
        test_conv = Conv2D(ic, oc, (3, 3), (2, 2), "same")
        x0 = QTensor(np.arange(1, b * ic * 5 * 5 + 1).reshape([b, ic, 5, 5]),
                    requires_grad=True,
                    dtype=vq.kfloat32)
        x = test_conv.forward(x0)
        print(x)

        # [
        # [[[4.3995643, 3.9317808, -2.0707254],
        #  [20.1951981, 21.6946659, 14.2591858],
        #  [38.4702759, 31.9730244, 24.5977650]],
        # [[-17.0607567, -31.5377998, -7.5618000],
        #  [-22.5664024, -40.3876266, -15.1564388],
        #  [-3.1080279, -18.5986233, -8.0648050]]],
        # [[[6.6493244, -13.4840755, -20.2554188],
        #  [54.4235802, 34.4462433, 26.8171902],
        #  [90.2827682, 62.9092331, 51.6892929]],
        # [[-22.3385429, -45.2448578, 5.7101378],
        #  [-32.9464149, -60.9557228, -10.4994345],
        #  [5.9029331, -20.5480480, -0.9379558]]]
        # ]

state_dict 
=================================

.. py:method:: pyvqnet.nn.module.Module.state_dict(destination=None, prefix='')

    Restituisce un dizionario contenente l'intero stato del modulo.

    Sono inclusi sia i parametri che i buffer persistenti (ad esempio le medie mobili).
    Le chiavi sono i nomi corrispondenti dei parametri e dei buffer.

    :param destination: un dizionario in cui verrà memorizzato lo stato
    :param prefix: il prefisso per i parametri e i buffer utilizzati in questo
        modulo

    :return: un dizionario contenente l'intero stato del modulo

    Esempio::

        from pyvqnet.nn import Conv2D
        test_conv = Conv2D(2,3,(3,3),(2,2),"same")
        print(test_conv.state_dict().keys())
        #odict_keys(['weights', 'bias'])


toGPU
=================================

.. py:function:: pyvqnet.nn.module.Module.toGPU(device: int = DEV_GPU_0)

    Sposta i parametri e i dati dei buffer di un modulo e dei suoi sottomoduli sul dispositivo GPU specificato.

    device specifica il dispositivo su cui memorizzare i dati interni. Quando device >= DEV_GPU_0, i dati vengono memorizzati sulla GPU. Se il tuo computer ha più GPU,
    puoi specificare dispositivi diversi per memorizzare i dati. Ad esempio, device = DEV_GPU_1, DEV_GPU_2, DEV_GPU_3, ... indica la memorizzazione su GPU con diversi numeri di serie.
    
    .. note::
        Il modulo non può essere calcolato su GPU diverse. Verrà sollevato un errore Cuda se tenti di creare un QTensor su una GPU il cui ID supera il numero massimo di GPU verificate.

    :param device: Il dispositivo su cui salvare correntemente il QTensor, default=DEV_GPU_0. device = pyvqnet.DEV_GPU_0, memorizzato nella prima GPU, device = DEV_GPU_1, memorizzato nella seconda GPU, e così via.
    :return: Modulo spostato sul dispositivo GPU.

    Esempi::

        from pyvqnet.nn.conv import ConvT2D 
        test_conv = ConvT2D(3, 2, [4,4], [2, 2], "same")
        test_conv = test_conv.toGPU()
        print(test_conv.backend)
        #1000


toCPU
=================================

.. py:function:: pyvqnet.nn.module.Module.toCPU()

    Sposta i parametri e i dati dei buffer di un modulo e dei suoi sottomoduli su un dispositivo CPU specifico.

    :return: Modulo spostato sul dispositivo CPU.

    Esempi::

        from pyvqnet.nn.conv import ConvT2D 
        test_conv = ConvT2D(3, 2, [4,4], [2, 2], "same")
        test_conv = test_conv.toCPU()
        print(test_conv.backend)
        #0

.. _save_parameters:

save_parameters
=================================

.. py:function:: pyvqnet.utils.storage.save_parameters(obj, f)

    Salva i parametri del modello in un file su disco.

    :param obj: OrderedDict salvato da ``state_dict()``
    :param f: una stringa o un oggetto os.PathLike contenente un nome di file
    :return: None

    Esempio::

        from pyvqnet.nn import Module,Conv2D
        import pyvqnet
        class Net(Module):
            def __init__(self):
                super(Net, self).__init__()
                self.conv1 = Conv2D(input_channels=1, output_channels=6, kernel_size=(5, 5), stride=(1, 1), padding="valid")

            def forward(self, x):
                return super().forward(x)

        model = Net()
        pyvqnet.utils.storage.save_parameters(model.state_dict(),"tmp.model")

load_parameters
=================================

.. py:function:: pyvqnet.utils.storage.load_parameters(f)

    Carica i parametri del modello da un file su disco.

    L'istanza del modello deve essere creata prima.

    :param f: una stringa o un oggetto os.PathLike contenente un nome di file
    :return: OrderedDict salvato per ``load_state_dict()``

    Esempio::

        from pyvqnet.nn import Module,Conv2D
        import pyvqnet

        class Net(Module):
            def __init__(self):
                super(Net, self).__init__()
                self.conv1 = Conv2D(input_channels=1, output_channels=6, kernel_size=(5, 5), stride=(1, 1), padding="valid")

            def forward(self, x):
                return super().forward(x)

        model = Net()
        model1 = Net()  # un altro oggetto Module
        pyvqnet.utils.storage.save_parameters( model.state_dict(),"tmp.model")
        model_para =  pyvqnet.utils.storage.load_parameters("tmp.model")
        model1.load_state_dict(model_para)



ModuleList
**************************************************************************************************************************************************************************

.. py:class:: pyvqnet.nn.module.ModuleList([pyvqnet.nn.module.Module])


    Salva i sottomoduli in una lista. ModuleList può essere indicizzata come una normale lista Python e i parametri interni del Module che contiene possono essere salvati.

    :param modules: lista di nn.Module

    :return: una lista di moduli

    Esempio::

        from pyvqnet.tensor import *
        from pyvqnet.nn import Module,Linear,ModuleList
        from pyvqnet.qnn import ProbsMeasure,QuantumLayer
        import pyqpanda as pq
        def pqctest (input,param,qubits,cbits,m_machine):
            circuit = pq.QCircuit()
            circuit.insert(pq.H(qubits[0]))
            circuit.insert(pq.H(qubits[1]))
            circuit.insert(pq.H(qubits[2]))
            circuit.insert(pq.H(qubits[3]))

            circuit.insert(pq.RZ(qubits[0],input[0]))
            circuit.insert(pq.RZ(qubits[1],input[1]))
            circuit.insert(pq.RZ(qubits[2],input[2]))
            circuit.insert(pq.RZ(qubits[3],input[3]))

            circuit.insert(pq.CNOT(qubits[0],qubits[1]))
            circuit.insert(pq.RZ(qubits[1],param[0]))
            circuit.insert(pq.CNOT(qubits[0],qubits[1]))

            circuit.insert(pq.CNOT(qubits[1],qubits[2]))
            circuit.insert(pq.RZ(qubits[2],param[1]))
            circuit.insert(pq.CNOT(qubits[1],qubits[2]))

            circuit.insert(pq.CNOT(qubits[2],qubits[3]))
            circuit.insert(pq.RZ(qubits[3],param[2]))
            circuit.insert(pq.CNOT(qubits[2],qubits[3]))


            prog = pq.QProg()
            prog.insert(circuit)

            rlt_prob = ProbsMeasure([0,2],prog,m_machine,qubits)
            return rlt_prob


        class M(Module):
            def __init__(self):
                super(M, self).__init__()
                self.pqc2 = ModuleList([QuantumLayer(pqctest,3,"cpu",4,1), Linear(4,1)
                ])

            def forward(self, x, *args, **kwargs):
                y = self.pqc2[0](x)  + self.pqc2[1](x)
                return y

        mm = M()
        print(mm.state_dict().keys())
        #odict_keys(['pqc2.0.m_para', 'pqc2.1.weights', 'pqc2.1.bias'])



ParameterList
*********************************************************
.. py:class:: pyvqnet.nn.module.ParameterList([pyvqnet.nn.module.Module])


    Memorizza i parametri in una lista. Una ParameterList può essere indicizzata come una normale lista Python e i parametri interni del Parameter che contiene possono essere memorizzati.

    :param modules: lista di nn.Parameter.

    :return: una lista di Parameter.

    Esempio::

        from pyvqnet import nn
        class MyModule(nn.Module):
            def __init__(self):
                super().__init__()
                self.params = nn.ParameterList([nn.Parameter((10, 10)) for i in range(10)])
            def forward(self, x):

                # ParameterList può essere usato come un iterabile o essere indicizzato con interi
                for i, p in enumerate(self.params):
                    x = self.params[i // 2] * x + p * x
                return x

        model = MyModule()
        print(model.state_dict().keys())


Sequential
*********************************************************
.. py:class:: pyvqnet.nn.module.Sequential([pyvqnet.nn.module.Module])

    I moduli vengono aggiunti nell'ordine in cui vengono passati. In alternativa, puoi passare un ``OrderedDict`` di moduli. Il metodo ``forward()`` di ``Sequential`` accetta qualsiasi input e lo inoltra al suo primo modulo.
    Successivamente, ``Sequential`` passa l'output all'input di ciascun modulo successivo e infine restituisce l'output dell'ultimo modulo.

    :param modules: moduli da aggiungere.

    :return: Sequential.

    Esempio::
        
        from pyvqnet import nn
        from collections import OrderedDict

        # Utilizzo di Sequential per creare un modello piccolo.
        model = nn.Sequential(
                  nn.Conv2D(1,20,(5, 5)),
                  nn.ReLu(),
                  nn.Conv2D(20,64,(5, 5)),
                  nn.ReLu()
                )
        print(model.state_dict().keys())

        # Utilizzo di Sequential con OrderedDict. Questo è funzionalmente equivalente al codice precedente
                
        model = nn.Sequential(OrderedDict([
                  ('conv1', nn.Conv2D(1,20,(5, 5))),
                  ('relu1', nn.ReLu()),
                  ('conv2', nn.Conv2D(20,64,(5, 5))),
                  ('relu2', nn.ReLu())
                ]))
        print(model.state_dict().keys())


Livello di Rete Neurale Classica
********************************************************

Conv1D
=================================

.. py:class:: pyvqnet.nn.Conv1D(input_channels:int,output_channels:int,kernel_size:int ,stride:int= 1,padding = "valid",use_bias:str = True,kernel_initializer = None,bias_initializer =None, dilation_rate: int = 1, group: int = 1, dtype=None, name='')

    Applica un kernel di convoluzione 1D su un input. Gli input del modulo di convoluzione hanno forma (batch_size, input_channels, height)

    :param input_channels: `int` - Numero di canali di input
    :param output_channels: `int` - Numero di kernel
    :param kernel_size: `int` - Dimensione di un singolo kernel. forma del kernel = [output_channels,input_channels/group,kernel_size,1]
    :param stride: `int` - Passo, default 1
    :param padding: `str|int` - opzione di padding, può essere una stringa {'valid', 'same'} o un intero che indica la quantità di padding implicito da applicare. Default "valid".
    :param use_bias: `bool` - se usare il bias, default True
    :param kernel_initializer: `callable` - Default None
    :param bias_initializer: `callable` - Default None
    :param dilation_rate: `int` - tasso di dilatazione, default: 1
    :param group: `int` -  numero di gruppi di convoluzioni raggruppate. Default: 1
    :param dtype: Il tipo di dato del parametro, default: None, utilizza il tipo di dato predefinito kfloat32, che rappresenta un numero a virgola mobile a 32 bit.
    :param name: Il nome del modulo, default: "".
    :return: una classe Conv1D

    .. note::
        ``padding='valid'`` è equivalente a nessun padding.

        ``padding='same'`` applica un padding all'input in modo che l'output abbia la stessa forma dell'input.

    Esempio::

        import numpy as np
        from pyvqnet.tensor import QTensor
        from pyvqnet.nn import Conv1D
        import pyvqnet
        b= 2
        ic =3
        oc = 2
        test_conv = Conv1D(ic,oc,3,2,"same")
        x0 = QTensor(np.arange(1,b*ic*5*5 +1).reshape([b,ic,25]),requires_grad=True,dtype=pyvqnet.kfloat32)
        x = test_conv.forward(x0)
        print(x)

        # [
        # [[12.4438553, 14.8618164, 15.5595102, 16.2572021, 16.9548950, 17.6525879, 18.3502808, 19.0479736, 19.7456665, 20.4433594, 21.1410522, 21.8387432, 10.5725441],
        #  [-13.7539215, 1.0263026, 1.2747254, 1.5231485, 1.7715728, 2.0199962, 2.2684195, 2.5168428, 2.7652662, 3.0136888, 3.2621140, 3.5105357, 14.0515862]],
        # [[47.4924164, 41.0252953, 41.7229881, 42.4206772, 43.1183739, 43.8160667, 44.5137596, 45.2114487, 45.9091415, 46.6068344, 47.3045311, 48.0022240, 18.3216572],
        #  [-47.2381554, 10.3421783, 10.5906038, 10.8390274, 11.0874519, 11.3358765, 11.5842953, 11.8327246, 12.0811434, 12.3295631, 12.5779924, 12.8264122, 39.4719162]]
        # ]

Conv2D
=================================

.. py:class:: pyvqnet.nn.Conv2D(input_channels:int,output_channels:int,kernel_size:tuple,stride:tuple=(1, 1),padding="valid",use_bias = True,kernel_initializer=None,bias_initializer=None, dilation_rate: int = 1, group: int = 1, dtype = None, name = "")

    Applica un kernel di convoluzione bidimensionale su un input. Gli input del modulo di convoluzione hanno forma (batch_size, input_channels, height, width)

    :param input_channels: `int` - Numero di canali di input
    :param output_channels: `int` - Numero di kernel
    :param kernel_size: `tuple|list` - Dimensione di un singolo kernel. forma del kernel = [output_channels,input_channels/group,kernel_size,kernel_size]
    :param stride: `tuple|list` - Passo, default (1, 1)|[1,1]
    :param padding: `str|tuple` - opzione di padding, può essere una stringa {'valid', 'same'} o una tupla di interi che indica la quantità di padding implicito da applicare su entrambi i lati. Default "valid".
    :param use_bias: `bool` - se usare il bias, default True
    :param kernel_initializer: `callable` - Default None
    :param bias_initializer: `callable` - Default None
    :param dilation_rate: `int` - tasso di dilatazione, default: 1
    :param group: `int` -  numero di gruppi di convoluzioni raggruppate. Default: 1.
    :param dtype: Il tipo di dato del parametro, default: None, utilizza il tipo di dato predefinito kfloat32, che rappresenta un numero a virgola mobile a 32 bit.
    :param name: Il nome del modulo, default: "".

    :return: una classe Conv2D

    .. note::
        ``padding='valid'`` è equivalente a nessun padding.

        ``padding='same'`` applica un padding all'input in modo che l'output abbia la stessa forma dell'input.

    Esempio::

        import numpy as np
        from pyvqnet.tensor import QTensor
        from pyvqnet.nn import Conv2D
        import pyvqnet
        b= 2
        ic =3
        oc = 2
        test_conv = Conv2D(ic,oc,(3,3),(2,2),"same")
        x0 = QTensor(np.arange(1,b*ic*5*5+1).reshape([b,ic,5,5]),requires_grad=True,dtype=pyvqnet.kfloat32)
        x = test_conv.forward(x0)
        print(x)

        # [
        # [[[-0.1256833, 23.8978596, 26.7449780],
        #  [-7.2959919, 33.4023743, 42.1283913],
        #  [-8.7684336, 25.2698975, 40.4024887]],
        # [[33.0653763, 40.3120155, 27.3781891],
        #  [39.2921371, 45.8685760, 38.1885109],
        #  [23.1873779, 12.0480318, 12.7278290]]],
        # [[[-0.9730744, 61.3967094, 79.0511856],
        #  [-29.3652401, 75.0349350, 112.7325439],
        #  [-26.4682808, 59.0924797, 104.2572098]],
        # [[66.8064194, 96.0953140, 72.9157486],
        #  [90.9154129, 110.7232437, 91.2616043],
        #  [56.8825951, 34.6904907, 30.1957760]]]
        # ]

ConvT2D
=================================

.. py:class:: pyvqnet.nn.ConvT2D(input_channels,output_channels,kernel_size,stride=[1, 1],padding="valid",use_bias="True", kernel_initializer=None,bias_initializer=None, dilation_rate: int = 1, out_padding=(0,0), group: int = 1, dtype=None, name='')

    Applica un kernel di convoluzione trasposta bidimensionale su un input. Gli input del modulo ConvT hanno forma (batch_size, input_channels, height, width)

    :param input_channels: `int` - Numero di canali di input
    :param output_channels: `int` - Numero di kernel
    :param kernel_size: `tuple|list` - Dimensione di un singolo kernel. forma del kernel = [input_channels,output_channels/group,kernel_size,kernel_size]
    :param stride: `tuple|list` - Passo, default (1, 1)|[1,1]
    :param padding: `str|tuple` - opzione di padding, può essere una stringa {'valid', 'same'} o una tupla di interi che indica la quantità di padding implicito da applicare su entrambi i lati. Default "valid".
    :param use_bias: `bool` - Se utilizzare un elemento di offset. Default: usa.
    :param kernel_initializer: `callable` - Default None
    :param bias_initializer: `callable` - Default None
    :param dilation_rate: `int` - tasso di dilatazione, default: 1.
    :param out_padding: Dimensione aggiuntiva aggiunta a un lato di ciascuna dimensione nella forma dell'output. Default: (0,0)
    :param group: `int` -  numero di gruppi di convoluzioni raggruppate. Default: 1.
    :param dtype: Il tipo di dato del parametro, default: None, utilizza il tipo di dato predefinito kfloat32, che rappresenta un numero a virgola mobile a 32 bit.
    :param name: Il nome del modulo, default: "".

    :return: una classe ConvT2D

    .. note::
        ``padding='valid'`` è equivalente a nessun padding.

        ``padding='same'`` applica un padding all'input in modo che l'output abbia la stessa forma dell'input.


    Esempio::

        import numpy as np
        from pyvqnet.tensor import QTensor
        from pyvqnet.nn import ConvT2D
        import pyvqnet
        test_conv = ConvT2D(3, 2, (3, 3), (1, 1), "valid")
        x = QTensor(np.arange(1, 1 * 3 * 5 * 5+1).reshape([1, 3, 5, 5]), requires_grad=True,dtype=pyvqnet.kfloat32)
        y = test_conv.forward(x)
        print(y)

        # [
        # [[[-3.3675897, 4.8476148, 14.2448473, 14.8897810, 15.5347166, 20.0420666, 10.9831696],
        #  [-14.0110836, -3.2500827, 6.4022207, 6.5149083, 6.6275964, 23.7946320, 12.1828709],
        #  [-22.2661152, -3.5112300, 12.9493723, 13.5486069, 14.1478367, 39.6327629, 18.8349991],
        #  [-24.4063797, -3.0093837, 15.9455290, 16.5447617, 17.1439915, 44.7691879, 21.3293095],
        #  [-26.5466480, -2.5075383, 18.9416828, 19.5409145, 20.1401463, 49.9056053, 23.8236179],
        #  [-24.7624626, -13.7395811, -7.9510674, -7.9967723, -8.0424776, 19.2783546, 7.0562835],
        #  [-3.5170188, 10.2280807, 16.1939259, 16.6804695, 17.1670132, 21.2262039, 6.2889833]],
        # [[-2.0570512, -9.5056667, -25.0429192, -25.9464111, -26.8499031, -24.7305946, -16.9881954],
        #  [-0.7620960, -18.3383904, -49.8948288, -51.2528229, -52.6108208, -52.2179604, -34.3664169],
        #  [-11.7121849, -27.1864738, -62.2154846, -63.6433640, -65.0712280, -52.6787071, -38.4497032],
        #  [-13.3643141, -29.0211792, -69.3548126, -70.7826691, -72.2105408, -58.1659012, -43.7543182],
        #  [-15.0164423, -30.8558884, -76.4941254, -77.9219971, -79.3498535, -63.6530838, -49.0589256],
        #  [-11.6070204, -14.1940546, -35.5471687, -36.0715408, -36.5959129, -23.9147663, -22.8668022],
        #  [-14.4390459, -4.9011412, -6.4719801, -6.5418491, -6.6117167, 9.3329525, -1.7254852]]]
        # ]

AvgPool1D
=================================

.. py:class:: pyvqnet.nn.AvgPool1D(kernel, stride, padding='valid', name='')

    Questa operazione applica un pooling medio 1D su un segnale di input composto da diversi piani di input.

    :param kernel: dimensione delle finestre di pooling medio
    :param strides: fattore di ridimensionamento
    :param padding: uno tra "valid", "same" o un intero che specifica il valore di padding, default "valid"
    :param name: nome del livello di output.

    :return: livello AvgPool1D

    .. note::
        ``padding='valid'`` è equivalente a nessun padding.

        ``padding='same'`` applica un padding all'input in modo che l'output abbia la stessa forma dell'input.



    Esempio::

        import numpy as np
        from pyvqnet.tensor import QTensor
        from pyvqnet.nn import AvgPool1D
        test_mp = AvgPool1D([3],[2],"same")
        x= QTensor(np.array([0, 1, 0, 4, 5,
                                    2, 3, 2, 1, 3,
                                    4, 4, 0, 4, 3,
                                    2, 5, 2, 6, 4,
                                    1, 0, 0, 5, 7],dtype=float).reshape([1,5,5]),requires_grad=True)

        y= test_mp.forward(x)
        print(y)
        # [
        # [[0.3333333, 1.6666666, 3],
        #  [1.6666666, 2, 1.3333334],
        #  [2.6666667, 2.6666667, 2.3333333],
        #  [2.3333333, 4.3333335, 3.3333333],
        #  [0.3333333, 1.6666666, 4]]
        # ]

MaxPool1D
=================================

.. py:class:: pyvqnet.nn.MaxPool1D(kernel, stride, padding='valid', dtype=None, name='')

    Questa operazione applica un pooling massimo 1D su un segnale di input composto da diversi piani di input.

    :param kernel: dimensione delle finestre di pooling massimo
    :param strides: fattore di ridimensionamento
    :param padding: uno tra "valid", "same" o un intero che specifica il valore di padding, default "valid"
    :param name: Il nome del modulo, default: "".

    :return: livello MaxPool1D

    .. note::

        ``padding='valid'`` è equivalente a nessun padding.

        ``padding='same'`` applica un padding all'input in modo che l'output abbia la stessa forma dell'input.


    Esempio::

        import numpy as np
        from pyvqnet.tensor import QTensor
        from pyvqnet.nn import MaxPool1D
        test_mp = MaxPool1D([3],[2],"same")
        x= QTensor(np.array([0, 1, 0, 4, 5,
                                    2, 3, 2, 1, 3,
                                    4, 4, 0, 4, 3,
                                    2, 5, 2, 6, 4,
                                    1, 0, 0, 5, 7],dtype=float).reshape([1,5,5]),requires_grad=True)

        y= test_mp.forward(x)
        print(y)
        #[[[1. 4. 5.]
        #   [3. 3. 3.]
        #   [4. 4. 4.]
        #   [5. 6. 6.]
        #   [1. 5. 7.]]]

AvgPool2D
=================================

.. py:class:: pyvqnet.nn.AvgPool2D(kernel, stride, padding='valid', name='')

    Questa operazione applica un pooling medio 2D sulle caratteristiche di input.

    :param kernel: dimensione delle finestre di pooling medio
    :param strides: fattori di ridimensionamento
    :param padding: uno tra "valid", "same" o una tupla con interi che specifica il valore di padding di colonna e riga, default "valid"
    :param name: nome del livello di output
    :return: livello AvgPool2D

    .. note::
        ``padding='valid'`` è equivalente a nessun padding.

        ``padding='same'`` applica un padding all'input in modo che l'output abbia la stessa forma dell'input.


    Esempio::

        import numpy as np
        from pyvqnet.tensor import QTensor
        from pyvqnet.nn import AvgPool2D
        test_mp = AvgPool2D([2,2],[2,2],"valid")
        x= QTensor(np.array([0, 1, 0, 4, 5,
                                    2, 3, 2, 1, 3,
                                    4, 4, 0, 4, 3,
                                    2, 5, 2, 6, 4,
                                    1, 0, 0, 5, 7],dtype=float).reshape([1,1,5,5]),requires_grad=True)

        y= test_mp.forward(x)
        print(y)
        #[[[[1.5  1.75]
        #    [3.75 3.  ]]]]

MaxPool2D
=================================

.. py:class:: pyvqnet.nn.MaxPool2D(kernel, stride, padding='valid', name='')

    Questa operazione applica un pooling massimo 2D sulle caratteristiche di input.

    :param kernel: dimensione delle finestre di pooling massimo
    :param strides: fattore di ridimensionamento
    :param padding: uno tra "valid", "same" o una tupla con interi che specifica il valore di padding di colonna e riga, default "valid"
    :param name: nome del livello di output
    :return: livello MaxPool2D

    .. note::
        ``padding='valid'`` è equivalente a nessun padding.

        ``padding='same'`` applica un padding all'input in modo che l'output abbia la stessa forma dell'input.


    Esempio::

        import numpy as np
        from pyvqnet.tensor import QTensor
        from pyvqnet.nn import MaxPool2D
        test_mp = MaxPool2D([2,2],[2,2],"valid")
        x= QTensor(np.array([0, 1, 0, 4, 5,
                                    2, 3, 2, 1, 3,
                                    4, 4, 0, 4, 3,
                                    2, 5, 2, 6, 4,
                                    1, 0, 0, 5, 7],dtype=float).reshape([1,1,5,5]),requires_grad=True)

        y= test_mp.forward(x)
        print(y)
        # [[[[3. 4.]
        #    [5. 6.]]]]

Embedding
=================================

.. py:class:: pyvqnet.nn.embedding.Embedding(num_embeddings, embedding_dim, weight_initializer=<function xavier_normal>,dtype=None, name: str = '')

    Questo modulo viene spesso utilizzato per memorizzare embedding di parole e recuperarli usando indici.
    L'input del modulo è una lista di indici, e l'output sono i corrispondenti
    embedding di parole.

    :param num_embeddings: `int` - dimensione del dizionario degli embedding.
    :param embedding_dim: `int` - la dimensione di ogni vettore di embedding.
    :param weight_initializer: `callable` - default normal.
    :param dtype: Il tipo di dato del parametro, default: None, utilizza il tipo di dato predefinito kfloat32, che rappresenta un numero a virgola mobile a 32 bit.
    :param name: nome del livello di output.

    :return: una classe Embedding

    Esempio::

        import numpy as np
        from pyvqnet.tensor import QTensor
        from pyvqnet.nn.embedding import Embedding
        import pyvqnet
        vlayer = Embedding(30,3)
        x = QTensor(np.arange(1,25).reshape([2,3,2,2]),dtype= pyvqnet.kint64)
        y = vlayer(x)
        print(y)

        # [
        # [[[[-0.3168081, 0.0329394, -0.2934906],
        #  [0.1057295, -0.2844988, -0.1687456]],
        # [[-0.2382513, -0.3642318, -0.2257225],
        #  [0.1563180, 0.1567665, 0.3038477]]],
        # [[[-0.4131152, -0.0564500, -0.2804018],
        #  [-0.2955172, -0.0009581, -0.1641144]],
        # [[0.0692555, 0.1094901, 0.4099118],
        #  [0.4348361, 0.0304361, -0.0061203]]],
        # [[[-0.3310401, -0.1836129, 0.1098949],
        #  [-0.1840732, 0.0332474, -0.0261806]],
        # [[-0.1489778, 0.2519453, 0.3299376],
        #  [-0.1942692, -0.1540277, -0.2335350]]]],
        # [[[[-0.2620637, -0.3181309, -0.1857461],
        #  [-0.0878164, -0.4180320, -0.1831555]],
        # [[-0.0738970, -0.1888980, -0.3034399],
        #  [0.1955448, -0.0409723, 0.3023460]]],
        # [[[0.2430045, 0.0880465, 0.4309453],
        #  [-0.1796514, -0.1432367, -0.1253638]],
        # [[-0.5266719, 0.2386262, -0.0329155],
        #  [0.1033449, -0.3442690, -0.0471130]]],
        # [[[-0.5336705, -0.1939755, -0.3000667],
        #  [0.0059001, 0.5567381, 0.1926173]],
        # [[-0.2385869, -0.3910453, 0.2521235],
        #  [-0.0246447, -0.0241158, -0.1402829]]]]
        # ]


BatchNorm2d
=================================

.. py:class:: pyvqnet.nn.BatchNorm2d(channel_num:int, momentum:float=0.1, epsilon:float = 1e-5, affine= True, beta_initializer=zeros, gamma_initializer=ones, dtype=None, name="")

    Applica la normalizzazione batch (Batch Normalization) su un input 4D (B,C,H,W) come descritto nell'articolo
    `Batch Normalization: Accelerating Deep Network Training by Reducing
    Internal Covariate Shift <https://arxiv.org/abs/1502.03167>`__ .

    .. math::

        y = \frac{x - \mathrm{E}[x]}{\sqrt{\mathrm{Var}[x] + \epsilon}} * \gamma + \beta

    dove :math:`\gamma` e :math:`\beta` sono parametri apprendibili. Inoltre, per default, durante l'addestramento questo layer mantiene stime
    progressive della media e varianza calcolate, che vengono poi utilizzate per la normalizzazione durante la valutazione.
    Le stime progressive vengono mantenute con un momentum predefinito di 0.1.

    :param channel_num: `int` - il numero di canali delle caratteristiche di input.
    :param momentum: `float` - momentum per il calcolo della media mobile pesata, default 0.1.
    :param epsilon: `float` - costante di stabilità numerica, default 1e-5.
    :param affine: Un valore booleano che, quando impostato a ``True``, fa sì che questo modulo abbia parametri affini apprendibili per canale, inizializzati a 1 (per i pesi) e 0 (per i bias). Default: ``True``.
    :param beta_initializer: `callable` - default zeros.
    :param gamma_initializer: `callable` - default ones.
    :param dtype: Il tipo di dato del parametro, default: None, utilizza il tipo di dato predefinito kfloat32, che rappresenta un numero a virgola mobile a 32 bit.
    :param name: nome del livello di output
    :return: una classe BatchNorm2d

    Esempio::

        import numpy as np
        from pyvqnet.tensor import QTensor
        from pyvqnet.nn import BatchNorm2d
        import pyvqnet
        b = 2
        ic = 2
        test_conv = BatchNorm2d(ic)

        x = QTensor(np.arange(1, 17).reshape([b, ic, 4, 1]),
                    requires_grad=True,
                    dtype=pyvqnet.kfloat32)
        y = test_conv.forward(x)
        print(y)

        # [
        # [[[-1.3242440],
        #  [-1.0834724],
        #  [-0.8427007],
        #  [-0.6019291]],
        # [[-1.3242440],
        #  [-1.0834724],
        #  [-0.8427007],
        #  [-0.6019291]]],
        # [[[0.6019291],
        #  [0.8427007],
        #  [1.0834724],
        #  [1.3242440]],
        # [[0.6019291],
        #  [0.8427007],
        #  [1.0834724],
        #  [1.3242440]]]
        # ]


BatchNorm1d
=================================

.. py:class:: pyvqnet.nn.BatchNorm1d(channel_num:int, momentum:float=0.1, epsilon:float = 1e-5, affine = True, beta_initializer=zeros, gamma_initializer=ones, dtype=None, name="")

    Applica la normalizzazione batch (Batch Normalization) su un input 2D (B,C) come descritto nell'articolo
    `Batch Normalization: Accelerating Deep Network Training by Reducing
    Internal Covariate Shift <https://arxiv.org/abs/1502.03167>`__ .

    .. math::

        y = \frac{x - \mathrm{E}[x]}{\sqrt{\mathrm{Var}[x] + \epsilon}} * \gamma + \beta

    dove :math:`\gamma` e :math:`\beta` sono parametri apprendibili. Inoltre, per default, durante l'addestramento questo layer mantiene stime
    progressive della media e varianza calcolate, che vengono poi utilizzate per la normalizzazione durante la valutazione.
    Le stime progressive vengono mantenute con un momentum predefinito di 0.1.


    :param channel_num: `int` - il numero di canali delle caratteristiche di input.
    :param momentum: `float` - momentum per il calcolo della media mobile pesata, default 0.1
    :param epsilon: `float` - costante di stabilità numerica, default 1e-5.
    :param affine: Un valore booleano che, quando impostato a ``True``, fa sì che questo modulo abbia parametri affini apprendibili per canale, inizializzati a 1 (per i pesi) e 0 (per i bias). Default: ``True``.
    :param beta_initializer: `callable` - default zeros.
    :param gamma_initializer: `callable` - default ones.
    :param dtype: Il tipo di dato del parametro, default: None, utilizza il tipo di dato predefinito kfloat32, che rappresenta un numero a virgola mobile a 32 bit.
    :param name: nome del livello di output
    :return: una classe BatchNorm1d

    Esempio::

        import numpy as np
        from pyvqnet.tensor import QTensor
        from pyvqnet.nn import BatchNorm1d
        import pyvqnet
        test_conv = BatchNorm1d(4)

        x = QTensor(np.arange(1, 17).reshape([4, 4]),
                    requires_grad=True,
                    dtype=pyvqnet.kfloat32)
        y = test_conv.forward(x)
        print(y)


        # [
        # [-1.3416405, -1.3416405, -1.3416405, -1.3416405],
        # [-0.4472135, -0.4472135, -0.4472135, -0.4472135],
        # [0.4472135, 0.4472135, 0.4472135, 0.4472135],
        # [1.3416405, 1.3416405, 1.3416405, 1.3416405]
        # ]



LayerNormNd
=================================

.. py:class:: pyvqnet.nn.layer_norm.LayerNormNd(normalized_shape: list, epsilon: float = 1e-5, affine = True, dtype=None,name="")

    La normalizzazione di layer viene eseguita sulle ultime dimensioni di qualsiasi input. Il metodo specifico è come descritto nell'articolo:
    `Layer Normalization <https://arxiv.org/abs/1607.06450>`__.

    .. math::
        y = \frac{x - \mathrm{E}[x]}{ \sqrt{\mathrm{Var}[x] + \epsilon}} * \gamma + \beta

    Per input come (B,C,H,W,D), ``norm_shape`` può essere [C,H,W,D],[H,W,D],[W,D] o [D] .

    :param norm_shape: `float` - forma da normalizzare.
    :param epsilon: `float` - costante di stabilità numerica, default 1e-5.
    :param affine: Un valore booleano che, quando impostato a ``True``, fa sì che questo modulo abbia parametri affini apprendibili per canale, inizializzati a 1 (per i pesi) e 0 (per i bias). Default: ``True``.
    :param dtype: Il tipo di dato del parametro, default: None, utilizza il tipo di dato predefinito kfloat32, che rappresenta un numero a virgola mobile a 32 bit.
    :param name: nome del livello di output.

    :return: una classe LayerNormNd.

    Esempio::

        import numpy as np
        from pyvqnet.tensor import QTensor
        from pyvqnet import kfloat32
        from pyvqnet.nn.layer_norm import LayerNormNd
        ic = 4
        test_conv = LayerNormNd([2,2])
        x = QTensor(np.arange(1,17).reshape([2,2,2,2]),requires_grad=True,dtype=kfloat32)
        y = test_conv.forward(x)
        print(y)
        # [
        # [[[-1.3416355, -0.4472118],
        #  [0.4472118, 1.3416355]],
        # [[-1.3416355, -0.4472118],
        #  [0.4472118, 1.3416355]]],
        # [[[-1.3416355, -0.4472118],
        #  [0.4472118, 1.3416355]],
        # [[-1.3416355, -0.4472118],
        #  [0.4472118, 1.3416355]]]
        # ]


LayerNorm2d
=================================

.. py:class:: pyvqnet.nn.layer_norm.LayerNorm2d(norm_size:int, epsilon:float = 1e-5, affine= True, dtype=None, name="")

    Applica la normalizzazione di layer su un mini-batch di input 4D come descritto nell'articolo
    `Layer Normalization <https://arxiv.org/abs/1607.06450>`__

    .. math::
        y = \frac{x - \mathrm{E}[x]}{ \sqrt{\mathrm{Var}[x] + \epsilon}} * \gamma + \beta

    La media e la deviazione standard vengono calcolate sulla dimensione delle ultime `D` dimensioni.

    Per input come (B,C,H,W), ``norm_size`` dovrebbe essere uguale a C * H * W.

    :param norm_size: `float` - dimensione di normalizzazione, uguale a C * H * W
    :param epsilon: `float` - costante di stabilità numerica, default 1e-5
    :param affine: Un valore booleano che, quando impostato a ``True``, fa sì che questo modulo abbia parametri affini apprendibili per canale, inizializzati a 1 (per i pesi) e 0 (per i bias). Default: ``True``.
    :param name: nome del livello di output
    :param dtype: Il tipo di dato del parametro, default: None, utilizza il tipo di dato predefinito kfloat32, che rappresenta un numero a virgola mobile a 32 bit.
    
    :return: una classe LayerNorm2d

    Esempio::

        import numpy as np
        import pyvqnet
        from pyvqnet.tensor import QTensor
        from pyvqnet.nn.layer_norm import LayerNorm2d
        ic = 4
        test_conv = LayerNorm2d(8)
        x = QTensor(np.arange(1,17).reshape([2,2,4,1]),requires_grad=True,dtype=pyvqnet.kfloat32)
        y = test_conv.forward(x)
        print(y)

        # [
        # [[[-1.5275238],
        #  [-1.0910884],
        #  [-0.6546531],
        #  [-0.2182177]],
        # [[0.2182177],
        #  [0.6546531],
        #  [1.0910884],
        #  [1.5275238]]],
        # [[[-1.5275238],
        #  [-1.0910884],
        #  [-0.6546531],
        #  [-0.2182177]],
        # [[0.2182177],
        #  [0.6546531],
        #  [1.0910884],
        #  [1.5275238]]]
        # ]

LayerNorm1d
=================================

.. py:class:: pyvqnet.nn.layer_norm.LayerNorm1d(norm_size:int, epsilon:float = 1e-5, affine= True, dtype=None,name="")

    Applica la normalizzazione di layer su un mini-batch di input 2D come descritto nell'articolo
    `Layer Normalization <https://arxiv.org/abs/1607.06450>`__

    .. math::
        y = \frac{x - \mathrm{E}[x]}{ \sqrt{\mathrm{Var}[x] + \epsilon}} * \gamma + \beta

    La media e la deviazione standard vengono calcolate sulla dimensione delle ultime dimensioni, dove ``norm_size``
    è il valore dell'ultima dimensione.

    :param norm_size: `float` - dimensione di normalizzazione, uguale all'ultima dimensione
    :param epsilon: `float` - costante di stabilità numerica, default 1e-5
    :param affine: Un valore booleano che, quando impostato a ``True``, fa sì che questo modulo abbia parametri affini apprendibili per canale, inizializzati a 1 (per i pesi) e 0 (per i bias). Default: ``True``.
    :param dtype: Il tipo di dato del parametro, default: None, utilizza il tipo di dato predefinito kfloat32, che rappresenta un numero a virgola mobile a 32 bit.
    :param name: nome del livello di output

    :return: una classe LayerNorm1d

    Esempio::

        import numpy as np
        import pyvqnet
        from pyvqnet.tensor import QTensor
        from pyvqnet.nn.layer_norm import LayerNorm1d
        test_conv = LayerNorm1d(4)
        x = QTensor(np.arange(1,17).reshape([4,4]),requires_grad=True,dtype=pyvqnet.kfloat32)
        y = test_conv.forward(x)
        print(y)

        # [
        # [-1.3416355, -0.4472118, 0.4472118, 1.3416355],
        # [-1.3416355, -0.4472118, 0.4472118, 1.3416355],
        # [-1.3416355, -0.4472118, 0.4472118, 1.3416355],
        # [-1.3416355, -0.4472118, 0.4472118, 1.3416355]
        # ]


GroupNorm
=============================================================

.. py:class:: pyvqnet.nn.group_norm.GroupNorm(num_groups: int, num_channels: int, epsilon = 1e-5, affine = True, dtype = None, name = "")

    Applica la normalizzazione di gruppo a un mini-batch di input. Input: :math:`(N, C, *)` dove :math:`C=\mathrm{num\_channels}` , Output: :math:`(N, C, *)` .

    Questo layer implementa l'operazione descritta nell'articolo `Group Normalization <https://arxiv.org/abs/1803.08494>`__

    .. math::

        y = \frac{x - \mathrm{E}[x]}{ \sqrt{\mathrm{Var}[x] + \epsilon}} * \gamma + \beta

    I canali di input sono divisi in :attr:`num_groups` gruppi, ciascuno contenente ``num_channels / num_groups`` canali. :attr:`num_channels` deve essere divisibile per :attr:`num_groups`. La media e la deviazione standard vengono calcolate separatamente per ciascun gruppo. Se :attr:`affine` è ``True``, allora :math:`\gamma` e :math:`\beta` sono apprendibili. Vettore di parametri di trasformazione affine per canale di dimensione :attr:`num_channels`.

    :param num_groups (int): Numero di gruppi in cui dividere i canali
    :param num_channels (int): Numero di canali previsti nell'input
    :param eps: Valore da aggiungere al denominatore per la stabilità numerica. Default: 1e-5
    :param affine: Un valore booleano che, quando impostato a ``True``, fa sì che questo modulo abbia parametri affini apprendibili per canale, inizializzati a 1 (per i pesi) e 0 (per i bias). Default: ``True``.
    :param dtype: Il tipo di dato del parametro, default: None, utilizza il tipo di dato predefinito kfloat32, che rappresenta un numero a virgola mobile a 32 bit.
    :param name: nome del livello di output

    :return: classe GroupNorm

    Esempio::

        import numpy as np
        from pyvqnet.tensor import QTensor
        from pyvqnet import kfloat32
        from pyvqnet.nn import GroupNorm
        test_conv = GroupNorm(2,10)
        x = QTensor(np.arange(0,60*2*5).reshape([2,10,3,2,5]),requires_grad=True,dtype=kfloat32)
        y = test_conv.forward(x)
        print(y)


Linear
=================================

.. py:class:: pyvqnet.nn.Linear(input_channels, output_channels, weight_initializer=None, bias_initializer=None,use_bias=True, dtype=None, name: str = "")

    Modulo lineare (livello completamente connesso).
    :math:`y = x@A.T + b`

    :param input_channels: `int` - numero di caratteristiche di input
    :param output_channels: `int` - numero di caratteristiche di output
    :param weight_initializer: `callable` - default normal
    :param bias_initializer: `callable` - default zeros
    :param use_bias: `bool` - default True
    :param dtype: Il tipo di dato del parametro, default: None, utilizza il tipo di dato predefinito kfloat32, che rappresenta un numero a virgola mobile a 32 bit.
    :param name: nome del livello di output

    :return: una classe Linear

    Esempio::

        import numpy as np
        import pyvqnet
        from pyvqnet.tensor import QTensor
        from pyvqnet.nn import Linear
        c1 =2
        c2 = 3
        cin = 7
        cout = 5
        n = Linear(cin,cout)
        input = QTensor(np.arange(1,c1*c2*cin+1).reshape((c1,c2,cin)),requires_grad=True,dtype=pyvqnet.kfloat32)
        y = n.forward(input)
        print(y)

        # [
        # [[4.3084583, -1.9228780, -0.3428757, 1.2840536, -0.5865945],
        #  [9.8339605, -5.5135884, -3.1228657, 4.3025794, -4.1492314],
        #  [15.3594627, -9.1042995, -5.9028554, 7.3211040, -7.7118683]],
        # [[20.8849659, -12.6950111, -8.6828451, 10.3396301, -11.2745066],
        #  [26.4104652, -16.2857227, -11.4628344, 13.3581581, -14.8371439],
        #  [31.9359703, -19.8764324, -14.2428246, 16.3766804, -18.3997803]]
        # ]


Dropout
=================================

.. py:class:: pyvqnet.nn.dropout.Dropout(dropout_rate = 0.5)

    Modulo Dropout. Il modulo dropout imposta casualmente a zero gli output di alcune unità, mentre ridimensiona le altre in base alla probabilità di dropout specificata.

    :param dropout_rate: `float` - probabilità che un neurone venga azzerato
    :return: una classe Dropout

    Esempio::

        from pyvqnet.nn.dropout import Dropout
        import numpy as np
        from pyvqnet.tensor import QTensor
        b = 2
        ic = 2
        x = QTensor(np.arange(-1*ic*2*2,(b-1)*ic*2*2.0).reshape([b,ic,2,2]),requires_grad=True)
        droplayer = Dropout(0.5)
        droplayer.train()
        y = droplayer(x)
        print(y)
        # [[[[-16. -14.]
        #    [-12.   0.]]

        #   [[ -8.  -6.]
        #    [ -4.  -2.]]]


        #  [[[  0.   2.]
        #    [  0.   6.]]

        #   [[  0.   0.]
        #    [  0.  14.]]]]

DropPath
=================================

.. py:class:: pyvqnet.nn.dropout.DropPath(dropout_rate = 0.5,name="")

    Il modulo DropPath elimina percorsi (in modo casuale profondo) su base campione per campione.

    :param dropout_rate: `float` - La probabilità che il neurone venga azzerato.
    :param name: Il nome di questo modulo, default "".

    :return: Istanza di DropPath.

    Esempio::

        import pyvqnet.nn as nn
        import pyvqnet.tensor as tensor

        x = tensor.randu([4])
        y = nn.DropPath()(x)
        print(y)
        #[0.9074978,0.9350062,0.6896403,0.3541051]


Pixel_Shuffle 
=================================

.. py:class:: pyvqnet.nn.pixel_shuffle.Pixel_Shuffle(upscale_factors)

    Riarrangia tensori di forma: (*, C * r^2, H, W) in un tensore di forma (*, C, H * r, W * r) dove r è il fattore di scala.

    :param upscale_factors: fattore per aumentare la trasformazione di scala

    :return:
            modulo Pixel_Shuffle

    Esempio::

        from pyvqnet.nn import Pixel_Shuffle
        from pyvqnet.tensor import tensor
        ps = Pixel_Shuffle(3)
        inx = tensor.ones([5,2,3,18,4,4])
        inx.requires_grad=  True
        y = ps(inx)
        print(y.shape)
        #[5, 2, 3, 2, 12, 12]

Pixel_Unshuffle 
=================================

.. py:class:: pyvqnet.nn.pixel_shuffle.Pixel_Unshuffle(downscale_factors)

    Inverte l'operazione Pixel_Shuffle riarrangiando gli elementi. Riarrangia un tensore di forma (*, C, H * r, W * r) in (*, C * r^2, H, W), dove r è il fattore di riduzione.
    
    :param downscale_factors: fattore per aumentare la trasformazione di scala

    :return:
            modulo Pixel_Unshuffle

    Esempio::

        from pyvqnet.nn import Pixel_Unshuffle
        from pyvqnet.tensor import tensor
        ps = Pixel_Unshuffle(3)
        inx = tensor.ones([5, 2, 3, 2, 12, 12])
        inx.requires_grad = True
        y = ps(inx)
        print(y.shape)
        #[5, 2, 3, 18, 4, 4]


GRU
=================================

.. py:class:: pyvqnet.nn.gru.GRU(input_size, hidden_size, num_layers=1, nonlinearity='tanh', batch_first=True, use_bias=True, bidirectional=False, dtype=None, name: str = '')


    Modulo Gated Recurrent Unit (GRU). Supporta impilamento multi-livello e configurazione bidirezionale.
    La formula di calcolo del GRU monodirezionale a singolo livello è la seguente:

    .. math::
        \begin{array}{ll}
            r_t = \sigma(W_{ir} x_t + b_{ir} + W_{hr} h_{(t-1)} + b_{hr}) \\
            z_t = \sigma(W_{iz} x_t + b_{iz} + W_{hz} h_{(t-1)} + b_{hz}) \\
            n_t = \tanh(W_{in} x_t + b_{in} + r_t * (W_{hn} h_{(t-1)}+ b_{hn})) \\
            h_t = (1 - z_t) * n_t + z_t * h_{(t-1)}
        \end{array}

    :param input_size: Dimensioni delle caratteristiche di input.
    :param hidden_size: Dimensioni delle caratteristiche nascoste.
    :param num_layers: Numero di livelli impilati. default: 1.
    :param batch_first: Se batch_first è True, la forma dell'input deve essere [batch_size,seq_len,feature_dim],
     se batch_first è False, la forma dell'input deve essere [seq_len,batch_size,feature_dim], default: True.
    :param use_bias: Se use_bias è False, questo modulo non conterrà bias. default: True.
    :param bidirectional: Se bidirectional è True, il modulo sarà un GRU bidirezionale. default: False.
    :param dtype: Il tipo di dato del parametro, default: None, utilizza il tipo di dato predefinito kfloat32, che rappresenta un numero a virgola mobile a 32 bit.
    :param name: nome del livello di output

    :return: Un'istanza del modulo GRU.

    Esempio::

        from pyvqnet.nn import GRU
        from pyvqnet.tensor import tensor

        rnn2 = GRU(4, 6, 2, batch_first=False, bidirectional=True)

        input = tensor.ones([5, 3, 4])
        h0 = tensor.ones([4, 3, 6])

        output, hn = rnn2(input, h0)
        print(output)
        print(hn)
        # [
        # [[0.2815045, 0.2056844, 0.0750246, 0.5802019, 0.3536537, 0.8136684, -0.0034523, 0.1634004, 0.6099871, 0.8451654, -0.2833570, 0.7294812],
        #  [0.2815045, 0.2056844, 0.0750246, 0.5802019, 0.3536537, 0.8136684, -0.0034523, 0.1634004, 0.6099871, 0.8451654, -0.2833570, 0.7294812],
        #  [0.2815045, 0.2056844, 0.0750246, 0.5802019, 0.3536537, 0.8136684, -0.0034523, 0.1634004, 0.6099871, 0.8451654, -0.2833570, 0.7294812]],
        # [[0.0490867, 0.0115325, -0.2797680, 0.4711050, -0.0687061, 0.7216146, 0.0258964, 0.0619203, 0.6341010, 0.8445141, -0.4164453, 0.7409840],
        #  [0.0490867, 0.0115325, -0.2797680, 0.4711050, -0.0687061, 0.7216146, 0.0258964, 0.0619203, 0.6341010, 0.8445141, -0.4164453, 0.7409840],
        #  [0.0490867, 0.0115325, -0.2797680, 0.4711050, -0.0687061, 0.7216146, 0.0258964, 0.0619203, 0.6341010, 0.8445141, -0.4164453, 0.7409840]],
        # [[0.0182974, -0.0536071, -0.4478674, 0.4315647, -0.2191887, 0.6492687, 0.1572548, 0.0839213, 0.6707115, 0.8444533, -0.3811499, 0.7448123],
        #  [0.0182974, -0.0536071, -0.4478674, 0.4315647, -0.2191887, 0.6492687, 0.1572548, 0.0839213, 0.6707115, 0.8444533, -0.3811499, 0.7448123],
        #  [0.0182974, -0.0536071, -0.4478674, 0.4315647, -0.2191887, 0.6492687, 0.1572548, 0.0839213, 0.6707115, 0.8444533, -0.3811499, 0.7448123]],
        # [[0.0722285, -0.0636698, -0.5457084, 0.3817562, -0.1890205, 0.5696942, 0.3855782, 0.2057217, 0.7370453, 0.8646453, -0.1967214, 0.7630759],
        #  [0.0722285, -0.0636698, -0.5457084, 0.3817562, -0.1890205, 0.5696942, 0.3855782, 0.2057217, 0.7370453, 0.8646453, -0.1967214, 0.7630759],
        #  [0.0722285, -0.0636698, -0.5457084, 0.3817562, -0.1890205, 0.5696942, 0.3855782, 0.2057217, 0.7370453, 0.8646453, -0.1967214, 0.7630759]],
        # [[0.1834545, -0.0489200, -0.6343678, 0.3061281, -0.0449328, 0.4901535, 0.6941375, 0.4570828, 0.8433002, 0.9152645, 0.2342478, 0.8299093],
        #  [0.1834545, -0.0489200, -0.6343678, 0.3061281, -0.0449328, 0.4901535, 0.6941375, 0.4570828, 0.8433002, 0.9152645, 0.2342478, 0.8299093],
        #  [0.1834545, -0.0489200, -0.6343678, 0.3061281, -0.0449328, 0.4901535, 0.6941375, 0.4570828, 0.8433002, 0.9152645, 0.2342478, 0.8299093]]
        # ]
        # [
        # [[-0.8070476, -0.5560303, 0.7575479, -0.2368367, 0.4228620, -0.2573725],
        #  [-0.8070476, -0.5560303, 0.7575479, -0.2368367, 0.4228620, -0.2573725],
        #  [-0.8070476, -0.5560303, 0.7575479, -0.2368367, 0.4228620, -0.2573725]],
        # [[-0.3857390, -0.3195596, 0.0281313, 0.8734715, -0.4499536, 0.2270730],
        #  [-0.3857390, -0.3195596, 0.0281313, 0.8734715, -0.4499536, 0.2270730],
        #  [-0.3857390, -0.3195596, 0.0281313, 0.8734715, -0.4499536, 0.2270730]],
        # [[0.1834545, -0.0489200, -0.6343678, 0.3061281, -0.0449328, 0.4901535],
        #  [0.1834545, -0.0489200, -0.6343678, 0.3061281, -0.0449328, 0.4901535],
        #  [0.1834545, -0.0489200, -0.6343678, 0.3061281, -0.0449328, 0.4901535]],
        # [[-0.0034523, 0.1634004, 0.6099871, 0.8451654, -0.2833570, 0.7294812],
        #  [-0.0034523, 0.1634004, 0.6099871, 0.8451654, -0.2833570, 0.7294812],
        #  [-0.0034523, 0.1634004, 0.6099871, 0.8451654, -0.2833570, 0.7294812]]
        # ]

RNN 
=================================

.. py:class:: pyvqnet.nn.rnn.RNN(input_size, hidden_size, num_layers=1, nonlinearity='tanh', batch_first=True, use_bias=True, bidirectional=False, dtype=None, name: str = '')


    Modulo Recurrent Neural Network (RNN), utilizza :math:`\tanh` o :math:`\text{ReLU}` come funzione di attivazione.
    Sono supportati RNN bidirezionali e RNN multistrato.
    La formula di calcolo di un RNN monodirezionale a singolo livello è la seguente:

    .. math::
        h_t = \tanh(W_{ih} x_t + b_{ih} + W_{hh} h_{(t-1)} + b_{hh})

    Se :attr:`nonlinearity` è ``'relu'``, allora :math:`\text{ReLU}` sostituirà :math:`\tanh`.

    :param input_size: Dimensioni delle caratteristiche di input.
    :param hidden_size: Dimensioni delle caratteristiche nascoste.
    :param num_layers: Numero di livelli impilati. default: 1.
    :param nonlinearity: funzione di attivazione non lineare, default: ``'tanh'`` .
    :param batch_first: Se batch_first è True, la forma dell'input deve essere [batch_size,seq_len,feature_dim],
     se batch_first è False, la forma dell'input deve essere [seq_len,batch_size,feature_dim], default: True.
    :param use_bias: Se use_bias è False, questo modulo non conterrà bias. default: True.
    :param bidirectional: Se bidirectional è True, il modulo sarà un RNN bidirezionale. default: False.
    :param dtype: Il tipo di dato del parametro, default: None, utilizza il tipo di dato predefinito kfloat32, che rappresenta un numero a virgola mobile a 32 bit.
    :param name: nome del livello di output

    :return: Un'istanza del modulo RNN.

    Esempio::

        from pyvqnet.nn import RNN
        from pyvqnet.tensor import tensor

        rnn2 = RNN(4, 6, 2, batch_first=False, bidirectional = True)

        input = tensor.ones([5, 3, 4])
        h0 = tensor.ones([4, 3, 6])
        output, hn = rnn2(input, h0)
        print(output)
        print(hn)
        # [
        # [[-0.4481719, 0.4345263, 0.0284741, 0.6886298, 0.8672314, -0.3574123, 0.8238092, -0.2751125, -0.4704098, 0.7624499, -0.4156595, -0.1646518],
        #  [-0.4481719, 0.4345263, 0.0284741, 0.6886298, 0.8672314, -0.3574123, 0.8238092, -0.2751125, -0.4704098, 0.7624499, -0.4156595, -0.1646518],
        #  [-0.4481719, 0.4345263, 0.0284741, 0.6886298, 0.8672314, -0.3574123, 0.8238092, -0.2751125, -0.4704098, 0.7624499, -0.4156595, -0.1646518]],
        # [[-0.5737326, 0.1401956, -0.6656274, 0.3557707, 0.4083472, 0.3605195, 0.6767184, -0.2054843, -0.2875977, 0.6573941, -0.3289444, -0.1988498],
        #  [-0.5737326, 0.1401956, -0.6656274, 0.3557707, 0.4083472, 0.3605195, 0.6767184, -0.2054843, -0.2875977, 0.6573941, -0.3289444, -0.1988498],
        #  [-0.5737326, 0.1401956, -0.6656274, 0.3557707, 0.4083472, 0.3605195, 0.6767184, -0.2054843, -0.2875977, 0.6573941, -0.3289444, -0.1988498]],
        # [[-0.4233001, 0.1252111, -0.7437832, 0.2092323, 0.5826398, 0.5207447, 0.7403980, -0.0006015, -0.4055642, 0.6553873, -0.0861093, -0.2096289],
        #  [-0.4233001, 0.1252111, -0.7437832, 0.2092323, 0.5826398, 0.5207447, 0.7403980, -0.0006015, -0.4055642, 0.6553873, -0.0861093, -0.2096289],
        #  [-0.4233001, 0.1252111, -0.7437832, 0.2092323, 0.5826398, 0.5207447, 0.7403980, -0.0006015, -0.4055642, 0.6553873, -0.0861093, -0.2096289]],
        # [[-0.3636788, 0.3627384, -0.6542842, 0.0563165, 0.5711210, 0.5174620, 0.4968840, -0.3591014, -0.5738643, 0.7505787, -0.1767489, 0.2954176], [-0.3636788, 0.3627384, -0.6542842, 0.0563165, 0.5711210, 0.5174620, 0.4968840, -0.3591014, -0.5738643, 0.7505787, -0.1767489, 0.2954176], [-0.3636788, 0.3627384, -0.6542842, 0.0563165, 0.5711210, 0.5174620, 0.4968840, -0.3591014, -0.5738643, 0.7505787, -0.1767489, 0.2954176]],
        # [[-0.1619987, 0.3079547, -0.5022690, -0.2989357, 0.2861646, 0.4965633, 0.4618312, -0.4173903, 0.1423969, -0.2332578, -0.4014739, 0.0601179],
        #  [-0.1619987, 0.3079547, -0.5022690, -0.2989357, 0.2861646, 0.4965633, 0.4618312, -0.4173903, 0.1423969, -0.2332578, -0.4014739, 0.0601179],
        #  [-0.1619987, 0.3079547, -0.5022690, -0.2989357, 0.2861646, 0.4965633, 0.4618312, -0.4173903, 0.1423969, -0.2332578, -0.4014739, 0.0601179]]
        # ]
        # [
        # [[-0.1878589, -0.5177042, -0.3672480, 0.1613673, 0.4321197, 0.6168041],
        #  [-0.1878589, -0.5177042, -0.3672480, 0.1613673, 0.4321197, 0.6168041],
        #  [-0.1878589, -0.5177042, -0.3672480, 0.1613673, 0.4321197, 0.6168041]],
        # [[-0.7923757, 0.0184400, -0.2851982, -0.6367047, 0.5933805, -0.6244841],
        #  [-0.7923757, 0.0184400, -0.2851982, -0.6367047, 0.5933805, -0.6244841],
        #  [-0.7923757, 0.0184400, -0.2851982, -0.6367047, 0.5933805, -0.6244841]],
        # [[-0.1619987, 0.3079547, -0.5022690, -0.2989357, 0.2861646, 0.4965633],
        #  [-0.1619987, 0.3079547, -0.5022690, -0.2989357, 0.2861646, 0.4965633],
        #  [-0.1619987, 0.3079547, -0.5022690, -0.2989357, 0.2861646, 0.4965633]],
        # [[0.8238092, -0.2751125, -0.4704098, 0.7624499, -0.4156595, -0.1646518],
        #  [0.8238092, -0.2751125, -0.4704098, 0.7624499, -0.4156595, -0.1646518],
        #  [0.8238092, -0.2751125, -0.4704098, 0.7624499, -0.4156595, -0.1646518]]
        # ]



LSTM
=================================

.. py:class:: pyvqnet.nn.lstm.LSTM(input_size, hidden_size, num_layers=1, batch_first=True, use_bias=True, bidirectional=False, dtype=None, name: str = '')

    Modulo Long Short-Term Memory (LSTM). Supporta LSTM bidirezionale, LSTM multistrato impilato e altre configurazioni.
    La formula di calcolo di un LSTM monodirezionale a singolo livello è la seguente:

    .. math::
        \begin{array}{ll} \\
            i_t = \sigma(W_{ii} x_t + b_{ii} + W_{hi} h_{t-1} + b_{hi}) \\
            f_t = \sigma(W_{if} x_t + b_{if} + W_{hf} h_{t-1} + b_{hf}) \\
            g_t = \tanh(W_{ig} x_t + b_{ig} + W_{hg} h_{t-1} + b_{hg}) \\
            o_t = \sigma(W_{io} x_t + b_{io} + W_{ho} h_{t-1} + b_{ho}) \\
            c_t = f_t \odot c_{t-1} + i_t \odot g_t \\
            h_t = o_t \odot \tanh(c_t) \\
        \end{array}

    :param input_size: Dimensioni delle caratteristiche di input.
    :param hidden_size: Dimensioni delle caratteristiche nascoste.
    :param num_layers: Numero di livelli impilati. default: 1.
    :param batch_first: Se batch_first è True, la forma dell'input deve essere [batch_size,seq_len,feature_dim],
     se batch_first è False, la forma dell'input deve essere [seq_len,batch_size,feature_dim], default: True.
    :param use_bias: Se use_bias è False, questo modulo non conterrà bias. default: True.
    :param bidirectional: Se bidirectional è True, il modulo sarà un LSTM bidirezionale. default: False.
    :param dtype: Il tipo di dato del parametro, default: None, utilizza il tipo di dato predefinito kfloat32, che rappresenta un numero a virgola mobile a 32 bit.
    :param name: nome del livello di output

    :return: Un'istanza del modulo LSTM.

    Esempio::

        from pyvqnet.nn import LSTM
        from pyvqnet.tensor import tensor

        rnn2 = LSTM(4, 6, 2, batch_first=False, bidirectional = True)

        input = tensor.ones([5, 3, 4])
        h0 = tensor.ones([4, 3, 6])
        c0 = tensor.ones([4, 3, 6])
        output, (hn, cn) = rnn2(input, (h0, c0))

        print(output)
        print(hn)
        print(cn)

        # [
        # [[0.1585344, 0.1758823, 0.4273642, 0.1640685, 0.1030634, 0.1657819, -0.0197110, 0.2073366, 0.0050953, -0.1467141, -0.1413236, -0.1404487], 
        #  [0.1585344, 0.1758823, 0.4273642, 0.1640685, 0.1030634, 0.1657819, -0.0197110, 0.2073366, 0.0050953, -0.1467141, -0.1413236, -0.1404487], 
        #  [0.1585344, 0.1758823, 0.4273642, 0.1640685, 0.1030634, 0.1657819, -0.0197110, 0.2073366, 0.0050953, -0.1467141, -0.1413236, -0.1404487]],[[0.0366294, 0.1421610, 0.2401645, 0.0672358, 0.2205958, 0.1306419, 0.0129892, 0.1626964, 0.0116193, -0.1181969, -0.1101109, -0.0844855],  
        #  [0.0366294, 0.1421610, 0.2401645, 0.0672358, 0.2205958, 0.1306419, 0.0129892, 0.1626964, 0.0116193, -0.1181969, -0.1101109, -0.0844855],  
        #  [0.0366294, 0.1421610, 0.2401645, 0.0672358, 0.2205958, 0.1306419, 0.0129892, 0.1626964, 0.0116193, -0.1181969, -0.1101109, -0.0844855]], 
        # [[0.0169496, 0.1236289, 0.1416115, -0.0382225, 0.2277734, 0.0378894, 0.0252284, 0.1317508, 0.0191879, -0.0379719, -0.0707748, -0.0134158],
        #  [0.0169496, 0.1236289, 0.1416115, -0.0382225, 0.2277734, 0.0378894, 0.0252284, 0.1317508, 0.0191879, -0.0379719, -0.0707748, -0.0134158],
        #  [0.0169496, 0.1236289, 0.1416115, -0.0382225, 0.2277734, 0.0378894, 0.0252284, 0.1317508, 0.0191879, -0.0379719, -0.0707748, -0.0134158]],[[0.0223647, 0.1227054, 0.0959055, -0.1043864, 0.2314414, -0.0289589, 0.0346038, 0.1147739, 0.0461321, 0.0998507, 0.0097069, 0.0886721],
        #  [0.0223647, 0.1227054, 0.0959055, -0.1043864, 0.2314414, -0.0289589, 0.0346038, 0.1147739, 0.0461321, 0.0998507, 0.0097069, 0.0886721],
        #  [0.0223647, 0.1227054, 0.0959055, -0.1043864, 0.2314414, -0.0289589, 0.0346038, 0.1147739, 0.0461321, 0.0998507, 0.0097069, 0.0886721]],
        # [[0.0345177, 0.1308527, 0.0884205, -0.1468191, 0.2236451, -0.0705002, 0.0672482, 0.1278620, 0.1676001, 0.2955882, 0.2448514, 0.1802391],
        #  [0.0345177, 0.1308527, 0.0884205, -0.1468191, 0.2236451, -0.0705002, 0.0672482, 0.1278620, 0.1676001, 0.2955882, 0.2448514, 0.1802391],
        #  [0.0345177, 0.1308527, 0.0884205, -0.1468191, 0.2236451, -0.0705002, 0.0672482, 0.1278620, 0.1676001, 0.2955882, 0.2448514, 0.1802391]]
        # ]
        # [
        # [[0.1687095, -0.2087553, 0.0254020, 0.3340017, 0.2515125, 0.2364762],
        #  [0.1687095, -0.2087553, 0.0254020, 0.3340017, 0.2515125, 0.2364762],
        #  [0.1687095, -0.2087553, 0.0254020, 0.3340017, 0.2515125, 0.2364762]],
        # [[0.2621196, 0.2436198, -0.1790378, 0.0883382, -0.0479185, -0.0838870],
        #  [0.2621196, 0.2436198, -0.1790378, 0.0883382, -0.0479185, -0.0838870],
        #  [0.2621196, 0.2436198, -0.1790378, 0.0883382, -0.0479185, -0.0838870]],
        # [[0.0345177, 0.1308527, 0.0884205, -0.1468191, 0.2236451, -0.0705002],
        #  [0.0345177, 0.1308527, 0.0884205, -0.1468191, 0.2236451, -0.0705002],
        #  [0.0345177, 0.1308527, 0.0884205, -0.1468191, 0.2236451, -0.0705002]],
        # [[-0.0197110, 0.2073366, 0.0050953, -0.1467141, -0.1413236, -0.1404487],
        #  [-0.0197110, 0.2073366, 0.0050953, -0.1467141, -0.1413236, -0.1404487],
        #  [-0.0197110, 0.2073366, 0.0050953, -0.1467141, -0.1413236, -0.1404487]]
        # ]
        # [
        # [[0.3588709, -0.3877619, 0.0519047, 0.5984558, 0.7709259, 1.0954115],
        #  [0.3588709, -0.3877619, 0.0519047, 0.5984558, 0.7709259, 1.0954115],
        #  [0.3588709, -0.3877619, 0.0519047, 0.5984558, 0.7709259, 1.0954115]],
        # [[0.4557160, 0.6420789, -0.4407433, 0.1704233, -0.1592798, -0.1966903],
        #  [0.4557160, 0.6420789, -0.4407433, 0.1704233, -0.1592798, -0.1966903],
        #  [0.4557160, 0.6420789, -0.4407433, 0.1704233, -0.1592798, -0.1966903]],
        # [[0.0681112, 0.4060420, 0.1333674, -0.3497016, 0.7122995, -0.1229735],
        #  [0.0681112, 0.4060420, 0.1333674, -0.3497016, 0.7122995, -0.1229735],
        #  [0.0681112, 0.4060420, 0.1333674, -0.3497016, 0.7122995, -0.1229735]],
        # [[-0.0378819, 0.4589431, 0.0142352, -0.3194987, -0.3059436, -0.3285254],
        #  [-0.0378819, 0.4589431, 0.0142352, -0.3194987, -0.3059436, -0.3285254],
        #  [-0.0378819, 0.4589431, 0.0142352, -0.3194987, -0.3059436, -0.3285254]]
        # ]


Dynamic_GRU
=================================

.. py:class:: pyvqnet.nn.gru.Dynamic_GRU(input_size,hidden_size, num_layers=1, batch_first=True, use_bias=True, bidirectional=False, dtype=None, name: str = '')
    
    Applica una GRU (Gated Recurrent Unit) RNN multistrato a una sequenza di input di lunghezza dinamica.

    Il primo input dovrebbe essere un input di sequenza batch di lunghezza variabile, definito
    attraverso la classe ``tensor.PackedSequence``.
    La classe ``tensor.PackedSequence`` puo essere costruita
    chiamando le seguenti funzioni in sequenza: ``pad_sequence``, ``pack_pad_sequence``.

    Il primo output di Dynamic_GRU e anche una classe ``tensor.PackedSequence``,
    che puo essere decompressa in un QTensor normale usando ``tensor.pad_pack_sequence``.

    Per ogni elemento nella sequenza di input, ogni livello calcola la seguente formula:

    .. math::
        \begin{array}{ll}
            r_t = \sigma(W_{ir} x_t + b_{ir} + W_{hr} h_{(t-1)} + b_{hr}) \
            z_t = \sigma(W_{iz} x_t + b_{iz} + W_{hz} h_{(t-1)} + b_{hz}) \
            n_t = \tanh(W_{in} x_t + b_{in} + r_t * (W_{hn} h_{(t-1)}+ b_{hn})) \
            h_t = (1 - z_t) * n_t + z_t * h_{(t-1)}
        \end{array}

    :param input_size: Dimensione delle caratteristiche di input.
    :param hidden_size: Dimensione delle caratteristiche nascoste.
    :param num_layers: Numero di livelli ricorrenti. Default: 1
    :param batch_first: Se True, la forma dell'input e [dimensione batch, lunghezza sequenza, dimensione caratteristiche]. Se False, la forma dell'input e [lunghezza sequenza, dimensione batch, dimensione caratteristiche], default True.
    :param use_bias: Se False, il livello non utilizza pesi di bias b_ih e b_hh. Default: true.
    :param bidirectional: Se true, diventa un GRU bidirezionale. Default: false.
    :param dtype: Il tipo di dato del parametro, default: None, utilizza il tipo di dato predefinito kfloat32, che rappresenta un numero a virgola mobile a 32 bit.
    :param name: nome del livello di output

    :return: Una classe Dynamic_GRU

    Esempio::

        from pyvqnet.nn import Dynamic_GRU
        from pyvqnet.tensor import tensor
        seq_len = [4,1,2]
        input_size = 4
        batch_size =3
        hidden_size = 2
        ml = 2
        rnn2 = Dynamic_GRU(input_size,
                        hidden_size=2,
                        num_layers=2,
                        batch_first=False,
                        bidirectional=True)

        a = tensor.arange(1, seq_len[0] * input_size + 1).reshape(
            [seq_len[0], input_size])
        b = tensor.arange(1, seq_len[1] * input_size + 1).reshape(
            [seq_len[1], input_size])
        c = tensor.arange(1, seq_len[2] * input_size + 1).reshape(
            [seq_len[2], input_size])

        y = tensor.pad_sequence([a, b, c], False)

        input = tensor.pack_pad_sequence(y,
                                        seq_len,
                                        batch_first=False,
                                        enforce_sorted=False)

        h0 = tensor.ones([ml * 2, batch_size, hidden_size])

        output, hn = rnn2(input, h0)

        seq_unpacked, lens_unpacked =         tensor.pad_packed_sequence(output, batch_first=False)
        print(seq_unpacked)
        print(lens_unpacked)
        # [
        # [[-0.3918380, 0.0056273, 0.9018179, 0.9006662],
        #  [-0.3715909, 0.0307644, 0.9756137, 0.9705784],
        #  [-0.3917399, 0.0057521, 0.9507942, 0.9456232]],
        # [[-0.6348240, -0.0603764, 0.9014163, 0.8903066],
        #  [0, 0, 0, 0],
        #  [-0.6333261, -0.0592172, 0.9660671, 0.9580816]],
        # [[-0.4571511, 0.0210018, 0.9151242, 0.9011748],
        #  [0, 0, 0, 0],
        #  [0, 0, 0, 0]],
        # [[-0.3585358, 0.0918219, 0.9496037, 0.9391552],
        #  [0, 0, 0, 0],
        #  [0, 0, 0, 0]]
        # ]
        # [4 1 2]


Dynamic_RNN 
=================================

.. py:class:: pyvqnet.nn.rnn.Dynamic_RNN(input_size, hidden_size, num_layers=1, nonlinearity='tanh', batch_first=True, use_bias=True, bidirectional=False, dtype=None, name: str = '')
    
    Applica reti neurali ricorrenti (RNN) a sequenze di input di lunghezza dinamica.

    Il primo input dovrebbe essere un input di sequenza batch di lunghezza variabile, definito
    attraverso la classe ``tensor.PackedSequence``.
    La classe ``tensor.PackedSequence`` puo essere costruita
    chiamando le seguenti funzioni in sequenza: ``pad_sequence``, ``pack_pad_sequence``.

    Il primo output di Dynamic_RNN e anche una classe ``tensor.PackedSequence``,
    che puo essere decompressa in un QTensor normale usando ``tensor.pad_pack_sequence``.

    Modulo Recurrent Neural Network (RNN), utilizza :math:`\tanh` o :math:`\text{ReLU}` come funzione di attivazione. Supporta configurazioni bidirezionali e multistrato.
    La formula di calcolo di un RNN monodirezionale a singolo livello e la seguente:

    .. math::
        h_t = \tanh(W_{ih} x_t + b_{ih} + W_{hh} h_{(t-1)} + b_{hh})
    
    Se :attr:`nonlinearity` e ``'relu'``, allora :math:`\text{ReLU}` sostituira :math:`\tanh`.

    :param input_size: Dimensione delle caratteristiche di input.
    :param hidden_size: Dimensione delle caratteristiche nascoste.
    :param num_layers: Numero di livelli RNN impilati, default: 1.
    :param nonlinearity: Funzione di attivazione non lineare, default ``'tanh'``.
    :param batch_first: Se True, la forma dell'input e [dimensione batch, lunghezza sequenza, dimensione caratteristiche],
      Se False, la forma dell'input e [lunghezza sequenza, dimensione batch, dimensione caratteristiche], default True.
    :param use_bias: Se False, il modulo non applica bias, default: True.
    :param bidirectional: Se True, diventa un RNN bidirezionale, default: False.
    :param dtype: Il tipo di dato del parametro, default: None, utilizza il tipo di dato predefinito kfloat32, che rappresenta un numero a virgola mobile a 32 bit.
    :param name: nome del livello di output

    :return: Istanza Dynamic_RNN

    Esempio::

        from pyvqnet.nn import Dynamic_RNN
        from pyvqnet.tensor import tensor
        seq_len = [4,1,2]
        input_size = 4
        batch_size =3
        hidden_size = 2
        ml = 2
        rnn2 = Dynamic_RNN(input_size,
                        hidden_size=2,
                        num_layers=2,
                        batch_first=False,
                        bidirectional=True,
                        nonlinearity='relu')

        a = tensor.arange(1, seq_len[0] * input_size + 1).reshape(
            [seq_len[0], input_size])
        b = tensor.arange(1, seq_len[1] * input_size + 1).reshape(
            [seq_len[1], input_size])
        c = tensor.arange(1, seq_len[2] * input_size + 1).reshape(
            [seq_len[2], input_size])

        y = tensor.pad_sequence([a, b, c], False)

        input = tensor.pack_pad_sequence(y,
                                        seq_len,
                                        batch_first=False,
                                        enforce_sorted=False)

        h0 = tensor.ones([ml * 2, batch_size, hidden_size])

        output, hn = rnn2(input, h0)

        seq_unpacked, lens_unpacked =         tensor.pad_packed_sequence(output, batch_first=False)
        print(seq_unpacked)
        print(lens_unpacked)

        # [
        # [[1.2980951, 0, 0, 0],
        #  [1.5040692, 0, 0, 0],
        #  [1.4927036, 0, 0, 0.1065927]],
        # [[2.6561704, 0, 0, 0.2532321],
        #  [0, 0, 0, 0],
        #  [3.1472805, 0, 0, 0]],
        # [[5.1231661, 0, 0, 0.7596353],
        #  [0, 0, 0, 0],
        #  [0, 0, 0, 0]],
        # [[8.4954977, 0, 0, 0.8191229],
        #  [0, 0, 0, 0],
        #  [0, 0, 0, 0]]
        # ]
        # [4 1 2]



Dynamic_LSTM
=================================

.. py:class:: pyvqnet.nn.lstm.Dynamic_LSTM(input_size, hidden_size, num_layers=1, batch_first=True, use_bias=True, bidirectional=False, dtype=None, name: str = '')
    
    Applica reti neurali ricorrenti LSTM (Long Short-Term Memory) a sequenze di input di lunghezza dinamica.

    Il primo input dovrebbe essere un input di sequenza batch di lunghezza variabile, definito
    attraverso la classe ``tensor.PackedSequence``.
    La classe ``tensor.PackedSequence`` puo essere costruita
    chiamando le seguenti funzioni in sequenza: ``pad_sequence``, ``pack_pad_sequence``.

    Il primo output di Dynamic_LSTM e anche una classe ``tensor.PackedSequence``,
    che puo essere decompressa in un QTensor normale usando ``tensor.pad_pack_sequence``.

    Modulo Recurrent Neural Network (RNN), utilizza :math:`\tanh` o :math:`\text{ReLU}` come funzione di attivazione. Supporta configurazioni bidirezionali e multistrato.
    La formula di calcolo di un RNN monodirezionale a singolo livello e la seguente:

    .. math::
        \begin{array}{ll} \
            i_t = \sigma(W_{ii} x_t + b_{ii} + W_{hi} h_{t-1} + b_{hi}) \
            f_t = \sigma(W_{if} x_t + b_{if} + W_{hf} h_{t-1} + b_{hf}) \
            g_t = \tanh(W_{ig} x_t + b_{ig} + W_{hg} h_{t-1} + b_{hg}) \
            o_t = \sigma(W_{io} x_t + b_{io} + W_{ho} h_{t-1} + b_{ho}) \
            c_t = f_t \odot c_{t-1} + i_t \odot g_t \
            h_t = o_t \odot \tanh(c_t) \
        \end{array}

    :param input_size: Dimensione delle caratteristiche di input.
    :param hidden_size: Dimensione delle caratteristiche nascoste.
    :param num_layers: Numero di livelli LSTM impilati, default: 1.
    :param batch_first: Se True, la forma dell'input e [dimensione batch, lunghezza sequenza, dimensione caratteristiche],
      Se False, la forma dell'input e [lunghezza sequenza, dimensione batch, dimensione caratteristiche], default True.
    :param use_bias: Se False, il modulo non applica bias, default: True.
    :param bidirectional: Se True, diventa un LSTM bidirezionale, default: False.
    :param dtype: Il tipo di dato del parametro, default: None, utilizza il tipo di dato predefinito kfloat32, che rappresenta un numero a virgola mobile a 32 bit.
    :param name: nome del livello di output

    :return: Istanza Dynamic_LSTM

    Esempio::

        from pyvqnet.nn import Dynamic_LSTM
        from pyvqnet.tensor import tensor

        input_size = 2
        hidden_size = 2
        ml = 2
        seq_len = [3, 4, 1]
        batch_size = 3
        rnn2 = Dynamic_LSTM(input_size,
                            hidden_size=hidden_size,
                            num_layers=ml,
                            batch_first=False,
                            bidirectional=True)

        a = tensor.arange(1, seq_len[0] * input_size + 1).reshape(
            [seq_len[0], input_size])
        b = tensor.arange(1, seq_len[1] * input_size + 1).reshape(
            [seq_len[1], input_size])
        c = tensor.arange(1, seq_len[2] * input_size + 1).reshape(
            [seq_len[2], input_size])
        a.requires_grad = True
        b.requires_grad = True
        c.requires_grad = True
        y = tensor.pad_sequence([a, b, c], False)

        input = tensor.pack_pad_sequence(y,
                                        seq_len,
                                        batch_first=False,
                                        enforce_sorted=False)

        h0 = tensor.ones([ml * 2, batch_size, hidden_size])
        c0 = tensor.ones([ml * 2, batch_size, hidden_size])

        output, (hn, cn) = rnn2(input, (h0, c0))

        seq_unpacked, lens_unpacked =         tensor.pad_packed_sequence(output, batch_first=False)

        print(seq_unpacked)
        print(lens_unpacked)

        # [
        # [[0.2038177, 0.1139005, 0.2312966, -0.1140076],
        #  [0.1992285, 0.1221137, 0.2277344, -0.3147154],
        #  [0.2293468, 0.0681745, 0.2426863, 0.2572871]],
        # [[0.1398094, -0.0150359, 0.2513067, 0.0783743],
        #  [0.1328388, -0.0031956, 0.2324090, -0.1962151],
        #  [0, 0, 0, 0]],
        # [[0.0898260, -0.0706460, 0.2396922, 0.2323916],
        #  [0.0817787, -0.0449937, 0.2388873, -0.0000469],
        #  [0, 0, 0, 0]],
        # [[0, 0, 0, 0],
        #  [0.0532839, -0.0870574, 0.2397324, 0.2103822],
        #  [0, 0, 0, 0]]
        # ]
        # [3 4 1]


Interpolate
=================================
.. py:class:: pyvqnet.nn.Interpolate(size, scale_factor, mode = "nearest", align_corners = None,  recompute_scale_factor = None, name = "")

    Campiona l'input verso il basso o verso l'alto.

    Attualmente sono supportati solo dati di input quadridimensionali.

    Le dimensioni di input vengono interpretate nella forma: `B x C x H x W`.

    Le modalita disponibili per il ridimensionamento sono: ``nearest``, ``bilinear``, ``bicubic``.

    :param size: dimensione spaziale dell'output.
    :param scale_factor: moltiplicatore per la dimensione spaziale.
    :param mode: algoritmo utilizzato per il sovracampionamento ``nearest`` | ``bilinear`` | ``bicubic``.
    :param align_corners: Geometricamente, consideriamo i pixel dell'input e dell'output
            come quadrati anziche punti. Se impostato a ``True``, i tensori di input e output
            sono allineati dai punti centrali dei loro pixel d'angolo, preservando i valori ai pixel d'angolo.
            Se impostato a ``False``, i tensori di input e output sono allineati dai punti d'angolo
            dei loro pixel d'angolo, e l'interpolazione utilizza il padding con valori al bordo
            per i valori fuori dai limiti, rendendo questa operazione *indipendente* dalla dimensione dell'input
            quando :attr:`scale_factor` rimane lo stesso. Ha effetto solo quando :attr:`mode`
            e ``bilinear``, ``bicubic``.
    :param recompute_scale_factor: ricalcola il scale_factor per l'uso nel calcolo dell'interpolazione.
    :param name: Nome del modulo.

    Esempio::

        from pyvqnet.nn import Interpolate
        from pyvqnet.tensor import tensor
        import pyvqnet
        pyvqnet.utils.set_random_seed(1)

        import numpy as np
        np.random.seed(0)

        np_ = np.random.randn(36).reshape((1, 1, 6, 6)).astype(np.float32)
        mode_ = "bilinear"
        size_ = 3

        class Model(pyvqnet.nn.Module):

            def __init__(self):
                super().__init__()
                self.inter = Interpolate(size = size_, mode=mode_)
                self.ln = pyvqnet.nn.Linear(9, 1)

            def forward(self, x):
                x = self.inter(x).reshape((1,-1))
                x = self.ln(x)
                return 2 * x

        input_vqnet = tensor.QTensor(np_,  dtype=pyvqnet.kfloat32, requires_grad=True)
        loss_pyvqnet = pyvqnet.nn.MeanSquaredError()
        model_vqnet = Model()
        output_vqnet = model_vqnet(input_vqnet)
        l = loss_pyvqnet(tensor.QTensor([[1.0]]), output_vqnet)
        l.backward()
        print(model_vqnet.parameters()[0].grad)


fuse_module
=================================
.. py:class:: pyvqnet.nn.fuse_module(model)

    Viene utilizzato per fondere i moduli vicini corrispondenti del modello nella fase di inferenza in un unico modulo,
    riducendo la quantita di calcolo nella fase di inferenza del modello e aumentando la velocita di inferenza del modello.

    Le sequenze di moduli attualmente supportate sono le seguenti:

    conv, bn

    linear, bn

    Le altre sequenze rimangono invariate, per cui il primo modulo nella lista viene sostituito con il modulo fuso, e gli altri vengono sostituiti con ``Identity``.

    :param input: Include la modellazione dei moduli di fusione.

    :return: Modello con moduli fusi.

    Esempi::
    
        from pyvqnet import tensor,kfloat32
        from pyvqnet.nn import Linear
        from pyvqnet.nn import Module, BatchNorm1d, BatchNorm2d, Conv1D, Conv2D

        from pyvqnet.qnn.vqc import *
        from pyvqnet.optim import Adam
        from pyvqnet.nn import Module,BinaryCrossEntropy, Sigmoid
        from pyvqnet.data import data_generator
        import numpy as np
        from pyvqnet.tensor import QTensor

        from time import time
        from pyvqnet.utils import set_random_seed
        from pyvqnet.nn import fuse_module

        def get_accuracy(result, label):
            result = (result > 0.5).astype(4)
            score = tensor.sums(result == label)
            return score.item()
            
        class Model(Module):
            def __init__(self):

                super(Model, self).__init__()

                self.conv1 = Conv2D(1,2,1)
                self.ban = BatchNorm2d(2)

                self.conv2 = Conv2D(2,1,1)
                self.li1 = Linear(64,1)
                self.ac = Sigmoid()
                
            def forward(self, x):
                x = self.conv1(x)
                x = self.ban(x)
                x = self.conv2(x).reshape([-1,64])
                x = self.li1(x)
                x = self.ac(x)

                return x
        X_train = np.random.randn(80, 1, 8, 8)
        y_train = np.random.choice([0,1], size=(80))
        
        model = Model().toGPU()
        optimizer = Adam(model.parameters(), lr = 0.001)
        batch_size = 20
        epoch = 80
        loss = BinaryCrossEntropy()
        print("start training..............")
        model.train()
        
        loss_history = []
        accuracy_history = []
        time2 = time()
        
        for i in range(epoch):
            count = 0
            sum_loss = 0
            accuary = 0
            t = 0
            for data, label in data_generator(X_train, y_train, batch_size, False):
                optimizer.zero_grad()
                data, label = QTensor(data,requires_grad=True).toGPU(), QTensor(label,
                                                    dtype=kfloat32,
                                                    requires_grad=False).toGPU()
                
                result = model(data)
                
                loss_b = loss(label.reshape([-1, 1]), result)
                
                loss_b.backward()
                optimizer._step()

                sum_loss += loss_b.item()
                count += batch_size
                accuary += get_accuracy(result, label.reshape([-1,1]))
                t = t + 1
            
            loss_history.append(sum_loss/count)
            accuracy_history.append(accuary/count)
            print(
                f"epoch:{i}, #### loss:{sum_loss/count} #####accuracy:{accuary/count}"
            )
        print(f"run time {time() - time2}")
        
        
        model.eval()

        input = tensor.randn((20, 1, 8, 8)).toGPU()
        print(list(model.named_children()))
        time_a = time()
        a = model(input)
        print(f"fuse before {time() - time_a}")
        fuse_module(model)
        model.toGPU()
        print(list(model.named_children()))
        time_b = time()
        b = model(input)
        print(f"fuse after {time() - time_b}")
        
        print(tensor.max(tensor.abs(a - b)).item())


SDPA
=================================
.. py:class:: pyvqnet.transformer.e2eqvit.SDPA(attn_mask=None,dropout_p=0.,scale=None,is_causal=False)

    Meccanismo di attenzione a prodotto scalare SDPA.

    :param attn_mask: Maschera di attenzione; la forma deve essere trasmissibile alla forma dei pesi di attenzione.
    :param dropout_p: Probabilita di dropout; se maggiore di 0.0, viene applicato il dropout.
    :param scale: Fattore di scala applicato prima della softmax.
    :param is_causal: Se true, assume una maschera di attenzione causale superiore sinistra e genera un errore se sia attn_mask che is_causal sono impostati.
    
    Esempi::
    
        from pyvqnet.transformer import SDPA
        from pyvqnet import tensor
        import pyvqnet
        from time import time
        import pyvqnet.nn as nn
        import numpy as np

        np.random.seed(42)

        query_np = np.random.randn(3, 3, 3, 5).astype(np.float32) 
        key_np = np.random.randn(3, 3, 3, 5).astype(np.float32)   
        value_np = np.random.randn(3, 3, 3, 5).astype(np.float32) 

        model = SDPA(tensor.QTensor([1.])).toGPU()

        query_p = tensor.QTensor(query_np, dtype=pyvqnet.kfloat32, requires_grad=True).toGPU()
        key_p = tensor.QTensor(key_np, dtype=pyvqnet.kfloat32, requires_grad=True).toGPU()
        value_p = tensor.QTensor(value_np, dtype=pyvqnet.kfloat32, requires_grad=True).toGPU()

        out_sdpa = model(query_p, key_p, value_p)

        out_sdpa.backward()


Livello di Funzione di Perdita
********************************************************

.. note::

        Nota che a differenza di PyTorch e altri framework, nella funzione forward della seguente funzione di perdita, il primo parametro e l'etichetta e il secondo parametro e il valore previsto.

MeanSquaredError
=================================

.. py:class:: pyvqnet.nn.MeanSquaredError

    Crea un criterio che misura l'errore quadratico medio (norma L2 al quadrato) tra
    ciascun elemento nell'input :math:`x` e nel target :math:`y`.

    La perdita non ridotta puo essere descritta come:

    .. math::
        \ell(x, y) = L = \{l_1,\dots,l_N\}^\top, \quad
        l_n = \left( x_n - y_n \right)^2,

    dove :math:`N` e la dimensione del batch. Quindi:

    .. math::
        \ell(x, y) =
            \operatorname{mean}(L)


    :math:`x` e :math:`y` sono QTensor di forme arbitrarie con un totale
    di :math:`n` elementi ciascuno.

    L'operazione di media opera ancora su tutti gli elementi e divide per :math:`n`.

    :param name: nome del livello di output

    :return: una classe MeanSquaredError

    Parametri per la funzione forward di perdita:

        x: :math:`(N, *)` dove :math:`*` significa un numero qualsiasi di dimensioni aggiuntive

        y: :math:`(N, *)`, stessa forma dell'input

    Esempio::
    
        from pyvqnet.tensor import QTensor
        from pyvqnet import kfloat64
        from pyvqnet.nn import MeanSquaredError
        y = QTensor([[0, 0, 1, 0, 0, 0, 0, 0, 0, 0]],
                    requires_grad=False,
                    dtype=kfloat64)
        x = QTensor([[0.1, 0.05, 0.7, 0, 0.05, 0.1, 0, 0, 0, 0]],
                    requires_grad=True,
                    dtype=kfloat64)

        loss_result = MeanSquaredError()
        result = loss_result(y, x)
        print(result)

        # [0.0115000]
        

BinaryCrossEntropy
=================================

.. py:class:: pyvqnet.nn.BinaryCrossEntropy

    Misura l'entropia incrociata binaria tra il target e l'output:

    La perdita non ridotta puo essere descritta come:

    .. math::
        \ell(x, y) = L = \{l_1,\dots,l_N\}^\top, \quad
        l_n = - w_n \left[ y_n \cdot \log x_n + (1 - y_n) \cdot \log (1 - x_n) \right],

    dove :math:`N` e la dimensione del batch.

    .. math::
        \ell(x, y) = \operatorname{mean}(L)

    :return: una classe BinaryCrossEntropy

    Parametri per la funzione forward di perdita:

        x: :math:`(N, *)` dove :math:`*` significa un numero qualsiasi di dimensioni aggiuntive

        y: :math:`(N, *)`, stessa forma dell'input

    Esempio::

        import pyvqnet
        from pyvqnet.tensor import QTensor
        x = QTensor([[0.3, 0.7, 0.2], [0.2, 0.3, 0.1]], requires_grad=True)
        y = QTensor([[0, 1.0, 0], [0, 0.0, 1]], requires_grad=True)

        loss_result = pyvqnet.nn.BinaryCrossEntropy()
        result = loss_result(y, x)
        result.backward()
        print(result)

        # [0.6364825]

CategoricalCrossEntropy
=================================

.. py:class:: pyvqnet.nn.CategoricalCrossEntropy

    Questo criterio combina LogSoftmax e NLLLoss in un'unica classe.

    La perdita puo essere descritta come segue, dove `class` e l'indice della classe del target:

    .. math::
        \text{loss}(x, class) = -\log\left(\frac{\exp(x[class])}{\sum_j \exp(x[j])}\right)
                       = -x[class] + \log\left(\sum_j \exp(x[j])\right)

    :return: una classe CategoricalCrossEntropy

    Parametri per la funzione forward di perdita:

        x: :math:`(N, *)` dove :math:`*` significa un numero qualsiasi di dimensioni aggiuntive

        y: :math:`(N, *)`, stessa forma dell'input, deve avere tipo di dato intero a 64 bit.

    Esempio::

        from pyvqnet.tensor import QTensor
        from pyvqnet import kfloat32,kint64
        from pyvqnet.nn import CategoricalCrossEntropy
        x = QTensor([[1, 2, 3, 4, 5],
        [1, 2, 3, 4, 5],
        [1, 2, 3, 4, 5]], requires_grad=True,dtype=kfloat32)
        y = QTensor([[0, 1, 0, 0, 0], [0, 1, 0, 0, 0], [1, 0, 0, 0, 0]], requires_grad=False,dtype=kint64)
        loss_result = CategoricalCrossEntropy()
        result = loss_result(y, x)
        print(result)

        # [3.7852428]


SoftmaxCrossEntropy
=================================

.. py:class:: pyvqnet.nn.SoftmaxCrossEntropy

    Questo criterio combina LogSoftmax e NLLLoss in un'unica classe con maggiore stabilita numerica.

    La perdita puo essere descritta come segue, dove `class` e l'indice della classe del target:

    .. math::
        \text{loss}(x, class) = -\log\left(\frac{\exp(x[class])}{\sum_j \exp(x[j])}\right)
                       = -x[class] + \log\left(\sum_j \exp(x[j])\right)

    :return: una classe SoftmaxCrossEntropy

    Parametri per la funzione forward di perdita:

        x: :math:`(N, *)` dove :math:`*` significa un numero qualsiasi di dimensioni aggiuntive

        y: :math:`(N, *)`, stessa forma dell'input, deve avere tipo di dato intero a 64 bit.

    Esempio::

        from pyvqnet.tensor import QTensor
        from pyvqnet import kfloat32, kint64
        from pyvqnet.nn import SoftmaxCrossEntropy
        x = QTensor([[1, 2, 3, 4, 5], [1, 2, 3, 4, 5], [1, 2, 3, 4, 5]],
                    requires_grad=True,
                    dtype=kfloat32)
        y = QTensor([[0, 1, 0, 0, 0], [0, 1, 0, 0, 0], [1, 0, 0, 0, 0]],
                    requires_grad=False,
                    dtype=kint64)
        loss_result = SoftmaxCrossEntropy()
        result = loss_result(y, x)
        result.backward()
        print(result)

        # [3.7852478]


NLL_Loss
=================================

.. py:class:: pyvqnet.nn.NLL_Loss()

    La perdita media di log-verosimiglianza negativa. E utile per addestrare un problema di classificazione con `C` classi.

    L'`x` fornito attraverso una chiamata forward deve contenere log-probabilita di ciascuna classe. `x` deve essere un tensore di dimensione :math:`(N, C)` o :math:`(N, C, d_1, d_2, ..., d_K)`
    con :math:`K \geq 1` per il caso `K`-dimensionale. La `y` attesa da questa perdita dovrebbe essere un indice di classe nell'intervallo :math:`[0, C-1]` dove `C = numero di classi`.

    .. math::

        \ell(x, y) = L = \{l_1,\dots,l_N\}^\top, \quad
        l_n = -
            \sum_{n=1}^N \frac{1}{N}x_{n,y_n}, \quad

    :return: una classe NLL_Loss

    Parametri per la funzione forward di perdita:

        x: :math:`(N, *)`, l'output della funzione di perdita, che puo essere una variabile multidimensionale.

        y: :math:`(N, *)`, il valore vero atteso dalla funzione di perdita, deve avere tipo di dato intero a 64 bit.


    Esempio::

        from pyvqnet.tensor import QTensor
        from pyvqnet import kfloat32,kint64
        from pyvqnet.nn import NLL_Loss

        x = QTensor([
            0.9476322568516703, 0.226547421131723, 0.5944201443911326,
            0.42830868492969476, 0.76414068655387, 0.00286059168094277,
            0.3574236812873617, 0.9096948856639084, 0.4560809854582528,
            0.9818027091583286, 0.8673569904602182, 0.9860275114020933,
            0.9232667066664217, 0.303693313961628, 0.8461034903175555
        ])
        x= x.reshape([1, 3, 1, 5])
        x.requires_grad = True
        y = QTensor([[[2, 1, 0, 0, 2]]], dtype=kint64)

        loss_result = NLL_Loss()
        result = loss_result(y, x)
        print(result)
        #[-0.6187226]

CrossEntropyLoss
=================================

.. py:class:: pyvqnet.nn.CrossEntropyLoss()

    Questo criterio combina LogSoftmax e NLLLoss in un'unica classe.

    `x` deve contenere punteggi grezzi e non normalizzati per ciascuna classe. `x` deve essere un tensore di dimensione :math:`(C)` per input non raggruppati, :math:`(N, C)` o :math:`(N, C, d_1, d_2, ..., d_K)` con :math:`K \geq 1` per il caso `K`-dimensionale.

    La perdita puo essere descritta come segue, dove `class` e l'indice della classe del target:

    .. math::

        \text{loss}(x, class) = -\log\left(\frac{\exp(x[class])}{\sum_j \exp(x[j])}\right)
                       = -x[class] + \log\left(\sum_j \exp(x[j])\right)

    :return: una classe CrossEntropyLoss

    Parametri per la funzione forward di perdita:

        x: :math:`(N, *)`, l'output della funzione di perdita, che puo essere una variabile multidimensionale.

        y: :math:`(N, *)`, il valore vero atteso dalla funzione di perdita, deve avere tipo di dato intero a 64 bit.


    Esempio::

        from pyvqnet.tensor import QTensor
        from pyvqnet import kfloat32,kint64
        from pyvqnet.nn import CrossEntropyLoss
        x = QTensor([
            0.9476322568516703, 0.226547421131723, 0.5944201443911326,
            0.42830868492969476, 0.76414068655387, 0.00286059168094277,
            0.3574236812873617, 0.9096948856639084, 0.4560809854582528,
            0.9818027091583286, 0.8673569904602182, 0.9860275114020933,
            0.9232667066664217, 0.303693313961628, 0.8461034903175555
        ])
        x.reshape_([1, 3, 1, 5])
        x.requires_grad = True
        y = QTensor([[[2, 1, 0, 0, 2]]], dtype=kint64)

        loss_result = CrossEntropyLoss()
        result = loss_result(y, x)
        print(result)

        #[1.1508200]



Funzione di Attivazione
********************************************************


Activation
=================================
.. py:class:: pyvqnet.nn.activation.Activation

    Classe base delle attivazioni. Le funzioni di attivazione specifiche ereditano da questa classe.

Sigmoid
=================================
.. py:class:: pyvqnet.nn.Sigmoid(name: str = '')

        Applica una funzione di attivazione sigmoide al livello specificato.

        .. math::
            \text{Sigmoid}(x) = \frac{1}{1 + \exp(-x)}

        :param name: nome del livello di output
        :return: livello di attivazione Sigmoid

        Esempi::

            from pyvqnet.nn import Sigmoid
            from pyvqnet.tensor import QTensor
            layer = Sigmoid()
            y = layer(QTensor([1.0, 2.0, 3.0, 4.0]))
            print(y)

            # [0.7310586, 0.8807970, 0.9525741, 0.9820138]

Softplus
=================================
.. py:class:: pyvqnet.nn.Softplus(name: str = '')

        Applica la funzione di attivazione softplus al livello specificato.

        .. math::
            \text{Softplus}(x) = \log(1 + \exp(x))

        :param name: nome del livello di output
        :return: livello di attivazione Softplus

    Esempi::

        from pyvqnet.nn import Softplus
        from pyvqnet.tensor import QTensor
        layer = Softplus()
        y = layer(QTensor([1.0, 2.0, 3.0, 4.0]))
        print(y)

        # [1.3132616, 2.1269281, 3.0485873, 4.0181499]

Softsign
=================================
.. py:class:: pyvqnet.nn.Softsign(name: str = '')

        Applica la funzione di attivazione softsign al livello specificato.

        .. math::
            \text{SoftSign}(x) = \frac{x}{ 1 + |x|}

        :param name: nome del livello di output
        :return: livello di attivazione Softsign

        Esempi::

            from pyvqnet.nn import Softsign
            from pyvqnet.tensor import QTensor
            layer = Softsign()
            y = layer(QTensor([1.0, 2.0, 3.0, 4.0]))
            print(y)

            # [0.5000000, 0.6666667, 0.7500000, 0.8000000]

Softmax
=================================
.. py:class:: pyvqnet.nn.Softmax(axis: int = - 1, name: str = '')

    Applica una funzione di attivazione softmax al livello specificato.

    .. math::
        \text{Softmax}(x_{i}) = \frac{\exp(x_i)}{\sum_j \exp(x_j)}


    :param axis: dimensione su cui operare (-1 per l'ultimo asse), default = -1
    :param name: nome del livello di output
    :return: livello di attivazione Softmax

    Esempi::

        from pyvqnet.nn import Softmax
        from pyvqnet.tensor import QTensor
        layer = Softmax()
        y = layer(QTensor([1.0, 2.0, 3.0, 4.0]))
        print(y)

        # [0.0320586, 0.0871443, 0.2368828, 0.6439142]


HardSigmoid
=================================
.. py:class:: pyvqnet.nn.HardSigmoid(name: str = '')

    Applica una funzione di attivazione hard sigmoid al livello specificato.

    .. math::
        \text{Hardsigmoid}(x) = \begin{cases}
            0 & \text{ if } x \le -3, \
            1 & \text{ if } x \ge +3, \
            x / 6 + 1 / 2 & \text{otherwise}
        \end{cases}

    :param name: nome del livello di output
    :return: livello di attivazione Hard Sigmoid

    Esempi::

        from pyvqnet.nn import HardSigmoid
        from pyvqnet.tensor import QTensor
        layer = HardSigmoid()
        y = layer(QTensor([1.0, 2.0, 3.0, 4.0]))
        print(y)

        # [0.6666667, 0.8333334, 1, 1]

ReLu
=================================
.. py:class:: pyvqnet.nn.ReLu(name: str = '')

    Applica una funzione di attivazione a unita lineare rettificata al livello specificato.

    .. math::
        \text{ReLu}(x) = \begin{cases}
        x, & \text{ if } x > 0\
        0, & \text{ if } x \leq 0
        \end{cases}


    :param name: nome del livello di output
    :return: livello di attivazione ReLu

    Esempi::

        from pyvqnet.nn import ReLu
        from pyvqnet.tensor import QTensor
        layer = ReLu()
        y = layer(QTensor([-1, 2.0, -3, 4.0]))
        print(y)

        # [0, 2, 0, 4]

LeakyReLu
=================================
.. py:class:: pyvqnet.nn.LeakyReLu(alpha: float = 0.01, name: str = '')

    Applica la versione leaky di una funzione di attivazione a unita lineare rettificata
    al livello specificato.

    .. math::
        \text{LeakyRelu}(x) =
        \begin{cases}
        x, & \text{ if } x \geq 0 \
        \alpha * x, & \text{ otherwise }
        \end{cases}

    :param alpha: coefficiente LeakyRelu, default: 0.01
    :param name: nome del livello di output
    :return: livello di attivazione Leaky ReLu

    Esempi::

        from pyvqnet.nn import LeakyReLu
        from pyvqnet.tensor import QTensor
        layer = LeakyReLu()
        y = layer(QTensor([-1, 2.0, -3, 4.0]))
        print(y)

        # [-0.0100000, 2, -0.0300000, 4]

Gelu
=================================
.. py:class:: pyvqnet.nn.Gelu(approximate="tanh", name="")
    
    Applica la funzione a unita lineare di errore gaussiano:

    .. math:: \text{GELU}(x) = x * \Phi(x)

    Quando il parametro di approssimazione e 'tanh', GELU viene stimato tramite:

    .. math:: \text{GELU}(x) = 0.5 * x * (1 + \text{Tanh}(\sqrt{2 / \pi} * (x + 0.044715 * x^3)))

    :param approximate: Metodo di calcolo approssimato, default "tanh".
    :param name: Nome del livello della funzione di attivazione, default "".

    :return: Istanza del livello della funzione di attivazione Gelu.

    Esempi::

        from pyvqnet.tensor import randu, ones_like
        from pyvqnet.nn import Gelu
        qa = randu([5,4])
        qb = Gelu()(qa)
        print(qb)
        # [[0.0292515,0.0668998,0.4036024,0.8369502],
        # [0.1929213,0.1981275,0.2358531,0.7790835],
        # [0.1754935,0.6204091,0.2354677,0.2409406],
        # [0.4238827,0.804715,0.1633414,0.2853],
        # [0.1959854,0.590143,0.553995,0.0008423]]

ELU
=================================
.. py:class:: pyvqnet.nn.ELU(alpha: float = 1.0, name: str = '')

    Applica la funzione di attivazione a unita lineare esponenziale al livello specificato.

    .. math::
        \text{ELU}(x) = \begin{cases}
        x, & \text{ if } x > 0\
        \alpha * (\exp(x) - 1), & \text{ if } x \leq 0
        \end{cases}

    :param alpha: coefficiente Elu, default: 1.0
    :param name: nome del livello di output
    :return: livello di attivazione Elu

    Esempi::

        from pyvqnet.nn import ELU
        from pyvqnet.tensor import QTensor
        layer = ELU()
        y = layer(QTensor([-1, 2.0, -3, 4.0]))
        print(y)

        # [-0.6321205, 2, -0.9502130, 4]

Tanh
=================================
.. py:class:: pyvqnet.nn.Tanh(name: str = '')

    Applica la funzione di attivazione tangente iperbolica al livello specificato.

    .. math::
        \text{Tanh}(x) = \frac{\exp(x) - \exp(-x)} {\exp(x) + \exp(-x)}

    :param name: nome del livello di output
    :return: livello di attivazione tangente iperbolica

    Esempi::

        from pyvqnet.nn import Tanh
        from pyvqnet.tensor import QTensor
        layer = Tanh()
        y = layer(QTensor([-1, 2.0, -3, 4.0]))
        print(y)

        # [-0.7615942, 0.9640276, -0.9950548, 0.9993293]



.. _Optimizer:

Modulo Ottimizzatore
********************************************************


Optimizer
=================================
.. py:class:: pyvqnet.optim.optimizer.Optimizer(params, lr=0.01)

    Classe base per tutti gli ottimizzatori.

    :param params: parametri del modello da ottimizzare
    :param lr: tasso di apprendimento del modello (default: 0.01)

adadelta
=================================
.. py:class:: pyvqnet.optim.adadelta.Adadelta(params, lr=0.01, beta=0.99, epsilon=1e-8)

    ADADELTA: Un metodo di tasso di apprendimento adattivo. riferimento: (https://arxiv.org/abs/1212.5701)

    .. math::

        E(g_t^2) &= \beta * E(g_{t-1}^2) + (1-\beta) * g^2\\
        Square\_avg &= \sqrt{ ( E(dx_{t-1}^2) + \epsilon ) / ( E(g_t^2) + \epsilon ) }\\
        E(dx_t^2) &= \beta * E(dx_{t-1}^2) + (1-\beta) * (-g*square\_avg)^2 \\
        param\_new &= param - lr * Square\_avg

    :param params: parametri del modello da ottimizzare
    :param lr: tasso di apprendimento del modello (default: 0.01)
    :param beta: per il calcolo della media mobile dei gradienti al quadrato (default: 0.99)
    :param epsilon: termine aggiunto al denominatore per migliorare la stabilita numerica (default: 1e-8)
    :return: un ottimizzatore Adadelta

    Esempio::

        import numpy as np
        from pyvqnet.optim import adadelta
        from pyvqnet.tensor import QTensor
        w = np.arange(24).reshape(1,2,3,4).astype(np.float64)    
        param = QTensor(w)
        param.grad = QTensor(np.arange(24).reshape(1, 2, 3, 4).astype(np.float64))
        params = [param]
        opti = adadelta.Adadelta(params)

        for i in range(1,3):
            opti._step()
            print(param)

        # [
        # [[[0, 0.9999900, 1.9999900, 2.9999900],    
        #  [3.9999900, 4.9999900, 5.9999900, 6.9999900],     
        #  [7.9999900, 8.9999905, 9.9999905, 10.9999905]],   
        # [[11.9999905, 12.9999905, 13.9999905, 14.9999905], 
        #  [15.9999905, 16.9999905, 17.9999905, 18.9999905], 
        #  [19.9999905, 20.9999905, 21.9999905, 22.9999905]]]
        # ]

        # [
        # [[[0, 0.9999800, 1.9999800, 2.9999800],    
        #  [3.9999800, 4.9999800, 5.9999800, 6.9999800],     
        #  [7.9999800, 8.9999800, 9.9999800, 10.9999800]],   
        # [[11.9999800, 12.9999800, 13.9999800, 14.9999800], 
        #  [15.9999800, 16.9999809, 17.9999809, 18.9999809], 
        #  [19.9999809, 20.9999809, 21.9999809, 22.9999809]]]
        # ]

adagrad
=================================
.. py:class:: pyvqnet.optim.adagrad.Adagrad(params, lr=0.01, epsilon=1e-8 )

    Implementa l'algoritmo Adagrad. riferimento: (https://databricks.com/glossary/adagrad)

    .. math::
        \begin{aligned}
        moment\_new &= moment + g * g\\param\_new
        &= param - \frac{lr * g}{\sqrt{moment\_new} + \epsilon}
        \end{aligned}

    :param params: parametri del modello da ottimizzare
    :param lr: tasso di apprendimento del modello (default: 0.01)
    :param epsilon: termine aggiunto al denominatore per migliorare la stabilita numerica (default: 1e-8)
    :return: un ottimizzatore Adagrad

    Esempio::

        import numpy as np
        from pyvqnet.optim import adagrad
        from pyvqnet.tensor import QTensor
        w = np.arange(24).reshape(1,2,3,4).astype(np.float64)    
        param = QTensor(w)
        param.grad = QTensor(np.arange(24).reshape(1, 2, 3, 4).astype(np.float64))
        params = [param]
        opti = adagrad.Adagrad(params)

        for i in range(1,3):
            opti._step() 
            print(param)

        # [
        # [[[0, 0.9900000, 1.9900000, 2.9900000],
        #  [3.9900000, 4.9899998, 5.9899998, 6.9899998],
        #  [7.9899998, 8.9899998, 9.9899998, 10.9899998]],
        # [[11.9899998, 12.9899998, 13.9899998, 14.9899998],
        #  [15.9899998, 16.9899998, 17.9899998, 18.9899998],
        #  [19.9899998, 20.9899998, 21.9899998, 22.9899998]]]
        # ]

        # [
        # [[[0, 0.9829289, 1.9829290, 2.9829290],
        #  [3.9829290, 4.9829288, 5.9829288, 6.9829288],
        #  [7.9829288, 8.9829283, 9.9829283, 10.9829283]],
        # [[11.9829283, 12.9829283, 13.9829283, 14.9829283],
        #  [15.9829283, 16.9829292, 17.9829292, 18.9829292],
        #  [19.9829292, 20.9829292, 21.9829292, 22.9829292]]]
        # ]


AdamW
=================================
.. py:class:: pyvqnet.optim.adam.AdamW(params, lr=0.01, beta1=0.9, beta2=0.999, epsilon=1e-8, weight_decay=0.01, amsgrad: bool = False)
    
    Implementa l'algoritmo AdamW.

    .. math::
        t=t+1

    .. math::
        param\_new = param - lr*weight\_decay*param
    .. math::
        moment\_1\_new=\beta1*moment\_1+(1-\beta1)g
    .. math::
        moment\_2\_new=\beta2*moment\_2+(1-\beta2)g*g
    .. math::
        lr = lr*\frac{\sqrt{1-\beta2^t}}{1-\beta1^t}
    
    Se il parametro amsgrad e True:

    .. math::
        moment\_2\_max = max(moment\_2\_max,moment\_2)
    .. math::
        param\_new=param\_new-lr*\frac{moment\_1}{\sqrt{moment\_2\_max}+\epsilon}
    
    altrimenti:

    .. math::
        param\_new=param\_new-lr*\frac{moment\_1}{\sqrt{moment\_2}+\epsilon}

    :param params: Parametri del modello che devono essere ottimizzati.
    :param lr: tasso di apprendimento (default: 0.01).
    :param beta1: Coefficiente utilizzato per calcolare la media mobile del gradiente e del suo quadrato (default: 0.9).
    :param beta2: Coefficiente utilizzato per calcolare la media mobile del gradiente e del suo quadrato (default: 0.999).
    :param epsilon: Costante da aggiungere al denominatore per migliorare la stabilita numerica (default: 1e-8).
    :param weight_decay: Coefficiente di decadimento del peso, default 0.01.
    :param amsgrad: Se utilizzare la variante AMSGrad di questo algoritmo (default: False).
    :return: Un ottimizzatore AdamW.

    Esempio::

        from pyvqnet.optim import adam
        import numpy as np
        from pyvqnet.tensor import QTensor
        w = np.arange(24).reshape(1,2,3,4).astype(np.float64)
        param = QTensor(w)
        param.grad = QTensor(np.arange(24).reshape(1,2,3,4).astype(np.float64))
        params = [param]
        opti = adam.AdamW(params, lr=0.5)

        for i in range(1,3):
            opti.step()
        print(param)
        # [[[[ 0. ,-0.007475 , 0.98255 , 1.972575 ],
        # [2.9626, 3.952625, 4.9426501, 5.9326751],
        # [6.9227001, 7.9127251, 8.9027501, 9.8927751]],

        # [[10.8828001,11.8728251,12.8628501,13.8528751],
        # [14.8429002,15.8329252,16.8229502,17.8129752],
        # [18.8030002,19.7930252,20.7830502,21.7730752]]]]

Adam
=================================
.. py:class:: pyvqnet.optim.adam.Adam(params, lr=0.01, beta1=0.9, beta2=0.999, epsilon=1e-8,weight_decay = 0, amsgrad: bool = False)

    Adam: Un metodo per l'ottimizzazione stocastica riferimento: (https://arxiv.org/abs/1412.6980), regola dinamicamente il tasso di apprendimento di ciascun parametro utilizzando le stime del primo e del secondo momento del gradiente.

    .. math::
        t = t + 1
    .. math::
        param  = param - lr*weight\_decay*param
    .. math::
        moment\_1\_new=\beta1*moment\_1+(1-\beta1)g
    .. math::
        moment\_2\_new=\beta2*moment\_2+(1-\beta2)g*g
    .. math::
        lr = lr*\frac{\sqrt{1-\beta2^t}}{1-\beta1^t}

    se amsgrad = True:
    
    .. math::
        moment\_2\_max = max(moment\_2\_max,moment\_2)
    .. math::
        param\_new=param-lr*\frac{moment\_1}{\sqrt{moment\_2\_max}+\epsilon} 

    altrimenti:

    .. math::
        param\_new=param-lr*\frac{moment\_1}{\sqrt{moment\_2}+\epsilon} 


    :param params: parametri del modello da ottimizzare
    :param lr: tasso di apprendimento del modello (default: 0.01)
    :param beta1: coefficienti utilizzati per calcolare le medie mobili del gradiente e del suo quadrato (default: 0.9)
    :param beta2: coefficienti utilizzati per calcolare le medie mobili del gradiente e del suo quadrato (default: 0.999)
    :param epsilon: termine aggiunto al denominatore per migliorare la stabilita numerica (default: 1e-8)
    :param weight_decay: Coefficiente di decadimento del peso, default 0.
    :param amsgrad: se utilizzare la variante AMSGrad di questo algoritmo (default: False)
    :return: un ottimizzatore Adam

    Esempio::

        import numpy as np
        from pyvqnet.optim import adam
        from pyvqnet.tensor import QTensor
        w = np.arange(24).reshape(1,2,3,4).astype(np.float64)    
        param = QTensor(w)
        param.grad = QTensor(np.arange(24).reshape(1, 2, 3, 4).astype(np.float64))
        params = [param]
        opti = adam.Adam(params)
        
        for i in range(1,3):
            opti._step()
            print(param)

        # [
        # [[[0, 0.9900000, 1.9900000, 2.9900000],
        #  [3.9900000, 4.9899998, 5.9899998, 6.9899998],
        #  [7.9899998, 8.9899998, 9.9899998, 10.9899998]],
        # [[11.9899998, 12.9899998, 13.9899998, 14.9899998],
        #  [15.9899998, 16.9899998, 17.9899998, 18.9899998],
        #  [19.9899998, 20.9899998, 21.9899998, 22.9899998]]]
        # ]

        # [
        # [[[0, 0.9800000, 1.9800000, 2.9800000],
        #  [3.9800000, 4.9799995, 5.9799995, 6.9799995],
        #  [7.9799995, 8.9799995, 9.9799995, 10.9799995]],
        # [[11.9799995, 12.9799995, 13.9799995, 14.9799995],
        #  [15.9799995, 16.9799995, 17.9799995, 18.9799995],
        #  [19.9799995, 20.9799995, 21.9799995, 22.9799995]]]
        # ]


adamax
=================================
.. py:class:: pyvqnet.optim.adamax.Adamax(params, lr=0.01, beta1=0.9, beta2=0.999, epsilon=1e-8)

    Implementa l'algoritmo Adamax (una variante di Adam basata sulla norma infinito). riferimento: (https://arxiv.org/abs/1412.6980)

    .. math::
        \\t = t + 1
    .. math::
        moment\_new=\beta1*moment+(1-\beta1)g
    .. math::
        norm\_new = \max{(\beta1*norm+\epsilon, \left|g\right|)}
    .. math::
        lr = \frac{lr}{1-\beta1^t}
    .. math::
        param\_new = param - lr*\frac{moment\_new}{norm\_new}\\

    :param params: parametri del modello da ottimizzare
    :param lr: tasso di apprendimento del modello (default: 0.01)
    :param beta1: coefficienti utilizzati per calcolare le medie mobili del gradiente e del suo quadrato (default: 0.9)
    :param beta2: coefficienti utilizzati per calcolare le medie mobili del gradiente e del suo quadrato (default: 0.999)
    :param epsilon: termine aggiunto al denominatore per migliorare la stabilita numerica (default: 1e-8)
    :return: un ottimizzatore Adamax

    Esempio::

        import numpy as np
        from pyvqnet.optim import adamax
        from pyvqnet.tensor import QTensor
        w = np.arange(24).reshape(1,2,3,4).astype(np.float64)    
        param = QTensor(w)
        param.grad = QTensor(np.arange(24).reshape(1,2,3,4).astype(np.float64))
        params = [param]
        opti = adamax.Adamax(params)
        
        for i in range(1,3):
            opti._step()
            print(param)

        # [
        # [[[0, 0.9900000, 1.9900000, 2.9900000],
        #  [3.9900000, 4.9899998, 5.9899998, 6.9899998],
        #  [7.9899998, 8.9899998, 9.9899998, 10.9899998]],
        # [[11.9899998, 12.9899998, 13.9899998, 14.9899998],
        #  [15.9899998, 16.9899998, 17.9899998, 18.9899998],
        #  [19.9899998, 20.9899998, 21.9899998, 22.9899998]]]
        # ]

        # [
        # [[[0, 0.9800000, 1.9800000, 2.9800000],
        #  [3.9800000, 4.9799995, 5.9799995, 6.9799995],
        #  [7.9799995, 8.9799995, 9.9799995, 10.9799995]],
        # [[11.9799995, 12.9799995, 13.9799995, 14.9799995],
        #  [15.9799995, 16.9799995, 17.9799995, 18.9799995],
        #  [19.9799995, 20.9799995, 21.9799995, 22.9799995]]]
        # ]

rmsprop
=================================
.. py:class:: pyvqnet.optim.rmsprop.RMSProp(params, lr=0.01, beta=0.99, epsilon=1e-8)

    Implementa l'algoritmo RMSprop. riferimento: (https://arxiv.org/pdf/1308.0850v5.pdf)

    .. math::
        s_{t+1} = s_{t} + (1 - \beta)*(g)^2

    .. math::
        param_new = param -  \frac{g}{\sqrt{s_{t+1}} + epsilon}

    :param params: parametri del modello da ottimizzare
    :param lr: tasso di apprendimento del modello (default: 0.01)
    :param beta: coefficienti utilizzati per calcolare le medie mobili del gradiente e del suo quadrato (default: 0.99)
    :param epsilon: termine aggiunto al denominatore per migliorare la stabilita numerica (default: 1e-8)
    :return: un ottimizzatore RMSProp

    Esempio::

        import numpy as np
        from pyvqnet.optim import rmsprop
        from pyvqnet.tensor import QTensor
        w = np.arange(24).reshape(1,2,3,4).astype(np.float64)    
        param = QTensor(w)
        param.grad = QTensor(np.arange(24).reshape(1,2,3,4).astype(np.float64))
        params = [param]
        opti = rmsprop.RMSProp(params)
        
        for i in range(1,3):
            opti._step()
            print(param)

        # [
        # [[[0, 0.9000000, 1.9000000, 2.8999999],
        #  [3.8999999, 4.9000001, 5.9000001, 6.9000001],
        #  [7.9000001, 8.8999996, 9.8999996, 10.8999996]],
        # [[11.8999996, 12.8999996, 13.8999996, 14.8999996],
        #  [15.8999996, 16.8999996, 17.8999996, 18.8999996],
        #  [19.8999996, 20.8999996, 21.8999996, 22.8999996]]]
        # ]

        # [
        # [[[0, 0.8291118, 1.8291118, 2.8291118],
        #  [3.8291118, 4.8291121, 5.8291121, 6.8291121],
        #  [7.8291121, 8.8291111, 9.8291111, 10.8291111]],
        # [[11.8291111, 12.8291111, 13.8291111, 14.8291111],
        #  [15.8291111, 16.8291111, 17.8291111, 18.8291111],
        #  [19.8291111, 20.8291111, 21.8291111, 22.8291111]]]
        # ]

sgd
=================================
.. py:class:: pyvqnet.optim.sgd.SGD(params, lr=0.01, momentum=0, nesterov=False)

    Implementa l'algoritmo SGD. riferimento: (https://en.wikipedia.org/wiki/Stochastic_gradient_descent)

    .. math::

        \\param\_new=param-lr*g\\

    :param params: parametri del modello da ottimizzare
    :param lr: tasso di apprendimento del modello (default: 0.01)
    :param momentum: fattore di momentum (default: 0)
    :param nesterov: abilita il momentum di Nesterov (default: False)
    :return: un ottimizzatore SGD

    Esempio::

        import numpy as np
        from pyvqnet.optim import sgd
        from pyvqnet.tensor import QTensor
        w = np.arange(24).reshape(1,2,3,4).astype(np.float64)    
        param = QTensor(w)
        param.grad = QTensor(np.arange(24).reshape(1,2,3,4).astype(np.float64))
        params = [param]
        opti = sgd.SGD(params)

        for i in range(1,3):
            opti._step()
            print(param) 

        # [
        # [[[0, 0.9900000, 1.9800000, 2.9700000],
        #  [3.9600000, 4.9499998, 5.9400001, 6.9299998],
        #  [7.9200001, 8.9099998, 9.8999996, 10.8900003]],
        # [[11.8800001, 12.8699999, 13.8599997, 14.8500004],
        #  [15.8400002, 16.8299999, 17.8199997, 18.8099995],
        #  [19.7999992, 20.7900009, 21.7800007, 22.7700005]]]
        # ]

        # [
        # [[[0, 0.9800000, 1.9600000, 2.9400001],
        #  [3.9200001, 4.8999996, 5.8800001, 6.8599997],
        #  [7.8400002, 8.8199997, 9.7999992, 10.7800007]],
        # [[11.7600002, 12.7399998, 13.7199993, 14.7000008],
        #  [15.6800003, 16.6599998, 17.6399994, 18.6199989],
        #  [19.5999985, 20.5800018, 21.5600014, 22.5400009]]]
        # ]



Metriche
********************************************************


MSE
=================================

.. py:class:: pyvqnet.utils.metrics.MSE(y_true_Qtensor, y_pred_Qtensor)

    MSE: Errore Quadratico Medio (Mean Squared Error).

    :param y_true_Qtensor: Un QTensor di forma (n_samples,) o (n_samples, n_outputs), valore target reale.
    :param y_pred_Qtensor: Un QTensor di forma (n_samples,) o (n_samples, n_outputs), valori target stimati.
    :return: restituisce un risultato float.

    Esempio::

            import numpy as np
            from pyvqnet.tensor import tensor
            from pyvqnet.utils import metrics as vqnet_metrics
            from pyvqnet import _core
            _vqnet = _core.vqnet

            y_true_Qtensor = tensor.arange(1, 12)
            y_pred_Qtensor = tensor.arange(4, 15)
            result = vqnet_metrics.MSE(y_true_Qtensor, y_pred_Qtensor)
            print(result)
            # 9.0

            y_true_Qtensor = tensor.arange(1, 13).reshape([3, 4])
            y_pred_Qtensor = tensor.arange(4, 16).reshape([3, 4])
            result = vqnet_metrics.MSE(y_true_Qtensor, y_pred_Qtensor)
            print(result)
            # 9.0


RMSE
=================================

.. py:class:: pyvqnet.utils.metrics.RMSE(y_true_Qtensor, y_pred_Qtensor)

    RMSE: Errore Quadratico Medio della Radice (Root Mean Squared Error).

    :param y_true_Qtensor: Un QTensor di forma (n_samples,) o (n_samples, n_outputs), valore target reale.
    :param y_pred_Qtensor: Un QTensor di forma (n_samples,) o (n_samples, n_outputs), valori target stimati.
    :return: restituisce un risultato float.

    Esempio::

            import numpy as np
            from pyvqnet.tensor import tensor
            from pyvqnet.utils import metrics as vqnet_metrics
            from pyvqnet import _core
            _vqnet = _core.vqnet

            y_true_Qtensor = tensor.arange(1, 12)
            y_pred_Qtensor = tensor.arange(4, 15)
            result = vqnet_metrics.RMSE(y_true_Qtensor, y_pred_Qtensor)
            print(result)
            # 3.0

            y_true_Qtensor = tensor.arange(1, 13).reshape([3, 4])
            y_pred_Qtensor = tensor.arange(4, 16).reshape([3, 4])
            result = vqnet_metrics.RMSE(y_true_Qtensor, y_pred_Qtensor)
            print(result)
            # 3.0



MAE
=================================

.. py:class:: pyvqnet.utils.metrics.MAE(y_true_Qtensor, y_pred_Qtensor)

    MAE: Errore Assoluto Medio (Mean Absolute Error).

    :param y_true_Qtensor: Un QTensor di forma (n_samples,) o (n_samples, n_outputs), valore target reale.
    :param y_pred_Qtensor: Un QTensor di forma (n_samples,) o (n_samples, n_outputs), valori target stimati.
    :return: restituisce un risultato float.

    Esempio::

            import numpy as np
            from pyvqnet.tensor import tensor
            from pyvqnet.utils import metrics as vqnet_metrics
            from pyvqnet import _core
            _vqnet = _core.vqnet

            y_true_Qtensor = tensor.arange(1, 12)
            y_pred_Qtensor = tensor.arange(4, 15)
            result = vqnet_metrics.MAE(y_true_Qtensor, y_pred_Qtensor)
            print(result)
            # 3.0

            y_true_Qtensor = tensor.arange(1, 13).reshape([3, 4])
            y_pred_Qtensor = tensor.arange(4, 16).reshape([3, 4])
            result = vqnet_metrics.MAE(y_true_Qtensor, y_pred_Qtensor)
            print(result)
            # 3.0


R_Square
=================================

.. py:class:: pyvqnet.utils.metrics.R_Square(y_true_Qtensor, y_pred_Qtensor, sample_weight=None)

    R_Square: R^2 (coefficiente di determinazione) funzione di punteggio di regressione.
    Il punteggio migliore possibile e 1.0, che puo essere negativo
    (poiche il modello puo deteriorarsi arbitrariamente).
    Un modello che predice sempre il valore atteso di y,
    ignorando le caratteristiche di input, otterra un punteggio R^2 di 0.0.
    
    :param y_true_Qtensor: Un QTensor di forma (n_samples,) o (n_samples, n_outputs), valore target reale.
    :param y_pred_Qtensor: Un QTensor di forma (n_samples,) o (n_samples, n_outputs), valori target stimati.
    :param sample_weight: Array di forma (n_samples,), peso campione opzionale, default:None.
    :return: restituisce un risultato float.

    Esempio::

            import numpy as np
            from pyvqnet.tensor import tensor
            from pyvqnet.utils import metrics as vqnet_metrics
            from pyvqnet import _core
            _vqnet = _core.vqnet

            y_true_Qtensor = tensor.arange(1, 12)
            y_pred_Qtensor = tensor.arange(4, 15)
            result = vqnet_metrics.R_Square(y_true_Qtensor, y_pred_Qtensor)
            print(result)
            # 0.09999999999999998

            y_true_Qtensor = tensor.arange(1, 13).reshape([3, 4])
            y_pred_Qtensor = tensor.arange(4, 16).reshape([3, 4])
            result = vqnet_metrics.R_Square(y_true_Qtensor, y_pred_Qtensor)
            print(result)
            # 0.15625


precision_recall_f1_2_score
=================================

.. py:class:: pyvqnet.utils.metrics.precision_recall_f1_2_score(y_true_Qtensor, y_pred_Qtensor)

    Calcola la precisione, il richiamo e il punteggio F1 dei valori previsti in un compito di classificazione binaria. I valori previsti e reali devono essere QTensor di forma simile (n_samples,), con valore 0 o 1, che rappresentano le etichette delle due classi.
    
    :param y_true_Qtensor: Un QTensor 1D, valore target reale.
    :param y_pred_Qtensor: Un QTensor 1D, valore target stimato.

    :returns: 
        - precision - risultato di precisione
        - recall - risultato di richiamo
        - f1 - punteggio F1

    Esempio::

            import numpy as np
            from pyvqnet.tensor import tensor
            from pyvqnet.utils import metrics as vqnet_metrics
            from pyvqnet import _core
            _vqnet = _core.vqnet

            y_true_Qtensor = tensor.QTensor([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
            y_pred_Qtensor = tensor.QTensor([0, 0, 1, 1, 1, 0, 0, 1, 1, 1])

            precision, recall, f1 = vqnet_metrics.precision_recall_f1_2_score(
                y_true_Qtensor, y_pred_Qtensor)
            print(precision, recall, f1)
            # 0.5 0.6 0.5454545454545454


precision_recall_f1_N_score
=================================

.. py:class:: pyvqnet.utils.metrics.precision_recall_f1_N_score(y_true_Qtensor, y_pred_Qtensor, N, average)

    Calcoli di precisione, richiamo e punteggio F1 per compiti di classificazione multiclasse. dove il valore previsto e il valore reale sono QTensor di forma simile (n_samples,), e i valori sono interi da 0 a N-1, che rappresentano le etichette di N classi.

    :param y_true_Qtensor: Un QTensor 1D, valore target reale.
    :param y_pred_Qtensor: Un QTensor 1D, valore target stimato.
    :param N: N classi (numero di classi).
    :param average: stringa, ['micro', 'macro', 'weighted'].
             Questo parametro e richiesto per target multiclasse/multi-etichetta.
             
             ``'micro'``: Calcola le metriche globalmente contando il totale dei veri positivi, falsi negativi e falsi positivi.
             
             ``'macro'``: Calcola la metrica per ciascuna etichetta e trova il suo valore non pesato. Significa che il bilanciamento delle etichette non viene considerato.
             
             ``'weighted'``: Calcola le metriche per ciascuna etichetta e trova la loro media (il numero di istanze reali di ciascuna etichetta). Questo modifica ``'macro'`` per tenere conto dello squilibrio delle etichette; questo puo comportare che i punteggi F non siano compresi tra precisione e richiamo.
    
    :returns: 
        - precision - risultato di precisione
        - recall - risultato di richiamo
        - f1 - punteggio F1

    Esempio::

                import numpy as np
                from pyvqnet.tensor import tensor
                from pyvqnet.utils import metrics as vqnet_metrics
                from pyvqnet import _core
                _vqnet = _core.vqnet

                reference_list = [1, 1, 2, 2, 2, 3, 3, 3, 3, 3]
                prediciton_list = [1, 2, 2, 2, 3, 1, 2, 3, 3, 3]
                y_true_Qtensor = tensor.QTensor(reference_list)
                y_pred_Qtensor = tensor.QTensor(prediciton_list)

                precision_micro, recall_micro, f1_micro = vqnet_metrics.precision_recall_f1_N_score(
                    y_true_Qtensor, y_pred_Qtensor, 3, average='micro')
                print(precision_micro, recall_micro, f1_micro)
                # 0.6 0.6 0.6

                precision_macro, recall_macro, f1_macro = vqnet_metrics.precision_recall_f1_N_score(
                    y_true_Qtensor, y_pred_Qtensor, 3, average='macro')
                print(precision_macro, recall_macro, f1_macro)
                # 0.5833333333333334 0.5888888888888889 0.5793650793650794

                precision_weighted, recall_weighted, f1_weighted = vqnet_metrics.precision_recall_f1_N_score(
                    y_true_Qtensor, y_pred_Qtensor, 3, average='weighted')
                print(precision_weighted, recall_weighted, f1_weighted)
                # 0.625 0.6 0.6047619047619047



precision_recall_f1_Multi_score
=================================

.. py:class:: pyvqnet.utils.metrics.precision_recall_f1_Multi_score(y_true_Qtensor, y_pred_Qtensor, N, average)

    Calcoli di precisione, richiamo e punteggio F1 per compiti di classificazione multiclasse. dove i valori previsti e reali sono QTensor di forma simile (n_samples, N), dove i valori sono valori di etichetta codificati one-hot N-dimensionali.

    :param y_true_Qtensor: Un QTensor 1D, valore target reale.
    :param y_pred_Qtensor: Un QTensor 1D, valore target stimato.
    :param N: N classi (numero di classi).
    :param average: stringa, ['micro', 'macro', 'weighted'].
             Questo parametro e richiesto per target multiclasse/multi-etichetta.
             
             ``'micro'``: Calcola le metriche globalmente contando il totale dei veri positivi, falsi negativi e falsi positivi.
             
             ``'macro'``: Calcola la metrica per ciascuna etichetta e trova il suo valore non pesato. Significa che il bilanciamento delle etichette non viene considerato.
             
             ``'weighted'``: Calcola le metriche per ciascuna etichetta e trova la loro media (il numero di istanze reali di ciascuna etichetta). Questo modifica ``'macro'`` per tenere conto dello squilibrio delle etichette; questo puo comportare che i punteggi F non siano compresi tra precisione e richiamo.
    
    :returns: 
        - precision - risultato di precisione
        - recall - risultato di richiamo
        - f1 - punteggio F1

    Esempio::


                    import numpy as np
                    from pyvqnet.tensor import tensor
                    from pyvqnet.utils import metrics as vqnet_metrics
                    from pyvqnet import _core
                    _vqnet = _core.vqnet

                    reference_list = [[1, 0], [0, 1], [0, 0], [1, 1], [1, 0]]
                    prediciton_list = [[1, 0], [0, 0], [1, 0], [0, 0], [0, 0]]
                    y_true_Qtensor = tensor.QTensor(reference_list)
                    y_pred_Qtensor = tensor.QTensor(prediciton_list)

                    micro_precision, micro_recall, micro_f1 = vqnet_metrics.precision_recall_f1_Multi_score(y_true_Qtensor,
                                y_pred_Qtensor, 2, average='micro')
                    print(micro_precision, micro_recall, micro_f1)
                    # 0.5 0.2 0.28571428571428575

                    macro_precision, macro_recall, macro_f1 = vqnet_metrics.precision_recall_f1_Multi_score(y_true_Qtensor,
                                y_pred_Qtensor, 2, average='macro')
                    print(macro_precision, macro_recall, macro_f1)
                    # 0.25 0.16666666666666666 0.2

                    weighted_precision, weighted_recall, weighted_f1 = vqnet_metrics.precision_recall_f1_Multi_score(y_true_Qtensor,
                                y_pred_Qtensor, 2, average='weighted')
                    print(weighted_precision, weighted_recall, weighted_f1)
                    # 0.3 0.19999999999999998 0.24

                    reference_list = [[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0], [1, 0, 1]]
                    prediciton_list = [[1, 0, 0], [1, 0, 0], [1, 1, 1], [1, 0, 0], [0, 1, 1]]
                    y_true_Qtensor = tensor.QTensor(reference_list)
                    y_pred_Qtensor = tensor.QTensor(prediciton_list)

                    micro_precision, micro_recall, micro_f1 = vqnet_metrics.precision_recall_f1_Multi_score(y_true_Qtensor,
                                y_pred_Qtensor, 3, average='micro')
                    print(micro_precision, micro_recall, micro_f1) # 0.5 0.5714285714285714 0.5333333333333333

                    macro_precision, macro_recall, macro_f1 = vqnet_metrics.precision_recall_f1_Multi_score(y_true_Qtensor,
                                y_pred_Qtensor, 3, average='macro')
                    print(macro_precision, macro_recall, macro_f1)
                    # 0.5 0.5555555555555555 0.5238095238095238

                    weighted_precision, weighted_recall, weighted_f1 = vqnet_metrics.precision_recall_f1_Multi_score(y_true_Qtensor,
                                y_pred_Qtensor, 3, average='weighted')
                    print(weighted_precision, weighted_recall, weighted_f1)
                    # 0.5 0.5714285714285714 0.5306122448979592



auc_calculate
=================================

.. py:class:: pyvqnet.utils.metrics.auc_calculate(y_true_Qtensor, y_pred_Qtensor, pos_label=None, sample_weight=None, drop_intermediate=True)

    Calcola la precisione, il richiamo e il punteggio F1 del compito di classificazione.

    :param y_true_Qtensor: Un QTensor di forma [n_samples].
                             Un'etichetta binaria reale. Se l'etichetta non e {1,1} o {0,1}, pos_label deve essere fornito esplicitamente.
    :param y_pred_Qtensor: Un QTensor di forma [n_samples].
                             Punteggio target, che puo essere una stima di probabilita positiva della classe, un valore di confidenza o una misura senza soglia della decisione (restituita da "decision_function" su alcuni classificatori).
    :param pos_label: int o str. L'etichetta della classe positiva. default=None.
                      Quando ``pos_label`` e None, se ``y_true_Qtensor`` e in {-1,1} o {0,1}, ``pos_label`` viene impostato a 1, altrimenti viene sollevato un errore.
    :param sample_weight: array di forma (n_samples,), default=None.
    :param drop_intermediate: booleano, opzionale (default=True).
                     Se eliminare alcune soglie subottimali che non appaiono sulla curva ROC tracciata.
    :return: restituisce un risultato float.

    Esempio::

                import numpy as np
                from pyvqnet.tensor import tensor
                from pyvqnet.utils import metrics as vqnet_metrics
                from pyvqnet import _core
                _vqnet = _core.vqnet

                y = np.array([1, 1, 1, 1, 0, 1, 0, 0, 0, 0])
                pred = np.array([0.9, 0.8, 0.7, 0.6, 0.6, 0.4, 0.4, 0.3, 0.2, 0.1])
                y_Qtensor = tensor.QTensor(y)
                pred_Qtensor = tensor.QTensor(pred)
                result = vqnet_metrics.auc_calculate(y_Qtensor, pred_Qtensor)
                print("auc:", result)
                # 0.92

                y = np.array([1, 1, 1, 1, 1, 0, 0, 1, 1, 1])
                pred = np.array([1, 0, 1, 1, 1, 1, 0, 1, 1, 0])
                y_Qtensor = tensor.QTensor(y)
                pred_Qtensor = tensor.QTensor(pred)
                result = vqnet_metrics.auc_calculate(y_Qtensor, pred_Qtensor)
                print("auc:", result)
                # 0.625

                y = [1, 2, 1, 1, 1, 0, 0, 1, 1, 1]
                pred = [1, 0, 2, 1, 1, 1, 0, 1, 1, 0]
                y_Qtensor = tensor.QTensor(y)
                pred_Qtensor = tensor.QTensor(pred)
                result = vqnet_metrics.auc_calculate(y_Qtensor, pred_Qtensor, pos_label=2)
                print("auc:", result)
                # 0.1111111111111111


Compatibilita Triton
*********************************************************

`triton <https://triton-lang.org/main/index.html>`_ e un linguaggio e compilatore per scrivere kernel GPU efficienti per il deep learning.
Gli utenti scrivono codice Python simile a NumPy, e Triton lo compila in codice GPU efficiente (simile a CUDA ma di livello superiore).
Triton dipende da alcune interfacce PyTorch, VQNet implementa un'interfaccia simile a PyTorch, permettendo l'integrazione con codice scritto in Triton per la propagazione forward e backward del modello.

Installazione di triton:

.. code-block::

    pip install triton

L'esempio seguente e modificato dall'esempio ufficiale di triton: `layer-norm <https://triton-lang.org/main/getting-started/tutorials/05-layer-norm.html>`_.
Questo esempio deve essere eseguito su Linux con una GPU, e richiede l'installazione di triton e pytorch (usato per confrontare la correttezza del calcolo):

.. code-block::

    import torch as real_torch
    import pyvqnet as torch
    import triton
    import triton.language as tl



    DEVICE = torch.DEV_GPU_0

    @triton.jit
    def _layer_norm_fwd_fused(
        X,  # puntatore all'input
        Y,  # puntatore all'output
        W,  # puntatore ai pesi
        B,  # puntatore ai bias
        Mean,  # puntatore alla media
        Rstd,  # puntatore a 1/std
        stride,  # di quanto aumentare il puntatore quando ci si sposta di 1 riga
        N,  # numero di colonne in X
        eps,  # epsilon per evitare la divisione per zero
        BLOCK_SIZE: tl.constexpr,
    ):
        # Mappa l'id del programma alla riga di X e Y che deve calcolare.
        row = tl.program_id(0)
        Y += row * stride
        X += row * stride
        # Calcola la media
        mean = 0
        _mean = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
        for off in range(0, N, BLOCK_SIZE):
            cols = off + tl.arange(0, BLOCK_SIZE)
            a = tl.load(X + cols, mask=cols < N, other=0.).to(tl.float32)
            _mean += a
        mean = tl.sum(_mean, axis=0) / N
        # Calcola la varianza
        _var = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
        for off in range(0, N, BLOCK_SIZE):
            cols = off + tl.arange(0, BLOCK_SIZE)
            x = tl.load(X + cols, mask=cols < N, other=0.).to(tl.float32)
            x = tl.where(cols < N, x - mean, 0.)
            _var += x * x
        var = tl.sum(_var, axis=0) / N
        rstd = 1 / tl.sqrt(var + eps)
        # Scrivi media / rstd
        tl.store(Mean + row, mean)
        tl.store(Rstd + row, rstd)
        # Normalizza e applica la trasformazione lineare
        for off in range(0, N, BLOCK_SIZE):
            cols = off + tl.arange(0, BLOCK_SIZE)
            mask = cols < N
            w = tl.load(W + cols, mask=mask)
            b = tl.load(B + cols, mask=mask)
            x = tl.load(X + cols, mask=mask, other=0.).to(tl.float32)
            x_hat = (x - mean) * rstd
            y = x_hat * w + b
            # Scrivi l'output
            tl.store(Y + cols, y, mask=mask)


    @triton.jit
    def _layer_norm_bwd_dx_fused(DX,  # puntatore al gradiente di input
                                DY,  # puntatore al gradiente di output
                                DW,  # puntatore alla somma parziale del gradiente dei pesi
                                DB,  # puntatore alla somma parziale del gradiente dei bias
                                X,  # puntatore all'input
                                W,  # puntatore ai pesi
                                Mean,  # puntatore alla media
                                Rstd,  # puntatore a 1/std
                                Lock,  # puntatore al lock
                                stride,  # di quanto aumentare il puntatore quando ci si sposta di 1 riga
                                N,  # numero di colonne in X
                                GROUP_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr):
        # Mappa l'id del programma agli elementi di X, DX, e DY che deve calcolare.
        row = tl.program_id(0)
        cols = tl.arange(0, BLOCK_SIZE_N)
        mask = cols < N
        X += row * stride
        DY += row * stride
        DX += row * stride
        # Offset dei lock e dei puntatori dei gradienti pesi/bias per la riduzione parallela
        lock_id = row % GROUP_SIZE_M
        Lock += lock_id
        Count = Lock + GROUP_SIZE_M
        DW = DW + lock_id * N + cols
        DB = DB + lock_id * N + cols
        # Carica i dati in SRAM

        x = tl.load(X + cols, mask=mask, other=0).to(tl.float32)
        dy = tl.load(DY + cols, mask=mask, other=0).to(tl.float32)
        w = tl.load(W + cols, mask=mask).to(tl.float32)
        mean = tl.load(Mean + row)
        rstd = tl.load(Rstd + row)
        # Calcola dx
        xhat = (x - mean) * rstd
        wdy = w * dy
        xhat = tl.where(mask, xhat, 0.)
        wdy = tl.where(mask, wdy, 0.)
        c1 = tl.sum(xhat * wdy, axis=0) / N
        c2 = tl.sum(wdy, axis=0) / N
        dx = (wdy - (xhat * c1 + c2)) * rstd
        # Scrivi dx
        tl.store(DX + cols, dx, mask=mask)

        # Accumula somme parziali per dw/db
        partial_dw = (dy * xhat).to(w.dtype)
        partial_db = (dy).to(w.dtype)
        while tl.atomic_cas(Lock, 0, 1) == 1:
            pass
        count = tl.load(Count)
        # Il primo store non accumula
        if count == 0:
            tl.atomic_xchg(Count, 1)
        else:
            partial_dw += tl.load(DW, mask=mask)
            partial_db += tl.load(DB, mask=mask)
        tl.store(DW, partial_dw, mask=mask)
        tl.store(DB, partial_db, mask=mask)

        # necessaria una barriera per garantire che tutti i thread abbiano terminato
        # prima di rilasciare il lock
        tl.debug_barrier()

        # Rilascia il lock
        tl.atomic_xchg(Lock, 0)


    @triton.jit
    def _layer_norm_bwd_dwdb(DW,  # puntatore alla somma parziale del gradiente dei pesi
                            DB,  # puntatore alla somma parziale del gradiente dei bias
                            FINAL_DW,  # puntatore al gradiente dei pesi
                            FINAL_DB,  # puntatore al gradiente dei bias
                            M,  # GROUP_SIZE_M
                            N,  # numero di colonne
                            BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr):
        # Mappa l'id del programma agli elementi di DW e DB che deve calcolare.
        pid = tl.program_id(0)
        cols = pid * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
        dw = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
        db = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
        # Itera attraverso le righe di DW e DB per sommare le somme parziali.
        for i in range(0, M, BLOCK_SIZE_M):
            rows = i + tl.arange(0, BLOCK_SIZE_M)
            mask = (rows[:, None] < M) & (cols[None, :] < N)
            offs = rows[:, None] * N + cols[None, :]
            dw += tl.load(DW + offs, mask=mask, other=0.)
            db += tl.load(DB + offs, mask=mask, other=0.)
        # Scrivi la somma finale nell'output.
        sum_dw = tl.sum(dw, axis=0)
        sum_db = tl.sum(db, axis=0)
        tl.store(FINAL_DW + cols, sum_dw, mask=cols < N)
        tl.store(FINAL_DB + cols, sum_db, mask=cols < N)


    class LayerNorm(torch.autograd.Function):

        @staticmethod
        def forward(ctx, x, normalized_shape, weight, bias, eps):
            # alloca l'output
            y = torch.empty_like(x)
            # rimodella i dati di input in un tensore 2D
            x_arg = x.reshape([-1, x.shape[-1]])
            M, N = x_arg.shape
            mean = torch.empty((M, ), dtype=torch.float32, device=x.device).data
            rstd = torch.empty((M, ), dtype=torch.float32, device=x.device).data
            # Meno di 64KB per feature: accoda il kernel fuso
            MAX_FUSED_SIZE = 65536 // x.element_size()
            BLOCK_SIZE = min(MAX_FUSED_SIZE, triton.next_power_of_2(N))
            if N > BLOCK_SIZE:
                raise RuntimeError("This layer norm doesn't support feature dim >= 64KB.");
            # euristiche per il numero di warp
            num_warps = min(max(BLOCK_SIZE // 256, 1), 8)
            # accoda il kernel
            _layer_norm_fwd_fused[(M, )](  #
                x_arg, y, weight, bias, mean, rstd,  #
                x_arg.stride[0], N, eps,  #
                BLOCK_SIZE=BLOCK_SIZE, num_warps=num_warps, num_ctas=1)
            ctx.save_for_backward(x, weight, bias, mean, rstd)
            ctx.BLOCK_SIZE = BLOCK_SIZE
            ctx.num_warps = num_warps
            ctx.eps = eps
            return y.data

        @staticmethod
        def backward(ctx, dy):
            x, w, b, m, v = ctx.saved_tensors
            # euristiche per la quantita di flusso di riduzione parallela per DW/DB
            N = w.shape[0]
            GROUP_SIZE_M = 64
            if N <= 8192: GROUP_SIZE_M = 96
            if N <= 4096: GROUP_SIZE_M = 128
            if N <= 1024: GROUP_SIZE_M = 256
            # alloca l'output
            locks = torch.zeros(2 * GROUP_SIZE_M, dtype=torch.int32, device=w.device).data
            _dw = torch.zeros((GROUP_SIZE_M, N), dtype=x.dtype, device=w.device).data
            _db = torch.zeros((GROUP_SIZE_M, N), dtype=x.dtype, device=w.device).data
            dw = torch.empty((N, ), dtype=w.dtype, device=w.device).data
            db = torch.empty((N, ), dtype=w.dtype, device=w.device).data
            dx = torch.empty_like(dy).data
            # accoda il kernel usando le euristiche del forward
            # calcola anche le somme parziali per DW e DB
            x_arg = x.reshape([-1, x.shape[-1]])
            M, N = x_arg.shape

            _layer_norm_bwd_dx_fused[(M, )](  #
                dx, dy, _dw, _db, x, w, m, v, locks,  #
                x_arg.stride[0], N,  #
                BLOCK_SIZE_N=ctx.BLOCK_SIZE,  #
                GROUP_SIZE_M=GROUP_SIZE_M,  #
                num_warps=ctx.num_warps)

            grid = lambda meta: (triton.cdiv(N, meta['BLOCK_SIZE_N']), )

            # accumula le somme parziali in un kernel separato
            _layer_norm_bwd_dwdb[grid](
                _dw, _db, dw, db, min(GROUP_SIZE_M, M), N,  #
                BLOCK_SIZE_M=32,  #
                BLOCK_SIZE_N=128, num_ctas=1)
            return dx, None, dw, db, None

    preprocess = LayerNorm.preprocess
    layer_norm = LayerNorm.apply


    def test_layer_norm(M, N, dtype, eps=1e-5, device=DEVICE):
        # crea i dati
        x_shape = (M, N)
        w_shape = (x_shape[-1], )
        weight = torch.rand(w_shape, dtype=dtype, device=device, requires_grad=True)
        bias = torch.rand(w_shape, dtype=dtype, device=device, requires_grad=True)
        x = -2.3 + 0.5 * torch.randn(x_shape, dtype=dtype, device=device)

        dy = .1 * torch.randn_like(x)
        x.requires_grad = True
        # forward pass
        xc, weightc, biasc = preprocess(x, weight, bias)
        y_tri = layer_norm(xc, w_shape, weightc, biasc, eps)
        y_tri = torch.to_tensor(y_tri)
        numpy_w = weight.detach().cpu().numpy()
        numpy_b = bias.detach().cpu().numpy()
        numpy_x = x.detach().cpu().numpy()
        numpy_dy = dy.detach().cpu().numpy()
        torch_weight = real_torch.tensor(numpy_w,device="cuda:1",requires_grad= True)
        torch_bias = real_torch.tensor(numpy_b,device="cuda:1",requires_grad= True)
        torch_x = real_torch.tensor(numpy_x,device="cuda:1",requires_grad= True)
        torch_dy = real_torch.tensor(numpy_dy,device="cuda:1",requires_grad= True)
        y_ref = real_torch.nn.functional.layer_norm(torch_x, w_shape, torch_weight, torch_bias, eps)
        # backward pass (triton)
        y_tri.backward(dy, retain_graph=True)
        dx_tri, dw_tri, db_tri = [_.grad.detach().cpu().numpy() for _ in [x, weight, bias]]
        x.grad, weight.grad, bias.grad = None, None, None
        # backward pass (torch)
        y_ref.backward(torch_dy, retain_graph=True)
        dx_ref, dw_ref, db_ref = [_.grad.detach().cpu().numpy() for _ in [torch_x, torch_weight, torch_bias]]
        # confronto
        import numpy as np
        assert np.allclose(y_tri.detach().cpu().numpy(), y_ref.detach().cpu().numpy(), atol=1e-2, rtol=0)
        assert np.allclose(dx_tri, dx_ref, atol=1e-2, rtol=0)
        assert np.allclose(db_tri, db_ref, atol=1e-2, rtol=0)
        assert np.allclose(dw_tri, dw_ref, atol=1e-2, rtol=0)
        print("same")
    test_layer_norm(12, 32, torch.float32)
