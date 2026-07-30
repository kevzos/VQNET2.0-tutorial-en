Klassisches Neuronales Netzwerk-Modul
#########################################

Die folgenden klassischen neuronalen Netzwerkmodelle unterstützen die automatische Rückwärtspropagation. Nach Ausführung der Vorwärtsfunktion kann der Gradient durch Aufruf der Rückwärtsfunktion berechnet werden. Ein einfaches Beispiel für die Faltungsschicht ist wie folgt:

.. code-block::

    from pyvqnet.tensor import arange
    from pyvqnet import kfloat32
    from pyvqnet.nn import Conv2D

    # ein Bild wird in eine zweidimensionale Faltungsschicht eingespeist
    b = 2        # Batch-Größe
    ic = 2       # Eingangskanäle
    oc = 2      # Ausgangskanäle
    hw = 4      # Eingangsbreite und -höhe

    # zweidimensionale Faltungsschicht
    test_conv = Conv2D(ic,oc,(2,2),(2,2),"same")

    # Eingabe der Form [b,ic,hw,hw]
    x0 = arange(1,b*ic*hw*hw+1,requires_grad=True,dtype=kfloat32)

    x1 = x0.reshape([b,ic,hw,hw])
    # Vorwärtsfunktion
    x = test_conv(x1)

    # Rückwärtsfunktion mit Autograd
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


Modul-Klasse
********************************************************

abstraktes Berechnungsmodul


Module
=================================

.. py:class:: pyvqnet.nn.module.Module

    Basisklasse für alle neuronalen Netzwerkmodelle, einschließlich Quantenmodule oder klassische Module.
    Ihre Modelle sollten ebenfalls eine Unterklasse dieser Klasse für die Autograd-Berechnung sein.

    Module können auch andere Module enthalten, was eine Verschachtelung in
    einer Baumstruktur ermöglicht. Sie können die Untermodule als reguläre Attribute zuweisen::

        class Model(Module):
            def __init__(self):
                super(Model, self).__init__()
                self.conv1 = pyvqnet.nn.Conv2d(1, 20, (5,5))
                self.conv2 = pyvqnet.nn.Conv2d(20, 20, (5,5))

            def forward(self, x):
                x = pyvqnet.nn.activation.relu(self.conv1(x))
                return pyvqnet.nn.activation.relu(self.conv2(x))

    Auf diese Weise zugewiesene Untermodule werden registriert.

forward
=================================

.. py:method:: pyvqnet.nn.module.Module.forward(x, *args, **kwargs)

    Abstrakte Methode, die den Vorwärtsdurchlauf ausführt.

    :param x: Eingabe-QTensor
    :param \*args: Ein parameterloser Variablenparameter
    :param \*\*kwargs: Ein Schlüsselwort-Variablenparameter
    :return: Modulausgabe

    Example::

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

    Gibt ein Wörterbuch zurück, das den gesamten Zustand des Moduls enthält.

    Sowohl Parameter als auch persistente Puffer (z.B. laufende Durchschnitte) sind
    enthalten. Die Schlüssel sind die entsprechenden Parameter- und Puffernamen.

    :param destination: Ein Dictionary, in dem der Zustand gespeichert wird
    :param prefix: Das Präfix für Parameter und Puffer, die in diesem
        Modul verwendet werden

    :return: Ein Dictionary, das den gesamten Zustand des Moduls enthält

    Example::

        from pyvqnet.nn import Conv2D
        test_conv = Conv2D(2,3,(3,3),(2,2),"same")
        print(test_conv.state_dict().keys())
        #odict_keys(['weights', 'bias'])


toGPU
=================================

.. py:function:: pyvqnet.nn.module.Module.toGPU(device: int = DEV_GPU_0)

    Verschiebt die Parameter- und Pufferdaten eines Moduls und seiner Untermodule auf das angegebene GPU-Gerät.

    device gibt das Gerät an, auf dem die internen Daten gespeichert werden. Wenn device >= DEV_GPU_0, werden die Daten auf der GPU gespeichert. Wenn Ihr Computer mehrere GPUs hat,
    können Sie verschiedene Geräte zur Datenspeicherung angeben. Zum Beispiel bedeutet device = DEV_GPU_1, DEV_GPU_2, DEV_GPU_3, ... die Speicherung auf GPUs mit unterschiedlichen Seriennummern.
    
    .. note::
        Module können nicht auf verschiedenen GPUs berechnet werden. Ein Cuda-Fehler wird ausgelöst, wenn Sie versuchen, einen QTensor auf einer GPU zu erstellen, deren ID die maximale Anzahl verifizierter GPUs überschreitet.

    :param device: Das Gerät, das aktuell den QTensor speichert, Standard=DEV_GPU_0. device = pyvqnet.DEV_GPU_0, gespeichert in der ersten GPU, device = DEV_GPU_1, gespeichert in der zweiten GPU, und so weiter.
    :return: Modul, das auf das GPU-Gerät verschoben wurde.

    Examples::

        from pyvqnet.nn.conv import ConvT2D 
        test_conv = ConvT2D(3, 2, [4,4], [2, 2], "same")
        test_conv = test_conv.toGPU()
        print(test_conv.backend)
        #1000


toCPU
=================================

.. py:function:: pyvqnet.nn.module.Module.toCPU()

    Verschiebt die Parameter- und Pufferdaten eines Moduls und seiner Untermodule auf ein bestimmtes CPU-Gerät.

    :return: Modul, das auf das CPU-Gerät verschoben wurde.

    Examples::

        from pyvqnet.nn.conv import ConvT2D 
        test_conv = ConvT2D(3, 2, [4,4], [2, 2], "same")
        test_conv = test_conv.toCPU()
        print(test_conv.backend)
        #0

.. _save_parameters:

save_parameters
=================================

.. py:function:: pyvqnet.utils.storage.save_parameters(obj, f)

    Speichert Modellparameter in einer Datei auf der Festplatte.

    :param obj: Gespeichertes OrderedDict von ``state_dict()``
    :param f: Ein String oder os.PathLike-Objekt, das einen Dateinamen enthält
    :return: None

    Example::

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

    Lädt Modellparameter aus einer Datei auf der Festplatte.

    Die Modellinstanz sollte zuerst erstellt werden.

    :param f: Ein String oder os.PathLike-Objekt, das einen Dateinamen enthält
    :return: Gespeichertes OrderedDict für ``load_state_dict()``

    Example::

        from pyvqnet.nn import Module,Conv2D
        import pyvqnet

        class Net(Module):
            def __init__(self):
                super(Net, self).__init__()
                self.conv1 = Conv2D(input_channels=1, output_channels=6, kernel_size=(5, 5), stride=(1, 1), padding="valid")

            def forward(self, x):
                return super().forward(x)

        model = Net()
        model1 = Net()  # another Module object
        pyvqnet.utils.storage.save_parameters( model.state_dict(),"tmp.model")
        model_para =  pyvqnet.utils.storage.load_parameters("tmp.model")
        model1.load_state_dict(model_para)



ModuleList
**************************************************************************************************************************************************************************

.. py:class:: pyvqnet.nn.module.ModuleList([pyvqnet.nn.module.Module])


    Speichert Untermodule in einer Liste. ModuleList kann wie eine normale Python-Liste indexiert werden, und die internen Parameter des darin enthaltenen Moduls können gespeichert werden.

    :param modules: Liste von nn.Module

    :return: Eine Liste von Modulen

    Example::

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


    Speichert Parameter in einer Liste. ParameterList kann wie eine normale Python-Liste indexiert werden, und die internen Parameter des darin enthaltenen Parameters können gespeichert werden.

    :param modules: nn.Parameter-Liste.

    :return: Eine Parameter-Liste.

    Example::

        from pyvqnet import nn
        class MyModule(nn.Module):
            def __init__(self):
                super().__init__()
                self.params = nn.ParameterList([nn.Parameter((10, 10)) for i in range(10)])
            def forward(self, x):

                # ParameterList kann als Iterable verwendet oder mit Integer-Indizes angesprochen werden
                for i, p in enumerate(self.params):
                    x = self.params[i // 2] * x + p * x
                return x

        model = MyModule()
        print(model.state_dict().keys())


Sequential
*********************************************************
.. py:class:: pyvqnet.nn.module.Sequential([pyvqnet.nn.module.Module])

    Module werden in der Reihenfolge hinzugefügt, in der sie übergeben werden. Alternativ kann ein ``OrderedDict`` von Modulen übergeben werden. Die ``forward()``-Methode von ``Sequential`` nimmt eine beliebige Eingabe und leitet sie an ihr erstes Modul weiter.
    Sie leitet dann die Ausgabe des ersten Moduls als Eingabe an das nächste Modul weiter und gibt schließlich die Ausgabe des letzten Moduls zurück.

    :param modules: Anzuhängendes Modul.

    :return: Sequential.

    Example::
        
        from pyvqnet import nn
        from collections import OrderedDict

        # Verwenden von Sequential zur Erstellung eines kleinen Modells.
        model = nn.Sequential(
                  nn.Conv2D(1,20,(5, 5)),
                  nn.ReLu(),
                  nn.Conv2D(20,64,(5, 5)),
                  nn.ReLu()
                )
        print(model.state_dict().keys())

        # Verwenden von Sequential mit OrderedDict. Dies ist funktional identisch zum obigen Code.
                
        model = nn.Sequential(OrderedDict([
                  ('conv1', nn.Conv2D(1,20,(5, 5))),
                  ('relu1', nn.ReLu()),
                  ('conv2', nn.Conv2D(20,64,(5, 5))),
                  ('relu2', nn.ReLu())
                ]))
        print(model.state_dict().keys())


Klassische Neuronale Netzwerk-Schicht
********************************************************

Conv1D
=================================

.. py:class:: pyvqnet.nn.Conv1D(input_channels:int,output_channels:int,kernel_size:int ,stride:int= 1,padding = "valid",use_bias:str = True,kernel_initializer = None,bias_initializer =None, dilation_rate: int = 1, group: int = 1, dtype=None, name='')

    Wendet einen 1-dimensionalen Faltungskernel auf eine Eingabe an. Die Eingaben für das Faltungsmodul haben die Form (batch_size, input_channels, height)

    :param input_channels: `int` - Anzahl der Eingabekanäle
    :param output_channels: `int` - Anzahl der Kernel
    :param kernel_size: `int` - Größe eines einzelnen Kernels. Kernel-Form = [output_channels, input_channels/group, kernel_size, 1]
    :param stride: `int` - Schrittweite, Standardwert ist 1
    :param padding: `str|int` - Padding-Option, kann ein String {'valid', 'same'} oder eine ganze Zahl sein, die den Betrag des impliziten Paddings angibt. Standard "valid".
    :param use_bias: `bool` - ob Bias verwendet wird, Standardwert ist True
    :param kernel_initializer: `callable` - Standardwert ist None
    :param bias_initializer: `callable` - Standardwert ist None
    :param dilation_rate: `int` - Dilatationsgröße, Standard: 1
    :param group: `int` - Anzahl der Gruppen für gruppierte Faltungen. Standard: 1
    :param dtype: Der Datentyp des Parameters, Standard: None, verwendet den Standard-Datentyp kfloat32, der eine 32-Bit-Gleitkommazahl darstellt.
    :param name: Der Name des Moduls, Standard: "".
    :return: Eine Conv1D-Klasse

    .. note::
        ``padding='valid'`` entspricht keiner Auffüllung.

        ``padding='same'`` füllt die Eingabe so auf, dass die Ausgabe die gleiche Form wie die Eingabe hat.

    Example::

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

    Wendet einen zweidimensionalen Faltungskernel auf eine Eingabe an. Die Eingaben für das Faltungsmodul haben die Form (batch_size, input_channels, height, width)

    :param input_channels: `int` - Anzahl der Eingabekanäle
    :param output_channels: `int` - Anzahl der Kernel
    :param kernel_size: `tuple|list` - Größe eines einzelnen Kernels. Kernel-Form = [output_channels, input_channels/group, kernel_size, kernel_size]
    :param stride: `tuple|list` - Schrittweite, Standardwert ist (1, 1)|[1,1]
    :param padding: `str|tuple` - Padding-Option, kann ein String {'valid', 'same'} oder ein Tupel von ganzen Zahlen sein, die den Betrag des impliziten Paddings auf beiden Seiten angeben. Standard "valid".
    :param use_bias: `bool` - ob Bias verwendet wird, Standardwert ist True
    :param kernel_initializer: `callable` - Standardwert ist None
    :param bias_initializer: `callable` - Standardwert ist None
    :param dilation_rate: `int` - Dilatationsgröße, Standard: 1
    :param group: `int` - Anzahl der Gruppen für gruppierte Faltungen. Standard: 1.
    :param dtype: Der Datentyp des Parameters, Standard: None, verwendet den Standard-Datentyp kfloat32, der eine 32-Bit-Gleitkommazahl darstellt.
    :param name: Der Name des Moduls, Standard: "".

    :return: Eine Conv2D-Klasse

    .. note::
        ``padding='valid'`` entspricht keiner Auffüllung.

        ``padding='same'`` füllt die Eingabe so auf, dass die Ausgabe die gleiche Form wie die Eingabe hat.

    Example::

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

    Wendet einen zweidimensionalen transponierten Faltungskernel auf eine Eingabe an. Die Eingaben für das ConvT-Modul haben die Form (batch_size, input_channels, height, width)

    :param input_channels: `int` - Anzahl der Eingabekanäle
    :param output_channels: `int` - Anzahl der Kernel
    :param kernel_size: `tuple|list` - Größe eines einzelnen Kernels. Kernel-Form = [input_channels, output_channels/group, kernel_size, kernel_size]
    :param stride: `tuple|list` - Schrittweite, Standardwert ist (1, 1)|[1,1]
    :param padding: `str|tuple` - Padding-Option, kann ein String {'valid', 'same'} oder ein Tupel von ganzen Zahlen sein, die den Betrag des impliziten Paddings auf beiden Seiten angeben. Standard "valid".
    :param use_bias: `bool` - Ob ein Offset-Element verwendet werden soll. Standardmäßig verwenden.
    :param kernel_initializer: `callable` - Standardwert ist None
    :param bias_initializer: `callable` - Standardwert ist None
    :param dilation_rate: `int` - Dilatationsgröße, Standard: 1.
    :param out_padding: Zusätzliche Größe, die auf einer Seite jeder Dimension der Ausgabeform hinzugefügt wird. Standard: (0,0) 
    :param group: `int` - Anzahl der Gruppen für gruppierte Faltungen. Standard: 1.
    :param dtype: Der Datentyp des Parameters, Standard: None, verwendet den Standard-Datentyp kfloat32, der eine 32-Bit-Gleitkommazahl darstellt.
    :param name: Der Name des Moduls, Standard: "".

    :return: Eine ConvT2D-Klasse

    .. note::
        ``padding='valid'`` entspricht keiner Auffüllung.

        ``padding='same'`` füllt die Eingabe so auf, dass die Ausgabe die gleiche Form wie die Eingabe hat.


    Example::

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

    Diese Operation wendet einen 1D-Durchschnittspooling auf ein Eingangssignal an, das aus mehreren Eingabeebenen besteht.

    :param kernel: Größe der Durchschnittspooling-Fenster
    :param strides: Faktor, um den herunterskaliert wird
    :param padding: Einer von "valid", "same" oder eine ganze Zahl, die den Padding-Wert angibt, Standardwert ist "valid"
    :param name: Name der Ausgabeschicht.

    :return: AvgPool1D-Schicht

    .. note::
        ``padding='valid'`` entspricht keiner Auffüllung.

        ``padding='same'`` füllt die Eingabe so auf, dass die Ausgabe die gleiche Form wie die Eingabe hat.



    Example::

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

    Diese Operation wendet ein 1D-Max-Pooling auf ein Eingangssignal an, das aus mehreren Eingabeebenen besteht.

    :param kernel: Größe der Max-Pooling-Fenster
    :param strides: Faktor, um den herunterskaliert wird
    :param padding: Einer von "valid", "same" oder eine ganze Zahl, die den Padding-Wert angibt, Standardwert ist "valid"
    :param name: Der Name des Moduls, Standard: "".

    :return: MaxPool1D-Schicht

    .. note::

        ``padding='valid'`` entspricht keiner Auffüllung.

        ``padding='same'`` füllt die Eingabe so auf, dass die Ausgabe die gleiche Form wie die Eingabe hat.


    Example::

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

    Diese Operation wendet 2D-Durchschnittspooling auf Eingabemerkmale an.

    :param kernel: Größe der Durchschnittspooling-Fenster
    :param strides: Faktoren, um die herunterskaliert wird
    :param padding: Einer von "valid", "same" oder ein Tupel mit ganzen Zahlen, die den Padding-Wert für Spalte und Zeile angeben, Standardwert ist "valid"
    :param name: Name der Ausgabeschicht
    :return: AvgPool2D-Schicht

    .. note::
        ``padding='valid'`` entspricht keiner Auffüllung.

        ``padding='same'`` füllt die Eingabe so auf, dass die Ausgabe die gleiche Form wie die Eingabe hat.


    Example::

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

    Diese Operation wendet 2D-Max-Pooling auf Eingabemerkmale an.

    :param kernel: Größe der Max-Pooling-Fenster
    :param strides: Faktor, um den herunterskaliert wird
    :param padding: Einer von "valid", "same" oder ein Tupel mit ganzen Zahlen, die den Padding-Wert für Spalte und Zeile angeben, Standardwert ist "valid"
    :param name: Name der Ausgabeschicht
    :return: MaxPool2D-Schicht

    .. note::
        ``padding='valid'`` entspricht keiner Auffüllung.

        ``padding='same'`` füllt die Eingabe so auf, dass die Ausgabe die gleiche Form wie die Eingabe hat.


    Example::

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

    Dieses Modul wird häufig verwendet, um Worteinbettungen zu speichern und sie über Indizes abzurufen.
    Die Eingabe für das Modul ist eine Liste von Indizes, und die Ausgabe sind die entsprechenden
    Worteinbettungen.

    :param num_embeddings: `int` - Größe des Einbettungswörterbuchs.
    :param embedding_dim: `int` - Die Größe jedes Einbettungsvektors.
    :param weight_initializer: `callable` - Standardwert ist normal.
    :param dtype: Der Datentyp des Parameters, Standard: None, verwendet den Standard-Datentyp kfloat32, der eine 32-Bit-Gleitkommazahl darstellt.
    :param name: Name der Ausgabeschicht.

    :return: Eine Embedding-Klasse

    Example::

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

    Wendet Batch-Normalisierung auf eine 4D-Eingabe (B,C,H,W) an, wie in dem Paper
    `Batch Normalization: Accelerating Deep Network Training by Reducing
    Internal Covariate Shift <https://arxiv.org/abs/1502.03167>`__ beschrieben.

    .. math::

        y = \frac{x - \mathrm{E}[x]}{\sqrt{\mathrm{Var}[x] + \epsilon}} * \gamma + \beta

    wobei :math:`\gamma` und :math:`\beta` lernbare Parameter sind. Standardmäßig behält diese Schicht während des Trainings laufende
    Schätzungen ihres berechneten Mittelwerts und ihrer Varianz bei, die dann während der Evaluierung zur Normalisierung verwendet werden.
    Die laufenden Schätzungen werden mit einem Standard-Momentum von 0,1 beibehalten.

    :param channel_num: `int` - Die Anzahl der Eingabemerkmal-Kanäle.
    :param momentum: `float` - Momentum bei der Berechnung des exponentiell gewichteten Durchschnitts, Standardwert ist 0,1.
    :param epsilon: `float` - Konstante für numerische Stabilität, Standardwert ist 1e-5.
    :param affine: Ein boolescher Wert, der, wenn auf ``True`` gesetzt, diesem Modul lernbare kanalweise affine Parameter zuweist, initialisiert mit 1 (für Gewichte) und 0 (für Biases). Standard: ``True``.
    :param beta_initializer: `callable` - Standardwert ist zeros.
    :param gamma_initializer: `callable` - Standardwert ist ones.
    :param dtype: Der Datentyp des Parameters, Standard: None, verwendet den Standard-Datentyp kfloat32, der eine 32-Bit-Gleitkommazahl darstellt.
    :param name: Name der Ausgabeschicht
    :return: Eine BatchNorm2d-Klasse

    Example::

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

    Wendet Batch-Normalisierung auf eine 2D-Eingabe (B,C) an, wie in dem Paper
    `Batch Normalization: Accelerating Deep Network Training by Reducing
    Internal Covariate Shift <https://arxiv.org/abs/1502.03167>`__ beschrieben.

    .. math::

        y = \frac{x - \mathrm{E}[x]}{\sqrt{\mathrm{Var}[x] + \epsilon}} * \gamma + \beta

    wobei :math:`\gamma` und :math:`\beta` lernbare Parameter sind. Standardmäßig behält diese Schicht während des Trainings laufende
    Schätzungen ihres berechneten Mittelwerts und ihrer Varianz bei, die dann während der Evaluierung zur Normalisierung verwendet werden.
    Die laufenden Schätzungen werden mit einem Standard-Momentum von 0,1 beibehalten.


    :param channel_num: `int` - Die Anzahl der Eingabemerkmal-Kanäle.
    :param momentum: `float` - Momentum bei der Berechnung des exponentiell gewichteten Durchschnitts, Standardwert ist 0,1
    :param epsilon: `float` - Konstante für numerische Stabilität, Standardwert ist 1e-5.
    :param affine: Ein boolescher Wert, der, wenn auf ``True`` gesetzt, diesem Modul lernbare kanalweise affine Parameter zuweist, initialisiert mit 1 (für Gewichte) und 0 (für Biases). Standard: ``True``.
    :param beta_initializer: `callable` - Standardwert ist zeros.
    :param gamma_initializer: `callable` - Standardwert ist ones.
    :param dtype: Der Datentyp des Parameters, Standard: None, verwendet den Standard-Datentyp kfloat32, der eine 32-Bit-Gleitkommazahl darstellt.
    :param name: Name der Ausgabeschicht
    :return: Eine BatchNorm1d-Klasse

    Example::

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

    Die Layer-Normalisierung wird auf die letzten Dimensionen einer beliebigen Eingabe angewendet. Die spezifische Methode ist wie in dem Paper beschrieben:
    `Layer Normalization <https://arxiv.org/abs/1607.06450>`__.

    .. math::
        y = \frac{x - \mathrm{E}[x]}{ \sqrt{\mathrm{Var}[x] + \epsilon}} * \gamma + \beta

    Für Eingaben wie (B,C,H,W,D) kann ``norm_shape`` [C,H,W,D], [H,W,D], [W,D] oder [D] sein.

    :param norm_shape: `float` - Die zu standardisierende Form.
    :param epsilon: `float` - Konstante für numerische Stabilität, Standardwert ist 1e-5.
    :param affine: Ein boolescher Wert, der, wenn auf ``True`` gesetzt, diesem Modul lernbare kanalweise affine Parameter zuweist, initialisiert mit 1 (für Gewichte) und 0 (für Biases). Standard: ``True``.
    :param dtype: Der Datentyp des Parameters, Standard: None, verwendet den Standard-Datentyp kfloat32, der eine 32-Bit-Gleitkommazahl darstellt.
    :param name: Name der Ausgabeschicht.

    :return: Eine LayerNormNd-Klasse.

    Example::

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

    Wendet Layer-Normalisierung auf einen Mini-Batch von 4D-Eingaben an, wie in
    dem Paper `Layer Normalization <https://arxiv.org/abs/1607.06450>`__ beschrieben.

    .. math::
        y = \frac{x - \mathrm{E}[x]}{ \sqrt{\mathrm{Var}[x] + \epsilon}} * \gamma + \beta

    Der Mittelwert und die Standardabweichung werden über die Größe der letzten `D` Dimensionen berechnet.

    Für Eingaben wie (B,C,H,W) sollte ``norm_size`` gleich C * H * W sein.

    :param norm_size: `float` - Normalisierungsgröße, gleich C * H * W
    :param epsilon: `float` - Konstante für numerische Stabilität, Standardwert ist 1e-5
    :param affine: Ein boolescher Wert, der, wenn auf ``True`` gesetzt, diesem Modul lernbare kanalweise affine Parameter zuweist, initialisiert mit 1 (für Gewichte) und 0 (für Biases). Standard: ``True``.
    :param name: Name der Ausgabeschicht
    :param dtype: Der Datentyp des Parameters, Standard: None, verwendet den Standard-Datentyp kfloat32, der eine 32-Bit-Gleitkommazahl darstellt.
    
    :return: Eine LayerNorm2d-Klasse

    Example::

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

    Wendet Layer-Normalisierung auf einen Mini-Batch von 2D-Eingaben an, wie in
    dem Paper `Layer Normalization <https://arxiv.org/abs/1607.06450>`__ beschrieben.

    .. math::
        y = \frac{x - \mathrm{E}[x]}{ \sqrt{\mathrm{Var}[x] + \epsilon}} * \gamma + \beta

    Der Mittelwert und die Standardabweichung werden über die Größe der letzten Dimensionen berechnet, wobei ``norm_size``
    der Wert der letzten Dimensionsgröße ist.

    :param norm_size: `float` - Normalisierungsgröße, gleich der letzten Dimension
    :param epsilon: `float` - Konstante für numerische Stabilität, Standardwert ist 1e-5
    :param affine: Ein boolescher Wert, der, wenn auf ``True`` gesetzt, diesem Modul lernbare kanalweise affine Parameter zuweist, initialisiert mit 1 (für Gewichte) und 0 (für Biases). Standard: ``True``.
    :param dtype: Der Datentyp des Parameters, Standard: None, verwendet den Standard-Datentyp kfloat32, der eine 32-Bit-Gleitkommazahl darstellt.
    :param name: Name der Ausgabeschicht

    :return: Eine LayerNorm1d-Klasse

    Example::

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

    Wendet Gruppen-Normalisierung auf einen Mini-Batch von Eingaben an. Eingabe: :math:`(N, C, *)` wobei :math:`C=\mathrm{num\_channels}` , Ausgabe: :math:`(N, C, *)` .

    Diese Schicht implementiert die in dem Paper `Group Normalization <https://arxiv.org/abs/1803.08494>`__ beschriebene Operation.

    .. math::

        y = \frac{x - \mathrm{E}[x]}{ \sqrt{\mathrm{Var}[x] + \epsilon}} * \gamma + \beta

    Die Eingabekanäle werden in :attr:`num_groups` Gruppen aufgeteilt, die jeweils ``num_channels / num_groups`` Kanäle enthalten. :attr:`num_channels` muss durch :attr:`num_groups` teilbar sein. Der Mittelwert und die Standardabweichung werden für jede Gruppe separat berechnet. Wenn :attr:`affine` ``True`` ist, dann sind :math:`\gamma` und :math:`\beta` lernbar. Kanalweiser affiner Transformationsparametervektor der Größe :attr:`num_channels`.

    :param num_groups (int): Anzahl der Gruppen, in die die Kanäle aufgeteilt werden
    :param num_channels (int): Anzahl der in der Eingabe erwarteten Kanäle
    :param eps: Wert, der zum Nenner für numerische Stabilität addiert wird. Standard: 1e-5
    :param affine: Ein boolescher Wert, der, wenn auf ``True`` gesetzt, diesem Modul lernbare kanalweise affine Parameter zuweist, initialisiert mit 1 (für Gewichte) und 0 (für Biases). Standard: ``True``.
    :param dtype: Der Datentyp des Parameters, Standard: None, verwendet den Standard-Datentyp kfloat32, der eine 32-Bit-Gleitkommazahl darstellt.
    :param name: Name der Ausgabeschicht

    :return: GroupNorm-Klasse

    Example::

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

    Lineares Modul (vollständig verbundene Schicht).
    :math:`y = x@A.T + b`

    :param input_channels: `int` - Anzahl der Eingabemerkmale
    :param output_channels: `int` - Anzahl der Ausgabemerkmale
    :param weight_initializer: `callable` - Standardwert ist normal
    :param bias_initializer: `callable` - Standardwert ist zeros
    :param use_bias: `bool` - Standardwert ist True
    :param dtype: Der Datentyp des Parameters, Standard: None, verwendet den Standard-Datentyp kfloat32, der eine 32-Bit-Gleitkommazahl darstellt.
    :param name: Name der Ausgabeschicht

    :return: Eine Linear-Klasse

    Example::

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

    Dropout-Modul. Das Dropout-Modul setzt die Ausgaben einiger Einheiten zufällig auf Null, während andere entsprechend der angegebenen Dropout-Wahrscheinlichkeit hochskaliert werden.

    :param dropout_rate: `float` - Wahrscheinlichkeit, dass ein Neuron auf Null gesetzt wird
    :return: Eine Dropout-Klasse

    Example::

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

    Das DropPath-Modul verwirft Pfade (zufällig tief) auf einer Probe-für-Probe-Basis.

    :param dropout_rate: `float` - Die Wahrscheinlichkeit, dass das Neuron auf Null gesetzt wird.
    :param name: Der Name dieses Moduls, Standard ist "".

    :return: DropPath-Instanz.

    Example::

        import pyvqnet.nn as nn
        import pyvqnet.tensor as tensor

        x = tensor.randu([4])
        y = nn.DropPath()(x)
        print(y)
        #[0.9074978,0.9350062,0.6896403,0.3541051]


Pixel_Shuffle 
=================================

.. py:class:: pyvqnet.nn.pixel_shuffle.Pixel_Shuffle(upscale_factors)

    Ordnet Tensoren der Form (*, C * r^2, H, W) in einen Tensor der Form (*, C, H * r, W * r) um, wobei r der Skalierungsfaktor ist.

    :param upscale_factors: Faktor zur Erhöhung der Skalierungstransformation

    :return:
            Pixel_Shuffle-Modul

    Example::

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

    Kehrt die Pixel_Shuffle-Operation um, indem die Elemente neu angeordnet werden. Mischt einen Tensor der Form (*, C, H * r, W * r) in (*, C * r^2, H, W), wobei r der Schrumpfungsfaktor ist.
    
    :param downscale_factors: Faktor zur Erhöhung der Skalierungstransformation

    :return:
            Pixel_Unshuffle-Modul

    Example::

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


    Gated Recurrent Unit (GRU) Modul. Unterstützt mehrschichtige Stapelung und bidirektionale Konfiguration.
    Die Berechnungsformel des einschichtigen, unidirektionalen GRU lautet wie folgt:

    .. math::
        \begin{array}{ll}
            r_t = \sigma(W_{ir} x_t + b_{ir} + W_{hr} h_{(t-1)} + b_{hr}) \\
            z_t = \sigma(W_{iz} x_t + b_{iz} + W_{hz} h_{(t-1)} + b_{hz}) \\
            n_t = \tanh(W_{in} x_t + b_{in} + r_t * (W_{hn} h_{(t-1)}+ b_{hn})) \\
            h_t = (1 - z_t) * n_t + z_t * h_{(t-1)}
        \end{array}

    :param input_size: Dimensionen der Eingabemerkmale.
    :param hidden_size: Dimensionen der versteckten Merkmale.
    :param num_layers: Anzahl der Stapelschichten. Standard: 1.
    :param batch_first: Wenn batch_first True ist, sollte die Eingabeform [batch_size, seq_len, feature_dim] sein,
     wenn batch_first False ist, sollte die Eingabeform [seq_len, batch_size, feature_dim] sein, Standard: True.
    :param use_bias: Wenn use_bias False ist, enthält dieses Modul kein Bias. Standard: True.
    :param bidirectional: Wenn bidirectional True ist, wird das Modul ein bidirektionales GRU. Standard: False.
    :param dtype: Der Datentyp des Parameters, Standard: None, verwendet den Standard-Datentyp kfloat32, der eine 32-Bit-Gleitkommazahl darstellt.
    :param name: Name der Ausgabeschicht

    :return: Eine GRU-Modulinstanz.

    Example::

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


    Rekurrentes Neuronales Netzwerk (RNN) Modul, verwendet :math:`\tanh` oder :math:`\text{ReLU}` als Aktivierungsfunktion.
    Bidirektionales RNN und mehrschichtiges RNN werden unterstützt.
    Die Berechnungsformel des einschichtigen unidirektionalen RNN lautet wie folgt:

    .. math::
        h_t = \tanh(W_{ih} x_t + b_{ih} + W_{hh} h_{(t-1)} + b_{hh})

    Wenn :attr:`nonlinearity` ``'relu'`` ist, dann ersetzt :math:`\text{ReLU}` :math:`\tanh`.

    :param input_size: Dimensionen der Eingabemerkmale.
    :param hidden_size: Dimensionen der versteckten Merkmale.
    :param num_layers: Anzahl der Stapelschichten. Standard: 1.
    :param nonlinearity: Nichtlineare Aktivierungsfunktion, Standard: ``'tanh'`` .
    :param batch_first: Wenn batch_first True ist, sollte die Eingabeform [batch_size, seq_len, feature_dim] sein,
     wenn batch_first False ist, sollte die Eingabeform [seq_len, batch_size, feature_dim] sein, Standard: True.
    :param use_bias: Wenn use_bias False ist, enthält dieses Modul kein Bias. Standard: True.
    :param bidirectional: Wenn bidirectional True ist, wird das Modul ein bidirektionales RNN. Standard: False.
    :param dtype: Der Datentyp des Parameters, Standard: None, verwendet den Standard-Datentyp kfloat32, der eine 32-Bit-Gleitkommazahl darstellt.
    :param name: Name der Ausgabeschicht

    :return: Eine RNN-Modulinstanz.

    Example::

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

    Long Short-Term Memory (LSTM) Modul. Unterstützt bidirektionales LSTM, gestapeltes mehrschichtiges LSTM und andere Konfigurationen.
    Die Berechnungsformel des einschichtigen unidirektionalen LSTM lautet wie folgt:

    .. math::
        \begin{array}{ll} \\
            i_t = \sigma(W_{ii} x_t + b_{ii} + W_{hi} h_{t-1} + b_{hi}) \\
            f_t = \sigma(W_{if} x_t + b_{if} + W_{hf} h_{t-1} + b_{hf}) \\
            g_t = \tanh(W_{ig} x_t + b_{ig} + W_{hg} h_{t-1} + b_{hg}) \\
            o_t = \sigma(W_{io} x_t + b_{io} + W_{ho} h_{t-1} + b_{ho}) \\
            c_t = f_t \odot c_{t-1} + i_t \odot g_t \\
            h_t = o_t \odot \tanh(c_t) \\
        \end{array}

    :param input_size: Dimensionen der Eingabemerkmale.
    :param hidden_size: Dimensionen der versteckten Merkmale.
    :param num_layers: Anzahl der Stapelschichten. Standard: 1.
    :param batch_first: Wenn batch_first True ist, sollte die Eingabeform [batch_size, seq_len, feature_dim] sein,
     wenn batch_first False ist, sollte die Eingabeform [seq_len, batch_size, feature_dim] sein, Standard: True.
    :param use_bias: Wenn use_bias False ist, enthält dieses Modul kein Bias. Standard: True.
    :param bidirectional: Wenn bidirectional True ist, wird das Modul ein bidirektionales LSTM. Standard: False.
    :param dtype: Der Datentyp des Parameters, Standard: None, verwendet den Standard-Datentyp kfloat32, der eine 32-Bit-Gleitkommazahl darstellt.
    :param name: Name der Ausgabeschicht

    :return: Eine LSTM-Modulinstanz.

    Example::

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
    
    Wendet ein mehrschichtiges Gated Recurrent Unit (GRU) RNN auf eine dynamisch lange Eingabesequenz an.

    Die erste Eingabe sollte eine als
    ``tensor.PackedSequence``-Klasse definierte
    variabel lange Batch-Sequenzeingabe sein.
    Die ``tensor.PackedSequence``-Klasse kann durch
    aufeinanderfolgendes Aufrufen von ``pad_sequence``, ``pack_pad_sequence`` konstruiert werden.

    Die erste Ausgabe von Dynamic_GRU ist ebenfalls eine ``tensor.PackedSequence``-Klasse,
    die mit ``tensor.pad_pack_sequence`` in einen normalen QTensor entpackt werden kann.

    Für jedes Element in der Eingabesequenz berechnet jede Schicht die folgende Formel:

    .. math::
        \begin{array}{ll}
            r_t = \sigma(W_{ir} x_t + b_{ir} + W_{hr} h_{(t-1)} + b_{hr}) \\
            z_t = \sigma(W_{iz} x_t + b_{iz} + W_{hz} h_{(t-1)} + b_{hz}) \\
            n_t = \tanh(W_{in} x_t + b_{in} + r_t * (W_{hn} h_{(t-1)}+ b_{hn})) \\
            h_t = (1 - z_t) * n_t + z_t * h_{(t-1)}
        \end{array}

    :param input_size: Dimension der Eingabemerkmale.
    :param hidden_size: Dimension der versteckten Merkmale.
    :param num_layers: Anzahl der Schleifenschichten. Standard: 1
    :param batch_first: Wenn True, wird die Eingabeform als [batch size, sequence length, feature dimension] angegeben. Wenn False, wird die Eingabeform als [sequence length, batch size, feature dimension] angegeben. Standard True.
    :param use_bias: Wenn False, verwendet die Schicht keine Bias-Gewichte b_ih und b_hh. Standard: True.
    :param bidirectional: Wenn True, wird ein bidirektionales GRU erstellt. Standard: False.
    :param dtype: Der Datentyp des Parameters, Standard: None, verwendet den Standard-Datentyp kfloat32, der eine 32-Bit-Gleitkommazahl darstellt.
    :param name: Name der Ausgabeschicht

    :return: Eine Dynamic_GRU-Klasse

    Example::

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

        seq_unpacked, lens_unpacked = \
        tensor.pad_packed_sequence(output, batch_first=False)
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
    
    Wendet rekurrente neuronale Netzwerke (RNNs) auf dynamisch lange Eingabesequenzen an.

    Die erste Eingabe sollte eine als
    ``tensor.PackedSequence``-Klasse definierte
    variabel lange Batch-Sequenzeingabe sein.
    Die ``tensor.PackedSequence``-Klasse kann durch
    aufeinanderfolgendes Aufrufen von ``pad_sequence``, ``pack_pad_sequence`` konstruiert werden.

    Die erste Ausgabe von Dynamic_RNN ist ebenfalls eine ``tensor.PackedSequence``-Klasse,
    die mit ``tensor.pad_pack_sequence`` in einen normalen QTensor entpackt werden kann.

    Rekurrentes Neuronales Netzwerk (RNN) Modul, verwendet :math:`\tanh` oder :math:`\text{ReLU}` als Aktivierungsfunktion. Unterstützt zweiseitige, mehrschichtige Konfiguration.
    Die Berechnungsformel des einschichtigen einseitigen RNN lautet wie folgt:

    .. math::
        h_t = \tanh(W_{ih} x_t + b_{ih} + W_{hh} h_{(t-1)} + b_{hh})
    
    Wenn :attr:`nonlinearity` ``'relu'`` ist, dann ersetzt :math:`\text{ReLU}` :math:`\tanh`.

    :param input_size: Dimension der Eingabemerkmale.
    :param hidden_size: Dimension der versteckten Merkmale.
    :param num_layers: Anzahl der gestapelten RNN-Schichten, Standard: 1.
    :param nonlinearity: Nicht-lineare Aktivierungsfunktion, Standard ist ``'tanh'``.
    :param batch_first: Wenn True, ist die Eingabeform [batch size, sequence length, feature dimension],
      Wenn False, ist die Eingabeform [sequence length, batch size, feature dimension], Standard True.
    :param use_bias: Wenn False, wendet das Modul keine Bias-Elemente an, Standard: True.
    :param bidirectional: Wenn True, wird es ein bidirektionales RNN, Standard: False.
    :param dtype: Der Datentyp des Parameters, Standard: None, verwendet den Standard-Datentyp kfloat32, der eine 32-Bit-Gleitkommazahl darstellt.
    :param name: Name der Ausgabeschicht

    :return: Dynamic_RNN-Instanz

    Example::

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

        seq_unpacked, lens_unpacked = \
        tensor.pad_packed_sequence(output, batch_first=False)
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
    
    Apply Long Short-Term Memory (LSTM) RNNs to dynamic-length input sequences.

    Die erste Eingabe sollte eine als
    ``tensor.PackedSequence``-Klasse definierte
    variabel lange Batch-Sequenzeingabe sein.
    Die ``tensor.PackedSequence``-Klasse kann durch
    aufeinanderfolgendes Aufrufen von ``pad_sequence``, ``pack_pad_sequence`` konstruiert werden.

    The first output of Dynamic_LSTM is also a ``tensor.PackedSequence`` class,
    It can be unpacked into a normal QTensor using ``tensor.pad_pack_sequence``.

    Rekurrentes Neuronales Netzwerk (RNN) Modul, verwendet :math:`\tanh` oder :math:`\text{ReLU}` als Aktivierungsfunktion. Unterstützt zweiseitige, mehrschichtige Konfiguration.
    Die Berechnungsformel des einschichtigen einseitigen RNN lautet wie folgt:

    .. math::
        \begin{array}{ll} \\
            i_t = \sigma(W_{ii} x_t + b_{ii} + W_{hi} h_{t-1} + b_{hi}) \\
            f_t = \sigma(W_{if} x_t + b_{if} + W_{hf} h_{t-1} + b_{hf}) \\
            g_t = \tanh(W_{ig} x_t + b_{ig} + W_{hg} h_{t-1} + b_{hg}) \\
            o_t = \sigma(W_{io} x_t + b_{io} + W_{ho} h_{t-1} + b_{ho}) \\
            c_t = f_t \odot c_{t-1} + i_t \odot g_t \\
            h_t = o_t \odot \tanh(c_t) \\
        \end{array}

    :param input_size: Dimension der Eingabemerkmale.
    :param hidden_size: Dimension der versteckten Merkmale.
    :param num_layers: Number of stacked LSTM layers, default: 1.
    :param batch_first: Wenn True, ist die Eingabeform [batch size, sequence length, feature dimension],
      Wenn False, ist die Eingabeform [sequence length, batch size, feature dimension], Standard True.
    :param use_bias: Wenn False, wendet das Modul keine Bias-Elemente an, Standard: True.
    :param bidirectional: Wenn True, wird es ein bidirektionales LSTM, Standard: False.
    :param dtype: Der Datentyp des Parameters, Standard: None, verwendet den Standard-Datentyp kfloat32, der eine 32-Bit-Gleitkommazahl darstellt.
    :param name: Name der Ausgabeschicht

    :return: Dynamic_LSTM-Instanz

    Example::

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

        seq_unpacked, lens_unpacked = \
        tensor.pad_packed_sequence(output, batch_first=False)

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

    Nimmt eine Herunter-/Hochskalierung der Eingabe vor.

    Derzeit werden nur vierdimensionale Eingabedaten unterstützt.

    Die Eingabedimensionen werden in der Form interpretiert: `B x C x H x W`.

    Die verfügbaren Modi zur Größenänderung sind: ``nearest``, ``bilinear``, ``bicubic``.

    :param size: Ausgabe-Raumgröße.
    :param scale_factor: Multiplikator für die Raumgröße.
    :param mode: Algorithmus, der für das Upsampling verwendet wird: ``nearest`` | ``bilinear`` | ``bicubic``.
    :param align_corners: Geometrisch betrachtet betrachten wir die Pixel der
            Eingabe und Ausgabe als Quadrate und nicht als Punkte.
            Wenn auf ``True`` gesetzt, werden die Eingabe- und Ausgabetensoren an den
            Mittelpunkten ihrer Eckpixel ausgerichtet, wodurch die Werte an den Eckpixeln erhalten bleiben.
            Wenn auf ``False`` gesetzt, werden die Eingabe- und Ausgabetensoren an den Eckpunkten
            ihrer Eckpixel ausgerichtet, und die Interpolation verwendet Randwert-Padding
            für außerhalb liegende Werte, was diese Operation *unabhängig* von der Eingabegröße macht,
            wenn :attr:`scale_factor` gleich bleibt. Dies wirkt sich nur aus, wenn :attr:`mode`
            ``bilinear`` oder ``bicubic`` ist.
    :param recompute_scale_factor: Den scale_factor für die Interpolationsberechnung neu berechnen.
    :param name: Modulname.

    Example::

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

    Wird verwendet, um die entsprechenden benachbarten Module des Modells in der Inferenzphase zu einem Modul zu fusionieren,
    was den Rechenaufwand in der Modell-Inferenzphase reduziert und die Geschwindigkeit der Modell-Inferenz erhöht.

    Die derzeit unterstützten Modulsequenzen sind wie folgt:

    conv, bn

    linear, bn

    Die anderen Sequenzen bleiben unverändert, wobei das erste Modul in der Liste durch das fusionierte Modul ersetzt wird und die anderen durch ``Identity`` ersetzt werden.

    :param input: Enthält die Modellierung der Fusionsmodule.

    :return: Modell mit fusionierten Modulen.

    Examples::
    
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

    SDPA Skalarprodukt-Aufmerksamkeitsmechanismus.

    :param attn_mask: Aufmerksamkeitsmaske; die Form muss mit der Form der Aufmerksamkeitsgewichte broadcast-bar sein.
    :param dropout_p: Dropout-Wahrscheinlichkeit; wenn groesser als 0.0, wird Dropout angewendet.
    :param scale: Skalierungsfaktor, der vor Softmax angewendet wird.
    :param is_causal: Wenn true, wird eine kausale Aufmerksamkeitsmaskierung in der oberen linken Ecke angenommen, und es wird ein Fehler ausgegeben, wenn sowohl attn_mask als auch is_causal gesetzt sind.
    
    Examples::
    
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


Verlustfunktions-Schicht
********************************************************

.. note::

        Bitte beachten Sie, dass im Gegensatz zu PyTorch und anderen Frameworks in der forward-Funktion der folgenden Verlustfunktionen der erste Parameter das Label und der zweite Parameter der vorhergesagte Wert ist.

MeanSquaredError
=================================

.. py:class:: pyvqnet.nn.MeanSquaredError

    Erstellt ein Kriterium, das den mittleren quadratischen Fehler (quadrierte L2-Norm) zwischen
    jedem Element in der Eingabe :math:`x` und dem Ziel :math:`y` misst.

    Der nicht-reduzierte Verlust kann wie folgt beschrieben werden:

    .. math::
        \ell(x, y) = L = \{l_1,\dots,l_N\}^\top, \quad
        l_n = \left( x_n - y_n \right)^2,

    wobei :math:`N` die Batch-Größe ist. Dann:

    .. math::
        \ell(x, y) =
            \operatorname{mean}(L)


    :math:`x` und :math:`y` sind QTensoren beliebiger Form mit insgesamt
    :math:`n` Elementen.

    Die Mittelwert-Operation wirkt weiterhin über alle Elemente und dividiert durch :math:`n`.

    :param name: Name der Ausgabeschicht

    :return: Eine MeanSquaredError-Klasse

    Parameter für die Verlust-Forward-Funktion:

        x: :math:`(N, *)` wobei :math:`*` eine beliebige Anzahl zusätzlicher Dimensionen bedeutet

        y: :math:`(N, *)`, gleiche Form wie die Eingabe

    Example::
    
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

    Misst die binäre Kreuzentropie zwischen dem Ziel und der Ausgabe:

    Der nicht-reduzierte Verlust kann wie folgt beschrieben werden:

    .. math::
        \ell(x, y) = L = \{l_1,\dots,l_N\}^\top, \quad
        l_n = - w_n \left[ y_n \cdot \log x_n + (1 - y_n) \cdot \log (1 - x_n) \right],

    wobei :math:`N` die Batch-Größe ist.

    .. math::
        \ell(x, y) = \operatorname{mean}(L)

    :return: Eine BinaryCrossEntropy-Klasse

    Parameter für die Verlust-Forward-Funktion:

        x: :math:`(N, *)` wobei :math:`*` eine beliebige Anzahl zusätzlicher Dimensionen bedeutet

        y: :math:`(N, *)`, gleiche Form wie die Eingabe

    Example::

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

    Dieses Kriterium kombiniert LogSoftmax und NLLLoss in einer einzigen Klasse.

    Der Verlust kann wie folgt beschrieben werden, wobei `class` der Index der Zielklasse ist:

    .. math::
        \text{loss}(x, class) = -\log\left(\frac{\exp(x[class])}{\sum_j \exp(x[j])}\right)
                       = -x[class] + \log\left(\sum_j \exp(x[j])\right)

    :return: Eine CategoricalCrossEntropy-Klasse

    Parameter für die Verlust-Forward-Funktion:

        x: :math:`(N, *)` wobei :math:`*` eine beliebige Anzahl zusätzlicher Dimensionen bedeutet

        y: :math:`(N, *)`, gleiche Form wie die Eingabe, sollte den Datentyp der 64-Bit-Ganzzahl haben.

    Example::

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

    Dieses Kriterium kombiniert LogSoftmax und NLLLoss in einer einzigen Klasse mit höherer numerischer Stabilität.

    Der Verlust kann wie folgt beschrieben werden, wobei `class` der Index der Zielklasse ist:

    .. math::
        \text{loss}(x, class) = -\log\left(\frac{\exp(x[class])}{\sum_j \exp(x[j])}\right)
                       = -x[class] + \log\left(\sum_j \exp(x[j])\right)

    :return: Eine SoftmaxCrossEntropy-Klasse

    Parameter für die Verlust-Forward-Funktion:

        x: :math:`(N, *)` wobei :math:`*` eine beliebige Anzahl zusätzlicher Dimensionen bedeutet

        y: :math:`(N, *)`, gleiche Form wie die Eingabe, sollte den Datentyp der 64-Bit-Ganzzahl haben.

    Example::

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

    Der durchschnittliche Negative-Log-Likelihood-Verlust. Er ist nützlich für das Training eines Klassifikationsproblems mit `C` Klassen.

    Das `x`, das durch einen forward-Aufruf übergeben wird, sollte Log-Wahrscheinlichkeiten jeder Klasse enthalten. `x` muss ein Tensor der Größe entweder :math:`(N, C)` oder :math:`(N, C, d_1, d_2, ..., d_K)` sein
    mit :math:`K \geq 1` für den `K`-dimensionalen Fall. Das `y`, das dieser Verlust erwartet, sollte ein Klassenindex im Bereich :math:`[0, C-1]` sein, wobei `C = number of classes`.

    .. math::

        \ell(x, y) = L = \{l_1,\dots,l_N\}^\top, \quad
        l_n = -
            \sum_{n=1}^N \frac{1}{N}x_{n,y_n}, \quad

    :return: Eine NLL_Loss-Klasse

    Parameter für die Verlust-Forward-Funktion:

        x: :math:`(N, *)`, die Ausgabe der Verlustfunktion, kann eine mehrdimensionale Variable sein.

        y: :math:`(N, *)`, der von der Verlustfunktion erwartete wahre Wert, sollte den Datentyp der 64-Bit-Ganzzahl haben.


    Example::

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

    Dieses Kriterium kombiniert LogSoftmax und NLLLoss in einer einzigen Klasse.

    `x` sollte rohe, nicht normalisierte Werte (Scores) für jede Klasse enthalten. `x` muss ein Tensor der Größe :math:`(C)` für ungebatchte Eingabe, :math:`(N, C)` oder :math:`(N, C, d_1, d_2, ..., d_K)` mit :math:`K \geq 1` für den `K`-dimensionalen Fall sein.

    Der Verlust kann wie folgt beschrieben werden, wobei `class` der Index der Zielklasse ist:

    .. math::

        \text{loss}(x, class) = -\log\left(\frac{\exp(x[class])}{\sum_j \exp(x[j])}\right)
                       = -x[class] + \log\left(\sum_j \exp(x[j])\right)

    :return: Eine CrossEntropyLoss-Klasse

    Parameter für die Verlust-Forward-Funktion:

        x: :math:`(N, *)`, die Ausgabe der Verlustfunktion, kann eine mehrdimensionale Variable sein.

        y: :math:`(N, *)`, der von der Verlustfunktion erwartete wahre Wert, sollte den Datentyp der 64-Bit-Ganzzahl haben.


    Example::

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



Aktivierungsfunktion
********************************************************


Activation
=================================
.. py:class:: pyvqnet.nn.activation.Activation

    Basisklasse der Aktivierung. Spezifische Aktivierungsfunktionen erben von dieser Klasse.

Sigmoid
=================================
.. py:class:: pyvqnet.nn.Sigmoid(name: str = '')

        Wendet eine Sigmoid-Aktivierungsfunktion auf die angegebene Schicht an.

        .. math::
            \text{Sigmoid}(x) = \frac{1}{1 + \exp(-x)}

        :param name: Name der Ausgabeschicht
        :return: Sigmoid-Aktivierungsschicht

        Examples::

            from pyvqnet.nn import Sigmoid
            from pyvqnet.tensor import QTensor
            layer = Sigmoid()
            y = layer(QTensor([1.0, 2.0, 3.0, 4.0]))
            print(y)

            # [0.7310586, 0.8807970, 0.9525741, 0.9820138]

Softplus
=================================
.. py:class:: pyvqnet.nn.Softplus(name: str = '')

        Wendet die Softplus-Aktivierungsfunktion auf die angegebene Schicht an.

        .. math::
            \text{Softplus}(x) = \log(1 + \exp(x))

        :param name: Name der Ausgabeschicht
        :return: Softplus-Aktivierungsschicht

    Examples::

        from pyvqnet.nn import Softplus
        from pyvqnet.tensor import QTensor
        layer = Softplus()
        y = layer(QTensor([1.0, 2.0, 3.0, 4.0]))
        print(y)

        # [1.3132616, 2.1269281, 3.0485873, 4.0181499]

Softsign
=================================
.. py:class:: pyvqnet.nn.Softsign(name: str = '')

        Wendet die Softsign-Aktivierungsfunktion auf die angegebene Schicht an.

        .. math::
            \text{SoftSign}(x) = \frac{x}{ 1 + |x|}

        :param name: Name der Ausgabeschicht
        :return: Softsign-Aktivierungsschicht

        Examples::

            from pyvqnet.nn import Softsign
            from pyvqnet.tensor import QTensor
            layer = Softsign()
            y = layer(QTensor([1.0, 2.0, 3.0, 4.0]))
            print(y)

            # [0.5000000, 0.6666667, 0.7500000, 0.8000000]

Softmax
=================================
.. py:class:: pyvqnet.nn.Softmax(axis: int = - 1, name: str = '')

    Wendet eine Softmax-Aktivierungsfunktion auf die angegebene Schicht an.

    .. math::
        \text{Softmax}(x_{i}) = \frac{\exp(x_i)}{\sum_j \exp(x_j)}


    :param axis: Dimension, auf der operiert wird (-1 für letzte Achse), Standard = -1
    :param name: Name der Ausgabeschicht
    :return: Softmax-Aktivierungsschicht

    Examples::

        from pyvqnet.nn import Softmax
        from pyvqnet.tensor import QTensor
        layer = Softmax()
        y = layer(QTensor([1.0, 2.0, 3.0, 4.0]))
        print(y)

        # [0.0320586, 0.0871443, 0.2368828, 0.6439142]


HardSigmoid
=================================
.. py:class:: pyvqnet.nn.HardSigmoid(name: str = '')

    Wendet eine Hard-Sigmoid-Aktivierungsfunktion auf die angegebene Schicht an.

    .. math::
        \text{Hardsigmoid}(x) = \begin{cases}
            0 & \text{ if } x \le -3, \\
            1 & \text{ if } x \ge +3, \\
            x / 6 + 1 / 2 & \text{otherwise}
        \end{cases}

    :param name: Name der Ausgabeschicht
    :return: Hard-Sigmoid-Aktivierungsschicht

    Examples::

        from pyvqnet.nn import HardSigmoid
        from pyvqnet.tensor import QTensor
        layer = HardSigmoid()
        y = layer(QTensor([1.0, 2.0, 3.0, 4.0]))
        print(y)

        # [0.6666667, 0.8333334, 1, 1]

ReLu
=================================
.. py:class:: pyvqnet.nn.ReLu(name: str = '')

    Wendet eine Rectified-Linear-Unit-Aktivierungsfunktion auf die angegebene Schicht an.

    .. math::
        \text{ReLu}(x) = \begin{cases}
        x, & \text{ if } x > 0\\
        0, & \text{ if } x \leq 0
        \end{cases}


    :param name: Name der Ausgabeschicht
    :return: ReLu-Aktivierungsschicht

    Examples::

        from pyvqnet.nn import ReLu
        from pyvqnet.tensor import QTensor
        layer = ReLu()
        y = layer(QTensor([-1, 2.0, -3, 4.0]))
        print(y)

        # [0, 2, 0, 4]

LeakyReLu
=================================
.. py:class:: pyvqnet.nn.LeakyReLu(alpha: float = 0.01, name: str = '')

    Wendet die Leaky-Version einer Rectified-Linear-Unit-Aktivierungs-
    funktion auf die angegebene Schicht an.

    .. math::
        \text{LeakyRelu}(x) =
        \begin{cases}
        x, & \text{ if } x \geq 0 \\
        \alpha * x, & \text{ otherwise }
        \end{cases}

    :param alpha: LeakyReLu-Koeffizient, Standard: 0.01
    :param name: Name der Ausgabeschicht
    :return: Leaky-ReLu-Aktivierungsschicht

    Examples::

        from pyvqnet.nn import LeakyReLu
        from pyvqnet.tensor import QTensor
        layer = LeakyReLu()
        y = layer(QTensor([-1, 2.0, -3, 4.0]))
        print(y)

        # [-0.0100000, 2, -0.0300000, 4]

Gelu
=================================
.. py:class:: pyvqnet.nn.Gelu(approximate="tanh", name="")
    
    Wendet die Gaußsche-Fehler-Lineare-Einheit-Funktion an:

    .. math:: \text{GELU}(x) = x * \Phi(x)

    Wenn der Näherungsparameter 'tanh' ist, wird GELU geschätzt durch:

    .. math:: \text{GELU}(x) = 0.5 * x * (1 + \text{Tanh}(\sqrt{2 / \pi} * (x + 0.044715 * x^3)))

    :param approximate: Näherungsberechnungsmethode, Standard ist "tanh".
    :param name: Name der Aktivierungsfunktionsschicht, Standard ist "".

    :return: Gelu-Aktivierungsfunktionsschicht-Instanz.

    Examples::

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

    Wendet die Exponential-Linear-Unit-Aktivierungsfunktion auf die angegebene Schicht an.

    .. math::
        \text{ELU}(x) = \begin{cases}
        x, & \text{ if } x > 0\\
        \alpha * (\exp(x) - 1), & \text{ if } x \leq 0
        \end{cases}

    :param alpha: ELU-Koeffizient, Standard: 1.0
    :param name: Name der Ausgabeschicht
    :return: ELU-Aktivierungsschicht

    Examples::

        from pyvqnet.nn import ELU
        from pyvqnet.tensor import QTensor
        layer = ELU()
        y = layer(QTensor([-1, 2.0, -3, 4.0]))
        print(y)

        # [-0.6321205, 2, -0.9502130, 4]

Tanh
=================================
.. py:class:: pyvqnet.nn.Tanh(name: str = '')

    Wendet die hyperbolische Tangens-Aktivierungsfunktion auf die angegebene Schicht an.

    .. math::
        \text{Tanh}(x) = \frac{\exp(x) - \exp(-x)} {\exp(x) + \exp(-x)}

    :param name: Name der Ausgabeschicht
    :return: Hyperbolische-Tangens-Aktivierungsschicht

    Examples::

        from pyvqnet.nn import Tanh
        from pyvqnet.tensor import QTensor
        layer = Tanh()
        y = layer(QTensor([-1, 2.0, -3, 4.0]))
        print(y)

        # [-0.7615942, 0.9640276, -0.9950548, 0.9993293]


.. _Optimizer:

Optimierer-Modul
********************************************************


Optimizer
=================================
.. py:class:: pyvqnet.optim.optimizer.Optimizer(params, lr=0.01)

    Basisklasse für alle Optimierer.

    :param params: Parameter des Modells, die optimiert werden müssen
    :param lr: Lernrate des Modells (Standard: 0.01)

adadelta
=================================
.. py:class:: pyvqnet.optim.adadelta.Adadelta(params, lr=0.01, beta=0.99, epsilon=1e-8)

    ADADELTA: Eine adaptive Lernraten-Methode. Referenz: (https://arxiv.org/abs/1212.5701)

    .. math::

        E(g_t^2) &= \beta * E(g_{t-1}^2) + (1-\beta) * g^2\\
        Square\_avg &= \sqrt{ ( E(dx_{t-1}^2) + \epsilon ) / ( E(g_t^2) + \epsilon ) }\\
        E(dx_t^2) &= \beta * E(dx_{t-1}^2) + (1-\beta) * (-g*square\_avg)^2 \\
        param\_new &= param - lr * Square\_avg

    :param params: Parameter des Modells, die optimiert werden müssen
    :param lr: Lernrate des Modells (Standard: 0.01)
    :param beta: Zur Berechnung eines laufenden Durchschnitts der quadrierten Gradienten (Standard: 0.99)
    :param epsilon: Zum Nenner hinzugefügter Term zur Verbesserung der numerischen Stabilität (Standard: 1e-8)
    :return: Ein Adadelta-Optimierer

    Example::

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

    Implementiert den Adagrad-Algorithmus. Referenz: (https://databricks.com/glossary/adagrad)

    .. math::
        \begin{aligned}
        moment\_new &= moment + g * g\\param\_new
        &= param - \frac{lr * g}{\sqrt{moment\_new} + \epsilon}
        \end{aligned}

    :param params: Parameter des Modells, die optimiert werden müssen
    :param lr: Lernrate des Modells (Standard: 0.01)
    :param epsilon: Zum Nenner hinzugefügter Term zur Verbesserung der numerischen Stabilität (Standard: 1e-8)
    :return: Ein Adagrad-Optimierer

    Example::

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
    
    Implementiert den AdamW-Algorithmus.

    .. math::
        t=t+1

    .. math::
        param\_new = param - lr*weight\_decay*param
    .. math::
        moment\_1\_new=\beta1*moment\_1+(1−\beta1)g
    .. math::
        moment\_2\_new=\beta2*moment\_2+(1−\beta2)g*g
    .. math::
        lr = lr*\frac{\sqrt{1-\beta2^t}}{1-\beta1^t}
    
    Wenn der Parameter amsgrad True ist

    .. math::
        moment\_2\_max = max(moment\_2\_max,moment\_2)
    .. math::
        param\_new=param\_new-lr*\frac{moment\_1}{\sqrt{moment\_2\_max}+\epsilon}
    
    andernfalls:

    .. math::
        param\_new=param\_new-lr*\frac{moment\_1}{\sqrt{moment\_2}+\epsilon}

    :param params: Modellparameter, die optimiert werden müssen.
    :param lr: Lernrate (Standard: 0.01).
    :param beta1: Koeffizient zur Berechnung des laufenden Durchschnitts des Gradienten und seines Quadrats (Standard: 0.9).
    :param beta2: Koeffizient zur Berechnung des laufenden Durchschnitts des Gradienten und seines Quadrats (Standard: 0.999).
    :param epsilon: Konstante, die zum Nenner addiert wird, um die numerische Stabilität zu verbessern (Standard: 1e-8).
    :param weight_decay: Gewichtsabnahmekoeffizient, Standard 0.01.
    :param amsgrad: Ob die AMSGrad-Variante dieses Algorithmus verwendet werden soll (Standard: False).
    :return: Ein AdamW-Optimierer.

    Example::

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

    Adam: Eine Methode zur stochastischen Optimierung. Referenz: (https://arxiv.org/abs/1412.6980). Sie kann die Lernrate jedes Parameters dynamisch an die Schätzungen des ersten und zweiten Moments des Gradienten anpassen.

    .. math::
        t = t + 1
    .. math::
        param  = param - lr*weight\_decay*param
    .. math::
        moment\_1\_new=\beta1∗moment\_1+(1−\beta1)g
    .. math::
        moment\_2\_new=\beta2∗moment\_2+(1−\beta2)g*g
    .. math::
        lr = lr*\frac{\sqrt{1-\beta2^t}}{1-\beta1^t}

    wenn amsgrad = True
    
    .. math::
        moment\_2\_max = max(moment\_2\_max,moment\_2)
    .. math::
        param\_new=param-lr*\frac{moment\_1}{\sqrt{moment\_2\_max}+\epsilon} 

    sonst

    .. math::
        param\_new=param-lr*\frac{moment\_1}{\sqrt{moment\_2}+\epsilon} 


    :param params: Parameter des Modells, die optimiert werden müssen
    :param lr: Lernrate des Modells (Standard: 0.01)
    :param beta1: Koeffizienten zur Berechnung der laufenden Durchschnitte des Gradienten und seines Quadrats (Standard: 0.9)
    :param beta2: Koeffizienten zur Berechnung der laufenden Durchschnitte des Gradienten und seines Quadrats (Standard: 0.999)
    :param epsilon: Zum Nenner hinzugefügter Term zur Verbesserung der numerischen Stabilität (Standard: 1e-8)
    :param weight_decay: Gewichtsabnahmekoeffizient, Standard 0.
    :param amsgrad: Ob die AMSGrad-Variante dieses Algorithmus verwendet werden soll (Standard: False)
    :return: Ein Adam-Optimierer

    Example::

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

    Implementiert den Adamax-Algorithmus (eine Variante von Adam basierend auf der Unendlichkeitsnorm). Referenz: (https://arxiv.org/abs/1412.6980)

    .. math::
        \\t = t + 1
    .. math::
        moment\_new=\beta1∗moment+(1−\beta1)g
    .. math::
        norm\_new = \max{(\beta1∗norm+\epsilon, \left|g\right|)}
    .. math::
        lr = \frac{lr}{1-\beta1^t}
    .. math::
        param\_new = param − lr*\frac{moment\_new}{norm\_new}\\

    :param params: Parameter des Modells, die optimiert werden müssen
    :param lr: Lernrate des Modells (Standard: 0.01)
    :param beta1: Koeffizienten zur Berechnung der laufenden Durchschnitte des Gradienten und seines Quadrats (Standard: 0.9)
    :param beta2: Koeffizienten zur Berechnung der laufenden Durchschnitte des Gradienten und seines Quadrats (Standard: 0.999)
    :param epsilon: Zum Nenner hinzugefügter Term zur Verbesserung der numerischen Stabilität (Standard: 1e-8)
    :return: Ein Adamax-Optimierer

    Example::

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

    Implementiert den RMSprop-Algorithmus. Referenz: (https://arxiv.org/pdf/1308.0850v5.pdf)

    .. math::
        s_{t+1} = s_{t} + (1 - \beta)*(g)^2

    .. math::
        param_new = param -  \frac{g}{\sqrt{s_{t+1}} + epsilon}

    :param params: Parameter des Modells, die optimiert werden müssen
    :param lr: Lernrate des Modells (Standard: 0.01)
    :param beta: Koeffizienten zur Berechnung der laufenden Durchschnitte des Gradienten und seines Quadrats (Standard: 0.99)
    :param epsilon: Zum Nenner hinzugefügter Term zur Verbesserung der numerischen Stabilität (Standard: 1e-8)
    :return: Ein RMSProp-Optimierer

    Example::

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

    Implementiert den SGD-Algorithmus. Referenz: (https://en.wikipedia.org/wiki/Stochastic_gradient_descent)

    .. math::

        \\param\_new=param-lr*g\\

    :param params: Parameter des Modells, die optimiert werden müssen
    :param lr: Lernrate des Modells (Standard: 0.01)
    :param momentum: Momentum-Faktor (Standard: 0)
    :param nesterov: Aktiviert Nesterov-Momentum (Standard: False)
    :return: Ein SGD-Optimierer

    Example::

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


Metriken
********************************************************


MSE
=================================

.. py:class:: pyvqnet.utils.metrics.MSE(y_true_Qtensor, y_pred_Qtensor)

    MSE: Mittlerer quadratischer Fehler (Mean Squared Error).

    :param y_true_Qtensor: Ein QTensor der Form (n_samples,) oder (n_samples, n_outputs), wahrer Zielwert.
    :param y_pred_Qtensor: Ein QTensor der Form (n_samples,) oder (n_samples, n_outputs), geschätzte Zielwerte.
    :return: Gibt ein float-Ergebnis zurück.

    Example::

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

    RMSE: Wurzel der mittleren quadratischen Abweichung (Root Mean Squared Error).

    :param y_true_Qtensor: Ein QTensor der Form (n_samples,) oder (n_samples, n_outputs), wahrer Zielwert.
    :param y_pred_Qtensor: Ein QTensor der Form (n_samples,) oder (n_samples, n_outputs), geschätzte Zielwerte.
    :return: Gibt ein float-Ergebnis zurück.

    Example::

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

    MAE: Mittlerer absoluter Fehler (Mean Absolute Error).

    :param y_true_Qtensor: Ein QTensor der Form (n_samples,) oder (n_samples, n_outputs), wahrer Zielwert.
    :param y_pred_Qtensor: Ein QTensor der Form (n_samples,) oder (n_samples, n_outputs), geschätzte Zielwerte.
    :return: Gibt ein float-Ergebnis zurück.

    Example::

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

    R_Square: R^2 (Bestimmtheitsmaß) Regressions-Bewertungsfunktion.
    Der bestmögliche Wert ist 1.0, der negativ sein kann
    (da das Modell beliebig schlecht werden kann).
    Einer, der immer den erwarteten Wert von y vorhersagt,
    ignoriert die Eingabemerkmale und erhält einen R^2-Wert von 0.0.
    
    :param y_true_Qtensor: Ein QTensor der Form (n_samples,) oder (n_samples, n_outputs), wahrer Zielwert.
    :param y_pred_Qtensor: Ein QTensor der Form (n_samples,) oder (n_samples, n_outputs), geschätzte Zielwerte.
    :param sample_weight: Array der Form (n_samples,), optionales Stichprobengewicht, Standard: None.
    :return: Gibt ein float-Ergebnis zurück.

    Example::

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

    Berechnet die Precision, Recall und den F1-Score der vorhergesagten Werte unter der 2-Klassifikationsaufgabe. Die vorhergesagten und wahren Werte müssen QTensoren ähnlicher Form (n_samples, ) sein, mit einem Wert von 0 oder 1, die die Labels der beiden Klassen darstellen.
    
    :param y_true_Qtensor: Ein 1D-QTensor, wahrer Zielwert.
    :param y_pred_Qtensor: Ein 1D-QTensor, geschätzter Zielwert.

    :returns: 
        - precision - Precision-Ergebnis
        - recall - Recall-Ergebnis
        - f1 - F1-Score

    Example::

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

    Precision-, Recall- und F1-Score-Berechnungen für Multiklassifikationsaufgaben. Der vorhergesagte Wert und der wahre Wert sind QTensoren ähnlicher Form (n_samples, ), und die Werte sind ganze Zahlen von 0 bis N-1, die die Labels von N Klassen darstellen.

    :param y_true_Qtensor: Ein 1D-QTensor, wahrer Zielwert.
    :param y_pred_Qtensor: Ein 1D-QTensor, geschätzter Zielwert.
    :param N: N Klassen (Anzahl der Klassen).
    :param average: String, ['micro', 'macro', 'weighted'].
             Dieser Parameter ist für Multi-Klassen/Multi-Label-Ziele erforderlich.
             
             ``'micro'``: Berechnet Metriken global durch Zählen der gesamten wahren Positiven, falschen Negative und falschen Positive.
             
             ``'macro'``: Berechnet die Metrik für jedes Label und ermittelt seinen ungewichteten Wert. Dies bedeutet, dass das Gleichgewicht der Labels nicht berücksichtigt wird.
             
             ``'weighted'``: Berechnet die Metriken für jedes Label und ermittelt ihren Durchschnitt (die Anzahl der wahren Instanzen jedes Labels). Dies ändert ``'macro'``, um das Label-Ungleichgewicht zu berücksichtigen; dies kann dazu führen, dass F-Scores nicht zwischen Precision und Recall liegen.
    
    :returns: 
        - precision - Precision-Ergebnis
        - recall - Recall-Ergebnis
        - f1 - F1-Score

    Example::

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

    Precision-, Recall- und F1-Score-Berechnungen für Multiklassifikationsaufgaben. Die vorhergesagten und wahren Werte sind QTensoren ähnlicher Form (n_samples, N), wobei die Werte N-dimensionale One-Hot-kodierte Labelwerte sind.

    :param y_true_Qtensor: Ein 1D-QTensor, wahrer Zielwert.
    :param y_pred_Qtensor: Ein 1D-QTensor, geschätzter Zielwert.
    :param N: N Klassen (Anzahl der Klassen).
    :param average: String, ['micro', 'macro', 'weighted'].
             Dieser Parameter ist für Multi-Klassen/Multi-Label-Ziele erforderlich.
             
             ``'micro'``: Berechnet Metriken global durch Zählen der gesamten wahren Positiven, falschen Negative und falschen Positive.
             
             ``'macro'``: Berechnet die Metrik für jedes Label und ermittelt seinen ungewichteten Wert. Dies bedeutet, dass das Gleichgewicht der Labels nicht berücksichtigt wird.
             
             ``'weighted'``: Berechnet die Metriken für jedes Label und ermittelt ihren Durchschnitt (die Anzahl der wahren Instanzen jedes Labels). Dies ändert ``'macro'``, um das Label-Ungleichgewicht zu berücksichtigen; dies kann dazu führen, dass F-Scores nicht zwischen Precision und Recall liegen.
    
    :returns: 
        - precision - Precision-Ergebnis
        - recall - Recall-Ergebnis
        - f1 - F1-Score

    Example::


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

    Berechnet die Precision, Recall und den F1-Score der Klassifikationsaufgabe.

    :param y_true_Qtensor: Ein QTensor der Form [n_samples].
                             Ein wahres binäres Label. Wenn das Label nicht {1,1} oder {0,1} ist, sollte pos_label explizit angegeben werden.
    :param y_pred_Qtensor: Ein QTensor der Form [n_samples].
                             Ziel-Score, der eine positive Wahrscheinlichkeitsschätzungsklasse, ein Konfidenzwert oder ein nicht-schwellenbasiertes Maß der Entscheidung sein kann (zurückgegeben von "decision_function" bei einigen Klassifikatoren)
    :param pos_label: int oder str. Das Label der positiven Klasse. Standard=None.
                      Wenn ``pos_label`` None ist und ``y_true_Qtensor`` in {-1,1} oder {0,1} liegt, wird ``pos_label`` auf 1 gesetzt, andernfalls wird ein Fehler ausgelöst.
    :param sample_weight: Array der Form (n_samples,), Standard=None.
    :param drop_intermediate: boolean, optional (Standard=True).
                     Ob einige suboptimale Schwellenwerte, die nicht auf der gezeichneten ROC-Kurve erscheinen, weggelassen werden sollen.
    :return: Gibt ein float-Ergebnis aus.

    Example::

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


Triton-Kompatibilität
*********************************************************

`triton <https://triton-lang.org/main/index.html>`_ ist eine Sprache und ein Compiler zum Schreiben effizienter GPU-Kernel für Deep Learning.
Benutzer schreiben Python-Code ähnlich wie NumPy, und Triton kompiliert ihn in effizienten GPU-Code (ähnlich wie CUDA, aber auf höherer Ebene).
Triton ist auf einige PyTorch-Schnittstellen angewiesen. VQNet implementiert eine Schnittstelle ähnlich wie PyTorch, die die Integration mit Triton-geschriebenem Code für die Vorwärts- und Rückwärtspropagation von Modellen ermöglicht.

Triton installieren:

.. code-block::

    pip install triton

Das folgende Beispiel ist modifiziert aus dem offiziellen Triton-Beispiel: `layer-norm <https://triton-lang.org/main/getting-started/tutorials/05-layer-norm.html>`_.
Dieses Beispiel muss auf Linux mit einer GPU ausgeführt werden, und triton sowie pytorch (zur Überprüfung der Berechnungskorrektheit) müssen installiert sein:

.. code-block::

    import torch as real_torch
    import pyvqnet as torch
    import triton
    import triton.language as tl



    DEVICE = torch.DEV_GPU_0

    @triton.jit
    def _layer_norm_fwd_fused(
        X,  # pointer to the input
        Y,  # pointer to the output
        W,  # pointer to the weights
        B,  # pointer to the biases
        Mean,  # pointer to the mean
        Rstd,  # pointer to the 1/std
        stride,  # how much to increase the pointer when moving by 1 row
        N,  # number of columns in X
        eps,  # epsilon to avoid division by zero
        BLOCK_SIZE: tl.constexpr,
    ):
        # Map the program id to the row of X and Y it should compute.
        row = tl.program_id(0)
        Y += row * stride
        X += row * stride
        # Compute mean
        mean = 0
        _mean = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
        for off in range(0, N, BLOCK_SIZE):
            cols = off + tl.arange(0, BLOCK_SIZE)
            a = tl.load(X + cols, mask=cols < N, other=0.).to(tl.float32)
            _mean += a
        mean = tl.sum(_mean, axis=0) / N
        # Compute variance
        _var = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
        for off in range(0, N, BLOCK_SIZE):
            cols = off + tl.arange(0, BLOCK_SIZE)
            x = tl.load(X + cols, mask=cols < N, other=0.).to(tl.float32)
            x = tl.where(cols < N, x - mean, 0.)
            _var += x * x
        var = tl.sum(_var, axis=0) / N
        rstd = 1 / tl.sqrt(var + eps)
        # Write mean / rstd
        tl.store(Mean + row, mean)
        tl.store(Rstd + row, rstd)
        # Normalize and apply linear transformation
        for off in range(0, N, BLOCK_SIZE):
            cols = off + tl.arange(0, BLOCK_SIZE)
            mask = cols < N
            w = tl.load(W + cols, mask=mask)
            b = tl.load(B + cols, mask=mask)
            x = tl.load(X + cols, mask=mask, other=0.).to(tl.float32)
            x_hat = (x - mean) * rstd
            y = x_hat * w + b
            # Write output
            tl.store(Y + cols, y, mask=mask)


    @triton.jit
    def _layer_norm_bwd_dx_fused(DX,  # pointer to the input gradient
                                DY,  # pointer to the output gradient
                                DW,  # pointer to the partial sum of weights gradient
                                DB,  # pointer to the partial sum of biases gradient
                                X,  # pointer to the input
                                W,  # pointer to the weights
                                Mean,  # pointer to the mean
                                Rstd,  # pointer to the 1/std
                                Lock,  # pointer to the lock
                                stride,  # how much to increase the pointer when moving by 1 row
                                N,  # number of columns in X
                                GROUP_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr):
        # Map the program id to the elements of X, DX, and DY it should compute.
        row = tl.program_id(0)
        cols = tl.arange(0, BLOCK_SIZE_N)
        mask = cols < N
        X += row * stride
        DY += row * stride
        DX += row * stride
        # Offset locks and weights/biases gradient pointer for parallel reduction
        lock_id = row % GROUP_SIZE_M
        Lock += lock_id
        Count = Lock + GROUP_SIZE_M
        DW = DW + lock_id * N + cols
        DB = DB + lock_id * N + cols
        # Load data to SRAM

        x = tl.load(X + cols, mask=mask, other=0).to(tl.float32)
        dy = tl.load(DY + cols, mask=mask, other=0).to(tl.float32)
        w = tl.load(W + cols, mask=mask).to(tl.float32)
        mean = tl.load(Mean + row)
        rstd = tl.load(Rstd + row)
        # Compute dx
        xhat = (x - mean) * rstd
        wdy = w * dy
        xhat = tl.where(mask, xhat, 0.)
        wdy = tl.where(mask, wdy, 0.)
        c1 = tl.sum(xhat * wdy, axis=0) / N
        c2 = tl.sum(wdy, axis=0) / N
        dx = (wdy - (xhat * c1 + c2)) * rstd
        # Write dx
        tl.store(DX + cols, dx, mask=mask)

        # Accumulate partial sums for dw/db
        partial_dw = (dy * xhat).to(w.dtype)
        partial_db = (dy).to(w.dtype)
        while tl.atomic_cas(Lock, 0, 1) == 1:
            pass
        count = tl.load(Count)
        # First store doesn't accumulate
        if count == 0:
            tl.atomic_xchg(Count, 1)
        else:
            partial_dw += tl.load(DW, mask=mask)
            partial_db += tl.load(DB, mask=mask)
        tl.store(DW, partial_dw, mask=mask)
        tl.store(DB, partial_db, mask=mask)

        # need a barrier to ensure all threads finished before
        # releasing the lock
        tl.debug_barrier()

        # Release the lock
        tl.atomic_xchg(Lock, 0)


    @triton.jit
    def _layer_norm_bwd_dwdb(DW,  # pointer to the partial sum of weights gradient
                            DB,  # pointer to the partial sum of biases gradient
                            FINAL_DW,  # pointer to the weights gradient
                            FINAL_DB,  # pointer to the biases gradient
                            M,  # GROUP_SIZE_M
                            N,  # number of columns
                            BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr):
        # Map the program id to the elements of DW and DB it should compute.
        pid = tl.program_id(0)
        cols = pid * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
        dw = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
        db = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
        # Iterate through the rows of DW and DB to sum the partial sums.
        for i in range(0, M, BLOCK_SIZE_M):
            rows = i + tl.arange(0, BLOCK_SIZE_M)
            mask = (rows[:, None] < M) & (cols[None, :] < N)
            offs = rows[:, None] * N + cols[None, :]
            dw += tl.load(DW + offs, mask=mask, other=0.)
            db += tl.load(DB + offs, mask=mask, other=0.)
        # Write the final sum to the output.
        sum_dw = tl.sum(dw, axis=0)
        sum_db = tl.sum(db, axis=0)
        tl.store(FINAL_DW + cols, sum_dw, mask=cols < N)
        tl.store(FINAL_DB + cols, sum_db, mask=cols < N)


    class LayerNorm(torch.autograd.Function):

        @staticmethod
        def forward(ctx, x, normalized_shape, weight, bias, eps):
            # allocate output
            y = torch.empty_like(x)
            # reshape input data into 2D tensor
            x_arg = x.reshape([-1, x.shape[-1]])
            M, N = x_arg.shape
            mean = torch.empty((M, ), dtype=torch.float32, device=x.device).data
            rstd = torch.empty((M, ), dtype=torch.float32, device=x.device).data
            # Less than 64KB per feature: enqueue fused kernel
            MAX_FUSED_SIZE = 65536 // x.element_size()
            BLOCK_SIZE = min(MAX_FUSED_SIZE, triton.next_power_of_2(N))
            if N > BLOCK_SIZE:
                raise RuntimeError("This layer norm doesn't support feature dim >= 64KB.");
            # heuristics for number of warps
            num_warps = min(max(BLOCK_SIZE // 256, 1), 8)
            # enqueue kernel
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
            # heuristics for amount of parallel reduction stream for DW/DB
            N = w.shape[0]
            GROUP_SIZE_M = 64
            if N <= 8192: GROUP_SIZE_M = 96
            if N <= 4096: GROUP_SIZE_M = 128
            if N <= 1024: GROUP_SIZE_M = 256
            # allocate output
            locks = torch.zeros(2 * GROUP_SIZE_M, dtype=torch.int32, device=w.device).data
            _dw = torch.zeros((GROUP_SIZE_M, N), dtype=x.dtype, device=w.device).data
            _db = torch.zeros((GROUP_SIZE_M, N), dtype=x.dtype, device=w.device).data
            dw = torch.empty((N, ), dtype=w.dtype, device=w.device).data
            db = torch.empty((N, ), dtype=w.dtype, device=w.device).data
            dx = torch.empty_like(dy).data
            # enqueue kernel using forward pass heuristics
            # also compute partial sums for DW and DB
            x_arg = x.reshape([-1, x.shape[-1]])
            M, N = x_arg.shape

            _layer_norm_bwd_dx_fused[(M, )](  #
                dx, dy, _dw, _db, x, w, m, v, locks,  #
                x_arg.stride[0], N,  #
                BLOCK_SIZE_N=ctx.BLOCK_SIZE,  #
                GROUP_SIZE_M=GROUP_SIZE_M,  #
                num_warps=ctx.num_warps)

            grid = lambda meta: (triton.cdiv(N, meta['BLOCK_SIZE_N']), )

            # accumulate partial sums in separate kernel
            _layer_norm_bwd_dwdb[grid](
                _dw, _db, dw, db, min(GROUP_SIZE_M, M), N,  #
                BLOCK_SIZE_M=32,  #
                BLOCK_SIZE_N=128, num_ctas=1)
            return dx, None, dw, db, None

    preprocess = LayerNorm.preprocess
    layer_norm = LayerNorm.apply


    def test_layer_norm(M, N, dtype, eps=1e-5, device=DEVICE):
        # create data
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
        # compare
        import numpy as np
        assert np.allclose(y_tri.detach().cpu().numpy(), y_ref.detach().cpu().numpy(), atol=1e-2, rtol=0)
        assert np.allclose(dx_tri, dx_ref, atol=1e-2, rtol=0)
        assert np.allclose(db_tri, db_ref, atol=1e-2, rtol=0)
        assert np.allclose(dw_tri, dw_ref, atol=1e-2, rtol=0)
        print("same")
    test_layer_norm(12, 32, torch.float32)
