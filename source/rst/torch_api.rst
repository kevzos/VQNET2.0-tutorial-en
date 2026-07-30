

.. _torch_api:

=============================================================
VQNet verwendet torch für Berechnungen auf niedriger Ebene
=============================================================

Ab Version 2.15.0 unterstützt diese Software die Verwendung von `torch` als Rechen-Backend für Operationen auf niedriger Ebene und kann mit Modellen, Code und Drittanbieterbibliotheken basierend auf `torch` für die Weiterentwicklung integriert werden.

    .. important::

        Um die folgenden Funktionen zu nutzen, installieren Sie bitte selbst torch>=2.11.0. Wenn Sie eine GPU-Version von torch installieren, müssen Sie eine mit CUDA 12.6 kompatible Version verwenden, da Ihre torch-Installation sonst aufgrund von Problemen mit der NVIDIA CUDA-Laufzeitbibliothek möglicherweise nicht funktioniert. Diese Software installiert torch während der Installation nicht automatisch.

    .. note::

        Die Funktionen zur variationellen Quantenberechnung (mit Kleinbuchstaben, wie `rx`, `ry`, `rz` usw.) in :ref:`vqc_api` sowie die grundlegenden Berechnungsfunktionen von QTensor in :ref:`qtensor_api` können
        nach Aufruf von ``pyvqnet.backends.set_backend("torch")`` einen `QTensor` als Eingabe verwenden, wobei das `data`-Mitglied von `QTensor` von pyvqnet's Tensor zu ``torch.Tensor`` für die Berechnung wechselt.

        ``pyvqnet.backends.set_backend("torch")`` und ``pyvqnet.backends.set_backend("pyvqnet")`` ändern das globale Rechen-Backend.
        ``QTensor``-Objekte, die unter verschiedenen Backend-Konfigurationen erstellt wurden, können nicht in Berechnungen gemischt werden.

Grundlegende Backend-Konfiguration
============================================

set_backend
------------------------------------------------

.. py:function:: pyvqnet.backends.set_backend(backend_name)

    Setzt das Backend für aktuelle Berechnungen und Datenspeicherung. Der Standardwert ist "pyvqnet-ad", es kann aber auch auf "torch", "torch-native" oder "pyvqnet-ad" gesetzt werden.
    
    Nach dem Aufruf von ``pyvqnet.backends.set_backend("torch")`` bleibt die Schnittstelle unverändert. Die ``data``-Membervariable von VQNets ``QTensor`` verwendet durchgängig ``torch.Tensor`` zur Datenspeicherung.
    :ref:`qtensor_api`, :ref:`vqc_api` und die ``pyvqnet.nn.torch``-Schnittstellen akzeptieren ``QTensor`` als Eingabe und geben ``QTensor`` als Ausgabe aus.

    Nach dem Aufruf von ``pyvqnet.backends.set_backend("torch-native")`` bleiben die Schnittstellen unverändert: :ref:`qtensor_api`, :ref:`vqc_api` und die `pyvqnet.nn.torch`-Schnittstelle.
    Eingaben können direkt ``torch.Tensor`` oder ``QTensor``-Typen akzeptieren, und Ausgaben sind ``torch.Tensor``, wodurch die Konvertierung zu ``QTensor`` entfällt und somit Datenkonvertierung reduziert wird.
    
    Nach dem Aufruf von ``pyvqnet.backends.set_backend("pyvqnet")`` speichert das ``data``-Mitglied von VQNets ``QTensor`` Daten mit ``pyvqnet._core.Tensor``, und Berechnungen verwenden die pyvqnet C++-Bibliothek.

    Nach dem Aufruf von ``pyvqnet.backends.set_backend("pyvqnet-ad")`` speichert das ``data``-Mitglied von VQNets ``QTensor`` Daten mit ``pyvqnet._core.Tensor``, und Berechnungen verwenden die pyvqnet C++-Bibliothek mit verbesserter Leistung.


    .. note::

        Diese Funktion ändert das aktuelle Rechen-Backend. ``QTensor``-Objekte, die unter verschiedenen Backends erstellt wurden, können nicht zusammen in Berechnungen verwendet werden.

    :param backend_name: Name des Backends, kann "pyvqnet" oder "torch" sein.

    Example::

        import pyvqnet
        pyvqnet.backends.set_backend("torch")

get_backend
-------------------------------

.. py:function:: pyvqnet.backends.get_backend(t=None)

    Wenn `t` None ist, wird das aktuelle Rechen-Backend abgerufen.
    Wenn `t` ein QTensor ist, wird das Backend zurückgegeben, das zum Erstellen des QTensor basierend auf seiner ``data``-Eigenschaft verwendet wurde.
    Wenn "torch" das Backend ist, wird das pyvqnet torchAPI-Backend zurückgegeben.
    Wenn "pyvqnet" das Backend ist, wird einfach "pyvqnet" zurückgegeben.
    
    :param t: Der aktuelle Tensor (Standard: None).
    :return: Das Backend. Standardmäßig wird "pyvqnet" zurückgegeben.

    Example::

        import pyvqnet
        pyvqnet.backends.set_backend("torch")
        pyvqnet.backends.get_backend()

QTensor-Funktionen
===================

Nach dem Setzen des Backends auf ``torch``:

.. code-block::

    import pyvqnet
    pyvqnet.backends.set_backend("torch")

Alle Member-Funktionen, Erstellungsfunktionen, mathematischen Funktionen, logischen Funktionen, Matrixtransformationen usw. unter :ref:`qtensor_api` verwenden torch für Berechnungen. Auf `QTensor.data` kann zugegriffen werden, um die Torch-Daten abzurufen.

Klassische Neuronale Netzwerke und Variationelle Quanten-Neuronale Netzwerke Module
==========================================================================================

Basisklasse
------------------------------------------------

TorchModule
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. py:class:: pyvqnet.nn.torch.TorchModule(*args, **kwargs)

    Die Basisklasse, die Modelle bei Verwendung des `torch`-Backends definiert. Diese Klasse erbt sowohl von ``pyvqnet.nn.Module`` als auch von ``torch.nn.Module``.
    Sie kann als Untermodul zu einem torch-Modell hinzugefügt werden.

    .. note::

        Diese Klasse und ihre abgeleiteten Klassen sind nur für die Verwendung mit ``pyvqnet.backends.set_backend("torch")`` geeignet.
        Nicht mit dem standardmäßigen ``pyvqnet.nn`` `Module` mischen.
    
        Die Daten in den ``_buffers`` dieser Klasse sind vom Typ ``torch.Tensor``.
        Die Daten in den ``_parameters`` dieser Klasse sind vom Typ ``torch.nn.Parameter``.

    .. py:method:: pyvqnet.nn.torch.TorchModule.forward(x, *args, **kwargs)

        Abstrakte Vorwärtsberechnungsfunktion für die TorchModule-Klasse.

        :param x: Eingabe-QTensor.
        :param args: Nicht-Schlüsselwort-Variablenargumente.
        :param kwargs: Schlüsselwort-Variablenargumente.

        :return: Ausgabe-QTensor, dessen interne `data` ein ``torch.Tensor`` ist.

        Example::

            import numpy as np
            from pyvqnet.tensor import QTensor
            import pyvqnet
            pyvqnet.backends.set_backend("torch")
            from pyvqnet.nn.torch import Conv2D
            b = 2
            ic = 3
            oc = 2
            test_conv = Conv2D(ic, oc, (3, 3), (2, 2), "valid")
            x0 = QTensor(np.arange(1, b * ic * 5 * 5 + 1).reshape([b, ic, 5, 5]),
                        requires_grad=True,
                        dtype=pyvqnet.kfloat32)
            x = test_conv.forward(x0)
            print(x)

    .. py:method:: pyvqnet.nn.torch.TorchModule.state_dict(destination=None, prefix='')

        Gibt ein Wörterbuch zurück, das den gesamten Zustand des Moduls enthält, einschließlich Parameter- und Pufferwerten.
        Die Schlüssel sind die Namen der entsprechenden Parameter und Puffer.

        :param destination: Das Wörterbuch zum Speichern der internen Modulparameter.
        :param prefix: Ein Präfix für die Namen der Parameter und Puffer.

        :return: Ein Wörterbuch mit dem gesamten Zustand des Moduls.

        Example::

            from pyvqnet.nn.torch import Conv2D
            import pyvqnet
            pyvqnet.backends.set_backend("torch")
            test_conv = Conv2D(2,3,(3,3),(2,2),"same")
            print(test_conv.state_dict().keys())

    .. py:method:: pyvqnet.nn.torch.TorchModule.load_state_dict(state_dict, strict=True)

        Kopiert Parameter und Puffer aus dem :attr:`state_dict` in dieses Modul und seine Untermodule.

        :param state_dict: Ein Wörterbuch mit Parametern und persistenten Puffern.
        :param strict: Gibt an, ob die Schlüssel im state_dict mit dem `state_dict()` des Modells übereinstimmen müssen. Standard: True.

        :return: Eine Fehlermeldung, falls ein Problem auftritt.

        Examples::

            from pyvqnet.nn.torch import TorchModule, Conv2D
            import pyvqnet
            import pyvqnet.utils
            pyvqnet.backends.set_backend("torch")

            class Net(TorchModule):
                def __init__(self):
                    super(Net, self).__init__()
                    self.conv1 = Conv2D(input_channels=1, output_channels=6, kernel_size=(5, 5),
                        stride=(1, 1), padding="valid")

                def forward(self, x):
                    return super().forward(x)

            model = Net()
            pyvqnet.utils.storage.save_parameters(model.state_dict(), "tmp.model")
            model_param = pyvqnet.utils.storage.load_parameters("tmp.model")
            model.load_state_dict(model_param)

    .. py:method:: pyvqnet.nn.torch.TorchModule.toGPU(device: int = DEV_GPU_0)

        Verschiebt die Parameter- und Pufferdaten des Moduls und seiner Untermodule auf das angegebene GPU-Gerät.

        Das Gerät gibt an, wo die internen Daten gespeichert werden. Wenn device >= DEV_GPU_0, werden Daten auf der GPU gespeichert.
        Wenn Ihr Computer mehrere GPUs hat, können Sie verschiedene Geräte zur Datenspeicherung angeben. Zum Beispiel bezieht sich device = DEV_GPU_1, DEV_GPU_2, DEV_GPU_3, ... auf GPUs mit unterschiedlichen Seriennummern.
        
        .. note::

            Module können keine Berechnungen über verschiedene GPUs hinweg durchführen.
            Wenn Sie versuchen, einen QTensor auf einer GPU-ID zu erstellen, die über dem maximal zulässigen Wert liegt, wird ein Cuda-Fehler ausgelöst.

        :param device: Das Gerät zum Speichern des QTensor. Standard: DEV_GPU_0. device = pyvqnet.DEV_GPU_0 speichert auf der ersten GPU, device = DEV_GPU_1 auf der zweiten GPU usw.
        :return: Das auf das GPU-Gerät verschobene Modul.

        Examples::

            from pyvqnet.nn.torch import ConvT2D
            import pyvqnet
            pyvqnet.backends.set_backend("torch")
            test_conv = ConvT2D(3, 2, [4, 4], [2, 2], (0, 0))
            test_conv = test_conv.toGPU()
            print(test_conv.backend)
            #1000

    .. py:method:: pyvqnet.torch.TorchModule.toCPU()

        Verschiebt die Parameter- und Pufferdaten des Moduls und seiner Untermodule auf ein bestimmtes CPU-Gerät.

        :return: Das auf das CPU-Gerät verschobene Modul.

        Examples::

            from pyvqnet.nn.torch import ConvT2D
            import pyvqnet
            pyvqnet.backends.set_backend("torch")
            test_conv = ConvT2D(3, 2, [4, 4], [2, 2], (0, 0))
            test_conv = test_conv.toCPU()
            print(test_conv.backend)
            #0


TorchModuleList
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. py:class:: pyvqnet.nn.torch.TorchModuleList(modules = None)

    Dieses Modul wird verwendet, um untergeordnete ``TorchModule``-Instanzen in einer Liste zu speichern. Die TorchModuleList kann wie eine normale Python-Liste indiziert werden, und die darin enthaltenen internen Parameter können gespeichert werden.
    
    Diese Klasse erbt von ``pyvqnet.nn.torch.TorchModule`` und ``pyvqnet.nn.ModuleList`` und kann als Untermodul zu einem torch-Modell hinzugefügt werden.

    :param modules: Eine Liste von ``pyvqnet.nn.torch.TorchModule``

    :return: Eine TorchModuleList-Klasse

    Example::

        from pyvqnet.tensor import *
        from pyvqnet.nn.torch import TorchModule, Linear, TorchModuleList

        import pyvqnet
        pyvqnet.backends.set_backend("torch")

        class M(TorchModule):
            def __init__(self):
                super(M, self).__init__()
                self.pqc2 = TorchModuleList([Linear(4, 1), Linear(4, 1)])

            def forward(self, x):
                y = self.pqc2[0](x) + self.pqc2[1](x)
                return y

        mm = M()

TorchParameterList
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. py:class:: pyvqnet.nn.torch.TorchParameterList(value=None)

    Dieses Modul wird verwendet, um untergeordnete ``pyvqnet.nn.Parameter``-Instanzen in einer Liste zu speichern. Die TorchParameterList kann wie eine normale Python-Liste indiziert werden, und die darin enthaltenen internen Parameter können gespeichert werden.
    
    Diese Klasse erbt von ``pyvqnet.nn.torch.TorchModule`` und ``pyvqnet.nn.ParameterList`` und kann als Untermodul zu einem torch-Modell hinzugefügt werden.

    :param value: Eine Liste von ``nn.Parameter``

    :return: Eine TorchParameterList-Klasse

    Example::

        from pyvqnet.tensor import *
        from pyvqnet.nn.torch import TorchModule, Linear, TorchParameterList
        import pyvqnet.nn as nn
        import pyvqnet
        pyvqnet.backends.set_backend("torch")

        class MyModule(TorchModule):
            def __init__(self):
                super().__init__()
                self.params = TorchParameterList([nn.Parameter((10, 10)) for i in range(10)])

            def forward(self, x):
                # ParameterList can act as an iterable, or be indexed using ints
                for i, p in enumerate(self.params):
                    x = self.params[i // 2] * x + p * x
                return x

        model = MyModule()
        print(model.state_dict().keys())

TorchSequential
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. py:class:: pyvqnet.nn.torch.TorchSequential(*args)

    Dieses Modul fügt Module in der Reihenfolge hinzu, in der sie übergeben werden. Alternativ kann ein ``OrderedDict`` von Modulen übergeben werden. Die ``forward()``-Methode der ``Sequential``-Klasse akzeptiert beliebige Eingaben und leitet sie an ihr erstes Modul weiter.
    Die Ausgabe wird dann sequenziell mit der Eingabe jedes nachfolgenden Moduls verknüpft, wobei die endgültige Ausgabe das Ergebnis des letzten Moduls ist.

    Diese Klasse erbt von ``pyvqnet.nn.torch.TorchModule`` und ``pyvqnet.nn.Sequential`` und kann als Untermodul zu einem torch-Modell hinzugefügt werden.

    :param args: Hinzuzufügende Module

    :return: Eine TorchSequential-Klasse

    Example::

        import pyvqnet
        from collections import OrderedDict
        from pyvqnet.tensor import *
        from pyvqnet.nn.torch import TorchModule, Conv2D, ReLu, \
            TorchSequential
        pyvqnet.backends.set_backend("torch")
        model = TorchSequential(
                    Conv2D(1, 20, (5, 5)),
                    ReLu(),
                    Conv2D(20, 64, (5, 5)),
                    ReLu()
                )
        print(model.state_dict().keys())

        model = TorchSequential(OrderedDict([
                    ('conv1', Conv2D(1, 20, (5, 5))),
                    ('relu1', ReLu()),
                    ('conv2', Conv2D(20, 64, (5, 5))),
                    ('relu2', ReLu())
                ]))
        print(model.state_dict().keys())

Speichern und Laden von Modellparametern
--------------------------------------------

Sie können :ref:`save_parameters` mit ``save_parameters`` und ``load_parameters`` verwenden, um die Parameter eines ``TorchModule``-Modells als Wörterbuch in einer Datei zu speichern, wobei die Werte als `numpy.ndarray` gespeichert werden. Alternativ können Sie die Parameterdatei von der Festplatte laden. Beachten Sie, dass die Modellstruktur nicht in der Datei gespeichert wird und Sie die Modellstruktur manuell rekonstruieren müssen. Sie können auch direkt ``torch.save`` und ``torch.load`` verwenden, um die ``torch``-Modellparameter zu lesen, da die Parameter von ``TorchModule`` als ``torch.Tensor``-Objekte gespeichert sind.


Klassische Neuronale Netzwerk-Module
--------------------------------------------

Die folgenden klassischen neuronalen Netzwerk-Module erben alle von ``pyvqnet.nn.Module`` und ``torch.nn.Module`` und können als Untermodule zu einem torch-Modell hinzugefügt werden.

Linear
^^^^^^^^^^^^^^^^^^^^^^^^

.. py:class:: pyvqnet.nn.torch.Linear(input_channels, output_channels, weight_initializer=None, bias_initializer=None, use_bias=True, dtype=None, name: str = "")

    Ein lineares Modul (voll verbundene Schicht), :math:`y = x@A.T + b`.
    Diese Klasse erbt von ``pyvqnet.nn.Module`` und ``torch.nn.Module`` und kann als Untermodul eines torch-Modells verwendet werden.

    Die Daten in den ``_buffers`` der Klasse sind vom Typ ``torch.Tensor``.
    Die Daten in den ``_parameters`` der Klasse sind vom Typ ``torch.nn.Parameter``.

    :param input_channels: `int` - Die Anzahl der Eingabekanäle.
    :param output_channels: `int` - Die Anzahl der Ausgabekanäle.
    :param weight_initializer: `callable` - Gewichtungsinitialisierungsfunktion, standardmäßig leer, verwendet he_uniform.
    :param bias_initializer: `callable` - Bias-Initialisierungsfunktion, standardmäßig leer, verwendet he_uniform.
    :param use_bias: `bool` - Ob der Bias-Term verwendet werden soll, Standard ist True.
    :param dtype: Datentyp für die Parameter, standardmäßig None, verwendet den Standarddatentyp `kfloat32`, der 32-Bit-Gleitkommazahlen repräsentiert.
    :param name: Der Name der linearen Schicht, Standard ist "".

    :return: Eine Instanz der linearen Schicht.

    Example::

        import numpy as np
        import pyvqnet
        from pyvqnet.tensor import QTensor
        from pyvqnet.nn.torch import Linear
        pyvqnet.backends.set_backend("torch")
        c1 = 2
        c2 = 3
        cin = 7
        cout = 5
        n = Linear(cin, cout)
        input = QTensor(np.arange(1, c1 * c2 * cin + 1).reshape((c1, c2, cin)), requires_grad=True, dtype=pyvqnet.kfloat32)
        y = n.forward(input)
        print(y)

Conv1D
^^^^^^^^^^^^^^^^^^^^^^^^

.. py:class:: pyvqnet.nn.torch.Conv1D(input_channels: int, output_channels: int, kernel_size: int, stride: int = 1, padding = "valid", use_bias: bool = True, kernel_initializer = None, bias_initializer = None, dilation_rate: int = 1, group: int = 1, dtype = None, name = "")

    Führt eine 1D-Faltung auf der Eingabe durch. Die Eingabe für das Conv1D-Modul hat die Form (batch_size, input_channels, in_height).
    Diese Klasse erbt von ``pyvqnet.nn.Module`` und ``torch.nn.Module`` und kann als Untermodul eines torch-Modells verwendet werden.

    Die Daten in den ``_buffers`` der Klasse sind vom Typ ``torch.Tensor``.
    Die Daten in den ``_parameters`` der Klasse sind vom Typ ``torch.nn.Parameter``.

    :param input_channels: `int` - Die Anzahl der Eingabekanäle.
    :param output_channels: `int` - Die Anzahl der Ausgabekanäle.
    :param kernel_size: `int` - Die Größe des Faltungskerns. Die Kernelform ist [output_channels, input_channels/group, kernel_size, 1].
    :param stride: `int` - Der Schritt, Standard ist 1.
    :param padding: `str|int` - Padding-Optionen, entweder ein String {'valid', 'same'} oder eine ganze Zahl, die den auf die Eingabe anzuwendenden Padding-Betrag angibt. Standard ist "valid".
    :param use_bias: `bool` - Ob der Bias-Term verwendet werden soll, Standard ist True.
    :param kernel_initializer: `callable` - Die Initialisierungsmethode für den Faltungskern. Standard ist leer, verwendet kaiming_uniform.
    :param bias_initializer: `callable` - Die Initialisierungsmethode für den Bias. Standard ist leer, verwendet kaiming_uniform.
    :param dilation_rate: `int` - Die Dilationsgröße, Standard ist 1.
    :param group: `int` - Die Anzahl der Gruppen in der gruppierten Faltung. Standard ist 1.
    :param dtype: Datentyp für die Parameter, standardmäßig None, verwendet den Standarddatentyp `kfloat32`, der 32-Bit-Gleitkommazahlen repräsentiert.
    :param name: Der Name des Moduls, Standard ist "".

    :return: Eine Instanz der 1D-Faltung.

    .. note::

        ``padding='valid'`` wendet kein Padding an.

        ``padding='same'`` wendet Zero-Padding auf die Eingabe an, wobei die Ausgabehöhe `out_height` gleich `ceil(in_height / stride)` ist, und unterstützt kein `stride > 1`.

    Example::

        import numpy as np
        import pyvqnet
        from pyvqnet.tensor import QTensor
        from pyvqnet.nn.torch import Conv1D
        pyvqnet.backends.set_backend("torch")
        b = 2
        ic = 3
        oc = 2
        test_conv = Conv1D(ic, oc, 3, 2)
        x0 = QTensor(np.arange(1, b * ic * 5 * 5 + 1).reshape([b, ic, 25]), requires_grad=True, dtype=pyvqnet.kfloat32)
        x = test_conv.forward(x0)
        print(x)

Conv2D
^^^^^^^^^^^^^^^^^^^^^^^^

.. py:class:: pyvqnet.nn.torch.Conv2D(input_channels: int, output_channels: int, kernel_size: tuple, stride: tuple = (1, 1), padding = "valid", use_bias = True, kernel_initializer = None, bias_initializer = None, dilation_rate: int = 1, group: int = 1, dtype = None, name = "")

    Führt eine 2D-Faltung auf der Eingabe durch. Die Eingabe für das Conv2D-Modul hat die Form (batch_size, input_channels, height, width).
    Diese Klasse erbt von ``pyvqnet.nn.Module`` und ``torch.nn.Module`` und kann als Untermodul eines torch-Modells verwendet werden.

    Die Daten in den ``_buffers`` der Klasse sind vom Typ ``torch.Tensor``.
    Die Daten in den ``_parameters`` der Klasse sind vom Typ ``torch.nn.Parameter``.

    :param input_channels: `int` - Die Anzahl der Eingabekanäle.
    :param output_channels: `int` - Die Anzahl der Ausgabekanäle.
    :param kernel_size: `tuple|list` - Die Größe des Faltungskerns. Die Kernelform ist [output_channels, input_channels/group, kernel_size, kernel_size].
    :param stride: `tuple|list` - Der Schritt, Standard ist (1, 1).
    :param padding: `str|tuple` - Padding-Optionen, entweder ein String {'valid', 'same'} oder ein Tupel, das das auf beide Seiten anzuwendende Padding angibt. Standard ist "valid".
    :param use_bias: `bool` - Ob der Bias-Term verwendet werden soll, Standard ist True.
    :param kernel_initializer: `callable` - Die Initialisierungsmethode für den Faltungskern. Standard ist leer, verwendet kaiming_uniform.
    :param bias_initializer: `callable` - Die Initialisierungsmethode für den Bias. Standard ist leer, verwendet kaiming_uniform.
    :param dilation_rate: `int` - Die Dilationsgröße, Standard ist 1.
    :param group: `int` - Die Anzahl der Gruppen in der gruppierten Faltung. Standard ist 1.
    :param dtype: Datentyp für die Parameter, standardmäßig None, verwendet den Standarddatentyp `kfloat32`, der 32-Bit-Gleitkommazahlen repräsentiert.
    :param name: Der Name des Moduls, Standard ist "".

    :return: Eine Instanz der 2D-Faltung.

    .. note::

        ``padding='valid'`` wendet kein Padding an.

        ``padding='same'`` wendet Zero-Padding auf die Eingabe an, wobei die Ausgabehöhe gleich `ceil(in_height / stride)` ist.

    Example::

        import numpy as np
        import pyvqnet
        from pyvqnet.tensor import QTensor
        from pyvqnet.nn.torch import Conv2D
        pyvqnet.backends.set_backend("torch")
        b = 2
        ic = 3
        oc = 2
        test_conv = Conv2D(ic, oc, (3, 3), (2, 2))
        x0 = QTensor(np.arange(1, b * ic * 5 * 5 + 1).reshape([b, ic, 5, 5]), requires_grad=True, dtype=pyvqnet.kfloat32)
        x = test_conv.forward(x0)
        print(x)

ConvT2D
^^^^^^^^^^^^^^^^^^^^^^^^

.. py:class:: pyvqnet.nn.torch.ConvT2D(input_channels, output_channels, kernel_size, stride=[1, 1], padding=(0, 0), use_bias="True", kernel_initializer=None, bias_initializer=None, dilation_rate: int = 1, out_padding=(0, 0), group=1, dtype=None, name="")

    Führt eine 2D-transponierte Faltung auf der Eingabe durch. Die Eingabe für das ConvT2D-Modul hat die Form (batch_size, input_channels, height, width).
    Diese Klasse erbt von ``pyvqnet.nn.Module`` und ``torch.nn.Module`` und kann als Untermodul eines torch-Modells verwendet werden.

    Die Daten in den ``_buffers`` der Klasse sind vom Typ ``torch.Tensor``.
    Die Daten in den ``_parameters`` der Klasse sind vom Typ ``torch.nn.Parameter``.

    :param input_channels: `int` - Die Anzahl der Eingabekanäle.
    :param output_channels: `int` - Die Anzahl der Ausgabekanäle.
    :param kernel_size: `tuple|list` - Die Größe des Faltungskerns, mit Kernelform = [input_channels, output_channels/group, kernel_size, kernel_size].
    :param stride: `tuple|list` - Der Schritt, Standard ist (1, 1).
    :param padding: `tuple` - Padding-Optionen, ein Tupel, das das auf beide Seiten anzuwendende Padding angibt. Standard ist (0, 0).
    :param use_bias: `bool` - Ob der Bias-Term verwendet werden soll, Standard ist True.
    :param kernel_initializer: `callable` - Die Initialisierungsmethode für den Faltungskern. Standard ist leer, verwendet kaiming_uniform.
    :param bias_initializer: `callable` - Die Initialisierungsmethode für den Bias. Standard ist leer, verwendet kaiming_uniform.
    :param dilation_rate: `int` - Die Dilationsgröße, Standard ist 1.
    :param out_padding: Zusätzliche Größe, die zur Ausgabeform für jede Dimension hinzugefügt wird. Standard ist (0, 0).
    :param group: `int` - Die Anzahl der Gruppen in der gruppierten Faltung. Standard ist 1.
    :param dtype: Datentyp für die Parameter, standardmäßig None, verwendet den Standarddatentyp `kfloat32`, der 32-Bit-Gleitkommazahlen repräsentiert.
    :param name: Der Name des Moduls, Standard ist "".

    :return: Eine Instanz der 2D-transponierten Faltung.

    .. note::

        ``padding='valid'`` wendet kein Padding an.

        ``padding='same'`` wendet Zero-Padding auf die Eingabe an, wobei die Ausgabehöhe gleich `ceil(height / stride)` ist.

    Example::

        import numpy as np
        import pyvqnet
        from pyvqnet.tensor import QTensor
        from pyvqnet.nn.torch import ConvT2D
        pyvqnet.backends.set_backend("torch")
        test_conv = ConvT2D(3, 2, (3, 3), (1, 1))
        x = QTensor(np.arange(1, 1 * 3 * 5 * 5 + 1).reshape([1, 3, 5, 5]), requires_grad=True, dtype=pyvqnet.kfloat32)
        y = test_conv.forward(x)
        print(y)

AvgPool1D
^^^^^^^^^^^^^^^^^^^^^^^^

.. py:class:: pyvqnet.nn.torch.AvgPool1D(kernel, stride, padding=0, name = "")

    Führt einen durchschnittlichen Pooling-Vorgang auf 1D-Eingabe durch. Die Eingabe hat die Form (batch_size, input_channels, in_height).
    Diese Klasse erbt von ``pyvqnet.nn.Module`` und ``torch.nn.Module`` und kann als Untermodul zu einem torch-Modell hinzugefügt werden.

    Die Daten in den ``_buffers`` der Klasse sind vom Typ ``torch.Tensor``.
    Die Daten in den ``_parameters`` der Klasse sind vom Typ ``torch.nn.Parameter``.

    :param kernel: Die Größe des Pooling-Fensters.
    :param stride: Die Schrittgröße für die Bewegung des Fensters.
    :param padding: Padding-Option, eine ganze Zahl, die die Padding-Länge angibt. Standard ist 0.
    :param name: Der Name des Moduls, Standard ist "".

    :return: Eine Instanz der 1D-Durchschnitts-Pooling-Schicht.

    Example::

        import numpy as np
        import pyvqnet
        from pyvqnet.tensor import QTensor
        from pyvqnet.nn.torch import AvgPool1D
        pyvqnet.backends.set_backend("torch")
        test_mp = AvgPool1D([3],[2],0)
        x = QTensor(np.array([0, 1, 0, 4, 5,
                             2, 3, 2, 1, 3,
                             4, 4, 0, 4, 3,
                             2, 5, 2, 6, 4,
                             1, 0, 0, 5, 7], dtype=float).reshape([1, 5, 5]), requires_grad=True)
        y = test_mp.forward(x)
        print(y)

MaxPool1D
^^^^^^^^^^^^^^^^^^^^^^^^

.. py:class:: pyvqnet.nn.torch.MaxPool1D(kernel, stride, padding=0, name="")

    Führt einen Max-Pooling-Vorgang auf 1D-Eingabe durch. Die Eingabe hat die Form (batch_size, input_channels, in_height).
    Diese Klasse erbt von ``pyvqnet.nn.Module`` und ``torch.nn.Module`` und kann als Untermodul zu einem torch-Modell hinzugefügt werden.

    Die Daten in den ``_buffers`` der Klasse sind vom Typ ``torch.Tensor``.
    Die Daten in den ``_parameters`` der Klasse sind vom Typ ``torch.nn.Parameter``.

    :param kernel: Die Größe des Pooling-Fensters.
    :param stride: Die Schrittgröße für die Bewegung des Fensters.
    :param padding: Padding-Option, eine ganze Zahl, die die Padding-Länge angibt. Standard ist 0.
    :param name: Der Name des Moduls, Standard ist "".

    :return: Eine Instanz der 1D-Max-Pooling-Schicht.

    Example::

        import numpy as np
        import pyvqnet
        from pyvqnet.tensor import QTensor
        from pyvqnet.nn.torch import MaxPool1D
        pyvqnet.backends.set_backend("torch")
        test_mp = MaxPool1D([3],[2],0)
        x = QTensor(np.array([0, 1, 0, 4, 5,
                             2, 3, 2, 1, 3,
                             4, 4, 0, 4, 3,
                             2, 5, 2, 6, 4,
                             1, 0, 0, 5, 7], dtype=float).reshape([1, 5, 5]), requires_grad=True)
        y = test_mp.forward(x)
        print(y)

AvgPool2D
^^^^^^^^^^^^^^^^^^^^^^^^

.. py:class:: pyvqnet.nn.torch.AvgPool2D(kernel, stride, padding=(0,0), name="")

    Führt einen durchschnittlichen Pooling-Vorgang auf 2D-Eingabe durch. Die Eingabe hat die Form (batch_size, input_channels, height, width).
    Diese Klasse erbt von ``pyvqnet.nn.Module`` und ``torch.nn.Module`` und kann als Untermodul zu einem torch-Modell hinzugefügt werden.

    Die Daten in den ``_buffers`` der Klasse sind vom Typ ``torch.Tensor``.
    Die Daten in den ``_parameters`` der Klasse sind vom Typ ``torch.nn.Parameter``.

    :param kernel: Die Größe des Pooling-Fensters.
    :param stride: Die Schrittgröße für die Bewegung des Fensters.
    :param padding: Padding-Option, ein Tupel mit zwei ganzen Zahlen, die das Padding für beide Dimensionen angeben. Standard ist (0,0).
    :param name: Der Name des Moduls, Standard ist "".

    :return: Eine Instanz der 2D-Durchschnitts-Pooling-Schicht.

    Example::

        import numpy as np
        import pyvqnet
        from pyvqnet.tensor import QTensor
        from pyvqnet.nn.torch import AvgPool2D
        pyvqnet.backends.set_backend("torch")
        test_mp = AvgPool2D([2, 2], [2, 2], 1)
        x = QTensor(np.array([0, 1, 0, 4, 5,
                             2, 3, 2, 1, 3,
                             4, 4, 0, 4, 3,
                             2, 5, 2, 6, 4,
                             1, 0, 0, 5, 7], dtype=float).reshape([1, 1, 5, 5]), requires_grad=True)
        y = test_mp.forward(x)
        print(y)

MaxPool2D
^^^^^^^^^^^^^^^^^^^^^^^^

.. py:class:: pyvqnet.nn.torch.MaxPool2D(kernel, stride, padding=(0,0), name="")

    Führt einen Max-Pooling-Vorgang auf 2D-Eingabe durch. Die Eingabe hat die Form (batch_size, input_channels, height, width).
    Diese Klasse erbt von ``pyvqnet.nn.Module`` und ``torch.nn.Module`` und kann als Untermodul zu einem torch-Modell hinzugefügt werden.

    Die Daten in den ``_buffers`` der Klasse sind vom Typ ``torch.Tensor``.
    Die Daten in den ``_parameters`` der Klasse sind vom Typ ``torch.nn.Parameter``.

    :param kernel: Die Größe des Pooling-Fensters.
    :param stride: Die Schrittgröße für die Bewegung des Fensters.
    :param padding: Padding-Option, ein Tupel mit zwei ganzen Zahlen, die das Padding für beide Dimensionen angeben. Standard ist (0,0).
    :param name: Der Name des Moduls, Standard ist "".

    :return: Eine Instanz der 2D-Max-Pooling-Schicht.

    Example::

        import numpy as np
        import pyvqnet
        from pyvqnet.tensor import QTensor
        from pyvqnet.nn.torch import MaxPool2D
        pyvqnet.backends.set_backend("torch")
        test_mp = MaxPool2D([2, 2], [2, 2], (0, 0))
        x = QTensor(np.array([0, 1, 0, 4, 5,
                             2, 3, 2, 1, 3,
                             4, 4, 0, 4, 3,
                             2, 5, 2, 6, 4,
                             1, 0, 0, 5, 7], dtype=float).reshape([1, 1, 5, 5]), requires_grad=True)
        y = test_mp.forward(x)
        print(y)

Embedding
^^^^^^^^^^^^^^^^^^^^^^^^

.. py:class:: pyvqnet.nn.torch.Embedding(num_embeddings, embedding_dim, weight_initializer=xavier_normal, dtype=None, name: str = "")

    Dieses Modul wird typischerweise verwendet, um Wort-Einbettungen zu speichern und sie über Indizes abzurufen. Die Eingabe des Moduls ist eine Liste von Indizes, und die Ausgabe sind die entsprechenden Wort-Einbettungen.
    Die Eingabe dieser Schicht sollte vom Typ `kint64` sein.
    Diese Klasse erbt von ``pyvqnet.nn.Module`` und ``torch.nn.Module`` und kann als Untermodul zu einem torch-Modell hinzugefügt werden.

    Die Daten in den ``_buffers`` der Klasse sind vom Typ ``torch.Tensor``.
    Die Daten in den ``_parameters`` der Klasse sind vom Typ ``torch.nn.Parameter``.

    :param num_embeddings: `int` - Die Größe des Einbettungswörterbuchs.
    :param embedding_dim: `int` - Die Größe jedes Einbettungsvektors.
    :param weight_initializer: `callable` - Die Gewichtungsinitialisierungsmethode, Standard ist Xavier Normal.
    :param dtype: Der Datentyp für die Parameter, standardmäßig None, verwendet den Standarddatentyp: `kfloat32` (32-Bit-Gleitkommazahl).
    :param name: Der Name der Einbettungsschicht, Standard ist "".

    :return: Eine Instanz der Embedding-Schicht.

    Example::

        import numpy as np
        import pyvqnet
        from pyvqnet.tensor import QTensor
        from pyvqnet.nn.torch import Embedding
        pyvqnet.backends.set_backend("torch")
        vlayer = Embedding(30, 3)
        x = QTensor(np.arange(1, 25).reshape([2, 3, 2, 2]), dtype=pyvqnet.kint64)
        y = vlayer(x)
        print(y)

BatchNorm2d
^^^^^^^^^^^^^^^^^^^^^^^^

.. py:class:: pyvqnet.nn.torch.BatchNorm2d(channel_num:int, momentum:float=0.1, epsilon:float = 1e-5, affine=True, beta_initializer=zeros, gamma_initializer=ones, dtype=None, name="")

    Wendet Batch-Normalisierung auf 4D-Eingabe (B, C, H, W) an. Siehe dazu das Paper
    `Batch Normalization: Accelerating Deep Network Training by Reducing
    Internal Covariate Shift <https://arxiv.org/abs/1502.03167>`__.

    Diese Klasse erbt von ``pyvqnet.nn.Module`` und ``torch.nn.Module`` und kann als Untermodul zu einem torch-Modell hinzugefügt werden.

    Die Daten in den ``_buffers`` der Klasse sind vom Typ ``torch.Tensor``.
    Die Daten in den ``_parameters`` der Klasse sind vom Typ ``torch.nn.Parameter``.

    .. math::
        y = \frac{x - \mathrm{E}[x]}{\sqrt{\mathrm{Var}[x] + \epsilon}} * \gamma + \beta

    wobei :math:`\gamma` und :math:`\beta` trainierbare Parameter sind. Zusätzlich schätzt die Schicht während des Trainings standardmäßig kontinuierlich den Mittelwert und die Varianz, die dann während der Evaluation zur Normalisierung verwendet werden. Der Momentum für die gleitenden Durchschnitte ist auf den Standardwert 0.1 gesetzt.

    :param channel_num: `int` - Die Anzahl der Eingabekanäle.
    :param momentum: `float` - Momentum für die Berechnung des gleitenden Durchschnitts, Standard ist 0.1.
    :param epsilon: `float` - Eine kleine Konstante für numerische Stabilität, Standard ist 1e-5.
    :param affine: `bool` - Ob lernbare affine Parameter für jeden Kanal eingefügt werden sollen. Standard ist `True`, was die Parameter als 1 für Gewichte und 0 für Biases initialisiert.
    :param beta_initializer: `callable` - Die Initialisierungsmethode für Beta, Standard ist Null-Initialisierung.
    :param gamma_initializer: `callable` - Die Initialisierungsmethode für Gamma, Standard ist Eins-Initialisierung.
    :param dtype: Der Datentyp für die Parameter, standardmäßig None, verwendet `kfloat32` (32-Bit-Gleitkommazahl).
    :param name: Der Name der Batch-Normalisierungsschicht, Standard ist "".

    :return: Eine Instanz der 2D-Batch-Normalisierungsschicht.

    Example::

        import numpy as np
        import pyvqnet
        from pyvqnet.tensor import QTensor
        from pyvqnet.nn.torch import BatchNorm2d
        pyvqnet.backends.set_backend("torch")
        b = 2
        ic = 2
        test_conv = BatchNorm2d(ic)
        x = QTensor(np.arange(1, 17).reshape([b, ic, 4, 1]), requires_grad=True, dtype=pyvqnet.kfloat32)
        y = test_conv.forward(x)
        print(y)

BatchNorm1d
^^^^^^^^^^^^^^^^^^^^^^^^

.. py:class:: pyvqnet.nn.torch.BatchNorm1d(channel_num:int, momentum:float=0.1, epsilon:float = 1e-5, affine=True, beta_initializer=zeros, gamma_initializer=ones, dtype=None, name="")

    Wendet Batch-Normalisierung auf 2D-Eingabe (B, C) an. Siehe dazu das Paper
    `Batch Normalization: Accelerating Deep Network Training by Reducing
    Internal Covariate Shift <https://arxiv.org/abs/1502.03167>`__.

    .. math::
        y = \frac{x - \mathrm{E}[x]}{\sqrt{\mathrm{Var}[x] + \epsilon}} * \gamma + \beta

    wobei :math:`\gamma` und :math:`\beta` trainierbare Parameter sind. Zusätzlich schätzt die Schicht während des Trainings standardmäßig kontinuierlich den Mittelwert und die Varianz, die dann während der Evaluation zur Normalisierung verwendet werden. Der Momentum für die gleitenden Durchschnitte ist auf den Standardwert 0.1 gesetzt.

    Diese Klasse erbt von ``pyvqnet.nn.Module`` und ``torch.nn.Module`` und kann als Untermodul zu einem torch-Modell hinzugefügt werden.

    Die Daten in den ``_buffers`` der Klasse sind vom Typ ``torch.Tensor``.
    Die Daten in den ``_parameters`` der Klasse sind vom Typ ``torch.nn.Parameter``.

    :param channel_num: `int` - Die Anzahl der Eingabekanäle.
    :param momentum: `float` - Momentum für die Berechnung des gleitenden Durchschnitts, Standard ist 0.1.
    :param epsilon: `float` - Eine kleine Konstante für numerische Stabilität, Standard ist 1e-5.
    :param affine: `bool` - Ob lernbare affine Parameter für jeden Kanal eingefügt werden sollen. Standard ist `True`, was die Parameter als 1 für Gewichte und 0 für Biases initialisiert.
    :param beta_initializer: `callable` - Die Initialisierungsmethode für Beta, Standard ist Null-Initialisierung.
    :param gamma_initializer: `callable` - Die Initialisierungsmethode für Gamma, Standard ist Eins-Initialisierung.
    :param dtype: Der Datentyp für die Parameter, standardmäßig None, verwendet `kfloat32` (32-Bit-Gleitkommazahl).
    :param name: Der Name der Batch-Normalisierungsschicht, Standard ist "".

    :return: Eine Instanz der 1D-Batch-Normalisierungsschicht.

    Example::

        import numpy as np
        import pyvqnet
        from pyvqnet.tensor import QTensor
        from pyvqnet.nn.torch import BatchNorm1d
        pyvqnet.backends.set_backend("torch")
        test_conv = BatchNorm1d(4)
        x = QTensor(np.arange(1, 17).reshape([4, 4]), requires_grad=True, dtype=pyvqnet.kfloat32)
        y = test_conv.forward(x)
        print(y)

LayerNormNd
^^^^^^^^^^^^^^^^^^^^^^^^

.. py:class:: pyvqnet.nn.torch.LayerNormNd(normalized_shape: list, epsilon: float = 1e-5, affine=True, dtype=None, name="")

    Wendet Layer-Normalisierung auf die letzten D Dimensionen einer beliebigen Eingabe an. Die genaue Methode wird in folgendem Paper beschrieben:
    `Layer Normalization <https://arxiv.org/abs/1607.06450>`__.

    .. math::
        y = \frac{x - \mathrm{E}[x]}{ \sqrt{\mathrm{Var}[x] + \epsilon}} * \gamma + \beta

    Für Eingaben wie (B, C, H, W, D) kann ``norm_shape`` [C, H, W, D], [H, W, D], [W, D] oder [D] sein.

    Diese Klasse erbt von ``pyvqnet.nn.Module`` und ``torch.nn.Module`` und kann als Untermodul zu einem torch-Modell hinzugefügt werden.

    Die Daten in den ``_buffers`` der Klasse sind vom Typ ``torch.Tensor``.
    Die Daten in den ``_parameters`` der Klasse sind vom Typ ``torch.nn.Parameter``.

    :param norm_shape: `list` - Die zu normalisierende Form.
    :param epsilon: `float` - Eine kleine Konstante für numerische Stabilität, Standard ist 1e-5.
    :param affine: `bool` - Wenn `True`, hat dieses Modul lernbare affine Parameter für jeden Kanal, initialisiert mit 1 (für Gewichte) und 0 (für Biases). Standard ist `True`.
    :param dtype: Der Datentyp für die Parameter, standardmäßig None, verwendet `kfloat32` (32-Bit-Gleitkommazahl).
    :param name: Der Name des Moduls, Standard ist "".

    :return: Eine Instanz der LayerNormNd-Klasse.

    Example::

        import numpy as np
        from pyvqnet.tensor import QTensor
        from pyvqnet import kfloat32
        from pyvqnet.nn.torch import LayerNormNd
        import pyvqnet
        pyvqnet.backends.set_backend("torch")
        ic = 4
        test_conv = LayerNormNd([2,2])
        x = QTensor(np.arange(1,17).reshape([2,2,2,2]), requires_grad=True, dtype=kfloat32)
        y = test_conv.forward(x)
        print(y)

LayerNorm2d
^^^^^^^^^^^^^^^^^^^^^^^^

.. py:class:: pyvqnet.nn.torch.LayerNorm2d(norm_size:int, epsilon:float = 1e-5, affine=True, dtype=None, name="")

    Wendet Layer-Normalisierung auf 4D-Eingaben an. Die genaue Methode wird in folgendem Paper beschrieben:
    `Layer Normalization <https://arxiv.org/abs/1607.06450>`__.

    .. math::
        y = \frac{x - \mathrm{E}[x]}{ \sqrt{\mathrm{Var}[x] + \epsilon}} * \gamma + \beta

    Mittelwert und Standardabweichung werden über die verbleibenden Dimensionen (außer der ersten) berechnet. Für Eingaben wie (B, C, H, W) sollte ``norm_size`` gleich C * H * W sein.

    Diese Klasse erbt von ``pyvqnet.nn.Module`` und ``torch.nn.Module`` und kann als Untermodul zu einem torch-Modell hinzugefügt werden.

    Die Daten in den ``_buffers`` der Klasse sind vom Typ ``torch.Tensor``.
    Die Daten in den ``_parameters`` der Klasse sind vom Typ ``torch.nn.Parameter``.

    :param norm_size: `int` - Die Größe der Normalisierung, sollte gleich C * H * W sein.
    :param epsilon: `float` - Eine kleine Konstante für numerische Stabilität, Standard ist 1e-5.
    :param affine: `bool` - Wenn `True`, hat dieses Modul lernbare affine Parameter für jeden Kanal, initialisiert mit 1 (für Gewichte) und 0 (für Biases). Standard ist `True`.
    :param dtype: Der Datentyp für die Parameter, standardmäßig None, verwendet `kfloat32` (32-Bit-Gleitkommazahl).
    :param name: Der Name des Moduls, Standard ist "".

    :return: Eine Instanz der 2D-Layer-Normalisierung.

    Example::

        import numpy as np
        import pyvqnet
        from pyvqnet.tensor import QTensor
        from pyvqnet.nn.torch import LayerNorm2d
        import pyvqnet
        pyvqnet.backends.set_backend("torch")
        ic = 4
        test_conv = LayerNorm2d(8)
        x = QTensor(np.arange(1,17).reshape([2,2,4,1]), requires_grad=True, dtype=pyvqnet.kfloat32)
        y = test_conv.forward(x)
        print(y)

LayerNorm1d
^^^^^^^^^^^^^^^^^^^^^^^^

.. py:class:: pyvqnet.nn.torch.LayerNorm1d(norm_size:int, epsilon:float = 1e-5, affine=True, dtype=None, name="")

    Wendet Layer-Normalisierung auf 2D-Eingaben an. Die genaue Methode wird in folgendem Paper beschrieben:
    `Layer Normalization <https://arxiv.org/abs/1607.06450>`__.

    .. math::
        y = \frac{x - \mathrm{E}[x]}{ \sqrt{\mathrm{Var}[x] + \epsilon}} * \gamma + \beta

    Mittelwert und Standardabweichung werden über die letzte Dimensionsgröße berechnet, wobei ``norm_size`` der Wert der letzten Dimension ist.

    Diese Klasse erbt von ``pyvqnet.nn.Module`` und ``torch.nn.Module`` und kann als Untermodul zu einem torch-Modell hinzugefügt werden.

    Die Daten in den ``_buffers`` der Klasse sind vom Typ ``torch.Tensor``.
    Die Daten in den ``_parameters`` der Klasse sind vom Typ ``torch.nn.Parameter``.

    :param norm_size: `int` - Die Größe der Normalisierung, sollte gleich der Größe der letzten Dimension sein.
    :param epsilon: `float` - Eine kleine Konstante für numerische Stabilität, Standard ist 1e-5.
    :param affine: `bool` - Wenn `True`, hat dieses Modul lernbare affine Parameter für jeden Kanal, initialisiert mit 1 (für Gewichte) und 0 (für Biases). Standard ist `True`.
    :param dtype: Der Datentyp für die Parameter, standardmäßig None, verwendet `kfloat32` (32-Bit-Gleitkommazahl).
    :param name: Der Name des Moduls, Standard ist "".

    :return: Eine Instanz der 1D-Layer-Normalisierung.

    Example::

        import numpy as np
        from pyvqnet.tensor import QTensor
        from pyvqnet.nn.torch import LayerNorm1d
        import pyvqnet
        pyvqnet.backends.set_backend("torch")
        test_conv = LayerNorm1d(4)
        x = QTensor(np.arange(1,17).reshape([4,4]), requires_grad=True, dtype=pyvqnet.kfloat32)
        y = test_conv.forward(x)
        print(y)

GroupNorm
^^^^^^^^^^^^^^^^^^^^^^^^

.. py:class:: pyvqnet.nn.torch.GroupNorm(num_groups: int, num_channels: int, epsilon = 1e-5, affine = True, dtype = None, name = "")

    Wendet Group-Normalisierung auf Mini-Batch-Eingaben an. Eingabe: :math:`(N, C, *)` wobei :math:`C=\mathrm{num\_channels}`, Ausgabe: :math:`(N, C, *)`.

    Diese Schicht implementiert die im Paper `Group Normalization <https://arxiv.org/abs/1803.08494>`__ beschriebene Operation.

    .. math::
        
        y = \frac{x - \mathrm{E}[x]}{ \sqrt{\mathrm{Var}[x] + \epsilon}} * \gamma + \beta

    Die Eingabekanäle werden in :attr:`num_groups` Gruppen aufgeteilt, die jeweils ``num_channels / num_groups`` Kanäle enthalten. :attr:`num_channels` muss durch :attr:`num_groups` teilbar sein. Mittelwert und Standardabweichung werden innerhalb jeder Gruppe separat berechnet. Wenn :attr:`affine` ``True`` ist, dann sind :math:`\gamma` und :math:`\beta` lernbar. Die affinen Transformationsparameter für jeden Kanal sind Vektoren der Größe :attr:`num_channels`.

    Diese Klasse erbt von ``pyvqnet.nn.Module`` und ``torch.nn.Module`` und kann als Untermodul zu einem torch-Modell hinzugefügt werden.

    Die Daten in den ``_buffers`` der Klasse sind vom Typ ``torch.Tensor``.
    Die Daten in den ``_parameters`` der Klasse sind vom Typ ``torch.nn.Parameter``.

    :param num_groups (int): Die Anzahl der Gruppen, in die die Kanäle aufgeteilt werden.
    :param num_channels (int): Die Anzahl der erwarteten Eingabekanäle.
    :param epsilon: Ein kleiner Wert, der zum Nenner für numerische Stabilität hinzugefügt wird. Standard ist 1e-5.
    :param affine: Ein boolescher Wert. Wenn auf ``True`` gesetzt, hat dieses Modul lernbare affine Parameter für jeden Kanal, initialisiert mit 1 (für Gewichte) und 0 (für Biases). Standard ist ``True``.
    :param dtype: Der Datentyp für die Parameter. Standardmäßig None, verwendet `kfloat32` (32-Bit-Gleitkommazahl).
    :param name: Der Name des Moduls. Standard ist "".

    :return: Eine Instanz der GroupNorm-Klasse.

    Example::

        import numpy as np
        from pyvqnet.tensor import QTensor, kfloat32
        from pyvqnet.nn.torch import GroupNorm
        import pyvqnet
        pyvqnet.backends.set_backend("torch")
        test_conv = GroupNorm(2, 10)
        x = QTensor(np.arange(0, 60*2*5).reshape([2, 10, 3, 2, 5]), requires_grad=True, dtype=kfloat32)
        y = test_conv.forward(x)
        print(y)

Dropout
^^^^^^^^^^^^^^^^^^^^^^^^

.. py:class:: pyvqnet.nn.torch.Dropout(dropout_rate = 0.5)

    Dropout-Modul. Das Dropout-Modul setzt die Ausgabe zufällig ausgewählter Einheiten auf Null und skaliert die verbleibenden Einheiten entsprechend der angegebenen dropout_rate Wahrscheinlichkeit.

    Diese Klasse erbt von ``pyvqnet.nn.Module`` und ``torch.nn.Module`` und kann als Untermodul zu einem torch-Modell hinzugefügt werden.

    :param dropout_rate: `float` - Die Wahrscheinlichkeit, Neuronen auf Null zu setzen.
    :param name: Der Name des Moduls. Standard ist "".

    :return: Eine Instanz der Dropout-Klasse.

    Example::

        import numpy as np
        from pyvqnet.nn.torch import Dropout
        from pyvqnet.tensor import arange
        import pyvqnet
        pyvqnet.backends.set_backend("torch")
        b = 2
        ic = 2
        x = arange(-1 * ic * 2 * 2.0, (b - 1) * ic * 2 * 2).reshape([b, ic, 2, 2])
        droplayer = Dropout(0.5)
        droplayer.train()
        y = droplayer(x)
        print(y)

DropPath
^^^^^^^^^^^^^^^^^^^^^^^^

.. py:class:: pyvqnet.nn.torch.DropPath(dropout_rate = 0.5, name="")

    DropPath-Modul wendet zufälliges Sample-Pfad-Dropout (zufällige Tiefe) an.

    Diese Klasse erbt von ``pyvqnet.nn.Module`` und ``torch.nn.Module`` und kann als Untermodul zu einem torch-Modell hinzugefügt werden.

    :param dropout_rate: `float` - Die Wahrscheinlichkeit, Neuronen auf Null zu setzen.
    :param name: Der Name des Moduls. Standard ist "".

    :return: Eine Instanz der DropPath-Klasse.

    Example::

        import pyvqnet.nn.torch as nn
        import pyvqnet.tensor as tensor
        import pyvqnet
        pyvqnet.backends.set_backend("torch")
        x = tensor.randu([4])
        y = nn.DropPath()(x)
        print(y)

Pixel_Shuffle
^^^^^^^^^^^^^^^^^^^^^^^^

.. py:class:: pyvqnet.nn.torch.Pixel_Shuffle(upscale_factors, name="")

    Ordnet einen Tensor der Form (*, C * r^2, H, W) in einen Tensor der Form (*, C, H * r, W * r) um, wobei r der Skalierungsfaktor ist.

    Diese Klasse erbt von ``pyvqnet.nn.Module`` und ``torch.nn.Module`` und kann als Untermodul zu einem torch-Modell hinzugefügt werden.

    :param upscale_factors: Der Skalierungsfaktor für die Transformation.
    :param name: Der Name des Moduls. Standard ist "".

    :return: Eine Instanz des Pixel_Shuffle-Moduls.

    Example::

        from pyvqnet.nn.torch import Pixel_Shuffle
        from pyvqnet.tensor import tensor
        import pyvqnet
        pyvqnet.backends.set_backend("torch")
        ps = Pixel_Shuffle(3)
        inx = tensor.ones([5, 2, 3, 18, 4, 4])
        inx.requires_grad = True
        y = ps(inx)

Pixel_Unshuffle
^^^^^^^^^^^^^^^^^^^^^^^^

.. py:class:: pyvqnet.nn.torch.Pixel_Unshuffle(downscale_factors, name="")

    Kehrt die Pixel_Shuffle-Operation durch Neuordnung der Elemente um. Wandelt einen Tensor der Form (*, C, H * r, W * r) in (*, C * r^2, H, W) um, wobei r der Herunterskalierungsfaktor ist.

    Diese Klasse erbt von ``pyvqnet.nn.Module`` und ``torch.nn.Module`` und kann als Untermodul zu einem torch-Modell hinzugefügt werden.

    :param downscale_factors: Der Herunterskalierungsfaktor für die Transformation.
    :param name: Der Name des Moduls. Standard ist "".

    :return: Eine Instanz des Pixel_Unshuffle-Moduls.

    Example::

        from pyvqnet.nn.torch import Pixel_Unshuffle
        from pyvqnet.tensor import tensor
        import pyvqnet
        pyvqnet.backends.set_backend("torch")
        ps = Pixel_Unshuffle(3)
        inx = tensor.ones([5, 2, 3, 2, 12, 12])
        inx.requires_grad = True
        y = ps(inx)

GRU
^^^^^^^^^^^^^^^^^^^^^^^^

.. py:class:: pyvqnet.nn.torch.GRU(input_size, hidden_size, num_layers=1, nonlinearity='tanh', batch_first=True, use_bias=True, bidirectional=False, dtype=None, name: str = "")

    Gated Recurrent Unit (GRU)-Modul. Unterstützt mehrschichtige Stapelung und bidirektionale Konfiguration. Die Formel für ein einschichtiges unidirektionales GRU lautet:

    .. math::
        \begin{array}{ll}
            r_t = \sigma(W_{ir} x_t + b_{ir} + W_{hr} h_{(t-1)} + b_{hr}) \\
            z_t = \sigma(W_{iz} x_t + b_{iz} + W_{hz} h_{(t-1)} + b_{hz}) \\
            n_t = \tanh(W_{in} x_t + b_{in} + r_t * (W_{hn} h_{(t-1)}+ b_{hn})) \\
            h_t = (1 - z_t) * n_t + z_t * h_{(t-1)}
        \end{array}

    Diese Klasse erbt von ``pyvqnet.nn.Module`` und ``torch.nn.Module`` und kann als Untermodul zu einem torch-Modell hinzugefügt werden.

    Die ``_buffers`` der Klasse enthalten ``torch.Tensor``-Daten, und die ``_parameters`` der Klasse enthalten ``torch.nn.Parameter``-Daten.

    :param input_size: Die Eingabemerkmal-Dimension.
    :param hidden_size: Die verborgene Merkmal-Dimension.
    :param num_layers: Die Anzahl der gestapelten GRU-Schichten, Standard: 1.
    :param batch_first: Wenn True, ist die Eingabeform [batch_size, seq_len, feature_dim], wenn False, ist die Form [seq_len, batch_size, feature_dim], Standard: True.
    :param use_bias: Wenn False, verwendet das Modul keine Bias-Terme, Standard: True.
    :param bidirectional: Wenn True, wird das GRU bidirektional, Standard: False.
    :param dtype: Der Datentyp der Parameter, standardmäßig None, verwendet den Standarddatentyp: kfloat32 (32-Bit-Gleitkommazahl).
    :param name: Der Name des Moduls, Standard: "".

    :return: Eine Instanz des GRU-Moduls.

    Example::

        import pyvqnet
        pyvqnet.backends.set_backend("torch")
        from pyvqnet.nn.torch import GRU
        from pyvqnet.tensor import tensor

        rnn2 = GRU(4, 6, 2, batch_first=False, bidirectional=True)

        input = tensor.ones([5, 3, 4])
        h0 = tensor.ones([4, 3, 6])

        output, hn = rnn2(input, h0)



RNN
^^^^^^^^^^^^^^^^^^^^^^^^

.. py:class:: pyvqnet.nn.torch.RNN(input_size, hidden_size, num_layers=1, nonlinearity='tanh', batch_first=True, use_bias=True, bidirectional=False, dtype=None, name: str = "")

    Modul für rekurrente neuronale Netze (RNN), das :math:`\tanh` oder :math:`\text{ReLU}` als Aktivierungsfunktion verwendet. Unterstützt bidirektionale und mehrschichtige Konfigurationen. Die Formel für ein einschichtiges unidirektionales RNN lautet:

    .. math::
        h_t = \tanh(W_{ih} x_t + b_{ih} + W_{hh} h_{(t-1)} + b_{hh})

    Wenn :attr:`nonlinearity` ``'relu'`` ist, ersetzt :math:`\text{ReLU}` :math:`\tanh`.

    Diese Klasse erbt von ``pyvqnet.nn.Module`` und ``torch.nn.Module`` und kann als Untermodul zu einem torch-Modell hinzugefügt werden.

    Die ``_buffers`` der Klasse enthalten ``torch.Tensor``-Daten, und die ``_parameters`` der Klasse enthalten ``torch.nn.Parameter``-Daten.

    :param input_size: Die Eingabemerkmal-Dimension.
    :param hidden_size: Die verborgene Merkmal-Dimension.
    :param num_layers: Die Anzahl der gestapelten RNN-Schichten, Standard: 1.
    :param nonlinearity: Die nichtlineare Aktivierungsfunktion, Standard: ``'tanh'``.
    :param batch_first: Wenn True, ist die Eingabeform [batch_size, seq_len, feature_dim], wenn False, ist die Form [seq_len, batch_size, feature_dim], Standard: True.
    :param use_bias: Wenn False, verwendet das Modul keine Bias-Terme, Standard: True.
    :param bidirectional: Wenn True, wird das RNN bidirektional, Standard: False.
    :param dtype: Der Datentyp der Parameter, standardmäßig None, verwendet den Standarddatentyp: kfloat32 (32-Bit-Gleitkommazahl).
    :param name: Der Name des Moduls, Standard: "".

    :return: Eine Instanz des RNN-Moduls.

    Example::

        import pyvqnet
        pyvqnet.backends.set_backend("torch")
        from pyvqnet.nn.torch import RNN
        from pyvqnet.tensor import tensor

        rnn2 = RNN(4, 6, 2, batch_first=False, bidirectional = True)

        input = tensor.ones([5, 3, 4])
        h0 = tensor.ones([4, 3, 6])
        output, hn = rnn2(input, h0)

LSTM
^^^^^^^^^^^^^^^^^^^^^^^^

.. py:class:: pyvqnet.nn.torch.LSTM(input_size, hidden_size, num_layers=1, batch_first=True, use_bias=True, bidirectional=False, dtype=None, name: str = "")

    Long Short-Term Memory (LSTM)-Modul. Unterstützt bidirektionales LSTM und gestapeltes mehrschichtiges LSTM. Die Formel für ein einschichtiges unidirektionales LSTM lautet wie folgt:

    .. math::
        \begin{array}{ll} \\
            i_t = \sigma(W_{ii} x_t + b_{ii} + W_{hi} h_{t-1} + b_{hi}) \\
            f_t = \sigma(W_{if} x_t + b_{if} + W_{hf} h_{t-1} + b_{hf}) \\
            g_t = \tanh(W_{ig} x_t + b_{ig} + W_{hg} h_{t-1} + b_{hg}) \\
            o_t = \sigma(W_{io} x_t + b_{io} + W_{ho} h_{t-1} + b_{ho}) \\
            c_t = f_t \odot c_{t-1} + i_t \odot g_t \\
            h_t = o_t \odot \tanh(c_t) \\
        \end{array}

    Diese Klasse erbt von ``pyvqnet.nn.Module`` und ``torch.nn.Module`` und kann als Untermodul zu einem torch-Modell hinzugefügt werden.

    Die ``_buffers`` der Klasse enthalten ``torch.Tensor``-Daten, und die ``_parameters`` der Klasse enthalten ``torch.nn.Parameter``-Daten.

    :param input_size: Die Eingabemerkmal-Dimension.
    :param hidden_size: Die verborgene Merkmal-Dimension.
    :param num_layers: Die Anzahl der gestapelten LSTM-Schichten, Standard: 1.
    :param batch_first: Wenn True, ist die Eingabeform [batch_size, seq_len, feature_dim], wenn False, ist die Form [seq_len, batch_size, feature_dim], Standard: True.
    :param use_bias: Wenn False, verwendet das Modul keine Bias-Terme, Standard: True.
    :param bidirectional: Wenn True, wird das LSTM bidirektional, Standard: False.
    :param dtype: Der Datentyp der Parameter, standardmäßig None, verwendet den Standarddatentyp: kfloat32 (32-Bit-Gleitkommazahl).
    :param name: Der Name des Moduls, Standard: "".

    :return: Eine Instanz des LSTM-Moduls.

    Example::

        import pyvqnet
        pyvqnet.backends.set_backend("torch")
        from pyvqnet.nn.torch import LSTM
        from pyvqnet.tensor import tensor

        rnn2 = LSTM(4, 6, 2, batch_first=False, bidirectional = True)

        input = tensor.ones([5, 3, 4])
        h0 = tensor.ones([4, 3, 6])
        c0 = tensor.ones([4, 3, 6])
        output, (hn, cn) = rnn2(input, (h0, c0))


Dynamic_GRU
^^^^^^^^^^^^^^^^^^^^^^^^

.. py:class:: pyvqnet.nn.torch.Dynamic_GRU(input_size,hidden_size, num_layers=1, batch_first=True, use_bias=True, bidirectional=False, dtype=None, name: str = "")

    Wendet ein mehrschichtiges Gated Recurrent Unit (GRU)-RNN auf Eingabesequenzen variabler Länge an.

    Die erste Eingabe sollte eine Batch-Sequenz-Eingabe mit variabler Länge sein, definiert
    über eine ``tensor.PackedSequence``-Klasse.

    Die ``tensor.PackedSequence``-Klasse kann durch aufeinanderfolgendes Aufrufen der folgenden Funktionen konstruiert werden: ``pad_sequence``, ``pack_pad_sequence``.

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

    Diese Klasse erbt von ``pyvqnet.nn.Module`` und ``torch.nn.Module`` und kann als Untermodul von ``torch.nn.Module`` zum torch-Modell hinzugefügt werden.

    The data in ``_buffers`` of this class is of ``torch.Tensor`` type.

    The data in ``_parameters`` of this class is of ``torch.nn.Parameter`` type.

    :param input_size: Input feature dimension.
    :param hidden_size: Hidden feature dimension.
    :param num_layers: Number of loop layers. Default value: 1
    :param batch_first: If True, the input shape is provided as [batch size, sequence length, feature dimension]. If False, the input shape is provided as [sequence length, batch size, feature dimension]. Default value: True.
    :param use_bias: If False, the bias weights b_ih and b_hh are not used for this layer. Default value: True.
    :param bidirectional: If true, it becomes a bidirectional GRU. Default value: False.
    :param dtype: The data type of the parameter, default: None, use the default data type: kfloat32, representing 32-bit floating point numbers.
    :param name: Der Name dieses Moduls, Standard ist "".

    :return: A Dynamic_GRU class

    Example::

        import pyvqnet
        pyvqnet.backends.set_backend("torch")
        from pyvqnet.nn.torch import Dynamic_GRU
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

Dynamic_RNN 
^^^^^^^^^^^^^^^^^^^^^^^^

.. py:class:: pyvqnet.nn.torch.Dynamic_RNN(input_size, hidden_size, num_layers=1, nonlinearity='tanh', batch_first=True, use_bias=True, bidirectional=False, dtype=None, name: str = "")


    Wendet ein rekurrentes neuronales Netzwerk (RNN) auf eine Eingabesequenz variabler Länge an.

    The first input should be a batch sequence input with variable length defined
    via the ``tensor.PackedSequence`` class.

    The ``tensor.PackedSequence`` class can be constructed by
    calling the next function in succession: ``pad_sequence``, ``pack_pad_sequence``.

    The first output of Dynamic_RNN is also a ``tensor.PackedSequence`` class,
    which can be unpacked to a normal QTensor using ``tensor.pad_pack_sequence``.

    Recurrent neural network (RNN) module, using :math:`\tanh` or :math:`\text{ReLU}` as activation function. Supports bidirectional, multi-layer configurations.
    The calculation formula of single-layer unidirectional RNN is as follows:

    .. math::
        h_t = \tanh(W_{ih} x_t + b_{ih} + W_{hh} h_{(t-1)} + b_{hh})

    If :attr:`nonlinearity` is ``'relu'``, then :math:`\text{ReLU}` will replace :math:`\tanh`.

    Diese Klasse erbt von ``pyvqnet.nn.Module`` und ``torch.nn.Module`` und kann als Untermodul von ``torch.nn.Module`` zum torch-Modell hinzugefügt werden.

    Die Daten in ``_buffers`` dieser Klasse sind vom Typ ``torch.Tensor``.

    Die Daten in ``_parameters`` dieser Klasse sind vom Typ ``torch.nn.Parameter``.

    :param input_size: Input feature dimension.
    :param hidden_size: Hidden feature dimension.
    :param num_layers: Number of stacked RNN layers, default: 1.
    :param nonlinearity: Nonlinear activation function, default is ``'tanh'``.
    :param batch_first: If True, the input shape is [batch size, sequence length, feature dimension],If False, the input shape is [sequence length, batch size, feature dimension], default is True.
    :param use_bias: If False, this module does not apply bias, default: True.
    :param bidirectional: If True, it becomes a bidirectional RNN, default: False.
    :param dtype: The data type of the parameter, default: None, use the default data type: kfloat32, representing 32-bit floating point numbers.
    :param name: Der Name dieses Moduls, Standard ist "".

    :return: Dynamic_RNN instance

    Example::

        import pyvqnet
        pyvqnet.backends.set_backend("torch")
        from pyvqnet.nn.torch import Dynamic_RNN
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




Dynamic_LSTM
^^^^^^^^^^^^^^^^^^^^^^^^

.. py:class:: pyvqnet.nn.torch.Dynamic_LSTM(input_size, hidden_size, num_layers=1, batch_first=True, use_bias=True, bidirectional=False, dtype=None, name: str = "")


    Wendet ein Long Short-Term Memory (LSTM)-RNN auf Eingabesequenzen variabler Länge an.

    The first input should be a batch sequence input with variable length defined
    via a ``tensor.PackedSequence`` class.

    The ``tensor.PackedSequence`` class can be constructed by
    calling the next functions in succession: ``pad_sequence``, ``pack_pad_sequence``.

    The first output of Dynamic_LSTM is also a ``tensor.PackedSequence`` class,
    which can be unpacked to a normal QTensor using ``tensor.pad_pack_sequence``.

    Recurrent Neural Network (RNN) module, using :math:`\tanh` or :math:`\text{ReLU}` as activation function. Supports bidirectional, multi-layer configurations.
    The calculation formula of single-layer one-way RNN is as follows: 
    
    
    .. math::
        \begin{array}{ll} \\
            i_t = \sigma(W_{ii} x_t + b_{ii} + W_{hi} h_{t-1} + b_{hi}) \\
            f_t = \sigma(W_{if} x_t + b_{if} + W_{hf} h_{t-1} + b_{hf}) \\
            g_t = \tanh(W_{ig} x_t + b_{ig} + W_{hg} h_{t-1} + b_{hg}) \\
            o_t = \sigma(W_{io} x_t + b_{io} + W_{ho} h_{t-1} + b_{ho}) \\
            c_t = f_t \odot c_{t-1} + i_t \odot g_t \\
            h_t = o_t \odot \tanh(c_t) \\
        \end{array}

    Diese Klasse erbt von ``pyvqnet.nn.Module`` und ``torch.nn.Module`` und kann als Untermodul von ``torch.nn.Module`` zum torch-Modell hinzugefügt werden.

    Die Daten in ``_buffers`` dieser Klasse sind vom Typ ``torch.Tensor``.

    Die Daten in ``_parameters`` dieser Klasse sind vom Typ ``torch.nn.Parameter``.

    :param input_size: Input feature dimension.
    :param hidden_size: Hidden feature dimension.
    :param num_layers: Number of stacked LSTM layers, default: 1.
    :param batch_first: If True, the input shape is [batch size, sequence length, feature dimension],If False, the input shape is [sequence length, batch size, feature dimension], default is True.
    :param use_bias: If False, this module does not apply bias, default: True.
    :param bidirectional: If True, it becomes a bidirectional LSTM, default: False.
    :param dtype: The data type of the parameter, default: None, use the default data type: kfloat32, representing 32-bit floating point numbers.
    :param name: Der Name dieses Moduls, Standard ist "".

    :return: Dynamic_LSTM instance

    Example::

        import pyvqnet
        pyvqnet.backends.set_backend("torch")
        from pyvqnet.nn.torch import Dynamic_LSTM
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

 


Interpolate
^^^^^^^^^^^^^^^^^^^^^^^^

.. py:class:: pyvqnet.nn.torch.Interpolate(size = None, scale_factor = None, mode = "nearest", align_corners = None,  recompute_scale_factor = None, name = "")

    Down/upsample the input.

    Currently only supports 4D input data.

    Die Eingabegröße wird als `B x C x H x W` interpretiert.

    The available `mode` options are ``nearest``, ``bilinear``, ``bicubic``.

    Diese Klasse erbt von ``pyvqnet.nn.Module`` und ``torch.nn.Module`` und kann als Untermodul von ``torch.nn.Module`` zum torch-Modell hinzugefügt werden.

    :param size: Output size, default is None.
    :param scale_factor: Scaling factor, default is None.
    :param mode: Algorithm used for upsampling ``nearest`` | ``bilinear`` | ``bicubic``.
    :param align_corners: From a geometric point of view, we treat the pixels of the input and output as squares instead of points. The pixels of the input and output are treated as squares instead of points.If set to `true`, the input and output tensors will be aligned by the center points of their corner pixels. Corner pixel center points are aligned, and the values ​​of the corner pixels are preserved.If set to `false`, the input and output tensors will be aligned by the corner points of their corner pixels, and the values ​​of the corner pixels are preserved. Corner pixel corner points are aligned, and interpolation will use edge values ​​for padding.Values ​​outside the boundaries are padded, making this operation independent of the input size.When ``scale_factor`` remains unchanged. This only works when ``mode`` is ``bilinear``.
    :param recompute_scale_factor: Recompute the scale factor for use in the interpolation calculation. When ``scale_factor`` is passed as an argument, it will be used to calculate the output size.
    :param name: Module name.

    Example::

        from pyvqnet.nn.torch import Interpolate
        from pyvqnet.tensor import tensor
        import pyvqnet
        pyvqnet.backends.set_backend("torch")
        pyvqnet.utils.set_random_seed(1)

        mode_ = "bilinear"
        size_ = 3

        model = Interpolate(size=size_, mode=mode_)
        input_vqnet = tensor.randu((1, 1, 6, 6),
                                dtype=pyvqnet.kfloat32,
                                requires_grad=True)
        output_vqnet = model(input_vqnet)

SDPA
^^^^^^^^^^^^^^^^^^^^^^^^

.. py:class:: pyvqnet.nn.torch.SDPA(attn_mask=None,dropout_p=0.,scale=None,is_causal=False)

    Konstruiert eine Klasse, die skalierte Punkt-Produkt-Aufmerksamkeit für Query-, Key- und Value-Tensoren berechnet.

    Diese Klasse erbt von ``pyvqnet.nn.Module`` und ``torch.nn.Module`` und kann als Untermodul von ``torch.nn.Module`` zum torch-Modell hinzugefügt werden.

    :param attn_mask: Aufmerksamkeitsmaske; Standardwert: None. Die Form muss mit der Form der Aufmerksamkeitsgewichte broadcast-kompatibel sein.
    :param dropout_p: Dropout-Wahrscheinlichkeit; Standardwert: 0, wenn größer als 0.0, wird Dropout angewendet.
    :param scale: Skalierungsfaktor, der vor Softmax angewendet wird, Standardwert: None.
    :param is_causal: Standardwert: False, wenn auf true gesetzt, ist die Aufmerksamkeitsmaske eine untere Dreiecksmatrix, wenn die Maske eine quadratische Matrix ist. Wenn sowohl attn_mask als auch is_causal gesetzt sind, wird ein Fehler ausgelöst.
    :return: Eine SDPA-Klasse

    Examples::
    
        from pyvqnet.nn.torch import SDPA
        from pyvqnet import tensor
        import pyvqnet
        pyvqnet.backends.set_backend("torch")
        model = SDPA(tensor.QTensor([1.]))

   .. py:method:: forward(query,key,value)

        Führt die Vorwärtsberechnung durch.

        :param query: Der Query-Eingabe-QTensor.
        :param key: Der Key-Eingabe-QTensor.
        :param value: Der Key-Eingabe-QTensor.
        :return: Der von der SDPA-Berechnung zurückgegebene QTensor.
        
        Examples::
        
            from pyvqnet.nn.torch import SDPA
            from pyvqnet import tensor
            import pyvqnet
            pyvqnet.backends.set_backend("torch")

            import numpy as np

            model = SDPA(tensor.QTensor([1.]))

            query_np = np.random.randn(3, 3, 3, 5).astype(np.float32) 
            key_np = np.random.randn(3, 3, 3, 5).astype(np.float32)   
            value_np = np.random.randn(3, 3, 3, 5).astype(np.float32) 

            query_p = tensor.QTensor(query_np, dtype=pyvqnet.kfloat32, requires_grad=True)
            key_p = tensor.QTensor(key_np, dtype=pyvqnet.kfloat32, requires_grad=True)
            value_p = tensor.QTensor(value_np, dtype=pyvqnet.kfloat32, requires_grad=True)

            out_sdpa = model(query_p, key_p, value_p)

            out_sdpa.backward()

Verlustfunktionen-API
------------------------

MeanSquaredError
^^^^^^^^^^^^^^^^^^^^^^^^

.. py:class:: pyvqnet.nn.torch.MeanSquaredError(name="")

    Berechnet den mittleren quadratischen Fehler zwischen der Eingabe :math:`x` and the target value :math:`y`.

    Wenn der quadratische Fehler durch die folgende Funktion beschrieben werden kann:

    .. math::
        \ell(x, y) = L = \{l_1,\dots,l_N\}^\top, \quad
        l_n = \left( x_n - y_n \right)^2,

    :math:`x` and :math:`y` are QTensor s of arbitrary shapes, and the root mean square error of the total :math:`n` elements is calculated as follows.

    .. math::
        \ell(x, y) =
        \operatorname{mean}(L)

    Diese Klasse erbt von ``pyvqnet.nn.Module`` und ``torch.nn.Module`` und kann als Untermodul von ``torch.nn.Module`` zum torch-Modell hinzugefügt werden.

    :param name: Der Name dieses Moduls, Standard ist "".
    :return: Eine RMS-Fehler-Instanz.

    Erforderliche Parameter für die RMS-Fehler-Vorwärtsberechnungsfunktion:

        x: :math:`(N, *)` predicted value, where :math:`*` represents any dimension.

        y: :math:`(N, *)`, target value, a QTensor of the same dimension as the input.

    .. note::

        Bitte beachten Sie, dass im Gegensatz zu Frameworks wie PyTorch bei der Forward-Funktion der folgenden MeanSquaredError-Funktion der erste Parameter der Zielwert und der zweite Parameter der Vorhersagewert ist.


    Example::

        from pyvqnet.tensor import QTensor
        from pyvqnet import kfloat64
        from pyvqnet.nn.torch import MeanSquaredError
        import pyvqnet
        pyvqnet.backends.set_backend("torch")
        y = QTensor([[0, 0, 1, 0, 0, 0, 0, 0, 0, 0]],
                    requires_grad=False,
                    dtype=kfloat64)
        x = QTensor([[0.1, 0.05, 0.7, 0, 0.05, 0.1, 0, 0, 0, 0]],
                    requires_grad=True,
                    dtype=kfloat64)

        loss_result = MeanSquaredError()
        result = loss_result(y, x)
        print(result)



BinaryCrossEntropy
^^^^^^^^^^^^^^^^^^^^^^^^

.. py:class:: pyvqnet.nn.torch.BinaryCrossEntropy(name="")

    Misst den durchschnittlichen binären Kreuzentropieverlust zwischen dem Ziel und der Eingabe.

    Die binäre Kreuzentropie ohne Mittelung lautet wie folgt:

    .. math::
        \ell(x, y) = L = \{l_1,\dots,l_N\}^\top, \quad
        l_n = - w_n \left[ y_n \cdot \log x_n + (1 - y_n) \cdot \log (1 - x_n) \right],

    where :math:`N` is the batch size.

    .. math::
        \ell(x, y) = \operatorname{mean}(L)

    This class inherits from ``pyvqnet.nn.Module`` and ``torch.nn.Module`` and can be added to torch models as a submodule of ``torch.nn.Module``.

    :param name: Der Name dieses Moduls, Standard ist "".
    :return: Eine Instanz der durchschnittlichen binären Kreuzentropie.

    Erforderliche Parameter für die Durchschnitts-Binärkreuzentropie-Vorwärtsberechnungsfunktion:

        x: :math:`(N, *)` predicted value, where :math:`*` represents any dimension.

        y: :math:`(N, *)`, target value, a QTensor of the same dimension as the input.

    .. note::

        Bitte beachten Sie, dass im Gegensatz zu Frameworks wie PyTorch bei der Forward-Funktion der BinaryCrossEntropy-Funktion der erste Parameter der Zielwert und der zweite Parameter der Vorhersagewert ist.
        
    Example::

        from pyvqnet.tensor import QTensor
        from pyvqnet.nn.torch import BinaryCrossEntropy
        import pyvqnet
        pyvqnet.backends.set_backend("torch")
        x = QTensor([[0.3, 0.7, 0.2], [0.2, 0.3, 0.1]], requires_grad=True)
        y = QTensor([[0.0, 1.0, 0], [0.0, 0, 1]], requires_grad=False)

        loss_result = BinaryCrossEntropy()
        result = loss_result(y, x)
        result.backward()
        print(result)


CategoricalCrossEntropy
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. py:class:: pyvqnet.nn.torch.CategoricalCrossEntropy(name="")

    Diese Verlustfunktion kombiniert LogSoftmax und NLLLoss zur Berechnung der durchschnittlichen kategorialen Kreuzentropie.

    Die Verlustfunktion wird wie folgt berechnet, wobei class die entsprechende Kategoriebezeichnung des Zielwerts ist:

    .. math::
        \text{loss}(x, y) = -\log\left(\frac{\exp(x[class])}{\sum_j \exp(x[j])}\right)
        = -x[class] + \log\left(\sum_j \exp(x[j])\right)

    :param name: Der Name dieses Moduls, Standard ist "".
    :return: Die durchschnittliche kategoriale Kreuzentropie-Instanz.

    Erforderliche Parameter für die Fehler-Vorwärtsberechnungsfunktion:

        x: :math:`(N, *)` Predicted value, where :math:`*` indicates any dimension.

        y: :math:`(N, *)`, target value, a QTensor of the same dimension as the input. Must be a 64-bit integer, kint64.

    .. note::

        Bitte beachten Sie, dass im Gegensatz zu PyTorch und anderen Frameworks bei der Forward-Funktion der CategoricalCrossEntropy-Funktion der erste Parameter der Zielwert und der zweite Parameter der Vorhersagewert ist.

        Diese Klasse erbt von ``pyvqnet.nn.Module`` und ``torch.nn.Module`` und kann als Untermodul von ``torch.nn.Module`` zum torch-Modell hinzugefügt werden.

    Example::

        from pyvqnet.tensor import QTensor
        from pyvqnet import kfloat32,kint64
        from pyvqnet.nn.torch import CategoricalCrossEntropy
        import pyvqnet
        pyvqnet.backends.set_backend("torch")
        x = QTensor([[1, 2, 3, 4, 5],
        [1, 2, 3, 4, 5],
        [1, 2, 3, 4, 5]], requires_grad=True,dtype=kfloat32)
        y = QTensor([[0, 1, 0, 0, 0], [0, 1, 0, 0, 0], [1, 0, 0, 0, 0]], requires_grad=False,dtype=kint64)
        loss_result = CategoricalCrossEntropy()
        result = loss_result(y, x)
        print(result)



SoftmaxCrossEntropy
^^^^^^^^^^^^^^^^^^^^^^^^

.. py:class:: pyvqnet.nn.torch.SoftmaxCrossEntropy(name="")

    Diese Verlustfunktion kombiniert LogSoftmax und NLLLoss zur Berechnung der durchschnittlichen Klassifikations-Kreuzentropie und bietet eine höhere numerische Stabilität.

    The loss function is calculated as follows, where class is the corresponding classification label of the target value:

    .. math::
        \text{loss}(x, y) = -\log\left(\frac{\exp(x[class])}{\sum_j \exp(x[j])}\right)
        = -x[class] + \log\left(\sum_j \exp(x[j])\right)

    :param name: Der Name dieses Moduls, Standard ist "".
    :return: Eine Softmax-Kreuzentropie-Verlustfunktionsinstanz

    Erforderliche Parameter für die Fehler-Vorwärtsberechnungsfunktion:

        x: :math:`(N, *)` predicted value, where :math:`*` indicates any dimension.

        y: :math:`(N, *)`, target value, a QTensor of the same dimension as the input. Must be a 64-bit integer, kint64.

    .. note::

        Bitte beachten Sie, dass im Gegensatz zu PyTorch und anderen Frameworks bei der Forward-Funktion der SoftmaxCrossEntropy-Funktion der erste Parameter der Zielwert und der zweite Parameter der Vorhersagewert ist.

        Diese Klasse erbt von ``pyvqnet.nn.Module`` und ``torch.nn.Module`` und kann als Untermodul von ``torch.nn.Module`` zum torch-Modell hinzugefügt werden.
        
    Example::

        from pyvqnet.tensor import QTensor
        from pyvqnet import kfloat32, kint64
        from pyvqnet.nn.torch import SoftmaxCrossEntropy
        import pyvqnet
        pyvqnet.backends.set_backend("torch")
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



NLL_Loss
^^^^^^^^^^^^^^^^^^^^^^^^

.. py:class:: pyvqnet.nn.torch.NLL_Loss(name="")

    
    Durchschnittlicher negativer Log-Likelihood-Verlust. Nützlich für Klassifikationsprobleme mit C Klassen.

    `x` is the probability likelihood given by the model. Its shape can be :math:`(N, C)` or :math:`(N, C, d_1, d_2, ..., d_K)`. `y` is the expected true value of the loss function, containing class indices in :math:`[0, C-1]`.

    .. math::
        \ell(x, y) = L = \{l_1,\dots,l_N\}^\top, \quad
        l_n = -
        \sum_{n=1}^N \frac{1}{N}x_{n,y_n} \quad

    :param name: Der Name dieses Moduls, Standard ist "".
    :return: Eine NLL_Loss-Verlustfunktionsinstanz

    Erforderliche Parameter für die Fehler-Vorwärtsberechnungsfunktion:

        x: :math:`(N, *)`, the output prediction value of the loss function, which can be a multi-dimensional variable.

        y: :math:`(N, *)`, the target value of the loss function. Must be a 64-bit integer, kint64.

    .. note::

        Bitte beachten Sie, dass im Gegensatz zu Frameworks wie PyTorch bei der Forward-Funktion der NLL_Loss-Funktion der erste Parameter der Zielwert und der zweite Parameter der Vorhersagewert ist.

        Diese Klasse erbt von ``pyvqnet.nn.Module`` und ``torch.nn.Module`` und kann als Untermodul von ``torch.nn.Module`` zum torch-Modell hinzugefügt werden.
            
    Example::

        from pyvqnet.tensor import QTensor
        from pyvqnet import kint64
        from pyvqnet.nn.torch import NLL_Loss
        import pyvqnet
        pyvqnet.backends.set_backend("torch")
        x = QTensor([
            0.9476322568516703, 0.226547421131723, 0.5944201443911326,
            0.42830868492969476, 0.76414068655387, 0.00286059168094277,
            0.3574236812873617, 0.9096948856639084, 0.4560809854582528,
            0.9818027091583286, 0.8673569904602182, 0.9860275114020933,
            0.9232667066664217, 0.303693313961628, 0.8461034903175555
        ])
        x=x.reshape([1, 3, 1, 5])
        x.requires_grad = True
        y = QTensor([[[2, 1, 0, 0, 2]]], dtype=kint64)

        loss_result = NLL_Loss()
        result = loss_result(y, x)
        print(result)


CrossEntropyLoss
^^^^^^^^^^^^^^^^^^^^^^^^

.. py:class:: pyvqnet.nn.torch.CrossEntropyLoss(name="")

    Diese Funktion berechnet den Verlust von LogSoftmax und NLL_Loss zusammen.

    `x` contains the unnormalized output. Its shape can be :math:`(C)`, :math:`(N, C)` two-dimensional or :math:`(N, C, d_1, d_2, ..., d_K)` multidimensional.

    Die Formel der Verlustfunktion lautet wie folgt, wobei class die entsprechende Klassenbezeichnung des Zielwerts ist:

    .. math::
        \text{loss}(x, y) = -\log\left(\frac{\exp(x[class])}{\sum_j \exp(x[j])}\right)
        = -x[class] + \log\left(\sum_j \exp(x[j])\right)

    :param name: Der Name dieses Moduls, Standard ist "".
    :return: Eine CrossEntropyLoss-Verlustfunktionsinstanz

    Erforderliche Parameter für die Fehler-Vorwärtsberechnungsfunktion:

        x: :math:`(N, *)`, the output of the loss function, which can be a multi-dimensional variable.

        y: :math:`(N, *)`, the expected true value of the loss function. Must be a 64-bit integer, kint64.

    .. note::

        Bitte beachten Sie, dass im Gegensatz zu Frameworks wie PyTorch bei der Forward-Funktion der CrossEntropyLoss-Funktion der erste Parameter der Zielwert und der zweite Parameter der Vorhersagewert ist.

        Diese Klasse erbt von ``pyvqnet.nn.Module`` und ``torch.nn.Module`` und kann als Untermodul von ``torch.nn.Module`` zum torch-Modell hinzugefügt werden.

    Example::

        from pyvqnet.tensor import QTensor
        from pyvqnet import kint64
        from pyvqnet.nn.torch import CrossEntropyLoss
        import pyvqnet
        pyvqnet.backends.set_backend("torch")
        x = QTensor([
            0.9476322568516703, 0.226547421131723, 0.5944201443911326,
            0.42830868492969476, 0.76414068655387, 0.00286059168094277,
            0.3574236812873617, 0.9096948856639084, 0.4560809854582528,
            0.9818027091583286, 0.8673569904602182, 0.9860275114020933,
            0.9232667066664217, 0.303693313961628, 0.8461034903175555
        ])
        x=x.reshape([1, 3, 1, 5])
        x.requires_grad = True
        y = QTensor([[[2, 1, 0, 0, 2]]], dtype=kint64)

        loss_result = CrossEntropyLoss()
        result = loss_result(y, x)
        print(result)


Aktivierungsfunktionen
----------------------

Sigmoid
^^^^^^^^^^^^^^^^^^^^^^^^

.. py:class:: pyvqnet.nn.torch.Sigmoid(name:str="")

    Sigmoid-Aktivierungsfunktionsschicht.

    .. math::
        \text{Sigmoid}(x) = \frac{1}{1 + \exp(-x)}

    Diese Klasse erbt von ``pyvqnet.nn.Module`` und ``torch.nn.Module`` und kann als Untermodul von ``torch.nn.Module`` zum torch-Modell hinzugefügt werden.

    :param name: Der Name der Aktivierungsfunktionsschicht, Standard ist "".

    :return: Eine Sigmoid-Aktivierungsfunktionsschicht-Instanz.

    Examples::

        from pyvqnet.nn.torch import Sigmoid
        from pyvqnet.tensor import QTensor
        import pyvqnet
        pyvqnet.backends.set_backend("torch")
        layer = Sigmoid()
        y = layer(QTensor([1.0, 2.0, 3.0, 4.0]))
        print(y)


Softplus
^^^^^^^^^^^^^^^^^^^^^^^^

.. py:class:: pyvqnet.nn.torch.Softplus(name:str="")

    Softplus 

    .. math::
        \text{Softplus}(x) = \log(1 + \exp(x))

    Diese Klasse erbt von ``pyvqnet.nn.Module`` und ``torch.nn.Module`` und kann als Untermodul von ``torch.nn.Module`` zum torch-Modell hinzugefügt werden.

    :param name: Der Name der Aktivierungsfunktionsschicht, Standard ist "".

    :return: eine Softplus-Instanz.

    Examples::

        from pyvqnet.nn.torch import Softplus
        from pyvqnet.tensor import QTensor
        import pyvqnet
        pyvqnet.backends.set_backend("torch")
        layer = Softplus()
        y = layer(QTensor([1.0, 2.0, 3.0, 4.0]))

Softsign
^^^^^^^^^^^^^^^^^^^^^^^^

.. py:class:: pyvqnet.nn.torch.Softsign(name:str="")

    Softsign.

    .. math::
        \text{SoftSign}(x) = \frac{x}{ 1 + |x|}


    Diese Klasse erbt von ``pyvqnet.nn.Module`` und ``torch.nn.Module`` und kann als Untermodul von ``torch.nn.Module`` zum torch-Modell hinzugefügt werden.


    :param name: Der Name der Aktivierungsfunktionsschicht, Standard ist "".

    :return: eine SoftSign-Instanz.

    Examples::

        from pyvqnet.nn.torch import Softsign
        from pyvqnet.tensor import QTensor
        import pyvqnet
        pyvqnet.backends.set_backend("torch")
        layer = Softsign()
        y = layer(QTensor([1.0, 2.0, 3.0, 4.0]))



Softmax
^^^^^^^^^^^^^^^^^^^^^^^^

.. py:class:: pyvqnet.nn.torch.Softmax(axis:int = -1,name:str="")

    Softmax 

    .. math::
        \text{Softmax}(x_{i}) = \frac{\exp(x_i)}{\sum_j \exp(x_j)}


    Diese Klasse erbt von ``pyvqnet.nn.Module`` und ``torch.nn.Module`` und kann als Untermodul von ``torch.nn.Module`` zum torch-Modell hinzugefügt werden.


    :param axis: the dimension to calculate (the last axis is -1), default value = -1.
    :param name: Der Name der Aktivierungsfunktionsschicht, Standard ist "".

    :return: eine Softmax-Instanz.

    Examples::

        from pyvqnet.nn.torch import Softmax
        from pyvqnet.tensor import QTensor
        import pyvqnet
        pyvqnet.backends.set_backend("torch")
        layer = Softmax()
        y = layer(QTensor([1.0, 2.0, 3.0, 4.0]))


HardSigmoid
^^^^^^^^^^^^^^^^^^^^^^^^

.. py:class:: pyvqnet.nn.torch.HardSigmoid(name:str="")

    HardSigmoid 

    .. math::
        \text{Hardsigmoid}(x) = \begin{cases}
            0 & \text{ if } x \le -3, \\
            1 & \text{ if } x \ge +3, \\
            x / 6 + 1 / 2 & \text{otherwise}
        \end{cases}


    Diese Klasse erbt von ``pyvqnet.nn.Module`` und ``torch.nn.Module`` und kann als Untermodul von ``torch.nn.Module`` zum torch-Modell hinzugefügt werden.


    :param name: Der Name der Aktivierungsfunktionsschicht, Standard ist "".

    :return: HardSigmoid-Instanz.

    Examples::

        from pyvqnet.nn.torch import HardSigmoid
        from pyvqnet.tensor import QTensor
        import pyvqnet
        pyvqnet.backends.set_backend("torch")
        layer = HardSigmoid()
        y = layer(QTensor([1.0, 2.0, 3.0, 4.0]))


ReLu
^^^^^^^^^^^^^^^^^^^^^^^^

.. py:class:: pyvqnet.nn.torch.ReLu(name:str="")

    ReLu.

    .. math::
        \text{ReLu}(x) = \begin{cases}
        x, & \text{ if } x > 0\\
        0, & \text{ if } x \leq 0
        \end{cases}


    Diese Klasse erbt von ``pyvqnet.nn.Module`` und ``torch.nn.Module`` und kann als Untermodul von ``torch.nn.Module`` zum torch-Modell hinzugefügt werden.


    :param name: Der Name der Aktivierungsfunktionsschicht, Standard ist "".

    :return: eine ReLu-Instanz.

    Examples::

        from pyvqnet.nn.torch import ReLu
        from pyvqnet.tensor import QTensor
        import pyvqnet
        pyvqnet.backends.set_backend("torch")
        layer = ReLu()
        y = layer(QTensor([-1, 2.0, -3, 4.0]))

        


LeakyReLu
^^^^^^^^^^^^^^^^^^^^^^^^

.. py:class:: pyvqnet.nn.torch.LeakyReLu(alpha:float=0.01,name:str="")

    LeakyReLu  

    .. math::
        \text{LeakyRelu}(x) =
        \begin{cases}
        x, & \text{ if } x \geq 0 \\
        \alpha * x, & \text{ otherwise }
        \end{cases}


    Diese Klasse erbt von ``pyvqnet.nn.Module`` und ``torch.nn.Module`` und kann als Untermodul von ``torch.nn.Module`` zum torch-Modell hinzugefügt werden.


    :param alpha: LeakyRelu coefficient, default: 0.01.
    :param name: Der Name der Aktivierungsfunktionsschicht, Standard ist "".

    :return: eine LeakyReLu-Aktivierungsinstanz.

    Examples::

        from pyvqnet.nn.torch import LeakyReLu
        from pyvqnet.tensor import QTensor
        import pyvqnet
        pyvqnet.backends.set_backend("torch")
        layer = LeakyReLu()
        y = layer(QTensor([-1, 2.0, -3, 4.0]))



Gelu
^^^^^^^^^^^^^^^^^^^^^^^^

.. py:class:: pyvqnet.nn.torch.Gelu(approximate="tanh", name="")
    
    Gelu:

    .. math:: \text{GELU}(x) = x * \Phi(x)

    Wenn der Approximationsparameter 'tanh' ist, wird GELU wie folgt geschätzt:

    .. math:: \text{GELU}(x) = 0.5 * x * (1 + \text{Tanh}(\sqrt{2 / \pi} * (x + 0.044715 * x^3)))


    Diese Klasse erbt von ``pyvqnet.nn.Module`` und ``torch.nn.Module`` und kann als Untermodul von ``torch.nn.Module`` zum torch-Modell hinzugefügt werden.


    :param approximate: Approximate calculation method, the default is "tanh".
    :param name: Der Name der Aktivierungsfunktionsschicht, Standard ist "".

    :return: Gelu-Aktivierungsinstanz.

    Examples::

        from pyvqnet.tensor import randu, ones_like
        from pyvqnet.nn.torch import Gelu
        import pyvqnet
        pyvqnet.backends.set_backend("torch")
        qa = randu([5,4])
        qb = Gelu()(qa)



ELU
^^^^^^^^^^^^^^^^^^^^^^^^

.. py:class:: pyvqnet.nn.torch.ELU(alpha:float=1,name:str="")

    ELU Exponential Linear Unit Aktivierungsfunktionsschicht.

    .. math::
        \text{ELU}(x) = \begin{cases}
        x, & \text{ if } x > 0\\
        \alpha * (\exp(x) - 1), & \text{ if } x \leq 0
        \end{cases}


    Diese Klasse erbt von ``pyvqnet.nn.Module`` und ``torch.nn.Module`` und kann als Untermodul von ``torch.nn.Module`` zum torch-Modell hinzugefügt werden.



    :param alpha: ELU Coefficient, default:1.
    :param name: Der Name der Aktivierungsfunktionsschicht, Standard ist "".

    :return: ELU-Aktivierungsinstanz.

    Examples::

        from pyvqnet.nn.torch import ELU
        from pyvqnet.tensor import QTensor
        import pyvqnet
        pyvqnet.backends.set_backend("torch")
        layer = ELU()
        y = layer(QTensor([-1, 2.0, -3, 4.0]))


Tanh
^^^^^^^^^^^^^^^^^^^^^^^^

.. py:class:: pyvqnet.nn.torch.Tanh(name:str="")

    Tanh hyperbolische Tangens-Aktivierungsfunktion.

    .. math::
        \text{Tanh}(x) = \frac{\exp(x) - \exp(-x)} {\exp(x) + \exp(-x)}


    Diese Klasse erbt von ``pyvqnet.nn.Module`` und ``torch.nn.Module`` und kann als Untermodul von ``torch.nn.Module`` zum torch-Modell hinzugefügt werden.



    :param name: Der Name der Aktivierungsfunktionsschicht, Standard ist "".

    :return: Tanh-Aktivierungsinstanz.

    Examples::

        from pyvqnet.nn.torch import Tanh
        from pyvqnet.tensor import QTensor
        import pyvqnet
        pyvqnet.backends.set_backend("torch")
        layer = Tanh()
        y = layer(QTensor([-1, 2.0, -3, 4.0]))



Optimierer-Modul
---------------------------------------------

Für klassische und Quantenschaltkreis-Module, die von `TorchModule` erben, können die Parameter `model.paramters()` weiterhin mit anderen Optimierern als `Rotosolve` unter :ref:`Optimizer` optimiert werden.



Verwendung von pyqpanda zur Ausführung variationeller Quantenschaltkreise
-------------------------------------------------------------------------

Im Folgenden wird die Trainingsschnittstelle für variationelle Quantenschaltkreise zur Schaltkreisberechnung mit pyqpanda und pyqpanda3 beschrieben.

.. warning::

    Der Quantenberechnungsteil des folgenden TorchQpandaQuantumLayer verwendet pyqpanda2.

    Aufgrund von Kompatibilitätsproblemen zwischen pyqpanda2 und pyqpanda3 müssen Sie pyqpnda2 selbst installieren, `pip install pyqpanda`

TorchQpandaQuantumLayer
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Wenn Sie mit der pyQPanda2-Syntax vertrauter sind, können Sie die Schnittstelle TorchQpandaQuantumLayer verwenden. Fügen Sie benutzerdefinierte Quantenbits ``qubits``, klassische Bits ``cbits`` und den Backend-Simulator ``machine`` zur ``qprog_with_measure``-Funktion von TorchQpandaQuantumLayer hinzu.

.. py:class:: pyvqnet.qnn.vqc.torch.TorchQpandaQuantumLayer(qprog_with_measure,para_num,diff_method:str = "parameter_shift",delta:float = 0.01,dtype=None,name="")

    Abstraktes Berechnungsmodul einer variationellen Quantenschicht. Verwendet pyQPanda2 zur Simulation eines parametrisierten Quantenschaltkreises und erhält die Messergebnisse. Diese variationelle Quantenschicht erbt das Gradientenberechnungsmodul des VQNet-Frameworks. Sie kann die Parameterdrift-Methode zur Berechnung des Gradienten der Schaltkreisparameter verwenden, variationelle Quantenschaltkreismodelle trainieren oder variationelle Quantenschaltkreise in hybride Quanten-Klassik-Modelle einbetten.

    :param qprog_with_measure: Quantum circuit operation and measurement functions built with pyQPand.
    :param para_num: `int` - number of parameters.
    :param diff_method: Method for solving quantum circuit parameter gradients, "parameter shift" or "finite difference", default parameter shift.
    :param delta: \delta when calculating gradients by finite difference.
    :param dtype: Data type of parameter, default: None, use default data type: kfloat32, representing 32-bit floating point numbers.
    :param name: Der Name dieses Moduls, Standard ist "".

    :return: A module that can calculate quantum circuits.

    .. note::

        qprog_with_measure ist eine in pyQPanda2 definierte Quantenschaltkreisfunktion.

        Diese Funktion muss die folgenden Parameter als Funktionseingabe enthalten (auch wenn ein Parameter nicht tatsächlich verwendet wird), da sie sonst in dieser Funktion nicht richtig funktioniert.

        Im Vergleich zu QuantumLayer soll der Benutzer im variationellen Schaltkreis, der über diese Schnittstelle übergeben wird, die Quantenbits und Simulatoren manuell erstellen.

        Wenn qprog_with_measure eine Quantenmessung erfordert, muss der Benutzer auch manuell cbits erstellen und zuweisen.

        Die Verwendung der Quantenschaltkreisfunktion qprog_with_measure (input, param, nqubits, ncbits) kann dem folgenden Beispiel entnommen werden.

        `input`: Eingabe eindimensionaler klassischer Daten. Wenn keine vorhanden, None eingeben.

        `param`: Eingabe der zu trainierenden eindimensionalen variationellen Quantenschaltkreisparameter.

    Example::

        import pyqpanda as pq
        from pyvqnet.qnn import ProbsMeasure
        import numpy as np
        from pyvqnet.tensor import QTensor
        import pyvqnet
        pyvqnet.backends.set_backend("torch")
        from pyvqnet.qnn.vqc.torch import TorchQpandaQuantumLayer
        def pqctest (input,param):
            num_of_qubits = 4

            m_machine = pq.CPUQVM()# außerhalb
            m_machine.init_qvm()# außerhalb
            qubits = m_machine.qAlloc_many(num_of_qubits)

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

        pqc = TorchQpandaQuantumLayer(pqctest,3)

        # klassische Daten als Eingabe
        input = QTensor([[1.0,2,3,4],[4,2,2,3],[3,3,2,2]],requires_grad=True)

        # Vorwärtsschaltkreise
        rlt = pqc(input)

        print(rlt)

        grad =  QTensor(np.ones(rlt.data.shape)*1000)
        # Rückwärtsschaltkreise
        rlt.backward(grad)

        print(pqc.m_para.grad)
        print(input.grad)


.. warning::

    Der Quantenberechnungsteil der folgenden TorchQcloud3QuantumLayer- und TorchQpanda3QuantumLayer-Schnittstellen verwendet pyqpanda3.

    Wenn Sie die QCloud-Funktion unter diesem Modul verwenden, treten Fehler beim Importieren von pyqpanda2 im Code oder bei der Verwendung der pyqpanda2-bezogenen Paketschnittstellen von pyvqnet auf.

TorchQcloud3QuantumLayer
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Wenn Sie die neueste Version von pyqpanda3 installiert haben, können Sie diese Schnittstelle verwenden, um einen variationellen Schaltkreis zu definieren und ihn zur Ausführung auf dem echten Chip von originqc einzureichen.

.. py:class:: pyvqnet.qnn.vqc.torch.TorchQcloud3QuantumLayer(origin_qprog_func, qcloud_token, para_num, pauli_str_dict=None, shots = 1000, initializer=None, dtype=None, name="", diff_method="parameter_shift", submit_kwargs={}, query_kwargs={})
    
    Ein abstraktes Berechnungsmodul für echte Chips unter Verwendung von originqc von pyqpanda3. Es reicht parametrisierte Quantenschaltkreise an echte Chips weiter und erhält Messergebnisse.
    Wenn diff_method == "random_coordinate_descent", wählt die Schicht zufällig einen einzelnen Parameter zur Berechnung des Gradienten aus, während andere Parameter Null bleiben. Referenz: https://arxiv.org/abs/2311.00088

    .. note::

        qcloud_token ist der API-Token, den Sie von der Cloud-Plattform beantragt haben.

        origin_qprog_func muss Daten vom Typ pypqanda3.core.QProg zurückgeben. Wenn pauli_str_dict nicht gesetzt ist, muss sichergestellt werden, dass die Messung in das QProg eingefügt wurde.

        origin_qprog_func muss das folgende Format haben:

        origin_qprog_func(input,param )

        `input`: Eingabe 1~2D klassischer Daten. Im 2D-Fall ist die erste Dimension die Batchgröße.

        `param`: Input the parameters to be trained of the 1D variational quantum circuit.

    .. warning::

        Diese Klasse erbt von ``pyvqnet.nn.Module`` und ``torch.nn.Module`` und kann als Untermodul von ``torch.nn.Module`` zum torch-Modell hinzugefügt werden.

        The data in ``_buffers`` of this class is of ``torch.Tensor`` type.

        The data in ``_parmeters`` of this class is of ``torch.nn.Parameter`` type.

    :param origin_qprog_func: The variational quantum circuit function built by QPanda, which must return QProg.
    :param qcloud_token: `str` - The type of quantum machine or cloud token for execution.
    :param para_num: `int` - The number of parameters, the parameter is a QTensor of size [para_num].
    :param pauli_str_dict: `dict|list` - Dictionary or list of dictionaries representing Pauli operators in quantum circuits. Defaults to "None", which means measurement operations are performed. If a dictionary of Pauli operators is entered, a single expectation or multiple expectations are calculated.
    :param shot: `int` - Number of measurements. The default value is 1000.
    :param initializer: Initializer for parameter values. The default value is "None", using a 0~2*pi normal distribution.
    :param dtype: Data type of the parameter. The default value is None, which means using the default data type pyvqnet.kfloat32.
    :param name: The name of the module. The default value is an empty string.
    :param diff_method: Differentiation method for gradient calculation. The default value is "parameter_shift", "random_coordinate_descent".
    :param submit_kwargs: Additional keyword parameters for submitting quantum circuits, default: {"if_print_qcloud_log":False,"chip_id":"WK_C180","is_amend":True,"is_mapping":True,"is_optimization":True,"compile_level":3,"default_task_group_size":200,"test_qcloud_fake":False,"":"server_ip_address"}, when test_qcloud_fake is set to True, local CPUQVM simulation.
    :param query_kwargs: Additional keyword parameters for querying quantum results, default: {"timeout":2,"print_query_info":True,"sub_circuits_split_size":1}.
    :return: A module that can calculate quantum circuits.


    Example::

        import pyqpanda3.core as pq
        import pyvqnet
        from pyvqnet.qnn.vqc.torch import TorchQcloud3QuantumLayer

        pyvqnet.backends.set_backend("torch")
        def qfun(input,param):

            m_qlist = range(6)
            cbits = range(6)
            measure_qubits = [0,2]
            m_prog = pq.QProg()
            cir = pq.QCircuit()
            cir<<pq.RZ(m_qlist[0],input[0])
            cir<<pq.CNOT(m_qlist[0],m_qlist[1])
            cir<<pq.RY(m_qlist[1],param[0])
            cir<<pq.CNOT(m_qlist[0],m_qlist[2])
            cir<<pq.RZ(m_qlist[1],input[1])
            cir<<pq.RY(m_qlist[2],param[1])
            cir<<pq.H(m_qlist[2])
            m_prog<<cir

            for idx, ele in enumerate(measure_qubits):
                m_prog << pq.measure(m_qlist[ele], cbits[idx])  # pylint: disable=expression-not-assigned
            return m_prog

        l = TorchQcloud3QuantumLayer(qfun,
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
            cbits = range(6)
            measure_qubits = [0,2]
            m_prog = pq.QProg()
            cir = pq.QCircuit()
            cir<<pq.RZ(m_qlist[0],input[0])
            cir<<pq.CNOT(m_qlist[0],m_qlist[1])
            cir<<pq.RY(m_qlist[1],param[0])
            cir<<pq.CNOT(m_qlist[0],m_qlist[2])
            cir<<pq.RZ(m_qlist[1],input[1])
            cir<<pq.RY(m_qlist[2],param[1])
            cir<<pq.H(m_qlist[2])
            m_prog<<cir

            return m_prog
        l = TorchQcloud3QuantumLayer(qfun2,
                "3047DE8A59764BEDAC9C3282093B16AF",
                2,

                pauli_str_dict={'Z0 X1':10,'':-0.5,'Y2':-0.543},
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

TorchQpanda3QuantumLayer
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Wenn Sie mit der pyQPanda3-Syntax vertrauter sind, können Sie die Schnittstelle TorchQpanda3QuantumLayer verwenden.

.. py:class:: pyvqnet.qnn.vqc.torch.TorchQpanda3QuantumLayer(qprog_with_measure,para_num,diff_method:str = "parameter_shift",delta:float = 0.01,dtype=None,name="")

    Abstraktes Berechnungsmodul einer variationellen Quantenschicht. Verwendet pyQPanda3 zur Simulation eines parametrisierten Quantenschaltkreises und erhält die Messergebnisse. Diese variationelle Quantenschicht erbt das Gradientenberechnungsmodul des VQNet-Frameworks. Sie können die Parameterdrift-Methode zur Berechnung des Gradienten der Schaltkreisparameter verwenden, das variationelle Quantenschaltkreismodell trainieren oder den variationellen Quantenschaltkreis in ein hybrides Quanten-Klassik-Modell einbetten.

    :param qprog_with_measure: Quantum circuit operation and measurement functions built with pyQPand.
    :param para_num: `int` - number of parameters.
    :param diff_method: method for solving quantum circuit parameter gradients, "parameter shift" or "finite difference", default parameter shift.
    :param delta: \delta when calculating gradients by finite difference.
    :param dtype: data type of parameter, default: None, use default data type: kfloat32, representing 32-bit floating point numbers.
    :param name: the name of this module, default is "".

    :return: a module that can calculate quantum circuits.

    .. note::

        qprog_with_measure ist eine in pyQPanda definierte Quantenschaltkreisfunktion.

        Diese Funktion muss die folgenden Parameter als Funktionseingaben enthalten (auch wenn ein Parameter nicht tatsächlich verwendet wird), da sie sonst in dieser Funktion nicht richtig funktioniert.

        The use of the quantum circuit function qprog_with_measure (input,param,nqubits,ncbits) can refer to the following example.

        `input`: Eingabe eindimensionaler klassischer Daten. Wenn nicht vorhanden, None eingeben.

        `param`: Eingabe der zu trainierenden Parameter für den eindimensionalen variationellen Quantenschaltkreis.

    Example::

        import pyqpanda3.core as pq
        from pyvqnet.qnn.pq3 import ProbsMeasure
        import numpy as np
        from pyvqnet.tensor import QTensor
        import pyvqnet
        pyvqnet.backends.set_backend("torch")
        from pyvqnet.qnn.vqc.torch import TorchQpanda3QuantumLayer
        def pqctest (input,param):
            num_of_qubits = 4

            m_machine = pq.CPUQVM()# außerhalb
        
            qubits =range(num_of_qubits)

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

        pqc = TorchQpanda3QuantumLayer(pqctest,3)

        # klassische Daten als Eingabe
        input = QTensor([[1.0,2,3,4],[4,2,2,3],[3,3,2,2]],requires_grad=True)

        # Vorwärtsschaltkreise
        rlt = pqc(input)

        print(rlt)

        grad =  QTensor(np.ones(rlt.data.shape)*1000)
        # Rückwärtsschaltkreise
        rlt.backward(grad)

        print(pqc.m_para.grad)
        print(input.grad)

Variationelles Quantenschaltkreis-Modul und Schnittstelle basierend auf automatischer Differentiation
-----------------------------------------------------------------------------------------------------
Basisklasse
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Das Schreiben eines variationellen Quantenschaltkreismodells erfordert die Vererbung von ``QModule``.

QModule
""""""""""""""""""

.. py:class:: pyvqnet.qnn.vqc.torch.QModule(name="")

    Wenn der Benutzer das `torch`-Backend verwendet, definiert dies die Basisklasse, die das Quanten-Variationsschaltkreis-Modell `Module` erben soll.
    Diese Klasse erbt von ``pyvqnet.nn.torch.TorchModule`` und ``torch.nn.Module``.

    Diese Klasse kann als Untermodul von ``torch.nn.Module`` zum torch-Modell hinzugefügt werden.

    .. note::

        Diese Klasse und ihre abgeleiteten Klassen sind nur für ``pyvqnet.backends.set_backend("torch")`` anwendbar, nicht mit dem ``Module`` unter dem standardmäßigen ``pyvqnet.nn`` mischen.

        The data in ``_buffers`` of this class is of ``torch.Tensor`` type.

        The data in ``_parmeters`` of this class is of ``torch.nn.Parameter`` type.


QMachine
""""""""""""""""""

.. py:class:: pyvqnet.qnn.vqc.torch.QMachine(num_wires, dtype=pyvqnet.kcomplex64,grad_mode="",save_ir=False)

    Simulatorklasse für variationelles Quantenrechnen, einschließlich Zustandsvektoren, deren Zustandsattribut Quantenschaltkreise sind.

    Diese Klasse erbt von ``pyvqnet.nn.torch.TorchModule`` und ``pyvqnet.qnn.QMachine``.

    Diese Klasse kann als Untermodul von ``torch.nn.Module`` zum torch-Modell hinzugefügt werden.

    .. note::

        Vor jedem Durchlauf des vollständigen Quantenschaltkreises müssen Sie `pyvqnet.qnn.vqc.QMachine.reset_states(batchsize)` verwenden, um den Anfangszustand im Simulator neu zu initialisieren und ihn auf
        (batchsize,*)-Dimensionen zu übertragen, um sich an Batch-Datentraining anzupassen.

    :param num_wires: Die Anzahl der Quantenbits.
    :param dtype: Der Datentyp der berechneten Daten. Der Standardwert ist pyvqnet.kcomplex64, die entsprechende Parameterpräzision ist pyvqnet.kfloat32.
    :param grad_mode: Der Gradientenberechnungsmodus, kann "adjoint" sein, der Standardwert: "", verwendet automatische Differentiation.
    :param save_ir: Wenn auf True gesetzt, wird die Operation in originIR gespeichert, der Standardwert: False.

    :return: Gibt ein QMachine-Objekt aus.

    Example::
        
        from pyvqnet.qnn.vqc.torch import QMachine
        import pyvqnet
        pyvqnet.backends.set_backend("torch")
        qm = QMachine(4)
        print(qm.states)


   .. py:method:: reset_states(batchsize)

        Initialisiert den Anfangszustand im Simulator neu und sendet ihn an
        (batchsize,*)-Dimensionen, um sich an Batch-Datentraining anzupassen.

        :param batchsize: Batch processing dimension.

Modul für variationelle Quantenlogikgatter
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Die folgenden Funktionsschnittstellen in ``pyvqnet.qnn.vqc`` unterstützen direkt ``QTensor`` des ``torch``-Backends für Berechnungen.

.. csv-table:: List of supported pyvqnet.qnn.vqc interfaces
    :file: ./images/same_apis_from_vqc.csv

Die folgenden Quantenschaltkreis-Module erben von ``pyvqnet.qnn.vqc.torch.QModule``, wobei Berechnungen mit ``torch.Tensor`` durchgeführt werden.

.. note::

    Diese Klasse und ihre abgeleiteten Klassen sind nur für ``pyvqnet.backends.set_backend("torch")`` anwendbar, nicht mit ``Module`` unter dem standardmäßigen ``pyvqnet.nn`` mischen.

    Wenn diese Klassen nicht-parametrische Member-Variablen ``_buffers`` haben, sind die darin enthaltenen Daten vom Typ ``torch.Tensor``.
    Wenn diese Klassen parametrische Member-Variablen ``_parameters`` haben, sind die darin enthaltenen Daten vom Typ ``torch.nn.Parameter``.

I
""""""""""""""""""

.. py:class:: pyvqnet.qnn.vqc.torch.I(has_params: bool = False,trainable: bool = False,init_params=None,wires=None,dtype=pyvqnet.kcomplex64,use_dagger=False)
    
    Definiert ein I-Quantengatter.

    Diese Klasse erbt von ``pyvqnet.qnn.vqc.torch.QModule`` und ``torch.nn.Module``.
    Diese Klasse kann als Untermodul von ``torch.nn.Module`` zum torch-Modell hinzugefügt werden.

    :param has_params: ob es Parameter hat, wie RX, RY und andere Gatter auf True gesetzt werden müssen, und solche ohne Parameter auf False gesetzt werden müssen, der Standard ist False.
    :param trainable: ob es zu trainierende Parameter hat. Wenn die Schicht externe Eingabedaten zum Aufbau der Logikgatter-Matrix verwendet, auf False setzen. Wenn die zu trainierenden Parameter von dieser Schicht initialisiert werden müssen, ist es True, der Standard ist False.
    :param init_params: Initialisierungsparameter zum Codieren klassischer Daten QTensor, der Standard ist None.
    :param wires: Bit-Index des Leitungseffekts, der Standard ist None.
    :param dtype: Die Datenpräzision der internen Matrix des Logikgatters kann auf pyvqnet.kcomplex64 oder pyvqnet.kcomplex128 gesetzt werden, entsprechend float- bzw. double-Eingabe.
    :param use_dagger: ob die transponiert-konjugierte Version des Gatters verwendet werden soll, der Standard ist False.
    :return: eine ``pyvqnet.qnn.vqc.torch.QModule``-Instanz

    Example::
        
        from pyvqnet.qnn.vqc.torch import I,QMachine
        import pyvqnet
        pyvqnet.backends.set_backend("torch")
        device = QMachine(4)
        layer = I(wires=0)
        batchsize = 1
        device.reset_states(1)
        layer(q_machine = device)
        print(device.states)


Hadamard
""""""""""""""""""

.. py:class:: pyvqnet.qnn.vqc.torch.Hadamard(has_params: bool = False,trainable: bool = False,init_params=None,wires=None,dtype=pyvqnet.kcomplex64,use_dagger=False)
    
    Definiert ein Hadamard-Quantengatter.

    Diese Klasse erbt von ``pyvqnet.qnn.vqc.torch.QModule`` und ``torch.nn.Module``.
    Diese Klasse kann als Untermodul von ``torch.nn.Module`` zum torch-Modell hinzugefügt werden.

    :param has_params: ob es Parameter hat, wie RX, RY und andere Gatter auf True gesetzt werden müssen, und solche ohne Parameter auf False gesetzt werden müssen, der Standard ist False.
    :param trainable: ob es zu trainierende Parameter hat. Wenn die Schicht externe Eingabedaten zum Aufbau der Logikgatter-Matrix verwendet, auf False setzen. Wenn die zu trainierenden Parameter von dieser Schicht initialisiert werden müssen, ist es True, der Standard ist False.
    :param init_params: Initialisierungsparameter zum Codieren klassischer Daten QTensor, der Standard ist None.
    :param wires: Bit-Index des Leitungseffekts, der Standard ist None.
    :param dtype: Die Datenpräzision der internen Matrix des Logikgatters kann auf pyvqnet.kcomplex64 oder pyvqnet.kcomplex128 gesetzt werden, entsprechend float- bzw. double-Eingabe.
    :param use_dagger: ob die transponiert-konjugierte Version des Gatters verwendet werden soll, der Standard ist False.
    :return: eine ``pyvqnet.qnn.vqc.torch.QModule``-Instanz

    Example::
        
        from pyvqnet.qnn.vqc.torch import Hadamard,QMachine
        import pyvqnet
        pyvqnet.backends.set_backend("torch")
        device = QMachine(4)
        layer = Hadamard(wires=0)
        batchsize = 1
        device.reset_states(1)
        layer(q_machine = device)
        print(device.states)


T
""""""""""""""""""

.. py:class:: pyvqnet.qnn.vqc.torch.T(has_params: bool = False,trainable: bool = False,init_params=None,wires=None,dtype=pyvqnet.kcomplex64,use_dagger=False)
    
    Definiert ein T-Quantengatter.

    Diese Klasse erbt von ``pyvqnet.qnn.vqc.torch.QModule`` und ``torch.nn.Module``.
    Diese Klasse kann als Untermodul von ``torch.nn.Module`` zum torch-Modell hinzugefügt werden.

    :param has_params: ob es Parameter hat, wie RX, RY und andere Gatter auf True gesetzt werden müssen, und solche ohne Parameter auf False gesetzt werden müssen, der Standard ist False.
    :param trainable: ob es zu trainierende Parameter hat. Wenn die Schicht externe Eingabedaten zum Aufbau der Logikgatter-Matrix verwendet, auf False setzen. Wenn die zu trainierenden Parameter von dieser Schicht initialisiert werden müssen, ist es True, der Standard ist False.
    :param init_params: Initialisierungsparameter zum Codieren klassischer Daten QTensor, der Standard ist None.
    :param wires: Bit-Index des Leitungseffekts, der Standard ist None.
    :param dtype: Die Datenpräzision der internen Matrix des Logikgatters kann auf pyvqnet.kcomplex64 oder pyvqnet.kcomplex128 gesetzt werden, entsprechend float- bzw. double-Eingabe.
    :param use_dagger: ob die transponiert-konjugierte Version des Gatters verwendet werden soll, der Standard ist False.
    :return: eine ``pyvqnet.qnn.vqc.torch.QModule``-Instanz

    Example::
        
        from pyvqnet.qnn.vqc.torch import T,QMachine
        import pyvqnet
        pyvqnet.backends.set_backend("torch")
        device = QMachine(4)
        layer = T(wires=0)
        batchsize = 1
        device.reset_states(1)
        layer(q_machine = device)
        print(device.states)



S
""""""""""""""""""

.. py:class:: pyvqnet.qnn.vqc.torch.S(has_params: bool = False,trainable: bool = False,init_params=None,wires=None,dtype=pyvqnet.kcomplex64,use_dagger=False)
    
    Definiert ein S-Quantengatter.

    Diese Klasse erbt von ``pyvqnet.qnn.vqc.torch.QModule`` und ``torch.nn.Module``.
    Diese Klasse kann als Untermodul von ``torch.nn.Module`` zum torch-Modell hinzugefügt werden.

    :param has_params: ob es Parameter hat, wie RX, RY und andere Gatter auf True gesetzt werden müssen, und solche ohne Parameter auf False gesetzt werden müssen, der Standard ist False.
    :param trainable: ob es zu trainierende Parameter hat. Wenn die Schicht externe Eingabedaten zum Aufbau der Logikgatter-Matrix verwendet, auf False setzen. Wenn die zu trainierenden Parameter von dieser Schicht initialisiert werden müssen, ist es True, der Standard ist False.
    :param init_params: Initialisierungsparameter zum Codieren klassischer Daten QTensor, der Standard ist None.
    :param wires: Bit-Index des Leitungseffekts, der Standard ist None.
    :param dtype: Die Datenpräzision der internen Matrix des Logikgatters kann auf pyvqnet.kcomplex64 oder pyvqnet.kcomplex128 gesetzt werden, entsprechend float- bzw. double-Eingabe.
    :param use_dagger: ob die transponiert-konjugierte Version des Gatters verwendet werden soll, der Standard ist False.
    :return: eine ``pyvqnet.qnn.vqc.torch.QModule``-Instanz

    Example::
        
        from pyvqnet.qnn.vqc.torch import S,QMachine
        import pyvqnet
        pyvqnet.backends.set_backend("torch")
        device = QMachine(4)
        layer = S(wires=0)
        batchsize = 1
        device.reset_states(1)
        layer(q_machine = device)
        print(device.states)


PauliX
""""""""""""""""""

.. py:class:: pyvqnet.qnn.vqc.torch.PauliX(has_params: bool = False,trainable: bool = False,init_params=None,wires=None,dtype=pyvqnet.kcomplex64,use_dagger=False)
    
    Definiert ein PauliX-Quantengatter.

    Diese Klasse erbt von ``pyvqnet.qnn.vqc.torch.QModule`` und ``torch.nn.Module``.
    Diese Klasse kann als Untermodul von ``torch.nn.Module`` zum torch-Modell hinzugefügt werden.


    :param has_params: ob es Parameter hat, wie RX, RY und andere Gatter auf True gesetzt werden müssen, und solche ohne Parameter auf False gesetzt werden müssen, der Standard ist False.
    :param trainable: ob es zu trainierende Parameter hat. Wenn die Schicht externe Eingabedaten zum Aufbau der Logikgatter-Matrix verwendet, auf False setzen. Wenn die zu trainierenden Parameter von dieser Schicht initialisiert werden müssen, ist es True, der Standard ist False.
    :param init_params: Initialisierungsparameter zum Codieren klassischer Daten QTensor, der Standard ist None.
    :param wires: Bit-Index des Leitungseffekts, der Standard ist None.
    :param dtype: Die Datenpräzision der internen Matrix des Logikgatters kann auf pyvqnet.kcomplex64 oder pyvqnet.kcomplex128 gesetzt werden, entsprechend float- bzw. double-Eingabe.
    :param use_dagger: ob die transponiert-konjugierte Version des Gatters verwendet werden soll, der Standard ist False.
    :return: eine ``pyvqnet.qnn.vqc.torch.QModule``-Instanz

    Example::
        
        from pyvqnet.qnn.vqc.torch import PauliX,QMachine
        import pyvqnet
        pyvqnet.backends.set_backend("torch")
        device = QMachine(4)
        layer = PauliX(wires=0)
        batchsize = 1
        device.reset_states(1)
        layer(q_machine = device)
        print(device.states)


PauliY
""""""""""""""""""

.. py:class:: pyvqnet.qnn.vqc.torch.PauliY(has_params: bool = False,trainable: bool = False,init_params=None,wires=None,dtype=pyvqnet.kcomplex64,use_dagger=False)
    
    Definiert ein PauliY-Quantengatter.

    Diese Klasse erbt von ``pyvqnet.qnn.vqc.torch.QModule`` und ``torch.nn.Module``.
    Diese Klasse kann als Untermodul von ``torch.nn.Module`` zum torch-Modell hinzugefügt werden.


    :param has_params: ob es Parameter hat, wie RX, RY und andere Gatter auf True gesetzt werden müssen, und solche ohne Parameter auf False gesetzt werden müssen, der Standard ist False.
    :param trainable: ob es zu trainierende Parameter hat. Wenn die Schicht externe Eingabedaten zum Aufbau der Logikgatter-Matrix verwendet, auf False setzen. Wenn die zu trainierenden Parameter von dieser Schicht initialisiert werden müssen, ist es True, der Standard ist False.
    :param init_params: Initialisierungsparameter zum Codieren klassischer Daten QTensor, der Standard ist None.
    :param wires: Bit-Index des Leitungseffekts, der Standard ist None.
    :param dtype: Die Datenpräzision der internen Matrix des Logikgatters kann auf pyvqnet.kcomplex64 oder pyvqnet.kcomplex128 gesetzt werden, entsprechend float- bzw. double-Eingabe.
    :param use_dagger: ob die transponiert-konjugierte Version des Gatters verwendet werden soll, der Standard ist False.
    :return: eine ``pyvqnet.qnn.vqc.torch.QModule``-Instanz

    Example::
        
        from pyvqnet.qnn.vqc.torch import PauliY,QMachine
        import pyvqnet
        pyvqnet.backends.set_backend("torch")
        device = QMachine(4)
        layer = PauliY(wires=0)
        batchsize = 1
        device.reset_states(1)
        layer(q_machine = device)
        print(device.states)



PauliZ
""""""""""""""""""

.. py:class:: pyvqnet.qnn.vqc.torch.PauliZ(has_params: bool = False,trainable: bool = False,init_params=None,wires=None,dtype=pyvqnet.kcomplex64,use_dagger=False)
    
    Definiert ein PauliZ-Quantengatter.

    Diese Klasse erbt von ``pyvqnet.qnn.vqc.torch.QModule`` und ``torch.nn.Module``.
    Diese Klasse kann als Untermodul von ``torch.nn.Module`` zum torch-Modell hinzugefügt werden.


    :param has_params: ob es Parameter hat, wie RX, RY und andere Gatter auf True gesetzt werden müssen, und solche ohne Parameter auf False gesetzt werden müssen, der Standard ist False.
    :param trainable: ob es zu trainierende Parameter hat. Wenn die Schicht externe Eingabedaten zum Aufbau der Logikgatter-Matrix verwendet, auf False setzen. Wenn die zu trainierenden Parameter von dieser Schicht initialisiert werden müssen, ist es True, der Standard ist False.
    :param init_params: Initialisierungsparameter zum Codieren klassischer Daten QTensor, der Standard ist None.
    :param wires: Bit-Index des Leitungseffekts, der Standard ist None.
    :param dtype: Die Datenpräzision der internen Matrix des Logikgatters kann auf pyvqnet.kcomplex64 oder pyvqnet.kcomplex128 gesetzt werden, entsprechend float- bzw. double-Eingabe.
    :param use_dagger: ob die transponiert-konjugierte Version des Gatters verwendet werden soll, der Standard ist False.
    :return: eine ``pyvqnet.qnn.vqc.torch.QModule``-Instanz

    Example::
        
        from pyvqnet.qnn.vqc.torch import PauliZ,QMachine
        import pyvqnet
        pyvqnet.backends.set_backend("torch")
        device = QMachine(4)
        layer = PauliZ(wires=0)
        batchsize = 1
        device.reset_states(1)
        layer(q_machine = device)
        print(device.states)



X1
""""""""""""""""""

.. py:class:: pyvqnet.qnn.vqc.torch.X1(has_params: bool = False,trainable: bool = False,init_params=None,wires=None,dtype=pyvqnet.kcomplex64,use_dagger=False)
    
    Definiert ein X1-Quantengatter.

    Diese Klasse erbt von ``pyvqnet.qnn.vqc.torch.QModule`` und ``torch.nn.Module``.
    Diese Klasse kann als Untermodul von ``torch.nn.Module`` zum torch-Modell hinzugefügt werden.

    :param has_params: ob es Parameter hat, wie RX, RY und andere Gatter auf True gesetzt werden müssen, und solche ohne Parameter auf False gesetzt werden müssen, der Standard ist False.
    :param trainable: ob es zu trainierende Parameter hat. Wenn die Schicht externe Eingabedaten zum Aufbau der Logikgatter-Matrix verwendet, auf False setzen. Wenn die zu trainierenden Parameter von dieser Schicht initialisiert werden müssen, ist es True, der Standard ist False.
    :param init_params: Initialisierungsparameter zum Codieren klassischer Daten QTensor, der Standard ist None.
    :param wires: Bit-Index des Leitungseffekts, der Standard ist None.
    :param dtype: Die Datenpräzision der internen Matrix des Logikgatters kann auf pyvqnet.kcomplex64 oder pyvqnet.kcomplex128 gesetzt werden, entsprechend float- bzw. double-Eingabe.
    :param use_dagger: ob die transponiert-konjugierte Version des Gatters verwendet werden soll, der Standard ist False.
    :return: eine ``pyvqnet.qnn.vqc.torch.QModule``-Instanz

    Example::
        
        from pyvqnet.qnn.vqc.torch import X1,QMachine
        import pyvqnet
        pyvqnet.backends.set_backend("torch")
        device = QMachine(4)
        layer = X1(wires=0)
        batchsize = 1
        device.reset_states(1)
        layer(q_machine = device)
        print(device.states)


RX
""""""""""""""""""

.. py:class:: pyvqnet.qnn.vqc.torch.RX(has_params: bool = False,trainable: bool = False,init_params=None,wires=None,dtype=pyvqnet.kcomplex64,use_dagger=False)
    
    Definiert ein RX-Quantengatter.

    Diese Klasse erbt von ``pyvqnet.qnn.vqc.torch.QModule`` und ``torch.nn.Module``.
    Diese Klasse kann als Untermodul von ``torch.nn.Module`` zum torch-Modell hinzugefügt werden.

    :param has_params: ob es Parameter hat, wie RX, RY und andere Gatter auf True gesetzt werden müssen, und solche ohne Parameter auf False gesetzt werden müssen, der Standard ist False.
    :param trainable: ob es zu trainierende Parameter hat. Wenn die Schicht externe Eingabedaten zum Aufbau der Logikgatter-Matrix verwendet, auf False setzen. Wenn die zu trainierenden Parameter von dieser Schicht initialisiert werden müssen, ist es True, der Standard ist False.
    :param init_params: Initialisierungsparameter zum Codieren klassischer Daten QTensor, der Standard ist None.
    :param wires: Bit-Index des Leitungseffekts, der Standard ist None.
    :param dtype: Die Datenpräzision der internen Matrix des Logikgatters kann auf pyvqnet.kcomplex64 oder pyvqnet.kcomplex128 gesetzt werden, entsprechend float- bzw. double-Eingabe.
    :param use_dagger: ob die transponiert-konjugierte Version des Gatters verwendet werden soll, der Standard ist False.
    :return: eine ``pyvqnet.qnn.vqc.torch.QModule``-Instanz

    Example::

        from pyvqnet.qnn.vqc.torch import RX,QMachine
        import pyvqnet
        pyvqnet.backends.set_backend("torch")
        device = QMachine(4)
        layer = RX(has_params= True, trainable= True, wires=0)
        batchsize = 2
        device.reset_states(batchsize)
        layer(q_machine = device)
        print(device.states)



RY
""""""""""""""""""

.. py:class:: pyvqnet.qnn.vqc.torch.RY(has_params: bool = False,trainable: bool = False,init_params=None,wires=None,dtype=pyvqnet.kcomplex64,use_dagger=False)
    
    Definiert ein RY-Quantengatter.

    Diese Klasse erbt von ``pyvqnet.qnn.vqc.torch.QModule`` und ``torch.nn.Module``.
    Diese Klasse kann als Untermodul von ``torch.nn.Module`` zum torch-Modell hinzugefügt werden.

    :param has_params: ob es Parameter hat, wie RX, RY und andere Gatter auf True gesetzt werden müssen, und solche ohne Parameter auf False gesetzt werden müssen, der Standard ist False.
    :param trainable: ob es zu trainierende Parameter hat. Wenn die Schicht externe Eingabedaten zum Aufbau der Logikgatter-Matrix verwendet, auf False setzen. Wenn die zu trainierenden Parameter von dieser Schicht initialisiert werden müssen, ist es True, der Standard ist False.
    :param init_params: Initialisierungsparameter zum Codieren klassischer Daten QTensor, der Standard ist None.
    :param wires: Bit-Index des Leitungseffekts, der Standard ist None.
    :param dtype: Die Datenpräzision der internen Matrix des Logikgatters kann auf pyvqnet.kcomplex64 oder pyvqnet.kcomplex128 gesetzt werden, entsprechend float- bzw. double-Eingabe.
    :param use_dagger: ob die transponiert-konjugierte Version des Gatters verwendet werden soll, der Standard ist False.
    :return: eine ``pyvqnet.qnn.vqc.torch.QModule``-Instanz

    Example::

        from pyvqnet.qnn.vqc.torch import RY,QMachine
        import pyvqnet
        pyvqnet.backends.set_backend("torch")
        device = QMachine(4)
        layer = RY(has_params= True, trainable= True, wires=0)
        batchsize = 2
        device.reset_states(batchsize)
        layer(q_machine = device)
        print(device.states)


RZ
""""""""""""""""""

.. py:class:: pyvqnet.qnn.vqc.torch.RZ(has_params: bool = False,trainable: bool = False,init_params=None,wires=None,dtype=pyvqnet.kcomplex64,use_dagger=False)
    
    Definiert ein RZ-Quantengatter.

    Diese Klasse erbt von ``pyvqnet.qnn.vqc.torch.QModule`` und ``torch.nn.Module``.
    Diese Klasse kann als Untermodul von ``torch.nn.Module`` zum torch-Modell hinzugefügt werden.

    :param has_params: ob es Parameter hat, wie RX, RY und andere Gatter auf True gesetzt werden müssen, und solche ohne Parameter auf False gesetzt werden müssen, der Standard ist False.
    :param trainable: ob es zu trainierende Parameter hat. Wenn die Schicht externe Eingabedaten zum Aufbau der Logikgatter-Matrix verwendet, auf False setzen. Wenn die zu trainierenden Parameter von dieser Schicht initialisiert werden müssen, ist es True, der Standard ist False.
    :param init_params: Initialisierungsparameter zum Codieren klassischer Daten QTensor, der Standard ist None.
    :param wires: Bit-Index des Leitungseffekts, der Standard ist None.
    :param dtype: Die Datenpräzision der internen Matrix des Logikgatters kann auf pyvqnet.kcomplex64 oder pyvqnet.kcomplex128 gesetzt werden, entsprechend float- bzw. double-Eingabe.
    :param use_dagger: ob die transponiert-konjugierte Version des Gatters verwendet werden soll, der Standard ist False.
    :return: eine ``pyvqnet.qnn.vqc.torch.QModule``-Instanz

    Example::

        from pyvqnet.qnn.vqc.torch import RZ,QMachine
        import pyvqnet
        pyvqnet.backends.set_backend("torch")
        device = QMachine(4)
        layer = RZ(has_params= True, trainable= True, wires=0)
        batchsize = 2
        device.reset_states(batchsize)
        layer(q_machine = device)
        print(device.states)


CRX
""""""""""""""""""

.. py:class:: pyvqnet.qnn.vqc.torch.CRX(has_params: bool = False,trainable: bool = False,init_params=None,wires=None,dtype=pyvqnet.kcomplex64,use_dagger=False)
    
    Definiert ein CRX-Quantengatter.

    Diese Klasse erbt von ``pyvqnet.qnn.vqc.torch.QModule`` und ``torch.nn.Module``.
    Diese Klasse kann als Untermodul von ``torch.nn.Module`` zum torch-Modell hinzugefügt werden.

    :param has_params: ob es Parameter hat, wie RX, RY und andere Gatter auf True gesetzt werden müssen, und solche ohne Parameter auf False gesetzt werden müssen, der Standard ist False.
    :param trainable: ob es zu trainierende Parameter hat. Wenn die Schicht externe Eingabedaten zum Aufbau der Logikgatter-Matrix verwendet, auf False setzen. Wenn die zu trainierenden Parameter von dieser Schicht initialisiert werden müssen, ist es True, der Standard ist False.
    :param init_params: Initialisierungsparameter zum Codieren klassischer Daten QTensor, der Standard ist None.
    :param wires: Bit-Index des Leitungseffekts, der Standard ist None.
    :param dtype: Die Datenpräzision der internen Matrix des Logikgatters kann auf pyvqnet.kcomplex64 oder pyvqnet.kcomplex128 gesetzt werden, entsprechend float- bzw. double-Eingabe.
    :param use_dagger: ob die transponiert-konjugierte Version des Gatters verwendet werden soll, der Standard ist False.
    :return: eine ``pyvqnet.qnn.vqc.torch.QModule``-Instanz

    Example::

        from pyvqnet.qnn.vqc.torch import CRX,QMachine
        import pyvqnet
        pyvqnet.backends.set_backend("torch")
        device = QMachine(4)
        layer = CRX(has_params= True, trainable= True, wires=[0,2])
        batchsize = 2
        device.reset_states(batchsize)
        layer(q_machine = device)
        print(device.states)


CRY
""""""""""""""""""


.. py:class:: pyvqnet.qnn.vqc.torch.CRY(has_params: bool = False,trainable: bool = False,init_params=None,wires=None,dtype=pyvqnet.kcomplex64,use_dagger=False)
    
    Definiert ein CRY-Quantengatter.

    Diese Klasse erbt von ``pyvqnet.qnn.vqc.torch.QModule`` und ``torch.nn.Module``.
    Diese Klasse kann als Untermodul von ``torch.nn.Module`` zum torch-Modell hinzugefügt werden.

    :param has_params: ob es Parameter hat, wie RX, RY und andere Gatter auf True gesetzt werden müssen, und solche ohne Parameter auf False gesetzt werden müssen, der Standard ist False.
    :param trainable: ob es zu trainierende Parameter hat. Wenn die Schicht externe Eingabedaten zum Aufbau der Logikgatter-Matrix verwendet, auf False setzen. Wenn die zu trainierenden Parameter von dieser Schicht initialisiert werden müssen, ist es True, der Standard ist False.
    :param init_params: Initialisierungsparameter zum Codieren klassischer Daten QTensor, der Standard ist None.
    :param wires: Bit-Index des Leitungseffekts, der Standard ist None.
    :param dtype: Die Datenpräzision der internen Matrix des Logikgatters kann auf pyvqnet.kcomplex64 oder pyvqnet.kcomplex128 gesetzt werden, entsprechend float- bzw. double-Eingabe.
    :param use_dagger: ob die transponiert-konjugierte Version des Gatters verwendet werden soll, der Standard ist False.
    :return: eine ``pyvqnet.qnn.vqc.torch.QModule``-Instanz

    Example::

        from pyvqnet.qnn.vqc.torch import CRY,QMachine
        import pyvqnet
        pyvqnet.backends.set_backend("torch")
        device = QMachine(4)
        layer = CRY(has_params= True, trainable= True, wires=[0,2])
        batchsize = 2
        device.reset_states(batchsize)
        layer(q_machine = device)
        print(device.states)


CRZ
""""""""""""""""""


.. py:class:: pyvqnet.qnn.vqc.torch.CRZ(has_params: bool = False,trainable: bool = False,init_params=None,wires=None,dtype=pyvqnet.kcomplex64,use_dagger=False)
    
    Definiert ein CRZ-Quantengatter.

    Diese Klasse erbt von ``pyvqnet.qnn.vqc.torch.QModule`` und ``torch.nn.Module``.
    Diese Klasse kann als Untermodul von ``torch.nn.Module`` zum torch-Modell hinzugefügt werden.

    :param has_params: ob es Parameter hat, wie RX, RY und andere Gatter auf True gesetzt werden müssen, und solche ohne Parameter auf False gesetzt werden müssen, der Standard ist False.
    :param trainable: ob es zu trainierende Parameter hat. Wenn die Schicht externe Eingabedaten zum Aufbau der Logikgatter-Matrix verwendet, auf False setzen. Wenn die zu trainierenden Parameter von dieser Schicht initialisiert werden müssen, ist es True, der Standard ist False.
    :param init_params: Initialisierungsparameter zum Codieren klassischer Daten QTensor, der Standard ist None.
    :param wires: Bit-Index des Leitungseffekts, der Standard ist None.
    :param dtype: Die Datenpräzision der internen Matrix des Logikgatters kann auf pyvqnet.kcomplex64 oder pyvqnet.kcomplex128 gesetzt werden, entsprechend float- bzw. double-Eingabe.
    :param use_dagger: ob die transponiert-konjugierte Version des Gatters verwendet werden soll, der Standard ist False.
    :return: eine ``pyvqnet.qnn.vqc.torch.QModule``-Instanz

    Example::

        from pyvqnet.qnn.vqc.torch import CRZ,QMachine
        import pyvqnet
        pyvqnet.backends.set_backend("torch")
        device = QMachine(4)
        layer = CRZ(has_params= True, trainable= True, wires=[0,2])
        batchsize = 2
        device.reset_states(batchsize)
        layer(q_machine = device)
        print(device.states)



U1
""""""""""""""""""

.. py:class:: pyvqnet.qnn.vqc.torch.U1(has_params: bool = False,trainable: bool = False,init_params=None,wires=None,dtype=pyvqnet.kcomplex64,use_dagger=False)
    
    Definiert ein U1-Quantengatter.

    Diese Klasse erbt von ``pyvqnet.qnn.vqc.torch.QModule`` und ``torch.nn.Module``.
    Diese Klasse kann als Untermodul von ``torch.nn.Module`` zum torch-Modell hinzugefügt werden.

    :param has_params: ob es Parameter hat, wie RX, RY und andere Gatter auf True gesetzt werden müssen, und solche ohne Parameter auf False gesetzt werden müssen, der Standard ist False.
    :param trainable: ob es zu trainierende Parameter hat. Wenn die Schicht externe Eingabedaten zum Aufbau der Logikgatter-Matrix verwendet, auf False setzen. Wenn die zu trainierenden Parameter von dieser Schicht initialisiert werden müssen, ist es True, der Standard ist False.
    :param init_params: Initialisierungsparameter zum Codieren klassischer Daten QTensor, der Standard ist None.
    :param wires: Bit-Index des Leitungseffekts, der Standard ist None.
    :param dtype: Die Datenpräzision der internen Matrix des Logikgatters kann auf pyvqnet.kcomplex64 oder pyvqnet.kcomplex128 gesetzt werden, entsprechend float- bzw. double-Eingabe.
    :param use_dagger: ob die transponiert-konjugierte Version des Gatters verwendet werden soll, der Standard ist False.
    :return: eine ``pyvqnet.qnn.vqc.torch.QModule``-Instanz

    Example::

        from pyvqnet.qnn.vqc.torch import U1,QMachine
        import pyvqnet
        pyvqnet.backends.set_backend("torch")
        device = QMachine(4)
        layer = U1(has_params= True, trainable= True, wires=0)
        batchsize = 2
        device.reset_states(batchsize)
        layer(q_machine = device)
        print(device.states)

U2
""""""""""""""""""


.. py:class:: pyvqnet.qnn.vqc.torch.U2(has_params: bool = False,trainable: bool = False,init_params=None,wires=None,dtype=pyvqnet.kcomplex64,use_dagger=False)
    
    Definiert ein U2-Quantengatter.

    Diese Klasse erbt von ``pyvqnet.qnn.vqc.torch.QModule`` und ``torch.nn.Module``.
    Diese Klasse kann als Untermodul von ``torch.nn.Module`` zum torch-Modell hinzugefügt werden.

    :param has_params: ob es Parameter hat, wie RX, RY und andere Gatter auf True gesetzt werden müssen, und solche ohne Parameter auf False gesetzt werden müssen, der Standard ist False.
    :param trainable: ob es zu trainierende Parameter hat. Wenn die Schicht externe Eingabedaten zum Aufbau der Logikgatter-Matrix verwendet, auf False setzen. Wenn die zu trainierenden Parameter von dieser Schicht initialisiert werden müssen, ist es True, der Standard ist False.
    :param init_params: Initialisierungsparameter zum Codieren klassischer Daten QTensor, der Standard ist None.
    :param wires: Bit-Index des Leitungseffekts, der Standard ist None.
    :param dtype: Die Datenpräzision der internen Matrix des Logikgatters kann auf pyvqnet.kcomplex64 oder pyvqnet.kcomplex128 gesetzt werden, entsprechend float- bzw. double-Eingabe.
    :param use_dagger: ob die transponiert-konjugierte Version des Gatters verwendet werden soll, der Standard ist False.
    :return: eine ``pyvqnet.qnn.vqc.torch.QModule``-Instanz

    Example::

        from pyvqnet.qnn.vqc.torch import U2,QMachine
        import pyvqnet
        pyvqnet.backends.set_backend("torch")
        device = QMachine(4)
        layer = U2(has_params= True, trainable= True, wires=1)
        batchsize = 2
        device.reset_states(batchsize)
        layer(q_machine = device)
        print(device.states)


U3
""""""""""""""""""


.. py:class:: pyvqnet.qnn.vqc.torch.U3(has_params: bool = False,trainable: bool = False,init_params=None,wires=None,dtype=pyvqnet.kcomplex64,use_dagger=False)
    
    Definiert ein U3-Quantengatter.

    Diese Klasse erbt von ``pyvqnet.qnn.vqc.torch.QModule`` und ``torch.nn.Module``.
    Diese Klasse kann als Untermodul von ``torch.nn.Module`` zum torch-Modell hinzugefügt werden.

    :param has_params: ob es Parameter hat, wie RX, RY und andere Gatter auf True gesetzt werden müssen, und solche ohne Parameter auf False gesetzt werden müssen, der Standard ist False.
    :param trainable: ob es zu trainierende Parameter hat. Wenn die Schicht externe Eingabedaten zum Aufbau der Logikgatter-Matrix verwendet, auf False setzen. Wenn die zu trainierenden Parameter von dieser Schicht initialisiert werden müssen, ist es True, der Standard ist False.
    :param init_params: Initialisierungsparameter zum Codieren klassischer Daten QTensor, der Standard ist None.
    :param wires: Bit-Index des Leitungseffekts, der Standard ist None.
    :param dtype: Die Datenpräzision der internen Matrix des Logikgatters kann auf pyvqnet.kcomplex64 oder pyvqnet.kcomplex128 gesetzt werden, entsprechend float- bzw. double-Eingabe.
    :param use_dagger: ob die transponiert-konjugierte Version des Gatters verwendet werden soll, der Standard ist False.
    :return: eine ``pyvqnet.qnn.vqc.torch.QModule``-Instanz

    Example::

        from pyvqnet.qnn.vqc.torch import U3,QMachine
        import pyvqnet
        pyvqnet.backends.set_backend("torch")
        device = QMachine(4)
        layer = U3(has_params= True, trainable= True, wires=1)
        batchsize = 2
        device.reset_states(batchsize)
        layer(q_machine = device)
        print(device.states)



CNOT
""""""""""""""""""

.. py:class:: pyvqnet.qnn.vqc.torch.CNOT(has_params: bool = False,trainable: bool = False,init_params=None,wires=None,dtype=pyvqnet.kcomplex64,use_dagger=False)
    
    Definiert ein CNOT-Quantengatter, Alias `CX`.

    Diese Klasse erbt von ``pyvqnet.qnn.vqc.torch.QModule`` und ``torch.nn.Module``.
    Diese Klasse kann als Untermodul von ``torch.nn.Module`` zum torch-Modell hinzugefügt werden.

    :param has_params: ob es Parameter hat, wie RX, RY und andere Gatter auf True gesetzt werden müssen, und solche ohne Parameter auf False gesetzt werden müssen, der Standard ist False.
    :param trainable: ob es zu trainierende Parameter hat. Wenn die Schicht externe Eingabedaten zum Aufbau der Logikgatter-Matrix verwendet, auf False setzen. Wenn die zu trainierenden Parameter von dieser Schicht initialisiert werden müssen, ist es True, der Standard ist False.
    :param init_params: Initialisierungsparameter zum Codieren klassischer Daten QTensor, der Standard ist None.
    :param wires: Bit-Index des Leitungseffekts, der Standard ist None.
    :param dtype: Die Datenpräzision der internen Matrix des Logikgatters kann auf pyvqnet.kcomplex64 oder pyvqnet.kcomplex128 gesetzt werden, entsprechend float- bzw. double-Eingabe.
    :param use_dagger: ob die transponiert-konjugierte Version des Gatters verwendet werden soll, der Standard ist False.
    :return: eine ``pyvqnet.qnn.vqc.torch.QModule``-Instanz

    Example::

        from pyvqnet.qnn.vqc.torch import CNOT,QMachine
        import pyvqnet
        pyvqnet.backends.set_backend("torch")
        device = QMachine(4)
        layer = CNOT(wires=[0,1])
        batchsize = 2
        device.reset_states(batchsize)
        layer(q_machine = device)
        print(device.states)

CY
""""""""""""""""""

.. py:class:: pyvqnet.qnn.vqc.torch.CY(has_params: bool = False,trainable: bool = False,init_params=None,wires=None,dtype=pyvqnet.kcomplex64,use_dagger=False)
    
    Definiert ein CY-Quantengatter.

    Diese Klasse erbt von ``pyvqnet.qnn.vqc.torch.QModule`` und ``torch.nn.Module``.
    Diese Klasse kann als Untermodul von ``torch.nn.Module`` zum torch-Modell hinzugefügt werden.

    :param has_params: ob es Parameter hat, wie RX, RY und andere Gatter auf True gesetzt werden müssen, und solche ohne Parameter auf False gesetzt werden müssen, der Standard ist False.
    :param trainable: ob es zu trainierende Parameter hat. Wenn die Schicht externe Eingabedaten zum Aufbau der Logikgatter-Matrix verwendet, auf False setzen. Wenn die zu trainierenden Parameter von dieser Schicht initialisiert werden müssen, ist es True, der Standard ist False.
    :param init_params: Initialisierungsparameter zum Codieren klassischer Daten QTensor, der Standard ist None.
    :param wires: Bit-Index des Leitungseffekts, der Standard ist None.
    :param dtype: Die Datenpräzision der internen Matrix des Logikgatters kann auf pyvqnet.kcomplex64 oder pyvqnet.kcomplex128 gesetzt werden, entsprechend float- bzw. double-Eingabe.
    :param use_dagger: ob die transponiert-konjugierte Version des Gatters verwendet werden soll, der Standard ist False.
    :return: eine ``pyvqnet.qnn.vqc.torch.QModule``-Instanz

    Example::

        from pyvqnet.qnn.vqc.torch import CY,QMachine
        import pyvqnet
        pyvqnet.backends.set_backend("torch")
        device = QMachine(4)
        layer = CY(wires=[0,1])
        batchsize = 2
        device.reset_states(batchsize)
        layer(q_machine = device)
        print(device.states)


CZ
""""""""""""""""""

.. py:class:: pyvqnet.qnn.vqc.torch.CZ(has_params: bool = False,trainable: bool = False,init_params=None,wires=None,dtype=pyvqnet.kcomplex64,use_dagger=False)
    
    Definiert ein CZ-Quantengatter.

    Diese Klasse erbt von ``pyvqnet.qnn.vqc.torch.QModule`` und ``torch.nn.Module``.
    Diese Klasse kann als Untermodul von ``torch.nn.Module`` zum torch-Modell hinzugefügt werden.

    :param has_params: ob es Parameter hat, wie RX, RY und andere Gatter auf True gesetzt werden müssen, und solche ohne Parameter auf False gesetzt werden müssen, der Standard ist False.
    :param trainable: ob es zu trainierende Parameter hat. Wenn die Schicht externe Eingabedaten zum Aufbau der Logikgatter-Matrix verwendet, auf False setzen. Wenn die zu trainierenden Parameter von dieser Schicht initialisiert werden müssen, ist es True, der Standard ist False.
    :param init_params: Initialisierungsparameter zum Codieren klassischer Daten QTensor, der Standard ist None.
    :param wires: Bit-Index des Leitungseffekts, der Standard ist None.
    :param dtype: Die Datenpräzision der internen Matrix des Logikgatters kann auf pyvqnet.kcomplex64 oder pyvqnet.kcomplex128 gesetzt werden, entsprechend float- bzw. double-Eingabe.
    :param use_dagger: ob die transponiert-konjugierte Version des Gatters verwendet werden soll, der Standard ist False.
    :return: eine ``pyvqnet.qnn.vqc.torch.QModule``-Instanz

    Example::

        from pyvqnet.qnn.vqc.torch import CZ,QMachine
        import pyvqnet
        pyvqnet.backends.set_backend("torch")
        device = QMachine(4)
        layer = CZ(wires=[0,1])
        batchsize = 2
        device.reset_states(batchsize)
        layer(q_machine = device)
        print(device.states)




CR
""""""""""""""""""

.. py:class:: pyvqnet.qnn.vqc.torch.CR(has_params: bool = False,trainable: bool = False,init_params=None,wires=None,dtype=pyvqnet.kcomplex64,use_dagger=False)
    
    Definiert ein CR-Quantengatter.

    Diese Klasse erbt von ``pyvqnet.qnn.vqc.torch.QModule`` und ``torch.nn.Module``.
    Diese Klasse kann als Untermodul von ``torch.nn.Module`` zum torch-Modell hinzugefügt werden.

    :param has_params: ob es Parameter hat, wie RX, RY und andere Gatter auf True gesetzt werden müssen, und solche ohne Parameter auf False gesetzt werden müssen, der Standard ist False.
    :param trainable: ob es zu trainierende Parameter hat. Wenn die Schicht externe Eingabedaten zum Aufbau der Logikgatter-Matrix verwendet, auf False setzen. Wenn die zu trainierenden Parameter von dieser Schicht initialisiert werden müssen, ist es True, der Standard ist False.
    :param init_params: Initialisierungsparameter zum Codieren klassischer Daten QTensor, der Standard ist None.
    :param wires: Bit-Index des Leitungseffekts, der Standard ist None.
    :param dtype: Die Datenpräzision der internen Matrix des Logikgatters kann auf pyvqnet.kcomplex64 oder pyvqnet.kcomplex128 gesetzt werden, entsprechend float- bzw. double-Eingabe.
    :param use_dagger: ob die transponiert-konjugierte Version des Gatters verwendet werden soll, der Standard ist False.
    :return: eine ``pyvqnet.qnn.vqc.torch.QModule``-Instanz

    Example::

        from pyvqnet.qnn.vqc.torch import CR,QMachine
        import pyvqnet
        pyvqnet.backends.set_backend("torch")
        device = QMachine(4)
        device = QMachine(4)
        layer = CR(has_params= True, trainable= True, wires=[0,2])
        batchsize = 2
        device.reset_states(batchsize)
        layer(q_machine = device)
        print(device.states)



SWAP
""""""""""""""""""


.. py:class:: pyvqnet.qnn.vqc.torch.SWAP(has_params: bool = False,trainable: bool = False,init_params=None,wires=None,dtype=pyvqnet.kcomplex64,use_dagger=False)
    
    Definiert ein SWAP-Quantengatter.

    Diese Klasse erbt von ``pyvqnet.qnn.vqc.torch.QModule`` und ``torch.nn.Module``.
    Diese Klasse kann als Untermodul von ``torch.nn.Module`` zum torch-Modell hinzugefügt werden.

    :param has_params: ob es Parameter hat, wie RX, RY und andere Gatter auf True gesetzt werden müssen, und solche ohne Parameter auf False gesetzt werden müssen, der Standard ist False.
    :param trainable: ob es zu trainierende Parameter hat. Wenn die Schicht externe Eingabedaten zum Aufbau der Logikgatter-Matrix verwendet, auf False setzen. Wenn die zu trainierenden Parameter von dieser Schicht initialisiert werden müssen, ist es True, der Standard ist False.
    :param init_params: Initialisierungsparameter zum Codieren klassischer Daten QTensor, der Standard ist None.
    :param wires: Bit-Index des Leitungseffekts, der Standard ist None.
    :param dtype: Die Datenpräzision der internen Matrix des Logikgatters kann auf pyvqnet.kcomplex64 oder pyvqnet.kcomplex128 gesetzt werden, entsprechend float- bzw. double-Eingabe.
    :param use_dagger: ob die transponiert-konjugierte Version des Gatters verwendet werden soll, der Standard ist False.
    :return: eine ``pyvqnet.qnn.vqc.torch.QModule``-Instanz

    Example::

        from pyvqnet.qnn.vqc.torch import SWAP,QMachine
        import pyvqnet
        pyvqnet.backends.set_backend("torch")
        device = QMachine(4)
        layer = SWAP(wires=[0,1])
        batchsize = 2
        device.reset_states(batchsize)
        layer(q_machine = device)
        print(device.states)


CSWAP
""""""""""""""""""

.. py:class:: pyvqnet.qnn.vqc.torch.CSWAP(has_params: bool = False,trainable: bool = False,init_params=None,wires=None,dtype=pyvqnet.kcomplex64,use_dagger=False)
    
    Definiert ein SWAP-Quantengatter.

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

    Diese Klasse erbt von ``pyvqnet.qnn.vqc.torch.QModule`` und ``torch.nn.Module``.
    Diese Klasse kann als Untermodul von ``torch.nn.Module`` zum torch-Modell hinzugefügt werden.

    :param has_params: ob es Parameter hat, wie RX, RY und andere Gatter auf True gesetzt werden müssen, und solche ohne Parameter auf False gesetzt werden müssen, der Standard ist False.
    :param trainable: ob es zu trainierende Parameter hat. Wenn die Schicht externe Eingabedaten zum Aufbau der Logikgatter-Matrix verwendet, auf False setzen. Wenn die zu trainierenden Parameter von dieser Schicht initialisiert werden müssen, ist es True, der Standard ist False.
    :param init_params: Initialisierungsparameter zum Codieren klassischer Daten QTensor, der Standard ist None.
    :param wires: Bit-Index des Leitungseffekts, der Standard ist None.
    :param dtype: Die Datenpräzision der internen Matrix des Logikgatters kann auf pyvqnet.kcomplex64 oder pyvqnet.kcomplex128 gesetzt werden, entsprechend float- bzw. double-Eingabe.
    :param use_dagger: ob die transponiert-konjugierte Version des Gatters verwendet werden soll, der Standard ist False.
    :return: eine ``pyvqnet.qnn.vqc.torch.QModule``-Instanz

    Example::

        from pyvqnet.qnn.vqc.torch import CSWAP,QMachine
        import pyvqnet
        pyvqnet.backends.set_backend("torch")
        device = QMachine(4)
        layer = CSWAP(wires=[0,1,2])
        batchsize = 2
        device.reset_states(batchsize)
        layer(q_machine = device)
        print(device.states)

RXX
""""""""""""""""""


.. py:class:: pyvqnet.qnn.vqc.torch.RXX(has_params: bool = False,trainable: bool = False,init_params=None,wires=None,dtype=pyvqnet.kcomplex64,use_dagger=False)
    
    Definiert ein RXX-Quantengatter.

    Diese Klasse erbt von ``pyvqnet.qnn.vqc.torch.QModule`` und ``torch.nn.Module``.
    Diese Klasse kann als Untermodul von ``torch.nn.Module`` zum torch-Modell hinzugefügt werden.

    :param has_params: ob es Parameter hat, wie RX, RY und andere Gatter auf True gesetzt werden müssen, und solche ohne Parameter auf False gesetzt werden müssen, der Standard ist False.
    :param trainable: ob es zu trainierende Parameter hat. Wenn die Schicht externe Eingabedaten zum Aufbau der Logikgatter-Matrix verwendet, auf False setzen. Wenn die zu trainierenden Parameter von dieser Schicht initialisiert werden müssen, ist es True, der Standard ist False.
    :param init_params: Initialisierungsparameter zum Codieren klassischer Daten QTensor, der Standard ist None.
    :param wires: Bit-Index des Leitungseffekts, der Standard ist None.
    :param dtype: Die Datenpräzision der internen Matrix des Logikgatters kann auf pyvqnet.kcomplex64 oder pyvqnet.kcomplex128 gesetzt werden, entsprechend float- bzw. double-Eingabe.
    :param use_dagger: ob die transponiert-konjugierte Version des Gatters verwendet werden soll, der Standard ist False.
    :return: eine ``pyvqnet.qnn.vqc.torch.QModule``-Instanz

    Example::

        from pyvqnet.qnn.vqc.torch import RXX,QMachine
        import pyvqnet
        pyvqnet.backends.set_backend("torch")
        device = QMachine(4)
        layer = RXX(has_params= True, trainable= True, wires=[0,2])
        batchsize = 2
        device.reset_states(batchsize)
        layer(q_machine = device)
        print(device.states)

RYY
""""""""""""""""""

.. py:class:: pyvqnet.qnn.vqc.torch.RYY(has_params: bool = False,trainable: bool = False,init_params=None,wires=None,dtype=pyvqnet.kcomplex64,use_dagger=False)
    
    Definiert ein RYY-Quantengatter.

    Diese Klasse erbt von ``pyvqnet.qnn.vqc.torch.QModule`` und ``torch.nn.Module``.
    Diese Klasse kann als Untermodul von ``torch.nn.Module`` zum torch-Modell hinzugefügt werden.

    :param has_params: ob es Parameter hat, wie RX, RY und andere Gatter auf True gesetzt werden müssen, und solche ohne Parameter auf False gesetzt werden müssen, der Standard ist False.
    :param trainable: ob es zu trainierende Parameter hat. Wenn die Schicht externe Eingabedaten zum Aufbau der Logikgatter-Matrix verwendet, auf False setzen. Wenn die zu trainierenden Parameter von dieser Schicht initialisiert werden müssen, ist es True, der Standard ist False.
    :param init_params: Initialisierungsparameter zum Codieren klassischer Daten QTensor, der Standard ist None.
    :param wires: Bit-Index des Leitungseffekts, der Standard ist None.
    :param dtype: Die Datenpräzision der internen Matrix des Logikgatters kann auf pyvqnet.kcomplex64 oder pyvqnet.kcomplex128 gesetzt werden, entsprechend float- bzw. double-Eingabe.
    :param use_dagger: ob die transponiert-konjugierte Version des Gatters verwendet werden soll, der Standard ist False.
    :return: eine ``pyvqnet.qnn.vqc.torch.QModule``-Instanz

    Example::

        from pyvqnet.qnn.vqc.torch import RYY,QMachine
        import pyvqnet
        pyvqnet.backends.set_backend("torch")
        device = QMachine(4)
        layer = RYY(has_params= True, trainable= True, wires=[0,2])
        batchsize = 2
        device.reset_states(batchsize)
        layer(q_machine = device)
        print(device.states)


RZZ
""""""""""""""""""

.. py:class:: pyvqnet.qnn.vqc.torch.RZZ(has_params: bool = False,trainable: bool = False,init_params=None,wires=None,dtype=pyvqnet.kcomplex64,use_dagger=False)
    
    Definiert ein RZZ-Quantengatter.

    Diese Klasse erbt von ``pyvqnet.qnn.vqc.torch.QModule`` und ``torch.nn.Module``.
    Diese Klasse kann als Untermodul von ``torch.nn.Module`` zum torch-Modell hinzugefügt werden.

    :param has_params: ob es Parameter hat, wie RX, RY und andere Gatter auf True gesetzt werden müssen, und solche ohne Parameter auf False gesetzt werden müssen, der Standard ist False.
    :param trainable: ob es zu trainierende Parameter hat. Wenn die Schicht externe Eingabedaten zum Aufbau der Logikgatter-Matrix verwendet, auf False setzen. Wenn die zu trainierenden Parameter von dieser Schicht initialisiert werden müssen, ist es True, der Standard ist False.
    :param init_params: Initialisierungsparameter zum Codieren klassischer Daten QTensor, der Standard ist None.
    :param wires: Bit-Index des Leitungseffekts, der Standard ist None.
    :param dtype: Die Datenpräzision der internen Matrix des Logikgatters kann auf pyvqnet.kcomplex64 oder pyvqnet.kcomplex128 gesetzt werden, entsprechend float- bzw. double-Eingabe.
    :param use_dagger: ob die transponiert-konjugierte Version des Gatters verwendet werden soll, der Standard ist False.
    :return: eine ``pyvqnet.qnn.vqc.torch.QModule``-Instanz

    Example::

        from pyvqnet.qnn.vqc.torch import RZZ,QMachine
        import pyvqnet
        pyvqnet.backends.set_backend("torch")
        device = QMachine(4)
        layer = RZZ(has_params= True, trainable= True, wires=[0,2])
        batchsize = 2
        device.reset_states(batchsize)
        layer(q_machine = device)
        print(device.states)



RZX
""""""""""""""""""

.. py:class:: pyvqnet.qnn.vqc.torch.RZX(has_params: bool = False,trainable: bool = False,init_params=None,wires=None,dtype=pyvqnet.kcomplex64,use_dagger=False)
    
    Definiert ein RZX-Quantengatter.

    Diese Klasse erbt von ``pyvqnet.qnn.vqc.torch.QModule`` und ``torch.nn.Module``.
    Diese Klasse kann als Untermodul von ``torch.nn.Module`` zum torch-Modell hinzugefügt werden.

    :param has_params: ob es Parameter hat, wie RX, RY und andere Gatter auf True gesetzt werden müssen, und solche ohne Parameter auf False gesetzt werden müssen, der Standard ist False.
    :param trainable: ob es zu trainierende Parameter hat. Wenn die Schicht externe Eingabedaten zum Aufbau der Logikgatter-Matrix verwendet, auf False setzen. Wenn die zu trainierenden Parameter von dieser Schicht initialisiert werden müssen, ist es True, der Standard ist False.
    :param init_params: Initialisierungsparameter zum Codieren klassischer Daten QTensor, der Standard ist None.
    :param wires: Bit-Index des Leitungseffekts, der Standard ist None.
    :param dtype: Die Datenpräzision der internen Matrix des Logikgatters kann auf pyvqnet.kcomplex64 oder pyvqnet.kcomplex128 gesetzt werden, entsprechend float- bzw. double-Eingabe.
    :param use_dagger: ob die transponiert-konjugierte Version des Gatters verwendet werden soll, der Standard ist False.
    :return: eine ``pyvqnet.qnn.vqc.torch.QModule``-Instanz

    Example::

        from pyvqnet.qnn.vqc.torch import RZX,QMachine
        import pyvqnet
        pyvqnet.backends.set_backend("torch")
        device = QMachine(4)
        layer = RZX(has_params= True, trainable= True, wires=[0,2])
        batchsize = 2
        device.reset_states(batchsize)
        layer(q_machine = device)
        print(device.states)

Toffoli
""""""""""""""""""

.. py:class:: pyvqnet.qnn.vqc.torch.Toffoli(has_params: bool = False,trainable: bool = False,init_params=None,wires=None,dtype=pyvqnet.kcomplex64,use_dagger=False)
    
    Definiert ein Toffoli-Quantengatter.

    Diese Klasse erbt von ``pyvqnet.qnn.vqc.torch.QModule`` und ``torch.nn.Module``.
    Diese Klasse kann als Untermodul von ``torch.nn.Module`` zum torch-Modell hinzugefügt werden.

    :param has_params: ob es Parameter hat, wie RX, RY und andere Gatter auf True gesetzt werden müssen, und solche ohne Parameter auf False gesetzt werden müssen, der Standard ist False.
    :param trainable: ob es zu trainierende Parameter hat. Wenn die Schicht externe Eingabedaten zum Aufbau der Logikgatter-Matrix verwendet, auf False setzen. Wenn die zu trainierenden Parameter von dieser Schicht initialisiert werden müssen, ist es True, der Standard ist False.
    :param init_params: Initialisierungsparameter zum Codieren klassischer Daten QTensor, der Standard ist None.
    :param wires: Bit-Index des Leitungseffekts, der Standard ist None.
    :param dtype: Die Datenpräzision der internen Matrix des Logikgatters kann auf pyvqnet.kcomplex64 oder pyvqnet.kcomplex128 gesetzt werden, entsprechend float- bzw. double-Eingabe.
    :param use_dagger: ob die transponiert-konjugierte Version des Gatters verwendet werden soll, der Standard ist False.
    :return: eine ``pyvqnet.qnn.vqc.torch.QModule``-Instanz

    Example::

        from pyvqnet.qnn.vqc.torch import Toffoli,QMachine
        import pyvqnet
        pyvqnet.backends.set_backend("torch")
        device = QMachine(4)
        layer = Toffoli(wires=[0,2,1])
        batchsize = 2
        device.reset_states(batchsize)
        layer(q_machine = device)
        print(device.states)

IsingXX
""""""""""""""""""


.. py:class:: pyvqnet.qnn.vqc.torch.IsingXX(has_params: bool = False,trainable: bool = False,init_params=None,wires=None,dtype=pyvqnet.kcomplex64,use_dagger=False)
    
    Definiert ein IsingXX-Quantengatter.

    Diese Klasse erbt von ``pyvqnet.qnn.vqc.torch.QModule`` und ``torch.nn.Module``.
    Diese Klasse kann als Untermodul von ``torch.nn.Module`` zum torch-Modell hinzugefügt werden.

    :param has_params: ob es Parameter hat, wie RX, RY und andere Gatter auf True gesetzt werden müssen, und solche ohne Parameter auf False gesetzt werden müssen, der Standard ist False.
    :param trainable: ob es zu trainierende Parameter hat. Wenn die Schicht externe Eingabedaten zum Aufbau der Logikgatter-Matrix verwendet, auf False setzen. Wenn die zu trainierenden Parameter von dieser Schicht initialisiert werden müssen, ist es True, der Standard ist False.
    :param init_params: Initialisierungsparameter zum Codieren klassischer Daten QTensor, der Standard ist None.
    :param wires: Bit-Index des Leitungseffekts, der Standard ist None.
    :param dtype: Die Datenpräzision der internen Matrix des Logikgatters kann auf pyvqnet.kcomplex64 oder pyvqnet.kcomplex128 gesetzt werden, entsprechend float- bzw. double-Eingabe.
    :param use_dagger: ob die transponiert-konjugierte Version des Gatters verwendet werden soll, der Standard ist False.
    :return: eine ``pyvqnet.qnn.vqc.torch.QModule``-Instanz

    Example::

        from pyvqnet.qnn.vqc.torch import IsingXX,QMachine
        import pyvqnet
        pyvqnet.backends.set_backend("torch")
        device = QMachine(4)
        layer = IsingXX(has_params= True, trainable= True, wires=[0,2])
        batchsize = 2
        device.reset_states(batchsize)
        layer(q_machine = device)
        print(device.states)


IsingYY
""""""""""""""""""


.. py:class:: pyvqnet.qnn.vqc.torch.IsingYY(has_params: bool = False,trainable: bool = False,init_params=None,wires=None,dtype=pyvqnet.kcomplex64,use_dagger=False)
    
    Definiert ein IsingYY-Quantengatter.

    Diese Klasse erbt von ``pyvqnet.qnn.vqc.torch.QModule`` und ``torch.nn.Module``.
    Diese Klasse kann als Untermodul von ``torch.nn.Module`` zum torch-Modell hinzugefügt werden.

    :param has_params: ob es Parameter hat, wie RX, RY und andere Gatter auf True gesetzt werden müssen, und solche ohne Parameter auf False gesetzt werden müssen, der Standard ist False.
    :param trainable: ob es zu trainierende Parameter hat. Wenn die Schicht externe Eingabedaten zum Aufbau der Logikgatter-Matrix verwendet, auf False setzen. Wenn die zu trainierenden Parameter von dieser Schicht initialisiert werden müssen, ist es True, der Standard ist False.
    :param init_params: Initialisierungsparameter zum Codieren klassischer Daten QTensor, der Standard ist None.
    :param wires: Bit-Index des Leitungseffekts, der Standard ist None.
    :param dtype: Die Datenpräzision der internen Matrix des Logikgatters kann auf pyvqnet.kcomplex64 oder pyvqnet.kcomplex128 gesetzt werden, entsprechend float- bzw. double-Eingabe.
    :param use_dagger: ob die transponiert-konjugierte Version des Gatters verwendet werden soll, der Standard ist False.
    :return: eine ``pyvqnet.qnn.vqc.torch.QModule``-Instanz

    Example::

        from pyvqnet.qnn.vqc.torch import IsingYY,QMachine
        import pyvqnet
        pyvqnet.backends.set_backend("torch")
        device = QMachine(4)
        layer = IsingYY(has_params= True, trainable= True, wires=[0,2])
        batchsize = 2
        device.reset_states(batchsize)
        layer(q_machine = device)
        print(device.states)


IsingZZ
""""""""""""""""""

.. py:class:: pyvqnet.qnn.vqc.torch.IsingZZ(has_params: bool = False,trainable: bool = False,init_params=None,wires=None,dtype=pyvqnet.kcomplex64,use_dagger=False)
    
    Definiert ein IsingZZ-Quantengatter.

    Diese Klasse erbt von ``pyvqnet.qnn.vqc.torch.QModule`` und ``torch.nn.Module``.
    Diese Klasse kann als Untermodul von ``torch.nn.Module`` zum torch-Modell hinzugefügt werden.

    :param has_params: ob es Parameter hat, wie RX, RY und andere Gatter auf True gesetzt werden müssen, und solche ohne Parameter auf False gesetzt werden müssen, der Standard ist False.
    :param trainable: ob es zu trainierende Parameter hat. Wenn die Schicht externe Eingabedaten zum Aufbau der Logikgatter-Matrix verwendet, auf False setzen. Wenn die zu trainierenden Parameter von dieser Schicht initialisiert werden müssen, ist es True, der Standard ist False.
    :param init_params: Initialisierungsparameter zum Codieren klassischer Daten QTensor, der Standard ist None.
    :param wires: Bit-Index des Leitungseffekts, der Standard ist None.
    :param dtype: Die Datenpräzision der internen Matrix des Logikgatters kann auf pyvqnet.kcomplex64 oder pyvqnet.kcomplex128 gesetzt werden, entsprechend float- bzw. double-Eingabe.
    :param use_dagger: ob die transponiert-konjugierte Version des Gatters verwendet werden soll, der Standard ist False.
    :return: eine ``pyvqnet.qnn.vqc.torch.QModule``-Instanz

    Example::

        from pyvqnet.qnn.vqc.torch import IsingZZ,QMachine
        import pyvqnet
        pyvqnet.backends.set_backend("torch")
        device = QMachine(4)
        layer = IsingZZ(has_params= True, trainable= True, wires=[0,2])
        batchsize = 2
        device.reset_states(batchsize)
        layer(q_machine = device)
        print(device.states)


IsingXY
""""""""""""""""""


.. py:class:: pyvqnet.qnn.vqc.torch.IsingXY(has_params: bool = False,trainable: bool = False,init_params=None,wires=None,dtype=pyvqnet.kcomplex64,use_dagger=False)
    
    Definiert ein IsingXY-Quantengatter.

    Diese Klasse erbt von ``pyvqnet.qnn.vqc.torch.QModule`` und ``torch.nn.Module``.
    Diese Klasse kann als Untermodul von ``torch.nn.Module`` zum torch-Modell hinzugefügt werden.

    :param has_params: ob es Parameter hat, wie RX, RY und andere Gatter auf True gesetzt werden müssen, und solche ohne Parameter auf False gesetzt werden müssen, der Standard ist False.
    :param trainable: ob es zu trainierende Parameter hat. Wenn die Schicht externe Eingabedaten zum Aufbau der Logikgatter-Matrix verwendet, auf False setzen. Wenn die zu trainierenden Parameter von dieser Schicht initialisiert werden müssen, ist es True, der Standard ist False.
    :param init_params: Initialisierungsparameter zum Codieren klassischer Daten QTensor, der Standard ist None.
    :param wires: Bit-Index des Leitungseffekts, der Standard ist None.
    :param dtype: Die Datenpräzision der internen Matrix des Logikgatters kann auf pyvqnet.kcomplex64 oder pyvqnet.kcomplex128 gesetzt werden, entsprechend float- bzw. double-Eingabe.
    :param use_dagger: ob die transponiert-konjugierte Version des Gatters verwendet werden soll, der Standard ist False.
    :return: eine ``pyvqnet.qnn.vqc.torch.QModule``-Instanz

    Example::

        from pyvqnet.qnn.vqc.torch import IsingXY,QMachine
        import pyvqnet
        pyvqnet.backends.set_backend("torch")
        device = QMachine(4)
        layer = IsingXY(has_params= True, trainable= True, wires=[0,2])
        batchsize = 2
        device.reset_states(batchsize)
        layer(q_machine = device)
        print(device.states)


PhaseShift
""""""""""""""""""


.. py:class:: pyvqnet.qnn.vqc.torch.PhaseShift(has_params: bool = False,trainable: bool = False,init_params=None,wires=None,dtype=pyvqnet.kcomplex64,use_dagger=False)
    
    Definiert ein PhaseShift-Quantengatter.

    Diese Klasse erbt von ``pyvqnet.qnn.vqc.torch.QModule`` und ``torch.nn.Module``.
    Diese Klasse kann als Untermodul von ``torch.nn.Module`` zum torch-Modell hinzugefügt werden.

    :param has_params: ob es Parameter hat, wie RX, RY und andere Gatter auf True gesetzt werden müssen, und solche ohne Parameter auf False gesetzt werden müssen, der Standard ist False.
    :param trainable: ob es zu trainierende Parameter hat. Wenn die Schicht externe Eingabedaten zum Aufbau der Logikgatter-Matrix verwendet, auf False setzen. Wenn die zu trainierenden Parameter von dieser Schicht initialisiert werden müssen, ist es True, der Standard ist False.
    :param init_params: Initialisierungsparameter zum Codieren klassischer Daten QTensor, der Standard ist None.
    :param wires: Bit-Index des Leitungseffekts, der Standard ist None.
    :param dtype: Die Datenpräzision der internen Matrix des Logikgatters kann auf pyvqnet.kcomplex64 oder pyvqnet.kcomplex128 gesetzt werden, entsprechend float- bzw. double-Eingabe.
    :param use_dagger: ob die transponiert-konjugierte Version des Gatters verwendet werden soll, der Standard ist False.
    :return: eine ``pyvqnet.qnn.vqc.torch.QModule``-Instanz

    Example::

        from pyvqnet.qnn.vqc.torch import PhaseShift,QMachine
        import pyvqnet
        pyvqnet.backends.set_backend("torch")
        device = QMachine(4)
        layer = PhaseShift(has_params= True, trainable= True, wires=1)
        batchsize = 2
        device.reset_states(batchsize)
        layer(q_machine = device)
        print(device.states)


MultiRZ
""""""""""""""""""

.. py:class:: pyvqnet.qnn.vqc.torch.MultiRZ(has_params: bool = False,trainable: bool = False,init_params=None,wires=None,dtype=pyvqnet.kcomplex64,use_dagger=False)
    
    Definiert ein MultiRZ-Quantengatter.

    Diese Klasse erbt von ``pyvqnet.qnn.vqc.torch.QModule`` und ``torch.nn.Module``.
    Diese Klasse kann als Untermodul von ``torch.nn.Module`` zum torch-Modell hinzugefügt werden.

    :param has_params: ob es Parameter hat, wie RX, RY und andere Gatter auf True gesetzt werden müssen, und solche ohne Parameter auf False gesetzt werden müssen, der Standard ist False.
    :param trainable: ob es zu trainierende Parameter hat. Wenn die Schicht externe Eingabedaten zum Aufbau der Logikgatter-Matrix verwendet, auf False setzen. Wenn die zu trainierenden Parameter von dieser Schicht initialisiert werden müssen, ist es True, der Standard ist False.
    :param init_params: Initialisierungsparameter zum Codieren klassischer Daten QTensor, der Standard ist None.
    :param wires: Bit-Index des Leitungseffekts, der Standard ist None.
    :param dtype: Die Datenpräzision der internen Matrix des Logikgatters kann auf pyvqnet.kcomplex64 oder pyvqnet.kcomplex128 gesetzt werden, entsprechend float- bzw. double-Eingabe.
    :param use_dagger: ob die transponiert-konjugierte Version des Gatters verwendet werden soll, der Standard ist False.
    :return: eine ``pyvqnet.qnn.vqc.torch.QModule``-Instanz

    Example::

        from pyvqnet.qnn.vqc.torch import MultiRZ,QMachine
        import pyvqnet
        pyvqnet.backends.set_backend("torch")
        device = QMachine(4)
        layer = MultiRZ(has_params= True, trainable= True, wires=[0,2])
        batchsize = 2
        device.reset_states(batchsize)
        layer(q_machine = device)
        print(device.states)



SDG
""""""""""""""""""


.. py:class:: pyvqnet.qnn.vqc.torch.SDG(has_params: bool = False,trainable: bool = False,init_params=None,wires=None,dtype=pyvqnet.kcomplex64,use_dagger=False)
    
    Definiert ein SDG-Quantengatter.

    Diese Klasse erbt von ``pyvqnet.qnn.vqc.torch.QModule`` und ``torch.nn.Module``.
    Diese Klasse kann als Untermodul von ``torch.nn.Module`` zum torch-Modell hinzugefügt werden.

    :param has_params: ob es Parameter hat, wie RX, RY und andere Gatter auf True gesetzt werden müssen, und solche ohne Parameter auf False gesetzt werden müssen, der Standard ist False.
    :param trainable: ob es zu trainierende Parameter hat. Wenn die Schicht externe Eingabedaten zum Aufbau der Logikgatter-Matrix verwendet, auf False setzen. Wenn die zu trainierenden Parameter von dieser Schicht initialisiert werden müssen, ist es True, der Standard ist False.
    :param init_params: Initialisierungsparameter zum Codieren klassischer Daten QTensor, der Standard ist None.
    :param wires: Bit-Index des Leitungseffekts, der Standard ist None.
    :param dtype: Die Datenpräzision der internen Matrix des Logikgatters kann auf pyvqnet.kcomplex64 oder pyvqnet.kcomplex128 gesetzt werden, entsprechend float- bzw. double-Eingabe.
    :param use_dagger: ob die transponiert-konjugierte Version des Gatters verwendet werden soll, der Standard ist False.
    :return: eine ``pyvqnet.qnn.vqc.torch.QModule``-Instanz

    Example::
        
        from pyvqnet.qnn.vqc.torch import SDG,QMachine
        import pyvqnet
        pyvqnet.backends.set_backend("torch")
        device = QMachine(4)
        layer = SDG(wires=0)
        batchsize = 1
        device.reset_states(1)
        layer(q_machine = device)
        print(device.states)




TDG
""""""""""""""""""

.. py:class:: pyvqnet.qnn.vqc.torch.TDG(has_params: bool = False,trainable: bool = False,init_params=None,wires=None,dtype=pyvqnet.kcomplex64,use_dagger=False)
    
    Definiert ein SDG-Quantengatter.

    Diese Klasse erbt von ``pyvqnet.qnn.vqc.torch.QModule`` und ``torch.nn.Module``.
    Diese Klasse kann als Untermodul von ``torch.nn.Module`` zum torch-Modell hinzugefügt werden.

    :param has_params: ob es Parameter hat, wie RX, RY und andere Gatter auf True gesetzt werden müssen, und solche ohne Parameter auf False gesetzt werden müssen, der Standard ist False.
    :param trainable: ob es zu trainierende Parameter hat. Wenn die Schicht externe Eingabedaten zum Aufbau der Logikgatter-Matrix verwendet, auf False setzen. Wenn die zu trainierenden Parameter von dieser Schicht initialisiert werden müssen, ist es True, der Standard ist False.
    :param init_params: Initialisierungsparameter zum Codieren klassischer Daten QTensor, der Standard ist None.
    :param wires: Bit-Index des Leitungseffekts, der Standard ist None.
    :param dtype: Die Datenpräzision der internen Matrix des Logikgatters kann auf pyvqnet.kcomplex64 oder pyvqnet.kcomplex128 gesetzt werden, entsprechend float- bzw. double-Eingabe.
    :param use_dagger: ob die transponiert-konjugierte Version des Gatters verwendet werden soll, der Standard ist False.
    :return: eine ``pyvqnet.qnn.vqc.torch.QModule``-Instanz

    Example::
        
        from pyvqnet.qnn.vqc.torch import TDG,QMachine
        import pyvqnet
        pyvqnet.backends.set_backend("torch")
        device = QMachine(4)
        layer = TDG(wires=0)
        batchsize = 1
        device.reset_states(1)
        layer(q_machine = device)
        print(device.states)



ControlledPhaseShift
"""""""""""""""""""""""""""""""""""""""""""""


.. py:class:: pyvqnet.qnn.vqc.torch.ControlledPhaseShift(has_params: bool = False,trainable: bool = False,init_params=None,wires=None,dtype=pyvqnet.kcomplex64,use_dagger=False)
    
    Definiert ein ControlledPhaseShift-Quantengatter.

    Diese Klasse erbt von ``pyvqnet.qnn.vqc.torch.QModule`` und ``torch.nn.Module``.
    Diese Klasse kann als Untermodul von ``torch.nn.Module`` zum torch-Modell hinzugefügt werden.

    :param has_params: ob es Parameter hat, wie RX, RY und andere Gatter auf True gesetzt werden müssen, und solche ohne Parameter auf False gesetzt werden müssen, der Standard ist False.
    :param trainable: ob es zu trainierende Parameter hat. Wenn die Schicht externe Eingabedaten zum Aufbau der Logikgatter-Matrix verwendet, auf False setzen. Wenn die zu trainierenden Parameter von dieser Schicht initialisiert werden müssen, ist es True, der Standard ist False.
    :param init_params: Initialisierungsparameter zum Codieren klassischer Daten QTensor, der Standard ist None.
    :param wires: Bit-Index des Leitungseffekts, der Standard ist None.
    :param dtype: Die Datenpräzision der internen Matrix des Logikgatters kann auf pyvqnet.kcomplex64 oder pyvqnet.kcomplex128 gesetzt werden, entsprechend float- bzw. double-Eingabe.
    :param use_dagger: ob die transponiert-konjugierte Version des Gatters verwendet werden soll, der Standard ist False.
    :return: eine ``pyvqnet.qnn.vqc.torch.QModule``-Instanz

    Example::

        from pyvqnet.qnn.vqc.torch import ControlledPhaseShift,QMachine
        import pyvqnet
        pyvqnet.backends.set_backend("torch")
        device = QMachine(4)
        layer = ControlledPhaseShift(has_params= True, trainable= True, wires=[0,2])
        batchsize = 2
        device.reset_states(batchsize)
        layer(q_machine = device)
        print(device.states)



MultiControlledX
"""""""""""""""""""""""""""""""""""""""""""""

.. py:class:: pyvqnet.qnn.vqc.torch.MultiControlledX(has_params: bool = False,trainable: bool = False,init_params=None,wires=None,dtype=pyvqnet.kcomplex64,use_dagger=False,control_values=None)
    
    Definiert ein MultiControlledX-Quantengatter.

    Diese Klasse erbt von ``pyvqnet.qnn.vqc.torch.QModule`` und ``torch.nn.Module``.
    Diese Klasse kann als Untermodul von ``torch.nn.Module`` zum torch-Modell hinzugefügt werden.
    
    :param has_params: ob es Parameter hat, wie RX, RY und andere Gatter auf True gesetzt werden müssen, und solche ohne Parameter auf False gesetzt werden müssen, der Standard ist False.
    :param trainable: ob es zu trainierende Parameter hat. Wenn die Schicht externe Eingabedaten zum Aufbau der Logikgatter-Matrix verwendet, auf False setzen. Wenn die zu trainierenden Parameter von dieser Schicht initialisiert werden müssen, ist es True, der Standard ist False.
    :param init_params: Initialisierungsparameter zum Codieren klassischer Daten QTensor, der Standard ist None.
    :param wires: Bit-Index des Leitungseffekts, der Standard ist None.
    :param dtype: Die Datenpräzision der internen Matrix des Logikgatters kann auf pyvqnet.kcomplex64 oder pyvqnet.kcomplex128 gesetzt werden, entsprechend float- bzw. double-Eingabe.
    :param use_dagger: ob die transponiert-konjugierte Version des Gatters verwendet werden soll, der Standard ist False.
    :param control_values: Control value, the default is None, when the bit is 1, it is controlled.

    :return: eine ``pyvqnet.qnn.vqc.torch.QModule``-Instanz

    Example::

        import pyvqnet
        pyvqnet.backends.set_backend("torch")
        from pyvqnet.qnn.vqc.torch import QMachine,MultiControlledX
        from pyvqnet.tensor import QTensor,kcomplex64
        qm = QMachine(4,dtype=kcomplex64)
        qm.reset_states(2)
        mcx = MultiControlledX( 
                        init_params=None,
                        wires=[2,3,0,1],
                        dtype=kcomplex64,
                        use_dagger=False,control_values=[1,0,0])
        y = mcx(q_machine = qm)
        print(qm.states)


Messungs-API
^^^^^^^^^^^^^^^^^^^^^^

Probability
"""""""""""""""""""""""""""""


.. py:class:: pyvqnet.qnn.vqc.torch.Probability(wires=None, name="")

    Berechnet das Wahrscheinlichkeitsmessungsergebnis des Quantenschaltkreises auf einem bestimmten Bit.

    Diese Klasse erbt von ``pyvqnet.qnn.vqc.torch.QModule`` und ``torch.nn.Module``.
    Diese Klasse kann als Untermodul von ``torch.nn.Module`` zum torch-Modell hinzugefügt werden.

    :param wires: Der Index des Messbits, Liste, Tupel oder Integer.
    :param name: Der Name des Moduls, Standard: "".
    :return: Das Messergebnis, QTensor.

    Example::

        import pyvqnet
        pyvqnet.backends.set_backend("torch")
        from pyvqnet.qnn.vqc.torch import Probability,rx,ry,cnot,QMachine,rz
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


MeasureAll
"""""""""""""""""""""""""""""

.. py:class:: pyvqnet.qnn.vqc.torch.MeasureAll(obs=None, name="")

    Berechnet die Messergebnisse von Quantenschaltkreisen, unterstützt Eingabe obs als mehrere oder einzelne Pauli-Operatoren oder Hamiltonians.
    For example:

    {\'X0\': 0.23} indicates a PauliX effect on qubit 0, with a coefficient of 0.23.

    {\'X1 Z2\': 2.4,\'Y2\': -0.5} corresponds to the observed value 2.4 * X1 @ Z2 - 0.5 * Y2.

    [{\'X1 Z2\': 4,\'Z1 Z0\': 3},{\'X1 Y2 Z0\': 3.5}] corresponds to the two observed values 4 * X1 @ Z2 + 3 * Z1 @ Z0 and 3.5 * X1 @ Y2 @ Z0.
        
        
    Diese Klasse erbt von ``pyvqnet.qnn.vqc.torch.QModule`` und ``torch.nn.Module``.

    Diese Klasse kann als Untermodul von ``torch.nn.Module`` zum torch-Modell hinzugefügt werden.

    :param obs: observable.
    :param name: module name, default: "".
    :return: measurement result, QTensor.

    Example::

        import pyvqnet
        pyvqnet.backends.set_backend("torch")
        from pyvqnet.qnn.vqc.torch import MeasureAll,rx,ry,cnot,QMachine,rz
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
            "Z0 Z1" :2
        }, {
            "X0 Z2" :1
        }]
        ma = MeasureAll(obs = obs_list)
        y = ma(q_machine=qm)
        print(y)



Samples
"""""""""""""""""""""""""""""

.. py:class:: pyvqnet.qnn.vqc.torch.Samples(wires=None, obs=None, shots = 1,name="")

    Erhält Stichprobenergebnisse mit Schuss auf bestimmten Drähten.

    Diese Klasse erbt von ``pyvqnet.qnn.vqc.torch.QModule`` und ``torch.nn.Module``.
    Diese Klasse kann als Untermodul von ``torch.nn.Module`` zum torch-Modell hinzugefügt werden.


    :param wires: Sample qubit index. Default value: None, use all bits of the simulator at runtime.
    :param obs: This value can only be None.
    :param shots: Sample repetition count, default value: 1.
    :param name: The name of this module, default value: "".
    :return: a measurement method class

    Example::

        import pyvqnet
        pyvqnet.backends.set_backend("torch")
        from pyvqnet.qnn.vqc.torch import Samples,rx,ry,cnot,QMachine,rz
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


HermitianExpval
"""""""""""""""""""""""""""""

.. py:class:: pyvqnet.qnn.vqc.torch.HermitianExpval(obs=None, name="")

    Berechnet den Erwartungswert einer hermiteschen Größe in einem Quantenschaltkreis.

    Diese Klasse erbt von ``pyvqnet.qnn.vqc.torch.QModule`` und ``torch.nn.Module``.
    Diese Klasse kann als Untermodul von ``torch.nn.Module`` zum torch-Modell hinzugefügt werden.


    :param obs: Hermitian quantity.
    :param name: module name, default: "".
    :return: expected result, QTensor.

    Example::

        import pyvqnet
        pyvqnet.backends.set_backend("torch")
        from pyvqnet.qnn.vqc.torch import QMachine, rx,ry,\
            RX, RY, CNOT, PauliX, PauliZ, VQC_RotCircuit,HermitianExpval
        from pyvqnet.tensor import QTensor, tensor
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

                rx(q_machine=self.qm, wires=0, params=x[:, [1]])
                ry(q_machine=self.qm, wires=1, params=x[:, [0]])
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

Häufige Vorlagen für Quantenschaltkreise
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

VQC_HardwareEfficientAnsatz
""""""""""""""""""""""""""""""""""""""""


.. py:class:: pyvqnet.qnn.vqc.torch.VQC_HardwareEfficientAnsatz(n_qubits,single_rot_gate_list,entangle_gate="CNOT",entangle_rules='linear',depth=1,initial=None,dtype=None)

    Implementierung des Hardware Efficient Ansatz, vorgestellt im Paper: `Hardware-efficient Variational Quantum Eigensolver for Small Molecules <https://arxiv.org/pdf/1704.05018.pdf>`__ .

    Diese Klasse erbt von ``pyvqnet.qnn.vqc.torch.QModule`` und ``torch.nn.Module``.

    Diese Klasse kann als Untermodul von ``torch.nn.Module`` zum torch-Modell hinzugefügt werden.


    :param n_qubits: Anzahl der Qubits.
    :param single_rot_gate_list: A single qubit rotation gate list is constructed by one or several rotation gate that act on every qubit.Currently support Rx, Ry, Rz.
    :param entangle_gate: The non parameterized entanglement gate.CNOT,CZ is supported.default:CNOT.
    :param entangle_rules: How entanglement gate is used in the circuit. 'linear' means the entanglement gate will be act on every neighboring qubits. 'all' means the entanglment gate will be act on any two qbuits. Default:linear.
    :param depth: The depth of ansatz, default:1.
    :param initial: initial one same value for paramaters,default:None,this module will initialize parameters randomly.
    :param dtype: data dtype of parameters.
    :return: a VQC_HardwareEfficientAnsatz instance.

    Example::

        from pyvqnet.nn.torch import TorchModule,Linear,TorchModuleList
        from pyvqnet.qnn.vqc.torch.qcircuit import VQC_HardwareEfficientAnsatz,RZZ,RZ
        from pyvqnet.qnn.vqc.torch import Probability,QMachine
        from pyvqnet import tensor
        import pyvqnet
        pyvqnet.backends.set_backend("torch")
        pyvqnet.utils.set_random_seed(25)

        class QM(TorchModule):
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



VQC_BasicEntanglerTemplate
""""""""""""""""""""""""""""""""""""""""

.. py:class:: pyvqnet.qnn.vqc.torch.VQC_BasicEntanglerTemplate(num_layer=1, num_qubits=1, rotation="RX", initial=None, dtype=None)

    Eine Schicht bestehend aus einer Ein-Parameter-Einzel-Qubit-Rotation auf jedem Qubit, gefolgt von mehreren CNOT-Gattern in einer geschlossenen Kette oder Ring-Kombination.

    Ein Ring von CNOT-Gattern verbindet jedes Qubit mit seinen Nachbarn, und schließlich wird das a-te Qubit als Nachbar des a-ten Qubits betrachtet.

    Diese Klasse erbt von ``pyvqnet.qnn.vqc.torch.QModule`` und ``torch.nn.Module``.

    Diese Klasse kann als Untermodul von ``torch.nn.Module`` zum torch-Modell hinzugefügt werden.

    :param num_layers: number of repeat layers, default: 1.
    :param num_qubits: Anzahl der Qubits, Standard: 1.
    :param rotation: zu verwendendes Ein-Parameter-Ein-Qubit-Gatter, Standard: `RX`
    :param initial: initialisiert gleichen Wert für alle Parameter. Standard: None, Parameter werden zufällig initialisiert.
    :param dtype: Datentyp des Parameters, Standard: None, verwende float32.
    :return: Eine VQC_BasicEntanglerTemplate-Instanz

    Example::

        import pyvqnet
        pyvqnet.backends.set_backend("torch")
        from pyvqnet.qnn.vqc.torch import QModule,\
            VQC_BasicEntanglerTemplate, Probability, QMachine
        from pyvqnet import tensor


        class QM(QModule):
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



VQC_StronglyEntanglingTemplate
""""""""""""""""""""""""""""""""""""""""

.. py:class:: pyvqnet.qnn.vqc.torch.VQC_StronglyEntanglingTemplate(num_layers=1, num_qubits=1, ranges=None,initial=None, dtype=None)

    Schichten bestehend aus Einzel-Qubit-Rotationen und Entanglern, wie im `circuit-centric classifier design <https://arxiv.org/abs/1804.00633>`__ .

    Diese Klasse erbt von ``pyvqnet.qnn.vqc.torch.QModule`` und ``torch.nn.Module``.
    Diese Klasse kann als Untermodul von ``torch.nn.Module`` zum torch-Modell hinzugefügt werden.


    :param num_layers: number of repeat layers, default: 1.
    :param num_qubits: Anzahl der Qubits, Standard: 1.
    :param ranges: sequence determining the range hyperparameter for each subsequent layer; default: None
                                using :math: `r=l \mod M` for the :math:`l` th layer and :math:`M` qubits.
    :param initial: initial value for all parameters.default: None,initialized randomly.
    :param dtype: data type of parameter, default:None,use float32.
    :return: A VQC_StronglyEntanglingTemplate instance.

    Example::

        from pyvqnet.nn.torch import TorchModule,Linear,TorchModuleList
        from pyvqnet.qnn.vqc.torch.qcircuit import VQC_StronglyEntanglingTemplate
        from pyvqnet.qnn.vqc.torch import Probability, QMachine
        from pyvqnet import tensor
        import pyvqnet

        pyvqnet.backends.set_backend("torch")
        pyvqnet.utils.set_random_seed(25)
        class QM(TorchModule):
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



VQC_QuantumEmbedding
""""""""""""""""""""""""""""""""""""""""


.. py:class:: pyvqnet.qnn.vqc.torch.VQC_QuantumEmbedding(qubits, machine, num_repetitions_input, depth_input, num_unitary_layers, num_repetitions,initial = None,dtype = None,name= "")
    
    Verwendet RZ, RY, RZ zur Erstellung variationeller Quantenschaltkreise zur Codierung klassischer Daten in Quantenzustände.
    Referenz `Quantum embeddings for machine learning <https://arxiv.org/abs/2001.03622>`_.

    Diese Klasse erbt von ``pyvqnet.qnn.vqc.torch.QModule`` und ``torch.nn.Module``.
    Diese Klasse kann als Untermodul von ``torch.nn.Module`` zum torch-Modell hinzugefügt werden.

 
    :param num_repetitions_input: Anzahl der Wiederholungen zur Codierung der Eingabe in einem Untermodul.
    :param depth_input: Anzahl der Eingabedimensionen.
    :param num_unitary_layers: Anzahl der Wiederholungen variationeller Quantengatter.
    :param num_repetitions: Anzahl der Wiederholungen des Untermoduls.
    :param initial: Parameter-Initialisierungswert, Standard ist None
    :param dtype: Parametertyp, Standard ist None, verwende float32.
    :param name: Klassenname
    :return: Eine VQC_QuantumEmbedding-Instanz.

    Example::

        from pyvqnet.nn.torch import TorchModule
        from pyvqnet.qnn.vqc.torch.qcircuit import VQC_QuantumEmbedding
        from pyvqnet.qnn.vqc.torch import Probability, QMachine, MeasureAll
        from pyvqnet import tensor
        import pyvqnet

        pyvqnet.backends.set_backend("torch")
        pyvqnet.utils.set_random_seed(25)
        depth_input = 2
        num_repetitions = 2
        num_repetitions_input = 2
        num_unitary_layers = 2
        nq = depth_input * num_repetitions_input
        bz = 12

        class QM(TorchModule):
            def __init__(self, name=""):
                super().__init__(name)

                self.ansatz = VQC_QuantumEmbedding(num_repetitions_input, depth_input,
                                                num_unitary_layers,
                                                num_repetitions, initial=tensor.full([1],12.0),dtype=pyvqnet.kfloat32)

                self.measure = MeasureAll(obs={f"Z{nq-1}":1})
                self.device = QMachine(nq)

            def forward(self, x, *args, **kwargs):
                self.device.reset_states(x.shape[0])
                self.ansatz(x,q_machine=self.device)
                return self.measure(q_machine=self.device)

        inputx = tensor.arange(1.0, bz * depth_input + 1).reshape([bz, depth_input])
        qlayer = QM()
        y = qlayer(inputx)
        y.backward()
        print(y)


ExpressiveEntanglingAnsatz
""""""""""""""""""""""""""""""""""""""""

.. py:class:: pyvqnet.qnn.vqc.torch.ExpressiveEntanglingAnsatz(type: int, num_wires: int, depth: int, dtype=None, name: str = "")

    19 verschiedene Ansätze aus dem Paper `Expressibility and entangling capability of parameterized quantum circuits for hybrid quantum-classical algorithms <https://arxiv.org/pdf/1905.10876.pdf>`_.

    Diese Klasse erbt von ``pyvqnet.qnn.vqc.torch.QModule`` und ``torch.nn.Module``.

    Diese Klasse kann als Untermodul von ``torch.nn.Module`` zum torch-Modell hinzugefügt werden.

    :param type: Circuit type from 1 to 19, a total of 19 lines.
    :param num_wires: Anzahl der Qubits.
    :param depth: Circuit depth.
    :param dtype: data type of parameter, default:None,use float32.
    :param name: Name, default "".

    :return:
        a ExpressiveEntanglingAnsatz instance

    Example::

        from pyvqnet.nn.torch import TorchModule
        from pyvqnet.qnn.vqc.torch.qcircuit import ExpressiveEntanglingAnsatz
        from pyvqnet.qnn.vqc.torch import Probability, QMachine, MeasureAll
        from pyvqnet import tensor
        import pyvqnet

        pyvqnet.backends.set_backend("torch")
        pyvqnet.utils.set_random_seed(25)

        class QModel(TorchModule):
            def __init__(self, num_wires, dtype,grad_mode=""):
                super(QModel, self).__init__()

                self._num_wires = num_wires
                self._dtype = dtype
                self.qm = QMachine(num_wires, dtype=dtype,grad_mode=grad_mode)
                self.c1 = ExpressiveEntanglingAnsatz(1,3,2)
                self.measure = MeasureAll(obs={
                    "Z1":1
                })

            def forward(self, x, *args, **kwargs):
                self.qm.reset_states(x.shape[0])
                self.c1(q_machine = self.qm)
                rlt = self.measure(q_machine=self.qm)
                return rlt
            

        input_x = tensor.QTensor([[0.1, 0.2, 0.3]])

        qunatum_model = QModel(num_wires=3, dtype=pyvqnet.kcomplex64)

        batch_y = qunatum_model(input_x)
        batch_y.backward()
        print(batch_y)



vqc_basis_embedding
""""""""""""""""""""""""""""""""""""""""

.. py:function:: pyvqnet.qnn.vqc.torch.vqc_basis_embedding(basis_state,q_machine)

    Codiert n binäre Merkmale in den n-Qubit-Basiszustand von ``q_machine``. Diese Funktion hat den Alias `VQC_BasisEmbedding`.

    For example, for ``basis_state=([0, 1, 1])``, the basis state in the quantum system is :math:`|011 \rangle`.

    :param basis_state: ``(n)`` size binary input.
    :param q_machine: quantum machine device.
    

    Example::

        import pyvqnet
        pyvqnet.backends.set_backend("torch")
        from pyvqnet.qnn.vqc.torch import vqc_basis_embedding,QMachine
        qm  = QMachine(3)
        vqc_basis_embedding(basis_state=[1,1,0],q_machine=qm)
        print(qm.states)




vqc_angle_embedding
""""""""""""""""""""""""""""""""""""""""


.. py:function:: pyvqnet.qnn.vqc.torch.vqc_angle_embedding(input_feat, wires, q_machine: pyvqnet.qnn.vqc.torch.QMachine, rotation: str = "X")

    Codiert :math:`N` Merkmale in den Rotationswinkel von :math:`n` Qubits, wobei :math:`N \leq n`.
    Diese Funktion hat den Alias `VQC_AngleEmbedding` .

    Die Rotation kann ausgewählt werden als: 'X', 'Y', 'Z', wie durch den ``rotation``-Parameter definiert:

    * ``rotation='X'`` Verwendet das Merkmal als Winkel der RX-Rotation.

    * ``rotation='Y'`` Verwendet das Merkmal als Winkel der RY-Rotation.

    * ``rotation='Z'`` Verwendet das Merkmal als Winkel der RZ-Rotation.

    ``wires`` repräsentiert den Index des Rotationsgatters auf dem Qubit.

    :param input_feat: Array representing parameters.
    :param wires: Qubit idx.
    :param q_machine: Quantum machine device.
    :param rotation: Rotation gate, default is "X".

    Example::

        import pyvqnet
        pyvqnet.backends.set_backend("torch")
        from pyvqnet.qnn.vqc.torch import vqc_angle_embedding, QMachine
        from pyvqnet.tensor import QTensor
        qm  = QMachine(2)
        vqc_angle_embedding(QTensor([2.2, 1]), [0, 1], q_machine=qm, rotation='X')
        print(qm.states)
        vqc_angle_embedding(QTensor([2.2, 1]), [0, 1], q_machine=qm, rotation='Y')
        print(qm.states)
        vqc_angle_embedding(QTensor([2.2, 1]), [0, 1], q_machine=qm, rotation='Z')
        print(qm.states)



vqc_amplitude_embedding
""""""""""""""""""""""""""""""""""""""""

.. py:function:: pyvqnet.qnn.vqc.torch.vqc_amplitude_embeddingVQC_AmplitudeEmbeddingCircuit(input_feature, q_machine)

    Codiert ein :math:`2^n`-Merkmal in einen Amplitudenvektor von :math:`n` Qubits. Diese Funktion hat den Alias `VQC_AmplitudeEmbedding`.

    :param input_feature: numpy array representing the parameter.
    :param q_machine: quantum machine device.
    

    Example::

        import pyvqnet
        pyvqnet.backends.set_backend("torch")
        from pyvqnet.qnn.vqc.torch import vqc_amplitude_embedding, QMachine
        from pyvqnet.tensor import QTensor
        qm  = QMachine(3)
        vqc_amplitude_embedding(QTensor([3.2,-2,-2,0.3,12,0.1,2,-1]), q_machine=qm)
        print(qm.states)



vqc_iqp_embedding
""""""""""""""""""""""""""""""""""""""""

.. py:function:: pyvqnet.qnn.vqc.vqc_iqp_embedding(input_feat, q_machine: pyvqnet.qnn.vqc.torch.QMachine, rep: int = 1)

    Codiert :math:`n` Merkmale in :math:`n` Qubits unter Verwendung diagonaler Gatter einer IQP-Schaltung. Alias: ``VQC_IQPEmbedding`` .

    The encoding is proposed by `Havlicek et al. (2018) <https://arxiv.org/pdf/1804.11326.pdf>`_.

    By specifying ``rep`` , the basic IQP circuit can be repeated.

    :param input_feat: Array of parameters.
    :param q_machine: Quantum machine machine.
    :param rep: Number of times to repeat the quantum circuit block, default is 1.

    Example::

        import pyvqnet
        pyvqnet.backends.set_backend("torch")
        from pyvqnet.qnn.vqc.torch import vqc_iqp_embedding, QMachine
        from pyvqnet.tensor import QTensor
        qm  = QMachine(3)
        vqc_iqp_embedding(QTensor([3.2,-2,-2]), q_machine=qm)
        print(qm.states)        



vqc_rotcircuit
""""""""""""""""""""""""""""""""""""""""

.. py:function:: pyvqnet.qnn.vqc.torch.vqc_rotcircuit(q_machine, wire, params)

    Beliebige Einzel-Qubit-Rotations-Quantenlogikgatter-Kombination. Diese Funktion hat den Alias: ``VQC_RotCircuit`` .

    .. math::
        R(\phi,\theta,\omega) = RZ(\omega)RY(\theta)RZ(\phi)= \begin{bmatrix}
        e^{-i(\phi+\omega)/2}\cos(\theta/2) & -e^{i(\phi-\omega)/2}\sin(\theta/2) \\
        e^{-i(\phi-\omega)/2}\sin(\theta/2) & e^{i(\phi+\omega)/2}\cos(\theta/2)
        \end{bmatrix}.

    :param q_machine: quantum virtual machine device.
    :param wire: quantum bit index.
    :param params: represents parameters :math:`[\phi, \theta, \omega]`.

    Example::

        import pyvqnet
        pyvqnet.backends.set_backend("torch")
        from pyvqnet.qnn.vqc.torch import vqc_rotcircuit, QMachine
        from pyvqnet.tensor import QTensor
        qm  = QMachine(3)
        vqc_rotcircuit(q_machine=qm, wire=[1],params=QTensor([2.0,1.5,2.1]))
        print(qm.states)


vqc_crot_circuit
""""""""""""""""""""""""""""""""""""""""


.. py:function:: pyvqnet.qnn.vqc.torch.vqc_crot_circuit(para,control_qubits,rot_wire,q_machine)

    Quantenlogikgatter-Kombination einer gesteuerten Rot-Einzel-Qubit-Rotation. Diese Funktion hat den Alias: ``VQC_CRotCircuit`` .

    .. math:: 
        CR(\phi, \theta, \omega) = \begin{bmatrix}
        1 & 0 & 0 & 0 \\
        0 & 1 & 0 & 0\\
        0 & 0 & e^{-i(\phi+\omega)/2}\cos(\theta/2) & -e^{i(\phi-\omega)/2}\sin(\theta/2)\\
        0 & 0 & e^{-i(\phi-\omega)/2}\sin(\theta/2) & e^{i(\phi+\omega)/2}\cos(\theta/2)
        \end{bmatrix}.

    :param para: represents the array of parameters.
    :param control_qubits: Control qubit index.
    :param rot_wire: Rot qubit index.
    :param q_machine: Quantum machine device.
    

    Example::

        import pyvqnet
        pyvqnet.backends.set_backend("torch")
        from pyvqnet.tensor import QTensor
        from pyvqnet.qnn.vqc.torch import vqc_crot_circuit,QMachine, MeasureAll
        p = QTensor([2, 3, 4.0])
        qm = QMachine(2)
        vqc_crot_circuit(p, 0, 1, qm)
        m = MeasureAll(obs={"Z0": 1})
        exp = m(q_machine=qm)
        print(exp)




vqc_controlled_hadamard
""""""""""""""""""""""""""""""""""""""""


.. py:function:: pyvqnet.qnn.vqc.torch.vqc_controlled_hadamard(wires, q_machine)

    Gesteuerter Hadamard-Logikgatter-Quantenschaltkreis. Diese Funktion hat den Alias: ``VQC_Controlled_Hadamard`` .

    .. math:: 
        CH = \begin{bmatrix}
        1 & 0 & 0 & 0 \\
        0 & 1 & 0 & 0 \\
        0 & 0 & \frac{1}{\sqrt{2}} & \frac{1}{\sqrt{2}} \\
        0 & 0 & \frac{1}{\sqrt{2}} & -\frac{1}{\sqrt{2}}
        \end{bmatrix}.

    :param wires: quantum bit index list, the first one is the control bit, the list length is 2.
    :param q_machine: quantum virtual machine device.

    Examples::

        import pyvqnet
        pyvqnet.backends.set_backend("torch")
        from pyvqnet.tensor import QTensor
        from pyvqnet.qnn.vqc.torch import vqc_controlled_hadamard,\
            QMachine, MeasureAll

        p = QTensor([0.2, 3, 4.0])
        qm = QMachine(3)
        vqc_controlled_hadamard([1, 0], qm)
        m = MeasureAll(obs={"Z0": 1})
        exp = m(q_machine=qm)
        print(exp)



vqc_ccz
""""""""""""""""""""""""""""""""""""""""

.. py:function:: pyvqnet.qnn.vqc.torch.vqc_ccz(wires, q_machine)

    Gesteuert-gesteuertes-Z-Logikgatter. Alias: ``VQC_CCZ`` .

    .. math::
        CCZ =
        \begin{pmatrix}
        1 & 0 & 0 & 0 & 0 & 0 & 0 & 0\\
        0 & 1 & 0 & 0 & 0 & 0 & 0 & 0 & 0\\
        0 & 0 & 1 & 0 & 0 & 0 & 0 & 0\\
        0 & 0 & 0 & 1 & 0 & 0 & 0 & 0\\
        0 & 0 & 0 & 0 & 1 & 0 & 0 & 0\\
        0 & 0 & 0 & 0 & 0 & 1 & 0 & 0 & 0\\
        0 & 0 & 0 & 0 & 0 & 1 & 0 & 0\\
        0 & 0 & 0 & 0 & 0 & 0 & 1 & 0 & 0\\
        0 & 0 & 0 & 0 & 0 & 0 & 0 & 1 & 0\\
        0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & -1
        \end{pmatrix}

    :param wires: quantum bit index list, the first one is the control bit. The list length is 3.
    :param q_machine: quantum virtual machine device.

    Example::

        import pyvqnet
        pyvqnet.backends.set_backend("torch")
        from pyvqnet.tensor import QTensor
        from pyvqnet.qnn.vqc.torch import vqc_ccz,QMachine, MeasureAll
        p = QTensor([0.2, 3, 4.0])

        qm = QMachine(3)

        vqc_ccz([1, 0, 2], qm)
        m = MeasureAll(obs={"Z0": 1})
        exp = m(q_machine=qm)
        print(exp)



vqc_fermionic_single_excitation
""""""""""""""""""""""""""""""""""""""""

.. py:function:: pyvqnet.qnn.vqc.torch.vqc_fermionic_single_excitation(weight, wires, q_machine)

    Gekoppelter-Cluster-Einfachanregungsoperator für das Tensorprodukt von Pauli-Matrizen. Matrix form is given by:

    .. math::
        \hat{U}_{pr}(\theta) = \mathrm{exp} \{ \theta_{pr} (\hat{c}_p^\dagger \hat{c}_r
        -\mathrm{H.c.}) \},

    Alias: ``VQC_FermionicSingleExcitation`` .

    :param weight: Parameter on qubit p, only a elements.
    :param wires: A subset of qubit indices in the interval [r, p]. Minimum length must be 2. The first index value is interpreted as r, and the last a index value is interpreted as p.The intermediate indices are acted upon by CNOT gates to compute the parity of the qubit set.
    :param q_machine: Quantum virtual machine device.

    

    Examples::

        import pyvqnet
        pyvqnet.backends.set_backend("torch")
        from pyvqnet.tensor import QTensor
        from pyvqnet.qnn.vqc.torch import vqc_fermionic_single_excitation,\
            QMachine, MeasureAll
        qm = QMachine(3)
        p0 = QTensor([0.5])

        vqc_fermionic_single_excitation(p0, [1, 0, 2], qm)
        m = MeasureAll(obs={"Z0": 1})
        exp = m(q_machine=qm)
        print(exp)

 


vqc_fermionic_double_excitation
""""""""""""""""""""""""""""""""""""""""


.. py:function:: pyvqnet.qnn.vqc.torch.vqc_fermionic_double_excitation(weight, wires1, wires2, q_machine)

    Gekoppelter-Cluster-Doppelanregungsoperator für das exponenzierte Tensorprodukt von Pauli-Matrizen, Matrixform gegeben durch:

    .. math::
        \hat{U}_{pqrs}(\theta) = \mathrm{exp} \{ \theta (\hat{c}_p^\dagger \hat{c}_q^\dagger
        \hat{c}_r \hat{c}_s - \mathrm{H.c.}) \},

    wobei :math:`\hat{c}` und :math:`\hat{c}^\dagger` Fermionen-Vernichtungs- und
    Erzeugungsoperatoren sind und :math:`r, s` und :math:`p, q` auf besetzten bzw.
    leeren Molekülorbitalen indizieren. Verwenden Sie die `Jordan-Wigner-Transformation
    <https://arxiv.org/abs/1208.5986>`_ The fermion operator defined above can be written as
    in terms of the Pauli matrix (see
    `arXiv:1805.04340 <https://arxiv.org/abs/1805.04340>`_ for more details)

    .. math::
        \hat{U}_{pqrs}(\theta) = \mathrm{exp} \Big\{
        \frac{i\theta}{8} \bigotimes_{b=s+1}^{r-1} \hat{Z}_b \bigotimes_{a=q+1}^{p-1}
        \hat{Z}_a (\hat{X}_s \hat{X}_r \hat{Y}_q \hat{X}_p +
        \hat{Y}_s \hat{X}_r \hat{Y}_q \hat{Y}_p +\\ \hat{X}_s \hat{Y}_r \hat{Y}_q \hat{Y}_p +
        \hat{X}_s \hat{X}_r \hat{X}_q \hat{Y}_p - \mathrm{H.c.} ) \Big\}

    This function is aliased as: ``VQC_FermionicDoubleExcitation`` .

    :param weight: variable parameter
    :param wires1: represents the subset of qubits in the index list interval [s, r]. The ath index is interpreted as s and the last index is interpreted as r. The CNOT gate operates on the middle indexes to calculate the parity of a group of qubits.
    :param wires2: represents the subset of qubits in the index list interval [q, p]. The first root index is interpreted as q and the last index is interpreted as p. The CNOT gate operates on the middle indexes to calculate the parity of a group of qubits.
    :param q_machine: Quantum virtual machine device.

    

    Examples::

        import pyvqnet
        pyvqnet.backends.set_backend("torch")
        from pyvqnet.tensor import QTensor
        from pyvqnet.qnn.vqc.torch import vqc_fermionic_double_excitation,\
            QMachine, MeasureAll
        qm = QMachine(5)
        p0 = QTensor([0.5])

        vqc_fermionic_double_excitation(p0, [0, 1], [2, 3], qm)
        m = MeasureAll(obs={"Z0": 1})
        exp = m(q_machine=qm)
        print(exp)
 

vqc_uccsd
""""""""""""""""""""""""""""""""""""""""


.. py:function:: pyvqnet.qnn.vqc.torch.vqc_uccsd(weights, wires, s_wires, d_wires, init_state, q_machine)

    Implementiert die Unitary Coupled Cluster Single and Double Excitations Simulation (UCCSD). UCCSD ist eine VQE-Simulation, die häufig für quantenchemische Simulationen verwendet wird.

    Within the first-order Trotter approximation, the UCCSD unitary function is given by:

    .. math::
        \hat{U}(\vec{\theta}) =
        \prod_{p > r} \mathrm{exp} \Big\{\theta_{pr}
        (\hat{c}_p^\dagger \hat{c}_r-\mathrm{H.c.}) \Big\}
        \prod_{p > q > r > s} \mathrm{exp} \Big\{\theta_{pqrs}
        (\hat{c}_p^\dagger \hat{c}_q^\dagger \hat{c}_r \hat{c}_s-\mathrm{H.c.}) \Big\}

    where :math:`\hat{c}` and :math:`\hat{c}^\dagger` are fermion annihilation and
    creation operators and index :math:`r, s` and :math:`p, q` on occupied and
    empty molecular orbitals respectively. (For more details see
    `arXiv:1805.04340 <https://arxiv.org/abs/1805.04340>`_):

    This function is aliased as: ``VQC_UCCSD`` .

    :param weights: tensor of size ``(len(s_wires)+ len(d_wires))`` containing the parameters :math:`\theta_{pr}` and :math:`\theta_{pqrs}` input Z rotations ``FermionicSingleExcitation`` and ``FermionicDoubleExcitation`` .
    :param wires: qubit indices for template action
    :param s_wires: sequence of lists containing qubit indices ``[r,...,p]`` generated by a single excitation :math:`\vert r, p \rangle = \hat{c}_p^\dagger \hat{c}_r \vert \mathrm{HF} \rangle`,where :math:`\vert \mathrm{HF} \rangle` denotes the Hartee-Fock reference state.
    :param d_wires: sequence of lists, each containing two lists specifying indices ``[s, ...,r]`` and ``[q,..., p]`` defining double excitation :math:`\vert s, r, q, p \rangle = \hat{c}_p^\dagger \hat{c}_q^\dagger \hat{c}_r\hat{c}_s \vert \mathrm{HF} \rangle` .
    :param init_state: occupation-number vector of length ``len(wires)`` representing the high-frequency state. ``init_state`` Initialization state of the qubit.
    :param q_machine: Quantum virtual machine device.

    Examples::

        import pyvqnet
        pyvqnet.backends.set_backend("torch")
        from pyvqnet.qnn.vqc.torch import vqc_uccsd, QMachine, MeasureAll
        from pyvqnet.tensor import QTensor
        p0 = QTensor([2, 0.5, -0.2, 0.3, -2, 1, 3, 0])
        s_wires = [[0, 1, 2], [0, 1, 2, 3, 4], [1, 2, 3], [1, 2, 3, 4, 5]]
        d_wires = [[[0, 1], [2, 3]], [[0, 1], [2, 3, 4, 5]], [[0, 1], [3, 4]],
                [[0, 1], [4, 5]]]
        qm = QMachine(6)

        vqc_uccsd(p0, range(6), s_wires, d_wires, QTensor([1.0, 1, 0, 0, 0, 0]), qm)
        m = MeasureAll(obs={"Z1": 1})
        exp = m(q_machine=qm)
        print(exp)

        # [[0.963802]]


vqc_zfeaturemap
""""""""""""""""""""""""""""""""""""""""

.. py:function:: pyvqnet.qnn.vqc.torch.vqc_zfeaturemap(input_feat, q_machine: pyvqnet.qnn.vqc.torch.QMachine, data_map_func=None, rep: int = 2)

    Pauli-Z-Evolutionsschaltung erster Ordnung.

    For 3 qubits and 2 repetitions, the circuit is represented as:

    .. parsed-literal::

        ┌───┐┌──────────────┐┌───┐┌──────────────┐
        ┤ H ├┤ U1(2.0*x[0]) ├┤ H ├┤ U1(2.0*x[0]) ├
        ├───┤├──────────────┤├───┤├──────────────┤
        ┤ H ├┤ U1(2.0*x[1]) ├┤ H ├┤ U1(2.0*x[1]) ├
        ├───┤├──────────────┤├───┤├──────────────┤
        ┤ H ├┤ U1(2.0*x[2]) ├┤ H ├┤ U1(2.0*x[2]) ├
        └───┘└──────────────┘└───┘└──────────────┘

    Die Pauli-Zeichenkette ist auf ``Z`` festgelegt. Daher wird die Expansion erster Ordnung eine Schaltung ohne Verschlingungsgatter sein.

    :param input_feat: Array, das Eingabeparameter repräsentiert.
    :param q_machine: Quanten-Virtuelle-Maschine.
    :param data_map_func: Parameter-Abbildungsmatrix, eine aufrufbare Funktion, entworfen als: ``data_map_func = lambda x: x``.
    :param rep: Anzahl der Wiederholungen des Moduls.

    Example::

        import pyvqnet
        pyvqnet.backends.set_backend("torch")
        from pyvqnet.qnn.vqc.torch import vqc_zfeaturemap, QMachine, hadamard
        from pyvqnet.tensor import QTensor
        qm = QMachine(3)
        for i in range(3):
            hadamard(q_machine=qm, wires=[i])
        vqc_zfeaturemap(input_feat=QTensor([[0.1,0.2,0.3]]),q_machine = qm)
        print(qm.states)
 

vqc_zzfeaturemap
""""""""""""""""""""""""""""""""""""""""

.. py:function:: pyvqnet.qnn.vqc.torch.vqc_zzfeaturemap(input_feat, q_machine: pyvqnet.qnn.vqc.torch.QMachine, data_map_func=None, entanglement: Union[str, List[List[int]],Callable[[int], List[int]]] = "full",rep: int = 2)

    Pauli-Z-Evolutionsschaltung zweiter Ordnung.

    For 3 qubits, 1 repeat, and linear entanglement, the circuit is represented as:

    .. parsed-literal::


        ┌───┐┌─────────────────┐
        ┤ H ├┤ U1(2.0*φ(x[0])) ├──■────────────────────────────■────────────────────────────────────
        ├───┤├─────────────────┤┌─┴─┐┌──────────────────────┐┌─┴─┐
        ┤ H ├┤ U1(2.0*φ(x[1])) ├┤ X ├┤ U1(2.0*φ(x[0],x[1])) ├┤ X ├──■────────────────────────────■──
        ├───┤├─────────────────┤└───┘└──────────────────────┘└───┘┌─┴─┐┌──────────────────────┐┌─┴─┐
        ┤ H ├┤ U1(2.0*φ(x[2])) ├──────────────────────────────────┤ X ├┤ U1(2.0*φ(x[1],x[2])) ├┤ X ├
        └───┘└─────────────────┘                                  └───┘└──────────────────────┘└───┘
    
    Where ``φ`` is a classic nonlinear function. If two values ​​are input, ``φ(x,y) = (pi - x)(pi - y)``, and if a is input, ``φ(x) = x``. It is expressed as follows using ``data_map_func``:

    .. code-block::

        def data_map_func(x):
            coeff = x if x.shape[-1] == 1 else ft.reduce(lambda x, y: (np.pi - x) * (np.pi - y), x)
            return coeff

    :param input_feat: Array, das Eingabeparameter repräsentiert.
    :param q_machine: Quanten-Virtuelle-Maschine.
    :param data_map_func: Parameter-Abbildungsmatrix, eine aufrufbare Funktion.
    :param entanglement: angegebene Verschlingungsstruktur.
    :param rep: Anzahl der Modulwiederholungen.
    
    Example::

        import pyvqnet
        pyvqnet.backends.set_backend("torch")
        from pyvqnet.qnn.vqc.torch import vqc_zzfeaturemap, QMachine
        from pyvqnet.tensor import QTensor

        qm = QMachine(3)
        vqc_zzfeaturemap(q_machine=qm, input_feat=QTensor([[0.1,0.2,0.3]]))
        print(qm.states)


vqc_allsinglesdoubles
""""""""""""""""""""""""""""""""""""""""

.. py:function:: pyvqnet.qnn.vqc.torch.vqc_allsinglesdoubles(weights, q_machine: pyvqnet.qnn.vqc.torch.QMachine, hf_state, wires, singles=None, doubles=None)

    In diesem Fall haben wir vier Einfach- und Doppelanregungen, um die gesamte Spin-Projektion des Hartree-Fock-Zustands zu erhalten.

    Die resultierende unitäre Matrix bewahrt die Teilchenpopulation und bereitet das n-Qubit-System in einer Überlagerung des anfänglichen Hartree-Fock-Zustands und anderer Zustände vor, die die Mehrfachanregungskonfiguration codieren.

    :param weights: Ein QTensor der Größe ``(len(singles) + len(doubles),)``, der die Winkel enthält, die nacheinander in die vqc.qCircuit.single_excitation- und vqc.qCircuit.double_excitation-Operationen eingehen
    :param q_machine: Die Quantenmaschine.
    :param hf_state: Ein Vektor der Länge ``len(wires)`` mit Besetzungszahlen, die den Hartree-Fock-Zustand repräsentieren, ``hf_state`` wird zur Initialisierung der Drähte verwendet.
    :param wires: Die Qubits, auf die gewirkt werden soll.
    :param singles: Eine Sequenz von Listen mit den Indizes der beiden Qubits, auf die die single_excitation-Operation wirkt.
    :param doubles: Listensequenz mit den Indizes der beiden Qubits, auf die die double_excitation-Operation wirkt.

    For example, the quantum circuit for two electrons and six qubits is shown below:

    .. image:: ./images/all_singles_doubles.png
        :width: 600 px
        :align: center

    |

    Example::

        import pyvqnet
        pyvqnet.backends.set_backend("torch")
        from pyvqnet.qnn.vqc.torch import vqc_allsinglesdoubles, QMachine

        from pyvqnet.tensor import QTensor
        qubits = 4
        qm = QMachine(qubits)

        vqc_allsinglesdoubles(q_machine=qm, weights=QTensor([0.55, 0.11, 0.53]), 
                              hf_state = QTensor([1,1,0,0]), singles=[[0, 2], [1, 3]], doubles=[[0, 1, 2, 3]], wires=[0,1,2,3])
        print(qm.states)

vqc_basisrotation
""""""""""""""""""""""""""""""""""""""""

.. py:function:: pyvqnet.qnn.vqc.torch.vqc_basisrotation(q_machine: pyvqnet.qnn.vqc.torch.QMachine, wires, unitary_matrix: QTensor, check=False)

    Implementiert eine Schaltung, die ein Ensemble bereitstellt, das zur Durchführung genauer Einzel-Einheits-Basisrotationen verwendet werden kann. The circuit is derived from the single-particle fermion-determined unitary transformation :math:`U(u)` given in `arXiv:1711.04789 <https://arxiv.org/abs/1711.04789>`_\ 
    
    .. math::
        U(u) = \exp{\left( \sum_{pq} \left[\log u \right]_{pq} (a_p^\dagger a_q - a_q^\dagger a_p) \right)}.

    :math:`U(u)` is obtained by using the scheme given in the paper `Optica, 3, 1460 (2016) <https://opg.optica.org/optica/fulltext.cfm?uri=optica-3-12-1460&id=355743>`_\ .

    :param q_machine: Quantenmaschine.
    :param wires: Qubits, auf die gewirkt werden soll.
    :param unitary_matrix: Matrix, die die Basis für die Transformation angibt.
    :param check: überprüft, ob `unitary_matrix` eine unitäre Matrix ist.

    Example::

        import pyvqnet

        pyvqnet.backends.set_backend("torch")
        from pyvqnet.qnn.vqc.torch import vqc_basisrotation, QMachine
        from pyvqnet.tensor import QTensor
        import numpy as np

        V = np.array([[0.73678 + 0.27511j, -0.5095 + 0.10704j, -0.06847 + 0.32515j],
                    [0.73678 + 0.27511j, -0.5095 + 0.10704j, -0.06847 + 0.32515j],
                    [-0.21271 + 0.34938j, -0.38853 + 0.36497j, 0.61467 - 0.41317j]])

        eigen_vals, eigen_vecs = np.linalg.eigh(V)
        umat = eigen_vecs.T
        wires = range(len(umat))

        qm = QMachine(len(umat))

        vqc_basisrotation(q_machine=qm,
                        wires=wires,
                        unitary_matrix=QTensor(umat, dtype=qm.state.dtype))

        print(qm.states)



vqc_quantumpooling_circuit
""""""""""""""""""""""""""""""""""""""""

.. py:function:: pyvqnet.qnn.vqc.torch.vqc_quantumpooling_circuit(ignored_wires, sinks_wires, params, q_machine)

    Quantenschaltkreis, der Daten herunterabtastet.

    Um die Anzahl der Qubits in der Schaltung zu reduzieren, werden zunächst Qubit-Paare im System erstellt. Nach dem anfänglichen Paaren aller Qubits wird eine verallgemeinerte 2-Qubit-Unitäre auf jedes Qubit-Paar angewendet. Und nach dem Anwenden dieser zwei-Qubit-Unitären wird ein Qubit in jedem Qubit-Paar für den Rest des neuronalen Netzwerks ignoriert.

    :param sources_wires: Quell-Qubit-Indizes, die ignoriert werden.
    :param sinks_wires: Ziel-Qubit-Indizes, die beibehalten werden.
    :param params: Eingabeparameter.
    :param q_machine: Quanten-Virtuelle-Maschinen-Gerät.

    Examples:: 

        import pyvqnet

        pyvqnet.backends.set_backend("torch")
        from pyvqnet.qnn.vqc.torch import vqc_quantumpooling_circuit, QMachine, MeasureAll
        from pyvqnet import tensor
        p = tensor.full([6], 0.35)
        qm = QMachine(4)
        vqc_quantumpooling_circuit(q_machine=qm,
                                ignored_wires=[0, 1],
                                sinks_wires=[2, 3],
                                params=p)
        m = MeasureAll(obs={"Z1": 1})
        exp = m(q_machine=qm)
        print(exp)



QuantumLayerAdjoint
""""""""""""""""""""""""""""""""""""""""

.. py:class:: pyvqnet.qnn.vqc.torch.QuantumLayerAdjoint(general_module: pyvqnet.nn.Module, use_qpanda=False, name="")


    Eine automatisch differenzierbare QuantumLayer-Schicht, die den adjungierten Matrix-Ansatz zur Berechnung von Gradienten verwendet, see `Efficient calculation of gradients in classical simulations of variational quantum algorithms <https://arxiv.org/abs/2009.02823>`_ .

    :param general_module: eine `pyvqnet.nn.Module`-Instanz, die nur mit der Quantenschaltkreis-Schnittstelle unter ``pyvqnet.qnn.vqc.torch`` erstellt wurde.
    :param use_qpanda: Ob die qpanda-Leitung für die Vorwärtsübertragung verwendet werden soll, Standard: False.
    :param name: Der Name der Schicht, Standard ist "".

    .. note::

        Die QMachine von general_module sollte grad_method = "adjoint" setzen.

        Unterstützt derzeit die folgenden parametrisierten Logikgatter `RX`, `RY`, `RZ`, `PhaseShift`, `RXX`, `RYY`, `RZZ`, `RZX`, `U1`, `U2`, `U3` und andere variationelle Schaltkreise, die aus nicht-parametrischen Logikgittern bestehen.


    Example::

        import pyvqnet
        pyvqnet.backends.set_backend("torch")
        from pyvqnet import tensor
        from pyvqnet.qnn.vqc.torch import QuantumLayerAdjoint, \
            QMachine, RX, RY, CNOT, T, \
                MeasureAll, RZ, VQC_HardwareEfficientAnsatz,\
                    QModule

        class QModel(QModule):
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
                self.measure = MeasureAll(obs={
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
        input_x = tensor.broadcast_to(input_x, [40, 3])
        input_x.requires_grad = True
        qunatum_model = QModel(num_wires=6,
                            dtype=pyvqnet.kcomplex64,
                            grad_mode="adjoint")
        adjoint_model = QuantumLayerAdjoint(qunatum_model, qunatum_model.qm)
        batch_y = adjoint_model(input_x)
        batch_y.backward()





Tensor-Netzwerk-Backend-Modul für variationelle Quantenschaltkreise
==========================================================================================

Tensor-Netzwerke (TN) reduzieren die Rechenkomplexität erheblich, indem sie einen komplexen Tensor in ein Netzwerk mehrerer niedrigdimensionaler Tensoren zerlegen.

Matrix Product State (MPS) ist eine spezielle Form des Tensor-Netzwerks. MPS stellt einen Quantenzustand als Produkt einer Reihe von Matrizen dar und reduziert so effektiv die Anzahl der Parameter und die Rechenkomplexität.

Die folgende Schnittstelle basiert auf dem ``torch``-Backend und bietet funktionale Unterstützung für die Konstruktion von Quantenschaltkreisen in Tensor-Netzwerken, einschließlich der Konstruktion von Quantenschaltkreis-Basisklassen, Quantenlogikgattern, Quantenschaltkreisen und Messungen sowie der Berechnung von Parametergradienten durch automatische differentielle Simulation anstelle der Parameterdrift-Methode.

Die Konstruktion von Quantenleitungen im MPS-Verfahren ergänzt die Unterstützung für die Konstruktion von Quantenleitungen mit großen Bit-Zahlen.

.. warning::

        Die Verwendung der folgenden Funktionen in diesem Modul erfordert die zusätzliche Installation von ``tensornetwork`` und ``torch``. Die Standardinstallation von ``pyvqnet`` enthält diese beiden Abhängigkeiten nicht. Bitte installieren Sie sie mit ``pip install tensornetwork torch``.

.. warning::

        Ermöglicht MPS den Aufbau von Quantenleitungen über den Parameter ``use_mps`` in ``TNQMachine``, was Quantenleitungs-Implementierungen mit großen Bit-Zahlen (100 und mehr) unterstützt.

.. warning::
        
        Batching wird anders verwendet als bei klassischen Modulen, basierend auf dem vmap-Ansatz, bei dem die Daten- und Parameterkonstruktionsleitungen eine Dimension tiefer eingegeben werden müssen, wie in der folgenden Beispielschnittstelle gezeigt, und die Batch-Ausführung muss sowohl auf ``TNQMachine`` als auch auf ``TNQModule`` basieren.

Basisklasse
------------------------------------------------

Das Schreiben eines variationellen Quantenschaltkreismodells auf einem Tensor-Netzwerk erfordert die Vererbung von ``TNQModule``.

TNQModule
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. py:class:: pyvqnet.qnn.vqc.tn.TNQModule(use_jit=False, vectorized_argnums=0, name="")

    .. note::

        Diese Klasse und ihre abgeleiteten Klassen sind nur für ``pyvqnet.backends.set_backend("torch")`` anwendbar, nicht mit dem ``Module`` unter dem standardmäßigen ``pyvqnet.nn`` mischen.

        The data in ``_buffers`` of this class is of ``torch.Tensor`` type.

        The data in ``_parmeters`` of this class is of ``torch.nn.Parameter`` type.

    :param use_jit: steuert die JIT-Kompilierungsfunktion des Quantenschaltkreises.
    :param vectorized_argnums: die zu vektorisierenden Argumente,
            diese Argumente sollten in der ersten Dimension die gleiche Batch-Form haben, Standard ist 0.
    :param name: Name des Moduls.

    Example::

        import pyvqnet
        from pyvqnet.nn import Parameter
        pyvqnet.backends.set_backend("torch")
        from pyvqnet.qnn.vqc.tn import TNQModule
        from pyvqnet.qnn.vqc.tn import TNQMachine, RX, RY, CNOT, PauliX, PauliZ,qmeasure,qcircuit,VQC_RotCircuit
        class QModel(TNQModule):
            def __init__(self, num_wires, dtype,batch_size=2):
                super(QModel, self).__init__()

                self._num_wires = num_wires
                self._dtype = dtype
                self.qm = TNQMachine(num_wires, dtype=dtype)

                self.w = Parameter((2,4,3),initializer=pyvqnet.utils.initializer.quantum_uniform)
                self.cnot = CNOT(wires=[0, 1])
                self.batch_size = batch_size
            def forward(self, x, *args, **kwargs):
                self.qm.reset_states(batchsize=self.batch_size)

                def get_cnot(nqubits,qm):
                    for i in range(len(nqubits) - 1):
                        CNOT(wires = [nqubits[i], nqubits[i + 1]])(q_machine = qm)
                    CNOT(wires = [nqubits[len(nqubits) - 1], nqubits[0]])(q_machine = qm)


                def build_circuit(weights, xx, nqubits,qm):
                    def Rot(weights_j, nqubits,qm):#pylint:disable=invalid-name
                        VQC_RotCircuit(qm,nqubits,weights_j)

                    def basisstate(qm,xx, nqubits):
                        for i in nqubits:
                            qcircuit.rz(q_machine=qm, wires=i, params=xx[i])
                            qcircuit.ry(q_machine=qm, wires=i, params=xx[i])
                            qcircuit.rz(q_machine=qm, wires=i, params=xx[i])

                    basisstate(qm,xx,nqubits)

                    for i in range(weights.shape[0]):

                        weights_i = weights[i, :, :]
                        for j in range(len(nqubits)):
                            weights_j = weights_i[j]
                            Rot(weights_j, nqubits[j],qm)
                        get_cnot(nqubits,qm)

                build_circuit(self.w, x,range(4),self.qm)

                y= qmeasure.MeasureAll(obs={'Z0': 1})(self.qm)
                return y


        x= pyvqnet.tensor.QTensor([[1,0,0,1],[1,1,0,1]],dtype=pyvqnet.kfloat32)
        model = QModel(4,pyvqnet.kcomplex64,2)
        y = model(x)
        y.backward()


TNQMachine
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. py:class:: pyvqnet.qnn.vqc.tn.TNQMachine(num_wires, dtype=pyvqnet.kcomplex64, use_mps=False)

    Simulatorklasse für variationelles Quantenrechnen, einschließlich Zustandsvektoren, deren Zustandsattribut Quantenschaltkreise sind.

    Diese Klasse erbt von ``pyvqnet.nn.torch.TorchModule`` und ``pyvqnet.qnn.QMachine``.

    Diese Klasse kann als Untermodul von ``torch.nn.Module`` zum torch-Modell hinzugefügt werden.

    .. warning::
        
        Im Quantenschaltkreis des Tensor-Netzwerks wird die ``vmap``-Funktion standardmäßig aktiviert, und die Batch-Dimension wird in den Logikgatter-Parametern auf der Leitung verworfen.
        Bei Verwendung des Aufrufparameters, wenn die Dimension [batch_size, \*] ist, wird die erste batch_size-Dimension verworfen, und die folgenden Dimensionen werden direkt verwendet, z.B. für die Eingabedaten x[:,1] -> x[1], und auch für den trainierbaren Parameter, siehe das folgende Beispiel für die Verwendung von xx, weights.

    .. note::

        Vor jedem Durchlauf des vollständigen Quantenschaltkreises müssen Sie `pyvqnet.qnn.vqc.QMachine.reset_states(batchsize)` verwenden, um den Anfangszustand im Simulator neu zu initialisieren und ihn auf
        (batchsize,*)-Dimensionen zu übertragen, um sich an Batch-Datentraining anzupassen.

    :param num_wires: Anzahl der zu verwendenden Qubits
    :param dtype: interner Datentyp für Berechnungen.
    :param use_mps: MPSCircuit für große Bit-Modelle öffnen.

    :return: Gibt ein TNQMachine-Objekt aus.

    Example::
        
        import pyvqnet
        from pyvqnet.nn import Parameter
        pyvqnet.backends.set_backend("torch")
        from pyvqnet.qnn.vqc.tn import TNQModule
        from pyvqnet.qnn.vqc.tn import TNQMachine, RX, RY, CNOT, PauliX, PauliZ,qmeasure,qcircuit,VQC_RotCircuit
        class QModel(TNQModule):
            def __init__(self, num_wires, dtype,batch_size=2):
                super(QModel, self).__init__()

                self._num_wires = num_wires
                self._dtype = dtype
                self.qm = TNQMachine(num_wires, dtype=dtype)

                self.w = Parameter((2,4,3),initializer=pyvqnet.utils.initializer.quantum_uniform)
                self.cnot = CNOT(wires=[0, 1])
                self.batch_size = batch_size
            def forward(self, x, *args, **kwargs):
                self.qm.reset_states(batchsize=self.batch_size)

                def get_cnot(nqubits,qm):
                    for i in range(len(nqubits) - 1):
                        CNOT(wires = [nqubits[i], nqubits[i + 1]])(q_machine = qm)
                    CNOT(wires = [nqubits[len(nqubits) - 1], nqubits[0]])(q_machine = qm)


                def build_circuit(weights, xx, nqubits,qm):
                    def Rot(weights_j, nqubits,qm):#pylint:disable=invalid-name
                        VQC_RotCircuit(qm,nqubits,weights_j)

                    def basisstate(qm,xx, nqubits):
                        for i in nqubits:
                            qcircuit.rz(q_machine=qm, wires=i, params=xx[i])
                            qcircuit.ry(q_machine=qm, wires=i, params=xx[i])
                            qcircuit.rz(q_machine=qm, wires=i, params=xx[i])

                    basisstate(qm,xx,nqubits)

                    for i in range(weights.shape[0]):

                        weights_i = weights[i, :, :]
                        for j in range(len(nqubits)):
                            weights_j = weights_i[j]
                            Rot(weights_j, nqubits[j],qm)
                        get_cnot(nqubits,qm)

                build_circuit(self.w, x,range(4),self.qm)

                y= qmeasure.MeasureAll(obs={'Z0': 1})(self.qm)
                return y


        x= pyvqnet.tensor.QTensor([[1,0,0,1],[1,1,0,1]],dtype=pyvqnet.kfloat32)
        model = QModel(4,pyvqnet.kcomplex64,2)
        y = model(x)
        y.backward()

    .. py:method:: get_states()

        Ruft die Zustände der Tensor-Netzwerk-Quantenmaschine ab.

Modul für variationelle Quantenlogikgatter
------------------------------------------------

Die folgenden Funktionsschnittstellen in ``pyvqnet.qnn.vqc`` unterstützen direkt ``QTensor`` des ``torch``-Backends für Berechnungen, Importpfad ``pyvqnet.qnn.vqc.tn``.

.. csv-table:: List of supported pyvqnet.qnn.vqc interfaces
    :file: ./images/same_apis_from_tn.csv

Die folgenden Quantenschaltkreis-Module erben von ``pyvqnet.qnn.vqc.tn.TNQModule``, wobei Berechnungen mit ``torch.Tensor`` durchgeführt werden.

.. note::

    Diese Klasse und ihre abgeleiteten Klassen sind nur für ``pyvqnet.backends.set_backend("torch")`` anwendbar, nicht mit ``Module`` unter dem standardmäßigen ``pyvqnet.nn`` mischen.

    Wenn diese Klassen nicht-parametrische Member-Variablen ``_buffers`` haben, sind die darin enthaltenen Daten vom Typ ``torch.Tensor``.
    Wenn diese Klassen parametrische Member-Variablen ``_parameters`` haben, sind die darin enthaltenen Daten vom Typ ``torch.nn.Parameter``.

I
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. py:class:: pyvqnet.qnn.vqc.tn.I(has_params: bool = False,trainable: bool = False,init_params=None,wires=None,dtype=pyvqnet.kcomplex64,use_dagger=False)
    
    Definiert ein I-Quantengatter.

    Diese Klasse erbt von ``pyvqnet.qnn.vqc.tn.QModule`` und ``torch.nn.Module``.
    Diese Klasse kann als Untermodul von ``torch.nn.Module`` zum torch-Modell hinzugefügt werden.

    :param has_params: ob es Parameter hat, wie RX, RY und andere Gatter auf True gesetzt werden müssen, und solche ohne Parameter auf False gesetzt werden müssen, der Standard ist False.
    :param trainable: ob es zu trainierende Parameter hat. Wenn die Schicht externe Eingabedaten zum Aufbau der Logikgatter-Matrix verwendet, auf False setzen. Wenn die zu trainierenden Parameter von dieser Schicht initialisiert werden müssen, ist es True, der Standard ist False.
    :param init_params: Initialisierungsparameter zum Codieren klassischer Daten QTensor, der Standard ist None.
    :param wires: Bit-Index des Leitungseffekts, der Standard ist None.
    :param dtype: Die Datenpräzision der internen Matrix des Logikgatters kann auf pyvqnet.kcomplex64 oder pyvqnet.kcomplex128 gesetzt werden, entsprechend float- bzw. double-Eingabe.
    :param use_dagger: ob die transponiert-konjugierte Version des Gatters verwendet werden soll, der Standard ist False.
    :return: eine ``pyvqnet.qnn.vqc.tn.QModule``-Instanz

    Example::
        
        from pyvqnet.qnn.vqc.tn import I,TNQMachine,TNQModule,MeasureAll, rx
        import pyvqnet
        pyvqnet.backends.set_backend("torch")

        class QModel(TNQModule):
            
            def __init__(self, num_wires, dtype,batch_size=2):
                super(QModel, self).__init__()
                self.device = TNQMachine(num_wires)
                self.layer = I(wires=0)
                self.batch_size = batch_size
                self.num_wires = num_wires
                
            def forward(self, x, *args, **kwargs):
                self.device.reset_states(batchsize=self.batch_size)
                for i in range(self.num_wires):
                    rx(self.device, wires=i, params=x[i])
                self.layer(q_machine = self.device)
                y = MeasureAll(obs={'Z0': 1})(self.device)
                return y

        x = pyvqnet.tensor.QTensor([[1,0,0,1],[1,1,0,1]],dtype=pyvqnet.kfloat32,requires_grad=True)
        model = QModel(4,pyvqnet.kcomplex64,2)
        y = model(x)
        print(y)



Hadamard
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. py:class:: pyvqnet.qnn.vqc.tn.Hadamard(has_params: bool = False,trainable: bool = False,init_params=None,wires=None,dtype=pyvqnet.kcomplex64,use_dagger=False)
    
    Definiert ein Hadamard-Quantengatter.

    Diese Klasse erbt von ``pyvqnet.qnn.vqc.tn.QModule`` und ``torch.nn.Module``.
    Diese Klasse kann als Untermodul von ``torch.nn.Module`` zum torch-Modell hinzugefügt werden.

    :param has_params: ob es Parameter hat, wie RX, RY und andere Gatter auf True gesetzt werden müssen, und solche ohne Parameter auf False gesetzt werden müssen, der Standard ist False.
    :param trainable: ob es zu trainierende Parameter hat. Wenn die Schicht externe Eingabedaten zum Aufbau der Logikgatter-Matrix verwendet, auf False setzen. Wenn die zu trainierenden Parameter von dieser Schicht initialisiert werden müssen, ist es True, der Standard ist False.
    :param init_params: Initialisierungsparameter zum Codieren klassischer Daten QTensor, der Standard ist None.
    :param wires: Bit-Index des Leitungseffekts, der Standard ist None.
    :param dtype: Die Datenpräzision der internen Matrix des Logikgatters kann auf pyvqnet.kcomplex64 oder pyvqnet.kcomplex128 gesetzt werden, entsprechend float- bzw. double-Eingabe.
    :param use_dagger: ob die transponiert-konjugierte Version des Gatters verwendet werden soll, der Standard ist False.
    :return: eine ``pyvqnet.qnn.vqc.tn.QModule``-Instanz

    Example::
        
        from pyvqnet.qnn.vqc.tn import Hadamard,TNQMachine,TNQModule,MeasureAll, rx
        import pyvqnet
        pyvqnet.backends.set_backend("torch")

        class QModel(TNQModule):
            
            def __init__(self, num_wires, dtype,batch_size=2):
                super(QModel, self).__init__()
                self.device = TNQMachine(num_wires)
                self.layer = Hadamard(wires=0)
                self.batch_size = batch_size
                self.num_wires = num_wires
                
            def forward(self, x, *args, **kwargs):
                self.device.reset_states(batchsize=self.batch_size)
                for i in range(self.num_wires):
                    rx(self.device, wires=i, params=x[i])
                self.layer(q_machine = self.device)
                y = MeasureAll(obs={'Z0': 1})(self.device)
                return y

        x = pyvqnet.tensor.QTensor([[1,0,0,1],[1,1,0,1]],dtype=pyvqnet.kfloat32,requires_grad=True)
        model = QModel(4,pyvqnet.kcomplex64,2)
        y = model(x)
        print(y)


T
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. py:class:: pyvqnet.qnn.vqc.tn.T(has_params: bool = False,trainable: bool = False,init_params=None,wires=None,dtype=pyvqnet.kcomplex64,use_dagger=False)
    
    Definiert ein T-Quantengatter.

    Diese Klasse erbt von ``pyvqnet.qnn.vqc.tn.QModule`` und ``torch.nn.Module``.
    Diese Klasse kann als Untermodul von ``torch.nn.Module`` zum torch-Modell hinzugefügt werden.

    :param has_params: ob es Parameter hat, wie RX, RY und andere Gatter auf True gesetzt werden müssen, und solche ohne Parameter auf False gesetzt werden müssen, der Standard ist False.
    :param trainable: ob es zu trainierende Parameter hat. Wenn die Schicht externe Eingabedaten zum Aufbau der Logikgatter-Matrix verwendet, auf False setzen. Wenn die zu trainierenden Parameter von dieser Schicht initialisiert werden müssen, ist es True, der Standard ist False.
    :param init_params: Initialisierungsparameter zum Codieren klassischer Daten QTensor, der Standard ist None.
    :param wires: Bit-Index des Leitungseffekts, der Standard ist None.
    :param dtype: Die Datenpräzision der internen Matrix des Logikgatters kann auf pyvqnet.kcomplex64 oder pyvqnet.kcomplex128 gesetzt werden, entsprechend float- bzw. double-Eingabe.
    :param use_dagger: ob die transponiert-konjugierte Version des Gatters verwendet werden soll, der Standard ist False.
    :return: eine ``pyvqnet.qnn.vqc.tn.QModule``-Instanz

    Example::
        
        from pyvqnet.qnn.vqc.tn import T,TNQMachine,TNQModule,MeasureAll, rx
        import pyvqnet
        pyvqnet.backends.set_backend("torch")

        class QModel(TNQModule):
            
            def __init__(self, num_wires, dtype,batch_size=2):
                super(QModel, self).__init__()
                self.device = TNQMachine(num_wires)
                self.layer = T(wires=0)
                self.batch_size = batch_size
                self.num_wires = num_wires
                
            def forward(self, x, *args, **kwargs):
                self.device.reset_states(batchsize=self.batch_size)
                for i in range(self.num_wires):
                    rx(self.device, wires=i, params=x[i])
                self.layer(q_machine = self.device)
                y = MeasureAll(obs={'Z0': 1})(self.device)
                return y

        x = pyvqnet.tensor.QTensor([[1,0,0,1],[1,1,0,1]],dtype=pyvqnet.kfloat32,requires_grad=True)
        model = QModel(4,pyvqnet.kcomplex64,2)
        y = model(x)
        print(y)



S
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. py:class:: pyvqnet.qnn.vqc.tn.S(has_params: bool = False,trainable: bool = False,init_params=None,wires=None,dtype=pyvqnet.kcomplex64,use_dagger=False)
    
    Definiert ein S-Quantengatter.

    Diese Klasse erbt von ``pyvqnet.qnn.vqc.tn.QModule`` und ``torch.nn.Module``.
    Diese Klasse kann als Untermodul von ``torch.nn.Module`` zum torch-Modell hinzugefügt werden.

    :param has_params: ob es Parameter hat, wie RX, RY und andere Gatter auf True gesetzt werden müssen, und solche ohne Parameter auf False gesetzt werden müssen, der Standard ist False.
    :param trainable: ob es zu trainierende Parameter hat. Wenn die Schicht externe Eingabedaten zum Aufbau der Logikgatter-Matrix verwendet, auf False setzen. Wenn die zu trainierenden Parameter von dieser Schicht initialisiert werden müssen, ist es True, der Standard ist False.
    :param init_params: Initialisierungsparameter zum Codieren klassischer Daten QTensor, der Standard ist None.
    :param wires: Bit-Index des Leitungseffekts, der Standard ist None.
    :param dtype: Die Datenpräzision der internen Matrix des Logikgatters kann auf pyvqnet.kcomplex64 oder pyvqnet.kcomplex128 gesetzt werden, entsprechend float- bzw. double-Eingabe.
    :param use_dagger: ob die transponiert-konjugierte Version des Gatters verwendet werden soll, der Standard ist False.
    :return: eine ``pyvqnet.qnn.vqc.tn.QModule``-Instanz

    Example::
        
        from pyvqnet.qnn.vqc.tn import S,TNQMachine,TNQModule,MeasureAll, rx
        import pyvqnet
        pyvqnet.backends.set_backend("torch")

        class QModel(TNQModule):
            
            def __init__(self, num_wires, dtype,batch_size=2):
                super(QModel, self).__init__()
                self.device = TNQMachine(num_wires)
                self.layer = S(wires=0)
                self.batch_size = batch_size
                self.num_wires = num_wires
                
            def forward(self, x, *args, **kwargs):
                self.device.reset_states(batchsize=self.batch_size)
                for i in range(self.num_wires):
                    rx(self.device, wires=i, params=x[i])
                self.layer(q_machine = self.device)
                y = MeasureAll(obs={'Z0': 1})(self.device)
                return y

        x = pyvqnet.tensor.QTensor([[1,0,0,1],[1,1,0,1]],dtype=pyvqnet.kfloat32,requires_grad=True)
        model = QModel(4,pyvqnet.kcomplex64,2)
        y = model(x)
        print(y)


PauliX
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. py:class:: pyvqnet.qnn.vqc.tn.PauliX(has_params: bool = False,trainable: bool = False,init_params=None,wires=None,dtype=pyvqnet.kcomplex64,use_dagger=False)
    
    Definiert ein PauliX-Quantengatter.

    Diese Klasse erbt von ``pyvqnet.qnn.vqc.tn.QModule`` und ``torch.nn.Module``.
    Diese Klasse kann als Untermodul von ``torch.nn.Module`` zum torch-Modell hinzugefügt werden.


    :param has_params: ob es Parameter hat, wie RX, RY und andere Gatter auf True gesetzt werden müssen, und solche ohne Parameter auf False gesetzt werden müssen, der Standard ist False.
    :param trainable: ob es zu trainierende Parameter hat. Wenn die Schicht externe Eingabedaten zum Aufbau der Logikgatter-Matrix verwendet, auf False setzen. Wenn die zu trainierenden Parameter von dieser Schicht initialisiert werden müssen, ist es True, der Standard ist False.
    :param init_params: Initialisierungsparameter zum Codieren klassischer Daten QTensor, der Standard ist None.
    :param wires: Bit-Index des Leitungseffekts, der Standard ist None.
    :param dtype: Die Datenpräzision der internen Matrix des Logikgatters kann auf pyvqnet.kcomplex64 oder pyvqnet.kcomplex128 gesetzt werden, entsprechend float- bzw. double-Eingabe.
    :param use_dagger: ob die transponiert-konjugierte Version des Gatters verwendet werden soll, der Standard ist False.
    :return: eine ``pyvqnet.qnn.vqc.tn.QModule``-Instanz

    Example::
        
        from pyvqnet.qnn.vqc.tn import PauliX,TNQMachine,TNQModule,MeasureAll, rx
        import pyvqnet
        pyvqnet.backends.set_backend("torch")

        class QModel(TNQModule):
            
            def __init__(self, num_wires, dtype,batch_size=2):
                super(QModel, self).__init__()
                self.device = TNQMachine(num_wires)
                self.layer = PauliX(wires=0)
                self.batch_size = batch_size
                self.num_wires = num_wires
                
            def forward(self, x, *args, **kwargs):
                self.device.reset_states(batchsize=self.batch_size)
                for i in range(self.num_wires):
                    rx(self.device, wires=i, params=x[i])
                self.layer(q_machine = self.device)
                y = MeasureAll(obs={'Z0': 1})(self.device)
                return y

        x = pyvqnet.tensor.QTensor([[1,0,0,1],[1,1,0,1]],dtype=pyvqnet.kfloat32,requires_grad=True)
        model = QModel(4,pyvqnet.kcomplex64,2)
        y = model(x)
        print(y)



PauliY
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. py:class:: pyvqnet.qnn.vqc.tn.PauliY(has_params: bool = False,trainable: bool = False,init_params=None,wires=None,dtype=pyvqnet.kcomplex64,use_dagger=False)
    
    Definiert ein PauliY-Quantengatter.

    Diese Klasse erbt von ``pyvqnet.qnn.vqc.tn.QModule`` und ``torch.nn.Module``.
    Diese Klasse kann als Untermodul von ``torch.nn.Module`` zum torch-Modell hinzugefügt werden.


    :param has_params: ob es Parameter hat, wie RX, RY und andere Gatter auf True gesetzt werden müssen, und solche ohne Parameter auf False gesetzt werden müssen, der Standard ist False.
    :param trainable: ob es zu trainierende Parameter hat. Wenn die Schicht externe Eingabedaten zum Aufbau der Logikgatter-Matrix verwendet, auf False setzen. Wenn die zu trainierenden Parameter von dieser Schicht initialisiert werden müssen, ist es True, der Standard ist False.
    :param init_params: Initialisierungsparameter zum Codieren klassischer Daten QTensor, der Standard ist None.
    :param wires: Bit-Index des Leitungseffekts, der Standard ist None.
    :param dtype: Die Datenpräzision der internen Matrix des Logikgatters kann auf pyvqnet.kcomplex64 oder pyvqnet.kcomplex128 gesetzt werden, entsprechend float- bzw. double-Eingabe.
    :param use_dagger: ob die transponiert-konjugierte Version des Gatters verwendet werden soll, der Standard ist False.
    :return: eine ``pyvqnet.qnn.vqc.tn.QModule``-Instanz

    Example::
        
        from pyvqnet.qnn.vqc.tn import PauliY,TNQMachine,TNQModule,MeasureAll, rx
        import pyvqnet
        pyvqnet.backends.set_backend("torch")

        class QModel(TNQModule):
            
            def __init__(self, num_wires, dtype,batch_size=2):
                super(QModel, self).__init__()
                self.device = TNQMachine(num_wires)
                self.layer = PauliY(wires=0)
                self.batch_size = batch_size
                self.num_wires = num_wires
                
            def forward(self, x, *args, **kwargs):
                self.device.reset_states(batchsize=self.batch_size)
                for i in range(self.num_wires):
                    rx(self.device, wires=i, params=x[i])
                self.layer(q_machine = self.device)
                y = MeasureAll(obs={'Z0': 1})(self.device)
                return y

        x = pyvqnet.tensor.QTensor([[1,0,0,1],[1,1,0,1]],dtype=pyvqnet.kfloat32,requires_grad=True)
        model = QModel(4,pyvqnet.kcomplex64,2)
        y = model(x)
        print(y)


PauliZ
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. py:class:: pyvqnet.qnn.vqc.tn.PauliZ(has_params: bool = False,trainable: bool = False,init_params=None,wires=None,dtype=pyvqnet.kcomplex64,use_dagger=False)
    
    Definiert ein PauliZ-Quantengatter.

    Diese Klasse erbt von ``pyvqnet.qnn.vqc.tn.QModule`` und ``torch.nn.Module``.
    Diese Klasse kann als Untermodul von ``torch.nn.Module`` zum torch-Modell hinzugefügt werden.


    :param has_params: ob es Parameter hat, wie RX, RY und andere Gatter auf True gesetzt werden müssen, und solche ohne Parameter auf False gesetzt werden müssen, der Standard ist False.
    :param trainable: ob es zu trainierende Parameter hat. Wenn die Schicht externe Eingabedaten zum Aufbau der Logikgatter-Matrix verwendet, auf False setzen. Wenn die zu trainierenden Parameter von dieser Schicht initialisiert werden müssen, ist es True, der Standard ist False.
    :param init_params: Initialisierungsparameter zum Codieren klassischer Daten QTensor, der Standard ist None.
    :param wires: Bit-Index des Leitungseffekts, der Standard ist None.
    :param dtype: Die Datenpräzision der internen Matrix des Logikgatters kann auf pyvqnet.kcomplex64 oder pyvqnet.kcomplex128 gesetzt werden, entsprechend float- bzw. double-Eingabe.
    :param use_dagger: ob die transponiert-konjugierte Version des Gatters verwendet werden soll, der Standard ist False.
    :return: eine ``pyvqnet.qnn.vqc.tn.QModule``-Instanz

    Example::
        
        from pyvqnet.qnn.vqc.tn import PauliZ,TNQMachine,TNQModule,MeasureAll, rx
        import pyvqnet
        pyvqnet.backends.set_backend("torch")

        class QModel(TNQModule):
            
            def __init__(self, num_wires, dtype,batch_size=2):
                super(QModel, self).__init__()
                self.device = TNQMachine(num_wires)
                self.layer = PauliZ(wires=0)
                self.batch_size = batch_size
                self.num_wires = num_wires
                
            def forward(self, x, *args, **kwargs):
                self.device.reset_states(batchsize=self.batch_size)
                for i in range(self.num_wires):
                    rx(self.device, wires=i, params=x[i])
                self.layer(q_machine = self.device)
                y = MeasureAll(obs={'Z0': 1})(self.device)
                return y

        x = pyvqnet.tensor.QTensor([[1,0,0,1],[1,1,0,1]],dtype=pyvqnet.kfloat32,requires_grad=True)
        model = QModel(4,pyvqnet.kcomplex64,2)
        y = model(x)
        print(y)




X1
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. py:class:: pyvqnet.qnn.vqc.tn.X1(has_params: bool = False,trainable: bool = False,init_params=None,wires=None,dtype=pyvqnet.kcomplex64,use_dagger=False)
    
    Definiert ein X1-Quantengatter.

    Diese Klasse erbt von ``pyvqnet.qnn.vqc.tn.QModule`` und ``torch.nn.Module``.
    Diese Klasse kann als Untermodul von ``torch.nn.Module`` zum torch-Modell hinzugefügt werden.

    :param has_params: ob es Parameter hat, wie RX, RY und andere Gatter auf True gesetzt werden müssen, und solche ohne Parameter auf False gesetzt werden müssen, der Standard ist False.
    :param trainable: ob es zu trainierende Parameter hat. Wenn die Schicht externe Eingabedaten zum Aufbau der Logikgatter-Matrix verwendet, auf False setzen. Wenn die zu trainierenden Parameter von dieser Schicht initialisiert werden müssen, ist es True, der Standard ist False.
    :param init_params: Initialisierungsparameter zum Codieren klassischer Daten QTensor, der Standard ist None.
    :param wires: Bit-Index des Leitungseffekts, der Standard ist None.
    :param dtype: Die Datenpräzision der internen Matrix des Logikgatters kann auf pyvqnet.kcomplex64 oder pyvqnet.kcomplex128 gesetzt werden, entsprechend float- bzw. double-Eingabe.
    :param use_dagger: ob die transponiert-konjugierte Version des Gatters verwendet werden soll, der Standard ist False.
    :return: eine ``pyvqnet.qnn.vqc.tn.QModule``-Instanz

    Example::
        
        from pyvqnet.qnn.vqc.tn import X1,TNQMachine,TNQModule,MeasureAll, rx
        import pyvqnet
        pyvqnet.backends.set_backend("torch")

        class QModel(TNQModule):
            
            def __init__(self, num_wires, dtype,batch_size=2):
                super(QModel, self).__init__()
                self.device = TNQMachine(num_wires)
                self.layer = X1(wires=0)
                self.batch_size = batch_size
                self.num_wires = num_wires
                
            def forward(self, x, *args, **kwargs):
                self.device.reset_states(batchsize=self.batch_size)
                for i in range(self.num_wires):
                    rx(self.device, wires=i, params=x[i])
                self.layer(q_machine = self.device)
                y = MeasureAll(obs={'Z0': 1})(self.device)
                return y

        x = pyvqnet.tensor.QTensor([[1,0,0,1],[1,1,0,1]],dtype=pyvqnet.kfloat32,requires_grad=True)
        model = QModel(4,pyvqnet.kcomplex64,2)
        y = model(x)
        print(y)



RX
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. py:class:: pyvqnet.qnn.vqc.tn.RX(has_params: bool = False,trainable: bool = False,init_params=None,wires=None,dtype=pyvqnet.kcomplex64,use_dagger=False)
    
    Definiert ein RX-Quantengatter.

    Diese Klasse erbt von ``pyvqnet.qnn.vqc.tn.QModule`` und ``torch.nn.Module``.
    Diese Klasse kann als Untermodul von ``torch.nn.Module`` zum torch-Modell hinzugefügt werden.

    :param has_params: ob es Parameter hat, wie RX, RY und andere Gatter auf True gesetzt werden müssen, und solche ohne Parameter auf False gesetzt werden müssen, der Standard ist False.
    :param trainable: ob es zu trainierende Parameter hat. Wenn die Schicht externe Eingabedaten zum Aufbau der Logikgatter-Matrix verwendet, auf False setzen. Wenn die zu trainierenden Parameter von dieser Schicht initialisiert werden müssen, ist es True, der Standard ist False.
    :param init_params: Initialisierungsparameter zum Codieren klassischer Daten QTensor, der Standard ist None.
    :param wires: Bit-Index des Leitungseffekts, der Standard ist None.
    :param dtype: Die Datenpräzision der internen Matrix des Logikgatters kann auf pyvqnet.kcomplex64 oder pyvqnet.kcomplex128 gesetzt werden, entsprechend float- bzw. double-Eingabe.
    :param use_dagger: ob die transponiert-konjugierte Version des Gatters verwendet werden soll, der Standard ist False.
    :return: eine ``pyvqnet.qnn.vqc.tn.QModule``-Instanz

    Example::

        from pyvqnet.qnn.vqc.tn import RX,TNQMachine,TNQModule,MeasureAll, rx
        import pyvqnet
        pyvqnet.backends.set_backend("torch")

        class QModel(TNQModule):
            
            def __init__(self, num_wires, dtype,batch_size=2):
                super(QModel, self).__init__()
                self.device = TNQMachine(num_wires)
                self.layer = RX(wires=0,has_params=True)
                self.batch_size = batch_size
                self.num_wires = num_wires
                
            def forward(self, x, *args, **kwargs):
                self.device.reset_states(batchsize=self.batch_size)
                for i in range(self.num_wires):
                    rx(self.device, wires=i, params=x[i])
                self.layer(q_machine = self.device)
                y = MeasureAll(obs={'Z0': 1})(self.device)
                return y

        x = pyvqnet.tensor.QTensor([[1,0,0,1],[1,1,0,1]],dtype=pyvqnet.kfloat32,requires_grad=True)
        model = QModel(4,pyvqnet.kcomplex64,2)
        y = model(x)
        print(y)





RY
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. py:class:: pyvqnet.qnn.vqc.tn.RY(has_params: bool = False,trainable: bool = False,init_params=None,wires=None,dtype=pyvqnet.kcomplex64,use_dagger=False)
    
    Definiert ein RY-Quantengatter.

    Diese Klasse erbt von ``pyvqnet.qnn.vqc.tn.QModule`` und ``torch.nn.Module``.
    Diese Klasse kann als Untermodul von ``torch.nn.Module`` zum torch-Modell hinzugefügt werden.

    :param has_params: ob es Parameter hat, wie RX, RY und andere Gatter auf True gesetzt werden müssen, und solche ohne Parameter auf False gesetzt werden müssen, der Standard ist False.
    :param trainable: ob es zu trainierende Parameter hat. Wenn die Schicht externe Eingabedaten zum Aufbau der Logikgatter-Matrix verwendet, auf False setzen. Wenn die zu trainierenden Parameter von dieser Schicht initialisiert werden müssen, ist es True, der Standard ist False.
    :param init_params: Initialisierungsparameter zum Codieren klassischer Daten QTensor, der Standard ist None.
    :param wires: Bit-Index des Leitungseffekts, der Standard ist None.
    :param dtype: Die Datenpräzision der internen Matrix des Logikgatters kann auf pyvqnet.kcomplex64 oder pyvqnet.kcomplex128 gesetzt werden, entsprechend float- bzw. double-Eingabe.
    :param use_dagger: ob die transponiert-konjugierte Version des Gatters verwendet werden soll, der Standard ist False.
    :return: eine ``pyvqnet.qnn.vqc.tn.QModule``-Instanz

    Example::

        from pyvqnet.qnn.vqc.tn import RY,TNQMachine,TNQModule,MeasureAll, rx
        import pyvqnet
        pyvqnet.backends.set_backend("torch")

        class QModel(TNQModule):
            
            def __init__(self, num_wires, dtype,batch_size=2):
                super(QModel, self).__init__()
                self.device = TNQMachine(num_wires)
                self.layer = RY(wires=0,has_params=True)
                self.batch_size = batch_size
                self.num_wires = num_wires
                
            def forward(self, x, *args, **kwargs):
                self.device.reset_states(batchsize=self.batch_size)
                for i in range(self.num_wires):
                    rx(self.device, wires=i, params=x[i])
                self.layer(q_machine = self.device)
                y = MeasureAll(obs={'Z0': 1})(self.device)
                return y

        x = pyvqnet.tensor.QTensor([[1,0,0,1],[1,1,0,1]],dtype=pyvqnet.kfloat32,requires_grad=True)
        model = QModel(4,pyvqnet.kcomplex64,2)
        y = model(x)
        print(y)


RZ
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. py:class:: pyvqnet.qnn.vqc.tn.RZ(has_params: bool = False,trainable: bool = False,init_params=None,wires=None,dtype=pyvqnet.kcomplex64,use_dagger=False)
    
    Definiert ein RZ-Quantengatter.

    Diese Klasse erbt von ``pyvqnet.qnn.vqc.tn.QModule`` und ``torch.nn.Module``.
    Diese Klasse kann als Untermodul von ``torch.nn.Module`` zum torch-Modell hinzugefügt werden.

    :param has_params: ob es Parameter hat, wie RX, RY und andere Gatter auf True gesetzt werden müssen, und solche ohne Parameter auf False gesetzt werden müssen, der Standard ist False.
    :param trainable: ob es zu trainierende Parameter hat. Wenn die Schicht externe Eingabedaten zum Aufbau der Logikgatter-Matrix verwendet, auf False setzen. Wenn die zu trainierenden Parameter von dieser Schicht initialisiert werden müssen, ist es True, der Standard ist False.
    :param init_params: Initialisierungsparameter zum Codieren klassischer Daten QTensor, der Standard ist None.
    :param wires: Bit-Index des Leitungseffekts, der Standard ist None.
    :param dtype: Die Datenpräzision der internen Matrix des Logikgatters kann auf pyvqnet.kcomplex64 oder pyvqnet.kcomplex128 gesetzt werden, entsprechend float- bzw. double-Eingabe.
    :param use_dagger: ob die transponiert-konjugierte Version des Gatters verwendet werden soll, der Standard ist False.
    :return: eine ``pyvqnet.qnn.vqc.tn.QModule``-Instanz

    Example::

        from pyvqnet.qnn.vqc.tn import RZ,TNQMachine,TNQModule,MeasureAll, rx
        import pyvqnet
        pyvqnet.backends.set_backend("torch")

        class QModel(TNQModule):
            
            def __init__(self, num_wires, dtype,batch_size=2):
                super(QModel, self).__init__()
                self.device = TNQMachine(num_wires)
                self.layer = RZ(wires=0,has_params=True)
                self.batch_size = batch_size
                self.num_wires = num_wires
                
            def forward(self, x, *args, **kwargs):
                self.device.reset_states(batchsize=self.batch_size)
                for i in range(self.num_wires):
                    rx(self.device, wires=i, params=x[i])
                self.layer(q_machine = self.device)
                y = MeasureAll(obs={'Z0': 1})(self.device)
                return y

        x = pyvqnet.tensor.QTensor([[1,0,0,1],[1,1,0,1]],dtype=pyvqnet.kfloat32,requires_grad=True)
        model = QModel(4,pyvqnet.kcomplex64,2)
        y = model(x)
        print(y)


CRX
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. py:class:: pyvqnet.qnn.vqc.tn.CRX(has_params: bool = False,trainable: bool = False,init_params=None,wires=None,dtype=pyvqnet.kcomplex64,use_dagger=False)
    
    Definiert ein CRX-Quantengatter.

    Diese Klasse erbt von ``pyvqnet.qnn.vqc.tn.QModule`` und ``torch.nn.Module``.
    Diese Klasse kann als Untermodul von ``torch.nn.Module`` zum torch-Modell hinzugefügt werden.

    :param has_params: ob es Parameter hat, wie RX, RY und andere Gatter auf True gesetzt werden müssen, und solche ohne Parameter auf False gesetzt werden müssen, der Standard ist False.
    :param trainable: ob es zu trainierende Parameter hat. Wenn die Schicht externe Eingabedaten zum Aufbau der Logikgatter-Matrix verwendet, auf False setzen. Wenn die zu trainierenden Parameter von dieser Schicht initialisiert werden müssen, ist es True, der Standard ist False.
    :param init_params: Initialisierungsparameter zum Codieren klassischer Daten QTensor, der Standard ist None.
    :param wires: Bit-Index des Leitungseffekts, der Standard ist None.
    :param dtype: Die Datenpräzision der internen Matrix des Logikgatters kann auf pyvqnet.kcomplex64 oder pyvqnet.kcomplex128 gesetzt werden, entsprechend float- bzw. double-Eingabe.
    :param use_dagger: ob die transponiert-konjugierte Version des Gatters verwendet werden soll, der Standard ist False.
    :return: eine ``pyvqnet.qnn.vqc.tn.QModule``-Instanz

    Example::

        from pyvqnet.qnn.vqc.tn import CRX,TNQMachine,TNQModule,MeasureAll, rx
        import pyvqnet
        pyvqnet.backends.set_backend("torch")

        class QModel(TNQModule):
            
            def __init__(self, num_wires, dtype,batch_size=2):
                super(QModel, self).__init__()
                self.device = TNQMachine(num_wires)
                self.layer = CRX(has_params= True, trainable= True, wires=[0,2])
                self.batch_size = batch_size
                self.num_wires = num_wires
                
            def forward(self, x, *args, **kwargs):
                self.device.reset_states(batchsize=self.batch_size)
                for i in range(self.num_wires):
                    rx(self.device, wires=i, params=x[i])
                self.layer(q_machine = self.device)
                y = MeasureAll(obs={'Z0': 1})(self.device)
                return y

        x = pyvqnet.tensor.QTensor([[1,0,0,1],[1,1,0,1]],dtype=pyvqnet.kfloat32,requires_grad=True)
        model = QModel(4,pyvqnet.kcomplex64,2)
        y = model(x)
        print(y)




CRY
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^


.. py:class:: pyvqnet.qnn.vqc.tn.CRY(has_params: bool = False,trainable: bool = False,init_params=None,wires=None,dtype=pyvqnet.kcomplex64,use_dagger=False)
    
    Definiert ein CRY-Quantengatter.

    Diese Klasse erbt von ``pyvqnet.qnn.vqc.tn.QModule`` und ``torch.nn.Module``.
    Diese Klasse kann als Untermodul von ``torch.nn.Module`` zum torch-Modell hinzugefügt werden.

    :param has_params: ob es Parameter hat, wie RX, RY und andere Gatter auf True gesetzt werden müssen, und solche ohne Parameter auf False gesetzt werden müssen, der Standard ist False.
    :param trainable: ob es zu trainierende Parameter hat. Wenn die Schicht externe Eingabedaten zum Aufbau der Logikgatter-Matrix verwendet, auf False setzen. Wenn die zu trainierenden Parameter von dieser Schicht initialisiert werden müssen, ist es True, der Standard ist False.
    :param init_params: Initialisierungsparameter zum Codieren klassischer Daten QTensor, der Standard ist None.
    :param wires: Bit-Index des Leitungseffekts, der Standard ist None.
    :param dtype: Die Datenpräzision der internen Matrix des Logikgatters kann auf pyvqnet.kcomplex64 oder pyvqnet.kcomplex128 gesetzt werden, entsprechend float- bzw. double-Eingabe.
    :param use_dagger: ob die transponiert-konjugierte Version des Gatters verwendet werden soll, der Standard ist False.
    :return: eine ``pyvqnet.qnn.vqc.tn.QModule``-Instanz

    Example::

        from pyvqnet.qnn.vqc.tn import CRY,TNQMachine,TNQModule,MeasureAll, rx
        import pyvqnet
        pyvqnet.backends.set_backend("torch")

        class QModel(TNQModule):
            
            def __init__(self, num_wires, dtype,batch_size=2):
                super(QModel, self).__init__()
                self.device = TNQMachine(num_wires)
                self.layer = CRY(has_params= True, trainable= True, wires=[0,2])
                self.batch_size = batch_size
                self.num_wires = num_wires
                
            def forward(self, x, *args, **kwargs):
                self.device.reset_states(batchsize=self.batch_size)
                for i in range(self.num_wires):
                    rx(self.device, wires=i, params=x[i])
                self.layer(q_machine = self.device)
                y = MeasureAll(obs={'Z0': 1})(self.device)
                return y

        x = pyvqnet.tensor.QTensor([[1,0,0,1],[1,1,0,1]],dtype=pyvqnet.kfloat32,requires_grad=True)
        model = QModel(4,pyvqnet.kcomplex64,2)
        y = model(x)
        print(y)


CRZ
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^


.. py:class:: pyvqnet.qnn.vqc.tn.CRZ(has_params: bool = False,trainable: bool = False,init_params=None,wires=None,dtype=pyvqnet.kcomplex64,use_dagger=False)
    
    Definiert ein CRZ-Quantengatter.

    Diese Klasse erbt von ``pyvqnet.qnn.vqc.tn.QModule`` und ``torch.nn.Module``.
    Diese Klasse kann als Untermodul von ``torch.nn.Module`` zum torch-Modell hinzugefügt werden.

    :param has_params: ob es Parameter hat, wie RX, RY und andere Gatter auf True gesetzt werden müssen, und solche ohne Parameter auf False gesetzt werden müssen, der Standard ist False.
    :param trainable: ob es zu trainierende Parameter hat. Wenn die Schicht externe Eingabedaten zum Aufbau der Logikgatter-Matrix verwendet, auf False setzen. Wenn die zu trainierenden Parameter von dieser Schicht initialisiert werden müssen, ist es True, der Standard ist False.
    :param init_params: Initialisierungsparameter zum Codieren klassischer Daten QTensor, der Standard ist None.
    :param wires: Bit-Index des Leitungseffekts, der Standard ist None.
    :param dtype: Die Datenpräzision der internen Matrix des Logikgatters kann auf pyvqnet.kcomplex64 oder pyvqnet.kcomplex128 gesetzt werden, entsprechend float- bzw. double-Eingabe.
    :param use_dagger: ob die transponiert-konjugierte Version des Gatters verwendet werden soll, der Standard ist False.
    :return: eine ``pyvqnet.qnn.vqc.tn.QModule``-Instanz

    Example::

        from pyvqnet.qnn.vqc.tn import CRZ,TNQMachine,TNQModule,MeasureAll, rx
        import pyvqnet
        pyvqnet.backends.set_backend("torch")

        class QModel(TNQModule):
            
            def __init__(self, num_wires, dtype,batch_size=2):
                super(QModel, self).__init__()
                self.device = TNQMachine(num_wires)
                self.layer = CRZ(has_params= True, trainable= True, wires=[0,2])
                self.batch_size = batch_size
                self.num_wires = num_wires
                
            def forward(self, x, *args, **kwargs):
                self.device.reset_states(batchsize=self.batch_size)
                for i in range(self.num_wires):
                    rx(self.device, wires=i, params=x[i])
                self.layer(q_machine = self.device)
                y = MeasureAll(obs={'Z0': 1})(self.device)
                return y

        x = pyvqnet.tensor.QTensor([[1,0,0,1],[1,1,0,1]],dtype=pyvqnet.kfloat32,requires_grad=True)
        model = QModel(4,pyvqnet.kcomplex64,2)
        y = model(x)
        print(y)



U1
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. py:class:: pyvqnet.qnn.vqc.tn.U1(has_params: bool = False,trainable: bool = False,init_params=None,wires=None,dtype=pyvqnet.kcomplex64,use_dagger=False)
    
    Definiert ein U1-Quantengatter.

    Diese Klasse erbt von ``pyvqnet.qnn.vqc.tn.QModule`` und ``torch.nn.Module``.
    Diese Klasse kann als Untermodul von ``torch.nn.Module`` zum torch-Modell hinzugefügt werden.

    :param has_params: ob es Parameter hat, wie RX, RY und andere Gatter auf True gesetzt werden müssen, und solche ohne Parameter auf False gesetzt werden müssen, der Standard ist False.
    :param trainable: ob es zu trainierende Parameter hat. Wenn die Schicht externe Eingabedaten zum Aufbau der Logikgatter-Matrix verwendet, auf False setzen. Wenn die zu trainierenden Parameter von dieser Schicht initialisiert werden müssen, ist es True, der Standard ist False.
    :param init_params: Initialisierungsparameter zum Codieren klassischer Daten QTensor, der Standard ist None.
    :param wires: Bit-Index des Leitungseffekts, der Standard ist None.
    :param dtype: Die Datenpräzision der internen Matrix des Logikgatters kann auf pyvqnet.kcomplex64 oder pyvqnet.kcomplex128 gesetzt werden, entsprechend float- bzw. double-Eingabe.
    :param use_dagger: ob die transponiert-konjugierte Version des Gatters verwendet werden soll, der Standard ist False.
    :return: eine ``pyvqnet.qnn.vqc.tn.QModule``-Instanz

    Example::

        from pyvqnet.qnn.vqc.tn import U1,TNQMachine,TNQModule,MeasureAll, rx
        import pyvqnet
        pyvqnet.backends.set_backend("torch")

        class QModel(TNQModule):
            
            def __init__(self, num_wires, dtype,batch_size=2):
                super(QModel, self).__init__()
                self.device = TNQMachine(num_wires)
                self.layer = U1(has_params= True, trainable= True, wires=0)
                self.batch_size = batch_size
                self.num_wires = num_wires
                
            def forward(self, x, *args, **kwargs):
                self.device.reset_states(batchsize=self.batch_size)
                for i in range(self.num_wires):
                    rx(self.device, wires=i, params=x[i])
                self.layer(q_machine = self.device)
                y = MeasureAll(obs={'Z0': 1})(self.device)
                return y

        x = pyvqnet.tensor.QTensor([[1,0,0,1],[1,1,0,1]],dtype=pyvqnet.kfloat32,requires_grad=True)
        model = QModel(4,pyvqnet.kcomplex64,2)
        y = model(x)
        print(y)


U2
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^


.. py:class:: pyvqnet.qnn.vqc.tn.U2(has_params: bool = False,trainable: bool = False,init_params=None,wires=None,dtype=pyvqnet.kcomplex64,use_dagger=False)
    
    Definiert ein U2-Quantengatter.

    Diese Klasse erbt von ``pyvqnet.qnn.vqc.tn.QModule`` und ``torch.nn.Module``.
    Diese Klasse kann als Untermodul von ``torch.nn.Module`` zum torch-Modell hinzugefügt werden.

    :param has_params: ob es Parameter hat, wie RX, RY und andere Gatter auf True gesetzt werden müssen, und solche ohne Parameter auf False gesetzt werden müssen, der Standard ist False.
    :param trainable: ob es zu trainierende Parameter hat. Wenn die Schicht externe Eingabedaten zum Aufbau der Logikgatter-Matrix verwendet, auf False setzen. Wenn die zu trainierenden Parameter von dieser Schicht initialisiert werden müssen, ist es True, der Standard ist False.
    :param init_params: Initialisierungsparameter zum Codieren klassischer Daten QTensor, der Standard ist None.
    :param wires: Bit-Index des Leitungseffekts, der Standard ist None.
    :param dtype: Die Datenpräzision der internen Matrix des Logikgatters kann auf pyvqnet.kcomplex64 oder pyvqnet.kcomplex128 gesetzt werden, entsprechend float- bzw. double-Eingabe.
    :param use_dagger: ob die transponiert-konjugierte Version des Gatters verwendet werden soll, der Standard ist False.
    :return: eine ``pyvqnet.qnn.vqc.tn.QModule``-Instanz

    Example::

        from pyvqnet.qnn.vqc.tn import U2,TNQMachine,TNQModule,MeasureAll, rx
        import pyvqnet
        pyvqnet.backends.set_backend("torch")

        class QModel(TNQModule):
            
            def __init__(self, num_wires, dtype,batch_size=2):
                super(QModel, self).__init__()
                self.device = TNQMachine(num_wires)
                self.layer = U2(has_params= True, trainable= True, wires=0)
                self.batch_size = batch_size
                self.num_wires = num_wires
                
            def forward(self, x, *args, **kwargs):
                self.device.reset_states(batchsize=self.batch_size)
                for i in range(self.num_wires):
                    rx(self.device, wires=i, params=x[i])
                self.layer(q_machine = self.device)
                y = MeasureAll(obs={'Z0': 1})(self.device)
                return y

        x = pyvqnet.tensor.QTensor([[1,0,0,1],[1,1,0,1]],dtype=pyvqnet.kfloat32,requires_grad=True)
        model = QModel(4,pyvqnet.kcomplex64,2)
        y = model(x)
        print(y)


U3
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^


.. py:class:: pyvqnet.qnn.vqc.tn.U3(has_params: bool = False,trainable: bool = False,init_params=None,wires=None,dtype=pyvqnet.kcomplex64,use_dagger=False)
    
    Definiert ein U3-Quantengatter.

    Diese Klasse erbt von ``pyvqnet.qnn.vqc.tn.QModule`` und ``torch.nn.Module``.
    Diese Klasse kann als Untermodul von ``torch.nn.Module`` zum torch-Modell hinzugefügt werden.

    :param has_params: ob es Parameter hat, wie RX, RY und andere Gatter auf True gesetzt werden müssen, und solche ohne Parameter auf False gesetzt werden müssen, der Standard ist False.
    :param trainable: ob es zu trainierende Parameter hat. Wenn die Schicht externe Eingabedaten zum Aufbau der Logikgatter-Matrix verwendet, auf False setzen. Wenn die zu trainierenden Parameter von dieser Schicht initialisiert werden müssen, ist es True, der Standard ist False.
    :param init_params: Initialisierungsparameter zum Codieren klassischer Daten QTensor, der Standard ist None.
    :param wires: Bit-Index des Leitungseffekts, der Standard ist None.
    :param dtype: Die Datenpräzision der internen Matrix des Logikgatters kann auf pyvqnet.kcomplex64 oder pyvqnet.kcomplex128 gesetzt werden, entsprechend float- bzw. double-Eingabe.
    :param use_dagger: ob die transponiert-konjugierte Version des Gatters verwendet werden soll, der Standard ist False.
    :return: eine ``pyvqnet.qnn.vqc.tn.QModule``-Instanz

    Example::

        from pyvqnet.qnn.vqc.tn import U3,TNQMachine,TNQModule,MeasureAll, rx
        import pyvqnet
        pyvqnet.backends.set_backend("torch")

        class QModel(TNQModule):
            
            def __init__(self, num_wires, dtype,batch_size=2):
                super(QModel, self).__init__()
                self.device = TNQMachine(num_wires)
                self.layer = U3(has_params= True, trainable= True, wires=0)
                self.batch_size = batch_size
                self.num_wires = num_wires
                
            def forward(self, x, *args, **kwargs):
                self.device.reset_states(batchsize=self.batch_size)
                for i in range(self.num_wires):
                    rx(self.device, wires=i, params=x[i])
                self.layer(q_machine = self.device)
                y = MeasureAll(obs={'Z0': 1})(self.device)
                return y

        x = pyvqnet.tensor.QTensor([[1,0,0,1],[1,1,0,1]],dtype=pyvqnet.kfloat32,requires_grad=True)
        model = QModel(4,pyvqnet.kcomplex64,2)
        y = model(x)
        print(y)



CNOT
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. py:class:: pyvqnet.qnn.vqc.tn.CNOT(has_params: bool = False,trainable: bool = False,init_params=None,wires=None,dtype=pyvqnet.kcomplex64,use_dagger=False)
    
    Definiert ein CNOT-Quantengatter, Alias `CX`.

    Diese Klasse erbt von ``pyvqnet.qnn.vqc.tn.QModule`` und ``torch.nn.Module``.
    Diese Klasse kann als Untermodul von ``torch.nn.Module`` zum torch-Modell hinzugefügt werden.

    :param has_params: ob es Parameter hat, wie RX, RY und andere Gatter auf True gesetzt werden müssen, und solche ohne Parameter auf False gesetzt werden müssen, der Standard ist False.
    :param trainable: ob es zu trainierende Parameter hat. Wenn die Schicht externe Eingabedaten zum Aufbau der Logikgatter-Matrix verwendet, auf False setzen. Wenn die zu trainierenden Parameter von dieser Schicht initialisiert werden müssen, ist es True, der Standard ist False.
    :param init_params: Initialisierungsparameter zum Codieren klassischer Daten QTensor, der Standard ist None.
    :param wires: Bit-Index des Leitungseffekts, der Standard ist None.
    :param dtype: Die Datenpräzision der internen Matrix des Logikgatters kann auf pyvqnet.kcomplex64 oder pyvqnet.kcomplex128 gesetzt werden, entsprechend float- bzw. double-Eingabe.
    :param use_dagger: ob die transponiert-konjugierte Version des Gatters verwendet werden soll, der Standard ist False.
    :return: eine ``pyvqnet.qnn.vqc.tn.QModule``-Instanz

    Example::

        from pyvqnet.qnn.vqc.tn import CNOT,TNQMachine,TNQModule,MeasureAll, rx
        import pyvqnet
        pyvqnet.backends.set_backend("torch")

        class QModel(TNQModule):
            
            def __init__(self, num_wires, dtype,batch_size=2):
                super(QModel, self).__init__()
                self.device = TNQMachine(num_wires)
                self.layer = CNOT(wires=[0,1])
                self.batch_size = batch_size
                self.num_wires = num_wires
                
            def forward(self, x, *args, **kwargs):
                self.device.reset_states(batchsize=self.batch_size)
                for i in range(self.num_wires):
                    rx(self.device, wires=i, params=x[i])
                self.layer(q_machine = self.device)
                y = MeasureAll(obs={'Z0': 1})(self.device)
                return y

        x = pyvqnet.tensor.QTensor([[1,0,0,1],[1,1,0,1]],dtype=pyvqnet.kfloat32,requires_grad=True)
        model = QModel(4,pyvqnet.kcomplex64,2)
        y = model(x)
        print(y)

CY
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. py:class:: pyvqnet.qnn.vqc.tn.CY(has_params: bool = False,trainable: bool = False,init_params=None,wires=None,dtype=pyvqnet.kcomplex64,use_dagger=False)
    
    Definiert ein CY-Quantengatter.

    Diese Klasse erbt von ``pyvqnet.qnn.vqc.tn.QModule`` und ``torch.nn.Module``.
    Diese Klasse kann als Untermodul von ``torch.nn.Module`` zum torch-Modell hinzugefügt werden.

    :param has_params: ob es Parameter hat, wie RX, RY und andere Gatter auf True gesetzt werden müssen, und solche ohne Parameter auf False gesetzt werden müssen, der Standard ist False.
    :param trainable: ob es zu trainierende Parameter hat. Wenn die Schicht externe Eingabedaten zum Aufbau der Logikgatter-Matrix verwendet, auf False setzen. Wenn die zu trainierenden Parameter von dieser Schicht initialisiert werden müssen, ist es True, der Standard ist False.
    :param init_params: Initialisierungsparameter zum Codieren klassischer Daten QTensor, der Standard ist None.
    :param wires: Bit-Index des Leitungseffekts, der Standard ist None.
    :param dtype: Die Datenpräzision der internen Matrix des Logikgatters kann auf pyvqnet.kcomplex64 oder pyvqnet.kcomplex128 gesetzt werden, entsprechend float- bzw. double-Eingabe.
    :param use_dagger: ob die transponiert-konjugierte Version des Gatters verwendet werden soll, der Standard ist False.
    :return: eine ``pyvqnet.qnn.vqc.tn.QModule``-Instanz

    Example::

        from pyvqnet.qnn.vqc.tn import CY,TNQMachine,TNQModule,MeasureAll, rx
        import pyvqnet
        pyvqnet.backends.set_backend("torch")

        class QModel(TNQModule):
            
            def __init__(self, num_wires, dtype,batch_size=2):
                super(QModel, self).__init__()
                self.device = TNQMachine(num_wires)
                self.layer = CY(wires=[0,1])
                self.batch_size = batch_size
                self.num_wires = num_wires
                
            def forward(self, x, *args, **kwargs):
                self.device.reset_states(batchsize=self.batch_size)
                for i in range(self.num_wires):
                    rx(self.device, wires=i, params=x[i])
                self.layer(q_machine = self.device)
                y = MeasureAll(obs={'Z0': 1})(self.device)
                return y

        x = pyvqnet.tensor.QTensor([[1,0,0,1],[1,1,0,1]],dtype=pyvqnet.kfloat32,requires_grad=True)
        model = QModel(4,pyvqnet.kcomplex64,2)
        y = model(x)
        print(y)


CZ
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. py:class:: pyvqnet.qnn.vqc.tn.CZ(has_params: bool = False,trainable: bool = False,init_params=None,wires=None,dtype=pyvqnet.kcomplex64,use_dagger=False)
    
    Definiert ein CZ-Quantengatter.

    Diese Klasse erbt von ``pyvqnet.qnn.vqc.tn.QModule`` und ``torch.nn.Module``.
    Diese Klasse kann als Untermodul von ``torch.nn.Module`` zum torch-Modell hinzugefügt werden.

    :param has_params: ob es Parameter hat, wie RX, RY und andere Gatter auf True gesetzt werden müssen, und solche ohne Parameter auf False gesetzt werden müssen, der Standard ist False.
    :param trainable: ob es zu trainierende Parameter hat. Wenn die Schicht externe Eingabedaten zum Aufbau der Logikgatter-Matrix verwendet, auf False setzen. Wenn die zu trainierenden Parameter von dieser Schicht initialisiert werden müssen, ist es True, der Standard ist False.
    :param init_params: Initialisierungsparameter zum Codieren klassischer Daten QTensor, der Standard ist None.
    :param wires: Bit-Index des Leitungseffekts, der Standard ist None.
    :param dtype: Die Datenpräzision der internen Matrix des Logikgatters kann auf pyvqnet.kcomplex64 oder pyvqnet.kcomplex128 gesetzt werden, entsprechend float- bzw. double-Eingabe.
    :param use_dagger: ob die transponiert-konjugierte Version des Gatters verwendet werden soll, der Standard ist False.
    :return: eine ``pyvqnet.qnn.vqc.tn.QModule``-Instanz

    Example::

        from pyvqnet.qnn.vqc.tn import CZ,TNQMachine,TNQModule,MeasureAll, rx
        import pyvqnet
        pyvqnet.backends.set_backend("torch")

        class QModel(TNQModule):
            
            def __init__(self, num_wires, dtype,batch_size=2):
                super(QModel, self).__init__()
                self.device = TNQMachine(num_wires)
                self.layer = CZ(wires=[0,1])
                self.batch_size = batch_size
                self.num_wires = num_wires
                
            def forward(self, x, *args, **kwargs):
                self.device.reset_states(batchsize=self.batch_size)
                for i in range(self.num_wires):
                    rx(self.device, wires=i, params=x[i])
                self.layer(q_machine = self.device)
                y = MeasureAll(obs={'Z0': 1})(self.device)
                return y

        x = pyvqnet.tensor.QTensor([[1,0,0,1],[1,1,0,1]],dtype=pyvqnet.kfloat32,requires_grad=True)
        model = QModel(4,pyvqnet.kcomplex64,2)
        y = model(x)
        print(y)





CR
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. py:class:: pyvqnet.qnn.vqc.tn.CR(has_params: bool = False,trainable: bool = False,init_params=None,wires=None,dtype=pyvqnet.kcomplex64,use_dagger=False)
    
    Definiert ein CR-Quantengatter.

    Diese Klasse erbt von ``pyvqnet.qnn.vqc.tn.QModule`` und ``torch.nn.Module``.
    Diese Klasse kann als Untermodul von ``torch.nn.Module`` zum torch-Modell hinzugefügt werden.

    :param has_params: ob es Parameter hat, wie RX, RY und andere Gatter auf True gesetzt werden müssen, und solche ohne Parameter auf False gesetzt werden müssen, der Standard ist False.
    :param trainable: ob es zu trainierende Parameter hat. Wenn die Schicht externe Eingabedaten zum Aufbau der Logikgatter-Matrix verwendet, auf False setzen. Wenn die zu trainierenden Parameter von dieser Schicht initialisiert werden müssen, ist es True, der Standard ist False.
    :param init_params: Initialisierungsparameter zum Codieren klassischer Daten QTensor, der Standard ist None.
    :param wires: Bit-Index des Leitungseffekts, der Standard ist None.
    :param dtype: Die Datenpräzision der internen Matrix des Logikgatters kann auf pyvqnet.kcomplex64 oder pyvqnet.kcomplex128 gesetzt werden, entsprechend float- bzw. double-Eingabe.
    :param use_dagger: ob die transponiert-konjugierte Version des Gatters verwendet werden soll, der Standard ist False.
    :return: eine ``pyvqnet.qnn.vqc.tn.QModule``-Instanz

    Example::

        from pyvqnet.qnn.vqc.tn import CR,TNQMachine,TNQModule,MeasureAll, rx
        import pyvqnet
        pyvqnet.backends.set_backend("torch")

        class QModel(TNQModule):
            
            def __init__(self, num_wires, dtype,batch_size=2):
                super(QModel, self).__init__()
                self.device = TNQMachine(num_wires)
                self.layer = CR(has_params= True, trainable= True, wires=[0,2])
                self.batch_size = batch_size
                self.num_wires = num_wires
                
            def forward(self, x, *args, **kwargs):
                self.device.reset_states(batchsize=self.batch_size)
                for i in range(self.num_wires):
                    rx(self.device, wires=i, params=x[i])
                self.layer(q_machine = self.device)
                y = MeasureAll(obs={'Z0': 1})(self.device)
                return y

        x = pyvqnet.tensor.QTensor([[1,0,0,1],[1,1,0,1]],dtype=pyvqnet.kfloat32,requires_grad=True)
        model = QModel(4,pyvqnet.kcomplex64,2)
        y = model(x)
        print(y)



SWAP
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^


.. py:class:: pyvqnet.qnn.vqc.tn.SWAP(has_params: bool = False,trainable: bool = False,init_params=None,wires=None,dtype=pyvqnet.kcomplex64,use_dagger=False)
    
    Definiert ein SWAP-Quantengatter.

    Diese Klasse erbt von ``pyvqnet.qnn.vqc.tn.QModule`` und ``torch.nn.Module``.
    Diese Klasse kann als Untermodul von ``torch.nn.Module`` zum torch-Modell hinzugefügt werden.

    :param has_params: ob es Parameter hat, wie RX, RY und andere Gatter auf True gesetzt werden müssen, und solche ohne Parameter auf False gesetzt werden müssen, der Standard ist False.
    :param trainable: ob es zu trainierende Parameter hat. Wenn die Schicht externe Eingabedaten zum Aufbau der Logikgatter-Matrix verwendet, auf False setzen. Wenn die zu trainierenden Parameter von dieser Schicht initialisiert werden müssen, ist es True, der Standard ist False.
    :param init_params: Initialisierungsparameter zum Codieren klassischer Daten QTensor, der Standard ist None.
    :param wires: Bit-Index des Leitungseffekts, der Standard ist None.
    :param dtype: Die Datenpräzision der internen Matrix des Logikgatters kann auf pyvqnet.kcomplex64 oder pyvqnet.kcomplex128 gesetzt werden, entsprechend float- bzw. double-Eingabe.
    :param use_dagger: ob die transponiert-konjugierte Version des Gatters verwendet werden soll, der Standard ist False.
    :return: eine ``pyvqnet.qnn.vqc.tn.QModule``-Instanz

    Example::

        from pyvqnet.qnn.vqc.tn import SWAP,TNQMachine,TNQModule,MeasureAll, rx
        import pyvqnet
        pyvqnet.backends.set_backend("torch")

        class QModel(TNQModule):
            
            def __init__(self, num_wires, dtype,batch_size=2):
                super(QModel, self).__init__()
                self.device = TNQMachine(num_wires)
                self.layer = SWAP(wires=[0,1])
                self.batch_size = batch_size
                self.num_wires = num_wires
                
            def forward(self, x, *args, **kwargs):
                self.device.reset_states(batchsize=self.batch_size)
                for i in range(self.num_wires):
                    rx(self.device, wires=i, params=x[i])
                self.layer(q_machine = self.device)
                y = MeasureAll(obs={'Z0': 1})(self.device)
                return y

        x = pyvqnet.tensor.QTensor([[1,0,0,1],[1,1,0,1]],dtype=pyvqnet.kfloat32,requires_grad=True)
        model = QModel(4,pyvqnet.kcomplex64,2)
        y = model(x)
        print(y)


CSWAP
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. py:class:: pyvqnet.qnn.vqc.tn.CSWAP(has_params: bool = False,trainable: bool = False,init_params=None,wires=None,dtype=pyvqnet.kcomplex64,use_dagger=False)
    
    Definiert ein SWAP-Quantengatter.

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

    Diese Klasse erbt von ``pyvqnet.qnn.vqc.tn.QModule`` und ``torch.nn.Module``.
    Diese Klasse kann als Untermodul von ``torch.nn.Module`` zum torch-Modell hinzugefügt werden.

    :param has_params: ob es Parameter hat, wie RX, RY und andere Gatter auf True gesetzt werden müssen, und solche ohne Parameter auf False gesetzt werden müssen, der Standard ist False.
    :param trainable: ob es zu trainierende Parameter hat. Wenn die Schicht externe Eingabedaten zum Aufbau der Logikgatter-Matrix verwendet, auf False setzen. Wenn die zu trainierenden Parameter von dieser Schicht initialisiert werden müssen, ist es True, der Standard ist False.
    :param init_params: Initialisierungsparameter zum Codieren klassischer Daten QTensor, der Standard ist None.
    :param wires: Bit-Index des Leitungseffekts, der Standard ist None.
    :param dtype: Die Datenpräzision der internen Matrix des Logikgatters kann auf pyvqnet.kcomplex64 oder pyvqnet.kcomplex128 gesetzt werden, entsprechend float- bzw. double-Eingabe.
    :param use_dagger: ob die transponiert-konjugierte Version des Gatters verwendet werden soll, der Standard ist False.
    :return: eine ``pyvqnet.qnn.vqc.tn.QModule``-Instanz

    Example::

        from pyvqnet.qnn.vqc.tn import CSWAP,TNQMachine,TNQModule,MeasureAll, rx
        import pyvqnet
        pyvqnet.backends.set_backend("torch")

        class QModel(TNQModule):
            
            def __init__(self, num_wires, dtype,batch_size=2):
                super(QModel, self).__init__()
                self.device = TNQMachine(num_wires)
                self.layer = CSWAP(wires=[0,1,2])
                self.batch_size = batch_size
                self.num_wires = num_wires
                
            def forward(self, x, *args, **kwargs):
                self.device.reset_states(batchsize=self.batch_size)
                for i in range(self.num_wires):
                    rx(self.device, wires=i, params=x[i])
                self.layer(q_machine = self.device)
                y = MeasureAll(obs={'Z0': 1})(self.device)
                return y

        x = pyvqnet.tensor.QTensor([[1,0,0,1],[1,1,0,1]],dtype=pyvqnet.kfloat32,requires_grad=True)
        model = QModel(4,pyvqnet.kcomplex64,2)
        y = model(x)
        print(y)

RXX
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^


.. py:class:: pyvqnet.qnn.vqc.tn.RXX(has_params: bool = False,trainable: bool = False,init_params=None,wires=None,dtype=pyvqnet.kcomplex64,use_dagger=False)
    
    Definiert ein RXX-Quantengatter.

    Diese Klasse erbt von ``pyvqnet.qnn.vqc.tn.QModule`` und ``torch.nn.Module``.
    Diese Klasse kann als Untermodul von ``torch.nn.Module`` zum torch-Modell hinzugefügt werden.

    :param has_params: ob es Parameter hat, wie RX, RY und andere Gatter auf True gesetzt werden müssen, und solche ohne Parameter auf False gesetzt werden müssen, der Standard ist False.
    :param trainable: ob es zu trainierende Parameter hat. Wenn die Schicht externe Eingabedaten zum Aufbau der Logikgatter-Matrix verwendet, auf False setzen. Wenn die zu trainierenden Parameter von dieser Schicht initialisiert werden müssen, ist es True, der Standard ist False.
    :param init_params: Initialisierungsparameter zum Codieren klassischer Daten QTensor, der Standard ist None.
    :param wires: Bit-Index des Leitungseffekts, der Standard ist None.
    :param dtype: Die Datenpräzision der internen Matrix des Logikgatters kann auf pyvqnet.kcomplex64 oder pyvqnet.kcomplex128 gesetzt werden, entsprechend float- bzw. double-Eingabe.
    :param use_dagger: ob die transponiert-konjugierte Version des Gatters verwendet werden soll, der Standard ist False.
    :return: eine ``pyvqnet.qnn.vqc.tn.QModule``-Instanz

    Example::

        from pyvqnet.qnn.vqc.tn import RXX,TNQMachine,TNQModule,MeasureAll, rx
        import pyvqnet
        pyvqnet.backends.set_backend("torch")

        class QModel(TNQModule):
            
            def __init__(self, num_wires, dtype,batch_size=2):
                super(QModel, self).__init__()
                self.device = TNQMachine(num_wires)
                self.layer = RXX(has_params= True, trainable= True, wires=[0,2])
                self.batch_size = batch_size
                self.num_wires = num_wires
                
            def forward(self, x, *args, **kwargs):
                self.device.reset_states(batchsize=self.batch_size)
                for i in range(self.num_wires):
                    rx(self.device, wires=i, params=x[i])
                self.layer(q_machine = self.device)
                y = MeasureAll(obs={'Z0': 1})(self.device)
                return y

        x = pyvqnet.tensor.QTensor([[1,0,0,1],[1,1,0,1]],dtype=pyvqnet.kfloat32,requires_grad=True)
        model = QModel(4,pyvqnet.kcomplex64,2)
        y = model(x)
        print(y)

RYY
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. py:class:: pyvqnet.qnn.vqc.tn.RYY(has_params: bool = False,trainable: bool = False,init_params=None,wires=None,dtype=pyvqnet.kcomplex64,use_dagger=False)
    
    Definiert ein RYY-Quantengatter.

    Diese Klasse erbt von ``pyvqnet.qnn.vqc.tn.QModule`` und ``torch.nn.Module``.
    Diese Klasse kann als Untermodul von ``torch.nn.Module`` zum torch-Modell hinzugefügt werden.

    :param has_params: ob es Parameter hat, wie RX, RY und andere Gatter auf True gesetzt werden müssen, und solche ohne Parameter auf False gesetzt werden müssen, der Standard ist False.
    :param trainable: ob es zu trainierende Parameter hat. Wenn die Schicht externe Eingabedaten zum Aufbau der Logikgatter-Matrix verwendet, auf False setzen. Wenn die zu trainierenden Parameter von dieser Schicht initialisiert werden müssen, ist es True, der Standard ist False.
    :param init_params: Initialisierungsparameter zum Codieren klassischer Daten QTensor, der Standard ist None.
    :param wires: Bit-Index des Leitungseffekts, der Standard ist None.
    :param dtype: Die Datenpräzision der internen Matrix des Logikgatters kann auf pyvqnet.kcomplex64 oder pyvqnet.kcomplex128 gesetzt werden, entsprechend float- bzw. double-Eingabe.
    :param use_dagger: ob die transponiert-konjugierte Version des Gatters verwendet werden soll, der Standard ist False.
    :return: eine ``pyvqnet.qnn.vqc.tn.QModule``-Instanz

    Example::

        from pyvqnet.qnn.vqc.tn import RYY,TNQMachine,TNQModule,MeasureAll, rx
        import pyvqnet
        pyvqnet.backends.set_backend("torch")

        class QModel(TNQModule):
            
            def __init__(self, num_wires, dtype,batch_size=2):
                super(QModel, self).__init__()
                self.device = TNQMachine(num_wires)
                self.layer = RYY(has_params= True, trainable= True, wires=[0,2])
                self.batch_size = batch_size
                self.num_wires = num_wires
                
            def forward(self, x, *args, **kwargs):
                self.device.reset_states(batchsize=self.batch_size)
                for i in range(self.num_wires):
                    rx(self.device, wires=i, params=x[i])
                self.layer(q_machine = self.device)
                y = MeasureAll(obs={'Z0': 1})(self.device)
                return y

        x = pyvqnet.tensor.QTensor([[1,0,0,1],[1,1,0,1]],dtype=pyvqnet.kfloat32,requires_grad=True)
        model = QModel(4,pyvqnet.kcomplex64,2)
        y = model(x)
        print(y)


RZZ
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. py:class:: pyvqnet.qnn.vqc.tn.RZZ(has_params: bool = False,trainable: bool = False,init_params=None,wires=None,dtype=pyvqnet.kcomplex64,use_dagger=False)
    
    Definiert ein RZZ-Quantengatter.

    Diese Klasse erbt von ``pyvqnet.qnn.vqc.tn.QModule`` und ``torch.nn.Module``.
    Diese Klasse kann als Untermodul von ``torch.nn.Module`` zum torch-Modell hinzugefügt werden.

    :param has_params: ob es Parameter hat, wie RX, RY und andere Gatter auf True gesetzt werden müssen, und solche ohne Parameter auf False gesetzt werden müssen, der Standard ist False.
    :param trainable: ob es zu trainierende Parameter hat. Wenn die Schicht externe Eingabedaten zum Aufbau der Logikgatter-Matrix verwendet, auf False setzen. Wenn die zu trainierenden Parameter von dieser Schicht initialisiert werden müssen, ist es True, der Standard ist False.
    :param init_params: Initialisierungsparameter zum Codieren klassischer Daten QTensor, der Standard ist None.
    :param wires: Bit-Index des Leitungseffekts, der Standard ist None.
    :param dtype: Die Datenpräzision der internen Matrix des Logikgatters kann auf pyvqnet.kcomplex64 oder pyvqnet.kcomplex128 gesetzt werden, entsprechend float- bzw. double-Eingabe.
    :param use_dagger: ob die transponiert-konjugierte Version des Gatters verwendet werden soll, der Standard ist False.
    :return: eine ``pyvqnet.qnn.vqc.tn.QModule``-Instanz

    Example::

        from pyvqnet.qnn.vqc.tn import RZZ,TNQMachine,TNQModule,MeasureAll, rx
        import pyvqnet
        pyvqnet.backends.set_backend("torch")

        class QModel(TNQModule):
            
            def __init__(self, num_wires, dtype,batch_size=2):
                super(QModel, self).__init__()
                self.device = TNQMachine(num_wires)
                self.layer = RZZ(has_params= True, trainable= True, wires=[0,2])
                self.batch_size = batch_size
                self.num_wires = num_wires
                
            def forward(self, x, *args, **kwargs):
                self.device.reset_states(batchsize=self.batch_size)
                for i in range(self.num_wires):
                    rx(self.device, wires=i, params=x[i])
                self.layer(q_machine = self.device)
                y = MeasureAll(obs={'Z0': 1})(self.device)
                return y

        x = pyvqnet.tensor.QTensor([[1,0,0,1],[1,1,0,1]],dtype=pyvqnet.kfloat32,requires_grad=True)
        model = QModel(4,pyvqnet.kcomplex64,2)
        y = model(x)
        print(y)



RZX
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. py:class:: pyvqnet.qnn.vqc.tn.RZX(has_params: bool = False,trainable: bool = False,init_params=None,wires=None,dtype=pyvqnet.kcomplex64,use_dagger=False)
    
    Definiert ein RZX-Quantengatter.

    Diese Klasse erbt von ``pyvqnet.qnn.vqc.tn.QModule`` und ``torch.nn.Module``.
    Diese Klasse kann als Untermodul von ``torch.nn.Module`` zum torch-Modell hinzugefügt werden.

    :param has_params: ob es Parameter hat, wie RX, RY und andere Gatter auf True gesetzt werden müssen, und solche ohne Parameter auf False gesetzt werden müssen, der Standard ist False.
    :param trainable: ob es zu trainierende Parameter hat. Wenn die Schicht externe Eingabedaten zum Aufbau der Logikgatter-Matrix verwendet, auf False setzen. Wenn die zu trainierenden Parameter von dieser Schicht initialisiert werden müssen, ist es True, der Standard ist False.
    :param init_params: Initialisierungsparameter zum Codieren klassischer Daten QTensor, der Standard ist None.
    :param wires: Bit-Index des Leitungseffekts, der Standard ist None.
    :param dtype: Die Datenpräzision der internen Matrix des Logikgatters kann auf pyvqnet.kcomplex64 oder pyvqnet.kcomplex128 gesetzt werden, entsprechend float- bzw. double-Eingabe.
    :param use_dagger: ob die transponiert-konjugierte Version des Gatters verwendet werden soll, der Standard ist False.
    :return: eine ``pyvqnet.qnn.vqc.tn.QModule``-Instanz

    Example::

        from pyvqnet.qnn.vqc.tn import RZX,TNQMachine,TNQModule,MeasureAll, rx
        import pyvqnet
        pyvqnet.backends.set_backend("torch")

        class QModel(TNQModule):
            
            def __init__(self, num_wires, dtype,batch_size=2):
                super(QModel, self).__init__()
                self.device = TNQMachine(num_wires)
                self.layer = RZX(has_params= True, trainable= True, wires=[0,2])
                self.batch_size = batch_size
                self.num_wires = num_wires
                
            def forward(self, x, *args, **kwargs):
                self.device.reset_states(batchsize=self.batch_size)
                for i in range(self.num_wires):
                    rx(self.device, wires=i, params=x[i])
                self.layer(q_machine = self.device)
                y = MeasureAll(obs={'Z0': 1})(self.device)
                return y

        x = pyvqnet.tensor.QTensor([[1,0,0,1],[1,1,0,1]],dtype=pyvqnet.kfloat32,requires_grad=True)
        model = QModel(4,pyvqnet.kcomplex64,2)
        y = model(x)
        print(y)

Toffoli
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. py:class:: pyvqnet.qnn.vqc.tn.Toffoli(has_params: bool = False,trainable: bool = False,init_params=None,wires=None,dtype=pyvqnet.kcomplex64,use_dagger=False)
    
    Definiert ein Toffoli-Quantengatter.

    Diese Klasse erbt von ``pyvqnet.qnn.vqc.tn.QModule`` und ``torch.nn.Module``.
    Diese Klasse kann als Untermodul von ``torch.nn.Module`` zum torch-Modell hinzugefügt werden.

    :param has_params: ob es Parameter hat, wie RX, RY und andere Gatter auf True gesetzt werden müssen, und solche ohne Parameter auf False gesetzt werden müssen, der Standard ist False.
    :param trainable: ob es zu trainierende Parameter hat. Wenn die Schicht externe Eingabedaten zum Aufbau der Logikgatter-Matrix verwendet, auf False setzen. Wenn die zu trainierenden Parameter von dieser Schicht initialisiert werden müssen, ist es True, der Standard ist False.
    :param init_params: Initialisierungsparameter zum Codieren klassischer Daten QTensor, der Standard ist None.
    :param wires: Bit-Index des Leitungseffekts, der Standard ist None.
    :param dtype: Die Datenpräzision der internen Matrix des Logikgatters kann auf pyvqnet.kcomplex64 oder pyvqnet.kcomplex128 gesetzt werden, entsprechend float- bzw. double-Eingabe.
    :param use_dagger: ob die transponiert-konjugierte Version des Gatters verwendet werden soll, der Standard ist False.
    :return: eine ``pyvqnet.qnn.vqc.tn.QModule``-Instanz

    Example::

        from pyvqnet.qnn.vqc.tn import Toffoli,TNQMachine,TNQModule,MeasureAll, rx
        import pyvqnet
        pyvqnet.backends.set_backend("torch")

        class QModel(TNQModule):
            
            def __init__(self, num_wires, dtype,batch_size=2):
                super(QModel, self).__init__()
                self.device = TNQMachine(num_wires)
                self.layer = Toffoli(wires=[0,2,1])
                self.batch_size = batch_size
                self.num_wires = num_wires
                
            def forward(self, x, *args, **kwargs):
                self.device.reset_states(batchsize=self.batch_size)
                for i in range(self.num_wires):
                    rx(self.device, wires=i, params=x[i])
                self.layer(q_machine = self.device)
                y = MeasureAll(obs={'Z0': 1})(self.device)
                return y

        x = pyvqnet.tensor.QTensor([[1,0,0,1],[1,1,0,1]],dtype=pyvqnet.kfloat32,requires_grad=True)
        model = QModel(4,pyvqnet.kcomplex64,2)
        y = model(x)
        print(y)

IsingXX
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^


.. py:class:: pyvqnet.qnn.vqc.tn.IsingXX(has_params: bool = False,trainable: bool = False,init_params=None,wires=None,dtype=pyvqnet.kcomplex64,use_dagger=False)
    
    Definiert ein IsingXX-Quantengatter.

    Diese Klasse erbt von ``pyvqnet.qnn.vqc.tn.QModule`` und ``torch.nn.Module``.
    Diese Klasse kann als Untermodul von ``torch.nn.Module`` zum torch-Modell hinzugefügt werden.

    :param has_params: ob es Parameter hat, wie RX, RY und andere Gatter auf True gesetzt werden müssen, und solche ohne Parameter auf False gesetzt werden müssen, der Standard ist False.
    :param trainable: ob es zu trainierende Parameter hat. Wenn die Schicht externe Eingabedaten zum Aufbau der Logikgatter-Matrix verwendet, auf False setzen. Wenn die zu trainierenden Parameter von dieser Schicht initialisiert werden müssen, ist es True, der Standard ist False.
    :param init_params: Initialisierungsparameter zum Codieren klassischer Daten QTensor, der Standard ist None.
    :param wires: Bit-Index des Leitungseffekts, der Standard ist None.
    :param dtype: Die Datenpräzision der internen Matrix des Logikgatters kann auf pyvqnet.kcomplex64 oder pyvqnet.kcomplex128 gesetzt werden, entsprechend float- bzw. double-Eingabe.
    :param use_dagger: ob die transponiert-konjugierte Version des Gatters verwendet werden soll, der Standard ist False.
    :return: eine ``pyvqnet.qnn.vqc.tn.QModule``-Instanz

    Example::

        from pyvqnet.qnn.vqc.tn import IsingXX,TNQMachine,TNQModule,MeasureAll, rx
        import pyvqnet
        pyvqnet.backends.set_backend("torch")

        class QModel(TNQModule):
            
            def __init__(self, num_wires, dtype,batch_size=2):
                super(QModel, self).__init__()
                self.device = TNQMachine(num_wires)
                self.layer = IsingXX(has_params= True, trainable= True, wires=[0,2])
                self.batch_size = batch_size
                self.num_wires = num_wires
                
            def forward(self, x, *args, **kwargs):
                self.device.reset_states(batchsize=self.batch_size)
                for i in range(self.num_wires):
                    rx(self.device, wires=i, params=x[i])
                self.layer(q_machine = self.device)
                y = MeasureAll(obs={'Z0': 1})(self.device)
                return y

        x = pyvqnet.tensor.QTensor([[1,0,0,1],[1,1,0,1]],dtype=pyvqnet.kfloat32,requires_grad=True)
        model = QModel(4,pyvqnet.kcomplex64,2)
        y = model(x)
        print(y)


IsingYY
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^


.. py:class:: pyvqnet.qnn.vqc.tn.IsingYY(has_params: bool = False,trainable: bool = False,init_params=None,wires=None,dtype=pyvqnet.kcomplex64,use_dagger=False)
    
    Definiert ein IsingYY-Quantengatter.

    Diese Klasse erbt von ``pyvqnet.qnn.vqc.tn.QModule`` und ``torch.nn.Module``.
    Diese Klasse kann als Untermodul von ``torch.nn.Module`` zum torch-Modell hinzugefügt werden.

    :param has_params: ob es Parameter hat, wie RX, RY und andere Gatter auf True gesetzt werden müssen, und solche ohne Parameter auf False gesetzt werden müssen, der Standard ist False.
    :param trainable: ob es zu trainierende Parameter hat. Wenn die Schicht externe Eingabedaten zum Aufbau der Logikgatter-Matrix verwendet, auf False setzen. Wenn die zu trainierenden Parameter von dieser Schicht initialisiert werden müssen, ist es True, der Standard ist False.
    :param init_params: Initialisierungsparameter zum Codieren klassischer Daten QTensor, der Standard ist None.
    :param wires: Bit-Index des Leitungseffekts, der Standard ist None.
    :param dtype: Die Datenpräzision der internen Matrix des Logikgatters kann auf pyvqnet.kcomplex64 oder pyvqnet.kcomplex128 gesetzt werden, entsprechend float- bzw. double-Eingabe.
    :param use_dagger: ob die transponiert-konjugierte Version des Gatters verwendet werden soll, der Standard ist False.
    :return: eine ``pyvqnet.qnn.vqc.tn.QModule``-Instanz

    Example::

        from pyvqnet.qnn.vqc.tn import IsingYY,TNQMachine,TNQModule,MeasureAll, rx
        import pyvqnet
        pyvqnet.backends.set_backend("torch")

        class QModel(TNQModule):
            
            def __init__(self, num_wires, dtype,batch_size=2):
                super(QModel, self).__init__()
                self.device = TNQMachine(num_wires)
                self.layer = IsingYY(has_params= True, trainable= True, wires=[0,2])
                self.batch_size = batch_size
                self.num_wires = num_wires
                
            def forward(self, x, *args, **kwargs):
                self.device.reset_states(batchsize=self.batch_size)
                for i in range(self.num_wires):
                    rx(self.device, wires=i, params=x[i])
                self.layer(q_machine = self.device)
                y = MeasureAll(obs={'Z0': 1})(self.device)
                return y

        x = pyvqnet.tensor.QTensor([[1,0,0,1],[1,1,0,1]],dtype=pyvqnet.kfloat32,requires_grad=True)
        model = QModel(4,pyvqnet.kcomplex64,2)
        y = model(x)
        print(y)


IsingZZ
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. py:class:: pyvqnet.qnn.vqc.tn.IsingZZ(has_params: bool = False,trainable: bool = False,init_params=None,wires=None,dtype=pyvqnet.kcomplex64,use_dagger=False)
    
    Definiert ein IsingZZ-Quantengatter.

    Diese Klasse erbt von ``pyvqnet.qnn.vqc.tn.QModule`` und ``torch.nn.Module``.
    Diese Klasse kann als Untermodul von ``torch.nn.Module`` zum torch-Modell hinzugefügt werden.

    :param has_params: ob es Parameter hat, wie RX, RY und andere Gatter auf True gesetzt werden müssen, und solche ohne Parameter auf False gesetzt werden müssen, der Standard ist False.
    :param trainable: ob es zu trainierende Parameter hat. Wenn die Schicht externe Eingabedaten zum Aufbau der Logikgatter-Matrix verwendet, auf False setzen. Wenn die zu trainierenden Parameter von dieser Schicht initialisiert werden müssen, ist es True, der Standard ist False.
    :param init_params: Initialisierungsparameter zum Codieren klassischer Daten QTensor, der Standard ist None.
    :param wires: Bit-Index des Leitungseffekts, der Standard ist None.
    :param dtype: Die Datenpräzision der internen Matrix des Logikgatters kann auf pyvqnet.kcomplex64 oder pyvqnet.kcomplex128 gesetzt werden, entsprechend float- bzw. double-Eingabe.
    :param use_dagger: ob die transponiert-konjugierte Version des Gatters verwendet werden soll, der Standard ist False.
    :return: eine ``pyvqnet.qnn.vqc.tn.QModule``-Instanz

    Example::

        from pyvqnet.qnn.vqc.tn import IsingZZ,TNQMachine,TNQModule,MeasureAll, rx
        import pyvqnet
        pyvqnet.backends.set_backend("torch")

        class QModel(TNQModule):
            
            def __init__(self, num_wires, dtype,batch_size=2):
                super(QModel, self).__init__()
                self.device = TNQMachine(num_wires)
                self.layer = IsingZZ(has_params= True, trainable= True, wires=[0,2])
                self.batch_size = batch_size
                self.num_wires = num_wires
                
            def forward(self, x, *args, **kwargs):
                self.device.reset_states(batchsize=self.batch_size)
                for i in range(self.num_wires):
                    rx(self.device, wires=i, params=x[i])
                self.layer(q_machine = self.device)
                y = MeasureAll(obs={'Z0': 1})(self.device)
                return y

        x = pyvqnet.tensor.QTensor([[1,0,0,1],[1,1,0,1]],dtype=pyvqnet.kfloat32,requires_grad=True)
        model = QModel(4,pyvqnet.kcomplex64,2)
        y = model(x)
        print(y)


IsingXY
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^


.. py:class:: pyvqnet.qnn.vqc.tn.IsingXY(has_params: bool = False,trainable: bool = False,init_params=None,wires=None,dtype=pyvqnet.kcomplex64,use_dagger=False)
    
    Definiert ein IsingXY-Quantengatter.

    Diese Klasse erbt von ``pyvqnet.qnn.vqc.tn.QModule`` und ``torch.nn.Module``.
    Diese Klasse kann als Untermodul von ``torch.nn.Module`` zum torch-Modell hinzugefügt werden.

    :param has_params: ob es Parameter hat, wie RX, RY und andere Gatter auf True gesetzt werden müssen, und solche ohne Parameter auf False gesetzt werden müssen, der Standard ist False.
    :param trainable: ob es zu trainierende Parameter hat. Wenn die Schicht externe Eingabedaten zum Aufbau der Logikgatter-Matrix verwendet, auf False setzen. Wenn die zu trainierenden Parameter von dieser Schicht initialisiert werden müssen, ist es True, der Standard ist False.
    :param init_params: Initialisierungsparameter zum Codieren klassischer Daten QTensor, der Standard ist None.
    :param wires: Bit-Index des Leitungseffekts, der Standard ist None.
    :param dtype: Die Datenpräzision der internen Matrix des Logikgatters kann auf pyvqnet.kcomplex64 oder pyvqnet.kcomplex128 gesetzt werden, entsprechend float- bzw. double-Eingabe.
    :param use_dagger: ob die transponiert-konjugierte Version des Gatters verwendet werden soll, der Standard ist False.
    :return: eine ``pyvqnet.qnn.vqc.tn.QModule``-Instanz

    Example::

        from pyvqnet.qnn.vqc.tn import IsingXY,TNQMachine,TNQModule,MeasureAll, rx
        import pyvqnet
        pyvqnet.backends.set_backend("torch")

        class QModel(TNQModule):
            
            def __init__(self, num_wires, dtype,batch_size=2):
                super(QModel, self).__init__()
                self.device = TNQMachine(num_wires)
                self.layer = IsingXY(has_params= True, trainable= True, wires=[0,2])
                self.batch_size = batch_size
                self.num_wires = num_wires
                
            def forward(self, x, *args, **kwargs):
                self.device.reset_states(batchsize=self.batch_size)
                for i in range(self.num_wires):
                    rx(self.device, wires=i, params=x[i])
                self.layer(q_machine = self.device)
                y = MeasureAll(obs={'Z0': 1})(self.device)
                return y

        x = pyvqnet.tensor.QTensor([[1,0,0,1],[1,1,0,1]],dtype=pyvqnet.kfloat32,requires_grad=True)
        model = QModel(4,pyvqnet.kcomplex64,2)
        y = model(x)
        print(y)


PhaseShift
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^


.. py:class:: pyvqnet.qnn.vqc.tn.PhaseShift(has_params: bool = False,trainable: bool = False,init_params=None,wires=None,dtype=pyvqnet.kcomplex64,use_dagger=False)
    
    Definiert ein PhaseShift-Quantengatter.

    Diese Klasse erbt von ``pyvqnet.qnn.vqc.tn.QModule`` und ``torch.nn.Module``.
    Diese Klasse kann als Untermodul von ``torch.nn.Module`` zum torch-Modell hinzugefügt werden.

    :param has_params: ob es Parameter hat, wie RX, RY und andere Gatter auf True gesetzt werden müssen, und solche ohne Parameter auf False gesetzt werden müssen, der Standard ist False.
    :param trainable: ob es zu trainierende Parameter hat. Wenn die Schicht externe Eingabedaten zum Aufbau der Logikgatter-Matrix verwendet, auf False setzen. Wenn die zu trainierenden Parameter von dieser Schicht initialisiert werden müssen, ist es True, der Standard ist False.
    :param init_params: Initialisierungsparameter zum Codieren klassischer Daten QTensor, der Standard ist None.
    :param wires: Bit-Index des Leitungseffekts, der Standard ist None.
    :param dtype: Die Datenpräzision der internen Matrix des Logikgatters kann auf pyvqnet.kcomplex64 oder pyvqnet.kcomplex128 gesetzt werden, entsprechend float- bzw. double-Eingabe.
    :param use_dagger: ob die transponiert-konjugierte Version des Gatters verwendet werden soll, der Standard ist False.
    :return: eine ``pyvqnet.qnn.vqc.tn.QModule``-Instanz

    Example::

        from pyvqnet.qnn.vqc.tn import PhaseShift,TNQMachine,TNQModule,MeasureAll, rx
        import pyvqnet
        pyvqnet.backends.set_backend("torch")

        class QModel(TNQModule):
            
            def __init__(self, num_wires, dtype,batch_size=2):
                super(QModel, self).__init__()
                self.device = TNQMachine(num_wires)
                self.layer = PhaseShift(has_params= True, trainable= True, wires=1)
                self.batch_size = batch_size
                self.num_wires = num_wires
                
            def forward(self, x, *args, **kwargs):
                self.device.reset_states(batchsize=self.batch_size)
                for i in range(self.num_wires):
                    rx(self.device, wires=i, params=x[i])
                self.layer(q_machine = self.device)
                y = MeasureAll(obs={'Z0': 1})(self.device)
                return y

        x = pyvqnet.tensor.QTensor([[1,0,0,1],[1,1,0,1]],dtype=pyvqnet.kfloat32,requires_grad=True)
        model = QModel(4,pyvqnet.kcomplex64,2)
        y = model(x)
        print(y)


MultiRZ
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. py:class:: pyvqnet.qnn.vqc.tn.MultiRZ(has_params: bool = False,trainable: bool = False,init_params=None,wires=None,dtype=pyvqnet.kcomplex64,use_dagger=False)
    
    Definiert ein MultiRZ-Quantengatter.

    Diese Klasse erbt von ``pyvqnet.qnn.vqc.tn.QModule`` und ``torch.nn.Module``.
    Diese Klasse kann als Untermodul von ``torch.nn.Module`` zum torch-Modell hinzugefügt werden.

    :param has_params: ob es Parameter hat, wie RX, RY und andere Gatter auf True gesetzt werden müssen, und solche ohne Parameter auf False gesetzt werden müssen, der Standard ist False.
    :param trainable: ob es zu trainierende Parameter hat. Wenn die Schicht externe Eingabedaten zum Aufbau der Logikgatter-Matrix verwendet, auf False setzen. Wenn die zu trainierenden Parameter von dieser Schicht initialisiert werden müssen, ist es True, der Standard ist False.
    :param init_params: Initialisierungsparameter zum Codieren klassischer Daten QTensor, der Standard ist None.
    :param wires: Bit-Index des Leitungseffekts, der Standard ist None.
    :param dtype: Die Datenpräzision der internen Matrix des Logikgatters kann auf pyvqnet.kcomplex64 oder pyvqnet.kcomplex128 gesetzt werden, entsprechend float- bzw. double-Eingabe.
    :param use_dagger: ob die transponiert-konjugierte Version des Gatters verwendet werden soll, der Standard ist False.
    :return: eine ``pyvqnet.qnn.vqc.tn.QModule``-Instanz

    Example::

        from pyvqnet.qnn.vqc.tn import MultiRZ,TNQMachine,TNQModule,MeasureAll, rx
        import pyvqnet
        pyvqnet.backends.set_backend("torch")

        class QModel(TNQModule):
            
            def __init__(self, num_wires, dtype,batch_size=2):
                super(QModel, self).__init__()
                self.device = TNQMachine(num_wires)
                self.layer = MultiRZ(has_params= True, trainable= True, wires=[0,2])
                self.batch_size = batch_size
                self.num_wires = num_wires
                
            def forward(self, x, *args, **kwargs):
                self.device.reset_states(batchsize=self.batch_size)
                for i in range(self.num_wires):
                    rx(self.device, wires=i, params=x[i])
                self.layer(q_machine = self.device)
                y = MeasureAll(obs={'Z0': 1})(self.device)
                return y

        x = pyvqnet.tensor.QTensor([[1,0,0,1],[1,1,0,1]],dtype=pyvqnet.kfloat32,requires_grad=True)
        model = QModel(4,pyvqnet.kcomplex64,2)
        y = model(x)
        print(y)



SDG
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^


.. py:class:: pyvqnet.qnn.vqc.tn.SDG(has_params: bool = False,trainable: bool = False,init_params=None,wires=None,dtype=pyvqnet.kcomplex64,use_dagger=False)
    
    Definiert ein SDG-Quantengatter.

    Diese Klasse erbt von ``pyvqnet.qnn.vqc.tn.QModule`` und ``torch.nn.Module``.
    Diese Klasse kann als Untermodul von ``torch.nn.Module`` zum torch-Modell hinzugefügt werden.

    :param has_params: ob es Parameter hat, wie RX, RY und andere Gatter auf True gesetzt werden müssen, und solche ohne Parameter auf False gesetzt werden müssen, der Standard ist False.
    :param trainable: ob es zu trainierende Parameter hat. Wenn die Schicht externe Eingabedaten zum Aufbau der Logikgatter-Matrix verwendet, auf False setzen. Wenn die zu trainierenden Parameter von dieser Schicht initialisiert werden müssen, ist es True, der Standard ist False.
    :param init_params: Initialisierungsparameter zum Codieren klassischer Daten QTensor, der Standard ist None.
    :param wires: Bit-Index des Leitungseffekts, der Standard ist None.
    :param dtype: Die Datenpräzision der internen Matrix des Logikgatters kann auf pyvqnet.kcomplex64 oder pyvqnet.kcomplex128 gesetzt werden, entsprechend float- bzw. double-Eingabe.
    :param use_dagger: ob die transponiert-konjugierte Version des Gatters verwendet werden soll, der Standard ist False.
    :return: eine ``pyvqnet.qnn.vqc.tn.QModule``-Instanz

    Example::
        
        from pyvqnet.qnn.vqc.tn import SDG,TNQMachine,TNQModule,MeasureAll, rx
        import pyvqnet
        pyvqnet.backends.set_backend("torch")

        class QModel(TNQModule):
            
            def __init__(self, num_wires, dtype,batch_size=2):
                super(QModel, self).__init__()
                self.device = TNQMachine(num_wires)
                self.layer = SDG(wires=0)
                self.batch_size = batch_size
                self.num_wires = num_wires
                
            def forward(self, x, *args, **kwargs):
                self.device.reset_states(batchsize=self.batch_size)
                for i in range(self.num_wires):
                    rx(self.device, wires=i, params=x[i])
                self.layer(q_machine = self.device)
                y = MeasureAll(obs={'Z0': 1})(self.device)
                return y

        x = pyvqnet.tensor.QTensor([[1,0,0,1],[1,1,0,1]],dtype=pyvqnet.kfloat32,requires_grad=True)
        model = QModel(4,pyvqnet.kcomplex64,2)
        y = model(x)
        print(y)




TDG
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. py:class:: pyvqnet.qnn.vqc.tn.TDG(has_params: bool = False,trainable: bool = False,init_params=None,wires=None,dtype=pyvqnet.kcomplex64,use_dagger=False)
    
    Definiert ein SDG-Quantengatter.

    Diese Klasse erbt von ``pyvqnet.qnn.vqc.tn.QModule`` und ``torch.nn.Module``.
    Diese Klasse kann als Untermodul von ``torch.nn.Module`` zum torch-Modell hinzugefügt werden.

    :param has_params: ob es Parameter hat, wie RX, RY und andere Gatter auf True gesetzt werden müssen, und solche ohne Parameter auf False gesetzt werden müssen, der Standard ist False.
    :param trainable: ob es zu trainierende Parameter hat. Wenn die Schicht externe Eingabedaten zum Aufbau der Logikgatter-Matrix verwendet, auf False setzen. Wenn die zu trainierenden Parameter von dieser Schicht initialisiert werden müssen, ist es True, der Standard ist False.
    :param init_params: Initialisierungsparameter zum Codieren klassischer Daten QTensor, der Standard ist None.
    :param wires: Bit-Index des Leitungseffekts, der Standard ist None.
    :param dtype: Die Datenpräzision der internen Matrix des Logikgatters kann auf pyvqnet.kcomplex64 oder pyvqnet.kcomplex128 gesetzt werden, entsprechend float- bzw. double-Eingabe.
    :param use_dagger: ob die transponiert-konjugierte Version des Gatters verwendet werden soll, der Standard ist False.
    :return: eine ``pyvqnet.qnn.vqc.tn.QModule``-Instanz

    Example::
        
        from pyvqnet.qnn.vqc.tn import TDG,TNQMachine,TNQModule,MeasureAll, rx
        import pyvqnet
        pyvqnet.backends.set_backend("torch")

        class QModel(TNQModule):
            
            def __init__(self, num_wires, dtype,batch_size=2):
                super(QModel, self).__init__()
                self.device = TNQMachine(num_wires)
                self.layer = TDG(wires=0)
                self.batch_size = batch_size
                self.num_wires = num_wires
                
            def forward(self, x, *args, **kwargs):
                self.device.reset_states(batchsize=self.batch_size)
                for i in range(self.num_wires):
                    rx(self.device, wires=i, params=x[i])
                self.layer(q_machine = self.device)
                y = MeasureAll(obs={'Z0': 1})(self.device)
                return y

        x = pyvqnet.tensor.QTensor([[1,0,0,1],[1,1,0,1]],dtype=pyvqnet.kfloat32,requires_grad=True)
        model = QModel(4,pyvqnet.kcomplex64,2)
        y = model(x)
        print(y)



ControlledPhaseShift
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^


.. py:class:: pyvqnet.qnn.vqc.tn.ControlledPhaseShift(has_params: bool = False,trainable: bool = False,init_params=None,wires=None,dtype=pyvqnet.kcomplex64,use_dagger=False)
    
    Definiert ein ControlledPhaseShift-Quantengatter.

    Diese Klasse erbt von ``pyvqnet.qnn.vqc.tn.QModule`` und ``torch.nn.Module``.
    Diese Klasse kann als Untermodul von ``torch.nn.Module`` zum torch-Modell hinzugefügt werden.

    :param has_params: ob es Parameter hat, wie RX, RY und andere Gatter auf True gesetzt werden müssen, und solche ohne Parameter auf False gesetzt werden müssen, der Standard ist False.
    :param trainable: ob es zu trainierende Parameter hat. Wenn die Schicht externe Eingabedaten zum Aufbau der Logikgatter-Matrix verwendet, auf False setzen. Wenn die zu trainierenden Parameter von dieser Schicht initialisiert werden müssen, ist es True, der Standard ist False.
    :param init_params: Initialisierungsparameter zum Codieren klassischer Daten QTensor, der Standard ist None.
    :param wires: Bit-Index des Leitungseffekts, der Standard ist None.
    :param dtype: Die Datenpräzision der internen Matrix des Logikgatters kann auf pyvqnet.kcomplex64 oder pyvqnet.kcomplex128 gesetzt werden, entsprechend float- bzw. double-Eingabe.
    :param use_dagger: ob die transponiert-konjugierte Version des Gatters verwendet werden soll, der Standard ist False.
    :return: eine ``pyvqnet.qnn.vqc.tn.QModule``-Instanz

    Example::

        from pyvqnet.qnn.vqc.tn import ControlledPhaseShift,TNQMachine,TNQModule,MeasureAll, rx
        import pyvqnet
        pyvqnet.backends.set_backend("torch")

        class QModel(TNQModule):
            
            def __init__(self, num_wires, dtype,batch_size=2):
                super(QModel, self).__init__()
                self.device = TNQMachine(num_wires)
                self.layer = ControlledPhaseShift(has_params= True, trainable= True, wires=[0,2])
                self.batch_size = batch_size
                self.num_wires = num_wires
                
            def forward(self, x, *args, **kwargs):
                self.device.reset_states(batchsize=self.batch_size)
                for i in range(self.num_wires):
                    rx(self.device, wires=i, params=x[i])
                self.layer(q_machine = self.device)
                y = MeasureAll(obs={'Z0': 1})(self.device)
                return y

        x = pyvqnet.tensor.QTensor([[1,0,0,1],[1,1,0,1]],dtype=pyvqnet.kfloat32,requires_grad=True)
        model = QModel(4,pyvqnet.kcomplex64,2)
        y = model(x)
        print(y)


Messungs-API
------------------------------

VQC_Purity
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^


.. py:function:: pyvqnet.qnn.vqc.tn.VQC_Purity(state, qubits_idx, num_wires, use_tn=False)

    Berechnet die Reinheit auf einem bestimmten Qubit ``qubits_idx`` aus dem Zustandsvektor ``state``.

    .. math::
        \gamma = \text{Tr}(\rho^2)

    where :math:`\rho` is a density matrix. The purity of a normalized quantum state satisfies :math:`\frac{1}{d} \leq \gamma \leq 1` ,
    where :math:`d` is the dimension of the Hilbert space.
    Die Reinheit des reinen Zustands ist 1.

    :param state: Quantum state obtained from TNQMachine.get_states()
    :param qubits_idx: Qubit index for which to calculate purity
    :param num_wires: Qubit idx
    :param use_tn: use tensornetwork need to be set True, default False

    :return: purity

    .. note::
        
        batch_size need TNQModule.

    Example::

        import pyvqnet
        from pyvqnet.qnn.vqc.tn import TNQMachine, qcircuit, TNQModule,VQC_Purity
        pyvqnet.backends.set_backend("torch")
        from pyvqnet.tensor import QTensor

        x = QTensor([[0.7, 0.4], [1.7, 2.4]], requires_grad=True).toGPU()

        class QM(TNQModule):
            def __init__(self, name=""):
                super().__init__(name)
                self.device = TNQMachine(3)
                
            def forward(self, x):
                self.device.reset_states(2)
                qcircuit.rx(q_machine=self.device, wires=0, params=x[0])
                qcircuit.ry(q_machine=self.device, wires=1, params=x[1])
                qcircuit.ry(q_machine=self.device, wires=2, params=x[1])
                qcircuit.cnot(q_machine=self.device, wires=[0, 1])
                qcircuit.cnot(q_machine=self.device, wires=[2, 1])
                return VQC_Purity(self.device.get_states(), [0, 1], num_wires=3, use_tn=True)

        model = QM().toGPU()
        y_tn = model(x)
        x.data.retain_grad()
        y_tn.backward()
        print(y_tn)

VQC_VarMeasure
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. py:function:: pyvqnet.qnn.vqc.tn.VQC_VarMeasure(q_machine, obs)

    Return the measurement variance of the provided observable ``obs`` in statevectors in ``q_machine`` .

    :param q_machine: Quantum state obtained from pyqpanda get_qstate()
    :param obs: observables

    :return: variance value

    Example::

        import pyvqnet
        from pyvqnet.qnn.vqc.tn import TNQMachine, qcircuit, VQC_VarMeasure, TNQModule,PauliY
        from pyvqnet.tensor import QTensor
        from pyvqnet import kfloat64
        pyvqnet.backends.set_backend("torch")
        x = QTensor([[0.7, 0.4], [0.6, 0.4]], requires_grad=True).toGPU()

        class QM(TNQModule):
            def __init__(self, name=""):
                super().__init__(name)
                self.device = TNQMachine(3)
                
            def forward(self, x):
                self.device.reset_states(2)
                qcircuit.rx(q_machine=self.device, wires=0, params=x[0])
                qcircuit.ry(q_machine=self.device, wires=1, params=x[1])
                qcircuit.ry(q_machine=self.device, wires=2, params=x[1])
                qcircuit.cnot(q_machine=self.device, wires=[0, 1])
                qcircuit.cnot(q_machine=self.device, wires=[2, 1])
                return VQC_VarMeasure(q_machine= self.device, obs=PauliY(wires=0))
            
        model = QM().toGPU()
        y = model(x)
        x.data.retain_grad()
        y.backward()
        print(y)

        # [[0.9370641],
        # [0.9516521]]


VQC_DensityMatrixFromQstate
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^


.. py:function:: pyvqnet.qnn.vqc.tn.VQC_DensityMatrixFromQstate(state, indices, use_tn=False)

    Berechnet die Dichtematrix von Quantenzuständen ``state`` über einen bestimmten Satz von Qubits ``indices``.

    :param state: A 1D list of state vectors. The size of this list should be ``(2**N,)`` For the number of qubits ``N``, qstate should start from 000 -> 111.
    :param indices: A list of qubit indices in the considered subsystem.
    :param use_tn: use tensornetwork need to be set True, default False.

    :return: A density matrix of size "(b, 2**len(indices), 2**len(indices))".

    Example::

        import pyvqnet
        from pyvqnet.tensor import QTensor
        from pyvqnet.qnn.vqc.tn import TNQMachine, qcircuit, VQC_DensityMatrixFromQstate,TNQModule
        pyvqnet.backends.set_backend("torch")
        x = QTensor([[0.7,0.4],[1.7,2.4]], requires_grad=True).toGPU()
        class QM(TNQModule):
            def __init__(self, name=""):
                super().__init__(name=name, use_jit=True)
                self.device = TNQMachine(3)
                
            def forward(self, x):
                self.device.reset_states(2)
                qcircuit.rx(q_machine=self.device, wires=0, params=x[0])
                qcircuit.ry(q_machine=self.device, wires=1, params=x[1])
                qcircuit.ry(q_machine=self.device, wires=2, params=x[1])
                qcircuit.cnot(q_machine=self.device, wires=[0, 1])
                qcircuit.cnot(q_machine=self.device, wires=[2, 1])
                return VQC_DensityMatrixFromQstate(self.device.get_states(),[0,1],use_tn=True)
            
        model = QM().toGPU()
        y = model(x)
        x.data.retain_grad()
        y.backward()
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
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^


.. py:class:: pyvqnet.qnn.vqc.tn.Probability(wires=None, name="")

    Berechnet das Wahrscheinlichkeitsmessungsergebnis des Quantenschaltkreises auf einem bestimmten Bit.

    Diese Klasse erbt von ``pyvqnet.qnn.vqc.tn.QModule`` und ``torch.nn.Module``.
    Diese Klasse kann als Untermodul von ``torch.nn.Module`` zum torch-Modell hinzugefügt werden.

    :param wires: Der Index des Messbits, Liste, Tupel oder Integer.
    :param name: Der Name des Moduls, Standard: "".
    :return: Das Messergebnis, QTensor.

    Example::

        import pyvqnet
        pyvqnet.backends.set_backend("torch")
        from pyvqnet.qnn.vqc.tn import Probability,rx,ry,cnot,TNQMachine,rz
        from pyvqnet.tensor import QTensor
        from pyvqnet import kfloat64
        x = QTensor([[0.56, 0.1],[0.56, 0.1]],requires_grad=True)
        qm = TNQMachine(4)
        qm.reset_states(2)
        rz(q_machine=qm,wires=0,params=x[:,[0]])
        rz(q_machine=qm,wires=1,params=x[:,[0]])
        cnot(q_machine=qm,wires=[0,1])
        ry(q_machine=qm,wires=2,params=x[:,[1]])
        cnot(q_machine=qm,wires=[0,2])
        rz(q_machine=qm,wires=3,params=x[:,[1]])
        ma = Probability(wires=1)
        y =ma(q_machine=qm)


MeasureAll
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. py:class:: pyvqnet.qnn.vqc.tn.MeasureAll(obs=None, name="")

    Berechnet die Messergebnisse von Quantenschaltkreisen, unterstützt Eingabe obs als mehrere oder einzelne Pauli-Operatoren oder Hamiltonians.
    
    For example:

    {\'X0\': 0.23} indicates a PauliX effect on qubit 0, with a coefficient of 0.23.

    {\'X1 Z2\': 2.4,\'Y2\': -0.5} corresponds to the observed value 2.4 * X1 @ Z2 - 0.5 * Y2.

    [{\'X1 Z2\': 4,\'Z1 Z0\': 3},{\'X1 Y2 Z0\': 3.5}] corresponds to the two observed values 4 * X1 @ Z2 + 3 * Z1 @ Z0 and 3.5 * X1 @ Y2 @ Z0.
        
    Diese Klasse erbt von ``pyvqnet.qnn.vqc.tn.QModule`` und ``torch.nn.Module``.

    Diese Klasse kann als Untermodul von ``torch.nn.Module`` zum torch-Modell hinzugefügt werden.

    :param obs: observable.
    :param name: module name, default: "".
    :return: measurement result, QTensor.

    Example::

        import pyvqnet
        pyvqnet.backends.set_backend("torch")
        from pyvqnet.qnn.vqc.tn import MeasureAll,rx,ry,cnot,TNQMachine,rz
        from pyvqnet.tensor import QTensor
        from pyvqnet import kfloat64
        x = QTensor([[0.56, 0.1],[0.56, 0.1]],requires_grad=True)
        qm = TNQMachine(4)
        qm.reset_states(2)
        rz(q_machine=qm,wires=0,params=x[:,[0]])
        rz(q_machine=qm,wires=1,params=x[:,[0]])
        cnot(q_machine=qm,wires=[0,1])
        ry(q_machine=qm,wires=2,params=x[:,[1]])
        cnot(q_machine=qm,wires=[0,2])
        rz(q_machine=qm,wires=3,params=x[:,[1]])
        obs_list = [{
            "Z0 Z1" :2
        }, {
            "X0 Z2" :1
        }]
        ma = MeasureAll(obs = obs_list)
        y = ma(q_machine=qm)
        print(y)



Samples
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. py:class:: pyvqnet.qnn.vqc.tn.Samples(wires=None, obs=None, shots = 1,name="")

    Erhält Stichprobenergebnisse mit Schuss auf bestimmten Drähten.

    Diese Klasse erbt von ``pyvqnet.qnn.vqc.tn.QModule`` und ``torch.nn.Module``.
    Diese Klasse kann als Untermodul von ``torch.nn.Module`` zum torch-Modell hinzugefügt werden.


    :param wires: Sample qubit index. Default value: None, use all bits of the simulator at runtime.
    :param obs: This value can only be None.
    :param shots: Sample repetition count, default value: 1.
    :param name: The name of this module, default value: "".
    :return: a measurement method class

    Example::

        import pyvqnet
        pyvqnet.backends.set_backend("torch")
        from pyvqnet.qnn.vqc.tn import Samples,rx,ry,cnot,TNQMachine,rz
        from pyvqnet.tensor import QTensor
        from pyvqnet import kfloat64
        x = QTensor([[0.56, 0.1],[0.56, 0.1]],requires_grad=True)

        qm = TNQMachine(4)
        qm.reset_states(2)
        rz(q_machine=qm,wires=0,params=x[:,[0]])
        rx(q_machine=qm,wires=1,params=x[:,[0]])
        cnot(q_machine=qm,wires=[0,1])

        cnot(q_machine=qm,wires=[0,2])
        ry(q_machine=qm,wires=3,params=x[:,[1]])


        ma = Samples(wires=[0,1,2],shots=3)
        y = ma(q_machine=qm)
        print(y)


HermitianExpval
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. py:class:: pyvqnet.qnn.vqc.tn.HermitianExpval(obs=None, name="")

    Berechnet den Erwartungswert einer hermiteschen Größe in einem Quantenschaltkreis.

    Diese Klasse erbt von ``pyvqnet.qnn.vqc.tn.QModule`` und ``torch.nn.Module``.
    Diese Klasse kann als Untermodul von ``torch.nn.Module`` zum torch-Modell hinzugefügt werden.


    :param obs: Hermitian quantity.
    :param name: module name, default: "".
    :return: expected result, QTensor.

    Example::

        import pyvqnet
        pyvqnet.backends.set_backend("torch")
        from pyvqnet.qnn.vqc.tn import TNQMachine, rx,ry,\
            RX, RY, CNOT, PauliX, PauliZ, VQC_RotCircuit,HermitianExpval, TNQModule
        from pyvqnet.tensor import QTensor, tensor
        from pyvqnet.nn import Parameter
        import numpy as np
        bsz = 3
        H = np.array([[8, 4, 0, -6], [4, 0, 4, 0], [0, 4, 8, 0], [-6, 0, 0, 0]])
        class QModel(TNQModule):
            def __init__(self, num_wires, dtype):
                super(QModel, self).__init__()
                self.rot_param = Parameter((3, ))
                self._num_wires = num_wires
                self._dtype = dtype
                self.qm = TNQMachine(num_wires, dtype=dtype)
                self.rx_layer1 = VQC_RotCircuit
                self.ry_layer2 = RY(has_params=True,
                                    trainable=True,
                                    wires=0,
                                    init_params=tensor.QTensor([-0.5]))
                self.xlayer = PauliX(wires=0)
                self.cnot = CNOT(wires=[0, 1])
                self.measure = HermitianExpval(obs = {'wires':(1,0),'observables':tensor.to_tensor(H)})

            def forward(self, x, *args, **kwargs):
                self.qm.reset_states(bsz)

                rx(q_machine=self.qm, wires=0, params=x[1])
                ry(q_machine=self.qm, wires=1, params=x[0])
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

Häufige Vorlagen für Quantenschaltkreise
--------------------------------------------

VQC_HardwareEfficientAnsatz
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^


.. py:class:: pyvqnet.qnn.vqc.tn.VQC_HardwareEfficientAnsatz(n_qubits,single_rot_gate_list,entangle_gate="CNOT",entangle_rules='linear',depth=1,initial=None,dtype=None)

    Implementierung des Hardware Efficient Ansatz, vorgestellt im Paper: `Hardware-efficient Variational Quantum Eigensolver for Small Molecules <https://arxiv.org/pdf/1704.05018.pdf>`__ .

    Diese Klasse erbt von ``pyvqnet.qnn.vqc.tn.QModule`` und ``torch.nn.Module``.

    Diese Klasse kann als Untermodul von ``torch.nn.Module`` zum torch-Modell hinzugefügt werden.


    :param n_qubits: Anzahl der Qubits.
    :param single_rot_gate_list: A single qubit rotation gate list is constructed by one or several rotation gate that act on every qubit.Currently support Rx, Ry, Rz.
    :param entangle_gate: The non parameterized entanglement gate.CNOT,CZ is supported.default:CNOT.
    :param entangle_rules: How entanglement gate is used in the circuit. 'linear' means the entanglement gate will be act on every neighboring qubits. 'all' means the entanglment gate will be act on any two qbuits. Default:linear.
    :param depth: The depth of ansatz, default:1.
    :param initial: initial one same value for paramaters,default:None,this module will initialize parameters randomly.
    :param dtype: data dtype of parameters.
    :return: a VQC_HardwareEfficientAnsatz instance.

    Example::

        from pyvqnet.nn.torch import Linear
        from pyvqnet.qnn.vqc.tn.qcircuit import VQC_HardwareEfficientAnsatz,RZZ,RZ
        from pyvqnet.qnn.vqc.tn import Probability,TNQMachine, TNQModule
        from pyvqnet import tensor
        import pyvqnet
        pyvqnet.backends.set_backend("torch")
        pyvqnet.utils.set_random_seed(25)

        class QM(TNQModule):
            def __init__(self, name=""):
                super().__init__(name)
                self.linearx = Linear(4,2)
                self.ansatz = VQC_HardwareEfficientAnsatz(4, ["rx", "RY", "rz"],
                                            entangle_gate="cnot",
                                            entangle_rules="linear",
                                            depth=2)
                self.encode1 = RZ(wires=0)
                self.encode2 = RZ(wires=1)
                self.measure = Probability(wires=[0, 2])
                self.device = TNQMachine(4)
            def forward(self, x, *args, **kwargs):
                self.device.reset_states(bz)
                y = self.linearx(x)
                self.encode1(params = y[0],q_machine = self.device,)
                self.encode2(params = y[1],q_machine = self.device,)
                self.ansatz(q_machine =self.device)
                return self.measure(q_machine =self.device)

        bz =3
        inputx = tensor.arange(1.0,bz*4+1).reshape([bz,4])
        inputx.requires_grad= True
        qlayer = QM()
        y = qlayer(inputx)
        y.backward()
        print(y)



VQC_BasicEntanglerTemplate
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. py:class:: pyvqnet.qnn.vqc.tn.VQC_BasicEntanglerTemplate(num_layer=1, num_qubits=1, rotation="RX", initial=None, dtype=None)

    Eine Schicht bestehend aus einer Ein-Parameter-Einzel-Qubit-Rotation auf jedem Qubit, gefolgt von mehreren CNOT-Gattern in einer geschlossenen Kette oder Ring-Kombination.

    Ein Ring von CNOT-Gattern verbindet jedes Qubit mit seinen Nachbarn, und schließlich wird das a-te Qubit als Nachbar des a-ten Qubits betrachtet.

    Diese Klasse erbt von ``pyvqnet.qnn.vqc.tn.QModule`` und ``torch.nn.Module``.

    Diese Klasse kann als Untermodul von ``torch.nn.Module`` zum torch-Modell hinzugefügt werden.

    :param num_layers: number of repeat layers, default: 1.
    :param num_qubits: Anzahl der Qubits, Standard: 1.
    :param rotation: zu verwendendes Ein-Parameter-Ein-Qubit-Gatter, Standard: `RX`
    :param initial: initialisiert gleichen Wert für alle Parameter. Standard: None, Parameter werden zufällig initialisiert.
    :param dtype: Datentyp des Parameters, Standard: None, verwende float32.
    :return: Eine VQC_BasicEntanglerTemplate-Instanz

    Example::

        import pyvqnet
        pyvqnet.backends.set_backend("torch")
        from pyvqnet.qnn.vqc.tn import TNQModule,\
            VQC_BasicEntanglerTemplate, Probability, TNQMachine
        from pyvqnet import tensor


        class QM(TNQModule):
            def __init__(self, name=""):
                super().__init__(name)

                self.ansatz = VQC_BasicEntanglerTemplate(2,
                                                    4,
                                                    "rz",
                                                    initial=tensor.ones([1, 1]))

                self.measure = Probability(wires=[0, 2])
                self.device = TNQMachine(4)

            def forward(self,x, *args, **kwargs):

                self.ansatz(q_machine=self.device)
                return self.measure(q_machine=self.device)

        bz = 1
        inputx = tensor.arange(1.0, bz * 4 + 1).reshape([bz, 4])
        qlayer = QM()
        y = qlayer(inputx)
        y.backward()
        print(y)



VQC_StronglyEntanglingTemplate
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. py:class:: pyvqnet.qnn.vqc.tn.VQC_StronglyEntanglingTemplate(num_layers=1, num_qubits=1, ranges=None,initial=None, dtype=None)

    Schichten bestehend aus Einzel-Qubit-Rotationen und Entanglern, wie im `circuit-centric classifier design <https://arxiv.org/abs/1804.00633>`__ .

    Diese Klasse erbt von ``pyvqnet.qnn.vqc.tn.QModule`` und ``torch.nn.Module``.
    Diese Klasse kann als Untermodul von ``torch.nn.Module`` zum torch-Modell hinzugefügt werden.


    :param num_layers: number of repeat layers, default: 1.
    :param num_qubits: Anzahl der Qubits, Standard: 1.
    :param ranges: sequence determining the range hyperparameter for each subsequent layer; default: None
                                using :math: `r=l \mod M` for the :math:`l` th layer and :math:`M` qubits.
    :param initial: initial value for all parameters.default: None,initialized randomly.
    :param dtype: data type of parameter, default:None,use float32.
    :return: A VQC_StronglyEntanglingTemplate instance.

    Example::

        from pyvqnet.nn.torch import TorchModule,Linear,TorchModuleList
        from pyvqnet.qnn.vqc.tn.qcircuit import VQC_StronglyEntanglingTemplate
        from pyvqnet.qnn.vqc.tn import Probability, TNQMachine, TNQModule
        from pyvqnet import tensor
        import pyvqnet

        pyvqnet.backends.set_backend("torch")
        pyvqnet.utils.set_random_seed(25)
        class QM(TNQModule):
            def __init__(self, name=""):
                super().__init__(name)

                self.ansatz = VQC_StronglyEntanglingTemplate(2,
                                                    4,
                                                    None,
                                                    initial=tensor.ones([1, 1]))

                self.measure = Probability(wires=[0, 1])
                self.device = TNQMachine(4)

            def forward(self,x, *args, **kwargs):

                self.ansatz(q_machine=self.device)
                return self.measure(q_machine=self.device)

        bz = 1
        inputx = tensor.arange(1.0, bz * 4 + 1).reshape([bz, 4])
        qlayer = QM()
        y = qlayer(inputx)
        y.backward()
        print(y)



VQC_QuantumEmbedding
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^


.. py:class:: pyvqnet.qnn.vqc.tn.VQC_QuantumEmbedding(qubits, machine, num_repetitions_input, depth_input, num_unitary_layers, num_repetitions,initial = None,dtype = None,name= "")
    
    Verwendet RZ, RY, RZ zur Erstellung variationeller Quantenschaltkreise zur Codierung klassischer Daten in Quantenzustände.
    Referenz `Quantum embeddings for machine learning <https://arxiv.org/abs/2001.03622>`_.

    Diese Klasse erbt von ``pyvqnet.qnn.vqc.tn.QModule`` und ``torch.nn.Module``.
    Diese Klasse kann als Untermodul von ``torch.nn.Module`` zum torch-Modell hinzugefügt werden.

 
    :param num_repetitions_input: Anzahl der Wiederholungen zur Codierung der Eingabe in einem Untermodul.
    :param depth_input: Anzahl der Eingabedimensionen.
    :param num_unitary_layers: Anzahl der Wiederholungen variationeller Quantengatter.
    :param num_repetitions: Anzahl der Wiederholungen des Untermoduls.
    :param initial: Parameter-Initialisierungswert, Standard ist None
    :param dtype: Parametertyp, Standard ist None, verwende float32.
    :param name: Klassenname
    :return: Eine VQC_QuantumEmbedding-Instanz.

    Example::

        from pyvqnet.qnn.vqc.tn.qcircuit import VQC_QuantumEmbedding
        from pyvqnet.qnn.vqc.tn import TNQMachine, MeasureAll, TNQModule
        from pyvqnet import tensor
        import pyvqnet

        pyvqnet.backends.set_backend("torch")
        pyvqnet.utils.set_random_seed(25)
        depth_input = 2
        num_repetitions = 2
        num_repetitions_input = 2
        num_unitary_layers = 2
        nq = depth_input * num_repetitions_input
        bz = 12

        class QM(TNQModule):
            def __init__(self, name=""):
                super().__init__(name)

                self.ansatz = VQC_QuantumEmbedding(num_repetitions_input, depth_input,
                                                num_unitary_layers,
                                                num_repetitions, initial=tensor.full([1],12.0),dtype=pyvqnet.kfloat32)

                self.measure = MeasureAll(obs={f"Z{nq-1}":1})
                self.device = TNQMachine(nq)

            def forward(self, x, *args, **kwargs):
                self.device.reset_states(bz)
                self.ansatz(x,q_machine=self.device)
                return self.measure(q_machine=self.device)

        inputx = tensor.arange(1.0, bz * depth_input + 1).reshape([bz, depth_input])
        qlayer = QM()
        y = qlayer(inputx)
        y.backward()
        print(y)


ExpressiveEntanglingAnsatz
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. py:class:: pyvqnet.qnn.vqc.tn.ExpressiveEntanglingAnsatz(type: int, num_wires: int, depth: int, dtype=None, name: str = "")

    19 verschiedene Ansätze aus dem Paper `Expressibility and entangling capability of parameterized quantum circuits for hybrid quantum-classical algorithms <https://arxiv.org/pdf/1905.10876.pdf>`_.

    Diese Klasse erbt von ``pyvqnet.qnn.vqc.tn.QModule`` und ``torch.nn.Module``.

    Diese Klasse kann als Untermodul von ``torch.nn.Module`` zum torch-Modell hinzugefügt werden.

    :param type: Circuit type from 1 to 19, a total of 19 lines.
    :param num_wires: Anzahl der Qubits.
    :param depth: Circuit depth.
    :param dtype: data type of parameter, default:None,use float32.
    :param name: Name, default "".

    :return:
        a ExpressiveEntanglingAnsatz instance

    Example::

        from pyvqnet.qnn.vqc.tn.qcircuit import ExpressiveEntanglingAnsatz
        from pyvqnet.qnn.vqc.tn import Probability, TNQMachine, MeasureAll, TNQModule
        from pyvqnet import tensor
        import pyvqnet

        pyvqnet.backends.set_backend("torch")
        pyvqnet.utils.set_random_seed(25)

        class QModel(TNQModule):
            def __init__(self, num_wires, dtype):
                super(QModel, self).__init__()

                self._num_wires = num_wires
                self._dtype = dtype
                self.qm = TNQMachine(num_wires, dtype=dtype)
                self.c1 = ExpressiveEntanglingAnsatz(1,3,2)
                self.measure = MeasureAll(obs={
                    "Z1":1
                })

            def forward(self, x, *args, **kwargs):
                self.qm.reset_states(1)
                self.c1(q_machine = self.qm)
                rlt = self.measure(q_machine=self.qm)
                return rlt
            

        input_x = tensor.QTensor([[0.1, 0.2, 0.3]])

        qunatum_model = QModel(num_wires=3, dtype=pyvqnet.kcomplex64)

        batch_y = qunatum_model(input_x)
        batch_y.backward()
        print(batch_y)


vqc_basis_embedding
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. py:function:: pyvqnet.qnn.vqc.tn.vqc_basis_embedding(basis_state,q_machine)

    Codiert n binäre Merkmale in den n-Qubit-Basiszustand von ``q_machine``. Diese Funktion hat den Alias `VQC_BasisEmbedding`.

    For example, for ``basis_state=([0, 1, 1])``, the basis state in the quantum system is :math:`|011 \rangle`.

    :param basis_state: ``(n)`` size binary input.
    :param q_machine: quantum machine device.
    

    Example::

        import pyvqnet
        pyvqnet.backends.set_backend("torch")
        from pyvqnet.qnn.vqc.tn import vqc_basis_embedding,TNQMachine
        qm  = TNQMachine(3)
        vqc_basis_embedding(basis_state=[1,1,0],q_machine=qm)
        print(qm.get_states())




vqc_angle_embedding
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^


.. py:function:: pyvqnet.qnn.vqc.tn.vqc_angle_embedding(input_feat, wires, q_machine: pyvqnet.qnn.vqc.tn.TNQMachine, rotation: str = "X")

    Codiert :math:`N` Merkmale in den Rotationswinkel von :math:`n` Qubits, wobei :math:`N \leq n`.
    Diese Funktion hat den Alias `VQC_AngleEmbedding` .

    Die Rotation kann ausgewählt werden als: 'X', 'Y', 'Z', wie durch den ``rotation``-Parameter definiert:

    * ``rotation='X'`` Verwendet das Merkmal als Winkel der RX-Rotation.

    * ``rotation='Y'`` Verwendet das Merkmal als Winkel der RY-Rotation.

    * ``rotation='Z'`` Verwendet das Merkmal als Winkel der RZ-Rotation.

    ``wires`` repräsentiert den Index des Rotationsgatters auf dem Qubit.

    :param input_feat: Array representing parameters.
    :param wires: Qubit idx.
    :param q_machine: Quantum machine device.
    :param rotation: Rotation gate, default is "X".

    Example::

        import pyvqnet
        pyvqnet.backends.set_backend("torch")
        from pyvqnet.qnn.vqc.tn import vqc_angle_embedding, TNQMachine
        from pyvqnet.tensor import QTensor
        qm  = TNQMachine(2)
        vqc_angle_embedding(QTensor([2.2, 1]), [0, 1], q_machine=qm, rotation='X')
        print(qm.get_states())
        vqc_angle_embedding(QTensor([2.2, 1]), [0, 1], q_machine=qm, rotation='Y')
        print(qm.get_states())
        vqc_angle_embedding(QTensor([2.2, 1]), [0, 1], q_machine=qm, rotation='Z')
        print(qm.get_states())



vqc_amplitude_embedding
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. py:function:: pyvqnet.qnn.vqc.tn.vqc_amplitude_embedding(input_feature, q_machine)

    Codiert ein :math:`2^n`-Merkmal in einen Amplitudenvektor von :math:`n` Qubits. Diese Funktion hat den Alias `VQC_AmplitudeEmbedding`.

    :param input_feature: numpy array representing the parameter.
    :param q_machine: quantum machine device.
    

    Example::

        import pyvqnet
        pyvqnet.backends.set_backend("torch")
        from pyvqnet.qnn.vqc.tn import vqc_amplitude_embedding, TNQMachine
        from pyvqnet.tensor import QTensor
        qm  = TNQMachine(3)
        vqc_amplitude_embedding(QTensor([3.2,-2,-2,0.3,12,0.1,2,-1]), q_machine=qm)
        print(qm.get_states())



vqc_iqp_embedding
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. py:function:: pyvqnet.qnn.vqc.tn.vqc_iqp_embedding(input_feat, q_machine: pyvqnet.qnn.vqc.tn.TNQMachine, rep: int = 1)

    Codiert :math:`n` Merkmale in :math:`n` Qubits unter Verwendung diagonaler Gatter einer IQP-Schaltung. Alias: ``VQC_IQPEmbedding`` .

    The encoding is proposed by `Havlicek et al. (2018) <https://arxiv.org/pdf/1804.11326.pdf>`_.

    By specifying ``rep`` , the basic IQP circuit can be repeated.

    :param input_feat: Array of parameters.
    :param q_machine: Quantum machine machine.
    :param rep: Number of times to repeat the quantum circuit block, default is 1.

    Example::

        import pyvqnet
        pyvqnet.backends.set_backend("torch")
        from pyvqnet.qnn.vqc.tn import vqc_iqp_embedding, TNQMachine
        from pyvqnet.tensor import QTensor
        qm  = TNQMachine(3)
        vqc_iqp_embedding(QTensor([3.2,-2,-2]), q_machine=qm)
        print(qm.get_states())        



vqc_rotcircuit
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. py:function:: pyvqnet.qnn.vqc.tn.vqc_rotcircuit(q_machine, wire, params)

    Beliebige Einzel-Qubit-Rotations-Quantenlogikgatter-Kombination. Diese Funktion hat den Alias: ``VQC_RotCircuit`` .

    .. math::
        R(\phi,\theta,\omega) = RZ(\omega)RY(\theta)RZ(\phi)= \begin{bmatrix}
        e^{-i(\phi+\omega)/2}\cos(\theta/2) & -e^{i(\phi-\omega)/2}\sin(\theta/2) \\
        e^{-i(\phi-\omega)/2}\sin(\theta/2) & e^{i(\phi+\omega)/2}\cos(\theta/2)
        \end{bmatrix}.

    :param q_machine: quantum virtual machine device.
    :param wire: quantum bit index.
    :param params: represents parameters :math:`[\phi, \theta, \omega]`.

    Example::

        import pyvqnet
        pyvqnet.backends.set_backend("torch")
        from pyvqnet.qnn.vqc.tn import vqc_rotcircuit, TNQMachine
        from pyvqnet.tensor import QTensor
        qm  = TNQMachine(3)
        vqc_rotcircuit(q_machine=qm, wire=[1],params=QTensor([2.0,1.5,2.1]))
        print(qm.get_states())


vqc_crot_circuit
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^


.. py:function:: pyvqnet.qnn.vqc.tn.vqc_crot_circuit(para,control_qubits,rot_wire,q_machine)

    Quantenlogikgatter-Kombination einer gesteuerten Rot-Einzel-Qubit-Rotation. Diese Funktion hat den Alias: ``VQC_CRotCircuit`` .

    .. math:: 
        CR(\phi, \theta, \omega) = \begin{bmatrix}
        1 & 0 & 0 & 0 \\
        0 & 1 & 0 & 0\\
        0 & 0 & e^{-i(\phi+\omega)/2}\cos(\theta/2) & -e^{i(\phi-\omega)/2}\sin(\theta/2)\\
        0 & 0 & e^{-i(\phi-\omega)/2}\sin(\theta/2) & e^{i(\phi+\omega)/2}\cos(\theta/2)
        \end{bmatrix}.

    :param para: represents the array of parameters.
    :param control_qubits: Control qubit index.
    :param rot_wire: Rot qubit index.
    :param q_machine: Quantum machine device.
    

    Example::

        import pyvqnet
        pyvqnet.backends.set_backend("torch")
        from pyvqnet.tensor import QTensor
        from pyvqnet.qnn.vqc.tn import vqc_crot_circuit,TNQMachine, MeasureAll
        p = QTensor([2, 3, 4.0])
        qm = TNQMachine(2)
        vqc_crot_circuit(p, 0, 1, qm)
        m = MeasureAll(obs={"Z0": 1})
        exp = m(q_machine=qm)
        print(exp)




vqc_controlled_hadamard
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^


.. py:function:: pyvqnet.qnn.vqc.tn.vqc_controlled_hadamard(wires, q_machine)

    Gesteuerter Hadamard-Logikgatter-Quantenschaltkreis. Diese Funktion hat den Alias: ``VQC_Controlled_Hadamard`` .

    .. math:: 
        CH = \begin{bmatrix}
        1 & 0 & 0 & 0 \\
        0 & 1 & 0 & 0 \\
        0 & 0 & \frac{1}{\sqrt{2}} & \frac{1}{\sqrt{2}} \\
        0 & 0 & \frac{1}{\sqrt{2}} & -\frac{1}{\sqrt{2}}
        \end{bmatrix}.

    :param wires: quantum bit index list, the first one is the control bit, the list length is 2.
    :param q_machine: quantum virtual machine device.

    Examples::

        import pyvqnet
        pyvqnet.backends.set_backend("torch")
        from pyvqnet.tensor import QTensor
        from pyvqnet.qnn.vqc.tn import vqc_controlled_hadamard,\
            TNQMachine, MeasureAll

        p = QTensor([0.2, 3, 4.0])
        qm = TNQMachine(3)
        vqc_controlled_hadamard([1, 0], qm)
        m = MeasureAll(obs={"Z0": 1})
        exp = m(q_machine=qm)
        print(exp)


vqc_ccz
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. py:function:: pyvqnet.qnn.vqc.tn.vqc_ccz(wires, q_machine)

    Gesteuert-gesteuertes-Z-Logikgatter. Alias: ``VQC_CCZ`` .

    .. math::
        CCZ =
        \begin{pmatrix}
        1 & 0 & 0 & 0 & 0 & 0 & 0 & 0\\
        0 & 1 & 0 & 0 & 0 & 0 & 0 & 0 & 0\\
        0 & 0 & 1 & 0 & 0 & 0 & 0 & 0\\
        0 & 0 & 0 & 1 & 0 & 0 & 0 & 0\\
        0 & 0 & 0 & 0 & 1 & 0 & 0 & 0\\
        0 & 0 & 0 & 0 & 0 & 1 & 0 & 0 & 0\\
        0 & 0 & 0 & 0 & 0 & 1 & 0 & 0\\
        0 & 0 & 0 & 0 & 0 & 0 & 1 & 0 & 0\\
        0 & 0 & 0 & 0 & 0 & 0 & 0 & 1 & 0\\
        0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & -1
        \end{pmatrix}

    :param wires: quantum bit index list, the first one is the control bit. The list length is 3.
    :param q_machine: quantum virtual machine device.

    Example::

        import pyvqnet
        pyvqnet.backends.set_backend("torch")
        from pyvqnet.tensor import QTensor
        from pyvqnet.qnn.vqc.tn import vqc_ccz,TNQMachine, MeasureAll
        p = QTensor([0.2, 3, 4.0])

        qm = TNQMachine(3)

        vqc_ccz([1, 0, 2], qm)
        m = MeasureAll(obs={"Z0": 1})
        exp = m(q_machine=qm)
        print(exp)



vqc_fermionic_single_excitation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. py:function:: pyvqnet.qnn.vqc.tn.vqc_fermionic_single_excitation(weight, wires, q_machine)

    Gekoppelter-Cluster-Einfachanregungsoperator für das Tensorprodukt von Pauli-Matrizen. Matrix form is given by:

    .. math::
        \hat{U}_{pr}(\theta) = \mathrm{exp} \{ \theta_{pr} (\hat{c}_p^\dagger \hat{c}_r
        -\mathrm{H.c.}) \},

    Alias: ``VQC_FermionicSingleExcitation`` .

    :param weight: Parameter on qubit p, only a elements.
    :param wires: A subset of qubit indices in the interval [r, p]. Minimum length must be 2. The first index value is interpreted as r, and the last a index value is interpreted as p.The intermediate indices are acted upon by CNOT gates to compute the parity of the qubit set.
    :param q_machine: Quantum virtual machine device.

    

    Examples::

        import pyvqnet
        pyvqnet.backends.set_backend("torch")
        from pyvqnet.tensor import QTensor
        from pyvqnet.qnn.vqc.tn import vqc_fermionic_single_excitation,\
            TNQMachine, MeasureAll
        qm = TNQMachine(3)
        p0 = QTensor([0.5])

        vqc_fermionic_single_excitation(p0, [1, 0, 2], qm)
        m = MeasureAll(obs={"Z0": 1})
        exp = m(q_machine=qm)
        print(exp)

 


vqc_fermionic_double_excitation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^


.. py:function:: pyvqnet.qnn.vqc.tn.vqc_fermionic_double_excitation(weight, wires1, wires2, q_machine)

    Gekoppelter-Cluster-Doppelanregungsoperator für das exponenzierte Tensorprodukt von Pauli-Matrizen, Matrixform gegeben durch:

    .. math::
        \hat{U}_{pqrs}(\theta) = \mathrm{exp} \{ \theta (\hat{c}_p^\dagger \hat{c}_q^\dagger
        \hat{c}_r \hat{c}_s - \mathrm{H.c.}) \},

    wobei :math:`\hat{c}` und :math:`\hat{c}^\dagger` Fermionen-Vernichtungs- und
    Erzeugungsoperatoren sind und :math:`r, s` und :math:`p, q` auf besetzten bzw.
    leeren Molekülorbitalen indizieren. Verwenden Sie die `Jordan-Wigner-Transformation
    <https://arxiv.org/abs/1208.5986>`_ The fermion operator defined above can be written as
    in terms of the Pauli matrix (see
    `arXiv:1805.04340 <https://arxiv.org/abs/1805.04340>`_ for more details)

    .. math::
        \hat{U}_{pqrs}(\theta) = \mathrm{exp} \Big\{
        \frac{i\theta}{8} \bigotimes_{b=s+1}^{r-1} \hat{Z}_b \bigotimes_{a=q+1}^{p-1}
        \hat{Z}_a (\hat{X}_s \hat{X}_r \hat{Y}_q \hat{X}_p +
        \hat{Y}_s \hat{X}_r \hat{Y}_q \hat{Y}_p +\\ \hat{X}_s \hat{Y}_r \hat{Y}_q \hat{Y}_p +
        \hat{X}_s \hat{X}_r \hat{X}_q \hat{Y}_p - \mathrm{H.c.} ) \Big\}

    This function is aliased as: ``VQC_FermionicDoubleExcitation`` .

    :param weight: variable parameter
    :param wires1: represents the subset of qubits in the index list interval [s, r]. The ath index is interpreted as s and the last index is interpreted as r. The CNOT gate operates on the middle indexes to calculate the parity of a group of qubits.
    :param wires2: represents the subset of qubits in the index list interval [q, p]. The first root index is interpreted as q and the last index is interpreted as p. The CNOT gate operates on the middle indexes to calculate the parity of a group of qubits.
    :param q_machine: Quantum virtual machine device.

    

    Examples::

        import pyvqnet
        pyvqnet.backends.set_backend("torch")
        from pyvqnet.tensor import QTensor
        from pyvqnet.qnn.vqc.tn import vqc_fermionic_double_excitation,\
            TNQMachine, MeasureAll
        qm = TNQMachine(5)
        p0 = QTensor([0.5])

        vqc_fermionic_double_excitation(p0, [0, 1], [2, 3], qm)
        m = MeasureAll(obs={"Z0": 1})
        exp = m(q_machine=qm)
        print(exp)
 

vqc_uccsd
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^


.. py:function:: pyvqnet.qnn.vqc.tn.vqc_uccsd(weights, wires, s_wires, d_wires, init_state, q_machine)

    Implementiert die Unitary Coupled Cluster Single and Double Excitations Simulation (UCCSD). UCCSD ist eine VQE-Simulation, die häufig für quantenchemische Simulationen verwendet wird.

    Within the first-order Trotter approximation, the UCCSD unitary function is given by:

    .. math::
        \hat{U}(\vec{\theta}) =
        \prod_{p > r} \mathrm{exp} \Big\{\theta_{pr}
        (\hat{c}_p^\dagger \hat{c}_r-\mathrm{H.c.}) \Big\}
        \prod_{p > q > r > s} \mathrm{exp} \Big\{\theta_{pqrs}
        (\hat{c}_p^\dagger \hat{c}_q^\dagger \hat{c}_r \hat{c}_s-\mathrm{H.c.}) \Big\}

    where :math:`\hat{c}` and :math:`\hat{c}^\dagger` are fermion annihilation and
    creation operators and index :math:`r, s` and :math:`p, q` on occupied and
    empty molecular orbitals respectively. (For more details see
    `arXiv:1805.04340 <https://arxiv.org/abs/1805.04340>`_):

    This function is aliased as: ``VQC_UCCSD`` .

    :param weights: tensor of size ``(len(s_wires)+ len(d_wires))`` containing the parameters :math:`\theta_{pr}` and :math:`\theta_{pqrs}` input Z rotations ``FermionicSingleExcitation`` and ``FermionicDoubleExcitation`` .
    :param wires: qubit indices for template action
    :param s_wires: sequence of lists containing qubit indices ``[r,...,p]`` generated by a single excitation :math:`\vert r, p \rangle = \hat{c}_p^\dagger \hat{c}_r \vert \mathrm{HF} \rangle`,where :math:`\vert \mathrm{HF} \rangle` denotes the Hartee-Fock reference state.
    :param d_wires: sequence of lists, each containing two lists specifying indices ``[s, ...,r]`` and ``[q,..., p]`` defining double excitation :math:`\vert s, r, q, p \rangle = \hat{c}_p^\dagger \hat{c}_q^\dagger \hat{c}_r\hat{c}_s \vert \mathrm{HF} \rangle` .
    :param init_state: occupation-number vector of length ``len(wires)`` representing the high-frequency state. ``init_state`` Initialization state of the qubit.
    :param q_machine: Quantum virtual machine device.

    Examples::

        import pyvqnet
        pyvqnet.backends.set_backend("torch")
        from pyvqnet.qnn.vqc.tn import vqc_uccsd, TNQMachine, MeasureAll
        from pyvqnet.tensor import QTensor
        p0 = QTensor([2, 0.5, -0.2, 0.3, -2, 1, 3, 0])
        s_wires = [[0, 1, 2], [0, 1, 2, 3, 4], [1, 2, 3], [1, 2, 3, 4, 5]]
        d_wires = [[[0, 1], [2, 3]], [[0, 1], [2, 3, 4, 5]], [[0, 1], [3, 4]],
                [[0, 1], [4, 5]]]
        qm = TNQMachine(6)

        vqc_uccsd(p0, range(6), s_wires, d_wires, QTensor([1.0, 1, 0, 0, 0, 0]), qm)
        m = MeasureAll(obs={"Z1": 1})
        exp = m(q_machine=qm)
        print(exp)

        # [[0.963802]]


vqc_zfeaturemap
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. py:function:: pyvqnet.qnn.vqc.tn.vqc_zfeaturemap(input_feat, q_machine: pyvqnet.qnn.vqc.tn.TNQMachine, data_map_func=None, rep: int = 2)

    Pauli-Z-Evolutionsschaltung erster Ordnung.

    For 3 qubits and 2 repetitions, the circuit is represented as:

    .. parsed-literal::

        ┌───┐┌──────────────┐┌───┐┌──────────────┐
        ┤ H ├┤ U1(2.0*x[0]) ├┤ H ├┤ U1(2.0*x[0]) ├
        ├───┤├──────────────┤├───┤├──────────────┤
        ┤ H ├┤ U1(2.0*x[1]) ├┤ H ├┤ U1(2.0*x[1]) ├
        ├───┤├──────────────┤├───┤├──────────────┤
        ┤ H ├┤ U1(2.0*x[2]) ├┤ H ├┤ U1(2.0*x[2]) ├
        └───┘└──────────────┘└───┘└──────────────┘

    Die Pauli-Zeichenkette ist auf ``Z`` festgelegt. Daher wird die Expansion erster Ordnung eine Schaltung ohne Verschlingungsgatter sein.

    :param input_feat: Array, das Eingabeparameter repräsentiert.
    :param q_machine: Quanten-Virtuelle-Maschine.
    :param data_map_func: Parameter-Abbildungsmatrix, eine aufrufbare Funktion, entworfen als: ``data_map_func = lambda x: x``.
    :param rep: Anzahl der Wiederholungen des Moduls.

    Example::

        import pyvqnet
        pyvqnet.backends.set_backend("torch")
        from pyvqnet.qnn.vqc.tn import vqc_zfeaturemap, TNQMachine, hadamard
        from pyvqnet.tensor import QTensor
        qm = TNQMachine(3)
        for i in range(3):
            hadamard(q_machine=qm, wires=[i])
        vqc_zfeaturemap(input_feat=QTensor([[0.1,0.2,0.3]]),q_machine = qm)
        print(qm.get_states())
 

vqc_zzfeaturemap
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. py:function:: pyvqnet.qnn.vqc.tn.vqc_zzfeaturemap(input_feat, q_machine: pyvqnet.qnn.vqc.tn.TNQMachine, data_map_func=None, entanglement: Union[str, List[List[int]],Callable[[int], List[int]]] = "full",rep: int = 2)

    Pauli-Z-Evolutionsschaltung zweiter Ordnung.

    For 3 qubits, 1 repeat, and linear entanglement, the circuit is represented as:

    .. parsed-literal::


        ┌───┐┌─────────────────┐
        ┤ H ├┤ U1(2.0*φ(x[0])) ├──■────────────────────────────■────────────────────────────────────
        ├───┤├─────────────────┤┌─┴─┐┌──────────────────────┐┌─┴─┐
        ┤ H ├┤ U1(2.0*φ(x[1])) ├┤ X ├┤ U1(2.0*φ(x[0],x[1])) ├┤ X ├──■────────────────────────────■──
        ├───┤├─────────────────┤└───┘└──────────────────────┘└───┘┌─┴─┐┌──────────────────────┐┌─┴─┐
        ┤ H ├┤ U1(2.0*φ(x[2])) ├──────────────────────────────────┤ X ├┤ U1(2.0*φ(x[1],x[2])) ├┤ X ├
        └───┘└─────────────────┘                                  └───┘└──────────────────────┘└───┘
    
    Where ``φ`` is a classic nonlinear function. If two values ​​are input, ``φ(x,y) = (pi - x)(pi - y)``, and if a is input, ``φ(x) = x``. It is expressed as follows using ``data_map_func``:

    .. code-block::

        def data_map_func(x):
            coeff = x if x.shape[-1] == 1 else ft.reduce(lambda x, y: (np.pi - x) * (np.pi - y), x)
            return coeff

    :param input_feat: Array, das Eingabeparameter repräsentiert.
    :param q_machine: Quanten-Virtuelle-Maschine.
    :param data_map_func: Parameter-Abbildungsmatrix, eine aufrufbare Funktion.
    :param entanglement: angegebene Verschlingungsstruktur.
    :param rep: Anzahl der Modulwiederholungen.
    
    Example::

        import pyvqnet
        pyvqnet.backends.set_backend("torch")
        from pyvqnet.qnn.vqc.tn import vqc_zzfeaturemap, TNQMachine
        from pyvqnet.tensor import QTensor

        qm = TNQMachine(3)
        vqc_zzfeaturemap(q_machine=qm, input_feat=QTensor([[0.1,0.2,0.3]]))
        print(qm.get_states())


vqc_allsinglesdoubles
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. py:function:: pyvqnet.qnn.vqc.tn.vqc_allsinglesdoubles(weights, q_machine: pyvqnet.qnn.vqc.tn.TNQMachine, hf_state, wires, singles=None, doubles=None)

    In diesem Fall haben wir vier Einfach- und Doppelanregungen, um die gesamte Spin-Projektion des Hartree-Fock-Zustands zu erhalten.

    Die resultierende unitäre Matrix bewahrt die Teilchenpopulation und bereitet das n-Qubit-System in einer Überlagerung des anfänglichen Hartree-Fock-Zustands und anderer Zustände vor, die die Mehrfachanregungskonfiguration codieren.

    :param weights: Ein QTensor der Größe ``(len(singles) + len(doubles),)``, der die Winkel enthält, die nacheinander in die vqc.qCircuit.single_excitation- und vqc.qCircuit.double_excitation-Operationen eingehen
    :param q_machine: Die Quantenmaschine.
    :param hf_state: Ein Vektor der Länge ``len(wires)`` mit Besetzungszahlen, die den Hartree-Fock-Zustand repräsentieren, ``hf_state`` wird zur Initialisierung der Drähte verwendet.
    :param wires: Die Qubits, auf die gewirkt werden soll.
    :param singles: Eine Sequenz von Listen mit den Indizes der beiden Qubits, auf die die single_excitation-Operation wirkt.
    :param doubles: Listensequenz mit den Indizes der beiden Qubits, auf die die double_excitation-Operation wirkt.

    For example, the quantum circuit for two electrons and six qubits is shown below:

    .. image:: ./images/all_singles_doubles.png
        :width: 600 px
        :align: center

    |

    Example::

        import pyvqnet
        pyvqnet.backends.set_backend("torch")
        from pyvqnet.qnn.vqc.tn import vqc_allsinglesdoubles, TNQMachine

        from pyvqnet.tensor import QTensor
        qubits = 4
        qm = TNQMachine(qubits)

        vqc_allsinglesdoubles(q_machine=qm, weights=QTensor([0.55, 0.11, 0.53]), 
                              hf_state = QTensor([1,1,0,0]), singles=[[0, 2], [1, 3]], doubles=[[0, 1, 2, 3]], wires=[0,1,2,3])
        print(qm.get_states())

vqc_basisrotation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. py:function:: pyvqnet.qnn.vqc.tn.vqc_basisrotation(q_machine: pyvqnet.qnn.vqc.tn.TNQMachine, wires, unitary_matrix: QTensor, check=False)

    Implementiert eine Schaltung, die ein Ensemble bereitstellt, das zur Durchführung genauer Einzel-Einheits-Basisrotationen verwendet werden kann. The circuit is derived from the single-particle fermion-determined unitary transformation :math:`U(u)` given in `arXiv:1711.04789 <https://arxiv.org/abs/1711.04789>`_\ 
    
    .. math::
        U(u) = \exp{\left( \sum_{pq} \left[\log u \right]_{pq} (a_p^\dagger a_q - a_q^\dagger a_p) \right)}.

    :math:`U(u)` is obtained by using the scheme given in the paper `Optica, 3, 1460 (2016) <https://opg.optica.org/optica/fulltext.cfm?uri=optica-3-12-1460&id=355743>`_\ .

    :param q_machine: Quantenmaschine.
    :param wires: Qubits, auf die gewirkt werden soll.
    :param unitary_matrix: Matrix, die die Basis für die Transformation angibt.
    :param check: überprüft, ob `unitary_matrix` eine unitäre Matrix ist.

    Example::

        import pyvqnet

        pyvqnet.backends.set_backend("torch")
        from pyvqnet.qnn.vqc.tn import vqc_basisrotation, TNQMachine
        from pyvqnet.tensor import QTensor
        import numpy as np

        V = np.array([[0.73678 + 0.27511j, -0.5095 + 0.10704j, -0.06847 + 0.32515j],
                    [0.73678 + 0.27511j, -0.5095 + 0.10704j, -0.06847 + 0.32515j],
                    [-0.21271 + 0.34938j, -0.38853 + 0.36497j, 0.61467 - 0.41317j]])

        eigen_vals, eigen_vecs = np.linalg.eigh(V)
        umat = eigen_vecs.T
        wires = range(len(umat))

        qm = TNQMachine(len(umat))

        vqc_basisrotation(q_machine=qm,
                        wires=wires,
                        unitary_matrix=QTensor(umat, dtype=qm.dtype))

        print(qm.get_states())



Verteilte Schnittstelle
================================================

Verteilte zusammenhängende Funktionen, die bei Verwendung des ``torch``-Rechen-Backends die ``torch.distributed``-Schnittstelle von torch kapseln,

.. note::

    Bitte beachten Sie `torch distributed <https://pytorch.org/docs/stable/distributed.html>`_, um die verteilte Methode zu starten.
    Bei Verwendung von CPU für die Verteilung verwenden Sie bitte ``gloo`` anstelle von ``mpi``.
    Bei Verwendung von GPU für die Verteilung verwenden Sie bitte ``nccl``.

    :ref:`vqnet_dist` VQNets eigene verteilte Schnittstelle ist nicht auf das ``torch``-Rechen-Backend anwendbar.

CommController
-------------------------

.. py:class:: pyvqnet.distributed.ControlComm.CommController(backend,rank=None,world_size=None)
    :no-index:
    
    CommController wird zur Steuerung des Datenkommunikationscontrollers unter CPU und GPU verwendet. Es erzeugt CPU- (gloo) und GPU- (nccl) Controller durch Setzen des Parameters `backend`.
    Diese Klasse ruft backend, rank, world_size auf, um ``torch.distributed.init_process_group(backend, rank, world_size)`` zu initialisieren.

    :param backend: wird verwendet, um einen CPU- oder GPU-Datenkommunikationscontroller zu erzeugen, 'gloo' oder 'nccl'.
    :param rank: die Prozessnummer des aktuellen Programms.
    :param world_size: die Anzahl aller globalen Prozesse.

    :return:
        CommController-Instanz.

    Examples::

        from pyvqnet.distributed import CommController
        import pyvqnet
        pyvqnet.backends.set_backend("torch")
        import os
        import multiprocessing as mp


        def init_process(rank, size):
            """ Initialize the distributed environment. """
            os.environ['MASTER_ADDR'] = '127.0.0.1'
            os.environ['MASTER_PORT'] = '29500'
            os.environ['LOCAL_RANK'] = f"{rank}"
            pp = CommController("gloo", rank=rank, world_size=size)
            
            local_rank = pp.get_rank()
            print(local_rank)


        if __name__ == "__main__":
            world_size = 2
            processes = []
            mp.set_start_method("spawn")
            for rank in range(world_size):
                p = mp.Process(target=init_process, args=(rank, world_size))
                p.start()
                processes.append(p)

            for p in processes:
                p.join()
 

    .. py:method:: getRank()
        :no-index:

        Wird verwendet, um die Prozess-ID des aktuellen Prozesses abzurufen.

        :return: Gibt die Prozess-ID des aktuellen Prozesses zurück.

        Examples::

            from pyvqnet.distributed import CommController
            import pyvqnet
            pyvqnet.backends.set_backend("torch")
            import os
            import multiprocessing as mp


            def init_process(rank, size):
                """ Initialize the distributed environment. """
                os.environ['MASTER_ADDR'] = '127.0.0.1'
                os.environ['MASTER_PORT'] = '29500'
                os.environ['LOCAL_RANK'] = f"{rank}"
                pp = CommController("gloo", rank=rank, world_size=size)
                
                local_rank = pp.getRank()
                print(local_rank)


            if __name__ == "__main__":
                world_size = 2
                processes = []
                mp.set_start_method("spawn")
                for rank in range(world_size):
                    p = mp.Process(target=init_process, args=(rank, world_size))
                    p.start()
                    processes.append(p)

                for p in processes:
                    p.join()
            


    .. py:method:: getSize()
        :no-index:

        Wird verwendet, um die Gesamtzahl der gestarteten Prozesse abzurufen.

        :return: Gibt die Gesamtzahl der Prozesse zurück.

        Examples::

                        from pyvqnet.distributed import CommController
            import pyvqnet
            pyvqnet.backends.set_backend("torch")
            import os
            import multiprocessing as mp


            def init_process(rank, size):
                """ Initialize the distributed environment. """
                os.environ['MASTER_ADDR'] = '127.0.0.1'
                os.environ['MASTER_PORT'] = '29500'
                os.environ['LOCAL_RANK'] = f"{rank}"
                pp = CommController("gloo", rank=rank, world_size=size)
                
                local_rank = pp.getSize()
                print(local_rank)


            if __name__ == "__main__":
                world_size = 2
                processes = []
                mp.set_start_method("spawn")
                for rank in range(world_size):
                    p = mp.Process(target=init_process, args=(rank, world_size))
                    p.start()
                    processes.append(p)

                for p in processes:
                    p.join()
            


    .. py:method:: getLocalRank()
        :no-index:

        In jedem Prozess wird die lokale Prozessnummer jeder Maschine über ``os.environ['LOCAL_RANK'] = rank`` abgerufen.

        Die Umgebungsvariable `LOCAL_RANK` muss im Voraus gesetzt werden.

        :return: Die aktuelle Prozessnummer auf der aktuellen Maschine.

        Examples::

            from pyvqnet.distributed import CommController
            import pyvqnet
            pyvqnet.backends.set_backend("torch")
            import os
            import multiprocessing as mp


            def init_process(rank, size):
                """ Initialize the distributed environment. """
                os.environ['MASTER_ADDR'] = '127.0.0.1'
                os.environ['MASTER_PORT'] = '29500'
                os.environ['LOCAL_RANK'] = f"{rank}"
                pp = CommController("gloo", rank=rank, world_size=size)
                
                local_rank = pp.getLocalRank()
                print(local_rank )


            if __name__ == "__main__":
                world_size = 2
                processes = []
                mp.set_start_method("spawn")
                for rank in range(world_size):
                    p = mp.Process(target=init_process, args=(rank, world_size))
                    p.start()
                    processes.append(p)

                for p in processes:
                    p.join()

 
    .. py:method:: split_groups(rankL)
        :no-index:

        Die gemäß dem Eingabeparameter festgelegte Prozessnummernliste wird verwendet, um mehrere Kommunikationsgruppen aufzuteilen.

        :param rankL: Prozessgruppenliste.
        :return: eine Liste, die ``torch.distributed.ProcessGroup`` enthält

        Examples::

            from pyvqnet.distributed import get_local_rank,CommController
            import pyvqnet
            import numpy as np
            from pyvqnet.tensor import tensor
            pyvqnet.backends.set_backend("torch")
            import os
            import multiprocessing as mp


            def init_process(rank, size):
                """ Initialize the distributed environment. """
                os.environ['MASTER_ADDR'] = '127.0.0.1'
                os.environ['MASTER_PORT'] = '29500'
                os.environ['LOCAL_RANK'] = f"{rank}"
                Comm_OP = CommController("gloo", rank=rank, world_size=size)

                group = Comm_OP.split_groups([[1,3]])

                num = tensor.to_tensor(np.random.rand(1, 5)+get_local_rank()*10)
                print(f"before rank {Comm_OP.getRank()}  {num}\n")
                
                Comm_OP.reduce_group(num, 1,"sum",group[0])
                print(f"after rank {Comm_OP.getRank()}  {num}\n")
                

            if __name__ == "__main__":
                world_size = 4
                processes = []
                mp.set_start_method("spawn")
                for rank in range(world_size):
                    p = mp.Process(target=init_process, args=(rank, world_size))
                    p.start()
                    processes.append(p)

                for p in processes:
                    p.join()
 


 
    .. py:method:: barrier()
        :no-index:

        Synchronisation verschiedener Prozesse.

        :return: Synchronisationsoperation.

        Examples::

            from pyvqnet.distributed import CommController
            import pyvqnet
            pyvqnet.backends.set_backend("torch")
            import os
            import multiprocessing as mp


            def init_process(rank, size):
                """ Initialize the distributed environment. """
                os.environ['MASTER_ADDR'] = '127.0.0.1'
                os.environ['MASTER_PORT'] = '29500'
                os.environ['LOCAL_RANK'] = f"{rank}"
                pp = CommController("gloo", rank=rank, world_size=size)
                
                pp.barrier()


            if __name__ == "__main__":
                world_size = 4
                processes = []
                mp.set_start_method("spawn")
                for rank in range(world_size):
                    p = mp.Process(target=init_process, args=(rank, world_size))
                    p.start()
                    processes.append(p)

                for p in processes:
                    p.join()

    .. py:method:: allreduce(tensor, c_op = "avg")
        :no-index:

        Unterstützt Allreduce-Kommunikation auf Daten.

        :param tensor: Eingabedaten.
        :param c_op: Berechnungsmethode.

        Examples::

            from pyvqnet.distributed import get_local_rank,CommController
            import pyvqnet
            import numpy as np
            from pyvqnet.tensor import tensor
            pyvqnet.backends.set_backend("torch")
            import os
            import multiprocessing as mp


            def init_process(rank, size):
                """ Initialize the distributed environment. """
                os.environ['MASTER_ADDR'] = '127.0.0.1'
                os.environ['MASTER_PORT'] = '29500'
                os.environ['LOCAL_RANK'] = f"{rank}"
                Comm_OP = CommController("gloo", rank=rank, world_size=size)

                num = tensor.to_tensor(np.random.rand(1, 5))
                print(f"rank {Comm_OP.getRank()}  {num}")

                Comm_OP.all_reduce(num, "sum")
                print(f"rank {Comm_OP.getRank()}  {num}")

            if __name__ == "__main__":
                world_size = 2
                processes = []
                mp.set_start_method("spawn")
                for rank in range(world_size):
                    p = mp.Process(target=init_process, args=(rank, world_size))
                    p.start()
                    processes.append(p)

                for p in processes:
                    p.join()

    .. py:method:: reduce(tensor, root = 0, c_op = "avg")
        :no-index:

        Unterstützt Reduce-Kommunikation auf Daten.

        :param tensor: Eingabedaten.
        :param root: Gibt den Knoten an, an den die Daten zurückgegeben werden.
        :param c_op: Berechnungsmethode.

        Examples::

            from pyvqnet.distributed import get_local_rank,CommController
            import pyvqnet
            import numpy as np
            from pyvqnet.tensor import tensor
            pyvqnet.backends.set_backend("torch")
            import os
            import multiprocessing as mp


            def init_process(rank, size):
                """ Initialize the distributed environment. """
                os.environ['MASTER_ADDR'] = '127.0.0.1'
                os.environ['MASTER_PORT'] = '29500'
                os.environ['LOCAL_RANK'] = f"{rank}"
                Comm_OP = CommController("gloo", rank=rank, world_size=size)

                num = tensor.to_tensor(np.random.rand(1, 5))
                print(f"before rank {Comm_OP.getRank()}  {num}")
                
                Comm_OP.reduce(num, 1,"sum")
                print(f"after rank {Comm_OP.getRank()}  {num}")
                

            if __name__ == "__main__":
                world_size = 3
                processes = []
                mp.set_start_method("spawn")
                for rank in range(world_size):
                    p = mp.Process(target=init_process, args=(rank, world_size))
                    p.start()
                    processes.append(p)

                for p in processes:
                    p.join()
 
 
    .. py:method:: broadcast(tensor, root = 0)
        :no-index:

        Sendet die Daten auf dem angegebenen Prozess-root an alle Prozesse.

        :param tensor: Input data.
        :param root: Der angegebene Knoten.

        Examples::

            from pyvqnet.distributed import get_local_rank,CommController
            import pyvqnet
            import numpy as np
            from pyvqnet.tensor import tensor
            pyvqnet.backends.set_backend("torch")
            import os
            import multiprocessing as mp


            def init_process(rank, size):
                """ Initialize the distributed environment. """
                os.environ['MASTER_ADDR'] = '127.0.0.1'
                os.environ['MASTER_PORT'] = '29500'
                os.environ['LOCAL_RANK'] = f"{rank}"
                Comm_OP = CommController("gloo", rank=rank, world_size=size)


                num = tensor.to_tensor(np.random.rand(1, 5))+ rank
                print(f"before rank {Comm_OP.getRank()}  {num}")
                
                Comm_OP.broadcast(num, 1)
                print(f"after rank {Comm_OP.getRank()}  {num}")
                

            if __name__ == "__main__":
                world_size = 3
                processes = []
                mp.set_start_method("spawn")
                for rank in range(world_size):
                    p = mp.Process(target=init_process, args=(rank, world_size))
                    p.start()
                    processes.append(p)

                for p in processes:
                    p.join()


    .. py:method:: allgather(tensor)
        :no-index:

        Gather all the data from all processes together. This interface only supports the nccl backend.

        :param tensor: Input data.

        Examples::

            from pyvqnet.distributed import get_local_rank,CommController,get_world_size
            import pyvqnet
            import numpy as np
            from pyvqnet.tensor import tensor
            pyvqnet.backends.set_backend("torch")
            import os
            import multiprocessing as mp


            def init_process(rank, size):
                """ Initialize the distributed environment. """
                os.environ['MASTER_ADDR'] = '127.0.0.1'
                os.environ['MASTER_PORT'] = '29500'
                os.environ['LOCAL_RANK'] = f"{rank}"
                Comm_OP = CommController("nccl", rank=rank, world_size=size)

                num = tensor.QTensor(np.random.rand(5,4),device=pyvqnet.DEV_GPU_0+rank)
                print(f"before rank {Comm_OP.getRank()}  {num}\n")

                num = Comm_OP.all_gather(num)
                print(f"after rank {Comm_OP.getRank()}  {num}\n")


            if __name__ == "__main__":
                world_size = 2
                processes = []
                mp.set_start_method("spawn")
                for rank in range(world_size):
                    p = mp.Process(target=init_process, args=(rank, world_size))
                    p.start()
                    processes.append(p)

                for p in processes:
                    p.join()


    .. py:method:: send(tensor, dest)
        :no-index:

        p2p communication interface.

        :param tensor: input data.
        :param dest: destination process.

        Examples::

            from pyvqnet.distributed import get_rank,CommController,get_world_size
            import pyvqnet
            import numpy as np
            from pyvqnet.tensor import tensor
            pyvqnet.backends.set_backend("torch")
            import os
            import multiprocessing as mp


            def init_process(rank, size):
                """ Initialize the distributed environment. """
                os.environ['MASTER_ADDR'] = '127.0.0.1'
                os.environ['MASTER_PORT'] = '29500'
                os.environ['LOCAL_RANK'] = f"{rank}"
                Comm_OP = CommController("gloo", rank=rank, world_size=size)

                num = tensor.to_tensor(np.random.rand(1, 5))
                recv = tensor.zeros_like(num)
                if get_rank() == 0:
                    Comm_OP.send(num, 1)
                elif get_rank() == 1:
                    Comm_OP.recv(recv, 0)
                print(f"before rank {Comm_OP.getRank()}  {num}")
                print(f"after rank {Comm_OP.getRank()}  {recv}")

            if __name__ == "__main__":
                world_size = 2
                processes = []
                mp.set_start_method("spawn")
                for rank in range(world_size):
                    p = mp.Process(target=init_process, args=(rank, world_size))
                    p.start()
                    processes.append(p)

                for p in processes:
                    p.join()
 
 
    .. py:method:: recv(tensor, source)
        :no-index:

        p2p communication interface.

        :param tensor: input data.
        :param source: receiving process.

        Examples::

            from pyvqnet.distributed import get_rank,CommController,get_world_size
            import pyvqnet
            import numpy as np
            from pyvqnet.tensor import tensor
            pyvqnet.backends.set_backend("torch")
            import os
            import multiprocessing as mp


            def init_process(rank, size):
                """ Initialize the distributed environment. """
                os.environ['MASTER_ADDR'] = '127.0.0.1'
                os.environ['MASTER_PORT'] = '29500'
                os.environ['LOCAL_RANK'] = f"{rank}"
                Comm_OP = CommController("gloo", rank=rank, world_size=size)

                num = tensor.to_tensor(np.random.rand(1, 5))
                recv = tensor.zeros_like(num)
                if get_rank() == 0:
                    Comm_OP.send(num, 1)
                elif get_rank() == 1:
                    Comm_OP.recv(recv, 0)
                print(f"before rank {Comm_OP.getRank()}  {num}")
                print(f"after rank {Comm_OP.getRank()}  {recv}")

            if __name__ == "__main__":
                world_size = 2
                processes = []
                mp.set_start_method("spawn")
                for rank in range(world_size):
                    p = mp.Process(target=init_process, args=(rank, world_size))
                    p.start()
                    processes.append(p)

                for p in processes:
                    p.join()
            

    .. py:method:: allreduce_group(tensor, c_op = "avg", group = None)
        :no-index:

        Intra-group allreduce communication interface.

        :param tensor: Input data.
        :param c_op: Berechnungsmethode.
        :param group: Communication group generated from `split_groups` or `init_groups` .

        Examples::

            from pyvqnet.distributed import get_local_rank,CommController
            import pyvqnet
            import numpy as np
            from pyvqnet.tensor import tensor
            pyvqnet.backends.set_backend("torch")
            import os
            import multiprocessing as mp


            def init_process(rank, size):
                """ Initialize the distributed environment. """
                os.environ['MASTER_ADDR'] = '127.0.0.1'
                os.environ['MASTER_PORT'] = '29500'
                os.environ['LOCAL_RANK'] = f"{rank}"
                Comm_OP = CommController("gloo", rank=rank, world_size=size)

                rankL = [[0,1],[2,3]]
                groups = Comm_OP.split_groups(rankL)
                num = tensor.to_tensor(np.ones(5)+get_local_rank()*1000)

                print(f"before rank {Comm_OP.getRank()}  {num}")
                if Comm_OP.getRank() in rankL[0]:
                    Comm_OP.all_reduce_group(num, "sum", groups[0])

                    print(f"after rank {Comm_OP.getRank()}  {num}")

                if Comm_OP.getRank() in rankL[1]:
                    Comm_OP.all_reduce_group(num, "sum", groups[1])

                    print(f"after rank {Comm_OP.getRank()}  {num}")

            if __name__ == "__main__":
                world_size = 4
                mp.set_start_method("spawn")
                processes = []
                for rank in range(world_size):
                    p = mp.Process(target=init_process, args=(rank, world_size))
                    p.start()
                    processes.append(p)

                for p in processes:
                    p.join()



    .. py:method:: reduce_group(tensor, root = 0, c_op = "avg", group = None)
        :no-index:

        Intra-group reduce communication interface.

        :param tensor: Input data.
        :param root: Specify the process number.
        :param c_op: Berechnungsmethode.
        :param group: Communication group generated from `split_groups` or `init_groups` .

        Examples::

            from pyvqnet.distributed import get_local_rank,CommController
            import pyvqnet
            import numpy as np
            from pyvqnet.tensor import tensor
            pyvqnet.backends.set_backend("torch")
            import os
            import multiprocessing as mp


            def init_process(rank, size):
                """ Initialize the distributed environment. """
                os.environ['MASTER_ADDR'] = '127.0.0.1'
                os.environ['MASTER_PORT'] = '29500'
                os.environ['LOCAL_RANK'] = f"{rank}"
                Comm_OP = CommController("gloo", rank=rank, world_size=size)
                rankL = [[1,3],[0,2]]
                group = Comm_OP.split_groups([[1,3],[0,2]])

                num = tensor.to_tensor(np.random.rand(1, 5)+get_local_rank()*10)
                print(f"before rank {Comm_OP.getRank()}  {num}\n")
                if Comm_OP.getRank() in rankL[0]:
                    Comm_OP.reduce_group(num, rankL[0][1],"sum",group[0])
                    print(f"after rank {Comm_OP.getRank()}  {num}\n")
                if Comm_OP.getRank() in rankL[1]:
                    Comm_OP.reduce_group(num, rankL[1][1],"sum",group[1])
                    print(f"after rank {Comm_OP.getRank()}  {num}\n")

            if __name__ == "__main__":
                world_size = 4
                processes = []
                mp.set_start_method("spawn")
                for rank in range(world_size):
                    p = mp.Process(target=init_process, args=(rank, world_size))
                    p.start()
                    processes.append(p)

                for p in processes:
                    p.join()

 
    .. py:method:: broadcast_group(tensor, root = 0, group = None)
        :no-index:

        Intra-group broadcast communication interface.

        :param tensor: Input data.
        :param root: Specify the process ID.
        :param group: Communication group generated from `split_groups` or `init_groups` .

        Examples::
            
            from pyvqnet.distributed import get_local_rank,CommController
            import pyvqnet
            import numpy as np
            from pyvqnet.tensor import tensor
            pyvqnet.backends.set_backend("torch")
            import os
            import multiprocessing as mp


            def init_process(rank, size):
                """ Initialize the distributed environment. """
                os.environ['MASTER_ADDR'] = '127.0.0.1'
                os.environ['MASTER_PORT'] = '29500'
                os.environ['LOCAL_RANK'] = f"{rank}"
                Comm_OP = CommController("gloo", rank=rank, world_size=size)

                rankL = [[2,3],[0,1,4]]
                group = Comm_OP.split_groups(rankL)

                num = tensor.to_tensor(np.random.rand(1, 5))+ rank*1000
                print(f"before rank {Comm_OP.getRank()}  {num}")
                
                if Comm_OP.getRank() in rankL[0]:
                    Comm_OP.broadcast_group(num, rankL[0][0],group[0])
                    print(f"after rank {Comm_OP.getRank()}  {num}")

                if Comm_OP.getRank() in rankL[1]:
                    Comm_OP.broadcast_group(num, rankL[1][1],group[1])
                    print(f"after rank {Comm_OP.getRank()}  {num}")

            if __name__ == "__main__":
                world_size = 5
                processes = []
                mp.set_start_method("spawn")
                for rank in range(world_size):
                    p = mp.Process(target=init_process, args=(rank, world_size))
                    p.start()
                    processes.append(p)

                for p in processes:
                    p.join()

 
    .. py:method:: allgather_group(tensor, group = None)
        :no-index:
        
        Allgather-Kommunikationsschnittstelle innerhalb der Gruppe.

        :param tensor: input data.
        :param group: Communication group generated from `split_groups` or `init_groups` .

        Examples::
            
            from pyvqnet.distributed import get_local_rank,CommController,get_world_size
            import pyvqnet
            import numpy as np
            from pyvqnet.tensor import tensor
            pyvqnet.backends.set_backend("torch")
            import os
            import multiprocessing as mp
            

            def init_process(rank, size ):
                """ Initialize the distributed environment. """
                os.environ['MASTER_ADDR'] = '127.0.0.1'
                os.environ['MASTER_PORT'] = '29500'
                os.environ['LOCAL_RANK'] = f"{rank}"
                Comm_OP = CommController("nccl", rank=rank, world_size=size)

                group = Comm_OP.split_groups([[0,1]])
                print(f"get_world_size {get_world_size()}")

                num = tensor.QTensor(np.random.rand(5,4)+get_local_rank()*100,device=pyvqnet.DEV_GPU_0+get_local_rank())
                print(f"before rank {Comm_OP.getRank()}  {num}\n")

                num = Comm_OP.allgather_group(num,group[0])
                print(f"after rank {Comm_OP.getRank()}  {num}\n")


            if __name__ == "__main__":
                world_size = 2
                processes = []
                mp.set_start_method("spawn")
                for rank in range(world_size):
                    p = mp.Process(target=init_process, args=(rank, world_size ))
                    p.start()
                    processes.append(p)

                for p in processes:
                    p.join()

