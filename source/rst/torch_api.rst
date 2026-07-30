

.. _torch_api:

=============================================================
VQNet utilizza torch per il calcolo a basso livello
=============================================================

A partire dalla versione 2.15.0, questo software supporta l'uso di `torch` come backend di calcolo per le operazioni a basso livello e puo essere integrato con modelli, codici e librerie di terze parti basate su `torch` per lo sviluppo secondario.

    .. important::

        Per utilizzare le seguenti funzionalita, installa torch>=2.11.0 autonomamente. Se installi una versione GPU di torch, devi usare una versione compatibile con CUDA 12.6, altrimenti il tuo torch potrebbe non funzionare a causa di problemi con la libreria runtime NVIDIA CUDA. Questo software non installa automaticamente torch durante l'installazione.

    .. note::

        Le funzioni di calcolo quantistico variazionale (con nome in minuscolo, come `rx`, `ry`, `rz`, ecc.) in :ref:`vqc_api`, cosi come le funzioni di calcolo base di QTensor in :ref:`qtensor_api`,
        possono accettare un `QTensor` come input dopo aver chiamato ``pyvqnet.backends.set_backend("torch")``, con il membro `data` di `QTensor` che passa dal Tensor di pyvqnet a ``torch.Tensor`` per il calcolo.

        ``pyvqnet.backends.set_backend("torch")`` e ``pyvqnet.backends.set_backend("pyvqnet")`` modificano il backend di calcolo globale.
        Gli oggetti ``QTensor`` creati con configurazioni di backend diverse non possono essere mescolati nei calcoli.

Configurazione Base del Backend
============================================

set_backend
------------------------------------------------

.. py:function:: pyvqnet.backends.set_backend(backend_name)

    Imposta il backend per i calcoli e l'archiviazione dei dati correnti. Il valore predefinito e "pyvqnet-ad", ma puo essere impostato su "torch", "torch-native", "pyvqnet-ad".
    
    Dopo aver chiamato ``pyvqnet.backends.set_backend("torch")``, l'interfaccia rimane invariata. La variabile membro ``data`` di ``QTensor`` di VQNet utilizza ``torch.Tensor`` per archiviare i dati.
    :ref:`qtensor_api`, :ref:`vqc_api` e le interfacce ``pyvqnet.nn.torch`` accettano ``QTensor`` come input e ``QTensor`` come output.

    Dopo aver chiamato ``pyvqnet.backends.set_backend("torch-native")``, le interfacce rimangono invariate: :ref:`qtensor_api`, :ref:`vqc_api` e l'interfaccia `pyvqnet.nn.torch`.
    Gli input possono accettare direttamente tipi ``torch.Tensor`` o ``QTensor`` e gli output sono ``torch.Tensor``, eliminando la necessita di conversione in ``QTensor``, riducendo cosi la conversione dei dati.
    
    Dopo aver chiamato ``pyvqnet.backends.set_backend("pyvqnet")``, il membro ``data`` di ``QTensor`` di VQNet archiviera i dati usando ``pyvqnet._core.Tensor`` e i calcoli utilizzeranno la libreria C++ di pyvqnet.

    Dopo aver chiamato ``pyvqnet.backends.set_backend("pyvqnet-ad")``, il membro ``data`` di ``QTensor`` di VQNet archiviera i dati usando ``pyvqnet._core.Tensor`` e i calcoli utilizzeranno la libreria C++ di pyvqnet con prestazioni migliorate.


    .. note::

        Questa funzione modifica il backend di calcolo corrente. Gli oggetti ``QTensor`` creati con backend diversi non possono essere utilizzati insieme nei calcoli.

    :param backend_name: Nome del backend, puo essere "pyvqnet" o "torch".

    Example::

        import pyvqnet
        pyvqnet.backends.set_backend("torch")

get_backend
-------------------------------

.. py:function:: pyvqnet.backends.get_backend(t=None)

    Se `t` e None, recupera il backend di calcolo corrente.
    Se `t` e un QTensor, restituisce il backend utilizzato per creare il QTensor in base alla sua proprieta ``data``.
    Se "torch" e il backend, restituisce il backend torchAPI di pyvqnet.
    Se "pyvqnet" e il backend, restituisce semplicemente "pyvqnet".
    
    :param t: Il tensore corrente (default: None).
    :return: Il backend. Per impostazione predefinita, restituisce "pyvqnet".

    Example::

        import pyvqnet
        pyvqnet.backends.set_backend("torch")
        pyvqnet.backends.get_backend()

Funzioni QTensor
===================

Dopo aver impostato il backend su ``torch``:

.. code-block::

    import pyvqnet
    pyvqnet.backends.set_backend("torch")

Tutte le funzioni membro, funzioni di creazione, funzioni matematiche, funzioni logiche, trasformazioni di matrici, ecc., in :ref:`qtensor_api` utilizzeranno torch per il calcolo. Si puo accedere a `QTensor.data` per recuperare i dati di torch.

Moduli di Rete Neurale Classica e Rete Neurale Quantistica Variazionale
==========================================================================================

Classe Base
------------------------------------------------

TorchModule
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. py:class:: pyvqnet.nn.torch.TorchModule(*args, **kwargs)

    La classe base che definisce i modelli quando si utilizza il backend `torch`. Questa classe eredita da ``pyvqnet.nn.Module`` e ``torch.nn.Module``.
    Puo essere aggiunta come sottomodulo a un torchmodel.

    .. note::

        Questa classe e le sue classi derivate sono adatte solo per l'uso con ``pyvqnet.backends.set_backend("torch")``.
        Non mescolare con il `Module` predefinito di ``pyvqnet.nn``.
    
        I dati in ``_buffers`` di questa classe sono di tipo ``torch.Tensor``.
        I dati in ``_parameters`` di questa classe sono di tipo ``torch.nn.Parameter``.

    .. py:method:: pyvqnet.nn.torch.TorchModule.forward(x, *args, **kwargs)

        Funzione astratta di calcolo forward per la classe TorchModule.

        :param x: QTensor di input.
        :param args: Argomenti variabili non nominati.
        :param kwargs: Argomenti variabili nominati.

        :return: QTensor di output, con `data` interno che e un ``torch.Tensor``.

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

        Restituisce un dizionario contenente l'intero stato del modulo, inclusi parametri e valori dei buffer.
        Le chiavi sono i nomi dei parametri e dei buffer corrispondenti.

        :param destination: Il dizionario in cui archiviare i parametri interni del modulo.
        :param prefix: Un prefisso utilizzato per i nomi dei parametri e dei buffer.

        :return: Un dizionario contenente l'intero stato del modulo.

        Example::

            from pyvqnet.nn.torch import Conv2D
            import pyvqnet
            pyvqnet.backends.set_backend("torch")
            test_conv = Conv2D(2,3,(3,3),(2,2),"same")
            print(test_conv.state_dict().keys())

    .. py:method:: pyvqnet.nn.torch.TorchModule.load_state_dict(state_dict, strict=True)

        Copia i parametri e i buffer da :attr:`state_dict` in questo modulo e nei suoi sottomoduli.

        :param state_dict: Un dizionario contenente parametri e buffer persistenti.
        :param strict: Se imporre che le chiavi in state_dict corrispondano a `state_dict()` del modello. Default: True.

        :return: Un messaggio di errore in caso di problemi.

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

        Sposta il modulo e i parametri dei sottomoduli e i dati dei buffer sul dispositivo GPU specificato.

        Il dispositivo specifica dove vengono archiviati i dati interni. Quando device >= DEV_GPU_0, i dati vengono archiviati sulla GPU.
        Se il tuo computer ha piu GPU, puoi specificare dispositivi diversi per archiviare i dati. Ad esempio, device = DEV_GPU_1, DEV_GPU_2, DEV_GPU_3, ... si riferisce all'archiviazione su GPU con numeri di serie diversi.
        
        .. note::

            I moduli non possono eseguire calcoli su GPU diverse.
            Se tenti di creare un QTensor su un ID GPU che supera il massimo consentito per la convalida, verra sollevato un errore Cuda.

        :param device: Il dispositivo su cui archiviare il QTensor. Default: DEV_GPU_0. device = pyvqnet.DEV_GPU_0 archivia sulla prima GPU, device = DEV_GPU_1 archivia sulla seconda GPU, e cosi via.
        :return: Il modulo spostato sul dispositivo GPU.

        Examples::

            from pyvqnet.nn.torch import ConvT2D
            import pyvqnet
            pyvqnet.backends.set_backend("torch")
            test_conv = ConvT2D(3, 2, [4, 4], [2, 2], (0, 0))
            test_conv = test_conv.toGPU()
            print(test_conv.backend)
            #1000

    .. py:method:: pyvqnet.torch.TorchModule.toCPU()

        Sposta il modulo e i parametri dei sottomoduli e i dati dei buffer su un dispositivo CPU specifico.

        :return: Il modulo spostato sul dispositivo CPU.

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

    Questo modulo viene utilizzato per archiviare istanze figlie di ``TorchModule`` in una lista. TorchModuleList puo essere indicizzata come una lista Python normale e i parametri interni che contiene possono essere salvati.
    
    Questa classe eredita da ``pyvqnet.nn.torch.TorchModule`` e ``pyvqnet.nn.ModuleList`` e puo essere aggiunta come sottomodulo a un torchmodel.

    :param modules: Una lista di ``pyvqnet.nn.torch.TorchModule``

    :return: Una classe TorchModuleList

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

    Questo modulo viene utilizzato per archiviare istanze figlie di ``pyvqnet.nn.Parameter`` in una lista. TorchParameterList puo essere indicizzata come una lista Python normale e i parametri interni che contiene possono essere salvati.
    
    Questa classe eredita da ``pyvqnet.nn.torch.TorchModule`` e ``pyvqnet.nn.ParameterList`` e puo essere aggiunta come sottomodulo a un torchmodel.

    :param value: Una lista di ``nn.Parameter``

    :return: Una classe TorchParameterList

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

    Il modulo aggiunge i moduli nell'ordine in cui vengono passati. In alternativa, puoi passare un ``OrderedDict`` di moduli. Il metodo ``forward()`` della classe ``Sequential`` accetta qualsiasi input e lo inoltra al suo primo modulo.
    L'output viene quindi collegato sequenzialmente all'input di ciascun modulo successivo, con l'output finale che e il risultato dell'ultimo modulo.

    Questa classe eredita da ``pyvqnet.nn.torch.TorchModule`` e ``pyvqnet.nn.Sequential`` e puo essere aggiunta come sottomodulo a un torchmodel.

    :param args: Moduli da aggiungere

    :return: Una classe TorchSequential

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

Salvataggio e Caricamento dei Parametri del Modello
---------------------------------------------------

Puoi usare ``save_parameters`` e ``load_parameters`` di :ref:`save_parameters` per salvare i parametri di un modello ``TorchModule`` come dizionario in un file, con i valori salvati come `numpy.ndarray`. In alternativa, puoi caricare il file dei parametri dal disco. Nota che la struttura del modello non viene salvata nel file e dovrai ricostruire manualmente la struttura del modello. Puoi anche usare direttamente ``torch.save`` e ``torch.load`` per leggere i parametri del modello ``torch`` poiche i parametri di ``TorchModule`` sono archiviati come oggetti ``torch.Tensor``.


Moduli di Rete Neurale Classica
--------------------------------------------

I seguenti moduli di rete neurale classica ereditano tutti da ``pyvqnet.nn.Module`` e ``torch.nn.Module`` e possono essere aggiunti come sottomoduli a un torchmodel.

Linear
^^^^^^^^^^^^^^^^^^^^^^^^

.. py:class:: pyvqnet.nn.torch.Linear(input_channels, output_channels, weight_initializer=None, bias_initializer=None, use_bias=True, dtype=None, name: str = "")

    Un modulo lineare (strato completamente connesso), :math:`y = x@A.T + b`.
    Questa classe eredita da ``pyvqnet.nn.Module`` e ``torch.nn.Module`` e puo essere utilizzata come sottomodulo di un torchmodel.

    I dati in ``_buffers`` della classe sono di tipo ``torch.Tensor``.
    I dati in ``_parameters`` della classe sono di tipo ``torch.nn.Parameter``.

    :param input_channels: `int` - Il numero di canali di input.
    :param output_channels: `int` - Il numero di canali di output.
    :param weight_initializer: `callable` - Funzione di inizializzazione dei pesi, default vuoto, utilizza he_uniform.
    :param bias_initializer: `callable` - Funzione di inizializzazione del bias, default vuoto, utilizza he_uniform.
    :param use_bias: `bool` - Se utilizzare il termine di bias, default True.
    :param dtype: Tipo di dato per i parametri, default None, utilizza il tipo di dato predefinito `kfloat32`, che rappresenta numeri a virgola mobile a 32 bit.
    :param name: Il nome dello strato lineare, default "".

    :return: Un'istanza dello strato Linear.

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

    Esegue una convoluzione 1D sull'input. L'input del modulo Conv1D ha la forma (batch_size, input_channels, in_height).
    Questa classe eredita da ``pyvqnet.nn.Module`` e ``torch.nn.Module`` e puo essere utilizzata come sottomodulo di un torchmodel.

    I dati in ``_buffers`` della classe sono di tipo ``torch.Tensor``.
    I dati in ``_parameters`` della classe sono di tipo ``torch.nn.Parameter``.

    :param input_channels: `int` - Il numero di canali di input.
    :param output_channels: `int` - Il numero di canali di output.
    :param kernel_size: `int` - La dimensione del kernel di convoluzione. La forma del kernel e [output_channels, input_channels/group, kernel_size, 1].
    :param stride: `int` - Il passo, default 1.
    :param padding: `str|int` - Opzioni di padding, puo essere una stringa {'valid', 'same'} o un intero che specifica la quantita di padding da applicare all'input. Default "valid".
    :param use_bias: `bool` - Se utilizzare il termine di bias, default True.
    :param kernel_initializer: `callable` - Il metodo di inizializzazione del kernel di convoluzione. Default vuoto, utilizza kaiming_uniform.
    :param bias_initializer: `callable` - Il metodo di inizializzazione del bias. Default vuoto, utilizza kaiming_uniform.
    :param dilation_rate: `int` - La dimensione della dilatazione, default 1.
    :param group: `int` - Il numero di gruppi nella convoluzione raggruppata. Default 1.
    :param dtype: Tipo di dato per i parametri, default None, utilizza il tipo di dato predefinito `kfloat32`, che rappresenta numeri a virgola mobile a 32 bit.
    :param name: Il nome del modulo, default "".

    :return: Un'istanza della convoluzione 1D.

    .. note::

        ``padding='valid'`` non applica padding.

        ``padding='same'`` applica zero-padding all'input, con `out_height` dell'output uguale a `ceil(in_height / stride)` e non supporta `stride > 1`.

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

    Esegue una convoluzione 2D sull'input. L'input del modulo Conv2D ha la forma (batch_size, input_channels, height, width).
    Questa classe eredita da ``pyvqnet.nn.Module`` e ``torch.nn.Module`` e puo essere utilizzata come sottomodulo di un torchmodel.

    I dati in ``_buffers`` della classe sono di tipo ``torch.Tensor``.
    I dati in ``_parameters`` della classe sono di tipo ``torch.nn.Parameter``.

    :param input_channels: `int` - Il numero di canali di input.
    :param output_channels: `int` - Il numero di canali di output.
    :param kernel_size: `tuple|list` - La dimensione del kernel di convoluzione. La forma del kernel e [output_channels, input_channels/group, kernel_size, kernel_size].
    :param stride: `tuple|list` - Il passo, default (1, 1).
    :param padding: `str|tuple` - Opzioni di padding, puo essere una stringa {'valid', 'same'} o una tupla che specifica il padding da applicare su entrambi i lati. Default "valid".
    :param use_bias: `bool` - Se utilizzare il termine di bias, default True.
    :param kernel_initializer: `callable` - Il metodo di inizializzazione del kernel di convoluzione. Default vuoto, utilizza kaiming_uniform.
    :param bias_initializer: `callable` - Il metodo di inizializzazione del bias. Default vuoto, utilizza kaiming_uniform.
    :param dilation_rate: `int` - La dimensione della dilatazione, default 1.
    :param group: `int` - Il numero di gruppi nella convoluzione raggruppata. Default 1.
    :param dtype: Tipo di dato per i parametri, default None, utilizza il tipo di dato predefinito `kfloat32`, che rappresenta numeri a virgola mobile a 32 bit.
    :param name: Il nome del modulo, default "".

    :return: Un'istanza della convoluzione 2D.

    .. note::

        ``padding='valid'`` non applica padding.

        ``padding='same'`` applica zero-padding all'input, con l'altezza dell'output uguale a `ceil(in_height / stride)`.

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

    Esegue una convoluzione trasposta 2D sull'input. L'input del modulo ConvT2D ha la forma (batch_size, input_channels, height, width).
    Questa classe eredita da ``pyvqnet.nn.Module`` e ``torch.nn.Module`` e puo essere utilizzata come sottomodulo di un torchmodel.

    I dati in ``_buffers`` della classe sono di tipo ``torch.Tensor``.
    I dati in ``_parameters`` della classe sono di tipo ``torch.nn.Parameter``.

    :param input_channels: `int` - Il numero di canali di input.
    :param output_channels: `int` - Il numero di canali di output.
    :param kernel_size: `tuple|list` - La dimensione del kernel di convoluzione, con forma del kernel = [input_channels, output_channels/group, kernel_size, kernel_size].
    :param stride: `tuple|list` - Il passo, default (1, 1).
    :param padding: `tuple` - Opzioni di padding, una tupla che specifica il padding da applicare su entrambi i lati. Default (0, 0).
    :param use_bias: `bool` - Se utilizzare il termine di bias, default True.
    :param kernel_initializer: `callable` - Il metodo di inizializzazione del kernel di convoluzione. Default vuoto, utilizza kaiming_uniform.
    :param bias_initializer: `callable` - Il metodo di inizializzazione del bias. Default vuoto, utilizza kaiming_uniform.
    :param dilation_rate: `int` - La dimensione della dilatazione, default 1.
    :param out_padding: Dimensione extra aggiunta alla forma dell'output per ogni dimensione. Default (0, 0).
    :param group: `int` - Il numero di gruppi nella convoluzione raggruppata. Default 1.
    :param dtype: Tipo di dato per i parametri, default None, utilizza il tipo di dato predefinito `kfloat32`, che rappresenta numeri a virgola mobile a 32 bit.
    :param name: Il nome del modulo, default "".

    :return: Un'istanza della convoluzione trasposta 2D.

    .. note::

        ``padding='valid'`` non applica padding.

        ``padding='same'`` applica zero-padding all'input, con l'altezza dell'output uguale a `ceil(height / stride)`.

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

    Esegue il pooling medio su input 1D. L'input ha la forma (batch_size, input_channels, in_height).
    Questa classe eredita da ``pyvqnet.nn.Module`` e ``torch.nn.Module`` e puo essere aggiunta come sottomodulo a un torchmodel.

    I dati in ``_buffers`` della classe sono di tipo ``torch.Tensor``.
    I dati in ``_parameters`` della classe sono di tipo ``torch.nn.Parameter``.

    :param kernel: La dimensione della finestra di pooling.
    :param stride: La dimensione del passo per lo spostamento della finestra.
    :param padding: Opzione di padding, un intero che specifica la lunghezza del padding. Default 0.
    :param name: Il nome del modulo, default "".

    :return: Un'istanza dello strato di pooling medio 1D.

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

    Esegue il max pooling su input 1D. L'input ha la forma (batch_size, input_channels, in_height).
    Questa classe eredita da ``pyvqnet.nn.Module`` e ``torch.nn.Module`` e puo essere aggiunta come sottomodulo a un torchmodel.

    I dati in ``_buffers`` della classe sono di tipo ``torch.Tensor``.
    I dati in ``_parameters`` della classe sono di tipo ``torch.nn.Parameter``.

    :param kernel: La dimensione della finestra di pooling.
    :param stride: La dimensione del passo per lo spostamento della finestra.
    :param padding: Opzione di padding, un intero che specifica la lunghezza del padding. Default 0.
    :param name: Il nome del modulo, default "".

    :return: Un'istanza dello strato di max pooling 1D.

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

    Esegue il pooling medio su input 2D. L'input ha la forma (batch_size, input_channels, height, width).
    Questa classe eredita da ``pyvqnet.nn.Module`` e ``torch.nn.Module`` e puo essere aggiunta come sottomodulo a un torchmodel.

    I dati in ``_buffers`` della classe sono di tipo ``torch.Tensor``.
    I dati in ``_parameters`` della classe sono di tipo ``torch.nn.Parameter``.

    :param kernel: La dimensione della finestra di pooling.
    :param stride: La dimensione del passo per lo spostamento della finestra.
    :param padding: Opzione di padding, una tupla contenente due interi che specificano il padding per entrambe le dimensioni. Default (0,0).
    :param name: Il nome del modulo, default "".

    :return: Un'istanza dello strato di pooling medio 2D.

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

    Esegue il max pooling su input 2D. L'input ha la forma (batch_size, input_channels, height, width).
    Questa classe eredita da ``pyvqnet.nn.Module`` e ``torch.nn.Module`` e puo essere aggiunta come sottomodulo a un torchmodel.

    I dati in ``_buffers`` della classe sono di tipo ``torch.Tensor``.
    I dati in ``_parameters`` della classe sono di tipo ``torch.nn.Parameter``.

    :param kernel: La dimensione della finestra di pooling.
    :param stride: La dimensione del passo per lo spostamento della finestra.
    :param padding: Opzione di padding, una tupla contenente due interi che specificano il padding per entrambe le dimensioni. Default (0,0).
    :param name: Il nome del modulo, default "".

    :return: Un'istanza dello strato di max pooling 2D.

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

    Questo modulo viene tipicamente utilizzato per archiviare embedding di parole e recuperarli usando indici. L'input del modulo e una lista di indici e l'output sono i corrispondenti embedding di parole.
    L'input di questo strato deve essere di tipo `kint64`. 
    Questa classe eredita da ``pyvqnet.nn.Module`` e ``torch.nn.Module`` e puo essere aggiunta come sottomodulo a un torchmodel.

    I dati in ``_buffers`` della classe sono di tipo ``torch.Tensor``.
    I dati in ``_parameters`` della classe sono di tipo ``torch.nn.Parameter``.

    :param num_embeddings: `int` - La dimensione del dizionario di embedding.
    :param embedding_dim: `int` - La dimensione di ogni vettore di embedding.
    :param weight_initializer: `callable` - Il metodo di inizializzazione dei pesi, default Xavier Normal.
    :param dtype: Il tipo di dato per i parametri, default None, utilizza il tipo di dato predefinito: `kfloat32` (virgola mobile a 32 bit).
    :param name: Il nome dello strato di embedding, default "".

    :return: Un'istanza dello strato Embedding.

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

    Applica la normalizzazione batch su input 4D (B, C, H, W). Fare riferimento all'articolo
    `Batch Normalization: Accelerating Deep Network Training by Reducing
    Internal Covariate Shift <https://arxiv.org/abs/1502.03167>`__.

    Questa classe eredita da ``pyvqnet.nn.Module`` e ``torch.nn.Module`` e puo essere aggiunta come sottomodulo a un torchmodel.

    I dati in ``_buffers`` della classe sono di tipo ``torch.Tensor``.
    I dati in ``_parameters`` della classe sono di tipo ``torch.nn.Parameter``.

    .. math::
        y = \frac{x - \mathrm{E}[x]}{\sqrt{\mathrm{Var}[x] + \epsilon}} * \gamma + \beta

    dove :math:`\gamma` e :math:`\beta` sono parametri addestrabili. Inoltre, per impostazione predefinita, durante l'addestramento, lo strato continua a stimare la media e la varianza, che vengono poi utilizzate per la normalizzazione durante la valutazione. Il momentum per le medie mobili e impostato al valore predefinito di 0.1.

    :param channel_num: `int` - Il numero di canali di input.
    :param momentum: `float` - Momentum per il calcolo della media mobile, default 0.1.
    :param epsilon: `float` - Una piccola costante per la stabilita numerica, default 1e-5.
    :param affine: `bool` - Se includere parametri affini apprendibili per ogni canale. Default `True`, che inizializza i parametri come 1 per i pesi e 0 per i bias.
    :param beta_initializer: `callable` - Il metodo di inizializzazione per beta, default inizializzazione a zero.
    :param gamma_initializer: `callable` - Il metodo di inizializzazione per gamma, default inizializzazione a uno.
    :param dtype: Il tipo di dato per i parametri, default None, utilizza `kfloat32` (virgola mobile a 32 bit).
    :param name: Il nome dello strato di normalizzazione batch, default "".

    :return: Un'istanza dello strato di normalizzazione batch 2D.

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

    Applica la normalizzazione batch su input 2D (B, C). Fare riferimento all'articolo
    `Batch Normalization: Accelerating Deep Network Training by Reducing
    Internal Covariate Shift <https://arxiv.org/abs/1502.03167>`__.

    .. math::
        y = \frac{x - \mathrm{E}[x]}{\sqrt{\mathrm{Var}[x] + \epsilon}} * \gamma + \beta

    dove :math:`\gamma` e :math:`\beta` sono parametri addestrabili. Inoltre, per impostazione predefinita, durante l'addestramento, lo strato continua a stimare la media e la varianza, che vengono poi utilizzate per la normalizzazione durante la valutazione. Il momentum per le medie mobili e impostato al valore predefinito di 0.1.

    Questa classe eredita da ``pyvqnet.nn.Module`` e ``torch.nn.Module`` e puo essere aggiunta come sottomodulo a un torchmodel.

    I dati in ``_buffers`` della classe sono di tipo ``torch.Tensor``.
    I dati in ``_parameters`` della classe sono di tipo ``torch.nn.Parameter``.

    :param channel_num: `int` - Il numero di canali di input.
    :param momentum: `float` - Momentum per il calcolo della media mobile, default 0.1.
    :param epsilon: `float` - Una piccola costante per la stabilita numerica, default 1e-5.
    :param affine: `bool` - Se includere parametri affini apprendibili per ogni canale. Default `True`, che inizializza i parametri come 1 per i pesi e 0 per i bias.
    :param beta_initializer: `callable` - Il metodo di inizializzazione per beta, default inizializzazione a zero.
    :param gamma_initializer: `callable` - Il metodo di inizializzazione per gamma, default inizializzazione a uno.
    :param dtype: Il tipo di dato per i parametri, default None, utilizza `kfloat32` (virgola mobile a 32 bit).
    :param name: Il nome dello strato di normalizzazione batch, default "".

    :return: Un'istanza dello strato di normalizzazione batch 1D.

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

    Applica la normalizzazione layer sulle ultime D dimensioni di qualsiasi input. Il metodo specifico e descritto nell'articolo:
    `Layer Normalization <https://arxiv.org/abs/1607.06450>`__.

    .. math::
        y = \frac{x - \mathrm{E}[x]}{ \sqrt{\mathrm{Var}[x] + \epsilon}} * \gamma + \beta

    Per input come (B, C, H, W, D), ``norm_shape`` puo essere [C, H, W, D], [H, W, D], [W, D] o [D].

    Questa classe eredita da ``pyvqnet.nn.Module`` e ``torch.nn.Module`` e puo essere aggiunta come sottomodulo a un torchmodel.

    I dati in ``_buffers`` della classe sono di tipo ``torch.Tensor``.
    I dati in ``_parameters`` della classe sono di tipo ``torch.nn.Parameter``.

    :param norm_shape: `list` - La forma da normalizzare.
    :param epsilon: `float` - Una piccola costante per la stabilita numerica, default 1e-5.
    :param affine: `bool` - Se `True`, questo modulo ha parametri affini apprendibili per ogni canale, inizializzati a 1 (per i pesi) e 0 (per i bias). Default `True`.
    :param dtype: Il tipo di dato per i parametri, default None, utilizza `kfloat32` (virgola mobile a 32 bit).
    :param name: Il nome del modulo, default "".

    :return: Un'istanza della classe LayerNormNd.

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

    Applica la normalizzazione layer su input 4D. Il metodo specifico e descritto nell'articolo:
    `Layer Normalization <https://arxiv.org/abs/1607.06450>`__.

    .. math::
        y = \frac{x - \mathrm{E}[x]}{ \sqrt{\mathrm{Var}[x] + \epsilon}} * \gamma + \beta

    La media e la deviazione standard vengono calcolate sulle dimensioni rimanenti, escludendo la prima. Per input come (B, C, H, W), ``norm_size`` dovrebbe essere uguale a C * H * W.

    Questa classe eredita da ``pyvqnet.nn.Module`` e ``torch.nn.Module`` e puo essere aggiunta come sottomodulo a un torchmodel.

    I dati in ``_buffers`` della classe sono di tipo ``torch.Tensor``.
    I dati in ``_parameters`` della classe sono di tipo ``torch.nn.Parameter``.

    :param norm_size: `int` - La dimensione della normalizzazione, dovrebbe essere uguale a C * H * W.
    :param epsilon: `float` - Una piccola costante per la stabilita numerica, default 1e-5.
    :param affine: `bool` - Se `True`, questo modulo ha parametri affini apprendibili per ogni canale, inizializzati a 1 (per i pesi) e 0 (per i bias). Default `True`.
    :param dtype: Il tipo di dato per i parametri, default None, utilizza `kfloat32` (virgola mobile a 32 bit).
    :param name: Il nome del modulo, default "".

    :return: Un'istanza della normalizzazione layer 2D.

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

    Applica la normalizzazione layer su input 2D. Il metodo specifico e descritto nell'articolo:
    `Layer Normalization <https://arxiv.org/abs/1607.06450>`__.

    .. math::
        y = \frac{x - \mathrm{E}[x]}{ \sqrt{\mathrm{Var}[x] + \epsilon}} * \gamma + \beta

    La media e la deviazione standard vengono calcolate sull'ultima dimensione, dove ``norm_size`` e il valore dell'ultima dimensione.

    Questa classe eredita da ``pyvqnet.nn.Module`` e ``torch.nn.Module`` e puo essere aggiunta come sottomodulo a un torchmodel.

    I dati in ``_buffers`` della classe sono di tipo ``torch.Tensor``.
    I dati in ``_parameters`` della classe sono di tipo ``torch.nn.Parameter``.

    :param norm_size: `int` - La dimensione della normalizzazione, dovrebbe essere uguale alla dimensione dell'ultima dimensione.
    :param epsilon: `float` - Una piccola costante per la stabilita numerica, default 1e-5.
    :param affine: `bool` - Se `True`, questo modulo ha parametri affini apprendibili per ogni canale, inizializzati a 1 (per i pesi) e 0 (per i bias). Default `True`.
    :param dtype: Il tipo di dato per i parametri, default None, utilizza `kfloat32` (virgola mobile a 32 bit).
    :param name: Il nome del modulo, default "".

    :return: Un'istanza della normalizzazione layer 1D.

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

    Applica la normalizzazione di gruppo su input di mini-batch. Input: :math:`(N, C, *)` dove :math:`C=\mathrm{num\_channels}`, Output: :math:`(N, C, *)`.

    Questo strato implementa l'operazione descritta nell'articolo `Group Normalization <https://arxiv.org/abs/1803.08494>`__.

    .. math::
        
        y = \frac{x - \mathrm{E}[x]}{ \sqrt{\mathrm{Var}[x] + \epsilon}} * \gamma + \beta

    I canali di input sono divisi in :attr:`num_groups` gruppi, ciascuno contenente ``num_channels / num_groups`` canali. :attr:`num_channels` deve essere divisibile per :attr:`num_groups`. La media e la deviazione standard vengono calcolate separatamente all'interno di ciascun gruppo. Se :attr:`affine` e ``True``, allora :math:`\gamma` e :math:`\beta` sono apprendibili. I parametri di trasformazione affine per ogni canale sono vettori di dimensione :attr:`num_channels`.

    Questa classe eredita da ``pyvqnet.nn.Module`` e ``torch.nn.Module`` e puo essere aggiunta come sottomodulo a un torchmodel.

    I dati in ``_buffers`` della classe sono di tipo ``torch.Tensor``.
    I dati in ``_parameters`` della classe sono di tipo ``torch.nn.Parameter``.

    :param num_groups (int): Il numero di gruppi in cui dividere i canali.
    :param num_channels (int): Il numero di canali di input previsti.
    :param epsilon: Un piccolo valore aggiunto al denominatore per la stabilita numerica. Default 1e-5.
    :param affine: Un valore booleano. Se impostato a ``True``, questo modulo ha parametri affini apprendibili per ogni canale, inizializzati a 1 (per i pesi) e 0 (per i bias). Default ``True``.
    :param dtype: Il tipo di dato per i parametri. Default None, utilizza `kfloat32` (virgola mobile a 32 bit).
    :param name: Il nome del modulo. Default "".

    :return: Un'istanza della classe GroupNorm.

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

    Modulo Dropout. Il modulo dropout imposta casualmente a zero l'output di alcune unita, ridimensionando le unita rimanenti in base alla probabilita dropout_rate.

    Questa classe eredita da ``pyvqnet.nn.Module`` e ``torch.nn.Module`` e puo essere aggiunta come sottomodulo a un torchmodel.

    :param dropout_rate: `float` - La probabilita di impostare i neuroni a zero.
    :param name: Il nome del modulo. Default "".

    :return: Un'istanza della classe Dropout.

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

    Il modulo DropPath applica un dropout casuale del percorso (profondita casuale).

    Questa classe eredita da ``pyvqnet.nn.Module`` e ``torch.nn.Module`` e puo essere aggiunta come sottomodulo a un torchmodel.

    :param dropout_rate: `float` - La probabilita di impostare i neuroni a zero.
    :param name: Il nome del modulo. Default "".

    :return: Un'istanza della classe DropPath.

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

    Riorganizza un tensore di forma: (*, C * r^2, H, W) in un tensore di forma (*, C, H * r, W * r), dove r e il fattore di scala.

    Questa classe eredita da ``pyvqnet.nn.Module`` e ``torch.nn.Module`` e puo essere aggiunta come sottomodulo a un torchmodel.

    :param upscale_factors: Il fattore di scala per la trasformazione.
    :param name: Il nome del modulo. Default "".

    :return: Un'istanza del modulo Pixel_Shuffle.

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

    Inverte l'operazione Pixel_Shuffle riorganizzando gli elementi. Trasforma un tensore di forma (*, C, H * r, W * r) in (*, C * r^2, H, W), dove r e il fattore di ridimensionamento.

    Questa classe eredita da ``pyvqnet.nn.Module`` e ``torch.nn.Module`` e puo essere aggiunta come sottomodulo a un torchmodel.

    :param downscale_factors: Il fattore di ridimensionamento per la trasformazione.
    :param name: Il nome del modulo. Default "".

    :return: Un'istanza del modulo Pixel_Unshuffle.

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

    Modulo Gated Recurrent Unit (GRU). Supporta l'impilamento multi-strato e la configurazione bidirezionale. La formula per un GRU unidirezionale a singolo strato e:

    .. math::
        \begin{array}{ll}
            r_t = \sigma(W_{ir} x_t + b_{ir} + W_{hr} h_{(t-1)} + b_{hr}) \\
            z_t = \sigma(W_{iz} x_t + b_{iz} + W_{hz} h_{(t-1)} + b_{hz}) \\
            n_t = \tanh(W_{in} x_t + b_{in} + r_t * (W_{hn} h_{(t-1)}+ b_{hn})) \\
            h_t = (1 - z_t) * n_t + z_t * h_{(t-1)}
        \end{array}

    Questa classe eredita da ``pyvqnet.nn.Module`` e ``torch.nn.Module`` e puo essere aggiunta come sottomodulo a un torchmodel.

    I ``_buffers`` della classe contengono dati ``torch.Tensor`` e i ``_parameters`` della classe contengono dati ``torch.nn.Parameter``.

    :param input_size: La dimensione delle caratteristiche di input.
    :param hidden_size: La dimensione delle caratteristiche nascoste.
    :param num_layers: Il numero di strati GRU impilati, default: 1.
    :param batch_first: Se True, la forma dell'input e [batch_size, seq_len, feature_dim], se False, la forma e [seq_len, batch_size, feature_dim], default: True.
    :param use_bias: Se False, il modulo non utilizza termini di bias, default: True.
    :param bidirectional: Se True, rende il GRU bidirezionale, default: False.
    :param dtype: Il tipo di dato dei parametri, default None, utilizza il tipo di dato predefinito: kfloat32 (float a 32 bit).
    :param name: Il nome del modulo, default: "".

    :return: Un'istanza del modulo GRU.

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

    Modulo Recurrent Neural Network (RNN), che utilizza :math:`\tanh` o :math:`\text{ReLU}` come funzione di attivazione. Supporta configurazioni bidirezionali e multi-strato. La formula per un RNN unidirezionale a singolo strato e:

    .. math::
        h_t = \tanh(W_{ih} x_t + b_{ih} + W_{hh} h_{(t-1)} + b_{hh})

    Se :attr:`nonlinearity` e ``'relu'``, allora :math:`\text{ReLU}` sostituira :math:`\tanh`.

    Questa classe eredita da ``pyvqnet.nn.Module`` e ``torch.nn.Module`` e puo essere aggiunta come sottomodulo a un torchmodel.

    I ``_buffers`` della classe contengono dati ``torch.Tensor`` e i ``_parameters`` della classe contengono dati ``torch.nn.Parameter``.

    :param input_size: La dimensione delle caratteristiche di input.
    :param hidden_size: La dimensione delle caratteristiche nascoste.
    :param num_layers: Il numero di strati RNN impilati, default: 1.
    :param nonlinearity: La funzione di attivazione non lineare, default: ``'tanh'``.
    :param batch_first: Se True, la forma dell'input e [batch_size, seq_len, feature_dim], se False, la forma e [seq_len, batch_size, feature_dim], default: True.
    :param use_bias: Se False, il modulo non utilizza termini di bias, default: True.
    :param bidirectional: Se True, rende il RNN bidirezionale, default: False.
    :param dtype: The data type of the parameters, defaults to None, using the default data type: kfloat32 (32-bit float).
    :param name: Il nome del modulo, default: "".

    :return: An instance of the RNN module.

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

    Modulo Long Short-Term Memory (LSTM). Supporta LSTM bidirezionale e configurazioni LSTM multi-strato impilate. La formula per un LSTM unidirezionale a singolo strato e la seguente:

    .. math::
        \begin{array}{ll} \\
            i_t = \sigma(W_{ii} x_t + b_{ii} + W_{hi} h_{t-1} + b_{hi}) \\
            f_t = \sigma(W_{if} x_t + b_{if} + W_{hf} h_{t-1} + b_{hf}) \\
            g_t = \tanh(W_{ig} x_t + b_{ig} + W_{hg} h_{t-1} + b_{hg}) \\
            o_t = \sigma(W_{io} x_t + b_{io} + W_{ho} h_{t-1} + b_{ho}) \\
            c_t = f_t \odot c_{t-1} + i_t \odot g_t \\
            h_t = o_t \odot \tanh(c_t) \\
        \end{array}

    Questa classe eredita da ``pyvqnet.nn.Module`` e ``torch.nn.Module`` e puo essere aggiunta come sottomodulo a un torchmodel.

    I ``_buffers`` della classe contengono dati ``torch.Tensor`` e i ``_parameters`` della classe contengono dati ``torch.nn.Parameter``.

    :param input_size: La dimensione delle caratteristiche di input.
    :param hidden_size: La dimensione delle caratteristiche nascoste.
    :param num_layers: Il numero di strati LSTM impilati, default: 1.
    :param batch_first: Se True, la forma dell'input e [batch_size, seq_len, feature_dim], se False, la forma e [seq_len, batch_size, feature_dim], default: True.
    :param use_bias: Se False, il modulo non utilizza termini di bias, default: True.
    :param bidirectional: Se True, rende il LSTM bidirezionale, default: False.
    :param dtype: Il tipo di dato dei parametri, default None, utilizza il tipo di dato predefinito: kfloat32 (float a 32 bit).
    :param name: Il nome del modulo, default: "".

    :return: Un'istanza del modulo LSTM.

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

    Applica un GRU (Gated Recurrent Unit) RNN multi-strato a sequenze di input di lunghezza dinamica.

    Il primo input dovrebbe essere un input di sequenza batch con lunghezza variabile definita
    tramite una classe ``tensor.PackedSequence``.

    La classe ``tensor.PackedSequence`` puo essere costruita
    chiamando consecutivamente le funzioni: ``pad_sequence``, ``pack_pad_sequence``.

    Il primo output di Dynamic_GRU e anche una classe ``tensor.PackedSequence``,
    che puo essere decompressa in un QTensor normale usando ``tensor.pad_pack_sequence``.

    Per ogni elemento nella sequenza di input, ogni strato calcola la seguente formula:

    .. math::
        \begin{array}{ll}
        r_t = \sigma(W_{ir} x_t + b_{ir} + W_{hr} h_{(t-1)} + b_{hr}) \\
        z_t = \sigma(W_{iz} x_t + b_{iz} + W_{hz} h_{(t-1)} + b_{hz}) \\
        n_t = \tanh(W_{in} x_t + b_{in} + r_t * (W_{hn} h_{(t-1)}+ b_{hn})) \\
        h_t = (1 - z_t) * n_t + z_t * h_{(t-1)}
        \end{array}

    Questa classe eredita da ``pyvqnet.nn.Module`` e ``torch.nn.Module`` e puo essere aggiunta al modello torch come sottomodulo di ``torch.nn.Module``.

    I dati in ``_buffers`` di questa classe sono di tipo ``torch.Tensor``.

    I dati in ``_parameters`` di questa classe sono di tipo ``torch.nn.Parameter``.

    :param input_size: Dimensione delle caratteristiche di input.
    :param hidden_size: Dimensione delle caratteristiche nascoste.
    :param num_layers: Numero di strati del ciclo. Valore predefinito: 1
    :param batch_first: Se True, la forma dell'input e [batch size, sequence length, feature dimension]. Se False, la forma dell'input e [sequence length, batch size, feature dimension]. Valore predefinito: True.
    :param use_bias: Se False, i pesi di bias b_ih e b_hh non vengono utilizzati per questo strato. Valore predefinito: True.
    :param bidirectional: Se true, diventa un GRU bidirezionale. Valore predefinito: False.
    :param dtype: Il tipo di dato del parametro, default: None, utilizza il tipo di dato predefinito: kfloat32, che rappresenta numeri a virgola mobile a 32 bit.
    :param name: Il nome di questo modulo, default "".

    :return: Una classe Dynamic_GRU

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


    Applica una rete neurale ricorrente (RNN) a una sequenza di input di lunghezza dinamica.

    Il primo input dovrebbe essere un input di sequenza batch con lunghezza variabile definita
    tramite la classe ``tensor.PackedSequence``.

    La classe ``tensor.PackedSequence`` puo essere costruita
    chiamando consecutivamente le funzioni: ``pad_sequence``, ``pack_pad_sequence``.

    Il primo output di Dynamic_RNN e anche una classe ``tensor.PackedSequence``,
    che puo essere decompressa in un QTensor normale usando ``tensor.pad_pack_sequence``.

    Modulo RNN, che utilizza :math:`\tanh` o :math:`\text{ReLU}` come funzione di attivazione. Supporta configurazioni bidirezionali e multi-strato.
    La formula di calcolo per RNN unidirezionale a singolo strato e la seguente:

    .. math::
        h_t = \tanh(W_{ih} x_t + b_{ih} + W_{hh} h_{(t-1)} + b_{hh})

    Se :attr:`nonlinearity` e ``'relu'``, allora :math:`\text{ReLU}` sostituira :math:`\tanh`.

    Questa classe eredita da ``pyvqnet.nn.Module`` e ``torch.nn.Module`` e puo essere aggiunta al modello torch come sottomodulo di ``torch.nn.Module``.

    I dati in ``_buffers`` di questa classe sono di tipo ``torch.Tensor``.

    I dati in ``_parmeters`` di questa classe sono di tipo ``torch.nn.Parameter``.

    :param input_size: Dimensione delle caratteristiche di input.
    :param hidden_size: Dimensione delle caratteristiche nascoste.
    :param num_layers: Numero di strati RNN impilati, default: 1.
    :param nonlinearity: Funzione di attivazione non lineare, default ``'tanh'``.
    :param batch_first: Se True, la forma dell'input e [batch size, sequence length, feature dimension]. Se False, la forma dell'input e [sequence length, batch size, feature dimension], default True.
    :param use_bias: Se False, questo modulo non applica bias, default: True.
    :param bidirectional: Se True, diventa un RNN bidirezionale, default: False.
    :param dtype: Il tipo di dato del parametro, default: None, utilizza il tipo di dato predefinito: kfloat32, che rappresenta numeri a virgola mobile a 32 bit.
    :param name: Il nome di questo modulo, default "".

    :return: Istanza Dynamic_RNN

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


    Applica un LSTM (Long Short-Term Memory) RNN a sequenze di input di lunghezza dinamica.

    Il primo input dovrebbe essere un input di sequenza batch con lunghezza variabile definita
    tramite una classe ``tensor.PackedSequence``.

    La classe ``tensor.PackedSequence`` puo essere costruita
    chiamando consecutivamente le funzioni: ``pad_sequence``, ``pack_pad_sequence``.

    Il primo output di Dynamic_LSTM e anche una classe ``tensor.PackedSequence``,
    che puo essere decompressa in un QTensor normale usando ``tensor.pad_pack_sequence``.

    Modulo RNN, che utilizza :math:`\tanh` o :math:`\text{ReLU}` come funzione di attivazione. Supporta configurazioni bidirezionali e multi-strato.
    La formula di calcolo per LSTM unidirezionale a singolo strato e la seguente: 
    
    
    .. math::
        \begin{array}{ll} \\
            i_t = \sigma(W_{ii} x_t + b_{ii} + W_{hi} h_{t-1} + b_{hi}) \\
            f_t = \sigma(W_{if} x_t + b_{if} + W_{hf} h_{t-1} + b_{hf}) \\
            g_t = \tanh(W_{ig} x_t + b_{ig} + W_{hg} h_{t-1} + b_{hg}) \\
            o_t = \sigma(W_{io} x_t + b_{io} + W_{ho} h_{t-1} + b_{ho}) \\
            c_t = f_t \odot c_{t-1} + i_t \odot g_t \\
            h_t = o_t \odot \tanh(c_t) \\
        \end{array}

    Questa classe eredita da ``pyvqnet.nn.Module`` e ``torch.nn.Module`` e puo essere aggiunta al modello torch come sottomodulo di ``torch.nn.Module``.

    I dati in ``_buffers`` di questa classe sono di tipo ``torch.Tensor``.

    I dati in ``_parmeters`` di questa classe sono di tipo ``torch.nn.Parameter``.

    :param input_size: Dimensione delle caratteristiche di input.
    :param hidden_size: Dimensione delle caratteristiche nascoste.
    :param num_layers: Numero di strati LSTM impilati, default: 1.
    :param batch_first: Se True, la forma dell'input e [batch size, sequence length, feature dimension]. Se False, la forma dell'input e [sequence length, batch size, feature dimension], default True.
    :param use_bias: Se False, questo modulo non applica bias, default: True.
    :param bidirectional: Se True, diventa un LSTM bidirezionale, default: False.
    :param dtype: Il tipo di dato del parametro, default: None, utilizza il tipo di dato predefinito: kfloat32, che rappresenta numeri a virgola mobile a 32 bit.
    :param name: Il nome di questo modulo, default "".

    :return: Istanza Dynamic_LSTM

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

    Ridimensiona l'input in scala inferiore/superiore.

    Attualmente supporta solo dati di input 4D.

    La dimensione dell'input e interpretata come `B x C x H x W`.

    Le opzioni `mode` disponibili sono ``nearest``, ``bilinear``, ``bicubic``.

    Questa classe eredita da ``pyvqnet.nn.Module`` e ``torch.nn.Module`` e puo essere aggiunta al modello torch come sottomodulo di ``torch.nn.Module``.

    :param size: Dimensione di output, default None.
    :param scale_factor: Fattore di scala, default None.
    :param mode: Algoritmo utilizzato per l'upsampling ``nearest`` | ``bilinear`` | ``bicubic``.
    :param align_corners: Da un punto di vista geometrico, trattiamo i pixel dell'input e dell'output come quadrati invece che come punti. Se impostato a `true`, i tensori di input e output saranno allineati dai punti centrali dei loro pixel d'angolo. I punti centrali dei pixel d'angolo sono allineati e i valori dei pixel d'angolo vengono preservati. Se impostato a `false`, i tensori di input e output saranno allineati dai punti d'angolo dei loro pixel d'angolo e i valori dei pixel d'angolo vengono preservati. I punti d'angolo dei pixel d'angolo sono allineati e l'interpolazione utilizzera i valori di bordo per il padding. I valori fuori dai confini vengono riempiti, rendendo questa operazione indipendente dalla dimensione dell'input. Quando ``scale_factor`` rimane invariato. Funziona solo quando ``mode`` e ``bilinear``.
    :param recompute_scale_factor: Ricalcola il fattore di scala per l'uso nel calcolo dell'interpolazione. Quando ``scale_factor`` viene passato come argomento, verra utilizzato per calcolare la dimensione dell'output.
    :param name: Nome del modulo.

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

    Costruisce una classe che calcola l'attenzione scalata del prodotto scalare per i tensori query, key e value.

    Questa classe eredita da ``pyvqnet.nn.Module`` e ``torch.nn.Module`` e puo essere aggiunta al modello torch come sottomodulo di ``torch.nn.Module``.

    :param attn_mask: Maschera di attenzione; valore predefinito: None. La forma deve essere trasmissibile alla forma dei pesi di attenzione.
    :param dropout_p: Probabilita di dropout; valore predefinito: 0, se maggiore di 0.0, viene applicato dropout.
    :param scale: Fattore di scala applicato prima di softmax, valore predefinito: None.
    :param is_causal: valore predefinito: False, se impostato a true, la maschera di attenzione e una matrice triangolare inferiore quando la maschera e una matrice quadrata. Se sia attn_mask che is_causal sono impostati, viene sollevato un errore.
    :return: Una classe SDPA

    Examples::
    
        from pyvqnet.nn.torch import SDPA
        from pyvqnet import tensor
        import pyvqnet
        pyvqnet.backends.set_backend("torch")
        model = SDPA(tensor.QTensor([1.]))

   .. py:method:: forward(query,key,value)

        Esegue il calcolo forward.

        :param query: Il QTensor di input query.
        :param key: Il QTensor di input key.
        :param value: Il QTensor di input key.
        :return: Il QTensor restituito dal calcolo SDPA.
        
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

API delle Funzioni di Perdita
-----------------------------

MeanSquaredError
^^^^^^^^^^^^^^^^^^^^^^^^

.. py:class:: pyvqnet.nn.torch.MeanSquaredError(name="")

    Calcola l'errore quadratico medio tra l'input :math:`x` e il valore target :math:`y`.

    Se l'errore quadratico puo essere descritto dalla seguente funzione:

    .. math::
        \ell(x, y) = L = \{l_1,\dots,l_N\}^\top, \quad
        l_n = \left( x_n - y_n \right)^2,

    :math:`x` e :math:`y` sono QTensor di forme arbitrarie e l'errore quadratico medio del totale :math:`n` elementi viene calcolato come segue.

    .. math::
        \ell(x, y) =
        \operatorname{mean}(L)

    Questa classe eredita da ``pyvqnet.nn.Module`` e ``torch.nn.Module`` e puo essere aggiunta al modello torch come sottomodulo di ``torch.nn.Module``.

    :param name: Il nome di questo modulo, default "".
    :return: Un'istanza di errore RMS.

    Parametri richiesti per la funzione di calcolo forward dell'errore RMS:

        x: :math:`(N, *)` valore previsto, dove :math:`*` rappresenta qualsiasi dimensione.

        y: :math:`(N, *)`, valore target, un QTensor della stessa dimensione dell'input.

    .. note::

        Si prega di notare che, a differenza di framework come pytorch, nella funzione forward della seguente funzione MeanSquaredError, il primo parametro e il valore target e il secondo parametro e il valore previsto.


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

    Calcola la perdita media di entropia incrociata binaria tra il target e l'input.

    L'entropia incrociata binaria senza media e la seguente:

    .. math::
        \ell(x, y) = L = \{l_1,\dots,l_N\}^\top, \quad
        l_n = - w_n \left[ y_n \cdot \log x_n + (1 - y_n) \cdot \log (1 - x_n) \right],

    dove :math:`N` e la dimensione del batch.

    .. math::
        \ell(x, y) = \operatorname{mean}(L)

    Questa classe eredita da ``pyvqnet.nn.Module`` e ``torch.nn.Module`` e puo essere aggiunta ai modelli torch come sottomodulo di ``torch.nn.Module``.

    :param name: Il nome di questo modulo, default "".
    :return: Un'istanza di entropia incrociata binaria media.

    Parametri richiesti per la funzione di calcolo forward dell'errore di entropia incrociata binaria media:

        x: :math:`(N, *)` valore previsto, dove :math:`*` rappresenta qualsiasi dimensione.

        y: :math:`(N, *)`, valore target, un QTensor della stessa dimensione dell'input.

    .. note::

        Si prega di notare che, a differenza di framework come pytorch, nella funzione forward della funzione BinaryCrossEntropy, il primo parametro e il valore target e il secondo parametro e il valore previsto.
        
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

    Questa funzione di perdita combina LogSoftmax e NLLLoss per calcolare l'entropia incrociata categoriale media.

    La funzione di perdita viene calcolata come segue, dove class e l'etichetta di categoria corrispondente del valore target:

    .. math::
        \text{loss}(x, y) = -\log\left(\frac{\exp(x[class])}{\sum_j \exp(x[j])}\right)
        = -x[class] + \log\left(\sum_j \exp(x[j])\right)

    :param name: Il nome di questo modulo, default "".
    :return: L'istanza di entropia incrociata categoriale media.

    Parametri richiesti per la funzione di calcolo forward dell'errore:

        x: :math:`(N, *)` Valore previsto, dove :math:`*` indica qualsiasi dimensione.

        y: :math:`(N, *)`, valore target, un QTensor della stessa dimensione dell'input. Deve essere un intero a 64 bit, kint64.

    .. note::

        Si prega di notare che, a differenza di pytorch e altri framework, nella funzione forward di CategoricalCrossEntropy, il primo parametro e il valore target e il secondo parametro e il valore previsto.

        Questa classe eredita da ``pyvqnet.nn.Module`` e ``torch.nn.Module`` e puo essere aggiunta al modello torch come sottomodulo di ``torch.nn.Module``.

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

    Questa funzione di perdita combina LogSoftmax e NLLLoss per calcolare l'entropia incrociata di classificazione media e ha una maggiore stabilita numerica.

    La funzione di perdita viene calcolata come segue, dove class e l'etichetta di classificazione corrispondente del valore target:

    .. math::
        \text{loss}(x, y) = -\log\left(\frac{\exp(x[class])}{\sum_j \exp(x[j])}\right)
        = -x[class] + \log\left(\sum_j \exp(x[j])\right)

    :param name: Il nome di questo modulo, default "".
    :return: Un'istanza della funzione di perdita Softmax cross entropy

    Parametri richiesti per la funzione di calcolo forward dell'errore:

        x: :math:`(N, *)` valore previsto, dove :math:`*` indica qualsiasi dimensione.

        y: :math:`(N, *)`, valore target, un QTensor della stessa dimensione dell'input. Deve essere un intero a 64 bit, kint64.

    .. note::

        Si prega di notare che, a differenza di pytorch e altri framework, nella funzione forward di SoftmaxCrossEntropy, il primo parametro e il valore target e il secondo parametro e il valore previsto.

        Questa classe eredita da ``pyvqnet.nn.Module`` e ``torch.nn.Module`` e puo essere aggiunta al modello torch come sottomodulo di ``torch.nn.Module``.
        
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

    
    Perdita media di log-verosimiglianza negativa. Utile per problemi di classificazione con C classi.

    `x` e la probabilita di verosimiglianza fornita dal modello. La sua forma puo essere :math:`(N, C)` o :math:`(N, C, d_1, d_2, ..., d_K)`. `y` e il valore vero atteso della funzione di perdita, contenente indici di classe in :math:`[0, C-1]`.

    .. math::
        \ell(x, y) = L = \{l_1,\dots,l_N\}^\top, \quad
        l_n = -
        \sum_{n=1}^N \frac{1}{N}x_{n,y_n} \quad

    :param name: Il nome di questo modulo, default "".
    :return: Un'istanza della funzione di perdita NLL_Loss

    Parametri richiesti per la funzione di calcolo forward dell'errore:

        x: :math:`(N, *)`, il valore di previsione di output della funzione di perdita, che puo essere una variabile multidimensionale.

        y: :math:`(N, *)`, il valore target della funzione di perdita. Deve essere un intero a 64 bit, kint64.

    .. note::

        Si prega di notare che, a differenza di framework come pytorch, nella funzione forward di NLL_Loss, il primo parametro e il valore target e il secondo parametro e il valore previsto.

        Questa classe eredita da ``pyvqnet.nn.Module`` e ``torch.nn.Module`` e puo essere aggiunta al modello torch come sottomodulo di ``torch.nn.Module``.
            
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

    Questa funzione calcola la perdita di LogSoftmax e NLL_Loss insieme.

    `x` contiene l'output non normalizzato. La sua forma puo essere :math:`(C)`, :math:`(N, C)` bidimensionale o :math:`(N, C, d_1, d_2, ..., d_K)` multidimensionale.

    La formula della funzione di perdita e la seguente, dove class e l'etichetta di classe corrispondente del valore target:

    .. math::
        \text{loss}(x, y) = -\log\left(\frac{\exp(x[class])}{\sum_j \exp(x[j])}\right)
        = -x[class] + \log\left(\sum_j \exp(x[j])\right)

    :param name: Il nome di questo modulo, default "".
    :return: Un'istanza della funzione di perdita CrossEntropyLoss

    Parametri richiesti per la funzione di calcolo forward dell'errore:

        x: :math:`(N, *)`, l'output della funzione di perdita, che puo essere una variabile multidimensionale.

        y: :math:`(N, *)`, il valore vero atteso della funzione di perdita. Deve essere un intero a 64 bit, kint64.

    .. note::

        Si prega di notare che, a differenza di framework come pytorch, nella funzione forward di CrossEntropyLoss, il primo parametro e il valore target e il secondo parametro e il valore previsto.

        Questa classe eredita da ``pyvqnet.nn.Module`` e ``torch.nn.Module`` e puo essere aggiunta al modello torch come sottomodulo di ``torch.nn.Module``.

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


Funzioni di Attivazione
-----------------------

Sigmoid
^^^^^^^^^^^^^^^^^^^^^^^^

.. py:class:: pyvqnet.nn.torch.Sigmoid(name:str="")

    Strato della funzione di attivazione Sigmoid.

    .. math::
        \text{Sigmoid}(x) = \frac{1}{1 + \exp(-x)}

    Questa classe eredita da ``pyvqnet.nn.Module`` e ``torch.nn.Module`` e puo essere aggiunta al modello torch come sottomodulo di ``torch.nn.Module``.

    :param name: Il nome dello strato della funzione di attivazione, default "".

    :return: Un'istanza dello strato della funzione di attivazione Sigmoid.

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

    Softplus.

    .. math::
        \text{Softplus}(x) = \log(1 + \exp(x))

    Questa classe eredita da ``pyvqnet.nn.Module`` e ``torch.nn.Module`` e puo essere aggiunta al modello torch come sottomodulo di ``torch.nn.Module``.

    :param name: Il nome dello strato della funzione di attivazione, default "".

    :return: Un'istanza Softplus.

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


    Questa classe eredita da ``pyvqnet.nn.Module`` e ``torch.nn.Module`` e puo essere aggiunta al modello torch come sottomodulo di ``torch.nn.Module``.


    :param name: Il nome dello strato della funzione di attivazione, default "".

    :return: Un'istanza SoftSign.

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

    Softmax.

    .. math::
        \text{Softmax}(x_{i}) = \frac{\exp(x_i)}{\sum_j \exp(x_j)}


    Questa classe eredita da ``pyvqnet.nn.Module`` e ``torch.nn.Module`` e puo essere aggiunta al modello torch come sottomodulo di ``torch.nn.Module``.


    :param axis: La dimensione su cui calcolare (l'ultimo asse e -1), valore predefinito = -1.
    :param name: Il nome dello strato della funzione di attivazione, default "".

    :return: Un'istanza Softmax.

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

    HardSigmoid.

    .. math::
        \text{Hardsigmoid}(x) = \begin{cases}
            0 & \text{ if } x \le -3, \\
            1 & \text{ if } x \ge +3, \\
            x / 6 + 1 / 2 & \text{otherwise}
        \end{cases}


    Questa classe eredita da ``pyvqnet.nn.Module`` e ``torch.nn.Module`` e puo essere aggiunta al modello torch come sottomodulo di ``torch.nn.Module``.


    :param name: Il nome dello strato della funzione di attivazione, default "".

    :return: Un'istanza HardSigmoid.

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


    Questa classe eredita da ``pyvqnet.nn.Module`` e ``torch.nn.Module`` e puo essere aggiunta al modello torch come sottomodulo di ``torch.nn.Module``.


    :param name: Il nome dello strato della funzione di attivazione, default "".

    :return: Un'istanza ReLu.

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

    LeakyReLu.

    .. math::
        \text{LeakyRelu}(x) =
        \begin{cases}
        x, & \text{ if } x \geq 0 \\
        \alpha * x, & \text{ otherwise }
        \end{cases}


    Questa classe eredita da ``pyvqnet.nn.Module`` e ``torch.nn.Module`` e puo essere aggiunta al modello torch come sottomodulo di ``torch.nn.Module``.


    :param alpha: Coefficiente LeakyRelu, default: 0.01.
    :param name: Il nome dello strato della funzione di attivazione, default "".

    :return: Un'istanza di attivazione LeakyReLu.

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

    Quando il parametro di approssimazione e 'tanh', GELU viene stimato come segue:

    .. math:: \text{GELU}(x) = 0.5 * x * (1 + \text{Tanh}(\sqrt{2 / \pi} * (x + 0.044715 * x^3)))


    Questa classe eredita da ``pyvqnet.nn.Module`` e ``torch.nn.Module`` e puo essere aggiunta al modello torch come sottomodulo di ``torch.nn.Module``.


    :param approximate: Metodo di calcolo approssimato, default "tanh".
    :param name: Il nome dello strato della funzione di attivazione, default "".

    :return: Istanza di attivazione Gelu.

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

    Strato della funzione di attivazione ELU (Exponential Linear Unit).

    .. math::
        \text{ELU}(x) = \begin{cases}
        x, & \text{ if } x > 0\\
        \alpha * (\exp(x) - 1), & \text{ if } x \leq 0
        \end{cases}


    Questa classe eredita da ``pyvqnet.nn.Module`` e ``torch.nn.Module`` e puo essere aggiunta al modello torch come sottomodulo di ``torch.nn.Module``.



    :param alpha: Coefficiente ELU, default: 1.
    :param name: Il nome dello strato della funzione di attivazione, default "".

    :return: Istanza di attivazione ELU.

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

    Funzione di attivazione tangente iperbolica Tanh.

    .. math::
        \text{Tanh}(x) = \frac{\exp(x) - \exp(-x)} {\exp(x) + \exp(-x)}


    Questa classe eredita da ``pyvqnet.nn.Module`` e ``torch.nn.Module`` e puo essere aggiunta al modello torch come sottomodulo di ``torch.nn.Module``.



    :param name: Il nome dello strato della funzione di attivazione, default "".

    :return: Istanza di attivazione Tanh.

    Examples::

        from pyvqnet.nn.torch import Tanh
        from pyvqnet.tensor import QTensor
        import pyvqnet
        pyvqnet.backends.set_backend("torch")
        layer = Tanh()
        y = layer(QTensor([-1, 2.0, -3, 4.0]))



Modulo Optimizer
---------------------------------------------

Per i modelli classici e di circuito quantistico che ereditano da `TorchModule`, i parametri `model.paramters()` possono continuare a essere ottimizzati utilizzando ottimizzatori diversi da `Rotosolve` presenti in :ref:`Optimizer`.



Utilizzo di pyqpanda per eseguire circuiti quantistici variazionali
-------------------------------------------------------------------------

La seguente e l'interfaccia per l'addestramento di circuiti quantistici variazionali che utilizza pyqpanda e pyqpanda3 per il calcolo.

.. warning::

    La parte di calcolo quantistico del seguente TorchQpandaQuantumLayer utilizza pyqpanda2.

    A causa dei problemi di compatibilita tra pyqpanda2 e pyqpanda3, devi installare pyqpanda2 autonomamente, `pip install pyqpanda`

TorchQpandaQuantumLayer
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Se hai piu familiarita con la sintassi pyQPanda2, puoi utilizzare l'interfaccia TorchQpandaQuantumLayer, aggiungendo bit quantistici personalizzati ``qubits``, bit classici ``cbits`` e simulatore backend ``machine`` al parametro della funzione ``qprog_with_measure`` di TorchQpandaQuantumLayer.

.. py:class:: pyvqnet.qnn.vqc.torch.TorchQpandaQuantumLayer(qprog_with_measure,para_num,diff_method:str = "parameter_shift",delta:float = 0.01,dtype=None,name="")

    Modulo di calcolo astratto dello strato quantistico variazionale. Utilizza pyQPanda2 per simulare un circuito quantistico parametrizzato e ottenere i risultati di misurazione. Questo strato quantistico variazionale eredita il modulo di calcolo del gradiente del framework VQNet. Puo utilizzare il metodo di spostamento dei parametri per calcolare il gradiente dei parametri del circuito, addestrare modelli di circuito quantistico variazionale o incorporare circuiti quantistici variazionali in modelli ibridi quantistici e classici.

    :param qprog_with_measure: Funzioni di operazione e misurazione del circuito quantistico costruite con pyQPanda.
    :param para_num: `int` - numero di parametri.
    :param diff_method: Metodo per risolvere i gradienti dei parametri del circuito quantistico, "parameter shift" o "finite difference", default parameter shift.
    :param delta: \delta nel calcolo dei gradienti per differenza finita.
    :param dtype: Tipo di dato del parametro, default: None, utilizza il tipo di dato predefinito: kfloat32, che rappresenta numeri a virgola mobile a 32 bit.
    :param name: Il nome di questo modulo, default "".

    :return: Un modulo in grado di calcolare circuiti quantistici.

    .. note::

        qprog_with_measure e una funzione di circuito quantistico definita in pyQPanda2.

        Questa funzione deve contenere i seguenti parametri come input della funzione (anche se un parametro non viene effettivamente utilizzato), altrimenti non funzionera correttamente in questa funzione.

        Rispetto a QuantumLayer, nella funzione di esecuzione del circuito variazionale passata da questa interfaccia, l'utente deve creare manualmente bit quantistici e simulatori.

        Se qprog_with_measure richiede una misura quantistica, l'utente deve anche creare e allocare manualmente i cbits.

        L'uso della funzione di circuito quantistico qprog_with_measure (input, param, nqubits, ncbits) puo fare riferimento al seguente esempio.

        `input`: Input di dati classici unidimensionali. Se assente, inserire None.

        `param`: Input dei parametri del circuito quantistico variazionale unidimensionali da addestrare.

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

            m_machine = pq.CPUQVM()# outside
            m_machine.init_qvm()# outside
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

        #classic data as input
        input = QTensor([[1.0,2,3,4],[4,2,2,3],[3,3,2,2]],requires_grad=True)

        #forward circuits
        rlt = pqc(input)

        print(rlt)

        grad =  QTensor(np.ones(rlt.data.shape)*1000)
        #backward circuits
        rlt.backward(grad)

        print(pqc.m_para.grad)
        print(input.grad)


.. warning::

    La parte di calcolo quantistico delle seguenti interfacce TorchQcloud3QuantumLayer e TorchQpanda3QuantumLayer utilizza pyqpanda3.

    Se utilizzi la funzione QCloud in questo modulo, ci saranno errori durante l'importazione di pyqpanda2 nel codice o l'utilizzo delle interfacce dei pacchetti correlati a pyqpanda2 di pyvqnet.

TorchQcloud3QuantumLayer
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Quando installi l'ultima versione di pyqpanda3, puoi utilizzare questa interfaccia per definire un circuito variazionale e inviarlo al chip reale di originqc per l'esecuzione.

.. py:class:: pyvqnet.qnn.vqc.torch.TorchQcloud3QuantumLayer(origin_qprog_func, qcloud_token, para_num, pauli_str_dict=None, shots = 1000, initializer=None, dtype=None, name="", diff_method="parameter_shift", submit_kwargs={}, query_kwargs={})
    
    Un modulo di calcolo astratto per chip reali che utilizza originqc di pyqpanda3. Invia circuiti quantistici parametrizzati a chip reali e ottiene i risultati di misurazione.
    Se diff_method == "random_coordinate_descent", lo strato selezionera casualmente un singolo parametro per calcolare il gradiente e gli altri parametri rimarranno a zero. Riferimento: https://arxiv.org/abs/2311.00088

    .. note::

        qcloud_token e il token API che hai richiesto dalla piattaforma cloud.

        origin_qprog_func deve restituire dati di tipo pypqanda3.core.QProg. Se pauli_str_dict non e impostato, e necessario assicurarsi che la misura sia stata inserita nel QProg.

        origin_qprog_func deve essere nel seguente formato:

        origin_qprog_func(input,param)

        `input`: Input di dati classici 1D~2D. Nel caso 2D, la prima dimensione e la dimensione del batch.

        `param`: Input dei parametri da addestrare del circuito quantistico variazionale 1D.

    .. warning::

        Questa classe eredita da ``pyvqnet.nn.Module`` e ``torch.nn.Module`` e puo essere aggiunta al modello torch come sottomodulo di ``torch.nn.Module``.

        I dati in ``_buffers`` di questa classe sono di tipo ``torch.Tensor``.

        I dati in ``_parmeters`` di questa classe sono di tipo ``torch.nn.Parameter``.

    :param origin_qprog_func: La funzione di circuito quantistico variazionale costruita da QPanda, che deve restituire QProg.
    :param qcloud_token: `str` - Il tipo di macchina quantistica o token cloud per l'esecuzione.
    :param para_num: `int` - Il numero di parametri, il parametro e un QTensor di dimensione [para_num].
    :param pauli_str_dict: `dict|list` - Dizionario o lista di dizionari che rappresentano operatori di Pauli nei circuiti quantistici. Default "None", il che significa che vengono eseguite operazioni di misurazione. Se viene inserito un dizionario di operatori di Pauli, viene calcolata una singola aspettativa o aspettative multiple.
    :param shot: `int` - Numero di misurazioni. Il valore predefinito e 1000.
    :param initializer: Inizializzatore per i valori dei parametri. Il valore predefinito e "None", utilizzando una distribuzione normale 0~2*pi.
    :param dtype: Tipo di dato del parametro. Il valore predefinito e None, che significa utilizzare il tipo di dato predefinito pyvqnet.kfloat32.
    :param name: Il nome del modulo. Il valore predefinito e una stringa vuota.
    :param diff_method: Metodo di differenziazione per il calcolo del gradiente. Il valore predefinito e "parameter_shift", "random_coordinate_descent".
    :param submit_kwargs: Parametri chiave aggiuntivi per l'invio di circuiti quantistici, default: {"if_print_qcloud_log":False,"chip_id":"WK_C180","is_amend":True,"is_mapping":True,"is_optimization":True,"compile_level":3,"default_task_group_size":200,"test_qcloud_fake":False,"":"server_ip_address"}, quando test_qcloud_fake e impostato a True, simulazione CPUQVM locale.
    :param query_kwargs: Parametri chiave aggiuntivi per interrogare i risultati quantistici, default: {"timeout":2,"print_query_info":True,"sub_circuits_split_size":1}.
    :return: Un modulo in grado di calcolare circuiti quantistici.


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

Se hai piu familiarita con la sintassi pyQPanda3, puoi utilizzare l'interfaccia TorchQpanda3QuantumLayer.

.. py:class:: pyvqnet.qnn.vqc.torch.TorchQpanda3QuantumLayer(qprog_with_measure,para_num,diff_method:str = "parameter_shift",delta:float = 0.01,dtype=None,name="")

    Modulo di calcolo astratto dello strato quantistico variazionale. Utilizza pyQPanda3 per simulare un circuito quantistico parametrizzato e ottenere i risultati di misurazione. Questo strato quantistico variazionale eredita il modulo di calcolo del gradiente del framework VQNet. Puoi utilizzare il metodo di spostamento dei parametri per calcolare il gradiente dei parametri del circuito, addestrare il modello di circuito quantistico variazionale o incorporare il circuito quantistico variazionale in un modello ibrido quantistico e classico.

    :param qprog_with_measure: Funzioni di operazione e misurazione del circuito quantistico costruite con pyQPanda.
    :param para_num: `int` - numero di parametri.
    :param diff_method: Metodo per risolvere i gradienti dei parametri del circuito quantistico, "parameter shift" o "finite difference", default parameter shift.
    :param delta: \delta nel calcolo dei gradienti per differenza finita.
    :param dtype: Tipo di dato del parametro, default: None, utilizza il tipo di dato predefinito: kfloat32, che rappresenta numeri a virgola mobile a 32 bit.
    :param name: Il nome di questo modulo, default "".

    :return: Un modulo in grado di calcolare circuiti quantistici.

    .. note::

        qprog_with_measure e una funzione di circuito quantistico definita in pyQPanda.

        Questa funzione deve includere i seguenti parametri come input della funzione (anche se un parametro non viene effettivamente utilizzato), altrimenti non funzionera correttamente in questa funzione.

        L'uso della funzione di circuito quantistico qprog_with_measure (input,param,nqubits,ncbits) puo fare riferimento al seguente esempio.

        `input`: Input di dati classici unidimensionali. Se assente, inserire None.

        `param`: Input dei parametri da addestrare per il circuito quantistico variazionale unidimensionale.

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

            m_machine = pq.CPUQVM()# outside
        
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

        #classic data as input
        input = QTensor([[1.0,2,3,4],[4,2,2,3],[3,3,2,2]],requires_grad=True)

        #forward circuits
        rlt = pqc(input)

        print(rlt)

        grad =  QTensor(np.ones(rlt.data.shape)*1000)
        #backward circuits
        rlt.backward(grad)

        print(pqc.m_para.grad)
        print(input.grad)

Modulo di circuito quantistico variazionale e interfaccia basati su differenziazione automatica
-----------------------------------------------------------------------------------------------
Classe Base
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Scrivere un modello di circuito quantistico variazionale richiede l'ereditarieta da ``QModule``.

QModule
""""""""""""""""""

.. py:class:: pyvqnet.qnn.vqc.torch.QModule(name="")

    Quando l'utente utilizza il backend `torch`, definisce la classe base che il `Module` del modello di circuito quantistico variazionale deve ereditare.
    Questa classe eredita da ``pyvqnet.nn.torch.TorchModule`` e ``torch.nn.Module``.

    Questa classe puo essere aggiunta al modello torch come sottomodulo di ``torch.nn.Module``.

    .. note::

        Questa classe e le sue classi derivate sono applicabili solo a ``pyvqnet.backends.set_backend("torch")``, non mescolare con il ``Module`` sotto il ``pyvqnet.nn`` predefinito.

        I dati in ``_buffers`` di questa classe sono di tipo ``torch.Tensor``.

        I dati in ``_parmeters`` di questa classe sono di tipo ``torch.nn.Parameter``.


QMachine
""""""""""""""""""

.. py:class:: pyvqnet.qnn.vqc.torch.QMachine(num_wires, dtype=pyvqnet.kcomplex64,grad_mode="",save_ir=False)

    Classe simulatore per il calcolo quantistico variazionale, inclusi statevector il cui attributo states sono circuiti quantistici.

    Questa classe eredita da ``pyvqnet.nn.torch.TorchModule`` e ``pyvqnet.qnn.QMachine``.

    Questa classe puo essere aggiunta al modello torch come sottomodulo di ``torch.nn.Module``.

    .. note::

        Prima di ogni esecuzione del circuito quantistico completo, devi usare `pyvqnet.qnn.vqc.QMachine.reset_states(batchsize)` per reinizializzare lo stato iniziale nel simulatore e trasmetterlo alle dimensioni
        (batchsize,*) per adattarsi all'addestramento batch dei dati.

    :param num_wires: Il numero di bit quantistici.
    :param dtype: Il tipo di dato dei dati calcolati. Il valore predefinito e pyvqnet.kcomplex64 e la precisione del parametro corrispondente e pyvqnet.kfloat32.
    :param grad_mode: La modalita di calcolo del gradiente, puo essere "adjoint", il valore predefinito: "", utilizza la differenziazione automatica.
    :param save_ir: Quando impostato a True, salva l'operazione in originIR, il valore predefinito: False.

    :return: Restituisce un oggetto QMachine.

    Example::
        
        from pyvqnet.qnn.vqc.torch import QMachine
        import pyvqnet
        pyvqnet.backends.set_backend("torch")
        qm = QMachine(4)
        print(qm.states)


   .. py:method:: reset_states(batchsize)

        Reinizializza lo stato iniziale nel simulatore e lo trasmette alle dimensioni
        (batchsize,*) per adattarsi all'addestramento batch dei dati.

        :param batchsize: Dimensione del batch di elaborazione.

Modulo di porte logiche quantistiche variazionali
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Le seguenti interfacce di funzione in ``pyvqnet.qnn.vqc`` supportano direttamente il ``QTensor`` del backend ``torch`` per il calcolo.

.. csv-table:: Elenco delle interfacce pyvqnet.qnn.vqc supportate
    :file: ./images/same_apis_from_vqc.csv

I seguenti moduli di circuito quantistico ereditano da ``pyvqnet.qnn.vqc.torch.QModule``, dove i calcoli vengono eseguiti utilizzando ``torch.Tensor``.

.. note::

    Questa classe e le sue classi derivate sono applicabili solo a ``pyvqnet.backends.set_backend("torch")``, non mescolare con ``Module`` sotto il ``pyvqnet.nn`` predefinito.

    Se queste classi hanno variabili membro non parametriche ``_buffers``, i dati in esse sono di tipo ``torch.Tensor``.
    Se queste classi hanno variabili membro parametriche ``_parmeters``, i dati in esse sono di tipo ``torch.nn.Parameter``.

I
""""""""""""""""""

.. py:class:: pyvqnet.qnn.vqc.torch.I(has_params: bool = False,trainable: bool = False,init_params=None,wires=None,dtype=pyvqnet.kcomplex64,use_dagger=False)
    
    definisce una porta quantistica I.

    Questa classe eredita da ``pyvqnet.qnn.vqc.torch.QModule`` e ``torch.nn.Module``.
    Questa classe puo essere aggiunta al modello torch come sottomodulo di ``torch.nn.Module``.

    :param has_params: se ha parametri, come RX, RY e altre porte devono essere impostate a True, e quelle senza parametri devono essere impostate a False, il default e False.
    :param trainable: se ha parametri da addestrare. Se lo strato utilizza dati di input esterni per costruire la matrice della porta logica, impostare a False. Se i parametri da addestrare devono essere inizializzati da questo strato, e True, il default e False.
    :param init_params: Parametri di inizializzazione utilizzati per codificare i dati classici QTensor, il default e None.
    :param wires: Indice del bit su cui agisce la porta, il default e None.
    :param dtype: La precisione dei dati della matrice interna della porta logica puo essere impostata a pyvqnet.kcomplex64 o pyvqnet.kcomplex128, corrispondenti rispettivamente a input float o double.
    :param use_dagger: se utilizzare la versione trasposta coniugata della porta, il default e False.
    :return: un'istanza di ``pyvqnet.qnn.vqc.torch.QModule``

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
    
    definisce una porta quantistica Hadamard.

    Questa classe eredita da ``pyvqnet.qnn.vqc.torch.QModule`` e ``torch.nn.Module``.
    Questa classe puo essere aggiunta al modello torch come sottomodulo di ``torch.nn.Module``.

    :param has_params: se ha parametri, come RX, RY e altre porte devono essere impostate a True, e quelle senza parametri devono essere impostate a False, il default e False.
    :param trainable: se ha parametri da addestrare. Se lo strato utilizza dati di input esterni per costruire la matrice della porta logica, impostare a False. Se i parametri da addestrare devono essere inizializzati da questo strato, e True, il default e False.
    :param init_params: Parametri di inizializzazione utilizzati per codificare i dati classici QTensor, il default e None.
    :param wires: Indice del bit su cui agisce la porta, il default e None.
    :param dtype: La precisione dei dati della matrice interna della porta logica puo essere impostata a pyvqnet.kcomplex64 o pyvqnet.kcomplex128, corrispondenti rispettivamente a input float o double.
    :param use_dagger: se utilizzare la versione trasposta coniugata della porta, il default e False.
    :return: un'istanza di ``pyvqnet.qnn.vqc.torch.QModule``

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
    
    definisce una porta quantistica T.

    Questa classe eredita da ``pyvqnet.qnn.vqc.torch.QModule`` e ``torch.nn.Module``.
    Questa classe puo essere aggiunta al modello torch come sottomodulo di ``torch.nn.Module``.

    :param has_params: se ha parametri, come RX, RY e altre porte devono essere impostate a True, e quelle senza parametri devono essere impostate a False, il default e False.
    :param trainable: se ha parametri da addestrare. Se lo strato utilizza dati di input esterni per costruire la matrice della porta logica, impostare a False. Se i parametri da addestrare devono essere inizializzati da questo strato, e True, il default e False.
    :param init_params: Parametri di inizializzazione utilizzati per codificare i dati classici QTensor, il default e None.
    :param wires: Indice del bit su cui agisce la porta, il default e None.
    :param dtype: La precisione dei dati della matrice interna della porta logica puo essere impostata a pyvqnet.kcomplex64 o pyvqnet.kcomplex128, corrispondenti rispettivamente a input float o double.
    :param use_dagger: se utilizzare la versione trasposta coniugata della porta, il default e False.
    :return: un'istanza di ``pyvqnet.qnn.vqc.torch.QModule``

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
    
    definisce una porta quantistica S.

    Questa classe eredita da ``pyvqnet.qnn.vqc.torch.QModule`` e ``torch.nn.Module``.
    Questa classe puo essere aggiunta al modello torch come sottomodulo di ``torch.nn.Module``.

    :param has_params: se ha parametri, come RX, RY e altre porte devono essere impostate a True, e quelle senza parametri devono essere impostate a False, il default e False.
    :param trainable: se ha parametri da addestrare. Se lo strato utilizza dati di input esterni per costruire la matrice della porta logica, impostare a False. Se i parametri da addestrare devono essere inizializzati da questo strato, e True, il default e False.
    :param init_params: Parametri di inizializzazione utilizzati per codificare i dati classici QTensor, il default e None.
    :param wires: Indice del bit su cui agisce la porta, il default e None.
    :param dtype: La precisione dei dati della matrice interna della porta logica puo essere impostata a pyvqnet.kcomplex64 o pyvqnet.kcomplex128, corrispondenti rispettivamente a input float o double.
    :param use_dagger: se utilizzare la versione trasposta coniugata della porta, il default e False.
    :return: un'istanza di ``pyvqnet.qnn.vqc.torch.QModule``

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
    
    definisce una porta quantistica PauliX.

    Questa classe eredita da ``pyvqnet.qnn.vqc.torch.QModule`` e ``torch.nn.Module``.
    Questa classe puo essere aggiunta al modello torch come sottomodulo di ``torch.nn.Module``.


    :param has_params: whether it has parameters, such as RX, RY and other gates need to be set to True, and those without parameters need to be set to False, the default is False.
    :param trainable: whether it has parameters to be trained. If the layer uses external input data to build the logic gate matrix, set to False. If the parameters to be trained need to be initialized from this layer, it is True, the default is False.
    :param init_params: Initialization parameters used to encode classic data QTensor, the default is None.
    :param wires: Bit index of the line effect, the default is None.
    :param dtype: The data precision of the internal matrix of the logic gate can be set to pyvqnet.kcomplex64 or pyvqnet.kcomplex128, corresponding to float input or double input respectively.
    :param use_dagger: whether to use the transposed conjugate version of the gate, the default is False.
    :return: un'istanza di ``pyvqnet.qnn.vqc.torch.QModule``

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
    
    definisce una porta quantistica PauliY.

    Questa classe eredita da ``pyvqnet.qnn.vqc.torch.QModule`` e ``torch.nn.Module``.
    Questa classe puo essere aggiunta al modello torch come sottomodulo di ``torch.nn.Module``.


    :param has_params: whether it has parameters, such as RX, RY and other gates need to be set to True, and those without parameters need to be set to False, the default is False.
    :param trainable: whether it has parameters to be trained. If the layer uses external input data to build the logic gate matrix, set to False. If the parameters to be trained need to be initialized from this layer, it is True, the default is False.
    :param init_params: Initialization parameters used to encode classic data QTensor, the default is None.
    :param wires: Bit index of the line effect, the default is None.
    :param dtype: The data precision of the internal matrix of the logic gate can be set to pyvqnet.kcomplex64 or pyvqnet.kcomplex128, corresponding to float input or double input respectively.
    :param use_dagger: whether to use the transposed conjugate version of the gate, the default is False.
    :return: un'istanza di ``pyvqnet.qnn.vqc.torch.QModule``

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
    
    definisce una porta quantistica PauliZ.

    Questa classe eredita da ``pyvqnet.qnn.vqc.torch.QModule`` e ``torch.nn.Module``.
    Questa classe puo essere aggiunta al modello torch come sottomodulo di ``torch.nn.Module``.


    :param has_params: whether it has parameters, such as RX, RY and other gates need to be set to True, and those without parameters need to be set to False, the default is False.
    :param trainable: whether it has parameters to be trained. If the layer uses external input data to build the logic gate matrix, set to False. If the parameters to be trained need to be initialized from this layer, it is True, the default is False.
    :param init_params: Initialization parameters used to encode classic data QTensor, the default is None.
    :param wires: Bit index of the line effect, the default is None.
    :param dtype: The data precision of the internal matrix of the logic gate can be set to pyvqnet.kcomplex64 or pyvqnet.kcomplex128, corresponding to float input or double input respectively.
    :param use_dagger: whether to use the transposed conjugate version of the gate, the default is False.
    :return: un'istanza di ``pyvqnet.qnn.vqc.torch.QModule``

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
    
    definisce una porta quantistica X1.

    Questa classe eredita da ``pyvqnet.qnn.vqc.torch.QModule`` e ``torch.nn.Module``.
    Questa classe puo essere aggiunta al modello torch come sottomodulo di ``torch.nn.Module``.

    :param has_params: se ha parametri, come RX, RY e altre porte devono essere impostate a True, e quelle senza parametri devono essere impostate a False, il default e False.
    :param trainable: se ha parametri da addestrare. Se lo strato utilizza dati di input esterni per costruire la matrice della porta logica, impostare a False. Se i parametri da addestrare devono essere inizializzati da questo strato, e True, il default e False.
    :param init_params: Parametri di inizializzazione utilizzati per codificare i dati classici QTensor, il default e None.
    :param wires: Indice del bit su cui agisce la porta, il default e None.
    :param dtype: La precisione dei dati della matrice interna della porta logica puo essere impostata a pyvqnet.kcomplex64 o pyvqnet.kcomplex128, corrispondenti rispettivamente a input float o double.
    :param use_dagger: se utilizzare la versione trasposta coniugata della porta, il default e False.
    :return: un'istanza di ``pyvqnet.qnn.vqc.torch.QModule``

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
    
    definisce una porta quantistica RX.

    Questa classe eredita da ``pyvqnet.qnn.vqc.torch.QModule`` e ``torch.nn.Module``.
    Questa classe puo essere aggiunta al modello torch come sottomodulo di ``torch.nn.Module``.

    :param has_params: se ha parametri, come RX, RY e altre porte devono essere impostate a True, e quelle senza parametri devono essere impostate a False, il default e False.
    :param trainable: se ha parametri da addestrare. Se lo strato utilizza dati di input esterni per costruire la matrice della porta logica, impostare a False. Se i parametri da addestrare devono essere inizializzati da questo strato, e True, il default e False.
    :param init_params: Parametri di inizializzazione utilizzati per codificare i dati classici QTensor, il default e None.
    :param wires: Indice del bit su cui agisce la porta, il default e None.
    :param dtype: La precisione dei dati della matrice interna della porta logica puo essere impostata a pyvqnet.kcomplex64 o pyvqnet.kcomplex128, corrispondenti rispettivamente a input float o double.
    :param use_dagger: se utilizzare la versione trasposta coniugata della porta, il default e False.
    :return: un'istanza di ``pyvqnet.qnn.vqc.torch.QModule``

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
    
    definisce una porta quantistica RY.

    Questa classe eredita da ``pyvqnet.qnn.vqc.torch.QModule`` e ``torch.nn.Module``.
    Questa classe puo essere aggiunta al modello torch come sottomodulo di ``torch.nn.Module``.

    :param has_params: se ha parametri, come RX, RY e altre porte devono essere impostate a True, e quelle senza parametri devono essere impostate a False, il default e False.
    :param trainable: se ha parametri da addestrare. Se lo strato utilizza dati di input esterni per costruire la matrice della porta logica, impostare a False. Se i parametri da addestrare devono essere inizializzati da questo strato, e True, il default e False.
    :param init_params: Parametri di inizializzazione utilizzati per codificare i dati classici QTensor, il default e None.
    :param wires: Indice del bit su cui agisce la porta, il default e None.
    :param dtype: La precisione dei dati della matrice interna della porta logica puo essere impostata a pyvqnet.kcomplex64 o pyvqnet.kcomplex128, corrispondenti rispettivamente a input float o double.
    :param use_dagger: se utilizzare la versione trasposta coniugata della porta, il default e False.
    :return: un'istanza di ``pyvqnet.qnn.vqc.torch.QModule``

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
    
    definisce una porta quantistica RZ.

    Questa classe eredita da ``pyvqnet.qnn.vqc.torch.QModule`` e ``torch.nn.Module``.
    Questa classe puo essere aggiunta al modello torch come sottomodulo di ``torch.nn.Module``.

    :param has_params: se ha parametri, come RX, RY e altre porte devono essere impostate a True, e quelle senza parametri devono essere impostate a False, il default e False.
    :param trainable: se ha parametri da addestrare. Se lo strato utilizza dati di input esterni per costruire la matrice della porta logica, impostare a False. Se i parametri da addestrare devono essere inizializzati da questo strato, e True, il default e False.
    :param init_params: Parametri di inizializzazione utilizzati per codificare i dati classici QTensor, il default e None.
    :param wires: Indice del bit su cui agisce la porta, il default e None.
    :param dtype: La precisione dei dati della matrice interna della porta logica puo essere impostata a pyvqnet.kcomplex64 o pyvqnet.kcomplex128, corrispondenti rispettivamente a input float o double.
    :param use_dagger: se utilizzare la versione trasposta coniugata della porta, il default e False.
    :return: un'istanza di ``pyvqnet.qnn.vqc.torch.QModule``

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
    
    definisce una porta quantistica CRX.

    Questa classe eredita da ``pyvqnet.qnn.vqc.torch.QModule`` e ``torch.nn.Module``.
    Questa classe puo essere aggiunta al modello torch come sottomodulo di ``torch.nn.Module``.

    :param has_params: se ha parametri, come RX, RY e altre porte devono essere impostate a True, e quelle senza parametri devono essere impostate a False, il default e False.
    :param trainable: se ha parametri da addestrare. Se lo strato utilizza dati di input esterni per costruire la matrice della porta logica, impostare a False. Se i parametri da addestrare devono essere inizializzati da questo strato, e True, il default e False.
    :param init_params: Parametri di inizializzazione utilizzati per codificare i dati classici QTensor, il default e None.
    :param wires: Indice del bit su cui agisce la porta, il default e None.
    :param dtype: La precisione dei dati della matrice interna della porta logica puo essere impostata a pyvqnet.kcomplex64 o pyvqnet.kcomplex128, corrispondenti rispettivamente a input float o double.
    :param use_dagger: se utilizzare la versione trasposta coniugata della porta, il default e False.
    :return: un'istanza di ``pyvqnet.qnn.vqc.torch.QModule``

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
    
    definisce una porta quantistica CRY.

    Questa classe eredita da ``pyvqnet.qnn.vqc.torch.QModule`` e ``torch.nn.Module``.
    Questa classe puo essere aggiunta al modello torch come sottomodulo di ``torch.nn.Module``.

    :param has_params: se ha parametri, come RX, RY e altre porte devono essere impostate a True, e quelle senza parametri devono essere impostate a False, il default e False.
    :param trainable: se ha parametri da addestrare. Se lo strato utilizza dati di input esterni per costruire la matrice della porta logica, impostare a False. Se i parametri da addestrare devono essere inizializzati da questo strato, e True, il default e False.
    :param init_params: Parametri di inizializzazione utilizzati per codificare i dati classici QTensor, il default e None.
    :param wires: Indice del bit su cui agisce la porta, il default e None.
    :param dtype: La precisione dei dati della matrice interna della porta logica puo essere impostata a pyvqnet.kcomplex64 o pyvqnet.kcomplex128, corrispondenti rispettivamente a input float o double.
    :param use_dagger: se utilizzare la versione trasposta coniugata della porta, il default e False.
    :return: un'istanza di ``pyvqnet.qnn.vqc.torch.QModule``

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
    
    definisce una porta quantistica CRZ.

    Questa classe eredita da ``pyvqnet.qnn.vqc.torch.QModule`` e ``torch.nn.Module``.
    Questa classe puo essere aggiunta al modello torch come sottomodulo di ``torch.nn.Module``.

    :param has_params: se ha parametri, come RX, RY e altre porte devono essere impostate a True, e quelle senza parametri devono essere impostate a False, il default e False.
    :param trainable: se ha parametri da addestrare. Se lo strato utilizza dati di input esterni per costruire la matrice della porta logica, impostare a False. Se i parametri da addestrare devono essere inizializzati da questo strato, e True, il default e False.
    :param init_params: Parametri di inizializzazione utilizzati per codificare i dati classici QTensor, il default e None.
    :param wires: Indice del bit su cui agisce la porta, il default e None.
    :param dtype: La precisione dei dati della matrice interna della porta logica puo essere impostata a pyvqnet.kcomplex64 o pyvqnet.kcomplex128, corrispondenti rispettivamente a input float o double.
    :param use_dagger: se utilizzare la versione trasposta coniugata della porta, il default e False.
    :return: un'istanza di ``pyvqnet.qnn.vqc.torch.QModule``

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
    
    definisce una porta quantistica U1.

    Questa classe eredita da ``pyvqnet.qnn.vqc.torch.QModule`` e ``torch.nn.Module``.
    Questa classe puo essere aggiunta al modello torch come sottomodulo di ``torch.nn.Module``.

    :param has_params: se ha parametri, come RX, RY e altre porte devono essere impostate a True, e quelle senza parametri devono essere impostate a False, il default e False.
    :param trainable: se ha parametri da addestrare. Se lo strato utilizza dati di input esterni per costruire la matrice della porta logica, impostare a False. Se i parametri da addestrare devono essere inizializzati da questo strato, e True, il default e False.
    :param init_params: Parametri di inizializzazione utilizzati per codificare i dati classici QTensor, il default e None.
    :param wires: Indice del bit su cui agisce la porta, il default e None.
    :param dtype: La precisione dei dati della matrice interna della porta logica puo essere impostata a pyvqnet.kcomplex64 o pyvqnet.kcomplex128, corrispondenti rispettivamente a input float o double.
    :param use_dagger: se utilizzare la versione trasposta coniugata della porta, il default e False.
    :return: un'istanza di ``pyvqnet.qnn.vqc.torch.QModule``

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
    
    definisce una porta quantistica U2.

    Questa classe eredita da ``pyvqnet.qnn.vqc.torch.QModule`` e ``torch.nn.Module``.
    Questa classe puo essere aggiunta al modello torch come sottomodulo di ``torch.nn.Module``.

    :param has_params: se ha parametri, come RX, RY e altre porte devono essere impostate a True, e quelle senza parametri devono essere impostate a False, il default e False.
    :param trainable: se ha parametri da addestrare. Se lo strato utilizza dati di input esterni per costruire la matrice della porta logica, impostare a False. Se i parametri da addestrare devono essere inizializzati da questo strato, e True, il default e False.
    :param init_params: Parametri di inizializzazione utilizzati per codificare i dati classici QTensor, il default e None.
    :param wires: Indice del bit su cui agisce la porta, il default e None.
    :param dtype: La precisione dei dati della matrice interna della porta logica puo essere impostata a pyvqnet.kcomplex64 o pyvqnet.kcomplex128, corrispondenti rispettivamente a input float o double.
    :param use_dagger: se utilizzare la versione trasposta coniugata della porta, il default e False.
    :return: un'istanza di ``pyvqnet.qnn.vqc.torch.QModule``

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
    
    definisce una porta quantistica U3.

    Questa classe eredita da ``pyvqnet.qnn.vqc.torch.QModule`` e ``torch.nn.Module``.
    Questa classe puo essere aggiunta al modello torch come sottomodulo di ``torch.nn.Module``.

    :param has_params: se ha parametri, come RX, RY e altre porte devono essere impostate a True, e quelle senza parametri devono essere impostate a False, il default e False.
    :param trainable: se ha parametri da addestrare. Se lo strato utilizza dati di input esterni per costruire la matrice della porta logica, impostare a False. Se i parametri da addestrare devono essere inizializzati da questo strato, e True, il default e False.
    :param init_params: Parametri di inizializzazione utilizzati per codificare i dati classici QTensor, il default e None.
    :param wires: Indice del bit su cui agisce la porta, il default e None.
    :param dtype: La precisione dei dati della matrice interna della porta logica puo essere impostata a pyvqnet.kcomplex64 o pyvqnet.kcomplex128, corrispondenti rispettivamente a input float o double.
    :param use_dagger: se utilizzare la versione trasposta coniugata della porta, il default e False.
    :return: un'istanza di ``pyvqnet.qnn.vqc.torch.QModule``

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
    
    definisce una porta quantistica CNOT, alias `CX`.

    Questa classe eredita da ``pyvqnet.qnn.vqc.torch.QModule`` e ``torch.nn.Module``.
    Questa classe puo essere aggiunta al modello torch come sottomodulo di ``torch.nn.Module``.

    :param has_params: se ha parametri, come RX, RY e altre porte devono essere impostate a True, e quelle senza parametri devono essere impostate a False, il default e False.
    :param trainable: se ha parametri da addestrare. Se lo strato utilizza dati di input esterni per costruire la matrice della porta logica, impostare a False. Se i parametri da addestrare devono essere inizializzati da questo strato, e True, il default e False.
    :param init_params: Parametri di inizializzazione utilizzati per codificare i dati classici QTensor, il default e None.
    :param wires: Indice del bit su cui agisce la porta, il default e None.
    :param dtype: La precisione dei dati della matrice interna della porta logica puo essere impostata a pyvqnet.kcomplex64 o pyvqnet.kcomplex128, corrispondenti rispettivamente a input float o double.
    :param use_dagger: se utilizzare la versione trasposta coniugata della porta, il default e False.
    :return: un'istanza di ``pyvqnet.qnn.vqc.torch.QModule``

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
    
    definisce una porta quantistica CY.

    Questa classe eredita da ``pyvqnet.qnn.vqc.torch.QModule`` e ``torch.nn.Module``.
    Questa classe puo essere aggiunta al modello torch come sottomodulo di ``torch.nn.Module``.

    :param has_params: se ha parametri, come RX, RY e altre porte devono essere impostate a True, e quelle senza parametri devono essere impostate a False, il default e False.
    :param trainable: se ha parametri da addestrare. Se lo strato utilizza dati di input esterni per costruire la matrice della porta logica, impostare a False. Se i parametri da addestrare devono essere inizializzati da questo strato, e True, il default e False.
    :param init_params: Parametri di inizializzazione utilizzati per codificare i dati classici QTensor, il default e None.
    :param wires: Indice del bit su cui agisce la porta, il default e None.
    :param dtype: La precisione dei dati della matrice interna della porta logica puo essere impostata a pyvqnet.kcomplex64 o pyvqnet.kcomplex128, corrispondenti rispettivamente a input float o double.
    :param use_dagger: se utilizzare la versione trasposta coniugata della porta, il default e False.
    :return: un'istanza di ``pyvqnet.qnn.vqc.torch.QModule``

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
    
    definisce una porta quantistica CZ.

    Questa classe eredita da ``pyvqnet.qnn.vqc.torch.QModule`` e ``torch.nn.Module``.
    Questa classe puo essere aggiunta al modello torch come sottomodulo di ``torch.nn.Module``.

    :param has_params: se ha parametri, come RX, RY e altre porte devono essere impostate a True, e quelle senza parametri devono essere impostate a False, il default e False.
    :param trainable: se ha parametri da addestrare. Se lo strato utilizza dati di input esterni per costruire la matrice della porta logica, impostare a False. Se i parametri da addestrare devono essere inizializzati da questo strato, e True, il default e False.
    :param init_params: Parametri di inizializzazione utilizzati per codificare i dati classici QTensor, il default e None.
    :param wires: Indice del bit su cui agisce la porta, il default e None.
    :param dtype: La precisione dei dati della matrice interna della porta logica puo essere impostata a pyvqnet.kcomplex64 o pyvqnet.kcomplex128, corrispondenti rispettivamente a input float o double.
    :param use_dagger: se utilizzare la versione trasposta coniugata della porta, il default e False.
    :return: un'istanza di ``pyvqnet.qnn.vqc.torch.QModule``

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
    
    definisce una porta quantistica CR.

    Questa classe eredita da ``pyvqnet.qnn.vqc.torch.QModule`` e ``torch.nn.Module``.
    Questa classe puo essere aggiunta al modello torch come sottomodulo di ``torch.nn.Module``.

    :param has_params: se ha parametri, come RX, RY e altre porte devono essere impostate a True, e quelle senza parametri devono essere impostate a False, il default e False.
    :param trainable: se ha parametri da addestrare. Se lo strato utilizza dati di input esterni per costruire la matrice della porta logica, impostare a False. Se i parametri da addestrare devono essere inizializzati da questo strato, e True, il default e False.
    :param init_params: Parametri di inizializzazione utilizzati per codificare i dati classici QTensor, il default e None.
    :param wires: Indice del bit su cui agisce la porta, il default e None.
    :param dtype: La precisione dei dati della matrice interna della porta logica puo essere impostata a pyvqnet.kcomplex64 o pyvqnet.kcomplex128, corrispondenti rispettivamente a input float o double.
    :param use_dagger: se utilizzare la versione trasposta coniugata della porta, il default e False.
    :return: un'istanza di ``pyvqnet.qnn.vqc.torch.QModule``

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
    
    definisce una porta quantistica SWAP.

    Questa classe eredita da ``pyvqnet.qnn.vqc.torch.QModule`` e ``torch.nn.Module``.
    Questa classe puo essere aggiunta al modello torch come sottomodulo di ``torch.nn.Module``.

    :param has_params: se ha parametri, come RX, RY e altre porte devono essere impostate a True, e quelle senza parametri devono essere impostate a False, il default e False.
    :param trainable: se ha parametri da addestrare. Se lo strato utilizza dati di input esterni per costruire la matrice della porta logica, impostare a False. Se i parametri da addestrare devono essere inizializzati da questo strato, e True, il default e False.
    :param init_params: Parametri di inizializzazione utilizzati per codificare i dati classici QTensor, il default e None.
    :param wires: Indice del bit su cui agisce la porta, il default e None.
    :param dtype: La precisione dei dati della matrice interna della porta logica puo essere impostata a pyvqnet.kcomplex64 o pyvqnet.kcomplex128, corrispondenti rispettivamente a input float o double.
    :param use_dagger: se utilizzare la versione trasposta coniugata della porta, il default e False.
    :return: un'istanza di ``pyvqnet.qnn.vqc.torch.QModule``

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
    
    definisce una porta quantistica SWAP.

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

    Questa classe eredita da ``pyvqnet.qnn.vqc.torch.QModule`` e ``torch.nn.Module``.
    Questa classe puo essere aggiunta al modello torch come sottomodulo di ``torch.nn.Module``.

    :param has_params: se ha parametri, come RX, RY e altre porte devono essere impostate a True, e quelle senza parametri devono essere impostate a False, il default e False.
    :param trainable: se ha parametri da addestrare. Se lo strato utilizza dati di input esterni per costruire la matrice della porta logica, impostare a False. Se i parametri da addestrare devono essere inizializzati da questo strato, e True, il default e False.
    :param init_params: Parametri di inizializzazione utilizzati per codificare i dati classici QTensor, il default e None.
    :param wires: Indice del bit su cui agisce la porta, il default e None.
    :param dtype: La precisione dei dati della matrice interna della porta logica puo essere impostata a pyvqnet.kcomplex64 o pyvqnet.kcomplex128, corrispondenti rispettivamente a input float o double.
    :param use_dagger: se utilizzare la versione trasposta coniugata della porta, il default e False.
    :return: un'istanza di ``pyvqnet.qnn.vqc.torch.QModule``

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
    
    definisce una porta quantistica RXX.

    Questa classe eredita da ``pyvqnet.qnn.vqc.torch.QModule`` e ``torch.nn.Module``.
    Questa classe puo essere aggiunta al modello torch come sottomodulo di ``torch.nn.Module``.

    :param has_params: se ha parametri, come RX, RY e altre porte devono essere impostate a True, e quelle senza parametri devono essere impostate a False, il default e False.
    :param trainable: se ha parametri da addestrare. Se lo strato utilizza dati di input esterni per costruire la matrice della porta logica, impostare a False. Se i parametri da addestrare devono essere inizializzati da questo strato, e True, il default e False.
    :param init_params: Parametri di inizializzazione utilizzati per codificare i dati classici QTensor, il default e None.
    :param wires: Indice del bit su cui agisce la porta, il default e None.
    :param dtype: La precisione dei dati della matrice interna della porta logica puo essere impostata a pyvqnet.kcomplex64 o pyvqnet.kcomplex128, corrispondenti rispettivamente a input float o double.
    :param use_dagger: se utilizzare la versione trasposta coniugata della porta, il default e False.
    :return: un'istanza di ``pyvqnet.qnn.vqc.torch.QModule``

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
    
    definisce una porta quantistica RYY.

    Questa classe eredita da ``pyvqnet.qnn.vqc.torch.QModule`` e ``torch.nn.Module``.
    Questa classe puo essere aggiunta al modello torch come sottomodulo di ``torch.nn.Module``.

    :param has_params: se ha parametri, come RX, RY e altre porte devono essere impostate a True, e quelle senza parametri devono essere impostate a False, il default e False.
    :param trainable: se ha parametri da addestrare. Se lo strato utilizza dati di input esterni per costruire la matrice della porta logica, impostare a False. Se i parametri da addestrare devono essere inizializzati da questo strato, e True, il default e False.
    :param init_params: Parametri di inizializzazione utilizzati per codificare i dati classici QTensor, il default e None.
    :param wires: Indice del bit su cui agisce la porta, il default e None.
    :param dtype: La precisione dei dati della matrice interna della porta logica puo essere impostata a pyvqnet.kcomplex64 o pyvqnet.kcomplex128, corrispondenti rispettivamente a input float o double.
    :param use_dagger: se utilizzare la versione trasposta coniugata della porta, il default e False.
    :return: un'istanza di ``pyvqnet.qnn.vqc.torch.QModule``

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
    
    definisce una porta quantistica RZZ.

    Questa classe eredita da ``pyvqnet.qnn.vqc.torch.QModule`` e ``torch.nn.Module``.
    Questa classe puo essere aggiunta al modello torch come sottomodulo di ``torch.nn.Module``.

    :param has_params: se ha parametri, come RX, RY e altre porte devono essere impostate a True, e quelle senza parametri devono essere impostate a False, il default e False.
    :param trainable: se ha parametri da addestrare. Se lo strato utilizza dati di input esterni per costruire la matrice della porta logica, impostare a False. Se i parametri da addestrare devono essere inizializzati da questo strato, e True, il default e False.
    :param init_params: Parametri di inizializzazione utilizzati per codificare i dati classici QTensor, il default e None.
    :param wires: Indice del bit su cui agisce la porta, il default e None.
    :param dtype: La precisione dei dati della matrice interna della porta logica puo essere impostata a pyvqnet.kcomplex64 o pyvqnet.kcomplex128, corrispondenti rispettivamente a input float o double.
    :param use_dagger: se utilizzare la versione trasposta coniugata della porta, il default e False.
    :return: un'istanza di ``pyvqnet.qnn.vqc.torch.QModule``

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
    
    definisce una porta quantistica RZX.

    Questa classe eredita da ``pyvqnet.qnn.vqc.torch.QModule`` e ``torch.nn.Module``.
    Questa classe puo essere aggiunta al modello torch come sottomodulo di ``torch.nn.Module``.

    :param has_params: se ha parametri, come RX, RY e altre porte devono essere impostate a True, e quelle senza parametri devono essere impostate a False, il default e False.
    :param trainable: se ha parametri da addestrare. Se lo strato utilizza dati di input esterni per costruire la matrice della porta logica, impostare a False. Se i parametri da addestrare devono essere inizializzati da questo strato, e True, il default e False.
    :param init_params: Parametri di inizializzazione utilizzati per codificare i dati classici QTensor, il default e None.
    :param wires: Indice del bit su cui agisce la porta, il default e None.
    :param dtype: La precisione dei dati della matrice interna della porta logica puo essere impostata a pyvqnet.kcomplex64 o pyvqnet.kcomplex128, corrispondenti rispettivamente a input float o double.
    :param use_dagger: se utilizzare la versione trasposta coniugata della porta, il default e False.
    :return: un'istanza di ``pyvqnet.qnn.vqc.torch.QModule``

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
    
    definisce una porta quantistica Toffoli.

    Questa classe eredita da ``pyvqnet.qnn.vqc.torch.QModule`` e ``torch.nn.Module``.
    Questa classe puo essere aggiunta al modello torch come sottomodulo di ``torch.nn.Module``.

    :param has_params: se ha parametri, come RX, RY e altre porte devono essere impostate a True, e quelle senza parametri devono essere impostate a False, il default e False.
    :param trainable: se ha parametri da addestrare. Se lo strato utilizza dati di input esterni per costruire la matrice della porta logica, impostare a False. Se i parametri da addestrare devono essere inizializzati da questo strato, e True, il default e False.
    :param init_params: Parametri di inizializzazione utilizzati per codificare i dati classici QTensor, il default e None.
    :param wires: Indice del bit su cui agisce la porta, il default e None.
    :param dtype: La precisione dei dati della matrice interna della porta logica puo essere impostata a pyvqnet.kcomplex64 o pyvqnet.kcomplex128, corrispondenti rispettivamente a input float o double.
    :param use_dagger: se utilizzare la versione trasposta coniugata della porta, il default e False.
    :return: un'istanza di ``pyvqnet.qnn.vqc.torch.QModule``

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
    
    definisce una porta quantistica IsingXX.

    Questa classe eredita da ``pyvqnet.qnn.vqc.torch.QModule`` e ``torch.nn.Module``.
    Questa classe puo essere aggiunta al modello torch come sottomodulo di ``torch.nn.Module``.

    :param has_params: se ha parametri, come RX, RY e altre porte devono essere impostate a True, e quelle senza parametri devono essere impostate a False, il default e False.
    :param trainable: se ha parametri da addestrare. Se lo strato utilizza dati di input esterni per costruire la matrice della porta logica, impostare a False. Se i parametri da addestrare devono essere inizializzati da questo strato, e True, il default e False.
    :param init_params: Parametri di inizializzazione utilizzati per codificare i dati classici QTensor, il default e None.
    :param wires: Indice del bit su cui agisce la porta, il default e None.
    :param dtype: La precisione dei dati della matrice interna della porta logica puo essere impostata a pyvqnet.kcomplex64 o pyvqnet.kcomplex128, corrispondenti rispettivamente a input float o double.
    :param use_dagger: se utilizzare la versione trasposta coniugata della porta, il default e False.
    :return: un'istanza di ``pyvqnet.qnn.vqc.torch.QModule``

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
    
    definisce una porta quantistica IsingYY.

    Questa classe eredita da ``pyvqnet.qnn.vqc.torch.QModule`` e ``torch.nn.Module``.
    Questa classe puo essere aggiunta al modello torch come sottomodulo di ``torch.nn.Module``.

    :param has_params: se ha parametri, come RX, RY e altre porte devono essere impostate a True, e quelle senza parametri devono essere impostate a False, il default e False.
    :param trainable: se ha parametri da addestrare. Se lo strato utilizza dati di input esterni per costruire la matrice della porta logica, impostare a False. Se i parametri da addestrare devono essere inizializzati da questo strato, e True, il default e False.
    :param init_params: Parametri di inizializzazione utilizzati per codificare i dati classici QTensor, il default e None.
    :param wires: Indice del bit su cui agisce la porta, il default e None.
    :param dtype: La precisione dei dati della matrice interna della porta logica puo essere impostata a pyvqnet.kcomplex64 o pyvqnet.kcomplex128, corrispondenti rispettivamente a input float o double.
    :param use_dagger: se utilizzare la versione trasposta coniugata della porta, il default e False.
    :return: un'istanza di ``pyvqnet.qnn.vqc.torch.QModule``

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
    
    definisce una porta quantistica IsingZZ.

    Questa classe eredita da ``pyvqnet.qnn.vqc.torch.QModule`` e ``torch.nn.Module``.
    Questa classe puo essere aggiunta al modello torch come sottomodulo di ``torch.nn.Module``.

    :param has_params: se ha parametri, come RX, RY e altre porte devono essere impostate a True, e quelle senza parametri devono essere impostate a False, il default e False.
    :param trainable: se ha parametri da addestrare. Se lo strato utilizza dati di input esterni per costruire la matrice della porta logica, impostare a False. Se i parametri da addestrare devono essere inizializzati da questo strato, e True, il default e False.
    :param init_params: Parametri di inizializzazione utilizzati per codificare i dati classici QTensor, il default e None.
    :param wires: Indice del bit su cui agisce la porta, il default e None.
    :param dtype: La precisione dei dati della matrice interna della porta logica puo essere impostata a pyvqnet.kcomplex64 o pyvqnet.kcomplex128, corrispondenti rispettivamente a input float o double.
    :param use_dagger: se utilizzare la versione trasposta coniugata della porta, il default e False.
    :return: un'istanza di ``pyvqnet.qnn.vqc.torch.QModule``

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
    
    definisce una porta quantistica IsingXY.

    Questa classe eredita da ``pyvqnet.qnn.vqc.torch.QModule`` e ``torch.nn.Module``.
    Questa classe puo essere aggiunta al modello torch come sottomodulo di ``torch.nn.Module``.

    :param has_params: se ha parametri, come RX, RY e altre porte devono essere impostate a True, e quelle senza parametri devono essere impostate a False, il default e False.
    :param trainable: se ha parametri da addestrare. Se lo strato utilizza dati di input esterni per costruire la matrice della porta logica, impostare a False. Se i parametri da addestrare devono essere inizializzati da questo strato, e True, il default e False.
    :param init_params: Parametri di inizializzazione utilizzati per codificare i dati classici QTensor, il default e None.
    :param wires: Indice del bit su cui agisce la porta, il default e None.
    :param dtype: La precisione dei dati della matrice interna della porta logica puo essere impostata a pyvqnet.kcomplex64 o pyvqnet.kcomplex128, corrispondenti rispettivamente a input float o double.
    :param use_dagger: se utilizzare la versione trasposta coniugata della porta, il default e False.
    :return: un'istanza di ``pyvqnet.qnn.vqc.torch.QModule``

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
    
    definisce una porta quantistica PhaseShift.

    Questa classe eredita da ``pyvqnet.qnn.vqc.torch.QModule`` e ``torch.nn.Module``.
    Questa classe puo essere aggiunta al modello torch come sottomodulo di ``torch.nn.Module``.

    :param has_params: se ha parametri, come RX, RY e altre porte devono essere impostate a True, e quelle senza parametri devono essere impostate a False, il default e False.
    :param trainable: se ha parametri da addestrare. Se lo strato utilizza dati di input esterni per costruire la matrice della porta logica, impostare a False. Se i parametri da addestrare devono essere inizializzati da questo strato, e True, il default e False.
    :param init_params: Parametri di inizializzazione utilizzati per codificare i dati classici QTensor, il default e None.
    :param wires: Indice del bit su cui agisce la porta, il default e None.
    :param dtype: La precisione dei dati della matrice interna della porta logica puo essere impostata a pyvqnet.kcomplex64 o pyvqnet.kcomplex128, corrispondenti rispettivamente a input float o double.
    :param use_dagger: se utilizzare la versione trasposta coniugata della porta, il default e False.
    :return: un'istanza di ``pyvqnet.qnn.vqc.torch.QModule``

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
    
    definisce una porta quantistica MultiRZ.

    Questa classe eredita da ``pyvqnet.qnn.vqc.torch.QModule`` e ``torch.nn.Module``.
    Questa classe puo essere aggiunta al modello torch come sottomodulo di ``torch.nn.Module``.

    :param has_params: se ha parametri, come RX, RY e altre porte devono essere impostate a True, e quelle senza parametri devono essere impostate a False, il default e False.
    :param trainable: se ha parametri da addestrare. Se lo strato utilizza dati di input esterni per costruire la matrice della porta logica, impostare a False. Se i parametri da addestrare devono essere inizializzati da questo strato, e True, il default e False.
    :param init_params: Parametri di inizializzazione utilizzati per codificare i dati classici QTensor, il default e None.
    :param wires: Indice del bit su cui agisce la porta, il default e None.
    :param dtype: La precisione dei dati della matrice interna della porta logica puo essere impostata a pyvqnet.kcomplex64 o pyvqnet.kcomplex128, corrispondenti rispettivamente a input float o double.
    :param use_dagger: se utilizzare la versione trasposta coniugata della porta, il default e False.
    :return: un'istanza di ``pyvqnet.qnn.vqc.torch.QModule``

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
    
    definisce una porta quantistica SDG.

    Questa classe eredita da ``pyvqnet.qnn.vqc.torch.QModule`` e ``torch.nn.Module``.
    Questa classe puo essere aggiunta al modello torch come sottomodulo di ``torch.nn.Module``.

    :param has_params: se ha parametri, come RX, RY e altre porte devono essere impostate a True, e quelle senza parametri devono essere impostate a False, il default e False.
    :param trainable: se ha parametri da addestrare. Se lo strato utilizza dati di input esterni per costruire la matrice della porta logica, impostare a False. Se i parametri da addestrare devono essere inizializzati da questo strato, e True, il default e False.
    :param init_params: Parametri di inizializzazione utilizzati per codificare i dati classici QTensor, il default e None.
    :param wires: Indice del bit su cui agisce la porta, il default e None.
    :param dtype: La precisione dei dati della matrice interna della porta logica puo essere impostata a pyvqnet.kcomplex64 o pyvqnet.kcomplex128, corrispondenti rispettivamente a input float o double.
    :param use_dagger: se utilizzare la versione trasposta coniugata della porta, il default e False.
    :return: un'istanza di ``pyvqnet.qnn.vqc.torch.QModule``

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
    
    definisce una porta quantistica SDG.

    Questa classe eredita da ``pyvqnet.qnn.vqc.torch.QModule`` e ``torch.nn.Module``.
    Questa classe puo essere aggiunta al modello torch come sottomodulo di ``torch.nn.Module``.

    :param has_params: se ha parametri, come RX, RY e altre porte devono essere impostate a True, e quelle senza parametri devono essere impostate a False, il default e False.
    :param trainable: se ha parametri da addestrare. Se lo strato utilizza dati di input esterni per costruire la matrice della porta logica, impostare a False. Se i parametri da addestrare devono essere inizializzati da questo strato, e True, il default e False.
    :param init_params: Parametri di inizializzazione utilizzati per codificare i dati classici QTensor, il default e None.
    :param wires: Indice del bit su cui agisce la porta, il default e None.
    :param dtype: La precisione dei dati della matrice interna della porta logica puo essere impostata a pyvqnet.kcomplex64 o pyvqnet.kcomplex128, corrispondenti rispettivamente a input float o double.
    :param use_dagger: se utilizzare la versione trasposta coniugata della porta, il default e False.
    :return: un'istanza di ``pyvqnet.qnn.vqc.torch.QModule``

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
    
    definisce una porta quantistica ControlledPhaseShift.

    Questa classe eredita da ``pyvqnet.qnn.vqc.torch.QModule`` e ``torch.nn.Module``.
    Questa classe puo essere aggiunta al modello torch come sottomodulo di ``torch.nn.Module``.

    :param has_params: se ha parametri, come RX, RY e altre porte devono essere impostate a True, e quelle senza parametri devono essere impostate a False, il default e False.
    :param trainable: se ha parametri da addestrare. Se lo strato utilizza dati di input esterni per costruire la matrice della porta logica, impostare a False. Se i parametri da addestrare devono essere inizializzati da questo strato, e True, il default e False.
    :param init_params: Parametri di inizializzazione utilizzati per codificare i dati classici QTensor, il default e None.
    :param wires: Indice del bit su cui agisce la porta, il default e None.
    :param dtype: La precisione dei dati della matrice interna della porta logica puo essere impostata a pyvqnet.kcomplex64 o pyvqnet.kcomplex128, corrispondenti rispettivamente a input float o double.
    :param use_dagger: se utilizzare la versione trasposta coniugata della porta, il default e False.
    :return: un'istanza di ``pyvqnet.qnn.vqc.torch.QModule``

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
    
    definisce una porta quantistica MultiControlledX.

    Questa classe eredita da ``pyvqnet.qnn.vqc.torch.QModule`` e ``torch.nn.Module``.
    Questa classe puo essere aggiunta al modello torch come sottomodulo di ``torch.nn.Module``.
    
    :param has_params: whether it has parameters, such as RX, RY and other gates need to be set to True, and those without parameters need to be set to False, the default is False.
    :param trainable: whether it has parameters to be trained. If the layer uses external input data to build the logic gate matrix, set to False. If the parameters to be trained need to be initialized from this layer, it is True, the default is False.
    :param init_params: Initialization parameters used to encode classic data QTensor, the default is None.
    :param wires: Bit index of the line effect, the default is None.
    :param dtype: The data precision of the internal matrix of the logic gate can be set to pyvqnet.kcomplex64 or pyvqnet.kcomplex128, corresponding to float input or double input respectively.
    :param use_dagger: whether to use the transposed conjugate version of the gate, the default is False.
    :param control_values: Control value, the default is None, when the bit is 1, it is controlled.

    :return: un'istanza di ``pyvqnet.qnn.vqc.torch.QModule``

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


API di Misurazione
^^^^^^^^^^^^^^^^^^^^^^

Probability
"""""""""""""""""""""""""""""


.. py:class:: pyvqnet.qnn.vqc.torch.Probability(wires=None, name="")

    Calcola il risultato della misurazione di probabilita del circuito quantistico su un bit specifico.

    Questa classe eredita da ``pyvqnet.qnn.vqc.torch.QModule`` e ``torch.nn.Module``.
    Questa classe puo essere aggiunta al modello torch come sottomodulo di ``torch.nn.Module``.

    :param wires: L'indice del bit di misurazione, lista, tupla o intero.
    :param name: Il nome del modulo, default: "".
    :return: The measurement result, QTensor.

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

    Calcola i risultati di misurazione dei circuiti quantistici, supporta l'input obs come operatori di Pauli multipli o singoli o Hamiltoniane.
    For example:

    {\'X0\': 0.23} indicates a PauliX effect on qubit 0, with a coefficient of 0.23.

    {\'X1 Z2\': 2.4,\'Y2\': -0.5} corresponds to the observed value 2.4 * X1 @ Z2 - 0.5 * Y2.

    [{\'X1 Z2\': 4,\'Z1 Z0\': 3},{\'X1 Y2 Z0\': 3.5}] corresponds to the two observed values 4 * X1 @ Z2 + 3 * Z1 @ Z0 and 3.5 * X1 @ Y2 @ Z0.
        
        
    Questa classe eredita da ``pyvqnet.qnn.vqc.torch.QModule`` e ``torch.nn.Module``.

    Questa classe puo essere aggiunta al modello torch come sottomodulo di ``torch.nn.Module``.

    :param obs: osservabile.
    :param name: nome del modulo, default: "".
    :return: risultato della misurazione, QTensor.

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

    Ottiene risultati campione con shot su wires specifici.

    Questa classe eredita da ``pyvqnet.qnn.vqc.torch.QModule`` e ``torch.nn.Module``.
    Questa classe puo essere aggiunta al modello torch come sottomodulo di ``torch.nn.Module``.


    :param wires: Indice del qubit campionato. Valore predefinito: None, utilizza tutti i bit del simulatore in fase di esecuzione.
    :param obs: This value can only be None.
    :param shots: Sample repetition count, default value: 1.
    :param name: Il nome di questo modulo, valore predefinito: "".
    :return: una classe di metodo di misurazione

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

    Calcola l'aspettativa di una quantita Hermitiana in un circuito quantistico.

    Questa classe eredita da ``pyvqnet.qnn.vqc.torch.QModule`` e ``torch.nn.Module``.
    Questa classe puo essere aggiunta al modello torch come sottomodulo di ``torch.nn.Module``.


    :param obs: quantita Hermitiana.
    :param name: nome del modulo, default: "".
    :return: risultato atteso, QTensor.

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

Modelli comuni per circuiti quantistici
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

VQC_HardwareEfficientAnsatz
""""""""""""""""""""""""""""""""""""""""


.. py:class:: pyvqnet.qnn.vqc.torch.VQC_HardwareEfficientAnsatz(n_qubits,single_rot_gate_list,entangle_gate="CNOT",entangle_rules='linear',depth=1,initial=None,dtype=None)

    Implementazione di Hardware Efficient Ansatz introdotto nell'articolo: `Hardware-efficient Variational Quantum Eigensolver for Small Molecules <https://arxiv.org/pdf/1704.05018.pdf>`__ .

    Questa classe eredita da ``pyvqnet.qnn.vqc.torch.QModule`` e ``torch.nn.Module``.

    Questa classe puo essere aggiunta al modello torch come sottomodulo di ``torch.nn.Module``.


    :param n_qubits: Numero di qubit.
    :param single_rot_gate_list: Una lista di porte di rotazione a singolo qubit costruita da una o piu porte di rotazione che agiscono su ogni qubit. Attualmente supporta Rx, Ry, Rz.
    :param entangle_gate: La porta di entanglement non parametrizzata. Sono supportati CNOT, CZ. Default: CNOT.
    :param entangle_rules: Come la porta di entanglement viene utilizzata nel circuito. 'linear' significa che la porta di entanglement agira su ogni qubit vicino. 'all' significa che la porta di entanglement agira su qualsiasi coppia di qubit. Default: linear.
    :param depth: La profondita dell'ansatz, default: 1.
    :param initial: valore iniziale uguale per tutti i parametri, default: None, questo modulo inizializzera i parametri casualmente.
    :param dtype: tipo di dato dei parametri.
    :return: un'istanza di VQC_HardwareEfficientAnsatz.

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

    Uno strato costituito da una rotazione a singolo qubit con singolo parametro su ogni qubit, seguito da molteplici porte CNOT in una combinazione a catena chiusa o ad anello.

    Un anello di porte CNOT collega ogni qubit ai suoi vicini e, infine, il qubit a e considerato vicino del qubit a.

    Questa classe eredita da ``pyvqnet.qnn.vqc.torch.QModule`` e ``torch.nn.Module``.

    Questa classe puo essere aggiunta al modello torch come sottomodulo di ``torch.nn.Module``.

    :param num_layers: numero di strati ripetuti, default: 1.
    :param num_qubits: number of qubits, default: 1.
    :param rotation: one-parameter single-qubit gate to use, default: `RX`
    :param initial: valore inizializzato uguale per tutti i parametri, default: None, i parametri verranno inizializzati casualmente.
    :param dtype: data type of parameter, default:None,use float32.
    :return: A VQC_BasicEntanglerTemplate instance

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

    Strati costituiti da rotazioni a singolo qubit e entangler, come in `circuit-centric classifier design <https://arxiv.org/abs/1804.00633>`__ .

    Questa classe eredita da ``pyvqnet.qnn.vqc.torch.QModule`` e ``torch.nn.Module``.
    Questa classe puo essere aggiunta al modello torch come sottomodulo di ``torch.nn.Module``.


    :param num_layers: numero di strati ripetuti, default: 1.
    :param num_qubits: number of qubits, default: 1.
    :param ranges: sequenza che determina l'iperparametro di intervallo per ogni strato successivo; default: None
                                using :math: `r=l \mod M` for the :math:`l` th layer and :math:`M` qubits.
    :param initial: valore iniziale per tutti i parametri, default: None, inizializzato casualmente.
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
    
    Usa RZ, RY, RZ per creare circuiti quantistici variazionali per codificare dati classici in stati quantistici.
    Reference `Quantum embeddings for machine learning <https://arxiv.org/abs/2001.03622>`_.

    Questa classe eredita da ``pyvqnet.qnn.vqc.torch.QModule`` e ``torch.nn.Module``.
    Questa classe puo essere aggiunta al modello torch come sottomodulo di ``torch.nn.Module``.

 
    :param num_repetitions_input: numero di volte per codificare l'input in un sottomodulo.
    :param depth_input: number of input dimension .
    :param num_unitary_layers: numero di volte delle porte quantistiche variazionali.
    :param num_repetitions: numero di volte del sottomodulo.
    :param initial: parameter initialization value, default is None
    :param dtype: parameter type, default is None, use float32.
    :param name: class name
    :return: A VQC_QuantumEmbedding instance.

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

    19 diversi ansatz dall'articolo `Expressibility and entangling capability of parameterized quantum circuits for hybrid quantum-classical algorithms <https://arxiv.org/pdf/1905.10876.pdf>`_.

    Questa classe eredita da ``pyvqnet.qnn.vqc.torch.QModule`` e ``torch.nn.Module``.

    Questa classe puo essere aggiunta al modello torch come sottomodulo di ``torch.nn.Module``.

    :param type: Circuit type from 1 to 19, a total of 19 lines.
    :param num_wires: Number of qubits.
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

    Codifica n caratteristiche binarie nello stato di base a n qubit di ``q_machine``. Questa funzione e anche chiamata `VQC_BasisEmbedding`.

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

    Codifica :math:`N` caratteristiche nell'angolo di rotazione di :math:`n` qubit, dove :math:`N \leq n`.
    This function is aliased as `VQC_AngleEmbedding` .

    The rotation can be selected as: 'X' , 'Y' , 'Z', as defined by the ``rotation`` parameter:

    * ``rotation='X'`` Use the feature as the angle of RX rotation.

    * ``rotation='Y'`` Use the feature as the angle of RY rotation.

    * ``rotation='Z'`` Use the feature as the angle of RZ rotation.

    ``wires`` represents the idx of the rotation gate on the qubit.

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

    Encodes a :math:`2^n` feature into an amplitude vector of :math:`n` qubits. This function is aliased as `VQC_AmplitudeEmbedding`.

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

    Encode :math:`n` features into :math:`n` qubits using diagonal gates of an IQP circuit. Alias: ``VQC_IQPEmbedding`` .

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

    Arbitrary single quantum bit rotation quantum logic gate combination. This function alias: ``VQC_RotCircuit`` .

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

    Quantum logic gate combination of controlled Rot single quantum bit rotation. This function alias: ``VQC_CRotCircuit`` .

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

    Controlled Hadamard logic gate quantum circuit. This function alias: ``VQC_Controlled_Hadamard`` .

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

    Controlled-controlled-Z logic gate. Alias: ``VQC_CCZ`` .

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

    Coupled cluster single excitation operator for tensor product of Pauli matrices. Matrix form is given by:

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

    Coupled clustered biexcitation operator for tensor product of Pauli matrices exponentiated, matrix form given by:

    .. math::
        \hat{U}_{pqrs}(\theta) = \mathrm{exp} \{ \theta (\hat{c}_p^\dagger \hat{c}_q^\dagger
        \hat{c}_r \hat{c}_s - \mathrm{H.c.}) \},

    where :math:`\hat{c}` and :math:`\hat{c}^\dagger` are fermion annihilation and
    operators are created and indexed :math:`r, s` and :math:`p, q` on occupied and
    empty molecular orbitals respectively. Use `Jordan-Wigner transformation
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

    Implements the Unitary Coupled Cluster Single and Double Excitations Simulation (UCCSD). UCCSD is a VQE simulation commonly used to run quantum chemistry simulations.

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

    First-order Pauli Z-evolution circuit.

    For 3 qubits and 2 repetitions, the circuit is represented as:

    .. parsed-literal::

        ┌───┐┌──────────────┐┌───┐┌──────────────┐
        ┤ H ├┤ U1(2.0*x[0]) ├┤ H ├┤ U1(2.0*x[0]) ├
        ├───┤├──────────────┤├───┤├──────────────┤
        ┤ H ├┤ U1(2.0*x[1]) ├┤ H ├┤ U1(2.0*x[1]) ├
        ├───┤├──────────────┤├───┤├──────────────┤
        ┤ H ├┤ U1(2.0*x[2]) ├┤ H ├┤ U1(2.0*x[2]) ├
        └───┘└──────────────┘└───┘└──────────────┘

    The Pauli string is fixed to ``Z``. Therefore, the first-order expansion will be a circuit without entanglement gates.

    :param input_feat: Array representing input parameters.
    :param q_machine: Quantum virtual machine.
    :param data_map_func: Parameter mapping matrix, a callable function, designed as: ``data_map_func = lambda x: x``.
    :param rep: Number of times the module is repeated.

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

    Second-order Pauli-Z evolution circuit.

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

    :param input_feat: Array representing input parameters.
    :param q_machine: Quantum virtual machine.
    :param data_map_func: parameter mapping matrix, a callable function.
    :param entanglement: specified entanglement structure.
    :param rep: module repetition times.
    
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

    In this case, we have four single excitations and double excitations to preserve the total spin projection of the Hartree-Fock state.

    The resulting unitary matrix preserves the particle population and prepares the n-qubit system in a superposition of the initial Hartree-Fock state and other states encoding the multi-excitation configuration.

    :param weights: A QTensor of size ``(len(singles) + len(doubles),)`` containing the angles that enter the vqc.qCircuit.single_excitation and vqc.qCircuit.double_excitation operations in sequence
    :param q_machine: The quantum machine.
    :param hf_state: A vector of length ``len(wires)`` occupancy numbers representing the Hartree-Fock state, ``hf_state`` used to initialize the wires.
    :param wires: The qubits to act on.
    :param singles: A sequence of lists with the indices of the two qubits acted on by the single_exitation operation.
    :param doubles: List sequence with the indices of the two qubits acted on by the double_exitation operation.

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

    Implement a circuit that provides an ensemble that can be used to perform accurate single-unit basis rotations. The circuit is derived from the single-particle fermion-determined unitary transformation :math:`U(u)` given in `arXiv:1711.04789 <https://arxiv.org/abs/1711.04789>`_\ 
    
    .. math::
        U(u) = \exp{\left( \sum_{pq} \left[\log u \right]_{pq} (a_p^\dagger a_q - a_q^\dagger a_p) \right)}.

    :math:`U(u)` is obtained by using the scheme given in the paper `Optica, 3, 1460 (2016) <https://opg.optica.org/optica/fulltext.cfm?uri=optica-3-12-1460&id=355743>`_\ .

    :param q_machine: quantum machine.
    :param wires: qubits to act on.
    :param unitary_matrix: matrix specifying the basis for the transformation.
    :param check: check if `unitary_matrix` is a unitary matrix.

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

    Quantum circuit that downsamples data.

    To reduce the number of qubits in the circuit, pairs of qubits are first created in the system. After initially pairing all qubits, a generalized 2-qubit unitary is applied to each pair of qubits. And after applying these two qubit unitaries, a qubit in each pair of qubits is ignored for the rest of the neural network.

    :param sources_wires: Source qubit indices that will be ignored.
    :param sinks_wires: Target qubit indices that will be retained.
    :param params: Input parameters.
    :param q_machine: Quantum virtual machine device.

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


    An automatically differentiable QuantumLayer layer that uses the adjoint matrix approach to calculate gradients, see `Efficient calculation of gradients in classical simulations of variational quantum algorithms <https://arxiv.org/abs/2009.02823>`_ .

    :param general_module: a `pyvqnet.nn.Module` instance built using only the quantum circuit interface under ``pyvqnet.qnn.vqc.torch``.
    :param use_qpanda: Whether to use qpanda line for forward transmission, default: False.
    :param name: Il nome dello strato, default "".

    .. note::

        The QMachine of general_module should set grad_method = "adjoint".

        Currently supports the following parameterized logic gates `RX`, `RY`, `RZ`, `PhaseShift`, `RXX`, `RYY`, `RZZ`, `RZX`, `U1`, `U2`, `U3` and other variational circuits consisting of non-parameter logic gates.


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





Tensor Network Backend Variational Quantum Circuit Module
==========================================================================================

Tensor Network (TN) significantly reduces computational complexity by decomposing a complex tensor into a network of multiple low-dimensional tensors.

Matrix Product State (MPS) is a special form of Tensor Network. MPS represents a quantum state as the product of a series of matrices, thus effectively reducing the number of parameters and the computational complexity.

The following interface is based on the ``torch`` backend, which provides functional support for constructing quantum circuits in tensor networks, including the construction of quantum circuit base classes, quantum logic gates, quantum circuits, and measurements, as well as calculating parameter gradients by automatic differential simulation instead of parameter drift method.

Constructing quantum lines in the MPS way makes up for the support for large-bit quantum line construction.

.. warning::

        Using the following features in this module requires additional installation of ``tensornetwork`` and ``torch``. The default installation of ``pyvqnet`` does not include these two dependencies. Please install them using ``pip install tensornetwork torch``.

.. warning::

        Enables MPS to build quantum lines via the ``use_mps`` parameter in ``TNQMachine``, which supports large-bit (100 and above) quantum line implementations.

.. warning::
        
        Batching is used differently than under classic modules, based on the vmap approach, where the data and parameter construction lines need to be entered in one dimension down, as shown in the sample interface below, and the batching execution must be based on both ``TNQMachine`` and ``TNQModule``.

Base Class
------------------------------------------------

Writing a  variational quantum circuit model on tensornetwork requires inheriting from ``TNQModule``.

TNQModule
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. py:class:: pyvqnet.qnn.vqc.tn.TNQModule(use_jit=False, vectorized_argnums=0, name="")

    .. note::

        This class and its derived classes are only applicable to ``pyvqnet.backends.set_backend("torch")``, do not mix with the ``Module`` under the default ``pyvqnet.nn``.

        The data in ``_buffers`` of this class is of ``torch.Tensor`` type.

        The data in ``_parmeters`` of this class is of ``torch.nn.Parameter`` type.

    :param use_jit: control quantum circuit jit compilation functionality.
    :param vectorized_argnums: the args to be vectorized,
            these arguments should share the same batch shape in the fist dimension,defaults to 0.
    :param name: name of Module.

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

    Classe simulatore per il calcolo quantistico variazionale, inclusi statevector il cui attributo states sono circuiti quantistici.

    Questa classe eredita da ``pyvqnet.nn.torch.TorchModule`` e ``pyvqnet.qnn.QMachine``.

    Questa classe puo essere aggiunta al modello torch come sottomodulo di ``torch.nn.Module``.

    .. warning::
        
        Nel circuito quantistico della rete tensoriale, la funzione ``vmap`` sara abilitata per impostazione predefinita e la dimensione del batch sara scartata nei parametri della porta logica sulla linea.
        Quando si utilizza il parametro di chiamata, se la dimensione e [batch_size, \*], la prima dimensione batch_size viene scartata e le dimensioni successive vengono utilizzate direttamente, ad esempio per i dati di input x[:,1] -> x[1], e anche per il parametro addestrabile, vedere il seguente esempio per l'uso di xx, weights.

    .. note::

        Prima di ogni esecuzione del circuito quantistico completo, devi usare `pyvqnet.qnn.vqc.QMachine.reset_states(batchsize)` per reinizializzare lo stato iniziale nel simulatore e trasmetterlo alle dimensioni
        (batchsize,*) per adattarsi all'addestramento batch dei dati.

    :param num_wires: numero di qubit da utilizzare
    :param dtype: tipo di dato interno utilizzato per il calcolo.
    :param use_mps: apre MPSCircuit per modelli a bit grandi.

    :return: Output a TNQMachine object.

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

        get tensornetwork qmachine states.

Variational quantum logic gate module
------------------------------------------------

The following function interfaces in ``pyvqnet.qnn.vqc`` directly support ``QTensor`` of ``torch`` backend for calculation, import path ``pyvqnet.qnn.vqc.tn``.

.. csv-table:: List of supported pyvqnet.qnn.vqc interfaces
    :file: ./images/same_apis_from_tn.csv

The following quantum circuit modules inherit from ``pyvqnet.qnn.vqc.tn.TNQModule``, where calculations are performed using ``torch.Tensor``.

.. note::

    This class and its derived classes are only applicable to ``pyvqnet.backends.set_backend("torch")``, do not mix with ``Module`` under the default ``pyvqnet.nn``.

    If these classes have non-parameter member variables ``_buffers``, the data in them is of type ``torch.Tensor``.
    If these classes have parameter member variables ``_parmeters``, the data in them is of type ``torch.nn.Parameter``.

I
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. py:class:: pyvqnet.qnn.vqc.tn.I(has_params: bool = False,trainable: bool = False,init_params=None,wires=None,dtype=pyvqnet.kcomplex64,use_dagger=False)
    
    definisce una porta quantistica I.

    Questa classe eredita da ``pyvqnet.qnn.vqc.tn.QModule`` e ``torch.nn.Module``.
    Questa classe puo essere aggiunta al modello torch come sottomodulo di ``torch.nn.Module``.

    :param has_params: whether it has parameters, such as RX, RY and other gates need to be set to True, and those without parameters need to be set to False, the default is False.
    :param trainable: whether it has parameters to be trained. If the layer uses external input data to build the logic gate matrix, set to False. If the parameters to be trained need to be initialized from this layer, it is True, the default is False.
    :param init_params: Initialization parameters used to encode classic data QTensor, the default is None.
    :param wires: Bit index of the line effect, the default is None.
    :param dtype: The data precision of the internal matrix of the logic gate can be set to pyvqnet.kcomplex64 or pyvqnet.kcomplex128, corresponding to float input or double input respectively.
    :param use_dagger: whether to use the transposed conjugate version of the gate, the default is False.
    :return: un'istanza di ``pyvqnet.qnn.vqc.tn.QModule``

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
    
    definisce una porta quantistica Hadamard.

    Questa classe eredita da ``pyvqnet.qnn.vqc.tn.QModule`` e ``torch.nn.Module``.
    Questa classe puo essere aggiunta al modello torch come sottomodulo di ``torch.nn.Module``.

    :param has_params: whether it has parameters, such as RX, RY and other gates need to be set to True, and those without parameters need to be set to False, the default is False.
    :param trainable: whether it has parameters to be trained. If the layer uses external input data to build the logic gate matrix, set to False. If the parameters to be trained need to be initialized from this layer, it is True, the default is False.
    :param init_params: Initialization parameters used to encode classic data QTensor, the default is None.
    :param wires: Bit index of the line effect, the default is None.
    :param dtype: The data precision of the internal matrix of the logic gate can be set to pyvqnet.kcomplex64 or pyvqnet.kcomplex128, corresponding to float input or double input respectively.
    :param use_dagger: whether to use the transposed conjugate version of the gate, the default is False.
    :return: un'istanza di ``pyvqnet.qnn.vqc.tn.QModule``

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
    
    definisce una porta quantistica T.

    Questa classe eredita da ``pyvqnet.qnn.vqc.tn.QModule`` e ``torch.nn.Module``.
    Questa classe puo essere aggiunta al modello torch come sottomodulo di ``torch.nn.Module``.

    :param has_params: whether it has parameters, such as RX, RY and other gates need to be set to True, and those without parameters need to be set to False, the default is False.
    :param trainable: whether it has parameters to be trained. If the layer uses external input data to build the logic gate matrix, set to False. If the parameters to be trained need to be initialized from this layer, it is True, the default is False.
    :param init_params: Initialization parameters used to encode classic data QTensor, the default is None.
    :param wires: Bit index of the line effect, the default is None.
    :param dtype: The data precision of the internal matrix of the logic gate can be set to pyvqnet.kcomplex64 or pyvqnet.kcomplex128, corresponding to float input or double input respectively.
    :param use_dagger: whether to use the transposed conjugate version of the gate, the default is False.
    :return: un'istanza di ``pyvqnet.qnn.vqc.tn.QModule``

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
    
    definisce una porta quantistica S.

    Questa classe eredita da ``pyvqnet.qnn.vqc.tn.QModule`` e ``torch.nn.Module``.
    Questa classe puo essere aggiunta al modello torch come sottomodulo di ``torch.nn.Module``.

    :param has_params: whether it has parameters, such as RX, RY and other gates need to be set to True, and those without parameters need to be set to False, the default is False.
    :param trainable: whether it has parameters to be trained. If the layer uses external input data to build the logic gate matrix, set to False. If the parameters to be trained need to be initialized from this layer, it is True, the default is False.
    :param init_params: Initialization parameters used to encode classic data QTensor, the default is None.
    :param wires: Bit index of the line effect, the default is None.
    :param dtype: The data precision of the internal matrix of the logic gate can be set to pyvqnet.kcomplex64 or pyvqnet.kcomplex128, corresponding to float input or double input respectively.
    :param use_dagger: whether to use the transposed conjugate version of the gate, the default is False.
    :return: un'istanza di ``pyvqnet.qnn.vqc.tn.QModule``

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
    
    definisce una porta quantistica PauliX.

    Questa classe eredita da ``pyvqnet.qnn.vqc.tn.QModule`` e ``torch.nn.Module``.
    Questa classe puo essere aggiunta al modello torch come sottomodulo di ``torch.nn.Module``.


    :param has_params: whether it has parameters, such as RX, RY and other gates need to be set to True, and those without parameters need to be set to False, the default is False.
    :param trainable: whether it has parameters to be trained. If the layer uses external input data to build the logic gate matrix, set to False. If the parameters to be trained need to be initialized from this layer, it is True, the default is False.
    :param init_params: Initialization parameters used to encode classic data QTensor, the default is None.
    :param wires: Bit index of the line effect, the default is None.
    :param dtype: The data precision of the internal matrix of the logic gate can be set to pyvqnet.kcomplex64 or pyvqnet.kcomplex128, corresponding to float input or double input respectively.
    :param use_dagger: whether to use the transposed conjugate version of the gate, the default is False.
    :return: un'istanza di ``pyvqnet.qnn.vqc.tn.QModule``

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
    
    definisce una porta quantistica PauliY.

    Questa classe eredita da ``pyvqnet.qnn.vqc.tn.QModule`` e ``torch.nn.Module``.
    Questa classe puo essere aggiunta al modello torch come sottomodulo di ``torch.nn.Module``.


    :param has_params: whether it has parameters, such as RX, RY and other gates need to be set to True, and those without parameters need to be set to False, the default is False.
    :param trainable: whether it has parameters to be trained. If the layer uses external input data to build the logic gate matrix, set to False. If the parameters to be trained need to be initialized from this layer, it is True, the default is False.
    :param init_params: Initialization parameters used to encode classic data QTensor, the default is None.
    :param wires: Bit index of the line effect, the default is None.
    :param dtype: The data precision of the internal matrix of the logic gate can be set to pyvqnet.kcomplex64 or pyvqnet.kcomplex128, corresponding to float input or double input respectively.
    :param use_dagger: whether to use the transposed conjugate version of the gate, the default is False.
    :return: un'istanza di ``pyvqnet.qnn.vqc.tn.QModule``

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
    
    definisce una porta quantistica PauliZ.

    Questa classe eredita da ``pyvqnet.qnn.vqc.tn.QModule`` e ``torch.nn.Module``.
    Questa classe puo essere aggiunta al modello torch come sottomodulo di ``torch.nn.Module``.


    :param has_params: whether it has parameters, such as RX, RY and other gates need to be set to True, and those without parameters need to be set to False, the default is False.
    :param trainable: whether it has parameters to be trained. If the layer uses external input data to build the logic gate matrix, set to False. If the parameters to be trained need to be initialized from this layer, it is True, the default is False.
    :param init_params: Initialization parameters used to encode classic data QTensor, the default is None.
    :param wires: Bit index of the line effect, the default is None.
    :param dtype: The data precision of the internal matrix of the logic gate can be set to pyvqnet.kcomplex64 or pyvqnet.kcomplex128, corresponding to float input or double input respectively.
    :param use_dagger: whether to use the transposed conjugate version of the gate, the default is False.
    :return: un'istanza di ``pyvqnet.qnn.vqc.tn.QModule``

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
    
    definisce una porta quantistica X1.

    Questa classe eredita da ``pyvqnet.qnn.vqc.tn.QModule`` e ``torch.nn.Module``.
    Questa classe puo essere aggiunta al modello torch come sottomodulo di ``torch.nn.Module``.

    :param has_params: whether it has parameters, such as RX, RY and other gates need to be set to True, and those without parameters need to be set to False, the default is False.
    :param trainable: whether it has parameters to be trained. If the layer uses external input data to build the logic gate matrix, set to False. If the parameters to be trained need to be initialized from this layer, it is True, the default is False.
    :param init_params: Initialization parameters used to encode classic data QTensor, the default is None.
    :param wires: Bit index of the line effect, the default is None.
    :param dtype: The data precision of the internal matrix of the logic gate can be set to pyvqnet.kcomplex64 or pyvqnet.kcomplex128, corresponding to float input or double input respectively.
    :param use_dagger: whether to use the transposed conjugate version of the gate, the default is False.
    :return: un'istanza di ``pyvqnet.qnn.vqc.tn.QModule``

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
    
    definisce una porta quantistica RX.

    Questa classe eredita da ``pyvqnet.qnn.vqc.tn.QModule`` e ``torch.nn.Module``.
    Questa classe puo essere aggiunta al modello torch come sottomodulo di ``torch.nn.Module``.

    :param has_params: whether it has parameters, such as RX, RY and other gates need to be set to True, and those without parameters need to be set to False, the default is False.
    :param trainable: whether it has parameters to be trained. If the layer uses external input data to build the logic gate matrix, set to False. If the parameters to be trained need to be initialized from this layer, it is True, the default is False.
    :param init_params: Initialization parameters used to encode classic data QTensor, the default is None.
    :param wires: Bit index of the line effect, the default is None.
    :param dtype: The data precision of the internal matrix of the logic gate can be set to pyvqnet.kcomplex64 or pyvqnet.kcomplex128, corresponding to float input or double input respectively.
    :param use_dagger: whether to use the transposed conjugate version of the gate, the default is False.
    :return: un'istanza di ``pyvqnet.qnn.vqc.tn.QModule``

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
    
    definisce una porta quantistica RY.

    Questa classe eredita da ``pyvqnet.qnn.vqc.tn.QModule`` e ``torch.nn.Module``.
    Questa classe puo essere aggiunta al modello torch come sottomodulo di ``torch.nn.Module``.

    :param has_params: whether it has parameters, such as RX, RY and other gates need to be set to True, and those without parameters need to be set to False, the default is False.
    :param trainable: whether it has parameters to be trained. If the layer uses external input data to build the logic gate matrix, set to False. If the parameters to be trained need to be initialized from this layer, it is True, the default is False.
    :param init_params: Initialization parameters used to encode classic data QTensor, the default is None.
    :param wires: Bit index of the line effect, the default is None.
    :param dtype: The data precision of the internal matrix of the logic gate can be set to pyvqnet.kcomplex64 or pyvqnet.kcomplex128, corresponding to float input or double input respectively.
    :param use_dagger: whether to use the transposed conjugate version of the gate, the default is False.
    :return: un'istanza di ``pyvqnet.qnn.vqc.tn.QModule``

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
    
    definisce una porta quantistica RZ.

    Questa classe eredita da ``pyvqnet.qnn.vqc.tn.QModule`` e ``torch.nn.Module``.
    Questa classe puo essere aggiunta al modello torch come sottomodulo di ``torch.nn.Module``.

    :param has_params: whether it has parameters, such as RX, RY and other gates need to be set to True, and those without parameters need to be set to False, the default is False.
    :param trainable: whether it has parameters to be trained. If the layer uses external input data to build the logic gate matrix, set to False. If the parameters to be trained need to be initialized from this layer, it is True, the default is False.
    :param init_params: Initialization parameters used to encode classic data QTensor, the default is None.
    :param wires: Bit index of the line effect, the default is None.
    :param dtype: The data precision of the internal matrix of the logic gate can be set to pyvqnet.kcomplex64 or pyvqnet.kcomplex128, corresponding to float input or double input respectively.
    :param use_dagger: whether to use the transposed conjugate version of the gate, the default is False.
    :return: un'istanza di ``pyvqnet.qnn.vqc.tn.QModule``

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
    
    definisce una porta quantistica CRX.

    Questa classe eredita da ``pyvqnet.qnn.vqc.tn.QModule`` e ``torch.nn.Module``.
    Questa classe puo essere aggiunta al modello torch come sottomodulo di ``torch.nn.Module``.

    :param has_params: whether it has parameters, such as RX, RY and other gates need to be set to True, and those without parameters need to be set to False, the default is False.
    :param trainable: whether it has parameters to be trained. If the layer uses external input data to build the logic gate matrix, set to False. If the parameters to be trained need to be initialized from this layer, it is True, the default is False.
    :param init_params: Initialization parameters used to encode classic data QTensor, the default is None.
    :param wires: Bit index of the line effect, the default is None.
    :param dtype: The data precision of the internal matrix of the logic gate can be set to pyvqnet.kcomplex64 or pyvqnet.kcomplex128, corresponding to float input or double input respectively.
    :param use_dagger: whether to use the transposed conjugate version of the gate, the default is False.
    :return: un'istanza di ``pyvqnet.qnn.vqc.tn.QModule``

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
    
    definisce una porta quantistica CRY.

    Questa classe eredita da ``pyvqnet.qnn.vqc.tn.QModule`` e ``torch.nn.Module``.
    Questa classe puo essere aggiunta al modello torch come sottomodulo di ``torch.nn.Module``.

    :param has_params: whether it has parameters, such as RX, RY and other gates need to be set to True, and those without parameters need to be set to False, the default is False.
    :param trainable: whether it has parameters to be trained. If the layer uses external input data to build the logic gate matrix, set to False. If the parameters to be trained need to be initialized from this layer, it is True, the default is False.
    :param init_params: Initialization parameters used to encode classic data QTensor, the default is None.
    :param wires: Bit index of the line effect, the default is None.
    :param dtype: The data precision of the internal matrix of the logic gate can be set to pyvqnet.kcomplex64 or pyvqnet.kcomplex128, corresponding to float input or double input respectively.
    :param use_dagger: whether to use the transposed conjugate version of the gate, the default is False.
    :return: un'istanza di ``pyvqnet.qnn.vqc.tn.QModule``

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
    
    definisce una porta quantistica CRZ.

    Questa classe eredita da ``pyvqnet.qnn.vqc.tn.QModule`` e ``torch.nn.Module``.
    Questa classe puo essere aggiunta al modello torch come sottomodulo di ``torch.nn.Module``.

    :param has_params: whether it has parameters, such as RX, RY and other gates need to be set to True, and those without parameters need to be set to False, the default is False.
    :param trainable: whether it has parameters to be trained. If the layer uses external input data to build the logic gate matrix, set to False. If the parameters to be trained need to be initialized from this layer, it is True, the default is False.
    :param init_params: Initialization parameters used to encode classic data QTensor, the default is None.
    :param wires: Bit index of the line effect, the default is None.
    :param dtype: The data precision of the internal matrix of the logic gate can be set to pyvqnet.kcomplex64 or pyvqnet.kcomplex128, corresponding to float input or double input respectively.
    :param use_dagger: whether to use the transposed conjugate version of the gate, the default is False.
    :return: un'istanza di ``pyvqnet.qnn.vqc.tn.QModule``

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
    
    definisce una porta quantistica U1.

    Questa classe eredita da ``pyvqnet.qnn.vqc.tn.QModule`` e ``torch.nn.Module``.
    Questa classe puo essere aggiunta al modello torch come sottomodulo di ``torch.nn.Module``.

    :param has_params: whether it has parameters, such as RX, RY and other gates need to be set to True, and those without parameters need to be set to False, the default is False.
    :param trainable: whether it has parameters to be trained. If the layer uses external input data to build the logic gate matrix, set to False. If the parameters to be trained need to be initialized from this layer, it is True, the default is False.
    :param init_params: Initialization parameters used to encode classic data QTensor, the default is None.
    :param wires: Bit index of the line effect, the default is None.
    :param dtype: The data precision of the internal matrix of the logic gate can be set to pyvqnet.kcomplex64 or pyvqnet.kcomplex128, corresponding to float input or double input respectively.
    :param use_dagger: whether to use the transposed conjugate version of the gate, the default is False.
    :return: un'istanza di ``pyvqnet.qnn.vqc.tn.QModule``

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
    
    definisce una porta quantistica U2.

    Questa classe eredita da ``pyvqnet.qnn.vqc.tn.QModule`` e ``torch.nn.Module``.
    Questa classe puo essere aggiunta al modello torch come sottomodulo di ``torch.nn.Module``.

    :param has_params: whether it has parameters, such as RX, RY and other gates need to be set to True, and those without parameters need to be set to False, the default is False.
    :param trainable: whether it has parameters to be trained. If the layer uses external input data to build the logic gate matrix, set to False. If the parameters to be trained need to be initialized from this layer, it is True, the default is False.
    :param init_params: Initialization parameters used to encode classic data QTensor, the default is None.
    :param wires: Bit index of the line effect, the default is None.
    :param dtype: The data precision of the internal matrix of the logic gate can be set to pyvqnet.kcomplex64 or pyvqnet.kcomplex128, corresponding to float input or double input respectively.
    :param use_dagger: whether to use the transposed conjugate version of the gate, the default is False.
    :return: un'istanza di ``pyvqnet.qnn.vqc.tn.QModule``

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
    
    definisce una porta quantistica U3.

    Questa classe eredita da ``pyvqnet.qnn.vqc.tn.QModule`` e ``torch.nn.Module``.
    Questa classe puo essere aggiunta al modello torch come sottomodulo di ``torch.nn.Module``.

    :param has_params: whether it has parameters, such as RX, RY and other gates need to be set to True, and those without parameters need to be set to False, the default is False.
    :param trainable: whether it has parameters to be trained. If the layer uses external input data to build the logic gate matrix, set to False. If the parameters to be trained need to be initialized from this layer, it is True, the default is False.
    :param init_params: Initialization parameters used to encode classic data QTensor, the default is None.
    :param wires: Bit index of the line effect, the default is None.
    :param dtype: The data precision of the internal matrix of the logic gate can be set to pyvqnet.kcomplex64 or pyvqnet.kcomplex128, corresponding to float input or double input respectively.
    :param use_dagger: whether to use the transposed conjugate version of the gate, the default is False.
    :return: un'istanza di ``pyvqnet.qnn.vqc.tn.QModule``

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
    
    definisce una porta quantistica CNOT, alias `CX`.

    Questa classe eredita da ``pyvqnet.qnn.vqc.tn.QModule`` e ``torch.nn.Module``.
    Questa classe puo essere aggiunta al modello torch come sottomodulo di ``torch.nn.Module``.

    :param has_params: whether it has parameters, such as RX, RY and other gates need to be set to True, and those without parameters need to be set to False, the default is False.
    :param trainable: whether it has parameters to be trained. If the layer uses external input data to build the logic gate matrix, set to False. If the parameters to be trained need to be initialized from this layer, it is True, the default is False.
    :param init_params: Initialization parameters used to encode classic data QTensor, the default is None.
    :param wires: Bit index of the line effect, the default is None.
    :param dtype: The data precision of the internal matrix of the logic gate can be set to pyvqnet.kcomplex64 or pyvqnet.kcomplex128, corresponding to float input or double input respectively.
    :param use_dagger: whether to use the transposed conjugate version of the gate, the default is False.
    :return: un'istanza di ``pyvqnet.qnn.vqc.tn.QModule``

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
    
    definisce una porta quantistica CY.

    Questa classe eredita da ``pyvqnet.qnn.vqc.tn.QModule`` e ``torch.nn.Module``.
    Questa classe puo essere aggiunta al modello torch come sottomodulo di ``torch.nn.Module``.

    :param has_params: whether it has parameters, such as RX, RY and other gates need to be set to True, and those without parameters need to be set to False, the default is False.
    :param trainable: whether it has parameters to be trained. If the layer uses external input data to build the logic gate matrix, set to False. If the parameters to be trained need to be initialized from this layer, it is True, the default is False.
    :param init_params: Initialization parameters used to encode classic data QTensor, the default is None.
    :param wires: Bit index of the line effect, the default is None.
    :param dtype: The data precision of the internal matrix of the logic gate can be set to pyvqnet.kcomplex64 or pyvqnet.kcomplex128, corresponding to float input or double input respectively.
    :param use_dagger: whether to use the transposed conjugate version of the gate, the default is False.
    :return: un'istanza di ``pyvqnet.qnn.vqc.tn.QModule``

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
    
    definisce una porta quantistica CZ.

    Questa classe eredita da ``pyvqnet.qnn.vqc.tn.QModule`` e ``torch.nn.Module``.
    Questa classe puo essere aggiunta al modello torch come sottomodulo di ``torch.nn.Module``.

    :param has_params: whether it has parameters, such as RX, RY and other gates need to be set to True, and those without parameters need to be set to False, the default is False.
    :param trainable: whether it has parameters to be trained. If the layer uses external input data to build the logic gate matrix, set to False. If the parameters to be trained need to be initialized from this layer, it is True, the default is False.
    :param init_params: Initialization parameters used to encode classic data QTensor, the default is None.
    :param wires: Bit index of the line effect, the default is None.
    :param dtype: The data precision of the internal matrix of the logic gate can be set to pyvqnet.kcomplex64 or pyvqnet.kcomplex128, corresponding to float input or double input respectively.
    :param use_dagger: whether to use the transposed conjugate version of the gate, the default is False.
    :return: un'istanza di ``pyvqnet.qnn.vqc.tn.QModule``

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
    
    definisce una porta quantistica CR.

    Questa classe eredita da ``pyvqnet.qnn.vqc.tn.QModule`` e ``torch.nn.Module``.
    Questa classe puo essere aggiunta al modello torch come sottomodulo di ``torch.nn.Module``.

    :param has_params: whether it has parameters, such as RX, RY and other gates need to be set to True, and those without parameters need to be set to False, the default is False.
    :param trainable: whether it has parameters to be trained. If the layer uses external input data to build the logic gate matrix, set to False. If the parameters to be trained need to be initialized from this layer, it is True, the default is False.
    :param init_params: Initialization parameters used to encode classic data QTensor, the default is None.
    :param wires: Bit index of the line effect, the default is None.
    :param dtype: The data precision of the internal matrix of the logic gate can be set to pyvqnet.kcomplex64 or pyvqnet.kcomplex128, corresponding to float input or double input respectively.
    :param use_dagger: whether to use the transposed conjugate version of the gate, the default is False.
    :return: un'istanza di ``pyvqnet.qnn.vqc.tn.QModule``

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
    
    definisce una porta quantistica SWAP.

    Questa classe eredita da ``pyvqnet.qnn.vqc.tn.QModule`` e ``torch.nn.Module``.
    Questa classe puo essere aggiunta al modello torch come sottomodulo di ``torch.nn.Module``.

    :param has_params: whether it has parameters, such as RX, RY and other gates need to be set to True, and those without parameters need to be set to False, the default is False.
    :param trainable: whether it has parameters to be trained. If the layer uses external input data to build the logic gate matrix, set to False. If the parameters to be trained need to be initialized from this layer, it is True, the default is False.
    :param init_params: Initialization parameters used to encode classic data QTensor, the default is None.
    :param wires: Bit index of the line effect, the default is None.
    :param dtype: The data precision of the internal matrix of the logic gate can be set to pyvqnet.kcomplex64 or pyvqnet.kcomplex128, corresponding to float input or double input respectively.
    :param use_dagger: whether to use the transposed conjugate version of the gate, the default is False.
    :return: un'istanza di ``pyvqnet.qnn.vqc.tn.QModule``

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
    
    definisce una porta quantistica SWAP.

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

    Questa classe eredita da ``pyvqnet.qnn.vqc.tn.QModule`` e ``torch.nn.Module``.
    Questa classe puo essere aggiunta al modello torch come sottomodulo di ``torch.nn.Module``.

    :param has_params: whether it has parameters, such as RX, RY and other gates need to be set to True, and those without parameters need to be set to False, the default is False.
    :param trainable: whether it has parameters to be trained. If the layer uses external input data to build the logic gate matrix, set to False. If the parameters to be trained need to be initialized from this layer, it is True, the default is False.
    :param init_params: Initialization parameters used to encode classic data QTensor, the default is None.
    :param wires: Bit index of the line effect, the default is None.
    :param dtype: The data precision of the internal matrix of the logic gate can be set to pyvqnet.kcomplex64 or pyvqnet.kcomplex128, corresponding to float input or double input respectively.
    :param use_dagger: whether to use the transposed conjugate version of the gate, the default is False.
    :return: un'istanza di ``pyvqnet.qnn.vqc.tn.QModule``

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
    
    definisce una porta quantistica RXX.

    Questa classe eredita da ``pyvqnet.qnn.vqc.tn.QModule`` e ``torch.nn.Module``.
    Questa classe puo essere aggiunta al modello torch come sottomodulo di ``torch.nn.Module``.

    :param has_params: whether it has parameters, such as RX, RY and other gates need to be set to True, and those without parameters need to be set to False, the default is False.
    :param trainable: whether it has parameters to be trained. If the layer uses external input data to build the logic gate matrix, set to False. If the parameters to be trained need to be initialized from this layer, it is True, the default is False.
    :param init_params: Initialization parameters used to encode classic data QTensor, the default is None.
    :param wires: Bit index of the line effect, the default is None.
    :param dtype: The data precision of the internal matrix of the logic gate can be set to pyvqnet.kcomplex64 or pyvqnet.kcomplex128, corresponding to float input or double input respectively.
    :param use_dagger: whether to use the transposed conjugate version of the gate, the default is False.
    :return: un'istanza di ``pyvqnet.qnn.vqc.tn.QModule``

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
    
    definisce una porta quantistica RYY.

    Questa classe eredita da ``pyvqnet.qnn.vqc.tn.QModule`` e ``torch.nn.Module``.
    Questa classe puo essere aggiunta al modello torch come sottomodulo di ``torch.nn.Module``.

    :param has_params: whether it has parameters, such as RX, RY and other gates need to be set to True, and those without parameters need to be set to False, the default is False.
    :param trainable: whether it has parameters to be trained. If the layer uses external input data to build the logic gate matrix, set to False. If the parameters to be trained need to be initialized from this layer, it is True, the default is False.
    :param init_params: Initialization parameters used to encode classic data QTensor, the default is None.
    :param wires: Bit index of the line effect, the default is None.
    :param dtype: The data precision of the internal matrix of the logic gate can be set to pyvqnet.kcomplex64 or pyvqnet.kcomplex128, corresponding to float input or double input respectively.
    :param use_dagger: whether to use the transposed conjugate version of the gate, the default is False.
    :return: un'istanza di ``pyvqnet.qnn.vqc.tn.QModule``

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
    
    definisce una porta quantistica RZZ.

    Questa classe eredita da ``pyvqnet.qnn.vqc.tn.QModule`` e ``torch.nn.Module``.
    Questa classe puo essere aggiunta al modello torch come sottomodulo di ``torch.nn.Module``.

    :param has_params: whether it has parameters, such as RX, RY and other gates need to be set to True, and those without parameters need to be set to False, the default is False.
    :param trainable: whether it has parameters to be trained. If the layer uses external input data to build the logic gate matrix, set to False. If the parameters to be trained need to be initialized from this layer, it is True, the default is False.
    :param init_params: Initialization parameters used to encode classic data QTensor, the default is None.
    :param wires: Bit index of the line effect, the default is None.
    :param dtype: The data precision of the internal matrix of the logic gate can be set to pyvqnet.kcomplex64 or pyvqnet.kcomplex128, corresponding to float input or double input respectively.
    :param use_dagger: whether to use the transposed conjugate version of the gate, the default is False.
    :return: un'istanza di ``pyvqnet.qnn.vqc.tn.QModule``

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
    
    definisce una porta quantistica RZX.

    Questa classe eredita da ``pyvqnet.qnn.vqc.tn.QModule`` e ``torch.nn.Module``.
    Questa classe puo essere aggiunta al modello torch come sottomodulo di ``torch.nn.Module``.

    :param has_params: whether it has parameters, such as RX, RY and other gates need to be set to True, and those without parameters need to be set to False, the default is False.
    :param trainable: whether it has parameters to be trained. If the layer uses external input data to build the logic gate matrix, set to False. If the parameters to be trained need to be initialized from this layer, it is True, the default is False.
    :param init_params: Initialization parameters used to encode classic data QTensor, the default is None.
    :param wires: Bit index of the line effect, the default is None.
    :param dtype: The data precision of the internal matrix of the logic gate can be set to pyvqnet.kcomplex64 or pyvqnet.kcomplex128, corresponding to float input or double input respectively.
    :param use_dagger: whether to use the transposed conjugate version of the gate, the default is False.
    :return: un'istanza di ``pyvqnet.qnn.vqc.tn.QModule``

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
    
    definisce una porta quantistica Toffoli.

    Questa classe eredita da ``pyvqnet.qnn.vqc.tn.QModule`` e ``torch.nn.Module``.
    Questa classe puo essere aggiunta al modello torch come sottomodulo di ``torch.nn.Module``.

    :param has_params: whether it has parameters, such as RX, RY and other gates need to be set to True, and those without parameters need to be set to False, the default is False.
    :param trainable: whether it has parameters to be trained. If the layer uses external input data to build the logic gate matrix, set to False. If the parameters to be trained need to be initialized from this layer, it is True, the default is False.
    :param init_params: Initialization parameters used to encode classic data QTensor, the default is None.
    :param wires: Bit index of the line effect, the default is None.
    :param dtype: The data precision of the internal matrix of the logic gate can be set to pyvqnet.kcomplex64 or pyvqnet.kcomplex128, corresponding to float input or double input respectively.
    :param use_dagger: whether to use the transposed conjugate version of the gate, the default is False.
    :return: un'istanza di ``pyvqnet.qnn.vqc.tn.QModule``

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
    
    definisce una porta quantistica IsingXX.

    Questa classe eredita da ``pyvqnet.qnn.vqc.tn.QModule`` e ``torch.nn.Module``.
    Questa classe puo essere aggiunta al modello torch come sottomodulo di ``torch.nn.Module``.

    :param has_params: whether it has parameters, such as RX, RY and other gates need to be set to True, and those without parameters need to be set to False, the default is False.
    :param trainable: whether it has parameters to be trained. If the layer uses external input data to build the logic gate matrix, set to False. If the parameters to be trained need to be initialized from this layer, it is True, the default is False.
    :param init_params: Initialization parameters used to encode classic data QTensor, the default is None.
    :param wires: Bit index of the line effect, the default is None.
    :param dtype: The data precision of the internal matrix of the logic gate can be set to pyvqnet.kcomplex64 or pyvqnet.kcomplex128, corresponding to float input or double input respectively.
    :param use_dagger: whether to use the transposed conjugate version of the gate, the default is False.
    :return: un'istanza di ``pyvqnet.qnn.vqc.tn.QModule``

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
    
    definisce una porta quantistica IsingYY.

    Questa classe eredita da ``pyvqnet.qnn.vqc.tn.QModule`` e ``torch.nn.Module``.
    Questa classe puo essere aggiunta al modello torch come sottomodulo di ``torch.nn.Module``.

    :param has_params: whether it has parameters, such as RX, RY and other gates need to be set to True, and those without parameters need to be set to False, the default is False.
    :param trainable: whether it has parameters to be trained. If the layer uses external input data to build the logic gate matrix, set to False. If the parameters to be trained need to be initialized from this layer, it is True, the default is False.
    :param init_params: Initialization parameters used to encode classic data QTensor, the default is None.
    :param wires: Bit index of the line effect, the default is None.
    :param dtype: The data precision of the internal matrix of the logic gate can be set to pyvqnet.kcomplex64 or pyvqnet.kcomplex128, corresponding to float input or double input respectively.
    :param use_dagger: whether to use the transposed conjugate version of the gate, the default is False.
    :return: un'istanza di ``pyvqnet.qnn.vqc.tn.QModule``

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
    
    definisce una porta quantistica IsingZZ.

    Questa classe eredita da ``pyvqnet.qnn.vqc.tn.QModule`` e ``torch.nn.Module``.
    Questa classe puo essere aggiunta al modello torch come sottomodulo di ``torch.nn.Module``.

    :param has_params: whether it has parameters, such as RX, RY and other gates need to be set to True, and those without parameters need to be set to False, the default is False.
    :param trainable: whether it has parameters to be trained. If the layer uses external input data to build the logic gate matrix, set to False. If the parameters to be trained need to be initialized from this layer, it is True, the default is False.
    :param init_params: Initialization parameters used to encode classic data QTensor, the default is None.
    :param wires: Bit index of the line effect, the default is None.
    :param dtype: The data precision of the internal matrix of the logic gate can be set to pyvqnet.kcomplex64 or pyvqnet.kcomplex128, corresponding to float input or double input respectively.
    :param use_dagger: whether to use the transposed conjugate version of the gate, the default is False.
    :return: un'istanza di ``pyvqnet.qnn.vqc.tn.QModule``

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
    
    definisce una porta quantistica IsingXY.

    Questa classe eredita da ``pyvqnet.qnn.vqc.tn.QModule`` e ``torch.nn.Module``.
    Questa classe puo essere aggiunta al modello torch come sottomodulo di ``torch.nn.Module``.

    :param has_params: whether it has parameters, such as RX, RY and other gates need to be set to True, and those without parameters need to be set to False, the default is False.
    :param trainable: whether it has parameters to be trained. If the layer uses external input data to build the logic gate matrix, set to False. If the parameters to be trained need to be initialized from this layer, it is True, the default is False.
    :param init_params: Initialization parameters used to encode classic data QTensor, the default is None.
    :param wires: Bit index of the line effect, the default is None.
    :param dtype: The data precision of the internal matrix of the logic gate can be set to pyvqnet.kcomplex64 or pyvqnet.kcomplex128, corresponding to float input or double input respectively.
    :param use_dagger: whether to use the transposed conjugate version of the gate, the default is False.
    :return: un'istanza di ``pyvqnet.qnn.vqc.tn.QModule``

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
    
    definisce una porta quantistica PhaseShift.

    Questa classe eredita da ``pyvqnet.qnn.vqc.tn.QModule`` e ``torch.nn.Module``.
    Questa classe puo essere aggiunta al modello torch come sottomodulo di ``torch.nn.Module``.

    :param has_params: whether it has parameters, such as RX, RY and other gates need to be set to True, and those without parameters need to be set to False, the default is False.
    :param trainable: whether it has parameters to be trained. If the layer uses external input data to build the logic gate matrix, set to False. If the parameters to be trained need to be initialized from this layer, it is True, the default is False.
    :param init_params: Initialization parameters used to encode classic data QTensor, the default is None.
    :param wires: Bit index of the line effect, the default is None.
    :param dtype: The data precision of the internal matrix of the logic gate can be set to pyvqnet.kcomplex64 or pyvqnet.kcomplex128, corresponding to float input or double input respectively.
    :param use_dagger: whether to use the transposed conjugate version of the gate, the default is False.
    :return: un'istanza di ``pyvqnet.qnn.vqc.tn.QModule``

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
    
    definisce una porta quantistica MultiRZ.

    Questa classe eredita da ``pyvqnet.qnn.vqc.tn.QModule`` e ``torch.nn.Module``.
    Questa classe puo essere aggiunta al modello torch come sottomodulo di ``torch.nn.Module``.

    :param has_params: whether it has parameters, such as RX, RY and other gates need to be set to True, and those without parameters need to be set to False, the default is False.
    :param trainable: whether it has parameters to be trained. If the layer uses external input data to build the logic gate matrix, set to False. If the parameters to be trained need to be initialized from this layer, it is True, the default is False.
    :param init_params: Initialization parameters used to encode classic data QTensor, the default is None.
    :param wires: Bit index of the line effect, the default is None.
    :param dtype: The data precision of the internal matrix of the logic gate can be set to pyvqnet.kcomplex64 or pyvqnet.kcomplex128, corresponding to float input or double input respectively.
    :param use_dagger: whether to use the transposed conjugate version of the gate, the default is False.
    :return: un'istanza di ``pyvqnet.qnn.vqc.tn.QModule``

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
    
    definisce una porta quantistica SDG.

    Questa classe eredita da ``pyvqnet.qnn.vqc.tn.QModule`` e ``torch.nn.Module``.
    Questa classe puo essere aggiunta al modello torch come sottomodulo di ``torch.nn.Module``.

    :param has_params: whether it has parameters, such as RX, RY and other gates need to be set to True, and those without parameters need to be set to False, the default is False.
    :param trainable: whether it has parameters to be trained. If the layer uses external input data to build the logic gate matrix, set to False. If the parameters to be trained need to be initialized from this layer, it is True, the default is False.
    :param init_params: Initialization parameters used to encode classic data QTensor, the default is None.
    :param wires: Bit index of the line effect, the default is None.
    :param dtype: The data precision of the internal matrix of the logic gate can be set to pyvqnet.kcomplex64 or pyvqnet.kcomplex128, corresponding to float input or double input respectively.
    :param use_dagger: whether to use the transposed conjugate version of the gate, the default is False.
    :return: un'istanza di ``pyvqnet.qnn.vqc.tn.QModule``

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
    
    definisce una porta quantistica SDG.

    Questa classe eredita da ``pyvqnet.qnn.vqc.tn.QModule`` e ``torch.nn.Module``.
    Questa classe puo essere aggiunta al modello torch come sottomodulo di ``torch.nn.Module``.

    :param has_params: whether it has parameters, such as RX, RY and other gates need to be set to True, and those without parameters need to be set to False, the default is False.
    :param trainable: whether it has parameters to be trained. If the layer uses external input data to build the logic gate matrix, set to False. If the parameters to be trained need to be initialized from this layer, it is True, the default is False.
    :param init_params: Initialization parameters used to encode classic data QTensor, the default is None.
    :param wires: Bit index of the line effect, the default is None.
    :param dtype: The data precision of the internal matrix of the logic gate can be set to pyvqnet.kcomplex64 or pyvqnet.kcomplex128, corresponding to float input or double input respectively.
    :param use_dagger: whether to use the transposed conjugate version of the gate, the default is False.
    :return: un'istanza di ``pyvqnet.qnn.vqc.tn.QModule``

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
    
    definisce una porta quantistica ControlledPhaseShift.

    Questa classe eredita da ``pyvqnet.qnn.vqc.tn.QModule`` e ``torch.nn.Module``.
    Questa classe puo essere aggiunta al modello torch come sottomodulo di ``torch.nn.Module``.

    :param has_params: whether it has parameters, such as RX, RY and other gates need to be set to True, and those without parameters need to be set to False, the default is False.
    :param trainable: whether it has parameters to be trained. If the layer uses external input data to build the logic gate matrix, set to False. If the parameters to be trained need to be initialized from this layer, it is True, the default is False.
    :param init_params: Initialization parameters used to encode classic data QTensor, the default is None.
    :param wires: Bit index of the line effect, the default is None.
    :param dtype: The data precision of the internal matrix of the logic gate can be set to pyvqnet.kcomplex64 or pyvqnet.kcomplex128, corresponding to float input or double input respectively.
    :param use_dagger: whether to use the transposed conjugate version of the gate, the default is False.
    :return: un'istanza di ``pyvqnet.qnn.vqc.tn.QModule``

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


Measurements API
------------------------------

VQC_Purity
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^


.. py:function:: pyvqnet.qnn.vqc.tn.VQC_Purity(state, qubits_idx, num_wires, use_tn=False)

    Calcola la purezza su un particolare qubit ``qubits_idx`` a partire dal vettore di stato ``state``.

    .. math::
        \gamma = \text{Tr}(\rho^2)

    where :math:`\rho` is a density matrix. The purity of a normalized quantum state satisfies :math:`\frac{1}{d} \leq \gamma \leq 1` ,
    where :math:`d` is the dimension of the Hilbert space.
    La purezza dello stato puro e 1.

    :param state: Stato quantistico ottenuto da TNQMachine.get_states()
    :param qubits_idx: Indice del qubit per cui calcolare la purezza
    :param num_wires: Indice del qubit
    :param use_tn: per utilizzare tensornetwork deve essere impostato a True, default False

    :return: purezza

    .. note::
        
        batch_size richiede TNQModule.

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

    Computes the density matrix of quantum states ``state`` over a specific set of qubits ``indices`` .

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

    Calcola il risultato della misurazione di probabilita del circuito quantistico su un bit specifico.

    Questa classe eredita da ``pyvqnet.qnn.vqc.tn.QModule`` e ``torch.nn.Module``.
    Questa classe puo essere aggiunta al modello torch come sottomodulo di ``torch.nn.Module``.

    :param wires: L'indice del bit di misurazione, lista, tupla o intero.
    :param name: Il nome del modulo, default: "".
    :return: The measurement result, QTensor.

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

    Calcola i risultati di misurazione dei circuiti quantistici, supporta l'input obs come operatori di Pauli multipli o singoli o Hamiltoniane.
    
    For example:

    {\'X0\': 0.23} indicates a PauliX effect on qubit 0, with a coefficient of 0.23.

    {\'X1 Z2\': 2.4,\'Y2\': -0.5} corresponds to the observed value 2.4 * X1 @ Z2 - 0.5 * Y2.

    [{\'X1 Z2\': 4,\'Z1 Z0\': 3},{\'X1 Y2 Z0\': 3.5}] corresponds to the two observed values 4 * X1 @ Z2 + 3 * Z1 @ Z0 and 3.5 * X1 @ Y2 @ Z0.
        
    Questa classe eredita da ``pyvqnet.qnn.vqc.tn.QModule`` e ``torch.nn.Module``.

    Questa classe puo essere aggiunta al modello torch come sottomodulo di ``torch.nn.Module``.

    :param obs: osservabile.
    :param name: nome del modulo, default: "".
    :return: risultato della misurazione, QTensor.

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

    Ottiene risultati campione con shot su wires specifici.

    Questa classe eredita da ``pyvqnet.qnn.vqc.tn.QModule`` e ``torch.nn.Module``.
    Questa classe puo essere aggiunta al modello torch come sottomodulo di ``torch.nn.Module``.


    :param wires: Indice del qubit campionato. Valore predefinito: None, utilizza tutti i bit del simulatore in fase di esecuzione.
    :param obs: This value can only be None.
    :param shots: Sample repetition count, default value: 1.
    :param name: Il nome di questo modulo, valore predefinito: "".
    :return: una classe di metodo di misurazione

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

    Calcola l'aspettativa di una quantita Hermitiana in un circuito quantistico.

    Questa classe eredita da ``pyvqnet.qnn.vqc.tn.QModule`` e ``torch.nn.Module``.
    Questa classe puo essere aggiunta al modello torch come sottomodulo di ``torch.nn.Module``.


    :param obs: quantita Hermitiana.
    :param name: nome del modulo, default: "".
    :return: risultato atteso, QTensor.

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

Modelli comuni per circuiti quantistici
--------------------------------------------

VQC_HardwareEfficientAnsatz
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^


.. py:class:: pyvqnet.qnn.vqc.tn.VQC_HardwareEfficientAnsatz(n_qubits,single_rot_gate_list,entangle_gate="CNOT",entangle_rules='linear',depth=1,initial=None,dtype=None)

    Implementazione di Hardware Efficient Ansatz introdotto nell'articolo: `Hardware-efficient Variational Quantum Eigensolver for Small Molecules <https://arxiv.org/pdf/1704.05018.pdf>`__ .

    Questa classe eredita da ``pyvqnet.qnn.vqc.tn.QModule`` e ``torch.nn.Module``.

    Questa classe puo essere aggiunta al modello torch come sottomodulo di ``torch.nn.Module``.


    :param n_qubits: Numero di qubit.
    :param single_rot_gate_list: Una lista di porte di rotazione a singolo qubit costruita da una o piu porte di rotazione che agiscono su ogni qubit. Attualmente supporta Rx, Ry, Rz.
    :param entangle_gate: La porta di entanglement non parametrizzata. Sono supportati CNOT, CZ. Default: CNOT.
    :param entangle_rules: Come la porta di entanglement viene utilizzata nel circuito. 'linear' significa che la porta di entanglement agira su ogni qubit vicino. 'all' significa che la porta di entanglement agira su qualsiasi coppia di qubit. Default: linear.
    :param depth: La profondita dell'ansatz, default: 1.
    :param initial: valore iniziale uguale per tutti i parametri, default: None, questo modulo inizializzera i parametri casualmente.
    :param dtype: tipo di dato dei parametri.
    :return: un'istanza di VQC_HardwareEfficientAnsatz.

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

    Uno strato costituito da una rotazione a singolo qubit con singolo parametro su ogni qubit, seguito da molteplici porte CNOT in una combinazione a catena chiusa o ad anello.

    Un anello di porte CNOT collega ogni qubit ai suoi vicini e, infine, il qubit a e considerato vicino del qubit a.

    Questa classe eredita da ``pyvqnet.qnn.vqc.tn.QModule`` e ``torch.nn.Module``.

    Questa classe puo essere aggiunta al modello torch come sottomodulo di ``torch.nn.Module``.

    :param num_layers: numero di strati ripetuti, default: 1.
    :param num_qubits: number of qubits, default: 1.
    :param rotation: one-parameter single-qubit gate to use, default: `RX`
    :param initial: valore inizializzato uguale per tutti i parametri, default: None, i parametri verranno inizializzati casualmente.
    :param dtype: data type of parameter, default:None,use float32.
    :return: A VQC_BasicEntanglerTemplate instance

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

    Strati costituiti da rotazioni a singolo qubit e entangler, come in `circuit-centric classifier design <https://arxiv.org/abs/1804.00633>`__ .

    Questa classe eredita da ``pyvqnet.qnn.vqc.tn.QModule`` e ``torch.nn.Module``.
    Questa classe puo essere aggiunta al modello torch come sottomodulo di ``torch.nn.Module``.


    :param num_layers: numero di strati ripetuti, default: 1.
    :param num_qubits: number of qubits, default: 1.
    :param ranges: sequenza che determina l'iperparametro di intervallo per ogni strato successivo; default: None
                                using :math: `r=l \mod M` for the :math:`l` th layer and :math:`M` qubits.
    :param initial: valore iniziale per tutti i parametri, default: None, inizializzato casualmente.
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
    
    Usa RZ, RY, RZ per creare circuiti quantistici variazionali per codificare dati classici in stati quantistici.
    Reference `Quantum embeddings for machine learning <https://arxiv.org/abs/2001.03622>`_.

    Questa classe eredita da ``pyvqnet.qnn.vqc.tn.QModule`` e ``torch.nn.Module``.
    Questa classe puo essere aggiunta al modello torch come sottomodulo di ``torch.nn.Module``.

 
    :param num_repetitions_input: numero di volte per codificare l'input in un sottomodulo.
    :param depth_input: number of input dimension .
    :param num_unitary_layers: numero di volte delle porte quantistiche variazionali.
    :param num_repetitions: numero di volte del sottomodulo.
    :param initial: parameter initialization value, default is None
    :param dtype: parameter type, default is None, use float32.
    :param name: class name
    :return: A VQC_QuantumEmbedding instance.

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

    19 diversi ansatz dall'articolo `Expressibility and entangling capability of parameterized quantum circuits for hybrid quantum-classical algorithms <https://arxiv.org/pdf/1905.10876.pdf>`_.

    Questa classe eredita da ``pyvqnet.qnn.vqc.tn.QModule`` e ``torch.nn.Module``.

    Questa classe puo essere aggiunta al modello torch come sottomodulo di ``torch.nn.Module``.

    :param type: Circuit type from 1 to 19, a total of 19 lines.
    :param num_wires: Number of qubits.
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

    Codifica n caratteristiche binarie nello stato di base a n qubit di ``q_machine``. Questa funzione e anche chiamata `VQC_BasisEmbedding`.

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

    Codifica :math:`N` caratteristiche nell'angolo di rotazione di :math:`n` qubit, dove :math:`N \leq n`.
    This function is aliased as `VQC_AngleEmbedding` .

    The rotation can be selected as: 'X' , 'Y' , 'Z', as defined by the ``rotation`` parameter:

    * ``rotation='X'`` Use the feature as the angle of RX rotation.

    * ``rotation='Y'`` Use the feature as the angle of RY rotation.

    * ``rotation='Z'`` Use the feature as the angle of RZ rotation.

    ``wires`` represents the idx of the rotation gate on the qubit.

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

    Encodes a :math:`2^n` feature into an amplitude vector of :math:`n` qubits. This function is aliased as `VQC_AmplitudeEmbedding`.

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

    Encode :math:`n` features into :math:`n` qubits using diagonal gates of an IQP circuit. Alias: ``VQC_IQPEmbedding`` .

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

    Arbitrary single quantum bit rotation quantum logic gate combination. This function alias: ``VQC_RotCircuit`` .

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

    Quantum logic gate combination of controlled Rot single quantum bit rotation. This function alias: ``VQC_CRotCircuit`` .

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

    Controlled Hadamard logic gate quantum circuit. This function alias: ``VQC_Controlled_Hadamard`` .

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

    Controlled-controlled-Z logic gate. Alias: ``VQC_CCZ`` .

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

    Coupled cluster single excitation operator for tensor product of Pauli matrices. Matrix form is given by:

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

    Coupled clustered biexcitation operator for tensor product of Pauli matrices exponentiated, matrix form given by:

    .. math::
        \hat{U}_{pqrs}(\theta) = \mathrm{exp} \{ \theta (\hat{c}_p^\dagger \hat{c}_q^\dagger
        \hat{c}_r \hat{c}_s - \mathrm{H.c.}) \},

    where :math:`\hat{c}` and :math:`\hat{c}^\dagger` are fermion annihilation and
    operators are created and indexed :math:`r, s` and :math:`p, q` on occupied and
    empty molecular orbitals respectively. Use `Jordan-Wigner transformation
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

    Implements the Unitary Coupled Cluster Single and Double Excitations Simulation (UCCSD). UCCSD is a VQE simulation commonly used to run quantum chemistry simulations.

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

    First-order Pauli Z-evolution circuit.

    For 3 qubits and 2 repetitions, the circuit is represented as:

    .. parsed-literal::

        ┌───┐┌──────────────┐┌───┐┌──────────────┐
        ┤ H ├┤ U1(2.0*x[0]) ├┤ H ├┤ U1(2.0*x[0]) ├
        ├───┤├──────────────┤├───┤├──────────────┤
        ┤ H ├┤ U1(2.0*x[1]) ├┤ H ├┤ U1(2.0*x[1]) ├
        ├───┤├──────────────┤├───┤├──────────────┤
        ┤ H ├┤ U1(2.0*x[2]) ├┤ H ├┤ U1(2.0*x[2]) ├
        └───┘└──────────────┘└───┘└──────────────┘

    The Pauli string is fixed to ``Z``. Therefore, the first-order expansion will be a circuit without entanglement gates.

    :param input_feat: Array representing input parameters.
    :param q_machine: Quantum virtual machine.
    :param data_map_func: Parameter mapping matrix, a callable function, designed as: ``data_map_func = lambda x: x``.
    :param rep: Number of times the module is repeated.

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

    Second-order Pauli-Z evolution circuit.

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

    :param input_feat: Array representing input parameters.
    :param q_machine: Quantum virtual machine.
    :param data_map_func: parameter mapping matrix, a callable function.
    :param entanglement: specified entanglement structure.
    :param rep: module repetition times.
    
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

    In this case, we have four single excitations and double excitations to preserve the total spin projection of the Hartree-Fock state.

    The resulting unitary matrix preserves the particle population and prepares the n-qubit system in a superposition of the initial Hartree-Fock state and other states encoding the multi-excitation configuration.

    :param weights: A QTensor of size ``(len(singles) + len(doubles),)`` containing the angles that enter the vqc.qCircuit.single_excitation and vqc.qCircuit.double_excitation operations in sequence
    :param q_machine: The quantum machine.
    :param hf_state: A vector of length ``len(wires)`` occupancy numbers representing the Hartree-Fock state, ``hf_state`` used to initialize the wires.
    :param wires: The qubits to act on.
    :param singles: A sequence of lists with the indices of the two qubits acted on by the single_exitation operation.
    :param doubles: List sequence with the indices of the two qubits acted on by the double_exitation operation.

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

    Implement a circuit that provides an ensemble that can be used to perform accurate single-unit basis rotations. The circuit is derived from the single-particle fermion-determined unitary transformation :math:`U(u)` given in `arXiv:1711.04789 <https://arxiv.org/abs/1711.04789>`_\ 
    
    .. math::
        U(u) = \exp{\left( \sum_{pq} \left[\log u \right]_{pq} (a_p^\dagger a_q - a_q^\dagger a_p) \right)}.

    :math:`U(u)` is obtained by using the scheme given in the paper `Optica, 3, 1460 (2016) <https://opg.optica.org/optica/fulltext.cfm?uri=optica-3-12-1460&id=355743>`_\ .

    :param q_machine: quantum machine.
    :param wires: qubits to act on.
    :param unitary_matrix: matrix specifying the basis for the transformation.
    :param check: check if `unitary_matrix` is a unitary matrix.

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



Distributed interface
================================================

Distributed related functions, when using the ``torch`` computing backend, encapsulate the ``torch.distributed`` interface of torch,

.. note::

    Please refer to `torch distributed <https://pytorch.org/docs/stable/distributed.html>`_ to start the distributed method.
    When using CPU for distribution, please use ``gloo`` instead of ``mpi``.
    When using GPU for distribution, please use ``nccl``.

    :ref:`vqnet_dist` VQNet's own distributed interface is not applicable to the ``torch`` computing backend.

CommController
-------------------------

.. py:class:: pyvqnet.distributed.ControlComm.CommController(backend,rank=None,world_size=None)
    :no-index:
    
    CommController is used to control the data communication controller under cpu and gpu. It generates cpu (gloo) and gpu (nccl) controllers by setting the parameter `backend`.
    This class will call backend, rank, world_size to initialize ``torch.distributed.init_process_group(backend, rank, world_size)`` .

    :param backend: used to generate cpu or gpu data communication controller, 'gloo' or 'nccl'.
    :param rank: the process number of the current program.
    :param world_size: the number of all global processes.

    :return:
        CommController instance.

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

        Used to get the process ID of the current process.

        :return: Returns the process ID of the current process.

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

        Used to get the total number of processes started.

        :return: Returns the total number of processes.

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

        In each process, get the local process number of each machine through ``os.environ['LOCAL_RANK'] = rank``.

        The environment variable `LOCAL_RANK` needs to be set in advance.

        :return: The current process number on the current machine.

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

        The process number list set according to the input parameter is used to divide multiple communication groups.

        :param rankL: process group list.
        :return: a list containing ``torch.distributed.ProcessGroup``

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

        Synchronization of different processes.

        :return: Synchronization operation.

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

        Supports allreduce communication on data.

        :param tensor: Input data.
        :param c_op: Calculation method.

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

        Supports reduce communication on data.

        :param tensor: Input data.
        :param root: Specifies the node where the data is returned.
        :param c_op: Calculation method.

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

        Broadcast the data on the specified process root to all processes.

        :param tensor: Input data.
        :param root: The specified node.

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
        :param c_op: Calculation method.
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
        :param c_op: Calculation method.
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
        
        Allgather communication interface within the group.

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

