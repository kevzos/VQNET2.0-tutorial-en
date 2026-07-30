

.. _torch_api:

=============================================================
VQNet usa torch para computacao de baixo nivel
=============================================================

A partir da versao 2.15.0, este software suporta o uso de `torch` como backend de computacao para operacoes de baixo nivel e pode ser integrado a modelos, codigos e bibliotecas de terceiros baseados em `torch` para desenvolvimento secundario.

    .. important::

        Para usar os recursos a seguir, instale torch>=2.11.0 por conta propria. Se estiver instalando uma versao GPU do torch, use uma versao compativel com CUDA 12.6, caso contrario seu torch pode nao funcionar devido a problemas com a biblioteca de tempo de execucao NVIDIA CUDA. Este software nao instala o torch automaticamente durante a instalacao.

    .. note::

        As funcoes de computacao quantica variacional (com nomes em minusculo, como `rx` , `ry` , `rz` , etc.) em :ref:`vqc_api`, bem como as funcoes de computacao basica do QTensor in :ref:`qtensor_api` ,
        podem receber um `QTensor` como entrada apos chamar ``pyvqnet.backends.set_backend("torch")`` , com o membro `data` do `QTensor` mudando de Tensor do pyvqnet para ``torch.Tensor`` para computacao.

        ``pyvqnet.backends.set_backend("torch")`` e ``pyvqnet.backends.set_backend("pyvqnet")`` modificam o backend de computacao global.
        Objetos ``QTensor`` criados sob diferentes configuracoes de backend nao podem ser misturados em computacoes.

Configuracao Basica do Backend
============================================

set_backend
------------------------------------------------

.. py:function:: pyvqnet.backends.set_backend(backend_name)

    Define o backend para computacoes atuais e armazenamento de dados. O padrao eh "pyvqnet-ad", mas pode ser definido como "torch", "torch-native", "pyvqnet-ad".
    
    Apos chamar ``pyvqnet.backends.set_backend("torch")``, a interface permanece inalterada. A variavel membro ``data`` do ``QTensor`` do VQNet usa ``torch.Tensor`` para armazenar dados.
    :ref:`qtensor_api` , :ref:`vqc_api` , e ``pyvqnet.nn.torch`` aceitam ``QTensor`` como entrada e ``QTensor`` como saida.

    Apos chamar ``pyvqnet.backends.set_backend("torch-native")``, as interfaces permanecem inalteradas: :ref:`qtensor_api`, :ref:`vqc_api`, e a interface `pyvqnet.nn.torch`.
    As entradas podem aceitar diretamente tipos ``torch.Tensor`` ou ``QTensor``, e as saidas sao ``torch.Tensor``, eliminando a necessidade de conversao para ``QTensor``, reduzindo assim a conversao de dados.
    
    Apos chamar ``pyvqnet.backends.set_backend("pyvqnet")``, o membro ``data`` do ``QTensor`` do VQNet armazenara dados usando ``pyvqnet._core.Tensor`` , e as computacoes usarao a biblioteca C++ do pyvqnet.

    Apos chamar ``pyvqnet.backends.set_backend("pyvqnet-ad")``, o membro ``data`` do ``QTensor`` do VQNet armazenara dados usando ``pyvqnet._core.Tensor`` , e as computacoes usarao a biblioteca C++ do pyvqnet com desempenho otimizado.


    .. note::

        Esta funcao modifica o backend de computacao atual. Objetos ``QTensor`` criados sob diferentes backends nao podem ser usados juntos em computacoes.

    :param backend_name: Nome do backend, pode ser "pyvqnet" ou "torch".

    Exemplo::

        import pyvqnet
        pyvqnet.backends.set_backend("torch")

get_backend
-------------------------------

.. py:function:: pyvqnet.backends.get_backend(t=None)

    Se `t` for None, obtem o backend de computacao atual.
    Se `t` for um QTensor, retorna o backend usado para criar o QTensor com base em sua propriedade ``data``.
    Se "torch" for o backend, retorna o backend torchAPI do pyvqnet.
    Se "pyvqnet" for o backend, simplesmente retorna "pyvqnet".
    
    :param t: O tensor atual (padrao: None).
    :return: O backend. Por padrao, retorna "pyvqnet".

    Exemplo::

        import pyvqnet
        pyvqnet.backends.set_backend("torch")
        pyvqnet.backends.get_backend()

Funcoes do QTensor
===================

Apos definir o backend como ``torch``:

.. code-block::

    import pyvqnet
    pyvqnet.backends.set_backend("torch")

Todas as funcoes membro, funcoes de criacao, funcoes matematicas, funcoes logicas, transformacoes de matriz, etc., em :ref:`qtensor_api` usarao torch para computacao. O `QTensor.data` pode ser acessado para recuperar os dados torch.

Modulos de Rede Neural Classica e Rede Neural Quantica Variacional
==========================================================================================

Classe Base
------------------------------------------------

TorchModule
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. py:class:: pyvqnet.nn.torch.TorchModule(*args, **kwargs)

    A classe base que define modelos ao usar o backend `torch`. Esta classe herda de ``pyvqnet.nn.Module`` e ``torch.nn.Module``.
    Pode ser adicionada como um submodulo a um torchmodel.

    .. note::

        Esta classe e suas classes derivadas sao adequadas apenas para uso com ``pyvqnet.backends.set_backend("torch")``.
        Nao misture com o `Module` padrao do ``pyvqnet.nn``.
    
        Os dados em ``_buffers`` desta classe sao do tipo ``torch.Tensor``.
        Os dados em ``_parameters`` desta classe sao do tipo ``torch.nn.Parameter``.

    .. py:method:: pyvqnet.nn.torch.TorchModule.forward(x, *args, **kwargs)

        Funcao abstrata de computacao direta para a classe TorchModule.

        :param x: QTensor de entrada.
        :param args: Argumentos variaveis nao nomeados.
        :param kwargs: Argumentos variaveis nomeados.

        :return: QTensor de saida, com o `data` interno sendo um ``torch.Tensor``.

        Exemplo::

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

        Retorna um dicionario contendo o estado completo do modulo, incluindo parametros e valores de buffers.
        As chaves sao os nomes dos parametros e buffers correspondentes.

        :param destination: O dicionario para armazenar os parametros internos do modulo.
        :param prefix: Um prefixo usado para os nomes dos parametros e buffers.

        :return: Um dicionario contendo o estado completo do modulo.

        Exemplo::

            from pyvqnet.nn.torch import Conv2D
            import pyvqnet
            pyvqnet.backends.set_backend("torch")
            test_conv = Conv2D(2,3,(3,3),(2,2),"same")
            print(test_conv.state_dict().keys())

    .. py:method:: pyvqnet.nn.torch.TorchModule.load_state_dict(state_dict, strict=True)

        Copia parametros e buffers do :attr:`state_dict` para este modulo e seus submodulos.

        :param state_dict: Um dicionario contendo parametros e buffers persistentes.
        :param strict: Se deve exigir que as chaves no state_dict correspondam ao `state_dict()` do modelo. Padrao: True.

        :return: Uma mensagem de erro se houver algum problema.

        Exemplos::

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

        Move o modulo e seus submodulos, parametros e dados de buffer para o dispositivo GPU especificado.

        O dispositivo especifica onde os dados internos sao armazenados. Quando device >= DEV_GPU_0, os dados sao armazenados na GPU.
        Se seu computador tiver varias GPUs, voce pode especificar dispositivos diferentes para armazenar dados. Por exemplo, device = DEV_GPU_1, DEV_GPU_2, DEV_GPU_3, ... refere-se a armazenamento em GPUs com diferentes numeros de serie.
        
        .. note::

            Modulos nao podem realizar computacoes entre diferentes GPUs.
            Se voce tentar criar um QTensor em um ID de GPU que exceda o maximo permitido para validacao, um erro Cuda sera gerado.

        :param device: O dispositivo para armazenar o QTensor. Padrao: DEV_GPU_0. device = pyvqnet.DEV_GPU_0 armazena na primeira GPU, device = DEV_GPU_1 armazena na segunda GPU, e assim por diante.
        :return: O Modulo movido para o dispositivo GPU.

        Exemplos::

            from pyvqnet.nn.torch import ConvT2D
            import pyvqnet
            pyvqnet.backends.set_backend("torch")
            test_conv = ConvT2D(3, 2, [4, 4], [2, 2], (0, 0))
            test_conv = test_conv.toGPU()
            print(test_conv.backend)
            #1000

    .. py:method:: pyvqnet.torch.TorchModule.toCPU()

        Move o modulo e seus submodulos, parametros e dados de buffer para um dispositivo CPU especifico.

        :return: O Modulo movido para o dispositivo CPU.

        Exemplos::

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

    Este modulo eh usado para armazenar instancias filhas ``TorchModule`` em uma lista. O TorchModuleList pode ser indexado como uma lista Python comum, e os parametros internos que ele contem podem ser salvos.
    
    Esta classe herda de ``pyvqnet.nn.torch.TorchModule`` e ``pyvqnet.nn.ModuleList``, e pode ser adicionada como um submodulo a um torchmodel.

    :param modules: Uma lista de ``pyvqnet.nn.torch.TorchModule``

    :return: Uma classe TorchModuleList

    Exemplo::

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

    Este modulo eh usado para armazenar instancias filhas ``pyvqnet.nn.Parameter`` em uma lista. O TorchParameterList pode ser indexado como uma lista Python comum, e os parametros internos que ele contem podem ser salvos.
    
    Esta classe herda de ``pyvqnet.nn.torch.TorchModule`` e ``pyvqnet.nn.ParameterList``, e pode ser adicionada como um submodulo a um torchmodel.

    :param value: Uma lista de ``nn.Parameter``

    :return: Uma classe TorchParameterList

    Exemplo::

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

    O modulo adiciona modulos na ordem em que sao passados. Alternativamente, voce pode passar um ``OrderedDict`` de modulos. O metodo ``forward()`` da classe ``Sequential`` aceita qualquer entrada e a encaminha para seu primeiro modulo.
    A saida entao eh sequencialmente ligada a entrada de cada modulo subsequente, com a saida final sendo o resultado do ultimo modulo.

    Esta classe herda de ``pyvqnet.nn.torch.TorchModule`` e ``pyvqnet.nn.Sequential``, e pode ser adicionada como um submodulo a um torchmodel.

    :param args: Modulos a serem adicionados

    :return: Uma classe TorchSequential

    Exemplo::

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

Salvando e Carregando Parametros do Modelo
--------------------------------------------

Voce pode usar :ref:`save_parameters` ``save_parameters`` e ``load_parameters`` para salvar os parametros de um modelo ``TorchModule`` como um dicionario em um arquivo, com os valores salvos como `numpy.ndarray`. Alternativamente, voce pode carregar o arquivo de parametros do disco. Note que a estrutura do modelo nao eh salva no arquivo, e voce precisara reconstruir manualmente a estrutura do modelo. Voce tambem pode usar diretamente ``torch.save`` e ``torch.load`` para ler os parametros do modelo ``torch``, ja que os parametros do ``TorchModule`` sao armazenados como objetos ``torch.Tensor``.


Modulos de Rede Neural Classica
--------------------------------------------

Os seguintes modulos de rede neural classica herdam todos de ``pyvqnet.nn.Module`` e ``torch.nn.Module``, e podem ser adicionados como submodulos a um torchmodel.

Linear
^^^^^^^^^^^^^^^^^^^^^^^^

.. py:class:: pyvqnet.nn.torch.Linear(input_channels, output_channels, weight_initializer=None, bias_initializer=None, use_bias=True, dtype=None, name: str = "")

    Um modulo linear (camada totalmente conectada), :math:`y = x@A.T + b`.
    Esta classe herda de ``pyvqnet.nn.Module`` e ``torch.nn.Module`` e pode ser usada como um submodulo de um torchmodel.

    Os dados em ``_buffers`` da classe sao do tipo ``torch.Tensor``.
    Os dados em ``_parameters`` da classe sao do tipo ``torch.nn.Parameter``.

    :param input_channels: `int` - O numero de canais de entrada.
    :param output_channels: `int` - O numero de canais de saida.
    :param weight_initializer: `callable` - Funcao de inicializacao de peso, padrao vazio, usando he_uniform.
    :param bias_initializer: `callable` - Funcao de inicializacao de vies, padrao vazio, usando he_uniform.
    :param use_bias: `bool` - Se deve usar o termo de vies, padrao True.
    :param dtype: Tipo de dados para os parametros, padrao None, usa o tipo de dados padrao `kfloat32`, que representa numeros de ponto flutuante de 32 bits.
    :param name: O nome da camada linear, padrao "".

    :return: Uma instancia da camada Linear.

    Exemplo::

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

    Realiza convolucao 1D na entrada. A entrada do modulo Conv1D tem a forma (batch_size, input_channels, in_height).
    Esta classe herda de ``pyvqnet.nn.Module`` e ``torch.nn.Module``, e pode ser usada como um submodulo de um torchmodel.

    Os dados em ``_buffers`` da classe sao do tipo ``torch.Tensor``.
    Os dados em ``_parameters`` da classe sao do tipo ``torch.nn.Parameter``.

    :param input_channels: `int` - O numero de canais de entrada.
    :param output_channels: `int` - O numero de canais de saida.
    :param kernel_size: `int` - O tamanho do kernel de convolucao. A forma do kernel eh [output_channels, input_channels/group, kernel_size, 1].
    :param stride: `int` - O passo, padrao eh 1.
    :param padding: `str|int` - Opcoes de preenchimento, pode ser uma string {'valid', 'same'} ou um inteiro especificando a quantidade de preenchimento a ser aplicada a entrada. Padrao eh "valid".
    :param use_bias: `bool` - Se deve usar o termo de vies, padrao True.
    :param kernel_initializer: `callable` - O metodo de inicializacao do kernel de convolucao. Padrao vazio, usando kaiming_uniform.
    :param bias_initializer: `callable` - O metodo de inicializacao do vies. Padrao vazio, usando kaiming_uniform.
    :param dilation_rate: `int` - O tamanho da dilatacao, padrao eh 1.
    :param group: `int` - O numero de grupos na convolucao agrupada. Padrao eh 1.
    :param dtype: Tipo de dados para os parametros, padrao None, usa o tipo de dados padrao `kfloat32`, que representa numeros de ponto flutuante de 32 bits.
    :param name: O nome do modulo, padrao "".

    :return: Uma instancia da convolucao 1D.

    .. note::

        ``padding='valid'`` nao aplica preenchimento.

        ``padding='same'`` aplica preenchimento zero a entrada, com o `out_height` da saida igual a `ceil(in_height / stride)`, e nao suporta `stride > 1`.

    Exemplo::

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

    Realiza convolucao 2D na entrada. A entrada do modulo Conv2D tem a forma (batch_size, input_channels, height, width).
    Esta classe herda de ``pyvqnet.nn.Module`` e ``torch.nn.Module``, e pode ser usada como um submodulo de um torchmodel.

    Os dados em ``_buffers`` da classe sao do tipo ``torch.Tensor``.
    Os dados em ``_parameters`` da classe sao do tipo ``torch.nn.Parameter``.

    :param input_channels: `int` - O numero de canais de entrada.
    :param output_channels: `int` - O numero de canais de saida.
    :param kernel_size: `tuple|list` - O tamanho do kernel de convolucao. A forma do kernel eh [output_channels, input_channels/group, kernel_size, kernel_size].
    :param stride: `tuple|list` - O passo, padrao eh (1, 1).
    :param padding: `str|tuple` - Opcoes de preenchimento, pode ser uma string {'valid', 'same'} ou uma tupla especificando o preenchimento a ser aplicado a ambos os lados. Padrao eh "valid".
    :param use_bias: `bool` - Se deve usar o termo de vies, padrao True.
    :param kernel_initializer: `callable` - O metodo de inicializacao do kernel de convolucao. Padrao vazio, usando kaiming_uniform.
    :param bias_initializer: `callable` - O metodo de inicializacao do vies. Padrao vazio, usando kaiming_uniform.
    :param dilation_rate: `int` - O tamanho da dilatacao, padrao eh 1.
    :param group: `int` - O numero de grupos na convolucao agrupada. Padrao eh 1.
    :param dtype: Tipo de dados para os parametros, padrao None, usa o tipo de dados padrao `kfloat32`, que representa numeros de ponto flutuante de 32 bits.
    :param name: O nome do modulo, padrao "".

    :return: Uma instancia da convolucao 2D.

    .. note::

        ``padding='valid'`` nao aplica preenchimento.

        ``padding='same'`` aplica preenchimento zero a entrada, com a altura da saida igual a `ceil(in_height / stride)`.

    Exemplo::

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

    Realiza convolucao transposta 2D na entrada. A entrada do modulo ConvT2D tem a forma (batch_size, input_channels, height, width).
    Esta classe herda de ``pyvqnet.nn.Module`` e ``torch.nn.Module``, e pode ser usada como um submodulo de um torchmodel.

    Os dados em ``_buffers`` da classe sao do tipo ``torch.Tensor``.
    Os dados em ``_parameters`` da classe sao do tipo ``torch.nn.Parameter``.

    :param input_channels: `int` - O numero de canais de entrada.
    :param output_channels: `int` - O numero de canais de saida.
    :param kernel_size: `tuple|list` - O tamanho do kernel de convolucao, com forma do kernel = [input_channels, output_channels/group, kernel_size, kernel_size].
    :param stride: `tuple|list` - O passo, padrao eh (1, 1).
    :param padding: `tuple` - Opcoes de preenchimento, uma tupla especificando o preenchimento a ser aplicado a ambos os lados. Padrao eh (0, 0).
    :param use_bias: `bool` - Se deve usar o termo de vies, padrao True.
    :param kernel_initializer: `callable` - O metodo de inicializacao do kernel de convolucao. Padrao vazio, usando kaiming_uniform.
    :param bias_initializer: `callable` - O metodo de inicializacao do vies. Padrao vazio, usando kaiming_uniform.
    :param dilation_rate: `int` - O tamanho da dilatacao, padrao eh 1.
    :param out_padding: Tamanho extra adicionado a forma da saida para cada dimensao. Padrao eh (0, 0).
    :param group: `int` - O numero de grupos na convolucao agrupada. Padrao eh 1.
    :param dtype: Tipo de dados para os parametros, padrao None, usa o tipo de dados padrao `kfloat32`, que representa numeros de ponto flutuante de 32 bits.
    :param name: O nome do modulo, padrao "".

    :return: Uma instancia da convolucao transposta 2D.

    .. note::

        ``padding='valid'`` nao aplica preenchimento.

        ``padding='same'`` aplica preenchimento zero a entrada, com a altura da saida igual a `ceil(height / stride)`.

    Exemplo::

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

    Realiza pooling medio na entrada 1D. A entrada tem a forma (batch_size, input_channels, in_height).
    Esta classe herda de ``pyvqnet.nn.Module`` e ``torch.nn.Module``, e pode ser adicionada como um submodulo a um torchmodel.

    Os dados em ``_buffers`` da classe sao do tipo ``torch.Tensor``.
    Os dados em ``_parameters`` da classe sao do tipo ``torch.nn.Parameter``.

    :param kernel: O tamanho da janela de pooling.
    :param stride: O tamanho do passo para mover a janela.
    :param padding: Opcao de preenchimento, um inteiro especificando o comprimento do preenchimento. Padrao eh 0.
    :param name: O nome do modulo, padrao "".

    :return: Uma instancia da camada de pooling medio 1D.

    Exemplo::

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

    Realiza pooling maximo na entrada 1D. A entrada tem a forma (batch_size, input_channels, in_height).
    Esta classe herda de ``pyvqnet.nn.Module`` e ``torch.nn.Module``, e pode ser adicionada como um submodulo a um torchmodel.

    Os dados em ``_buffers`` da classe sao do tipo ``torch.Tensor``.
    Os dados em ``_parameters`` da classe sao do tipo ``torch.nn.Parameter``.

    :param kernel: O tamanho da janela de pooling.
    :param stride: O tamanho do passo para mover a janela.
    :param padding: Opcao de preenchimento, um inteiro especificando o comprimento do preenchimento. Padrao eh 0.
    :param name: O nome do modulo, padrao "".

    :return: Uma instancia da camada de pooling maximo 1D.

    Exemplo::

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

    Realiza pooling medio na entrada 2D. A entrada tem a forma (batch_size, input_channels, height, width).
    Esta classe herda de ``pyvqnet.nn.Module`` e ``torch.nn.Module``, e pode ser adicionada como um submodulo a um torchmodel.

    Os dados em ``_buffers`` da classe sao do tipo ``torch.Tensor``.
    Os dados em ``_parameters`` da classe sao do tipo ``torch.nn.Parameter``.

    :param kernel: O tamanho da janela de pooling.
    :param stride: O tamanho do passo para mover a janela.
    :param padding: Opcao de preenchimento, uma tupla contendo dois inteiros especificando preenchimento para ambas as dimensoes. Padrao eh (0,0).
    :param name: O nome do modulo, padrao "".

    :return: Uma instancia da camada de pooling medio 2D.

    Exemplo::

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

    Realiza pooling maximo na entrada 2D. A entrada tem a forma (batch_size, input_channels, height, width).
    Esta classe herda de ``pyvqnet.nn.Module`` e ``torch.nn.Module``, e pode ser adicionada como um submodulo a um torchmodel.

    Os dados em ``_buffers`` da classe sao do tipo ``torch.Tensor``.
    Os dados em ``_parameters`` da classe sao do tipo ``torch.nn.Parameter``.

    :param kernel: O tamanho da janela de pooling.
    :param stride: O tamanho do passo para mover a janela.
    :param padding: Opcao de preenchimento, uma tupla contendo dois inteiros especificando preenchimento para ambas as dimensoes. Padrao eh (0,0).
    :param name: O nome do modulo, padrao "".

    :return: Uma instancia da camada de pooling maximo 2D.

    Exemplo::

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

    Este modulo eh tipicamente usado para armazenar embeddings de palavras e recupera-las usando indices. A entrada do modulo eh uma lista de indices, e a saida sao os embeddings de palavras correspondentes.
    A entrada desta camada deve ser do tipo `kint64`.
    Esta classe herda de ``pyvqnet.nn.Module`` e ``torch.nn.Module``, e pode ser adicionada como um submodulo a um torchmodel.

    Os dados em ``_buffers`` da classe sao do tipo ``torch.Tensor``.
    Os dados em ``_parameters`` da classe sao do tipo ``torch.nn.Parameter``.

    :param num_embeddings: `int` - O tamanho do dicionario de embedding.
    :param embedding_dim: `int` - O tamanho de cada vetor de embedding.
    :param weight_initializer: `callable` - O metodo de inicializacao do peso, padrao eh Xavier Normal.
    :param dtype: O tipo de dados para os parametros, padrao None, que usa o tipo de dados padrao: `kfloat32` (ponto flutuante de 32 bits).
    :param name: O nome da camada de embedding, padrao "".

    :return: Uma instancia da camada Embedding.

    Exemplo::

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

    Aplica normalizacao de lote em entrada 4D (B, C, H, W). Consulte o artigo
    `Batch Normalization: Accelerating Deep Network Training by Reducing
    Internal Covariate Shift <https://arxiv.org/abs/1502.03167>`__.

    Esta classe herda de ``pyvqnet.nn.Module`` e ``torch.nn.Module``, e pode ser adicionada como um submodulo a um torchmodel.

    Os dados em ``_buffers`` da classe sao do tipo ``torch.Tensor``.
    Os dados em ``_parameters`` da classe sao do tipo ``torch.nn.Parameter``.

    .. math::
        y = \frac{x - \mathrm{E}[x]}{\sqrt{\mathrm{Var}[x] + \epsilon}} * \gamma + \beta

    onde :math:`\gamma` e :math:`\beta` sao parametros treinaveis. Alem disso, por padrao, durante o treinamento, a camada continua estimando a media e a variancia, que sao entao usadas para normalizacao durante a avaliacao. O momentum para as medias moveis eh definido como o valor padrao de 0.1.

    :param channel_num: `int` - O numero de canais de entrada.
    :param momentum: `float` - Momentum para o calculo da media movel, padrao eh 0.1.
    :param epsilon: `float` - Uma pequena constante para estabilidade numerica, padrao eh 1e-5.
    :param affine: `bool` - Se deve incluir parametros afins treinaveis para cada canal. Padrao eh `True`, que inicializa os parametros como 1 para pesos e 0 para vieses.
    :param beta_initializer: `callable` - O metodo de inicializacao para beta, padrao eh inicializacao zero.
    :param gamma_initializer: `callable` - O metodo de inicializacao para gamma, padrao eh inicializacao um.
    :param dtype: O tipo de dados para os parametros, padrao None, usando `kfloat32` (ponto flutuante de 32 bits).
    :param name: O nome da camada de normalizacao de lote, padrao "".

    :return: Uma instancia da camada de normalizacao de lote 2D.

    Exemplo::

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

    Aplica normalizacao de lote em entrada 2D (B, C). Consulte o artigo
    `Batch Normalization: Accelerating Deep Network Training by Reducing
    Internal Covariate Shift <https://arxiv.org/abs/1502.03167>`__.

    .. math::
        y = \frac{x - \mathrm{E}[x]}{\sqrt{\mathrm{Var}[x] + \epsilon}} * \gamma + \beta

    onde :math:`\gamma` e :math:`\beta` sao parametros treinaveis. Alem disso, por padrao, durante o treinamento, a camada continua estimando a media e a variancia, que sao entao usadas para normalizacao durante a avaliacao. O momentum para as medias moveis eh definido como o valor padrao de 0.1.

    Esta classe herda de ``pyvqnet.nn.Module`` e ``torch.nn.Module``, e pode ser adicionada como um submodulo a um torchmodel.

    Os dados em ``_buffers`` da classe sao do tipo ``torch.Tensor``.
    Os dados em ``_parameters`` da classe sao do tipo ``torch.nn.Parameter``.

    :param channel_num: `int` - O numero de canais de entrada.
    :param momentum: `float` - Momentum para o calculo da media movel, padrao eh 0.1.
    :param epsilon: `float` - Uma pequena constante para estabilidade numerica, padrao eh 1e-5.
    :param affine: `bool` - Se deve incluir parametros afins treinaveis para cada canal. Padrao eh `True`, que inicializa os parametros como 1 para pesos e 0 para vieses.
    :param beta_initializer: `callable` - O metodo de inicializacao para beta, padrao eh inicializacao zero.
    :param gamma_initializer: `callable` - O metodo de inicializacao para gamma, padrao eh inicializacao um.
    :param dtype: O tipo de dados para os parametros, padrao None, usando `kfloat32` (ponto flutuante de 32 bits).
    :param name: O nome da camada de normalizacao de lote, padrao "".

    :return: Uma instancia da camada de normalizacao de lote 1D.

    Exemplo::

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

    Aplica normalizacao de camada nas ultimas D dimensoes de qualquer entrada. O metodo especifico eh descrito no artigo:
    `Layer Normalization <https://arxiv.org/abs/1607.06450>`__.

    .. math::
        y = \frac{x - \mathrm{E}[x]}{ \sqrt{\mathrm{Var}[x] + \epsilon}} * \gamma + \beta

    Para entradas como (B, C, H, W, D), o ``norm_shape`` pode ser [C, H, W, D], [H, W, D], [W, D], ou [D].

    Esta classe herda de ``pyvqnet.nn.Module`` e ``torch.nn.Module``, e pode ser adicionada como um submodulo a um torchmodel.

    Os dados em ``_buffers`` da classe sao do tipo ``torch.Tensor``.
    Os dados em ``_parameters`` da classe sao do tipo ``torch.nn.Parameter``.

    :param norm_shape: `list` - A forma a ser normalizada.
    :param epsilon: `float` - Uma pequena constante para estabilidade numerica, padrao eh 1e-5.
    :param affine: `bool` - Se `True`, este modulo tem parametros afins treinaveis para cada canal, inicializados como 1 (para pesos) e 0 (para vieses). Padrao eh `True`.
    :param dtype: O tipo de dados para os parametros, padrao None, usando `kfloat32` (ponto flutuante de 32 bits).
    :param name: O nome do modulo, padrao "".

    :return: Uma instancia da classe LayerNormNd.

    Exemplo::

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

    Aplica normalizacao de camada em entradas 4D. O metodo especifico eh descrito no artigo:
    `Layer Normalization <https://arxiv.org/abs/1607.06450>`__.

    .. math::
        y = \frac{x - \mathrm{E}[x]}{ \sqrt{\mathrm{Var}[x] + \epsilon}} * \gamma + \beta

    A media e o desvio padrao sao calculados nas dimensoes restantes, excluindo a primeira. Para entradas como (B, C, H, W), ``norm_size`` deve ser igual a C * H * W.

    Esta classe herda de ``pyvqnet.nn.Module`` e ``torch.nn.Module``, e pode ser adicionada como um submodulo a um torchmodel.

    Os dados em ``_buffers`` da classe sao do tipo ``torch.Tensor``.
    Os dados em ``_parameters`` da classe sao do tipo ``torch.nn.Parameter``.

    :param norm_size: `int` - O tamanho da normalizacao, deve ser igual a C * H * W.
    :param epsilon: `float` - Uma pequena constante para estabilidade numerica, padrao eh 1e-5.
    :param affine: `bool` - Se `True`, este modulo tem parametros afins treinaveis para cada canal, inicializados como 1 (para pesos) e 0 (para vieses). Padrao eh `True`.
    :param dtype: O tipo de dados para os parametros, padrao None, usando `kfloat32` (ponto flutuante de 32 bits).
    :param name: O nome do modulo, padrao "".

    :return: Uma instancia da normalizacao de camada 2D.

    Exemplo::

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

    Aplica normalizacao de camada em entradas 2D. O metodo especifico eh descrito no artigo:
    `Layer Normalization <https://arxiv.org/abs/1607.06450>`__.

    .. math::
        y = \frac{x - \mathrm{E}[x]}{ \sqrt{\mathrm{Var}[x] + \epsilon}} * \gamma + \beta

    A media e o desvio padrao sao calculados no tamanho da ultima dimensao, onde ``norm_size`` eh o valor da ultima dimensao.

    Esta classe herda de ``pyvqnet.nn.Module`` e ``torch.nn.Module``, e pode ser adicionada como um submodulo a um torchmodel.

    Os dados em ``_buffers`` da classe sao do tipo ``torch.Tensor``.
    Os dados em ``_parameters`` da classe sao do tipo ``torch.nn.Parameter``.

    :param norm_size: `int` - O tamanho da normalizacao, deve ser igual ao tamanho da ultima dimensao.
    :param epsilon: `float` - Uma pequena constante para estabilidade numerica, padrao eh 1e-5.
    :param affine: `bool` - Se `True`, este modulo tem parametros afins treinaveis para cada canal, inicializados como 1 (para pesos) e 0 (para vieses). Padrao eh `True`.
    :param dtype: O tipo de dados para os parametros, padrao None, usando `kfloat32` (ponto flutuante de 32 bits).
    :param name: O nome do modulo, padrao "".

    :return: Uma instancia da normalizacao de camada 1D.

    Exemplo::

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

    Aplica normalizacao de grupo em entradas de mini-lote. Entrada: :math:`(N, C, *)` onde :math:`C=\mathrm{num\_channels}`, Saida: :math:`(N, C, *)`.

    Esta camada implementa a operacao descrita no artigo `Group Normalization <https://arxiv.org/abs/1803.08494>`__.

    .. math::
        
        y = \frac{x - \mathrm{E}[x]}{ \sqrt{\mathrm{Var}[x] + \epsilon}} * \gamma + \beta

    Os canais de entrada sao divididos em :attr:`num_groups` grupos, cada um contendo ``num_channels / num_groups`` canais. O :attr:`num_channels` deve ser divisivel por :attr:`num_groups`. A media e o desvio padrao sao calculados separadamente dentro de cada grupo. Se :attr:`affine` for ``True``, entao :math:`\gamma` e :math:`\beta` sao treinaveis. Os parametros de transformacao afim para cada canal sao vetores de tamanho :attr:`num_channels`.

    Esta classe herda de ``pyvqnet.nn.Module`` e ``torch.nn.Module``, e pode ser adicionada como um submodulo a um torchmodel.

    Os dados em ``_buffers`` da classe sao do tipo ``torch.Tensor``.
    Os dados em ``_parameters`` da classe sao do tipo ``torch.nn.Parameter``.

    :param num_groups (int): The number of groups to divide the channels into.
    :param num_channels (int): The number of expected input channels.
    :param epsilon: Um pequeno valor adicionado ao denominador para estabilidade numerica. Padrao eh 1e-5.
    :param affine: Um valor booleano. Se definido como ``True``, este modulo tem parametros afins treinaveis para cada canal, inicializados como 1 (para pesos) e 0 (para vieses). Padrao eh ``True``.
    :param dtype: O tipo de dados para os parametros. Padrao None, usando `kfloat32` (ponto flutuante de 32 bits).
    :param name: O nome do modulo. Padrao "".

    :return: Uma instancia da classe GroupNorm.

    Exemplo::

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

    Modulo Dropout. O modulo dropout define aleatoriamente a saida de algumas unidades como zero, enquanto dimensiona as unidades restantes pela probabilidade dada de dropout_rate.

    Esta classe herda de ``pyvqnet.nn.Module`` e ``torch.nn.Module``, e pode ser adicionada como um submodulo a um torchmodel.

    :param dropout_rate: `float` - A probabilidade de definir neuronios como zero.
    :param name: O nome do modulo. Padrao "".

    :return: Uma instancia da classe Dropout.

    Exemplo::

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

    O modulo DropPath aplica dropout de caminho aleatorio (profundidade aleatoria).

    Esta classe herda de ``pyvqnet.nn.Module`` e ``torch.nn.Module``, e pode ser adicionada como um submodulo a um torchmodel.

    :param dropout_rate: `float` - A probabilidade de definir neuronios como zero.
    :param name: O nome do modulo. Padrao "".

    :return: Uma instancia da classe DropPath.

    Exemplo::

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

    Reorganiza um tensor de forma: (*, C * r^2, H, W) para um tensor de forma (*, C, H * r, W * r), onde r eh o fator de escala.

    Esta classe herda de ``pyvqnet.nn.Module`` e ``torch.nn.Module``, e pode ser adicionada como um submodulo a um torchmodel.

    :param upscale_factors: O fator de escala para a transformacao.
    :param name: O nome do modulo. Padrao "".

    :return: Uma instancia do modulo Pixel_Shuffle.

    Exemplo::

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

    Inverte a operacao Pixel_Shuffle reorganizando elementos. Transforma um tensor de forma (*, C, H * r, W * r) para (*, C * r^2, H, W), onde r eh o fator de reducao de escala.

    Esta classe herda de ``pyvqnet.nn.Module`` e ``torch.nn.Module``, e pode ser adicionada como um submodulo a um torchmodel.

    :param downscale_factors: O fator de reducao de escala para a transformacao.
    :param name: O nome do modulo. Padrao "".

    :return: Uma instancia do modulo Pixel_Unshuffle.

    Exemplo::

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

    Modulo Gated Recurrent Unit (GRU). Suporta empilhamento de multiplas camadas e configuracao bidirecional. A formula para um GRU unidirecional de camada unica eh:

    .. math::
        \begin{array}{ll}
            r_t = \sigma(W_{ir} x_t + b_{ir} + W_{hr} h_{(t-1)} + b_{hr}) \\
            z_t = \sigma(W_{iz} x_t + b_{iz} + W_{hz} h_{(t-1)} + b_{hz}) \\
            n_t = \tanh(W_{in} x_t + b_{in} + r_t * (W_{hn} h_{(t-1)}+ b_{hn})) \\
            h_t = (1 - z_t) * n_t + z_t * h_{(t-1)}
        \end{array}

    Esta classe herda de ``pyvqnet.nn.Module`` e ``torch.nn.Module``, e pode ser adicionada como um submodulo a um torchmodel.

    Os ``_buffers`` da classe contem dados ``torch.Tensor``, e os ``_parameters`` da classe contem dados ``torch.nn.Parameter``.

    :param input_size: A dimensao da caracteristica de entrada.
    :param hidden_size: A dimensao da caracteristica oculta.
    :param num_layers: O numero de camadas GRU empilhadas, padrao: 1.
    :param batch_first: Se True, a forma de entrada eh [batch_size, seq_len, feature_dim]; se False, a forma eh [seq_len, batch_size, feature_dim]; padrao: True.
    :param use_bias: Se False, o modulo nao usa termos de vies; padrao: True.
    :param bidirectional: Se True, torna o GRU bidirecional; padrao: False.
    :param dtype: O tipo de dados dos parametros, padrao None, usando o tipo de dados padrao: kfloat32 (float de 32 bits).
    :param name: O nome do modulo, padrao: "".

    :return: Uma instancia do modulo GRU.

    Exemplo::

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

    Modulo de Rede Neural Recorrente (RNN), usando :math:`\tanh` ou :math:`\text{ReLU}` como funcao de ativacao. Suporta configuracao bidirecional e multiplas camadas. A formula para um RNN unidirecional de camada unica eh:

    .. math::
        h_t = \tanh(W_{ih} x_t + b_{ih} + W_{hh} h_{(t-1)} + b_{hh})

    Se :attr:`nonlinearity` for ``'relu'``, entao :math:`\text{ReLU}` substituira :math:`\tanh`.

    Esta classe herda de ``pyvqnet.nn.Module`` e ``torch.nn.Module``, e pode ser adicionada como um submodulo a um torchmodel.

    Os ``_buffers`` da classe contem dados ``torch.Tensor``, e os ``_parameters`` da classe contem dados ``torch.nn.Parameter``.

    :param input_size: A dimensao da caracteristica de entrada.
    :param hidden_size: A dimensao da caracteristica oculta.
    :param num_layers: O numero de camadas RNN empilhadas, padrao: 1.
    :param nonlinearity: A funcao de ativacao nao linear, padrao: ``'tanh'``.
    :param batch_first: Se True, a forma de entrada eh [batch_size, seq_len, feature_dim]; se False, a forma eh [seq_len, batch_size, feature_dim]; padrao: True.
    :param use_bias: Se False, o modulo nao usa termos de vies; padrao: True.
    :param bidirectional: Se True, torna o RNN bidirecional; padrao: False.
    :param dtype: O tipo de dados dos parametros, padrao None, usando o tipo de dados padrao: kfloat32 (float de 32 bits).
    :param name: O nome do modulo, padrao: "".

    :return: Uma instancia do modulo RNN.

    Exemplo::

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

    Modulo Long Short-Term Memory (LSTM). Suporta LSTM bidirecional e configuracoes LSTM de multiplas camadas empilhadas. A formula para um LSTM unidirecional de camada unica eh a seguinte:

    .. math::
        \begin{array}{ll} \\
            i_t = \sigma(W_{ii} x_t + b_{ii} + W_{hi} h_{t-1} + b_{hi}) \\
            f_t = \sigma(W_{if} x_t + b_{if} + W_{hf} h_{t-1} + b_{hf}) \\
            g_t = \tanh(W_{ig} x_t + b_{ig} + W_{hg} h_{t-1} + b_{hg}) \\
            o_t = \sigma(W_{io} x_t + b_{io} + W_{ho} h_{t-1} + b_{ho}) \\
            c_t = f_t \odot c_{t-1} + i_t \odot g_t \\
            h_t = o_t \odot \tanh(c_t) \\
        \end{array}

    Esta classe herda de ``pyvqnet.nn.Module`` e ``torch.nn.Module``, e pode ser adicionada como um submodulo a um torchmodel.

    Os ``_buffers`` da classe contem dados ``torch.Tensor``, e os ``_parameters`` da classe contem dados ``torch.nn.Parameter``.

    :param input_size: A dimensao da caracteristica de entrada.
    :param hidden_size: A dimensao da caracteristica oculta.
    :param num_layers: O numero de camadas LSTM empilhadas, padrao: 1.
    :param batch_first: Se True, a forma de entrada eh [batch_size, seq_len, feature_dim]; se False, a forma eh [seq_len, batch_size, feature_dim]; padrao: True.
    :param use_bias: Se False, o modulo nao usa termos de vies; padrao: True.
    :param bidirectional: Se True, torna o LSTM bidirecional; padrao: False.
    :param dtype: O tipo de dados dos parametros, padrao None, usando o tipo de dados padrao: kfloat32 (float de 32 bits).
    :param name: O nome do modulo, padrao: "".

    :return: Uma instancia do modulo LSTM.

    Exemplo::

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

    Aplica um GRU RNN multicamadas a sequencias de entrada de comprimento dinamicamente variavel.

    A primeira entrada deve ser uma sequencia em lote com comprimento variavel definida
    atraves de uma classe ``tensor.PackedSequence``.

    A classe ``tensor.PackedSequence`` pode ser construida
    chamando as seguintes funcoes consecutivamente: ``pad_sequence``, ``pack_pad_sequence``.

    A primeira saida do Dynamic_GRU tambem eh uma classe ``tensor.PackedSequence``,
    que pode ser desempacotada para um QTensor normal usando ``tensor.pad_pack_sequence``.

    Para cada elemento na sequencia de entrada, cada camada calcula a seguinte formula:

    .. math::
        \begin{array}{ll}
        r_t = \sigma(W_{ir} x_t + b_{ir} + W_{hr} h_{(t-1)} + b_{hr}) \\
        z_t = \sigma(W_{iz} x_t + b_{iz} + W_{hz} h_{(t-1)} + b_{hz}) \\
        n_t = \tanh(W_{in} x_t + b_{in} + r_t * (W_{hn} h_{(t-1)}+ b_{hn})) \\
        h_t = (1 - z_t) * n_t + z_t * h_{(t-1)}
        \end{array}

    Esta classe herda de ``pyvqnet.nn.Module`` e ``torch.nn.Module``, pode ser adicionada ao modelo torch como um submodulo de ``torch.nn.Module``.

    Os dados em ``_buffers`` desta classe sao do tipo ``torch.Tensor``.

    Os dados em ``_parameters`` desta classe sao do tipo ``torch.nn.Parameter``.

    :param input_size: Input feature dimension.
    :param hidden_size: Hidden feature dimension.
    :param num_layers: Number of loop layers. Default value: 1
    :param batch_first: If True, the input shape is provided as [batch size, sequence length, feature dimension]. If False, the input shape is provided as [sequence length, batch size, feature dimension]. Default value: True.
    :param use_bias: Se False, os pesos de vies b_ih e b_hh nao sao usados para esta camada. Valor padrao: True.
    :param bidirectional: Se verdadeiro, torna-se um GRU bidirecional. Valor padrao: False.
    :param dtype: O tipo de dados do parametro, padrao: None, usa o tipo de dados padrao: kfloat32, representando numeros de ponto flutuante de 32 bits.
    :param name: O nome deste modulo, padrao "".

    :return: Uma classe Dynamic_GRU

    Exemplo::

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


    Aplica uma rede neural recorrente (RNN) a uma sequencia de entrada de comprimento variavel.

    A primeira entrada deve ser uma sequencia em lote com comprimento variavel definida
    atraves da classe ``tensor.PackedSequence``.

    A classe ``tensor.PackedSequence`` pode ser construida
    chamando a seguinte funcao em sucessao: ``pad_sequence``, ``pack_pad_sequence``.

    A primeira saida do Dynamic_RNN tambem eh uma classe ``tensor.PackedSequence``,
    que pode ser desempacotada para um QTensor normal usando ``tensor.pad_pack_sequence``.

    Modulo de rede neural recorrente (RNN), usando :math:`\tanh` ou :math:`\text{ReLU}` como funcao de ativacao. Suporta configuracao bidirecional e multiplas camadas.
    A formula de calculo do RNN unidirecional de camada unica eh a seguinte:

    .. math::
        h_t = \tanh(W_{ih} x_t + b_{ih} + W_{hh} h_{(t-1)} + b_{hh})

    Se :attr:`nonlinearity` for ``'relu'``, entao :math:`\text{ReLU}` substituira :math:`\tanh`.

    Esta classe herda de ``pyvqnet.nn.Module`` e ``torch.nn.Module``, e pode ser adicionada ao modelo torch como um submodulo de ``torch.nn.Module``.

    Os dados em ``_buffers`` desta classe sao do tipo ``torch.Tensor``.

    Os dados em ``_parmeters`` desta classe sao do tipo ``torch.nn.Parameter``.

    :param input_size: Input feature dimension.
    :param hidden_size: Hidden feature dimension.
    :param num_layers: Number of stacked RNN layers, default: 1.
    :param nonlinearity: Funcao de ativacao nao linear, padrao eh ``'tanh'``.
    :param batch_first: Se True, a forma de entrada eh [tamanho do lote, comprimento da sequencia, dimensao da caracteristica]; se False, a forma de entrada eh [comprimento da sequencia, tamanho do lote, dimensao da caracteristica]; padrao eh True.
    :param use_bias: Se False, este modulo nao aplica vies; padrao: True.
    :param bidirectional: Se True, torna-se um RNN bidirecional; padrao: False.
    :param dtype: O tipo de dados do parametro, padrao: None, usa o tipo de dados padrao: kfloat32, representando numeros de ponto flutuante de 32 bits.
    :param name: O nome deste modulo, padrao "".

    :return: Instancia Dynamic_RNN

    Exemplo::

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


    Aplica um LSTM RNN a sequencias de entrada de comprimento variavel.

    A primeira entrada deve ser uma sequencia em lote com comprimento variavel definida
    atraves de uma classe ``tensor.PackedSequence``.

    A classe ``tensor.PackedSequence`` pode ser construida
    chamando as seguintes funcoes em sucessao: ``pad_sequence``, ``pack_pad_sequence``.

    A primeira saida do Dynamic_LSTM tambem eh uma classe ``tensor.PackedSequence``,
    que pode ser desempacotada para um QTensor normal usando ``tensor.pad_pack_sequence``.

    Modulo de Rede Neural Recorrente (RNN), usando :math:`\tanh` ou :math:`\text{ReLU}` como funcao de ativacao. Suporta configuracao bidirecional e multiplas camadas.
    A formula de calculo do RNN unidirecional de camada unica eh a seguinte:
    
    
    .. math::
        \begin{array}{ll} \\
            i_t = \sigma(W_{ii} x_t + b_{ii} + W_{hi} h_{t-1} + b_{hi}) \\
            f_t = \sigma(W_{if} x_t + b_{if} + W_{hf} h_{t-1} + b_{hf}) \\
            g_t = \tanh(W_{ig} x_t + b_{ig} + W_{hg} h_{t-1} + b_{hg}) \\
            o_t = \sigma(W_{io} x_t + b_{io} + W_{ho} h_{t-1} + b_{ho}) \\
            c_t = f_t \odot c_{t-1} + i_t \odot g_t \\
            h_t = o_t \odot \tanh(c_t) \\
        \end{array}

    Esta classe herda de ``pyvqnet.nn.Module`` e ``torch.nn.Module``, e pode ser adicionada ao modelo torch como um submodulo de ``torch.nn.Module``.

    Os dados em ``_buffers`` desta classe sao do tipo ``torch.Tensor``.

    Os dados em ``_parmeters`` desta classe sao do tipo ``torch.nn.Parameter``.

    :param input_size: Input feature dimension.
    :param hidden_size: Hidden feature dimension.
    :param num_layers: Number of stacked LSTM layers, default: 1.
    :param batch_first: Se True, a forma de entrada eh [tamanho do lote, comprimento da sequencia, dimensao da caracteristica]; se False, a forma de entrada eh [comprimento da sequencia, tamanho do lote, dimensao da caracteristica]; padrao eh True.
    :param use_bias: Se False, este modulo nao aplica vies; padrao: True.
    :param bidirectional: Se True, torna-se um LSTM bidirecional; padrao: False.
    :param dtype: O tipo de dados do parametro, padrao: None, usa o tipo de dados padrao: kfloat32, representando numeros de ponto flutuante de 32 bits.
    :param name: O nome deste modulo, padrao "".

    :return: Instancia Dynamic_LSTM

    Exemplo::

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

    Reduz/aumenta a amostragem da entrada.

    Atualmente, suporta apenas dados de entrada 4D.

    O tamanho da entrada eh interpretado como `B x C x H x W`.

    As opcoes de `mode` disponiveis sao ``nearest``, ``bilinear``, ``bicubic``.

    Esta classe herda de ``pyvqnet.nn.Module`` e ``torch.nn.Module``, e pode ser adicionada ao modelo torch como um submodulo de ``torch.nn.Module``.

    :param size: Tamanho de saida, padrao None.
    :param scale_factor: Fator de escala, padrao None.
    :param mode: Algoritmo usado para upsampling ``nearest`` | ``bilinear`` | ``bicubic``.
    :param align_corners: From a geometric point of view, we treat the pixels of the input and output as squares instead of points. The pixels of the input and output are treated as squares instead of points.If set to `true`, the input and output tensors will be aligned by the center points of their corner pixels. Corner pixel center points are aligned, and the values ​​of the corner pixels are preserved.If set to `false`, the input and output tensors will be aligned by the corner points of their corner pixels, and the values ​​of the corner pixels are preserved. Corner pixel corner points are aligned, and interpolation will use edge values ​​for padding.Values ​​outside the boundaries are padded, making this operation independent of the input size.When ``scale_factor`` remains unchanged. This only works when ``mode`` is ``bilinear``.
    :param recompute_scale_factor: Recalcular o fator de escala para uso no calculo de interpolacao. Quando ``scale_factor`` eh passado como argumento, ele sera usado para calcular o tamanho de saida.
    :param name: Nome do modulo.

    Exemplo::

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

    Constroi uma classe que calcula atencao de produto escalar escalado para tensores de consulta, chave e valor.

    Esta classe herda de ``pyvqnet.nn.Module`` e ``torch.nn.Module``, e pode ser adicionada ao modelo torch como um submodulo de ``torch.nn.Module``.

    :param attn_mask: Mascara de atencao; valor padrao: None. A forma deve ser transmitivel para a forma dos pesos de atencao.
    :param dropout_p: Dropout probability; default value: 0, if greater than 0.0, dropout is applied.
    :param scale: Scaling factor applied before softmax, default value: None.
    :param is_causal: valor padrao: False, se definido como true, a mascara de atencao eh uma matriz triangular inferior quando a mascara eh uma matriz quadrada. Se ambos attn_mask e is_causal forem definidos, um erro sera gerado.
    :return: Uma classe SDPA

    Exemplos::
    
        from pyvqnet.nn.torch import SDPA
        from pyvqnet import tensor
        import pyvqnet
        pyvqnet.backends.set_backend("torch")
        model = SDPA(tensor.QTensor([1.]))

   .. py:method:: forward(query,key,value)

        Realiza a computacao direta.

        :param query: O QTensor de entrada da consulta.
        :param key: O QTensor de entrada da chave.
        :param value: O QTensor de entrada da chave.
        :return: O QTensor retornado pelo calculo SDPA.
        
        Exemplos::
        
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

API de Funcoes de Perda
------------------------

MeanSquaredError
^^^^^^^^^^^^^^^^^^^^^^^^

.. py:class:: pyvqnet.nn.torch.MeanSquaredError(name="")

    Calcula o erro quadratico medio entre a entrada :math:`x` e o valor alvo :math:`y`.

    Se o erro quadratico pode ser descrito pela seguinte funcao:

    .. math::
        \ell(x, y) = L = \{l_1,\dots,l_N\}^\top, \quad
        l_n = \left( x_n - y_n \right)^2,

    :math:`x` e :math:`y` sao QTensor de formas arbitrarias, e o erro quadratico medio do total de :math:`n` elementos eh calculado da seguinte forma.

    .. math::
        \ell(x, y) =
        \operatorname{mean}(L)

    Esta classe herda de ``pyvqnet.nn.Module`` e ``torch.nn.Module``, e pode ser adicionada ao modelo torch como um submodulo de ``torch.nn.Module``.

    :param name: O nome deste modulo, padrao "".
    :return: Uma instancia de erro RMS.

    Parametros necessarios para a funcao de calculo direto do erro RMS:

        x: :math:`(N, *)` valor previsto, onde :math:`*` representa qualquer dimensao.

        y: :math:`(N, *)`, valor alvo, um QTensor da mesma dimensao que a entrada.

    .. note::

        Note que, diferentemente de frameworks como pytorch, na funcao direta da funcao MeanSquaredError a seguir, o primeiro parametro eh o valor alvo e o segundo parametro eh o valor previsto.


    Exemplo::

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

    Mede a perda media de entropia cruzada binaria entre o alvo e a entrada.

    A entropia cruzada binaria sem media eh a seguinte:

    .. math::
        \ell(x, y) = L = \{l_1,\dots,l_N\}^\top, \quad
        l_n = - w_n \left[ y_n \cdot \log x_n + (1 - y_n) \cdot \log (1 - x_n) \right],

    where :math:`N` is the batch size.

    .. math::
        \ell(x, y) = \operatorname{mean}(L)

    Esta classe herda de ``pyvqnet.nn.Module`` e ``torch.nn.Module``, e pode ser adicionada a modelos torch como um submodulo de ``torch.nn.Module``.

    :param name: O nome deste modulo, padrao "".
    :return: Uma instancia de entropia cruzada binaria media.

    Parametros necessarios para a funcao de calculo direto do erro de entropia cruzada binaria media:

        x: :math:`(N, *)` valor previsto, onde :math:`*` representa qualquer dimensao.

        y: :math:`(N, *)`, valor alvo, um QTensor da mesma dimensao que a entrada.

    .. note::

        Note que, diferentemente de frameworks como pytorch, na funcao direta da funcao BinaryCrossEntropy, o primeiro parametro eh o valor alvo e o segundo parametro eh o valor previsto.
        
    Exemplo::

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

    Esta funcao de perda combina LogSoftmax e NLLLoss para calcular a entropia cruzada categorica media.

    A funcao de perda eh calculada da seguinte forma, onde class eh o rotulo de categoria correspondente do valor alvo:

    .. math::
        \text{loss}(x, y) = -\log\left(\frac{\exp(x[class])}{\sum_j \exp(x[j])}\right)
        = -x[class] + \log\left(\sum_j \exp(x[j])\right)

    :param name: O nome deste modulo, padrao "".
    :return: A instancia de entropia cruzada categorica media.

    Parametros necessarios para a funcao de calculo direto do erro:

        x: :math:`(N, *)` Predicted value, where :math:`*` indicates any dimension.

        y: :math:`(N, *)`, target value, a QTensor of the same dimension as the input. Must be a 64-bit integer, kint64.

    .. note::

        Note que, diferentemente de pytorch e outros frameworks, na funcao direta da funcao CategoricalCrossEntropy, o primeiro parametro eh o valor alvo e o segundo parametro eh o valor previsto.

        Esta classe herda de ``pyvqnet.nn.Module`` e ``torch.nn.Module``, e pode ser adicionada ao modelo torch como um submodulo de ``torch.nn.Module``.

    Exemplo::

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

    Esta funcao de perda combina LogSoftmax e NLLLoss para calcular a entropia cruzada de classificacao media, e tem maior estabilidade numerica.

    The loss function is calculated as follows, where class is the corresponding classification label of the target value:

    .. math::
        \text{loss}(x, y) = -\log\left(\frac{\exp(x[class])}{\sum_j \exp(x[j])}\right)
        = -x[class] + \log\left(\sum_j \exp(x[j])\right)

    :param name: O nome deste modulo, padrao "".
    :return: Uma instancia de funcao de perda de entropia cruzada Softmax

    Parametros necessarios para a funcao de calculo direto do erro:

        x: :math:`(N, *)` predicted value, where :math:`*` indicates any dimension.

        y: :math:`(N, *)`, target value, a QTensor of the same dimension as the input. Must be a 64-bit integer, kint64.

    .. note::

        Note que, diferentemente de pytorch e outros frameworks, na funcao direta da funcao SoftmaxCrossEntropy, o primeiro parametro eh o valor alvo e o segundo parametro eh o valor previsto.

        Esta classe herda de ``pyvqnet.nn.Module`` e ``torch.nn.Module``, e pode ser adicionada ao modelo torch como um submodulo de ``torch.nn.Module``.
        
    Exemplo::

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

    
    Perda media de log-verossimilhanca negativa. Util para problemas de classificacao com C classes.

    `x` eh a probabilidade de verossimilhanca fornecida pelo modelo. Sua forma pode ser :math:`(N, C)` ou :math:`(N, C, d_1, d_2, ..., d_K)`. `y` eh o valor verdadeiro esperado da funcao de perda, contendo indices de classe em :math:`[0, C-1]`.

    .. math::
        \ell(x, y) = L = \{l_1,\dots,l_N\}^\top, \quad
        l_n = -
        \sum_{n=1}^N \frac{1}{N}x_{n,y_n} \quad

    :param name: O nome deste modulo, padrao "".
    :return: Uma instancia de funcao de perda NLL_Loss

    Parametros necessarios para a funcao de calculo direto do erro:

        x: :math:`(N, *)`, o valor de previsao de saida da funcao de perda, que pode ser uma variavel multidimensional.

        y: :math:`(N, *)`, o valor alvo da funcao de perda. Deve ser um inteiro de 64 bits, kint64.

    .. note::

        Note que, diferentemente de frameworks como pytorch, na funcao direta da funcao NLL_Loss, o primeiro parametro eh o valor alvo e o segundo parametro eh o valor de previsao.

        Esta classe herda de ``pyvqnet.nn.Module`` e ``torch.nn.Module``, e pode ser adicionada ao modelo torch como um submodulo de ``torch.nn.Module``.
            
    Exemplo::

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

    Esta funcao calcula a perda de LogSoftmax e NLL_Loss juntas.

    `x` contem a saida nao normalizada. Sua forma pode ser :math:`(C)`, :math:`(N, C)` bidimensional ou :math:`(N, C, d_1, d_2, ..., d_K)` multidimensional.

    The formula of the loss function is as follows, where class is the corresponding class label of the target value:

    .. math::
        \text{loss}(x, y) = -\log\left(\frac{\exp(x[class])}{\sum_j \exp(x[j])}\right)
        = -x[class] + \log\left(\sum_j \exp(x[j])\right)

    :param name: O nome deste modulo, padrao "".
    :return: Uma instancia de funcao de perda CrossEntropyLoss

    Parametros necessarios para a funcao de calculo direto do erro:

        x: :math:`(N, *)`, a saida da funcao de perda, que pode ser uma variavel multidimensional.

        y: :math:`(N, *)`, o valor verdadeiro esperado da funcao de perda. Deve ser um inteiro de 64 bits, kint64.

    .. note::

        Note que, diferentemente de frameworks como pytorch, na funcao direta da funcao CrossEntropyLoss, o primeiro parametro eh o valor alvo e o segundo parametro eh o valor previsto.

        Esta classe herda de ``pyvqnet.nn.Module`` e ``torch.nn.Module``, e pode ser adicionada ao modelo torch como um submodulo de ``torch.nn.Module``.

    Exemplo::

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


Funcoes de Ativacao
---------------------

Sigmoid
^^^^^^^^^^^^^^^^^^^^^^^^

.. py:class:: pyvqnet.nn.torch.Sigmoid(name:str="")

    Camada de funcao de ativacao Sigmoid.

    .. math::
        \text{Sigmoid}(x) = \frac{1}{1 + \exp(-x)}

    Esta classe herda de ``pyvqnet.nn.Module`` e ``torch.nn.Module``, e pode ser adicionada ao modelo torch como um submodulo de ``torch.nn.Module``.

    :param name: O nome da camada de funcao de ativacao, padrao "".

    :return: Uma instancia de camada de funcao de ativacao Sigmoid.

    Exemplos::

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

    Esta classe herda de ``pyvqnet.nn.Module`` e ``torch.nn.Module``, e pode ser adicionada ao modelo torch como um submodulo de ``torch.nn.Module``.

    :param name: O nome da camada de funcao de ativacao, padrao "".

    :return: uma instancia Softplus.

    Exemplos::

        from pyvqnet.nn.torch import Softplus
        from pyvqnet.tensor import QTensor
        import pyvqnet
        pyvqnet.backends.set_backend("torch")
        layer = Softplus()
        y = layer(QTensor([1.0, 2.0, 3.0, 4.0]))

Softsign
^^^^^^^^^^^^^^^^^^^^^^^^

.. py:class:: pyvqnet.nn.torch.Softsign(name:str="")

    Softsign .

    .. math::
        \text{SoftSign}(x) = \frac{x}{ 1 + |x|}


    Esta classe herda de ``pyvqnet.nn.Module`` e ``torch.nn.Module``, e pode ser adicionada ao modelo torch como um submodulo de ``torch.nn.Module``.


    :param name: O nome da camada de funcao de ativacao, padrao "".

    :return: uma instancia SoftSign.

    Exemplos::

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


    Esta classe herda de ``pyvqnet.nn.Module`` e ``torch.nn.Module``, e pode ser adicionada ao modelo torch como um submodulo de ``torch.nn.Module``.


    :param axis: a dimensao para calcular (o ultimo eixo eh -1); valor padrao = -1.
    :param name: O nome da camada de funcao de ativacao, padrao "".

    :return: uma instancia Softmax.

    Exemplos::

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


    Esta classe herda de ``pyvqnet.nn.Module`` e ``torch.nn.Module``, e pode ser adicionada ao modelo torch como um submodulo de ``torch.nn.Module``.


    :param name: O nome da camada de funcao de ativacao, padrao "".

    :return: Instancia HardSigmoid.

    Exemplos::

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


    Esta classe herda de ``pyvqnet.nn.Module`` e ``torch.nn.Module``, e pode ser adicionada ao modelo torch como um submodulo de ``torch.nn.Module``.


    :param name: O nome da camada de funcao de ativacao, padrao "".

    :return: uma instancia ReLu.

    Exemplos::

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


    Esta classe herda de ``pyvqnet.nn.Module`` e ``torch.nn.Module``, e pode ser adicionada ao modelo torch como um submodulo de ``torch.nn.Module``.


    :param alpha: LeakyRelu coefficient, default: 0.01.
    :param name: O nome da camada de funcao de ativacao, padrao "".

    :return: uma instancia de ativacao LeakyReLu.

    Exemplos::

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

    When the approximation parameter is 'tanh', GELU is estimated as follows:

    .. math:: \text{GELU}(x) = 0.5 * x * (1 + \text{Tanh}(\sqrt{2 / \pi} * (x + 0.044715 * x^3)))


    Esta classe herda de ``pyvqnet.nn.Module`` e ``torch.nn.Module``, e pode ser adicionada ao modelo torch como um submodulo de ``torch.nn.Module``.


    :param approximate: Metodo de calculo de aproximacao; padrao eh "tanh".
    :param name: O nome da camada de funcao de ativacao, padrao "".

    :return: Instancia de ativacao Gelu.

    Exemplos::

        from pyvqnet.tensor import randu, ones_like
        from pyvqnet.nn.torch import Gelu
        import pyvqnet
        pyvqnet.backends.set_backend("torch")
        qa = randu([5,4])
        qb = Gelu()(qa)



ELU
^^^^^^^^^^^^^^^^^^^^^^^^

.. py:class:: pyvqnet.nn.torch.ELU(alpha:float=1,name:str="")

    ELU Camada de funcao de ativacao de Unidade Linear Exponencial.

    .. math::
        \text{ELU}(x) = \begin{cases}
        x, & \text{ if } x > 0\\
        \alpha * (\exp(x) - 1), & \text{ if } x \leq 0
        \end{cases}


    Esta classe herda de ``pyvqnet.nn.Module`` e ``torch.nn.Module``, e pode ser adicionada ao modelo torch como um submodulo de ``torch.nn.Module``.



    :param alpha: Coeficiente ELU; padrao:1.
    :param name: O nome da camada de funcao de ativacao, padrao "".

    :return: Instancia de ativacao ELU.

    Exemplos::

        from pyvqnet.nn.torch import ELU
        from pyvqnet.tensor import QTensor
        import pyvqnet
        pyvqnet.backends.set_backend("torch")
        layer = ELU()
        y = layer(QTensor([-1, 2.0, -3, 4.0]))


Tanh
^^^^^^^^^^^^^^^^^^^^^^^^

.. py:class:: pyvqnet.nn.torch.Tanh(name:str="")

    Tanh funcao de ativacao tangente hiperbolica.

    .. math::
        \text{Tanh}(x) = \frac{\exp(x) - \exp(-x)} {\exp(x) + \exp(-x)}


    Esta classe herda de ``pyvqnet.nn.Module`` e ``torch.nn.Module``, e pode ser adicionada ao modelo torch como um submodulo de ``torch.nn.Module``.



    :param name: O nome da camada de funcao de ativacao, padrao "".

    :return: Instancia de ativacao Tanh.

    Exemplos::

        from pyvqnet.nn.torch import Tanh
        from pyvqnet.tensor import QTensor
        import pyvqnet
        pyvqnet.backends.set_backend("torch")
        layer = Tanh()
        y = layer(QTensor([-1, 2.0, -3, 4.0]))



Modulo Otimizador
---------------------------------------------

Para modulos de circuito quantico classico e quantico herdados de `TorchModule`, os parametros `model.paramters()` podem continuar a ser otimizados usando otimizadores diferentes de `Rotosolve` em :ref:`Optimizer`.



Usando pyqpanda para executar circuito quantico variacional
-------------------------------------------------------------------------

A seguir, a interface de circuito quantico variacional de treinamento para calculo de circuito usando pyqpanda e pyqpanda3.

.. warning::

    A parte de computacao quantica do TorchQpandaQuantumLayer a seguir usa pyqpanda2.

    Devido a problemas de compatibilidade entre pyqpanda2 e pyqpanda3, voce precisa instalar o pyqpnda2 por conta propria: `pip install pyqpanda`

TorchQpandaQuantumLayer
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Se voce esta mais familiarizado com a sintaxe pyQPanda2, pode usar a interface TorchQpandaQuantumLayer, adicionando bits quanticos personalizados ``qubits``, bits classicos ``cbits`` e simulador de backend ``machine`` a funcao de parametro ``qprog_with_measure`` do TorchQpandaQuantumLayer.

.. py:class:: pyvqnet.qnn.vqc.torch.TorchQpandaQuantumLayer(qprog_with_measure,para_num,diff_method:str = "parameter_shift",delta:float = 0.01,dtype=None,name="")

    Modulo de computacao abstrato da camada quantica variacional. Use pyQPanda2 para simular um circuito quantico parametrizado e obter os resultados de medicao. Esta camada quantica variacional herda o modulo de calculo de gradiente do framework VQNet. Pode usar o metodo de desvio de parametro para calcular o gradiente dos parametros do circuito, treinar modelos de circuito quantico variacional ou incorporar circuitos quanticos variacionais em modelos quanticos e classicos hibridos.

    :param qprog_with_measure: Quantum circuit operation and measurement functions built with pyQPand.
    :param para_num: `int` - number of parameters.
    :param diff_method: Method for solving quantum circuit parameter gradients, "parameter shift" or "finite difference", default parameter shift.
    :param delta: \delta when calculating gradients by finite difference.
    :param dtype: Data type of parameter, default: None, use default data type: kfloat32, representing 32-bit floating point numbers.
    :param name: O nome deste modulo, padrao "".

    :return: Um modulo que pode calcular circuitos quanticos.

    .. note::

        qprog_with_measure eh uma funcao de circuito quantico definida em pyQPanda2.

        Esta funcao deve conter os seguintes parametros como entrada de funcao (mesmo que um parametro nao seja realmente usado), caso contrario, nao funcionara corretamente nesta funcao.

        Comparado com QuantumLayer, na funcao de execucao de circuito variacional passada por esta interface, o usuario deve criar manualmente bits quanticos e simuladores.

        Se qprog_with_measure requer medicao quantica, o usuario tambem precisa criar e alocar cbits manualmente.

        O uso da funcao de circuito quantico qprog_with_measure (input, param, nqubits, ncbits) pode consultar o exemplo a seguir.

        `input`: Insira dados classicos unidimensionais. Se nao houver, insira None.

        `param`: Insira parametros de circuito quantico variacional unidimensionais a serem treinados.

    Exemplo::

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

    The quantum computing part of the following TorchQcloud3QuantumLayer and TorchQpanda3QuantumLayer interfaces uses pyqpanda3.

    Se voce usar a funcao QCloud neste modulo, havera erros ao importar pyqpanda2 no codigo ou usar interfaces de pacotes relacionados a pyqpanda2 do pyvqnet.

TorchQcloud3QuantumLayer
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Quando voce instala a versao mais recente do pyqpanda3, pode usar esta interface para definir um circuito variacional e submete-lo ao chip real do originqc para operacao.

.. py:class:: pyvqnet.qnn.vqc.torch.TorchQcloud3QuantumLayer(origin_qprog_func, qcloud_token, para_num, pauli_str_dict=None, shots = 1000, initializer=None, dtype=None, name="", diff_method="parameter_shift", submit_kwargs={}, query_kwargs={})
    
    Um modulo de computacao abstrato para chips reais usando originqc do pyqpanda3. Ele submete circuitos quanticos parametrizados a chips reais e obtem resultados de medicao.
    Se diff_method == "random_coordinate_descent", a camada selecionara aleatoriamente um unico parametro para calcular o gradiente, e outros parametros permanecerao zero. Referencia: https://arxiv.org/abs/2311.00088

    .. note::

        qcloud_token eh o token de api que voce solicitou da plataforma em nuvem.

        origin_qprog_func needs to return data of type pypqanda3.core.QProg. If pauli_str_dict is not set, it is necessary to ensure that the measure has been inserted into the QProg.

        origin_qprog_func must be in the following format:

        origin_qprog_func(input,param )

        `input`: Input 1~2D classical data. In the case of 2D, the first dimension is the batch size.

        `param`: Input the parameters to be trained of the 1D variational quantum circuit.

    .. warning::

        Esta classe herda de ``pyvqnet.nn.Module`` e ``torch.nn.Module``, e pode ser adicionada ao modelo torch como um submodulo de ``torch.nn.Module``.

        Os dados em ``_buffers`` desta classe sao do tipo ``torch.Tensor``.

        Os dados em ``_parmeters`` desta classe sao do tipo ``torch.nn.Parameter``.

    :param origin_qprog_func: A funcao de circuito quantico variacional construida por QPanda, que deve retornar QProg.
    :param qcloud_token: `str` - O tipo de maquina quantica ou token de nuvem para execucao.
    :param para_num: `int` - O numero de parametros, o parametro eh um QTensor de tamanho [para_num].
    :param pauli_str_dict: `dict|list` - Dicionario ou lista de dicionarios representando operadores de Pauli em circuitos quanticos. Padrao "None", o que significa que operacoes de medicao sao realizadas. Se um dicionario de operadores de Pauli for inserido, uma expectativa unica ou multiplas expectativas sao calculadas.
    :param shot: `int` - Numero de medicoes. O valor padrao eh 1000.
    :param initializer: Inicializador para valores de parametro. O valor padrao eh "None", usando uma distribuicao normal 0~2*pi.
    :param dtype: Tipo de dados do parametro. O valor padrao eh None, que significa usar o tipo de dados padrao pyvqnet.kfloat32.
    :param name: O nome do modulo. O valor padrao eh uma string vazia.
    :param diff_method: Metodo de diferenciacao para calculo de gradiente. O valor padrao eh "parameter_shift", "random_coordinate_descent".
    :param submit_kwargs: Additional keyword parameters for submitting quantum circuits, default: {"if_print_qcloud_log":False,"chip_id":"WK_C180","is_amend":True,"is_mapping":True,"is_optimization":True,"compile_level":3,"default_task_group_size":200,"test_qcloud_fake":False,"":"server_ip_address"}, when test_qcloud_fake is set to True, local CPUQVM simulation.
    :param query_kwargs: Additional keyword parameters for querying quantum results, default: {"timeout":2,"print_query_info":True,"sub_circuits_split_size":1}.
    :return: Um modulo que pode calcular circuitos quanticos.


    Exemplo::

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

Se voce esta mais familiarizado com a sintaxe pyQPanda3, pode usar a interface TorchQpanda3QuantumLayer.

.. py:class:: pyvqnet.qnn.vqc.torch.TorchQpanda3QuantumLayer(qprog_with_measure,para_num,diff_method:str = "parameter_shift",delta:float = 0.01,dtype=None,name="")

    Modulo de computacao abstrato da camada quantica variacional. Use pyQPanda3 para simular um circuito quantico parametrizado e obter os resultados de medicao. Esta camada quantica variacional herda o modulo de computacao de gradiente do framework VQNet. Voce pode usar o metodo de desvio de parametro para calcular o gradiente dos parametros do circuito, treinar o modelo de circuito quantico variacional ou incorporar o circuito quantico variacional em um modelo quantico e classico hibrido.

    :param qprog_with_measure: Quantum circuit operation and measurement functions built with pyQPand.
    :param para_num: `int` - number of parameters.
    :param diff_method: method for solving quantum circuit parameter gradients, "parameter shift" or "finite difference", default parameter shift.
    :param delta: \delta when calculating gradients by finite difference.
    :param dtype: data type of parameter, default: None, use default data type: kfloat32, representing 32-bit floating point numbers.
    :param name: o nome deste modulo, padrao "".

    :return: um modulo que pode calcular circuitos quanticos.

    .. note::

        qprog_with_measure is a quantum circuit function defined in pyQPanda.

        Esta funcao deve incluir os seguintes parametros como entradas de funcao (mesmo que um parametro nao seja realmente usado), caso contrario, nao funcionara corretamente nesta funcao.

        The use of the quantum circuit function qprog_with_measure (input,param,nqubits,ncbits) can refer to the following example.

        `input`: Insira dados classicos unidimensionais. Se nao houver, insira None.

        `param`: Insira os parametros a serem treinados para o circuito quantico variacional unidimensional.

    Exemplo::

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

Modulo e interface de circuito quantico variacional baseado em diferenciacao automatica
---------------------------------------------------------------------------------------------
Classe Base
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Escrever um modelo de circuito quantico variacional requer herdar de ``QModule``.

QModule
""""""""""""""""""

.. py:class:: pyvqnet.qnn.vqc.torch.QModule(name="")

    Quando o usuario usa o backend `torch`, define a classe base que o modelo de circuito quantico variacional `Module` deve herdar.
    Esta classe herda de ``pyvqnet.nn.torch.TorchModule`` e ``torch.nn.Module``.

    Esta classe pode ser adicionada ao modelo torch como um submodulo de ``torch.nn.Module``.

    .. note::

        Esta classe e suas classes derivadas sao aplicaveis apenas a ``pyvqnet.backends.set_backend("torch")``, nao misture com o ``Module`` sob o ``pyvqnet.nn`` padrao.

        Os dados em ``_buffers`` desta classe sao do tipo ``torch.Tensor``.

        Os dados em ``_parmeters`` desta classe sao do tipo ``torch.nn.Parameter``.


QMachine
""""""""""""""""""

.. py:class:: pyvqnet.qnn.vqc.torch.QMachine(num_wires, dtype=pyvqnet.kcomplex64,grad_mode="",save_ir=False)

    Classe simuladora para computacao quantica variacional, incluindo vetores de estado cujo atributo states sao circuitos quanticos.

    Esta classe herda de ``pyvqnet.nn.torch.TorchModule`` e ``pyvqnet.qnn.QMachine``.

    Esta classe pode ser adicionada ao modelo torch como um submodulo de ``torch.nn.Module``.

    .. note::

        Before each run of the complete quantum circuit, you must use `pyvqnet.qnn.vqc.QMachine.reset_states(batchsize)` to reinitialize the initial state in the simulator and broadcast it to
        dimensoes (batchsize,*) para se adaptar ao treinamento de dados em lote.

    :param num_wires: O numero de bits quanticos.
    :param dtype: The data type of the calculated data. The default value is pyvqnet. kcomplex64, and the corresponding parameter precision is pyvqnet.kfloat32.
    :param grad_mode: O modo de calculo de gradiente, que pode ser "adjoint", o valor padrao: "", usa diferenciacao automatica.
    :param save_ir: Quando definido como True, salva a operacao em originIR, o valor padrao: False.

    :return: Saida um objeto QMachine.

    Exemplo::
        
        from pyvqnet.qnn.vqc.torch import QMachine
        import pyvqnet
        pyvqnet.backends.set_backend("torch")
        qm = QMachine(4)
        print(qm.states)


   .. py:method:: reset_states(batchsize)

        Reinicializa o estado inicial no simulador e o transmite para
        dimensoes (batchsize,*) para se adaptar ao treinamento de dados em lote.

        :param batchsize: Dimensao de processamento em lote.

Modulo de porta logica quantica variacional
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The following function interfaces in ``pyvqnet.qnn.vqc`` directly support ``QTensor`` of ``torch`` backend for calculation.

.. csv-table:: List of supported pyvqnet.qnn.vqc interfaces
    :file: ./images/same_apis_from_vqc.csv

Os seguintes modulos de circuito quantico herdam de ``pyvqnet.qnn.vqc.torch.QModule``, onde os calculos sao realizados usando ``torch.Tensor``.

.. note::

    This class and its derived classes are only applicable to ``pyvqnet.backends.set_backend("torch")``, do not mix with ``Module`` under the default ``pyvqnet.nn``.

    Se estas classes tiverem variaveis membro nao parametro ``_buffers``, os dados nelas sao do tipo ``torch.Tensor``.
    Se estas classes tiverem variaveis membro de parametro ``_parmeters``, os dados nelas sao do tipo ``torch.nn.Parameter``.

I
""""""""""""""""""

.. py:class:: pyvqnet.qnn.vqc.torch.I(has_params: bool = False,trainable: bool = False,init_params=None,wires=None,dtype=pyvqnet.kcomplex64,use_dagger=False)
    
    define a I quantum gate .

    Esta classe herda de ``pyvqnet.qnn.vqc.torch.QModule`` e ``torch.nn.Module``.
    Esta classe pode ser adicionada ao modelo torch como um submodulo de ``torch.nn.Module``.

    :param has_params: se possui parametros, como portas RX, RY e outras precisam ser definidas como True, e aquelas sem parametros precisam ser definidas como False, o padrao eh False.
    :param trainable: se possui parametros a serem treinados. Se a camada usa dados de entrada externos para construir a matriz da porta logica, defina como False. Se os parametros a serem treinados precisam ser inicializados a partir desta camada, eh True, o padrao eh False.
    :param init_params: Parametros de inicializacao usados para codificar dados classicos QTensor, o padrao eh None.
    :param wires: Indice do bit do efeito de linha, o padrao eh None.
    :param dtype: A precisao dos dados da matriz interna da porta logica pode ser definida como pyvqnet.kcomplex64 ou pyvqnet.kcomplex128, correspondendo a entrada float ou double respectivamente.
    :param use_dagger: se deve usar a versao conjugada transposta da porta, o padrao eh False.
    :return: uma instancia ``pyvqnet.qnn.vqc.torch.QModule``

    Exemplo::
        
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
    
    define a Hadamard quantum gate .

    Esta classe herda de ``pyvqnet.qnn.vqc.torch.QModule`` e ``torch.nn.Module``.
    Esta classe pode ser adicionada ao modelo torch como um submodulo de ``torch.nn.Module``.

    :param has_params: se possui parametros, como portas RX, RY e outras precisam ser definidas como True, e aquelas sem parametros precisam ser definidas como False, o padrao eh False.
    :param trainable: se possui parametros a serem treinados. Se a camada usa dados de entrada externos para construir a matriz da porta logica, defina como False. Se os parametros a serem treinados precisam ser inicializados a partir desta camada, eh True, o padrao eh False.
    :param init_params: Parametros de inicializacao usados para codificar dados classicos QTensor, o padrao eh None.
    :param wires: Indice do bit do efeito de linha, o padrao eh None.
    :param dtype: A precisao dos dados da matriz interna da porta logica pode ser definida como pyvqnet.kcomplex64 ou pyvqnet.kcomplex128, correspondendo a entrada float ou double respectivamente.
    :param use_dagger: se deve usar a versao conjugada transposta da porta, o padrao eh False.
    :return: uma instancia ``pyvqnet.qnn.vqc.torch.QModule``

    Exemplo::
        
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
    
    define a T quantum gate .

    Esta classe herda de ``pyvqnet.qnn.vqc.torch.QModule`` e ``torch.nn.Module``.
    Esta classe pode ser adicionada ao modelo torch como um submodulo de ``torch.nn.Module``.

    :param has_params: se possui parametros, como portas RX, RY e outras precisam ser definidas como True, e aquelas sem parametros precisam ser definidas como False, o padrao eh False.
    :param trainable: se possui parametros a serem treinados. Se a camada usa dados de entrada externos para construir a matriz da porta logica, defina como False. Se os parametros a serem treinados precisam ser inicializados a partir desta camada, eh True, o padrao eh False.
    :param init_params: Parametros de inicializacao usados para codificar dados classicos QTensor, o padrao eh None.
    :param wires: Indice do bit do efeito de linha, o padrao eh None.
    :param dtype: A precisao dos dados da matriz interna da porta logica pode ser definida como pyvqnet.kcomplex64 ou pyvqnet.kcomplex128, correspondendo a entrada float ou double respectivamente.
    :param use_dagger: se deve usar a versao conjugada transposta da porta, o padrao eh False.
    :return: uma instancia ``pyvqnet.qnn.vqc.torch.QModule``

    Exemplo::
        
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
    
    define a S quantum gate .

    Esta classe herda de ``pyvqnet.qnn.vqc.torch.QModule`` e ``torch.nn.Module``.
    Esta classe pode ser adicionada ao modelo torch como um submodulo de ``torch.nn.Module``.

    :param has_params: se possui parametros, como portas RX, RY e outras precisam ser definidas como True, e aquelas sem parametros precisam ser definidas como False, o padrao eh False.
    :param trainable: se possui parametros a serem treinados. Se a camada usa dados de entrada externos para construir a matriz da porta logica, defina como False. Se os parametros a serem treinados precisam ser inicializados a partir desta camada, eh True, o padrao eh False.
    :param init_params: Parametros de inicializacao usados para codificar dados classicos QTensor, o padrao eh None.
    :param wires: Indice do bit do efeito de linha, o padrao eh None.
    :param dtype: A precisao dos dados da matriz interna da porta logica pode ser definida como pyvqnet.kcomplex64 ou pyvqnet.kcomplex128, correspondendo a entrada float ou double respectivamente.
    :param use_dagger: se deve usar a versao conjugada transposta da porta, o padrao eh False.
    :return: uma instancia ``pyvqnet.qnn.vqc.torch.QModule``

    Exemplo::
        
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
    
    define a PauliX quantum gate .

    Esta classe herda de ``pyvqnet.qnn.vqc.torch.QModule`` e ``torch.nn.Module``.
    Esta classe pode ser adicionada ao modelo torch como um submodulo de ``torch.nn.Module``.


    :param has_params: se possui parametros, como portas RX, RY e outras precisam ser definidas como True, e aquelas sem parametros precisam ser definidas como False, o padrao eh False.
    :param trainable: se possui parametros a serem treinados. Se a camada usa dados de entrada externos para construir a matriz da porta logica, defina como False. Se os parametros a serem treinados precisam ser inicializados a partir desta camada, eh True, o padrao eh False.
    :param init_params: Parametros de inicializacao usados para codificar dados classicos QTensor, o padrao eh None.
    :param wires: Indice do bit do efeito de linha, o padrao eh None.
    :param dtype: A precisao dos dados da matriz interna da porta logica pode ser definida como pyvqnet.kcomplex64 ou pyvqnet.kcomplex128, correspondendo a entrada float ou double respectivamente.
    :param use_dagger: se deve usar a versao conjugada transposta da porta, o padrao eh False.
    :return: uma instancia ``pyvqnet.qnn.vqc.torch.QModule``

    Exemplo::
        
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
    
    define a PauliY quantum gate .

    Esta classe herda de ``pyvqnet.qnn.vqc.torch.QModule`` e ``torch.nn.Module``.
    Esta classe pode ser adicionada ao modelo torch como um submodulo de ``torch.nn.Module``.


    :param has_params: se possui parametros, como portas RX, RY e outras precisam ser definidas como True, e aquelas sem parametros precisam ser definidas como False, o padrao eh False.
    :param trainable: se possui parametros a serem treinados. Se a camada usa dados de entrada externos para construir a matriz da porta logica, defina como False. Se os parametros a serem treinados precisam ser inicializados a partir desta camada, eh True, o padrao eh False.
    :param init_params: Parametros de inicializacao usados para codificar dados classicos QTensor, o padrao eh None.
    :param wires: Indice do bit do efeito de linha, o padrao eh None.
    :param dtype: A precisao dos dados da matriz interna da porta logica pode ser definida como pyvqnet.kcomplex64 ou pyvqnet.kcomplex128, correspondendo a entrada float ou double respectivamente.
    :param use_dagger: se deve usar a versao conjugada transposta da porta, o padrao eh False.
    :return: uma instancia ``pyvqnet.qnn.vqc.torch.QModule``

    Exemplo::
        
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
    
    define a PauliZ quantum gate .

    Esta classe herda de ``pyvqnet.qnn.vqc.torch.QModule`` e ``torch.nn.Module``.
    Esta classe pode ser adicionada ao modelo torch como um submodulo de ``torch.nn.Module``.


    :param has_params: se possui parametros, como portas RX, RY e outras precisam ser definidas como True, e aquelas sem parametros precisam ser definidas como False, o padrao eh False.
    :param trainable: se possui parametros a serem treinados. Se a camada usa dados de entrada externos para construir a matriz da porta logica, defina como False. Se os parametros a serem treinados precisam ser inicializados a partir desta camada, eh True, o padrao eh False.
    :param init_params: Parametros de inicializacao usados para codificar dados classicos QTensor, o padrao eh None.
    :param wires: Indice do bit do efeito de linha, o padrao eh None.
    :param dtype: A precisao dos dados da matriz interna da porta logica pode ser definida como pyvqnet.kcomplex64 ou pyvqnet.kcomplex128, correspondendo a entrada float ou double respectivamente.
    :param use_dagger: se deve usar a versao conjugada transposta da porta, o padrao eh False.
    :return: uma instancia ``pyvqnet.qnn.vqc.torch.QModule``

    Exemplo::
        
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
    
    define a X1 quantum gate .

    Esta classe herda de ``pyvqnet.qnn.vqc.torch.QModule`` e ``torch.nn.Module``.
    Esta classe pode ser adicionada ao modelo torch como um submodulo de ``torch.nn.Module``.

    :param has_params: se possui parametros, como portas RX, RY e outras precisam ser definidas como True, e aquelas sem parametros precisam ser definidas como False, o padrao eh False.
    :param trainable: se possui parametros a serem treinados. Se a camada usa dados de entrada externos para construir a matriz da porta logica, defina como False. Se os parametros a serem treinados precisam ser inicializados a partir desta camada, eh True, o padrao eh False.
    :param init_params: Parametros de inicializacao usados para codificar dados classicos QTensor, o padrao eh None.
    :param wires: Indice do bit do efeito de linha, o padrao eh None.
    :param dtype: A precisao dos dados da matriz interna da porta logica pode ser definida como pyvqnet.kcomplex64 ou pyvqnet.kcomplex128, correspondendo a entrada float ou double respectivamente.
    :param use_dagger: se deve usar a versao conjugada transposta da porta, o padrao eh False.
    :return: uma instancia ``pyvqnet.qnn.vqc.torch.QModule``

    Exemplo::
        
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
    
    define a RX quantum gate .

    Esta classe herda de ``pyvqnet.qnn.vqc.torch.QModule`` e ``torch.nn.Module``.
    Esta classe pode ser adicionada ao modelo torch como um submodulo de ``torch.nn.Module``.

    :param has_params: se possui parametros, como portas RX, RY e outras precisam ser definidas como True, e aquelas sem parametros precisam ser definidas como False, o padrao eh False.
    :param trainable: se possui parametros a serem treinados. Se a camada usa dados de entrada externos para construir a matriz da porta logica, defina como False. Se os parametros a serem treinados precisam ser inicializados a partir desta camada, eh True, o padrao eh False.
    :param init_params: Parametros de inicializacao usados para codificar dados classicos QTensor, o padrao eh None.
    :param wires: Indice do bit do efeito de linha, o padrao eh None.
    :param dtype: A precisao dos dados da matriz interna da porta logica pode ser definida como pyvqnet.kcomplex64 ou pyvqnet.kcomplex128, correspondendo a entrada float ou double respectivamente.
    :param use_dagger: se deve usar a versao conjugada transposta da porta, o padrao eh False.
    :return: uma instancia ``pyvqnet.qnn.vqc.torch.QModule``

    Exemplo::

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
    
    define a RY quantum gate .

    Esta classe herda de ``pyvqnet.qnn.vqc.torch.QModule`` e ``torch.nn.Module``.
    Esta classe pode ser adicionada ao modelo torch como um submodulo de ``torch.nn.Module``.

    :param has_params: se possui parametros, como portas RX, RY e outras precisam ser definidas como True, e aquelas sem parametros precisam ser definidas como False, o padrao eh False.
    :param trainable: se possui parametros a serem treinados. Se a camada usa dados de entrada externos para construir a matriz da porta logica, defina como False. Se os parametros a serem treinados precisam ser inicializados a partir desta camada, eh True, o padrao eh False.
    :param init_params: Parametros de inicializacao usados para codificar dados classicos QTensor, o padrao eh None.
    :param wires: Indice do bit do efeito de linha, o padrao eh None.
    :param dtype: A precisao dos dados da matriz interna da porta logica pode ser definida como pyvqnet.kcomplex64 ou pyvqnet.kcomplex128, correspondendo a entrada float ou double respectivamente.
    :param use_dagger: se deve usar a versao conjugada transposta da porta, o padrao eh False.
    :return: uma instancia ``pyvqnet.qnn.vqc.torch.QModule``

    Exemplo::

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
    
    define a RZ quantum gate .

    Esta classe herda de ``pyvqnet.qnn.vqc.torch.QModule`` e ``torch.nn.Module``.
    Esta classe pode ser adicionada ao modelo torch como um submodulo de ``torch.nn.Module``.

    :param has_params: se possui parametros, como portas RX, RY e outras precisam ser definidas como True, e aquelas sem parametros precisam ser definidas como False, o padrao eh False.
    :param trainable: se possui parametros a serem treinados. Se a camada usa dados de entrada externos para construir a matriz da porta logica, defina como False. Se os parametros a serem treinados precisam ser inicializados a partir desta camada, eh True, o padrao eh False.
    :param init_params: Parametros de inicializacao usados para codificar dados classicos QTensor, o padrao eh None.
    :param wires: Indice do bit do efeito de linha, o padrao eh None.
    :param dtype: A precisao dos dados da matriz interna da porta logica pode ser definida como pyvqnet.kcomplex64 ou pyvqnet.kcomplex128, correspondendo a entrada float ou double respectivamente.
    :param use_dagger: se deve usar a versao conjugada transposta da porta, o padrao eh False.
    :return: uma instancia ``pyvqnet.qnn.vqc.torch.QModule``

    Exemplo::

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
    
    define a CRX quantum gate .

    Esta classe herda de ``pyvqnet.qnn.vqc.torch.QModule`` e ``torch.nn.Module``.
    Esta classe pode ser adicionada ao modelo torch como um submodulo de ``torch.nn.Module``.

    :param has_params: se possui parametros, como portas RX, RY e outras precisam ser definidas como True, e aquelas sem parametros precisam ser definidas como False, o padrao eh False.
    :param trainable: se possui parametros a serem treinados. Se a camada usa dados de entrada externos para construir a matriz da porta logica, defina como False. Se os parametros a serem treinados precisam ser inicializados a partir desta camada, eh True, o padrao eh False.
    :param init_params: Parametros de inicializacao usados para codificar dados classicos QTensor, o padrao eh None.
    :param wires: Indice do bit do efeito de linha, o padrao eh None.
    :param dtype: A precisao dos dados da matriz interna da porta logica pode ser definida como pyvqnet.kcomplex64 ou pyvqnet.kcomplex128, correspondendo a entrada float ou double respectivamente.
    :param use_dagger: se deve usar a versao conjugada transposta da porta, o padrao eh False.
    :return: uma instancia ``pyvqnet.qnn.vqc.torch.QModule``

    Exemplo::

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
    
    define a CRY quantum gate .

    Esta classe herda de ``pyvqnet.qnn.vqc.torch.QModule`` e ``torch.nn.Module``.
    Esta classe pode ser adicionada ao modelo torch como um submodulo de ``torch.nn.Module``.

    :param has_params: se possui parametros, como portas RX, RY e outras precisam ser definidas como True, e aquelas sem parametros precisam ser definidas como False, o padrao eh False.
    :param trainable: se possui parametros a serem treinados. Se a camada usa dados de entrada externos para construir a matriz da porta logica, defina como False. Se os parametros a serem treinados precisam ser inicializados a partir desta camada, eh True, o padrao eh False.
    :param init_params: Parametros de inicializacao usados para codificar dados classicos QTensor, o padrao eh None.
    :param wires: Indice do bit do efeito de linha, o padrao eh None.
    :param dtype: A precisao dos dados da matriz interna da porta logica pode ser definida como pyvqnet.kcomplex64 ou pyvqnet.kcomplex128, correspondendo a entrada float ou double respectivamente.
    :param use_dagger: se deve usar a versao conjugada transposta da porta, o padrao eh False.
    :return: uma instancia ``pyvqnet.qnn.vqc.torch.QModule``

    Exemplo::

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
    
    define a CRZ quantum gate .

    Esta classe herda de ``pyvqnet.qnn.vqc.torch.QModule`` e ``torch.nn.Module``.
    Esta classe pode ser adicionada ao modelo torch como um submodulo de ``torch.nn.Module``.

    :param has_params: se possui parametros, como portas RX, RY e outras precisam ser definidas como True, e aquelas sem parametros precisam ser definidas como False, o padrao eh False.
    :param trainable: se possui parametros a serem treinados. Se a camada usa dados de entrada externos para construir a matriz da porta logica, defina como False. Se os parametros a serem treinados precisam ser inicializados a partir desta camada, eh True, o padrao eh False.
    :param init_params: Parametros de inicializacao usados para codificar dados classicos QTensor, o padrao eh None.
    :param wires: Indice do bit do efeito de linha, o padrao eh None.
    :param dtype: A precisao dos dados da matriz interna da porta logica pode ser definida como pyvqnet.kcomplex64 ou pyvqnet.kcomplex128, correspondendo a entrada float ou double respectivamente.
    :param use_dagger: se deve usar a versao conjugada transposta da porta, o padrao eh False.
    :return: uma instancia ``pyvqnet.qnn.vqc.torch.QModule``

    Exemplo::

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
    
    define a U1 quantum gate .

    Esta classe herda de ``pyvqnet.qnn.vqc.torch.QModule`` e ``torch.nn.Module``.
    Esta classe pode ser adicionada ao modelo torch como um submodulo de ``torch.nn.Module``.

    :param has_params: se possui parametros, como portas RX, RY e outras precisam ser definidas como True, e aquelas sem parametros precisam ser definidas como False, o padrao eh False.
    :param trainable: se possui parametros a serem treinados. Se a camada usa dados de entrada externos para construir a matriz da porta logica, defina como False. Se os parametros a serem treinados precisam ser inicializados a partir desta camada, eh True, o padrao eh False.
    :param init_params: Parametros de inicializacao usados para codificar dados classicos QTensor, o padrao eh None.
    :param wires: Indice do bit do efeito de linha, o padrao eh None.
    :param dtype: A precisao dos dados da matriz interna da porta logica pode ser definida como pyvqnet.kcomplex64 ou pyvqnet.kcomplex128, correspondendo a entrada float ou double respectivamente.
    :param use_dagger: se deve usar a versao conjugada transposta da porta, o padrao eh False.
    :return: uma instancia ``pyvqnet.qnn.vqc.torch.QModule``

    Exemplo::

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
    
    define a U2 quantum gate .

    Esta classe herda de ``pyvqnet.qnn.vqc.torch.QModule`` e ``torch.nn.Module``.
    Esta classe pode ser adicionada ao modelo torch como um submodulo de ``torch.nn.Module``.

    :param has_params: se possui parametros, como portas RX, RY e outras precisam ser definidas como True, e aquelas sem parametros precisam ser definidas como False, o padrao eh False.
    :param trainable: se possui parametros a serem treinados. Se a camada usa dados de entrada externos para construir a matriz da porta logica, defina como False. Se os parametros a serem treinados precisam ser inicializados a partir desta camada, eh True, o padrao eh False.
    :param init_params: Parametros de inicializacao usados para codificar dados classicos QTensor, o padrao eh None.
    :param wires: Indice do bit do efeito de linha, o padrao eh None.
    :param dtype: A precisao dos dados da matriz interna da porta logica pode ser definida como pyvqnet.kcomplex64 ou pyvqnet.kcomplex128, correspondendo a entrada float ou double respectivamente.
    :param use_dagger: se deve usar a versao conjugada transposta da porta, o padrao eh False.
    :return: uma instancia ``pyvqnet.qnn.vqc.torch.QModule``

    Exemplo::

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
    
    define a U3 quantum gate .

    Esta classe herda de ``pyvqnet.qnn.vqc.torch.QModule`` e ``torch.nn.Module``.
    Esta classe pode ser adicionada ao modelo torch como um submodulo de ``torch.nn.Module``.

    :param has_params: se possui parametros, como portas RX, RY e outras precisam ser definidas como True, e aquelas sem parametros precisam ser definidas como False, o padrao eh False.
    :param trainable: se possui parametros a serem treinados. Se a camada usa dados de entrada externos para construir a matriz da porta logica, defina como False. Se os parametros a serem treinados precisam ser inicializados a partir desta camada, eh True, o padrao eh False.
    :param init_params: Parametros de inicializacao usados para codificar dados classicos QTensor, o padrao eh None.
    :param wires: Indice do bit do efeito de linha, o padrao eh None.
    :param dtype: A precisao dos dados da matriz interna da porta logica pode ser definida como pyvqnet.kcomplex64 ou pyvqnet.kcomplex128, correspondendo a entrada float ou double respectivamente.
    :param use_dagger: se deve usar a versao conjugada transposta da porta, o padrao eh False.
    :return: uma instancia ``pyvqnet.qnn.vqc.torch.QModule``

    Exemplo::

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
    
    define a CNOT quantum gate , alias `CX` .

    Esta classe herda de ``pyvqnet.qnn.vqc.torch.QModule`` e ``torch.nn.Module``.
    Esta classe pode ser adicionada ao modelo torch como um submodulo de ``torch.nn.Module``.

    :param has_params: se possui parametros, como portas RX, RY e outras precisam ser definidas como True, e aquelas sem parametros precisam ser definidas como False, o padrao eh False.
    :param trainable: se possui parametros a serem treinados. Se a camada usa dados de entrada externos para construir a matriz da porta logica, defina como False. Se os parametros a serem treinados precisam ser inicializados a partir desta camada, eh True, o padrao eh False.
    :param init_params: Parametros de inicializacao usados para codificar dados classicos QTensor, o padrao eh None.
    :param wires: Indice do bit do efeito de linha, o padrao eh None.
    :param dtype: A precisao dos dados da matriz interna da porta logica pode ser definida como pyvqnet.kcomplex64 ou pyvqnet.kcomplex128, correspondendo a entrada float ou double respectivamente.
    :param use_dagger: se deve usar a versao conjugada transposta da porta, o padrao eh False.
    :return: uma instancia ``pyvqnet.qnn.vqc.torch.QModule``

    Exemplo::

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
    
    define a CY quantum gate .

    Esta classe herda de ``pyvqnet.qnn.vqc.torch.QModule`` e ``torch.nn.Module``.
    Esta classe pode ser adicionada ao modelo torch como um submodulo de ``torch.nn.Module``.

    :param has_params: se possui parametros, como portas RX, RY e outras precisam ser definidas como True, e aquelas sem parametros precisam ser definidas como False, o padrao eh False.
    :param trainable: se possui parametros a serem treinados. Se a camada usa dados de entrada externos para construir a matriz da porta logica, defina como False. Se os parametros a serem treinados precisam ser inicializados a partir desta camada, eh True, o padrao eh False.
    :param init_params: Parametros de inicializacao usados para codificar dados classicos QTensor, o padrao eh None.
    :param wires: Indice do bit do efeito de linha, o padrao eh None.
    :param dtype: A precisao dos dados da matriz interna da porta logica pode ser definida como pyvqnet.kcomplex64 ou pyvqnet.kcomplex128, correspondendo a entrada float ou double respectivamente.
    :param use_dagger: se deve usar a versao conjugada transposta da porta, o padrao eh False.
    :return: uma instancia ``pyvqnet.qnn.vqc.torch.QModule``

    Exemplo::

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
    
    define a CZ quantum gate .

    Esta classe herda de ``pyvqnet.qnn.vqc.torch.QModule`` e ``torch.nn.Module``.
    Esta classe pode ser adicionada ao modelo torch como um submodulo de ``torch.nn.Module``.

    :param has_params: se possui parametros, como portas RX, RY e outras precisam ser definidas como True, e aquelas sem parametros precisam ser definidas como False, o padrao eh False.
    :param trainable: se possui parametros a serem treinados. Se a camada usa dados de entrada externos para construir a matriz da porta logica, defina como False. Se os parametros a serem treinados precisam ser inicializados a partir desta camada, eh True, o padrao eh False.
    :param init_params: Parametros de inicializacao usados para codificar dados classicos QTensor, o padrao eh None.
    :param wires: Indice do bit do efeito de linha, o padrao eh None.
    :param dtype: A precisao dos dados da matriz interna da porta logica pode ser definida como pyvqnet.kcomplex64 ou pyvqnet.kcomplex128, correspondendo a entrada float ou double respectivamente.
    :param use_dagger: se deve usar a versao conjugada transposta da porta, o padrao eh False.
    :return: uma instancia ``pyvqnet.qnn.vqc.torch.QModule``

    Exemplo::

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
    
    define a CR quantum gate .

    Esta classe herda de ``pyvqnet.qnn.vqc.torch.QModule`` e ``torch.nn.Module``.
    Esta classe pode ser adicionada ao modelo torch como um submodulo de ``torch.nn.Module``.

    :param has_params: se possui parametros, como portas RX, RY e outras precisam ser definidas como True, e aquelas sem parametros precisam ser definidas como False, o padrao eh False.
    :param trainable: se possui parametros a serem treinados. Se a camada usa dados de entrada externos para construir a matriz da porta logica, defina como False. Se os parametros a serem treinados precisam ser inicializados a partir desta camada, eh True, o padrao eh False.
    :param init_params: Parametros de inicializacao usados para codificar dados classicos QTensor, o padrao eh None.
    :param wires: Indice do bit do efeito de linha, o padrao eh None.
    :param dtype: A precisao dos dados da matriz interna da porta logica pode ser definida como pyvqnet.kcomplex64 ou pyvqnet.kcomplex128, correspondendo a entrada float ou double respectivamente.
    :param use_dagger: se deve usar a versao conjugada transposta da porta, o padrao eh False.
    :return: uma instancia ``pyvqnet.qnn.vqc.torch.QModule``

    Exemplo::

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
    
    define a SWAP quantum gate .

    Esta classe herda de ``pyvqnet.qnn.vqc.torch.QModule`` e ``torch.nn.Module``.
    Esta classe pode ser adicionada ao modelo torch como um submodulo de ``torch.nn.Module``.

    :param has_params: se possui parametros, como portas RX, RY e outras precisam ser definidas como True, e aquelas sem parametros precisam ser definidas como False, o padrao eh False.
    :param trainable: se possui parametros a serem treinados. Se a camada usa dados de entrada externos para construir a matriz da porta logica, defina como False. Se os parametros a serem treinados precisam ser inicializados a partir desta camada, eh True, o padrao eh False.
    :param init_params: Parametros de inicializacao usados para codificar dados classicos QTensor, o padrao eh None.
    :param wires: Indice do bit do efeito de linha, o padrao eh None.
    :param dtype: A precisao dos dados da matriz interna da porta logica pode ser definida como pyvqnet.kcomplex64 ou pyvqnet.kcomplex128, correspondendo a entrada float ou double respectivamente.
    :param use_dagger: se deve usar a versao conjugada transposta da porta, o padrao eh False.
    :return: uma instancia ``pyvqnet.qnn.vqc.torch.QModule``

    Exemplo::

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
    
    define a SWAP quantum gate .

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

    Esta classe herda de ``pyvqnet.qnn.vqc.torch.QModule`` e ``torch.nn.Module``.
    Esta classe pode ser adicionada ao modelo torch como um submodulo de ``torch.nn.Module``.

    :param has_params: se possui parametros, como portas RX, RY e outras precisam ser definidas como True, e aquelas sem parametros precisam ser definidas como False, o padrao eh False.
    :param trainable: se possui parametros a serem treinados. Se a camada usa dados de entrada externos para construir a matriz da porta logica, defina como False. Se os parametros a serem treinados precisam ser inicializados a partir desta camada, eh True, o padrao eh False.
    :param init_params: Parametros de inicializacao usados para codificar dados classicos QTensor, o padrao eh None.
    :param wires: Indice do bit do efeito de linha, o padrao eh None.
    :param dtype: A precisao dos dados da matriz interna da porta logica pode ser definida como pyvqnet.kcomplex64 ou pyvqnet.kcomplex128, correspondendo a entrada float ou double respectivamente.
    :param use_dagger: se deve usar a versao conjugada transposta da porta, o padrao eh False.
    :return: uma instancia ``pyvqnet.qnn.vqc.torch.QModule``

    Exemplo::

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
    
    define a RXX quantum gate .

    Esta classe herda de ``pyvqnet.qnn.vqc.torch.QModule`` e ``torch.nn.Module``.
    Esta classe pode ser adicionada ao modelo torch como um submodulo de ``torch.nn.Module``.

    :param has_params: se possui parametros, como portas RX, RY e outras precisam ser definidas como True, e aquelas sem parametros precisam ser definidas como False, o padrao eh False.
    :param trainable: se possui parametros a serem treinados. Se a camada usa dados de entrada externos para construir a matriz da porta logica, defina como False. Se os parametros a serem treinados precisam ser inicializados a partir desta camada, eh True, o padrao eh False.
    :param init_params: Parametros de inicializacao usados para codificar dados classicos QTensor, o padrao eh None.
    :param wires: Indice do bit do efeito de linha, o padrao eh None.
    :param dtype: A precisao dos dados da matriz interna da porta logica pode ser definida como pyvqnet.kcomplex64 ou pyvqnet.kcomplex128, correspondendo a entrada float ou double respectivamente.
    :param use_dagger: se deve usar a versao conjugada transposta da porta, o padrao eh False.
    :return: uma instancia ``pyvqnet.qnn.vqc.torch.QModule``

    Exemplo::

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
    
    define a RYY quantum gate .

    Esta classe herda de ``pyvqnet.qnn.vqc.torch.QModule`` e ``torch.nn.Module``.
    Esta classe pode ser adicionada ao modelo torch como um submodulo de ``torch.nn.Module``.

    :param has_params: se possui parametros, como portas RX, RY e outras precisam ser definidas como True, e aquelas sem parametros precisam ser definidas como False, o padrao eh False.
    :param trainable: se possui parametros a serem treinados. Se a camada usa dados de entrada externos para construir a matriz da porta logica, defina como False. Se os parametros a serem treinados precisam ser inicializados a partir desta camada, eh True, o padrao eh False.
    :param init_params: Parametros de inicializacao usados para codificar dados classicos QTensor, o padrao eh None.
    :param wires: Indice do bit do efeito de linha, o padrao eh None.
    :param dtype: A precisao dos dados da matriz interna da porta logica pode ser definida como pyvqnet.kcomplex64 ou pyvqnet.kcomplex128, correspondendo a entrada float ou double respectivamente.
    :param use_dagger: se deve usar a versao conjugada transposta da porta, o padrao eh False.
    :return: uma instancia ``pyvqnet.qnn.vqc.torch.QModule``

    Exemplo::

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
    
    define a RZZ quantum gate .

    Esta classe herda de ``pyvqnet.qnn.vqc.torch.QModule`` e ``torch.nn.Module``.
    Esta classe pode ser adicionada ao modelo torch como um submodulo de ``torch.nn.Module``.

    :param has_params: se possui parametros, como portas RX, RY e outras precisam ser definidas como True, e aquelas sem parametros precisam ser definidas como False, o padrao eh False.
    :param trainable: se possui parametros a serem treinados. Se a camada usa dados de entrada externos para construir a matriz da porta logica, defina como False. Se os parametros a serem treinados precisam ser inicializados a partir desta camada, eh True, o padrao eh False.
    :param init_params: Parametros de inicializacao usados para codificar dados classicos QTensor, o padrao eh None.
    :param wires: Indice do bit do efeito de linha, o padrao eh None.
    :param dtype: A precisao dos dados da matriz interna da porta logica pode ser definida como pyvqnet.kcomplex64 ou pyvqnet.kcomplex128, correspondendo a entrada float ou double respectivamente.
    :param use_dagger: se deve usar a versao conjugada transposta da porta, o padrao eh False.
    :return: uma instancia ``pyvqnet.qnn.vqc.torch.QModule``

    Exemplo::

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
    
    define a RZX quantum gate .

    Esta classe herda de ``pyvqnet.qnn.vqc.torch.QModule`` e ``torch.nn.Module``.
    Esta classe pode ser adicionada ao modelo torch como um submodulo de ``torch.nn.Module``.

    :param has_params: se possui parametros, como portas RX, RY e outras precisam ser definidas como True, e aquelas sem parametros precisam ser definidas como False, o padrao eh False.
    :param trainable: se possui parametros a serem treinados. Se a camada usa dados de entrada externos para construir a matriz da porta logica, defina como False. Se os parametros a serem treinados precisam ser inicializados a partir desta camada, eh True, o padrao eh False.
    :param init_params: Parametros de inicializacao usados para codificar dados classicos QTensor, o padrao eh None.
    :param wires: Indice do bit do efeito de linha, o padrao eh None.
    :param dtype: A precisao dos dados da matriz interna da porta logica pode ser definida como pyvqnet.kcomplex64 ou pyvqnet.kcomplex128, correspondendo a entrada float ou double respectivamente.
    :param use_dagger: se deve usar a versao conjugada transposta da porta, o padrao eh False.
    :return: uma instancia ``pyvqnet.qnn.vqc.torch.QModule``

    Exemplo::

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
    
    define a Toffoli quantum gate .

    Esta classe herda de ``pyvqnet.qnn.vqc.torch.QModule`` e ``torch.nn.Module``.
    Esta classe pode ser adicionada ao modelo torch como um submodulo de ``torch.nn.Module``.

    :param has_params: se possui parametros, como portas RX, RY e outras precisam ser definidas como True, e aquelas sem parametros precisam ser definidas como False, o padrao eh False.
    :param trainable: se possui parametros a serem treinados. Se a camada usa dados de entrada externos para construir a matriz da porta logica, defina como False. Se os parametros a serem treinados precisam ser inicializados a partir desta camada, eh True, o padrao eh False.
    :param init_params: Parametros de inicializacao usados para codificar dados classicos QTensor, o padrao eh None.
    :param wires: Indice do bit do efeito de linha, o padrao eh None.
    :param dtype: A precisao dos dados da matriz interna da porta logica pode ser definida como pyvqnet.kcomplex64 ou pyvqnet.kcomplex128, correspondendo a entrada float ou double respectivamente.
    :param use_dagger: se deve usar a versao conjugada transposta da porta, o padrao eh False.
    :return: uma instancia ``pyvqnet.qnn.vqc.torch.QModule``

    Exemplo::

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
    
    define a IsingXX quantum gate .

    Esta classe herda de ``pyvqnet.qnn.vqc.torch.QModule`` e ``torch.nn.Module``.
    Esta classe pode ser adicionada ao modelo torch como um submodulo de ``torch.nn.Module``.

    :param has_params: se possui parametros, como portas RX, RY e outras precisam ser definidas como True, e aquelas sem parametros precisam ser definidas como False, o padrao eh False.
    :param trainable: se possui parametros a serem treinados. Se a camada usa dados de entrada externos para construir a matriz da porta logica, defina como False. Se os parametros a serem treinados precisam ser inicializados a partir desta camada, eh True, o padrao eh False.
    :param init_params: Parametros de inicializacao usados para codificar dados classicos QTensor, o padrao eh None.
    :param wires: Indice do bit do efeito de linha, o padrao eh None.
    :param dtype: A precisao dos dados da matriz interna da porta logica pode ser definida como pyvqnet.kcomplex64 ou pyvqnet.kcomplex128, correspondendo a entrada float ou double respectivamente.
    :param use_dagger: se deve usar a versao conjugada transposta da porta, o padrao eh False.
    :return: uma instancia ``pyvqnet.qnn.vqc.torch.QModule``

    Exemplo::

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
    
    define a IsingYY quantum gate .

    Esta classe herda de ``pyvqnet.qnn.vqc.torch.QModule`` e ``torch.nn.Module``.
    Esta classe pode ser adicionada ao modelo torch como um submodulo de ``torch.nn.Module``.

    :param has_params: se possui parametros, como portas RX, RY e outras precisam ser definidas como True, e aquelas sem parametros precisam ser definidas como False, o padrao eh False.
    :param trainable: se possui parametros a serem treinados. Se a camada usa dados de entrada externos para construir a matriz da porta logica, defina como False. Se os parametros a serem treinados precisam ser inicializados a partir desta camada, eh True, o padrao eh False.
    :param init_params: Parametros de inicializacao usados para codificar dados classicos QTensor, o padrao eh None.
    :param wires: Indice do bit do efeito de linha, o padrao eh None.
    :param dtype: A precisao dos dados da matriz interna da porta logica pode ser definida como pyvqnet.kcomplex64 ou pyvqnet.kcomplex128, correspondendo a entrada float ou double respectivamente.
    :param use_dagger: se deve usar a versao conjugada transposta da porta, o padrao eh False.
    :return: uma instancia ``pyvqnet.qnn.vqc.torch.QModule``

    Exemplo::

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
    
    define a IsingZZ quantum gate .

    Esta classe herda de ``pyvqnet.qnn.vqc.torch.QModule`` e ``torch.nn.Module``.
    Esta classe pode ser adicionada ao modelo torch como um submodulo de ``torch.nn.Module``.

    :param has_params: se possui parametros, como portas RX, RY e outras precisam ser definidas como True, e aquelas sem parametros precisam ser definidas como False, o padrao eh False.
    :param trainable: se possui parametros a serem treinados. Se a camada usa dados de entrada externos para construir a matriz da porta logica, defina como False. Se os parametros a serem treinados precisam ser inicializados a partir desta camada, eh True, o padrao eh False.
    :param init_params: Parametros de inicializacao usados para codificar dados classicos QTensor, o padrao eh None.
    :param wires: Indice do bit do efeito de linha, o padrao eh None.
    :param dtype: A precisao dos dados da matriz interna da porta logica pode ser definida como pyvqnet.kcomplex64 ou pyvqnet.kcomplex128, correspondendo a entrada float ou double respectivamente.
    :param use_dagger: se deve usar a versao conjugada transposta da porta, o padrao eh False.
    :return: uma instancia ``pyvqnet.qnn.vqc.torch.QModule``

    Exemplo::

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
    
    define a IsingXY quantum gate .

    Esta classe herda de ``pyvqnet.qnn.vqc.torch.QModule`` e ``torch.nn.Module``.
    Esta classe pode ser adicionada ao modelo torch como um submodulo de ``torch.nn.Module``.

    :param has_params: se possui parametros, como portas RX, RY e outras precisam ser definidas como True, e aquelas sem parametros precisam ser definidas como False, o padrao eh False.
    :param trainable: se possui parametros a serem treinados. Se a camada usa dados de entrada externos para construir a matriz da porta logica, defina como False. Se os parametros a serem treinados precisam ser inicializados a partir desta camada, eh True, o padrao eh False.
    :param init_params: Parametros de inicializacao usados para codificar dados classicos QTensor, o padrao eh None.
    :param wires: Indice do bit do efeito de linha, o padrao eh None.
    :param dtype: A precisao dos dados da matriz interna da porta logica pode ser definida como pyvqnet.kcomplex64 ou pyvqnet.kcomplex128, correspondendo a entrada float ou double respectivamente.
    :param use_dagger: se deve usar a versao conjugada transposta da porta, o padrao eh False.
    :return: uma instancia ``pyvqnet.qnn.vqc.torch.QModule``

    Exemplo::

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
    
    define a PhaseShift quantum gate .

    Esta classe herda de ``pyvqnet.qnn.vqc.torch.QModule`` e ``torch.nn.Module``.
    Esta classe pode ser adicionada ao modelo torch como um submodulo de ``torch.nn.Module``.

    :param has_params: se possui parametros, como portas RX, RY e outras precisam ser definidas como True, e aquelas sem parametros precisam ser definidas como False, o padrao eh False.
    :param trainable: se possui parametros a serem treinados. Se a camada usa dados de entrada externos para construir a matriz da porta logica, defina como False. Se os parametros a serem treinados precisam ser inicializados a partir desta camada, eh True, o padrao eh False.
    :param init_params: Parametros de inicializacao usados para codificar dados classicos QTensor, o padrao eh None.
    :param wires: Indice do bit do efeito de linha, o padrao eh None.
    :param dtype: A precisao dos dados da matriz interna da porta logica pode ser definida como pyvqnet.kcomplex64 ou pyvqnet.kcomplex128, correspondendo a entrada float ou double respectivamente.
    :param use_dagger: se deve usar a versao conjugada transposta da porta, o padrao eh False.
    :return: uma instancia ``pyvqnet.qnn.vqc.torch.QModule``

    Exemplo::

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
    
    define a MultiRZ quantum gate .

    Esta classe herda de ``pyvqnet.qnn.vqc.torch.QModule`` e ``torch.nn.Module``.
    Esta classe pode ser adicionada ao modelo torch como um submodulo de ``torch.nn.Module``.

    :param has_params: se possui parametros, como portas RX, RY e outras precisam ser definidas como True, e aquelas sem parametros precisam ser definidas como False, o padrao eh False.
    :param trainable: se possui parametros a serem treinados. Se a camada usa dados de entrada externos para construir a matriz da porta logica, defina como False. Se os parametros a serem treinados precisam ser inicializados a partir desta camada, eh True, o padrao eh False.
    :param init_params: Parametros de inicializacao usados para codificar dados classicos QTensor, o padrao eh None.
    :param wires: Indice do bit do efeito de linha, o padrao eh None.
    :param dtype: A precisao dos dados da matriz interna da porta logica pode ser definida como pyvqnet.kcomplex64 ou pyvqnet.kcomplex128, correspondendo a entrada float ou double respectivamente.
    :param use_dagger: se deve usar a versao conjugada transposta da porta, o padrao eh False.
    :return: uma instancia ``pyvqnet.qnn.vqc.torch.QModule``

    Exemplo::

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
    
    define a SDG quantum gate .

    Esta classe herda de ``pyvqnet.qnn.vqc.torch.QModule`` e ``torch.nn.Module``.
    Esta classe pode ser adicionada ao modelo torch como um submodulo de ``torch.nn.Module``.

    :param has_params: se possui parametros, como portas RX, RY e outras precisam ser definidas como True, e aquelas sem parametros precisam ser definidas como False, o padrao eh False.
    :param trainable: se possui parametros a serem treinados. Se a camada usa dados de entrada externos para construir a matriz da porta logica, defina como False. Se os parametros a serem treinados precisam ser inicializados a partir desta camada, eh True, o padrao eh False.
    :param init_params: Parametros de inicializacao usados para codificar dados classicos QTensor, o padrao eh None.
    :param wires: Indice do bit do efeito de linha, o padrao eh None.
    :param dtype: A precisao dos dados da matriz interna da porta logica pode ser definida como pyvqnet.kcomplex64 ou pyvqnet.kcomplex128, correspondendo a entrada float ou double respectivamente.
    :param use_dagger: se deve usar a versao conjugada transposta da porta, o padrao eh False.
    :return: uma instancia ``pyvqnet.qnn.vqc.torch.QModule``

    Exemplo::
        
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
    
    define a SDG quantum gate .

    Esta classe herda de ``pyvqnet.qnn.vqc.torch.QModule`` e ``torch.nn.Module``.
    Esta classe pode ser adicionada ao modelo torch como um submodulo de ``torch.nn.Module``.

    :param has_params: se possui parametros, como portas RX, RY e outras precisam ser definidas como True, e aquelas sem parametros precisam ser definidas como False, o padrao eh False.
    :param trainable: se possui parametros a serem treinados. Se a camada usa dados de entrada externos para construir a matriz da porta logica, defina como False. Se os parametros a serem treinados precisam ser inicializados a partir desta camada, eh True, o padrao eh False.
    :param init_params: Parametros de inicializacao usados para codificar dados classicos QTensor, o padrao eh None.
    :param wires: Indice do bit do efeito de linha, o padrao eh None.
    :param dtype: A precisao dos dados da matriz interna da porta logica pode ser definida como pyvqnet.kcomplex64 ou pyvqnet.kcomplex128, correspondendo a entrada float ou double respectivamente.
    :param use_dagger: se deve usar a versao conjugada transposta da porta, o padrao eh False.
    :return: uma instancia ``pyvqnet.qnn.vqc.torch.QModule``

    Exemplo::
        
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
    
    define a ControlledPhaseShift quantum gate .

    Esta classe herda de ``pyvqnet.qnn.vqc.torch.QModule`` e ``torch.nn.Module``.
    Esta classe pode ser adicionada ao modelo torch como um submodulo de ``torch.nn.Module``.

    :param has_params: se possui parametros, como portas RX, RY e outras precisam ser definidas como True, e aquelas sem parametros precisam ser definidas como False, o padrao eh False.
    :param trainable: se possui parametros a serem treinados. Se a camada usa dados de entrada externos para construir a matriz da porta logica, defina como False. Se os parametros a serem treinados precisam ser inicializados a partir desta camada, eh True, o padrao eh False.
    :param init_params: Parametros de inicializacao usados para codificar dados classicos QTensor, o padrao eh None.
    :param wires: Indice do bit do efeito de linha, o padrao eh None.
    :param dtype: A precisao dos dados da matriz interna da porta logica pode ser definida como pyvqnet.kcomplex64 ou pyvqnet.kcomplex128, correspondendo a entrada float ou double respectivamente.
    :param use_dagger: se deve usar a versao conjugada transposta da porta, o padrao eh False.
    :return: uma instancia ``pyvqnet.qnn.vqc.torch.QModule``

    Exemplo::

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
    
    define a MultiControlledX quantum gate .

    Esta classe herda de ``pyvqnet.qnn.vqc.torch.QModule`` e ``torch.nn.Module``.
    Esta classe pode ser adicionada ao modelo torch como um submodulo de ``torch.nn.Module``.
    
    :param has_params: se possui parametros, como portas RX, RY e outras precisam ser definidas como True, e aquelas sem parametros precisam ser definidas como False, o padrao eh False.
    :param trainable: se possui parametros a serem treinados. Se a camada usa dados de entrada externos para construir a matriz da porta logica, defina como False. Se os parametros a serem treinados precisam ser inicializados a partir desta camada, eh True, o padrao eh False.
    :param init_params: Parametros de inicializacao usados para codificar dados classicos QTensor, o padrao eh None.
    :param wires: Indice do bit do efeito de linha, o padrao eh None.
    :param dtype: A precisao dos dados da matriz interna da porta logica pode ser definida como pyvqnet.kcomplex64 ou pyvqnet.kcomplex128, correspondendo a entrada float ou double respectivamente.
    :param use_dagger: se deve usar a versao conjugada transposta da porta, o padrao eh False.
    :param control_values: Valor de controle, o padrao eh None, quando o bit eh 1, eh controlado.

    :return: uma instancia ``pyvqnet.qnn.vqc.torch.QModule``

    Exemplo::

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


API de Medicoes
^^^^^^^^^^^^^^^^^^^^^^

Probability
"""""""""""""""""""""""""""""


.. py:class:: pyvqnet.qnn.vqc.torch.Probability(wires=None, name="")

    Calcula o resultado da medicao de probabilidade do circuito quantico em um bit especifico.

    Esta classe herda de ``pyvqnet.qnn.vqc.torch.QModule`` e ``torch.nn.Module``.
    Esta classe pode ser adicionada ao modelo torch como um submodulo de ``torch.nn.Module``.

    :param wires: O indice do bit de medicao, lista, tupla ou inteiro.
    :param name: O nome do modulo, padrao: "".
    :return: O resultado da medicao, QTensor.

    Exemplo::

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

    Calcula os resultados de medicao de circuitos quanticos, suporta entrada obs como multiplos ou unicos operadores Pauli ou Hamiltonianos.
    For example:

    {\'X0\': 0.23} indicates a PauliX effect on qubit 0, with a coefficient of 0.23.

    {\'X1 Z2\': 2.4,\'Y2\': -0.5} corresponds to the observed value 2.4 * X1 @ Z2 - 0.5 * Y2.

    [{\'X1 Z2\': 4,\'Z1 Z0\': 3},{\'X1 Y2 Z0\': 3.5}] corresponds to the two observed values 4 * X1 @ Z2 + 3 * Z1 @ Z0 and 3.5 * X1 @ Y2 @ Z0.
        
        
    Esta classe herda de ``pyvqnet.qnn.vqc.torch.QModule`` e ``torch.nn.Module``.

    Esta classe pode ser adicionada ao modelo torch como um submodulo de ``torch.nn.Module``.

    :param obs: observable.
    :param name: module name, default: "".
    :return: measurement result, QTensor.

    Exemplo::

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

    Get sample results with shot on  specific wires.

    Esta classe herda de ``pyvqnet.qnn.vqc.torch.QModule`` e ``torch.nn.Module``.
    Esta classe pode ser adicionada ao modelo torch como um submodulo de ``torch.nn.Module``.


    :param wires: Indice do qubit de amostragem. Valor padrao: None, usa todos os bits do simulador em tempo de execucao.
    :param obs: Este valor so pode ser None.
    :param shots: Sample repetition count, default value: 1.
    :param name: O nome deste modulo, valor padrao: "".
    :return: a measurement method class

    Exemplo::

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

    Compute the expectation of a Hermitian quantity in a quantum circuit.

    Esta classe herda de ``pyvqnet.qnn.vqc.torch.QModule`` e ``torch.nn.Module``.
    Esta classe pode ser adicionada ao modelo torch como um submodulo de ``torch.nn.Module``.


    :param obs: Hermitian quantity.
    :param name: module name, default: "".
    :return: expected result, QTensor.

    Exemplo::

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

Modelos comuns para circuitos quanticos
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

VQC_HardwareEfficientAnsatz
""""""""""""""""""""""""""""""""""""""""


.. py:class:: pyvqnet.qnn.vqc.torch.VQC_HardwareEfficientAnsatz(n_qubits,single_rot_gate_list,entangle_gate="CNOT",entangle_rules='linear',depth=1,initial=None,dtype=None)

    Implementation of Hardware Efficient Ansatz introduced in the paper: `Hardware-efficient Variational Quantum Eigensolver for Small Molecules <https://arxiv.org/pdf/1704.05018.pdf>`__ .

    Esta classe herda de ``pyvqnet.qnn.vqc.torch.QModule`` e ``torch.nn.Module``.

    Esta classe pode ser adicionada ao modelo torch como um submodulo de ``torch.nn.Module``.


    :param n_qubits: Number of qubits.
    :param single_rot_gate_list: Uma lista de portas de rotacao de unico qubit eh construida por uma ou varias portas de rotacao que atuam em cada qubit. Atualmente suporta Rx, Ry, Rz.
    :param entangle_gate: A porta de entrelacamento nao parametrizada. CNOT, CZ sao suportados. Padrao: CNOT.
    :param entangle_rules: Como a porta de entrelacamento eh usada no circuito. 'linear' significa que a porta de entrelacamento atuara em cada qubit vizinho. 'all' significa que a porta de entrelacamento atuara em quaisquer dois qubits. Padrao: linear.
    :param depth: A profundidade do ansatz, padrao: 1.
    :param initial: valor inicial unico para parametros, padrao: None, este modulo inicializara parametros aleatoriamente.
    :param dtype: data dtype of parameters.
    :return: a VQC_HardwareEfficientAnsatz instance.

    Exemplo::

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

    A layer consisting of a single-parameter single-qubit rotation on each qubit, followed by multiple CNOT gates in a closed chain or ring combination.

    A ring of CNOT gates connects each qubit to its neighbors, and finally the a qubit is considered to be the neighbor of the a th qubit.

    Esta classe herda de ``pyvqnet.qnn.vqc.torch.QModule`` e ``torch.nn.Module``.

    Esta classe pode ser adicionada ao modelo torch como um submodulo de ``torch.nn.Module``.

    :param num_layers: number of repeat layers, default: 1.
    :param num_qubits: number of qubits, default: 1.
    :param rotation: one-parameter single-qubit gate to use, default: `RX`
    :param initial: valor inicial igual para todos os parametros. padrao: None, os parametros serao inicializados aleatoriamente.
    :param dtype: data type of parameter, default:None,use float32.
    :return: A VQC_BasicEntanglerTemplate instance

    Exemplo::

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

    Layers consisting of single qubit rotations and entanglers, as in `circuit-centric classifier design <https://arxiv.org/abs/1804.00633>`__ .

    Esta classe herda de ``pyvqnet.qnn.vqc.torch.QModule`` e ``torch.nn.Module``.
    Esta classe pode ser adicionada ao modelo torch como um submodulo de ``torch.nn.Module``.


    :param num_layers: number of repeat layers, default: 1.
    :param num_qubits: number of qubits, default: 1.
    :param ranges: sequencia determinando o hiperparametro de intervalo para cada camada subsequente; padrao: None
                                using :math: `r=l \mod M` for the :math:`l` th layer and :math:`M` qubits.
    :param initial: initial value for all parameters.default: None,initialized randomly.
    :param dtype: data type of parameter, default:None,use float32.
    :return: A VQC_StronglyEntanglingTemplate instance.

    Exemplo::

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
    
    Use RZ,RY,RZ to create variational quantum circuits to encode classical data into quantum states.
    Reference `Quantum embeddings for machine learning <https://arxiv.org/abs/2001.03622>`_.

    Esta classe herda de ``pyvqnet.qnn.vqc.torch.QModule`` e ``torch.nn.Module``.
    Esta classe pode ser adicionada ao modelo torch como um submodulo de ``torch.nn.Module``.

 
    :param num_repetitions_input: number of repeat times to encode input in a submodule.
    :param depth_input: number of input dimension .
    :param num_unitary_layers: number of repeat times of variational quantum gates.
    :param num_repetitions: number of repeat times of submodule.
    :param initial: parameter initialization value, default is None
    :param dtype: parameter type, default is None, use float32.
    :param name: class name
    :return: A VQC_QuantumEmbedding instance.

    Exemplo::

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

    19 ansatz diferentes do artigo `Expressibility and entangling capability of parameterized quantum circuits for hybrid quantum-classical algorithms <https://arxiv.org/pdf/1905.10876.pdf>`_.

    Esta classe herda de ``pyvqnet.qnn.vqc.torch.QModule`` e ``torch.nn.Module``.

    Esta classe pode ser adicionada ao modelo torch como um submodulo de ``torch.nn.Module``.

    :param type: Tipo de circuito de 1 a 19, um total de 19 linhas.
    :param num_wires: Number of qubits.
    :param depth: Circuit depth.
    :param dtype: data type of parameter, default:None,use float32.
    :param name: Name, default "".

    :return:
        a ExpressiveEntanglingAnsatz instance

    Exemplo::

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

    Codifica n caracteristicas binarias no estado base de n-qubits de ``q_machine``. Esta funcao tem o alias `VQC_BasisEmbedding`.

    For example, for ``basis_state=([0, 1, 1])``, the basis state in the quantum system is :math:`|011 \rangle`.

    :param basis_state: ``(n)`` size binary input.
    :param q_machine: quantum machine device.
    

    Exemplo::

        import pyvqnet
        pyvqnet.backends.set_backend("torch")
        from pyvqnet.qnn.vqc.torch import vqc_basis_embedding,QMachine
        qm  = QMachine(3)
        vqc_basis_embedding(basis_state=[1,1,0],q_machine=qm)
        print(qm.states)




vqc_angle_embedding
""""""""""""""""""""""""""""""""""""""""


.. py:function:: pyvqnet.qnn.vqc.torch.vqc_angle_embedding(input_feat, wires, q_machine: pyvqnet.qnn.vqc.torch.QMachine, rotation: str = "X")

    Encodes :math:`N` features into the rotation angle of :math:`n` qubits, where :math:`N \leq n`.
    Esta funcao tem o alias `VQC_AngleEmbedding` .

    The rotation can be selected as: 'X' , 'Y' , 'Z', as defined by the ``rotation`` parameter:

    * ``rotation='X'`` Use the feature as the angle of RX rotation.

    * ``rotation='Y'`` Use the feature as the angle of RY rotation.

    * ``rotation='Z'`` Use the feature as the angle of RZ rotation.

    ``wires`` represents the idx of the rotation gate on the qubit.

    :param input_feat: Array representing parameters.
    :param wires: Qubit idx.
    :param q_machine: Quantum machine device.
    :param rotation: Rotation gate, default is "X".

    Exemplo::

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

    Codifica uma caracteristica :math:`2^n` em um vetor de amplitude de :math:`n` qubits. Esta funcao tem o alias `VQC_AmplitudeEmbedding`.

    :param input_feature: array numpy representando o parametro.
    :param q_machine: quantum machine device.
    

    Exemplo::

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
    :param rep: Numero de vezes para repetir o bloco de circuito quantico, padrao eh 1.

    Exemplo::

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

    Combinacao arbitraria de porta logica quantica de rotacao de unico bit quantico. Esta funcao tem o alias: ``VQC_RotCircuit`` .

    .. math::
        R(\phi,\theta,\omega) = RZ(\omega)RY(\theta)RZ(\phi)= \begin{bmatrix}
        e^{-i(\phi+\omega)/2}\cos(\theta/2) & -e^{i(\phi-\omega)/2}\sin(\theta/2) \\
        e^{-i(\phi-\omega)/2}\sin(\theta/2) & e^{i(\phi+\omega)/2}\cos(\theta/2)
        \end{bmatrix}.

    :param q_machine: quantum virtual machine device.
    :param wire: quantum bit index.
    :param params: represents parameters :math:`[\phi, \theta, \omega]`.

    Exemplo::

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

    Combinacao de porta logica quantica de rotacao controlada de unico bit quantico. Esta funcao tem o alias: ``VQC_CRotCircuit`` .

    .. math:: 
        CR(\phi, \theta, \omega) = \begin{bmatrix}
        1 & 0 & 0 & 0 \\
        0 & 1 & 0 & 0\\
        0 & 0 & e^{-i(\phi+\omega)/2}\cos(\theta/2) & -e^{i(\phi-\omega)/2}\sin(\theta/2)\\
        0 & 0 & e^{-i(\phi-\omega)/2}\sin(\theta/2) & e^{i(\phi+\omega)/2}\cos(\theta/2)
        \end{bmatrix}.

    :param para: representa o array de parametros.
    :param control_qubits: Control qubit index.
    :param rot_wire: Rot qubit index.
    :param q_machine: Quantum machine device.
    

    Exemplo::

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

    Circuito quantico de porta logica Hadamard controlada. Esta funcao tem o alias: ``VQC_Controlled_Hadamard`` .

    .. math:: 
        CH = \begin{bmatrix}
        1 & 0 & 0 & 0 \\
        0 & 1 & 0 & 0 \\
        0 & 0 & \frac{1}{\sqrt{2}} & \frac{1}{\sqrt{2}} \\
        0 & 0 & \frac{1}{\sqrt{2}} & -\frac{1}{\sqrt{2}}
        \end{bmatrix}.

    :param wires: lista de indices de bits quanticos, o primeiro eh o bit de controle, o comprimento da lista eh 2.
    :param q_machine: quantum virtual machine device.

    Exemplos::

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

    :param wires: lista de indices de bits quanticos, o primeiro eh o bit de controle. O comprimento da lista eh 3.
    :param q_machine: quantum virtual machine device.

    Exemplo::

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
    :param wires: Um subconjunto de indices de qubit no intervalo [r, p]. O comprimento minimo deve ser 2. O primeiro valor de indice eh interpretado como r, e o ultimo valor de indice eh interpretado como p. Os indices intermediarios sao afetados por portas CNOT para calcular a paridade do conjunto de qubits.
    :param q_machine: Quantum virtual machine device.

    

    Exemplos::

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

    Esta funcao tem o alias: ``VQC_FermionicDoubleExcitation`` .

    :param weight: variable parameter
    :param wires1: representa o subconjunto de qubits no intervalo de lista de indices [s, r]. O indice a eh interpretado como s e o ultimo indice eh interpretado como r. A porta CNOT opera nos indices do meio para calcular a paridade de um grupo de qubits.
    :param wires2: representa o subconjunto de qubits no intervalo de lista de indices [q, p]. O primeiro indice raiz eh interpretado como q e o ultimo indice eh interpretado como p. A porta CNOT opera nos indices do meio para calcular a paridade de um grupo de qubits.
    :param q_machine: Quantum virtual machine device.

    

    Exemplos::

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

    Dentro da aproximacao de Trotter de primeira ordem, a funcao unitaria UCCSD eh dada por:

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

    Esta funcao tem o alias: ``VQC_UCCSD`` .

    :param weights: tensor of size ``(len(s_wires)+ len(d_wires))`` containing the parameters :math:`\theta_{pr}` and :math:`\theta_{pqrs}` input Z rotations ``FermionicSingleExcitation`` and ``FermionicDoubleExcitation`` .
    :param wires: qubit indices for template action
    :param s_wires: sequence of lists containing qubit indices ``[r,...,p]`` generated by a single excitation :math:`\vert r, p \rangle = \hat{c}_p^\dagger \hat{c}_r \vert \mathrm{HF} \rangle`,where :math:`\vert \mathrm{HF} \rangle` denotes the Hartee-Fock reference state.
    :param d_wires: sequence of lists, each containing two lists specifying indices ``[s, ...,r]`` and ``[q,..., p]`` defining double excitation :math:`\vert s, r, q, p \rangle = \hat{c}_p^\dagger \hat{c}_q^\dagger \hat{c}_r\hat{c}_s \vert \mathrm{HF} \rangle` .
    :param init_state: occupation-number vector of length ``len(wires)`` representing the high-frequency state. ``init_state`` Initialization state of the qubit.
    :param q_machine: Quantum virtual machine device.

    Exemplos::

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

    A string de Pauli eh fixada em ``Z``. Portanto, a expansao de primeira ordem sera um circuito sem portas de entrelacamento.

    :param input_feat: Array representing input parameters.
    :param q_machine: Quantum virtual machine.
    :param data_map_func: Parameter mapping matrix, a callable function, designed as: ``data_map_func = lambda x: x``.
    :param rep: Numero de vezes que o modulo eh repetido.

    Exemplo::

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
    
    Exemplo::

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

    Neste caso, temos quatro excitacoes simples e duplas para preservar a projecao de spin total do estado de Hartree-Fock.

    The resulting unitary matrix preserves the particle population and prepares the n-qubit system in a superposition of the initial Hartree-Fock state and other states encoding the multi-excitation configuration.

    :param weights: Um QTensor de tamanho ``(len(singles) + len(doubles),)`` contendo os angulos que entram nas operacoes vqc.qCircuit.single_excitation e vqc.qCircuit.double_excitation em sequencia
    :param q_machine: A maquina quantica.
    :param hf_state: Um vetor de comprimento ``len(wires)`` de numeros de ocupacao representando o estado de Hartree-Fock, ``hf_state`` usado para inicializar os fios.
    :param wires: Os qubits sobre os quais atuar.
    :param singles: Uma sequencia de listas com os indices dos dois qubits sobre os quais a operacao single_excitation atua.
    :param doubles: Sequencia de lista com os indices dos dois qubits sobre os quais a operacao double_excitation atua.

    For example, the quantum circuit for two electrons and six qubits is shown below:

    .. image:: ./images/all_singles_doubles.png
        :width: 600 px
        :align: center

    |

    Exemplo::

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

    Implementa um circuito que fornece um conjunto que pode ser usado para realizar rotacoes precisas de base de unidade unica. O circuito eh derivado da transformacao unitaria fermionica de unica particula :math:`U(u)` dada em `arXiv:1711.04789 <https://arxiv.org/abs/1711.04789>`_\
    
    .. math::
        U(u) = \exp{\left( \sum_{pq} \left[\log u \right]_{pq} (a_p^\dagger a_q - a_q^\dagger a_p) \right)}.

    :math:`U(u)` is obtained by using the scheme given in the paper `Optica, 3, 1460 (2016) <https://opg.optica.org/optica/fulltext.cfm?uri=optica-3-12-1460&id=355743>`_\ .

    :param q_machine: quantum machine.
    :param wires: qubits to act on.
    :param unitary_matrix: matriz especificando a base para a transformacao.
    :param check: check if `unitary_matrix` is a unitary matrix.

    Exemplo::

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

    Para reduzir o numero de qubits no circuito, pares de qubits sao primeiro criados no sistema. Apos emparelhar inicialmente todos os qubits, uma unitaria generalizada de 2 qubits eh aplicada a cada par de qubits. E apos aplicar essas unitarias de dois qubits, um qubit em cada par de qubits eh ignorado para o resto da rede neural.

    :param sources_wires: Indices de qubits de origem que serao ignorados.
    :param sinks_wires: Indices de qubits de destino que serao mantidos.
    :param params: Input parameters.
    :param q_machine: Quantum virtual machine device.

    Exemplos:: 

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


    Uma camada QuantumLayer automaticamente diferenciavel que usa a abordagem de matriz adjunta para calcular gradientes, veja `Efficient calculation of gradients in classical simulations of variational quantum algorithms <https://arxiv.org/abs/2009.02823>`_ .

    :param general_module: uma instancia `pyvqnet.nn.Module` construida usando apenas a interface de circuito quantico sob ``pyvqnet.qnn.vqc.torch``.
    :param use_qpanda: Whether to use qpanda line for forward transmission, default: False.
    :param name: O nome da camada, padrao "".

    .. note::

        O QMachine do general_module deve definir grad_method = "adjoint".

        Currently supports the following parameterized logic gates `RX`, `RY`, `RZ`, `PhaseShift`, `RXX`, `RYY`, `RZZ`, `RZX`, `U1`, `U2`, `U3` and other variational circuits consisting of non-parameter logic gates.


    Exemplo::

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





Modulo de Circuito Quantico Variacional com Backend Tensor Network
==========================================================================================

Tensor Network (TN) significantly reduces computational complexity by decomposing a complex tensor into a network of multiple low-dimensional tensors.

Matrix Product State (MPS) is a special form of Tensor Network. MPS represents a quantum state as the product of a series of matrices, thus effectively reducing the number of parameters and the computational complexity.

A interface a seguir eh baseada no backend ``torch``, que fornece suporte funcional para construir circuitos quanticos em redes tensorais, incluindo a construcao de classes base de circuito quantico, portas logicas quanticas, circuitos quanticos e medicoes, alem de calcular gradientes de parametros por simulacao diferencial automatica em vez do metodo de desvio de parametro.

Constructing quantum lines in the MPS way makes up for the support for large-bit quantum line construction.

.. warning::

        Usar os recursos a seguir neste modulo requer instalacao adicional de ``tensornetwork`` e ``torch``. A instalacao padrao do ``pyvqnet`` nao inclui essas duas dependencias. Instale-as usando ``pip install tensornetwork torch``.

.. warning::

        Habilita MPS para construir linhas quanticas via o parametro ``use_mps`` em ``TNQMachine``, que suporta implementacoes de linhas quanticas de grandes bits (100 e acima).

.. warning::
        
        Batching is used differently than under classic modules, based on the vmap approach, where the data and parameter construction lines need to be entered in one dimension down, as shown in the sample interface below, and the batching execution must be based on both ``TNQMachine`` and ``TNQModule``.

Classe Base
------------------------------------------------

Escrever um modelo de circuito quantico variacional em tensornetwork requer herdar de ``TNQModule``.

TNQModule
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. py:class:: pyvqnet.qnn.vqc.tn.TNQModule(use_jit=False, vectorized_argnums=0, name="")

    .. note::

        Esta classe e suas classes derivadas sao aplicaveis apenas a ``pyvqnet.backends.set_backend("torch")``, nao misture com o ``Module`` sob o ``pyvqnet.nn`` padrao.

        Os dados em ``_buffers`` desta classe sao do tipo ``torch.Tensor``.

        Os dados em ``_parmeters`` desta classe sao do tipo ``torch.nn.Parameter``.

    :param use_jit: control quantum circuit jit compilation functionality.
    :param vectorized_argnums: os argumentos a serem vetorizados,
            estes argumentos devem compartilhar a mesma forma de lote na primeira dimensao, padrao 0.
    :param name: name of Module.

    Exemplo::

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

    Classe simuladora para computacao quantica variacional, incluindo vetores de estado cujo atributo states sao circuitos quanticos.

    Esta classe herda de ``pyvqnet.nn.torch.TorchModule`` e ``pyvqnet.qnn.QMachine``.

    Esta classe pode ser adicionada ao modelo torch como um submodulo de ``torch.nn.Module``.

    .. warning::
        
        No circuito quantico da rede tensorial, a funcao ``vmap`` sera habilitada por padrao, e a dimensao do lote sera descartada nos parametros da porta logica na linha.
        Ao usar o parametro de chamada, se a dimensao for [batch_size, \*], a primeira dimensao batch_size eh descartada, e as dimensoes seguintes sao usadas diretamente, por exemplo, para os dados de entrada x[:,1] -> x[1], e para o parametro treinavel tambem, veja o exemplo a seguir para o uso de xx, weights.

    .. note::

        Before each run of the complete quantum circuit, you must use `pyvqnet.qnn.vqc.QMachine.reset_states(batchsize)` to reinitialize the initial state in the simulator and broadcast it to
        dimensoes (batchsize,*) para se adaptar ao treinamento de dados em lote.

    :param num_wires: number of qubits to use
    :param dtype: internal data type used to calculate.
    :param use_mps: open MPSCircuit for large bit models.

    :return: Output a TNQMachine object.

    Exemplo::
        
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

Modulo de porta logica quantica variacional
------------------------------------------------

The following function interfaces in ``pyvqnet.qnn.vqc`` directly support ``QTensor`` of ``torch`` backend for calculation, import path ``pyvqnet.qnn.vqc.tn``.

.. csv-table:: List of supported pyvqnet.qnn.vqc interfaces
    :file: ./images/same_apis_from_tn.csv

Os seguintes modulos de circuito quantico herdam de ``pyvqnet.qnn.vqc.tn.TNQModule``, onde os calculos sao realizados usando ``torch.Tensor``.

.. note::

    This class and its derived classes are only applicable to ``pyvqnet.backends.set_backend("torch")``, do not mix with ``Module`` under the default ``pyvqnet.nn``.

    Se estas classes tiverem variaveis membro nao parametro ``_buffers``, os dados nelas sao do tipo ``torch.Tensor``.
    Se estas classes tiverem variaveis membro de parametro ``_parmeters``, os dados nelas sao do tipo ``torch.nn.Parameter``.

I
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. py:class:: pyvqnet.qnn.vqc.tn.I(has_params: bool = False,trainable: bool = False,init_params=None,wires=None,dtype=pyvqnet.kcomplex64,use_dagger=False)
    
    define a I quantum gate .

    Esta classe herda de ``pyvqnet.qnn.vqc.tn.QModule`` e ``torch.nn.Module``.
    Esta classe pode ser adicionada ao modelo torch como um submodulo de ``torch.nn.Module``.

    :param has_params: se possui parametros, como portas RX, RY e outras precisam ser definidas como True, e aquelas sem parametros precisam ser definidas como False, o padrao eh False.
    :param trainable: se possui parametros a serem treinados. Se a camada usa dados de entrada externos para construir a matriz da porta logica, defina como False. Se os parametros a serem treinados precisam ser inicializados a partir desta camada, eh True, o padrao eh False.
    :param init_params: Parametros de inicializacao usados para codificar dados classicos QTensor, o padrao eh None.
    :param wires: Indice do bit do efeito de linha, o padrao eh None.
    :param dtype: A precisao dos dados da matriz interna da porta logica pode ser definida como pyvqnet.kcomplex64 ou pyvqnet.kcomplex128, correspondendo a entrada float ou double respectivamente.
    :param use_dagger: se deve usar a versao conjugada transposta da porta, o padrao eh False.
    :return: a ``pyvqnet.qnn.vqc.tn.QModule`` instance

    Exemplo::
        
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
    
    define a Hadamard quantum gate .

    Esta classe herda de ``pyvqnet.qnn.vqc.tn.QModule`` e ``torch.nn.Module``.
    Esta classe pode ser adicionada ao modelo torch como um submodulo de ``torch.nn.Module``.

    :param has_params: se possui parametros, como portas RX, RY e outras precisam ser definidas como True, e aquelas sem parametros precisam ser definidas como False, o padrao eh False.
    :param trainable: se possui parametros a serem treinados. Se a camada usa dados de entrada externos para construir a matriz da porta logica, defina como False. Se os parametros a serem treinados precisam ser inicializados a partir desta camada, eh True, o padrao eh False.
    :param init_params: Parametros de inicializacao usados para codificar dados classicos QTensor, o padrao eh None.
    :param wires: Indice do bit do efeito de linha, o padrao eh None.
    :param dtype: A precisao dos dados da matriz interna da porta logica pode ser definida como pyvqnet.kcomplex64 ou pyvqnet.kcomplex128, correspondendo a entrada float ou double respectivamente.
    :param use_dagger: se deve usar a versao conjugada transposta da porta, o padrao eh False.
    :return: a ``pyvqnet.qnn.vqc.tn.QModule`` instance

    Exemplo::
        
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
    
    define a T quantum gate .

    Esta classe herda de ``pyvqnet.qnn.vqc.tn.QModule`` e ``torch.nn.Module``.
    Esta classe pode ser adicionada ao modelo torch como um submodulo de ``torch.nn.Module``.

    :param has_params: se possui parametros, como portas RX, RY e outras precisam ser definidas como True, e aquelas sem parametros precisam ser definidas como False, o padrao eh False.
    :param trainable: se possui parametros a serem treinados. Se a camada usa dados de entrada externos para construir a matriz da porta logica, defina como False. Se os parametros a serem treinados precisam ser inicializados a partir desta camada, eh True, o padrao eh False.
    :param init_params: Parametros de inicializacao usados para codificar dados classicos QTensor, o padrao eh None.
    :param wires: Indice do bit do efeito de linha, o padrao eh None.
    :param dtype: A precisao dos dados da matriz interna da porta logica pode ser definida como pyvqnet.kcomplex64 ou pyvqnet.kcomplex128, correspondendo a entrada float ou double respectivamente.
    :param use_dagger: se deve usar a versao conjugada transposta da porta, o padrao eh False.
    :return: a ``pyvqnet.qnn.vqc.tn.QModule`` instance

    Exemplo::
        
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
    
    define a S quantum gate .

    Esta classe herda de ``pyvqnet.qnn.vqc.tn.QModule`` e ``torch.nn.Module``.
    Esta classe pode ser adicionada ao modelo torch como um submodulo de ``torch.nn.Module``.

    :param has_params: se possui parametros, como portas RX, RY e outras precisam ser definidas como True, e aquelas sem parametros precisam ser definidas como False, o padrao eh False.
    :param trainable: se possui parametros a serem treinados. Se a camada usa dados de entrada externos para construir a matriz da porta logica, defina como False. Se os parametros a serem treinados precisam ser inicializados a partir desta camada, eh True, o padrao eh False.
    :param init_params: Parametros de inicializacao usados para codificar dados classicos QTensor, o padrao eh None.
    :param wires: Indice do bit do efeito de linha, o padrao eh None.
    :param dtype: A precisao dos dados da matriz interna da porta logica pode ser definida como pyvqnet.kcomplex64 ou pyvqnet.kcomplex128, correspondendo a entrada float ou double respectivamente.
    :param use_dagger: se deve usar a versao conjugada transposta da porta, o padrao eh False.
    :return: a ``pyvqnet.qnn.vqc.tn.QModule`` instance

    Exemplo::
        
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
    
    define a PauliX quantum gate .

    Esta classe herda de ``pyvqnet.qnn.vqc.tn.QModule`` e ``torch.nn.Module``.
    Esta classe pode ser adicionada ao modelo torch como um submodulo de ``torch.nn.Module``.


    :param has_params: se possui parametros, como portas RX, RY e outras precisam ser definidas como True, e aquelas sem parametros precisam ser definidas como False, o padrao eh False.
    :param trainable: se possui parametros a serem treinados. Se a camada usa dados de entrada externos para construir a matriz da porta logica, defina como False. Se os parametros a serem treinados precisam ser inicializados a partir desta camada, eh True, o padrao eh False.
    :param init_params: Parametros de inicializacao usados para codificar dados classicos QTensor, o padrao eh None.
    :param wires: Indice do bit do efeito de linha, o padrao eh None.
    :param dtype: A precisao dos dados da matriz interna da porta logica pode ser definida como pyvqnet.kcomplex64 ou pyvqnet.kcomplex128, correspondendo a entrada float ou double respectivamente.
    :param use_dagger: se deve usar a versao conjugada transposta da porta, o padrao eh False.
    :return: a ``pyvqnet.qnn.vqc.tn.QModule`` instance

    Exemplo::
        
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
    
    define a PauliY quantum gate .

    Esta classe herda de ``pyvqnet.qnn.vqc.tn.QModule`` e ``torch.nn.Module``.
    Esta classe pode ser adicionada ao modelo torch como um submodulo de ``torch.nn.Module``.


    :param has_params: se possui parametros, como portas RX, RY e outras precisam ser definidas como True, e aquelas sem parametros precisam ser definidas como False, o padrao eh False.
    :param trainable: se possui parametros a serem treinados. Se a camada usa dados de entrada externos para construir a matriz da porta logica, defina como False. Se os parametros a serem treinados precisam ser inicializados a partir desta camada, eh True, o padrao eh False.
    :param init_params: Parametros de inicializacao usados para codificar dados classicos QTensor, o padrao eh None.
    :param wires: Indice do bit do efeito de linha, o padrao eh None.
    :param dtype: A precisao dos dados da matriz interna da porta logica pode ser definida como pyvqnet.kcomplex64 ou pyvqnet.kcomplex128, correspondendo a entrada float ou double respectivamente.
    :param use_dagger: se deve usar a versao conjugada transposta da porta, o padrao eh False.
    :return: a ``pyvqnet.qnn.vqc.tn.QModule`` instance

    Exemplo::
        
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
    
    define a PauliZ quantum gate .

    Esta classe herda de ``pyvqnet.qnn.vqc.tn.QModule`` e ``torch.nn.Module``.
    Esta classe pode ser adicionada ao modelo torch como um submodulo de ``torch.nn.Module``.


    :param has_params: se possui parametros, como portas RX, RY e outras precisam ser definidas como True, e aquelas sem parametros precisam ser definidas como False, o padrao eh False.
    :param trainable: se possui parametros a serem treinados. Se a camada usa dados de entrada externos para construir a matriz da porta logica, defina como False. Se os parametros a serem treinados precisam ser inicializados a partir desta camada, eh True, o padrao eh False.
    :param init_params: Parametros de inicializacao usados para codificar dados classicos QTensor, o padrao eh None.
    :param wires: Indice do bit do efeito de linha, o padrao eh None.
    :param dtype: A precisao dos dados da matriz interna da porta logica pode ser definida como pyvqnet.kcomplex64 ou pyvqnet.kcomplex128, correspondendo a entrada float ou double respectivamente.
    :param use_dagger: se deve usar a versao conjugada transposta da porta, o padrao eh False.
    :return: a ``pyvqnet.qnn.vqc.tn.QModule`` instance

    Exemplo::
        
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
    
    define a X1 quantum gate .

    Esta classe herda de ``pyvqnet.qnn.vqc.tn.QModule`` e ``torch.nn.Module``.
    Esta classe pode ser adicionada ao modelo torch como um submodulo de ``torch.nn.Module``.

    :param has_params: se possui parametros, como portas RX, RY e outras precisam ser definidas como True, e aquelas sem parametros precisam ser definidas como False, o padrao eh False.
    :param trainable: se possui parametros a serem treinados. Se a camada usa dados de entrada externos para construir a matriz da porta logica, defina como False. Se os parametros a serem treinados precisam ser inicializados a partir desta camada, eh True, o padrao eh False.
    :param init_params: Parametros de inicializacao usados para codificar dados classicos QTensor, o padrao eh None.
    :param wires: Indice do bit do efeito de linha, o padrao eh None.
    :param dtype: A precisao dos dados da matriz interna da porta logica pode ser definida como pyvqnet.kcomplex64 ou pyvqnet.kcomplex128, correspondendo a entrada float ou double respectivamente.
    :param use_dagger: se deve usar a versao conjugada transposta da porta, o padrao eh False.
    :return: a ``pyvqnet.qnn.vqc.tn.QModule`` instance

    Exemplo::
        
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
    
    define a RX quantum gate .

    Esta classe herda de ``pyvqnet.qnn.vqc.tn.QModule`` e ``torch.nn.Module``.
    Esta classe pode ser adicionada ao modelo torch como um submodulo de ``torch.nn.Module``.

    :param has_params: se possui parametros, como portas RX, RY e outras precisam ser definidas como True, e aquelas sem parametros precisam ser definidas como False, o padrao eh False.
    :param trainable: se possui parametros a serem treinados. Se a camada usa dados de entrada externos para construir a matriz da porta logica, defina como False. Se os parametros a serem treinados precisam ser inicializados a partir desta camada, eh True, o padrao eh False.
    :param init_params: Parametros de inicializacao usados para codificar dados classicos QTensor, o padrao eh None.
    :param wires: Indice do bit do efeito de linha, o padrao eh None.
    :param dtype: A precisao dos dados da matriz interna da porta logica pode ser definida como pyvqnet.kcomplex64 ou pyvqnet.kcomplex128, correspondendo a entrada float ou double respectivamente.
    :param use_dagger: se deve usar a versao conjugada transposta da porta, o padrao eh False.
    :return: a ``pyvqnet.qnn.vqc.tn.QModule`` instance

    Exemplo::

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
    
    define a RY quantum gate .

    Esta classe herda de ``pyvqnet.qnn.vqc.tn.QModule`` e ``torch.nn.Module``.
    Esta classe pode ser adicionada ao modelo torch como um submodulo de ``torch.nn.Module``.

    :param has_params: se possui parametros, como portas RX, RY e outras precisam ser definidas como True, e aquelas sem parametros precisam ser definidas como False, o padrao eh False.
    :param trainable: se possui parametros a serem treinados. Se a camada usa dados de entrada externos para construir a matriz da porta logica, defina como False. Se os parametros a serem treinados precisam ser inicializados a partir desta camada, eh True, o padrao eh False.
    :param init_params: Parametros de inicializacao usados para codificar dados classicos QTensor, o padrao eh None.
    :param wires: Indice do bit do efeito de linha, o padrao eh None.
    :param dtype: A precisao dos dados da matriz interna da porta logica pode ser definida como pyvqnet.kcomplex64 ou pyvqnet.kcomplex128, correspondendo a entrada float ou double respectivamente.
    :param use_dagger: se deve usar a versao conjugada transposta da porta, o padrao eh False.
    :return: a ``pyvqnet.qnn.vqc.tn.QModule`` instance

    Exemplo::

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
    
    define a RZ quantum gate .

    Esta classe herda de ``pyvqnet.qnn.vqc.tn.QModule`` e ``torch.nn.Module``.
    Esta classe pode ser adicionada ao modelo torch como um submodulo de ``torch.nn.Module``.

    :param has_params: se possui parametros, como portas RX, RY e outras precisam ser definidas como True, e aquelas sem parametros precisam ser definidas como False, o padrao eh False.
    :param trainable: se possui parametros a serem treinados. Se a camada usa dados de entrada externos para construir a matriz da porta logica, defina como False. Se os parametros a serem treinados precisam ser inicializados a partir desta camada, eh True, o padrao eh False.
    :param init_params: Parametros de inicializacao usados para codificar dados classicos QTensor, o padrao eh None.
    :param wires: Indice do bit do efeito de linha, o padrao eh None.
    :param dtype: A precisao dos dados da matriz interna da porta logica pode ser definida como pyvqnet.kcomplex64 ou pyvqnet.kcomplex128, correspondendo a entrada float ou double respectivamente.
    :param use_dagger: se deve usar a versao conjugada transposta da porta, o padrao eh False.
    :return: a ``pyvqnet.qnn.vqc.tn.QModule`` instance

    Exemplo::

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
    
    define a CRX quantum gate .

    Esta classe herda de ``pyvqnet.qnn.vqc.tn.QModule`` e ``torch.nn.Module``.
    Esta classe pode ser adicionada ao modelo torch como um submodulo de ``torch.nn.Module``.

    :param has_params: se possui parametros, como portas RX, RY e outras precisam ser definidas como True, e aquelas sem parametros precisam ser definidas como False, o padrao eh False.
    :param trainable: se possui parametros a serem treinados. Se a camada usa dados de entrada externos para construir a matriz da porta logica, defina como False. Se os parametros a serem treinados precisam ser inicializados a partir desta camada, eh True, o padrao eh False.
    :param init_params: Parametros de inicializacao usados para codificar dados classicos QTensor, o padrao eh None.
    :param wires: Indice do bit do efeito de linha, o padrao eh None.
    :param dtype: A precisao dos dados da matriz interna da porta logica pode ser definida como pyvqnet.kcomplex64 ou pyvqnet.kcomplex128, correspondendo a entrada float ou double respectivamente.
    :param use_dagger: se deve usar a versao conjugada transposta da porta, o padrao eh False.
    :return: a ``pyvqnet.qnn.vqc.tn.QModule`` instance

    Exemplo::

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
    
    define a CRY quantum gate .

    Esta classe herda de ``pyvqnet.qnn.vqc.tn.QModule`` e ``torch.nn.Module``.
    Esta classe pode ser adicionada ao modelo torch como um submodulo de ``torch.nn.Module``.

    :param has_params: se possui parametros, como portas RX, RY e outras precisam ser definidas como True, e aquelas sem parametros precisam ser definidas como False, o padrao eh False.
    :param trainable: se possui parametros a serem treinados. Se a camada usa dados de entrada externos para construir a matriz da porta logica, defina como False. Se os parametros a serem treinados precisam ser inicializados a partir desta camada, eh True, o padrao eh False.
    :param init_params: Parametros de inicializacao usados para codificar dados classicos QTensor, o padrao eh None.
    :param wires: Indice do bit do efeito de linha, o padrao eh None.
    :param dtype: A precisao dos dados da matriz interna da porta logica pode ser definida como pyvqnet.kcomplex64 ou pyvqnet.kcomplex128, correspondendo a entrada float ou double respectivamente.
    :param use_dagger: se deve usar a versao conjugada transposta da porta, o padrao eh False.
    :return: a ``pyvqnet.qnn.vqc.tn.QModule`` instance

    Exemplo::

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
    
    define a CRZ quantum gate .

    Esta classe herda de ``pyvqnet.qnn.vqc.tn.QModule`` e ``torch.nn.Module``.
    Esta classe pode ser adicionada ao modelo torch como um submodulo de ``torch.nn.Module``.

    :param has_params: se possui parametros, como portas RX, RY e outras precisam ser definidas como True, e aquelas sem parametros precisam ser definidas como False, o padrao eh False.
    :param trainable: se possui parametros a serem treinados. Se a camada usa dados de entrada externos para construir a matriz da porta logica, defina como False. Se os parametros a serem treinados precisam ser inicializados a partir desta camada, eh True, o padrao eh False.
    :param init_params: Parametros de inicializacao usados para codificar dados classicos QTensor, o padrao eh None.
    :param wires: Indice do bit do efeito de linha, o padrao eh None.
    :param dtype: A precisao dos dados da matriz interna da porta logica pode ser definida como pyvqnet.kcomplex64 ou pyvqnet.kcomplex128, correspondendo a entrada float ou double respectivamente.
    :param use_dagger: se deve usar a versao conjugada transposta da porta, o padrao eh False.
    :return: a ``pyvqnet.qnn.vqc.tn.QModule`` instance

    Exemplo::

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
    
    define a U1 quantum gate .

    Esta classe herda de ``pyvqnet.qnn.vqc.tn.QModule`` e ``torch.nn.Module``.
    Esta classe pode ser adicionada ao modelo torch como um submodulo de ``torch.nn.Module``.

    :param has_params: se possui parametros, como portas RX, RY e outras precisam ser definidas como True, e aquelas sem parametros precisam ser definidas como False, o padrao eh False.
    :param trainable: se possui parametros a serem treinados. Se a camada usa dados de entrada externos para construir a matriz da porta logica, defina como False. Se os parametros a serem treinados precisam ser inicializados a partir desta camada, eh True, o padrao eh False.
    :param init_params: Parametros de inicializacao usados para codificar dados classicos QTensor, o padrao eh None.
    :param wires: Indice do bit do efeito de linha, o padrao eh None.
    :param dtype: A precisao dos dados da matriz interna da porta logica pode ser definida como pyvqnet.kcomplex64 ou pyvqnet.kcomplex128, correspondendo a entrada float ou double respectivamente.
    :param use_dagger: se deve usar a versao conjugada transposta da porta, o padrao eh False.
    :return: a ``pyvqnet.qnn.vqc.tn.QModule`` instance

    Exemplo::

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
    
    define a U2 quantum gate .

    Esta classe herda de ``pyvqnet.qnn.vqc.tn.QModule`` e ``torch.nn.Module``.
    Esta classe pode ser adicionada ao modelo torch como um submodulo de ``torch.nn.Module``.

    :param has_params: se possui parametros, como portas RX, RY e outras precisam ser definidas como True, e aquelas sem parametros precisam ser definidas como False, o padrao eh False.
    :param trainable: se possui parametros a serem treinados. Se a camada usa dados de entrada externos para construir a matriz da porta logica, defina como False. Se os parametros a serem treinados precisam ser inicializados a partir desta camada, eh True, o padrao eh False.
    :param init_params: Parametros de inicializacao usados para codificar dados classicos QTensor, o padrao eh None.
    :param wires: Indice do bit do efeito de linha, o padrao eh None.
    :param dtype: A precisao dos dados da matriz interna da porta logica pode ser definida como pyvqnet.kcomplex64 ou pyvqnet.kcomplex128, correspondendo a entrada float ou double respectivamente.
    :param use_dagger: se deve usar a versao conjugada transposta da porta, o padrao eh False.
    :return: a ``pyvqnet.qnn.vqc.tn.QModule`` instance

    Exemplo::

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
    
    define a U3 quantum gate .

    Esta classe herda de ``pyvqnet.qnn.vqc.tn.QModule`` e ``torch.nn.Module``.
    Esta classe pode ser adicionada ao modelo torch como um submodulo de ``torch.nn.Module``.

    :param has_params: se possui parametros, como portas RX, RY e outras precisam ser definidas como True, e aquelas sem parametros precisam ser definidas como False, o padrao eh False.
    :param trainable: se possui parametros a serem treinados. Se a camada usa dados de entrada externos para construir a matriz da porta logica, defina como False. Se os parametros a serem treinados precisam ser inicializados a partir desta camada, eh True, o padrao eh False.
    :param init_params: Parametros de inicializacao usados para codificar dados classicos QTensor, o padrao eh None.
    :param wires: Indice do bit do efeito de linha, o padrao eh None.
    :param dtype: A precisao dos dados da matriz interna da porta logica pode ser definida como pyvqnet.kcomplex64 ou pyvqnet.kcomplex128, correspondendo a entrada float ou double respectivamente.
    :param use_dagger: se deve usar a versao conjugada transposta da porta, o padrao eh False.
    :return: a ``pyvqnet.qnn.vqc.tn.QModule`` instance

    Exemplo::

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
    
    define a CNOT quantum gate , alias `CX` .

    Esta classe herda de ``pyvqnet.qnn.vqc.tn.QModule`` e ``torch.nn.Module``.
    Esta classe pode ser adicionada ao modelo torch como um submodulo de ``torch.nn.Module``.

    :param has_params: se possui parametros, como portas RX, RY e outras precisam ser definidas como True, e aquelas sem parametros precisam ser definidas como False, o padrao eh False.
    :param trainable: se possui parametros a serem treinados. Se a camada usa dados de entrada externos para construir a matriz da porta logica, defina como False. Se os parametros a serem treinados precisam ser inicializados a partir desta camada, eh True, o padrao eh False.
    :param init_params: Parametros de inicializacao usados para codificar dados classicos QTensor, o padrao eh None.
    :param wires: Indice do bit do efeito de linha, o padrao eh None.
    :param dtype: A precisao dos dados da matriz interna da porta logica pode ser definida como pyvqnet.kcomplex64 ou pyvqnet.kcomplex128, correspondendo a entrada float ou double respectivamente.
    :param use_dagger: se deve usar a versao conjugada transposta da porta, o padrao eh False.
    :return: a ``pyvqnet.qnn.vqc.tn.QModule`` instance

    Exemplo::

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
    
    define a CY quantum gate .

    Esta classe herda de ``pyvqnet.qnn.vqc.tn.QModule`` e ``torch.nn.Module``.
    Esta classe pode ser adicionada ao modelo torch como um submodulo de ``torch.nn.Module``.

    :param has_params: se possui parametros, como portas RX, RY e outras precisam ser definidas como True, e aquelas sem parametros precisam ser definidas como False, o padrao eh False.
    :param trainable: se possui parametros a serem treinados. Se a camada usa dados de entrada externos para construir a matriz da porta logica, defina como False. Se os parametros a serem treinados precisam ser inicializados a partir desta camada, eh True, o padrao eh False.
    :param init_params: Parametros de inicializacao usados para codificar dados classicos QTensor, o padrao eh None.
    :param wires: Indice do bit do efeito de linha, o padrao eh None.
    :param dtype: A precisao dos dados da matriz interna da porta logica pode ser definida como pyvqnet.kcomplex64 ou pyvqnet.kcomplex128, correspondendo a entrada float ou double respectivamente.
    :param use_dagger: se deve usar a versao conjugada transposta da porta, o padrao eh False.
    :return: a ``pyvqnet.qnn.vqc.tn.QModule`` instance

    Exemplo::

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
    
    define a CZ quantum gate .

    Esta classe herda de ``pyvqnet.qnn.vqc.tn.QModule`` e ``torch.nn.Module``.
    Esta classe pode ser adicionada ao modelo torch como um submodulo de ``torch.nn.Module``.

    :param has_params: se possui parametros, como portas RX, RY e outras precisam ser definidas como True, e aquelas sem parametros precisam ser definidas como False, o padrao eh False.
    :param trainable: se possui parametros a serem treinados. Se a camada usa dados de entrada externos para construir a matriz da porta logica, defina como False. Se os parametros a serem treinados precisam ser inicializados a partir desta camada, eh True, o padrao eh False.
    :param init_params: Parametros de inicializacao usados para codificar dados classicos QTensor, o padrao eh None.
    :param wires: Indice do bit do efeito de linha, o padrao eh None.
    :param dtype: A precisao dos dados da matriz interna da porta logica pode ser definida como pyvqnet.kcomplex64 ou pyvqnet.kcomplex128, correspondendo a entrada float ou double respectivamente.
    :param use_dagger: se deve usar a versao conjugada transposta da porta, o padrao eh False.
    :return: a ``pyvqnet.qnn.vqc.tn.QModule`` instance

    Exemplo::

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
    
    define a CR quantum gate .

    Esta classe herda de ``pyvqnet.qnn.vqc.tn.QModule`` e ``torch.nn.Module``.
    Esta classe pode ser adicionada ao modelo torch como um submodulo de ``torch.nn.Module``.

    :param has_params: se possui parametros, como portas RX, RY e outras precisam ser definidas como True, e aquelas sem parametros precisam ser definidas como False, o padrao eh False.
    :param trainable: se possui parametros a serem treinados. Se a camada usa dados de entrada externos para construir a matriz da porta logica, defina como False. Se os parametros a serem treinados precisam ser inicializados a partir desta camada, eh True, o padrao eh False.
    :param init_params: Parametros de inicializacao usados para codificar dados classicos QTensor, o padrao eh None.
    :param wires: Indice do bit do efeito de linha, o padrao eh None.
    :param dtype: A precisao dos dados da matriz interna da porta logica pode ser definida como pyvqnet.kcomplex64 ou pyvqnet.kcomplex128, correspondendo a entrada float ou double respectivamente.
    :param use_dagger: se deve usar a versao conjugada transposta da porta, o padrao eh False.
    :return: a ``pyvqnet.qnn.vqc.tn.QModule`` instance

    Exemplo::

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
    
    define a SWAP quantum gate .

    Esta classe herda de ``pyvqnet.qnn.vqc.tn.QModule`` e ``torch.nn.Module``.
    Esta classe pode ser adicionada ao modelo torch como um submodulo de ``torch.nn.Module``.

    :param has_params: se possui parametros, como portas RX, RY e outras precisam ser definidas como True, e aquelas sem parametros precisam ser definidas como False, o padrao eh False.
    :param trainable: se possui parametros a serem treinados. Se a camada usa dados de entrada externos para construir a matriz da porta logica, defina como False. Se os parametros a serem treinados precisam ser inicializados a partir desta camada, eh True, o padrao eh False.
    :param init_params: Parametros de inicializacao usados para codificar dados classicos QTensor, o padrao eh None.
    :param wires: Indice do bit do efeito de linha, o padrao eh None.
    :param dtype: A precisao dos dados da matriz interna da porta logica pode ser definida como pyvqnet.kcomplex64 ou pyvqnet.kcomplex128, correspondendo a entrada float ou double respectivamente.
    :param use_dagger: se deve usar a versao conjugada transposta da porta, o padrao eh False.
    :return: a ``pyvqnet.qnn.vqc.tn.QModule`` instance

    Exemplo::

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
    
    define a SWAP quantum gate .

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

    Esta classe herda de ``pyvqnet.qnn.vqc.tn.QModule`` e ``torch.nn.Module``.
    Esta classe pode ser adicionada ao modelo torch como um submodulo de ``torch.nn.Module``.

    :param has_params: se possui parametros, como portas RX, RY e outras precisam ser definidas como True, e aquelas sem parametros precisam ser definidas como False, o padrao eh False.
    :param trainable: se possui parametros a serem treinados. Se a camada usa dados de entrada externos para construir a matriz da porta logica, defina como False. Se os parametros a serem treinados precisam ser inicializados a partir desta camada, eh True, o padrao eh False.
    :param init_params: Parametros de inicializacao usados para codificar dados classicos QTensor, o padrao eh None.
    :param wires: Indice do bit do efeito de linha, o padrao eh None.
    :param dtype: A precisao dos dados da matriz interna da porta logica pode ser definida como pyvqnet.kcomplex64 ou pyvqnet.kcomplex128, correspondendo a entrada float ou double respectivamente.
    :param use_dagger: se deve usar a versao conjugada transposta da porta, o padrao eh False.
    :return: a ``pyvqnet.qnn.vqc.tn.QModule`` instance

    Exemplo::

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
    
    define a RXX quantum gate .

    Esta classe herda de ``pyvqnet.qnn.vqc.tn.QModule`` e ``torch.nn.Module``.
    Esta classe pode ser adicionada ao modelo torch como um submodulo de ``torch.nn.Module``.

    :param has_params: se possui parametros, como portas RX, RY e outras precisam ser definidas como True, e aquelas sem parametros precisam ser definidas como False, o padrao eh False.
    :param trainable: se possui parametros a serem treinados. Se a camada usa dados de entrada externos para construir a matriz da porta logica, defina como False. Se os parametros a serem treinados precisam ser inicializados a partir desta camada, eh True, o padrao eh False.
    :param init_params: Parametros de inicializacao usados para codificar dados classicos QTensor, o padrao eh None.
    :param wires: Indice do bit do efeito de linha, o padrao eh None.
    :param dtype: A precisao dos dados da matriz interna da porta logica pode ser definida como pyvqnet.kcomplex64 ou pyvqnet.kcomplex128, correspondendo a entrada float ou double respectivamente.
    :param use_dagger: se deve usar a versao conjugada transposta da porta, o padrao eh False.
    :return: a ``pyvqnet.qnn.vqc.tn.QModule`` instance

    Exemplo::

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
    
    define a RYY quantum gate .

    Esta classe herda de ``pyvqnet.qnn.vqc.tn.QModule`` e ``torch.nn.Module``.
    Esta classe pode ser adicionada ao modelo torch como um submodulo de ``torch.nn.Module``.

    :param has_params: se possui parametros, como portas RX, RY e outras precisam ser definidas como True, e aquelas sem parametros precisam ser definidas como False, o padrao eh False.
    :param trainable: se possui parametros a serem treinados. Se a camada usa dados de entrada externos para construir a matriz da porta logica, defina como False. Se os parametros a serem treinados precisam ser inicializados a partir desta camada, eh True, o padrao eh False.
    :param init_params: Parametros de inicializacao usados para codificar dados classicos QTensor, o padrao eh None.
    :param wires: Indice do bit do efeito de linha, o padrao eh None.
    :param dtype: A precisao dos dados da matriz interna da porta logica pode ser definida como pyvqnet.kcomplex64 ou pyvqnet.kcomplex128, correspondendo a entrada float ou double respectivamente.
    :param use_dagger: se deve usar a versao conjugada transposta da porta, o padrao eh False.
    :return: a ``pyvqnet.qnn.vqc.tn.QModule`` instance

    Exemplo::

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
    
    define a RZZ quantum gate .

    Esta classe herda de ``pyvqnet.qnn.vqc.tn.QModule`` e ``torch.nn.Module``.
    Esta classe pode ser adicionada ao modelo torch como um submodulo de ``torch.nn.Module``.

    :param has_params: se possui parametros, como portas RX, RY e outras precisam ser definidas como True, e aquelas sem parametros precisam ser definidas como False, o padrao eh False.
    :param trainable: se possui parametros a serem treinados. Se a camada usa dados de entrada externos para construir a matriz da porta logica, defina como False. Se os parametros a serem treinados precisam ser inicializados a partir desta camada, eh True, o padrao eh False.
    :param init_params: Parametros de inicializacao usados para codificar dados classicos QTensor, o padrao eh None.
    :param wires: Indice do bit do efeito de linha, o padrao eh None.
    :param dtype: A precisao dos dados da matriz interna da porta logica pode ser definida como pyvqnet.kcomplex64 ou pyvqnet.kcomplex128, correspondendo a entrada float ou double respectivamente.
    :param use_dagger: se deve usar a versao conjugada transposta da porta, o padrao eh False.
    :return: a ``pyvqnet.qnn.vqc.tn.QModule`` instance

    Exemplo::

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
    
    define a RZX quantum gate .

    Esta classe herda de ``pyvqnet.qnn.vqc.tn.QModule`` e ``torch.nn.Module``.
    Esta classe pode ser adicionada ao modelo torch como um submodulo de ``torch.nn.Module``.

    :param has_params: se possui parametros, como portas RX, RY e outras precisam ser definidas como True, e aquelas sem parametros precisam ser definidas como False, o padrao eh False.
    :param trainable: se possui parametros a serem treinados. Se a camada usa dados de entrada externos para construir a matriz da porta logica, defina como False. Se os parametros a serem treinados precisam ser inicializados a partir desta camada, eh True, o padrao eh False.
    :param init_params: Parametros de inicializacao usados para codificar dados classicos QTensor, o padrao eh None.
    :param wires: Indice do bit do efeito de linha, o padrao eh None.
    :param dtype: A precisao dos dados da matriz interna da porta logica pode ser definida como pyvqnet.kcomplex64 ou pyvqnet.kcomplex128, correspondendo a entrada float ou double respectivamente.
    :param use_dagger: se deve usar a versao conjugada transposta da porta, o padrao eh False.
    :return: a ``pyvqnet.qnn.vqc.tn.QModule`` instance

    Exemplo::

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
    
    define a Toffoli quantum gate .

    Esta classe herda de ``pyvqnet.qnn.vqc.tn.QModule`` e ``torch.nn.Module``.
    Esta classe pode ser adicionada ao modelo torch como um submodulo de ``torch.nn.Module``.

    :param has_params: se possui parametros, como portas RX, RY e outras precisam ser definidas como True, e aquelas sem parametros precisam ser definidas como False, o padrao eh False.
    :param trainable: se possui parametros a serem treinados. Se a camada usa dados de entrada externos para construir a matriz da porta logica, defina como False. Se os parametros a serem treinados precisam ser inicializados a partir desta camada, eh True, o padrao eh False.
    :param init_params: Parametros de inicializacao usados para codificar dados classicos QTensor, o padrao eh None.
    :param wires: Indice do bit do efeito de linha, o padrao eh None.
    :param dtype: A precisao dos dados da matriz interna da porta logica pode ser definida como pyvqnet.kcomplex64 ou pyvqnet.kcomplex128, correspondendo a entrada float ou double respectivamente.
    :param use_dagger: se deve usar a versao conjugada transposta da porta, o padrao eh False.
    :return: a ``pyvqnet.qnn.vqc.tn.QModule`` instance

    Exemplo::

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
    
    define a IsingXX quantum gate .

    Esta classe herda de ``pyvqnet.qnn.vqc.tn.QModule`` e ``torch.nn.Module``.
    Esta classe pode ser adicionada ao modelo torch como um submodulo de ``torch.nn.Module``.

    :param has_params: se possui parametros, como portas RX, RY e outras precisam ser definidas como True, e aquelas sem parametros precisam ser definidas como False, o padrao eh False.
    :param trainable: se possui parametros a serem treinados. Se a camada usa dados de entrada externos para construir a matriz da porta logica, defina como False. Se os parametros a serem treinados precisam ser inicializados a partir desta camada, eh True, o padrao eh False.
    :param init_params: Parametros de inicializacao usados para codificar dados classicos QTensor, o padrao eh None.
    :param wires: Indice do bit do efeito de linha, o padrao eh None.
    :param dtype: A precisao dos dados da matriz interna da porta logica pode ser definida como pyvqnet.kcomplex64 ou pyvqnet.kcomplex128, correspondendo a entrada float ou double respectivamente.
    :param use_dagger: se deve usar a versao conjugada transposta da porta, o padrao eh False.
    :return: a ``pyvqnet.qnn.vqc.tn.QModule`` instance

    Exemplo::

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
    
    define a IsingYY quantum gate .

    Esta classe herda de ``pyvqnet.qnn.vqc.tn.QModule`` e ``torch.nn.Module``.
    Esta classe pode ser adicionada ao modelo torch como um submodulo de ``torch.nn.Module``.

    :param has_params: se possui parametros, como portas RX, RY e outras precisam ser definidas como True, e aquelas sem parametros precisam ser definidas como False, o padrao eh False.
    :param trainable: se possui parametros a serem treinados. Se a camada usa dados de entrada externos para construir a matriz da porta logica, defina como False. Se os parametros a serem treinados precisam ser inicializados a partir desta camada, eh True, o padrao eh False.
    :param init_params: Parametros de inicializacao usados para codificar dados classicos QTensor, o padrao eh None.
    :param wires: Indice do bit do efeito de linha, o padrao eh None.
    :param dtype: A precisao dos dados da matriz interna da porta logica pode ser definida como pyvqnet.kcomplex64 ou pyvqnet.kcomplex128, correspondendo a entrada float ou double respectivamente.
    :param use_dagger: se deve usar a versao conjugada transposta da porta, o padrao eh False.
    :return: a ``pyvqnet.qnn.vqc.tn.QModule`` instance

    Exemplo::

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
    
    define a IsingZZ quantum gate .

    Esta classe herda de ``pyvqnet.qnn.vqc.tn.QModule`` e ``torch.nn.Module``.
    Esta classe pode ser adicionada ao modelo torch como um submodulo de ``torch.nn.Module``.

    :param has_params: se possui parametros, como portas RX, RY e outras precisam ser definidas como True, e aquelas sem parametros precisam ser definidas como False, o padrao eh False.
    :param trainable: se possui parametros a serem treinados. Se a camada usa dados de entrada externos para construir a matriz da porta logica, defina como False. Se os parametros a serem treinados precisam ser inicializados a partir desta camada, eh True, o padrao eh False.
    :param init_params: Parametros de inicializacao usados para codificar dados classicos QTensor, o padrao eh None.
    :param wires: Indice do bit do efeito de linha, o padrao eh None.
    :param dtype: A precisao dos dados da matriz interna da porta logica pode ser definida como pyvqnet.kcomplex64 ou pyvqnet.kcomplex128, correspondendo a entrada float ou double respectivamente.
    :param use_dagger: se deve usar a versao conjugada transposta da porta, o padrao eh False.
    :return: a ``pyvqnet.qnn.vqc.tn.QModule`` instance

    Exemplo::

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
    
    define a IsingXY quantum gate .

    Esta classe herda de ``pyvqnet.qnn.vqc.tn.QModule`` e ``torch.nn.Module``.
    Esta classe pode ser adicionada ao modelo torch como um submodulo de ``torch.nn.Module``.

    :param has_params: se possui parametros, como portas RX, RY e outras precisam ser definidas como True, e aquelas sem parametros precisam ser definidas como False, o padrao eh False.
    :param trainable: se possui parametros a serem treinados. Se a camada usa dados de entrada externos para construir a matriz da porta logica, defina como False. Se os parametros a serem treinados precisam ser inicializados a partir desta camada, eh True, o padrao eh False.
    :param init_params: Parametros de inicializacao usados para codificar dados classicos QTensor, o padrao eh None.
    :param wires: Indice do bit do efeito de linha, o padrao eh None.
    :param dtype: A precisao dos dados da matriz interna da porta logica pode ser definida como pyvqnet.kcomplex64 ou pyvqnet.kcomplex128, correspondendo a entrada float ou double respectivamente.
    :param use_dagger: se deve usar a versao conjugada transposta da porta, o padrao eh False.
    :return: a ``pyvqnet.qnn.vqc.tn.QModule`` instance

    Exemplo::

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
    
    define a PhaseShift quantum gate .

    Esta classe herda de ``pyvqnet.qnn.vqc.tn.QModule`` e ``torch.nn.Module``.
    Esta classe pode ser adicionada ao modelo torch como um submodulo de ``torch.nn.Module``.

    :param has_params: se possui parametros, como portas RX, RY e outras precisam ser definidas como True, e aquelas sem parametros precisam ser definidas como False, o padrao eh False.
    :param trainable: se possui parametros a serem treinados. Se a camada usa dados de entrada externos para construir a matriz da porta logica, defina como False. Se os parametros a serem treinados precisam ser inicializados a partir desta camada, eh True, o padrao eh False.
    :param init_params: Parametros de inicializacao usados para codificar dados classicos QTensor, o padrao eh None.
    :param wires: Indice do bit do efeito de linha, o padrao eh None.
    :param dtype: A precisao dos dados da matriz interna da porta logica pode ser definida como pyvqnet.kcomplex64 ou pyvqnet.kcomplex128, correspondendo a entrada float ou double respectivamente.
    :param use_dagger: se deve usar a versao conjugada transposta da porta, o padrao eh False.
    :return: a ``pyvqnet.qnn.vqc.tn.QModule`` instance

    Exemplo::

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
    
    define a MultiRZ quantum gate .

    Esta classe herda de ``pyvqnet.qnn.vqc.tn.QModule`` e ``torch.nn.Module``.
    Esta classe pode ser adicionada ao modelo torch como um submodulo de ``torch.nn.Module``.

    :param has_params: se possui parametros, como portas RX, RY e outras precisam ser definidas como True, e aquelas sem parametros precisam ser definidas como False, o padrao eh False.
    :param trainable: se possui parametros a serem treinados. Se a camada usa dados de entrada externos para construir a matriz da porta logica, defina como False. Se os parametros a serem treinados precisam ser inicializados a partir desta camada, eh True, o padrao eh False.
    :param init_params: Parametros de inicializacao usados para codificar dados classicos QTensor, o padrao eh None.
    :param wires: Indice do bit do efeito de linha, o padrao eh None.
    :param dtype: A precisao dos dados da matriz interna da porta logica pode ser definida como pyvqnet.kcomplex64 ou pyvqnet.kcomplex128, correspondendo a entrada float ou double respectivamente.
    :param use_dagger: se deve usar a versao conjugada transposta da porta, o padrao eh False.
    :return: a ``pyvqnet.qnn.vqc.tn.QModule`` instance

    Exemplo::

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
    
    define a SDG quantum gate .

    Esta classe herda de ``pyvqnet.qnn.vqc.tn.QModule`` e ``torch.nn.Module``.
    Esta classe pode ser adicionada ao modelo torch como um submodulo de ``torch.nn.Module``.

    :param has_params: se possui parametros, como portas RX, RY e outras precisam ser definidas como True, e aquelas sem parametros precisam ser definidas como False, o padrao eh False.
    :param trainable: se possui parametros a serem treinados. Se a camada usa dados de entrada externos para construir a matriz da porta logica, defina como False. Se os parametros a serem treinados precisam ser inicializados a partir desta camada, eh True, o padrao eh False.
    :param init_params: Parametros de inicializacao usados para codificar dados classicos QTensor, o padrao eh None.
    :param wires: Indice do bit do efeito de linha, o padrao eh None.
    :param dtype: A precisao dos dados da matriz interna da porta logica pode ser definida como pyvqnet.kcomplex64 ou pyvqnet.kcomplex128, correspondendo a entrada float ou double respectivamente.
    :param use_dagger: se deve usar a versao conjugada transposta da porta, o padrao eh False.
    :return: a ``pyvqnet.qnn.vqc.tn.QModule`` instance

    Exemplo::
        
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
    
    define a SDG quantum gate .

    Esta classe herda de ``pyvqnet.qnn.vqc.tn.QModule`` e ``torch.nn.Module``.
    Esta classe pode ser adicionada ao modelo torch como um submodulo de ``torch.nn.Module``.

    :param has_params: se possui parametros, como portas RX, RY e outras precisam ser definidas como True, e aquelas sem parametros precisam ser definidas como False, o padrao eh False.
    :param trainable: se possui parametros a serem treinados. Se a camada usa dados de entrada externos para construir a matriz da porta logica, defina como False. Se os parametros a serem treinados precisam ser inicializados a partir desta camada, eh True, o padrao eh False.
    :param init_params: Parametros de inicializacao usados para codificar dados classicos QTensor, o padrao eh None.
    :param wires: Indice do bit do efeito de linha, o padrao eh None.
    :param dtype: A precisao dos dados da matriz interna da porta logica pode ser definida como pyvqnet.kcomplex64 ou pyvqnet.kcomplex128, correspondendo a entrada float ou double respectivamente.
    :param use_dagger: se deve usar a versao conjugada transposta da porta, o padrao eh False.
    :return: a ``pyvqnet.qnn.vqc.tn.QModule`` instance

    Exemplo::
        
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
    
    define a ControlledPhaseShift quantum gate .

    Esta classe herda de ``pyvqnet.qnn.vqc.tn.QModule`` e ``torch.nn.Module``.
    Esta classe pode ser adicionada ao modelo torch como um submodulo de ``torch.nn.Module``.

    :param has_params: se possui parametros, como portas RX, RY e outras precisam ser definidas como True, e aquelas sem parametros precisam ser definidas como False, o padrao eh False.
    :param trainable: se possui parametros a serem treinados. Se a camada usa dados de entrada externos para construir a matriz da porta logica, defina como False. Se os parametros a serem treinados precisam ser inicializados a partir desta camada, eh True, o padrao eh False.
    :param init_params: Parametros de inicializacao usados para codificar dados classicos QTensor, o padrao eh None.
    :param wires: Indice do bit do efeito de linha, o padrao eh None.
    :param dtype: A precisao dos dados da matriz interna da porta logica pode ser definida como pyvqnet.kcomplex64 ou pyvqnet.kcomplex128, correspondendo a entrada float ou double respectivamente.
    :param use_dagger: se deve usar a versao conjugada transposta da porta, o padrao eh False.
    :return: a ``pyvqnet.qnn.vqc.tn.QModule`` instance

    Exemplo::

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


API de Medicoes
------------------------------

VQC_Purity
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^


.. py:function:: pyvqnet.qnn.vqc.tn.VQC_Purity(state, qubits_idx, num_wires, use_tn=False)

    Calcula a pureza em um qubit especifico ``qubits_idx`` a partir do vetor de estado ``state``.

    .. math::
        \gamma = \text{Tr}(\rho^2)

    where :math:`\rho` is a density matrix. The purity of a normalized quantum state satisfies :math:`\frac{1}{d} \leq \gamma \leq 1` ,
    where :math:`d` is the dimension of the Hilbert space.
    The purity of the pure state is 1.

    :param state: Estado quantico obtido de TNQMachine.get_states()
    :param qubits_idx: Qubit index for which to calculate purity
    :param num_wires: Qubit idx
    :param use_tn: use tensornetwork need to be set True, default False

    :return: purity

    .. note::
        
        batch_size need TNQModule.

    Exemplo::

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

    :param q_machine: Estado quantico obtido de pyqpanda get_qstate()
    :param obs: observables

    :return: variance value

    Exemplo::

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

    Exemplo::

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

    Calcula o resultado da medicao de probabilidade do circuito quantico em um bit especifico.

    Esta classe herda de ``pyvqnet.qnn.vqc.tn.QModule`` e ``torch.nn.Module``.
    Esta classe pode ser adicionada ao modelo torch como um submodulo de ``torch.nn.Module``.

    :param wires: O indice do bit de medicao, lista, tupla ou inteiro.
    :param name: O nome do modulo, padrao: "".
    :return: O resultado da medicao, QTensor.

    Exemplo::

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

    Calcula os resultados de medicao de circuitos quanticos, suporta entrada obs como multiplos ou unicos operadores Pauli ou Hamiltonianos.
    
    For example:

    {\'X0\': 0.23} indicates a PauliX effect on qubit 0, with a coefficient of 0.23.

    {\'X1 Z2\': 2.4,\'Y2\': -0.5} corresponds to the observed value 2.4 * X1 @ Z2 - 0.5 * Y2.

    [{\'X1 Z2\': 4,\'Z1 Z0\': 3},{\'X1 Y2 Z0\': 3.5}] corresponds to the two observed values 4 * X1 @ Z2 + 3 * Z1 @ Z0 and 3.5 * X1 @ Y2 @ Z0.
        
    Esta classe herda de ``pyvqnet.qnn.vqc.tn.QModule`` e ``torch.nn.Module``.

    Esta classe pode ser adicionada ao modelo torch como um submodulo de ``torch.nn.Module``.

    :param obs: observable.
    :param name: module name, default: "".
    :return: measurement result, QTensor.

    Exemplo::

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

    Get sample results with shot on  specific wires.

    Esta classe herda de ``pyvqnet.qnn.vqc.tn.QModule`` e ``torch.nn.Module``.
    Esta classe pode ser adicionada ao modelo torch como um submodulo de ``torch.nn.Module``.


    :param wires: Indice do qubit de amostragem. Valor padrao: None, usa todos os bits do simulador em tempo de execucao.
    :param obs: Este valor so pode ser None.
    :param shots: Sample repetition count, default value: 1.
    :param name: O nome deste modulo, valor padrao: "".
    :return: a measurement method class

    Exemplo::

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

    Compute the expectation of a Hermitian quantity in a quantum circuit.

    Esta classe herda de ``pyvqnet.qnn.vqc.tn.QModule`` e ``torch.nn.Module``.
    Esta classe pode ser adicionada ao modelo torch como um submodulo de ``torch.nn.Module``.


    :param obs: Hermitian quantity.
    :param name: module name, default: "".
    :return: expected result, QTensor.

    Exemplo::

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

Modelos comuns para circuitos quanticos
--------------------------------------------

VQC_HardwareEfficientAnsatz
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^


.. py:class:: pyvqnet.qnn.vqc.tn.VQC_HardwareEfficientAnsatz(n_qubits,single_rot_gate_list,entangle_gate="CNOT",entangle_rules='linear',depth=1,initial=None,dtype=None)

    Implementation of Hardware Efficient Ansatz introduced in the paper: `Hardware-efficient Variational Quantum Eigensolver for Small Molecules <https://arxiv.org/pdf/1704.05018.pdf>`__ .

    Esta classe herda de ``pyvqnet.qnn.vqc.tn.QModule`` e ``torch.nn.Module``.

    Esta classe pode ser adicionada ao modelo torch como um submodulo de ``torch.nn.Module``.


    :param n_qubits: Number of qubits.
    :param single_rot_gate_list: Uma lista de portas de rotacao de unico qubit eh construida por uma ou varias portas de rotacao que atuam em cada qubit. Atualmente suporta Rx, Ry, Rz.
    :param entangle_gate: A porta de entrelacamento nao parametrizada. CNOT, CZ sao suportados. Padrao: CNOT.
    :param entangle_rules: Como a porta de entrelacamento eh usada no circuito. 'linear' significa que a porta de entrelacamento atuara em cada qubit vizinho. 'all' significa que a porta de entrelacamento atuara em quaisquer dois qubits. Padrao: linear.
    :param depth: A profundidade do ansatz, padrao: 1.
    :param initial: valor inicial unico para parametros, padrao: None, este modulo inicializara parametros aleatoriamente.
    :param dtype: data dtype of parameters.
    :return: a VQC_HardwareEfficientAnsatz instance.

    Exemplo::

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

    A layer consisting of a single-parameter single-qubit rotation on each qubit, followed by multiple CNOT gates in a closed chain or ring combination.

    A ring of CNOT gates connects each qubit to its neighbors, and finally the a qubit is considered to be the neighbor of the a th qubit.

    Esta classe herda de ``pyvqnet.qnn.vqc.tn.QModule`` e ``torch.nn.Module``.

    Esta classe pode ser adicionada ao modelo torch como um submodulo de ``torch.nn.Module``.

    :param num_layers: number of repeat layers, default: 1.
    :param num_qubits: number of qubits, default: 1.
    :param rotation: one-parameter single-qubit gate to use, default: `RX`
    :param initial: valor inicial igual para todos os parametros. padrao: None, os parametros serao inicializados aleatoriamente.
    :param dtype: data type of parameter, default:None,use float32.
    :return: A VQC_BasicEntanglerTemplate instance

    Exemplo::

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

    Layers consisting of single qubit rotations and entanglers, as in `circuit-centric classifier design <https://arxiv.org/abs/1804.00633>`__ .

    Esta classe herda de ``pyvqnet.qnn.vqc.tn.QModule`` e ``torch.nn.Module``.
    Esta classe pode ser adicionada ao modelo torch como um submodulo de ``torch.nn.Module``.


    :param num_layers: number of repeat layers, default: 1.
    :param num_qubits: number of qubits, default: 1.
    :param ranges: sequencia determinando o hiperparametro de intervalo para cada camada subsequente; padrao: None
                                using :math: `r=l \mod M` for the :math:`l` th layer and :math:`M` qubits.
    :param initial: initial value for all parameters.default: None,initialized randomly.
    :param dtype: data type of parameter, default:None,use float32.
    :return: A VQC_StronglyEntanglingTemplate instance.

    Exemplo::

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
    
    Use RZ,RY,RZ to create variational quantum circuits to encode classical data into quantum states.
    Reference `Quantum embeddings for machine learning <https://arxiv.org/abs/2001.03622>`_.

    Esta classe herda de ``pyvqnet.qnn.vqc.tn.QModule`` e ``torch.nn.Module``.
    Esta classe pode ser adicionada ao modelo torch como um submodulo de ``torch.nn.Module``.

 
    :param num_repetitions_input: number of repeat times to encode input in a submodule.
    :param depth_input: number of input dimension .
    :param num_unitary_layers: number of repeat times of variational quantum gates.
    :param num_repetitions: number of repeat times of submodule.
    :param initial: parameter initialization value, default is None
    :param dtype: parameter type, default is None, use float32.
    :param name: class name
    :return: A VQC_QuantumEmbedding instance.

    Exemplo::

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

    19 ansatz diferentes do artigo `Expressibility and entangling capability of parameterized quantum circuits for hybrid quantum-classical algorithms <https://arxiv.org/pdf/1905.10876.pdf>`_.

    Esta classe herda de ``pyvqnet.qnn.vqc.tn.QModule`` e ``torch.nn.Module``.

    Esta classe pode ser adicionada ao modelo torch como um submodulo de ``torch.nn.Module``.

    :param type: Tipo de circuito de 1 a 19, um total de 19 linhas.
    :param num_wires: Number of qubits.
    :param depth: Circuit depth.
    :param dtype: data type of parameter, default:None,use float32.
    :param name: Name, default "".

    :return:
        a ExpressiveEntanglingAnsatz instance

    Exemplo::

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

    Codifica n caracteristicas binarias no estado base de n-qubits de ``q_machine``. Esta funcao tem o alias `VQC_BasisEmbedding`.

    For example, for ``basis_state=([0, 1, 1])``, the basis state in the quantum system is :math:`|011 \rangle`.

    :param basis_state: ``(n)`` size binary input.
    :param q_machine: quantum machine device.
    

    Exemplo::

        import pyvqnet
        pyvqnet.backends.set_backend("torch")
        from pyvqnet.qnn.vqc.tn import vqc_basis_embedding,TNQMachine
        qm  = TNQMachine(3)
        vqc_basis_embedding(basis_state=[1,1,0],q_machine=qm)
        print(qm.get_states())




vqc_angle_embedding
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^


.. py:function:: pyvqnet.qnn.vqc.tn.vqc_angle_embedding(input_feat, wires, q_machine: pyvqnet.qnn.vqc.tn.TNQMachine, rotation: str = "X")

    Encodes :math:`N` features into the rotation angle of :math:`n` qubits, where :math:`N \leq n`.
    Esta funcao tem o alias `VQC_AngleEmbedding` .

    The rotation can be selected as: 'X' , 'Y' , 'Z', as defined by the ``rotation`` parameter:

    * ``rotation='X'`` Use the feature as the angle of RX rotation.

    * ``rotation='Y'`` Use the feature as the angle of RY rotation.

    * ``rotation='Z'`` Use the feature as the angle of RZ rotation.

    ``wires`` represents the idx of the rotation gate on the qubit.

    :param input_feat: Array representing parameters.
    :param wires: Qubit idx.
    :param q_machine: Quantum machine device.
    :param rotation: Rotation gate, default is "X".

    Exemplo::

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

    Codifica uma caracteristica :math:`2^n` em um vetor de amplitude de :math:`n` qubits. Esta funcao tem o alias `VQC_AmplitudeEmbedding`.

    :param input_feature: array numpy representando o parametro.
    :param q_machine: quantum machine device.
    

    Exemplo::

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
    :param rep: Numero de vezes para repetir o bloco de circuito quantico, padrao eh 1.

    Exemplo::

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

    Combinacao arbitraria de porta logica quantica de rotacao de unico bit quantico. Esta funcao tem o alias: ``VQC_RotCircuit`` .

    .. math::
        R(\phi,\theta,\omega) = RZ(\omega)RY(\theta)RZ(\phi)= \begin{bmatrix}
        e^{-i(\phi+\omega)/2}\cos(\theta/2) & -e^{i(\phi-\omega)/2}\sin(\theta/2) \\
        e^{-i(\phi-\omega)/2}\sin(\theta/2) & e^{i(\phi+\omega)/2}\cos(\theta/2)
        \end{bmatrix}.

    :param q_machine: quantum virtual machine device.
    :param wire: quantum bit index.
    :param params: represents parameters :math:`[\phi, \theta, \omega]`.

    Exemplo::

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

    Combinacao de porta logica quantica de rotacao controlada de unico bit quantico. Esta funcao tem o alias: ``VQC_CRotCircuit`` .

    .. math:: 
        CR(\phi, \theta, \omega) = \begin{bmatrix}
        1 & 0 & 0 & 0 \\
        0 & 1 & 0 & 0\\
        0 & 0 & e^{-i(\phi+\omega)/2}\cos(\theta/2) & -e^{i(\phi-\omega)/2}\sin(\theta/2)\\
        0 & 0 & e^{-i(\phi-\omega)/2}\sin(\theta/2) & e^{i(\phi+\omega)/2}\cos(\theta/2)
        \end{bmatrix}.

    :param para: representa o array de parametros.
    :param control_qubits: Control qubit index.
    :param rot_wire: Rot qubit index.
    :param q_machine: Quantum machine device.
    

    Exemplo::

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

    Circuito quantico de porta logica Hadamard controlada. Esta funcao tem o alias: ``VQC_Controlled_Hadamard`` .

    .. math:: 
        CH = \begin{bmatrix}
        1 & 0 & 0 & 0 \\
        0 & 1 & 0 & 0 \\
        0 & 0 & \frac{1}{\sqrt{2}} & \frac{1}{\sqrt{2}} \\
        0 & 0 & \frac{1}{\sqrt{2}} & -\frac{1}{\sqrt{2}}
        \end{bmatrix}.

    :param wires: lista de indices de bits quanticos, o primeiro eh o bit de controle, o comprimento da lista eh 2.
    :param q_machine: quantum virtual machine device.

    Exemplos::

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

    :param wires: lista de indices de bits quanticos, o primeiro eh o bit de controle. O comprimento da lista eh 3.
    :param q_machine: quantum virtual machine device.

    Exemplo::

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
    :param wires: Um subconjunto de indices de qubit no intervalo [r, p]. O comprimento minimo deve ser 2. O primeiro valor de indice eh interpretado como r, e o ultimo valor de indice eh interpretado como p. Os indices intermediarios sao afetados por portas CNOT para calcular a paridade do conjunto de qubits.
    :param q_machine: Quantum virtual machine device.

    

    Exemplos::

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

    Esta funcao tem o alias: ``VQC_FermionicDoubleExcitation`` .

    :param weight: variable parameter
    :param wires1: representa o subconjunto de qubits no intervalo de lista de indices [s, r]. O indice a eh interpretado como s e o ultimo indice eh interpretado como r. A porta CNOT opera nos indices do meio para calcular a paridade de um grupo de qubits.
    :param wires2: representa o subconjunto de qubits no intervalo de lista de indices [q, p]. O primeiro indice raiz eh interpretado como q e o ultimo indice eh interpretado como p. A porta CNOT opera nos indices do meio para calcular a paridade de um grupo de qubits.
    :param q_machine: Quantum virtual machine device.

    

    Exemplos::

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

    Dentro da aproximacao de Trotter de primeira ordem, a funcao unitaria UCCSD eh dada por:

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

    Esta funcao tem o alias: ``VQC_UCCSD`` .

    :param weights: tensor of size ``(len(s_wires)+ len(d_wires))`` containing the parameters :math:`\theta_{pr}` and :math:`\theta_{pqrs}` input Z rotations ``FermionicSingleExcitation`` and ``FermionicDoubleExcitation`` .
    :param wires: qubit indices for template action
    :param s_wires: sequence of lists containing qubit indices ``[r,...,p]`` generated by a single excitation :math:`\vert r, p \rangle = \hat{c}_p^\dagger \hat{c}_r \vert \mathrm{HF} \rangle`,where :math:`\vert \mathrm{HF} \rangle` denotes the Hartee-Fock reference state.
    :param d_wires: sequence of lists, each containing two lists specifying indices ``[s, ...,r]`` and ``[q,..., p]`` defining double excitation :math:`\vert s, r, q, p \rangle = \hat{c}_p^\dagger \hat{c}_q^\dagger \hat{c}_r\hat{c}_s \vert \mathrm{HF} \rangle` .
    :param init_state: occupation-number vector of length ``len(wires)`` representing the high-frequency state. ``init_state`` Initialization state of the qubit.
    :param q_machine: Quantum virtual machine device.

    Exemplos::

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

    A string de Pauli eh fixada em ``Z``. Portanto, a expansao de primeira ordem sera um circuito sem portas de entrelacamento.

    :param input_feat: Array representing input parameters.
    :param q_machine: Quantum virtual machine.
    :param data_map_func: Parameter mapping matrix, a callable function, designed as: ``data_map_func = lambda x: x``.
    :param rep: Numero de vezes que o modulo eh repetido.

    Exemplo::

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
    
    Exemplo::

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

    Neste caso, temos quatro excitacoes simples e duplas para preservar a projecao de spin total do estado de Hartree-Fock.

    The resulting unitary matrix preserves the particle population and prepares the n-qubit system in a superposition of the initial Hartree-Fock state and other states encoding the multi-excitation configuration.

    :param weights: Um QTensor de tamanho ``(len(singles) + len(doubles),)`` contendo os angulos que entram nas operacoes vqc.qCircuit.single_excitation e vqc.qCircuit.double_excitation em sequencia
    :param q_machine: A maquina quantica.
    :param hf_state: Um vetor de comprimento ``len(wires)`` de numeros de ocupacao representando o estado de Hartree-Fock, ``hf_state`` usado para inicializar os fios.
    :param wires: Os qubits sobre os quais atuar.
    :param singles: Uma sequencia de listas com os indices dos dois qubits sobre os quais a operacao single_excitation atua.
    :param doubles: Sequencia de lista com os indices dos dois qubits sobre os quais a operacao double_excitation atua.

    For example, the quantum circuit for two electrons and six qubits is shown below:

    .. image:: ./images/all_singles_doubles.png
        :width: 600 px
        :align: center

    |

    Exemplo::

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

    Implementa um circuito que fornece um conjunto que pode ser usado para realizar rotacoes precisas de base de unidade unica. O circuito eh derivado da transformacao unitaria fermionica de unica particula :math:`U(u)` dada em `arXiv:1711.04789 <https://arxiv.org/abs/1711.04789>`_\
    
    .. math::
        U(u) = \exp{\left( \sum_{pq} \left[\log u \right]_{pq} (a_p^\dagger a_q - a_q^\dagger a_p) \right)}.

    :math:`U(u)` is obtained by using the scheme given in the paper `Optica, 3, 1460 (2016) <https://opg.optica.org/optica/fulltext.cfm?uri=optica-3-12-1460&id=355743>`_\ .

    :param q_machine: quantum machine.
    :param wires: qubits to act on.
    :param unitary_matrix: matriz especificando a base para a transformacao.
    :param check: check if `unitary_matrix` is a unitary matrix.

    Exemplo::

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
    Esta classe chamara backend, rank, world_size para inicializar ``torch.distributed.init_process_group(backend, rank, world_size)`` .

    :param backend: used to generate cpu or gpu data communication controller, 'gloo' or 'nccl'.
    :param rank: the process number of the current program.
    :param world_size: the number of all global processes.

    :return:
        CommController instance.

    Exemplos::

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

        Exemplos::

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

        Exemplos::

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

        Exemplos::

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

        Exemplos::

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

        Exemplos::

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

        Exemplos::

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

        Exemplos::

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

        Exemplos::

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

        Reune todos os dados de todos os processos. Esta interface so suporta o backend nccl.

        :param tensor: Input data.

        Exemplos::

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

        Exemplos::

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

        Exemplos::

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

        Exemplos::

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

        Exemplos::

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

        Exemplos::
            
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

        Exemplos::
            
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

