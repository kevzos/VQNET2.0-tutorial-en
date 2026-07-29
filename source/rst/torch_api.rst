

.. _torch_api:

=============================================================
VQNet usa torch para el cálculo de bajo nivel
=============================================================

A partir de la versión 2.15.0, este software permite usar `torch` como motor de cálculo para operaciones de bajo nivel y puede integrarse con modelos, códigos y librerías de terceros basados en `torch` para desarrollo secundario.

    .. important::

        Para usar las siguientes funcionalidades, instale torch>=2.11.0 usted mismo. Si instala una versión GPU de torch, debe usar una versión compatible con CUDA 12.6; de lo contrario, torch podría no funcionar debido a problemas con la librería de tiempo de ejecución NVIDIA CUDA. Este software no instala torch automáticamente durante la instalación.

    .. note::

        Las funciones de computación cuántica variacional (con nombres en minúscula, como `rx`, `ry`, `rz`, etc.) en :ref:`vqc_api`, así como las funciones básicas de cálculo de QTensor en :ref:`qtensor_api`,
        pueden aceptar un `QTensor` como entrada después de llamar a ``pyvqnet.backends.set_backend("torch")``, con el miembro `data` de `QTensor` cambiando del Tensor de pyvqnet a ``torch.Tensor`` para el cálculo.

        ``pyvqnet.backends.set_backend("torch")`` y ``pyvqnet.backends.set_backend("pyvqnet")`` modifican el motor de cálculo global.
        Los objetos ``QTensor`` creados bajo diferentes configuraciones de motor no se pueden mezclar en los cálculos.

Configuración Básica del Motor
============================================

set_backend
------------------------------------------------

.. py:function:: pyvqnet.backends.set_backend(backend_name)

    Establece el motor para los cálculos actuales y el almacenamiento de datos. El valor predeterminado es "pyvqnet-ad", pero se puede configurar como "torch", "torch-native", "pyvqnet-ad".
    
    Después de llamar a ``pyvqnet.backends.set_backend("torch")``, la interfaz permanece sin cambios. La variable miembro ``data`` de ``QTensor`` de VQNet usa ``torch.Tensor`` para almacenar datos.
    :ref:`qtensor_api`, :ref:`vqc_api`, y las interfaces de ``pyvqnet.nn.torch`` aceptan ``QTensor`` como entrada y producen ``QTensor`` como salida.

    Después de llamar a ``pyvqnet.backends.set_backend("torch-native")``, las interfaces permanecen sin cambios: :ref:`qtensor_api`, :ref:`vqc_api`, y la interfaz `pyvqnet.nn.torch`.
    Las entradas pueden aceptar directamente tipos ``torch.Tensor`` o ``QTensor``, y las salidas son ``torch.Tensor``, eliminando la necesidad de conversión a ``QTensor``, reduciendo así la conversión de datos.
    
    Después de llamar a ``pyvqnet.backends.set_backend("pyvqnet")``, el miembro ``data`` de ``QTensor`` de VQNet almacenará datos usando ``pyvqnet._core.Tensor``, y los cálculos usarán la librería C++ de pyvqnet.

    Después de llamar a ``pyvqnet.backends.set_backend("pyvqnet-ad")``, el miembro ``data`` de ``QTensor`` de VQNet almacenará datos usando ``pyvqnet._core.Tensor``, y los cálculos usarán la librería C++ de pyvqnet con rendimiento mejorado.


    .. note::

        Esta función modifica el motor de cálculo actual. Los objetos ``QTensor`` creados bajo diferentes motores no se pueden usar juntos en los cálculos.

    :param backend_name: Nombre del motor, puede ser "pyvqnet" o "torch".

    Example::

        import pyvqnet
        pyvqnet.backends.set_backend("torch")

get_backend
-------------------------------

.. py:function:: pyvqnet.backends.get_backend(t=None)

    Si `t` es None, obtiene el motor de cálculo actual.
    Si `t` es un QTensor, devuelve el motor usado para crear el QTensor según su propiedad ``data``.
    Si el motor es "torch", devuelve el motor torchAPI de pyvqnet.
    Si el motor es "pyvqnet", simplemente devuelve "pyvqnet".
    
    :param t: El tensor actual (valor predeterminado: None).
    :return: El motor. Por defecto, devuelve "pyvqnet".

    Example::

        import pyvqnet
        pyvqnet.backends.set_backend("torch")
        pyvqnet.backends.get_backend()

Funciones de QTensor
====================

Después de configurar el motor como ``torch``:

.. code-block::

    import pyvqnet
    pyvqnet.backends.set_backend("torch")

Todas las funciones miembro, funciones de creación, funciones matemáticas, funciones lógicas, transformaciones matriciales, etc., en :ref:`qtensor_api` usarán torch para el cálculo. Se puede acceder a `QTensor.data` para obtener los datos de torch.

Módulos de Red Neuronal Clásica y Red Neuronal Cuántica Variacional
==========================================================================================

Clase Base
------------------------------------------------

TorchModule
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. py:class:: pyvqnet.nn.torch.TorchModule(*args, **kwargs)

    La clase base que define modelos cuando se usa el motor `torch`. Esta clase hereda tanto de ``pyvqnet.nn.Module`` como de ``torch.nn.Module``.
    Se puede agregar como submódulo a un torchmodel.

    .. note::

        Esta clase y sus clases derivadas solo son adecuadas para usar con ``pyvqnet.backends.set_backend("torch")``.
        No mezclar con el `Module` predeterminado de ``pyvqnet.nn``.
    
        Los datos en ``_buffers`` de esta clase son de tipo ``torch.Tensor``.
        Los datos en ``_parameters`` de esta clase son de tipo ``torch.nn.Parameter``.

    .. py:method:: pyvqnet.nn.torch.TorchModule.forward(x, *args, **kwargs)

        Función de cálculo forward abstracta para la clase TorchModule.

        :param x: QTensor de entrada.
        :param args: Argumentos variables sin palabra clave.
        :param kwargs: Argumentos variables con palabra clave.

        :return: QTensor de salida, con el `data` interno siendo un ``torch.Tensor``.

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

        Devuelve un diccionario que contiene el estado completo del módulo, incluyendo parámetros y valores de búfer.
        Las claves son los nombres de los parámetros y búferes correspondientes.

        :param destination: El diccionario donde almacenar los parámetros internos del módulo.
        :param prefix: Un prefijo usado para los nombres de los parámetros y búferes.

        :return: Un diccionario que contiene el estado completo del módulo.

        Example::

            from pyvqnet.nn.torch import Conv2D
            import pyvqnet
            pyvqnet.backends.set_backend("torch")
            test_conv = Conv2D(2,3,(3,3),(2,2),"same")
            print(test_conv.state_dict().keys())

    .. py:method:: pyvqnet.nn.torch.TorchModule.load_state_dict(state_dict, strict=True)

        Copia los parámetros y búferes del :attr:`state_dict` a este módulo y sus submódulos.

        :param state_dict: Un diccionario que contiene parámetros y búferes persistentes.
        :param strict: Si se debe exigir que las claves del state_dict coincidan con las del `state_dict()` del modelo. Valor predeterminado: True.

        :return: Un mensaje de error si hay algún problema.

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

        Mueve el módulo y los datos de parámetros y búferes de sus submódulos al dispositivo GPU especificado.

        El dispositivo especifica dónde se almacenan los datos internos. Cuando device >= DEV_GPU_0, los datos se almacenan en la GPU.
        Si su computadora tiene múltiples GPUs, puede especificar diferentes dispositivos para almacenar datos. Por ejemplo, device = DEV_GPU_1, DEV_GPU_2, DEV_GPU_3, ... se refiere al almacenamiento en GPUs con diferentes números de serie.
        
        .. note::

            Los módulos no pueden realizar cálculos entre diferentes GPUs.
            Si intenta crear un QTensor en un ID de GPU que supera el máximo permitido para validación, se generará un error de Cuda.

        :param device: El dispositivo donde almacenar el QTensor. Valor predeterminado: DEV_GPU_0. device = pyvqnet.DEV_GPU_0 almacena en la primera GPU, device = DEV_GPU_1 almacena en la segunda GPU, y así sucesivamente.
        :return: El módulo movido al dispositivo GPU.

        Examples::

            from pyvqnet.nn.torch import ConvT2D
            import pyvqnet
            pyvqnet.backends.set_backend("torch")
            test_conv = ConvT2D(3, 2, [4, 4], [2, 2], (0, 0))
            test_conv = test_conv.toGPU()
            print(test_conv.backend)
            #1000

    .. py:method:: pyvqnet.torch.TorchModule.toCPU()

        Mueve el módulo y los datos de parámetros y búferes de sus submódulos a un dispositivo CPU específico.

        :return: El módulo movido al dispositivo CPU.

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

    Este módulo se usa para almacenar instancias hijas de ``TorchModule`` en una lista. TorchModuleList se puede indexar como una lista normal de Python, y los parámetros internos que contiene se pueden guardar.
    
    Esta clase hereda de ``pyvqnet.nn.torch.TorchModule`` y ``pyvqnet.nn.ModuleList``, y se puede agregar como submódulo a un torchmodel.

    :param modules: Una lista de ``pyvqnet.nn.torch.TorchModule``

    :return: Una clase TorchModuleList

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

    Este módulo se usa para almacenar instancias hijas de ``pyvqnet.nn.Parameter`` en una lista. TorchParameterList se puede indexar como una lista normal de Python, y los parámetros internos que contiene se pueden guardar.
    
    Esta clase hereda de ``pyvqnet.nn.torch.TorchModule`` y ``pyvqnet.nn.ParameterList``, y se puede agregar como submódulo a un torchmodel.

    :param value: Una lista de ``nn.Parameter``

    :return: Una clase TorchParameterList

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
                # ParameterList puede actuar como iterable, o indexarse con enteros
                for i, p in enumerate(self.params):
                    x = self.params[i // 2] * x + p * x
                return x

        model = MyModule()
        print(model.state_dict().keys())

TorchSequential
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. py:class:: pyvqnet.nn.torch.TorchSequential(*args)

    Este módulo agrega módulos en el orden en que se pasan. Alternativamente, puede pasar un ``OrderedDict`` de módulos. El método ``forward()`` de la clase ``Sequential`` acepta cualquier entrada y la reenvía a su primer módulo.
    La salida se vincula secuencialmente a la entrada de cada módulo subsiguiente, siendo la salida final el resultado del último módulo.

    Esta clase hereda de ``pyvqnet.nn.torch.TorchModule`` y ``pyvqnet.nn.Sequential``, y se puede agregar como submódulo a un torchmodel.

    :param args: Módulos a agregar

    :return: Una clase TorchSequential

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

Guardar y Cargar Parámetros del Modelo
--------------------------------------------

Puede usar ``save_parameters`` y ``load_parameters`` de :ref:`save_parameters` para guardar los parámetros de un modelo ``TorchModule`` como un diccionario en un archivo, con los valores guardados como `numpy.ndarray`. Alternativamente, puede cargar el archivo de parámetros desde el disco. Tenga en cuenta que la estructura del modelo no se guarda en el archivo, y deberá reconstruir manualmente la estructura del modelo. También puede usar directamente ``torch.save`` y ``torch.load`` para leer los parámetros del modelo ``torch`` ya que los parámetros de ``TorchModule`` se almacenan como objetos ``torch.Tensor``.


Módulos de Red Neuronal Clásica
--------------------------------------------

Los siguientes módulos de red neuronal clásica heredan todos de ``pyvqnet.nn.Module`` y ``torch.nn.Module``, y se pueden agregar como submódulos a un torchmodel.

Linear
^^^^^^^^^^^^^^^^^^^^^^^^

.. py:class:: pyvqnet.nn.torch.Linear(input_channels, output_channels, weight_initializer=None, bias_initializer=None, use_bias=True, dtype=None, name: str = "")

    Un módulo lineal (capa completamente conectada), :math:`y = x@A.T + b`.
    Esta clase hereda de ``pyvqnet.nn.Module`` y ``torch.nn.Module`` y se puede usar como submódulo de un torchmodel.

    Los datos en los ``_buffers`` de la clase son de tipo ``torch.Tensor``.
    Los datos en los ``_parameters`` de la clase son de tipo ``torch.nn.Parameter``.

    :param input_channels: `int` - Número de canales de entrada.
    :param output_channels: `int` - Número de canales de salida.
    :param weight_initializer: `callable` - Función de inicialización de pesos, valor predeterminado vacío, usando he_uniform.
    :param bias_initializer: `callable` - Función de inicialización de sesgo, valor predeterminado vacío, usando he_uniform.
    :param use_bias: `bool` - Si usar el término de sesgo, valor predeterminado True.
    :param dtype: Tipo de datos para los parámetros, predeterminado None, usa el tipo de datos predeterminado `kfloat32`, que representa números de punto flotante de 32 bits.
    :param name: El nombre de la capa lineal, valor predeterminado "".

    :return: Una instancia de la capa Linear.

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

    Realiza convolución 1D en la entrada. La entrada al módulo Conv1D tiene la forma (batch_size, input_channels, in_height).
    Esta clase hereda de ``pyvqnet.nn.Module`` y ``torch.nn.Module``, y se puede usar como submódulo de un torchmodel.

    Los datos en los ``_buffers`` de la clase son de tipo ``torch.Tensor``.
    Los datos en los ``_parameters`` de la clase son de tipo ``torch.nn.Parameter``.

    :param input_channels: `int` - Número de canales de entrada.
    :param output_channels: `int` - Número de canales de salida.
    :param kernel_size: `int` - Tamaño del kernel de convolución. La forma del kernel es [output_channels, input_channels/group, kernel_size, 1].
    :param stride: `int` - El paso (stride), valor predeterminado 1.
    :param padding: `str|int` - Opciones de relleno, puede ser una cadena {'valid', 'same'} o un entero que especifica la cantidad de relleno a aplicar a la entrada. Valor predeterminado "valid".
    :param use_bias: `bool` - Si usar el término de sesgo, valor predeterminado True.
    :param kernel_initializer: `callable` - Método de inicialización del kernel de convolución. Valor predeterminado vacío, usando kaiming_uniform.
    :param bias_initializer: `callable` - Método de inicialización del sesgo. Valor predeterminado vacío, usando kaiming_uniform.
    :param dilation_rate: `int` - Tamaño de dilatación, valor predeterminado 1.
    :param group: `int` - Número de grupos en la convolución agrupada. Valor predeterminado 1.
    :param dtype: Tipo de datos para los parámetros, predeterminado None, usa el tipo de datos predeterminado `kfloat32`, que representa números de punto flotante de 32 bits.
    :param name: El nombre del módulo, valor predeterminado "".

    :return: Una instancia de la convolución 1D.

    .. note::

        ``padding='valid'`` no aplica relleno.

        ``padding='same'`` aplica relleno con ceros a la entrada, con el `out_height` de salida igual a `ceil(in_height / stride)`, y no soporta `stride > 1`.

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

    Realiza convolución 2D en la entrada. La entrada al módulo Conv2D tiene la forma (batch_size, input_channels, height, width).
    Esta clase hereda de ``pyvqnet.nn.Module`` y ``torch.nn.Module``, y se puede usar como submódulo de un torchmodel.

    Los datos en los ``_buffers`` de la clase son de tipo ``torch.Tensor``.
    Los datos en los ``_parameters`` de la clase son de tipo ``torch.nn.Parameter``.

    :param input_channels: `int` - Número de canales de entrada.
    :param output_channels: `int` - Número de canales de salida.
    :param kernel_size: `tuple|list` - Tamaño del kernel de convolución. La forma del kernel es [output_channels, input_channels/group, kernel_size, kernel_size].
    :param stride: `tuple|list` - El paso (stride), valor predeterminado (1, 1).
    :param padding: `str|tuple` - Opciones de relleno, puede ser una cadena {'valid', 'same'} o una tupla que especifica el relleno a aplicar en ambos lados. Valor predeterminado "valid".
    :param use_bias: `bool` - Si usar el término de sesgo, valor predeterminado True.
    :param kernel_initializer: `callable` - Método de inicialización del kernel de convolución. Valor predeterminado vacío, usando kaiming_uniform.
    :param bias_initializer: `callable` - Método de inicialización del sesgo. Valor predeterminado vacío, usando kaiming_uniform.
    :param dilation_rate: `int` - Tamaño de dilatación, valor predeterminado 1.
    :param group: `int` - Número de grupos en la convolución agrupada. Valor predeterminado 1.
    :param dtype: Tipo de datos para los parámetros, predeterminado None, usa el tipo de datos predeterminado `kfloat32`, que representa números de punto flotante de 32 bits.
    :param name: El nombre del módulo, valor predeterminado "".

    :return: Una instancia de la convolución 2D.

    .. note::

        ``padding='valid'`` no aplica relleno.

        ``padding='same'`` aplica relleno con ceros a la entrada, con la altura de salida igual a `ceil(in_height / stride)`.

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

    Realiza convolución transpuesta 2D en la entrada. La entrada al módulo ConvT2D tiene la forma (batch_size, input_channels, height, width).
    Esta clase hereda de ``pyvqnet.nn.Module`` y ``torch.nn.Module``, y se puede usar como submódulo de un torchmodel.

    Los datos en los ``_buffers`` de la clase son de tipo ``torch.Tensor``.
    Los datos en los ``_parameters`` de la clase son de tipo ``torch.nn.Parameter``.

    :param input_channels: `int` - Número de canales de entrada.
    :param output_channels: `int` - Número de canales de salida.
    :param kernel_size: `tuple|list` - Tamaño del kernel de convolución, con forma del kernel = [input_channels, output_channels/group, kernel_size, kernel_size].
    :param stride: `tuple|list` - El paso (stride), valor predeterminado (1, 1).
    :param padding: `tuple` - Opciones de relleno, una tupla que especifica el relleno a aplicar en ambos lados. Valor predeterminado (0, 0).
    :param use_bias: `bool` - Si usar el término de sesgo, valor predeterminado True.
    :param kernel_initializer: `callable` - Método de inicialización del kernel de convolución. Valor predeterminado vacío, usando kaiming_uniform.
    :param bias_initializer: `callable` - Método de inicialización del sesgo. Valor predeterminado vacío, usando kaiming_uniform.
    :param dilation_rate: `int` - Tamaño de dilatación, valor predeterminado 1.
    :param out_padding: Tamaño extra agregado a la forma de salida para cada dimensión. Valor predeterminado (0, 0).
    :param group: `int` - Número de grupos en la convolución agrupada. Valor predeterminado 1.
    :param dtype: Tipo de datos para los parámetros, predeterminado None, usa el tipo de datos predeterminado `kfloat32`, que representa números de punto flotante de 32 bits.
    :param name: El nombre del módulo, valor predeterminado "".

    :return: Una instancia de la convolución transpuesta 2D.

    .. note::

        ``padding='valid'`` no aplica relleno.

        ``padding='same'`` aplica relleno con ceros a la entrada, con la altura de salida igual a `ceil(height / stride)`.

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

    Realiza pooling promedio en entrada 1D. La entrada tiene la forma (batch_size, input_channels, in_height).
    Esta clase hereda de ``pyvqnet.nn.Module`` y ``torch.nn.Module``, y se puede agregar como submódulo a un torchmodel.

    Los datos en los ``_buffers`` de la clase son de tipo ``torch.Tensor``.
    Los datos en los ``_parameters`` de la clase son de tipo ``torch.nn.Parameter``.

    :param kernel: El tamaño de la ventana de pooling.
    :param stride: El tamaño de paso para mover la ventana.
    :param padding: Opción de relleno, un entero que especifica la longitud del relleno. Valor predeterminado 0.
    :param name: El nombre del módulo, valor predeterminado "".

    :return: Una instancia de la capa de pooling promedio 1D.

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

    Realiza pooling máximo en entrada 1D. La entrada tiene la forma (batch_size, input_channels, in_height).
    Esta clase hereda de ``pyvqnet.nn.Module`` y ``torch.nn.Module``, y se puede agregar como submódulo a un torchmodel.

    Los datos en los ``_buffers`` de la clase son de tipo ``torch.Tensor``.
    Los datos en los ``_parameters`` de la clase son de tipo ``torch.nn.Parameter``.

    :param kernel: El tamaño de la ventana de pooling.
    :param stride: El tamaño de paso para mover la ventana.
    :param padding: Opción de relleno, un entero que especifica la longitud del relleno. Valor predeterminado 0.
    :param name: El nombre del módulo, valor predeterminado "".

    :return: Una instancia de la capa de pooling máximo 1D.

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

    Realiza pooling promedio en entrada 2D. La entrada tiene la forma (batch_size, input_channels, height, width).
    Esta clase hereda de ``pyvqnet.nn.Module`` y ``torch.nn.Module``, y se puede agregar como submódulo a un torchmodel.

    Los datos en los ``_buffers`` de la clase son de tipo ``torch.Tensor``.
    Los datos en los ``_parameters`` de la clase son de tipo ``torch.nn.Parameter``.

    :param kernel: El tamaño de la ventana de pooling.
    :param stride: El tamaño de paso para mover la ventana.
    :param padding: Opción de relleno, una tupla que contiene dos enteros especificando el relleno para ambas dimensiones. Valor predeterminado (0,0).
    :param name: El nombre del módulo, valor predeterminado "".

    :return: Una instancia de la capa de pooling promedio 2D.

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

    Realiza pooling máximo en entrada 2D. La entrada tiene la forma (batch_size, input_channels, height, width).
    Esta clase hereda de ``pyvqnet.nn.Module`` y ``torch.nn.Module``, y se puede agregar como submódulo a un torchmodel.

    Los datos en los ``_buffers`` de la clase son de tipo ``torch.Tensor``.
    Los datos en los ``_parameters`` de la clase son de tipo ``torch.nn.Parameter``.

    :param kernel: El tamaño de la ventana de pooling.
    :param stride: El tamaño de paso para mover la ventana.
    :param padding: Opción de relleno, una tupla que contiene dos enteros especificando el relleno para ambas dimensiones. Valor predeterminado (0,0).
    :param name: El nombre del módulo, valor predeterminado "".

    :return: Una instancia de la capa de pooling máximo 2D.

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

    Este módulo se usa típicamente para almacenar embeddings de palabras y recuperarlas usando índices. La entrada al módulo es una lista de índices, y la salida son los embeddings de palabras correspondientes.
    La entrada a esta capa debe ser de tipo `kint64`. 
    Esta clase hereda de ``pyvqnet.nn.Module`` y ``torch.nn.Module``, y se puede agregar como submódulo a un torchmodel.

    Los datos en los ``_buffers`` de la clase son de tipo ``torch.Tensor``.
    Los datos en los ``_parameters`` de la clase son de tipo ``torch.nn.Parameter``.

    :param num_embeddings: `int` - El tamaño del diccionario de embeddings.
    :param embedding_dim: `int` - El tamaño de cada vector de embedding.
    :param weight_initializer: `callable` - El método de inicialización de pesos, valor predeterminado Xavier Normal.
    :param dtype: El tipo de datos para los parámetros, predeterminado None, que usa el tipo de datos predeterminado: `kfloat32` (punto flotante de 32 bits).
    :param name: El nombre de la capa de embedding, valor predeterminado "".

    :return: Una instancia de la capa Embedding.

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

    Aplica normalización por lotes en entrada 4D (B, C, H, W). Consulte el artículo
    `Batch Normalization: Accelerating Deep Network Training by Reducing
    Internal Covariate Shift <https://arxiv.org/abs/1502.03167>`__.

    Esta clase hereda de ``pyvqnet.nn.Module`` y ``torch.nn.Module``, y se puede agregar como submódulo a un torchmodel.

    Los datos en los ``_buffers`` de la clase son de tipo ``torch.Tensor``.
    Los datos en los ``_parameters`` de la clase son de tipo ``torch.nn.Parameter``.

    .. math::
        y = \frac{x - \mathrm{E}[x]}{\sqrt{\mathrm{Var}[x] + \epsilon}} * \gamma + \beta

    donde :math:`\gamma` y :math:`\beta` son parámetros entrenables. Además, por defecto, durante el entrenamiento, la capa continúa estimando la media y la varianza, que luego se usan para la normalización durante la evaluación. El momentum para los promedios móviles se establece en el valor predeterminado de 0.1.

    :param channel_num: `int` - Número de canales de entrada.
    :param momentum: `float` - Momentum para el cálculo del promedio móvil, valor predeterminado 0.1.
    :param epsilon: `float` - Una pequeña constante para estabilidad numérica, valor predeterminado 1e-5.
    :param affine: `bool` - Si incluir parámetros afines entrenables para cada canal. Valor predeterminado `True`, que inicializa los parámetros como 1 para pesos y 0 para sesgos.
    :param beta_initializer: `callable` - Método de inicialización para beta, valor predeterminado inicialización cero.
    :param gamma_initializer: `callable` - Método de inicialización para gamma, valor predeterminado inicialización uno.
    :param dtype: Tipo de datos para los parámetros, predeterminado None, usando `kfloat32` (punto flotante de 32 bits).
    :param name: El nombre de la capa de normalización por lotes, valor predeterminado "".

    :return: Una instancia de la capa de normalización por lotes 2D.

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

    Aplica normalización por lotes en entrada 2D (B, C). Consulte el artículo
    `Batch Normalization: Accelerating Deep Network Training by Reducing
    Internal Covariate Shift <https://arxiv.org/abs/1502.03167>`__.

    .. math::
        y = \frac{x - \mathrm{E}[x]}{\sqrt{\mathrm{Var}[x] + \epsilon}} * \gamma + \beta

    donde :math:`\gamma` y :math:`\beta` son parámetros entrenables. Además, por defecto, durante el entrenamiento, la capa continúa estimando la media y la varianza, que luego se usan para la normalización durante la evaluación. El momentum para los promedios móviles se establece en el valor predeterminado de 0.1.

    Esta clase hereda de ``pyvqnet.nn.Module`` y ``torch.nn.Module``, y se puede agregar como submódulo a un torchmodel.

    Los datos en los ``_buffers`` de la clase son de tipo ``torch.Tensor``.
    Los datos en los ``_parameters`` de la clase son de tipo ``torch.nn.Parameter``.

    :param channel_num: `int` - Número de canales de entrada.
    :param momentum: `float` - Momentum para el cálculo del promedio móvil, valor predeterminado 0.1.
    :param epsilon: `float` - Una pequeña constante para estabilidad numérica, valor predeterminado 1e-5.
    :param affine: `bool` - Si incluir parámetros afines entrenables para cada canal. Valor predeterminado `True`, que inicializa los parámetros como 1 para pesos y 0 para sesgos.
    :param beta_initializer: `callable` - Método de inicialización para beta, valor predeterminado inicialización cero.
    :param gamma_initializer: `callable` - Método de inicialización para gamma, valor predeterminado inicialización uno.
    :param dtype: Tipo de datos para los parámetros, predeterminado None, usando `kfloat32` (punto flotante de 32 bits).
    :param name: El nombre de la capa de normalización por lotes, valor predeterminado "".

    :return: Una instancia de la capa de normalización por lotes 1D.

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

    Aplica normalización de capa en las últimas D dimensiones de cualquier entrada. El método específico se describe en el artículo:
    `Layer Normalization <https://arxiv.org/abs/1607.06450>`__.

    .. math::
        y = \frac{x - \mathrm{E}[x]}{ \sqrt{\mathrm{Var}[x] + \epsilon}} * \gamma + \beta

    Para entradas como (B, C, H, W, D), ``norm_shape`` puede ser [C, H, W, D], [H, W, D], [W, D] o [D].

    Esta clase hereda de ``pyvqnet.nn.Module`` y ``torch.nn.Module``, y se puede agregar como submódulo a un torchmodel.

    Los datos en los ``_buffers`` de la clase son de tipo ``torch.Tensor``.
    Los datos en los ``_parameters`` de la clase son de tipo ``torch.nn.Parameter``.

    :param norm_shape: `list` - La forma a normalizar.
    :param epsilon: `float` - Una pequeña constante para estabilidad numérica, valor predeterminado 1e-5.
    :param affine: `bool` - Si es `True`, este módulo tiene parámetros afines entrenables para cada canal, inicializados en 1 (para pesos) y 0 (para sesgos). Valor predeterminado `True`.
    :param dtype: Tipo de datos para los parámetros, predeterminado None, usando `kfloat32` (punto flotante de 32 bits).
    :param name: El nombre del módulo, valor predeterminado "".

    :return: Una instancia de la clase LayerNormNd.

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

    Aplica normalización de capa en entradas 4D. El método específico se describe en el artículo:
    `Layer Normalization <https://arxiv.org/abs/1607.06450>`__.

    .. math::
        y = \frac{x - \mathrm{E}[x]}{ \sqrt{\mathrm{Var}[x] + \epsilon}} * \gamma + \beta

    La media y la desviación estándar se calculan en las dimensiones restantes, excluyendo la primera. Para entradas como (B, C, H, W), ``norm_size`` debe ser igual a C * H * W.

    Esta clase hereda de ``pyvqnet.nn.Module`` y ``torch.nn.Module``, y se puede agregar como submódulo a un torchmodel.

    Los datos en los ``_buffers`` de la clase son de tipo ``torch.Tensor``.
    Los datos en los ``_parameters`` de la clase son de tipo ``torch.nn.Parameter``.

    :param norm_size: `int` - El tamaño de la normalización, debe ser igual a C * H * W.
    :param epsilon: `float` - Una pequeña constante para estabilidad numérica, valor predeterminado 1e-5.
    :param affine: `bool` - Si es `True`, este módulo tiene parámetros afines entrenables para cada canal, inicializados en 1 (para pesos) y 0 (para sesgos). Valor predeterminado `True`.
    :param dtype: Tipo de datos para los parámetros, predeterminado None, usando `kfloat32` (punto flotante de 32 bits).
    :param name: El nombre del módulo, valor predeterminado "".

    :return: Una instancia de la normalización de capa 2D.

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

    Aplica normalización de capa en entradas 2D. El método específico se describe en el artículo:
    `Layer Normalization <https://arxiv.org/abs/1607.06450>`__.

    .. math::
        y = \frac{x - \mathrm{E}[x]}{ \sqrt{\mathrm{Var}[x] + \epsilon}} * \gamma + \beta

    La media y la desviación estándar se calculan en el tamaño de la última dimensión, donde ``norm_size`` es el valor de la última dimensión.

    Esta clase hereda de ``pyvqnet.nn.Module`` y ``torch.nn.Module``, y se puede agregar como submódulo a un torchmodel.

    Los datos en los ``_buffers`` de la clase son de tipo ``torch.Tensor``.
    Los datos en los ``_parameters`` de la clase son de tipo ``torch.nn.Parameter``.

    :param norm_size: `int` - El tamaño de la normalización, debe ser igual al tamaño de la última dimensión.
    :param epsilon: `float` - Una pequeña constante para estabilidad numérica, valor predeterminado 1e-5.
    :param affine: `bool` - Si es `True`, este módulo tiene parámetros afines entrenables para cada canal, inicializados en 1 (para pesos) y 0 (para sesgos). Valor predeterminado `True`.
    :param dtype: Tipo de datos para los parámetros, predeterminado None, usando `kfloat32` (punto flotante de 32 bits).
    :param name: El nombre del módulo, valor predeterminado "".

    :return: Una instancia de la normalización de capa 1D.

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

    Aplica normalización de grupo en entradas de mini-lote. Entrada: :math:`(N, C, *)` donde :math:`C=\mathrm{num\_channels}`, Salida: :math:`(N, C, *)`.

    Esta capa implementa la operación descrita en el artículo `Group Normalization <https://arxiv.org/abs/1803.08494>`__.

    .. math::
        
        y = \frac{x - \mathrm{E}[x]}{ \sqrt{\mathrm{Var}[x] + \epsilon}} * \gamma + \beta

    Los canales de entrada se dividen en :attr:`num_groups` grupos, cada uno conteniendo ``num_channels / num_groups`` canales. :attr:`num_channels` debe ser divisible por :attr:`num_groups`. La media y la desviación estándar se calculan por separado dentro de cada grupo. Si :attr:`affine` es ``True``, entonces :math:`\gamma` y :math:`\beta` son entrenables. Los parámetros de transformación afín para cada canal son vectores de tamaño :attr:`num_channels`.

    Esta clase hereda de ``pyvqnet.nn.Module`` y ``torch.nn.Module``, y se puede agregar como submódulo a un torchmodel.

    Los datos en los ``_buffers`` de la clase son de tipo ``torch.Tensor``.
    Los datos en los ``_parameters`` de la clase son de tipo ``torch.nn.Parameter``.

    :param num_groups (int): El número de grupos en los que dividir los canales.
    :param num_channels (int): El número de canales de entrada esperados.
    :param epsilon: Un valor pequeño añadido al denominador para estabilidad numérica. Valor predeterminado 1e-5.
    :param affine: Un valor booleano. Si se establece en ``True``, este módulo tiene parámetros afines entrenables para cada canal, inicializados en 1 (para pesos) y 0 (para sesgos). Valor predeterminado ``True``.
    :param dtype: El tipo de datos para los parámetros. Predeterminado None, usando `kfloat32` (punto flotante de 32 bits).
    :param name: El nombre del módulo. Valor predeterminado "".

    :return: Una instancia de la clase GroupNorm.

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

    Módulo Dropout. El módulo dropout establece aleatoriamente la salida de algunas unidades a cero, mientras escala las unidades restantes según la probabilidad dropout_rate.

    Esta clase hereda de ``pyvqnet.nn.Module`` y ``torch.nn.Module``, y se puede agregar como submódulo a un torchmodel.

    :param dropout_rate: `float` - La probabilidad de establecer neuronas a cero.
    :param name: El nombre del módulo. Valor predeterminado "".

    :return: Una instancia de la clase Dropout.

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

    El módulo DropPath aplica dropout de ruta de muestra aleatoria (profundidad aleatoria).

    Esta clase hereda de ``pyvqnet.nn.Module`` y ``torch.nn.Module``, y se puede agregar como submódulo a un torchmodel.

    :param dropout_rate: `float` - La probabilidad de establecer neuronas a cero.
    :param name: El nombre del módulo. Valor predeterminado "".

    :return: Una instancia de la clase DropPath.

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

    Reorganiza un tensor de forma: (*, C * r^2, H, W) a un tensor de forma (*, C, H * r, W * r), donde r es el factor de escala.

    Esta clase hereda de ``pyvqnet.nn.Module`` y ``torch.nn.Module``, y se puede agregar como submódulo a un torchmodel.

    :param upscale_factors: El factor de escala para la transformación.
    :param name: El nombre del módulo. Valor predeterminado "".

    :return: Una instancia del módulo Pixel_Shuffle.

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

    Invierte la operación Pixel_Shuffle reorganizando elementos. Transforma un tensor de forma (*, C, H * r, W * r) a (*, C * r^2, H, W), donde r es el factor de reducción de escala.

    Esta clase hereda de ``pyvqnet.nn.Module`` y ``torch.nn.Module``, y se puede agregar como submódulo a un torchmodel.

    :param downscale_factors: El factor de reducción de escala para la transformación.
    :param name: El nombre del módulo. Valor predeterminado "".

    :return: Una instancia del módulo Pixel_Unshuffle.

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

    Módulo de Unidad Recurrente Cerrada (GRU). Soporta apilamiento multicapa y configuración bidireccional. La fórmula para una GRU unidireccional de una sola capa es:

    .. math::
        \begin{array}{ll}
            r_t = \sigma(W_{ir} x_t + b_{ir} + W_{hr} h_{(t-1)} + b_{hr}) \\
            z_t = \sigma(W_{iz} x_t + b_{iz} + W_{hz} h_{(t-1)} + b_{hz}) \\
            n_t = \tanh(W_{in} x_t + b_{in} + r_t * (W_{hn} h_{(t-1)}+ b_{hn})) \\
            h_t = (1 - z_t) * n_t + z_t * h_{(t-1)}
        \end{array}

    Esta clase hereda de ``pyvqnet.nn.Module`` y ``torch.nn.Module``, y se puede agregar como submódulo a un torchmodel.

    Los ``_buffers`` de la clase contienen datos ``torch.Tensor``, y los ``_parameters`` de la clase contienen datos ``torch.nn.Parameter``.

    :param input_size: La dimensión de la característica de entrada.
    :param hidden_size: La dimensión de la característica oculta.
    :param num_layers: El número de capas GRU apiladas, valor predeterminado: 1.
    :param batch_first: Si es True, la forma de entrada es [batch_size, seq_len, feature_dim]; si es False, la forma es [seq_len, batch_size, feature_dim], valor predeterminado: True.
    :param use_bias: Si es False, el módulo no usa términos de sesgo, valor predeterminado: True.
    :param bidirectional: Si es True, hace la GRU bidireccional, valor predeterminado: False.
    :param dtype: El tipo de datos de los parámetros, predeterminado None, usando el tipo de datos predeterminado: kfloat32 (float de 32 bits).
    :param name: El nombre del módulo, valor predeterminado: "".

    :return: Una instancia del módulo GRU.

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

    Módulo de Red Neuronal Recurrente (RNN), usando :math:`\tanh` o :math:`\text{ReLU}` como función de activación. Soporta configuraciones bidireccionales y multicapa. La fórmula para una RNN unidireccional de una sola capa es:

    .. math::
        h_t = \tanh(W_{ih} x_t + b_{ih} + W_{hh} h_{(t-1)} + b_{hh})

    Si :attr:`nonlinearity` es ``'relu'``, entonces :math:`\text{ReLU}` reemplazará a :math:`\tanh`.

    Esta clase hereda de ``pyvqnet.nn.Module`` y ``torch.nn.Module``, y se puede agregar como submódulo a un torchmodel.

    Los ``_buffers`` de la clase contienen datos ``torch.Tensor``, y los ``_parameters`` de la clase contienen datos ``torch.nn.Parameter``.

    :param input_size: La dimensión de la característica de entrada.
    :param hidden_size: La dimensión de la característica oculta.
    :param num_layers: El número de capas RNN apiladas, valor predeterminado: 1.
    :param nonlinearity: La función de activación no lineal, valor predeterminado: ``'tanh'``.
    :param batch_first: Si es True, la forma de entrada es [batch_size, seq_len, feature_dim]; si es False, la forma es [seq_len, batch_size, feature_dim], valor predeterminado: True.
    :param use_bias: Si es False, el módulo no usa términos de sesgo, valor predeterminado: True.
    :param bidirectional: Si es True, hace la RNN bidireccional, valor predeterminado: False.
    :param dtype: El tipo de datos de los parámetros, predeterminado None, usando el tipo de datos predeterminado: kfloat32 (float de 32 bits).
    :param name: El nombre del módulo, valor predeterminado: "".

    :return: Una instancia del módulo RNN.

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

    Módulo de Memoria a Largo Plazo (LSTM). Soporta LSTM bidireccional y configuraciones LSTM apiladas multicapa. La fórmula para una LSTM unidireccional de una sola capa es la siguiente:

    .. math::
        \begin{array}{ll} \\
            i_t = \sigma(W_{ii} x_t + b_{ii} + W_{hi} h_{t-1} + b_{hi}) \\
            f_t = \sigma(W_{if} x_t + b_{if} + W_{hf} h_{t-1} + b_{hf}) \\
            g_t = \tanh(W_{ig} x_t + b_{ig} + W_{hg} h_{t-1} + b_{hg}) \\
            o_t = \sigma(W_{io} x_t + b_{io} + W_{ho} h_{t-1} + b_{ho}) \\
            c_t = f_t \odot c_{t-1} + i_t \odot g_t \\
            h_t = o_t \odot \tanh(c_t) \\
        \end{array}

    Esta clase hereda de ``pyvqnet.nn.Module`` y ``torch.nn.Module``, y se puede agregar como submódulo a un torchmodel.

    Los ``_buffers`` de la clase contienen datos ``torch.Tensor``, y los ``_parameters`` de la clase contienen datos ``torch.nn.Parameter``.

    :param input_size: La dimensión de la característica de entrada.
    :param hidden_size: La dimensión de la característica oculta.
    :param num_layers: El número de capas LSTM apiladas, valor predeterminado: 1.
    :param batch_first: Si es True, la forma de entrada es [batch_size, seq_len, feature_dim]; si es False, la forma es [seq_len, batch_size, feature_dim], valor predeterminado: True.
    :param use_bias: Si es False, el módulo no usa términos de sesgo, valor predeterminado: True.
    :param bidirectional: Si es True, hace la LSTM bidireccional, valor predeterminado: False.
    :param dtype: El tipo de datos de los parámetros, predeterminado None, usando el tipo de datos predeterminado: kfloat32 (float de 32 bits).
    :param name: El nombre del módulo, valor predeterminado: "".

    :return: Una instancia del módulo LSTM.

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

    Aplica una RNN de Unidad Recurrente Cerrada (GRU) multicapa a secuencias de entrada de longitud dinámica.

    La primera entrada debe ser una entrada de secuencia por lotes con longitud variable definida
    a través de una clase ``tensor.PackedSequence``.

    La clase ``tensor.PackedSequence`` se puede construir
    llamando a las siguientes funciones consecutivamente: ``pad_sequence``, ``pack_pad_sequence``.

    La primera salida de Dynamic_GRU también es una clase ``tensor.PackedSequence``,
    que se puede desempaquetar a un QTensor normal usando ``tensor.pad_pack_sequence``.

    Para cada elemento en la secuencia de entrada, cada capa calcula la siguiente fórmula:

    .. math::
        \begin{array}{ll}
        r_t = \sigma(W_{ir} x_t + b_{ir} + W_{hr} h_{(t-1)} + b_{hr}) \\
        z_t = \sigma(W_{iz} x_t + b_{iz} + W_{hz} h_{(t-1)} + b_{hz}) \\
        n_t = \tanh(W_{in} x_t + b_{in} + r_t * (W_{hn} h_{(t-1)}+ b_{hn})) \\
        h_t = (1 - z_t) * n_t + z_t * h_{(t-1)}
        \end{array}

    Esta clase hereda de ``pyvqnet.nn.Module`` y ``torch.nn.Module`` y se puede agregar al modelo torch como submódulo de ``torch.nn.Module``.

    Los datos en ``_buffers`` de esta clase son de tipo ``torch.Tensor``.

    Los datos en ``_parameters`` de esta clase son de tipo ``torch.nn.Parameter``.

    :param input_size: Dimensión de la característica de entrada.
    :param hidden_size: Dimensión de la característica oculta.
    :param num_layers: Número de capas de bucle. Valor predeterminado: 1
    :param batch_first: Si es True, la forma de entrada se proporciona como [tamaño de lote, longitud de secuencia, dimensión de característica]. Si es False, la forma de entrada se proporciona como [longitud de secuencia, tamaño de lote, dimensión de característica]. Valor predeterminado: True.
    :param use_bias: Si es False, los pesos de sesgo b_ih y b_hh no se usan para esta capa. Valor predeterminado: True.
    :param bidirectional: Si es true, se convierte en una GRU bidireccional. Valor predeterminado: False.
    :param dtype: El tipo de datos del parámetro, predeterminado: None, use el tipo de datos predeterminado: kfloat32, que representa números de punto flotante de 32 bits.
    :param name: El nombre de este módulo, valor predeterminado "".

    :return: Una clase Dynamic_GRU

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


    Aplica una red neuronal recurrente (RNN) a una secuencia de entrada de longitud dinámica.

    La primera entrada debe ser una entrada de secuencia por lotes con longitud variable definida
    a través de la clase ``tensor.PackedSequence``.

    La clase ``tensor.PackedSequence`` se puede construir
    llamando a la siguiente función en sucesión: ``pad_sequence``, ``pack_pad_sequence``.

    La primera salida de Dynamic_RNN también es una clase ``tensor.PackedSequence``,
    que se puede desempaquetar a un QTensor normal usando ``tensor.pad_pack_sequence``.

    Módulo de red neuronal recurrente (RNN), usando :math:`\tanh` o :math:`\text{ReLU}` como función de activación. Soporta configuraciones bidireccionales y multicapa.
    La fórmula de cálculo de la RNN unidireccional de una sola capa es la siguiente:

    .. math::
        h_t = \tanh(W_{ih} x_t + b_{ih} + W_{hh} h_{(t-1)} + b_{hh})

    Si :attr:`nonlinearity` es ``'relu'``, entonces :math:`\text{ReLU}` reemplazará a :math:`\tanh`.

    Esta clase hereda de ``pyvqnet.nn.Module`` y ``torch.nn.Module``, y se puede agregar al modelo torch como submódulo de ``torch.nn.Module``.

    Los datos en ``_buffers`` de esta clase son de tipo ``torch.Tensor``.

    Los datos en ``_parmeters`` de esta clase son de tipo ``torch.nn.Parameter``.

    :param input_size: Dimensión de la característica de entrada.
    :param hidden_size: Dimensión de la característica oculta.
    :param num_layers: Número de capas RNN apiladas, valor predeterminado: 1.
    :param nonlinearity: Función de activación no lineal, valor predeterminado ``'tanh'``.
    :param batch_first: Si es True, la forma de entrada es [tamaño de lote, longitud de secuencia, dimensión de característica]; si es False, la forma de entrada es [longitud de secuencia, tamaño de lote, dimensión de característica], valor predeterminado True.
    :param use_bias: Si es False, este módulo no aplica sesgo, valor predeterminado: True.
    :param bidirectional: Si es True, se convierte en una RNN bidireccional, valor predeterminado: False.
    :param dtype: El tipo de datos del parámetro, predeterminado: None, use el tipo de datos predeterminado: kfloat32, que representa números de punto flotante de 32 bits.
    :param name: El nombre de este módulo, valor predeterminado "".

    :return: Instancia de Dynamic_RNN

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


    Aplica una RNN de Memoria a Largo Plazo (LSTM) a secuencias de entrada de longitud dinámica.

    La primera entrada debe ser una entrada de secuencia por lotes con longitud variable definida
    a través de una clase ``tensor.PackedSequence``.

    La clase ``tensor.PackedSequence`` se puede construir
    llamando a las siguientes funciones en sucesión: ``pad_sequence``, ``pack_pad_sequence``.

    La primera salida de Dynamic_LSTM también es una clase ``tensor.PackedSequence``,
    que se puede desempaquetar a un QTensor normal usando ``tensor.pad_pack_sequence``.

    Módulo de Red Neuronal Recurrente (RNN), usando :math:`\tanh` o :math:`\text{ReLU}` como función de activación. Soporta configuraciones bidireccionales y multicapa.
    La fórmula de cálculo de la RNN unidireccional de una sola capa es la siguiente: 
    
    
    .. math::
        \begin{array}{ll} \\
            i_t = \sigma(W_{ii} x_t + b_{ii} + W_{hi} h_{t-1} + b_{hi}) \\
            f_t = \sigma(W_{if} x_t + b_{if} + W_{hf} h_{t-1} + b_{hf}) \\
            g_t = \tanh(W_{ig} x_t + b_{ig} + W_{hg} h_{t-1} + b_{hg}) \\
            o_t = \sigma(W_{io} x_t + b_{io} + W_{ho} h_{t-1} + b_{ho}) \\
            c_t = f_t \odot c_{t-1} + i_t \odot g_t \\
            h_t = o_t \odot \tanh(c_t) \\
        \end{array}

    Esta clase hereda de ``pyvqnet.nn.Module`` y ``torch.nn.Module``, y se puede agregar al modelo torch como submódulo de ``torch.nn.Module``.

    Los datos en ``_buffers`` de esta clase son de tipo ``torch.Tensor``.

    Los datos en ``_parmeters`` de esta clase son de tipo ``torch.nn.Parameter``.

    :param input_size: Dimensión de la característica de entrada.
    :param hidden_size: Dimensión de la característica oculta.
    :param num_layers: Número de capas LSTM apiladas, valor predeterminado: 1.
    :param batch_first: Si es True, la forma de entrada es [tamaño de lote, longitud de secuencia, dimensión de característica]; si es False, la forma de entrada es [longitud de secuencia, tamaño de lote, dimensión de característica], valor predeterminado True.
    :param use_bias: Si es False, este módulo no aplica sesgo, valor predeterminado: True.
    :param bidirectional: Si es True, se convierte en una LSTM bidireccional, valor predeterminado: False.
    :param dtype: El tipo de datos del parámetro, predeterminado: None, use el tipo de datos predeterminado: kfloat32, que representa números de punto flotante de 32 bits.
    :param name: El nombre de este módulo, valor predeterminado "".

    :return: Instancia de Dynamic_LSTM

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

    Reduce/aumenta la resolución de la entrada.

    Actualmente solo soporta datos de entrada 4D.

    El tamaño de entrada se interpreta como `B x C x H x W`.

    Las opciones de `mode` disponibles son ``nearest``, ``bilinear``, ``bicubic``.

    Esta clase hereda de ``pyvqnet.nn.Module`` y ``torch.nn.Module`` y se puede agregar al modelo torch como submódulo de ``torch.nn.Module``.

    :param size: Tamaño de salida, valor predeterminado None.
    :param scale_factor: Factor de escala, valor predeterminado None.
    :param mode: Algoritmo usado para el sobremuestreo ``nearest`` | ``bilinear`` | ``bicubic``.
    :param align_corners: Desde un punto de vista geométrico, tratamos los píxeles de entrada y salida como cuadrados en lugar de puntos. Si se establece en `true`, los tensores de entrada y salida se alinearán por los puntos centrales de sus píxeles de esquina. Los puntos centrales de los píxeles de esquina están alineados y se conservan los valores de los píxeles de esquina. Si se establece en `false`, los tensores de entrada y salida se alinearán por los puntos de esquina de sus píxeles de esquina, y se conservan los valores de los píxeles de esquina. Los puntos de esquina de los píxeles de esquina están alineados, y la interpolación usará valores de borde para el relleno. Los valores fuera de los límites se rellenan, haciendo esta operación independiente del tamaño de entrada. Cuando ``scale_factor`` permanece sin cambios. Esto solo funciona cuando ``mode`` es ``bilinear``.
    :param recompute_scale_factor: Recalcular el factor de escala para usar en el cálculo de interpolación. Cuando se pasa ``scale_factor`` como argumento, se usará para calcular el tamaño de salida.
    :param name: Nombre del módulo.

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

    Construye una clase que calcula la atención de producto punto escalado para los tensores query, key y value.

    Esta clase hereda de ``pyvqnet.nn.Module`` y ``torch.nn.Module``, y se puede agregar al modelo torch como submódulo de ``torch.nn.Module``.

    :param attn_mask: Máscara de atención; valor predeterminado: None. La forma debe ser transmisible a la forma de los pesos de atención.
    :param dropout_p: Probabilidad de dropout; valor predeterminado: 0, si es mayor que 0.0, se aplica dropout.
    :param scale: Factor de escala aplicado antes de softmax, valor predeterminado: None.
    :param is_causal: valor predeterminado: False, si se establece en true, la máscara de atención es una matriz triangular inferior cuando la máscara es una matriz cuadrada. Si tanto attn_mask como is_causal están configurados, se genera un error.
    :return: Una clase SDPA

    Examples::
    
        from pyvqnet.nn.torch import SDPA
        from pyvqnet import tensor
        import pyvqnet
        pyvqnet.backends.set_backend("torch")
        model = SDPA(tensor.QTensor([1.]))

   .. py:method:: forward(query,key,value)

        Realiza el cálculo forward.

        :param query: El QTensor de entrada query.
        :param key: El QTensor de entrada key.
        :param value: El QTensor de entrada key.
        :return: El QTensor devuelto por el cálculo SDPA.
        
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

Loss Functions API
------------------------

MeanSquaredError
^^^^^^^^^^^^^^^^^^^^^^^^

.. py:class:: pyvqnet.nn.torch.MeanSquaredError(name="")

    Calcula el error cuadrático medio entre la entrada :math:`x` y el valor objetivo :math:`y`.

    Si el error cuadrático se puede describir mediante la siguiente función:

    .. math::
        \ell(x, y) = L = \{l_1,\dots,l_N\}^\top, \quad
        l_n = \left( x_n - y_n \right)^2,

    :math:`x` y :math:`y` son QTensor de formas arbitrarias, y el error cuadrático medio del total de :math:`n` elementos se calcula de la siguiente manera.

    .. math::
        \ell(x, y) =
        \operatorname{mean}(L)

    Esta clase hereda de ``pyvqnet.nn.Module`` y ``torch.nn.Module``, y se puede agregar al modelo torch como submódulo de ``torch.nn.Module``.

    :param name: El nombre de este módulo, valor predeterminado "".
    :return: Una instancia de error RMS.

    Parámetros requeridos para la función de cálculo forward del error RMS:

        x: :math:`(N, *)` valor predicho, donde :math:`*` representa cualquier dimensión.

        y: :math:`(N, *)`, valor objetivo, un QTensor de la misma dimensión que la entrada.

    .. note::

        Tenga en cuenta que, a diferencia de frameworks como pytorch, en la función forward de la siguiente función MeanSquaredError, el primer parámetro es el valor objetivo y el segundo parámetro es el valor predicho.


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

    Mide la pérdida de entropía cruzada binaria promedio entre el objetivo y la entrada.

    La entropía cruzada binaria sin promediar es la siguiente:

    .. math::
        \ell(x, y) = L = \{l_1,\dots,l_N\}^\top, \quad
        l_n = - w_n \left[ y_n \cdot \log x_n + (1 - y_n) \cdot \log (1 - x_n) \right],

    donde :math:`N` es el tamaño del lote.

    .. math::
        \ell(x, y) = \operatorname{mean}(L)

    Esta clase hereda de ``pyvqnet.nn.Module`` y ``torch.nn.Module`` y se puede agregar a modelos torch como submódulo de ``torch.nn.Module``.

    :param name: El nombre de este módulo, valor predeterminado "".
    :return: Una instancia de entropía cruzada binaria promedio.

    Parámetros requeridos para la función de cálculo forward del error de entropía cruzada binaria promedio:

        x: :math:`(N, *)` valor predicho, donde :math:`*` representa cualquier dimensión.

        y: :math:`(N, *)`, valor objetivo, un QTensor de la misma dimensión que la entrada.

    .. note::

        Tenga en cuenta que, a diferencia de frameworks como pytorch, en la función forward de la función BinaryCrossEntropy, el primer parámetro es el valor objetivo y el segundo parámetro es el valor predicho.
        
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

    Esta función de pérdida combina LogSoftmax y NLLLoss para calcular la entropía cruzada categórica promedio.

    La función de pérdida se calcula de la siguiente manera, donde class es la etiqueta de categoría correspondiente del valor objetivo:

    .. math::
        \text{loss}(x, y) = -\log\left(\frac{\exp(x[class])}{\sum_j \exp(x[j])}\right)
        = -x[class] + \log\left(\sum_j \exp(x[j])\right)

    :param name: El nombre de este módulo, valor predeterminado "".
    :return: La instancia de entropía cruzada categórica promedio.

    Parámetros requeridos para la función de cálculo forward del error:

        x: :math:`(N, *)` Valor predicho, donde :math:`*` indica cualquier dimensión.

        y: :math:`(N, *)`, valor objetivo, un QTensor de la misma dimensión que la entrada. Debe ser un entero de 64 bits, kint64.

    .. note::

        Tenga en cuenta que, a diferencia de pytorch y otros frameworks, en la función forward de la función CategoricalCrossEntropy, el primer parámetro es el valor objetivo y el segundo parámetro es el valor predicho.

        Esta clase hereda de ``pyvqnet.nn.Module`` y ``torch.nn.Module``, y se puede agregar al modelo torch como submódulo de ``torch.nn.Module``.

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

    Esta función de pérdida combina LogSoftmax y NLLLoss para calcular la entropía cruzada de clasificación promedio, y tiene mayor estabilidad numérica.

    La función de pérdida se calcula de la siguiente manera, donde class es la etiqueta de clasificación correspondiente del valor objetivo:

    .. math::
        \text{loss}(x, y) = -\log\left(\frac{\exp(x[class])}{\sum_j \exp(x[j])}\right)
        = -x[class] + \log\left(\sum_j \exp(x[j])\right)

    :param name: El nombre de este módulo, valor predeterminado "".
    :return: Una instancia de función de pérdida de entropía cruzada Softmax

    Parámetros requeridos para la función de cálculo forward del error:

        x: :math:`(N, *)` valor predicho, donde :math:`*` indica cualquier dimensión.

        y: :math:`(N, *)`, valor objetivo, un QTensor de la misma dimensión que la entrada. Debe ser un entero de 64 bits, kint64.

    .. note::

        Tenga en cuenta que, a diferencia de pytorch y otros frameworks, en la función forward de la función SoftmaxCrossEntropy, el primer parámetro es el valor objetivo y el segundo parámetro es el valor predicho.

        Esta clase hereda de ``pyvqnet.nn.Module`` y ``torch.nn.Module``, y se puede agregar al modelo torch como submódulo de ``torch.nn.Module``.
        
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

    
    Pérdida de log-verosimilitud negativa promedio. Útil para problemas de clasificación con C clases.

    `x` es la verosimilitud de probabilidad dada por el modelo. Su forma puede ser :math:`(N, C)` o :math:`(N, C, d_1, d_2, ..., d_K)`. `y` es el valor verdadero esperado de la función de pérdida, que contiene índices de clase en :math:`[0, C-1]`.

    .. math::
        \ell(x, y) = L = \{l_1,\dots,l_N\}^\top, \quad
        l_n = -
        \sum_{n=1}^N \frac{1}{N}x_{n,y_n} \quad

    :param name: El nombre de este módulo, valor predeterminado "".
    :return: Una instancia de función de pérdida NLL_Loss

    Parámetros requeridos para la función de cálculo forward del error:

        x: :math:`(N, *)`, el valor de predicción de salida de la función de pérdida, que puede ser una variable multidimensional.

        y: :math:`(N, *)`, el valor objetivo de la función de pérdida. Debe ser un entero de 64 bits, kint64.

    .. note::

        Tenga en cuenta que, a diferencia de frameworks como pytorch, en la función forward de la función NLL_Loss, el primer parámetro es el valor objetivo y el segundo parámetro es el valor de predicción.

        Esta clase hereda de ``pyvqnet.nn.Module`` y ``torch.nn.Module``, y se puede agregar al modelo torch como submódulo de ``torch.nn.Module``.
            
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

    Esta función calcula la pérdida de LogSoftmax y NLL_Loss juntos.

    `x` contiene la salida no normalizada. Su forma puede ser :math:`(C)`, :math:`(N, C)` bidimensional o :math:`(N, C, d_1, d_2, ..., d_K)` multidimensional.

    La fórmula de la función de pérdida es la siguiente, donde class es la etiqueta de clase correspondiente del valor objetivo:

    .. math::
        \text{loss}(x, y) = -\log\left(\frac{\exp(x[class])}{\sum_j \exp(x[j])}\right)
        = -x[class] + \log\left(\sum_j \exp(x[j])\right)

    :param name: El nombre de este módulo, valor predeterminado "".
    :return: Una instancia de función de pérdida CrossEntropyLoss

    Parámetros requeridos para la función de cálculo forward del error:

        x: :math:`(N, *)`, la salida de la función de pérdida, que puede ser una variable multidimensional.

        y: :math:`(N, *)`, el valor verdadero esperado de la función de pérdida. Debe ser un entero de 64 bits, kint64.

    .. note::

        Tenga en cuenta que, a diferencia de frameworks como pytorch, en la función forward de la función CrossEntropyLoss, el primer parámetro es el valor objetivo y el segundo parámetro es el valor predicho.

        Esta clase hereda de ``pyvqnet.nn.Module`` y ``torch.nn.Module``, y se puede agregar al modelo torch como submódulo de ``torch.nn.Module``.

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


Funciones de Activación
-----------------------

Sigmoid
^^^^^^^^^^^^^^^^^^^^^^^^

.. py:class:: pyvqnet.nn.torch.Sigmoid(name:str="")

    Capa de función de activación Sigmoide.

    .. math::
        \text{Sigmoid}(x) = \frac{1}{1 + \exp(-x)}

    Esta clase hereda de ``pyvqnet.nn.Module`` y ``torch.nn.Module``, y se puede agregar al modelo torch como submódulo de ``torch.nn.Module``.

    :param name: El nombre de la capa de función de activación, valor predeterminado "".

    :return: Una instancia de capa de función de activación Sigmoide.

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

    Esta clase hereda de ``pyvqnet.nn.Module`` y ``torch.nn.Module``, y se puede agregar al modelo torch como submódulo de ``torch.nn.Module``.

    :param name: El nombre de la capa de función de activación, valor predeterminado "".

    :return: Una instancia de Softplus.

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


    Esta clase hereda de ``pyvqnet.nn.Module`` y ``torch.nn.Module``, y se puede agregar al modelo torch como submódulo de ``torch.nn.Module``.


    :param name: El nombre de la capa de función de activación, valor predeterminado "".

    :return: Una instancia de SoftSign.

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


    Esta clase hereda de ``pyvqnet.nn.Module`` y ``torch.nn.Module``, y se puede agregar al modelo torch como submódulo de ``torch.nn.Module``.


    :param axis: la dimensión a calcular (el último eje es -1), valor predeterminado = -1.
    :param name: El nombre de la capa de función de activación, valor predeterminado "".

    :return: Una instancia de Softmax.

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


    Esta clase hereda de ``pyvqnet.nn.Module`` y ``torch.nn.Module``, y se puede agregar al modelo torch como submódulo de ``torch.nn.Module``.


    :param name: El nombre de la capa de función de activación, valor predeterminado "".

    :return: Instancia de HardSigmoid.

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


    Esta clase hereda de ``pyvqnet.nn.Module`` y ``torch.nn.Module``, y se puede agregar al modelo torch como submódulo de ``torch.nn.Module``.


    :param name: El nombre de la capa de función de activación, valor predeterminado "".

    :return: Una instancia de ReLu.

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


    Esta clase hereda de ``pyvqnet.nn.Module`` y ``torch.nn.Module``, y se puede agregar al modelo torch como submódulo de ``torch.nn.Module``.


    :param alpha: Coeficiente de LeakyRelu, valor predeterminado: 0.01.
    :param name: El nombre de la capa de función de activación, valor predeterminado "".

    :return: Una instancia de activación LeakyReLu.

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

    Cuando el parámetro de aproximación es 'tanh', GELU se estima de la siguiente manera:

    .. math:: \text{GELU}(x) = 0.5 * x * (1 + \text{Tanh}(\sqrt{2 / \pi} * (x + 0.044715 * x^3)))


    Esta clase hereda de ``pyvqnet.nn.Module`` y ``torch.nn.Module``, y se puede agregar al modelo torch como submódulo de ``torch.nn.Module``.


    :param approximate: Método de cálculo aproximado, el valor predeterminado es "tanh".
    :param name: El nombre de la capa de función de activación, valor predeterminado "".

    :return: Instancia de activación Gelu.

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

    Capa de función de activación ELU (Unidad Lineal Exponencial).

    .. math::
        \text{ELU}(x) = \begin{cases}
        x, & \text{ if } x > 0\\
        \alpha * (\exp(x) - 1), & \text{ if } x \leq 0
        \end{cases}


    Esta clase hereda de ``pyvqnet.nn.Module`` y ``torch.nn.Module``, y se puede agregar al modelo torch como submódulo de ``torch.nn.Module``.



    :param alpha: Coeficiente ELU, valor predeterminado: 1.
    :param name: El nombre de la capa de función de activación, valor predeterminado "".

    :return: Instancia de activación ELU.

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

    Función de activación Tanh (tangente hiperbólica).

    .. math::
        \text{Tanh}(x) = \frac{\exp(x) - \exp(-x)} {\exp(x) + \exp(-x)}


    Esta clase hereda de ``pyvqnet.nn.Module`` y ``torch.nn.Module``, y se puede agregar al modelo torch como submódulo de ``torch.nn.Module``.



    :param name: El nombre de la capa de función de activación, valor predeterminado "".

    :return: Instancia de activación Tanh.

    Examples::

        from pyvqnet.nn.torch import Tanh
        from pyvqnet.tensor import QTensor
        import pyvqnet
        pyvqnet.backends.set_backend("torch")
        layer = Tanh()
        y = layer(QTensor([-1, 2.0, -3, 4.0]))



Módulo Optimizador
---------------------------------------------

Para los módulos de circuitos clásicos y cuánticos que heredan de `TorchModule`, los parámetros `model.paramters()` se pueden seguir optimizando usando optimizadores distintos de `Rotosolve` en :ref:`Optimizer`.



Usando pyqpanda para ejecutar circuitos cuánticos variacionales
-------------------------------------------------------------------------

La siguiente es la interfaz de circuito cuántico variacional de entrenamiento para el cálculo de circuitos usando pyqpanda y pyqpanda3.

.. warning::

    La parte de computación cuántica del siguiente TorchQpandaQuantumLayer usa pyqpanda2.

    Debido a problemas de compatibilidad entre pyqpanda2 y pyqpanda3, debe instalar pyqpnda2 usted mismo, `pip install pyqpanda`

TorchQpandaQuantumLayer
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Si está más familiarizado con la sintaxis de pyQPanda2, puede usar la interfaz TorchQpandaQuantumLayer, agregando bits cuánticos personalizados ``qubits``, bits clásicos ``cbits`` y el simulador backend ``machine`` a la función de parámetro ``qprog_with_measure`` de TorchQpandaQuantumLayer.

.. py:class:: pyvqnet.qnn.vqc.torch.TorchQpandaQuantumLayer(qprog_with_measure,para_num,diff_method:str = "parameter_shift",delta:float = 0.01,dtype=None,name="")

    Módulo de cómputo abstracto de capa cuántica variacional. Usa pyQPanda2 para simular un circuito cuántico parametrizado y obtener los resultados de medición. Esta capa cuántica variacional hereda el módulo de cálculo de gradiente del framework VQNet. Puede usar el método de desplazamiento de parámetros para calcular el gradiente de los parámetros del circuito, entrenar modelos de circuitos cuánticos variacionales o incrustar circuitos cuánticos variacionales en modelos híbridos cuánticos y clásicos.

    :param qprog_with_measure: Funciones de operación y medición de circuitos cuánticos construidas con pyQPand.
    :param para_num: `int` - número de parámetros.
    :param diff_method: Método para resolver gradientes de parámetros de circuitos cuánticos, "parameter shift" o "finite difference", valor predeterminado parameter shift.
    :param delta: \delta al calcular gradientes por diferencias finitas.
    :param dtype: Tipo de datos del parámetro, valor predeterminado: None, usa el tipo de datos predeterminado: kfloat32, que representa números de punto flotante de 32 bits.
    :param name: El nombre de este módulo, valor predeterminado "".

    :return: Un módulo que puede calcular circuitos cuánticos.

    .. note::

        qprog_with_measure es una función de circuito cuántico definida en pyQPanda2.

        Esta función debe contener los siguientes parámetros como entrada de función (incluso si un parámetro no se usa realmente), de lo contrario no funcionará correctamente en esta función.

        En comparación con QuantumLayer, en la función de ejecución de circuito variacional pasada por esta interfaz, el usuario debe crear manualmente los bits cuánticos y los simuladores.

        Si qprog_with_measure requiere medición cuántica, el usuario también necesita crear y asignar cbits manualmente.

        El uso de la función de circuito cuántico qprog_with_measure (input, param, nqubits, ncbits) puede consultar el siguiente ejemplo.

        `input`: Datos clásicos unidimensionales de entrada. Si no hay ninguno, ingrese None.

        `param`: Parámetros del circuito cuántico variacional unidimensional a entrenar.

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

    La parte de computación cuántica de las siguientes interfaces TorchQcloud3QuantumLayer y TorchQpanda3QuantumLayer usa pyqpanda3.

    Si usa la función QCloud bajo este módulo, ocurrirán errores al importar pyqpanda2 en el código o al usar las interfaces de paquetes relacionados con pyqpanda2 de pyvqnet.

TorchQcloud3QuantumLayer
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Cuando instale la versión más reciente de pyqpanda3, puede usar esta interfaz para definir un circuito variacional y enviarlo al chip real de originqc para su ejecución.

.. py:class:: pyvqnet.qnn.vqc.torch.TorchQcloud3QuantumLayer(origin_qprog_func, qcloud_token, para_num, pauli_str_dict=None, shots = 1000, initializer=None, dtype=None, name="", diff_method="parameter_shift", submit_kwargs={}, query_kwargs={})
    
    Un módulo de cómputo abstracto para chips reales usando originqc de pyqpanda3. Envía circuitos cuánticos parametrizados a chips reales y obtiene resultados de medición.
    Si diff_method == "random_coordinate_descent", la capa seleccionará aleatoriamente un solo parámetro para calcular el gradiente, y otros parámetros permanecerán en cero. Referencia: https://arxiv.org/abs/2311.00088

    .. note::

        qcloud_token es el token de API que solicitó desde la plataforma en la nube.

        origin_qprog_func necesita devolver datos de tipo pypqanda3.core.QProg. Si no se establece pauli_str_dict, es necesario asegurarse de que la medición se haya insertado en el QProg.

        origin_qprog_func debe tener el siguiente formato:

        origin_qprog_func(input,param)

        `input`: Datos clásicos de entrada 1~2D. En el caso 2D, la primera dimensión es el tamaño del lote.

        `param`: Parámetros a entrenar del circuito cuántico variacional 1D.

    .. warning::

        Esta clase hereda de ``pyvqnet.nn.Module`` y ``torch.nn.Module``, y se puede agregar al modelo torch como submódulo de ``torch.nn.Module``.

        Los datos en ``_buffers`` de esta clase son de tipo ``torch.Tensor``.

        Los datos en ``_parmeters`` de esta clase son de tipo ``torch.nn.Parameter``.

    :param origin_qprog_func: La función de circuito cuántico variacional construida por QPanda, que debe devolver QProg.
    :param qcloud_token: `str` - El tipo de máquina cuántica o token de nube para ejecución.
    :param para_num: `int` - El número de parámetros, el parámetro es un QTensor de tamaño [para_num].
    :param pauli_str_dict: `dict|list` - Diccionario o lista de diccionarios que representan operadores Pauli en circuitos cuánticos. Valor predeterminado "None", lo que significa que se realizan operaciones de medición. Si se ingresa un diccionario de operadores Pauli, se calcula una expectativa única o múltiples expectativas.
    :param shot: `int` - Número de mediciones. El valor predeterminado es 1000.
    :param initializer: Inicializador para valores de parámetros. El valor predeterminado es "None", usando una distribución normal 0~2*pi.
    :param dtype: Tipo de datos del parámetro. El valor predeterminado es None, lo que significa usar el tipo de datos predeterminado pyvqnet.kfloat32.
    :param name: El nombre del módulo. El valor predeterminado es una cadena vacía.
    :param diff_method: Método de diferenciación para el cálculo del gradiente. El valor predeterminado es "parameter_shift", "random_coordinate_descent".
    :param submit_kwargs: Parámetros clave adicionales para enviar circuitos cuánticos, predeterminado: {"if_print_qcloud_log":False,"chip_id":"WK_C180","is_amend":True,"is_mapping":True,"is_optimization":True,"compile_level":3,"default_task_group_size":200,"test_qcloud_fake":False,"":"server_ip_address"}, cuando test_qcloud_fake se establece en True, simulación local CPUQVM.
    :param query_kwargs: Parámetros clave adicionales para consultar resultados cuánticos, predeterminado: {"timeout":2,"print_query_info":True,"sub_circuits_split_size":1}.
    :return: Un módulo que puede calcular circuitos cuánticos.


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

Si está más familiarizado con la sintaxis de pyQPanda3, puede usar la interfaz TorchQpanda3QuantumLayer.

.. py:class:: pyvqnet.qnn.vqc.torch.TorchQpanda3QuantumLayer(qprog_with_measure,para_num,diff_method:str = "parameter_shift",delta:float = 0.01,dtype=None,name="")

    Módulo de cómputo abstracto de capa cuántica variacional. Usa pyQPanda3 para simular un circuito cuántico parametrizado y obtener los resultados de medición. Esta capa cuántica variacional hereda el módulo de cálculo de gradiente del framework VQNet. Puede usar el método de desplazamiento de parámetros para calcular el gradiente de los parámetros del circuito, entrenar el modelo de circuito cuántico variacional o incrustar el circuito cuántico variacional en un modelo híbrido cuántico y clásico.

    :param qprog_with_measure: Funciones de operación y medición de circuitos cuánticos construidas con pyQPand.
    :param para_num: `int` - número de parámetros.
    :param diff_method: método para resolver gradientes de parámetros de circuitos cuánticos, "parameter shift" o "finite difference", valor predeterminado parameter shift.
    :param delta: \delta al calcular gradientes por diferencias finitas.
    :param dtype: tipo de datos del parámetro, valor predeterminado: None, usa el tipo de datos predeterminado: kfloat32, que representa números de punto flotante de 32 bits.
    :param name: el nombre de este módulo, valor predeterminado "".

    :return: un módulo que puede calcular circuitos cuánticos.

    .. note::

        qprog_with_measure es una función de circuito cuántico definida en pyQPanda.

        Esta función debe incluir los siguientes parámetros como entradas de función (incluso si un parámetro no se usa realmente), de lo contrario no funcionará correctamente en esta función.

        El uso de la función de circuito cuántico qprog_with_measure (input, param, nqubits, ncbits) puede consultar el siguiente ejemplo.

        `input`: Datos clásicos unidimensionales de entrada. Si no hay ninguno, ingrese None.

        `param`: Parámetros a entrenar para el circuito cuántico variacional unidimensional.

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

Módulo de circuito cuántico variacional e interfaz basados en diferenciación automática
---------------------------------------------------------------------------------------------
Clase Base
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Escribir un modelo de circuito cuántico variacional requiere heredar de ``QModule``.

QModule
""""""""""""""""""

.. py:class:: pyvqnet.qnn.vqc.torch.QModule(name="")

    Define la clase base que debe heredar el modelo de circuito cuántico variacional `Module` cuando el usuario usa el motor `torch`.
    Esta clase hereda de ``pyvqnet.nn.torch.TorchModule`` y ``torch.nn.Module``.

    Esta clase se puede agregar al modelo torch como submódulo de ``torch.nn.Module``.

    .. note::

        Esta clase y sus clases derivadas solo son aplicables con ``pyvqnet.backends.set_backend("torch")``, no mezclar con el ``Module`` bajo el ``pyvqnet.nn`` predeterminado.

        Los datos en ``_buffers`` de esta clase son de tipo ``torch.Tensor``.

        Los datos en ``_parmeters`` de esta clase son de tipo ``torch.nn.Parameter``.


QMachine
""""""""""""""""""

.. py:class:: pyvqnet.qnn.vqc.torch.QMachine(num_wires, dtype=pyvqnet.kcomplex64,grad_mode="",save_ir=False)

    Clase de simulador para computación cuántica variacional, que incluye vectores de estado cuyo atributo states son circuitos cuánticos.

    Esta clase hereda de ``pyvqnet.nn.torch.TorchModule`` y ``pyvqnet.qnn.QMachine``.

    Esta clase se puede agregar al modelo torch como submódulo de ``torch.nn.Module``.

    .. note::

        Antes de cada ejecución del circuito cuántico completo, debe usar `pyvqnet.qnn.vqc.QMachine.reset_states(batchsize)` para reinicializar el estado inicial en el simulador y transmitirlo a las
        dimensiones (batchsize,*) para adaptarse al entrenamiento con datos por lotes.

    :param num_wires: El número de bits cuánticos.
    :param dtype: El tipo de datos de los datos calculados. El valor predeterminado es pyvqnet.kcomplex64, y la precisión del parámetro correspondiente es pyvqnet.kfloat32.
    :param grad_mode: El modo de cálculo de gradiente, que puede ser "adjoint", el valor predeterminado: "", usa diferenciación automática.
    :param save_ir: Cuando se establece en True, guarda la operación en originIR, el valor predeterminado: False.

    :return: Produce un objeto QMachine.

    Example::
        
        from pyvqnet.qnn.vqc.torch import QMachine
        import pyvqnet
        pyvqnet.backends.set_backend("torch")
        qm = QMachine(4)
        print(qm.states)


   .. py:method:: reset_states(batchsize)

        Reinicializa el estado inicial en el simulador y lo transmite a las
        dimensiones (batchsize,*) para adaptarse al entrenamiento con datos por lotes.

        :param batchsize: Dimensión de procesamiento por lotes.

Módulo de compuerta lógica cuántica variacional
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The following function interfaces in ``pyvqnet.qnn.vqc`` directly support ``QTensor`` of ``torch`` backend for calculation.

.. csv-table:: Lista de interfaces pyvqnet.qnn.vqc soportadas
    :file: ./images/same_apis_from_vqc.csv

The following quantum circuit modules inherit from ``pyvqnet.qnn.vqc.torch.QModule``, where calculations are performed using ``torch.Tensor``.

.. note::

    Esta clase y sus clases derivadas solo son aplicables con ``pyvqnet.backends.set_backend("torch")``, no mezclar con el ``Module`` bajo el ``pyvqnet.nn`` predeterminado.

    Si estas clases tienen variables miembro no paramétricas ``_buffers``, los datos en ellas son de tipo ``torch.Tensor``.
    Si estas clases tienen variables miembro paramétricas ``_parmeters``, los datos en ellas son de tipo ``torch.nn.Parameter``.

I
""""""""""""""""""

.. py:class:: pyvqnet.qnn.vqc.torch.I(has_params: bool = False,trainable: bool = False,init_params=None,wires=None,dtype=pyvqnet.kcomplex64,use_dagger=False)
    
    define una compuerta cuántica I.

    Esta clase hereda de ``pyvqnet.qnn.vqc.torch.QModule`` y ``torch.nn.Module``.
    Esta clase se puede agregar al modelo torch como submódulo de ``torch.nn.Module``.

    :param has_params: si tiene parámetros, compuertas como RX, RY y otras deben establecerse en True, y las que no tienen parámetros deben establecerse en False, el valor predeterminado es False.
    :param trainable: si tiene parámetros para entrenar. Si la capa usa datos de entrada externos para construir la matriz de la compuerta lógica, establezca en False. Si los parámetros a entrenar necesitan inicializarse desde esta capa, es True, el valor predeterminado es False.
    :param init_params: Parámetros de inicialización usados para codificar datos clásicos QTensor, el valor predeterminado es None.
    :param wires: Índice de bit del efecto de línea, el valor predeterminado es None.
    :param dtype: La precisión de datos de la matriz interna de la compuerta lógica puede establecerse en pyvqnet.kcomplex64 o pyvqnet.kcomplex128, correspondiendo a entrada float o double respectivamente.
    :param use_dagger: si usar la versión transpuesta conjugada de la compuerta, el valor predeterminado es False.
    :return: una instancia de ``pyvqnet.qnn.vqc.torch.QModule``

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
    
    define una compuerta cuántica Hadamard .

    Esta clase hereda de ``pyvqnet.qnn.vqc.torch.QModule`` y ``torch.nn.Module``.
    Esta clase se puede agregar al modelo torch como submódulo de ``torch.nn.Module``.

    :param has_params: si tiene parámetros, compuertas como RX, RY y otras deben establecerse en True, y las que no tienen parámetros deben establecerse en False, el valor predeterminado es False.
    :param trainable: si tiene parámetros para entrenar. Si la capa usa datos de entrada externos para construir la matriz de la compuerta lógica, establezca en False. Si los parámetros a entrenar necesitan inicializarse desde esta capa, es True, el valor predeterminado es False.
    :param init_params: Parámetros de inicialización usados para codificar datos clásicos QTensor, el valor predeterminado es None.
    :param wires: Índice de bit del efecto de línea, el valor predeterminado es None.
    :param dtype: La precisión de datos de la matriz interna de la compuerta lógica puede establecerse en pyvqnet.kcomplex64 o pyvqnet.kcomplex128, correspondiendo a entrada float o double respectivamente.
    :param use_dagger: si usar la versión transpuesta conjugada de la compuerta, el valor predeterminado es False.
    :return: una instancia de ``pyvqnet.qnn.vqc.torch.QModule``

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
    
    define una compuerta cuántica T .

    Esta clase hereda de ``pyvqnet.qnn.vqc.torch.QModule`` y ``torch.nn.Module``.
    Esta clase se puede agregar al modelo torch como submódulo de ``torch.nn.Module``.

    :param has_params: si tiene parámetros, compuertas como RX, RY y otras deben establecerse en True, y las que no tienen parámetros deben establecerse en False, el valor predeterminado es False.
    :param trainable: si tiene parámetros para entrenar. Si la capa usa datos de entrada externos para construir la matriz de la compuerta lógica, establezca en False. Si los parámetros a entrenar necesitan inicializarse desde esta capa, es True, el valor predeterminado es False.
    :param init_params: Parámetros de inicialización usados para codificar datos clásicos QTensor, el valor predeterminado es None.
    :param wires: Índice de bit del efecto de línea, el valor predeterminado es None.
    :param dtype: La precisión de datos de la matriz interna de la compuerta lógica puede establecerse en pyvqnet.kcomplex64 o pyvqnet.kcomplex128, correspondiendo a entrada float o double respectivamente.
    :param use_dagger: si usar la versión transpuesta conjugada de la compuerta, el valor predeterminado es False.
    :return: una instancia de ``pyvqnet.qnn.vqc.torch.QModule``

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
    
    define una compuerta cuántica S .

    Esta clase hereda de ``pyvqnet.qnn.vqc.torch.QModule`` y ``torch.nn.Module``.
    Esta clase se puede agregar al modelo torch como submódulo de ``torch.nn.Module``.

    :param has_params: si tiene parámetros, compuertas como RX, RY y otras deben establecerse en True, y las que no tienen parámetros deben establecerse en False, el valor predeterminado es False.
    :param trainable: si tiene parámetros para entrenar. Si la capa usa datos de entrada externos para construir la matriz de la compuerta lógica, establezca en False. Si los parámetros a entrenar necesitan inicializarse desde esta capa, es True, el valor predeterminado es False.
    :param init_params: Parámetros de inicialización usados para codificar datos clásicos QTensor, el valor predeterminado es None.
    :param wires: Índice de bit del efecto de línea, el valor predeterminado es None.
    :param dtype: La precisión de datos de la matriz interna de la compuerta lógica puede establecerse en pyvqnet.kcomplex64 o pyvqnet.kcomplex128, correspondiendo a entrada float o double respectivamente.
    :param use_dagger: si usar la versión transpuesta conjugada de la compuerta, el valor predeterminado es False.
    :return: una instancia de ``pyvqnet.qnn.vqc.torch.QModule``

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
    
    define una compuerta cuántica PauliX .

    Esta clase hereda de ``pyvqnet.qnn.vqc.torch.QModule`` y ``torch.nn.Module``.
    Esta clase se puede agregar al modelo torch como submódulo de ``torch.nn.Module``.


    :param has_params: si tiene parámetros, compuertas como RX, RY y otras deben establecerse en True, y las que no tienen parámetros deben establecerse en False, el valor predeterminado es False.
    :param trainable: si tiene parámetros para entrenar. Si la capa usa datos de entrada externos para construir la matriz de la compuerta lógica, establezca en False. Si los parámetros a entrenar necesitan inicializarse desde esta capa, es True, el valor predeterminado es False.
    :param init_params: Parámetros de inicialización usados para codificar datos clásicos QTensor, el valor predeterminado es None.
    :param wires: Índice de bit del efecto de línea, el valor predeterminado es None.
    :param dtype: La precisión de datos de la matriz interna de la compuerta lógica puede establecerse en pyvqnet.kcomplex64 o pyvqnet.kcomplex128, correspondiendo a entrada float o double respectivamente.
    :param use_dagger: si usar la versión transpuesta conjugada de la compuerta, el valor predeterminado es False.
    :return: una instancia de ``pyvqnet.qnn.vqc.torch.QModule``

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
    
    define una compuerta cuántica PauliY .

    Esta clase hereda de ``pyvqnet.qnn.vqc.torch.QModule`` y ``torch.nn.Module``.
    Esta clase se puede agregar al modelo torch como submódulo de ``torch.nn.Module``.


    :param has_params: si tiene parámetros, compuertas como RX, RY y otras deben establecerse en True, y las que no tienen parámetros deben establecerse en False, el valor predeterminado es False.
    :param trainable: si tiene parámetros para entrenar. Si la capa usa datos de entrada externos para construir la matriz de la compuerta lógica, establezca en False. Si los parámetros a entrenar necesitan inicializarse desde esta capa, es True, el valor predeterminado es False.
    :param init_params: Parámetros de inicialización usados para codificar datos clásicos QTensor, el valor predeterminado es None.
    :param wires: Índice de bit del efecto de línea, el valor predeterminado es None.
    :param dtype: La precisión de datos de la matriz interna de la compuerta lógica puede establecerse en pyvqnet.kcomplex64 o pyvqnet.kcomplex128, correspondiendo a entrada float o double respectivamente.
    :param use_dagger: si usar la versión transpuesta conjugada de la compuerta, el valor predeterminado es False.
    :return: una instancia de ``pyvqnet.qnn.vqc.torch.QModule``

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
    
    define una compuerta cuántica PauliZ .

    Esta clase hereda de ``pyvqnet.qnn.vqc.torch.QModule`` y ``torch.nn.Module``.
    Esta clase se puede agregar al modelo torch como submódulo de ``torch.nn.Module``.


    :param has_params: si tiene parámetros, compuertas como RX, RY y otras deben establecerse en True, y las que no tienen parámetros deben establecerse en False, el valor predeterminado es False.
    :param trainable: si tiene parámetros para entrenar. Si la capa usa datos de entrada externos para construir la matriz de la compuerta lógica, establezca en False. Si los parámetros a entrenar necesitan inicializarse desde esta capa, es True, el valor predeterminado es False.
    :param init_params: Parámetros de inicialización usados para codificar datos clásicos QTensor, el valor predeterminado es None.
    :param wires: Índice de bit del efecto de línea, el valor predeterminado es None.
    :param dtype: La precisión de datos de la matriz interna de la compuerta lógica puede establecerse en pyvqnet.kcomplex64 o pyvqnet.kcomplex128, correspondiendo a entrada float o double respectivamente.
    :param use_dagger: si usar la versión transpuesta conjugada de la compuerta, el valor predeterminado es False.
    :return: una instancia de ``pyvqnet.qnn.vqc.torch.QModule``

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
    
    define una compuerta cuántica X1 .

    Esta clase hereda de ``pyvqnet.qnn.vqc.torch.QModule`` y ``torch.nn.Module``.
    Esta clase se puede agregar al modelo torch como submódulo de ``torch.nn.Module``.

    :param has_params: si tiene parámetros, compuertas como RX, RY y otras deben establecerse en True, y las que no tienen parámetros deben establecerse en False, el valor predeterminado es False.
    :param trainable: si tiene parámetros para entrenar. Si la capa usa datos de entrada externos para construir la matriz de la compuerta lógica, establezca en False. Si los parámetros a entrenar necesitan inicializarse desde esta capa, es True, el valor predeterminado es False.
    :param init_params: Parámetros de inicialización usados para codificar datos clásicos QTensor, el valor predeterminado es None.
    :param wires: Índice de bit del efecto de línea, el valor predeterminado es None.
    :param dtype: La precisión de datos de la matriz interna de la compuerta lógica puede establecerse en pyvqnet.kcomplex64 o pyvqnet.kcomplex128, correspondiendo a entrada float o double respectivamente.
    :param use_dagger: si usar la versión transpuesta conjugada de la compuerta, el valor predeterminado es False.
    :return: una instancia de ``pyvqnet.qnn.vqc.torch.QModule``

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
    
    define una compuerta cuántica RX .

    Esta clase hereda de ``pyvqnet.qnn.vqc.torch.QModule`` y ``torch.nn.Module``.
    Esta clase se puede agregar al modelo torch como submódulo de ``torch.nn.Module``.

    :param has_params: si tiene parámetros, compuertas como RX, RY y otras deben establecerse en True, y las que no tienen parámetros deben establecerse en False, el valor predeterminado es False.
    :param trainable: si tiene parámetros para entrenar. Si la capa usa datos de entrada externos para construir la matriz de la compuerta lógica, establezca en False. Si los parámetros a entrenar necesitan inicializarse desde esta capa, es True, el valor predeterminado es False.
    :param init_params: Parámetros de inicialización usados para codificar datos clásicos QTensor, el valor predeterminado es None.
    :param wires: Índice de bit del efecto de línea, el valor predeterminado es None.
    :param dtype: La precisión de datos de la matriz interna de la compuerta lógica puede establecerse en pyvqnet.kcomplex64 o pyvqnet.kcomplex128, correspondiendo a entrada float o double respectivamente.
    :param use_dagger: si usar la versión transpuesta conjugada de la compuerta, el valor predeterminado es False.
    :return: una instancia de ``pyvqnet.qnn.vqc.torch.QModule``

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
    
    define una compuerta cuántica RY .

    Esta clase hereda de ``pyvqnet.qnn.vqc.torch.QModule`` y ``torch.nn.Module``.
    Esta clase se puede agregar al modelo torch como submódulo de ``torch.nn.Module``.

    :param has_params: si tiene parámetros, compuertas como RX, RY y otras deben establecerse en True, y las que no tienen parámetros deben establecerse en False, el valor predeterminado es False.
    :param trainable: si tiene parámetros para entrenar. Si la capa usa datos de entrada externos para construir la matriz de la compuerta lógica, establezca en False. Si los parámetros a entrenar necesitan inicializarse desde esta capa, es True, el valor predeterminado es False.
    :param init_params: Parámetros de inicialización usados para codificar datos clásicos QTensor, el valor predeterminado es None.
    :param wires: Índice de bit del efecto de línea, el valor predeterminado es None.
    :param dtype: La precisión de datos de la matriz interna de la compuerta lógica puede establecerse en pyvqnet.kcomplex64 o pyvqnet.kcomplex128, correspondiendo a entrada float o double respectivamente.
    :param use_dagger: si usar la versión transpuesta conjugada de la compuerta, el valor predeterminado es False.
    :return: una instancia de ``pyvqnet.qnn.vqc.torch.QModule``

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
    
    define una compuerta cuántica RZ .

    Esta clase hereda de ``pyvqnet.qnn.vqc.torch.QModule`` y ``torch.nn.Module``.
    Esta clase se puede agregar al modelo torch como submódulo de ``torch.nn.Module``.

    :param has_params: si tiene parámetros, compuertas como RX, RY y otras deben establecerse en True, y las que no tienen parámetros deben establecerse en False, el valor predeterminado es False.
    :param trainable: si tiene parámetros para entrenar. Si la capa usa datos de entrada externos para construir la matriz de la compuerta lógica, establezca en False. Si los parámetros a entrenar necesitan inicializarse desde esta capa, es True, el valor predeterminado es False.
    :param init_params: Parámetros de inicialización usados para codificar datos clásicos QTensor, el valor predeterminado es None.
    :param wires: Índice de bit del efecto de línea, el valor predeterminado es None.
    :param dtype: La precisión de datos de la matriz interna de la compuerta lógica puede establecerse en pyvqnet.kcomplex64 o pyvqnet.kcomplex128, correspondiendo a entrada float o double respectivamente.
    :param use_dagger: si usar la versión transpuesta conjugada de la compuerta, el valor predeterminado es False.
    :return: una instancia de ``pyvqnet.qnn.vqc.torch.QModule``

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
    
    define una compuerta cuántica CRX .

    Esta clase hereda de ``pyvqnet.qnn.vqc.torch.QModule`` y ``torch.nn.Module``.
    Esta clase se puede agregar al modelo torch como submódulo de ``torch.nn.Module``.

    :param has_params: si tiene parámetros, compuertas como RX, RY y otras deben establecerse en True, y las que no tienen parámetros deben establecerse en False, el valor predeterminado es False.
    :param trainable: si tiene parámetros para entrenar. Si la capa usa datos de entrada externos para construir la matriz de la compuerta lógica, establezca en False. Si los parámetros a entrenar necesitan inicializarse desde esta capa, es True, el valor predeterminado es False.
    :param init_params: Parámetros de inicialización usados para codificar datos clásicos QTensor, el valor predeterminado es None.
    :param wires: Índice de bit del efecto de línea, el valor predeterminado es None.
    :param dtype: La precisión de datos de la matriz interna de la compuerta lógica puede establecerse en pyvqnet.kcomplex64 o pyvqnet.kcomplex128, correspondiendo a entrada float o double respectivamente.
    :param use_dagger: si usar la versión transpuesta conjugada de la compuerta, el valor predeterminado es False.
    :return: una instancia de ``pyvqnet.qnn.vqc.torch.QModule``

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
    
    define una compuerta cuántica CRY .

    Esta clase hereda de ``pyvqnet.qnn.vqc.torch.QModule`` y ``torch.nn.Module``.
    Esta clase se puede agregar al modelo torch como submódulo de ``torch.nn.Module``.

    :param has_params: si tiene parámetros, compuertas como RX, RY y otras deben establecerse en True, y las que no tienen parámetros deben establecerse en False, el valor predeterminado es False.
    :param trainable: si tiene parámetros para entrenar. Si la capa usa datos de entrada externos para construir la matriz de la compuerta lógica, establezca en False. Si los parámetros a entrenar necesitan inicializarse desde esta capa, es True, el valor predeterminado es False.
    :param init_params: Parámetros de inicialización usados para codificar datos clásicos QTensor, el valor predeterminado es None.
    :param wires: Índice de bit del efecto de línea, el valor predeterminado es None.
    :param dtype: La precisión de datos de la matriz interna de la compuerta lógica puede establecerse en pyvqnet.kcomplex64 o pyvqnet.kcomplex128, correspondiendo a entrada float o double respectivamente.
    :param use_dagger: si usar la versión transpuesta conjugada de la compuerta, el valor predeterminado es False.
    :return: una instancia de ``pyvqnet.qnn.vqc.torch.QModule``

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
    
    define una compuerta cuántica CRZ .

    Esta clase hereda de ``pyvqnet.qnn.vqc.torch.QModule`` y ``torch.nn.Module``.
    Esta clase se puede agregar al modelo torch como submódulo de ``torch.nn.Module``.

    :param has_params: si tiene parámetros, compuertas como RX, RY y otras deben establecerse en True, y las que no tienen parámetros deben establecerse en False, el valor predeterminado es False.
    :param trainable: si tiene parámetros para entrenar. Si la capa usa datos de entrada externos para construir la matriz de la compuerta lógica, establezca en False. Si los parámetros a entrenar necesitan inicializarse desde esta capa, es True, el valor predeterminado es False.
    :param init_params: Parámetros de inicialización usados para codificar datos clásicos QTensor, el valor predeterminado es None.
    :param wires: Índice de bit del efecto de línea, el valor predeterminado es None.
    :param dtype: La precisión de datos de la matriz interna de la compuerta lógica puede establecerse en pyvqnet.kcomplex64 o pyvqnet.kcomplex128, correspondiendo a entrada float o double respectivamente.
    :param use_dagger: si usar la versión transpuesta conjugada de la compuerta, el valor predeterminado es False.
    :return: una instancia de ``pyvqnet.qnn.vqc.torch.QModule``

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
    
    define una compuerta cuántica U1 .

    Esta clase hereda de ``pyvqnet.qnn.vqc.torch.QModule`` y ``torch.nn.Module``.
    Esta clase se puede agregar al modelo torch como submódulo de ``torch.nn.Module``.

    :param has_params: si tiene parámetros, compuertas como RX, RY y otras deben establecerse en True, y las que no tienen parámetros deben establecerse en False, el valor predeterminado es False.
    :param trainable: si tiene parámetros para entrenar. Si la capa usa datos de entrada externos para construir la matriz de la compuerta lógica, establezca en False. Si los parámetros a entrenar necesitan inicializarse desde esta capa, es True, el valor predeterminado es False.
    :param init_params: Parámetros de inicialización usados para codificar datos clásicos QTensor, el valor predeterminado es None.
    :param wires: Índice de bit del efecto de línea, el valor predeterminado es None.
    :param dtype: La precisión de datos de la matriz interna de la compuerta lógica puede establecerse en pyvqnet.kcomplex64 o pyvqnet.kcomplex128, correspondiendo a entrada float o double respectivamente.
    :param use_dagger: si usar la versión transpuesta conjugada de la compuerta, el valor predeterminado es False.
    :return: una instancia de ``pyvqnet.qnn.vqc.torch.QModule``

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
    
    define una compuerta cuántica U2 .

    Esta clase hereda de ``pyvqnet.qnn.vqc.torch.QModule`` y ``torch.nn.Module``.
    Esta clase se puede agregar al modelo torch como submódulo de ``torch.nn.Module``.

    :param has_params: si tiene parámetros, compuertas como RX, RY y otras deben establecerse en True, y las que no tienen parámetros deben establecerse en False, el valor predeterminado es False.
    :param trainable: si tiene parámetros para entrenar. Si la capa usa datos de entrada externos para construir la matriz de la compuerta lógica, establezca en False. Si los parámetros a entrenar necesitan inicializarse desde esta capa, es True, el valor predeterminado es False.
    :param init_params: Parámetros de inicialización usados para codificar datos clásicos QTensor, el valor predeterminado es None.
    :param wires: Índice de bit del efecto de línea, el valor predeterminado es None.
    :param dtype: La precisión de datos de la matriz interna de la compuerta lógica puede establecerse en pyvqnet.kcomplex64 o pyvqnet.kcomplex128, correspondiendo a entrada float o double respectivamente.
    :param use_dagger: si usar la versión transpuesta conjugada de la compuerta, el valor predeterminado es False.
    :return: una instancia de ``pyvqnet.qnn.vqc.torch.QModule``

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
    
    define una compuerta cuántica U3 .

    Esta clase hereda de ``pyvqnet.qnn.vqc.torch.QModule`` y ``torch.nn.Module``.
    Esta clase se puede agregar al modelo torch como submódulo de ``torch.nn.Module``.

    :param has_params: si tiene parámetros, compuertas como RX, RY y otras deben establecerse en True, y las que no tienen parámetros deben establecerse en False, el valor predeterminado es False.
    :param trainable: si tiene parámetros para entrenar. Si la capa usa datos de entrada externos para construir la matriz de la compuerta lógica, establezca en False. Si los parámetros a entrenar necesitan inicializarse desde esta capa, es True, el valor predeterminado es False.
    :param init_params: Parámetros de inicialización usados para codificar datos clásicos QTensor, el valor predeterminado es None.
    :param wires: Índice de bit del efecto de línea, el valor predeterminado es None.
    :param dtype: La precisión de datos de la matriz interna de la compuerta lógica puede establecerse en pyvqnet.kcomplex64 o pyvqnet.kcomplex128, correspondiendo a entrada float o double respectivamente.
    :param use_dagger: si usar la versión transpuesta conjugada de la compuerta, el valor predeterminado es False.
    :return: una instancia de ``pyvqnet.qnn.vqc.torch.QModule``

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
    
    define una compuerta cuántica CNOT , alias `CX` .

    Esta clase hereda de ``pyvqnet.qnn.vqc.torch.QModule`` y ``torch.nn.Module``.
    Esta clase se puede agregar al modelo torch como submódulo de ``torch.nn.Module``.

    :param has_params: si tiene parámetros, compuertas como RX, RY y otras deben establecerse en True, y las que no tienen parámetros deben establecerse en False, el valor predeterminado es False.
    :param trainable: si tiene parámetros para entrenar. Si la capa usa datos de entrada externos para construir la matriz de la compuerta lógica, establezca en False. Si los parámetros a entrenar necesitan inicializarse desde esta capa, es True, el valor predeterminado es False.
    :param init_params: Parámetros de inicialización usados para codificar datos clásicos QTensor, el valor predeterminado es None.
    :param wires: Índice de bit del efecto de línea, el valor predeterminado es None.
    :param dtype: La precisión de datos de la matriz interna de la compuerta lógica puede establecerse en pyvqnet.kcomplex64 o pyvqnet.kcomplex128, correspondiendo a entrada float o double respectivamente.
    :param use_dagger: si usar la versión transpuesta conjugada de la compuerta, el valor predeterminado es False.
    :return: una instancia de ``pyvqnet.qnn.vqc.torch.QModule``

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
    
    define una compuerta cuántica CY .

    Esta clase hereda de ``pyvqnet.qnn.vqc.torch.QModule`` y ``torch.nn.Module``.
    Esta clase se puede agregar al modelo torch como submódulo de ``torch.nn.Module``.

    :param has_params: si tiene parámetros, compuertas como RX, RY y otras deben establecerse en True, y las que no tienen parámetros deben establecerse en False, el valor predeterminado es False.
    :param trainable: si tiene parámetros para entrenar. Si la capa usa datos de entrada externos para construir la matriz de la compuerta lógica, establezca en False. Si los parámetros a entrenar necesitan inicializarse desde esta capa, es True, el valor predeterminado es False.
    :param init_params: Parámetros de inicialización usados para codificar datos clásicos QTensor, el valor predeterminado es None.
    :param wires: Índice de bit del efecto de línea, el valor predeterminado es None.
    :param dtype: La precisión de datos de la matriz interna de la compuerta lógica puede establecerse en pyvqnet.kcomplex64 o pyvqnet.kcomplex128, correspondiendo a entrada float o double respectivamente.
    :param use_dagger: si usar la versión transpuesta conjugada de la compuerta, el valor predeterminado es False.
    :return: una instancia de ``pyvqnet.qnn.vqc.torch.QModule``

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
    
    define una compuerta cuántica CZ .

    Esta clase hereda de ``pyvqnet.qnn.vqc.torch.QModule`` y ``torch.nn.Module``.
    Esta clase se puede agregar al modelo torch como submódulo de ``torch.nn.Module``.

    :param has_params: si tiene parámetros, compuertas como RX, RY y otras deben establecerse en True, y las que no tienen parámetros deben establecerse en False, el valor predeterminado es False.
    :param trainable: si tiene parámetros para entrenar. Si la capa usa datos de entrada externos para construir la matriz de la compuerta lógica, establezca en False. Si los parámetros a entrenar necesitan inicializarse desde esta capa, es True, el valor predeterminado es False.
    :param init_params: Parámetros de inicialización usados para codificar datos clásicos QTensor, el valor predeterminado es None.
    :param wires: Índice de bit del efecto de línea, el valor predeterminado es None.
    :param dtype: La precisión de datos de la matriz interna de la compuerta lógica puede establecerse en pyvqnet.kcomplex64 o pyvqnet.kcomplex128, correspondiendo a entrada float o double respectivamente.
    :param use_dagger: si usar la versión transpuesta conjugada de la compuerta, el valor predeterminado es False.
    :return: una instancia de ``pyvqnet.qnn.vqc.torch.QModule``

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
    
    define una compuerta cuántica CR .

    Esta clase hereda de ``pyvqnet.qnn.vqc.torch.QModule`` y ``torch.nn.Module``.
    Esta clase se puede agregar al modelo torch como submódulo de ``torch.nn.Module``.

    :param has_params: si tiene parámetros, compuertas como RX, RY y otras deben establecerse en True, y las que no tienen parámetros deben establecerse en False, el valor predeterminado es False.
    :param trainable: si tiene parámetros para entrenar. Si la capa usa datos de entrada externos para construir la matriz de la compuerta lógica, establezca en False. Si los parámetros a entrenar necesitan inicializarse desde esta capa, es True, el valor predeterminado es False.
    :param init_params: Parámetros de inicialización usados para codificar datos clásicos QTensor, el valor predeterminado es None.
    :param wires: Índice de bit del efecto de línea, el valor predeterminado es None.
    :param dtype: La precisión de datos de la matriz interna de la compuerta lógica puede establecerse en pyvqnet.kcomplex64 o pyvqnet.kcomplex128, correspondiendo a entrada float o double respectivamente.
    :param use_dagger: si usar la versión transpuesta conjugada de la compuerta, el valor predeterminado es False.
    :return: una instancia de ``pyvqnet.qnn.vqc.torch.QModule``

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
    
    define una compuerta cuántica SWAP .

    Esta clase hereda de ``pyvqnet.qnn.vqc.torch.QModule`` y ``torch.nn.Module``.
    Esta clase se puede agregar al modelo torch como submódulo de ``torch.nn.Module``.

    :param has_params: si tiene parámetros, compuertas como RX, RY y otras deben establecerse en True, y las que no tienen parámetros deben establecerse en False, el valor predeterminado es False.
    :param trainable: si tiene parámetros para entrenar. Si la capa usa datos de entrada externos para construir la matriz de la compuerta lógica, establezca en False. Si los parámetros a entrenar necesitan inicializarse desde esta capa, es True, el valor predeterminado es False.
    :param init_params: Parámetros de inicialización usados para codificar datos clásicos QTensor, el valor predeterminado es None.
    :param wires: Índice de bit del efecto de línea, el valor predeterminado es None.
    :param dtype: La precisión de datos de la matriz interna de la compuerta lógica puede establecerse en pyvqnet.kcomplex64 o pyvqnet.kcomplex128, correspondiendo a entrada float o double respectivamente.
    :param use_dagger: si usar la versión transpuesta conjugada de la compuerta, el valor predeterminado es False.
    :return: una instancia de ``pyvqnet.qnn.vqc.torch.QModule``

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
    
    define una compuerta cuántica SWAP .

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

    Esta clase hereda de ``pyvqnet.qnn.vqc.torch.QModule`` y ``torch.nn.Module``.
    Esta clase se puede agregar al modelo torch como submódulo de ``torch.nn.Module``.

    :param has_params: si tiene parámetros, compuertas como RX, RY y otras deben establecerse en True, y las que no tienen parámetros deben establecerse en False, el valor predeterminado es False.
    :param trainable: si tiene parámetros para entrenar. Si la capa usa datos de entrada externos para construir la matriz de la compuerta lógica, establezca en False. Si los parámetros a entrenar necesitan inicializarse desde esta capa, es True, el valor predeterminado es False.
    :param init_params: Parámetros de inicialización usados para codificar datos clásicos QTensor, el valor predeterminado es None.
    :param wires: Índice de bit del efecto de línea, el valor predeterminado es None.
    :param dtype: La precisión de datos de la matriz interna de la compuerta lógica puede establecerse en pyvqnet.kcomplex64 o pyvqnet.kcomplex128, correspondiendo a entrada float o double respectivamente.
    :param use_dagger: si usar la versión transpuesta conjugada de la compuerta, el valor predeterminado es False.
    :return: una instancia de ``pyvqnet.qnn.vqc.torch.QModule``

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
    
    define una compuerta cuántica RXX .

    Esta clase hereda de ``pyvqnet.qnn.vqc.torch.QModule`` y ``torch.nn.Module``.
    Esta clase se puede agregar al modelo torch como submódulo de ``torch.nn.Module``.

    :param has_params: si tiene parámetros, compuertas como RX, RY y otras deben establecerse en True, y las que no tienen parámetros deben establecerse en False, el valor predeterminado es False.
    :param trainable: si tiene parámetros para entrenar. Si la capa usa datos de entrada externos para construir la matriz de la compuerta lógica, establezca en False. Si los parámetros a entrenar necesitan inicializarse desde esta capa, es True, el valor predeterminado es False.
    :param init_params: Parámetros de inicialización usados para codificar datos clásicos QTensor, el valor predeterminado es None.
    :param wires: Índice de bit del efecto de línea, el valor predeterminado es None.
    :param dtype: La precisión de datos de la matriz interna de la compuerta lógica puede establecerse en pyvqnet.kcomplex64 o pyvqnet.kcomplex128, correspondiendo a entrada float o double respectivamente.
    :param use_dagger: si usar la versión transpuesta conjugada de la compuerta, el valor predeterminado es False.
    :return: una instancia de ``pyvqnet.qnn.vqc.torch.QModule``

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
    
    define una compuerta cuántica RYY .

    Esta clase hereda de ``pyvqnet.qnn.vqc.torch.QModule`` y ``torch.nn.Module``.
    Esta clase se puede agregar al modelo torch como submódulo de ``torch.nn.Module``.

    :param has_params: si tiene parámetros, compuertas como RX, RY y otras deben establecerse en True, y las que no tienen parámetros deben establecerse en False, el valor predeterminado es False.
    :param trainable: si tiene parámetros para entrenar. Si la capa usa datos de entrada externos para construir la matriz de la compuerta lógica, establezca en False. Si los parámetros a entrenar necesitan inicializarse desde esta capa, es True, el valor predeterminado es False.
    :param init_params: Parámetros de inicialización usados para codificar datos clásicos QTensor, el valor predeterminado es None.
    :param wires: Índice de bit del efecto de línea, el valor predeterminado es None.
    :param dtype: La precisión de datos de la matriz interna de la compuerta lógica puede establecerse en pyvqnet.kcomplex64 o pyvqnet.kcomplex128, correspondiendo a entrada float o double respectivamente.
    :param use_dagger: si usar la versión transpuesta conjugada de la compuerta, el valor predeterminado es False.
    :return: una instancia de ``pyvqnet.qnn.vqc.torch.QModule``

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
    
    define una compuerta cuántica RZZ .

    Esta clase hereda de ``pyvqnet.qnn.vqc.torch.QModule`` y ``torch.nn.Module``.
    Esta clase se puede agregar al modelo torch como submódulo de ``torch.nn.Module``.

    :param has_params: si tiene parámetros, compuertas como RX, RY y otras deben establecerse en True, y las que no tienen parámetros deben establecerse en False, el valor predeterminado es False.
    :param trainable: si tiene parámetros para entrenar. Si la capa usa datos de entrada externos para construir la matriz de la compuerta lógica, establezca en False. Si los parámetros a entrenar necesitan inicializarse desde esta capa, es True, el valor predeterminado es False.
    :param init_params: Parámetros de inicialización usados para codificar datos clásicos QTensor, el valor predeterminado es None.
    :param wires: Índice de bit del efecto de línea, el valor predeterminado es None.
    :param dtype: La precisión de datos de la matriz interna de la compuerta lógica puede establecerse en pyvqnet.kcomplex64 o pyvqnet.kcomplex128, correspondiendo a entrada float o double respectivamente.
    :param use_dagger: si usar la versión transpuesta conjugada de la compuerta, el valor predeterminado es False.
    :return: una instancia de ``pyvqnet.qnn.vqc.torch.QModule``

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
    
    define una compuerta cuántica RZX .

    Esta clase hereda de ``pyvqnet.qnn.vqc.torch.QModule`` y ``torch.nn.Module``.
    Esta clase se puede agregar al modelo torch como submódulo de ``torch.nn.Module``.

    :param has_params: si tiene parámetros, compuertas como RX, RY y otras deben establecerse en True, y las que no tienen parámetros deben establecerse en False, el valor predeterminado es False.
    :param trainable: si tiene parámetros para entrenar. Si la capa usa datos de entrada externos para construir la matriz de la compuerta lógica, establezca en False. Si los parámetros a entrenar necesitan inicializarse desde esta capa, es True, el valor predeterminado es False.
    :param init_params: Parámetros de inicialización usados para codificar datos clásicos QTensor, el valor predeterminado es None.
    :param wires: Índice de bit del efecto de línea, el valor predeterminado es None.
    :param dtype: La precisión de datos de la matriz interna de la compuerta lógica puede establecerse en pyvqnet.kcomplex64 o pyvqnet.kcomplex128, correspondiendo a entrada float o double respectivamente.
    :param use_dagger: si usar la versión transpuesta conjugada de la compuerta, el valor predeterminado es False.
    :return: una instancia de ``pyvqnet.qnn.vqc.torch.QModule``

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
    
    define una compuerta cuántica Toffoli .

    Esta clase hereda de ``pyvqnet.qnn.vqc.torch.QModule`` y ``torch.nn.Module``.
    Esta clase se puede agregar al modelo torch como submódulo de ``torch.nn.Module``.

    :param has_params: si tiene parámetros, compuertas como RX, RY y otras deben establecerse en True, y las que no tienen parámetros deben establecerse en False, el valor predeterminado es False.
    :param trainable: si tiene parámetros para entrenar. Si la capa usa datos de entrada externos para construir la matriz de la compuerta lógica, establezca en False. Si los parámetros a entrenar necesitan inicializarse desde esta capa, es True, el valor predeterminado es False.
    :param init_params: Parámetros de inicialización usados para codificar datos clásicos QTensor, el valor predeterminado es None.
    :param wires: Índice de bit del efecto de línea, el valor predeterminado es None.
    :param dtype: La precisión de datos de la matriz interna de la compuerta lógica puede establecerse en pyvqnet.kcomplex64 o pyvqnet.kcomplex128, correspondiendo a entrada float o double respectivamente.
    :param use_dagger: si usar la versión transpuesta conjugada de la compuerta, el valor predeterminado es False.
    :return: una instancia de ``pyvqnet.qnn.vqc.torch.QModule``

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
    
    define una compuerta cuántica IsingXX .

    Esta clase hereda de ``pyvqnet.qnn.vqc.torch.QModule`` y ``torch.nn.Module``.
    Esta clase se puede agregar al modelo torch como submódulo de ``torch.nn.Module``.

    :param has_params: si tiene parámetros, compuertas como RX, RY y otras deben establecerse en True, y las que no tienen parámetros deben establecerse en False, el valor predeterminado es False.
    :param trainable: si tiene parámetros para entrenar. Si la capa usa datos de entrada externos para construir la matriz de la compuerta lógica, establezca en False. Si los parámetros a entrenar necesitan inicializarse desde esta capa, es True, el valor predeterminado es False.
    :param init_params: Parámetros de inicialización usados para codificar datos clásicos QTensor, el valor predeterminado es None.
    :param wires: Índice de bit del efecto de línea, el valor predeterminado es None.
    :param dtype: La precisión de datos de la matriz interna de la compuerta lógica puede establecerse en pyvqnet.kcomplex64 o pyvqnet.kcomplex128, correspondiendo a entrada float o double respectivamente.
    :param use_dagger: si usar la versión transpuesta conjugada de la compuerta, el valor predeterminado es False.
    :return: una instancia de ``pyvqnet.qnn.vqc.torch.QModule``

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
    
    define una compuerta cuántica IsingYY .

    Esta clase hereda de ``pyvqnet.qnn.vqc.torch.QModule`` y ``torch.nn.Module``.
    Esta clase se puede agregar al modelo torch como submódulo de ``torch.nn.Module``.

    :param has_params: si tiene parámetros, compuertas como RX, RY y otras deben establecerse en True, y las que no tienen parámetros deben establecerse en False, el valor predeterminado es False.
    :param trainable: si tiene parámetros para entrenar. Si la capa usa datos de entrada externos para construir la matriz de la compuerta lógica, establezca en False. Si los parámetros a entrenar necesitan inicializarse desde esta capa, es True, el valor predeterminado es False.
    :param init_params: Parámetros de inicialización usados para codificar datos clásicos QTensor, el valor predeterminado es None.
    :param wires: Índice de bit del efecto de línea, el valor predeterminado es None.
    :param dtype: La precisión de datos de la matriz interna de la compuerta lógica puede establecerse en pyvqnet.kcomplex64 o pyvqnet.kcomplex128, correspondiendo a entrada float o double respectivamente.
    :param use_dagger: si usar la versión transpuesta conjugada de la compuerta, el valor predeterminado es False.
    :return: una instancia de ``pyvqnet.qnn.vqc.torch.QModule``

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
    
    define una compuerta cuántica IsingZZ .

    Esta clase hereda de ``pyvqnet.qnn.vqc.torch.QModule`` y ``torch.nn.Module``.
    Esta clase se puede agregar al modelo torch como submódulo de ``torch.nn.Module``.

    :param has_params: si tiene parámetros, compuertas como RX, RY y otras deben establecerse en True, y las que no tienen parámetros deben establecerse en False, el valor predeterminado es False.
    :param trainable: si tiene parámetros para entrenar. Si la capa usa datos de entrada externos para construir la matriz de la compuerta lógica, establezca en False. Si los parámetros a entrenar necesitan inicializarse desde esta capa, es True, el valor predeterminado es False.
    :param init_params: Parámetros de inicialización usados para codificar datos clásicos QTensor, el valor predeterminado es None.
    :param wires: Índice de bit del efecto de línea, el valor predeterminado es None.
    :param dtype: La precisión de datos de la matriz interna de la compuerta lógica puede establecerse en pyvqnet.kcomplex64 o pyvqnet.kcomplex128, correspondiendo a entrada float o double respectivamente.
    :param use_dagger: si usar la versión transpuesta conjugada de la compuerta, el valor predeterminado es False.
    :return: una instancia de ``pyvqnet.qnn.vqc.torch.QModule``

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
    
    define una compuerta cuántica IsingXY .

    Esta clase hereda de ``pyvqnet.qnn.vqc.torch.QModule`` y ``torch.nn.Module``.
    Esta clase se puede agregar al modelo torch como submódulo de ``torch.nn.Module``.

    :param has_params: si tiene parámetros, compuertas como RX, RY y otras deben establecerse en True, y las que no tienen parámetros deben establecerse en False, el valor predeterminado es False.
    :param trainable: si tiene parámetros para entrenar. Si la capa usa datos de entrada externos para construir la matriz de la compuerta lógica, establezca en False. Si los parámetros a entrenar necesitan inicializarse desde esta capa, es True, el valor predeterminado es False.
    :param init_params: Parámetros de inicialización usados para codificar datos clásicos QTensor, el valor predeterminado es None.
    :param wires: Índice de bit del efecto de línea, el valor predeterminado es None.
    :param dtype: La precisión de datos de la matriz interna de la compuerta lógica puede establecerse en pyvqnet.kcomplex64 o pyvqnet.kcomplex128, correspondiendo a entrada float o double respectivamente.
    :param use_dagger: si usar la versión transpuesta conjugada de la compuerta, el valor predeterminado es False.
    :return: una instancia de ``pyvqnet.qnn.vqc.torch.QModule``

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
    
    define una compuerta cuántica PhaseShift .

    Esta clase hereda de ``pyvqnet.qnn.vqc.torch.QModule`` y ``torch.nn.Module``.
    Esta clase se puede agregar al modelo torch como submódulo de ``torch.nn.Module``.

    :param has_params: si tiene parámetros, compuertas como RX, RY y otras deben establecerse en True, y las que no tienen parámetros deben establecerse en False, el valor predeterminado es False.
    :param trainable: si tiene parámetros para entrenar. Si la capa usa datos de entrada externos para construir la matriz de la compuerta lógica, establezca en False. Si los parámetros a entrenar necesitan inicializarse desde esta capa, es True, el valor predeterminado es False.
    :param init_params: Parámetros de inicialización usados para codificar datos clásicos QTensor, el valor predeterminado es None.
    :param wires: Índice de bit del efecto de línea, el valor predeterminado es None.
    :param dtype: La precisión de datos de la matriz interna de la compuerta lógica puede establecerse en pyvqnet.kcomplex64 o pyvqnet.kcomplex128, correspondiendo a entrada float o double respectivamente.
    :param use_dagger: si usar la versión transpuesta conjugada de la compuerta, el valor predeterminado es False.
    :return: una instancia de ``pyvqnet.qnn.vqc.torch.QModule``

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
    
    define una compuerta cuántica MultiRZ .

    Esta clase hereda de ``pyvqnet.qnn.vqc.torch.QModule`` y ``torch.nn.Module``.
    Esta clase se puede agregar al modelo torch como submódulo de ``torch.nn.Module``.

    :param has_params: si tiene parámetros, compuertas como RX, RY y otras deben establecerse en True, y las que no tienen parámetros deben establecerse en False, el valor predeterminado es False.
    :param trainable: si tiene parámetros para entrenar. Si la capa usa datos de entrada externos para construir la matriz de la compuerta lógica, establezca en False. Si los parámetros a entrenar necesitan inicializarse desde esta capa, es True, el valor predeterminado es False.
    :param init_params: Parámetros de inicialización usados para codificar datos clásicos QTensor, el valor predeterminado es None.
    :param wires: Índice de bit del efecto de línea, el valor predeterminado es None.
    :param dtype: La precisión de datos de la matriz interna de la compuerta lógica puede establecerse en pyvqnet.kcomplex64 o pyvqnet.kcomplex128, correspondiendo a entrada float o double respectivamente.
    :param use_dagger: si usar la versión transpuesta conjugada de la compuerta, el valor predeterminado es False.
    :return: una instancia de ``pyvqnet.qnn.vqc.torch.QModule``

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
    
    define una compuerta cuántica SDG .

    Esta clase hereda de ``pyvqnet.qnn.vqc.torch.QModule`` y ``torch.nn.Module``.
    Esta clase se puede agregar al modelo torch como submódulo de ``torch.nn.Module``.

    :param has_params: si tiene parámetros, compuertas como RX, RY y otras deben establecerse en True, y las que no tienen parámetros deben establecerse en False, el valor predeterminado es False.
    :param trainable: si tiene parámetros para entrenar. Si la capa usa datos de entrada externos para construir la matriz de la compuerta lógica, establezca en False. Si los parámetros a entrenar necesitan inicializarse desde esta capa, es True, el valor predeterminado es False.
    :param init_params: Parámetros de inicialización usados para codificar datos clásicos QTensor, el valor predeterminado es None.
    :param wires: Índice de bit del efecto de línea, el valor predeterminado es None.
    :param dtype: La precisión de datos de la matriz interna de la compuerta lógica puede establecerse en pyvqnet.kcomplex64 o pyvqnet.kcomplex128, correspondiendo a entrada float o double respectivamente.
    :param use_dagger: si usar la versión transpuesta conjugada de la compuerta, el valor predeterminado es False.
    :return: una instancia de ``pyvqnet.qnn.vqc.torch.QModule``

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
    
    define una compuerta cuántica SDG .

    Esta clase hereda de ``pyvqnet.qnn.vqc.torch.QModule`` y ``torch.nn.Module``.
    Esta clase se puede agregar al modelo torch como submódulo de ``torch.nn.Module``.

    :param has_params: si tiene parámetros, compuertas como RX, RY y otras deben establecerse en True, y las que no tienen parámetros deben establecerse en False, el valor predeterminado es False.
    :param trainable: si tiene parámetros para entrenar. Si la capa usa datos de entrada externos para construir la matriz de la compuerta lógica, establezca en False. Si los parámetros a entrenar necesitan inicializarse desde esta capa, es True, el valor predeterminado es False.
    :param init_params: Parámetros de inicialización usados para codificar datos clásicos QTensor, el valor predeterminado es None.
    :param wires: Índice de bit del efecto de línea, el valor predeterminado es None.
    :param dtype: La precisión de datos de la matriz interna de la compuerta lógica puede establecerse en pyvqnet.kcomplex64 o pyvqnet.kcomplex128, correspondiendo a entrada float o double respectivamente.
    :param use_dagger: si usar la versión transpuesta conjugada de la compuerta, el valor predeterminado es False.
    :return: una instancia de ``pyvqnet.qnn.vqc.torch.QModule``

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
    
    define una compuerta cuántica ControlledPhaseShift .

    Esta clase hereda de ``pyvqnet.qnn.vqc.torch.QModule`` y ``torch.nn.Module``.
    Esta clase se puede agregar al modelo torch como submódulo de ``torch.nn.Module``.

    :param has_params: si tiene parámetros, compuertas como RX, RY y otras deben establecerse en True, y las que no tienen parámetros deben establecerse en False, el valor predeterminado es False.
    :param trainable: si tiene parámetros para entrenar. Si la capa usa datos de entrada externos para construir la matriz de la compuerta lógica, establezca en False. Si los parámetros a entrenar necesitan inicializarse desde esta capa, es True, el valor predeterminado es False.
    :param init_params: Parámetros de inicialización usados para codificar datos clásicos QTensor, el valor predeterminado es None.
    :param wires: Índice de bit del efecto de línea, el valor predeterminado es None.
    :param dtype: La precisión de datos de la matriz interna de la compuerta lógica puede establecerse en pyvqnet.kcomplex64 o pyvqnet.kcomplex128, correspondiendo a entrada float o double respectivamente.
    :param use_dagger: si usar la versión transpuesta conjugada de la compuerta, el valor predeterminado es False.
    :return: una instancia de ``pyvqnet.qnn.vqc.torch.QModule``

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
    
    define una compuerta cuántica MultiControlledX .

    Esta clase hereda de ``pyvqnet.qnn.vqc.torch.QModule`` y ``torch.nn.Module``.
    Esta clase se puede agregar al modelo torch como submódulo de ``torch.nn.Module``.
    
    :param has_params: si tiene parámetros, compuertas como RX, RY y otras deben establecerse en True, y las que no tienen parámetros deben establecerse en False, el valor predeterminado es False.
    :param trainable: si tiene parámetros para entrenar. Si la capa usa datos de entrada externos para construir la matriz de la compuerta lógica, establezca en False. Si los parámetros a entrenar necesitan inicializarse desde esta capa, es True, el valor predeterminado es False.
    :param init_params: Parámetros de inicialización usados para codificar datos clásicos QTensor, el valor predeterminado es None.
    :param wires: Índice de bit del efecto de línea, el valor predeterminado es None.
    :param dtype: La precisión de datos de la matriz interna de la compuerta lógica puede establecerse en pyvqnet.kcomplex64 o pyvqnet.kcomplex128, correspondiendo a entrada float o double respectivamente.
    :param use_dagger: si usar la versión transpuesta conjugada de la compuerta, el valor predeterminado es False.
    :param control_values: Valor de control, el valor predeterminado es None, cuando el bit es 1, est00e1 controlado.

    :return: una instancia de ``pyvqnet.qnn.vqc.torch.QModule``

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


Measurements API
^^^^^^^^^^^^^^^^^^^^^^

Probability
"""""""""""""""""""""""""""""


.. py:class:: pyvqnet.qnn.vqc.torch.Probability(wires=None, name="")

    Calcula el resultado de medición de probabilidad del circuito cuántico en un bit específico.

    Esta clase hereda de ``pyvqnet.qnn.vqc.torch.QModule`` y ``torch.nn.Module``.
    Esta clase se puede agregar al modelo torch como submódulo de ``torch.nn.Module``.

    :param wires: El índice del bit de medición, lista, tupla o entero.
    :param name: El nombre del módulo, valor predeterminado: "".
    :return: El resultado de la medición, QTensor.

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

    Calcula los resultados de medición de circuitos cuánticos, soporta entrada obs como múltiples o simples operadores Pauli o Hamiltonianos.
    Por ejemplo:

    {\'X0\': 0.23} indica un efecto PauliX en el qubit 0, con un coeficiente de 0.23.

    {\'X1 Z2\': 2.4,\'Y2\': -0.5} corresponde al valor observado 2.4 * X1 @ Z2 - 0.5 * Y2.

    [{\'X1 Z2\': 4,\'Z1 Z0\': 3},{\'X1 Y2 Z0\': 3.5}] corresponde a los dos valores observados 4 * X1 @ Z2 + 3 * Z1 @ Z0 y 3.5 * X1 @ Y2 @ Z0.
        
        
    Esta clase hereda de ``pyvqnet.qnn.vqc.torch.QModule`` y ``torch.nn.Module``.

    Esta clase se puede agregar al modelo torch como submódulo de ``torch.nn.Module``.

    :param obs: observable.
    :param name: nombre del módulo, valor predeterminado: "".
    :return: resultado de la medición, QTensor.

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

    Obtiene resultados de muestreo con disparos en cables específicos.

    Esta clase hereda de ``pyvqnet.qnn.vqc.torch.QModule`` y ``torch.nn.Module``.
    Esta clase se puede agregar al modelo torch como submódulo de ``torch.nn.Module``.


    :param wires: Índice del qubit de muestreo. Valor predeterminado: None, usa todos los bits del simulador en tiempo de ejecución.
    :param obs: Este valor solo puede ser None.
    :param shots: Recuento de repeticiones de muestreo, valor predeterminado: 1.
    :param name: El nombre de este módulo, valor predeterminado: "".
    :return: una clase de método de medición

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

    Calcula la expectativa de una cantidad Hermitiana en un circuito cuántico.

    Esta clase hereda de ``pyvqnet.qnn.vqc.torch.QModule`` y ``torch.nn.Module``.
    Esta clase se puede agregar al modelo torch como submódulo de ``torch.nn.Module``.


    :param obs: Cantidad Hermitiana.
    :param name: nombre del módulo, valor predeterminado: "".
    :return: resultado esperado, QTensor.

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

Plantillas comunes para circuitos cu00e1nticos
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

VQC_HardwareEfficientAnsatz
""""""""""""""""""""""""""""""""""""""""


.. py:class:: pyvqnet.qnn.vqc.torch.VQC_HardwareEfficientAnsatz(n_qubits,single_rot_gate_list,entangle_gate="CNOT",entangle_rules='linear',depth=1,initial=None,dtype=None)

    Implementación de Hardware Efficient Ansatz presentado en el artículo: `Hardware-efficient Variational Quantum Eigensolver for Small Molecules <https://arxiv.org/pdf/1704.05018.pdf>`__ .

    Esta clase hereda de ``pyvqnet.qnn.vqc.torch.QModule`` y ``torch.nn.Module``.

    Esta clase se puede agregar al modelo torch como submódulo de ``torch.nn.Module``.


    :param n_qubits: Número de qubits.
    :param single_rot_gate_list: Una lista de compuertas de rotación de un solo qubit construida por una o varias compuertas de rotación que actúan en cada qubit. Actualmente soporta Rx, Ry, Rz.
    :param entangle_gate: La compuerta de entrelazamiento no parametrizada. Se soporta CNOT, CZ. Valor predeterminado: CNOT.
    :param entangle_rules: Cómo se usa la compuerta de entrelazamiento en el circuito. 'linear' significa que la compuerta de entrelazamiento actuará en cada par de qubits vecinos. 'all' significa que la compuerta de entrelazamiento actuará en cualquier par de qubits. Valor predeterminado: linear.
    :param depth: La profundidad del ansatz, valor predeterminado: 1.
    :param initial: valor inicial único para los parámetros, valor predeterminado: None, este módulo inicializará los parámetros aleatoriamente.
    :param dtype: tipo de datos de los parámetros.
    :return: una instancia de VQC_HardwareEfficientAnsatz.

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

    Una capa que consiste en una rotación de un solo qubit con un solo parámetro en cada qubit, seguida de múltiples compuertas CNOT en una combinación de cadena cerrada o anillo.

    Un anillo de compuertas CNOT conecta cada qubit con sus vecinos, y finalmente el qubit a se considera vecino del qubit a-ésimo.

    Esta clase hereda de ``pyvqnet.qnn.vqc.torch.QModule`` y ``torch.nn.Module``.

    Esta clase se puede agregar al modelo torch como submódulo de ``torch.nn.Module``.

    :param num_layers: número de capas repetidas, valor predeterminado: 1.
    :param num_qubits: número de qubits, valor predeterminado: 1.
    :param rotation: compuerta de un solo qubit con un parámetro a usar, valor predeterminado: `RX`
    :param initial: valor inicial igual para todos los parámetros. valor predeterminado: None, los parámetros se inicializarán aleatoriamente.
    :param dtype: tipo de datos del parámetro, valor predeterminado: None, usa float32.
    :return: Una instancia de VQC_BasicEntanglerTemplate

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

    Capas que consisten en rotaciones de un solo qubit y entrelazadores, como en `circuit-centric classifier design <https://arxiv.org/abs/1804.00633>`__ .

    Esta clase hereda de ``pyvqnet.qnn.vqc.torch.QModule`` y ``torch.nn.Module``.
    Esta clase se puede agregar al modelo torch como submódulo de ``torch.nn.Module``.


    :param num_layers: número de capas repetidas, valor predeterminado: 1.
    :param num_qubits: número de qubits, valor predeterminado: 1.
    :param ranges: secuencia que determina el hiperparámetro de rango para cada capa subsiguiente; valor predeterminado: None
                                usando :math: `r=l \mod M` para la :math:`l`-ésima capa y :math:`M` qubits.
    :param initial: valor inicial para todos los parámetros. valor predeterminado: None, inicializado aleatoriamente.
    :param dtype: tipo de datos del parámetro, valor predeterminado: None, usa float32.
    :return: Una instancia de VQC_StronglyEntanglingTemplate.

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
    
    Usa RZ, RY, RZ para crear circuitos cuánticos variacionales que codifican datos clásicos en estados cuánticos.
    Referencia `Quantum embeddings for machine learning <https://arxiv.org/abs/2001.03622>`_.

    Esta clase hereda de ``pyvqnet.qnn.vqc.torch.QModule`` y ``torch.nn.Module``.
    Esta clase se puede agregar al modelo torch como submódulo de ``torch.nn.Module``.

 
    :param num_repetitions_input: número de repeticiones para codificar la entrada en un submódulo.
    :param depth_input: número de dimensiones de entrada.
    :param num_unitary_layers: número de repeticiones de compuertas cuánticas variacionales.
    :param num_repetitions: número de repeticiones del submódulo.
    :param initial: valor de inicialización de parámetros, valor predeterminado None.
    :param dtype: tipo de parámetro, valor predeterminado None, usa float32.
    :param name: nombre de la clase.
    :return: Una instancia de VQC_QuantumEmbedding.

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

    19 ansatz diferentes del artículo `Expressibility and entangling capability of parameterized quantum circuits for hybrid quantum-classical algorithms <https://arxiv.org/pdf/1905.10876.pdf>`_.

    Esta clase hereda de ``pyvqnet.qnn.vqc.torch.QModule`` y ``torch.nn.Module``.

    Esta clase se puede agregar al modelo torch como submódulo de ``torch.nn.Module``.

    :param type: Tipo de circuito del 1 al 19, un total de 19 líneas.
    :param num_wires: Número de qubits.
    :param depth: Profundidad del circuito.
    :param dtype: tipo de datos del parámetro, valor predeterminado: None, usa float32.
    :param name: Nombre, valor predeterminado "".

    :return:
        una instancia de ExpressiveEntanglingAnsatz

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

    Codifica n características binarias en el estado base de n qubits de ``q_machine``. Esta función tiene el alias `VQC_BasisEmbedding`.

    Por ejemplo, para ``basis_state=([0, 1, 1])``, el estado base en el sistema cuántico es :math:`|011 \rangle`.

    :param basis_state: entrada binaria de tamaño ``(n)``.
    :param q_machine: dispositivo de máquina cuántica.
    

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

    Codifica :math:`N` características en el ángulo de rotación de :math:`n` qubits, where :math:`N \leq n`.
    Esta función tiene el alias `VQC_AngleEmbedding` .

    La rotación se puede seleccionar como: 'X' , 'Y' , 'Z', as defined by the ``rotation`` parameter:

    * ``rotation='X'`` Usa la caracter00edstica como el 00e1ngulo de rotaci00f3n RX.

    * ``rotation='Y'`` Usa la caracter00edstica como el 00e1ngulo de rotaci00f3n RY.

    * ``rotation='Z'`` Usa la caracter00edstica como el 00e1ngulo de rotaci00f3n RZ.

    ``wires`` representa el 00edndice de la compuerta de rotaci00f3n en el qubit.

    :param input_feat: Arreglo que representa los parámetros.
    :param wires: Índice del qubit.
    :param q_machine: Dispositivo de máquina cuántica.
    :param rotation: Compuerta de rotación, valor predeterminado "X".

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

    Codifica una caracter00edstica de :math:`2^n` en un vector de amplitudes de :math:`n` qubits. Esta funci00f3n tiene el alias `VQC_AmplitudeEmbedding`.

    :param input_feature: arreglo numpy que representa el par00e1metro.
    :param q_machine: dispositivo de m00e1quina cu00e1ntica.
    

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

    Codifica :math:`n` caracter00edsticas en :math:`n` qubits usando compuertas diagonales de un circuito IQP. Alias: ``VQC_IQPEmbedding`` .

    The encoding is proposed by `Havlicek et al. (2018) <https://arxiv.org/pdf/1804.11326.pdf>`_.

    By specifying ``rep`` , the basic IQP circuit can be repeated.

    :param input_feat: Array of parameters.
    :param q_machine: M00e1quina cu00e1ntica.
    :param rep: N00famero de veces para repetir el bloque de circuito cu00e1ntico, valor predeterminado 1.

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

    Combinaci00f3n arbitraria de compuertas l00f3gicas de rotaci00f3n de un solo bit cu00e1ntico. Esta funci00f3n tiene el alias: ``VQC_RotCircuit`` .

    .. math::
        R(\phi,\theta,\omega) = RZ(\omega)RY(\theta)RZ(\phi)= \begin{bmatrix}
        e^{-i(\phi+\omega)/2}\cos(\theta/2) & -e^{i(\phi-\omega)/2}\sin(\theta/2) \\
        e^{-i(\phi-\omega)/2}\sin(\theta/2) & e^{i(\phi+\omega)/2}\cos(\theta/2)
        \end{bmatrix}.

    :param q_machine: dispositivo de m00e1quina virtual cu00e1ntica.
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

    Combinaci00f3n de compuertas l00f3gicas de rotaci00f3n controlada de un solo bit cu00e1ntico. Esta funci00f3n tiene el alias: ``VQC_CRotCircuit`` .

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
    :param q_machine: Dispositivo de máquina cuántica.
    

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

    Circuito cu00e1ntico de compuerta l00f3gica Hadamard controlada. Esta funci00f3n tiene el alias: ``VQC_Controlled_Hadamard`` .

    .. math:: 
        CH = \begin{bmatrix}
        1 & 0 & 0 & 0 \\
        0 & 1 & 0 & 0 \\
        0 & 0 & \frac{1}{\sqrt{2}} & \frac{1}{\sqrt{2}} \\
        0 & 0 & \frac{1}{\sqrt{2}} & -\frac{1}{\sqrt{2}}
        \end{bmatrix}.

    :param wires: lista de 00edndices de bits cu00e1nticos, el primero es el bit de control, la longitud de la lista es 2.
    :param q_machine: dispositivo de m00e1quina virtual cu00e1ntica.

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

    :param wires: lista de 00edndices de bits cu00e1nticos, el primero es el bit de control. La longitud de la lista es 3.
    :param q_machine: dispositivo de m00e1quina virtual cu00e1ntica.

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

    Operador de excitaci00f3n 00fanica de cluster acoplado para producto tensorial de matrices Pauli. La forma matricial est00e1 dada por:

    .. math::
        \hat{U}_{pr}(\theta) = \mathrm{exp} \{ \theta_{pr} (\hat{c}_p^\dagger \hat{c}_r
        -\mathrm{H.c.}) \},

    Alias: ``VQC_FermionicSingleExcitation`` .

    :param weight: Par00e1metro en el qubit p, solo un elemento.
    :param wires: Un subconjunto de 00edndices de qubits en el intervalo [r, p]. La longitud m00ednima debe ser 2. El primer valor de 00edndice se interpreta como r, y el 00faltimo valor de 00edndice se interpreta como p. Los 00edndices intermedios son afectados por compuertas CNOT para calcular la paridad del conjunto de qubits.
    :param q_machine: Dispositivo de m00e1quina virtual cu00e1ntica.

    

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

    Operador de bi-excitaci00f3n de cluster acoplado para producto tensorial de matrices Pauli exponenciado, forma matricial dada por:

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

    Esta funci00f3n tiene el alias: ``VQC_FermionicDoubleExcitation`` .

    :param weight: par00e1metro variable
    :param wires1: representa el subconjunto de qubits en el intervalo de lista de 00edndices [s, r]. El 00edndice a-th se interpreta como s y el 00faltimo 00edndice se interpreta como r. La compuerta CNOT opera en los 00edndices intermedios para calcular la paridad de un grupo de qubits.
    :param wires2: representa el subconjunto de qubits en el intervalo de lista de 00edndices [q, p]. El primer 00edndice ra00edz se interpreta como q y el 00faltimo 00edndice se interpreta como p. La compuerta CNOT opera en los 00edndices intermedios para calcular la paridad de un grupo de qubits.
    :param q_machine: Dispositivo de m00e1quina virtual cu00e1ntica.

    

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

    Implementa la Simulaci00f3n de Excitaciones 00danicas y Dobles del Cluster Acoplado Unitario (UCCSD). UCCSD es una simulaci00f3n VQE com00fanmente usada para ejecutar simulaciones de qu00edmica cu00e1ntica.

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

    Esta funci00f3n tiene el alias: ``VQC_UCCSD`` .

    :param weights: tensor de tama00f1o ``(len(s_wires)+ len(d_wires))`` que contiene los par00e1metros :math:`\theta_{pr}` and :math:`\theta_{pqrs}` input Z rotations ``FermionicSingleExcitation`` and ``FermionicDoubleExcitation`` .
    :param wires: 00edndices de qubits para la acci00f3n de la plantilla
    :param s_wires: secuencia de listas que contienen 00edndices de qubits ``[r,...,p]`` generados por una excitaci00f3n simple :math:`\vert r, p \rangle = \hat{c}_p^\dagger \hat{c}_r \vert \mathrm{HF} \rangle`,where :math:`\vert \mathrm{HF} \rangle` denotes the Hartee-Fock reference state.
    :param d_wires: secuencia de listas, cada una conteniendo dos listas que especifican 00edndices ``[s,...,r]`` y ``[q,...,p]`` definiendo doble excitaci00f3n :math:`\vert s, r, q, p \rangle = \hat{c}_p^\dagger \hat{c}_q^\dagger \hat{c}_r\hat{c}_s \vert \mathrm{HF} \rangle` .
    :param init_state: vector de n00fameros de ocupaci00f3n de longitud ``len(wires)`` que representa el estado de alta frecuencia. ``init_state`` Estado de inicializaci00f3n del qubit.
    :param q_machine: Dispositivo de m00e1quina virtual cu00e1ntica.

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

    Circuito de evoluci00f3n Pauli Z de primer orden.

    For 3 qubits and 2 repetitions, the circuit is represented as:

    .. parsed-literal::

        ┌───┐┌──────────────┐┌───┐┌──────────────┐
        ┤ H ├┤ U1(2.0*x[0]) ├┤ H ├┤ U1(2.0*x[0]) ├
        ├───┤├──────────────┤├───┤├──────────────┤
        ┤ H ├┤ U1(2.0*x[1]) ├┤ H ├┤ U1(2.0*x[1]) ├
        ├───┤├──────────────┤├───┤├──────────────┤
        ┤ H ├┤ U1(2.0*x[2]) ├┤ H ├┤ U1(2.0*x[2]) ├
        └───┘└──────────────┘└───┘└──────────────┘

    La cadena Pauli est00e1 fijada a ``Z``. Por lo tanto, la expansi00f3n de primer orden ser00e1 un circuito sin compuertas de entrelazamiento.

    :param input_feat: Arreglo que representa los parámetros de entrada.
    :param q_machine: M00e1quina virtual cu00e1ntica.
    :param data_map_func: Matriz de mapeo de par00e1metros, una funci00f3n invocable, dise00f1ada como: ``data_map_func = lambda x: x``.
    :param rep: N00famero de veces que se repite el m00f3dulo.

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

    Circuito de evoluci00f3n Pauli-Z de segundo orden.

    For 3 qubits, 1 repeat, and linear entanglement, the circuit is represented as:

    .. parsed-literal::


        ┌───┐┌─────────────────┐
        ┤ H ├┤ U1(2.0*φ(x[0])) ├──■────────────────────────────■────────────────────────────────────
        ├───┤├─────────────────┤┌─┴─┐┌──────────────────────┐┌─┴─┐
        ┤ H ├┤ U1(2.0*φ(x[1])) ├┤ X ├┤ U1(2.0*φ(x[0],x[1])) ├┤ X ├──■────────────────────────────■──
        ├───┤├─────────────────┤└───┘└──────────────────────┘└───┘┌─┴─┐┌──────────────────────┐┌─┴─┐
        ┤ H ├┤ U1(2.0*φ(x[2])) ├──────────────────────────────────┤ X ├┤ U1(2.0*φ(x[1],x[2])) ├┤ X ├
        └───┘└─────────────────┘                                  └───┘└──────────────────────┘└───┘
    
    Donde ``φ`` es una funci00f3n no lineal cl00e1sica. Si se ingresan dos valores, ``φ(x,y) = (pi - x)(pi - y)``, y si se ingresa uno, ``φ(x) = x``. Se expresa de la siguiente manera usando ``data_map_func``:

    .. code-block::

        def data_map_func(x):
            coeff = x if x.shape[-1] == 1 else ft.reduce(lambda x, y: (np.pi - x) * (np.pi - y), x)
            return coeff

    :param input_feat: Arreglo que representa los parámetros de entrada.
    :param q_machine: M00e1quina virtual cu00e1ntica.
    :param data_map_func: matriz de mapeo de par00e1metros, una funci00f3n invocable.
    :param entanglement: estructura de entrelazamiento especificada.
    :param rep: veces de repetici00f3n del m00f3dulo.
    
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

    En este caso, tenemos cuatro excitaciones simples y dobles para preservar la proyecci00f3n de esp00edn total del estado de Hartree-Fock.

    La matriz unitaria resultante preserva la poblaci00f3n de part00edculas y prepara el sistema de n-qubits en una superposici00f3n del estado inicial de Hartree-Fock y otros estados que codifican la configuraci00f3n de m00faltiples excitaciones.

    :param weights: Un QTensor de tama00f1o ``(len(singles) + len(doubles),)`` que contiene los 00e1ngulos que entran en las operaciones vqc.qCircuit.single_excitation y vqc.qCircuit.double_excitation en secuencia
    :param q_machine: La m00e1quina cu00e1ntica.
    :param hf_state: Un vector de longitud ``len(wires)`` de n00fameros de ocupaci00f3n que representa el estado de Hartree-Fock, ``hf_state`` usado para inicializar los wires.
    :param wires: Los qubits sobre los que actuar.
    :param singles: Una secuencia de listas con los 00edndices de los dos qubits sobre los que act00faa la operaci00f3n single_excitation.
    :param doubles: Secuencia de listas con los 00edndices de los dos qubits sobre los que act00faa la operaci00f3n double_excitation.

    Por ejemplo, el circuito cu00e1ntico para dos electrones y seis qubits se muestra a continuaci00f3n:

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

    Implementa un circuito que proporciona un conjunto que puede usarse para realizar rotaciones precisas de base unitaria simple. The circuit is derived from the single-particle fermion-determined unitary transformation :math:`U(u)` given in `arXiv:1711.04789 <https://arxiv.org/abs/1711.04789>`_\ 
    
    .. math::
        U(u) = \exp{\left( \sum_{pq} \left[\log u \right]_{pq} (a_p^\dagger a_q - a_q^\dagger a_p) \right)}.

    :math:`U(u)` is obtained by using the scheme given in the paper `Optica, 3, 1460 (2016) <https://opg.optica.org/optica/fulltext.cfm?uri=optica-3-12-1460&id=355743>`_\ .

    :param q_machine: m00e1quina cu00e1ntica.
    :param wires: qubits sobre los que actuar.
    :param unitary_matrix: matriz que especifica la base para la transformaci00f3n.
    :param check: verificar si `unitary_matrix` es una matriz unitaria.

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

    Circuito cu00e1ntico que reduce la resoluci00f3n de los datos.

    Para reducir el n00famero de qubits en el circuito, primero se crean pares de qubits en el sistema. Despu00e9s de emparejar inicialmente todos los qubits, se aplica una unitaria generalizada de 2 qubits a cada par de qubits. Y despu00e9s de aplicar estas unitarias de dos qubits, se ignora un qubit en cada par de qubits para el resto de la red neuronal.

    :param sources_wires: 00cdndices de qubits fuente que ser00e1n ignorados.
    :param sinks_wires: 00cdndices de qubits destino que ser00e1n retenidos.
    :param params: Par00e1metros de entrada.
    :param q_machine: Dispositivo de m00e1quina virtual cu00e1ntica.

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


    Una capa QuantumLayer automáticamente diferenciable que usa el enfoque de matriz adjunta para calcular gradientes, see `Efficient calculation of gradients in classical simulations of variational quantum algorithms <https://arxiv.org/abs/2009.02823>`_ .

    :param general_module: una instancia de `pyvqnet.nn.Module` construida usando solo la interfaz de circuito cuántico bajo ``pyvqnet.qnn.vqc.torch``.
    :param use_qpanda: Si usar la línea qpanda para la transmisión forward, valor predeterminado: False.
    :param name: El nombre de la capa, valor predeterminado ""

    .. note::

        El QMachine de general_module debe establecer grad_method = "adjoint".

        Actualmente soporta las siguientes compuertas lógicas parametrizadas `RX`, `RY`, `RZ`, `PhaseShift`, `RXX`, `RYY`, `RZZ`, `RZX`, `U1`, `U2`, `U3` y otros circuitos variacionales que consisten en compuertas lógicas no paramétricas.


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





M00f3dulo de Circuito Cu00e1ntico Variacional con Backend de Red de Tensores
==========================================================================================

La Red de Tensores (TN) reduce significativamente la complejidad computacional descomponiendo un tensor complejo en una red de múltiples tensores de baja dimensión.

El Estado de Producto Matricial (MPS) es una forma especial de Red de Tensores. MPS representa un estado cuántico como el producto de una serie de matrices, reduciendo así efectivamente el número de parámetros y la complejidad computacional.

La siguiente interfaz se basa en el motor ``torch``, que proporciona soporte funcional para construir circuitos cuánticos en redes de tensores, incluyendo la construcción de clases base de circuitos cuánticos, compuertas lógicas cuánticas, circuitos cuánticos y mediciones, así como el cálculo de gradientes de parámetros mediante simulación de diferenciación automática en lugar del método de desplazamiento de parámetros.

La construcción de líneas cuánticas en modo MPS compensa el soporte para la construcción de líneas cuánticas de gran número de bits.

.. warning::

        El uso de las siguientes funcionalidades en este módulo requiere la instalación adicional de ``tensornetwork`` y ``torch``. La instalación predeterminada de ``pyvqnet`` no incluye estas dos dependencias. Instálelas usando ``pip install tensornetwork torch``.

.. warning::

        Habilita MPS para construir líneas cuánticas a través del parámetro ``use_mps`` en ``TNQMachine``, que soporta implementaciones de líneas cuánticas de gran número de bits (100 y más).

.. warning::
        
        El procesamiento por lotes se usa de manera diferente a los módulos clásicos, basado en el enfoque vmap, donde los datos y las líneas de construcción de parámetros deben ingresarse una dimensión menos, como se muestra en la interfaz de ejemplo a continuación, y la ejecución por lotes debe basarse tanto en ``TNQMachine`` como en ``TNQModule``.

Clase Base
------------------------------------------------

Escribir un modelo de circuito cuántico variacional en una red de tensores requiere heredar de ``TNQModule``.

TNQModule
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. py:class:: pyvqnet.qnn.vqc.tn.TNQModule(use_jit=False, vectorized_argnums=0, name="")

    .. note::

        Esta clase y sus clases derivadas solo son aplicables con ``pyvqnet.backends.set_backend("torch")``, no mezclar con el ``Module`` bajo el ``pyvqnet.nn`` predeterminado.

        Los datos en ``_buffers`` de esta clase son de tipo ``torch.Tensor``.

        Los datos en ``_parmeters`` de esta clase son de tipo ``torch.nn.Parameter``.

    :param use_jit: controla la funcionalidad de compilación jit del circuito cuántico.
    :param vectorized_argnums: los args a vectorizar,
            estos argumentos deben compartir la misma forma de lote en la primera dimensión, valor predeterminado 0.
    :param name: nombre del Módulo.

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

    Clase de simulador para computación cuántica variacional, que incluye vectores de estado cuyo atributo states son circuitos cuánticos.

    Esta clase hereda de ``pyvqnet.nn.torch.TorchModule`` y ``pyvqnet.qnn.QMachine``.

    Esta clase se puede agregar al modelo torch como submódulo de ``torch.nn.Module``.

    .. warning::
        
        In the quantum circuit of the tensor network, the ``vmap`` function will be enabled by default, and the batch dimension will be discarded in the logic gate parameters on the line.
        When using the call parameter, if the dimension is [batch_size, \*], the first batch_size dimension is discarded, and the following dimensions are used directly, e.g., for the input data x[:,1] -> x[1], and for the trainable parameter as well, see the following example for the usage of xx, weights.

    .. note::

        Before each run of the complete quantum circuit, you must use `pyvqnet.qnn.vqc.QMachine.reset_states(batchsize)` to reinitialize the initial state in the simulator and broadcast it to
        (batchsize,*) dimensions to adapt to batch data training.

    :param num_wires: número de qubits a usar
    :param dtype: tipo de datos interno usado para calcular.
    :param use_mps: abre MPSCircuit para modelos de gran número de bits.

    :return: Produce un objeto TNQMachine.

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

        obtiene los estados de la máquina cuántica de la red de tensores.

Módulo de compuerta lógica cuántica variacional
------------------------------------------------

The following function interfaces in ``pyvqnet.qnn.vqc`` directly support ``QTensor`` of ``torch`` backend for calculation, import path ``pyvqnet.qnn.vqc.tn``.

.. csv-table:: Lista de interfaces pyvqnet.qnn.vqc soportadas
    :file: ./images/same_apis_from_tn.csv

Los siguientes módulos de circuitos cuánticos heredan de ``pyvqnet.qnn.vqc.tn.TNQModule``, donde los cálculos se realizan usando ``torch.Tensor``.

.. note::

    Esta clase y sus clases derivadas solo son aplicables con ``pyvqnet.backends.set_backend("torch")``, no mezclar con el ``Module`` bajo el ``pyvqnet.nn`` predeterminado.

    Si estas clases tienen variables miembro no paramétricas ``_buffers``, los datos en ellas son de tipo ``torch.Tensor``.
    Si estas clases tienen variables miembro paramétricas ``_parmeters``, los datos en ellas son de tipo ``torch.nn.Parameter``.

I
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. py:class:: pyvqnet.qnn.vqc.tn.I(has_params: bool = False,trainable: bool = False,init_params=None,wires=None,dtype=pyvqnet.kcomplex64,use_dagger=False)
    
    define a I quantum gate .

    Esta clase hereda de ``pyvqnet.qnn.vqc.tn.QModule`` y ``torch.nn.Module``.
    Esta clase se puede agregar al modelo torch como submódulo de ``torch.nn.Module``.

    :param has_params: si tiene parámetros, compuertas como RX, RY y otras deben establecerse en True, y las que no tienen parámetros deben establecerse en False, el valor predeterminado es False.
    :param trainable: si tiene parámetros para entrenar. Si la capa usa datos de entrada externos para construir la matriz de la compuerta lógica, establezca en False. Si los parámetros a entrenar necesitan inicializarse desde esta capa, es True, el valor predeterminado es False.
    :param init_params: Parámetros de inicialización usados para codificar datos clásicos QTensor, el valor predeterminado es None.
    :param wires: Índice de bit del efecto de línea, el valor predeterminado es None.
    :param dtype: La precisión de datos de la matriz interna de la compuerta lógica puede establecerse en pyvqnet.kcomplex64 o pyvqnet.kcomplex128, correspondiendo a entrada float o double respectivamente.
    :param use_dagger: si usar la versión transpuesta conjugada de la compuerta, el valor predeterminado es False.
    :return: una instancia de ``pyvqnet.qnn.vqc.tn.QModule``

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
    
    define una compuerta cuántica Hadamard .

    Esta clase hereda de ``pyvqnet.qnn.vqc.tn.QModule`` y ``torch.nn.Module``.
    Esta clase se puede agregar al modelo torch como submódulo de ``torch.nn.Module``.

    :param has_params: si tiene parámetros, compuertas como RX, RY y otras deben establecerse en True, y las que no tienen parámetros deben establecerse en False, el valor predeterminado es False.
    :param trainable: si tiene parámetros para entrenar. Si la capa usa datos de entrada externos para construir la matriz de la compuerta lógica, establezca en False. Si los parámetros a entrenar necesitan inicializarse desde esta capa, es True, el valor predeterminado es False.
    :param init_params: Parámetros de inicialización usados para codificar datos clásicos QTensor, el valor predeterminado es None.
    :param wires: Índice de bit del efecto de línea, el valor predeterminado es None.
    :param dtype: La precisión de datos de la matriz interna de la compuerta lógica puede establecerse en pyvqnet.kcomplex64 o pyvqnet.kcomplex128, correspondiendo a entrada float o double respectivamente.
    :param use_dagger: si usar la versión transpuesta conjugada de la compuerta, el valor predeterminado es False.
    :return: una instancia de ``pyvqnet.qnn.vqc.tn.QModule``

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
    
    define una compuerta cuántica T .

    Esta clase hereda de ``pyvqnet.qnn.vqc.tn.QModule`` y ``torch.nn.Module``.
    Esta clase se puede agregar al modelo torch como submódulo de ``torch.nn.Module``.

    :param has_params: si tiene parámetros, compuertas como RX, RY y otras deben establecerse en True, y las que no tienen parámetros deben establecerse en False, el valor predeterminado es False.
    :param trainable: si tiene parámetros para entrenar. Si la capa usa datos de entrada externos para construir la matriz de la compuerta lógica, establezca en False. Si los parámetros a entrenar necesitan inicializarse desde esta capa, es True, el valor predeterminado es False.
    :param init_params: Parámetros de inicialización usados para codificar datos clásicos QTensor, el valor predeterminado es None.
    :param wires: Índice de bit del efecto de línea, el valor predeterminado es None.
    :param dtype: La precisión de datos de la matriz interna de la compuerta lógica puede establecerse en pyvqnet.kcomplex64 o pyvqnet.kcomplex128, correspondiendo a entrada float o double respectivamente.
    :param use_dagger: si usar la versión transpuesta conjugada de la compuerta, el valor predeterminado es False.
    :return: una instancia de ``pyvqnet.qnn.vqc.tn.QModule``

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
    
    define una compuerta cuántica S .

    Esta clase hereda de ``pyvqnet.qnn.vqc.tn.QModule`` y ``torch.nn.Module``.
    Esta clase se puede agregar al modelo torch como submódulo de ``torch.nn.Module``.

    :param has_params: si tiene parámetros, compuertas como RX, RY y otras deben establecerse en True, y las que no tienen parámetros deben establecerse en False, el valor predeterminado es False.
    :param trainable: si tiene parámetros para entrenar. Si la capa usa datos de entrada externos para construir la matriz de la compuerta lógica, establezca en False. Si los parámetros a entrenar necesitan inicializarse desde esta capa, es True, el valor predeterminado es False.
    :param init_params: Parámetros de inicialización usados para codificar datos clásicos QTensor, el valor predeterminado es None.
    :param wires: Índice de bit del efecto de línea, el valor predeterminado es None.
    :param dtype: La precisión de datos de la matriz interna de la compuerta lógica puede establecerse en pyvqnet.kcomplex64 o pyvqnet.kcomplex128, correspondiendo a entrada float o double respectivamente.
    :param use_dagger: si usar la versión transpuesta conjugada de la compuerta, el valor predeterminado es False.
    :return: una instancia de ``pyvqnet.qnn.vqc.tn.QModule``

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
    
    define una compuerta cuántica PauliX .

    Esta clase hereda de ``pyvqnet.qnn.vqc.tn.QModule`` y ``torch.nn.Module``.
    Esta clase se puede agregar al modelo torch como submódulo de ``torch.nn.Module``.


    :param has_params: si tiene parámetros, compuertas como RX, RY y otras deben establecerse en True, y las que no tienen parámetros deben establecerse en False, el valor predeterminado es False.
    :param trainable: si tiene parámetros para entrenar. Si la capa usa datos de entrada externos para construir la matriz de la compuerta lógica, establezca en False. Si los parámetros a entrenar necesitan inicializarse desde esta capa, es True, el valor predeterminado es False.
    :param init_params: Parámetros de inicialización usados para codificar datos clásicos QTensor, el valor predeterminado es None.
    :param wires: Índice de bit del efecto de línea, el valor predeterminado es None.
    :param dtype: La precisión de datos de la matriz interna de la compuerta lógica puede establecerse en pyvqnet.kcomplex64 o pyvqnet.kcomplex128, correspondiendo a entrada float o double respectivamente.
    :param use_dagger: si usar la versión transpuesta conjugada de la compuerta, el valor predeterminado es False.
    :return: una instancia de ``pyvqnet.qnn.vqc.tn.QModule``

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
    
    define una compuerta cuántica PauliY .

    Esta clase hereda de ``pyvqnet.qnn.vqc.tn.QModule`` y ``torch.nn.Module``.
    Esta clase se puede agregar al modelo torch como submódulo de ``torch.nn.Module``.


    :param has_params: si tiene parámetros, compuertas como RX, RY y otras deben establecerse en True, y las que no tienen parámetros deben establecerse en False, el valor predeterminado es False.
    :param trainable: si tiene parámetros para entrenar. Si la capa usa datos de entrada externos para construir la matriz de la compuerta lógica, establezca en False. Si los parámetros a entrenar necesitan inicializarse desde esta capa, es True, el valor predeterminado es False.
    :param init_params: Parámetros de inicialización usados para codificar datos clásicos QTensor, el valor predeterminado es None.
    :param wires: Índice de bit del efecto de línea, el valor predeterminado es None.
    :param dtype: La precisión de datos de la matriz interna de la compuerta lógica puede establecerse en pyvqnet.kcomplex64 o pyvqnet.kcomplex128, correspondiendo a entrada float o double respectivamente.
    :param use_dagger: si usar la versión transpuesta conjugada de la compuerta, el valor predeterminado es False.
    :return: una instancia de ``pyvqnet.qnn.vqc.tn.QModule``

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
    
    define una compuerta cuántica PauliZ .

    Esta clase hereda de ``pyvqnet.qnn.vqc.tn.QModule`` y ``torch.nn.Module``.
    Esta clase se puede agregar al modelo torch como submódulo de ``torch.nn.Module``.


    :param has_params: si tiene parámetros, compuertas como RX, RY y otras deben establecerse en True, y las que no tienen parámetros deben establecerse en False, el valor predeterminado es False.
    :param trainable: si tiene parámetros para entrenar. Si la capa usa datos de entrada externos para construir la matriz de la compuerta lógica, establezca en False. Si los parámetros a entrenar necesitan inicializarse desde esta capa, es True, el valor predeterminado es False.
    :param init_params: Parámetros de inicialización usados para codificar datos clásicos QTensor, el valor predeterminado es None.
    :param wires: Índice de bit del efecto de línea, el valor predeterminado es None.
    :param dtype: La precisión de datos de la matriz interna de la compuerta lógica puede establecerse en pyvqnet.kcomplex64 o pyvqnet.kcomplex128, correspondiendo a entrada float o double respectivamente.
    :param use_dagger: si usar la versión transpuesta conjugada de la compuerta, el valor predeterminado es False.
    :return: una instancia de ``pyvqnet.qnn.vqc.tn.QModule``

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
    
    define una compuerta cuántica X1 .

    Esta clase hereda de ``pyvqnet.qnn.vqc.tn.QModule`` y ``torch.nn.Module``.
    Esta clase se puede agregar al modelo torch como submódulo de ``torch.nn.Module``.

    :param has_params: si tiene parámetros, compuertas como RX, RY y otras deben establecerse en True, y las que no tienen parámetros deben establecerse en False, el valor predeterminado es False.
    :param trainable: si tiene parámetros para entrenar. Si la capa usa datos de entrada externos para construir la matriz de la compuerta lógica, establezca en False. Si los parámetros a entrenar necesitan inicializarse desde esta capa, es True, el valor predeterminado es False.
    :param init_params: Parámetros de inicialización usados para codificar datos clásicos QTensor, el valor predeterminado es None.
    :param wires: Índice de bit del efecto de línea, el valor predeterminado es None.
    :param dtype: La precisión de datos de la matriz interna de la compuerta lógica puede establecerse en pyvqnet.kcomplex64 o pyvqnet.kcomplex128, correspondiendo a entrada float o double respectivamente.
    :param use_dagger: si usar la versión transpuesta conjugada de la compuerta, el valor predeterminado es False.
    :return: una instancia de ``pyvqnet.qnn.vqc.tn.QModule``

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
    
    define una compuerta cuántica RX .

    Esta clase hereda de ``pyvqnet.qnn.vqc.tn.QModule`` y ``torch.nn.Module``.
    Esta clase se puede agregar al modelo torch como submódulo de ``torch.nn.Module``.

    :param has_params: si tiene parámetros, compuertas como RX, RY y otras deben establecerse en True, y las que no tienen parámetros deben establecerse en False, el valor predeterminado es False.
    :param trainable: si tiene parámetros para entrenar. Si la capa usa datos de entrada externos para construir la matriz de la compuerta lógica, establezca en False. Si los parámetros a entrenar necesitan inicializarse desde esta capa, es True, el valor predeterminado es False.
    :param init_params: Parámetros de inicialización usados para codificar datos clásicos QTensor, el valor predeterminado es None.
    :param wires: Índice de bit del efecto de línea, el valor predeterminado es None.
    :param dtype: La precisión de datos de la matriz interna de la compuerta lógica puede establecerse en pyvqnet.kcomplex64 o pyvqnet.kcomplex128, correspondiendo a entrada float o double respectivamente.
    :param use_dagger: si usar la versión transpuesta conjugada de la compuerta, el valor predeterminado es False.
    :return: una instancia de ``pyvqnet.qnn.vqc.tn.QModule``

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
    
    define una compuerta cuántica RY .

    Esta clase hereda de ``pyvqnet.qnn.vqc.tn.QModule`` y ``torch.nn.Module``.
    Esta clase se puede agregar al modelo torch como submódulo de ``torch.nn.Module``.

    :param has_params: si tiene parámetros, compuertas como RX, RY y otras deben establecerse en True, y las que no tienen parámetros deben establecerse en False, el valor predeterminado es False.
    :param trainable: si tiene parámetros para entrenar. Si la capa usa datos de entrada externos para construir la matriz de la compuerta lógica, establezca en False. Si los parámetros a entrenar necesitan inicializarse desde esta capa, es True, el valor predeterminado es False.
    :param init_params: Parámetros de inicialización usados para codificar datos clásicos QTensor, el valor predeterminado es None.
    :param wires: Índice de bit del efecto de línea, el valor predeterminado es None.
    :param dtype: La precisión de datos de la matriz interna de la compuerta lógica puede establecerse en pyvqnet.kcomplex64 o pyvqnet.kcomplex128, correspondiendo a entrada float o double respectivamente.
    :param use_dagger: si usar la versión transpuesta conjugada de la compuerta, el valor predeterminado es False.
    :return: una instancia de ``pyvqnet.qnn.vqc.tn.QModule``

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
    
    define una compuerta cuántica RZ .

    Esta clase hereda de ``pyvqnet.qnn.vqc.tn.QModule`` y ``torch.nn.Module``.
    Esta clase se puede agregar al modelo torch como submódulo de ``torch.nn.Module``.

    :param has_params: si tiene parámetros, compuertas como RX, RY y otras deben establecerse en True, y las que no tienen parámetros deben establecerse en False, el valor predeterminado es False.
    :param trainable: si tiene parámetros para entrenar. Si la capa usa datos de entrada externos para construir la matriz de la compuerta lógica, establezca en False. Si los parámetros a entrenar necesitan inicializarse desde esta capa, es True, el valor predeterminado es False.
    :param init_params: Parámetros de inicialización usados para codificar datos clásicos QTensor, el valor predeterminado es None.
    :param wires: Índice de bit del efecto de línea, el valor predeterminado es None.
    :param dtype: La precisión de datos de la matriz interna de la compuerta lógica puede establecerse en pyvqnet.kcomplex64 o pyvqnet.kcomplex128, correspondiendo a entrada float o double respectivamente.
    :param use_dagger: si usar la versión transpuesta conjugada de la compuerta, el valor predeterminado es False.
    :return: una instancia de ``pyvqnet.qnn.vqc.tn.QModule``

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
    
    define una compuerta cuántica CRX .

    Esta clase hereda de ``pyvqnet.qnn.vqc.tn.QModule`` y ``torch.nn.Module``.
    Esta clase se puede agregar al modelo torch como submódulo de ``torch.nn.Module``.

    :param has_params: si tiene parámetros, compuertas como RX, RY y otras deben establecerse en True, y las que no tienen parámetros deben establecerse en False, el valor predeterminado es False.
    :param trainable: si tiene parámetros para entrenar. Si la capa usa datos de entrada externos para construir la matriz de la compuerta lógica, establezca en False. Si los parámetros a entrenar necesitan inicializarse desde esta capa, es True, el valor predeterminado es False.
    :param init_params: Parámetros de inicialización usados para codificar datos clásicos QTensor, el valor predeterminado es None.
    :param wires: Índice de bit del efecto de línea, el valor predeterminado es None.
    :param dtype: La precisión de datos de la matriz interna de la compuerta lógica puede establecerse en pyvqnet.kcomplex64 o pyvqnet.kcomplex128, correspondiendo a entrada float o double respectivamente.
    :param use_dagger: si usar la versión transpuesta conjugada de la compuerta, el valor predeterminado es False.
    :return: una instancia de ``pyvqnet.qnn.vqc.tn.QModule``

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
    
    define una compuerta cuántica CRY .

    Esta clase hereda de ``pyvqnet.qnn.vqc.tn.QModule`` y ``torch.nn.Module``.
    Esta clase se puede agregar al modelo torch como submódulo de ``torch.nn.Module``.

    :param has_params: si tiene parámetros, compuertas como RX, RY y otras deben establecerse en True, y las que no tienen parámetros deben establecerse en False, el valor predeterminado es False.
    :param trainable: si tiene parámetros para entrenar. Si la capa usa datos de entrada externos para construir la matriz de la compuerta lógica, establezca en False. Si los parámetros a entrenar necesitan inicializarse desde esta capa, es True, el valor predeterminado es False.
    :param init_params: Parámetros de inicialización usados para codificar datos clásicos QTensor, el valor predeterminado es None.
    :param wires: Índice de bit del efecto de línea, el valor predeterminado es None.
    :param dtype: La precisión de datos de la matriz interna de la compuerta lógica puede establecerse en pyvqnet.kcomplex64 o pyvqnet.kcomplex128, correspondiendo a entrada float o double respectivamente.
    :param use_dagger: si usar la versión transpuesta conjugada de la compuerta, el valor predeterminado es False.
    :return: una instancia de ``pyvqnet.qnn.vqc.tn.QModule``

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
    
    define una compuerta cuántica CRZ .

    Esta clase hereda de ``pyvqnet.qnn.vqc.tn.QModule`` y ``torch.nn.Module``.
    Esta clase se puede agregar al modelo torch como submódulo de ``torch.nn.Module``.

    :param has_params: si tiene parámetros, compuertas como RX, RY y otras deben establecerse en True, y las que no tienen parámetros deben establecerse en False, el valor predeterminado es False.
    :param trainable: si tiene parámetros para entrenar. Si la capa usa datos de entrada externos para construir la matriz de la compuerta lógica, establezca en False. Si los parámetros a entrenar necesitan inicializarse desde esta capa, es True, el valor predeterminado es False.
    :param init_params: Parámetros de inicialización usados para codificar datos clásicos QTensor, el valor predeterminado es None.
    :param wires: Índice de bit del efecto de línea, el valor predeterminado es None.
    :param dtype: La precisión de datos de la matriz interna de la compuerta lógica puede establecerse en pyvqnet.kcomplex64 o pyvqnet.kcomplex128, correspondiendo a entrada float o double respectivamente.
    :param use_dagger: si usar la versión transpuesta conjugada de la compuerta, el valor predeterminado es False.
    :return: una instancia de ``pyvqnet.qnn.vqc.tn.QModule``

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
    
    define una compuerta cuántica U1 .

    Esta clase hereda de ``pyvqnet.qnn.vqc.tn.QModule`` y ``torch.nn.Module``.
    Esta clase se puede agregar al modelo torch como submódulo de ``torch.nn.Module``.

    :param has_params: si tiene parámetros, compuertas como RX, RY y otras deben establecerse en True, y las que no tienen parámetros deben establecerse en False, el valor predeterminado es False.
    :param trainable: si tiene parámetros para entrenar. Si la capa usa datos de entrada externos para construir la matriz de la compuerta lógica, establezca en False. Si los parámetros a entrenar necesitan inicializarse desde esta capa, es True, el valor predeterminado es False.
    :param init_params: Parámetros de inicialización usados para codificar datos clásicos QTensor, el valor predeterminado es None.
    :param wires: Índice de bit del efecto de línea, el valor predeterminado es None.
    :param dtype: La precisión de datos de la matriz interna de la compuerta lógica puede establecerse en pyvqnet.kcomplex64 o pyvqnet.kcomplex128, correspondiendo a entrada float o double respectivamente.
    :param use_dagger: si usar la versión transpuesta conjugada de la compuerta, el valor predeterminado es False.
    :return: una instancia de ``pyvqnet.qnn.vqc.tn.QModule``

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
    
    define una compuerta cuántica U2 .

    Esta clase hereda de ``pyvqnet.qnn.vqc.tn.QModule`` y ``torch.nn.Module``.
    Esta clase se puede agregar al modelo torch como submódulo de ``torch.nn.Module``.

    :param has_params: si tiene parámetros, compuertas como RX, RY y otras deben establecerse en True, y las que no tienen parámetros deben establecerse en False, el valor predeterminado es False.
    :param trainable: si tiene parámetros para entrenar. Si la capa usa datos de entrada externos para construir la matriz de la compuerta lógica, establezca en False. Si los parámetros a entrenar necesitan inicializarse desde esta capa, es True, el valor predeterminado es False.
    :param init_params: Parámetros de inicialización usados para codificar datos clásicos QTensor, el valor predeterminado es None.
    :param wires: Índice de bit del efecto de línea, el valor predeterminado es None.
    :param dtype: La precisión de datos de la matriz interna de la compuerta lógica puede establecerse en pyvqnet.kcomplex64 o pyvqnet.kcomplex128, correspondiendo a entrada float o double respectivamente.
    :param use_dagger: si usar la versión transpuesta conjugada de la compuerta, el valor predeterminado es False.
    :return: una instancia de ``pyvqnet.qnn.vqc.tn.QModule``

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
    
    define una compuerta cuántica U3 .

    Esta clase hereda de ``pyvqnet.qnn.vqc.tn.QModule`` y ``torch.nn.Module``.
    Esta clase se puede agregar al modelo torch como submódulo de ``torch.nn.Module``.

    :param has_params: si tiene parámetros, compuertas como RX, RY y otras deben establecerse en True, y las que no tienen parámetros deben establecerse en False, el valor predeterminado es False.
    :param trainable: si tiene parámetros para entrenar. Si la capa usa datos de entrada externos para construir la matriz de la compuerta lógica, establezca en False. Si los parámetros a entrenar necesitan inicializarse desde esta capa, es True, el valor predeterminado es False.
    :param init_params: Parámetros de inicialización usados para codificar datos clásicos QTensor, el valor predeterminado es None.
    :param wires: Índice de bit del efecto de línea, el valor predeterminado es None.
    :param dtype: La precisión de datos de la matriz interna de la compuerta lógica puede establecerse en pyvqnet.kcomplex64 o pyvqnet.kcomplex128, correspondiendo a entrada float o double respectivamente.
    :param use_dagger: si usar la versión transpuesta conjugada de la compuerta, el valor predeterminado es False.
    :return: una instancia de ``pyvqnet.qnn.vqc.tn.QModule``

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
    
    define una compuerta cuántica CNOT , alias `CX` .

    Esta clase hereda de ``pyvqnet.qnn.vqc.tn.QModule`` y ``torch.nn.Module``.
    Esta clase se puede agregar al modelo torch como submódulo de ``torch.nn.Module``.

    :param has_params: si tiene parámetros, compuertas como RX, RY y otras deben establecerse en True, y las que no tienen parámetros deben establecerse en False, el valor predeterminado es False.
    :param trainable: si tiene parámetros para entrenar. Si la capa usa datos de entrada externos para construir la matriz de la compuerta lógica, establezca en False. Si los parámetros a entrenar necesitan inicializarse desde esta capa, es True, el valor predeterminado es False.
    :param init_params: Parámetros de inicialización usados para codificar datos clásicos QTensor, el valor predeterminado es None.
    :param wires: Índice de bit del efecto de línea, el valor predeterminado es None.
    :param dtype: La precisión de datos de la matriz interna de la compuerta lógica puede establecerse en pyvqnet.kcomplex64 o pyvqnet.kcomplex128, correspondiendo a entrada float o double respectivamente.
    :param use_dagger: si usar la versión transpuesta conjugada de la compuerta, el valor predeterminado es False.
    :return: una instancia de ``pyvqnet.qnn.vqc.tn.QModule``

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
    
    define una compuerta cuántica CY .

    Esta clase hereda de ``pyvqnet.qnn.vqc.tn.QModule`` y ``torch.nn.Module``.
    Esta clase se puede agregar al modelo torch como submódulo de ``torch.nn.Module``.

    :param has_params: si tiene parámetros, compuertas como RX, RY y otras deben establecerse en True, y las que no tienen parámetros deben establecerse en False, el valor predeterminado es False.
    :param trainable: si tiene parámetros para entrenar. Si la capa usa datos de entrada externos para construir la matriz de la compuerta lógica, establezca en False. Si los parámetros a entrenar necesitan inicializarse desde esta capa, es True, el valor predeterminado es False.
    :param init_params: Parámetros de inicialización usados para codificar datos clásicos QTensor, el valor predeterminado es None.
    :param wires: Índice de bit del efecto de línea, el valor predeterminado es None.
    :param dtype: La precisión de datos de la matriz interna de la compuerta lógica puede establecerse en pyvqnet.kcomplex64 o pyvqnet.kcomplex128, correspondiendo a entrada float o double respectivamente.
    :param use_dagger: si usar la versión transpuesta conjugada de la compuerta, el valor predeterminado es False.
    :return: una instancia de ``pyvqnet.qnn.vqc.tn.QModule``

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
    
    define una compuerta cuántica CZ .

    Esta clase hereda de ``pyvqnet.qnn.vqc.tn.QModule`` y ``torch.nn.Module``.
    Esta clase se puede agregar al modelo torch como submódulo de ``torch.nn.Module``.

    :param has_params: si tiene parámetros, compuertas como RX, RY y otras deben establecerse en True, y las que no tienen parámetros deben establecerse en False, el valor predeterminado es False.
    :param trainable: si tiene parámetros para entrenar. Si la capa usa datos de entrada externos para construir la matriz de la compuerta lógica, establezca en False. Si los parámetros a entrenar necesitan inicializarse desde esta capa, es True, el valor predeterminado es False.
    :param init_params: Parámetros de inicialización usados para codificar datos clásicos QTensor, el valor predeterminado es None.
    :param wires: Índice de bit del efecto de línea, el valor predeterminado es None.
    :param dtype: La precisión de datos de la matriz interna de la compuerta lógica puede establecerse en pyvqnet.kcomplex64 o pyvqnet.kcomplex128, correspondiendo a entrada float o double respectivamente.
    :param use_dagger: si usar la versión transpuesta conjugada de la compuerta, el valor predeterminado es False.
    :return: una instancia de ``pyvqnet.qnn.vqc.tn.QModule``

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
    
    define una compuerta cuántica CR .

    Esta clase hereda de ``pyvqnet.qnn.vqc.tn.QModule`` y ``torch.nn.Module``.
    Esta clase se puede agregar al modelo torch como submódulo de ``torch.nn.Module``.

    :param has_params: si tiene parámetros, compuertas como RX, RY y otras deben establecerse en True, y las que no tienen parámetros deben establecerse en False, el valor predeterminado es False.
    :param trainable: si tiene parámetros para entrenar. Si la capa usa datos de entrada externos para construir la matriz de la compuerta lógica, establezca en False. Si los parámetros a entrenar necesitan inicializarse desde esta capa, es True, el valor predeterminado es False.
    :param init_params: Parámetros de inicialización usados para codificar datos clásicos QTensor, el valor predeterminado es None.
    :param wires: Índice de bit del efecto de línea, el valor predeterminado es None.
    :param dtype: La precisión de datos de la matriz interna de la compuerta lógica puede establecerse en pyvqnet.kcomplex64 o pyvqnet.kcomplex128, correspondiendo a entrada float o double respectivamente.
    :param use_dagger: si usar la versión transpuesta conjugada de la compuerta, el valor predeterminado es False.
    :return: una instancia de ``pyvqnet.qnn.vqc.tn.QModule``

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
    
    define una compuerta cuántica SWAP .

    Esta clase hereda de ``pyvqnet.qnn.vqc.tn.QModule`` y ``torch.nn.Module``.
    Esta clase se puede agregar al modelo torch como submódulo de ``torch.nn.Module``.

    :param has_params: si tiene parámetros, compuertas como RX, RY y otras deben establecerse en True, y las que no tienen parámetros deben establecerse en False, el valor predeterminado es False.
    :param trainable: si tiene parámetros para entrenar. Si la capa usa datos de entrada externos para construir la matriz de la compuerta lógica, establezca en False. Si los parámetros a entrenar necesitan inicializarse desde esta capa, es True, el valor predeterminado es False.
    :param init_params: Parámetros de inicialización usados para codificar datos clásicos QTensor, el valor predeterminado es None.
    :param wires: Índice de bit del efecto de línea, el valor predeterminado es None.
    :param dtype: La precisión de datos de la matriz interna de la compuerta lógica puede establecerse en pyvqnet.kcomplex64 o pyvqnet.kcomplex128, correspondiendo a entrada float o double respectivamente.
    :param use_dagger: si usar la versión transpuesta conjugada de la compuerta, el valor predeterminado es False.
    :return: una instancia de ``pyvqnet.qnn.vqc.tn.QModule``

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
    
    define una compuerta cuántica SWAP .

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

    Esta clase hereda de ``pyvqnet.qnn.vqc.tn.QModule`` y ``torch.nn.Module``.
    Esta clase se puede agregar al modelo torch como submódulo de ``torch.nn.Module``.

    :param has_params: si tiene parámetros, compuertas como RX, RY y otras deben establecerse en True, y las que no tienen parámetros deben establecerse en False, el valor predeterminado es False.
    :param trainable: si tiene parámetros para entrenar. Si la capa usa datos de entrada externos para construir la matriz de la compuerta lógica, establezca en False. Si los parámetros a entrenar necesitan inicializarse desde esta capa, es True, el valor predeterminado es False.
    :param init_params: Parámetros de inicialización usados para codificar datos clásicos QTensor, el valor predeterminado es None.
    :param wires: Índice de bit del efecto de línea, el valor predeterminado es None.
    :param dtype: La precisión de datos de la matriz interna de la compuerta lógica puede establecerse en pyvqnet.kcomplex64 o pyvqnet.kcomplex128, correspondiendo a entrada float o double respectivamente.
    :param use_dagger: si usar la versión transpuesta conjugada de la compuerta, el valor predeterminado es False.
    :return: una instancia de ``pyvqnet.qnn.vqc.tn.QModule``

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
    
    define una compuerta cuántica RXX .

    Esta clase hereda de ``pyvqnet.qnn.vqc.tn.QModule`` y ``torch.nn.Module``.
    Esta clase se puede agregar al modelo torch como submódulo de ``torch.nn.Module``.

    :param has_params: si tiene parámetros, compuertas como RX, RY y otras deben establecerse en True, y las que no tienen parámetros deben establecerse en False, el valor predeterminado es False.
    :param trainable: si tiene parámetros para entrenar. Si la capa usa datos de entrada externos para construir la matriz de la compuerta lógica, establezca en False. Si los parámetros a entrenar necesitan inicializarse desde esta capa, es True, el valor predeterminado es False.
    :param init_params: Parámetros de inicialización usados para codificar datos clásicos QTensor, el valor predeterminado es None.
    :param wires: Índice de bit del efecto de línea, el valor predeterminado es None.
    :param dtype: La precisión de datos de la matriz interna de la compuerta lógica puede establecerse en pyvqnet.kcomplex64 o pyvqnet.kcomplex128, correspondiendo a entrada float o double respectivamente.
    :param use_dagger: si usar la versión transpuesta conjugada de la compuerta, el valor predeterminado es False.
    :return: una instancia de ``pyvqnet.qnn.vqc.tn.QModule``

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
    
    define una compuerta cuántica RYY .

    Esta clase hereda de ``pyvqnet.qnn.vqc.tn.QModule`` y ``torch.nn.Module``.
    Esta clase se puede agregar al modelo torch como submódulo de ``torch.nn.Module``.

    :param has_params: si tiene parámetros, compuertas como RX, RY y otras deben establecerse en True, y las que no tienen parámetros deben establecerse en False, el valor predeterminado es False.
    :param trainable: si tiene parámetros para entrenar. Si la capa usa datos de entrada externos para construir la matriz de la compuerta lógica, establezca en False. Si los parámetros a entrenar necesitan inicializarse desde esta capa, es True, el valor predeterminado es False.
    :param init_params: Parámetros de inicialización usados para codificar datos clásicos QTensor, el valor predeterminado es None.
    :param wires: Índice de bit del efecto de línea, el valor predeterminado es None.
    :param dtype: La precisión de datos de la matriz interna de la compuerta lógica puede establecerse en pyvqnet.kcomplex64 o pyvqnet.kcomplex128, correspondiendo a entrada float o double respectivamente.
    :param use_dagger: si usar la versión transpuesta conjugada de la compuerta, el valor predeterminado es False.
    :return: una instancia de ``pyvqnet.qnn.vqc.tn.QModule``

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
    
    define una compuerta cuántica RZZ .

    Esta clase hereda de ``pyvqnet.qnn.vqc.tn.QModule`` y ``torch.nn.Module``.
    Esta clase se puede agregar al modelo torch como submódulo de ``torch.nn.Module``.

    :param has_params: si tiene parámetros, compuertas como RX, RY y otras deben establecerse en True, y las que no tienen parámetros deben establecerse en False, el valor predeterminado es False.
    :param trainable: si tiene parámetros para entrenar. Si la capa usa datos de entrada externos para construir la matriz de la compuerta lógica, establezca en False. Si los parámetros a entrenar necesitan inicializarse desde esta capa, es True, el valor predeterminado es False.
    :param init_params: Parámetros de inicialización usados para codificar datos clásicos QTensor, el valor predeterminado es None.
    :param wires: Índice de bit del efecto de línea, el valor predeterminado es None.
    :param dtype: La precisión de datos de la matriz interna de la compuerta lógica puede establecerse en pyvqnet.kcomplex64 o pyvqnet.kcomplex128, correspondiendo a entrada float o double respectivamente.
    :param use_dagger: si usar la versión transpuesta conjugada de la compuerta, el valor predeterminado es False.
    :return: una instancia de ``pyvqnet.qnn.vqc.tn.QModule``

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
    
    define una compuerta cuántica RZX .

    Esta clase hereda de ``pyvqnet.qnn.vqc.tn.QModule`` y ``torch.nn.Module``.
    Esta clase se puede agregar al modelo torch como submódulo de ``torch.nn.Module``.

    :param has_params: si tiene parámetros, compuertas como RX, RY y otras deben establecerse en True, y las que no tienen parámetros deben establecerse en False, el valor predeterminado es False.
    :param trainable: si tiene parámetros para entrenar. Si la capa usa datos de entrada externos para construir la matriz de la compuerta lógica, establezca en False. Si los parámetros a entrenar necesitan inicializarse desde esta capa, es True, el valor predeterminado es False.
    :param init_params: Parámetros de inicialización usados para codificar datos clásicos QTensor, el valor predeterminado es None.
    :param wires: Índice de bit del efecto de línea, el valor predeterminado es None.
    :param dtype: La precisión de datos de la matriz interna de la compuerta lógica puede establecerse en pyvqnet.kcomplex64 o pyvqnet.kcomplex128, correspondiendo a entrada float o double respectivamente.
    :param use_dagger: si usar la versión transpuesta conjugada de la compuerta, el valor predeterminado es False.
    :return: una instancia de ``pyvqnet.qnn.vqc.tn.QModule``

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
    
    define una compuerta cuántica Toffoli .

    Esta clase hereda de ``pyvqnet.qnn.vqc.tn.QModule`` y ``torch.nn.Module``.
    Esta clase se puede agregar al modelo torch como submódulo de ``torch.nn.Module``.

    :param has_params: si tiene parámetros, compuertas como RX, RY y otras deben establecerse en True, y las que no tienen parámetros deben establecerse en False, el valor predeterminado es False.
    :param trainable: si tiene parámetros para entrenar. Si la capa usa datos de entrada externos para construir la matriz de la compuerta lógica, establezca en False. Si los parámetros a entrenar necesitan inicializarse desde esta capa, es True, el valor predeterminado es False.
    :param init_params: Parámetros de inicialización usados para codificar datos clásicos QTensor, el valor predeterminado es None.
    :param wires: Índice de bit del efecto de línea, el valor predeterminado es None.
    :param dtype: La precisión de datos de la matriz interna de la compuerta lógica puede establecerse en pyvqnet.kcomplex64 o pyvqnet.kcomplex128, correspondiendo a entrada float o double respectivamente.
    :param use_dagger: si usar la versión transpuesta conjugada de la compuerta, el valor predeterminado es False.
    :return: una instancia de ``pyvqnet.qnn.vqc.tn.QModule``

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
    
    define una compuerta cuántica IsingXX .

    Esta clase hereda de ``pyvqnet.qnn.vqc.tn.QModule`` y ``torch.nn.Module``.
    Esta clase se puede agregar al modelo torch como submódulo de ``torch.nn.Module``.

    :param has_params: si tiene parámetros, compuertas como RX, RY y otras deben establecerse en True, y las que no tienen parámetros deben establecerse en False, el valor predeterminado es False.
    :param trainable: si tiene parámetros para entrenar. Si la capa usa datos de entrada externos para construir la matriz de la compuerta lógica, establezca en False. Si los parámetros a entrenar necesitan inicializarse desde esta capa, es True, el valor predeterminado es False.
    :param init_params: Parámetros de inicialización usados para codificar datos clásicos QTensor, el valor predeterminado es None.
    :param wires: Índice de bit del efecto de línea, el valor predeterminado es None.
    :param dtype: La precisión de datos de la matriz interna de la compuerta lógica puede establecerse en pyvqnet.kcomplex64 o pyvqnet.kcomplex128, correspondiendo a entrada float o double respectivamente.
    :param use_dagger: si usar la versión transpuesta conjugada de la compuerta, el valor predeterminado es False.
    :return: una instancia de ``pyvqnet.qnn.vqc.tn.QModule``

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
    
    define una compuerta cuántica IsingYY .

    Esta clase hereda de ``pyvqnet.qnn.vqc.tn.QModule`` y ``torch.nn.Module``.
    Esta clase se puede agregar al modelo torch como submódulo de ``torch.nn.Module``.

    :param has_params: si tiene parámetros, compuertas como RX, RY y otras deben establecerse en True, y las que no tienen parámetros deben establecerse en False, el valor predeterminado es False.
    :param trainable: si tiene parámetros para entrenar. Si la capa usa datos de entrada externos para construir la matriz de la compuerta lógica, establezca en False. Si los parámetros a entrenar necesitan inicializarse desde esta capa, es True, el valor predeterminado es False.
    :param init_params: Parámetros de inicialización usados para codificar datos clásicos QTensor, el valor predeterminado es None.
    :param wires: Índice de bit del efecto de línea, el valor predeterminado es None.
    :param dtype: La precisión de datos de la matriz interna de la compuerta lógica puede establecerse en pyvqnet.kcomplex64 o pyvqnet.kcomplex128, correspondiendo a entrada float o double respectivamente.
    :param use_dagger: si usar la versión transpuesta conjugada de la compuerta, el valor predeterminado es False.
    :return: una instancia de ``pyvqnet.qnn.vqc.tn.QModule``

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
    
    define una compuerta cuántica IsingZZ .

    Esta clase hereda de ``pyvqnet.qnn.vqc.tn.QModule`` y ``torch.nn.Module``.
    Esta clase se puede agregar al modelo torch como submódulo de ``torch.nn.Module``.

    :param has_params: si tiene parámetros, compuertas como RX, RY y otras deben establecerse en True, y las que no tienen parámetros deben establecerse en False, el valor predeterminado es False.
    :param trainable: si tiene parámetros para entrenar. Si la capa usa datos de entrada externos para construir la matriz de la compuerta lógica, establezca en False. Si los parámetros a entrenar necesitan inicializarse desde esta capa, es True, el valor predeterminado es False.
    :param init_params: Parámetros de inicialización usados para codificar datos clásicos QTensor, el valor predeterminado es None.
    :param wires: Índice de bit del efecto de línea, el valor predeterminado es None.
    :param dtype: La precisión de datos de la matriz interna de la compuerta lógica puede establecerse en pyvqnet.kcomplex64 o pyvqnet.kcomplex128, correspondiendo a entrada float o double respectivamente.
    :param use_dagger: si usar la versión transpuesta conjugada de la compuerta, el valor predeterminado es False.
    :return: una instancia de ``pyvqnet.qnn.vqc.tn.QModule``

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
    
    define una compuerta cuántica IsingXY .

    Esta clase hereda de ``pyvqnet.qnn.vqc.tn.QModule`` y ``torch.nn.Module``.
    Esta clase se puede agregar al modelo torch como submódulo de ``torch.nn.Module``.

    :param has_params: si tiene parámetros, compuertas como RX, RY y otras deben establecerse en True, y las que no tienen parámetros deben establecerse en False, el valor predeterminado es False.
    :param trainable: si tiene parámetros para entrenar. Si la capa usa datos de entrada externos para construir la matriz de la compuerta lógica, establezca en False. Si los parámetros a entrenar necesitan inicializarse desde esta capa, es True, el valor predeterminado es False.
    :param init_params: Parámetros de inicialización usados para codificar datos clásicos QTensor, el valor predeterminado es None.
    :param wires: Índice de bit del efecto de línea, el valor predeterminado es None.
    :param dtype: La precisión de datos de la matriz interna de la compuerta lógica puede establecerse en pyvqnet.kcomplex64 o pyvqnet.kcomplex128, correspondiendo a entrada float o double respectivamente.
    :param use_dagger: si usar la versión transpuesta conjugada de la compuerta, el valor predeterminado es False.
    :return: una instancia de ``pyvqnet.qnn.vqc.tn.QModule``

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
    
    define una compuerta cuántica PhaseShift .

    Esta clase hereda de ``pyvqnet.qnn.vqc.tn.QModule`` y ``torch.nn.Module``.
    Esta clase se puede agregar al modelo torch como submódulo de ``torch.nn.Module``.

    :param has_params: si tiene parámetros, compuertas como RX, RY y otras deben establecerse en True, y las que no tienen parámetros deben establecerse en False, el valor predeterminado es False.
    :param trainable: si tiene parámetros para entrenar. Si la capa usa datos de entrada externos para construir la matriz de la compuerta lógica, establezca en False. Si los parámetros a entrenar necesitan inicializarse desde esta capa, es True, el valor predeterminado es False.
    :param init_params: Parámetros de inicialización usados para codificar datos clásicos QTensor, el valor predeterminado es None.
    :param wires: Índice de bit del efecto de línea, el valor predeterminado es None.
    :param dtype: La precisión de datos de la matriz interna de la compuerta lógica puede establecerse en pyvqnet.kcomplex64 o pyvqnet.kcomplex128, correspondiendo a entrada float o double respectivamente.
    :param use_dagger: si usar la versión transpuesta conjugada de la compuerta, el valor predeterminado es False.
    :return: una instancia de ``pyvqnet.qnn.vqc.tn.QModule``

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
    
    define una compuerta cuántica MultiRZ .

    Esta clase hereda de ``pyvqnet.qnn.vqc.tn.QModule`` y ``torch.nn.Module``.
    Esta clase se puede agregar al modelo torch como submódulo de ``torch.nn.Module``.

    :param has_params: si tiene parámetros, compuertas como RX, RY y otras deben establecerse en True, y las que no tienen parámetros deben establecerse en False, el valor predeterminado es False.
    :param trainable: si tiene parámetros para entrenar. Si la capa usa datos de entrada externos para construir la matriz de la compuerta lógica, establezca en False. Si los parámetros a entrenar necesitan inicializarse desde esta capa, es True, el valor predeterminado es False.
    :param init_params: Parámetros de inicialización usados para codificar datos clásicos QTensor, el valor predeterminado es None.
    :param wires: Índice de bit del efecto de línea, el valor predeterminado es None.
    :param dtype: La precisión de datos de la matriz interna de la compuerta lógica puede establecerse en pyvqnet.kcomplex64 o pyvqnet.kcomplex128, correspondiendo a entrada float o double respectivamente.
    :param use_dagger: si usar la versión transpuesta conjugada de la compuerta, el valor predeterminado es False.
    :return: una instancia de ``pyvqnet.qnn.vqc.tn.QModule``

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
    
    define una compuerta cuántica SDG .

    Esta clase hereda de ``pyvqnet.qnn.vqc.tn.QModule`` y ``torch.nn.Module``.
    Esta clase se puede agregar al modelo torch como submódulo de ``torch.nn.Module``.

    :param has_params: si tiene parámetros, compuertas como RX, RY y otras deben establecerse en True, y las que no tienen parámetros deben establecerse en False, el valor predeterminado es False.
    :param trainable: si tiene parámetros para entrenar. Si la capa usa datos de entrada externos para construir la matriz de la compuerta lógica, establezca en False. Si los parámetros a entrenar necesitan inicializarse desde esta capa, es True, el valor predeterminado es False.
    :param init_params: Parámetros de inicialización usados para codificar datos clásicos QTensor, el valor predeterminado es None.
    :param wires: Índice de bit del efecto de línea, el valor predeterminado es None.
    :param dtype: La precisión de datos de la matriz interna de la compuerta lógica puede establecerse en pyvqnet.kcomplex64 o pyvqnet.kcomplex128, correspondiendo a entrada float o double respectivamente.
    :param use_dagger: si usar la versión transpuesta conjugada de la compuerta, el valor predeterminado es False.
    :return: una instancia de ``pyvqnet.qnn.vqc.tn.QModule``

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
    
    define una compuerta cuántica SDG .

    Esta clase hereda de ``pyvqnet.qnn.vqc.tn.QModule`` y ``torch.nn.Module``.
    Esta clase se puede agregar al modelo torch como submódulo de ``torch.nn.Module``.

    :param has_params: si tiene parámetros, compuertas como RX, RY y otras deben establecerse en True, y las que no tienen parámetros deben establecerse en False, el valor predeterminado es False.
    :param trainable: si tiene parámetros para entrenar. Si la capa usa datos de entrada externos para construir la matriz de la compuerta lógica, establezca en False. Si los parámetros a entrenar necesitan inicializarse desde esta capa, es True, el valor predeterminado es False.
    :param init_params: Parámetros de inicialización usados para codificar datos clásicos QTensor, el valor predeterminado es None.
    :param wires: Índice de bit del efecto de línea, el valor predeterminado es None.
    :param dtype: La precisión de datos de la matriz interna de la compuerta lógica puede establecerse en pyvqnet.kcomplex64 o pyvqnet.kcomplex128, correspondiendo a entrada float o double respectivamente.
    :param use_dagger: si usar la versión transpuesta conjugada de la compuerta, el valor predeterminado es False.
    :return: una instancia de ``pyvqnet.qnn.vqc.tn.QModule``

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
    
    define una compuerta cuántica ControlledPhaseShift .

    Esta clase hereda de ``pyvqnet.qnn.vqc.tn.QModule`` y ``torch.nn.Module``.
    Esta clase se puede agregar al modelo torch como submódulo de ``torch.nn.Module``.

    :param has_params: si tiene parámetros, compuertas como RX, RY y otras deben establecerse en True, y las que no tienen parámetros deben establecerse en False, el valor predeterminado es False.
    :param trainable: si tiene parámetros para entrenar. Si la capa usa datos de entrada externos para construir la matriz de la compuerta lógica, establezca en False. Si los parámetros a entrenar necesitan inicializarse desde esta capa, es True, el valor predeterminado es False.
    :param init_params: Parámetros de inicialización usados para codificar datos clásicos QTensor, el valor predeterminado es None.
    :param wires: Índice de bit del efecto de línea, el valor predeterminado es None.
    :param dtype: La precisión de datos de la matriz interna de la compuerta lógica puede establecerse en pyvqnet.kcomplex64 o pyvqnet.kcomplex128, correspondiendo a entrada float o double respectivamente.
    :param use_dagger: si usar la versión transpuesta conjugada de la compuerta, el valor predeterminado es False.
    :return: una instancia de ``pyvqnet.qnn.vqc.tn.QModule``

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

    Calculate the purity on a particular qubit ``qubits_idx`` from the state vector ``state``.

    .. math::
        \gamma = \text{Tr}(\rho^2)

    where :math:`\rho` is a density matrix. The purity of a normalized quantum state satisfies :math:`\frac{1}{d} \leq \gamma \leq 1` ,
    where :math:`d` is the dimension of the Hilbert space.
    The purity of the pure state is 1.

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

    :param q_machine: Estado cu00e1ntico obtenido de pyqpanda get_qstate()
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

    Calcula el resultado de medici00f3n de probabilidad del circuito cu00e1ntico en un bit espec00edfico.

    Esta clase hereda de ``pyvqnet.qnn.vqc.tn.QModule`` y ``torch.nn.Module``.
    Esta clase se puede agregar al modelo torch como submódulo de ``torch.nn.Module``.

    :param wires: The index of the measurement bit, list, tuple or integer.
    :param name: El nombre del m00f3dulo, valor predeterminado: ""
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

    Calcula los resultados de medici00f3n de circuitos cu00e1nticos, soporta entrada obs como m00faltiples o simples operadores Pauli o Hamiltonianos.
    
    For example:

    {\'X0\': 0.23} indicates a PauliX effect on qubit 0, with a coefficient of 0.23.

    {\'X1 Z2\': 2.4,\'Y2\': -0.5} corresponds to the observed value 2.4 * X1 @ Z2 - 0.5 * Y2.

    [{\'X1 Z2\': 4,\'Z1 Z0\': 3},{\'X1 Y2 Z0\': 3.5}] corresponds to the two observed values 4 * X1 @ Z2 + 3 * Z1 @ Z0 and 3.5 * X1 @ Y2 @ Z0.
        
    Esta clase hereda de ``pyvqnet.qnn.vqc.tn.QModule`` y ``torch.nn.Module``.

    Esta clase se puede agregar al modelo torch como submódulo de ``torch.nn.Module``.

    :param obs: observable.
    :param name: nombre del m00f3dulo, valor predeterminado: ""
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

    Get sample results with shot on  specific wires.

    Esta clase hereda de ``pyvqnet.qnn.vqc.tn.QModule`` y ``torch.nn.Module``.
    Esta clase se puede agregar al modelo torch como submódulo de ``torch.nn.Module``.


    :param wires: 00cdndice del qubit de muestreo. Valor predeterminado: None, usa todos los bits del simulador en tiempo de ejecuci00f3n.
    :param obs: Este valor solo puede ser None.
    :param shots: Recuento de repeticiones de muestreo, valor predeterminado: 1.
    :param name: El nombre de este m00f3dulo, valor predeterminado: ""
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

    Calcula la expectativa de una cantidad Hermitiana en un circuito cu00e1ntico.

    Esta clase hereda de ``pyvqnet.qnn.vqc.tn.QModule`` y ``torch.nn.Module``.
    Esta clase se puede agregar al modelo torch como submódulo de ``torch.nn.Module``.


    :param obs: Cantidad Hermitiana.
    :param name: nombre del m00f3dulo, valor predeterminado: ""
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

Plantillas comunes para circuitos cu00e1nticos
----------------------------------------------

VQC_HardwareEfficientAnsatz
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^


.. py:class:: pyvqnet.qnn.vqc.tn.VQC_HardwareEfficientAnsatz(n_qubits,single_rot_gate_list,entangle_gate="CNOT",entangle_rules='linear',depth=1,initial=None,dtype=None)

    Implementation of Hardware Efficient Ansatz introduced in the paper: `Hardware-efficient Variational Quantum Eigensolver for Small Molecules <https://arxiv.org/pdf/1704.05018.pdf>`__ .

    Esta clase hereda de ``pyvqnet.qnn.vqc.tn.QModule`` y ``torch.nn.Module``.

    Esta clase se puede agregar al modelo torch como submódulo de ``torch.nn.Module``.


    :param n_qubits: Number of qubits.
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

    A layer consisting of a single-parameter single-qubit rotation on each qubit, followed by multiple CNOT gates in a closed chain or ring combination.

    A ring of CNOT gates connects each qubit to its neighbors, and finally the a qubit is considered to be the neighbor of the a th qubit.

    Esta clase hereda de ``pyvqnet.qnn.vqc.tn.QModule`` y ``torch.nn.Module``.

    Esta clase se puede agregar al modelo torch como submódulo de ``torch.nn.Module``.

    :param num_layers: number of repeat layers, default: 1.
    :param num_qubits: number of qubits, default: 1.
    :param rotation: one-parameter single-qubit gate to use, default: `RX`
    :param initial: initialized same value for all paramters. default:None,parameters will be initialized randomly.
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

    Layers consisting of single qubit rotations and entanglers, as in `circuit-centric classifier design <https://arxiv.org/abs/1804.00633>`__ .

    Esta clase hereda de ``pyvqnet.qnn.vqc.tn.QModule`` y ``torch.nn.Module``.
    Esta clase se puede agregar al modelo torch como submódulo de ``torch.nn.Module``.


    :param num_layers: number of repeat layers, default: 1.
    :param num_qubits: number of qubits, default: 1.
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
    
    Use RZ,RY,RZ to create variational quantum circuits to encode classical data into quantum states.
    Reference `Quantum embeddings for machine learning <https://arxiv.org/abs/2001.03622>`_.

    Esta clase hereda de ``pyvqnet.qnn.vqc.tn.QModule`` y ``torch.nn.Module``.
    Esta clase se puede agregar al modelo torch como submódulo de ``torch.nn.Module``.

 
    :param num_repetitions_input: number of repeat times to encode input in a submodule.
    :param depth_input: number of input dimension .
    :param num_unitary_layers: number of repeat times of variational quantum gates.
    :param num_repetitions: number of repeat times of submodule.
    :param initial: valor de inicializaci00f3n de par00e1metros, valor predeterminado None
    :param dtype: tipo de par00e1metro, valor predeterminado None, usa float32.
    :param name: nombre de la clase
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

    19 different ansatz from the paper `Expressibility and entangling capability of parameterized quantum circuits for hybrid quantum-classical algorithms <https://arxiv.org/pdf/1905.10876.pdf>`_.

    Esta clase hereda de ``pyvqnet.qnn.vqc.tn.QModule`` y ``torch.nn.Module``.

    Esta clase se puede agregar al modelo torch como submódulo de ``torch.nn.Module``.

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

    Codifica n caracter00edsticas binarias en el estado base de n qubits de ``q_machine``. Esta funci00f3n tiene el alias `VQC_BasisEmbedding`.

    For example, for ``basis_state=([0, 1, 1])``, the basis state in the quantum system is :math:`|011 \rangle`.

    :param basis_state: ``(n)`` size binary input.
    :param q_machine: dispositivo de m00e1quina cu00e1ntica.
    

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

    Codifica :math:`N` características en el ángulo de rotación de :math:`n` qubits, where :math:`N \leq n`.
    Esta función tiene el alias `VQC_AngleEmbedding` .

    La rotación se puede seleccionar como: 'X' , 'Y' , 'Z', as defined by the ``rotation`` parameter:

    * ``rotation='X'`` Usa la caracter00edstica como el 00e1ngulo de rotaci00f3n RX.

    * ``rotation='Y'`` Usa la caracter00edstica como el 00e1ngulo de rotaci00f3n RY.

    * ``rotation='Z'`` Usa la caracter00edstica como el 00e1ngulo de rotaci00f3n RZ.

    ``wires`` representa el 00edndice de la compuerta de rotaci00f3n en el qubit.

    :param input_feat: Arreglo que representa los parámetros.
    :param wires: Índice del qubit.
    :param q_machine: Dispositivo de máquina cuántica.
    :param rotation: Compuerta de rotación, valor predeterminado "X".

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

    Codifica una caracter00edstica de :math:`2^n` en un vector de amplitudes de :math:`n` qubits. Esta funci00f3n tiene el alias `VQC_AmplitudeEmbedding`.

    :param input_feature: arreglo numpy que representa el par00e1metro.
    :param q_machine: dispositivo de m00e1quina cu00e1ntica.
    

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

    Codifica :math:`n` caracter00edsticas en :math:`n` qubits usando compuertas diagonales de un circuito IQP. Alias: ``VQC_IQPEmbedding`` .

    The encoding is proposed by `Havlicek et al. (2018) <https://arxiv.org/pdf/1804.11326.pdf>`_.

    By specifying ``rep`` , the basic IQP circuit can be repeated.

    :param input_feat: Array of parameters.
    :param q_machine: M00e1quina cu00e1ntica.
    :param rep: N00famero de veces para repetir el bloque de circuito cu00e1ntico, valor predeterminado 1.

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

    Combinaci00f3n arbitraria de compuertas l00f3gicas de rotaci00f3n de un solo bit cu00e1ntico. Esta funci00f3n tiene el alias: ``VQC_RotCircuit`` .

    .. math::
        R(\phi,\theta,\omega) = RZ(\omega)RY(\theta)RZ(\phi)= \begin{bmatrix}
        e^{-i(\phi+\omega)/2}\cos(\theta/2) & -e^{i(\phi-\omega)/2}\sin(\theta/2) \\
        e^{-i(\phi-\omega)/2}\sin(\theta/2) & e^{i(\phi+\omega)/2}\cos(\theta/2)
        \end{bmatrix}.

    :param q_machine: dispositivo de m00e1quina virtual cu00e1ntica.
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

    Combinaci00f3n de compuertas l00f3gicas de rotaci00f3n controlada de un solo bit cu00e1ntico. Esta funci00f3n tiene el alias: ``VQC_CRotCircuit`` .

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
    :param q_machine: Dispositivo de máquina cuántica.
    

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

    Circuito cu00e1ntico de compuerta l00f3gica Hadamard controlada. Esta funci00f3n tiene el alias: ``VQC_Controlled_Hadamard`` .

    .. math:: 
        CH = \begin{bmatrix}
        1 & 0 & 0 & 0 \\
        0 & 1 & 0 & 0 \\
        0 & 0 & \frac{1}{\sqrt{2}} & \frac{1}{\sqrt{2}} \\
        0 & 0 & \frac{1}{\sqrt{2}} & -\frac{1}{\sqrt{2}}
        \end{bmatrix}.

    :param wires: lista de 00edndices de bits cu00e1nticos, el primero es el bit de control, la longitud de la lista es 2.
    :param q_machine: dispositivo de m00e1quina virtual cu00e1ntica.

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

    :param wires: lista de 00edndices de bits cu00e1nticos, el primero es el bit de control. La longitud de la lista es 3.
    :param q_machine: dispositivo de m00e1quina virtual cu00e1ntica.

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

    Operador de excitaci00f3n 00fanica de cluster acoplado para producto tensorial de matrices Pauli. La forma matricial est00e1 dada por:

    .. math::
        \hat{U}_{pr}(\theta) = \mathrm{exp} \{ \theta_{pr} (\hat{c}_p^\dagger \hat{c}_r
        -\mathrm{H.c.}) \},

    Alias: ``VQC_FermionicSingleExcitation`` .

    :param weight: Par00e1metro en el qubit p, solo un elemento.
    :param wires: Un subconjunto de 00edndices de qubits en el intervalo [r, p]. La longitud m00ednima debe ser 2. El primer valor de 00edndice se interpreta como r, y el 00faltimo valor de 00edndice se interpreta como p. Los 00edndices intermedios son afectados por compuertas CNOT para calcular la paridad del conjunto de qubits.
    :param q_machine: Dispositivo de m00e1quina virtual cu00e1ntica.

    

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

    Operador de bi-excitaci00f3n de cluster acoplado para producto tensorial de matrices Pauli exponenciado, forma matricial dada por:

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

    Esta funci00f3n tiene el alias: ``VQC_FermionicDoubleExcitation`` .

    :param weight: par00e1metro variable
    :param wires1: representa el subconjunto de qubits en el intervalo de lista de 00edndices [s, r]. El 00edndice a-th se interpreta como s y el 00faltimo 00edndice se interpreta como r. La compuerta CNOT opera en los 00edndices intermedios para calcular la paridad de un grupo de qubits.
    :param wires2: representa el subconjunto de qubits en el intervalo de lista de 00edndices [q, p]. El primer 00edndice ra00edz se interpreta como q y el 00faltimo 00edndice se interpreta como p. La compuerta CNOT opera en los 00edndices intermedios para calcular la paridad de un grupo de qubits.
    :param q_machine: Dispositivo de m00e1quina virtual cu00e1ntica.

    

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

    Implementa la Simulaci00f3n de Excitaciones 00danicas y Dobles del Cluster Acoplado Unitario (UCCSD). UCCSD es una simulaci00f3n VQE com00fanmente usada para ejecutar simulaciones de qu00edmica cu00e1ntica.

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

    Esta funci00f3n tiene el alias: ``VQC_UCCSD`` .

    :param weights: tensor de tama00f1o ``(len(s_wires)+ len(d_wires))`` que contiene los par00e1metros :math:`\theta_{pr}` and :math:`\theta_{pqrs}` input Z rotations ``FermionicSingleExcitation`` and ``FermionicDoubleExcitation`` .
    :param wires: 00edndices de qubits para la acci00f3n de la plantilla
    :param s_wires: secuencia de listas que contienen 00edndices de qubits ``[r,...,p]`` generados por una excitaci00f3n simple :math:`\vert r, p \rangle = \hat{c}_p^\dagger \hat{c}_r \vert \mathrm{HF} \rangle`,where :math:`\vert \mathrm{HF} \rangle` denotes the Hartee-Fock reference state.
    :param d_wires: secuencia de listas, cada una conteniendo dos listas que especifican 00edndices ``[s,...,r]`` y ``[q,...,p]`` definiendo doble excitaci00f3n :math:`\vert s, r, q, p \rangle = \hat{c}_p^\dagger \hat{c}_q^\dagger \hat{c}_r\hat{c}_s \vert \mathrm{HF} \rangle` .
    :param init_state: vector de n00fameros de ocupaci00f3n de longitud ``len(wires)`` que representa el estado de alta frecuencia. ``init_state`` Estado de inicializaci00f3n del qubit.
    :param q_machine: Dispositivo de m00e1quina virtual cu00e1ntica.

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

    Circuito de evoluci00f3n Pauli Z de primer orden.

    For 3 qubits and 2 repetitions, the circuit is represented as:

    .. parsed-literal::

        ┌───┐┌──────────────┐┌───┐┌──────────────┐
        ┤ H ├┤ U1(2.0*x[0]) ├┤ H ├┤ U1(2.0*x[0]) ├
        ├───┤├──────────────┤├───┤├──────────────┤
        ┤ H ├┤ U1(2.0*x[1]) ├┤ H ├┤ U1(2.0*x[1]) ├
        ├───┤├──────────────┤├───┤├──────────────┤
        ┤ H ├┤ U1(2.0*x[2]) ├┤ H ├┤ U1(2.0*x[2]) ├
        └───┘└──────────────┘└───┘└──────────────┘

    La cadena Pauli est00e1 fijada a ``Z``. Por lo tanto, la expansi00f3n de primer orden ser00e1 un circuito sin compuertas de entrelazamiento.

    :param input_feat: Arreglo que representa los parámetros de entrada.
    :param q_machine: M00e1quina virtual cu00e1ntica.
    :param data_map_func: Matriz de mapeo de par00e1metros, una funci00f3n invocable, dise00f1ada como: ``data_map_func = lambda x: x``.
    :param rep: N00famero de veces que se repite el m00f3dulo.

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

    Circuito de evoluci00f3n Pauli-Z de segundo orden.

    For 3 qubits, 1 repeat, and linear entanglement, the circuit is represented as:

    .. parsed-literal::


        ┌───┐┌─────────────────┐
        ┤ H ├┤ U1(2.0*φ(x[0])) ├──■────────────────────────────■────────────────────────────────────
        ├───┤├─────────────────┤┌─┴─┐┌──────────────────────┐┌─┴─┐
        ┤ H ├┤ U1(2.0*φ(x[1])) ├┤ X ├┤ U1(2.0*φ(x[0],x[1])) ├┤ X ├──■────────────────────────────■──
        ├───┤├─────────────────┤└───┘└──────────────────────┘└───┘┌─┴─┐┌──────────────────────┐┌─┴─┐
        ┤ H ├┤ U1(2.0*φ(x[2])) ├──────────────────────────────────┤ X ├┤ U1(2.0*φ(x[1],x[2])) ├┤ X ├
        └───┘└─────────────────┘                                  └───┘└──────────────────────┘└───┘
    
    Donde ``φ`` es una funci00f3n no lineal cl00e1sica. Si se ingresan dos valores, ``φ(x,y) = (pi - x)(pi - y)``, y si se ingresa uno, ``φ(x) = x``. Se expresa de la siguiente manera usando ``data_map_func``:

    .. code-block::

        def data_map_func(x):
            coeff = x if x.shape[-1] == 1 else ft.reduce(lambda x, y: (np.pi - x) * (np.pi - y), x)
            return coeff

    :param input_feat: Arreglo que representa los parámetros de entrada.
    :param q_machine: M00e1quina virtual cu00e1ntica.
    :param data_map_func: matriz de mapeo de par00e1metros, una funci00f3n invocable.
    :param entanglement: estructura de entrelazamiento especificada.
    :param rep: veces de repetici00f3n del m00f3dulo.
    
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

    En este caso, tenemos cuatro excitaciones simples y dobles para preservar la proyecci00f3n de esp00edn total del estado de Hartree-Fock.

    La matriz unitaria resultante preserva la poblaci00f3n de part00edculas y prepara el sistema de n-qubits en una superposici00f3n del estado inicial de Hartree-Fock y otros estados que codifican la configuraci00f3n de m00faltiples excitaciones.

    :param weights: Un QTensor de tama00f1o ``(len(singles) + len(doubles),)`` que contiene los 00e1ngulos que entran en las operaciones vqc.qCircuit.single_excitation y vqc.qCircuit.double_excitation en secuencia
    :param q_machine: La m00e1quina cu00e1ntica.
    :param hf_state: Un vector de longitud ``len(wires)`` de n00fameros de ocupaci00f3n que representa el estado de Hartree-Fock, ``hf_state`` usado para inicializar los wires.
    :param wires: Los qubits sobre los que actuar.
    :param singles: Una secuencia de listas con los 00edndices de los dos qubits sobre los que act00faa la operaci00f3n single_excitation.
    :param doubles: Secuencia de listas con los 00edndices de los dos qubits sobre los que act00faa la operaci00f3n double_excitation.

    Por ejemplo, el circuito cu00e1ntico para dos electrones y seis qubits se muestra a continuaci00f3n:

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

    Implementa un circuito que proporciona un conjunto que puede usarse para realizar rotaciones precisas de base unitaria simple. The circuit is derived from the single-particle fermion-determined unitary transformation :math:`U(u)` given in `arXiv:1711.04789 <https://arxiv.org/abs/1711.04789>`_\ 
    
    .. math::
        U(u) = \exp{\left( \sum_{pq} \left[\log u \right]_{pq} (a_p^\dagger a_q - a_q^\dagger a_p) \right)}.

    :math:`U(u)` is obtained by using the scheme given in the paper `Optica, 3, 1460 (2016) <https://opg.optica.org/optica/fulltext.cfm?uri=optica-3-12-1460&id=355743>`_\ .

    :param q_machine: m00e1quina cu00e1ntica.
    :param wires: qubits sobre los que actuar.
    :param unitary_matrix: matriz que especifica la base para la transformaci00f3n.
    :param check: verificar si `unitary_matrix` es una matriz unitaria.

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
    Esta clase llamar00e1 a backend, rank, world_size para inicializar ``torch.distributed.init_process_group(backend, rank, world_size)`` .

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

