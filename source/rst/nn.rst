Módulo de Red Neuronal Clásica
#########################################

Los siguientes módulos de red neuronal clásica soportan el cálculo automático de propagación hacia atrás. Después de ejecutar la función forward, puede calcular el gradiente ejecutando la función backward. Un ejemplo simple de la capa de convolución es el siguiente:

.. code-block::

    from pyvqnet.tensor import arange
    from pyvqnet import kfloat32
    from pyvqnet.nn import Conv2D

    # una imagen se introduce en una capa de convolución bidimensional
    b = 2        # tamaño del lote
    ic = 2       # canales de entrada
    oc = 2      # canales de salida
    hw = 4      # altura y anchura de entrada

    # capa de convolución bidimensional
    test_conv = Conv2D(ic,oc,(2,2),(2,2),"same")

    # entrada de forma [b,ic,hw,hw]
    x0 = arange(1,b*ic*hw*hw+1,requires_grad=True,dtype=kfloat32)

    x1 = x0.reshape([b,ic,hw,hw])
    #función forward
    x = test_conv(x1)

    #función backward con autograd
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


Clase Módulo
********************************************************

módulo de cálculo abstracto


Module
=================================

.. py:class:: pyvqnet.nn.module.Module

    Clase base para todos los módulos de red neuronal, incluyendo módulos cuánticos o módulos clásicos.
    Sus modelos también deben ser subclases de esta clase para el cálculo con autograd.

    Los módulos también pueden contener otros módulos, permitiendo anidarlos en
    una estructura de árbol. Puede asignar los submódulos como atributos regulares::

        class Model(Module):
            def __init__(self):
                super(Model, self).__init__()
                self.conv1 = pyvqnet.nn.Conv2d(1, 20, (5,5))
                self.conv2 = pyvqnet.nn.Conv2d(20, 20, (5,5))

            def forward(self, x):
                x = pyvqnet.nn.activation.relu(self.conv1(x))
                return pyvqnet.nn.activation.relu(self.conv2(x))

    Los submódulos asignados de esta manera serán registrados

forward
=================================

.. py:method:: pyvqnet.nn.module.Module.forward(x, *args, **kwargs)

    Método abstracto que realiza el paso forward (propagación hacia adelante).

    :param x: QTensor de entrada
    :param \*args: Un parámetro variable no nombrado (keyword)
    :param \*\*kwargs: Un parámetro variable nombrado (keyword)
    :return: salida del módulo

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

    Retorna un diccionario que contiene el estado completo del módulo.

    Se incluyen tanto los parámetros como los búferes persistentes (por ejemplo, promedios móviles).
    Las claves son los nombres correspondientes de parámetros y búferes.

    :param destination: un diccionario donde se almacenará el estado
    :param prefix: el prefijo para los parámetros y búferes usados en este
        módulo

    :return: un diccionario que contiene el estado completo del módulo

    Example::

        from pyvqnet.nn import Conv2D
        test_conv = Conv2D(2,3,(3,3),(2,2),"same")
        print(test_conv.state_dict().keys())
        #odict_keys(['weights', 'bias'])


toGPU
=================================

.. py:function:: pyvqnet.nn.module.Module.toGPU(device: int = DEV_GPU_0)

    Mueve los parámetros y datos de búfer de un módulo y sus submódulos al dispositivo GPU especificado.

    device especifica el dispositivo donde se almacenan los datos internos. Cuando device >= DEV_GPU_0, los datos se almacenan en la GPU. Si su computadora tiene múltiples GPUs,
    puede especificar diferentes dispositivos para almacenar datos. Por ejemplo, device = DEV_GPU_1, DEV_GPU_2, DEV_GPU_3, ... significa que se almacena en GPUs con diferentes números de serie.
    
    .. note::
        El módulo no se puede calcular en diferentes GPUs. Se generará un error Cuda si intenta crear un QTensor en una GPU cuyo ID excede el número máximo de GPUs verificadas.

    :param device: El dispositivo que actualmente guarda QTensor, por defecto=DEV_GPU_0. device = pyvqnet.DEV_GPU_0, almacenado en la primera GPU, device = DEV_GPU_1, almacenado en la segunda GPU, y así sucesivamente.
    :return: Módulo movido al dispositivo GPU.

    Examples::

        from pyvqnet.nn.conv import ConvT2D 
        test_conv = ConvT2D(3, 2, [4,4], [2, 2], "same")
        test_conv = test_conv.toGPU()
        print(test_conv.backend)
        #1000


toCPU
=================================

.. py:function:: pyvqnet.nn.module.Module.toCPU()

    Mueve los parámetros y datos de búfer de un módulo y sus submódulos a un dispositivo CPU específico.

    :return: Módulo movido al dispositivo CPU.

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

    Guarda los parámetros del modelo en un archivo de disco.

    :param obj: OrderedDict guardado desde ``state_dict()``
    :param f: una cadena o un objeto os.PathLike que contiene un nombre de archivo
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

    Carga los parámetros del modelo desde un archivo de disco.

    La instancia del modelo debe crearse primero.

    :param f: una cadena o un objeto os.PathLike que contiene un nombre de archivo
    :return: OrderedDict guardado para ``load_state_dict()``

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


    Guarda submódulos en una lista. ModuleList se puede indexar como una lista normal de Python, y los parámetros internos del módulo que contiene se pueden guardar.

    :param modules: lista de nn.Module

    :return: una lista de módulos

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


    Almacena parámetros en una lista. ParameterList se puede indexar como una lista normal de Python, y los parámetros internos del Parameter que contiene se pueden almacenar.

    :param modules: lista de nn.Parameter.

    :return: una lista de Parameter.

    Example::

        from pyvqnet import nn
        class MyModule(nn.Module):
            def __init__(self):
                super().__init__()
                self.params = nn.ParameterList([nn.Parameter((10, 10)) for i in range(10)])
            def forward(self, x):

                # ParameterList can act as an iterable, or be indexed using ints
                for i, p in enumerate(self.params):
                    x = self.params[i // 2] * x + p * x
                return x

        model = MyModule()
        print(model.state_dict().keys())


Sequential
*********************************************************
.. py:class:: pyvqnet.nn.module.Sequential([pyvqnet.nn.module.Module])

    Los módulos se agregarán en el orden en que se pasen. Alternativamente, se puede pasar un ``OrderedDict`` de módulos. El método ``forward()`` de ``Sequential`` toma cualquier entrada y la envía a su primer módulo.
    Luego ``Sequential`` pasa la salida como entrada a cada módulo subsiguiente en orden, y finalmente retorna la salida del último módulo.

    :param modules: módulo a agregar.

    :return: Sequential.

    Example::
        
        from pyvqnet import nn
        from collections import OrderedDict

        # Usando Sequential para crear un modelo pequeño.
        model = nn.Sequential(
                  nn.Conv2D(1,20,(5, 5)),
                  nn.ReLu(),
                  nn.Conv2D(20,64,(5, 5)),
                  nn.ReLu()
                )
        print(model.state_dict().keys())

        # Usando Sequential con OrderedDict. Esto es funcionalmente igual al código anterior
                
        model = nn.Sequential(OrderedDict([
                  ('conv1', nn.Conv2D(1,20,(5, 5))),
                  ('relu1', nn.ReLu()),
                  ('conv2', nn.Conv2D(20,64,(5, 5))),
                  ('relu2', nn.ReLu())
                ]))
        print(model.state_dict().keys())


Capa de Red Neuronal Clásica
********************************************************

Conv1D
=================================

.. py:class:: pyvqnet.nn.Conv1D(input_channels:int,output_channels:int,kernel_size:int ,stride:int= 1,padding = "valid",use_bias:str = True,kernel_initializer = None,bias_initializer =None, dilation_rate: int = 1, group: int = 1, dtype=None, name='')

    Aplica un kernel de convolución unidimensional sobre una entrada. Las entradas al módulo conv tienen forma (batch_size, input_channels, height)

    :param input_channels: `int` - Número de canales de entrada
    :param output_channels: `int` - Número de kernels
    :param kernel_size: `int` - Tamaño de un solo kernel. forma del kernel = [output_channels, input_channels/group, kernel_size, 1]
    :param stride: `int` - Paso (stride), por defecto 1
    :param padding: `str|int` - opción de relleno, que puede ser una cadena {'valid', 'same'} o un entero que indica la cantidad de relleno implícito a aplicar. Por defecto "valid".
    :param use_bias: `bool` - si usa sesgo (bias), por defecto True
    :param kernel_initializer: `callable` - Por defecto None
    :param bias_initializer: `callable` - Por defecto None
    :param dilation_rate: `int` - tamaño de dilatación, por defecto: 1
    :param group: `int` - número de grupos de convoluciones agrupadas. Por defecto: 1
    :param dtype: El tipo de dato del parámetro, por defecto: None, usa el tipo de dato predeterminado kfloat32, que representa un número de punto flotante de 32 bits.
    :param name: El nombre del módulo, por defecto: "".
    :return: una clase Conv1D

    .. note::
        ``padding='valid'`` es equivalente a sin relleno.

        ``padding='same'`` rellena la entrada para que la salida tenga la misma forma que la entrada.

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

    Aplica un kernel de convolución bidimensional sobre una entrada. Las entradas al módulo conv tienen forma (batch_size, input_channels, height, width)

    :param input_channels: `int` - Número de canales de entrada
    :param output_channels: `int` - Número de kernels
    :param kernel_size: `tuple|list` - Tamaño de un solo kernel. forma del kernel = [output_channels, input_channels/group, kernel_size, kernel_size]
    :param stride: `tuple|list` - Paso (stride), por defecto (1, 1)|[1,1]
    :param padding: `str|tuple` - opción de relleno, que puede ser una cadena {'valid', 'same'} o una tupla de enteros que indica la cantidad de relleno implícito a aplicar en ambos lados. Por defecto "valid".
    :param use_bias: `bool` - si usa sesgo (bias), por defecto True
    :param kernel_initializer: `callable` - Por defecto None
    :param bias_initializer: `callable` - Por defecto None
    :param dilation_rate: `int` - tamaño de dilatación, por defecto: 1
    :param group: `int` - número de grupos de convoluciones agrupadas. Por defecto: 1.
    :param dtype: El tipo de dato del parámetro, por defecto: None, usa el tipo de dato predeterminado kfloat32, que representa un número de punto flotante de 32 bits.
    :param name: El nombre del módulo, por defecto: "".

    :return: una clase Conv2D

    .. note::
        ``padding='valid'`` es equivalente a sin relleno.

        ``padding='same'`` rellena la entrada para que la salida tenga la misma forma que la entrada.

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

    Aplica un kernel de convolución transpuesta bidimensional sobre una entrada. Las entradas al módulo convT tienen forma (batch_size, input_channels, height, width)

    :param input_channels: `int` - Número de canales de entrada
    :param output_channels: `int` - Número de kernels
    :param kernel_size: `tuple|list` - Tamaño de un solo kernel. forma del kernel = [input_channels, output_channels/group, kernel_size, kernel_size]
    :param stride: `tuple|list` - Paso (stride), por defecto (1, 1)|[1,1]
    :param padding: `str|tuple` - opción de relleno, que puede ser una cadena {'valid', 'same'} o una tupla de enteros que indica la cantidad de relleno implícito a aplicar en ambos lados. Por defecto "valid".
    :param use_bias: `bool` - Si usar un elemento de desplazamiento (offset). Por defecto True
    :param kernel_initializer: `callable` - Por defecto None
    :param bias_initializer: `callable` - Por defecto None
    :param dilation_rate: `int` - tamaño de dilatación, por defecto: 1.
    :param out_padding: Tamaño adicional añadido a un lado de cada dimensión en la forma de salida. Por defecto: (0,0)
    :param group: `int` - número de grupos de convoluciones agrupadas. Por defecto: 1.
    :param dtype: El tipo de dato del parámetro, por defecto: None, usa el tipo de dato predeterminado kfloat32, que representa un número de punto flotante de 32 bits.
    :param name: El nombre del módulo, por defecto: "".

    :return: una clase ConvT2D

    .. note::
        ``padding='valid'`` es equivalente a sin relleno.

        ``padding='same'`` rellena la entrada para que la salida tenga la misma forma que la entrada.


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

    Esta operación aplica un promedio pooling 1D sobre una señal de entrada compuesta por varios planos de entrada.

    :param kernel: tamaño de las ventanas de promedio pooling
    :param strides: factor de reducción de escala
    :param padding: uno de "valid", "same" o un entero que especifica el valor de relleno, por defecto "valid"
    :param name: nombre de la capa de salida.

    :return: capa AvgPool1D

    .. note::
        ``padding='valid'`` es equivalente a sin relleno.

        ``padding='same'`` rellena la entrada para que la salida tenga la misma forma que la entrada.



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

    Esta operación aplica un max pooling 1D sobre una señal de entrada compuesta por varios planos de entrada.

    :param kernel: tamaño de las ventanas de max pooling
    :param strides: factor de reducción de escala
    :param padding: uno de "valid", "same" o un entero que especifica el valor de relleno, por defecto "valid"
    :param name: El nombre del módulo, por defecto: "".

    :return: capa MaxPool1D

    .. note::

        ``padding='valid'`` es equivalente a sin relleno.

        ``padding='same'`` rellena la entrada para que la salida tenga la misma forma que la entrada.


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

    Esta operación aplica un promedio pooling 2D sobre las características de entrada.

    :param kernel: tamaño de las ventanas de promedio pooling
    :param strides: factores de reducción de escala
    :param padding: uno de "valid", "same" o una tupla con enteros que especifica el valor de relleno de columna y fila, por defecto "valid"
    :param name: nombre de la capa de salida
    :return: capa AvgPool2D

    .. note::
        ``padding='valid'`` es equivalente a sin relleno.

        ``padding='same'`` rellena la entrada para que la salida tenga la misma forma que la entrada.


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

    Esta operación aplica un max pooling 2D sobre las características de entrada.

    :param kernel: tamaño de las ventanas de max pooling
    :param strides: factor de reducción de escala
    :param padding: uno de "valid", "same" o una tupla con enteros que especifica el valor de relleno de columna y fila, por defecto "valid"
    :param name: nombre de la capa de salida
    :return: capa MaxPool2D

    .. note::
        ``padding='valid'`` es equivalente a sin relleno.

        ``padding='same'`` rellena la entrada para que la salida tenga la misma forma que la entrada.


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

    Este módulo se usa a menudo para almacenar embeddings de palabras y recuperarlos usando índices.
    La entrada al módulo es una lista de índices, y la salida son los
    embeddings de palabras correspondientes.

    :param num_embeddings: `int` - tamaño del diccionario de embeddings.
    :param embedding_dim: `int` - el tamaño de cada vector de embedding.
    :param weight_initializer: `callable` - por defecto normal.
    :param dtype: El tipo de dato del parámetro, por defecto: None, usa el tipo de dato predeterminado kfloat32, que representa un número de punto flotante de 32 bits.
    :param name: nombre de la capa de salida.

    :return: una clase Embedding

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

    Aplica normalización por lotes (Batch Normalization) sobre una entrada 4D (B,C,H,W) como se describe en el artículo
    `Batch Normalization: Accelerating Deep Network Training by Reducing
    Internal Covariate Shift <https://arxiv.org/abs/1502.03167>`__ .

    .. math::

        y = \frac{x - \mathrm{E}[x]}{\sqrt{\mathrm{Var}[x] + \epsilon}} * \gamma + \beta

    donde :math:`\gamma` y :math:`\beta` son parámetros aprendibles. También por defecto, durante el entrenamiento esta capa mantiene estimaciones
    actualizadas de su media y varianza calculadas, que luego se usan para la normalización durante la evaluación.
    Las estimaciones actualizadas se mantienen con un momentum por defecto de 0.1.

    :param channel_num: `int` - el número de canales de características de entrada.
    :param momentum: `float` - momentum al calcular el promedio ponderado exponencial, por defecto 0.1.
    :param epsilon: `float` - constante de estabilidad numérica, por defecto 1e-5.
    :param affine: Un valor booleano que, cuando se establece en ``True``, hace que este módulo tenga parámetros afines aprendibles por canal, inicializados en 1 (para pesos) y 0 (para sesgos). Por defecto: ``True``.
    :param beta_initializer: `callable` - por defecto zeros.
    :param gamma_initializer: `callable` - por defecto ones.
    :param dtype: El tipo de dato del parámetro, por defecto: None, usa el tipo de dato predeterminado kfloat32, que representa un número de punto flotante de 32 bits.
    :param name: nombre de la capa de salida
    :return: una clase BatchNorm2d

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

    Aplica normalización por lotes (Batch Normalization) sobre una entrada 2D (B,C) como se describe en el artículo
    `Batch Normalization: Accelerating Deep Network Training by Reducing
    Internal Covariate Shift <https://arxiv.org/abs/1502.03167>`__ .

    .. math::

        y = \frac{x - \mathrm{E}[x]}{\sqrt{\mathrm{Var}[x] + \epsilon}} * \gamma + \beta

    donde :math:`\gamma` y :math:`\beta` son parámetros aprendibles. También por defecto, durante el entrenamiento esta capa mantiene estimaciones
    actualizadas de su media y varianza calculadas, que luego se usan para la normalización durante la evaluación.
    Las estimaciones actualizadas se mantienen con un momentum por defecto de 0.1.


    :param channel_num: `int` - el número de canales de características de entrada.
    :param momentum: `float` - momentum al calcular el promedio ponderado exponencial, por defecto 0.1
    :param epsilon: `float` - constante de estabilidad numérica, por defecto 1e-5.
    :param affine: Un valor booleano que, cuando se establece en ``True``, hace que este módulo tenga parámetros afines aprendibles por canal, inicializados en 1 (para pesos) y 0 (para sesgos). Por defecto: ``True``.
    :param beta_initializer: `callable` - por defecto zeros.
    :param gamma_initializer: `callable` - por defecto ones.
    :param dtype: El tipo de dato del parámetro, por defecto: None, usa el tipo de dato predeterminado kfloat32, que representa un número de punto flotante de 32 bits.
    :param name: nombre de la capa de salida
    :return: una clase BatchNorm1d

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

    La normalización de capa (Layer Normalization) se realiza en las últimas dimensiones de cualquier entrada. El método específico es como se describe en el artículo:
    `Layer Normalization <https://arxiv.org/abs/1607.06450>`__.

    .. math::
        y = \frac{x - \mathrm{E}[x]}{ \sqrt{\mathrm{Var}[x] + \epsilon}} * \gamma + \beta

    Para entradas como (B,C,H,W,D), ``norm_shape`` puede ser [C,H,W,D], [H,W,D], [W,D] o [D].

    :param norm_shape: `float` - estandarizar la forma.
    :param epsilon: `float` - constante de estabilidad numérica, por defecto 1e-5.
    :param affine: Un valor booleano que, cuando se establece en ``True``, hace que este módulo tenga parámetros afines aprendibles por canal, inicializados en 1 (para pesos) y 0 (para sesgos). Por defecto: ``True``.
    :param dtype: El tipo de dato del parámetro, por defecto: None, usa el tipo de dato predeterminado kfloat32, que representa un número de punto flotante de 32 bits.
    :param name: nombre de la capa de salida.

    :return: una clase LayerNormNd.

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

    Aplica normalización de capa (Layer Normalization) sobre un minilote de entradas 4D como se describe en
    el artículo `Layer Normalization <https://arxiv.org/abs/1607.06450>`__

    .. math::
        y = \frac{x - \mathrm{E}[x]}{ \sqrt{\mathrm{Var}[x] + \epsilon}} * \gamma + \beta

    La media y la desviación estándar se calculan sobre el tamaño de las últimas `D` dimensiones.

    Para entrada como (B,C,H,W), ``norm_size`` debe ser igual a C * H * W.

    :param norm_size: `float` - tamaño de normalización, igual a C * H * W
    :param epsilon: `float` - constante de estabilidad numérica, por defecto 1e-5
    :param affine: Un valor booleano que, cuando se establece en ``True``, hace que este módulo tenga parámetros afines aprendibles por canal, inicializados en 1 (para pesos) y 0 (para sesgos). Por defecto: ``True``.
    :param name: nombre de la capa de salida
    :param dtype: El tipo de dato del parámetro, por defecto: None, usa el tipo de dato predeterminado kfloat32, que representa un número de punto flotante de 32 bits.
    
    :return: una clase LayerNorm2d

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

    Aplica normalización de capa (Layer Normalization) sobre un minilote de entradas 2D como se describe en
    el artículo `Layer Normalization <https://arxiv.org/abs/1607.06450>`__

    .. math::
        y = \frac{x - \mathrm{E}[x]}{ \sqrt{\mathrm{Var}[x] + \epsilon}} * \gamma + \beta

    La media y la desviación estándar se calculan sobre el tamaño de la última dimensión, donde ``norm_size``
    es el valor del tamaño de la última dimensión.

    :param norm_size: `float` - tamaño de normalización, igual a la última dimensión
    :param epsilon: `float` - constante de estabilidad numérica, por defecto 1e-5
    :param affine: Un valor booleano que, cuando se establece en ``True``, hace que este módulo tenga parámetros afines aprendibles por canal, inicializados en 1 (para pesos) y 0 (para sesgos). Por defecto: ``True``.
    :param dtype: El tipo de dato del parámetro, por defecto: None, usa el tipo de dato predeterminado kfloat32, que representa un número de punto flotante de 32 bits.
    :param name: nombre de la capa de salida

    :return: una clase LayerNorm1d

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

    Aplica normalización por grupos (Group Normalization) a un minilote de entradas. Entrada: :math:`(N, C, *)` donde :math:`C=\mathrm{num\_channels}`, Salida: :math:`(N, C, *)`.

    Esta capa implementa la operación descrita en el artículo `Group Normalization <https://arxiv.org/abs/1803.08494>`__

    .. math::

        y = \frac{x - \mathrm{E}[x]}{ \sqrt{\mathrm{Var}[x] + \epsilon}} * \gamma + \beta

    Los canales de entrada se dividen en :attr:`num_groups` grupos, cada uno conteniendo ``num_channels / num_groups`` canales. :attr:`num_channels` debe ser divisible por :attr:`num_groups`. La media y la desviación estándar se calculan por separado para cada grupo. Si :attr:`affine` es ``True``, entonces :math:`\gamma` y :math:`\beta` son aprendibles. Vector de parámetros de transformación afín por canal de tamaño :attr:`num_channels`.

    :param num_groups (int): Número de grupos en los que dividir los canales
    :param num_channels (int): Número de canales esperados en la entrada
    :param eps: Valor a agregar al denominador para estabilidad numérica. Por defecto: 1e-5
    :param affine: Un valor booleano que, cuando se establece en ``True``, hace que este módulo tenga parámetros afines aprendibles por canal, inicializados en 1 (para pesos) y 0 (para sesgos). Por defecto: ``True``.
    :param dtype: El tipo de dato del parámetro, por defecto: None, usa el tipo de dato predeterminado kfloat32, que representa un número de punto flotante de 32 bits.
    :param name: nombre de la capa de salida

    :return: clase GroupNorm

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

    Módulo lineal (capa totalmente conectada).
    :math:`y = x@A.T + b`

    :param input_channels: `int` - número de características de entrada
    :param output_channels: `int` - número de características de salida
    :param weight_initializer: `callable` - por defecto normal
    :param bias_initializer: `callable` - por defecto zeros
    :param use_bias: `bool` - por defecto True
    :param dtype: El tipo de dato del parámetro, por defecto: None, usa el tipo de dato predeterminado kfloat32, que representa un número de punto flotante de 32 bits.
    :param name: nombre de la capa de salida

    :return: una clase Linear

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

    Módulo Dropout. El módulo dropout establece aleatoriamente las salidas de algunas unidades a cero, mientras escala otras según la probabilidad de dropout dada.

    :param dropout_rate: `float` - probabilidad de que una neurona se establezca a cero
    :return: una clase Dropout

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

    El módulo DropPath descarta caminos (aleatoriamente profundos) muestra por muestra.

    :param dropout_rate: `float` - La probabilidad de que la neurona se establezca a cero.
    :param name: El nombre de este módulo, por defecto es "".

    :return: instancia de DropPath.

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

    Reorganiza tensores de forma: (*, C * r^2, H, W) a un tensor de forma (*, C, H * r, W * r) donde r es el factor de escala.

    :param upscale_factors: factor para aumentar la transformación de escala

    :return:
            módulo Pixel_Shuffle

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

    Invierte la operación Pixel_Shuffle reorganizando los elementos. Reordena un tensor de forma (*, C, H * r, W * r) a (*, C * r^2, H, W), donde r es el factor de reducción.
    
    :param downscale_factors: factor para aumentar la transformación de escala

    :return:
            módulo Pixel_Unshuffle

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


    Módulo de Unidad Recurrente Cerrada (GRU). Soporta apilamiento multicapa y configuración bidireccional.
    La fórmula de cálculo del GRU unidireccional de una sola capa es la siguiente:

    .. math::
        \begin{array}{ll}
            r_t = \sigma(W_{ir} x_t + b_{ir} + W_{hr} h_{(t-1)} + b_{hr}) \\
            z_t = \sigma(W_{iz} x_t + b_{iz} + W_{hz} h_{(t-1)} + b_{hz}) \\
            n_t = \tanh(W_{in} x_t + b_{in} + r_t * (W_{hn} h_{(t-1)}+ b_{hn})) \\
            h_t = (1 - z_t) * n_t + z_t * h_{(t-1)}
        \end{array}

    :param input_size: Dimensiones de las características de entrada.
    :param hidden_size: Dimensiones de las características ocultas.
    :param num_layers: Número de capas apiladas. por defecto: 1.
    :param batch_first: Si batch_first es True, la forma de entrada debe ser [batch_size, seq_len, feature_dim],
     si batch_first es False, la forma de entrada debe ser [seq_len, batch_size, feature_dim], por defecto: True.
    :param use_bias: Si use_bias es False, este módulo no contendrá sesgo. por defecto: True.
    :param bidirectional: Si bidirectional es True, el módulo será un GRU bidireccional. por defecto: False.
    :param dtype: El tipo de dato del parámetro, por defecto: None, usa el tipo de dato predeterminado kfloat32, que representa un número de punto flotante de 32 bits.
    :param name: nombre de la capa de salida

    :return: Una instancia del módulo GRU.

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


    Módulo de Red Neuronal Recurrente (RNN), usa :math:`\tanh` o :math:`\text{ReLU}` como función de activación.
    Se soporta RNN bidireccional y RNN multicapa.
    La fórmula de cálculo del RNN unidireccional de una sola capa es la siguiente:

    .. math::
        h_t = \tanh(W_{ih} x_t + b_{ih} + W_{hh} h_{(t-1)} + b_{hh})

    Si :attr:`nonlinearity` es ``'relu'``, entonces :math:`\text{ReLU}` reemplazará a :math:`\tanh`.

    :param input_size: Dimensiones de las características de entrada.
    :param hidden_size: Dimensiones de las características ocultas.
    :param num_layers: Número de capas apiladas. por defecto: 1.
    :param nonlinearity: función de activación no lineal, por defecto: ``'tanh'``.
    :param batch_first: Si batch_first es True, la forma de entrada debe ser [batch_size, seq_len, feature_dim],
     si batch_first es False, la forma de entrada debe ser [seq_len, batch_size, feature_dim], por defecto: True.
    :param use_bias: Si use_bias es False, este módulo no contendrá sesgo. por defecto: True.
    :param bidirectional: Si bidirectional es True, el módulo será un RNN bidireccional. por defecto: False.
    :param dtype: El tipo de dato del parámetro, por defecto: None, usa el tipo de dato predeterminado kfloat32, que representa un número de punto flotante de 32 bits.
    :param name: nombre de la capa de salida

    :return: Una instancia del módulo RNN.

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

    Módulo de Memoria a Largo Plazo (LSTM). Soporta LSTM bidireccional, LSTM multicapa apilada y otras configuraciones.
    La fórmula de cálculo del LSTM unidireccional de una sola capa es la siguiente:

    .. math::
        \begin{array}{ll} \\
            i_t = \sigma(W_{ii} x_t + b_{ii} + W_{hi} h_{t-1} + b_{hi}) \\
            f_t = \sigma(W_{if} x_t + b_{if} + W_{hf} h_{t-1} + b_{hf}) \\
            g_t = \tanh(W_{ig} x_t + b_{ig} + W_{hg} h_{t-1} + b_{hg}) \\
            o_t = \sigma(W_{io} x_t + b_{io} + W_{ho} h_{t-1} + b_{ho}) \\
            c_t = f_t \odot c_{t-1} + i_t \odot g_t \\
            h_t = o_t \odot \tanh(c_t) \\
        \end{array}

    :param input_size: Dimensiones de las características de entrada.
    :param hidden_size: Dimensiones de las características ocultas.
    :param num_layers: Número de capas apiladas. por defecto: 1.
    :param batch_first: Si batch_first es True, la forma de entrada debe ser [batch_size, seq_len, feature_dim],
     si batch_first es False, la forma de entrada debe ser [seq_len, batch_size, feature_dim], por defecto: True.
    :param use_bias: Si use_bias es False, este módulo no contendrá sesgo. por defecto: True.
    :param bidirectional: Si bidirectional es True, el módulo será un LSTM bidireccional. por defecto: False.
    :param dtype: El tipo de dato del parámetro, por defecto: None, usa el tipo de dato predeterminado kfloat32, que representa un número de punto flotante de 32 bits.
    :param name: nombre de la capa de salida

    :return: Una instancia del módulo LSTM.

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
    
    Aplica una RNN de unidad recurrente cerrada (GRU) multicapa a una secuencia de entrada de longitud dinámica.

    La primera entrada debe ser una secuencia de lotes de longitud variable definida
    a través de la clase ``tensor.PackedSequence``.
    La clase ``tensor.PackedSequence`` se puede construir
    llamando a las funciones en sucesión: ``pad_sequence``, ``pack_pad_sequence``.

    La primera salida de Dynamic_GRU también es una clase ``tensor.PackedSequence``,
    que se puede desempaquetar en un QTensor normal usando ``tensor.pad_pack_sequence``.

    Para cada elemento en la secuencia de entrada, cada capa calcula la siguiente fórmula:

    .. math::
        \begin{array}{ll}
            r_t = \sigma(W_{ir} x_t + b_{ir} + W_{hr} h_{(t-1)} + b_{hr}) \\
            z_t = \sigma(W_{iz} x_t + b_{iz} + W_{hz} h_{(t-1)} + b_{hz}) \\
            n_t = \tanh(W_{in} x_t + b_{in} + r_t * (W_{hn} h_{(t-1)}+ b_{hn})) \\
            h_t = (1 - z_t) * n_t + z_t * h_{(t-1)}
        \end{array}

    :param input_size: Dimensión de las características de entrada.
    :param hidden_size: Dimensión de las características ocultas.
    :param num_layers: Número de capas recurrentes. Por defecto: 1
    :param batch_first: Si es True, la forma de entrada es [tamaño del lote, longitud de secuencia, dimensión de características]. Si es False, la forma de entrada es [longitud de secuencia, tamaño del lote, dimensión de características], por defecto True.
    :param use_bias: Si es False, la capa no usa pesos de sesgo b_ih y b_hh. Por defecto: true.
    :param bidirectional: Si es true, se convierte en un GRU bidireccional. Por defecto: false.
    :param dtype: El tipo de dato del parámetro, por defecto: None, usa el tipo de dato predeterminado kfloat32, que representa un número de punto flotante de 32 bits.
    :param name: nombre de la capa de salida

    :return: Una clase Dynamic_GRU

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
    
    Aplica redes neuronales recurrentes (RNN) a secuencias de entrada de longitud dinámica.

    La primera entrada debe ser una secuencia de lotes de longitud variable definida
    a través de la clase ``tensor.PackedSequence``.
    La clase ``tensor.PackedSequence`` se puede construir
    llamando a las funciones en sucesión: ``pad_sequence``, ``pack_pad_sequence``.

    La primera salida de Dynamic_RNN también es una clase ``tensor.PackedSequence``,
    que se puede desempaquetar en un QTensor normal usando ``tensor.pad_pack_sequence``.

    Módulo de Red Neuronal Recurrente (RNN), usando :math:`\tanh` o :math:`\text{ReLU}` como función de activación. Soporta configuración bidireccional y multicapa.
    La fórmula de cálculo del RNN unidireccional de una sola capa es la siguiente:

    .. math::
        h_t = \tanh(W_{ih} x_t + b_{ih} + W_{hh} h_{(t-1)} + b_{hh})
    
    Si :attr:`nonlinearity` es ``'relu'``, entonces :math:`\text{ReLU}` reemplazará a :math:`\tanh`.

    :param input_size: Dimensión de las características de entrada.
    :param hidden_size: Dimensión de las características ocultas.
    :param num_layers: Número de capas RNN apiladas, por defecto: 1.
    :param nonlinearity: Función de activación no lineal, por defecto es ``'tanh'``.
    :param batch_first: Si es True, la forma de entrada es [tamaño del lote, longitud de secuencia, dimensión de características],
      Si es False, la forma de entrada es [longitud de secuencia, tamaño del lote, dimensión de características], por defecto True.
    :param use_bias: Si es False, el módulo no aplica elementos de sesgo, por defecto: True.
    :param bidirectional: Si es True, se convierte en un RNN bidireccional, por defecto: False.
    :param dtype: El tipo de dato del parámetro, por defecto: None, usa el tipo de dato predeterminado kfloat32, que representa un número de punto flotante de 32 bits.
    :param name: nombre de la capa de salida

    :return: instancia de Dynamic_RNN

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
    
    Aplica RNN de Memoria a Largo Plazo (LSTM) a secuencias de entrada de longitud dinámica.

    La primera entrada debe ser una secuencia de lotes de longitud variable definida
    a través de la clase ``tensor.PackedSequence``.
    La clase ``tensor.PackedSequence`` se puede construir
    llamando a las funciones en sucesión: ``pad_sequence``, ``pack_pad_sequence``.

    La primera salida de Dynamic_LSTM también es una clase ``tensor.PackedSequence``,
    que se puede desempaquetar en un QTensor normal usando ``tensor.pad_pack_sequence``.

    Módulo de Red Neuronal Recurrente (RNN), usando :math:`\tanh` o :math:`\text{ReLU}` como función de activación. Soporta configuración bidireccional y multicapa.
    La fórmula de cálculo del LSTM unidireccional de una sola capa es la siguiente:

    .. math::
        \begin{array}{ll} \\
            i_t = \sigma(W_{ii} x_t + b_{ii} + W_{hi} h_{t-1} + b_{hi}) \\
            f_t = \sigma(W_{if} x_t + b_{if} + W_{hf} h_{t-1} + b_{hf}) \\
            g_t = \tanh(W_{ig} x_t + b_{ig} + W_{hg} h_{t-1} + b_{hg}) \\
            o_t = \sigma(W_{io} x_t + b_{io} + W_{ho} h_{t-1} + b_{ho}) \\
            c_t = f_t \odot c_{t-1} + i_t \odot g_t \\
            h_t = o_t \odot \tanh(c_t) \\
        \end{array}

    :param input_size: Dimensión de las características de entrada.
    :param hidden_size: Dimensión de las características ocultas.
    :param num_layers: Número de capas LSTM apiladas, por defecto: 1.
    :param batch_first: Si es True, la forma de entrada es [tamaño del lote, longitud de secuencia, dimensión de características],
      Si es False, la forma de entrada es [longitud de secuencia, tamaño del lote, dimensión de características], por defecto True.
    :param use_bias: Si es False, el módulo no aplica elementos de sesgo, por defecto: True.
    :param bidirectional: Si es True, se convierte en un LSTM bidireccional, por defecto: False.
    :param dtype: El tipo de dato del parámetro, por defecto: None, usa el tipo de dato predeterminado kfloat32, que representa un número de punto flotante de 32 bits.
    :param name: nombre de la capa de salida

    :return: instancia de Dynamic_LSTM

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

    Reduce/aumenta la resolución de la entrada.

    Actualmente solo se soportan datos de entrada tetradimensionales.

    Las dimensiones de entrada se interpretan en la forma: `B x C x H x W`.

    Los modos disponibles para redimensionar son: ``nearest``, ``bilinear``, ``bicubic``.

    :param size: tamaño espacial de salida.
    :param scale_factor: multiplicador para el tamaño espacial.
    :param mode: algoritmo usado para el sobremuestreo ``nearest`` | ``bilinear`` | ``bicubic``.
    :param align_corners: Geométricamente, consideramos los píxeles de la
            entrada y salida como cuadrados en lugar de puntos.
            Si se establece en ``True``, los tensores de entrada y salida se alinean por los
            puntos centrales de sus píxeles de esquina, preservando los valores en las esquinas.
            Si se establece en ``False``, los tensores de entrada y salida se alinean por los
            puntos de esquina de sus píxeles de esquina, y la interpolación usa relleno con valor de borde
            para valores fuera de los límites, haciendo esta operación *independiente* del tamaño de entrada
            cuando :attr:`scale_factor` se mantiene igual. Esto solo tiene efecto cuando :attr:`mode`
            es ``bilinear``, ``bicubic``.
    :param recompute_scale_factor: recalcula el scale_factor para usarlo en el cálculo de interpolación.
    :param name: Nombre del módulo.

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

    Se utiliza para fusionar los módulos vecinos correspondientes del modelo en la etapa de inferencia en un solo módulo,
    lo que reduce la cantidad de cálculo en la etapa de inferencia del modelo y aumenta su velocidad.

    Las secuencias de módulos actualmente soportadas son las siguientes:

    conv, bn

    linear, bn

    Las otras secuencias permanecen sin cambios, donde el primer módulo de la lista se reemplaza con el módulo fusionado, y los demás se reemplazan con ``Identity``.

    :param input: Incluye el modelado de los módulos de fusión.

    :return: Modelo con módulos fusionados.

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

    Mecanismo de atención de producto punto escalado SDPA.

    :param attn_mask: Máscara de atención; la forma debe ser transmisible a la forma de los pesos de atención.
    :param dropout_p: Probabilidad de dropout; si es mayor que 0.0, se aplica dropout.
    :param scale: Factor de escala aplicado antes de softmax.
    :param is_causal: Si es true, asume enmascaramiento de atención causal en la esquina superior izquierda y da error si se establecen tanto attn_mask como is_causal.
    
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


Capa de Función de Pérdida
********************************************************

.. note::

        Tenga en cuenta que, a diferencia de pytorch y otros frameworks, en la función forward de la siguiente función de pérdida, el primer parámetro es la etiqueta y el segundo es el valor predicho.

MeanSquaredError
=================================

.. py:class:: pyvqnet.nn.MeanSquaredError

    Crea un criterio que mide el error cuadrático medio (norma L2 al cuadrado) entre
    cada elemento en la entrada :math:`x` y el objetivo :math:`y`.

    La pérdida no reducida se puede describir como:

    .. math::
        \ell(x, y) = L = \{l_1,\dots,l_N\}^\top, \quad
        l_n = \left( x_n - y_n \right)^2,

    donde :math:`N` es el tamaño del lote. luego:

    .. math::
        \ell(x, y) =
            \operatorname{mean}(L)


    :math:`x` y :math:`y` son QTensors de formas arbitrarias con un total
    de :math:`n` elementos cada uno.

    La operación de media aún opera sobre todos los elementos, y divide por :math:`n`.

    :param name: nombre de la capa de salida

    :return: una clase MeanSquaredError

    Parámetros para la función forward de pérdida:

        x: :math:`(N, *)` donde :math:`*` significa cualquier número de dimensiones adicionales

        y: :math:`(N, *)`, misma forma que la entrada

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

    Mide la entropía cruzada binaria entre el objetivo y la salida:

    La pérdida no reducida se puede describir como:

    .. math::
        \ell(x, y) = L = \{l_1,\dots,l_N\}^\top, \quad
        l_n = - w_n \left[ y_n \cdot \log x_n + (1 - y_n) \cdot \log (1 - x_n) \right],

    donde :math:`N` es el tamaño del lote.

    .. math::
        \ell(x, y) = \operatorname{mean}(L)

    :return: una clase BinaryCrossEntropy

    Parámetros para la función forward de pérdida:

        x: :math:`(N, *)` donde :math:`*` significa cualquier número de dimensiones adicionales

        y: :math:`(N, *)`, misma forma que la entrada

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

    Este criterio combina LogSoftmax y NLLLoss en una sola clase.

    La pérdida se puede describir como sigue, donde `class` es el índice de la clase objetivo:

    .. math::
        \text{loss}(x, class) = -\log\left(\frac{\exp(x[class])}{\sum_j \exp(x[j])}\right)
                       = -x[class] + \log\left(\sum_j \exp(x[j])\right)

    :return: una clase CategoricalCrossEntropy

    Parámetros para la función forward de pérdida:

        x: :math:`(N, *)` donde :math:`*` significa cualquier número de dimensiones adicionales

        y: :math:`(N, *)`, misma forma que la entrada, debe tener tipo de dato entero de 64 bits.

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

    Este criterio combina LogSoftmax y NLLLoss en una sola clase con mayor estabilidad numérica.

    La pérdida se puede describir como sigue, donde `class` es el índice de la clase objetivo:

    .. math::
        \text{loss}(x, class) = -\log\left(\frac{\exp(x[class])}{\sum_j \exp(x[j])}\right)
                       = -x[class] + \log\left(\sum_j \exp(x[j])\right)

    :return: una clase SoftmaxCrossEntropy

    Parámetros para la función forward de pérdida:

        x: :math:`(N, *)` donde :math:`*` significa cualquier número de dimensiones adicionales

        y: :math:`(N, *)`, misma forma que la entrada, debe tener tipo de dato entero de 64 bits.

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

    La pérdida de log-verosimilitud negativa promedio. Es útil para entrenar un problema de clasificación con `C` clases.

    Se espera que `x` proporcionado a través de una llamada forward contenga log-probabilidades de cada clase. `x` debe ser un tensor de tamaño :math:`(N, C)` o :math:`(N, C, d_1, d_2, ..., d_K)`
    con :math:`K \geq 1` para el caso `K`-dimensional. `y` que esta pérdida espera debe ser un índice de clase en el rango :math:`[0, C-1]` donde `C = número de clases`.

    .. math::

        \ell(x, y) = L = \{l_1,\dots,l_N\}^\top, \quad
        l_n = -
            \sum_{n=1}^N \frac{1}{N}x_{n,y_n}, \quad

    :return: una clase NLL_Loss

    Parámetros para la función forward de pérdida:

        x: :math:`(N, *)`, la salida de la función de pérdida, que puede ser una variable multidimensional.

        y: :math:`(N, *)`, el valor verdadero esperado por la función de pérdida, debe tener tipo de dato entero de 64 bits.


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

    Este criterio combina LogSoftmax y NLLLoss en una sola clase.

    Se espera que `x` contenga puntuaciones brutas y no normalizadas para cada clase. `x` debe ser un tensor de tamaño :math:`(C)` para entrada sin lote, :math:`(N, C)` o :math:`(N, C, d_1, d_2, ..., d_K)` con :math:`K \geq 1` para el caso `K`-dimensional.

    La pérdida se puede describir como sigue, donde `class` es el índice de la clase objetivo:

    .. math::

        \text{loss}(x, class) = -\log\left(\frac{\exp(x[class])}{\sum_j \exp(x[j])}\right)
                       = -x[class] + \log\left(\sum_j \exp(x[j])\right)

    :return: una clase CrossEntropyLoss

    Parámetros para la función forward de pérdida:

        x: :math:`(N, *)`, la salida de la función de pérdida, que puede ser una variable multidimensional.

        y: :math:`(N, *)`, el valor verdadero esperado por la función de pérdida, debe tener tipo de dato entero de 64 bits.


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



Función de Activación
********************************************************


Activation
=================================
.. py:class:: pyvqnet.nn.activation.Activation

    Clase base de activación. Las funciones de activación específicas heredan de esta clase.

Sigmoid
=================================
.. py:class:: pyvqnet.nn.Sigmoid(name: str = '')

        Aplica una función de activación sigmoide a la capa dada.

        .. math::
            \text{Sigmoid}(x) = \frac{1}{1 + \exp(-x)}

        :param name: nombre de la capa de salida
        :return: capa de activación sigmoide

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

        Aplica la función de activación softplus a la capa dada.

        .. math::
            \text{Softplus}(x) = \log(1 + \exp(x))

        :param name: nombre de la capa de salida
        :return: capa de activación softplus

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

        Aplica la función de activación softsign a la capa dada.

        .. math::
            \text{SoftSign}(x) = \frac{x}{ 1 + |x|}

        :param name: nombre de la capa de salida
        :return: capa de activación softsign

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

    Aplica una función de activación softmax a la capa dada.

    .. math::
        \text{Softmax}(x_{i}) = \frac{\exp(x_i)}{\sum_j \exp(x_j)}


    :param axis: dimensión sobre la cual operar (-1 para el último eje), por defecto = -1
    :param name: nombre de la capa de salida
    :return: capa de activación softmax

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

    Aplica una función de activación sigmoide dura a la capa dada.

    .. math::
        \text{Hardsigmoid}(x) = \begin{cases}
            0 & \text{ if } x \le -3, \\
            1 & \text{ if } x \ge +3, \\
            x / 6 + 1 / 2 & \text{otherwise}
        \end{cases}

    :param name: nombre de la capa de salida
    :return: capa de activación sigmoide dura

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

    Aplica una función de activación de unidad lineal rectificada a la capa dada.

    .. math::
        \text{ReLu}(x) = \begin{cases}
        x, & \text{ if } x > 0\\
        0, & \text{ if } x \leq 0
        \end{cases}


    :param name: nombre de la capa de salida
    :return: capa de activación ReLu

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

    Aplica la versión con fuga de una función de activación de unidad lineal rectificada
    a la capa dada.

    .. math::
        \text{LeakyRelu}(x) =
        \begin{cases}
        x, & \text{ if } x \geq 0 \\
        \alpha * x, & \text{ otherwise }
        \end{cases}

    :param alpha: coeficiente de LeakyRelu, por defecto: 0.01
    :param name: nombre de la capa de salida
    :return: capa de activación ReLu con fuga

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
    
    Aplica la función de unidad lineal de error gaussiano:

    .. math:: \text{GELU}(x) = x * \Phi(x)

    Cuando el parámetro de aproximación es 'tanh', GELU se estima mediante:

    .. math:: \text{GELU}(x) = 0.5 * x * (1 + \text{Tanh}(\sqrt{2 / \pi} * (x + 0.044715 * x^3)))

    :param approximate: Método de cálculo aproximado, por defecto es "tanh".
    :param name: Nombre de la capa de función de activación, por defecto es "".

    :return: Instancia de capa de función de activación Gelu.

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

    Aplica la función de activación de unidad lineal exponencial a la capa dada.

    .. math::
        \text{ELU}(x) = \begin{cases}
        x, & \text{ if } x > 0\\
        \alpha * (\exp(x) - 1), & \text{ if } x \leq 0
        \end{cases}

    :param alpha: coeficiente de Elu, por defecto: 1.0
    :param name: nombre de la capa de salida
    :return: capa de activación Elu

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

    Aplica la función de activación de tangente hiperbólica a la capa dada.

    .. math::
        \text{Tanh}(x) = \frac{\exp(x) - \exp(-x)} {\exp(x) + \exp(-x)}

    :param name: nombre de la capa de salida
    :return: capa de activación de tangente hiperbólica

    Examples::

        from pyvqnet.nn import Tanh
        from pyvqnet.tensor import QTensor
        layer = Tanh()
        y = layer(QTensor([-1, 2.0, -3, 4.0]))
        print(y)

        # [-0.7615942, 0.9640276, -0.9950548, 0.9993293]


.. _Optimizer:

Módulo Optimizador
********************************************************


Optimizer
=================================
.. py:class:: pyvqnet.optim.optimizer.Optimizer(params, lr=0.01)

    Clase base para todos los optimizadores.

    :param params: parámetros del modelo que deben ser optimizados
    :param lr: tasa de aprendizaje del modelo (por defecto: 0.01)

adadelta
=================================
.. py:class:: pyvqnet.optim.adadelta.Adadelta(params, lr=0.01, beta=0.99, epsilon=1e-8)

    ADADELTA: Un Método de Tasa de Aprendizaje Adaptativa. referencia: (https://arxiv.org/abs/1212.5701)

    .. math::

        E(g_t^2) &= \beta * E(g_{t-1}^2) + (1-\beta) * g^2\\
        Square\_avg &= \sqrt{ ( E(dx_{t-1}^2) + \epsilon ) / ( E(g_t^2) + \epsilon ) }\\
        E(dx_t^2) &= \beta * E(dx_{t-1}^2) + (1-\beta) * (-g*square\_avg)^2 \\
        param\_new &= param - lr * Square\_avg

    :param params: parámetros del modelo que deben ser optimizados
    :param lr: tasa de aprendizaje del modelo (por defecto: 0.01)
    :param beta: para calcular un promedio móvil de los gradientes al cuadrado (por defecto: 0.99)
    :param epsilon: término agregado al denominador para mejorar la estabilidad numérica (por defecto: 1e-8)
    :return: un optimizador Adadelta

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

    Implementa el algoritmo Adagrad. referencia: (https://databricks.com/glossary/adagrad)

    .. math::
        \begin{aligned}
        moment\_new &= moment + g * g\\param\_new
        &= param - \frac{lr * g}{\sqrt{moment\_new} + \epsilon}
        \end{aligned}

    :param params: parámetros del modelo que deben ser optimizados
    :param lr: tasa de aprendizaje del modelo (por defecto: 0.01)
    :param epsilon: término agregado al denominador para mejorar la estabilidad numérica (por defecto: 1e-8)
    :return: un optimizador Adagrad

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
    
    Implementa el algoritmo AdamW.

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
    
    Si el parámetro amsgrad es True

    .. math::
        moment\_2\_max = max(moment\_2\_max,moment\_2)
    .. math::
        param\_new=param\_new-lr*\frac{moment\_1}{\sqrt{moment\_2\_max}+\epsilon}
    
    de lo contrario:

    .. math::
        param\_new=param\_new-lr*\frac{moment\_1}{\sqrt{moment\_2}+\epsilon}

    :param params: Parámetros del modelo que necesitan ser optimizados.
    :param lr: tasa de aprendizaje (por defecto: 0.01).
    :param beta1: Coeficiente usado para calcular el promedio móvil del gradiente y su cuadrado (por defecto: 0.9).
    :param beta2: Coeficiente usado para calcular el promedio móvil del gradiente y su cuadrado (por defecto: 0.999).
    :param epsilon: Constante para agregar al denominador para mejorar la estabilidad numérica (por defecto: 1e-8).
    :param weight_decay: Coeficiente de decaimiento de peso, por defecto 0.01.
    :param amsgrad: Si usar la variante AMSGrad de este algoritmo (por defecto: False).
    :return: Un optimizador AdamW.

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

    Adam: Un Método para Optimización Estocástica referencia: (https://arxiv.org/abs/1412.6980), ajusta dinámicamente la tasa de aprendizaje de cada parámetro usando las estimaciones del primer y segundo momento del gradiente.

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

    si amsgrad = True
    
    .. math::
        moment\_2\_max = max(moment\_2\_max,moment\_2)
    .. math::
        param\_new=param-lr*\frac{moment\_1}{\sqrt{moment\_2\_max}+\epsilon} 

    si no

    .. math::
        param\_new=param-lr*\frac{moment\_1}{\sqrt{moment\_2}+\epsilon} 


    :param params: parámetros del modelo que deben ser optimizados
    :param lr: tasa de aprendizaje del modelo (por defecto: 0.01)
    :param beta1: coeficientes usados para calcular promedios móviles del gradiente y su cuadrado (por defecto: 0.9)
    :param beta2: coeficientes usados para calcular promedios móviles del gradiente y su cuadrado (por defecto: 0.999)
    :param epsilon: término agregado al denominador para mejorar la estabilidad numérica (por defecto: 1e-8)
    :param weight_decay: Coeficiente de decaimiento de peso, por defecto 0.
    :param amsgrad: si usar la variante AMSGrad de este algoritmo (por defecto: False)
    :return: un optimizador Adam

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

    Implementa el algoritmo Adamax (una variante de Adam basada en la norma infinito). referencia: (https://arxiv.org/abs/1412.6980)

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

    :param params: parámetros del modelo que deben ser optimizados
    :param lr: tasa de aprendizaje del modelo (por defecto: 0.01)
    :param beta1: coeficientes usados para calcular promedios móviles del gradiente y su cuadrado (por defecto: 0.9)
    :param beta2: coeficientes usados para calcular promedios móviles del gradiente y su cuadrado (por defecto: 0.999)
    :param epsilon: término agregado al denominador para mejorar la estabilidad numérica (por defecto: 1e-8)
    :return: un optimizador Adamax

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

    Implementa el algoritmo RMSprop. referencia: (https://arxiv.org/pdf/1308.0850v5.pdf)

    .. math::
        s_{t+1} = s_{t} + (1 - \beta)*(g)^2

    .. math::
        param_new = param -  \frac{g}{\sqrt{s_{t+1}} + epsilon}

    :param params: parámetros del modelo que deben ser optimizados
    :param lr: tasa de aprendizaje del modelo (por defecto: 0.01)
    :param beta: coeficientes usados para calcular promedios móviles del gradiente y su cuadrado (por defecto: 0.99)
    :param epsilon: término agregado al denominador para mejorar la estabilidad numérica (por defecto: 1e-8)
    :return: un optimizador RMSProp

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

    Implementa el algoritmo SGD. referencia: (https://en.wikipedia.org/wiki/Stochastic_gradient_descent)

    .. math::

        \\param\_new=param-lr*g\\

    :param params: parámetros del modelo que deben ser optimizados
    :param lr: tasa de aprendizaje del modelo (por defecto: 0.01)
    :param momentum: factor de momento (por defecto: 0)
    :param nesterov: habilita el momento de Nesterov (por defecto: False)
    :return: un optimizador SGD

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


Métricas
********************************************************


MSE
=================================

.. py:class:: pyvqnet.utils.metrics.MSE(y_true_Qtensor, y_pred_Qtensor)

    MSE: Error Cuadrático Medio.

    :param y_true_Qtensor: Un QTensor de forma como (n_samples,) o (n_samples, n_outputs), valor objetivo verdadero.
    :param y_pred_Qtensor: Un QTensor de forma como (n_samples,) o (n_samples, n_outputs), valores objetivo estimados.
    :return: retorna un resultado float.

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

    RMSE: Raíz del Error Cuadrático Medio.

    :param y_true_Qtensor: Un QTensor de forma como (n_samples,) o (n_samples, n_outputs), valor objetivo verdadero.
    :param y_pred_Qtensor: Un QTensor de forma como (n_samples,) o (n_samples, n_outputs), valores objetivo estimados.
    :return: retorna un resultado float.

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

    MAE: Error Absoluto Medio.

    :param y_true_Qtensor: Un QTensor de forma como (n_samples,) o (n_samples, n_outputs), valor objetivo verdadero.
    :param y_pred_Qtensor: Un QTensor de forma como (n_samples,) o (n_samples, n_outputs), valores objetivo estimados.
    :return: retorna un resultado float.

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

    R_Square: función de puntuación de regresión R^2 (coeficiente de determinación).
    La mejor puntuación posible es 1.0, que puede ser negativa
    (ya que el modelo puede deteriorarse arbitrariamente).
    Uno que siempre predice el valor esperado de y,
    ignorando las características de entrada, obtendrá una puntuación R^2 de 0.0.
    
    :param y_true_Qtensor: Un QTensor de forma como (n_samples,) o (n_samples, n_outputs), valor objetivo verdadero.
    :param y_pred_Qtensor: Un QTensor de forma como (n_samples,) o (n_samples, n_outputs), valores objetivo estimados.
    :param sample_weight: Arreglo de forma como (n_samples,), peso de muestra opcional, por defecto: None.
    :return: retorna un resultado float.

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

    Calcula la precisión, exhaustividad (recall) y puntuación F1 de los valores predichos en una tarea de clasificación binaria. Los valores predichos y verdaderos deben ser QTensors de forma similar (n_samples,), con valores 0 o 1, representando las etiquetas de las dos clases.
    
    :param y_true_Qtensor: Un QTensor 1D, valor objetivo verdadero.
    :param y_pred_Qtensor: Un QTensor 1D, valor objetivo estimado.

    :returns: 
        - precision - resultado de precisión
        - recall - resultado de exhaustividad
        - f1 - puntuación f1

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

    Cálculos de precisión, exhaustividad (recall) y puntuación F1 para tareas de clasificación múltiple. donde el valor predicho y el valor verdadero son QTensors de forma similar (n_samples,), y los valores son enteros de 0 a N-1, que representan las etiquetas de N clases.

    :param y_true_Qtensor: Un QTensor 1D, valor objetivo verdadero.
    :param y_pred_Qtensor: Un QTensor 1D, valor objetivo estimado.
    :param N: N clases (número de clases).
    :param average: cadena, ['micro', 'macro', 'weighted'].
             Este parámetro es obligatorio para objetivos multiclase/multietiqueta.
             
             ``'micro'``: Calcula las métricas globalmente contando los totales verdaderos, falsos negativos y falsos positivos.
             
             ``'macro'``: Calcula la métrica para cada etiqueta y encuentra su valor no ponderado. Esto significa que no se considera el balance de las etiquetas.
             
             ``'weighted'``: Calcula las métricas para cada etiqueta y encuentra su promedio (el número de instancias verdaderas de cada etiqueta). Esto cambia ``'macro'`` para tener en cuenta el desbalance de etiquetas; esto puede resultar en puntuaciones F que no estén entre precisión y exhaustividad.
    
    :returns: 
        - precision - resultado de precisión
        - recall - resultado de exhaustividad
        - f1 - puntuación f1

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

    Cálculos de precisión, exhaustividad (recall) y puntuación F1 para tareas de clasificación múltiple. donde los valores predichos y verdaderos son QTensors de forma similar (n_samples, N), donde los valores son etiquetas codificadas one-hot de N dimensiones.

    :param y_true_Qtensor: Un QTensor 1D, valor objetivo verdadero.
    :param y_pred_Qtensor: Un QTensor 1D, valor objetivo estimado.
    :param N: N clases (número de clases).
    :param average: cadena, ['micro', 'macro', 'weighted'].
             Este parámetro es obligatorio para objetivos multiclase/multietiqueta.
             
             ``'micro'``: Calcula las métricas globalmente contando los totales verdaderos, falsos negativos y falsos positivos.
             
             ``'macro'``: Calcula la métrica para cada etiqueta y encuentra su valor no ponderado. Esto significa que no se considera el balance de las etiquetas.
             
             ``'weighted'``: Calcula las métricas para cada etiqueta y encuentra su promedio (el número de instancias verdaderas de cada etiqueta). Esto cambia ``'macro'`` para tener en cuenta el desbalance de etiquetas; esto puede resultar en puntuaciones F que no estén entre precisión y exhaustividad.
    
    :returns: 
        - precision - resultado de precisión
        - recall - resultado de exhaustividad
        - f1 - puntuación f1

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

    Calcula la precisión, exhaustividad y puntuación F1 de la tarea de clasificación.

    :param y_true_Qtensor: Un QTensor de forma [n_samples].
                             Una etiqueta binaria verdadera. Si la etiqueta no es {1,1} o {0,1}, pos_label debe darse explícitamente.
    :param y_pred_Qtensor: Un QTensor de forma [n_samples].
                             Puntuación objetivo, que puede ser una estimación de probabilidad de clase positiva, valor de confianza o una medida sin umbral de la decisión (devuelta por "decision_function" en algunos clasificadores)
    :param pos_label: int o str. La etiqueta de la clase positiva. por defecto=None.
                      Cuando ``pos_label`` es None, si ``y_true_Qtensor`` está en {-1,1} o {0,1}, ``pos_label`` se establece en 1, de lo contrario se generará un error.
    :param sample_weight: arreglo de forma (n_samples,), por defecto=None.
    :param drop_intermediate: booleano, opcional (por defecto=True).
                     Si se deben reducir algunos umbrales subóptimos que no aparecen en la curva ROC dibujada.
    :return: resultado float de salida.

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


Compatibilidad con Triton
*********************************************************

`triton <https://triton-lang.org/main/index.html>`_ es un lenguaje y compilador para escribir kernels GPU eficientes para aprendizaje profundo.
Los usuarios escriben código Python similar a NumPy, y luego Triton lo compila en código GPU eficiente (similar a CUDA pero de más alto nivel).
Triton depende de algunas interfaces de PyTorch. VQNet implementa una interfaz similar a PyTorch, permitiendo la integración con código escrito en Triton para la propagación hacia adelante y hacia atrás del modelo.

Instalar triton:

.. code-block::

    pip install triton

El siguiente ejemplo está modificado del ejemplo oficial de triton: `layer-norm <https://triton-lang.org/main/getting-started/tutorials/05-layer-norm.html>`_.
Este ejemplo debe ejecutarse en Linux con GPU, y deben instalarse triton y pytorch (usado para comparar la corrección del cálculo):

.. code-block::

    import torch as real_torch
    import pyvqnet as torch
    import triton
    import triton.language as tl



    DEVICE = torch.DEV_GPU_0

    @triton.jit
    def _layer_norm_fwd_fused(
        X,  # puntero a la entrada
        Y,  # puntero a la salida
        W,  # puntero a los pesos
        B,  # puntero a los sesgos
        Mean,  # puntero a la media
        Rstd,  # puntero a 1/desviación estándar
        stride,  # cuánto aumentar el puntero al moverse 1 fila
        N,  # número de columnas en X
        eps,  # épsilon para evitar división por cero
        BLOCK_SIZE: tl.constexpr,
    ):
        # Mapea el id del programa a la fila de X e Y que debe calcular.
        row = tl.program_id(0)
        Y += row * stride
        X += row * stride
        # Calcular la media
        mean = 0
        _mean = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
        for off in range(0, N, BLOCK_SIZE):
            cols = off + tl.arange(0, BLOCK_SIZE)
            a = tl.load(X + cols, mask=cols < N, other=0.).to(tl.float32)
            _mean += a
        mean = tl.sum(_mean, axis=0) / N
        # Calcular la varianza
        _var = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
        for off in range(0, N, BLOCK_SIZE):
            cols = off + tl.arange(0, BLOCK_SIZE)
            x = tl.load(X + cols, mask=cols < N, other=0.).to(tl.float32)
            x = tl.where(cols < N, x - mean, 0.)
            _var += x * x
        var = tl.sum(_var, axis=0) / N
        rstd = 1 / tl.sqrt(var + eps)
        # Escribir media / rstd
        tl.store(Mean + row, mean)
        tl.store(Rstd + row, rstd)
        # Normalizar y aplicar transformación lineal
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
    def _layer_norm_bwd_dx_fused(DX,  # puntero al gradiente de entrada
                                DY,  # puntero al gradiente de salida
                                DW,  # puntero a la suma parcial del gradiente de pesos
                                DB,  # puntero a la suma parcial del gradiente de sesgos
                                X,  # puntero a la entrada
                                W,  # puntero a los pesos
                                Mean,  # puntero a la media
                                Rstd,  # puntero a 1/desviación estándar
                                Lock,  # puntero al bloqueo
                                stride,  # cuánto aumentar el puntero al moverse 1 fila
                                N,  # número de columnas en X
                                GROUP_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr):
        # Mapea el id del programa a los elementos de X, DX y DY que debe calcular.
        row = tl.program_id(0)
        cols = tl.arange(0, BLOCK_SIZE_N)
        mask = cols < N
        X += row * stride
        DY += row * stride
        DX += row * stride
        # Desplazar bloqueos y punteros de gradiente de pesos/sesgos para reducción paralela
        lock_id = row % GROUP_SIZE_M
        Lock += lock_id
        Count = Lock + GROUP_SIZE_M
        DW = DW + lock_id * N + cols
        DB = DB + lock_id * N + cols
        # Cargar datos a SRAM

        x = tl.load(X + cols, mask=mask, other=0).to(tl.float32)
        dy = tl.load(DY + cols, mask=mask, other=0).to(tl.float32)
        w = tl.load(W + cols, mask=mask).to(tl.float32)
        mean = tl.load(Mean + row)
        rstd = tl.load(Rstd + row)
        # Calcular dx
        xhat = (x - mean) * rstd
        wdy = w * dy
        xhat = tl.where(mask, xhat, 0.)
        wdy = tl.where(mask, wdy, 0.)
        c1 = tl.sum(xhat * wdy, axis=0) / N
        c2 = tl.sum(wdy, axis=0) / N
        dx = (wdy - (xhat * c1 + c2)) * rstd
        # Escribir dx
        tl.store(DX + cols, dx, mask=mask)

        # Acumular sumas parciales para dw/db
        partial_dw = (dy * xhat).to(w.dtype)
        partial_db = (dy).to(w.dtype)
        while tl.atomic_cas(Lock, 0, 1) == 1:
            pass
        count = tl.load(Count)
        # El primer almacenamiento no acumula
        if count == 0:
            tl.atomic_xchg(Count, 1)
        else:
            partial_dw += tl.load(DW, mask=mask)
            partial_db += tl.load(DB, mask=mask)
        tl.store(DW, partial_dw, mask=mask)
        tl.store(DB, partial_db, mask=mask)

        # se necesita una barrera para asegurar que todos los hilos terminen antes de
        # liberar el bloqueo
        tl.debug_barrier()

        # Liberar el bloqueo
        tl.atomic_xchg(Lock, 0)


    @triton.jit
    def _layer_norm_bwd_dwdb(DW,  # puntero a la suma parcial del gradiente de pesos
                            DB,  # puntero a la suma parcial del gradiente de sesgos
                            FINAL_DW,  # puntero al gradiente de pesos
                            FINAL_DB,  # puntero al gradiente de sesgos
                            M,  # GROUP_SIZE_M
                            N,  # número de columnas
                            BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr):
        # Mapea el id del programa a los elementos de DW y DB que debe calcular.
        pid = tl.program_id(0)
        cols = pid * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
        dw = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
        db = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
        # Iterar a través de las filas de DW y DB para sumar las sumas parciales.
        for i in range(0, M, BLOCK_SIZE_M):
            rows = i + tl.arange(0, BLOCK_SIZE_M)
            mask = (rows[:, None] < M) & (cols[None, :] < N)
            offs = rows[:, None] * N + cols[None, :]
            dw += tl.load(DW + offs, mask=mask, other=0.)
            db += tl.load(DB + offs, mask=mask, other=0.)
        # Escribir la suma final a la salida.
        sum_dw = tl.sum(dw, axis=0)
        sum_db = tl.sum(db, axis=0)
        tl.store(FINAL_DW + cols, sum_dw, mask=cols < N)
        tl.store(FINAL_DB + cols, sum_db, mask=cols < N)


    class LayerNorm(torch.autograd.Function):

        @staticmethod
        def forward(ctx, x, normalized_shape, weight, bias, eps):
            # asignar salida
            y = torch.empty_like(x)
            # reformatear datos de entrada en tensor 2D
            x_arg = x.reshape([-1, x.shape[-1]])
            M, N = x_arg.shape
            mean = torch.empty((M, ), dtype=torch.float32, device=x.device).data
            rstd = torch.empty((M, ), dtype=torch.float32, device=x.device).data
            # Menos de 64KB por característica: encolar kernel fusionado
            MAX_FUSED_SIZE = 65536 // x.element_size()
            BLOCK_SIZE = min(MAX_FUSED_SIZE, triton.next_power_of_2(N))
            if N > BLOCK_SIZE:
                raise RuntimeError("Esta normalización de capa no soporta dimensión de características >= 64KB.");
            # heurística para número de warps
            num_warps = min(max(BLOCK_SIZE // 256, 1), 8)
            # encolar kernel
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
            # heurística para la cantidad de flujo de reducción paralela para DW/DB
            N = w.shape[0]
            GROUP_SIZE_M = 64
            if N <= 8192: GROUP_SIZE_M = 96
            if N <= 4096: GROUP_SIZE_M = 128
            if N <= 1024: GROUP_SIZE_M = 256
            # asignar salida
            locks = torch.zeros(2 * GROUP_SIZE_M, dtype=torch.int32, device=w.device).data
            _dw = torch.zeros((GROUP_SIZE_M, N), dtype=x.dtype, device=w.device).data
            _db = torch.zeros((GROUP_SIZE_M, N), dtype=x.dtype, device=w.device).data
            dw = torch.empty((N, ), dtype=w.dtype, device=w.device).data
            db = torch.empty((N, ), dtype=w.dtype, device=w.device).data
            dx = torch.empty_like(dy).data
            # encolar kernel usando heurísticas del pase forward
            # también calcular sumas parciales para DW y DB
            x_arg = x.reshape([-1, x.shape[-1]])
            M, N = x_arg.shape

            _layer_norm_bwd_dx_fused[(M, )](  #
                dx, dy, _dw, _db, x, w, m, v, locks,  #
                x_arg.stride[0], N,  #
                BLOCK_SIZE_N=ctx.BLOCK_SIZE,  #
                GROUP_SIZE_M=GROUP_SIZE_M,  #
                num_warps=ctx.num_warps)

            grid = lambda meta: (triton.cdiv(N, meta['BLOCK_SIZE_N']), )

            # acumular sumas parciales en un kernel separado
            _layer_norm_bwd_dwdb[grid](
                _dw, _db, dw, db, min(GROUP_SIZE_M, M), N,  #
                BLOCK_SIZE_M=32,  #
                BLOCK_SIZE_N=128, num_ctas=1)
            return dx, None, dw, db, None

    preprocess = LayerNorm.preprocess
    layer_norm = LayerNorm.apply


    def test_layer_norm(M, N, dtype, eps=1e-5, device=DEVICE):
        # crear datos
        x_shape = (M, N)
        w_shape = (x_shape[-1], )
        weight = torch.rand(w_shape, dtype=dtype, device=device, requires_grad=True)
        bias = torch.rand(w_shape, dtype=dtype, device=device, requires_grad=True)
        x = -2.3 + 0.5 * torch.randn(x_shape, dtype=dtype, device=device)

        dy = .1 * torch.randn_like(x)
        x.requires_grad = True
        # pase forward
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
        # pase backward (triton)
        y_tri.backward(dy, retain_graph=True)
        dx_tri, dw_tri, db_tri = [_.grad.detach().cpu().numpy() for _ in [x, weight, bias]]
        x.grad, weight.grad, bias.grad = None, None, None
        # pase backward (torch)
        y_ref.backward(torch_dy, retain_graph=True)
        dx_ref, dw_ref, db_ref = [_.grad.detach().cpu().numpy() for _ in [torch_x, torch_weight, torch_bias]]
        # comparar
        import numpy as np
        assert np.allclose(y_tri.detach().cpu().numpy(), y_ref.detach().cpu().numpy(), atol=1e-2, rtol=0)
        assert np.allclose(dx_tri, dx_ref, atol=1e-2, rtol=0)
        assert np.allclose(db_tri, db_ref, atol=1e-2, rtol=0)
        assert np.allclose(dw_tri, dw_ref, atol=1e-2, rtol=0)
        print("same")
    test_layer_norm(12, 32, torch.float32)
