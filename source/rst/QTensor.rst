.. _qtensor_api:

Módulo QTensor
###########################

VQNet aprendizaje automático cuántico utiliza la estructura de datos QTensor que es una interfaz de Python. QTensor soporta operaciones comunes de matrices multidimensionales, incluyendo funciones de creación, funciones matemáticas, funciones lógicas, transformaciones de matrices, etc.



Funciones y Atributos de QTensor
******************************************


QTensor
==============================

.. py:class:: pyvqnet.tensor.tensor.QTensor(data, requires_grad=False, nodes=None, device=0, dtype=None, name='')

    Wrapper de estructura de datos con construcción dinámica de grafos computacionales
    y diferenciación automática.

    :param data: _core.Tensor o array numpy que representa un QTensor
    :param requires_grad: indica si se debe rastrear el gradiente del tensor, por defecto False
    :param nodes: lista de sucesores en el grafo computacional, por defecto None
    :param device: dispositivo actual para guardar QTensor, default = 0, usar CPU.
    :param dtype: El tipo de dato del parámetro, por defecto None, usa el tipo de dato por defecto: kfloat32, que representa un número de punto flotante de 32 bits.
    :param name: El nombre del QTensor, por defecto: "".

    :return: QTensor de salida


    Ejemplo::

        from pyvqnet.tensor import QTensor
        from pyvqnet.dtype import *
        import numpy as np

        t1 = QTensor(np.ones([2,3]))
        t2 =  QTensor([2,3,4j,5])
        t3 =  QTensor([[[2,3,4,5],[2,3,4,5]]],dtype=kbool)
        print(t1)
        print(t2)
        print(t3)
        # [[1. 1. 1.]
        #  [1. 1. 1.]]
        # [2.+0.j 3.+0.j 0.+4.j 5.+0.j]
        # [[[ True  True  True  True]
        #   [ True  True  True  True]]]

    .. py:attribute:: ndim

        Devuelve el número de dimensiones de un tensor.

        :return: El número de dimensiones de un tensor.

        Ejemplo::

            from pyvqnet.tensor import QTensor

            a = QTensor([2.0, 3.0, 4.0, 5.0], requires_grad=True)
            print(a.ndim)

            # 1

    .. py:attribute:: shape

        Devuelve las dimensiones de un tensor

        :return: Una lista de las dimensiones del tensor

        Ejemplo::

            from pyvqnet.tensor import QTensor

            a = QTensor([2.0, 3.0, 4.0, 5.0], requires_grad=True)
            print(a.shape)

            # [4]

    .. py:attribute:: size

        Devuelve el número de elementos de un tensor.

        :return: El número de elementos de un tensor.

        Ejemplo::

            from pyvqnet.tensor import QTensor

            a = QTensor([2.0, 3.0, 4.0, 5.0], requires_grad=True)
            print(a.size)

            # 4

    .. py:method:: numel

        Devuelve el número de elementos en un tensor.

        :return: El número de elementos en un tensor.

        Ejemplo::

            from pyvqnet.tensor import QTensor

            a = QTensor([2.0, 3.0, 4.0, 5.0], requires_grad=True)
            print(a.numel())

            # 4

    .. py:attribute:: dtype

        Devuelve el tipo de dato de un tensor.

        Los tipos de datos soportados son los siguientes:

            =========================================  ===============================
            dtype                                      descripción
            =========================================  ===============================
            ``pyvqnet.kbool``                          Variable booleana
            ``pyvqnet.kuint8``                         8 bits enteros (sin signo)
            ``pyvqnet.kint8``                          8 bits enteros (con signo)
            ``pyvqnet.kint16``                         16 bits enteros (con signo)
            ``pyvqnet.kint32``                         32 bits enteros (con signo)
            ``pyvqnet.kint64``                         64 bits enteros (con signo)
            ``pyvqnet.kfloat32``                       Punto flotante de 32 bits, ver https://en.wikipedia.org/wiki/IEEE_754
            ``pyvqnet.kfloat64``                       Punto flotante de 64 bits, ver https://en.wikipedia.org/wiki/IEEE_754
            ``pyvqnet.kcomplex64``                     Número complejo de 64 bits, compuesto por dos `float32`
            ``pyvqnet.kcomplex128``                    Número complejo de 128 bits, compuesto por dos `float64`
            ``pyvqnet.kbfloat16``                      Punto flotante de 16 bits, a veces llamado formato Brain Floating Point, con asignación de 1 bit de signo, 8 bits de exponente y 7 bits de mantisa
            =========================================  ===============================

        :return: El tipo de dato del tensor.

        Ejemplo::

            from pyvqnet.tensor import QTensor

            a = QTensor([2, 3, 4, 5])
            print(a.dtype)
            # 4


    .. py:method:: zero_grad()

        Establece el gradiente a cero. Será usado por el optimizador en el proceso de optimización.

        :return: None

        Ejemplo::

            from pyvqnet.tensor import tensor
            from pyvqnet.tensor import QTensor
            t3  =  QTensor([2.0,3.0,4.0,5.0],requires_grad = True)
            t3.zero_grad()
            print(t3.grad)

            # [0, 0, 0, 0]


 

    .. py:method:: backward(grad=None)

        Calcula el gradiente del QTensor actual.

        :return: None

        Ejemplo::

            from pyvqnet.tensor import tensor
            from pyvqnet.tensor import QTensor

            target = QTensor([[0, 0, 1, 0, 0, 0, 0, 0, 0, 0.2]], requires_grad=True)
            y = 2*target + 3
            y.backward()
            print(target.grad)
            #[[2. 2. 2. 2. 2. 2. 2. 2. 2. 2.]]

 

    .. py:method:: to_numpy()

        Copia los datos propios a un nuevo numpy.array.

        :return: un nuevo numpy.array que contiene los datos del QTensor

        .. note::

            numpy no soporta el tipo bfloat16, debe convertirlo a otros tipos de datos soportados por numpy, como float32, antes de llamar a esta interfaz.

        Ejemplo::

            from pyvqnet.tensor import tensor
            from pyvqnet.tensor import QTensor
            t3  =  QTensor([2.0,3.0,4.0,5.0],requires_grad = True)
            t4 = t3.to_numpy()
            print(t4)

            # [2. 3. 4. 5.]

 

    .. py:method:: item()

            Devuelve el único elemento del QTensor. Lanza 'RuntimeError' si el QTensor tiene más de 1 elemento.

            :return: único dato de este objeto

            Ejemplo::

                from pyvqnet.tensor import tensor

                t = tensor.ones([1])
                print(t.item())

                # 1.0


    .. py:method:: argmax(*kargs)

        Devuelve los índices del valor máximo de todos los elementos en el QTensor de entrada, o
        Devuelve los índices de los valores máximos de un QTensor a lo largo de una dimensión.

        :param dim: dim (int) - la dimensión a reducir, solo acepta un solo eje. si dim == None, devuelve los índices del valor máximo de todos los elementos en el tensor de entrada. El rango válido de dim es [-R, R), donde R es el ndim de entrada. cuando dim < 0, funciona igual que dim + R.
        :param keepdims: indica si el QTensor de salida mantiene la dimensión o no.

        :return: los índices del valor máximo en el QTensor de entrada.

        Ejemplo::

            from pyvqnet.tensor import tensor
            from pyvqnet.tensor import QTensor
            a = QTensor([[1.3398, 0.2663, -0.2686, 0.2450],
                        [-0.7401, -0.8805, -0.3402, -1.1936],
                        [0.4907, -1.3948, -1.0691, -0.3132],
                        [-1.6092, 0.5419, -0.2993, 0.3195]])
            flag = a.argmax()
            print(flag)
            
            # [0]

            flag_0 = a.argmax([0], True)
            print(flag_0)

            # [
            # [0, 3, 0, 3]
            # ]

            flag_1 = a.argmax([1], True)
            print(flag_1)

            # [
            # [0],
            # [2],
            # [0],
            # [1]
            # ]

 

    .. py:method:: argmin(*kargs)

        Devuelve los índices del valor mínimo de todos los elementos en el QTensor de entrada, o
        Devuelve los índices de los valores mínimos de un QTensor a lo largo de una dimensión.

        :param dim: dim (int) - la dimensión a reducir, solo acepta un solo eje. si dim == None, devuelve los índices del valor mínimo de todos los elementos en el tensor de entrada. El rango válido de dim es [-R, R), donde R es el ndim de entrada. cuando dim < 0, funciona igual que dim + R.
        :param keepdims: indica si el QTensor de salida mantiene la dimensión o no.

        :return: los índices del valor mínimo en el QTensor de entrada.

        Ejemplo::

            from pyvqnet.tensor import tensor
            from pyvqnet.tensor import QTensor
            a = QTensor([[1.3398, 0.2663, -0.2686, 0.2450],
                        [-0.7401, -0.8805, -0.3402, -1.1936],
                        [0.4907, -1.3948, -1.0691, -0.3132],
                        [-1.6092, 0.5419, -0.2993, 0.3195]])
            flag = a.argmin()
            print(flag)

            # [12]

            flag_0 = a.argmin([0], True)
            print(flag_0)

            # [
            # [3, 2, 2, 1]
            # ]

            flag_1 = a.argmin([1], False)
            print(flag_1)

            # [2, 3, 1, 0]



    .. py:method:: fill_(v)

            Llena el QTensor con el valor especificado in situ.

            :param v: un valor escalar
            :return: None

            Ejemplo::

                from pyvqnet.tensor import tensor
                from pyvqnet.tensor import QTensor
                shape = [2, 3]
                value = 42
                t = tensor.zeros(shape)
                t.fill_(value)
                print(t)

                # [
                # [42, 42, 42],
                # [42, 42, 42]
                # ]

    
    .. py:method:: all()

            Devuelve True, si todos los valores del QTensor son distintos de cero.

            :return: True, si todos los valores del QTensor son distintos de cero.

            Ejemplo::

                from pyvqnet.tensor import tensor
                from pyvqnet.tensor import QTensor
                shape = [2, 3]
                t = tensor.zeros(shape)
                t.fill_(1.0)
                flag = t.all()
                print(flag)

                # True

 

    .. py:method:: any()

            Devuelve True, si algún valor del QTensor es distinto de cero.

            :return: True, si algún valor del QTensor es distinto de cero.

            Ejemplo::

                from pyvqnet.tensor import tensor
                from pyvqnet.tensor import QTensor

                shape = [2, 3]
                t = tensor.ones(shape)
                t.fill_(1.0)
                flag = t.any()
                print(flag)

                # True

 

    .. py:method:: fill_rand_binary_(v=0.5)

        Llena un QTensor con valores muestreados aleatoriamente de una distribución binomial.

        Si los datos generados aleatoriamente según la distribución binomial son mayores que el umbral de binarización, entonces el número de las posiciones correspondientes del QTensor se establece en 1, de lo contrario en 0.

        :param v: Umbral de binarización
        :return: None

        Ejemplo::

            from pyvqnet.tensor import tensor
            from pyvqnet.tensor import QTensor
            import numpy as np
            a = np.arange(6).reshape(2, 3).astype(np.float32)
            t = QTensor(a)
            t.fill_rand_binary_(2)
            print(t)

            # [
            # [1, 1, 1],
            # [1, 1, 1]
            # ]

 

    .. py:method:: fill_rand_signed_uniform_(v=1)

        Llena un QTensor con valores muestreados aleatoriamente de una distribución uniforme con signo.

        Factor de escala de los valores generados por la distribución uniforme con signo.

        :param v: un valor escalar
        :return: None

        Ejemplo::

            from pyvqnet.tensor import tensor
            from pyvqnet.tensor import QTensor
            import numpy as np
            a = np.arange(6).reshape(2, 3).astype(np.float32)
            t = QTensor(a)
            value = 42

            t.fill_rand_signed_uniform_(value)
            print(t)

            # [
            # [12.8852444, 4.4327269, 4.8489408],
            # [-24.3309803, 26.8036957, 39.4903450]
            # ]

 

    .. py:method:: fill_rand_uniform_(v=1)

        Llena un QTensor con valores muestreados aleatoriamente de una distribución uniforme.

        Factor de escala de los valores generados por la distribución uniforme.

        :param v: un valor escalar
        :return: None

        Ejemplo::

            from pyvqnet.tensor import tensor
            from pyvqnet.tensor import QTensor
            import numpy as np
            a = np.arange(6).reshape(2, 3).astype(np.float32)
            t = QTensor(a)
            value = 42
            t.fill_rand_uniform_(value)
            print(t)

            # [
            # [20.0404720, 14.4064417, 40.2955666],
            # [5.5692234, 26.2520485, 35.3326073]
            # ]



    .. py:method:: fill_rand_normal_(m=0, s=1, fast_math=True)

        Llena un QTensor con valores muestreados aleatoriamente de una distribución normal.
        Media de la distribución normal. Desviación estándar de la distribución normal.
        Indica si se debe usar o no el modo matemático rápido.

        :param m: media de la distribución normal
        :param s: desviación estándar de la distribución normal
        :param fast_math: True si se usa fast-math
        :return: None

        Ejemplo::

            from pyvqnet.tensor import tensor
            from pyvqnet.tensor import QTensor
            import numpy as np
            a = np.arange(6).reshape(2, 3).astype(np.float32)
            t = QTensor(a)
            t.fill_rand_normal_(2, 10, True)
            print(t)

            # [
            # [-10.4446531    4.9158096   2.9204607],
            # [ -7.2682705   8.1267328    6.2758742 ],
            # ]



    .. py:method:: transpose(new_dims=None)

        Invierte o permuta los ejes de un array. Si new_dims = None, invierte las dimensiones.

        :param new_dims: el nuevo orden de las dimensiones (lista de enteros).
        :return: QTensor resultante.

        Ejemplo::

            from pyvqnet.tensor import tensor
            from pyvqnet.tensor import QTensor
            import numpy as np
            R, C = 3, 4
            a = np.arange(R * C).reshape([2, 2, 3]).astype(np.float32)
            t = QTensor(a)
            rlt = t.transpose([2,0,1])
            print(rlt)
            # [
            # [[0, 3],
            #  [6, 9]],
            # [[1, 4],
            #  [7, 10]],
            # [[2, 5],
            #  [8, 11]]
            # ]



    .. py:method:: reshape(new_shape)

        Cambia la forma del tensor, devuelve un nuevo QTensor.

        :param new_shape: la nueva forma (lista de enteros)
        :return: un nuevo QTensor

        Ejemplo::

            from pyvqnet.tensor import tensor
            from pyvqnet.tensor import QTensor
            import numpy as np
            R, C = 3, 4
            a = np.arange(R * C).reshape(R, C).astype(np.float32)
            t = QTensor(a)
            reshape_t = t.reshape([C, R])
            print(reshape_t)
            # [
            # [0, 1, 2],
            # [3, 4, 5],
            # [6, 7, 8],
            # [9, 10, 11]
            # ]

    .. py:method:: reshape_(new_shape)

        Cambia la forma del QTensor actual in situ. Esta interfaz primero intentará transformar sin cambiar los datos de memoria originales. Si falla, los datos actuales se copiarán en la nueva memoria.

        .. warning::

            Se recomienda usar la interfaz reshape. En algunos casos, la ubicación real de la memoria subyacente se copiará en lugar de modificarse in situ.

        :param new_shape: la nueva forma (lista de enteros)
        :return: None

        Ejemplo::

            from pyvqnet.tensor import tensor
            from pyvqnet.tensor import QTensor
            import numpy as np
            R, C = 3, 4
            a = np.arange(R * C).reshape(R, C).astype(np.float32)
            t = QTensor(a)
            t.reshape_([C, R])
            print(t)

            # [
            # [0, 1, 2],
            # [3, 4, 5],
            # [6, 7, 8],
            # [9, 10, 11]
            # ]


    .. py:method:: getdata()

            Obtiene los datos del QTensor como un array NumPy.

            :return: un array NumPy

            Ejemplo::


                from pyvqnet.tensor import tensor
                from pyvqnet.tensor import QTensor

                t = tensor.ones([3, 4])
                a = t.getdata()
                print(a)

                # [[1. 1. 1. 1.]
                #  [1. 1. 1. 1.]
                #  [1. 1. 1. 1.]]

 

    .. py:method:: __getitem__()

            Se soporta el indexado por segmentos de QTensor, o usar QTensor como índice de acceso avanzado. Se devolverá un nuevo QTensor.

            Los parámetros start, stop y step pueden separarse por dos puntos, como start:stop:step, donde start, stop y step pueden omitirse.

            Como QTensor 1-D, el indexado o segmentado solo puede hacerse en un solo eje.

            Como QTensor 2-D y QTensor multidimensional, el indexado o segmentado puede hacerse en múltiples ejes.

            Si usa QTensor como índice para indexado avanzado, consulte numpy para `indexado avanzado <https://docs.scipy.org/doc/numpy-1.10.1/reference/arrays.indexing.html>`_ .

            Si su QTensor como índice es el resultado de una operación lógica, entonces realiza un indexado booleano.

            .. note:: 
                
                Usamos una forma de índice como a[3,4,1], pero la forma a[3][4][1] no está soportada.

            :param item: Un entero o QTensor como índice.

            :return: Un nuevo QTensor.

            Ejemplo::

                from pyvqnet.tensor import tensor, QTensor
                aaa = tensor.arange(1, 61)
                aaa = aaa.reshape([4, 5, 3])
                print(aaa[0:2, 3, :2])
                # [
                # [10, 11],
                #  [25, 26]
                # ]
                print(aaa[3, 4, 1])
                #[59]
                print(aaa[:, 2, :])
                # [
                # [7, 8, 9],
                #  [22, 23, 24],
                #  [37, 38, 39],
                #  [52, 53, 54]
                # ]
                print(aaa[2])
                # [
                # [31, 32, 33],
                #  [34, 35, 36],
                #  [37, 38, 39],
                #  [40, 41, 42],
                #  [43, 44, 45]
                # ]
                print(aaa[0:2, ::3, 2:])
                # [
                # [[3],
                #  [12]],
                # [[18],
                #  [27]]
                # ]
                a = tensor.ones([2, 2])
                b = QTensor([[1, 1], [0, 1]])
                b = b > 0
                c = a[b]
                print(c)
                #[1, 1, 1]
                tt = tensor.arange(1, 56 * 2 * 4 * 4 + 1).reshape([2, 8, 4, 7, 4])
                tt.requires_grad = True
                index_sample1 = tensor.arange(0, 3).reshape([3, 1])
                index_sample2 = QTensor([0, 1, 0, 2, 3, 2, 2, 3, 3]).reshape([3, 3])
                gg = tt[:, index_sample1, 3:, index_sample2, 2:]
                print(gg)
                # [
                # [[[[87, 88]],
                # [[983, 984]]],
                # [[[91, 92]],
                # [[987, 988]]],
                # [[[87, 88]],
                # [[983, 984]]]],
                # [[[[207, 208]],
                # [[1103, 1104]]],
                # [[[211, 212]],
                # [[1107, 1108]]],
                # [[[207, 208]],
                # [[1103, 1104]]]],
                # [[[[319, 320]],
                # [[1215, 1216]]],
                # [[[323, 324]],
                # [[1219, 1220]]],
                # [[[323, 324]],
                # [[1219, 1220]]]]
                # ]

 

    .. py:method:: __setitem__()

        Se soporta el indexado por segmentos de QTensor, o usar QTensor como índice de acceso avanzado. Se devolverá un nuevo QTensor.

        Los parámetros start, stop y step pueden separarse por dos puntos, como start:stop:step, donde start, stop y step pueden omitirse.

        Como QTensor 1-D, el indexado o segmentado solo puede hacerse en un solo eje.

        Como QTensor 2-D y QTensor multidimensional, el indexado o segmentado puede hacerse en múltiples ejes.

        Si usa QTensor como índice para indexado avanzado, consulte numpy para `indexado avanzado <https://docs.scipy.org/doc/numpy-1.10.1/reference/arrays.indexing.html>`_ .

        Si su QTensor como índice es el resultado de una operación lógica, entonces realiza un indexado booleano.

        .. note:: 
            
            Usamos una forma de índice como a[3,4,1], pero la forma a[3][4][1] no está soportada.

        :param item: Un entero o QTensor como índice

        :return: None


        Ejemplo::

            from pyvqnet.tensor import tensor
            aaa = tensor.arange(1, 61)
            aaa = aaa.reshape([4, 5, 3])
            vqnet_a2 = aaa[3, 4, 1]
            aaa[3, 4, 1] = tensor.arange(10001,
                                            10001 + vqnet_a2.size).reshape(vqnet_a2.shape)
            print(aaa)
            # [
            # [[1, 2, 3],
            #  [4, 5, 6],
            #  [7, 8, 9],
            #  [10, 11, 12],
            #  [13, 14, 15]],
            # [[16, 17, 18],
            #  [19, 20, 21],
            #  [22, 23, 24],
            #  [25, 26, 27],
            #  [28, 29, 30]],
            # [[31, 32, 33],
            #  [34, 35, 36],
            #  [37, 38, 39],
            #  [40, 41, 42],
            #  [43, 44, 45]],
            # [[46, 47, 48],
            #  [49, 50, 51],
            #  [52, 53, 54],
            #  [55, 56, 57],
            #  [58, 10001, 60]]
            # ]
            aaa = tensor.arange(1, 61)
            aaa = aaa.reshape([4, 5, 3])
            vqnet_a3 = aaa[:, 2, :]
            aaa[:, 2, :] = tensor.arange(10001,
                                            10001 + vqnet_a3.size).reshape(vqnet_a3.shape)
            print(aaa)
            # [
            # [[1, 2, 3],
            #  [4, 5, 6],
            #  [10001, 10002, 10003],
            #  [10, 11, 12],
            #  [13, 14, 15]],
            # [[16, 17, 18],
            #  [19, 20, 21],
            #  [10004, 10005, 10006],
            #  [25, 26, 27],
            #  [28, 29, 30]],
            # [[31, 32, 33],
            #  [34, 35, 36],
            #  [10007, 10008, 10009],
            #  [40, 41, 42],
            #  [43, 44, 45]],
            # [[46, 47, 48],
            #  [49, 50, 51],
            #  [10010, 10011, 10012],
            #  [55, 56, 57],
            #  [58, 59, 60]]
            # ]
            aaa = tensor.arange(1, 61)
            aaa = aaa.reshape([4, 5, 3])
            vqnet_a4 = aaa[2, :]
            aaa[2, :] = tensor.arange(10001,
                                        10001 + vqnet_a4.size).reshape(vqnet_a4.shape)
            print(aaa)
            # [
            # [[1, 2, 3],
            #  [4, 5, 6],
            #  [7, 8, 9],
            #  [10, 11, 12],
            #  [13, 14, 15]],
            # [[16, 17, 18],
            #  [19, 20, 21],
            #  [22, 23, 24],
            #  [25, 26, 27],
            #  [28, 29, 30]],
            # [[10001, 10002, 10003],
            #  [10004, 10005, 10006],
            #  [10007, 10008, 10009],
            #  [10010, 10011, 10012],
            #  [10013, 10014, 10015]],
            # [[46, 47, 48],
            #  [49, 50, 51],
            #  [52, 53, 54],
            #  [55, 56, 57],
            #  [58, 59, 60]]
            # ]
            aaa = tensor.arange(1, 61)
            aaa = aaa.reshape([4, 5, 3])
            vqnet_a5 = aaa[0:2, ::2, 1:2]
            aaa[0:2, ::2,
                1:2] = tensor.arange(10001,
                                        10001 + vqnet_a5.size).reshape(vqnet_a5.shape)
            print(aaa)
            # [
            # [[1, 10001, 3],
            #  [4, 5, 6],
            #  [7, 10002, 9],
            #  [10, 11, 12],
            #  [13, 10003, 15]],
            # [[16, 10004, 18],
            #  [19, 20, 21],
            #  [22, 10005, 24],
            #  [25, 26, 27],
            #  [28, 10006, 30]],
            # [[31, 32, 33],
            #  [34, 35, 36],
            #  [37, 38, 39],
            #  [40, 41, 42],
            #  [43, 44, 45]],
            # [[46, 47, 48],
            #  [49, 50, 51],
            #  [52, 53, 54],
            #  [55, 56, 57],
            #  [58, 59, 60]]
            # ]
            a = tensor.ones([2, 2])
            b = tensor.QTensor([[1, 1], [0, 1]])
            b = b > 0
            x = tensor.QTensor([1001, 2001, 3001])

            a[b] = x
            print(a)
            # [
            # [1001, 2001],
            #  [1, 3001]
            # ]
 

    .. py:method:: GPU(device: int = DEV_GPU_0)

        Clona QTensor al dispositivo GPU especificado.

        device especifica el dispositivo donde se almacenan los datos internos. Cuando device >= DEV_GPU_0, los datos se almacenan en la GPU.
        Si su computadora tiene múltiples GPUs, puede designar diferentes dispositivos para almacenar datos.
        Por ejemplo, device = DEV_GPU_1, DEV_GPU_2, DEV_GPU_3, ... indica almacenamiento en GPUs con diferentes números de serie.
        
        .. note::
            QTensor no puede realizar cálculos en diferentes GPUs.
            Se generará un error de Cuda si intenta crear un QTensor en una GPU cuyo ID excede el número máximo de GPUs verificadas.

        :param device: El dispositivo que actualmente guarda QTensor, default=DEV_GPU_0,

        device = pyvqnet.DEV_GPU_0, almacenado en la primera GPU, device = DEV_GPU_1,
        almacenado en la segunda GPU, y así sucesivamente.

        :return: Clona QTensor al dispositivo GPU.

        Ejemplos::

            from pyvqnet.tensor import QTensor
            a = QTensor([2])
            b = a.GPU()
            print(b.device)
            #1000

 

    .. py:method:: CPU()

        Clona QTensor al dispositivo CPU específico.

        :return: Clona QTensor al dispositivo CPU.

        Ejemplos::

            from pyvqnet.tensor import QTensor
            a = QTensor([2])
            b = a.CPU()
            print(b.device)
            # 0

 
    .. py:method:: toGPU(device: int = DEV_GPU_0)

        Mueve QTensor al dispositivo GPU especificado.

        device especifica el dispositivo donde se almacenan los datos internos. Cuando device >= DEV_GPU, los datos se almacenan en la GPU.
        Si su computadora tiene múltiples GPUs, puede designar diferentes dispositivos para almacenar datos.
        Por ejemplo, device = DEV_GPU_1, DEV_GPU_2, DEV_GPU_3, ... indica almacenamiento en GPUs con diferentes números de serie.

        .. note::

            QTensor no puede realizar cálculos en diferentes GPUs. Se generará un error de Cuda si intenta crear un QTensor en una GPU cuyo ID excede el número máximo de GPUs verificadas.

        :param device: El dispositivo que actualmente guarda QTensor, default=DEV_GPU_0. device = pyvqnet.DEV_GPU_0, almacenado en la primera GPU, device = DEV_GPU_1, almacenado en la segunda GPU, y así sucesivamente.
        :return: QTensor movido al dispositivo GPU.

        Ejemplos::

            from pyvqnet.tensor import QTensor
            a = QTensor([2])
            a = a.toGPU()
            print(a.device)
            #1000


    
    .. py:method:: toCPU()

        Mueve QTensor a CPU.

        :return: QTensor movido al dispositivo CPU.

        Ejemplos::

            from pyvqnet.tensor import QTensor
            a = QTensor([2])
            b = a.toCPU()
            print(b.device)
            # 0

    
    .. py:method:: isGPU()

        Indica si los datos de este QTensor están almacenados en la memoria del host GPU.

        :return: Indica si los datos de este QTensor están almacenados en la memoria del host GPU.

        Ejemplos::
        
            from pyvqnet.tensor import QTensor
            a = QTensor([2])
            a = a.isGPU()
            print(a)
            # False
 
    .. py:method:: isCPU()

        Indica si los datos de este QTensor están almacenados en la memoria del host CPU.

        :return: Indica si los datos de este QTensor están almacenados en la memoria del host CPU.

        Ejemplos::
        
            from pyvqnet.tensor import QTensor
            a = QTensor([2])
            a = a.isCPU()
            print(a)
            # True


Funciones de Creación
*****************************************************


ones
==============================

.. py:function:: pyvqnet.tensor.ones(shape,device=0,dtype-None)

    Devuelve un tensor de unos con la forma de entrada.

    :param shape: forma de entrada
    :param device: dispositivo en el que almacenar, por defecto 0, CPU.
    :param dtype: El tipo de dato del parámetro, por defecto None, usa el tipo de dato por defecto: kfloat32, que representa un número de punto flotante de 32 bits.
    
    :return: QTensor de salida con la forma de entrada.

    Ejemplo::

        from pyvqnet.tensor import tensor
        from pyvqnet.tensor import QTensor
        x = tensor.ones([2,3])
        print(x)

        # [
        # [1, 1, 1],
        # [1, 1, 1]
        # ]

ones_like
==============================

.. py:function:: pyvqnet.tensor.ones_like(t: pyvqnet.tensor.QTensor,device=0,dtype=None)

    Devuelve un tensor de unos con la misma forma que el QTensor de entrada.

    :param t: QTensor de entrada
    :param device: dispositivo en el que almacenar, por defecto 0, CPU.
    :param dtype: El tipo de dato del parámetro, por defecto None, usa el tipo de dato por defecto: kfloat32, que representa un número de punto flotante de 32 bits.
    
    :return: QTensor de salida


    Ejemplo::

        from pyvqnet.tensor import tensor
        from pyvqnet.tensor import QTensor
        t = QTensor([1, 2, 3])
        x = tensor.ones_like(t)
        print(x)

        # [1, 1, 1]

full
==============================

.. py:function:: pyvqnet.tensor.full(shape, value, device=0, dtype=None)

    Crea un QTensor de la forma especificada y lo llena con el valor.

    :param shape: forma del QTensor a crear
    :param value: valor con el que llenar el QTensor.
    :param device: dispositivo a usar, default = 0, usar dispositivo cpu.
    :param dtype: El tipo de dato del parámetro, por defecto None, usa el tipo de dato por defecto: kfloat32, que representa un número de punto flotante de 32 bits.
    
    :return: QTensor de salida

    Ejemplo::

        from pyvqnet.tensor import tensor
        from pyvqnet.tensor import QTensor
        shape = [2, 3]
        value = 42
        t = tensor.full(shape, value)
        print(t)
        # [
        # [42, 42, 42],
        # [42, 42, 42]
        # ]

full_like
==============================

.. py:function:: pyvqnet.tensor.full_like(t, value, device: int = 0, dtype=None)

    Crea un QTensor de la forma especificada y lo llena con el valor.

    :param t: QTensor de entrada
    :param value: valor con el que llenar el QTensor.
    :param device: dispositivo a usar, default = 0, usar dispositivo cpu.
    :param dtype: El tipo de dato del parámetro, por defecto None, usa el tipo de dato por defecto: kfloat32, que representa un número de punto flotante de 32 bits.
    
    :return: QTensor de salida

    Ejemplo::

        from pyvqnet.tensor import tensor
        from pyvqnet.tensor import QTensor
        a = tensor.randu([3,5])
        value = 42
        t = tensor.full_like(a, value)
        print(t)
        # [
        # [42, 42, 42, 42, 42],
        # [42, 42, 42, 42, 42],
        # [42, 42, 42, 42, 42]
        # ]

zeros
==============================

.. py:function:: pyvqnet.tensor.zeros(shape,device = 0,dtype=None)

    Devuelve un tensor de ceros con la forma de entrada.

    :param shape: forma del tensor
    :param device: dispositivo a usar, default = 0, usar dispositivo cpu
    :param dtype: El tipo de dato del parámetro, por defecto None, usa el tipo de dato por defecto: kfloat32, que representa un número de punto flotante de 32 bits.
    
    :return: QTensor de salida

    Ejemplo::

        from pyvqnet.tensor import tensor
        from pyvqnet.tensor import QTensor
        t = tensor.zeros([2, 3, 4])
        print(t)
        # [
        # [[0, 0, 0, 0],
        #  [0, 0, 0, 0],
        #  [0, 0, 0, 0]],
        # [[0, 0, 0, 0],
        #  [0, 0, 0, 0],
        #  [0, 0, 0, 0]]
        # ]


zeros_like
==============================

.. py:function:: pyvqnet.tensor.zeros_like(t: pyvqnet.tensor.QTensor,device: int = 0,dtype=None))

    Devuelve un tensor de ceros con la misma forma que el QTensor de entrada.

    :param t: QTensor de entrada
    :param device: dispositivo a usar, default = 0, usar dispositivo cpu
    :param dtype: El tipo de dato del parámetro, por defecto None, usa el tipo de dato por defecto: kfloat32, que representa un número de punto flotante de 32 bits.
    
    :return: QTensor de salida

    Ejemplo::

        from pyvqnet.tensor import tensor
        from pyvqnet.tensor import QTensor
        t = QTensor([1, 2, 3])
        x = tensor.zeros_like(t)
        print(x)

        # [0, 0, 0]

arange
==============================

.. py:function:: pyvqnet.tensor.arange(start, end, step=1, device: int = 0,dtype=None, requires_grad=False)

    Crea un QTensor 1D con valores espaciados uniformemente dentro de un intervalo dado.

    :param start: inicio del intervalo
    :param end: fin del intervalo
    :param step: espaciado entre valores
    :param device: dispositivo a usar, default = 0, usar dispositivo cpu
    :param dtype: El tipo de dato del parámetro, por defecto None, usa el tipo de dato por defecto: kfloat32, que representa un número de punto flotante de 32 bits.
    :param requires_grad: indica si se debe rastrear el gradiente del tensor, por defecto False
    :return: QTensor de salida

    Ejemplo::

        from pyvqnet.tensor import tensor
        from pyvqnet.tensor import QTensor
        t = tensor.arange(2, 30,4)
        print(t)

        # [ 2,  6, 10, 14, 18, 22, 26]

linspace
==============================

.. py:function:: pyvqnet.tensor.linspace(start, end, num, device: int = 0,dtype=None, requires_grad= False)

    Crea un QTensor 1D con valores espaciados uniformemente dentro de un intervalo dado.

    :param start: valor inicial
    :param end: valor final
    :param nums: número de muestras a generar
    :param device: dispositivo a usar, default = 0, usar dispositivo cpu
    :param dtype: El tipo de dato del parámetro, por defecto None, usa el tipo de dato por defecto: kfloat32, que representa un número de punto flotante de 32 bits.
    :param requires_grad: indica si se debe rastrear el gradiente del tensor, por defecto False
    :return: QTensor de salida

    Ejemplo::

        from pyvqnet.tensor import tensor
        from pyvqnet.tensor import QTensor
        start, stop, steps = -2.5, 10, 10
        t = tensor.linspace(start, stop, steps)
        print(t)
        #[-2.5000000, -1.1111112, 0.2777777, 1.6666665, 3.0555553, 4.4444442, 5.8333330, 7.2222219, 8.6111107, 10]

logspace
==============================

.. py:function:: pyvqnet.tensor.logspace(start, end, num, base, device: int = 0,dtype=None,  requires_grad)

    Crea un QTensor 1D con valores espaciados uniformemente en una escala logarítmica.

    :param start: ``base ** start`` es el valor inicial
    :param end: ``base ** end`` es el valor final de la secuencia
    :param nums: número de muestras a generar
    :param base: la base del espacio logarítmico
    :param device: dispositivo a usar, default = 0, usar dispositivo cpu
    :param dtype: El tipo de dato del parámetro, por defecto None, usa el tipo de dato por defecto: kfloat32, que representa un número de punto flotante de 32 bits.
    :param requires_grad: indica si se debe rastrear el gradiente del tensor, por defecto False
    :return: QTensor de salida

    Ejemplo::

        from pyvqnet.tensor import tensor
        from pyvqnet.tensor import QTensor
        start, stop, num, base = 0.1, 1.0, 5, 10.0
        t = tensor.logspace(start, stop, num, base)
        print(t)

        # [1.2589254, 2.1134889, 3.5481336, 5.9566211, 10]

eye
==============================

.. py:function:: pyvqnet.tensor.eye(size, offset: int = 0, device=0,dtype=None)

    Crea un QTensor de tamaño size x size con unos en la diagonal y ceros
    en el resto.

    :param size: tamaño del QTensor (cuadrado) a crear
    :param offset: Índice de la diagonal: 0 (por defecto) se refiere a la diagonal principal, un valor positivo se refiere a una diagonal superior, y un valor negativo a una diagonal inferior.
    :param device: dispositivo a usar, default = 0, usar dispositivo cpu
    :param dtype: El tipo de dato del parámetro, por defecto None, usa el tipo de dato por defecto: kfloat32, que representa un número de punto flotante de 32 bits.
    
    :return: QTensor de salida

    Ejemplo::

        from pyvqnet.tensor import tensor
        from pyvqnet.tensor import QTensor
        size = 3
        t = tensor.eye(size)
        print(t)

        # [
        # [1, 0, 0],
        # [0, 1, 0],
        # [0, 0, 1]
        # ]


diagonal
==============================

.. py:function:: pyvqnet.tensor.diagonal(t: QTensor, offset: int = 0, dim1=0, dim2=1)


    Devuelve una vista parcial de :attr:`t` con los elementos de la diagonal añadidos como dimensiones al final de la forma relativas a :attr:`dim1` y :attr:`dim2`.
    :attr:`offset` es el desplazamiento de la diagonal principal.

    :param t: tensor de entrada
    :param offset: desplazamiento (0 significa diagonal principal, valores positivos significan la enésima diagonal por encima de la diagonal principal, valores negativos significan la enésima diagonal por debajo de la diagonal principal)
    :param dim1: primera dimensión para tomar la diagonal. Por defecto: 0.
    :param dim2: segunda dimensión para tomar la diagonal. Por defecto: 1.

    Ejemplo::

        from pyvqnet.tensor import randn,diagonal

        x = randn((2, 5, 4, 2))
        diagonal_elements = diagonal(x, offset=-1, dim1=1, dim2=2)
        print(diagonal_elements)
        # [[[-0.4641751,-0.1410288,-0.1215512, 0.5423283],
        #   [ 0.9556418, 0.0376572, 1.2571657, 0.8268463]],

        #  [[-0.7972266, 0.2080281,-0.1157126,-0.7342224],
        #   [ 1.1039937, 0.4700735, 1.0219841,-0.146358 ]]]


diag
==============================

.. py:function:: pyvqnet.tensor.diag(t, k: int = 0)

    Selecciona elementos diagonales o construye un QTensor diagonal.

    Si la entrada es un QTensor 2-D, devuelve un nuevo tensor 1D que contiene los elementos diagonales seleccionados. Si la entrada es un QTensor 1-D, devuelve un nuevo tensor 2D cuyos elementos diagonales seleccionados son los valores de entrada y el resto son 0.

    :param t: QTensor de entrada
    :param k: desplazamiento (0 para la diagonal principal, positivo para la enésima diagonal por encima de la principal, negativo para la enésima diagonal por debajo de la principal)
    :return: QTensor de salida

    Ejemplo::

        from pyvqnet.tensor import tensor
        from pyvqnet.tensor import QTensor
        import numpy as np
        a = np.arange(16).reshape(4, 4).astype(np.float32)
        t = QTensor(a)
        for k in range(-3, 4):
            u = tensor.diag(t,k=k)
            print(u)
        # [12.]
        # <QTensor [1] DEV_CPU kfloat32>

        # [ 8.,13.]
        # <QTensor [2] DEV_CPU kfloat32>

        # [ 4., 9.,14.]
        # <QTensor [3] DEV_CPU kfloat32>

        # [ 0., 5.,10.,15.]
        # <QTensor [4] DEV_CPU kfloat32>

        # [ 1., 6.,11.]
        # <QTensor [3] DEV_CPU kfloat32>

        # [2.,7.]
        # <QTensor [2] DEV_CPU kfloat32>

        # [3.]
        # <QTensor [1] DEV_CPU kfloat32>

randu
==============================

.. py:function:: pyvqnet.tensor.randu(shape,min=0.0,max=1.0, device: int = 0, dtype=None, requires_grad=False)

    Crea un QTensor con valores aleatorios distribuidos uniformemente.

    :param shape: forma del QTensor a crear
    :param min: valor mínimo de la distribución uniforme, por defecto: 0.
    :param max: valor máximo de la distribución uniforme, por defecto: 1.
    :param device: dispositivo a usar, default = 0, usar dispositivo cpu
    :param dtype: El tipo de dato del parámetro, por defecto None, usa el tipo de dato por defecto: kfloat32, que representa un número de punto flotante de 32 bits.
    :param requires_grad: indica si se debe rastrear el gradiente del tensor, por defecto False
    :return: QTensor de salida


    Ejemplo::

        from pyvqnet.tensor import tensor
        from pyvqnet.tensor import QTensor
        shape = [2, 3]
        t = tensor.randu(shape)
        print(t)

        # [
        # [0.0885886, 0.9570093, 0.8304565],
        # [0.6055251, 0.8721224, 0.1927866]
        # ]

randn
==============================

.. py:function:: pyvqnet.tensor.randn(shape, mean=0.0,std=1.0, device: int = 0, dtype=None, requires_grad=False)

    Crea un QTensor con valores aleatorios distribuidos normalmente.

    :param shape: forma del QTensor a crear
    :param mean: valor medio de la distribución normal, por defecto: 0.
    :param std: valor de varianza estándar de la distribución normal, por defecto: 1.
    :param device: dispositivo a usar, default = 0, usar dispositivo cpu
    :param dtype: El tipo de dato del parámetro, por defecto None, usa el tipo de dato por defecto: kfloat32, que representa un número de punto flotante de 32 bits.
    :param requires_grad: indica si se debe rastrear el gradiente del tensor, por defecto False
    :return: QTensor de salida

    Ejemplo::

        from pyvqnet.tensor import tensor
        from pyvqnet.tensor import QTensor
        shape = [2, 3]
        t = tensor.randn(shape)
        print(t)

        # [
        # [-0.9529880, -0.4947567, -0.6399882],
        # [-0.6987777, -0.0089036, -0.5084590]
        # ]

binomial
==============================
.. py:function:: pyvqnet.tensor.binomial(total_counts, probs)
    
    Crea una distribución binomial parametrizada por :attr:total_count y :attr:probs.

    :param total_counts: Número de ensayos de Bernoulli.
    :param probs: Probabilidades de evento.

    :return:
        QTensor para la distribución binomial.

    Ejemplo::

        import pyvqnet.tensor as tensor

        a = tensor.randu([3,4])
        b = 1000

        c = tensor.binomial(b,a)
        print(c)

        # [[221.,763., 30.,339.],
        # [803.,899.,105.,356.],
        # [550.,688.,828.,493.]]

multinomial
==============================

.. py:function:: pyvqnet.tensor.multinomial(t, num_samples)

    Devuelve un Tensor donde cada fila contiene num_samples muestras indexadas.
    De la distribución de probabilidad multinomial ubicada en la fila correspondiente del tensor de entrada.

    :param t: Distribución de probabilidad de entrada.
    :param num_samples: número de muestras.

    :return:
        índice de muestra de salida

    Ejemplos::

        from pyvqnet import tensor
        weights = tensor.QTensor([0.1,10, 3, 1]) 
        idx = tensor.multinomial(weights,3)
        print(idx)

        from pyvqnet import tensor
        weights = tensor.QTensor([0,10, 3, 2.2,0.0]) 
        idx = tensor.multinomial(weights,3)
        print(idx)

        # [1 0 3]
        # [1 3 2]

triu
==============================

.. py:function:: pyvqnet.tensor.triu(t, diagonal=0)

    Devuelve la matriz triangular superior de t de entrada, con el resto establecido a 0.

    :param t: QTensor de entrada
    :param diagonal: El desplazamiento default = 0. La diagonal principal es 0, positivo es desplazamiento hacia arriba, y negativo es desplazamiento hacia abajo

    :return: QTensor de salida

    Ejemplos::

        from pyvqnet.tensor import tensor
        a = tensor.arange(1.0, 2 * 6 * 5 + 1.0).reshape([2, 6, 5])
        u = tensor.triu(a, 1)
        print(u)
        # [
        # [[0, 2, 3, 4, 5],
        #  [0, 0, 8, 9, 10],
        #  [0, 0, 0, 14, 15],
        #  [0, 0, 0, 0, 20],
        #  [0, 0, 0, 0, 0],
        #  [0, 0, 0, 0, 0]],
        # [[0, 32, 33, 34, 35],
        #  [0, 0, 38, 39, 40],
        #  [0, 0, 0, 44, 45],
        #  [0, 0, 0, 0, 50],
        #  [0, 0, 0, 0, 0],
        #  [0, 0, 0, 0, 0]]
        # ]

tril
==============================

.. py:function:: pyvqnet.tensor.tril(t, diagonal=0)

    Devuelve la matriz triangular inferior de t de entrada, con el resto establecido a 0.

    :param t: QTensor de entrada
    :param diagonal: El desplazamiento default = 0. La diagonal principal es 0, positivo es desplazamiento hacia arriba, y negativo es desplazamiento hacia abajo

    :return: QTensor de salida

    Ejemplos::

        from pyvqnet.tensor import tensor
        a = tensor.arange(1.0, 2 * 6 * 5 + 1.0).reshape([12, 5])
        u = tensor.tril(a, 1)
        print(u)
        # [
        # [1, 2, 0, 0, 0],
        #  [6, 7, 8, 0, 0],
        #  [11, 12, 13, 14, 0],
        #  [16, 17, 18, 19, 20],
        #  [21, 22, 23, 24, 25],
        #  [26, 27, 28, 29, 30],
        #  [31, 32, 33, 34, 35],
        #  [36, 37, 38, 39, 40],
        #  [41, 42, 43, 44, 45],
        #  [46, 47, 48, 49, 50],
        #  [51, 52, 53, 54, 55],
        #  [56, 57, 58, 59, 60]
        # ]


Funciones Matemáticas
*****************************************************


floor
==============================

.. py:function:: pyvqnet.tensor.floor(t)

    Devuelve un nuevo QTensor con la parte entera inferior (floor) de los elementos de entrada, el entero más grande menor o igual que cada elemento.

    :param t: QTensor de entrada
    :return: QTensor de salida

    Ejemplo::

        from pyvqnet.tensor import tensor

        t = tensor.arange(-2.0, 2.0, 0.25)
        u = tensor.floor(t)
        print(u)

        # [-2, -2, -2, -2, -1, -1, -1, -1, 0, 0, 0, 0, 1, 1, 1, 1]

ceil
==============================

.. py:function:: pyvqnet.tensor.ceil(t)

    Devuelve un nuevo QTensor con la parte entera superior (ceil) de los elementos de entrada, el entero más pequeño mayor o igual que cada elemento.

    :param t: QTensor de entrada
    :return: QTensor de salida

    Ejemplo::

        from pyvqnet.tensor import tensor

        t = tensor.arange(-2.0, 2.0, 0.25)
        u = tensor.ceil(t)
        print(u)

        # [-2, -1, -1, -1, -1, -0, -0, -0, 0, 1, 1, 1, 1, 2, 2, 2]

round
==============================

.. py:function:: pyvqnet.tensor.round(t)

    Redondea los valores del QTensor al entero más cercano.

    :param t: QTensor de entrada
    :return: QTensor de salida

    Ejemplo::

        from pyvqnet.tensor import tensor

        t = tensor.arange(-2.0, 2.0, 0.4)
        u = tensor.round(t)
        print(u)

        # [-2, -2, -1, -1, -0, -0, 0, 1, 1, 2]

sort
==============================

.. py:function:: pyvqnet.tensor.sort(t, axis: int, descending=False, stable=True)

    Ordena QTensor a lo largo del eje.

    :param t: QTensor de entrada
    :param axis: eje de ordenación
    :param descending: orden descendente si es desc
    :param stable: Indica si se debe usar ordenación estable o no
    :return: QTensor de salida

    Ejemplo::

        from pyvqnet.tensor import tensor
        from pyvqnet.tensor import QTensor
        import numpy as np
        a = np.random.randint(10, size=24).reshape(3,8).astype(np.float32)
        A = QTensor(a)
        AA = tensor.sort(A,1,False)
        print(AA)

        # [
        # [0, 1, 2, 4, 6, 7, 8, 8],
        # [2, 5, 5, 8, 9, 9, 9, 9],
        # [1, 2, 5, 5, 5, 6, 7, 7]
        # ]

argsort
==============================

.. py:function:: pyvqnet.tensor.argsort(t, axis: int, descending=False, stable=True)

    Devuelve un array de índices de la misma forma que la entrada que indexa los datos a lo largo del eje dado en orden ordenado.

    :param t: QTensor de entrada
    :param axis: eje de ordenación
    :param descending: orden descendente si es desc
    :param stable: Indica si se debe usar ordenación estable o no
    :return: QTensor de salida

    Ejemplo::

        from pyvqnet.tensor import tensor
        from pyvqnet.tensor import QTensor
        import numpy as np
        a = np.random.randint(10, size=24).reshape(3,8).astype(np.float32)
        A = QTensor(a)
        bb = tensor.argsort(A,1,False)
        print(bb)

        # [
        # [4, 0, 1, 7, 5, 3, 2, 6], 
        #  [3, 0, 7, 6, 2, 1, 4, 5],
        #  [4, 7, 5, 0, 2, 1, 3, 6]
        # ]

topK
==============================

.. py:function:: pyvqnet.tensor.topK(t, k, axis=-1, if_descent=True)

    Devuelve los k elementos más grandes del tensor de entrada a lo largo del eje dado.

    Si if_descent es False, devuelve los k elementos más pequeños.

    :param t: QTensor de entrada
    :param k: número de elementos más grandes o más pequeños
    :param axis: eje de ordenación, default = -1, el último eje
    :param if_descent: orden de ordenación, por defecto True

    :return: Un nuevo QTensor

    Ejemplos::

        from pyvqnet.tensor import tensor, QTensor
        x = QTensor([
            24., 13., 15., 4., 3., 8., 11., 3., 6., 15., 24., 13., 15., 3., 3., 8., 7.,
            3., 6., 11.
        ])
        x= x.reshape([2, 5, 1, 2])
        x.requires_grad = True
        y = tensor.topK(x, 3, 1)
        print(y)
        # [
        # [[[24, 15]],
        # [[15, 13]],
        # [[11, 8]]],
        # [[[24, 13]],
        # [[15, 11]],
        # [[7, 8]]]
        # ]

argtopK
==============================

.. py:function:: pyvqnet.tensor.argtopK(t, k, axis=-1, if_descent=True)

    Devuelve el índice de los k elementos más grandes a lo largo del eje dado del tensor de entrada.

    Si if_descent es False, devuelve el índice de los k elementos más pequeños.

    :param t: QTensor de entrada
    :param k: número de elementos más grandes o más pequeños
    :param axis: eje de ordenación, default = -1, el último eje
    :param if_descent: orden de ordenación, por defecto True

    :return: Un nuevo QTensor

    Ejemplos::

        from pyvqnet.tensor import tensor, QTensor
        x = QTensor([
            24., 13., 15., 4., 3., 8., 11., 3., 6., 15., 24., 13., 15., 3., 3., 8., 7.,
            3., 6., 11.
        ])
        x= x.reshape([2, 5, 1, 2])
        x.requires_grad = True
        y = tensor.argtopK(x, 3, 1)
        print(y)
        # [
        # [[[0, 4]],
        # [[1, 0]],
        # [[3, 2]]],
        # [[[0, 0]],
        # [[1, 4]],
        # [[3, 2]]]
        # ]



add
==============================

.. py:function:: pyvqnet.tensor.add(t1: pyvqnet.tensor.QTensor, t2: pyvqnet.tensor.QTensor)

    Suma elemento a elemento dos QTensors, equivalente a t1 + t2.

    :param t1: primer QTensor
    :param t2: segundo QTensor
    :return: QTensor de salida

    Ejemplo::

        from pyvqnet.tensor import tensor
        from pyvqnet.tensor import QTensor
        t1 = QTensor([1, 2, 3])
        t2 = QTensor([4, 5, 6])
        x = tensor.add(t1, t2)
        print(x)

        # [5, 7, 9]

sub
==============================

.. py:function:: pyvqnet.tensor.sub(t1: pyvqnet.tensor.QTensor, t2: pyvqnet.tensor.QTensor)

    Resta elemento a elemento dos QTensors, equivalente a t1 - t2.


    :param t1: primer QTensor
    :param t2: segundo QTensor
    :return: QTensor de salida

    Ejemplo::

        from pyvqnet.tensor import tensor
        from pyvqnet.tensor import QTensor
        t1 = QTensor([1, 2, 3])
        t2 = QTensor([4, 5, 6])
        x = tensor.sub(t1, t2)
        print(x)

        # [-3, -3, -3]

mul
==============================

.. py:function:: pyvqnet.tensor.mul(t1: pyvqnet.tensor.QTensor, t2: pyvqnet.tensor.QTensor)

    Multiplica elemento a elemento dos QTensors, equivalente a t1 * t2.

    :param t1: primer QTensor
    :param t2: segundo QTensor
    :return: QTensor de salida


    Ejemplo::

        from pyvqnet.tensor import tensor
        from pyvqnet.tensor import QTensor
        t1 = QTensor([1, 2, 3])
        t2 = QTensor([4, 5, 6])
        x = tensor.mul(t1, t2)
        print(x)

        # [4, 10, 18]

divide
==============================

.. py:function:: pyvqnet.tensor.divide(t1: pyvqnet.tensor.QTensor, t2: pyvqnet.tensor.QTensor)

    Divide elemento a elemento dos QTensors, equivalente a t1 / t2.


    :param t1: primer QTensor
    :param t2: segundo QTensor
    :return: QTensor de salida


    Ejemplo::

        from pyvqnet.tensor import tensor
        from pyvqnet.tensor import QTensor
        t1 = QTensor([1, 2, 3])
        t2 = QTensor([4, 5, 6])
        x = tensor.divide(t1, t2)
        print(x)

        # [0.2500000, 0.4000000, 0.5000000]

sums
==============================

.. py:function:: pyvqnet.tensor.sums(t: pyvqnet.tensor.QTensor, axis: Optional[int] = None, keepdims=False)

    Suma todos los elementos en QTensor a lo largo del eje dado. Si axis = None, suma todos los elementos en QTensor.

    :param t: QTensor de entrada
    :param axis: eje usado para sumar, por defecto None
    :param keepdims: indica si el tensor de salida mantiene la dimensión o no. Por defecto False
    :return: QTensor de salida


    Ejemplo::

        from pyvqnet.tensor import tensor
        from pyvqnet.tensor import QTensor
        t = QTensor(([1, 2, 3], [4, 5, 6]))
        x = tensor.sums(t)
        print(x)

        # [21]



cumsum
==============================

.. py:function:: pyvqnet.tensor.cumsum(t, axis=-1)

    Devuelve la suma acumulativa de los elementos de entrada en la dimensión del eje.

    :param t: el QTensor de entrada
    :param axis: Cálculo del eje, por defecto -1, usar el último eje

    :return: QTensor de salida.

    Ejemplo::

       from pyvqnet.tensor import tensor, QTensor
       t = QTensor(([1, 2, 3], [4, 5, 6]))
       x = tensor.cumsum(t,-1)
       print(x)
       # [
       # [1, 3, 6],
       # [4, 9, 15]
       # ]


mean
==============================

.. py:function:: pyvqnet.tensor.mean(t: pyvqnet.tensor.QTensor, axis=None, keepdims=False)

    Obtiene los valores medios en el QTensor a lo largo del eje.

    :param t: el QTensor de entrada.
    :param axis: la dimensión a reducir.
    :param keepdims: indica si el QTensor de salida mantiene la dimensión o no, por defecto False.
    :return: devuelve el valor medio del QTensor de entrada.

    Ejemplo::

        from pyvqnet.tensor import tensor
        from pyvqnet.tensor import QTensor
        t = QTensor([[1, 2, 3], [4, 5, 6.0]])
        x = tensor.mean(t, axis=1)
        print(x)

        # [2., 5.]

median
==============================

.. py:function:: pyvqnet.tensor.median(t: pyvqnet.tensor.QTensor, axis=None, keepdims=False)

    Obtiene el valor mediano en el QTensor.

    :param t: el QTensor de entrada
    :param axis: Un eje para promediar, por defecto None
    :param keepdims: indica si el QTensor de salida mantiene la dimensión o no, por defecto False

    :return: Devuelve la mediana de los valores en la entrada o QTensor.

    Ejemplo::

        from pyvqnet.tensor import tensor
        from pyvqnet.tensor import QTensor

        a = QTensor([[1.5219, -1.5212,  0.2202]])
        median_a = tensor.median(a)
        print(median_a)

        # [0.2202000]

        b = QTensor([[0.2505, -0.3982, -0.9948,  0.3518, -1.3131],
                    [0.3180, -0.6993,  1.0436,  0.0438,  0.2270],
                    [-0.2751,  0.7303,  0.2192,  0.3321,  0.2488],
                    [1.0778, -1.9510,  0.7048,  0.4742, -0.7125]])
        median_b = tensor.median(b,1, False)
        print(median_b)

        # [-0.3982000, 0.2270000, 0.2488000, 0.4742000]

std
==============================

.. py:function:: pyvqnet.tensor.std(t: pyvqnet.tensor.QTensor, axis=None, keepdims=False, unbiased=True)

    Obtiene el valor de varianza estándar en el QTensor.


    :param t: el QTensor de entrada
    :param axis: el eje usado para calcular la desviación estándar, por defecto None
    :param keepdims: indica si el QTensor de salida mantiene la dimensión o no, por defecto False
    :param unbiased: indica si se debe usar la corrección de Bessel, por defecto true
    :return: Devuelve la varianza estándar de los valores en la entrada o QTensor

    Ejemplo::

        from pyvqnet.tensor import tensor
        from pyvqnet.tensor import QTensor

        a = QTensor([[-0.8166, -1.3802, -0.3560]])
        std_a = tensor.std(a)
        print(std_a)

        # [0.5129624]

        b = QTensor([[0.2505, -0.3982, -0.9948,  0.3518, -1.3131],
                    [0.3180, -0.6993,  1.0436,  0.0438,  0.2270],
                    [-0.2751,  0.7303,  0.2192,  0.3321,  0.2488],
                    [1.0778, -1.9510,  0.7048,  0.4742, -0.7125]])
        std_b = tensor.std(b, 1, False, False)
        print(std_b)

        # [0.6593542, 0.5583112, 0.3206565, 1.1103367]

var
==============================

.. py:function:: pyvqnet.tensor.var(t: pyvqnet.tensor.QTensor, axis=None, keepdims=False, unbiased=True)

    Obtiene la varianza en el QTensor.


    :param t: el QTensor de entrada.
    :param axis: El eje usado para calcular la varianza, por defecto None
    :param keepdims: indica si el QTensor de salida mantiene la dimensión o no, por defecto False.
    :param unbiased: indica si se debe usar la corrección de Bessel, por defecto true.


    :return: Obtiene la varianza en el QTensor.

    Ejemplo::

        from pyvqnet.tensor import tensor
        from pyvqnet.tensor import QTensor

        a = QTensor([[-0.8166, -1.3802, -0.3560]])
        a_var = tensor.var(a)
        print(a_var)

        # [0.2631305]

matmul
==============================

.. py:function:: pyvqnet.tensor.matmul(t1: pyvqnet.tensor.QTensor, t2: pyvqnet.tensor.QTensor)

    Multiplicación de matrices de matrices 2d, 3d y 4d.

    :param t1: primer QTensor
    :param t2: segundo QTensor
    :return: QTensor de salida

    Ejemplo::

        from pyvqnet.tensor import tensor
        from pyvqnet.tensor import QTensor
        t1 = tensor.ones([2,3])
        t1.requires_grad = True
        t2 = tensor.ones([3,4])
        t2.requires_grad = True
        t3  = tensor.matmul(t1,t2)
        t3.backward(tensor.ones_like(t3))
        print(t1.grad)

        # [
        # [4, 4, 4],
        #  [4, 4, 4]
        # ]

        print(t2.grad)

        # [
        # [2, 2, 2, 2],
        #  [2, 2, 2, 2],
        #  [2, 2, 2, 2]
        # ]kron

=============================

.. py:function:: pyvqnet.tensor.kron(t1: pyvqnet.tensor.QTensor, t2: pyvqnet.tensor.QTensor)

    Calcula el producto de Kronecker de ``t1`` y ``t2``, expresado como :math:`\otimes` . Si ``t1`` es un tensor :math:`(a_0 \times a_1 \times \dots \times a_n)` y ``t2`` es un tensor :math:`(b_0 \times b_1 \times \dots \ times b_n)`, el resultado será un tensor :math:`(a_0*b_0 \times a_1*b_1 \times \dots \times a_n*b_n)` con las siguientes entradas:

    .. math::
          (\text{input} \otimes \text{other})_{k_0, k_1, \dots, k_n} =
              \text{input}_{i_0, i_1, \dots, i_n} * \text{other}_{j_0, j_1, \dots, j_n},

    donde :math:`k_t = i_t * b_t + j_t` es :math:`0 \leq t \leq n`.
    Si un tensor tiene menos dimensiones que el otro, se desempaquetará hasta que tenga la misma dimensionalidad.

    :param t1: El primer QTensor.
    :param t2: El segundo QTensor.

    :return: QTensor de salida.

    Ejemplo::

        from pyvqnet import tensor
        a = tensor.arange(1,1+ 24).reshape([2,1,2,3,2])
        b = tensor.arange(1,1+ 24).reshape([6,4])

        c = tensor.kron(a,b)
        print(c)


        # [[[[[  1.   2.   3.   4.   2.   4.   6.   8.]
        #     [  5.   6.   7.   8.  10.  12.  14.  16.]
        #     [  9.  10.  11.  12.  18.  20.  22.  24.]
        #     [ 13.  14.  15.  16.  26.  28.  30.  32.]
        #     [ 17.  18.  19.  20.  34.  36.  38.  40.]
        #     [ 21.  22.  23.  24.  42.  44.  46.  48.]
        #     [  3.   6.   9.  12.   4.   8.  12.  16.]
        #     [ 15.  18.  21.  24.  20.  24.  28.  32.]
        #     [ 27.  30.  33.  36.  36.  40.  44.  48.]
        #     [ 39.  42.  45.  48.  52.  56.  60.  64.]
        #     [ 51.  54.  57.  60.  68.  72.  76.  80.]
        #     [ 63.  66.  69.  72.  84.  88.  92.  96.]
        #     [  5.  10.  15.  20.   6.  12.  18.  24.]
        #     [ 25.  30.  35.  40.  30.  36.  42.  48.]
        #     [ 45.  50.  55.  60.  54.  60.  66.  72.]
        #     [ 65.  70.  75.  80.  78.  84.  90.  96.]
        #     [ 85.  90.  95. 100. 102. 108. 114. 120.]
        #     [105. 110. 115. 120. 126. 132. 138. 144.]]

        #    [[  7.  14.  21.  28.   8.  16.  24.  32.]
        #     [ 35.  42.  49.  56.  40.  48.  56.  64.]
        #     [ 63.  70.  77.  84.  72.  80.  88.  96.]
        #     [ 91.  98. 105. 112. 104. 112. 120. 128.]
        #     [119. 126. 133. 140. 136. 144. 152. 160.]
        #     [147. 154. 161. 168. 168. 176. 184. 192.]
        #     [  9.  18.  27.  36.  10.  20.  30.  40.]
        #     [ 45.  54.  63.  72.  50.  60.  70.  80.]
        #     [ 81.  90.  99. 108.  90. 100. 110. 120.]
        #     [117. 126. 135. 144. 130. 140. 150. 160.]
        #     [153. 162. 171. 180. 170. 180. 190. 200.]
        #     [189. 198. 207. 216. 210. 220. 230. 240.]
        #     [ 11.  22.  33.  44.  12.  24.  36.  48.]
        #     [ 55.  66.  77.  88.  60.  72.  84.  96.]
        #     [ 99. 110. 121. 132. 108. 120. 132. 144.]
        #     [143. 154. 165. 176. 156. 168. 180. 192.]
        #     [187. 198. 209. 220. 204. 216. 228. 240.]
        #     [231. 242. 253. 264. 252. 264. 276. 288.]]]



        #  [[[[ 13.  26.  39.  52.  14.  28.  42.  56.]
        #     [ 65.  78.  91. 104.  70.  84.  98. 112.]
        #     [117. 130. 143. 156. 126. 140. 154. 168.]
        #     [169. 182. 195. 208. 182. 196. 210. 224.]
        #     [221. 234. 247. 260. 238. 252. 266. 280.]
        #     [273. 286. 299. 312. 294. 308. 322. 336.]
        #     [ 15.  30.  45.  60.  16.  32.  48.  64.]
        #     [ 75.  90. 105. 120.  80.  96. 112. 128.]
        #     [135. 150. 165. 180. 144. 160. 176. 192.]
        #     [195. 210. 225. 240. 208. 224. 240. 256.]
        #     [255. 270. 285. 300. 272. 288. 304. 320.]
        #     [315. 330. 345. 360. 336. 352. 368. 384.]
        #     [ 17.  34.  51.  68.  18.  36.  54.  72.]
        #     [ 85. 102. 119. 136.  90. 108. 126. 144.]
        #     [153. 170. 187. 204. 162. 180. 198. 216.]
        #     [221. 238. 255. 272. 234. 252. 270. 288.]
        #     [289. 306. 323. 340. 306. 324. 342. 360.]
        #     [357. 374. 391. 408. 378. 396. 414. 432.]]

        #    [[ 19.  38.  57.  76.  20.  40.  60.  80.]
        #     [ 95. 114. 133. 152. 100. 120. 140. 160.]
        #     [171. 190. 209. 228. 180. 200. 220. 240.]
        #     [247. 266. 285. 304. 260. 280. 300. 320.]
        #     [323. 342. 361. 380. 340. 360. 380. 400.]
        #     [399. 418. 437. 456. 420. 440. 460. 480.]
        #     [ 21.  42.  63.  84.  22.  44.  66.  88.]
        #     [105. 126. 147. 168. 110. 132. 154. 176.]
        #     [189. 210. 231. 252. 198. 220. 242. 264.]
        #     [273. 294. 315. 336. 286. 308. 330. 352.]
        #     [357. 378. 399. 420. 374. 396. 418. 440.]
        #     [441. 462. 483. 504. 462. 484. 506. 528.]
        #     [ 23.  46.  69.  92.  24.  48.  72.  96.]
        #     [115. 138. 161. 184. 120. 144. 168. 192.]
        #     [207. 230. 253. 276. 216. 240. 264. 288.]
        #     [299. 322. 345. 368. 312. 336. 360. 384.]
        #     [391. 414. 437. 460. 408. 432. 456. 480.]
        #     [483. 506. 529. 552. 504. 528. 552. 576.]]]]]


einsum
==============================

.. py:function:: pyvqnet.tensor.einsum(equation, *operands)

    Suma los productos de los elementos de los operandos de entrada a lo largo de la dimensión especificada usando una notación basada en la convención de suma de Einstein.

    .. note::

        Esta función usa opt_einsum (https://optimized-einsum.readthedocs.io/en/stable/) para acelerar el cálculo o reducir el consumo de memoria optimizando el orden de contracción. Esta optimización ocurre cuando hay al menos tres entradas.

        Para operaciones `einsum` más complejas, se puede importar opt_einsum adicionalmente para calcular directamente en QTensor.

    :param equation: El subíndice de la suma de Einstein.
    :param operands: El tensor sobre el cual se calculará la suma de Einstein.

    :return:

        El resultado QTensor.

    Ejemplo::

        from pyvqnet import tensor

        vqneta = tensor.randn((3, 5, 4))
        vqnetl = tensor.randn((2, 5))
        vqnetr = tensor.randn((2, 4))
        z = tensor.einsum('bn,anm,bm->ba',  vqnetl, vqneta,vqnetr)
        print(z.shape)
        #[2, 3]
        vqneta = tensor.randn((20,30,40,50))
        z = tensor.einsum('...ij->...ji', vqneta)
        print(z.shape)
        #[20, 30, 50, 40]

reciprocal
==============================

.. py:function:: pyvqnet.tensor.reciprocal(t)

    Calcula el recíproco elemento a elemento del QTensor.

    :param t: QTensor de entrada
    :return: QTensor de salida

    Ejemplo::

        from pyvqnet.tensor import tensor
        from pyvqnet.tensor import QTensor

        t = tensor.arange(1, 10, 1)
        u = tensor.reciprocal(t)
        print(u)

        #[1, 0.5000000, 0.3333333, 0.2500000, 0.2000000, 0.1666667, 0.1428571, 0.1250000, 0.1111111]

sign
==============================

.. py:function:: pyvqnet.tensor.sign(t)

    Devuelve un nuevo QTensor con los signos de los elementos de entrada. La función signo devuelve -1 si t < 0, 0 si t==0, 1 si t > 0.

    :param t: QTensor de entrada
    :return: QTensor de salida


    Ejemplo::

        from pyvqnet.tensor import tensor
        from pyvqnet.tensor import QTensor

        t = tensor.arange(-5, 5, 1)
        u = tensor.sign(t)
        print(u)

        # [-1, -1, -1, -1, -1, 0, 1, 1, 1, 1]


neg
==============================

.. py:function:: pyvqnet.tensor.neg(t: pyvqnet.tensor.QTensor)

    Negación unaria de elementos de QTensor.

    :param t: QTensor de entrada
    :return: QTensor de salida

    Ejemplo::

        from pyvqnet.tensor import tensor
        from pyvqnet.tensor import QTensor
        t = QTensor([1, 2, 3])
        x = tensor.neg(t)
        print(x)

        # [-1, -2, -3]

trace
==============================

.. py:function:: pyvqnet.tensor.trace(t, k: int = 0)

    Devuelve la suma de los elementos de la diagonal de la matriz 2-D de entrada.

    :param t: QTensor 2-D de entrada
    :param k: desplazamiento (0 para la diagonal principal, positivo para la enésima
        diagonal por encima de la principal, negativo para la enésima diagonal por debajo de la
        principal)
    :return: la suma de los elementos de la diagonal de la matriz 2-D de entrada

    Ejemplo::

        from pyvqnet.tensor import tensor
        from pyvqnet.tensor import QTensor

        t = tensor.randn([4,4])
        for k in range(-3, 4):
            u=tensor.trace(t,k=k)
            print(u)

        # 0.07717618346214294
        # -1.9287869930267334
        # 0.6111435890197754
        # 2.8094992637634277
        # 0.6388946771621704
        # -1.3400784730911255
        # 0.26980453729629517

exp
==============================

.. py:function:: pyvqnet.tensor.exp(t: pyvqnet.tensor.QTensor)

    Aplica la función exponencial a todos los elementos del QTensor de entrada.

    :param t: QTensor de entrada
    :return: QTensor de salida

    Ejemplo::

        from pyvqnet.tensor import tensor
        from pyvqnet.tensor import QTensor
        t = QTensor([1, 2, 3])
        x = tensor.exp(t)
        print(x)

        # [2.7182817, 7.3890562, 20.0855369]

acos
==============================

.. py:function:: pyvqnet.tensor.acos(t: pyvqnet.tensor.QTensor)

    Calcula el coseno inverso elemento a elemento del QTensor.

    :param t: QTensor de entrada
    :return: QTensor de salida

    Ejemplo::

        from pyvqnet.tensor import tensor
        from pyvqnet.tensor import QTensor
        import numpy as np
        a = np.arange(36).reshape(2,6,3).astype(np.float32)
        a =a/100
        A = QTensor(a,requires_grad = True)
        y = tensor.acos(A)
        print(y)

        # [
        # [[1.5707964, 1.5607961, 1.5507950],
        #  [1.5407919, 1.5307857, 1.5207754],
        #  [1.5107603, 1.5007390, 1.4907107],
        #  [1.4806744, 1.4706289, 1.4605733],
        #  [1.4505064, 1.4404273, 1.4303349],
        #  [1.4202280, 1.4101057, 1.3999666]],
        # [[1.3898098, 1.3796341, 1.3694384],
        #  [1.3592213, 1.3489819, 1.3387187],
        #  [1.3284305, 1.3181161, 1.3077742],
        #  [1.2974033, 1.2870022, 1.2765695],
        #  [1.2661036, 1.2556033, 1.2450669],
        #  [1.2344928, 1.2238795, 1.2132252]]
        # ]

asin
==============================

.. py:function:: pyvqnet.tensor.asin(t: pyvqnet.tensor.QTensor)

    Calcula el seno inverso elemento a elemento del QTensor.

    :param t: QTensor de entrada
    :return: QTensor de salida

    Ejemplo::

        from pyvqnet.tensor import tensor
        from pyvqnet.tensor import QTensor

        t = tensor.arange(-1, 1, .5)
        u = tensor.asin(t)
        print(u)

        #[-1.5707964, -0.5235988, 0, 0.5235988]

atan
==============================

.. py:function:: pyvqnet.tensor.atan(t: pyvqnet.tensor.QTensor)

    Calcula la tangente inversa elemento a elemento del QTensor.

    :param t: QTensor de entrada
    :return: QTensor de salida

    Ejemplo::

        from pyvqnet.tensor import tensor
        from pyvqnet.tensor import QTensor

        t = tensor.arange(-1, 1, .5)
        u = tensor.atan(t)
        print(u)

        # [-0.7853981, -0.4636476, 0.0000, 0.4636476]

sin
==============================

.. py:function:: pyvqnet.tensor.sin(t: pyvqnet.tensor.QTensor)

    Aplica la función seno a todos los elementos del QTensor de entrada.


    :param t: QTensor de entrada
    :return: QTensor de salida

    Ejemplo::

        from pyvqnet.tensor import tensor
        from pyvqnet.tensor import QTensor
        t = QTensor([1, 2, 3])
        x = tensor.sin(t)
        print(x)

        # [0.8414709, 0.9092974, 0.1411200]

cos
==============================

.. py:function:: pyvqnet.tensor.cos(t: pyvqnet.tensor.QTensor)

    Aplica la función coseno a todos los elementos del QTensor de entrada.


    :param t: QTensor de entrada
    :return: QTensor de salida

    Ejemplo::

        from pyvqnet.tensor import tensor
        from pyvqnet.tensor import QTensor
        t = QTensor([1, 2, 3])
        x = tensor.cos(t)
        print(x)

        # [0.5403022, -0.4161468, -0.9899924]

tan 
==============================

.. py:function:: pyvqnet.tensor.tan(t: pyvqnet.tensor.QTensor)

    Aplica la función tangente a todos los elementos del QTensor de entrada.


    :param t: QTensor de entrada
    :return: QTensor de salida

    Ejemplo::

        from pyvqnet.tensor import tensor
        from pyvqnet.tensor import QTensor
        t = QTensor([1, 2, 3])
        x = tensor.tan(t)
        print(x)

        # [1.5574077, -2.1850397, -0.1425465]

tanh
==============================

.. py:function:: pyvqnet.tensor.tanh(t: pyvqnet.tensor.QTensor)

    Aplica la función tangente hiperbólica a todos los elementos del QTensor de entrada.

    :param t: QTensor de entrada
    :return: QTensor de salida

    Ejemplo::

        from pyvqnet.tensor import tensor
        from pyvqnet.tensor import QTensor
        t = QTensor([1, 2, 3])
        x = tensor.tanh(t)
        print(x)

        # [0.7615941, 0.9640275, 0.9950547]

sinh
==============================

.. py:function:: pyvqnet.tensor.sinh(t: pyvqnet.tensor.QTensor)

    Aplica la función seno hiperbólico a todos los elementos del QTensor de entrada.


    :param t: QTensor de entrada
    :return: QTensor de salida

    Ejemplo::

        from pyvqnet.tensor import tensor
        from pyvqnet.tensor import QTensor
        t = QTensor([1, 2, 3])
        x = tensor.sinh(t)
        print(x)

        # [1.1752011, 3.6268603, 10.0178747]

cosh
==============================

.. py:function:: pyvqnet.tensor.cosh(t: pyvqnet.tensor.QTensor)

    Aplica la función coseno hiperbólico a todos los elementos del QTensor de entrada.


    :param t: QTensor de entrada
    :return: QTensor de salida

    Ejemplo::

        from pyvqnet.tensor import tensor
        from pyvqnet.tensor import QTensor
        t = QTensor([1, 2, 3])
        x = tensor.cosh(t)
        print(x)

        # [1.5430806, 3.7621955, 10.0676622]

power
==============================

.. py:function:: pyvqnet.tensor.power(t1: pyvqnet.tensor.QTensor, t2: pyvqnet.tensor.QTensor)

    Eleva el primer QTensor a la potencia del segundo QTensor.

    :param t1: primer QTensor
    :param t2: segundo QTensor
    :return: QTensor de salida

    Ejemplo::

        from pyvqnet.tensor import tensor
        from pyvqnet.tensor import QTensor
        t1 = QTensor([1, 4, 3])
        t2 = QTensor([2, 5, 6])
        x = tensor.power(t1, t2)
        print(x)

        # [1, 1024, 729]

abs
==============================

.. py:function:: pyvqnet.tensor.abs(t: pyvqnet.tensor.QTensor)

    Aplica la función valor absoluto a todos los elementos del QTensor de entrada.

    :param t: QTensor de entrada
    :return: QTensor de salida

    Ejemplo::

        from pyvqnet.tensor import tensor
        from pyvqnet.tensor import QTensor
        t = QTensor([1, -2, 3])
        x = tensor.abs(t)
        print(x)

        # [1, 2, 3]

log
==============================

.. py:function:: pyvqnet.tensor.log(t: pyvqnet.tensor.QTensor)

    Aplica la función log (ln) a todos los elementos del QTensor de entrada.

    :param t: QTensor de entrada
    :return: QTensor de salida

    Ejemplo::

        from pyvqnet.tensor import tensor
        from pyvqnet.tensor import QTensor
        t = QTensor([1, 2, 3])
        x = tensor.log(t)
        print(x)

        # [0, 0.6931471, 1.0986123]

log_softmax
==============================

.. py:function:: pyvqnet.tensor.log_softmax(t, axis=-1)

    Calcula secuencialmente los resultados de la función softmax y la función log en el eje axis.

    :param t: QTensor de entrada.
    :param axis: El eje usado para calcular softmax, el valor por defecto es -1.

    :return: QTensor de salida.

    Ejemplo::

        from pyvqnet import tensor
        output = tensor.arange(1,13).reshape([3,2,2])
        t = tensor.log_softmax(output,1)
        print(t)
        # [
        # [[-2.1269281, -2.1269281],
        #  [-0.1269280, -0.1269280]],
        # [[-2.1269281, -2.1269281],
        #  [-0.1269280, -0.1269280]],
        # [[-2.1269281, -2.1269281],
        #  [-0.1269280, -0.1269280]]
        # ]

sqrt
==============================

.. py:function:: pyvqnet.tensor.sqrt(t: pyvqnet.tensor.QTensor)

    Aplica la función raíz cuadrada a todos los elementos del QTensor de entrada.


    :param t: QTensor de entrada
    :return: QTensor de salida

    Ejemplo::

        from pyvqnet.tensor import tensor
        from pyvqnet.tensor import QTensor
        t = QTensor([1, 2, 3])
        x = tensor.sqrt(t)
        print(x)

        # [1, 1.4142135, 1.7320507]

square
==============================

.. py:function:: pyvqnet.tensor.square(t: pyvqnet.tensor.QTensor)

    Aplica la función cuadrado a todos los elementos del QTensor de entrada.


    :param t: QTensor de entrada
    :return: QTensor de salida

    Ejemplo::

        from pyvqnet.tensor import tensor
        from pyvqnet.tensor import QTensor
        t = QTensor([1, 2, 3])
        x = tensor.square(t)
        print(x)
        # [1, 4, 9]



eigh
==============================

.. py:function:: pyvqnet.tensor.eigh(t: QTensor)
 
    Devuelve los valores propios y vectores propios de una matriz compleja hermítica (simétrica conjugada) o real simétrica.

    Devuelve dos objetos, un array 1D que contiene los valores propios de a,
    y una matriz cuadrada 2D o matriz (dependiendo del tipo de entrada) de los vectores propios correspondientes (en columnas).

    :param: QTensor de entrada.
    :param: Valores propios y vectores propios de t.
    :return:

        Devuelve valores propios y vectores propios

    Ejemplos::

        import numpy as np
        import pyvqnet
        from pyvqnet import tensor


        def generate_random_symmetric_matrix(n):
                A = pyvqnet.tensor.randn((n, n))
                A = A + A.transpose()
                return A

        n = 3
        symmetric_matrix = generate_random_symmetric_matrix(n)

        evs,vecs = pyvqnet.tensor.eigh(symmetric_matrix)
        print(evs)
        print(vecs)
        # [-4.0669565,-1.9191254,-1.3642329]
        # <QTensor [3] DEV_CPU kfloat32>

        # [[-0.9889652, 0.0325959,-0.1445187],
        #  [ 0.0912495, 0.9025176,-0.4208745],
        #  [ 0.1167119,-0.4294176,-0.8955328]]
        # <QTensor [3, 3] DEV_CPU kfloat32>

frobenius_norm
==============================

.. py:function:: pyvqnet.tensor.frobenius_norm(t: QTensor, axis: int = None, keepdims=False)

    Calcula la norma F del tensor en el QTensor de entrada a lo largo del eje establecido por axis,
    si axis es None, devuelve la norma F de todos los elementos.

    :param t: QTensor de entrada.
    :param axis: El eje usado para encontrar la norma F, el valor por defecto es None.
    :param keepdims: Indica si el tensor de salida preserva la dimensionalidad reducida. El valor por defecto es False.
    :return: Devuelve un QTensor o valor de norma F.


    Ejemplo::

        from pyvqnet import tensor,QTensor
        t = QTensor([[[1., 2., 3.], [4., 5., 6.]], [[7., 8., 9.], [10., 11., 12.]],
                    [[13., 14., 15.], [16., 17., 18.]]])
        t.requires_grad = True
        result = tensor.frobenius_norm(t, -2, False)
        print(result)
        # [
        # [4.1231055, 5.3851647, 6.7082038],
        #  [12.2065554, 13.6014709, 15],
        #  [20.6155281, 22.0227146, 23.4307499]
        # ]



Funciones Lógicas
**************************

maximum
==============================

.. py:function:: pyvqnet.tensor.maximum(t1: pyvqnet.tensor.QTensor, t2: pyvqnet.tensor.QTensor)

    Máximo elemento a elemento de dos tensores.


    :param t1: primer QTensor
    :param t2: segundo QTensor
    :return: QTensor de salida

    Ejemplo::

        from pyvqnet.tensor import tensor
        from pyvqnet.tensor import QTensor
        t1 = QTensor([6, 4, 3])
        t2 = QTensor([2, 5, 7])
        x = tensor.maximum(t1, t2)
        print(x)

        # [6, 5, 7]

minimum
==============================

.. py:function:: pyvqnet.tensor.minimum(t1: pyvqnet.tensor.QTensor, t2: pyvqnet.tensor.QTensor)

    Mínimo elemento a elemento de dos tensores.


    :param t1: primer QTensor
    :param t2: segundo QTensor
    :return: QTensor de salida

    Ejemplo::

        from pyvqnet.tensor import tensor
        from pyvqnet.tensor import QTensor
        t1 = QTensor([6, 4, 3])
        t2 = QTensor([2, 5, 7])
        x = tensor.minimum(t1, t2)
        print(x)

        # [2, 4, 3]

min
==============================

.. py:function:: pyvqnet.tensor.min(t: pyvqnet.tensor.QTensor, axis=None, keepdims=False)

    Devuelve los elementos mínimos del QTensor de entrada a lo largo del eje dado.
    Si axis == None, devuelve el valor mínimo de todos los elementos en el tensor.

    :param t: QTensor de entrada
    :param axis: eje usado para el mínimo, por defecto None
    :param keepdims: indica si el tensor de salida mantiene la dimensión o no. Por defecto False
    :return: QTensor de salida

    Ejemplo::

        from pyvqnet.tensor import tensor
        from pyvqnet.tensor import QTensor
        t = QTensor([[1, 2, 3], [4, 5, 6]])
        x = tensor.min(t, axis=1, keepdims=True)
        print(x)

        # [
        # [1],
        #  [4]
        # ]

max
==============================

.. py:function:: pyvqnet.tensor.max(t: pyvqnet.tensor.QTensor, axis=None, keepdims=False)

    Devuelve los elementos máximos del QTensor de entrada a lo largo del eje dado.
    Si axis == None, devuelve el valor máximo de todos los elementos en el tensor.

    :param t: QTensor de entrada
    :param axis: eje usado para el máximo, por defecto None
    :param keepdims: indica si el tensor de salida mantiene la dimensión o no. Por defecto False
    :return: QTensor de salida

    Ejemplo::

        from pyvqnet.tensor import tensor
        from pyvqnet.tensor import QTensor
        t = QTensor([[1, 2, 3], [4, 5, 6]])
        x = tensor.max(t, axis=1, keepdims=True)
        print(x)

        # [[3],
        # [6]]

clip
==============================

.. py:function:: pyvqnet.tensor.clip(t: pyvqnet.tensor.QTensor, min_val, max_val)

    Recorta el QTensor de entrada a los valores mínimo y máximo.

    :param t: QTensor de entrada
    :param min_val: valor mínimo
    :param max_val: valor máximo
    :return: QTensor de salida

    Ejemplo::

        from pyvqnet.tensor import tensor
        from pyvqnet.tensor import QTensor
        t = QTensor([2, 4, 6])
        x = tensor.clip(t, 3, 8)
        print(x)

        # [3, 4, 6]

where
==============================

.. py:function:: pyvqnet.tensor.where(condition: pyvqnet.tensor.QTensor, t1: pyvqnet.tensor.QTensor, t2: pyvqnet.tensor.QTensor)

    Devuelve elementos seleccionados de x o y dependiendo de la condición.

    :param condition: tensor de condición, debe tener tipo de dato kbool.
    :param t1: QTensor del cual tomar elementos si se cumple la condición, por defecto None
    :param t2: QTensor del cual tomar elementos si no se cumple la condición, por defecto None
    :return: QTensor de salida

    Ejemplo::

        from pyvqnet.tensor import tensor
        from pyvqnet.tensor import QTensor
        t1 = QTensor([1, 2, 3])
        t2 = QTensor([4, 5, 6])
        x = tensor.where(t1 < 2, t1, t2)
        print(x)

        # [1, 5, 6]

nonzero
==============================

.. py:function:: pyvqnet.tensor.nonzero(t)

    Devuelve un QTensor que contiene los índices de los elementos distintos de cero.

    :param t: QTensor de entrada
    :return: QTensor de salida que contiene los índices de los elementos distintos de cero.

    Ejemplo::

        from pyvqnet.tensor import tensor
        from pyvqnet.tensor import QTensor
        t = QTensor([[0.6, 0.0, 0.0, 0.0],
                                    [0.0, 0.4, 0.0, 0.0],
                                    [0.0, 0.0, 1.2, 0.0],
                                    [0.0, 0.0, 0.0,-0.4]])
        t = tensor.nonzero(t)
        print(t)
        # [
        # [0, 0],
        # [1, 1],
        # [2, 2],
        # [3, 3]
        # ]

isfinite
==============================

.. py:function:: pyvqnet.tensor.isfinite(t)

    Prueba elemento a elemento si es finito (no infinito o no es un número).

    :param t: QTensor de entrada
    :return: QTensor de salida, que devuelve True cuando el elemento en la posición correspondiente cumple la condición, de lo contrario devuelve False.

    Ejemplo::

        from pyvqnet.tensor import tensor
        from pyvqnet.tensor import QTensor

        t = QTensor([1, float('inf'), 2, float('-inf'), float('nan')])
        flag = tensor.isfinite(t)
        print(flag)

        #[ True False  True False False]

isinf
==============================

.. py:function:: pyvqnet.tensor.isinf(t)

    Prueba elemento a elemento si es infinito positivo o negativo.

    :param t: QTensor de entrada
    :return: QTensor de salida, que devuelve True cuando el elemento en la posición correspondiente cumple la condición, de lo contrario devuelve False.

    Ejemplo::

        from pyvqnet.tensor import tensor
        from pyvqnet.tensor import QTensor

        t = QTensor([1, float('inf'), 2, float('-inf'), float('nan')])
        flag = tensor.isinf(t)
        print(flag)

        # [False  True False  True False]

isnan
==============================

.. py:function:: pyvqnet.tensor.isnan(t)

    Prueba elemento a elemento si es NaN.

    :param t: QTensor de entrada
    :return: QTensor de salida, que devuelve True cuando el elemento en la posición correspondiente cumple la condición, de lo contrario devuelve False.

    Ejemplo::

        from pyvqnet.tensor import tensor
        from pyvqnet.tensor import QTensor

        t = QTensor([1, float('inf'), 2, float('-inf'), float('nan')])
        flag = tensor.isnan(t)
        print(flag)

        # [False False False False  True]

isneginf
==============================

.. py:function:: pyvqnet.tensor.isneginf(t)

    Prueba elemento a elemento si es infinito negativo.

    :param t: QTensor de entrada
    :return: QTensor de salida, que devuelve True cuando el elemento en la posición correspondiente cumple la condición, de lo contrario devuelve False.

    Ejemplo::

        from pyvqnet.tensor import tensor
        from pyvqnet.tensor import QTensor

        t = QTensor([1, float('inf'), 2, float('-inf'), float('nan')])
        flag = tensor.isneginf(t)
        print(flag)

        # [False False False  True False]

isposinf
==============================

.. py:function:: pyvqnet.tensor.isposinf(t)

    Prueba elemento a elemento si es infinito positivo.

    :param t: QTensor de entrada
    :return: QTensor de salida, que devuelve True cuando el elemento en la posición correspondiente cumple la condición, de lo contrario devuelve False.

    Ejemplo::

        from pyvqnet.tensor import tensor
        from pyvqnet.tensor import QTensor

        t = QTensor([1, float('inf'), 2, float('-inf'), float('nan')])
        flag = tensor.isposinf(t)
        print(flag)

        # [False  True False False False]

logical_and
==============================

.. py:function:: pyvqnet.tensor.logical_and(t1, t2)

    Calcula el valor de verdad de ``t1`` y ``t2`` elemento a elemento.

    :param t1: QTensor de entrada
    :param t2: QTensor de entrada
    :return: QTensor de salida, que devuelve True cuando el elemento en la posición correspondiente cumple la condición, de lo contrario devuelve False.

    Ejemplo::

        from pyvqnet.tensor import tensor
        from pyvqnet.tensor import QTensor

        a = QTensor([0, 1, 10, 0])
        b = QTensor([4, 0, 1, 0])
        flag = tensor.logical_and(a,b)
        print(flag)

        # [False False  True False]

logical_or
==============================

.. py:function:: pyvqnet.tensor.logical_or(t1, t2)

    Calcula el valor de verdad de ``t1 or t2`` elemento a elemento.

    :param t1: QTensor de entrada
    :param t2: QTensor de entrada
    :return: QTensor de salida, que devuelve True cuando el elemento en la posición correspondiente cumple la condición, de lo contrario devuelve False.

    Ejemplo::

        from pyvqnet.tensor import tensor
        from pyvqnet.tensor import QTensor

        a = QTensor([0, 1, 10, 0])
        b = QTensor([4, 0, 1, 0])
        flag = tensor.logical_or(a,b)
        print(flag)

        # [ True  True  True False]

logical_not
==============================

.. py:function:: pyvqnet.tensor.logical_not(t)

    Calcula el valor de verdad de ``not t`` elemento a elemento.

    :param t: QTensor de entrada
    :return: QTensor de salida, que devuelve True cuando el elemento en la posición correspondiente cumple la condición, de lo contrario devuelve False.

    Ejemplo::

        from pyvqnet.tensor import tensor
        from pyvqnet.tensor import QTensor

        a = QTensor([0, 1, 10, 0])
        flag = tensor.logical_not(a)
        print(flag)

        # [ True False False  True]

logical_xor
==============================

.. py:function:: pyvqnet.tensor.logical_xor(t1, t2)

    Calcula el valor de verdad de ``t1 xor t2`` elemento a elemento.

    :param t1: QTensor de entrada
    :param t2: QTensor de entrada

    :return: QTensor de salida, que devuelve True cuando el elemento en la posición correspondiente cumple la condición, de lo contrario devuelve False.

    Ejemplo::

        from pyvqnet.tensor import tensor
        from pyvqnet.tensor import QTensor

        a = QTensor([0, 1, 10, 0])
        b = QTensor([4, 0, 1, 0])
        flag = tensor.logical_xor(a,b)
        print(flag)

        # [ True  True False False]

greater
==============================

.. py:function:: pyvqnet.tensor.greater(t1, t2)

    Devuelve el valor de verdad de ``t1 > t2`` elemento a elemento.


    :param t1: QTensor de entrada
    :param t2: QTensor de entrada
    :return: QTensor de salida, que devuelve True cuando el elemento en la posición correspondiente cumple la condición, de lo contrario devuelve False.

    Ejemplo::

        from pyvqnet.tensor import tensor
        from pyvqnet.tensor import QTensor

        a = QTensor([[1, 2], [3, 4]])
        b = QTensor([[1, 1], [4, 4]])
        flag = tensor.greater(a,b)
        print(flag)

        # [[False  True]
        #  [False False]]

greater_equal
==============================

.. py:function:: pyvqnet.tensor.greater_equal(t1, t2)

    Devuelve el valor de verdad de ``t1 >= t2`` elemento a elemento.

    :param t1: QTensor de entrada
    :param t2: QTensor de entrada
    :return: QTensor de salida, que devuelve True cuando el elemento en la posición correspondiente cumple la condición, de lo contrario devuelve False.

    Ejemplo::

        from pyvqnet.tensor import tensor
        from pyvqnet.tensor import QTensor

        a = QTensor([[1, 2], [3, 4]])
        b = QTensor([[1, 1], [4, 4]])
        flag = tensor.greater_equal(a,b)
        print(flag)

        #[[ True  True]
        # [False  True]]

less
==============================

.. py:function:: pyvqnet.tensor.less(t1, t2)

    Devuelve el valor de verdad de ``t1 < t2`` elemento a elemento.

    :param t1: QTensor de entrada
    :param t2: QTensor de entrada
    :return: QTensor de salida, que devuelve True cuando el elemento en la posición correspondiente cumple la condición, de lo contrario devuelve False.

    Ejemplo::

        from pyvqnet.tensor import tensor
        from pyvqnet.tensor import QTensor

        a = QTensor([[1, 2], [3, 4]])
        b = QTensor([[1, 1], [4, 4]])
        flag = tensor.less(a,b)
        print(flag)

        #[[False False]
        # [ True False]]

less_equal
==============================

.. py:function:: pyvqnet.tensor.less_equal(t1, t2)

    Devuelve el valor de verdad de ``t1 <= t2`` elemento a elemento.

    :param t1: QTensor de entrada
    :param t2: QTensor de entrada
    :return: QTensor de salida, que devuelve True cuando el elemento en la posición correspondiente cumple la condición, de lo contrario devuelve False.

    Ejemplo::

        from pyvqnet.tensor import tensor
        from pyvqnet.tensor import QTensor

        a = QTensor([[1, 2], [3, 4]])
        b = QTensor([[1, 1], [4, 4]])
        flag = tensor.less_equal(a,b)
        print(flag)

        # [[ True False]
        #  [ True  True]]

equal
==============================

.. py:function:: pyvqnet.tensor.equal(t1, t2)

    Devuelve el valor de verdad de ``t1 == t2`` elemento a elemento.

    :param t1: QTensor de entrada
    :param t2: QTensor de entrada
    :return: QTensor de salida, que devuelve True cuando el elemento en la posición correspondiente cumple la condición, de lo contrario devuelve False.

    Ejemplo::

        from pyvqnet.tensor import tensor
        from pyvqnet.tensor import QTensor

        a = QTensor([[1, 2], [3, 4]])
        b = QTensor([[1, 1], [4, 4]])
        flag = tensor.equal(a,b)
        print(flag)

        #[[ True False]
        # [False  True]]

not_equal
==============================

.. py:function:: pyvqnet.tensor.not_equal(t1, t2)

    Devuelve el valor de verdad de ``t1 != t2`` elemento a elemento.

    :param t1: QTensor de entrada
    :param t2: QTensor de entrada
    :return: QTensor de salida, que devuelve True cuando el elemento en la posición correspondiente cumple la condición, de lo contrario devuelve False.

    Ejemplo::

        from pyvqnet.tensor import tensor
        from pyvqnet.tensor import QTensor

        a = QTensor([[1, 2], [3, 4]])
        b = QTensor([[1, 1], [4, 4]])
        flag = tensor.not_equal(a,b)
        print(flag)


        #[[False  True]
        # [ True False]]


bitwise_and
==============================

.. py:function:: pyvqnet.tensor.bitwise_and(t1, t2)
 
    Calcula el AND bit a bit de dos elementos QTensor.

    :param t1: QTensor de entrada t1. Solo enteros o booleanos son entradas válidas.
    :param t2: QTensor de entrada t2. Solo enteros o booleanos son entradas válidas.
    :return:
        QTensor resultante

    Ejemplo::

        from pyvqnet.tensor import *
        import numpy as np
        from pyvqnet.dtype import *
        powers_of_two = 1 << np.arange(14, dtype=np.int64)[::-1]
        samples = tensor.QTensor([23],dtype=kint8)
        samples = samples.unsqueeze(-1)
        states_sampled_base_ten = samples & tensor.QTensor(powers_of_two,dtype = samples.dtype, device = samples.device)
        print(states_sampled_base_ten)
        #[[ 0, 0, 0, 0, 0, 0, 0, 0, 0,16, 0, 4, 2, 1]]



**********************

select
==============================

.. py:function:: pyvqnet.tensor.select(t: pyvqnet.tensor.QTensor, index)

    Devuelve QTensor en el QTensor en el eje dado. La siguiente operación obtiene el valor del mismo resultado.

    :param t: QTensor de entrada
    :param index: una cadena que contiene la dimensión de salida
    :return: QTensor de salida

    Ejemplo::

        from pyvqnet.tensor import tensor
        from pyvqnet.tensor import QTensor
        import numpy as np
        t = QTensor(np.arange(1,25.0).reshape(2,3,4))
              
        indx = [":", "0", ":"]
        t.requires_grad = True
        t.zero_grad()
        ts = tensor.select(t,indx)
        print(ts)  
        # [
        # [[1., 2., 3., 4.]],
        # [[13., 14., 15., 16.]]
        # ]


broadcast
==============================

.. py:function:: pyvqnet.tensor.broadcast(t1: pyvqnet.tensor.QTensor, t2: pyvqnet.tensor.QTensor)

    Sujeto a ciertas restricciones, los arrays más pequeños se colocan en arrays más grandes para que tengan formas compatibles. Esta interfaz puede realizar diferenciación automática en los tensores de parámetros de entrada.

    Referencia https://numpy.org/doc/stable/user/basics.broadcasting.html

    :param t1: QTensor de entrada 1
    :param t2: QTensor de entrada 2

    :return t11: t1 con nueva forma de broadcast.
    :return t22: t2 con nueva forma de broadcast.

    Ejemplo::

        from pyvqnet.tensor import tensor
        t1 = tensor.ones([5, 4])
        t2 = tensor.ones([4])

        t11, t22 = tensor.broadcast(t1, t2)

        print(t11.shape)
        print(t22.shape)

        t1 = tensor.ones([5, 4])
        t2 = tensor.ones([1])

        t11, t22 = tensor.broadcast(t1, t2)

        print(t11.shape)
        print(t22.shape)

        t1 = tensor.ones([5, 4])
        t2 = tensor.ones([2, 1, 4])

        t11, t22 = tensor.broadcast(t1, t2)

        print(t11.shape)
        print(t22.shape)


        # [5, 4]
        # [5, 4]
        # [5, 4]
        # [5, 4]
        # [2, 5, 4]
        # [2, 5, 4]

concatenate
==============================

.. py:function:: pyvqnet.tensor.concatenate(args: list, axis=1)

    Concatena el QTensor de entrada a lo largo del eje y devuelve un nuevo QTensor.

    :param args: lista compuesta de QTensors de entrada
    :param axis: dimensión para concatenar. Debe estar entre 0 y el número de dimensiones de los tensores a concatenar.
    :return: QTensor de salida

    Ejemplo::

        from pyvqnet.tensor import tensor
        from pyvqnet.tensor import QTensor
        x = QTensor([[1.0, 2, 3],[4,5,6]], requires_grad=True) 
        y = 1-x  
        x = tensor.concatenate([x,y],1)
        print(x)

        # [
        # [1, 2, 3, 0, -1, -2],
        # [4, 5, 6, -3, -4, -5]
        # ]

stack
==============================

.. py:function:: pyvqnet.tensor.stack(QTensors: list, axis) 

    Une una secuencia de arrays a lo largo de un nuevo eje, devuelve un nuevo QTensor.

    :param QTensors: lista que contiene QTensors
    :param axis: dimensión a insertar. Debe estar entre 0 y el número de dimensiones de los tensores apilados.
    :return: QTensor de salida

    Ejemplo::

        from pyvqnet.tensor import tensor
        from pyvqnet.tensor import QTensor
        import numpy as np
        R, C = 3, 4
        a = np.arange(R * C).reshape(R, C).astype(np.float32)
        t11 = QTensor(a)
        t22 = QTensor(a)
        t33 = QTensor(a)
        rlt1 = tensor.stack([t11,t22,t33],2)
        print(rlt1)

        # [
        # [[0, 0, 0],
        #  [1, 1, 1],
        #  [2, 2, 2],
        #  [3, 3, 3]],
        # [[4, 4, 4],
        #  [5, 5, 5],
        #  [6, 6, 6],
        #  [7, 7, 7]],
        # [[8, 8, 8],
        #  [9, 9, 9],
        #  [10, 10, 10],
        #  [11, 11, 11]]
        # ]

permute
==============================

.. py:function:: pyvqnet.tensor.permute(t: pyvqnet.tensor.QTensor, dim: list)

    Invierte o permuta los ejes de un array.

    :param t: QTensor de entrada
    :param dim: el nuevo orden de las dimensiones (lista de enteros)
    :return: QTensor de salida

    Ejemplo::

        from pyvqnet.tensor import tensor
        from pyvqnet.tensor import QTensor
        import numpy as np
        R, C = 3, 4
        a = np.arange(R * C).reshape([2,2,3]).astype(np.float32)
        t = QTensor(a)
        tt = tensor.permute(t,[2,0,1])
        print(tt)

        # [
        # [[0, 3],
        #  [6, 9]],
        # [[1, 4],
        #  [7, 10]],
        # [[2, 5],
        #  [8, 11]]
        # ]

transpose
==============================

.. py:function:: pyvqnet.tensor.transpose(t: pyvqnet.tensor.QTensor, dim: list)

    Transpone los ejes de un array. Si dim = None, invierte las dimensiones. Esta función es igual que permute.

    :param t: QTensor de entrada
    :param dim: el nuevo orden de las dimensiones (lista de enteros)
    :return: QTensor de salida

    Ejemplo::

        from pyvqnet.tensor import tensor
        from pyvqnet.tensor import QTensor
        import numpy as np
        R, C = 3, 4
        a = np.arange(R * C).reshape([2,2,3]).astype(np.float32)
        t = QTensor(a)
        tt = tensor.transpose(t,[2,0,1])
        print(tt)

        # [
        # [[0, 3],
        #  [6, 9]],
        # [[1, 4],
        #  [7, 10]],
        # [[2, 5],
        #  [8, 11]]
        # ]

tile
==============================

.. py:function:: pyvqnet.tensor.tile(t: pyvqnet.tensor.QTensor, reps: list)

    Construye un QTensor repitiendo QTensor el número de veces dado por reps.

    Si reps tiene longitud d, el QTensor resultante tendrá dimensión de max(d, t.ndim).

    Si t.ndim < d, t se expande para ser d-dimensional insertando nuevos ejes desde la dimensión inicial.
    Así, un array de forma (3,) se promociona a (1, 3) para replicación 2-D, o forma (1, 1, 3) para replicación 3-D.

    Si t.ndim > d, reps se expande a t.ndim insertando 1's en él.

    Así, para un t de forma (2, 3, 4, 5), un reps de (4, 3) se trata como (1, 1, 4, 3).

    :param t: QTensor de entrada
    :param reps: el número de repeticiones por dimensión.
    :return: un nuevo QTensor

    Ejemplo::

        from pyvqnet.tensor import tensor
        from pyvqnet.tensor import QTensor

        import numpy as np
        a = np.arange(6).reshape(2,3).astype(np.float32)
        A = QTensor(a)
        reps = [2,2]
        B = tensor.tile(A,reps)
        print(B)

        # [
        # [0, 1, 2, 0, 1, 2],
        # [3, 4, 5, 3, 4, 5],
        # [0, 1, 2, 0, 1, 2],
        # [3, 4, 5, 3, 4, 5]
        # ]

squeeze
==============================

.. py:function:: pyvqnet.tensor.squeeze(t: pyvqnet.tensor.QTensor, axis: int = - 1)

    Elimina ejes de longitud uno.

    :param t: QTensor de entrada
    :param axis: eje para hacer squeeze, si axis = -1, hace squeeze de todas las dimensiones que tienen tamaño 1.
    :return: QTensor de salida

    Ejemplo::

        from pyvqnet.tensor import tensor
        from pyvqnet.tensor import QTensor
        import numpy as np
        a = np.arange(6).reshape(1,6,1).astype(np.float32)
        A = QTensor(a)
        AA = tensor.squeeze(A,0)
        print(AA)

        # [
        # [0],
        # [1],
        # [2],
        # [3],
        # [4],
        # [5]
        # ]

unsqueeze
==============================

.. py:function:: pyvqnet.tensor.unsqueeze(t: pyvqnet.tensor.QTensor, axis: int = 0)

    Devuelve un nuevo QTensor con una dimensión de tamaño uno insertada en la posición especificada.

    :param t: QTensor de entrada
    :param axis: eje para hacer unsqueeze, donde se insertará la dimensión.
    :return: QTensor de salida

    Ejemplo::

        from pyvqnet.tensor import tensor
        from pyvqnet.tensor import QTensor
        import numpy as np
        a = np.arange(24).reshape(2,1,1,4,3).astype(np.float32)
        A = QTensor(a)
        AA = tensor.unsqueeze(A,1)
        print(AA)

        # [
        # [[[[[0, 1, 2],
        #  [3, 4, 5],
        #  [6, 7, 8],
        #  [9, 10, 11]]]]],
        # [[[[[12, 13, 14],
        #  [15, 16, 17],
        #  [18, 19, 20],
        #  [21, 22, 23]]]]]
        # ]


moveaxis
===============================

.. py:function:: pyvqnet.tensor.moveaxis(t, source: int, destination: int)

    Mueve las dimensiones de `t` desde las posiciones en `source` a las posiciones en `destination`.

    Otras dimensiones de `t` que no se mueven explícitamente mantienen su orden original y aparecen en posiciones no especificadas en `destination`.

    :param t: QTensor de entrada.
    :param source: (entero o tupla de enteros) Las posiciones originales de las dimensiones a mover. Estas posiciones deben ser únicas.
    :param destination: (entero o tupla de enteros) Las posiciones de destino para cada dimensión original. Estas posiciones también deben ser únicas.

    :return:
        Nuevo QTensor


    Ejemplo::

        from pyvqnet import QTensor,tensor
        a = tensor.arange(0,24).reshape((2,3,4))
        b = tensor.moveaxis(a,(1, 2), (0, 1))
        print(b.shape)


swapaxis
==============================

.. py:function:: pyvqnet.tensor.swapaxis(t, axis1: int, axis2: int)

    Intercambia dos ejes de un array. Las dimensiones dadas axis1 y axis2 se intercambian.

    :param t: QTensor de entrada
    :param axis1: Primer eje.
    :param axis2: Posición de destino para el eje original. También deben ser únicas.
    :return: QTensor de salida

    Ejemplo::

        from pyvqnet.tensor import tensor
        from pyvqnet.tensor import QTensor
        import numpy as np
        a = np.arange(24).reshape(2,3,4).astype(np.float32)
        A = QTensor(a)
        AA = tensor.swapaxis(A,2,1)
        print(AA)

        # [
        # [[0, 4, 8],
        #  [1, 5, 9],
        #  [2, 6, 10],
        #  [3, 7, 11]],
        # [[12, 16, 20],
        #  [13, 17, 21],
        #  [14, 18, 22],
        #  [15, 19, 23]]
        # ]

masked_fill
==============================

.. py:function:: pyvqnet.tensor.masked_fill(t, mask, value)

    Si mask == 1, llena con el valor especificado. La forma de mask debe ser transmisible desde la forma del QTensor de entrada.

    :param t: QTensor de entrada
    :param mask: Un QTensor
    :param value: valor especificado
    :return: Un QTensor

    Ejemplos::

        from pyvqnet.tensor import tensor
        import numpy as np
        a = tensor.ones([2, 2, 2, 2])
        mask = np.random.randint(0, 2, size=4).reshape([2, 2])
        b = tensor.QTensor(mask==1)
        c = tensor.masked_fill(a, b, 13)
        print(c)
        # [
        # [[[1, 1],
        #  [13, 13]],
        # [[1, 1],
        #  [13, 13]]],
        # [[[1, 1],
        #  [13, 13]],
        # [[1, 1],
        #  [13, 13]]]
        # ]


flatten
==============================

.. py:function:: pyvqnet.tensor.flatten(t: pyvqnet.tensor.QTensor, start: int = 0, end: int = - 1)

    Aplana QTensor desde la dimensión start hasta la dimensión end.

    :param t: QTensor de entrada
    :param start: dimensión inicial, default = 0, empezar desde la primera dimensión.
    :param end: dimensión final, default = -1, terminar con la última dimensión.
    :return: QTensor de salida

    Ejemplo::

        from pyvqnet.tensor import tensor
        from pyvqnet.tensor import QTensor
        t = QTensor([1, 2, 3])
        x = tensor.flatten(t)
        print(x)

        # [1, 2, 3]


reshape
==============================

.. py:function:: pyvqnet.tensor.reshape(t: pyvqnet.tensor.QTensor,new_shape)

    Cambia la forma del QTensor, devuelve un QTensor con nueva forma.

    :param t: QTensor de entrada.
    :param new_shape: nueva forma

    :return: un QTensor con nueva forma.

    Ejemplo::

        from pyvqnet.tensor import tensor
        from pyvqnet.tensor import QTensor
        import numpy as np
        R, C = 3, 4
        a = np.arange(R * C).reshape(R, C).astype(np.float32)
        t = QTensor(a)
        reshape_t = tensor.reshape(t, [C, R])
        print(reshape_t)
        # [
        # [0, 1, 2],
        # [3, 4, 5],
        # [6, 7, 8],
        # [9, 10, 11]
        # ]

flip
==============================

.. py:function:: pyvqnet.tensor.flip(t, flip_dims)

    Invierte el QTensor a lo largo del eje especificado, devolviendo un nuevo tensor.

    :param t: QTensor de entrada.
    :param flip_dims: El eje o lista de ejes a invertir.

    :return: QTensor de salida.

    Ejemplo::

        from pyvqnet import tensor
        t = tensor.arange(1, 3 * 2 *2 * 2 + 1).reshape([3, 2, 2, 2])
        t.requires_grad = True
        y = tensor.flip(t, [0, -1])
        print(y)
        # [
        # [[[18, 17], 
        #  [20, 19]], 
        # [[22, 21],  
        #  [24, 23]]],
        # [[[10, 9],  
        #  [12, 11]], 
        # [[14, 13],  
        #  [16, 15]]],
        # [[[2, 1],   
        #  [4, 3]],   
        # [[6, 5],    
        #  [8, 7]]]   
        # ]


gather
=============================

.. py:function:: pyvqnet.tensor.gather(t, dim, index)

    Recoge valores a lo largo del eje especificado por 'dim'.

    Para tensores 3-D, la salida se especifica por:

    .. math::

        out[i][j][k] = t[index[i][j][k]][j][k] , if dim == 0 \

        out[i][j][k] = t[i][index[i][j][k]][k] , if dim == 1 \

        out[i][j][k] = t[i][j][index[i][j][k]] , if dim == 2 \

    :param t: QTensor de entrada.
    :param dim: El eje de agregación.
    :param index: QTensor índice, debe tener el mismo tamaño de dimensión que la entrada.

    :return: el resultado agregado

    Ejemplo::

        from pyvqnet.tensor import gather,QTensor,tensor
        import numpy as np
        np.random.seed(25)
        npx = np.random.randn( 3, 4,6)
        npindex = np.array([2,3,1,2,1,2,3,0,2,3,1,2,3,2,0,1]).reshape([2,2,4]).astype(np.int64)

        x1 = QTensor(npx)
        indices1 =  QTensor(npindex)
        x1.requires_grad = True
        y1 = gather(x1,1,indices1)
        y1.backward(tensor.arange(0,y1.numel()).reshape(y1.shape))

        print(y1)
        # [
        # [[2.1523438, -0.4196777, -2.0527344, -1.2460938],
        #  [-0.6201172, -1.3349609, 2.2949219, -0.5913086]],
        # [[0.2170410, -0.7055664, 1.6074219, -1.9394531],
        #  [0.2430420, -0.6333008, 0.5332031, 0.3881836]]
        # ]

scatter
=============================

.. py:function:: pyvqnet.tensor.scatter(input, dim, index, src)

    Escribe todos los valores en el tensor src en input en los índices especificados en el tensor indices.

    Para tensores 3-D, la salida se especifica por:

    .. math::

        input[indices[i][j][k]][j][k] = src[i][j][k] , if dim == 0 \
        input[i][indices[i][j][k]][k] = src[i][j][k] , if dim == 1 \
        input[i][j][indices[i][j][k]] = src[i][j][k] , if dim == 2 \

    :param input: QTensor de entrada.
    :param dim: Eje de dispersión.
    :param indices: QTensor índice, debe tener el mismo tamaño de dimensión que la entrada.
    :param src: El tensor fuente para dispersar.

    Ejemplo::

        from pyvqnet.tensor import scatter, QTensor
        import numpy as np
        np.random.seed(25)
        npx = np.random.randn(3, 2, 4, 2)
        npindex = np.array([2, 3, 1, 2, 1, 2, 3, 0, 2, 3, 1, 2, 3, 2, 0,
                            1]).reshape([2, 2, 4, 1]).astype(np.int64)
        x1 = QTensor(npx)
        npsrc = QTensor(np.full_like(npindex, 200), dtype=x1.dtype)
        npsrc.requires_grad = True
        indices1 = QTensor(npindex)
        y1 = scatter(x1, 2, indices1, npsrc)
        print(y1)

        # [[[[  0.2282731   1.0268903]
        #    [200.         -0.5911815]
        #    [200.         -0.2223257]
        #    [200.          1.8379046]]

        #   [[200.          0.8685831]
        #    [200.         -0.2323119]
        #    [200.         -1.3346615]
        #    [200.         -1.2460893]]]


        #  [[[  1.2022723  -1.0499416]
        #    [200.         -0.4196777]
        #    [200.         -2.5944874]
        #    [200.          0.6808889]]

        #   [[200.         -1.9762536]
        #    [200.         -0.2908697]
        #    [200.          1.9826261]
        #    [200.         -1.839905 ]]]


        #  [[[  1.6076708   0.3882919]
        #    [  0.3997321   0.4054766]
        #    [  0.2170018  -0.6334391]
        #    [  0.2466215  -1.9395455]]

        #   [[  0.1140596  -1.8853414]
        #    [  0.2430805  -0.7054807]
        #    [  0.3646276  -0.5029522]
        #    [ -0.2257515  -0.5655377]]]]

broadcast_to
=============================

.. py:function:: pyvqnet.tensor.broadcast_to(t, ref)

    Sujeto a ciertas restricciones, el array t se "transmite" a la forma de referencia para que tengan formas compatibles.

    https://numpy.org/doc/stable/user/basics.broadcasting.html

    :param t: QTensor de entrada
    :param ref: Forma de referencia.

    :return: El QTensor de t recién transmitido.

    Ejemplo::

        from pyvqnet.tensor.tensor import QTensor
        from pyvqnet.tensor import *
        ref = [2,3,4]
        a = ones([4])
        b = tensor.broadcast_to(a,ref)
        print(b.shape)
        #[2, 3, 4]



Funciones de Utilidad
*****************************************************


to_tensor
==============================

.. py:function:: pyvqnet.tensor.to_tensor(x)

    Convierte el array de entrada a QTensor si aún no lo es.

    :param x: entero, flotante o numpy.array
    :return: QTensor de salida

    Ejemplo::

        from pyvqnet.tensor import tensor
        t = tensor.to_tensor(10.0)
        print(t)
        # [10]


pad_sequence
==============================

.. py:function:: pyvqnet.tensor.pad_sequence(qtensor_list, batch_first=False, padding_value=0)

    Rellena una lista de tensores de longitud variable con ``padding_value``. ``pad_sequence`` apila listas de tensores a lo largo de nuevas dimensiones y los rellena hasta igualar la longitud.
    La entrada es una secuencia de listas de tamaño ``L x *``. L es de longitud variable.

    :param qtensor_list: `list[QTensor]` - lista de secuencias de longitud variable.
    :param batch_first: 'bool' - Si es true, la salida será ``batch size x longest sequence length x *``, de lo contrario ``longest sequence length x batch size x *``. Por defecto: False.
    :param padding_value: 'float' - valor de relleno. Valor por defecto: 0.

    :return:
         Si batch_first es ``False``, el tamaño del tensor es ``batch size x longest sequence length x *``.
         De lo contrario, el tamaño del tensor es ``longest sequence length x batch size x *``.

    Ejemplos::

        from pyvqnet.tensor import tensor
        a = tensor.ones([4, 2,3])
        b = tensor.ones([1, 2,3])
        c = tensor.ones([2, 2,3])
        a.requires_grad = True
        b.requires_grad = True
        c.requires_grad = True
        y = tensor.pad_sequence([a, b, c], True)

        print(y)
        # [
        # [[[1, 1, 1],
        #  [1, 1, 1]],
        # [[1, 1, 1],
        #  [1, 1, 1]],
        # [[1, 1, 1],
        #  [1, 1, 1]],
        # [[1, 1, 1],
        #  [1, 1, 1]]],
        # [[[1, 1, 1],
        #  [1, 1, 1]],
        # [[0, 0, 0],
        #  [0, 0, 0]],
        # [[0, 0, 0],
        #  [0, 0, 0]],
        # [[0, 0, 0],
        #  [0, 0, 0]]],
        # [[[1, 1, 1],
        #  [1, 1, 1]],
        # [[1, 1, 1],
        #  [1, 1, 1]],
        # [[0, 0, 0],
        #  [0, 0, 0]],
        # [[0, 0, 0],
        #  [0, 0, 0]]]
        # ]


pad_packed_sequence
==============================

.. py:function:: pyvqnet.tensor.pad_packed_sequence(sequence, batch_first=False, padding_value=0, total_length=None)

    Rellena un lote de secuencias empaquetadas de longitud variable. Es la inversa de `pack_pad_sequence`.
    Cuando ``batch_first`` es True, devuelve un tensor de forma ``B x T x *``, de lo contrario devuelve ``T x B x *``.
    Donde `T` es la longitud de secuencia más larga y `B` es el tamaño del lote.

    :param sequence: 'QTensor' - los datos a procesar.
    :param batch_first: 'bool' - Si ``True``, el lote será la primera dimensión de la entrada. Valor por defecto: False.
    :param padding_value: 'bool' - valor de relleno. Por defecto: 0.
    :param total_length: 'bool' - Si no es ``None``, la salida se rellenará hasta la longitud :attr:`total_length`. Por defecto: None.
    :return:
        Una tupla de tensores que contienen las secuencias rellenadas, y una lista de longitudes para cada secuencia en el lote. Los elementos del lote se reordenarán en su orden original.

    Ejemplos::

        from pyvqnet.tensor import tensor
        a = tensor.ones([4, 2,3])
        b = tensor.ones([2, 2,3])
        c = tensor.ones([1, 2,3])
        a.requires_grad = True
        b.requires_grad = True
        c.requires_grad = True
        y = tensor.pad_sequence([a, b, c], True)
        seq_len = [4, 2, 1]
        data = tensor.pack_pad_sequence(y,
                                seq_len,
                                batch_first=True,
                                enforce_sorted=True)

        seq_unpacked, lens_unpacked = tensor.pad_packed_sequence(data, batch_first=True)
        print(seq_unpacked)
        # [[[[1. 1. 1.]
        #    [1. 1. 1.]]

        #   [[1. 1. 1.]
        #    [1. 1. 1.]]

        #   [[1. 1. 1.]
        #    [1. 1. 1.]]

        #   [[1. 1. 1.]
        #    [1. 1. 1.]]]


        #  [[[1. 1. 1.]
        #    [1. 1. 1.]]

        #   [[1. 1. 1.]
        #    [1. 1. 1.]]

        #   [[0. 0. 0.]
        #    [0. 0. 0.]]

        #   [[0. 0. 0.]
        #    [0. 0. 0.]]]


        #  [[[1. 1. 1.]
        #    [1. 1. 1.]]

        #   [[0. 0. 0.]
        #    [0. 0. 0.]]

        #   [[0. 0. 0.]
        #    [0. 0. 0.]]
        #   [[0. 0. 0.]
        #    [0. 0. 0.]]]]
        print(lens_unpacked)
        # [4, 2, 1]


pack_pad_sequence
==============================

.. py:function:: pyvqnet.tensor.pack_pad_sequence(input, lengths, batch_first=False, enforce_sorted=True)

    Empaqueta un Tensor que contiene secuencias rellenadas de longitud variable. Si batch_first es True, `input` debe tener forma [batch size, length,*], de lo contrario forma [length, batch size,*].

    Para secuencias no ordenadas, use ``enforce_sorted`` como False. Si :attr:`enforce_sorted` es ``True``, las secuencias deben estar ordenadas en orden descendente por longitud.

    :param input: 'QTensor' - lotes de secuencias de longitud variable para rellenar.
    :param lengths: 'list' - lista de longitudes de secuencia para cada elemento del lote.
    :param batch_first: 'bool' - si ``True``, se espera que la entrada esté en formato ``B x T x *``, por defecto: False.
    :param enforce_sorted: 'bool' - si ``True``, la entrada debe contener secuencias en orden descendente de longitud. Si ``False``, la entrada se ordenará incondicionalmente. Por defecto: True.

    :return: Un objeto :class:`PackedSequence`.

    Ejemplos::

        from pyvqnet.tensor import tensor
        a = tensor.ones([4, 2,3])
        c = tensor.ones([1, 2,3])
        b = tensor.ones([2, 2,3])
        a.requires_grad = True
        b.requires_grad = True
        c.requires_grad = True
        y = tensor.pad_sequence([a, b, c], True)
        seq_len = [4, 2, 1]
        data = tensor.pack_pad_sequence(y,
                                seq_len,
                                batch_first=True,
                                enforce_sorted=False)
        print(data.data)

        # [[[1. 1. 1.]
        #   [1. 1. 1.]]

        #  [[1. 1. 1.]
        #   [1. 1. 1.]]

        #  [[1. 1. 1.]
        #   [1. 1. 1.]]

        #  [[1. 1. 1.]
        #   [1. 1. 1.]]

        #  [[1. 1. 1.]
        #   [1. 1. 1.]]

        #  [[1. 1. 1.]
        #   [1. 1. 1.]]

        #  [[1. 1. 1.]
        #   [1. 1. 1.]]]

        print(data.batch_sizes)
        # [3, 2, 1, 1]


functional_conv2d
==============================
.. py:function:: pyvqnet.nn.functional.functional_conv2d(x, weight, bias, stride=(1,1), padding=(0,0), dilation=(1,1), groups=1)

    Realiza una convolución 2D en una imagen de entrada que consta de múltiples planos de entrada.

    :param x: Tensor de entrada 4D.
    :param weight: Tensor de kernel 4D.

    :param stride: `tuple` - paso, por defecto (1, 1)
    :param padding: Relleno, controla la cantidad de relleno en la entrada. Puede ser una cadena {'valid', 'same'} o una tupla de enteros que especifica la cantidad de relleno implícito a aplicar a la entrada, por defecto (0,0).
    :param dilation: `tuple` - Espaciado entre elementos del kernel. Por defecto: (0,0)
    :param groups: `int` - Número de grupos. Valor por defecto: 1

    :return: qtensor


    Ejemplos::

        from pyvqnet.nn.functional import functional_conv2d 
        from pyvqnet.tensor import arange,ones 
        from pyvqnet import kfloat32 
        from pyvqnet.nn import Module,Parameter 


        classTM(Module): 
            def __init__(self, *args, **kwargs): 
                super().__init__(*args, **kwargs) 
                self.w = ones([5,4,2,2]) 
                self.w.requires_grad = True 
                self.b = ones([5,]) 
                self.b.requires_grad = True 

            def forward(self,x): 
                weight, bias, = self.w, self.b 
                return functional_conv2d(x, weight, bias) 


        x = arange(0,7*4*12*12,dtype=kfloat32).reshape([7,4,12,12]) 
        l = TM() 
        y = l(x) 

        y.backward( )

no_grad
==============================

.. py:function:: pyvqnet.no_grad()

    Deshabilita el registro de nodos de retropropagación durante el cálculo hacia adelante.

    Ejemplo::

        import pyvqnet.tensor as tensor
        from pyvqnet import no_grad

        with no_grad():
            x = tensor.QTensor([1.0, 2.0, 3.0],requires_grad=True)
            y = tensor.tan(x)
            y.backward()
        """
        RuntimeError: The output tensor does not require gradients (output.requires_grad == False). This may occur if you used a non-autograd function in the forward pass. To enable gradient computation, ensure that all operations are performed on tensors with requires_grad=True, or use autograd-compatible functions.
        """