.. _qtensor_api:

Módulo QTensor
###########################

O aprendizado de máquina quântico do VQNet utiliza a estrutura de dados QTensor, que é uma interface Python. O QTensor suporta operações matriciais multidimensionais comuns, incluindo funções de criação, funções matemáticas, funções lógicas, transformações matriciais, etc.




Funções e Atributos do QTensor
******************************************


QTensor
==============================

.. py:class:: pyvqnet.tensor.tensor.QTensor(data, requires_grad=False, nodes=None, device=0, dtype=None, name='')

    Encapsulamento da estrutura de dados com construção dinâmica de grafo computacional
    e diferenciação automática.

    :param data: _core.Tensor ou array numpy que representa um QTensor
    :param requires_grad: indica se o gradiente do tensor deve ser rastreado, padrão: False
    :param nodes: lista de sucessores no grafo computacional, padrão: None
    :param device: dispositivo atual para salvar o QTensor, padrão = 0, usa CPU.
    :param dtype: O tipo de dados do parâmetro, padrão None, usa o tipo de dados padrão: kfloat32, que representa um número de ponto flutuante de 32 bits.
    :param name: O nome do QTensor, padrão: "".

    :return: QTensor de saída


    Example::

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

        Retorna o número de dimensões de um tensor.

        :return: O número de dimensões de um tensor.

        Example::

            from pyvqnet.tensor import QTensor

            a = QTensor([2.0, 3.0, 4.0, 5.0], requires_grad=True)
            print(a.ndim)

            # 1

    .. py:attribute:: shape

        Retorna as dimensões de um tensor

        :return: Uma lista das dimensões do tensor

        Example::

            from pyvqnet.tensor import QTensor

            a = QTensor([2.0, 3.0, 4.0, 5.0], requires_grad=True)
            print(a.shape)

            # [4]

    .. py:attribute:: size

        Retorna o número de elementos de um tensor.

        :return: O número de elementos de um tensor.

        Example::

            from pyvqnet.tensor import QTensor

            a = QTensor([2.0, 3.0, 4.0, 5.0], requires_grad=True)
            print(a.size)

            # 4

    .. py:method:: numel

        Retorna o número de elementos em um tensor.

        :return: O número de elementos em um tensor.

        Example::

            from pyvqnet.tensor import QTensor

            a = QTensor([2.0, 3.0, 4.0, 5.0], requires_grad=True)
            print(a.numel())

            # 4

    .. py:attribute:: dtype

        Retorna o tipo de dados de um tensor.

        Os tipos de dados suportados são os seguintes:

            =========================================  ===============================
            dtype                                      descrição
            =========================================  ===============================
            ``pyvqnet.kbool``                          Variável booleana
            ``pyvqnet.kuint8``                         Inteiro de 8 bits (sem sinal)
            ``pyvqnet.kint8``                          Inteiro de 8 bits (com sinal)
            ``pyvqnet.kint16``                         Inteiro de 16 bits (com sinal)
            ``pyvqnet.kint32``                         Inteiro de 32 bits (com sinal)
            ``pyvqnet.kint64``                         Inteiro de 64 bits (com sinal)
            ``pyvqnet.kfloat32``                       Ponto flutuante de 32 bits, veja https://en.wikipedia.org/wiki/IEEE_754
            ``pyvqnet.kfloat64``                       Ponto flutuante de 64 bits, veja https://en.wikipedia.org/wiki/IEEE_754
            ``pyvqnet.kcomplex64``                     Número complexo de 64 bits, composto por dois `float32`
            ``pyvqnet.kcomplex128``                    Número complexo de 128 bits, composto por dois `float64`
            ``pyvqnet.kbfloat16``                      Ponto flutuante de 16 bits, também chamado de formato Brain floating point, com alocação de bits: 1 bit de sinal, 8 bits de expoente e 7 bits de mantissa
            =========================================  ===============================

        :return: O tipo de dados do tensor.

        Example::

            from pyvqnet.tensor import QTensor

            a = QTensor([2, 3, 4, 5])
            print(a.dtype)
            # 4


    .. py:method:: zero_grad()

        Define o gradiente como zero. Será usado pelo otimizador no processo de otimização.

        :return: None

        Example::

            from pyvqnet.tensor import tensor
            from pyvqnet.tensor import QTensor
            t3  =  QTensor([2.0,3.0,4.0,5.0],requires_grad = True)
            t3.zero_grad()
            print(t3.grad)

            # [0, 0, 0, 0]


 
    .. py:method:: backward(grad=None)

        Calcula o gradiente do QTensor atual.

        :return: None

        Example::

            from pyvqnet.tensor import tensor
            from pyvqnet.tensor import QTensor

            target = QTensor([[0, 0, 1, 0, 0, 0, 0, 0, 0, 0.2]], requires_grad=True)
            y = 2*target + 3
            y.backward()
            print(target.grad)
            #[[2. 2. 2. 2. 2. 2. 2. 2. 2. 2.]]

 

    .. py:method:: to_numpy()

        Copia os dados para um novo numpy.array.

        :return: um novo numpy.array contendo os dados do QTensor

        .. note::

            O numpy não suporta o tipo bfloat16; você precisa converter para outros tipos de dados suportados pelo numpy, como float32, antes de chamar esta interface.

        Example::

            from pyvqnet.tensor import tensor
            from pyvqnet.tensor import QTensor
            t3  =  QTensor([2.0,3.0,4.0,5.0],requires_grad = True)
            t4 = t3.to_numpy()
            print(t4)

            # [2. 3. 4. 5.]

 
    .. py:method:: item()

            Retorna o único elemento do QTensor. Lança 'RuntimeError' se o QTensor tiver mais de 1 elemento.

            :return: único dado deste objeto

            Example::

                from pyvqnet.tensor import tensor

                t = tensor.ones([1])
                print(t.item())

                # 1.0

 
    .. py:method:: argmax(*kargs)

        Retorna os índices do valor máximo de todos os elementos no QTensor de entrada, ou
        Retorna os índices dos valores máximos de um QTensor ao longo de uma dimensão.

        :param dim: dim (int) – a dimensão a ser reduzida, aceita apenas um único eixo. se dim == None, retorna os índices do valor máximo de todos os elementos no tensor de entrada. O intervalo válido de dim é [-R, R), onde R é o ndim da entrada. quando dim < 0, funciona da mesma forma que dim + R.
        :param keepdims: indica se a dimensão do QTensor de saída é mantida ou não.

        :return: os índices do valor máximo no QTensor de entrada.

        Example::

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

        Retorna os índices do valor mínimo de todos os elementos no QTensor de entrada, ou
        Retorna os índices dos valores mínimos de um QTensor ao longo de uma dimensão.

        :param dim: dim (int) – a dimensão a ser reduzida, aceita apenas um único eixo. se dim == None, retorna os índices do valor mínimo de todos os elementos no tensor de entrada. O intervalo válido de dim é [-R, R), onde R é o ndim da entrada. quando dim < 0, funciona da mesma forma que dim + R.
        :param keepdims: indica se a dimensão do QTensor de saída é mantida ou não.

        :return: os índices do valor mínimo no QTensor de entrada.

        Example::

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

            Preenche o QTensor com o valor especificado, modificando-o internamente.

            :param v: um valor escalar
            :return: None

            Example::

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

            Retorna True se todos os valores do QTensor forem diferentes de zero.

            :return: True se todos os valores do QTensor forem diferentes de zero.

            Example::

                from pyvqnet.tensor import tensor
                from pyvqnet.tensor import QTensor
                shape = [2, 3]
                t = tensor.zeros(shape)
                t.fill_(1.0)
                flag = t.all()
                print(flag)

                # True

 
    .. py:method:: any()

            Retorna True se algum valor do QTensor for diferente de zero.

            :return: True se algum valor do QTensor for diferente de zero.

            Example::

                from pyvqnet.tensor import tensor
                from pyvqnet.tensor import QTensor

                shape = [2, 3]
                t = tensor.ones(shape)
                t.fill_(1.0)
                flag = t.any()
                print(flag)

                # True

 
    .. py:method:: fill_rand_binary_(v=0.5)

        Preenche um QTensor com valores amostrados aleatoriamente de uma distribuição binomial.

        Se os dados gerados aleatoriamente pela distribuição binomial forem maiores que o limiar de binarização, o número nas posições correspondentes do QTensor é definido como 1, caso contrário, 0.

        :param v: Limiar de binarização
        :return: None

        Example::

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

        Preenche um QTensor com valores amostrados aleatoriamente de uma distribuição uniforme com sinal.

        Fator de escala dos valores gerados pela distribuição uniforme com sinal.

        :param v: um valor escalar
        :return: None

        Example::

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

        Preenche um QTensor com valores amostrados aleatoriamente de uma distribuição uniforme.

        Fator de escala dos valores gerados pela distribuição uniforme.

        :param v: um valor escalar
        :return: None

        Example::

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

        Preenche um QTensor com valores amostrados aleatoriamente de uma distribuição normal.
        Média da distribuição normal. Desvio padrão da distribuição normal.
        Indica se deve usar o modo fast math ou não.

        :param m: média da distribuição normal
        :param s: desvio padrão da distribuição normal
        :param fast_math: True se usar fast-math
        :return: None

        Example::

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

        Inverte ou permuta os eixos de um array. Se new_dims = None, inverte as dimensões.

        :param new_dims: a nova ordem das dimensões (lista de inteiros).
        :return: QTensor resultante.

        Example::

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

        Altera a forma do tensor e retorna um novo QTensor.

        :param new_shape: a nova forma (lista de inteiros)
        :return: um novo QTensor

        Example::

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

        Altera a forma do QTensor atual internamente. Esta interface primeiro tenta transformar sem alterar os dados originais na memória. Se falhar, os dados atuais serão copiados para a nova memória.

        .. warning::

            Recomenda-se usar a interface reshape. Em alguns casos, a localização real da memória subjacente será copiada em vez de modificada internamente.

        :param new_shape: a nova forma (lista de inteiros)
        :return: None

        Example::

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

            Obtém os dados do QTensor como um array NumPy.

            :return: um array NumPy

            Example::


                from pyvqnet.tensor import tensor
                from pyvqnet.tensor import QTensor

                t = tensor.ones([3, 4])
                a = t.getdata()
                print(a)

                # [[1. 1. 1. 1.]
                #  [1. 1. 1. 1.]
                #  [1. 1. 1. 1.]]

 

    .. py:method:: __getitem__()

            O índice por fatia (slicing) do QTensor é suportado, assim como o uso do QTensor como índice de acesso avançado. Um novo QTensor será retornado.

            Os parâmetros start, stop e step podem ser separados por dois pontos, como start:stop:step, onde start, stop e step podem ser omitidos.

            Como um QTensor 1-D, a indexação ou fatia só pode ser feita em um único eixo.

            Como um QTensor 2-D e um QTensor multidimensional, a indexação ou fatia pode ser feita em múltiplos eixos.

            Se você usar um QTensor como índice para indexação avançada, veja numpy para `advanced indexing <https://docs.scipy.org/doc/numpy-1.10.1/reference/arrays.indexing.html>`_ .

            Se o seu QTensor como índice é o resultado de uma operação lógica, então você faz uma indexação booleana.

            .. note:: 
                
                Usamos uma forma de índice como a[3,4,1], mas a forma a[3][4][1] não é suportada.

            :param item: Um inteiro ou QTensor usado como índice.

            :return: Um novo QTensor.

            Example::

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

        O índice por fatia (slicing) do QTensor é suportado, assim como o uso do QTensor como índice de acesso avançado. Um novo QTensor será retornado.

        Os parâmetros start, stop e step podem ser separados por dois pontos, como start:stop:step, onde start, stop e step podem ser omitidos.

        Como um QTensor 1-D, a indexação ou fatia só pode ser feita em um único eixo.

        Como um QTensor 2-D e um QTensor multidimensional, a indexação ou fatia pode ser feita em múltiplos eixos.

        Se você usar um QTensor como índice para indexação avançada, veja numpy para `advanced indexing <https://docs.scipy.org/doc/numpy-1.10.1/reference/arrays.indexing.html>`_ .

        Se o seu QTensor como índice é o resultado de uma operação lógica, então você faz uma indexação booleana.

        .. note:: 
            
            Usamos uma forma de índice como a[3,4,1], mas a forma a[3][4][1] não é suportada.

        :param item: Um inteiro ou QTensor usado como índice

        :return: None


        Example::

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

        Clona o QTensor para o dispositivo GPU especificado.

        device especifica o dispositivo onde os dados internos são armazenados. Quando device >= DEV_GPU_0, os dados são armazenados na GPU.
        Se o seu computador tiver múltiplas GPUs, você pode designar diferentes dispositivos para armazenar dados.
        Por exemplo, device = DEV_GPU_1, DEV_GPU_2, DEV_GPU_3, ... indica armazenamento em GPUs com diferentes números de série.
        
        .. note::
            O QTensor não pode realizar cálculos em GPUs diferentes.
            Um erro Cuda será gerado se você tentar criar um QTensor em uma GPU cujo ID exceda o número máximo de GPUs verificadas.

        :param device: O dispositivo que está salvando o QTensor atualmente, padrão=DEV_GPU_0,

        device = pyvqnet.DEV_GPU_0, armazenado na primeira GPU, device = DEV_GPU_1,
        armazenado na segunda GPU, e assim por diante.

        :return: Clona o QTensor para o dispositivo GPU.

        Examples::

            from pyvqnet.tensor import QTensor
            a = QTensor([2])
            b = a.GPU()
            print(b.device)
            #1000

 

    .. py:method:: CPU()

        Clona o QTensor para o dispositivo CPU específico

        :return: Clona o QTensor para o dispositivo CPU.

        Examples::

            from pyvqnet.tensor import QTensor
            a = QTensor([2])
            b = a.CPU()
            print(b.device)
            # 0

 
    .. py:method:: toGPU(device: int = DEV_GPU_0)

        Move o QTensor para o dispositivo GPU especificado.

        device especifica o dispositivo onde os dados internos são armazenados. Quando device >= DEV_GPU, os dados são armazenados na GPU.
        Se o seu computador tiver múltiplas GPUs, você pode designar diferentes dispositivos para armazenar dados.
        Por exemplo, device = DEV_GPU_1, DEV_GPU_2, DEV_GPU_3, ... indica armazenamento em GPUs com diferentes números de série.

        .. note::

            O QTensor não pode realizar cálculos em GPUs diferentes. Um erro Cuda será gerado se você tentar criar um QTensor em uma GPU cujo ID exceda o número máximo de GPUs verificadas.

        :param device: O dispositivo que está salvando o QTensor atualmente, padrão=DEV_GPU_0. device = pyvqnet.DEV_GPU_0, armazenado na primeira GPU, device = DEV_GPU_1, armazenado na segunda GPU, e assim por diante.
        :return: QTensor movido para o dispositivo GPU.

        Examples::

            from pyvqnet.tensor import QTensor
            a = QTensor([2])
            a = a.toGPU()
            print(a.device)
            #1000


    
    .. py:method:: toCPU()

        Move o QTensor para a CPU

        :return: QTensor movido para o dispositivo CPU.

        Examples::

            from pyvqnet.tensor import QTensor
            a = QTensor([2])
            b = a.toCPU()
            print(b.device)
            # 0

    
    .. py:method:: isGPU()

        Verifica se os dados deste QTensor estão armazenados na memória da GPU.

        :return: True se os dados deste QTensor estiverem armazenados na memória da GPU.

        Examples::
        
            from pyvqnet.tensor import QTensor
            a = QTensor([2])
            a = a.isGPU()
            print(a)
            # False
 
    .. py:method:: isCPU()

        Verifica se os dados deste QTensor estão armazenados na memória da CPU.

        :return: True se os dados deste QTensor estiverem armazenados na memória da CPU.

        Examples::
        
            from pyvqnet.tensor import QTensor
            a = QTensor([2])
            a = a.isCPU()
            print(a)
            # True


Funções de Criação
*****************************************************


ones
==============================

.. py:function:: pyvqnet.tensor.ones(shape,device=0,dtype-None)

    Retorna um tensor preenchido com 1 com a forma especificada.

    :param shape: forma de entrada
    :param device: dispositivo de armazenamento, padrão 0, CPU.
    :param dtype: O tipo de dados do parâmetro, padrão None, usa o tipo de dados padrão: kfloat32, que representa um número de ponto flutuante de 32 bits.
    
    :return: QTensor de saída com a forma especificada.

    Example::

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

    Retorna um tensor preenchido com 1 com a mesma forma do QTensor de entrada.

    :param t: QTensor de entrada
    :param device: dispositivo de armazenamento, padrão 0, CPU.
    :param dtype: O tipo de dados do parâmetro, padrão None, usa o tipo de dados padrão: kfloat32, que representa um número de ponto flutuante de 32 bits.
    
    :return: QTensor de saída


    Example::

        from pyvqnet.tensor import tensor
        from pyvqnet.tensor import QTensor
        t = QTensor([1, 2, 3])
        x = tensor.ones_like(t)
        print(x)

        # [1, 1, 1]

full
==============================

.. py:function:: pyvqnet.tensor.full(shape, value, device=0, dtype=None)

    Cria um QTensor com a forma especificada e o preenche com o valor indicado.

    :param shape: forma do QTensor a ser criado
    :param value: valor para preencher o QTensor.
    :param device: dispositivo a ser usado, padrão = 0, usa dispositivo CPU.
    :param dtype: O tipo de dados do parâmetro, padrão None, usa o tipo de dados padrão: kfloat32, que representa um número de ponto flutuante de 32 bits.
    
    :return: QTensor de saída

    Example::

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

    Cria um QTensor com a forma do tensor de entrada e o preenche com o valor indicado.

    :param t: QTensor de entrada
    :param value: valor para preencher o QTensor.
    :param device: dispositivo a ser usado, padrão = 0, usa dispositivo CPU.
    :param dtype: O tipo de dados do parâmetro, padrão None, usa o tipo de dados padrão: kfloat32, que representa um número de ponto flutuante de 32 bits.
    
    :return: QTensor de saída

    Example::

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

    Retorna um tensor preenchido com 0 com a forma especificada.

    :param shape: forma do tensor
    :param device: dispositivo a ser usado, padrão = 0, usa dispositivo CPU
    :param dtype: O tipo de dados do parâmetro, padrão None, usa o tipo de dados padrão: kfloat32, que representa um número de ponto flutuante de 32 bits.
    
    :return: QTensor de saída

    Example::

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

    Retorna um tensor preenchido com 0 com a mesma forma do QTensor de entrada.

    :param t: QTensor de entrada
    :param device: dispositivo a ser usado, padrão = 0, usa dispositivo CPU
    :param dtype: O tipo de dados do parâmetro, padrão None, usa o tipo de dados padrão: kfloat32, que representa um número de ponto flutuante de 32 bits.
    
    :return: QTensor de saída

    Example::

        from pyvqnet.tensor import tensor
        from pyvqnet.tensor import QTensor
        t = QTensor([1, 2, 3])
        x = tensor.zeros_like(t)
        print(x)

        # [0, 0, 0]

arange
==============================

.. py:function:: pyvqnet.tensor.arange(start, end, step=1, device: int = 0,dtype=None, requires_grad=False)

    Cria um QTensor 1D com valores uniformemente espaçados dentro de um intervalo.

    :param start: início do intervalo
    :param end: fim do intervalo
    :param step: espaçamento entre valores
    :param device: dispositivo a ser usado, padrão = 0, usa dispositivo CPU
    :param dtype: O tipo de dados do parâmetro, padrão None, usa o tipo de dados padrão: kfloat32, que representa um número de ponto flutuante de 32 bits.
    :param requires_grad: indica se o gradiente do tensor deve ser rastreado, padrão: False
    :return: QTensor de saída

    Example::

        from pyvqnet.tensor import tensor
        from pyvqnet.tensor import QTensor
        t = tensor.arange(2, 30,4)
        print(t)

        # [ 2,  6, 10, 14, 18, 22, 26]

linspace
==============================

.. py:function:: pyvqnet.tensor.linspace(start, end, num, device: int = 0,dtype=None, requires_grad= False)

    Cria um QTensor 1D com valores uniformemente espaçados dentro de um intervalo.

    :param start: valor inicial
    :param end: valor final
    :param nums: número de amostras a serem geradas
    :param device: dispositivo a ser usado, padrão = 0, usa dispositivo CPU
    :param dtype: O tipo de dados do parâmetro, padrão None, usa o tipo de dados padrão: kfloat32, que representa um número de ponto flutuante de 32 bits.
    :param requires_grad: indica se o gradiente do tensor deve ser rastreado, padrão: False
    :return: QTensor de saída

    Example::

        from pyvqnet.tensor import tensor
        from pyvqnet.tensor import QTensor
        start, stop, steps = -2.5, 10, 10
        t = tensor.linspace(start, stop, steps)
        print(t)
        #[-2.5000000, -1.1111112, 0.2777777, 1.6666665, 3.0555553, 4.4444442, 5.8333330, 7.2222219, 8.6111107, 10]

logspace
==============================

.. py:function:: pyvqnet.tensor.logspace(start, end, num, base, device: int = 0,dtype=None,  requires_grad)

    Cria um QTensor 1D com valores uniformemente espaçados em uma escala logarítmica.

    :param start: ``base ** start`` é o valor inicial
    :param end: ``base ** end`` é o valor final da sequência
    :param nums: número de amostras a serem geradas
    :param base: a base do espaço logarítmico
    :param device: dispositivo a ser usado, padrão = 0, usa dispositivo CPU
    :param dtype: O tipo de dados do parâmetro, padrão None, usa o tipo de dados padrão: kfloat32, que representa um número de ponto flutuante de 32 bits.
    :param requires_grad: indica se o gradiente do tensor deve ser rastreado, padrão: False
    :return: QTensor de saída

    Example::

        from pyvqnet.tensor import tensor
        from pyvqnet.tensor import QTensor
        start, stop, num, base = 0.1, 1.0, 5, 10.0
        t = tensor.logspace(start, stop, num, base)
        print(t)

        # [1.2589254, 2.1134889, 3.5481336, 5.9566211, 10]

eye
==============================

.. py:function:: pyvqnet.tensor.eye(size, offset: int = 0, device=0,dtype=None)

    Cria um QTensor de tamanho size x size com 1 na diagonal e 0
    nas demais posições.

    :param size: tamanho do QTensor (quadrado) a ser criado
    :param offset: Índice da diagonal: 0 (padrão) refere-se à diagonal principal, um valor positivo refere-se a uma diagonal superior e um valor negativo a uma diagonal inferior.
    :param device: dispositivo a ser usado, padrão = 0, usa dispositivo CPU
    :param dtype: O tipo de dados do parâmetro, padrão None, usa o tipo de dados padrão: kfloat32, que representa um número de ponto flutuante de 32 bits.
    
    :return: QTensor de saída

    Example::

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


    Retorna uma visão parcial de :attr:`t` com os elementos da diagonal anexados como dimensões ao final da forma em relação a :attr:`dim1` e :attr:`dim2`.
    :attr:`offset` é o deslocamento da diagonal principal.

    :param t: tensor de entrada
    :param offset: deslocamento (0 significa diagonal principal, valores positivos significam a n-ésima diagonal acima da diagonal principal, valores negativos significam a n-ésima diagonal abaixo da diagonal principal)
    :param dim1: primeira dimensão para obter a diagonal. Padrão: 0.
    :param dim2: segunda dimensão para obter a diagonal. Padrão: 1.

    Example::

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

    Seleciona elementos da diagonal ou constrói um QTensor diagonal.

    Insira um QTensor 2-D e retorna um novo tensor 1D contendo os elementos da diagonal selecionados. Insira um QTensor 1-D e retorna um novo tensor 2D cujos elementos da diagonal selecionados são os valores de entrada e o restante é 0.

    :param t: QTensor de entrada
    :param k: deslocamento (0 para a diagonal principal, positivo para a n-ésima
        diagonal acima da principal, negativo para a n-ésima diagonal abaixo da
        principal)
    :return: QTensor de saída

    Example::

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

    Cria um QTensor com valores aleatórios uniformemente distribuídos.

    :param shape: forma do QTensor a ser criado
    :param min: valor mínimo da distribuição uniforme, padrão: 0.
    :param max: valor máximo da distribuição uniforme, padrão: 1.
    :param device: dispositivo a ser usado, padrão = 0, usa dispositivo CPU
    :param dtype: O tipo de dados do parâmetro, padrão None, usa o tipo de dados padrão: kfloat32, que representa um número de ponto flutuante de 32 bits.
    :param requires_grad: indica se o gradiente do tensor deve ser rastreado, padrão: False
    :return: QTensor de saída


    Example::

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

    Cria um QTensor com valores aleatórios normalmente distribuídos.

    :param shape: forma do QTensor a ser criado
    :param mean: valor médio da distribuição normal, padrão: 0.
    :param std: desvio padrão da distribuição normal, padrão: 1.
    :param device: dispositivo a ser usado, padrão = 0, usa dispositivo CPU
    :param dtype: O tipo de dados do parâmetro, padrão None, usa o tipo de dados padrão: kfloat32, que representa um número de ponto flutuante de 32 bits.
    :param requires_grad: indica se o gradiente do tensor deve ser rastreado, padrão: False
    :return: QTensor de saída

    Example::

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
    
    Cria uma distribuição binomial parametrizada por :attr:total_count e :attr:probs.

    :param total_counts: Número de tentativas de Bernoulli.
    :param probs: Probabilidades dos eventos.

    :return:
        QTensor para distribuição binomial.

    Example::

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

    Retorna um Tensor onde cada linha contém num_samples amostras indexadas.
    A partir da distribuição de probabilidade multinomial localizada na linha correspondente do tensor de entrada.

    :param t: Distribuição de probabilidade de entrada.
    :param num_samples: número de amostras.

    :return:
        índice das amostras de saída

    Examples::

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

    Retorna a matriz triangular superior do tensor de entrada t, com o restante definido como 0.

    :param t: entrada de um QTensor
    :param diagonal: O deslocamento, padrão = 0. Diagonal principal é 0, positivo é deslocado para cima e negativo é deslocado para baixo

    :return: saída de um QTensor

    Examples::

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

    Retorna a matriz triangular inferior do tensor de entrada t, com o restante definido como 0.

    :param t: entrada de um QTensor
    :param diagonal: O deslocamento, padrão = 0. Diagonal principal é 0, positivo é deslocado para cima e negativo é deslocado para baixo

    :return: saída de um QTensor

    Examples::

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


Funções Matemáticas
*****************************************************


floor
==============================

.. py:function:: pyvqnet.tensor.floor(t)

    Retorna um novo QTensor com o piso (floor) dos elementos de entrada, o maior inteiro menor ou igual a cada elemento.

    :param t: QTensor de entrada
    :return: QTensor de saída

    Example::

        from pyvqnet.tensor import tensor

        t = tensor.arange(-2.0, 2.0, 0.25)
        u = tensor.floor(t)
        print(u)

        # [-2, -2, -2, -2, -1, -1, -1, -1, 0, 0, 0, 0, 1, 1, 1, 1]

ceil
==============================

.. py:function:: pyvqnet.tensor.ceil(t)

    Retorna um novo QTensor com o teto (ceil) dos elementos de entrada, o menor inteiro maior ou igual a cada elemento.

    :param t: QTensor de entrada
    :return: QTensor de saída

    Example::

        from pyvqnet.tensor import tensor

        t = tensor.arange(-2.0, 2.0, 0.25)
        u = tensor.ceil(t)
        print(u)

        # [-2, -1, -1, -1, -1, -0, -0, -0, 0, 1, 1, 1, 1, 2, 2, 2]

round
==============================

.. py:function:: pyvqnet.tensor.round(t)

    Arredonda os valores do QTensor para o inteiro mais próximo.

    :param t: QTensor de entrada
    :return: QTensor de saída

    Example::

        from pyvqnet.tensor import tensor

        t = tensor.arange(-2.0, 2.0, 0.4)
        u = tensor.round(t)
        print(u)

        # [-2, -2, -1, -1, -0, -0, 0, 1, 1, 2]

sort
==============================

.. py:function:: pyvqnet.tensor.sort(t, axis: int, descending=False, stable=True)

    Ordena o QTensor ao longo do eixo

    :param t: QTensor de entrada
    :param axis: eixo de ordenação
    :param descending: ordem decrescente se True
    :param stable: indica se deve usar ordenação estável ou não
    :return: QTensor de saída

    Example::

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

    Retorna um array de índices com a mesma forma da entrada que indexa os dados ao longo do eixo fornecido em ordem classificada.

    :param t: QTensor de entrada
    :param axis: eixo de ordenação
    :param descending: ordem decrescente se True
    :param stable: indica se deve usar ordenação estável ou não
    :return: QTensor de saída

    Example::

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

    Retorna os k maiores elementos do tensor de entrada ao longo do eixo fornecido.

    Se if_descent for False, retorna os k menores elementos.

    :param t: entrada de um QTensor
    :param k: número de maiores ou menores elementos
    :param axis: eixo de ordenação, padrão = -1, o último eixo
    :param if_descent: ordem decrescente, padrão True

    :return: Um novo QTensor

    Examples::

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

    Retorna o índice dos k maiores elementos ao longo do eixo fornecido do tensor de entrada.

    Se if_descent for False, retorna o índice dos k menores elementos.

    :param t: entrada de um QTensor
    :param k: número de maiores ou menores elementos
    :param axis: eixo de ordenação, padrão = -1, o último eixo
    :param if_descent: ordem decrescente, padrão True

    :return: Um novo QTensor

    Examples::

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

    Soma dois QTensors elemento a elemento, equivalente a t1 + t2.

    :param t1: primeiro QTensor
    :param t2: segundo QTensor
    :return: QTensor de saída

    Example::

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

    Subtrai dois QTensors elemento a elemento, equivalente a t1 - t2.


    :param t1: primeiro QTensor
    :param t2: segundo QTensor
    :return: QTensor de saída

    Example::

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

    Multiplica dois QTensors elemento a elemento, equivalente a t1 * t2.

    :param t1: primeiro QTensor
    :param t2: segundo QTensor
    :return: QTensor de saída


    Example::

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

    Divide dois QTensors elemento a elemento, equivalente a t1 / t2.


    :param t1: primeiro QTensor
    :param t2: segundo QTensor
    :return: QTensor de saída


    Example::

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

    Soma todos os elementos no QTensor ao longo do eixo fornecido. Se axis = None, soma todos os elementos no QTensor.

    :param t: QTensor de entrada
    :param axis: eixo usado para soma, padrão None
    :param keepdims: indica se a dimensão do tensor de saída é mantida ou não. - padrão False
    :return: QTensor de saída


    Example::

        from pyvqnet.tensor import tensor
        from pyvqnet.tensor import QTensor
        t = QTensor(([1, 2, 3], [4, 5, 6]))
        x = tensor.sums(t)
        print(x)

        # [21]



cumsum
==============================

.. py:function:: pyvqnet.tensor.cumsum(t, axis=-1)

    Retorna a soma cumulativa dos elementos de entrada na dimensão do eixo.

    :param t: o QTensor de entrada
    :param axis: eixo de cálculo, padrão -1, usa o último eixo

    :return: QTensor de saída.

    Example::

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

    Obtém os valores médios no QTensor ao longo do eixo.

    :param t: o QTensor de entrada.
    :param axis: a dimensão a ser reduzida.
    :param keepdims: indica se a dimensão do QTensor de saída é mantida ou não, padrão False.
    :return: retorna o valor médio do QTensor de entrada.

    Example::

        from pyvqnet.tensor import tensor
        from pyvqnet.tensor import QTensor
        t = QTensor([[1, 2, 3], [4, 5, 6.0]])
        x = tensor.mean(t, axis=1)
        print(x)

        # [2., 5.]

median
==============================

.. py:function:: pyvqnet.tensor.median(t: pyvqnet.tensor.QTensor, axis=None, keepdims=False)

    Obtém o valor mediano no QTensor.

    :param t: o QTensor de entrada
    :param axis: um eixo para cálculo da mediana, padrão None
    :param keepdims: indica se a dimensão do QTensor de saída é mantida ou não, padrão False

    :return: Retorna a mediana dos valores no QTensor de entrada.

    Example::

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

    Obtém o valor do desvio padrão no QTensor.


    :param t: o QTensor de entrada
    :param axis: o eixo usado para calcular o desvio padrão, padrão None
    :param keepdims: indica se a dimensão do QTensor de saída é mantida ou não, padrão False
    :param unbiased: indica se deve usar a correção de Bessel, padrão True
    :return: Retorna o desvio padrão dos valores no QTensor de entrada

    Example::

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

    Obtém a variância no QTensor.


    :param t: o QTensor de entrada.
    :param axis: o eixo usado para calcular a variância, padrão None
    :param keepdims: indica se a dimensão do QTensor de saída é mantida ou não, padrão False.
    :param unbiased: indica se deve usar a correção de Bessel, padrão True.


    :return: Obtém a variância no QTensor.

    Example::

        from pyvqnet.tensor import tensor
        from pyvqnet.tensor import QTensor

        a = QTensor([[-0.8166, -1.3802, -0.3560]])
        a_var = tensor.var(a)
        print(a_var)

        # [0.2631305]

matmul
==============================

.. py:function:: pyvqnet.tensor.matmul(t1: pyvqnet.tensor.QTensor, t2: pyvqnet.tensor.QTensor)

    Multiplicação matricial de duas matrizes 2D, 3D, 4D.

    :param t1: primeiro QTensor
    :param t2: segundo QTensor
    :return: QTensor de saída

    Example::

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
        # ]

kron
=============================

.. py:function:: pyvqnet.tensor.kron(t1: pyvqnet.tensor.QTensor, t2: pyvqnet.tensor.QTensor)

    Calcula o produto de Kronecker de ``t1`` e ``t2``, expresso em :math:`\otimes` . Se ``t1`` é um tensor :math:`(a_0 \times a_1 \times \dots \times a_n)` e ``t2`` é um tensor :math:`(b_0 \times b_1 \times \dots \ times b_n)`, o resultado será um tensor :math:`(a_0*b_0 \times a_1*b_1 \times \dots \times a_n*b_n)` com as seguintes entradas:
    
    .. math::
          (\text{input} \otimes \text{other})_{k_0, k_1, \dots, k_n} =
              \text{input}_{i_0, i_1, \dots, i_n} * \text{other}_{j_0, j_1, \dots, j_n},

    onde :math:`k_t = i_t * b_t + j_t` é :math:`0 \leq t \leq n`.
    Se um tensor tiver menos dimensões que o outro, ele será descompactado até ter a mesma dimensionalidade.

    :param t1: O primeiro QTensor.
    :param t2: O segundo QTensor.
    
    :return: QTensor de saída.

    Example::

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
        #     [231. 242. 253. 264. 252. 264. 276. 288.]]]]



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
    
    Soma os produtos dos elementos dos operandos de entrada ao longo da dimensão especificada usando uma notação baseada na convenção de soma de Einstein.

    .. note::

        Esta função usa opt_einsum (https://optimized-einsum.readthedocs.io/en/stable/) para acelerar o cálculo ou reduzir o consumo de memória otimizando a ordem de contração. Esta otimização ocorre quando há pelo menos três entradas.

        Para `einsum` mais complexos, opt_einsum pode ser importado adicionalmente para calcular diretamente no QTensor.

    :param equation: O subscrito da soma de Einstein.
    :param operands: O tensor no qual a soma de Einstein será calculada.

    :return:

        O resultado QTensor.

    Example::

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

    Calcula o recíproco elemento a elemento do QTensor.

    :param t: QTensor de entrada
    :return: QTensor de saída

    Example::

        from pyvqnet.tensor import tensor
        from pyvqnet.tensor import QTensor

        t = tensor.arange(1, 10, 1)
        u = tensor.reciprocal(t)
        print(u)

        #[1, 0.5000000, 0.3333333, 0.2500000, 0.2000000, 0.1666667, 0.1428571, 0.1250000, 0.1111111]

sign
==============================

.. py:function:: pyvqnet.tensor.sign(t)

    Retorna um novo QTensor com os sinais dos elementos de entrada. A função sinal retorna -1 se t < 0, 0 se t == 0, 1 se t > 0.

    :param t: QTensor de entrada
    :return: QTensor de saída


    Example::

        from pyvqnet.tensor import tensor
        from pyvqnet.tensor import QTensor

        t = tensor.arange(-5, 5, 1)
        u = tensor.sign(t)
        print(u)

        # [-1, -1, -1, -1, -1, 0, 1, 1, 1, 1]


neg
==============================

.. py:function:: pyvqnet.tensor.neg(t: pyvqnet.tensor.QTensor)

    Negação unária dos elementos do QTensor.

    :param t: QTensor de entrada
    :return: QTensor de saída

    Example::

        from pyvqnet.tensor import tensor
        from pyvqnet.tensor import QTensor
        t = QTensor([1, 2, 3])
        x = tensor.neg(t)
        print(x)

        # [-1, -2, -3]

trace
==============================

.. py:function:: pyvqnet.tensor.trace(t, k: int = 0)

    Retorna a soma dos elementos da diagonal da matriz 2-D de entrada.

    :param t: QTensor 2-D de entrada
    :param k: deslocamento (0 para a diagonal principal, positivo para a n-ésima
        diagonal acima da principal, negativo para a n-ésima diagonal abaixo da
        principal)
    :return: a soma dos elementos da diagonal da matriz 2-D de entrada

    Example::

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

    Aplica a função exponencial a todos os elementos do QTensor de entrada.

    :param t: QTensor de entrada
    :return: QTensor de saída

    Example::

        from pyvqnet.tensor import tensor
        from pyvqnet.tensor import QTensor
        t = QTensor([1, 2, 3])
        x = tensor.exp(t)
        print(x)

        # [2.7182817, 7.3890562, 20.0855369]

acos
==============================

.. py:function:: pyvqnet.tensor.acos(t: pyvqnet.tensor.QTensor)

    Calcula o cosseno inverso elemento a elemento do QTensor.

    :param t: QTensor de entrada
    :return: QTensor de saída

    Example::

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

    Calcula o seno inverso elemento a elemento do QTensor.

    :param t: QTensor de entrada
    :return: QTensor de saída

    Example::

        from pyvqnet.tensor import tensor
        from pyvqnet.tensor import QTensor

        t = tensor.arange(-1, 1, .5)
        u = tensor.asin(t)
        print(u)

        #[-1.5707964, -0.5235988, 0, 0.5235988]

atan
==============================

.. py:function:: pyvqnet.tensor.atan(t: pyvqnet.tensor.QTensor)

    Calcula a tangente inversa elemento a elemento do QTensor.

    :param t: QTensor de entrada
    :return: QTensor de saída

    Example::

        from pyvqnet.tensor import tensor
        from pyvqnet.tensor import QTensor

        t = tensor.arange(-1, 1, .5)
        u = tensor.atan(t)
        print(u)

        # [-0.7853981, -0.4636476, 0.0000, 0.4636476]

sin
==============================

.. py:function:: pyvqnet.tensor.sin(t: pyvqnet.tensor.QTensor)

    Aplica a função seno a todos os elementos do QTensor de entrada.


    :param t: QTensor de entrada
    :return: QTensor de saída

    Example::

        from pyvqnet.tensor import tensor
        from pyvqnet.tensor import QTensor
        t = QTensor([1, 2, 3])
        x = tensor.sin(t)
        print(x)

        # [0.8414709, 0.9092974, 0.1411200]

cos
==============================

.. py:function:: pyvqnet.tensor.cos(t: pyvqnet.tensor.QTensor)

    Aplica a função cosseno a todos os elementos do QTensor de entrada.


    :param t: QTensor de entrada
    :return: QTensor de saída

    Example::

        from pyvqnet.tensor import tensor
        from pyvqnet.tensor import QTensor
        t = QTensor([1, 2, 3])
        x = tensor.cos(t)
        print(x)

        # [0.5403022, -0.4161468, -0.9899924]

tan 
==============================

.. py:function:: pyvqnet.tensor.tan(t: pyvqnet.tensor.QTensor)

    Aplica a função tangente a todos os elementos do QTensor de entrada.


    :param t: QTensor de entrada
    :return: QTensor de saída

    Example::

        from pyvqnet.tensor import tensor
        from pyvqnet.tensor import QTensor
        t = QTensor([1, 2, 3])
        x = tensor.tan(t)
        print(x)

        # [1.5574077, -2.1850397, -0.1425465]

tanh
==============================

.. py:function:: pyvqnet.tensor.tanh(t: pyvqnet.tensor.QTensor)

    Aplica a função tangente hiperbólica a todos os elementos do QTensor de entrada.

    :param t: QTensor de entrada
    :return: QTensor de saída

    Example::

        from pyvqnet.tensor import tensor
        from pyvqnet.tensor import QTensor
        t = QTensor([1, 2, 3])
        x = tensor.tanh(t)
        print(x)

        # [0.7615941, 0.9640275, 0.9950547]

sinh
==============================

.. py:function:: pyvqnet.tensor.sinh(t: pyvqnet.tensor.QTensor)

    Aplica a função seno hiperbólico a todos os elementos do QTensor de entrada.


    :param t: QTensor de entrada
    :return: QTensor de saída

    Example::

        from pyvqnet.tensor import tensor
        from pyvqnet.tensor import QTensor
        t = QTensor([1, 2, 3])
        x = tensor.sinh(t)
        print(x)

        # [1.1752011, 3.6268603, 10.0178747]

cosh
==============================

.. py:function:: pyvqnet.tensor.cosh(t: pyvqnet.tensor.QTensor)

    Aplica a função cosseno hiperbólico a todos os elementos do QTensor de entrada.


    :param t: QTensor de entrada
    :return: QTensor de saída

    Example::

        from pyvqnet.tensor import tensor
        from pyvqnet.tensor import QTensor
        t = QTensor([1, 2, 3])
        x = tensor.cosh(t)
        print(x)

        # [1.5430806, 3.7621955, 10.0676622]

power
==============================

.. py:function:: pyvqnet.tensor.power(t1: pyvqnet.tensor.QTensor, t2: pyvqnet.tensor.QTensor)

    Eleva o primeiro QTensor à potência do segundo QTensor.

    :param t1: primeiro QTensor
    :param t2: segundo QTensor
    :return: QTensor de saída

    Example::

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

    Aplica a função valor absoluto a todos os elementos do QTensor de entrada.

    :param t: QTensor de entrada
    :return: QTensor de saída

    Example::

        from pyvqnet.tensor import tensor
        from pyvqnet.tensor import QTensor
        t = QTensor([1, -2, 3])
        x = tensor.abs(t)
        print(x)

        # [1, 2, 3]

log
==============================

.. py:function:: pyvqnet.tensor.log(t: pyvqnet.tensor.QTensor)

    Aplica a função log (ln) a todos os elementos do QTensor de entrada.

    :param t: QTensor de entrada
    :return: QTensor de saída

    Example::

        from pyvqnet.tensor import tensor
        from pyvqnet.tensor import QTensor
        t = QTensor([1, 2, 3])
        x = tensor.log(t)
        print(x)

        # [0, 0.6931471, 1.0986123]

log_softmax
==============================

.. py:function:: pyvqnet.tensor.log_softmax(t, axis=-1)
    
    Calcula sequencialmente os resultados da função softmax e da função log no eixo especificado.

    :param t: QTensor de entrada.
    :param axis: O eixo usado para calcular softmax, o padrão é -1.

    :return: QTensor de saída.

    Example::

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

    Aplica a função raiz quadrada a todos os elementos do QTensor de entrada.


    :param t: QTensor de entrada
    :return: QTensor de saída

    Example::

        from pyvqnet.tensor import tensor
        from pyvqnet.tensor import QTensor
        t = QTensor([1, 2, 3])
        x = tensor.sqrt(t)
        print(x)

        # [1, 1.4142135, 1.7320507]

square
==============================

.. py:function:: pyvqnet.tensor.square(t: pyvqnet.tensor.QTensor)

    Aplica a função quadrado a todos os elementos do QTensor de entrada.


    :param t: QTensor de entrada
    :return: QTensor de saída

    Example::

        from pyvqnet.tensor import tensor
        from pyvqnet.tensor import QTensor
        t = QTensor([1, 2, 3])
        x = tensor.square(t)
        print(x)
        # [1, 4, 9]



eigh
==============================

.. py:function:: pyvqnet.tensor.eigh(t: QTensor)
 
    Retorna os autovalores e autovetores de uma matriz Hermitiana complexa (simétrica conjugada) ou matriz simétrica real.

    Retorna dois objetos, um array 1D contendo os autovalores,
    e uma matriz quadrada 2D ou matriz (dependendo do tipo de entrada) dos autovetores correspondentes (em colunas).

    :param t: QTensor de entrada.
    :return:

        Retorna autovalores e autovetores

    Examples::

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

    Calcula a norma F do tensor no QTensor de entrada ao longo do eixo definido por axis.
    Se axis for None, retorna a norma F de todos os elementos.

    :param t: QTensor de entrada.
    :param axis: O eixo usado para calcular a norma F, o padrão é None.
    :param keepdims: Se o tensor de saída preserva a dimensionalidade reduzida. O padrão é False.
    :return: Saída de um QTensor ou valor da norma F.


    Example::

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



Funções Lógicas
**************************

maximum
==============================

.. py:function:: pyvqnet.tensor.maximum(t1: pyvqnet.tensor.QTensor, t2: pyvqnet.tensor.QTensor)

    Máximo elemento a elemento de dois tensores.


    :param t1: primeiro QTensor
    :param t2: segundo QTensor
    :return: QTensor de saída

    Example::

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

    Mínimo elemento a elemento de dois tensores.


    :param t1: primeiro QTensor
    :param t2: segundo QTensor
    :return: QTensor de saída

    Example::

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

    Retorna os elementos mínimos do QTensor de entrada ao longo do eixo fornecido.
    se axis == None, retorna o valor mínimo de todos os elementos no tensor.

    :param t: QTensor de entrada
    :param axis: eixo usado para min, padrão None
    :param keepdims: indica se a dimensão do tensor de saída é mantida ou não. - padrão False
    :return: QTensor de saída

    Example::

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

    Retorna os elementos máximos do QTensor de entrada ao longo do eixo fornecido.
    se axis == None, retorna o valor máximo de todos os elementos no tensor.

    :param t: QTensor de entrada
    :param axis: eixo usado para max, padrão None
    :param keepdims: indica se a dimensão do tensor de saída é mantida ou não. - padrão False
    :return: QTensor de saída

    Example::

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

    Limita o QTensor de entrada aos valores mínimo e máximo.

    :param t: QTensor de entrada
    :param min_val: valor mínimo
    :param max_val: valor máximo
    :return: QTensor de saída

    Example::

        from pyvqnet.tensor import tensor
        from pyvqnet.tensor import QTensor
        t = QTensor([2, 4, 6])
        x = tensor.clip(t, 3, 8)
        print(x)

        # [3, 4, 6]

where
==============================

.. py:function:: pyvqnet.tensor.where(condition: pyvqnet.tensor.QTensor, t1: pyvqnet.tensor.QTensor, t2: pyvqnet.tensor.QTensor)

    Retorna elementos escolhidos de x ou y dependendo da condição.

    :param condition: tensor de condição, precisa ter tipo de dados kbool.
    :param t1: QTensor do qual extrair elementos se a condição for atendida, padrão None
    :param t2: QTensor do qual extrair elementos se a condição não for atendida, padrão None
    :return: QTensor de saída

    Example::

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

    Retorna um QTensor contendo os índices dos elementos diferentes de zero.

    :param t: QTensor de entrada
    :return: QTensor de saída contendo os índices dos elementos diferentes de zero.

    Example::
    
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

    Testa elemento a elemento se é finito (não é infinito nem NaN).

    :param t: QTensor de entrada
    :return: QTensor de saída, que retorna True quando o elemento na posição correspondente atende à condição, caso contrário retorna False.

    Example::

        from pyvqnet.tensor import tensor
        from pyvqnet.tensor import QTensor

        t = QTensor([1, float('inf'), 2, float('-inf'), float('nan')])
        flag = tensor.isfinite(t)
        print(flag)

        #[ True False  True False False]

isinf
==============================

.. py:function:: pyvqnet.tensor.isinf(t)

    Testa elemento a elemento se é infinito positivo ou negativo.

    :param t: QTensor de entrada
    :return: QTensor de saída, que retorna True quando o elemento na posição correspondente atende à condição, caso contrário retorna False.
    
    Example::

        from pyvqnet.tensor import tensor
        from pyvqnet.tensor import QTensor

        t = QTensor([1, float('inf'), 2, float('-inf'), float('nan')])
        flag = tensor.isinf(t)
        print(flag)

        # [False  True False  True False]

isnan
==============================

.. py:function:: pyvqnet.tensor.isnan(t)

    Testa elemento a elemento se é NaN.

    :param t: QTensor de entrada
    :return: QTensor de saída, que retorna True quando o elemento na posição correspondente atende à condição, caso contrário retorna False.
    
    Example::

        from pyvqnet.tensor import tensor
        from pyvqnet.tensor import QTensor

        t = QTensor([1, float('inf'), 2, float('-inf'), float('nan')])
        flag = tensor.isnan(t)
        print(flag)

        # [False False False False  True]

isneginf
==============================

.. py:function:: pyvqnet.tensor.isneginf(t)

    Testa elemento a elemento se é infinito negativo.

    :param t: QTensor de entrada
    :return: QTensor de saída, que retorna True quando o elemento na posição correspondente atende à condição, caso contrário retorna False.
    
    Example::

        from pyvqnet.tensor import tensor
        from pyvqnet.tensor import QTensor

        t = QTensor([1, float('inf'), 2, float('-inf'), float('nan')])
        flag = tensor.isneginf(t)
        print(flag)

        # [False False False  True False]

isposinf
==============================

.. py:function:: pyvqnet.tensor.isposinf(t)

    Testa elemento a elemento se é infinito positivo.

    :param t: QTensor de entrada
    :return: QTensor de saída, que retorna True quando o elemento na posição correspondente atende à condição, caso contrário retorna False.
    
    Example::

        from pyvqnet.tensor import tensor
        from pyvqnet.tensor import QTensor

        t = QTensor([1, float('inf'), 2, float('-inf'), float('nan')])
        flag = tensor.isposinf(t)
        print(flag)

        # [False  True False False False]

logical_and
==============================

.. py:function:: pyvqnet.tensor.logical_and(t1, t2)

    Calcula o valor verdade de ``t1`` e ``t2`` elemento a elemento.

    :param t1: QTensor de entrada
    :param t2: QTensor de entrada
    :return: QTensor de saída, que retorna True quando o elemento na posição correspondente atende à condição, caso contrário retorna False.
    
    Example::

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

    Calcula o valor verdade de ``t1 or t2`` elemento a elemento.

    :param t1: QTensor de entrada
    :param t2: QTensor de entrada
    :return: QTensor de saída, que retorna True quando o elemento na posição correspondente atende à condição, caso contrário retorna False.

    Example::

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

    Calcula o valor verdade de ``not t`` elemento a elemento.

    :param t: QTensor de entrada
    :return: QTensor de saída, que retorna True quando o elemento na posição correspondente atende à condição, caso contrário retorna False.

    Example::

        from pyvqnet.tensor import tensor
        from pyvqnet.tensor import QTensor

        a = QTensor([0, 1, 10, 0])
        flag = tensor.logical_not(a)
        print(flag)

        # [ True False False  True]

logical_xor
==============================

.. py:function:: pyvqnet.tensor.logical_xor(t1, t2)

    Calcula o valor verdade de ``t1 xor t2`` elemento a elemento.

    :param t1: QTensor de entrada
    :param t2: QTensor de entrada

    :return: QTensor de saída, que retorna True quando o elemento na posição correspondente atende à condição, caso contrário retorna False.

    Example::

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

    Retorna o valor verdade de ``t1 > t2`` elemento a elemento.


    :param t1: QTensor de entrada
    :param t2: QTensor de entrada
    :return: QTensor de saída, que retorna True quando o elemento na posição correspondente atende à condição, caso contrário retorna False.

    Example::

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

    Retorna o valor verdade de ``t1 >= t2`` elemento a elemento.

    :param t1: QTensor de entrada
    :param t2: QTensor de entrada
    :return: QTensor de saída, que retorna True quando o elemento na posição correspondente atende à condição, caso contrário retorna False.

    Example::

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

    Retorna o valor verdade de ``t1 < t2`` elemento a elemento.

    :param t1: QTensor de entrada
    :param t2: QTensor de entrada
    :return: QTensor de saída, que retorna True quando o elemento na posição correspondente atende à condição, caso contrário retorna False.

    Example::

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

    Retorna o valor verdade de ``t1 <= t2`` elemento a elemento.

    :param t1: QTensor de entrada
    :param t2: QTensor de entrada
    :return: QTensor de saída, que retorna True quando o elemento na posição correspondente atende à condição, caso contrário retorna False.

    Example::

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

    Retorna o valor verdade de ``t1 == t2`` elemento a elemento.

    :param t1: QTensor de entrada
    :param t2: QTensor de entrada
    :return: QTensor de saída, que retorna True quando o elemento na posição correspondente atende à condição, caso contrário retorna False.
    
    Example::

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

    Retorna o valor verdade de ``t1 != t2`` elemento a elemento.

    :param t1: QTensor de entrada
    :param t2: QTensor de entrada
    :return: QTensor de saída, que retorna True quando o elemento na posição correspondente atende à condição, caso contrário retorna False.
    
    Example::

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
 
    Calcula o AND bit a bit de dois elementos QTensor.

    :param t1: QTensor de entrada t1. Apenas inteiros ou booleanos são entradas válidas.
    :param t2: QTensor de entrada t2. Apenas inteiros ou booleanos são entradas válidas.
    :return:
        QTensor resultante

    Example::

        from pyvqnet.tensor import *
        import numpy as np
        from pyvqnet.dtype import *
        powers_of_two = 1 << np.arange(14, dtype=np.int64)[::-1]
        samples = tensor.QTensor([23],dtype=kint8)
        samples = samples.unsqueeze(-1)
        states_sampled_base_ten = samples & tensor.QTensor(powers_of_two,dtype = samples.dtype, device = samples.device)
        print(states_sampled_base_ten)
        #[[ 0, 0, 0, 0, 0, 0, 0, 0, 0,16, 0, 4, 2, 1]]


Operações Matriciais
**********************

select
==============================

.. py:function:: pyvqnet.tensor.select(t: pyvqnet.tensor.QTensor, index)

    Retorna QTensor no QTensor no eixo fornecido. A operação a seguir obtém o mesmo valor de resultado.

    :param t: QTensor de entrada
    :param index: uma string contendo a dimensão de saída
    :return: QTensor de saída

    Example::

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

    Sujeito a certas restrições, arrays menores são distribuídos em arrays maiores para que tenham formas compatíveis. Esta interface pode realizar diferenciação automática nos tensores de parâmetros de entrada.

    Referência https://numpy.org/doc/stable/user/basics.broadcasting.html

    :param t1: QTensor de entrada 1
    :param t2: QTensor de entrada 2

    :return t11: t1 com nova forma de broadcast.
    :return t22: t2 com nova forma de broadcast.

    Example::

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

    Concatena os QTensors de entrada ao longo do eixo e retorna um novo QTensor.

    :param args: lista de QTensors de entrada
    :param axis: dimensão para concatenar. Deve estar entre 0 e o número de dimensões dos tensores a serem concatenados.
    :return: QTensor de saída

    Example::

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

    Junta uma sequência de arrays ao longo de um novo eixo, retorna um novo QTensor.

    :param QTensors: lista contendo QTensors
    :param axis: dimensão a ser inserida. Deve estar entre 0 e o número de dimensões dos tensores empilhados.
    :return: QTensor de saída

    Example::

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

    Inverte ou permuta os eixos de um array.

    :param t: QTensor de entrada
    :param dim: a nova ordem das dimensões (lista de inteiros)
    :return: QTensor de saída

    Example::

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

    Transpõe os eixos de um array. Se dim = None, inverte as dimensões. Esta função é igual a permute.

    :param t: QTensor de entrada
    :param dim: a nova ordem das dimensões (lista de inteiros)
    :return: QTensor de saída

    Example::

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

    Constrói um QTensor repetindo o QTensor o número de vezes dado por reps.

    Se reps tem comprimento d, o QTensor resultante terá dimensão max(d, t.ndim).

    Se t.ndim < d, t é expandido para d-dimensional inserindo novos eixos a partir da dimensão inicial.
    Assim, um array de forma (3,) é promovido para (1, 3) para replicação 2-D, ou forma (1, 1, 3) para replicação 3-D.

    Se t.ndim > d, reps é expandido para t.ndim inserindo 1's.

    Assim, para um t de forma (2, 3, 4, 5), um reps de (4, 3) é tratado como (1, 1, 4, 3).

    :param t: QTensor de entrada
    :param reps: o número de repetições por dimensão.
    :return: um novo QTensor

    Example::

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

    Remove eixos de comprimento um.

    :param t: QTensor de entrada
    :param axis: eixo para squeeze, se axis = -1, remove todas as dimensões que têm tamanho 1.
    :return: QTensor de saída

    Example::

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

    Retorna um novo QTensor com uma dimensão de tamanho um inserida na posição especificada.

    :param t: QTensor de entrada
    :param axis: eixo para unsqueeze, onde a dimensão será inserida.
    :return: QTensor de saída

    Example::

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

    Move as dimensões de `t` das posições em `source` para as posições em `destination`.

    Outras dimensões de `t` que não são explicitamente movidas mantêm sua ordem original e aparecem em posições não especificadas em `destination`.

    :param t: QTensor de entrada.
    :param source: (inteiro ou tupla de inteiros) As posições originais das dimensões a serem movidas. Estas posições devem ser únicas.
    :param destination: (inteiro ou tupla de inteiros) As posições de destino para cada dimensão original. Estas posições também devem ser únicas.

    :return:
        Novo QTensor


    Example::

        from pyvqnet import QTensor,tensor
        a = tensor.arange(0,24).reshape((2,3,4))
        b = tensor.moveaxis(a,(1, 2), (0, 1))
        print(b.shape)


swapaxis
==============================

.. py:function:: pyvqnet.tensor.swapaxis(t, axis1: int, axis2: int)

    Troca dois eixos de um array. As dimensões fornecidas axis1 e axis2 são permutadas.

    :param t: QTensor de entrada
    :param axis1: Primeiro eixo.
    :param axis2: Posição de destino para o eixo original. Estas também devem ser únicas.
    :return: QTensor de saída

    Example::

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

    Se mask == 1, preenche com o valor especificado. A forma da máscara deve ser compatível para broadcast com a forma do QTensor de entrada.

    :param t: QTensor de entrada
    :param mask: Um QTensor
    :param value: valor especificado
    :return: Um QTensor

    Examples::

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

    Achata o QTensor da dimensão start até a dimensão end.

    :param t: QTensor de entrada
    :param start: dimensão inicial, padrão = 0, começa da primeira dimensão.
    :param end: dimensão final, padrão = -1, termina na última dimensão.
    :return: QTensor de saída

    Example::

        from pyvqnet.tensor import tensor
        from pyvqnet.tensor import QTensor
        t = QTensor([1, 2, 3])
        x = tensor.flatten(t)
        print(x)

        # [1, 2, 3]


reshape
==============================

.. py:function:: pyvqnet.tensor.reshape(t: pyvqnet.tensor.QTensor,new_shape)

    Altera a forma do QTensor, retorna um QTensor com a nova forma

    :param t: QTensor de entrada.
    :param new_shape: nova forma

    :return: um QTensor com a nova forma.

    Example::

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
    
    Inverte o QTensor ao longo do eixo especificado, retornando um novo tensor.

    :param t: QTensor de entrada.
    :param flip_dims: O eixo ou lista de eixos para inverter.

    :return: QTensor de saída.

    Example::

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

    Coleta valores ao longo do eixo especificado por 'dim'.

    Para tensores 3-D, a saída é especificada por:

    .. math::

        out[i][j][k] = t[index[i][j][k]][j][k] , if dim == 0 \\

        out[i][j][k] = t[i][index[i][j][k]][k] , if dim == 1 \\

        out[i][j][k] = t[i][j][index[i][j][k]] , if dim == 2 \\

    :param t: QTensor de entrada.
    :param dim: O eixo de agregação.
    :param index: QTensor de índice, deve ter o mesmo tamanho de dimensão que a entrada.

    :return: o resultado agregado

    Example::

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

    Escreve todos os valores no tensor src na entrada nos índices especificados no tensor de índices.

    Para tensores 3-D, a saída é especificada por:

    .. math::

        input[indices[i][j][k]][j][k] = src[i][j][k] , if dim == 0 \\
        input[i][indices[i][j][k]][k] = src[i][j][k] , if dim == 1 \\
        input[i][j][indices[i][j][k]] = src[i][j][k] , if dim == 2 \\

    :param input: QTensor de entrada.
    :param dim: Eixo de dispersão.
    :param indices: QTensor de índice, deve ter o mesmo tamanho de dimensão que a entrada.
    :param src: O tensor de origem a ser disperso.

    Example::

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

    Sujeito a certas restrições, o array t é "transmitido" (broadcast) para a forma de referência para que tenham formas compatíveis.

    https://numpy.org/doc/stable/user/basics.broadcasting.html

    :param t: QTensor de entrada
    :param ref: Forma de referência.
    
    :return: O QTensor de t recém-transmitido.

    Example::

        from pyvqnet.tensor.tensor import QTensor
        from pyvqnet.tensor import *
        ref = [2,3,4]
        a = ones([4])
        b = tensor.broadcast_to(a,ref)
        print(b.shape)
        #[2, 3, 4]



Funções Utilitárias
*****************************************************


to_tensor
==============================

.. py:function:: pyvqnet.tensor.to_tensor(x)

    Converte o array de entrada para QTensor, se ainda não for.

    :param x: inteiro, float ou numpy.array
    :return: QTensor de saída

    Example::

        from pyvqnet.tensor import tensor
        t = tensor.to_tensor(10.0)
        print(t)
        # [10]


pad_sequence
==============================

.. py:function:: pyvqnet.tensor.pad_sequence(qtensor_list, batch_first=False, padding_value=0)

    Preenche uma lista de tensores de comprimento variável com ``padding_value``. ``pad_sequence`` empilha listas de tensores ao longo de novas dimensões e as preenche para que tenham o mesmo comprimento.
    A entrada é uma sequência de listas de tamanho ``L x *``. L é de comprimento variável.

    :param qtensor_list: `list[QTensor]` - lista de sequências de comprimento variável.
    :param batch_first: 'bool' - Se True, a saída será ``tamanho do lote x comprimento da sequência mais longa x *``, caso contrário ``comprimento da sequência mais longa x tamanho do lote x *``. Padrão: False.
    :param padding_value: 'float' - valor de preenchimento. Valor padrão: 0.

    :return:
         Se batch_first for ``False``, o tamanho do tensor é ``tamanho do lote x comprimento da sequência mais longa x *``.
         Caso contrário, o tamanho do tensor é ``comprimento da sequência mais longa x tamanho do lote x *``.

    Examples::

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
    
    Preenche um lote de sequências empacotadas de comprimento variável. É o inverso de `pack_pad_sequence`.
    Quando ``batch_first`` é True, retorna um tensor de forma ``B x T x *``, caso contrário retorna ``T x B x *``.
    Onde `T` é o comprimento da sequência mais longa e `B` é o tamanho do lote.

    :param sequence: 'QTensor' - os dados a serem processados.
    :param batch_first: 'bool' - Se ``True``, o lote será a primeira dimensão da entrada. Valor padrão: False.
    :param padding_value: 'bool' - valor de preenchimento. Padrão: 0.
    :param total_length: 'bool' - Se não for ``None``, a saída será preenchida até o comprimento :attr:`total_length`. Padrão: None.
    :return:
        Uma tupla de tensores contendo as sequências preenchidas e uma lista de comprimentos para cada sequência no lote. Os elementos do lote serão reordenados em sua ordem original.
    
    Examples::

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
    
    Empacota um Tensor contendo sequências preenchidas de comprimento variável. Se batch_first for True, a `entrada` deve ter a forma [tamanho do lote, comprimento, ...], caso contrário, forma [comprimento, tamanho do lote, ...].

    Para sequências não ordenadas, use ``enforce_sorted`` como False. Se :attr:`enforce_sorted` for ``True``, as sequências devem ser ordenadas em ordem decrescente por comprimento.
    
    :param input: 'QTensor' - lotes de sequências de comprimento variável para preenchimento.
    :param lengths: 'list' - lista de comprimentos de sequência para cada elemento
         do lote.
    :param batch_first: 'bool' - se ``True``, espera-se que a entrada esteja no formato ``B x T x *``
         , padrão: False.
    :param enforce_sorted: 'bool' - se ``True``, a entrada deve
         conter sequências em ordem decrescente de comprimento. Se ``False``, a entrada será ordenada incondicionalmente. Padrão: True.

    :return: Um objeto :class:`PackedSequence`.

    Examples::

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
    
    Realiza uma convolução 2D em uma imagem de entrada composta por múltiplos planos de entrada.

    :param x: Tensor de entrada 4D.
    :param weight: Tensor do kernel 4D.

    :param stride: `tuple` - passo, padrão (1, 1)
    :param padding: Preenchimento, controla a quantidade de padding na entrada. Pode ser uma string {'valid', 'same'} ou uma tupla de inteiros especificando a quantidade de padding implícito a ser aplicado à entrada, padrão (0,0).
    :param dilation: `tuple` - Espaçamento entre elementos do kernel. Padrão: (0,0)
    :param groups: `int` - Número de grupos. Valor padrão: 1 

    :return: qtensor 


    Examples:: 

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

    Desabilita o registro de nós de retropropagação durante o cálculo direto.

    Example::

        import pyvqnet.tensor as tensor
        from pyvqnet import no_grad

        with no_grad():
            x = tensor.QTensor([1.0, 2.0, 3.0],requires_grad=True)
            y = tensor.tan(x)
            y.backward()
        """
        RuntimeError: The output tensor does not require gradients (output.requires_grad == False). This may occur if you used a non-autograd function in the forward pass. To enable gradient computation, ensure that all operations are performed on tensors with requires_grad=True, or use autograd-compatible functions.
        """
