.. _qtensor_api:

Modulo QTensor
###########################

Il machine learning quantistico di VQNet utilizza la struttura dati QTensor tramite interfaccia Python. QTensor supporta le comuni operazioni su matrici multidimensionali, incluse funzioni di creazione, funzioni matematiche, funzioni logiche, trasformazioni di matrici, ecc.




Funzioni e Attributi di QTensor
******************************************


QTensor
==============================

.. py:class:: pyvqnet.tensor.tensor.QTensor(data, requires_grad=False, nodes=None, device=0, dtype=None, name='')

    Wrapper della struttura dati con costruzione dinamica del grafo computazionale
    e differenziazione automatica.

    :param data: _core.Tensor o array numpy che rappresenta un QTensor
    :param requires_grad: indica se tracciare il gradiente del tensore, default False
    :param nodes: lista dei successori nel grafo computazionale, default None
    :param device: dispositivo corrente per salvare QTensor, default = 0, usa CPU.
    :param dtype: tipo di dato del parametro, default None, usa il tipo di dato predefinito: kfloat32, che rappresenta un numero a virgola mobile a 32 bit.
    :param name: nome del QTensor, default: "".

    :return: QTensor di output


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

        Restituisce il numero di dimensioni di un tensore.

        :return: Il numero di dimensioni di un tensore.

        Example::

            from pyvqnet.tensor import QTensor

            a = QTensor([2.0, 3.0, 4.0, 5.0], requires_grad=True)
            print(a.ndim)

            # 1

    .. py:attribute:: shape

        Restituisce le dimensioni di un tensore

        :return: Una lista delle dimensioni del tensore

        Example::

            from pyvqnet.tensor import QTensor

            a = QTensor([2.0, 3.0, 4.0, 5.0], requires_grad=True)
            print(a.shape)

            # [4]

    .. py:attribute:: size

        Restituisce il numero di elementi di un tensore.

        :return: Il numero di elementi di un tensore.

        Example::

            from pyvqnet.tensor import QTensor

            a = QTensor([2.0, 3.0, 4.0, 5.0], requires_grad=True)
            print(a.size)

            # 4

    .. py:method:: numel

        Restituisce il numero di elementi in un tensore.

        :return: Il numero di elementi in un tensore.

        Example::

            from pyvqnet.tensor import QTensor

            a = QTensor([2.0, 3.0, 4.0, 5.0], requires_grad=True)
            print(a.numel())

            # 4

    .. py:attribute:: dtype

        Restituisce il tipo di dato di un tensore.

        I tipi di dati supportati sono i seguenti:

            =========================================  ===============================
            dtype                                      descrizione
            =========================================  ===============================
            ``pyvqnet.kbool``                          Variabile booleana
            ``pyvqnet.kuint8``                         Intero a 8 bit (senza segno)
            ``pyvqnet.kint8``                          Intero a 8 bit (con segno)
            ``pyvqnet.kint16``                         Intero a 16 bit (con segno)
            ``pyvqnet.kint32``                         Intero a 32 bit (con segno)
            ``pyvqnet.kint64``                         Intero a 64 bit (con segno)
            ``pyvqnet.kfloat32``                       Virgola mobile a 32 bit, vedi https://en.wikipedia.org/wiki/IEEE_754
            ``pyvqnet.kfloat64``                       Virgola mobile a 64 bit, vedi https://en.wikipedia.org/wiki/IEEE_754
            ``pyvqnet.kcomplex64``                     Numero complesso a 64 bit, composto da due `float32`
            ``pyvqnet.kcomplex128``                    Numero complesso a 128 bit, composto da due `float64`
            ``pyvqnet.kbfloat16``                      Virgola mobile a 16 bit, talvolta chiamato formato Brain floating point, con allocazione di 1 bit di segno, 8 bit di esponente e 7 bit di mantissa
            =========================================  ===============================

        :return: Il tipo di dato del tensore.

        Example::

            from pyvqnet.tensor import QTensor

            a = QTensor([2, 3, 4, 5])
            print(a.dtype)
            # 4


    .. py:method:: zero_grad()

        Azzera il gradiente. Verrà utilizzato dall'ottimizzatore nel processo di ottimizzazione.

        :return: None

        Example::

            from pyvqnet.tensor import tensor
            from pyvqnet.tensor import QTensor
            t3  =  QTensor([2.0,3.0,4.0,5.0],requires_grad = True)
            t3.zero_grad()
            print(t3.grad)

            # [0, 0, 0, 0]


 
    .. py:method:: backward(grad=None)

        Calcola il gradiente del QTensor corrente.

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

        Copia i propri dati in un nuovo numpy.array.

        :return: un nuovo numpy.array contenente i dati del QTensor

        .. note::

            numpy non supporta il tipo bfloat16, è necessario convertire in altri tipi di dati supportati da numpy come float32 prima di chiamare questa interfaccia.

        Example::

            from pyvqnet.tensor import tensor
            from pyvqnet.tensor import QTensor
            t3  =  QTensor([2.0,3.0,4.0,5.0],requires_grad = True)
            t4 = t3.to_numpy()
            print(t4)

            # [2. 3. 4. 5.]

 
    .. py:method:: item()

            Restituisce l'unico elemento dal QTensor. Genera 'RuntimeError' se QTensor ha piu' di 1 elemento.

            :return: unico dato di questo oggetto

            Example::

                from pyvqnet.tensor import tensor

                t = tensor.ones([1])
                print(t.item())

                # 1.0

 
    .. py:method:: argmax(*kargs)

        Restituisce gli indici del valore massimo di tutti gli elementi nel QTensor di input, oppure
        Restituisce gli indici dei valori massimi di un QTensor lungo una dimensione.

        :param dim: dim (int) – la dimensione da ridurre, accetta solo un singolo asse. se dim == None, restituisce gli indici del valore massimo di tutti gli elementi nel tensore di input. L'intervallo valido per dim è [-R, R), dove R è ndim dell'input. quando dim < 0, funziona come dim + R.
        :param keepdims: indica se la dimensione del QTensor di output viene mantenuta o meno.

        :return: gli indici del valore massimo nel QTensor di input.

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

        Restituisce gli indici del valore minimo di tutti gli elementi nel QTensor di input, oppure
        Restituisce gli indici dei valori minimi di un QTensor lungo una dimensione.

        :param dim: dim (int) – la dimensione da ridurre, accetta solo un singolo asse. se dim == None, restituisce gli indici del valore minimo di tutti gli elementi nel tensore di input. L'intervallo valido per dim è [-R, R), dove R è ndim dell'input. quando dim < 0, funziona come dim + R.
        :param keepdims: indica se la dimensione del QTensor di output viene mantenuta o meno.

        :return: gli indici del valore minimo nel QTensor di input.

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

            Riempie il QTensor con il valore specificato, in-place.

            :param v: un valore scalare
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

            Restituisce True se tutti i valori del QTensor sono diversi da zero.

            :return: True, se tutti i valori del QTensor sono diversi da zero.

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

            Restituisce True se qualsiasi valore del QTensor e' diverso da zero.

            :return: True, se qualsiasi valore del QTensor e' diverso da zero.

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

        Riempie un QTensor con valori campionati casualmente da una distribuzione binomiale.

        Se i dati generati casualmente dopo la distribuzione binomiale sono maggiori della soglia di binarizzazione, il valore nelle posizioni corrispondenti del QTensor viene impostato a 1, altrimenti 0.

        :param v: Soglia di binarizzazione
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

        Riempie un QTensor con valori campionati casualmente da una distribuzione uniforme con segno.

        Fattore di scala dei valori generati dalla distribuzione uniforme con segno.

        :param v: un valore scalare
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

        Riempie un QTensor con valori campionati casualmente da una distribuzione uniforme.

        Fattore di scala dei valori generati dalla distribuzione uniforme.

        :param v: un valore scalare
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

        Riempie un QTensor con valori campionati casualmente da una distribuzione normale.
        Media della distribuzione normale. Deviazione standard della distribuzione normale.
        Indica se utilizzare o meno la modalita' fast-math.

        :param m: media della distribuzione normale
        :param s: deviazione standard della distribuzione normale
        :param fast_math: True se si utilizza fast-math
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

        Inverte o permuta gli assi di un array. Se new_dims = None, inverte le dimensioni.

        :param new_dims: il nuovo ordine delle dimensioni (lista di interi).
        :return: QTensor risultante.

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

        Modifica la forma del tensore e restituisce un nuovo QTensor.

        :param new_shape: la nuova forma (lista di interi)
        :return: un nuovo QTensor

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

        Modifica la forma del QTensor corrente in-place. Questa interfaccia tenta prima di trasformare senza modificare i dati originali in memoria. Se fallisce, i dati correnti vengono copiati nella nuova memoria.

        .. warning::

            Si consiglia di utilizzare l'interfaccia reshape. In alcuni casi, la posizione effettiva della memoria sottostante viene copiata invece di essere modificata in-place.

        :param new_shape: la nuova forma (lista di interi)
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

            Ottiene i dati del QTensor come array NumPy.

            :return: un array NumPy

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

            L'indicizzazione per slicing di QTensor e' supportata, cosi' come l'uso di QTensor come indice avanzato. Verra' restituito un nuovo QTensor.

            I parametri start, stop e step possono essere separati da due punti, ad esempio start:stop:step, dove start, stop e step possono essere omessi.

            Per un QTensor 1-D, l'indicizzazione o lo slicing possono essere effettuati solo su un singolo asse.

            Per un QTensor 2-D e un QTensor multidimensionale, l'indicizzazione o lo slicing possono essere effettuati su piu' assi.

            Se si utilizza QTensor come indice per indicizzazione avanzata, vedere numpy per `advanced indexing <https://docs.scipy.org/doc/numpy-1.10.1/reference/arrays.indexing.html>`_ .

            Se il QTensor usato come indice e' il risultato di un'operazione logica, si effettua un indicizzazione booleana.

            .. note:: 
                
                Utilizziamo una forma di indice come a[3,4,1], ma la forma a[3][4][1] non e' supportata.

            :param item: Un intero o QTensor usato come indice.

            :return: Un nuovo QTensor.

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

        L'indicizzazione per slicing di QTensor e' supportata, cosi' come l'uso di QTensor come indice avanzato. Verra' restituito un nuovo QTensor.

        I parametri start, stop e step possono essere separati da due punti, ad esempio start:stop:step, dove start, stop e step possono essere omessi.

        Per un QTensor 1-D, l'indicizzazione o lo slicing possono essere effettuati solo su un singolo asse.

        Per un QTensor 2-D e un QTensor multidimensionale, l'indicizzazione o lo slicing possono essere effettuati su piu' assi.

        Se si utilizza QTensor come indice per indicizzazione avanzata, vedere numpy per `advanced indexing <https://docs.scipy.org/doc/numpy-1.10.1/reference/arrays.indexing.html>`_ .

        Se il QTensor usato come indice e' il risultato di un'operazione logica, si effettua un indicizzazione booleana.

        .. note:: 
            
            Utilizziamo una forma di indice come a[3,4,1], ma la forma a[3][4][1] non e' supportata.

        :param item: Un intero o QTensor usato come indice

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

        Clona QTensor sul dispositivo GPU specificato.

        device specifica il dispositivo in cui memorizzare i dati interni. Quando device >= DEV_GPU_0, i dati vengono memorizzati sulla GPU.
        Se il computer ha piu' GPU, e' possibile designare dispositivi diversi per memorizzare i dati.
        Ad esempio, device = DEV_GPU_1, DEV_GPU_2, DEV_GPU_3, ... indica la memorizzazione su GPU con numeri di serie diversi.
        
        .. note::
            QTensor non puo' eseguire calcoli su GPU diverse.
            Verra' sollevato un errore Cuda se si tenta di creare un QTensor su una GPU il cui ID supera il numero massimo di GPU verificate.

        :param device: Il dispositivo che salva correntemente QTensor, default=DEV_GPU_0,

        device = pyvqnet.DEV_GPU_0, memorizzato nella prima GPU, device = DEV_GPU_1,
        memorizzato nella seconda GPU, e cosi' via.

        :return: Clona QTensor sul dispositivo GPU.

        Examples::

            from pyvqnet.tensor import QTensor
            a = QTensor([2])
            b = a.GPU()
            print(b.device)
            #1000

 

    .. py:method:: CPU()

        Clona QTensor sul dispositivo CPU specificato

        :return: Clona QTensor sul dispositivo CPU.

        Examples::

            from pyvqnet.tensor import QTensor
            a = QTensor([2])
            b = a.CPU()
            print(b.device)
            # 0

 
    .. py:method:: toGPU(device: int = DEV_GPU_0)

        Sposta QTensor sul dispositivo GPU specificato.

        device specifica il dispositivo in cui memorizzare i dati interni. Quando device >= DEV_GPU, i dati vengono memorizzati sulla GPU.
        Se il computer ha piu' GPU, e' possibile designare dispositivi diversi per memorizzare i dati.
        Ad esempio, device = DEV_GPU_1, DEV_GPU_2, DEV_GPU_3, ... indica la memorizzazione su GPU con numeri di serie diversi.

        .. note::

            QTensor non puo' eseguire calcoli su GPU diverse. Verra' sollevato un errore Cuda se si tenta di creare un QTensor su una GPU il cui ID supera il numero massimo di GPU verificate.

        :param device: Il dispositivo che salva correntemente QTensor, default=DEV_GPU_0. device = pyvqnet.DEV_GPU_0, memorizzato nella prima GPU, device = DEV_GPU_1, memorizzato nella seconda GPU, e cosi' via.
        :return: QTensor spostato sul dispositivo GPU.

        Examples::

            from pyvqnet.tensor import QTensor
            a = QTensor([2])
            a = a.toGPU()
            print(a.device)
            #1000


    
    .. py:method:: toCPU()

        Sposta QTensor sulla CPU

        :return: QTensor spostato sul dispositivo CPU.

        Examples::

            from pyvqnet.tensor import QTensor
            a = QTensor([2])
            b = a.toCPU()
            print(b.device)
            # 0

    
    .. py:method:: isGPU()

        Indica se i dati di questo QTensor sono memorizzati nella memoria della GPU.

        :return: Indica se i dati di questo QTensor sono memorizzati nella memoria della GPU.

        Examples::
        
            from pyvqnet.tensor import QTensor
            a = QTensor([2])
            a = a.isGPU()
            print(a)
            # False
 
    .. py:method:: isCPU()

        Indica se i dati di questo QTensor sono memorizzati nella memoria della CPU.

        :return: Indica se i dati di questo QTensor sono memorizzati nella memoria della CPU.

        Examples::
        
            from pyvqnet.tensor import QTensor
            a = QTensor([2])
            a = a.isCPU()
            print(a)
            # True


Funzioni di Creazione
*****************************************************


ones
==============================

.. py:function:: pyvqnet.tensor.ones(shape,device=0,dtype-None)

    Restituisce un tensore di uno con la forma specificata.

    :param shape: forma di input
    :param device: dispositivo su cui memorizzare, default 0, CPU.
    :param dtype: tipo di dato del parametro, default None, usa il tipo di dato predefinito: kfloat32, che rappresenta un numero a virgola mobile a 32 bit.
    
    :return: QTensor di output con la forma specificata.

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

    Restituisce un tensore di uno con la stessa forma del QTensor di input.

    :param t: QTensor di input
    :param device: dispositivo su cui memorizzare, default 0, CPU.
    :param dtype: tipo di dato del parametro, default None, usa il tipo di dato predefinito: kfloat32, che rappresenta un numero a virgola mobile a 32 bit.
    
    :return: QTensor di output


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

    Crea un QTensor della forma specificata e lo riempie con il valore indicato.

    :param shape: forma del QTensor da creare
    :param value: valore con cui riempire il QTensor.
    :param device: dispositivo da utilizzare, default = 0, usa dispositivo CPU.
    :param dtype: tipo di dato del parametro, default None, usa il tipo di dato predefinito: kfloat32, che rappresenta un numero a virgola mobile a 32 bit.
    
    :return: QTensor di output

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

    Crea un QTensor della forma specificata e lo riempie con il valore indicato.

    :param t: QTensor di input
    :param value: valore con cui riempire il QTensor.
    :param device: dispositivo da utilizzare, default = 0, usa dispositivo CPU.
    :param dtype: tipo di dato del parametro, default None, usa il tipo di dato predefinito: kfloat32, che rappresenta un numero a virgola mobile a 32 bit.
    
    :return: QTensor di output

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

    Restituisce un tensore di zeri con la forma specificata.

    :param shape: forma del tensore
    :param device: dispositivo da utilizzare, default = 0, usa dispositivo CPU
    :param dtype: tipo di dato del parametro, default None, usa il tipo di dato predefinito: kfloat32, che rappresenta un numero a virgola mobile a 32 bit.
    
    :return: QTensor di output

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

    Restituisce un tensore di zeri con la stessa forma del QTensor di input.

    :param t: QTensor di input
    :param device: dispositivo da utilizzare, default = 0, usa dispositivo CPU
    :param dtype: tipo di dato del parametro, default None, usa il tipo di dato predefinito: kfloat32, che rappresenta un numero a virgola mobile a 32 bit.
    
    :return: QTensor di output

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

    Crea un QTensor 1D con valori equidistanti all'interno di un intervallo specificato.

    :param start: inizio dell'intervallo
    :param end: fine dell'intervallo
    :param step: spaziatura tra i valori
    :param device: dispositivo da utilizzare, default = 0, usa dispositivo CPU
    :param dtype: tipo di dato del parametro, default None, usa il tipo di dato predefinito: kfloat32, che rappresenta un numero a virgola mobile a 32 bit.
    :param requires_grad: indica se tracciare il gradiente del tensore, default False
    :return: QTensor di output

    Example::

        from pyvqnet.tensor import tensor
        from pyvqnet.tensor import QTensor
        t = tensor.arange(2, 30,4)
        print(t)

        # [ 2,  6, 10, 14, 18, 22, 26]

linspace
==============================

.. py:function:: pyvqnet.tensor.linspace(start, end, num, device: int = 0,dtype=None, requires_grad= False)

    Crea un QTensor 1D con valori equidistanti all'interno di un intervallo specificato.

    :param start: valore iniziale
    :param end: valore finale
    :param nums: numero di campioni da generare
    :param device: dispositivo da utilizzare, default = 0, usa dispositivo CPU
    :param dtype: tipo di dato del parametro, default None, usa il tipo di dato predefinito: kfloat32, che rappresenta un numero a virgola mobile a 32 bit.
    :param requires_grad: indica se tracciare il gradiente del tensore, default False
    :return: QTensor di output

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

    Crea un QTensor 1D con valori equidistanti su scala logaritmica.

    :param start: ``base ** start`` e' il valore iniziale
    :param end: ``base ** end`` e' il valore finale della sequenza
    :param nums: numero di campioni da generare
    :param base: la base dello spazio logaritmico
    :param device: dispositivo da utilizzare, default = 0, usa dispositivo CPU
    :param dtype: tipo di dato del parametro, default None, usa il tipo di dato predefinito: kfloat32, che rappresenta un numero a virgola mobile a 32 bit.
    :param requires_grad: indica se tracciare il gradiente del tensore, default False
    :return: QTensor di output

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

    Crea un QTensor size x size con uno sulla diagonale e zero
    altrove.

    :param size: dimensione del QTensor (quadrato) da creare
    :param offset: Indice della diagonale: 0 (default) si riferisce alla diagonale principale, un valore positivo si riferisce a una diagonale superiore, un valore negativo a una diagonale inferiore.
    :param device: dispositivo da utilizzare, default = 0, usa dispositivo CPU
    :param dtype: tipo di dato del parametro, default None, usa il tipo di dato predefinito: kfloat32, che rappresenta un numero a virgola mobile a 32 bit.
    
    :return: QTensor di output

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


    Restituisce una vista parziale di :attr:`t` con gli elementi della diagonale aggiunti come dimensioni alla fine della forma rispetto a :attr:`dim1` e :attr:`dim2`.
    :attr:`offset` e' l'offset della diagonale principale.

    :param t: tensore di input
    :param offset: offset (0 indica la diagonale principale, valori positivi indicano l'n-esima diagonale sopra la diagonale principale, valori negativi indicano l'n-esima diagonale sotto la diagonale principale)
    :param dim1: prima dimensione da cui prendere la diagonale. Default: 0.
    :param dim2: seconda dimensione da cui prendere la diagonale. Default: 1.

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

    Seleziona gli elementi diagonali o costruisce un QTensor diagonale.

    Inserisci un QTensor 2-D e restituisce un nuovo tensore 1D contenente gli elementi diagonali selezionati. Inserisci un QTensor 1-D e restituisce un nuovo tensore 2D i cui elementi diagonali selezionati sono i valori di input e il resto e' 0.

    :param t: QTensor di input
    :param k: offset (0 per la diagonale principale, positivo per l'n-esima
        diagonale sopra la principale, negativo per l'n-esima diagonale sotto la
        principale)
    :return: QTensor di output

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

    Crea un QTensor con valori casuali distribuiti uniformemente.

    :param shape: forma del QTensor da creare
    :param min: valore minimo della distribuzione uniforme, default: 0.
    :param max: valore massimo della distribuzione uniforme, default: 1.
    :param device: dispositivo da utilizzare, default = 0, usa dispositivo CPU
    :param dtype: tipo di dato del parametro, default None, usa il tipo di dato predefinito: kfloat32, che rappresenta un numero a virgola mobile a 32 bit.
    :param requires_grad: indica se tracciare il gradiente del tensore, default False
    :return: QTensor di output


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

    Crea un QTensor con valori casuali distribuiti normalmente.

    :param shape: forma del QTensor da creare
    :param mean: media della distribuzione normale, default: 0.
    :param std: deviazione standard della distribuzione normale, default: 1.
    :param device: dispositivo da utilizzare, default = 0, usa dispositivo CPU
    :param dtype: tipo di dato del parametro, default None, usa il tipo di dato predefinito: kfloat32, che rappresenta un numero a virgola mobile a 32 bit.
    :param requires_grad: indica se tracciare il gradiente del tensore, default False
    :return: QTensor di output

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
    
    Crea una distribuzione binomiale parametrizzata da :attr:total_count e :attr:probs.

    :param total_counts: Numero di prove Bernoulliane.
    :param probs: Probabilita' degli eventi.

    :return:
        QTensor per la distribuzione binomiale.

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

    Restituisce un tensore in cui ogni riga contiene num_samples campioni indicizzati.
    Dalla distribuzione di probabilita' multinomiale situata nella riga corrispondente del tensore di input.

    :param t: Distribuzione di probabilita' di input.
    :param num_samples: numero di campioni.

    :return:
        indice del campione di output

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

    Restituisce la matrice triangolare superiore dell'input t, con il resto impostato a 0.

    :param t: QTensor di input
    :param diagonal: Offset, default = 0. La diagonale principale e' 0, positivo e' offset verso l'alto, negativo e' offset verso il basso

    :return: QTensor di output

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

    Restituisce la matrice triangolare inferiore dell'input t, con il resto impostato a 0.

    :param t: QTensor di input
    :param diagonal: Offset, default = 0. La diagonale principale e' 0, positivo e' offset verso l'alto, negativo e' offset verso il basso

    :return: QTensor di output

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


Funzioni Matematiche
*****************************************************


floor
==============================

.. py:function:: pyvqnet.tensor.floor(t)

    Restituisce un nuovo QTensor con la parte intera inferiore degli elementi di input, il piu' grande intero minore o uguale a ciascun elemento.

    :param t: QTensor di input
    :return: QTensor di output

    Example::

        from pyvqnet.tensor import tensor

        t = tensor.arange(-2.0, 2.0, 0.25)
        u = tensor.floor(t)
        print(u)

        # [-2, -2, -2, -2, -1, -1, -1, -1, 0, 0, 0, 0, 1, 1, 1, 1]

ceil
==============================

.. py:function:: pyvqnet.tensor.ceil(t)

    Restituisce un nuovo QTensor con la parte intera superiore degli elementi di input, il piu' piccolo intero maggiore o uguale a ciascun elemento.

    :param t: QTensor di input
    :return: QTensor di output

    Example::

        from pyvqnet.tensor import tensor

        t = tensor.arange(-2.0, 2.0, 0.25)
        u = tensor.ceil(t)
        print(u)

        # [-2, -1, -1, -1, -1, -0, -0, -0, 0, 1, 1, 1, 1, 2, 2, 2]

round
==============================

.. py:function:: pyvqnet.tensor.round(t)

    Arrotonda i valori del QTensor all'intero piu' vicino.

    :param t: QTensor di input
    :return: QTensor di output

    Example::

        from pyvqnet.tensor import tensor

        t = tensor.arange(-2.0, 2.0, 0.4)
        u = tensor.round(t)
        print(u)

        # [-2, -2, -1, -1, -0, -0, 0, 1, 1, 2]

sort
==============================

.. py:function:: pyvqnet.tensor.sort(t, axis: int, descending=False, stable=True)

    Ordina il QTensor lungo l'asse

    :param t: QTensor di input
    :param axis: asse di ordinamento
    :param descending: ordine di ordinamento se decrescente
    :param stable: indica se utilizzare l'ordinamento stabile o meno
    :return: QTensor di output

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

    Restituisce un array di indici della stessa forma dell'input che indicizzano i dati lungo l'asse specificato in ordine ordinato.

    :param t: QTensor di input
    :param axis: asse di ordinamento
    :param descending: ordine di ordinamento se decrescente
    :param stable: indica se utilizzare l'ordinamento stabile o meno
    :return: QTensor di output

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

    Restituisce i k elementi piu' grandi del tensore di input lungo l'asse specificato.

    Se if_descent e' False, restituisce i k elementi piu' piccoli.

    :param t: QTensor di input
    :param k: numero di elementi piu' grandi o piu' piccoli
    :param axis: asse di ordinamento, default = -1, l'ultimo asse
    :param if_descent: ordine di ordinamento, default True

    :return: Un nuovo QTensor

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

    Restituisce l'indice dei k elementi piu' grandi lungo l'asse specificato del tensore di input.

    Se if_descent e' False, restituisce l'indice dei k elementi piu' piccoli.

    :param t: QTensor di input
    :param k: numero di elementi piu' grandi o piu' piccoli
    :param axis: asse di ordinamento, default = -1, l'ultimo asse
    :param if_descent: ordine di ordinamento, default True

    :return: Un nuovo QTensor

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

    Somma elemento per elemento due QTensor, equivalente a t1 + t2.

    :param t1: primo QTensor
    :param t2: secondo QTensor
    :return: QTensor di output

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

    Sottrae elemento per elemento due QTensor, equivalente a t1 - t2.


    :param t1: primo QTensor
    :param t2: secondo QTensor
    :return: QTensor di output

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

    Moltiplica elemento per elemento due QTensor, equivalente a t1 * t2.

    :param t1: primo QTensor
    :param t2: secondo QTensor
    :return: QTensor di output


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

    Divide elemento per elemento due QTensor, equivalente a t1 / t2.


    :param t1: primo QTensor
    :param t2: secondo QTensor
    :return: QTensor di output


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

    Somma tutti gli elementi nel QTensor lungo l'asse specificato. Se axis = None, somma tutti gli elementi nel QTensor.

    :param t: QTensor di input
    :param axis: asse utilizzato per la somma, default None
    :param keepdims: indica se la dimensione del tensore di output viene mantenuta o meno, default False
    :return: QTensor di output


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

    Restituisce la somma cumulativa degli elementi di input lungo la dimensione axis.

    :param t: il QTensor di input
    :param axis: Asse di calcolo, default -1, usa l'ultimo asse

    :return: QTensor di output.

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

    Ottiene i valori medi nel QTensor lungo l'asse.

    :param t: il QTensor di input.
    :param axis: la dimensione da ridurre.
    :param keepdims: indica se la dimensione del QTensor di output viene mantenuta o meno, default False.
    :return: restituisce il valore medio del QTensor di input.

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

    Ottiene il valore mediano nel QTensor.

    :param t: il QTensor di input
    :param axis: un asse per la media, default None
    :param keepdims: indica se la dimensione del QTensor di output viene mantenuta o meno, default False

    :return: Restituisce la mediana dei valori nel QTensor di input.

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

    Ottiene il valore di deviazione standard nel QTensor.


    :param t: il QTensor di input
    :param axis: l'asse utilizzato per calcolare la deviazione standard, default None
    :param keepdims: indica se la dimensione del QTensor di output viene mantenuta o meno, default False
    :param unbiased: indica se utilizzare la correzione di Bessel, default True
    :return: Restituisce la deviazione standard dei valori nel QTensor di input

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

    Ottiene la varianza nel QTensor.


    :param t: il QTensor di input.
    :param axis: l'asse utilizzato per calcolare la varianza, default None
    :param keepdims: indica se la dimensione del QTensor di output viene mantenuta o meno, default False.
    :param unbiased: indica se utilizzare la correzione di Bessel, default True.


    :return: Ottiene la varianza nel QTensor.

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

    Moltiplicazione matriciale di due matrici 2D, 3D, 4D.

    :param t1: primo QTensor
    :param t2: secondo QTensor
    :return: QTensor di output

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

    Calcola il prodotto di Kronecker di ``t1`` e ``t2``, espresso come :math:`\otimes` . Se ``t1`` e' un tensore :math:`(a_0 \times a_1 \times \dots \times a_n)` e ``t2`` e' un tensore :math:`(b_0 \times b_1 \times \dots \times b_n)`, il risultato sara' un tensore :math:`(a_0*b_0 \times a_1*b_1 \times \dots \times a_n*b_n)` con le seguenti voci:
    
    .. math::
          (\text{input} \otimes \text{other})_{k_0, k_1, \dots, k_n} =
              \text{input}_{i_0, i_1, \dots, i_n} * \text{other}_{j_0, j_1, \dots, j_n},

    dove :math:`k_t = i_t * b_t + j_t` con :math:`0 \leq t \leq n`.
    Se un tensore ha meno dimensioni dell'altro, verra' espanso fino ad avere la stessa dimensionalita'.

    :param t1: Il primo QTensor.
    :param t2: Il secondo QTensor.
    
    :return: QTensor di output.

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
    
    Somma i prodotti degli elementi degli operandi di input lungo la dimensione specificata utilizzando una notazione basata sulla convenzione di sommatoria di Einstein.

    .. note::

        Questa funzione utilizza opt_einsum (https://optimized-einsum.readthedocs.io/en/stable/) per accelerare il calcolo o ridurre il consumo di memoria ottimizzando l'ordine di contrazione. Questa ottimizzazione avviene quando ci sono almeno tre input.

        Per `einsum` piu' complessi, e' possibile importare ulteriormente opt_einsum per calcolare direttamente su QTensor.

    :param equation: Il pedice della sommatoria di Einstein.
    :param operands: Il tensore su cui calcolare la sommatoria di Einstein.

    :return:

        Il QTensor risultante.

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

    Calcola il reciproco elemento per elemento del QTensor.

    :param t: QTensor di input
    :return: QTensor di output

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

    Restituisce un nuovo QTensor con i segni degli elementi di input. La funzione segno restituisce -1 se t < 0, 0 se t==0, 1 se t > 0.

    :param t: QTensor di input
    :return: QTensor di output


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

    Negazione unaria degli elementi del QTensor.

    :param t: QTensor di input
    :return: QTensor di output

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

    Restituisce la somma degli elementi della diagonale della matrice 2D di input.

    :param t: QTensor 2D di input
    :param k: offset (0 per la diagonale principale, positivo per l'n-esima
        diagonale sopra la principale, negativo per l'n-esima diagonale sotto la
        principale)
    :return: la somma degli elementi della diagonale della matrice 2D di input

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

    Applica la funzione esponenziale a tutti gli elementi del QTensor di input.

    :param t: QTensor di input
    :return: QTensor di output

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

    Calcola l'arcocoseno elemento per elemento del QTensor.

    :param t: QTensor di input
    :return: QTensor di output

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

    Calcola l'arcoseno elemento per elemento del QTensor.

    :param t: QTensor di input
    :return: QTensor di output

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

    Calcola l'arcotangente elemento per elemento del QTensor.

    :param t: QTensor di input
    :return: QTensor di output

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

    Applica la funzione seno a tutti gli elementi del QTensor di input.


    :param t: QTensor di input
    :return: QTensor di output

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

    Applica la funzione coseno a tutti gli elementi del QTensor di input.


    :param t: QTensor di input
    :return: QTensor di output

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

    Applica la funzione tangente a tutti gli elementi del QTensor di input.


    :param t: QTensor di input
    :return: QTensor di output

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

    Applica la funzione tangente iperbolica a tutti gli elementi del QTensor di input.

    :param t: QTensor di input
    :return: QTensor di output

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

    Applica la funzione seno iperbolico a tutti gli elementi del QTensor di input.


    :param t: QTensor di input
    :return: QTensor di output

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

    Applica la funzione coseno iperbolico a tutti gli elementi del QTensor di input.


    :param t: QTensor di input
    :return: QTensor di output

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

    Eleva il primo QTensor alla potenza del secondo QTensor.

    :param t1: primo QTensor
    :param t2: secondo QTensor
    :return: QTensor di output

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

    Applica la funzione valore assoluto a tutti gli elementi del QTensor di input.

    :param t: QTensor di input
    :return: QTensor di output

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

    Applica la funzione log (ln) a tutti gli elementi del QTensor di input.

    :param t: QTensor di input
    :return: QTensor di output

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
    
    Calcola sequenzialmente i risultati della funzione softmax e della funzione log sull'asse axis.

    :param t: QTensor di input.
    :param axis: L'asse utilizzato per calcolare softmax, il default e' -1.

    :return: QTensor di output.

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

    Applica la funzione radice quadrata a tutti gli elementi del QTensor di input.


    :param t: QTensor di input
    :return: QTensor di output

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

    Applica la funzione quadrato a tutti gli elementi del QTensor di input.


    :param t: QTensor di input
    :return: QTensor di output

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
 
    Restituisce gli autovalori e gli autovettori di una matrice complessa Hermitiana (coniugata simmetrica) o reale simmetrica.

    Restituisce due oggetti: un array 1D contenente gli autovalori di a,
    e una matrice quadrata 2D o matrice (a seconda del tipo di input) dei corrispondenti autovettori (in colonne).

    :param: QTensor di input.
    :param: Autovalori e autovettori di t.
    :return:

        Restituisce autovalori e autovettori

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

    Calcola la norma F del tensore sul QTensor di input lungo l'asse specificato da axis,
    se axis e' None, restituisce la norma F di tutti gli elementi.

    :param t: QTensor di input.
    :param axis: L'asse utilizzato per calcolare la norma F, il default e' None.
    :param keepdims: Indica se il tensore di output mantiene la dimensionalita' ridotta. Il default e' False.
    :return: Restituisce un QTensor o un valore di norma F.


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



Funzioni Logiche
**************************

maximum
==============================

.. py:function:: pyvqnet.tensor.maximum(t1: pyvqnet.tensor.QTensor, t2: pyvqnet.tensor.QTensor)

    Massimo elemento per elemento di due tensori.


    :param t1: primo QTensor
    :param t2: secondo QTensor
    :return: QTensor di output

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

    Minimo elemento per elemento di due tensori.


    :param t1: primo QTensor
    :param t2: secondo QTensor
    :return: QTensor di output

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

    Restituisce gli elementi minimi del QTensor di input lungo l'asse specificato.
    Se axis == None, restituisce il valore minimo di tutti gli elementi del tensore.

    :param t: QTensor di input
    :param axis: asse utilizzato per il minimo, default None
    :param keepdims: indica se la dimensione del tensore di output viene mantenuta o meno, default False
    :return: QTensor di output

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

    Restituisce gli elementi massimi del QTensor di input lungo l'asse specificato.
    Se axis == None, restituisce il valore massimo di tutti gli elementi del tensore.

    :param t: QTensor di input
    :param axis: asse utilizzato per il massimo, default None
    :param keepdims: indica se la dimensione del tensore di output viene mantenuta o meno, default False
    :return: QTensor di output

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

    Limita il QTensor di input ai valori minimo e massimo.

    :param t: QTensor di input
    :param min_val: valore minimo
    :param max_val: valore massimo
    :return: QTensor di output

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

    Restituisce elementi scelti da x o y in base alla condizione.

    :param condition: tensore di condizione, deve avere tipo di dato kbool.
    :param t1: QTensor da cui prendere gli elementi se la condizione e' vera, default None
    :param t2: QTensor da cui prendere gli elementi se la condizione e' falsa, default None
    :return: QTensor di output

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

    Restituisce un QTensor contenente gli indici degli elementi diversi da zero.

    :param t: QTensor di input
    :return: QTensor di output contenente gli indici degli elementi diversi da zero.

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

    Verifica elemento per elemento se e' finito (non infinito e non NaN).

    :param t: QTensor di input
    :return: QTensor di output, restituisce True quando l'elemento nella posizione corrispondente soddisfa la condizione, altrimenti False.

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

    Verifica elemento per elemento se e' infinito positivo o negativo.

    :param t: QTensor di input
    :return: QTensor di output, restituisce True quando l'elemento nella posizione corrispondente soddisfa la condizione, altrimenti False.
    
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

    Verifica elemento per elemento se e' NaN.

    :param t: QTensor di input
    :return: QTensor di output, restituisce True quando l'elemento nella posizione corrispondente soddisfa la condizione, altrimenti False.
    
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

    Verifica elemento per elemento se e' infinito negativo.

    :param t: QTensor di input
    :return: QTensor di output, restituisce True quando l'elemento nella posizione corrispondente soddisfa la condizione, altrimenti False.
    
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

    Verifica elemento per elemento se e' infinito positivo.

    :param t: QTensor di input
    :return: QTensor di output, restituisce True quando l'elemento nella posizione corrispondente soddisfa la condizione, altrimenti False.
    
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

    Calcola il valore di verita' di ``t1`` and ``t2`` elemento per elemento.

    :param t1: QTensor di input
    :param t2: QTensor di input
    :return: QTensor di output, restituisce True quando l'elemento nella posizione corrispondente soddisfa la condizione, altrimenti False.
    
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

    Calcola il valore di verita' di ``t1 or t2`` elemento per elemento.

    :param t1: QTensor di input
    :param t2: QTensor di input
    :return: QTensor di output, restituisce True quando l'elemento nella posizione corrispondente soddisfa la condizione, altrimenti False.

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

    Calcola il valore di verita' di ``not t`` elemento per elemento.

    :param t: QTensor di input
    :return: QTensor di output, restituisce True quando l'elemento nella posizione corrispondente soddisfa la condizione, altrimenti False.

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

    Calcola il valore di verita' di ``t1 xor t2`` elemento per elemento.

    :param t1: QTensor di input
    :param t2: QTensor di input

    :return: QTensor di output, restituisce True quando l'elemento nella posizione corrispondente soddisfa la condizione, altrimenti False.

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

    Restituisce il valore di verita' di ``t1 > t2`` elemento per elemento.


    :param t1: QTensor di input
    :param t2: QTensor di input
    :return: QTensor di output, restituisce True quando l'elemento nella posizione corrispondente soddisfa la condizione, altrimenti False.

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

    Restituisce il valore di verita' di ``t1 >= t2`` elemento per elemento.

    :param t1: QTensor di input
    :param t2: QTensor di input
    :return: QTensor di output, restituisce True quando l'elemento nella posizione corrispondente soddisfa la condizione, altrimenti False.

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

    Restituisce il valore di verita' di ``t1 < t2`` elemento per elemento.

    :param t1: QTensor di input
    :param t2: QTensor di input
    :return: QTensor di output, restituisce True quando l'elemento nella posizione corrispondente soddisfa la condizione, altrimenti False.

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

    Restituisce il valore di verita' di ``t1 <= t2`` elemento per elemento.

    :param t1: QTensor di input
    :param t2: QTensor di input
    :return: QTensor di output, restituisce True quando l'elemento nella posizione corrispondente soddisfa la condizione, altrimenti False.

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

    Restituisce il valore di verita' di ``t1 == t2`` elemento per elemento.

    :param t1: QTensor di input
    :param t2: QTensor di input
    :return: QTensor di output, restituisce True quando l'elemento nella posizione corrispondente soddisfa la condizione, altrimenti False.
    
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

    Restituisce il valore di verita' di ``t1 != t2`` elemento per elemento.

    :param t1: QTensor di input
    :param t2: QTensor di input
    :return: QTensor di output, restituisce True quando l'elemento nella posizione corrispondente soddisfa la condizione, altrimenti False.
    
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
 
    Calcola l'AND bit a bit di due elementi QTensor.

    :param t1: QTensor di input t1. Solo interi o booleani sono input validi.
    :param t2: QTensor di input t2. Solo interi o booleani sono input validi.
    :return:
        QTensor risultante

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


Operazioni Matriciali
**********************

select
==============================

.. py:function:: pyvqnet.tensor.select(t: pyvqnet.tensor.QTensor, index)

    Restituisce QTensor nel QTensor all'asse specificato. L'operazione seguente ottiene lo stesso valore del risultato.

    :param t: QTensor di input
    :param index: una stringa contenente la dimensione di output
    :return: QTensor di output

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

    Fatte salve alcune restrizioni, gli array piu' piccoli vengono distribuiti su array piu' grandi in modo da avere forme compatibili. Questa interfaccia puo' eseguire la differenziazione automatica sui tensori dei parametri di input.

    Riferimento https://numpy.org/doc/stable/user/basics.broadcasting.html

    :param t1: QTensor di input 1
    :param t2: QTensor di input 2

    :return t11: t1 con la nuova forma broadcast.
    :return t22: t2 con la nuova forma broadcast.

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

    Concatena il QTensor di input lungo l'asse e restituisce un nuovo QTensor.

    :param args: lista composta da QTensor di input
    :param axis: dimensione per la concatenazione. Deve essere compresa tra 0 e il numero di dimensioni dei tensori da concatenare.
    :return: QTensor di output

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

    Unisce una sequenza di array lungo un nuovo asse, restituisce un nuovo QTensor.

    :param QTensors: lista contenente QTensors
    :param axis: dimensione da inserire. Deve essere compresa tra 0 e il numero di dimensioni dei tensori impilati.
    :return: QTensor di output

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

    Inverte o permuta gli assi di un array.

    :param t: QTensor di input
    :param dim: il nuovo ordine delle dimensioni (lista di interi)
    :return: QTensor di output

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

    Traspone gli assi di un array. Se dim = None, inverte le dimensioni. Questa funzione e' uguale a permute.

    :param t: QTensor di input
    :param dim: il nuovo ordine delle dimensioni (lista di interi)
    :return: QTensor di output

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

    Costruisce un QTensor ripetendo il QTensor il numero di volte specificato da reps.

    Se reps ha lunghezza d, il QTensor risultante avra' dimensione max(d, t.ndim).

    Se t.ndim < d, t viene espanso a d-dimensioni inserendo nuovi assi dalla dimensione iniziale.
    Quindi un array di forma (3,) viene promosso a (1, 3) per la replica 2D, o forma (1, 1, 3) per la replica 3D.

    Se t.ndim > d, reps viene espanso a t.ndim inserendo 1 in esso.

    Quindi per un t di forma (2, 3, 4, 5), un reps di (4, 3) viene trattato come (1, 1, 4, 3).

    :param t: QTensor di input
    :param reps: il numero di ripetizioni per dimensione.
    :return: un nuovo QTensor

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

    Rimuove gli assi di lunghezza unitaria.

    :param t: QTensor di input
    :param axis: asse da comprimere, se axis = -1, comprime tutte le dimensioni che hanno dimensione 1.
    :return: QTensor di output

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

    Restituisce un nuovo QTensor con una dimensione di dimensione unitaria inserita nella posizione specificata.

    :param t: QTensor di input
    :param axis: asse per unsqueeze, in cui verra' inserita la dimensione.
    :return: QTensor di output

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

    Sposta le dimensioni di `t` dalle posizioni in `source` alle posizioni in `destination`.

    Le altre dimensioni di `t` che non vengono spostate esplicitamente mantengono il loro ordine originale e appaiono nelle posizioni non specificate in `destination`.

    :param t: QTensor di input.
    :param source: (intero o tupla di interi) Le posizioni originali delle dimensioni da spostare. Queste posizioni devono essere uniche.
    :param destination: (intero o tupla di interi) Le posizioni di destinazione per ogni dimensione originale. Anche queste posizioni devono essere uniche.

    :return:
        Nuovo QTensor


    Example::

        from pyvqnet import QTensor,tensor
        a = tensor.arange(0,24).reshape((2,3,4))
        b = tensor.moveaxis(a,(1, 2), (0, 1))
        print(b.shape)


swapaxis
==============================

.. py:function:: pyvqnet.tensor.swapaxis(t, axis1: int, axis2: int)

    Scambia due assi di un array. Le dimensioni specificate axis1 e axis2 vengono scambiate.

    :param t: QTensor di input
    :param axis1: Primo asse.
    :param axis2: Posizione di destinazione per l'asse originale. Devono anche essere univoci.
    :return: QTensor di output

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

    Se mask == 1, riempie con il valore specificato. La forma di mask deve essere broadcastable dalla forma del QTensor di input.

    :param t: QTensor di input
    :param mask: Un QTensor
    :param value: valore specificato
    :return: Un QTensor

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

    Appiattisce QTensor dalla dimensione start alla dimensione end.

    :param t: QTensor di input
    :param start: dimensione di inizio, default = 0, parte dalla prima dimensione.
    :param end: dimensione di fine, default = -1, termina con l'ultima dimensione.
    :return: QTensor di output

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

    Modifica la forma del QTensor, restituisce un QTensor con la nuova forma.

    :param t: QTensor di input.
    :param new_shape: nuova forma

    :return: un QTensor con la nuova forma.

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
    
    Inverte il QTensor lungo l'asse specificato, restituendo un nuovo tensore.

    :param t: QTensor di input.
    :param flip_dims: L'asse o la lista di assi da invertire.

    :return: QTensor di output.

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

    Raccoglie i valori lungo l'asse specificato da 'dim'.

    Per tensori 3D, l'output e' specificato da:

    .. math::

        out[i][j][k] = t[index[i][j][k]][j][k] , if dim == 0 \\

        out[i][j][k] = t[i][index[i][j][k]][k] , if dim == 1 \\

        out[i][j][k] = t[i][j][index[i][j][k]] , if dim == 2 \\

    :param t: QTensor di input.
    :param dim: L'asse di aggregazione.
    :param index: QTensor di indice, deve avere la stessa dimensione dell'input.

    :return: il risultato aggregato

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

    Scrive tutti i valori del tensore src in input agli indici specificati nel tensore indices.

    Per tensori 3D, l'output e' specificato da:

    .. math::

        input[indices[i][j][k]][j][k] = src[i][j][k] , if dim == 0 \\
        input[i][indices[i][j][k]][k] = src[i][j][k] , if dim == 1 \\
        input[i][j][indices[i][j][k]] = src[i][j][k] , if dim == 2 \\

    :param input: QTensor di input.
    :param dim: Asse di scattering.
    :param indices: QTensor di indice, deve avere la stessa dimensione dell'input.
    :param src: Il tensore sorgente da distribuire.

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

    Fatte salve alcune limitazioni, l'array t viene "broadcast" alla forma di riferimento in modo da avere forme compatibili.

    https://numpy.org/doc/stable/user/basics.broadcasting.html

    :param t: QTensor di input
    :param ref: Forma di riferimento.
    
    :return: Il QTensor di t appena broadcastato.

    Example::

        from pyvqnet.tensor.tensor import QTensor
        from pyvqnet.tensor import *
        ref = [2,3,4]
        a = ones([4])
        b = tensor.broadcast_to(a,ref)
        print(b.shape)
        #[2, 3, 4]



Funzioni di Utilita'
*****************************************************


to_tensor
==============================

.. py:function:: pyvqnet.tensor.to_tensor(x)

    Converte l'array di input in QTensor se non lo e' gia'.

    :param x: intero, float o numpy.array
    :return: QTensor di output

    Example::

        from pyvqnet.tensor import tensor
        t = tensor.to_tensor(10.0)
        print(t)
        # [10]


pad_sequence
==============================

.. py:function:: pyvqnet.tensor.pad_sequence(qtensor_list, batch_first=False, padding_value=0)

    Riempie una lista di tensori di lunghezza variabile con ``padding_value``. ``pad_sequence`` impila liste di tensori lungo nuove dimensioni e le riempie fino a renderle di uguale lunghezza.
    L'input e' una sequenza di liste di dimensione ``L x *``. L e' di lunghezza variabile.

    :param qtensor_list: `list[QTensor]` - lista di sequenze di lunghezza variabile.
    :param batch_first: 'bool' - Se True, l'output sara' ``batch size x longest sequence length x *``, altrimenti ``longest sequence length x batch size x *``. Default: False.
    :param padding_value: 'float' - valore di padding. Default: 0.

    :return:
         Se batch_first e' ``False``, la dimensione del tensore e' ``batch size x longest sequence length x *``.
         Altrimenti la dimensione del tensore e' ``longest sequence length x batch size x *``.

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
    
    Riempie un batch di sequenze impacchettate di lunghezza variabile. E' l'inverso di `pack_pad_sequence`.
    Quando ``batch_first`` e' True, restituisce un tensore di forma ``B x T x *``, altrimenti restituisce ``T x B x *``.
    Dove `T` e' la lunghezza massima della sequenza e `B` e' la dimensione del batch.

    :param sequence: 'QTensor' - i dati da elaborare.
    :param batch_first: 'bool' - Se ``True``, il batch sara' la prima dimensione dell'input. Default: False.
    :param padding_value: 'bool' - valore di padding. Default: 0.
    :param total_length: 'bool' - Se non ``None``, l'output sara' riempito fino alla lunghezza :attr:`total_length`. Default: None.
    :return:
        Una tupla di tensori contenente le sequenze riempite e una lista di lunghezze per ogni sequenza nel batch. Gli elementi del batch verranno riordinati nel loro ordine originale.
    
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
    
    Impacchetta un tensore contenente sequenze riempite di lunghezza variabile. Se batch_first e' True, `input` deve avere forma [batch size, length,*], altrimenti forma [length, batch size,*].

    Per sequenze non ordinate, usare ``enforce_sorted`` = False. Se :attr:`enforce_sorted` e' ``True``, le sequenze devono essere ordinate in ordine decrescente per lunghezza.
    
    :param input: 'QTensor' - batch di sequenze di lunghezza variabile da riempire.
    :param lengths: 'list' - lista delle lunghezze delle sequenze per ogni elemento
         del batch.
    :param batch_first: 'bool' - se ``True``, l'input deve essere nel formato ``B x T x *``
         default: False.
    :param enforce_sorted: 'bool' - se ``True``, l'input deve
         contenere sequenze in ordine decrescente di lunghezza. Se ``False``, l'input verra' ordinato incondizionatamente. Default: True.

    :return: Un oggetto :class:`PackedSequence`.

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
    
    Esegue una convoluzione 2D su un'immagine di input composta da piu' piani di input.

    :param x: Tensore di input 4D.
    :param weight: Tensore kernel 4D.

    :param stride: `tuple` - passo, default (1, 1)
    :param padding: Padding, controlla la quantita' di padding sull'input. Puo' essere una stringa {'valid', 'same'} o una tupla di interi che specifica la quantita' di padding implicito da applicare all'input, default (0,0).
    :param dilation: `tuple` - Spaziatura tra gli elementi del kernel. Default: (0,0)
    :param groups: `int` - Numero di gruppi. Default: 1 

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

    Registra i nodi di backpropagation quando il calcolo forward e' disabilitato.

    Esempio::

        import pyvqnet.tensor as tensor
        from pyvqnet import no_grad

        with no_grad():
            x = tensor.QTensor([1.0, 2.0, 3.0],requires_grad=True)
            y = tensor.tan(x)
            y.backward()
        """
        RuntimeError: The output tensor does not require gradients (output.requires_grad == False). This may occur if you used a non-autograd function in the forward pass. To enable gradient computation, ensure that all operations are performed on tensors with requires_grad=True, or use autograd-compatible functions.
        """
