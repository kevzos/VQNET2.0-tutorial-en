.. _qtensor_api:

QTensor-Modul
###########################

VQNet Quantum Machine Learning verwendet die Datenstruktur QTensor, die eine Python-Schnittstelle bereitstellt. QTensor unterstützt übliche mehrdimensionale Matrixoperationen, einschließlich Erzeugungsfunktionen, mathematische Funktionen, logische Funktionen, Matrixtransformationen usw.




Funktionen und Attribute von QTensor
******************************************


QTensor
==============================

.. py:class:: pyvqnet.tensor.tensor.QTensor(data, requires_grad=False, nodes=None, device=0, dtype=None, name='')

    Wrapper einer Datenstruktur mit dynamischer Erstellung von Berechnungsgraphen
    und automatischer Differentiation.

    :param data: _core.Tensor oder numpy-Array, das einen QTensor darstellt
    :param requires_grad: ob der Gradient des Tensors verfolgt werden soll, Standardwert False
    :param nodes: Liste der Nachfolger im Berechnungsgraph, Standardwert None
    :param device: aktuelles Gerät zum Speichern des QTensor, Standard = 0, CPU verwenden.
    :param dtype: Der Datentyp des Parameters, Standardwert None, verwendet den Standarddatentyp: kfloat32, der eine 32-Bit-Gleitkommazahl darstellt.
    :param name: Der Name des QTensor, Standard: "".

    :return: Ausgabe-QTensor


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

        Gibt die Anzahl der Dimensionen eines Tensors zurück.

        :return: Die Anzahl der Dimensionen eines Tensors.

        Example::

            from pyvqnet.tensor import QTensor

            a = QTensor([2.0, 3.0, 4.0, 5.0], requires_grad=True)
            print(a.ndim)

            # 1

    .. py:attribute:: shape

        Gibt die Dimensionen eines Tensors zurück.

        :return: Eine Liste der Dimensionen des Tensors

        Example::

            from pyvqnet.tensor import QTensor

            a = QTensor([2.0, 3.0, 4.0, 5.0], requires_grad=True)
            print(a.shape)

            # [4]

    .. py:attribute:: size

        Gibt die Anzahl der Elemente eines Tensors zurück.

        :return: Die Anzahl der Elemente eines Tensors.

        Example::

            from pyvqnet.tensor import QTensor

            a = QTensor([2.0, 3.0, 4.0, 5.0], requires_grad=True)
            print(a.size)

            # 4

    .. py:method:: numel

        Gibt die Anzahl der Elemente in einem Tensor zurück.

        :return: Die Anzahl der Elemente in einem Tensor.

        Example::

            from pyvqnet.tensor import QTensor

            a = QTensor([2.0, 3.0, 4.0, 5.0], requires_grad=True)
            print(a.numel())

            # 4

    .. py:attribute:: dtype

        Gibt den Datentyp eines Tensors zurück.

        Die unterstützten Datentypen sind wie folgt:

            =========================================  ===============================
            dtype                                      Beschreibung
            =========================================  ===============================
            ``pyvqnet.kbool``                          Boolescher Wert
            ``pyvqnet.kuint8``                         8-Bit-Integer (vorzeichenlos)
            ``pyvqnet.kint8``                          8-Bit-Integer (mit Vorzeichen)
            ``pyvqnet.kint16``                         16-Bit-Integer (mit Vorzeichen)
            ``pyvqnet.kint32``                         32-Bit-Integer (mit Vorzeichen)
            ``pyvqnet.kint64``                         64-Bit-Integer (mit Vorzeichen)
            ``pyvqnet.kfloat32``                       32-Bit-Gleitkommazahl, siehe https://en.wikipedia.org/wiki/IEEE_754
            ``pyvqnet.kfloat64``                       64-Bit-Gleitkommazahl, siehe https://en.wikipedia.org/wiki/IEEE_754
            ``pyvqnet.kcomplex64``                     64-Bit-Komplexzahl, bestehend aus zwei `float32`
            ``pyvqnet.kcomplex128``                    128-Bit-Komplexzahl, bestehend aus zwei `float64`
            ``pyvqnet.kbfloat16``                      16-Bit-Gleitkommazahl, manchmal als Brain-Floating-Point-Format bezeichnet, mit Bit-Aufteilung von 1 Vorzeichenbit, 8 Exponentenbits und 7 Mantissenbits
            =========================================  ===============================

        :return: Der Datentyp des Tensors.

        Example::

            from pyvqnet.tensor import QTensor

            a = QTensor([2, 3, 4, 5])
            print(a.dtype)
            # 4


    .. py:method:: zero_grad()

        Setzt den Gradienten auf Null. Wird vom Optimierer im Optimierungsprozess verwendet.

        :return: None

        Example::

            from pyvqnet.tensor import tensor
            from pyvqnet.tensor import QTensor
            t3  =  QTensor([2.0,3.0,4.0,5.0],requires_grad = True)
            t3.zero_grad()
            print(t3.grad)

            # [0, 0, 0, 0]


 
    .. py:method:: backward(grad=None)

        Berechnet den Gradienten des aktuellen QTensor.

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

        Kopiert die eigenen Daten in ein neues numpy.array.

        :return: ein neues numpy.array, das die QTensor-Daten enth�lt

        .. note::

            numpy unterst�tzt den Typ bfloat16 nicht. Sie m�ssen zuerst in einen anderen von numpy unterst�tzten Datentyp wie float32 konvertieren, bevor Sie diese Schnittstelle aufrufen.

        Example::

            from pyvqnet.tensor import tensor
            from pyvqnet.tensor import QTensor
            t3  =  QTensor([2.0,3.0,4.0,5.0],requires_grad = True)
            t4 = t3.to_numpy()
            print(t4)

            # [2. 3. 4. 5.]

 
    .. py:method:: item()

            Gibt das einzige Element des QTensor zur�ck. L�st einen 'RuntimeError' aus, wenn der QTensor mehr als 1 Element enth�lt.

            :return: die einzigen Daten dieses Objekts

            Example::

                from pyvqnet.tensor import tensor

                t = tensor.ones([1])
                print(t.item())

                # 1.0

 
    .. py:method:: argmax(*kargs)

        Gibt die Indizes des Maximalwerts aller Elemente im Eingabe-QTensor zur�ck, oder
        Gibt die Indizes der Maximalwerte eines QTensor �ber eine Dimension zur�ck.

        :param dim: dim (int) - die zu reduzierende Dimension, akzeptiert nur eine einzelne Achse. Wenn dim == None, werden die Indizes des Maximalwerts aller Elemente im Eingabe-Tensor zur�ckgegeben. Der g�ltige dim-Bereich ist [-R, R), wobei R = ndim der Eingabe ist. Wenn dim < 0, funktioniert es wie dim + R.
        :param keepdims: ob die Dimension des Ausgabe-QTensor beibehalten wird oder nicht.

        :return: die Indizes des Maximalwerts im Eingabe-QTensor.

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

        Gibt die Indizes des Minimalwerts aller Elemente im Eingabe-QTensor zur�ck, oder
        Gibt die Indizes der Minimalwerte eines QTensor �ber eine Dimension zur�ck.

        :param dim: dim (int) - die zu reduzierende Dimension, akzeptiert nur eine einzelne Achse. Wenn dim == None, werden die Indizes des Minimalwerts aller Elemente im Eingabe-Tensor zur�ckgegeben. Der g�ltige dim-Bereich ist [-R, R), wobei R = ndim der Eingabe ist. Wenn dim < 0, funktioniert es wie dim + R.
        :param keepdims: ob die Dimension des Ausgabe-QTensor beibehalten wird oder nicht.

        :return: die Indizes des Minimalwerts im Eingabe-QTensor.

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

            F�llt den QTensor mit dem angegebenen Wert an Ort und Stelle.

            :param v: ein skalarer Wert
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

            Gibt True zur�ck, wenn alle QTensor-Werte ungleich Null sind.

            :return: True, wenn alle QTensor-Werte ungleich Null sind.

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

            Gibt True zur�ck, wenn irgendein QTensor-Wert ungleich Null ist.

            :return: True, wenn irgendein QTensor-Wert ungleich Null ist.

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

        F�llt einen QTensor mit Werten, die zuf�llig aus einer Binomialverteilung gezogen werden.

        Wenn die nach der Binomialverteilung zuf�llig erzeugten Daten gr��er als der Binarisierungsschwellwert sind, wird die Zahl an den entsprechenden Positionen des QTensor auf 1 gesetzt, andernfalls auf 0.

        :param v: Binarisierungsschwellwert
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

        F�llt einen QTensor mit Werten, die zuf�llig aus einer vorzeichenbehafteten Gleichverteilung gezogen werden.

        Skalierungsfaktor der von der vorzeichenbehafteten Gleichverteilung erzeugten Werte.

        :param v: ein skalarer Wert
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

        F�llt einen QTensor mit Werten, die zuf�llig aus einer Gleichverteilung gezogen werden.

        Skalierungsfaktor der von der Gleichverteilung erzeugten Werte.

        :param v: ein skalarer Wert
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

        F�llt einen QTensor mit Werten, die zuf�llig aus einer Normalverteilung gezogen werden.
        Mittelwert der Normalverteilung. Standardabweichung der Normalverteilung.
        Gibt an, ob der schnelle Mathematikmodus verwendet werden soll.

        :param m: Mittelwert der Normalverteilung
        :param s: Standardabweichung der Normalverteilung
        :param fast_math: True, wenn schnelle Mathematik verwendet werden soll
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

        Kehrt die Achsen eines Arrays um oder vertauscht sie. Wenn new_dims = None, werden die Dimensionen umgekehrt.

        :param new_dims: die neue Reihenfolge der Dimensionen (Liste von Integers).
        :return: Ergebnis-QTensor.

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

        Ändert die Form des Tensors und gibt einen neuen QTensor zurück.

        :param new_shape: die neue Form (Liste von Integers)
        :return: ein neuer QTensor

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

        �ndert die Form des aktuellen QTensor an Ort und Stelle. Diese Schnittstelle versucht zun�chst, die Transformation ohne �nderung der urspr�nglichen Speicherdaten durchzuf�hren. Wenn dies fehlschl�gt, werden die aktuellen Daten in den neuen Speicher kopiert.

        .. warning::

            Es wird empfohlen, die reshape-Schnittstelle zu verwenden. In einigen F�llen wird der tats�chliche zugrunde liegende Speicherort kopiert, anstatt die �nderung an Ort und Stelle vorzunehmen.

        :param new_shape: die neue Form (Liste von Integers)
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

            Holt die Daten des QTensor als NumPy-Array.

            :return: ein NumPy-Array

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

        Slicing-Indizierung von QTensor wird unterst�tzt, ebenso die Verwendung von QTensor als erweiterter Indexzugriff. Ein neuer QTensor wird zur�ckgegeben.

        Die Parameter start, stop und step k�nnen durch einen Doppelpunkt getrennt werden, wie start:stop:step, wobei start, stop und step Standardwerte haben k�nnen.

        Als 1-D-QTensor kann die Indizierung oder das Slicing nur auf einer einzelnen Achse durchgef�hrt werden.

        Als 2-D-QTensor und mehrdimensionaler QTensor kann die Indizierung oder das Slicing auf mehreren Achsen durchgef�hrt werden.

        Wenn Sie QTensor als Index f�r die erweiterte Indizierung verwenden, siehe numpy f�r `erweiterte Indizierung <https://docs.scipy.org/doc/numpy-1.10.1/reference/arrays.indexing.html>`_ .

        Wenn Ihr QTensor als Index das Ergebnis einer logischen Operation ist, f�hren Sie eine boolesche Indizierung durch.

        .. note:: 
            
            Wir verwenden eine Indexform wie a[3,4,1], aber die Form a[3][4][1] wird nicht unterst�tzt.

        :param item: Ein Integer oder QTensor als Index.

        :return: Ein neuer QTensor.

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

        Slicing-Indizierung von QTensor wird unterst�tzt, ebenso die Verwendung von QTensor als erweiterter Indexzugriff. Ein neuer QTensor wird zur�ckgegeben.

        Die Parameter start, stop und step k�nnen durch einen Doppelpunkt getrennt werden, wie start:stop:step, wobei start, stop und step Standardwerte haben k�nnen.

        Als 1-D-QTensor kann die Indizierung oder das Slicing nur auf einer einzelnen Achse durchgef�hrt werden.

        Als 2-D-QTensor und mehrdimensionaler QTensor kann die Indizierung oder das Slicing auf mehreren Achsen durchgef�hrt werden.

        Wenn Sie QTensor als Index f�r die erweiterte Indizierung verwenden, siehe numpy f�r `erweiterte Indizierung <https://docs.scipy.org/doc/numpy-1.10.1/reference/arrays.indexing.html>`_ .

        Wenn Ihr QTensor als Index das Ergebnis einer logischen Operation ist, f�hren Sie eine boolesche Indizierung durch.

        .. note:: 
            
            Wir verwenden eine Indexform wie a[3,4,1], aber die Form a[3][4][1] wird nicht unterst�tzt.

        :param item: Ein Integer oder QTensor als Index.

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

        Klont QTensor auf das angegebene GPU-Ger�t.

        device gibt das Ger�t an, auf dem die internen Daten gespeichert werden. Wenn device >= DEV_GPU_0, werden die Daten auf der GPU gespeichert.
        Wenn Ihr Computer �ber mehrere GPUs verf�gt, k�nnen Sie verschiedene Ger�te zur Datenspeicherung zuweisen.
        Zum Beispiel device = DEV_GPU_1, DEV_GPU_2, DEV_GPU_3, ... zeigt die Speicherung auf GPUs mit unterschiedlichen Seriennummern an.
        
        .. note::
            QTensor kann keine Berechnungen auf verschiedenen GPUs durchf�hren.
            Ein Cuda-Fehler wird ausgel�st, wenn Sie versuchen, einen QTensor auf einer GPU zu erstellen, deren ID die maximale Anzahl verifizierter GPUs �berschreitet.

        :param device: Das Ger�t, das aktuell den QTensor speichert, Standard=DEV_GPU_0,

        device = pyvqnet.DEV_GPU_0, gespeichert auf der ersten GPU, device = DEV_GPU_1,
        gespeichert auf der zweiten GPU, und so weiter.

        :return: Klont QTensor auf das GPU-Ger�t.

        Examples::

            from pyvqnet.tensor import QTensor
            a = QTensor([2])
            b = a.GPU()
            print(b.device)
            #1000

 

    .. py:method:: CPU()

        Klont QTensor auf das angegebene CPU-Ger�t.

        :return: Klont QTensor auf das CPU-Ger�t.

        Examples::

            from pyvqnet.tensor import QTensor
            a = QTensor([2])
            b = a.CPU()
            print(b.device)
            # 0

 
    .. py:method:: toGPU(device: int = DEV_GPU_0)

        Verschiebt QTensor auf das angegebene GPU-Ger�t.

        device gibt das Ger�t an, auf dem die internen Daten gespeichert werden. Wenn device >= DEV_GPU, werden die Daten auf der GPU gespeichert.
        Wenn Ihr Computer �ber mehrere GPUs verf�gt, k�nnen Sie verschiedene Ger�te zur Datenspeicherung zuweisen.
        Zum Beispiel device = DEV_GPU_1, DEV_GPU_2, DEV_GPU_3, ... zeigt die Speicherung auf GPUs mit unterschiedlichen Seriennummern an.

        .. note::

            QTensor kann keine Berechnungen auf verschiedenen GPUs durchf�hren. Ein Cuda-Fehler wird ausgel�st, wenn Sie versuchen, einen QTensor auf einer GPU zu erstellen, deren ID die maximale Anzahl verifizierter GPUs �berschreitet.

        :param device: Das Ger�t, das aktuell den QTensor speichert, Standard=DEV_GPU_0. device = pyvqnet.DEV_GPU_0, gespeichert auf der ersten GPU, device = DEV_GPU_1, gespeichert auf der zweiten GPU, und so weiter.
        :return: QTensor auf das GPU-Ger�t verschoben.

        Examples::

            from pyvqnet.tensor import QTensor
            a = QTensor([2])
            a = a.toGPU()
            print(a.device)
            #1000


    
    .. py:method:: toCPU()

        Verschiebt QTensor zur CPU.

        :return: QTensor auf das CPU-Ger�t verschoben.

        Examples::

            from pyvqnet.tensor import QTensor
            a = QTensor([2])
            b = a.toCPU()
            print(b.device)
            # 0

    
    .. py:method:: isGPU()

        Gibt an, ob die Daten dieses QTensor im GPU-Hostspeicher gespeichert sind.

        :return: Gibt an, ob die Daten dieses QTensor im GPU-Hostspeicher gespeichert sind.

        Examples::
        
            from pyvqnet.tensor import QTensor
            a = QTensor([2])
            a = a.isGPU()
            print(a)
            # False
 
    .. py:method:: isCPU()

        Gibt an, ob die Daten dieses QTensor im CPU-Hostspeicher gespeichert sind.

        :return: Gibt an, ob die Daten dieses QTensor im CPU-Hostspeicher gespeichert sind.

        Examples::
        
            from pyvqnet.tensor import QTensor
            a = QTensor([2])
            a = a.isCPU()
            print(a)
            # True


Erzeugungsfunktionen
*****************************************************


ones
==============================

.. py:function:: pyvqnet.tensor.ones(shape,device=0,dtype-None)

    Gibt einen Einsen-Tensor mit der angegebenen Form zur�ck.

    :param shape: Eingabeform
    :param device: Auf welchem Ger�t gespeichert werden soll, Standard 0, CPU.
    :param dtype: Der Datentyp des Parameters, Standardwert None, verwendet den Standarddatentyp: kfloat32, der eine 32-Bit-Gleitkommazahl darstellt.
    
    :return: Ausgabe-QTensor mit der angegebenen Form.

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

    Gibt einen Einsen-Tensor mit derselben Form wie der Eingabe-QTensor zur�ck.

    :param t: Eingabe-QTensor
    :param device: Auf welchem Ger�t gespeichert werden soll, Standard 0, CPU.
    :param dtype: Der Datentyp des Parameters, Standardwert None, verwendet den Standarddatentyp: kfloat32, der eine 32-Bit-Gleitkommazahl darstellt.
    
    :return: Ausgabe-QTensor


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

    Erstellt einen QTensor mit der angegebenen Form und f�llt ihn mit dem Wert.

    :param shape: Form des zu erstellenden QTensor
    :param value: Wert, mit dem der QTensor gef�llt werden soll.
    :param device: Zu verwendendes Ger�t, Standard = 0, CPU verwenden.
    :param dtype: Der Datentyp des Parameters, Standardwert None, verwendet den Standarddatentyp: kfloat32, der eine 32-Bit-Gleitkommazahl darstellt.
    
    :return: Ausgabe-QTensor

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

    Erstellt einen QTensor mit der angegebenen Form und f�llt ihn mit dem Wert.

    :param t: Eingabe-QTensor
    :param value: Wert, mit dem der QTensor gef�llt werden soll.
    :param device: Zu verwendendes Ger�t, Standard = 0, CPU verwenden.
    :param dtype: Der Datentyp des Parameters, Standardwert None, verwendet den Standarddatentyp: kfloat32, der eine 32-Bit-Gleitkommazahl darstellt.
    
    :return: Ausgabe-QTensor

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

    Gibt einen Null-Tensor mit der angegebenen Form zur�ck.

    :param shape: Form des Tensors
    :param device: Zu verwendendes Ger�t, Standard = 0, CPU verwenden.
    :param dtype: Der Datentyp des Parameters, Standardwert None, verwendet den Standarddatentyp: kfloat32, der eine 32-Bit-Gleitkommazahl darstellt.
    
    :return: Ausgabe-QTensor

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

    Gibt einen Null-Tensor mit derselben Form wie der Eingabe-QTensor zur�ck.

    :param t: Eingabe-QTensor
    :param device: Zu verwendendes Ger�t, Standard = 0, CPU verwenden.
    :param dtype: Der Datentyp des Parameters, Standardwert None, verwendet den Standarddatentyp: kfloat32, der eine 32-Bit-Gleitkommazahl darstellt.
    
    :return: Ausgabe-QTensor

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

    Erstellt einen 1D-QTensor mit gleichm��ig verteilten Werten innerhalb eines gegebenen Intervalls.

    :param start: Start des Intervalls
    :param end: Ende des Intervalls
    :param step: Abstand zwischen den Werten
    :param device: Zu verwendendes Ger�t, Standard = 0, CPU verwenden.
    :param dtype: Der Datentyp des Parameters, Standardwert None, verwendet den Standarddatentyp: kfloat32, der eine 32-Bit-Gleitkommazahl darstellt.
    :param requires_grad: ob der Gradient des Tensors verfolgt werden soll, Standardwert False
    :return: Ausgabe-QTensor

    Example::

        from pyvqnet.tensor import tensor
        from pyvqnet.tensor import QTensor
        t = tensor.arange(2, 30,4)
        print(t)

        # [ 2,  6, 10, 14, 18, 22, 26]

linspace
==============================

.. py:function:: pyvqnet.tensor.linspace(start, end, num, device: int = 0,dtype=None, requires_grad= False)

    Erstellt einen 1D-QTensor mit gleichm��ig verteilten Werten innerhalb eines gegebenen Intervalls.

    :param start: Startwert
    :param end: Endwert
    :param nums: Anzahl der zu erzeugenden Stichproben
    :param device: Zu verwendendes Ger�t, Standard = 0, CPU verwenden.
    :param dtype: Der Datentyp des Parameters, Standardwert None, verwendet den Standarddatentyp: kfloat32, der eine 32-Bit-Gleitkommazahl darstellt.
    :param requires_grad: ob der Gradient des Tensors verfolgt werden soll, Standardwert False
    :return: Ausgabe-QTensor

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

    Erstellt einen 1D-QTensor mit gleichm��ig verteilten Werten auf einer logarithmischen Skala.

    :param start: ``base ** start`` ist der Startwert
    :param end: ``base ** end`` ist der Endwert der Sequenz
    :param nums: Anzahl der zu erzeugenden Stichproben
    :param base: die Basis des logarithmischen Raums
    :param device: Zu verwendendes Ger�t, Standard = 0, CPU verwenden.
    :param dtype: Der Datentyp des Parameters, Standardwert None, verwendet den Standarddatentyp: kfloat32, der eine 32-Bit-Gleitkommazahl darstellt.
    :param requires_grad: ob der Gradient des Tensors verfolgt werden soll, Standardwert False
    :return: Ausgabe-QTensor

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

    Erstellt einen size x size QTensor mit Einsen auf der Diagonalen und Nullen
    ansonsten.

    :param size: Gr��e des (quadratischen) zu erstellenden QTensor
    :param offset: Index der Diagonale: 0 (Standard) bezieht sich auf die Hauptdiagonale, ein positiver Wert auf eine obere Diagonale und ein negativer Wert auf eine untere Diagonale.
    :param device: Zu verwendendes Ger�t, Standard = 0, CPU verwenden.
    :param dtype: Der Datentyp des Parameters, Standardwert None, verwendet den Standarddatentyp: kfloat32, der eine 32-Bit-Gleitkommazahl darstellt.
    
    :return: Ausgabe-QTensor

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


    Gibt eine partielle Ansicht von :attr:`t` zur�ck, bei der die Diagonalenelemente als Dimensionen am Ende der Form relativ zu :attr:`dim1` und :attr:`dim2` angeh�ngt werden.
    :attr:`offset` ist der Versatz der Hauptdiagonale.

    :param t: Eingabe-Tensor
    :param offset: Versatz (0 bedeutet Hauptdiagonale, positive Werte bedeuten die n-te Diagonale oberhalb der Hauptdiagonale, negative Werte bedeuten die n-te Diagonale unterhalb der Hauptdiagonale)
    :param dim1: erste Dimension, aus der die Diagonale entnommen wird. Standard: 0.
    :param dim2: zweite Dimension, aus der die Diagonale entnommen wird. Standard: 1.

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

    W�hlt Diagonalelemente aus oder erstellt einen diagonalen QTensor.

    Bei Eingabe eines 2-D-QTensor wird ein neuer 1D-Tensor zur�ckgegeben, der die ausgew�hlten Diagonalelemente enth�lt. Bei Eingabe eines 1-D-QTensor wird ein neuer 2D-Tensor zur�ckgegeben, dessen ausgew�hlte Diagonalelemente die Eingabewerte sind und der Rest 0 ist.

    :param t: Eingabe-QTensor
    :param k: Versatz (0 f�r die Hauptdiagonale, positiv f�r die n-te
        Diagonale oberhalb der Hauptdiagonale, negativ f�r die n-te Diagonale unterhalb der
        Hauptdiagonale)
    :return: Ausgabe-QTensor

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

    Erstellt einen QTensor mit gleichverteilten Zufallswerten.

    :param shape: Form des zu erstellenden QTensor
    :param min: Minimalwert der Gleichverteilung, Standard: 0.
    :param max: Maximalwert der Gleichverteilung, Standard: 1.
    :param device: Zu verwendendes Ger�t, Standard = 0, CPU verwenden.
    :param dtype: Der Datentyp des Parameters, Standardwert None, verwendet den Standarddatentyp: kfloat32, der eine 32-Bit-Gleitkommazahl darstellt.
    :param requires_grad: ob der Gradient des Tensors verfolgt werden soll, Standardwert False
    :return: Ausgabe-QTensor


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

    Erstellt einen QTensor mit normalverteilten Zufallswerten.

    :param shape: Form des zu erstellenden QTensor
    :param mean: Mittelwert der Normalverteilung, Standard: 0.
    :param std: Standardabweichung der Normalverteilung, Standard: 1.
    :param device: Zu verwendendes Ger�t, Standard = 0, CPU verwenden.
    :param dtype: Der Datentyp des Parameters, Standardwert None, verwendet den Standarddatentyp: kfloat32, der eine 32-Bit-Gleitkommazahl darstellt.
    :param requires_grad: ob der Gradient des Tensors verfolgt werden soll, Standardwert False
    :return: Ausgabe-QTensor

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
    
    Erstellt eine Binomialverteilung, parametrisiert durch :attr:total_count und :attr:probs.

    :param total_counts: Anzahl der Bernoulli-Versuche.
    :param probs: Ereigniswahrscheinlichkeiten.

    :return:
        QTensor f�r die Binomialverteilung.

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

    Gibt einen Tensor zur�ck, in dem jede Zeile num_samples indizierte Stichproben enth�lt.
    Aus der multinomialen Wahrscheinlichkeitsverteilung, die sich in der entsprechenden Zeile der Tensoreingabe befindet.

    :param t: Eingabe-Wahrscheinlichkeitsverteilung.
    :param num_samples: Anzahl der Stichproben.

    :return:
        Ausgabe-Stichprobenindex

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

    Gibt die obere Dreiecksmatrix der Eingabe t zur�ck, der Rest wird auf 0 gesetzt.

    :param t: Eingabe eines QTensor
    :param diagonal: Der Versatz, Standard =0. Hauptdiagonale ist 0, positiv ist Versatz nach oben, negativ ist Versatz nach unten.

    :return: Ausgabe eines QTensor

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

    Gibt die untere Dreiecksmatrix der Eingabe t zur�ck, der Rest wird auf 0 gesetzt.

    :param t: Eingabe eines QTensor
    :param diagonal: Der Versatz, Standard =0. Hauptdiagonale ist 0, positiv ist Versatz nach oben, negativ ist Versatz nach unten.

    :return: Ausgabe eines QTensor

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


Mathematische Funktionen
*****************************************************


floor
==============================

.. py:function:: pyvqnet.tensor.floor(t)

    Gibt einen neuen QTensor mit der Abrundung der Elemente der Eingabe zur�ck, der gr��ten ganzen Zahl, die kleiner oder gleich jedem Element ist.

    :param t: Eingabe-QTensor
    :return: Ausgabe-QTensor

    Example::

        from pyvqnet.tensor import tensor

        t = tensor.arange(-2.0, 2.0, 0.25)
        u = tensor.floor(t)
        print(u)

        # [-2, -2, -2, -2, -1, -1, -1, -1, 0, 0, 0, 0, 1, 1, 1, 1]

ceil
==============================

.. py:function:: pyvqnet.tensor.ceil(t)

    Gibt einen neuen QTensor mit der Aufrundung der Elemente der Eingabe zur�ck, der kleinsten ganzen Zahl, die gr��er oder gleich jedem Element ist.

    :param t: Eingabe-QTensor
    :return: Ausgabe-QTensor

    Example::

        from pyvqnet.tensor import tensor

        t = tensor.arange(-2.0, 2.0, 0.25)
        u = tensor.ceil(t)
        print(u)

        # [-2, -1, -1, -1, -1, -0, -0, -0, 0, 1, 1, 1, 1, 2, 2, 2]

round
==============================

.. py:function:: pyvqnet.tensor.round(t)

    Rundet QTensor-Werte auf die n�chste ganze Zahl.

    :param t: Eingabe-QTensor
    :return: Ausgabe-QTensor

    Example::

        from pyvqnet.tensor import tensor

        t = tensor.arange(-2.0, 2.0, 0.4)
        u = tensor.round(t)
        print(u)

        # [-2, -2, -1, -1, -0, -0, 0, 1, 1, 2]

sort
==============================

.. py:function:: pyvqnet.tensor.sort(t, axis: int, descending=False, stable=True)

    Sortiert QTensor entlang der Achse.

    :param t: Eingabe-QTensor
    :param axis: Sortierachse
    :param descending: Sortierreihenfolge, wenn absteigend
    :param stable: Ob stabile Sortierung verwendet werden soll oder nicht
    :return: Ausgabe-QTensor

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

    Gibt ein Array von Indizes derselben Form wie die Eingabe zur�ck, die Daten entlang der angegebenen Achse in sortierter Reihenfolge indizieren.

    :param t: Eingabe-QTensor
    :param axis: Sortierachse
    :param descending: Sortierreihenfolge, wenn absteigend
    :param stable: Ob stabile Sortierung verwendet werden soll oder nicht
    :return: Ausgabe-QTensor

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

    Gibt die k gr��ten Elemente des Eingabe-Tensors entlang der angegebenen Achse zur�ck.

    Wenn if_descent False ist, werden die k kleinsten Elemente zur�ckgegeben.

    :param t: Eingabe eines QTensor
    :param k: Anzahl der gr��ten oder kleinsten Elemente
    :param axis: Sortierachse, Standard = -1, die letzte Achse
    :param if_descent: Sortierreihenfolge, Standardwert True

    :return: Ein neuer QTensor

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

    Gibt den Index der k gr��ten Elemente entlang der angegebenen Achse des Eingabe-Tensors zur�ck.

    Wenn if_descent False ist, wird der Index der k kleinsten Elemente zur�ckgegeben.

    :param t: Eingabe eines QTensor
    :param k: Anzahl der gr��ten oder kleinsten Elemente
    :param axis: Sortierachse, Standard = -1, die letzte Achse
    :param if_descent: Sortierreihenfolge, Standardwert True

    :return: Ein neuer QTensor

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

    Addiert zwei QTensoren elementweise, entspricht t1 + t2.

    :param t1: erster QTensor
    :param t2: zweiter QTensor
    :return: Ausgabe-QTensor

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

    Subtrahiert zwei QTensoren elementweise, entspricht t1 - t2.


    :param t1: erster QTensor
    :param t2: zweiter QTensor
    :return: Ausgabe-QTensor

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

    Multipliziert zwei QTensoren elementweise, entspricht t1 * t2.

    :param t1: erster QTensor
    :param t2: zweiter QTensor
    :return: Ausgabe-QTensor


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

    Dividiert zwei QTensoren elementweise, entspricht t1 / t2.


    :param t1: erster QTensor
    :param t2: zweiter QTensor
    :return: Ausgabe-QTensor


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

    Summiert alle Elemente im QTensor entlang der angegebenen Achse. Wenn axis = None, werden alle Elemente im QTensor summiert.

    :param t: Eingabe-QTensor
    :param axis: Achse, die f�r die Summierung verwendet wird, Standardwert None
    :param keepdims: ob die Dimension des Ausgabe-Tensors beibehalten wird oder nicht. Standardwert False
    :return: Ausgabe-QTensor


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

    Gibt die kumulative Summe der Eingabeelemente in der Dimensionsachse zur�ck.

    :param t: der Eingabe-QTensor
    :param axis: Achse f�r die Berechnung, Standardwert -1, die letzte Achse

    :return: Ausgabe-QTensor.

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

    Ermittelt die Mittelwerte im QTensor entlang der Achse.

    :param t: der Eingabe-QTensor.
    :param axis: die zu reduzierende Dimension.
    :param keepdims: ob die Dimension des Ausgabe-QTensor beibehalten wird oder nicht, Standardwert False.
    :return: gibt den Mittelwert des Eingabe-QTensor zur�ck.

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

    Ermittelt den Medianwert im QTensor.

    :param t: der Eingabe-QTensor
    :param axis: eine Achse f�r die Mittelung, Standardwert None
    :param keepdims: ob die Dimension des Ausgabe-QTensor beibehalten wird oder nicht, Standardwert False

    :return: Gibt den Median der Werte in der Eingabe oder dem QTensor zur�ck.

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

    Ermittelt die Standardabweichung im QTensor.


    :param t: der Eingabe-QTensor
    :param axis: die Achse zur Berechnung der Standardabweichung, Standardwert None
    :param keepdims: ob die Dimension des Ausgabe-QTensor beibehalten wird oder nicht, Standardwert False
    :param unbiased: ob die Bessel-Korrektur verwendet werden soll, Standardwert True
    :return: Gibt die Standardabweichung der Werte in der Eingabe oder dem QTensor zur�ck

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

    Ermittelt die Varianz im QTensor.


    :param t: der Eingabe-QTensor.
    :param axis: Die Achse zur Berechnung der Varianz, Standardwert None
    :param keepdims: ob die Dimension des Ausgabe-QTensor beibehalten wird oder nicht, Standardwert False.
    :param unbiased: ob die Bessel-Korrektur verwendet werden soll, Standardwert True.


    :return: Ermittelt die Varianz im QTensor.

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

    Matrixmultiplikationen von zwei 2D-, 3D-, 4D-Matrizen.

    :param t1: erster QTensor
    :param t2: zweiter QTensor
    :return: Ausgabe-QTensor

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

    Berechnet das Kronecker-Produkt von ``t1`` und ``t2``, ausgedr�ckt als :math:`\\otimes` . Wenn ``t1`` ein :math:`(a_0 \\times a_1 \\times \\dots \\times a_n)`-Tensor ist und ``t2`` ein :math:`(b_0 \\times b_1 \\times \\dots \\times b_n)`-Tensor, wird das Ergebnis ein :math:`(a_0*b_0 \\times a_1*b_1 \\times \\dots \\times a_n*b_n)`-Tensor mit den folgenden Eintr�gen sein:
    
    .. math::
          (\\text{input} \\otimes \\text{other})_{k_0, k_1, \\dots, k_n} =
              \\text{input}_{i_0, i_1, \\dots, i_n} * \\text{other}_{j_0, j_1, \\dots, j_n},

    wobei :math:`k_t = i_t * b_t + j_t` und :math:`0 \\leq t \\leq n`.
    Wenn ein Tensor weniger Dimensionen als der andere hat, wird er entpackt, bis er die gleiche Dimensionalit�t hat.

    :param t1: Der erste QTensor.
    :param t2: Der zweite QTensor.
    
    :return: Ausgabe-QTensor.

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
    
    Summiert die Produkte der Elemente der Eingabeoperanden entlang der angegebenen Dimension unter Verwendung einer Notation, die auf der Einstein'schen Summenkonvention basiert.

    .. note::

        Diese Funktion verwendet opt_einsum (https://optimized-einsum.readthedocs.io/en/stable/), um die Berechnung zu beschleunigen oder den Speicherverbrauch durch Optimierung der Kontraktionsreihenfolge zu reduzieren. Diese Optimierung erfolgt, wenn mindestens drei Eingaben vorhanden sind.

        F�r komplexere `einsum`-Operationen kann opt_einsum zus�tzlich importiert werden, um direkt auf QTensor zu rechnen.

    :param equation: Das Subskript der Einstein-Summe.
    :param operands: Der Tensor, auf dem die Einstein-Summe berechnet werden soll.

    :return:

        Das QTensor-Ergebnis.

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

    Berechnet den elementweisen Kehrwert des QTensor.

    :param t: Eingabe-QTensor
    :return: Ausgabe-QTensor

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

    Gibt einen neuen QTensor mit den Vorzeichen der Elemente der Eingabe zur�ck. Die Sign-Funktion gibt -1 zur�ck, wenn t < 0, 0, wenn t==0, 1, wenn t > 0.

    :param t: Eingabe-QTensor
    :return: Ausgabe-QTensor


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

    Un�re Negation der QTensor-Elemente.

    :param t: Eingabe-QTensor
    :return: Ausgabe-QTensor

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

    Gibt die Summe der Elemente der Diagonalen der eingegebenen 2-D-Matrix zur�ck.

    :param t: Eingabe-2-D-QTensor
    :param k: Versatz (0 f�r die Hauptdiagonale, positiv f�r die n-te
        Diagonale oberhalb der Hauptdiagonale, negativ f�r die n-te Diagonale unterhalb der
        Hauptdiagonale)
    :return: die Summe der Elemente der Diagonalen der eingegebenen 2-D-Matrix

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

    Wendet die Exponentialfunktion auf alle Elemente des Eingabe-QTensor an.

    :param t: Eingabe-QTensor
    :return: Ausgabe-QTensor

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

    Berechnet den elementweisen Arkuskosinus des QTensor.

    :param t: Eingabe-QTensor
    :return: Ausgabe-QTensor

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

    Berechnet den elementweisen Arkussinus des QTensor.

    :param t: Eingabe-QTensor
    :return: Ausgabe-QTensor

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

    Berechnet den elementweisen Arkustangens des QTensor.

    :param t: Eingabe-QTensor
    :return: Ausgabe-QTensor

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

    Wendet die Sinusfunktion auf alle Elemente des Eingabe-QTensor an.


    :param t: Eingabe-QTensor
    :return: Ausgabe-QTensor

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

    Wendet die Kosinusfunktion auf alle Elemente des Eingabe-QTensor an.


    :param t: Eingabe-QTensor
    :return: Ausgabe-QTensor

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

    Wendet die Tangensfunktion auf alle Elemente des Eingabe-QTensor an.


    :param t: Eingabe-QTensor
    :return: Ausgabe-QTensor

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

    Wendet die hyperbolische Tangensfunktion auf alle Elemente des Eingabe-QTensor an.

    :param t: Eingabe-QTensor
    :return: Ausgabe-QTensor

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

    Wendet die hyperbolische Sinusfunktion auf alle Elemente des Eingabe-QTensor an.


    :param t: Eingabe-QTensor
    :return: Ausgabe-QTensor

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

    Wendet die hyperbolische Kosinusfunktion auf alle Elemente des Eingabe-QTensor an.


    :param t: Eingabe-QTensor
    :return: Ausgabe-QTensor

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

    Potenziert den ersten QTensor mit dem zweiten QTensor.

    :param t1: erster QTensor
    :param t2: zweiter QTensor
    :return: Ausgabe-QTensor

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

    Wendet die Absolutwertfunktion auf alle Elemente des Eingabe-QTensor an.

    :param t: Eingabe-QTensor
    :return: Ausgabe-QTensor

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

    Wendet die Logarithmusfunktion (ln) auf alle Elemente des Eingabe-QTensor an.

    :param t: Eingabe-QTensor
    :return: Ausgabe-QTensor

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
    
    Berechnet nacheinander die Ergebnisse der Softmax-Funktion und der Logarithmusfunktion auf der Achse axis.

    :param t: Eingabe-QTensor.
    :param axis: Die Achse, die zur Berechnung von softmax verwendet wird, Standard ist -1.

    :return: Ausgabe-QTensor.

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

    Wendet die Quadratwurzelfunktion auf alle Elemente des Eingabe-QTensor an.


    :param t: Eingabe-QTensor
    :return: Ausgabe-QTensor

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

    Wendet die Quadratfunktion auf alle Elemente des Eingabe-QTensor an.


    :param t: Eingabe-QTensor
    :return: Ausgabe-QTensor

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
 
    Gibt die Eigenwerte und Eigenvektoren einer komplexen Hermiteschen (konjugiert symmetrischen) oder reellen symmetrischen Matrix zur�ck.

    Gibt zwei Objekte zur�ck, ein 1D-Array mit den Eigenwerten von a,
    und eine 2D-quadratische Matrix oder Matrix (je nach Eingabetyp) der entsprechenden Eigenvektoren (in Spalten).

    :param: Eingabe-QTensor.
    :param: Eigenwerte und Eigenvektoren von t.
    :return:

        Gibt Eigenwerte und Eigenvektoren zur�ck

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

    Berechnet die F-Norm des Tensors auf dem Eingabe-QTensor entlang der durch axis festgelegten Achse.
    Wenn axis None ist, wird die F-Norm aller Elemente zur�ckgegeben.

    :param t: Eingabe-QTensor.
    :param axis: Die Achse zur Berechnung der F-Norm, Standard ist None.
    :param keepdims: Gibt an, ob der Ausgabe-Tensor die reduzierte Dimensionalit�t beibeh�lt. Standard ist False.
    :return: Gibt einen QTensor oder F-Norm-Wert zur�ck.


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



Logische Funktionen
**************************

maximum
==============================

.. py:function:: pyvqnet.tensor.maximum(t1: pyvqnet.tensor.QTensor, t2: pyvqnet.tensor.QTensor)

    Elementweises Maximum zweier Tensoren.


    :param t1: erster QTensor
    :param t2: zweiter QTensor
    :return: Ausgabe-QTensor

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

    Elementweises Minimum zweier Tensoren.


    :param t1: erster QTensor
    :param t2: zweiter QTensor
    :return: Ausgabe-QTensor

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

    Gibt die minimalen Elemente des Eingabe-QTensor entlang der angegebenen Achse zur�ck.
    Wenn axis == None, wird der minimale Wert aller Elemente im Tensor zur�ckgegeben.

    :param t: Eingabe-QTensor
    :param axis: Achse f�r das Minimum, Standardwert None
    :param keepdims: ob die Dimension des Ausgabe-Tensors beibehalten wird oder nicht. Standardwert False
    :return: Ausgabe-QTensor

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

    Gibt die maximalen Elemente des Eingabe-QTensor entlang der angegebenen Achse zur�ck.
    Wenn axis == None, wird der maximale Wert aller Elemente im Tensor zur�ckgegeben.

    :param t: Eingabe-QTensor
    :param axis: Achse f�r das Maximum, Standardwert None
    :param keepdims: ob die Dimension des Ausgabe-Tensors beibehalten wird oder nicht. Standardwert False
    :return: Ausgabe-QTensor

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

    Begrenzt den Eingabe-QTensor auf Minimal- und Maximalwert.

    :param t: Eingabe-QTensor
    :param min_val: Minimalwert
    :param max_val: Maximalwert
    :return: Ausgabe-QTensor

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

    Gibt Elemente aus x oder y zur�ck, abh�ngig von der Bedingung.

    :param condition: Bedingungs-Tensor, muss den Datentyp kbool haben.
    :param t1: QTensor, aus dem Elemente entnommen werden, wenn die Bedingung erf�llt ist, Standardwert None
    :param t2: QTensor, aus dem Elemente entnommen werden, wenn die Bedingung nicht erf�llt ist, Standardwert None
    :return: Ausgabe-QTensor

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

    Gibt einen QTensor zur�ck, der die Indizes der Nicht-Null-Elemente enth�lt.

    :param t: Eingabe-QTensor
    :return: Ausgabe-QTensor enth�lt Indizes von Nicht-Null-Elementen.

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

    Testet elementweise auf Endlichkeit (nicht unendlich und nicht keine Zahl).

    :param t: Eingabe-QTensor
    :return: Ausgabe-QTensor, der True zur�ckgibt, wenn das entsprechende Positionselement die Bedingung erf�llt, andernfalls False.

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

    Testet elementweise auf positive oder negative Unendlichkeit.

    :param t: Eingabe-QTensor
    :return: Ausgabe-QTensor, der True zur�ckgibt, wenn das entsprechende Positionselement die Bedingung erf�llt, andernfalls False.
    
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

    Testet elementweise auf NaN (keine Zahl).

    :param t: Eingabe-QTensor
    :return: Ausgabe-QTensor, der True zur�ckgibt, wenn das entsprechende Positionselement die Bedingung erf�llt, andernfalls False.
    
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

    Testet elementweise auf negative Unendlichkeit.

    :param t: Eingabe-QTensor
    :return: Ausgabe-QTensor, der True zur�ckgibt, wenn das entsprechende Positionselement die Bedingung erf�llt, andernfalls False.
    
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

    Testet elementweise auf positive Unendlichkeit.

    :param t: Eingabe-QTensor
    :return: Ausgabe-QTensor, der True zur�ckgibt, wenn das entsprechende Positionselement die Bedingung erf�llt, andernfalls False.
    
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

    Berechnet den Wahrheitswert von ``t1 and t2`` elementweise.

    :param t1: Eingabe-QTensor
    :param t2: Eingabe-QTensor
    :return: Ausgabe-QTensor, der True zur�ckgibt, wenn das entsprechende Positionselement die Bedingung erf�llt, andernfalls False.
    
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

    Berechnet den Wahrheitswert von ``t1 or t2`` elementweise.

    :param t1: Eingabe-QTensor
    :param t2: Eingabe-QTensor
    :return: Ausgabe-QTensor, der True zur�ckgibt, wenn das entsprechende Positionselement die Bedingung erf�llt, andernfalls False.

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

    Berechnet den Wahrheitswert von ``not t`` elementweise.

    :param t: Eingabe-QTensor
    :return: Ausgabe-QTensor, der True zur�ckgibt, wenn das entsprechende Positionselement die Bedingung erf�llt, andernfalls False.

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

    Berechnet den Wahrheitswert von ``t1 xor t2`` elementweise.

    :param t1: Eingabe-QTensor
    :param t2: Eingabe-QTensor

    :return: Ausgabe-QTensor, der True zur�ckgibt, wenn das entsprechende Positionselement die Bedingung erf�llt, andernfalls False.

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

    Gibt den Wahrheitswert von ``t1 > t2`` elementweise zur�ck.


    :param t1: Eingabe-QTensor
    :param t2: Eingabe-QTensor
    :return: Ausgabe-QTensor, der True zur�ckgibt, wenn das entsprechende Positionselement die Bedingung erf�llt, andernfalls False.

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

    Gibt den Wahrheitswert von ``t1 >= t2`` elementweise zur�ck.

    :param t1: Eingabe-QTensor
    :param t2: Eingabe-QTensor
    :return: Ausgabe-QTensor, der True zur�ckgibt, wenn das entsprechende Positionselement die Bedingung erf�llt, andernfalls False.

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

    Gibt den Wahrheitswert von ``t1 < t2`` elementweise zur�ck.

    :param t1: Eingabe-QTensor
    :param t2: Eingabe-QTensor
    :return: Ausgabe-QTensor, der True zur�ckgibt, wenn das entsprechende Positionselement die Bedingung erf�llt, andernfalls False.

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

    Gibt den Wahrheitswert von ``t1 <= t2`` elementweise zur�ck.

    :param t1: Eingabe-QTensor
    :param t2: Eingabe-QTensor
    :return: Ausgabe-QTensor, der True zur�ckgibt, wenn das entsprechende Positionselement die Bedingung erf�llt, andernfalls False.

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

    Gibt den Wahrheitswert von ``t1 == t2`` elementweise zur�ck.

    :param t1: Eingabe-QTensor
    :param t2: Eingabe-QTensor
    :return: Ausgabe-QTensor, der True zur�ckgibt, wenn das entsprechende Positionselement die Bedingung erf�llt, andernfalls False.
    
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

    Gibt den Wahrheitswert von ``t1 != t2`` elementweise zur�ck.

    :param t1: Eingabe-QTensor
    :param t2: Eingabe-QTensor
    :return: Ausgabe-QTensor, der True zur�ckgibt, wenn das entsprechende Positionselement die Bedingung erf�llt, andernfalls False.
    
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
 
    F�hrt das bitweise UND zweier QTensor-Elemente durch.

    :param t1: Eingabe-QTensor t1. Nur Integer oder Boolesche Werte sind g�ltige Eingaben.
    :param t2: Eingabe-QTensor t2. Nur Integer oder Boolesche Werte sind g�ltige Eingaben.
    :return:
        Ergebnis-QTensor

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


Matrixoperationen
**********************

select
==============================

.. py:function:: pyvqnet.tensor.select(t: pyvqnet.tensor.QTensor, index)

    Gibt einen QTensor im QTensor an der angegebenen Achse zur�ck. Die folgende Operation liefert das gleiche Ergebnis.

    :param t: Eingabe-QTensor
    :param index: Ein String, der die Ausgabedimension enth�lt
    :return: Ausgabe-QTensor

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

    Vorbehaltlich bestimmter Einschr�nkungen werden kleinere Arrays in gr��eren Arrays platziert, sodass sie kompatible Formen haben. Diese Schnittstelle kann automatische Differentiation auf den Eingabe-Parameter-Tensoren durchf�hren.

    Referenz https://numpy.org/doc/stable/user/basics.broadcasting.html

    :param t1: Eingabe-QTensor 1
    :param t2: Eingabe-QTensor 2

    :return t11: t1 mit neuer Broadcast-Form.
    :return t22: t2 mit neuer Broadcast-Form.

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

    Verkettet die Eingabe-QTensoren entlang der Achse und gibt einen neuen QTensor zur�ck.

    :param args: Liste von Eingabe-QTensoren
    :param axis: Dimension zum Verketten. Muss zwischen 0 und der Anzahl der Dimensionen der zu verkettenden Tensoren liegen.
    :return: Ausgabe-QTensor

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

    F�gt eine Sequenz von Arrays entlang einer neuen Achse zusammen und gibt einen neuen QTensor zur�ck.

    :param QTensors: Liste von QTensoren
    :param axis: Dimension zum Einf�gen. Muss zwischen 0 und der Anzahl der Dimensionen der zu stapelnden Tensoren liegen.
    :return: Ausgabe-QTensor

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

    Kehrt die Achsen eines Arrays um oder vertauscht sie.

    :param t: Eingabe-QTensor
    :param dim: die neue Reihenfolge der Dimensionen (Liste von Integers)
    :return: Ausgabe-QTensor

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

    Transponiert die Achsen eines Arrays. Wenn dim = None, werden die Dimensionen umgekehrt. Diese Funktion ist identisch mit permute.

    :param t: Eingabe-QTensor
    :param dim: die neue Reihenfolge der Dimensionen (Liste von Integers)
    :return: Ausgabe-QTensor

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

    Erstellt einen QTensor durch wiederholtes Aneinanderh�ngen des QTensor entsprechend der Anzahl in reps.

    Wenn reps die L�nge d hat, hat der Ergebnis-QTensor die Dimension max(d, t.ndim).

    Wenn t.ndim < d, wird t auf d-dimensional erweitert, indem neue Achsen von der Anfangsdimension eingef�gt werden.
    So wird ein Array der Form (3,) f�r 2-D-Wiederholung auf (1, 3) erh�ht, oder auf Form (1, 1, 3) f�r 3-D-Wiederholung.

    Wenn t.ndim > d, wird reps auf t.ndim erweitert, indem 1-en eingef�gt werden.

    F�r einen Tensor t der Form (2, 3, 4, 5) wird reps (4, 3) als (1, 1, 4, 3) behandelt.

    :param t: Eingabe-QTensor
    :param reps: die Anzahl der Wiederholungen pro Dimension.
    :return: ein neuer QTensor

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

    Entfernt Achsen der L�nge eins.

    :param t: Eingabe-QTensor
    :param axis: Stauchungsachse, wenn axis = -1, werden alle Dimensionen mit der Gr��e 1 gestaucht.
    :return: Ausgabe-QTensor

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

    Gibt einen neuen QTensor mit einer Dimension der Gr��e eins an der angegebenen Position zur�ck.

    :param t: Eingabe-QTensor
    :param axis: Achse f�r die Erweiterung, an der die Dimension eingef�gt wird.
    :return: Ausgabe-QTensor

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

    Verschiebt Dimensionen von `t` von Positionen in `source` zu Positionen in `destination`.

    Andere Dimensionen von `t`, die nicht explizit verschoben werden, behalten ihre urspr�ngliche Reihenfolge und erscheinen an nicht in `destination` angegebenen Positionen.

    :param t: Eingabe-QTensor.
    :param source: (Integer oder Tupel von Integers) Die urspr�nglichen Positionen der zu verschiebenden Dimensionen. Diese Positionen m�ssen eindeutig sein.
    :param destination: (Integer oder Tupel von Integers) Die Zielpositionen f�r jede urspr�ngliche Dimension. Diese Positionen m�ssen ebenfalls eindeutig sein.

    :return:
        Neuer QTensor


    Example::

        from pyvqnet import QTensor,tensor
        a = tensor.arange(0,24).reshape((2,3,4))
        b = tensor.moveaxis(a,(1, 2), (0, 1))
        print(b.shape)


swapaxis
==============================

.. py:function:: pyvqnet.tensor.swapaxis(t, axis1: int, axis2: int)

    Vertauscht zwei Achsen eines Arrays. Die gegebenen Dimensionen axis1 und axis2 werden getauscht.

    :param t: Eingabe-QTensor
    :param axis1: Erste Achse.
    :param axis2: Zielposition f�r die urspr�ngliche Achse. Diese muss ebenfalls eindeutig sein.
    :return: Ausgabe-QTensor

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

    Wenn mask == 1, mit dem angegebenen Wert f�llen. Die Form der Maske muss von der Form des Eingabe-QTensor broadcastbar sein.

    :param t: Eingabe-QTensor
    :param mask: Ein QTensor
    :param value: angegebener Wert
    :return: Ein QTensor

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

    Flacht QTensor von Dimension start bis Dimension end ab.

    :param t: Eingabe-QTensor
    :param start: Startdimension, Standard = 0, beginnt mit der ersten Dimension.
    :param end: Enddimension, Standard = -1, endet mit der letzten Dimension.
    :return: Ausgabe-QTensor

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

    �ndert die Form des QTensor, gibt einen QTensor mit neuer Form zur�ck.

    :param t: Eingabe-QTensor.
    :param new_shape: neue Form

    :return: Ein QTensor mit neuer Form.

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
    
    Kehrt den QTensor entlang der angegebenen Achse um und gibt einen neuen Tensor zur�ck.

    :param t: Eingabe-QTensor.
    :param flip_dims: Die Achse oder Liste der Achsen zum Spiegeln.

    :return: Ausgabe-QTensor.

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

    Sammelt Werte entlang der durch 'dim' angegebenen Achse.

    F�r 3-D-Tensoren wird die Ausgabe wie folgt angegeben:

    .. math::

        out[i][j][k] = t[index[i][j][k]][j][k] , if dim == 0 \\

        out[i][j][k] = t[i][index[i][j][k]][k] , if dim == 1 \\

        out[i][j][k] = t[i][j][index[i][j][k]] , if dim == 2 \\

    :param t: Eingabe-QTensor.
    :param dim: Die Aggregationsachse.
    :param index: Index-QTensor, sollte die gleiche Dimensionsgr��e wie die Eingabe haben.

    :return: das aggregierte Ergebnis

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

    Schreibt alle Werte des Tensors src in input an den im Index-Tensor angegebenen Positionen.

    F�r 3-D-Tensoren wird die Ausgabe wie folgt angegeben:

    .. math::

        input[indices[i][j][k]][j][k] = src[i][j][k] , if dim == 0 \\
        input[i][indices[i][j][k]][k] = src[i][j][k] , if dim == 1 \\
        input[i][j][indices[i][j][k]] = src[i][j][k] , if dim == 2 \\

    :param input: Eingabe-QTensor.
    :param dim: Streuungsachse.
    :param indices: Index-QTensor, sollte die gleiche Dimensionsgr��e wie die Eingabe haben.
    :param src: Der Quell-Tensor zum Streuen.

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

    Vorbehaltlich bestimmter Einschr�nkungen wird der Array t auf die Referenzform "gesendet" (broadcast), sodass sie kompatible Formen haben.

    https://numpy.org/doc/stable/user/basics.broadcasting.html

    :param t: Eingabe-QTensor
    :param ref: Referenzform.
    
    :return: Der QTensor des neu gebroadcasteten t.

    Example::

        from pyvqnet.tensor.tensor import QTensor
        from pyvqnet.tensor import *
        ref = [2,3,4]
        a = ones([4])
        b = tensor.broadcast_to(a,ref)
        print(b.shape)
        #[2, 3, 4]



Hilfsfunktionen
*****************************************************


to_tensor
==============================

.. py:function:: pyvqnet.tensor.to_tensor(x)

    Konvertiert das Eingabe-Array in einen QTensor, falls es noch keiner ist.

    :param x: Integer, Float oder numpy.array
    :return: Ausgabe-QTensor

    Example::

        from pyvqnet.tensor import tensor
        t = tensor.to_tensor(10.0)
        print(t)
        # [10]


pad_sequence
==============================

.. py:function:: pyvqnet.tensor.pad_sequence(qtensor_list, batch_first=False, padding_value=0)

    F�llt eine Liste von Tensoren variabler L�nge mit ``padding_value``. ``pad_sequence`` stapelt Listen von Tensoren entlang neuer Dimensionen und f�llt sie auf gleiche L�nge auf.
    Die Eingabe ist eine Sequenz von Listen der Gr��e ``L x *``. L ist die variable L�nge.

    :param qtensor_list: `list[QTensor]` - Liste von Sequenzen variabler L�nge.
    :param batch_first: 'bool' - Wenn True, hat die Ausgabe die Form ``batch size x longest sequence length x *``, andernfalls ``longest sequence length x batch size x *``. Standard: False.
    :param padding_value: 'float' - Auff�llwert. Standardwert: 0.

    :return:
         Wenn batch_first ``False`` ist, hat der Tensor die Gr��e ``batch size x longest sequence length x *``.
         Andernfalls hat der Tensor die Gr��e ``longest sequence length x batch size x *``.

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
    
    F�llt einen Stapel von gepackten Sequenzen variabler L�nge auf. Dies ist die Umkehrung von `pack_pad_sequence`.
    Wenn ``batch_first`` True ist, wird ein Tensor der Form ``B x T x *`` zur�ckgegeben, andernfalls ``T x B x *``.
    Dabei ist `T` die l�ngste Sequenzl�nge und `B` die Stapelgr��e.

    :param sequence: 'QTensor' - die zu verarbeitenden Daten.
    :param batch_first: 'bool' - Wenn ``True``, ist der Stapel die erste Dimension der Eingabe. Standardwert: False.
    :param padding_value: 'bool' - Auff�llwert. Standard: 0.
    :param total_length: 'bool' - Wenn nicht ``None``, wird die Ausgabe auf die L�nge :attr:`total_length` aufgef�llt. Standard: None.
    :return:
        Ein Tupel von Tensoren, das die aufgef�llten Sequenzen enth�lt, sowie eine Liste von L�ngen f�r jede Sequenz im Stapel. Stapelelemente werden in ihrer urspr�nglichen Reihenfolge angeordnet.
    
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
    
    Packt einen Tensor, der aufgefüllte Sequenzen variabler Länge enthält. Wenn batch_first True ist, sollte `input` die Form (Stapelgröße, Länge, ...) haben, andernfalls die Form (Länge, Stapelgröße, ...).

    F�r unsortierte Sequenzen verwenden Sie ``enforce_sorted`` = False. Wenn :attr:`enforce_sorted` ``True`` ist, sollten die Sequenzen in absteigender Reihenfolge nach L�nge sortiert sein.
    
    :param input: 'QTensor' - Stapel von Sequenzen variabler L�nge zum Auff�llen.
    :param lengths: 'list' - Liste der Sequenzl�ngen f�r jedes Stapelelement.
    :param batch_first: 'bool' - Wenn ``True``, wird die Eingabe im Format ``B x T x *`` erwartet, Standard: False.
    :param enforce_sorted: 'bool' - Wenn ``True``, sollte die Eingabe Sequenzen in absteigender Reihenfolge der L�nge enthalten. Wenn ``False``, wird die Eingabe unsortiert verarbeitet. Standard: True.

    :return: Ein :class:`PackedSequence`-Objekt.

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
    
    F�hrt eine 2D-Faltung auf einem Eingabebild durch, das aus mehreren Eingabeebenen besteht.

    :param x: 4D-Eingabe-Tensor.
    :param weight: 4D-Kernel-Tensor.

    :param stride: `tuple` - Schrittweite, Standardwert (1, 1)
    :param padding: Auff�llung, steuert den Umfang der Auff�llung der Eingabe. Dies kann ein String {'valid', 'same'} oder ein Tupel von Integers sein, das den Umfang der impliziten Auff�llung angibt, die auf die Eingabe angewendet wird, Standardwert (0,0).
    :param dilation: `tuple` - Abstand zwischen Kernel-Elementen. Standard: (0,0)
    :param groups: `int` - Anzahl der Gruppen. Standardwert: 1 

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

    Deaktiviert die Aufzeichnung von R�ckw�rtsdurchlauf-Knoten bei der Vorw�rtsberechnung.

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
