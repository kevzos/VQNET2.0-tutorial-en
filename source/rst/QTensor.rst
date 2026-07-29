.. _qtensor_api:

Module QTensor
###########################

L'apprentissage automatique quantique de VQNet utilise la structure de donnees QTensor qui est une interface Python. QTensor prend en charge les operations matricielles multidimensionnelles courantes, notamment les fonctions de creation, les fonctions mathematiques, les fonctions logiques, les transformations matricielles, etc.



Fonctions et attributs de QTensor
******************************************


QTensor
==============================

.. py:class:: pyvqnet.tensor.tensor.QTensor(data, requires_grad=False, nodes=None, device=0, dtype=None, name='')

    Encapsuleur de structure de donnees avec construction de graphe de calcul dynamique
    et differenciation automatique.

    :param data: _core.Tensor ou tableau numpy representant un QTensor
    :param requires_grad: indique si le gradient du tenseur doit etre suivi, par defaut False
    :param nodes: liste des successeurs dans le graphe de calcul, par defaut None
    :param device: peripherique actuel pour sauvegarder le QTensor, par defaut = 0, utilise le CPU.
    :param dtype: Le type de donnees du parametre, par defaut None, utilise le type de donnees par defaut : kfloat32, qui represente un nombre a virgule flottante 32 bits.
    :param name: Le nom du QTensor, par defaut : "".

    :return: QTensor de sortie


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

        Retourne le nombre de dimensions d'un tenseur.

        :return: Le nombre de dimensions d'un tenseur.

        Example::

            from pyvqnet.tensor import QTensor

            a = QTensor([2.0, 3.0, 4.0, 5.0], requires_grad=True)
            print(a.ndim)

            # 1

    .. py:attribute:: shape

        Retourne les dimensions d'un tenseur.

        :return: Une liste des dimensions du tenseur.

        Example::

            from pyvqnet.tensor import QTensor

            a = QTensor([2.0, 3.0, 4.0, 5.0], requires_grad=True)
            print(a.shape)

            # [4]

    .. py:attribute:: size

        Retourne le nombre d'elements d'un tenseur.

        :return: Le nombre d'elements d'un tenseur.

        Example::

            from pyvqnet.tensor import QTensor

            a = QTensor([2.0, 3.0, 4.0, 5.0], requires_grad=True)
            print(a.size)

            # 4

    .. py:method:: numel

        Retourne le nombre d'elements dans un tenseur.

        :return: Le nombre d'elements dans un tenseur.

        Example::

            from pyvqnet.tensor import QTensor

            a = QTensor([2.0, 3.0, 4.0, 5.0], requires_grad=True)
            print(a.numel())

            # 4

    .. py:attribute:: dtype

        Retourne le type de donnees d'un tenseur.

        Les types de donnees pris en charge sont les suivants :

            =========================================  ===============================
            dtype                                      description
            =========================================  ===============================
            ``pyvqnet.kbool``                          Variable booleenne
            ``pyvqnet.kuint8``                         Entier 8 bits (non signe)
            ``pyvqnet.kint8``                          Entier 8 bits (signe)
            ``pyvqnet.kint16``                         Entier 16 bits (signe)
            ``pyvqnet.kint32``                         Entier 32 bits (signe)
            ``pyvqnet.kint64``                         Entier 64 bits (signe)
            ``pyvqnet.kfloat32``                       Virgule flottante 32 bits, voir https://en.wikipedia.org/wiki/IEEE_754
            ``pyvqnet.kfloat64``                       Virgule flottante 64 bits, voir https://en.wikipedia.org/wiki/IEEE_754
            ``pyvqnet.kcomplex64``                     Nombre complexe 64 bits, compose de deux `float32`
            ``pyvqnet.kcomplex128``                    Nombre complexe 128 bits, compose de deux `float64`
            ``pyvqnet.kbfloat16``                      Virgule flottante 16 bits, parfois appelee format Brain floating point, avec une allocation de bits de 1 bit de signe, 8 bits d'exposant et 7 bits de mantisse
            =========================================  ===============================

        :return: Le type de donnees du tenseur.

        Example::

            from pyvqnet.tensor import QTensor

            a = QTensor([2, 3, 4, 5])
            print(a.dtype)
            # 4


    .. py:method:: zero_grad()

        Met le gradient a zero. Sera utilise par l'optimiseur dans le processus d'optimisation.

        :return: None

        Example::

            from pyvqnet.tensor import tensor
            from pyvqnet.tensor import QTensor
            t3  =  QTensor([2.0,3.0,4.0,5.0],requires_grad = True)
            t3.zero_grad()
            print(t3.grad)

            # [0, 0, 0, 0]


 

    .. py:method:: backward(grad=None)

        Calcule le gradient du QTensor actuel.

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

        Copie les donnees dans un nouveau numpy.array.

        :return: un nouveau numpy.array contenant les donnees du QTensor

        .. note::

            numpy ne prend pas en charge le type bfloat16, vous devez d'abord convertir vers d'autres types de donnees pris en charge par numpy tels que float32 avant d'appeler cette interface.

        Example::

            from pyvqnet.tensor import tensor
            from pyvqnet.tensor import QTensor
            t3  =  QTensor([2.0,3.0,4.0,5.0],requires_grad = True)
            t4 = t3.to_numpy()
            print(t4)

            # [2. 3. 4. 5.]

 
    .. py:method:: item()

            Retourne le seul element du QTensor. Leve 'RuntimeError' si le QTensor a plus d'un element.

            :return: uniquement les donnees de cet objet

            Example::

                from pyvqnet.tensor import tensor

                t = tensor.ones([1])
                print(t.item())

                # 1.0

 
    .. py:method:: argmax(*kargs)

        Retourne les indices de la valeur maximale de tous les elements du QTensor d'entree, ou
        Retourne les indices des valeurs maximales d'un QTensor le long d'une dimension.

        :param dim: dim (int) – la dimension a reduire, accepte uniquement un seul axe. si dim == None, retourne les indices de la valeur maximale de tous les elements du tenseur d'entree. La plage valide pour dim est [-R, R), ou R est le ndim de l'entree. quand dim < 0, cela fonctionne de la meme maniere que dim + R.
        :param keepdims: indique si la dimension du QTensor de sortie est conservee ou non.

        :return: les indices de la valeur maximale dans le QTensor d'entree.

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

        Retourne les indices de la valeur minimale de tous les elements du QTensor d'entree, ou
        Retourne les indices des valeurs minimales d'un QTensor le long d'une dimension.

        :param dim: dim (int) – la dimension a reduire, accepte uniquement un seul axe. si dim == None, retourne les indices de la valeur minimale de tous les elements du tenseur d'entree. La plage valide pour dim est [-R, R), ou R est le ndim de l'entree. quand dim < 0, cela fonctionne de la meme maniere que dim + R.
        :param keepdims: indique si la dimension du QTensor de sortie est conservee ou non.

        :return: les indices de la valeur minimale dans le QTensor d'entree.

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

            Remplit le QTensor avec la valeur specifiee sur place.

            :param v: une valeur scalaire
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

            Retourne True si toutes les valeurs du QTensor sont non nulles.

            :return: True si toutes les valeurs du QTensor sont non nulles.

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

            Retourne True si une valeur quelconque du QTensor est non nulle.

            :return: True si une valeur quelconque du QTensor est non nulle.

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

        Remplit un QTensor avec des valeurs echantillonnees aleatoirement selon une distribution binomiale.

        Si les donnees generees aleatoirement selon la distribution binomiale sont superieures au seuil de binarisation, alors le nombre correspondant de positions du QTensor est defini a 1, sinon a 0.

        :param v: Seuil de binarisation
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

        Remplit un QTensor avec des valeurs echantillonnees aleatoirement selon une distribution uniforme signee.

        Facteur d'echelle des valeurs generees par la distribution uniforme signee.

        :param v: une valeur scalaire
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

        Remplit un QTensor avec des valeurs echantillonnees aleatoirement selon une distribution uniforme.

        Facteur d'echelle des valeurs generees par la distribution uniforme.

        :param v: une valeur scalaire
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

        Remplit un QTensor avec des valeurs echantillonnees aleatoirement selon une distribution normale.
        Moyenne de la distribution normale. Ecart type de la distribution normale.
        Indique s'il faut utiliser le mode math rapide ou non.

        :param m: moyenne de la distribution normale
        :param s: ecart type de la distribution normale
        :param fast_math: True si utilisation du mode math rapide
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

        Inverse ou permute les axes d'un tableau. Si new_dims = None, inverse les dimensions.

        :param new_dims: le nouvel ordre des dimensions (liste d'entiers).
        :return: QTensor de resultat.

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

        Change la forme du tenseur et retourne un nouveau QTensor.

        :param new_shape: la nouvelle forme (liste d'entiers)
        :return: un nouveau QTensor

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

        Change la forme du QTensor actuel sur place. Cette interface tentera d'abord de transformer sans modifier les donnees memoire d'origine. En cas d'echec, les donnees actuelles seront copiees dans la nouvelle memoire.

        .. warning::

            Il est recommande d'utiliser l'interface reshape. Dans certains cas, l'emplacement memoire sous-jacent reel sera copie au lieu d'etre modifie sur place.

        :param new_shape: la nouvelle forme (liste d'entiers)
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

            Recupere les donnees du QTensor sous forme de tableau NumPy.

            :return: un tableau NumPy

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

            Le decoupage et l'indexation de QTensor sont pris en charge, ainsi que l'utilisation de QTensor comme index d'acces avance. Un nouveau QTensor sera retourne.

            Les parametres start, stop et step peuvent etre separes par deux-points, comme start:stop:step, ou start, stop et step peuvent etre par defaut.

            En tant que QTensor 1-D, l'indexation ou le decoupage ne peut se faire que sur un seul axe.

            En tant que QTensor 2-D et QTensor multidimensionnel, l'indexation ou le decoupage peut se faire sur plusieurs axes.

            Si vous utilisez QTensor comme index pour l'indexation avancee, voir numpy pour `advanced indexing <https://docs.scipy.org/doc/numpy-1.10.1/reference/arrays.indexing.html>`_ .

            Si votre QTensor utilise comme index est le resultat d'une operation logique, alors vous faites une indexation booleenne.

            .. note:: 
                
                Nous utilisons une forme d'index comme a[3,4,1], mais la forme a[3][4][1] n'est pas prise en charge.

            :param item: Un entier ou un QTensor utilise comme index.

            :return: Un nouveau QTensor.

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

        Le decoupage et l'indexation de QTensor sont pris en charge, ainsi que l'utilisation de QTensor comme index d'acces avance. Un nouveau QTensor sera retourne.

        Les parametres start, stop et step peuvent etre separes par deux-points, comme start:stop:step, ou start, stop et step peuvent etre par defaut.

        En tant que QTensor 1-D, l'indexation ou le decoupage ne peut se faire que sur un seul axe.

        En tant que QTensor 2-D et QTensor multidimensionnel, l'indexation ou le decoupage peut se faire sur plusieurs axes.

        Si vous utilisez QTensor comme index pour l'indexation avancee, voir numpy pour `advanced indexing <https://docs.scipy.org/doc/numpy-1.10.1/reference/arrays.indexing.html>`_ .

        Si votre QTensor utilise comme index est le resultat d'une operation logique, alors vous faites une indexation booleenne.

        .. note:: 
            
            Nous utilisons une forme d'index comme a[3,4,1], mais la forme a[3][4][1] n'est pas prise en charge.

        :param item: Un entier ou un QTensor utilise comme index.

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

        Clone le QTensor vers le peripherique GPU specifie.

        device specifie le peripherique sur lequel les donnees internes sont stockees. Lorsque device >= DEV_GPU_0, les donnees sont stockees sur le GPU.
        Si votre ordinateur dispose de plusieurs GPU, vous pouvez designer differents peripheriques pour stocker les donnees. 
        Par exemple, device = DEV_GPU_1, DEV_GPU_2, DEV_GPU_3, ... indique un stockage sur des GPU avec differents numeros de serie.
        
        .. note::
            QTensor ne peut pas effectuer de calculs sur differents GPU.
            Une erreur Cuda sera levee si vous essayez de creer un QTensor sur un GPU dont l'ID depasse le nombre maximum de GPU verifies.

        :param device: Le peripherique qui sauvegarde actuellement le QTensor, par defaut=DEV_GPU_0,

        device = pyvqnet.DEV_GPU_0, stocke dans le premier GPU, device = DEV_GPU_1,
        stocke dans le deuxieme GPU, et ainsi de suite.

        :return: Clone le QTensor vers le peripherique GPU.

        Examples::

            from pyvqnet.tensor import QTensor
            a = QTensor([2])
            b = a.GPU()
            print(b.device)
            #1000

 

    .. py:method:: CPU()

        Clone le QTensor vers le peripherique CPU specifie.

        :return: Clone le QTensor vers le peripherique CPU.

        Examples::

            from pyvqnet.tensor import QTensor
            a = QTensor([2])
            b = a.CPU()
            print(b.device)
            # 0

 
    .. py:method:: toGPU(device: int = DEV_GPU_0)

        Deplace le QTensor vers le peripherique GPU specifie.

        device specifie le peripherique sur lequel les donnees internes sont stockees. Lorsque device >= DEV_GPU, les donnees sont stockees sur le GPU.
        Si votre ordinateur dispose de plusieurs GPU, vous pouvez designer differents peripheriques pour stocker les donnees.
        Par exemple, device = DEV_GPU_1, DEV_GPU_2, DEV_GPU_3, ... indique un stockage sur des GPU avec differents numeros de serie.

        .. note::

            QTensor ne peut pas effectuer de calculs sur differents GPU. Une erreur Cuda sera levee si vous essayez de creer un QTensor sur un GPU dont l'ID depasse le nombre maximum de GPU verifies.

        :param device: Le peripherique qui sauvegarde actuellement le QTensor, par defaut=DEV_GPU_0. device = pyvqnet.DEV_GPU_0, stocke dans le premier GPU, device = DEV_GPU_1, stocke dans le deuxieme GPU, et ainsi de suite.
        :return: QTensor deplace vers le peripherique GPU.

        Examples::

            from pyvqnet.tensor import QTensor
            a = QTensor([2])
            a = a.toGPU()
            print(a.device)
            #1000


    
    .. py:method:: toCPU()

        Deplace le QTensor vers le CPU.

        :return: QTensor deplace vers le peripherique CPU.

        Examples::

            from pyvqnet.tensor import QTensor
            a = QTensor([2])
            b = a.toCPU()
            print(b.device)
            # 0

    
    .. py:method:: isGPU()

        Indique si les donnees de ce QTensor sont stockees dans la memoire hote du GPU.

        :return: Indique si les donnees de ce QTensor sont stockees dans la memoire hote du GPU.

        Examples::
        
            from pyvqnet.tensor import QTensor
            a = QTensor([2])
            a = a.isGPU()
            print(a)
            # False
 
    .. py:method:: isCPU()

        Indique si les donnees de ce QTensor sont stockees dans la memoire hote du CPU.

        :return: Indique si les donnees de ce QTensor sont stockees dans la memoire hote du CPU.

        Examples::
        
            from pyvqnet.tensor import QTensor
            a = QTensor([2])
            a = a.isCPU()
            print(a)
            # True


Fonctions de creation
*****************************************************


ones
==============================

.. py:function:: pyvqnet.tensor.ones(shape,device=0,dtype-None)

    Retourne un tenseur rempli de 1 avec la forme d'entree.

    :param shape: forme d'entree
    :param device: peripherique de stockage, par defaut 0, CPU.
    :param dtype: Le type de donnees du parametre, par defaut None, utilise le type de donnees par defaut : kfloat32, qui represente un nombre a virgule flottante 32 bits.
    
    :return: QTensor de sortie avec la forme d'entree.

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

    Retourne un tenseur rempli de 1 avec la meme forme que le QTensor d'entree.

    :param t: QTensor d'entree
    :param device: peripherique de stockage, par defaut 0, CPU.
    :param dtype: Le type de donnees du parametre, par defaut None, utilise le type de donnees par defaut : kfloat32, qui represente un nombre a virgule flottante 32 bits.
    
    :return: QTensor de sortie


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

    Cree un QTensor de la forme specifiee et le remplit avec la valeur.

    :param shape: forme du QTensor a creer
    :param value: valeur pour remplir le QTensor
    :param device: peripherique a utiliser, par defaut = 0, utilise le peripherique CPU.
    :param dtype: Le type de donnees du parametre, par defaut None, utilise le type de donnees par defaut : kfloat32, qui represente un nombre a virgule flottante 32 bits.
    
    :return: QTensor de sortie

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

    Cree un QTensor de la forme specifiee et le remplit avec la valeur.

    :param t: QTensor d'entree
    :param value: valeur pour remplir le QTensor
    :param device: peripherique a utiliser, par defaut = 0, utilise le peripherique CPU.
    :param dtype: Le type de donnees du parametre, par defaut None, utilise le type de donnees par defaut : kfloat32, qui represente un nombre a virgule flottante 32 bits.
    
    :return: QTensor de sortie

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

    Retourne un tenseur rempli de 0 de la forme d'entree.

    :param shape: forme du tenseur
    :param device: peripherique a utiliser, par defaut = 0, utilise le peripherique CPU.
    :param dtype: Le type de donnees du parametre, par defaut None, utilise le type de donnees par defaut : kfloat32, qui represente un nombre a virgule flottante 32 bits.
    
    :return: QTensor de sortie

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

    Retourne un tenseur rempli de 0 avec la meme forme que le QTensor d'entree.

    :param t: QTensor d'entree
    :param device: peripherique a utiliser, par defaut = 0, utilise le peripherique CPU.
    :param dtype: Le type de donnees du parametre, par defaut None, utilise le type de donnees par defaut : kfloat32, qui represente un nombre a virgule flottante 32 bits.
    
    :return: QTensor de sortie

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

    Cree un QTensor 1D avec des valeurs espaces regulierement dans un intervalle donne.

    :param start: debut de l'intervalle
    :param end: fin de l'intervalle
    :param step: espacement entre les valeurs
    :param device: peripherique a utiliser, par defaut = 0, utilise le peripherique CPU.
    :param dtype: Le type de donnees du parametre, par defaut None, utilise le type de donnees par defaut : kfloat32, qui represente un nombre a virgule flottante 32 bits.
    :param requires_grad: indique si le gradient du tenseur doit etre suivi, par defaut False
    :return: QTensor de sortie

    Example::

        from pyvqnet.tensor import tensor
        from pyvqnet.tensor import QTensor
        t = tensor.arange(2, 30,4)
        print(t)

        # [ 2,  6, 10, 14, 18, 22, 26]

linspace
==============================

.. py:function:: pyvqnet.tensor.linspace(start, end, num, device: int = 0,dtype=None, requires_grad= False)

    Cree un QTensor 1D avec des valeurs espaces regulierement dans un intervalle donne.

    :param start: valeur de debut
    :param end: valeur de fin
    :param nums: nombre d'echantillons a generer
    :param device: peripherique a utiliser, par defaut = 0, utilise le peripherique CPU.
    :param dtype: Le type de donnees du parametre, par defaut None, utilise le type de donnees par defaut : kfloat32, qui represente un nombre a virgule flottante 32 bits.
    :param requires_grad: indique si le gradient du tenseur doit etre suivi, par defaut False
    :return: QTensor de sortie

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

    Cree un QTensor 1D avec des valeurs espaces regulierement sur une echelle logarithmique.

    :param start: ``base ** start`` est la valeur de debut
    :param end: ``base ** end`` est la valeur finale de la sequence
    :param nums: nombre d'echantillons a generer
    :param base: la base de l'espace logarithmique
    :param device: peripherique a utiliser, par defaut = 0, utilise le peripherique CPU.
    :param dtype: Le type de donnees du parametre, par defaut None, utilise le type de donnees par defaut : kfloat32, qui represente un nombre a virgule flottante 32 bits.
    :param requires_grad: indique si le gradient du tenseur doit etre suivi, par defaut False
    :return: QTensor de sortie

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

    Cree un QTensor de taille size x size avec des 1 sur la diagonale et des 0
    ailleurs.

    :param size: taille du QTensor (carre) a creer
    :param offset: Index de la diagonale : 0 (par defaut) fait reference a la diagonale principale, une valeur positive fait reference a une diagonale superieure et une valeur negative a une diagonale inferieure.
    :param device: peripherique a utiliser, par defaut = 0, utilise le peripherique CPU.
    :param dtype: Le type de donnees du parametre, par defaut None, utilise le type de donnees par defaut : kfloat32, qui represente un nombre a virgule flottante 32 bits.
    
    :return: QTensor de sortie

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


    Retourne une vue partielle de :attr:`t` avec les elements diagonaux ajoutes comme dimensions a la fin de la forme relative a :attr:`dim1` et :attr:`dim2`.
    :attr:`offset` est le decalage de la diagonale principale.

    :param t: tenseur d'entree
    :param offset: decalage (0 signifie diagonale principale, les valeurs positives signifient la nieme diagonale au-dessus de la diagonale principale, les valeurs negatives signifient la nieme diagonale en dessous de la diagonale principale)
    :param dim1: premiere dimension pour prendre la diagonale. Par defaut : 0.
    :param dim2: deuxieme dimension pour prendre la diagonale. Par defaut : 1.

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

    Selectionne les elements diagonaux ou construit un QTensor diagonal.

    Entrez un QTensor 2-D et retourne un nouveau tenseur 1D contenant les elements diagonaux selectionnes. Entrez un QTensor 1-D et retourne un nouveau tenseur 2D dont les elements diagonaux selectionnes sont les valeurs d'entree et le reste est 0.

    :param t: QTensor d'entree
    :param k: decalage (0 pour la diagonale principale, positif pour la nieme diagonale au-dessus de la diagonale principale, negatif pour la nieme diagonale en dessous de la diagonale principale)
    :return: QTensor de sortie

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

    Cree un QTensor avec des valeurs aleatoires uniformement distribuees.

    :param shape: forme du QTensor a creer
    :param min: valeur minimale de la distribution uniforme, par defaut : 0.
    :param max: valeur maximale de la distribution uniforme, par defaut : 1.
    :param device: peripherique a utiliser, par defaut = 0, utilise le peripherique CPU.
    :param dtype: Le type de donnees du parametre, par defaut None, utilise le type de donnees par defaut : kfloat32, qui represente un nombre a virgule flottante 32 bits.
    :param requires_grad: indique si le gradient du tenseur doit etre suivi, par defaut False
    :return: QTensor de sortie


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

    Cree un QTensor avec des valeurs aleatoires normalement distribuees.

    :param shape: forme du QTensor a creer
    :param mean: valeur moyenne de la distribution normale, par defaut : 0.
    :param std: valeur de l'ecart type de la distribution normale, par defaut : 1.
    :param device: peripherique a utiliser, par defaut = 0, utilise le peripherique CPU.
    :param dtype: Le type de donnees du parametre, par defaut None, utilise le type de donnees par defaut : kfloat32, qui represente un nombre a virgule flottante 32 bits.
    :param requires_grad: indique si le gradient du tenseur doit etre suivi, par defaut False
    :return: QTensor de sortie

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
    
    Cree une distribution binomiale parametree par :attr:total_count et :attr:probs.

    :param total_counts: Nombre d'essais de Bernoulli.
    :param probs: Probabilites des evenements.

    :return:
        QTensor pour la distribution binomiale.

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

    Retourne un Tenseur dont chaque ligne contient num_samples echantillons indexes.
    A partir de la distribution de probabilite multinomiale situee dans la ligne correspondante du tenseur d'entree.

    :param t: Distribution de probabilite d'entree.
    :param num_samples: nombre d'echantillons.

    :return:
        index des echantillons de sortie

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

    Retourne la matrice triangulaire superieure de l'entree t, le reste etant defini a 0.

    :param t: entrez un QTensor
    :param diagonal: Le decalage par defaut = 0. La diagonale principale est 0, positif est un decalage vers le haut, et negatif est un decalage vers le bas.

    :return: sortie d'un QTensor

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

    Retourne la matrice triangulaire inferieure de l'entree t, le reste etant defini a 0.

    :param t: entrez un QTensor
    :param diagonal: Le decalage par defaut = 0. La diagonale principale est 0, positif est un decalage vers le haut, et negatif est un decalage vers le bas.

    :return: sortie d'un QTensor

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


Fonctions mathematiques
*****************************************************


floor
==============================

.. py:function:: pyvqnet.tensor.floor(t)

    Retourne un nouveau QTensor avec la partie entiere par defaut des elements d'entree, le plus grand entier inferieur ou egal a chaque element.

    :param t: QTensor d'entree
    :return: QTensor de sortie

    Example::

        from pyvqnet.tensor import tensor

        t = tensor.arange(-2.0, 2.0, 0.25)
        u = tensor.floor(t)
        print(u)

        # [-2, -2, -2, -2, -1, -1, -1, -1, 0, 0, 0, 0, 1, 1, 1, 1]

ceil
==============================

.. py:function:: pyvqnet.tensor.ceil(t)

    Retourne un nouveau QTensor avec la partie entiere par exces des elements d'entree, le plus petit entier superieur ou egal a chaque element.

    :param t: QTensor d'entree
    :return: QTensor de sortie

    Example::

        from pyvqnet.tensor import tensor

        t = tensor.arange(-2.0, 2.0, 0.25)
        u = tensor.ceil(t)
        print(u)

        # [-2, -1, -1, -1, -1, -0, -0, -0, 0, 1, 1, 1, 1, 2, 2, 2]

round
==============================

.. py:function:: pyvqnet.tensor.round(t)

    Arrondit les valeurs du QTensor a l'entier le plus proche.

    :param t: QTensor d'entree
    :return: QTensor de sortie

    Example::

        from pyvqnet.tensor import tensor

        t = tensor.arange(-2.0, 2.0, 0.4)
        u = tensor.round(t)
        print(u)

        # [-2, -2, -1, -1, -0, -0, 0, 1, 1, 2]

sort
==============================

.. py:function:: pyvqnet.tensor.sort(t, axis: int, descending=False, stable=True)

    Trie le QTensor le long de l'axe.

    :param t: QTensor d'entree
    :param axis: axe de tri
    :param descending: ordre de tri descendant
    :param stable: Indique s'il faut utiliser un tri stable ou non
    :return: QTensor de sortie

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

    Retourne un tableau d'indices de la meme forme que l'entree qui indexe les donnees le long de l'axe donne dans l'ordre trie.

    :param t: QTensor d'entree
    :param axis: axe de tri
    :param descending: ordre de tri descendant
    :param stable: Indique s'il faut utiliser un tri stable ou non
    :return: QTensor de sortie

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

    Retourne les k plus grands elements du tenseur d'entree le long de l'axe donne.

    Si if_descent est False, retourne les k plus petits elements.

    :param t: entrez un QTensor
    :param k: nombre des plus grands elements ou des plus petits elements
    :param axis: axe de tri, par defaut = -1, le dernier axe
    :param if_descent: ordre de tri, par defaut True

    :return: Un nouveau QTensor

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

    Retourne l'index des k plus grands elements le long de l'axe donne du tenseur d'entree.

    Si if_descent est False, retourne l'index des k plus petits elements.

    :param t: entrez un QTensor
    :param k: nombre des plus grands elements ou des plus petits elements
    :param axis: axe de tri, par defaut = -1, le dernier axe
    :param if_descent: ordre de tri, par defaut True

    :return: Un nouveau QTensor

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

    Additionne element par element deux QTensors, equivalent a t1 + t2.

    :param t1: premier QTensor
    :param t2: deuxieme QTensor
    :return: QTensor de sortie

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

    Soustrait element par element deux QTensors, equivalent a t1 - t2.


    :param t1: premier QTensor
    :param t2: deuxieme QTensor
    :return: QTensor de sortie

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

    Multiplie element par element deux QTensors, equivalent a t1 * t2.

    :param t1: premier QTensor
    :param t2: deuxieme QTensor
    :return: QTensor de sortie


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

    Divise element par element deux QTensors, equivalent a t1 / t2.


    :param t1: premier QTensor
    :param t2: deuxieme QTensor
    :return: QTensor de sortie


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

    Additionne tous les elements du QTensor le long de l'axe donne. Si axis = None, additionne tous les elements du QTensor.

    :param t: QTensor d'entree
    :param axis: axe utilise pour la somme, par defaut None
    :param keepdims: indique si la dimension du tenseur de sortie est conservee ou non. - par defaut False
    :return: QTensor de sortie


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

    Retourne la somme cumulee des elements d'entree dans la dimension de l'axe.

    :param t: le QTensor d'entree
    :param axis: Calcul de l'axe, par defaut -1, utilise le dernier axe

    :return: QTensor de sortie

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

    Obtient les valeurs moyennes dans le QTensor le long de l'axe.

    :param t: le QTensor d'entree
    :param axis: la dimension a reduire
    :param keepdims: indique si la dimension du QTensor de sortie est conservee ou non, par defaut False.
    :return: retourne la valeur moyenne du QTensor d'entree.

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

    Obtient la valeur mediane dans le QTensor.

    :param t: le QTensor d'entree
    :param axis: Un axe pour le calcul de la moyenne, par defaut None
    :param keepdims: indique si la dimension du QTensor de sortie est conservee ou non, par defaut False

    :return: Retourne la mediane des valeurs dans l'entree ou le QTensor.

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

    Obtient la valeur de l'ecart type dans le QTensor.


    :param t: le QTensor d'entree
    :param axis: l'axe utilise pour calculer l'ecart type, par defaut None
    :param keepdims: indique si la dimension du QTensor de sortie est conservee ou non, par defaut False
    :param unbiased: indique s'il faut utiliser la correction de Bessel, par defaut true
    :return: Retourne l'ecart type des valeurs dans l'entree ou le QTensor

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

    Obtient la variance dans le QTensor.


    :param t: le QTensor d'entree
    :param axis: L'axe utilise pour calculer la variance, par defaut None
    :param keepdims: indique si la dimension du QTensor de sortie est conservee ou non, par defaut False.
    :param unbiased: indique s'il faut utiliser la correction de Bessel, par defaut true.


    :return: Obtient la variance dans le QTensor.

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

    Multiplication matricielle de deux matrices 2d, 3d, 4d.

    :param t1: premier QTensor
    :param t2: deuxieme QTensor
    :return: QTensor de sortie

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

    Calcule le produit de Kronecker de ``t1`` et ``t2``, exprime par :math:`\otimes` . Si ``t1`` est un tenseur :math:`(a_0 \times a_1 \times \dots \times a_n)` et ``t2`` est un tenseur :math:`(b_0 \times b_1 \times \dots \ times b_n)`, le resultat sera un tenseur :math:`(a_0*b_0 \times a_1*b_1 \times \dots \times a_n*b_n)` avec les entrees suivantes :
    
    .. math::
          (\text{input} \otimes \text{other})_{k_0, k_1, \dots, k_n} =
              \text{input}_{i_0, i_1, \dots, i_n} * \text{other}_{j_0, j_1, \dots, j_n},

    ou :math:`k_t = i_t * b_t + j_t` est :math:`0 \leq t \leq n`.
    Si un tenseur a moins de dimensions que l'autre, il sera developpe jusqu'a avoir la meme dimensionalite.

    :param t1: Le premier QTensor.
    :param t2: Le deuxieme QTensor.
    
    :return: QTensor de sortie.

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
    
    Additionne les produits des elements des operandes d'entree le long de la dimension specifiee en utilisant une notation basee sur la convention de sommation d'Einstein.

    .. note::

        Cette fonction utilise opt_einsum (https://optimized-einsum.readthedocs.io/en/stable/) pour accelerer le calcul ou reduire la consommation memoire en optimisant l'ordre de contraction. Cette optimisation se produit lorsqu'il y a au moins trois entrees.

        Pour des `einsum` plus complexes, opt_einsum peut etre importe supplementairement pour calculer directement sur le QTensor.

    :param equation: L'indice de la sommation d'Einstein.
    :param operands: Le tenseur sur lequel la sommation d'Einstein doit etre calculee.

    :return:

        Le resultat QTensor.

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

    Calcule l'inverse element par element du QTensor.

    :param t: QTensor d'entree
    :return: QTensor de sortie

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

    Retourne un nouveau QTensor avec les signes des elements d'entree. La fonction signe retourne -1 si t < 0, 0 si t==0, 1 si t > 0.

    :param t: QTensor d'entree
    :return: QTensor de sortie


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

    Negation unaire des elements du QTensor.

    :param t: QTensor d'entree
    :return: QTensor de sortie

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

    Retourne la somme des elements de la diagonale de la matrice 2-D d'entree.

    :param t: QTensor 2-D d'entree
    :param k: decalage (0 pour la diagonale principale, positif pour la nieme diagonale au-dessus de la diagonale principale, negatif pour la nieme diagonale en dessous de la diagonale principale)
    :return: la somme des elements de la diagonale de la matrice 2-D d'entree

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

    Applique la fonction exponentielle a tous les elements du QTensor d'entree.

    :param t: QTensor d'entree
    :return: QTensor de sortie

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

    Calcule l'arc cosinus element par element du QTensor.

    :param t: QTensor d'entree
    :return: QTensor de sortie

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

    Calcule l'arc sinus element par element du QTensor.

    :param t: QTensor d'entree
    :return: QTensor de sortie

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

    Calcule l'arc tangente element par element du QTensor.

    :param t: QTensor d'entree
    :return: QTensor de sortie

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

    Applique la fonction sinus a tous les elements du QTensor d'entree.


    :param t: QTensor d'entree
    :return: QTensor de sortie

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

    Applique la fonction cosinus a tous les elements du QTensor d'entree.


    :param t: QTensor d'entree
    :return: QTensor de sortie

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

    Applique la fonction tangente a tous les elements du QTensor d'entree.


    :param t: QTensor d'entree
    :return: QTensor de sortie

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

    Applique la fonction tangente hyperbolique a tous les elements du QTensor d'entree.

    :param t: QTensor d'entree
    :return: QTensor de sortie

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

    Applique la fonction sinus hyperbolique a tous les elements du QTensor d'entree.


    :param t: QTensor d'entree
    :return: QTensor de sortie

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

    Applique la fonction cosinus hyperbolique a tous les elements du QTensor d'entree.


    :param t: QTensor d'entree
    :return: QTensor de sortie

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

    Eleve le premier QTensor a la puissance du deuxieme QTensor.

    :param t1: premier QTensor
    :param t2: deuxieme QTensor
    :return: QTensor de sortie

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

    Applique la fonction valeur absolue a tous les elements du QTensor d'entree.

    :param t: QTensor d'entree
    :return: QTensor de sortie

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

    Applique la fonction log (ln) a tous les elements du QTensor d'entree.

    :param t: QTensor d'entree
    :return: QTensor de sortie

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
    
    Calcule sequentiellement les resultats de la fonction softmax et de la fonction log sur l'axe axis.

    :param t: QTensor d'entree.
    :param axis: L'axe utilise pour calculer softmax, par defaut -1.

    :return: QTensor de sortie.

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

    Applique la fonction racine carree a tous les elements du QTensor d'entree.


    :param t: QTensor d'entree
    :return: QTensor de sortie

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

    Applique la fonction carre a tous les elements du QTensor d'entree.


    :param t: QTensor d'entree
    :return: QTensor de sortie

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
 
    Retourne les valeurs propres et vecteurs propres d'une matrice complexe hermitienne (symetrique conjuguee) ou reelle symetrique.

    Retourne deux objets, un tableau 1D contenant les valeurs propres de a,
    et une matrice carree 2D ou matrice (selon le type d'entree) des vecteurs propres correspondants (en colonnes).

    :param: QTensor d'entree.
    :param: Valeurs propres et vecteurs propres de t.
    :return:

        Retourne les valeurs propres et vecteurs propres

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

    Calcule la norme F du tenseur sur le QTensor d'entree le long de l'axe defini par axis,
    si axis est None, retourne la norme F de tous les elements.

    :param t: QTensor d'entree.
    :param axis: L'axe utilise pour trouver la norme F, par defaut None.
    :param keepdims: Indique si le tenseur de sortie preserve la dimensionnalite reduite. Par defaut False.
    :return: Sort un QTensor ou une valeur de norme F.


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




Fonctions logiques
**************************

maximum
==============================

.. py:function:: pyvqnet.tensor.maximum(t1: pyvqnet.tensor.QTensor, t2: pyvqnet.tensor.QTensor)

    Maximum element par element de deux tenseurs.


    :param t1: premier QTensor
    :param t2: deuxieme QTensor
    :return: QTensor de sortie

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

    Minimum element par element de deux tenseurs.


    :param t1: premier QTensor
    :param t2: deuxieme QTensor
    :return: QTensor de sortie

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

    Retourne les elements minimaux du QTensor d'entree le long de l'axe donne.
    Si axis == None, retourne la valeur minimale de tous les elements du tenseur.

    :param t: QTensor d'entree
    :param axis: axe utilise pour le min, par defaut None
    :param keepdims: indique si la dimension du tenseur de sortie est conservee ou non. - par defaut False
    :return: QTensor de sortie

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

    Retourne les elements maximaux du QTensor d'entree le long de l'axe donne.
    Si axis == None, retourne la valeur maximale de tous les elements du tenseur.

    :param t: QTensor d'entree
    :param axis: axe utilise pour le max, par defaut None
    :param keepdims: indique si la dimension du tenseur de sortie est conservee ou non. - par defaut False
    :return: QTensor de sortie

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

    Limite le QTensor d'entree aux valeurs minimale et maximale.

    :param t: QTensor d'entree
    :param min_val: valeur minimale
    :param max_val: valeur maximale
    :return: QTensor de sortie

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

    Retourne les elements choisis parmi x ou y en fonction de la condition.

    :param condition: tenseur de condition, doit avoir le type de donnees kbool.
    :param t1: QTensor a partir duquel prendre les elements si la condition est vraie, par defaut None
    :param t2: QTensor a partir duquel prendre les elements si la condition est fausse, par defaut None
    :return: QTensor de sortie

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

    Retourne un QTensor contenant les indices des elements non nuls.

    :param t: QTensor d'entree
    :return: QTensor de sortie contenant les indices des elements non nuls.

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

    Teste element par element la finitude (pas infini et pas Not a Number).

    :param t: QTensor d'entree
    :return: QTensor de sortie, qui retourne True lorsque l'element a la position correspondante satisfait la condition, sinon retourne False.

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

    Teste element par element l'infini positif ou negatif.

    :param t: QTensor d'entree
    :return: QTensor de sortie, qui retourne True lorsque l'element a la position correspondante satisfait la condition, sinon retourne False.
    
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

    Teste element par element pour Nan.

    :param t: QTensor d'entree
    :return: QTensor de sortie, qui retourne True lorsque l'element a la position correspondante satisfait la condition, sinon retourne False.
    
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

    Teste element par element l'infini negatif.

    :param t: QTensor d'entree
    :return: QTensor de sortie, qui retourne True lorsque l'element a la position correspondante satisfait la condition, sinon retourne False.
    
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

    Teste element par element l'infini positif.

    :param t: QTensor d'entree
    :return: QTensor de sortie, qui retourne True lorsque l'element a la position correspondante satisfait la condition, sinon retourne False.
    
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

    Calcule la valeur de verite de ``t1 et t2`` element par element.

    :param t1: QTensor d'entree
    :param t2: QTensor d'entree
    :return: QTensor de sortie, qui retourne True lorsque l'element a la position correspondante satisfait la condition, sinon retourne False.
    
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

    Calcule la valeur de verite de ``t1 ou t2`` element par element.

    :param t1: QTensor d'entree
    :param t2: QTensor d'entree
    :return: QTensor de sortie, qui retourne True lorsque l'element a la position correspondante satisfait la condition, sinon retourne False.

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

    Calcule la valeur de verite de ``non t`` element par element.

    :param t: QTensor d'entree
    :return: QTensor de sortie, qui retourne True lorsque l'element a la position correspondante satisfait la condition, sinon retourne False.

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

    Calcule la valeur de verite de ``t1 xor t2`` element par element.

    :param t1: QTensor d'entree
    :param t2: QTensor d'entree

    :return: QTensor de sortie, qui retourne True lorsque l'element a la position correspondante satisfait la condition, sinon retourne False.

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

    Retourne la valeur de verite de ``t1 > t2`` element par element.


    :param t1: QTensor d'entree
    :param t2: QTensor d'entree
    :return: QTensor de sortie, qui retourne True lorsque l'element a la position correspondante satisfait la condition, sinon retourne False.

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

    Retourne la valeur de verite de ``t1 >= t2`` element par element.

    :param t1: QTensor d'entree
    :param t2: QTensor d'entree
    :return: QTensor de sortie, qui retourne True lorsque l'element a la position correspondante satisfait la condition, sinon retourne False.

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

    Retourne la valeur de verite de ``t1 < t2`` element par element.

    :param t1: QTensor d'entree
    :param t2: QTensor d'entree
    :return: QTensor de sortie, qui retourne True lorsque l'element a la position correspondante satisfait la condition, sinon retourne False.

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

    Retourne la valeur de verite de ``t1 <= t2`` element par element.

    :param t1: QTensor d'entree
    :param t2: QTensor d'entree
    :return: QTensor de sortie, qui retourne True lorsque l'element a la position correspondante satisfait la condition, sinon retourne False.

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

    Retourne la valeur de verite de ``t1 == t2`` element par element.

    :param t1: QTensor d'entree
    :param t2: QTensor d'entree
    :return: QTensor de sortie, qui retourne True lorsque l'element a la position correspondante satisfait la condition, sinon retourne False.
    
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

    Retourne la valeur de verite de ``t1 != t2`` element par element.

    :param t1: QTensor d'entree
    :param t2: QTensor d'entree
    :return: QTensor de sortie, qui retourne True lorsque l'element a la position correspondante satisfait la condition, sinon retourne False.
    
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
 
    Calcule le ET bit a bit de deux elements QTensor.

    :param t1: QTensor d'entree t1. Seuls les entiers ou les booleens sont des entrees valides.
    :param t2: QTensor d'entree t2. Seuls les entiers ou les booleens sont des entrees valides.
    :return:
        QTensor de resultat

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


Operations matricielles
***********************

select
==============================

.. py:function:: pyvqnet.tensor.select(t: pyvqnet.tensor.QTensor, index)

    Retourne le QTensor dans le QTensor a l'axe donne. L'operation suivante obtient la valeur du meme resultat.

    :param t: QTensor d'entree
    :param index: une chaine contenant la dimension de sortie
    :return: QTensor de sortie

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

    Sous certaines restrictions, les tableaux plus petits sont places dans des tableaux plus grands afin qu'ils aient des formes compatibles. Cette interface peut effectuer une differenciation automatique sur les tenseurs de parametres d'entree.

    Reference https://numpy.org/doc/stable/user/basics.broadcasting.html

    :param t1: QTensor d'entree 1
    :param t2: QTensor d'entree 2

    :return t11: t1 avec la nouvelle forme de diffusion.
    :return t22: t2 avec la nouvelle forme de diffusion.

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

    Concatene le QTensor d'entree le long de l'axe et retourne un nouveau QTensor.

    :param args: liste composee de QTensors d'entree
    :param axis: dimension a concatener. Doit etre entre 0 et le nombre de dimensions des tenseurs a concatener.
    :return: QTensor de sortie

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

    Rejoint une sequence de tableaux le long d'un nouvel axe, retourne un nouveau QTensor.

    :param QTensors: liste contenant des QTensors
    :param axis: dimension a inserer. Doit etre entre 0 et le nombre de dimensions des tenseurs a empiler.
    :return: QTensor de sortie

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

    Inverse ou permute les axes d'un tableau.

    :param t: QTensor d'entree
    :param dim: le nouvel ordre des dimensions (liste d'entiers)
    :return: QTensor de sortie

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

    Transpose les axes d'un tableau. Si dim = None, inverse les dimensions. Cette fonction est identique a permute.

    :param t: QTensor d'entree
    :param dim: le nouvel ordre des dimensions (liste d'entiers)
    :return: QTensor de sortie

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

    Construit un QTensor en repetant le QTensor le nombre de fois donne par reps.

    Si reps a une longueur d, le QTensor resultat aura une dimension de max(d, t.ndim).

    Si t.ndim < d, t est etendu pour etre d-dimensionnel en inserant de nouveaux axes a partir de la dimension de debut.
    Ainsi un tableau de forme (3,) est promu en (1, 3) pour une replication 2-D, ou en forme (1, 1, 3) pour une replication 3-D.

    Si t.ndim > d, reps est etendu a t.ndim en y inserant des 1.

    Ainsi pour un t de forme (2, 3, 4, 5), un reps de (4, 3) est traite comme (1, 1, 4, 3).

    :param t: QTensor d'entree
    :param reps: le nombre de repetitions par dimension.
    :return: un nouveau QTensor

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

    Supprime les axes de longueur un.

    :param t: QTensor d'entree
    :param axis: axe de compression, si axis = -1, supprime toutes les dimensions qui ont une taille de 1.
    :return: QTensor de sortie

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

    Retourne un nouveau QTensor avec une dimension de taille un inseree a la position specifiee.

    :param t: QTensor d'entree
    :param axis: axe de decompression, qui inserera la dimension.
    :return: QTensor de sortie

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

    Deplace les dimensions de `t` des positions dans `source` vers les positions dans `destination`.

    Les autres dimensions de `t` qui ne sont pas explicitement deplacees conservent leur ordre d'origine et apparaissent aux positions non specifiees dans `destination`.

    :param t: QTensor d'entree.
    :param source: (entier ou tuple d'entiers) Les positions d'origine des dimensions a deplacer. Ces positions doivent etre uniques.
    :param destination: (entier ou tuple d'entiers) Les positions de destination pour chaque dimension d'origine. Ces positions doivent egalement etre uniques.

    :return:
        Nouveau QTensor


    Example::

        from pyvqnet import QTensor,tensor
        a = tensor.arange(0,24).reshape((2,3,4))
        b = tensor.moveaxis(a,(1, 2), (0, 1))
        print(b.shape)


swapaxis
==============================

.. py:function:: pyvqnet.tensor.swapaxis(t, axis1: int, axis2: int)

    Echange deux axes d'un tableau. Les dimensions donnees axis1 et axis2 sont echangees.

    :param t: QTensor d'entree
    :param axis1: Premier axe.
    :param axis2: Position de destination pour l'axe d'origine. Ceux-ci doivent egalement etre uniques.
    :return: QTensor de sortie

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

    Si mask == 1, remplit avec la valeur specifiee. La forme du masque doit pouvoir etre diffusee a partir de la forme du QTensor d'entree.

    :param t: QTensor d'entree
    :param mask: Un QTensor
    :param value: valeur specifiee
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

    Aplatit le QTensor de la dimension start a la dimension end.

    :param t: QTensor d'entree
    :param start: dimension de debut, par defaut = 0, commence a la premiere dimension.
    :param end: dimension de fin, par defaut = -1, se termine a la derniere dimension.
    :return: QTensor de sortie

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

    Change la forme du QTensor, retourne un QTensor de nouvelle forme.

    :param t: QTensor d'entree.
    :param new_shape: nouvelle forme

    :return: un QTensor de nouvelle forme.

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
    
    Inverse le QTensor le long de l'axe specifie, retournant un nouveau tenseur.

    :param t: QTensor d'entree.
    :param flip_dims: L'axe ou la liste des axes a inverser.

    :return: QTensor de sortie.

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

    Collecte les valeurs le long de l'axe specifie par 'dim'.

    Pour les tenseurs 3-D, la sortie est specifiee par :

    .. math::

        out[i][j][k] = t[index[i][j][k]][j][k] , if dim == 0 \\

        out[i][j][k] = t[i][index[i][j][k]][k] , if dim == 1 \\

        out[i][j][k] = t[i][j][index[i][j][k]] , if dim == 2 \\

    :param t: QTensor d'entree.
    :param dim: L'axe d'agregation.
    :param index: QTensor d'index, doit avoir la meme taille de dimension que l'entree.

    :return: le resultat agrege

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

    Ecrit toutes les valeurs du tenseur src dans l'entree aux indices specifies dans le tenseur d'indices.

    Pour les tenseurs 3-D, la sortie est specifiee par :

    .. math::

        input[indices[i][j][k]][j][k] = src[i][j][k] , if dim == 0 \\
        input[i][indices[i][j][k]][k] = src[i][j][k] , if dim == 1 \\
        input[i][j][indices[i][j][k]] = src[i][j][k] , if dim == 2 \\

    :param input: QTensor d'entree.
    :param dim: Axe de dispersion.
    :param indices: QTensor d'index, doit avoir la meme taille de dimension que l'entree.
    :param src: Le tenseur source a disperser.

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

    Sous certaines contraintes, le tableau t est \"diffuse\" vers la forme de reference afin qu'ils aient des formes compatibles.

    https://numpy.org/doc/stable/user/basics.broadcasting.html

    :param t: QTensor d'entree
    :param ref: Forme de reference.
    
    :return: Le QTensor de t nouvellement diffuse.

    Example::

        from pyvqnet.tensor.tensor import QTensor
        from pyvqnet.tensor import *
        ref = [2,3,4]
        a = ones([4])
        b = tensor.broadcast_to(a,ref)
        print(b.shape)
        #[2, 3, 4]



Fonctions utilitaires
*****************************************************


to_tensor
==============================

.. py:function:: pyvqnet.tensor.to_tensor(x)

    Convertit le tableau d'entree en Qtensor s'il ne l'est pas deja.

    :param x: entier, flottant ou numpy.array
    :return: QTensor de sortie

    Example::

        from pyvqnet.tensor import tensor
        t = tensor.to_tensor(10.0)
        print(t)
        # [10]



pad_sequence
==============================

.. py:function:: pyvqnet.tensor.pad_sequence(qtensor_list, batch_first=False, padding_value=0)

    Remplit une liste de tenseurs de longueurs variables avec ``padding_value``. ``pad_sequence`` empile les listes de tenseurs le long de nouvelles dimensions et les remplit pour qu'elles aient une longueur egale.
    L'entree est une sequence de listes de taille ``L x *``. L est de longueur variable.

    :param qtensor_list: `list[QTensor]` - liste de sequences de longueurs variables.
    :param batch_first: 'bool' - Si vrai, la sortie sera ``taille du lot x longueur de la plus longue sequence x *``, sinon ``longueur de la plus longue sequence x taille du lot x *``. Par defaut : False.
    :param padding_value: 'float' - valeur de remplissage. Valeur par defaut : 0.

    :return:
         Si batch_first est ``False``, la taille du tenseur est ``taille du lot x longueur de la plus longue sequence x *``.
         Sinon, la taille du tenseur est ``longueur de la plus longue sequence x taille du lot x *``.

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
    
    Remplit un lot de sequences compressees de longueurs variables. C'est l'inverse de `pack_pad_sequence`.
    Quand ``batch_first`` est True, retourne un tenseur de forme ``B x T x *``, sinon retourne ``T x B x *``.
    Ou `T` est la longueur de la plus longue sequence et `B` est la taille du lot.

    :param sequence: 'QTensor' - les donnees a traiter.
    :param batch_first: 'bool' - Si ``True``, le lot sera la premiere dimension de l'entree. Valeur par defaut : False.
    :param padding_value: 'bool' - valeur de remplissage. Par defaut : 0.
    :param total_length: 'bool' - Si non ``None``, la sortie sera remplie a la longueur :attr:`total_length`. Par defaut : None.
    :return:
        Un tuple de tenseurs contenant les sequences rembourrees et une liste de longueurs pour chaque sequence du lot. Les elements du lot seront reordonnes dans leur ordre d'origine.
    
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
    
     Compresse un Tenseur contenant des sequences rembourrees de longueurs variables. Si batch_first est True, `input` doit avoir la forme (taille du lot, longueur, ...), sinon la forme (longueur, taille du lot, ...).

    Pour les sequences non triees, utilisez ``enforce_sorted`` a False. Si :attr:`enforce_sorted` est ``True``, les sequences doivent etre triees par ordre decroissant de longueur.
    
    :param input: 'QTensor' - lots de sequences de longueurs variables pour le remplissage.
    :param lengths: 'list' - liste des longueurs de sequence pour chaque element du lot.
    :param batch_first: 'bool' - si ``True``, l'entree doit etre au format ``B x T x *``, par defaut : False.
    :param enforce_sorted: 'bool' - si ``True``, l'entree doit contenir des sequences dans l'ordre decroissant de longueur. Si ``False``, l'entree sera triee inconditionnellement. Par defaut : True.

    :return: Un objet :class:`PackedSequence`.

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
    
    Effectue une convolution 2D sur une image d'entree composee de plusieurs plans d'entree.

    :param x: Tenseur d'entree 4D.
    :param weight: Tenseur de noyau 4D.

    :param stride: `tuple` - pas, par defaut (1, 1)
    :param padding: Rembourrage, controle la quantite de rembourrage sur l'entree. Cela peut etre une chaine {'valid', 'same'} ou un tuple d'entiers specifiant la quantite de rembourrage implicite a appliquer a l'entree, par defaut (0,0).
    :param dilation: `tuple` - Espacement entre les elements du noyau. Par defaut : (0,0)
    :param groups: `int` - Nombre de groupes. Valeur par defaut : 1 

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

    Desactive l'enregistrement des nœuds de retropropagation lors du calcul avant.

    Example::

        import pyvqnet.tensor as tensor
        from pyvqnet import no_grad

        with no_grad():
            x = tensor.QTensor([1.0, 2.0, 3.0],requires_grad=True)
            y = tensor.tan(x)
            y.backward()
        \"\"\"
        RuntimeError: The output tensor does not require gradients (output.requires_grad == False). This may occur if you used a non-autograd function in the forward pass. To enable gradient computation, ensure that all operations are performed on tensors with requires_grad=True, or use autograd-compatible functions.
        \"\"\"
