.. _qtensor_api:

QTensor 模块
###########################

VQNet 量子机器学习使用 Python 接口的数据结构 QTensor。QTensor 支持常见的多维矩阵操作，包括创建函数、数学函数、逻辑函数、矩阵变换等。




QTensor 的函数和属性
******************************************


QTensor
==============================

.. py:class:: pyvqnet.tensor.tensor.QTensor(data, requires_grad=False, nodes=None, device=0, dtype=None, name='')

    具有动态计算图构建和自动微分功能的数据结构包装器。

    :param data: _core.Tensor 或 numpy 数组，表示一个 QTensor
    :param requires_grad: 是否跟踪张量的梯度，默认为 False
    :param nodes: 计算图中后继节点的列表，默认为 None
    :param device: 保存 QTensor 的当前设备，默认 = 0，使用 CPU。
    :param dtype: 参数的数据类型，默认为 None，使用默认数据类型：kfloat32，表示 32 位浮点数。
    :param name: QTensor 的名称，默认为 ""。

    :return: 输出 QTensor


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

        返回张量的维度数量。

        :return: 张量的维度数量。

        Example::

            from pyvqnet.tensor import QTensor

            a = QTensor([2.0, 3.0, 4.0, 5.0], requires_grad=True)
            print(a.ndim)

            # 1

    .. py:attribute:: shape

        返回张量的形状。

        :return: 张量各维度大小的列表。

        Example::

            from pyvqnet.tensor import QTensor

            a = QTensor([2.0, 3.0, 4.0, 5.0], requires_grad=True)
            print(a.shape)

            # [4]

    .. py:attribute:: size

        返回张量中元素的数量。

        :return: 张量中元素的数量。

        Example::

            from pyvqnet.tensor import QTensor

            a = QTensor([2.0, 3.0, 4.0, 5.0], requires_grad=True)
            print(a.size)

            # 4

    .. py:method:: numel

        返回张量中的元素数量。

        :return: 张量中的元素数量。

        Example::

            from pyvqnet.tensor import QTensor

            a = QTensor([2.0, 3.0, 4.0, 5.0], requires_grad=True)
            print(a.numel())

            # 4

    .. py:attribute:: dtype

        返回张量的数据类型。

        支持的数据类型如下：

            =========================================  ===============================
            dtype                                      描述
            =========================================  ===============================
            ``pyvqnet.kbool``                          布尔变量
            ``pyvqnet.kuint8``                         8 位无符号整数
            ``pyvqnet.kint8``                          8 位有符号整数
            ``pyvqnet.kint16``                         16 位有符号整数
            ``pyvqnet.kint32``                         32 位有符号整数
            ``pyvqnet.kint64``                         64 位有符号整数
            ``pyvqnet.kfloat32``                       32 位浮点数，参见 https://en.wikipedia.org/wiki/IEEE_754
            ``pyvqnet.kfloat64``                       64 位浮点数，参见 https://en.wikipedia.org/wiki/IEEE_754
            ``pyvqnet.kcomplex64``                     64 位复数，由两个 `float32` 组成
            ``pyvqnet.kcomplex128``                    128 位复数，由两个 `float64` 组成
            ``pyvqnet.kbfloat16``                      16 位浮点数，有时称为 Brain 浮点格式，位分配为 1 位符号位、8 位指数位和 7 位尾数位
            =========================================  ===============================

        :return: 张量的数据类型。

        Example::

            from pyvqnet.tensor import QTensor

            a = QTensor([2, 3, 4, 5])
            print(a.dtype)
            # 4


    .. py:method:: zero_grad()

        将梯度设置为零。将在优化过程中被优化器使用。

        :return: None

        Example::

            from pyvqnet.tensor import tensor
            from pyvqnet.tensor import QTensor
            t3  =  QTensor([2.0,3.0,4.0,5.0],requires_grad = True)
            t3.zero_grad()
            print(t3.grad)

            # [0, 0, 0, 0]


 
    .. py:method:: backward(grad=None)

        计算当前 QTensor 的梯度。

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

        将自身数据复制到新的 numpy.array。

        :return: 包含 QTensor 数据的新 numpy.array

        .. note::

            numpy 不支持 bfloat16 类型，您需要先转换为其他 numpy 支持的数据类型（如 float32）再调用此接口。

        Example::

            from pyvqnet.tensor import tensor
            from pyvqnet.tensor import QTensor
            t3  =  QTensor([2.0,3.0,4.0,5.0],requires_grad = True)
            t4 = t3.to_numpy()
            print(t4)

            # [2. 3. 4. 5.]

 
    .. py:method:: item()

            返回 QTensor 中的唯一元素。如果 QTensor 包含多个元素，则引发 'RuntimeError'。

            :return: 此对象的唯一数据

            Example::

                from pyvqnet.tensor import tensor

                t = tensor.ones([1])
                print(t.item())

                # 1.0

 
    .. py:method:: argmax(*kargs)

        返回输入 QTensor 中所有元素最大值的索引，或者
        返回 QTensor 在指定维度上最大值的索引。

        :param dim: dim (int) – 要缩减的维度，仅接受单个轴。如果 dim == None，则返回输入张量中所有元素最大值的索引。有效的 dim 范围为 [-R, R)，其中 R 是输入的 ndim。当 dim < 0 时，其工作方式与 dim + R 相同。
        :param keepdims:  whether the 输出 QTensor has dim retained or not.

        :return: 输入 QTensor 中最大值的索引。

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

        返回输入 QTensor 中所有元素最小值的索引，或者
        返回 QTensor 在指定维度上最小值的索引。

        :param dim: dim (int) – the dimension to reduce,only accepts single axis. if dim == None, returns the indices of the 最小值 of all elements in the 输入张量.The valid dim range is [-R, R), where R is input's ndim. when dim < 0, it works the same way as dim + R.
        :param keepdims:  whether the 输出 QTensor has dim retained or not.

        :return: 输入 QTensor 中最小值的索引。

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

            用指定值就地填充 QTensor。

            :param v: 标量值
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

            如果所有 QTensor 值均为非零，则返回 True。

            :return: 如果所有 QTensor 值均为非零，则返回 True。

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

            如果任意 QTensor 值为非零，则返回 True。

            :return: 如果任意 QTensor 值为非零，则返回 True。

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

        用从二项分布随机采样的值填充 QTensor。

        如果二项分布随机生成的数据大于二值化阈值，则将 QTensor 对应位置的值设为 1，否则为 0。

        :param v: 二值化阈值
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

        用从有符号均匀分布随机采样的值填充 QTensor。

        有符号均匀分布生成值的缩放因子。

        :param v: 标量值
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

        用从均匀分布随机采样的值填充 QTensor。

        均匀分布生成值的缩放因子。

        :param v: 标量值
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

        用从正态分布随机采样的值填充 QTensor。
        正态分布的均值。正态分布的标准差。
        是否使用快速数学模式。

        :param m: 正态分布的均值
        :param s: 正态分布的标准差
        :param fast_math: 是否使用快速数学模式
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

        反转或排列数组的轴。如果 new_dims = None，则反转维度。

        :param new_dims: 维度的新顺序（整数列表）。
        :return:  结果 QTensor。

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

        更改张量的形状，返回一个新的 QTensor。

        :param new_shape: 新形状（整数列表）
        :return: 一个新的 QTensor

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

        就地更改当前 QTensor 的形状。此接口将首先尝试在不更改原始内存数据的情况下进行变换。如果失败，当前数据将被复制到新内存中。

        .. warning::

            建议使用 reshape 接口。在某些情况下，实际底层内存位置会被复制而非就地修改。

        :param new_shape: 新形状（整数列表）
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

            获取 QTensor 的数据作为 NumPy 数组。

            :return: 一个 NumPy 数组

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

            支持 QTensor 的切片索引，或使用 QTensor 作为高级索引访问输入。将返回一个新的 QTensor。

            参数 start、stop 和 step 可以用冒号分隔，如 start:stop:step，其中 start、stop 和 step 可以省略。

            作为一维 QTensor，索引或切片只能在单个轴上进行。

            作为二维 QTensor 和多维 QTensor，索引或切片可以在多个轴上进行。

            如果您使用 QTensor 作为高级索引的索引，请参阅 numpy 的 `advanced indexing <https://docs.scipy.org/doc/numpy-1.10.1/reference/arrays.indexing.html>`_ 。

            如果用作索引的 QTensor 是逻辑运算的结果，则执行布尔索引。

            .. note:: 
                
                我们使用 a[3,4,1] 这样的索引形式，但不支持 a[3][4][1] 这种形式。

            :param item: 用作索引的整数或 QTensor。

            :return: 一个新的 QTensor。

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

        支持 QTensor 的切片索引，或使用 QTensor 作为高级索引访问输入。将返回一个新的 QTensor。

        参数 start、stop 和 step 可以用冒号分隔，如 start:stop:step，其中 start、stop 和 step 可以省略。

        作为一维 QTensor，索引或切片只能在单个轴上进行。

        作为二维 QTensor 和多维 QTensor，索引或切片可以在多个轴上进行。

        如果您使用 QTensor 作为高级索引的索引，请参阅 numpy 的 `advanced indexing <https://docs.scipy.org/doc/numpy-1.10.1/reference/arrays.indexing.html>`_ 。

        如果用作索引的 QTensor 是逻辑运算的结果，则执行布尔索引。

        .. note:: 
            
            我们使用 a[3,4,1] 这样的索引形式，但不支持 a[3][4][1] 这种形式。

        :param item: 用作索引的整数或 QTensor

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

        将 QTensor 克隆到指定的 GPU 设备。

        device 指定内部数据存储的设备。当 device >= DEV_GPU_0 时，数据存储在 GPU 上。
        如果您的计算机有多个 GPU，您可以指定不同的设备来存储数据。
        例如，device = DEV_GPU_1, DEV_GPU_2, DEV_GPU_3, ... 表示存储在不同编号的 GPU 上。
        
        .. note::
            QTensor 不能在不同的 GPU 上执行计算。
            如果您尝试在 ID 超过已验证 GPU 最大数量的 GPU 上创建 QTensor，将引发 Cuda 错误。

        :param device: 当前保存 QTensor 的设备，默认为 DEV_GPU_0，

        device = pyvqnet.DEV_GPU_0，存储在第一个 GPU 中，device = DEV_GPU_1，
        存储在第二个 GPU 中，以此类推。

        :return: 克隆到 GPU 设备的 QTensor。

        Examples::

            from pyvqnet.tensor import QTensor
            a = QTensor([2])
            b = a.GPU()
            print(b.device)
            #1000

 

    .. py:method:: CPU()

        将 QTensor 克隆到指定的 CPU 设备。

        :return: 克隆到 CPU 设备的 QTensor。

        Examples::

            from pyvqnet.tensor import QTensor
            a = QTensor([2])
            b = a.CPU()
            print(b.device)
            # 0

 
    .. py:method:: toGPU(device: int = DEV_GPU_0)

        将 QTensor 移动到指定的 GPU 设备。

        device specifies the device whose internal data is stored. When device >= DEV_GPU, the data is stored on the GPU.
        If your computer has multiple GPUs, you can designate different devices to store data on.
        例如，device = DEV_GPU_1, DEV_GPU_2, DEV_GPU_3, ... 表示存储在不同编号的 GPU 上。

        .. note::

            QTensor 不能在不同的 GPU 上执行计算。 如果您尝试在 ID 超过已验证 GPU 最大数量的 GPU 上创建 QTensor，将引发 Cuda 错误。

        :param device: The device currently saving QTensor, default=DEV_GPU_0. device = pyvqnet.DEV_GPU_0，存储在第一个 GPU 中，device = DEV_GPU_1， 存储在第二个 GPU 中，以此类推。
        :return: 移动到 GPU 设备的 QTensor。

        Examples::

            from pyvqnet.tensor import QTensor
            a = QTensor([2])
            a = a.toGPU()
            print(a.device)
            #1000


    
    .. py:method:: toCPU()

        将 QTensor 移动到 CPU。

        :return: 移动到 CPU 设备的 QTensor。

        Examples::

            from pyvqnet.tensor import QTensor
            a = QTensor([2])
            b = a.toCPU()
            print(b.device)
            # 0

    
    .. py:method:: isGPU()

        此 QTensor 的数据是否存储在 GPU 主机内存中。

        :return: 此 QTensor 的数据是否存储在 GPU 主机内存中。

        Examples::
        
            from pyvqnet.tensor import QTensor
            a = QTensor([2])
            a = a.isGPU()
            print(a)
            # False
 
    .. py:method:: isCPU()

        此 QTensor 的数据是否存储在 CPU 主机内存中。

        :return: 此 QTensor 的数据是否存储在 CPU 主机内存中。

        Examples::
        
            from pyvqnet.tensor import QTensor
            a = QTensor([2])
            a = a.isCPU()
            print(a)
            # True


创建函数
*****************************************************


ones
==============================

.. py:function:: pyvqnet.tensor.ones(shape,device=0,dtype-None)

    返回具有输入形状的全一张量。

    :param shape: 输入形状
    :param device: 存储在哪个设备中，默认为 0，即 CPU。
    :param dtype: 参数的数据类型，默认为 None，使用默认数据类型：kfloat32，表示 32 位浮点数。
    
    :return: 输出 QTensor with the 输入形状.

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

    返回与输入 QTensor 形状相同的全一张量。

    :param t: 输入 QTensor
    :param device: 存储在哪个设备中，默认为 0，即 CPU。
    :param dtype: 参数的数据类型，默认为 None，使用默认数据类型：kfloat32，表示 32 位浮点数。
    
    :return:  输出 QTensor


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

    创建指定形状的 QTensor 并用指定值填充它。

    :param shape: 要创建的 QTensor 的形状
    :param value: 用于填充 QTensor 的值。
    :param device: 使用的设备，默认为 0，使用 CPU 设备。
    :param dtype: 参数的数据类型，默认为 None，使用默认数据类型：kfloat32，表示 32 位浮点数。
    
    :return: 输出 QTensor

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

    创建指定形状的 QTensor 并用指定值填充它。

    :param t:  输入 QTensor
    :param value: 用于填充 QTensor 的值。
    :param device: 使用的设备，默认为 0，使用 CPU 设备。
    :param dtype: 参数的数据类型，默认为 None，使用默认数据类型：kfloat32，表示 32 位浮点数。
    
    :return: 输出 QTensor

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

    返回输入形状的零张量。

    :param shape: 张量的形状
    :param device: 使用的设备，默认为 0，使用 CPU 设备。
    :param dtype: 参数的数据类型，默认为 None，使用默认数据类型：kfloat32，表示 32 位浮点数。
    
    :return: 输出 QTensor

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

    返回与输入 QTensor 形状相同的零张量。

    :param t: 输入 QTensor
    :param device: 使用的设备，默认为 0，使用 CPU 设备。
    :param dtype: 参数的数据类型，默认为 None，使用默认数据类型：kfloat32，表示 32 位浮点数。
    
    :return:  输出 QTensor

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

    在给定区间内创建一个包含等间距值的一维 QTensor。

    :param start: 区间的起始值
    :param end: 区间的结束值
    :param step: 值之间的间距
    :param device: 使用的设备，默认为 0，使用 CPU 设备。
    :param dtype: 参数的数据类型，默认为 None，使用默认数据类型：kfloat32，表示 32 位浮点数。
    :param requires_grad: should tensor’s gradient be tracked, defaults to False
    :return: 输出 QTensor

    Example::

        from pyvqnet.tensor import tensor
        from pyvqnet.tensor import QTensor
        t = tensor.arange(2, 30,4)
        print(t)

        # [ 2,  6, 10, 14, 18, 22, 26]

linspace
==============================

.. py:function:: pyvqnet.tensor.linspace(start, end, num, device: int = 0,dtype=None, requires_grad= False)

    在给定区间内创建一个包含等间距值的一维 QTensor。

    :param start: 起始值
    :param end: 结束值
    :param nums: 生成的样本数量
    :param device: 使用的设备，默认为 0，使用 CPU 设备。
    :param dtype: 参数的数据类型，默认为 None，使用默认数据类型：kfloat32，表示 32 位浮点数。
    :param requires_grad: should tensor’s gradient be tracked, defaults to False
    :return: 输出 QTensor

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

    在对数刻度上创建一个包含等间距值的一维 QTensor。

    :param start: ``base ** start`` is the 起始值
    :param end: ``base ** end`` 是序列的最终值
    :param nums: 生成的样本数量
    :param base: 对数空间的底数
    :param device: 使用的设备，默认为 0，使用 CPU 设备。
    :param dtype: 参数的数据类型，默认为 None，使用默认数据类型：kfloat32，表示 32 位浮点数。
    :param requires_grad: should tensor’s gradient be tracked, defaults to False
    :return: 输出 QTensor

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

    创建一个 size x size 的 QTensor，对角线上为 1，其余位置为 0。

    :param size: 要创建的（方形）QTensor 的大小
    :param offset: 对角线索引：0（默认）表示主对角线，正值表示上对角线，负值表示下对角线。
    :param device: 使用的设备，默认为 0，使用 CPU 设备。
    :param dtype: 参数的数据类型，默认为 None，使用默认数据类型：kfloat32，表示 32 位浮点数。
    
    :return: 输出 QTensor

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


    返回 :attr:`t` 的一个部分视图，其中对角线元素作为维度附加到形状末尾，相对于 :attr:`dim1` 和 :attr:`dim2`。
    :attr:`offset` 是主对角线的偏移量。

    :param t: 输入张量
    :param offset: offset (0 means main diagonal, positive values ​​mean the nth diagonal above the main diagonal, negative values ​​mean the nth diagonal below the main diagonal)
    :param dim1: 取对角线的第一个维度。默认值：0。
    :param dim2: 取对角线的第二个维度。默认值：1。

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

    选择对角线元素或构造一个对角 QTensor。

    输入一个二维 QTensor，返回一个包含所选对角线元素的新一维张量。输入一个一维 QTensor，返回一个新的二维张量，其选定对角线元素为输入值，其余为 0。

    :param t: 输入 QTensor
    :param k: offset (0 for the main diagonal, positive for the nth
        diagonal above the main one, negative for the nth diagonal below the
        main one)
    :return: 输出 QTensor

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

    创建一个具有均匀分布随机值的 QTensor。

    :param shape: 要创建的 QTensor 的形状
    :param min: 均匀分布的最小值，默认值：0。
    :param max: 均匀分布的最大值，默认值：1。
    :param device: 使用的设备，默认为 0，使用 CPU 设备。
    :param dtype: 参数的数据类型，默认为 None，使用默认数据类型：kfloat32，表示 32 位浮点数。
    :param requires_grad: should tensor’s gradient be tracked, defaults to False
    :return: 输出 QTensor


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

    创建一个具有正态分布随机值的 QTensor。

    :param shape: 要创建的 QTensor 的形状
    :param mean: 正态分布的均值，默认值：0。
    :param std: 正态分布的标准差，默认值：1。
    :param device: 使用的设备，默认为 0，使用 CPU 设备。
    :param dtype: 参数的数据类型，默认为 None，使用默认数据类型：kfloat32，表示 32 位浮点数。
    :param requires_grad: should tensor’s gradient be tracked, defaults to False
    :return: 输出 QTensor

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
    
    创建一个由 :attr:total_count 和 :attr:probs 参数化的二项分布。

    :param total_counts: 伯努利试验次数。
    :param probs: 事件概率。

    :return:
        二项分布的 QTensor。

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

    返回一个张量，其中每行包含 num_samples 个索引样本。
    来自张量输入对应行的多项概率分布。

    :param t: 输入概率分布。
    :param num_samples: 样本数量。

    :return:
        输出样本索引

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

    返回输入 t 的上三角矩阵，其余部分设置为 0。

    :param t: 输入 QTensor
    :param diagonal: 偏移量，默认为 0。主对角线为 0，正值表示向上偏移，负值表示向下偏移。

    :return: 输出 QTensor

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

    返回输入 t 的下三角矩阵，其余部分设置为 0。

    :param t: 输入 QTensor
    :param diagonal: 偏移量，默认为 0。主对角线为 0，正值表示向上偏移，负值表示向下偏移。

    :return: 输出 QTensor

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


数学函数
*****************************************************


floor
==============================

.. py:function:: pyvqnet.tensor.floor(t)

    返回一个新的 QTensor，其中包含输入元素的向下取整值，即小于或等于每个元素的最大整数。

    :param t: 输入 QTensor
    :return: 输出 QTensor

    Example::

        from pyvqnet.tensor import tensor

        t = tensor.arange(-2.0, 2.0, 0.25)
        u = tensor.floor(t)
        print(u)

        # [-2, -2, -2, -2, -1, -1, -1, -1, 0, 0, 0, 0, 1, 1, 1, 1]

ceil
==============================

.. py:function:: pyvqnet.tensor.ceil(t)

    返回一个新的 QTensor，其中包含输入元素的向上取整值，即大于或等于每个元素的最小整数。

    :param t: 输入 QTensor
    :return: 输出 QTensor

    Example::

        from pyvqnet.tensor import tensor

        t = tensor.arange(-2.0, 2.0, 0.25)
        u = tensor.ceil(t)
        print(u)

        # [-2, -1, -1, -1, -1, -0, -0, -0, 0, 1, 1, 1, 1, 2, 2, 2]

round
==============================

.. py:function:: pyvqnet.tensor.round(t)

    将 QTensor 值四舍五入到最接近的整数。

    :param t: 输入 QTensor
    :return: 输出 QTensor

    Example::

        from pyvqnet.tensor import tensor

        t = tensor.arange(-2.0, 2.0, 0.4)
        u = tensor.round(t)
        print(u)

        # [-2, -2, -1, -1, -0, -0, 0, 1, 1, 2]

sort
==============================

.. py:function:: pyvqnet.tensor.sort(t, axis: int, descending=False, stable=True)

    沿指定轴对 QTensor 进行排序。

    :param t: 输入 QTensor
    :param axis: 排序轴
    :param descending: 是否降序排序
    :param stable:  是否使用稳定排序
    :return: 输出 QTensor

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

    返回一个与输入形状相同的索引数组，这些索引按排序顺序索引给定轴上的数据。

    :param t: 输入 QTensor
    :param axis: 排序轴
    :param descending: 是否降序排序
    :param stable:  是否使用稳定排序
    :return: 输出 QTensor

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

    返回输入张量沿给定轴的前 k 个最大元素。

    如果 if_descent 为 False，则返回前 k 个最小元素。

    :param t: 输入 QTensor
    :param k: 最大元素或最小元素的数量
    :param axis: 排序轴，默认为 -1，即最后一个轴
    :param if_descent: 排序顺序，默认为 True

    :return: 一个新的 QTensor

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

    返回输入张量沿给定轴的前 k 个最大元素的索引。

    如果 if_descent 为 False，则返回前 k 个最小元素的索引。

    :param t: 输入 QTensor
    :param k: 最大元素或最小元素的数量
    :param axis: 排序轴，默认为 -1，即最后一个轴
    :param if_descent: 排序顺序，默认为 True

    :return: 一个新的 QTensor

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

    逐元素相加两个 QTensor，等价于 t1 + t2。

    :param t1: 第一个 QTensor
    :param t2: 第二个 QTensor
    :return:  输出 QTensor

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

    逐元素相减两个 QTensor，等价于 t1 - t2。


    :param t1: 第一个 QTensor
    :param t2: 第二个 QTensor
    :return:  输出 QTensor

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

    逐元素相乘两个 QTensor，等价于 t1 * t2。

    :param t1: 第一个 QTensor
    :param t2: 第二个 QTensor
    :return:  输出 QTensor


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

    逐元素相除两个 QTensor，等价于 t1 / t2。


    :param t1: 第一个 QTensor
    :param t2: 第二个 QTensor
    :return:  输出 QTensor


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

    沿给定轴对 QTensor 中的所有元素求和。如果 axis = None，则对 QTensor 中所有元素求和。

    :param t: 输入 QTensor
    :param axis:  用于求和的轴，默认为 None
    :param keepdims:  输出张量是否保留维度，默认为 False
    :return:  输出 QTensor


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

    返回输入元素在指定维度轴上的累积和。

    :param t:  the 输入 QTensor
    :param axis:  计算的轴，默认为 -1，使用最后一个轴。

    :return:  输出 QTensor.

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

    获取 QTensor 沿指定轴的均值。

    :param t:  the 输入 QTensor.
    :param axis: 要缩减的维度。
    :param keepdims:  whether the 输出 QTensor has dim retained or not, defaults to False.
    :return: returns the mean value of the 输入 QTensor.

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

    获取 QTensor 中的中位数。

    :param t: the 输入 QTensor
    :param axis:  用于平均的轴，默认为 None
    :param keepdims:  whether the 输出 QTensor has dim retained or not, defaults to False

    :return: 返回输入或 QTensor 中值的中位数。

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

    获取 QTensor 中的标准差。


    :param t:  the 输入 QTensor
    :param axis:  用于计算标准差的轴，默认为 None
    :param keepdims:  whether the 输出 QTensor has dim retained or not, defaults to False
    :param unbiased:  whether to use Bessel’s correction,default true
    :return: 返回输入或 QTensor 中值的标准差。

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

    获取 QTensor 中的方差。


    :param t:  the 输入 QTensor.
    :param axis:  用于计算方差的轴，默认为 None。
    :param keepdims:  whether the 输出 QTensor has dim retained or not, defaults to False.
    :param unbiased:  whether to use Bessel’s correction,default true.


    :return: 获取 QTensor 中的方差。

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

    两个二维、三维、四维矩阵的矩阵乘法。

    :param t1: 第一个 QTensor
    :param t2: 第二个 QTensor
    :return:  输出 QTensor

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

    计算 ``t1`` 和 ``t2`` 的 Kronecker 积，用 :math:`\otimes` 表示。如果 ``t1`` 是一个 :math:`(a_0 \times a_1 \times \dots \times a_n)` 张量，``t2`` 是一个 :math:`(b_0 \times b_1 \times \dots \ times b_n)` 张量，结果将是一个 :math:`(a_0*b_0 \times a_1*b_1 \times \dots \times a_n*b_n)` 张量，其条目如下：
    
    .. math::
          (\text{input} \otimes \text{other})_{k_0, k_1, \dots, k_n} =
              \text{input}_{i_0, i_1, \dots, i_n} * \text{other}_{j_0, j_1, \dots, j_n},

    where :math:`k_t = i_t * b_t + j_t` is :math:`0 \leq t \leq n`.
    如果一个张量的维度少于另一个，它将被展开直至具有相同的维度数。

    :param t1: The 第一个 QTensor.
    :param t2: The 第二个 QTensor.
    
    :return: 输出 QTensor。

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
    
    使用基于爱因斯坦求和约定的符号，沿指定维度对输入操作数的元素乘积求和。

    .. note::

        此函数使用 opt_einsum (https://optimized-einsum.readthedocs.io/en/stable/) 来通过优化收缩顺序加速计算或减少内存消耗。当至少有三个输入时，会进行此优化。

        对于更复杂的 `einsum`，可以额外导入 opt_einsum 以直接在 QTensor 上计算。

    :param equation: 爱因斯坦求和的下标。
    :param operands: 要计算爱因斯坦求和的张量。

    :return:

        QTensor 结果。

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

    计算 QTensor 的逐元素倒数。

    :param t: 输入 QTensor
    :return: 输出 QTensor

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

    返回一个新的 QTensor，其中包含输入元素的符号。符号函数在 t < 0 时返回 -1，t==0 时返回 0，t > 0 时返回 1。

    :param t: 输入 QTensor
    :return: 输出 QTensor


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

    对 QTensor 元素进行一元取反。

    :param t: 输入 QTensor
    :return:  输出 QTensor

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

    返回输入的二维矩阵对角线元素之和。

    :param t: 输入二维 QTensor
    :param k: offset (0 for the main diagonal, positive for the nth
        diagonal above the main one, negative for the nth diagonal below the
        main one)
    :return: 输入的二维矩阵对角线元素之和。

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

    对输入 QTensor 的所有元素应用指数函数。

    :param t: 输入 QTensor
    :return:  输出 QTensor

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

    计算 QTensor 的逐元素反余弦。

    :param t: 输入 QTensor
    :return: 输出 QTensor

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

    计算 QTensor 的逐元素反正弦。

    :param t: 输入 QTensor
    :return: 输出 QTensor

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

    计算 QTensor 的逐元素反正切。

    :param t: 输入 QTensor
    :return: 输出 QTensor

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

    对输入 QTensor 的所有元素应用正弦函数。


    :param t: 输入 QTensor
    :return:  输出 QTensor

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

    对输入 QTensor 的所有元素应用余弦函数。


    :param t: 输入 QTensor
    :return:  输出 QTensor

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

    对输入 QTensor 的所有元素应用正切函数。


    :param t: 输入 QTensor
    :return:  输出 QTensor

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

    对输入 QTensor 的所有元素应用双曲正切函数。

    :param t: 输入 QTensor
    :return:  输出 QTensor

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

    对输入 QTensor 的所有元素应用双曲正弦函数。


    :param t: 输入 QTensor
    :return:  输出 QTensor

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

    对输入 QTensor 的所有元素应用双曲余弦函数。


    :param t: 输入 QTensor
    :return:  输出 QTensor

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

    Raises 第一个 QTensor to the power of 第二个 QTensor.

    :param t1: 第一个 QTensor
    :param t2: 第二个 QTensor
    :return:  输出 QTensor

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

    对输入 QTensor 的所有元素应用绝对值函数。

    :param t: 输入 QTensor
    :return:  输出 QTensor

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

    对输入 QTensor 的所有元素应用自然对数 (ln) 函数。

    :param t: 输入 QTensor
    :return:  输出 QTensor

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
    
    依次计算 softmax 函数和 log 函数在 axis 轴上的结果。

    :param t: 输入 QTensor .
    :param axis: 用于计算 softmax 的轴，默认为 -1。

    :return: 输出 QTensor。

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

    对输入 QTensor 的所有元素应用平方根函数。


    :param t: 输入 QTensor
    :return:  输出 QTensor

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

    对输入 QTensor 的所有元素应用平方函数。


    :param t: 输入 QTensor
    :return:  输出 QTensor

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
 
    返回复 Hermitian（共轭对称）或实对称矩阵的特征值和特征向量。

    返回两个对象，一个包含特征值的一维数组，
    以及对应的特征向量（按列排列）的二维方阵或矩阵（取决于输入类型）。

    :param: 输入 QTensor。
    :param: t 的特征值和特征向量。
    :return:

        返回特征值和特征向量。

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

    计算输入 QTensor 沿 axis 轴设置的张量的 F-范数，
    如果 axis 为 None，则返回所有元素的 F-范数。

    :param t: 输入 QTensor。
    :param axis: 用于计算 F 范数的轴，默认为 None。
    :param keepdims: 输出张量是否保留缩减后的维度，默认为 False。
    :return: 输出 QTensor 或 F-范数值。


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



逻辑函数
**************************

maximum
==============================

.. py:function:: pyvqnet.tensor.maximum(t1: pyvqnet.tensor.QTensor, t2: pyvqnet.tensor.QTensor)

    两个张量的逐元素最大值。


    :param t1: 第一个 QTensor
    :param t2: 第二个 QTensor
    :return:  输出 QTensor

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

    两个张量的逐元素最小值。


    :param t1: 第一个 QTensor
    :param t2: 第二个 QTensor
    :return:  输出 QTensor

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

    返回输入 QTensor 沿给定轴的最小元素。
    如果 axis == None，返回张量中所有元素的最小值。

    :param t: 输入 QTensor
    :param axis: 用于求最小值的轴，默认为 None
    :param keepdims:  输出张量是否保留维度，默认为 False
    :return: 输出 QTensor

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

    返回输入 QTensor 沿给定轴的最大元素。
    如果 axis == None，返回张量中所有元素的最大值。

    :param t: 输入 QTensor
    :param axis: 用于求最大值的轴，默认为 None
    :param keepdims:  输出张量是否保留维度，默认为 False
    :return: 输出 QTensor

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

    Clips 输入 QTensor to minimum and 最大值.

    :param t: 输入 QTensor
    :param min_val:  最小值
    :param max_val:  最大值
    :return:  输出 QTensor

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

    根据条件从 x 或 y 中选择元素返回。

    :param condition: 条件张量，数据类型必须为 kbool。
    :param t1: 条件满足时从中取元素的 QTensor，默认为 None
    :param t2: 条件不满足时从中取元素的 QTensor，默认为 None
    :return: 输出 QTensor

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

    返回包含非零元素索引的 QTensor。

    :param t: 输入 QTensor
    :return: 输出 QTensor contains indices of nonzero elements.

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

    逐元素测试有限性（不是无穷大也不是非数字）。

    :param t: 输入 QTensor
    :return: 输出 QTensor，当对应位置元素满足条件时返回 True，否则返回 False。

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

    逐元素测试正无穷或负无穷。

    :param t: 输入 QTensor
    :return: 输出 QTensor，当对应位置元素满足条件时返回 True，否则返回 False。
    
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

    逐元素测试是否为 NaN。

    :param t: 输入 QTensor
    :return: 输出 QTensor，当对应位置元素满足条件时返回 True，否则返回 False。
    
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

    逐元素测试负无穷。

    :param t: 输入 QTensor
    :return: 输出 QTensor，当对应位置元素满足条件时返回 True，否则返回 False。
    
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

    逐元素测试正无穷。

    :param t: 输入 QTensor
    :return: 输出 QTensor，当对应位置元素满足条件时返回 True，否则返回 False。
    
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

    逐元素计算 ``t1`` 与 ``t2`` 的真值。

    :param t1: 输入 QTensor
    :param t2: 输入 QTensor
    :return: 输出 QTensor，当对应位置元素满足条件时返回 True，否则返回 False。
    
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

    逐元素计算 ``t1 or t2`` 的真值。

    :param t1: 输入 QTensor
    :param t2: 输入 QTensor
    :return: 输出 QTensor，当对应位置元素满足条件时返回 True，否则返回 False。

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

    逐元素计算 ``not t`` 的真值。

    :param t: 输入 QTensor
    :return: 输出 QTensor，当对应位置元素满足条件时返回 True，否则返回 False。

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

    逐元素计算 ``t1 xor t2`` 的真值。

    :param t1: 输入 QTensor
    :param t2: 输入 QTensor

    :return: 输出 QTensor，当对应位置元素满足条件时返回 True，否则返回 False。

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

    逐元素返回 ``t1 > t2`` 的真值。


    :param t1: 输入 QTensor
    :param t2: 输入 QTensor
    :return: 输出 QTensor，当对应位置元素满足条件时返回 True，否则返回 False。

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

    逐元素返回 ``t1 >= t2`` 的真值。

    :param t1: 输入 QTensor
    :param t2: 输入 QTensor
    :return: 输出 QTensor，当对应位置元素满足条件时返回 True，否则返回 False。

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

    逐元素返回 ``t1 < t2`` 的真值。

    :param t1: 输入 QTensor
    :param t2: 输入 QTensor
    :return: 输出 QTensor，当对应位置元素满足条件时返回 True，否则返回 False。

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

    逐元素返回 ``t1 <= t2`` 的真值。

    :param t1: 输入 QTensor
    :param t2: 输入 QTensor
    :return: 输出 QTensor，当对应位置元素满足条件时返回 True，否则返回 False。

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

    逐元素返回 ``t1 == t2`` 的真值。

    :param t1: 输入 QTensor
    :param t2: 输入 QTensor
    :return: 输出 QTensor，当对应位置元素满足条件时返回 True，否则返回 False。
    
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

    逐元素返回 ``t1 != t2`` 的真值。

    :param t1: 输入 QTensor
    :param t2: 输入 QTensor
    :return: 输出 QTensor，当对应位置元素满足条件时返回 True，否则返回 False。
    
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
 
    计算两个 QTensor 元素的按位与。

    :param t1: 输入 QTensor t1。仅接受整数或布尔值作为有效输入。
    :param t2: 输入 QTensor t2。仅接受整数或布尔值作为有效输入。
    :return:
        结果 QTensor

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


矩阵操作
**********************

select
==============================

.. py:function:: pyvqnet.tensor.select(t: pyvqnet.tensor.QTensor, index)

    在 QTensor 中按给定轴返回 QTensor。以下操作获得相同结果的值。

    :param t: 输入 QTensor
    :param index: 包含输出维度的字符串
    :return: 输出 QTensor

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

    在特定限制下，较小的数组被放置到较大的数组中，使它们具有兼容的形状。此接口可以对输入参数张量进行自动微分。

    参考 https://numpy.org/doc/stable/user/basics.broadcasting.html

    :param t1: 输入 QTensor 1
    :param t2: 输入 QTensor 2

    :return t11: 具有新广播形状的 t1。
    :return t22: 具有新广播形状的 t2。

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

    Concatenate the 输入 QTensor along the axis and return 一个新的 QTensor.

    :param args: list consist of 输入 QTensors
    :param axis: 要连接的维度。必须在 0 和连接张量的维度数之间。
    :return: 输出 QTensor

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

    Join a sequence of arrays along a new axis,return 一个新的 QTensor.

    :param QTensors: 包含 QTensor 的列表
    :param axis: 要插入的维度。必须在 0 和堆叠张量的维度数之间。
    :return: 输出 QTensor

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

    反转或排列数组的轴。

    :param t: 输入 QTensor
    :param dim: 维度的新顺序（整数列表）
    :return: 输出 QTensor

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

    转置数组的轴。如果 dim = None，则反转维度。此函数与 permute 相同。

    :param t: 输入 QTensor
    :param dim: 维度的新顺序（整数列表）
    :return: 输出 QTensor

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

    通过重复 QTensor 指定次数来构造一个新的 QTensor。

    如果 reps 长度为 d，则结果 QTensor 的维度为 max(d, t.ndim)。

    如果 t.ndim < d，则 t 通过从起始维度插入新轴扩展到 d 维。
    因此形状为 (3,) 的数组在二维复制时被提升为 (1, 3)，或在三维复制时提升为 (1, 1, 3)。

    如果 t.ndim > d，则 reps 通过在其中插入 1 扩展到 t.ndim。

    因此对于形状为 (2, 3, 4, 5) 的 t，reps 为 (4, 3) 被视为 (1, 1, 4, 3)。

    :param t: 输入 QTensor
    :param reps: 每个维度的重复次数。
    :return: 一个新的 QTensor

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

    移除长度为 1 的轴。

    :param t: 输入 QTensor
    :param axis: 挤压轴，如果 axis = -1，则挤压所有大小为 1 的维度。
    :return: 输出 QTensor

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

    Return 一个新的 QTensor with a dimension of size one inserted at the specified position.

    :param t: 输入 QTensor
    :param axis: 要插入维度的轴。
    :return: 输出 QTensor

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

    将 `t` 的维度从 `source` 中的位置移动到 `destination` 中的位置。

    未明确移动的 `t` 的其他维度保留其原始顺序，并出现在 `destination` 中未指定的位置。

    :param t: 输入 QTensor。
    :param source: (整数或整数元组) 要移动的维度的原始位置。这些位置必须唯一。
    :param destination: (整数或整数元组) 每个原始维度的目标位置。这些位置也必须唯一。

    :return:
        新的 QTensor


    Example::

        from pyvqnet import QTensor,tensor
        a = tensor.arange(0,24).reshape((2,3,4))
        b = tensor.moveaxis(a,(1, 2), (0, 1))
        print(b.shape)


swapaxis
==============================

.. py:function:: pyvqnet.tensor.swapaxis(t, axis1: int, axis2: int)

    交换数组的两个轴。给定的维度 axis1 和 axis2 被互换。

    :param t: 输入 QTensor
    :param axis1: 第一个轴。
    :param axis2:  原始轴的目标位置。这些也必须唯一。
    :return: 输出 QTensor

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

    If mask == 1, fill with the 指定值. The shape of the mask must be broadcastable from the shape of the 输入 QTensor.

    :param t: 输入 QTensor
    :param mask: 一个 QTensor
    :param value: 指定值
    :return:  一个 QTensor

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

    从 start 维度到 end 维度展平 QTensor。

    :param t: 输入 QTensor
    :param start: 起始维度，默认为 0，从第一个维度开始。
    :param end: 结束维度，默认为 -1，到最后一个维度结束。
    :return:  输出 QTensor

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

    改变 QTensor 的形状，返回一个新形状的 QTensor。

    :param t: 输入 QTensor.
    :param new_shape: 新形状。

    :return: a 新形状。 QTensor.

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
    
    沿指定轴反转 QTensor，返回一个新的张量。

    :param t: 输入 QTensor。
    :param flip_dims: 要翻转的轴或轴列表。

    :return: 输出 QTensor。

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

    收集沿 'dim' 指定轴的值。

    对于三维张量，输出由以下公式指定：

    .. math::

        out[i][j][k] = t[index[i][j][k]][j][k] , if dim == 0 \\

        out[i][j][k] = t[i][index[i][j][k]][k] , if dim == 1 \\

        out[i][j][k] = t[i][j][index[i][j][k]] , if dim == 2 \\

    :param t: 输入 QTensor。
    :param dim: 聚合轴。
    :param index: 索引 QTensor，应与输入具有相同的维度大小。

    :return: 聚合结果。

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

    将张量 src 中的所有值写入 input 中由索引张量 indices 指定的位置。

    对于三维张量，输出由以下公式指定：

    .. math::

        input[indices[i][j][k]][j][k] = src[i][j][k] , if dim == 0 \\
        input[i][indices[i][j][k]][k] = src[i][j][k] , if dim == 1 \\
        input[i][j][indices[i][j][k]] = src[i][j][k] , if dim == 2 \\

    :param input: 输入 QTensor。
    :param dim: 散射轴。
    :param indices: 索引 QTensor，应与输入具有相同的维度大小。
    :param src: 要散射的源张量。

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

    在特定约束下，数组 t 被"广播"到参考形状，使它们具有兼容的形状。

    https://numpy.org/doc/stable/user/basics.broadcasting.html

    :param t: 输入 QTensor
    :param ref: 参考形状。
    
    :return: 新广播后的 t 的 QTensor。

    Example::

        from pyvqnet.tensor.tensor import QTensor
        from pyvqnet.tensor import *
        ref = [2,3,4]
        a = ones([4])
        b = tensor.broadcast_to(a,ref)
        print(b.shape)
        #[2, 3, 4]



实用函数
*****************************************************


to_tensor
==============================

.. py:function:: pyvqnet.tensor.to_tensor(x)

    如果输入数组还不是 Qtensor，则将其转换。

    :param x: 整数、浮点数或 numpy.array
    :return: 输出 QTensor

    Example::

        from pyvqnet.tensor import tensor
        t = tensor.to_tensor(10.0)
        print(t)
        # [10]


pad_sequence
==============================

.. py:function:: pyvqnet.tensor.pad_sequence(qtensor_list, batch_first=False, padding_value=0)

    用 ``padding_value`` 填充变长张量列表。``pad_sequence`` 沿新维度堆叠张量列表并将它们填充到等长。
    输入是大小为 ``L x *`` 的列表序列。L 是可变长度。

    :param qtensor_list: `list[QTensor]` - 变长序列的列表。
    :param batch_first: 'bool' - 如果为 True，输出将为 ``batch size x longest sequence length x *``，否则为 ``longest sequence length x batch size x *``。默认值：False。
    :param padding_value: 'float' - 填充值。默认值：0。

    :return:
         如果 batch_first 为 ``False``，张量大小为 ``batch size x longest sequence length x *``。
         否则张量大小为 ``longest sequence length x batch size x *``。

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
    
    填充一批打包的变长序列。它是 `pack_pad_sequence` 的逆操作。
    当 ``batch_first`` 为 True 时，返回形状为 ``B x T x *`` 的张量，否则返回 ``T x B x *``。
    其中 `T` 是最长序列长度，`B` 是批量大小。

    :param sequence: 'QTensor' - 要处理的数据。
    :param batch_first: 'bool' - 如果 ``True``，批量将是输入的第一个维度。默认值：False。
    :param padding_value: 'bool' - 填充值。默认值：0。
    :param total_length: 'bool' - 如果不为 ``None``，输出将被填充到长度 :attr:`total_length`。默认值：None。
    :return:
        包含填充序列的张量元组，以及批次中每个序列的长度列表。批次元素将按其原始顺序重新排列。
    
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
    
    打包包含变长填充序列的张量。如果 batch_first 为 True，`input` 的形状应为 [batch size, length,*]，否则形状为 [length, batch size,*]。

    对于未排序的序列，使用 ``enforce_sorted`` 为 False。如果 :attr:`enforce_sorted` 为 ``True``，序列应按长度降序排列。
    
    :param input: 'QTensor' - 用于填充的变长序列批次。
    :param lengths: 'list' - list of sequence lengths for each batch
         element.
    :param batch_first: 'bool' - if ``True``, the input is expected to be ``B x T x *``
         format, default: False.
    :param enforce_sorted: 'bool' - if ``True``, the input should be
         Contains sequences in descending order of length. If ``False``, the input will be sorted unconditionally. Default: True.

    :return: 一个 :class:`PackedSequence` 对象。

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
    
    在由多个输入平面组成的输入图像上执行二维卷积。

    :param x: 4D 输入张量.
    :param weight: 4D 卷积核张量。

    :param stride: `tuple` - 步长，默认为 (1, 1)
    :param padding: 填充，控制输入上的填充量。可以是字符串 {'valid', 'same'} 或整数元组，指定应用于输入的隐式填充量，默认为 (0,0)。
    :param dilation: `tuple` - 卷积核元素之间的间距。默认值：(0,0)
    :param groups: `int` - 分组数。默认值：1

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

    在前向计算禁用时记录反向传播节点。

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
