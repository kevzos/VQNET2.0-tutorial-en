Usar el módulo de aprendizaje automático cuántico pyQPanda3
###########################################################

.. warning::

    La parte de computación cuántica de la siguiente interfaz utiliza pyqpanda3.

    Si usa la función QCloud bajo este módulo, se producirán errores al importar pyqpanda2 en el código o al usar la interfaz del paquete relacionado con pyqpanda2 de pyvqnet.

Capa de computación cuántica
***********************************

.. _QuantumLayer_pq3:

QuantumLayer
============================

Si está familiarizado con la sintaxis de pyQPanda3, puede usar la interfaz QuantumLayer para personalizar el simulador pyqpanda3 y realizar cálculos.

.. py:class:: pyvqnet.qnn.pq3.quantumlayer.QuantumLayer(qprog_with_measure,para_num,diff_method:str = "parameter_shift",delta:float = 0.01,dtype=None,name="")

    Módulo de cómputo abstracto de capa variacional cuántica. Use pyQPanda3 para simular un circuito cuántico parametrizado y obtener los resultados de medición. Esta capa variacional cuántica hereda el módulo de cálculo de gradientes del framework VQNet. Puede usar el método de desplazamiento de parámetros para calcular el gradiente de los parámetros del circuito, entrenar modelos de circuitos cuánticos variacionales o incrustar circuitos cuánticos variacionales en modelos híbridos cuántico-clásicos.

    :param qprog_with_measure: Operaciones de circuito cuántico y funciones de medición construidas con pyQPanda.
    :param para_num: `int` - número de parámetros.
    :param diff_method: Método para calcular los gradientes de los parámetros del circuito cuántico, "parameter shift" o "finite difference", por defecto desplazamiento de parámetros.
    :param delta: \delta al calcular gradientes por diferencias finitas.
    :param dtype: tipo de datos del parámetro, por defecto: None, usa el tipo de datos predeterminado: kfloat32, que representa números de punto flotante de 32 bits.
    :param name: el nombre de este módulo, por defecto "".

    :return: un módulo que puede calcular circuitos cuánticos.

    .. note::

        qprog_with_measure es una función de circuito cuántico definida en pyQPanda3.

        Esta función debe contener dos parámetros, input y parameter, como entrada de la función (incluso si un parámetro no se usa realmente), y la salida es el resultado de medición o el valor esperado del circuito (debe ser np.ndarray o una lista que contenga valores), de lo contrario no funcionará correctamente en QpandaQCircuitVQCLayerLite.

        El uso de la función de circuito cuántico qprog_with_measure (input, param) se puede consultar en el siguiente ejemplo.

        `input`: Datos clásicos unidimensionales de entrada. Si no hay entrada, ingrese None

        `param`: Parámetros del circuito cuántico variacional unidimensional a entrenar

    .. note::

        Esta clase tiene los alias `QuantumLayerV2`, `QpandaQCircuitVQCLayerLite`.

    Ejemplo::

        from pyvqnet.qnn.pq3.measure import ProbsMeasure
        from pyvqnet.qnn.pq3.quantumlayer import QuantumLayer
        from pyvqnet.tensor import QTensor,ones
        import pyqpanda3.core as pq
        def pqctest (input,param):
            num_of_qubits = 4

            m_machine = pq.CPUQVM()

            qubits = range(num_of_qubits)

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
        pqc = QuantumLayer(pqctest,3)

        #datos clásicos como entrada
        input = QTensor([[1,2,3,4],[4,2,2,3],[3.0,3,2,2]] )

        #circuitos hacia adelante
        rlt = pqc(input)

        print(rlt)

        grad = ones(rlt.data.shape)*1000
        #circuitos hacia atrás
        rlt.backward(grad)

        print(pqc.m_para.grad)

QpandaQProgVQCLayer
=============================

.. py:class:: pyvqnet.qnn.pq3.quantumlayer.QuantumLayerV3(origin_qprog_func,para_num,qvm_type="cpu", pauli_str_dict=None, shots=1000, initializer=None,dtype=None,name="")

    Envía el circuito cuántico parametrizado al simulador local de amplitud completa QPanda3 para su cálculo y entrena los parámetros del circuito.
    Soporta datos por lotes y usa la regla de desplazamiento de parámetros para estimar el gradiente de los parámetros.
    Para CRX, CRY, CRZ, esta capa usa la fórmula de https://iopscience.iop.org/article/10.1088/1367-2630/ac2cb3, y el resto de las puertas lógicas usan el método de desplazamiento de parámetros predeterminado para calcular el gradiente.

    :param origin_qprog_func: La función de circuito cuántico invocable construida con QPanda.
    :param para_num: `int` - Número de parámetros; los parámetros son unidimensionales.
    :param qvm_type: `str` - Tipo de simulador pyqpanda3 a usar, `cpu` o `gpu`, por defecto `cpu`.
    :param pauli_str_dict: `dict|list` - Diccionario o lista de diccionarios que representan los operadores de Pauli en el circuito cuántico. Por defecto es None.
    :param shots: `int` - Número de mediciones. Por defecto es 1000.
    :param initializer: Inicializador para los valores de los parámetros. Por defecto es None.
    :param dtype: Tipo de datos del parámetro. Por defecto es None, lo que significa usar el tipo de datos predeterminado.
    :param name: Nombre del módulo. Por defecto es una cadena vacía.

    :return: Devuelve una clase QpandaQProgVQCLayer

    .. note::

        origin_qprog_func es una función de circuito cuántico definida por el usuario usando pyQPanda3.

        Esta función debe contener dos parámetros, input y parameter, como entrada de la función (incluso si un parámetro no se usa realmente), y la salida son datos de tipo pyqpanda3.core.QProg, de lo contrario no puede ejecutarse correctamente en QuantumLayerV3.

        origin_qprog_func (input,param )

        `input`: datos de clase de arreglo definidos por el usuario para datos clásicos unidimensionales

        `param`: parámetros de circuito cuántico unidimensionales definidos por el usuario tipo arreglo

    .. note::

        Esta clase tiene el alias `QuantumLayerV3` .

    Ejemplo::

        import pyqpanda3.core as pq
        import pyvqnet
        from pyvqnet.qnn.pq3.quantumlayer import  QuantumLayerV3


        def qfun(input, param ):
            m_qlist = range(3)
            cbits = range(3)
            measure_qubits = [0,1, 2]
            m_prog = pq.QProg()
            cir = pq.QCircuit(3)

            cir<<pq.RZ(m_qlist[0], input[0])
            cir<<pq.RX(m_qlist[2], input[2])
            
            qcir = pq.RX(m_qlist[1], param[1]).control(m_qlist[0])
        
            cir<<qcir

            qcir = pq.RY(m_qlist[0], param[2]).control(m_qlist[1])
        
            cir<<qcir

            cir<<pq.RY(m_qlist[0], input[1])

            qcir = pq.RZ(m_qlist[0], param[3]).control(m_qlist[1])
        
            cir<<qcir
            m_prog<<cir

            for idx, ele in enumerate(measure_qubits):
                m_prog << pq.measure(m_qlist[ele], cbits[idx])  # pylint: disable=expression-not-assigned
            return m_prog

        from pyvqnet.utils.initializer import ones
        l = QuantumLayerV3(qfun,
                        4,
                        "cpu",
                        pauli_str_dict=None,
                        shots=1000,
                        initializer=ones,
                        name="")
        x = pyvqnet.tensor.QTensor(
            [[2.56, 1.2,-3]],
            requires_grad=True)
        y = l(x)

        y.backward()
        print(l.m_para.grad.to_numpy())
        print(x.grad.to_numpy())

QuantumBatchAsyncQcloudLayer
==================================

Cuando instala la última versión de pyqpanda3, puede usar esta interfaz para definir un circuito variacional y enviarlo al chip real originqc para su ejecución.

.. py:class:: pyvqnet.qnn.pq3.quantumlayer.QuantumBatchAsyncQcloudLayer(origin_qprog_func, qcloud_token, para_num, pauli_str_dict=None, shots = 1000, initializer=None, dtype=None, name="", diff_method="parameter_shift", submit_kwargs={}, query_kwargs={})
    
    Un módulo de cómputo abstracto para el chip real originqc usando pyqpanda3 QCLOUD. Envía circuitos cuánticos parametrizados al chip real y obtiene los resultados de medición.
    Si diff_method == "random_coordinate_descent", la capa seleccionará aleatoriamente un solo parámetro para calcular el gradiente, y los otros parámetros permanecerán en cero. Referencia: https://arxiv.org/abs/2311.00088

    .. note::

        qcloud_token es el token de API que solicitó desde la plataforma en la nube.

        origin_qprog_func debe devolver datos de tipo pyqpanda3.core.QProg. Si no se establece pauli_str_dict, es necesario asegurarse de que la medición se haya insertado en el QProg.

        origin_qprog_func debe tener el siguiente formato:

        origin_qprog_func(input,param)

            `input`: Datos clásicos de entrada de 1~2 dimensiones. En el caso de 2D, la primera dimensión es el tamaño del lote

            `param`: Parámetros de entrada para entrenar el circuito cuántico variacional 1D

    .. note::

        En la versión actual, el tiempo de espera total predeterminado para el envío de un solo circuito a QCloud es de 60 segundos. Si se produce un tiempo de espera debido a que QCloud está ocupado, puede establecer el valor de la clave `total_timeout` en ``query_kwargs`` al número deseado de segundos de espera.

    :param origin_qprog_func: La función de circuito cuántico variacional construida con QPanda, que debe devolver un QProg.
    :param qcloud_token: `str` - El tipo de máquina cuántica o el token de la nube utilizado para la ejecución.
    :param para_num: `int` - El número de parámetros, el parámetro es un QTensor de tamaño [para_num].
    :param pauli_str_dict: `dict|list` - Un diccionario o lista de diccionarios que representan los operadores de Pauli en el circuito cuántico. El valor predeterminado es "None", que realiza operaciones de medición. Si se ingresa un diccionario de operadores de Pauli, se calculará una expectativa única o múltiples expectativas.
    :param shots: `int` - El número de mediciones. El valor predeterminado es 1000.
    :param initializer: Inicializador para los valores de los parámetros. El valor predeterminado es "None", que usa una distribución normal 0~2*pi.
    :param dtype: El tipo de datos del parámetro. El valor predeterminado es None, lo que significa usar el tipo de datos predeterminado pyvqnet.kfloat32.
    :param name: El nombre del módulo. El valor predeterminado es una cadena vacía.
    :param diff_method: Método de diferenciación para el cálculo del gradiente. El valor predeterminado es "parameter_shift", "random_coordinate_descent".
    :param submit_kwargs: Parámetros clave adicionales para enviar circuitos cuánticos, predeterminado: {"if_print_qcloud_log":False,"chip_id":"WK_C180","is_amend":True,"is_mapping":True,"is_optimization":True,"compile_level":3,"default_task_group_size":200,"test_qcloud_fake":False,"":"server_ip_address"}, cuando test_qcloud_fake se establece en True, simulación local con CPUQVM.
    :param query_kwargs: Parámetros clave adicionales para consultar resultados cuánticos, predeterminado: {"timeout":1,"total_timeout":60,"print_query_info":True,"sub_circuits_split_size":1}.
    :return: Un módulo que puede calcular circuitos cuánticos.

    Ejemplo::

        import pyqpanda3.core as pq
        import pyvqnet
        from pyvqnet.qnn.pq3.quantumlayer import QuantumBatchAsyncQcloudLayer

        def qfun(input,param):
            measure_qubits = [0,2]
            m_qlist = range(6)
            cir = pq.QCircuit(6)
            cir << (pq.RZ(m_qlist[0],input[0]))
            cir << pq.CNOT(m_qlist[0],m_qlist[1])
            cir << pq.RY(m_qlist[1],param[0])
            cir << pq.CNOT(m_qlist[0],m_qlist[2])
            cir << pq.RZ(m_qlist[1],input[1])
            cir << pq.RY(m_qlist[2],param[1])
            cir << pq.H(m_qlist[2])
            m_prog = pq.QProg(cir)


            for idx, ele in enumerate(measure_qubits):
                m_prog << pq.measure(m_qlist[ele], m_qlist[idx])  # pylint: disable=expression-not-assigned
            return m_prog

        l = QuantumBatchAsyncQcloudLayer(qfun,
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
            cir = pq.QCircuit(6)
            cir<<pq.RZ(m_qlist[0],input[0])
            cir<<pq.CNOT(m_qlist[0],m_qlist[1])
            cir<<pq.RY(m_qlist[1],param[0])
            cir<<pq.CNOT(m_qlist[0],m_qlist[2])
            cir<<pq.RZ(m_qlist[1],input[1])
            cir<<pq.RY(m_qlist[2],param[1])
            cir<<pq.H(m_qlist[2])
            m_prog = pq.QProg(cir)

        
            
            return m_prog
        l = QuantumBatchAsyncQcloudLayer(qfun2,
                "3047DE8A59764BEDAC9C3282093B16AF",
                2,

                pauli_str_dict={'Z0 X1':10,'':-0.5,'Y2':-0.543,"":3333},
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


QuantumLayerAdjoint
============================

.. py:class:: pyvqnet.qnn.pq3.quantumlayer.QuantumLayerAdjoint(pq3_vqc_circuit, param_num, pauli_dicts, dtype = None, name="")

    Esta clase usa la interfaz VQCircuit de pyqpanda3 para calcular los gradientes de los parámetros en un circuito cuántico con respecto al hamiltoniano usando el método adjunto.

    Esta clase soporta entrada por lotes y múltiples salidas hamiltonianas.

    .. note::

        Al usar esta interfaz, debe construir el circuito usando puertas lógicas de VQCircuit.

        Actualmente, solo se soportan puertas lógicas limitadas; se lanzará una excepción si no son compatibles.

        El parámetro de entrada ``pq3_vqc_circuit`` solo puede contener dos parámetros, `x` y `param`, que deben ser un arreglo o lista unidimensional.

        En la función ``pq3_vqc_circuit``, los usuarios deben usar ``pyqpanda3.vqcircuit.VQCircuit().set_Param`` para personalizar cómo se manejan las entradas y los parámetros.

        Además, los usuarios deben ingresar previamente el número de parámetros en ``param_num``. Esta interfaz inicializará un parámetro ``m_para`` con una longitud de ``param_num``.

        Vea el ejemplo a continuación.

    :param pq3_vqc_circuit: Personaliza el circuito VQCircuit de pyqpanda3.
    :param param_num: Número de parámetros. :param pauli_dicts: Observaciones esperadas, puede ser una lista.
    :param dtype: Tipo de parámetro, kfloat32 o kfloat64, por defecto: None, usa kfloat32.
    :param name: El nombre de esta interfaz.
    :return: Devuelve una instancia de QuantumLayerAdjoint


    Ejemplo::

        from pyvqnet.qnn.pq3 import QuantumLayerAdjoint
        from pyvqnet import tensor

        from pyqpanda3.vqcircuit import VQCircuit
        import pyqpanda3 as pq3

        l = 3
        n = 7
        def pqctest(x,param):
            vqc = VQCircuit()
            vqc.set_Param([len(param) +len(x)])
            w_offset = len(x)
            for j in range(len(x)):
                vqc << pq3.core.RX(j, vqc.Param([j  ]))
            for j in range(l):
                for i in range(n - 1):
                    vqc << pq3.core.CNOT(i, i + 1)
                for i in range(n):
                    vqc << pq3.core.RX(i, vqc.Param([w_offset + 3 * n * j + i]))
                        
                    vqc << pq3.core.RZ(i, vqc.Param([w_offset + 3 * n * j + i + n]))
                    vqc << pq3.core.RY(i, vqc.Param([w_offset + 3 * n * j + i + 2 * n]))
            
            return vqc

        Xn_string = ' '.join([f'X{i}' for i in range(n)])
        pauli_dict  = {Xn_string:1.}

        layer = QuantumLayerAdjoint(pqctest,3*l*n,pauli_dict)

        x = tensor.randn([2,5])
        x.requires_grad = True
        y = layer(x)
        y.backward()
        print(layer.m_para.grad)
        print(x.grad)

        Xn_string = ' '.join([f'X{i}' for i in range(n)])
        Zn_string = ' '.join([f'Z{i}' for i in range(n)])
        pauli_dict  = {Xn_string:1.,Zn_string:0.5}

        layer = QuantumLayerAdjoint(pqctest,3*l*n,pauli_dict)

        x = tensor.randn([2,5])
        x.requires_grad = True
        y = layer(x)
        y.backward()
        print(layer.m_para.grad)
        print(x.grad)

        Xn_string = ' '.join([f'X{i}' for i in range(n)])
        Zn_string = ' '.join([f'Z{i}' for i in range(n)])
        pauli_dict  = {Xn_string:1.,Zn_string:0.5}

        layer = QuantumLayerAdjoint(pqctest,3*l*n,pauli_dict)

        x = tensor.randn([1,5])
        x.requires_grad = True
        y = layer(x)
        y.backward()
        print(layer.m_para.grad)
        print(x.grad)

        Xn_string = ' '.join([f'X{i}' for i in range(n)])
        Zn_string = ' '.join([f'Z{i}' for i in range(n)])
        pauli_dict  = [{Xn_string:1.,Zn_string:0.5},{Xn_string:1.,Zn_string:0.5}]

        layer = QuantumLayerAdjoint(pqctest,3*l*n,pauli_dict)

        x = tensor.randn([1,5])
        x.requires_grad = True
        y = layer(x)
        y.backward()
        print(layer.m_para.grad)
        print(x.grad)

        Xn_string = ' '.join([f'X{i}' for i in range(n)])
        Zn_string = ' '.join([f'Z{i}' for i in range(n)])
        pauli_dict  = [{Xn_string:1.,Zn_string:0.5},{Xn_string:1.,Zn_string:0.5}]

        layer = QuantumLayerAdjoint(pqctest,3*l*n,pauli_dict)

        x = tensor.randn([2,5])
        x.requires_grad = True
        y = layer(x)
        y.backward()
        print(layer.m_para.grad)
        print(x.grad)
        """
        [-0.1086438, 0.1805159, 0.2619071,..., 0.1508062, 0.0329617,-0.0043367]
        <QTensor [63] DEV_CPU kfloat32>

        [[-0.0425088, 0.0187212,-0.0326243, 0.1314874,-0.0729216],
        [-0.0972663,-0.0371378,-0.0455299,-0.0170686,-0.0328533]]
        <QTensor [2, 5] DEV_CPU kfloat32>

        [ 0.0706403,-0.1070583, 0.0547093,...,-0.0183769,-0.0742296, 0.0026942]
        <QTensor [63] DEV_CPU kfloat32>

        [[-0.07577  ,-0.1364278, 0.0220043, 0.0690343, 0.0281384],
        [ 0.0075356,-0.1627405,-0.0381604, 0.1185545, 0.1409108]]
        <QTensor [2, 5] DEV_CPU kfloat32>

        [-0.0634308,-0.0128268, 0.0396237,...,-0.0350691,-0.116307 , 0.0164972]
        <QTensor [63] DEV_CPU kfloat32>

        [[-0.0823639,-0.0418629, 0.0105356, 0.0699336, 0.041226 ]]
        <QTensor [1, 5] DEV_CPU kfloat32>

        [-0.1281752, 0.0852512, 0.0678721,...,-0.080481 , 0.0202518,-0.0348869]
        <QTensor [63] DEV_CPU kfloat32>

        [[-0.0339751,-0.0330053,-0.0651799, 0.2171837,-0.1267595]]
        <QTensor [1, 5] DEV_CPU kfloat32>

        [ 0.305574 , 0.2730191, 0.0605986,...,-0.2138517,-0.2475468, 0.174026 ]
        <QTensor [63] DEV_CPU kfloat32>

        [[ 0.1867954,-0.0704528,-0.0603823,-0.0123921,-0.0938597],
        [-0.041001 ,-0.2520995, 0.0683114,-0.0986969, 0.1000023]]
        <QTensor [2, 5] DEV_CPU kfloat32>
        """

VQCQCloudLayer
============================

.. py:class:: pyvqnet.qnn.pq3.vqc_qcloud_layer.VQCQCloudLayer(vqc_module, qcloud_token, pauli_str_dict=None, shots=1000, name="", submit_kwargs={}, query_kwargs={})

    Envía el Módulo VQC al chip real QCloud o al simulador local pyqpanda3 para su ejecución.

    Propagación hacia adelante: En lugar de ejecutar el cálculo del circuito cuántico variacional de VQNet, llama al chip cuántico real o al simulador local qpanda para realizar el cálculo.

    Propagación hacia atrás: Usa la regla de desplazamiento de parámetros para calcular gradientes. Para cada dimensión de entrada y cada parámetro entrenable en el VQC,
    genera circuitos desplazados en +/- pi/2 y los envía para su cálculo, recupera los resultados para calcular el jacobiano. Los gradientes se establecen en el tensor de entrada y los Parámetros entrenables del VQC.

    .. note::

        El tiempo de espera total predeterminado para el envío de un solo circuito a QCloud es de 60 segundos. Si se produce un tiempo de espera debido a que QCloud está ocupado, puede establecer la clave ``total_timeout`` en ``query_kwargs`` para especificar los segundos de espera.

    .. note::

        No puede definir una función de medición (como ``MeasureAll``) en ``vqc_module``. La medición debe especificarse mediante el parámetro ``pauli_str_dict`` para indicar los observables.
        Por ejemplo: ``VQCQCloudLayer(vqc_module, token, pauli_str_dict={'Z0': 1, 'Z1': 1})``.

    :param vqc_module: Módulo VQC de VQNet, debe incluir un QMachine con save_ir=True.
    :param qcloud_token: Token de API de QCloud. Pase una cadena vacía si usa un simulador local.
    :param pauli_str_dict: Diccionario de operadores de Pauli para el cálculo del valor esperado. Por defecto es None, lo que realiza la operación de medición.
    :param shots: Número de mediciones. Por defecto es 1000.
    :param name: Nombre del módulo. Por defecto es una cadena vacía.
    :param submit_kwargs: Parámetros clave adicionales para enviar circuitos cuánticos, predeterminado: {"if_print_qcloud_log":False,"chip_id":"WK_C180","is_amend":True,"is_mapping":True,"is_optimization":True,"compile_level":3,"default_task_group_size":200,"test_qcloud_fake":False,"":"server_ip_address"}, cuando test_qcloud_fake se establece en True, simulación local con CPUQVM.
    :param query_kwargs: Argumentos clave adicionales para consultar resultados cuánticos. Predeterminado: {"timeout":1,"total_timeout":60, "print_query_info":True,"sub_circuits_split_size":1}.

    Ejemplo::

        from pyvqnet.qnn.vqc import *
        from pyvqnet.qnn.pq3 import VQCQCloudLayer
        from pyvqnet.nn import Module
        import pyvqnet

        class QModel(Module):
            def __init__(self, num_wires, dtype):
                super(QModel, self).__init__()
                self._num_wires = num_wires
                self._dtype = dtype
                self.qm = QMachine(num_wires, dtype=dtype, save_ir=True)
                self.rx_layer = RX(has_params=True, trainable=False, wires=0)
                self.u1 = U1(has_params=True, trainable=True, wires=[1])
                self.cnot = CNOT(wires=[0, 1])

            def forward(self, x, *args, **kwargs):
                self.qm.reset_states(x.shape[0])
                self.rx_layer(params=x[:, [0]], q_machine=self.qm)
                self.cnot(q_machine=self.qm)
                self.u1(q_machine=self.qm)
                return x

        qmodel = QModel(num_wires=2, dtype=pyvqnet.kcomplex64)
        layer = VQCQCloudLayer(
            qmodel,
            "your_qcloud_token",
            pauli_str_dict={'Z0': 1, 'Z1': 1},
            shots=1000,
            submit_kwargs={"test_qcloud_fake": True},
        )
        x = pyvqnet.tensor.QTensor([[0.5, 0.3], [0.5, 0.3]], requires_grad=True)
        y = layer(x)
        y.backward()
        print(x.grad)
        print(qmodel.u1.params.grad)

grad
===============
.. py:function:: pyvqnet.qnn.pq3.quantumlayer.grad(quantum_prog_func, input_params, *args)

    La función grad proporciona una interfaz para calcular el gradiente de los parámetros del circuito cuántico parametrizado diseñado por el usuario.
    Los usuarios pueden usar pyQPanda3 para diseñar la función de ejecución del circuito ``quantum_prog_func`` como se muestra a continuación, y pasarla como parámetro a la función grad.
    El segundo parámetro de la función grad son las coordenadas del gradiente del parámetro de la puerta lógica cuántica que desea calcular.
    La forma del valor de retorno es [número de parámetros, número de salidas].

    :param quantum_prog_func: función de ejecución del circuito cuántico diseñada con pyQPanda3.
    :param input_params: parámetros para los cuales se calculará el gradiente.
    :param \*args: otros parámetros de entrada para la función quantum_prog_func.
    :return:
        Gradiente de los parámetros


    Ejemplos::

        from pyvqnet.qnn.pq3 import grad, ProbsMeasure
        import pyqpanda3.core as pq

        def pqctest(param):
            machine = pq.CPUQVM()
        
            qubits = range(2)
            circuit = pq.QCircuit(2)

            circuit<<pq.RX(qubits[0], param[0])

            circuit<<pq.RY(qubits[1], param[1])
            circuit<<pq.CNOT(qubits[0], qubits[1])

            circuit<<pq.RX(qubits[1], param[2])

            prog = pq.QProg()
            prog<<circuit

            EXP = ProbsMeasure(machine,prog,[1])
            return EXP


        g = grad(pqctest, [0.1,0.2, 0.3])
        print(g)
        exp = pqctest([0.1,0.2, 0.3])
        print(exp)





QLinear
==============

QLinear implementa un algoritmo de conexión completa cuántico. Primero, los datos se codifican en un estado cuántico, y luego se realiza la operación de evolución y la medición a través de circuitos cuánticos para obtener el resultado final de conexión completa.

.. image:: ./images/qlinear_cir.png

.. py:class:: pyvqnet.qnn.qlinear.QLinear(input_channels,output_channels,machine: str = "CPU")

    Módulo de conexión completa cuántico. La entrada al módulo de conexión completa tiene forma (canales de entrada, canales de salida). Tenga en cuenta que esta capa no toma parámetros cuánticos variacionales.

    :param input_channels: `int` - Número de canales de entrada.
    :param output_channels: `int` - Número de canales de salida.
    :param machine: `str` - La máquina virtual a usar, se usa simulación CPU por defecto.
    :return: Capa de conexión completa cuántica.

    Ejemplo::

        from pyvqnet.tensor import QTensor
        from pyvqnet.qnn.qlinear import QLinear
        params = [[0.37454012, 0.95071431, 0.73199394, 0.59865848, 0.15601864, 0.15599452], 
        [1.37454012, 0.95071431, 0.73199394, 0.59865848, 0.15601864, 0.15599452],
        [1.37454012, 1.95071431, 0.73199394, 0.59865848, 0.15601864, 0.15599452],
        [1.37454012, 1.95071431, 1.73199394, 1.59865848, 0.15601864, 0.15599452]]
        m = QLinear(6, 2)
        input = QTensor(params, requires_grad=True)
        output = m(input)
        output.backward()
        print(output)

        #[
        #[0.0568473, 0.1264389],
        #[0.1524036, 0.1264389],
        #[0.1524036, 0.1442845],
        #[0.1524036, 0.1442845]
        #]



Qconv
==========================

    Qconv es una interfaz de algoritmo de convolución cuántica.
    La operación de convolución cuántica usa circuitos cuánticos para realizar operaciones de convolución sobre datos clásicos. No necesita calcular operaciones de multiplicación y suma. Solo necesita codificar los datos en estados cuánticos, y luego realizar operaciones de evolución y medición a través de circuitos cuánticos para obtener los resultados finales de convolución.
    Solicita el mismo número de bits cuánticos según el número de datos de entrada en el rango del kernel de convolución, y luego construye circuitos cuánticos para el cálculo.

    .. image:: ./images/qcnn.png

    El circuito cuántico se codifica insertando primero puertas :math:`RY` y :math:`RZ` en cada qubit, y luego usando :math:`Z` y :math:`U3` en dos qubits cualesquiera para entrelazar e intercambiar información. A continuación se muestra un ejemplo de 4 qubits

    .. image:: ./images/qcnn_cir.png

.. py:class:: pyvqnet.qnn.qcnn.qconv.QConv(input_channels,output_channels,quantum_number,stride=(1, 1),padding=(0, 0),kernel_initializer=normal,machine:str = "CPU", dtype=None, name ="")

    Módulo de convolución cuántica. Reemplaza el kernel Conv2D con un circuito cuántico. La entrada del módulo conv tiene forma (tamaño de lote, canales de entrada, altura, ancho) `Samuel et al. (2020) <https://arxiv.org/abs/2012.12177>`_ .

        :param input_channels: `int` - Número de canales de entrada.
        :param output_channels: `int` - Número de canales de salida.
        :param quantum_number: `int` - El tamaño de un solo kernel.
        :param stride: `tuple` - El paso, predeterminado (1,1).
        :param padding: `tuple` - Relleno, predeterminado (0,0).
        :param kernel_initializer: `callable` - Predeterminado a distribución normal.
        :param machine: `str` - La máquina virtual a usar, predeterminado a simulación CPU.
        :param dtype: El tipo de datos del parámetro, por defecto: None, usa el tipo de datos predeterminado: kfloat32, que representa números de punto flotante de 32 bits.
        :param name: El nombre de este módulo, predeterminado "".

        :return: Capa de convolución cuántica.


        Ejemplo::

            from pyvqnet.tensor import tensor
            from pyvqnet.qnn.qcnn.qconv import QConv
            x = tensor.ones([1,3,4,4])
            layer = QConv(input_channels=3, output_channels=2, quantum_number=4, stride=(2, 2))
            y = layer(x)
            print(y)

            # [
            # [[[-0.0889078, -0.0889078],
            #  [-0.0889078, -0.0889078]],
            # [[0.7992646, 0.7992646],
            #  [0.7992646, 0.7992646]]]
            # ]

Puertas lógicas cuánticas
************************************

La forma de procesar los bits cuánticos es mediante puertas lógicas cuánticas. Usando puertas lógicas cuánticas, evolucionamos conscientemente los estados cuánticos. La puerta lógica cuántica es la base del algoritmo cuántico.

Puerta lógica cuántica básica
=============================

En esta sección, usamos varias puertas lógicas de pyqpanda3 desarrolladas por Origin Quantum para construir circuitos cuánticos y realizar simulaciones cuánticas.
Las puertas lógicas actualmente soportadas por pyQPanda3 se pueden consultar en la definición de Puerta lógica cuántica de pyQPanda3.
Además, VQNet también encapsula algunas combinaciones de puertas lógicas cuánticas comúnmente usadas en aprendizaje automático cuántico:


BasicEmbeddingCircuit
============================

.. py:function:: pyvqnet.qnn.pq3.template.BasicEmbeddingCircuit(input_feat,qlist)
    
    Codifica n características binarias en el estado base de n qubits.

    Por ejemplo, para ``features=([0, 1, 1])``, el estado base es :math:`|011 \rangle` en un sistema cuántico.

    :param input_feat: entrada binaria de tamaño ``(n)``.
    :param qlist: qubits para construir el circuito de plantilla.
    :return: circuito cuántico.


    Ejemplo::

            from pyvqnet.qnn.pq3.template import BasicEmbeddingCircuit
            import pyqpanda3.core as pq
            from pyvqnet import tensor
            input_feat = tensor.QTensor([1,1,0])
            
            qlist = range(3)
            circuit = BasicEmbeddingCircuit(input_feat,qlist)
            print(circuit)


AngleEmbeddingCircuit
============================

.. py:function:: pyvqnet.qnn.pq3.template.AngleEmbeddingCircuit(input_feat,qubits,rotation:str='X')

    Codifica :math:`N` características en el ángulo de rotación de :math:`n` qubits, donde :math:`N \leq n`.

    La rotación se puede elegir como: 'X', 'Y', 'Z', según lo define el parámetro ``rotation``:

    * ``rotation='X'`` Usa la característica como el ángulo de la rotación RX.

    * ``rotation='Y'`` Usa la característica como el ángulo de la rotación RY.

    * ``rotation='Z'`` Usa la característica como el ángulo de la rotación RZ.

    La longitud de ``features`` debe ser menor o igual al número de qubits. Si la longitud en ``features`` es menor que los qubits, el circuito no aplica las puertas de rotación restantes.

    :param input_feat: arreglo numpy que representa los parámetros.
    :param qubits: índices de los qubits.
    :param rotation: qué rotación usar, predeterminado "X".
    :return: circuito cuántico.

    Ejemplo::

        from pyvqnet.qnn.pq3.template import AngleEmbeddingCircuit
        import numpy as np 
        m_qlist = range(2)

        input_feat = np.array([2.2, 1])
        C = AngleEmbeddingCircuit(input_feat,m_qlist,'X')
        print(C)
        C = AngleEmbeddingCircuit(input_feat,m_qlist,'Y')
        print(C)
        C = AngleEmbeddingCircuit(input_feat,m_qlist,'Z')
        print(C)

IQPEmbeddingCircuits
============================

.. py:function:: pyvqnet.qnn.pq3.template.IQPEmbeddingCircuits(input_feat,qubits,rep:int = 1)
    
    Codifica :math:`n` características en :math:`n` qubits usando puertas diagonales de un circuito IQP.

    La codificación es propuesta por `Havlicek et al. (2018) <https://arxiv.org/pdf/1804.11326.pdf>`_.

    El circuito IQP básico se puede repetir especificando ``n_repeats``.

    :param input_feat: arreglo numpy que representa los parámetros.
    :param qubits: lista de índices de qubits.
    :param rep: Repite el bloque del circuito cuántico, el número predeterminado de veces es 1.
    :return: circuito cuántico.

    Ejemplo::

        import numpy as np
        from pyvqnet.qnn.pq3.template import IQPEmbeddingCircuits
        input_feat = np.arange(1,100)
        qlist = range(3)
        circuit = IQPEmbeddingCircuits(input_feat,qlist,rep = 3)
        print(circuit)


RotCircuit
============================

.. py:function:: pyvqnet.qnn.pq3.template.RotCircuit(para,qubits)

    Rotación arbitraria de un solo qubit. El número de qlists debe ser 1, y el número de parámetros debe ser 3.

    .. math::

        R(\phi,\theta,\omega) = RZ(\omega)RY(\theta)RZ(\phi)= \begin{bmatrix}
        e^{-i(\phi+\omega)/2}\cos(\theta/2) & -e^{i(\phi-\omega)/2}\sin(\theta/2) \\
        e^{-i(\phi-\omega)/2}\sin(\theta/2) & e^{i(\phi+\omega)/2}\cos(\theta/2)
        \end{bmatrix}.

    :param para: arreglo numpy que representa los parámetros :math:`[\phi, \theta, \omega]`.
    :param qubits: índice del qubit, solo se acepta un solo qubit.
    :return: circuito cuántico.

    Ejemplo::

        from pyvqnet.qnn.pq3.template import RotCircuit
        import pyqpanda3.core as pq
        from pyvqnet import tensor

        m_qlist = 1

        param =tensor.QTensor([3,4,5])
        c = RotCircuit(param,m_qlist)
        print(c)

CRotCircuit
============================

.. py:function:: pyvqnet.qnn.pq3.template.CRotCircuit(para,control_qubits,rot_qubits)

    Operador Rot controlado.

    .. math:: 
        
        CR(\phi, \theta, \omega) = \begin{bmatrix}
        1 & 0 & 0 & 0 \\
        0 & 1 & 0 & 0\\
        0 & 0 & e^{-i(\phi+\omega)/2}\cos(\theta/2) & -e^{i(\phi-\omega)/2}\sin(\theta/2)\\
        0 & 0 & e^{-i(\phi-\omega)/2}\sin(\theta/2) & e^{i(\phi+\omega)/2}\cos(\theta/2)
        \end{bmatrix}.

    :param para: Un arreglo numpy que representa los parámetros :math:`[\phi, \theta, \omega]`.
    :param control_qubits: índice del qubit de control, el número de qubits debe ser 1.
    :param rot_qubits: índice del qubit de rotación, el número de qubits debe ser 1.
    :return: circuito cuántico.

    Ejemplo::

        from pyvqnet.qnn.pq3.template import CRotCircuit
        import pyqpanda3.core as pq
        import numpy as np
        m_qlist = range(1)
        control_qlist = [1]
        param = np.array([3,4,5])
        cir = CRotCircuit(param,control_qlist,m_qlist)

        print(cir)


CSWAPcircuit
============================

.. py:function:: pyvqnet.qnn.pq3.template.CSWAPcircuit(qubits)

    Circuito SWAP controlado.

    .. math:: 
        
        CSWAP = \begin{bmatrix}
        1 & 0 & 0 & 0 & 0 & 0 & 0 & 0 \\
        0 & 1 & 0 & 0 & 0 & 0 & 0 & 0 \\
        0 & 0 & 1 & 0 & 0 & 0 & 0 & 0 \\
        0 & 0 & 0 & 1 & 0 & 0 & 0 & 0 \\
        0 & 0 & 0 & 0 & 1 & 0 & 0 & 0 \\
        0 & 0 & 0 & 0 & 0 & 0 & 1 & 0 \\
        0 & 0 & 0 & 0 & 0 & 1 & 0 & 0 \\
        0 & 0 & 0 & 0 & 0 & 0 & 0 & 1
        \end{bmatrix}.

    .. note:: 
        
        El primer qubit proporcionado corresponde al **qubit de control** .

    :param qubits: lista de índices de qubits. El primer qubit es el qubit de control. La longitud de qlist debe ser 3.
    :return: El circuito cuántico.

    Ejemplo::

        from pyvqnet.qnn.pq3 import CSWAPcircuit
        import pyqpanda3.core as pq
        m_machine = pq.CPUQVM()

        m_qlist = range(3)

        c =CSWAPcircuit([m_qlist[1],m_qlist[2],m_qlist[0]])
        print(c)


Controlled_Hadamard
=======================

.. py:function:: pyvqnet.qnn.pq3.template.Controlled_Hadamard(qubits)
    
    Puerta lógica Hadamard controlada

    .. math:: 
        CH = \begin{bmatrix}
        1 & 0 & 0 & 0 \\
        0 & 1 & 0 & 0 \\
        0 & 0 & \frac{1}{\sqrt{2}} & \frac{1}{\sqrt{2}} \\
        0 & 0 & \frac{1}{\sqrt{2}} & -\frac{1}{\sqrt{2}}
        \end{bmatrix}.

    :param qubits: índice del qubit.

    Ejemplos::

        import pyqpanda3.core as pq

        machine = pq.CPUQVM()
        
        qubits =range(2)
        from pyvqnet.qnn.pq3 import Controlled_Hadamard

        cir = Controlled_Hadamard(qubits)
        print(cir)

CCZ
==============

.. py:function:: pyvqnet.qnn.pq3.template.CCZ(qubits)

    Puerta lógica controlada-controlada-Z.

    .. math::

        CCZ =
        \begin{pmatrix}
        1 & 0 & 0 & 0 & 0 & 0 & 0 & 0\\
        0 & 1 & 0 & 0 & 0 & 0 & 0 & 0 & 0\\
        0 & 0 & 1 & 0 & 0 & 0 & 0 & 0 & 0\\
        0 & 0 & 0 & 1 & 0 & 0 & 0 & 0\\
        0 & 0 & 0 & 0 & 0 & 1 & 0 & 0 & 0\\
        0 & 0 & 0 & 0 & 0 & 0 & 1 & 0 & 0\\
        0 & 0 & 0 & 0 & 0 & 0 & 0 & 1 & 0 & 0\\
        0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & -1
        \end{pmatrix}

    :param qubits: índice del qubit.

    :return:
        QCircuit de pyQPanda3

    Ejemplo::

        import pyqpanda3.core as pq

        machine = pq.CPUQVM()

        qubits = range(3)

        from pyvqnet.qnn.pq3 import CCZ

        cir = CCZ(qubits)


FermionicSingleExcitation
===========================

.. py:function:: pyvqnet.qnn.pq3.template.FermionicSingleExcitation(weight, wires, qubits)

    Operador de excitación simple de cluster acoplado para la exponenciación de productos tensoriales de matrices de Pauli. La forma matricial está dada por:

    .. math::

        \hat{U}_{pr}(\theta) = \mathrm{exp} \{ \theta_{pr} (\hat{c}_p^\dagger \hat{c}_r
        -\mathrm{H.c.}) \},

    :param weight: parámetro variable en el qubit p.
    :param wires: representa un subconjunto de índices de qubits en el intervalo [r, p]. La longitud mínima debe ser 2. El primer valor de índice se interpreta como r y el último valor de índice se interpreta como p.
        Los índices intermedios son operados por puertas CNOT para calcular la paridad del conjunto de qubits.
    :param qubits: índices de qubits.

    :return:
        QCircuit de pyQPanda3

    Ejemplos::

        from pyvqnet.qnn.pq3 import FermionicSingleExcitation, expval

        weight=0.5
        import pyqpanda3.core as pq
        machine = pq.CPUQVM()

        qlists = range(3)

        cir = FermionicSingleExcitation(weight, [1, 0, 2], qlists)


FermionicDoubleExcitation
============================

.. py:function:: pyvqnet.qnn.pq3.template.FermionicDoubleExcitation(weight, wires1, wires2, qubits)

    Operador de doble excitación de cluster acoplado para el producto tensorial de matrices de Pauli, la forma matricial está dada por:

    .. math::

        \hat{U}_{pqrs}(\theta) = \mathrm{exp} \{ \theta (\hat{c}_p^\dagger \hat{c}_q^\dagger
        \hat{c}_r \hat{c}_s - \mathrm{H.c.}) \},

    donde :math:`\hat{c}` y :math:`\hat{c}^\dagger` son los operadores de aniquilación y
    creación de fermiones, y los índices :math:`r, s` y :math:`p, q` corresponden a orbitales moleculares ocupados y
    vacíos respectivamente. Usando la `transformación de Jordan-Wigner
    <https://arxiv.org/abs/1208.5986>`_ el operador fermiónico definido anteriormente se puede escribir
    en términos de la matriz de Pauli (consulte
    `arXiv:1805.04340 <https://arxiv.org/abs/1805.04340>`_ para más detalles)

    .. math::

        \hat{U}_{pqrs}(\theta) = \mathrm{exp} \Big\{
        \frac{i\theta}{8} \bigotimes_{b=s+1}^{r-1} \hat{Z}_b \bigotimes_{a=q+1}^{p-1}
        \hat{Z}_a (\hat{X}_s \hat{X}_r \hat{Y}_q \hat{X}_p +
        \hat{Y}_s \hat{X}_r \hat{Y}_q \hat{Y}_p +\\ \hat{X}_s \hat{Y}_r \hat{Y}_q \hat{Y}_p +
        \hat{X}_s \hat{X}_r \hat{X}_q \hat{Y}_p - \mathrm{H.c.} ) \Big\}

    :param weight: parámetro variable
    :param wires1: representa el subconjunto de qubits en el intervalo [s, r] ocupado por la lista de índices de qubits. El primer índice se interpreta como s y el último índice como r. La puerta CNOT opera en los índices intermedios para calcular la paridad de un conjunto de qubits.
    :param wires2: representa el subconjunto de qubits en el intervalo [q, p] ocupado por la lista de índices de qubits. El primer índice raíz se interpreta como q y el último índice como p. La puerta CNOT opera en el índice intermedio para calcular la paridad de un conjunto de qubits.
    :param qubits: índices de qubits.

    :return:
        QCircuit de pyQPanda3

    Ejemplos::

        import pyqpanda3.core as pq
        from pyvqnet.qnn.pq3 import FermionicDoubleExcitation, expval
        machine = pq.CPUQVM()
        
        qlists = range(5)
        weight = 1.5
        cir = FermionicDoubleExcitation(weight,
                                        wires1=[0, 1],
                                        wires2=[2, 3, 4],
                                        qubits=qlists)


UCCSD
===================

.. py:function:: pyvqnet.qnn.pq3.template.UCCSD(weights, wires, s_wires, d_wires, init_state, qubits)

    Implementa la Simulación de Excitación Simple y Doble de Cluster Acoplado Unitario (UCCSD). UCCSD es una simulación VQE, comúnmente utilizada para ejecutar simulaciones de química cuántica.

    En la aproximación de Trotter de primer orden, la función unitaria UCCSD está dada por:

    .. math::

        \hat{U}(\vec{\theta}) =
        \prod_{p > r} \mathrm{exp} \Big\{\theta_{pr}
        (\hat{c}_p^\dagger \hat{c}_r-\mathrm{H.c.}) \Big\}
        \prod_{p > q > r > s} \mathrm{exp} \Big\{\theta_{pqrs}
        (\hat{c}_p^\dagger \hat{c}_q^\dagger \hat{c}_r \hat{c}_s-\mathrm{H.c.}) \Big\}
    
    donde :math:`\hat{c}` y :math:`\hat{c}^\dagger` son los operadores de aniquilación y

    creación de fermiones, y los índices :math:`r, s` y :math:`p, q` son los orbitales moleculares ocupados y
    vacíos, respectivamente. (Para más detalles, consulte
    `arXiv:1805.04340 <https://arxiv.org/abs/1805.04340>`_):

    :param weights: tensor de tamaño ``(len(s_wires)+ len(d_wires))`` que contiene los parámetros
     :math: `\theta_{pr}` y :math: `\theta_{pqrs}` de entrada para las rotaciones Z ``FermionicSingleExcitation`` y ``FermionicDoubleExcitation`` .
    :param wires: índices de qubits a ser plantillados
    :param s_wires: secuencia de listas que contiene índices de qubits ``[r,...,p]`` generados por una excitación simple
     :math: `\vert r, p \rangle = \hat{c}_p^\dagger \hat{c}_r \vert \mathrm{HF} \rangle`, donde :math:`\vert \mathrm{HF} \rangle` denota el estado de referencia de Hartree-Fock.
    :param d_wires: secuencia de listas, cada una conteniendo dos listas. Especifica los índices ``[s, ...,r]`` y ``[q,..., p]`` . Define la excitación dual :math:`\vert s, r, q, p \rangle = \hat{c}_p^\dagger \hat{c}_q^\dagger \hat{c}_r\hat{c}_s \vert \mathrm{HF} \rangle` .
    :param init_state: vector de número de ocupación de longitud ``len(wires)`` que representa el estado de alta frecuencia. ``init_state`` Inicializa el estado del qubit.
    :param qubits: Índice del qubit.

    Ejemplos::
        
        import pyqpanda3.core as pq
        from pyvqnet.tensor import tensor
        from pyvqnet.qnn.pq3 import UCCSD, expval
        machine = pq.CPUQVM()
        
        qlists = range(6)
        weight = tensor.zeros([8])
        cir = UCCSD(weight,wires = [0,1,2,3,4,5,6],
                                        s_wires=[[0, 1, 2], [0, 1, 2, 3, 4], [1, 2, 3], [1, 2, 3, 4, 5]],
                                        d_wires=[[[0, 1], [2, 3]], [[0, 1], [2, 3, 4, 5]], [[0, 1], [3, 4]], [[0, 1], [4, 5]]],
                                        init_state=[1, 1, 0, 0, 0, 0],
                                        qubits=qlists)

QuantumPoolingCircuit
============================

.. py:function:: pyvqnet.qnn.pq3.template.QuantumPoolingCircuit(sources_wires, sinks_wires, params,qubits)

    Circuito cuántico que reduce la muestra de datos.

    Para reducir el número de qubits en el circuito, primero se crean pares de qubits en el sistema. Después de emparejar inicialmente todos los qubits, se aplica la unitaria generalizada de 2 qubits a cada par de qubits. Y después de aplicar estas unitarias de dos qubits, se ignora un qubit en cada par de qubits para el resto de la red neuronal.

    :param sources_wires: Índices de qubits fuente que se ignorarán.
    :param sinks_wires: Índices de qubits destino que se conservarán.
    :param params: Parámetros de entrada.
    :param qubits: Índices de qubits.

    :return:
        QCircuit de pyQPanda3

    Ejemplos::

        from pyvqnet.qnn.pq3.template import QuantumPoolingCircuit
        import pyqpanda3.core as pq
        from pyvqnet import tensor

        qlists = range(4)
        p = tensor.full([6], 0.35)
        cir = QuantumPoolingCircuit([0, 1], [2, 3], p, qlists)
        print(cir)

Combinaciones de circuitos cuánticos de uso común
***********************************************************
VQNet proporciona algunos circuitos cuánticos comúnmente utilizados en la investigación de aprendizaje automático cuántico

HardwareEfficientAnsatz
=============================

.. py:class:: pyvqnet.qnn.pq3.ansatz.HardwareEfficientAnsatz(qubits,single_rot_gate_list,entangle_gate="CNOT",entangle_rules='linear',depth=1)

    Implementación de Hardware Efficient Ansatz introducido en el artículo: `Hardware-efficient Variational Quantum Eigensolver for Small Molecules <https://arxiv.org/pdf/1704.05018.pdf>`__ .

    :param qubits: índice del qubit.
    :param single_rot_gate_list: Lista de puertas de rotación de un solo qubit que consiste en una o más puertas de rotación que actúan sobre cada qubit. Actualmente se soportan Rx, Ry, Rz.
    :param entangle_gate: Puerta de entrelazamiento no paramétrica. Soporta CNOT, CZ. Predeterminado: CNOT.
    :param entangle_rules: Cómo se usa la puerta de entrelazamiento en el circuito. ``linear`` significa que la puerta de entrelazamiento actuará sobre cada qubit adyacente. ``all`` significa que la puerta de entrelazamiento actuará sobre cualquier par de qubits. Predeterminado: ``linear``.
    :param depth: Profundidad del ansatz, predeterminado: 1.

    :return:
        Una instancia de HardwareEfficientAnsatz

    Ejemplo::

        import pyqpanda3.core as pq
        from pyvqnet.tensor import QTensor,tensor
        from pyvqnet.qnn.pq3.ansatz import HardwareEfficientAnsatz
        machine = pq.CPUQVM()

        qlist = range(4)
        c = HardwareEfficientAnsatz(qlist,["rx", "RY", "rz"],
                                entangle_gate="cnot",
                                entangle_rules="linear",
                                depth=1)
        w = tensor.ones([c.get_para_num()])

        cir = c.create_ansatz(w)
        print(cir)

BasicEntanglerTemplate
============================

.. py:class:: pyvqnet.qnn.pq3.template.BasicEntanglerTemplate(weights=None, num_qubits=1, rotation=pyqpanda3.RX)
    
    Una capa que consiste en rotaciones de un solo qubit con un solo parámetro en cada qubit, seguida de múltiples puertas CNOT combinadas en una cadena o anillo cerrado.
    El anillo de puertas CNOT conecta cada qubit con sus vecinos, considerando el último qubit como vecino del primero.
    El número de capas :math:`L` está determinado por la primera dimensión del parámetro ``weights``.

    :param weights: Un tensor de pesos de forma `(L, len(qubits))`. Cada peso se usa como parámetro en una puerta cuántica paramétrica. El valor predeterminado es: ``None``, entonces se usan números aleatorios de distribución normal `(1,1)` como pesos.
    :param num_qubits: El número de qubits, predeterminado es 1.
    :param rotation: Usa una puerta de un solo qubit con un solo parámetro, ``pyqpanda3.RX`` se usa como valor predeterminado.
    :return:
        Una instancia de BasicEntanglerTemplate

    Ejemplo::

        import pyqpanda3.core as pq
        import numpy as np
        from pyvqnet.qnn.pq3 import BasicEntanglerTemplate
        np.random.seed(42)
        num_qubits = 5
        shape = [1, num_qubits]
        weights = np.random.random(size=shape)

        machine = pq.CPUQVM()

        qubits = range(num_qubits)

        circuit = BasicEntanglerTemplate(weights=weights, num_qubits=num_qubits, rotation=pq.RZ)
        result = circuit.compute_circuit()
        circuit.print_circuit(qubits)


StronglyEntanglingTemplate
============================

.. py:class:: pyvqnet.qnn.pq3.template.StronglyEntanglingTemplate(weights=None, num_qubits=1, ranges=None)

    Capas que consisten en rotaciones de un solo qubit y entrelazadores, como en el `diseño de clasificador centrado en circuitos <https://arxiv.org/abs/1804.00633>`__ .
    El parámetro ``weights`` contiene los pesos de cada capa. Por lo tanto, el número de capas :math:`L` es igual a la primera dimensión de ``weights``.
    Contiene puertas CNOT de 2 qubits que actúan sobre :math:`M` qubits, :math:`i = 1,...,M`. El segundo número de qubit de cada puerta está dado por la fórmula :math:`(i+r)\mod M`, donde :math:`r` es un hiperparámetro llamado ``range``, y :math:`0 < r < M`.

    :param weights: Tensor de pesos de forma ``(L, M, 3)``, valor predeterminado: None, usa un tensor aleatorio de forma ``(1,1,3)``.
    :param num_qubits: Número de qubits, valor predeterminado: 1.
    :param ranges: Secuencia que determina los hiperparámetros de rango para cada capa subsiguiente; valor predeterminado: None, usa :math:`r=l \mod M` como valor de ranges.
    :return:
        Una instancia de StronglyEntanglingTemplate

    Ejemplo::

        from pyvqnet.qnn.pq3 import StronglyEntanglingTemplate
        import pyqpanda3.core as pq
        from pyvqnet.tensor import *
        import numpy as np
        np.random.seed(42)
        num_qubits = 3
        shape = [2, num_qubits, 3]
        weights = np.random.random(size=shape)

        machine = pq.CPUQVM()

        qubits = range(num_qubits)

        circuit = StronglyEntanglingTemplate(weights, num_qubits=num_qubits )
        result = circuit.compute_circuit()
        print(result)
        circuit.print_circuit(qubits)


ComplexEntanglingTemplate
============================

.. py:class:: pyvqnet.qnn.pq3.ComplexEntanglingTemplate(weights,num_qubits,depth)

    Capa fuertemente entrelazada que consiste en puertas U3 y puertas CNOT.
    Esta plantilla de circuito proviene del siguiente artículo: https://arxiv.org/abs/1804.00633.

    :param weights: parámetros, forma [depth,num_qubits,3]
    :param num_qubits: número de qubits.
    :param depth: profundidad del subcircuito.
    :return:
        Una instancia de ComplexEntanglingTemplate

    Ejemplo::

        from pyvqnet.qnn.pq3 import ComplexEntanglingTemplate
        import pyqpanda3.core as pq
        from pyvqnet.tensor import *
        depth=3
        num_qubits = 8
        shape = [depth, num_qubits, 3]
        weights = tensor.randn(shape)

        machine = pq.CPUQVM()

        qubits = range(num_qubits)

        circuit = ComplexEntanglingTemplate(weights, num_qubits=num_qubits,depth=depth)
        result = circuit.create_circuit(qubits)
        circuit.print_circuit(qubits)


Quantum_Embedding
============================

.. py:class:: pyvqnet.qnn.pq3.Quantum_Embedding(qubits, machine, num_repetitions_input, depth_input, num_unitary_layers, num_repetitions)

    Usa RZ, RY, RZ para crear un circuito cuántico variacional que codifica datos clásicos en estados cuánticos.
    Referencia `Quantum embeddings for machine learning <https://arxiv.org/abs/2001.03622>`_.

    Después de inicializar la clase, su función miembro ``compute_circuit`` es la función de ejecución, que se puede usar como parámetro de entrada para la clase ``QuantumLayerV2`` para formar una capa del modelo de aprendizaje automático cuántico.

    :param qubits: Los bits cuánticos solicitados por pyQPanda3.
    :param machine: Máquina virtual cuántica solicitada por pyQPanda3.
    :param num_repetitions_input: El número de repeticiones de codificación de la entrada en el submódulo.
    :param depth_input: La dimensión de características de los datos de entrada.
    :param num_unitary_layers: El número de repeticiones de la puerta cuántica variacional en cada submódulo.
    :param num_repetitions: El número de repeticiones del submódulo.
    :return:
        Una instancia de Quantum_Embedding

    Ejemplo::

        from pyvqnet.qnn.pq3 import QuantumLayerV2,Quantum_Embedding
        from pyvqnet.tensor import tensor
        import pyqpanda3.core as pq
        depth_input = 2
        num_repetitions = 2
        num_repetitions_input = 2
        num_unitary_layers = 2

        loacl_machine = pq.CPUQVM()

        nq = depth_input * num_repetitions_input
        qubits = range(nq)
        cubes = range(nq)

        data_in = tensor.ones([12, depth_input])
        data_in.requires_grad = True

        qe = Quantum_Embedding(nq, loacl_machine, num_repetitions_input,
        depth_input, num_unitary_layers, num_repetitions)
        qlayer = QuantumLayerV2(qe.compute_circuit,
        qe.param_num)

        data_in.requires_grad = True
        y = qlayer.forward(data_in)
        y.backward()
        print(data_in.grad)


Medición de circuitos cuánticos
************************************

expval
============================

.. py:function:: pyvqnet.qnn.pq3.measure.expval(machine,prog,pauli_str_dict,shots=1000,noise_model=None)

    El valor esperado de la observación hamiltoniana proporcionada.

    Si la observación es :math:`0.7Z\otimes X\otimes I+0.2I\otimes Z\otimes I`,
    entonces el diccionario hamiltoniano será ``{{'Z0, X1':0.7} ,{'Z1':0.2}}``.

    La API expval ahora soporta el simulador pyQPanda3.

    :param machine: La máquina cuántica creada por pyQPanda3.
    :param prog: El programa cuántico creado por pyQPanda3.
    :param pauli_str_dict: Valor observado hamiltoniano.
    :param shots: Número de mediciones, predeterminado es 1000.
    :param noise_model: Modelo de ruido a aplicar, predeterminado es None (simulación ideal).

    :return: valor esperado.

    Ejemplo::

        import pyqpanda3.core as pq
        from pyvqnet.qnn.pq3.measure import expval
        input = [0.56, 0.1]
        m_machine = pq.CPUQVM()

        m_qlist = range(3)
        cir = pq.QCircuit(3)
        cir<<pq.RZ(m_qlist[0],input[0])
        cir<<pq.CNOT(m_qlist[0],m_qlist[1])
        cir<<pq.RY(m_qlist[1],input[1])
        cir<<pq.CNOT(m_qlist[0],m_qlist[2])
        m_prog = pq.QProg(cir)

        pauli_dict = {'Z0 X1':10,'Y2':-0.543}
        exp2 = expval(m_machine,m_prog,pauli_dict)
        print(exp2)
 


QuantumMeasure
=============================

.. py:function:: pyvqnet.qnn.pq3.measure.QuantumMeasure(machine,prog,measure_qubits:list,shots:int = 1000, qcloud_option="",noise_model=None)

    Calcula las mediciones de circuitos cuánticos. Devuelve las mediciones obtenidas mediante métodos de Monte Carlo.

    Para más detalles sobre la medición, consulte la documentación de pyQPanda3.

    La API QuantumMeasure actualmente solo soporta ``CPUQVM`` o ``QCloud`` de pyQPanda3.

    :param machine: La máquina virtual cuántica asignada por pyQPanda3.
    :param prog: El programa cuántico creado por pyQPanda3.
    :param measure_qubits: Lista que contiene los índices de los bits de medición.
    :param shots: El número de mediciones, el valor predeterminado es 1000.
    :param qcloud_option: Establece la configuración de qcloud, el valor predeterminado es "", puede pasar una clase QCloudOptions, que solo es útil cuando se usa qcloud.
    :param noise_model: Modelo de ruido a aplicar, predeterminado es None (simulación ideal).
    :return: Devuelve los resultados de medición obtenidos por el método de Monte Carlo.

    Ejemplo::

        from pyqpanda3.core import *
        circuit = QCircuit(3)
        circuit << H(0)
        circuit << P(2, 0.2)
        circuit << RX(1, 0.9)
        circuit << RX(0, 0.6)
        circuit << RX(1, 0.3)
        circuit << RY(1, 0.3)
        circuit << RY(2, 2.7)
        circuit << RX(0, 1.5)

        prog = QProg()
        prog.append(circuit)

        machine = CPUQVM()


        from pyvqnet.qnn.pq3.measure import probs_measure,quantum_measure

        measure_result = quantum_measure(machine,prog,[2,0])
        print(measure_result)


ProbsMeasure
============================

.. py:function:: pyvqnet.qnn.pq3.measure.ProbsMeasure(machine,prog,measure_qubits:list,shots=1,noise_model=None)

    Calcula las mediciones de probabilidad del circuito.

    Para más detalles, consulte la documentación de pyQPanda3 sobre medición de probabilidad.

    La API ProbsMeasure actualmente solo soporta ``CPUQVM`` o ``QCloud`` de pyQPanda.

    :param measure_qubits: Lista que contiene los índices de los qubits de medición.
    :param prog: El programa cuántico creado por qpanda.
    :param machine: La máquina virtual cuántica asignada por pyQPanda.
    :param shots: Número de mediciones, predeterminado es 1, que calcula el valor teórico.
    :param noise_model: Modelo de ruido a aplicar, predeterminado es None (simulación ideal).
    :return: Mide los qubits en orden lexicográfico.


    Ejemplo::

        from pyqpanda3.core import *
        from pyvqnet.qnn.pq3.measure import probs_measure
        circuit = QCircuit(3)
        circuit << H(0)
        circuit << P(2, 0.2)
        circuit << RX(1, 0.9)
        circuit << RX(0, 0.6)
        circuit << RX(1, 0.3)
        circuit << RY(1, 0.3)
        circuit << RY(2, 2.7)
        circuit << RX(0, 1.5)

        prog = QProg()
        prog.append(circuit)
        prog.append(measure(0, 0))
        prog.append(measure(1, 1))
        prog.append(measure(2, 2))

        machine = CPUQVM()

        measure_result = probs_measure(machine,prog,[2,0])
        print(measure_result)
        #[0.04796392899146941, 0, 0.4760180355042653, 0.4760180355042653]


DensityMatrixFromQstate
===========================
.. py:function:: pyvqnet.qnn.pq3.measure.DensityMatrixFromQstate(state, indices)
    
    Calcula la matriz de densidad de un estado cuántico sobre un conjunto específico de qubits.

    :param state: Lista 1D de vectores de estado. El tamaño de esta lista debe ser ``(2**N,)`` Para un número de qubits ``N``, qstate debe comenzar desde 000 -> 111.
    :param indices: Lista de índices de qubits en el subsistema considerado.
    :return: 
        Matriz de densidad de tamaño ``(2**len(indices), 2**len(indices))`` .

    Ejemplo::

        from pyvqnet.qnn.pq3.measure import DensityMatrixFromQstate
        qstate = [(0.9306699299765968+0j), (0.18865613455240968+0j), (0.1886561345524097+0j), (0.03824249173404786+0j), -0.048171819846746615j, -0.00976491131165138j, -0.23763904794287155j, -0.048171819846746615j]
        print(DensityMatrixFromQstate(qstate,[0,1]))
        # [[0.86846704+0.j 0.1870241 +0.j 0.17604699+0.j 0.03791166+0.j]
        # [0.1870241 +0.j 0.09206345+0.j 0.03791166+0.j 0.01866219+0.j]
        # [0.17604699+0.j 0.03791166+0.j 0.03568649+0.j 0.00768507+0.j]
        # [0.03791166+0.j 0.01866219+0.j 0.00768507+0.j 0.00378301+0.j]]


VN_Entropy
==============
.. py:function:: pyvqnet.qnn.pq3.measure.VN_Entropy(state, indices, base=None)
    
    Calcula la entropía de Von Neumann dado un vector de estado en una lista dada de qubits.

    .. math::

        S( \rho ) = -\text{Tr}( \rho \log ( \rho ))

    :param state: Lista 1D de vectores de estado. El tamaño de esta lista debe ser ``(2**N,)`` Para un número de qubits ``N``, qstate debe comenzar desde 000 -> 111.
    :param indices: Lista de índices de qubits en el subsistema bajo consideración.
    :param base: Base del logaritmo. Si es None, se usa el logaritmo natural.

    :return: valor de punto flotante de la entropía de von Neumann.

    Ejemplo::

        from pyvqnet.qnn.pq3.measure import VN_Entropy
        qstate = [(0.9022961387408862 + 0j), -0.06676534788028633j,
        (0.18290448232350312 + 0j), -0.3293638014158896j,
        (0.03707657410649268 + 0j), -0.06676534788028635j,
        (0.18290448232350312 + 0j), -0.013534006039561714j] 
        print(VN_Entropy(qstate, [0, 1]))
        #0.14592917648464448

Mutal_Info
==============
.. py:function:: pyvqnet.qnn.pq3.measure.Mutal_Info(state, indices0, indices1, base=None)

    Calcula la información mutua dado el vector de estado en dos listas de sub-qubits.

    .. math::

        I(A, B) = S(\rho^A) + S(\rho^B) - S(\rho^{AB})
        donde :math:`S` es la entropía de von Neumann.

    La información mutua es una medida de la correlación entre dos subsistemas. Más específicamente, cuantifica la cantidad de información que un sistema puede obtener al medir el otro.

    Cada estado se puede dar como un vector de estado en la base de computación.

    :param state: Lista 1D de vectores de estado. El tamaño de esta lista debe ser ``(2**N,)`` Para un número de qubits ``N``, qstate debe comenzar desde 000 -> 111.
    :param indices0: Lista de índices de qubits en el primer subsistema.
    :param indices1: Lista de índices de qubits en el segundo subsistema.
    :param base: Base de los logaritmos. Si es None, se usan logaritmos naturales. El valor predeterminado es None.

    :return: Información mutua entre subsistemas

    Ejemplo::

        from pyvqnet.qnn.pq3.measure import Mutal_Info
        qstate = [(0.9022961387408862 + 0j), -0.06676534788028633j,
        (0.18290448232350312 + 0j), -0.3293638014158896j,
        (0.03707657410649268 + 0j), -0.06676534788028635j,
        (0.18290448232350312 + 0j), -0.013534006039561714j]
        print(Mutal_Info(qstate, [0], [2], 2))
        #0.13763425302805887

Purity
=========================

.. py:function:: pyvqnet.qnn.pq3.measure.Purity(state, qubits_idx)

    Calcula la pureza de un qubit particular a partir del vector de estado.

    .. math::

        \gamma = \text{Tr}(\rho^2)
        
    donde :math:`\rho` es la matriz de densidad. La pureza de un estado cuántico normalizado satisface :math:`\frac{1}{d} \leq \gamma \leq 1` ,
    donde :math:`d` es la dimensión del espacio de Hilbert.
    La pureza de un estado puro es 1.

    :param state: estado cuántico obtenido de pyqpanda3
    :param qubits_idx: índice del qubit para el cual se calculará la pureza

    :return:
        pureza

    Ejemplos::

        from pyvqnet.qnn.pq3.measure import Purity
        qstate = [(0.9306699299765968 + 0j), (0.18865613455240968 + 0j),
        (0.1886561345524097 + 0j), (0.03824249173404786 + 0j),
        -0.048171819846746615j, -0.00976491131165138j, -0.23763904794287155j,
        -0.048171819846746615j]
        pp = Purity(qstate, [1])
        print(pp)
        #0.902503479761881
