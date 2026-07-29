Preguntas Frecuentes
=============================

**P: ¿Cuáles son las características de VQNet?**

R: VQNet es un conjunto de herramientas de aprendizaje automático cuántico desarrollado por Origin Quantum basado en pyQPanda. VQNet proporciona un rico conjunto de interfaces fáciles de usar para módulos de redes neuronales clásicas, permitiendo una optimización conveniente del aprendizaje automático.
El método de definición del modelo es consistente con los frameworks de aprendizaje automático mainstream, lo que reduce la curva de aprendizaje para los usuarios.
Al mismo tiempo, basado en pyQPanda, un simulador cuántico de alto rendimiento desarrollado por Origin Quantum, VQNet también puede soportar la operación de un gran número de bits cuánticos en computadoras portátiles personales. Finalmente, VQNet también proporciona ricos ejemplos :doc:`./qml_demo` para su referencia y aprendizaje.

**P: ¿Cómo usar VQNet para entrenar modelos de aprendizaje automático cuántico?**

R: Existe un tipo de algoritmo de aprendizaje automático cuántico que construye modelos diferenciables basados en circuitos variacionales cuánticos.
VQNet puede usar el método de descenso de gradiente para entrenar este tipo de modelo. Los pasos generales son los siguientes: Primero, en la computadora local, los usuarios pueden construir una máquina virtual a través de pyQPanda y combinar las interfaces proporcionadas en VQNet para construir un modelo híbrido cuántico-clásico ``Module``; segundo, llamar a ``forward()`` del ``Module`` puede realizar la simulación del circuito cuántico y el cálculo forward de la red neuronal clásica según el modo de operación definido por el usuario;
Al llamar a ``backward()`` del ``Module``, el modelo construido por el usuario puede diferenciarse automáticamente como en los frameworks de aprendizaje automático clásicos como PyTorch, y calcular los gradientes de los parámetros en los circuitos variacionales cuánticos y las capas de cálculo clásicas; finalmente, combine la función ``step()`` del optimizador para optimizar los parámetros.

En VQNet, utilizamos `parameter-shift <https://arxiv.org/abs/1803.00745>`_ para calcular el gradiente de los circuitos variacionales cuánticos. Los usuarios pueden usar la interfaz bajo :ref:`QuantumLayer_pq3` proporcionada por VQNet para encapsular la diferenciación automática de los circuitos variacionales cuánticos. Los usuarios solo necesitan definir los circuitos variacionales cuánticos como parámetros en un cierto formato para construir las clases anteriores.

En VQNet, también podemos usar el método basado en diferenciación automática para calcular el gradiente de los circuitos variacionales cuánticos. Los usuarios pueden usar la interfaz en :ref:`vqc_api` para construir un circuito entrenable. Este circuito no depende de pyQPanda, sino que divide la codificación, las operaciones de puerta y la medición en el circuito en operadores diferenciables, para lograr la función de calcular el gradiente de los parámetros.

Para más detalles, consulte las interfaces relevantes y los códigos de ejemplo en este documento.

**P: En Windows, encontré un error al instalar VQNet: "importError: DLL load failed while importing _core: The specified module could not be found."**

R: Es posible que los usuarios necesiten instalar la biblioteca de tiempo de ejecución de VC++ en Windows.
Consulte https://learn.microsoft.com/en-us/cpp/windows/latest-supported-vc-redist?view=msvc-170 para instalar la biblioteca de tiempo de ejecución adecuada.
Además, VQNet actualmente solo soporta la versión python3.10, así que confirme su versión de python.

**P: ¿Cómo llamar a la nube cuántica original y al chip cuántico para el cálculo?**

R: Puede usar el clúster de computación de alto rendimiento de Origin Quantum o computadoras cuánticas reales para la simulación de circuitos cuánticos, reemplazando la simulación local de circuitos cuánticos con computación en la nube.
En VQNet, los usuarios pueden usar ``QuantumBatchAsyncQcloudLayer`` para construir un módulo de circuito variacional cuántico, ingresar las CLAVES API solicitadas en el sitio web oficial de Origin y enviar la tarea a la máquina real para su ejecución.

**P: ¿Por qué los parámetros del modelo que definí no se actualizan durante el entrenamiento?**

R: Para construir un modelo VQNet, es necesario asegurarse de que todos los módulos utilizados sean diferenciables. Cuando un módulo en el modelo no puede calcular el gradiente, ese módulo y los módulos anteriores no podrán calcular el gradiente utilizando la regla de la cadena.
Si el usuario personaliza un circuito variacional cuántico, use la interfaz bajo :ref:`QuantumLayer_pq3` proporcionada por VQNet. Para los módulos de aprendizaje automático clásicos, debe usar las interfaces definidas por :doc:`./QTensor` y :doc:`./nn`. Estas interfaces encapsulan las funciones de cálculo de gradiente, y VQNet puede realizar la diferenciación automática.

Si el usuario desea usar una lista que contiene múltiples módulos como submódulo en `Module`, no use la lista Python incorporada. En su lugar, use pyvqnet.nn.module.ModuleList. De esta manera, los parámetros de entrenamiento de los submódulos pueden registrarse en todo el modelo, permitiendo el entrenamiento por diferenciación automática. Aquí hay un ejemplo:

     Example::

         from pyvqnet. tensor import *
         from pyvqnet.nn import Module,Linear,ModuleList
         from pyvqnet.qnn import ProbsMeasure, QuantumLayer
         import pyqpanda as pq
         def pqctest(input, param, qubits, cbits, m_machine):
             circuit = pq. QCircuit()
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
             prog. insert(circuit)

             rlt_prob = ProbsMeasure([0,2],prog,m_machine,qubits)
             return rlt_prob


         class M(Module):
             def __init__(self):
                 super(M, self).__init__()
                 #Should be built using ModuleList
                 self.pqc2 = ModuleList([QuantumLayer(pqctest,3,"cpu",4,1), Linear(4,1)
                 ])
                 #Direct use of list cannot save the parameters in pqc3.
                 #self.pqc3 = [QuantumLayer(pqctest,3,"cpu",4,1), Linear(4,1)
                 #]
             def forward(self, x, *args, **kwargs):
                 y = self.pqc2[0](x) + self.pqc2[1](x)
                 return y

         mm = M()
         print(mm. state_dict(). keys())

**P: ¿Por qué el código original no funcionaba en la versión 2.0.7?**

R: En la versión v2.0.7, agregamos diferentes tipos de datos y atributos dtype a QTensor, y restringimos los formatos de entrada según las convenciones de PyTorch. Por ejemplo, la entrada de la capa Embedding debe ser kint64, y las etiquetas para las capas CategoricalCrossEntropy, CrossEntropyLoss, SoftmaxCrossEntropy y NLL_Loss deben ser kint64.

Puede usar la interfaz 'astype()' para convertir el tipo al tipo de datos especificado, o inicializar el QTensor usando una matriz numpy del tipo de datos correspondiente.

**P: ¿VQNet depende de torch?**

R: VQNet no depende de torch, ni instala torch automáticamente.

Para usar las siguientes funciones, debe instalar torch>=2.11.0 usted mismo. Desde v2.15.0, soportamos el uso de `torch >=2.11.0 <https://docs.pytorch.org/docs/stable/index.html>`_ como backend de computación para redes neuronales clásicas, circuitos variacionales cuánticos, computación distribuida, etc.
Después de llamar a ``pyvqnet.backends.set_backend("torch")``, la interfaz permanece sin cambios, pero las variables miembro ``data`` del ``QTensor`` de VQNet utilizan ``torch.Tensor`` para almacenar datos,
y usan torch para la computación. Las clases bajo ``pyvqnet.nn.torch`` y ``pyvqnet.qnn.vqc.torch`` heredan de ``torch.nn.Module`` y pueden formar modelos ``torch``.
