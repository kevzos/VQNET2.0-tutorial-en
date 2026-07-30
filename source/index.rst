.. VQNet documentation master file, created by
   sphinx-quickstart on Tue Jul 27 15:25:07 2021.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

VQNet
=================================

Características principales de VQNet
--------------------------------------

Compatibilidad multiplataforma y soporte multi-entorno
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

VQNet permite a los usuarios realizar investigación y desarrollo de aprendizaje automático cuántico en una variedad de entornos de hardware y sistemas operativos. Ya sea usando CPU o GPU para simulación de computación cuántica, o llamando a chips cuánticos reales a través del servicio en la nube Benyuan Quantum, VQNet ofrece soporte sin interrupciones. Actualmente, VQNet es compatible con Python 3.10 en sistemas Windows, Linux y macOS.

Diseño de interfaz perfecto y facilidad de uso
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

VQNet utiliza Python como lenguaje frontal, proporciona una interfaz de funciones similar a PyTorch y permite elegir libremente entre múltiples motores de computación para implementar la función de diferenciación automática de modelos clásicos y cuánticos de aprendizaje automático. El framework incluye integradas: más de 100 interfaces de computación Tensor de uso común, más de 100 interfaces de computación de circuitos variacionales cuánticos y más de 50 interfaces de redes neuronales clásicas. Estas interfaces cubren el proceso completo de desarrollo, desde el aprendizaje automático clásico hasta el aprendizaje automático cuántico, y se actualizarán continuamente.

Rendimiento de cálculo eficiente y capacidades de expansión
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- **Soporte para experimentos con chips cuánticos reales**: Para los usuarios que necesitan experimentos con chips cuánticos reales, VQNet integra la interfaz pyQPanda original y combina las capacidades de programación eficientes de Sinan para lograr cálculos rápidos de simulación de circuitos cuánticos y operación en chips reales.
- **Optimización de cómputo local**: Para necesidades de cómputo local, VQNet proporciona una interfaz de programación de aprendizaje automático cuántico basada en CPU o GPU, y utiliza tecnología de diferenciación automática para realizar cálculos de gradiente de circuitos variacionales cuánticos, que es significativamente más rápida que los métodos de desplazamiento de parámetros. Los detalles se encuentran en :ref:`benchmarks`.
- **Soporte de cómputo distribuido**: VQNet admite cómputo distribuido basado en MPI, lo que permite entrenar modelos híbridos de redes neuronales cuántico-clásicas a gran escala en múltiples nodos.

Escenarios de aplicación ricos y soporte de ejemplos
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

VQNet no es solo una herramienta de desarrollo potente, sino que también se utiliza ampliamente en múltiples proyectos dentro de la empresa, incluidos la optimización de energía, el análisis de datos médicos, el procesamiento de imágenes y otros campos. Para ayudar a los usuarios a comenzar rápidamente, VQNet proporciona una variedad de escenarios que van desde tutoriales básicos hasta aplicaciones avanzadas en el sitio web oficial y la documentación en línea de la API. Estos recursos permiten a los usuarios comprender fácilmente cómo usar VQNet para resolver problemas prácticos y construir rápidamente sus propias aplicaciones de aprendizaje automático cuántico.

.. toctree::
    :caption: Guía de instalación
    :maxdepth: 2

    rst/install.rst

.. toctree::
    :caption: Ejemplos prácticos
    :maxdepth: 2

    rst/vqc_demo.rst
    rst/qml_demo.rst

.. toctree::
    :caption: API de red neuronal clásica
    :maxdepth: 2

    rst/QTensor.rst
    rst/nn.rst
    rst/utils.rst

.. toctree::
    :caption: API QNN integrada con pyqpanda
    :maxdepth: 2

    rst/qnn.rst
    rst/qnn_pq3.rst

.. toctree::
    :caption: API QNN con Autograd
    :maxdepth: 2

    rst/vqc.rst

.. toctree:: 
    :caption: Otros 
    :maxdepth: 2 
    
    rst/torch_api.rst
    rst/vqnet_dist.rst
    rst/FAQ.rst 
    rst/CHANGELOG.rst




