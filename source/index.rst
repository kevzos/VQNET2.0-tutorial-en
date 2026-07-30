.. VQNet documentation master file, created by
   sphinx-quickstart on Tue Jul 27 15:25:07 2021.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

VQNet
=================================

Recursos principais do VQNet
------------------------------

Compatibilidade multiplataforma e suporte multi-ambiente
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

O VQNet permite que os usuários realizem pesquisa e desenvolvimento de aprendizado de máquina quântico em diversos ambientes de hardware e sistemas operacionais. Seja usando CPU ou GPU para simulação de computação quântica, ou chamando chips quânticos reais através do serviço de nuvem Benyuan Quantum, o VQNet oferece suporte sem interrupções. Atualmente, o VQNet é compatível com Python 3.10 nos sistemas Windows, Linux e macOS.

Design de interface perfeito e facilidade de uso
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

O VQNet utiliza Python como linguagem front-end, fornece uma interface de funções semelhante ao PyTorch e permite escolher livremente entre vários backends de computação para implementar a função de diferenciação automática de modelos clássicos e quânticos de aprendizado de máquina. O framework inclui integrados: mais de 100 interfaces de computação Tensor comumente usadas, mais de 100 interfaces de computação de circuitos variacionais quânticos e mais de 50 interfaces de redes neurais clássicas. Essas interfaces cobrem o processo completo de desenvolvimento, desde o aprendizado de máquina clássico até o aprendizado de máquina quântico, e serão continuamente atualizadas.

Desempenho de computação eficiente e capacidades de expansão
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- **Suporte para experimentos com chips quânticos reais**: Para usuários que precisam de experimentos com chips quânticos reais, o VQNet integra a interface pyQPanda original e combina as capacidades de agendamento eficientes do Sinan para realizar cálculos rápidos de simulação de circuitos quânticos e operação em chips reais.
- **Otimização de computação local**: Para necessidades de computação local, o VQNet fornece uma interface de programação de aprendizado de máquina quântico baseada em CPU ou GPU e utiliza tecnologia de diferenciação automática para realizar cálculos de gradiente de circuitos variacionais quânticos, que é significativamente mais rápida do que os métodos de deslocamento de parâmetros. Os detalhes estão disponíveis em :ref:`benchmarks`.
- **Suporte para computação distribuída**: O VQNet suporta computação distribuída baseada em MPI, que permite o treinamento de modelos híbridos de redes neurais quântico-clássicas em larga escala em vários nós.

Cenários de aplicação ricos e suporte de exemplos
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

O VQNet não é apenas uma ferramenta de desenvolvimento poderosa, mas também é amplamente utilizado em vários projetos dentro da empresa, incluindo otimização de energia, análise de dados médicos, processamento de imagens e outros campos. Para ajudar os usuários a começar rapidamente, o VQNet oferece uma variedade de cenários que vão desde tutoriais básicos até aplicações avançadas no site oficial e na documentação online da API. Esses recursos permitem que os usuários compreendam facilmente como usar o VQNet para resolver problemas práticos e construir rapidamente suas próprias aplicações de aprendizado de máquina quântico.

.. toctree::
    :caption: Installation Guide
    :maxdepth: 2

    rst/install.rst

.. toctree::
    :caption: Hands-on Examples
    :maxdepth: 2

    rst/vqc_demo.rst
    rst/qml_demo.rst

.. toctree::
    :caption: Classic neural network API
    :maxdepth: 2

    rst/QTensor.rst
    rst/nn.rst
    rst/utils.rst

.. toctree::
    :caption: QNN API integrated with pyqpanda
    :maxdepth: 2

    rst/qnn.rst
    rst/qnn_pq3.rst

.. toctree::
    :caption: Autograd QNN API
    :maxdepth: 2

    rst/vqc.rst

.. toctree:: 
    :caption: Others 
    :maxdepth: 2 
    
    rst/torch_api.rst
    rst/vqnet_dist.rst
    rst/FAQ.rst 
    rst/CHANGELOG.rst




