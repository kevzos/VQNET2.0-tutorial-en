Perguntas Frequentes
=============================

**P: Quais são as características do VQNet?**

R: O VQNet é um conjunto de ferramentas de aprendizado de máquina quântico desenvolvido pela Origin Quantum baseado no pyQPanda. O VQNet fornece um rico conjunto de interfaces fáceis de usar para módulos de redes neurais clássicas, permitindo uma otimização conveniente do aprendizado de máquina.
O método de definição do modelo é consistente com os frameworks de aprendizado de máquina mainstream, o que reduz a curva de aprendizado para os usuários.
Ao mesmo tempo, baseado no pyQPanda, um simulador quântico de alto desempenho desenvolvido pela Origin Quantum, o VQNet também pode suportar a operação de um grande número de bits quânticos em laptops pessoais. Por fim, o VQNet também fornece ricos exemplos :doc:`./qml_demo` para sua referência e aprendizado.

**P: Como usar o VQNet para treinar modelos de aprendizado de máquina quântico?**

R: Existe um tipo de algoritmo de aprendizado de máquina quântico que constrói modelos diferenciáveis baseados em circuitos variacionais quânticos.
O VQNet pode usar o método de descida de gradiente para treinar esse tipo de modelo. As etapas gerais são as seguintes: Primeiro, no computador local, os usuários podem construir uma máquina virtual através do pyQPanda e combinar as interfaces fornecidas no VQNet para construir um modelo híbrido quântico-clássico ``Module``; segundo, chamar ``forward()`` do ``Module`` pode realizar a simulação do circuito quântico e o cálculo forward da rede neural clássica de acordo com o modo de operação definido pelo usuário;
Ao chamar ``backward()`` do ``Module``, o modelo construído pelo usuário pode ser diferenciado automaticamente como em frameworks de aprendizado de máquina clássicos como o PyTorch, e calcular os gradientes dos parâmetros nos circuitos variacionais quânticos e nas camadas de cálculo clássicas; por fim, combine a função ``step()`` do otimizador para otimizar os parâmetros.

No VQNet, usamos `parameter-shift <https://arxiv.org/abs/1803.00745>`_ para calcular o gradiente dos circuitos variacionais quânticos. Os usuários podem usar a interface sob :ref:`QuantumLayer_pq3` fornecida pelo VQNet para encapsular a diferenciação automática dos circuitos variacionais quânticos. Os usuários só precisam definir os circuitos variacionais quânticos como parâmetros em um determinado formato para construir as classes acima.

No VQNet, também podemos usar o método baseado em diferenciação automática para calcular o gradiente dos circuitos variacionais quânticos. Os usuários podem usar a interface em :ref:`vqc_api` para construir um circuito treinável. Este circuito não depende do pyQPanda, mas divide a codificação, as operações de porta e a medição no circuito em operadores diferenciáveis, para alcançar a função de calcular o gradiente dos parâmetros.

Para mais detalhes, consulte as interfaces relevantes e os códigos de exemplo neste documento.

**P: No Windows, encontrei um erro ao instalar o VQNet: "importError: DLL load failed while importing _core: The specified module could not be found."**

R: Os usuários podem precisar instalar a biblioteca de tempo de execução VC++ no Windows.
Consulte https://learn.microsoft.com/en-us/cpp/windows/latest-supported-vc-redist?view=msvc-170 para instalar a biblioteca de tempo de execução apropriada.
Além disso, o VQNet atualmente suporta apenas a versão python3.10, portanto, confirme sua versão do python.

**P: Como chamar a nuvem quântica original e o chip quântico para cálculo?**

R: Você pode usar o cluster de computação de alto desempenho da Origin Quantum ou computadores quânticos reais para simulação de circuitos quânticos, substituindo a simulação local de circuitos quânticos pela computação em nuvem.
No VQNet, os usuários podem usar ``QuantumBatchAsyncQcloudLayer`` para construir um módulo de circuito variacional quântico, inserir as CHAVES API solicitadas no site oficial da Origin e enviar a tarefa para a máquina real para execução.

**P: Por que os parâmetros do modelo que defini não são atualizados durante o treinamento?**

R: Para construir um modelo VQNet, é necessário garantir que todos os módulos utilizados sejam diferenciáveis. Quando um módulo no modelo não pode calcular o gradiente, esse módulo e os módulos anteriores não poderão calcular o gradiente usando a regra da cadeia.
Se o usuário personalizar um circuito variacional quântico, use a interface sob :ref:`QuantumLayer_pq3` fornecida pelo VQNet. Para módulos de aprendizado de máquina clássicos, você precisa usar as interfaces definidas por :doc:`./QTensor` e :doc:`./nn`. Essas interfaces encapsulam as funções de cálculo de gradiente, e o VQNet pode realizar a diferenciação automática.

Se o usuário quiser usar uma lista contendo vários módulos como submódulo em `Module`, não use a lista Python integrada. Em vez disso, use pyvqnet.nn.module.ModuleList. Dessa forma, os parâmetros de treinamento dos submódulos podem ser registrados em todo o modelo, permitindo o treinamento por diferenciação automática. Aqui está um exemplo:

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

**P: Por que o código original não funcionava na versão 2.0.7?**

R: Na versão v2.0.7, adicionamos diferentes tipos de dados e atributos dtype ao QTensor, e restringimos os formatos de entrada com base nas convenções do PyTorch. Por exemplo, a entrada da camada Embedding precisa ser kint64, e os rótulos para as camadas CategoricalCrossEntropy, CrossEntropyLoss, SoftmaxCrossEntropy e NLL_Loss precisam ser kint64.

Você pode usar a interface 'astype()' para converter o tipo para o tipo de dados especificado, ou inicializar o QTensor usando uma matriz numpy do tipo de dados correspondente.

**P: O VQNet depende do torch?**

R: O VQNet não depende do torch, nem instala o torch automaticamente.

Para usar os seguintes recursos, você precisa instalar o torch>=2.11.0 por conta própria. Desde a v2.15.0, oferecemos suporte ao uso do `torch >=2.11.0 <https://docs.pytorch.org/docs/stable/index.html>`_ como backend de computação para redes neurais clássicas, circuitos variacionais quânticos, computação distribuída, etc.
Após chamar ``pyvqnet.backends.set_backend("torch")``, a interface permanece inalterada, mas as variáveis membro ``data`` do ``QTensor`` do VQNet passam a usar ``torch.Tensor`` para armazenar dados,
e usam torch para computação. As classes sob ``pyvqnet.nn.torch`` e ``pyvqnet.qnn.vqc.torch`` herdam de ``torch.nn.Module`` e podem formar modelos ``torch``.
