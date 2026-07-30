Domande Frequenti
=============================

**D: Quali sono le caratteristiche di VQNet?**

R: VQNet è un insieme di strumenti di apprendimento automatico quantistico sviluppato da Origin Quantum basato su pyQPanda. VQNet fornisce un ricco insieme di interfacce facili da usare per i moduli di reti neurali classiche, consentendo una comoda ottimizzazione dell'apprendimento automatico.
Il metodo di definizione del modello è coerente con i framework di apprendimento automatico mainstream, riducendo la curva di apprendimento per gli utenti.
Allo stesso tempo, basato su pyQPanda, un simulatore quantistico ad alte prestazioni sviluppato da Origin Quantum, VQNet può anche supportare il funzionamento di un gran numero di bit quantistici su laptop personali. Infine, VQNet fornisce anche ricchi esempi :doc:`./qml_demo` come riferimento e apprendimento.

**D: Come utilizzare VQNet per addestrare modelli di apprendimento automatico quantistico?**

R: Esiste un tipo di algoritmo di apprendimento automatico quantistico che costruisce modelli differenziabili basati su circuiti variazionali quantistici.
VQNet può utilizzare il metodo della discesa del gradiente per addestrare questo tipo di modello. I passaggi generali sono i seguenti: Innanzitutto, sul computer locale, gli utenti possono costruire una macchina virtuale tramite pyQPanda e combinare le interfacce fornite in VQNet per costruire un modello ibrido quantistico-classico ``Module``; in secondo luogo, chiamando ``forward()`` del ``Module`` è possibile eseguire la simulazione del circuito quantistico e il calcolo forward della rete neurale classica secondo la modalità operativa definita dall'utente;
Chiamando ``backward()`` del ``Module``, il modello costruito dall'utente può essere differenziato automaticamente come nei framework di apprendimento automatico classici come PyTorch, e calcolare i gradienti dei parametri nei circuiti variazionali quantistici e negli strati di calcolo classici; infine, combinare la funzione ``step()`` dell'ottimizzatore per ottimizzare i parametri.

In VQNet, utilizziamo `parameter-shift <https://arxiv.org/abs/1803.00745>`_ per calcolare il gradiente dei circuiti variazionali quantistici. Gli utenti possono utilizzare l'interfaccia sotto :ref:`QuantumLayer_pq3` fornita da VQNet per incapsulare la differenziazione automatica dei circuiti variazionali quantistici. Gli utenti devono solo definire i circuiti variazionali quantistici come parametri in un certo formato per costruire le classi sopra descritte.

In VQNet, possiamo anche utilizzare il metodo basato sulla differenziazione automatica per calcolare il gradiente dei circuiti variazionali quantistici. Gli utenti possono utilizzare l'interfaccia in :ref:`vqc_api` per costruire un circuito addestrabile. Questo circuito non dipende da pyQPanda, ma suddivide la codifica, le operazioni di gate e la misurazione nel circuito in operatori differenziabili, realizzando così la funzione di calcolo del gradiente dei parametri.

Per i dettagli, consultare le interfacce pertinenti e i codici di esempio in questo documento.

**D: In Windows, ho riscontrato un errore durante l'installazione di VQNet: "importError: DLL load failed while importing _core: The specified module could not be found."**

R: Gli utenti potrebbero dover installare la libreria runtime VC++ su Windows.
Consultare https://learn.microsoft.com/en-us/cpp/windows/latest-supported-vc-redist?view=msvc-170 per installare la libreria runtime appropriata.
Inoltre, VQNet attualmente supporta solo la versione python3.10, quindi si prega di confermare la propria versione di python.

**D: Come chiamare il cloud quantistico originale e il chip quantistico per il calcolo?**

R: È possibile utilizzare il cluster di calcolo ad alte prestazioni di Origin Quantum o computer quantistici reali per la simulazione di circuiti quantistici, sostituendo la simulazione locale con il cloud computing.
In VQNet, gli utenti possono utilizzare ``QuantumBatchAsyncQcloudLayer`` per costruire un modulo di circuito variazionale quantistico, inserire le CHIAVI API richieste sul sito web ufficiale di Origin e inviare il task alla macchina reale per l'esecuzione.

**D: Perché i parametri del modello che ho definito non vengono aggiornati durante l'addestramento?**

R: Per costruire un modello VQNet, è necessario assicurarsi che tutti i moduli utilizzati siano differenziabili. Quando un modulo nel modello non può calcolare il gradiente, quel modulo e i moduli precedenti non potranno calcolare il gradiente utilizzando la regola della catena.
Se l'utente personalizza un circuito variazionale quantistico, si prega di utilizzare l'interfaccia sotto :ref:`QuantumLayer_pq3` fornita da VQNet. Per i moduli di apprendimento automatico classici, è necessario utilizzare le interfacce definite da :doc:`./QTensor` e :doc:`./nn`. Queste interfacce incapsulano le funzioni di calcolo del gradiente e VQNet può eseguire la differenziazione automatica.

Se l'utente desidera utilizzare un elenco contenente più moduli come sottomodulo in `Module`, non utilizzare l'elenco Python integrato. Utilizzare invece pyvqnet.nn.module.ModuleList. In questo modo, i parametri di addestramento dei sottomoduli possono essere registrati nell'intero modello, consentendo l'addestramento con differenziazione automatica. Ecco un esempio:

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

**D: Perché il codice originale non funzionava nella versione 2.0.7?**

R: Nella versione v2.0.7, abbiamo aggiunto diversi tipi di dati e attributi dtype a QTensor, e limitato i formati di input in base alle convenzioni di PyTorch. Ad esempio, l'input del livello Embedding deve essere kint64 e le etichette per i livelli CategoricalCrossEntropy, CrossEntropyLoss, SoftmaxCrossEntropy e NLL_Loss devono essere kint64.

È possibile utilizzare l'interfaccia 'astype()' per convertire il tipo nel tipo di dati specificato, oppure inizializzare il QTensor utilizzando un array numpy del tipo di dati corrispondente.

**D: VQNet dipende da torch?**

R: VQNet non dipende da torch, né installa torch automaticamente.

Per utilizzare le seguenti funzionalità, è necessario installare torch>=2.11.0 autonomamente. Dalla v2.15.0, supportiamo l'uso di `torch >=2.11.0 <https://docs.pytorch.org/docs/stable/index.html>`_ come backend di calcolo per reti neurali classiche, circuiti variazionali quantistici, calcolo distribuito, ecc.
Dopo aver chiamato ``pyvqnet.backends.set_backend("torch")``, l'interfaccia rimane invariata, ma le variabili membro ``data`` del ``QTensor`` di VQNet utilizzano tutte ``torch.Tensor`` per memorizzare i dati,
e usano torch per il calcolo. Le classi sotto ``pyvqnet.nn.torch`` e ``pyvqnet.qnn.vqc.torch`` ereditano da ``torch.nn.Module`` e possono formare modelli ``torch``.
