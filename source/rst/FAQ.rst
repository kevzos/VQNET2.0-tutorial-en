Häufig gestellte Fragen
=============================

**F: Was sind die Merkmale von VQNet?**

A: VQNet ist ein Werkzeugsatz für quantenmaschinelles Lernen, der von Origin Quantum basierend auf pyQPanda entwickelt wurde. VQNet bietet eine umfangreiche Sammlung benutzerfreundlicher Schnittstellen für klassische neuronale Netzwerkmodelle und ermöglicht eine bequeme Optimierung des maschinellen Lernens.
Die Modelldefinitionsmethode ist konsistent mit gängigen Frameworks für maschinelles Lernen, was die Lernkurve für Benutzer verkürzt.
Gleichzeitig basiert VQNet auf pyQPanda, einem von Origin Quantum entwickelten Hochleistungs-Quantensimulator, und kann den Betrieb einer großen Anzahl von Quantenbits auf persönlichen Laptops unterstützen. Schließlich bietet VQNet auch umfangreiche :doc:`./qml_demo`-Beispiele als Referenz und Lernmaterial.

**F: Wie verwendet man VQNet zum Trainieren von quantenmaschinellen Lernmodellen?**

A: Es gibt eine Art von quantenmaschinellem Lernalgorithmus, der differenzierbare Modelle basierend auf Quanten-Variationsschaltkreisen erstellt.
VQNet kann die Gradientenabstiegsmethode verwenden, um diese Art von Modell zu trainieren. Die allgemeinen Schritte sind wie folgt: Zunächst können Benutzer auf dem lokalen Computer eine virtuelle Maschine über pyQPanda erstellen und die in VQNet bereitgestellten Schnittstellen kombinieren, um ein hybrides Quanten-Klassik-Modell ``Module`` zu erstellen; zweitens kann der Aufruf von ``forward()`` des ``Module`` die Quantenschaltkreis-Simulation und die klassische neuronale Netzwerk-Forward-Berechnung gemäß der benutzerdefinierten Betriebsart durchführen;
Beim Aufruf von ``backward()`` des ``Module`` kann das benutzerdefinierte Modell automatisch differenziert werden, ähnlich wie bei klassischen Frameworks wie PyTorch, und die Parametergradienten in Quanten-Variationsschaltkreisen und klassischen Berechnungsschichten berechnen; schließlich kombinieren Sie die ``step()``-Funktion des Optimierers, um die Parameter zu optimieren.

In VQNet verwenden wir `parameter-shift <https://arxiv.org/abs/1803.00745>`_, um den Gradienten von Quanten-Variationsschaltkreisen zu berechnen. Benutzer können die unter :ref:`QuantumLayer_pq3` bereitgestellte Schnittstelle von VQNet verwenden, um die automatische Differenzierung von Quanten-Variationsschaltkreisen zu kapseln. Benutzer müssen nur Quanten-Variationsschaltkreise als Parameter in einem bestimmten Format definieren, um die obigen Klassen zu erstellen.

In VQNet können wir auch die auf automatischer Differenzierung basierende Methode verwenden, um den Gradienten von Quanten-Variationsschaltkreisen zu berechnen. Benutzer können die Schnittstelle in :ref:`vqc_api` verwenden, um einen trainierbaren Schaltkreis zu erstellen. Dieser Schaltkreis ist nicht auf pyQPanda angewiesen, sondern teilt die Codierung, Gatteroperationen und Messung im Schaltkreis in differenzierbare Operatoren auf, um die Funktion der Gradientenberechnung zu realisieren.

Details entnehmen Sie bitte den entsprechenden Schnittstellen und Beispielcodes in diesem Dokument.

**F: Unter Windows erhalte ich einen Fehler bei der Installation von VQNet: "importError: DLL load failed while importing _core: The specified module could not be found."**

A: Benutzer müssen möglicherweise die VC++-Laufzeitbibliothek unter Windows installieren.
Besuchen Sie https://learn.microsoft.com/en-us/cpp/windows/latest-supported-vc-redist?view=msvc-170, um die entsprechende Laufzeitbibliothek zu installieren.
VQNet unterstützt derzeit nur Python 3.10, bitte überprüfen Sie Ihre Python-Version.

**F: Wie rufe ich die ursprüngliche Quantencloud und den Quantenchip zur Berechnung auf?**

A: Sie können den Hochleistungsrechner-Cluster von Origin Quantum oder echte Quantencomputer für die Quantenschaltkreis-Simulation nutzen und die lokale Simulation durch Cloud-Computing ersetzen.
In VQNet können Benutzer ``QuantumBatchAsyncQcloudLayer`` verwenden, um ein Modul für Variationsschaltkreise zu erstellen, die auf der offiziellen Website von Origin beantragten API-SCHLÜSSEL eingeben und die Aufgabe zur Ausführung an die echte Maschine senden.

**F: Warum werden die von mir definierten Modellparameter während des Trainings nicht aktualisiert?**

A: Um ein VQNet-Modell zu erstellen, muss sichergestellt sein, dass alle darin verwendeten Module differenzierbar sind. Wenn ein Modul im Modell den Gradienten nicht berechnen kann, können dieses Modul und die vorhergehenden Module den Gradienten nicht mit der Kettenregel berechnen.
Wenn der Benutzer einen Quanten-Variationsschaltkreis anpasst, verwenden Sie bitte die unter :ref:`QuantumLayer_pq3` bereitgestellte Schnittstelle von VQNet. Für klassische Module des maschinellen Lernens müssen Sie die Schnittstellen verwenden, die von :doc:`./QTensor` und :doc:`./nn` definiert werden. Diese Schnittstellen kapseln die Funktionen der Gradientenberechnung, und VQNet kann die automatische Differenzierung durchführen.

Wenn der Benutzer eine Liste mit mehreren Modulen als Untermodul in `Module` verwenden möchte, verwenden Sie bitte nicht die integrierte Python-Liste. Verwenden Sie stattdessen pyvqnet.nn.module.ModuleList. Auf diese Weise können die Trainingsparameter der Untermodelle im gesamten Modell registriert werden, was ein automatisches Differenzierungstraining ermöglicht. Hier ist ein Beispiel:

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

**F: Warum funktionierte der ursprüngliche Code in Version 2.0.7 nicht?**

A: In Version v2.0.7 haben wir verschiedene Datentypen und dtype-Attribute zu QTensor hinzugefügt und die Eingabeformate gemäß den PyTorch-Konventionen eingeschränkt. Beispielsweise muss die Eingabe der Embedding-Schicht vom Typ kint64 sein, und die Labels für die Schichten CategoricalCrossEntropy, CrossEntropyLoss, SoftmaxCrossEntropy und NLL_Loss müssen vom Typ kint64 sein.

Sie können die 'astype()'-Schnittstelle verwenden, um den Typ in den angegebenen Datentyp zu konvertieren, oder den QTensor mit einem numpy-Array des entsprechenden Datentyps initialisieren.

**F: Hängt VQNet von torch ab?**

A: VQNet hängt nicht von torch ab und installiert torch nicht automatisch.

Um die folgenden Funktionen zu nutzen, müssen Sie torch>=2.11.0 selbst installieren. Seit v2.15.0 unterstützen wir die Verwendung von `torch >=2.11.0 <https://docs.pytorch.org/docs/stable/index.html>`_ als Rechen-Backend für klassische neuronale Netzwerke, Quanten-Variationsschaltkreise, verteiltes Rechnen usw.
Nach dem Aufruf von ``pyvqnet.backends.set_backend("torch")`` bleibt die Schnittstelle unverändert, aber die ``data``-Mitgliedsvariablen des ``QTensor`` von VQNet verwenden alle ``torch.Tensor`` zur Datenspeicherung
und verwenden torch für die Berechnung. Die Klassen unter ``pyvqnet.nn.torch`` und ``pyvqnet.qnn.vqc.torch`` erben von ``torch.nn.Module`` und können ``torch``-Modelle bilden.
