.. VQNet documentation master file, created by
   sphinx-quickstart on Tue Jul 27 15:25:07 2021.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

VQNet
=================================

Hauptfunktionen von VQNet
--------------------------

Multiplattform-Kompatibilität und Umgebungsunterstützung
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

VQNet ermöglicht es Benutzern, Forschung und Entwicklung im Bereich des quantenmaschinellen Lernens in einer Vielzahl von Hardware- und Betriebssystemumgebungen durchzuführen. Ob mit CPU oder GPU für die Quantencomputersimulation oder durch Aufruf echter Quantenchips über den Benyuan Quantum Cloud Service – VQNet bietet nahtlose Unterstützung. Derzeit ist VQNet mit Python 3.10 unter Windows, Linux und macOS kompatibel.

Perfektes Schnittstellendesign und Benutzerfreundlichkeit
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

VQNet verwendet Python als Frontend-Sprache, bietet eine Funktionenschnittstelle ähnlich wie PyTorch und ermöglicht die freie Auswahl verschiedener Rechen-Backends zur Implementierung der automatischen Differenzierung klassischer und quantenmaschineller Lernmodelle. Das Framework enthält integriert: über 100 häufig verwendete Tensor-Computing-Schnittstellen, über 100 Quanten-Variationsschaltkreis-Schnittstellen und über 50 klassische neuronale Netzwerkschnittstellen. Diese Schnittstellen decken den gesamten Entwicklungsprozess vom klassischen maschinellen Lernen bis zum quantenmaschinellen Lernen ab und werden kontinuierlich aktualisiert.

Effiziente Rechenleistung und Erweiterungsmöglichkeiten
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- **Unterstützung für echte Quantenchip-Experimente**: Für Benutzer, die echte Quantenchip-Experimente benötigen, integriert VQNet die ursprüngliche pyQPanda-Schnittstelle und kombiniert die effizienten Planungsfähigkeiten von Sinan, um schnelle Quantenschaltkreis-Simulationsberechnungen und den Betrieb auf echten Chips zu ermöglichen.
- **Lokale Rechenoptimierung**: Für lokale Rechenanforderungen bietet VQNet eine Programmier-schnittstelle für quantenmaschinelles Lernen basierend auf CPU oder GPU und verwendet automatische Differenzierungstechnologie zur Berechnung von Gradienten in Quanten-Variationsschaltkreisen, was deutlich schneller ist als Parameter-Shift-Methoden. Details finden Sie unter :ref:`benchmarks`.
- **Unterstützung für verteiltes Rechnen**: VQNet unterstützt MPI-basiertes verteiltes Rechnen, wodurch das Training großer hybrider Quanten-Klassik-Neuronalnetzmodelle auf mehreren Knoten realisiert werden kann.

Reichhaltige Anwendungsszenarien und Beispielunterstützung
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

VQNet ist nicht nur ein leistungsstarkes Entwicklungstool, sondern wird auch in zahlreichen unternehmensinternen Projekten eingesetzt, darunter Energieoptimierung, medizinische Datenanalyse, Bildverarbeitung und andere Bereiche. Um Benutzern den schnellen Einstieg zu erleichtern, bietet VQNet auf der offiziellen Website und in der API-Online-Dokumentation eine Vielzahl von Szenarien, die von grundlegenden Tutorials bis zu fortgeschrittenen Anwendungen reichen. Diese Ressourcen ermöglichen es Benutzern, leicht zu verstehen, wie VQNet zur Lösung praktischer Probleme eingesetzt werden kann, und schnell eigene Anwendungen für quantenmaschinelles Lernen zu erstellen.

.. toctree::
    :caption: Installationsanleitung
    :maxdepth: 2

    rst/install.rst

.. toctree::
    :caption: Praxisbeispiele
    :maxdepth: 2

    rst/vqc_demo.rst
    rst/qml_demo.rst

.. toctree::
    :caption: API für klassische neuronale Netze
    :maxdepth: 2

    rst/QTensor.rst
    rst/nn.rst
    rst/utils.rst

.. toctree::
    :caption: QNN-API integriert mit pyqpanda
    :maxdepth: 2

    rst/qnn.rst
    rst/qnn_pq3.rst

.. toctree::
    :caption: Autograd QNN-API
    :maxdepth: 2

    rst/vqc.rst

.. toctree:: 
    :caption: Sonstiges 
    :maxdepth: 2 
    
    rst/torch_api.rst
    rst/vqnet_dist.rst
    rst/FAQ.rst 
    rst/CHANGELOG.rst




