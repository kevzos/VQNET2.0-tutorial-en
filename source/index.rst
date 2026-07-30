.. VQNet documentation master file, created by
   sphinx-quickstart on Tue Jul 27 15:25:07 2021.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

VQNet
=================================

Caratteristiche principali di VQNet
--------------------------------------

Compatibilità multipiattaforma e supporto multi-ambiente
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

VQNet consente agli utenti di condurre ricerca e sviluppo di apprendimento automatico quantistico in una varietà di ambienti hardware e sistemi operativi. Sia che si utilizzi CPU o GPU per la simulazione di calcolo quantistico, o che si richiamino chip quantistici reali tramite il servizio cloud Benyuan Quantum, VQNet offre un supporto senza interruzioni. Attualmente, VQNet è compatibile con Python 3.10 su sistemi Windows, Linux e macOS.

Design dell'interfaccia perfetto e facilità d'uso
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

VQNet utilizza Python come linguaggio front-end, fornisce un'interfaccia simile a PyTorch e consente di scegliere liberamente tra vari motori di calcolo per implementare la funzione di differenziazione automatica dei modelli classici e quantistici di apprendimento automatico. Il framework include: oltre 100 interfacce di calcolo Tensor di uso comune, oltre 100 interfacce di calcolo per circuiti variazionali quantistici e oltre 50 interfacce di reti neurali classiche. Queste interfacce coprono l'intero processo di sviluppo, dall'apprendimento automatico classico a quello quantistico, e saranno continuamente aggiornate.

Prestazioni di calcolo efficienti e capacità di espansione
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- **Supporto per esperimenti con chip quantistici reali**: Per gli utenti che necessitano di esperimenti con chip quantistici reali, VQNet integra l'interfaccia pyQPanda originale e combina le efficienti capacità di pianificazione di Sinan per realizzare calcoli rapidi di simulazione di circuiti quantistici e operazioni su chip reali.
- **Ottimizzazione del calcolo locale**: Per le esigenze di calcolo locale, VQNet fornisce un'interfaccia di programmazione per l'apprendimento automatico quantistico basata su CPU o GPU e utilizza la tecnologia di differenziazione automatica per eseguire calcoli di gradiente dei circuiti variazionali quantistici, significativamente più veloci dei metodi di parameter shift. I dettagli sono disponibili in :ref:`benchmarks`.
- **Supporto per il calcolo distribuito**: VQNet supporta il calcolo distribuito basato su MPI, che consente l'addestramento di modelli ibridi di reti neurali quantistico-classiche su larga scala su più nodi.

Scenari applicativi ricchi e supporto di esempi
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

VQNet non è solo un potente strumento di sviluppo, ma è anche ampiamente utilizzato in numerosi progetti interni all'azienda, tra cui l'ottimizzazione energetica, l'analisi di dati medici, l'elaborazione delle immagini e altri campi. Per aiutare gli utenti a iniziare rapidamente, VQNet offre una varietà di scenari che vanno dai tutorial di base alle applicazioni avanzate sul sito web ufficiale e nella documentazione online dell'API. Queste risorse consentono agli utenti di comprendere facilmente come utilizzare VQNet per risolvere problemi pratici e costruire rapidamente le proprie applicazioni di apprendimento automatico quantistico.

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




