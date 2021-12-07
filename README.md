# Achterbahnsimulation
Projekt des Kurses "Computergraphik" an der HAW im SS 21.

Python 3.7;
IDE: PyCharm 2021.1.3;
Windows 10

Abhängigkeiten: 
- argparse >= 1.4.0 
- numpy >= 1.20.3
- matplotlib >= 3.4.3


## Ziel des Projektes ist die Simulation einer Achterbahn mittels Bezier-Interpolation und Visualisierung differentialgeometrischer Größen.

Das Projekt besteht aus zwei Teilen: 

1. Erstellen und samplen eines Torus-Knoten nach individuell einstellbaren Parametern, und
2. Berechnung und Visualisierung einer Achterbahn durch Bezierkurven, welche entlang der (oder anderen) Samples laufen sollen.

Um das Programm zu nutzen, kann es z.B. als Projekt in Pycharm geöffnet und dort ausgeführt werden. Alternativ können die Befehle (nachfolgend erklärt) über die Konsole ausgeführt werden. 

### 1. Torus-Knoten Sampling

Das Torus-Knoten Sampling geschieht mit der `TorusKnotGeneration.py`. Nach erfolgreicher Ausführung wird eine .csv-Datei mit dreidimensionalen Samples dieses Knotens erstellt und im selben Ordner unter einem passenden Namen gespeichert.

Um einen Torus-Knoten zu definieren, muss ein p (`-p`) und ein q (`-q`) definiert werden. Diese Parameter müssen ganzzahlig und teilerfrei sein, mehr Informationen dazu hier: https://de.wikipedia.org/wiki/Torusknoten. Möglichkeiten wären also beispielsweise 2, 3 oder 3, 8.
Für das Sampling muss dann nur noch die Anzahl der Samples mit `-nSamples` (ganzzahlig) angegeben werden, also z.B. 20.

In der Console wäre also eine Ausführung mit `python TorusKnotGeneration.py -p 3 -q 8 -nSamples 20` möglich. In Pycharm können diese Parameter in den zum Skript gehörigen Run/Debug Configurations unter "Parameters" eingetragen werden. Die erstellte Datei trägt dann den Namen "TorusSamples_3_8_20.csv".

### 2. Achterbahn Visualisierung

Die Achterbahn wird mit `RollerCoasterVisualization.py` visualisiert, dafür braucht es Samples. Hierfür werden .csv- und .trk-Dateien unterstützt. 
Entweder können die Torus-Samples aus der `TorusKnotGeneration.py`, eine eigene .csv-Datei oder eine .trk-Datei genutzt werden. 

Die .trk- als auch die .csv-Datei muss jeweils 3 Werte (Koordinaten) in einer Zeile enthalten. Eine .trk-Datei muss mit dem keyword "Track" anfangen, die nachfolgende Zeile sollte dann leer sein oder z.B. einen Kommentar oder die Anzahl der Samples enthalten (diese zweite Zeile wird ignoriert) und darauffolgend enthält jede Zeile die drei Werte (Koordinaten). Kommentare mit `#` beginnend werden ebenfalls ignoriert.

Der Pfad soll mit dem Parameter `-filepath` angegeben werden, entweder lokal oder z.B. so: `-filepath C:/Users/UserX/Desktop/trk_dateien/_WildeMaus2.trk`
Wie bei dem Torus-Sampling kann das ganz einfach über die Konsole oder innheralb von z.B. Pycharm (siehe oben) passieren. 
In der Console könnte das Skript also beispielweise mit `python RollerCoasterVisualization.py -filepath TorusSamples_3_8_20.csv` ausgeführt werden.

Es resultiert ein Fenster mit einer 3d-Ansicht und Animation der Achterbahn-Strecke. Auf der linken Seite des Fensters sind einige geometrische Größen in Graphen veranschaulticht (Größen sind beschriftet), während sich auf der rechten Seite die 3d-Ansicht befindet (interagierbar mit der Maus), wobei der Wagon der Achterbahn auf der Strecke fährt und sich die Krümmung der Strecke zum jeweiligen Zeitpunkt in der Farbe des Wagons wiederspiegelt. 



