# Lokale LLM-Benchmarks auf einer Threadripper-Workstation

Messreihen zu großen Sprachmodellen unter [llama.cpp](https://github.com/ggml-org/llama.cpp)
auf eigener Hardware — mit Fokus darauf, **wo die Engpässe tatsächlich sitzen**
und welche Stellschrauben etwas bringen. Jede Zahl ist gemessen, nicht geschätzt;
wo Vorhersagen danebenlagen, steht das mit dabei.

> **Einheiten:** Alle Angaben in GB, binär gezählt — so wie `nvidia-smi`, `free`
> und die Karten selbst zählen (eine „16-GB-Karte" = 16 GB). Hugging-Face-Angaben
> sind auf diese Skala umgerechnet und liegen deshalb rund 7 % unter den dort
> genannten Zahlen.

## Hardware

| | |
|---|---|
| CPU | AMD Ryzen Threadripper PRO 3945WX — 12 Kerne / 24 Threads, 8-Kanal DDR4-3200 |
| RAM | 61 GiB (8 × 8 GB, alle Kanäle belegt) |
| GPU | 2 × NVIDIA GeForce RTX 5070 Ti — je 16 GiB, zusammen **31,8 GiB VRAM** |
| Speicher | 937 GB NVMe |
| System | Ubuntu 26.04, NVIDIA-Treiber 595.x, CUDA 13.3 |

## Die Messreihen

| Datum | Inhalt |
|---|---|
| [25.07.](FINDINGS-2026-07-25.md) | Erste Bestandsaufnahme, CPU-only, Modellvergleich, Thread-Tuning |
| [28.07.](FINDINGS-2026-07-28.md) | CUDA-Build, eine Karte, Layer-Sweep, MTP |
| [01.08.](FINDINGS-2026-08-01.md) | Zweite Karte, Laguna S 2.1 (118B), Nemotron 3 Super (120B), DeepSeek V4 Flash |
| [02.08.](FINDINGS-2026-08-02.md) | KV-Quantisierung und MTP am dichten 27B |
| [04.08.](FINDINGS-2026-08-04.md) | DeepSeek V4 Flash in IQ2_M, Layer-Split ausgereizt, spekulatives Dekodieren |
| [22.08.](FINDINGS-2026-08-22.md) | Qwen3.8-27B, KV-Cache-Quantisierung, Kosten einer freigehaltenen GPU |

Jede Datei verlinkt ihren Vorgänger und schließt mit einer Liste offener Punkte,
die in die nächste übernommen wird.

---

## Die wichtigsten Erkenntnisse

### Bei extremen Quantisierungen ist die kleinere Datei nicht die schnellere

DeepSeek V4 Flash (260B), dasselbe Modell, nur andere Quantisierung:

| | Größe | Decode | Praxislauf |
|---|---:|---:|---|
| UD-IQ1_S | 76,9 GiB | 2,8 t/s | 111 Zeilen |
| **UD-IQ2_M** | 84,7 GiB | **15,17 t/s** | **254 Zeilen** |

Unterhalb von etwa 3 Bit kostet das Entpacken der Gewichte mehr Zeit als das
Lesen der Bytes. Eine Stufe gröber ist dann **schneller und besser zugleich**.
Bandbreitenrechnungen gelten nur für Q4 und ähnliche Stufen — bei IQ1/IQ2
dominiert die Rechenzeit. Eine vorab erstellte Prognose lag deshalb um Faktor 16
daneben.

### Beim KV-Cache gilt dasselbe — mit einer scharfen Schwelle

Der KV-Cache wird bei **jedem** Token vollständig gelesen, teures Entpacken
schlägt hier also besonders durch. Zwei Messungen, die zusammen die Schwelle
zeigen.

Von f16 auf `q8_0` (Qwen3.6-27B, `llama-bench`) — **kostenlos**, halber Speicher:

| | pp512 | tg128 |
|---|---:|---:|
| KV f16, ohne Flash-Attention | 1723,58 ± 53,55 | 42,92 ± 0,03 |
| KV q8_0 + Flash-Attention | 1737,44 ± 53,69 | 42,88 ± 0,03 |

Eine Stufe tiefer kippt es (Qwen3.8-27B in Q4_K_M, beide Karten):

| | 64k Kontext | 128k Kontext |
|---|---:|---:|
| **`q8_0`** (genau) | **73,9 t/s** | **69,1 t/s** |
| **`iq4_nl`** (sparsam) | 43,5 t/s | 46,5 t/s |

**`iq4_nl` kostet 40 %. Doppelter Kontext kostet 6 %.**

Merke: **`q8_0` verwenden, solange er in den Speicher passt.** Reicht der Platz
nicht, lieber Kontext opfern als Cache-Genauigkeit — Kontext ist billig.

### Spekulatives Dekodieren zahlt nur bei bandbreitenlimitierten Modellen

| Modell | Verfahren | Wirkung |
|---|---|---|
| Qwen3.6-27B (dicht, ganz im VRAM) | MTP | **× 1,76** |
| Qwen3.6-35B-A3B (MoE, ganz im VRAM) | MTP | + 9 % |
| DeepSeek V4 Flash (MoE, teils CPU) | DSpark | **× 0,86 — langsamer** |

Der Gewinn entsteht daraus, dass das Prüfen mehrerer Token kaum mehr kostet als
eines: Die Gewichte müssen ohnehin einmal durch den Speicher. **Bei MoE gilt das
nicht.** Jeder Token wählt eigene 6 von 256 Experten, vier Token berühren bis zu
24 verschiedene Blöcke — statt einmal wird viermal entpackt. Die Trefferquote lag
mit 50,6 % gut, sie half nur nichts.

### Bei ungleichen Layern teilt `-ts` falsch

`--n-cpu-moe` macht die Layer extrem unterschiedlich schwer: die einen haben ihre
Experten im RAM, die anderen auf der GPU. `-ts` teilt aber nach **Anzahl**, nicht
nach Größe. Dazu trägt Karte 0 allein den Grundstock — Embedding, Ausgabeschicht,
Attention aller Layer, hier ~9,6 GiB.

Daraus folgt das Gegenintuitive: **Die Karte ohne Grundstock muss die meisten
Layer bekommen.** Eine gleichmäßige Aufteilung lässt entweder Karte 1 halb leer
oder sprengt sie.

| | naheliegend | ausgereizt |
|---|---:|---:|
| Karte 0 | 15637 MiB | 14455 MiB |
| Karte 1 | 8066 MiB | **15173 MiB** |
| gesamt | 23,1 GiB | **28,9 GiB** |

### Mehr Parallelität ist oft langsamer

- **Threads:** Bei 8-Kanal-DDR4 sättigen etwa 8 Kerne die Bandbreite. Ein
  80B-MoE lieferte mit 8 Threads 65 t/s, mit 12 noch 61, mit den vorgegebenen 48
  nur 24 t/s.
- **Zwei Karten:** Bei Layer-Split rechnen sie abwechselnd, nie gleichzeitig. Der
  Gewinn der zweiten Karte ist **Kapazität**, nicht Tempo — nur eben Kapazität,
  die verhindert, dass etwas über den RAM muss, und das wirkt dann stark.
- **Auslagern in den RAM:** Eine Karte für andere Aufgaben freizuhalten und die
  Gewichte teilweise in den RAM zu legen kostete **Faktor 5,5** (46,5 → 8,5 t/s).
  Die Kurve ist dabei fast flach — der RAM-Anteil beherrscht alles, einzelne
  Schichten zu verschieben bringt kaum etwas.
- **Bauen:** `-j $(nproc)` sprengt bei CUDA den Speicher; 24 parallele
  `nvcc`-Prozesse für Template-Instanzen brauchen je mehrere GiB. Der OOM-Killer
  meldet sich als `Error 137` und sieht aus wie ein Compilerfehler. `-j 8` läuft
  durch.

---

## Fallen, die Messungen unbrauchbar machen

**Ein CPU-only-Build ignoriert `-ngl` schweigend.** Kein Fehler, keine Warnung —
das Modell rechnet auf der CPU und ist ein Zehntel so schnell. Verlässliche
Kontrolle ist `nvidia-smi` während des Laufs: Steht dort 0 MiB, läuft nichts auf
der GPU. Mehrere Build-Verzeichnisse nebeneinander (`build/`, `build-cuda/`,
`build-vulkan/`) machen das zur echten Stolperfalle.

**Prefill-Werte aus `llama-cli` sind wertlos.** Am selben Modell: 111–128 t/s aus
`llama-cli` mit kurzem Prompt gegen 1723 t/s aus `llama-bench` — Faktor 14. Bei
kurzen Eingaben bestimmt der Startaufwand die Rate.

**Decode-Werte sind nur bei gleicher Ausgabelänge vergleichbar.** Identische
Konfiguration lieferte 12,25 t/s über 800 Token und 15,17 t/s über 2751 Token.
Der Kaltstart verteilt sich mit der Länge.

**Mindestens ~1,5 GiB VRAM je Karte freilassen.** Sonst scheitert `cublasCreate_v2`
mit „resource allocation failed" — und zwar erst beim ersten Token, nicht beim
Laden, weil cuBLAS seinen Arbeitsbereich erst beim ersten Matrixprodukt anlegt.

---

## Umfang ist nicht Qualität

Derselbe offene Programmierauftrag („baue ein Auto-Rennspiel") an alle Modelle:

| Modell | Umfang | lauffähig |
|---|---|---|
| Qwen3.6-27B (Q4, dicht) | mittel | ja |
| Laguna S 2.1 (Q4, MoE, 118B) | 692 Zeilen, 3 Dateien | erst nach Fix |
| DeepSeek V4 Flash (IQ1_S, 260B) | 111 Zeilen, 1 Datei | ja |
| DeepSeek V4 Flash (IQ2_M, 260B) | 254 Zeilen, 1 Datei | ja |

Das ambitionierteste Ergebnis lief nicht, das kleinste lief sofort. Und ein
260B-Modell liefert bei offener Aufgabe nichts, was ein 27B in Q4 nicht auch
hinbekommt — solange die Quantisierung der begrenzende Faktor bleibt.

---

## Was daraus für eine Aufrüstung folgt

Der Engpass ist die **CPU**, nicht der Speicher und nicht die GPU: Bei
teilausgelagerten Modellen standen zwölf Kerne am Anschlag, während die Karten
sich bei 3–5 % Auslastung langweilten und die SSD mit 22 MB/s vor sich hin las.

| Maßnahme | Bewertung |
|---|---|
| Mehr Kerne | greift genau am Engpass an |
| RAM auf 128 GB | ermöglicht höhere Quantisierungen — die sind **auch schneller**, nicht nur besser |
| dritte GPU | schwächste Option, solange der RAM-Anteil dominiert |

Die mittlere Zeile ist eine Korrektur: Zuerst stand dort „Qualität, kein Tempo".
Der Sprung IQ1 → IQ2 hat beides gebracht.
