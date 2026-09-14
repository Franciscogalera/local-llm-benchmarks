# Raylight auf zwei RTX 5070 Ti — was funktioniert und was nicht

Stand: 29.08.2026. Alle Zahlen gemessen auf dem WRX80 mit zwei RTX 5070 Ti
(je 16 GiB, beide PCIe 4.0 x16), ComfyUI mit raylight, Modell Wan 2.2 S2V 14B.

Diese Datei ist die Kurzfassung fuer den naechsten Anlauf — damit nicht wieder
von vorn probiert wird.

---

## Die drei Verfahren und wofuer sie taugen

| Verfahren | teilt auf | macht schneller | macht laengere Videos |
|---|---|---|---|
| **USP / Sequenz** (`ulysses_degree`) | die Datenkette des Videos | ja, 1,49× | ja — **das einzige** |
| **CFG** (`cfg_degree`) | die beiden Prompt-Durchgaenge | nein (gemessen: 0 %) | nein |
| **FSDP** | die Modellgewichte | nein | indirekt, macht Platz frei |
| **DP** (`dp_degree`) | ganze Bilder/Videos | mehr Videos je Stunde | nein |

Merksatz: **Fuer die Laenge eines einzelnen Videos hilft nur Sequenz-Parallelitaet**
(direkt) **oder FSDP** (indirekt, weil es Gewichte von der Karte nimmt).
CFG hilft dafuer nie — jede Karte haelt dort weiterhin die volle Datenkette.

---

## Wichtigste Falle: der Audio-Zweig bricht die Aufteilung

Sequenz-Parallelitaet zerschneidet die Datenkette des Videos. Beim S2V-Modell
haengen aber zusaetzliche Token fuer Audio und Referenzbild daran, und dadurch
zerfaellt die Kette nicht mehr sauber in Bildabschnitte:

    Input tensor shape: [1, 95040, 5120]   Additional info: {'t': 53}
    Shape mismatch, can't divide axis of length 95040 in chunks of 53
    einops.EinopsError: pattern "b (t n) c -> (b t) n c"

Beobachtet:

| Aufbau | 2 Karten, Sequenz |
|---|---|
| S2V, 480 × 832, 49 Bilder | laeuft |
| S2V, 720 × 1280, **jede** Laenge | scheitert |
| S2V, 704 × 1280, **jede** Laenge | scheitert |

**Korrektur einer falschen Faehrte:** Zuerst sah es nach einem
Teilbarkeitsproblem der Aufloesung aus (720/16 = 45, ungerade). Die Messung bei
704 (44, gerade) scheiterte aber genauso. Es liegt am Audio-Zweig, nicht an der
Kantenlaenge.

**Konsequenz: Fuer Multi-GPU den S2V-Ablauf meiden.** `WanImageToVideo` statt
`WanSoundImageToVideo` verwenden — ohne Audio zerfaellt die Kette sauber.

## Wie der Autor von raylight es einstellt

Aus seinen Beispiel-Ablaeufen (`example_workflows/`), alle identisch:

    RayInitializer:  GPU=2, ulysses_degree=2, ring_degree=1, cfg_degree=1
    Sampler:         XFuserKSamplerAdvanced
    Modell:          fp8-safetensors ueber RayUNETLoader
    FSDP:            aus
    cfg im Sampler:  1.0   bei nur 6 Schritten

Zwei Dinge daran sind entscheidend:

1. **Immer Sequenz-Parallelitaet, nie CFG.** In keinem einzigen Beispiel ist
   `cfg_degree > 1` gesetzt.
2. **`cfg = 1.0`** — er arbeitet mit destillierten Modellen, die ohne
   Prompt-Verstaerkung auskommen. Damit gibt es nur *einen* Durchgang, und
   CFG-Parallelitaet waere ohnehin sinnlos.

Wer mit `cfg = 5.0` und 20 Schritten arbeitet, ist ausserhalb dessen, wofuer
raylight gebaut wurde.

---

## Was mit dem S2V-Modell schiefging

Der Avatar-Ablauf (Wan 2.2 S2V 14B als GGUF, Bild + Audio → Video, cfg 5.0)
weicht in drei Punkten von allen Beispielen ab: GGUF statt safetensors,
cfg 5.0 statt 1.0, und ein Audio-Zweig, den kein Beispiel hat.

**Sequenz-Parallelitaet:** laeuft, 176 s statt 262 s (Faktor 1,49) — aber das
Ergebnis ist ein **anderes Gesicht**. Im Vergleichsbild bei gleichem Startwert
fehlten Brille und Handbewegung, der Hintergrund wechselte von grau auf weiss.
Fuer Avatare unbrauchbar. Ob das am S2V-Zweig, am GGUF-Format oder am
cfg-Wert liegt, ist **nicht** isoliert.

**CFG-Parallelitaet:** wirkungslos. Und der Grund ist belegt — der Sampler
prueft, ob der Datenblock zwei Eintraege hat:

    # cfg_utils.py, _validate_cfg_batch
    if tensor is None or tensor.ndim == 0 or tensor.shape[0] != cfg_world_size:
        raise ValueError("CFG = 1.0, disables guidance. ...")

Die Meldung fuehrt in die Irre: Es geht nicht um den cfg-Wert, sondern um die
Stapelgroesse. Bei diesem Modell rechnet ComfyUI die beiden Durchgaenge
**nacheinander als getrennte Aufrufe**, nicht als Paket — es gibt also nichts
zu verteilen.

**FSDP:** geht mit GGUF grundsaetzlich nicht (steht in der README). Mit der
fp8-safetensors-Fassung laedt es, aber nur mit `FSDP_CPU_OFFLOAD = True`;
ohne den Schalter bricht es schon beim Zerlegen der Gewichte mit
`OutOfMemory` ab.

---

## Stolpersteine, die Zeit gekostet haben

**`cfg_degree` am falschen Sampler.** Der gespeicherte Ablauf hatte
`cfg_degree = 2` — aber `XFuserKSamplerAdvanced` liest diesen Wert gar nicht,
er ist der Sampler fuer Sequenz-Parallelitaet. Die zweite Karte war damit nie
beteiligt, ohne dass irgendetwas darauf hingewiesen haette. Fuer CFG braucht
es `UnifiedParallelSampler`.

**`UnifiedParallelSampler` + `RandomNoise` stuerzt ab.** Er verlangt eine
Rausch-Quelle als eigenen Knoten. ComfyUIs eingebauter `RandomNoise` laesst
sich nicht an die Ray-Arbeiter uebertragen:

    ModuleNotFoundError: No module named
    '/home/.../ComfyUI/comfy_extras/nodes_custom_sampler'

Grund: ComfyUI registriert seine Zusatzknoten unter einem *Dateipfad* statt
unter einem Modulnamen, und Ray kann das Objekt beim Empfaenger nicht wieder
aufbauen. Abhilfe: `DPNoiseList` aus raylight selbst verwenden.

**Das Produkt der Grade muss der Kartenzahl entsprechen.**
`ulysses × ring × cfg = GPU`. Die README-Empfehlung, Ulysses und Ring bei FSDP
auf 0 zu setzen, gilt nur fuer den Data-Parallel-Sampler; hier fuehrt sie zu:

    ERROR, parallel product of ulysses_degree=0 x ring_degree=0 x cfg_degree=2 is 0.

**ComfyUI neu starten will Geduld.** Wird zu frueh neu gestartet, bricht die
neue Instanz mit *"Port 8188 is already in use"* ab, und wenn die alte dann
endet, laeuft gar nichts mehr. `~/comfy-start.sh` wartet korrekt auf Port und
VRAM.

**`pkill -f MUSTER` erwischt die eigene SSH-Sitzung**, wenn das Muster in der
Befehlszeile steht. Immer ueber die Prozessnummer beenden.

---

## Messwerte

Wan 2.2 S2V 14B (GGUF Q4), 480 × 832, 49 Bilder, 10 Schritte:

| Aufbau | Zeit | Gesicht erhalten |
|---|---:|---|
| 1 Karte | 262 s | — |
| 2 Karten, Sequenz | 176 s (1,49×) | **nein** |
| 2 Karten, CFG (XFuser-Sampler) | 264 s | ja (wirkungslos) |
| 2 Karten, CFG (Unified-Sampler) | 267 s | ja (wirkungslos) |

Laengengrenze bei 720 × 1280, 2 Schritte:

| Aufbau | maximale Laenge |
|---|---|
| 1 Karte | 145 Bilder = 9,1 s |
| 2 Karten, CFG | 145 Bilder = 9,1 s |
| 2 Karten, Sequenz | scheitert bei dieser Aufloesung ganz (siehe oben) |

Der Abbruch bei 145 → 177 ist echter `torch.OutOfMemoryError` bei 13,70 GiB.

---

## Offen fuer den naechsten Anlauf

* Sequenz-Parallelitaet bei **704 × 1280** ausreizen — dort greift die
  Teilbarkeit, und erst diese Messung sagt, ob zwei Karten laengere Videos
  ermoeglichen.
* Den Ablauf des Autors (`WAN_22_Raylight`) mit dem Wan 2.2 A14B nachbauen:
  zwei Sampler hintereinander (Schritte 0–3 und 3–Ende), `ulysses_degree = 2`,
  `cfg = 1`. Das ist die Kombination, fuer die raylight gebaut wurde — dort
  koennte auch die Gesichtsveraenderung ausbleiben.
* Pruefen, ob die Gesichtsveraenderung am S2V-Zweig, am GGUF-Format oder am
  hohen cfg-Wert liegt. Drei Ursachen, bisher keine isoliert.
* Datendurchsatz messen: zwei Videos gleichzeitig (`dp_degree = 2`) gegen zwei
  nacheinander. Das ist der einzige Nutzen zusaetzlicher Karten, der bisher
  ueberhaupt nicht geprueft wurde.

---

# Nachtrag 30.08./15.09.2026 — MiniMax H3 laeuft doch, und die Gesichtsfrage ist geklaert

## Der Fork von Karmabu loest den H3-Absturz

Mit dem offiziellen raylight bricht MiniMax H3 beim Laden mit FSDP ab:

    KeyError: "attribute 'adaln_t_table' already exists"

Der Fork **github.com/karmabu/raylight** behebt das. Danach laeuft H3 ueber beide
Karten mit FSDP und Sequenz-Parallelitaet. Wichtig beim Installieren: Es darf nur
**ein** `raylight` unter `custom_nodes` liegen — zwei Fassungen kollidieren, die
alte gehoert aus dem Verzeichnis heraus, nicht nur umbenannt.

| | eine Karte | zwei Karten (Fork) |
|---|---:|---:|
| Arbeitsspeicher | 48 GiB | **26–36 GiB** |
| 672 × 1216, 8,7 s, 4 Schritte | 255 s | 220 s |
| 672 × 1216, **13,7 s** | friert die Maschine ein | 420 s |

Der Zeitgewinn ist mit etwa 14 Prozent klein — und der 220-s-Lauf hatte noch keine
Referenzbilder zu verarbeiten, real sind es eher zehn. **Der eigentliche Gewinn ist
der Arbeitsspeicher.** Auf einer Karte brauchte schon der 8-Sekunden-Lauf 48 GiB,
und 15 Sekunden legten den ganzen Rechner lahm. Mit der Aufteilung sind 13,7
Sekunden kein Problem mehr.

## Die Gesichtsveraenderung lag NICHT an raylight

Unter "Offen fuer den naechsten Anlauf" stand die Frage, ob die veraenderten
Gesichter am S2V-Zweig, am GGUF-Format oder am cfg-Wert liegen. Bei H3 war die
Antwort banal und peinlich: **Im selbst gebauten Ablauf war der Eingang
`ref_images` gar nicht verdrahtet.** Der `LoadImage`-Knoten lag im Graphen herum,
war aber mit nichts verbunden. Das Modell hat sich das Gesicht jedes Mal frei
ausgedacht — nur anhand der Wortbeschreibung im Prompt, weshalb die Kleidung immer
passte und das Gesicht nie.

ComfyUI meldet einen leeren optionalen Eingang **nicht**, der Lauf geht ganz normal
durch. Mit zwei angeschlossenen Referenzbildern (wie in der offiziellen Vorlage)
bleibt das Gesicht ueber raylight erhalten: gleiche Brille, gleiche Frisur, gleiche
Kleidung.

**Merksatz: Bevor ein Verfahren beschuldigt wird, den selbst gebauten Graphen gegen
die offizielle Vorlage diffen.** Der schnellste Weg dafuer ist, beide mit `ui2api.py`
ins API-Format zu bringen und nur die verdrahteten Eingaenge je Knoten auszugeben.

## Schrittzahl entscheidet ueber den Bildausschnitt, nicht der Sampler

Bei 13,7 Sekunden faehrt H3 langsam an das Gesicht heran und wandert dabei zur
Seite, bis nur noch ein angeschnittener Kopf im Bild steht. Dagegen half nichts von
dem, was naheliegt:

* Prompt-Formulierungen ("no zoom", "stays centred") sind wirkungslos — bei
  `cfg = 1` gibt es keinen Negativ-Durchgang.
* Das Referenzbild weiter aufziehen (Motiv auf 80 %, 70 %, 58 % einer gleich
  grossen Flaeche, unten buendig, Hintergrund glatt) verschiebt nur den Startpunkt.
  Bei 6 Schritten landet das Modell **unabhaengig davon** im selben engen
  Ausschnitt.
* Der Sampler ist nicht die Ursache. `res_multistep` und `euler` faehrt beide gleich
  hart heran, sobald 6 Schritte gerechnet werden.

Der Ausloeser ist die **Schrittzahl**. Mit 4 Schritten bleibt der Ausschnitt weit
und ruhig, weil das Modell seine eigene Kamerabewegung gar nicht zu Ende ausfuehrt.
Mit 6 Schritten hat es genug Durchgaenge dafuer.

    4 Schritte: ruhiger, weiter Ausschnitt  —  weichere Haut
    6 Schritte: feine Haut und Bartstruktur  —  enge Grossaufnahme nach wenigen Sekunden

Beides zusammen gibt es nicht. Wer den weiten Ausschnitt will, nimmt 4 Schritte und
setzt das Referenzbild auf etwa 62 Prozent, damit die kleine Restbewegung nicht aus
dem Bild fuehrt.

## Weitere Fallen

**Hochskalieren macht weicher, nicht schaerfer.** `4x-UltraSharp` glaettet Haut und
Bart wachsartig; die 1080×1920-Fassung sieht im direkten Vergleich weniger
detailliert aus als das 672×1216-Original. Fuer Gesichter besser die Aufloesung
beim Rechnen erhoehen — soweit der Grafikspeicher mitspielt.

**768 × 1376 bei 13,7 Sekunden sprengt 16 GiB.** Abbruch mit
`torch.OutOfMemoryError` bei 12,11 GiB belegt und 2,82 GiB angefordert.

**Nach einem OOM geben die Ray-Arbeiter den Grafikspeicher nicht frei.** Beide
Karten blieben mit 15,8 GiB belegt, und der naechste Lauf scheiterte nach 0,2
Sekunden mit `RuntimeError: VRAM grow failed`. ComfyUI muss dann neu gestartet
werden — der Fehler sieht aus wie ein Modellproblem, ist aber nur eine Leiche.

**Ausgabedateien nach Parametern benennen.** Alle Ergebnisse heissen sonst
`gen_00001`, `gen_00002`, und hinterher weiss niemand mehr, welche Aufloesung,
Laenge oder Schrittzahl dahintersteckt. `starte.py` setzt den `filename_prefix`
jetzt aus Kartenzahl, Aufloesung, Bildzahl, Sekunden und Schritten zusammen und
loest dafuer auch verdrahtete Werte auf (ResolutionSelector, Rechenformeln,
Umschalter).
