# Genesis Dog Walking

Reinforcement Learning für den vierbeinigen **Unitree Go2** im Genesis-Simulator.
Eine mit PPO trainierte Policy steuert zwölf Gelenke und soll vorgegebene
Vorwärtsgeschwindigkeiten auf unebenem Terrain einhalten. Das aktive Projekt
liegt in [`third_prototype/`](third_prototype/); `old/` enthält lokale frühere Versuche.

## Projektaufbau

| Datei / Verzeichnis | Aufgabe |
| --- | --- |
| [`third_prototype/runner.py`](third_prototype/runner.py) | Training starten, Run-Verzeichnis anlegen, Checkpoints und finale Videos erzeugen |
| [`third_prototype/config.py`](third_prototype/config.py) | PPO, Reward, Curriculum, Seeds und Umgebung konfigurieren |
| [`third_prototype/make_environment.py`](third_prototype/make_environment.py) | Genesis-Szene, Go2, Observations, Gelenksteuerung und Episodenabbrüche |
| [`third_prototype/reward.py`](third_prototype/reward.py) | Geschwindigkeits-Tracking und Strafen für unerwünschtes Verhalten |
| [`third_prototype/rsl_rl/`](third_prototype/rsl_rl/) | Im Repository enthaltene, angepasste RL-Implementierung |
| [`third_prototype/eval_video.py`](third_prototype/eval_video.py) | Einen vorhandenen Checkpoint als Videos auswerten |
| [`third_prototype/speed_eval.py`](third_prototype/speed_eval.py) | Geschwindigkeitsmetriken eines Checkpoints in der Konsole ausgeben |
| [`third_prototype/train_script.sh`](third_prototype/train_script.sh) | Training mit der lokalen Conda-/Slurm-Konfiguration |
| `third_prototype/logs/` | TensorBoard, Checkpoints, W&B-Metadaten und Slurm-Ausgaben |
| `third_prototype/video/` | Finale Videos pro Run bzw. separater Videoauswertung |
| [`docs/runs/`](docs/runs/) | Auswertungen und reproduzierbare Analyseartefakte |

Logs, Checkpoints, Videos und `old/` sind per `.gitignore` ausgeschlossen und
stehen nach einem frischen Clone nicht automatisch zur Verfügung.

## Umgebung einrichten

Der für A21 protokollierte Lauf verwendete Linux, Python 3.10.20 und eine
NVIDIA RTX 4090. Der vorhandene Umgebungsexport
[`environment_test.yml`](environment_test.yml) enthält unter anderem Genesis
0.4.0, PyTorch 2.10.0, TensorDict 0.12.1 und TensorBoard 2.20.0.

Vom Repository-Hauptverzeichnis aus:

```bash
conda env create --name test --file environment_test.yml
conda activate test

# Rendering ohne sichtbares Fenster, wie in den vorhandenen Startskripten:
export PYOPENGL_PLATFORM=egl
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

python -c "import torch; print('CUDA verfügbar:', torch.cuda.is_available())"
```

Der Export ist ein Linux-Umgebungssnapshot mit einem lokalen `prefix` am
Dateiende; für eine andere Installation diesen gegebenenfalls entfernen bzw.
anpassen. Eine Neuinstallation wurde im Rahmen der A21-Auswertung nicht getestet.
Genesis benötigt auch für Videorendering ohne Fenster passende Grafikbibliotheken.
Der Code besitzt einen CPU-Fallback; das Standardtraining mit 4.096 parallelen
Umgebungen ist für eine GPU ausgelegt.

[`requirements.txt`](requirements.txt), [`requirements_2.txt`](requirements_2.txt)
und [`conda-env.yml`](conda-env.yml) sind zusätzliche ältere Abhängigkeitslisten.
Sie sind keine gleichwertigen Reproduktionen der `test`-Umgebung; insbesondere
fehlt dort die explizite TensorDict-Abhängigkeit. Die lokale Version unter
`third_prototype/rsl_rl/` wird direkt importiert.

## Training starten

```bash
conda activate test
export PYOPENGL_PLATFORM=egl
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

# W&B wird im Runner immer initialisiert. Offline bleiben alle Daten lokal.
export WANDB_MODE=offline
python third_prototype/runner.py --config_name config_A
```

Für Online-Tracking bei Weights & Biases stattdessen `WANDB_MODE=online` setzen
und vorher `wandb login` ausführen. Das Projekt heißt `Genesis_Dog_Walking`;
TensorBoard-Dateien werden zusätzlich geschrieben.

Jeder Start reserviert die nächste freie Version, beispielsweise
`third_prototype/logs/go2_genesis_rsl_rl_config_A_v022/`, sofern `v001` bis `v021`
bereits existieren. Auch eine Fortsetzung erhält einen eigenen Run-Ordner.

### Training ab Checkpoint fortsetzen

`train_script.sh` ist aktuell auf B36 ab `model_6999.pt` eingestellt und trainiert
2.000 zusätzliche Iterationen: **7000 bis 8999**, final `model_8999.pt` im neuen
Run-Ordner. Vorher lässt sich der Plan ohne Simulator prüfen:

```bash
bash third_prototype/train_script.sh --dry-run
```

Der entsprechende direkte Aufruf aus dem Repository-Hauptverzeichnis:

```bash
python third_prototype/runner.py \
  --resume-from third_prototype/logs/go2_genesis_rsl_rl_config_B_v036/model_6999.pt \
  --iterations 2000 --video-interval 0
```

Bei Resume bedeutet `--iterations` immer **zusätzliche** Iterationen und ist
verpflichtend. Actor, Critic, Observation-Normalisierung, Adam-Zustand und
Lernrate werden geladen. Der nächste Update verwendet die Iteration nach dem
Checkpoint; 6999 wird nicht wiederholt. Gespeicherte Rewards, Kommandobereiche
und PPO-Einstellungen kommen aus der zum Run gehörenden W&B-Konfiguration.
Fehlt dieser Snapshot, kann er mit `--saved-config PFAD_ZUR_CONFIG_YAML`
ausdrücklich angegeben werden. Änderungen an `config.py` überschreiben die
Resume-Konfiguration nicht.

Die neue Run-Version enthält `resume.json` mit Herkunft und Iterationsbereich.
TensorBoard setzt mit Iteration 7000 fort, W&B verwendet einen eigenen Run.
Simulatorzustand, laufende Episoden und damalige Zufallszustände sind im alten
Checkpoint nicht enthalten und starten neu; es ist keine bitgenaue Fortsetzung.
Ein aktives Curriculum wird deshalb für Resume derzeit abgewiesen, weil dessen
Fortschrittszustand ebenfalls fehlt. B36 hat kein aktives Curriculum.

Für andere zusätzliche Laufzeiten z.B.:

```bash
ADDITIONAL_ITERATIONS=1000 sbatch third_prototype/train_script.sh
```

Ein anderer Startcheckpoint kann mit `RESUME_CHECKPOINT=/ABSOLUTER/PFAD/model_N.pt`
gewählt werden. Weitere Runner-Optionen werden vom Shell-Skript weitergereicht,
etwa `--skip-final-eval` oder `--iterations 1000`. Ohne `--resume-from` startet
`runner.py --config_name config_B` weiterhin ein neues Training.

### Aktuelle Konfiguration

Die folgenden Werte beschreiben den ausführbaren Stand von `config.py`, nicht
die teilweise älteren Kommentare darin:

| Einstellung | Wert |
| --- | --- |
| Lernverfahren | PPO; Actor und Critic mit jeweils 512 / 256 / 128 Neuronen, ELU, Observation-Normalisierung |
| Training | aktuell 5.000 Iterationen, 4.096 Umgebungen, 96 Schritte pro Iteration |
| Episode / Steuerung | maximal 30 s, 50 Hz, zwei Simulations-Substeps |
| Zustand / Aktion | 50 Observations, zwölf Gelenkaktionen; auf ±1 begrenzt, Positionsoffset ±0,25 rad |
| Vorwärtskommando | aktuell fest 0,4 m/s; seitliches und Drehkommando jeweils 0 |
| Terrain | fraktales Terrain, `use_terrain=True` |
| Lernrate / Entropiekoeffizient | fest `3e-4` / `0.002` |
| Policy-Standardabweichung | lernbar, initial 0,5; Grenzen 0,05–0,75 |
| Curriculum | ausgeschaltet |
| Fortschritts-Reward | signierte Geschwindigkeit in Welt-X-Richtung; Gewicht 0,5, aktiv bei Soll > 0,1 m/s |
| Checkpoints | alle 100 Iterationen und am Ende; bei 5.000 Iterationen `model_4999.pt` |


### Slurm

`train_script.sh` aktiviert `~/miniconda3/envs/test` und setzt aktuell B39 ab
Checkpoint 6999 fort. Conda-Pfad, Projektpfad, Partition und GPU-Anforderung
bei Bedarf an die eigene Maschine anpassen. Vom Repository-Hauptverzeichnis:

```bash
mkdir -p third_prototype/logs/slurm
WANDB_MODE=offline sbatch third_prototype/train_script.sh
```

Slurm schreibt nach `third_prototype/logs/slurm/<job-id>.out` über den im Skript
hinterlegten absoluten Pfad. Die abschließenden Runner-Videos verwenden den
neuen Run und dessen finalen Checkpoint. Das frühere zusätzliche Eval mit dem
fest eingetragenen alten B36-Checkpoint ist aus dem Startskript entfernt.
Die historische [`SWEEP.md`](third_prototype/SWEEP.md) beschreibt einen anderen
Stand des Startskripts.

## Training beobachten

```bash
tensorboard --logdir third_prototype/logs
```

| TensorBoard-Metrik | Bedeutung |
| --- | --- |
| `Train/mean_reward` | mittlerer Gesamt-Reward der letzten bis zu 4.096 abgeschlossenen Episoden |
| `Train/mean_episode_length` | mittlere Episodenlänge in Schritten; mit 0,02 multiplizieren für Sekunden |
| `Tracking/lin_vel_x_score` | mittleres `exp(-(Soll-vx − Ist-vx)² / 0.1)` der Trainingsrollouts; keine Fehlerangabe in m/s |
| `Policy/mean_std` | mittlere Standardabweichung der Aktionsverteilung |
| `Loss/value` | Fehler des Value-Netzes; gemeinsam mit Reward und Episodendauer beurteilen |
| `Gait/heading_error_abs_mean` | mittlere absolute Richtungsabweichung in Radiant |
| `Gait/diagonal_support` | Anteil der Umgebungsschritte mit genau zwei diagonal aufsetzenden Füßen |
| `Gait/rear_both_air` | Anteil mit beiden Hinterfüßen ohne Kontakt |
| `Gait/undesired_contact_fraction` | Anteil mit unerwünschtem Körperkontakt |

Die Trainingsvideos unter `logs/<run>/videos/` zeigen alle 200 Iterationen
einen stochastischen Rollout von nur 96 Schritten (ca. 1,92 s).
Die abschließenden Videos unter `video/<run>/` verwenden deterministische
Aktionen, feste Kommandos in 0,1-m/s-Schritten und höchstens 800 Schritte
(ca. 16 s). Sie enden bei einem Episodenabbruch vorzeitig.

## Checkpoint auswerten

Die Beispiele setzen einen lokal vorhandenen A21-Checkpoint voraus:

```bash
python third_prototype/eval_video.py \
  --config_name config_A \
  --model_path third_prototype/logs/go2_genesis_rsl_rl_config_A_v021/model_3999.pt

python third_prototype/speed_eval.py \
  --config_name config_A \
  --model_path third_prototype/logs/go2_genesis_rsl_rl_config_A_v021/model_3999.pt
```

`eval_video.py` schreibt nach `third_prototype/video/EVAL_VIDEO_vNNN/`.
`speed_eval.py` misst pro Kommando eine deterministische Episode über höchstens
1.000 Schritte (20 s), mit 100 Schritten Anlaufzeit für Geschwindigkeitsmittel.
Es berichtet Netto-Weg in Welt-X, getrennten Vorwärts- und Rückwärtsweg,
lokale Geschwindigkeit und Fehler, Yaw-Rate, Kontaktanteile je Roboterglied,
Reward und den Abbruchgrund. Die Werte des letzten Schritts werden **vor** dem
automatischen Reset erfasst; nach einem Abbruch endet die Messung. Ein einzelner
Versuch pro Kommando ist keine statistische Erfolgsrate. Die Diagnose meldet
sowohl Welt-Z als auch die interpolierte Höhe über dem lokalen Terrain.

Für feste Kommandos und eine JSON-Datei, etwa mit dem A27-Checkpoint:

```bash
python third_prototype/speed_eval.py \
  --model_path third_prototype/logs/go2_genesis_rsl_rl_config_A_v027/model_4999.pt \
  --commands 0.4 0.8 --json-out /tmp/a27_diagnostics.json
```

Beide Skripte laden die **aktuelle** Konfiguration aus `config.py`, nicht
automatisch die gespeicherte Run-Konfiguration. Für ältere Checkpoints müssen
Konfiguration und Umgebung zum damaligen Training passen. Beide verwenden derzeit deterministische Aktionen.
Die Shell-Wrapper `eval_script.sh` und `speed_eval.sh` enthalten fest eingetragene
Pfade zu älteren Runs; die Python-Aufrufe oben wählen A21 ausdrücklich aus.
