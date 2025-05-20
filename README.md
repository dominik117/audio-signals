# Pattern Recognition in Audio-Signals

Welcome to **Pattern Recognition in Audio-Signals**, a modular CLI-based Python project for exploring audio perception, acoustics, and sound synthesis. Created as part of the course *Pattern Recognition in Audio Signals* taught by **Kilian Schuster**, the **Master of Sound**, at **Lucerne University of Applied Sciences and Arts**.

This work was developed as part of the **Master of Science in Applied Information and Data Science**.

---

## Overview

This project is a structured submission platform for weekly exercises based on lecture content.

You can explore each week’s submission via a command-line interface, including interactive demos, audio recordings, analysis, and more.

---

## Quick Start

### 1. **Clone the repository**

```bash
git clone [https://github.com/dominik117/audio-signals.git](https://github.com/dominik117/audio-signals.git)
cd audio-signals
```

### 2. **Install Poetry (if you haven't)**

```bash
curl -sSL [https://install.python-poetry.org](https://install.python-poetry.org) | python3 -
```

Make sure it's in your path:

```bash
poetry --version
```

### 3. **Install dependencies**

```bash
poetry install
```

### 4. **Run the CLI**

```bash
poetry run python main.py
```

You'll be greeted with an interactive menu where you can explore each week's submission, play audio, read notes, or generate synthesized sounds.

---

## Audio Compatibility Notes

* This project uses **PyAudio**, which requires **PortAudio**.

### macOS

```bash
brew install portaudio
```

### Ubuntu/Debian

```bash
sudo apt install portaudio19-dev
```

---

## Project Structure

* `main.py` – Entry point with CLI interface
* `submissions/` – Weekly submissions organized by folder (e.g., `week01/`)
* `utils/` – Helper functions (audio playback, file loading, menu navigation)
* `poetry.lock`, `pyproject.toml` – Dependency and environment configuration


---

## Acknowledgments

Big thanks to **Kilian Schuster** for teaching this excellent course and for the sonic inspiration. 
