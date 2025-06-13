# Pattern Recognition in Audio-Signals

Welcome to **Pattern Recognition in Audio-Signals**, a modular CLI-based Python project for exploring audio perception, acoustics, and sound synthesis. Created as part of the course *Pattern Recognition in Audio Signals* taught by **Kilian Schuster**, the **Master of Sound**, at **Lucerne University of Applied Sciences and Arts**.
This work was developed for the **Master of Science in Applied Information and Data Science**.

---

## Overview

This project is a structured submission platform for weekly exercises based on lecture content.
You can explore each week’s submission via a command-line interface, including interactive demos, audio recordings, analysis, and more.

---

## Prerequisites

Before you begin, make sure you have the following installed on your system:

### 1. Python ≥ 3.10

* **macOS (with Homebrew)**

  1. If you don’t have Homebrew yet, install it by running:
     '''bash
     /bin/bash -c "\$(curl -fsSL [https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh](https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh))"
     '''
  2. Install Python 3.10 (or later):
     '''bash
     brew install python\@3.10
     '''
  3. Verify:
     '''bash
     python3 --version
     '''

* **Ubuntu / Debian**
  '''bash
  sudo apt update
  sudo apt install python3.10 python3.10-venv python3-pip
  '''

* **Windows**

  1. Download the latest installer from [python.org](https://www.python.org/downloads/windows/).
  2. Run the installer and **check “Add Python to PATH”**.
  3. Verify in PowerShell or CMD:
     '''powershell
     python --version
     '''

### 2. Homebrew (macOS only)

If you skipped this above, install Homebrew first:
'''bash
/bin/bash -c "\$(curl -fsSL [https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh](https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh))"
'''

### 3. PortAudio (for PyAudio)

* **macOS**
  '''bash
  brew install portaudio
  '''
* **Ubuntu / Debian**
  '''bash
  sudo apt install portaudio19-dev
  '''
* **Windows**
  PyAudio wheels usually bundle PortAudio; if you run into issues, you can install Visual Studio Build Tools and then:
  '''powershell
  pip install pipwin
  pipwin install pyaudio
  '''

---

## Quick Start

1. **Clone the repository**
   '''bash
   git clone [https://github.com/dominik117/audio-signals.git](https://github.com/dominik117/audio-signals.git)
   cd audio-signals
   '''

2. **Install Poetry** (if you haven't)
   '''bash
   curl -sSL [https://install.python-poetry.org](https://install.python-poetry.org) | python3 -
   '''
   Make sure it’s in your path:
   '''bash
   poetry --version
   '''

3. **Install project dependencies**
   '''bash
   poetry install
   '''

4. **Run the CLI**
   '''bash
   poetry run python main.py
   '''
   You’ll be greeted with an interactive menu where you can explore each week’s submission, play audio, read notes, or generate synthesized sounds.

---

## Project Structure

* **`main.py`** – Entry point with CLI interface
* **`submissions/`** – Weekly submissions organized by folder (e.g., `week01/`, `week02/`, …)
* **`utils/`** – Helper functions (audio playback, file loading, menu navigation)
* **`pyproject.toml`**, **`poetry.lock`** – Dependency and environment configuration

---

## Audio Compatibility Notes

* This project uses **PyAudio**, which requires **PortAudio** (see Prerequisites).
* On Windows, if `pip install pyaudio` fails, use `pipwin` as shown above.

---

## Acknowledgments

Big thanks to **Kilian Schuster**, our **Master of Sound**, for teaching this excellent course and for the sonic inspiration.

---

Enjoy exploring the world of audio pattern recognition!
