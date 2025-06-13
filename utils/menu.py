import os
from utils.loader import show_text_file, open_link
from utils.audio import play_audio, load_wav
from submissions.week01 import playground
from submissions.week02 import analysis 
from submissions.week03 import morse
from submissions.week04 import print_summary


def show_main_menu():
    while True:
        print("\n=== Dominik Bacher – Audio Submissions ===")
        print("1. Week 1 – Introduction to Acoustics")
        print("2. Week 2 – Signal Analysis")
        print("3. Week 3 – Hidden Markov Models")
        print("4. Week 4 – Neural Networks")
        print("5. Exit")
        choice = input("> ")

        if choice == "1":
            show_week01_menu()
        elif choice == "2":
            run_week02()
        elif choice == "3":
            show_week03_menu()
        elif choice == "4":
            show_week04_menu()
        elif choice == "5":
            print("Goodbye!")
            break
        else:
            print("Invalid input. Try again.")

def show_week01_menu():
    while True:
        print("\n-- Week 1: Introduction to Acoustics --")
        print("1. Sine Wave Playground")
        print("2. Play provided environment sample recording (may break the CLI)")
        print("3. Read description of acoustic features from the environment sample")
        print("4. View A4 drawing illustrating the acoustic scene")
        print("5. View article insights")
        print("6. Play custom recording (linked to Exercise 2)")
        print("7. Back")

        choice = input("> ")
        base = "submissions/week01"

        if choice == "1":
            playground.run_playground()
        elif choice == "2":
            print("\nIf the program crashes, restart with: poetry run python main.py")
            play_audio(os.path.join(base, "Track-5.wav"))
        elif choice == "3":
            show_text_file(os.path.join(base, "description.txt"))
        elif choice == "4":
            print("View the image at: https://ibb.co/cccfp2w4")
            input("Press Enter to continue...")
        elif choice == "5":
            show_text_file(os.path.join(base, "insights.txt"))
        elif choice == "6":
            play_audio(os.path.join(base, "environment_recording.wav"))
        elif choice == "7":
            break
        else:
            print("Invalid input. Try again.")


def run_week02():
    base = "submissions/week02"
    wav, fs = load_wav(os.path.join("submissions/week01", "environment_recording.wav"))

    while True:
        print("\n-- Week 2: Signal Analysis --")
        print("1. Play recording")
        print("2. Show waveform")
        print("3. Show spectrogram")
        print("4. Print time-domain features")
        print("5. Print spectral features")
        print("6. Show analysis of recording")
        print("7. Detect Chopin piano notes (pseudo-code)")
        print("8. Detect Chopin piano notes (executable-code)")
        print("9. Show audio analysis infographic")
        print("10. Back")
        choice = input("> ")

        if choice == "1":
            play_audio(os.path.join("submissions/week01", "environment_recording.wav"))
        elif choice == "2":
            analysis.plot_waveform(wav, fs)
        elif choice == "3":
            analysis.plot_spectrogram(wav, fs)
        elif choice == "4":
            feats = analysis.compute_time_features(wav, fs)
            for k, v in feats.items():
                print(f"{k}: {v:.4g}")
            input("Press Enter to continue...")
        elif choice == "5":
            feats = analysis.compute_spectral_features(wav, fs)
            for k, v in feats.items():
                print(f"{k}: {v:.4g}")
            input("Press Enter to continue...")
        elif choice == "6":
            text = open("submissions/week02/analysis.txt", "r").read()
            analysis.show_stepwise(text)
            input("Press Enter to continue...")
        elif choice == "7":
            print("Pseudo-code for detecting piano notes:\n")
            analysis.print_pseudo_code()
            input("Press Enter to continue...")
        elif choice == "8":
            print("\nDetecting Chopin piano notes…")
            onsets = analysis.detect_chopin_onsets(wav, fs)
            if not onsets:
                print("No piano onsets found.")
            else:
                for idx, t in enumerate(onsets, start=1):
                    print(f"  {idx}. Piano note at {t:.2f} s")
            input("Press Enter to continue...")
        elif choice == "9":
            print("View the image at: https://ibb.co/ccv8n2CC")
            input("Press Enter to continue...")
        elif choice == "10": 
            break
        else:
            print("Invalid input.")

def show_week03_menu():

    trained = False

    while True:
        print("\n-- Week 3: Hidden Markov Models --")
        print("1. Train all letter models")
        print("2. Inspect A, B, π for one letter")
        print("3. Decode a test morse sequence")
        print("4. Hand-craft & test ‘!’ model")
        print("5. Enforce start/stop states")
        print("6. Sweep hyperparameters")
        print("7. Back")

        choice = input("> ").strip()
        if choice=="1": 
            morse.train_all_models()
            trained = True
            print("Now you can proceed to the next steps. Thank you for training the model!")
            input("Press Enter to continue...")
        elif choice in {"2", "3", "4", "5", "6"}:
            if not trained:
                print("Please train the models first with option 1.")
                input("Press Enter to continue...")
                continue
            if choice == "2":
                morse.inspect_model()
            elif choice == "3":
                morse.decode_sequence()
            elif choice == "4":
                morse.show_exclamation_hmm("submissions/week03/hmm_exclamation.png")
                input("Press Enter to continue...")
            elif choice == "5":
                morse.enforce_start_stop()
                input("Press Enter to continue...")
            else: 
                morse.hyperparam_experiments()
        elif choice == "7":
            break

        else:
            print("Invalid input, please pick 1–7.")


def show_week04_menu():
    while True:
        print("\n-- Week 4: Sequence-to-Sequence Learning --")
        print("1. Train and test the tuned model")
        print("2. Read summary of reference paper")
        print("3. Read tuning insights")
        print("4. Back")

        choice = input("> ").strip()

        if choice == "1":
            run_model_tuned()
        elif choice == "2":
            print_summary.print_backprop_paper_summary()
        elif choice == "3":
            show_text_file("submissions/week04/tuning_insights.txt")
        elif choice == "4":
            break
        else:
            print("Invalid input..")

import subprocess

def run_model_tuned():
    print("\nRunning model_tuned.py...\n")
    try:
        subprocess.run(["poetry", "run", "python", "submissions/week04/model_tuned.py"], check=True)
    except subprocess.CalledProcessError as e:
        print("Error running model:", e)
