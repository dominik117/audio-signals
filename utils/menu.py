import os
from utils.loader import show_text_file, open_link
from utils.audio import play_audio, load_wav
from submissions.week01 import playground
from submissions.week02 import analysis 


def show_main_menu():
    while True:
        print("\n=== Dominik Bacher – Audio Submissions ===")
        print("1. Week 1 – Introduction to Acoustics")
        print("2. Week 2 – Signal Analysis")
        print("3. Exit")
        choice = input("> ")

        if choice == "1":
            show_week01_menu()
        elif choice == "2":
            run_week02()
        elif choice == "3":
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
        print("9. Show image imagining features on the time and frequency domain")
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
            print("View the image at: https://ibb.co/https://ibb.co/twMrgWQx")
            input("Press Enter to continue...")
        elif choice == "10":
            
            break
        else:
            print("Invalid input.")
