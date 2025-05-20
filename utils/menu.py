import os
from utils.loader import show_text_file, open_link
from utils.audio import play_audio
from submissions.week01 import synth

def show_main_menu():
    while True:
        print("\n=== Dominik Bacher – Audio Submissions ===")
        print("1. Week 1 – Introduction to Acoustics")
        print("2. Exit")
        choice = input("> ")

        if choice == "1":
            show_week01_menu()
        elif choice == "2":
            print("Goodbye!")
            break
        else:
            print("Invalid input. Try again.")

def show_week01_menu():
    while True:
        print("\n-- Week 1: Introduction to Acoustics --")
        print("1. View A4 drawing link")
        print("2. Read description of acoustic features")
        print("3. Play recorded audio")
        print("4. View article insights")
        print("5. Generate improved audio snippet")
        print("6. Back")
        choice = input("> ")

        base = "submissions/week01"

        if choice == "1":
            show_text_file(os.path.join(base, "drawing.txt"))
        elif choice == "2":
            show_text_file(os.path.join(base, "description.txt"))
        elif choice == "3":
            play_audio(os.path.join(base, "Track-5.wav"))
        elif choice == "4":
            show_text_file(os.path.join(base, "insights.txt"))
        elif choice == "5":
            synth.generate()
        elif choice == "6":
            break
        else:
            print("Invalid input. Try again.")
