import os
from utils.loader import show_text_file, open_link
from utils.audio import play_audio
from submissions.week01 import playground

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
        print("1. Sine Wave Playground")
        print("2. View A4 drawing link")
        print("3. Read description of acoustic features")
        print("4. View article insights")
        print("5. Play environment recording")
        print("6. Back")
        choice = input("> ")

        base = "submissions/week01"

        if choice == "1":
            playground.run_playground()
        elif choice == "2":
            print("View the image at: https://ibb.co/cccfp2w4")
        elif choice == "3":
            show_text_file(os.path.join(base, "description.txt"))
        elif choice == "4":
            show_text_file(os.path.join(base, "insights.txt"))
        elif choice == "5":
            play_audio(os.path.join(base, "environment_recording.wav"))
        elif choice == "6":
            break
        else:
            print("Invalid input. Try again.")

