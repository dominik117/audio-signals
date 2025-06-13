import re

def show_text_file(filepath):
    try:
        with open(filepath, "r") as file:
            print("\n" + file.read())
    except FileNotFoundError:
        print("File not found.")
    input("\nPress Enter to continue...")

def open_link(link):
    print(f"\nOpen this link in your browser: {link}")
    input("Press Enter to continue...")

def show_stepwise(text):
    """
    Display each sentence in `text` one at a time.
    User must press Enter to advance.
    """
    # Split on sentence-ending punctuation, keeping the delimiter
    sentences = re.findall(r'[^.?!]+[.?!]', text.strip())
    total = len(sentences)

    for idx, sentence in enumerate(sentences, start=1):
        print(f"\n• Analysis {idx}/{total} {sentence.strip()}")
        input("Press Enter to continue…")

def prompt(message: str) -> str:
    """
    Print the given prompt message and return what the user types.
    """
    try:
        return input(message)
    except KeyboardInterrupt:
        print("\nInterrupted. Returning empty string.")
        return ""
