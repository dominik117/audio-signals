def show_text_file(filepath):
    try:
        with open(filepath, "r") as file:
            print("\n" + file.read())
    except FileNotFoundError:
        print("File not found.")

def open_link(link):
    import webbrowser
    webbrowser.open(link)
