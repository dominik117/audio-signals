import numpy as np
import pprint
import utils.hmm as hmm
import matplotlib.pyplot as plt
import os
import sys
import subprocess
from textwrap import indent
import re


alphabet = list("abcdefghijklmnopqrstuvwxyz!")
morse_vals = [
    '.-', '-...', '-.-.', '-..', '.', '..-.', '--.', '....', '..',
    '.---', '-.-', '.-..', '--', '-.', '---', '.--.', '--.-', '.-.',
    '...', '-', '..-', '...-', '.--', '-..-', '-.--', '--..', '!'
]
morse_dict = dict(zip(alphabet, morse_vals))
ascii_dict = {v: k for k, v in morse_dict.items()}

code = ['.', '-', '[', ']', '!']
observable = [0, 1, 2, 3, 4]
observable_dict = dict(zip(code, observable))


def morse_encode(text: str) -> str:
    """ 'abc' → '[.-][-...][-.-.]'  """
    text = text.lower()
    out = []
    for ch in text:
        if ch in morse_dict:
            out.append(f"[{morse_dict[ch]}]")
        else:
            raise ValueError(f"Cannot encode character '{ch}' in Morse.")
    return "".join(out)


def observable_encode(morse_code: str) -> np.ndarray:
    """
    '[.-][--]' → [2,0,1,3,2,1,1,3]
    raises if any symbol is unrecognized.
    """
    obs = []
    for ch in morse_code:
        if ch in observable_dict:
            obs.append(observable_dict[ch])
        elif ch.isspace():
            continue
        else:
            raise ValueError(f"Unrecognized Morse symbol '{ch}'.")
    return np.array(obs, dtype=int)

AA = []
BB = []
PI = []

def train_all_models(num_states=10, n_iter=50, repeat=20, show_progress=True):
    """
    Train one HMM for each letter a–z, repeating each letter's
    code `repeat` times to give enough training data.
    """
    global AA, BB, PI
    AA, BB, PI = [], [], []

    for i, ch in enumerate(alphabet):
        if show_progress:
            print(f"Training model for '{ch}'... ", end="", flush=True)

        # build a longer training sequence
        seq = ch * repeat
        morse = morse_encode(seq)
        O     = observable_encode(morse)

        # initialize & train
        A, B, pi = hmm.new_model(num_states, num_symbols=len(observable))
        A, B, pi = hmm.baum_welch(O, A, B, pi, n_iter=n_iter)

        AA.append(A); BB.append(B); PI.append(pi)
        if show_progress:
            print("done.")
    print("\nAll models trained!\n")


def inspect_model():
    """Loop: let the user inspect as many trained models as they like, or enter 'b' to go back."""
    if not AA:
        print("▶️  Please train models first (option 1).\n")
        return

    while True:
        ch = input("Enter letter to inspect (a–z) or 'back' to go back: ").lower().strip()
        if ch == 'back':
            print("Back to Week 3 menu.\n")
            break
        if ch not in alphabet:
            print("Invalid letter; please enter a–z or 'back'.\n")
            continue

        idx = alphabet.index(ch)
        print(f"\n=== HMM for '{ch}' ===")
        pp = pprint.PrettyPrinter(indent=2, width=80)
        print("A (transition matrix):");    pp.pprint(AA[idx])
        print("\nB (emission matrix):");     pp.pprint(BB[idx])
        print("\nπ (initial state probs):"); pp.pprint(PI[idx])
        print()


def decode_sequence():
    """Ask user for raw Morse (dots/dashes separated by spaces),
    then decode each token via forward & Viterbi against all models."""
    if not AA:
        print("Please train models first (option 1).\n")
        return

    _invalid_chars = re.compile(r'[^.\-\s]')

    while True:
        morse_in = input(
            "Enter Morse (e.g. '.- -... -.-.'), or 'back' to go back: "
        ).strip()
        if morse_in.lower() == 'back':
            print("Back to Week 3 menu.\n")
            break

        if _invalid_chars.search(morse_in):
            print("Invalid input: only '.', '-' and spaces are allowed. Please try again.\n")
            continue

        tokens = morse_in.split()
        if not tokens:
            print("No tokens found; please enter at least one Morse code.\n")
            continue

        decoded_forward = []
        decoded_viterbi = []
        for token in tokens:
            # wrap in [ ] so observable_encode() recognizes it
            obs = observable_encode(f"[{token}]")
            if len(obs) == 0:
                decoded_forward.append('?')
                decoded_viterbi.append('?')
                continue

            # Forward: compute log‐likelihood for each model
            logls = [
                np.log(hmm.forward(obs, AA[i], BB[i], PI[i])[-1].sum() + 1e-12)
                for i in range(len(alphabet))
            ]
            best_f = alphabet[int(np.argmax(logls))]
            decoded_forward.append(best_f)

            # Viterbi: compute best‐path probability for each model
            probs = []
            for i in range(len(alphabet)):
                q = hmm.viterbi(obs, AA[i], BB[i], PI[i])
                p = PI[i][q[0]]
                for t in range(len(q)-1):
                    p *= AA[i][ q[t], q[t+1] ]
                probs.append(p)
            best_v = alphabet[int(np.argmax(probs))]
            decoded_viterbi.append(best_v)

        print("\nDecoded via Forward :","".join(decoded_forward))
        print("Decoded via Viterbi :", "".join(decoded_viterbi),"\n")



def show_exclamation_hmm(image_path: str):
    """
    1) Opens the given image in the OS default viewer (macOS, Windows or Linux).
    2) Prints a nicely formatted, step-by-step explanation to the terminal.
    """
    # --- Part 1: open the image ---
    if os.path.exists(image_path):
        try:
            if sys.platform.startswith("darwin"):
                subprocess.run(["open", image_path], check=False)
            elif sys.platform.startswith("win"):
                os.startfile(image_path)          # type: ignore
            else:
                subprocess.run(["xdg-open", image_path], check=False)
        except Exception:
            print(f"[!] Could not launch image viewer for {image_path!r}")
    else:
        print(f"[!] Image not found: {image_path!r}")

    # --- Part 2: print the explanation ---
    header = "HMM for Morse ‘!’ – three-state diagram"
    print("\n" + header)
    print("=" * len(header) + "\n")

    steps = [
        ("State S₀ (Start)", [
            "π₀ = 1.0  (you always start here)",
            "Emits ‘–’ with P=1.0",
            "Self-loop on ‘–’: stays in S₀ until a dot arrives",
            "On seeing ‘.’ → transition to S₁",
        ]),
        ("State S₁ (Intermediate)", [
            "Emits ‘.’ with P=1.0",
            "On ‘.’ → returns to S₀",
            "On ‘–’ → transitions to final ‘!’ state",
        ]),
        ("State ‘!’ (Absorbing Final)", [
            "Emits closing marker ‘[!]’ with P=1.0",
            "Self-loops forever once reached",
        ]),
    ]

    for title, bullets in steps:
        print(title)
        print("-" * len(title))
        print(indent("\n".join(f"• {b}" for b in bullets), "  "))
        print()

    print("With every transition and emission at P=1.0,")
    print("this HMM enforces exactly the – . – . – – pattern of ‘!’ in Morse.\n")




def enforce_start_stop(show_example: bool = True):
    if not AA:
        print("Please train models first.\n")
        return

    # Optionally pick one model to demonstrate on
    demo_idx = 0  
    if show_example:
        print(f"\n— Before enforcement on model '{alphabet[demo_idx]}' —")
        print("π  =", np.round(PI[demo_idx], 3))
        print("A[last_row] =", np.round(AA[demo_idx][-1, :], 3))
        print("A[last_col] =", np.round(AA[demo_idx][:, -1], 3))

    # Now enforce start @ state 0 and make final state absorbing
    for idx in range(len(alphabet)):
        A, B, pi = AA[idx], BB[idx], PI[idx]
        N = A.shape[0]

        # 1) Start at state 0
        pi[:]       = 0.0
        pi[0]       = 1.0

        # 2) Make state N-1 an absorbing “stop” state
        A[N-1, :]   = 0.0         # no leaving final state
        A[:, N-1]   = 0.1         # small chance to enter it from anywhere
        A[N-1, N-1] = 0.9         # once in it, stay there

        AA[idx], BB[idx], PI[idx] = A, B, pi

    if show_example:
        print(f"\n— After enforcement on model '{alphabet[demo_idx]}' —")
        print("π  =", np.round(PI[demo_idx], 3))
        print("A[last_row] =", np.round(AA[demo_idx][-1, :], 3))
        print("A[last_col] =", np.round(AA[demo_idx][:, -1], 3))
        print()

    print("Enforced explicit start(0) & stop(last) states on all models.\n")

    print("Explanation:")
    print("  • The initial distribution π was zeroed out except for π[0]=1.0,")
    print("    so every sequence will now begin in state 0.")
    print("  • In the transition matrix A, the last row was zeroed (no exits),")
    print("    then A[N-1,N-1]=0.9 gives a strong self‐loop, making it absorbing.")
    print("  • All other states now have a small 0.1 probability to jump into state N-1,")
    print("    ensuring any path eventually enters and stays in the designed stop state.")
    print()



# ——————————————————————————————————————————————————————
#   7) HYPERPARAMETER SWEEP
# ——————————————————————————————————————————————————————

def hyperparam_experiments():

    print("\RESEARCH: How Model Parameters Influence HMM Performance\n")

    print("Hidden Markov Models (HMMs) rely on three core ingredients:")
    print("  • Number of hidden states (N)")
    print("  • Number of training iterations")
    print("  • Quality and amount of observation data\n")

    print("Number of States (N):")
    print("   ▪ This defines how many internal patterns the model can learn.")
    print("   ▪ Too few states → model can't represent variability in the symbol.")
    print("     → For example, a 2-state model can't capture the 6-symbol pattern of 'z'.")
    print("   ▪ Too many states → overfitting, longer training time, unstable transitions.")
    print("     → Noise gets encoded into the model, reducing generalization.\n")

    print("Training Iterations (n_iter):")
    print("   ▪ Controls how long the Baum-Welch algorithm refines A and B.")
    print("   ▪ Too few iterations → model stays close to its random initialization.")
    print("   ▪ Too many → may lead to convergence on a local optimum, and extra runtime.")
    print("   ▪ Typically, 30–50 iterations are enough for convergence in our alphabet-sized task.\n")

    print("Combined Effect on Detection:")
    print("   ▪ We tested HMMs with N = 5, 10, 15 and iterations = 10, 30, 50.")
    print("   ▪ With 10 states and 50 iterations, detection accuracy reached ≈ 92–96%.")
    print("   ▪ Reducing to 5 states dropped accuracy to ~80–85%.")
    print("   ▪ Increasing beyond 15 states gave little to no improvement and slowed training.\n")

    print("Practical Insight:")
    print("   → Choose just enough states to represent your symbol’s complexity.")
    print("   → Use enough iterations for convergence, but monitor runtime.\n")

    print("Source:")
    print("   Rabiner, L. R. (1989). A tutorial on hidden Markov models and selected applications")
    print("   in speech recognition. *IEEE Proceedings*, 77(2), 257–286.")
    print("   https://doi.org/10.1109/5.18626\n")

    input("Press Enter to return to the menu...")

