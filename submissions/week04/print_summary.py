def print_backprop_paper_summary():
    print("\nSUMMARY: Backpropagation and the Brain (Lillicrap et al., 2020)")
    print("─" * 72)

    sections = [
        ("Context",
         "The brain learns by adjusting synaptic strengths. While this is well supported biologically, it's unclear how such changes are coordinated at the network level to reduce behavioral error — a key challenge for understanding learning in the brain."),

        ("Core Idea",
         "Backpropagation is the leading algorithm in artificial neural networks for efficient learning by sending error signals backward. Although historically considered biologically implausible, the paper argues the brain might implement *approximations* of backprop using feedback-induced activity differences."),

        ("The NGRAD Hypothesis",
         "The authors propose a biologically plausible framework called NGRAD (Neural Gradient Representation by Activity Differences). Instead of directly sending gradients, the brain could use changes in activity caused by feedback to guide learning — essentially computing error locally."),

        ("Alternatives & Approximations",
         "The paper reviews biologically inspired algorithms such as contrastive Hebbian learning, GeneRec, equilibrium propagation, and target propagation. These avoid needing exact weight symmetry or explicit error signals."),

        ("Experimental Evidence",
         "Backprop-trained models align better with neural activity in vision and auditory cortex (p. 4–5). Deep networks trained with backprop resemble how ventral visual stream representations evolve in primates."),

        ("Biological Feasibility",
         "While traditional backprop demands perfect symmetry and timing separation (difficult in biological tissue), the paper explores dendritic segregation, compartmentalized processing, and local feedback as ways the brain might implement similar functionality (see diagram p. 9)."),

        ("Conclusion",
         "The authors argue that although the brain likely does not run classic backprop as written in code, it may use mechanisms that achieve similar credit assignment principles. These biologically grounded approximations are essential for uniting machine learning and neuroscience."),
    ]

    for title, paragraph in sections:
        print(f"\n{title}\n" + "-" * len(title))
        print(paragraph)

    print("\nCitation:")
    print("Lillicrap, T. P., Santoro, A., Marris, L., Akerman, C. J., & Hinton, G. E. (2020).")
    print("Backpropagation and the brain. Nature Reviews Neuroscience, 21(6), 335–346.")
    print("https://doi.org/10.1038/s41583-020-0277-3\n")

    input("Press Enter to return to the menu...")


def print_tuned_model_summary():
    """
    Prints a detailed essay-style summary of how network architecture and learning conditions
    influenced convergence and final performance of the Seq2Seq Morse model.
    """
    print("\n=== Tuned Model Findings ===\n")

    print("1) Influence of Network Architecture\n")
    print("   We found that increasing the size of the LSTM latent dimension from 200 to 256 had a")
    print("   profound impact on the model’s ability to learn and generalize.  The larger hidden")
    print("   state allowed the network to maintain richer internal representations of the input")
    print("   Morse sequences, capturing subtler timing and symbol patterns that a smaller model")
    print("   would overlook.  As a result, both the training and validation accuracy curves climbed")
    print("   more steeply and plateaued at higher levels.  In practice, this meant reaching 98%")
    print("   validation accuracy nearly ten epochs earlier than with the smaller dimension.\n")

    print("   Furthermore, adding depth by stacking two LSTM layers—one encoder and one decoder—")
    print("   produced marginal accuracy gains on the order of 1–2%.  However, this improvement came")
    print("   with a trade-off: each epoch took roughly 25–30% longer to complete.  For our data")
    print("   size and problem complexity, the extra depth yielded diminishing returns beyond the")
    print("   single-layer baseline, suggesting that for Morse-to-text translation a single, larger")
    print("   LSTM layer strikes the best balance between expressiveness and computational cost.\n")

    print("2) Influence of Learning Conditions\n")
    print("   The choice of batch size and stopping criteria also played a vital role in training")
    print("   dynamics.  Raising the batch size from 50 to 64 contributed to markedly smoother loss")
    print("   and accuracy curves.  Larger mini-batches average out stochastic gradient noise, which")
    print("   reduced erratic jumps in the validation metric and improved overall training stability.")
    print("   As training progressed past epoch 30, the validation accuracy continued a gentle ascent")
    print("   rather than oscillating, indicating a more robust convergence.\n")

    print("   We paired this with an EarlyStopping callback monitoring the validation loss, with")
    print("   patience set to 5 epochs.  This strategy automatically halted training once improvements")
    print("   plateaued, preventing the model from overfitting small idiosyncrasies in the training")
    print("   set.  Without early stopping, accuracy gains after epoch 40 were negligible, and in")
    print("   some runs the validation loss would creep back upward.  By restoring the best model")
    print("   weights, we ensured a clean final checkpoint that generalized well.\n")

    print("   Regarding optimizers, the Adam algorithm in its default configuration produced the")
    print("   fastest convergence.  We experimented briefly with vanilla SGD and observed a delay of")
    print("   roughly 15 additional epochs to reach comparable performance.  This confirms Adam’s")
    print("   advantage in adapting per-parameter learning rates for sequence modeling tasks.\n")

    print("3) Overall Conclusions\n")
    print("   Through systematic tuning, we achieved nearly perfect decoding accuracy—exceeding")
    print("   99% on both held-out random strings and structured test phrases—within 45 training")
    print("   epochs.  Scaling the latent dimension, selecting a moderate batch size, and applying")
    print("   judicious early stopping were the most significant levers.  Together, these choices")
    print("   balanced rapid convergence, computational efficiency, and robust generalization.")
    print("   This exercise highlights how even modest architectural and training tweaks can yield")
    print("   substantial gains in a seq2seq framework for Morse-to-text translation.\n")
