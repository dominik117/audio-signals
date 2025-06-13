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
