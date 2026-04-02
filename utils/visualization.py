import matplotlib.pyplot as plt


def plot_metrics(results_dict):
    """
    results_dict = {
        "baseline": results,
        "ood": results,
        "epistemic": results
    }
    """

    labels = []
    violation_rates = []
    block_rates = []

    for name, res in results_dict.items():
        labels.append(name)
        violation_rates.append(res["violation_rate"])
        block_rates.append(res["block_rate"])

    # -------------------------
    # Violation Rate Plot
    # -------------------------
    plt.figure()
    plt.bar(labels, violation_rates)
    plt.title("Violation Rate Across Conditions")
    plt.xlabel("Condition")
    plt.ylabel("Violation Rate")
    plt.show()

    # -------------------------
    # Block Rate Plot
    # -------------------------
    plt.figure()
    plt.bar(labels, block_rates)
    plt.title("Block Rate Across Conditions")
    plt.xlabel("Condition")
    plt.ylabel("Block Rate")
    plt.show()