import os
import json
import matplotlib.pyplot as plt
import numpy as np

def extract_metric_from_json(directories, metric_name="frac_novel_systems", filename="metrics.json"):
    """
    Given a list of directories, read the specified metric from each metrics.json.
    Returns a dict: { directory_path: metric_value }
    """
    results = {}
    for d in directories:
        file_path = os.path.join(d, filename)
        if not os.path.exists(file_path):
            print(f"⚠️ Skipping {d}: {filename} not found.")
            continue

        try:
            with open(file_path, "r") as f:
                data = json.load(f)

            if metric_name in data and "value" in data[metric_name]:
                results[d] = data[metric_name]["value"]
            else:
                print(f"⚠️ Metric '{metric_name}' not found in {file_path}.")
        except Exception as e:
            print(f"❌ Error reading {file_path}: {e}")
    return results


# --- Directory setup ---
directories = [f"./results/ml_bulk_modulus/64batch/{i}" for i in range(40, 440, 40)]

# --- Extract metric values ---
metric_name = "frac_novel_systems"
metric_data = extract_metric_from_json(directories, metric_name=metric_name, filename="metrics.json")

# --- Prepare data for plotting ---
x_vals = []
y_vals = []

for dir_path, val in metric_data.items():
    try:
        i_val = int(os.path.basename(dir_path))
        x_vals.append(i_val)
        y_vals.append(val)
    except ValueError:
        print(f"⚠️ Could not parse integer from {dir_path}")

# --- Sort by x for consistent plotting ---
x_vals, y_vals = zip(*sorted(zip(x_vals, y_vals)))

# --- Plot ---
plt.figure(figsize=(8, 5))
plt.plot(x_vals, y_vals, marker="o", color="teal", linewidth=2, label="Fraction of Novel Systems")

plt.xlabel("Conditioned Bulk Modulus", fontsize=13)
plt.ylabel("Fraction of Novel Systems", fontsize=13)
plt.title("Novel Chemical Systems Fraction", fontsize=15, weight="bold")
plt.grid(True, linestyle="--", alpha=0.6)
plt.legend(frameon=True, loc="best", fontsize=11)
plt.tight_layout()

# --- Save and show ---
plt.savefig("frac_novel_systems.png", dpi=300, bbox_inches="tight")
plt.show()

