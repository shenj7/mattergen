import os
import json
import matplotlib.pyplot as plt
import numpy as np
import re

num_values = 16

def num_unique(file_path):
    total = 0
    with open(file_path, "r") as f:
        for line in f:
            line = line.strip()
            if "num_rows" in line:
                # Extract the number after "num_rows:"
                parts = line.split("num_rows:")
                if len(parts) > 1:
                    num_str = parts[1].split("}")[0].strip().strip(",")
                    try:
                        total += int(num_str)
                    except ValueError:
                        pass
    return num_values-total

def extract_metric_from_json(directories, filename="properties_space_group.txt"):
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
            results[d] = num_unique(file_path)/16
        except Exception as e:
            print(f"❌ Error reading {file_path}: {e}")
    return results


# --- Directory setup ---
directories = [f"./results/ml_bulk_modulus/{i}" for i in range(40, 440, 40)]

# --- Extract metric values ---
metric_data = extract_metric_from_json(directories, filename="properties_space_group.txt")

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
plt.savefig("frac_novel_systems_space_group.png", dpi=300, bbox_inches="tight")
plt.show()

