import os
import re
import numpy as np
import matplotlib.pyplot as plt

def extract_bulk_moduli_from_file(file_path):
    """Extract all bulk modulus values from a single text file."""
    with open(file_path, "r") as f:
        text = f.read()

    pattern = r"Bulk Modulus:\s*Column\(\[([^\]]+)\]\)"
    matches = re.findall(pattern, text)

    bulk_moduli = []
    for match in matches:
        numbers = [float(x.strip()) for x in match.split(",")]
        bulk_moduli.extend(numbers)
    return bulk_moduli


def extract_bulk_moduli_from_dirs(directories, filename="properties.txt"):
    """Extract bulk moduli from multiple directories."""
    results = {}
    for d in directories:
        file_path = os.path.join(d, filename)
        if not os.path.exists(file_path):
            print(f"⚠️ Skipping {d}: file '{filename}' not found.")
            continue
        moduli = extract_bulk_moduli_from_file(file_path)
        results[d] = moduli
    return results


# Example: you can change these directories to your ML vs conditioned sources
directories = [f"./results/ml_bulk_modulus/{i}" for i in range(40, 440, 40)]
all_bulk_moduli = extract_bulk_moduli_from_dirs(directories, filename="properties.txt")

# Prepare data
x_vals = []  # conditioned (dataset)
y_vals = []  # ML (predicted)
all_points_x = []
all_points_y = []

for dir_path, moduli in all_bulk_moduli.items():
    try:
        i_val = int(os.path.basename(dir_path))  # could represent "conditioned" value
        if moduli:
            mean_modulus = np.mean(moduli)
            x_vals.append(i_val)
            y_vals.append(mean_modulus)
            for m in moduli:
                all_points_x.append(i_val + np.random.uniform(-2, 2))
                all_points_y.append(m)
    except ValueError:
        print(f"⚠️ Could not parse integer from {dir_path}")

# Plot
plt.figure(figsize=(8, 6))

# Scatter distribution
plt.scatter(all_points_x, all_points_y, color="gray", alpha=0.5, s=40, label="Individual Samples")

# Mean trend
plt.plot(x_vals, y_vals, marker="o", color="steelblue", linewidth=2.5, label="Mean Bulk Modulus")

# Expected-value line x=y
min_val = min(min(x_vals), min(all_points_y))
max_val = max(max(x_vals), max(all_points_y))
plt.plot([min_val, max_val], [min_val, max_val], "r--", linewidth=2, label="Expected Value (x = y)")

# Labels, title, and grid
plt.xlabel("Conditioned Bulk Modulus (GPa)", fontsize=13)
plt.ylabel("ML Bulk Modulus (GPa)", fontsize=13)
plt.title("Conditioned vs Dataset Bulk Modulus", fontsize=15, weight="bold")

plt.grid(True, linestyle="--", alpha=0.6)
plt.legend(frameon=True, loc="best", fontsize=11)
plt.tight_layout()

# Save and show
plt.savefig("bulk_modulus_distribution_vs_i.png", dpi=300, bbox_inches="tight")
plt.show()

