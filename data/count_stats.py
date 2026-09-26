import pandas as pd

# Read CSV
df = pd.read_csv("data/processed/labels.csv")

# Define signer groups
fsl = df[df["signer"].isin(["S0", "S1", "S2", "S3"])]
smp = df[df["signer"].isin(["S4", "S5", "S6", "S7"])]
cmb = pd.concat([fsl, smp])

def generate_report(name, data):
    lines = []
    total = len(data)
    not_occ = len(data[data["occluded"] == 0])
    occ = len(data[data["occluded"] == 1])
    lines.append(f"{name}: Total={total}, Not Occluded={not_occ}, Occluded={occ}\nDetailed Report:\n")

    # --- Category breakdown ---
    lines.append("By Category:\n")
    for cat, group in data.groupby("cat"):
        lines.append(
            f"  Category {cat}: Total={len(group)}, Not Occluded={len(group[group.occluded==0])}, Occluded={len(group[group.occluded==1])}"
        )

    # --- Gloss breakdown ---
    lines.append("\nBy Gloss:\n")
    for gloss, group in data.groupby("gloss"):
        lines.append(
            f"  Gloss {gloss}: Total={len(group)}, Not Occluded={len(group[group.occluded==0])}, Occluded={len(group[group.occluded==1])}"
        )

    lines.append("\n" + "=" * 80 + "\n")
    return "\n".join(lines)

# Generate reports
output = []
output.append(generate_report("FSL-105", fsl))
output.append(generate_report("SMP-105", smp))
output.append(generate_report("CMB-105", cmb))

# Write to file
with open("counting result.txt", "w") as f:
    f.write("\n".join(output))

print("✅ Counting complete — results saved to 'counting result.txt'")
