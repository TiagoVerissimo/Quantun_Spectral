import csv
import matplotlib.pyplot as plt
import numpy as np
import os

os.makedirs("results", exist_ok=True)

data = {}
with open("section7_results.csv", "r") as f:
    reader = csv.DictReader(f)
    for row in reader:
        # Robust reading of the CSV to handle both old shifted and new format
        if "wall_validation" not in row or row.get("wall_validation") is None:
            if row.get("wall") is None or row.get("wall") == "":
                row["wall"] = row["validated"]
                row["validated"] = "False"
            row["wall_validation"] = "0.0"
            
        N = int(row["N"])
        if N not in data:
            data[N] = []
            
        parsed_row = {}
        for k, v in row.items():
            if k == "validated":
                parsed_row[k] = v.strip().lower() == "true"
            elif v == "" or v is None:
                parsed_row[k] = 0.0
            else:
                parsed_row[k] = float(v)
        data[N].append(parsed_row)

# Sort by N
sorted_Ns = sorted(data.keys())

means = {
    "N": [],
    "solves_hybrid": [],
    "solves_uniform": [],
    "frac_weyl": [],
    "frac_floor": [],
    "frac_floor_poly": [],
    "frac_horn": [],
    "wall": [],
    "wall_validation": []
}
stds = {
    "solves_hybrid": [],
    "solves_uniform": [],
    "wall": []
}

for N in sorted_Ns:
    instances = data[N]
    num = len(instances)
    means["N"].append(N)
    
    hybrids = [inst["solves_hybrid"] for inst in instances]
    uniforms = [inst["solves_uniform"] for inst in instances]
    
    means["solves_hybrid"].append(np.mean(hybrids))
    stds["solves_hybrid"].append(np.std(hybrids))
    means["solves_uniform"].append(np.mean(uniforms))
    stds["solves_uniform"].append(np.std(uniforms))
    
    means["frac_weyl"].append(sum(inst["frac_weyl"] for inst in instances) / num)
    means["frac_floor"].append(sum(inst["frac_floor"] for inst in instances) / num)
    means["frac_floor_poly"].append(sum(inst["frac_floor_poly"] for inst in instances) / num)
    means["frac_horn"].append(sum(inst["frac_horn"] for inst in instances) / num)
    
    # Calculate mean and standard deviation of production wall time
    walls = [inst["wall"] for inst in instances]
    means["wall"].append(np.mean(walls))
    stds["wall"].append(np.std(walls))
    
    means["wall_validation"].append(sum(inst["wall_validation"] for inst in instances) / num)

# ----------------- Plot 1: Efficiency Comparison (Hybrid vs. Uniform) -----------------
plt.figure(figsize=(8, 5))
bar_width = 0.35
r1 = np.arange(len(means["N"]))
r2 = [x + bar_width for x in r1]

plt.bar(r1, means["solves_hybrid"], yerr=stds["solves_hybrid"], capsize=5, color="#3a86c8", width=bar_width, edgecolor="grey", label="Hybrid Certified Sweep (Alg 1)")
plt.bar(r2, means["solves_uniform"], yerr=stds["solves_uniform"], capsize=5, color="#f25c54", width=bar_width, edgecolor="grey", label="Uniform Equal-Rigor Grid")

plt.xlabel("Qubits (N)", fontweight="bold", fontsize=11)
plt.ylabel("Mean Matrix Solves (Oracle Calls)", fontweight="bold", fontsize=11)
plt.xticks([r + bar_width/2 for r in range(len(means["N"]))], [str(N) for N in means["N"]])
plt.title("Oracle Query Efficiency: Hybrid vs. Uniform Grid", fontsize=13, fontweight="bold", pad=15)
plt.grid(True, linestyle=":", alpha=0.6)
plt.legend(fontsize=10)
plt.tight_layout()
plt.savefig("results/efficiency_comparison.png", dpi=160)
plt.close()

# ----------------- Plot 2: Certificate Path Coverage vs N -----------------
plt.figure(figsize=(8, 5))
plt.plot(means["N"], means["frac_weyl"], "o-", color="#9ea1a5", lw=2, label="Weyl Endpoint Prop (Prop 5.4)")
plt.plot(means["N"], means["frac_floor_poly"], "s-", color="#4895ef", lw=2, label="PSD Floor - Poly Input (Prop 5.6)")
plt.plot(means["N"], means["frac_floor"], "d-", color="#3a86c8", lw=2, label="PSD Floor - Oracle (Prop 5.6)")
plt.plot(means["N"], means["frac_horn"], "^-", color="#f25c54", lw=2, label="Horn T_1^n Bound (Prop 6.4)")

plt.xlabel("Qubits (N)", fontweight="bold", fontsize=11)
plt.ylabel("Fraction of Path Certified Positive", fontweight="bold", fontsize=11)
plt.xticks(means["N"])
plt.ylim(0, 0.35)
plt.title("Endpoint Certificate Path Coverage vs. System Size", fontsize=13, fontweight="bold", pad=15)
plt.grid(True, linestyle=":", alpha=0.6)
plt.legend(fontsize=9, loc="upper right")
plt.tight_layout()
plt.savefig("results/certificate_coverage.png", dpi=160)
plt.close()

# ----------------- Plot 3: Execution Runtime Scaling -----------------
plt.figure(figsize=(8, 5))

means_wall = np.array(means["wall"])
stds_wall = np.array(stds["wall"])

# Since y-scale is logarithmic, we must make sure error bar bounds are positive
yerr_lower = np.minimum(stds_wall, means_wall - 1e-4)  # prevent non-positive values in log scale
yerr = [yerr_lower, stds_wall]

plt.errorbar(means["N"], means["wall"], yerr=yerr, fmt="o-", color="#2b2d42", 
             lw=2.5, capsize=5, label="Mean Production Wall Time (Alg 1)")

# Plot theoretical scaling O(2^N * N) reference line, matched at N=12
idx_12 = sorted_Ns.index(12)
t_12 = means["wall"][idx_12]
c_ref = t_12 / (2**12 * 12)
ref_curve = c_ref * (2**np.array(means["N"])) * np.array(means["N"])

plt.plot(means["N"], ref_curve, "--", color="#e76f51", lw=2, 
         label=r"Theoretical Complexity $O(2^N \cdot N)$")

plt.yscale("log", base=2)
plt.xlabel("Qubits (N)", fontweight="bold", fontsize=11)
plt.ylabel("Serial Wall-clock Time (s)", fontweight="bold", fontsize=11)
plt.xticks(means["N"])
plt.title("Execution Runtime Scaling vs. Theoretical Complexity", fontsize=13, fontweight="bold", pad=15)
plt.grid(True, which="both", linestyle=":", alpha=0.6)
plt.legend(fontsize=10)
plt.tight_layout()
plt.savefig("results/runtime_scaling.png", dpi=160)
plt.close()

print("Generated new graphs based on section7_results.csv successfully.")
