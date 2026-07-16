import csv
import matplotlib.pyplot as plt
import numpy as np
import os

os.makedirs("results", exist_ok=True)

data = {}
with open("section7_results.csv", "r") as f:
    reader = csv.DictReader(f)
    for row in reader:
        N = int(row["N"])
        if N not in data:
            data[N] = []

        parsed_row = {}
        for k, v in row.items():
            if k in ["validated", "weylA_certified"]:
                parsed_row[k] = str(v).strip().lower() == "true"
            elif k in ["sector"]:
                parsed_row[k] = str(v)
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

    hybrids = [inst.get("weylA_oracle_calls", 0) for inst in instances]
    uniforms = [inst.get("uniform_solves", 0) for inst in instances]

    means["solves_hybrid"].append(np.mean(hybrids))
    stds["solves_hybrid"].append(np.std(hybrids))
    means["solves_uniform"].append(np.mean(uniforms))
    stds["solves_uniform"].append(np.std(uniforms))

    means["frac_weyl"].append(sum(inst.get("weyl0_frac_pos", 0) for inst in instances) / num)
    means["frac_floor"].append(sum(inst.get("psd_frac_pos", 0) for inst in instances) / num)

    means["frac_floor_poly"].append(sum(inst.get("frac_floor_poly", 0) for inst in instances) / num)

    # Calculate mean and standard deviation of production wall time
    walls = [inst.get("t_weylA", 0) for inst in instances]
    means["wall"].append(np.mean(walls))
    stds["wall"].append(np.std(walls))

# ----------------- Plot 1: Efficiency Comparison (Hybrid vs. Uniform) -----------------
plt.figure(figsize=(8, 5))
bar_width = 0.35
r1 = np.arange(len(sorted_Ns))
r2 = [x + bar_width for x in r1]

# Calculate medians and IQR
medians_hybrid = []
yerr_hybrid_lower = []
yerr_hybrid_upper = []

medians_uniform = []
yerr_uniform_lower = []
yerr_uniform_upper = []

for N in sorted_Ns:
    instances = data[N]
    hybrids = np.array([inst.get("weylA_oracle_calls", 0) for inst in instances])
    uniforms = np.array([inst.get("uniform_solves", 0) for inst in instances])

    # Hybrid
    med_h = np.median(hybrids)
    p25_h = np.percentile(hybrids, 25)
    p75_h = np.percentile(hybrids, 75)
    medians_hybrid.append(med_h)
    yerr_hybrid_lower.append(med_h - p25_h)
    yerr_hybrid_upper.append(p75_h - med_h)

    # Uniform
    med_u = np.median(uniforms)
    p25_u = np.percentile(uniforms, 25)
    p75_u = np.percentile(uniforms, 75)
    medians_uniform.append(med_u)
    yerr_uniform_lower.append(med_u - p25_u)
    yerr_uniform_upper.append(p75_u - med_u)

yerr_hybrid = np.array([yerr_hybrid_lower, yerr_hybrid_upper])
yerr_uniform = np.array([yerr_uniform_lower, yerr_uniform_upper])

plt.bar(r1, medians_hybrid, yerr=yerr_hybrid, capsize=5, color="#3a86c8", width=bar_width, edgecolor="grey", label="Hybrid Certified Sweep (Alg 1)")
plt.bar(r2, medians_uniform, yerr=yerr_uniform, capsize=5, color="#f25c54", width=bar_width, edgecolor="grey", label="Uniform Workload Reference Grid")

plt.xlabel("Qubits (N)", fontweight="bold", fontsize=11)
plt.ylabel("Median Matrix Solves (Oracle Calls)", fontweight="bold", fontsize=11)
plt.xticks([r + bar_width/2 for r in range(len(sorted_Ns))], [str(N) for N in sorted_Ns])
plt.title("Solve-Count Workload: Adaptive vs. Uniform Grid", fontsize=13, fontweight="bold", pad=15)
plt.grid(True, linestyle=":", alpha=0.6)
plt.legend(fontsize=10)
plt.tight_layout()
plt.savefig("results/efficiency_comparison.png", dpi=160)
plt.close()

# ----------------- Plot 2: Certificate Path Coverage vs N -----------------
plt.figure(figsize=(8, 5))
plt.plot(means["N"], means["frac_weyl"], "o-", color="#9ea1a5", lw=2, label="Weyl Endpoint Prop (Prop 5.4)")
plt.plot(means["N"], means["frac_floor"], "d-", color="#3a86c8", lw=2, label="PSD Floor - Oracle (Prop 5.6)")


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

# Plot theoretical scaling O(2^N * N) reference line, matched at max N
idx_max = -1
max_N = means["N"][-1]
t_max = means["wall"][-1]
c_ref = t_max / (2**max_N * max_N)
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
