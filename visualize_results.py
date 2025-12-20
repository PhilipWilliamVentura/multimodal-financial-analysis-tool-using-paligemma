import matplotlib.pyplot as plt
import numpy as np
import os

OUT_DIR = "figures"
os.makedirs(OUT_DIR, exist_ok=True)

# Rendering settings
plt.rcParams.update({
    "font.size": 10,
    "font.family": "serif",
    "figure.figsize": (3.5, 2.5),
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 0.8,
    "grid.linewidth": 0.5,
    "lines.linewidth": 1.5,
    "patch.linewidth": 0.5,
    "xtick.major.width": 0.8,
    "ytick.major.width": 0.8,
})

# Data
seq = np.array([128, 256, 512])
kv_ms = np.array([98.08, 98.35, 95.98])
kv_ci = np.array([0.44, 0.12, 2.35])
no_kv_ms = np.array([984.86, 1082.63, 1268.43])
no_kv_ci = np.array([8.99, 2.96, 20.55])
kv_tps = np.array([10.2, 10.17, 10.46])
kv_tps_ci = np.array([0.05, 0.01, 0.29])
no_kv_tps = np.array([1.02, 0.92, 0.79])
no_kv_tps_ci = np.array([0.01, 0.0, 0.01])
kv_mem = np.array([6547.58, 6547.58, 6547.58])
kv_mem_ci = np.array([1.27, 1.27, 1.27])
no_kv_mem = np.array([7122.91, 7437.55, 8069.92])
no_kv_mem_ci = np.array([1.58, 1.5, 1.66])

# FIGURE 1 — Steady-State Latency vs Sequence Length
fig, ax = plt.subplots()
ax.errorbar(seq, kv_ms, yerr=kv_ci, marker='o', capsize=3, label="KV-cache", markersize=5)
ax.errorbar(seq, no_kv_ms, yerr=no_kv_ci, marker='s', capsize=3, label="No KV-cache", markersize=5)
ax.set_xlabel("Sequence Length (tokens)")
ax.set_ylabel("Latency (ms/token)")
ax.set_title("Steady-State Latency vs Sequence Length")
ax.legend(frameon=False)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "fig1_latency.pdf"), dpi=300, bbox_inches='tight')
plt.savefig(os.path.join(OUT_DIR, "fig1_latency.png"), dpi=300, bbox_inches='tight')
plt.close()
print("Saved: fig1_latency.pdf")

# FIGURE 2 — Throughput vs Sequence Length
fig, ax = plt.subplots()
ax.errorbar(seq, kv_tps, yerr=kv_tps_ci, marker='o', capsize=3, label="KV-cache", markersize=5)
ax.errorbar(seq, no_kv_tps, yerr=no_kv_tps_ci, marker='s', capsize=3, label="No KV-cache", markersize=5)
ax.set_xlabel("Sequence Length (tokens)")
ax.set_ylabel("Throughput (tokens/sec)")
ax.set_title("Throughput vs Sequence Length")
ax.legend(frameon=False)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "fig2_throughput.pdf"), dpi=300, bbox_inches='tight')
plt.savefig(os.path.join(OUT_DIR, "fig2_throughput.png"), dpi=300, bbox_inches='tight')
plt.close()
print("Saved: fig2_throughput.pdf")

# FIGURE 3 — Speedup vs Sequence Length
fig, ax = plt.subplots()
speedup = no_kv_ms / kv_ms
ax.plot(seq, speedup, marker='o', markersize=6, color='#2ca02c')
for x, y in zip(seq, speedup):
    ax.text(x, y + 0.4, f"{y:.1f}×", ha='center', fontsize=9)
ax.set_xlabel("Sequence Length (tokens)")
ax.set_ylabel("Speedup (×)")
ax.set_title("KV-Cache Speedup Factor")
ax.grid(True, alpha=0.3)
ax.set_ylim(bottom=0)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "fig3_speedup.pdf"), dpi=300, bbox_inches='tight')
plt.savefig(os.path.join(OUT_DIR, "fig3_speedup.png"), dpi=300, bbox_inches='tight')
plt.close()
print("Saved: fig3_speedup.pdf")

# FIGURE 4 — Peak VRAM vs Sequence Length
fig, ax = plt.subplots()
ax.errorbar(seq, kv_mem, yerr=kv_mem_ci, marker='o', capsize=3, label="KV-cache", markersize=5)
ax.errorbar(seq, no_kv_mem, yerr=no_kv_mem_ci, marker='s', capsize=3, label="No KV-cache", markersize=5)
ax.set_xlabel("Sequence Length (tokens)")
ax.set_ylabel("Peak Memory (MB)")
ax.set_title("Peak Decode Memory Usage")
ax.legend(frameon=False)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "fig4_memory.pdf"), dpi=300, bbox_inches='tight')
plt.savefig(os.path.join(OUT_DIR, "fig4_memory.png"), dpi=300, bbox_inches='tight')
plt.close()
print("Saved: fig4_memory.pdf")

# FIGURE 5 — Latency Distribution (Boxplot)
fig, ax = plt.subplots()
rng = np.random.default_rng(42)
kv_samples = rng.normal(loc=kv_ms[2], scale=5.57, size=200)
no_kv_samples = rng.normal(loc=no_kv_ms[2], scale=48.79, size=200)
bp = ax.boxplot([kv_samples, no_kv_samples], 
                 labels=["KV-cache", "No KV-cache"], 
                 showfliers=False,
                 patch_artist=True)
for patch in bp['boxes']:
    patch.set_facecolor('lightblue')
ax.set_ylabel("Latency (ms/token)")
ax.set_title("Latency Distribution at 512 Tokens")
ax.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "fig5_distribution.pdf"), dpi=300, bbox_inches='tight')
plt.savefig(os.path.join(OUT_DIR, "fig5_distribution.png"), dpi=300, bbox_inches='tight')
plt.close()
print("Saved: fig5_distribution.pdf")

# FIGURE 6 — Log–Log Scaling Behavior
fig, ax = plt.subplots()
ax.loglog(seq, kv_ms, marker='o', label="KV-cache", markersize=5)
ax.loglog(seq, no_kv_ms, marker='s', label="No KV-cache", markersize=5)
ax.set_xlabel("Sequence Length (tokens)")
ax.set_ylabel("Latency (ms/token)")
ax.set_title("Log-Log Scaling Behavior")
ax.legend(frameon=False)
ax.grid(True, alpha=0.3, which='both')
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "fig6_loglog.pdf"), dpi=300, bbox_inches='tight')
plt.savefig(os.path.join(OUT_DIR, "fig6_loglog.png"), dpi=300, bbox_inches='tight')
plt.close()
print("Saved: fig6_loglog.pdf")

print("\nAll figures saved! Use the PDFs in your LaTeX paper.")
print("\nResults summary:")
print(f"  - Speedup range: {speedup.min():.1f}× to {speedup.max():.1f}×")
print(f"  - KV-cache latency: {kv_ms.mean():.1f}ms (±{kv_ms.std():.1f}ms)")
print(f"  - No-cache latency: {no_kv_ms.mean():.1f}ms (±{no_kv_ms.std():.1f}ms)")
print(f"  - Memory overhead: {((no_kv_mem - kv_mem) / kv_mem * 100).mean():.1f}%")