# visualize_results.py
# Run AFTER ablation_study.py completes
# Creates publication-ready plots

import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

OUTPUT_DIR = "ablation_results"

def load_results():
    """Load results from JSON"""
    with open(f"{OUTPUT_DIR}/results.json", "r") as f:
        results = json.load(f)
    with open(f"{OUTPUT_DIR}/summary.json", "r") as f:
        summary = json.load(f)
    return results, summary

def plot_latency_comparison(summary):
    """Bar chart comparing latency across configs"""
    configs = list(summary.keys())
    latencies = [summary[c]["avg_latency_ms"] for c in configs]
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(configs, latencies, color='steelblue', alpha=0.8, edgecolor='black', linewidth=1.2)
    
    # Highlight baseline and no_kv_cache
    if "baseline" in configs:
        baseline_idx = configs.index("baseline")
        bars[baseline_idx].set_color('coral')
    if "no_kv_cache" in configs:
        no_cache_idx = configs.index("no_kv_cache")
        bars[no_cache_idx].set_color('crimson')
    
    plt.xlabel('Configuration', fontsize=13, fontweight='bold')
    plt.ylabel('Average Latency (ms)', fontsize=13, fontweight='bold')
    plt.title('Inference Latency Across Configurations', fontsize=15, fontweight='bold')
    plt.xticks(rotation=45, ha='right', fontsize=11)
    plt.yticks(fontsize=11)
    plt.grid(axis='y', alpha=0.3, linestyle='--')
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/latency_comparison.png", dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {OUTPUT_DIR}/latency_comparison.png")
    plt.close()

def plot_memory_usage(summary):
    """Bar chart for memory usage"""
    configs = list(summary.keys())
    memory = [summary[c]["avg_memory_mb"] for c in configs]
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(configs, memory, color='mediumseagreen', alpha=0.8, edgecolor='black', linewidth=1.2)
    
    if "baseline" in configs:
        baseline_idx = configs.index("baseline")
        bars[baseline_idx].set_color('coral')
    if "no_kv_cache" in configs:
        no_cache_idx = configs.index("no_kv_cache")
        bars[no_cache_idx].set_color('crimson')
    
    plt.xlabel('Configuration', fontsize=13, fontweight='bold')
    plt.ylabel('Memory Usage (MB)', fontsize=13, fontweight='bold')
    plt.title('GPU Memory Usage Across Configurations', fontsize=15, fontweight='bold')
    plt.xticks(rotation=45, ha='right', fontsize=11)
    plt.yticks(fontsize=11)
    plt.grid(axis='y', alpha=0.3, linestyle='--')
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/memory_usage.png", dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {OUTPUT_DIR}/memory_usage.png")
    plt.close()

def plot_kv_cache_impact(summary):
    """Side-by-side comparison of baseline vs no_kv_cache"""
    if "baseline" not in summary or "no_kv_cache" not in summary:
        print("⚠ Missing baseline or no_kv_cache, skipping comparison")
        return
    
    baseline = summary["baseline"]
    no_cache = summary["no_kv_cache"]
    
    metrics = ['Latency (ms)', 'Memory (MB)']
    baseline_vals = [baseline["avg_latency_ms"], baseline["avg_memory_mb"]]
    no_cache_vals = [no_cache["avg_latency_ms"], no_cache["avg_memory_mb"]]
    
    x = np.arange(len(metrics))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(10, 6))
    bars1 = ax.bar(x - width/2, baseline_vals, width, label='With KV Cache (Baseline)', 
                   color='coral', alpha=0.8, edgecolor='black', linewidth=1.2)
    bars2 = ax.bar(x + width/2, no_cache_vals, width, label='Without KV Cache',
                   color='crimson', alpha=0.8, edgecolor='black', linewidth=1.2)
    
    ax.set_xlabel('Metric', fontsize=13, fontweight='bold')
    ax.set_ylabel('Value', fontsize=13, fontweight='bold')
    ax.set_title('KV Cache Impact on Performance', fontsize=15, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics, fontsize=12)
    ax.legend(fontsize=11, loc='upper left')
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.1f}',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3),
                       textcoords="offset points",
                       ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/kv_cache_comparison.png", dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {OUTPUT_DIR}/kv_cache_comparison.png")
    plt.close()

def plot_temperature_analysis(summary):
    """Compare temperature effects"""
    temp_configs = {k: v for k, v in summary.items() if 'temp' in k or k == 'baseline'}
    
    if len(temp_configs) < 2:
        print("⚠ Not enough temperature configs, skipping")
        return
    
    configs = list(temp_configs.keys())
    latencies = [temp_configs[c]["avg_latency_ms"] for c in configs]
    tokens = [temp_configs[c]["avg_tokens"] for c in configs]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Latency
    ax1.bar(configs, latencies, color='steelblue', alpha=0.8, edgecolor='black', linewidth=1.2)
    ax1.set_xlabel('Configuration', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Avg Latency (ms)', fontsize=12, fontweight='bold')
    ax1.set_title('Latency vs Temperature', fontsize=13, fontweight='bold')
    ax1.tick_params(axis='x', rotation=45)
    ax1.grid(axis='y', alpha=0.3, linestyle='--')
    
    # Tokens
    ax2.bar(configs, tokens, color='orange', alpha=0.8, edgecolor='black', linewidth=1.2)
    ax2.set_xlabel('Configuration', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Avg Tokens Generated', fontsize=12, fontweight='bold')
    ax2.set_title('Token Count vs Temperature', fontsize=13, fontweight='bold')
    ax2.tick_params(axis='x', rotation=45)
    ax2.grid(axis='y', alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/temperature_analysis.png", dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {OUTPUT_DIR}/temperature_analysis.png")
    plt.close()

def generate_latex_table(summary):
    """Generate LaTeX table for paper"""
    latex = "\\begin{table}[ht]\n\\centering\n"
    latex += "\\caption{PaliGemma Inference Performance Across Configurations}\n"
    latex += "\\label{tab:ablation_results}\n"
    latex += "\\begin{tabular}{lcccc}\n\\hline\\hline\n"
    latex += "\\textbf{Configuration} & \\textbf{Latency (ms)} & \\textbf{Tokens} & \\textbf{ms/token} & \\textbf{Memory (MB)} \\\\\n\\hline\n"
    
    for config, data in summary.items():
        ms_per_token = data['avg_latency_ms'] / data['avg_tokens'] if data['avg_tokens'] > 0 else 0
        latex += f"{config.replace('_', '\\_')} & {data['avg_latency_ms']:.1f} & {data['avg_tokens']:.1f} & "
        latex += f"{ms_per_token:.1f} & {data['avg_memory_mb']:.1f} \\\\\n"
    
    # Add speedup if available
    if "baseline" in summary and "no_kv_cache" in summary:
        speedup = summary["no_kv_cache"]["avg_latency_ms"] / summary["baseline"]["avg_latency_ms"]
        latex += "\\hline\n"
        latex += f"\\multicolumn{{5}}{{l}}{{\\textit{{KV Cache Speedup: {speedup:.2f}x}}}} \\\\\n"
    
    latex += "\\hline\\hline\n\\end{tabular}\n\\end{table}"
    
    with open(f"{OUTPUT_DIR}/results_table.tex", "w") as f:
        f.write(latex)
    
    print(f"✓ Saved: {OUTPUT_DIR}/results_table.tex")

def generate_paper_summary(summary):
    """Generate text summary for paper"""
    summary_text = "# ABLATION STUDY SUMMARY FOR PAPER\n\n"
    summary_text += "## Key Findings\n\n"
    
    if "baseline" in summary and "no_kv_cache" in summary:
        baseline_lat = summary["baseline"]["avg_latency_ms"]
        no_cache_lat = summary["no_kv_cache"]["avg_latency_ms"]
        speedup = no_cache_lat / baseline_lat
        
        baseline_mem = summary["baseline"]["avg_memory_mb"]
        no_cache_mem = summary["no_kv_cache"]["avg_memory_mb"]
        mem_ratio = no_cache_mem / baseline_mem
        
        summary_text += f"**KV Cache Performance:**\n"
        summary_text += f"- Baseline (with KV cache): {baseline_lat:.1f} ms, {baseline_mem:.1f} MB\n"
        summary_text += f"- Without KV cache: {no_cache_lat:.1f} ms, {no_cache_mem:.1f} MB\n"
        summary_text += f"- **Speedup: {speedup:.2f}x faster** with KV caching enabled\n"
        summary_text += f"- **Memory overhead: {mem_ratio:.1f}x more memory** without caching\n\n"
    
    # Temperature analysis
    if "temp_0" in summary and "temp_1" in summary:
        temp0_lat = summary["temp_0"]["avg_latency_ms"]
        temp1_lat = summary["temp_1"]["avg_latency_ms"]
        summary_text += f"**Temperature Impact:**\n"
        summary_text += f"- Greedy decoding (temp=0.0): {temp0_lat:.1f} ms\n"
        summary_text += f"- Sampling (temp=1.0): {temp1_lat:.1f} ms\n"
        summary_text += f"- Difference: {abs(temp1_lat - temp0_lat):.1f} ms ({abs(temp1_lat - temp0_lat)/temp0_lat*100:.1f}% change)\n\n"
    
    summary_text += "## All Configurations\n\n"
    summary_text += "| Configuration | Latency (ms) | Memory (MB) | Tokens |\n"
    summary_text += "|--------------|--------------|-------------|--------|\n"
    
    for config, data in sorted(summary.items(), key=lambda x: x[1]['avg_latency_ms']):
        summary_text += f"| {config} | {data['avg_latency_ms']:.1f} | {data['avg_memory_mb']:.1f} | {data['avg_tokens']:.1f} |\n"
    
    with open(f"{OUTPUT_DIR}/paper_summary.txt", "w") as f:
        f.write(summary_text)
    
    print(f"✓ Saved: {OUTPUT_DIR}/paper_summary.txt")
    
    # Print to console
    print("\n" + "="*80)
    print(summary_text)
    print("="*80)

def main():
    print("="*80)
    print("GENERATING PAPER-READY VISUALIZATIONS")
    print("="*80)
    
    # Load results
    print("\nLoading results...")
    results, summary = load_results()
    print(f"✓ Loaded {len(results)} results across {len(summary)} configurations\n")
    
    # Generate plots
    print("Creating publication-quality plots...")
    plot_latency_comparison(summary)
    plot_memory_usage(summary)
    plot_kv_cache_impact(summary)
    plot_temperature_analysis(summary)
    
    # Generate tables
    print("\nGenerating paper tables...")
    generate_latex_table(summary)
    generate_paper_summary(summary)
    
    print("\n" + "="*80)
    print("✅ ALL VISUALIZATIONS COMPLETE!")
    print("="*80)
    print("\nGenerated files for your paper:")
    print("  📊 latency_comparison.png")
    print("  📊 memory_usage.png")
    print("  📊 kv_cache_comparison.png")
    print("  📊 temperature_analysis.png")
    print("  📄 results_table.tex (LaTeX table)")
    print("  📄 paper_summary.txt (key findings)")
    print(f"\n📁 All files in: {OUTPUT_DIR}/")

if __name__ == "__main__":
    main()