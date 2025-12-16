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
    bars = plt.bar(configs, latencies, color='steelblue', alpha=0.8)
    
    # Highlight baseline
    if "baseline" in configs:
        baseline_idx = configs.index("baseline")
        bars[baseline_idx].set_color('coral')
    
    plt.xlabel('Configuration', fontsize=12)
    plt.ylabel('Average Latency (ms)', fontsize=12)
    plt.title('Inference Latency Across Configurations', fontsize=14, fontweight='bold')
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/latency_comparison.png", dpi=300)
    print(f"✓ Saved: {OUTPUT_DIR}/latency_comparison.png")
    plt.close()

def plot_latency_per_token(summary):
    """Bar chart for latency per token"""
    configs = list(summary.keys())
    latencies_per_token = [summary[c]["avg_latency_per_token_ms"] for c in configs]
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(configs, latencies_per_token, color='mediumseagreen', alpha=0.8)
    
    if "baseline" in configs:
        baseline_idx = configs.index("baseline")
        bars[baseline_idx].set_color('coral')
    
    plt.xlabel('Configuration', fontsize=12)
    plt.ylabel('Latency per Token (ms)', fontsize=12)
    plt.title('Token Generation Speed Across Configurations', fontsize=14, fontweight='bold')
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/latency_per_token.png", dpi=300)
    print(f"✓ Saved: {OUTPUT_DIR}/latency_per_token.png")
    plt.close()

def plot_memory_usage(summary):
    """Bar chart for memory usage"""
    configs = list(summary.keys())
    memory = [summary[c]["avg_memory_mb"] for c in configs]
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(configs, memory, color='mediumpurple', alpha=0.8)
    
    if "baseline" in configs:
        baseline_idx = configs.index("baseline")
        bars[baseline_idx].set_color('coral')
    
    plt.xlabel('Configuration', fontsize=12)
    plt.ylabel('Memory Usage (MB)', fontsize=12)
    plt.title('GPU Memory Usage Across Configurations', fontsize=14, fontweight='bold')
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/memory_usage.png", dpi=300)
    print(f"✓ Saved: {OUTPUT_DIR}/memory_usage.png")
    plt.close()

def plot_speedup_comparison(summary):
    """Speedup relative to baseline"""
    if "baseline" not in summary:
        print("⚠ No baseline found, skipping speedup plot")
        return
    
    baseline_latency = summary["baseline"]["avg_latency_ms"]
    
    configs = [c for c in summary.keys() if c != "baseline"]
    speedups = [baseline_latency / summary[c]["avg_latency_ms"] for c in configs]
    
    plt.figure(figsize=(10, 6))
    colors = ['green' if s > 1 else 'red' for s in speedups]
    bars = plt.bar(configs, speedups, color=colors, alpha=0.7)
    
    plt.axhline(y=1.0, color='black', linestyle='--', linewidth=1, label='Baseline')
    plt.xlabel('Configuration', fontsize=12)
    plt.ylabel('Speedup (relative to baseline)', fontsize=12)
    plt.title('Speedup Comparison (>1 is faster, <1 is slower)', fontsize=14, fontweight='bold')
    plt.xticks(rotation=45, ha='right')
    plt.legend()
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/speedup_comparison.png", dpi=300)
    print(f"✓ Saved: {OUTPUT_DIR}/speedup_comparison.png")
    plt.close()

def plot_tradeoff_scatter(summary):
    """Scatter plot: latency vs memory"""
    configs = list(summary.keys())
    latencies = [summary[c]["avg_latency_ms"] for c in configs]
    memory = [summary[c]["avg_memory_mb"] for c in configs]
    
    plt.figure(figsize=(10, 8))
    plt.scatter(latencies, memory, s=200, alpha=0.6, c='steelblue', edgecolors='black')
    
    # Annotate points
    for i, config in enumerate(configs):
        plt.annotate(config, (latencies[i], memory[i]), 
                    xytext=(5, 5), textcoords='offset points', fontsize=9)
    
    plt.xlabel('Average Latency (ms)', fontsize=12)
    plt.ylabel('Memory Usage (MB)', fontsize=12)
    plt.title('Latency vs Memory Tradeoff', fontsize=14, fontweight='bold')
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/tradeoff_scatter.png", dpi=300)
    print(f"✓ Saved: {OUTPUT_DIR}/tradeoff_scatter.png")
    plt.close()

def generate_latex_table(summary):
    """Generate LaTeX table for paper"""
    latex = "\\begin{table}[h]\n\\centering\n\\begin{tabular}{lcccc}\n\\hline\n"
    latex += "Configuration & Latency (ms) & Tokens & ms/token & Memory (MB) \\\\\n\\hline\n"
    
    for config, data in summary.items():
        latex += f"{config} & {data['avg_latency_ms']:.1f} & {data['avg_tokens']:.1f} & "
        latex += f"{data['avg_latency_per_token_ms']:.2f} & {data['avg_memory_mb']:.1f} \\\\\n"
    
    latex += "\\hline\n\\end{tabular}\n"
    latex += "\\caption{Inference performance across different configurations}\n"
    latex += "\\label{tab:results}\n\\end{table}"
    
    with open(f"{OUTPUT_DIR}/results_table.tex", "w") as f:
        f.write(latex)
    
    print(f"✓ Saved: {OUTPUT_DIR}/results_table.tex")

def generate_markdown_report(results, summary):
    """Generate markdown report"""
    md = "# PaliGemma Ablation Study Results\n\n"
    md += f"**Date**: {results[0]['config_name'] if results else 'Unknown'}\n\n"
    md += "## Summary Statistics\n\n"
    md += "| Configuration | Avg Latency (ms) | Avg Tokens | ms/token | Memory (MB) |\n"
    md += "|--------------|------------------|------------|----------|-------------|\n"
    
    for config, data in summary.items():
        md += f"| {config} | {data['avg_latency_ms']:.1f} | {data['avg_tokens']:.1f} | "
        md += f"{data['avg_latency_per_token_ms']:.2f} | {data['avg_memory_mb']:.1f} |\n"
    
    md += "\n## Key Findings\n\n"
    
    if "baseline" in summary and "no_kv_cache" in summary:
        speedup = summary["no_kv_cache"]["avg_latency_ms"] / summary["baseline"]["avg_latency_ms"]
        md += f"- **KV-cache speedup**: {speedup:.2f}x faster with caching enabled\n"
    
    if "baseline" in summary and "fp32" in summary:
        speedup = summary["fp32"]["avg_latency_ms"] / summary["baseline"]["avg_latency_ms"]
        md += f"- **FP16 speedup**: {speedup:.2f}x faster than FP32\n"
    
    md += "\n## Individual Results\n\n"
    
    for result in results[:10]:  # Show first 10
        md += f"### {result['config_name']} - Image {result['image_idx']}\n"
        md += f"- **Prompt**: {result['prompt']}\n"
        md += f"- **Output**: {result['output']}\n"
        md += f"- **Latency**: {result['latency_ms']:.1f} ms ({result['tokens_generated']} tokens)\n\n"
    
    with open(f"{OUTPUT_DIR}/report.md", "w") as f:
        f.write(md)
    
    print(f"✓ Saved: {OUTPUT_DIR}/report.md")

def main():
    print("="*80)
    print("GENERATING VISUALIZATIONS")
    print("="*80)
    
    # Load results
    print("\nLoading results...")
    results, summary = load_results()
    print(f"✓ Loaded {len(results)} results across {len(summary)} configurations\n")
    
    # Generate plots
    print("Creating plots...")
    plot_latency_comparison(summary)
    plot_latency_per_token(summary)
    plot_memory_usage(summary)
    plot_speedup_comparison(summary)
    plot_tradeoff_scatter(summary)
    
    # Generate tables
    print("\nGenerating tables...")
    generate_latex_table(summary)
    generate_markdown_report(results, summary)
    
    print("\n" + "="*80)
    print("VISUALIZATION COMPLETE!")
    print(f"All files saved to: {OUTPUT_DIR}/")
    print("="*80)
    print("\nGenerated files:")
    print("  📊 latency_comparison.png")
    print("  📊 latency_per_token.png")
    print("  📊 memory_usage.png")
    print("  📊 speedup_comparison.png")
    print("  📊 tradeoff_scatter.png")
    print("  📄 results_table.tex (for LaTeX)")
    print("  📄 report.md (markdown summary)")

if __name__ == "__main__":
    main()