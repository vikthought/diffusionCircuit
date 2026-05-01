"""
Plotting Utilities
==================

Consolidated plotting functions for the diffusionCircuit pipeline.
"""

from pathlib import Path
from typing import Dict, List, Optional, Union, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.patches as mpatches
import networkx as nx

# Visual Constants
EXCITATORY_COLOR = "#E63946"  # Red
INHIBITORY_COLOR = "#457B9D"  # Blue
NEUTRAL_COLOR = "#A8DADC"     # Light teal
EDGE_ALPHA = 0.7


def plot_connectome_heatmaps(
    matrices: Dict[str, np.ndarray],
    nodes: List[str],
    output_dir: Path,
) -> None:
    """
    Generate heatmap visualizations of connectome matrices.
    """
    for name, matrix in matrices.items():
        fig, ax = plt.subplots(figsize=(12, 10))
        
        # Use log scale for better visualization
        plot_matrix = np.log10(matrix + 1)
        
        sns.heatmap(
            plot_matrix,
            ax=ax,
            cmap="viridis",
            cbar_kws={"label": "log₁₀(weight + 1)"},
            xticklabels=False,
            yticklabels=False,
        )
        
        ax.set_title(f"{name} Connectome (Cook et al. 2019)", fontsize=14)
        ax.set_xlabel("Pre-synaptic neuron", fontsize=12)
        ax.set_ylabel("Post-synaptic neuron", fontsize=12)
        
        # Add edge count annotation
        edge_count = int((matrix > 0).sum())
        ax.text(
            0.02, 0.98,
            f"Edges: {edge_count}",
            transform=ax.transAxes,
            fontsize=10,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8)
        )
        
        plt.tight_layout()
        
        filename = output_dir / f"{name.lower().replace(' ', '_')}_heatmap.png"
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"  Saved: {filename.name}")


def create_network_graph(
    adj: np.ndarray,
    node_names: List[str],
    title: str,
    ax: plt.Axes,
    show_legend: bool = True
) -> None:
    """
    Create network visualization with inhibitory/excitatory coloring.
    """
    n = adj.shape[0]
    
    # Create NetworkX graph
    G = nx.DiGraph()
    G.add_nodes_from(range(n))
    
    excitatory_edges = []
    inhibitory_edges = []
    
    for i in range(n):
        for j in range(n):
            if adj[i, j] > 0:
                excitatory_edges.append((i, j))
            elif adj[i, j] < 0:
                inhibitory_edges.append((i, j))
    
    G.add_edges_from(excitatory_edges + inhibitory_edges)
    
    # Layout
    if n <= 30:
        pos = nx.circular_layout(G)
    else:
        # Use seed for reproducibility
        pos = nx.spring_layout(G, k=2/np.sqrt(n), iterations=50, seed=42)
    
    # Draw nodes
    nx.draw_networkx_nodes(G, pos, ax=ax, node_size=50, node_color='lightgray', 
                           edgecolors='black', linewidths=0.5)
    
    # Draw edges
    if excitatory_edges:
        nx.draw_networkx_edges(G, pos, edgelist=excitatory_edges, ax=ax,
                               edge_color=EXCITATORY_COLOR, alpha=EDGE_ALPHA,
                               width=0.5, arrows=True, arrowsize=5,
                               connectionstyle="arc3,rad=0.1")
    
    if inhibitory_edges:
        nx.draw_networkx_edges(G, pos, edgelist=inhibitory_edges, ax=ax,
                               edge_color=INHIBITORY_COLOR, alpha=EDGE_ALPHA,
                               width=0.5, arrows=True, arrowsize=5,
                               connectionstyle="arc3,rad=0.1")
    
    ax.set_title(title, fontsize=10, fontweight='bold')
    ax.axis('off')
    
    # Legend
    if show_legend:
        excit_patch = mpatches.Patch(color=EXCITATORY_COLOR, label=f'Excitatory ({len(excitatory_edges)})')
        inhib_patch = mpatches.Patch(color=INHIBITORY_COLOR, label=f'Inhibitory ({len(inhibitory_edges)})')
        ax.legend(handles=[excit_patch, inhib_patch], loc='upper right', fontsize=7,
                  framealpha=0.9)


def create_phase_grid_figure(
    phase_adjacencies: Dict[str, np.ndarray],
    node_names: Optional[List[str]],
    output_path: Path
) -> None:
    """
    Create grid figure showing connectivity graphs for each phase.
    """
    phases = list(phase_adjacencies.keys())
    n_phases = len(phases)
    
    # Determine grid size
    n_cols = min(4, n_phases)
    n_rows = (n_phases + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows))
    
    if n_rows == 1 and n_cols == 1:
        axes = np.array([[axes]])
    elif n_rows == 1:
        axes = axes.reshape(1, -1)
    elif n_cols == 1:
        axes = axes.reshape(-1, 1)
    
    # Create consistent layout using first adjacency
    first_adj = list(phase_adjacencies.values())[0]
    n = first_adj.shape[0]
    
    # Use circular layout for consistency
    G_template = nx.DiGraph()
    G_template.add_nodes_from(range(n))
    pos = nx.circular_layout(G_template)
    
    for idx, (phase_name, adj) in enumerate(phase_adjacencies.items()):
        row = idx // n_cols
        col = idx % n_cols
        ax = axes[row, col]
        
        # Count edges
        n_excit = (adj > 0).sum()
        n_inhib = (adj < 0).sum()
        
        # Create graph
        G = nx.DiGraph()
        G.add_nodes_from(range(n))
        
        excitatory_edges = []
        inhibitory_edges = []
        
        for i in range(n):
            for j in range(n):
                if adj[i, j] > 0:
                    excitatory_edges.append((i, j))
                elif adj[i, j] < 0:
                    inhibitory_edges.append((i, j))
        
        # Draw
        nx.draw_networkx_nodes(G, pos, ax=ax, node_size=30, node_color='lightgray',
                               edgecolors='black', linewidths=0.3)
        
        if excitatory_edges:
            G.add_edges_from(excitatory_edges)
            nx.draw_networkx_edges(G, pos, edgelist=excitatory_edges, ax=ax,
                                   edge_color=EXCITATORY_COLOR, alpha=0.5,
                                   width=0.3, arrows=True, arrowsize=3,
                                   connectionstyle="arc3,rad=0.1")
        
        if inhibitory_edges:
            G.add_edges_from(inhibitory_edges)
            nx.draw_networkx_edges(G, pos, edgelist=inhibitory_edges, ax=ax,
                                   edge_color=INHIBITORY_COLOR, alpha=0.5,
                                   width=0.3, arrows=True, arrowsize=3,
                                   connectionstyle="arc3,rad=0.1")
        
        ax.set_title(f"{phase_name}\n({n_excit} excit, {n_inhib} inhib)", 
                     fontsize=9, fontweight='bold')
        ax.axis('off')
        
        # Legend in first panel only
        if idx == 0:
            excit_patch = mpatches.Patch(color=EXCITATORY_COLOR, label='Excitatory (+)')
            inhib_patch = mpatches.Patch(color=INHIBITORY_COLOR, label='Inhibitory (-)')
            ax.legend(handles=[excit_patch, inhib_patch], loc='upper right', fontsize=7,
                      framealpha=0.9)
    
    # Hide unused axes
    for idx in range(n_phases, n_rows * n_cols):
        row = idx // n_cols
        col = idx % n_cols
        axes[row, col].axis('off')
    
    plt.suptitle("Connectivity Graphs Across Phases\n(Excitatory=Red, Inhibitory=Blue)", 
                 fontsize=12, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()


def create_difference_heatmap(
    adj1: np.ndarray,
    adj2: np.ndarray,
    name1: str,
    name2: str,
    output_path: Path
) -> None:
    """Create heatmap showing difference between two adjacency matrices."""
    diff = adj2 - adj1
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # First adjacency
    im1 = axes[0].imshow(adj1, cmap='RdBu_r', vmin=-1, vmax=1)
    axes[0].set_title(f'{name1}', fontsize=10)
    axes[0].axis('off')
    plt.colorbar(im1, ax=axes[0], fraction=0.046)
    
    # Second adjacency
    im2 = axes[1].imshow(adj2, cmap='RdBu_r', vmin=-1, vmax=1)
    axes[1].set_title(f'{name2}', fontsize=10)
    axes[1].axis('off')
    plt.colorbar(im2, ax=axes[1], fraction=0.046)
    
    # Difference
    max_diff = max(abs(diff.min()), abs(diff.max()))
    im3 = axes[2].imshow(diff, cmap='RdBu_r', vmin=-max_diff, vmax=max_diff)
    axes[2].set_title(f'Difference ({name2} - {name1})', fontsize=10)
    axes[2].axis('off')
    plt.colorbar(im3, ax=axes[2], fraction=0.046)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def create_strategy_comparison_plot(
    metrics_df: pd.DataFrame,
    output_dir: Path
) -> None:
    """
    Create bar plot comparing F1 scores across strategies using metrics dataframe.
    """
    # Filter for known strategies and create display labels
    df = metrics_df.copy()
    
    strategy_map = {
        "global": "Global",
        "stimulus_only": "Stimulus Only",
        "global_finetuned": "Global + Finetune"
    }
    
    # Only keep main comparison strategies
    df = df[df["strategy"].isin(strategy_map.keys())]
    if df.empty:
        return
        
    df["Strategy Display"] = df["strategy"].map(strategy_map)
    df["Condition Display"] = df["condition"].apply(lambda x: x.replace("_", " ").capitalize())
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Color by strategy display
    colors = {
        "Global": "#2E86AB", 
        "Stimulus Only": "#A23B72", 
        "Global + Finetune": "#F18F01"
    }
    
    x = np.arange(len(df))
    # Safe color mapping
    bar_colors = [colors.get(s, "#333333") for s in df["Strategy Display"]]
    
    bars = ax.bar(x, df["f1_score"], color=bar_colors)
    
    ax.set_xticks(x)
    labels = [f"{row['Condition Display']}\n({row['Strategy Display']})" for _, row in df.iterrows()]
    ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=9)
    
    ax.set_ylabel("F1 Score vs Structural Connectome")
    ax.set_title("Comparison of Training Strategies", fontsize=12, fontweight='bold')
    
    if not df.empty:
        ax.set_ylim(0, df["f1_score"].max() * 1.2)
    
    # Add value labels
    for bar, val in zip(bars, df["f1_score"]):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                f'{val:.3f}', ha='center', va='bottom', fontsize=8)
    
    # Legend
    handles = [mpatches.Patch(color=c, label=s) for s, c in colors.items() if s in df["Strategy Display"].unique()]
    ax.legend(handles=handles, loc='upper right')
    
    plt.tight_layout()
    plt.savefig(output_dir / "strategy_comparison.png", dpi=150, bbox_inches='tight')
    plt.close()
