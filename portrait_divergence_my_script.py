    #     #!/usr/bin/env python
# # -*- coding: utf-8 -*-

# portrait_divergence_visualization.py
# Modified to include portrait plots and divergence heatmaps

import sys, os
import argparse
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import entropy
from collections import Counter

def portrait_py(graph):
    """Return matrix B where B[i,j] is the number of starting nodes in graph
    with j nodes in shell i.
    """
    dia = 500
    N = graph.number_of_nodes()
    B = np.zeros((dia+1,N)) 
    
    max_path = 1
    adj = graph.adj
    for starting_node in graph.nodes():
        nodes_visited = {starting_node:0}
        search_queue = [starting_node]
        d = 1
        while search_queue:
            next_depth = []
            extend = next_depth.extend
            for n in search_queue:
                l = [i for i in adj[n] if i not in nodes_visited] 
                extend(l)
                for j in l:
                    nodes_visited[j] = d
            search_queue = next_depth
            d += 1
            
        node_distances = nodes_visited.values()
        max_node_distances = max(node_distances)
        
        curr_max_path = max_node_distances
        if curr_max_path > max_path:
            max_path = curr_max_path
        
        dict_distribution = dict.fromkeys(node_distances, 0)
        for d in node_distances:
            dict_distribution[d] += 1
        for shell,count in dict_distribution.items():
            B[shell][count] += 1
        
        max_shell = dia
        while max_shell > max_node_distances:
            B[max_shell][0] += 1
            max_shell -= 1
    
    return B[:max_path+1,:]


def pad_portraits_to_same_size(B1,B2):
    """Make sure that two matrices are padded with zeros and/or trimmed of
    zeros to be the same dimensions.
    """
    ns,ms = B1.shape
    nl,ml = B2.shape
    
    lastcol1 = max(np.nonzero(B1)[1])
    lastcol2 = max(np.nonzero(B2)[1])
    lastcol = max(lastcol1,lastcol2)
    B1 = B1[:,:lastcol+1]
    B2 = B2[:,:lastcol+1]
    
    BigB1 = np.zeros((max(ns,nl), lastcol+1))
    BigB2 = np.zeros((max(ns,nl), lastcol+1))
    
    BigB1[:B1.shape[0],:B1.shape[1]] = B1
    BigB2[:B2.shape[0],:B2.shape[1]] = B2
    
    return BigB1, BigB2


def portrait_divergence(BG, BH):
    """Compute the network portrait divergence between two portraits."""
    BG, BH = pad_portraits_to_same_size(BG, BH)
    
    L, K = BG.shape
    V = np.tile(np.arange(K),(L,1))
    
    XG = BG*V / (BG*V).sum()
    XH = BH*V / (BH*V).sum()
    
    P = XG.ravel()
    Q = XH.ravel()
    
    M = 0.5*(P+Q)
    KLDpm = entropy(P, M, base=2)
    KLDqm = entropy(Q, M, base=2)
    JSDpq = 0.5*(KLDpm + KLDqm)
    
    return JSDpq


def plot_portrait(portrait_matrix, title, output_file):
    """Plot a single network portrait."""
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Only plot non-zero portions
    nonzero_rows = np.any(portrait_matrix != 0, axis=1)
    nonzero_cols = np.any(portrait_matrix != 0, axis=0)
    trimmed = portrait_matrix[nonzero_rows][:, nonzero_cols]
    
    im = ax.imshow(trimmed, aspect='auto', cmap='viridis', interpolation='nearest')
    ax.set_xlabel('Number of nodes at distance k')
    ax.set_ylabel('Distance (shell number)')
    ax.set_title(title)
    plt.colorbar(im, ax=ax, label='Count')
    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()


def compute_all_portraits(networks_array):
    """Compute portraits for all networks in the array."""
    n_networks = networks_array.shape[0]
    portraits = []
    
    print(f"Computing portraits for {n_networks} networks...")
    for i in range(n_networks):
        network = networks_array[i, :, :]
        G = nx.from_numpy_array(network)
        B = portrait_py(G)
        portraits.append(B)
        if (i + 1) % 10 == 0:
            print(f"  Processed {i + 1}/{n_networks} networks")
    
    return portraits


def compute_divergence_matrix(portraits):
    """Compute pairwise portrait divergences."""
    n = len(portraits)
    divergence_matrix = np.zeros((n, n))
    
    print(f"Computing {n}x{n} divergence matrix...")
    for i in range(n):
        for j in range(i, n):
            if i == j:
                divergence_matrix[i, j] = 0
            else:
                div = portrait_divergence(portraits[i], portraits[j])
                divergence_matrix[i, j] = div
                divergence_matrix[j, i] = div
        if (i + 1) % 10 == 0:
            print(f"  Processed {i + 1}/{n} rows")
    
    return divergence_matrix


def plot_divergence_heatmap(divergence_matrix, labels, output_file, metadata_df=None, 
                           color_by=None, draw_lines_by=None):
    """Plot heatmap of portrait divergences with optional coloring and separation lines."""
    
    # Phylogenetic ordering columns in hierarchical order
    phylo_columns = ['phylogenetic_group', 'super_order', 'order', 'sub_order', 
                     'family', 'sub_family', 'genus', 'species']
    
    # Sort by phylogenetic hierarchy
    if metadata_df is not None:
        # Use available phylogenetic columns for sorting
        sort_columns = [col for col in phylo_columns if col in metadata_df.columns]
        if sort_columns:
            metadata_sorted = metadata_df.sort_values(by=sort_columns)
            sort_order = metadata_sorted.index.values
            divergence_matrix = divergence_matrix[sort_order][:, sort_order]
            labels = [labels[i] for i in sort_order]
            metadata_df = metadata_sorted.reset_index(drop=True)
            print(f"Heatmap ordered by phylogenetic hierarchy: {', '.join(sort_columns)}")
    
    fig, ax = plt.subplots(figsize=(18, 16))
    
    # Create the heatmap
    sns.heatmap(divergence_matrix, 
                xticklabels=labels, 
                yticklabels=labels,
                cmap='Blues_r', # viridis',
                square=True,
                cbar_kws={'label': 'Portrait Divergence'},
                ax=ax)
    
    # Color tick labels by specified column
    if color_by and metadata_df is not None and color_by in metadata_df.columns:
        unique_groups = metadata_df[color_by].unique()
        colors = plt.cm.tab20(np.linspace(0, 1, len(unique_groups)))
        color_map = dict(zip(unique_groups, colors))
        
        for idx, (label, group) in enumerate(zip(ax.get_yticklabels(), metadata_df[color_by])):
            label.set_color(color_map[group])
        for idx, (label, group) in enumerate(zip(ax.get_xticklabels(), metadata_df[color_by])):
            label.set_color(color_map[group])
        
        print(f"Colored labels by: {color_by}")
    
    # Draw separation lines between groups
    if draw_lines_by and metadata_df is not None and draw_lines_by in metadata_df.columns:
        boundaries = []
        prev_group = None
        for idx, group in enumerate(metadata_df[draw_lines_by]):
            if prev_group is not None and group != prev_group:
                boundaries.append(idx)
            prev_group = group
        
        for boundary in boundaries:
            color="white"
            ax.axhline(y=boundary, color=color, linewidth=2, linestyle='--', alpha=0.8)
            ax.axvline(x=boundary, color=color, linewidth=2, linestyle='--', alpha=0.8)

        print(f"Drew {len(boundaries)} separation lines by: {draw_lines_by}")
    
    ax.set_title('Network Portrait Divergence Matrix', fontsize=16, pad=20)
    fontsize = 5
    plt.xticks(rotation=90, ha='right', fontsize=fontsize)
    plt.yticks(rotation=0, fontsize=fontsize)
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved heatmap to {output_file}")


def main():
    BASE_PATH = "/Users/adrian/Documents/01_projects/14_4D_lab/14_4D_lab_code/"
    
    # data_type = "01_connectomes/01_consensus_bin_density_10_percent_50" 
    # data_type = "01_connectomes/00_connectomes_50" 
    # data_type = "01_connectomes/00_connectomes_orig" 
    # data_type = "01_connectomes/01_consensus_bin_density_10_percent_50"
    data_type = "02_distance_matrices/distance_matrix_50"
    # data_type = "02_distance_matrices/distance_matrix_orig"
    
    CONNECTOMES = BASE_PATH + f"data/preprocessed/suarez_MaMI_dataset/{data_type}.npy"
    METADATA = BASE_PATH + "data/preprocessed/suarez_MaMI_dataset/04_further_info/names_of_animals_with_preprocessed_connectomes_50.csv"
    OUTPUT_DIR = BASE_PATH + f"output/data_analysis/suarez_MaMI_dataset/network_portrait_divergence_{data_type.split("/")[-1]}/"
    # OUTPUT_DIR = BASE_PATH + f"output/empirical_analysis/network_portrait_divergence/network_portrait_divergence_{data_type}/"
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    parser = argparse.ArgumentParser(description='Compute and visualize network portrait divergences')
    parser.add_argument('--networks_file', default=CONNECTOMES, help='Path to .npy file with shape (n_networks, 100, 100)')
    parser.add_argument('--metadata_file', default=METADATA, help='Path to CSV file with metadata')
    parser.add_argument('--output_dir', default=OUTPUT_DIR, help='Output directory for plots')
    parser.add_argument('--label_column', default='common_name', help='Column to use for labels')
    parser.add_argument('--order_by', default=None, help='Column to order heatmap by (e.g., "order", "family")')
    parser.add_argument('--skip_individual_plots', 
                        default=False,  # True, 
                        action='store_true',
                        help='Skip individual portrait plots')
    parser.add_argument('--color_by', default="order", help='Column to color heatmap by (e.g., "order", "family")')
    parser.add_argument('--draw_lines_by', default="order", help='Column to draw separation lines by (e.g., "order", "family")')

    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load data
    print("Loading networks...")
    networks = np.load(args.networks_file)
    print(f"Loaded networks with shape: {networks.shape}")
    
    print("Loading metadata...")
    metadata = pd.read_csv(args.metadata_file)
    print(f"Loaded metadata with {len(metadata)} entries")
    
    # Get labels
    if args.label_column in metadata.columns:
        labels = metadata[args.label_column].tolist()
    else:
        print(f"Warning: '{args.label_column}' not found. Using indices.")
        labels = [f"Network_{i}" for i in range(len(networks))]
    
    # Compute portraits
    portraits = compute_all_portraits(networks)
    
    # Plot individual portraits
    if not args.skip_individual_plots:
        print("\nGenerating individual portrait plots...")
        portraits_dir = os.path.join(args.output_dir, 'individual_portraits')
        os.makedirs(portraits_dir, exist_ok=True)
        
        for i, (portrait, label) in enumerate(zip(portraits, labels)):
            output_file = os.path.join(portraits_dir, f'portrait_{i:03d}_{label}.png')
            plot_portrait(portrait, f'Portrait: {label}', output_file)
            if (i + 1) % 10 == 0:
                print(f"  Saved {i + 1}/{len(portraits)} plots")
    
    # Compute divergence matrix
    print("\nComputing divergence matrix...")
    divergence_matrix = compute_divergence_matrix(portraits)
    
    # Save divergence matrix
    divergence_file = os.path.join(args.output_dir, 'divergence_matrix.npy')
    np.save(divergence_file, divergence_matrix)
    print(f"Saved divergence matrix to {divergence_file}")
    
    # Plot heatmap with phylogenetic ordering and coloring
    print("\nGenerating divergence heatmap...")
    heatmap_file = os.path.join(args.output_dir, 'divergence_heatmap.png')
    plot_divergence_heatmap(divergence_matrix, labels, heatmap_file, 
                           metadata_df=metadata,
                           color_by=args.color_by,
                           draw_lines_by=args.draw_lines_by)
    
    # If order_by specified, also save ordered version as CSV
    if args.order_by and args.order_by in metadata.columns:
        sort_order = metadata[args.order_by].argsort()
        ordered_labels = [labels[i] for i in sort_order]
        ordered_matrix = divergence_matrix[sort_order][:, sort_order]
        
        df_divergence = pd.DataFrame(ordered_matrix, 
                                     index=ordered_labels, 
                                     columns=ordered_labels)
        csv_file = os.path.join(args.output_dir, f'divergence_matrix_ordered_by_{args.order_by}.csv')
        df_divergence.to_csv(csv_file)
        print(f"Saved ordered divergence matrix to {csv_file}")
    
    print("\nDone!")


if __name__ == '__main__':
    main()