import argparse
import os
from collections import defaultdict
from glob import glob
import pickle as pkl

import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
from sklearn.manifold import TSNE
import matplotlib as mpl

# Colormap
CMAP = mpl.colormaps['tab10']

# Colors for each image type (this ensures consistent colors across multiple visualizations)
COLORS = {
    'original': CMAP(0.0),
    'poisoned': CMAP(0.3),
    'target': CMAP(0.7)
}


def parse_args():
    """
    Parse program arguments.

    :return: Program arguments
    """

    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--concept-pairs-file', default='datasets/concept_pairs.csv',
        help='Path to file containing concept pairs.'
    )
    parser.add_argument(
        '--embed-dir',
        help='Directory containing the CLIP embeddings obtained by the parse_images.py script.'
    )
    parser.add_argument(
        '--plot-dir', default='plots',
        help='Directory where the plot shall be saved to.'
    )
    parser.add_argument(
        '--types', nargs='+', default=['original', 'poisoned'],
        help='List of image types, for which the projection shall be calculated.'
    )
    parser.add_argument(
        '--concepts', nargs='+', required=False, default=None,
        help='Filter to plot specific concepts only. If not specified, all concepts will be plotted.'
    )

    return parser.parse_args()


def plot_embeddings(args):
    """
    Plots a 2-dimensional t-SNE projection of CLIP embeddings.

    :param args: Program arguments
    """

    # Gather figure parameters
    nrows = 2  # Number of rows (intra and inter)
    ncols = 5  # Number of columns (5 concepts each)
    figsize = (15, 5.5)  # Figure size
    legend_y = 0.03  # Legend Y offset
    if args.concepts is not None:
        nrows = 1
        ncols = len(args.concepts)
        figsize = (0.5 + 3 * ncols, 3.0)
        legend_y = -0.03

    # Create figure and subplot axes
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, sharex=True, sharey=True)
    axes = axes.T.flatten()
    # fig.suptitle('t-SNE analysis of CLIP embeddings', fontsize=16, y=sup_y)

    # Read concept pairs
    concept_pairs = pd.read_csv(args.concept_pairs_file, index_col='concat_name').to_dict('index')

    # Hold values for embeddings and image types per concept
    embeddings = defaultdict(list)
    embedding_types = defaultdict(list)

    # Iterate over all pickle files containing the CLIP embeddings and meta information
    for path in glob(os.path.join(args.embed_dir, '*.pkl')):
        # Get concept
        concept = os.path.abspath(path)[:-17]
        concept = '_'.join(concept.split('_')[2:])
        print(f'Collecting embeddings for {concept}')

        # Load pickle file
        with open(path, 'rb') as f:
            _embeddings = pkl.load(f)

        # Iterate over all embeddings
        for _embedding in _embeddings:
            # Get image type
            image_type = _embedding['image_type']

            # Get target concept
            target_concept = concept_pairs[concept]['target_concept']

            if image_type in args.types and (args.concepts is None or target_concept in args.concepts):
                # If embedding shall be included in plot, collect embedding and corresponding image type
                embeddings[concept].append(_embedding['clip_embedding'].squeeze())
                embedding_types[concept].append(image_type)

    # Hold values for intra and inter class indices
    inter = 0
    intra = 0

    # Get all concepts
    concepts = sorted(list(embeddings.keys()))

    # Iterate over each concept
    for i, concept in enumerate(concepts):
        print(f'Compute t-SNE for {concept}')

        # Get concept pair type (intra or inter) and original and target concept
        concept_type = concept_pairs[concept]['type']
        original_concept, target_concept = concept_pairs[concept]['original_concept'], concept_pairs[concept]['target_concept']

        # Determine index of subplot axis
        index = i
        if args.concepts is None:
            if concept_type == 'intra':
                index = 2 * intra
                intra += 1
            else:
                index = 2 * inter + 1
                inter += 1

        # Get subplot and set title and X and Y ticks
        ax = axes[index]
        ax.set_title(f"{original_concept} $\\rightarrow$ {target_concept}")
        ax.set_xticks([])
        ax.set_yticks([])

        # Calculate t-SNE projection
        x = np.array(embeddings[concept])
        tsne_x = TSNE(n_components=2, random_state=0, perplexity=30).fit_transform(x)

        # For each image type, plot the projected 2D embedding separately
        embedding_t = np.array(embedding_types[concept])
        ts = args.types
        for t in ts:
            # Get indices that belong to the image type
            indices = np.argwhere(embedding_t == t)

            # Add labels to first subplot only
            ax_kwargs = {'label': t + ' images'} if i == 0 else {}

            # Plot 2D embeddings
            ax.scatter(tsne_x[indices, 0], tsne_x[indices, 1], color=COLORS[t], s=0.75, **ax_kwargs)

    # Add legend to figure
    fig.legend(loc='lower center', fancybox=True, ncol=len(args.types), bbox_to_anchor=(0.5, legend_y), markerscale=4)

    # Save plot
    filename = f'tsne_embeddings_{"_".join(args.types)}'
    if args.concepts is not None:
        filename += f'_{"_".join(args.concepts)}'
    path = os.path.join(args.plot_dir, f'{filename}.png')
    fig.savefig(path, bbox_inches='tight', dpi=200)


def main():
    """
    Main program.
    """

    # Parse program arguments
    args = parse_args()

    # Plot embeddings
    plot_embeddings(args)


if __name__ == '__main__':
    main()
