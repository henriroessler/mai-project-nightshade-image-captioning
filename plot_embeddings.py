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

CMAP = mpl.colormaps['tab10']

COLORS = {
    'original': CMAP(0.0),
    'poisoned': CMAP(0.3),
    'target': CMAP(0.2)
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--concept-pairs-file', default='shared/concept_pairs.csv')
    parser.add_argument('--embed-dir')
    parser.add_argument('--plot-dir', default='plots')
    parser.add_argument('--types', nargs='+', default=['original', 'poisoned'])
    return parser.parse_args()


def plot_embeddings(args):
    fig, axes = plt.subplots(5, 2, figsize=(8, 11.5), sharex=True, sharey=True)
    axes = axes.flatten()
    fig.suptitle('t-SNE analysis of CLIP embeddings', fontsize=16, y=0.94)

    concept_pairs = pd.read_csv(args.concept_pairs_file, index_col='concat_name').to_dict('index')

    embeddings = defaultdict(list)
    embedding_types = defaultdict(list)

    for path in glob(os.path.join(args.embed_dir, '*.pkl')):
        concept = os.path.abspath(path)[:-17]
        concept = '_'.join(concept.split('_')[2:])
        print(f'Collecting embeddings for {concept}')
        with open(path, 'rb') as f:
            _embeddings = pkl.load(f)

        for _embedding in _embeddings:
            image_type = _embedding['image_type']

            if image_type in args.types:
                embeddings[concept].append(_embedding['clip_embedding'].squeeze())
                embedding_types[concept].append(image_type)

    inter = 0
    intra = 0

    for i, concept in enumerate(list(embeddings.keys())):
        print(f'Compute t-SNE for {concept}')

        concept_type = concept_pairs[concept]['type']
        original_concept, target_concept = concept_pairs[concept]['original_concept'], concept_pairs[concept]['target_concept']

        if concept_type == 'intra':
            index = 2 * intra
            intra += 1
        else:
            index = 2 * inter + 1
            inter += 1

        ax = axes[index]
        ax.set_title(f"{original_concept} $\\rightarrow$ {target_concept}")
        ax.set_xticks([])
        ax.set_yticks([])

        x = np.array(embeddings[concept])
        tsne_x = TSNE(n_components=2, random_state=0, perplexity=30).fit_transform(x)

        embedding_t = np.array(embedding_types[concept])
        ts = set(embedding_t)
        for t in ts:
            indices = np.argwhere(embedding_t == t)
            ax_kwargs = {'label': t + ' images'} if i == 0 else {}
            ax.scatter(tsne_x[indices, 0], tsne_x[indices, 1], color=COLORS[t], s=0.75, **ax_kwargs)

    fig.legend(loc='lower center', fancybox=True, ncol=len(args.types), bbox_to_anchor=(0.5, 0.06), markerscale=4)
    path = os.path.join(args.plot_dir, f'tsne_embeddings_{"_".join(args.types)}.png')
    fig.savefig(path, bbox_inches='tight', dpi=200)


def main():
    args = parse_args()

    plot_embeddings(args)


if __name__ == '__main__':
    main()
