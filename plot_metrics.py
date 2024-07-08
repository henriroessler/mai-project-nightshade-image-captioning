import argparse
import os

from matplotlib import pyplot as plt
import matplotlib as mpl
import pandas as pd

CMAP = mpl.colormaps['tab20']

COLORS = {
    'orange': CMAP(0.6),
    'banana': CMAP(0.6),
    'horse': CMAP(0.4),
    'cow': CMAP(0.4),
    'cup': CMAP(0.1),
    'bottle': CMAP(0.1),
    'skateboard': CMAP(0.5),
    'surfboard': CMAP(0.5),
    'car': CMAP(0.8),
    'motorcycle': CMAP(0.8),
    'person': CMAP(0.2),
    'airplane': CMAP(0.2),
    'boat': CMAP(0.3),
    'sandwich': CMAP(0.3),
    'cat': CMAP(0.0),
    'train': CMAP(0.0),
    'wine glass': CMAP(0.7),
    'traffic light': CMAP(0.7),
    'sink': CMAP(0.9),
    'backpack': CMAP(0.9),
    'inter': CMAP(0.0),
    'intra': CMAP(0.1)
}

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--file')
    parser.add_argument('--plot-dir')
    parser.add_argument('--concept-pairs-file')
    parser.add_argument('--group-by', default='target_concept')
    parser.add_argument('--filter', nargs='+', required=False, default=None)
    parser.add_argument('--legend-offset', type=float, default=1.13)
    parser.add_argument('--legend-auto', action='store_true')
    return parser.parse_args()


def plot_metric(args, metric: str, ylabel: str, ylim = (None, None)):
    plt.rc("lines", linewidth=2)

    concept_pairs_df = pd.read_csv(args.concept_pairs_file)

    df = pd.read_csv(args.file)
    df = df.merge(concept_pairs_df[['model', 'target_concept', 'type']], on='target_concept')
    df['asr'] = (df['attack_original'] + df['attack_both']) / df['pretrained_target'] * 100.0
    df['asr2'] = (df['finetuned_original'] + df['finetuned_both']) / df['num_images'] * 100.0

    sup_groups = df.groupby('finetune_type')
    for finetune_type, sup_df in sup_groups:

        for model in ('origpairs', 'switchpairs'):
            if finetune_type == 'origpairs':
                sup_df = sup_df[sup_df['model'] == model]
            # add pretrained results
            if finetune_type == 'origpairs':
                sup_df = pd.concat([sup_df, df[(df['frac'] == 0) & (df['model'] == model)]], ignore_index=True)
            else:
                sup_df = pd.concat([sup_df, df[df['frac'] == 0]], ignore_index=True)
            fig, ax = plt.subplots(figsize=(6, 3))
            #concept_title = '$X$' if model == 'switchpairs' else '$Y$'
            #pair_title = '$X \\rightarrow Y$' if finetune_type == 'origpairs' else '$X \\rightarrow Y$ and $Y \\rightarrow X$'
            #ax.set_title(f'Results of concepts {concept_title} for model finetuned on poisoned {pair_title} pairs')
            ax.set_ylabel(ylabel)
            ax.set_xlabel('Fraction of poisoned images [%]')

            if args.filter is not None:
                sup_df = sup_df[sup_df[args.group_by].isin(args.filter)]

            for group_by_key, sub_df in sup_df.groupby(args.group_by):
                aggregated_df = sub_df.groupby('frac').mean(numeric_only=True).sort_index()
                x = aggregated_df.index
                y = aggregated_df[metric]
                marker = 'o-'
                ax.plot(x, y, marker, color=COLORS.get(group_by_key), label=group_by_key)

            ax.set_ylim(*ylim)
            # ax.set_xlim(0, 100)
            ax.set_xticks(labels=['0', '5', '10', '25', '50', '100'], ticks=[0, 5, 10, 25, 50, 100])
            ax.tick_params(direction='in')
            legend_kwargs = dict(fancybox=True, ncol=1, loc='best')
            if not args.legend_auto:
                legend_kwargs.update({'loc': 'right', 'bbox_to_anchor': (args.legend_offset, 0.5)})
            ax.legend(**legend_kwargs)

            filename = f'{metric}_{model}_{finetune_type}_{args.group_by}'
            if args.filter is not None:
                filename += f'_{"_".join(args.filter)}'
            path = os.path.join(args.plot_dir, f'{filename}.v2.png')
            fig.savefig(path, bbox_inches='tight', dpi=200)


def main():
    args = parse_args()

    plot_metric(args, 'finetuned_bleu', 'BLEU score', (0.0, 0.4))
    plot_metric(args, 'asr', 'Attack success rate [%]', (-5, 105))
    plot_metric(args, 'asr2', 'Attack success rate [%]', (-5, 105))


if __name__ == '__main__':
    main()
