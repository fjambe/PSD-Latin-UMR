#! /usr/bin/env python3
import matplotlib.pyplot as plt
from collections import Counter


def extract_and_plot(plm, all_or_first):
    with open(f'/home/federica/vallex-pokus/predicting_frames/sallust-bert-GH/{plm.lower()}_constrained_candidate_senses.tsv', 'r', encoding='utf8') as f:
        data = []
        next(f)
        for line in f:
            line = line.split('\t')[6]

            # first option: all retrieved candidates
            if all_or_first == 'all':
                verbs = [v.strip('\n') for v in line.split(';')]
                data.extend(verbs)
                how_many = 40
                extra = 8

            # second option: only first retrieved candidate
            elif all_or_first == 'first':
                verbs = [v.strip('\n') for v in line.split(';')][0]
                data.append(verbs)
                how_many = 20
                extra = 0.25

        counted_data = Counter(data)
        counted_data = counted_data.most_common(how_many)
        counted_data = {pair[0]: pair[1] for pair in counted_data}
        if all_or_first == 'first':
            counted_data = {k: v for k, v in counted_data.items() if v > 4}

        # PLOT
        # figure size and adjust layout
        if all_or_first == 'all':
            plt.figure(figsize=(20, 12))
            plt.subplots_adjust(bottom=0.25)

        elif all_or_first == 'first':
            plt.figure(figsize=(12, 15))  # Adjust the height to provide more space for title and values
            plt.subplots_adjust(left=0.3, right=0.9)  # Adjust left and right margins

        plt.barh(list(counted_data.keys()), list(counted_data.values()))  # plt.barh() for horizontal bars
        plt.yticks(rotation='horizontal', fontsize=20)  # Rotate y-axis labels and increase font size

        # adding text annotations
        for key, value in counted_data.items():
            plt.text(value + extra, key, str(value), ha='left', va='center', rotation='horizontal', fontsize=15)

        plt.title(p)
        plt.tight_layout()
        plt.savefig(f'./plots/{plm.lower()}_{all_or_first}_lemmas.png')


if __name__ == "__main__":

    plms = ['mBERT', 'Latin-BERT', 'PhilBERTa', 'PhilTa']
    for p in plms:
        extract_and_plot(p, all_or_first='all')
        extract_and_plot(p, all_or_first='first')
