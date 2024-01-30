#! /usr/bin/env python3
import matplotlib.pyplot as plt
from collections import Counter

plm = ['mBERT', 'Latin-BERT', 'PhilBERTa', 'PhilTa']
for p in plm:
    with open(f'/home/federica/vallex-pokus/predicting_frames/sallust-bert-GH/{p.lower()}_constrained_candidate_senses.csv', 'r', encoding='utf8') as f:
        data = []
        next(f)
        for line in f:
            line = line.split(',')[6]
            verbs = [v.strip('" ').strip('\n') for v in line.split(';')]
            data.extend(verbs)

    counted_data = Counter(data)
    counted_data = counted_data.most_common(50)
    counted_data = {pair[0]: pair[1] for pair in counted_data}

    # Increase figure size and adjust layout
    plt.figure(figsize=(20, 12))
    plt.subplots_adjust(bottom=0.25)

    plt.bar(counted_data.keys(), counted_data.values())
    plt.xticks(rotation='vertical')

    # Adding text annotations
    for key, value in counted_data.items():
        plt.text(key, value + 8, str(value), ha='center', va='bottom', rotation='vertical')

    plt.title(p)
    plt.tight_layout()
    plt.show()

