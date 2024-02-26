#! /usr/bin/env python3
import matplotlib.pyplot as plt
import numpy as np


"""
## First: sysnet general
models = ['mBERT', 'LatinBERT', 'PhilBERTa', 'PhilTa']
# values = [610.99, 758.39, 699.79, 860.6]
values = [738.18, 960.78, 866.45, 1073.23]

barWidth = 0.18
plt.figure(figsize=(8, 7))  # Adjust figure size if needed
plt.bar(models, values)
for i, value in enumerate(values):
    plt.text(i, value + 10, f'{value:.2f}', ha='center', va='bottom', fontsize=14)

plt.xlabel('PLM', fontweight='bold', fontsize=19)
plt.ylabel('Synset candidates', fontweight='bold', fontsize=18)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.show()
"""

## Second: synset examples
permota = [328, 592, 1168, 1240]
peperit = [68, 28, 15, 324]

barWidth = 0.18
plt.figure(figsize=(8, 7))

br1 = np.arange(len(permota))
br2 = [x + barWidth for x in br1]

plt.bar(br1, permota, color='sandybrown', width=barWidth,
        edgecolor='sandybrown', label='permota')
plt.bar(br2, peperit, color='steelblue', width=barWidth,
        edgecolor='steelblue', label='peperit')

for i, value in enumerate(permota):
    plt.text(i, value + 10, f'{value:.0f}', ha='center', va='bottom', fontsize=11)
for i, value in enumerate(peperit):
    plt.text(i + barWidth, value + 10, f'{value:.0f}', ha='center', va='bottom', fontsize=11)

plt.xlabel('PLM', fontweight='bold', fontsize=15)
plt.ylabel('Synset candidates', fontweight='bold', fontsize=15)
plt.xticks([r + barWidth for r in range(len(permota))],
           ['mBERT', 'LatinBERT', 'PhilBERTa', 'PhilTa'], fontsize=14)
plt.yticks(fontsize=14)

plt.legend(fontsize=13)
plt.show()

