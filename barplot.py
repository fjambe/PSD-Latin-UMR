#! /usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt

# set width of bar
barWidth = 0.25
fig = plt.subplots(figsize=(12, 8))

# set height of bar
guess_tot = [889.65, 950.47, 860.45, 1068.88]
guess_nohapax = [785.88, 861.31, 749.68, 1008.13]
guess_freq = [279.81, 404.85, 269.52, 570.04]

# Set position of bar on X axis
br1 = np.arange(len(guess_tot))
br2 = [x + barWidth for x in br1]
br3 = [x + barWidth for x in br2]
br4 = [x + barWidth for x in br3]

# Make the plot
plt.bar(br1, guess_tot, color='darkred', width=barWidth,
        edgecolor='darkred', label='guess_tot')
plt.bar(br2, guess_nohapax, color='sandybrown', width=barWidth,
        edgecolor='sandybrown', label='guess_no_hapax')
plt.bar(br3, guess_freq, color='steelblue', width=barWidth,
        edgecolor='steelblue', label='guess_frequent')

for i, value in enumerate(guess_tot):
    plt.text(i, value + 10, f'{value:.2f}', ha='center', va='bottom')
for i, value in enumerate(guess_nohapax):
    plt.text(i + barWidth, value + 10, f'{value:.2f}', ha='center', va='bottom')
for i, value in enumerate(guess_freq):
    plt.text(i + 2 * barWidth, value + 10, f'{value:.2f}', ha='center', va='bottom')

# Adding labels
plt.xlabel('PLM', fontweight='bold', fontsize=15)
plt.ylabel('Candidates', fontweight='bold', fontsize=15)
plt.xticks([r + barWidth for r in range(len(guess_tot))],
           ['mBERT', 'LatinBERT', 'PhilBERTa', 'PhilTa'])

plt.legend()
plt.show()
