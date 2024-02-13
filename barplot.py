#! /usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt

# set width of bar
barWidth = 0.2
fig = plt.subplots(figsize=(14, 9))

# set height of bar
# guess_tot = [906.81, 955.6, 876.05, 1057.71]
guess_tot = [1158.59, 1228.24, 1127.13, 1352.18]
# guess_nohapax = [804.13, 866.59, 764.74, 997.34]
guess_nohapax = [1021.79, 1111.02, 981.48, 1269.83]
# guess_seen = [751.37, 846.96, 730.56, 958.8]
guess_seen = [963.58, 1084.63, 929.84, 1214.21]
# guess_freq = [188.5, 356.68, 159.5, 545.27]
guess_freq = [362.67, 491.19, 248, 679.67]

# Set position of bar on X axis
br1 = np.arange(len(guess_tot))
br2 = [x + barWidth for x in br1]
br3 = [x + barWidth for x in br2]
br4 = [x + barWidth for x in br3]
br5 = [x + barWidth for x in br4]

# Make the plot
plt.bar(br1, guess_tot, color='darkred', width=barWidth,
        edgecolor='darkred', label='guess_tot')
plt.bar(br2, guess_nohapax, color='sandybrown', width=barWidth,
        edgecolor='sandybrown', label='guess_no_hapax')
plt.bar(br3, guess_seen, color='steelblue', width=barWidth,
        edgecolor='steelblue', label='guess_seen')
plt.bar(br4, guess_freq, color='lightpink', width=barWidth,
        edgecolor='lightpink', label='guess_frequent')


for i, value in enumerate(guess_tot):
    plt.text(i, value + 10, f'{value:.2f}', ha='center', va='bottom')
for i, value in enumerate(guess_nohapax):
    plt.text(i + barWidth, value + 10, f'{value:.2f}', ha='center', va='bottom')
for i, value in enumerate(guess_seen):
    plt.text(i + 2 * barWidth, value + 10, f'{value:.2f}', ha='center', va='bottom')
for i, value in enumerate(guess_freq):
    plt.text(i + 3 * barWidth, value + 10, f'{value:.2f}', ha='center', va='bottom')

# Adding labels
plt.xlabel('PLM', fontweight='bold', fontsize=15)
plt.ylabel('Candidates', fontweight='bold', fontsize=15)
plt.xticks([r + barWidth for r in range(len(guess_tot))],
           ['mBERT', 'LatinBERT', 'PhilBERTa', 'PhilTa'])

plt.legend()
plt.show()
