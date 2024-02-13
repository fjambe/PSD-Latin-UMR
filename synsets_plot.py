#! /usr/bin/env python3
import matplotlib.pyplot as plt

models = ['mBERT', 'LatinBERT', 'PhilBERTa', 'PhilTa']
# values = [610.99, 758.39, 699.79, 860.6]
values = [738.18, 960.78, 866.45, 1073.23]

barWidth = 0.18
plt.figure(figsize=(8, 7))  # Adjust figure size if needed
plt.bar(models, values)
for i, value in enumerate(values):
    plt.text(i, value + 10, f'{value:.2f}', ha='center', va='bottom')

plt.xlabel('PLM', fontweight='bold', fontsize=15)
plt.ylabel('Synset candidates', fontweight='bold', fontsize=15)
plt.show()
