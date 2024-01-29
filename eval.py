#! /usr/bin/env python3
# Copyright © 2024 Federica Gamba <gamba@ufal.mff.cuni.cz>
import argparse
import csv
from statistics import mean
from collections import Counter
from lxml import etree


def read_filename(filename):
    """Function to read the input CSV file.
    Expected format (6 columns):token, token_id, id_tect, possible_synsets, wrong_number, wrong_guesses."""
    stored_infile = {}
    with open(filename, 'r') as infile:
        reader = csv.DictReader(infile)
        for row in reader:
            token = row['token']
            token_id = row['token_id']
            lemma = row['lemma']
            id_tect = row['id_tect']
            possible_synsets = row['possible_synsets']
            wrong_number = row['wrong_number']
            wrong_guesses = row['wrong_guesses']

            stored_infile[token_id] = {
                'token': token,
                'lemma': lemma,
                'id_tect': id_tect,
                'possible_synsets': possible_synsets,
                'wrong_number': int(wrong_number),
                'wrong_guesses': wrong_guesses}
    return stored_infile


def get_lemmas_from_tlayer(t_filename, prefix='.//{http://ufal.mff.cuni.cz/pdt/pml/}'):
    """Function to retrieve all nodes with a valency frame in PDT tectogrammatical layer."""
    filename = f'/home/federica/vallex-pokus/LDT_PML_tectogrammatical_130317/LDT_Sallust/Sallust_all_files/{t_filename}'
    t_tree = etree.parse(filename)
    t_elem = t_tree.getroot()
    frames = t_elem.findall(f'{prefix}val_frame.rf')
    pdt_verbs = {}
    for fr in frames:
        t = fr.getparent()
        if t.find(f'{prefix}sempos').text == 'v':  # adding constraint on POS (verbs only)
            pdt_verbs[t.attrib['id']] = t.find(f'{prefix}t_lemma').text
    return pdt_verbs


def frequent_verbs(no_sum=True, no_habeo=True):
    """Function that retrieves the 10 (or n) most frequent lemmas in the whole corpus."""
    all_lemmas = get_lemmas_from_tlayer('sallust-libri1-10.afun.normalized.t')
    all_lemmas.update(get_lemmas_from_tlayer('sallust-libri11-20.afun.normalized.t'))
    all_lemmas.update(get_lemmas_from_tlayer('sallust-libri21-30.afun.normalized.t'))
    all_lemmas.update(get_lemmas_from_tlayer('sallust-libri31-40.afun.normalized.t'))
    all_lemmas.update(get_lemmas_from_tlayer('sallust-libri41-51.afun.normalized.t'))
    all_lemmas.update(get_lemmas_from_tlayer('sallust-libri52-61.afun.normalized.t'))
    if no_sum:
        all_lemmas = {k: lm for k, lm in all_lemmas.items() if lm != 'sum'}
    if no_habeo:
        all_lemmas = {k: lm for k, lm in all_lemmas.items() if lm != 'habeo'}
    print(all_lemmas)
    return Counter(all_lemmas.values())


def oov_guesses(stored_infile, exclude=False, lemma_filter=None):
    """
    Function to compute the average number (arithmetic mean) of candidate guesses extracted
    before retrieving one with the same lemma as the token to be annotated.
    Option to exclude cases of lemmas occurring only once (number of guesses = 1485).
    """
    if exclude:
        numbers = [item['wrong_number'] for item in stored_infile.values() if item['wrong_number'] != 1322]
    else:
        numbers = [item['wrong_number'] for item in stored_infile.values()]

    if lemma_filter:  # lemma_filter is a list of lemmas
        stored_infile = {k: v for k, v in stored_infile.items() if v.get('lemma') in lemma_filter}
        numbers = [item['wrong_number'] for item in stored_infile.values()]
    return round(mean(numbers), 2)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("file", type=str, help="File to be evaluated.")
    args = parser.parse_args()

    # Load and extract information
    info = read_filename(args.file)
    frequency = frequent_verbs()  # e.g. [('facio', 87), ('habeo', 83), ...]
    most_freq = [pair[0] for pair in frequency.most_common(10)]  # i.e., lemmas
    print(most_freq)

    # Compute and print out statistics
    avg_guess_tot = oov_guesses(info)
    avg_guess = oov_guesses(info, exclude=True)
    avg_freq_guess = oov_guesses(info, lemma_filter=most_freq)

    all_entries = len([k for k in info])  # 279
    oov = len([v for v in info.values() if v['wrong_number'] == 1322])  # 54
    oov_rate = round(oov / all_entries * 100, 2)  # 19.35%

    print('Average number of retrieved candidates:', avg_guess_tot,
          '\nAverage number of retrieved candidates excluding hapaxes:', avg_guess,
          '\nAverage number of retrieved candidates with most frequent verbs:', avg_freq_guess,
          '\nOOV rate:', oov_rate, '%')
