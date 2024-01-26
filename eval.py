#! /usr/bin/env python3
# Copyright © 2024 Federica Gamba <gamba@ufal.mff.cuni.cz>
import argparse
import csv
from statistics import mean
from collections import Counter


def read_filename(filename):
    """Function to read the input CSV file.
    Expected format (6 columns):token, token_id, id_tect, possible_synsets, wrong_number, wrong_guesses."""
    result_dict = {}
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

            result_dict[token_id] = {
                'token': token,
                'lemma': lemma,
                'id_tect': id_tect,
                'possible_synsets': possible_synsets,
                'wrong_number': int(wrong_number),
                'wrong_guesses': wrong_guesses}
    return result_dict


def frequent_verbs(stored_infile, no_sum=True):
    """Function that retrieves the 10 (or n) most frequent lemmas."""
    lemmas = [item['lemma'] for item in stored_infile.values()]
    if no_sum:
        lemmas = [lm for lm in lemmas if lm != 'sum']
    return Counter(lemmas)


def oov_guesses(stored_infile, exclude=False, lemma_filter=None):
    """
    Function to compute the average number (arithmetic mean) of candidate guesses extracted
    before retrieving one with the same lemma as the token to be annotated.
    Option to exclude cases of lemmas occurring only once (number of guesses = 1485).
    """
    if exclude:
        numbers = [item['wrong_number'] for item in stored_infile.values() if item['wrong_number'] != 1485]
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

    # Load and extract necessary information
    info = read_filename(args.file)
    frequency = frequent_verbs(info)  # e.g. [('facio', 83), ('habeo', 74), ...]
    most_freq = [pair[0] for pair in frequency.most_common(10)]  # i.e., lemmas

    # Compute and print out statistics
    avg_guess_tot = oov_guesses(info)
    avg_guess = oov_guesses(info, exclude=True)
    avg_freq_guess = oov_guesses(info, lemma_filter=most_freq)
    print('Average number of retrieved candidates:', avg_guess_tot,
          '\nAverage number of retrieved candidates excluding hapaxes:', avg_guess,
          '\nAverage number of retrieved candidates with most frequent verbs:', avg_freq_guess)
