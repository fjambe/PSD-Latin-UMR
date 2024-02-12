#! /usr/bin/env python3
# Copyright © 2024 Federica Gamba <gamba@ufal.mff.cuni.cz>
import argparse
import csv
from statistics import mean
from collections import Counter
from lxml import etree

parser = argparse.ArgumentParser()
parser.add_argument("file", type=str, help="File to be evaluated.")


def read_filename(filename):
    """
    Function to read the input TSV file.
    Expected format (7 columns):token, token_id, id_tect, not_constrained_candidates, possible_synsets, wrong_number, wrong_guesses.
    """
    stored_infile = {}
    with open(filename, 'r') as infile:
        reader = csv.DictReader(infile, delimiter='\t')
        for row in reader:
            token = row['token']
            token_id = row['token_id']
            lemma = row['lemma']
            id_tect = row['id_tect']
            all_candidates = row['not_constrained_candidates'].split("), (")
            possible_synsets = row['possible_synsets']
            wrong_number = row['wrong_number']
            wrong_guesses = row['wrong_guesses']

            stored_infile[id_tect] = {
                'token': token,
                'lemma': lemma,
                'token_id': token_id,
                'all_candidates': all_candidates,
                'possible_synsets': possible_synsets,
                'wrong_number': int(wrong_number),
                'wrong_guesses': wrong_guesses}
    return stored_infile


def retrieve_annotated_verbs(filename):
    """Function to retrieve only annotated predicates."""
    annotated = {}
    with open(filename, 'r', encoding='utf8') as infile:
        for line in infile.readlines():
            line = line.strip().split('\t')
            # discard entries with len < 5, as they have no synset assigned
            if len(line) == 5:  # standard case
                if line[4].startswith('v#'):  # only verbal frames
                    annotated[line[0]] = (line[1], line[3], line[4])  # (lemma, UMR_id, synset_id)
                    # synset_id may be multiple (separator: /)
            elif len(line) == 6:  # cases where a new definitions was assigned
                if line[4].startswith('v#'):
                    annotated[line[0]] = (line[1], line[3], line[4])  # (lemma, UMR_id, synset_id)
                    # synset_id may be absent
        return annotated


def count_verbs(r_verbs, t_verbs, no_sum=True, no_habeo=True):
    """Function that retrieves the 10 (or n) most frequent lemmas in the whole corpus."""
    all_lemmas = {**r_verbs, **t_verbs}
    if no_sum:
        all_lemmas = {k: lm[0] for k, lm in r_verbs.items() if lm != 'sum'}
    if no_habeo:
        all_lemmas = {k: lm[0] for k, lm in r_verbs.items() if lm != 'habeo'}
    return Counter(all_lemmas.values())


def count_only_once_verbs(r_verbs, t_verbs):
    """Function that retrieves the target verbs occurring in the reference corpus only once."""
    ref_lemmas = Counter({k: v[0] for k, v in r_verbs.items()}.values())
    return [lm[0] for t_id, lm in t_verbs.items() if ref_lemmas.get(t_verbs[t_id][0]) == 1]


def interpret_guesses(stored_infile, exclude=False, lemma_filter=None, seen=None, no_zero_no_one=None):
    """
    Function to compute the average number (arithmetic mean) of candidate guesses extracted
    before retrieving one with the same lemma as the token to be annotated.
    Options:
    1. exclude cases of lemmas occurring only once (number of guesses = 1273).
    2. consider only a given subset of lemmas.
    3. consider only verbs whose synset is also present in the reference corpus.
    4. exclude cases of lemmas occurring either zero or one time(s) in the reference corpus.
    """

    if lemma_filter:  # lemma_filter is a list of lemmas
        stored_infile = {k: v for k, v in stored_infile.items() if v.get('lemma') in lemma_filter}
    elif seen:
        seen_synsets = [vr[1] for vr in seen.values()]
        to_keep = [k for k, v in tgt_verbs.items() if v[1] in seen_synsets]  # v[1] = synset-XX
        stored_infile = {k: v for k, v in stored_infile.items() if k in to_keep}
    elif no_zero_no_one:
        stored_infile = {k: v for k, v in stored_infile.items() if v['lemma'] not in no_zero_no_one}

    if exclude or no_zero_no_one:
        numbers = [item['wrong_number'] for item in stored_infile.values() if item['wrong_number'] != 1273]
    else:
        numbers = [item['wrong_number'] for item in stored_infile.values()]

    return round(mean(numbers), 2)


def compare_synsets(stored_infile, annotated, seen):  # simplified setup
    seen_synsets = [vr[1] for vr in seen.values()]
    annotated = {k: v for k, v in annotated.items() if v[1] in seen_synsets}

    cand_synsets = {}
    for t_id_tgt, tup_tgt in annotated.items():  # for each entry, i.e. row, in my tgt (annotated) file
        count_synsets = 0
        for id_tect, values in stored_infile.items():
            if id_tect == t_id_tgt:
                for cand in values['all_candidates']:
                    cand = cand.replace("(", "").replace("[", "").replace("]", "").split("', '")
                    cand = [c.replace("'", "") for c in cand]
                    if tup_tgt[2] == cand[1]:  # synset_id in tgt == retrieved_synset_id
                        cand_synsets[t_id_tgt] = count_synsets
                        break
                    else:
                        count_synsets += 1
    return cand_synsets


if __name__ == "__main__":
    args = parser.parse_args()

    # Load and extract information
    info = read_filename(args.file)
    ref_verbs = retrieve_annotated_verbs('/home/federica/vallex-pokus/predicting_frames/sallust-bert-GH'
                                         '/polished_total_frames_no31-40.tsv')
    tgt_verbs = retrieve_annotated_verbs('/home/federica/vallex-pokus/predicting_frames/sallust-bert-GH'
                                         '/polished_frames_only31-40.tsv')
    frequency = count_verbs(ref_verbs, tgt_verbs)  # e.g. [('facio', 81), ('dico', 39), ...]
    most_freq = [pair[0] for pair in frequency.most_common(10)]  # i.e., lemmas
    only_once = count_only_once_verbs(ref_verbs, tgt_verbs)  # 37, 13.7%

    # Compute and print out statistics
    avg_guess_tot = interpret_guesses(info)
    avg_guess_no_oov = interpret_guesses(info, exclude=True)
    avg_not_once_guess = interpret_guesses(info, no_zero_no_one=only_once)
    avg_freq_guess = interpret_guesses(info, lemma_filter=most_freq)
    avg_guess_seen = interpret_guesses(info, seen=ref_verbs)

    all_entries = len(tgt_verbs)  # 270
    oov = len([v for v in info.values() if v['wrong_number'] == 1273])  # 53
    oov_rate = round(oov / all_entries * 100, 2)  # 19.63%

    synset_comparison = compare_synsets(info, tgt_verbs, ref_verbs)
    print(synset_comparison)
    avg_synsets = round(mean(synset_comparison.values()), 2)

    print('10 most frequent verbs:', most_freq,
          '\nAverage number of retrieved candidates:', avg_guess_tot,
          '\nAverage number of retrieved candidates excluding hapaxes:', avg_guess_no_oov,
          '\nAverage number of retrieved candidates with only seen synsets:', avg_guess_seen,
          '\nAverage number of retrieved candidates with most frequent verbs:', avg_freq_guess,
          '\nAverage number of retrieved synsets in simplified setup:', avg_synsets,
          '\nOOV rate:', oov_rate, '%',
          '\nNumber of target predicates occurring only once in the reference corpus:', len(only_once),
          '\nPercentage of such verbs:', round(len(only_once) / len(tgt_verbs) * 100, 2), '%')
