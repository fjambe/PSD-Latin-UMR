#! /usr/bin/env python3
# Copyright © 2024 Federica Gamba <gamba@ufal.mff.cuni.cz>
import argparse
import pandas as pd
from lxml import etree
from collections import defaultdict
from flair.embeddings import TransformerWordEmbeddings, TransformerDocumentEmbeddings
from flair.data import Sentence
from sklearn.metrics.pairwise import cosine_similarity
pd.set_option('display.max_columns', None)


parser = argparse.ArgumentParser()
parser.add_argument("bert_model", type=str, help="BERT model to be used."
                                                 "Suggested options: mbert, latin, laberta, philberta."
                                                 "To use a different model, specify its HF name as 'namespace/model'."
                                                 "An exception is raised if the passed model is not retrievable.")
args = parser.parse_args()
if args.bert_model == 'mbert':
    bert = 'bert-base-multilingual-cased'
elif args.bert_model == 'latin':
    bert = '/home/federica/hf-latin-bert/bert-base-latin-uncased/'
elif args.bert_model == 'laberta':
    bert = 'bowphs/LaBerta'  # BERT model trained on Latin (Corpus Corporum)
elif args.bert_model == 'philberta':
    bert = 'bowphs/PhilBerta'  # BERT model trained on Latin and Greek
else:
    bert = args.bert_model


# loading sentences (WN definitions)
def read_wordnet(wn_path):
    """Function to load Latin WordNet."""
    with open(wn_path, 'r', encoding='utf8') as wordn:
        lwn = pd.read_csv(wordn, header=0, usecols=lambda x: x not in ['id', 'type'])
        lwn['Transf_synset'] = lwn['id_synset'].str.removeprefix('http://wordnet-rdf.princeton.edu/wn30/').str.split('-')
        lwn['Transf_synset'] = lwn['Transf_synset'].str[1] + '#' + lwn['Transf_synset'].str[0]
    return lwn


def retrieve_definitions(filename, wn):
    """Function to load all definitions that have been assigned to annotated predicates."""
    defs = {}
    annotated = {}
    placeholder = 1
    with open(filename, 'r', encoding='utf8') as infile:
        for line in infile.readlines():
            line = line.strip().split('\t')
            # discard entries with len < 5, as they have no synset assigned
            if len(line) == 5:
                for s in line[4].split('/'):
                    definition = set(wn.loc[(wn['Transf_synset'] == s), 'definition'].values)
                    annotated[line[0]] = line[4]
                    # before data correction: try, except KeyError, continue
                    if s.startswith('v#'):  # only verbal frames
                        defs[s] = definition.pop()

            elif len(line) == 6:  # cases where a new definitions was assigned
                if line[4] not in ['', 'TBD', '-']:  # synset available
                    synset = line[4].strip(' ')  # strip to deal with extra spaces
                    if synset.startswith('v#'):
                        defs[synset] = set(wn.loc[(wn['Transf_synset'] == synset), 'definition'].values).pop()
                        annotated[line[0]] = line[4]
                else:
                    defs['v#' + str(placeholder)] = line[5]
                    annotated[line[0]] = 'v#' + str(placeholder)
                    placeholder += 1
    return defs, annotated


def get_token_from_mlayer(m_filename, prefix='.//{http://ufal.mff.cuni.cz/pdt/pml/}'):
    """Function to retrieve all tokens from PDT morphological layer."""
    filename = f'/home/federica/vallex-pokus/LDT_PML_tectogrammatical_130317/LDT_Sallust/Sallust_all_files/{m_filename}'
    m_tree = etree.parse(filename)
    m_elem = m_tree.getroot()
    xml_m_tokens = m_elem.findall(f'{prefix}m')

    pdt_tokens = {}
    for t in xml_m_tokens:
        form = t.find(f'{prefix}form')
        m_id = t.find(f'{prefix}w.rf')
        lemma = t.find(f'{prefix}lemma')
        pdt_tokens[m_id.text] = {'form': form.text, 'lemma': lemma.text}

    return pdt_tokens


def get_frames_from_tlayer(t_filename, pdt_tokens, prefix='.//{http://ufal.mff.cuni.cz/pdt/pml/}'):
    """Function to retrieve all nodes with a valency frame in PDT tectogrammatical layer."""
    filename = f'/home/federica/vallex-pokus/LDT_PML_tectogrammatical_130317/LDT_Sallust/Sallust_all_files/{t_filename}'
    t_tree = etree.parse(filename)
    t_elem = t_tree.getroot()

    frames = t_elem.findall(f'{prefix}val_frame.rf')
    pdt_verbs = defaultdict(dict)
    for fr in frames:
        t = fr.getparent()
        pdt_verbs[t.attrib['id']]['frame'] = fr.text
        pdt_verbs[t.attrib['id']]['w_id'] = t.find(f'{prefix}lex.rf').text.replace('a#a', 'w#w')

        for w_id, v in pdt_tokens.items():
            if w_id == pdt_verbs[t.attrib['id']]['w_id']:
                pdt_verbs[t.attrib['id']]['form'] = pdt_tokens[w_id]['form']
                v['id_tect'] = t.attrib['id']

    # control because one annotation error was found in PDT data (11-20.t)
    pdt_verbs = {k: v for k, v in pdt_verbs.items() if len(v) == 3}
    return pdt_verbs, pdt_tokens


def embeddings_in_df(embedder, tokens_or_document, sentence_processing=False):
    """Function to compute embeddings and store all information in a pd dataframe."""
    # tokens is the dictionary returned by get_token_from_mlayer. Only if sentence_processing is False.
    # document is a list of sentences. Only if sentence_processing is True.
    info = {}

    if not sentence_processing:
        # first, restoring raw text
        sents = {}
        sentenced_tokens = defaultdict(dict)
        for t, v in tokens_or_document.items():  # tokens_dict
            sent_id = t.strip('w#w-').split('W')[0]
            if sent_id not in sents:
                sents[sent_id] = [v['form'] for t, v in tokens_or_document.items() if t.strip('w#w-').split('W')[0] == sent_id]
            sentenced_tokens[sent_id][t] = v

        # calculating embeddings
        for sent_id, sent in sents.items():
            sentence = Sentence(sent)
            embedder.embed(sentence)

            for tk in sentence:
                info[tk.text] = {'token': tk.text, 'embedding': tk.embedding}
                for item, v in sentenced_tokens[sent_id].items():
                    if v['form'] == tk.text:
                        info[tk.text].update({'token_id': item})
                        info[tk.text].update({'lemma': v['lemma']})
                        if v.get('id_tect'):
                            info[tk.text]['id_tect'] = v['id_tect']

    else:
        progr_sent_number = 0
        for sent in tokens_or_document:  # document
            key = {d for d in defs if defs[d] == sent}
            sentence = Sentence(sent)
            embedder.embed(sentence)
            info[progr_sent_number] = {'sent_text': sentence.text, 'wn_synset_id': key.pop(),
                                       'sent_emb': sentence.embedding}
            progr_sent_number += 1

    return pd.DataFrame.from_dict(info, orient='index')


def similarity(lemma, emb1):
    """Function to compute cosine similarity between embeddings and output candidates."""
    simils = {}
    reasonable_cand = {}
    for index, row in ref_verbal_embeddings.iterrows():
        t_id = row.loc['id_tect']
        emb2 = row.loc['embedding']
        sim_score = cosine_similarity(emb1.reshape(1, -1), emb2.reshape(1, -1))  # reshape to a 2D tensor if needed
        simils[str(t_id)] = (sim_score, row.loc['lemma'])

        # add constraint on the lemma, i.e., compare token under scrutiny only to tokens with the same lemma
        if row.loc['lemma'] == lemma:
            reasonable_cand[str(t_id)] = sim_score

    return simils, reasonable_cand


def apply_similarity_and_sort(row):
    simscore, reasonable_cand = similarity(row['lemma'], row['embedding'])
    sorted_simscore = dict(sorted(simscore.items(), key=lambda x: x[1]))
    sorted_reasonable_cand = dict(sorted(reasonable_cand.items(), key=lambda x: x[1]))
    return pd.Series({'all_candidates': sorted_simscore, 'constrained_candidates': sorted_reasonable_cand})


def get_wrong_guesses(candidates, lemma):
    """
    Function to retrieve all candidate tokens that have a similarity score higher than
    the first token with constrained lemma.
    """
    wrong = []
    # found_lemma = False

    for candidate_data in candidates.values():
        if candidate_data[1] == lemma:
            # found_lemma = True
            break

        else:
        # if not found_lemma:
            wrong.append(candidate_data[1])

    return wrong


def process_candidates(candidates):
    return [(el, defs.get(annotation.get(el))) for el in candidates if annotation.get(el) and defs.get(annotation.get(el))]


# Load files
# polish files: cat xxx.tsv | cut -f ... (keep fields for lemma, pdt_frame, synset, def) | tail -n +4 | grep -Pv '^\t'
# possibly to integrate in the python code, once the file format will be standardized.
wordnet = read_wordnet('/home/federica/vallex-pokus/files/LiLa_LatinWordnet.csv')
defs, annotation = retrieve_definitions('/home/federica/Downloads/polished_total_frames_no31-40.tsv', wordnet)
definitions = list(set(defs.values()))


"""
# SENTENCE EMBEDDINGS for synset definitions
Computation of sentence embeddings for definitions that I manually assigned to verbal forms in Sallust.
TODO: restructure `definitions` by adding all synset-definition pairs + definitions added by me,
instead of storing only definitions already observed in the text and manually annotated.

AS OF NOW: implemented but not actually exploited.
"""

# FLAIR embeddings exploiting Transformers architecture for sentences
sent_embedding = TransformerDocumentEmbeddings(bert, seed=42)
sent_embeddings = embeddings_in_df(sent_embedding, definitions, sentence_processing=True)


"""
Compute contextual embeddings of verbal tokens in Latin.
1. Load Sallust 31-40 as test file: file for which I will try to extract synset/definition candidates.
2. Identify verbal tokens thanks to PDT frame annotation (morphological + tectogrammatical layers) and
reconstruct the full text (sequence of sentences) from there.
"""

# retrieving tokens from file
tokens = get_token_from_mlayer('sallust-libri31-40.afun.normalized.m')
# extracting predicates
verbs, tokens = get_frames_from_tlayer('sallust-libri31-40.afun.normalized.t', tokens)

"""
3. Select BERT model for Latin. Which one?

> `bert-base-multilingual-cased`:
from the inference API fill-mask, performances on the sentence *'[MASK] it ad forum'* look very bad.
I tried a RoBERTa model for Latin but I get the following warning:
"Some weights of RobertaModel were not initialized from the model checkpoint at `pstroe/roberta-base-latin-cased`
and are newly initialized. You should probably TRAIN this model on a down-stream task to be able to use it
for predictions and inference."

**Subtoken_pooling** (Flair library) is used to convert subword embeddings to word embeddings.
3 more options are available to perform this transformation, besides `mean`: `first`, `last`, `first_last`
(see https://flairnlp.github.io/docs/tutorial-embeddings/transformer-embeddings#Pooling-operation).

TODO: experiment with different options.
"""

# FLAIR embeddings exploiting Transformers architecture for tokens
embedding = TransformerWordEmbeddings(bert, repo_type= 'model', subtoken_pooling='mean', seed=42)
word_embeddings = embeddings_in_df(embedding, tokens)

"""Then, filter the dataframe only for verbal entries, i.e. entries that are found in the dictionary `verbs`."""
# Keeping only verbal tokens, which are assigned a frame
verbal = [k for k in verbs]
verbal_embeddings = word_embeddings[word_embeddings['id_tect'].isin(verbal)]


"""
Token similarity
For each verbal token in Sallust31-40 (or any text to annotate), compute similarity wrt other verbal tokens in the text.
This means that text (i.e., context) needs to be retrieved also for Sallust 1-30 + 41-61 (reference corpus).
"""

# retrieving reference tokens from file
ref_tokens = get_token_from_mlayer('sallust-libri1-10.afun.normalized.m')
ref_tokens.update(get_token_from_mlayer('sallust-libri11-20.afun.normalized.m'))
ref_tokens.update(get_token_from_mlayer('sallust-libri21-30.afun.normalized.m'))
ref_tokens.update(get_token_from_mlayer('sallust-libri41-51.afun.normalized.m'))
ref_tokens.update(get_token_from_mlayer('sallust-libri52-61.afun.normalized.m'))

# extracting predicates
ref_verbs, ref_tokens = get_frames_from_tlayer('sallust-libri1-10.afun.normalized.t', ref_tokens)
temp_verbs, temp_tokens = get_frames_from_tlayer('sallust-libri11-20.afun.normalized.t', ref_tokens)
ref_tokens.update(temp_tokens), ref_verbs.update(temp_verbs)
temp_verbs, temp_tokens = get_frames_from_tlayer('sallust-libri21-30.afun.normalized.t', ref_tokens)
ref_tokens.update(temp_tokens), ref_verbs.update(temp_verbs)
temp_verbs, temp_tokens = get_frames_from_tlayer('sallust-libri41-51.afun.normalized.t', ref_tokens)
ref_tokens.update(temp_tokens), ref_verbs.update(temp_verbs)
temp_verbs, temp_tokens = get_frames_from_tlayer('sallust-libri52-61.afun.normalized.t', ref_tokens)
ref_tokens.update(temp_tokens), ref_verbs.update(temp_verbs)

"""
For each token in `verbs`, find its n (n = 5) closest neighbours which are also contained in the `verbs` dictionary.
Therefore, compute embeddings for `ref_verbs` need to be computed.
"""

# Compute embeddings for ref_verbs
ref_word_embeddings = embeddings_in_df(embedding, ref_tokens)

# Keeping only verbal tokens, which are assigned a frame
# ref_verbal = [v['form'] for v in ref_verbs.values()]
ref_verbal = [k for k in ref_verbs]
ref_verbal_embeddings = ref_word_embeddings[ref_word_embeddings['id_tect'].isin(ref_verbal)]


"""
Now token similarity can be computed.
From https://intellica-ai.medium.com/comparison-of-different-word-embeddings-on-text-similarity-a-use-case-in-nlp-e83e08469c1c:
"Once we will have vectors of the given text chunk, to compute the similarity between generated vectors,
statistical methods for the vector similarity can be used. Such techniques are cosine similarity, Euclidean distance,
Jaccard distance, word mover’s distance. Cosine similarity is the technique that is being widely used for
text similarity + explanations on other measures."

TODO: check.
"""

result = verbal_embeddings.apply(apply_similarity_and_sort, axis=1)
# Concatenate the result with the original DataFrame
verbal_embeddings = pd.concat([verbal_embeddings, result], axis=1)

# retrieve candidates that are selected as appropriate before the first one with constrained lemma
verbal_embeddings['wrong_guesses'] = verbal_embeddings.apply(lambda row: get_wrong_guesses(row['all_candidates'], row['lemma']), axis=1)
verbal_embeddings['wrong_number'] = verbal_embeddings['wrong_guesses'].apply(len)


# extract only the 5 closest neighbours based on the similarity score value (with token constrained on the lemma only)
verbal_embeddings['constrained_candidates'] = verbal_embeddings['constrained_candidates'].apply(lambda constrained_candidates: dict(list(constrained_candidates.items())[:5]))
verbal_embeddings['possible_synsets'] = verbal_embeddings['constrained_candidates'].apply(process_candidates)
verbal_embeddings.to_csv(f'{args.bert_model}_constrained_candidate_senses.csv', columns=['token', 'token_id', 'id_tect', 'possible_synsets', 'wrong_number', 'wrong_guesses'], encoding='utf-8', index=False)

# TODO: among the possible definitions for my token's lemma, find the closest to each of the candidates.


# list of possible definition for my token: given a lemma, I have column definition in dataframe wordnet
# for each of these definition I can already retrieve its embeddings in sent_embeddings df,
# where I have sent_text - wn_synset_id - sent_emb
# each of the candidates: each el in possible_synsets
# for each definition (d) of my token, compute sentence similarity with each of the candidates (c)
# how to compute sentence similarity? do I need sentences or embeddings?
# possibly reduce the number of candidates to 3.

