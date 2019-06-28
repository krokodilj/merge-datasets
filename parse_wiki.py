import pandas as pd
import numpy as np
import json

from util import average_tokens

TRAIN_PATH = './WikiPassageQA/train.tsv'
TEST_PATH = './WikiPassageQA/test.tsv'
DEV_PATH = './WikiPassageQA/dev.tsv'

DOC_PATH = './WikiPassageQA/document_passages.json'

passages_json = dict()

with open(DOC_PATH) as f:
    passages_json = json.load(f)


train_queries = pd.read_csv(TRAIN_PATH, sep='\t')
test_queries = pd.read_csv(TEST_PATH, sep='\t')
dev_queries = pd.read_csv(DEV_PATH, sep='\t')

queries = pd.concat([train_queries, test_queries, dev_queries])


def flatten_passages(passages_json):
    result = list()

    for doc_id in passages_json:
        doc = passages_json[doc_id]

        for p_id in doc:
            passage = doc[p_id]
            
            # clear tabs and newlines
            passage = passage.replace('\n',' ') 
            passage = passage.replace('\t',' ') 

            result.append(['-'.join([doc_id, p_id]), passage])

    return pd.DataFrame(result, columns=['answer_id', 'answer_text'])


def find_relevant_question(answers, questions):

    results = list()

    for i, a in enumerate(answers):
        doc_id, p_id = a.split('-')

        is_relevant = (questions['DocumentID'] == int(doc_id)) & (
            questions['RelevantPassages'].str.contains(p_id))

        temp = questions[is_relevant]

        if not temp.empty:
            results.append(temp.iloc[0]['QID'])
        else:
            results.append(np.NaN)

        if i % 300 == 0:
            print("{:.2f} %".format(i*100/len(answers)))

    return results


flat_passages = flatten_passages(passages_json)

flat_passages['relevant_question'] = find_relevant_question(
    flat_passages['answer_id'], queries)

flat_passages = flat_passages[ ['answer_id', 'relevant_question', 'answer_text'] ]

flat_passages.to_csv('wiki_labeled.txt', sep='\t', index=False, na_rep='NaN')