import pandas as pd
import numpy as np
from util import average_tokens


PATH = "./Antique/antique-collection.txt"
TRAIN_QREL_PATH = "./Antique/antique-train.qrelnew" # Formatted .qrels
TEST_QREL_PATH = "./Antique/antique-test.qrelnew"   # Formatted .qrels

data_columns = ['answer_id', 'answer_text']
qrel_columns = ['question_id', 'something', 'answer_id', 'relevance']

data = pd.read_csv(PATH, sep="\t", header=None, keep_default_na=False)
data.columns = data_columns

train_qrel = pd.read_csv(TRAIN_QREL_PATH, sep="\t", header=None, keep_default_na=False)
train_qrel.columns = qrel_columns

test_qrel = pd.read_csv(TEST_QREL_PATH, sep="\t", header=None, keep_default_na=False)
test_qrel.columns = qrel_columns

qrel = pd.concat([train_qrel, test_qrel])


def parse_qrel(path, new_path):
    """
    Data is not formatted correctly
    Format the data, use tab spacing
    """
    data = list()

    f = open(path, 'r')
    for line in f.readlines():
        data.append([l.strip() for l in line.split()])
    f.close()

    f = open(new_path, 'w')
    for values in data:
        f.write("\t".join(values)+"\n")
    f.close()

# parse_qrel(TEST_QREL_PATH, TEST_QREL_PATH+'new')

# Create new format
# AnswerID AnswerText RelevantQuestionIDs
# {
#     answer_id: '',
#     answer_text: '',
#     relevant_question_id: ''
# }


def find_relevant_questions(answer_ids, q_rel):

    result = list()

    for i, answer_id in enumerate(answer_ids):

        is_relevant = (q_rel['answer_id'] == answer_id) & (q_rel['relevance'] >= 3)

        temp = q_rel[is_relevant]

        if not temp.empty:
            result.append(temp['question_id'].iloc[0])
        else:
            result.append(np.NaN)

        if i % 300 == 0:
            print("{:.2f} %".format(i*100/len(answer_ids)))

    return result

# data = data[:4000]

data['relevant_question'] = find_relevant_questions(data['answer_id'], qrel)

data = data[ ['answer_id', 'relevant_question', 'answer_text'] ]

data.to_csv('antique_labeled.txt', sep='\t', index=False, na_rep='NaN')

average_tokens(data['answer_text'])


