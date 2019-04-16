import re
from tleval import load_corpus

import sys
import pickle

def clean_doc(doc):
    for idx, sent in enumerate(doc):
        #print(sent.as_tokenized_string())
        if (
            re.search(r"By .* \|", sent.as_tokenized_string())
            or re.search(r"\|.*\| Report abuse", sent.as_tokenized_string())
        ):
            #print("Deleting", sent.as_tokenized_string())
            doc.sentences = doc.sentences[:idx]
            break
        elif "|" in sent.as_tokenized_string():
            print(sent.as_tokenized_string())


if __name__ == "__main__":
    corpus = load_corpus(sys.argv[1])

    for doc in corpus:
        clean_doc(doc)

    with open(sys.argv[2], "wb") as f_out:
        pickle.dump(corpus, f_out)
