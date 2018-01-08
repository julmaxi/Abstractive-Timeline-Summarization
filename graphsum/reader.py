import os
import sys
import glob
import itertools as it

from xml.etree import ElementTree as ET
from subprocess import Popen


from collections.abc import Sequence


class TokenListView(Sequence):
    def __init__(self, container, attributes):
        self.container = container
        self.attributes = attributes

    def __getitem__(self, idx):
        tok = self.container.tokens[idx]

        if isinstance(idx, slice):
            return [tuple([getattr(t, arg) for arg in self.attributes]) for t in tok]
        else:
            return tuple([getattr(tok, arg) for arg in self.attributes])

    def __len__(self):
        return len(self.container.tokens)


class TokenAttrListView(Sequence):
    def __init__(self, container, attr):
        self.container = container
        self.attr = attr

    def __getitem__(self, idx):
        tok = self.container.tokens[idx]

        if isinstance(idx, slice):
            return [getattr(t, self.attr) for t in tok]
        else:
            return getattr(tok, self.attr)

    def __len__(self):
        return len(self.container.tokens)


class Document:
    def __init__(self, new_sentences, name = None):
        self.name = name
        self.sentences = []
        self._tok_list = None

        for sent in new_sentences:
            self.add_sentence(sent)

    def __iter__(self):
        return iter(self.sentences)

    def __len__(self):
        return len(self.sentences)

    @property
    def tokens(self):
        if self._tok_list is None:
            self._tok_list = list(it.chain(*map(lambda s: s.tokens, self.sentences)))
        return self._tok_list

    @property
    def basename(self):
        return self.name.split("/")[-1].split(".")[0]

    def add_sentence(self, sentence):
        sentence.idx = len(self.sentences)
        self.sentences.append(sentence)
        sentence.document = self

    def as_token_tuple_sequence(self, *args):
        return TokenListView(self, args)

    def as_token_attr_sequence(self, arg):
        return TokenAttrListView(self, arg)


class Sentence:
    def __init__(self, tokens,
                 dependency_tree=None, idx=None,
                 document=None, zero_based_indexing=True):
        self.tokens = tokens
        self.idx = idx
        self.dependency_tree = dependency_tree
        self.document = document
        self.zero_based_indexing = zero_based_indexing

        for token in self.tokens:
            token.sentence = self

    def as_tokenized_string(self):
        return " ".join(map(lambda t: t.form, self.tokens))

    def __iter__(self):
        return iter(self.tokens)

    def __getitem__(self, idx):
        if not self.zero_based_indexing:
            idx -= 1
        return self.tokens[idx]

    def as_token_tuple_sequence(self, *args):
        return TokenListView(self, args)

    def as_token_attr_sequence(self, arg):
        return TokenAttrListView(self, arg)

class Token:
    def __init__(self,
                 form=None,
                 lemma=None,
                 pos=None,
                 coarse_pos=None,
                 idx=None,
                 coref_chain=None):
        self.form = form
        self.lemma = lemma
        self.pos = pos
        self.coarse_pos = coarse_pos
        self.idx = idx
        self.sentence = None
        self.coref_chain = coref_chain

    def __repr__(self):
        return "Token(idx = {!r}, {!r}, sentence_idx = {!r})".format(
        self.idx, self.form, self.sentence.idx)

    def __eq__(self, other):
        return self.form.lower() == other.form.lower()

    def __hash__(self):
        return hash(self.form.lower())

    @property
    def form_lowercase(self):
        return self.form.lower()


class StanfordXMLReader:
    def __init__(self):
        pass

    def run(self, filename):
        tree = ET.parse(filename)
        root = tree.getroot()

        sents = []

        for xml_sent in root.find("document").find("sentences").iter("sentence"):
            sents.append(self.process_sentence(xml_sent))

        self.process_coref(root.find("document").find("coreference"), sents)

        return Document(sents, filename)

    def process_coref(self, all_coref_xml, sents):
        if all_coref_xml is None:
            return

        for coref_xml in all_coref_xml.iter("coreference"):
            coref_chain = []
            for mention in coref_xml.iter("mention"):
                sent_idx = int(mention.find("sentence").text) - 1
                head_idx = int(mention.find("head").text) - 1

                tok = sents[sent_idx][head_idx]

                tok.coref_chain = coref_chain

    def process_sentence(self, xml_sent):
        tokens = []
        token_idx_map = {}
        for token in xml_sent.find("tokens").iter("token"):
            pos = token.find("POS").text

            if pos.startswith("N"):
                coarse_pos = "NOUN"
            elif pos.startswith("V"):
                coarse_pos = "VERB"
            elif pos.startswith("A"):
                coarse_pos = "ADJ"
            else:
                coarse_pos = None

            tid = token.attrib["id"]
            token = Token(
                idx=tid,
                form=token.find("word").text,
                lemma=token.find("lemma").text,
                pos=pos,
                coarse_pos=coarse_pos)

            tokens.append(token)
            token_idx_map[tid] = token

        for deps in xml_sent.iter("dependencies"):
            if deps.attrib["type"] == "enhanced-plus-plus-dependencies":
                dep_tree = self.process_dependencies(deps, tokens)
                break

        return Sentence(tokens, zero_based_indexing=False)

    def process_dependencies(self, xml_deps, tokens):
        tree = DependencyTree()
        node_idx_map = {}
        for tok in tokens:
            node_idx_map[tok.idx] = tree.append_token(tok)

        for dep in xml_deps.iter("dep"):
            gov_id = dep.find("governor").attrib["idx"]
            dep_id = dep.find("dependent").attrib["idx"]
            extra = dep.attrib.get("extra", False)
            if extra:
                continue
            edge_type = dep.attrib["type"]

            if "subj" in edge_type:
                edge_type = "SB"
            elif "obj" in edge_type:
                edge_type = "OA"

            if edge_type != "root":
                dep_tok = node_idx_map[dep_id]

                if dep_tok.parent is None:
                    node_idx_map[gov_id].add_child(dep_tok, edge_type)

        return tree


class DUC2005Reader:
    def __init__(self, basedir):
        self.basedir = basedir

    def read_document_cluster(self, cluster_id):
        doc_dir = self._find_cluster_doc_dir(cluster_id)

        all_docs = []

        for doc_fname in os.listdir(doc_dir):
            full_path = os.path.join(doc_dir, doc_fname)
            doc = self.read_document(full_path)

            all_docs.append(doc)

        return cluster_id, all_docs

    @property
    def docs_path(self):
        return os.path.join(
            self.basedir,
            "duc2005_docs"
        )

    def _find_cluster_doc_dir(self, cluster_id):
        for path in os.listdir(self.docs_path):
            if path.startswith("d" + str(cluster_id)):
                return os.path.join(self.docs_path, path)

        return None

    def read_all_document_clusters(self):
        for doc_dirname in os.listdir(self.docs_path):
            num = doc_dirname[1:-1]
            yield self.read_document_cluster(num)

    def read_document(self, doc_path):
        tree = ET.parse(doc_path)
        paragraphs = tree.getroot().findall(".//P")
        if len(paragraphs) == 0:
            text = tree.getroot().find("TEXT")
            if text is not None:
                paragraphs = [text]
        return os.path.basename(doc_path), "\n".join([p.text for p in paragraphs])


if __name__ == "__main__":#java -cp "*" -Xmx2g edu.stanford.nlp.pipeline.StanfordCoreNLP -annotators tokenize,ssplit,pos,lemma -file input.txt
    reader = DUC2005Reader(sys.argv[1])
    target_dir = sys.argv[2]

    os.mkdir(target_dir)

    for cluster_id, docs in reader.read_all_document_clusters():
        cluster_dir = os.path.join(target_dir, cluster_id)
        os.mkdir(cluster_dir)
        for doc_name, doc in docs:
            with open(os.path.join(cluster_dir, doc_name), "w") as f_out:
                f_out.write(doc)
            doc_full_path = os.path.join(cluster_dir, doc_name)
            proc = Popen(["java", "-cp", '/Users/juliussteen/Documents/Studium/master/libs/stanford-corenlp-full-2015-12-09/*', "-Xmx2g", "edu.stanford.nlp.pipeline.StanfordCoreNLP", "-annotators", "tokenize,ssplit,pos,lemma", "-file", doc_full_path, "-outputDirectory", cluster_dir], cwd=os.curdir)
            proc.wait()
