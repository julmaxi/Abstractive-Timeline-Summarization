import os
import sys
import glob
import itertools as it

from xml.etree import ElementTree as ET
from subprocess import Popen
import datetime

from collections.abc import Sequence
from collections import defaultdict

from tlgraphsum.utils import iter_dirs, iter_files, fst
from calendar import monthrange

import pickle

HARDCODED_PROBLEMATIC_DOCS = {'crisis-libya': {'768', '4746', '156', '981', '434', '1001', '5456', '1086', '825'}, 'tl17-libya': {'308', '257', '3', '212', '341', '309'}, 'tl17-haiti': {'820'}, 'tl17-bpoil': {'124', '364', '489', '2131', '1908', '1965'}, 'tl17-h1n1': {'519'}, 'tl17-finan': {'2124'}, 'crisis-egypt': {'2342', '506', '1808', '3312', '4831', '4983', '5047', '1268', '4933', '1569', '3110', '2510', '2190', '608', '385', '4633'}, 'crisis-yemen': {'3664', '592', '2787'}, 'tl17-syria': {'328', '283', '293'}, 'tl17-mj': {'176'}}


def load_corpus(fname, filter_blacklist=True):
    with open(fname, "rb") as f:
        corpus = pickle.load(f)
        corpus.name = fname
        corpus.basename = os.path.basename(fname)

    corpus_id, _ = os.path.basename(fname).split(".", 1)
    print(corpus.num_documents)
    if filter_blacklist:
        corpus.per_date_documents = filter_corpus_by_blacklist(corpus.per_date_documents, HARDCODED_PROBLEMATIC_DOCS.get(corpus_id, []))
    print(corpus.num_documents)

    return corpus


def filter_corpus_by_blacklist(per_date_documents, blacklist):
    new_per_date_documents = {}
    for date, docs in per_date_documents.items():
        new_per_date_documents[date] = [doc for doc in docs if doc.docid not in blacklist]
    return new_per_date_documents


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

    def __repr__(self):
        content = ", ".join(map(lambda x: repr(x), self))
        return "TokenListView([{}])".format(content)


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


class TimelineCorpus:
    def __init__(self, per_date_documents, blacklist=set()):
        super(TimelineCorpus, self).__init__()
        self.per_date_documents = dict()

        for date, docs in per_date_documents.items():
            self.per_date_documents[date] = [doc for doc in docs if doc.docid not in blacklist]

    def __iter__(self):
        return (doc for _, docs in sorted(self.per_date_documents.items()) for doc in docs)

    @property
    def num_documents(self):
        return sum(map(lambda l: len(l), self.per_date_documents.values()))

    def iter_dates(self):
        return iter(self.per_date_documents.keys())

    def docs_for_date(self, date):
        return self.per_date_documents[date]

    def as_docs_attr_sequence(self, arg):
        return [doc.as_token_attr_sequence(arg) for doc in self]

    @property
    def plaintext_docs(self):
        plaintext_docs = []
        for doc in self:
            plaintext_docs.append(doc.plaintext)

        return plaintext_docs

    @property
    def sentences(self):
        return [sent for doc in self for sent in doc]

    @property
    def sorted_sentences(self):
        return [sent for doc in sorted(self, key=lambda d: d.name) for sent in doc]

    def __len__(self):
        return sum(map(len, self.per_date_documents.values()))

    def docs_between_dates(self, first_date, last_date):
        relevant_docs = []

        for date, docs in self.per_date_documents.items():
            if date >= first_date and date <= last_date:
                relevant_docs.extend(docs)

        return relevant_docs

    def docs_at_dates(self, dates):
        relevant_docs = []

        for date, docs in self.per_date_documents.items():
            if date in dates:
                relevant_docs.extend(docs)

        return relevant_docs


class Document:
    def __init__(self, new_sentences, name=None):
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
    def docid(self):
        did, _ = os.path.basename(self.name).split(".", 1)

        return did

    @property
    def plaintext(self):
        return " ".join(self.as_token_attr_sequence("form"))

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

    def as_sentence_attr_sequence(self, arg):
        return [sent.as_token_attr_sequence(arg) for sent in self.sentences]


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

    @property
    def plaintext(self):
        return self.as_tokenized_string()

    def as_tokenized_string(self):
        return " ".join(map(lambda t: t.form, self.tokens))

    def as_tokenized_string_with_attribute(self, attr):
        toks = []
        for t in self.tokens:
            str_repr = "/".join((t.form, getattr(t, attr)))
            toks.append(str_repr)
        return " ".join(toks)

    def __iter__(self):
        return iter(self.tokens)

    def __getitem__(self, idx):
        if not self.zero_based_indexing:
            idx -= 1
        return self.tokens[idx]

    def __len__(self):
        return len(self.tokens)

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
                 coref_chain=None,
                 timex=None):
        self.form = form
        self.lemma = lemma
        self.pos = pos
        self.coarse_pos = coarse_pos
        self.idx = idx
        self.sentence = None
        self.coref_chain = coref_chain
        self.timex = timex

    def __repr__(self):
        return "Token(idx = {!r}, {!r}, sentence_idx = {!r})".format(
        self.idx, self.form, self.sentence.idx if self.sentence is not None else None)

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
            sent = self.process_sentence(xml_sent)
            sents.append(sent)

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
                idx=int(tid) - 1,
                form=token.find("word").text,
                lemma=token.find("lemma").text,
                pos=pos,
                coarse_pos=coarse_pos)

            tokens.append(token)
            token_idx_map[tid] = token

        dep_tree = None
        for deps in xml_sent.iter("dependencies"):
            if deps.attrib["type"] == "enhanced-plus-plus-dependencies":
                dep_tree = self.process_dependencies(deps, tokens)
                break

        return Sentence(tokens, zero_based_indexing=False, dependency_tree=dep_tree)

    def process_dependencies(self, xml_deps, tokens):
        tree = DependencyTree()
        node_idx_map = {}
        for tok in tokens:
            tree.add_token(tok)

        for dep in xml_deps.iter("dep"):
            # CoreNLP uses 1 based indexing
            gov_id = int(dep.find("governor").attrib["idx"]) - 1
            dep_id = int(dep.find("dependent").attrib["idx"]) - 1
            extra = dep.attrib.get("extra", False)
            if extra:
                continue
            edge_type = dep.attrib["type"]

            #if "subj" in edge_type:
            #    edge_type = "SB"
            #elif "obj" in edge_type:
            #    edge_type = "OA"

            if edge_type != "root":
                gov_tok = tokens[gov_id]
                dep_tok = tokens[dep_id]
                tree.add_edge(gov_tok, dep_tok, edge_type)

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

from xml.etree import ElementTree as ET
from collections import namedtuple

TimeEX = namedtuple("TimeEX", "tid type_ value tokens")


class TimeMLReader:
    def __init__(self, preserve_linebreaks=False):
        self.preserve_linebreaks = preserve_linebreaks

    def run(self, fname):
        etree = ET.parse(fname)
        root = etree.getroot()
        if not self.preserve_linebreaks:
            tokens = [(tok, None) for tok in root.text.split()]
        else:
            tokens = []
            lines = root.text.split("\n")
            for line in lines:
                tokens.extend([(tok, None) for tok in line.split(" ")])
                tokens.append(("\n", None))

        for elem in root:
            time_ex = TimeEX(
                elem.attrib["tid"],
                elem.attrib["type"],
                elem.attrib["value"],
                elem.text.split()
            )
            for tok in elem.text.split():
                tokens.append((tok, time_ex))
            if not self.preserve_linebreaks:
                for tok in elem.tail.split():
                    tokens.append((tok, None))
            else:
                lines = elem.tail.split("\n")
                for line in lines:
                    tokens.extend([(tok, None) for tok in line.split(" ")])
                    tokens.append(("\n", None))

        return tokens


def compute_month_week_range(year, month):
    try:
        _, start_week, _ = datetime.date(year, month, 1).isocalendar()
        _, month_len = monthrange(2011, 2)
        _, end_week, _ = datetime.date(year, month, month_len).isocalendar()
    except ValueError:
        pass
        #logger.warning("Invalid date reference {}-{}".format(year, month))
    else:
        return range(start_week, end_week + 1)

    return set()


def compute_week_from_day(year, month, day):
    try:
        _, week, _ = datetime.date(year, month, day).isocalendar()
    except ValueError:
        pass
        #logger.warning("Invalid date reference {}-{}".format(year, month))
    else:
        return week

    return None


class DateTag:
    YEAR = 1
    MONTH = 2
    DAY = 3
    WEEK = 4

    @staticmethod
    def from_timex(expr):
        parts = expr.split("-")

        if len(parts) == 1:
            if not parts[0].isdigit():
                raise ValueError("Expression {!r} is not a date tag".format(expr))

            return DateTag(DateTag.YEAR, int(parts[0]))

        if len(parts) == 3:
            return DateTag(
                DateTag.DAY,
                int(parts[0]),
                int(parts[1]),
                int(parts[2]))

        if len(parts) == 2:
            if parts[1].startswith("W"):
                return DateTag(DateTag.WEEK, int(parts[0]), int(parts[1][1:]))
            else:
                return DateTag(DateTag.MONTH, int(parts[0]), int(parts[1]))

        raise ValueError("Expression {!r} is not a date tag".format(expr))

    def __init__(self, dtype, *args):
        self.dtype = dtype

        self.year = None
        self.month = None
        self.week = None
        self.day = None

        if dtype == DateTag.YEAR:
            self.year = args[0]
        elif dtype == DateTag.MONTH:
            self.year = args[0]
            self.month = args[1]
        elif dtype == DateTag.DAY:
            self.year = args[0]
            self.month = args[1]
            self.day = args[2]
        elif dtype == DateTag.WEEK:
            self.year = args[0]
            self.week = args[1]

    def __contains__(self, other):
        if self.dtype == other.dtype:
            return self == other
        elif self.dtype == DateTag.DAY and other.dtype in (DateTag.YEAR, DateTag.MONTH, DateTag.WEEK):
            return False
        elif self.dtype == DateTag.WEEK and other.dtype in (DateTag.YEAR, DateTag.MONTH):
            return False
        elif self.dtype == DateTag.MONTH and other.dtype == DateTag.YEAR:
            return False

        if other.dtype == DateTag.MONTH:
            return self.month == other.month and self.year == other.year
        elif other.dtype == DateTag.WEEK:
            if self.dtype == DateTag.YEAR:
                return other.year == self.year
            elif self.dtype == DateTag.MONTH:
                return other.week in compute_month_week_range(self.year, self.month)
            assert False
        elif other.dtype == DateTag.DAY:
            if self.dtype == DateTag.MONTH:
                return self.year == other.year and self.month == other.month
            elif self.dtype == DateTag.WEEK:
                return self.week == compute_week_from_day(other.year, other.month, other.day)
            elif self.dtype == DateTag.YEAR:
                return self.year == other.year

        assert False

    def __eq__(self, other):
        return self.dtype == other.dtype and self.year == other.year and self.month == other.month and self.day == other.day and self.week == other.week

    def __hash__(self):
        return hash((self.dtype, self.year, self.month, self.week, self.day))

    def __repr__(self):
        if self.dtype != DateTag.WEEK:
            return "DateTag({}, {}, {}, {})".format(self.dtype, self.year, self.month, self.day)
        else:
            return "DateTag({}, {}, {})".format(self.dtype, self.year, self.week)

    def __str__(self):
        if self.dtype != DateTag.WEEK:
            return "{}-{}-{}".format(self.year, self.month, self.day)
        else:
            return "{}-W{}".format(self.year, self.week)

    @property
    def datetime(self):
        return datetime.datetime(self.year, self.month, self.day)


class DatedSentenceReader:
    def __init__(self):
        self.stanford_reader = StanfordXMLReader()
        self.time_ml_reader = TimeMLReader()

    def read(self, stanford_path, timeml_path, dct):
        doc = self.stanford_reader.run(stanford_path)

        timeml_tokens = self.time_ml_reader.run(timeml_path)

        doc_tok_iter = (tok for sent in doc for tok in sent)

        for sent in doc.sentences:
            sent.time_expressions = []

        doc_tok_iter = (tok for sent in doc for tok in sent)

        for timetok, doc_tok in zip(timeml_tokens, doc_tok_iter):
            (time_tok, timex) = timetok
            normalized_form = time_tok.replace("&", "&amp;")
            #if normalized_form != doc_tok.form:
            #    print("WARNING, unmatched text")
            assert time_tok == doc_tok.form, "{} != {}".format(time_tok, doc_tok.form)

            if timex is not None:
                doc_tok.timex = timex

                if len(doc_tok.sentence.time_expressions) == 0 or doc_tok.sentence.time_expressions[-1] != timex:
                    doc_tok.sentence.time_expressions.append(timex)

        doc.dct_tag = DateTag(DateTag.DAY, dct.year, dct.month, dct.day)
        doc.all_date_tags = set()
        for sent in doc.sentences:
            available_timeexs = filter(lambda tx: tx.type_ == "DATE", sent.time_expressions)

            possible_exact_dates = []
            all_date_tags = set()
            for timeex in available_timeexs:
                try:
                    tag = DateTag.from_timex(timeex.value)
                except ValueError:
                    continue
                all_date_tags.add(tag)
                doc.all_date_tags.add(tag)
                if tag.dtype == DateTag.DAY:
                    possible_exact_dates.append(tag)

            sent.all_date_tags = all_date_tags
            sent.exact_date_references = possible_exact_dates

            if len(possible_exact_dates) > 0:
                # TODO: Find better heuristic
                pred_date = possible_exact_dates[0]
                try:
                    sent.predicted_date = datetime.date(pred_date.year, pred_date.month, pred_date.day)
                except ValueError:
                    sent.predicted_date = dct
            else:
                sent.predicted_date = dct

        return doc


class DependencyTree:
    def __init__(self):
        self.token_heads = {}
#        self.nodes = []
#        self.roots = []

    def add_token(self, token):
        self.token_heads[token] = (None, None)

    def add_edge(self, head_token, token, edge_type):
        self.token_heads[token] = (head_token, edge_type)

    def __str__(self):
        return "\n".join(map(lambda n: str(n), self.roots))

    def as_head_idx_sequence(self):
        seq = []
        for token, (head_token, edge_type) in self.token_heads.items():
            if head_token is None:
                seq.append((token.idx, -1))
            else:
                seq.append((token.idx, head_token.idx))

        seq.sort()
        return [head_idx for _, head_idx in seq]

    def extract_svo_tuples(self):
        possible_tuples = defaultdict(dict)

        obj_token = None
        subj_token = None
        for token, (head, label) in self.token_heads.items():
            if label == "dobj":
                possible_tuples[head]["dobj"] = token
            elif label == "nsubj":
                possible_tuples[head]["subj"] = token

        tuples = []
        for root, args in possible_tuples.items():
            if args.get("subj") is not None:
                tuples.append(
                    (root, args.get("subj"), args.get("dobj"))
                )
        return tuples



#class DependencyTreeNode:
#    def __init__(self, token, tree, idx):
#        self.token = token
#        self.children = []
#        self.tree = tree
#        self.parent = None
#        self.incoming_edge_type = None
#        self.idx = idx
#
#    def add_child(self, child, edge_type):
#        if child.parent is None:
#            self.tree.roots.remove(child)
#        else:
#            child.parent.remove_child(child)
#        self.children.append((child, edge_type))
#        child.parent = self
#        child.incoming_edge_type = edge_type
#
#    def line_repr(self, indent=0):
#        if indent == 0:
#            indent_str = ""
#        elif indent == 1:
#            indent_str = "+----"
#        else:
#            indent_str = "     " * (indent - 1) + "+----"
#        own_line = indent_str + "({}) {}".format(
#            self.incoming_edge_type, self.token)
#        lines = [(self.idx, own_line)]
#        for child, _ in self.children:
#            lines += child.line_repr(indent + 1)
#
#        return lines
#
#    def __str__(self, indent=0):
#        return "\n".join(map(lambda t: t[1], sorted(self.line_repr())))


class DatedTimelineCorpusReader:
    def __init__(self, parsed_suffix=".tokenized", timeml_suffix=".tokenized.timeml"):
        self.reader = DatedSentenceReader()

        self.parsed_suffix = parsed_suffix
        self.timeml_suffix = timeml_suffix

    def run(self, document_dir, timeml_dir, ):
        date_dict = defaultdict(list)
        for date_dir in iter_dirs(document_dir):
            print("Reading", date_dir)
            dir_date = datetime.datetime.strptime(os.path.basename(date_dir), "%Y-%m-%d").date()

            for doc_fname in iter_files(date_dir, self.parsed_suffix):
                fname = os.path.basename(doc_fname)
                prefix = fname[:-len(self.parsed_suffix)]
                timeml_fname = os.path.join(timeml_dir, os.path.basename(date_dir), prefix + self.timeml_suffix)
                sentences = self.reader.read(doc_fname, timeml_fname, dir_date)
                date_dict[dir_date].append(sentences)

        return TimelineCorpus(date_dict)


if __name__ == "__main__":#java -cp "*" -Xmx2g edu.stanford.nlp.pipeline.StanfordCoreNLP -annotators tokenize,ssplit,pos,lemma -file input.txt
    reader = TimeMLReader(True)
    outdir = sys.argv[2]
    for crisis_dir in iter_dirs(sys.argv[1]):
        crisis_outdir = os.path.join(outdir, os.path.basename(crisis_dir))
        os.mkdir(crisis_outdir)

        doc_dir = os.path.join(crisis_dir, "articles")
        for date_dir in iter_dirs(doc_dir):
            date_out_dir = os.path.join(crisis_outdir, os.path.basename(date_dir))
            os.mkdir(date_out_dir)
            for fname in iter_files(date_dir, "timeml"):
                basename = os.path.basename(fname)[:-7]
                toks = reader.run(fname)

                with open(date_out_dir + "/" + basename + ".tokenized", "w") as f_out:
                    f_out.write(" ".join(map(fst, toks)))



    #print(DatedSentenceReader().read(sys.argv[1], sys.argv[2]))
#
    #sys.exit()
#
    #reader = DUC2005Reader(sys.argv[1])
    #target_dir = sys.argv[2]
#
    #os.mkdir(target_dir)
#
    #for cluster_id, docs in reader.read_all_document_clusters():
    #    cluster_dir = os.path.join(target_dir, cluster_id)
    #    os.mkdir(cluster_dir)
    #    for doc_name, doc in docs:
    #        with open(os.path.join(cluster_dir, doc_name), "w") as f_out:
    #            f_out.write(doc)
    #        doc_full_path = os.path.join(cluster_dir, doc_name)
    #        proc = Popen(["java", "-cp", '/Users/juliussteen/Documents/Studium/master/libs/stanford-corenlp-full-2015-12-09/*', "-Xmx2g", "edu.stanford.nlp.pipeline.StanfordCoreNLP", "-annotators", "tokenize,ssplit,pos,lemma", "-file", doc_full_path, "-outputDirectory", cluster_dir], cwd=os.curdir)
    #        proc.wait()
