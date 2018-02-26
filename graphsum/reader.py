import os
import sys
import glob
import itertools as it

from xml.etree import ElementTree as ET
from subprocess import Popen
import datetime

from collections.abc import Sequence
from collections import defaultdict

from utils import iter_dirs, iter_files

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


class TimelineCorpus:
    def __init__(self, per_date_documents):
        super(TimelineCorpus, self).__init__()
        self.per_date_documents = per_date_documents

    def __iter__(self):
        return (doc for docs in self.per_date_documents.values() for doc in docs)

    def iter_dates(self):
        return iter(self.per_date_documents.keys())

    def docs_for_date(self, date):
        return self.per_date_documents[date]

    @property
    def sentences(self):
        return [sent for doc in self for sent in doc]

    @property
    def sorted_sentences(self):
        return [sent for doc in sorted(self, key=lambda d: d.name) for sent in doc]


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

        print(expr)
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

    def __eq__(self, other):
        return self.dtype == other.dtype and self.year == other.year and self.month == other.month and self.day == other.day and self.week == other.week

    def __hash__(self):
        return hash((self.dtype, self.year, self.month, self.week, self.day))

    def __repr__(self):
        if self.dtype != DateTag.WEEK:
            return "DateTag({}, {}, {}, {})".format(self.dtype, self.year, self.month, self.day)
        else:
            return "DateTag({}, {}, {})".format(self.dtype, self.year, self.week)

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
                #try:
                #    date = datetime.datetime.strptime(timeex.value, "%Y-%m-%d").date()
                #    possible_exact_dates.append(date)
                #    parts = timeex.value.split("-")
                #    tag = (int(parts[0]), int(parts[1]), int(parts[2]))
                #    all_date_tags.add(tag)
                #    doc.all_date_tags.add(tag)
                #except ValueError:
                #    pass
                #else:
                #    continue
#
                #try:
                #    date = datetime.datetime.strptime(timeex.value, "%Y-%m").date()
                #    parts = timeex.value.split("-")
                #    tag = (int(parts[0]), int(parts[1]), None)
                #    all_date_tags.add(tag)
                #    doc.all_date_tags.add(tag)
                #except ValueError:
                #    pass
                #else:
                #    continue
#
                #try:
                #    date = datetime.datetime.strptime(timeex.value, "%Y").date()
                #    tag = (int(timeex.value), None, None)
                #    all_date_tags.add(tag)
                #    doc.all_date_tags.add(tag)
                #except ValueError:
                #    pass
                #else:
                #    continue

            sent.all_date_tags = all_date_tags
            sent.exact_date_references = possible_exact_dates

            if len(possible_exact_dates) > 0:
                # TODO: Find better heuristic
                #print(sent.time_expressions[0])
                sent.predicted_date = possible_exact_dates[0]
            else:
                sent.predicted_date = dct


        #from collections import deque
#
        #context = deque([], 10)
#
        #while True:
        #    try:
        #        next_doc_tok = next(doc_tok_iter)
        #        next_time_ml_tok = next(timeml_iter)
        #    except StopIteration:
        #        break
#
        #    context.append(next_doc_tok.form)
#
#
        #    if isinstance(next_time_ml_tok, TimeEX):
        #        tid, type_, value, tokens = next_time_ml_tok
#
        #        for tok in tokens[1:]:
        #            try:
        #                next_doc_tok = next(tok)
#
        #                context.append(next_doc_tok.form)
#
        #                print(next_doc_tok.form, tok)
        #                if next_doc_tok.form != tok:
        #                    raise RuntimeError("Parse Error: {} != {}".format(
        #                        next_doc_tok.form,
        #                        tok
        #                    ))
#
        #            except StopIteration:
        #                break
        #    elif next_doc_tok.form != next_time_ml_tok:
        #        if next_doc_tok.form == "-LRB-" and next_time_ml_tok == "(" \
        #            or next_doc_tok.form == "-RRB-" and next_time_ml_tok == ")":
        #            continue
#
        #        if next_doc_tok.form.startswith(next_time_ml_tok):
        #            accumulated_time_ml = next_time_ml_tok
#
#
        #        raise RuntimeError("Parse Error: {} != {}. Context: {}".format(
        #            next_doc_tok.form,
        #            next_time_ml_tok,
        #            " ".join(context)
        #        ))

        return doc


class DependencyTree:
    def __init__(self):
        self.nodes = []
        self.roots = []

    def append_token(self, token, parent=None, edge_type=None, **args):
        node = DependencyTreeNode(token, self, len(self.nodes), **args)
        self.nodes.append(node)
        if parent:
            parent.add_child(node, edge_type)
        else:
            self.roots.append(node)

        return node

    def __str__(self):
        return "\n".join(map(lambda n: str(n), self.roots))

    def node_for_token(self, token):
        for node in self.nodes:
            if node.token == token:
                return node
        else:
            return None


class DependencyTreeNode:
    def __init__(self, token, tree, idx):
        self.token = token
        self.children = []
        self.tree = tree
        self.parent = None
        self.incoming_edge_type = None
        self.idx = idx

    def add_child(self, child, edge_type):
        if child.parent is None:
            self.tree.roots.remove(child)
        else:
            child.parent.remove_child(child)
        self.children.append((child, edge_type))
        child.parent = self
        child.incoming_edge_type = edge_type

    def line_repr(self, indent=0):
        if indent == 0:
            indent_str = ""
        elif indent == 1:
            indent_str = "+----"
        else:
            indent_str = "     " * (indent - 1) + "+----"
        own_line = indent_str + "({}) {}".format(
            self.incoming_edge_type, self.token)
        lines = [(self.idx, own_line)]
        for child, _ in self.children:
            lines += child.line_repr(indent + 1)

        return lines

    def __str__(self, indent=0):
        return "\n".join(map(lambda t: t[1], sorted(self.line_repr())))


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
    from utils import iter_dirs, iter_files, fst
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

                with open(date_out_dir + "/" + basename, "w") as f_out:
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
