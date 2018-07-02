import argparse
import os
import gzip
import xml.sax


class StanfordPOSContentHandler(xml.sax.ContentHandler):
    def __init__(self):
        xml.sax.ContentHandler.__init__(self)
        self.curr_sent = []
        self.is_pos = False

        self.all_sents = []

        self.curr_pos_parts = []

    def startElement(self, name, attrs):
        if name.lower() == "pos":
            self.is_pos = True

    def endElement(self, name):
        if name.lower() == "pos":
            self.is_pos = False
            self.curr_sent.append("".join(self.curr_pos_parts))
            self.curr_pos_parts = []
        elif name.lower() == "sentence":
            if len(self.curr_sent) > 0:
                self.all_sents.append(self.curr_sent)
                self.curr_sent = []

    def characters(self, content):
        if self.is_pos:
            self.curr_pos_parts.append(content)


def iter_gigaword_dir(gw_dir, fname_filter=lambda f: True):
    for fname in os.listdir(gw_dir):
        if not fname.endswith(".xml.gz") or not fname_filter(fname):
            continue
        with gzip.open(os.path.join(gw_dir, fname), 'rb') as f:
            content = f.read()
            yield content


def filter_pos_from_raw_xml(content):
    handler = StanfordPOSContentHandler()
    xml.sax.parseString(content, handler)
    return handler.all_sents


def read_gigaword_pos(gw_dir):
    for file_content in iter_gigaword_dir(gw_dir, lambda fname: fname.startswith("afp")):
        pos_sentences = filter_pos_from_raw_xml(file_content)

        for sent in pos_sentences:
            yield sent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("gw_dir")
    parser.add_argument("outfile")
    args = parser.parse_args()

    with open(args.outfile, "w") as f_out:
        for pos_sent in read_gigaword_pos(args.gw_dir):
            f_out.write(" ".join(pos_sent))
            f_out.write("\n")


if __name__ == "__main__":
    main()
