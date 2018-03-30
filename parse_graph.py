import sys
from collections import namedtuple, Counter


class AMRGraph:
    def __init__(self, raw_text=""):
        self.raw_text = raw_text
        self.nodes = {}
        self.edges = {}






class StringParser:
    def __init__(self, raw_str):
        self.idx = 0
        self.raw_str = raw_str

    def __iter__(self):
        return self

    def __next__(self):
        if self.idx == len(self.raw_str):
            raise StopIteration()

        ch = self.raw_str[self.idx]

        self.idx += 1

        return ch

    def skip_whitespace(self):
        for ch in self:
            if ch not in " \n\t":
                break

        self.idx -= 1

    def skip_past_next(self, chars):
        for ch in self:
            if ch in chars:
                self.idx -= 1
                return ch

    def consume_to_chars(self, chars):
        read_chars = []
        for ch in self:
            if ch in chars:
                break

            read_chars.append(ch)

        self.idx -= 1

        return "".join(read_chars)

    def consume_to_whitespace(self):
        return self.consume_to_chars(" \n\t")

    def assert_and_consume(self, assert_str):
        for ch1, ch2 in zip(assert_str, self):
            assert ch1 == ch2, "{} != {}".format(ch1, ch2)

    @property
    def current_char(self):
        return self.raw_str[self.idx]








def parse_graph_description(graph_str_lines):
    graph_str_iter = iter(graph_str_lines)

    uncommented_graph_lines = []

    for line in graph_str_lines:
        if not line.startswith("#"):
            uncommented_graph_lines.append(line)

    #print(" ".join(uncommented_graph_lines))
    parser = StringParser(" ".join(uncommented_graph_lines))
    parser.skip_whitespace()

    return parse_node(parser)




def parse_child(parser):
    parser.skip_whitespace()
    if parser.current_char == "(":
        content = parse_node(parser)
    elif parser.current_char == '"':
        content = parse_literal(parser)
    elif parser.current_char in "0123456789":
        content = parse_number(parser)
    elif parser.current_char.lower() in "abcdefghijklmnopqrstuvwxyz":
        content = parse_reference(parser)
    elif parser.current_char.lower() in "+-":
        content = parse_polarity(parser)
        #print("\n", repr(parser.current_char), parser.raw_str[parser.idx - 5: parser.idx + 25])

    return content


Node = namedtuple("Node", "node_id content children")
NodeReference = namedtuple("NodeReference", "ref_id")
Polarity = namedtuple("Polarity", "pol")


def parse_reference(parser):
    content = parser.consume_to_chars(" )")

    return NodeReference(content)


def parse_number(parser):
    content = parser.consume_to_whitespace()

    return content


def parse_polarity(parser):
    return Polarity(parser.consume_to_whitespace())


def parse_literal(parser):
    parser.assert_and_consume('"')
    content = parser.consume_to_chars('"')
    parser.assert_and_consume('"')
    return content


def parse_node(parser):
    #PARSE_ID = 0
    #PARSE_CONTENT = 1
    #PARSE_CHILDREN = 2
#
    #parse_state = PARSE_ID

    #print(parser.raw_str[parser.idx:parser.idx+25])
    parser.assert_and_consume("(")
    #print(parser.raw_str[parser.idx:parser.idx+25])
    parser.skip_whitespace()
    node_id = parser.consume_to_whitespace()
    parser.skip_whitespace()
    parser.assert_and_consume("/")
    parser.skip_whitespace()
    node_content = parser.consume_to_whitespace()

    children = []
    while True:
        next_char = parser.skip_past_next([":", ")"])
        if next_char == ")":
            break
        parser.skip_whitespace()
        label = parser.consume_to_whitespace()
        parser.skip_whitespace()

        #print(parser.raw_str[parser.idx:parser.idx+25])
        child = parse_child(parser)
        children.append((label, child))


    return Node(node_id, node_content, children)



    ##parse_state = PARSE_CONTENT
#
    #content_chars = []
    #for ch in char_iter:
    #    if ch == " ":
    #        break
    #    else:
    #        content_chars.append(ch)
#
    #node_content = "".join(content_chars)
#
    #skip_whitespace(char_iter)
    #children = []
    #for ch in char_iter:
    #    if ch == ":":
    #        children.append(parse_child(char_iter))
#
    #return (node_id, node_content, children)
#

def read_amr_graphs_file(f):
    current_graph = []
    for line in f:
        line = line.strip()

        if len(line) == 0:
            yield parse_graph_description(current_graph)
            current_graph = []
        else:
            current_graph.append(line)

    if len(current_graph) > 0:
        yield parse_graph_description(current_graph)


def gather_children_args(children):
    args = []
    for key, child in children:
        if isinstance(child, Node):
            args.append(child)
            args.extend(gather_children_args(child[2]))

    return args


def find_child(node, node_type):
    for name, child in node.children:
        if name == node_type:
            return child


if __name__ == "__main__":
    print(parse_graph_description(
        """
(p3 / phase
      :ARG1-of (o / oppose-01
            :ARG0 (c3 / country
                  :ARG1-of (h / have-org-role-91
                        :ARG0 (p2 / person
                              :ARG0-of (p / post-01
                                    :op1-of (c / circle)
                                    :op1-of (i / inner))
                              :ARG0-of (b / bomb-01
                                    :ARG1 (s / suicide))
                              :ARG2-of (w / wake-up-02
                                    :ARG0 (f / final)
                                    :ARG1 (t / this))
                              :name (n / name
                                    :op1 "Assad")
                              :wiki "Bashar_al-Assad")
                        :ARG2 (c2 / chief))
                  :name (n2 / name
                        :op1 "Syria")
                  :wiki "Syria")))        """.strip().split()
    ))

    root_counter = Counter()
    root_arg_counter = Counter()

    with open(sys.argv[1]) as f:
        for idx, graph in enumerate(read_amr_graphs_file(f)):
            root_counter[graph[1]] += 1

            arg0 = find_child(graph, ":ARG1")
            print(arg0)
            if arg0 is not None and isinstance(arg0, Node):
                root_arg_counter[(graph[1], arg0[1])] += 1

#            for child in graph[2]:
                #print(child[1])
#                root_counter[child[1]] += 1

            for child in gather_children_args(graph[2]):
                root_counter[child[1]] += 1
            print(graph)



    print(sorted(root_arg_counter.items(), key=lambda x: x[1]))




