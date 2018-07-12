from collections import Counter, deque, defaultdict
import itertools as it
import math
import re
import sys

from kenlm import Model


class KenLMLanguageModel:
    @staticmethod
    def from_file(filename):
        return KenLMLanguageModel(Model(filename))

    def __init__(self, kenlm_model):
        self.kenlm_model = kenlm_model

    def to_str(self, sent):
        return " ".join(map(lambda x: x[0], sent))

    def estimate_sent_log_proba(self, sent):
        log10score = self.kenlm_model.score(self.to_str(sent)) / len(sent)

        return log10score / math.log(2)

    def estimate_full_sent_log_probas(self, sent):
        return [proba for proba, _, _ in self.kenlm_model.full_scores(self.to_str(sent))]


class KenLMPOSLanguageModel:
    def to_str(self, sent):
        return " ".join(map(lambda x: x[1], sent))


class StoredLanguageModel:
    @staticmethod
    def from_file(filename):
        ngrams_by_size = defaultdict(dict)
        current_ngram_size = None
        with open(filename) as f:
            for line in f:
                line = line.strip()
                if line == "\\end\\":
                    break
                if len(line) == 0:
                    continue
                match = None
                if line[0] == "\\":
                    match = re.match(r"\\(\d+)-grams:", line)
                if match:
                    current_ngram_size = int(match.group(1))
                elif current_ngram_size is not None:
                    parts = line.split()

                    if len(parts) == current_ngram_size + 1:
                        ngrams_by_size[current_ngram_size][tuple(parts[1:])] = (float(parts[0]),)
                    elif len(parts) == current_ngram_size + 2:
                        ngrams_by_size[current_ngram_size][tuple(parts[1:-1])] = (float(parts[0]), float(parts[-1]))
                    else:
                        print("Warning, parsing error at line ", line)

        return StoredLanguageModel(ngrams_by_size)

    def __init__(self, ngrams_by_size):
        self.ngrams_by_size = ngrams_by_size
        self.n = max(ngrams_by_size.keys())

    def estimate_log_proba(self, context, word):
        ngram = context + (word,)
        proba = self.ngrams_by_size[len(ngram)].get(ngram)
        if proba is not None:
            return proba[0]

        proba = self.estimate_log_proba(context[1:], word)
        backoff = self.ngrams_by_size[len(ngram)].get(context)
        if backoff is None or len(backoff) != 2:
            backoff = 0.0  # Remember, we work in log-space
        else:
            backoff = backoff[1]

        return proba + backoff

    def _transform_sentence(self, sent):
        transformed_sent = ["<s>"]
        for tok in sent:
            tok = tok.lower()
            if (tok,) in self.ngrams_by_size[1]:
                transformed_sent.append(tok)
            else:
                transformed_sent.append("<unk>")

        transformed_sent.append("</s>")

        return transformed_sent

    def estimate_sent_log_proba(self, sent):
        sent = self._transform_sentence(sent)
        proba = 0.0
        for t_idx, tok in enumerate(sent):
            if t_idx == 0:
                continue
            context = tuple(sent[max(t_idx - self.n, 0):t_idx])
            proba += self.estimate_log_proba(context, tok)

        return proba / len(sent)


class KatzLanguageModel:
    def __init__(self, max_context=2, k=5):
        self.max_context = max_context
        self.occurence_counter = defaultdict(Counter)
        self.num_ngrams_counter = Counter()

        self.k = k

    def train(self, sentences):

        for sent in sentences:
            context = deque([None] * self.max_context)

            for tok in sent:
                context_words = tuple(context)
                for start_idx in range(self.max_context + 1):
                    ngram = context_words[start_idx:]
                    self.occurence_counter[ngram][tok] += 1
                    self.num_ngrams_counter[len(ngram) + 1] += 1

                context.popleft()
                context.append(tok)

        ngrams_with_length = []
        for context, counter in self.occurence_counter.items():
            for count in counter.values():
                ngrams_with_length.append((len(context) + 1, count))

        self.ngram_size_counter = Counter(ngrams_with_length)

    def estimate_propa(self, context, token):
        if len(context) == 0:
            return self.occurence_counter[tuple()][token] / sum(self.occurence_counter[tuple()].values())

        occurence_count = self.occurence_counter[context][token]
        if occurence_count > self.k:
            return self.occurence_counter[context][token] / sum(self.occurence_counter[context].values())
        elif occurence_count > 0:
            modified_r = (occurence_count + 1) * (
                self.ngram_size_counter[(len(context) + 1, occurence_count + 1)] /
                self.ngram_size_counter[(len(context) + 1, occurence_count)]
            )

            numerator = modified_r / occurence_count - ((self.k + 1) * self.ngram_size_counter[(self.k + 1, len(context) + 1)]) / self.ngram_size_counter[(len(context) + 1, 1)]
            denomiator = 1 - (self.k + 1) * self.ngram_size_counter[(self.k + 1, len(context) + 1)] / self.ngram_size_counter[(len(context) + 1, 1)]

            d = numerator / denomiator

            return d * occurence_count / self.occurence_counter[tuple(context)][token]
        else:
            estimation_sum = 0

            for tok, count in self.occurence_counter[context].items():
                estimation_sum += self.estimate_propa(context, tok)

            normalize_sum = 0
            for tok, count in self.occurence_counter[context[1:]].items():
                normalize_sum += self.estimate_propa(context[1:], tok)

            alpha = (1 - estimation_sum) / (1 - normalize_sum)

            return alpha * self.estimate_propa(context[1:], token)


class UNK:
    pass


class KeyserNeyLanguageModel:
    def __init__(self, n=3, d=0.75, cutoff_freq=5):
        self.n = n
        self.d = d

        self.vocabulary = set()
        self.cutoff_freq = cutoff_freq

    def _transform_sentence(self, sent):
        transformed_sent = []
        for tok in sent:
            if tok in self.vocabulary:
                transformed_sent.append(tok)
            else:
                transformed_sent.append(UNK)

        return transformed_sent

    def train(self, sentences):
        token_counter = Counter()
        transformed_sentences = []
        for sent in sentences:
            for tok in sent:
                token_counter[tok] += 1

        self.vocabulary = set(map(lambda x: x[0], filter(lambda x: x[1] > self.cutoff_freq, token_counter.items())))

        transformed_sentences = []
        for sent in sentences:
            transformed_sentences.append(self._transform_sentence(sent))

        sentences = transformed_sentences

        self.precedence_counts = defaultdict(Counter)
        self.counts = defaultdict(Counter)
        self.all_preceding_words_by_ngram_length = defaultdict(lambda: defaultdict(set))

        for sent in sentences:
            context = deque([None] * (self.n - 1))

            for t_idx, tok in enumerate(it.chain(sent, [None] * (self.n - 1))):
                self.vocabulary.add(tok)

                context_words = tuple(context)
                for start_idx in range(self.n):
                    context_ngram = context_words[start_idx:]
                    self.counts[context_ngram][tok] += 1

                    if t_idx == 0 or t_idx > len(sent):
                        precedent = None
                    else:
                        precedent = sent[t_idx - 1]
                    self.precedence_counts[context_ngram + (tok,)][precedent] += 1
                    self.all_preceding_words_by_ngram_length[len(context_ngram) + 1][context_ngram + (tok,)].add(precedent)

                context.popleft()
                context.append(tok)

    def estimate_proba(self, context, word, use_freq=True):
        if len(context) == 0:
            context_counts = sum(list(map(lambda x: len(x[1]), self.all_preceding_words_by_ngram_length[1].items())))

            return len(self.precedence_counts[(word,)]) / context_counts

        count = self.counts[context][word]
        count_normalizer = sum(self.counts[context].values())

        if not use_freq:
            count = len(self.precedence_counts[context + (word,)])
            count_normalizer = len(self.precedence_counts[context])

        if count_normalizer > 0:
            discounted_frequency = max(
                count - self.d, 0) / count_normalizer

            normalizer = self.compute_lambda(context, use_freq=use_freq)
            return discounted_frequency + normalizer * self.estimate_proba(context[1:], word, use_freq=False)
        else:
            # See if this is correct
            discounted_frequency = (1.0 - self.d) / len(self.vocabulary)
            normalizer = self.d
            return discounted_frequency + normalizer * self.estimate_proba(context[1:], word, use_freq=False)

    def compute_lambda(self, ngram, use_freq=True):
        continuation_counts = len(self.counts[ngram])

        if not use_freq:
            normalized_discount = self.d * len(self.counts[ngram])
            return normalized_discount * continuation_counts

        normalized_discount = self.d / sum(
            self.counts[ngram].values())

        return normalized_discount * continuation_counts

    def estimate_sent_log_proba(self, sent):
        sent = self._transform_sentence(sent)
        proba = 0.0
        for t_idx, tok in enumerate(it.chain(sent, [None] * (self.n - 1))):
            context = tuple(sent[max(t_idx - self.n, 0):t_idx])
            if len(context) < self.n:
                context = tuple([None] * (len(context) - self.n)) + context
            proba += self.estimate_log_proba(context, tok)

        return proba

    def estimate_log_proba(self, context, token):
        return math.log(self.estimate_proba(context, token))


if __name__ == "__main__":
    model = StoredLanguageModel.from_file(sys.argv[1])

    print(model.estimate_log_proba(("has", "a",), "dog"))
    print(model.estimate_sent_log_proba("The man eats a steak".split()))
    print(model.estimate_sent_log_proba("The a man eats steak".split()))

    sys.exit(0)

    print(sum([model.estimate_proba(("has", "a",), v) for v in model.vocabulary]))
    print(model.estimate_sent_log_proba(["The", "dog", "has", "cat"]))
    print(model.estimate_sent_log_proba(["The", "dog", "has", "a", "cat"]))
    print(model.estimate_sent_log_proba(["The", "dog", "dog", "has", "a", "cat"]))
