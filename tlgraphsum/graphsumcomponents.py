from component import *


class GenericCachableClusteringComponent:
    def compute(self, input_data, context, *args, **kwargs):
        pass


class APComponent(GenericCachableClusteringComponent):
    def compute_clusters(self, input_data, context):
        clusters = cluster_sentences_ap(
            corpus.sentences,
            include_uncertain_date_edges=self.include_uncertain_date_edges,
            predicted_tag_only=self.predicted_tag_only)


@component("scoring.basic")
class SentenceScorerComponent:
    @classmethod
    def get_parameters(self):
        return {
            "scorers": ListVaĺue(ComponentValue())
        }

    def __init__(self, scorers):
        self.scorers = scorers

    def score_corpus_candidates(self, corpus, clusters):
        total_scores = [[1 for _ in cluster] for cluster in clusters]

        for scorer in self.scorers:
            scores = scorer.score_corpus_candidates(corpus, clusters)

            for cluster_idx, cluster_scores in enumerate(scores):
                for cand_idx, score in enumerate(cluster_scores):
                    total_scores[cluster_idx][cand_idx] *= score

        return total_scores


class BasicScorer:
    def score_corpus_candidates(self, corpus, clusters):
        scores = []
        for cluster in clusters:
            scores.append(self.score_cluster_candidates(clusters))

        return scores

    def score_cluster_candidates(self, cluster):
        scores = []
        for cand in cluster:
            scores.append(self.score_candidate(cand))

        return scores

    def score_candidate(self, cand):
        pass


@component("scorer.lm")
class LMScorerComponent(BasicScorer):
    @classmethod
    def get_parameters(self):
        return {
        }

    def score_candidate(self, cand):
        pass


@component("scorer.local_tr")
class LocalClusterTRComponent(BasicScorer):
    @classmethod
    def get_parameters(self):
        return {
        }

    def score_cluster_candidates(self, cluster):
        pass
