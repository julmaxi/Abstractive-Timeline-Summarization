from graphsum import calculate_informativeness, STOPWORDS, KenLMLanguageModel, read_cluster_from_premade_files, SklearnTfIdfCosineSimilarityModel, calculate_keyword_text_rank, TfidfVectorizer, cosine_similarity, SentenceCompressionGraph

import sys


def eval_path_measures_banerjee():
    lm = KenLMLanguageModel.from_file("langmodel20k_vp_3.bin")

    clusters = read_cluster_from_premade_files(sys.argv[1])

    global_sim_model = SklearnTfIdfCosineSimilarityModel(stem=False)
    global_sim_model.fit([sent.as_token_attr_sequence("form") for cl in clusters.values() for sent in cl])

    for cluster_name, cluster in clusters.items():
        if cluster_name != "cluster_005.txt":
            continue
        tr_scores = calculate_keyword_text_rank([sent.as_token_tuple_sequence("form", "pos") for sent in cluster])

        sim_model = TfidfVectorizer()
        orig_vecs = sim_model.fit_transform([" ".join(sent.as_token_attr_sequence("form")) for sent in cluster])

        def check_closeness(sent):
            vec_1 = sim_model.transform(map(lambda x: x[0], sent))
            similarities = cosine_similarity(vec_1, orig_vecs)

            return all(similarities[0,:] < 0.8)

        word_graph = SentenceCompressionGraph(STOPWORDS)
        word_graph.add_sentences(sent.as_token_tuple_sequence("form", "pos") for sent in cluster)

        top_path_weight_candidates = []
        top_informativeness_candidates = []
        top_ling_candidates = []
        top_score_candidates = []

        for sent in word_graph.generate_compression_candidates(n=250, filterfunc=check_closeness):
            lm_score = 1 / (1.0 - lm.estimate_sent_log_proba(list(map(lambda x: x[0], sent))))
            informativeness_score = calculate_informativeness(sent, tr_scores)
            score = informativeness_score * lm_score / len(sent)

            sent = " ".join(map(lambda x: x[0], sent))

            top_path_weight_candidates.append(sent)
            top_informativeness_candidates.append((sent, informativeness_score))
            top_ling_candidates.append((sent, lm_score))
            top_score_candidates.append((sent, score))

        top_score_candidates_no_weight = []
        shortest_no_weight = []
        for sent in word_graph.generate_compression_candidates(n=250, filterfunc=check_closeness, use_weighting=False):
            lm_score = 1 / (1.0 - lm.estimate_sent_log_proba(list(map(lambda x: x[0], sent))))
            informativeness_score = calculate_informativeness(sent, tr_scores)
            score = informativeness_score * lm_score / len(sent)
            sent = " ".join(map(lambda x: x[0], sent))
            top_score_candidates_no_weight.append((sent, score))

            shortest_no_weight.append(sent)

        with open("path-weight.txt", "w") as f_out:
            f_out.write("\n".join(top_path_weight_candidates))

        with open("informativeness.txt", "w") as f_out:
            top_informativeness_candidates.sort(key=lambda x: x[1], reverse=True)
            f_out.write("\n".join(map(lambda x: x[0], top_informativeness_candidates)))

        with open("lingusitic.txt", "w") as f_out:
            top_ling_candidates.sort(key=lambda x: x[1], reverse=True)
            f_out.write("\n".join(map(lambda x: x[0], top_ling_candidates)))

        with open("score.txt", "w") as f_out:
            top_score_candidates.sort(key=lambda x: x[1], reverse=True)
            f_out.write("\n".join(map(lambda x: x[0], top_ling_candidates)))

        with open("shortest-path.txt", "w") as f_out:
            f_out.write("\n".join(shortest_no_weight))

        with open("shortest-path-score.txt", "w") as f_out:
            top_score_candidates_no_weight.sort(key=lambda x: x[1], reverse=True)
            f_out.write("\n".join(map(lambda x: x[0], top_score_candidates_no_weight)))


if __name__ == "__main__":
    eval_path_measures_banerjee()