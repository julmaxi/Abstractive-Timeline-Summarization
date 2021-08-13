# Requirements

Requirements are listed in the requirements.txt file.
To use AP clustering, place this repository https://github.com/nojima/affinity-propagation-sparse in the libs/affinity-propagation-sparse directory.

# Instructions for using the model with the original datasets

1. Download tilse from https://github.com/smartschat/tilse
2. Run ``get-and-preprocess-data`` from the tilse library. There is no need to wait for the preprocessing, we only need the downloaded data.
3. Use the raw data (in the raw directory created by tilse) and parse it using Stanford CoreNLP
4. Run the save_tl_corpus.py script. It needs the location of both timeml files from Heideltime and the location of the CoreNLP parsed files. You can either run Heideltime yourself or let tilse do the job. This needs to be done seperately for each corpus.
5. Download the Language Model from https://www.keithv.com/software/giga/lm_giga_20k_vp_3gram.zip and put it in your home directory
6. Run the tleval.py script. It requires the path of the corpus file created in step 4 and a config file as arguments. Example config files can be found in the configs directory.
Use the ap-abstractive-datetr-dateref-clsize-path-depfiltered-greedy-redundancy.json to for the setup described in the paper.

# Instructions for applying the system to new datasets

The system requires corpora to be saved as pickle files.
The core component to produce these is ``DatedTimelineCorpusReader``
It expects two directories. One with TimeML files for each article and one with Stanford CoreNLP XML files.

Both directories should be structured as follows:
    corpus/
    date1/
      article1.suffix
      article2.suffix
    date2/
      article3.suffix
      article4.suffix
    ...

Dates should be written as yyyy-mm-dd. Suffixes for both filetypes can be configured by passing them as arguments to ``DatedTimelineCorpusReader``.

Once the data is saved, the system can be invoked as follows:

``
python tlgraphsum/tleval.py -t [GOLD TL 1, GOLD TL 2]  -- [CORPUS PICKLE] [CONFIG FILE]
``

Parameters are derived from the timelines as described in the paper. If the run is successful, results will be written to the "systems_timelines" and "evaluation_results" directories respectively.
